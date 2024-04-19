import os
import csv
import shutil
import torch.hub as hub
import librosa
import numpy as np
import torch
import pickle
import argparse
import torchaudio
from tqdm import tqdm
from itertools import islice, zip_longest
import s3prl.hub as hub

# Global variables
root_dir = "/content/drive/MyDrive/sep28k"
input_csv_path = "/content/drive/MyDrive/sep28k/SEP-28k_labels.csv"
fluency_csv_path = "/content/drive/MyDrive/sep28k/fluencybank_labels.csv"
output_csv_path = "/content/drive/MyDrive/sep28k/SEP-28k_final_labels.csv"
considered_columns = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords']

# Function definitions
def gather_wav_files(directory):
    all_files = set()
    for podcast in os.listdir(directory):
        if podcast.endswith('.csv'):
            continue
        podcast_path = os.path.join(directory, podcast)
        if os.path.isdir(podcast_path):
            for episode in os.listdir(podcast_path):
                episode_path = os.path.join(podcast_path, episode)
                if os.path.isdir(episode_path):
                    for filename in os.listdir(episode_path):
                        if filename.endswith(".wav"):
                            all_files.add(filename)
    return all_files

def create_file_paths_dict(root_dir, available_files):
    file_paths = {} 
    for podcast in os.listdir(root_dir):
        if podcast.endswith('.csv'):
            continue
        podcast_path = os.path.join(root_dir, podcast)
        if os.path.isdir(podcast_path):
            for episode in os.listdir(podcast_path):
                episode_path = os.path.join(podcast_path, episode)
                if os.path.isdir(episode_path):
                    for filename in os.listdir(episode_path):
                        if filename.endswith(".wav") and filename in available_files:
                            full_path = os.path.join(episode_path, filename)
                            file_paths[filename] = full_path
    return file_paths

def process_and_write_rows(reader, writer, available_files, all_files):
    for row in reader:
        filename = f"{row['Show'].strip()}_{row['EpId'].strip()}_{row['ClipId'].strip()}.wav"
        if filename in all_files:
            for col in considered_columns:
                if int(row[col]) >= 2:
                    final_label = col
                    break
            else:
                final_label = "None"

            if final_label != "None":
                row["Final_Label"] = final_label
                writer.writerow(row)
                available_files.add(filename)

def create_labels_dict():
    labels_dict = {}
    with open(output_csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            filename = f"{row['Show'].strip()}_{row['EpId'].strip()}_{row['ClipId'].strip()}.wav"
            label = row['Final_Label']
            labels_dict[filename] = label
    return labels_dict

def load_audio(file_path, sr=16000):
    y, _ = librosa.load(file_path, sr=sr)
    return y

def extract_mfb_features(y, sr=16000, n_mels=40):
    mfb = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmin=0, fmax=8000)
    return mfb

def extract_f0_features(y, sr=16000, target_feats=94):
    f0, voiced_flag = librosa.core.piptrack(y=y, sr=sr)
    # Choosing the pitch with the highest magnitude
    indices = np.argmax(f0, axis=0)
    #print('indices:',len(indices))
    f0 = f0[indices, np.arange(f0.shape[1])]
    #print('f0:',len(f0))
    #print('flag:',len(voiced_flag))
    # Select voiced_flag based on indices of chosen pitches
    voiced_flag = np.array([voiced_flag[indices[t], t] for t in range(len(indices))])

      # Computing pitch-delta
    f0_delta = np.diff(f0, prepend=f0[0])

    # Combine features
    f0_features = np.column_stack((f0, f0_delta, voiced_flag))

    # Adjust the number of features to the target number
    num_features = f0_features.shape[1]
    if num_features < target_feats:
        # Pad with zeros if there are fewer features than target
        pad_width = target_feats - num_features
        f0_features = np.pad(f0_features, ((0, 0), (0, pad_width)), mode='constant')
    elif num_features > target_feats:
        # Truncate if there are more features than target
        f0_features = f0_features[:, :target_feats]

    return f0_features

def batch(iterable, n=1):
    args = [iter(iterable)] * n
    return zip_longest(*args)

def extract_w2v_features(file_paths, features_dict, device='cuda'):
    wav2vec = getattr(hub, 'wav2vec2')().to(device)  # Load the wav2vec model

    lay = 8  # Desired layer

    # Process files in batches
    for batch_files in tqdm(batch(file_paths.items(), 2)):
        batch_wavs = []
        valid_files = []

        for bf in batch_files:
            if bf is not None:
                file, path = bf
                wav, sr = torchaudio.load(path)  # Load the WAV file
                batch_wavs.append(wav.squeeze(dim=0).to(device))
                valid_files.append(file)

        if not batch_wavs:
            continue

        # Forward pass through wav2vec
        with torch.no_grad():
            batch_reps = wav2vec(batch_wavs)["hidden_states"][lay]

        # Update features_dict with wav2vec representations
        for j, rep in enumerate(batch_reps):
            rep_avg = np.mean(rep.cpu().numpy(), axis=1)
            features_dict[valid_files[j]]['wav2vec'] = rep_avg

    return features_dict
# Additional function definitions for feature extraction, preprocessing audio files, etc.

def filter_and_save_dictionaries(features_dict, labels_dict, file_paths):
    # Get the sets of keys from all dictionaries
    features_keys = set(features_dict.keys())
    labels_keys = set(labels_dict.keys())
    file_paths_keys = set(file_paths.keys())

    # Find the intersection of the keys
    matching_keys = features_keys.intersection(labels_keys).intersection(file_paths_keys)

    # Now filter all dictionaries to only keep matching keys
    filtered_features_dict = {key: features_dict[key] for key in matching_keys}
    filtered_labels_dict = {key: labels_dict[key] for key in matching_keys}
    filtered_file_paths = {key: file_paths[key] for key in matching_keys}

    # Save the filtered dictionaries to pickle files
    with open('filtered_features_dict.pkl', 'wb') as f:
        pickle.dump(filtered_features_dict, f)
    with open('filtered_labels_dict.pkl', 'wb') as f:
        pickle.dump(filtered_labels_dict, f)
    with open('filtered_file_paths.pkl', 'wb') as f:
        pickle.dump(filtered_file_paths, f)

    return filtered_features_dict, filtered_labels_dict, filtered_file_paths

def preprocess_audio_files(file_paths):
    features_dict = {}

    for file, path in file_paths.items():
        audio = load_audio(path)
        mfb_features = extract_mfb_features(audio)
        f0_features = extract_f0_features(audio)
        features_dict[file] = {
            'MFB': mfb_features,
            'F0': f0_features
        }

    with open('audio_features.pkl', 'wb') as f:
        pickle.dump(features_dict, f)

    return features_dict

def main():
    parser = argparse.ArgumentParser(description="Preprocess audio data.")
    parser.add_argument('--action', choices=['gather', 'process_csv', 'extract_features','filter_and_save'], required=True, help='Specify the action to perform')
    args = parser.parse_args()

    if args.action == 'gather':
        all_files = gather_wav_files(root_dir)
        print(f"Found {len(all_files)} WAV files.")

    elif args.action == 'process_csv':
        available_files = set()
        with open(input_csv_path, 'r') as csv_input1, open(fluency_csv_path, 'r') as csv_input2, open(output_csv_path, 'w', newline='') as csv_output:
            reader1 = csv.DictReader(csv_input1)
            reader2 = csv.DictReader(csv_input2)
            fieldnames = reader1.fieldnames + ["Final_Label"]
            writer = csv.DictWriter(csv_output, fieldnames=fieldnames)

            writer.writeheader()
            all_files = gather_wav_files(root_dir)
            process_and_write_rows(reader1, writer, available_files, all_files)
            process_and_write_rows(reader2, writer, available_files, all_files)
        
        file_paths = create_file_paths_dict(root_dir, available_files)

    elif args.action == 'extract_features':
        labels_dict = create_labels_dict()

        pass

    elif args.action == 'filter_and_save':
        # Assuming features_dict, labels_dict, and file_paths are already defined
        filtered_features_dict, filtered_labels_dict, filtered_file_paths = filter_and_save_dictionaries(features_dict, labels_dict, file_paths)
        print("Filtered dictionaries saved to pickle files.")

if __name__ == "__main__":
    main()

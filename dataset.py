#create dataset
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def fix_length(data, size):
    if data.shape[1] > size:
        return data[:, :size]  # Truncate
    elif data.shape[1] < size:
        padding = size - data.shape[1]
        return np.pad(data, ((0, 0), (0, padding)), mode='constant')  # Pad
    return data

class AudioFeaturesDataset(Dataset):
    def __init__(self, features_dict, labels_dict, sequence_length, considered_columns):
        self.features_dict = features_dict  # Dictionary with preprocessed features
        self.labels_dict = labels_dict
        self.sequence_length = sequence_length
        self.considered_columns = considered_columns
        self.fluent_index = self.considered_columns.index('NoStutteredWords')

    def __len__(self):
        return len(self.features_dict)

    def __getitem__(self, idx):
        file_name = list(self.features_dict.keys())[idx]

        # Retrieve preprocessed MFB and F0 features
        mfb_features = self.features_dict[file_name]['MFB']
        f0_features = self.features_dict[file_name]['F0']
        w2v_features = self.features_dict[file_name]['wav2vec']
        w2v_features = w2v_features.reshape(1,-1)
        # Ensure features are the correct length
        mfb = fix_length(mfb_features, self.sequence_length)
        f0 = fix_length(f0_features, self.sequence_length)
        w2v = fix_length(w2v_features, self.sequence_length)

        # Process labels
        label_text = self.labels_dict[file_name]
        label_index = self.considered_columns.index(label_text)
        bin_label = 0 if label_index == self.fluent_index else 1

        return torch.tensor(mfb, dtype=torch.float32), torch.tensor(f0, dtype=torch.float32), torch.tensor(w2v, dtype=torch.float32),torch.tensor(label_index, dtype=torch.long), torch.tensor(bin_label, dtype=torch.long)
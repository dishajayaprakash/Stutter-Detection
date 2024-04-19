import torch
import torchmetrics
import pickle
import torch.nn as nn
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from models import *  
from dataset import AudioFeaturesDataset  

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def load_data():
    with open('features_dict.pkl', 'rb') as f:
        features_dict = pickle.load(f)
    with open('labels_dict.pkl', 'rb') as f:
        labels_dict = pickle.load(f)
    with open('file_paths.pkl', 'rb') as f:
        file_paths = pickle.load(f)
    return features_dict, labels_dict, file_paths

def split_data(file_paths, labels_dict):
    fn_train_val, fn_test, labels_train_val, labels_test = train_test_split(
        list(file_paths.keys()), [labels_dict[fn] for fn in file_paths.keys()], test_size=0.2, random_state=42)
    fn_train, fn_val, labels_train, labels_val = train_test_split(
        fn_train_val, labels_train_val, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2
    return fn_train, fn_val, fn_test, labels_train, labels_val, labels_test

def create_datasets(fn_train, fn_val, fn_test, labels_train, labels_val, labels_test, features_dict, sequence_length, considered_columns):
    train_dataset = AudioFeaturesDataset({fn: features_dict[fn] for fn in fn_train}, {fn: labels_dict[fn] for fn in fn_train}, sequence_length, considered_columns)
    val_dataset = AudioFeaturesDataset({fn: features_dict[fn] for fn in fn_val}, {fn: labels_dict[fn] for fn in fn_val}, sequence_length, considered_columns)
    test_dataset = AudioFeaturesDataset({fn: features_dict[fn] for fn in fn_test}, {fn: labels_dict[fn] for fn in fn_test}, sequence_length, considered_columns)
    return train_dataset, val_dataset, test_dataset

def train_model(train_loader, val_loader, model_name,num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f1_binary = torchmetrics.F1Score(task='binary',num_classes=2).to(device)
    f1_multiclass = torchmetrics.F1Score(task='multiclass',num_classes=6).to(device)
    bin_loss = nn.CrossEntropyLoss()
    multi_loss = nn.CrossEntropyLoss()
    early_stopper = EarlyStopper()
    # Dynamically select the model based on the model name
    model_class = globals()[model_name]
    model = model_class(input_dim_mfb=40, 
                        input_dim_f0=94, 
                        input_dim_w2v=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_bin_losses = []
    train_multi_losses = []
    train_f1_bins = []
    train_f1_multis = []
    val_bin_losses = []
    val_multi_losses = []
    val_f1_bins = []
    val_f1_multis = []

    # Assuming `train_dataloader` is your dataloader with batches of data
    for epoch in range(num_epochs):
        model.train()
        total_bin_loss = 0.0
        total_multi_loss = 0.0
        f1_binary.reset()
        f1_multiclass.reset()

        for mfb_features, f0_features, w2v_features, multi_lab, bin_lab in train_loader:
            optimizer.zero_grad()

            # Send features and labels to the device
            mfb_features, f0_features, w2v_features = mfb_features.to(device), f0_features.to(device),w2v_features.to(device)
            multi_lab, bin_lab = multi_lab.to(device), bin_lab.to(device)

            # Forward pass
            out_fluent, out_events = model(mfb_features, f0_features, w2v_features)

            # Compute the loss
            loss_fluent = bin_loss(out_fluent, bin_lab)
            loss_multi = multi_loss(out_events, multi_lab)

            total_bin_loss += loss_fluent.item()
            total_multi_loss += loss_multi.item()

            pred_bin = torch.argmax(torch.softmax(out_fluent, dim=1), dim=1)
            pred_multi = torch.argmax(torch.softmax(out_events, dim=1), dim=1)

            f1_binary.update(pred_bin, bin_lab)
            f1_multiclass.update(pred_multi, multi_lab)

            # Backward pass and optimization
            combined_loss = loss_fluent + loss_multi
            combined_loss.backward()
            #loss_multi.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_bin_loss = total_bin_loss / len(train_loader)
        train_multi_loss = total_multi_loss / len(train_loader)
        train_f1_bin = f1_binary.compute()
        train_f1_multi = f1_multiclass.compute()

        #val
        model.eval()
        total_val_bin_loss = 0.0
        total_val_multi_loss = 0.0
        f1_binary.reset()
        f1_multiclass.reset()

        with torch.no_grad():
            for mfb_features, f0_features, w2v_features, multi_lab, bin_lab  in val_loader:
                # Send features and labels to the device
                mfb_features, f0_features, w2v_features = mfb_features.to(device), f0_features.to(device),w2v_features.to(device)
                multi_lab, bin_lab = multi_lab.to(device), bin_lab.to(device)

                # Forward pass
                out_fluent, out_events = model(mfb_features, f0_features, w2v_features)

                # Compute the loss
                loss_fluent = bin_loss(out_fluent, bin_lab)
                loss_multi = multi_loss(out_events, multi_lab)

                total_val_bin_loss += loss_fluent.item()
                total_val_multi_loss += loss_multi.item()

                pred_bin = torch.argmax(torch.softmax(out_fluent, dim=1), dim=1)
                pred_multi = torch.argmax(torch.softmax(out_events, dim=1), dim=1)

                f1_binary.update(pred_bin, bin_lab)
                f1_multiclass.update(pred_multi, multi_lab)

        val_bin_loss = total_val_bin_loss / len(val_loader)
        val_multi_loss = total_val_multi_loss / len(val_loader)
        val_f1_bin = f1_binary.compute()
        val_f1_multi = f1_multiclass.compute()

        if early_stopper.early_stop(val_multi_loss):
            break


        train_bin_losses.append(train_bin_loss)
        train_multi_losses.append(train_multi_loss)
        train_f1_bins.append(train_f1_bin)
        train_f1_multis.append(train_f1_multi)
        val_bin_losses.append(val_bin_loss)
        val_multi_losses.append(val_multi_loss)
        val_f1_bins.append(val_f1_bin)
        val_f1_multis.append(val_f1_multi)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Training - Binary Loss: {train_bin_loss}, Multiclass Loss: {train_multi_loss}, Binary F1: {train_f1_bin}, Multiclass F1: {train_f1_multi}")
        print(f"Validation - Binary Loss: {val_bin_loss}, Multiclass Loss: {val_multi_loss}, Binary F1: {val_f1_bin}, Multiclass F1: {val_f1_multi}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stutter Detection Model")
    parser.add_argument('--model', type=str, help='Model to use for training', required=True)
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
    args = parser.parse_args()

    features_dict, labels_dict, file_paths = load_data()

    fn_train, fn_val, fn_test, labels_train, labels_val, labels_test = split_data(file_paths, labels_dict)
    sequence_length = 200  # Set this to your desired sequence length
    considered_columns = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords']
    train_dataset, val_dataset, test_dataset = create_datasets(fn_train, fn_val, fn_test, labels_train, labels_val, labels_test, features_dict, sequence_length, considered_columns)

    # Create DataLoader with selected batch size
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Train model
    train_model(train_loader, val_loader, args.model, args.num_epochs)

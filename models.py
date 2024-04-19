import torch.nn as nn
import torch.nn.init as init
import torch

class ConvLSTM(nn.Module):
    def __init__(self, input_dim_mfb, input_dim_f0,input_dim_w2v,dropout_rate=0.4):
        super(ConvLSTM, self).__init__()

         # Convolution for MFB features
        self.conv_mfb = nn.Sequential(
            nn.Conv1d(input_dim_mfb, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Convolution for F0 features
        self.conv_f0 = nn.Sequential(
            nn.Conv1d(input_dim_f0, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )


        # Convolution for F0 features
        self.conv_w2v = nn.Sequential(
            nn.Conv1d(input_dim_w2v, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # LSTM layer
        self.lstm = nn.LSTM(192, 64, batch_first=True)  # 128 for concatenated features

        # Fully Connected layers for classification
        self.fc_fluent = nn.Linear(64, 2)  # Binary classification: Fluent/Dysfluent
        self.fc_events = nn.Linear(64, 6)  # 6 dysfluency types

         # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x_mfb, x_f0, x_w2v):
        #print("x_mfb shape:", x_mfb.shape)
        #print("x_f0 shape:", x_f0.shape)
        x_mfb = self.conv_mfb(x_mfb)
        x_f0 = self.conv_f0(x_f0)
        x_w2v = self.conv_w2v(x_w2v)
        x_w2v = x_w2v.transpose(1, 2)
        x_mfb = x_mfb.transpose(1, 2)  # Now shape: (batch, 100, 94)
        x_f0 = x_f0.transpose(1, 2)
        #print("x_mfb after conv shape:", x_mfb.shape)
        #print("x_f0 after conv shape:", x_f0.shape)

        # Combine features and LSTM processing
        x_combined = torch.cat((x_mfb, x_f0,x_w2v), dim=2)
        #print("x_combined shape:", x_combined.shape)
        self.lstm.flatten_parameters()
        _, (h_n, _) = self.lstm(x_combined)
        #print("Output shape after LSTM:", h_n.shape)
        lstm_out = h_n.squeeze(0)

        # Classification
        out_fluent = self.fc_fluent(lstm_out)
        out_events = self.fc_events(lstm_out)

        return out_fluent, out_events

class LSTM_base(nn.Module):
    def __init__(self):
        super(LSTM_base, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(135, 64, batch_first=True)  # 128 for concatenated features

        # Fully Connected layers for classification
        self.fc_fluent = nn.Linear(64, 2)  # Binary classification: Fluent/Dysfluent
        self.fc_events = nn.Linear(64, 6)  # 6 dysfluency types

    def forward(self, x_mfb, x_f0, x_w2v):
        #print("x_mfb shape:", x_mfb.shape)
        #print("x_f0 shape:", x_f0.shape)
        #x_mfb = self.conv_mfb(x_mfb)
        #x_f0 = self.conv_f0(x_f0)
        x_mfb = x_mfb.transpose(1, 2)  # Now shape: (batch, 100, 94)
        x_f0 = x_f0.transpose(1, 2)
        x_w2v = x_w2v.transpose(1, 2)
        #print("x_mfb after conv shape:", x_mfb.shape)
        #print("x_f0 after conv shape:", x_f0.shape)

        # Combine features and LSTM processing
        x_combined = torch.cat((x_mfb, x_f0,x_w2v), dim=2)
        #print("x_combined shape:", x_combined.shape)
        self.lstm.flatten_parameters()
        _, (h_n, _) = self.lstm(x_combined)
        #print("Output shape after LSTM:", h_n.shape)
        lstm_out = h_n.squeeze(0)

        # Classification
        out_fluent = self.fc_fluent(lstm_out)
        out_events = self.fc_events(lstm_out)

        return out_fluent, out_events

class ResNetStutterDetectionModel(nn.Module):
    def __init__(self, input_dim_mfb, input_dim_f0, num_mfb_features, num_f0_features):
        super(ResNetStutterDetectionModel, self).__init__()

        # ResNet Blocks for MFB features
        self.resnet_mfb = ResNetBlock(input_dim_mfb, num_mfb_features)

        # ResNet Blocks for F0 features
        self.resnet_f0 = ResNetBlock(input_dim_f0, num_f0_features)

        # Calculate the total number of features after flattening the output from both ResNet blocks
        self.num_combined_features = num_mfb_features * num_mfb_features + num_f0_features * num_f0_features

        # Fully Connected layers for classification
        self.fc_fluent = nn.Linear(self.num_combined_features, 2)  # Binary classification: Fluent/Dysfluent
        self.fc_events = nn.Linear(self.num_combined_features, 6)  # 6 dysfluency types

    def forward(self, x_mfb, x_f0):
        x_mfb = self.resnet_mfb(x_mfb)
        x_f0 = self.resnet_f0(x_f0)

        # Flatten the outputs
        x_mfb = x_mfb.view(x_mfb.size(0), -1)
        x_f0 = x_f0.view(x_f0.size(0), -1)

        # Combine features
        x_combined = torch.cat((x_mfb, x_f0), dim=1)

        # Classification
        out_fluent = self.fc_fluent(x_combined)
        out_events = self.fc_events(x_combined)

        return out_fluent, out_events

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, 1)  # Skip connection
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # Skip Connection
        out = self.relu(out)
        out = self.dropout(out)
        
        return out
model = ResNetStutterDetectionModel(input_dim_mfb=40, input_dim_f0=94, num_mfb_features=200, num_f0_features=200)

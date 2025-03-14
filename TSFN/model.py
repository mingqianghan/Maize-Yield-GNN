import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv

class TSFN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN Feature Extractors for vegetation and CWSI
        self.veg_cnn = self._create_cnn(in_channels=5, out_channels=16)  # Input: 5 channels, Output: 16 channels
        self.cwsi_cnn = self._create_cnn(in_channels=1, out_channels=8)  # Input: 1 channel, Output: 8 channels

        # Embedding for irrigation (static feature)
        self.irrigation_embed = nn.Embedding(num_embeddings=3, embedding_dim=16)

        # Temporal component (LSTM) for processing time-varying features
        # self.temporal = nn.LSTM(input_size=16*5*5 + 8*5*5, hidden_size=64, num_layers=2, batch_first=True)
        # Temporal component (GRU) for processing time-varying features
        self.gru = nn.GRU(input_size=16*5*5 + 8*5*5, hidden_size=64, num_layers=1, batch_first=True)
        
        # GraphSAGE layers
        self.sage1 = SAGEConv(64 + 16, 64, aggr='max')  # Input: LSTM output + irrigation embedding
        self.sage2 = SAGEConv(64, 32, aggr='max')

        # Final fully connected layer: maps 32 features to the target.
        self.fc = nn.Linear(32, 1)

    def _create_cnn(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5))  # Output size: (5, 5)
        )

    def forward(self, veg, cwsi, irrigation, edge_index):
        # veg shape: (batch_size, num_timepoints, channels, height, width)
        # cwsi shape: (batch_size, num_timepoints, channels, height, width)
        # irrigation shape: (batch_size, 1)

        batch_size, num_timepoints, veg_channels, veg_height, veg_width = veg.shape
        _, _, cwsi_channels, cwsi_height, cwsi_width = cwsi.shape

        # Process vegetation and CWSI for each time point
        veg_features = []
        cwsi_features = []
        for t in range(num_timepoints):
            # Extract vegetation and CWSI for the current time point
            veg_t = veg[:, t, :, :, :]  # Shape: (batch_size, channels, height, width)
            cwsi_t = cwsi[:, t, :, :, :]  # Shape: (batch_size, channels, height, width)

            # Pass through CNNs
            veg_features_t = self.veg_cnn(veg_t).flatten(1)  # Shape: (batch_size, 16*5*5)
            cwsi_features_t = self.cwsi_cnn(cwsi_t).flatten(1)  # Shape: (batch_size, 8*5*5)

            # Append to lists
            veg_features.append(veg_features_t)
            cwsi_features.append(cwsi_features_t)

        # Stack features across time points
        veg_features = torch.stack(veg_features, dim=1)  # Shape: (batch_size, num_timepoints, 16*5*5)
        cwsi_features = torch.stack(cwsi_features, dim=1)  # Shape: (batch_size, num_timepoints, 8*5*5)

        # Combine vegetation and CWSI features
        combined_features = torch.cat([veg_features, cwsi_features], dim=2)  # Shape: (batch_size, num_timepoints, 16*5*5 + 8*5*5)

        # Temporal processing with GRU
        gru_out, _ = self.gru(combined_features)  # Shape: (batch_size, num_timepoints, 64)
        temporal_out = gru_out[:, -1, :]  # Use the last time step's output (batch_size, 64)

        # Embed irrigation (static feature)
        irrigation = irrigation.squeeze().long()  # Ensure irrigation is a 1D tensor of indices
        irrigation_features = self.irrigation_embed(irrigation)  # Shape: (batch_size, 16)

        # Combine temporal output with irrigation features
        combined = torch.cat([temporal_out, irrigation_features], dim=1)  # Shape: (batch_size, 64 + 16)

        # GraphSAGE layers process the combined features.
        x1 = F.relu(self.sage1(combined, edge_index))  # Output: (batch_size, 64)
        x_final = F.relu(self.sage2(x1, edge_index))          # Output: (batch_size, 32)

        # Final prediction
        return self.fc(x_final)  # Output: (batch_size, 1)
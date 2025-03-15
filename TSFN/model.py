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

        # Temporal component (GRU) for processing time-varying features
        self.gru = nn.GRU(input_size=16*5*5 + 8*5*5, hidden_size=128, num_layers=2, batch_first=True)
        
        # Attention layer for weighting timepoints
        self.attention_fc1 = nn.Linear(128, 64)
        self.attention_fc2 = nn.Linear(64, 1)
        
        # GraphSAGE layers
        self.sage1 = SAGEConv(128 + 16, 64, aggr='max')  # Input: LSTM output + irrigation embedding
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
        # temporal_out = gru_out[:, -1, :]  # Use the last time step's output (batch_size, 64)
        
        # Compute attention weights for each timepoint
        # energy: (batch_size, num_timepoints, 1)
        # energy = self.attention_layer(gru_out)
        attn_hidden = torch.tanh(self.attention_fc1(gru_out))  # (batch_size, num_timepoints, 64)
        energy = self.attention_fc2(attn_hidden)               # (batch_size, num_timepoints, 1)
        attention_weights = F.softmax(energy, dim=1)
        context_vector = torch.sum(attention_weights * gru_out, dim=1)  # (batch_size, hidden_size)
        
     
        # Embed irrigation (static feature)
        irrigation = irrigation.squeeze().long()  # Ensure irrigation is a 1D tensor of indices
        irrigation_features = self.irrigation_embed(irrigation)  # Shape: (batch_size, 16)

        # Combine temporal output with irrigation features
        combined = torch.cat([context_vector, irrigation_features], dim=1)  # Shape: (batch_size, 64 + 16)

        # GraphSAGE layers process the combined features.
        x1 = F.relu(self.sage1(combined, edge_index))  # Output: (batch_size, 64)
        x_final = F.relu(self.sage2(x1, edge_index))          # Output: (batch_size, 32)

        # Final prediction
        return self.fc(x_final)  # Output: (batch_size, 1)
    
    def get_attention_weights(self, veg, cwsi):
        """
        Compute and return the attention weights.
        """
        batch_size, num_timepoints, veg_channels, veg_height, veg_width = veg.shape

        # Process each timepoint with CNNs.
        veg_features = []
        cwsi_features = []
        for t in range(num_timepoints):
            veg_t = veg[:, t, :, :, :]
            cwsi_t = cwsi[:, t, :, :, :]
            veg_features_t = self.veg_cnn(veg_t).flatten(1)
            cwsi_features_t = self.cwsi_cnn(cwsi_t).flatten(1)
            veg_features.append(veg_features_t)
            cwsi_features.append(cwsi_features_t)

        veg_features = torch.stack(veg_features, dim=1)
        cwsi_features = torch.stack(cwsi_features, dim=1)
        combined_features = torch.cat([veg_features, cwsi_features], dim=2)

        # Temporal processing with GRU.
        gru_out, _ = self.gru(combined_features)
        
        attn_hidden = torch.tanh(self.attention_fc1(gru_out))
        energy = self.attention_fc2(attn_hidden)
        attention_weights = F.softmax(energy, dim=1)
        return attention_weights
    
    
    
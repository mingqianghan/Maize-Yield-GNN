import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv

# Temporal-Spatial Fusion Network (TSFN)
class TSFN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN feature extractors (shared across timepoints)
        self.veg_cnn = self._create_cnn(in_channels=5, out_channels=16)
        self.cwsi_cnn = self._create_cnn(in_channels=1, out_channels=8)
        
        # Irrigation is a categorical variable; embedding dimension is set to 16.
        self.irrigation_embed = nn.Embedding(num_embeddings=3, embedding_dim=16)
        
        # After CNN and irrigation embedding, the per–timepoint feature dimension:
        # vegetation: 16 channels * 5 * 5 = 400
        # cwsi: 8 channels * 5 * 5 = 200
        # irrigation: 16
        # Total: 400 + 200 + 16 = 616
        #
        # GraphSAGE layers:
        self.sage1 = SAGEConv(616, 64, aggr='max')
        self.sage2 = SAGEConv(64, 32, aggr='max')
        
        # LSTM for temporal aggregation:
        # Input dimension is 32 (from GNN output) and hidden size is set to 64.
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        
        # Final fully connected layer mapping from LSTM output to the prediction.
        self.fc = nn.Linear(64, 1)
    
    def _create_cnn(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5))
        )
    
    def forward(self, veg, cwsi, irrigation, edge_index):
        """
        Inputs:
          veg: Tensor of shape (N, T, 5, H, W)         -- vegetation images
          cwsi: Tensor of shape (N, T, 1, H, W)        -- CWSI images
          irrigation: Tensor of shape (N, T)           -- categorical irrigation data per timepoint
          edge_index: Graph connectivity information for the N nodes.
          
        Process:
          1. For each timepoint t, extract features using CNNs and irrigation embedding.
          2. Process the concatenated features with GraphSAGE layers (per timepoint).
          3. Stack the per–timepoint GNN outputs to form a temporal sequence.
          4. Use an LSTM to aggregate temporal information.
          5. Produce the final prediction.
        """
        N, T = veg.shape[0], veg.shape[1]
        gnn_embeddings_list = []
        '''
        # Process each timepoint separately.
        for t in range(T):
            # Extract data for timepoint t.
            veg_t = veg[:, t, :, :, :]           # (N, 5, H, W)
            cwsi_t = cwsi[:, t, :, :, :]           # (N, 1, H, W)
            irr_t = irrigation[:, t]               # (N,)
            
            # CNN feature extraction.
            veg_feat = self.veg_cnn(veg_t).view(N, -1)    # (N, 400)
            cwsi_feat = self.cwsi_cnn(cwsi_t).view(N, -1)   # (N, 200)
            irr_feat = self.irrigation_embed(irr_t)         # (N, 16)
            
            # Concatenate the features.
            features = torch.cat([veg_feat, cwsi_feat, irr_feat], dim=1)  # (N, 616)
            
            # Apply GraphSAGE layers to incorporate spatial information.
            x = F.relu(self.sage1(features, edge_index))  # (N, 64)
            x = F.relu(self.sage2(x, edge_index))           # (N, 32)
            gnn_embeddings_list.append(x)
        '''
            
        for t in range(T):
            # Extract data for timepoint t.
            veg_t = veg[:, t, :, :, :]           # Expected shape: (N, 5, H, W)
            cwsi_t = cwsi[:, t, :, :, :]           # Expected shape: (N, 1, H, W)
            irr_t = irrigation[:, t].squeeze(-1)   # Squeeze to ensure shape: (N,)
            
            # CNN feature extraction.
            veg_feat = self.veg_cnn(veg_t).view(N, -1)    # (N, 400)
            cwsi_feat = self.cwsi_cnn(cwsi_t).view(N, -1)   # (N, 200)
            irr_feat = self.irrigation_embed(irr_t)         # (N, 16)
        
            # Concatenate the features.
            features = torch.cat([veg_feat, cwsi_feat, irr_feat], dim=1)  # (N, 616)
            
            # Apply GraphSAGE layers to incorporate spatial information.
            x = F.relu(self.sage1(features, edge_index))  # (N, 64)
            x = F.relu(self.sage2(x, edge_index))           # (N, 32)
            gnn_embeddings_list.append(x)
        
        # Stack the GNN embeddings across time: (N, T, 32)
        gnn_time_embeddings = torch.stack(gnn_embeddings_list, dim=1)
        
        # Use LSTM to aggregate the temporal dynamics.
        # lstm_out: (N, T, 64), h_n: (num_layers, N, 64)
        lstm_out, (h_n, _) = self.lstm(gnn_time_embeddings)
        # Take the output from the last time step.
        temporal_embedding = lstm_out[:, -1, :]  # (N, 64)
        
        # Final prediction.
        output = self.fc(temporal_embedding)
        return output



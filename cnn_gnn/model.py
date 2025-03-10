import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv

# ============================================
#              Model Architecture
# ============================================
class CNN_GNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN Feature Extractors
        self.veg_cnn = self._create_cnn(5, 16)
        self.cwsi_cnn = self._create_cnn(1, 8)

        self.irrigation_embed = nn.Embedding(num_embeddings=3, embedding_dim=16)

        self.sage1 = SAGEConv(16*5*5 + 8*5*5 + 16, 64, aggr='max')
        self.sage2 = SAGEConv(64, 32, aggr='max')

        # Final fully connected layer: maps 32 features to the target.
        self.fc = nn.Linear(32, 1)

    def _create_cnn(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5)))

    def forward(self, veg, cwsi, irrigation, edge_index):
        # Feature extraction
        veg_features = self.veg_cnn(veg).flatten(1)
        cwsi_features = self.cwsi_cnn(cwsi).flatten(1)

        irrigation = irrigation.squeeze().long()
        irrigation_features = self.irrigation_embed(irrigation)

        combined = torch.cat([veg_features, cwsi_features, irrigation_features], dim=1)

        # GraphSAGE layers process the combined features.
        x1 = F.relu(self.sage1(combined, edge_index))  # Output: (n, 64)
        x_final = F.relu(self.sage2(x1, edge_index))          # Output: (n, 32)

        # Final prediction
        return self.fc(x_final)
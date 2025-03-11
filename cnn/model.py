import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN Feature Extractors
        self.veg_cnn = self._create_cnn(5, 16)
        self.cwsi_cnn = self._create_cnn(1, 8)

        # Embedding for irrigation: 
        # Suppose irrigation has 3 possible categories -> embedding of size 16
        self.irrigation_embed = nn.Embedding(num_embeddings=3, embedding_dim=16)

        # Fully connected layers to process combined features
        input_dim = 16 * 5 * 5 + 8 * 5 * 5 + 16  # Sum of all flattened feature dims
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def _create_cnn(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5))  # Output shape = (out_channels, 5, 5)
        )

    def forward(self, veg, cwsi, irrigation):
        # Extract features via CNNs
        veg_features = self.veg_cnn(veg)     # (batch_size, 16, 5, 5)
        cwsi_features = self.cwsi_cnn(cwsi)  # (batch_size, 8,  5, 5)

        # Flatten feature maps
        veg_features = veg_features.flatten(start_dim=1)
        cwsi_features = cwsi_features.flatten(start_dim=1)

        # Embedding for irrigation
        irrigation = irrigation.squeeze().long()
        irrigation_features = self.irrigation_embed(irrigation)

        # Combine all features
        combined = torch.cat([veg_features, cwsi_features, irrigation_features], dim=1)

        # Pass through fully connected layers
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv

class TSFN_Model(nn.Module):
    """
    TSFN_Model (Temporal Spatial Fusion Network) implements deep learning architecture for Maize Yield Prediction.
    
    It processes time-varying image data (vegetation and CWSI) with CNNs and a GRU,
    applies an attention mechanism over time, integrates static irrigation features via embeddings,
    and then uses GraphSAGE layers to produce a final prediction.
    """
    def __init__(self):
        super().__init__()
        # CNN feature extractor for vegetation images.
        # Input: 5 channels, Output: 16 channels.
        self.veg_cnn = self._create_cnn(in_channels=5, out_channels=16)
        # CNN feature extractor for CWSI images.
        # Input: 1 channel, Output: 8 channels.
        self.cwsi_cnn = self._create_cnn(in_channels=1, out_channels=8)

        # Embedding for static irrigation feature.
        # There are 3 possible irrigation types, each embedded into a 16-dimensional vector.
        self.irrigation_embed = nn.Embedding(num_embeddings=3, embedding_dim=16)

        # GRU for temporal processing of combined CNN features.
        # The input size is determined by the flattened CNN outputs (16*5*5 for vegetation and 8*5*5 for CWSI).
        self.gru = nn.GRU(input_size=16*5*5 + 8*5*5, hidden_size=128, num_layers=2, batch_first=True)
        
        # Attention layer to compute scalar scores for each time step's GRU output.
        self.attention_layer = nn.Linear(128, 1)
        
        # GraphSAGE layers to integrate temporal features with static irrigation features.
        # First layer: combines GRU output (128) with irrigation embedding (16) to produce 64 features.
        self.sage1 = SAGEConv(128 + 16, 64, aggr='max')
        # Second layer: reduces the feature dimension from 64 to 32.
        self.sage2 = SAGEConv(64, 32, aggr='max')

        # Final fully connected layer mapping 32 features to the prediction target.
        self.fc = nn.Linear(32, 1)

    def _create_cnn(self, in_channels, out_channels):
        """
        Create a CNN submodule to extract features from input images.
        
        Architecture:
            Conv2d -> ReLU -> MaxPool2d -> Conv2d -> ReLU -> AdaptiveAvgPool2d
        The adaptive pooling ensures that the spatial dimensions are fixed to (5, 5).
        
        Parameters:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels after the second convolution.
        
        Returns:
            nn.Sequential: A CNN module.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5))  # Fix output feature map size to 5x5.
        )

    def forward(self, veg, cwsi, irrigation, edge_index):
        """
        Perform a forward pass through the model.
        
        Parameters:
            veg (Tensor): Vegetation images of shape (batch_size, num_timepoints, channels, height, width).
            cwsi (Tensor): CWSI images of shape (batch_size, num_timepoints, channels, height, width).
            irrigation (Tensor): Static irrigation data as indices (batch_size, 1).
            edge_index (Tensor): Graph edge index for the GraphSAGE layers.
        
        Returns:
            Tensor: Final prediction tensor of shape (batch_size, 1).
        """
        _, num_timepoints, _, _, _ = veg.shape

        # Process each timepoint separately using the CNNs.
        veg_features = []
        cwsi_features = []
        for t in range(num_timepoints):
            # Extract images for the current timepoint.
            veg_t = veg[:, t, :, :, :]   # Shape: (batch_size, channels, height, width)
            cwsi_t = cwsi[:, t, :, :, :]   # Shape: (batch_size, channels, height, width)

            # Extract features and flatten spatial dimensions.
            veg_features_t = self.veg_cnn(veg_t).flatten(1)  # Shape: (batch_size, 16*5*5)
            cwsi_features_t = self.cwsi_cnn(cwsi_t).flatten(1)  # Shape: (batch_size, 8*5*5)

            veg_features.append(veg_features_t)
            cwsi_features.append(cwsi_features_t)

        # Stack features over time (along dimension 1).
        veg_features = torch.stack(veg_features, dim=1)  # Shape: (batch_size, num_timepoints, 16*5*5)
        cwsi_features = torch.stack(cwsi_features, dim=1)  # Shape: (batch_size, num_timepoints, 8*5*5)

        # Concatenate vegetation and CWSI features along the feature dimension.
        combined_features = torch.cat([veg_features, cwsi_features], dim=2)  # Shape: (batch_size, num_timepoints, 16*5*5 + 8*5*5)

        # Process the time-varying features with the GRU.
        gru_out, _ = self.gru(combined_features)  # Output shape: (batch_size, num_timepoints, 128)
        
        # Compute attention scores for each timepoint.
        energy = self.attention_layer(gru_out)     # Shape: (batch_size, num_timepoints, 1)
        attention_weights = F.softmax(energy, dim=1)  # Normalize scores over the time dimension.

        # Compute a context vector as a weighted sum of the GRU outputs.
        context_vector = torch.sum(attention_weights * gru_out, dim=1)  # Shape: (batch_size, 128)

        # Process the static irrigation feature.
        irrigation = irrigation.squeeze().long()  # Ensure shape is (batch_size,) with integer indices.
        irrigation_features = self.irrigation_embed(irrigation)  # Shape: (batch_size, 16)

        # Combine the temporal context with the static irrigation embedding.
        combined = torch.cat([context_vector, irrigation_features], dim=1)  # Shape: (batch_size, 128 + 16)

        # Process the combined features using GraphSAGE layers.
        x1 = F.relu(self.sage1(combined, edge_index))  # Intermediate output: (batch_size, 64)
        x_final = F.relu(self.sage2(x1, edge_index))     # Final GraphSAGE output: (batch_size, 32)

        # Final prediction layer.
        return self.fc(x_final)  # Shape: (batch_size, 1)
    
    def get_attention_weights(self, veg, cwsi):
        """
        Compute and return attention weights for each timepoint based on the input images.
        
        Parameters:
            veg (Tensor): Vegetation images of shape (batch_size, num_timepoints, channels, height, width).
            cwsi (Tensor): CWSI images of shape (batch_size, num_timepoints, channels, height, width).
        
        Returns:
            Tensor: Attention weights with shape (batch_size, num_timepoints, 1).
        """
        batch_size, num_timepoints, _, _, _ = veg.shape

        veg_features = []
        cwsi_features = []
        for t in range(num_timepoints):
            # Process each timepoint with the CNNs.
            veg_t = veg[:, t, :, :, :]
            cwsi_t = cwsi[:, t, :, :, :]
            veg_features_t = self.veg_cnn(veg_t).flatten(1)
            cwsi_features_t = self.cwsi_cnn(cwsi_t).flatten(1)
            veg_features.append(veg_features_t)
            cwsi_features.append(cwsi_features_t)

        # Stack features over time.
        veg_features = torch.stack(veg_features, dim=1)
        cwsi_features = torch.stack(cwsi_features, dim=1)
        combined_features = torch.cat([veg_features, cwsi_features], dim=2)

        # Process combined features with the GRU.
        gru_out, _ = self.gru(combined_features)
        
        # Compute and return the attention weights.
        energy = self.attention_layer(gru_out)
        attention_weights = F.softmax(energy, dim=1)
        return attention_weights
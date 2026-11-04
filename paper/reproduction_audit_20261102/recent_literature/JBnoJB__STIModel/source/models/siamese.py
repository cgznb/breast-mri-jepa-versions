"""
Siamese network architecture for breast tumor response prediction.
"""

import torch
import torch.nn as nn

from models.attention import AttentionModule
from models.temporal import TemporalTransformerBlock


class SiameseNetwork(nn.Module):
    """
    Siamese network for comparing pre- and post-treatment MRI scans.
    
    This model processes paired MRI scans and their sub-regions to predict
    pathologic complete response (pCR) in breast cancer patients.
    """
    def __init__(self, resnet, pretrained_path=''):
        """
        Initialize the Siamese network.
        
        Args:
            resnet: ResNet model for feature extraction
            pretrained_path (str): Path to pretrained weights
        """
        super(SiameseNetwork, self).__init__()
        # Initialize ResNet
        self.resnet = resnet
        
        if pretrained_path:  # Load pretrained weights if provided
            self._load_pretrained_weights(pretrained_path)

        # Define model structure
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        self.attention = AttentionModule(512)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Temporal transformer module
        self.temporal_transformer = TemporalTransformerBlock(
            feature_dim=512,
            num_heads=8,
            num_layers=2
        )

        # Sub-region feature fusion layer
        self.region_fusion = nn.Sequential(
            nn.Linear(512 * 3, 512),  # Fuse 3 sub-regions
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512)
        )

        # Final classification layer
        self.fc = nn.Sequential(
            nn.Linear(512 * 4, 512),  # Global features + Sub-region features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def _load_pretrained_weights(self, pretrained_path):
        """
        Load pretrained weights for the model.
        
        Args:
            pretrained_path (str): Path to pretrained weights
        """
        device = next(self.parameters()).device
        checkpoint = torch.load(pretrained_path, map_location=device)
        
        # Handle different state dict formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Load pretrained weights
        self.load_state_dict(state_dict, strict=False)

    def forward_one(self, x):
        """
        Process a single MRI scan.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, 1, D, H, W]
            
        Returns:
            torch.Tensor: Feature vector of shape [B, 512]
        """
        x = self.features(x)
        x = self.attention(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward_regions(self, regions):
        """
        Process sub-region features.
        
        Args:
            regions (torch.Tensor): Input tensor of shape [B, n_regions, 1, D, H, W]
            
        Returns:
            torch.Tensor: Fused region features of shape [B, 512]
        """
        region_features = []
        for i in range(regions.size(1)):  # Iterate over each sub-region
            region = regions[:, i]  # [batch_size, 1, D, H, W]
            feat = self.forward_one(region)
            region_features.append(feat)

        # Fuse all sub-region features
        region_features = torch.stack(region_features, dim=1)  # [batch_size, n_regions, 512]
        region_features = region_features.view(region_features.size(0), -1)  # [batch_size, n_regions * 512]
        fused_region_features = self.region_fusion(region_features)  # [batch_size, 512]
        return fused_region_features

    def forward(self, x1, x2, pre_regions, post_regions, diff_time):
        """
        Process paired pre- and post-treatment MRI scans.
        
        Args:
            x1 (torch.Tensor): Pre-treatment scan of shape [B, 1, D, H, W]
            x2 (torch.Tensor): Post-treatment scan of shape [B, 1, D, H, W]
            pre_regions (torch.Tensor): Pre-treatment sub-regions of shape [B, n_regions, 1, D, H, W]
            post_regions (torch.Tensor): Post-treatment sub-regions of shape [B, n_regions, 1, D, H, W]
            diff_time (torch.Tensor): Time difference between scans of shape [B]
            
        Returns:
            torch.Tensor: Classification outputs of shape [B, 2]
        """
        # Process global images
        out1 = self.forward_one(x1)  # Pre-treatment global features
        out2 = self.forward_one(x2)  # Post-treatment global features

        # Process sub-regions
        pre_region_features = self.forward_regions(pre_regions)   # Pre-treatment sub-region features
        post_region_features = self.forward_regions(post_regions) # Post-treatment sub-region features

        diff_region_features = post_region_features - pre_region_features

        # Calculate global feature differences
        global_diff = out2 - out1

        # Use temporal transformer to enhance difference features
        time_enhanced_features = self.temporal_transformer(global_diff, diff_time)
        region_time_diff = self.temporal_transformer(diff_region_features, diff_time)

        # Concatenate all features
        combined_features = torch.cat([
            out1, out2, time_enhanced_features,  # Global features and time-enhanced features
            region_time_diff  # Sub-region features
        ], dim=1)

        # Final classification
        out = self.fc(combined_features)
        return out 
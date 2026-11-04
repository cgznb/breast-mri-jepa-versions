"""
Attention modules for the breast tumor response prediction model.
"""

import torch
import torch.nn as nn


class AttentionModule(nn.Module):
    """
    Channel attention module for 3D medical images.
    
    This module applies channel-wise attention to the input features,
    which helps the model focus on relevant regions of the image.
    """
    def __init__(self, in_channels):
        """
        Initialize the attention module.
        
        Args:
            in_channels (int): Number of input channels
        """
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the attention module.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, D, H, W]
            
        Returns:
            torch.Tensor: Attention-weighted tensor of the same shape
        """
        attention_weights = self.attention(x)
        return x * attention_weights 
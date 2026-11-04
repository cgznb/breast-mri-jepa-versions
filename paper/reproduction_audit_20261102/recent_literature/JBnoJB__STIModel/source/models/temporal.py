"""
Temporal modules for processing time information between pre- and post-treatment scans.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """
    Module for embedding time differences into a feature space.
    """
    def __init__(self, embedding_dim):
        """
        Initialize the time embedding module.
        
        Args:
            embedding_dim (int): Dimension of the embedding space
        """
        super(TimeEmbedding, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, diff_time):
        """
        Embed time differences into a feature space.
        
        Args:
            diff_time (torch.Tensor): Time differences of shape [batch_size]
            
        Returns:
            torch.Tensor: Embedded features of shape [batch_size, embedding_dim]
        """
        return self.embedding(diff_time.unsqueeze(1))


def positional_encoding(diff_time, hidden_size):
    """
    Generate sinusoidal positional encoding for time differences.
    
    Args:
        diff_time (torch.Tensor): Time differences of shape [batch_size]
        hidden_size (int): Size of the encoding
        
    Returns:
        torch.Tensor: Positional encoding of shape [batch_size, hidden_size]
    """
    pe = torch.zeros(diff_time.size(0), hidden_size)
    position = diff_time.unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.to(diff_time.device)


class MultiHeadTemporalAttention(nn.Module):
    """
    Multi-head attention mechanism for temporal feature enhancement.
    
    This module integrates time information with image features using
    a transformer-like attention mechanism.
    """
    def __init__(self, feature_dim, num_heads=8):
        """
        Initialize the multi-head temporal attention module.
        
        Args:
            feature_dim (int): Dimension of the feature space
            num_heads (int): Number of attention heads
        """
        super(MultiHeadTemporalAttention, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads
        assert self.head_dim * num_heads == feature_dim, "Feature dimension must be divisible by number of heads"

        # Time encoding projection
        self.time_proj = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        # Q, K, V projection layers
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)

        self.out_proj = nn.Linear(feature_dim, feature_dim)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 4, feature_dim)
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, diff_time):
        """
        Forward pass of the multi-head temporal attention module.
        
        Args:
            x (torch.Tensor): Input features of shape [batch_size, seq_len, feature_dim]
            diff_time (torch.Tensor): Time differences of shape [batch_size]
            
        Returns:
            torch.Tensor: Enhanced features of shape [batch_size, feature_dim]
        """
        batch_size = x.size(0)

        # Generate time encoding
        time_encoding = self.time_proj(diff_time.unsqueeze(-1))  # [batch_size, feature_dim]

        # Residual connection and Layer Norm
        residual = x
        x = self.norm1(x)

        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(time_encoding).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(time_encoding).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.feature_dim)
        x = self.out_proj(x)

        # First residual connection
        x = x + residual

        # Second residual connection and FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + residual

        return x.squeeze(1)  # Remove sequence dimension


class TemporalTransformerBlock(nn.Module):
    """
    Transformer block for temporal feature enhancement.
    
    This module consists of multiple layers of temporal attention
    and a fusion mechanism to combine original and enhanced features.
    """
    def __init__(self, feature_dim, num_heads=8, num_layers=2):
        """
        Initialize the temporal transformer block.
        
        Args:
            feature_dim (int): Dimension of the feature space
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
        """
        super(TemporalTransformerBlock, self).__init__()
        self.layers = nn.ModuleList([
            MultiHeadTemporalAttention(feature_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, x, diff_time):
        """
        Forward pass of the temporal transformer block.
        
        Args:
            x (torch.Tensor): Input features of shape [batch_size, feature_dim]
            diff_time (torch.Tensor): Time differences of shape [batch_size]
            
        Returns:
            torch.Tensor: Enhanced features of shape [batch_size, feature_dim]
        """
        # Save original features
        original_features = x

        # Process through multiple temporal attention layers
        for layer in self.layers:
            x = layer(x.unsqueeze(1), diff_time)

        # Fuse original and enhanced features
        x = torch.cat([original_features, x], dim=-1)
        x = self.fusion(x)

        return x 
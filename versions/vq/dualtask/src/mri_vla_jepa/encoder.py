"""Independent visit MRI encoding and portable spatial-grid pooling."""
from __future__ import annotations

import math
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .vq_encoder import VQContinuousFrontend


def _groups(channels):
    return math.gcd(channels, 8)


def _separable_grid_pool(value, grid):
    """Exact adaptive-average bins using native reductions on each spatial axis.

    MPS lacks adaptive_avg_pool3d; its 1D/2D versions also reject some grids.
    Separating the three averaging axes implements the same bin definitions
    without CPU transfers, so gradients and non-divisible volumes still work.
    """
    for axis, target in enumerate(grid, start=2):
        length = value.shape[axis]
        if length == target:
            continue
        bins = []
        for index in range(target):
            slices = [slice(None)] * 5
            slices[axis] = slice(index * length // target,
                                 ((index + 1) * length + target - 1) // target)
            bins.append(value[tuple(slices)].mean(axis))
        value = torch.stack(bins, axis)
    return value


class _SpatialGridPool(nn.Module):
    def __init__(self, grid):
        super().__init__()
        self.grid = tuple(grid)

    def forward(self, value):
        if value.device.type == "mps":
            return _separable_grid_pool(value, self.grid)
        return F.adaptive_avg_pool3d(value, self.grid)


class VisitMRIEncoder(nn.Module):
    """Spatial tokens for one visit; no attention or normalization across visits."""

    def __init__(self, cfg):
        super().__init__()
        self.grid = tuple(cfg.token_grid)
        hidden = max(8, cfg.dim // 2)
        self.stem = nn.Sequential(
            nn.Conv3d(3, hidden, 3, stride=2, padding=1),
            nn.GroupNorm(_groups(hidden), hidden), nn.GELU(),
            nn.Conv3d(hidden, cfg.dim, 3, stride=2, padding=1),
            nn.GroupNorm(_groups(cfg.dim), cfg.dim), nn.GELU(),
            _SpatialGridPool(self.grid),
        )
        self.position = nn.Parameter(torch.randn(1, math.prod(self.grid), cfg.dim) * .02)
        layer = nn.TransformerEncoderLayer(
            cfg.dim, cfg.heads, cfg.dim * 4, cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.spatial = nn.TransformerEncoder(layer, cfg.encoder_depth,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.dim)

    def forward(self, images):
        if images.ndim != 5 or images.shape[1] != 3:
            raise ValueError("Visit encoder expects [visits,3,D,H,W]")
        tokens = self.stem(images).flatten(2).transpose(1, 2)
        return self.norm(self.spatial(tokens + self.position))


class VQVisitMRIEncoder(nn.Module):
    """Shared pretrained per-phase features, followed by trainable MRI tokens.

    [O,3,D,H,W] -> [O,3*embedding_dim,D/4,H/4,W/4]
    -> grid pooling -> projection -> the existing spatial Transformer.
    """

    def __init__(self, cfg):
        super().__init__()
        self.grid = tuple(cfg.token_grid)
        self.frontend = VQContinuousFrontend(cfg)
        self.activation_checkpointing = cfg.vq_activation_checkpointing
        self.pool = _SpatialGridPool(self.grid)
        self.projection = nn.Conv3d(3 * cfg.vq_embedding_dim, cfg.dim, 1)
        self.position = nn.Parameter(torch.randn(1, math.prod(self.grid), cfg.dim) * .02)
        layer = nn.TransformerEncoderLayer(
            cfg.dim, cfg.heads, cfg.dim * 4, cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.spatial = nn.TransformerEncoder(layer, cfg.encoder_depth,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.dim)
        self.set_pretrained_trainable(False)

    def pretrained_parameters(self):
        """Only parameters originating in the VQ checkpoint, excluding adapter."""
        return self.frontend.parameters()

    def set_pretrained_trainable(self, trainable):
        if not isinstance(trainable, bool):
            raise ValueError("Pretrained encoder trainability must be boolean")
        self.pretrained_trainable = trainable
        self.frontend.requires_grad_(trainable)
        if not trainable:
            for parameter in self.frontend.parameters():
                parameter.grad = None
        self.frontend.train(self.training if trainable else False)
        return self

    def train(self, mode=True):
        super().train(mode)
        if not self.pretrained_trainable:
            self.frontend.eval()
        return self

    def initialize_pretrained(self, path):
        return self.frontend.initialize_pretrained(path)

    def forward(self, images):
        if images.ndim != 5 or images.shape[1] != 3:
            raise ValueError("Visit encoder expects [visits,3,D,H,W]")
        # Each phase uses the same one-channel pretrained convolution weights.
        # Frozen features keep no autograd graph; adapters still receive grads.
        context = nullcontext() if self.pretrained_trainable else torch.no_grad()
        with context:
            features = []
            for phase in images.split(1, dim=1):
                if (self.activation_checkpointing and self.pretrained_trainable
                        and torch.is_grad_enabled()):
                    encoded = checkpoint(self.frontend, phase, use_reentrant=False)
                else:
                    encoded = self.frontend(phase)
                features.append(encoded)
            features = torch.cat(features, dim=1)
        tokens = self.projection(self.pool(features)).flatten(2).transpose(1, 2)
        return self.norm(self.spatial(tokens + self.position))


def build_visit_encoder(cfg):
    if cfg.encoder_type == "raw":
        return VisitMRIEncoder(cfg)
    if cfg.encoder_type == "vq_prequant":
        return VQVisitMRIEncoder(cfg)
    raise ValueError("Unsupported MRI encoder_type")

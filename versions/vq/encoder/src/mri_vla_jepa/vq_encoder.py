"""Continuous front end of the pretrained single-channel MRI VQ-GAN.

The convolution/padding/residual architecture and checkpoint names are adapted
from MeWM via cgznb/symm-fm and local trjepa/codec.py. This derived component
retains their Creative Commons Attribution-NonCommercial 4.0 terms:
https://creativecommons.org/licenses/by-nc/4.0/legalcode
Upstream: https://github.com/scott-yjyang/MeWM and
https://github.com/cgznb/symm-fm

Changes: retain only encoder and pre_quant; expose gradient-enabled continuous
encoding and weights-only initialization. No quantizer, decoder, discriminator,
Lightning dependency, or codebook updates are present.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


def _triple(value):
    return (value, value, value) if isinstance(value, int) else tuple(value)


class SamePadConv3d(nn.Conv3d):
    """Replication padding with the original codec's asymmetric SAME rule."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        kernel, steps = _triple(kernel_size), _triple(stride)
        super().__init__(in_channels, out_channels, kernel, stride=steps, padding=0)
        differences = [size - step for size, step in zip(kernel, steps)]
        pairs = [(difference // 2 + difference % 2, difference // 2)
                 for difference in differences]
        self.pad_input = tuple(value for pair in reversed(pairs) for value in pair)

    def forward(self, value):
        if any(self.pad_input):
            value = F.pad(value, self.pad_input, mode="replicate")
        return super().forward(value)


class ResidualBlock3D(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels, eps=1e-6)
        self.conv1 = SamePadConv3d(channels, channels, 3)
        self.norm2 = nn.GroupNorm(groups, channels, eps=1e-6)
        self.conv2 = SamePadConv3d(channels, channels, 3)

    def forward(self, value):
        hidden = self.conv1(F.silu(self.norm1(value)))
        return value + self.conv2(F.silu(self.norm2(hidden)))


class SinglePhaseVQEncoder(nn.Module):
    """Preserve original encoder.* parameter names and GroupNorm semantics."""

    def __init__(self, cfg):
        super().__init__()
        hidden, groups = cfg.vq_hidden_channels, cfg.vq_num_groups
        self.input = SamePadConv3d(1, hidden, 3)
        self.downsamples = nn.ModuleList([
            SamePadConv3d(hidden, hidden * 2, 4, 2),
            SamePadConv3d(hidden * 2, hidden * 4, 4, 2),
        ])
        self.residuals = nn.ModuleList([
            ResidualBlock3D(hidden * 2, groups),
            ResidualBlock3D(hidden * 4, groups),
        ])
        self.bottleneck = nn.ModuleList([
            ResidualBlock3D(hidden * 4, groups)
            for _ in range(cfg.vq_bottleneck_blocks)
        ])
        for block in self.bottleneck:
            nn.init.zeros_(block.conv2.weight)
            nn.init.zeros_(block.conv2.bias)
        self.final_norm = nn.GroupNorm(groups, hidden * 4, eps=1e-6)
        self.output_channels = hidden * 4

    def forward(self, images):
        hidden = self.input(images)
        for downsample, residual in zip(self.downsamples, self.residuals, strict=True):
            hidden = residual(downsample(hidden))
        for residual in self.bottleneck:
            hidden = residual(hidden)
        return F.silu(self.final_norm(hidden))


class VQContinuousFrontend(nn.Module):
    """Single-channel continuous features before the discrete VQ codebook."""

    def __init__(self, cfg):
        super().__init__()
        self.encoder = SinglePhaseVQEncoder(cfg)
        self.pre_quant = SamePadConv3d(self.encoder.output_channels,
                                       cfg.vq_embedding_dim, 1)
        self.architecture = {
            "image_channels": 1,
            "hidden_channels": cfg.vq_hidden_channels,
            "embedding_dim": cfg.vq_embedding_dim,
            "downsample_factor": 4,
            "bottleneck_blocks": cfg.vq_bottleneck_blocks,
            "num_groups": cfg.vq_num_groups,
        }

    def forward(self, images):
        if (images.ndim != 5 or images.shape[1] != 1 or
                any(size < 4 or size % 4 for size in images.shape[-3:])):
            raise ValueError("VQ front end expects [visits,1,D,H,W] divisible by four")
        return self.pre_quant(self.encoder(images))

    def initialize_pretrained(self, path):
        """Strictly load only encoder/pre_quant from the audited safe export."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or payload.get("schema") != "symm_world_codec_v2":
            raise ValueError("Expected a weights-only symm_world_codec_v2 export")
        contract = "registered_ROI32_shared_training_DCE0_normalization"
        if payload.get("numeric_contract") != contract:
            raise ValueError("Expected the registered ROI32 DCE0 normalization contract")
        architecture = payload.get("model_config")
        if not isinstance(architecture, Mapping):
            raise ValueError("Codec export is missing its model_config")
        for name, expected in self.architecture.items():
            if architecture.get(name) != expected:
                raise ValueError(f"Pretrained codec {name} differs from model architecture")
        source = payload.get("codec_state")
        if not isinstance(source, Mapping):
            raise ValueError("Codec export is missing its codec_state")
        state = {name: value for name, value in source.items()
                 if name.startswith(("encoder.", "pre_quant."))}
        expected_state = self.state_dict()
        if state.keys() != expected_state.keys():
            raise ValueError("Pretrained codec encoder/pre_quant keys differ from architecture")
        for name, expected in expected_state.items():
            value = state[name]
            if not isinstance(value, torch.Tensor) or value.shape != expected.shape:
                raise ValueError(f"Pretrained codec tensor {name} has an incompatible shape")
            if not torch.isfinite(value).all():
                raise ValueError(f"Pretrained codec tensor {name} contains nonfinite values")
        self.load_state_dict(state, strict=True)
        return {"schema": payload["schema"], "numeric_contract": contract,
                "architecture": dict(self.architecture),
                "loaded_parameter_tensors": len(state)}

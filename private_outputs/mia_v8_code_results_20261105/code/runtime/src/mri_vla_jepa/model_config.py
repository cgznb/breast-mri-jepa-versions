"""Validated model hyperparameters; voxel-flow generation is optional."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .phases import selected_phases


MODEL_SIZE_PRESETS = {
    "small": dict(dim=128, encoder_depth=2, fusion_depth=2, predictor_depth=2,
                  heads=4, attention_head_dim=None, ffn_dim=512),
    "medium": dict(dim=768, encoder_depth=8, fusion_depth=8, predictor_depth=8,
                   heads=16, attention_head_dim=64, ffn_dim=3072),
    "large": dict(dim=1024, encoder_depth=12, fusion_depth=12, predictor_depth=12,
                  heads=16, attention_head_dim=64, ffn_dim=4096),
}
_LEGACY_STRUCTURE = dict(dim=128, encoder_depth=2, fusion_depth=2,
                         predictor_depth=2, heads=4)


def resolve_model_config(value):
    """Expand an explicitly named preset before strict training-config parsing."""
    if not isinstance(value, dict):
        raise ValueError("VLA model configuration must be a mapping")
    resolved = dict(value)
    size = resolved.get("model_size")
    if size is None:
        return resolved
    if not isinstance(size, str) or size not in MODEL_SIZE_PRESETS:
        raise ValueError("model_size must be small, medium, large or null")
    for name, expected in MODEL_SIZE_PRESETS[size].items():
        supplied = resolved.get(name)
        if supplied is not None and supplied != expected:
            raise ValueError(f"{name}={supplied!r} conflicts with model_size={size!r} "
                             f"(expected {expected!r})")
        resolved[name] = expected
    return resolved


@dataclass
class RawJEPAConfig:
    clinical_dim: int = 4
    dim: int | None = None
    token_grid: tuple[int, int, int] = (2, 4, 4)
    encoder_depth: int | None = None
    fusion_depth: int | None = None
    predictor_depth: int | None = None
    heads: int | None = None
    state_queries: int = 8
    pcr_queries: int = 4
    fm_channels: int = 24
    ema_decay: float = .99
    dropout: float = 0.
    enable_flow: bool = True
    enable_world_model: bool = True
    encoder_type: str = "raw"
    vq_hidden_channels: int = 16
    vq_embedding_dim: int = 8
    vq_bottleneck_blocks: int = 1
    vq_num_groups: int = 32
    vq_activation_checkpointing: bool = False
    fixed_transition_queries: bool = False
    pcr_readout: str = "direct"
    spatial_readout: str = "mean"
    pcr_delta_reference: str = "current"
    world_residual: bool = False
    history_delta: bool = False
    lesion_world_weighting: bool = False
    lesion_pooling: bool = False
    lesion_weight_strength: float = 4.
    model_size: str | None = None
    attention_head_dim: int | None = None
    ffn_dim: int | None = None
    mri_phases: str = "all"

    def __post_init__(self):
        # None distinguishes an omitted structure field from an explicit value.
        # This supports presets without reinterpreting historical custom models.
        if self.model_size is not None:
            if not isinstance(self.model_size, str) or self.model_size not in MODEL_SIZE_PRESETS:
                raise ValueError("model_size must be small, medium, large or null")
            for name, expected in MODEL_SIZE_PRESETS[self.model_size].items():
                value = getattr(self, name)
                if value is not None and value != expected:
                    raise ValueError(f"{name}={value!r} conflicts with model_size={self.model_size!r} "
                                     f"(expected {expected!r})")
                setattr(self, name, expected)
        else:
            for name, default in _LEGACY_STRUCTURE.items():
                if getattr(self, name) is None:
                    setattr(self, name, default)

    def validate(self):
        for name in ("clinical_dim", "dim", "encoder_depth", "fusion_depth",
                     "predictor_depth", "heads", "state_queries", "pcr_queries",
                     "fm_channels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.attention_head_dim is None and self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        for name in ("attention_head_dim", "ffn_dim"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer or null")
        if self.model_size is not None:
            if not isinstance(self.model_size, str) or self.model_size not in MODEL_SIZE_PRESETS:
                raise ValueError("model_size must be small, medium, large or null")
            for name, expected in MODEL_SIZE_PRESETS[self.model_size].items():
                if getattr(self, name) != expected:
                    raise ValueError(f"{name} disagrees with model_size={self.model_size!r}")
        selected_phases(self)
        if (len(self.token_grid) != 3 or
                any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                    for v in self.token_grid)):
            raise ValueError("token_grid must have three positive integer dimensions")
        if not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must lie in [0,1)")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0,1)")
        if not isinstance(self.enable_flow, bool):
            raise ValueError("enable_flow must be boolean")
        if not isinstance(self.enable_world_model, bool):
            raise ValueError("enable_world_model must be boolean")
        if not isinstance(self.encoder_type, str) or self.encoder_type not in {"raw", "vq_prequant", "vq_encoder"}:
            raise ValueError("encoder_type must be raw, vq_prequant or vq_encoder")
        for name in ("fixed_transition_queries", "world_residual", "history_delta",
                     "lesion_world_weighting", "lesion_pooling"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.pcr_readout not in {"direct", "zero_state", "copy", "forecast_detach", "forecast"}:
            raise ValueError("Unsupported pcr_readout")
        if not self.enable_world_model and (self.pcr_readout not in {"direct", "zero_state"} or self.enable_flow):
            raise ValueError("Disabled world model requires direct pCR readout and enable_flow=false")
        if self.pcr_readout == "zero_state" and self.enable_world_model:
            raise ValueError("zero_state pCR readout requires enable_world_model=false")
        if not isinstance(self.spatial_readout, str) or self.spatial_readout not in {"mean", "attention"}:
            raise ValueError("spatial_readout must be mean or attention")
        if (not isinstance(self.pcr_delta_reference, str) or
                self.pcr_delta_reference not in {"current", "t0"}):
            raise ValueError("pcr_delta_reference must be current or t0")
        if self.pcr_readout not in {"direct", "zero_state"} and not self.fixed_transition_queries:
            raise ValueError("State pCR readouts require fixed_transition_queries")
        if (isinstance(self.lesion_weight_strength, bool) or
                not math.isfinite(self.lesion_weight_strength) or self.lesion_weight_strength < 0):
            raise ValueError("lesion_weight_strength must be finite and nonnegative")
        if not isinstance(self.vq_activation_checkpointing, bool):
            raise ValueError("vq_activation_checkpointing must be boolean")
        for name in ("vq_hidden_channels", "vq_embedding_dim", "vq_num_groups"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.vq_bottleneck_blocks, bool) or
                not isinstance(self.vq_bottleneck_blocks, int) or self.vq_bottleneck_blocks < 0):
            raise ValueError("vq_bottleneck_blocks must be a nonnegative integer")
        if self.vq_hidden_channels * 2 % self.vq_num_groups:
            raise ValueError("vq_num_groups must divide both VQ residual channel counts")
        return self


@dataclass
class RawVLAJEPAConfig(RawJEPAConfig):
    enable_flow: bool = False

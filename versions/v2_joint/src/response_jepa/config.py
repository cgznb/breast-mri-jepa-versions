"""Strict, serializable configurations: reject misspellings instead of ignoring them."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any
import math
import yaml

@dataclass
class ModelConfig:
    encoder: str = "multiscale"           # multiscale | legacy_small
    dim: int = 128
    heads: int = 4
    token_grid: list[int] = field(default_factory=lambda: [4, 8, 8])
    widths: list[int] = field(default_factory=lambda: [24, 48, 96, 128])
    depths: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    spatial_depth: int = 2
    fusion_depth: int = 2
    predictor_depth: int = 4
    context_queries: int = 32
    state_queries: int = 8
    pcr_queries: int = 4
    dropout: float = 0.1
    world_arch: str = "transformer"       # transformer | linear (control)
    residual_world: bool = True
    readout: str = "gated"                # direct | forecast | gated | copy_gated
    forecast_detach: bool = False
    use_mri: bool = True
    use_clinical: bool = True
    use_arm: bool = True
    encoder_chunk: int = 4
    activation_checkpoint: bool = False
    gate_bias: float = -2.0

@dataclass
class LossConfig:
    masked: float = 1.0
    variance: float = 1.0
    covariance: float = 0.04
    token_variance: float = 0.1
    repr_pcr: float = 0.0                 # nonzero explicitly makes A label-assisted
    tf: float = 0.25
    rollout: float = 1.0
    fused: float = 1.0
    direct: float = 0.5
    forecast: float = 0.25
    align: float = 0.0                    # observed online -> frozen teacher when fine-tuning
    mask_ratio: float = 0.35
    variance_floor: float = 1.0
    token_variance_floor: float = 0.25
    consistency: float = 0.0
    consistency_warmup_updates: int = 500

@dataclass
class TrainConfig:
    phase: str = "C"                     # A representation | B world | C task | joint
    profile: str = "t0"                  # t0 | dynamic | single_t0 | single_t3
    seed: int = 17
    epochs: int = 60
    patience: int = 15
    min_delta: float = 0.00001
    batch_size: int = 4
    accumulation: int = 4
    lr: float = 0.0001
    backbone_lr: float = 0.00003
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.05
    clip_grad: float = 1.0
    precision: str = "bf16"              # CPU always FP32; no unscaled FP16 path
    device: str = "cuda"
    workers: int = 0
    pin_memory: bool = True
    freeze_encoder: bool = True
    ema: float = 0.996                    # A/joint; teacher is fixed in staged B/C
    prefix_sampling: str = "balanced_cycle" # balanced_cycle | random | all_prefixes
    augment: bool = False
    threads: int = 4
    deterministic: bool = True
    eval_batch_size: int = 4
    world_eval: bool = True
    save_predictions: bool = False
    allow_synthetic: bool = False
    image_shape: list[int] = field(default_factory=lambda: [32, 128, 128])
    image_cache: str | None = None
    sample_weighting: str = "sample"      # sample | patient_equal (all-prefix joint)
    reference_batch_size: int | None = None
    max_updates: int | None = None         # opt-in update-boundary joint runner
    warmup_updates: int = 200
    eval_every_updates: int = 100
    patience_updates: int = 500
    train_eval_patients: int = 0

@dataclass
class Config:
    schema: str = "breast_response_jepa_v2"
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> "Config":
        m, l, t = self.model, self.loss, self.train
        if self.schema != "breast_response_jepa_v2":
            raise ValueError("Wrong config schema; upstream configs are not v2 configs")
        if m.encoder not in {"multiscale", "legacy_small"}:
            raise ValueError("Unknown encoder")
        for name in ("dim", "heads", "spatial_depth", "fusion_depth", "predictor_depth",
                     "context_queries", "state_queries", "pcr_queries", "encoder_chunk"):
            if type(getattr(m, name)) is not int or getattr(m, name) < 1:
                raise ValueError(f"model.{name} must be a positive integer")
        if m.dim % m.heads:
            raise ValueError("dim must be divisible by heads")
        for name, length in (("token_grid", 3), ("widths", 4), ("depths", 4)):
            value = getattr(m, name)
            if len(value) != length or any(type(v) is not int or v < 1 for v in value):
                raise ValueError(f"Invalid model.{name}")
        if not 0 <= m.dropout < 1 or not math.isfinite(m.gate_bias):
            raise ValueError("Invalid dropout/gate_bias")
        if m.world_arch not in {"transformer", "linear"}:
            raise ValueError("Unknown world architecture")
        if m.readout not in {"direct", "forecast", "gated", "copy_gated"}:
            raise ValueError("Unknown readout")
        if t.phase not in {"A", "B", "C", "joint"} or t.profile not in {"t0", "dynamic", "single_t0", "single_t3"}:
            raise ValueError("Unknown phase/profile")
        if t.profile.startswith("single_") and (t.phase != "C" or m.readout != "direct"):
            raise ValueError("Single-visit references are C/direct only")
        if not m.use_mri and (t.phase != "C" or m.readout != "direct" or l.tf or l.rollout):
            raise ValueError("Clinical baselines must be C/direct with tf=rollout=0")
        if t.phase not in {"A", "joint"} and t.augment:
            raise ValueError("Augmentation is allowed only in A/joint representation branches")
        if t.phase == "B" and not t.freeze_encoder:
            raise ValueError("B requires a frozen encoder and fixed target coordinates")
        if t.phase == "C" and not t.freeze_encoder and (l.tf or l.rollout) and l.align <= 0:
            raise ValueError("C encoder fine-tuning with world losses requires align>0")
        if t.phase == "B" and not (l.tf or l.rollout):
            raise ValueError("B needs at least one world objective")
        if t.phase == "A" and (t.batch_size < 2 or not m.use_mri):
            raise ValueError("A needs MRI and >=2 physical patients per batch; accumulation does not fix variance")
        if t.phase == "joint":
            if not m.use_mri or t.freeze_encoder or t.batch_size < 2:
                raise ValueError("Joint training requires MRI, an unfrozen encoder and >=2 physical patients")
            if not (l.tf or l.rollout) or not (l.masked or l.variance or l.covariance or l.token_variance):
                raise ValueError("Joint training requires world and representation objectives")
            if l.fused <= 0:
                raise ValueError("Joint training requires the pCR objective")
        for name in ("epochs", "patience", "batch_size", "accumulation", "eval_batch_size", "threads"):
            if type(getattr(t, name)) is not int or getattr(t, name) < 1:
                raise ValueError(f"train.{name} must be positive")
        if type(t.workers) is not int or t.workers < 0:
            raise ValueError("workers must be nonnegative")
        if not 0 <= t.warmup_epochs < t.epochs:
            raise ValueError("warmup_epochs must be >=0 and <epochs")
        if len(t.image_shape) != 3 or any(type(v) is not int or v < 8 for v in t.image_shape):
            raise ValueError("image_shape must have three integer dimensions >=8")
        if any(g > s for g, s in zip(m.token_grid, t.image_shape)):
            raise ValueError("Token grid cannot exceed the input grid")
        for name in ("lr", "backbone_lr", "clip_grad"):
            if not math.isfinite(getattr(t, name)) or getattr(t, name) <= 0:
                raise ValueError(f"Invalid train.{name}")
        if any(not math.isfinite(getattr(t, k)) or getattr(t, k) < 0 for k in ("min_delta", "weight_decay")):
            raise ValueError("min_delta and weight_decay must be finite and nonnegative")
        if not 0 <= t.ema < 1 or not 0 < t.min_lr_ratio <= 1:
            raise ValueError("Invalid EMA or min LR ratio")
        if t.precision not in {"fp32", "bf16"} or t.prefix_sampling not in {"balanced_cycle", "random", "all_prefixes"}:
            raise ValueError("Invalid precision/sampling")
        if t.prefix_sampling == "all_prefixes" and (t.profile != "dynamic" or t.phase not in {"B", "C", "joint"}):
            raise ValueError("All-prefix sampling requires dynamic profile and B/C/joint phase")
        if t.sample_weighting not in {"sample", "patient_equal"}:
            raise ValueError("Unknown sample weighting")
        if t.sample_weighting == "patient_equal":
            if t.phase != "joint" or (t.profile == "dynamic" and t.prefix_sampling != "all_prefixes"):
                raise ValueError("Patient-equal weighting requires joint and all dynamic prefixes")
        if t.reference_batch_size is not None and (type(t.reference_batch_size) is not int or t.reference_batch_size < 1):
            raise ValueError("reference_batch_size must be a positive integer")
        for name in ("warmup_updates", "train_eval_patients"):
            if type(getattr(t, name)) is not int or getattr(t, name) < 0:
                raise ValueError(f"train.{name} must be a nonnegative integer")
        for name in ("eval_every_updates", "patience_updates"):
            if type(getattr(t, name)) is not int or getattr(t, name) < 1:
                raise ValueError(f"train.{name} must be a positive integer")
        if t.max_updates is not None:
            if type(t.max_updates) is not int or t.max_updates < 1 or t.warmup_updates >= t.max_updates:
                raise ValueError("max_updates must exceed nonnegative warmup_updates")
            if t.phase != "joint" or t.accumulation != 1:
                raise ValueError("Update-based training currently requires joint and accumulation=1")
        if l.consistency and (t.phase != "joint" or t.max_updates is None):
            raise ValueError("Consistency requires update-based joint training")
        if type(l.consistency_warmup_updates) is not int or l.consistency_warmup_updates < 0:
            raise ValueError("consistency_warmup_updates must be a nonnegative integer")
        if not 0 < l.mask_ratio < 1:
            raise ValueError("mask_ratio must be in (0,1)")
        for f in fields(l):
            value = getattr(l, f.name)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid loss.{f.name}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strict(cls, values: dict):
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} must be an object")
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def from_dict(d: dict) -> Config:
    if set(d) - {"schema", "model", "loss", "train"}:
        raise ValueError("Unknown top-level config keys")
    return Config(d.get("schema", "breast_response_jepa_v2"),
                  _strict(ModelConfig, d.get("model", {})),
                  _strict(LossConfig, d.get("loss", {})),
                  _strict(TrainConfig, d.get("train", {}))).validate()


def load_config(path: str | Path, overrides: list[str] = ()) -> Config:
    with open(path, encoding="utf-8") as f:
        cfg = from_dict(yaml.safe_load(f))
    d = cfg.to_dict()
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or key.count(".") != 1:
            raise ValueError("Overrides require section.key=value, e.g. train.device=cpu")
        section, name = key.split(".")
        if section not in {"model", "loss", "train"} or name not in d[section]:
            raise ValueError(f"Unknown override: {key}")
        d[section][name] = yaml.safe_load(value)
    return from_dict(d)

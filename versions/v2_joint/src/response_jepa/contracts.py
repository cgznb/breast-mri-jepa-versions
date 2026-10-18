"""Keep legal inputs separate from future supervision, including during training."""
from __future__ import annotations
from dataclasses import dataclass, fields
import torch

class TensorRecord:
    def to(self, device, non_blocking=False):
        return type(self)(**{f.name: getattr(self, f.name).to(device, non_blocking=non_blocking)
                             for f in fields(self)})
    def pin_memory(self):
        return type(self)(**{f.name: getattr(self, f.name).pin_memory() for f in fields(self)})

@dataclass(frozen=True)
class Inputs(TensorRecord):
    images: torch.Tensor          # [B,4,3,D,H,W], all unobserved slots exactly zero
    observed: torch.Tensor        # [B,4], legal known history only
    landmark: torch.Tensor        # [B], T0..T3
    clinical: torch.Tensor        # [B,4], normalized age, HR, HER2, MP
    clinical_mask: torch.Tensor   # [B,4]
    arm_id: torch.Tensor          # [B], 0 unknown, 1..13 official category
    arm_known_at: torch.Tensor    # [B], -1 unknown
    query_mask: torch.Tensor      # [B,4], OUTPUT REQUEST ONLY; not used by backbone

    def validate(self):
        if self.images.ndim != 6 or tuple(self.images.shape[1:3]) != (4, 3):
            raise ValueError("images must have shape [B,4,3,D,H,W]")
        b = len(self.images)
        if b == 0 or any(x < 1 for x in self.images.shape):
            raise ValueError("Empty input")
        for name in ("observed", "clinical_mask", "query_mask"):
            x = getattr(self, name)
            if x.shape != (b, 4) or x.dtype != torch.bool:
                raise ValueError(f"{name} must be bool [B,4]")
        for name in ("landmark", "arm_id", "arm_known_at"):
            x = getattr(self, name)
            if x.shape != (b,) or x.dtype != torch.long:
                raise ValueError(f"{name} must be int64 [B]")
        if self.clinical.shape != (b, 4):
            raise ValueError("clinical must have shape [B,4]")
        stages = torch.arange(4, device=self.images.device)[None]
        if ((self.landmark < 0) | (self.landmark > 3)).any():
            raise ValueError("Invalid landmark")
        if (self.observed & (stages > self.landmark[:, None])).any():
            raise ValueError("Future MRI entered the legal input")
        if not self.observed.any(1).all():
            raise ValueError("Every patient needs an observed visit")
        if (self.query_mask & (stages <= self.landmark[:, None])).any():
            raise ValueError("Only future stages can be requested")
        if self.images[~self.observed].count_nonzero():
            raise ValueError("Hidden image slots must be zero")
        if self.clinical[~self.clinical_mask].count_nonzero():
            raise ValueError("Unavailable clinical values must be zero")
        if not torch.isfinite(self.images).all() or not torch.isfinite(self.clinical).all():
            raise ValueError("Non-finite legal input")
        if ((self.arm_id < 0) | (self.arm_id > 13)).any():
            raise ValueError("Arm category is out of range")
        known = self.arm_id > 0
        if (known & ((self.arm_known_at < 0) | (self.arm_known_at > self.landmark))).any():
            raise ValueError("Arm is not available at the landmark")
        if ((~known) & (self.arm_known_at != -1)).any():
            raise ValueError("Unknown arm must have known_at=-1")
        return self

    def anchor(self):
        stages = torch.arange(4, device=self.images.device)[None].expand_as(self.observed)
        return stages.masked_fill(~self.observed, -1).max(1).values

@dataclass(frozen=True)
class Targets(TensorRecord):
    future: torch.Tensor           # [B,4,3,D,H,W], separate from Inputs
    future_mask: torch.Tensor      # [B,4], actually available target scans
    label: torch.Tensor            # [B], final pCR, zero where missing
    label_mask: torch.Tensor       # [B]

    def validate(self, inp):
        b = len(inp.images)
        if self.future.shape != inp.images.shape or self.future_mask.shape != (b, 4):
            raise ValueError("Wrong future target shape")
        if self.future_mask.dtype != torch.bool or self.label_mask.dtype != torch.bool:
            raise ValueError("Supervision masks must be boolean")
        stages = torch.arange(4, device=inp.images.device)[None]
        if (self.future_mask & (stages <= inp.landmark[:, None])).any():
            raise ValueError("Future supervision must be strictly after landmark")
        # No dependency on inp.query_mask: changing requested OUTPUTS must not change training.
        if self.label.shape != (b,) or self.label_mask.shape != (b,):
            raise ValueError("Wrong label shape")
        if not torch.isfinite(self.future[self.future_mask]).all():
            raise ValueError("Non-finite observed target")
        label = self.label[self.label_mask]
        if not torch.isfinite(label).all() or ((label != 0) & (label != 1)).any():
            raise ValueError("pCR must be binary or missing")
        return self

@dataclass
class Batch:
    inputs: Inputs
    targets: Targets
    keys: tuple[str, ...]
    sample_weight: torch.Tensor | None = None  # [B], fixed population weight; never renormalize per batch
    def to(self, device, non_blocking=False):
        weight = None if self.sample_weight is None else self.sample_weight.to(device, non_blocking=non_blocking)
        return Batch(self.inputs.to(device, non_blocking), self.targets.to(device, non_blocking), self.keys, weight)
    def pin_memory(self):
        weight = None if self.sample_weight is None else self.sample_weight.pin_memory()
        return Batch(self.inputs.pin_memory(), self.targets.pin_memory(), self.keys, weight)

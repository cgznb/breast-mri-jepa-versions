"""Patient-axis SIGReg on online MRI states, adapted from LeWM.

Reference: lucas-maes/le-wm, module.py, commit
8edfeb336732b5f3ce7b8b210d0ba370a09e2cac. The statistic, Gaussian window,
integration weights and sample-count scaling follow the reference. The local
adapter treats each stage/token as a group and patients as the sample axis.
There are no learned parameters, input normalization or extra encoder passes.

MIT License
Copyright (c) 2026 Lucas Maes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def _validate_settings(num_projections, num_points):
    if type(num_projections) is not int or num_projections < 1:
        raise ValueError("sigreg_num_projections must be a positive integer")
    if type(num_points) is not int or num_points < 2:
        raise ValueError("sigreg_num_points must be an integer >= 2")


def sigreg_loss(features, *, num_projections=256, num_points=17, projections=None):
    """Return LeWM's SIGReg for ``[groups, patients, channels]`` in FP32.

    Each group has its own empirical distribution along the patient axis.
    Gaussian random directions are sampled on every call using PyTorch's
    checkpointed RNG. ``projections`` optionally supplies raw direction columns
    for reproducible diagnostics and reference tests; they are L2 normalized,
    detached, and never optimized. The input features are not normalized.
    """
    _validate_settings(num_projections, num_points)
    if (not isinstance(features, torch.Tensor) or features.ndim != 3
            or min(features.shape) < 1 or not features.is_floating_point()):
        raise ValueError("SIGReg features must be a nonempty floating [groups, patients, channels] tensor")
    if projections is not None and (
            not isinstance(projections, torch.Tensor) or not projections.is_floating_point()
            or projections.shape != (features.shape[-1], num_projections)):
        raise ValueError("SIGReg projection directions must have shape [channels, num_projections]")
    # Trigonometric functions and empirical characteristic functions need FP32
    # even when called inside the model's CUDA BF16 autocast context.
    with torch.autocast(device_type=features.device.type, enabled=False):
        x = features.float()
        if projections is None:
            directions = torch.randn(x.shape[-1], num_projections, device=x.device,
                                     dtype=torch.float32)
        else:
            directions = projections.detach().to(device=x.device, dtype=torch.float32)
            if not bool(torch.isfinite(directions).all()) or bool((directions.norm(dim=0) == 0).any()):
                raise ValueError("SIGReg directions must be finite and nonzero")
        directions = F.normalize(directions, p=2, dim=0)
        t = torch.linspace(0, 3, num_points, device=x.device, dtype=torch.float32)
        dt = 3 / (num_points - 1)
        weights = torch.full((num_points,), 2 * dt, device=x.device, dtype=torch.float32)
        weights[[0, -1]] = dt
        gaussian_cf = torch.exp(-t.square() / 2)
        weights = weights * gaussian_cf
        angles = (x @ directions).unsqueeze(-1) * t
        error = ((angles.cos().mean(dim=-3) - gaussian_cf).square()
                 + angles.sin().mean(dim=-3).square())
        # Match the upstream doubled trapezoid weights and patient-count factor.
        return ((error @ weights) * x.shape[-2]).mean()


def observed_state_sigreg(observed, observed_mask, *, num_projections=256, num_points=17):
    """Regularize matching stage/token states across distinct patients.

    ``observed`` is the existing online encoder output ``[M, tokens, channels]``
    in boolean-index order, where M equals the number of True entries in the
    ``[patients, stages]`` mask. Each stage is processed independently, so
    spatial or timepoint differences cannot stand in for patient diversity.
    Stages with fewer than two patients are skipped. Eligible stages receive
    equal weight, matching the existing patient-axis variance regularizer.
    """
    _validate_settings(num_projections, num_points)
    if (not isinstance(observed, torch.Tensor) or observed.ndim != 3
            or not observed.is_floating_point() or min(observed.shape[1:]) < 1):
        raise ValueError("Online observed states must have shape [observed visits, tokens, channels]")
    if (not isinstance(observed_mask, torch.Tensor) or observed_mask.ndim != 2
            or observed_mask.dtype != torch.bool or observed_mask.device != observed.device):
        raise ValueError("Observed mask must be a boolean [patients, stages] tensor on the states' device")
    stages = observed_mask.nonzero(as_tuple=True)[1]
    if len(stages) != len(observed):
        raise ValueError("Online observed states do not match the observed-mask visit count")
    penalties = []
    for stage in range(observed_mask.shape[1]):
        values = observed[stages == stage]
        if len(values) >= 2:
            penalties.append(sigreg_loss(values.transpose(0, 1), num_projections=num_projections,
                                         num_points=num_points))
    return torch.stack(penalties).mean() if penalties else observed.float().sum() * 0

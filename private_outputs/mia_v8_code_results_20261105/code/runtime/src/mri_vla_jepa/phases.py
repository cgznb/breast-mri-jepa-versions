"""Canonical MRI phase selection within the unchanged three-phase contract."""
from __future__ import annotations

import torch


PHASES = ("pre_aqc0", "first_post_aqc1", "metadata_late")


def selected_phases(cfg_or_name):
    """Return the ordered phase names selected by a config or phase string."""
    name = cfg_or_name if isinstance(cfg_or_name, str) else cfg_or_name.mri_phases
    if name == "all":
        return PHASES
    if not isinstance(name, str) or name not in PHASES:
        raise ValueError("mri_phases must be all, pre_aqc0, first_post_aqc1 or metadata_late")
    return (name,)


def phase_indices(cfg_or_name):
    return tuple(PHASES.index(name) for name in selected_phases(cfg_or_name))


def select_phases(images, cfg_or_name, channel_dim=-4):
    """Select model channels from [*,3,D,H,W] without altering data storage.

    The explicit three-channel check catches accidental double-selection and
    keeps online and EMA encoder inputs on exactly the same data contract.
    """
    if images.ndim < 4:
        raise ValueError("MRI phase selection expects at least four dimensions")
    channel_dim %= images.ndim
    if images.shape[channel_dim] != len(PHASES):
        raise ValueError("MRI phase selection expects the original three-phase input")
    indices = phase_indices(cfg_or_name)
    if len(indices) == len(PHASES):
        return images
    return images.index_select(channel_dim, torch.tensor(indices, device=images.device))

"""Prediction-source profiles; future MRI targets retain all four visits.

Profiles restrict sampled/evaluated landmarks only. They never alter a patient's
visits, future query mask, teacher states or JEPA targets.
"""
from __future__ import annotations


def profile_stages(profile):
    """Return the permitted source stages for a named profile or one stage."""
    if type(profile) is int and 0 <= profile <= 3:
        return (profile,)
    if isinstance(profile, str):
        profiles = {"t0": (0,), "t0_t2": (0, 1, 2), "all_observed": (0, 1, 2, 3)}
        if profile in profiles:
            return profiles[profile]
    raise ValueError("Landmark profile must be t0, t0_t2, all_observed or an integer T0..T3")


def allowed_landmarks(store, index, profile):
    """Keep only actual MRI availability stages admitted by the profile."""
    admitted = profile_stages(profile)
    return [stage for stage in store.allowed_landmarks(index) if stage in admitted]


def sample_landmark(store, index, profile, rng):
    """Sample a legal source stage without changing historical fixed-T0 RNG use."""
    stages = allowed_landmarks(store, index, profile)
    if not stages:
        raise ValueError(f"Patient index {index} has no available landmark in profile {profile!r}")
    return 0 if profile == "t0" else rng.choice(stages)

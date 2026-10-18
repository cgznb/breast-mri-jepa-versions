from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

from .geometry import sitk_image_from_array
from .models import Geometry
from .rigid import RigidRegistrationError, RigidResult, estimate_rigid
from .transforms import TransformApplicationError, resample_rigid


class IntraVisitRegistrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class RegistrationRegion:
    slices_zyx: tuple[slice, slice, slice]
    used_full_fov: bool


@dataclass(frozen=True)
class PhaseRegistration:
    phase_id: int
    transform: sitk.Euler3DTransform
    metric_value: float
    iterations: int
    stop_condition: str
    sampling_percentage: float
    seed: int


@dataclass(frozen=True)
class PhaseAlignmentResult:
    phase_ids: tuple[int, ...]
    aligned_phases: tuple[np.ndarray, ...]
    registrations: tuple[PhaseRegistration, ...]
    region: RegistrationRegion


def _full_region(shape: tuple[int, int, int]) -> RegistrationRegion:
    return RegistrationRegion(
        slices_zyx=tuple(slice(0, size) for size in shape),
        used_full_fov=True,
    )


def registration_region(
    ftv_mask_zyx: np.ndarray,
    *,
    xy_margin: int = 10,
    z_margin: int = 5,
) -> RegistrationRegion:
    mask = np.asarray(ftv_mask_zyx)
    if mask.ndim != 3:
        raise ValueError("FTV mask must be a 3D z,y,x array")
    if xy_margin < 0 or z_margin < 0:
        raise ValueError("registration margins must be nonnegative")
    foreground = np.isfinite(mask) & (mask > 0)
    if not np.any(foreground):
        return _full_region(mask.shape)
    indices = np.nonzero(foreground)
    margins = (z_margin, xy_margin, xy_margin)
    slices = tuple(
        slice(
            max(0, int(np.min(axis)) - margin),
            min(size, int(np.max(axis)) + margin + 1),
        )
        for axis, margin, size in zip(indices, margins, mask.shape, strict=True)
    )
    return RegistrationRegion(slices_zyx=slices, used_full_fov=False)


def _clean_phase(array: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    phase = np.asarray(array, dtype=np.float32)
    if phase.shape != shape:
        raise ValueError("all DCE phases must match the visit geometry")
    clean = phase.copy()
    clean[~np.isfinite(clean)] = 0.0
    clean[clean < 0.0] = 0.0
    return clean


def _usable_driver(array: np.ndarray) -> bool:
    values = np.asarray(array)
    return values.size > 1 and bool(np.max(values) > np.min(values))


def _region_center_physical(
    geometry: Geometry,
    slices_zyx: tuple[slice, slice, slice],
) -> tuple[float, float, float]:
    z_slice, y_slice, x_slice = slices_zyx
    index_xyz = (
        (x_slice.start + x_slice.stop - 1.0) / 2.0,
        (y_slice.start + y_slice.stop - 1.0) / 2.0,
        (z_slice.start + z_slice.stop - 1.0) / 2.0,
    )
    image = sitk_image_from_array(
        np.zeros(geometry.shape_zyx, dtype=np.uint8), geometry
    )
    return image.TransformContinuousIndexToPhysicalPoint(index_xyz)


def _registration_for_phase(
    phase_id: int,
    result: RigidResult,
    center: tuple[float, float, float],
    seed: int,
) -> PhaseRegistration:
    transform = sitk.Euler3DTransform(result.transform)
    transform.SetCenter(center)
    return PhaseRegistration(
        phase_id=phase_id,
        transform=transform,
        metric_value=result.metric_value,
        iterations=result.iterations,
        stop_condition=result.stop_condition,
        sampling_percentage=result.sampling_percentage,
        seed=seed,
    )


def align_dce_phases(
    phases: Sequence[np.ndarray],
    phase_ids: Sequence[int],
    ftv_mask_zyx: np.ndarray,
    geometry: Geometry,
    *,
    samples: int = 5000,
    iterations: int = 500,
    seed: int = 20260719,
    learning_rate: float = 2.0,
    min_step: float = 1e-4,
    relaxation_factor: float = 0.5,
    shrink_factors: tuple[int, ...] = (4, 2, 1),
    smoothing_sigmas: tuple[float, ...] = (2.0, 1.0, 0.0),
) -> PhaseAlignmentResult:
    ids = tuple(int(value) for value in phase_ids)
    if (
        len(ids) != len(phases)
        or not ids
        or ids[0] != 0
        or len(set(ids)) != len(ids)
        or any(value < 0 for value in ids)
    ):
        raise ValueError("phase IDs must be unique, start at zero, and match DCE paths")
    mask = np.asarray(ftv_mask_zyx)
    if mask.shape != geometry.shape_zyx:
        raise ValueError("FTV mask must match the visit geometry")
    clean_phases = tuple(_clean_phase(phase, geometry.shape_zyx) for phase in phases)
    region = registration_region(mask)
    fixed_crop = clean_phases[0][region.slices_zyx]
    if not _usable_driver(fixed_crop) or any(
        not _usable_driver(phase[region.slices_zyx]) for phase in clean_phases[1:]
    ):
        region = _full_region(geometry.shape_zyx)

    def align_in_region(selected_region: RegistrationRegion) -> PhaseAlignmentResult:
        fixed_selected = clean_phases[0][selected_region.slices_zyx]
        selected_geometry = geometry.with_shape(fixed_selected.shape)
        center = _region_center_physical(geometry, selected_region.slices_zyx)
        fixed_image = sitk_image_from_array(fixed_selected, selected_geometry)
        aligned: list[np.ndarray] = [clean_phases[0]]
        registrations: list[PhaseRegistration] = []
        for phase_id, phase in zip(ids[1:], clean_phases[1:], strict=True):
            phase_seed = seed + phase_id
            try:
                result = estimate_rigid(
                    fixed_image,
                    sitk_image_from_array(
                        phase[selected_region.slices_zyx], selected_geometry
                    ),
                    samples=samples,
                    iterations=iterations,
                    seed=phase_seed,
                    learning_rate=learning_rate,
                    min_step=min_step,
                    relaxation_factor=relaxation_factor,
                    shrink_factors=shrink_factors,
                    smoothing_sigmas=smoothing_sigmas,
                )
                registration = _registration_for_phase(
                    phase_id, result, center, phase_seed
                )
                registered = resample_rigid(
                    phase,
                    geometry,
                    geometry,
                    registration.transform,
                    interpolation="linear",
                )
            except (
                RigidRegistrationError,
                TransformApplicationError,
                RuntimeError,
                ValueError,
            ) as exc:
                raise IntraVisitRegistrationError(
                    f"phase {phase_id} rigid registration failed: {exc}"
                ) from exc
            if not np.isfinite(registered).all() or not np.any(registered):
                raise IntraVisitRegistrationError(
                    f"phase {phase_id} rigid registration returned an empty or nonfinite image"
                )
            aligned.append(registered.astype(np.float32, copy=False))
            registrations.append(registration)
        return PhaseAlignmentResult(
            phase_ids=ids,
            aligned_phases=tuple(aligned),
            registrations=tuple(registrations),
            region=selected_region,
        )

    try:
        return align_in_region(region)
    except IntraVisitRegistrationError:
        if region.used_full_fov:
            raise
        return align_in_region(_full_region(geometry.shape_zyx))

"""Physical-geometry preprocessing primitives for longitudinal I-SPY2 MRI."""

from __future__ import annotations

import math
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .motfm_logging import get_logger


LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])
TARGET_SPACING_XYZ = (0.7032, 0.7032, 2.0)
OUTPUT_SHAPE_ZYX = (128, 128, 128)
NORMALIZATION_PERCENTILES = (0.5, 99.5)

logger = get_logger(__name__)


def _finite_vector(value: Any, length: int, field: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain {length} finite numbers") from exc
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"{field} must contain {length} finite numbers")
    return vector


def _positive_vector(value: Any, length: int, field: str) -> np.ndarray:
    vector = _finite_vector(value, length, field)
    if np.any(vector <= 0.0):
        raise ValueError(f"{field} values must be positive")
    return vector


@dataclass(frozen=True)
class PhysicalGeometry:
    """Geometry for a Z/Y/X array described by DICOM LPS metadata."""

    spacing_xyz: tuple[float, float, float]
    origin_lps: tuple[float, float, float]
    direction_lps_axes: tuple[tuple[float, float, float], ...]

    @property
    def direction_lps(self) -> np.ndarray:
        return np.asarray(self.direction_lps_axes, dtype=np.float64)

    @property
    def direction_ras(self) -> np.ndarray:
        return LPS_TO_RAS @ self.direction_lps

    @property
    def origin_ras(self) -> np.ndarray:
        return LPS_TO_RAS @ np.asarray(self.origin_lps, dtype=np.float64)

    @property
    def affine_ras(self) -> np.ndarray:
        affine = np.eye(4, dtype=np.float64)
        affine[:3, :3] = self.direction_ras @ np.diag(self.spacing_xyz)
        affine[:3, 3] = self.origin_ras
        return affine


def geometry_from_meta(meta: Mapping[str, Any]) -> PhysicalGeometry:
    """Parse required DICOM geometry without spacing/orientation fallbacks."""

    pixel_spacing = _positive_vector(meta.get("pixel_spacing"), 2, "pixel_spacing")
    slice_spacing = _positive_vector(
        [meta.get("spacing_between_slices")], 1, "spacing_between_slices"
    )[0]
    orientation = _finite_vector(
        meta.get("image_orientation_patient"), 6, "image_orientation_patient"
    )
    origin = _finite_vector(
        meta.get("image_position_patient_first"), 3, "image_position_patient_first"
    )

    x_axis = orientation[:3]
    y_axis = orientation[3:]
    if not np.isclose(np.linalg.norm(x_axis), 1.0, atol=1e-4):
        raise ValueError("image_orientation_patient first axis must be unit length")
    if not np.isclose(np.linalg.norm(y_axis), 1.0, atol=1e-4):
        raise ValueError("image_orientation_patient second axis must be unit length")
    if not np.isclose(float(np.dot(x_axis, y_axis)), 0.0, atol=1e-4):
        raise ValueError("image_orientation_patient axes must be orthogonal")
    z_axis = np.cross(x_axis, y_axis)
    if not np.isclose(np.linalg.norm(z_axis), 1.0, atol=1e-4):
        raise ValueError("image_orientation_patient axes do not define a slice direction")
    direction = np.column_stack((x_axis, y_axis, z_axis))
    if not np.isclose(np.linalg.det(direction), 1.0, atol=1e-4):
        raise ValueError("image_orientation_patient must define a proper rotation")

    return PhysicalGeometry(
        spacing_xyz=(float(pixel_spacing[1]), float(pixel_spacing[0]), float(slice_spacing)),
        origin_lps=tuple(float(value) for value in origin),
        direction_lps_axes=tuple(tuple(float(value) for value in row) for row in direction),
    )


def resample_zyx_to_ras(
    array: np.ndarray,
    geometry: PhysicalGeometry,
    target_spacing_xyz: Sequence[float] = TARGET_SPACING_XYZ,
    *,
    is_mask: bool,
) -> tuple[np.ndarray, PhysicalGeometry]:
    """Resample a Z/Y/X volume to an axis-aligned RAS grid with nibabel."""

    try:
        import nibabel as nib
        from nibabel.processing import resample_to_output
    except ImportError as exc:
        raise ImportError("I-SPY2 physical resampling requires nibabel") from exc

    source = np.asarray(array)
    if source.ndim != 3:
        raise ValueError(f"expected a 3D Z/Y/X array, got {source.shape}")
    if not np.isfinite(source).all():
        raise ValueError("resampling input contains non-finite values")
    target_spacing = _positive_vector(target_spacing_xyz, 3, "target_spacing_xyz")
    source_xyz = np.transpose(source, (2, 1, 0))
    image = nib.Nifti1Image(source_xyz.astype(np.float32, copy=False), geometry.affine_ras)
    resampled = resample_to_output(
        image,
        voxel_sizes=tuple(float(value) for value in target_spacing),
        order=0 if is_mask else 1,
        mode="constant",
        cval=0.0,
    )
    output_xyz = np.asarray(resampled.dataobj, dtype=np.float32)
    output_zyx = np.transpose(output_xyz, (2, 1, 0))
    if is_mask:
        output_zyx = (output_zyx > 0.5).astype(np.uint8)

    affine = np.asarray(resampled.affine, dtype=np.float64)
    axes = affine[:3, :3]
    spacing = np.linalg.norm(axes, axis=0)
    direction_ras = axes / spacing[None, :]
    output_geometry = PhysicalGeometry(
        spacing_xyz=tuple(float(value) for value in spacing),
        origin_lps=tuple(float(value) for value in (LPS_TO_RAS @ affine[:3, 3])),
        direction_lps_axes=tuple(
            tuple(float(value) for value in row) for row in (LPS_TO_RAS @ direction_ras)
        ),
    )
    return output_zyx, output_geometry


def robust_nonzero_percentile_scale(
    array: np.ndarray,
    *,
    percentiles: Sequence[float] = NORMALIZATION_PERCENTILES,
    eps: float = 1e-6,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Clip and scale finite nonzero voxels while preserving zero background."""

    source = np.asarray(array, dtype=np.float32)
    if not np.isfinite(source).all():
        raise ValueError("MRI volume contains non-finite values")
    percentile_values = _finite_vector(percentiles, 2, "percentiles")
    low_percentile, high_percentile = (float(value) for value in percentile_values)
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
    nonzero = source != 0.0
    foreground = source[nonzero]
    if foreground.size == 0:
        raise ValueError("MRI volume has no finite nonzero voxels")
    low, high = np.percentile(
        foreground.astype(np.float64), (low_percentile, high_percentile)
    )
    if not np.isfinite(low) or not np.isfinite(high) or float(high - low) <= float(eps):
        raise ValueError("MRI nonzero percentile range has no usable dynamic range")

    output = np.zeros_like(source, dtype=np.float32)
    clipped = np.clip(source[nonzero], low, high)
    output[nonzero] = ((clipped - low) / (high - low)).astype(np.float32)
    return output, {
        "low": float(low),
        "high": float(high),
        "low_percentile": low_percentile,
        "high_percentile": high_percentile,
        "nonzero_count": int(foreground.size),
    }


def relative_orientation_angle_degrees(
    source_meta: Mapping[str, Any],
    target_meta: Mapping[str, Any],
) -> float:
    """Return the geodesic SO(3) angle between two DICOM orientations."""

    source = geometry_from_meta(source_meta).direction_lps
    target = geometry_from_meta(target_meta).direction_lps
    relative = source.T @ target
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def orientation_within_tolerance(
    source_meta: Mapping[str, Any],
    target_meta: Mapping[str, Any],
    *,
    max_degrees: float = 5.0,
) -> bool:
    if not math.isfinite(float(max_degrees)) or float(max_degrees) < 0.0:
        raise ValueError("max_degrees must be finite and nonnegative")
    angle = relative_orientation_angle_degrees(source_meta, target_meta)
    return angle <= float(max_degrees) + 1e-9


def crop_start_for_bbox(
    minimum: int,
    maximum: int,
    *,
    output_length: int,
    input_length: int,
) -> int:
    """Choose a centered crop start that includes an inclusive bbox completely."""

    minimum = int(minimum)
    maximum = int(maximum)
    output_length = int(output_length)
    input_length = int(input_length)
    if output_length <= 0 or input_length <= 0:
        raise ValueError("crop lengths must be positive")
    if minimum < 0 or maximum < minimum or maximum >= input_length:
        raise ValueError("bbox must be a valid inclusive interval inside the input")
    if maximum - minimum + 1 > output_length:
        raise ValueError("bbox is larger than the requested crop")

    spatial_lower = min(0, input_length - output_length)
    spatial_upper = max(0, input_length - output_length)
    lower = max(maximum - output_length + 1, spatial_lower)
    upper = min(minimum, spatial_upper)
    if lower > upper:
        raise RuntimeError("no valid crop start can retain the bbox")
    desired = (minimum + maximum + 1 - output_length) // 2
    return min(max(desired, lower), upper)


def _centered_start(center: int, output_length: int, input_length: int) -> int:
    desired = int(center) - int(output_length) // 2
    lower = min(0, int(input_length) - int(output_length))
    upper = max(0, int(input_length) - int(output_length))
    return min(max(desired, lower), upper)


def _crop_or_pad_at_starts(
    array: np.ndarray,
    starts_zyx: Sequence[int],
    output_shape_zyx: Sequence[int],
) -> np.ndarray:
    output_shape = tuple(int(value) for value in output_shape_zyx)
    output = np.zeros((*array.shape[:-3], *output_shape), dtype=array.dtype)
    source_slices: list[slice] = []
    output_slices: list[slice] = []
    for start, requested, available in zip(starts_zyx, output_shape, array.shape[-3:]):
        stop = int(start) + requested
        source_start = max(int(start), 0)
        source_stop = min(stop, int(available))
        output_start = source_start - int(start)
        output_stop = output_start + max(source_stop - source_start, 0)
        source_slices.append(slice(source_start, source_stop))
        output_slices.append(slice(output_start, output_stop))
    output[(Ellipsis, *output_slices)] = array[(Ellipsis, *source_slices)]
    return output


def tumor_center_crop(
    image_czyx: np.ndarray,
    mask_zyx: np.ndarray,
    output_shape_zyx: Sequence[int] = OUTPUT_SHAPE_ZYX,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Crop/pad a visit around its tumor while retaining every voxel when possible."""

    image = np.asarray(image_czyx)
    mask = np.asarray(mask_zyx)
    output_shape = tuple(int(value) for value in output_shape_zyx)
    if len(output_shape) != 3 or any(value <= 0 for value in output_shape):
        raise ValueError("output_shape_zyx must contain three positive integers")
    if image.ndim != 4 or mask.ndim != 3 or image.shape[-3:] != mask.shape:
        raise ValueError(
            f"expected C/Z/Y/X image and matching Z/Y/X mask, got {image.shape} and {mask.shape}"
        )
    binary = mask > 0
    positive = np.argwhere(binary)
    if positive.size == 0:
        raise ValueError("cannot tumor-center a visit with an empty mask")
    minimum = positive.min(axis=0)
    maximum = positive.max(axis=0)
    center = (minimum + maximum) // 2
    bbox_shape = maximum - minimum + 1
    starts: list[int] = []
    for axis in range(3):
        if int(bbox_shape[axis]) <= output_shape[axis]:
            start = crop_start_for_bbox(
                int(minimum[axis]),
                int(maximum[axis]),
                output_length=output_shape[axis],
                input_length=mask.shape[axis],
            )
        else:
            start = _centered_start(
                int(center[axis]), output_shape[axis], int(mask.shape[axis])
            )
        starts.append(start)

    cropped_image = _crop_or_pad_at_starts(image, starts, output_shape)
    cropped_mask = _crop_or_pad_at_starts(binary.astype(np.uint8), starts, output_shape)
    original_count = int(binary.sum())
    cropped_count = int(cropped_mask.sum())
    return cropped_image, cropped_mask, {
        "coordinate_frame": "tumor_centered_per_visit",
        "bbox_min_zyx": [int(value) for value in minimum],
        "bbox_max_zyx": [int(value) for value in maximum],
        "bbox_shape_zyx": [int(value) for value in bbox_shape],
        "center_zyx": [int(value) for value in center],
        "crop_start_zyx": starts,
        "mask_voxel_count_before_crop": original_count,
        "mask_voxel_count_after_crop": cropped_count,
        "mask_fully_retained": original_count == cropped_count,
        "whole_breast_inverse_mapping": False,
    }


def _mask_crop_plan(
    mask_zyx: np.ndarray,
    output_shape_zyx: Sequence[int],
) -> dict[str, Any]:
    binary = np.asarray(mask_zyx) > 0
    output_shape = tuple(int(value) for value in output_shape_zyx)
    if binary.ndim != 3:
        raise ValueError(f"expected a 3D mask, got {binary.shape}")
    if len(output_shape) != 3 or any(value <= 0 for value in output_shape):
        raise ValueError("output_shape_zyx must contain three positive integers")
    positive = np.argwhere(binary)
    if positive.size == 0:
        return {
            "resampled_bbox_min_zyx": [],
            "resampled_bbox_max_zyx": [],
            "resampled_bbox_shape_zyx": [],
            "crop_start_zyx": [],
            "mask_fully_retained": False,
            "mask_retention_reason": "empty_after_nearest_resample",
        }
    minimum = positive.min(axis=0)
    maximum = positive.max(axis=0)
    center = (minimum + maximum) // 2
    bbox_shape = maximum - minimum + 1
    fully_retained = all(
        int(bbox_shape[axis]) <= output_shape[axis] for axis in range(3)
    )
    starts: list[int] = []
    for axis in range(3):
        if int(bbox_shape[axis]) <= output_shape[axis]:
            start = crop_start_for_bbox(
                int(minimum[axis]),
                int(maximum[axis]),
                output_length=output_shape[axis],
                input_length=binary.shape[axis],
            )
        else:
            start = _centered_start(
                int(center[axis]), output_shape[axis], int(binary.shape[axis])
            )
        starts.append(start)
    return {
        "resampled_bbox_min_zyx": [int(value) for value in minimum],
        "resampled_bbox_max_zyx": [int(value) for value in maximum],
        "resampled_bbox_shape_zyx": [int(value) for value in bbox_shape],
        "crop_start_zyx": starts,
        "mask_fully_retained": fully_retained,
        "mask_retention_reason": (
            "retained" if fully_retained else "bbox_exceeds_output_shape"
        ),
    }


def assess_visit_preprocessing(
    visits: pd.DataFrame,
    *,
    target_spacing_xyz: Sequence[float] = TARGET_SPACING_XYZ,
    output_shape_zyx: Sequence[int] = OUTPUT_SHAPE_ZYX,
    progress_every: int = 100,
) -> pd.DataFrame:
    """Validate and resample every visit mask, recording crop-retention evidence."""

    required = {
        "visit_id",
        "study_instance_uid",
        "dce0_path",
        "ser_path",
        "mask_path",
        "meta_path",
        "ftv_voxel_count",
    }
    missing = sorted(required.difference(visits.columns))
    if missing:
        raise ValueError(f"visits are missing preprocessing fields: {missing}")
    if type(progress_every) is not int or progress_every <= 0:
        raise ValueError("progress_every must be a positive integer")
    try:
        import nibabel as nib
        from nibabel.filebasedimages import ImageFileError
    except ImportError as exc:
        raise ImportError("I-SPY2 mask assessment requires nibabel") from exc

    output_rows: list[dict[str, Any]] = []
    source_rows = visits.to_dict(orient="records")
    total = len(source_rows)
    for completed, source in enumerate(source_rows, start=1):
        record_label = f"visit assessment record {completed}"
        meta_path = Path(str(source["meta_path"]))
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise ValueError(f"{record_label}: meta is not readable JSON") from None
        if not isinstance(meta, dict):
            raise ValueError(f"{record_label}: meta must be a JSON object")
        meta_uid = str(meta.get("study_uid") or meta.get("study_instance_uid") or "").strip()
        if meta_uid != str(source["study_instance_uid"]).strip():
            raise ValueError(f"{record_label}: study UID mismatch")
        geometry = geometry_from_meta(meta)
        try:
            mask_image = nib.load(str(source["mask_path"]))
        except (OSError, ValueError, ImageFileError):
            raise ValueError(f"{record_label}: mask NIfTI header is unreadable") from None
        mask_shape = tuple(int(value) for value in mask_image.shape)
        if len(mask_shape) != 3:
            raise ValueError(f"{record_label}: mask must be a 3D NIfTI")
        for channel in ("dce0", "ser"):
            try:
                channel_image = nib.load(str(source[f"{channel}_path"]))
            except (OSError, ValueError, ImageFileError):
                raise ValueError(
                    f"{record_label}: {channel} NIfTI header is unreadable"
                ) from None
            channel_shape = tuple(int(value) for value in channel_image.shape)
            if channel_shape != mask_shape:
                raise ValueError(
                    f"{record_label}: {channel} native shape does not match mask"
                )
        mask = np.asanyarray(mask_image.dataobj)
        if mask.ndim != 3 or not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all():
            raise ValueError(
                f"{record_label}: mask must be a finite binary 3D array"
            )
        source_count = int(np.count_nonzero(mask))
        if source_count != int(source["ftv_voxel_count"]):
            raise ValueError(f"{record_label}: mask voxel count mismatch")
        resampled, _ = resample_zyx_to_ras(
            mask,
            geometry,
            target_spacing_xyz=target_spacing_xyz,
            is_mask=True,
        )
        plan = _mask_crop_plan(resampled, output_shape_zyx)
        row = dict(source)
        row.update(
            {
                "orientation_lps_json": json.dumps(
                    geometry.direction_lps.tolist(), separators=(",", ":")
                ),
                "resampled_shape_zyx_json": json.dumps(
                    [int(value) for value in resampled.shape], separators=(",", ":")
                ),
                "mask_voxel_count_resampled": int(np.count_nonzero(resampled)),
                "resampled_bbox_min_zyx_json": json.dumps(
                    plan["resampled_bbox_min_zyx"], separators=(",", ":")
                ),
                "resampled_bbox_max_zyx_json": json.dumps(
                    plan["resampled_bbox_max_zyx"], separators=(",", ":")
                ),
                "resampled_bbox_shape_zyx_json": json.dumps(
                    plan["resampled_bbox_shape_zyx"], separators=(",", ":")
                ),
                "crop_start_zyx_json": json.dumps(
                    plan["crop_start_zyx"], separators=(",", ":")
                ),
                "mask_fully_retained": bool(plan["mask_fully_retained"]),
                "mask_retention_reason": str(plan["mask_retention_reason"]),
            }
        )
        output_rows.append(row)
        if completed % progress_every == 0 or completed == total:
            logger.info(
                "I-SPY2 visit assessment: completed %d/%d visits",
                completed,
                total,
            )
    return pd.DataFrame(output_rows)


def preprocess_contract(
    *,
    output_shape_zyx: Sequence[int] = OUTPUT_SHAPE_ZYX,
    target_spacing_xyz: Sequence[float] = TARGET_SPACING_XYZ,
) -> dict[str, Any]:
    """Return the immutable preprocessing policy recorded in every bundle."""

    output_shape = [int(value) for value in output_shape_zyx]
    target_spacing = [float(value) for value in target_spacing_xyz]
    if len(output_shape) != 3 or any(value <= 0 for value in output_shape):
        raise ValueError("output_shape_zyx must contain three positive integers")
    _positive_vector(target_spacing, 3, "target_spacing_xyz")
    return {
        "schema_version": "motfm_ispy2_preprocess_v1",
        "array_order": "zyx",
        "source_geometry_fields": [
            "pixel_spacing",
            "spacing_between_slices",
            "image_orientation_patient",
            "image_position_patient_first",
        ],
        "source_coordinate_system": "DICOM_LPS",
        "target_coordinate_system": "axis_aligned_RAS",
        "target_spacing_xyz": target_spacing,
        "output_shape_zyx": output_shape,
        "image_interpolation": "linear",
        "mask_interpolation": "nearest",
        "normalization": {
            "scope": "finite_nonzero_voxels_per_volume",
            "nonzero_percentiles": list(NORMALIZATION_PERCENTILES),
            "output_range": [0.0, 1.0],
            "zero_background_preserved": True,
        },
        "crop_policy": "bbox_complete_tumor_centered_per_visit",
        "full_resampled_mask_retention_required": True,
    }

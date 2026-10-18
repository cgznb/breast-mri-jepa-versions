"""Lazy longitudinal I-SPY2 dataset and validated per-visit ROI cache."""

from __future__ import annotations

import fcntl
import json
import math
import numbers
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from scripts.build_ispy2_longitudinal_bundle import ARTIFACT_FILENAMES, verify_bundle

from .ispy2_preprocessing import (
    NORMALIZATION_PERCENTILES,
    geometry_from_meta,
    preprocess_contract,
    resample_zyx_to_ras,
    robust_nonzero_percentile_scale,
    tumor_center_crop,
)
from .ispy2_registry import ContractError, sha256_file, sha256_json
from .motfm_logging import get_logger


CACHE_SCHEMA = "motfm_ispy2_visit_roi_cache_v2"
CHANNEL_ORDER = ("dce0", "ser")
VALID_FOLDS = ("train", "val", "test")
SOURCE_ASSET_FIELDS = {
    "dce0": "dce0_path",
    "ser": "ser_path",
    "mask": "mask_path",
    "meta": "meta_path",
}
STAT_SIGNATURE_KEYS = {"st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"}
MAX_STABLE_CACHE_READ_ATTEMPTS = 2
STAGE_BY_TRANSITION = {"T0->T1": 1, "T1->T2": 2, "T2->T3": 3}
MODEL_INPUT_KEYS = {
    "target_image",
    "spatial_condition",
    "clinical_text",
    "action_text",
    "delta_days",
    "stage_id",
    "source_ftv_volume_cc",
}
PAYLOAD_KEYS = {
    "schema_version",
    "cache_key",
    "bundle_contract_sha256",
    "bundle_preprocess_sha256",
    "effective_preprocess_sha256",
    "source_asset_sha256",
    "source_asset_stat_signatures",
    "source_asset_contract_sha256",
    "channel_order",
    "mri",
    "mask",
    "preprocess_metadata",
}
PREPROCESS_METADATA_KEYS = {
    "source_shape_zyx",
    "resampled_shape_zyx",
    "target_spacing_xyz",
    "output_shape_zyx",
    "normalization",
    "crop",
}
NORMALIZATION_METADATA_KEYS = {
    "low",
    "high",
    "low_percentile",
    "high_percentile",
    "nonzero_count",
}
CROP_METADATA_KEYS = {
    "coordinate_frame",
    "bbox_min_zyx",
    "bbox_max_zyx",
    "bbox_shape_zyx",
    "center_zyx",
    "crop_start_zyx",
    "mask_voxel_count_before_crop",
    "mask_voxel_count_after_crop",
    "mask_fully_retained",
    "whole_breast_inverse_mapping",
}

logger = get_logger(__name__)


class DatasetContractError(ValueError):
    """Raised when bundle-backed model inputs violate the dataset contract."""


class ISPY2CacheError(RuntimeError):
    """Raised when a visit cache cannot be safely loaded or created."""


def _positive_triplet(values: Sequence[float], field: str) -> tuple[float, float, float]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise DatasetContractError(f"{field} must contain three positive values") from exc
    if len(result) != 3 or not all(math.isfinite(value) and value > 0 for value in result):
        raise DatasetContractError(f"{field} must contain three positive values")
    return result


def _positive_shape(values: Sequence[int], field: str) -> tuple[int, int, int]:
    try:
        source = tuple(values)
    except TypeError:
        raise DatasetContractError(
            f"{field} must contain three positive integers"
        ) from None
    if len(source) != 3:
        raise DatasetContractError(f"{field} must contain three positive integers")
    result: list[int] = []
    for value in source:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise DatasetContractError(f"{field} must contain three positive integers")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            raise DatasetContractError(
                f"{field} must contain three positive integers"
            ) from None
        if not math.isfinite(numeric) or not numeric.is_integer() or numeric <= 0:
            raise DatasetContractError(f"{field} must contain three positive integers")
        result.append(int(numeric))
    return tuple(result)


def _canonical_bool(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    return None


def _require_positive_finite(value: Any, message: str) -> float:
    if isinstance(value, bool):
        raise DatasetContractError(message)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        raise DatasetContractError(message) from None
    if not math.isfinite(numeric) or numeric <= 0:
        raise DatasetContractError(message)
    return numeric


def _metadata_int_vector(value: Any, field: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(type(item) is not int for item in value)
    ):
        raise DatasetContractError(f"cache preprocess {field} must contain three integers")
    return tuple(value)


def _validate_preprocess_metadata(
    metadata: Any,
    *,
    expected_shape: tuple[int, int, int],
    expected_spacing: tuple[float, float, float],
    mask_voxel_count: int,
) -> None:
    if not isinstance(metadata, dict) or set(metadata) != PREPROCESS_METADATA_KEYS:
        raise DatasetContractError("cache preprocess metadata fields do not match the schema")
    source_shape = _metadata_int_vector(metadata["source_shape_zyx"], "source shape")
    resampled_shape = _metadata_int_vector(
        metadata["resampled_shape_zyx"], "resampled shape"
    )
    output_shape = _metadata_int_vector(metadata["output_shape_zyx"], "output shape")
    if any(value <= 0 for value in (*source_shape, *resampled_shape, *output_shape)):
        raise DatasetContractError("cache preprocess shapes must be positive")
    if output_shape != expected_shape:
        raise DatasetContractError("cache preprocess output shape mismatch")
    spacing = metadata["target_spacing_xyz"]
    if (
        not isinstance(spacing, list)
        or len(spacing) != 3
        or any(type(value) is not float for value in spacing)
        or tuple(spacing) != expected_spacing
    ):
        raise DatasetContractError("cache preprocess target spacing mismatch")

    normalization = metadata["normalization"]
    if not isinstance(normalization, dict) or set(normalization) != set(CHANNEL_ORDER):
        raise DatasetContractError("cache normalization channels do not match channel order")
    resampled_voxel_count = math.prod(resampled_shape)
    for channel in CHANNEL_ORDER:
        statistics = normalization[channel]
        if not isinstance(statistics, dict) or set(statistics) != NORMALIZATION_METADATA_KEYS:
            raise DatasetContractError("cache normalization statistics fields are invalid")
        numeric_fields = ("low", "high", "low_percentile", "high_percentile")
        if any(type(statistics[field]) is not float for field in numeric_fields):
            raise DatasetContractError("cache normalization statistics types are invalid")
        values = tuple(float(statistics[field]) for field in numeric_fields)
        if not all(math.isfinite(value) for value in values):
            raise DatasetContractError("cache normalization statistics must be finite")
        low, high, low_percentile, high_percentile = values
        if high - low <= 1e-6:
            raise DatasetContractError("cache normalization intensity range is invalid")
        if (low_percentile, high_percentile) != tuple(NORMALIZATION_PERCENTILES):
            raise DatasetContractError("cache normalization percentiles mismatch")
        nonzero_count = statistics["nonzero_count"]
        if (
            type(nonzero_count) is not int
            or nonzero_count <= 0
            or nonzero_count > resampled_voxel_count
        ):
            raise DatasetContractError("cache normalization nonzero count is invalid")

    crop = metadata["crop"]
    if not isinstance(crop, dict) or set(crop) != CROP_METADATA_KEYS:
        raise DatasetContractError("cache crop metadata fields do not match the schema")
    if crop["coordinate_frame"] != "tumor_centered_per_visit":
        raise DatasetContractError("cache crop coordinate frame mismatch")
    minimum = _metadata_int_vector(crop["bbox_min_zyx"], "crop bbox minimum")
    maximum = _metadata_int_vector(crop["bbox_max_zyx"], "crop bbox maximum")
    bbox_shape = _metadata_int_vector(crop["bbox_shape_zyx"], "crop bbox shape")
    center = _metadata_int_vector(crop["center_zyx"], "crop center")
    starts = _metadata_int_vector(crop["crop_start_zyx"], "crop start")
    for axis in range(3):
        if minimum[axis] < 0 or maximum[axis] < minimum[axis]:
            raise DatasetContractError("cache crop bbox interval is invalid")
        if maximum[axis] >= resampled_shape[axis]:
            raise DatasetContractError("cache crop bbox exceeds the resampled shape")
        if bbox_shape[axis] != maximum[axis] - minimum[axis] + 1:
            raise DatasetContractError("cache crop bbox shape is inconsistent")
        if bbox_shape[axis] > output_shape[axis]:
            raise DatasetContractError("cache crop bbox exceeds the output shape")
        if center[axis] != (minimum[axis] + maximum[axis]) // 2:
            raise DatasetContractError("cache crop center is inconsistent")
        lower_start = min(0, resampled_shape[axis] - output_shape[axis])
        upper_start = max(0, resampled_shape[axis] - output_shape[axis])
        if not lower_start <= starts[axis] <= upper_start:
            raise DatasetContractError("cache crop start is outside the valid range")
        if starts[axis] > minimum[axis] or starts[axis] + output_shape[axis] <= maximum[axis]:
            raise DatasetContractError("cache crop does not contain the full bbox")

    before_count = crop["mask_voxel_count_before_crop"]
    after_count = crop["mask_voxel_count_after_crop"]
    if type(before_count) is not int or type(after_count) is not int:
        raise DatasetContractError("cache crop mask counts must be integers")
    if (
        before_count <= 0
        or before_count > resampled_voxel_count
        or after_count != before_count
        or after_count != mask_voxel_count
        or after_count > math.prod(output_shape)
    ):
        raise DatasetContractError("cache crop mask counts are inconsistent")
    if crop["mask_fully_retained"] is not True:
        raise DatasetContractError("cache crop must retain the full mask")
    if crop["whole_breast_inverse_mapping"] is not False:
        raise DatasetContractError("cache crop inverse mapping policy mismatch")


def _load_nifti_array(path: Any, label: str) -> np.ndarray:
    try:
        import nibabel as nib
        from nibabel.filebasedimages import ImageFileError
    except ImportError as exc:
        raise ImportError("I-SPY2 ROI preprocessing requires nibabel") from exc
    try:
        image = nib.load(str(path))
        if len(image.shape) != 3:
            raise ValueError
        array = np.asanyarray(image.dataobj)
    except (OSError, ValueError, ImageFileError):
        raise ValueError(f"{label} must be a readable 3D NIfTI") from None
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite 3D NIfTI")
    return array


def _preprocess_visit(
    visit: Mapping[str, Any],
    *,
    target_spacing_xyz: Sequence[float],
    output_shape_zyx: Sequence[int],
) -> dict[str, Any]:
    """Preprocess one visit without retaining any other visit volume in memory."""

    try:
        meta = json.loads(Path(str(visit["meta_path"])).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ValueError("meta must be readable JSON") from None
    if not isinstance(meta, dict):
        raise ValueError("meta must be a JSON object")
    meta_uid = str(meta.get("study_uid") or meta.get("study_instance_uid") or "").strip()
    if meta_uid != str(visit["study_instance_uid"]).strip():
        raise ValueError("meta study UID mismatch")
    geometry = geometry_from_meta(meta)

    dce0 = _load_nifti_array(visit["dce0_path"], "dce0")
    ser = _load_nifti_array(visit["ser_path"], "ser")
    mask = _load_nifti_array(visit["mask_path"], "mask")
    if dce0.shape != mask.shape or ser.shape != mask.shape:
        raise ValueError("MRI channels and mask must have identical native shapes")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("mask must be binary")
    if int(np.count_nonzero(mask)) != int(visit["ftv_voxel_count"]):
        raise ValueError("mask voxel count mismatch")

    resampled_mask, mask_geometry = resample_zyx_to_ras(
        mask,
        geometry,
        target_spacing_xyz=target_spacing_xyz,
        is_mask=True,
    )
    normalized_channels: list[np.ndarray] = []
    normalization: dict[str, Mapping[str, float | int]] = {}
    resampled_shape: tuple[int, ...] | None = None
    for name, array in zip(CHANNEL_ORDER, (dce0, ser)):
        resampled, image_geometry = resample_zyx_to_ras(
            array,
            geometry,
            target_spacing_xyz=target_spacing_xyz,
            is_mask=False,
        )
        if resampled.shape != resampled_mask.shape:
            raise ValueError("resampled MRI and mask shapes differ")
        if not np.allclose(image_geometry.affine_ras, mask_geometry.affine_ras, atol=1e-6):
            raise ValueError("resampled MRI and mask geometry differs")
        normalized, statistics = robust_nonzero_percentile_scale(resampled)
        normalized_channels.append(normalized)
        normalization[name] = statistics
        resampled_shape = tuple(int(value) for value in resampled.shape)

    cropped_mri, cropped_mask, crop = tumor_center_crop(
        np.stack(normalized_channels, axis=0),
        resampled_mask,
        output_shape_zyx=output_shape_zyx,
    )
    if not bool(crop["mask_fully_retained"]):
        raise ValueError("tumor mask is not fully retained by the requested crop")
    return {
        "mri": torch.from_numpy(cropped_mri.astype(np.float16, copy=False)).contiguous(),
        "mask": torch.from_numpy(
            cropped_mask[np.newaxis].astype(np.uint8, copy=False)
        ).contiguous(),
        "preprocess_metadata": {
            "source_shape_zyx": [int(value) for value in mask.shape],
            "resampled_shape_zyx": list(resampled_shape or ()),
            "target_spacing_xyz": [float(value) for value in target_spacing_xyz],
            "output_shape_zyx": [int(value) for value in output_shape_zyx],
            "normalization": normalization,
            "crop": crop,
        },
    }


@contextmanager
def _cache_lock(path: Path):
    descriptor: int | None = None
    handle = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        handle = os.fdopen(descriptor, "a+b", closefd=True)
        descriptor = None
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    except (OSError, ValueError):
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        elif descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise ISPY2CacheError("cache lock setup failed") from None
    try:
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
        except (OSError, ValueError):
            raise ISPY2CacheError("cache lock release failed") from None


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    descriptor: int | None = None
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        raise ISPY2CacheError("atomic cache publication failed") from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


class _VisitCache:
    def __init__(
        self,
        *,
        cache_dir: Path,
        visits: Sequence[Mapping[str, Any]],
        bundle_contract_sha256: str,
        bundle_preprocess_sha256: str,
        effective_preprocess_sha256: str,
        target_spacing_xyz: Sequence[float],
        output_shape_zyx: Sequence[int],
        read_only: bool,
    ) -> None:
        self.cache_dir = cache_dir
        self.bundle_contract_sha256 = bundle_contract_sha256
        self.bundle_preprocess_sha256 = bundle_preprocess_sha256
        self.effective_preprocess_sha256 = effective_preprocess_sha256
        self.target_spacing_xyz = tuple(float(value) for value in target_spacing_xyz)
        self.output_shape_zyx = tuple(int(value) for value in output_shape_zyx)
        self.read_only = bool(read_only)
        self._visits: dict[str, tuple[int, Mapping[str, Any]]] = {}
        for ordinal, visit in enumerate(visits, start=1):
            visit_id = str(visit["visit_id"])
            if visit_id in self._visits:
                raise DatasetContractError("published visits contain duplicate visit IDs")
            self._visits[visit_id] = (ordinal, visit)

    def _visit(self, visit_id: str) -> tuple[int, Mapping[str, Any]]:
        try:
            return self._visits[visit_id]
        except KeyError:
            raise DatasetContractError("transition references an unavailable visit") from None

    def _asset_hashes(self, visit: Mapping[str, Any], ordinal: int) -> dict[str, str]:
        try:
            return {
                name: sha256_file(visit[field])
                for name, field in SOURCE_ASSET_FIELDS.items()
            }
        except (OSError, ValueError):
            raise ISPY2CacheError(
                f"visit cache record {ordinal}: source asset hashing failed"
            ) from None

    def _asset_stats(
        self, visit: Mapping[str, Any], ordinal: int
    ) -> dict[str, dict[str, int]]:
        try:
            signatures: dict[str, dict[str, int]] = {}
            for name, field in SOURCE_ASSET_FIELDS.items():
                status = Path(str(visit[field])).stat()
                signatures[name] = {
                    "st_dev": int(status.st_dev),
                    "st_ino": int(status.st_ino),
                    "st_size": int(status.st_size),
                    "st_mtime_ns": int(status.st_mtime_ns),
                    "st_ctime_ns": int(status.st_ctime_ns),
                }
            return signatures
        except (OSError, ValueError):
            raise ISPY2CacheError(
                f"visit cache record {ordinal}: source asset stat failed"
            ) from None

    def _cache_key(self, visit: Mapping[str, Any]) -> str:
        identity_sha256 = sha256_json(
            {
                "visit_id": str(visit["visit_id"]),
                "study_instance_uid": str(visit["study_instance_uid"]),
            }
        )
        return sha256_json(
            {
                "schema_version": CACHE_SCHEMA,
                "visit_identity_sha256": identity_sha256,
                "channel_order": list(CHANNEL_ORDER),
                "effective_preprocess_sha256": self.effective_preprocess_sha256,
                "output_shape_zyx": list(self.output_shape_zyx),
                "target_spacing_xyz": list(self.target_spacing_xyz),
            }
        )

    def _static_expected(self, visit: Mapping[str, Any]) -> dict[str, str]:
        return {
            "schema_version": CACHE_SCHEMA,
            "cache_key": self._cache_key(visit),
            "effective_preprocess_sha256": self.effective_preprocess_sha256,
        }

    def _validate_payload(
        self,
        payload: Any,
        expected: Mapping[str, Any],
        expected_stats: Mapping[str, Mapping[str, int]],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS:
            raise DatasetContractError("cache payload fields do not match the schema")
        for field, value in expected.items():
            if payload.get(field) != value:
                raise DatasetContractError(f"cache payload {field} mismatch")
        for field in ("bundle_contract_sha256", "bundle_preprocess_sha256"):
            value = payload.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise DatasetContractError(f"cache payload {field} is invalid")
        hashes = payload.get("source_asset_sha256")
        if not isinstance(hashes, dict) or set(hashes) != set(SOURCE_ASSET_FIELDS):
            raise DatasetContractError("cache source asset SHA256 fields are invalid")
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes.values()
        ):
            raise DatasetContractError("cache source asset SHA256 values are invalid")
        signatures = payload.get("source_asset_stat_signatures")
        if not isinstance(signatures, dict) or set(signatures) != set(SOURCE_ASSET_FIELDS):
            raise DatasetContractError("cache source asset stat fields are invalid")
        for signature in signatures.values():
            if (
                not isinstance(signature, dict)
                or set(signature) != STAT_SIGNATURE_KEYS
                or any(type(value) is not int or value < 0 for value in signature.values())
                or signature["st_size"] <= 0
            ):
                raise DatasetContractError("cache source asset stat signature is invalid")
        if signatures != dict(expected_stats):
            raise DatasetContractError("cache source asset stat signature mismatch")
        asset_contract = sha256_json(
            {
                "source_asset_sha256": hashes,
                "source_asset_stat_signatures": signatures,
            }
        )
        if payload.get("source_asset_contract_sha256") != asset_contract:
            raise DatasetContractError("cache source asset contract SHA256 mismatch")
        if payload.get("channel_order") != list(CHANNEL_ORDER):
            raise DatasetContractError("cache payload channel order mismatch")
        mri = payload.get("mri")
        mask = payload.get("mask")
        expected_shape = self.output_shape_zyx
        if (
            not isinstance(mri, torch.Tensor)
            or mri.device.type != "cpu"
            or mri.dtype != torch.float16
            or tuple(mri.shape) != (2, *expected_shape)
        ):
            raise DatasetContractError("cache MRI tensor shape or dtype mismatch")
        if not bool(torch.isfinite(mri).all()) or bool(((mri < 0) | (mri > 1)).any()):
            raise DatasetContractError("cache MRI tensor must be finite in [0, 1]")
        if (
            not isinstance(mask, torch.Tensor)
            or mask.device.type != "cpu"
            or mask.dtype != torch.uint8
            or tuple(mask.shape) != (1, *expected_shape)
        ):
            raise DatasetContractError("cache mask tensor shape or dtype mismatch")
        if not bool(((mask == 0) | (mask == 1)).all()) or int(mask.sum()) <= 0:
            raise DatasetContractError("cache mask tensor must be nonempty and binary")
        _validate_preprocess_metadata(
            payload.get("preprocess_metadata"),
            expected_shape=expected_shape,
            expected_spacing=self.target_spacing_xyz,
            mask_voxel_count=int(mask.sum()),
        )
        return payload

    def _try_load(
        self,
        path: Path,
        expected: Mapping[str, Any],
        expected_stats: Mapping[str, Mapping[str, int]],
    ) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            return self._validate_payload(payload, expected, expected_stats)
        except Exception:
            return None

    def _try_load_with_stable_assets(
        self,
        path: Path,
        expected: Mapping[str, Any],
        visit: Mapping[str, Any],
        ordinal: int,
    ) -> tuple[dict[str, Any] | None, dict[str, dict[str, int]], bool]:
        before = self._asset_stats(visit, ordinal)
        payload = self._try_load(path, expected, before)
        after = self._asset_stats(visit, ordinal)
        stable = before == after
        return (payload if stable else None), after, stable

    def load(self, visit_id: str) -> tuple[dict[str, Any], str]:
        ordinal, visit = self._visit(visit_id)
        expected = self._static_expected(visit)
        cache_path = self.cache_dir / f"{expected['cache_key']}.pt"
        payload, _, _ = self._try_load_with_stable_assets(
            cache_path, expected, visit, ordinal
        )
        if payload is not None:
            return payload, "hit"
        if self.read_only:
            raise ISPY2CacheError(
                f"visit cache record {ordinal}: cache is missing, stale, or invalid "
                "in read-only mode"
            ) from None

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise ISPY2CacheError(
                f"visit cache record {ordinal}: cache directory creation failed"
            ) from None
        lock_path = self.cache_dir / f".{expected['cache_key']}.lock"
        with _cache_lock(lock_path):
            for _ in range(MAX_STABLE_CACHE_READ_ATTEMPTS):
                payload, stats, stable = self._try_load_with_stable_assets(
                    cache_path, expected, visit, ordinal
                )
                if stable:
                    break
            else:
                raise ISPY2CacheError(
                    f"visit cache record {ordinal}: source assets changed repeatedly "
                    "during cache validation"
                ) from None
            if payload is not None:
                return payload, "hit"
            assets = self._asset_hashes(visit, ordinal)
            if self._asset_stats(visit, ordinal) != stats:
                raise ISPY2CacheError(
                    f"visit cache record {ordinal}: source assets changed during hashing"
                ) from None
            status = "rebuilt" if cache_path.exists() else "built"
            try:
                processed = _preprocess_visit(
                    visit,
                    target_spacing_xyz=self.target_spacing_xyz,
                    output_shape_zyx=self.output_shape_zyx,
                )
            except Exception:
                raise ISPY2CacheError(
                    f"visit cache record {ordinal}: preprocessing failed"
                ) from None
            if self._asset_stats(visit, ordinal) != stats:
                raise ISPY2CacheError(
                    f"visit cache record {ordinal}: source assets changed during preprocessing"
                ) from None
            asset_contract = sha256_json(
                {
                    "source_asset_sha256": assets,
                    "source_asset_stat_signatures": stats,
                }
            )
            payload = {
                **expected,
                "bundle_contract_sha256": self.bundle_contract_sha256,
                "bundle_preprocess_sha256": self.bundle_preprocess_sha256,
                "source_asset_sha256": assets,
                "source_asset_stat_signatures": stats,
                "source_asset_contract_sha256": asset_contract,
                "channel_order": list(CHANNEL_ORDER),
                **processed,
            }
            self._validate_payload(payload, expected, stats)
            _atomic_torch_save(payload, cache_path)
            return payload, status


class ISPY2LongitudinalDataset(Dataset):
    """Select verified transitions by fold and load only requested visit ROIs."""

    def __init__(
        self,
        bundle_json: str | Path,
        *,
        cache_dir: str | Path,
        fold: str | Sequence[str],
        output_shape_zyx: Sequence[int] | None = None,
        target_spacing_xyz: Sequence[float] | None = None,
        read_only: bool = False,
        include_evaluation_metadata: bool = False,
        source_records: Sequence[Mapping[str, Any]] | None = None,
        verified_bundle: Mapping[str, Any] | None = None,
    ) -> None:
        bundle_path = Path(bundle_json).resolve()
        self._source_only = source_records is not None
        if self._source_only:
            if not isinstance(verified_bundle, Mapping):
                raise DatasetContractError(
                    "source-only dataset requires a preflight-verified bundle"
                )
            try:
                published_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
                bundle = deepcopy(dict(verified_bundle))
                if published_bundle != bundle:
                    raise ValueError
                expected_self = sha256_json(
                    {
                        key: value
                        for key, value in bundle.items()
                        if key != "bundle_contract_sha256"
                    }
                )
                if bundle.get("bundle_contract_sha256") != expected_self:
                    raise ValueError
                for artifact_name in ("visits", "preprocess"):
                    artifact = bundle["artifacts"][artifact_name]
                    filename = ARTIFACT_FILENAMES[artifact_name]
                    if (
                        artifact.get("path") != filename
                        or artifact.get("sha256")
                        != sha256_file(bundle_path.parent / filename)
                    ):
                        raise ValueError
            except (KeyError, OSError, UnicodeError, json.JSONDecodeError, ValueError):
                raise DatasetContractError(
                    "published source-only bundle verification failed"
                ) from None
        else:
            if verified_bundle is not None:
                raise DatasetContractError(
                    "verified_bundle is only valid with source_records"
                )
            try:
                bundle = verify_bundle(bundle_path)
            except (ContractError, OSError, ValueError):
                raise DatasetContractError("published bundle verification failed") from None
        try:
            published_preprocess = json.loads(
                (bundle_path.parent / ARTIFACT_FILENAMES["preprocess"]).read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError):
            raise DatasetContractError("published preprocess contract is unreadable") from None
        if published_preprocess != preprocess_contract():
            raise DatasetContractError("published preprocess contract is unsupported")

        published_shape = published_preprocess["output_shape_zyx"]
        published_spacing = published_preprocess["target_spacing_xyz"]
        self.output_shape_zyx = _positive_shape(
            output_shape_zyx if output_shape_zyx is not None else published_shape,
            "output_shape_zyx",
        )
        self.target_spacing_xyz = _positive_triplet(
            target_spacing_xyz if target_spacing_xyz is not None else published_spacing,
            "target_spacing_xyz",
        )
        effective_preprocess = preprocess_contract(
            output_shape_zyx=self.output_shape_zyx,
            target_spacing_xyz=self.target_spacing_xyz,
        )

        requested_folds = (fold,) if isinstance(fold, str) else tuple(fold)
        if not requested_folds or any(value not in VALID_FOLDS for value in requested_folds):
            raise DatasetContractError("fold must contain only train, val, or test")
        if len(set(requested_folds)) != len(requested_folds):
            raise DatasetContractError("fold selection contains duplicates")
        try:
            visits = pd.read_csv(bundle_path.parent / ARTIFACT_FILENAMES["visits"])
            if self._source_only:
                if bundle["artifacts"]["visits"].get("row_count") != len(visits):
                    raise ValueError
                self._transitions = [
                    deepcopy(dict(record)) for record in source_records or ()
                ]
            else:
                transitions = pd.read_csv(
                    bundle_path.parent / ARTIFACT_FILENAMES["transitions"]
                )
                selected = transitions[transitions["fold"].isin(requested_folds)].copy()
                self._transitions = selected.to_dict(orient="records")
        except (KeyError, OSError, pd.errors.ParserError, TypeError, ValueError):
            raise DatasetContractError("published bundle tables are unreadable") from None
        self.include_evaluation_metadata = bool(include_evaluation_metadata)
        if self._source_only:
            if self.include_evaluation_metadata:
                raise DatasetContractError(
                    "source-only dataset cannot include evaluation metadata"
                )
            self._validate_source_records(requested_folds)
            visit_ids = set(visits["visit_id"].astype(str))
            if any(
                str(record["source_visit_id"]) not in visit_ids
                for record in self._transitions
            ):
                raise DatasetContractError(
                    "source-only record does not match a published visit"
                )
        else:
            self._validate_transitions()
        self._cache = _VisitCache(
            cache_dir=Path(cache_dir).resolve(),
            visits=visits.to_dict(orient="records"),
            bundle_contract_sha256=str(bundle["bundle_contract_sha256"]),
            bundle_preprocess_sha256=str(bundle["artifacts"]["preprocess"]["sha256"]),
            effective_preprocess_sha256=sha256_json(effective_preprocess),
            target_spacing_xyz=self.target_spacing_xyz,
            output_shape_zyx=self.output_shape_zyx,
            read_only=read_only,
        )

    def _validate_source_records(self, requested_folds: Sequence[str]) -> None:
        required_strings = (
            "transition_id",
            "source_visit_id",
            "fold",
            "clinical_text",
            "action_text",
        )
        transition_ids: set[str] = set()
        source_visit_ids: set[str] = set()
        for row in self._transitions:
            if not isinstance(row, Mapping) or any(
                "target" in str(key).lower() for key in row
            ):
                raise DatasetContractError(
                    "source-only records must not contain target fields"
                )
            try:
                delta_days = int(row["delta_days"])
                source_ftv = float(row["source_ftv_volume_cc"])
            except (KeyError, TypeError, ValueError, OverflowError):
                raise DatasetContractError(
                    "source-only numeric conditions are invalid"
                ) from None
            if delta_days <= 0 or float(row["delta_days"]) != float(delta_days):
                raise DatasetContractError(
                    "source-only delta_days must be a positive integer"
                )
            if not math.isfinite(source_ftv) or source_ftv <= 0:
                raise DatasetContractError(
                    "source-only FTV must be positive and finite"
                )
            if str(row.get("transition_type", "")) not in STAGE_BY_TRANSITION:
                raise DatasetContractError("source-only transition stage is unsupported")
            if str(row.get("fold", "")) not in requested_folds:
                raise DatasetContractError("source-only record fold is not selected")
            for field in required_strings:
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise DatasetContractError(
                        "source-only record string fields must be nonempty"
                    )
            transition_id = str(row["transition_id"])
            source_visit_id = str(row["source_visit_id"])
            if transition_id in transition_ids or source_visit_id in source_visit_ids:
                raise DatasetContractError(
                    "source-only transition and source visit identities must be unique"
                )
            transition_ids.add(transition_id)
            source_visit_ids.add(source_visit_id)

    def _validate_transitions(self) -> None:
        for row in self._transitions:
            try:
                delta_days = int(row["delta_days"])
                source_ftv = float(row["source_ftv_volume_cc"])
            except (KeyError, TypeError, ValueError, OverflowError):
                raise DatasetContractError("transition numeric conditions are invalid") from None
            if delta_days <= 0 or float(row["delta_days"]) != float(delta_days):
                raise DatasetContractError("transition delta_days must be a positive integer")
            if not math.isfinite(source_ftv) or source_ftv <= 0:
                raise DatasetContractError("transition source FTV must be positive and finite")
            transition_type = str(row.get("transition_type", ""))
            if transition_type not in STAGE_BY_TRANSITION:
                raise DatasetContractError("transition stage is unsupported")
            if _canonical_bool(row.get("source_ftv_is_condition")) is not True:
                raise DatasetContractError("transition source FTV condition flag is invalid")
            if _canonical_bool(row.get("target_ftv_is_condition")) is not False:
                raise DatasetContractError("transition target FTV condition flag is invalid")
            if _canonical_bool(row.get("target_ftv_is_audit_only")) is not True:
                raise DatasetContractError("transition target FTV audit flag is invalid")
            for field in (
                "transition_id",
                "source_visit_id",
                "target_visit_id",
                "fold",
                "clinical_text",
                "action_text",
            ):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise DatasetContractError("transition string fields must be nonempty")

    def __len__(self) -> int:
        return len(self._transitions)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._source_only:
            raise DatasetContractError(
                "source-only dataset does not support indexed target access"
            )
        row = self._transitions[index]
        source, _ = self._cache.load(str(row["source_visit_id"]))
        target, _ = self._cache.load(str(row["target_visit_id"]))
        item: dict[str, Any] = {
            "model_inputs": {
                "target_image": target["mri"][0:1].to(dtype=torch.float32),
                "spatial_condition": source["mri"].to(dtype=torch.float32),
                "clinical_text": str(row["clinical_text"]),
                "action_text": str(row["action_text"]),
                "delta_days": int(row["delta_days"]),
                "stage_id": STAGE_BY_TRANSITION[str(row["transition_type"])],
                "source_ftv_volume_cc": float(row["source_ftv_volume_cc"]),
            },
            "metadata": {
                "transition_id": str(row["transition_id"]),
                "source_visit_id": str(row["source_visit_id"]),
                "target_visit_id": str(row["target_visit_id"]),
                "fold": str(row["fold"]),
            },
        }
        if self.include_evaluation_metadata:
            item["evaluation_metadata"] = {
                "target_mask": target["mask"].clone(),
                "target_ftv_volume_cc": float(row["target_ftv_volume_cc"]),
            }
        _validate_item(item)
        return item

    def source_condition(
        self,
        source_visit_id: str,
        *,
        source_ftv_volume_cc: float,
    ) -> dict[str, Any]:
        """Load one locked transition's source inputs without touching its target visit."""

        matches = [
            row
            for row in self._transitions
            if str(row["source_visit_id"]) == source_visit_id
        ]
        if len(matches) != 1:
            raise DatasetContractError(
                "source visit must identify exactly one locked transition"
            )
        source_ftv = _require_positive_finite(
            source_ftv_volume_cc,
            "source FTV volume must be positive and finite",
        )
        row = matches[0]
        source, _ = self._cache.load(source_visit_id)
        spatial = source["mri"].to(dtype=torch.float32)
        mask = source["mask"].clone()
        if (
            tuple(spatial.shape) != (2, *self.output_shape_zyx)
            or not bool(torch.isfinite(spatial).all())
            or bool(((spatial < 0) | (spatial > 1)).any())
        ):
            raise DatasetContractError("source spatial condition is invalid")
        if (
            tuple(mask.shape) != (1, *self.output_shape_zyx)
            or mask.dtype != torch.uint8
            or not bool(((mask == 0) | (mask == 1)).all())
            or int(mask.sum()) <= 0
        ):
            raise DatasetContractError("source mask is invalid")
        return {
            "model_inputs": {
                "spatial_condition": spatial,
                "clinical_text": str(row["clinical_text"]),
                "action_text": str(row["action_text"]),
                "delta_days": int(row["delta_days"]),
                "stage_id": STAGE_BY_TRANSITION[str(row["transition_type"])],
                "source_ftv_volume_cc": source_ftv,
            },
            "source_geometry": {
                "mask": mask,
                "preprocess_metadata": deepcopy(source["preprocess_metadata"]),
            },
            "metadata": {
                "transition_id": str(row["transition_id"]),
                "source_visit_id": source_visit_id,
                "fold": str(row["fold"]),
                "transition_type": str(row["transition_type"]),
            },
        }

    def build_cache(self, *, progress_every: int = 100) -> dict[str, int]:
        if self._source_only:
            raise DatasetContractError(
                "source-only dataset does not support full cache traversal"
            )
        if type(progress_every) is not int or progress_every <= 0:
            raise DatasetContractError("progress_every must be a positive integer")
        visit_ids: list[str] = []
        seen: set[str] = set()
        for row in self._transitions:
            for field in ("source_visit_id", "target_visit_id"):
                visit_id = str(row[field])
                if visit_id not in seen:
                    seen.add(visit_id)
                    visit_ids.append(visit_id)
        counts = {"total": len(visit_ids), "cache_hits": 0, "built": 0, "rebuilt": 0}
        for completed, visit_id in enumerate(visit_ids, start=1):
            _, status = self._cache.load(visit_id)
            key = "cache_hits" if status == "hit" else status
            counts[key] += 1
            if completed % progress_every == 0 or completed == len(visit_ids):
                logger.info(
                    "I-SPY2 ROI cache: completed %d/%d visits",
                    completed,
                    len(visit_ids),
                )
        logger.info(
            "I-SPY2 ROI cache summary: total=%d cache_hits=%d built=%d rebuilt=%d",
            counts["total"],
            counts["cache_hits"],
            counts["built"],
            counts["rebuilt"],
        )
        return counts


def _validate_item(item: Mapping[str, Any]) -> None:
    expected_top = {"model_inputs", "metadata"}
    if "evaluation_metadata" in item:
        expected_top.add("evaluation_metadata")
    if set(item) != expected_top:
        raise DatasetContractError("dataset item boundary is invalid")
    inputs = item.get("model_inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != MODEL_INPUT_KEYS:
        raise DatasetContractError("model input fields are invalid")
    target = inputs["target_image"]
    spatial = inputs["spatial_condition"]
    if (
        not isinstance(target, torch.Tensor)
        or target.dtype != torch.float32
        or target.ndim != 4
        or target.shape[0] != 1
    ):
        raise DatasetContractError("target_image must be float32 [1,D,H,W]")
    if (
        not isinstance(spatial, torch.Tensor)
        or spatial.dtype != torch.float32
        or spatial.ndim != 4
        or spatial.shape[0] != 2
        or tuple(spatial.shape[1:]) != tuple(target.shape[1:])
    ):
        raise DatasetContractError("spatial_condition must be float32 [2,D,H,W]")
    for tensor in (target, spatial):
        if not bool(torch.isfinite(tensor).all()):
            raise DatasetContractError("model image tensors must be finite")
        if bool(((tensor < 0) | (tensor > 1)).any()):
            raise DatasetContractError("model image tensors must be in [0, 1]")
    if not isinstance(inputs["clinical_text"], str) or not isinstance(
        inputs["action_text"], str
    ):
        raise DatasetContractError("model text conditions must be strings")
    if type(inputs["delta_days"]) is not int or inputs["delta_days"] <= 0:
        raise DatasetContractError("delta_days must be a positive integer")
    if type(inputs["stage_id"]) is not int or inputs["stage_id"] not in (1, 2, 3):
        raise DatasetContractError("stage_id must be 1, 2, or 3")
    _require_positive_finite(
        inputs["source_ftv_volume_cc"],
        "source FTV volume must be positive and finite",
    )
    metadata = item.get("metadata")
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "transition_id",
        "source_visit_id",
        "target_visit_id",
        "fold",
    }:
        raise DatasetContractError("dataset metadata fields are invalid")
    if "evaluation_metadata" in item:
        evaluation = item["evaluation_metadata"]
        if not isinstance(evaluation, Mapping) or set(evaluation) != {
            "target_mask",
            "target_ftv_volume_cc",
        }:
            raise DatasetContractError("evaluation metadata fields are invalid")
        mask = evaluation["target_mask"]
        if (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.uint8
            or tuple(mask.shape) != tuple(target.shape)
            or not bool(((mask == 0) | (mask == 1)).all())
        ):
            raise DatasetContractError("evaluation target mask is invalid")
        _require_positive_finite(
            evaluation["target_ftv_volume_cc"],
            "evaluation target FTV must be positive and finite",
        )


def collate_ispy2_longitudinal(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate validated dataset items without flattening evaluation-only metadata."""

    if not batch:
        raise DatasetContractError("cannot collate an empty batch")
    for item in batch:
        _validate_item(item)
    has_evaluation = "evaluation_metadata" in batch[0]
    if any(("evaluation_metadata" in item) != has_evaluation for item in batch):
        raise DatasetContractError("batch mixes training and evaluation item boundaries")
    spatial_shape = tuple(batch[0]["model_inputs"]["target_image"].shape)
    if any(tuple(item["model_inputs"]["target_image"].shape) != spatial_shape for item in batch):
        raise DatasetContractError("batch target image shapes differ")
    try:
        delta_days = torch.tensor(
            [float(item["model_inputs"]["delta_days"]) for item in batch],
            dtype=torch.float32,
        )
        stage_id = torch.tensor(
            [item["model_inputs"]["stage_id"] for item in batch], dtype=torch.long
        )
        source_ftv = torch.tensor(
            [float(item["model_inputs"]["source_ftv_volume_cc"]) for item in batch],
            dtype=torch.float32,
        )
    except (TypeError, ValueError, OverflowError, RuntimeError):
        raise DatasetContractError("collated numeric tensor construction failed") from None
    result: dict[str, Any] = {
        "model_inputs": {
            "target_image": torch.stack(
                [item["model_inputs"]["target_image"] for item in batch]
            ),
            "spatial_condition": torch.stack(
                [item["model_inputs"]["spatial_condition"] for item in batch]
            ),
            "clinical_text": [item["model_inputs"]["clinical_text"] for item in batch],
            "action_text": [item["model_inputs"]["action_text"] for item in batch],
            "delta_days": delta_days,
            "stage_id": stage_id,
            "source_ftv_volume_cc": source_ftv,
        },
        "metadata": [dict(item["metadata"]) for item in batch],
    }
    if has_evaluation:
        try:
            target_ftv = torch.tensor(
                [
                    float(item["evaluation_metadata"]["target_ftv_volume_cc"])
                    for item in batch
                ],
                dtype=torch.float32,
            )
        except (TypeError, ValueError, OverflowError, RuntimeError):
            raise DatasetContractError("collated numeric tensor construction failed") from None
        result["evaluation_metadata"] = {
            "target_mask": torch.stack(
                [item["evaluation_metadata"]["target_mask"] for item in batch]
            ),
            "target_ftv_volume_cc": target_ftv,
        }
    numeric_inputs = result["model_inputs"]
    if not bool(torch.isfinite(numeric_inputs["delta_days"]).all()) or bool(
        (numeric_inputs["delta_days"] <= 0).any()
    ):
        raise DatasetContractError("collated delta_days tensor is invalid")
    if bool(
        ((numeric_inputs["stage_id"] < 1) | (numeric_inputs["stage_id"] > 3)).any()
    ):
        raise DatasetContractError("collated stage_id tensor is invalid")
    if not bool(torch.isfinite(numeric_inputs["source_ftv_volume_cc"]).all()) or bool(
        (numeric_inputs["source_ftv_volume_cc"] <= 0).any()
    ):
        raise DatasetContractError("collated source FTV tensor is invalid")
    if has_evaluation:
        target_ftv = result["evaluation_metadata"]["target_ftv_volume_cc"]
        if not bool(torch.isfinite(target_ftv).all()) or bool((target_ftv <= 0).any()):
            raise DatasetContractError("collated target FTV tensor is invalid")
    return result

"""Contracts shared by registered Strict-A bundle and cache tooling."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


BUNDLE_SCHEMA = "motfm_ispy2_registered_strict_a_bundle_v1"
CACHE_SCHEMA = "motfm_ispy2_t0_fixed_roi_cache_v1"
STATISTICS_SCHEMA = "motfm_ispy2_train_foreground_stats_v1"
COORDINATE_FRAME = "t0_fixed_registered_local_v1"
TARGET_SPACING_XYZ = (0.7032, 0.7032, 2.0)
OUTPUT_SHAPE_ZYX = (96, 256, 256)
VALID_FOLDS = ("train", "val")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def visit_fails_strict_a(visit: Mapping[str, Any]) -> bool:
    """Apply the frozen Strict-A v1 non-T0 quality rules."""

    if bool(visit.get("is_t0")):
        return False
    if bool(visit.get("algorithm_anomaly")):
        return True
    fov = float(visit["in_plane_fov_ratio"])
    non_tumor = float(visit["selected_non_tumor_ncc"])
    protected = float(visit["selected_protected_ncc"])
    rotation = float(visit["rotation_degrees"])
    translation = float(visit["translation_mm"])
    values = (fov, non_tumor, protected, rotation, translation)
    if not all(math.isfinite(value) for value in values):
        return True
    if fov >= 1.5:
        return True
    if non_tumor < 0.333385858 and protected < 0.056044488:
        return True
    low_selected = non_tumor < 0.5 or protected < 0.0
    extreme_motion = rotation >= 12.383351912 or translation >= 49.690795908
    return bool(
        (extreme_motion and low_selected)
        or (bool(visit.get("rigid_maximum_iterations_reached")) and low_selected)
    )


def selected_registration_metrics(
    registration_status: str, qc: Mapping[str, Any]
) -> tuple[float, float]:
    if registration_status == "deformable":
        prefix = "deformable"
    elif registration_status == "rigid_fallback":
        prefix = "rigid"
    else:
        raise ValueError("selected registration metrics require a non-T0 terminal state")
    try:
        values = (
            float(qc[f"{prefix}_non_tumor_ncc"]),
            float(qc[f"{prefix}_protected_ncc"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        raise ValueError("selected registration metrics are unavailable") from None
    if not all(math.isfinite(item) for item in values):
        raise ValueError("selected registration metrics must be finite")
    return values


def reconstruct_transition_universe(
    visits: pd.DataFrame,
    accepted: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Restore accepted and formerly excluded adjacent transitions from the frozen bundle."""

    required_visits = {
        "patient_id",
        "visit_id",
        "visit",
        "study_instance_uid",
        "visit_date",
        "ftv_volume_cc",
        "clinical_text",
        "trial_arm",
    }
    if not required_visits.issubset(visits.columns):
        raise ValueError("frozen visits are missing transition reconstruction fields")
    by_visit = {
        str(row["visit_id"]): row for row in visits.to_dict(orient="records")
    }
    if len(by_visit) != len(visits):
        raise ValueError("frozen visit identities must be unique")
    output: dict[str, dict[str, Any]] = {}
    for row in accepted.to_dict(orient="records"):
        transition_id = str(row.get("transition_id", ""))
        if not transition_id:
            raise ValueError("accepted transition identity is invalid")
        output[transition_id] = dict(row)
    for row in exclusions.to_dict(orient="records"):
        source_id = str(row.get("source_visit_id", ""))
        target_id = str(row.get("target_visit_id", ""))
        if source_id not in by_visit or target_id not in by_visit:
            raise ValueError("excluded transition references an unavailable visit")
        source, target = by_visit[source_id], by_visit[target_id]
        transition_type = str(row["transition_type"])
        transition_id = f"{row['patient_id']}:{transition_type}"
        if transition_id in output:
            raise ValueError("accepted and excluded transition identities overlap")
        try:
            detail = json.loads(str(row.get("detail", "{}")))
            delta_days = int(detail.get("delta_days"))
        except (json.JSONDecodeError, TypeError, ValueError, OverflowError):
            source_date = np.datetime64(str(source["visit_date"]))
            target_date = np.datetime64(str(target["visit_date"]))
            delta_days = int((target_date - source_date).astype(int))
        output[transition_id] = {
            "transition_id": transition_id,
            "patient_id": str(row["patient_id"]),
            "transition_type": transition_type,
            "source_visit_id": source_id,
            "target_visit_id": target_id,
            "source_visit": str(source["visit"]),
            "target_visit": str(target["visit"]),
            "source_study_instance_uid": str(source["study_instance_uid"]),
            "target_study_instance_uid": str(target["study_instance_uid"]),
            "delta_days": delta_days,
            "source_ftv_volume_cc": float(source["ftv_volume_cc"]),
            "target_ftv_volume_cc": float(target["ftv_volume_cc"]),
            "source_ftv_is_condition": True,
            "target_ftv_is_condition": False,
            "target_ftv_is_audit_only": True,
            "clinical_text": str(source["clinical_text"]),
            "action_text": f"treatment arm {source['trial_arm']}",
        }
    rows = sorted(output.values(), key=lambda row: str(row["transition_id"]))
    if len(rows) != len(accepted) + len(exclusions):
        raise RuntimeError("frozen transition universe is not an exact union")
    return rows


@dataclass(frozen=True)
class CropPlan:
    coordinate_frame: str
    input_shape_zyx: tuple[int, int, int]
    output_shape_zyx: tuple[int, int, int]
    bbox_min_zyx: tuple[int, int, int]
    bbox_max_zyx: tuple[int, int, int]
    center_zyx: tuple[int, int, int]
    crop_start_zyx: tuple[int, int, int]
    t0_mask_voxel_count: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in (
            "input_shape_zyx",
            "output_shape_zyx",
            "bbox_min_zyx",
            "bbox_max_zyx",
            "center_zyx",
            "crop_start_zyx",
        ):
            value[key] = list(value[key])
        return value


def _shape3(value: Sequence[int], *, name: str) -> tuple[int, int, int]:
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must contain three positive integers") from None
    if len(result) != 3 or any(item <= 0 for item in result):
        raise ValueError(f"{name} must contain three positive integers")
    return result


def t0_fixed_crop_plan(
    t0_mask_zyx: np.ndarray,
    *,
    output_shape_zyx: Sequence[int] = OUTPUT_SHAPE_ZYX,
) -> CropPlan:
    mask = np.asarray(t0_mask_zyx)
    output_shape = _shape3(output_shape_zyx, name="output_shape_zyx")
    if mask.ndim != 3 or not np.isfinite(mask).all():
        raise ValueError("T0 Mask must be a finite 3D array")
    binary = mask > 0
    positive = np.argwhere(binary)
    if positive.size == 0:
        raise ValueError("T0 Mask must be nonempty")
    minimum = positive.min(axis=0)
    maximum = positive.max(axis=0)
    center = (minimum + maximum) // 2
    starts = tuple(int(center[i]) - output_shape[i] // 2 for i in range(3))
    return CropPlan(
        coordinate_frame=COORDINATE_FRAME,
        input_shape_zyx=tuple(int(item) for item in mask.shape),
        output_shape_zyx=output_shape,
        bbox_min_zyx=tuple(int(item) for item in minimum),
        bbox_max_zyx=tuple(int(item) for item in maximum),
        center_zyx=tuple(int(item) for item in center),
        crop_start_zyx=starts,
        t0_mask_voxel_count=int(binary.sum()),
    )


def crop_or_pad_zyx(
    array: np.ndarray,
    starts_zyx: Sequence[int],
    output_shape_zyx: Sequence[int],
) -> np.ndarray:
    source = np.asarray(array)
    starts = tuple(int(item) for item in starts_zyx)
    output_shape = _shape3(output_shape_zyx, name="output_shape_zyx")
    if len(starts) != 3 or source.ndim < 3:
        raise ValueError("crop requires an array with three spatial dimensions")
    output = np.zeros((*source.shape[:-3], *output_shape), dtype=source.dtype)
    source_slices: list[slice] = []
    output_slices: list[slice] = []
    for start, requested, available in zip(starts, output_shape, source.shape[-3:]):
        source_start = max(start, 0)
        source_stop = min(start + requested, int(available))
        target_start = source_start - start
        target_stop = target_start + max(source_stop - source_start, 0)
        source_slices.append(slice(source_start, source_stop))
        output_slices.append(slice(target_start, target_stop))
    output[(Ellipsis, *output_slices)] = source[(Ellipsis, *source_slices)]
    return output


@dataclass(frozen=True)
class ChannelStatistics:
    count: int
    mean: float
    std: float

    def to_dict(self) -> dict[str, int | float]:
        return {"count": self.count, "mean": self.mean, "std": self.std}


class _Moments:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values: np.ndarray) -> None:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("normalization values must be nonempty and finite")
        count = int(vector.size)
        mean = float(vector.mean(dtype=np.float64))
        m2 = float(np.square(vector - mean, dtype=np.float64).sum(dtype=np.float64))
        if self.count == 0:
            self.count, self.mean, self.m2 = count, mean, m2
            return
        delta = mean - self.mean
        total = self.count + count
        self.m2 += m2 + delta * delta * self.count * count / total
        self.mean += delta * count / total
        self.count = total

    def finish(self) -> ChannelStatistics:
        if self.count <= 0:
            raise ValueError("normalization statistics are empty")
        std = math.sqrt(self.m2 / self.count)
        if not math.isfinite(std) or std <= 0:
            raise ValueError("normalization standard deviation must be positive")
        return ChannelStatistics(self.count, float(self.mean), float(std))


@dataclass(frozen=True)
class ForegroundStatistics:
    schema: str
    visit_ids: tuple[str, ...]
    channels: Mapping[str, ChannelStatistics]

    @classmethod
    def from_visit_arrays(
        cls, arrays: Mapping[str, tuple[np.ndarray, np.ndarray]]
    ) -> "ForegroundStatistics":
        dce0_moments = _Moments()
        ser_moments = _Moments()
        visit_ids = tuple(sorted(str(value) for value in arrays))
        if len(visit_ids) != len(arrays) or not visit_ids:
            raise ValueError("normalization visit identities must be unique and nonempty")
        for visit_id in visit_ids:
            dce0, ser = (np.asarray(value) for value in arrays[visit_id])
            if dce0.shape != ser.shape or dce0.ndim != 3:
                raise ValueError("DCE0 and SER normalization arrays must be matching 3D volumes")
            valid = np.isfinite(dce0) & (dce0 != 0)
            if not valid.any() or not np.isfinite(ser[valid]).all():
                raise ValueError("DCE0 foreground and corresponding SER must be finite")
            dce0_moments.update(dce0[valid])
            ser_moments.update(ser[valid])
        return cls(
            schema=STATISTICS_SCHEMA,
            visit_ids=visit_ids,
            channels={"dce0": dce0_moments.finish(), "ser": ser_moments.finish()},
        )

    @classmethod
    def from_visit_loader(cls, visit_ids: Sequence[str], loader) -> "ForegroundStatistics":
        identifiers = tuple(sorted(str(value) for value in visit_ids))
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise ValueError("normalization visit identities must be unique and nonempty")
        dce0_moments = _Moments()
        ser_moments = _Moments()
        for visit_id in identifiers:
            dce0, ser = (np.asarray(value) for value in loader(visit_id))
            if dce0.shape != ser.shape or dce0.ndim != 3:
                raise ValueError("DCE0 and SER normalization arrays must be matching 3D volumes")
            valid = np.isfinite(dce0) & (dce0 != 0)
            if not valid.any() or not np.isfinite(ser[valid]).all():
                raise ValueError("DCE0 foreground and corresponding SER must be finite")
            dce0_moments.update(dce0[valid])
            ser_moments.update(ser[valid])
        return cls(
            schema=STATISTICS_SCHEMA,
            visit_ids=identifiers,
            channels={"dce0": dce0_moments.finish(), "ser": ser_moments.finish()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "visit_ids": list(self.visit_ids),
            "channels": {name: value.to_dict() for name, value in self.channels.items()},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ForegroundStatistics":
        if value.get("schema") != STATISTICS_SCHEMA:
            raise ValueError("normalization statistics schema is invalid")
        channels = value.get("channels")
        if not isinstance(channels, Mapping) or set(channels) != {"dce0", "ser"}:
            raise ValueError("normalization statistics channels are invalid")
        if any(
            not isinstance(channels[name], Mapping)
            or set(channels[name]) != {"count", "mean", "std"}
            or type(channels[name]["count"]) is not int
            for name in ("dce0", "ser")
        ):
            raise ValueError("normalization statistics values are invalid")
        try:
            parsed = {
                name: ChannelStatistics(
                    count=channels[name]["count"],
                    mean=float(channels[name]["mean"]),
                    std=float(channels[name]["std"]),
                )
                for name in ("dce0", "ser")
            }
        except (KeyError, TypeError, ValueError, OverflowError):
            raise ValueError("normalization statistics values are invalid") from None
        if any(
            item.count <= 0
            or not math.isfinite(item.mean)
            or not math.isfinite(item.std)
            or item.std <= 0
            for item in parsed.values()
        ):
            raise ValueError("normalization statistics values are invalid")
        visits = tuple(str(item) for item in value.get("visit_ids", ()))
        if not visits or len(visits) != len(set(visits)):
            raise ValueError("normalization visit identities are invalid")
        return cls(STATISTICS_SCHEMA, visits, parsed)


def apply_global_zscore(
    dce0_zyx: np.ndarray,
    ser_zyx: np.ndarray,
    statistics: ForegroundStatistics,
) -> tuple[np.ndarray, np.ndarray]:
    dce0 = np.asarray(dce0_zyx, dtype=np.float32)
    ser = np.asarray(ser_zyx, dtype=np.float32)
    if dce0.shape != ser.shape or dce0.ndim != 3:
        raise ValueError("DCE0 and SER must be matching 3D arrays")
    valid = np.isfinite(dce0) & (dce0 != 0)
    if not valid.any():
        raise ValueError("DCE0 foreground must be nonempty")
    if not np.isfinite(ser[valid]).all():
        raise ValueError("SER must be finite on the DCE0 foreground")
    output = np.zeros((2, *dce0.shape), dtype=np.float32)
    for index, (name, array) in enumerate((("dce0", dce0), ("ser", ser))):
        channel = statistics.channels[name]
        output[index, valid] = (array[valid] - channel.mean) / channel.std
    if not np.isfinite(output).all():
        raise ValueError("normalized MRI contains non-finite values")
    return output, valid


def invert_global_zscore(
    normalized_zyx: np.ndarray,
    statistics: ChannelStatistics,
    valid_foreground_zyx: np.ndarray,
) -> np.ndarray:
    normalized = np.asarray(normalized_zyx, dtype=np.float32)
    valid = np.asarray(valid_foreground_zyx, dtype=bool)
    if normalized.shape != valid.shape or normalized.ndim != 3:
        raise ValueError("inverse normalization arrays must be matching 3D volumes")
    output = np.zeros_like(normalized, dtype=np.float32)
    output[valid] = normalized[valid] * statistics.std + statistics.mean
    if not np.isfinite(output).all():
        raise ValueError("inverse normalized MRI contains non-finite values")
    return output


def deterministic_patient_split(
    patient_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 2026,
    val_fraction: float = 0.1,
) -> dict[str, Any]:
    if seed != 2026 or not math.isclose(float(val_fraction), 0.1):
        raise ValueError("registered Strict-A split requires seed 2026 and val fraction 0.1")
    columns = ("pCR", "HR", "HER2", "MP", "trial_arm", "visit_pattern")
    normalized: list[dict[str, str]] = []
    for row in patient_rows:
        patient_id = str(row.get("patient_id", "")).strip()
        if not patient_id:
            raise ValueError("split patient_id must be nonempty")
        normalized.append(
            {
                "patient_id": patient_id,
                **{name: str(row.get(name, "missing")) for name in columns},
            }
        )
    if not normalized or len({row["patient_id"] for row in normalized}) != len(normalized):
        raise ValueError("split patients must be unique and nonempty")
    n_val = max(1, int(math.floor(len(normalized) * 0.1 + 0.5)))
    capacities = {"train": len(normalized) - n_val, "val": n_val}
    fractions = {"train": 0.9, "val": 0.1}
    totals = Counter(
        f"{name}={row[name]}" for row in normalized for name in columns
    )
    rng = np.random.default_rng(seed)
    jitter = {row["patient_id"]: float(rng.random()) for row in normalized}
    ordered = sorted(
        normalized,
        key=lambda row: (
            -sum(1.0 / totals[f"{name}={row[name]}"] for name in columns),
            jitter[row["patient_id"]],
            row["patient_id"],
        ),
    )
    split: dict[str, list[str]] = {"train": [], "val": []}
    assigned = {fold: Counter() for fold in split}
    for row in ordered:
        features = [f"{name}={row[name]}" for name in columns]
        choices = [fold for fold in split if len(split[fold]) < capacities[fold]]

        def cost(fold: str) -> tuple[float, int, str]:
            score = 0.0
            for feature in features:
                target = totals[feature] * fractions[fold]
                current = assigned[fold][feature]
                score += ((current + 1 - target) ** 2 - (current - target) ** 2) / max(
                    target, 1.0
                )
            return score, len(split[fold]), fold

        selected = min(choices, key=cost)
        split[selected].append(row["patient_id"])
        assigned[selected].update(features)
    for fold in split:
        split[fold].sort()
    return {
        "schema": "motfm_ispy2_registered_patient_split_v1",
        "seed": seed,
        "train_fraction": 0.9,
        "val_fraction": 0.1,
        "balance_columns": list(columns),
        "train": split["train"],
        "val": split["val"],
    }

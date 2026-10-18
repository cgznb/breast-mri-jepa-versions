#!/usr/bin/env python3
"""Build the registered, T0-fixed, Strict-A longitudinal training bundle."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk

from scripts.build_ispy2_longitudinal_bundle import (
    verify_bundle as verify_legacy_bundle,
)
from utils.ispy2_preprocessing import geometry_from_meta, resample_zyx_to_ras
from utils.ispy2_registry import load_clinical_table, normalize_patient_id
from utils.ispy2_registered import (
    BUNDLE_SCHEMA,
    COORDINATE_FRAME,
    OUTPUT_SHAPE_ZYX,
    TARGET_SPACING_XYZ,
    CropPlan,
    ForegroundStatistics,
    crop_or_pad_zyx,
    deterministic_patient_split,
    reconstruct_transition_universe,
    selected_registration_metrics,
    sha256_file,
    sha256_json,
    t0_fixed_crop_plan,
    visit_fails_strict_a,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path("/root/autodl-tmp")
DATA_ROOT = WORKSPACE / "Datasets/manifest-1781750940386/ISPY2"
DEFAULT_LEGACY_BUNDLE = (
    WORKSPACE / "MAM/MOTFM/runs/ispy2_longitudinal_3d/data_bundle/bundle.json"
)
DEFAULT_REGISTERED_MANIFEST = (
    DATA_ROOT
    / "processed_ispy2_world_phase3_scan_phase_aligned_multires_registered_to_T0"
    / "manifest/ispy2_breastdcedl_registered_to_T0_manifest.csv"
)
DEFAULT_ORIGINAL_MANIFEST = (
    DATA_ROOT
    / "processed_ispy2_world_phase3_scan/manifest/ispy2_breastdcedl_full_manifest.csv"
)
DEFAULT_STRICT_RULES = (
    WORKSPACE
    / (
        "MAM/Quantitative-MRI-of-breast-cancer-patients-to-forecast-response-"
        "to-therapy-registration-t0-python"
    )
    / "docs/ispy2_strict_a_selection_rules.md"
)
DEFAULT_CLINICAL = (
    WORKSPACE
    / "Datasets/manifest-1781750940386/ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx"
)
TERMINAL_STATES = {"fixed_reference", "deformable", "rigid_fallback"}
ALGORITHM_ANOMALIES = {"bspline_failed", "jacobian_nonfinite", "deformable_nonfinite"}
REGISTERED_ORIENTATION_LPS = (
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
)
BASE_SOURCE_CONTRACT_KEYS = {
    "legacy_bundle_sha256",
    "registered_manifest_sha256",
    "original_manifest_sha256",
    "clinical_sha256",
    "strict_a_rules_sha256",
}
EXPANDED_SOURCE_CONTRACT_KEYS = BASE_SOURCE_CONTRACT_KEYS | {
    "base_bundle_sha256",
    "expanded_registered_manifest_sha256",
    "expanded_original_manifest_sha256",
    "incremental_metadata_sha256",
    "incremental_patient_allowlist_sha256",
}
BASE_DATA_CONTRACT_SCHEMA = "motfm_ispy2_frozen_base_data_v1"
INCREMENT_CONTRACT_SCHEMA = "motfm_ispy2_registered_increment_v1"
EXPANDED_SPLIT_POLICIES = {
    "frozen_base_val_new_train_v1",
    "frozen_base_val_new_patient_holdout_v1",
}
INCREMENT_FILENAMES = {
    "new_patient_ids_sha256": "increment/new_patient_ids.txt",
    "new_visit_ids_sha256": "increment/new_visit_ids.txt",
    "new_transition_ids_sha256": "increment/new_transition_ids.txt",
    "new_exclusions_sha256": "increment/new_exclusions.csv",
    "summary_sha256": "increment/summary.json",
}
BASE_VISIT_GEOMETRY_FIELDS = {
    "orientation_lps_json",
    "resampled_shape_zyx_json",
    "resampled_bbox_min_zyx_json",
    "resampled_bbox_max_zyx_json",
    "resampled_bbox_shape_zyx_json",
    "mask_retention_reason",
}
ARTIFACT_FILENAMES = {
    "visits": "visits.csv",
    "transitions": "transitions.csv",
    "exclusions": "exclusions.csv",
    "split": "split.json",
    "preprocess": "preprocess.json",
    "normalization": "normalization.json",
    "crop_plans": "crop_plans.json",
}
VISIT_COLUMNS = (
    "patient_id",
    "visit_id",
    "visit",
    "visit_index",
    "study_instance_uid",
    "visit_date",
    "visit_date_source",
    "qc_status",
    "dce0_path",
    "ser_path",
    "mask_path",
    "meta_path",
    "ftv_voxel_count",
    "row_spacing_mm",
    "column_spacing_mm",
    "slice_spacing_mm",
    "ftv_voxel_volume_mm3",
    "ftv_volume_cc",
    "HR",
    "HER2",
    "MP",
    "Age_at_Screening",
    "menopausal_status",
    "trial_arm",
    "clinical_text",
    "fold",
    "orientation_lps_json",
    "resampled_shape_zyx_json",
    "mask_voxel_count_resampled",
    "resampled_bbox_min_zyx_json",
    "resampled_bbox_max_zyx_json",
    "resampled_bbox_shape_zyx_json",
    "crop_start_zyx_json",
    "mask_fully_retained",
    "mask_retention_reason",
    "original_meta_path",
    "registration_status",
    "registration_json_path",
    "is_t0",
    "in_plane_fov_ratio",
    "selected_non_tumor_ncc",
    "selected_protected_ncc",
    "rotation_degrees",
    "translation_mm",
    "rigid_maximum_iterations_reached",
    "algorithm_anomaly",
    "algorithm_anomaly_codes",
    "quality_pass",
    "mask_voxel_count_after_crop",
    "quality_evidence_json",
)
TRANSITION_COLUMNS = (
    "transition_id",
    "patient_id",
    "fold",
    "transition_type",
    "source_visit_id",
    "target_visit_id",
    "source_visit",
    "target_visit",
    "source_study_instance_uid",
    "target_study_instance_uid",
    "delta_days",
    "orientation_angle_degrees",
    "source_dce0_path",
    "source_ser_path",
    "source_mask_path",
    "source_meta_path",
    "target_dce0_path",
    "target_mask_path",
    "target_meta_path",
    "source_ftv_volume_cc",
    "target_ftv_volume_cc",
    "source_ftv_is_condition",
    "target_ftv_is_condition",
    "target_ftv_is_audit_only",
    "clinical_text",
    "action_text",
)
EXCLUSION_COLUMNS = (
    "transition_id",
    "patient_id",
    "source_visit_id",
    "target_visit_id",
    "transition_type",
    "reason",
    "detail",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError(f"JSON is unreadable: {path}") from None
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def require_source_files(paths: Mapping[str, str | Path]) -> None:
    """Fail before volume processing when a frozen source is unavailable."""

    for name, path in paths.items():
        if not Path(path).expanduser().is_file():
            raise FileNotFoundError(f"registered Strict-A source is missing: {name}")


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _dce0_path(value: Any) -> str:
    text = str(value)
    paths = [item for item in text.split(";") if item]
    if not paths:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = []
        paths = parsed if isinstance(parsed, list) else []
    if not paths:
        raise ValueError("registered DCE path list is empty")
    return str(paths[0])


def _fov_xy(meta_path: str | Path) -> tuple[float, float]:
    meta = _read_json(meta_path)
    try:
        spacing = tuple(float(value) for value in meta["pixel_spacing"])
        rows, cols = int(meta["rows"]), int(meta["cols"])
    except (KeyError, TypeError, ValueError, OverflowError):
        raise ValueError("original visit FOV metadata is invalid") from None
    if len(spacing) != 2 or min(*spacing, rows, cols) <= 0:
        raise ValueError("original visit FOV metadata is invalid")
    return cols * spacing[1], rows * spacing[0]


def _fov_ratio(visit_fov: Sequence[float], t0_fov: Sequence[float]) -> float:
    ratios = [a / b for a, b in zip(visit_fov, t0_fov)] + [
        b / a for a, b in zip(visit_fov, t0_fov)
    ]
    if any(not math.isfinite(value) or value <= 0 for value in ratios):
        raise ValueError("original in-plane FOV ratio is invalid")
    return max(ratios)


def _rigid_motion(path: str | Path) -> tuple[float, float]:
    try:
        transform = sitk.ReadTransform(str(path))
        if hasattr(transform, "GetBackTransform"):
            transform = transform.GetBackTransform()
        matrix = np.asarray(transform.GetMatrix(), dtype=np.float64).reshape(3, 3)
        translation = np.asarray(transform.GetTranslation(), dtype=np.float64)
    except (RuntimeError, ValueError, AttributeError):
        raise ValueError(f"rigid transform is unreadable: {path}") from None
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine)), float(np.linalg.norm(translation))


def visit_quality_evidence(
    *,
    visit: str,
    registration: Mapping[str, Any],
    in_plane_fov_ratio: float,
    rotation_degrees: float,
    translation_mm: float,
) -> dict[str, Any]:
    if visit == "T0":
        return {
            "is_t0": True,
            "registration_status": "fixed_reference",
            "in_plane_fov_ratio": 1.0,
            "selected_non_tumor_ncc": None,
            "selected_protected_ncc": None,
            "rotation_degrees": 0.0,
            "translation_mm": 0.0,
            "rigid_maximum_iterations_reached": False,
            "algorithm_anomaly": False,
            "algorithm_anomaly_codes": [],
            "quality_pass": True,
        }
    status = str(registration.get("registration_status", ""))
    qc = registration.get("qc")
    if not isinstance(qc, Mapping):
        raise ValueError("registration QC is unavailable")
    codes = {
        str(item)
        for key in ("reasons", "qc_warning_codes")
        for item in (qc.get(key) or [])
    }
    anomaly_codes = sorted(codes.intersection(ALGORITHM_ANOMALIES))
    try:
        selected_non_tumor, selected_protected = selected_registration_metrics(
            status, qc
        )
    except ValueError:
        if not anomaly_codes:
            raise
        selected_non_tumor = selected_protected = None
    stop = str(qc.get("rigid_stop_condition", "")).lower()
    maximum_iterations = int(qc.get("rigid_iterations", 0) or 0) >= 500 or (
        "maximum number of iterations" in stop
    )
    evidence = {
        "is_t0": False,
        "registration_status": status,
        "in_plane_fov_ratio": float(in_plane_fov_ratio),
        "selected_non_tumor_ncc": selected_non_tumor,
        "selected_protected_ncc": selected_protected,
        "rotation_degrees": float(rotation_degrees),
        "translation_mm": float(translation_mm),
        "rigid_maximum_iterations_reached": maximum_iterations,
        "algorithm_anomaly": bool(anomaly_codes),
        "algorithm_anomaly_codes": anomaly_codes,
    }
    evidence["quality_pass"] = not visit_fails_strict_a(evidence)
    return evidence


def strict_a_transition_reasons(
    transition: Mapping[str, Any],
    visits: Mapping[str, Mapping[str, Any]],
    *,
    patient_has_t0: bool,
) -> list[str]:
    reasons: list[str] = []
    if not patient_has_t0:
        reasons.append("patient_has_no_T0")
    if int(transition.get("delta_days", 0)) <= 0:
        reasons.append("nonpositive_delta_days")
    source = visits[str(transition["source_visit_id"])]
    target = visits[str(transition["target_visit_id"])]
    if str(source.get("registration_status")) not in TERMINAL_STATES:
        reasons.append("source_registration_not_terminal")
    if str(target.get("registration_status")) not in TERMINAL_STATES:
        reasons.append("target_registration_not_terminal")
    if not bool(source.get("quality_pass")):
        reasons.append("source_visit_quality_failed")
    if not bool(target.get("quality_pass")):
        reasons.append("target_visit_quality_failed")
    if patient_has_t0:
        t0 = visits[f"{transition['patient_id']}:T0"]
        if not bool(t0.get("mask_fully_retained")):
            reasons.append("T0_mask_not_fully_retained")
    if not bool(source.get("mask_fully_retained")):
        reasons.append("source_mask_not_fully_retained")
    if not bool(target.get("mask_fully_retained")):
        reasons.append("target_mask_not_fully_retained")
    return reasons


def count_evaluable_transitions(
    *, selected_count: int, excluded: Sequence[Mapping[str, Any]]
) -> int:
    """Count transitions with T0, including quality and crop exclusions."""

    return int(selected_count) + sum(
        "patient_has_no_T0" not in str(item["reason"]).split("+")
        for item in excluded
    )


def _load_resampled(
    path: str | Path,
    meta_path: str | Path,
    *,
    is_mask: bool,
    target_spacing_xyz: Sequence[float],
) -> np.ndarray:
    try:
        array = np.asanyarray(nib.load(str(path)).dataobj)
    except (OSError, ValueError):
        raise ValueError(f"NIfTI is unreadable: {path}") from None
    meta = _read_json(meta_path)
    resampled, _ = resample_zyx_to_ras(
        array,
        geometry_from_meta(meta),
        target_spacing_xyz=target_spacing_xyz,
        is_mask=is_mask,
    )
    return np.asarray(resampled)


def _crop_visit_images(
    visit: Mapping[str, Any],
    plan: CropPlan,
    *,
    target_spacing_xyz: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    dce0 = _load_resampled(
        visit["dce0_path"],
        visit["meta_path"],
        is_mask=False,
        target_spacing_xyz=target_spacing_xyz,
    )
    ser = _load_resampled(
        visit["ser_path"], visit["meta_path"], is_mask=False, target_spacing_xyz=target_spacing_xyz
    )
    if dce0.shape != ser.shape or tuple(dce0.shape) != plan.input_shape_zyx:
        raise ValueError("registered visit images do not share the patient T0 grid")
    return (
        crop_or_pad_zyx(dce0, plan.crop_start_zyx, plan.output_shape_zyx),
        crop_or_pad_zyx(ser, plan.crop_start_zyx, plan.output_shape_zyx),
    )


def _registered_mask_geometry(binary: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(binary)
    if mask.ndim != 3:
        raise ValueError("registered mask geometry requires a 3D array")
    coordinates = np.argwhere(mask > 0)
    if coordinates.size:
        minimum = coordinates.min(axis=0).astype(int)
        maximum = coordinates.max(axis=0).astype(int)
        bbox_shape = maximum - minimum + 1
        bbox_min = minimum.tolist()
        bbox_max = maximum.tolist()
        bbox_size = bbox_shape.tolist()
    else:
        bbox_min = bbox_max = bbox_size = []
    return {
        "orientation_lps_json": json.dumps(
            REGISTERED_ORIENTATION_LPS, separators=(",", ":")
        ),
        "resampled_shape_zyx_json": json.dumps(
            [int(value) for value in mask.shape], separators=(",", ":")
        ),
        "resampled_bbox_min_zyx_json": json.dumps(
            bbox_min, separators=(",", ":")
        ),
        "resampled_bbox_max_zyx_json": json.dumps(
            bbox_max, separators=(",", ":")
        ),
        "resampled_bbox_shape_zyx_json": json.dumps(
            bbox_size, separators=(",", ":")
        ),
    }


def _registered_rows(
    legacy_visits: pd.DataFrame, registered_manifest: pd.DataFrame
) -> pd.DataFrame:
    required = {
        "patient_id",
        "visit",
        "dce_paths",
        "ser_path",
        "mask_path",
        "meta_path",
        "registration_status",
    }
    if not required.issubset(registered_manifest.columns):
        raise ValueError("registered manifest fields are incomplete")
    registered = {
        (str(row["patient_id"]), str(row["visit"])): row
        for row in registered_manifest.to_dict(orient="records")
    }
    if len(registered) != len(registered_manifest):
        raise ValueError("registered manifest visit identities are duplicated")
    rows: list[dict[str, Any]] = []
    for legacy in legacy_visits.to_dict(orient="records"):
        key = (str(legacy["patient_id"]), str(legacy["visit"]))
        if key not in registered:
            raise ValueError(f"frozen visit is missing from registered manifest: {key}")
        current = registered[key]
        row = dict(legacy)
        row.update(
            {
                "original_meta_path": str(legacy["meta_path"]),
                "dce0_path": _dce0_path(current["dce_paths"]),
                "ser_path": str(current["ser_path"]),
                "mask_path": str(current["mask_path"]),
                "meta_path": str(current["meta_path"]),
                "registration_status": str(current["registration_status"]),
                "registration_json_path": str(
                    Path(str(current["meta_path"])).parents[1] / "registration.json"
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _assess_registered_visits(
    visits: pd.DataFrame,
    *,
    target_spacing_xyz: Sequence[float],
    output_shape_zyx: Sequence[int],
    progress_every: int,
    mask_loader=None,
    crop_plan_builder: Callable[..., CropPlan] = t0_fixed_crop_plan,
) -> tuple[pd.DataFrame, dict[str, CropPlan]]:
    rows = visits.to_dict(orient="records")
    by_patient: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_patient.setdefault(str(row["patient_id"]), []).append(row)
    output: list[dict[str, Any]] = []
    plans: dict[str, CropPlan] = {}
    completed = 0
    for patient_id in sorted(by_patient):
        patient_rows = by_patient[patient_id]
        t0_rows = [row for row in patient_rows if str(row["visit"]) == "T0"]
        plan = None
        t0_fov = None
        if len(t0_rows) == 1 and str(t0_rows[0]["registration_status"]) == "fixed_reference":
            t0 = t0_rows[0]
            t0_mask = (
                mask_loader(t0)
                if mask_loader is not None
                else _load_resampled(
                    t0["mask_path"],
                    t0["meta_path"],
                    is_mask=True,
                    target_spacing_xyz=target_spacing_xyz,
                )
            )
            plan = crop_plan_builder(t0_mask, output_shape_zyx=output_shape_zyx)
            plans[patient_id] = plan
            t0_fov = _fov_xy(t0["original_meta_path"])
        for source in sorted(patient_rows, key=lambda row: int(row["visit_index"])):
            row = dict(source)
            if plan is None:
                row.update(
                    {
                        "mask_voxel_count_resampled": 0,
                        "mask_voxel_count_after_crop": 0,
                        "mask_fully_retained": False,
                        "crop_start_zyx_json": "[]",
                        "quality_pass": False,
                        "quality_evidence_json": json.dumps(
                            {"reason": "patient_has_no_T0"}, separators=(",", ":")
                        ),
                    }
                )
            else:
                mask = (
                    mask_loader(row)
                    if mask_loader is not None
                    else _load_resampled(
                        row["mask_path"],
                        row["meta_path"],
                        is_mask=True,
                        target_spacing_xyz=target_spacing_xyz,
                    )
                )
                if tuple(mask.shape) != plan.input_shape_zyx:
                    raise ValueError(f"registered patient grid mismatch: {row['visit_id']}")
                binary = (mask > 0).astype(np.uint8)
                cropped = crop_or_pad_zyx(
                    binary, plan.crop_start_zyx, plan.output_shape_zyx
                )
                before, after = int(binary.sum()), int(cropped.sum())
                if str(row["visit"]) == "T0":
                    registration = {"registration_status": "fixed_reference", "qc": {}}
                    motion = (0.0, 0.0)
                    ratio = 1.0
                else:
                    registration = _read_json(row["registration_json_path"])
                    transform_path = registration.get("transform_path")
                    rigid_path = Path(str(transform_path)).parent / "rigid.tfm"
                    motion = _rigid_motion(rigid_path)
                    ratio = _fov_ratio(_fov_xy(row["original_meta_path"]), t0_fov)
                evidence = visit_quality_evidence(
                    visit=str(row["visit"]),
                    registration=registration,
                    in_plane_fov_ratio=ratio,
                    rotation_degrees=motion[0],
                    translation_mm=motion[1],
                )
                row.update(evidence)
                row.update(_registered_mask_geometry(binary))
                row.update(
                    {
                        "mask_voxel_count_resampled": before,
                        "mask_voxel_count_after_crop": after,
                        "mask_fully_retained": before == after,
                        "mask_retention_reason": (
                            "fully_retained"
                            if before == after
                            else "truncated_by_t0_fixed_crop"
                        ),
                        "crop_start_zyx_json": json.dumps(list(plan.crop_start_zyx)),
                        "quality_evidence_json": json.dumps(
                            evidence, sort_keys=True, separators=(",", ":")
                        ),
                    }
                )
            output.append(row)
            completed += 1
            if completed % progress_every == 0:
                print(json.dumps({"event": "mask_assessment", "completed": completed}), flush=True)
    return pd.DataFrame(output), plans


def _patient_split_rows(
    eligible_patients: set[str], visits: pd.DataFrame, clinical_path: Path
) -> list[dict[str, Any]]:
    clinical = load_clinical_table(clinical_path)
    clinical_by_id = {
        normalize_patient_id(row["Patient_ID"]): row
        for row in clinical.to_dict(orient="records")
    }
    rows: list[dict[str, Any]] = []
    for patient_id in sorted(eligible_patients):
        if patient_id not in clinical_by_id:
            raise ValueError(f"eligible patient is missing clinical data: {patient_id}")
        source = visits.loc[visits["patient_id"].astype(str) == patient_id]
        example = source.iloc[0]
        clinical_row = clinical_by_id[patient_id]
        pattern = "+".join(
            sorted(set(source["visit"].astype(str)), key=lambda value: int(value[1:]))
        )
        rows.append(
            {
                "patient_id": patient_id,
                "pCR": clinical_row.get("pCR", "missing"),
                "HR": example.get("HR", "missing"),
                "HER2": example.get("HER2", "missing"),
                "MP": example.get("MP", "missing"),
                "trial_arm": example.get("trial_arm", "missing"),
                "visit_pattern": pattern,
            }
        )
    return rows


def _artifact(path: Path, *, row_count: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"path": path.name, "sha256": sha256_file(path)}
    if row_count is not None:
        value["row_count"] = int(row_count)
    return value


def _canonical_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("registered Strict-A base record contains non-finite data")
        return float(value)
    if isinstance(value, (bool, int, str)):
        return value
    raise ValueError("registered Strict-A base record contains unsupported data")


def _records_sha256(
    frame: pd.DataFrame, *, identity: str, columns: Sequence[str]
) -> str:
    if identity not in frame.columns or any(column not in frame.columns for column in columns):
        raise ValueError("registered Strict-A base record schema is invalid")
    records = []
    for row in frame.sort_values(identity).to_dict(orient="records"):
        records.append(
            {column: _canonical_scalar(row[column]) for column in columns}
        )
    return sha256_json(records)


def build_base_data_contract(
    *,
    bundle_contract_sha256: str,
    visits: pd.DataFrame,
    transitions: pd.DataFrame,
    exclusions: pd.DataFrame,
    split: Mapping[str, Any],
    crop_plans: Mapping[str, Any],
) -> dict[str, Any]:
    visit_columns = [
        column for column in VISIT_COLUMNS if column not in BASE_VISIT_GEOMETRY_FIELDS
    ]
    normalized_split = {
        "train": sorted(str(value) for value in split.get("train", [])),
        "val": sorted(str(value) for value in split.get("val", [])),
    }
    normalized_plans = {
        str(patient): crop_plans[patient] for patient in sorted(crop_plans)
    }
    return {
        "schema": BASE_DATA_CONTRACT_SCHEMA,
        "bundle_contract_sha256": str(bundle_contract_sha256),
        "split_sha256": sha256_json(normalized_split),
        "visits_semantic_sha256": _records_sha256(
            visits,
            identity="visit_id",
            columns=visit_columns,
        ),
        "transitions_sha256": _records_sha256(
            transitions,
            identity="transition_id",
            columns=TRANSITION_COLUMNS,
        ),
        "exclusion_row_count": len(exclusions),
        "exclusions_sha256": _records_sha256(
            exclusions,
            identity="transition_id",
            columns=EXCLUSION_COLUMNS,
        ),
        "crop_plans_sha256": sha256_json(normalized_plans),
    }


def build_increment_contract(bundle_root: str | Path) -> dict[str, str]:
    root = Path(bundle_root)
    return {
        "schema": INCREMENT_CONTRACT_SCHEMA,
        **{
            field: sha256_file(root / relative_path)
            for field, relative_path in INCREMENT_FILENAMES.items()
        },
    }


def build_bundle(
    *,
    legacy_bundle: Path,
    registered_manifest_path: Path,
    original_manifest_path: Path,
    clinical_path: Path,
    strict_rules_path: Path,
    output_dir: Path,
    target_spacing_xyz: Sequence[float] = TARGET_SPACING_XYZ,
    output_shape_zyx: Sequence[int] = OUTPUT_SHAPE_ZYX,
    progress_every: int = 100,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if (output_dir / "bundle.json").is_file():
        return verify_bundle(output_dir / "bundle.json")
    if output_dir.exists():
        raise FileExistsError("incomplete registered bundle directory already exists")
    require_source_files(
        {
            "legacy_bundle": legacy_bundle,
            "registered_manifest": registered_manifest_path,
            "original_manifest": original_manifest_path,
            "clinical": clinical_path,
            "strict_a_rules": strict_rules_path,
        }
    )
    legacy = verify_legacy_bundle(legacy_bundle)
    legacy_root = legacy_bundle.parent
    legacy_visits = pd.read_csv(legacy_root / legacy["artifacts"]["visits"]["path"])
    accepted = pd.read_csv(legacy_root / legacy["artifacts"]["transitions"]["path"])
    prior_exclusions = pd.read_csv(legacy_root / legacy["artifacts"]["exclusions"]["path"])
    if (len(accepted), len(prior_exclusions)) != (1196, 358):
        raise ValueError("frozen transition universe must contain 1196 accepted and 358 excluded")
    universe = reconstruct_transition_universe(legacy_visits, accepted, prior_exclusions)
    if len(universe) != 1554:
        raise ValueError("frozen transition universe must contain 1554 transitions")
    registered_manifest = pd.read_csv(registered_manifest_path)
    registered_visits = _registered_rows(legacy_visits, registered_manifest)
    assessed, plans = _assess_registered_visits(
        registered_visits,
        target_spacing_xyz=target_spacing_xyz,
        output_shape_zyx=output_shape_zyx,
        progress_every=progress_every,
    )
    by_visit = {
        str(row["visit_id"]): row for row in assessed.to_dict(orient="records")
    }
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for transition in universe:
        patient_id = str(transition["patient_id"])
        reasons = strict_a_transition_reasons(
            transition,
            by_visit,
            patient_has_t0=patient_id in plans,
        )
        if reasons:
            excluded.append(
                {
                    "transition_id": transition["transition_id"],
                    "patient_id": patient_id,
                    "source_visit_id": transition["source_visit_id"],
                    "target_visit_id": transition["target_visit_id"],
                    "transition_type": transition["transition_type"],
                    "reason": "+".join(reasons),
                    "detail": json.dumps({"reasons": reasons}, separators=(",", ":")),
                }
            )
            continue
        source = by_visit[str(transition["source_visit_id"])]
        target = by_visit[str(transition["target_visit_id"])]
        row = dict(transition)
        row.update(
            {
                "source_dce0_path": source["dce0_path"],
                "source_ser_path": source["ser_path"],
                "source_mask_path": source["mask_path"],
                "source_meta_path": source["meta_path"],
                "target_dce0_path": target["dce0_path"],
                "target_mask_path": target["mask_path"],
                "target_meta_path": target["meta_path"],
            }
        )
        selected.append(row)
    eligible_patients = {str(row["patient_id"]) for row in selected}
    split = deterministic_patient_split(
        _patient_split_rows(eligible_patients, assessed, clinical_path), seed=2026
    )
    fold_by_patient = {
        patient: fold for fold in ("train", "val") for patient in split[fold]
    }
    for row in selected:
        row["fold"] = fold_by_patient[str(row["patient_id"])]
    selected.sort(key=lambda row: str(row["transition_id"]))
    referenced_ids = {
        str(row[key])
        for row in selected
        for key in ("source_visit_id", "target_visit_id")
    }
    referenced_ids.update(f"{patient}:T0" for patient in eligible_patients)
    published_visits = assessed.loc[assessed["visit_id"].astype(str).isin(referenced_ids)].copy()
    published_visits["fold"] = published_visits["patient_id"].astype(str).map(fold_by_patient)
    published_visits = published_visits.sort_values(["patient_id", "visit_index"])
    visit_records = {
        str(row["visit_id"]): row for row in published_visits.to_dict(orient="records")
    }
    train_visit_ids = sorted(
        {
            str(row[key])
            for row in selected
            if row["fold"] == "train"
            for key in ("source_visit_id", "target_visit_id")
        }
    )

    def load_for_statistics(visit_id: str):
        visit = visit_records[visit_id]
        return _crop_visit_images(
            visit,
            plans[str(visit["patient_id"])],
            target_spacing_xyz=target_spacing_xyz,
        )

    statistics = ForegroundStatistics.from_visit_loader(train_visit_ids, load_for_statistics)
    staging = output_dir.with_name(f".{output_dir.name}.building")
    if staging.exists():
        raise FileExistsError("registered bundle staging directory already exists")
    staging.mkdir(parents=True)
    try:
        visits_path = staging / "visits.csv"
        transitions_path = staging / "transitions.csv"
        exclusions_path = staging / "exclusions.csv"
        split_path = staging / "split.json"
        normalization_path = staging / "normalization.json"
        crop_path = staging / "crop_plans.json"
        preprocess_path = staging / "preprocess.json"
        published_visits.to_csv(visits_path, index=False)
        pd.DataFrame(selected).to_csv(transitions_path, index=False)
        pd.DataFrame(excluded).to_csv(exclusions_path, index=False)
        _json_dump(split_path, split)
        _json_dump(normalization_path, statistics.to_dict())
        _json_dump(
            crop_path,
            {
                patient: plan.to_dict()
                for patient, plan in sorted(plans.items())
                if patient in eligible_patients
            },
        )
        preprocess = {
            "schema": "motfm_ispy2_registered_preprocess_v1",
            "coordinate_frame": COORDINATE_FRAME,
            "target_spacing_xyz": [float(value) for value in target_spacing_xyz],
            "output_shape_zyx": [int(value) for value in output_shape_zyx],
            "crop_policy": "single_T0_mask_bbox_center_reused_for_all_visits",
            "image_interpolation": "linear",
            "mask_interpolation": "nearest",
            "normalization": {
                "schema": statistics.schema,
                "scope": "unique_train_transition_visits",
                "foreground": "finite_nonzero_DCE0",
                "SER_foreground": "same_as_DCE0",
                "outside_fov_and_padding": 0,
                "normalization_sha256": sha256_file(normalization_path),
            },
        }
        _json_dump(preprocess_path, preprocess)
        evaluable = count_evaluable_transitions(
            selected_count=len(selected), excluded=excluded
        )
        counts = {
            "universe_transition_count": 1554,
            "evaluable_transition_count": evaluable,
            "transition_count": len(selected),
            "exclusion_count": len(excluded),
            "patient_count": len(eligible_patients),
            "visit_count": len(published_visits),
            "fold_transition_counts": {
                fold: sum(row["fold"] == fold for row in selected) for fold in ("train", "val")
            },
            "fold_patient_counts": {fold: len(split[fold]) for fold in ("train", "val")},
            "transition_type_counts": dict(
                sorted(
                    pd.Series([row["transition_type"] for row in selected])
                    .value_counts()
                    .to_dict()
                    .items()
                )
            ),
            "empty_source_or_target_mask_transitions": sum(
                int(by_visit[str(row["source_visit_id"])]["mask_voxel_count_resampled"]) == 0
                or int(by_visit[str(row["target_visit_id"])]["mask_voxel_count_resampled"]) == 0
                for row in selected
            ),
        }
        artifacts = {
            "visits": _artifact(visits_path, row_count=len(published_visits)),
            "transitions": _artifact(transitions_path, row_count=len(selected)),
            "exclusions": _artifact(exclusions_path, row_count=len(excluded)),
            "split": _artifact(split_path),
            "preprocess": _artifact(preprocess_path),
            "normalization": _artifact(normalization_path),
            "crop_plans": _artifact(crop_path),
        }
        source_contracts = {
            "legacy_bundle_sha256": sha256_file(legacy_bundle),
            "registered_manifest_sha256": sha256_file(registered_manifest_path),
            "original_manifest_sha256": sha256_file(original_manifest_path),
            "clinical_sha256": sha256_file(clinical_path),
            "strict_a_rules_sha256": sha256_file(strict_rules_path),
        }
        contract = {
            "schema_version": BUNDLE_SCHEMA,
            "counts": counts,
            "artifacts": artifacts,
            "source_contracts": source_contracts,
            "strict_a_rule_version": "v1",
            "split_policy": "patient_level_90_10_seed_2026",
            "text_policy": legacy.get("text_policy", {}),
        }
        contract["bundle_contract_sha256"] = sha256_json(contract)
        _json_dump(staging / "bundle.json", contract)
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify_bundle(output_dir / "bundle.json")


def _registered_artifact_frames(
    bundle_path: Path, artifacts: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for name, columns in (
        ("visits", VISIT_COLUMNS),
        ("transitions", TRANSITION_COLUMNS),
        ("exclusions", EXCLUSION_COLUMNS),
    ):
        try:
            frame = pd.read_csv(bundle_path.parent / ARTIFACT_FILENAMES[name])
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            raise ValueError(
                f"registered Strict-A {name} artifact is unreadable"
            ) from None
        if tuple(frame.columns) != columns:
            raise ValueError(f"registered Strict-A {name} schema is invalid")
        if len(frame) != artifacts[name]["row_count"]:
            raise ValueError(f"registered Strict-A {name} row count is invalid")
        frames.append(frame)
    return frames[0], frames[1], frames[2]


def _identifier_set(values: Any, *, name: str, allow_empty: bool = False) -> set[str]:
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError(f"registered Strict-A {name} identities are invalid")
    identifiers = set(values)
    if len(identifiers) != len(values) or (not allow_empty and not identifiers):
        raise ValueError(f"registered Strict-A {name} identities are invalid")
    return identifiers


def _json_int_triplet(value: Any, *, name: str) -> list[int]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        raise ValueError(f"registered Strict-A {name} is invalid") from None
    if (
        not isinstance(parsed, list)
        or len(parsed) != 3
        or any(type(item) is not int for item in parsed)
    ):
        raise ValueError(f"registered Strict-A {name} is invalid")
    return parsed


def _json_orientation(value: Any, *, name: str) -> list[list[float]]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
        matrix = np.asarray(parsed, dtype=np.float64)
    except (json.JSONDecodeError, TypeError, ValueError, OverflowError):
        raise ValueError(f"registered Strict-A {name} is invalid") from None
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"registered Strict-A {name} is invalid")
    if not np.allclose(matrix, np.asarray(REGISTERED_ORIENTATION_LPS), atol=1e-8):
        raise ValueError(f"registered Strict-A {name} is invalid")
    return matrix.tolist()


def _strict_csv_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"registered Strict-A {name} is invalid")


def _finite_csv_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"registered Strict-A {name} is invalid") from None
    if not math.isfinite(result):
        raise ValueError(f"registered Strict-A {name} is invalid")
    return result


def _validate_registered_preprocess(
    preprocess: Mapping[str, Any], normalization_sha256: str
) -> None:
    expected_keys = {
        "schema",
        "coordinate_frame",
        "target_spacing_xyz",
        "output_shape_zyx",
        "crop_policy",
        "image_interpolation",
        "mask_interpolation",
        "normalization",
    }
    normalization = preprocess.get("normalization")
    if (
        set(preprocess) != expected_keys
        or preprocess.get("schema") != "motfm_ispy2_registered_preprocess_v1"
        or preprocess.get("coordinate_frame") != COORDINATE_FRAME
        or preprocess.get("target_spacing_xyz") != list(TARGET_SPACING_XYZ)
        or preprocess.get("output_shape_zyx") != list(OUTPUT_SHAPE_ZYX)
        or preprocess.get("crop_policy")
        != "single_T0_mask_bbox_center_reused_for_all_visits"
        or preprocess.get("image_interpolation") != "linear"
        or preprocess.get("mask_interpolation") != "nearest"
        or not isinstance(normalization, Mapping)
        or set(normalization)
        != {
            "schema",
            "scope",
            "foreground",
            "SER_foreground",
            "outside_fov_and_padding",
            "normalization_sha256",
        }
        or normalization.get("schema") != "motfm_ispy2_train_foreground_stats_v1"
        or normalization.get("scope") != "unique_train_transition_visits"
        or normalization.get("foreground") != "finite_nonzero_DCE0"
        or normalization.get("SER_foreground") != "same_as_DCE0"
        or normalization.get("outside_fov_and_padding") != 0
        or normalization.get("normalization_sha256") != normalization_sha256
    ):
        raise ValueError("registered Strict-A preprocess contract is invalid")


def _validate_registered_relations(
    *,
    visits: pd.DataFrame,
    transitions: pd.DataFrame,
    exclusions: pd.DataFrame,
    split: Mapping[str, Any],
    crop_plans: Mapping[str, Any],
    statistics: ForegroundStatistics,
    counts: Mapping[str, Any],
    normalization_policy: str | None,
) -> None:
    if set(split) != {"train", "val"}:
        raise ValueError("registered Strict-A split schema is invalid")
    train_patients = _identifier_set(split["train"], name="train patient")
    val_patients = _identifier_set(split["val"], name="val patient")
    if train_patients.intersection(val_patients):
        raise ValueError("registered Strict-A patient split overlaps")
    split_patients = train_patients | val_patients

    if visits.empty or transitions.empty:
        raise ValueError("registered Strict-A bundle is empty")
    visit_ids = visits["visit_id"].astype(str)
    transition_ids = transitions["transition_id"].astype(str)
    exclusion_ids = exclusions["transition_id"].astype(str)
    if (
        visit_ids.duplicated().any()
        or transition_ids.duplicated().any()
        or exclusion_ids.duplicated().any()
        or set(transition_ids).intersection(exclusion_ids)
    ):
        raise ValueError("registered Strict-A record identities are invalid")
    visit_by_id = {
        str(row["visit_id"]): row for row in visits.to_dict(orient="records")
    }
    visit_patients = set(visits["patient_id"].astype(str))
    transition_patients = set(transitions["patient_id"].astype(str))
    if visit_patients != split_patients or transition_patients != split_patients:
        raise ValueError("registered Strict-A split does not cover bundle patients")
    fold_patients = {"train": train_patients, "val": val_patients}
    for row in visits.to_dict(orient="records"):
        patient = str(row["patient_id"])
        fold = str(row["fold"])
        if fold not in fold_patients or patient not in fold_patients[fold]:
            raise ValueError("registered Strict-A visit fold is inconsistent")

    expected_paths = {
        "source_dce0_path": "dce0_path",
        "source_ser_path": "ser_path",
        "source_mask_path": "mask_path",
        "source_meta_path": "meta_path",
        "target_dce0_path": "dce0_path",
        "target_mask_path": "mask_path",
        "target_meta_path": "meta_path",
    }
    allowed_transitions = {"T0->T1", "T1->T2", "T2->T3"}
    for row in transitions.to_dict(orient="records"):
        patient = str(row["patient_id"])
        fold = str(row["fold"])
        source_id = str(row["source_visit_id"])
        target_id = str(row["target_visit_id"])
        source = visit_by_id.get(source_id)
        target = visit_by_id.get(target_id)
        if (
            fold not in fold_patients
            or patient not in fold_patients[fold]
            or source is None
            or target is None
            or source_id == target_id
            or str(row["transition_type"]) not in allowed_transitions
            or int(row["delta_days"]) <= 0
        ):
            raise ValueError("registered Strict-A transition relation is invalid")
        expected_type = f"{source['visit']}->{target['visit']}"
        try:
            expected_delta_days = int(
                (
                    pd.Timestamp(str(target["visit_date"]))
                    - pd.Timestamp(str(source["visit_date"]))
                ).days
            )
        except (TypeError, ValueError, OverflowError):
            raise ValueError("registered Strict-A transition dates are invalid") from None
        if (
            str(row["transition_type"]) != expected_type
            or str(row["transition_id"]) != f"{patient}:{expected_type}"
            or int(target["visit_index"]) != int(source["visit_index"]) + 1
            or int(row["delta_days"]) != expected_delta_days
            or expected_delta_days <= 0
            or not math.isclose(
                _finite_csv_float(
                    row["source_ftv_volume_cc"], name="source FTV volume"
                ),
                _finite_csv_float(source["ftv_volume_cc"], name="visit FTV volume"),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                _finite_csv_float(
                    row["target_ftv_volume_cc"], name="target FTV volume"
                ),
                _finite_csv_float(target["ftv_volume_cc"], name="visit FTV volume"),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or str(row["clinical_text"]) != str(source["clinical_text"])
            or str(row["action_text"]) != f"treatment arm {source['trial_arm']}"
            or _strict_csv_bool(
                row["source_ftv_is_condition"], name="source FTV condition flag"
            )
            is not True
            or _strict_csv_bool(
                row["target_ftv_is_condition"], name="target FTV condition flag"
            )
            is not False
            or _strict_csv_bool(
                row["target_ftv_is_audit_only"], name="target FTV audit flag"
            )
            is not True
        ):
            raise ValueError("registered Strict-A transition semantics are inconsistent")
        for endpoint in (source, target):
            if str(endpoint["patient_id"]) != patient or str(endpoint["fold"]) != fold:
                raise ValueError("registered Strict-A transition endpoint is inconsistent")
        if (
            str(row["source_visit"]) != str(source["visit"])
            or str(row["target_visit"]) != str(target["visit"])
            or str(row["source_study_instance_uid"])
            != str(source["study_instance_uid"])
            or str(row["target_study_instance_uid"])
            != str(target["study_instance_uid"])
        ):
            raise ValueError("registered Strict-A transition endpoint is inconsistent")
        for transition_key, visit_key in expected_paths.items():
            endpoint = source if transition_key.startswith("source_") else target
            if str(row[transition_key]) != str(endpoint[visit_key]):
                raise ValueError("registered Strict-A transition path is inconsistent")

    if not isinstance(crop_plans, Mapping) or set(crop_plans) != split_patients:
        raise ValueError("registered Strict-A crop plan patients are invalid")
    plan_keys = {
        "bbox_max_zyx",
        "bbox_min_zyx",
        "center_zyx",
        "coordinate_frame",
        "crop_start_zyx",
        "input_shape_zyx",
        "output_shape_zyx",
        "t0_mask_voxel_count",
    }
    normalized_plans = {}
    for patient, value in crop_plans.items():
        if (
            not isinstance(value, Mapping)
            or set(value) != plan_keys
            or value.get("coordinate_frame") != COORDINATE_FRAME
            or value.get("output_shape_zyx") != list(OUTPUT_SHAPE_ZYX)
            or type(value.get("t0_mask_voxel_count")) is not int
            or value["t0_mask_voxel_count"] <= 0
        ):
            raise ValueError("registered Strict-A crop plan is invalid")
        parsed_geometry = {}
        for key in (
            "bbox_max_zyx",
            "bbox_min_zyx",
            "center_zyx",
            "crop_start_zyx",
            "input_shape_zyx",
        ):
            parsed_geometry[key] = _json_int_triplet(
                value.get(key), name=f"crop plan {key}"
            )
        input_shape = parsed_geometry["input_shape_zyx"]
        bbox_min = parsed_geometry["bbox_min_zyx"]
        bbox_max = parsed_geometry["bbox_max_zyx"]
        if (
            any(length <= 0 for length in input_shape)
            or any(
                low < 0 or high < low or high >= length
                for low, high, length in zip(bbox_min, bbox_max, input_shape)
            )
        ):
            raise ValueError("registered Strict-A crop plan geometry is invalid")
        normalized_plans[str(patient)] = {
            **parsed_geometry,
            "t0_mask_voxel_count": int(value["t0_mask_voxel_count"]),
        }
    t0_counts = {patient: 0 for patient in split_patients}
    for row in visits.to_dict(orient="records"):
        patient = str(row["patient_id"])
        plan = normalized_plans[patient]
        start = _json_int_triplet(
            row["crop_start_zyx_json"], name="visit crop start"
        )
        shape = _json_int_triplet(
            row["resampled_shape_zyx_json"], name="visit resampled shape"
        )
        bbox_min = _json_int_triplet(
            row["resampled_bbox_min_zyx_json"], name="visit bbox minimum"
        )
        bbox_max = _json_int_triplet(
            row["resampled_bbox_max_zyx_json"], name="visit bbox maximum"
        )
        bbox_shape = _json_int_triplet(
            row["resampled_bbox_shape_zyx_json"], name="visit bbox shape"
        )
        _json_orientation(row["orientation_lps_json"], name="visit orientation")
        try:
            before = int(row["mask_voxel_count_resampled"])
            after = int(row["mask_voxel_count_after_crop"])
        except (TypeError, ValueError, OverflowError):
            raise ValueError("registered Strict-A visit Mask counts are invalid") from None
        visit_name = str(row["visit"])
        is_t0 = _strict_csv_bool(row["is_t0"], name="visit T0 flag")
        if is_t0:
            t0_counts[patient] += 1
        if (
            str(row["visit_id"]) != f"{patient}:{visit_name}"
            or int(row["visit_index"]) != int(visit_name[1:])
            or is_t0 != (visit_name == "T0")
            or shape != plan["input_shape_zyx"]
            or start != plan["crop_start_zyx"]
            or any(
                low < 0 or high < low or high >= length
                for low, high, length in zip(bbox_min, bbox_max, shape)
            )
            or bbox_shape
            != [high - low + 1 for low, high in zip(bbox_min, bbox_max)]
            or before <= 0
            or after != before
            or not _strict_csv_bool(
                row["mask_fully_retained"], name="visit Mask retention flag"
            )
            or str(row["mask_retention_reason"]) != "fully_retained"
            or not _strict_csv_bool(row["quality_pass"], name="visit quality flag")
        ):
            raise ValueError("registered Strict-A visit crop plan is inconsistent")
        if visit_name == "T0" and (
            bbox_min != plan["bbox_min_zyx"]
            or bbox_max != plan["bbox_max_zyx"]
            or before != plan["t0_mask_voxel_count"]
            or str(row["registration_status"]) != "fixed_reference"
        ):
            raise ValueError("registered Strict-A T0 visit geometry is inconsistent")
    if any(count != 1 for count in t0_counts.values()):
        raise ValueError("registered Strict-A patients must have exactly one T0 visit")

    train_endpoint_ids = sorted(
        {
            str(row[key])
            for row in transitions.to_dict(orient="records")
            if str(row["fold"]) == "train"
            for key in ("source_visit_id", "target_visit_id")
        }
    )
    normalization_visits = list(statistics.visit_ids)
    if normalization_policy is None:
        if normalization_visits != train_endpoint_ids:
            raise ValueError("registered Strict-A normalization visits are inconsistent")
    elif normalization_policy == "frozen_base_train_v1":
        if (
            normalization_visits != sorted(normalization_visits)
            or not set(normalization_visits) < set(train_endpoint_ids)
        ):
            raise ValueError(
                "registered Strict-A frozen normalization visits are inconsistent"
            )
    else:
        raise ValueError("registered Strict-A normalization policy is invalid")

    empty_endpoints = sum(
        int(visit_by_id[str(row["source_visit_id"])]["mask_voxel_count_resampled"])
        == 0
        or int(visit_by_id[str(row["target_visit_id"])]["mask_voxel_count_resampled"])
        == 0
        for row in transitions.to_dict(orient="records")
    )
    transition_type_counts = {
        str(name): int(value)
        for name, value in transitions["transition_type"]
        .astype(str)
        .value_counts()
        .sort_index()
        .items()
    }
    recomputed = {
        "universe_transition_count": len(transitions) + len(exclusions),
        "evaluable_transition_count": count_evaluable_transitions(
            selected_count=len(transitions),
            excluded=exclusions.to_dict(orient="records"),
        ),
        "transition_count": len(transitions),
        "exclusion_count": len(exclusions),
        "patient_count": len(split_patients),
        "visit_count": len(visits),
        "fold_transition_counts": {
            fold: int((transitions["fold"].astype(str) == fold).sum())
            for fold in ("train", "val")
        },
        "fold_patient_counts": {
            "train": len(train_patients),
            "val": len(val_patients),
        },
        "transition_type_counts": transition_type_counts,
        "empty_source_or_target_mask_transitions": empty_endpoints,
    }
    if dict(counts) != recomputed:
        raise ValueError("registered Strict-A bundle counts are inconsistent")


def _read_increment_identities(path: Path, *, name: str) -> set[str]:
    try:
        values = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        raise ValueError(f"registered Strict-A increment {name} is unreadable") from None
    if (
        not values
        or values != sorted(values)
        or len(values) != len(set(values))
        or any(not value.strip() for value in values)
    ):
        raise ValueError(f"registered Strict-A increment {name} is invalid")
    return set(values)


def _validate_increment_contract(
    *,
    bundle_path: Path,
    source_contracts: Mapping[str, Any],
    increment_contract: Mapping[str, Any],
    expansion: Mapping[str, Any],
    base_data_contract: Mapping[str, Any],
    exclusions: pd.DataFrame,
) -> dict[str, Any]:
    expected_keys = {"schema", *INCREMENT_FILENAMES}
    if (
        set(increment_contract) != expected_keys
        or increment_contract.get("schema") != INCREMENT_CONTRACT_SCHEMA
    ):
        raise ValueError("registered Strict-A increment contract is invalid")
    for field, relative_path in INCREMENT_FILENAMES.items():
        path = bundle_path.parent / relative_path
        value = increment_contract[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            or not path.is_file()
            or sha256_file(path) != value
        ):
            raise ValueError("registered Strict-A increment artifact is invalid")
    if (
        increment_contract["new_patient_ids_sha256"]
        != source_contracts.get("incremental_patient_allowlist_sha256")
    ):
        raise ValueError("registered Strict-A increment allowlist is inconsistent")

    identities = {
        "patient_ids": _read_increment_identities(
            bundle_path.parent / INCREMENT_FILENAMES["new_patient_ids_sha256"],
            name="patient identities",
        ),
        "visit_ids": _read_increment_identities(
            bundle_path.parent / INCREMENT_FILENAMES["new_visit_ids_sha256"],
            name="visit identities",
        ),
        "transition_ids": _read_increment_identities(
            bundle_path.parent / INCREMENT_FILENAMES["new_transition_ids_sha256"],
            name="transition identities",
        ),
    }
    try:
        new_exclusions = pd.read_csv(
            bundle_path.parent / INCREMENT_FILENAMES["new_exclusions_sha256"]
        )
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        raise ValueError("registered Strict-A increment exclusions are unreadable") from None
    if tuple(new_exclusions.columns) != EXCLUSION_COLUMNS:
        raise ValueError("registered Strict-A increment exclusion schema is invalid")
    base_exclusion_count = base_data_contract.get("exclusion_row_count")
    if type(base_exclusion_count) is not int:
        raise ValueError("registered Strict-A increment base exclusion count is invalid")
    published_new_exclusions = exclusions.iloc[base_exclusion_count:]
    if _records_sha256(
        new_exclusions,
        identity="transition_id",
        columns=EXCLUSION_COLUMNS,
    ) != _records_sha256(
        published_new_exclusions,
        identity="transition_id",
        columns=EXCLUSION_COLUMNS,
    ):
        raise ValueError("registered Strict-A increment exclusions are inconsistent")
    summary = _read_json(bundle_path.parent / INCREMENT_FILENAMES["summary_sha256"])
    if summary != dict(expansion):
        raise ValueError("registered Strict-A increment summary is inconsistent")
    identities["exclusions"] = new_exclusions
    return identities


def _validate_expanded_relations(
    *,
    visits: pd.DataFrame,
    transitions: pd.DataFrame,
    exclusions: pd.DataFrame,
    split: Mapping[str, Any],
    crop_plans: Mapping[str, Any],
    statistics: ForegroundStatistics,
    expansion: Mapping[str, Any],
    base_cache_contract: Mapping[str, Any],
    base_data_contract: Mapping[str, Any],
    increment_identities: Mapping[str, Any],
    split_policy: str,
) -> None:
    try:
        base_train_patients = {
            visit_id.rsplit(":", 1)[0] for visit_id in statistics.visit_ids
        }
    except (AttributeError, IndexError):
        raise ValueError("registered Strict-A expansion normalization IDs are invalid") from None
    train_patients = {str(value) for value in split["train"]}
    val_patients = {str(value) for value in split["val"]}
    split_patients = train_patients | val_patients
    candidate_patients = set(increment_identities["patient_ids"])
    selected_new_patients = split_patients.intersection(candidate_patients)
    new_val_patients = selected_new_patients.intersection(val_patients)
    base_val_patients = val_patients.difference(selected_new_patients)
    actual_base_train_patients = train_patients.difference(selected_new_patients)
    if (
        not base_train_patients
        or base_train_patients != actual_base_train_patients
        or base_train_patients.intersection(val_patients)
        or not selected_new_patients
        or (
            split_policy == "frozen_base_val_new_train_v1"
            and new_val_patients
        )
        or (
            split_policy == "frozen_base_val_new_patient_holdout_v1"
            and not new_val_patients
        )
    ):
        raise ValueError("registered Strict-A expansion base train split is invalid")
    base_patients = base_train_patients | base_val_patients
    base_exclusion_count = base_data_contract.get("exclusion_row_count")
    if (
        type(base_exclusion_count) is not int
        or base_exclusion_count < 0
        or base_exclusion_count > len(exclusions)
    ):
        raise ValueError("registered Strict-A base data exclusions are invalid")
    new_transitions = transitions.loc[
        transitions["patient_id"].astype(str).isin(selected_new_patients)
    ]
    base_exclusions = exclusions.iloc[:base_exclusion_count]
    new_exclusions = exclusions.iloc[base_exclusion_count:]
    published_new_visit_ids = set(
        visits.loc[
            visits["patient_id"].astype(str).isin(selected_new_patients), "visit_id"
        ].astype(str)
    )
    if (
        not selected_new_patients <= candidate_patients
        or not set(new_exclusions["patient_id"].astype(str)) <= candidate_patients
        or set(new_transitions["transition_id"].astype(str))
        != set(increment_identities["transition_ids"])
        or published_new_visit_ids != set(increment_identities["visit_ids"])
    ):
        raise ValueError("registered Strict-A increment identities are inconsistent")
    derived_expansion = {
        "candidate_patient_count": len(candidate_patients),
        "candidate_transition_count": len(new_transitions) + len(new_exclusions),
        "excluded_transition_count": len(new_exclusions),
        "selected_patient_count": len(selected_new_patients),
        "selected_transition_count": len(new_transitions),
    }
    if dict(expansion) != derived_expansion:
        raise ValueError("registered Strict-A expansion counts are inconsistent")

    base_visits = visits.loc[visits["patient_id"].astype(str).isin(base_patients)]
    base_transitions = transitions.loc[
        transitions["patient_id"].astype(str).isin(base_patients)
    ]
    base_plans = {
        patient: crop_plans[patient] for patient in sorted(base_patients)
    }
    expected_base_data = build_base_data_contract(
        bundle_contract_sha256=str(base_data_contract.get("bundle_contract_sha256", "")),
        visits=base_visits,
        transitions=base_transitions,
        exclusions=base_exclusions,
        split={
            "train": sorted(base_train_patients),
            "val": sorted(base_val_patients),
        },
        crop_plans=base_plans,
    )
    if (
        dict(base_data_contract) != expected_base_data
        or base_data_contract.get("bundle_contract_sha256")
        != base_cache_contract.get("bundle_contract_sha256")
    ):
        raise ValueError("registered Strict-A base data contract is inconsistent")


def _load_hashed_bundle_for_repair(bundle_path: Path) -> dict[str, Any]:
    bundle = _read_json(bundle_path)
    unsigned = dict(bundle)
    digest = unsigned.pop("bundle_contract_sha256", None)
    if (
        bundle.get("schema_version") != BUNDLE_SCHEMA
        or digest != sha256_json(unsigned)
    ):
        raise ValueError("registered Strict-A repair source contract is invalid")
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        ARTIFACT_FILENAMES
    ):
        raise ValueError("registered Strict-A repair artifacts are invalid")
    for name, filename in ARTIFACT_FILENAMES.items():
        artifact = artifacts[name]
        expected_keys = {"path", "sha256"} | (
            {"row_count"} if filename.endswith(".csv") else set()
        )
        path = bundle_path.parent / filename
        if (
            not isinstance(artifact, Mapping)
            or set(artifact) != expected_keys
            or artifact.get("path") != filename
            or not path.is_file()
            or sha256_file(path) != artifact.get("sha256")
        ):
            raise ValueError("registered Strict-A repair artifact changed")
    return bundle


def repair_bundle_geometry(
    bundle_path: str | Path,
    *,
    output_dir: str | Path,
    mask_loader: Callable[[Mapping[str, Any]], np.ndarray] | None = None,
    progress_every: int = 100,
) -> dict[str, Any]:
    """Write a verified copy with registered Mask geometry recomputed."""

    if type(progress_every) is not int or progress_every <= 0:
        raise ValueError("registered Strict-A repair progress interval is invalid")
    source_bundle_path = Path(bundle_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError("registered Strict-A repair output already exists")
    if (
        output == source_bundle_path.parent
        or source_bundle_path.parent in output.parents
    ):
        raise ValueError("registered Strict-A repair output must be outside the source")
    bundle = _load_hashed_bundle_for_repair(source_bundle_path)
    visits = pd.read_csv(source_bundle_path.parent / ARTIFACT_FILENAMES["visits"])
    if tuple(visits.columns) != VISIT_COLUMNS:
        raise ValueError("registered Strict-A repair visit schema is invalid")
    try:
        with (source_bundle_path.parent / ARTIFACT_FILENAMES["visits"]).open(
            newline="", encoding="utf-8"
        ) as handle:
            reader = csv.DictReader(handle)
            raw_fieldnames = reader.fieldnames
            raw_visits = list(reader)
    except (OSError, UnicodeError, csv.Error):
        raise ValueError("registered Strict-A repair visits are unreadable") from None
    if (
        tuple(raw_fieldnames or ()) != VISIT_COLUMNS
        or len(raw_visits) != len(visits)
        or [row["visit_id"] for row in raw_visits]
        != visits["visit_id"].astype(str).tolist()
    ):
        raise ValueError("registered Strict-A repair raw visits are inconsistent")
    crop_plans = _read_json(
        source_bundle_path.parent / ARTIFACT_FILENAMES["crop_plans"]
    )
    staging = output.with_name(f".{output.name}.repairing")
    if staging.exists():
        raise FileExistsError("registered Strict-A repair staging already exists")
    shutil.copytree(source_bundle_path.parent, staging)
    try:
        for index, source in enumerate(visits.to_dict(orient="records"), start=1):
            row = dict(source)
            patient = str(row["patient_id"])
            plan_value = crop_plans.get(patient)
            if not isinstance(plan_value, Mapping):
                raise ValueError("registered Strict-A repair crop plan is missing")
            try:
                plan = CropPlan(
                    coordinate_frame=str(plan_value["coordinate_frame"]),
                    input_shape_zyx=tuple(
                        int(value) for value in plan_value["input_shape_zyx"]
                    ),
                    output_shape_zyx=tuple(
                        int(value) for value in plan_value["output_shape_zyx"]
                    ),
                    bbox_min_zyx=tuple(
                        int(value) for value in plan_value["bbox_min_zyx"]
                    ),
                    bbox_max_zyx=tuple(
                        int(value) for value in plan_value["bbox_max_zyx"]
                    ),
                    center_zyx=tuple(int(value) for value in plan_value["center_zyx"]),
                    crop_start_zyx=tuple(
                        int(value) for value in plan_value["crop_start_zyx"]
                    ),
                    t0_mask_voxel_count=int(plan_value["t0_mask_voxel_count"]),
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                raise ValueError("registered Strict-A repair crop plan is invalid") from None
            mask = (
                np.asarray(mask_loader(row))
                if mask_loader is not None
                else _load_resampled(
                    row["mask_path"],
                    row["meta_path"],
                    is_mask=True,
                    target_spacing_xyz=TARGET_SPACING_XYZ,
                )
            )
            if mask.ndim != 3 or tuple(mask.shape) != plan.input_shape_zyx:
                raise ValueError("registered Strict-A repair grid changed")
            binary = (mask > 0).astype(np.uint8)
            cropped = crop_or_pad_zyx(
                binary, plan.crop_start_zyx, plan.output_shape_zyx
            )
            before = int(binary.sum())
            after = int(cropped.sum())
            if (
                before <= 0
                or after != before
                or int(row["mask_voxel_count_resampled"]) != before
                or int(row["mask_voxel_count_after_crop"]) != after
                or _json_int_triplet(
                    row["crop_start_zyx_json"], name="repair crop start"
                )
                != list(plan.crop_start_zyx)
                or not _strict_csv_bool(
                    row["mask_fully_retained"], name="repair Mask retention"
                )
                or not _strict_csv_bool(row["quality_pass"], name="repair quality")
            ):
                raise ValueError("registered Strict-A repair would change selection semantics")
            geometry = _registered_mask_geometry(binary)
            if str(row["visit"]) == "T0" and (
                json.loads(geometry["resampled_bbox_min_zyx_json"])
                != list(plan.bbox_min_zyx)
                or json.loads(geometry["resampled_bbox_max_zyx_json"])
                != list(plan.bbox_max_zyx)
                or before != plan.t0_mask_voxel_count
            ):
                raise ValueError("registered Strict-A repair T0 plan changed")
            row.update(geometry)
            row["mask_retention_reason"] = "fully_retained"
            raw_row = raw_visits[index - 1]
            for field, value in geometry.items():
                raw_row[field] = str(value)
            raw_row["mask_retention_reason"] = "fully_retained"
            if index % progress_every == 0:
                print(
                    json.dumps(
                        {"event": "geometry_repair", "completed": index},
                        sort_keys=True,
                    ),
                    flush=True,
                )
        visits_path = staging / ARTIFACT_FILENAMES["visits"]
        with visits_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(raw_fieldnames or ()),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(raw_visits)
        repaired_visits = pd.read_csv(visits_path)
        bundle["artifacts"]["visits"] = _artifact(
            visits_path, row_count=len(repaired_visits)
        )

        if bundle.get("split_policy") in EXPANDED_SPLIT_POLICIES:
            transitions = pd.read_csv(staging / ARTIFACT_FILENAMES["transitions"])
            exclusions = pd.read_csv(staging / ARTIFACT_FILENAMES["exclusions"])
            split = _read_json(staging / ARTIFACT_FILENAMES["split"])
            normalization = _read_json(staging / ARTIFACT_FILENAMES["normalization"])
            statistics = ForegroundStatistics.from_dict(normalization)
            base_train_patients = {
                visit_id.rsplit(":", 1)[0] for visit_id in statistics.visit_ids
            }
            candidate_patients = _read_increment_identities(
                staging / INCREMENT_FILENAMES["new_patient_ids_sha256"],
                name="patient identities",
            )
            base_val_patients = {
                str(value) for value in split["val"]
            }.difference(candidate_patients)
            base_patients = base_train_patients | base_val_patients
            base_exclusion_count = int(
                bundle["base_data_contract"]["exclusion_row_count"]
            )
            if base_exclusion_count < 0:
                raise ValueError("registered Strict-A repair expansion is invalid")
            bundle["base_data_contract"] = build_base_data_contract(
                bundle_contract_sha256=str(
                    bundle["base_cache_contract"]["bundle_contract_sha256"]
                ),
                visits=repaired_visits.loc[
                    repaired_visits["patient_id"].astype(str).isin(base_patients)
                ],
                transitions=transitions.loc[
                    transitions["patient_id"].astype(str).isin(base_patients)
                ],
                exclusions=exclusions.iloc[:base_exclusion_count],
                split={
                    "train": sorted(base_train_patients),
                    "val": sorted(base_val_patients),
                },
                crop_plans={
                    patient: crop_plans[patient] for patient in sorted(base_patients)
                },
            )
            bundle["increment_contract"] = build_increment_contract(staging)

        unsigned = dict(bundle)
        unsigned.pop("bundle_contract_sha256", None)
        bundle["bundle_contract_sha256"] = sha256_json(unsigned)
        _json_dump(staging / "bundle.json", bundle)
        verified = verify_bundle(staging / "bundle.json")
        staging.replace(output)
        return verified
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_bundle(path: str | Path) -> dict[str, Any]:
    bundle_path = Path(path).expanduser().resolve()
    bundle = _read_json(bundle_path)
    if bundle.get("schema_version") != BUNDLE_SCHEMA:
        raise ValueError("registered Strict-A bundle schema is invalid")
    expected = dict(bundle)
    digest = expected.pop("bundle_contract_sha256", None)
    if digest != sha256_json(expected):
        raise ValueError("registered Strict-A bundle contract SHA is invalid")
    required_bundle_keys = {
        "schema_version",
        "bundle_contract_sha256",
        "counts",
        "artifacts",
        "source_contracts",
        "strict_a_rule_version",
        "split_policy",
        "text_policy",
    }
    optional_bundle_keys = {
        "base_cache_contract",
        "base_data_contract",
        "expansion",
        "increment_contract",
        "normalization_policy",
    }
    split_policy = bundle.get("split_policy")
    expanded = split_policy in EXPANDED_SPLIT_POLICIES
    expected_source_contract_keys = (
        EXPANDED_SOURCE_CONTRACT_KEYS if expanded else BASE_SOURCE_CONTRACT_KEYS
    )
    if (
        not required_bundle_keys <= set(bundle)
        or set(bundle).difference(required_bundle_keys | optional_bundle_keys)
        or bundle.get("strict_a_rule_version") != "v1"
        or split_policy
        not in {"patient_level_90_10_seed_2026", *EXPANDED_SPLIT_POLICIES}
        or not isinstance(bundle.get("text_policy"), Mapping)
        or not isinstance(bundle.get("source_contracts"), Mapping)
        or set(bundle["source_contracts"]) != expected_source_contract_keys
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in bundle["source_contracts"].values()
        )
    ):
        raise ValueError("registered Strict-A source contract or bundle fields are invalid")
    if expanded:
        base_cache = bundle.get("base_cache_contract")
        base_data = bundle.get("base_data_contract")
        expansion = bundle.get("expansion")
        increment = bundle.get("increment_contract")
        if (
            bundle.get("normalization_policy") != "frozen_base_train_v1"
            or not isinstance(base_cache, Mapping)
            or set(base_cache)
            != {
                "schema",
                "bundle_contract_sha256",
                "preprocess_sha256",
                "normalization_sha256",
            }
            or base_cache.get("schema") != "motfm_ispy2_t0_fixed_roi_cache_v1"
            or not isinstance(base_data, Mapping)
            or set(base_data)
            != {
                "schema",
                "bundle_contract_sha256",
                "split_sha256",
                "visits_semantic_sha256",
                "transitions_sha256",
                "exclusion_row_count",
                "exclusions_sha256",
                "crop_plans_sha256",
            }
            or base_data.get("schema") != BASE_DATA_CONTRACT_SCHEMA
            or not isinstance(increment, Mapping)
            or not isinstance(expansion, Mapping)
            or set(expansion)
            != {
                "candidate_patient_count",
                "candidate_transition_count",
                "excluded_transition_count",
                "selected_patient_count",
                "selected_transition_count",
            }
            or any(type(value) is not int or value < 0 for value in expansion.values())
            or expansion["candidate_transition_count"]
            != expansion["selected_transition_count"]
            + expansion["excluded_transition_count"]
        ):
            raise ValueError("registered Strict-A expansion contract is invalid")
        for key in (
            "bundle_contract_sha256",
            "preprocess_sha256",
            "normalization_sha256",
        ):
            value = base_cache[key]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError("registered Strict-A base cache contract is invalid")
        if (
            type(base_data.get("exclusion_row_count")) is not int
            or base_data["exclusion_row_count"] < 0
        ):
            raise ValueError("registered Strict-A base data contract is invalid")
        for key in set(base_data).difference({"schema", "exclusion_row_count"}):
            value = base_data[key]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError("registered Strict-A base data contract is invalid")
    elif set(bundle).intersection(optional_bundle_keys):
        raise ValueError("registered Strict-A base bundle has expansion fields")
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(ARTIFACT_FILENAMES):
        raise ValueError("registered Strict-A bundle artifacts are invalid")
    for name, filename in ARTIFACT_FILENAMES.items():
        artifact = artifacts[name]
        expected_keys = {"path", "sha256"} | (
            {"row_count"} if filename.endswith(".csv") else set()
        )
        if not isinstance(artifact, Mapping) or set(artifact) != expected_keys:
            raise ValueError("registered Strict-A bundle artifact fields are invalid")
        if artifact.get("path") != filename:
            raise ValueError("registered Strict-A bundle artifact path is invalid")
        artifact_path = bundle_path.parent / filename
        if not artifact_path.is_file() or sha256_file(artifact_path) != artifact["sha256"]:
            raise ValueError("registered Strict-A bundle artifact is missing or changed")
    if expanded and (
        bundle["base_cache_contract"]["preprocess_sha256"]
        != artifacts["preprocess"]["sha256"]
        or bundle["base_cache_contract"]["normalization_sha256"]
        != artifacts["normalization"]["sha256"]
    ):
        raise ValueError("registered Strict-A frozen cache artifacts changed")
    visits, transitions, exclusions = _registered_artifact_frames(
        bundle_path, artifacts
    )
    split = _read_json(bundle_path.parent / ARTIFACT_FILENAMES["split"])
    preprocess = _read_json(bundle_path.parent / ARTIFACT_FILENAMES["preprocess"])
    normalization = _read_json(
        bundle_path.parent / ARTIFACT_FILENAMES["normalization"]
    )
    crop_plans = _read_json(bundle_path.parent / ARTIFACT_FILENAMES["crop_plans"])
    _validate_registered_preprocess(
        preprocess, artifacts["normalization"]["sha256"]
    )
    try:
        statistics = ForegroundStatistics.from_dict(normalization)
    except (KeyError, TypeError, ValueError, OverflowError):
        raise ValueError("registered Strict-A normalization contract is invalid") from None
    counts = bundle.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("registered Strict-A bundle counts are invalid")
    _validate_registered_relations(
        visits=visits,
        transitions=transitions,
        exclusions=exclusions,
        split=split,
        crop_plans=crop_plans,
        statistics=statistics,
        counts=counts,
        normalization_policy=bundle.get("normalization_policy"),
    )
    if expanded:
        increment_identities = _validate_increment_contract(
            bundle_path=bundle_path,
            source_contracts=bundle["source_contracts"],
            increment_contract=bundle["increment_contract"],
            expansion=bundle["expansion"],
            base_data_contract=bundle["base_data_contract"],
            exclusions=exclusions,
        )
        _validate_expanded_relations(
            visits=visits,
            transitions=transitions,
            exclusions=exclusions,
            split=split,
            crop_plans=crop_plans,
            statistics=statistics,
            expansion=bundle["expansion"],
            base_cache_contract=bundle["base_cache_contract"],
            base_data_contract=bundle["base_data_contract"],
            increment_identities=increment_identities,
            split_policy=str(split_policy),
        )
    return bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--legacy-bundle", type=Path, default=DEFAULT_LEGACY_BUNDLE)
    build.add_argument("--registered-manifest", type=Path, default=DEFAULT_REGISTERED_MANIFEST)
    build.add_argument("--original-manifest", type=Path, default=DEFAULT_ORIGINAL_MANIFEST)
    build.add_argument("--clinical", type=Path, default=DEFAULT_CLINICAL)
    build.add_argument("--strict-rules", type=Path, default=DEFAULT_STRICT_RULES)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--progress-every", type=int, default=100)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--bundle", type=Path, required=True)
    repair = subparsers.add_parser("repair-geometry")
    repair.add_argument("--bundle", type=Path, required=True)
    repair.add_argument("--output-dir", type=Path, required=True)
    repair.add_argument("--progress-every", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        bundle = build_bundle(
            legacy_bundle=args.legacy_bundle,
            registered_manifest_path=args.registered_manifest,
            original_manifest_path=args.original_manifest,
            clinical_path=args.clinical,
            strict_rules_path=args.strict_rules,
            output_dir=args.output_dir,
            progress_every=args.progress_every,
        )
    elif args.command == "verify":
        bundle = verify_bundle(args.bundle)
    else:
        bundle = repair_bundle_geometry(
            args.bundle,
            output_dir=args.output_dir,
            progress_every=args.progress_every,
        )
    print(json.dumps({"counts": bundle["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

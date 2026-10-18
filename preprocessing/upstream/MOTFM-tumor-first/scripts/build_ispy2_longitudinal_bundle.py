#!/usr/bin/env python3
"""Build and verify the MOTFM I-SPY2 longitudinal metadata bundle."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.ispy2_preprocessing import assess_visit_preprocessing, preprocess_contract
from utils.ispy2_registry import (
    ContractError,
    build_split_and_visit_records,
    join_registry_sources,
    load_clinical_table,
    load_image_manifest,
    load_verified_registries,
    sha256_file,
    sha256_json,
)


BUNDLE_SCHEMA = "motfm_ispy2_longitudinal_bundle_v1"
ARTIFACT_FILENAMES = {
    "visits": "visits.csv",
    "transitions": "transitions.csv",
    "split": "split.json",
    "preprocess": "preprocess.json",
    "exclusions": "exclusions.csv",
}
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
    "patient_id",
    "source_visit_id",
    "target_visit_id",
    "transition_type",
    "reason",
    "detail",
)
LOCKED_REAL_COUNTS = {
    "visit_count": 2194,
    "patient_count": 600,
    "transition_count": 1196,
    "split_fold_patients": {"train": 420, "val": 90, "test": 90},
    "fold_transition_counts": {"train": 849, "val": 179, "test": 168},
    "transition_type_counts": {"T0->T1": 436, "T1->T2": 399, "T2->T3": 361},
}
TEXT_POLICY = {
    "clinical_text_fields": ["age", "HR", "HER2", "MP", "menopausal_status"],
    "action_text_template": "treatment arm {trial_arm}",
    "model_text_excludes": [
        "patient_id",
        "pCR",
        "source_ftv_volume_cc",
        "target_ftv_volume_cc",
        "target_visit",
        "delta_days",
    ],
    "pcr_use": "transient_split_balancing_only",
    "source_ftv_volume_cc": "numeric_condition_metadata",
    "target_ftv_volume_cc": "numeric_audit_only_not_condition",
}


def _orientation_matrix(value: Any) -> np.ndarray:
    try:
        parsed = json.loads(str(value)) if isinstance(value, str) else value
        matrix = np.asarray(parsed, dtype=np.float64)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("orientation_lps_json must contain a 3x3 finite matrix") from exc
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("orientation_lps_json must contain a 3x3 finite matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-4) or not np.isclose(
        np.linalg.det(matrix), 1.0, atol=1e-4
    ):
        raise ValueError("orientation_lps_json must contain a proper rotation")
    return matrix


def _relative_angle_degrees(source: Any, target: Any) -> float:
    relative = _orientation_matrix(source).T @ _orientation_matrix(target)
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _validate_text(patient_id: str, clinical_text: str, action_text: str) -> None:
    lowered = f"{clinical_text} {action_text}".lower()
    forbidden = (patient_id.lower(), "pcr", "ftv", "target", "days")
    if any(token and token in lowered for token in forbidden):
        raise ValueError("model text violates the I-SPY2 leakage policy")


def build_adjacent_transitions(
    visits: pd.DataFrame,
    *,
    max_orientation_degrees: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build only valid T0->T1, T1->T2, and T2->T3 transitions."""

    required = {
        "patient_id",
        "visit_id",
        "visit",
        "visit_index",
        "study_instance_uid",
        "visit_date",
        "fold",
        "dce0_path",
        "ser_path",
        "mask_path",
        "meta_path",
        "ftv_volume_cc",
        "clinical_text",
        "trial_arm",
        "orientation_lps_json",
        "mask_fully_retained",
    }
    missing = sorted(required.difference(visits.columns))
    if missing:
        raise ValueError(f"visits are missing transition fields: {missing}")
    if not math.isfinite(float(max_orientation_degrees)) or float(max_orientation_degrees) < 0:
        raise ValueError("max_orientation_degrees must be finite and nonnegative")

    transition_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    for group_index, (patient_id, group) in enumerate(
        visits.groupby("patient_id", sort=True),
        start=1,
    ):
        by_index = {int(row["visit_index"]): row for row in group.to_dict(orient="records")}
        if len(by_index) != len(group):
            raise ValueError(
                f"duplicate visit index in patient group {group_index}"
            )
        for source_index in range(3):
            if source_index not in by_index or source_index + 1 not in by_index:
                continue
            source = by_index[source_index]
            target = by_index[source_index + 1]
            source_visit = f"T{source_index}"
            target_visit = f"T{source_index + 1}"
            if source["visit"] != source_visit or target["visit"] != target_visit:
                raise ValueError("visit labels and indices conflict")
            if source["fold"] != target["fold"]:
                raise ValueError("patient visits cannot span split folds")
            delta_days = (pd.Timestamp(target["visit_date"]) - pd.Timestamp(source["visit_date"])).days
            angle = _relative_angle_degrees(
                source["orientation_lps_json"], target["orientation_lps_json"]
            )
            reasons: list[str] = []
            if delta_days <= 0:
                reasons.append("nonpositive_delta_days")
            if not bool(source["mask_fully_retained"]):
                reasons.append("source_mask_not_fully_retained")
            if not bool(target["mask_fully_retained"]):
                reasons.append("target_mask_not_fully_retained")
            if angle > float(max_orientation_degrees) + 1e-9:
                reasons.append("orientation_angle_exceeds_limit")
            transition_type = f"{source_visit}->{target_visit}"
            if reasons:
                exclusion_rows.append(
                    {
                        "patient_id": patient_id,
                        "source_visit_id": source["visit_id"],
                        "target_visit_id": target["visit_id"],
                        "transition_type": transition_type,
                        "reason": "+".join(reasons),
                        "detail": json.dumps(
                            {
                                "delta_days": int(delta_days),
                                "orientation_angle_degrees": float(angle),
                                "source_mask_fully_retained": bool(
                                    source["mask_fully_retained"]
                                ),
                                "target_mask_fully_retained": bool(
                                    target["mask_fully_retained"]
                                ),
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    }
                )
                continue
            clinical_text = str(source["clinical_text"])
            action_text = f"treatment arm {source['trial_arm']}"
            _validate_text(str(patient_id), clinical_text, action_text)
            transition_rows.append(
                {
                    "transition_id": f"{patient_id}:{transition_type}",
                    "patient_id": patient_id,
                    "fold": source["fold"],
                    "transition_type": transition_type,
                    "source_visit_id": source["visit_id"],
                    "target_visit_id": target["visit_id"],
                    "source_visit": source_visit,
                    "target_visit": target_visit,
                    "source_study_instance_uid": source["study_instance_uid"],
                    "target_study_instance_uid": target["study_instance_uid"],
                    "delta_days": int(delta_days),
                    "orientation_angle_degrees": float(angle),
                    "source_dce0_path": source["dce0_path"],
                    "source_ser_path": source["ser_path"],
                    "source_mask_path": source["mask_path"],
                    "source_meta_path": source["meta_path"],
                    "target_dce0_path": target["dce0_path"],
                    "target_mask_path": target["mask_path"],
                    "target_meta_path": target["meta_path"],
                    "source_ftv_volume_cc": float(source["ftv_volume_cc"]),
                    "target_ftv_volume_cc": float(target["ftv_volume_cc"]),
                    "source_ftv_is_condition": True,
                    "target_ftv_is_condition": False,
                    "target_ftv_is_audit_only": True,
                    "clinical_text": clinical_text,
                    "action_text": action_text,
                }
            )
    return (
        pd.DataFrame(transition_rows, columns=TRANSITION_COLUMNS),
        pd.DataFrame(exclusion_rows, columns=EXCLUSION_COLUMNS),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            _json_safe(payload),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g", na_rep="")


def _bundle_counts(
    visits: pd.DataFrame,
    transitions: pd.DataFrame,
    exclusions: pd.DataFrame,
    split: Mapping[str, Any],
) -> dict[str, Any]:
    folds = ("train", "val", "test")
    transition_types = ("T0->T1", "T1->T2", "T2->T3")
    return {
        "visit_count": int(len(visits)),
        "patient_count": int(visits["patient_id"].nunique()),
        "transition_count": int(len(transitions)),
        "exclusion_count": int(len(exclusions)),
        "split_fold_patients": {fold: len(split[fold]) for fold in folds},
        "fold_transition_counts": {
            fold: int((transitions["fold"] == fold).sum()) for fold in folds
        },
        "transition_type_counts": {
            transition_type: int(
                (transitions["transition_type"] == transition_type).sum()
            )
            for transition_type in transition_types
        },
    }


def _validate_published_records(visits: pd.DataFrame, transitions: pd.DataFrame) -> None:
    if any(str(column).lower() == "pcr" for column in visits.columns):
        raise ValueError("pCR must not be published in visit records")
    for row in visits.to_dict(orient="records"):
        _validate_text(str(row["patient_id"]), str(row["clinical_text"]), "")
    for row in transitions.to_dict(orient="records"):
        expected_action = f"treatment arm {visits.loc[visits['visit_id'] == row['source_visit_id'], 'trial_arm'].item()}"
        if row["action_text"] != expected_action:
            raise ValueError("action_text must contain only the treatment arm")
        _validate_text(str(row["patient_id"]), str(row["clinical_text"]), str(row["action_text"]))


def _validate_bundle_relations(
    visits: pd.DataFrame,
    transitions: pd.DataFrame,
    split: Mapping[str, Any],
) -> None:
    folds = ("train", "val", "test")
    required_visits = {
        "patient_id",
        "visit_id",
        "visit",
        "visit_index",
        "study_instance_uid",
        "visit_date",
        "fold",
        "dce0_path",
        "ser_path",
        "mask_path",
        "meta_path",
        "ftv_volume_cc",
        "clinical_text",
        "trial_arm",
    }
    missing_visits = sorted(required_visits.difference(visits.columns))
    if missing_visits:
        raise ContractError(f"visits are missing relational fields: {missing_visits}")
    missing_transitions = sorted(set(TRANSITION_COLUMNS).difference(transitions.columns))
    if missing_transitions:
        raise ContractError(
            f"transitions are missing relational fields: {missing_transitions}"
        )
    if visits["visit_id"].duplicated().any():
        raise ContractError("visits contain duplicate visit_id values")
    if transitions["transition_id"].duplicated().any():
        raise ContractError("transitions contain duplicate transition_id values")

    split_lists: dict[str, list[str]] = {}
    for fold in folds:
        patients = split.get(fold)
        if not isinstance(patients, list) or any(not isinstance(item, str) for item in patients):
            raise ContractError(f"split {fold} must be a list of patient IDs")
        split_lists[fold] = patients
    split_patients = [patient for fold in folds for patient in split_lists[fold]]
    if len(split_patients) != len(set(split_patients)):
        raise ContractError("split is not patient-disjoint")
    visit_patients = set(visits["patient_id"].astype(str))
    if set(split_patients) != visit_patients:
        raise ContractError("split patients do not exactly cover visit patients")
    fold_by_patient = {
        patient: fold for fold in folds for patient in split_lists[fold]
    }
    for row in visits.to_dict(orient="records"):
        patient_id = str(row["patient_id"])
        if str(row["fold"]) != fold_by_patient[patient_id]:
            raise ContractError("visit fold does not agree with patient split")

    visits_by_id = {
        str(row["visit_id"]): row for row in visits.to_dict(orient="records")
    }
    allowed_pairs = {("T0", "T1"), ("T1", "T2"), ("T2", "T3")}
    for row in transitions.to_dict(orient="records"):
        source_id = str(row["source_visit_id"])
        target_id = str(row["target_visit_id"])
        if source_id not in visits_by_id:
            raise ContractError("transition source_visit_id does not exist in visits")
        if target_id not in visits_by_id:
            raise ContractError("transition target_visit_id does not exist in visits")
        source = visits_by_id[source_id]
        target = visits_by_id[target_id]
        patient_id = str(row["patient_id"])
        if patient_id != str(source["patient_id"]) or patient_id != str(target["patient_id"]):
            raise ContractError("transition patient does not match source and target visits")
        if str(row["fold"]) != str(source["fold"]) or str(row["fold"]) != str(target["fold"]):
            raise ContractError("transition fold does not match source and target visits")
        source_visit = str(source["visit"])
        target_visit = str(target["visit"])
        if (source_visit, target_visit) not in allowed_pairs:
            raise ContractError("transition source/target stages are not adjacent and allowed")
        expected_type = f"{source_visit}->{target_visit}"
        if (
            str(row["source_visit"]) != source_visit
            or str(row["target_visit"]) != target_visit
            or str(row["transition_type"]) != expected_type
            or int(target["visit_index"]) != int(source["visit_index"]) + 1
        ):
            raise ContractError("transition type/stage does not match source and target visits")
        if str(row["transition_id"]) != f"{patient_id}:{expected_type}":
            raise ContractError("transition_id does not match patient and transition type")
        if (
            str(row["source_study_instance_uid"])
            != str(source["study_instance_uid"])
            or str(row["target_study_instance_uid"])
            != str(target["study_instance_uid"])
        ):
            raise ContractError("transition study UIDs do not match source and target visits")
        path_relations = (
            ("source_dce0_path", source, "dce0_path"),
            ("source_ser_path", source, "ser_path"),
            ("source_mask_path", source, "mask_path"),
            ("source_meta_path", source, "meta_path"),
            ("target_dce0_path", target, "dce0_path"),
            ("target_mask_path", target, "mask_path"),
            ("target_meta_path", target, "meta_path"),
        )
        if any(str(row[field]) != str(visit_row[visit_field]) for field, visit_row, visit_field in path_relations):
            raise ContractError("transition paths do not match source and target visits")
        if not math.isclose(
            float(row["source_ftv_volume_cc"]),
            float(source["ftv_volume_cc"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(row["target_ftv_volume_cc"]),
            float(target["ftv_volume_cc"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ContractError("transition FTV metadata does not match source and target visits")
        expected_days = (
            pd.Timestamp(target["visit_date"]) - pd.Timestamp(source["visit_date"])
        ).days
        if expected_days <= 0 or int(row["delta_days"]) != expected_days:
            raise ContractError("transition delta_days does not match positive visit dates")
        if str(row["clinical_text"]) != str(source["clinical_text"]):
            raise ContractError("transition clinical_text does not match source visit")
        if str(row["action_text"]) != f"treatment arm {source['trial_arm']}":
            raise ContractError("transition action_text does not match source treatment arm")


def _require_locked_real_counts(counts: Mapping[str, Any]) -> None:
    for key in (
        "transition_count",
        "fold_transition_counts",
        "transition_type_counts",
        "visit_count",
        "patient_count",
        "split_fold_patients",
    ):
        expected = LOCKED_REAL_COUNTS[key]
        if counts.get(key) != expected:
            raise ContractError(
                f"locked real-data {key} mismatch: expected {expected}, got {counts.get(key)}"
            )


def publish_bundle(
    output_dir: str | Path,
    *,
    visits: pd.DataFrame,
    transitions: pd.DataFrame,
    split: Mapping[str, Any],
    exclusions: pd.DataFrame,
    source_contracts: Mapping[str, Any],
    enforce_locked_counts: bool = True,
) -> dict[str, Any]:
    """Publish all six artifacts as one atomic, hash-bound directory."""

    _validate_published_records(visits, transitions)
    _validate_bundle_relations(visits, transitions, split)
    counts = _bundle_counts(visits, transitions, exclusions, split)
    if enforce_locked_counts:
        _require_locked_real_counts(counts)
    destination = Path(output_dir).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"refusing to overwrite non-empty bundle: {destination}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    published = False
    try:
        preprocess = preprocess_contract()
        _write_csv(staging / ARTIFACT_FILENAMES["visits"], visits)
        _write_csv(staging / ARTIFACT_FILENAMES["transitions"], transitions)
        _write_json(staging / ARTIFACT_FILENAMES["split"], split)
        _write_json(staging / ARTIFACT_FILENAMES["preprocess"], preprocess)
        _write_csv(staging / ARTIFACT_FILENAMES["exclusions"], exclusions)
        artifacts: dict[str, Any] = {}
        frames = {"visits": visits, "transitions": transitions, "exclusions": exclusions}
        for name, filename in ARTIFACT_FILENAMES.items():
            entry: dict[str, Any] = {
                "path": filename,
                "sha256": sha256_file(staging / filename),
            }
            if name in frames:
                entry["row_count"] = len(frames[name])
            artifacts[name] = entry
        bundle: dict[str, Any] = {
            "schema_version": BUNDLE_SCHEMA,
            "artifacts": artifacts,
            "source_contracts": dict(source_contracts),
            "counts": counts,
            "text_policy": TEXT_POLICY,
        }
        bundle["bundle_contract_sha256"] = sha256_json(bundle)
        _write_json(staging / "bundle.json", bundle)
        verify_bundle(staging / "bundle.json")
        if destination.exists():
            destination.rmdir()
        os.replace(staging, destination)
        published = True
        return verify_bundle(destination / "bundle.json")
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def verify_bundle(
    bundle_json: str | Path,
    *,
    locked_real_counts: bool = False,
) -> dict[str, Any]:
    """Verify every artifact hash/count and optionally enforce the locked cohort."""

    path = Path(bundle_json).resolve()
    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"bundle is not readable JSON: {path}") from exc
    if not isinstance(bundle, dict) or bundle.get("schema_version") != BUNDLE_SCHEMA:
        raise ContractError("unsupported bundle schema")
    expected_self = sha256_json(
        {key: value for key, value in bundle.items() if key != "bundle_contract_sha256"}
    )
    if bundle.get("bundle_contract_sha256") != expected_self:
        raise ContractError("bundle contract SHA256 mismatch")
    root = path.parent
    for name, filename in ARTIFACT_FILENAMES.items():
        artifact = bundle.get("artifacts", {}).get(name, {})
        if artifact.get("path") != filename:
            raise ContractError(f"{name} artifact path mismatch")
        artifact_path = root / filename
        if artifact.get("sha256") != sha256_file(artifact_path):
            raise ContractError(f"{name} artifact SHA256 mismatch")

    visits = pd.read_csv(root / ARTIFACT_FILENAMES["visits"])
    transitions = pd.read_csv(root / ARTIFACT_FILENAMES["transitions"])
    exclusions = pd.read_csv(root / ARTIFACT_FILENAMES["exclusions"])
    split = json.loads((root / ARTIFACT_FILENAMES["split"]).read_text(encoding="utf-8"))
    for name, frame in (
        ("visits", visits),
        ("transitions", transitions),
        ("exclusions", exclusions),
    ):
        if bundle["artifacts"][name].get("row_count") != len(frame):
            raise ContractError(f"{name} artifact row count mismatch")
    _validate_bundle_relations(visits, transitions, split)
    counts = _bundle_counts(visits, transitions, exclusions, split)
    if counts != bundle.get("counts"):
        raise ContractError("bundle counts mismatch")
    _validate_published_records(visits, transitions)
    if locked_real_counts:
        _require_locked_real_counts(counts)
    return bundle


def build_source_contracts(
    contracts: Mapping[str, Mapping[str, Any]],
    *,
    manifest_csv: str | Path,
    clinical_path: str | Path,
) -> dict[str, str]:
    """Bind every immutable upstream registry and source table to the bundle."""

    return {
        "manifest_sha256": sha256_file(manifest_csv),
        "clinical_sha256": sha256_file(clinical_path),
        "date_registry_sha256": str(contracts["date"]["date_registry_sha256"]),
        "volume_contract_sha256": str(
            contracts["volume"]["volume_contract_sha256"]
        ),
        "qc_visits_contract_sha256": str(
            contracts["qc"]["qc_visits_contract_sha256"]
        ),
    }


def build_longitudinal_bundle(
    *,
    date_contract_dir: str | Path,
    volume_contract_dir: str | Path,
    qc_contract_dir: str | Path,
    manifest_csv: str | Path,
    clinical_path: str | Path,
    processed_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    registries = load_verified_registries(
        date_contract_dir, volume_contract_dir, qc_contract_dir, manifest_csv
    )
    images = load_image_manifest(
        manifest_csv, processed_root=processed_root, require_files=True
    )
    joined = join_registry_sources(
        registries.date, registries.volume, registries.qc, images
    )
    clinical = load_clinical_table(clinical_path)
    records, split = build_split_and_visit_records(joined, clinical, seed=2026)
    assessed = assess_visit_preprocessing(records)
    transitions, exclusions = build_adjacent_transitions(assessed)
    source_contracts = build_source_contracts(
        registries.contracts,
        manifest_csv=manifest_csv,
        clinical_path=clinical_path,
    )
    return publish_bundle(
        output_dir,
        visits=assessed,
        transitions=transitions,
        split=split,
        exclusions=exclusions,
        source_contracts=source_contracts,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build a verified bundle")
    build.add_argument("--date-contract-dir", type=Path, required=True)
    build.add_argument("--volume-contract-dir", type=Path, required=True)
    build.add_argument("--qc-contract-dir", type=Path, required=True)
    build.add_argument("--manifest-csv", type=Path, required=True)
    build.add_argument("--clinical", type=Path, required=True)
    build.add_argument("--processed-root", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="verify an existing bundle")
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--locked-real-counts", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        bundle = build_longitudinal_bundle(
            date_contract_dir=args.date_contract_dir,
            volume_contract_dir=args.volume_contract_dir,
            qc_contract_dir=args.qc_contract_dir,
            manifest_csv=args.manifest_csv,
            clinical_path=args.clinical,
            processed_root=args.processed_root,
            output_dir=args.output_dir,
        )
    else:
        bundle = verify_bundle(
            args.bundle, locked_real_counts=bool(args.locked_real_counts)
        )
    print(json.dumps({"counts": bundle["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

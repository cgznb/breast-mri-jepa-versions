#!/usr/bin/env python3
"""Stage an append-only expansion of the registered Strict-A bundle."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from scripts.build_ispy2_registered_strict_a_bundle import (
    EXPANDED_SPLIT_POLICIES,
    _assess_registered_visits,
    _artifact,
    build_base_data_contract,
    build_increment_contract,
    _json_dump,
    _registered_rows,
    require_source_files,
    strict_a_transition_reasons,
    verify_bundle,
)
from utils.ispy2_preprocessing import (
    assess_visit_preprocessing,
    geometry_from_meta,
    relative_orientation_angle_degrees,
)
from utils.ispy2_registered import (
    CACHE_SCHEMA,
    OUTPUT_SHAPE_ZYX,
    TARGET_SPACING_XYZ,
    CropPlan,
    reconstruct_transition_universe,
    sha256_file,
    sha256_json,
)
from utils.ispy2_registry import (
    build_split_and_visit_records,
    load_clinical_table,
    load_image_manifest,
)


PATIENT_ID_PATTERN = re.compile(r"ISPY2-\d+")
EMPTY_FTV_WARNINGS = {
    "ftv_mask is empty after applying analysis_mask_raw == 0",
    "ftv_mask_is_empty",
}


def load_patient_allowlist(
    path: Path, *, expected_count: int | None = 58
) -> tuple[str, ...]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values = [value for value in values if value]
    if not values or len(set(values)) != len(values):
        raise ValueError("incremental patient allowlist must contain unique IDs")
    if expected_count is not None and len(values) != expected_count:
        raise ValueError(
            f"incremental patient allowlist must contain {expected_count} unique IDs"
        )
    if any(PATIENT_ID_PATTERN.fullmatch(value) is None for value in values):
        raise ValueError("incremental patient allowlist contains an invalid ID")
    return tuple(sorted(values))


def study_dates_from_tcia_metadata(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"Subject ID", "Study UID", "Study Date"}
    if not required.issubset(frame.columns):
        raise ValueError("TCIA metadata is missing Study UID date fields")
    for field in required:
        frame[field] = frame[field].astype(str).str.strip()
    output: dict[str, str] = {}
    for study_uid, group in frame.groupby("Study UID", sort=True):
        try:
            dates = {
                datetime.strptime(value, "%m-%d-%Y").date().isoformat()
                for value in group["Study Date"]
                if value
            }
        except ValueError:
            raise ValueError(
                f"TCIA Study UID date is missing or conflicting: {study_uid}"
            ) from None
        if not study_uid or len(dates) != 1:
            raise ValueError(
                f"TCIA Study UID date is missing or conflicting: {study_uid}"
            )
        output[str(study_uid)] = next(iter(dates))
    return output


def reconstruct_new_visit_records(
    *,
    original_manifest_path: Path,
    metadata_path: Path,
    clinical_path: Path,
    patient_ids: tuple[str, ...],
    progress_every: int = 25,
) -> pd.DataFrame:
    requested = {str(value) for value in patient_ids}
    if not requested or "" in requested or len(requested) != len(patient_ids):
        raise ValueError("incremental patient identities must be nonempty and unique")
    source_manifest = pd.read_csv(
        original_manifest_path, dtype=str, keep_default_na=False
    )
    required = {"patient_id", "visit", "ftv_voxel_count", "qc_status"}
    if not required.issubset(source_manifest.columns):
        raise ValueError("incremental source manifest fields are incomplete")
    source_manifest = source_manifest.loc[
        source_manifest["patient_id"].astype(str).isin(requested)
    ].copy()
    source_by_visit = {
        (str(row["patient_id"]), str(row["visit"])): row
        for row in source_manifest.to_dict(orient="records")
    }
    if len(source_by_visit) != len(source_manifest):
        raise ValueError("incremental source manifest visit identities are duplicated")

    images = load_image_manifest(original_manifest_path, require_files=True)
    images = images.loc[images["patient_id"].astype(str).isin(requested)].copy()
    if set(images["patient_id"].astype(str)) != requested:
        raise ValueError("incremental image manifest patient coverage is incomplete")
    dates = study_dates_from_tcia_metadata(metadata_path)
    rows: list[dict[str, Any]] = []
    for image in images.to_dict(orient="records"):
        key = (str(image["patient_id"]), str(image["visit"]))
        if key not in source_by_visit:
            raise ValueError(f"incremental source visit is missing: {key}")
        source = source_by_visit[key]
        try:
            meta = json.loads(Path(str(image["meta_path"])).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ValueError(
                f"incremental visit meta is unreadable: {image['visit_id']}"
            ) from None
        if not isinstance(meta, dict):
            raise ValueError(f"incremental visit meta is invalid: {image['visit_id']}")
        geometry = geometry_from_meta(meta)
        study_uid = str(image["study_instance_uid"])
        if study_uid not in dates:
            raise ValueError(f"incremental Study UID has no TCIA date: {study_uid}")
        try:
            voxel_count = int(source["ftv_voxel_count"])
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f"incremental FTV voxel count is invalid: {image['visit_id']}"
            ) from None
        qc_status = str(source["qc_status"])
        qc_warning = str(source.get("qc_warnings", "")).strip()
        positive_ftv = voxel_count > 0 and qc_status == "ok"
        auditable_empty_ftv = (
            voxel_count == 0
            and qc_status == "warning"
            and qc_warning in EMPTY_FTV_WARNINGS
        )
        if voxel_count < 0 or not (positive_ftv or auditable_empty_ftv):
            raise ValueError(f"incremental visit is not FTV/QC eligible: {image['visit_id']}")
        column_spacing, row_spacing, slice_spacing = geometry.spacing_xyz
        voxel_volume = row_spacing * column_spacing * slice_spacing
        rows.append(
            {
                "patient_id": str(image["patient_id"]),
                "visit_id": str(image["visit_id"]),
                "visit": str(image["visit"]),
                "visit_index": int(image["visit_index"]),
                "study_instance_uid": study_uid,
                "visit_date": dates[study_uid],
                "visit_date_source": "tcia_metadata_study_uid",
                "qc_status": qc_status,
                "dce0_path": str(image["dce0_path"]),
                "ser_path": str(image["ser_path"]),
                "mask_path": str(image["mask_path"]),
                "meta_path": str(image["meta_path"]),
                "ftv_voxel_count": voxel_count,
                "row_spacing_mm": row_spacing,
                "column_spacing_mm": column_spacing,
                "slice_spacing_mm": slice_spacing,
                "ftv_voxel_volume_mm3": voxel_volume,
                "ftv_volume_cc": voxel_count * voxel_volume / 1000.0,
            }
        )
    records, _ = build_split_and_visit_records(
        pd.DataFrame(rows), load_clinical_table(clinical_path), seed=2026
    )
    records["fold"] = "train"
    return assess_visit_preprocessing(records, progress_every=progress_every)


def adjacent_transition_universe(visits: pd.DataFrame) -> list[dict[str, Any]]:
    skeletons: list[dict[str, Any]] = []
    for patient_id, group in visits.groupby("patient_id", sort=True):
        records = group.to_dict(orient="records")
        by_index = {int(row["visit_index"]): row for row in records}
        if len(by_index) != len(records):
            raise ValueError(f"incremental patient has duplicate visit indices: {patient_id}")
        for source_index in range(3):
            if source_index not in by_index or source_index + 1 not in by_index:
                continue
            source = by_index[source_index]
            target = by_index[source_index + 1]
            transition_type = f"T{source_index}->T{source_index + 1}"
            delta_days = int(
                (
                    pd.Timestamp(str(target["visit_date"]))
                    - pd.Timestamp(str(source["visit_date"]))
                ).days
            )
            skeletons.append(
                {
                    "patient_id": str(patient_id),
                    "source_visit_id": str(source["visit_id"]),
                    "target_visit_id": str(target["visit_id"]),
                    "transition_type": transition_type,
                    "detail": json.dumps(
                        {"delta_days": delta_days}, separators=(",", ":")
                    ),
                }
            )
    transitions = reconstruct_transition_universe(
        visits,
        pd.DataFrame(columns=["transition_id"]),
        pd.DataFrame(skeletons),
    )
    by_visit = {
        str(row["visit_id"]): row for row in visits.to_dict(orient="records")
    }
    for transition in transitions:
        source = by_visit[str(transition["source_visit_id"])]
        target = by_visit[str(transition["target_visit_id"])]
        try:
            source_meta = json.loads(
                Path(str(source["meta_path"])).read_text(encoding="utf-8")
            )
            target_meta = json.loads(
                Path(str(target["meta_path"])).read_text(encoding="utf-8")
            )
        except (KeyError, OSError, UnicodeError, json.JSONDecodeError):
            raise ValueError(
                f"incremental transition orientation metadata is unreadable: "
                f"{transition['transition_id']}"
            ) from None
        transition["orientation_angle_degrees"] = relative_orientation_angle_degrees(
            source_meta, target_meta
        )
    return transitions


def validate_real_incremental_cohort(
    *,
    patient_ids: Sequence[str],
    visits: pd.DataFrame,
    transitions: Sequence[Mapping[str, Any]],
) -> None:
    if len(patient_ids) == 58 and (len(visits), len(transitions)) != (218, 155):
        raise ValueError(
            "incremental real cohort must contain 218 visits and 155 adjacent transitions"
        )


def _write_lines(path: Path, values: Sequence[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def _read_sorted_lines(path: Path) -> tuple[str, ...]:
    values = tuple(path.read_text(encoding="utf-8").splitlines())
    if (
        not values
        or values != tuple(sorted(values))
        or len(values) != len(set(values))
    ):
        raise ValueError(f"incremental identity list is invalid: {path}")
    return values


def stage_expanded_bundle(
    *,
    base_bundle_path: Path,
    assessed_new_visits: pd.DataFrame,
    new_transition_universe: Sequence[Mapping[str, Any]],
    new_crop_plans: Mapping[str, CropPlan],
    patient_ids: Sequence[str],
    source_contracts: Mapping[str, str],
    output_dir: Path,
) -> dict[str, Any]:
    base_bundle_path = base_bundle_path.expanduser().resolve()
    base = verify_bundle(base_bundle_path)
    base_root = base_bundle_path.parent
    flatten_existing_expansion = base.get("split_policy") in EXPANDED_SPLIT_POLICIES
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("expanded registered bundle output already exists")
    staging = output_dir.with_name(f".{output_dir.name}.building")
    if staging.exists():
        raise FileExistsError("expanded registered bundle staging already exists")

    requested_patients = {str(value) for value in patient_ids}
    assessed_patients = set(assessed_new_visits["patient_id"].astype(str))
    if not requested_patients or assessed_patients != requested_patients:
        raise ValueError("incremental assessed patient coverage is invalid")
    by_visit = {
        str(row["visit_id"]): row
        for row in assessed_new_visits.to_dict(orient="records")
    }
    if len(by_visit) != len(assessed_new_visits):
        raise ValueError("incremental assessed visit identities are duplicated")

    selected_new: list[dict[str, Any]] = []
    excluded_new: list[dict[str, Any]] = []
    conclusions: set[str] = set()
    for source_transition in new_transition_universe:
        transition = dict(source_transition)
        transition_id = str(transition["transition_id"])
        if transition_id in conclusions:
            raise ValueError("incremental transition identities are duplicated")
        patient_id = str(transition["patient_id"])
        reasons = strict_a_transition_reasons(
            transition,
            by_visit,
            patient_has_t0=patient_id in new_crop_plans,
        )
        source = by_visit[str(transition["source_visit_id"])]
        target = by_visit[str(transition["target_visit_id"])]
        if (
            int(source["mask_voxel_count_resampled"]) <= 0
            or int(target["mask_voxel_count_resampled"]) <= 0
        ):
            reasons.append("empty_source_or_target_mask")
        conclusions.add(transition_id)
        if reasons:
            excluded_new.append(
                {
                    "transition_id": transition_id,
                    "patient_id": patient_id,
                    "source_visit_id": str(transition["source_visit_id"]),
                    "target_visit_id": str(transition["target_visit_id"]),
                    "transition_type": str(transition["transition_type"]),
                    "reason": "+".join(reasons),
                    "detail": json.dumps({"reasons": reasons}, separators=(",", ":")),
                }
            )
            continue
        transition.update(
            {
                "fold": "train",
                "source_dce0_path": source["dce0_path"],
                "source_ser_path": source["ser_path"],
                "source_mask_path": source["mask_path"],
                "source_meta_path": source["meta_path"],
                "target_dce0_path": target["dce0_path"],
                "target_mask_path": target["mask_path"],
                "target_meta_path": target["meta_path"],
            }
        )
        selected_new.append(transition)
    if len(conclusions) != len(new_transition_universe):
        raise RuntimeError("incremental transition conclusions are incomplete")

    base_visits_path = base_root / base["artifacts"]["visits"]["path"]
    base_transitions_path = base_root / base["artifacts"]["transitions"]["path"]
    base_exclusions_path = base_root / base["artifacts"]["exclusions"]["path"]
    base_visits = pd.read_csv(base_visits_path)
    base_transitions = pd.read_csv(base_transitions_path)
    base_exclusions = pd.read_csv(base_exclusions_path)
    prior_candidate_ids: set[str] = set()
    prior_visit_ids: set[str] = set()
    prior_transition_ids: set[str] = set()
    prior_new_exclusions = pd.DataFrame(columns=base_exclusions.columns)
    if flatten_existing_expansion:
        prior_increment = base_root / "increment"
        prior_candidate_ids = set(
            _read_sorted_lines(prior_increment / "new_patient_ids.txt")
        )
        prior_visit_ids = set(
            _read_sorted_lines(prior_increment / "new_visit_ids.txt")
        )
        prior_transition_ids = set(
            _read_sorted_lines(prior_increment / "new_transition_ids.txt")
        )
        prior_new_exclusions = pd.read_csv(prior_increment / "new_exclusions.csv")
        if tuple(prior_new_exclusions.columns) != tuple(base_exclusions.columns):
            raise ValueError(
                "prior incremental exclusion fields differ from the base schema"
            )
        if prior_candidate_ids.intersection(requested_patients):
            raise ValueError("incremental patients overlap the prior increment")
    base_ids = set(base_visits["patient_id"].astype(str))
    if base_ids.intersection(requested_patients):
        raise ValueError("incremental patients overlap the frozen base bundle")

    transition_columns = list(base_transitions.columns)
    if any(set(row) != set(transition_columns) for row in selected_new):
        raise ValueError("incremental selected transition fields differ from the base schema")
    eligible_patients = {str(row["patient_id"]) for row in selected_new}
    referenced_ids = {
        str(row[key])
        for row in selected_new
        for key in ("source_visit_id", "target_visit_id")
    }
    referenced_ids.update(f"{patient_id}:T0" for patient_id in eligible_patients)
    published_new = assessed_new_visits.loc[
        assessed_new_visits["visit_id"].astype(str).isin(referenced_ids)
    ].copy()
    published_new["fold"] = "train"
    published_new = published_new.sort_values(["patient_id", "visit_index"])
    if set(published_new["visit_id"].astype(str)) != referenced_ids:
        raise ValueError("incremental published visit coverage is incomplete")
    visit_columns = list(base_visits.columns)
    if set(published_new.columns) != set(visit_columns):
        raise ValueError("incremental visit fields differ from the base schema")
    published_new = published_new[visit_columns]
    selected_frame = pd.DataFrame(selected_new, columns=transition_columns)
    all_visits = pd.concat([base_visits, published_new], ignore_index=True)
    all_transitions = pd.concat([base_transitions, selected_frame], ignore_index=True)
    if excluded_new:
        if set(excluded_new[0]) != set(base_exclusions.columns):
            raise ValueError("incremental exclusion fields differ from the base schema")
        new_exclusion_frame = pd.DataFrame(excluded_new, columns=base_exclusions.columns)
    else:
        new_exclusion_frame = pd.DataFrame(columns=base_exclusions.columns)
    all_exclusions = pd.concat(
        [base_exclusions, new_exclusion_frame], ignore_index=True
    )
    cumulative_candidate_ids = prior_candidate_ids | requested_patients
    cumulative_visit_ids = prior_visit_ids | referenced_ids
    cumulative_transition_ids = prior_transition_ids | {
        str(row["transition_id"]) for row in selected_new
    }
    cumulative_new_exclusions = pd.concat(
        [prior_new_exclusions, new_exclusion_frame], ignore_index=True
    )

    base_split = json.loads(
        (base_root / base["artifacts"]["split"]["path"]).read_text(encoding="utf-8")
    )
    split = {
        "train": sorted([str(value) for value in base_split["train"]] + list(eligible_patients)),
        "val": [str(value) for value in base_split["val"]],
    }
    base_plans = json.loads(
        (base_root / base["artifacts"]["crop_plans"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    merged_plans = dict(base_plans)
    for patient_id in sorted(eligible_patients):
        merged_plans[patient_id] = new_crop_plans[patient_id].to_dict()

    staging.mkdir(parents=True)
    try:
        visits_path = staging / "visits.csv"
        transitions_path = staging / "transitions.csv"
        exclusions_path = staging / "exclusions.csv"
        split_path = staging / "split.json"
        preprocess_path = staging / "preprocess.json"
        normalization_path = staging / "normalization.json"
        crop_path = staging / "crop_plans.json"
        all_visits.to_csv(visits_path, index=False)
        all_transitions.to_csv(transitions_path, index=False)
        all_exclusions.to_csv(exclusions_path, index=False)
        _json_dump(split_path, split)
        shutil.copyfile(
            base_root / base["artifacts"]["preprocess"]["path"], preprocess_path
        )
        shutil.copyfile(
            base_root / base["artifacts"]["normalization"]["path"],
            normalization_path,
        )
        _json_dump(crop_path, merged_plans)
        artifacts = {
            "visits": _artifact(visits_path, row_count=len(all_visits)),
            "transitions": _artifact(transitions_path, row_count=len(all_transitions)),
            "exclusions": _artifact(exclusions_path, row_count=len(all_exclusions)),
            "split": _artifact(split_path),
            "preprocess": _artifact(preprocess_path),
            "normalization": _artifact(normalization_path),
            "crop_plans": _artifact(crop_path),
        }
        transition_types = Counter(all_transitions["transition_type"].astype(str))
        counts = {
            "universe_transition_count": int(base["counts"]["universe_transition_count"])
            + len(new_transition_universe),
            "evaluable_transition_count": int(base["counts"]["evaluable_transition_count"])
            + len(new_transition_universe),
            "transition_count": len(all_transitions),
            "exclusion_count": len(all_exclusions),
            "patient_count": len(set(all_transitions["patient_id"].astype(str))),
            "visit_count": len(all_visits),
            "fold_transition_counts": {
                fold: int((all_transitions["fold"].astype(str) == fold).sum())
                for fold in ("train", "val")
            },
            "fold_patient_counts": {fold: len(split[fold]) for fold in ("train", "val")},
            "transition_type_counts": dict(sorted(transition_types.items())),
            "empty_source_or_target_mask_transitions": 0,
        }
        if flatten_existing_expansion:
            base_cache_contract = dict(base["base_cache_contract"])
            base_data_contract = dict(base["base_data_contract"])
            prior_expansion = dict(base["expansion"])
            base_bundle_sha256 = str(base["source_contracts"]["base_bundle_sha256"])
        else:
            base_cache_contract = {
                "schema": CACHE_SCHEMA,
                "bundle_contract_sha256": str(base["bundle_contract_sha256"]),
                "preprocess_sha256": str(base["artifacts"]["preprocess"]["sha256"]),
                "normalization_sha256": str(base["artifacts"]["normalization"]["sha256"]),
            }
            base_data_contract = build_base_data_contract(
                bundle_contract_sha256=str(base["bundle_contract_sha256"]),
                visits=base_visits,
                transitions=base_transitions,
                exclusions=base_exclusions,
                split=base_split,
                crop_plans=base_plans,
            )
            prior_expansion = {
                "candidate_patient_count": 0,
                "candidate_transition_count": 0,
                "selected_patient_count": 0,
                "selected_transition_count": 0,
                "excluded_transition_count": 0,
            }
            base_bundle_sha256 = sha256_file(base_bundle_path)
        expansion = {
            "candidate_patient_count": int(prior_expansion["candidate_patient_count"])
            + len(requested_patients),
            "candidate_transition_count": int(
                prior_expansion["candidate_transition_count"]
            )
            + len(new_transition_universe),
            "selected_patient_count": int(prior_expansion["selected_patient_count"])
            + len(eligible_patients),
            "selected_transition_count": int(
                prior_expansion["selected_transition_count"]
            )
            + len(selected_new),
            "excluded_transition_count": int(
                prior_expansion["excluded_transition_count"]
            )
            + len(excluded_new),
        }
        contract = {
            "schema_version": str(base["schema_version"]),
            "counts": counts,
            "artifacts": artifacts,
            "source_contracts": {
                **dict(base.get("source_contracts", {})),
                **{str(key): str(value) for key, value in source_contracts.items()},
                "base_bundle_sha256": base_bundle_sha256,
            },
            "strict_a_rule_version": str(base["strict_a_rule_version"]),
            "split_policy": (
                str(base["split_policy"])
                if flatten_existing_expansion
                else "frozen_base_val_new_train_v1"
            ),
            "normalization_policy": "frozen_base_train_v1",
            "text_policy": dict(base.get("text_policy", {})),
            "base_cache_contract": base_cache_contract,
            "base_data_contract": base_data_contract,
            "expansion": expansion,
        }
        increment = staging / "increment"
        increment.mkdir()
        _write_lines(increment / "new_patient_ids.txt", sorted(cumulative_candidate_ids))
        _write_lines(increment / "new_visit_ids.txt", sorted(cumulative_visit_ids))
        _write_lines(
            increment / "new_transition_ids.txt",
            sorted(cumulative_transition_ids),
        )
        cumulative_new_exclusions.to_csv(
            increment / "new_exclusions.csv", index=False
        )
        _json_dump(increment / "summary.json", contract["expansion"])
        if sha256_file(increment / "new_patient_ids.txt") != contract[
            "source_contracts"
        ]["incremental_patient_allowlist_sha256"]:
            raise ValueError("combined incremental patient allowlist is inconsistent")
        contract["increment_contract"] = build_increment_contract(staging)
        contract["bundle_contract_sha256"] = sha256_json(contract)
        _json_dump(staging / "bundle.json", contract)
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify_bundle(output_dir / "bundle.json")


def build_expanded_bundle(
    *,
    base_bundle_path: Path,
    registered_manifest_path: Path,
    original_manifest_path: Path,
    metadata_path: Path,
    clinical_path: Path,
    strict_rules_path: Path,
    patient_allowlist_path: Path,
    output_dir: Path,
    progress_every: int = 25,
    expected_patient_count: int | None = 58,
) -> dict[str, Any]:
    require_source_files(
        {
            "base_bundle": base_bundle_path,
            "registered_manifest": registered_manifest_path,
            "original_manifest": original_manifest_path,
            "metadata": metadata_path,
            "clinical": clinical_path,
            "strict_a_rules": strict_rules_path,
            "patient_allowlist": patient_allowlist_path,
        }
    )
    if expected_patient_count == 58:
        combined_patient_ids = load_patient_allowlist(patient_allowlist_path)
    else:
        combined_patient_ids = load_patient_allowlist(
            patient_allowlist_path, expected_count=expected_patient_count
        )
    base = verify_bundle(base_bundle_path)
    if base.get("split_policy") in EXPANDED_SPLIT_POLICIES:
        prior_patient_ids = set(
            _read_sorted_lines(base_bundle_path.parent / "increment/new_patient_ids.txt")
        )
        if not prior_patient_ids < set(combined_patient_ids):
            raise ValueError(
                "combined incremental allowlist must contain the complete prior increment"
            )
        patient_ids = tuple(
            value for value in combined_patient_ids if value not in prior_patient_ids
        )
    else:
        patient_ids = combined_patient_ids
    reconstructed = reconstruct_new_visit_records(
        original_manifest_path=original_manifest_path,
        metadata_path=metadata_path,
        clinical_path=clinical_path,
        patient_ids=patient_ids,
        progress_every=progress_every,
    )
    if set(reconstructed["patient_id"].astype(str)) != set(patient_ids):
        raise ValueError("incremental reconstructed patient coverage is incomplete")
    universe = adjacent_transition_universe(reconstructed)
    validate_real_incremental_cohort(
        patient_ids=patient_ids,
        visits=reconstructed,
        transitions=universe,
    )
    registered_manifest = pd.read_csv(registered_manifest_path)
    registered_rows = _registered_rows(reconstructed, registered_manifest)
    assessed, plans = _assess_registered_visits(
        registered_rows,
        target_spacing_xyz=TARGET_SPACING_XYZ,
        output_shape_zyx=OUTPUT_SHAPE_ZYX,
        progress_every=progress_every,
    )
    return stage_expanded_bundle(
        base_bundle_path=base_bundle_path,
        assessed_new_visits=assessed,
        new_transition_universe=universe,
        new_crop_plans=plans,
        patient_ids=patient_ids,
        source_contracts={
            "expanded_registered_manifest_sha256": sha256_file(registered_manifest_path),
            "expanded_original_manifest_sha256": sha256_file(original_manifest_path),
            "incremental_metadata_sha256": sha256_file(metadata_path),
            "incremental_patient_allowlist_sha256": sha256_file(patient_allowlist_path),
            "clinical_sha256": sha256_file(clinical_path),
            "strict_a_rules_sha256": sha256_file(strict_rules_path),
        },
        output_dir=output_dir,
    )

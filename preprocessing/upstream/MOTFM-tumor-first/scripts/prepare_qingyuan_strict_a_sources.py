#!/usr/bin/env python3
"""Prepare localized manifests for a versioned Qingyuan Strict-A expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from utils.ispy2_registry import load_clinical_table, normalize_patient_id


TERMINAL_STATES = {"fixed_reference", "deformable", "rigid_fallback"}
MANIFEST_FIELDS = (
    "patient_id",
    "visit",
    "study_instance_uid",
    "dce_paths",
    "ser_path",
    "mask_path",
    "meta_path",
    "ftv_voxel_count",
    "qc_status",
    "qc_warnings",
    "registration_status",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_lines(path: Path, values: Iterable[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def _single_path(root: Path, pattern: str, label: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1 or not matches[0].is_file():
        raise ValueError(f"expected one {label} under {root}, found {len(matches)}")
    return matches[0].resolve()


def _read_ids(path: Path) -> tuple[str, ...]:
    values = tuple(
        sorted(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    )
    if not values or len(values) != len(set(values)):
        raise ValueError(f"patient identity file is empty or duplicated: {path}")
    if any(normalize_patient_id(value) != value for value in values):
        raise ValueError(f"patient identity file is not canonical: {path}")
    return values


def _normalize_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.columns = [str(column).strip() for column in output.columns]
    output["patient_id"] = output["patient_id"].map(normalize_patient_id)
    return output


def _adjacent_count(visits: Iterable[str]) -> int:
    values = set(str(value) for value in visits)
    return sum(f"T{index}" in values and f"T{index + 1}" in values for index in range(3))


def prepare_sources(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Qingyuan Strict-A source output already exists: {output}")
    output.mkdir(parents=True)
    try:
        terminal = _normalize_manifest(
            pd.read_csv(args.terminal_rows, dtype=str, keep_default_na=False)
        )
        preprocessed = _normalize_manifest(
            pd.read_csv(args.preprocessed_manifest, dtype=str, keep_default_na=False)
        )
        visit_metadata = pd.read_csv(
            args.visit_metadata, dtype=str, keep_default_na=False
        )
        visit_metadata["patient_id"] = visit_metadata["patient_id"].map(
            normalize_patient_id
        )
        clinical = load_clinical_table(args.clinical)
        clinical_ids = set(clinical["patient_id"].astype(str))
        old_ids = set(_read_ids(args.old_patient_allowlist))
        active_visits = pd.read_csv(args.active_bundle_visits, dtype=str)
        active_ids = set(active_visits["patient_id"].astype(str))

        stage_ids = set(terminal["patient_id"].astype(str))
        eligible_ids = stage_ids.intersection(clinical_ids)
        missing_clinical_ids = stage_ids.difference(clinical_ids)
        if eligible_ids.intersection(active_ids) or eligible_ids.intersection(old_ids):
            raise ValueError("Qingyuan patients overlap the active or historical increment")
        combined_ids = old_ids | eligible_ids

        terminal_keys = list(zip(terminal["patient_id"], terminal["visit"]))
        preprocessed_keys = list(zip(preprocessed["patient_id"], preprocessed["visit"]))
        metadata_keys = list(zip(visit_metadata["patient_id"], visit_metadata["visit"]))
        if len(terminal_keys) != len(set(terminal_keys)):
            raise ValueError("terminal Qingyuan visit identities are duplicated")
        if len(preprocessed_keys) != len(set(preprocessed_keys)):
            raise ValueError("preprocessed Qingyuan visit identities are duplicated")
        if len(metadata_keys) != len(set(metadata_keys)):
            raise ValueError("Qingyuan StudyDate identities are duplicated")
        preprocessed_by_key = {
            key: row for key, row in zip(preprocessed_keys, preprocessed.to_dict("records"))
        }
        metadata_by_key = {
            key: row for key, row in zip(metadata_keys, visit_metadata.to_dict("records"))
        }

        new_original: list[dict[str, str]] = []
        new_registered: list[dict[str, str]] = []
        for row in terminal.to_dict("records"):
            patient_id = str(row["patient_id"])
            if patient_id not in eligible_ids:
                continue
            visit = str(row["visit"])
            key = (patient_id, visit)
            source = preprocessed_by_key.get(key)
            metadata = metadata_by_key.get(key)
            if source is None or metadata is None:
                raise ValueError(f"Qingyuan source identity is incomplete: {key}")
            source_patient_id = str(row.get("source_patient_id") or row.get("patient_id"))
            source_patient_id = str(source.get("source_patient_id") or source.get("patient_id"))
            source_patient_id = str(source.get("patient_id_raw") or source_patient_id)
            raw_terminal_patient = str(row.get("patient_id_raw") or "")
            if not raw_terminal_patient:
                raw_terminal_patient = str(
                    visit_metadata.loc[
                        (visit_metadata["patient_id"] == patient_id)
                        & (visit_metadata["visit"] == visit),
                        "source_patient_id",
                    ].iloc[0]
                )
            visit_root = args.data_root.expanduser().resolve() / raw_terminal_patient / visit
            dce0 = _single_path(visit_root, "dce/*_dce_aqc_0.nii.gz", "registered DCE0")
            ser = _single_path(visit_root, "ser/*.nii.gz", "registered SER")
            mask = _single_path(visit_root, "mask/*.nii.gz", "registered Mask")
            registered_meta = _single_path(visit_root, "meta/*.json", "registered meta")
            registration_json = visit_root / "registration.json"
            if not registration_json.is_file():
                raise FileNotFoundError(registration_json)
            original_meta_root = args.original_meta_root.expanduser().resolve()
            original_meta = _single_path(
                original_meta_root / raw_terminal_patient / visit,
                "meta/*.json",
                "original meta",
            )
            original_image_root = args.original_image_root.expanduser().resolve()
            original_visit_root = original_image_root / raw_terminal_patient / visit
            original_dce0 = _single_path(
                original_visit_root, "dce/*_dce_aqc_0.nii.gz", "original DCE0"
            )
            original_ser = _single_path(
                original_visit_root, "ser/*.nii.gz", "original SER"
            )
            original_mask = _single_path(
                original_visit_root, "mask/*.nii.gz", "original Mask"
            )
            status = str(row.get("registration_status", ""))
            if status not in TERMINAL_STATES:
                raise ValueError(f"Qingyuan registration is not terminal: {key}")
            study_uid = str(metadata["Study UID"])
            try:
                localized_meta_value = json.loads(original_meta.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                raise ValueError(f"Qingyuan original meta is unreadable: {key}") from None
            if not isinstance(localized_meta_value, dict):
                raise ValueError(f"Qingyuan original meta is invalid: {key}")
            existing_uid = str(
                localized_meta_value.get("study_uid")
                or localized_meta_value.get("study_instance_uid")
                or ""
            ).strip()
            if existing_uid and existing_uid != study_uid:
                raise ValueError(f"Qingyuan original meta Study UID conflicts: {key}")
            localized_meta_value["study_uid"] = study_uid
            localized_meta = (
                output
                / "localized_original_meta"
                / raw_terminal_patient
                / visit
                / original_meta.name
            )
            localized_meta.parent.mkdir(parents=True, exist_ok=True)
            _write_json(localized_meta, localized_meta_value)

            try:
                registration_value = json.loads(
                    registration_json.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError):
                raise ValueError(
                    f"Qingyuan registration JSON is unreadable: {key}"
                ) from None
            if not isinstance(registration_value, dict):
                raise ValueError(f"Qingyuan registration JSON is invalid: {key}")
            if visit != "T0":
                rigid_path = visit_root / "transforms" / "rigid.tfm"
                if not rigid_path.is_file():
                    raise FileNotFoundError(rigid_path)
                registration_value["transform_path"] = str(
                    visit_root / "transforms" / "combined_transform.json"
                )
            localized_registered_root = (
                output / "localized_registered" / raw_terminal_patient / visit
            )
            localized_registered_meta = (
                localized_registered_root / "meta" / registered_meta.name
            )
            localized_registered_meta.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(registered_meta, localized_registered_meta)
            _write_json(
                localized_registered_root / "registration.json", registration_value
            )

            common = {
                "patient_id": patient_id,
                "visit": visit,
                "study_instance_uid": study_uid,
                "ftv_voxel_count": str(source["ftv_voxel_count"]),
                "qc_warnings": str(source.get("qc_warnings", "")),
            }
            new_original.append(
                {
                    **common,
                    "dce_paths": str(original_dce0),
                    "ser_path": str(original_ser),
                    "mask_path": str(original_mask),
                    "meta_path": str(localized_meta),
                    "qc_status": str(source["qc_status"]),
                    "registration_status": "",
                }
            )
            new_registered.append(
                {
                    **common,
                    "dce_paths": str(dce0),
                    "ser_path": str(ser),
                    "mask_path": str(mask),
                    "meta_path": str(localized_registered_meta),
                    "qc_status": str(source["qc_status"]),
                    "registration_status": status,
                }
            )

        base_original = _normalize_manifest(
            pd.read_csv(args.base_original_manifest, dtype=str, keep_default_na=False)
        )
        base_registered = _normalize_manifest(
            pd.read_csv(args.base_registered_manifest, dtype=str, keep_default_na=False)
        )
        old_original = base_original.loc[base_original["patient_id"].isin(old_ids)].copy()
        old_registered = base_registered.loc[
            base_registered["patient_id"].isin(old_ids)
        ].copy()
        if set(old_original["patient_id"]) != old_ids or set(old_registered["patient_id"]) != old_ids:
            raise ValueError("historical 58-patient source coverage is incomplete")
        for frame in (old_original, old_registered):
            for field in MANIFEST_FIELDS:
                if field not in frame.columns:
                    frame[field] = ""
        original = pd.concat(
            [old_original[list(MANIFEST_FIELDS)], pd.DataFrame(new_original)],
            ignore_index=True,
        )
        registered = pd.concat(
            [old_registered[list(MANIFEST_FIELDS)], pd.DataFrame(new_registered)],
            ignore_index=True,
        )
        original = original.sort_values(["patient_id", "visit"])
        registered = registered.sort_values(["patient_id", "visit"])
        original_path = output / "combined_original_manifest.csv"
        registered_path = output / "combined_registered_manifest.csv"
        original.to_csv(original_path, index=False)
        registered.to_csv(registered_path, index=False)

        old_metadata = pd.read_csv(
            args.base_metadata, dtype=str, keep_default_na=False, skipinitialspace=True
        )
        old_metadata.columns = [str(column).strip() for column in old_metadata.columns]
        for field in ("Subject ID", "Study UID", "Study Date"):
            old_metadata[field] = old_metadata[field].astype(str).str.strip()
        old_metadata["patient_id"] = old_metadata["Subject ID"].map(normalize_patient_id)
        old_metadata = old_metadata.loc[old_metadata["patient_id"].isin(old_ids)]
        old_metadata = old_metadata[["Subject ID", "Study UID", "Study Date"]]
        new_metadata = visit_metadata.loc[
            visit_metadata["patient_id"].isin(eligible_ids),
            ["source_patient_id", "Study UID", "Study Date"],
        ].rename(columns={"source_patient_id": "Subject ID"})
        for field in ("Subject ID", "Study UID", "Study Date"):
            new_metadata[field] = new_metadata[field].astype(str).str.strip()
        metadata = pd.concat([old_metadata, new_metadata], ignore_index=True)
        metadata = metadata.drop_duplicates().sort_values(["Subject ID", "Study Date", "Study UID"])
        metadata_path = output / "combined_study_metadata.csv"
        metadata.to_csv(metadata_path, index=False)

        patient_visits = terminal.groupby("patient_id")["visit"].agg(list)
        missing_rows = [
            {
                "patient_id": patient_id,
                "visit_count": len(patient_visits[patient_id]),
                "adjacent_transition_count": _adjacent_count(patient_visits[patient_id]),
                "reason": "missing_required_clinical_row",
            }
            for patient_id in sorted(missing_clinical_ids)
        ]
        missing_path = output / "qingyuan_pre_strict_a_exclusions.csv"
        pd.DataFrame(missing_rows).to_csv(missing_path, index=False)
        _write_lines(output / "qingyuan_clinical_eligible_patient_ids.txt", sorted(eligible_ids))
        _write_lines(output / "combined_patient_ids.txt", sorted(combined_ids))

        selected_terminal = terminal.loc[terminal["patient_id"].isin(eligible_ids)]
        visit_counts = Counter(
            selected_terminal.groupby("patient_id")["visit"].count().astype(int)
        )
        report = {
            "schema": "qingyuan_strict_a_source_preflight_v1",
            "staging_patient_count": len(stage_ids),
            "staging_visit_count": len(terminal),
            "clinical_eligible_patient_count": len(eligible_ids),
            "clinical_eligible_visit_count": len(selected_terminal),
            "clinical_eligible_adjacent_transition_count": sum(
                _adjacent_count(visits) for visits in selected_terminal.groupby("patient_id")["visit"].agg(list)
            ),
            "missing_clinical_patient_count": len(missing_clinical_ids),
            "missing_clinical_adjacent_transition_count": sum(
                row["adjacent_transition_count"] for row in missing_rows
            ),
            "historical_increment_patient_count": len(old_ids),
            "combined_candidate_patient_count": len(combined_ids),
            "localized_original_meta_count": len(new_original),
            "localized_registered_meta_count": len(new_registered),
            "localized_registration_json_count": len(new_registered),
            "active_patient_overlap_count": len(eligible_ids.intersection(active_ids)),
            "eligible_visit_count_distribution": {
                str(key): int(value) for key, value in sorted(visit_counts.items())
            },
            "artifacts": {},
        }
        for path in (
            original_path,
            registered_path,
            metadata_path,
            missing_path,
            output / "qingyuan_clinical_eligible_patient_ids.txt",
            output / "combined_patient_ids.txt",
        ):
            report["artifacts"][path.name] = {"sha256": _sha256(path)}
        _write_json(output / "preflight.json", report)
        return report
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-rows", type=Path, required=True)
    parser.add_argument("--preprocessed-manifest", type=Path, required=True)
    parser.add_argument("--visit-metadata", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--original-meta-root", type=Path, required=True)
    parser.add_argument("--original-image-root", type=Path, required=True)
    parser.add_argument("--clinical", type=Path, required=True)
    parser.add_argument("--old-patient-allowlist", type=Path, required=True)
    parser.add_argument("--base-original-manifest", type=Path, required=True)
    parser.add_argument("--base-registered-manifest", type=Path, required=True)
    parser.add_argument("--base-metadata", type=Path, required=True)
    parser.add_argument("--active-bundle-visits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = prepare_sources(args)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

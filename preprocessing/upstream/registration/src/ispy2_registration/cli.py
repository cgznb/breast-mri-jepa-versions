from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from .config import (
    DEFAULT_CONFIG_PATH,
    ConfigurationError,
    PathConfig,
    RegistrationConfig,
    load_toml_config,
)
from .dicom_prepare import (
    audit_dicom_root,
    prepare_dicom_root,
    write_audit_outputs,
    write_combined_manifest,
)
from .fallback_visualization import generate_rigid_fallback_examples
from .manifest import DataValidationError, group_by_patient, load_manifest
from .models import VisitRecord
from .report import generate_final_report
from .pipeline import (
    CohortResult,
    register_cohort,
    register_patient,
    write_failure_manifest,
    write_output_manifest,
)
from .transforms import TransformApplicationError
from .tcia import (
    inventory_dicom_series,
    missing_series_uids,
    read_tcia_manifest,
    write_inventory_csv,
    write_tcia_manifest,
    write_uid_status_csv,
)


def audit_records(
    records: Sequence[VisitRecord], config: RegistrationConfig
) -> dict[str, Any]:
    patients = group_by_patient(records)
    with_t0 = {
        patient_id
        for patient_id, patient in patients.items()
        if config.fixed_visit in patient.visits
    }
    pairs = sum(
        1
        for record in records
        if record.patient_id in with_t0 and record.visit in config.moving_visits
    )
    empty_future = sum(
        1
        for record in records
        if record.visit in config.moving_visits and record.ftv_voxel_count == 0
    )
    registrable_empty_future = sum(
        1
        for record in records
        if record.patient_id in with_t0
        and record.visit in config.moving_visits
        and record.ftv_voxel_count == 0
    )
    return {
        "patients": len(patients),
        "visits": len(records),
        "patients_with_T0": len(with_t0),
        "future_to_T0_pairs": pairs,
        "empty_future_ftv": empty_future,
        "registrable_empty_future_ftv": registrable_empty_future,
        "patients_without_T0": sorted(set(patients) - with_t0),
    }


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)


def _add_registration_options(parser: argparse.ArgumentParser) -> None:
    _add_paths(parser)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--elastix-threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-dwi", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ispy2-register")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    _add_paths(audit)
    patient = subparsers.add_parser("register-patient")
    patient.add_argument("patient_id")
    _add_registration_options(patient)
    cohort = subparsers.add_parser("register-cohort")
    cohort.add_argument("--patients", default="")
    _add_registration_options(cohort)
    visualize = subparsers.add_parser("visualize-rigid-fallbacks")
    _add_paths(visualize)
    visualize.add_argument("--destination", type=Path, required=True)
    dicom_audit = subparsers.add_parser("audit-dicom")
    dicom_audit.add_argument("--dicom-root", type=Path, nargs="+", required=True)
    dicom_audit.add_argument("--output-root", type=Path, required=True)
    dicom_audit.add_argument("--collection", required=True)
    dicom_audit.add_argument("--deep", action="store_true")
    dicom_audit.add_argument("--workers", type=int, default=1)
    prepare = subparsers.add_parser("prepare-dicom")
    prepare.add_argument("--dicom-root", type=Path, nargs="+", required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--collection", required=True)
    prepare.add_argument("--workers", type=int, default=4)
    prepare.add_argument("--resume", action="store_true")
    prepare.add_argument("--skip-dwi", action="store_true")
    prepare.add_argument("--patients", default="")
    inventory = subparsers.add_parser("inventory-tcia")
    inventory.add_argument("--dicom-root", type=Path, nargs="+", required=True)
    inventory.add_argument("--source-manifest", type=Path, required=True)
    inventory.add_argument("--inventory-csv", type=Path, required=True)
    inventory.add_argument("--repair-manifest", type=Path, required=True)
    inventory.add_argument("--uid-status-csv", type=Path, default=None)
    inventory.add_argument("--mark-missing-excluded", action="store_true")
    inventory.add_argument("--deep-headers", action="store_true")
    inventory.add_argument("--verify-md5", action="store_true")
    combine = subparsers.add_parser("combine-manifests")
    combine.add_argument("--manifests", type=Path, nargs="+", required=True)
    combine.add_argument("--destination", type=Path, required=True)
    report = subparsers.add_parser("generate-final-report")
    report.add_argument("--registered-manifest", type=Path, required=True)
    report.add_argument("--uid-status", type=Path, nargs="+", required=True)
    report.add_argument("--failures", type=Path, nargs="+", required=True)
    report.add_argument("--output-root", type=Path, required=True)
    return parser


def _config(
    args: argparse.Namespace,
    defaults: RegistrationConfig,
) -> RegistrationConfig:
    overrides = {
        name: value
        for name in ("workers", "elastix_threads", "seed")
        if (value := getattr(args, name, None)) is not None
    }
    return replace(defaults, **overrides)


def _runtime_config(
    args: argparse.Namespace,
) -> tuple[PathConfig, RegistrationConfig]:
    paths, registration = load_toml_config(args.config)
    path_overrides: dict[str, Path] = {}
    if args.manifest is not None:
        path_overrides["manifest"] = args.manifest
    if args.output_root is not None:
        path_overrides["output_root"] = args.output_root
    if path_overrides:
        paths = replace(paths, **path_overrides)
    return paths, _config(args, registration)


def _output_manifest_path(output_root: Path) -> Path:
    return output_root / "manifest" / "ispy2_breastdcedl_registered_to_T0_manifest.csv"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inventory-tcia":
        try:
            manifest = read_tcia_manifest(args.source_manifest)
            rows = inventory_dicom_series(
                args.dicom_root,
                deep_headers=args.deep_headers,
                verify_md5=args.verify_md5,
            )
            inventory_path = write_inventory_csv(args.inventory_csv, rows)
            missing = missing_series_uids(manifest, rows)
            repair_path = write_tcia_manifest(args.repair_manifest, manifest, missing)
            uid_status_path = (
                write_uid_status_csv(
                    args.uid_status_csv,
                    manifest,
                    rows,
                    mark_missing_excluded=args.mark_missing_excluded,
                )
                if args.uid_status_csv is not None
                else None
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(json.dumps({"status": "inventory_failed", "error": str(exc)}), file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "expected_series": len(manifest.series_uids),
                    "present_series": len(
                        {
                            str(row["series_instance_uid"])
                            for row in rows
                            if row.get("series_instance_uid")
                        }
                    ),
                    "missing_series": len(missing),
                    "inventory_csv": str(inventory_path),
                    "repair_manifest": str(repair_path),
                    "uid_status_csv": str(uid_status_path) if uid_status_path else None,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "audit-dicom":
        try:
            audit = audit_dicom_root(
                args.dicom_root, deep=args.deep, workers=args.workers
            )
            outputs = write_audit_outputs(args.output_root, args.collection, audit)
        except (OSError, RuntimeError, ValueError) as exc:
            print(json.dumps({"status": "dicom_audit_failed", "error": str(exc)}), file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    key: audit[key]
                    for key in (
                        "patients",
                        "visits",
                        "series",
                        "patients_with_complete_T0",
                        "dce_complete_visits",
                        "dwi_complete_visits",
                        "multiple_complete_dwi_visits",
                    )
                }
                | {"outputs": {key: str(path) for key, path in outputs.items()}},
                indent=2,
                sort_keys=True,
            )
        )
        return 1 if audit["discovery_errors"] else 0
    if args.command == "prepare-dicom":
        try:
            summary = prepare_dicom_root(
                args.dicom_root,
                args.output_root,
                collection=args.collection,
                workers=args.workers,
                resume=args.resume,
                include_dwi=not args.skip_dwi,
                selected_patients=(value for value in args.patients.split(",") if value),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(json.dumps({"status": "dicom_preparation_failed", "error": str(exc)}), file=sys.stderr)
            return 2
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1 if summary["failed_items"] else 0
    if args.command == "combine-manifests":
        try:
            destination = write_combined_manifest(args.manifests, args.destination)
        except (OSError, RuntimeError, ValueError) as exc:
            print(json.dumps({"status": "manifest_combine_failed", "error": str(exc)}), file=sys.stderr)
            return 2
        print(json.dumps({"manifest": str(destination)}, indent=2))
        return 0
    if args.command == "generate-final-report":
        try:
            report = generate_final_report(
                args.registered_manifest,
                args.uid_status,
                args.failures,
                args.output_root,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(json.dumps({"status": "report_failed", "error": str(exc)}), file=sys.stderr)
            return 2
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1 if report["acceptance_status"] != "passed" else 0
    try:
        paths, config = _runtime_config(args)
    except ConfigurationError as exc:
        print(
            json.dumps({"status": "configuration_failed", "error": str(exc)}),
            file=sys.stderr,
        )
        return 2
    try:
        records = load_manifest(paths.manifest_path)
    except DataValidationError as exc:
        print(json.dumps({"status": "audit_failed", "error": str(exc)}), file=sys.stderr)
        return 2
    if getattr(args, "skip_dwi", False):
        records = [
            replace(
                record,
                dwi_paths=(),
                dwi_b_values=(),
                adc_path=None,
                dwi_mask_path=None,
                dwi_source_group=None,
                dwi_registration_status=(
                    "deferred"
                    if record.dwi_paths
                    or record.dwi_registration_status
                    not in {"", "not_available"}
                    else "not_available"
                ),
                dwi_geometry=None,
                dwi_origin_xyz=None,
                dwi_frame_of_reference_uid=None,
            )
            for record in records
        ]
    if args.command == "audit":
        print(json.dumps(audit_records(records, config), indent=2, sort_keys=True))
        return 0
    if args.command == "visualize-rigid-fallbacks":
        try:
            summary = generate_rigid_fallback_examples(
                records,
                paths.output_root,
                config,
                args.destination,
            )
        except (
            FileExistsError,
            FileNotFoundError,
            RuntimeError,
            TransformApplicationError,
            ValueError,
        ) as exc:
            print(
                json.dumps(
                    {"status": "visualization_failed", "error": str(exc)},
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 2
        print(
            json.dumps(
                {
                    "destination": str(summary.destination),
                    "files": len(summary.files),
                    "categories": dict(summary.categories),
                    "pairs_by_category": {
                        key: list(value)
                        for key, value in summary.pairs_by_category.items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    patients = group_by_patient(records)
    if args.patient_id not in patients if args.command == "register-patient" else False:
        print(
            json.dumps({"status": "unknown_patient", "patient_id": args.patient_id}),
            file=sys.stderr,
        )
        return 2
    output_root = paths.output_root
    if args.command == "register-patient":
        result = register_patient(
            patients[args.patient_id], output_root, config, resume=args.resume
        )
        write_output_manifest(
            paths.manifest_path,
            _output_manifest_path(output_root),
            (result,),
        )
        print(
            json.dumps(
                {
                    "patient_id": result.patient_id,
                    "status": result.status,
                    "pairs": [pair.registration_status for pair in result.pairs],
                },
                indent=2,
            )
        )
        return 1 if result.status in {"partial_failure", "failed_reference"} else 0
    selected = [value for value in args.patients.split(",") if value] or None
    try:
        cohort_result: CohortResult = register_cohort(
            patients,
            output_root,
            config,
            resume=args.resume,
            selected_patient_ids=selected,
        )
    except ValueError as exc:
        print(json.dumps({"status": "invalid_cohort", "error": str(exc)}), file=sys.stderr)
        return 2
    write_output_manifest(
        paths.manifest_path,
        _output_manifest_path(output_root),
        cohort_result.patients,
    )
    failure_path = write_failure_manifest(output_root, cohort_result)
    summary = {
        "patients": len(cohort_result.patients),
        "failed_pairs": len(cohort_result.failed_pairs),
        "failed_patients": len(cohort_result.failed_patients),
        "failure_manifest": str(failure_path),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if cohort_result.has_failures else 0


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()

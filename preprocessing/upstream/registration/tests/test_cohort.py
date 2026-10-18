from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import ispy2_registration.cli as registration_cli
import ispy2_registration.pipeline as registration_pipeline
from ispy2_registration.cli import audit_records, build_parser, main
from ispy2_registration.config import RegistrationConfig
from ispy2_registration.fallback_visualization import GenerationSummary
from ispy2_registration.models import Geometry, PatientVisits, VisitRecord
from ispy2_registration.phase_alignment import IntraVisitRegistrationError
from ispy2_registration.pipeline import PairResult, register_patient, write_output_manifest


GEOMETRY = Geometry(
    shape_zyx=(2, 3, 4),
    spacing_xyz=(1.0, 1.0, 2.0),
    direction=tuple(np.eye(3).ravel()),
)


def _record(tmp_path: Path, patient: str, visit: str, *, empty_ftv: bool = False) -> VisitRecord:
    root = tmp_path / "input" / patient / visit

    def image(modality: str, name: str, array: np.ndarray) -> Path:
        path = root / modality / name
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(array, np.eye(4)), path)
        return path

    base = np.arange(24, dtype=np.float32).reshape(GEOMETRY.shape_zyx)
    dce = image("dce", "phase.nii.gz", base)
    ftv_array = np.zeros(GEOMETRY.shape_zyx, dtype=np.uint8)
    if not empty_ftv:
        ftv_array[0, 1, 1] = 1
    ftv = image("mask", "ftv.nii.gz", ftv_array)
    raw = image("raw_mask", "raw.nii.gz", np.full(GEOMETRY.shape_zyx, 49, np.uint8))
    ser = image("ser", "ser.nii.gz", base)
    pe = image("pe2", "pe2.nii.gz", base)
    meta = root / "meta/meta.json"
    meta.parent.mkdir(parents=True)
    metadata = {
        "patient_id": patient,
        "visit": visit,
        "n_times": 1,
        "n_slices": 2,
        "rows": 3,
        "cols": 4,
        "dce_paths": [str(dce)],
        "dce_phase_ids": [0],
        "mask_path": str(ftv),
        "raw_mask_path": str(raw),
        "ser_path": str(ser),
        "pe_paths": {"pe2": str(pe)},
        "pixel_spacing": [1.0, 1.0],
        "spacing_between_slices": 2.0,
        "slice_thickness": 2.0,
        "image_orientation_patient": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "image_position_patient_first": [0.0, 0.0, 0.0],
        "ftv_voxel_count": int(ftv_array.sum()),
        "qc_status": "ok",
        "qc_warnings": [],
    }
    meta.write_text(json.dumps(metadata), encoding="utf-8")
    return VisitRecord(
        patient_id=patient,
        visit=visit,
        n_times=1,
        geometry=GEOMETRY,
        dce_paths=(dce,),
        ftv_mask_path=ftv,
        raw_mask_path=raw,
        ser_path=ser,
        pe_paths={"pe2": pe},
        meta_path=meta,
        ftv_voxel_count=int(ftv_array.sum()),
        qc_status="ok",
        qc_warnings=(),
        metadata=metadata,
    )


def test_audit_records_counts_pairs_empty_ftv_and_missing_t0(tmp_path: Path) -> None:
    records = [
        _record(tmp_path, "P1", "T0"),
        _record(tmp_path, "P1", "T1"),
        _record(tmp_path, "P1", "T2", empty_ftv=True),
        _record(tmp_path, "P2", "T1"),
    ]

    report = audit_records(records, RegistrationConfig())

    assert report["patients"] == 2
    assert report["visits"] == 4
    assert report["patients_with_T0"] == 1
    assert report["future_to_T0_pairs"] == 2
    assert report["empty_future_ftv"] == 1
    assert report["registrable_empty_future_ftv"] == 1
    assert report["patients_without_T0"] == ["P2"]


def test_register_patient_skips_without_t0(tmp_path: Path) -> None:
    patient = PatientVisits("P2", {"T1": _record(tmp_path, "P2", "T1")})

    result = register_patient(
        patient,
        tmp_path / "output",
        RegistrationConfig(workers=1, elastix_threads=1),
        resume=False,
    )

    assert result.status == "skipped_no_T0"
    assert result.reference is None
    assert result.pairs == ()


def test_register_patient_continues_after_one_pair_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patient = PatientVisits(
        "P1",
        {
            "T0": _record(tmp_path, "P1", "T0"),
            "T1": _record(tmp_path, "P1", "T1"),
            "T2": _record(tmp_path, "P1", "T2"),
        },
    )

    fixed_alignments = []

    def pair_runner(
        fixed,
        moving,
        output_root,
        config,
        *,
        fixed_phase_alignment,
        moving_phase_alignment,
        resume,
    ):
        fixed_alignments.append(fixed_phase_alignment)
        if moving.visit == "T1":
            raise RuntimeError("synthetic pair failure")
        return PairResult(
            patient_id=moving.patient_id,
            visit=moving.visit,
            registration_status="deformable",
            registered_to_visit="T0",
            output_dir=Path(output_root) / moving.patient_id / moving.visit,
            meta_path=None,
            transform_path=None,
            rigidity_applied=True,
            qc_status="deformable",
        )

    monkeypatch.setattr("ispy2_registration.pipeline.register_pair", pair_runner)

    result = register_patient(
        patient,
        tmp_path / "output",
        RegistrationConfig(workers=1, elastix_threads=1),
        resume=False,
    )

    assert result.status == "partial_failure"
    assert [pair.registration_status for pair in result.pairs] == ["failed", "deformable"]
    assert "synthetic pair failure" in result.pairs[0].message
    assert fixed_alignments[0] is fixed_alignments[1]


def test_register_patient_marks_only_failed_intra_visit_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patient = PatientVisits(
        "P1",
        {
            "T0": _record(tmp_path, "P1", "T0"),
            "T1": _record(tmp_path, "P1", "T1"),
            "T2": _record(tmp_path, "P1", "T2"),
        },
    )
    original = registration_pipeline._align_record_phases

    def align(record, config):
        if record.visit == "T1":
            raise IntraVisitRegistrationError("phase 5 synthetic failure")
        return original(record, config)

    def pair_runner(
        fixed,
        moving,
        output_root,
        config,
        *,
        fixed_phase_alignment,
        moving_phase_alignment,
        resume,
    ):
        return PairResult(
            patient_id=moving.patient_id,
            visit=moving.visit,
            registration_status="deformable",
            registered_to_visit="T0",
            output_dir=Path(output_root) / moving.patient_id / moving.visit,
            meta_path=None,
            transform_path=None,
            rigidity_applied=True,
            qc_status="deformable",
        )

    monkeypatch.setattr("ispy2_registration.pipeline._align_record_phases", align)
    monkeypatch.setattr("ispy2_registration.pipeline.register_pair", pair_runner)

    result = register_patient(
        patient,
        tmp_path / "output",
        RegistrationConfig(workers=1, elastix_threads=1),
        resume=False,
    )

    assert result.status == "partial_failure"
    assert [pair.registration_status for pair in result.pairs] == [
        "failed_intra_visit",
        "deformable",
    ]
    assert "phase 5" in result.pairs[0].message


def _mark_pair_complete(
    output_root: Path,
    record: VisitRecord,
    status: str,
) -> None:
    pair_dir = output_root / record.patient_id / record.visit
    declared = pair_dir / "complete.txt"
    declared.parent.mkdir(parents=True, exist_ok=True)
    declared.write_text("complete\n", encoding="utf-8")
    (pair_dir / "registration.json").write_text(
        json.dumps(
            {
                "patient_id": record.patient_id,
                "visit": record.visit,
                "registration_status": status,
                "registered_to_visit": "T0",
                "output_dir": str(pair_dir),
                "meta_path": None,
                "transform_path": None,
                "rigidity_applied": False,
                "qc_status": status,
                "output_files": [str(declared)],
                "message": "",
            }
        ),
        encoding="utf-8",
    )


def test_resumed_complete_patient_performs_no_phase_alignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t0 = _record(tmp_path, "P1", "T0")
    t1 = _record(tmp_path, "P1", "T1")
    output_root = tmp_path / "output"
    _mark_pair_complete(output_root, t0, "fixed_reference")
    _mark_pair_complete(output_root, t1, "deformable")
    patient = PatientVisits("P1", {"T0": t0, "T1": t1})
    align_calls: list[str] = []
    monkeypatch.setattr(
        registration_pipeline,
        "_align_record_phases",
        lambda record, config: align_calls.append(record.visit),
    )

    result = register_patient(
        patient,
        output_root,
        RegistrationConfig(workers=1, elastix_threads=1),
        resume=True,
    )

    assert align_calls == []
    assert result.reference is not None and result.reference.resumed
    assert all(pair.resumed for pair in result.pairs)


def test_resume_aligns_only_t0_and_pending_moving_visits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t0 = _record(tmp_path, "P1", "T0")
    t1 = _record(tmp_path, "P1", "T1")
    t2 = _record(tmp_path, "P1", "T2")
    output_root = tmp_path / "output"
    _mark_pair_complete(output_root, t0, "fixed_reference")
    _mark_pair_complete(output_root, t1, "deformable")
    patient = PatientVisits("P1", {"T0": t0, "T1": t1, "T2": t2})
    align_calls: list[str] = []

    def align(record, config):
        align_calls.append(record.visit)
        return object()

    def pair_runner(fixed, moving, output_root, config, **kwargs):
        return PairResult(
            patient_id=moving.patient_id,
            visit=moving.visit,
            registration_status="deformable",
            registered_to_visit="T0",
            output_dir=Path(output_root) / moving.patient_id / moving.visit,
            meta_path=None,
            transform_path=None,
            rigidity_applied=True,
            qc_status="deformable",
        )

    monkeypatch.setattr(registration_pipeline, "_align_record_phases", align)
    monkeypatch.setattr(registration_pipeline, "register_pair", pair_runner)

    result = register_patient(
        patient,
        output_root,
        RegistrationConfig(workers=1, elastix_threads=1),
        resume=True,
    )

    assert align_calls == ["T0", "T2"]
    assert result.pairs[0].resumed
    assert result.pairs[1].registration_status == "deformable"


def _manifest_row(record: VisitRecord) -> dict[str, str]:
    return {
        "patient_id": record.patient_id,
        "visit": record.visit,
        "n_times": str(record.n_times),
        "n_slices": str(record.geometry.shape_zyx[0]),
        "rows": str(record.geometry.shape_zyx[1]),
        "cols": str(record.geometry.shape_zyx[2]),
        "dce_source_dir": "source_dce",
        "mask_source_dir": "source_mask",
        "dce_output_dir": str(record.dce_paths[0].parent),
        "dce_paths": ";".join(str(path) for path in record.dce_paths),
        "mask_path": str(record.ftv_mask_path),
        "raw_mask_path": str(record.raw_mask_path),
        "ser_path": str(record.ser_path),
        "pe_paths_json": json.dumps({k: str(v) for k, v in record.pe_paths.items()}),
        "derived_paths_json": json.dumps({"ser": str(record.ser_path)}),
        "derived_source_dirs_json": "{}",
        "meta_path": str(record.meta_path),
        "ftv_voxel_count": str(record.ftv_voxel_count),
        "qc_status": "ok",
        "qc_warnings": "",
        "pe2_path": str(record.pe_paths["pe2"]),
    }


def test_output_manifest_preserves_original_columns_and_appends_registration(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path, "P2", "T1")
    row = _manifest_row(record)
    source_manifest = tmp_path / "manifest.csv"
    with source_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    patient_result = register_patient(
        PatientVisits("P2", {"T1": record}),
        tmp_path / "output",
        RegistrationConfig(workers=1, elastix_threads=1),
        resume=False,
    )
    output_manifest = tmp_path / "registered_manifest.csv"

    write_output_manifest(source_manifest, output_manifest, (patient_result,))

    with output_manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        output_row = next(reader)
        assert reader.fieldnames[: len(row)] == list(row)
        assert reader.fieldnames[-10:] == [
            "registration_status",
            "registered_to_visit",
            "transform_path",
            "rigidity_applied",
            "candidate_outside_protected_roi_folding",
            "outside_protected_roi_folding_accepted",
            "protected_roi_fold_fraction",
            "outside_protected_roi_fold_fraction",
            "global_fold_fraction",
            "qc_warning_codes",
        ]
    assert output_row["registration_status"] == "skipped_no_T0"
    assert output_row["candidate_outside_protected_roi_folding"] == "false"
    assert output_row["outside_protected_roi_folding_accepted"] == "false"
    assert output_row["qc_warning_codes"] == ""


def test_cli_returns_nonzero_for_unknown_patient(tmp_path: Path) -> None:
    record = _record(tmp_path, "P1", "T0")
    row = _manifest_row(record)
    source_manifest = tmp_path / "manifest.csv"
    with source_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    exit_code = main(
        [
            "register-patient",
            "UNKNOWN",
            "--manifest",
            str(source_manifest),
            "--output-root",
            str(tmp_path / "output"),
        ]
    )

    assert exit_code != 0


def test_parser_accepts_visualization_destination() -> None:
    args = build_parser().parse_args(
        ["visualize-rigid-fallbacks", "--destination", "/tmp/examples"]
    )

    assert args.command == "visualize-rigid-fallbacks"
    assert args.destination == Path("/tmp/examples")


def test_cli_dispatches_visualization_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = _record(tmp_path, "P1", "T0")
    row = _manifest_row(record)
    source_manifest = tmp_path / "manifest.csv"
    with source_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    destination = tmp_path / "examples"
    captured: dict[str, object] = {}

    def fake_generate(records, output_root, config, requested_destination):
        captured["records"] = records
        captured["output_root"] = output_root
        captured["destination"] = requested_destination
        return GenerationSummary(
            destination=requested_destination,
            files=(),
            categories={},
            pairs_by_category={},
        )

    monkeypatch.setattr(
        registration_cli,
        "generate_rigid_fallback_examples",
        fake_generate,
    )

    exit_code = main(
        [
            "visualize-rigid-fallbacks",
            "--manifest",
            str(source_manifest),
            "--output-root",
            str(tmp_path / "output"),
            "--destination",
            str(destination),
        ]
    )

    assert exit_code == 0
    assert captured["destination"] == destination
    assert captured["output_root"] == tmp_path / "output"
    assert json.loads(capsys.readouterr().out)["destination"] == str(destination)

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

from ispy2_registration import pipeline as registration_pipeline
from ispy2_registration.config import RegistrationConfig
from ispy2_registration.elastix import ElastixRegistrationError
from ispy2_registration.io import read_nifti
from ispy2_registration.models import Geometry, PatientVisits, VisitRecord
from ispy2_registration.phase_alignment import (
    PhaseAlignmentResult,
    PhaseRegistration,
    RegistrationRegion,
)
from ispy2_registration.pipeline import (
    _pair_satisfies_requested_modalities,
    _register_dwi_to_t0,
    _qc_payload,
    build_output_metadata,
    copy_reference_visit,
    plan_patient,
    register_pair,
)
from ispy2_registration.qc import QCResult
from ispy2_registration.transforms import TransformApplicationError


GEOMETRY = Geometry(
    shape_zyx=(2, 3, 4),
    spacing_xyz=(0.7, 0.8, 2.5),
    direction=(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0),
)


def _image(path: Path, value: int = 0, dtype=np.float32) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(np.full(GEOMETRY.shape_zyx, value, dtype=dtype), np.eye(4)),
        path,
    )
    return path


def _record(tmp_path: Path, visit: str) -> VisitRecord:
    root = tmp_path / "input/ISPY2-1" / visit
    dce_paths = tuple(_image(root / "dce" / f"phase_{index}.nii.gz", index) for index in range(2))
    ftv = _image(root / "mask" / "ftv.nii.gz", 1, np.uint8)
    raw = _image(root / "raw_mask" / "raw.nii.gz", 49, np.uint8)
    ser = _image(root / "ser" / "ser.nii.gz", 3)
    pe = _image(root / "pe2" / "pe2.nii.gz", 4)
    meta = root / "meta" / "meta.json"
    meta.parent.mkdir(parents=True)
    metadata = {
        "patient_id": "ISPY2-1",
        "visit": visit,
        "n_times": 2,
        "n_slices": 2,
        "rows": 3,
        "cols": 4,
        "dce_paths": [str(path) for path in dce_paths],
        "dce_phase_ids": [0, 1],
        "mask_path": str(ftv),
        "raw_mask_path": str(raw),
        "ser_path": str(ser),
        "pe_paths": {"pe2": str(pe)},
        "derived_paths": {"ser": str(ser), "pe2": str(pe)},
        "pixel_spacing": [0.8, 0.7],
        "spacing_between_slices": 2.5,
        "slice_thickness": 2.5,
        "image_orientation_patient": [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        "image_position_patient_first": [10.0, 20.0, 30.0],
        "ftv_voxel_count": int(np.prod(GEOMETRY.shape_zyx)),
        "qc_status": "ok",
        "qc_warnings": [],
    }
    meta.write_text(json.dumps(metadata), encoding="utf-8")
    return VisitRecord(
        patient_id="ISPY2-1",
        visit=visit,
        n_times=2,
        geometry=GEOMETRY,
        dce_paths=dce_paths,
        ftv_mask_path=ftv,
        raw_mask_path=raw,
        ser_path=ser,
        pe_paths={"pe2": pe},
        meta_path=meta,
        ftv_voxel_count=int(np.prod(GEOMETRY.shape_zyx)),
        qc_status="ok",
        qc_warnings=(),
        metadata=metadata,
    )


def _registration_record(tmp_path: Path, visit: str) -> VisitRecord:
    shape = (12, 16, 18)
    geometry = Geometry(
        shape_zyx=shape,
        spacing_xyz=(1.0, 1.0, 1.5),
        direction=tuple(np.eye(3).ravel()),
    )
    root = tmp_path / "registration_input/ISPY2-2" / visit
    z, y, x = np.mgrid[: shape[0], : shape[1], : shape[2]]
    gaussian = np.exp(
        -((x - 8.0) ** 2 + (y - 7.0) ** 2 + (z - 5.0) ** 2) / 18.0
    ).astype(np.float32)

    def save(path: Path, array: np.ndarray) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(array, np.eye(4)), path)
        return path

    dce_paths = (
        save(root / "dce/phase_0.nii.gz", gaussian * 100),
        save(root / "dce/phase_1.nii.gz", gaussian * 140),
    )
    ftv_array = np.zeros(shape, dtype=np.uint8)
    ftv_array[4:7, 6:9, 7:10] = 1
    ftv = save(root / "mask/ftv.nii.gz", ftv_array)
    raw_array = np.full(shape, 49, dtype=np.uint8)
    raw_array[ftv_array > 0] = 0
    raw = save(root / "raw_mask/raw.nii.gz", raw_array)
    ser = save(root / "ser/ser.nii.gz", gaussian * 2)
    pe = save(root / "pe2/pe2.nii.gz", gaussian * 3)
    meta = root / "meta/meta.json"
    meta.parent.mkdir(parents=True)
    metadata = {
        "patient_id": "ISPY2-2",
        "visit": visit,
        "n_times": 2,
        "n_slices": shape[0],
        "rows": shape[1],
        "cols": shape[2],
        "dce_paths": [str(path) for path in dce_paths],
        "dce_phase_ids": [0, 1],
        "mask_path": str(ftv),
        "raw_mask_path": str(raw),
        "ser_path": str(ser),
        "pe_paths": {"pe2": str(pe)},
        "pixel_spacing": [1.0, 1.0],
        "spacing_between_slices": 1.5,
        "slice_thickness": 1.5,
        "image_orientation_patient": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "image_position_patient_first": [0.0, 0.0, 0.0],
        "ftv_voxel_count": int(ftv_array.sum()),
        "qc_status": "ok",
        "qc_warnings": [],
    }
    meta.write_text(json.dumps(metadata), encoding="utf-8")
    return VisitRecord(
        patient_id="ISPY2-2",
        visit=visit,
        n_times=2,
        geometry=geometry,
        dce_paths=dce_paths,
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


def _identity_alignment(
    record: VisitRecord,
    *,
    aligned_second_value: float | None = None,
) -> PhaseAlignmentResult:
    phases = [read_nifti(path, dtype=np.float32) for path in record.dce_paths]
    if aligned_second_value is not None:
        phases[1] = np.full(record.geometry.shape_zyx, aligned_second_value, np.float32)
    registrations = tuple(
        PhaseRegistration(
            phase_id=phase_id,
            transform=sitk.Euler3DTransform(),
            metric_value=-0.5,
            iterations=3,
            stop_condition="synthetic",
            sampling_percentage=0.25,
            seed=20260719 + phase_id,
        )
        for phase_id in record.dce_phase_ids[1:]
    )
    return PhaseAlignmentResult(
        phase_ids=record.dce_phase_ids,
        aligned_phases=tuple(phases),
        registrations=registrations,
        region=RegistrationRegion(
            slices_zyx=tuple(slice(0, value) for value in record.geometry.shape_zyx),
            used_full_fov=True,
        ),
    )


def test_plan_patient_without_t0_is_explicit_skip(tmp_path: Path) -> None:
    patient = PatientVisits("ISPY2-1", {"T1": _record(tmp_path, "T1")})

    plan = plan_patient(patient, RegistrationConfig())

    assert plan.status == "skipped_no_T0"
    assert plan.fixed is None
    assert plan.moving == ()


def test_plan_patient_uses_only_available_future_visits(tmp_path: Path) -> None:
    patient = PatientVisits(
        "ISPY2-1",
        {"T3": _record(tmp_path, "T3"), "T0": _record(tmp_path, "T0")},
    )

    plan = plan_patient(patient, RegistrationConfig())

    assert plan.status == "ready"
    assert plan.fixed.visit == "T0"
    assert tuple(record.visit for record in plan.moving) == ("T3",)


def test_build_output_metadata_replaces_paths_and_uses_t0_geometry(tmp_path: Path) -> None:
    moving = _record(tmp_path, "T1")
    fixed = _record(tmp_path, "T0")
    final_dir = tmp_path / "registered/ISPY2-1/T1"
    output_paths = {
        "dce": [final_dir / "dce/phase_0.nii.gz", final_dir / "dce/phase_1.nii.gz"],
        "mask": final_dir / "mask/ftv.nii.gz",
        "raw_mask": final_dir / "raw_mask/raw.nii.gz",
        "ser": final_dir / "ser/ser.nii.gz",
        "pe": {"pe2": final_dir / "pe2/pe2.nii.gz"},
    }

    metadata = build_output_metadata(
        moving,
        fixed,
        output_paths,
        registration_status="deformable",
        transform_path=final_dir / "transforms/combined_transform.json",
        rigidity_applied=True,
        qc_payload={"status": "deformable"},
    )

    assert metadata["dce_paths"] == [str(path) for path in output_paths["dce"]]
    assert metadata["mask_path"] == str(output_paths["mask"])
    assert metadata["pe_paths"]["pe2"] == str(output_paths["pe"]["pe2"])
    assert metadata["registered_to_visit"] == "T0"
    assert metadata["registration_status"] == "deformable"
    assert metadata["n_slices"] == fixed.geometry.shape_zyx[0]
    assert metadata["pixel_spacing"] == [0.8, 0.7]
    assert metadata["source_geometry"]["shape_zyx"] == list(moving.geometry.shape_zyx)


def test_copy_reference_visit_is_atomic_and_complete(tmp_path: Path) -> None:
    fixed = _record(tmp_path, "T0")
    output_root = tmp_path / "registered"
    alignment = _identity_alignment(fixed, aligned_second_value=9.0)

    result = copy_reference_visit(
        fixed,
        output_root,
        phase_alignment=alignment,
        resume=False,
    )

    assert result.registration_status == "fixed_reference"
    assert result.meta_path.is_file()
    assert (output_root / "ISPY2-1/T0/registration.json").is_file()
    payload = json.loads(result.meta_path.read_text(encoding="utf-8"))
    assert payload["registration_status"] == "fixed_reference"
    assert all(Path(path).is_file() for path in payload["dce_paths"])
    assert np.array_equal(read_nifti(payload["dce_paths"][0]), read_nifti(fixed.dce_paths[0]))
    assert read_nifti(payload["dce_paths"][1]) == pytest.approx(9.0)
    assert read_nifti(payload["ser_path"]) == pytest.approx(read_nifti(fixed.ser_path))
    assert (
        output_root
        / "ISPY2-1/T0/transforms/intravisit/phase_1_to_phase_0.tfm"
    ).is_file()
    assert (
        output_root / "ISPY2-1/T0/transforms/intravisit/phase_alignment.json"
    ).is_file()
    assert (output_root / "ISPY2-1/T0/qc/intravisit/phase_1_to_phase_0.png").is_file()

    resumed = copy_reference_visit(
        fixed,
        output_root,
        phase_alignment=alignment,
        resume=True,
    )
    assert resumed.resumed


def test_dwi_rigid_fallback_failure_is_isolated_from_dce(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record(tmp_path, "T0")
    dwi = _image(tmp_path / "input/ISPY2-1/T0/dwi/b0.nii.gz", 2)
    dwi_mask = _image(
        tmp_path / "input/ISPY2-1/T0/dwi_mask/seg.nii.gz", 1, np.uint8
    )
    metadata = dict(record.metadata)
    metadata["frame_of_reference_uid"] = "dce-frame"
    with_dwi = replace(
        record,
        metadata=metadata,
        dwi_paths=(dwi,),
        dwi_b_values=(0.0,),
        dwi_mask_path=dwi_mask,
        dwi_geometry=record.geometry,
        dwi_origin_xyz=(10.0, 20.0, 30.0),
        dwi_frame_of_reference_uid="different-dwi-frame",
    )

    def fail_rigid(*args, **kwargs):
        raise RuntimeError("synthetic DWI rigid failure")

    monkeypatch.setattr(registration_pipeline, "estimate_rigid", fail_rigid)
    field = np.zeros((*record.geometry.shape_zyx, 3), dtype=np.float32)

    result = _register_dwi_to_t0(with_dwi, record, field, _fast_config())

    assert result.status == "failed"
    assert "synthetic DWI rigid failure" in result.message
    assert all(path.is_file() for path in record.dce_paths)


def test_deferred_dwi_status_is_preserved_without_loading_dwi(tmp_path: Path) -> None:
    record = replace(_record(tmp_path, "T0"), dwi_registration_status="deferred")
    field = np.zeros((*record.geometry.shape_zyx, 3), dtype=np.float32)

    result = _register_dwi_to_t0(record, record, field, _fast_config())

    assert result.status == "deferred"


def test_resume_reopens_deferred_pair_after_dwi_is_prepared(tmp_path: Path) -> None:
    record = _record(tmp_path, "T0")
    final_dir = tmp_path / "registered/ISPY2-1/T0"
    output = final_dir / "dce/phase_0.nii.gz"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"nifti-placeholder")
    (final_dir / "registration.json").write_text(
        json.dumps(
            {
                "registration_status": "fixed_reference",
                "dwi_registration_status": "deferred",
                "output_files": [str(output)],
            }
        ),
        encoding="utf-8",
    )
    dwi_path = _image(tmp_path / "input/ISPY2-1/T0/dwi/b0.nii.gz", 1)

    assert _pair_satisfies_requested_modalities(final_dir, record)
    assert not _pair_satisfies_requested_modalities(
        final_dir,
        replace(record, dwi_paths=(dwi_path,), dwi_b_values=(0.0,)),
    )


def _fast_config() -> RegistrationConfig:
    return RegistrationConfig(
        workers=1,
        elastix_threads=1,
        rigid_samples=200,
        rigid_iterations=3,
        bspline_resolutions=1,
        bspline_iterations=(0,),
        bspline_spatial_samples=100,
        similarity_tolerance=0.01,
    )


def test_qc_payload_serializes_protected_roi_folding_warning() -> None:
    result = QCResult(
        accepted=True,
        status="deformable",
        qc_status="warning_outside_protected_roi_folding",
        reasons=(),
        qc_warning_codes=(
            "candidate_outside_protected_roi_folding",
            "outside_protected_roi_folding_accepted",
        ),
        rigid_non_tumor_ncc=0.2,
        deformable_non_tumor_ncc=0.4,
        rigid_protected_ncc=0.3,
        deformable_protected_ncc=0.5,
        jacobian_min=-0.2,
        jacobian_max=2.0,
        protected_roi_fold_fraction=0.0,
        outside_protected_roi_fold_fraction=0.01,
        global_fold_fraction=0.009,
        candidate_outside_protected_roi_folding=True,
        outside_protected_roi_folding_accepted=True,
        ftv_volume_change=0.05,
    )

    payload = _qc_payload(result)

    assert payload["qc_status"] == "warning_outside_protected_roi_folding"
    assert payload["protected_roi_fold_fraction"] == 0.0
    assert payload["outside_protected_roi_fold_fraction"] == 0.01
    assert payload["global_fold_fraction"] == 0.009
    assert payload["candidate_outside_protected_roi_folding"] is True
    assert payload["outside_protected_roi_folding_accepted"] is True
    assert payload["qc_warning_codes"] == [
        "candidate_outside_protected_roi_folding",
        "outside_protected_roi_folding_accepted",
    ]


def test_register_pair_writes_all_modalities_and_transform_artifacts(
    tmp_path: Path,
) -> None:
    fixed = _registration_record(tmp_path, "T0")
    moving = _registration_record(tmp_path, "T1")
    output_root = tmp_path / "registered"

    result = register_pair(fixed, moving, output_root, _fast_config(), resume=False)

    assert result.registration_status == "deformable"
    assert result.rigidity_applied
    assert result.transform_path.is_file()
    assert result.meta_path.is_file()
    assert all(path.is_file() for path in result.output_files)
    output_dir = output_root / "ISPY2-2/T1"
    assert len(list((output_dir / "dce").glob("*.nii.gz"))) == 2
    assert read_nifti(output_dir / "mask/ftv.nii.gz").dtype == np.uint8
    assert set(np.unique(read_nifti(output_dir / "raw_mask/raw.nii.gz"))) <= {0, 49, 128}
    assert read_nifti(output_dir / "jacobian/jacobian.nii.gz").shape == fixed.geometry.shape_zyx
    assert read_nifti(output_dir / "deformation/displacement_field.nii.gz").shape == (
        *fixed.geometry.shape_zyx,
        3,
    )
    rigidity_path = output_dir / "transforms/elastix/moving_rigidity.mha"
    transform_parameters = output_dir / "transforms/elastix/TransformParameters.0.txt"
    requested_parameters = output_dir / "transforms/elastix/BSplineParameters.txt"
    assert rigidity_path.is_file()
    assert ".T1.tmp-" not in transform_parameters.read_text(encoding="utf-8")
    assert ".T1.tmp-" not in requested_parameters.read_text(encoding="utf-8")
    assert str(rigidity_path) in requested_parameters.read_text(encoding="utf-8")


def test_register_pair_uses_128_padding_only_for_raw_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = _registration_record(tmp_path, "T0")
    moving = _registration_record(tmp_path, "T1")
    output_root = tmp_path / "registered"
    real_apply_selected = registration_pipeline._apply_selected
    padding_values: list[float] = []

    def tracking_apply_selected(*args, default_value=0.0, **kwargs):
        padding_values.append(float(default_value))
        return real_apply_selected(
            *args,
            default_value=default_value,
            **kwargs,
        )

    monkeypatch.setattr(
        registration_pipeline,
        "_apply_selected",
        tracking_apply_selected,
    )

    result = register_pair(fixed, moving, output_root, _fast_config(), resume=False)

    assert result.registration_status in {"deformable", "rigid_fallback"}
    assert padding_values.count(128.0) == 1
    assert all(value == 0.0 for value in padding_values if value != 128.0)


def test_register_pair_uses_phase_aligned_drivers_and_writes_intra_visit_artifacts(
    tmp_path: Path,
) -> None:
    fixed = _registration_record(tmp_path, "T0")
    moving = _registration_record(tmp_path, "T1")
    fixed_alignment = _identity_alignment(fixed, aligned_second_value=210.0)
    moving_alignment = _identity_alignment(moving, aligned_second_value=310.0)
    output_root = tmp_path / "registered"

    result = register_pair(
        fixed,
        moving,
        output_root,
        _fast_config(),
        fixed_phase_alignment=fixed_alignment,
        moving_phase_alignment=moving_alignment,
        resume=False,
    )

    assert result.registration_status == "deformable"
    output_dir = output_root / "ISPY2-2/T1"
    alignment_path = output_dir / "transforms/intravisit/phase_alignment.json"
    assert alignment_path.is_file()
    alignment_payload = json.loads(alignment_path.read_text(encoding="utf-8"))
    assert alignment_payload["phase_ids"] == [0, 1]
    metadata = json.loads(result.meta_path.read_text(encoding="utf-8"))
    assert metadata["intra_visit_registration_path"] == str(alignment_path)
    combined = json.loads(result.transform_path.read_text(encoding="utf-8"))
    assert combined["dce_composition"] == (
        "T0 fixed point -> visit phase 0 via longitudinal transform -> "
        "original DCE phase via intra-visit rigid transform"
    )


def test_register_pair_falls_back_to_rigid_when_bspline_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = _registration_record(tmp_path, "T0")
    moving = _registration_record(tmp_path, "T1")

    def fail_bspline(*args, **kwargs):
        raise ElastixRegistrationError("synthetic B-spline failure")

    monkeypatch.setattr("ispy2_registration.pipeline.estimate_bspline", fail_bspline)

    result = register_pair(
        fixed,
        moving,
        tmp_path / "registered",
        _fast_config(),
        resume=False,
    )

    assert result.registration_status == "rigid_fallback"
    status = json.loads(
        (tmp_path / "registered/ISPY2-2/T1/registration.json").read_text(encoding="utf-8")
    )
    assert "bspline_failed" in status["qc"]["reasons"]


def test_register_pair_falls_back_when_deformable_output_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = _registration_record(tmp_path, "T0")
    moving = _registration_record(tmp_path, "T1")

    def fail_deformation(*args, **kwargs):
        raise TransformApplicationError("synthetic nonfinite deformation field")

    monkeypatch.setattr(
        "ispy2_registration.pipeline.displacement_and_jacobian",
        fail_deformation,
    )

    result = register_pair(
        fixed,
        moving,
        tmp_path / "registered",
        _fast_config(),
        resume=False,
    )

    assert result.registration_status == "rigid_fallback"
    output_dir = tmp_path / "registered/ISPY2-2/T1"
    status = json.loads((output_dir / "registration.json").read_text(encoding="utf-8"))
    assert status["qc"]["reasons"] == ["deformable_output_invalid"]
    assert "synthetic nonfinite deformation field" in status["qc"]["bspline_error"]
    assert np.isfinite(read_nifti(output_dir / "deformation/displacement_field.nii.gz")).all()
    assert np.isfinite(read_nifti(output_dir / "jacobian/jacobian.nii.gz")).all()

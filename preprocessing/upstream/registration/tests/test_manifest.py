from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from ispy2_registration.manifest import DataValidationError, load_manifest


def _write_nifti(path: Path, shape: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4)), path)
    return str(path)


def _write_manifest_case(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, str]]:
    patient = "ISPY2-000001"
    visit = "T0"
    shape = (3, 4, 5)
    dce_paths = [
        _write_nifti(tmp_path / patient / visit / "dce" / f"dce_{phase}.nii.gz", shape)
        for phase in range(2)
    ]
    mask_path = _write_nifti(tmp_path / patient / visit / "mask" / "mask.nii.gz", shape)
    raw_mask_path = _write_nifti(
        tmp_path / patient / visit / "raw_mask" / "raw_mask.nii.gz", shape
    )
    ser_path = _write_nifti(tmp_path / patient / visit / "ser" / "ser.nii.gz", shape)
    pe_path = _write_nifti(tmp_path / patient / visit / "pe2" / "pe2.nii.gz", shape)
    meta_path = tmp_path / patient / visit / "meta" / "meta.json"
    meta_path.parent.mkdir(parents=True)
    metadata: dict[str, object] = {
        "patient_id": patient,
        "visit": visit,
        "n_times": 2,
        "n_slices": shape[0],
        "rows": shape[1],
        "cols": shape[2],
        "dce_paths": dce_paths,
        "dce_phase_ids": [0, 2],
        "mask_path": mask_path,
        "raw_mask_path": raw_mask_path,
        "ser_path": ser_path,
        "pe_paths": {"pe2": pe_path},
        "pixel_spacing": [0.8, 0.7],
        "spacing_between_slices": 2.5,
        "image_orientation_patient": [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        "ftv_voxel_count": 12,
        "qc_status": "ok",
        "qc_warnings": [],
    }
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    row = {
        "patient_id": patient,
        "visit": visit,
        "n_times": "2",
        "n_slices": str(shape[0]),
        "rows": str(shape[1]),
        "cols": str(shape[2]),
        "dce_paths": ";".join(dce_paths),
        "mask_path": mask_path,
        "raw_mask_path": raw_mask_path,
        "ser_path": ser_path,
        "pe_paths_json": json.dumps({"pe2": pe_path}),
        "meta_path": str(meta_path),
        "ftv_voxel_count": "12",
        "qc_status": "ok",
        "qc_warnings": "",
    }
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return manifest_path, metadata, row


def test_load_manifest_uses_json_geometry_and_paths(tmp_path: Path) -> None:
    manifest_path, _, _ = _write_manifest_case(tmp_path)

    records = load_manifest(manifest_path)

    assert len(records) == 1
    record = records[0]
    assert record.patient_id == "ISPY2-000001"
    assert record.visit == "T0"
    assert record.geometry.shape_zyx == (3, 4, 5)
    assert record.geometry.spacing_xyz == pytest.approx((0.7, 0.8, 2.5))
    assert len(record.dce_paths) == 2
    assert record.dce_phase_ids == (0, 2)
    assert record.pe_paths["pe2"].is_file()


def test_load_manifest_rejects_csv_json_patient_mismatch(tmp_path: Path) -> None:
    manifest_path, metadata, row = _write_manifest_case(tmp_path)
    metadata["patient_id"] = "ISPY2-WRONG"
    Path(row["meta_path"]).write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(DataValidationError, match="patient_id"):
        load_manifest(manifest_path)


def test_load_manifest_rejects_path_mismatch_without_directory_guessing(
    tmp_path: Path,
) -> None:
    manifest_path, metadata, row = _write_manifest_case(tmp_path)
    metadata["ser_path"] = str(tmp_path / "different_ser.nii.gz")
    Path(row["meta_path"]).write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(DataValidationError, match="ser_path"):
        load_manifest(manifest_path)


def test_load_manifest_rejects_missing_declared_file(tmp_path: Path) -> None:
    manifest_path, _, row = _write_manifest_case(tmp_path)
    Path(row["mask_path"]).unlink()

    with pytest.raises(DataValidationError, match="does not exist"):
        load_manifest(manifest_path)


def test_load_manifest_rejects_nifti_shape_mismatch(tmp_path: Path) -> None:
    manifest_path, _, row = _write_manifest_case(tmp_path)
    nib.save(
        nib.Nifti1Image(np.zeros((2, 4, 5), dtype=np.float32), np.eye(4)),
        row["ser_path"],
    )

    with pytest.raises(DataValidationError, match="shape"):
        load_manifest(manifest_path)


@pytest.mark.parametrize("phase_ids", [None, [0], [1, 2], [0, 0]])
def test_load_manifest_rejects_invalid_json_phase_ids(
    tmp_path: Path,
    phase_ids: list[int] | None,
) -> None:
    manifest_path, metadata, row = _write_manifest_case(tmp_path)
    if phase_ids is None:
        metadata.pop("dce_phase_ids")
    else:
        metadata["dce_phase_ids"] = phase_ids
    Path(row["meta_path"]).write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(DataValidationError, match="dce_phase_ids"):
        load_manifest(manifest_path)


def test_load_manifest_accepts_optional_dwi_on_a_different_native_grid(tmp_path: Path) -> None:
    manifest_path, metadata, row = _write_manifest_case(tmp_path)
    dwi_shape = (2, 3, 4)
    dwi_paths = []
    for b_value in (0, 800):
        path = tmp_path / f"dwi_b{b_value}.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros(dwi_shape, dtype=np.float32), np.eye(4)), path)
        dwi_paths.append(path)
    adc = tmp_path / "adc.nii.gz"
    mask = tmp_path / "dwi_seg.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros(dwi_shape, dtype=np.float32), np.eye(4)), adc)
    nib.save(nib.Nifti1Image(np.zeros(dwi_shape, dtype=np.uint8), np.eye(4)), mask)
    metadata.update(
        {
            "collection": "ACRIN-6698",
            "dwi_paths": [str(path) for path in dwi_paths],
            "dwi_b_values": [0, 800],
            "adc_path": str(adc),
            "dwi_mask_path": str(mask),
            "dwi_source_group": 5,
            "dwi_registration_status": "prepared",
            "dwi": {
                "native_geometry": {
                    "shape_zyx": list(dwi_shape),
                    "spacing_xyz": [1.2, 1.3, 4.0],
                    "direction": list(np.eye(3).ravel()),
                    "origin_xyz": [10.0, 20.0, 30.0],
                },
                "frame_of_reference_uid": "1.2.3",
            },
        }
    )
    row.update(
        {
            "collection": "ACRIN-6698",
            "dwi_paths": ";".join(str(path) for path in dwi_paths),
            "dwi_b_values_json": json.dumps([0, 800]),
            "adc_path": str(adc),
            "dwi_mask_path": str(mask),
            "dwi_source_group": "5",
            "dwi_registration_status": "prepared",
        }
    )
    Path(row["meta_path"]).write_text(json.dumps(metadata), encoding="utf-8")
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    record = load_manifest(manifest_path)[0]

    assert record.collection == "ACRIN-6698"
    assert record.dwi_b_values == (0.0, 800.0)
    assert record.dwi_geometry is not None
    assert record.dwi_geometry.shape_zyx == dwi_shape
    assert record.dwi_origin_xyz == (10.0, 20.0, 30.0)

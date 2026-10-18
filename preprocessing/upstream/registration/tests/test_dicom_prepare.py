from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from ispy2_registration.dicom_prepare import (
    DwiCandidateData,
    SeriesInfo,
    VolumeData,
    _decode_seg_on_grid,
    _read_dwi_trace,
    _read_scalar_series,
    classify_series,
    discover_series,
    prepared_visit_complete,
    score_dwi_candidates,
)
from ispy2_registration.models import Geometry


def _file_dataset(path: Path, *, sop_class_uid: str | None = None) -> FileDataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class_uid or "1.2.840.10008.5.1.4.1.1.4"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    return dataset


def _write_mr(
    path: Path,
    *,
    patient_id: str,
    visit: str,
    description: str,
    series_uid: str,
    series_number: int,
    z: float,
    pixels: np.ndarray,
    b_value: float | None = None,
    slope: float | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = _file_dataset(path)
    ds.PatientID = patient_id
    ds.StudyDescription = f"ACRIN-6698_ISPY2_MRI_{visit}"
    ds.SeriesDescription = description
    ds.Modality = "MR"
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = generate_uid()
    ds.FrameOfReferenceUID = "1.2.3.4"
    ds.SeriesNumber = series_number
    ds.InstanceNumber = int(z + 10)
    ds.Manufacturer = "TEST"
    ds.Rows, ds.Columns = pixels.shape
    ds.PixelSpacing = [1.0, 1.0]
    ds.SpacingBetweenSlices = 2.0
    ds.SliceThickness = 2.0
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, z]
    if b_value is not None:
        ds.DiffusionBValue = b_value
    if slope is not None:
        ds.RescaleSlope = slope
        ds.RescaleIntercept = 1.0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = np.asarray(pixels, dtype=np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)
    return path


def _series_info(path: Path, kind: str, uid: str, group: int = 5) -> SeriesInfo:
    return SeriesInfo(
        path=path,
        patient_id="ACRIN-6698-000001",
        visit="T0",
        collection="ACRIN-6698",
        kind=kind,
        source_group=group,
        series_uid=uid,
        study_uid="1.2.3",
        frame_of_reference_uid="1.2.3.4",
        series_number=group,
        description=kind,
        manufacturer="TEST",
        file_count=len(list(path.glob("*.dcm"))),
    )


def _write_seg(path: Path, positions: list[float], frames: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = _file_dataset(path, sop_class_uid="1.2.840.10008.5.1.4.1.1.66.4")
    ds.PatientID = "ACRIN-6698-000001"
    ds.StudyDescription = "ACRIN-6698_ISPY2_MRI_T0"
    ds.SeriesDescription = "ACRIN-6698: DWI SEG: from S5: Whole Tumor Manual"
    ds.Modality = "SEG"
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.FrameOfReferenceUID = "1.2.3.4"
    ds.SeriesNumber = 59001
    ds.Rows, ds.Columns = frames.shape[1:]
    ds.NumberOfFrames = frames.shape[0]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    shared = Dataset()
    orientation = Dataset()
    orientation.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    shared.PlaneOrientationSequence = Sequence([orientation])
    ds.SharedFunctionalGroupsSequence = Sequence([shared])
    per_frame = []
    for z in positions:
        group = Dataset()
        plane = Dataset()
        plane.ImagePositionPatient = [0, 0, z]
        group.PlanePositionSequence = Sequence([plane])
        per_frame.append(group)
    ds.PerFrameFunctionalGroupsSequence = Sequence(per_frame)
    ds.PixelData = np.asarray(frames, dtype=np.uint8).tobytes()
    ds.save_as(path, enforce_file_format=True)
    return path


def test_classification_and_discovery_ignore_literal_quote_directory_names(tmp_path: Path) -> None:
    quoted = tmp_path / '"ACRIN-6698"' / '"ACRIN-6698-000001"' / '"study"' / '"series"'
    uid = generate_uid()
    _write_mr(
        quoted / "1.dcm",
        patient_id="ACRIN-6698-000001",
        visit="T0",
        description="ISPY2: VOLSER: uni-lateral cropped: original DCE",
        series_uid=uid,
        series_number=91800,
        z=0,
        pixels=np.ones((2, 3), dtype=np.uint16),
    )

    series, errors = discover_series(tmp_path)

    assert errors == []
    assert [(item.patient_id, item.visit, item.kind) for item in series] == [
        ("ACRIN-6698-000001", "T0", "dce")
    ]
    assert classify_series("ACRIN-6698: DWI TRACE: from S5", "MR") == "dwi_trace"
    assert classify_series("ACRIN-6698: ADC: from S5", "MR") == "adc"


def test_dwi_is_sorted_by_b_value_and_adc_rescale_is_applied(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_uid = generate_uid()
    for b_value in (800.0, 0.0):
        for z in (2.0, 0.0):
            _write_mr(
                trace_dir / f"b{b_value:g}_z{z:g}.dcm",
                patient_id="ACRIN-6698-000001",
                visit="T0",
                description="ACRIN-6698: DWI TRACE: from S5: bVals=0,800",
                series_uid=trace_uid,
                series_number=50100,
                z=z,
                b_value=b_value,
                pixels=np.full((2, 3), int(b_value + z), dtype=np.uint16),
            )
    adc_dir = tmp_path / "adc"
    adc_uid = generate_uid()
    for z in (2.0, 0.0):
        _write_mr(
            adc_dir / f"z{z:g}.dcm",
            patient_id="ACRIN-6698-000001",
            visit="T0",
            description="ACRIN-6698: ADC: from S5: bVals=0,800",
            series_uid=adc_uid,
            series_number=50200,
            z=z,
            slope=0.001,
            pixels=np.full((2, 3), 2000, dtype=np.uint16),
        )

    volumes, b_values = _read_dwi_trace(_series_info(trace_dir, "dwi_trace", trace_uid))
    adc = _read_scalar_series(_series_info(adc_dir, "adc", adc_uid), apply_rescale=True)

    assert b_values == (0.0, 800.0)
    assert volumes[0].array[:, 0, 0].tolist() == [0.0, 2.0]
    assert adc.array == pytest.approx(np.full((2, 2, 3), 3.0, dtype=np.float32))


def test_sparse_seg_is_reconstructed_on_full_dwi_grid(tmp_path: Path) -> None:
    geometry = Geometry((4, 2, 3), (1.0, 1.0, 2.0), tuple(np.eye(3).ravel()))
    reference = VolumeData(
        array=np.zeros(geometry.shape_zyx, dtype=np.float32),
        geometry=geometry,
        origin_xyz=(0.0, 0.0, 0.0),
        frame_of_reference_uid="1.2.3.4",
        series_uid="trace",
        metadata={},
    )
    frames = np.zeros((2, 2, 3), dtype=np.uint8)
    frames[0, 0, 1] = 1
    frames[1, 1, 2] = 1
    seg_dir = tmp_path / "seg"
    seg_path = _write_seg(seg_dir / "1.dcm", [2.0, 6.0], frames)
    header = __import__("pydicom").dcmread(seg_path, stop_before_pixels=True)
    info = _series_info(seg_dir, "dwi_seg", str(header.SeriesInstanceUID))

    result = _decode_seg_on_grid(info, reference)

    assert result.array.shape == geometry.shape_zyx
    assert result.array[1, 0, 1] == 1
    assert result.array[3, 1, 2] == 1
    assert result.metadata["mapped_slice_indices"] == [1, 3]


def test_candidate_selection_prioritizes_adc_then_dice_distance_and_source() -> None:
    geometry = Geometry((2, 3, 4), (1.0, 1.0, 1.0), tuple(np.eye(3).ravel()))
    dce = VolumeData(
        np.zeros(geometry.shape_zyx), geometry, (0, 0, 0), "frame", "dce", {}
    )
    ftv = np.zeros(geometry.shape_zyx, dtype=np.uint8)
    ftv[0, 1, 1] = 1

    def candidate(group: int, *, adc: bool, exact: bool) -> DwiCandidateData:
        mask = np.zeros(geometry.shape_zyx, dtype=np.uint8)
        mask[0, 1, 1 if exact else 2] = 1
        volume = VolumeData(np.zeros(geometry.shape_zyx), geometry, (0, 0, 0), "frame", str(group), {})
        segment = VolumeData(mask, geometry, (0, 0, 0), "frame", str(group), {})
        return DwiCandidateData(
            group,
            (volume,),
            (0.0,),
            volume if adc else None,
            segment,
            {},
            {},
        )

    no_adc_exact = candidate(4, adc=False, exact=True)
    adc_offset = candidate(5, adc=True, exact=False)
    selected, scores = score_dwi_candidates([no_adc_exact, adc_offset], dce, ftv)

    assert selected is adc_offset
    assert scores[0]["source_group"] == 5
    assert scores[0]["selected"] is True
    assert scores[1]["ftv_dice"] == 1.0


def test_deferred_dwi_is_complete_for_dce_first_but_resumed_for_enrichment(
    tmp_path: Path,
) -> None:
    final_dir = tmp_path / "preprocessed/ACRIN-6698-000001/T0"
    output = final_dir / "dce/phase_0.nii.gz"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"nifti-placeholder")
    (final_dir / "preparation.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "dwi_registration_status": "deferred",
                "output_files": [str(output)],
            }
        ),
        encoding="utf-8",
    )

    assert prepared_visit_complete(final_dir)
    assert not prepared_visit_complete(final_dir, require_dwi=True)

from __future__ import annotations

import csv
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.pixels import apply_modality_lut

from .geometry import direction_from_iop
from .io import atomic_output_directory, write_json, write_nifti_identity
from .models import Geometry


VISIT_ORDER = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}
SEG_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.66.4"


@dataclass(frozen=True)
class SeriesInfo:
    path: Path
    patient_id: str
    visit: str
    collection: str
    kind: str
    source_group: int | None
    series_uid: str
    study_uid: str
    frame_of_reference_uid: str
    series_number: int | None
    description: str
    manufacturer: str
    file_count: int


@dataclass
class VisitPlan:
    patient_id: str
    visit: str
    collection: str
    series: dict[str, list[SeriesInfo]] = field(default_factory=lambda: defaultdict(list))
    discovery_errors: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        return (self.patient_id, self.visit)


@dataclass(frozen=True)
class VolumeData:
    array: np.ndarray
    geometry: Geometry
    origin_xyz: tuple[float, float, float]
    frame_of_reference_uid: str
    series_uid: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class DwiCandidateData:
    source_group: int
    dwi: tuple[VolumeData, ...]
    b_values: tuple[float, ...]
    adc: VolumeData | None
    segmentation: VolumeData
    validation: Mapping[str, Any]
    series: Mapping[str, SeriesInfo]


class DicomPreparationError(RuntimeError):
    pass


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("_", " ").replace("-", " ").replace(":", " ")
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_visit(*values: Any) -> str | None:
    for value in values:
        text = "" if value is None else str(value)
        matches = re.findall(
            r"(?:^|[^A-Za-z0-9])(T[0-3])(?:$|[^A-Za-z0-9])",
            text,
            flags=re.IGNORECASE,
        )
        if matches:
            return matches[-1].upper()
        compact = re.sub(r"[^A-Za-z0-9]", "", text).upper()
        match = re.search(r"MRI(T[0-3])", compact)
        if match:
            return match.group(1)
    return None


def normalize_patient_id(value: Any) -> str:
    patient_id = str(value or "").strip().strip('"')
    if re.fullmatch(r"\d{6}", patient_id):
        return f"ISPY2-{patient_id}"
    if patient_id.startswith("ACRIN-6698-") or patient_id.startswith("ISPY2-"):
        return patient_id
    return patient_id


def collection_from_patient(patient_id: str) -> str:
    if patient_id.startswith("ACRIN-6698-"):
        return "ACRIN-6698"
    if patient_id.startswith("ISPY2-"):
        return "ISPY2"
    return "unknown"


def source_group_from_description(value: Any) -> int | None:
    match = re.search(r"\bfrom\s+S\s*([0-9]+)\b", str(value or ""), re.IGNORECASE)
    return int(match.group(1)) if match else None


def classify_series(description: Any, modality: Any) -> str | None:
    text = normalize_text(description)
    modality_text = str(modality or "").upper()
    if "dwi trace" in text and modality_text == "MR":
        return "dwi_trace"
    if re.search(r"\badc\b", text) and "from s" in text and modality_text == "MR":
        return "adc"
    if "dwi seg" in text and modality_text == "SEG":
        return "dwi_seg"
    if "dwi mask" in text and modality_text == "MR":
        return "dwi_mask_mr"
    if "volser" in text and "original dce" in text and modality_text == "MR":
        return "dce"
    if "volser" in text and "analysis mask" in text and modality_text == "SEG":
        return "analysis_mask"
    if "volser" in text and re.search(r"\bser\b", text):
        return "ser"
    pe_match = re.search(r"\bpe\s*([0-9]+)\b", text)
    if "volser" in text and pe_match:
        return f"pe{int(pe_match.group(1))}"
    return None


def list_dicom_files(series_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in series_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".dcm"
    )


def _header(path: Path):
    return pydicom.dcmread(str(path), stop_before_pixels=True, force=False)


def _dataset(path: Path):
    return pydicom.dcmread(str(path), stop_before_pixels=False, force=False)


def _roots(value: Path | str | Sequence[Path | str]) -> tuple[Path, ...]:
    values = (value,) if isinstance(value, (str, Path)) else tuple(value)
    roots = tuple(Path(item) for item in values)
    if not roots or any(not root.is_dir() for root in roots):
        raise ValueError(f"one or more DICOM roots do not exist: {roots}")
    return roots


def discover_series(
    dicom_root: Path | str | Sequence[Path | str],
) -> tuple[list[SeriesInfo], list[dict[str, str]]]:
    found: list[SeriesInfo] = []
    errors: list[dict[str, str]] = []
    for root in _roots(dicom_root):
        for directory, _, filenames in os.walk(root):
            names = sorted(name for name in filenames if name.lower().endswith(".dcm"))
            if not names:
                continue
            series_dir = Path(directory)
            first_path = series_dir / names[0]
            try:
                ds = _header(first_path)
                description = str(getattr(ds, "SeriesDescription", ""))
                kind = classify_series(description, getattr(ds, "Modality", ""))
                if kind is None:
                    continue
                patient_id = normalize_patient_id(getattr(ds, "PatientID", ""))
                visit = extract_visit(
                    getattr(ds, "StudyDescription", ""),
                    description,
                    str(series_dir),
                )
                if not patient_id or not visit:
                    raise ValueError("cannot identify patient or visit from DICOM headers")
                series_number_value = getattr(ds, "SeriesNumber", None)
                try:
                    series_number = int(float(str(series_number_value)))
                except (TypeError, ValueError):
                    series_number = None
                found.append(
                    SeriesInfo(
                        path=series_dir,
                        patient_id=patient_id,
                        visit=visit,
                        collection=collection_from_patient(patient_id),
                        kind=kind,
                        source_group=source_group_from_description(description),
                        series_uid=str(getattr(ds, "SeriesInstanceUID", "")),
                        study_uid=str(getattr(ds, "StudyInstanceUID", "")),
                        frame_of_reference_uid=str(getattr(ds, "FrameOfReferenceUID", "")),
                        series_number=series_number,
                        description=description,
                        manufacturer=str(getattr(ds, "Manufacturer", "")),
                        file_count=len(names),
                    )
                )
            except Exception as exc:
                errors.append(
                    {
                        "series_dir": str(series_dir),
                        "first_file": str(first_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return sorted(
        found,
        key=lambda item: (
            item.patient_id,
            VISIT_ORDER[item.visit],
            item.kind,
            item.series_number if item.series_number is not None else 10**12,
            item.series_uid,
        ),
    ), errors


def build_visit_plans(series: Iterable[SeriesInfo]) -> list[VisitPlan]:
    plans: dict[tuple[str, str], VisitPlan] = {}
    seen_uids: dict[str, Path] = {}
    for item in series:
        if item.series_uid in seen_uids and seen_uids[item.series_uid] != item.path:
            raise DicomPreparationError(
                f"SeriesInstanceUID {item.series_uid} appears in multiple directories"
            )
        seen_uids[item.series_uid] = item.path
        plan = plans.setdefault(
            (item.patient_id, item.visit),
            VisitPlan(item.patient_id, item.visit, item.collection),
        )
        plan.series[item.kind].append(item)
    return sorted(
        plans.values(), key=lambda plan: (plan.patient_id, VISIT_ORDER[plan.visit])
    )


def _float_values(value: Any) -> tuple[float, ...]:
    if value is None:
        return ()
    try:
        return tuple(float(item) for item in value)
    except TypeError:
        return (float(value),)


def _normalized(vector: Sequence[float]) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(values)
    if not np.isfinite(norm) or norm == 0:
        raise DicomPreparationError("invalid zero or nonfinite DICOM direction")
    return values / norm


def _slice_axis(iop: Sequence[float]) -> np.ndarray:
    if len(iop) != 6:
        raise DicomPreparationError("ImageOrientationPatient must have six values")
    return _normalized(np.cross(_normalized(iop[:3]), _normalized(iop[3:])))


def _position(ds: Any) -> tuple[float, float, float]:
    values = _float_values(getattr(ds, "ImagePositionPatient", None))
    if len(values) != 3:
        raise DicomPreparationError("invalid ImagePositionPatient")
    return values  # type: ignore[return-value]


def _spacing_from_positions(
    positions: Sequence[tuple[float, float, float]], axis: np.ndarray, ds: Any
) -> float:
    projected = sorted(float(np.dot(position, axis)) for position in positions)
    differences = np.diff(projected)
    nonzero = np.abs(differences[np.abs(differences) > 1e-5])
    if nonzero.size:
        return float(np.median(nonzero))
    for name in ("SpacingBetweenSlices", "SliceThickness"):
        value = getattr(ds, name, None)
        if value is not None and float(value) > 0:
            return float(value)
    raise DicomPreparationError("cannot determine slice spacing")


def _series_slices(info: SeriesInfo) -> tuple[list[tuple[Path, Any]], Geometry, tuple[float, float, float]]:
    files = list_dicom_files(info.path)
    if not files:
        raise DicomPreparationError(f"no DICOM files in {info.path}")
    headers = [(path, _header(path)) for path in files]
    first = headers[0][1]
    iop = _float_values(getattr(first, "ImageOrientationPatient", None))
    axis = _slice_axis(iop)
    keyed = sorted(
        headers,
        key=lambda item: (
            float(np.dot(_position(item[1]), axis)),
            int(getattr(item[1], "InstanceNumber", 10**12)),
            item[0].name,
        ),
    )
    positions = [_position(ds) for _, ds in keyed]
    unique = {round(float(np.dot(value, axis)), 5) for value in positions}
    if len(unique) != len(positions):
        raise DicomPreparationError(f"duplicate slice positions in {info.path}")
    pixel_spacing = _float_values(getattr(first, "PixelSpacing", None))
    if len(pixel_spacing) != 2:
        raise DicomPreparationError(f"invalid PixelSpacing in {info.path}")
    geometry = Geometry(
        shape_zyx=(
            len(keyed),
            int(getattr(first, "Rows")),
            int(getattr(first, "Columns")),
        ),
        spacing_xyz=(
            float(pixel_spacing[1]),
            float(pixel_spacing[0]),
            _spacing_from_positions(positions, axis, first),
        ),
        direction=direction_from_iop(iop),
    )
    return keyed, geometry, positions[0]


def _volume_metadata(info: SeriesInfo, geometry: Geometry, origin: Sequence[float]) -> dict[str, Any]:
    return {
        "source_dir": str(info.path),
        "series_uid": info.series_uid,
        "series_number": info.series_number,
        "series_description": info.description,
        "frame_of_reference_uid": info.frame_of_reference_uid,
        "manufacturer": info.manufacturer,
        "shape_zyx": list(geometry.shape_zyx),
        "spacing_xyz": list(geometry.spacing_xyz),
        "direction": list(geometry.direction),
        "origin_xyz": [float(value) for value in origin],
    }


def _read_scalar_series(info: SeriesInfo, *, apply_rescale: bool) -> VolumeData:
    records, geometry, origin = _series_slices(info)
    arrays: list[np.ndarray] = []
    for path, _ in records:
        ds = _dataset(path)
        pixels = np.asarray(ds.pixel_array)
        if pixels.ndim != 2:
            raise DicomPreparationError(f"expected 2D pixels in {path}, got {pixels.shape}")
        if apply_rescale:
            pixels = np.asarray(apply_modality_lut(pixels, ds), dtype=np.float32)
        arrays.append(np.asarray(pixels, dtype=np.float32))
    array = np.stack(arrays, axis=0)
    if not np.isfinite(array).all():
        raise DicomPreparationError(f"nonfinite pixels in {info.path}")
    metadata = _volume_metadata(info, geometry, origin)
    metadata["rescale_policy"] = (
        "pydicom_apply_modality_lut" if apply_rescale else "stored_pixel_values"
    )
    return VolumeData(
        array=array,
        geometry=geometry,
        origin_xyz=origin,
        frame_of_reference_uid=info.frame_of_reference_uid,
        series_uid=info.series_uid,
        metadata=metadata,
    )


def _frame_position(ds: Any, index: int) -> tuple[float, float, float] | None:
    groups = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if not groups or index >= len(groups):
        return None
    sequence = getattr(groups[index], "PlanePositionSequence", None)
    if not sequence:
        return None
    values = _float_values(getattr(sequence[0], "ImagePositionPatient", None))
    return values if len(values) == 3 else None  # type: ignore[return-value]


def _shared_iop(ds: Any) -> tuple[float, ...]:
    shared = getattr(ds, "SharedFunctionalGroupsSequence", None)
    if shared:
        sequence = getattr(shared[0], "PlaneOrientationSequence", None)
        if sequence:
            values = _float_values(getattr(sequence[0], "ImageOrientationPatient", None))
            if values:
                return values
    return _float_values(getattr(ds, "ImageOrientationPatient", None))


def _decode_seg_on_grid(info: SeriesInfo, reference: VolumeData) -> VolumeData:
    files = list_dicom_files(info.path)
    if len(files) != 1:
        raise DicomPreparationError(f"expected one SEG object in {info.path}, got {len(files)}")
    ds = _dataset(files[0])
    if str(getattr(ds, "Modality", "")).upper() != "SEG" and str(
        getattr(ds, "SOPClassUID", "")
    ) != SEG_SOP_CLASS_UID:
        raise DicomPreparationError(f"not a DICOM SEG object: {files[0]}")
    frames = np.asarray(ds.pixel_array)
    if frames.ndim == 2:
        frames = frames[np.newaxis]
    if frames.ndim != 3:
        raise DicomPreparationError(f"invalid SEG pixel shape {frames.shape}")
    if tuple(frames.shape[1:]) != reference.geometry.shape_zyx[1:]:
        raise DicomPreparationError("SEG rows/columns do not match reference grid")
    iop = _shared_iop(ds)
    if not np.allclose(
        np.asarray(direction_from_iop(iop)).reshape(3, 3),
        np.asarray(reference.geometry.direction).reshape(3, 3),
        atol=1e-5,
    ):
        raise DicomPreparationError("SEG orientation does not match reference grid")
    axis = _slice_axis(iop)
    reference_positions = [
        np.asarray(reference.origin_xyz, dtype=np.float64)
        + np.asarray(axis, dtype=np.float64) * reference.geometry.spacing_xyz[2] * index
        for index in range(reference.geometry.shape_zyx[0])
    ]
    reference_projection = np.asarray(
        [np.dot(position, axis) for position in reference_positions], dtype=np.float64
    )
    full = np.zeros(reference.geometry.shape_zyx, dtype=np.uint8)
    mapped: list[int] = []
    for index, frame in enumerate(frames):
        position = _frame_position(ds, index)
        if position is None:
            raise DicomPreparationError("SEG frame is missing ImagePositionPatient")
        distances = np.abs(reference_projection - float(np.dot(position, axis)))
        target = int(np.argmin(distances))
        if distances[target] > max(1e-3, reference.geometry.spacing_xyz[2] * 0.2):
            raise DicomPreparationError(
                f"SEG frame {index} is not on the reference slice grid"
            )
        full[target] |= (np.asarray(frame) > 0).astype(np.uint8)
        mapped.append(target)
    metadata = _volume_metadata(info, reference.geometry, reference.origin_xyz)
    metadata.update(
        {
            "sparse_frame_count": int(frames.shape[0]),
            "mapped_slice_indices": mapped,
            "reconstruction_policy": "per_frame_ImagePositionPatient_to_full_DWI_grid",
            "source_unique_values": [int(value) for value in np.unique(frames)],
        }
    )
    return VolumeData(
        array=full,
        geometry=reference.geometry,
        origin_xyz=reference.origin_xyz,
        frame_of_reference_uid=info.frame_of_reference_uid,
        series_uid=info.series_uid,
        metadata=metadata,
    )


def _decode_full_analysis_mask(info: SeriesInfo, reference: VolumeData) -> tuple[np.ndarray, dict[str, Any]]:
    files = list_dicom_files(info.path)
    if len(files) != 1:
        raise DicomPreparationError(f"expected one Analysis Mask SEG in {info.path}")
    ds = _dataset(files[0])
    frames = np.asarray(ds.pixel_array)
    if frames.ndim == 2:
        frames = frames[np.newaxis]
    if frames.ndim != 3 or tuple(frames.shape[1:]) != reference.geometry.shape_zyx[1:]:
        raise DicomPreparationError(f"invalid Analysis Mask shape {frames.shape}")
    iop = _shared_iop(ds)
    axis = _slice_axis(iop)
    projection = np.asarray(
        [
            np.dot(
                np.asarray(reference.origin_xyz)
                + axis * reference.geometry.spacing_xyz[2] * index,
                axis,
            )
            for index in range(reference.geometry.shape_zyx[0])
        ]
    )
    raw = np.full(reference.geometry.shape_zyx, 128, dtype=np.uint8)
    mapped: list[int] = []
    for frame_index, frame in enumerate(frames):
        position = _frame_position(ds, frame_index)
        if position is None:
            raise DicomPreparationError("Analysis Mask frame lacks ImagePositionPatient")
        distances = np.abs(projection - float(np.dot(position, axis)))
        target = int(np.argmin(distances))
        if distances[target] > max(1e-3, reference.geometry.spacing_xyz[2] * 0.2):
            raise DicomPreparationError("Analysis Mask frame is off the DCE grid")
        raw[target] = np.asarray(frame, dtype=np.uint8)
        mapped.append(target)
    if len(set(mapped)) != reference.geometry.shape_zyx[0]:
        raise DicomPreparationError(
            f"Analysis Mask covers {len(set(mapped))}/{reference.geometry.shape_zyx[0]} DCE slices"
        )
    ftv = (raw == 0).astype(np.uint8)
    metadata = _volume_metadata(info, reference.geometry, reference.origin_xyz)
    metadata.update(
        {
            "mask_rule": "ftv_mask = (analysis_mask_raw == 0)",
            "raw_unique_values": [int(value) for value in np.unique(raw)],
            "ftv_voxel_count": int(ftv.sum()),
            "mapped_slice_indices": mapped,
        }
    )
    return raw, metadata


def _read_dce(info: SeriesInfo) -> tuple[tuple[VolumeData, ...], tuple[int, ...]]:
    files = list_dicom_files(info.path)
    if not files:
        raise DicomPreparationError(f"no DICOM files in {info.path}")
    grouped: defaultdict[int, list[tuple[Path, Any]]] = defaultdict(list)
    for path in files:
        ds = _header(path)
        try:
            phase = int(float(str(getattr(ds, "TemporalPositionIdentifier"))))
        except (TypeError, ValueError):
            raise DicomPreparationError(f"missing TemporalPositionIdentifier in {path}") from None
        grouped[phase].append((path, ds))
    phase_ids = tuple(sorted(grouped))
    if not phase_ids or phase_ids[0] != 0 or phase_ids != tuple(range(phase_ids[-1] + 1)):
        raise DicomPreparationError(f"invalid DCE phase IDs {phase_ids}")
    volumes: list[VolumeData] = []
    reference_positions: tuple[float, ...] | None = None
    for phase in phase_ids:
        first = grouped[phase][0][1]
        iop = _float_values(getattr(first, "ImageOrientationPatient", None))
        axis = _slice_axis(iop)
        ordered = sorted(
            grouped[phase], key=lambda item: float(np.dot(_position(item[1]), axis))
        )
        positions = tuple(round(float(np.dot(_position(ds), axis)), 5) for _, ds in ordered)
        if len(set(positions)) != len(positions):
            raise DicomPreparationError(f"duplicate DCE slice positions in phase {phase}")
        if reference_positions is None:
            reference_positions = positions
        elif positions != reference_positions:
            raise DicomPreparationError(f"DCE positions differ in phase {phase}")
        pixel_spacing = _float_values(getattr(first, "PixelSpacing", None))
        geometry = Geometry(
            shape_zyx=(len(ordered), int(first.Rows), int(first.Columns)),
            spacing_xyz=(
                pixel_spacing[1],
                pixel_spacing[0],
                _spacing_from_positions([_position(ds) for _, ds in ordered], axis, first),
            ),
            direction=direction_from_iop(iop),
        )
        arrays = [np.asarray(_dataset(path).pixel_array, dtype=np.float32) for path, _ in ordered]
        array = np.stack(arrays)
        origin = _position(ordered[0][1])
        metadata = _volume_metadata(info, geometry, origin)
        metadata["phase_id"] = phase
        volumes.append(
            VolumeData(
                array=array,
                geometry=geometry,
                origin_xyz=origin,
                frame_of_reference_uid=info.frame_of_reference_uid,
                series_uid=info.series_uid,
                metadata=metadata,
            )
        )
    return tuple(volumes), phase_ids


def _read_dwi_trace(info: SeriesInfo) -> tuple[tuple[VolumeData, ...], tuple[float, ...]]:
    files = list_dicom_files(info.path)
    grouped: defaultdict[float, list[tuple[Path, Any]]] = defaultdict(list)
    for path in files:
        ds = _header(path)
        value = getattr(ds, "DiffusionBValue", None)
        if value is None:
            raise DicomPreparationError(f"missing DiffusionBValue in {path}")
        grouped[float(value)].append((path, ds))
    b_values = tuple(sorted(grouped))
    if not b_values or b_values[0] != 0:
        raise DicomPreparationError(f"DWI b-values must include b0, got {b_values}")
    volumes: list[VolumeData] = []
    expected_geometry: Geometry | None = None
    expected_positions: tuple[float, ...] | None = None
    for b_value in b_values:
        first = grouped[b_value][0][1]
        iop = _float_values(getattr(first, "ImageOrientationPatient", None))
        axis = _slice_axis(iop)
        ordered = sorted(
            grouped[b_value], key=lambda item: float(np.dot(_position(item[1]), axis))
        )
        positions = tuple(round(float(np.dot(_position(ds), axis)), 5) for _, ds in ordered)
        pixel_spacing = _float_values(getattr(first, "PixelSpacing", None))
        geometry = Geometry(
            shape_zyx=(len(ordered), int(first.Rows), int(first.Columns)),
            spacing_xyz=(
                pixel_spacing[1],
                pixel_spacing[0],
                _spacing_from_positions([_position(ds) for _, ds in ordered], axis, first),
            ),
            direction=direction_from_iop(iop),
        )
        if expected_geometry is None:
            expected_geometry, expected_positions = geometry, positions
        elif geometry != expected_geometry or positions != expected_positions:
            raise DicomPreparationError(f"DWI b={b_value:g} grid differs from b0")
        origin = _position(ordered[0][1])
        array = np.stack(
            [np.asarray(_dataset(path).pixel_array, dtype=np.float32) for path, _ in ordered]
        )
        metadata = _volume_metadata(info, geometry, origin)
        metadata["b_value"] = b_value
        volumes.append(
            VolumeData(
                array=array,
                geometry=geometry,
                origin_xyz=origin,
                frame_of_reference_uid=info.frame_of_reference_uid,
                series_uid=info.series_uid,
                metadata=metadata,
            )
        )
    return tuple(volumes), b_values


def _select_single(plan: VisitPlan, kind: str, *, required: bool) -> SeriesInfo | None:
    values = plan.series.get(kind, [])
    if not values:
        if required:
            raise DicomPreparationError(f"{plan.patient_id}/{plan.visit}: missing {kind}")
        return None
    if len(values) != 1:
        raise DicomPreparationError(
            f"{plan.patient_id}/{plan.visit}: expected one {kind}, got {len(values)}"
        )
    return values[0]


def _candidate_series(plan: VisitPlan) -> dict[int, dict[str, SeriesInfo]]:
    grouped: defaultdict[int, dict[str, SeriesInfo]] = defaultdict(dict)
    for kind in ("dwi_trace", "adc", "dwi_seg", "dwi_mask_mr"):
        for info in plan.series.get(kind, []):
            if info.source_group is None:
                continue
            if kind in grouped[info.source_group]:
                raise DicomPreparationError(
                    f"{plan.patient_id}/{plan.visit}: duplicate {kind} for S{info.source_group}"
                )
            grouped[info.source_group][kind] = info
    return dict(grouped)


def _validate_sparse_mr_mask(info: SeriesInfo, segmentation: VolumeData) -> dict[str, Any]:
    records, geometry, _ = _series_slices(info)
    if geometry.shape_zyx[1:] != segmentation.geometry.shape_zyx[1:]:
        return {"status": "geometry_mismatch", "dice": None}
    axis = np.asarray(segmentation.geometry.direction).reshape(3, 3)[:, 2]
    segmentation_projection = np.asarray(
        [
            np.dot(
                np.asarray(segmentation.origin_xyz)
                + axis * segmentation.geometry.spacing_xyz[2] * index,
                axis,
            )
            for index in range(segmentation.geometry.shape_zyx[0])
        ]
    )
    overlap = intersection = mask_count = seg_count = 0
    mapped: list[int] = []
    for path, header in records:
        projection = float(np.dot(_position(header), axis))
        index = int(np.argmin(np.abs(segmentation_projection - projection)))
        pixels = np.asarray(_dataset(path).pixel_array) > 0
        segment = segmentation.array[index] > 0
        intersection += int(np.logical_and(pixels, segment).sum())
        mask_count += int(pixels.sum())
        seg_count += int(segment.sum())
        overlap += int(np.logical_or(pixels, segment).sum())
        mapped.append(index)
    denominator = mask_count + seg_count
    return {
        "status": "match" if denominator and 2 * intersection == denominator else "different",
        "dice": 1.0 if denominator == 0 else 2.0 * intersection / denominator,
        "intersection_voxels": intersection,
        "union_voxels": overlap,
        "mapped_slice_indices": mapped,
        "source_series_uid": info.series_uid,
    }


def decode_dwi_candidates(
    plan: VisitPlan, errors: list[dict[str, Any]] | None = None
) -> list[DwiCandidateData]:
    decoded: list[DwiCandidateData] = []
    for source_group, values in sorted(_candidate_series(plan).items()):
        if "dwi_trace" not in values or "dwi_seg" not in values:
            continue
        try:
            dwi, b_values = _read_dwi_trace(values["dwi_trace"])
            segmentation = _decode_seg_on_grid(values["dwi_seg"], dwi[0])
            adc = (
                _read_scalar_series(values["adc"], apply_rescale=True)
                if "adc" in values
                else None
            )
            if adc is not None and not _same_grid(adc, dwi[0]):
                raise DicomPreparationError(
                    f"{plan.patient_id}/{plan.visit}/S{source_group}: ADC grid differs from DWI"
                )
            if segmentation.frame_of_reference_uid != dwi[0].frame_of_reference_uid:
                raise DicomPreparationError(
                    f"{plan.patient_id}/{plan.visit}/S{source_group}: SEG frame differs from DWI"
                )
            validation = (
                _validate_sparse_mr_mask(values["dwi_mask_mr"], segmentation)
                if "dwi_mask_mr" in values
                else {"status": "not_available", "dice": None}
            )
            decoded.append(
                DwiCandidateData(
                    source_group=source_group,
                    dwi=dwi,
                    b_values=b_values,
                    adc=adc,
                    segmentation=segmentation,
                    validation=validation,
                    series=dict(values),
                )
            )
        except Exception as exc:
            if errors is None:
                raise
            errors.append(
                {
                    "source_group": source_group,
                    "selected": False,
                    "status": "candidate_preparation_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "series_uids": {
                        kind: info.series_uid for kind, info in values.items()
                    },
                }
            )
    return decoded


def _physical_image(volume: VolumeData, *, pixel_type: int = sitk.sitkFloat32) -> sitk.Image:
    image = sitk.GetImageFromArray(np.asarray(volume.array, dtype=np.float32))
    image.SetSpacing(volume.geometry.spacing_xyz)
    image.SetDirection(volume.geometry.direction)
    image.SetOrigin(volume.origin_xyz)
    return sitk.Cast(image, pixel_type)


def _resample_physical_mask(source: VolumeData, target: VolumeData) -> np.ndarray:
    result = sitk.Resample(
        _physical_image(source),
        _physical_image(target),
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkNearestNeighbor,
        0.0,
        sitk.sitkUInt8,
    )
    return sitk.GetArrayFromImage(result).astype(np.uint8)


def _centroid_mm(mask: np.ndarray, geometry: Geometry, origin: Sequence[float]) -> np.ndarray | None:
    coordinates = np.argwhere(np.asarray(mask) > 0)
    if not coordinates.size:
        return None
    index_xyz = np.mean(coordinates[:, ::-1], axis=0)
    direction = np.asarray(geometry.direction).reshape(3, 3)
    return np.asarray(origin) + direction @ (index_xyz * np.asarray(geometry.spacing_xyz))


def score_dwi_candidates(
    candidates: Sequence[DwiCandidateData], dce_reference: VolumeData, ftv_mask: np.ndarray
) -> tuple[DwiCandidateData | None, list[dict[str, Any]]]:
    scores: list[tuple[tuple[float, ...], DwiCandidateData, dict[str, Any]]] = []
    ftv_volume = VolumeData(
        array=(np.asarray(ftv_mask) > 0).astype(np.uint8),
        geometry=dce_reference.geometry,
        origin_xyz=dce_reference.origin_xyz,
        frame_of_reference_uid=dce_reference.frame_of_reference_uid,
        series_uid="FTV",
        metadata={},
    )
    ftv_centroid = _centroid_mm(ftv_volume.array, ftv_volume.geometry, ftv_volume.origin_xyz)
    for candidate in candidates:
        shared_frame = (
            bool(candidate.segmentation.frame_of_reference_uid)
            and candidate.segmentation.frame_of_reference_uid
            == dce_reference.frame_of_reference_uid
        )
        if shared_frame:
            mapped = _resample_physical_mask(candidate.segmentation, ftv_volume)
            intersection = int(np.logical_and(mapped > 0, ftv_mask > 0).sum())
            denominator = int((mapped > 0).sum() + (ftv_mask > 0).sum())
            dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
            dwi_centroid = _centroid_mm(
                candidate.segmentation.array,
                candidate.segmentation.geometry,
                candidate.segmentation.origin_xyz,
            )
            centroid_distance = (
                float(np.linalg.norm(dwi_centroid - ftv_centroid))
                if dwi_centroid is not None and ftv_centroid is not None
                else math.inf
            )
        else:
            dice = 0.0
            centroid_distance = math.inf
        payload = {
            "source_group": candidate.source_group,
            "has_adc": candidate.adc is not None,
            "shared_frame_of_reference": shared_frame,
            "ftv_dice": dice,
            "centroid_distance_mm": (
                centroid_distance if math.isfinite(centroid_distance) else None
            ),
            "dwi_mask_cross_validation": dict(candidate.validation),
            "series_uids": {
                kind: info.series_uid for kind, info in candidate.series.items()
            },
        }
        rank = (
            0.0 if candidate.adc is not None else 1.0,
            -float(dice),
            centroid_distance,
            float(candidate.source_group),
        )
        scores.append((rank, candidate, payload))
    scores.sort(key=lambda item: item[0])
    payloads = [item[2] for item in scores]
    if payloads:
        payloads[0]["selected"] = True
        for payload in payloads[1:]:
            payload["selected"] = False
    return (scores[0][1] if scores else None), payloads


def _output_paths(root: Path, plan: VisitPlan, b_values: Sequence[float]) -> dict[str, Any]:
    base = f"{plan.patient_id}_{plan.visit}"
    return {
        "dce": [root / "dce" / f"{base}_dce_aqc_{index}.nii.gz" for index in range(64)],
        "mask": root / "mask" / f"{base}_mask.nii.gz",
        "raw_mask": root / "raw_mask" / f"{base}_analysis_mask_raw.nii.gz",
        "ser": root / "ser" / f"{base}_ser.nii.gz",
        "pe": {f"pe{index}": root / f"pe{index}" / f"{base}_pe{index}.nii.gz" for index in range(2, 8)},
        "dwi": [root / "dwi" / f"{base}_dwi_b{value:g}.nii.gz" for value in b_values],
        "adc": root / "adc" / f"{base}_adc.nii.gz",
        "dwi_mask": root / "dwi_mask" / f"{base}_dwi_seg.nii.gz",
        "meta": root / "meta" / f"{base}_meta.json",
        "status": root / "preparation.json",
    }


def _same_grid(left: VolumeData, right: VolumeData) -> bool:
    return (
        left.geometry == right.geometry
        and np.allclose(left.origin_xyz, right.origin_xyz, atol=1e-4)
        and left.frame_of_reference_uid == right.frame_of_reference_uid
    )


def _row_from_status(status_path: Path, *, resumed: bool) -> dict[str, Any]:
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    row = dict(payload["manifest_row"])
    row["preparation_resumed"] = str(resumed).lower()
    return row


def prepared_visit_complete(final_dir: Path, *, require_dwi: bool = False) -> bool:
    status = final_dir / "preparation.json"
    if not status.is_file():
        return False
    try:
        payload = json.loads(status.read_text(encoding="utf-8"))
        if require_dwi and payload.get("dwi_registration_status") == "deferred":
            return False
        return payload.get("status") == "completed" and all(
            Path(path).is_file() for path in payload.get("output_files", ())
        )
    except (OSError, ValueError, TypeError):
        return False


def convert_visit(
    plan: VisitPlan,
    output_root: Path,
    *,
    resume: bool,
    include_dwi: bool = True,
) -> dict[str, Any]:
    final_dir = Path(output_root) / plan.patient_id / plan.visit
    if resume and prepared_visit_complete(final_dir, require_dwi=include_dwi):
        return _row_from_status(final_dir / "preparation.json", resumed=True)
    dce_info = _select_single(plan, "dce", required=True)
    mask_info = _select_single(plan, "analysis_mask", required=True)
    ser_info = _select_single(plan, "ser", required=True)
    assert dce_info is not None and mask_info is not None and ser_info is not None
    dce, phase_ids = _read_dce(dce_info)
    reference = dce[0]
    raw_mask, mask_metadata = _decode_full_analysis_mask(mask_info, reference)
    ftv_mask = (raw_mask == 0).astype(np.uint8)
    ser = _read_scalar_series(ser_info, apply_rescale=False)
    if not _same_grid(ser, reference):
        raise DicomPreparationError(f"{plan.patient_id}/{plan.visit}: SER grid differs from DCE")
    pe_volumes: dict[str, VolumeData] = {}
    for kind in sorted(
        (key for key in plan.series if re.fullmatch(r"pe[0-9]+", key)),
        key=lambda key: int(key[2:]),
    ):
        info = _select_single(plan, kind, required=False)
        if info is None:
            continue
        volume = _read_scalar_series(info, apply_rescale=False)
        if not _same_grid(volume, reference):
            raise DicomPreparationError(f"{plan.patient_id}/{plan.visit}: {kind} grid differs from DCE")
        pe_volumes[kind] = volume
    if not pe_volumes:
        raise DicomPreparationError(f"{plan.patient_id}/{plan.visit}: no PE series")

    dwi_error = ""
    selected_dwi: DwiCandidateData | None = None
    candidate_scores: list[dict[str, Any]] = []
    if include_dwi:
        try:
            candidate_errors: list[dict[str, Any]] = []
            candidates = decode_dwi_candidates(plan, candidate_errors)
            selected_dwi, candidate_scores = score_dwi_candidates(
                candidates, reference, ftv_mask
            )
            candidate_scores.extend(candidate_errors)
            if selected_dwi is not None:
                dwi_status = "prepared"
            elif candidate_errors:
                dwi_status = "preparation_failed"
                dwi_error = "all complete DWI candidates failed preparation"
            else:
                dwi_status = "not_available"
        except Exception as exc:
            dwi_status = "preparation_failed"
            dwi_error = f"{type(exc).__name__}: {exc}"
    else:
        candidate_groups = _candidate_series(plan)
        has_complete_dwi = any(
            "dwi_trace" in values and "dwi_seg" in values
            for values in candidate_groups.values()
        )
        dwi_status = "deferred" if has_complete_dwi else "not_available"
    b_values = selected_dwi.b_values if selected_dwi is not None else ()
    paths = _output_paths(final_dir, plan, b_values)
    dce_paths = paths["dce"][: len(dce)]
    pe_paths = {kind: paths["pe"][kind] for kind in pe_volumes}
    dwi_paths = paths["dwi"] if selected_dwi is not None else []
    adc_path = paths["adc"] if selected_dwi is not None and selected_dwi.adc is not None else None
    dwi_mask_path = paths["dwi_mask"] if selected_dwi is not None else None

    dwi_metadata: dict[str, Any] | None = None
    if selected_dwi is not None:
        dwi_metadata = {
            "source_group": selected_dwi.source_group,
            "b_values": list(selected_dwi.b_values),
            "paths": [str(path) for path in dwi_paths],
            "adc_path": str(adc_path) if adc_path else None,
            "mask_path": str(dwi_mask_path),
            "native_geometry": {
                **selected_dwi.dwi[0].geometry.as_dict(),
                "origin_xyz": list(selected_dwi.dwi[0].origin_xyz),
            },
            "frame_of_reference_uid": selected_dwi.dwi[0].frame_of_reference_uid,
            "candidate_scores": candidate_scores,
            "selected_series_uids": {
                kind: info.series_uid for kind, info in selected_dwi.series.items()
            },
            "dwi_mask_cross_validation": dict(selected_dwi.validation),
        }
    metadata = {
        "patient_id": plan.patient_id,
        "visit": plan.visit,
        "collection": plan.collection,
        "n_times": len(dce),
        "n_slices": reference.geometry.shape_zyx[0],
        "rows": reference.geometry.shape_zyx[1],
        "cols": reference.geometry.shape_zyx[2],
        "dce_paths": [str(path) for path in dce_paths],
        "dce_phase_ids": list(phase_ids),
        "mask_path": str(paths["mask"]),
        "raw_mask_path": str(paths["raw_mask"]),
        "ser_path": str(paths["ser"]),
        "pe_paths": {kind: str(path) for kind, path in pe_paths.items()},
        "derived_paths": {
            "ser": str(paths["ser"]),
            **{kind: str(path) for kind, path in pe_paths.items()},
        },
        "pixel_spacing": [reference.geometry.spacing_xyz[1], reference.geometry.spacing_xyz[0]],
        "spacing_between_slices": reference.geometry.spacing_xyz[2],
        "slice_thickness": reference.geometry.spacing_xyz[2],
        "image_orientation_patient": list(
            np.asarray(reference.geometry.direction).reshape(3, 3)[:, :2].T.ravel()
        ),
        "image_position_patient_first": list(reference.origin_xyz),
        "frame_of_reference_uid": reference.frame_of_reference_uid,
        "dce_native_geometry": {
            **reference.geometry.as_dict(),
            "origin_xyz": list(reference.origin_xyz),
        },
        "ftv_voxel_count": int(ftv_mask.sum()),
        "qc_status": "ok" if int(ftv_mask.sum()) else "warning",
        "qc_warnings": [] if int(ftv_mask.sum()) else ["ftv_mask_is_empty"],
        "dwi_paths": [str(path) for path in dwi_paths],
        "dwi_b_values": list(b_values),
        "adc_path": str(adc_path) if adc_path else "",
        "dwi_mask_path": str(dwi_mask_path) if dwi_mask_path else "",
        "dwi_source_group": selected_dwi.source_group if selected_dwi else None,
        "dwi_registration_status": dwi_status,
        "dwi_preparation_error": dwi_error,
        "dwi": dwi_metadata,
        "mask_meta": mask_metadata,
        "source_series_uids": {
            kind: [info.series_uid for info in values]
            for kind, values in sorted(plan.series.items())
        },
        "source_series_dirs": {
            kind: [str(info.path) for info in values]
            for kind, values in sorted(plan.series.items())
        },
        "affine_policy": "identity_for_existing_DCE_registration_compatibility",
    }
    row: dict[str, Any] = {
        "patient_id": plan.patient_id,
        "visit": plan.visit,
        "collection": plan.collection,
        "n_times": len(dce),
        "n_slices": reference.geometry.shape_zyx[0],
        "rows": reference.geometry.shape_zyx[1],
        "cols": reference.geometry.shape_zyx[2],
        "dce_source_dir": str(dce_info.path),
        "mask_source_dir": str(mask_info.path),
        "dce_output_dir": str(dce_paths[0].parent),
        "dce_paths": ";".join(str(path) for path in dce_paths),
        "mask_path": str(paths["mask"]),
        "raw_mask_path": str(paths["raw_mask"]),
        "ser_path": str(paths["ser"]),
        "pe_paths_json": json.dumps({kind: str(path) for kind, path in pe_paths.items()}, sort_keys=True),
        "derived_paths_json": json.dumps(metadata["derived_paths"], sort_keys=True),
        "meta_path": str(paths["meta"]),
        "ftv_voxel_count": int(ftv_mask.sum()),
        "qc_status": metadata["qc_status"],
        "qc_warnings": "|".join(metadata["qc_warnings"]),
        "dwi_paths": ";".join(str(path) for path in dwi_paths),
        "dwi_b_values_json": json.dumps(list(b_values)),
        "adc_path": str(adc_path) if adc_path else "",
        "dwi_mask_path": str(dwi_mask_path) if dwi_mask_path else "",
        "dwi_source_group": selected_dwi.source_group if selected_dwi else "",
        "dwi_registration_status": dwi_status,
        "preparation_resumed": "false",
    }
    for index in range(2, 8):
        row[f"pe{index}_path"] = str(pe_paths.get(f"pe{index}", ""))
    output_files: list[Path] = [
        *dce_paths,
        paths["mask"],
        paths["raw_mask"],
        paths["ser"],
        *pe_paths.values(),
        *dwi_paths,
        *((adc_path,) if adc_path else ()),
        *((dwi_mask_path,) if dwi_mask_path else ()),
        paths["meta"],
        paths["status"],
    ]
    status_payload = {
        "status": "completed",
        "patient_id": plan.patient_id,
        "visit": plan.visit,
        "dwi_registration_status": dwi_status,
        "dwi_error": dwi_error,
        "manifest_row": row,
        "output_files": [str(path) for path in output_files],
    }
    with atomic_output_directory(final_dir) as temporary:
        def temporary_path(path: Path) -> Path:
            return temporary / path.relative_to(final_dir)

        for volume, path in zip(dce, dce_paths, strict=True):
            write_nifti_identity(temporary_path(path), volume.array.astype(np.float32))
        write_nifti_identity(temporary_path(paths["raw_mask"]), raw_mask.astype(np.uint8))
        write_nifti_identity(temporary_path(paths["mask"]), ftv_mask.astype(np.uint8))
        write_nifti_identity(temporary_path(paths["ser"]), ser.array.astype(np.float32))
        for kind, volume in pe_volumes.items():
            write_nifti_identity(temporary_path(pe_paths[kind]), volume.array.astype(np.float32))
        if selected_dwi is not None:
            for volume, path in zip(selected_dwi.dwi, dwi_paths, strict=True):
                write_nifti_identity(temporary_path(path), volume.array.astype(np.float32))
            if selected_dwi.adc is not None and adc_path is not None:
                write_nifti_identity(temporary_path(adc_path), selected_dwi.adc.array.astype(np.float32))
            assert dwi_mask_path is not None
            write_nifti_identity(
                temporary_path(dwi_mask_path), selected_dwi.segmentation.array.astype(np.uint8)
            )
        write_json(temporary_path(paths["meta"]), metadata)
        write_json(temporary_path(paths["status"]), status_payload)
        if not all(temporary_path(path).is_file() for path in output_files):
            raise DicomPreparationError(f"incomplete temporary output for {plan.patient_id}/{plan.visit}")
    if not prepared_visit_complete(final_dir, require_dwi=include_dwi):
        raise DicomPreparationError(f"published output incomplete for {plan.patient_id}/{plan.visit}")
    return row


def _deep_audit_plan(plan: VisitPlan) -> dict[str, Any]:
    try:
        dce_info = _select_single(plan, "dce", required=True)
        mask_info = _select_single(plan, "analysis_mask", required=True)
        assert dce_info is not None and mask_info is not None
        dce, _ = _read_dce(dce_info)
        raw, _ = _decode_full_analysis_mask(mask_info, dce[0])
        if raw.shape != dce[0].geometry.shape_zyx:
            raise DicomPreparationError("Analysis Mask shape differs from DCE")
        dwi_errors: list[dict[str, Any]] = []
        decode_dwi_candidates(plan, dwi_errors)
        return {
            "patient_id": plan.patient_id,
            "visit": plan.visit,
            "deep_status": "passed",
            "deep_error": "",
            "dwi_deep_errors": dwi_errors,
        }
    except Exception as exc:
        return {
            "patient_id": plan.patient_id,
            "visit": plan.visit,
            "deep_status": "failed",
            "deep_error": f"{type(exc).__name__}: {exc}",
            "dwi_deep_errors": [],
        }


def audit_dicom_root(
    dicom_root: Path | str | Sequence[Path | str], *, deep: bool, workers: int = 1
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be at least one")
    series, discovery_errors = discover_series(dicom_root)
    plans = build_visit_plans(series)
    exclusions: list[dict[str, Any]] = []
    visit_rows: list[dict[str, Any]] = []
    for plan in plans:
        missing = [
            kind for kind in ("dce", "analysis_mask", "ser") if len(plan.series.get(kind, ())) != 1
        ]
        pe_count = sum(
            len(values) for kind, values in plan.series.items() if re.fullmatch(r"pe[0-9]+", kind)
        )
        if pe_count == 0:
            missing.append("pe")
        dwi_groups = _candidate_series(plan)
        complete_dwi_groups = [
            group
            for group, values in dwi_groups.items()
            if "dwi_trace" in values and "dwi_seg" in values
        ]
        row = {
            "patient_id": plan.patient_id,
            "visit": plan.visit,
            "collection": plan.collection,
            "dce_complete": not missing,
            "missing_or_ambiguous_dce_components": missing,
            "dwi_candidate_groups": sorted(dwi_groups),
            "complete_dwi_candidate_groups": complete_dwi_groups,
            "series_counts": {kind: len(values) for kind, values in sorted(plan.series.items())},
            "deep_status": "not_run",
            "deep_error": "",
        }
        if missing:
            exclusions.append(
                {
                    "patient_id": plan.patient_id,
                    "visit": plan.visit,
                    "collection": plan.collection,
                    "scope": "visit",
                    "reason": "missing_or_ambiguous_DCE_components",
                    "details": missing,
                }
            )
        visit_rows.append(row)
    if deep:
        eligible = [
            plan
            for plan, row in zip(plans, visit_rows, strict=True)
            if bool(row["dce_complete"])
        ]
        if workers == 1:
            deep_results = [_deep_audit_plan(plan) for plan in eligible]
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(eligible) or 1)) as executor:
                deep_results = list(executor.map(_deep_audit_plan, eligible))
        row_by_key = {
            (str(row["patient_id"]), str(row["visit"])): row for row in visit_rows
        }
        collection_by_key = {plan.key: plan.collection for plan in plans}
        for result in deep_results:
            key = (str(result["patient_id"]), str(result["visit"]))
            row_by_key[key].update(result)
            if result["deep_status"] == "failed":
                exclusions.append(
                    {
                        "patient_id": key[0],
                        "visit": key[1],
                        "collection": collection_by_key[key],
                        "scope": "visit",
                        "reason": "deep_pixel_audit_failed",
                        "details": result["deep_error"],
                    }
                )
    patients = {plan.patient_id for plan in plans}
    with_t0 = {plan.patient_id for plan in plans if plan.visit == "T0" and not any(
        item["patient_id"] == plan.patient_id and item["visit"] == "T0" for item in exclusions
    )}
    for patient_id in sorted(patients - with_t0):
        collection = next(plan.collection for plan in plans if plan.patient_id == patient_id)
        exclusions.append(
            {
                "patient_id": patient_id,
                "visit": None,
                "collection": collection,
                "scope": "patient",
                "reason": "no_complete_T0",
                "details": "patient cannot be registered to T0",
            }
        )
    return {
        "dicom_roots": [str(path) for path in _roots(dicom_root)],
        "deep": deep,
        "patients": len(patients),
        "visits": len(plans),
        "series": len(series),
        "patients_with_complete_T0": len(with_t0),
        "dce_complete_visits": sum(bool(row["dce_complete"]) for row in visit_rows),
        "dwi_complete_visits": sum(bool(row["complete_dwi_candidate_groups"]) for row in visit_rows),
        "multiple_complete_dwi_visits": sum(
            len(row["complete_dwi_candidate_groups"]) > 1 for row in visit_rows
        ),
        "series_kind_counts": dict(Counter(item.kind for item in series)),
        "discovery_errors": discovery_errors,
        "visit_audit": visit_rows,
        "exclusions": exclusions,
    }


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                        for key, value in row.items()
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return path


def write_audit_outputs(output_root: Path, collection: str, audit: Mapping[str, Any]) -> dict[str, Path]:
    manifest_dir = Path(output_root) / "manifest"
    stem = collection.lower().replace("-", "_")
    json_path = write_json(manifest_dir / f"{stem}_dicom_audit.json", dict(audit))
    visit_path = _write_csv_atomic(
        manifest_dir / f"{stem}_visit_audit.csv", audit.get("visit_audit", [])
    )
    exclusion_path = _write_csv_atomic(
        manifest_dir / f"{stem}_preparation_exclusions.csv", audit.get("exclusions", [])
    )
    return {"audit": json_path, "visits": visit_path, "exclusions": exclusion_path}


def _convert_worker(
    args: tuple[VisitPlan, Path, bool, bool],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    plan, output_root, resume, include_dwi = args
    try:
        return convert_visit(
            plan,
            output_root,
            resume=resume,
            include_dwi=include_dwi,
        ), None
    except Exception as exc:
        return None, {
            "patient_id": plan.patient_id,
            "visit": plan.visit,
            "collection": plan.collection,
            "status": "preparation_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def prepare_dicom_root(
    dicom_root: Path | str | Sequence[Path | str],
    output_root: Path | str,
    *,
    collection: str,
    workers: int = 4,
    resume: bool = False,
    include_dwi: bool = True,
    selected_patients: Iterable[str] | None = None,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be at least one")
    series, discovery_errors = discover_series(dicom_root)
    plans = build_visit_plans(series)
    selected = set(selected_patients or ())
    if selected:
        unknown = selected - {plan.patient_id for plan in plans}
        if unknown:
            raise ValueError(f"unknown patient IDs: {', '.join(sorted(unknown))}")
        plans = [plan for plan in plans if plan.patient_id in selected]
    output = Path(output_root)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = list(discovery_errors)
    jobs = [(plan, output, resume, include_dwi) for plan in plans]
    if workers == 1:
        results = map(_convert_worker, jobs)
        for row, error in results:
            if row is not None:
                rows.append(row)
            if error is not None:
                errors.append(error)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs) or 1)) as executor:
            futures = {executor.submit(_convert_worker, job): job[0].key for job in jobs}
            for future in as_completed(futures):
                row, error = future.result()
                if row is not None:
                    rows.append(row)
                if error is not None:
                    errors.append(error)
    rows.sort(key=lambda row: (str(row["patient_id"]), VISIT_ORDER[str(row["visit"])]))
    errors.sort(key=lambda row: (str(row.get("patient_id", "")), str(row.get("visit", ""))))
    manifest_dir = output / "manifest"
    stem = collection.lower().replace("-", "_")
    manifest_path = _write_csv_atomic(manifest_dir / f"{stem}_preprocessed_manifest.csv", rows)
    error_path = _write_csv_atomic(manifest_dir / f"{stem}_preparation_failures.csv", errors)
    summary = {
        "collection": collection,
        "dicom_roots": [str(path) for path in _roots(dicom_root)],
        "output_root": str(output),
        "planned_visits": len(plans),
        "prepared_visits": len(rows),
        "failed_items": len(errors),
        "dwi_prepared_visits": sum(row["dwi_registration_status"] == "prepared" for row in rows),
        "dwi_failed_visits": sum(
            row["dwi_registration_status"] == "preparation_failed" for row in rows
        ),
        "dwi_deferred_visits": sum(
            row["dwi_registration_status"] == "deferred" for row in rows
        ),
        "resumed_visits": sum(row["preparation_resumed"] == "true" for row in rows),
        "manifest_path": str(manifest_path),
        "failure_path": str(error_path),
    }
    summary_path = write_json(manifest_dir / f"{stem}_preparation_summary.json", summary)
    summary["summary_path"] = str(summary_path)
    return summary


def write_combined_manifest(manifests: Sequence[Path | str], destination: Path | str) -> Path:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for manifest in manifests:
        with Path(manifest).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for name in reader.fieldnames or ():
                if name not in fieldnames:
                    fieldnames.append(name)
            rows.extend(dict(row) for row in reader)
    normalized = [{name: row.get(name, "") for name in fieldnames} for row in rows]
    normalized.sort(key=lambda row: (row.get("patient_id", ""), row.get("visit", "")))
    return _write_csv_atomic(Path(destination), normalized)

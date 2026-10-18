from __future__ import annotations

import copy
import csv
import json
import multiprocessing
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import itk
import matplotlib
import numpy as np
import SimpleITK as sitk

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .config import RegistrationConfig
from .elastix import (
    ElastixRegistrationError,
    bspline_parameter_map,
    estimate_bspline,
)
from .geometry import itk_image_from_array, sitk_image_from_array
from .io import (
    atomic_output_directory,
    pair_is_complete,
    read_nifti,
    write_json,
    write_nifti_identity,
)
from .models import PatientVisits, VisitRecord
from .phase_alignment import (
    IntraVisitRegistrationError,
    PhaseAlignmentResult,
    align_dce_phases,
)
from .preprocess import prepare_pair_drivers
from .qc import (
    QCInputs,
    QCResult,
    QCThresholds,
    evaluate_deformable_qc,
    protected_roi_mask,
)
from .rigid import estimate_rigid
from .transforms import (
    TransformApplicationError,
    apply_transform,
    dicom_grid_transform,
    resample_dce_phases_once,
    displacement_and_jacobian,
    resample_multimodal_once,
    resample_rigid,
    rigid_displacement_and_jacobian,
)


@dataclass(frozen=True)
class PatientPlan:
    patient_id: str
    status: str
    fixed: VisitRecord | None
    moving: tuple[VisitRecord, ...]


@dataclass(frozen=True)
class PairResult:
    patient_id: str
    visit: str
    registration_status: str
    registered_to_visit: str | None
    output_dir: Path | None
    meta_path: Path | None
    transform_path: Path | None
    rigidity_applied: bool
    qc_status: str
    output_files: tuple[Path, ...] = ()
    resumed: bool = False
    message: str = ""


@dataclass(frozen=True)
class PatientResult:
    patient_id: str
    status: str
    reference: PairResult | None
    pairs: tuple[PairResult, ...]
    message: str = ""


@dataclass(frozen=True)
class CohortResult:
    patients: tuple[PatientResult, ...]

    @property
    def failed_pairs(self) -> tuple[PairResult, ...]:
        return tuple(
            pair
            for patient in self.patients
            for pair in patient.pairs
            if pair.registration_status in {"failed", "failed_intra_visit"}
        )

    @property
    def failed_patients(self) -> tuple[PatientResult, ...]:
        return tuple(
            patient
            for patient in self.patients
            if patient.status in {"failed_reference", "worker_failure"}
        )

    @property
    def has_failures(self) -> bool:
        return bool(self.failed_pairs or self.failed_patients)


@dataclass(frozen=True)
class DwiRegistrationResult:
    status: str
    dwi: tuple[np.ndarray, ...] = ()
    adc: np.ndarray | None = None
    mask: np.ndarray | None = None
    grid_transform: sitk.Transform | None = None
    qc: Mapping[str, Any] = field(default_factory=dict)
    message: str = ""


def plan_patient(patient: PatientVisits, config: RegistrationConfig) -> PatientPlan:
    fixed = patient.get(config.fixed_visit)
    if fixed is None:
        return PatientPlan(
            patient_id=patient.patient_id,
            status="skipped_no_T0",
            fixed=None,
            moving=(),
        )
    moving = tuple(
        patient.visits[visit]
        for visit in config.moving_visits
        if visit in patient.visits
    )
    return PatientPlan(
        patient_id=patient.patient_id,
        status="ready",
        fixed=fixed,
        moving=moving,
    )


def _source_paths(record: VisitRecord) -> dict[str, object]:
    return {
        "dce_paths": [str(path) for path in record.dce_paths],
        "mask_path": str(record.ftv_mask_path),
        "raw_mask_path": str(record.raw_mask_path),
        "ser_path": str(record.ser_path),
        "pe_paths": {name: str(path) for name, path in record.pe_paths.items()},
        "meta_path": str(record.meta_path),
        "dwi_paths": [str(path) for path in record.dwi_paths],
        "adc_path": str(record.adc_path) if record.adc_path is not None else None,
        "dwi_mask_path": (
            str(record.dwi_mask_path) if record.dwi_mask_path is not None else None
        ),
    }


def build_output_metadata(
    source: VisitRecord,
    fixed: VisitRecord,
    output_paths: Mapping[str, Any],
    *,
    registration_status: str,
    transform_path: Path | None,
    rigidity_applied: bool,
    qc_payload: Mapping[str, Any],
    ftv_voxel_count: int | None = None,
    dwi_registration_status: str | None = None,
    dwi_transform_path: Path | None = None,
    dwi_qc: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = copy.deepcopy(dict(source.metadata))
    target = fixed.geometry
    dce_paths = [str(path) for path in output_paths["dce"]]
    pe_paths = {name: str(path) for name, path in output_paths["pe"].items()}
    ser_path = str(output_paths["ser"])
    dwi_paths = [str(path) for path in output_paths.get("dwi", ())]
    adc_path_value = output_paths.get("adc")
    dwi_mask_value = output_paths.get("dwi_mask")
    dwi_status = dwi_registration_status or source.dwi_registration_status
    metadata.update(
        {
            "dce_paths": dce_paths,
            "mask_path": str(output_paths["mask"]),
            "raw_mask_path": str(output_paths["raw_mask"]),
            "ser_path": ser_path,
            "pe_paths": pe_paths,
            "collection": source.collection or metadata.get("collection", ""),
            "dwi_paths": dwi_paths,
            "dwi_b_values": list(source.dwi_b_values) if dwi_paths else [],
            "adc_path": str(adc_path_value) if adc_path_value is not None else "",
            "dwi_mask_path": str(dwi_mask_value) if dwi_mask_value is not None else "",
            "dwi_source_group": source.dwi_source_group,
            "dwi_registration_status": dwi_status,
            "derived_paths": {"ser": ser_path, **pe_paths},
            "n_times": len(dce_paths),
            "n_slices": target.shape_zyx[0],
            "rows": target.shape_zyx[1],
            "cols": target.shape_zyx[2],
            "pixel_spacing": [target.spacing_xyz[1], target.spacing_xyz[0]],
            "spacing_between_slices": target.spacing_xyz[2],
            "slice_thickness": fixed.metadata.get(
                "slice_thickness", target.spacing_xyz[2]
            ),
            "image_orientation_patient": list(
                fixed.metadata.get("image_orientation_patient", ())
            ),
            "image_position_patient_first": list(
                fixed.metadata.get("image_position_patient_first", ())
            ),
            "registered_to_visit": fixed.visit,
            "registration_status": registration_status,
            "transform_path": str(transform_path) if transform_path is not None else None,
            "rigidity_applied": rigidity_applied,
            "source_geometry": source.geometry.as_dict(),
            "target_geometry": target.as_dict(),
            "source_paths": _source_paths(source),
            "source_qc_warnings": list(source.qc_warnings),
            "registration": {
                "status": registration_status,
                "registered_to_visit": fixed.visit,
                "transform_path": str(transform_path) if transform_path is not None else None,
                "rigidity_applied": rigidity_applied,
                "qc": dict(qc_payload),
            },
        }
    )
    if ftv_voxel_count is not None:
        metadata["ftv_voxel_count"] = int(ftv_voxel_count)
    dwi_metadata = metadata.get("dwi")
    if isinstance(dwi_metadata, dict):
        dwi_metadata.update(
            {
                "paths": dwi_paths,
                "adc_path": str(adc_path_value) if adc_path_value is not None else None,
                "mask_path": str(dwi_mask_value) if dwi_mask_value is not None else None,
                "registration_status": dwi_status,
                "transform_path": (
                    str(dwi_transform_path) if dwi_transform_path is not None else None
                ),
                "registration_qc": dict(dwi_qc or {}),
                "registered_geometry": (
                    {
                        **target.as_dict(),
                        "origin_xyz": list(
                            fixed.metadata.get("image_position_patient_first", ())
                        ),
                    }
                    if dwi_paths
                    else None
                ),
                "registered_frame_of_reference_uid": (
                    fixed.metadata.get("frame_of_reference_uid") if dwi_paths else None
                ),
            }
        )
    derived_meta = metadata.get("derived_meta")
    if isinstance(derived_meta, dict):
        for name, path in {"ser": ser_path, **pe_paths}.items():
            item = derived_meta.get(name)
            if isinstance(item, dict):
                item["path"] = path
                item["shape_zyx"] = list(target.shape_zyx)
                item["pixel_spacing"] = [target.spacing_xyz[1], target.spacing_xyz[0]]
                item["spacing_between_slices"] = target.spacing_xyz[2]
                item["image_orientation_patient"] = list(
                    fixed.metadata.get("image_orientation_patient", ())
                )
                item["image_position_patient_first"] = list(
                    fixed.metadata.get("image_position_patient_first", ())
                )
    return metadata


def _visit_output_paths(final_dir: Path, record: VisitRecord) -> dict[str, Any]:
    return {
        "dce": [final_dir / "dce" / path.name for path in record.dce_paths],
        "mask": final_dir / "mask" / record.ftv_mask_path.name,
        "raw_mask": final_dir / "raw_mask" / record.raw_mask_path.name,
        "ser": final_dir / "ser" / record.ser_path.name,
        "pe": {
            name: final_dir / name / path.name for name, path in record.pe_paths.items()
        },
        "dwi": [final_dir / "dwi" / path.name for path in record.dwi_paths],
        "adc": (
            final_dir / "adc" / record.adc_path.name
            if record.adc_path is not None
            else None
        ),
        "dwi_mask": (
            final_dir / "dwi_mask" / record.dwi_mask_path.name
            if record.dwi_mask_path is not None
            else None
        ),
    }


def _flatten_output_paths(output_paths: Mapping[str, Any]) -> list[Path]:
    return [
        *output_paths["dce"],
        output_paths["mask"],
        output_paths["raw_mask"],
        output_paths["ser"],
        *output_paths["pe"].values(),
        *output_paths.get("dwi", ()),
        *((output_paths.get("adc"),) if output_paths.get("adc") is not None else ()),
        *((output_paths.get("dwi_mask"),) if output_paths.get("dwi_mask") is not None else ()),
    ]


def _temporary_path(final_path: Path, final_dir: Path, temporary_dir: Path) -> Path:
    return temporary_dir / final_path.relative_to(final_dir)


def _normalized_slice(array: np.ndarray) -> np.ndarray:
    clean = np.asarray(array, dtype=np.float32).copy()
    clean[~np.isfinite(clean)] = 0.0
    low, high = np.percentile(clean, (1.0, 99.0))
    if high <= low:
        return np.zeros(clean.shape, dtype=np.float32)
    return np.clip((clean - low) / (high - low), 0.0, 1.0)


def _phase_overlay(reference: np.ndarray, comparison: np.ndarray) -> np.ndarray:
    fixed = _normalized_slice(reference)
    moving = _normalized_slice(comparison)
    return np.stack((fixed, moving, fixed), axis=-1)


def _write_phase_qc_figure(
    path: Path,
    reference: np.ndarray,
    original: np.ndarray,
    aligned: np.ndarray,
    z_index: int,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(_normalized_slice(reference[z_index]), cmap="gray")
    axes[0].set_title("Phase 0")
    axes[1].imshow(_phase_overlay(reference[z_index], original[z_index]))
    axes[1].set_title("Before")
    axes[2].imshow(_phase_overlay(reference[z_index], aligned[z_index]))
    axes[2].set_title("After rigid")
    for axis in axes:
        axis.set_axis_off()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _write_phase_alignment_artifacts(
    record: VisitRecord,
    alignment: PhaseAlignmentResult,
    final_dir: Path,
    temporary: Path,
) -> tuple[Path, tuple[Path, ...], dict[str, Any]]:
    if alignment.phase_ids != record.dce_phase_ids:
        raise ValueError("phase alignment IDs do not match the visit record")
    if len(alignment.aligned_phases) != len(record.dce_paths):
        raise ValueError("phase alignment arrays do not match DCE paths")
    final_alignment = final_dir / "transforms" / "intravisit" / "phase_alignment.json"
    final_paths: list[Path] = []
    entries: list[dict[str, Any]] = []
    z_slice = alignment.region.slices_zyx[0]
    z_index = int((z_slice.start + z_slice.stop - 1) // 2)
    original_phases = tuple(read_nifti(path, dtype=np.float32) for path in record.dce_paths)
    for registration, original, aligned in zip(
        alignment.registrations,
        original_phases[1:],
        alignment.aligned_phases[1:],
        strict=True,
    ):
        transform_path = (
            final_dir
            / "transforms"
            / "intravisit"
            / f"phase_{registration.phase_id}_to_phase_0.tfm"
        )
        qc_path = (
            final_dir
            / "qc"
            / "intravisit"
            / f"phase_{registration.phase_id}_to_phase_0.png"
        )
        temporary_transform = _temporary_path(transform_path, final_dir, temporary)
        temporary_transform.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteTransform(registration.transform, str(temporary_transform))
        _write_phase_qc_figure(
            _temporary_path(qc_path, final_dir, temporary),
            alignment.aligned_phases[0],
            original,
            aligned,
            z_index,
        )
        final_paths.extend((transform_path, qc_path))
        entries.append(
            {
                "phase_id": registration.phase_id,
                "registered_to_phase_id": 0,
                "transform_path": str(transform_path),
                "qc_figure_path": str(qc_path),
                "metric_value": registration.metric_value,
                "iterations": registration.iterations,
                "stop_condition": registration.stop_condition,
                "sampling_percentage": registration.sampling_percentage,
                "seed": registration.seed,
            }
        )
    payload = {
        "status": "completed",
        "reference_phase_id": 0,
        "phase_ids": list(alignment.phase_ids),
        "used_full_fov": alignment.region.used_full_fov,
        "region_zyx": [
            [axis.start, axis.stop] for axis in alignment.region.slices_zyx
        ],
        "registrations": entries,
    }
    write_json(_temporary_path(final_alignment, final_dir, temporary), payload)
    return final_alignment, tuple((*final_paths, final_alignment)), payload


def _pair_result_from_status(status_path: Path, *, resumed: bool) -> PairResult:
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    transform_value = payload.get("transform_path")
    meta_value = payload.get("meta_path")
    output_dir_value = payload.get("output_dir")
    return PairResult(
        patient_id=payload["patient_id"],
        visit=payload["visit"],
        registration_status=payload["registration_status"],
        registered_to_visit=payload.get("registered_to_visit"),
        output_dir=Path(output_dir_value) if output_dir_value else None,
        meta_path=Path(meta_value) if meta_value else None,
        transform_path=Path(transform_value) if transform_value else None,
        rigidity_applied=bool(payload.get("rigidity_applied", False)),
        qc_status=str(payload.get("qc_status", "")),
        output_files=tuple(Path(path) for path in payload.get("output_files", ())),
        resumed=resumed,
        message=str(payload.get("message", "")),
    )


def _pair_satisfies_requested_modalities(
    final_dir: Path,
    record: VisitRecord,
) -> bool:
    if not pair_is_complete(final_dir):
        return False
    if not record.dwi_paths:
        return True
    try:
        payload = json.loads((final_dir / "registration.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return payload.get("dwi_registration_status") != "deferred"


def _completed_pair_result(
    record: VisitRecord,
    output_root: Path,
    *,
    resume: bool,
) -> PairResult | None:
    if not resume:
        return None
    final_dir = Path(output_root) / record.patient_id / record.visit
    if not _pair_satisfies_requested_modalities(final_dir, record):
        return None
    return _pair_result_from_status(final_dir / "registration.json", resumed=True)


def _dce_dicom_origin(record: VisitRecord) -> tuple[float, float, float]:
    try:
        values = tuple(
            float(value) for value in record.metadata["image_position_patient_first"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{record.patient_id}/{record.visit}: missing DCE DICOM origin"
        ) from exc
    if len(values) != 3 or not np.isfinite(values).all():
        raise ValueError(f"{record.patient_id}/{record.visit}: invalid DCE DICOM origin")
    return values  # type: ignore[return-value]


def _register_dwi_to_t0(
    record: VisitRecord,
    fixed: VisitRecord,
    longitudinal_field: np.ndarray,
    config: RegistrationConfig,
) -> DwiRegistrationResult:
    if not record.dwi_paths:
        return DwiRegistrationResult(
            status=(
                "deferred"
                if record.dwi_registration_status == "deferred"
                else "not_available"
            )
        )
    if (
        record.dwi_geometry is None
        or record.dwi_origin_xyz is None
        or record.dwi_mask_path is None
    ):
        return DwiRegistrationResult(
            status="failed",
            message="DWI paths are present but native geometry or SEG is missing",
        )
    try:
        dwi_arrays = tuple(read_nifti(path, dtype=np.float32) for path in record.dwi_paths)
        dwi_mask = (read_nifti(record.dwi_mask_path) > 0).astype(np.uint8)
        shared_frame = bool(record.dwi_frame_of_reference_uid) and (
            record.dwi_frame_of_reference_uid
            == str(record.metadata.get("frame_of_reference_uid", ""))
        )
        if shared_frame:
            grid_transform: sitk.Transform = dicom_grid_transform(
                record.geometry,
                _dce_dicom_origin(record),
                record.dwi_geometry,
                record.dwi_origin_xyz,
            )
            status = "registered_physical_coordinates"
            rigid_qc: dict[str, Any] = {}
        else:
            dce0 = read_nifti(record.dce_paths[0], dtype=np.float32)
            rigid = estimate_rigid(
                sitk_image_from_array(dce0, record.geometry),
                sitk_image_from_array(dwi_arrays[0], record.dwi_geometry),
                samples=config.rigid_samples,
                iterations=config.rigid_iterations,
                seed=config.seed,
                learning_rate=config.rigid_learning_rate,
                min_step=config.rigid_min_step,
                relaxation_factor=config.rigid_relaxation_factor,
                shrink_factors=config.rigid_shrink_factors,
                smoothing_sigmas=config.rigid_smoothing_sigmas,
            )
            grid_transform = rigid.transform
            status = "registered_rigid_fallback"
            rigid_qc = {
                "metric_value": rigid.metric_value,
                "iterations": rigid.iterations,
                "stop_condition": rigid.stop_condition,
            }
        registered_dwi = tuple(
            resample_multimodal_once(
                array,
                record.dwi_geometry,
                fixed.geometry,
                grid_transform,
                longitudinal_field,
                interpolation="linear",
            )
            for array in dwi_arrays
        )
        registered_adc = (
            resample_multimodal_once(
                read_nifti(record.adc_path, dtype=np.float32),
                record.dwi_geometry,
                fixed.geometry,
                grid_transform,
                longitudinal_field,
                interpolation="linear",
            )
            if record.adc_path is not None
            else None
        )
        registered_mask = resample_multimodal_once(
            dwi_mask,
            record.dwi_geometry,
            fixed.geometry,
            grid_transform,
            longitudinal_field,
            interpolation="nearest",
        )
        if not set(np.unique(registered_mask).astype(int)) <= {0, 1}:
            raise RuntimeError("DWI SEG is not binary after nearest-neighbor resampling")
        if np.any(dwi_mask > 0) and not np.any(registered_mask > 0):
            raise RuntimeError("nonempty source DWI SEG has no overlap with the T0 grid")
        qc = {
            "status": status,
            "shared_frame_of_reference": shared_frame,
            "interpolation": {"dwi": "linear", "adc": "linear", "seg": "nearest"},
            "single_resampling": True,
            "source_geometry": record.dwi_geometry.as_dict(),
            "source_origin_xyz": list(record.dwi_origin_xyz),
            "target_geometry": fixed.geometry.as_dict(),
            "registered_seg_voxel_count": int(np.count_nonzero(registered_mask)),
            "rigid_fallback": rigid_qc,
        }
        return DwiRegistrationResult(
            status=status,
            dwi=registered_dwi,
            adc=registered_adc,
            mask=registered_mask,
            grid_transform=grid_transform,
            qc=qc,
        )
    except Exception as exc:
        return DwiRegistrationResult(
            status="failed",
            qc={"status": "failed", "single_resampling": True},
            message=f"{type(exc).__name__}: {exc}",
        )


def _write_dwi_qc_figure(
    path: Path,
    dce0: np.ndarray,
    dwi0: np.ndarray,
    dwi_mask: np.ndarray,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    center = _slice_indices(dwi_mask)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, index, panel, label in zip(
        range(3), center, axes, ("Axial", "Coronal", "Sagittal"), strict=True
    ):
        fixed_slice = _display_slice(dce0, axis, index)
        dwi_slice = _display_slice(dwi0, axis, index)
        mask_slice = _display_slice(dwi_mask, axis, index)
        panel.imshow(_phase_overlay(fixed_slice, dwi_slice))
        if np.any(mask_slice > 0):
            panel.contour(mask_slice > 0, levels=[0.5], colors=["yellow"], linewidths=0.8)
        panel.set_title(label)
        panel.set_axis_off()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _successful_dwi_paths(
    output_paths: Mapping[str, Any], result: DwiRegistrationResult
) -> dict[str, Any]:
    paths = dict(output_paths)
    if not result.status.startswith("registered_"):
        paths["dwi"] = []
        paths["adc"] = None
        paths["dwi_mask"] = None
    return paths


def _write_dwi_outputs(
    result: DwiRegistrationResult,
    output_paths: Mapping[str, Any],
    final_dir: Path,
    temporary: Path,
    transform_path: Path,
    qc_figure_path: Path,
    dce0: np.ndarray,
) -> tuple[Path, ...]:
    if not result.status.startswith("registered_"):
        return ()
    if result.grid_transform is None or result.mask is None:
        raise RuntimeError("successful DWI registration lacks transform or SEG")
    for array, path in zip(result.dwi, output_paths["dwi"], strict=True):
        write_nifti_identity(
            _temporary_path(path, final_dir, temporary), array.astype(np.float32)
        )
    if result.adc is not None:
        if output_paths["adc"] is None:
            raise RuntimeError("registered ADC has no output path")
        write_nifti_identity(
            _temporary_path(output_paths["adc"], final_dir, temporary),
            result.adc.astype(np.float32),
        )
    write_nifti_identity(
        _temporary_path(output_paths["dwi_mask"], final_dir, temporary),
        result.mask.astype(np.uint8),
    )
    temporary_transform = _temporary_path(transform_path, final_dir, temporary)
    temporary_transform.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(result.grid_transform, str(temporary_transform))
    _write_dwi_qc_figure(
        _temporary_path(qc_figure_path, final_dir, temporary),
        dce0,
        result.dwi[0],
        result.mask,
    )
    return tuple(
        (
            *output_paths["dwi"],
            *((output_paths["adc"],) if output_paths["adc"] is not None else ()),
            output_paths["dwi_mask"],
            transform_path,
            qc_figure_path,
        )
    )


def copy_reference_visit(
    fixed: VisitRecord,
    output_root: Path,
    *,
    phase_alignment: PhaseAlignmentResult,
    resume: bool,
    config: RegistrationConfig | None = None,
) -> PairResult:
    final_dir = Path(output_root) / fixed.patient_id / fixed.visit
    if resume and _pair_satisfies_requested_modalities(final_dir, fixed):
        return _pair_result_from_status(final_dir / "registration.json", resumed=True)
    output_paths = _visit_output_paths(final_dir, fixed)
    final_meta = final_dir / "meta" / fixed.meta_path.name
    final_status = final_dir / "registration.json"
    final_dwi_transform = final_dir / "transforms" / "dwi_to_dce0.tfm"
    final_dwi_qc = final_dir / "qc" / f"{fixed.patient_id}_{fixed.visit}_dwi_to_T0.png"
    identity_field = np.zeros((*fixed.geometry.shape_zyx, 3), dtype=np.float32)
    dwi_result = _register_dwi_to_t0(
        fixed, fixed, identity_field, config or RegistrationConfig()
    )
    published_paths = _successful_dwi_paths(output_paths, dwi_result)
    image_outputs = _flatten_output_paths(published_paths)
    phase_artifact_paths: tuple[Path, ...] = ()
    metadata = build_output_metadata(
        fixed,
        fixed,
        published_paths,
        registration_status="fixed_reference",
        transform_path=None,
        rigidity_applied=False,
        qc_payload={"status": "fixed_reference"},
        ftv_voxel_count=fixed.ftv_voxel_count,
        dwi_registration_status=dwi_result.status,
        dwi_transform_path=(
            final_dwi_transform if dwi_result.status.startswith("registered_") else None
        ),
        dwi_qc=dwi_result.qc | ({"error": dwi_result.message} if dwi_result.message else {}),
    )
    source_and_final = [
        (fixed.dce_paths[0], output_paths["dce"][0]),
        (fixed.ftv_mask_path, output_paths["mask"]),
        (fixed.raw_mask_path, output_paths["raw_mask"]),
        (fixed.ser_path, output_paths["ser"]),
        *(
            (fixed.pe_paths[name], output_paths["pe"][name])
            for name in fixed.pe_paths
        ),
    ]
    status_payload = {
        "patient_id": fixed.patient_id,
        "visit": fixed.visit,
        "registration_status": "fixed_reference",
        "registered_to_visit": fixed.visit,
        "output_dir": str(final_dir),
        "meta_path": str(final_meta),
        "transform_path": None,
        "rigidity_applied": False,
        "qc_status": "fixed_reference",
        "output_files": [],
        "message": "T0 DCE phases aligned to phase 0; derived maps copied",
    }
    with atomic_output_directory(final_dir) as temporary:
        for source_path, final_path in source_and_final:
            destination = _temporary_path(final_path, final_dir, temporary)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
        for aligned, final_path in zip(
            phase_alignment.aligned_phases[1:], output_paths["dce"][1:], strict=True
        ):
            write_nifti_identity(
                _temporary_path(final_path, final_dir, temporary),
                np.asarray(aligned, dtype=np.float32),
            )
        dwi_artifact_paths = _write_dwi_outputs(
            dwi_result,
            published_paths,
            final_dir,
            temporary,
            final_dwi_transform,
            final_dwi_qc,
            np.asarray(phase_alignment.aligned_phases[0], dtype=np.float32),
        )
        phase_alignment_path, phase_artifact_paths, phase_payload = (
            _write_phase_alignment_artifacts(
                fixed, phase_alignment, final_dir, temporary
            )
        )
        metadata["intra_visit_registration"] = phase_payload
        metadata["intra_visit_registration_path"] = str(phase_alignment_path)
        metadata["registration"]["intra_visit_registration_path"] = str(
            phase_alignment_path
        )
        output_files = tuple(
            dict.fromkeys(
                (*image_outputs, *dwi_artifact_paths, *phase_artifact_paths, final_meta, final_status)
            )
        )
        status_payload["intra_visit_registration_path"] = str(phase_alignment_path)
        status_payload["output_files"] = [str(path) for path in output_files]
        status_payload["dwi_registration_status"] = dwi_result.status
        status_payload["dwi_message"] = dwi_result.message
        write_json(_temporary_path(final_meta, final_dir, temporary), metadata)
        write_json(_temporary_path(final_status, final_dir, temporary), status_payload)
        temporary_files = [
            _temporary_path(path, final_dir, temporary) for path in output_files
        ]
        if not all(path.is_file() for path in temporary_files):
            raise RuntimeError(f"incomplete T0 copy for {fixed.patient_id}")
    if not pair_is_complete(final_dir):
        raise RuntimeError(f"published T0 output is incomplete for {fixed.patient_id}")
    return _pair_result_from_status(final_status, resumed=False)


def _qc_payload(result: QCResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "qc_status": result.qc_status,
        "accepted": result.accepted,
        "reasons": list(result.reasons),
        "qc_warning_codes": list(result.qc_warning_codes),
        "rigid_non_tumor_ncc": result.rigid_non_tumor_ncc,
        "deformable_non_tumor_ncc": result.deformable_non_tumor_ncc,
        "rigid_protected_ncc": result.rigid_protected_ncc,
        "deformable_protected_ncc": result.deformable_protected_ncc,
        "jacobian_min": result.jacobian_min,
        "jacobian_max": result.jacobian_max,
        "protected_roi_fold_fraction": result.protected_roi_fold_fraction,
        "outside_protected_roi_fold_fraction": (
            result.outside_protected_roi_fold_fraction
        ),
        "global_fold_fraction": result.global_fold_fraction,
        "candidate_outside_protected_roi_folding": (
            result.candidate_outside_protected_roi_folding
        ),
        "outside_protected_roi_folding_accepted": (
            result.outside_protected_roi_folding_accepted
        ),
        "ftv_volume_change": result.ftv_volume_change,
    }


def _write_transform_parameter_files(transform_parameters, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    count = transform_parameters.GetNumberOfParameterMaps()
    for index in range(count):
        filename = (
            "InitialTransformParameters.0.txt"
            if index == 0
            else f"TransformParameters.{index - 1}.txt"
        )
        path = directory / filename
        itk.ParameterObject.WriteParameterFile(
            transform_parameters.GetParameterMap(index), str(path)
        )
        paths.append(path)
    return paths


def _write_requested_bspline_parameters(
    parameter_map: Mapping[str, list[str]], path: Path
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    parameter_object = itk.ParameterObject.New()
    parameter_object.SetParameterMap(dict(parameter_map))
    parameter_object.WriteParameterFile(str(path))
    return path


def _slice_indices(mask: np.ndarray) -> tuple[int, int, int]:
    coordinates = np.argwhere(np.asarray(mask) > 0)
    if coordinates.size:
        center = np.rint(np.mean(coordinates, axis=0)).astype(int)
    else:
        center = np.asarray(mask.shape) // 2
    return tuple(int(value) for value in center)


def _display_slice(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return array[index]
    if axis == 1:
        return array[:, index, :]
    return array[:, :, index]


def _write_qc_figure(
    path: Path,
    fixed_image: np.ndarray,
    rigid_image: np.ndarray,
    selected_image: np.ndarray,
    fixed_ftv: np.ndarray,
    selected_ftv: np.ndarray,
    candidate_jacobian: np.ndarray,
    protected_roi: np.ndarray,
    status: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    z_index, y_index, x_index = _slice_indices(fixed_ftv | selected_ftv)
    axes_and_indices = ((0, z_index), (1, y_index), (2, x_index))
    images = (fixed_image, rigid_image, selected_image)
    labels = ("T0", "Rigid", status)
    finite_values = np.concatenate(
        [np.asarray(image)[np.isfinite(image)] for image in images]
    )
    low, high = np.percentile(finite_values, (1, 99))
    if high <= low:
        high = low + 1.0
    figure, axes = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
    for row, (axis, index) in enumerate(axes_and_indices):
        for column, (image, label) in enumerate(zip(images, labels, strict=True)):
            panel = axes[row, column]
            panel.imshow(
                _display_slice(image, axis, index),
                cmap="gray",
                vmin=low,
                vmax=high,
                origin="lower",
            )
            mask = fixed_ftv if column == 0 else selected_ftv
            mask_slice = _display_slice(mask, axis, index)
            if np.any(mask_slice):
                panel.contour(mask_slice, levels=[0.5], colors=["#ff3b30"], linewidths=0.8)
            if row == 0:
                panel.set_title(label, fontsize=10)
            panel.set_axis_off()
        jacobian_panel = axes[row, 3]
        jacobian_slice = _display_slice(candidate_jacobian, axis, index)
        jacobian_panel.imshow(
            np.clip(jacobian_slice, 0.0, 2.0),
            cmap="coolwarm",
            vmin=0.0,
            vmax=2.0,
            origin="lower",
        )
        protected_slice = _display_slice(protected_roi, axis, index)
        if np.any(protected_slice):
            jacobian_panel.contour(
                protected_slice,
                levels=[0.5],
                colors=["#ffd60a"],
                linewidths=0.8,
            )
        folding_slice = jacobian_slice <= 0
        if np.any(folding_slice):
            overlay = np.zeros((*folding_slice.shape, 4), dtype=np.float32)
            overlay[..., 0] = 1.0
            overlay[..., 3] = folding_slice.astype(np.float32) * 0.75
            jacobian_panel.imshow(overlay, origin="lower")
        if row == 0:
            jacobian_panel.set_title("B-spline Jacobian", fontsize=10)
        jacobian_panel.set_axis_off()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _apply_selected(
    array: np.ndarray,
    moving: VisitRecord,
    fixed: VisitRecord,
    rigid_transform: sitk.Transform,
    transform_parameters,
    status: str,
    interpolation: str,
    *,
    default_value: float = 0.0,
) -> np.ndarray:
    if status == "deformable":
        return apply_transform(
            array,
            moving.geometry,
            fixed.geometry,
            transform_parameters,
            interpolation=interpolation,
            default_value=default_value,
        )
    return resample_rigid(
        array,
        moving.geometry,
        fixed.geometry,
        rigid_transform,
        interpolation=interpolation,
        default_value=default_value,
    )


def _align_record_phases(
    record: VisitRecord,
    config: RegistrationConfig,
) -> PhaseAlignmentResult:
    phases = tuple(read_nifti(path, dtype=np.float32) for path in record.dce_paths)
    ftv = (read_nifti(record.ftv_mask_path) > 0).astype(np.uint8)
    return align_dce_phases(
        phases,
        record.dce_phase_ids,
        ftv,
        record.geometry,
        samples=config.rigid_samples,
        iterations=config.rigid_iterations,
        seed=config.seed,
        learning_rate=config.rigid_learning_rate,
        min_step=config.rigid_min_step,
        relaxation_factor=config.rigid_relaxation_factor,
        shrink_factors=config.rigid_shrink_factors,
        smoothing_sigmas=config.rigid_smoothing_sigmas,
    )


def register_pair(
    fixed: VisitRecord,
    moving: VisitRecord,
    output_root: Path,
    config: RegistrationConfig,
    *,
    fixed_phase_alignment: PhaseAlignmentResult | None = None,
    moving_phase_alignment: PhaseAlignmentResult | None = None,
    resume: bool,
) -> PairResult:
    if fixed.patient_id != moving.patient_id:
        raise ValueError("fixed and moving records must belong to the same patient")
    if fixed.visit != config.fixed_visit:
        raise ValueError(f"fixed visit must be {config.fixed_visit}")
    final_dir = Path(output_root) / moving.patient_id / moving.visit
    final_status = final_dir / "registration.json"
    if resume and pair_is_complete(final_dir):
        return _pair_result_from_status(final_status, resumed=True)

    output_paths = _visit_output_paths(final_dir, moving)
    final_meta = final_dir / "meta" / moving.meta_path.name
    final_rigid = final_dir / "transforms" / "rigid.tfm"
    final_combined = final_dir / "transforms" / "combined_transform.json"
    final_parameter_dir = final_dir / "transforms" / "elastix"
    final_requested_parameters = final_parameter_dir / "BSplineParameters.txt"
    final_field = final_dir / "deformation" / "displacement_field.nii.gz"
    final_jacobian = final_dir / "jacobian" / "jacobian.nii.gz"
    final_qc_figure = final_dir / "qc" / f"{moving.patient_id}_{moving.visit}_to_{fixed.visit}.png"
    final_dwi_transform = final_dir / "transforms" / "dwi_to_dce0.tfm"
    final_dwi_qc = final_dir / "qc" / f"{moving.patient_id}_{moving.visit}_dwi_to_T0.png"
    final_phase_alignment = (
        final_dir / "transforms" / "intravisit" / "phase_alignment.json"
    )

    with atomic_output_directory(final_dir) as temporary:
        fixed_phases = tuple(read_nifti(path, dtype=np.float32) for path in fixed.dce_paths)
        moving_phases = tuple(read_nifti(path, dtype=np.float32) for path in moving.dce_paths)
        fixed_ftv = (read_nifti(fixed.ftv_mask_path) > 0).astype(np.uint8)
        moving_ftv = (read_nifti(moving.ftv_mask_path) > 0).astype(np.uint8)
        if fixed_phase_alignment is None:
            fixed_phase_alignment = align_dce_phases(
                fixed_phases,
                fixed.dce_phase_ids,
                fixed_ftv,
                fixed.geometry,
                samples=config.rigid_samples,
                iterations=config.rigid_iterations,
                seed=config.seed,
                learning_rate=config.rigid_learning_rate,
                min_step=config.rigid_min_step,
                relaxation_factor=config.rigid_relaxation_factor,
                shrink_factors=config.rigid_shrink_factors,
                smoothing_sigmas=config.rigid_smoothing_sigmas,
            )
        if moving_phase_alignment is None:
            moving_phase_alignment = align_dce_phases(
                moving_phases,
                moving.dce_phase_ids,
                moving_ftv,
                moving.geometry,
                samples=config.rigid_samples,
                iterations=config.rigid_iterations,
                seed=config.seed,
                learning_rate=config.rigid_learning_rate,
                min_step=config.rigid_min_step,
                relaxation_factor=config.rigid_relaxation_factor,
                shrink_factors=config.rigid_shrink_factors,
                smoothing_sigmas=config.rigid_smoothing_sigmas,
            )
        drivers = prepare_pair_drivers(
            fixed_phase_alignment.aligned_phases,
            moving_phase_alignment.aligned_phases,
            fixed_ftv,
            moving_ftv,
            weight=config.tumor_weight,
        )
        fixed_padded_geometry = fixed.geometry.with_shape(drivers.fixed_rigid.shape)
        moving_padded_geometry = moving.geometry.with_shape(drivers.moving_rigid.shape)
        rigid_result = estimate_rigid(
            sitk_image_from_array(drivers.fixed_rigid, fixed_padded_geometry),
            sitk_image_from_array(drivers.moving_rigid, moving_padded_geometry),
            samples=config.rigid_samples,
            iterations=config.rigid_iterations,
            seed=config.seed,
            learning_rate=config.rigid_learning_rate,
            min_step=config.rigid_min_step,
            relaxation_factor=config.rigid_relaxation_factor,
            shrink_factors=config.rigid_shrink_factors,
            smoothing_sigmas=config.rigid_smoothing_sigmas,
        )
        temporary_rigid = _temporary_path(final_rigid, final_dir, temporary)
        temporary_rigid.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteTransform(rigid_result.transform, str(temporary_rigid))

        temporary_parameter_dir = _temporary_path(
            final_parameter_dir, final_dir, temporary
        )
        requested_rigidity_path = (
            temporary_parameter_dir / "moving_rigidity.mha"
            if drivers.rigidity_applied
            else None
        )
        parameter_map = bspline_parameter_map(
            requested_rigidity_path,
            resolutions=config.bspline_resolutions,
            iterations=config.bspline_iterations,
            spatial_samples=config.bspline_spatial_samples,
            seed=config.seed,
        )
        published_parameter_map = {
            key: list(value) for key, value in parameter_map.items()
        }
        final_rigidity_path = (
            final_parameter_dir / "moving_rigidity.mha"
            if drivers.rigidity_applied
            else None
        )
        if final_rigidity_path is not None:
            published_parameter_map["MovingRigidityImageName"] = [
                str(final_rigidity_path)
            ]
        _write_requested_bspline_parameters(
            published_parameter_map,
            _temporary_path(final_requested_parameters, final_dir, temporary),
        )

        transform_parameters = None
        bspline_result = None
        bspline_failure = ""
        moving_rigidity_image = (
            itk_image_from_array(
                drivers.moving_rigidity.astype(np.float32),
                moving_padded_geometry,
            )
            if drivers.rigidity_applied
            else None
        )
        if moving_rigidity_image is not None and requested_rigidity_path is not None:
            requested_rigidity_path.parent.mkdir(parents=True, exist_ok=True)
            itk.imwrite(
                moving_rigidity_image,
                str(requested_rigidity_path),
                compression=True,
            )
        try:
            bspline_result = estimate_bspline(
                itk_image_from_array(drivers.fixed_bspline, fixed_padded_geometry),
                itk_image_from_array(drivers.moving_bspline, moving_padded_geometry),
                rigid_result.transform,
                output_dir=temporary_parameter_dir,
                threads=config.elastix_threads,
                moving_rigidity=moving_rigidity_image,
                parameter_map=parameter_map,
            )
            transform_parameters = bspline_result.transform_parameters
        except ElastixRegistrationError as exc:
            bspline_failure = str(exc)

        fixed_anatomical = drivers.fixed_anatomical[1:-1]
        moving_anatomical = drivers.moving_anatomical[1:-1]
        rigid_anatomical = resample_rigid(
            moving_anatomical,
            moving.geometry,
            fixed.geometry,
            rigid_result.transform,
            interpolation="linear",
        )
        rigid_ftv = resample_rigid(
            moving_ftv,
            moving.geometry,
            fixed.geometry,
            rigid_result.transform,
            interpolation="nearest",
        )
        deformable_error = ""
        if transform_parameters is not None:
            try:
                deformable_anatomical = apply_transform(
                    moving_anatomical,
                    moving.geometry,
                    fixed.geometry,
                    transform_parameters,
                    interpolation="linear",
                )
                deformable_ftv = apply_transform(
                    moving_ftv,
                    moving.geometry,
                    fixed.geometry,
                    transform_parameters,
                    interpolation="nearest",
                )
                deformable_field, deformable_jacobian = displacement_and_jacobian(
                    moving.geometry,
                    fixed.geometry,
                    transform_parameters,
                )
                qc_inputs = QCInputs(
                    fixed_image=fixed_anatomical,
                    rigid_image=rigid_anatomical,
                    deformable_image=deformable_anatomical,
                    fixed_ftv=fixed_ftv,
                    rigid_ftv=rigid_ftv,
                    deformable_ftv=deformable_ftv,
                    source_moving_ftv=moving_ftv,
                    source_geometry=moving.geometry,
                    target_geometry=fixed.geometry,
                    jacobian=deformable_jacobian,
                )
                qc_result = evaluate_deformable_qc(
                    qc_inputs,
                    QCThresholds(
                        max_ftv_volume_change=config.max_ftv_volume_change,
                        similarity_tolerance=config.similarity_tolerance,
                        protected_margin_mm=config.protected_margin_mm,
                        max_outside_fold_fraction=config.max_outside_fold_fraction,
                    ),
                )
                status = qc_result.status
                qc_status = qc_result.qc_status
                qc_payload = _qc_payload(qc_result)
                candidate_protected_roi = protected_roi_mask(
                    qc_inputs, config.protected_margin_mm
                )
            except (TransformApplicationError, RuntimeError, ValueError) as exc:
                deformable_error = f"{type(exc).__name__}: {exc}"
        if transform_parameters is None or deformable_error:
            status = "rigid_fallback"
            qc_status = status
            deformable_anatomical = rigid_anatomical
            deformable_ftv = rigid_ftv
            deformable_field, deformable_jacobian = rigid_displacement_and_jacobian(
                fixed.geometry, rigid_result.transform
            )
            qc_inputs = QCInputs(
                fixed_image=fixed_anatomical,
                rigid_image=rigid_anatomical,
                deformable_image=deformable_anatomical,
                fixed_ftv=fixed_ftv,
                rigid_ftv=rigid_ftv,
                deformable_ftv=deformable_ftv,
                source_moving_ftv=moving_ftv,
                source_geometry=moving.geometry,
                target_geometry=fixed.geometry,
                jacobian=deformable_jacobian,
            )
            candidate_protected_roi = protected_roi_mask(
                qc_inputs, config.protected_margin_mm
            )
            qc_payload = {
                "status": status,
                "qc_status": qc_status,
                "accepted": False,
                "reasons": [
                    "deformable_output_invalid"
                    if deformable_error
                    else "bspline_failed"
                ],
                "qc_warning_codes": [],
                "bspline_error": deformable_error or bspline_failure,
                "jacobian_min": float(np.min(deformable_jacobian)),
                "jacobian_max": float(np.max(deformable_jacobian)),
                "rigid_non_tumor_ncc": None,
                "deformable_non_tumor_ncc": None,
                "rigid_protected_ncc": None,
                "deformable_protected_ncc": None,
                "protected_roi_fold_fraction": 0.0,
                "outside_protected_roi_fold_fraction": 0.0,
                "global_fold_fraction": 0.0,
                "candidate_outside_protected_roi_folding": False,
                "outside_protected_roi_folding_accepted": False,
                "ftv_volume_change": None,
            }

        if status == "deformable":
            selected_anatomical = deformable_anatomical
            selected_ftv = deformable_ftv
            selected_field = deformable_field
            selected_jacobian = deformable_jacobian
        else:
            selected_anatomical = rigid_anatomical
            selected_ftv = rigid_ftv
            selected_field, selected_jacobian = rigid_displacement_and_jacobian(
                fixed.geometry, rigid_result.transform
            )

        written_parameter_paths: list[Path] = []
        final_parameter_paths: list[Path] = []
        if transform_parameters is not None:
            written_parameter_paths = _write_transform_parameter_files(
                transform_parameters, temporary_parameter_dir
            )
            final_parameter_paths = [
                final_parameter_dir / path.name for path in written_parameter_paths
            ]

        rigidity_applied = bool(
            bspline_result is not None and bspline_result.rigidity_applied
        )
        dwi_result = _register_dwi_to_t0(
            moving, fixed, selected_field, config
        )
        published_paths = _successful_dwi_paths(output_paths, dwi_result)
        combined_payload = {
            "patient_id": moving.patient_id,
            "moving_visit": moving.visit,
            "fixed_visit": fixed.visit,
            "selected_transform": status,
            "rigid_transform_path": str(final_rigid),
            "elastix_parameter_paths": [str(path) for path in final_parameter_paths],
            "requested_bspline_parameters_path": str(final_requested_parameters),
            "composition": "fixed_point_to_moving_point; rigid composed before B-spline output resampling",
            "intra_visit_registration_path": str(final_phase_alignment),
            "dce_composition": (
                "T0 fixed point -> visit phase 0 via longitudinal transform -> "
                "original DCE phase via intra-visit rigid transform"
            ),
            "rigidity_applied": rigidity_applied,
            "dwi_registration_status": dwi_result.status,
            "dwi_grid_transform_path": (
                str(final_dwi_transform)
                if dwi_result.status.startswith("registered_")
                else None
            ),
            "dwi_composition": (
                "T0 fixed point -> visit DCE0 via selected longitudinal field -> "
                "native DWI centered grid via DICOM physical coordinates or b0 rigid fallback"
            ),
        }
        write_json(
            _temporary_path(final_combined, final_dir, temporary), combined_payload
        )

        phase_transforms = {
            registration.phase_id: registration.transform
            for registration in moving_phase_alignment.registrations
        }
        registered_phases = resample_dce_phases_once(
            moving_phases,
            moving.dce_phase_ids,
            phase_transforms,
            moving.geometry,
            fixed.geometry,
            selected_field,
        )
        for registered, final_path in zip(
            registered_phases, output_paths["dce"], strict=True
        ):
            write_nifti_identity(
                _temporary_path(final_path, final_dir, temporary),
                registered.astype(np.float32),
            )
        write_nifti_identity(
            _temporary_path(output_paths["mask"], final_dir, temporary),
            selected_ftv.astype(np.uint8),
        )
        raw_mask = read_nifti(moving.raw_mask_path)
        registered_raw_mask = _apply_selected(
            raw_mask,
            moving,
            fixed,
            rigid_result.transform,
            transform_parameters,
            status,
            "nearest",
            default_value=128,
        )
        source_labels = set(np.unique(raw_mask).astype(int)) | {128}
        output_labels = set(np.unique(registered_raw_mask).astype(int))
        if not output_labels <= source_labels:
            raise RuntimeError("raw analysis mask labels changed during registration")
        write_nifti_identity(
            _temporary_path(output_paths["raw_mask"], final_dir, temporary),
            registered_raw_mask.astype(np.uint8),
        )
        ser = _apply_selected(
            read_nifti(moving.ser_path, dtype=np.float32),
            moving,
            fixed,
            rigid_result.transform,
            transform_parameters,
            status,
            "linear",
        )
        write_nifti_identity(
            _temporary_path(output_paths["ser"], final_dir, temporary),
            ser.astype(np.float32),
        )
        for name, source_path in moving.pe_paths.items():
            pe = _apply_selected(
                read_nifti(source_path, dtype=np.float32),
                moving,
                fixed,
                rigid_result.transform,
                transform_parameters,
                status,
                "linear",
            )
            write_nifti_identity(
                _temporary_path(output_paths["pe"][name], final_dir, temporary),
                pe.astype(np.float32),
            )
        try:
            dwi_artifact_paths = _write_dwi_outputs(
                dwi_result,
                published_paths,
                final_dir,
                temporary,
                final_dwi_transform,
                final_dwi_qc,
                fixed_anatomical,
            )
        except Exception as exc:
            for path in (
                *output_paths.get("dwi", ()),
                *((output_paths.get("adc"),) if output_paths.get("adc") is not None else ()),
                *((output_paths.get("dwi_mask"),) if output_paths.get("dwi_mask") is not None else ()),
                final_dwi_transform,
                final_dwi_qc,
            ):
                _temporary_path(path, final_dir, temporary).unlink(missing_ok=True)
            dwi_result = DwiRegistrationResult(
                status="failed",
                qc={"status": "failed", "single_resampling": True},
                message=f"output_write_failed: {type(exc).__name__}: {exc}",
            )
            published_paths = _successful_dwi_paths(output_paths, dwi_result)
            dwi_artifact_paths = ()
        combined_payload["dwi_registration_status"] = dwi_result.status
        combined_payload["dwi_grid_transform_path"] = (
            str(final_dwi_transform)
            if dwi_result.status.startswith("registered_")
            else None
        )
        write_json(
            _temporary_path(final_combined, final_dir, temporary), combined_payload
        )
        write_nifti_identity(
            _temporary_path(final_field, final_dir, temporary),
            selected_field.astype(np.float32),
        )
        write_nifti_identity(
            _temporary_path(final_jacobian, final_dir, temporary),
            selected_jacobian.astype(np.float32),
        )
        _write_qc_figure(
            _temporary_path(final_qc_figure, final_dir, temporary),
            fixed_anatomical,
            rigid_anatomical,
            selected_anatomical,
            fixed_ftv,
            selected_ftv,
            deformable_jacobian,
            candidate_protected_roi,
            status,
        )
        phase_alignment_path, phase_artifact_paths, phase_payload = (
            _write_phase_alignment_artifacts(
                moving, moving_phase_alignment, final_dir, temporary
            )
        )

        qc_payload["rigid_metric_value"] = rigid_result.metric_value
        qc_payload["rigid_iterations"] = rigid_result.iterations
        qc_payload["rigid_stop_condition"] = rigid_result.stop_condition
        metadata = build_output_metadata(
            moving,
            fixed,
            published_paths,
            registration_status=status,
            transform_path=final_combined,
            rigidity_applied=rigidity_applied,
            qc_payload=qc_payload,
            ftv_voxel_count=int(np.count_nonzero(selected_ftv)),
            dwi_registration_status=dwi_result.status,
            dwi_transform_path=(
                final_dwi_transform
                if dwi_result.status.startswith("registered_")
                else None
            ),
            dwi_qc=dwi_result.qc
            | ({"error": dwi_result.message} if dwi_result.message else {}),
        )
        metadata["registration"].update(
            {
                "rigid_transform_path": str(final_rigid),
                "elastix_parameter_paths": [str(path) for path in final_parameter_paths],
                "requested_bspline_parameters_path": str(final_requested_parameters),
                "displacement_field_path": str(final_field),
                "jacobian_path": str(final_jacobian),
                "qc_figure_path": str(final_qc_figure),
                "intra_visit_registration_path": str(phase_alignment_path),
            }
        )
        metadata["intra_visit_registration"] = phase_payload
        metadata["intra_visit_registration_path"] = str(phase_alignment_path)
        write_json(_temporary_path(final_meta, final_dir, temporary), metadata)

        image_outputs = _flatten_output_paths(published_paths)
        output_files = tuple(
            dict.fromkeys(
                (
                    *image_outputs,
                    *dwi_artifact_paths,
                    final_meta,
                    final_rigid,
                    final_combined,
                    final_requested_parameters,
                    *final_parameter_paths,
                    *((final_rigidity_path,) if final_rigidity_path is not None else ()),
                    final_field,
                    final_jacobian,
                    final_qc_figure,
                    *phase_artifact_paths,
                    final_status,
                )
            )
        )
        status_payload = {
            "patient_id": moving.patient_id,
            "visit": moving.visit,
            "registration_status": status,
            "registered_to_visit": fixed.visit,
            "output_dir": str(final_dir),
            "meta_path": str(final_meta),
            "transform_path": str(final_combined),
            "rigidity_applied": rigidity_applied,
            "qc_status": qc_status,
            "qc": qc_payload,
            "intra_visit_registration_path": str(phase_alignment_path),
            "output_files": [str(path) for path in output_files],
            "message": "",
            "dwi_registration_status": dwi_result.status,
            "dwi_message": dwi_result.message,
        }
        write_json(_temporary_path(final_status, final_dir, temporary), status_payload)
        if not all(
            _temporary_path(path, final_dir, temporary).is_file()
            for path in output_files
        ):
            raise RuntimeError(
                f"incomplete registered output for {moving.patient_id}/{moving.visit}"
            )
    if not pair_is_complete(final_dir):
        raise RuntimeError(
            f"published output is incomplete for {moving.patient_id}/{moving.visit}"
        )
    return _pair_result_from_status(final_status, resumed=False)


def register_patient(
    patient: PatientVisits,
    output_root: Path,
    config: RegistrationConfig,
    *,
    resume: bool,
) -> PatientResult:
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(config.elastix_threads)
    plan = plan_patient(patient, config)
    if plan.status == "skipped_no_T0":
        return PatientResult(
            patient_id=patient.patient_id,
            status=plan.status,
            reference=None,
            pairs=(),
            message=f"{patient.patient_id} has no {config.fixed_visit}",
        )
    assert plan.fixed is not None

    resumed_reference = _completed_pair_result(
        plan.fixed,
        output_root,
        resume=resume,
    )
    resumed_pairs = {
        moving.visit: result
        for moving in plan.moving
        if (
            result := _completed_pair_result(
                moving,
                output_root,
                resume=resume,
            )
        )
        is not None
    }
    pending = tuple(
        moving for moving in plan.moving if moving.visit not in resumed_pairs
    )
    if resumed_reference is not None and not pending:
        return PatientResult(
            patient_id=patient.patient_id,
            status="completed",
            reference=resumed_reference,
            pairs=tuple(resumed_pairs[moving.visit] for moving in plan.moving),
        )

    try:
        fixed_phase_alignment = _align_record_phases(plan.fixed, config)
        reference = (
            resumed_reference
            if resumed_reference is not None
            else copy_reference_visit(
                plan.fixed,
                output_root,
                phase_alignment=fixed_phase_alignment,
                resume=resume,
                config=config,
            )
        )
    except Exception as exc:
        return PatientResult(
            patient_id=patient.patient_id,
            status="failed_reference",
            reference=None,
            pairs=(),
            message=str(exc),
        )
    pair_results: list[PairResult] = []
    for moving in plan.moving:
        if moving.visit in resumed_pairs:
            pair_results.append(resumed_pairs[moving.visit])
            continue
        try:
            moving_phase_alignment = _align_record_phases(moving, config)
        except IntraVisitRegistrationError as exc:
            pair_results.append(
                PairResult(
                    patient_id=moving.patient_id,
                    visit=moving.visit,
                    registration_status="failed_intra_visit",
                    registered_to_visit=plan.fixed.visit,
                    output_dir=None,
                    meta_path=None,
                    transform_path=None,
                    rigidity_applied=False,
                    qc_status="failed_intra_visit",
                    message=str(exc),
                )
            )
            continue
        try:
            pair_results.append(
                register_pair(
                    plan.fixed,
                    moving,
                    output_root,
                    config,
                    fixed_phase_alignment=fixed_phase_alignment,
                    moving_phase_alignment=moving_phase_alignment,
                    resume=resume,
                )
            )
        except Exception as exc:
            pair_results.append(
                PairResult(
                    patient_id=moving.patient_id,
                    visit=moving.visit,
                    registration_status="failed",
                    registered_to_visit=plan.fixed.visit,
                    output_dir=None,
                    meta_path=None,
                    transform_path=None,
                    rigidity_applied=False,
                    qc_status="failed",
                    message=str(exc),
                )
            )
    status = (
        "partial_failure"
        if any(
            result.registration_status in {"failed", "failed_intra_visit"}
            for result in pair_results
        )
        else "completed"
    )
    return PatientResult(
        patient_id=patient.patient_id,
        status=status,
        reference=reference,
        pairs=tuple(pair_results),
    )


def _register_patient_worker(args) -> PatientResult:
    patient, output_root, config, resume = args
    return register_patient(patient, output_root, config, resume=resume)


def register_cohort(
    patients: Mapping[str, PatientVisits],
    output_root: Path,
    config: RegistrationConfig,
    *,
    resume: bool,
    selected_patient_ids: Iterable[str] | None = None,
) -> CohortResult:
    selected = (
        sorted(patients)
        if selected_patient_ids is None
        else sorted(set(selected_patient_ids))
    )
    unknown = [patient_id for patient_id in selected if patient_id not in patients]
    if unknown:
        raise ValueError(f"unknown patient IDs: {', '.join(unknown)}")
    jobs = [
        (patients[patient_id], Path(output_root), config, resume)
        for patient_id in selected
    ]
    if config.workers == 1 or len(jobs) <= 1:
        return CohortResult(
            patients=tuple(_register_patient_worker(job) for job in jobs)
        )
    results: dict[str, PatientResult] = {}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(config.workers, len(jobs)), mp_context=context
    ) as executor:
        future_to_patient = {
            executor.submit(_register_patient_worker, job): job[0].patient_id
            for job in jobs
        }
        for future in as_completed(future_to_patient):
            patient_id = future_to_patient[future]
            try:
                results[patient_id] = future.result()
            except Exception as exc:
                results[patient_id] = PatientResult(
                    patient_id=patient_id,
                    status="worker_failure",
                    reference=None,
                    pairs=(),
                    message=str(exc),
                )
    return CohortResult(patients=tuple(results[patient_id] for patient_id in selected))


def _result_index(
    patient_results: Iterable[PatientResult],
) -> tuple[dict[tuple[str, str], PairResult], dict[str, PatientResult]]:
    pair_results: dict[tuple[str, str], PairResult] = {}
    patients: dict[str, PatientResult] = {}
    for patient in patient_results:
        patients[patient.patient_id] = patient
        if patient.reference is not None:
            pair_results[(patient.patient_id, patient.reference.visit)] = patient.reference
        for pair in patient.pairs:
            pair_results[(patient.patient_id, pair.visit)] = pair
    return pair_results, patients


def _update_manifest_row_from_metadata(
    row: dict[str, str], metadata: Mapping[str, Any]
) -> None:
    dce_paths = [str(path) for path in metadata["dce_paths"]]
    row["dce_paths"] = ";".join(dce_paths)
    row["dce_output_dir"] = str(Path(dce_paths[0]).parent)
    row["mask_path"] = str(metadata["mask_path"])
    row["raw_mask_path"] = str(metadata["raw_mask_path"])
    row["ser_path"] = str(metadata["ser_path"])
    row["pe_paths_json"] = json.dumps(metadata["pe_paths"], sort_keys=True)
    row["collection"] = str(metadata.get("collection", row.get("collection", "")))
    row["dwi_paths"] = ";".join(str(path) for path in metadata.get("dwi_paths", ()))
    row["dwi_b_values_json"] = json.dumps(metadata.get("dwi_b_values", ()))
    row["adc_path"] = str(metadata.get("adc_path", ""))
    row["dwi_mask_path"] = str(metadata.get("dwi_mask_path", ""))
    row["dwi_source_group"] = str(metadata.get("dwi_source_group") or "")
    row["dwi_registration_status"] = str(
        metadata.get("dwi_registration_status", "not_available")
    )
    row["derived_paths_json"] = json.dumps(
        metadata.get("derived_paths", {}), sort_keys=True
    )
    row["n_times"] = str(metadata["n_times"])
    row["n_slices"] = str(metadata["n_slices"])
    row["rows"] = str(metadata["rows"])
    row["cols"] = str(metadata["cols"])
    row["ftv_voxel_count"] = str(metadata.get("ftv_voxel_count", ""))
    for index in range(2, 8):
        key = f"pe{index}"
        column = f"pe{index}_path"
        if column in row:
            row[column] = str(metadata.get("pe_paths", {}).get(key, ""))


def write_output_manifest(
    source_manifest: Path,
    output_manifest: Path,
    patient_results: Iterable[PatientResult],
) -> Path:
    with Path(source_manifest).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        source_fieldnames = list(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    appended = [
        "collection",
        "dwi_paths",
        "dwi_b_values_json",
        "adc_path",
        "dwi_mask_path",
        "dwi_source_group",
        "dwi_registration_status",
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
    fieldnames = source_fieldnames + [name for name in appended if name not in source_fieldnames]
    pair_index, patient_index = _result_index(patient_results)
    for row in rows:
        key = (row["patient_id"], row["visit"])
        pair = pair_index.get(key)
        patient = patient_index.get(row["patient_id"])
        if pair is None:
            if patient is None:
                status = "not_processed"
            elif patient.status == "skipped_no_T0":
                status = "skipped_no_T0"
            elif patient.status in {"failed_reference", "worker_failure"}:
                status = patient.status
            else:
                status = "not_processed"
            row.update(
                {
                    "registration_status": status,
                    "registered_to_visit": "",
                    "transform_path": "",
                    "rigidity_applied": "false",
                    "candidate_outside_protected_roi_folding": "false",
                    "outside_protected_roi_folding_accepted": "false",
                    "protected_roi_fold_fraction": "",
                    "outside_protected_roi_fold_fraction": "",
                    "global_fold_fraction": "",
                    "qc_warning_codes": "",
                    "dwi_registration_status": row.get(
                        "dwi_registration_status", "not_processed"
                    ),
                }
            )
            if "qc_status" in row:
                row["qc_status"] = status
            continue
        if pair.meta_path is not None and pair.meta_path.is_file():
            metadata = json.loads(pair.meta_path.read_text(encoding="utf-8"))
            _update_manifest_row_from_metadata(row, metadata)
            row["meta_path"] = str(pair.meta_path)
            qc = metadata.get("registration", {}).get("qc", {})
        else:
            qc = {}
        row.update(
            {
                "registration_status": pair.registration_status,
                "registered_to_visit": pair.registered_to_visit or "",
                "transform_path": str(pair.transform_path or ""),
                "rigidity_applied": str(pair.rigidity_applied).lower(),
                "candidate_outside_protected_roi_folding": str(
                    bool(qc.get("candidate_outside_protected_roi_folding", False))
                ).lower(),
                "outside_protected_roi_folding_accepted": str(
                    bool(qc.get("outside_protected_roi_folding_accepted", False))
                ).lower(),
                "protected_roi_fold_fraction": str(
                    qc.get("protected_roi_fold_fraction", "")
                ),
                "outside_protected_roi_fold_fraction": str(
                    qc.get("outside_protected_roi_fold_fraction", "")
                ),
                "global_fold_fraction": str(qc.get("global_fold_fraction", "")),
                "qc_warning_codes": ";".join(qc.get("qc_warning_codes", ())),
            }
        )
        if "qc_status" in row:
            row["qc_status"] = pair.qc_status
    destination = Path(output_manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def write_failure_manifest(output_root: Path, result: CohortResult) -> Path:
    path = Path(output_root) / "manifest" / "registration_failures.json"
    failures = [
        {
            "patient_id": pair.patient_id,
            "visit": pair.visit,
            "status": pair.registration_status,
            "message": pair.message,
        }
        for pair in result.failed_pairs
    ]
    failures.extend(
        {
            "patient_id": patient.patient_id,
            "visit": None,
            "status": patient.status,
            "message": patient.message,
        }
        for patient in result.failed_patients
    )
    for patient in result.patients:
        pairs = (
            ((patient.reference,) if patient.reference is not None else ())
            + patient.pairs
        )
        for pair in pairs:
            if pair.meta_path is None or not pair.meta_path.is_file():
                continue
            metadata = json.loads(pair.meta_path.read_text(encoding="utf-8"))
            if metadata.get("dwi_registration_status") == "failed":
                failures.append(
                    {
                        "patient_id": pair.patient_id,
                        "visit": pair.visit,
                        "status": "dwi_registration_failed",
                        "message": metadata.get("dwi", {})
                        .get("registration_qc", {})
                        .get("error", ""),
                    }
                )
    return write_json(path, {"failures": failures, "count": len(failures)})

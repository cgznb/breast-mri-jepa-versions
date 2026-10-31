#!/usr/bin/env python3
"""MAMA-MIA five-fold inference on audited I-SPY1 original DICOM volumes.

The source DICOM files remain immutable. Orientation is a reversible axis
permutation/flip (including preservation of obliquity), and masks are returned
to the exact selected native grid. A completed receipt is committed last.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import fcntl
import gc
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import SimpleITK as sitk
import torch
from scipy import ndimage

SCHEMA = "ispy1_mamamia_native_segmentation_v1"
ORDERED_SCHEMA = "ispy1_mamamia_ordered_original_segmentation_v1"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path("/public_fast_storage/cxx26/autodl/MAM/mamamia_segmentation_runtime_20260913/model")
DEFAULT_NNUNET = REPO / "preprocessing/upstream/nnUNet-MAMA-MIA"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(serializable(value), handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def git_revision(path: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def geometry(image: sitk.Image) -> dict[str, Any]:
    return {
        "shape_zyx": list(reversed(image.GetSize())),
        "spacing_xyz_mm": list(image.GetSpacing()),
        "origin_lps_mm": list(image.GetOrigin()),
        "direction_lps": list(image.GetDirection()),
        "orientation": sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(image.GetDirection()),
    }


def same_grid(left: sitk.Image, right: sitk.Image) -> bool:
    return left.GetSize() == right.GetSize() and all(
        np.allclose(a, b, atol=2e-4, rtol=1e-6)
        for a, b in (
            (left.GetSpacing(), right.GetSpacing()),
            (left.GetOrigin(), right.GetOrigin()),
            (left.GetDirection(), right.GetDirection()),
        )
    )


def write_image(path: Path, image: sitk.Image) -> None:
    temporary = path.with_name(path.name.removesuffix(".nii.gz") + f".{os.getpid()}.tmp.nii.gz")
    sitk.WriteImage(image, str(temporary), True)
    reread = sitk.ReadImage(str(temporary))
    if not same_grid(image, reread) or not np.array_equal(
        sitk.GetArrayViewFromImage(image), sitk.GetArrayViewFromImage(reread)
    ):
        raise ValueError("NIfTI round-trip changed image values or physical geometry")
    temporary.replace(path)


def selected_volume(case: dict[str, Any]) -> dict[str, Any]:
    """Keep the verified first-POST and order-only input contracts distinct."""
    mode = case.get("selection_mode", "verified_first_post")
    if mode == "ordered_original":
        if "first_post" in case:
            raise ValueError("Order-only acquisition must not be labeled first_post")
        selected = case["selected_volume"]
        if selected.get("contrast_state") not in ("unknown", "known_later_post", "verified_post", "verified_pre"):
            raise ValueError("Order-only acquisition requires explicit contrast provenance")
        order = selected.get("acquisition_order_zero_based")
        if type(order) is not int or order < 0 or selected.get("order_label") != f"P{order:02d}":
            raise ValueError("Order-only acquisition requires a consistent P00-based order label")
        if not case.get("parent_case_id"):
            raise ValueError("Order-only acquisition requires its parent examination ID")
        if not selected.get("sop_instance_uids"):
            raise ValueError("Order-only acquisition requires audited source SOP identities")
        return selected
    if mode != "verified_first_post" or "selected_volume" in case:
        raise ValueError("Unknown or mixed acquisition selection contract")
    selected = case["first_post"]
    if selected.get("phase_index_zero_based") != 1:
        raise ValueError("Inventory does not identify acquisition 1 as first postcontrast")
    return selected


def acquisition_provenance(case: dict[str, Any]) -> dict[str, Any]:
    if case.get("selection_mode") != "ordered_original":
        return {}
    return {"selection_mode": "ordered_original", "parent_case_id": case["parent_case_id"],
            "source_role": "ordered_original_acquisition"}


def read_native_dicom(case: dict[str, Any]) -> tuple[sitk.Image, dict[str, Any]]:
    """Decode exactly one selected phase and reconstruct a regular native grid."""
    selected = selected_volume(case)
    ordered_original = case.get("selection_mode") == "ordered_original"
    paths = [Path(p) for p in selected["files"]]
    if len(paths) < 2 or len(set(paths)) != len(paths):
        raise ValueError("A phase requires at least two distinct DICOM slices")
    reference = None
    slices = []
    slopes, intercepts, photometric = [], [], set()
    transform_modes = Counter()
    raw_rescale = {key: set() for key in ("RescaleSlope", "RescaleIntercept")}
    source_stats = []
    expected_sops = selected.get("sop_instance_uids")
    if expected_sops is not None and len(expected_sops) != len(paths):
        raise ValueError("Inventory source SOP/file counts differ")
    if expected_sops is not None and len(set(expected_sops)) != len(expected_sops):
        raise ValueError("Inventory contains duplicate source SOP identities")
    frames = set()
    for file_index, path in enumerate(paths):
        before = file_metadata(path)
        ds = pydicom.dcmread(str(path))
        if int(getattr(ds, "NumberOfFrames", 1)) != 1:
            raise ValueError("Enhanced/multiframe DICOM requires explicit frame selection")
        if str(ds.Modality) != "MR" or int(getattr(ds, "SamplesPerPixel", 1)) != 1:
            raise ValueError("Expected single-channel MR source")
        if "DERIVED" in getattr(ds, "ImageType", []):
            raise ValueError("Derived PE/SER/subtraction images are not source MR inputs")
        if ordered_original:
            if not {"ORIGINAL", "PRIMARY"}.issubset(set(getattr(ds, "ImageType", []))):
                raise ValueError("Order-only inputs require ORIGINAL PRIMARY images")
            if str(getattr(ds, "MRAcquisitionType", "")) != "3D":
                raise ValueError("Order-only inputs require original 3D MR acquisitions")
            if str(getattr(ds, "PatientID", "")) != case["patient_id"]:
                raise ValueError("Selected slice belongs to another patient")
        if expected_sops is not None and str(ds.SOPInstanceUID) != expected_sops[file_index]:
            raise ValueError("Source SOP identity differs from audited inventory")
        frames.add(str(getattr(ds, "FrameOfReferenceUID", "")))
        iop = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
        ipp = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
        spacing = np.asarray(ds.PixelSpacing, dtype=np.float64)
        shape = (int(ds.Rows), int(ds.Columns))
        if (iop.shape, ipp.shape, spacing.shape) != ((6,), (3,), (2,)):
            raise ValueError("Incomplete DICOM geometry")
        if not np.isfinite(np.r_[iop, ipp, spacing]).all() or min(spacing) <= 0:
            raise ValueError("Invalid DICOM geometry")
        direction = np.column_stack((iop[:3], iop[3:], np.cross(iop[:3], iop[3:])))
        if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4):
            raise ValueError("Non-orthonormal DICOM direction")
        series = str(ds.SeriesInstanceUID)
        if series != str(selected["series_uid"]):
            raise ValueError("Selected slice belongs to another series")
        if str(ds.StudyInstanceUID) != str(case["study_uid"]):
            raise ValueError("Selected slice belongs to another study")
        if reference is None:
            reference = (iop, spacing, shape, direction)
        elif shape != reference[2] or not np.allclose(iop, reference[0], atol=1e-5) or not np.allclose(spacing, reference[1], atol=1e-5):
            raise ValueError("Inconsistent slice orientation, matrix or spacing")
        raw_slope = getattr(ds, "RescaleSlope", None)
        raw_intercept = getattr(ds, "RescaleIntercept", None)
        for key in raw_rescale:
            value = getattr(ds, key, None)
            raw_rescale[key].add("absent" if key not in ds else "empty" if value is None or str(value).strip() == "" else str(value))
        if ordered_original and (raw_slope not in (None, "")) != (raw_intercept not in (None, "")):
            raise ValueError("Only one intensity rescale field is defined")
        if ordered_original and raw_slope in (None, "") and raw_intercept in (None, ""):
            # Conventional MR stored magnitude remains a usable model input.
            # This explicitly selects stored pixels; it does not reconstruct
            # missing quantitative calibration or assert DICOM values of 1/0.
            explicitly_empty = "RescaleSlope" in ds or "RescaleIntercept" in ds
            if explicitly_empty and any(key in ds for key in ("ModalityLUTSequence", "RealWorldValueMappingSequence", "PixelValueTransformationSequence")):
                raise ValueError("Undefined rescale with another intensity transform requires explicit review")
            if explicitly_empty and ("PHILIPS" in str(getattr(ds, "Manufacturer", "")).upper() or any(ds.get(tag) is not None and ds[tag].value not in (None, b"", "") for tag in ((0x0029, 0x1010), (0x0029, 0x1020), (0x0029, 0x1110), (0x0029, 0x1120)))):
                raise ValueError("Undefined rescale with private intensity metadata requires explicit review")
            slope, intercept = 1.0, 0.0  # arithmetic identity, not source calibration
            transform_mode = "stored_magnitude_no_rescale_defined"
        else:
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            if not np.isfinite([slope, intercept]).all() or slope <= 0:
                raise ValueError("Invalid intensity rescale")
            transform_mode = "dicom_rescale"
            slopes.append(slope)
            intercepts.append(intercept)
        transform_modes[transform_mode] += 1
        pixels = ds.pixel_array
        if pixels.shape != shape:
            raise ValueError("Decoded DICOM pixel shape differs")
        pixels = pixels.astype(np.float32) * slope + intercept
        slices.append((float(ipp @ direction[:, 2]), ipp, pixels, str(path.resolve()), before))
        photometric.add(str(getattr(ds, "PhotometricInterpretation", "")))
        if before != file_metadata(path):
            raise ValueError("Source DICOM changed during decoding")
    if photometric != {"MONOCHROME2"}:
        raise ValueError(f"Unsupported photometric interpretation: {sorted(photometric)}")
    if len(frames) != 1:
        raise ValueError("Source slices use different physical frames")
    slices.sort(key=lambda row: row[0])
    positions = np.stack([row[1] for row in slices])
    projected = np.array([row[0] for row in slices])
    delta = np.diff(projected)
    slice_spacing = float(np.median(delta))
    if slice_spacing <= 0 or not np.allclose(delta, slice_spacing, atol=0.02, rtol=0.005):
        raise ValueError("Duplicate, missing or irregular spatial slices")
    direction = reference[3]
    expected = positions[0] + (projected - projected[0])[:, None] * direction[:, 2]
    if float(np.max(np.abs(positions - expected))) > 0.05:
        raise ValueError("In-plane slice drift cannot be represented by one regular native grid")
    data = np.stack([row[2] for row in slices])
    if not np.isfinite(data).all() or float(np.ptp(data)) <= 0:
        raise ValueError("Source volume is nonfinite or constant")
    image = sitk.GetImageFromArray(data)
    image.SetSpacing((float(reference[1][1]), float(reference[1][0]), slice_spacing))
    image.SetOrigin(tuple(positions[0]))
    image.SetDirection(tuple(direction.ravel()))
    source_stats = [row[4] for row in slices]
    source = {
        "source_files": [row[3] for row in slices],
        "source_file_metadata": source_stats,
        "pixel_policy": "pydicom stored pixels * per-slice RescaleSlope + RescaleIntercept; float32; no VOI LUT/windowing/inversion; no external normalization",
        "rescale_slope_values": sorted(set(slopes)),
        "rescale_intercept_values": sorted(set(intercepts)),
        "photometric_interpretation": sorted(photometric),
        "frame_of_reference_uid": next(iter(frames)),
        "slice_spacing_max_deviation_mm": float(np.max(np.abs(delta - slice_spacing))),
        "slice_position_max_drift_mm": float(np.max(np.abs(positions - expected))),
        "intensity_min": float(data.min()),
        "intensity_max": float(data.max()),
    }
    if ordered_original:
        if len(transform_modes) > 1:
            raise ValueError("Mixed defined and undefined intensity calibration within a volume")
        source["pixel_policy"] = "Defined DICOM rescale: stored pixels * RescaleSlope + RescaleIntercept; otherwise paired undefined rescale with no alternative transform: explicitly use stored magnitude pixels. float32; no VOI LUT/windowing/inversion or external normalization. Undefined fields are not imputed quantitative calibration."
        source["intensity_transform_modes"] = dict(transform_modes)
        source["raw_rescale_field_values"] = {key: sorted(values) for key, values in raw_rescale.items()}
    return image, source


def orient_input(native: sitk.Image) -> tuple[sitk.Image, dict[str, Any]]:
    direction = np.asarray(native.GetDirection()).reshape(3, 3)
    plane_axis = int(np.argmax(np.abs(direction[:, 2])))
    if plane_axis not in (0, 2):
        raise ValueError("Coronal acquisition requires separate orientation review")
    plane, target = ("sagittal", "PSR") if plane_axis == 0 else ("axial", "LAS")
    oriented = sitk.DICOMOrient(native, target)
    restored = sitk.DICOMOrient(oriented, geometry(native)["orientation"])
    if not same_grid(native, restored) or not np.array_equal(sitk.GetArrayViewFromImage(native), sitk.GetArrayViewFromImage(restored)):
        raise ValueError("Reversible orientation failed to preserve native voxels/grid")
    return oriented, {"acquisition_plane": plane, "target_orientation": target, "spatial_interpolation": False, "obliquity_preserved": True}


def qc_mask(mask: sitk.Image) -> dict[str, Any]:
    data = sitk.GetArrayFromImage(mask)
    if not set(np.unique(data)).issubset({0, 1}):
        raise ValueError("Mask is not binary")
    binary = data > 0
    count = int(binary.sum())
    _, n_components = ndimage.label(binary)
    boundary = np.zeros(binary.shape, dtype=bool)
    for axis in range(3):
        for index in (0, -1):
            selection = [slice(None)] * 3
            selection[axis] = index
            boundary[tuple(selection)] = True
    edges = int(np.count_nonzero(binary & boundary))
    coords = np.where(binary)
    bbox = [[int(c.min()), int(c.max()) + 1] for c in coords] if count else None
    flags = []
    if count == 0:
        flags.append("empty_prediction_requires_review")
    if edges:
        flags.append("prediction_touches_image_boundary")
    if n_components > 1:
        flags.append("multiple_components_preserved")
    return {
        "empty": count == 0,
        "voxel_count": count,
        "volume_ml": float(count * np.prod(mask.GetSpacing()) / 1000),
        "component_count": int(n_components),
        "component_connectivity": 6,
        "bbox_zyx": bbox,
        "edge_voxel_count": edges,
        "review_flags": flags,
        "postprocessing": "all predicted components retained; no pruning",
    }


def write_overlay(path: Path, image: sitk.Image, mask: sitk.Image, case_id: str) -> None:
    # Canonical LPS panels make sagittal and axial acquisitions comparable in QC.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    im = sitk.DICOMOrient(image, "LPS")
    ma = sitk.DICOMOrient(mask, "LPS")
    data, seg = sitk.GetArrayFromImage(im), sitk.GetArrayFromImage(ma)
    nonzero = data[data != 0]
    lo, hi = np.percentile(nonzero if nonzero.size else data, [1, 99.5])
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), facecolor="black")
    for axis, title in enumerate(("Axial", "Coronal", "Sagittal")):
        support = seg.sum(axis=tuple(i for i in range(3) if i != axis))
        index = int(np.argmax(support)) if support.max() else seg.shape[axis] // 2
        panel = np.take(data, index, axis=axis)
        label = np.take(seg, index, axis=axis)
        remaining = [i for i in range(3) if i != axis]
        spacing_zyx = list(reversed(im.GetSpacing()))
        aspect = spacing_zyx[remaining[0]] / spacing_zyx[remaining[1]]
        for row in range(2):
            ax = axes[row, axis]
            ax.imshow(panel, cmap="gray", vmin=lo, vmax=hi, aspect=aspect)
            if row == 1 and label.any():
                ax.imshow(np.ma.masked_where(label == 0, label), cmap="autumn", vmin=0, vmax=1, alpha=0.45, aspect=aspect)
            ax.set_title(f"{title}, index {index}" + (" + mask" if row else ""), color="white")
            ax.axis("off")
    fig.suptitle(case_id + " | native physical frame; display in LPS", color="white")
    fig.tight_layout()
    temporary = path.with_name(path.stem + f".{os.getpid()}.tmp.png")
    fig.savefig(temporary, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    temporary.replace(path)


def load_predictor(args: argparse.Namespace):
    sys.path.insert(0, str(args.nnunet_source.resolve()))
    os.environ.setdefault("nnUNet_compile", "false")
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    plans = read_json(args.model / "plans.json")
    dataset = read_json(args.model / "dataset.json")
    manager = PlansManager(plans)
    config = manager.get_configuration("3d_fullres")
    if list(config.patch_size) != [128] * 3 or list(config.spacing) != [1] * 3:
        raise ValueError("Unexpected MAMA-MIA patch/spacing contract")
    if list(config.normalization_schemes) != ["ZScoreNormalization"] or list(config.use_mask_for_norm) != [False]:
        raise ValueError("Unexpected MAMA-MIA normalization contract")
    if dataset["channel_names"] != {"0": "T1"} or dataset["labels"] != {"background": 0, "tumor": 1}:
        raise ValueError("Unexpected MAMA-MIA channel/label contract")
    network = get_network_from_plans(
        config.network_arch_class_name,
        {**config.network_arch_init_kwargs, "deep_supervision": False},
        config.network_arch_init_kwargs_req_import, 1, 2, allow_init=False,
    )
    parameters = []
    for fold in range(5):
        # Official local author checkpoints are trusted. PyTorch >=2.6 otherwise
        # changes torch.load's default and rejects their saved NumPy metadata.
        checkpoint = torch.load(args.model / f"fold_{fold}/checkpoint_final.pth", map_location="cpu", weights_only=False)
        trainer = checkpoint["trainer_name"]
        config_name = checkpoint.get("init_args", {}).get("configuration")
        axes = tuple(checkpoint["inference_allowed_mirroring_axes"])
        if trainer != "nnUNetTrainer_4000epochs" or config_name != "3d_fullres" or axes != (0, 1, 2):
            raise ValueError(f"Fold {fold} does not match the official inference contract")
        network.load_state_dict(checkpoint["network_weights"], strict=True)
        parameters.append(checkpoint["network_weights"])
        del checkpoint
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        perform_everything_on_device=False, device=torch.device(args.device),
        verbose=False, verbose_preprocessing=False, allow_tqdm=False,
    )
    predictor.manual_initialization(network, manager, config, parameters, dataset, "nnUNetTrainer_4000epochs", (0, 1, 2))
    # The author predictor enables benchmarking, which may temporarily allocate
    # large workspaces. Keep this shared training GPU's allocator bound small.
    torch.backends.cudnn.benchmark = False
    return predictor


def inference_contract(args: argparse.Namespace) -> dict[str, Any]:
    ordered_original = args.selection_mode == "ordered_original"
    contract = {
        "schema": ORDERED_SCHEMA if ordered_original else SCHEMA,
        "model_files": [file_metadata(args.model / name) for name in ("plans.json", "dataset.json", *[f"fold_{i}/checkpoint_final.pth" for i in range(5)])],
        "nnunet_source": str(args.nnunet_source.resolve()),
        "source_version": {
            "runner_git_revision": git_revision(REPO),
            "runner_file": file_metadata(Path(__file__).resolve()),
            "nnunet_git_revision": git_revision(args.nnunet_source),
            "nnunet_files": [file_metadata(args.nnunet_source / relative) for relative in (
                "nnunetv2/inference/predict_from_raw_data.py", "nnunetv2/inference/export_prediction.py",
                "nnunetv2/preprocessing/preprocessors/default_preprocessor.py",
                "nnunetv2/preprocessing/normalization/default_normalization_schemes.py",
                "nnunetv2/utilities/plans_handling/plans_handler.py",
                "nnunetv2/utilities/get_network_from_plans.py",
            )],
        },
        "orientation": {"sagittal": "PSR", "axial": "LAS", "interpolation": "axis permutation/flips only; preserve obliquity"},
        "input": "one complete ORIGINAL PRIMARY 3D MR acquisition; retain temporal order without requiring first-POST identity; DICOM rescale; no external z-score" if ordered_original else "one full first-post MR phase; DICOM rescale; no external z-score",
        "preprocessing": {"implementation": "author DefaultPreprocessor", "nonzero_bbox_crop": True, "normalization": "single-image z-score over all voxels of cropped bounding box", "use_mask_for_norm": False, "spacing_mm": [1, 1, 1]},
        "inference": {"folds": [0, 1, 2, 3, 4], "patch_size": [128, 128, 128], "mirror_axes": [0, 1, 2], "tile_step_size": 0.5, "gaussian_blending": True, "fold_aggregation": "mean logits", "gpu_accumulation": False, "mirroring": "sequential author implementation"},
        "output": "binary all-component mask in the selected acquisition's native grid; no registration or component pruning" if ordered_original else "binary all-component mask in native first-post grid; no registration or component pruning",
    }
    if ordered_original:
        contract["selection_mode"] = "ordered_original"
        contract["selection_policy"] = "Exclude DERIVED and incomplete grids; select original acquisitions from formerly ambiguous/missing-source examinations. Pxx denotes acquisition order only. Never combine phases to fill missing slices."
    return contract


def completed_case(case: dict[str, Any], directory: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    path = directory / "receipt.json"
    if not path.exists():
        return None
    receipt = read_json(path)
    if receipt.get("status") != "complete" or receipt.get("contract") != contract:
        raise ValueError(f"Existing receipt contract/status differs: {case['case_id']}")
    if set(receipt["source_files"]) != set(selected_volume(case)["files"]):
        raise ValueError("Completed case source selection differs")
    if receipt.get("source_phase") != selected_volume(case):
        raise ValueError("Completed case acquisition provenance differs")
    for key, value in acquisition_provenance(case).items():
        if receipt.get(key) != value:
            raise ValueError("Completed case selection mode or parent differs")
    for stat in receipt["source_file_metadata"] + receipt["output_file_metadata"]:
        if file_metadata(Path(stat["path"])) != stat:
            raise ValueError(f"Completed case artifact/source changed: {stat['path']}")
    return receipt


def process_case(case: dict[str, Any], directory: Path, predictor: Any, contract: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
    directory.mkdir(parents=True, exist_ok=True)
    started = now()
    start = time.monotonic()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    write_json(directory / "status.json", {"case_id": case["case_id"], "status": "running", "started_at": started, "stage": "dicom_decode"})
    native, source = read_native_dicom(case)
    oriented, orientation = orient_input(native)
    paths = {key: directory / name for key, name in {
        "native_image": "native_image.nii.gz", "oriented_input": "input_oriented.nii.gz",
        "native_mask": "mask_native.nii.gz", "overlay": "qc.png", "preprocessing": "preprocessing.json",
    }.items()}
    write_image(paths["native_image"], native)
    write_image(paths["oriented_input"], oriented)
    write_json(directory / "status.json", {"case_id": case["case_id"], "status": "running", "started_at": started, "stage": "preprocessing"})
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
    data, _, properties = preprocessor.run_case([str(paths["oriented_input"])], None, predictor.plans_manager, predictor.configuration_manager, predictor.dataset_json)
    if data.shape[0] != 1 or not np.isfinite(data).all():
        raise ValueError("Invalid preprocessed single-channel input")
    write_json(paths["preprocessing"], {"properties": properties, "preprocessed_shape_czyx": list(data.shape), "dtype": str(data.dtype), "normalization_contract": contract["preprocessing"]})
    write_json(directory / "status.json", {"case_id": case["case_id"], "status": "running", "started_at": started, "stage": "five_fold_inference", "preprocessed_shape_czyx": list(data.shape)})
    with torch.inference_mode():
        logits = predictor.predict_logits_from_preprocessed_data(torch.from_numpy(data))
        segmentation = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits, predictor.plans_manager, predictor.configuration_manager,
            predictor.label_manager, properties, return_probabilities=False,
            num_threads_torch=args.cpu_threads,
        )
    del logits, data
    label = sitk.GetImageFromArray(np.asarray(segmentation, dtype=np.uint8))
    if label.GetSize() != oriented.GetSize():
        raise ValueError("Export did not restore oriented input dimensions")
    label.CopyInformation(oriented)
    mask = sitk.DICOMOrient(label, geometry(native)["orientation"])
    if not same_grid(mask, native):
        raise ValueError("Mask failed restoration to native source grid")
    # This path is permutation/flips only; nearest-neighbor interpolation is not
    # needed and no geometry is manufactured by replacing a mismatched header.
    write_image(paths["native_mask"], mask)
    qc = qc_mask(mask)
    if case.get("selection_mode") == "ordered_original":
        qc["review_flags"].append("ordered_original_input_not_verified_first_post")
        if selected_volume(case)["contrast_state"] == "unknown":
            qc["review_flags"].append("contrast_identity_unknown")
    if case.get("normalized_visit") not in ("T0", "baseline"):
        qc["review_flags"].append("postbaseline_visit_model_trained_on_baseline_mri")
    write_overlay(paths["overlay"], native, mask, case["case_id"])
    receipt = {
        "schema": contract["schema"], "case_id": case["case_id"], "patient_id": case["patient_id"],
        "study_uid": case["study_uid"], "normalized_visit": case.get("normalized_visit"),
        "original_visit": case.get("original_visit"), "status": "complete", "started_at": started,
        "completed_at": now(), "elapsed_seconds": time.monotonic() - start,
        "contract": contract, "inventory_provenance": file_metadata(args.inventory), **source,
        **acquisition_provenance(case), "source_phase": selected_volume(case),
        "source_geometry": geometry(native), "input_geometry": geometry(oriented),
        "mask_geometry": geometry(mask), "orientation_transform": orientation,
        "preprocessing_properties": properties, "output_paths": paths,
        "output_file_metadata": [file_metadata(path) for path in paths.values()],
        "qc": qc, "runtime": {"device": args.device, "gpu_memory_fraction": args.gpu_memory_fraction, "torch_version": torch.__version__, "cpu_threads": args.cpu_threads},
    }
    if args.device.startswith("cuda"):
        device = torch.device(args.device)
        receipt["runtime"]["cuda_memory_bytes"] = {
            "peak_allocated": torch.cuda.max_memory_allocated(device),
            "peak_reserved": torch.cuda.max_memory_reserved(device),
            "allocated_at_completion": torch.cuda.memory_allocated(device),
            "reserved_at_completion": torch.cuda.memory_reserved(device),
            "device_total": torch.cuda.get_device_properties(device).total_memory,
        }
    write_json(directory / "receipt.json", receipt)
    write_json(directory / "status.json", {"case_id": case["case_id"], "status": "complete", "completed_at": receipt["completed_at"], "qc": qc})
    return serializable(receipt)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--nnunet-source", type=Path, default=DEFAULT_NNUNET)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpu-memory-fraction", type=float, default=0.04)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--case-id", action="append", default=[], help="Repeat or comma-separate exact case IDs")
    p.add_argument("--patient-id", action="append", default=[], help="Repeat or comma-separate patient IDs")
    p.add_argument("--visit", action="append", default=[], help="Repeat normalized visit(s)")
    p.add_argument("--max-cases", type=int, help="Maximum new cases this invocation; complete cases do not count")
    p.add_argument("--prepare-only", action="store_true", help="Validate native/oriented reconstruction without loading network")
    p.add_argument("--fail-fast", action="store_true")
    return p


def split_options(values: list[str]) -> set[str]:
    return {part for value in values for part in value.split(",") if part}


def main() -> int:
    args = parser().parse_args()
    args.inventory, args.output = args.inventory.resolve(), args.output.resolve()
    args.model, args.nnunet_source = args.model.resolve(), args.nnunet_source.resolve()
    if args.cpu_threads < 1 or not 0 < args.gpu_memory_fraction <= 1 or (args.max_cases is not None and args.max_cases < 1):
        raise ValueError("Invalid resource/case limits")
    torch.set_num_threads(args.cpu_threads)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(args.cpu_threads)
    if args.device.startswith("cuda") and not args.prepare_only:
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA inference but CUDA is unavailable")
        torch.cuda.set_device(torch.device(args.device))
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, torch.device(args.device))
    manifest = read_json(args.inventory)
    all_cases = manifest["cases"]
    if len({case["case_id"] for case in all_cases}) != len(all_cases):
        raise ValueError("Duplicate inventory case IDs")
    case_ids, patients, visits = map(split_options, (args.case_id, args.patient_id, args.visit))
    unknown = case_ids - {case["case_id"] for case in all_cases}
    if unknown:
        raise ValueError(f"Unknown case IDs: {sorted(unknown)}")
    cases = [case for case in all_cases if case["status"] == "ready" and (not case_ids or case["case_id"] in case_ids) and (not patients or case["patient_id"] in patients) and (not visits or case.get("normalized_visit") in visits)]
    if not cases:
        raise ValueError("No ready cases matched selection")
    modes = {case.get("selection_mode", "verified_first_post") for case in cases}
    if len(modes) != 1:
        raise ValueError("Separate verified first-POST and ordered-original runs/output directories")
    args.selection_mode = next(iter(modes))
    for case in cases:
        selected_volume(case)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", case["case_id"]) or case["case_id"] in (".", ".."):
            raise ValueError("Unsafe case identifier")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another inference runner holds this output directory") from None
        contract = inference_contract(args)
        contract_path = args.output / "run_contract.json"
        if contract_path.exists() and read_json(contract_path) != contract:
            raise ValueError("Output directory belongs to a different model/inventory contract")
        write_json(contract_path, contract)
        status = {
            "schema": contract["schema"], "started_at": now(), "status": "running",
            "inventory_provenance": file_metadata(args.inventory),
            "inventory_cases": len(all_cases), "inventory_ready_cases": sum(c["status"] == "ready" for c in all_cases),
            "selected_cases": len(cases), "resumed_complete": [], "completed_this_invocation": [],
            "failed": [], "prepared_only": [], "runtime": {"device": args.device, "gpu_memory_fraction": args.gpu_memory_fraction, "cpu_threads": args.cpu_threads},
        }
        predictor = None
        for case in cases:
            if args.max_cases is not None and len(status["completed_this_invocation"]) + len(status["failed"]) + len(status["prepared_only"]) >= args.max_cases:
                break
            case_id = case["case_id"]
            directory = args.output / "cases" / case_id
            status["active_case"] = case_id
            status["updated_at"] = now()
            write_json(args.output / "run_status.json", status)
            try:
                complete = completed_case(case, directory, contract)
                if complete is not None:
                    status["resumed_complete"].append(case_id)
                    print(json.dumps({"case_id": case_id, "status": "resumed_complete"}), flush=True)
                    continue
                if args.prepare_only:
                    directory.mkdir(parents=True, exist_ok=True)
                    native, source = read_native_dicom(case)
                    oriented, transform = orient_input(native)
                    write_image(directory / "native_image.nii.gz", native)
                    write_image(directory / "input_oriented.nii.gz", oriented)
                    write_json(directory / "preparation.json", {"case_id": case_id, "patient_id": case["patient_id"], "study_uid": case["study_uid"], "status": "prepared_only", "inventory_provenance": file_metadata(args.inventory), **source, **acquisition_provenance(case), "source_phase": selected_volume(case), "source_geometry": geometry(native), "input_geometry": geometry(oriented), "orientation_transform": transform})
                    status["prepared_only"].append(case_id)
                    print(json.dumps({"case_id": case_id, "status": "prepared_only"}), flush=True)
                    continue
                if predictor is None:
                    predictor = load_predictor(args)
                receipt = process_case(case, directory, predictor, contract, args)
                status["completed_this_invocation"].append(case_id)
                print(json.dumps({"case_id": case_id, "status": "complete", "elapsed_seconds": round(receipt["elapsed_seconds"], 2), "qc": receipt["qc"]}), flush=True)
            except Exception as error:
                failure = {"case_id": case_id, "status": "failed", "failed_at": now(), "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
                write_json(directory / "status.json", failure)
                status["failed"].append(failure)
                print(json.dumps(failure), file=sys.stderr, flush=True)
                if args.fail_fast:
                    break
            finally:
                gc.collect()
                if args.device.startswith("cuda") and not args.prepare_only:
                    torch.cuda.empty_cache()
                status["updated_at"] = now()
                write_json(args.output / "run_status.json", status)
        status.pop("active_case", None)
        status["status"] = "failed" if status["failed"] else ("prepared_only" if args.prepare_only else "invocation_complete")
        status["completed_at"] = now()
        status["selected_pending"] = len(cases) - len(status["resumed_complete"]) - len(status["completed_this_invocation"]) - len(status["prepared_only"])
        status["all_selected_complete"] = not args.prepare_only and status["selected_pending"] == 0 and not status["failed"]
        write_json(args.output / "run_status.json", status)
    return 1 if status["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

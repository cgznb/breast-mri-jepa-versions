#!/usr/bin/env python3
"""Independently verify I-SPY1 output voxels against original DICOM geometry.

This checks engineering integrity, not biological segmentation accuracy. It
does not import the conversion/inference implementation or invoke a model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import numpy as np
import pydicom
from scipy import ndimage
import SimpleITK as sitk

DICOM_POSITION_TOLERANCE_MM = 0.051


def require(condition, message):
    if not condition:
        raise ValueError(message)


def grid(image):
    return np.asarray(image.GetOrigin()), np.asarray(image.GetDirection()).reshape(3, 3), np.asarray(image.GetSpacing())


def physical(image, indices_xyz):
    origin, direction, spacing = grid(image)
    return origin + (np.asarray(indices_xyz) * spacing) @ direction.T


def continuous_index(image, points_lps):
    origin, direction, spacing = grid(image)
    return (np.asarray(points_lps) - origin) @ np.linalg.inv(direction).T / spacing


def same_grid(left, right):
    require(left.GetSize() == right.GetSize(), "Image/mask shape mismatch")
    for a, b in zip(grid(left), grid(right)):
        require(np.allclose(a, b, atol=3e-4, rtol=1e-6), "Image/mask physical grid mismatch")


def check_recorded_geometry(image, recorded):
    require(list(reversed(image.GetSize())) == recorded["shape_zyx"], "Receipt shape mismatch")
    for actual, name in [(image.GetSpacing(), "spacing_xyz_mm"), (image.GetOrigin(), "origin_lps_mm"), (image.GetDirection(), "direction_lps")]:
        require(np.allclose(actual, recorded[name], atol=3e-4, rtol=1e-6), "Receipt geometry mismatch: " + name)


def check_files(metadata):
    for item in metadata:
        path = Path(item["path"])
        stat = path.stat()
        require(stat.st_size == item["size_bytes"] and stat.st_mtime_ns == item["mtime_ns"], "Recorded file changed: " + str(path))


def check_source(native, source_files, record):
    data = sitk.GetArrayFromImage(native)
    require(np.isfinite(data).all(), "Native image contains nonfinite voxels")
    require(len(source_files) == data.shape[0], "Source slice count differs from native volume")
    require(len(set(source_files)) == len(source_files), "Repeated source file")
    seen_z, seen_sop = set(), set()
    maximum_coordinate_error = 0.0
    maximum_slice_origin_error = 0.0
    maximum_intensity_error = 0.0
    for source_file in source_files:
        ds = pydicom.dcmread(source_file)
        require(str(ds.StudyInstanceUID) == record["study_uid"], "Source study mismatch")
        if record.get("patient_id"):
            require(str(ds.PatientID) == record["patient_id"], "Source patient mismatch")
        require("ORIGINAL" in ds.ImageType and "DERIVED" not in ds.ImageType, "Source is not original MR")
        require(str(ds.Modality) == "MR" and int(getattr(ds, "NumberOfFrames", 1)) == 1, "Unsupported source modality/frames")
        require(str(ds.SOPInstanceUID) not in seen_sop, "Repeated SOP instance")
        seen_sop.add(str(ds.SOPInstanceUID))
        ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
        iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
        ps = np.asarray(ds.PixelSpacing, dtype=float)
        index = continuous_index(native, ipp)
        z = int(np.rint(index[2]))
        require(0 <= z < data.shape[0] and z not in seen_z, "Duplicated/out-of-grid native source slice")
        # DICOM IPP is rounded per slice. The adapter declares a 0.05 mm
        # regular-grid position allowance; add 0.001 mm only for NIfTI float
        # serialization. A fixed voxel-index tolerance would vary with native
        # spacing and reject valid 0.0035 mm rounding at ~2.1 mm spacing.
        origin_error = float(np.max(np.abs(physical(native, np.array([0, 0, z])) - ipp)))
        maximum_slice_origin_error = max(maximum_slice_origin_error, origin_error)
        require(origin_error <= DICOM_POSITION_TOLERANCE_MM, "DICOM origin differs from nearest native slice grid beyond physical tolerance")
        seen_z.add(z)
        rows, columns = int(ds.Rows), int(ds.Columns)
        require(data.shape[1:] == (rows, columns), "Source matrix shape mismatch")
        # DICOM IOP first vector advances columns, second advances rows.
        xy = np.asarray([[0, 0], [columns - 1, 0], [0, rows - 1], [columns - 1, rows - 1], [(columns - 1) / 2, (rows - 1) / 2]], dtype=float)
        expected = ipp + xy[:, :1] * ps[1] * iop[:3] + xy[:, 1:] * ps[0] * iop[3:]
        actual = physical(native, np.column_stack((xy, np.full(len(xy), z))))
        error = float(np.max(np.abs(expected - actual)))
        maximum_coordinate_error = max(maximum_coordinate_error, error)
        require(error <= DICOM_POSITION_TOLERANCE_MM, "DICOM/native corner physical coordinates differ")
        expected_pixels = ds.pixel_array.astype(np.float64) * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
        actual_pixels = data[z]
        require(np.allclose(expected_pixels, actual_pixels, atol=1e-4, rtol=1e-6), "Native voxel values differ from rescaled DICOM")
        maximum_intensity_error = max(maximum_intensity_error, float(np.max(np.abs(expected_pixels - actual_pixels))))
    require(seen_z == set(range(data.shape[0])), "Native slices do not have one-to-one source coverage")
    return {"source_slices_checked": len(seen_z), "source_voxels_checked": int(data.size), "dicom_native_coordinate_tolerance_mm": DICOM_POSITION_TOLERANCE_MM, "coordinate_error_metric": "maximum_absolute_LPS_component", "maximum_dicom_native_slice_origin_error_mm": maximum_slice_origin_error, "maximum_dicom_native_coordinate_error_mm": maximum_coordinate_error, "maximum_dicom_native_intensity_error": maximum_intensity_error}


def check_orientation(native, oriented):
    require(np.prod(native.GetSize()) == np.prod(oriented.GetSize()), "Orientation changed voxel count")
    size = np.asarray(oriented.GetSize())
    rng = np.random.default_rng(73)
    corners = np.asarray(np.meshgrid(*[(0, n - 1) for n in size], indexing="ij")).reshape(3, -1).T
    sample = np.concatenate((corners, rng.integers(0, size, size=(4096, 3))))
    transformed = continuous_index(native, physical(oriented, sample))
    indices = np.rint(transformed).astype(int)
    require(np.allclose(transformed, indices, atol=5e-4), "Oriented voxel does not map to integer native voxel")
    require(np.all(indices >= 0) and np.all(indices < native.GetSize()), "Orientation maps outside native volume")
    source_data, target_data = sitk.GetArrayFromImage(native), sitk.GetArrayFromImage(oriented)
    require(np.array_equal(target_data[sample[:, 2], sample[:, 1], sample[:, 0]], source_data[indices[:, 2], indices[:, 1], indices[:, 0]]), "Orientation changed physical voxel intensities")
    # Derive the full integer permutation from physical grids, then compare
    # every voxel, rather than relying only on sampled coordinates.
    _, native_direction, native_spacing = grid(native)
    _, oriented_direction, oriented_spacing = grid(oriented)
    matrix = np.diag(1 / native_spacing) @ np.linalg.inv(native_direction) @ oriented_direction @ np.diag(oriented_spacing)
    integer = np.rint(matrix).astype(int)
    require(np.allclose(matrix, integer, atol=1e-5) and np.all(np.abs(integer).sum(axis=0) == 1) and np.all(np.abs(integer).sum(axis=1) == 1), "Orientation includes interpolation/scaling")
    native_axes_for_oriented = np.argmax(np.abs(integer), axis=0)
    # Arrays are z,y,x; perform an independently derived transpose and flips.
    axes = [2 - int(native_axes_for_oriented[j]) for j in (2, 1, 0)]
    expected = source_data.transpose(axes)
    for array_axis, oriented_axis in enumerate((2, 1, 0)):
        if integer[native_axes_for_oriented[oriented_axis], oriented_axis] < 0:
            expected = np.flip(expected, array_axis)
    require(np.array_equal(expected, target_data), "Complete oriented array differs from physically derived native permutation")
    return {"orientation_sampled_physical_points": int(len(sample)), "orientation_all_voxels_equal": True, "native_to_input_voxel_matrix": integer.tolist()}


def verify_case(directory, allow_prepared=False):
    receipt_path = directory / "receipt.json"
    prepared = not receipt_path.exists()
    require(not prepared or allow_prepared, "No completed receipt")
    record = json.loads((directory / ("preparation.json" if prepared else "receipt.json")).read_text())
    require(record["status"] == ("prepared_only" if prepared else "complete"), "Unexpected record status")
    if prepared:
        paths = {"native_image": directory / "native_image.nii.gz", "oriented_input": directory / "input_oriented.nii.gz"}
        # preparation.json intentionally contains no cohort fields.
        header = pydicom.dcmread(record["source_files"][0], stop_before_pixels=True)
        record["study_uid"] = str(header.StudyInstanceUID)
        record["patient_id"] = str(header.PatientID)
    else:
        paths = {key: Path(value) for key, value in record["output_paths"].items()}
        check_files(record["output_file_metadata"])
        require(record["source_phase"]["phase_index_zero_based"] == 1, "Source selection is not first postcontrast")
        require(set(record["source_phase"]["files"]) == set(record["source_files"]), "Receipt source selection differs")
    check_files(record["source_file_metadata"])
    native, oriented = sitk.ReadImage(str(paths["native_image"])), sitk.ReadImage(str(paths["oriented_input"]))
    check_recorded_geometry(native, record["source_geometry"])
    check_recorded_geometry(oriented, record["input_geometry"])
    result = {"case_id": record["case_id"], "status": "preparation_verified" if prepared else "complete_verified", **check_source(native, record["source_files"], record), **check_orientation(native, oriented)}
    if not prepared:
        mask = sitk.ReadImage(str(paths["native_mask"]))
        same_grid(native, mask)
        check_recorded_geometry(mask, record["mask_geometry"])
        data = sitk.GetArrayFromImage(mask)
        require(np.isfinite(data).all() and set(np.unique(data)).issubset({0, 1}), "Mask is not finite binary")
        count = int(np.count_nonzero(data))
        components = int(ndimage.label(data)[1])
        require(count == record["qc"]["voxel_count"], "QC voxel count differs")
        require(components == record["qc"]["component_count"], "QC component count differs")
        require(np.isclose(count * np.prod(mask.GetSpacing()) / 1000, record["qc"]["volume_ml"], atol=1e-5), "QC volume differs")
        result.update({"mask_labels": np.unique(data).tolist(), "mask_voxel_count": count, "mask_volume_ml": record["qc"]["volume_ml"], "mask_component_count": components, "review_flags": record["qc"]["review_flags"]})
    return result


def synthetic_adapter_check():
    """Exercise the actual adapter with asymmetric voxels and oblique frames."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
    import infer

    results = []
    with tempfile.TemporaryDirectory(prefix="ispy1_geometry_check_") as temporary:
        root = Path(temporary)
        theta = np.deg2rad(11)
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        sagittal = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], dtype=float)
        for name, direction in [("axial_oblique", rotation), ("sagittal_oblique", rotation @ sagittal)]:
            directory = root / name
            directory.mkdir()
            study_uid, series_uid = generate_uid(), generate_uid()
            origin = np.array([-132.25, 71.5, 23.75])
            spacing = np.array([0.73, 1.19, 2.31])
            # All voxels have distinct values; left/right flips are detectable.
            values = np.arange(9 * 13 * 17, dtype=np.uint16).reshape(9, 13, 17) + 7
            files = []
            for z in range(values.shape[0]):
                path = directory / f"{z}.dcm"
                meta = FileMetaDataset()
                meta.TransferSyntaxUID = ExplicitVRLittleEndian
                meta.MediaStorageSOPClassUID = MRImageStorage
                meta.MediaStorageSOPInstanceUID = generate_uid()
                meta.ImplementationClassUID = generate_uid()
                ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
                ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
                ds.PatientID, ds.StudyInstanceUID, ds.SeriesInstanceUID = "SYNTHETIC", study_uid, series_uid
                ds.Modality, ds.ImageType = "MR", ["ORIGINAL", "PRIMARY", "OTHER"]
                ds.ImageOrientationPatient = [float(x) for x in np.r_[direction[:, 0], direction[:, 1]]]
                ds.ImagePositionPatient = [float(x) for x in origin + z * spacing[2] * direction[:, 2]]
                ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]
                ds.Rows, ds.Columns = values.shape[1:]
                ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
                ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
                ds.RescaleSlope, ds.RescaleIntercept = 2, -31
                ds.PixelData = values[z].tobytes()
                ds.save_as(path, enforce_file_format=True)
                files.append(str(path))
            case = {"case_id": name, "study_uid": study_uid, "first_post": {"files": files[::-1], "series_uid": series_uid, "phase_index_zero_based": 1}}
            native, source = infer.read_native_dicom(case)
            oriented, _ = infer.orient_input(native)
            # Exercise NIfTI serialization as used by the production adapter.
            for filename, image in [("native.nii.gz", native), ("oriented.nii.gz", oriented)]:
                infer.write_image(directory / filename, image)
            native, oriented = sitk.ReadImage(str(directory / "native.nii.gz")), sitk.ReadImage(str(directory / "oriented.nii.gz"))
            source_result = check_source(native, source["source_files"], {"study_uid": study_uid, "patient_id": "SYNTHETIC"})
            orientation_result = check_orientation(native, oriented)
            corrupt = sitk.Image(native)
            corrupt.SetOrigin(tuple(np.asarray(corrupt.GetOrigin()) + np.array([3.0, 0.0, 0.0])))
            rejected = False
            try:
                check_source(corrupt, source["source_files"], {"study_uid": study_uid})
            except ValueError:
                rejected = True
            require(rejected, "Synthetic verifier failed to reject corrupted physical origin")
            results.append({"case": name, "passed": True, "corrupt_origin_rejected": rejected, **source_result, **orientation_result})
    return {"schema": "ispy1_oblique_adapter_selfcheck_v1", "passed": True, "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Inference output root containing cases/")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--allow-prepared", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--self-test", action="store_true", help="Check the actual adapter on two asymmetric oblique synthetic DICOM stacks; CPU only")
    args = parser.parse_args()
    if args.self_test:
        report = synthetic_adapter_check()
        print(json.dumps(report, indent=2), flush=True)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        return 0
    if args.output is None:
        parser.error("--output is required unless --self-test is used")
    cases = [args.output / "cases" / name for name in args.case_id] if args.case_id else sorted((args.output / "cases").glob("*"))
    results = []
    for directory in cases:
        if not directory.is_dir():
            continue
        if not args.case_id and not (directory / "receipt.json").exists() and not (args.allow_prepared and (directory / "preparation.json").exists()):
            continue
        try:
            result = verify_case(directory, args.allow_prepared)
        except Exception as error:
            result = {"case_id": directory.name, "status": "failed", "error_type": type(error).__name__, "error": str(error)}
        results.append(result)
        print(json.dumps(result), flush=True)
    failed = [r for r in results if r["status"] == "failed"]
    report = {"schema": "ispy1_independent_geometry_validation_v1", "validated_at_utc": datetime.now(timezone.utc).isoformat(), "output_root": str(args.output.resolve()), "scope": "engineering integrity against original DICOM; biological segmentation accuracy requires expert reference", "checked": len(results), "failed": len(failed), "all_checked_passed": bool(results) and not failed, "cases": results}
    path = args.report or args.output / "independent_validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return 0 if results and not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

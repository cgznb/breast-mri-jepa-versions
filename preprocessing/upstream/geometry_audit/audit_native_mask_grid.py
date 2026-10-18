"""Compare T0 masks against saved acquisition metadata, not raw NIfTI geometry."""

import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk


ROOT = Path("/root/autodl-tmp/Datasets/ispy2_first_post_mamamia_segmentation_v1")
SOURCE_MANIFEST = ROOT.parent / "ispy2_first_post_unregistered/manifest.json"
METADATA_ROOTS = {
    "ispy2_conversion_metadata": ROOT.parent
    / "manifest-1781750940386/ISPY2/processed_ispy2_world_phase3_scan",
    "acrin_original_metadata": ROOT.parent
    / "qingyuan_acrin_registered_minimal_20260804/metadata/original_meta",
}
OUT = Path(__file__).resolve().parent


def saved_native_geometry(record, source_record, counts):
    path = Path(record["metadata_source"]).resolve()
    assert path == Path(source_record["metadata_source"]).resolve()
    roots = [name for name, root in METADATA_ROOTS.items() if path.is_relative_to(root)]
    assert len(roots) == 1, "Unexpected original metadata location"
    metadata = json.loads(path.read_text())
    key = (record["patient_id"], record["visit"])
    assert key == (metadata["patient_id"], metadata["visit"])
    assert record["relative_path"] == source_record["relative_path"]
    assert Path(record["source_image"]).resolve() == (
        SOURCE_MANIFEST.parent / source_record["relative_path"]
    ).resolve()
    phase_ids = metadata["dce_phase_ids"]
    assert len(phase_ids) == len(metadata["dce_paths"]) == metadata["n_times"]
    assert phase_ids.count(1) == 1
    phase_name = Path(metadata["dce_paths"][phase_ids.index(1)]).name
    assert phase_name == Path(record["source_image"]).name
    assert phase_name.endswith("_dce_aqc_1.nii.gz")

    # All audited T0 metadata explicitly record slice spacing. Do not replace
    # it with slice thickness, which differs for one examination.
    native = {
        "shape_zyx": [metadata["n_slices"], metadata["rows"], metadata["cols"]],
        "pixel_spacing_yx_mm": metadata["pixel_spacing"],
        "slice_spacing_mm": metadata["spacing_between_slices"],
        "image_orientation_patient": metadata["image_orientation_patient"],
        "image_position_patient_first": metadata["image_position_patient_first"],
    }
    for field, value in native.items():
        for item in (source_record, record):
            assert np.array_equal(item[field], value), (record["case_id"], field)
    size = np.asarray(native["shape_zyx"])
    spacing = np.asarray([*native["pixel_spacing_yx_mm"], native["slice_spacing_mm"]])
    iop = np.asarray(native["image_orientation_patient"])
    origin = np.asarray(native["image_position_patient_first"])
    assert size.shape == (3,) and np.all(size > 0) and np.all(size == size.astype(int))
    assert spacing.shape == (3,) and np.isfinite(spacing).all() and min(spacing) > 0
    assert iop.shape == (6,) and origin.shape == (3,)
    assert np.isfinite(np.r_[iop, origin]).all()
    direction = np.column_stack((iop[:3], iop[3:], np.cross(iop[:3], iop[3:])))
    assert np.allclose(direction.T @ direction, np.eye(3), atol=1e-4)
    if "dce_native_geometry" in metadata:
        redundant = metadata["dce_native_geometry"]
        assert np.array_equal(redundant["shape_zyx"], size)
        assert np.allclose(redundant["spacing_xyz"], spacing[[1, 0, 2]], atol=1e-8, rtol=1e-7)
        assert np.allclose(redundant["origin_xyz"], origin, atol=1e-8, rtol=1e-7)
        assert np.allclose(redundant["direction"], direction.ravel(), atol=1e-8, rtol=1e-7)
        counts["redundant_saved_dce_native_geometry_matches"] += 1
    counts[roots[0]] += 1
    counts["manifest_and_inventory_exactly_match_original_metadata"] += 1
    counts["explicit_spacing_between_slices"] += 1
    counts["slice_spacing_differs_from_slice_thickness"] += (
        metadata["spacing_between_slices"] != metadata["slice_thickness"]
    )
    return native


def expected_lps(record):
    iop = np.asarray(record["image_orientation_patient"], dtype=float)
    direction = np.column_stack((iop[:3], iop[3:], np.cross(iop[:3], iop[3:])))
    spacing = np.array([record["pixel_spacing_yx_mm"][1], record["pixel_spacing_yx_mm"][0], record["slice_spacing_mm"]])
    origin = np.asarray(record["image_position_patient_first"], dtype=float)
    size = np.asarray(record["shape_zyx"])[::-1]
    tiny = sitk.Image([1, 1, 1], sitk.sitkUInt8)
    tiny.SetDirection(direction.ravel().tolist())
    tiny.SetSpacing(spacing.tolist())
    tiny.SetOrigin(origin.tolist())
    orient = sitk.DICOMOrientImageFilter()
    orient.SetDesiredCoordinateOrientation("LPS")
    standard = orient.Execute(tiny)
    order = np.asarray(orient.GetPermuteOrder())
    flips = np.asarray(orient.GetFlipAxes())
    # Obtain the library's permutation from a one-voxel image, then shift the
    # origin using the full image sizes without allocating a full MRI array.
    offset = np.zeros(3)
    offset[order[flips]] = size[order[flips]] - 1
    return {
        "size_xyz": size[order],
        "spacing_xyz": spacing[order],
        "origin_lps": origin + direction @ (offset * spacing),
        "direction_lps": np.asarray(standard.GetDirection()),
        "order": order,
        "flips": flips,
        "native_orientation": sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(direction.ravel().tolist()),
    }


def main():
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    inventory = json.loads((ROOT / "inventory.json").read_text())
    records = [r for r in inventory["records"] if r["visit"] == "T0"]
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    assert source_manifest["registered"] is False
    assert source_manifest["phase_index"] == 1 and source_manifest["array_axis_order"] == "zyx"
    source_records = [r for r in source_manifest["records"] if r["visit"] == "T0"]
    source_index = {(r["patient_id"], r["visit"]): r for r in source_records}
    assert len(source_index) == len(source_records) == len(records) == 1095
    assert {(r["patient_id"], r["visit"]) for r in records} == set(source_index)
    counts = Counter()
    provenance_counts = Counter()
    orientations = Counter()
    examples = {}
    max_errors = Counter()
    for record in records:
        native = saved_native_geometry(
            record, source_index[record["patient_id"], record["visit"]], provenance_counts
        )
        expected = expected_lps(native)
        # The raw header is checked only for array dimensions and the known
        # identity-affine storage contract; expected geometry is already set.
        raw = nib.load(record["source_image"])
        assert tuple(raw.shape) == tuple(native["shape_zyx"])
        assert np.allclose(raw.affine, np.eye(4))
        path = ROOT / "masks" / (record["case_id"] + ".nii.gz")
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        assert np.array_equal(reader.GetSize(), expected["size_xyz"])
        for name, actual in [("spacing_xyz", reader.GetSpacing()), ("origin_lps", reader.GetOrigin()), ("direction_lps", reader.GetDirection())]:
            error = float(np.max(np.abs(np.asarray(actual) - expected[name])))
            max_errors[name] = max(max_errors[name], error)
            assert np.allclose(actual, expected[name], atol=1e-4, rtol=1e-6), (record["case_id"], name)
        assert int(np.prod(reader.GetSize())) == int(np.prod(raw.shape))
        orientation = expected["native_orientation"]
        orientations[orientation] += 1
        counts["grid_matches_after_lps_orientation"] += 1
        permutation = not np.array_equal(expected["order"], [0, 1, 2])
        flip = bool(expected["flips"].any())
        counts["requires_axis_permutation"] += permutation
        counts["requires_axis_flip"] += flip
        counts["requires_any_orientation_change"] += permutation or flip
        mask_zyx = tuple(reversed(reader.GetSize()))
        counts["raw_zyx_shape_equals_lps_mask_zyx_shape"] += tuple(raw.shape) == mask_zyx
        counts["nibabel_raw_shape_equals_nibabel_mask_shape"] += tuple(raw.shape) == tuple(reader.GetSize())
        if orientation not in examples:
            examples[orientation] = {
                "raw_nibabel_shape_stored_zyx": list(raw.shape),
                "mask_sitk_array_shape_zyx": list(mask_zyx),
                "mask_nifti_header_shape_xyz": list(reader.GetSize()),
                "permute_order_xyz": expected["order"].tolist(),
                "flip_axes_after_permutation_xyz": expected["flips"].tolist(),
            }
    report = {
        "scope": "T0 same-examination unregistered first-post aqc1 MRI and its predicted segmentation mask",
        "t0_visits": len(records),
        "geometry_authority": {
            "source": "Saved original per-examination acquisition metadata JSON, opened through metadata_source",
            "source_manifest": str(SOURCE_MANIFEST),
            "metadata_roots": {name: str(path) for name, path in METADATA_ROOTS.items()},
            "source_fields": {
                "shape_zyx": ["n_slices", "rows", "cols"],
                "in_plane_spacing_yx_mm": "pixel_spacing",
                "slice_spacing_mm": "spacing_between_slices",
                "direction": "image_orientation_patient; slice axis is the cross product of the two saved axes",
                "origin_lps_mm": "image_position_patient_first",
            },
            "provenance_counts": dict(provenance_counts),
            "manifest_and_inventory_comparison": "Exact equality for all five geometry fields against the original metadata",
            "raw_mri_nifti_geometry_used_as_authority": False,
            "raw_mri_header_use": "Array shape and identity-affine storage-contract checks only",
            "mask_header_use": "Output grid under test, compared with the saved-metadata-derived LPS grid",
        },
        "counts": dict(counts),
        "native_orientation_counts": dict(orientations),
        "maximum_header_geometry_error": dict(max_errors),
        "examples_by_native_orientation": examples,
        "interpretation": "The masks retain the native first-post grid defined by saved original acquisition metadata after axis permutation/flips into LPS. Expected geometry is calculated directly from the original metadata JSON; its manifest and segmentation-inventory copies are independently checked for exact agreement. Original converted MRI NIfTI arrays are stored ZYX with identity affine, which does not encode true acquisition geometry. NIfTI mask arrays use standard XYZ storage. Direct raw-array multiplication is not generally aligned.",
        "limitations": "This is geometry verification, not segmentation accuracy, cross-phase registration, or cross-visit registration. No image interpolation or new segmentation was performed by this audit.",
    }
    (OUT / "native_t0_mask_grid_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

"""Audit current T0 predictions in the exact historical registered grid."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from nibabel.spaces import vox2out_vox

import audit

sys.path.insert(0, str(audit.ROOT / "MAM/longitudinal_temporal_pillar"))
from src.generated_dce0 import native_geometry_from_meta, strict_grid_reference


def affine_apply(matrix, points):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def centered_affine(geometry):
    direction = np.asarray(geometry["direction"]).reshape(3, 3)
    spacing = np.asarray(geometry["spacing_xyz"])
    matrix = np.eye(4)
    matrix[:3, :3] = direction @ np.diag(spacing)
    matrix[:3, 3] = -direction @ ((np.asarray(geometry["shape_zyx"][::-1]) - 1) * spacing / 2)
    return matrix


def native_affine(record):
    iop = np.asarray(record["image_orientation_patient"])
    direction = np.column_stack((iop[:3], iop[3:], np.cross(iop[:3], iop[3:])))
    matrix = np.eye(4)
    matrix[:3, :3] = direction @ np.diag([record["pixel_spacing_yx_mm"][1], record["pixel_spacing_yx_mm"][0], record["slice_spacing_mm"]])
    matrix[:3, 3] = record["image_position_patient_first"]
    return matrix


def transform_affine(transform):
    if not transform.IsLinear():
        raise ValueError("T0 intravisit mapping must be linear")
    matrix = np.eye(4)
    matrix[:3, 3] = transform.TransformPoint((0, 0, 0))
    for column in range(3):
        matrix[:3, column] = np.asarray(transform.TransformPoint(tuple(np.eye(3)[column]))) - matrix[:3, 3]
    return matrix


def fetch_missing(mappings):
    remote_root = audit.OUT / "retained_t0_phase_transforms"
    missing = []
    for row in mappings:
        relative = Path(row["native_patient_id"]) / "T0/transforms/intravisit/phase_1_to_phase_0.tfm"
        local = Path(row["phase_transform_path"])
        retained = remote_root / relative
        if not local.exists():
            row["phase_transform_path"] = str(retained)
            if not retained.exists():
                missing.append(relative.as_posix())
    if missing:
        remote_root.mkdir(exist_ok=True)
        print(f"Reading {len(missing)} retained T0 phase transforms", flush=True)
        for attempt in range(3):
            try:
                subprocess.run(["rsync", "-aR", "--files-from=-", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=30", "qingyuan:/data1/cxx/organized_v1/03_registered_nifti/", str(remote_root) + "/"], input="\n".join(missing) + "\n", text=True, check=True, timeout=120)
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                if attempt == 2:
                    raise
                print(f"Retrying retained transform read ({attempt + 2}/3)", flush=True)
    assert all(Path(row["phase_transform_path"]).is_file() for row in mappings)


def process(mapping, seg_index, visit_index, plans):
    native_id = mapping["native_patient_id"]
    registered_id = mapping["registered_patient_id"]
    record = seg_index[native_id, "T0"]
    report = audit.read(audit.SEG / "case_reports" / (record["case_id"] + ".json"))
    qc = report["qc"]
    if not qc["tumor_voxels"]:
        return {"patient_id": registered_id, "native_patient_id": native_id, "fold": mapping["fold"], "status": "empty_new_t0_mask"}
    meta = audit.read(Path(mapping["meta_path"]))
    assert meta["registration"]["status"] == "fixed_reference"
    fixed_geometry = meta["target_geometry"]
    geometry = native_geometry_from_meta(meta, fixed_geometry["shape_zyx"])
    # Use nibabel's geometry-only primitive from resample_to_output.
    shape_xyz, strict_affine = vox2out_vox((geometry.shape_zyx[::-1], geometry.affine_ras), voxel_sizes=audit.SPACING[::-1])
    assert tuple(shape_xyz[::-1]) == tuple(plans[registered_id]["input_shape_zyx"])
    strict_to_centered = centered_affine(fixed_geometry) @ np.linalg.inv(geometry.affine_ras) @ strict_affine
    centered_to_lps = native_affine(record) @ np.linalg.inv(centered_affine(meta["source_geometry"]))
    phase = sitk.ReadTransform(mapping["phase_transform_path"])
    phase_affine = transform_affine(phase)
    strict_to_native = centered_to_lps @ phase_affine @ strict_to_centered
    image = sitk.ReadImage(str(audit.SEG / "masks" / (record["case_id"] + ".nii.gz")))
    data = sitk.GetArrayViewFromImage(image)
    coordinates = np.asarray(np.where(data), dtype=float)[::-1].T
    assert len(coordinates) == qc["tumor_voxels"]
    bounds = np.column_stack((coordinates.min(axis=0), coordinates.max(axis=0) + 1))[::-1]
    assert np.array_equal(bounds, qc["bbox_zyx_exclusive"])
    assert np.allclose(image.GetSpacing(), qc["geometry"]["spacing_xyz_mm"])
    assert np.allclose(image.GetDirection(), qc["geometry"]["direction_lps"])
    assert np.allclose(image.GetOrigin(), qc["geometry"]["origin_lps_mm"])
    mask_to_native = np.eye(4)
    mask_to_native[:3, :3] = np.asarray(image.GetDirection()).reshape(3, 3) @ np.diag(image.GetSpacing())
    mask_to_native[:3, 3] = image.GetOrigin()
    mask_to_strict = np.linalg.inv(strict_to_native) @ mask_to_native
    points = affine_apply(mask_to_strict, coordinates)
    half_faces = np.abs(mask_to_strict[:3, :3]).sum(axis=1) / 2
    low, high = points.min(axis=0) - half_faces, points.max(axis=0) + half_faces
    center = (low + high) / 2
    extent = (high - low)[::-1] * audit.SPACING
    probes = np.unique(np.linspace(0, len(coordinates) - 1, 5, dtype=int))
    mapping_error = []
    for index in probes:
        expected = np.asarray(image.TransformContinuousIndexToPhysicalPoint(tuple(coordinates[index])))
        centered = affine_apply(strict_to_centered, points[index:index+1])[0]
        actual = affine_apply(centered_to_lps, np.asarray([phase.TransformPoint(tuple(centered))]))[0]
        mapping_error.append(float(np.max(np.abs(expected - actual))))
    assert max(mapping_error) < 1e-5
    visits = visit_index[registered_id]
    required = []
    required_t0 = None
    legacy_bounds = []
    for visit in visits:
        lo = np.array(json.loads(visit["resampled_bbox_min_zyx_json"]), dtype=float)[::-1] - .5
        hi = np.array(json.loads(visit["resampled_bbox_max_zyx_json"]), dtype=float)[::-1] + .5
        requirement = 2 * np.maximum(np.abs(lo - center), np.abs(hi - center))[::-1] * audit.SPACING
        required.append(requirement)
        legacy_bounds.append((visit["visit"], lo, hi))
        if visit["visit"] == "T0":
            required_t0 = requirement
    assert required_t0 is not None
    sizes = {}
    for margin in audit.MARGINS[1:]:
        size = np.ceil(((extent + 2 * margin) / audit.SPACING + 1 - 1e-8) / 32).astype(int) * 32
        assert np.all((size - 1) * audit.SPACING >= extent + 2 * margin - 1e-6)
        sizes[str(margin)] = size.tolist()
    base = {
        "patient_id": registered_id,
        "native_patient_id": native_id,
        "fold": mapping["fold"],
        "status": "mapped",
        "extent_zyx_mm": extent.tolist(),
        "native_spacing_zyx_mm": list(reversed(image.GetSpacing())),
        "components": qc["connected_components_26"],
        "center_strict_zyx": center[::-1].tolist(),
        "longitudinal_required_zyx_mm": np.maximum(extent, np.max(required, axis=0)).tolist(),
        "legacy_t0_required_zyx_mm": required_t0.tolist(),
        "available_visits": len(visits),
        "sizes_by_margin": sizes,
        "point_mapping_error_mm": max(mapping_error),
        "transform_path": mapping["phase_transform_path"],
        "mask_to_strict_index_xyz": mask_to_strict.tolist(),
        "new_mask_native_voxels": len(coordinates),
    }
    t0_legacy_low, t0_legacy_high = next((a,b) for visit,a,b in legacy_bounds if visit == "T0")
    union_low, union_high = np.minimum(low, t0_legacy_low), np.maximum(high, t0_legacy_high)
    union_center = (union_low + union_high) / 2
    union_extent = (union_high - union_low)[::-1] * audit.SPACING
    union_required = [2 * np.maximum(np.abs(lo - union_center), np.abs(hi - union_center))[::-1] * audit.SPACING for _,lo,hi in legacy_bounds]
    union = {**base, "extent_zyx_mm": union_extent.tolist(), "center_strict_zyx": union_center[::-1].tolist(), "longitudinal_required_zyx_mm": np.maximum(union_extent,np.max(union_required,axis=0)).tolist(), "localization": "T0_only_union_of_new_prediction_and_historical_T0_mask"}
    union["legacy_t0_required_zyx_mm"] = (2 * np.maximum(np.abs(t0_legacy_low - union_center), np.abs(t0_legacy_high - union_center))[::-1] * audit.SPACING).tolist()
    union["sizes_by_margin"] = {str(m): (np.ceil(((union_extent + 2 * m) / audit.SPACING + 1 - 1e-8) / 32).astype(int) * 32).tolist() for m in audit.MARGINS[1:]}
    return base, union


def main():
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    mappings = audit.read(audit.OUT / "registered_new_mask_mapping_inventory.json")["patients"]
    requested_count = len(mappings)
    if "--local-only" in sys.argv:
        for row in mappings:
            retained = audit.OUT / "retained_t0_phase_transforms" / row["native_patient_id"] / "T0/transforms/intravisit/phase_1_to_phase_0.tfm"
            if not Path(row["phase_transform_path"]).exists() and retained.exists():
                row["phase_transform_path"] = str(retained)
        mappings = [r for r in mappings if Path(r["phase_transform_path"]).is_file()]
    else:
        fetch_missing(mappings)
    inventory = audit.read(audit.SEG / "inventory.json")
    seg_index = {(r["patient_id"], r["visit"]): r for r in inventory["records"]}
    with (audit.BUNDLE / "visits.csv").open(newline="") as handle:
        visit_index = defaultdict(list)
        for row in csv.DictReader(handle):
            visit_index[row["patient_id"]].append(row)
    plans = audit.read(audit.BUNDLE / "crop_plans.json")
    new_rows, union_rows, excluded = [], [], []
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = executor.map(lambda mapping: process(mapping, seg_index, visit_index, plans), mappings)
        for index, result in enumerate(results, 1):
            if isinstance(result, tuple):
                new, union = result
                new_rows.append(new)
                union_rows.append(union)
            else:
                excluded.append(result)
            if index % 50 == 0:
                print(f"Mapped and checked {index}/{len(mappings)} registered T0 masks", flush=True)
    groups = {}
    for name, rows in [("registered_new_prediction",new_rows), ("registered_t0_union",union_rows)]:
        for fold in ["train", "val"]:
            groups[name + "_" + fold] = [r for r in rows if r["fold"] == fold]
    candidates = [r for name,rows in groups.items() for r in audit.candidate_rows(name, rows)]
    audit.csv_save("registered_new_mask_candidate_sizes.csv", candidates)
    audit.save("registered_new_mask_patient_geometry.json", {"new_prediction": new_rows, "t0_union": union_rows, "excluded": excluded})
    geometry_checks = []
    for mapping in mappings[::175]:
        meta = audit.read(Path(mapping["meta_path"]))
        geometry = native_geometry_from_meta(meta, meta["target_geometry"]["shape_zyx"])
        actual_shape, actual_affine = strict_grid_reference(geometry, audit.SPACING[::-1])
        shape, affine = vox2out_vox((geometry.shape_zyx[::-1], geometry.affine_ras), voxel_sizes=audit.SPACING[::-1])
        assert tuple(shape[::-1]) == actual_shape
        assert np.array_equal(affine, actual_affine)
        geometry_checks.append({"registered_patient_id": mapping["registered_patient_id"], "existing_strict_grid_helper_exact_match": True})
    summary = {
        "scope": "Existing T0 segmentation predictions mapped into the registered Strict-A grid; follow-up coverage uses historical registered masks, not newly mapped T1-T3 predictions.",
        "selection": "train only; val descriptive",
        "registered_patients_requested": requested_count,
        "registered_patients_with_available_transform": len(mappings),
        "unavailable_transform_count": requested_count - len(mappings),
        "mapped_nonempty_t0": len(new_rows),
        "excluded": excluded,
        "groups": {name: audit.describe(rows) for name, rows in groups.items()},
        "maximum_point_mapping_error_mm": max(r["point_mapping_error_mm"] for r in new_rows),
        "strict_grid_helper_checks": geometry_checks,
        "limitations": ["T0-union statistics are an alternative localization proposal, not a changed training policy.", "New predictions and historical FTV masks have different definitions; their agreement is not a clinical accuracy metric.", "No longitudinal registration was recomputed. No existing dataset, crop, checkpoint, or training process was changed."],
    }
    audit.save("registered_summary.json", summary)
    print(json.dumps({"mapped_nonempty_t0":len(new_rows),"excluded":len(excluded),"groups":{name:len(rows) for name,rows in groups.items()},"max_mapping_error_mm":summary["maximum_point_mapping_error_mm"]}),flush=True)


if __name__ == "__main__":
    main()

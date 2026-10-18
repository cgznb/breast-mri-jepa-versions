"""Assess fixed crops centered on the largest predicted T0 component."""

import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import affine_transform, label

OUT = Path(__file__).resolve().parent
SEG = Path("/root/autodl-tmp/Datasets/ispy2_first_post_mamamia_segmentation_v1")
SPACING = np.array([2.0, .7032, .7032])
SHAPES = ([64, 64, 64], [64, 96, 96], [64, 128, 128], [64, 160, 160], [64, 192, 192], [96, 128, 128],
          [96, 160, 160], [96, 192, 192], [96, 256, 256])


def read(path):
    return json.loads(Path(path).read_text())


def write_clipped_128(rows):
    fields = ["patient_id", "fold", "components", "source_mask_voxels", "largest_component_voxels",
              "center_strict_zyx", "largest_extent_zyx_mm", "largest_fits_64x160x160", "largest_fits_64x192x192"]
    clipped = [row for row in rows if row["largest_fits_64x128x128"] in (False, "False")]
    assert len(clipped) == 22 and all(row["fold"] == "train" for row in clipped)
    with (OUT / "largest_component_64x128x128_clipped_cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(row[key]) if isinstance(row[key], list) else row[key] for key in fields} for row in clipped)


def main():
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    source = read(OUT / "registered_new_mask_patient_geometry.json")
    records = {(r["patient_id"], r["visit"]): r for r in read(SEG / "inventory.json")["records"]}
    results, tied = [], 0
    reverse = np.eye(3)[::-1]
    for index, row in enumerate(source["new_prediction"], 1):
        record = records[row["native_patient_id"], "T0"]
        mask_image = sitk.ReadImage(str(SEG / "masks" / (record["case_id"] + ".nii.gz")))
        data = sitk.GetArrayViewFromImage(mask_image)
        coords = np.argwhere(data)
        assert len(coords) == row["new_mask_native_voxels"]
        roi_start, roi_stop = coords.min(axis=0), coords.max(axis=0) + 1
        region = data[tuple(slice(int(lo), int(hi)) for lo, hi in zip(roi_start, roi_stop))]
        components, component_count = label(region, structure=np.ones((3, 3, 3), dtype=np.uint8))
        assert component_count == row["components"]
        ids = components[tuple((coords - roi_start).T)]
        sizes = np.bincount(ids, minlength=component_count + 1)
        sizes[0] = 0
        largest_id = int(np.argmax(sizes))
        tied += int((sizes == sizes[largest_id]).sum() > 1)
        largest = ids == largest_id
        matrix = np.asarray(row["mask_to_strict_index_xyz"])
        points = (coords[:, ::-1] @ matrix[:3, :3].T + matrix[:3, 3])[:, ::-1]
        half_cell = np.abs(matrix[:3, :3]).sum(axis=1)[::-1] / 2
        all_low, all_high = points.min(axis=0) - half_cell, points.max(axis=0) + half_cell
        assert np.allclose((all_high - all_low) * SPACING, row["extent_zyx_mm"], atol=1e-6, rtol=0)
        assert np.allclose((all_high + all_low) / 2, row["center_strict_zyx"], atol=1e-6, rtol=0)
        major_points = points[largest]
        low, high = major_points.min(axis=0) - half_cell, major_points.max(axis=0) + half_cell
        center, extent = (low + high) / 2, (high - low) * SPACING
        all_requirement = 2 * np.maximum(np.abs(all_low - center), np.abs(all_high - center)) * SPACING
        offset = np.abs(points - center)
        inside = np.all(offset <= np.array(SHAPES[0]) / 2 + 1e-7, axis=1)
        inverse = np.linalg.inv(matrix)
        crop_start_xyz = (center - (np.array(SHAPES[0]) - 1) / 2)[::-1]
        output_to_input = reverse @ inverse[:3, :3] @ reverse
        crop_offset = (inverse[:3, :3] @ crop_start_xyz + inverse[:3, 3])[::-1]
        # Keep native array boundaries so interpolation also preserves voxel faces at the ROI edges.
        full_components = np.zeros(data.shape, dtype=components.dtype)
        full_components[tuple(slice(int(lo), int(hi)) for lo, hi in zip(roi_start, roi_stop))] = components
        cropped = affine_transform(full_components, output_to_input, crop_offset, output_shape=tuple(SHAPES[0]),
                                   order=0, mode="constant", cval=0, prefilter=False)
        result = {"patient_id": row["patient_id"], "native_patient_id": row["native_patient_id"], "fold": row["fold"],
                  "components": component_count, "largest_component_label": largest_id,
                  "source_mask_voxels": len(coords), "largest_component_voxels": int(largest.sum()),
                  "largest_fraction_of_source_mask": float(largest.mean()), "center_strict_zyx": center.tolist(),
                  "largest_extent_zyx_mm": extent.tolist(), "all_mask_required_extent_zyx_mm": all_requirement.tolist(),
                  "largest_source_centers_outside_64cube": int((~inside[largest]).sum()),
                  "largest_outside_fraction_64cube": float((~inside[largest]).mean()),
                  "all_source_centers_outside_64cube": int((~inside).sum()),
                  "all_outside_fraction_64cube": float((~inside).mean()),
                  "actual_cropped_largest_mask_voxels_64cube": int((cropped == largest_id).sum()),
                  "actual_cropped_all_mask_voxels_64cube": int(np.count_nonzero(cropped)),
                  "center_shift_from_whole_bbox_mm": float(np.linalg.norm((center - row["center_strict_zyx"]) * SPACING))}
        for shape in SHAPES:
            key = "x".join(map(str, shape))
            result["largest_fits_" + key] = bool(np.all(extent <= np.asarray(shape) * SPACING + 1e-6))
            result["all_mask_fits_" + key] = bool(np.all(all_requirement <= np.asarray(shape) * SPACING + 1e-6))
        results.append(result)
        if index % 50 == 0:
            print(json.dumps({"checked": index, "total": len(source["new_prediction"])}), flush=True)
    summaries = {}
    for fold in ("train", "val", "all"):
        rows = [r for r in results if fold == "all" or r["fold"] == fold]
        extents = np.asarray([r["largest_extent_zyx_mm"] for r in rows])
        all_extents = np.asarray([r["all_mask_required_extent_zyx_mm"] for r in rows])
        summaries[fold] = {"nonempty_patients": len(rows), "multiple_components": sum(r["components"] > 1 for r in rows),
                           "minimum_32rounded_shape_for_all_largest_regions_no_margin": (np.ceil((extents.max(axis=0) / SPACING - 1e-8) / 32).astype(int) * 32).tolist(),
                           "minimum_32rounded_shape_for_all_original_predictions_no_margin": (np.ceil((all_extents.max(axis=0) / SPACING - 1e-8) / 32).astype(int) * 32).tolist(),
                           "largest_extent_zyx_mm_percentiles": {str(q): np.percentile(extents, q, axis=0).tolist() for q in (50, 90, 95, 99, 100)},
                           "empty_largest_actual_crops_64cube": sum(r["actual_cropped_largest_mask_voxels_64cube"] == 0 for r in rows),
                           "empty_all_mask_actual_crops_64cube": sum(r["actual_cropped_all_mask_voxels_64cube"] == 0 for r in rows),
                           "largest_loses_over_half_64cube": sum(r["largest_outside_fraction_64cube"] > .5 for r in rows),
                           "shapes": []}
        for shape in SHAPES:
            key = "x".join(map(str, shape))
            largest_fits = sum(r["largest_fits_" + key] for r in rows)
            all_fits = sum(r["all_mask_fits_" + key] for r in rows)
            summaries[fold]["shapes"].append({"shape_zyx": shape, "largest_fully_contained": largest_fits,
                                              "largest_truncated": len(rows) - largest_fits,
                                              "largest_fully_contained_percent": largest_fits / len(rows) * 100,
                                              "all_predictions_fully_contained": all_fits,
                                              "all_predictions_truncated": len(rows) - all_fits})
    assert len(results) == 866
    with (OUT / "largest_component_crop_sizes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()} for row in results)
    write_clipped_128(results)
    report = {"scope": "All 866 nonempty model T0 masks in the registered historical SymmFlow cohort",
              "spacing_zyx_mm": SPACING.tolist(), "64cube_physical_extent_zyx_mm": (np.array(SHAPES[0]) * SPACING).tolist(),
              "center": "Largest 26-connected native model-mask component mapped into registered Strict-A voxel-cell bbox center",
              "largest_component_ties": tied, "tie_rule": "Lowest native connected-component label",
              "geometry": "Previously verified mapping from saved real metadata and retained T0 phase transforms; source mask counts, full bounds and centers rechecked for every mask",
              "containment": "Voxel-cell support within actual crop voxel faces; no added margin",
              "components_removed_from_source_data": False, "all_866_actual_64cube_masks_resampled": True,
              "originally_empty_predictions_unassessable": len(source["excluded"]), "summaries": summaries,
              "limitations": "Largest predicted component is not adjudicated whole tumor. Other components are assessed separately, not declared false positives. T1-T3 not assessed. No training or production data change."}
    (OUT / "largest_component_crop_sizes.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summaries"]["all"], indent=2), flush=True)


if __name__ == "__main__":
    main()

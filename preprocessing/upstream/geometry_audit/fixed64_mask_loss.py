"""Measure model T0 mask truncation by a fixed registered 64x128x128 crop."""

import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import affine_transform

OUT = Path(__file__).resolve().parent
SEG = Path("/root/autodl-tmp/Datasets/ispy2_first_post_mamamia_segmentation_v1")
SHAPE_ZYX = np.array([64, 128, 128])
SPACING_ZYX = np.array([2.0, .7032, .7032])


def read(path):
    return json.loads(Path(path).read_text())


def main():
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    source = read(OUT / "registered_new_mask_patient_geometry.json")
    records = {(r["patient_id"], r["visit"]): r for r in read(SEG / "inventory.json")["records"]}
    results = []
    masks_read = 0
    for row in source["new_prediction"]:
        extent = np.asarray(row["extent_zyx_mm"])
        support_outside = bool(np.any(extent > SHAPE_ZYX * SPACING_ZYX + 1e-6))
        count = row["new_mask_native_voxels"]
        lost_centers = partial_cells = 0
        cropped_mask_voxels = None
        if support_outside:
            record = records[row["native_patient_id"], "T0"]
            image = sitk.ReadImage(str(SEG / "masks" / (record["case_id"] + ".nii.gz")))
            data = sitk.GetArrayViewFromImage(image)
            coordinates = np.asarray(np.where(data), dtype=float)[::-1].T
            assert len(coordinates) == count
            matrix = np.asarray(row["mask_to_strict_index_xyz"])
            points = coordinates @ matrix[:3, :3].T + matrix[:3, 3]
            half_cell = np.abs(matrix[:3, :3]).sum(axis=1) / 2
            lo, hi = points.min(axis=0) - half_cell, points.max(axis=0) + half_cell
            center = np.asarray(row["center_strict_zyx"])[::-1]
            assert np.allclose((lo + hi) / 2, center, atol=1e-6, rtol=0)
            assert np.allclose((hi - lo)[::-1] * SPACING_ZYX, extent, atol=1e-6, rtol=0)
            offset = np.abs(points - center)
            inside_centers = np.all(offset <= SHAPE_ZYX[::-1] / 2 + 1e-7, axis=1)
            inside_cells = np.all(offset + half_cell <= SHAPE_ZYX[::-1] / 2 + 1e-7, axis=1)
            assert not inside_cells.all()
            lost_centers = int((~inside_centers).sum())
            partial_cells = int((inside_centers & ~inside_cells).sum())
            inverse = np.linalg.inv(matrix)
            start_xyz = center - (SHAPE_ZYX[::-1] - 1) / 2
            reverse = np.eye(3)[::-1]
            output_to_input = reverse @ inverse[:3, :3] @ reverse
            offset_zyx = (inverse[:3, :3] @ start_xyz + inverse[:3, 3])[::-1]
            cropped = affine_transform(data, output_to_input, offset_zyx,
                                       output_shape=tuple(SHAPE_ZYX), order=0,
                                       mode="constant", cval=0, prefilter=False)
            cropped_mask_voxels = int(np.count_nonzero(cropped))
            masks_read += 1
            if masks_read % 25 == 0:
                print(f"Checked actual mask voxels for {masks_read} affected patients", flush=True)
        results.append({"patient_id": row["patient_id"], "fold": row["fold"], "source_mask_voxels": count,
                        "predicted_mask_support_truncated": support_outside,
                        "source_voxel_centers_outside_crop": lost_centers,
                        "outside_center_fraction": lost_centers / count,
                        "actual_cropped_mask_voxels_for_affected_case": cropped_mask_voxels,
                        "inside_centers_with_partially_clipped_cells": partial_cells,
                        "new_mask_components": row["components"]})
    summaries = {}
    for fold in ("train", "val", "all"):
        rows = [r for r in results if fold == "all" or r["fold"] == fold]
        affected = [r for r in rows if r["predicted_mask_support_truncated"]]
        fractions = np.asarray([r["outside_center_fraction"] for r in rows])
        affected_fractions = np.asarray([r["outside_center_fraction"] for r in affected])
        summaries[fold] = {"nonempty_t0_patients": len(rows), "mask_support_fully_contained": len(rows) - len(affected),
                           "patients_with_any_mask_support_truncated": len(affected),
                           "patients_with_any_mask_support_truncated_percent": len(affected) / len(rows) * 100,
                           "patients_with_any_voxel_center_outside": int((fractions > 0).sum()),
                           "patients_with_only_partial_cell_clipping": sum(r["source_voxel_centers_outside_crop"] == 0 for r in affected),
                           "outside_fraction_above_10_percent_patients": int((fractions > .1).sum()),
                           "outside_fraction_above_25_percent_patients": int((fractions > .25).sum()),
                           "outside_fraction_above_50_percent_patients": int((fractions > .5).sum()),
                           "all_source_mask_centers_outside_patients": int((fractions == 1).sum()),
                           "empty_actual_cropped_masks_among_affected_patients": sum(r["actual_cropped_mask_voxels_for_affected_case"] == 0 for r in affected),
                           "affected_patient_median_outside_center_fraction": float(np.median(affected_fractions)),
                           "all_patient_mean_outside_center_fraction": float(fractions.mean()),
                           "empty_t0_predictions_unassessable": sum(fold == "all" or r["fold"] == fold for r in source["excluded"])}
    assert len(results) == 866 and masks_read == summaries["all"]["patients_with_any_mask_support_truncated"]
    with (OUT / "fixed64_model_t0_mask_loss.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    report = {"scope": "All registered-cohort new model T0 predictions; no historical FTV constraint and no required extra margin",
              "shape_zyx": SHAPE_ZYX.tolist(), "spacing_zyx_mm": SPACING_ZYX.tolist(),
              "crop_extent_zyx_mm": (SHAPE_ZYX * SPACING_ZYX).tolist(),
              "center": "Each patient's mapped new T0 prediction bbox center",
              "geometry": "Previously verified saved-metadata and retained-phase-transform mapping; MRI NIfTI headers are not used",
              "method": "Crop boundaries are voxel faces. Mask-support truncation uses transformed source voxel-cell bounds. Source voxel centers outside the crop quantify lost predicted region, with partial cell clipping counted separately. Guaranteed-contained masks use their audited bounds and counts; all affected mask arrays are read and rechecked.",
              "affected_masks_read": masks_read, "affected_masks_resampled_to_actual_crop": masks_read,
              "actual_mask_crop_interpolation": "Nearest neighbor at the exact 64x128x128 grid centered on the mapped new T0 bbox",
              "summaries": summaries,
              "limitations": "Predicted mask geometry, not clinical tumor accuracy. Outside-center fractions are not exact partial-cell clipped volumes. T1-T3 model predictions are not included. Empty T0 prediction cannot be assessed by this rule.",
              "production_data_or_training_changed": False}
    (OUT / "fixed64_model_t0_mask_loss.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

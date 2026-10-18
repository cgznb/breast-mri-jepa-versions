"""Measure T0-centered ROI requirements without changing data or training."""

from __future__ import annotations

import csv
import itertools
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


ROOT = Path("/root/autodl-tmp")
OUT = Path(__file__).resolve().parent
SEG = ROOT / "Datasets/ispy2_first_post_mamamia_segmentation_v1"
CROPS = ROOT / "MAM/MeWM-ISPY2/runs/first_post_all_visits_90_10_tumor_v1_20260913/data"
BUNDLE = ROOT / "MAM/MOTFM-tumor-first/runs/registered_strict_a_z96_target_oracle/data_bundle.qingyuan_20260804.val102_high_transition"
WORLD = ROOT / "MAM/MeWM-ISPY2-DCE0-BiFlowNet-RF/runs/three_phase_symmflow_all_pairs_t0_v2_20260915/inventory.json"
SPACING = np.array([2.0, 0.7032, 0.7032])
OLD_SHAPE = np.array([96, 256, 256])
MARGINS = [0, 10, 15, 20, 25, 30]
SHAPES = [(z, xy, xy) for z in [32, 64, 96] for xy in [96, 128, 160, 192, 224, 256, 320, 384, 448, 480]]


def read(path):
    return json.loads(path.read_text())


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def csv_save(name, rows):
    if not rows:
        return
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentiles(values):
    return {str(q): np.percentile(values, q, axis=0).tolist() for q in [0, 25, 50, 75, 90, 95, 97.5, 99, 100]}


def corners(geometry, bbox):
    bounds = np.asarray(bbox)[::-1] - 0.5
    index = np.asarray(list(itertools.product(*bounds)))
    direction = np.asarray(geometry["direction_lps"]).reshape(3, 3)
    return (index * geometry["spacing_xyz_mm"]) @ direction.T + geometry["origin_lps_mm"]


def candidate_rows(name, rows):
    extent = np.asarray([r["extent_zyx_mm"] for r in rows])
    longitudinal = np.asarray([r["longitudinal_required_zyx_mm"] for r in rows])
    for margin, shape in itertools.product(MARGINS, SHAPES):
        shape = np.asarray(shape)
        span = (shape - 1) * SPACING
        required = extent + 2 * margin
        scales = np.maximum(1, np.max(required / span, axis=1))
        yield {
            "cohort": name,
            "patients": len(rows),
            "margin_per_side_mm": margin,
            "shape_zyx": "x".join(map(str, shape)),
            "volume_ratio": float(np.prod(shape) / np.prod(OLD_SHAPE)),
            "fov_center_span_z_mm": span[0],
            "fov_center_span_y_mm": span[1],
            "fov_center_span_x_mm": span[2],
            "t0_all_bbox_and_margin_fit": int(np.all(required <= span + 1e-6, axis=1).sum()),
            "t0_all_bbox_and_margin_fit_percent": float(np.mean(np.all(required <= span + 1e-6, axis=1)) * 100),
            "all_nonempty_visits_bbox_and_margin_fit": int(np.all(longitudinal + 2 * margin <= span + 1e-6, axis=1).sum()),
            "adaptive_spacing_scale_p50": float(np.percentile(scales, 50)),
            "adaptive_spacing_scale_p90": float(np.percentile(scales, 90)),
            "adaptive_spacing_scale_p95": float(np.percentile(scales, 95)),
            "adaptive_spacing_scale_max": float(scales.max()),
        }


def describe(rows):
    extent = np.asarray([r["extent_zyx_mm"] for r in rows])
    result = {
        "patients": len(rows),
        "t0_bbox_extent_zyx_mm_percentiles": percentiles(extent),
        "native_t0_spacing_zyx_mm_percentiles": percentiles([r["native_spacing_zyx_mm"] for r in rows]),
        "t0_connected_component_counts": dict(Counter(str(r["components"]) for r in rows)),
        "t0_multiple_components": sum(r["components"] > 1 for r in rows),
        "longitudinal_required_zyx_mm_percentiles": percentiles([r["longitudinal_required_zyx_mm"] for r in rows]),
        "minimum_uniform_margin_to_include_all_visit_bboxes_mm_percentiles": percentiles(np.maximum(0, np.max((np.asarray([r["longitudinal_required_zyx_mm"] for r in rows]) - extent) / 2, axis=1))),
        "buckets": {},
    }
    for margin in MARGINS[1:]:
        shapes = (np.ceil(((extent + 2 * margin) / SPACING + 1 - 1e-8) / 32) * 32).astype(int)
        assert np.all((shapes - 1) * SPACING >= extent + 2 * margin - 1e-6)
        volumes = shapes.prod(axis=1) / OLD_SHAPE.prod()
        counts = Counter("x".join(map(str, s)) for s in shapes)
        result["buckets"][str(margin)] = {
            "rounding": "per-axis to multiples of 32, no spacing change",
            "unique_shapes": len(counts),
            "most_common_shapes": counts.most_common(15),
            "voxel_ratio_mean": float(volumes.mean()),
            "voxel_ratio_percentiles": percentiles(volumes),
            "all_axes_smaller_than_original": int(np.all(shapes < OLD_SHAPE, axis=1).sum()),
            "voxel_count_smaller_than_original": int((volumes < 1).sum()),
            "max_shape_per_axis_zyx": shapes.max(axis=0).tolist(),
            "all_available_nonempty_visit_bboxes_fit": int(np.all(np.asarray([r["longitudinal_required_zyx_mm"] for r in rows]) <= (shapes - 1) * SPACING + 1e-6, axis=1).sum()),
        }
    return result


def native_rows():
    inventory = read(SEG / "inventory.json")
    manifest = read(CROPS / "manifest.json")
    crop_index = {(r["patient_id"], r["visit"]): r for r in manifest["records"]}
    all_records = {}
    patients = defaultdict(dict)
    mask_sizes_verified = 0
    for record in inventory["records"]:
        key = record["patient_id"], record["visit"]
        report = read(SEG / "case_reports" / (record["case_id"] + ".json"))
        if report["status"] != "completed":
            raise ValueError("Incomplete segmentation report")
        mask_path = SEG / "masks" / (record["case_id"] + ".nii.gz")
        assert mask_path.stat().st_size == report["mask_size_bytes"]
        mask_sizes_verified += 1
        qc = report["qc"]
        crop = crop_index[key]
        assert qc["tumor_voxels"] == crop["crop"]["source_mask_voxels"]
        if qc["tumor_voxels"]:
            assert np.allclose(qc["bbox_extent_xyz_mm"], crop["crop"]["bbox_extent_xyz_mm"], atol=1e-4)
        info = {"record": record, "qc": qc, "fold": crop["fold"], "mask_path": mask_path}
        patients[key[0]][key[1]] = info
        all_records[key] = info
    world = read(WORLD)
    formal = {(r["patient_id"], r["visit"]): r for r in world["visits"]}
    groups = defaultdict(list)
    empty_t0, missing_t0 = [], []
    output_rows = []
    for patient, visits in sorted(patients.items()):
        if "T0" not in visits:
            missing_t0.append(patient)
            continue
        baseline = visits["T0"]
        qc = baseline["qc"]
        if not qc["tumor_voxels"]:
            empty_t0.append(patient)
            continue
        geometry = qc["geometry"]
        direction = np.asarray(geometry["direction_lps"]).reshape(3, 3)
        t0_corners = corners(geometry, qc["bbox_zyx_exclusive"])
        center = t0_corners.mean(axis=0)
        required = []
        centers = []
        for visit in visits.values():
            vqc = visit["qc"]
            if not vqc["tumor_voxels"]:
                continue
            points = corners(vqc["geometry"], vqc["bbox_zyx_exclusive"])
            offsets = np.linalg.solve(direction, (points - center).T).T
            required.append(2 * np.max(np.abs(offsets), axis=0)[::-1])
            centers.append(float(np.linalg.norm(points.mean(axis=0) - center)))
        row = {
            "patient_id": patient,
            "fold": baseline["fold"],
            "extent_zyx_mm": list(reversed(qc["bbox_extent_xyz_mm"])),
            "native_spacing_zyx_mm": list(reversed(geometry["spacing_xyz_mm"])),
            "components": qc["connected_components_26"],
            "longitudinal_required_zyx_mm": np.max(required, axis=0).tolist(),
            "max_native_center_shift_mm": max(centers),
            "available_visits": len(visits),
            "nonempty_visits": len(required),
            "formal_member": (patient, "T0") in formal,
        }
        output_rows.append(row)
        groups["native_all_" + row["fold"]].append(row)
        if row["formal_member"]:
            formal_fold = formal[(patient, "T0")]["split"]
            groups["native_formal_" + formal_fold].append(row)
    details = {
        "segmentation_visits": len(all_records),
        "segmentation_patients": len(patients),
        "t0_count": len(output_rows) + len(empty_t0),
        "nonempty_t0_count": len(output_rows),
        "empty_t0_count": len(empty_t0),
        "missing_t0_count": len(missing_t0),
        "all_mask_files_size_verified": mask_sizes_verified,
        "longitudinal_semantics": "unregistered native LPS transfer projected into T0 axes; not registered anatomical correspondence; conservative bbox corner containment",
    }
    return groups, output_rows, all_records, details


def registered_rows():
    with (BUNDLE / "visits.csv").open(newline="") as handle:
        source = list(csv.DictReader(handle))
    patient_visits = defaultdict(list)
    for row in source:
        patient_visits[row["patient_id"]].append(row)
    plans = read(BUNDLE / "crop_plans.json")
    groups = defaultdict(list)
    output = []
    for patient, visits in sorted(patient_visits.items()):
        baseline = next(v for v in visits if v["visit"] == "T0")
        low = np.array(json.loads(baseline["resampled_bbox_min_zyx_json"]), dtype=float)
        high = np.array(json.loads(baseline["resampled_bbox_max_zyx_json"]), dtype=float)
        shape = np.array(json.loads(baseline["resampled_bbox_shape_zyx_json"]), dtype=float)
        assert np.array_equal(high - low + 1, shape)
        center = (low + high) / 2
        plan = plans[patient]
        assert np.array_equal(low, plan["bbox_min_zyx"])
        assert np.array_equal(high, plan["bbox_max_zyx"])
        required = []
        for visit in visits:
            lo = np.array(json.loads(visit["resampled_bbox_min_zyx_json"]), dtype=float) - .5
            hi = np.array(json.loads(visit["resampled_bbox_max_zyx_json"]), dtype=float) + .5
            required.append(2 * np.maximum(np.abs(lo - center), np.abs(hi - center)) * SPACING)
        row = {
            "patient_id": patient,
            "fold": baseline["fold"],
            "extent_zyx_mm": (shape * SPACING).tolist(),
            "native_spacing_zyx_mm": [float(baseline[k]) for k in ["slice_spacing_mm", "row_spacing_mm", "column_spacing_mm"]],
            "components": 0,
            "longitudinal_required_zyx_mm": np.max(required, axis=0).tolist(),
            "available_visits": len(visits),
        }
        output.append(row)
        groups["registered_legacy_" + row["fold"]].append(row)
    return groups, output


def mapping_inventory(all_records):
    native_ids = {key[0] for key in all_records}
    rows = []
    examples = {}
    with (BUNDLE / "visits.csv").open(newline="") as handle:
        source = [r for r in csv.DictReader(handle) if r["visit"] == "T0"]
    for record in source:
        matches = set()
        for key in ["dce0_path", "mask_path", "original_meta_path"]:
            matches.update(set(Path(record[key]).parts) & native_ids)
        if len(matches) > 1:
            raise ValueError("Ambiguous native identity")
        native_id = next(iter(matches), None)
        meta_path = Path(record["meta_path"])
        transform = Path(record["dce0_path"]).parent.parent / "transforms/intravisit/phase_1_to_phase_0.tfm"
        if len(examples) < 2 and native_id:
            collection = "ACRIN" if native_id.startswith("ACRIN") else "ISPY2"
            if collection not in examples:
                meta = read(meta_path)
                examples[collection] = {"meta_keys": list(meta), "meta_path": str(meta_path), "transform_path": str(transform), "source_geometry": meta.get("source_geometry"), "target_geometry": meta.get("target_geometry")}
        rows.append({"registered_patient_id": record["patient_id"], "native_patient_id": native_id, "fold": record["fold"], "t0_mask_found": (native_id, "T0") in all_records, "phase_transform_found": transform.is_file(), "meta_path": str(meta_path), "phase_transform_path": str(transform)})
    save("registered_new_mask_mapping_inventory.json", {"patients": rows, "examples": examples, "t0_masks_found": sum(r["t0_mask_found"] for r in rows), "phase_transforms_found": sum(r["phase_transform_found"] for r in rows)})
    return {"registered_patients": len(rows), "t0_masks_found": sum(r["t0_mask_found"] for r in rows), "phase_transforms_found": sum(r["phase_transform_found"] for r in rows)}


def verify_mask_samples(all_records, rows):
    ordered = sorted(rows, key=lambda r: max(r["extent_zyx_mm"][1:]))
    indices = sorted(set(np.linspace(0, len(ordered) - 1, 12, dtype=int).tolist()))
    results = []
    for index in indices:
        row = ordered[index]
        info = all_records[row["patient_id"], "T0"]
        image = sitk.ReadImage(str(info["mask_path"]))
        data = sitk.GetArrayViewFromImage(image)
        coords = np.where(data)
        bounds = [[int(a.min()), int(a.max()) + 1] for a in coords]
        qc = info["qc"]
        assert int(np.count_nonzero(data)) == qc["tumor_voxels"]
        assert bounds == qc["bbox_zyx_exclusive"]
        assert np.allclose(image.GetSpacing(), qc["geometry"]["spacing_xyz_mm"])
        assert np.allclose(image.GetOrigin(), qc["geometry"]["origin_lps_mm"])
        assert np.allclose(image.GetDirection(), qc["geometry"]["direction_lps"])
        results.append({"patient_id": row["patient_id"], "count_bbox_geometry_match": True})
    return results


def plot(groups, candidates):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for name, color, label in [
        ("native_all_train", "#147d75", "Existing T0 predictions"),
        ("registered_legacy_train", "#c45b30", "Historical registered masks"),
    ]:
        rows = groups[name]
        sizes = sorted(max(r["extent_zyx_mm"][1:]) for r in rows)
        axes[0].plot(sizes, np.arange(1, len(sizes) + 1) / len(sizes) * 100, color=color, label=label)
    axes[0].set(xlabel="T0 bounding-box max in-plane extent (mm)", ylabel="Training patients (%)", ylim=(0, 101))
    axes[0].legend(loc="lower right", fontsize=9)
    for z, color in [(32, "#986132"), (64, "#147d75"), (96, "#4667b2")]:
        r = [v for v in candidates if v["cohort"] == "native_all_train" and v["margin_per_side_mm"] == 20 and v["shape_zyx"].startswith(str(z) + "x") and int(v["shape_zyx"].split("x")[1]) <= 256]
        axes[1].plot([int(v["shape_zyx"].split("x")[1]) for v in r], [v["t0_all_bbox_and_margin_fit_percent"] for v in r], "o-", color=color, label=f"{z} slices")
    axes[1].set(xlabel="Square crop width / height (voxels)", ylabel="Full T0 mask + 20 mm margin fit (%)", ylim=(0, 101))
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
    fig.savefig(OUT / "coverage_tradeoff.png", dpi=180)
    plt.close(fig)


def main():
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    native, native_patients, all_records, details = native_rows()
    registered, registered_patients = registered_rows()
    groups = {**native, **registered}
    candidates = [r for name, rows in groups.items() for r in candidate_rows(name, rows)]
    csv_save("candidate_sizes.csv", candidates)
    save("patient_geometry.json", {"native": native_patients, "registered_legacy": registered_patients})
    verification = verify_mask_samples(all_records, native_patients)
    mapping = mapping_inventory(all_records)
    summary = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_policy": "training geometry only; validation rows reported separately",
        "scope": "T0 bbox geometry audit, no model inference or training; full predicted mask union retained",
        "spacing_zyx_mm": SPACING.tolist(),
        "baseline_shape_zyx": OLD_SHAPE.tolist(),
        "native": details,
        "groups": {name: describe(rows) for name, rows in groups.items()},
        "mask_sample_verification": verification,
        "registered_new_mask_mapping_inventory": mapping,
        "limitations": [
            "New native first-post predictions and historical registered masks have different mask provenance.",
            "A T0 crop can retain all available mask voxels without establishing clinical segmentation accuracy.",
            "Projected native follow-up geometry is not longitudinal registration.",
            "Native formal inventory precedes ongoing acquisition-coverage admission.",
            "Volume ratios describe voxel counts, not measured GPU memory or wall time.",
        ],
    }
    save("summary.json", summary)
    plot(groups, candidates)
    print(json.dumps({"native": details, "cohorts": {k: len(v) for k, v in groups.items()}, "candidate_rows": len(candidates), "sample_checks": len(verification)}), flush=True)


if __name__ == "__main__":
    main()

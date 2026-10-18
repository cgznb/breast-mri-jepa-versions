"""Render completed ROI32 forecasts from saved samples and prepared real crops."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from mewm_ispy2 import registered_roi32_runtime as runtime
from mewm_ispy2 import registered_roi32_vq as vq
from mewm_ispy2.registered_roi32_data import (
    CropDataset, SPACING, file_identity, read_config, read_json, timestamp,
    verify_identity, write_json,
)
from mewm_ispy2.registered_roi32_evaluation import gallery

METRICS = ("l1", "valid_l1", "t0_roi_l1", "ssim_clamped_neg1_pos3", "lpips_three_slices_sum")
TIME_PAIRS = ("T0->T1", "T0->T2", "T0->T3", "T1->T2", "T1->T3", "T2->T3")
NAMES = ("Source / copy baseline", "Real target", "VQ target reconstruction", "FM sample 1", "FM mean of 8")


def select_cases(rows):
    by_pair = defaultdict(list)
    for row in rows:
        if row["method"] == "sample":
            by_pair[row["pair_id"]].append(row)
    candidates = []
    for pair_id, samples in by_pair.items():
        if sorted(int(row["sample_index"]) for row in samples) != list(range(8)):
            raise ValueError("Expected eight distinct generated samples per pair")
        candidates.append({"pair_id": pair_id, "patient_id": samples[0]["patient_id"],
                           "time_pair": samples[0]["time_pair"],
                           "largest_clipped": samples[0]["t0_largest_clipped"] == "True",
                           "mean_sample_l1": float(np.mean([float(row["l1"]) for row in samples]))})
    chosen, seen = [], set()
    for time_pair in TIME_PAIRS:
        clipped = time_pair in ("T0->T2", "T0->T3")
        pool = [row for row in candidates if row["time_pair"] == time_pair and row["largest_clipped"] == clipped]
        median = float(np.median([row["mean_sample_l1"] for row in pool]))
        available = [row for row in pool if row["patient_id"] not in seen]
        if not available:
            raise ValueError(f"No distinct patient available for {time_pair}")
        selected = min(available, key=lambda row: (abs(row["mean_sample_l1"] - median), row["pair_id"]))
        chosen.append({**selected, "stratum_median_l1": median, "stratum_pairs": len(pool)})
        seen.add(selected["patient_id"])
    return chosen


def draw_panel(panel, volume, mask, axis, anchor, window):
    plane = np.take(volume, anchor, axis=axis)
    contour = np.take(mask, anchor, axis=axis)
    spacing = [SPACING[i] for i in range(3) if i != axis]
    panel.imshow(plane, cmap="gray", origin="lower", interpolation="nearest",
                 vmin=window[0], vmax=window[1], aspect=spacing[0] / spacing[1])
    if contour.any() and not contour.all():
        panel.contour(contour, levels=[0.5], colors=["#38c896"], linewidths=0.6, origin="lower")
    panel.set_axis_off()


def render_case(path, case):
    figure, axes = plt.subplots(3, 5, figsize=(18, 9.2), constrained_layout=True)
    status = "Largest T0 region clipped" if case["largest_clipped"] else "Largest T0 region contained"
    window = case["window"]
    figure.suptitle(f"{case['pair_id']} | {status}\nGreen: fixed T0 model mask | Shared window [{window[0]:.2f}, {window[1]:.2f}]", fontsize=12)
    for axis, plane in enumerate(("Axial", "Coronal", "Sagittal")):
        for column, volume in enumerate(case["volumes"]):
            panel = axes[axis, column]
            draw_panel(panel, volume, case["mask"], axis, case["anchors"][axis], window)
            panel.set_title(f"{NAMES[column]}\n{plane} {case['anchors'][axis]}", fontsize=9)
    figure.savefig(path, dpi=140, facecolor="white")
    plt.close(figure)


def render_overview(path, cases):
    figure, axes = plt.subplots(len(cases), 5, figsize=(14, 3 * len(cases)), constrained_layout=True)
    figure.suptitle("Registered DCE0 | Six validation patients\nGreen: fixed T0 first-post model mask", fontsize=13)
    for row, case in enumerate(cases):
        for column, volume in enumerate(case["volumes"]):
            panel = axes[row, column]
            draw_panel(panel, volume, case["mask"], 0, case["anchors"][0], case["window"])
            if row == 0:
                panel.set_title(NAMES[column], fontsize=9)
        status = "clipped" if case["largest_clipped"] else "contained"
        axes[row, 0].text(0, -0.06, f"{case['pair_id']} | {status}", transform=axes[row, 0].transAxes, fontsize=8)
    figure.savefig(path, dpi=140, facecolor="white")
    plt.close(figure)


def render_curves(path, events):
    loss = [event for event in events if event["event"] == "validation"]
    endpoints = [event for event in events if event["event"] == "endpoint_validation"]
    best = min(endpoints, key=lambda event: event["endpoint_candidate_mae"])
    def x_values(items):
        return [item.get("reference_step", item["step"]) * 4 / 1000 for item in items]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), constrained_layout=True)
    axes[0].plot(x_values(loss), [item["loss"] for item in loss], color="#187b83", linewidth=1.8)
    axes[0].set_title("EMA validation FM loss")
    for key, label, color in (("endpoint_candidate_mae", "Single sample (selection metric)", "#b44f56"),
                              ("endpoint_mean_mae", "Mean of 2 samples", "#187b83"),
                              ("endpoint_source_copy_mae", "Source copy", "#737373")):
        axes[1].plot(x_values(endpoints), [item[key] for item in endpoints], label=label, color=color)
    axes[1].axvline(x_values([best])[0], color="#555555", linestyle=":", label="Selected checkpoint")
    axes[1].set_title("Endpoint latent MAE | Heun 25 steps")
    axes[1].legend(fontsize=8)
    for panel in axes:
        panel.set_xlabel("Sampled training pairs (thousands)")
        panel.set_ylabel("Lower is better")
        panel.grid(alpha=0.2)
    figure.savefig(path, dpi=150, facecolor="white")
    plt.close(figure)
    return best


@torch.inference_mode()
def preview(config, output, device):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    root = Path(config["output_dir"])
    evaluation = root / "fm_evaluation"
    complete = read_json(evaluation / "COMPLETE.json")
    if complete["status"] != "passed":
        raise ValueError("Final FM evaluation has not passed")
    runtime.verify_contract_files(complete["contract"])
    for identity in complete["artifacts"]:
        verify_identity(identity)
    with (evaluation / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = read_json(evaluation / "summary.json")
    if len(rows) != summary["pairs"] * 11 or not all(np.isfinite(float(row[key])) for row in rows for key in METRICS):
        raise ValueError("Incomplete or non-finite final metrics")
    aggregates = [group for group in summary["groups"] if len(group["group"]) == 1]
    for group in aggregates:
        subset = [row for row in rows if row["method"] == group["group"]["method"]]
        for key in METRICS:
            if abs(float(np.mean([float(row[key]) for row in subset])) - group[key]) > 1e-12:
                raise ValueError("Final summary does not match per-pair metrics")
    selected = select_cases(rows)
    dataset = CropDataset(config, "val")
    indices = {record["visit_id"]: index for index, record in enumerate(dataset.records)}
    codec, codec_identity = vq.load_frozen(config, device)
    before = {name: value.detach().clone() for name, value in codec.quantizer.state_dict().items()}
    output.mkdir(parents=True, exist_ok=False)
    cases, entries, provenance = [], [], []
    for selection in selected:
        name = selection["pair_id"].replace(":", "_").replace("->", "_")
        saved = read_json(evaluation / "cases" / (name + ".json"))
        if saved["contract"] != complete["contract"]:
            raise ValueError("Saved case belongs to another evaluation")
        verify_identity(saved["samples_identity"])
        report = read_json(root / "data/patients" / (selection["patient_id"] + ".json"))
        with np.load(saved["samples_identity"]["path"], allow_pickle=False) as archive:
            source_id, target_id = str(archive["earlier_visit_id"]), str(archive["later_visit_id"])
            sample = archive["samples"][0, 0].astype(np.float32)
            mean = archive["predictive_mean"][0].astype(np.float32)
            if archive["samples"].shape != (8, 1, 32, 128, 128):
                raise ValueError("Unexpected saved forecast dimensions")
            if not np.allclose(archive["affine_lps"], report["crop_affine_lps"], rtol=0, atol=1e-9):
                raise ValueError("Saved forecast geometry differs from the real crop")
        source, target = dataset[indices[source_id]], dataset[indices[target_id]]
        tensor = target["image"][None].to(device)
        with runtime.autocast(device):
            reconstruction, _ = codec(tensor)
        original = target["image"][0].numpy()
        mask = target["mask"][0].numpy()
        window = np.percentile(original[target["valid"][0].numpy()], [0.5, 99.5]).tolist()
        volumes = [source["image"][0].numpy(), original, reconstruction[0, 0].float().cpu().numpy(), sample, mean]
        if not mask.any() or window[1] - window[0] < 1e-6 or not all(np.isfinite(volume).all() for volume in volumes):
            raise ValueError("Invalid case image, mask or display range")
        anchors = [int(mask.sum(axis=tuple(i for i in range(3) if i != axis)).argmax()) for axis in range(3)]
        case = {**selection, "volumes": volumes, "mask": mask, "window": window, "anchors": anchors}
        render_case(output / (name + ".png"), case)
        cases.append(case)
        entries.append((name + ".png", f"{selection['pair_id']} | Mean single-sample L1 {selection['mean_sample_l1']:.4f}"))
        provenance.append({**selection, "source_visit_id": source_id, "target_visit_id": target_id,
                           "display_window": window, "slice_indices_zyx": anchors, "samples": saved["samples_identity"],
                           "geometry": file_identity(root / "data/patients" / (selection["patient_id"] + ".json"))})
        print(json.dumps({"completed": len(cases), "total": len(selected)}), flush=True)
    if any(not torch.equal(value, codec.quantizer.state_dict()[name]) for name, value in before.items()):
        raise RuntimeError("Preview changed the frozen VQ codebook")
    render_overview(output / "overview.png", cases)
    events = [json.loads(line) for line in (root / "fm/metrics.jsonl").read_text().splitlines()]
    best = render_curves(output / "training_curves.png", events)
    gallery(output / "index.html", "Registered ROI32 | Completed Symm FM | Six validation patients", entries)
    for path in output.glob("*.png"):
        with Image.open(path) as rendered:
            if min(rendered.size) < 500 or np.asarray(rendered.convert("RGB")).std() < 10:
                raise ValueError(f"Blank or undersized preview: {path.name}")
    write_json(output / "report.json", {
        "status": "passed", "created_at": timestamp(), "experiment": str(root), "split": "val", "test_split": None,
        "final_evaluation": file_identity(evaluation / "summary.json"), "codec": codec_identity,
        "selected_flow": complete["contract"]["flow"], "selected_endpoint_event": best,
        "selection": "One per time pair, nearest median single-sample L1 within clipping stratum, six unique patients; T0->T2 and T0->T3 use clipped cases",
        "display": "Target valid-voxel percentiles 0.5/99.5 shared across all five columns; physical spacing preserved",
        "contour": "Fixed T0 first-post model mask, not follow-up or generated segmentation",
        "forecast_source": "Original final evaluation samples, sample_index=0 and stored mean of all eight; no FM resampling",
        "vq_display_precision": "bf16" if device.type == "cuda" else "fp32", "device": str(device),
        "aggregates": aggregates, "examples": provenance, "metric_replay": "passed", "codebook_unchanged": True,
    })
    print(json.dumps({"status": "passed", "gallery": str(output / "index.html")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO / "configs/registered_dce0_roi32_firstpostmask_v1.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    preview(read_config(args.config), Path(args.output).resolve(), torch.device(args.device))


if __name__ == "__main__":
    main()

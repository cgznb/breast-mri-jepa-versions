"""Package the completed step sweep as an offline scientific image browser."""

from __future__ import annotations

import base64
import argparse
import io
import itertools
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from mewm_ispy2.registered_roi32_data import file_identity, read_json, verify_identity, write_json
from mewm_ispy2.registered_three_phase_data import PHASES

ROOT = REPO / "runs/registered_three_phase_roi32_v1/fm_review_30cases_20260917"
TEMPLATE = REPO / "scripts/three_phase_review_web"
LUCIDE = Path("/root/autodl-tmp/.codex/euler20_review_browser/offline_test_20260914/Local Results/ispy2_mu_results_offline_20260914/lucide.min.js")
LABELS = ("Precontrast", "First-post", "Late")


def encode_volume(volume):
    value = np.asarray(volume, dtype=np.float32)
    assert value.shape == (3, 32, 128, 128) and np.isfinite(value).all()
    lower, upper = float(value.min()), float(value.max())
    scale = max((upper - lower) / 65535, 1e-10)
    packed = np.rint((value.astype(np.float64) - lower) / scale).clip(0, 65535).astype(np.uint16)
    reconstructed = packed.astype(np.float64) * scale + lower
    error = float(np.max(np.abs(reconstructed - value)))
    assert error <= scale / 2 + 1e-7
    raster = packed.reshape(4096, 384)
    rgb = np.zeros((4096, 384, 3), dtype=np.uint8)
    rgb[:, :, 0], rgb[:, :, 1] = raster >> 8, raster & 255
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG", compress_level=3)
    return {"png": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
            "lower": lower, "scale": scale, "shape": list(value.shape), "max_error": error}


def encode_mask(mask):
    buffer = io.BytesIO()
    Image.fromarray((mask.reshape(4096, 128) * 255).astype(np.uint8)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def write_pack(path, key, volumes, **extra):
    payload = {"volumes": {name: encode_volume(value) for name, value in volumes.items()}, **extra}
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    path.write_text(f"window.registerReviewData({json.dumps(key)},{encoded});\n")
    return [value["max_error"] for value in payload["volumes"].values()]


def aggregate(cases, name, keys):
    groups = defaultdict(list)
    for case in cases:
        for row in case[name]:
            groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for key, values in sorted(groups.items()):
        row = dict(zip(keys, key, strict=True))
        for field, value in values[0].items():
            if field not in keys and (isinstance(value, (int, float)) or value is None):
                finite = [v[field] for v in values if v[field] is not None]
                row[field] = float(np.mean(finite)) if finite else None
        row["observations"] = len(values)
        result.append(row)
    return result


def charts(destination, summary):
    table = {(r["steps"], r["method"], r["phase"]): r for r in summary["image_metrics"]}
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, phase, label in zip(axes, PHASES, LABELS, strict=True):
        for method, title, color in (("sample", "Individual samples", "#ae475a"), ("mean", "Mean of 4", "#168678")):
            axis.plot([10, 20, 30, 50], [table[step, method, phase]["t0_target_mae"] for step in (10, 20, 30, 50)],
                      marker="o", color=color, label=title)
        axis.axhline(table[0, "source_copy", phase]["t0_target_mae"], color="#777777", label="Source copy")
        axis.set_title(label, fontsize=12)
        axis.set_xlabel("Heun integration steps")
        axis.set_ylabel("Fixed T0-region MAE")
        axis.set_xticks([10, 20, 30, 50])
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3)
    figure.suptitle("30 distinct validation patients | matched noise at all step counts", fontsize=13)
    figure.savefig(destination / "step_metrics.png", dpi=150)
    plt.close(figure)
    noise = {(r["steps"], r["phase"]): r for r in summary["diversity"]}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for phase, label, color in zip(PHASES, LABELS, ("#168678", "#ae475a", "#397eb0"), strict=True):
        axes[0].plot([10, 20, 30, 50], [noise[step, phase]["t0_pairwise_mae"] for step in (10, 20, 30, 50)],
                     marker="o", color=color, label=label)
        axes[1].plot([10, 20, 30, 50], [table[step, "mean", phase]["t0_change_ratio"] for step in (10, 20, 30, 50)],
                     marker="o", color=color, label=label)
    axes[0].set_ylabel("Between-noise T0-region MAE")
    axes[0].set_title("Four independent draws per patient", fontsize=12)
    axes[1].set_ylabel("Predicted / observed T0-region change")
    axes[1].set_title("Mean-of-4 change relative to source", fontsize=12)
    axes[1].axhline(1, color="#777777", linestyle="--")
    for axis in axes:
        axis.set_xlabel("Heun integration steps")
        axis.set_xticks([10, 20, 30, 50])
        axis.grid(alpha=0.2)
        axis.legend(fontsize=9)
    figure.savefig(destination / "diversity_metrics.png", dpi=150)
    plt.close(figure)


def report(destination, summary):
    table = {(r["steps"], r["method"], r["phase"]): r for r in summary["image_metrics"]}
    noise = {(r["steps"], r["phase"]): r for r in summary["diversity"]}
    lines = ["# Thirty-patient registered three-phase forecast review", "",
             "Thirty distinct validation patients, five per forward transition. Selection used a fixed "
             "seed and no image-quality ranking. Every patient has four independent noise draws at "
             "10, 20, 30 and 50 Heun steps, with identical noise and batch size across steps. "
             "Heun uses two network evaluations per step. The checkpoint, codec, crop and "
             "normalization are unchanged; no model was retrained.", "",
             "Both best-endpoint and final checkpoints are included in the browser. They are "
             "both step6300: all577 saved model tensors and556 EMA tensors are exactly equal, "
             "including the condition encoder. Both use EMA velocity weights under the same "
             "inference policy. Their matched-noise predictions therefore share the same "
             "verified results; the two labels do not represent different fitted models. "
             "[Weight comparison](weight_comparison.json)", "",
             "[Offline browser](index.html)", "", "## Step comparison", "",
             "MAE uses the original unclipped volumes and the fixed T0 predicted mask. Each patient "
             "has one selected pair and equal weight; individual-sample metrics average four draws. "
             "Mean4 here differs from the previous full-cohort Mean8 evaluation.", "",
             "| Phase | Source | Sample10 | Sample20 | Sample30 | Sample50 | Mean4@50 | Noise diversity@50 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for phase, label in zip(PHASES, LABELS, strict=True):
        values = [table[0, "source_copy", phase]["t0_target_mae"]]
        values += [table[step, "sample", phase]["t0_target_mae"] for step in (10, 20, 30, 50)]
        values += [table[50, "mean", phase]["t0_target_mae"], noise[50, phase]["t0_pairwise_mae"]]
        lines.append("| " + label + " | " + " | ".join(f"{value:.6f}" for value in values) + " |")
    lines += ["", "## Collapse and copying diagnostics", "",
              "Noise diversity is the average absolute difference over all six pairs of four "
              "independent samples, within each patient. Nonzero differences refute exact "
              "noise-independent output for these inputs; they do not prove that generated "
              "variation captures real biological uncertainty or exclude partial conditional collapse.", "",
              "| Phase | Mean50 to source, T0 MAE | Observed change, T0 MAE | Mean50 change ratio | "
              "Change cosine | Mean50 / target edge magnitude |",
              "|---|---:|---:|---:|---:|---:|"]
    for phase, label in zip(PHASES, LABELS, strict=True):
        mean, target = table[50, "mean", phase], table[0, "target", phase]
        lines.append(f"| {label} | {mean['t0_source_mae']:.6f} | {mean['t0_real_change_mae']:.6f} | "
                     f"{mean['t0_change_ratio']:.6f} | {mean['t0_change_cosine']:.6f} | "
                     f"{mean['xy_edge_magnitude'] / target['xy_edge_magnitude']:.6f} |")
    lines += ["", "The model starts with joint [noise, source] and predicts the later branch after "
              "ODE integration; no direct source-copy shortcut exists in the sampling wrapper. "
              "Stable anatomy and source conditioning explain some similarity. The training loss "
              "is an equal-weight two-branch latent velocity MSE, without an explicit tumor-change "
              "or image-texture loss. A model that underfits change can therefore preserve anatomy "
              "while missing response details. Averaging stochastic images further smooths detail. "
              "These mechanisms are plausible explanations, not causal ablation results.", "",
              "A condition control changes the future stage and recorded interval to another "
              "available T0-forward pair from the same patient, with source, noise, solver and "
              "batch fixed. It measures sensitivity to those conditions, not accuracy of a new "
              "clinical outcome. Control results are in summary.json.", "",
              "## Axial neighborhood", "",
              "The browser keeps a real reference slice fixed and displays offsets -10..-1 and "
              "+1..+10. The same-index prediction is separate, giving twenty neighbors plus "
              "one central prediction. The reference can switch between true follow-up and source. "
              "The neighborhood can also switch to true follow-up to inspect registration itself. "
              "The default anchor is the maximal T0-mask area, limited to z=10..21 so both "
              "ten-slice neighborhoods fit inside the 32-slice crop. Moving outside this range "
              "shows explicit unavailable positions; slices are never repeated to fill gaps.", "",
              "Nearby-oracle MAE takes the minimum over 21 candidate prediction slices, including "
              "the center, against a fixed true-target slice. It is a target-informed exploratory "
              "measure and will decrease through extra search flexibility; it is not a replacement "
              "for the original validation score or evidence of correct anatomical correspondence. "
              "The 2-mm axial spacing makes ten slices 20 mm. This only explores axial offset; "
              "it does not correct in-plane or deformable misregistration.", "",
              "The browser's grayscale window is shared across all compared volumes. Packed "
              "16-bit PNG values preserve the full exported range; display quantization errors "
              "are recorded separately from the original-volume metrics. The green contour "
              "is a fixed T0 predicted mask, not follow-up lesion ground truth. This is a "
              "balanced validation subset, not an independent test cohort.", ""]
    (destination / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-only", action="store_true")
    args = parser.parse_args()
    if args.assets_only:
        destination = ROOT / "web"
        for filename in ("index.html", "viewer.js", "viewer.css"):
            shutil.copyfile(TEMPLATE / filename, destination / filename)
        verification = read_json(destination / "verification.json")
        verification.update(renderer=file_identity(__file__), source_templates=[file_identity(TEMPLATE / f) for f in ("index.html", "viewer.js", "viewer.css")])
        write_json(destination / "verification.json", verification)
        print("Web assets refreshed; forecast packages unchanged")
        return
    complete = read_json(ROOT / "COMPLETE.json")
    assert complete["status"] == "passed" and complete["patients"] == complete["cases"] == 30
    verify_identity(complete["results"])
    verify_identity(complete["protocol"])
    results = read_json(ROOT / "results.json")
    best_path, final_path = ROOT.parent / "fm/best-endpoint.pt", ROOT.parent / "fm/last.pt"
    best = torch.load(best_path, map_location="cpu", mmap=True, weights_only=False)
    final = torch.load(final_path, map_location="cpu", mmap=True, weights_only=False)
    weight_checks = {}
    for name, first, second in (("model", best["model"], final["model"]),
                                ("ema", best["ema"]["shadow"], final["ema"]["shadow"])):
        assert set(first) == set(second)
        different = [key for key in first if not torch.equal(first[key], second[key])]
        if different:
            raise ValueError("Final weights differ; separate final-checkpoint inference is required")
        weight_checks[name] = {"tensors_compared": len(first), "different_tensors": 0}
    assert best["model_description"] == final["model_description"]
    weight_checks.update(best_step=best["training_state"]["step"], final_step=final["training_state"]["step"],
                         identical=True, best=file_identity(best_path), final=file_identity(final_path),
                         inference="EMA velocity parameters with saved condition encoder for both checkpoints")
    del best, final
    destination = ROOT / "web"
    destination.mkdir(exist_ok=True)
    write_json(destination / "weight_comparison.json", weight_checks)
    for filename in ("index.html", "viewer.js", "viewer.css"):
        shutil.copyfile(TEMPLATE / filename, destination / filename)
    shutil.copyfile(LUCIDE, destination / "lucide.min.js")
    summary = {"patients": 30, "samples_per_step": 4, "steps": [10, 20, 30, 50],
               "weight_comparison": weight_checks,
               "image_metrics": aggregate(results["cases"], "image_metrics", ("steps", "method", "phase")),
               "diversity": aggregate(results["cases"], "diversity", ("steps", "phase")),
               "step_differences": aggregate(results["cases"], "step_differences", ("steps", "phase"))}
    controls = [case for case in results["cases"] if case["condition_control"] is not None]
    summary["condition_control_cases"] = len(controls)
    summary["condition_control"] = aggregate(controls, "condition_control", ("phase",))
    case_list, errors = [], []
    for index, case in enumerate(results["cases"]):
        record = case["record"]
        name = record["pair_id"].replace(":", "_").replace("->", "_to_")
        directory, data = ROOT / "cases" / name, destination / "data" / name
        data.mkdir(parents=True, exist_ok=True)
        for identity in case["inference"]["artifacts"] + [case["reference"]]:
            verify_identity(identity)
        with np.load(directory / "reference.npz", allow_pickle=False) as arrays:
            reference = {key: arrays[key].copy() for key in ("source", "target", "source_vq", "target_vq")}
            mask = arrays["mask"].copy()
        enhancement = np.stack([v[[1, 2]] - v[0] for v in (reference["source"], reference["target"])])
        limit = max(float(np.percentile(np.abs(enhancement[:, :, mask]), 99.5)), 0.1)
        errors.extend(write_pack(data / "reference.js", name + "/reference", reference, mask=encode_mask(mask)))
        for step in (10, 20, 30, 50):
            with np.load(directory / f"steps_{step}.npz", allow_pickle=False) as arrays:
                volumes = {f"sample{i + 1}": arrays["samples"][i].astype(np.float32) for i in range(4)}
                volumes["mean"] = arrays["mean"].copy()
            errors.extend(write_pack(data / f"step{step}.js", name + f"/step{step}", volumes))
        case_list.append({"id": name, "patient": record["patient_id"], "transition": record["earlier_stage"] + " → " + record["later_stage"],
                          "earlier": record["earlier_stage"], "later": record["later_stage"],
                          "days": record["delta_days"], "interval_missing": record["interval_missing"],
                          "anchor": case["anchor_z"], "maxMaskZ": case["max_mask_z"],
                          "window": case["window"], "enhancementWindow": [-limit, limit],
                          "registration": case["inference"]["registration_status"],
                          "clipped": case["inference"]["largest_clipped"],
                          "metrics": case["image_metrics"], "diversity": case["diversity"],
                          "seed": case["inference"]["seed"]})
        print({"packaged_cases": index + 1, "total": 30}, flush=True)
    write_json(destination / "summary.json", summary)
    charts(destination, summary)
    report(destination, summary)
    manifest = {"cases": case_list, "summary": summary, "createdAt": complete["updated_at"],
                "checkpoints": [{"key": "best", "label": "最优权重", "step": weight_checks["best_step"]},
                                {"key": "final", "label": "最终权重", "step": weight_checks["final_step"]}],
                "weightsIdentical": weight_checks["identical"],
                "shape": [3, 32, 128, 128], "spacing": [2, 0.7032, 0.7032]}
    (destination / "manifest.js").write_text("window.REVIEW_MANIFEST=" + json.dumps(manifest, ensure_ascii=True, separators=(",", ":")) + ";\n")
    write_json(destination / "verification.json", {"status": "passed", "patients": len(case_list),
               "encoded_volumes": len(errors), "maximum_display_quantization_error": max(errors),
               "inference_completion": file_identity(ROOT / "COMPLETE.json"),
               "renderer": file_identity(__file__), "source_templates": [file_identity(TEMPLATE / f) for f in ("index.html", "viewer.js", "viewer.css")]})
    print({"status": "passed", "web": str(destination / "index.html"), "quantization_error": max(errors)}, flush=True)


if __name__ == "__main__":
    main()

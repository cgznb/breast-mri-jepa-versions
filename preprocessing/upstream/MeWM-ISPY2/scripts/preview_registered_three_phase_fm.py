"""Render completed forecasts and replay their saved aggregate metrics on CPU."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mewm_ispy2 import registered_roi32_runtime as runtime
from mewm_ispy2.registered_roi32_data import file_identity, read_json, verify_identity, write_json
from mewm_ispy2.registered_roi32_evaluation import gallery
from mewm_ispy2.registered_three_phase_data import PHASES, ThreePhaseCrops, load_config

LABELS = ("Precontrast", "First-post", "Late")
IMAGE_METRICS = ("l1", "valid_l1", "t0_roi_l1", "ssim_clamped_neg1_pos3", "lpips_three_slices_sum")


def read_table(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def replay(rows, summary, keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    checks = 0
    for aggregate in summary:
        selected = groups[tuple(aggregate[key] for key in keys)]
        assert len(selected) == aggregate["visits"]
        assert len({row["patient_id"] for row in selected}) == aggregate["patients"]
        for metric in metrics:
            patients = defaultdict(list)
            for row in selected:
                if row[metric] != "":
                    value = float(row[metric])
                    assert math.isfinite(value)
                    patients[row["patient_id"]].append(value)
            visit_mean = statistics.mean(value for values in patients.values() for value in values) if patients else None
            macro = statistics.mean(statistics.mean(values) for values in patients.values()) if patients else None
            for suffix, value in (("", macro), ("_visit_mean", visit_mean)):
                expected = aggregate[metric + suffix]
                assert value is expected if value is None or expected is None else abs(value - expected) < 1e-10
                checks += 1
    return checks


def curves(path, events):
    validation = [row for row in events if row["event"] == "validation"]
    endpoint = [row for row in events if row["event"] == "endpoint_validation"]
    assert len(validation) == 100 and len(endpoint) == 20
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot([row["reference_step"] * 4 / 1000 for row in validation],
                 [row["loss"] for row in validation], color="#227c70", label="Validation flow loss")
    for key, label, color in (("endpoint_candidate_mae", "Single sample", "#ac4458"),
                              ("endpoint_mean_mae", "Mean of 2 samples", "#227c70"),
                              ("endpoint_source_copy_mae", "Source copy", "#6c6c6c")):
        axes[1].plot([row["reference_step"] * 4 / 1000 for row in endpoint],
                     [row[key] for row in endpoint], marker=".", label=label, color=color)
    for panel in axes:
        panel.set_xlabel("Training pair draws (thousands)")
        panel.grid(alpha=0.18)
        panel.legend(fontsize=9)
    axes[0].set_ylabel("Symmetric flow loss")
    axes[1].set_ylabel("Normalized latent MAE")
    figure.suptitle("Registered three-phase ROI32 | all 538 validation pairs", fontsize=12)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def comparison(path, title, volumes, mask, *, enhancement=False):
    z = int(np.argmax(mask.sum(axis=(1, 2))))
    if enhancement:
        shown = {key: np.stack((value[1] - value[0], value[2] - value[0])) for key, value in volumes.items()}
        labels = ("First-post - pre", "Late - pre")
    else:
        shown, labels = volumes, LABELS
    real = shown["True target"]
    source = shown["Source"]
    if enhancement:
        limit = max(float(np.percentile(np.abs(np.concatenate((real[:, mask], source[:, mask]))), 99.5)),
                    float(np.percentile(np.abs(np.stack((real, source))), 99.5)), 0.1)
        lower, upper, cmap = -limit, limit, "coolwarm"
    else:
        foreground = np.concatenate((real[real != 0], source[source != 0]))
        lower = float(np.percentile(foreground, 1))
        upper = max(float(np.percentile(foreground, 99.5)),
                    float(np.percentile(np.concatenate((real[:, mask], source[:, mask])), 99.5)), lower + 0.1)
        cmap = "gray"
    figure, axes = plt.subplots(len(labels), 5, figsize=(17, 3.4 * len(labels)), constrained_layout=True)
    figure.suptitle(title + f" | axial z={z} | fixed T0 mask contour", fontsize=12)
    for row, label in enumerate(labels):
        for column, (name, data) in enumerate(shown.items()):
            panel = axes[row, column]
            picture = panel.imshow(data[row, z], cmap=cmap, vmin=lower, vmax=upper, origin="lower")
            panel.set_title(name, fontsize=10)
            if mask[z].any() and not mask[z].all():
                panel.contour(mask[z], levels=[0.5], colors=["#3be0ad"], linewidths=0.65, origin="lower")
            panel.set_axis_off()
        axes[row, 0].text(-0.04, 0.5, label, transform=axes[row, 0].transAxes, rotation=90,
                          ha="right", va="center", fontsize=10)
        error = np.abs(shown["Mean of 8"] - real)
        panel = axes[row, 4]
        error_picture = panel.imshow(error[row, z], cmap="magma", vmin=0, vmax=upper - lower, origin="lower")
        panel.set_title("Absolute error of mean", fontsize=10)
        panel.set_axis_off()
        figure.colorbar(picture, ax=list(axes[row, :4]), shrink=0.7, label="Shared intensity units")
        figure.colorbar(error_picture, ax=panel, shrink=0.7)
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return {"slice_z": z, "lower": lower, "upper": upper}


def metric_plot(path, summary):
    table = {(row["method"], row["phase"]): row for row in summary["phase_metrics"]}
    methods = (("source_copy", "Source copy", "#858585"), ("sample", "Single sample", "#ac4458"),
               ("sample_mean", "Mean of 8", "#227c70"), ("vq_target", "Target VQ reconstruction", "#3375a3"))
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for panel, metric, title in zip(axes, ("valid_l1", "t0_roi_l1", "lpips_three_slices_sum"),
                                  ("Foreground MAE", "T0-mask MAE", "LPIPS, three-slice sum"), strict=True):
        for index, (method, label, color) in enumerate(methods):
            panel.bar(np.arange(3) + (index - 1.5) * 0.19,
                      [table[method, phase][metric] for phase in PHASES], 0.19, label=label, color=color)
        panel.set_xticks(range(3), LABELS)
        panel.set_title(title + " (lower is better)", fontsize=11)
        panel.grid(axis="y", alpha=0.18)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9)
    figure.suptitle("102 validation patients / 538 pairs | equal patient weighting", fontsize=12)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def report(path, complete, evaluation_complete, summary, events, verification):
    validation = [row for row in events if row["event"] == "validation"]
    table = {(row["method"], row["phase"]): row for row in summary["phase_metrics"]}
    lines = ["# Completed registered three-phase ROI32 forecasts", "",
             f"Training completed at {complete['updated_at']}; evaluation completed at "
             f"{evaluation_complete['updated_at']}. Physical/effective batch was 64, with shortened "
             f"batches at validation boundaries: {complete['updates']} optimizer updates, "
             f"{complete['reference_updates']} batch4 reference updates and "
             f"{complete['sampled_pairs']} sampled pairs. The original VQ remained frozen.", "",
             f"Full validation flow loss decreased from {validation[0]['loss']:.6f} to "
             f"{validation[-1]['loss']:.6f}. The final forecast evaluation covers 102 validation "
             "patients / 538 pairs, with 8 generated samples per pair. Metrics average pairs and "
             "samples within each patient, then weight patients equally. This is the development "
             "validation cohort used for checkpoint selection, not an independent test set.", "",
             "## Forecast quality", "",
             "| Phase | Source foreground MAE | Mean8 foreground MAE | Source T0-mask MAE | "
             "Single-sample T0 MAE | Mean8 T0 MAE | Target VQ T0 MAE |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for phase, label in zip(PHASES, LABELS, strict=True):
        source, sample, mean, vq = (table[method, phase] for method in
                                    ("source_copy", "sample", "sample_mean", "vq_target"))
        lines.append(f"| {label} | {source['valid_l1']:.6f} | {mean['valid_l1']:.6f} | "
                     f"{source['t0_roi_l1']:.6f} | {sample['t0_roi_l1']:.6f} | "
                     f"{mean['t0_roi_l1']:.6f} | {vq['t0_roi_l1']:.6f} |")
    lines += ["", "Mean8 improves foreground and fixed-T0-region MAE compared with copying the source. "
              "Individual generated samples improve T0-region MAE but have slightly worse foreground "
              "MAE than source copy in every phase. Both generated samples and their mean have worse "
              "LPIPS than source copy. Mean8 is visibly smoother, and the displayed cases do not "
              "accurately recover local follow-up morphology or enhancement detail. Lower average "
              "intensity error alone does not establish high-quality longitudinal prediction.", "",
              "Target VQ reconstruction errors are much smaller than forecast errors. That diagnostic "
              "uses the true target and is not a forecast. It supports first investigating the "
              "generator's longitudinal modeling; it does not rule out benefits from later codec "
              "fine-tuning. The separate full-visit VQ audit has different visit weighting, so its "
              "values are not directly interchangeable with this target-pair diagnostic.", "",
              "## Figures and verification", "",
              "[All comparisons](index.html) | [Training curves](training_curves.png) | "
              "[Full-cohort metrics](validation_metrics.png)", "",
              "Each phase figure shows source, true target, sample 1, mean of 8 and absolute error "
              "of the mean. Rows are precontrast, first-post and late. The green contour is the "
              "fixed T0 predicted mask, not follow-up lesion ground truth. Enhancement figures "
              "show first-post minus precontrast and late minus precontrast.", "",
              f"The gallery includes all {verification['example_pairs']} prespecified saved pairs "
              f"from only {verification['example_patients']} patients. Sample 1 is fixed without "
              "quality selection; these examples are not a representative cohort sample. Plots use "
              "the maximal-area T0-mask axial slice and one shared source/target-derived intensity "
              "window across phases and prediction columns for each pair. Display windows differ "
              "between pairs; numerical MAE remains unclipped. SSIM clips to [-1,3], and inherited "
              "LPIPS clips to [-1,1]. T0 masks may be cropped at the fixed box boundary.", "",
              f"CPU-only replay passed {verification['aggregate_metric_checks']} aggregate metric "
              f"checks and {verification['saved_volume_metric_checks']} saved-volume MAE checks. "
              "Training/evaluation source and artifact identities passed. No new training or "
              "forecast inference was run to create these views. "
              "[Verification details](verification.json)", ""]
    path.write_text("\n".join(lines))


def main():
    config, baseline = load_config(REPO / "configs/registered_three_phase_roi32_v1.yaml")
    root = Path(config["output_dir"])
    for name in ("fm", "fm_evaluation"):
        marker = read_json(root / name / "COMPLETE.json")
        assert marker["status"] == "passed"
        runtime.verify_contract_files(marker["contract"])
        for artifact in marker["artifacts"]:
            verify_identity(artifact)
    complete = read_json(root / "fm/COMPLETE.json")
    evaluation = root / "fm_evaluation"
    summary = read_json(evaluation / "summary.json")
    phase_rows, enhancement_rows = read_table(evaluation / "images.csv"), read_table(evaluation / "enhancement.csv")
    assert len({row["pair_id"] for row in phase_rows}) == 538
    assert len({row["patient_id"] for row in phase_rows}) == 102
    assert Counter(row["method"] for row in phase_rows) == {"sample": 538 * 8 * 3, "sample_mean": 538 * 3,
                                                          "source_copy": 538 * 3, "vq_target": 538 * 3}
    checks = replay(phase_rows, summary["phase_metrics"], ("method", "phase"), IMAGE_METRICS)
    checks += replay(phase_rows, summary["transition_metrics"], ("method", "phase", "transition"), IMAGE_METRICS)
    checks += replay(enhancement_rows, summary["enhancement_metrics"], ("method", "enhancement", "region"),
                     ("mae", "relative_mae", "signed_bias", "spatial_correlation"))
    destination = evaluation / "review"
    destination.mkdir(exist_ok=True)
    events = [json.loads(line) for line in (root / "fm/metrics.jsonl").read_text().splitlines()]
    curves(destination / "training_curves.png", events)
    metric_plot(destination / "validation_metrics.png", summary)
    manifest = read_json(root / "inventory.json")
    records = [row for row in manifest["pairs"] if row["split"] == "val"][:8]
    visit_ids = {row[key] for row in records for key in ("earlier_visit_id", "later_visit_id")}
    selected = [row for row in manifest["visits"] if row["visit_id"] in visit_ids]
    torch.set_num_threads(2)
    dataset = ThreePhaseCrops(baseline, manifest, records=selected)
    images = {item["visit_id"]: item for item in (dataset[index] for index in range(len(dataset)))}
    figures = [("training_curves.png", "Training and normalized-latent validation"),
               ("validation_metrics.png", "Full-cohort metrics and baselines")]
    windows, image_checks, mean_errors = [], 0, []
    for row in records:
        name = row["pair_id"].replace(":", "_").replace("->", "_to_")
        with np.load(evaluation / "selected_samples" / (name + ".npz"), allow_pickle=False) as saved:
            samples, mean = saved["samples"].astype(np.float32), saved["mean"].copy()
        assert samples.shape == (8, 3, 32, 128, 128) and mean.shape == (3, 32, 128, 128)
        assert np.isfinite(samples).all() and np.isfinite(mean).all()
        mean_error = float(np.max(np.abs(samples.mean(0) - mean)))
        assert mean_error < 0.01, mean_error
        mean_errors.append(mean_error)
        source, target = images[row["earlier_visit_id"]], images[row["later_visit_id"]]
        mask = target["mask"][0].numpy()
        for index, phase in enumerate(PHASES):
            error = np.abs(mean[index] - target["image"][index].numpy())
            recorded = next(record for record in phase_rows if record["pair_id"] == row["pair_id"]
                            and record["method"] == "sample_mean" and record["phase"] == phase)
            for metric, value in (("l1", error.mean()), ("valid_l1", error[target["valid"][index].numpy()].mean()),
                                  ("t0_roi_l1", error[mask].mean())):
                assert math.isclose(float(value), float(recorded[metric]), rel_tol=2e-5, abs_tol=1e-6)
                image_checks += 1
        volumes = {"Source": source["image"].numpy(), "True target": target["image"].numpy(),
                   "Sample 1 of 8": samples[0], "Mean of 8": mean}
        for enhancement, suffix in ((False, "phases"), (True, "enhancement")):
            filename = name + "_" + suffix + ".png"
            window = comparison(destination / filename, row["pair_id"] + " | " + suffix,
                                volumes, mask, enhancement=enhancement)
            figures.append((filename, row["pair_id"] + " | " + suffix))
            windows.append({"pair_id": row["pair_id"], "view": suffix, **window})
    gallery(destination / "index.html", "Registered three-phase FM | completed forecasts", figures)
    verification = {"status": "passed", "aggregate_metric_checks": checks,
               "saved_volume_metric_checks": image_checks, "saved_sample_mean_max_error": max(mean_errors),
               "training_updates": complete["updates"], "sampled_pairs": complete["sampled_pairs"],
               "selection": "all eight prespecified saved forecast pairs; first sample is not selected by quality",
               "example_pairs": len(records), "example_patients": len({row["patient_id"] for row in records}),
               "windows": windows, "source": file_identity(__file__),
               "evaluation_summary": file_identity(evaluation / "summary.json")}
    write_json(destination / "verification.json", verification)
    report(destination / "report.md", complete, read_json(evaluation / "COMPLETE.json"), summary,
           events, verification)
    print({"status": "passed", "aggregate_checks": checks, "volume_checks": image_checks,
           "figures": len(figures), "gallery": str(destination / "index.html")})


if __name__ == "__main__":
    main()

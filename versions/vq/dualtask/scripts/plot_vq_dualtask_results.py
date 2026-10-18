#!/usr/bin/env python3
"""Plot completed dual-task aggregates and epoch histories without patient data."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "runs/vq_dualtask_20261012"
GROUPS = ("raw", "frozen", "finetune")
SEEDS = (17, 43)
KINDS = ("best", "last")
HORIZONS = (1, 2, 3)
LABELS = {"raw": "Raw encoder", "frozen": "Frozen VQ (A)", "finetune": "Fine-tuned VQ (B)"}
SEED_COLORS = {17: "#0072B2", 43: "#D55E00"}


def _read(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label}: expected a finite, nonnegative number")
    return float(value)


def _integer(value, label, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label}: expected an integer >= {minimum}")
    return value


def load_results(output):
    """Require the exact six completed runs and validated best/last world reports."""
    output = output.resolve()
    analysis = _read(output / "analysis.json")
    if (analysis.get("schema") != "vq_dualtask_analysis_v1" or
            analysis.get("status") != "completed" or
            analysis.get("completed_training_runs") != 6 or
            analysis.get("validated_world_reports") != 12):
        raise ValueError("Formal figures require completed analysis: 6 training runs and 12 validated world reports")
    if analysis.get("independent_test") is not False:
        raise ValueError("Expected development validation, not an independent-test analysis")
    _integer(analysis.get("validation_patients"), "validation_patients", minimum=1)
    records = analysis.get("runs")
    if not isinstance(records, list) or len(records) != 6:
        raise ValueError("Expected exactly six run records")
    index, cohort = {}, None
    for record in records:
        key = (record.get("group"), record.get("seed"))
        if key[0] not in GROUPS or type(key[1]) is not int or key[1] not in SEEDS or key in index:
            raise ValueError("Duplicate or unexpected group/seed in analysis")
        name = f"{key[0]}_dynamic_seed{key[1]}"
        run = output / name
        if (record.get("name") != name or record.get("training_completed") is not True or
                Path(record.get("source", "")).resolve() != run):
            raise ValueError(f"{name}: run name, completion, or source path mismatch")
        history = _read(run / "epoch_history.json")
        if not isinstance(history, list) or not history:
            raise ValueError(f"{name}: empty or invalid epoch history")
        train, validation = [], []
        for epoch, row in enumerate(history, start=1):
            if type(row.get("epoch")) is not int or row["epoch"] != epoch:
                raise ValueError(f"{name}: epoch history must be continuous from 1")
            train.append(_number(row["training"]["task"], f"{name} epoch {epoch} pCR BCE"))
            score = row["validation"]
            if score.get("selection_metric") != "patient_equal_prefix_nll":
                raise ValueError(f"{name}: unexpected validation selection metric")
            validation.append(_number(score["selection_score"], f"{name} epoch {epoch} validation NLL"))
        pcr = record["pcr"]
        for kind in KINDS:
            saved = pcr[kind]
            epoch = _integer(saved["epoch"], f"{name} {kind} epoch", minimum=1)
            nll = _number(saved["selection_nll"], f"{name} {kind} NLL")
            if epoch > len(history) or not math.isclose(nll, validation[epoch - 1], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"{name}: {kind} NLL does not match its saved epoch")
        if pcr["last"]["epoch"] != len(history):
            raise ValueError(f"{name}: final history epoch differs from last checkpoint")
        if not math.isclose(pcr["best"]["selection_nll"], min(validation), rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{name}: pCR best is not the minimum recorded validation NLL")
        ratios = {}
        for kind in KINDS:
            world = record["world"][kind]
            if world.get("identity_validated") is not True:
                raise ValueError(f"{name}: {kind} world checkpoint identity not validated")
            values, counts = [], []
            for horizon in HORIZONS:
                metric = world["metrics"][f"horizon_{horizon}"]
                patients = _integer(metric["patients"], f"{name} H{horizon} patients")
                targets = _integer(metric["prefix_targets"], f"{name} H{horizon} prefix targets")
                near = _integer(metric["autonomous_near_zero_copy_patients"], f"{name} H{horizon} near-zero copy")
                if near > patients or patients > analysis["validation_patients"] or targets < patients:
                    raise ValueError(f"{name}: invalid horizon coverage")
                ratio = metric["autonomous_relative_to_copy"]
                values.append(None if ratio is None else _number(ratio, f"{name} {kind} H{horizon} ratio"))
                if patients == 0 and ratio is not None:
                    raise ValueError(f"{name}: empty horizon must have an undefined ratio")
                counts.append((patients, targets))
            if cohort is None:
                cohort = counts
            elif counts != cohort:
                raise ValueError("Horizon cohorts differ between runs/checkpoints")
            ratios[kind] = values
        index[key] = {"record": record, "train": train, "validation": validation, "ratios": ratios}
    return analysis, index, cohort


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "grid.color": "#D9D9D9", "grid.alpha": 0.6,
        "grid.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })


def _save(fig, figures, name):
    paths = []
    for suffix in ("png", "pdf"):
        path = figures / f"{name}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_learning_curves(analysis, index, figures):
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharey=True)
    ceiling = max(value for run in index.values() for key in ("train", "validation")
                  for value in run[key]) * 1.12
    for row, group in enumerate(GROUPS):
        for col, seed in enumerate(SEEDS):
            ax, run = axes[row, col], index[group, seed]
            epochs = range(1, len(run["train"]) + 1)
            ax.plot(epochs, run["train"], color="#0072B2", linewidth=1.6, label="Train pCR BCE")
            ax.plot(epochs, run["validation"], color="#D55E00", linewidth=1.6,
                    label="Validation patient-equal prefix NLL")
            best = run["record"]["pcr"]["best"]
            ax.axvline(best["epoch"], color="#777777", linestyle=":", linewidth=1)
            ax.scatter(best["epoch"], best["selection_nll"], marker="*", s=100,
                       color="#222222", zorder=4, label="pCR-selected best")
            ax.set(title=f"{LABELS[group]} | seed {seed} | best epoch {best['epoch']}",
                   xlabel="Epoch within this stage",
                   xlim=(0.5, len(run["train"]) + 0.5))
            ax.set_ylim(0, max(ceiling, 0.1))
            if col == 0:
                ax.set_ylabel("BCE / NLL (lower is better)")
            ax.grid(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3, fontsize=9)
    fig.suptitle("pCR learning curves: six dual-task runs", y=0.995, fontsize=15)
    fig.text(0.5, 0.02,
             f"Development validation: {analysis['validation_patients']} patients; no independent test. "
             "Training BCE uses sampled legal prefixes.\n"
             "Validation NLL averages legal prefixes within each patient, then patients. A/B epochs restart independently.",
             ha="center", va="bottom", fontsize=9)
    fig.subplots_adjust(top=0.90, bottom=0.10, hspace=0.43, wspace=0.16)
    return _save(fig, figures, "pcr_learning_curves")


def plot_copy_ratios(analysis, index, cohort, figures):
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.4), sharey=True)
    finite = [value for run in index.values() for values in run["ratios"].values()
              for value in values if value is not None]
    ceiling = max([1.0, *finite]) * 1.14
    for ax, group in zip(axes, GROUPS):
        undefined = []
        for seed in SEEDS:
            for kind in KINDS:
                values = index[group, seed]["ratios"][kind]
                ax.plot(HORIZONS, [math.nan if value is None else value for value in values],
                        color=SEED_COLORS[seed], linestyle="-" if kind == "best" else "--",
                        marker="o" if kind == "best" else "s", markersize=5, linewidth=1.6)
                missing = [f"H{horizon}" for horizon, value in zip(HORIZONS, values) if value is None]
                if missing:
                    undefined.append(f"s{seed} {kind}: {', '.join(missing)}")
        ax.axhline(1, color="#555555", linestyle=":", linewidth=1.3)
        ax.set(title=LABELS[group], xlabel="Autonomous forecast horizon", xticks=HORIZONS,
               xlim=(0.8, 3.2), ylim=(0, ceiling))
        ax.grid(True)
        if undefined:
            ax.text(0.03, 0.03, "Undefined ratio (empty/near-zero copy):\n" + "\n".join(undefined),
                    transform=ax.transAxes, fontsize=8, va="bottom",
                    bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"})
    axes[0].set_ylabel("Autonomous L1 / own copy L1")
    handles = [Line2D([], [], color=SEED_COLORS[seed], linestyle="-" if kind == "best" else "--",
                      marker="o" if kind == "best" else "s", label=f"Seed {seed} {kind}")
               for seed in SEEDS for kind in KINDS]
    handles.append(Line2D([], [], color="#555555", linestyle=":", label="Copy baseline = 1"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=5, fontsize=9)
    fig.suptitle("Autonomous latent forecasting relative to copy", y=0.995, fontsize=15)
    coverage = "; ".join(f"H{h}: {patients} patients / {targets} prefix-targets"
                         for h, (patients, targets) in zip(HORIZONS, cohort))
    fig.text(0.5, 0.015,
             f"Development validation ({analysis['validation_patients']} patients); {coverage}.\n"
             "Each ratio uses its checkpoint's EMA teacher; copy holds the initial true latent. "
             "Ratio < 1 means better than own copy.\n"
             "Best is selected by pCR NLL. Ratios describe each latent space; raw L1 is not pooled across teachers.",
             ha="center", va="bottom", fontsize=8.5)
    fig.subplots_adjust(top=0.80, bottom=0.25, wspace=0.16)
    return _save(fig, figures, "autonomous_copy_ratios")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Completed experiment directory containing analysis.json and six run directories")
    parser.add_argument("--figures", type=Path, help="Figure directory (default: OUTPUT/figures)")
    args = parser.parse_args(argv)
    try:
        analysis, index, cohort = load_results(args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Cannot plot: {error}\n")
    figures = args.figures or args.output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _style()
    for path in [*plot_learning_curves(analysis, index, figures),
                 *plot_copy_ratios(analysis, index, cohort, figures)]:
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

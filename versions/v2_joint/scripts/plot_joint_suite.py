#!/usr/bin/env python3
"""Plot aggregate-only joint training metadata without loading patient data/models."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("t0", "dynamic")
SEEDS = (17, 43, 71)
COLORS = {"t0": "#2166ac", "dynamic": "#b35806"}
CAPTION = ("Reused development validation: 102 patients; no independent test. "
           "T0: 48 updates/epoch; dynamic all-prefix: 167 updates/epoch.")


def read_json(path, default=None):
    """A missing file is expected for queued runs; malformed files remain errors."""
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def save_figure(fig, output, stem):
    paths = []
    for extension in ("png", "pdf"):
        path = output / f"{stem}.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_histories(suite_root, output, status, aggregate):
    states = {r["name"]: r.get("state", "unknown") for r in status.get("runs", [])}
    completed = sum(v == "completed" for v in states.values())
    complete = completed == 6 and aggregate.get("complete", False)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4), sharex="row", sharey=True)
    for row, profile in enumerate(PROFILES):
        for col, seed in enumerate(SEEDS):
            ax = axes[row, col]
            name = f"joint_{profile}_seed{seed}"
            history = read_json(suite_root / name / "history.json", [])
            state = states.get(name, "queued")
            title = "Fixed T0" if profile == "t0" else "Dynamic all-prefix"
            ax.set_title(f"{title} | seed {seed} | {state}", fontsize=10)
            valid = [h for h in history if finite(h.get("updates"))
                     and finite(h.get("train", {}).get("fused"))
                     and finite(h.get("validation", {}).get("selection"))]
            if valid:
                x = [h["updates"] for h in valid]
                ax.plot(x, [h["train"]["fused"] for h in valid], color="#4575b4",
                        linewidth=1.8, label="Train fused BCE")
                ax.plot(x, [h["validation"]["selection"] for h in valid], color="#d73027",
                        linewidth=1.8, label="Val patient-equal fused NLL")
                # The stored early-stopping counter identifies accepted best updates,
                # including min_delta; the raw minimum can select a different epoch.
                accepted = [h for h in valid if h.get("bad_epochs") == 0]
                best = accepted[-1] if accepted else min(valid, key=lambda h: h["validation"]["selection"])
                ax.scatter([best["updates"]], [best["validation"]["selection"]],
                           marker="*", s=115, color="#d73027", edgecolor="white",
                           linewidth=0.5, zorder=4, label="Selected best (so far)")
                ax.axvline(best["updates"], color="#777777", alpha=0.45, linestyle=":", linewidth=1)
                ax.text(0.97, 0.05, f"Best: epoch {best['epoch']}\nNLL {best['validation']['selection']:.4f}",
                        ha="right", va="bottom", transform=ax.transAxes, fontsize=9)
                ax.legend(fontsize=7.5, loc="upper right")
            else:
                ax.text(0.5, 0.5, "No completed epoch yet", ha="center", va="center",
                        transform=ax.transAxes, color="#666666")
            ax.grid(alpha=0.22)
            ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
            ax.set_xlabel("Cumulative optimizer updates")
            if col == 0:
                ax.set_ylabel("BCE / NLL (lower is better)")
    prefix = "Complete" if complete else "Partial"
    fig.suptitle(f"{prefix}: joint training curves ({completed}/6 runs completed)", fontsize=15, y=0.985)
    fig.text(0.5, 0.012, CAPTION, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return save_figure(fig, output, "training_curves"), complete


def plot_completed_stages(output, aggregate):
    """Require all three verified terminal runs before showing a profile mean."""
    verified = {}
    for profile in PROFILES:
        verified[profile] = {r.get("seed") for r in aggregate.get("runs", [])
                             if r.get("profile") == profile and r.get("state") == "completed"
                             and r.get("verified") is True}
    ready = [p for p in PROFILES if verified[p] == set(SEEDS)]
    if not ready:
        # Avoid leaving a stale summary when a copied/partial suite replaces inputs.
        for ext in ("png", "pdf"):
            (output / f"stage_fused_three_seed.{ext}").unlink(missing_ok=True)
        return [], ready
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for ax, metric, label in zip(axes, ("nll", "auroc"),
                                 ("Fused pCR NLL (lower is better)", "Fused pCR AUROC (higher is better)")):
        for profile in ready:
            rows = [r for r in aggregate.get("aggregate", []) if r.get("profile") == profile
                    and r.get("branch") == "fused" and r.get("metric") == metric
                    and r.get("three_seeds_complete") is True and r.get("seeds") == 3
                    and finite(r.get("mean")) and finite(r.get("sample_std"))]
            rows.sort(key=lambda r: r["stage"])
            if not rows:
                continue
            x = [int(r["stage"][1:]) for r in rows]
            display = "Fixed T0" if profile == "t0" else "Dynamic all-prefix"
            ax.errorbar(x, [r["mean"] for r in rows], yerr=[r["sample_std"] for r in rows],
                        color=COLORS[profile], marker="o", markersize=6, capsize=4,
                        linewidth=1.8, label=f"{display} (3 seeds)")
        ax.set_xticks(range(4), ("T0", "T1", "T2", "T3"))
        ax.set_xlim(-0.2, 3.2)
        ax.set_xlabel("Observed prefix landmark")
        ax.set_ylabel(label)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=9)
    partial = "Partial: " if len(ready) != 2 or not aggregate.get("complete") else ""
    fig.suptitle(f"{partial}completed profile results, selected best checkpoints", fontsize=13)
    fig.text(0.5, 0.035, "Markers: 3-seed mean; error bars: sample SD (ddof=1), not confidence intervals.",
             ha="center", fontsize=9)
    fig.text(0.5, 0.005, "Development validation only; no independent test. Dynamic stage cohorts: 102 / 100 / 94 / 86.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.085, 1, 0.92))
    return save_figure(fig, output, "stage_fused_three_seed"), ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, default=ROOT / "runs/joint_three_seed_20261009")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/joint_three_seed_20261009/figures")
    args = parser.parse_args()
    suite_root, output = args.suite_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status = read_json(suite_root / "status.json", {})
    aggregate = read_json(output.parent / "aggregate.json", {})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    curves, complete = plot_histories(suite_root, output, status, aggregate)
    stages, ready = plot_completed_stages(output, aggregate)
    print(json.dumps({"complete": complete, "three_seed_profiles": ready,
                      "figures": curves + stages}, indent=2))


if __name__ == "__main__":
    main()

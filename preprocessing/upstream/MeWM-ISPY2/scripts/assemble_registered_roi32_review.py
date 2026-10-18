"""Combine best/final ROI32 review artifacts without changing model outputs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


def assemble(root):
    collection = {}
    for key, directory in (("best", root), ("last", root / "last")):
        complete = json.loads((directory / "COMPLETE.json").read_text())
        if complete["status"] != "passed":
            raise ValueError(f"{key} review has not passed")
        manifest = json.loads((directory / "review-manifest.json").read_text())
        if len(manifest["cases"]) != 30 or len({row["patient_id"] for row in manifest["cases"]}) != 30:
            raise ValueError("Each checkpoint needs 30 different patients")
        if manifest["checkpoint"] != key:
            raise ValueError("Checkpoint label is incorrect")
        if key == "last":
            for case in manifest["cases"]:
                case["frames"] = "last/" + case["frames"]
        manifest["diagnostics_path"] = ("last/" if key == "last" else "") + "diagnostics.json"
        collection[key] = manifest
    for best, last in zip(collection["best"]["cases"], collection["last"]["cases"], strict=True):
        for name in ("patient_id", "pair_id", "anchor", "window", "seeds"):
            if best[name] != last[name]:
                raise ValueError(f"Checkpoint comparison changed {name}")
    script = "window.REVIEW_COLLECTION = " + json.dumps(collection, ensure_ascii=True, allow_nan=False) + ";\nwindow.REVIEW_DATA = window.REVIEW_COLLECTION.best;\n"
    (root / "review-data.js").write_text(script)
    template = Path(__file__).with_name("roi32_review")
    for filename in ("index.html", "review.css", "review.js"):
        shutil.copy2(template / filename, root / filename)
    images = []
    for directory in (root / "frames", root / "last/frames"):
        images.extend(directory.rglob("*.png"))
    for path in images:
        with Image.open(path) as image:
            if image.size != (1024, 512):
                raise ValueError(f"Wrong atlas dimensions: {path.name}")
            image.verify()
    lines = ["# ROI32: 30-Patient Best and Final Checkpoint Review", "",
             "Each checkpoint uses the same 30 validation patients, one pair per patient, five pairs of each time type. "
             "Two clipping cases are deliberately included. Selection is seeded and does not use generation quality. "
             "This is a diagnostic subset of the development validation cohort, not an independent test.", "",
             "Both checkpoints use their saved velocity-network EMA and associated condition encoder. "
             "Best selects reference35000/actual11988; final is reference100000/actual20113. "
             "The VQ codec, source images, seeds and Heun solver remain the same. "
             "10/20/30/50 integration steps use20/40/60/100 network evaluations. "
             "Two noises per requested step and eight additional matched noises at25steps were generated per patient per checkpoint.", "",
             "## Source Similarity and Diversity", "",
             "Metrics use the intersection of source and target valid voxels. Each patient has equal weight. "
             "Diversity is the mean absolute difference over all28 pairs of eight25-step samples. "
             "Nonzero diversity rules out identical outputs for these sampled noises; it does not establish calibrated uncertainty or rule out partial mode collapse.", "",
             "| Checkpoint | Source to target MAE | Sample to source MAE | Sample to target MAE | Pairwise sample MAE | Mean8 to target MAE | Closer to source | Near-zero diversity |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key in ("best", "last"):
        diagnostic = collection[key]["diagnostics"]
        aggregate = diagnostic["aggregate"]
        values = [aggregate[name]["mean"] for name in ("source_target_mae", "sample_source_mae", "sample_target_mae", "sample_pairwise_mae", "mean_target_mae")]
        lines.append("| " + key + " | " + " | ".join(f"{value:.6f}" for value in values) + f" | {diagnostic['closer_to_source_cases']}/30 | {diagnostic['near_zero_diversity_cases']}/30 |")
    lines += ["", "## Matched-Noise Step Comparison", "", "| Checkpoint | Steps | Sample to target MAE | Sample to source MAE | Difference from50steps |", "|---|---:|---:|---:|---:|"]
    for key in ("best", "last"):
        for row in collection[key]["diagnostics"]["step_comparison"]:
            lines.append(f"| {key} | {row['steps']} | {row['target_mae']:.6f} | {row['source_mae']:.6f} | {row['difference_from_50']:.6f} |")
    lines += ["", "## Slice Position Sensitivity", "",
              "The reference Z maximizes fixed T0-mask area. Generated and real target slices can be inspected "
              "from Z-10 through Z+10. The center checkbox selects21positions or exactly20neighbors. "
              "Positions outside0-31 remain empty. Z spacing is2mm, so the window extends20mm each way. "
              "A minimum slice error uses the real target to pick the position and is optimistic. "
              "It is not a valid replacement for fixed-coordinate forecast metrics or proof of misregistration.", "",
              "| Checkpoint | Same-slice generation MAE | Best-window generation MAE | Same-slice source MAE | Best-window source MAE |", "|---|---:|---:|---:|---:|"]
    for key in ("best", "last"):
        aggregate = collection[key]["diagnostics"]["aggregate"]
        values = [aggregate[name]["mean"] for name in ("same_slice_generated_mae", "best_window_generated_mae", "same_slice_source_mae", "best_window_source_mae")]
        lines.append("| " + key + " | " + " | ".join(f"{value:.6f}" for value in values) + " |")
    lines += ["", "## Interpretation Limits", "",
              "The actual SymmFlow path conditions the target branch on the observed source branch; preserving patient anatomy is expected. "
              "Its supervised velocity target contains the real later latent, not a source-copy objective. "
              "Training-condition coverage includes all3462pairs for treatment, phase and interval; entire condition fields are not missing. "
              "DCE0 contains less direct enhancement information than postcontrast imaging. "
              "Learning shared anatomy more strongly than individual response, residual misregistration and ambiguous response given baseline "
              "conditions are plausible contributors. Their causal shares have not been established by this review. "
              "Mean8 smooths variable predictions and can lower voxel errors while losing detail; it is not a single forecast.", "",
              "All displayed intensities come from prepared registered crops and actual decoded samples. "
              "The joint display window uses pooled source/target foreground percentiles0.5/99.5, shared across methods; "
              "a fixed[-1,3] window is also available. Contours are the original fixed T0 first-post model mask, not follow-up segmentations. "
              "No NIfTI header was used to define or move slice geometry.", "",
              "[Offline viewer](index.html)", ""]
    (root / "report.md").write_text("\n".join(lines))
    result = {"status": "passed", "checkpoints": 2, "unique_patients": 30, "pngs_checked": len(images),
              "matched_patients_noise_windows": True, "requested_step_predictions": 480, "noise_diagnostic_predictions": 480,
              "selection_per_time_pair": {pair: sum(row["time_pair"] == pair for row in collection["best"]["cases"])
                                          for pair in sorted({row["time_pair"] for row in collection["best"]["cases"]})}}
    (root / "assembly_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    assemble(parser.parse_args().root.resolve())

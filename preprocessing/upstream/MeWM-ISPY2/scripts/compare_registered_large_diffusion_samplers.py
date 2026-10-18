from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mewm_ispy2.vqgan_visualization import orthogonal_planes


IMAGE_WINDOW = (-2.0, 3.0)
GROUP_LABELS = ("DDPM-200", "DDIM-200", "DDIM-50")


def _load_group(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, torch.Tensor]]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    tensors = {
        stage_id: torch.load(
            path / f"stage-{stage_id}-tensors.pt",
            map_location="cpu",
            weights_only=True,
        )
        for stage_id in (1, 2, 3)
    }
    return manifest, tensors


def _axial(volume: torch.Tensor, centroid: tuple[int, int, int]) -> np.ndarray:
    return (
        orthogonal_planes(volume, centroid)["axial"]
        .detach()
        .float()
        .cpu()
        .numpy()
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare fixed registered-large cases across diffusion samplers"
    )
    parser.add_argument("--ddpm-dir", type=Path, required=True)
    parser.add_argument("--ddim200-dir", type=Path, required=True)
    parser.add_argument("--ddim50-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    groups = [
        _load_group(path)
        for path in (args.ddpm_dir, args.ddim200_dir, args.ddim50_dir)
    ]
    reference_cases = groups[0][0]["cases"]
    reference_keys = [
        (case["transition_id"], case["seed"], case["stage_id"])
        for case in reference_cases
    ]
    for manifest, tensors in groups[1:]:
        keys = [
            (case["transition_id"], case["seed"], case["stage_id"])
            for case in manifest["cases"]
        ]
        if keys != reference_keys or set(tensors) != {1, 2, 3}:
            raise ValueError("sampler outputs do not contain the same fixed cases")

    figure, axes = plt.subplots(3, 5, figsize=(18, 10), constrained_layout=True)
    column_names = ("Source", *GROUP_LABELS, "Target")
    for row, reference_case in enumerate(reference_cases):
        stage_id = reference_case["stage_id"]
        centroid = tuple(reference_case["centroid_zyx"])
        reference_tensor = groups[0][1][stage_id]
        source = reference_tensor["source_zscore"]
        target = reference_tensor["target_zscore"]
        mask = reference_tensor["source_mask"]
        for _manifest, tensors in groups[1:]:
            candidate = tensors[stage_id]
            if not torch.equal(candidate["source_zscore"], source):
                raise ValueError("source tensors differ across sampler outputs")
            if not torch.equal(candidate["target_zscore"], target):
                raise ValueError("target tensors differ across sampler outputs")
        volumes = (
            source,
            groups[0][1][stage_id]["generated_zscore"],
            groups[1][1][stage_id]["generated_zscore"],
            groups[2][1][stage_id]["generated_zscore"],
            target,
        )
        mask_plane = _axial(mask, centroid)
        for column, (axis, volume) in enumerate(zip(axes[row], volumes, strict=True)):
            axis.imshow(
                _axial(volume, centroid),
                cmap="gray",
                vmin=IMAGE_WINDOW[0],
                vmax=IMAGE_WINDOW[1],
                origin="lower",
            )
            if np.any(mask_plane > 0):
                axis.contour(
                    mask_plane,
                    levels=[0.5],
                    colors=["#00FFFF"],
                    linewidths=0.8,
                    origin="lower",
                )
            if row == 0:
                axis.set_title(column_names[column])
            axis.axis("off")
        for column, (manifest, _tensors) in enumerate(groups, start=1):
            case = manifest["cases"][row]
            axes[row, column].text(
                0.02,
                0.98,
                f"MAE {case['whole_volume_target_mae']:.4f}\n"
                f"{case['sampling_seconds']:.1f} s",
                transform=axes[row, column].transAxes,
                va="top",
                ha="left",
                color="white",
                fontsize=9,
                bbox={
                    "facecolor": "black",
                    "alpha": 0.65,
                    "pad": 3,
                    "edgecolor": "none",
                },
            )
        axes[row, 0].text(
            0.02,
            0.98,
            f"Stage {stage_id} | seed {reference_case['seed']}",
            transform=axes[row, 0].transAxes,
            va="top",
            ha="left",
            color="white",
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 3, "edgecolor": "none"},
        )

    figure.suptitle(
        "Epoch 135 fixed-case sampler comparison (axial tumor-centered slices)",
        fontsize=15,
        fontweight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Matched real/frozen/mixed-phase reconstruction and enhancement figures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mewm_ispy2 import registered_roi32_vq as vq
from mewm_ispy2.registered_roi32_data import file_identity, read_json, verify_identity, write_json
from mewm_ispy2.registered_roi32_evaluation import gallery
from mewm_ispy2.registered_three_phase_data import PHASES, ThreePhaseCrops, load_config


def plot(path, title, volumes, mask, *, difference=False):
    z = int(np.argmax(mask.sum(axis=(1, 2))))
    if difference:
        displayed = {name: np.stack((value[1] - value[0], value[2] - value[0])) for name, value in volumes.items()}
        row_names = ("First-post minus pre", "Late minus pre")
    else:
        displayed, row_names = volumes, PHASES
    real = displayed["Real"]
    if difference:
        upper = max(float(np.percentile(np.abs(real[:, mask]), 99.5)), 0.1)
        lower, cmap = -upper, "coolwarm"
    else:
        foreground = real[real != 0]
        lower = float(np.percentile(foreground, 1))
        upper = max(float(np.percentile(foreground, 99.5)), float(np.percentile(real[:, mask], 99.5)), lower + 0.1)
        cmap = "gray"
    figure, axes = plt.subplots(len(row_names), 3, figsize=(12, 3.8 * len(row_names)), constrained_layout=True)
    figure.suptitle(title, fontsize=12)
    for row, phase in enumerate(row_names):
        for column, (method, value) in enumerate(displayed.items()):
            panel = axes[row, column]
            picture = panel.imshow(value[row, z], cmap=cmap, vmin=lower, vmax=upper)
            if mask[z].any():
                panel.contour(mask[z], levels=[0.5], colors=["lime"], linewidths=0.6)
            panel.set_title(f"{phase}: {method}", fontsize=10)
            panel.set_axis_off()
        figure.colorbar(picture, ax=list(axes[row]), shrink=0.7, label="Frozen intensity units")
    figure.savefig(path, dpi=120)
    plt.close(figure)


@torch.no_grad()
def main():
    config, baseline = load_config(REPO / "configs/registered_three_phase_roi32_v1.yaml")
    root = Path(config["output_dir"])
    marker = read_json(root / "vq_mixed_probe/COMPLETE.json")
    for value in marker["artifacts"]:
        verify_identity(value)
    protocol = marker["protocol"]
    torch.set_num_threads(4)
    manifest = read_json(root / "inventory.json")
    by_id = {row["visit_id"]: row for row in manifest["visits"]}
    patients = sorted({by_id[visit]["patient_id"] for visit in protocol["val_visits"]})[:2]
    chosen = []
    for patient in patients:
        records = sorted([by_id[visit] for visit in protocol["val_visits"] if by_id[visit]["patient_id"] == patient], key=lambda row: row["visit_id"])
        chosen.extend([records[0], records[-1]] if len(records) > 1 else records)
    dataset = ThreePhaseCrops(baseline, manifest, records=chosen)
    original, identity = vq.load_frozen(baseline, torch.device("cpu"))
    if identity != protocol["base_codec"]:
        raise ValueError("Paired figure base codec changed")
    payload = torch.load(root / "vq_mixed_probe/last.pt", map_location="cpu", weights_only=False)
    if payload["step"] != 100 or payload["protocol"] != protocol:
        raise ValueError("Paired figure mixed-phase checkpoint differs")
    tuned = vq.build_autoencoder(baseline).eval().requires_grad_(False)
    prefix = "autoencoder."
    tuned.load_state_dict({key[len(prefix):]: value for key, value in payload["model"].items() if key.startswith(prefix)}, strict=True)
    del payload
    destination = root / "vq_mixed_probe/figures"
    destination.mkdir(exist_ok=True)
    figures = []
    for index in range(len(dataset)):
        item = dataset[index]
        image = item["image"][:, None]
        before, _ = original(image)
        after, _ = tuned(image)
        volumes = {"Real": image[:, 0].numpy(), "Frozen": before[:, 0].numpy(), "Mixed 100": after[:, 0].numpy()}
        if any(not np.isfinite(value).all() for value in volumes.values()):
            raise ValueError("Non-finite paired reconstruction")
        name = item["visit_id"].replace(":", "_")
        for difference, suffix in ((False, "phases"), (True, "enhancement")):
            path = destination / f"{name}_{suffix}.png"
            plot(path, item["visit_id"] + " | " + suffix, volumes, item["mask"][0].numpy(), difference=difference)
            figures.append((path.name, item["visit_id"] + " | " + suffix))
    gallery(destination / "index.html", "Registered ROI32 | frozen versus mixed-phase VQ", figures)
    write_json(destination / "selection.json", {"policy": "first_two_sorted_validation_patients_first_and_last_available_visits",
               "visit_ids": [row["visit_id"] for row in chosen], "base_codec": identity,
               "mixed_codec": file_identity(root / "vq_mixed_probe/last.pt"),
               "window": "shared_real_derived_foreground_and_T0_ROI_99.5_percentile", "examples": len(chosen)})


if __name__ == "__main__":
    main()

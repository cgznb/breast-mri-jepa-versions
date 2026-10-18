"""Prespecified short mixed-phase fine-tuning comparison on a VQ copy."""

from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from mewm_ispy2 import registered_roi32_runtime as runtime
from mewm_ispy2 import registered_roi32_vq as vq
from mewm_ispy2.registered_roi32_data import file_identity, read_json, verify_identity, write_json
from mewm_ispy2.registered_roi32_evaluation import image_metrics, write_csv
from mewm_ispy2.registered_three_phase_audit import enhancement_metrics, grouped_metrics
from mewm_ispy2.registered_three_phase_data import PHASES, ThreePhaseCrops, load_config


def select_records(manifest, split, patients, visits_per_patient=None):
    groups = defaultdict(list)
    for row in manifest["visits"]:
        if row["fold"] == split and all(row["phase_sources"]):
            groups[row["patient_id"]].append(row)
    rng = random.Random(2026091603 if split == "train" else 2026091604)
    selected = []
    for patient in rng.sample(sorted(groups), patients):
        records = sorted(groups[patient], key=lambda row: row["visit_id"])
        if visits_per_patient is not None and len(records) > visits_per_patient:
            records = rng.sample(records, visits_per_patient)
        selected.extend(records)
    return sorted(selected, key=lambda row: row["visit_id"])


def main():
    config, baseline = load_config(REPO / "configs/registered_three_phase_roi32_v1.yaml")
    root = Path(config["output_dir"]) / "vq_mixed_probe"
    root.mkdir(parents=True, exist_ok=True)
    if (root / "COMPLETE.json").exists():
        complete = read_json(root / "COMPLETE.json")
        if complete["status"] != "passed":
            raise ValueError("Mixed-phase probe completion is invalid")
        verify_identity(complete["protocol"]["base_codec"])
        for artifact in complete["artifacts"]:
            verify_identity(artifact)
        print(json.dumps({"stage": "already_complete", "artifacts_verified": True}), flush=True)
        return
    device = torch.device("cpu")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    runtime.seed_all(2026091605)
    manifest = read_json(Path(config["output_dir"]) / "inventory.json")
    if (root / "protocol.json").exists():
        saved_selection = read_json(root / "protocol.json")
        visits = {row["visit_id"]: row for row in manifest["visits"]}
        training = [visits[visit] for visit in saved_selection["train_visits"]]
        validation = [visits[visit] for visit in saved_selection["val_visits"]]
    else:
        training = select_records(manifest, "train", 32, visits_per_patient=2)
        validation = select_records(manifest, "val", 16)
    original = file_identity(Path(baseline["output_dir"]) / "vq/best.pt")
    protocol = {"schema": "registered_three_phase_mixed_vq_probe_v1", "base_codec": original,
                "train_visits": [r["visit_id"] for r in training], "val_visits": [r["visit_id"] for r in validation],
                "train_patients": 32, "val_patients": 16, "updates": 100, "batch": 3,
                "sampling": "uniform_training_patient_then_available_selected_visit_all_three_phases",
                "learning_rate": 1e-5, "discriminator_learning_rate": 1e-5,
                "loss": "original_full_VQ_GAN_L1_LPIPS_commitment_2D_3D_GAN_feature_matching",
                "initialization": "copy_selected_autoencoder_codebook_discriminators_fresh_optimizers",
                "normalization": "frozen_original_training_DCE0_shared_across_phases", "precision": "cpu_fp32",
                "selection": "fixed_100_updates_no_validation_checkpoint_selection", "runtime_source": file_identity(__file__),
                "limitation": "Small development-only feasibility comparison; not a full codec training or independent test."}
    with runtime.output_lock(root):
        if (root / "protocol.json").exists() and read_json(root / "protocol.json") != protocol:
            raise ValueError("Mixed-phase probe protocol changed")
        if not (root / "protocol.json").exists():
            write_json(root / "protocol.json", protocol)
        if (root / "COMPLETE.json").exists():
            print(json.dumps({"stage": "already_complete"}), flush=True)
            return
        dataset = ThreePhaseCrops(baseline, manifest, records=training)
        prepared, patients = [], defaultdict(list)
        for index in range(len(dataset)):
            item = dataset[index]
            patients[item["patient_id"]].append(index)
            prepared.append(item)
        print(json.dumps({"stage": "prepared", "training_visits": len(prepared), "validation_visits": len(validation)}), flush=True)
        settings = copy.deepcopy(baseline)
        settings["vq"].update(learning_rate=1e-5, discriminator_learning_rate=1e-5,
                               discriminator_start=0, discriminator_ramp=1, lr_decay_update=1000000)
        system = vq.ROI32VQ(settings).to(device)
        payload = torch.load(original["path"], map_location="cpu", weights_only=False)
        system.load_state_dict(payload["model"], strict=True)
        del payload
        generator, discriminator = vq.optimizers(system, settings)
        checkpoint = root / "last.pt"
        completed, history = 0, []
        if checkpoint.exists():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload["protocol"] != protocol:
                raise ValueError("Probe recovery protocol differs")
            system.load_state_dict(payload["model"])
            generator.load_state_dict(payload["generator"])
            discriminator.load_state_dict(payload["discriminator"])
            completed, history = payload["step"], payload["history"]
            runtime.restore_rng(payload["rng"])
            del payload
        patient_ids = sorted(patients)
        for step in range(completed + 1, protocol["updates"] + 1):
            patient = random.choice(patient_ids)
            item = prepared[random.choice(patients[patient])]
            batch = {"image": item["image"][:, None]}
            metrics = vq.update(system, [batch], generator, discriminator, step=step, device=device, config=settings)
            if any(not math.isfinite(float(value)) for value in metrics.values()):
                raise FloatingPointError("Non-finite mixed-phase VQ update")
            history.append({"step": step, **metrics})
            if step == 1 or step % 10 == 0:
                print(json.dumps({"stage": "fine_tuning", "step": step, "loss": metrics["generator_total"]}), flush=True)
            if step % 25 == 0 or runtime.STOP_REQUESTED:
                runtime.save_checkpoint(checkpoint, {"protocol": protocol, "model": system.state_dict(),
                                         "generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
                                         "step": step, "history": history, "rng": runtime.rng_state()})
            if runtime.STOP_REQUESTED:
                return
        verify_identity(original)
        write_csv(root / "training.csv", history)
        system.eval().requires_grad_(False)
        validation_data = ThreePhaseCrops(baseline, manifest, records=validation)
        rows, enhancements, frozen_rows, frozen_enhancements = [], [], [], []
        with torch.no_grad():
            for index in range(len(validation_data)):
                item = validation_data[index]
                image = item["image"][:, None]
                prediction, _ = system.autoencoder(image)
                common = {"patient_id": item["patient_id"], "visit_id": item["visit_id"], "visit": item["visit"]}
                for phase_index, phase in enumerate(PHASES):
                    metrics = image_metrics(prediction[phase_index:phase_index + 1], image[phase_index:phase_index + 1],
                                            item["mask"][None], item["valid"][phase_index:phase_index + 1, None],
                                            system.perceptual, item["visit_id"])
                    rows.append({**common, "phase": phase, **metrics})
                enhancements += [{**common, **value} for value in enhancement_metrics(prediction[:, 0], image[:, 0], item["mask"][0], item["valid"])]
                frozen = read_json(Path(config["output_dir"]) / "vq_audit/cases" / (item["visit_id"].replace(":", "_") + ".json"))
                frozen_rows.extend(frozen["phase_rows"])
                frozen_enhancements.extend(frozen["enhancement_rows"])
                if index % 10 == 0:
                    print(json.dumps({"stage": "evaluating", "completed": index + 1, "total": len(validation_data)}), flush=True)
        metrics = ("l1", "valid_l1", "t0_roi_l1", "ssim_clamped_neg1_pos3", "lpips_three_slices_sum")
        enhancement_fields = ("mae", "relative_mae", "signed_bias", "spatial_correlation")
        summary = {"status": "passed", "protocol": protocol, "train_visits": len(training), "val_visits": len(validation),
                   "frozen": grouped_metrics(frozen_rows, ["phase"], metrics),
                   "fine_tuned": grouped_metrics(rows, ["phase"], metrics),
                   "frozen_enhancement": grouped_metrics(frozen_enhancements, ["enhancement", "region"], enhancement_fields),
                   "fine_tuned_enhancement": grouped_metrics(enhancements, ["enhancement", "region"], enhancement_fields),
                   "base_codec_unchanged": file_identity(original["path"]) == original}
        write_csv(root / "phases.csv", rows)
        write_csv(root / "enhancement.csv", enhancements)
        write_json(root / "summary.json", summary)
        write_json(root / "COMPLETE.json", {"status": "passed", "protocol": protocol,
                   "artifacts": [file_identity(root / name) for name in ("summary.json", "last.pt", "phases.csv", "enhancement.csv")]})
        print(json.dumps({"stage": "complete", "updates": 100, "validation_visits": len(validation)}), flush=True)


if __name__ == "__main__":
    runtime.install_signals()
    main()

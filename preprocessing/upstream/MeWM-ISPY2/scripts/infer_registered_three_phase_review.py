"""Matched-noise step sweep on thirty distinct validation patients."""

from __future__ import annotations

import argparse
import itertools
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from mewm_ispy2 import registered_roi32_runtime as runtime
from mewm_ispy2 import registered_roi32_vq as vq
from mewm_ispy2.registered_roi32_data import file_identity, read_json, save_npz, verify_identity, write_json
from mewm_ispy2.registered_roi32_fm import bridge
from mewm_ispy2.registered_three_phase_data import PHASES, ThreePhaseCrops, load_config
from mewm_ispy2.registered_three_phase_latents import PhaseLatentPairs
from mewm_ispy2.registered_three_phase_model import SharedPhaseCodec
from mewm_ispy2.registered_three_phase_training import load_selected

STEPS = (10, 20, 30, 50)
SAMPLES = 4
SELECTION_SEED = 20260917


def select_cases(records, seed=SELECTION_SEED, per_transition=5):
    groups = defaultdict(list)
    for record in records:
        if record["split"] == "val":
            groups[(record["earlier_stage"], record["later_stage"])].append(record)
    if len(groups) != 6:
        raise ValueError("Expected all six forward validation transitions")
    rng, used, selected = random.Random(seed), set(), []
    for key in sorted(groups):
        groups[key].sort(key=lambda row: row["pair_id"])
        rng.shuffle(groups[key])
    for _ in range(per_transition):
        for key in sorted(groups):
            record = next((row for row in groups[key] if row["patient_id"] not in used), None)
            if record is None:
                raise ValueError("Insufficient distinct patients for the prespecified strata")
            used.add(record["patient_id"])
            selected.append(record)
    return selected


def reference_slice(mask):
    if mask.shape != (32, 128, 128) or not mask.any():
        raise ValueError("Expected a nonempty ROI32 T0 mask")
    maximum = int(mask.sum(axis=(1, 2)).argmax())
    return min(21, max(10, maximum)), maximum


def cosine(a, b):
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-10 else None


def image_measurements(prediction, source, target, valid, mask, anchor):
    result = []
    for phase, name in enumerate(PHASES):
        p, s, t = (value[phase] for value in (prediction, source, target))
        metrics = {"phase": name}
        for region, keep in (("foreground", valid[phase]), ("t0", mask)):
            pv, sv, tv = (value[keep].astype(np.float64) for value in (p, s, t))
            dp, dt = pv - sv, tv - sv
            real_change = float(np.mean(np.abs(dt)))
            metrics.update({region + "_target_mae": float(np.mean(np.abs(pv - tv))),
                            region + "_source_mae": float(np.mean(np.abs(dp))),
                            region + "_real_change_mae": real_change,
                            region + "_change_ratio": float(np.mean(np.abs(dp))) / max(real_change, 1e-10),
                            region + "_change_cosine": cosine(dp, dt)})
        edge_values = []
        for axis in (1, 2):
            a, b = [slice(None)] * 3, [slice(None)] * 3
            a[axis], b[axis] = slice(1, None), slice(None, -1)
            edge_keep = valid[phase][tuple(a)] & valid[phase][tuple(b)]
            edge_values.append(float(np.abs(np.diff(p, axis=axis))[edge_keep].mean()))
        metrics["xy_edge_magnitude"] = float(np.mean(edge_values))
        keep = valid[phase, anchor]
        nearby = [float(np.abs(p[z] - t[anchor])[keep].mean()) for z in range(anchor - 10, anchor + 11)]
        metrics.update(same_slice_mae=nearby[10], nearby_oracle_mae=min(nearby),
                       nearby_oracle_offset=int(np.argmin(nearby)) - 10)
        result.append(metrics)
    return result


def diversity(samples, target, valid, mask):
    result = []
    for phase, name in enumerate(PHASES):
        pairs = [np.abs(samples[a, phase] - samples[b, phase])
                 for a, b in itertools.combinations(range(len(samples)), 2)]
        result.append({"phase": name,
                       "foreground_pairwise_mae": float(np.mean([p[valid[phase]].mean() for p in pairs])),
                       "t0_pairwise_mae": float(np.mean([p[mask].mean() for p in pairs])),
                       "foreground_voxel_std": float(samples[:, phase].std(axis=0)[valid[phase]].mean()),
                       "target_foreground_std": float(target[phase][valid[phase]].std()),
                       "identical_sample_pairs": sum(bool(np.array_equal(samples[a, phase], samples[b, phase]))
                                                     for a, b in itertools.combinations(range(len(samples)), 2))})
    return result


def analyze_case(directory, record):
    with np.load(directory / "reference.npz", allow_pickle=False) as data:
        source, target, mask, valid = (data[key].copy() for key in ("source", "target", "mask", "valid"))
        source_vq, target_vq = data["source_vq"].copy(), data["target_vq"].copy()
    anchor, mask_max = reference_slice(mask)
    rows, noise = [], []
    for method, value in (("source_copy", source), ("source_vq", source_vq), ("target_vq", target_vq), ("target", target)):
        rows.extend({"method": method, "steps": 0, "sample": -1, **row}
                    for row in image_measurements(value, source, target, valid, mask, anchor))
    step_differences = []
    with np.load(directory / "steps_50.npz", allow_pickle=False) as data:
        final = data["samples"].astype(np.float32)
    for steps in STEPS:
        with np.load(directory / f"steps_{steps}.npz", allow_pickle=False) as data:
            samples, mean = data["samples"].astype(np.float32), data["mean"].copy()
        for sample, value in enumerate(samples):
            rows.extend({"method": "sample", "steps": steps, "sample": sample, **row}
                        for row in image_measurements(value, source, target, valid, mask, anchor))
        rows.extend({"method": "mean", "steps": steps, "sample": -1, **row}
                    for row in image_measurements(mean, source, target, valid, mask, anchor))
        noise.extend({"steps": steps, **row} for row in diversity(samples, target, valid, mask))
        for phase, name in enumerate(PHASES):
            difference = np.abs(samples[:, phase] - final[:, phase])
            step_differences.append({"steps": steps, "phase": name,
                                     "foreground_mae_to_50": float(difference[:, valid[phase]].mean()),
                                     "t0_mae_to_50": float(difference[:, mask].mean())})
    conditioning = None
    if (directory / "condition_control.npz").exists():
        with np.load(directory / "condition_control.npz", allow_pickle=False) as data:
            alternate = data["samples"].astype(np.float32)
        conditioning = [{"phase": name,
                         "foreground_mae": float(np.abs(alternate[:, phase] - final[:, phase])[:, valid[phase]].mean()),
                         "t0_mae": float(np.abs(alternate[:, phase] - final[:, phase])[:, mask].mean())}
                        for phase, name in enumerate(PHASES)]
    real = np.stack((source, target))
    foreground = real[real != 0]
    window = [float(np.percentile(foreground, 1)),
              max(float(np.percentile(foreground, 99.5)), float(np.percentile(real[:, :, mask], 99.5)))]
    receipt = read_json(directory / "inference.json")
    result = {"record": record, "anchor_z": anchor, "max_mask_z": mask_max, "window": window,
              "image_metrics": rows, "diversity": noise, "step_differences": step_differences,
              "condition_control": conditioning, "inference": receipt,
              "reference": file_identity(directory / "reference.npz")}
    write_json(directory / "metrics.json", result)
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config, baseline = load_config(REPO / "configs/registered_three_phase_roi32_v1.yaml")
    root = Path(config["output_dir"])
    output = args.output or root / "fm_review_30cases_20260917"
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(root / "inventory.json")
    selected = select_cases(manifest["pairs"])
    protocol = {"schema": "three_phase_review30_v1", "selection_seed": SELECTION_SEED,
                "selection": "five per forward transition, thirty distinct validation patients, no outcome ranking",
                "pair_ids": [row["pair_id"] for row in selected], "steps": list(STEPS), "samples": SAMPLES,
                "solver": "heun", "seed_base": config["fm"]["validation_seed"],
                "same_noise_across_steps": True, "inference_batch": SAMPLES,
                "checkpoint": file_identity(root / "fm/best-endpoint.pt"),
                "training_completion": file_identity(root / "fm/COMPLETE.json"),
                "source": file_identity(__file__)}
    binding = output / "protocol.json"
    if binding.exists() and read_json(binding) != protocol:
        raise ValueError("Review inference protocol changed; use a new output directory")
    if not binding.exists():
        write_json(binding, protocol)
    for stage in ("fm", "fm_evaluation"):
        marker = read_json(root / stage / "COMPLETE.json")
        assert marker["status"] == "passed"
        runtime.verify_contract_files(marker["contract"])
        for identity in marker["artifacts"]:
            verify_identity(identity)
    torch.set_num_threads(4)
    runtime.seed_all(config["seed"])
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    pairs = PhaseLatentPairs(config, baseline, manifest, "val")
    positions = {row["pair_id"]: i for i, row in enumerate(pairs.records)}
    module = bridge(config)
    model = load_selected(config, baseline, manifest, device)
    codec, _ = vq.load_frozen(baseline, device)
    codec = SharedPhaseCodec(codec)
    visits = {row["visit_id"]: row for row in manifest["visits"]}
    records = [visits[visit] for visit in sorted({row[key] for row in selected for key in ("earlier_visit_id", "later_visit_id")})]
    crops = ThreePhaseCrops(baseline, manifest, records=records)
    crop_index = {row["visit_id"]: i for i, row in enumerate(crops.records)}
    started = time.monotonic()
    cases = []
    for index, record in enumerate(selected):
        name = record["pair_id"].replace(":", "_").replace("->", "_to_")
        directory = output / "cases" / name
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "metrics.json").exists():
            result = read_json(directory / "metrics.json")
            for identity in result["inference"]["artifacts"] + [result["reference"]]:
                verify_identity(identity)
            cases.append(result)
            continue
        seed = config["fm"]["validation_seed"] + positions[record["pair_id"]]
        source = pairs.read(record["earlier_visit_id"])[None].to(device)
        source_batch = source.repeat(SAMPLES, 1, 1, 1, 1)
        items = [{"record": record, "earlier_latent": source[0], "later_latent": source[0]}] * SAMPLES
        conditions = module.collate_pairs(items)["conditions"]
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.cat([torch.randn(source.shape, generator=generator, device=device) for _ in range(SAMPLES)])
        artifacts = []
        for steps in STEPS:
            destination = directory / f"steps_{steps}.npz"
            if not destination.exists():
                with runtime.autocast(device):
                    latent = model.sample(source_batch, conditions, noise, steps=steps, solver="heun")
                    decoded = codec.decode(latent, pairs.statistics)
                samples = decoded.float().cpu().numpy()
                assert samples.shape == (SAMPLES, 3, 32, 128, 128) and np.isfinite(samples).all()
                stored = samples.astype(np.float16)
                assert np.isfinite(stored).all() and np.array_equal(stored.astype(np.float32), samples)
                save_npz(destination, samples=stored, mean=samples.mean(0),
                         latent=latent.float().cpu().numpy().astype(np.float16))
                del latent, decoded
            artifacts.append(file_identity(destination))
            progress = {"status": "running", "completed_cases": index, "total_cases": len(selected),
                        "case_number": index + 1, "steps": steps, "elapsed_seconds": time.monotonic() - started,
                        "updated_at": datetime.now(timezone.utc).isoformat()}
            write_json(output / "progress.json", progress)
            print(progress, flush=True)
        alternative = None
        if index < 12 and record["earlier_stage"] == "T0":
            alternative = next((row for row in pairs.records if row["patient_id"] == record["patient_id"]
                                and row["earlier_stage"] == "T0" and row["later_stage"] != record["later_stage"]), None)
        if alternative is not None:
            destination = directory / "condition_control.npz"
            if not destination.exists():
                items = [{"record": alternative, "earlier_latent": source[0], "later_latent": source[0]}] * SAMPLES
                conditions_alt = module.collate_pairs(items)["conditions"]
                with runtime.autocast(device):
                    latent = model.sample(source_batch, conditions_alt, noise, steps=50, solver="heun")
                    decoded = codec.decode(latent, pairs.statistics)
                save_npz(destination, samples=decoded.float().cpu().numpy().astype(np.float16))
                del latent, decoded
            artifacts.append(file_identity(destination))
        # Only the source latent and forecast metadata are read before all predictions return.
        if not (directory / "reference.npz").exists():
            source_item = crops[crop_index[record["earlier_visit_id"]]]
            target_item = crops[crop_index[record["later_visit_id"]]]
            with runtime.autocast(device):
                source_recon = codec.decode(source, pairs.statistics)[0].cpu().numpy()
                target_recon = codec.decode(pairs.read(record["later_visit_id"])[None].to(device), pairs.statistics)[0].cpu().numpy()
            assert torch.equal(source_item["mask"], target_item["mask"])
            save_npz(directory / "reference.npz", source=source_item["image"].numpy(),
                     target=target_item["image"].numpy(), mask=target_item["mask"][0].numpy(),
                     valid=target_item["valid"].numpy(), source_vq=source_recon, target_vq=target_recon)
        metadata = read_json(visits[record["later_visit_id"]]["metadata_source"]["path"])
        report = read_json(visits[record["later_visit_id"]]["crop_report"]["path"])
        write_json(directory / "inference.json", {"seed": seed, "samples": SAMPLES, "artifacts": artifacts,
                   "target_access": "target tensors first opened after all forecasts returned",
                   "largest_clipped": report["largest_clipped"],
                   "registration_status": metadata.get("registration_status"),
                   "registered_to_visit": metadata.get("registered_to_visit"),
                   "condition_control_pair": alternative["pair_id"] if alternative else None})
        result = analyze_case(directory, record)
        cases.append(result)
        verify_identity(protocol["checkpoint"])
        verify_identity(protocol["source"])
    write_json(output / "results.json", {"protocol": protocol, "cases": cases})
    write_json(output / "COMPLETE.json", {"status": "passed", "cases": len(cases), "patients": len({row["record"]["patient_id"] for row in cases}),
               "main_forecast_volumes": len(cases) * len(STEPS) * SAMPLES,
               "condition_control_cases": sum(row["condition_control"] is not None for row in cases),
               "updated_at": datetime.now(timezone.utc).isoformat(), "protocol": file_identity(binding),
               "results": file_identity(output / "results.json")})
    write_json(output / "progress.json", {"status": "complete", "completed_cases": len(cases),
               "elapsed_seconds": time.monotonic() - started, "updated_at": datetime.now(timezone.utc).isoformat()})
    print({"status": "passed", "output": str(output)}, flush=True)


if __name__ == "__main__":
    main()

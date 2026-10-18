"""Matched-noise sampling and offline slice review for 30 ROI32 patients."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_erosion

from mewm_ispy2 import registered_roi32_fm as fm
from mewm_ispy2 import registered_roi32_runtime as runtime
from mewm_ispy2 import registered_roi32_vq as vq
from mewm_ispy2.registered_roi32_data import (
    CropDataset, file_identity, read_config, read_json, save_npz, timestamp,
    verify_identity, write_json,
)
from mewm_ispy2.registered_roi32_latents import LatentPairs, decode_normalized

STEPS = (10, 20, 30, 50)
TIME_PAIRS = ("T0->T1", "T0->T2", "T0->T3", "T1->T2", "T1->T3", "T2->T3")
SELECTION_SEED = 20260917


def case_name(record):
    return record["pair_id"].replace(":", "_").replace("->", "_")


def select_cases(config):
    root = Path(config["output_dir"])
    inventory = read_json(root / "data/inventory.json")
    records = [row for row in inventory["pairs"] if row["split"] == "val"]
    clipped = {row["patient_id"]: read_json(root / "data/patients" / (row["patient_id"] + ".json"))["largest_clipped"]
               for row in inventory["patients"] if row["patient_id"] in {r["patient_id"] for r in records}}
    rng = np.random.default_rng(SELECTION_SEED)
    selected, used = [], set()
    # Reserve two clipping examples before drawing the remaining distinct patients.
    for pair in ("T0->T2", "T0->T3"):
        pool = [i for i, row in enumerate(records) if row["earlier_stage"] + "->" + row["later_stage"] == pair
                and clipped[row["patient_id"]] and row["patient_id"] not in used]
        index = int(rng.choice(pool))
        selected.append(index)
        used.add(records[index]["patient_id"])
    for pair in TIME_PAIRS:
        count = sum(records[i]["earlier_stage"] + "->" + records[i]["later_stage"] == pair for i in selected)
        pool = [i for i, row in enumerate(records) if row["earlier_stage"] + "->" + row["later_stage"] == pair
                and not clipped[row["patient_id"]] and row["patient_id"] not in used]
        for index in rng.permutation(pool)[:5 - count]:
            selected.append(int(index))
            used.add(records[index]["patient_id"])
    if len(selected) != 30 or len(used) != 30:
        raise ValueError("Expected 30 distinct validation patients")
    selected.sort(key=lambda i: (TIME_PAIRS.index(records[i]["earlier_stage"] + "->" + records[i]["later_stage"]), records[i]["patient_id"]))
    return [{"dataset_index": index, "patient_id": records[index]["patient_id"],
             "pair_id": records[index]["pair_id"], "key": case_name(records[index]),
             "time_pair": records[index]["earlier_stage"] + "->" + records[index]["later_stage"],
             "source_visit": records[index]["earlier_visit_id"], "target_visit": records[index]["later_visit_id"],
             "largest_clipped": clipped[records[index]["patient_id"]]}
            for index in selected]


def setup(config, output, batch_patients, checkpoint):
    root = Path(config["output_dir"])
    completion = read_json(root / "fm_evaluation/COMPLETE.json")
    if completion["status"] != "passed":
        raise ValueError("Final evaluation is incomplete")
    runtime.verify_contract_files(completion["contract"])
    for identity in completion["artifacts"]:
        verify_identity(identity)
    contract = {"schema": "roi32_slice_review_v1", "experiment": str(root),
                "flow": file_identity(root / "fm" / (checkpoint + ".pt")), "codec": file_identity(root / "vq/best.pt"),
                "data": [file_identity(root / "data" / name) for name in ("inventory.json", "normalization.json")],
                "latent_statistics": file_identity(root / "latents/statistics.json"),
                "steps": list(STEPS), "samples": 2, "solver": "heun", "selection_seed": SELECTION_SEED,
                "noise": "original validation_seed + full_validation_pair_index * 8 + sample_index; draw each [1,8,8,32,32] separately",
                "batch_patients": batch_patients, "precision": "bf16"}
    path = output / "selection.json"
    if path.exists():
        selected = read_json(path)
        if selected["contract"] != contract:
            raise ValueError("Existing review has incompatible inputs or settings")
    else:
        selected = {"contract": contract, "cases": select_cases(config),
                    "selection_policy": "Five per time pair, 30 unique validation patients; two clipping cases reserved, otherwise seeded random selection without looking at prediction scores"}
        output.mkdir(parents=True, exist_ok=True)
        write_json(path, selected)
    return selected


@torch.inference_mode()
def generate(config, output, batch_patients, device, checkpoint):
    selected = setup(config, output, batch_patients, checkpoint)
    root = Path(config["output_dir"])
    if device.type != "cuda":
        raise ValueError("Matched BF16 sampling requires a CUDA device")
    torch.set_num_threads(2)
    runtime.seed_all(config["seed"])
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(0.45, device)
    dataset = LatentPairs(config, "val")
    images = CropDataset(config, "val")
    indices = {row["visit_id"]: i for i, row in enumerate(images.records)}
    module = fm.bridge(config)
    model = fm.load_selected(config, device)
    if checkpoint == "last":
        payload = runtime.load_checkpoint(root / "fm/last.pt", fm.contract_for(config), "cpu")
        if payload["model_description"] != model.description:
            raise ValueError("Final checkpoint has a different model configuration")
        model.load_state_dict(payload["model"])
        parameters = dict(model.velocity_model.named_parameters())
        for name, value in payload["ema"]["shadow"].items():
            parameters[name].copy_(value.to(device))
        del payload
    codec, _ = vq.load_frozen(config, device)
    codebook = {name: value.clone() for name, value in codec.quantizer.state_dict().items()}
    missing = []
    for record in selected["cases"]:
        report_path = output / "cases" / (record["key"] + ".json")
        if report_path.exists():
            report = read_json(report_path)
            if report["contract"] != selected["contract"]:
                raise ValueError("Case cache has an incompatible contract")
            verify_identity(report["volumes"])
        else:
            missing.append(record)
    for start in range(0, len(missing), batch_patients):
        group = missing[start:start + batch_patients]
        entries = [dataset[record["dataset_index"]] for record in group for _ in range(2)]
        batch = module.collate_pairs(entries)
        source = batch["earlier_latent"].to(device)
        noises, seeds = [], []
        for record in group:
            case_seeds = []
            for sample in range(2):
                seed = config["fm"]["validation_seed"] + record["dataset_index"] * 8 + sample
                generator = torch.Generator(device=device).manual_seed(seed)
                noises.append(torch.randn((1, 8, 8, 32, 32), device=device, generator=generator))
                case_seeds.append(seed)
            seeds.append(case_seeds)
        noise = torch.cat(noises)
        predicted_steps, latent_steps = [], []
        for steps in STEPS:
            with runtime.autocast(device):
                latent = model.sample(source, batch["conditions"], noise.clone(), steps=steps, solver="heun")
            predictions = [decode_normalized(codec, latent[i:i + 1], dataset.statistics)[0, 0].cpu().numpy()
                           for i in range(len(entries))]
            predicted_steps.append(np.stack(predictions).astype(np.float16))
            latent_steps.append(latent.cpu().numpy().astype(np.float16))
            print(json.dumps({"event": "sampling", "batch_start": start, "patients": len(group), "steps": steps,
                              "completed_patients": len(selected["cases"]) - len(missing) + start}), flush=True)
        generated = np.stack(predicted_steps)
        generated_latents = np.stack(latent_steps)
        for offset, record in enumerate(group):
            source_item, target_item = images[indices[record["source_visit"]]], images[indices[record["target_visit"]]]
            geometry_path = root / "data/patients" / (record["patient_id"] + ".json")
            geometry = read_json(geometry_path)
            prior = read_json(root / "fm_evaluation/cases" / (record["key"] + ".json"))
            verify_identity(prior["samples_identity"])
            with np.load(prior["samples_identity"]["path"], allow_pickle=False) as archive:
                old_samples = archive["samples"][:, 0].copy()
                old_mean = archive["predictive_mean"][0].copy()
                if str(archive["earlier_visit_id"]) != record["source_visit"] or str(archive["later_visit_id"]) != record["target_visit"]:
                    raise ValueError("Saved forecast visit mismatch")
                np.testing.assert_allclose(archive["affine_lps"], geometry["crop_affine_lps"], rtol=0, atol=1e-9)
            with runtime.autocast(device):
                reconstruction, _ = codec(target_item["image"][None].to(device))
            arrays = {"source": source_item["image"][0].numpy(), "target": target_item["image"][0].numpy(),
                      "vq": reconstruction[0, 0].float().cpu().numpy().astype(np.float16),
                      "source_valid": source_item["valid"][0].numpy(), "target_valid": target_item["valid"][0].numpy(),
                      "mask": target_item["mask"][0].numpy(), "generated": generated[:, offset * 2:offset * 2 + 2],
                      "generated_latents": generated_latents[:, offset * 2:offset * 2 + 2],
                      "saved25_samples": old_samples, "saved25_mean": old_mean,
                      "affine_lps": np.asarray(geometry["crop_affine_lps"]), "seeds": np.asarray(seeds[offset])}
            if not all(np.isfinite(value).all() for value in arrays.values()):
                raise FloatingPointError("Review contains non-finite arrays")
            destination = output / "cases" / (record["key"] + ".npz")
            save_npz(destination, **arrays)
            write_json(destination.with_suffix(".json"), {"contract": selected["contract"], "case": record,
                       "volumes": file_identity(destination), "original_samples": prior["samples_identity"],
                       "geometry": file_identity(geometry_path), "created_at": timestamp()})
        write_json(output / "progress.json", {"event": "sampling", "completed": len(selected["cases"]) - len(missing) + start + len(group),
                                               "total": 30, "updated_at": timestamp()})
    for number, record in enumerate(selected["cases"], start=1):
        destination = output / "noise25" / (record["key"] + ".npz")
        marker = destination.with_suffix(".json")
        if marker.exists():
            saved = read_json(marker)
            if saved["contract"] != selected["contract"]:
                raise ValueError("Noise comparison cache has a different checkpoint")
            verify_identity(saved["volumes"])
            continue
        batch = module.collate_pairs([dataset[record["dataset_index"]] for _ in range(8)])
        noise = torch.cat([torch.randn((1, 8, 8, 32, 32), device=device,
                          generator=torch.Generator(device=device).manual_seed(config["fm"]["validation_seed"] + record["dataset_index"] * 8 + sample))
                           for sample in range(8)])
        with runtime.autocast(device):
            latent = model.sample(batch["earlier_latent"].to(device), batch["conditions"], noise, steps=25, solver="heun")
        samples = np.stack([decode_normalized(codec, latent[i:i + 1], dataset.statistics)[0, 0].cpu().numpy()
                            for i in range(8)]).astype(np.float16)
        save_npz(destination, samples=samples, predictive_mean=samples.astype(np.float32).mean(0))
        write_json(marker, {"contract": selected["contract"], "volumes": file_identity(destination),
                           "steps": 25, "samples": 8, "physical_batch": 8, "created_at": timestamp()})
        print(json.dumps({"event": "noise_comparison", "checkpoint": checkpoint, "completed": number, "total": 30}), flush=True)
    if any(not torch.equal(value, codec.quantizer.state_dict()[name]) for name, value in codebook.items()):
        raise RuntimeError("Frozen codebook changed")
    from ispy2_symmflow.training.schema import fit_condition_schema
    _, schema_provenance = fit_condition_schema(dataset.inventory["pairs"], config["fm"]["conditions"])
    coverage = {name: {key: value for key, value in details.items() if key != "observed_categories"}
                for name, details in schema_provenance["train_observed_coverage"].items()}
    write_json(output / "generation_complete.json", {"status": "passed", "patients": 30, "predictions": 240,
               "contract": selected["contract"], "codebook_unchanged": True, "updated_at": timestamp(),
               "noise_comparison": "Fresh matched-seed Heun25, eight samples in one batch per patient, for both checkpoints",
               "condition_coverage": coverage, "unavailable_conditions": schema_provenance["unavailable_fields"]})


def mae(first, second, support):
    return float(np.mean(np.abs(first - second)[support], dtype=np.float64))


def diagnose(values, anchor):
    source, target = values["source"], values["target"]
    support = values["source_valid"] & values["target_valid"]
    region = values["mask"] & support
    samples = values["saved25_samples"].astype(np.float32)
    true_change = target - source
    mean = values["saved25_mean"]
    pairs = list(itertools.combinations(range(8), 2))
    source_error = mae(source, target, support)
    sample_target = float(np.mean([mae(sample, target, support) for sample in samples]))
    sample_source = float(np.mean([mae(sample, source, support) for sample in samples]))
    diversity = float(np.mean([mae(samples[i], samples[j], support) for i, j in pairs]))
    denominator = float(np.sum(true_change[support].astype(np.float64) ** 2))
    projection = float(np.sum((mean - source)[support].astype(np.float64) * true_change[support]) / denominator) if denominator > 1e-12 else None
    window = range(max(0, anchor - 10), min(32, anchor + 11))
    target_slice = target[anchor]
    slice_support = values["target_valid"][anchor]
    shifted_generated = [mae(samples[0, z], target_slice, slice_support) for z in window]
    shifted_source = [mae(source[z], target_slice, slice_support) for z in window]
    return {"source_target_mae": source_error, "sample_target_mae": sample_target,
            "sample_source_mae": sample_source, "mean_target_mae": mae(mean, target, support),
            "mean_source_mae": mae(mean, source, support), "sample_pairwise_mae": diversity,
            "sample_std_mean": float(samples.std(axis=0)[support].mean()),
            "diversity_to_true_change": diversity / max(source_error, 1e-12),
            "mean_change_projection": projection,
            "roi_sample_target_mae": float(np.mean([mae(sample, target, region) for sample in samples])) if region.any() else None,
            "roi_source_target_mae": mae(source, target, region) if region.any() else None,
            "same_slice_generated_mae": mae(samples[0, anchor], target_slice, slice_support),
            "best_window_generated_mae": min(shifted_generated),
            "best_window_generated_offset": list(window)[int(np.argmin(shifted_generated))] - anchor,
            "same_slice_source_mae": mae(source[anchor], target_slice, slice_support),
            "best_window_source_mae": min(shifted_source),
            "best_window_source_offset": list(window)[int(np.argmin(shifted_source))] - anchor}


def atlas(volume, window):
    gray = np.rint(np.clip((volume.astype(np.float32) - window[0]) / (window[1] - window[0]), 0, 1) * 255).astype(np.uint8)
    return np.concatenate([np.concatenate([np.flipud(gray[z]) for z in range(row * 8, row * 8 + 8)], axis=1)
                           for row in range(4)], axis=0)


def publish(output, icon_bundle):
    selected = read_json(output / "selection.json")
    completed = read_json(output / "generation_complete.json")
    if completed["status"] != "passed" or completed["contract"] != selected["contract"]:
        raise ValueError("Generation gate has not passed")
    runtime.verify_contract_files(selected["contract"])
    cases, diagnostics, step_rows = [], [], []
    for number, record in enumerate(selected["cases"], start=1):
        artifact = read_json(output / "cases" / (record["key"] + ".json"))
        verify_identity(artifact["volumes"])
        with np.load(artifact["volumes"]["path"], allow_pickle=False) as archive:
            values = {name: archive[name] for name in archive.files}
        noise_marker = read_json(output / "noise25" / (record["key"] + ".json"))
        verify_identity(noise_marker["volumes"])
        with np.load(noise_marker["volumes"]["path"], allow_pickle=False) as archive:
            values["saved25_samples"] = archive["samples"]
            values["saved25_mean"] = archive["predictive_mean"]
        mask = values["mask"]
        anchor = int(mask.sum(axis=(1, 2)).argmax())
        pooled = np.concatenate([values["source"][values["source_valid"]], values["target"][values["target_valid"]]])
        window = np.percentile(pooled, [0.5, 99.5]).tolist()
        if window[1] <= window[0]:
            raise ValueError("Degenerate image display window")
        frames = output / "frames" / record["key"]
        frames.mkdir(parents=True, exist_ok=True)
        volumes = {key: values[key] for key in ("source", "target", "vq")}
        for step_index, steps in enumerate(STEPS):
            for sample in range(2):
                volumes[f"s{steps}_{sample}"] = values["generated"][step_index, sample]
        for sample in range(8):
            volumes[f"s25_{sample}"] = values["saved25_samples"][sample]
        volumes["mean25"] = values["saved25_mean"]
        for key, volume in volumes.items():
            for preset, limits in (("joint", window), ("fixed", [-1, 3])):
                Image.fromarray(atlas(volume, limits)).save(frames / f"{key}_{preset}.png", optimize=True)
        edges = np.stack([plane & ~binary_erosion(plane) for plane in mask])
        edge_atlas = atlas(edges.astype(np.float32), [0, 1])
        rgba = np.zeros((*edge_atlas.shape, 4), dtype=np.uint8)
        rgba[..., :3] = (57, 210, 163)
        rgba[..., 3] = edge_atlas
        Image.fromarray(rgba).save(frames / "mask.png", optimize=True)
        diagnostic = diagnose(values, anchor)
        diagnostics.append({"pair_id": record["pair_id"], **diagnostic})
        support = values["source_valid"] & values["target_valid"]
        case_steps = []
        for step_index, steps in enumerate(STEPS):
            generated = values["generated"][step_index].astype(np.float32)
            row = {"steps": steps, "nfe": steps * 2,
                   "target_mae": float(np.mean([mae(sample, values["target"], support) for sample in generated])),
                   "source_mae": float(np.mean([mae(sample, values["source"], support) for sample in generated])),
                   "sample_diversity_mae": mae(generated[0], generated[1], support),
                   "difference_from_50": float(np.mean([mae(generated[i], values["generated"][-1, i].astype(np.float32), support) for i in range(2)]))}
            case_steps.append(row)
            step_rows.append({"pair_id": record["pair_id"], **row})
        cases.append({**record, "number": number, "anchor": anchor, "window": window,
                      "seeds": values["seeds"].tolist(), "frames": f"frames/{record['key']}",
                      "metrics": diagnostic, "steps": case_steps, "mask_voxels": int(mask.sum())})
        print(json.dumps({"event": "rendering", "completed": number, "total": 30}), flush=True)
    aggregates = {}
    for key in diagnostics[0]:
        if key == "pair_id":
            continue
        finite = [row[key] for row in diagnostics if row[key] is not None]
        aggregates[key] = {"mean": float(np.mean(finite)), "median": float(np.median(finite)),
                           "min": float(np.min(finite)), "max": float(np.max(finite))}
    step_aggregates = []
    for steps in STEPS:
        rows = [row for row in step_rows if row["steps"] == steps]
        step_aggregates.append({"steps": steps, **{key: float(np.mean([row[key] for row in rows]))
                                                  for key in ("target_mae", "source_mae", "sample_diversity_mae", "difference_from_50")}})
    assessment = {"patients": 30, "split": "val", "independent_test": False,
                  "diagnostic_sampling_steps": 25, "diagnostic_samples_per_pair": 8,
                  "support": "Intersection of source and target valid nonzero voxels; ROI also intersects the fixed T0 mask",
                  "near_zero_diversity_cases": sum(row["sample_pairwise_mae"] < 1e-6 for row in diagnostics),
                  "closer_to_source_cases": sum(row["sample_source_mae"] < row["sample_target_mae"] for row in diagnostics),
                  "window_caution": "Minimum slice error over +/-10 is target-informed exploratory matching, not a replacement for same-slice evaluation or proof of registration error",
                  "aggregate": aggregates, "step_comparison": step_aggregates,
                  "condition_coverage": completed["condition_coverage"], "unavailable_conditions": completed["unavailable_conditions"],
                  "cases": diagnostics}
    write_json(output / "diagnostics.json", assessment)
    manifest = {"experiment": "registered_dce0_roi32_firstpostmask_v1", "shape": [32, 128, 128],
                "spacing": [2.0, 0.7032, 0.7032], "steps": list(STEPS), "cases": cases,
                "checkpoint_reference_step": 100000 if Path(selected["contract"]["flow"]["path"]).name == "last.pt" else 35000,
                "checkpoint": "last" if Path(selected["contract"]["flow"]["path"]).name == "last.pt" else "best",
                "selection_policy": selected["selection_policy"],
                "diagnostics": {key: value for key, value in assessment.items() if key != "cases"}}
    write_json(output / "review-manifest.json", manifest)
    (output / "review-data.js").write_text("window.REVIEW_DATA = " + json.dumps(manifest, ensure_ascii=True, allow_nan=False) + ";\n")
    assets = Path(__file__).with_name("roi32_review")
    for name in ("index.html", "review.css", "review.js"):
        shutil.copy2(assets / name, output / name)
    if not icon_bundle.is_file():
        raise ValueError("Local Lucide icon bundle is missing")
    shutil.copy2(icon_bundle, output / "lucide.js")
    write_json(output / "COMPLETE.json", {"status": "passed", "updated_at": timestamp(), "patients": 30,
               "generated_predictions": 240, "noise_comparison_predictions": 240, "pngs": sum(1 for _ in (output / "frames").rglob("*.png")),
               "selection": file_identity(output / "selection.json"), "diagnostics": file_identity(output / "diagnostics.json")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO / "configs/registered_dce0_roi32_firstpostmask_v1.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=("generate", "publish", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-patients", type=int, default=4)
    parser.add_argument("--checkpoint", choices=("best-endpoint", "last"), default="best-endpoint")
    parser.add_argument("--icons", default="/root/autodl-tmp/.codex/roi32_viewer_browser_20260917/node_modules/lucide/dist/umd/lucide.js")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.batch_patients < 1:
        parser.error("--batch-patients must be positive")
    if args.stage in ("generate", "all"):
        generate(read_config(args.config), output, args.batch_patients, torch.device(args.device), args.checkpoint)
    if args.stage in ("publish", "all"):
        publish(output, Path(args.icons))


if __name__ == "__main__":
    main()

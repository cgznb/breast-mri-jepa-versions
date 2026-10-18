"""Rebuild real reduced MRI views on the existing T0 grids; reuse raw VQ latents."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
import argparse
import csv
import json
import os
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from trjepa import PHASES
from trjepa.data import (SCHEMA, CacheReader, atomic_npz, file_identity, read_json,
                         validate_inventory, verify_identity, write_json)


def prepare_view(job):
    upstream, output, visit, latent, normalization = job
    sys.path.insert(0, upstream)
    from mewm_ispy2.registered_three_phase_data import crop_phase, normalize_phase
    from mewm_ispy2.registered_roi32_data import visit_filename
    torch.set_num_threads(1)
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    crop = read_json(visit["crop_report"]["path"])
    metadata = read_json(visit["metadata_source"]["path"])
    name = "views/" + visit_filename(visit["visit_id"])
    path = Path(output)/name
    receipt = path.with_suffix(".json")
    baseline = Path(visit["crop_report"]["path"]).parent.parent
    pre = baseline/"images"/visit_filename(visit["visit_id"])
    roi_path = baseline/"patients"/(visit["patient_id"]+".npz")
    inputs = [visit["crop_report"], visit["metadata_source"], latent, *visit["phase_sources"]]
    cached = {a["path"]: a for a in crop["cache_identities"]}
    inputs += [cached[str(pre)], cached[str(roi_path)]]
    for identity in inputs:
        verify_identity(identity)
    binding = {"inputs": inputs, "normalization": normalization}
    if receipt.exists():
        previous = read_json(receipt)
        if previous["binding"] != binding:
            raise ValueError("Existing MRI preparation binding changed")
        verify_identity(previous["output"])
        return name, previous["output"]
    with np.load(pre, allow_pickle=False) as archive:
        first = normalize_phase(archive["image"].astype(np.float32), archive["valid"].astype(bool), normalization)
        images, support = [first], [archive["coverage"].astype(bool)]
    for source in visit["phase_sources"][1:]:
        raw, valid, covered = crop_phase(source, metadata, crop)
        images.append(normalize_phase(raw, valid, normalization))
        support.append(covered)
    with np.load(roi_path, allow_pickle=False) as archive:
        roi = archive["mask"].astype(bool)[None]
    z = np.load(latent["path"], allow_pickle=False)
    if z.shape != (24, 8, 32, 32) or z.dtype != np.float16 or not np.isfinite(z).all():
        raise ValueError("Expected the original finite raw ROI32 VQ cache")
    atomic_npz(path, image=np.stack(images), latent=z, roi=roi, support=np.stack(support))
    CacheReader(output).read(name)
    identity = file_identity(path)
    write_json(receipt, {"binding": binding, "output": identity})
    return name, identity


def build(args):
    root, output = Path(args.source).resolve(), Path(args.output).resolve()
    inv = read_json(root/"inventory.json")
    cache = read_json(root/"latents/cache_manifest.json")
    previous = read_json(root/"latents/statistics.json")
    if (inv["schema"] != "registered_three_phase_roi32_v1" or tuple(inv["phase_order"]) != PHASES
            or inv["missing_sources"] or read_json(root/"latents/COMPLETE.json")["status"] != "passed"):
        raise ValueError("Expected the completed reduced registered three-phase cohort")
    verify_identity(cache["codec"])
    safe = torch.load(args.codec, map_location="cpu", weights_only=True)
    if safe.get("schema") != "symm_world_codec_v2" or safe["source_identity"] != cache["codec"]:
        raise ValueError("Frozen codec does not belong to the existing latent cache")
    norm = {k: inv["image_normalization"][k] for k in ("mean", "std", "fit_split", "scope")}
    if norm["fit_split"] != "train" or norm["std"] <= 0:
        raise ValueError("Invalid frozen image normalization")
    output.mkdir(parents=True, exist_ok=True)
    source_files = [root/"inventory.json", root/"latents/cache_manifest.json", root/"latents/statistics.json",
                    Path(args.codec), Path(args.labels), Path(__file__), REPO/"trjepa/data.py"]
    source_files += [Path(args.upstream)/"mewm_ispy2"/p for p in
                     ("registered_roi32_data.py", "registered_three_phase_data.py")]
    binding = {"sources": [file_identity(p) for p in source_files], "normalization": norm}
    contract = output/"preparation_contract.json"
    if contract.exists() and read_json(contract) != binding:
        raise ValueError("Preparation code or source contract changed; use a new output directory")
    if not contract.exists():
        write_json(contract, binding)
    if (output/"audit.json").exists():
        from trjepa.data import ensure_assets
        ensure_assets(output, read_json(output/"inventory.json"))
        return read_json(output/"audit.json")
    latents = {Path(a["path"]).name: a for a in cache["files"]}
    if len(latents) != len(inv["visits"]):
        raise ValueError("Visit/latent coverage differs")
    labels = {}
    with Path(args.labels).open() as handle:
        for row in csv.DictReader(handle):
            if row["pid"] in labels:
                raise ValueError("Duplicate outcome patient")
            value = float(row["pCR"])
            if value not in (0, 1):
                raise ValueError("Nonbinary observed pCR label")
            labels[row["pid"]] = int(value)
    by_id = {v["visit_id"]: v for v in inv["visits"]}
    metadata, crops, approved = {}, {}, {}
    status = Counter()
    totals, squares, count = np.zeros(24), np.zeros(24), 0
    for visit in inv["visits"]:
        verify_identity(visit["metadata_source"]); verify_identity(visit["crop_report"])
        meta = read_json(visit["metadata_source"]["path"])
        crop = read_json(visit["crop_report"]["path"])
        if (crop["shape_zyx"] != [32, 128, 128] or crop["spacing_zyx_mm"] != [2., .7032, .7032]
                or meta["registered_to_visit"] != "T0" or meta["qc_status"] != "ok"
                or meta["intra_visit_registration"]["status"] != "completed"):
            raise ValueError("Reduced source-grid geometry or phase registration is unverified")
        indices = visit["phase_indices"]
        registered = {r["phase_id"] for r in meta["intra_visit_registration"]["registrations"]
                      if r["registered_to_phase_id"] == 0}
        if indices[:2] != [0, 1] or indices[2] <= 1 or not set(indices[1:]) <= registered:
            raise ValueError("Incorrect registered pre/first-post/metadata-late selection")
        vid = visit["visit_id"]
        metadata[vid], crops[vid] = meta, crop
        qc = meta["registration"]["qc"]
        approved[vid] = (meta["registration_status"] == "fixed_reference" or
                         (meta["registration_status"] == "deformable" and qc.get("accepted") is True))
        status[meta["registration_status"]] += 1
        latent = latents[vid.replace(":", "_")+".npy"]
        verify_identity(latent)
        z = np.load(latent["path"], allow_pickle=False)
        if z.shape != (24, 8, 32, 32) or not np.isfinite(z).all():
            raise ValueError("Invalid raw latent")
        if visit["fold"] == "train":
            z = z.astype(np.float64).reshape(24, -1)
            totals += z.sum(1); squares += np.square(z).sum(1); count += z.shape[1]
    training_ids = sorted(v["visit_id"] for v in inv["visits"] if v["fold"] == "train")
    mean = totals/count; std = np.sqrt(np.maximum(0, squares/count-mean**2))
    if (previous["fit_visit_ids"] != training_ids or previous["fit_split"] != "train"
            or not np.allclose(mean, previous["mean"], atol=1e-12, rtol=0)
            or not np.allclose(std, previous["std"], atol=1e-12, rtol=0)):
        raise ValueError("Train-only latent normalization did not replay")
    visits = [{"visit_id": v["visit_id"], "patient_id": v["patient_id"], "split": v["fold"],
               "file": "views/"+v["visit_id"].replace(":", "_")+".npz"} for v in inv["visits"]]
    visit_index = {v["visit_id"]: v for v in visits}
    baselines = {v["patient_id"]: v for v in inv["visits"] if v["visit"] == "T0"}
    pairs = []
    for pair in inv["pairs"]:
        source, target = (by_id[pair[k]] for k in ("earlier_visit_id", "later_visit_id"))
        source_id, target_id = source["visit_id"], target["visit_id"]
        base = baselines[pair["patient_id"]]
        day = lambda v: (datetime.fromisoformat(v["visit_date"])-datetime.fromisoformat(base["visit_date"])).days
        sd, td = day(source), day(target)
        if (pair["interval_missing"] or td-sd != pair["delta_days"] or td <= sd
                or source["patient_id"] != target["patient_id"] or source["fold"] != target["fold"]
                or metadata[source_id]["target_geometry"] != metadata[target_id]["target_geometry"]
                or crops[source_id]["crop_affine_ras"] != crops[target_id]["crop_affine_ras"]
                or source["trial_arm"] != base["trial_arm"] or target["trial_arm"] != base["trial_arm"]):
            raise ValueError("Pair geometry, clinical interval or baseline regimen changed")
        clinical = dict(pair["baseline_clinical"])
        clinical["source_stage"] = source["visit"]
        arm = pair["treatment"].get("treatment_arm")
        spatial = approved[source_id] and approved[target_id]
        pairs.append({"pair_id": pair["pair_id"], "patient_id": pair["patient_id"], "split": pair["split"],
                      "source_day": sd, "target_day": td, "source_file": visit_index[source_id]["file"],
                      "target_file": visit_index[target_id]["file"], "transition": source["visit"]+"->"+target["visit"],
                      "conditions": {"clinical": clinical, "source_day": sd, "delta_days": td-sd,
                                     "horizon_kind": "retrospective_observed",
                                     "treatment": {"regimen": arm, "drugs": [], "known_at_source": True}},
                      "labels": {"pcr": labels[pair["patient_id"]]},
                      "spatial_correspondence_verified": spatial,
                      "registration_qc_note": "Fixed T0 ROI; spatial JEPA only when both registrations are fixed/accepted deformable",
                      "source_geometry": {k: crops[source_id][k] for k in ("shape_zyx", "spacing_zyx_mm", "crop_affine_ras")}})
    result = {"schema": SCHEMA, "phase_order": list(PHASES), "spatial_policy": "source_defined_grid",
              "latent_mode": "prequant", "codec_identity": file_identity(args.codec),
              "image_normalization": norm, "visits": visits, "pairs": pairs,
              "representation_train_files": [v["file"] for v in visits if v["split"] == "train"],
              "roi_policy": "Fixed source-available T0 predicted ROI at every visit; no follow-up segmentation claim",
              "support_policy": "Measured original acquisition coverage resampled to the fixed T0 crop",
              "registration_policy": "Rigid fallback retained; spatial_valid false on any pair touching a fallback",
              "clinical_policy": "Baseline trial arm only, no invented drug schedule; pCR supervision only",
              "provenance": binding}
    validate_inventory(result)
    if shutil.disk_usage(output).free < 25*1024**3:
        raise RuntimeError("Need 25 GiB free for reduced MRI cache and training checkpoints")
    jobs = [(str(Path(args.upstream).resolve()), str(output), v,
             latents[v["visit_id"].replace(":", "_")+".npy"], norm) for v in inv["visits"]]
    identities = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (name, identity) in enumerate(pool.map(prepare_view, jobs), 1):
            identities[name] = identity
            if index == 1 or index % 50 == 0 or index == len(jobs):
                progress = {"status": "preparing", "views": index, "total": len(jobs)}
                write_json(output/"progress.json", progress); print(json.dumps(progress), flush=True)
    for identity in binding["sources"]:
        verify_identity(identity)
    result["cache_identities"] = identities
    write_json(output/"inventory.json", result)
    statistics = {"fit_split": "train", "mean": mean.tolist(), "std": std.tolist(),
                  "unique_files": result["representation_train_files"], "latent_mode": "prequant",
                  "codec_identity": file_identity(args.codec), "inventory_identity": file_identity(output/"inventory.json")}
    write_json(output/"statistics.json", statistics)
    report = {"passed": True, "views": len(visits), "pairs": len(pairs), "views_by_split": dict(Counter(v["split"] for v in visits)),
              "pairs_by_split": dict(Counter(p["split"] for p in pairs)),
              "patients_by_split": {s: len({v["patient_id"] for v in visits if v["split"] == s}) for s in ("train", "val")},
              "image_shape": [3, 32, 128, 128], "latent_shape": [24, 8, 32, 32],
              "registration": dict(status), "spatial_verified_pairs": sum(p["spatial_correspondence_verified"] for p in pairs),
              "train_statistics_replayed": True, "pcr_supervision": True, "response_survival_labels": False,
              "independent_test_split": False, "source_grid": "fixed_T0", "actual_coverage_masks": True}
    write_json(output/"audit.json", report); write_json(output/"progress.json", {"status": "complete", **report})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True); parser.add_argument("--upstream", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--codec", required=True)
    parser.add_argument("--labels", required=True); parser.add_argument("--workers", type=int, default=6)
    print(json.dumps(build(parser.parse_args()), indent=2))

"""Benchmark disposable checkpoint copies while the formal flow is paused."""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mewm_ispy2 import registered_roi32_fm as fm
from mewm_ispy2 import registered_roi32_fm_tuning as tuning
from mewm_ispy2 import registered_roi32_runtime as runtime
from mewm_ispy2.registered_roi32_data import file_identity, read_config, read_json, verify_identity, write_json
from mewm_ispy2.registered_roi32_latents import LatentPairs

import torch


def snapshot(config, destination):
    root = Path(config["output_dir"])
    contract = fm.contract_for(config)
    runtime.verify_contract_files(contract)
    sources = []
    for identity in contract["runtime_sources"]:
        copied = destination / "sources" / Path(identity["path"]).name
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(identity["path"], copied)
        sources.append({"original": identity, "archived": file_identity(copied)})
    checkpoints = {}
    for name in ("last.pt", "best-loss.pt", "best-endpoint.pt"):
        path = root / "fm" / name
        if not path.exists():
            continue
        saved = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        if saved["contract"] != contract:
            raise ValueError("Paused checkpoint does not match the original runtime")
        step = saved["training_state"]["step"]
        del saved
        copied = destination / "checkpoints" / name
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copied)
        checkpoints[name] = {"original": file_identity(path), "archived": file_identity(copied), "step": step}
    result = {"contract": contract, "sources": sources, "checkpoints": checkpoints}
    write_json(destination / "baseline.json", result)
    return result


def benchmark(config, destination, batches, warmup, measured):
    baseline = snapshot(config, destination)
    saved = torch.load(baseline["checkpoints"]["last.pt"]["archived"]["path"], map_location="cpu", mmap=True, weights_only=False)
    dataset = LatentPairs(config, "train")
    module = fm.bridge(config)
    device = torch.device("cuda:0")
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.use_deterministic_algorithms(config["runtime"]["deterministic_algorithms"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    for size in batches:
        stream = None
        model = optimizer = scheduler = ema = trainer = None
        try:
            model, optimizer, scheduler, ema, trainer = fm.build(config, dataset.inventory["pairs"], device)
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            ema.load_state_dict(saved["ema"])
            trainer.micro_step = saved["micro_step"]
            trainer.scaler.load_state_dict(saved["scaler"])
            runtime.restore_rng(saved["rng"])
            previous = tuning.reference_step(saved["training_state"])
            state = saved["stream"]
            stream = runtime.TrainingBatches(dataset, batch_size=state["batch_size"], effective_batch=state["effective_batch"],
                                             seed=state["seed"], workers=config["runtime"]["loader_workers"],
                                             balanced=state["balanced"], collate_fn=module.collate_pairs)
            stream.load_state_dict(state)
            for _ in range(warmup):
                metrics, previous = tuning.update(trainer, tuning.next_batches(stream, size, size, size), config, previous)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            for _ in range(measured):
                metrics, previous = tuning.update(trainer, tuning.next_batches(stream, size, size, size), config, previous)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            row = {"batch_size": size, "status": "passed", "updates": measured, "seconds": elapsed,
                   "seconds_per_update": elapsed / measured, "pairs_per_second": size * measured / elapsed,
                   "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                   "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                   "last_loss": metrics["loss"]}
        except torch.cuda.OutOfMemoryError:
            row = {"batch_size": size, "status": "out_of_memory"}
        finally:
            if stream is not None:
                stream.close()
            del model, optimizer, scheduler, ema, trainer, stream
            gc.collect()
            torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps(row), flush=True)
        write_json(destination / "benchmark.json", {"baseline": file_identity(destination / "baseline.json"),
                   "warmup_updates": warmup, "measured_updates": measured, "results": rows,
                   "scope": "disposable optimizer/model states; actual training data, sampler, BF16 and deterministic settings"})
    for checkpoint in baseline["checkpoints"].values():
        verify_identity(checkpoint["original"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batches", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--measured", type=int, default=30)
    args = parser.parse_args()
    config = read_config(args.config)
    destination = Path(args.output).resolve()
    if (destination / "baseline.json").exists():
        raise ValueError("Use a fresh benchmark directory to preserve the archived baseline")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with runtime.output_lock(Path(config["output_dir"])), Path(f"/tmp/registered_roi32_gpu_{args.gpu}.lock").open("a+") as gpu_lock:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        benchmark(config, destination, args.batches, args.warmup, args.measured)


if __name__ == "__main__":
    main()

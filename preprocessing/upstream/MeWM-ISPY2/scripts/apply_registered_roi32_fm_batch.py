"""Explicitly migrate archived, paused FM checkpoints to a measured batch size."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from mewm_ispy2 import registered_roi32_fm as fm
from mewm_ispy2 import registered_roi32_fm_tuning as tuning
from mewm_ispy2 import registered_roi32_runtime as runtime
from mewm_ispy2.registered_roi32_data import file_identity, read_config, read_json, verify_identity, write_json


def assert_identical(expected, actual):
    if isinstance(expected, torch.Tensor):
        if not torch.equal(expected, actual):
            raise AssertionError("Migration changed a tensor")
    elif isinstance(expected, np.ndarray):
        if not np.array_equal(expected, actual):
            raise AssertionError("Migration changed a NumPy state")
    elif isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError("Migration changed state keys")
        for key in expected:
            assert_identical(expected[key], actual[key])
    elif isinstance(expected, (tuple, list)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            raise AssertionError("Migration changed sequence structure")
        for one, two in zip(expected, actual, strict=True):
            assert_identical(one, two)
    elif expected != actual:
        raise AssertionError("Migration changed a scalar")


def apply(config, archive, batch_size):
    baseline = read_json(archive / "baseline.json")
    benchmark = read_json(archive / "benchmark.json")
    if "batch_tuning" in baseline["contract"]:
        raise ValueError("This migration handles the first batch change only; a tuned checkpoint needs its existing sample clock")
    verify_identity(benchmark["baseline"])
    if not any(row["batch_size"] == batch_size and row["status"] == "passed" for row in benchmark["results"]):
        raise ValueError("The requested batch has not passed the real-data benchmark")
    if baseline["contract"]["configuration"] != config:
        raise ValueError("The base experiment changed after benchmarking")
    for source in baseline["sources"]:
        verify_identity(source["archived"])
        if Path(source["original"]["path"]).resolve() != Path(fm.__file__).resolve():
            verify_identity(source["original"])
    old = torch.load(baseline["checkpoints"]["last.pt"]["archived"]["path"], map_location="cpu", mmap=True, weights_only=False)
    options = {
        "schema": "registered_roi32_fm_batch_tuning_v1",
        "batch_size": batch_size,
        "fallback_batch_size": max(config["fm"]["batch_size"], batch_size // 2),
        "reference_batch_size": config["fm"]["batch_size"],
        "reference_max_updates": config["fm"]["max_updates"],
        "validation_batch_size": old["stream"]["batch_size"],
        "sampler_batch_size": old["stream"]["batch_size"],
        "sampler_effective_batch": old["stream"]["effective_batch"],
        "migration_step": old["training_state"]["step"],
        "target_optimizer_updates": old["training_state"]["step"] + tuning.remaining_updates(config, old["training_state"]["step"], batch_size),
        "sample_budget_pairs": config["fm"]["max_updates"] * config["fm"]["batch_size"],
        "policy": "preserve original sampler, pair budget, sample-based LR/EMA horizon and validation batch/noise",
        "benchmark": file_identity(archive / "benchmark.json"),
        "baseline": file_identity(archive / "baseline.json"),
    }
    del old
    path = tuning.tuning_path(config)
    if path.exists():
        if read_json(path) != options:
            raise ValueError("Another batch tuning policy is already present")
    else:
        for entry in baseline["checkpoints"].values():
            verify_identity(entry["original"])
        write_json(path, options)
    contract = fm.contract_for(config)
    results = {}
    for name, entry in baseline["checkpoints"].items():
        verify_identity(entry["archived"])
        saved = torch.load(entry["archived"]["path"], map_location="cpu", mmap=True, weights_only=False)
        if saved["contract"] != baseline["contract"]:
            raise ValueError("Archived checkpoint has another original contract")
        current_path = Path(entry["original"]["path"])
        current = torch.load(current_path, map_location="cpu", mmap=True, weights_only=False)
        if current["contract"] == contract and current.get("batch_migration") == entry["archived"]:
            for key in saved:
                if key not in ("contract", "training_state"):
                    assert_identical(saved[key], current[key])
            assert_identical({**saved["training_state"], "reference_step": saved["training_state"]["step"]}, current["training_state"])
        else:
            if current["contract"] != baseline["contract"]:
                raise ValueError("Live checkpoint is neither the archived nor the migrated version")
            assert_identical(saved, current)
            del current
            changed = dict(saved)
            changed["contract"] = contract
            changed["training_state"] = {**saved["training_state"], "reference_step": saved["training_state"]["step"]}
            changed["batch_migration"] = entry["archived"]
            runtime.save_checkpoint(current_path, changed)
            current = torch.load(current_path, map_location="cpu", mmap=True, weights_only=False)
            assert_identical(changed, current)
            del changed
        results[name] = {"identity": file_identity(current_path), "original_step": entry["step"], "all_existing_training_state_exact": True}
        del saved, current
        gc.collect()
    report = {"status": "passed", "options": options, "contract": contract, "checkpoints": results}
    write_json(archive / "migration.json", report)
    print(f"Migrated {len(results)} checkpoints; batch={batch_size}, target_updates={options['target_optimizer_updates']}, pairs={options['sample_budget_pairs']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    torch.set_num_threads(2)
    with runtime.output_lock(Path(config["output_dir"])):
        apply(config, Path(args.archive).resolve(), args.batch_size)


if __name__ == "__main__":
    main()

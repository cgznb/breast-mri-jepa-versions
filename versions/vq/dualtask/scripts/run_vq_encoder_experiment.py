#!/usr/bin/env python3
"""Queue frozen VQ encoder runs followed by matched low-LR fine-tuning runs."""
from __future__ import annotations

import argparse
import runpy
import copy
import importlib.util
from pathlib import Path
import sys

from mri_vla_jepa.train_config import load_config


ROOT = Path(__file__).resolve().parents[1]
_release = runpy.run_path(str(ROOT / "scripts/release_runtime.py"))
ORIGINAL = _release["version_directory"](ROOT, "v1")
DEFAULT_OUTPUT = ROOT / "runs/vq_encoder_20261012"
DEFAULT_MANIFEST = ORIGINAL / "data/registered_roi32/manifest.json"
CODEC = Path("/root/autodl-tmp/MAM/symm-fm-world-v2/data/registered_roi32/codec.pt")
# Real B16 gates plus CUDA context/fragmentation margin; see reports/vq_encoder_cuda_20261012.json.
MEMORY_MIB = {"frozen": 9216, "finetune": 13312}


def jobs(output, suite="vq_dynamic"):
    profiles = {"vq_dynamic": ("dynamic",), "vq_t0": ("t0",),
                "vq_both": ("t0", "dynamic")}.get(suite)
    if profiles is None:
        raise ValueError("This launcher accepts only the new VQ encoder suites")
    result = []
    for seed in (17, 43):
        for profile in profiles:
            reference = load_config(ORIGINAL / f"configs/registered_roi32_epoch200_l1_{profile}_seed{seed}.yaml").to_dict()
            reference["model"].update(encoder_type="vq_prequant", vq_hidden_channels=16,
                                      vq_embedding_dim=8, vq_bottleneck_blocks=1,
                                      vq_num_groups=32, vq_activation_checkpointing=True)
            reference["image_cache"] = str(ORIGINAL / "data/registered_roi32/image_cache_20261004")
            frozen_name = f"frozen_{profile}_seed{seed}"
            for stage in ("frozen", "finetune"):
                name = f"{stage}_{profile}_seed{seed}"
                config = ROOT / f"configs/vq_{name}.yaml"
                cfg = load_config(config)
                expected = copy.deepcopy(reference)
                expected.update(encoder_training=stage,
                                encoder_checkpoint=str(CODEC) if stage == "frozen" else None,
                                encoder_lr=None if stage == "frozen" else 1e-5,
                                initialize_from_checkpoint=(str(DEFAULT_OUTPUT / frozen_name / "best.pt")
                                                            if stage == "finetune" else None))
                if _release["portable_config"](cfg.to_dict(), ROOT) != expected:
                    raise ValueError(f"{config.name} differs from its matched L1/control protocol; "
                                     "resource paths must identify the matched parent checkpoint")
                result.append({"name": name, "profile": profile, "seed": seed,
                               "experiment": f"vq_{stage}", "stage": stage,
                               "config": str(config), "output": str(output / name),
                               "status": "queued", "world_device": "cpu",
                               "depends_on": frozen_name if stage == "finetune" else None,
                               "reference": str(ORIGINAL / "runs/registered_roi32_l1_l4_20261004" /
                                                f"l1_{profile}_seed{seed}")})
    return result


def _load_queue():
    path = ROOT / "scripts/run_registered_roi32_experiments.py"
    spec = importlib.util.spec_from_file_location("vq_experiment_queue", path)
    queue = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = queue
    spec.loader.exec_module(queue)
    # Reuse the existing lock/process-adoption/checkpoint-validation machinery.
    queue.SUITE_OUTPUTS = {f"vq_{profile}": DEFAULT_OUTPUT for profile in ("dynamic", "t0", "both")}
    queue.jobs = jobs
    queue.training_memory_mib = lambda job: MEMORY_MIB[job["stage"]]
    queue.__file__ = str(Path(__file__).resolve())
    queue.__doc__ = __doc__
    return queue


def main():
    profile_parser = argparse.ArgumentParser(add_help=False)
    profile_parser.add_argument("--profile", choices=("dynamic", "t0", "both"), default="dynamic",
                                help="MRI prefix protocol to run (default: dynamic)")
    parser = argparse.ArgumentParser(add_help=False, parents=[profile_parser])
    parser.add_argument("--suite", choices=("vq_dynamic", "vq_t0", "vq_both"), help=argparse.SUPPRESS)
    args, remaining = parser.parse_known_args()
    if args.suite is not None and "--worker" not in remaining:
        raise ValueError("Choose --profile rather than --suite with this launcher")
    if args.suite is not None:
        args.profile = args.suite.removeprefix("vq_")
    queue = _load_queue()
    # Detached workers reconstruct exactly the same suite from the encoded profile.
    suite = f"vq_{args.profile}"
    # The queue launches its own --suite argument; accept that only for its worker.
    original_argv = sys.argv
    sys.argv = [str(Path(__file__).resolve()), "--suite", suite, *remaining]
    if not any(value == "--manifest" or value.startswith("--manifest=") for value in remaining):
        sys.argv.extend(["--manifest", str(DEFAULT_MANIFEST)])
    try:
        return queue.main(parent_parsers=[profile_parser], hide_suite=True)
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Queue matched dynamic pCR + JEPA controls, frozen VQ, and VQ fine-tuning."""
from __future__ import annotations

import argparse
import runpy
import copy
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys

from mri_vla_jepa.io import write_json
from mri_vla_jepa.train_config import load_config


ROOT = Path(__file__).resolve().parents[1]
_release = runpy.run_path(str(ROOT / "scripts/release_runtime.py"))
ORIGINAL = _release["version_directory"](ROOT, "v1")
DEFAULT_OUTPUT = ROOT / "runs/vq_dualtask_20261012"
DEFAULT_MANIFEST = ORIGINAL / "data/registered_roi32/manifest.json"
CODEC = Path("/root/autodl-tmp/MAM/symm-fm-world-v2/data/registered_roi32/codec.pt")
SUITE = "vq_dualtask_dynamic"
# Real B16 CUDA gates plus context/cache margin; see reports/vq_dualtask_cuda_20261012.json.
MEMORY_MIB = {"raw": 10240, "frozen": 9216, "finetune": 13312}
WORLD_MEMORY_MIB = 10240


def _analyze_completed(output):
    # Load only after all training and world evaluation artifacts are complete.
    path = ROOT / "scripts/analyze_vq_dualtask_experiment.py"
    spec = importlib.util.spec_from_file_location("vq_dualtask_analysis", path)
    analysis = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = analysis
    spec.loader.exec_module(analysis)
    return analysis.analyze(Path(output), require_complete=True)


def _work_with_analysis(work, output, *args, **kwargs):
    code = work(output, *args, **kwargs)
    if code != 0:
        return code
    try:
        _analyze_completed(output)
    except Exception as exc:
        # A reporting failure must never invalidate completed training or cause
        # the next invocation to train the six jobs again.
        write_json(Path(output) / "analysis.json", {
            "schema": "vq_dualtask_analysis_failure_v1", "status": "failed",
            "output": str(Path(output).resolve()),
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "error_type": type(exc).__name__, "error": str(exc),
        })
        print(f"Dual-task analysis failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Dual-task aggregate analysis completed: {Path(output) / 'analysis.json'}")
    return code


def jobs(output, suite=SUITE):
    if suite != SUITE:
        raise ValueError("This launcher accepts only the dynamic VQ dual-task suite")
    result = []
    for seed in (17, 43):
        reference = load_config(ORIGINAL / f"configs/registered_roi32_epoch200_l1_dynamic_seed{seed}.yaml").to_dict()
        reference.update(jepa_weight=.5, image_cache=str(ORIGINAL / "data/registered_roi32/image_cache_20261004"))
        frozen_name = f"frozen_dynamic_seed{seed}"
        for stage in ("raw", "frozen", "finetune"):
            name = f"{stage}_dynamic_seed{seed}"
            config = ROOT / f"configs/vq_dualtask_{name}.yaml"
            cfg = load_config(config)
            expected = copy.deepcopy(reference)
            if stage != "raw":
                expected["model"].update(encoder_type="vq_prequant", vq_hidden_channels=16,
                                         vq_embedding_dim=8, vq_bottleneck_blocks=1,
                                         vq_num_groups=32, vq_activation_checkpointing=True)
                expected.update(encoder_training=stage,
                                encoder_checkpoint=str(CODEC) if stage == "frozen" else None,
                                encoder_lr=None if stage == "frozen" else 1e-5,
                                initialize_from_checkpoint=(str(DEFAULT_OUTPUT / frozen_name / "best.pt")
                                                            if stage == "finetune" else None))
            if _release["portable_config"](cfg.to_dict(), ROOT) != expected:
                raise ValueError(f"{config.name} differs from the matched L1 plus JEPA protocol; "
                                 "resource paths must identify the matched parent checkpoint")
            result.append({"name": name, "profile": "dynamic", "seed": seed,
                           "experiment": f"vq_dualtask_{stage}", "stage": stage,
                           "config": str(config), "output": str(output / name),
                           "status": "queued", "world_device": "cuda",
                           "need_world_evaluation": True, "require_world_head_trained": True,
                           "depends_on": frozen_name if stage == "finetune" else None,
                           "reference": str(ORIGINAL / "runs/registered_roi32_l1_l4_20261004" /
                                            f"l1_dynamic_seed{seed}")})
    return result


def _load_queue():
    path = ROOT / "scripts/run_registered_roi32_experiments.py"
    spec = importlib.util.spec_from_file_location("vq_dualtask_queue", path)
    queue = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = queue
    spec.loader.exec_module(queue)
    queue.SUITE_OUTPUTS = {SUITE: DEFAULT_OUTPUT}
    queue.jobs = jobs
    queue.training_memory_mib = lambda job: MEMORY_MIB[job["stage"]]
    queue.WORLD_MEMORY_MIB = WORLD_MEMORY_MIB
    queue.__file__ = str(Path(__file__).resolve())
    queue.__doc__ = __doc__
    training_work = queue.work
    queue.work = lambda output, *args, **kwargs: _work_with_analysis(training_work, output, *args, **kwargs)
    return queue


def main():
    profile_parser = argparse.ArgumentParser(add_help=False)
    profile_parser.add_argument("--profile", choices=("dynamic",), default="dynamic",
                                help="MRI prefix protocol (default: dynamic)")
    parser = argparse.ArgumentParser(add_help=False, parents=[profile_parser])
    parser.add_argument("--suite", choices=(SUITE,), help=argparse.SUPPRESS)
    args, remaining = parser.parse_known_args()
    if args.suite is not None and "--worker" not in remaining:
        raise ValueError("The suite argument is reserved for detached workers")
    queue = _load_queue()
    original_argv = sys.argv
    sys.argv = [str(Path(__file__).resolve()), "--suite", SUITE, *remaining]
    if not any(value == "--manifest" or value.startswith("--manifest=") for value in remaining):
        sys.argv.extend(["--manifest", str(DEFAULT_MANIFEST)])
    try:
        return queue.main(parent_parsers=[profile_parser], hide_suite=True)
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())

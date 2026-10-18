#!/usr/bin/env python3
"""Run the four ordered V1 architecture experiments without altering prior runs."""
from __future__ import annotations

import argparse
import runpy
import copy
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

from mri_vla_jepa.io import read_json, write_json
from mri_vla_jepa.train_config import load_config
from mri_vla_jepa.training import _source_snapshot


ROOT = Path(__file__).resolve().parents[1]
_release = runpy.run_path(str(ROOT / "scripts/release_runtime.py"))
ORIGINAL = _release["version_directory"](ROOT, "v1")
DEFAULT_OUTPUT = ROOT / "runs/v1_evolution_20261012"
DEFAULT_MANIFEST = ORIGINAL / "data/registered_roi32/manifest.json"
CODEC = Path("/root/autodl-tmp/MAM/symm-fm-world-v2/data/registered_roi32/codec.pt")
SUITE = "v1_evolution_dynamic"
SEEDS = (17, 43)
# Dependencies are correctness/report barriers, never checkpoint warm starts or
# automatic statistical decisions based on the repeatedly used validation set.
VARIANTS = (
    {"name": "raw_fixed_direct", "stage": 1, "step": "fixed", "control": "l1_pcr_only", "changes": {"encoder_type": "raw"}},
    {"name": "vq_fixed_direct", "stage": 1, "step": "fixed", "control": "raw_fixed_direct", "changes": {}},
    {"name": "vq_copy", "stage": 2, "step": "readout", "control": "vq_fixed_direct", "changes": {"pcr_readout": "copy"}},
    {"name": "vq_forecast_detach", "stage": 2, "step": "readout", "control": "vq_copy", "changes": {"pcr_readout": "forecast_detach"}},
    {"name": "vq_forecast", "stage": 2, "step": "readout", "control": "vq_forecast_detach", "changes": {"pcr_readout": "forecast"}},
    {"name": "vq_forecast_history", "stage": 3, "step": "history", "control": "vq_forecast", "changes": {"pcr_readout": "forecast", "history_delta": True}},
    {"name": "vq_forecast_history_residual", "stage": 3, "step": "residual", "control": "vq_forecast_history", "changes": {"pcr_readout": "forecast", "history_delta": True, "world_residual": True}},
    {"name": "vq_forecast_history_residual_free", "stage": 3, "step": "free", "control": "vq_forecast_history_residual", "changes": {"pcr_readout": "forecast", "history_delta": True, "world_residual": True}, "rollout_weight": .5},
    {"name": "vq_forecast_history_residual_free_lesion", "stage": 4, "step": "features", "control": "vq_forecast_history_residual_free", "changes": {"pcr_readout": "forecast", "history_delta": True, "world_residual": True, "lesion_world_weighting": True}, "rollout_weight": .5},
    {"name": "vq_forecast_history_residual_free_encoder64", "stage": 4, "step": "features", "control": "vq_forecast_history_residual_free", "changes": {"pcr_readout": "forecast", "history_delta": True, "world_residual": True, "encoder_type": "vq_encoder"}, "rollout_weight": .5},
)
STEP_PREREQUISITES = {"fixed": None, "readout": "fixed", "history": "readout", "residual": "history", "free": "residual", "features": "free"}
# Real B16 CUDA train/FP32 world gates measured raw 8088 MiB, VQ <=7854,
# encoder64 8216. Admission adds context/cache margin; diagnostics also gated.
# A budget can admit only two workers even when --max-concurrent=3.
TRAINING_MEMORY_MIB = {variant["name"]: (10240 if variant["name"] in
    {"raw_fixed_direct", "vq_forecast_history_residual_free_encoder64"} else 9216) for variant in VARIANTS}
WORLD_MEMORY_MIB = 10240


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def expected_config(variant, seed):
    cfg = copy.deepcopy(load_config(ORIGINAL / f"configs/registered_roi32_epoch200_l1_dynamic_seed{seed}.yaml").to_dict())
    cfg.update(jepa_weight=.5, rollout_weight=variant.get("rollout_weight", 0.),
               image_cache=str(ORIGINAL / "data/registered_roi32/image_cache_20261004"))
    cfg["model"].update(encoder_type="vq_prequant", fixed_transition_queries=True, pcr_readout="direct", history_delta=False,
                         world_residual=False, lesion_world_weighting=False, lesion_pooling=False,
                         lesion_weight_strength=4.)
    cfg["model"].update(variant["changes"])
    if cfg["model"]["encoder_type"] != "raw":
        cfg["model"].update(encoder_type=variant["changes"].get("encoder_type", "vq_prequant"),
                             vq_hidden_channels=16, vq_embedding_dim=8, vq_bottleneck_blocks=1,
                             vq_num_groups=32, vq_activation_checkpointing=True)
        cfg.update(encoder_training="frozen", encoder_checkpoint=str(CODEC), encoder_lr=None,
                   initialize_from_checkpoint=None)
    return cfg


def jobs(output, suite=SUITE):
    if suite != SUITE:
        raise ValueError("This launcher accepts only the V1 evolution suite")
    if not (ROOT / "train.py").is_file():
        raise FileNotFoundError(f"Missing standalone training entry point: {ROOT / 'train.py'}")
    output = Path(output)
    result = []
    for variant in VARIANTS:
        prerequisite = STEP_PREREQUISITES[variant["step"]]
        dependencies = [f"{parent['name']}_seed{seed}" for parent in VARIANTS
                        if parent["step"] == prerequisite for seed in SEEDS]
        for seed in SEEDS:
            name = f"{variant['name']}_seed{seed}"
            config = ROOT / f"configs/evolution_{name}.yaml"
            cfg = load_config(config)
            if _release["portable_config"](cfg.to_dict(), ROOT) != expected_config(variant, seed):
                raise ValueError(f"{config.name} differs from the locked fresh-init evolution protocol")
            packaged = ROOT / "src/mri_vla_jepa/resources/configs" / config.name
            if packaged.read_bytes() != config.read_bytes():
                raise ValueError(f"Packaged configuration differs: {config.name}")
            result.append({"name": name, "variant": variant["name"], "stage": variant["stage"],
                           "step": variant["step"], "control": variant["control"], "profile": "dynamic",
                           "seed": seed, "experiment": "v1_evolution", "config": str(config),
                           "output": str(output / name), "status": "queued", "world_device": "cuda",
                           "need_world_evaluation": True, "require_world_head_trained": True,
                           "depends_on": dependencies or None, "initialization": "fresh",
                           "reference": str(ORIGINAL / "runs/registered_roi32_l1_l4_20261004" /
                                            f"l1_dynamic_seed{seed}")})
    return result


def dependency_reason(job, planned_jobs):
    names = job.get("depends_on") or []
    if isinstance(names, str):
        names = [names]
    if len(names) != len(set(names)) or job["name"] in names:
        raise ValueError("Dependencies must be distinct prerequisites")
    prerequisites = []
    for name in names:
        matches = [candidate for candidate in planned_jobs if candidate["name"] == name]
        if len(matches) != 1:
            raise ValueError("Each dependency must identify one job in this queue")
        prerequisites.append(matches[0])
    if any(parent["status"] in {"failed", "evaluation_failed", "blocked"} for parent in prerequisites):
        return "dependency_failed"
    # Unlike the VQ fine-tuning queue, require BOTH trained checkpoint validation
    # and completed best/last world evaluation before advancing a design step.
    if any(parent["status"] != "completed" or parent.get("training_validated") is not True
           for parent in prerequisites):
        return "waiting_for_dependency"
    return None


def source_freeze(output):
    """Record exact source text/configs; do not generate or emit checksums."""
    output = Path(output)
    paths = [Path(__file__).resolve(), ROOT / "train.py", ROOT / "scripts/analyze_evolution_experiment.py",
             ROOT / "scripts/run_registered_roi32_experiments.py",
             ROOT / "scripts/analyze_vq_dualtask_experiment.py",
             ROOT / "scripts/evaluate_registered_roi32_world.py", ROOT / "scripts/release_runtime.py"]
    paths += [ROOT / f"configs/evolution_{variant['name']}_seed{seed}.yaml"
              for variant in VARIANTS for seed in SEEDS]
    value = {"schema": "v1_evolution_source_freeze_v1", "runtime_source": _source_snapshot(),
             "files": {str(path.relative_to(ROOT)): path.read_text(encoding="utf-8") for path in paths}}
    path = output / "source_freeze.json"
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Evolution source/configuration changed after queue freeze; use a new experiment")
    else:
        write_json(path, value)
        path.chmod(0o600)
    return value


def _analyze(output, require_complete, manifest=DEFAULT_MANIFEST):
    return _load(ROOT / "scripts/analyze_evolution_experiment.py", "evolution_final_analysis").analyze(
        Path(output), require_complete=require_complete, manifest=manifest)


def _work_with_analysis(work, output, *args, **kwargs):
    source_freeze(output)
    code = work(output, *args, **kwargs)
    try:
        manifest = args[0] if args else kwargs.get("manifest", DEFAULT_MANIFEST)
        _analyze(output, require_complete=code == 0, manifest=manifest)
    except Exception as exc:
        write_json(Path(output) / "analysis_failure.json", {
            "schema": "v1_evolution_analysis_failure_v1", "status": "failed",
            "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
            "error_type": type(exc).__name__, "error": str(exc)})
        print(f"Evolution analysis failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return code


def _load_queue():
    queue = _load(ROOT / "scripts/run_registered_roi32_experiments.py", "v1_evolution_queue")
    queue.SUITE_OUTPUTS = {SUITE: DEFAULT_OUTPUT}
    queue.jobs = jobs
    queue.dependency_reason = dependency_reason
    queue.training_memory_mib = lambda job: TRAINING_MEMORY_MIB[job["variant"]]
    queue.WORLD_MEMORY_MIB = WORLD_MEMORY_MIB
    queue.__file__ = str(Path(__file__).resolve())
    queue.__doc__ = __doc__
    training_work = queue.work
    queue.work = lambda output, *args, **kwargs: _work_with_analysis(training_work, output, *args, **kwargs)
    return queue


def main():
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--profile", choices=("dynamic",), default="dynamic")
    parent.add_argument("--plan", action="store_true", help="Print the validated 20-run plan without starting workers")
    parser = argparse.ArgumentParser(add_help=False, parents=[parent])
    parser.add_argument("--suite", choices=(SUITE,), help=argparse.SUPPRESS)
    args, remaining = parser.parse_known_args()
    if args.plan:
        plan_parser = argparse.ArgumentParser(add_help=False)
        plan_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        planned, extra = plan_parser.parse_known_args(remaining)
        if extra:
            raise ValueError(f"Unsupported plan arguments: {extra}")
        print(json.dumps({"suite": SUITE, "jobs": jobs(planned.output.resolve())}, ensure_ascii=False, indent=2))
        return 0
    if args.suite is not None and "--worker" not in remaining:
        raise ValueError("The suite argument is reserved for detached workers")
    queue = _load_queue()
    original_argv = sys.argv
    sys.argv = [str(Path(__file__).resolve()), "--suite", SUITE, *remaining]
    if not any(value == "--manifest" or value.startswith("--manifest=") for value in remaining):
        sys.argv.extend(["--manifest", str(DEFAULT_MANIFEST)])
    try:
        return queue.main(parent_parsers=[parent], hide_suite=True)
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())

"""One locked, portable T0-CV train/evaluate/profile process.

The scheduler owns matrix state and GPU admission.  This worker never imports
the historical queue/controller and only deletes a run's receipted weights.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import gc
import importlib.util
import math
import os
from pathlib import Path
import random
import re
import resource
import shutil
import tempfile

import numpy as np
import torch
from torch import nn

from .ablation_suite import complete_evaluation_receipt, delete_evaluated_checkpoints
from .data import RawMRIStore
from .diagnostics import _world_head_trained, evaluate_world
from .io import digest, load_checkpoint, read_json, seed_all, stable_hash, write_json
from .model import RawVLAJEPA
from .train_config import load_config
from .training import (CHECKPOINT_SCHEMA, _checkpoint_config_matches, _configure,
                       _make_optimizer, _make_scheduler, _warmup_cosine_factor,
                       _patient_partitions, _protocol, _selection_protocol,
                       _source_snapshot, evaluate, load_trained, train)

PAIRED_INITIALIZATION = "forecast_constructor_then_rng_preserved_direct_head_v1"
EVALUATIONS = (("best", "val"), ("best", "test"), ("last", "val"))


def install_paired_readout_initialization():
    """Align every shared parameter and the post-construction CPU RNG stream."""
    if getattr(RawVLAJEPA.__init__, "__paired_readout_initialization__", False):
        return
    original = RawVLAJEPA.__init__

    def paired_init(self, cfg):
        if cfg.pcr_readout == "direct" and cfg.enable_world_model:
            original(self, replace(cfg, pcr_readout="forecast"))
            with torch.random.fork_rng(devices=[]):
                self.pcr_head = nn.Sequential(nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, 1))
            self.cfg = cfg.validate()
        else:
            original(self, cfg)

    paired_init.__paired_readout_initialization__ = True
    paired_init.__original_constructor__ = original
    RawVLAJEPA.__init__ = paired_init


def durable_json(path, value):
    path = Path(path)
    write_json(path, value)
    path.chmod(0o600)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _inside(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if parent not in path.parents:
        raise ValueError(f"Worker output must be strictly inside suite: {path}")
    return path


@contextmanager
def job_lock(run, phase):
    """A surviving worker excludes a replacement controller's duplicate."""
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (run / ".worker.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another worker holds the job lock: {run}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} phase={phase}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _store(manifest, cfg, normalization=None):
    return RawMRIStore(manifest, cfg.image_shape, cfg.allow_synthetic, normalization,
        image_cache=cfg.image_cache,
        require_lesion_weights=bool(cfg.model.lesion_world_weighting or cfg.model.lesion_pooling))


def _release(device):
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_context(suite, name):
    suite = Path(suite).resolve()
    state = read_json(suite / "matrix.json")
    matches = [job for job in state["jobs"] if job["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one matrix job named {name}")
    job = matches[0]
    _inside(job["output"], suite)
    cfg = load_config(job["config"])
    if cfg.digest != job["config_digest"] or cfg.seed != job["seed"]:
        raise ValueError("Job configuration digest/seed changed")
    if digest(job["manifest"]) != job["manifest_sha256"]:
        raise ValueError("Job manifest changed")
    if digest(state["codec"]) != state["codec_sha256"]:
        raise ValueError("Original codec changed")
    if cfg.encoder_checkpoint and Path(cfg.encoder_checkpoint).resolve() != Path(state["codec"]).resolve():
        raise ValueError("Configured codec differs from the pinned original codec")
    runtime = Path(state["runtime"]).resolve()
    for relative, expected in state["runtime_hashes"].items():
        path = _inside(runtime / relative, runtime)
        if digest(path) != expected:
            raise ValueError(f"Frozen runtime changed: {relative}")
    worker_hash = state["runtime_hashes"].get("src/mri_vla_jepa/t0_cv_worker.py")
    if worker_hash is not None and digest(__file__) != worker_hash:
        raise ValueError("Loaded worker differs from the frozen worker")
    if state["source_identity"] != stable_hash(_source_snapshot()):
        raise ValueError("Loaded training source differs from the frozen source identity")
    if str(cfg.device).startswith("cuda") and cfg.device != "cuda:0":
        raise ValueError("A portable worker must use process-local cuda:0")
    return state, job, cfg


def _worker_identity(state, job):
    return {"schema": "t0_cv_worker_identity_v1", "config_digest": job["config_digest"],
        "manifest_sha256": job["manifest_sha256"], "codec_sha256": state["codec_sha256"],
        "source_identity": state["source_identity"], "worker_sha256": digest(__file__),
        "paired_readout_initialization": PAIRED_INITIALIZATION}


def ensure_worker_identity(state, job, *, create=False):
    path = Path(job["output"]) / "worker_identity.json"
    expected = _worker_identity(state, job)
    if path.exists():
        if read_json(path) != expected:
            raise ValueError("Worker/config/data identity changed since this run started")
    elif create:
        durable_json(path, expected)
    else:
        raise FileNotFoundError("Run lacks its frozen worker identity")
    return expected


def train_job(state, job, cfg):
    install_paired_readout_initialization()
    torch.set_num_threads(cfg.threads)
    run = Path(job["output"])
    ensure_worker_identity(state, job, create=True)
    report_path = run / "train_report.json"
    if report_path.exists() and read_json(report_path).get("completed") is True:
        # A completed, evaluated run may legitimately have no weights left.
        if stable_hash(read_json(run / "config.json")) != cfg.digest:
            raise ValueError("Completed training configuration changed")
        if (run / "best_only_recovery.json").exists() and (run / "best.pt").exists():
            _repair_best_validation(job, cfg, read_json(report_path))
        return {"status": "train_report_completed", "report": str(report_path)}
    if (run / "evaluation_complete.json").exists() or (run / "evaluation_done.json").exists():
        raise ValueError("Evaluated run has no completed training report; refusing to retrain")
    store = _store(job["manifest"], cfg)
    if (run / "best.pt").exists() and not (run / "last.pt").exists():
        _recover_best_only(job, cfg, store)
    report = train(store, cfg, run, resume=(run / "last.pt").is_file())
    if report.get("completed") is True and (run / "best_only_recovery.json").exists():
        _repair_best_validation(job, cfg, report)
    return report


def _validated_best_snapshot(path, cfg, store):
    """Require a complete, internally consistent best-at-epoch-boundary state."""
    saved = load_checkpoint(path)
    if (saved.get("schema") != CHECKPOINT_SCHEMA or cfg.training_unit != "epochs"
            or not _checkpoint_config_matches(saved["config"], cfg)
            or saved.get("source_snapshot") != _source_snapshot()
            or saved.get("data_signature") != store.signature()
            or saved.get("normalization") != store.normalization_state()
            or saved.get("patient_partitions") != _patient_partitions(store)
            or saved.get("landmarks") != cfg.landmarks
            or saved.get("protocol") != _protocol(cfg.landmarks)
            or saved.get("selection_protocol") != _selection_protocol(cfg.landmarks, cfg.selection_metric)
            or saved.get("selection_metric") != cfg.selection_metric
            or saved.get("prediction_training") != cfg.prediction_training
            or saved.get("f1_threshold") != cfg.f1_threshold):
        raise ValueError("Best-only checkpoint config/data/source identity differs")
    epoch, completed = saved.get("epoch_state"), saved.get("completed_steps")
    batches = math.ceil(len(store.by_split["train"]) / cfg.batch_size)
    if (not isinstance(epoch, dict) or type(completed) is not int
            or type(epoch.get("completed_epochs")) is not int
            or not 1 <= epoch["completed_epochs"] <= cfg.max_epochs
            or completed != epoch["completed_epochs"] * batches
            or saved.get("best_epoch") != epoch["completed_epochs"]
            or saved.get("best_step") != completed or saved.get("training_unit") != "epochs"
            or saved.get("phase") != cfg.phase(completed)
            or any(epoch.get(key) != expected for key, expected in {
                "indices": [], "position": 0, "batches_completed": 0, "examples_seen": 0,
                "term_sums": {}, "gradient_norm_sum": 0., "bad_epochs": 0,
                "stopped_early": False}.items())):
        raise ValueError("Best-only checkpoint is not a complete selected epoch boundary")
    history, epochs = saved.get("history"), saved.get("epoch_history")
    if (not isinstance(history, list) or [row.get("step") for row in history] != list(range(1, completed + 1))
            or not isinstance(epochs, list)
            or [row.get("epoch") for row in epochs] != list(range(1, epoch["completed_epochs"] + 1))
            or any(row.get("completed_steps") != row["epoch"] * batches for row in epochs)
            or epochs[-1].get("improved") is not True
            or epochs[-1]["validation"]["selection_score"] != saved.get("best_selection_score")
            or epochs[-1]["validation"]["metrics"]["nll"] != saved.get("best_nll")
            or not math.isfinite(saved["best_selection_score"]) or not math.isfinite(saved["best_nll"])):
        raise ValueError("Best-only checkpoint history/selection is incomplete")
    random.Random().setstate(saved["sampler_state"])
    rng = saved["rng"]
    random.Random().setstate(rng["python"])
    value = rng["numpy"]
    np.random.RandomState().set_state((value[0], np.asarray(value[1], dtype=np.uint32), value[2], value[3], value[4]))
    torch.Generator(device="cpu").set_state(rng["torch"])
    if (not isinstance(rng.get("cuda"), list)
            or any(not isinstance(t, torch.Tensor) or t.dtype != torch.uint8 or t.ndim != 1 for t in rng["cuda"])
            or (str(cfg.device).startswith("cuda") and len(rng["cuda"]) != 1)):
        raise ValueError("Best-only checkpoint CUDA RNG state is incomplete")
    if str(cfg.device).startswith("cuda"):
        torch.Generator(device=cfg.device).set_state(rng["cuda"][0])
    # Construct on CPU and preserve construction RNG. Actual state_dict loading
    # checks every key/shape; Adam moments are checked against their parameters.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = RawVLAJEPA(cfg.model)
    model.load_state_dict(saved["model"], strict=True)
    if any(not bool(torch.isfinite(value).all()) for value in saved["model"].values()):
        raise ValueError("Best-only checkpoint contains non-finite model tensors")
    _configure(model, cfg, cfg.phase(0))
    optimizer = _make_optimizer(model, cfg)
    scheduler = _make_scheduler(optimizer, cfg, batches)
    base_lrs = list(scheduler.base_lrs) if cfg.lr_scheduler == "warmup_cosine" else None
    options = [{key: value for key, value in group.items() if key not in {"params", "lr", "initial_lr"}}
               for group in optimizer.param_groups]
    optimizer.load_state_dict(saved["optimizer"])
    if not optimizer.state:
        raise ValueError("Best-only checkpoint has no initialized optimizer state")
    for group, expected in zip(optimizer.param_groups, options, strict=True):
        if any(group.get(key) != value for key, value in expected.items()):
            raise ValueError("Best-only optimizer options differ from the configuration")
        if group["weight_decay"] != cfg.weight_decay or not math.isfinite(group["lr"]) or group["lr"] <= 0:
            raise ValueError("Best-only optimizer configuration differs")
        for parameter in group["params"]:
            moments = optimizer.state.get(parameter)
            if not moments:  # Unused world parameters legitimately have no moments.
                continue
            step = float(moments["step"])
            if not math.isfinite(step) or not 0 < step <= completed or not step.is_integer():
                raise ValueError("Best-only optimizer step is invalid")
            for key in ("exp_avg", "exp_avg_sq"):
                tensor = moments[key]
                if (tensor.shape != parameter.shape or tensor.dtype != parameter.dtype
                        or not bool(torch.isfinite(tensor).all())
                        or (key == "exp_avg_sq" and bool((tensor < 0).any()))):
                    raise ValueError("Best-only optimizer moments are incomplete")
    saved_scheduler = saved.get("scheduler_state")
    if scheduler is None:
        if saved_scheduler is not None or any(g["lr"] != (cfg.encoder_lr or cfg.lr)
                if g.get("name") == "pretrained_encoder" else g["lr"] != cfg.lr for g in optimizer.param_groups):
            raise ValueError("Best-only unscheduled optimizer/scheduler state differs")
    else:
        if not isinstance(saved_scheduler, dict):
            raise ValueError("Best-only checkpoint lacks its scheduler state")
        if cfg.lr_scheduler == "plateau":
            initial = scheduler.state_dict()
            if any(saved_scheduler.get(key) != initial[key] for key in (
                    "factor", "min_lrs", "patience", "verbose", "cooldown", "mode",
                    "threshold", "threshold_mode", "eps", "mode_worse")):
                raise ValueError("Best-only plateau scheduler options differ from the configuration")
        scheduler.load_state_dict(saved_scheduler)
        progress = completed if cfg.lr_scheduler == "warmup_cosine" else epoch["completed_epochs"]
        if (scheduler.last_epoch != progress
                or scheduler.get_last_lr() != [g["lr"] for g in optimizer.param_groups]):
            raise ValueError("Best-only scheduler progress differs from optimizer")
        if cfg.lr_scheduler == "warmup_cosine" and (scheduler.base_lrs != base_lrs
                or scheduler._step_count != completed + 1
                or saved_scheduler.get("lr_lambdas") != [None] * len(base_lrs)
                or scheduler.get_last_lr() != [lr * _warmup_cosine_factor(cfg, batches, completed) for lr in base_lrs]):
            raise ValueError("Best-only cosine scheduler state is inconsistent")
    diagnostics = saved.get("diagnostics_state")
    if cfg.diagnostics_every_epochs:
        from .diagnostics import make_probe_tasks
        expected = list(range(cfg.diagnostics_every_epochs, epoch["completed_epochs"] + 1, cfg.diagnostics_every_epochs))
        if (not isinstance(diagnostics, dict) or tuple(diagnostics.get("tasks", ())) != make_probe_tasks(
                store, cfg.landmarks, patients=cfg.diagnostics_patients, seed=cfg.diagnostics_seed)
                or [row.get("epoch") for row in diagnostics.get("reports", [])] != expected
                or (expected and diagnostics.get("previous", {}).get("epoch") != expected[-1])):
            raise ValueError("Best-only diagnostic resume state is incomplete")
    result = {"completed_epochs": epoch["completed_epochs"], "completed_steps": completed,
              "diagnostics": diagnostics["reports"] if diagnostics else None}
    del model, optimizer, scheduler, saved
    gc.collect()
    return result


def _recover_best_only(job, cfg, store):
    run = Path(job["output"])
    best, last = run / "best.pt", run / "last.pt"
    if best.is_symlink() or last.exists():
        raise ValueError("Best-only recovery requires a regular best and absent last")
    before = best.stat()
    identity = {"path": str(best.resolve()), "size": before.st_size, "sha256": digest(best)}
    details = _validated_best_snapshot(best, cfg, store)
    reports = details.pop("diagnostics")
    record = {"schema": "t0_cv_best_only_recovery_v1", "status": "source_validated",
        "source": identity, "destination": str(last.resolve()), **details,
        "config_digest": cfg.digest, "created_at": datetime.now(timezone.utc).isoformat()}
    durable_json(run / "best_only_recovery.json", record)
    with tempfile.NamedTemporaryFile(dir=run, prefix="last.recovered.", suffix=".pt.tmp", delete=False) as target:
        temporary = Path(target.name)
        with best.open("rb") as source:
            shutil.copyfileobj(source, target, length=8 * 2**20)
        target.flush()
        os.fsync(target.fileno())
    if (digest(temporary) != identity["sha256"] or best.stat().st_mtime_ns != before.st_mtime_ns
            or best.stat().st_size != before.st_size or last.exists()):
        raise ValueError("Best-only checkpoint changed during atomic recovery")
    os.replace(temporary, last)
    directory = os.open(run, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    record.update(status="last_restored", destination_sha256=identity["sha256"])
    durable_json(run / "best_only_recovery.json", record)
    if reports is not None:
        durable_json(run / "diagnostics.json", reports)


def _repair_best_validation(job, cfg, report):
    run = Path(job["output"])
    target = run / "best_validation.json"
    try:
        previous = read_json(target)
    except (OSError, ValueError):
        previous = None
    if isinstance(previous, dict) and previous.get("selection_score") == report["best_selection_score"]:
        return
    value = evaluate(run / "best.pt", job["manifest"], target, split="val", device=cfg.device,
                     allow_synthetic=cfg.allow_synthetic, bootstrap=0)
    if not math.isclose(value["selection_score"], report["best_selection_score"], rel_tol=0, abs_tol=1e-7):
        raise ValueError("Recovered best validation score differs from the selected checkpoint")
    durable_json(target, value)
    durable_json(run / "best_validation_recovery.json", {"status": "restored", "split": "val",
        "checkpoint_sha256": digest(run / "best.pt"), "checkpoint_epoch": report["best_epoch"],
        "previous_selection_score": previous.get("selection_score") if isinstance(previous, dict) else None,
        "selection_score": value["selection_score"]})
    _release(cfg.device)


def validate_completed_run(job, cfg):
    """Independently bind completed best/last weights to this fold and source."""
    run = Path(job["output"])
    for filename in ("best.pt", "last.pt", "train_report.json", "best_validation.json"):
        if not (run / filename).is_file():
            raise FileNotFoundError(f"Completed run lacks {filename}")
    report = read_json(run / "train_report.json")
    if (report.get("completed") is not True or report.get("landmarks") != cfg.landmarks
            or report.get("selection_protocol") != _selection_protocol(cfg.landmarks, cfg.selection_metric)
            or report.get("stop_reason") not in {"early_stopping", "max_epochs"}
            or report.get("selection_metric") != cfg.selection_metric
            or report.get("f1_threshold") != cfg.f1_threshold
            or report.get("prediction_training") != cfg.prediction_training):
        raise ValueError("Completed training report differs from this experiment")
    epochs = {"best": report.get("best_epoch"), "last": report.get("completed_epochs")}
    if (any(type(epoch) is not int or epoch < 1 for epoch in epochs.values())
            or epochs["best"] > epochs["last"] or epochs["last"] > cfg.max_epochs):
        raise ValueError("Completed report has invalid best/last epochs")
    store = _store(job["manifest"], cfg)
    signature, normalization = store.signature(), store.normalization_state()
    partitions, sources = _patient_partitions(store), _source_snapshot()
    del store
    for kind in ("best", "last"):
        saved = load_checkpoint(run / f"{kind}.pt")
        if (saved.get("schema") != CHECKPOINT_SCHEMA
                or not _checkpoint_config_matches(saved["config"], cfg)
                or saved.get("source_snapshot") != sources
                or saved.get("data_signature") != signature
                or saved.get("normalization") != normalization
                or saved.get("patient_partitions") != partitions
                or saved.get("protocol") != _protocol(cfg.landmarks)
                or saved.get("landmarks") != cfg.landmarks
                or saved.get("selection_protocol") != _selection_protocol(cfg.landmarks, cfg.selection_metric)):
            raise ValueError("Completed checkpoint config/data/source identity differs")
        if (saved.get("epoch_state", {}).get("completed_epochs") != epochs[kind]
                or saved.get("best_nll") != report["best_validation_nll"]
                or saved.get("best_selection_score") != report["best_selection_score"]
                or saved.get("selection_metric") != cfg.selection_metric
                or saved.get("f1_threshold") != cfg.f1_threshold):
            raise ValueError("Completed checkpoint/report epoch or selection differs")
        if cfg.lr_scheduler != "none" and saved.get("scheduler_state") is None:
            raise ValueError("Scheduled training checkpoint lacks scheduler state")
        if kind == "last" and saved.get("completed_steps") != report["completed_steps"]:
            raise ValueError("Completed checkpoint/report progress differs")
        del saved
    if read_json(run / "best_validation.json")["selection_score"] != report["best_selection_score"]:
        raise ValueError("Saved best validation score differs from training report")
    return report


def _training_diagnostics(run, cfg, report):
    path = run / "diagnostics.json"
    reports = read_json(path) if path.exists() else []
    expected = list(range(cfg.diagnostics_every_epochs, report["completed_epochs"] + 1,
                          cfg.diagnostics_every_epochs)) if cfg.diagnostics_every_epochs else []
    if not isinstance(reports, list) or [item.get("epoch") for item in reports] != expected:
        raise ValueError("Training diagnostics do not cover scheduled completed epochs")
    record = run / "diagnostics_record.json"
    durable_json(record, {"schema": "verified_training_diagnostics_v1",
        "completed_epochs": report["completed_epochs"], "every_epochs": cfg.diagnostics_every_epochs,
        "reports": reports})
    return record


def _finish_cleanup(suite, state, job):
    run = Path(job["output"])
    cleanup = delete_evaluated_checkpoints(run, suite, protected_paths=[state["codec"]])
    if not cleanup["all_receipted_weights_absent"]:
        raise ValueError("Receipted checkpoint cleanup is incomplete")
    result = {"completed": True, "config_digest": job["config_digest"],
        "manifest_sha256": job["manifest_sha256"], "cleanup": cleanup,
        "completed_at": datetime.now(timezone.utc).isoformat()}
    done = run / "evaluation_done.json"
    if done.exists():
        prior = read_json(done)
        if any(prior.get(key) != result[key] for key in ("completed", "config_digest", "manifest_sha256", "cleanup")):
            raise ValueError("Existing evaluation completion record differs")
        return prior
    durable_json(done, result)
    return result


def evaluate_job(suite, state, job, cfg):
    install_paired_readout_initialization()
    torch.set_num_threads(cfg.threads)
    ensure_worker_identity(state, job)
    run = Path(job["output"])
    # The receipt commits every prediction and checkpoint identity.  Retrying a
    # partial deletion must not load a checkpoint that is already gone.
    if (run / "evaluation_complete.json").exists():
        return _finish_cleanup(suite, state, job)
    if (run / "best_only_recovery.json").exists() and (run / "train_report.json").exists():
        completed_report = read_json(run / "train_report.json")
        if completed_report.get("completed") is True:
            _repair_best_validation(job, cfg, completed_report)
    report = validate_completed_run(job, cfg)
    epochs = {"best": report["best_epoch"], "last": report["completed_epochs"]}
    checkpoint_hashes = {kind: digest(run / f"{kind}.pt") for kind in epochs}
    diagnostic_record = _training_diagnostics(run, cfg, report)
    required = {"training": {"path": str(run / "train_report.json"), "kind": "json"},
                "diagnostics": {"path": str(diagnostic_record), "kind": "json"}}
    for kind, split in EVALUATIONS:
        target = run / f"evaluation_{kind}_{split}.json"
        result = evaluate(run / f"{kind}.pt", job["manifest"], target,
            split=split, device=cfg.device, allow_synthetic=cfg.allow_synthetic, bootstrap=0)
        if kind == "best" and split == "val" and not math.isclose(
                result["selection_score"], report["best_selection_score"], rel_tol=0, abs_tol=1e-7):
            raise ValueError("Recomputed best validation score differs from selected checkpoint")
        result.update(checkpoint_kind=kind, checkpoint_epoch=epochs[kind],
                      checkpoint_sha256=checkpoint_hashes[kind],
                      variant=job["variant"], fold_id=job["fold_id"],
                      train_seed=cfg.seed, config_digest=cfg.digest)
        durable_json(target, result)
        required[f"{kind}_{split}"] = {"path": str(target), "kind": "classification", "split": split}
        _release(cfg.device)
    trained = bool(cfg.model.enable_world_model and _world_head_trained(cfg))
    for kind, split in EVALUATIONS:
        target = run / f"world_{kind}_{split}.json"
        if not trained:
            result = {"status": "not_applicable", "world_head_trained": False,
                "world_architecture_present": cfg.model.enable_world_model,
                "teacher_present": cfg.model.enable_world_model,
                "reason": "Direct readout with zero world losses has no predictor gradient path",
                "split": split, "landmarks": cfg.landmarks}
        else:
            model, saved_cfg, saved = load_trained(run / f"{kind}.pt", cfg.device)
            if (saved.get("source_snapshot") != _source_snapshot()
                    or saved.get("epoch_state", {}).get("completed_epochs") != epochs[kind]):
                raise ValueError("World checkpoint source/epoch differs from completed report")
            torch.set_num_threads(saved_cfg.threads)
            store = _store(job["manifest"], saved_cfg, saved["normalization"])
            result = evaluate_world(model, store, replace(saved_cfg, device=cfg.device, precision="fp32"),
                                    split=split, batch_size=saved_cfg.validation_batch_size)
            result.update(status="evaluated", runtime_source_matches_checkpoint=True)
            del model, saved, store
        result.update(checkpoint_kind=kind, checkpoint_epoch=epochs[kind],
                      checkpoint_sha256=checkpoint_hashes[kind],
                      variant=job["variant"], fold_id=job["fold_id"],
                      train_seed=cfg.seed, config_digest=cfg.digest,
                      independence_scope="downstream_only", independent_test=False)
        durable_json(target, result)
        required[f"world_{kind}_{split}"] = {"path": str(target), "kind": "world",
                                               "split": split, "applicable": trained}
        _release(cfg.device)
    # Validate every saved classification/world output again, including last_val,
    # before the receipt helper checks complete manifest coverage and fsyncs them.
    for kind, split in EVALUATIONS:
        for prefix in ("evaluation", "world"):
            value = read_json(run / f"{prefix}_{kind}_{split}.json")
            if (value.get("checkpoint_kind") != kind or value.get("checkpoint_epoch") != epochs[kind]
                    or value.get("checkpoint_sha256") != checkpoint_hashes[kind]
                    or value.get("split") != split):
                raise ValueError("Saved evaluation checkpoint/split metadata differs")
            if prefix == "world" and trained and value.get("precision") != "fp32":
                raise ValueError("World evaluation must be saved in FP32")
    complete_evaluation_receipt(run, suite, required, manifest=job["manifest"])
    return _finish_cleanup(suite, state, job)


def _capacity_profiler(runtime):
    script = Path(runtime) / "scripts" / "profile_capacity_phase.py"
    spec = importlib.util.spec_from_file_location("_t0_cv_frozen_capacity_profile", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.profile


def _hardware(device):
    if not str(device).startswith("cuda"):
        return {"gpu_name": "cpu", "total_mib": 0, "bf16_supported": False}
    properties = torch.cuda.get_device_properties(device)
    return {"gpu_name": properties.name, "total_mib": properties.total_memory / 2**20,
            "bf16_supported": bool(torch.cuda.is_bf16_supported())}


def _profile_world(cfg, manifest):
    if not _world_head_trained(cfg):
        return {"world_peak_reserved_mib": 0., "world_peak_allocated_mib": 0.,
                "world_profile": {"status": "not_applicable", "precision": "fp32"}}
    _release(cfg.device)
    seed_all(cfg.seed, threads=cfg.threads, deterministic=cfg.deterministic)
    store = _store(manifest, cfg)
    complete = [i for i in store.by_split["train"]
                if all(visit is not None for visit in store.patients[i]["visits"])]
    if len(complete) < cfg.validation_batch_size:
        raise ValueError("World profiling needs a full batch of complete training-patient sequences")
    # A local store view only: no manifest/split files are rewritten. Complete
    # sequences ensure every TF target and free-rollout horizon is exercised.
    store.by_split["val"] = complete[:cfg.validation_batch_size]
    torch.cuda.reset_peak_memory_stats()
    model = RawVLAJEPA(cfg.model).to(cfg.device)
    if cfg.encoder_checkpoint:
        model.initialize_pretrained_encoder(cfg.encoder_checkpoint)
    _configure(model, cfg, "joint")
    model.eval()
    value = evaluate_world(model, store, replace(cfg, precision="fp32"), split="val",
                           batch_size=cfg.validation_batch_size)
    torch.cuda.synchronize()
    result = {"world_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "world_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "world_profile": {"status": "passed", "precision": "fp32",
            "scope": "actual evaluate_world on one full complete-training-patient batch",
            "batch_size": cfg.validation_batch_size, "patients": value["patients"],
            "legal_prefixes": value["legal_prefixes"]}}
    del model, store, value
    _release(cfg.device)
    return result


def checkpoint_estimate(total_parameters, trainable_parameters):
    # FP32 model (including EMA teacher), two FP32 Adam moments for every
    # trainable parameter, then history/RNG/serialization headroom. Some unused
    # parameters have no moments; allocating for all is deliberately conservative.
    raw = 4 * total_parameters + 8 * trainable_parameters
    return math.ceil(raw * 1.10 + 64 * 2**20)


def profile_job(state, job, cfg, output):
    install_paired_readout_initialization()
    torch.set_num_threads(cfg.threads)
    hardware = _hardware(cfg.device)
    durable_json(output, {"status": "profiling", "config_digest": job["config_digest"]})
    if str(cfg.device).startswith("cuda") and cfg.precision == "bf16" and not hardware["bf16_supported"]:
        raise RuntimeError("Configured BF16 training is unsupported by this GPU")
    profile = _capacity_profiler(state["runtime"])
    stages_output = Path(output).with_name(Path(output).name + ".stages.json")
    result = profile(job["config"], job["manifest"], stages_output, benchmark_seconds=0)
    _release(cfg.device)
    world = _profile_world(cfg, job["manifest"])
    result.update(world)
    result["peak_reserved_mib"] = max(result.get("peak_reserved_mib") or 0,
        world["world_peak_reserved_mib"], *(result.get(key) or 0 for key in
        ("training_peak_reserved_mib", "inference_peak_reserved_mib", "diagnostic_peak_reserved_mib")))
    result["peak_allocated_mib"] = max(result.get("peak_allocated_mib") or 0, world["world_peak_allocated_mib"])
    result.update(schema="t0_cv_capacity_profile_v1", hardware=hardware,
        host_peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        checkpoint_bytes=checkpoint_estimate(result["total_parameters"], result["trainable_parameters"]),
        checkpoint_bytes_formula="ceil(1.10 * (4 * total_parameters + 8 * trainable_parameters) + 64 MiB)",
        config_digest=job["config_digest"], manifest_sha256=job["manifest_sha256"],
        paired_readout_initialization=PAIRED_INITIALIZATION,
        source_identity=state["source_identity"])
    result["host_reservation_mib"] = max(result["host_peak_mib"], 2 * result["checkpoint_bytes"] / 2**20 + 1024)
    result["host_reservation_formula"] = "max(measured process RSS MiB, 2 * checkpoint_bytes / 2**20 + 1024)"
    if result.get("status") != "passed" or result.get("batch_size") != cfg.batch_size:
        raise ValueError("Capacity profile did not pass at the configured batch size")
    durable_json(output, result)
    return result


def run_job(suite, name, phase, *, profile_output=None):
    install_paired_readout_initialization()
    state, job, cfg = load_context(suite, name)
    with job_lock(job["output"], phase):
        if phase == "train":
            return train_job(state, job, cfg)
        if phase == "evaluate":
            return evaluate_job(Path(suite).resolve(), state, job, cfg)
        if phase == "profile":
            target = Path(profile_output) if profile_output else Path(job["output"]) / "capacity_profile.json"
            target = _inside(target, suite)
            return profile_job(state, job, cfg, target)
        raise ValueError("phase must be train, evaluate or profile")


def _attempt_context(suite):
    identifier = os.environ.get("T0_CV_ATTEMPT_ID")
    target = os.environ.get("T0_CV_ATTEMPT_STATUS")
    if identifier is None and target is None:
        return None
    if (identifier is None or target is None
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", identifier) is None
            or Path(identifier).name != identifier):
        raise ValueError("Attempt identity must be a safe basename with a status path")
    suite = Path(suite).resolve()
    path = _inside(target, suite)
    if path != suite / "attempts" / f"{identifier}.json":
        raise ValueError("Attempt status must be suite/attempts/<attempt_id>.json")
    return identifier, path


def _attempt_exit(attempt, args, error=None):
    if attempt is None:
        return
    identifier, target = attempt
    durable_json(target, {"schema": "t0_cv_attempt_exit_v1", "attempt_id": identifier,
        "pid": os.getpid(), "phase": args.phase, "job": args.job,
        "exit_code": 1 if error is not None else 0,
        "error_type": type(error).__name__ if error is not None else None,
        "error": str(error) if error is not None else None,
        "completed_at": datetime.now(timezone.utc).isoformat()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--phase", choices=("train", "evaluate", "profile"), required=True)
    parser.add_argument("--profile-output", type=Path)
    args = parser.parse_args(argv)
    attempt = _attempt_context(args.suite)
    try:
        result = run_job(args.suite, args.job, args.phase, profile_output=args.profile_output)
        print(result, flush=True)
    except BaseException as error:
        try:
            _attempt_exit(attempt, args, error)
        except BaseException as receipt_error:
            # Preserve the actual train/evaluation failure even when quota or
            # filesystem failure also prevents the controller's exit receipt.
            try:
                error.add_note(f"Attempt receipt write failed: {type(receipt_error).__name__}: {receipt_error}")
            except BaseException:
                pass
        raise
    _attempt_exit(attempt, args)


if __name__ == "__main__":
    main()

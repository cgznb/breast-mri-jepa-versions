#!/usr/bin/env python3
"""Measure real-data training, inference and diagnostic memory in one GPU worker."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time

import torch

from mri_vla_jepa.data import RawMRIStore
from mri_vla_jepa.diagnostics import evaluate_world, run_training_probe
from mri_vla_jepa.io import autocast, seed_all, write_json
from mri_vla_jepa.model import RawVLAJEPA
from mri_vla_jepa.train_config import load_config
from mri_vla_jepa.training import _configure, _make_optimizer, _source_snapshot, evaluate, load_trained


def loss_arguments(cfg):
    return {**{name: getattr(cfg, name) for name in (
        "task_weight", "jepa_weight", "rollout_weight", "flow_weight",
        "reconstruction_weight", "variance_weight", "variance_definition")},
        "prediction_training": cfg.prediction_training}


def profile_stages(cfg):
    """Profile only the observation prefixes that this run actually trains."""
    return (0,) if cfg.landmarks == "t0" else (0, 1, 2, 3)


def needs_future_supervision(cfg):
    return bool(cfg.jepa_weight or cfg.rollout_weight or cfg.flow_weight or
                (cfg.prediction_training == "teacher_forcing" and cfg.task_weight and
                 cfg.model.pcr_readout in {"forecast", "forecast_detach"}))


def profile_patients(store, cfg):
    # Forecast workloads include real future targets; pure direct T0 needs no
    # follow-up scans and can profile patients with only a baseline visit.
    required = range(4) if needs_future_supervision(cfg) or cfg.landmarks != "t0" else (0,)
    return [i for i in store.by_split["train"]
            if all(store.patients[i]["visits"][stage] is not None for stage in required)]


def profile(config, manifest, output, *, benchmark_seconds=0, ready=None, start=None):
    cfg = load_config(config)
    seed_all(cfg.seed, threads=cfg.threads, deterministic=cfg.deterministic)
    store = RawMRIStore(manifest, cfg.image_shape, cfg.allow_synthetic, image_cache=cfg.image_cache)
    stages = profile_stages(cfg)
    complete = profile_patients(store, cfg)
    needed = max(cfg.batch_size, cfg.validation_batch_size,
                 cfg.diagnostics_patients if cfg.diagnostics_every_epochs else 0)
    if len(complete) < needed:
        raise ValueError("Preflight needs enough real patients with all scans required by this configuration")
    model = RawVLAJEPA(cfg.model).to(cfg.device)
    model.initialize_pretrained_encoder(cfg.encoder_checkpoint)
    _configure(model, cfg, "joint")
    optimizer = _make_optimizer(model, cfg)
    torch.cuda.reset_peak_memory_stats()
    inp, sup = store.batch([(i, 0) for i in complete[:cfg.batch_size]],
                           future_supervision=needs_future_supervision(cfg))
    inp, sup = inp.to(cfg.device), sup.to(cfg.device)

    def update(*, sample_update=False):
        optimizer.zero_grad(set_to_none=True)
        with autocast(cfg.device, cfg.precision):
            loss, _ = model.compute_loss(inp, sup, **loss_arguments(cfg))
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite preflight loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite preflight gradients")
        parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        before = [p.detach().clone() for p in parameters] if sample_update else []
        optimizer.step()
        if sample_update:
            torch.stack([(p.detach().float() - old.float()).square().sum()
                         for p, old in zip(parameters, before, strict=True)]).sum().sqrt()
        del before, parameters
        if model.teacher_encoder is not None:
            model.update_teacher()
        return float(loss.detach())

    update()  # Include allocated AdamW moments and real backward buffers.
    torch.cuda.synchronize()
    if ready:
        write_json(ready, {"ready": True})
    if start:
        deadline = time.monotonic() + 300
        while not Path(start).exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Benchmark start barrier was not released")
            time.sleep(.1)
    began = time.monotonic()
    iterations, last_loss = 0, None
    while True:
        if benchmark_seconds:
            # Include real NPZ assembly/transfers and all landmark workloads.
            inp, sup = store.batch([(i, stages[iterations % len(stages)]) for i in complete[:cfg.batch_size]],
                                   future_supervision=needs_future_supervision(cfg))
            inp, sup = inp.to(cfg.device), sup.to(cfg.device)
        last_loss = update()
        iterations += 1
        torch.cuda.synchronize()
        if time.monotonic() - began >= max(benchmark_seconds, .001):
            break
    elapsed = time.monotonic() - began
    train_peak = torch.cuda.max_memory_reserved() / 2**20
    inference_peak, diagnostic_peak = None, None
    if not benchmark_seconds:
        # Intermediate landmarks combine several online visits and prediction.
        for stage in stages:
            inp, sup = store.batch([(i, stage) for i in complete[:cfg.batch_size]],
                                   future_supervision=needs_future_supervision(cfg))
            inp, sup = inp.to(cfg.device), sup.to(cfg.device)
            update(sample_update=bool(cfg.diagnostics_every_epochs))
        torch.cuda.synchronize()
        train_peak = torch.cuda.max_memory_reserved() / 2**20
        # Match validation: public inference with no supervision and FP32.
        model.eval()
        with torch.no_grad():
            for stage in stages:
                vi, _ = store.batch([(i, stage) for i in complete[:cfg.validation_batch_size]], supervised=False)
                model(vi.to(cfg.device))
        torch.cuda.synchronize()
        inference_peak = torch.cuda.max_memory_reserved() / 2**20
        if cfg.diagnostics_every_epochs:
            for stage in stages:
                tasks = tuple((i, stage) for i in complete[:cfg.diagnostics_batch_size])
                run_training_probe(model, store, cfg, cfg.diagnostics_every_epochs, tasks)
            torch.cuda.synchronize()
            diagnostic_peak = torch.cuda.max_memory_reserved() / 2**20
    result = {"status": "passed", "config": str(Path(config).resolve()),
              "gpu_name": torch.cuda.get_device_name(), "batch_size": cfg.batch_size,
              "landmarks": cfg.landmarks, "profile_stages": list(stages),
              "future_supervision": needs_future_supervision(cfg),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "total_parameters": sum(p.numel() for p in model.parameters()),
              "training_peak_reserved_mib": train_peak,
              "inference_peak_reserved_mib": inference_peak,
              "diagnostic_peak_reserved_mib": diagnostic_peak,
              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "iterations": iterations, "elapsed_seconds": elapsed,
              "patients_per_second": iterations * cfg.batch_size / elapsed,
              "last_loss": last_loss, "benchmark_seconds": benchmark_seconds,
              "benchmark_scope": "real NPZ assembly/transfers, complete sequences, cyclic T0-T3"}
    write_json(output, result)
    print(result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-seconds", type=float, default=0)
    parser.add_argument("--ready", type=Path)
    parser.add_argument("--start", type=Path)
    parser.add_argument("--evaluate-checkpoint", type=Path)
    parser.add_argument("--world-checkpoint", type=Path)
    args = parser.parse_args()
    if args.world_checkpoint:
        model, cfg, state = load_trained(args.world_checkpoint, "cuda:0")
        if state.get("source_snapshot") != _source_snapshot():
            raise ValueError("World evaluation source differs from checkpoint")
        torch.set_num_threads(cfg.threads)
        store = RawMRIStore(args.manifest, cfg.image_shape, cfg.allow_synthetic, state["normalization"])
        result = evaluate_world(model, store, replace(cfg, device="cuda:0", precision="fp32"),
                                split="val", batch_size=cfg.validation_batch_size)
        result.update(checkpoint=str(args.world_checkpoint), split="val", independent_test=False)
        write_json(args.output, result)
    elif args.evaluate_checkpoint:
        evaluate(args.evaluate_checkpoint, args.manifest, args.output,
                 split="val", device="cuda:0", bootstrap=0)
    else:
        profile(args.config, args.manifest, args.output,
                benchmark_seconds=args.benchmark_seconds, ready=args.ready, start=args.start)


if __name__ == "__main__":
    main()

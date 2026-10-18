"""Joint training with patient weighting and exact optimizer-update checkpoints."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import time

import torch

from .data import Store, planned_tasks
from .engine import (SCHEMA, autocast, build_model, evaluate, init_from, load_checkpoint,
                     make_loader, optimizer_for, restore_rng, rng_state, runtime_info,
                     save_checkpoint, setup, write_csv, write_json)
from .objectives import ADDITIVE_TERMS, joint_loss
from .provenance import source_snapshot


def schedule_updates(optimizer, cfg, update):
    """The learning rate for optimizer update 1..max_updates."""
    t = cfg.train
    if t.warmup_updates and update <= t.warmup_updates:
        factor = update / t.warmup_updates
    else:
        progress = min(1., max(0., (update-t.warmup_updates)/(t.max_updates-t.warmup_updates)))
        factor = t.min_lr_ratio + (1-t.min_lr_ratio)*.5*(1+math.cos(math.pi*progress))
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"]*factor


def _probe_tasks(store, cfg):
    count = cfg.train.train_eval_patients
    if not count:
        return []
    # Fixed stratified subset, determined without looking at validation results.
    import random
    groups = defaultdict(list)
    for i in store.splits["train"]:
        groups[store.patients[i].get("target", {}).get("pcr")].append(i)
    rng = random.Random(20261009)
    for ids in groups.values():
        rng.shuffle(ids)
    count = min(count, len(store.splits["train"]))
    population = len(store.splits["train"])
    quotas = {group: count*len(ids)/population for group, ids in groups.items()}
    sizes = {group: int(quota) for group, quota in quotas.items()}
    for group in sorted(groups, key=lambda group: (quotas[group]-sizes[group], str(group)), reverse=True):
        if sum(sizes.values()) >= count:
            break
        sizes[group] += 1
    chosen = [i for group, ids in groups.items() for i in ids[:sizes[group]]]
    return [(i, landmark) for i in chosen for landmark in store.allowed(i, cfg.train.profile)]


def train_updates(cfg, manifest, output, *, init=None, resume=False, stop_after_updates=None):
    cfg.validate()
    setup(cfg)
    t = cfg.train
    outdir = Path(output).resolve()
    outdir.mkdir(parents=True, exist_ok=True, mode=0o700)
    last = outdir / "last.pt"
    if (outdir / "config.json").exists() and not resume:
        raise FileExistsError("Output already contains a run; use --resume or a new directory")
    if resume and (not last.exists() or init is not None):
        raise ValueError("Resume requires last.pt and cannot be combined with --init")
    store = Store(manifest, t.image_shape, allow_synthetic=t.allow_synthetic, image_cache=t.image_cache)
    if not store.splits["train"] or not store.splits["val"]:
        raise ValueError("Training requires train AND val; test is never used for selection")
    store.audit(verify_pixels=False)
    fingerprint, implementation = store.fingerprint(), source_snapshot()
    model = build_model(cfg)
    state = load_checkpoint(last) if resume else None
    if state is not None:
        if (state["config"] != cfg.to_dict() or state["data_fingerprint"] != fingerprint
                or state["source_identity"] != implementation or state["environment"] != runtime_info()):
            raise ValueError("Resume config/data/source/runtime identity differs")
        if state.get("runner") != "optimizer_updates_v1":
            raise ValueError("Cannot resume an epoch-boundary run as an update-boundary run")
        model.load_state_dict(state["model"], strict=True)
        init_info = state["init"]
    else:
        init_info = init_from(model, cfg, init, store)
    model.world_trained = bool(cfg.loss.tf or cfg.loss.rollout)
    model.target_coordinate_origin = (state["target_coordinate_origin"] if state is not None else
        {"mode": "ema", "run": str(outdir), "phase": "joint", "seed": t.seed, "epoch": 0, "updates": 0})
    model.to(t.device)
    model.configure_trainable("joint", False)
    if cfg.model.readout == "direct":
        for module in (model.terminal_pool, model.terminal_head, model.gate):
            module.requires_grad_(False)
    optimizer = optimizer_for(model, cfg)
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
    updates = state["updates"] if state else 0
    epoch = state["cursor"]["epoch"] if state else 0
    cursor = state["cursor"]["batch"] if state else 0
    best = state["best_score"] if state else float("inf")
    best_update = state["best_update"] if state else 0
    history = state["history"] if state else []
    sums = defaultdict(float, state.get("interval_sums", {}) if state else {})
    interval_n = state.get("interval_samples", 0) if state else 0
    interval_updates = state.get("interval_updates", 0) if state else 0
    samples_seen = state.get("samples_seen", 0) if state else 0
    probe_tasks = _probe_tasks(store, cfg)
    write_json(outdir / "config.json", cfg.to_dict())
    write_json(outdir / "data_audit.json", store.audit(False))
    write_json(outdir / "provenance.json", {
        "source_identity": implementation, "environment": runtime_info(),
        "data_fingerprint": fingerprint, "normalization": store.normalization,
        "init": init_info, "parameters_total": sum(p.numel() for p in model.parameters()),
        "parameters_requires_grad": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "checkpoint_resume_boundary": "completed optimizer update; deterministic epoch and batch cursor",
        "teacher_updates": "EMA once after each AdamW update", "runner": "optimizer_updates_v1",
        "train_probe_patients": len({i for i, _ in probe_tasks}),
        "additive_reduction": "sum of fixed sample weights / reference batch size"})
    clock_start = time.monotonic()

    def checkpoint():
        return {"schema": SCHEMA, "runner": "optimizer_updates_v1", "config": cfg.to_dict(),
            "epoch": epoch, "cursor": {"epoch": epoch, "batch": cursor}, "updates": updates,
            "samples_seen": samples_seen, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "rng": rng_state(), "best_score": best, "best_update": best_update,
            "bad_epochs": 0, "bad_updates": updates-best_update, "history": history,
            "interval_sums": dict(sums), "interval_samples": interval_n, "interval_updates": interval_updates,
            "normalization": store.normalization, "data_fingerprint": fingerprint,
            "source_identity": implementation, "environment": runtime_info(), "world_trained": model.world_trained,
            "init": init_info, "target_coordinate_origin": model.target_coordinate_origin}

    def assess():
        nonlocal best, best_update, interval_n, interval_updates, clock_start
        metrics, _, _ = evaluate(model, store, cfg, "val")
        train_probe = evaluate(model, store, cfg, "train", tasks=probe_tasks, world=False)[0] if probe_tasks else None
        score = metrics["selection"]
        if not math.isfinite(score):
            raise FloatingPointError("Non-finite validation selection; checkpoint not committed")
        improved = score < best-t.min_delta
        if improved:
            best, best_update = score, updates
        train_terms = {}
        for key, value in sums.items():
            denominator = interval_updates if key in {"objective", "gradient_norm", "consistency_weight"} else interval_n
            train_terms[key] = value/max(1, denominator)
        history.append({"epoch": epoch, "batch_cursor": cursor, "updates": updates,
            "samples_seen": samples_seen, "training_patients_or_prefixes": interval_n,
            "train": train_terms, "train_probe": train_probe, "validation": metrics,
            "best_score": best, "best_update": best_update, "bad_updates": updates-best_update,
            "seconds": time.monotonic()-clock_start})
        sums.clear()
        interval_n = interval_updates = 0
        data = checkpoint()
        save_checkpoint(last, data)
        if improved:
            save_checkpoint(outdir / "best.pt", data)
        write_json(outdir / "history.json", history)
        write_json(outdir / "progress.json", {"state": "running", "updates": updates, "max_updates": t.max_updates,
            "epoch": epoch, "batch_cursor": cursor, "samples_seen": samples_seen, "best_update": best_update,
            "best_score": best, "bad_updates": updates-best_update})
        print(f"update={updates} epoch={epoch} batch={cursor} val={score:.5f} "
              f"best={best:.5f}@{best_update} bad_updates={updates-best_update}", flush=True)
        clock_start = time.monotonic()

    print(f"Update joint / {t.profile}; train={len(store.splits['train'])} val={len(store.splits['val'])}; "
          f"max_updates={t.max_updates}; output={outdir}", flush=True)
    if not history:
        assess()
    paused = False
    while updates < t.max_updates and updates-best_update < t.patience_updates:
        tasks = planned_tasks(store, cfg, "train", epoch)
        loader = make_loader(store, cfg, tasks, training=True, world=bool(cfg.loss.tf or cfg.loss.rollout or cfg.loss.align))
        if cursor > len(loader):
            raise ValueError("Checkpoint batch cursor exceeds deterministic loader")
        iterator = iter(loader)
        for _ in range(cursor):
            next(iterator)
        for cpu_batch in iterator:
            model.train()
            batch = cpu_batch.to(t.device, non_blocking=t.pin_memory)
            n = len(batch.keys)
            optimizer.zero_grad(set_to_none=True)
            schedule_updates(optimizer, cfg, updates+1)
            with autocast(cfg):
                loss, terms, prediction = joint_loss(model, batch, cfg, augment=t.augment, updates=updates+1)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite joint objective; no optimizer update committed")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.clip_grad, error_if_nonfinite=True)
            optimizer.step()
            model.update_teacher(t.ema)
            updates += 1
            cursor += 1
            samples_seen += n
            model.target_coordinate_origin = {"mode": "ema", "run": str(outdir), "phase": "joint", "seed": t.seed,
                                              "epoch": epoch, "updates": updates}
            sums["objective"] += float(loss.detach())
            sums["gradient_norm"] += float(norm)
            for key, value in terms.items():
                scalar = float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
                multiplier = (t.reference_batch_size or n) if key in ADDITIVE_TERMS else n
                sums[key] += scalar*(1 if key == "consistency_weight" else multiplier)
            interval_n += n
            interval_updates += 1
            del prediction, batch, loss, terms
            if (updates % t.eval_every_updates == 0 or updates == t.max_updates
                    or updates-best_update >= t.patience_updates):
                assess()
            if stop_after_updates is not None and updates >= stop_after_updates:
                save_checkpoint(last, checkpoint())
                paused = True
                break
            if updates >= t.max_updates or updates-best_update >= t.patience_updates:
                break
        if paused or updates >= t.max_updates or updates-best_update >= t.patience_updates:
            break
        epoch += 1
        cursor = 0
    if paused:
        return {"output": str(outdir), "state": "paused_for_test", "updates": updates}
    if history[-1]["updates"] != updates:
        assess()
    best_state = load_checkpoint(outdir / "best.pt")
    model.load_state_dict(best_state["model"], strict=True)
    model.target_coordinate_origin = best_state["target_coordinate_origin"]
    metrics, rows, wrows = evaluate(model, store, cfg, "val")
    write_json(outdir / "metrics_best.json", metrics)
    if t.save_predictions:
        write_csv(outdir / "predictions_val.csv", rows)
        write_csv(outdir / "world_val.csv", wrows)
    reason = "max_updates" if updates >= t.max_updates else "early_stop"
    summary = {"state": "completed", "stop_reason": reason, "updates": updates, "max_updates": t.max_updates,
               "best_update": best_update, "best_score": best, "samples_seen": samples_seen,
               "bad_updates": updates-best_update, "output": str(outdir)}
    write_json(outdir / "completion.json", summary)
    write_json(outdir / "progress.json", summary)
    return {**summary, "metrics": metrics}

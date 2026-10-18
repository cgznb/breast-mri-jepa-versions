"""Staged or joint training, epoch-boundary resume, and leakage-safe evaluation."""
from __future__ import annotations
from contextlib import nullcontext
from dataclasses import replace
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .config import Config, from_dict
from .data import Store, VisitDataset, PrefixDataset, collate_prefix, planned_tasks
from .model import ResponseJEPA
from .objectives import joint_loss, representation_loss, task_world_loss
from .metrics import task_summary, world_summary
from .provenance import file_identity, record_metadata, source_snapshot

SCHEMA = "response_jepa_checkpoint_local_v1"


def jsonable(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().item() if x.numel() == 1 else x.detach().cpu().tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [jsonable(v) for v in x]
    return x


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temp, path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o600)  # per-patient predictions are private research records


def runtime_info():
    return {"python": sys.version.split()[0], "torch": str(torch.__version__),
            "numpy": np.__version__, "torch_cuda": torch.version.cuda}


def tensor_hash(state):
    # In-memory fixed-teacher check only. Digest values are never saved or printed.
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rng_state():
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def setup(cfg):
    t = cfg.train
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(t.threads)
    random.seed(t.seed)
    np.random.seed(t.seed)
    torch.manual_seed(t.seed)
    if t.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use train.device=cpu for the synthetic smoke test")
        if t.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU does not support BF16; set train.precision=fp32")
        torch.cuda.manual_seed_all(t.seed)
    elif t.device != "cpu":
        raise ValueError("This validated v2 runner supports CPU/CUDA, not an untested MPS path")
    torch.use_deterministic_algorithms(t.deterministic, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = t.deterministic


def autocast(cfg):
    return (torch.autocast("cuda", dtype=torch.bfloat16)
            if cfg.train.device.startswith("cuda") and cfg.train.precision == "bf16" else nullcontext())


def load_checkpoint(path):
    # Full optimizer/RNG snapshots use pickle. Only load your OWN trusted checkpoints.
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("schema") != SCHEMA:
        raise ValueError("Not a v2 checkpoint. Upstream best.pt cannot be resumed or silently converted.")
    return data


def build_model(cfg):
    return ResponseJEPA(cfg.model)


def init_from(model, cfg, path, store):
    if cfg.train.phase == "joint":
        if path is not None:
            raise ValueError("Joint training starts from scratch; use --resume for its own last.pt")
        return None
    if path is None:
        if cfg.train.phase == "B":
            raise ValueError("B requires --init pointing to A/best.pt")
        if cfg.train.phase == "C" and cfg.model.use_mri and cfg.train.freeze_encoder:
            raise ValueError("A frozen random encoder is forbidden: provide --init A/B or unfreeze a direct baseline")
        if cfg.train.phase == "C" and (cfg.loss.tf or cfg.loss.rollout or cfg.model.readout != "direct"):
            raise ValueError("World-based C requires a trained B checkpoint")
        return None
    state = load_checkpoint(path)
    previous = from_dict(state["config"])
    if state["data_fingerprint"] != store.fingerprint():
        raise ValueError("Warm-start cohort/split/asset identity differs. Re-train A/B inside the current fold.")
    if state["normalization"] != store.normalization:
        raise ValueError("Warm-start normalization differs")
    shape_keys = ("encoder", "dim", "heads", "token_grid", "widths", "depths", "spatial_depth")
    if any(getattr(previous.model, k) != getattr(cfg.model, k) for k in shape_keys):
        raise ValueError("Encoder architecture differs from the initialization checkpoint")
    if previous.train.phase == "A" and cfg.train.phase in {"B", "C"}:
        if cfg.train.phase == "C" and (cfg.loss.tf or cfg.loss.rollout or cfg.model.readout != "direct"):
            raise ValueError("A->C is a DIRECT baseline. A->B->C is required for world-based training.")
        teacher = {k.removeprefix("target_encoder."): v for k, v in state["model"].items()
                   if k.startswith("target_encoder.")}
        model.encoder.load_state_dict(teacher, strict=True)
        model.target_encoder.load_state_dict(teacher, strict=True)
    elif previous.train.phase == "B" and cfg.train.phase == "C":
        if (previous.model.residual_world != cfg.model.residual_world
                or previous.model.world_arch != cfg.model.world_arch):
            raise ValueError("Cannot reinterpret an absolute B predictor as residual, or vice versa")
        model.load_state_dict(state["model"], strict=True)
    else:
        raise ValueError("Supported warm starts are A->B, A->C direct, B->C; use --resume within a phase")
    return {"path": str(Path(path).resolve()), "phase": previous.train.phase,
            "checkpoint_identity": file_identity(path),
            "target_coordinate_origin": state["target_coordinate_origin"]}


def stage_batches(tasks, batch_size, *, patient_ids=None):
    """Group by actual stage, optionally keeping repeated patients apart.

    A patient can have multiple legal prefixes with the same MRI anchor when
    visits become available late. These must not share a patient-axis variance
    batch. Preserve the original packing when every patient is already unique.
    """
    if patient_ids is not None and len(patient_ids) != len(tasks):
        raise ValueError("Patient identities must match the task list")
    groups = defaultdict(list)
    for i, task in enumerate(tasks):
        groups[task[1]].append(i)
    batches = []
    for group in groups.values():
        repeated = patient_ids is not None and len({patient_ids[i] for i in group}) < len(group)
        if repeated:
            chunks, members = [], []
            for i in group:
                patient = patient_ids[i]
                for chunk, patients in zip(chunks, members):
                    if len(chunk) < batch_size and patient not in patients:
                        chunk.append(i)
                        patients.add(patient)
                        break
                else:
                    chunks.append([i])
                    members.append({patient})
        else:
            chunks = [group[i:i+batch_size] for i in range(0, len(group), batch_size)]
        # Keep every sample. Merge a singleton only into a distinct-patient batch.
        if len(chunks) > 1 and len(chunks[-1]) == 1:
            for previous in reversed(chunks[:-1]):
                if patient_ids is None or patient_ids[chunks[-1][0]] not in {patient_ids[i] for i in previous}:
                    previous.extend(chunks.pop())
                    break
        batches.extend(chunks)
    return batches


def make_loader(store, cfg, tasks, *, phase=None, training=False, world=False, labels=True):
    phase = phase or cfg.train.phase
    t = cfg.train
    batch_size = t.batch_size if training else t.eval_batch_size
    common = dict(num_workers=t.workers, pin_memory=t.pin_memory and t.device.startswith("cuda"))
    # Use an independent generator so creating a DataLoader does not advance model dropout RNG.
    common["generator"] = torch.Generator().manual_seed(t.seed + (17 if training else 29))
    if phase == "A":
        ds = VisitDataset(store, tasks)
        return DataLoader(ds, batch_sampler=stage_batches(tasks, batch_size), **common)
    ds = PrefixDataset(store, tasks, future=world, labels=labels,
                       single=t.profile.startswith("single_"), load_images=cfg.model.use_mri,
                       sample_weighting=t.sample_weighting if training else "sample",
                       profile=t.profile, phase=phase)
    if phase == "joint" and training:
        anchor_tasks = [(pos, store.observed_stage(i, landmark)) for pos, (i, landmark) in enumerate(tasks)]
        batches = stage_batches(anchor_tasks, batch_size, patient_ids=[i for i, _ in tasks])
        # Interleave stages without consuming model/dropout RNG.
        random.Random(t.seed).shuffle(batches)
        return DataLoader(ds, batch_sampler=batches, collate_fn=collate_prefix, **common)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_prefix, **common)


def move_repr(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate_representation(model, store, cfg, split="val"):
    tasks = planned_tasks(store, cfg, split, 0)
    model.eval()
    total, count, sums = 0.0, 0, defaultdict(float)
    # Fixed validation masks shared across epochs; do not consume the training RNG stream.
    devices = [torch.device(cfg.train.device).index or 0] if cfg.train.device.startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(20261008)
        for batch in make_loader(store, cfg, tasks, phase="A"):
            batch = move_repr(batch, cfg.train.device)
            # Evaluation always FP32 (including on CUDA).
            loss, terms = representation_loss(model, batch, cfg, augment=False)
            n = len(batch["image"])
            total += float(loss)*n
            count += n
            for k, v in terms.items():
                sums[k] += float(v)*n
    return {"selection": total/count, "selection_name": "fixed_mask_same_visit_objective",
            "patients": count, "terms": {k: v/count for k, v in sums.items()}}


@torch.no_grad()
def evaluate(model, store, cfg, split="val", *, tasks=None, world=None):
    model.eval()
    if cfg.train.phase == "A":
        return evaluate_representation(model, store, cfg, split), [], []
    if tasks is None:
        tasks = planned_tasks(store, cfg, split, 0)
    if world is None:
        world = cfg.train.phase == "B" or cfg.train.world_eval
    world = bool(world and cfg.model.use_mri and getattr(model, "world_trained", True))
    rows, wrows, tfrows = [], [], []
    for batch in make_loader(store, cfg, tasks, world=world):
        batch = batch.to(cfg.train.device)
        inp, sup = batch.inputs, batch.targets
        out = model(inp, need_world=world)
        for i, key in enumerate(batch.keys):
            rows.append({"patient_key": key, "landmark": int(inp.landmark[i]),
                         "label": float(sup.label[i]) if sup.label_mask[i] else None,
                         "direct": float(out["direct_logit"][i].float().sigmoid()),
                         "forecast": float(out["forecast_logit"][i].float().sigmoid())
                             if out["forecast_logit"] is not None else None,
                         "fused": float(out["logit"][i].float().sigmoid()),
                         "gate": float(out["gate"][i]) if out["gate"] is not None else None})
        if world:
            tf, states, known, pair_mask = model.teacher_forcing(inp, sup, out["transitions"])
            anchors = inp.anchor()
            for i, key in enumerate(batch.keys):
                anchor, lm = int(anchors[i]), int(inp.landmark[i])
                for target in range(lm+1, 4):
                    if not sup.future_mask[i, target]:
                        continue
                    truth = states[i, target].float()
                    wrows.append({"patient_key": key, "landmark": lm, "target_stage": target,
                                  "horizon": target-anchor,
                                  "complete_chain": bool(known[i, anchor:target+1].all()),
                                  "free_l1": float((out["states"][i, target].float()-truth).abs().mean()),
                                  "copy_l1": float((states[i, anchor].float()-truth).abs().mean())})
                for j in range(3):
                    if pair_mask[i, j]:
                        tfrows.append({"patient_key": key,
                                       "free_l1": float((tf[i, j].float()-states[i, j+1].float()).abs().mean()),
                                       "copy_l1": float((states[i, j].float()-states[i, j+1].float()).abs().mean())})
    target_space = ("EMA teacher at this evaluated joint checkpoint; moves across epochs; not MRI pixel error"
                    if cfg.train.phase == "joint" else "fixed Stage-A EMA encoder in B/C; not MRI pixel error")
    result = {"classification": task_summary(rows) if cfg.train.phase in {"C", "joint"} else None,
              "world": world_summary(wrows, tfrows, target_space=target_space) if world else None,
              "target_coordinate_origin": getattr(model, "target_coordinate_origin", None) if world else None,
              "world_output_mode": "learned",
              "task_forecast_source": "anchor_copy" if cfg.model.readout == "copy_gated" else "predicted_terminal"}
    if cfg.train.phase == "B":
        value = result["world"]["all"]["free_l1"]
        result.update(selection=value, selection_name="fixed_coordinate_free_rollout_l1")
    else:
        fused = result["classification"]["fused"]
        result.update(selection=fused["patient_equal"]["nll"] if fused else None,
                      selection_name="patient_equal_pcr_nll")
    if result["selection"] is None or not math.isfinite(result["selection"]):
        raise ValueError("No valid selection score: inspect labels or future-target coverage")
    return result, rows, wrows


def optimizer_for(model, cfg):
    grouped = defaultdict(list)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        head = name.startswith(("direct_head", "terminal_head", "terminal_pool", "gate", "pcr_queries"))
        lr = cfg.train.lr if cfg.train.phase not in {"C", "joint"} or head else cfg.train.backbone_lr
        no_decay = param.ndim < 2 or "position" in name or "queries" in name
        grouped[(lr, 0.0 if no_decay else cfg.train.weight_decay)].append(param)
    groups = [{"params": params, "lr": lr, "base_lr": lr, "weight_decay": wd}
              for (lr, wd), params in grouped.items()]
    if not groups:
        raise ValueError("No trainable parameters")
    return torch.optim.AdamW(groups)


def schedule_lr(optimizer, cfg, progress):
    t = cfg.train
    if t.warmup_epochs and progress < t.warmup_epochs:
        factor = max(1e-6, progress/t.warmup_epochs)
    else:
        pos = min(1.0, max(0.0, (progress-t.warmup_epochs)/(t.epochs-t.warmup_epochs)))
        factor = t.min_lr_ratio + (1-t.min_lr_ratio)*.5*(1+math.cos(math.pi*pos))
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"]*factor


def save_checkpoint(path, data):
    path = Path(path)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(data, temporary)
    temporary.chmod(0o600)
    os.replace(temporary, path)


def train(cfg: Config, manifest, output, *, init=None, resume=False, stop_after_epoch=None,
          stop_after_updates=None):
    cfg.validate()
    if cfg.train.max_updates is not None:
        from .update_engine import train_updates
        return train_updates(cfg, manifest, output, init=init, resume=resume,
                             stop_after_updates=stop_after_updates)
    setup(cfg)
    outdir = Path(output).resolve()
    outdir.mkdir(parents=True, exist_ok=True, mode=0o700)
    last = outdir/"last.pt"
    if (outdir/"config.json").exists() and not resume:
        raise FileExistsError("Output already contains a run. Use --resume or a new directory.")
    if resume and (not last.exists() or init is not None):
        raise ValueError("Resume requires last.pt and must not be combined with --init")
    store = Store(manifest, cfg.train.image_shape, allow_synthetic=cfg.train.allow_synthetic,
                  image_cache=cfg.train.image_cache)
    if not store.splits["train"] or not store.splits["val"]:
        raise ValueError("Training requires train AND val; test is never used for selection")
    store.audit(verify_pixels=False)
    fingerprint, implementation = store.fingerprint(), source_snapshot()
    model = build_model(cfg)
    init_info = None
    start, best, bad, updates, history = 0, float("inf"), 0, 0, []
    state = None
    if resume:
        state = load_checkpoint(last)
        if state["config"] != cfg.to_dict() or state["data_fingerprint"] != fingerprint:
            raise ValueError("Resume config/data identity differs (including max epochs, seed and batch size)")
        if state.get("environment", runtime_info()) != runtime_info():
            raise ValueError("Resume runtime versions changed; use the same Python/PyTorch/NumPy environment")
        if state["source_identity"] != implementation:
            raise ValueError("Resume source changed. Start a new run rather than mixing implementations.")
        model.load_state_dict(state["model"], strict=True)
        start, best, bad, updates = state["epoch"], state["best_score"], state["bad_epochs"], state["updates"]
        history, init_info = state["history"], state["init"]
    else:
        init_info = init_from(model, cfg, init, store)
    model.target_coordinate_origin = (state["target_coordinate_origin"] if state is not None else
                                     init_info["target_coordinate_origin"] if init_info is not None else
                                     {"mode": "ema" if cfg.train.phase in {"A", "joint"} else "initial",
                                      "run": str(outdir), "phase": cfg.train.phase, "seed": cfg.train.seed,
                                      "epoch": 0, "updates": 0})
    model.world_trained = (cfg.train.phase == "B" or bool(cfg.loss.tf or cfg.loss.rollout)
                           or bool(init_info and init_info["phase"] == "B")) if cfg.train.phase != "A" else False
    if state is not None:
        model.world_trained = state.get("world_trained", model.world_trained)
    model.to(cfg.train.device)
    model.configure_trainable(cfg.train.phase, cfg.train.freeze_encoder)
    if cfg.train.phase in {"C", "joint"} and cfg.model.readout == "direct":
        for module in (model.terminal_pool, model.terminal_head, model.gate):
            module.requires_grad_(False)
        if not (cfg.loss.tf or cfg.loss.rollout):
            model.world.requires_grad_(False)
    optimizer = optimizer_for(model, cfg)
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
    frozen_target = tensor_hash(model.target_encoder.state_dict()) if cfg.train.phase in {"B", "C"} else None
    write_json(outdir/"config.json", cfg.to_dict())
    write_json(outdir/"data_audit.json", store.audit(False))
    write_json(outdir/"provenance.json", {
        "upstream_project": "cgznb/breast-mri-vla-jepa",
        "source_identity": implementation, "environment": runtime_info(),
        "data_fingerprint": fingerprint, "normalization": store.normalization,
        "init": init_info, "parameters_total": sum(p.numel() for p in model.parameters()),
        "parameters_requires_grad": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "actual_training_precision": cfg.train.precision if cfg.train.device.startswith("cuda") else "fp32",
        "checkpoint_resume_boundary": "end of completed epoch only",
        "teacher_updates": "EMA after every optimizer update" if cfg.train.phase in {"A", "joint"} else "fixed",
        "not_claimed": ["clinical efficacy", "causal treatment effects", "future MRI synthesis", "CVPR acceptance"]})
    if cfg.train.phase == "C" and cfg.model.use_mri and cfg.train.freeze_encoder and init_info is None and not resume:
        raise ValueError("Refusing frozen random MRI encoder")
    print(f"Phase {cfg.train.phase} / {cfg.train.profile}; train={len(store.splits['train'])}, "
          f"val={len(store.splits['val'])}; output={outdir}", flush=True)
    for epoch in range(start, cfg.train.epochs):
        if bad >= cfg.train.patience:
            break
        model.train()
        epoch_start = time.monotonic()
        tasks = planned_tasks(store, cfg, "train", epoch)
        needs_targets = bool(cfg.loss.tf or cfg.loss.rollout or cfg.loss.align) and cfg.train.phase != "A"
        loader = make_loader(store, cfg, tasks, training=True, world=needs_targets)
        iterator = iter(loader)
        sums, total_n = defaultdict(float), 0
        num_groups = math.ceil(len(loader)/cfg.train.accumulation)
        group_index = 0
        while True:
            group = []
            for _ in range(cfg.train.accumulation):
                try:
                    group.append(next(iterator))
                except StopIteration:
                    break
            if not group:
                break
            counts = [len(x["image"]) if cfg.train.phase == "A" else len(x.keys) for x in group]
            denominator = sum(counts)  # exact partial-tail weighting, not always /accumulation
            optimizer.zero_grad(set_to_none=True)
            schedule_lr(optimizer, cfg, epoch+(group_index+1)/num_groups)
            for cpu_batch, n in zip(group, counts):
                with autocast(cfg):
                    if cfg.train.phase == "A":
                        batch = move_repr(cpu_batch, cfg.train.device)
                        loss, terms = representation_loss(model, batch, cfg, augment=cfg.train.augment)
                    else:
                        batch = cpu_batch.to(cfg.train.device, non_blocking=cfg.train.pin_memory)
                        if cfg.train.phase == "joint":
                            loss, terms, prediction = joint_loss(model, batch, cfg, augment=cfg.train.augment)
                        else:
                            loss, terms, prediction = task_world_loss(model, batch, cfg)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite objective. No optimizer update was committed.")
                (loss*(n/denominator)).backward()
                sums["loss"] += float(loss.detach())*n
                for key, value in terms.items():
                    sums[key] += float(value.detach() if isinstance(value, torch.Tensor) else value)*n
                total_n += n
                # Gradients now live on parameters. Release completed batch/graph
                # references before the next GPU transfer or FP32 validation.
                if cfg.train.phase != "A":
                    del prediction
                del batch, loss, terms
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad, error_if_nonfinite=True)
            optimizer.step()
            if cfg.train.phase in {"A", "joint"}:
                model.update_teacher(cfg.train.ema)
                model.target_coordinate_origin = {"mode": "ema", "run": str(outdir), "phase": cfg.train.phase, "seed": cfg.train.seed,
                                                  "epoch": epoch+1, "updates": updates+1}
            sums["gradient_norm"] += float(norm)*denominator
            updates += 1
            group_index += 1
        metrics, _, _ = evaluate(model, store, cfg, "val")
        score = metrics["selection"]
        improved = score < best-cfg.train.min_delta
        if improved:
            best, bad = score, 0
        else:
            bad += 1
        if frozen_target is not None and tensor_hash(model.target_encoder.state_dict()) != frozen_target:
            raise RuntimeError("Fixed target coordinates changed in B/C")
        entry = {"epoch": epoch+1, "updates": updates, "training_patients_or_prefixes": total_n,
                 "train": {k: v/max(1, total_n) for k, v in sums.items()},
                 "validation": metrics, "best_score": best, "bad_epochs": bad,
                 "seconds": time.monotonic()-epoch_start}
        history.append(entry)
        data = {"schema": SCHEMA, "config": cfg.to_dict(), "epoch": epoch+1, "updates": updates,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng_state(),
                "best_score": best, "bad_epochs": bad, "history": history,
                "normalization": store.normalization, "data_fingerprint": fingerprint,
                "source_identity": implementation, "environment": runtime_info(),
                "world_trained": model.world_trained, "init": init_info,
                "target_coordinate_origin": model.target_coordinate_origin}
        save_checkpoint(last, data)
        if improved:
            save_checkpoint(outdir/"best.pt", data)
        write_json(outdir/"history.json", history)
        print(f"epoch={epoch+1} loss={entry['train']['loss']:.5f} "
              f"val={score:.5f} best={best:.5f} bad_epochs={bad}", flush=True)
        if stop_after_epoch is not None and epoch+1 >= stop_after_epoch:
            break
    best_state = load_checkpoint(outdir/"best.pt")
    model.load_state_dict(best_state["model"], strict=True)
    model.target_coordinate_origin = best_state["target_coordinate_origin"]
    metrics, rows, wrows = evaluate(model, store, cfg, "val")
    write_json(outdir/"metrics_best.json", metrics)
    if cfg.train.save_predictions and cfg.train.phase in {"C", "joint"}:
        write_csv(outdir/"predictions_val.csv", rows)
        write_csv(outdir/"world_val.csv", wrows)
    return {"output": str(outdir), "best_epoch": best_state["epoch"], "metrics": metrics}


def load_for_inference(checkpoint_path, manifest, *, device="cpu", image_cache=None, allow_synthetic=False):
    state = load_checkpoint(checkpoint_path)
    cfg = from_dict(state["config"])
    cfg.train.device = device
    cfg.train.precision = "fp32"
    cfg.train.workers = 0
    setup(cfg)
    model = build_model(cfg).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.target_coordinate_origin = state["target_coordinate_origin"]
    model.encoder_frozen = True
    model.world_trained = state.get("world_trained", cfg.train.phase == "B" or bool(cfg.loss.tf or cfg.loss.rollout))
    model.eval()
    store = Store(manifest, cfg.train.image_shape, allow_synthetic=allow_synthetic,
                  normalization=state["normalization"], image_cache=image_cache)
    # No cohort-equality requirement for external inference, but normalization is fixed.
    if record_metadata(store.manifest.get("image_normalization")) != state["data_fingerprint"].get("image_normalization"):
        raise ValueError("Inference image normalization differs from training. Apply the saved training scale.")
    return model, store, cfg, state

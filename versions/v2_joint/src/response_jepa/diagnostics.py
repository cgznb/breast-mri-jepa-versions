"""Fixed, read-only gradient and representation probes. No checkpoint selection."""
from __future__ import annotations
from copy import deepcopy
import random
import numpy as np
import torch
from .data import planned_tasks
from .engine import make_loader
from .objectives import joint_loss, task_world_loss


def effective_rank(x):
    x = x.detach().double().cpu()
    if len(x) < 2:
        return None
    singular = torch.linalg.svdvals(x-x.mean(0))
    power = singular.square()
    if float(power.sum()) <= 1e-15:
        return 0.0
    p = power[power > 1e-15]/power.sum()
    return float(torch.exp(-(p*p.log()).sum()))


def diagnose(model, store, cfg, *, patients=32):
    if cfg.train.phase not in {"C", "joint"}:
        raise ValueError("This task/world gradient probe requires a C or joint checkpoint")
    if not cfg.model.use_mri:
        raise ValueError("MRI representation diagnostics require an MRI-enabled checkpoint")
    cfg = deepcopy(cfg)
    # This fixed subset probe reports sample-mean gradients, not the complete
    # training population's weighted risk; the evaluation loader has no weights.
    cfg.train.sample_weighting = "sample"
    cfg.train.reference_batch_size = None
    model.configure_trainable(cfg.train.phase, cfg.train.freeze_encoder)
    model.eval()
    tasks = planned_tasks(store, cfg, "train", 0)
    rng = random.Random(20261008)
    rng.shuffle(tasks)
    tasks = tasks[:patients]
    params = [p for n, p in model.named_parameters() if n.startswith("fusion.") and p.requires_grad]
    records, states = [], []
    for batch in make_loader(store, cfg, tasks, world=True):
        batch = batch.to(cfg.train.device)
        with torch.enable_grad():
            if cfg.train.phase == "joint":
                _, terms, out = joint_loss(model, batch, cfg, augment=False)
            else:
                _, terms, out = task_world_loss(model, batch, cfg)
            task = sum(getattr(cfg.loss, k)*terms[k] for k in ("fused", "direct", "forecast"))
            world = cfg.loss.tf*terms["tf"] + cfg.loss.rollout*terms["rollout"]
            def vector(value):
                g = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True) if value.requires_grad else [None]*len(params)
                return torch.cat([(v.detach().float() if v is not None else torch.zeros_like(p)).flatten()
                                  for p, v in zip(params, g)])
            a, b = vector(task), vector(world)
            na, nb = float(a.norm()), float(b.norm())
            records.append({"patients": len(batch.keys), "task_gradient_norm_fusion": na,
                            "world_gradient_norm_fusion": nb,
                            "cosine_fusion": float(torch.dot(a, b)/(na*nb)) if na and nb else None})
        with torch.no_grad():
            anchors = batch.inputs.anchor()
            mask = torch.nn.functional.one_hot(anchors, 4).bool()
            value = model.encode_selected(batch.inputs.images, mask, teacher=True)[mask]
            states.append(value.detach().float().cpu())
    values = torch.cat(states)
    pooled = values.mean(1)
    return {"patients": len(values), "gradient_batches": records,
            "anchor_patient_pooled_effective_rank": effective_rank(pooled),
            "anchor_patient_pooled_variance": float(pooled.var(0, unbiased=False).mean()),
            "anchor_same_token_patient_variance": float(values.var(0, unbiased=False).mean()),
            "notes": ["No optimizer steps performed", "Probe anchors can have different stages in dynamic mode",
                      "Gradient probe uses unweighted sample means on its fixed subset",
                      "Low rank or negative gradient cosine is diagnostic, not proof of clinical failure"]}

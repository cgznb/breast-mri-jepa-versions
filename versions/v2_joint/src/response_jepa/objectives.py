"""FP32 loss reductions and explicit teacher-forced vs free-rollout supervision."""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F


# These losses have a per-prefix value before reduction. The batch-statistic
# regularizers intentionally remain outside patient weighting.
ADDITIVE_TERMS = frozenset({"masked", "repr_pcr", "tf", "rollout", "align",
                            "direct", "forecast", "fused", "consistency"})


def additive_average(values, sample_weight=None, reference_batch_size=None):
    """Sum fixed sample weights; never normalize by their within-batch sum.

    A fixed reference denominator keeps short stage/tail batches from increasing
    each patient's contribution. None retains the original physical-batch mean.
    """
    if values.ndim != 1 or not len(values):
        raise ValueError("Additive losses must have one value per nonempty sample")
    values = values.float()
    if sample_weight is not None:
        if (sample_weight.shape != values.shape or not torch.isfinite(sample_weight).all()
                or (sample_weight < 0).any()):
            raise ValueError("Sample weights must be finite nonnegative [B]")
        values = values * sample_weight.to(device=values.device, dtype=torch.float32)
    if reference_batch_size is None:
        return values.mean()
    if type(reference_batch_size) is not int or reference_batch_size < 1:
        raise ValueError("reference_batch_size must be a positive integer or None")
    return values.sum() / reference_batch_size


def _reduction(batch, cfg):
    weight = None
    if getattr(cfg.train, "sample_weighting", "sample") == "patient_equal":
        weight = (batch.get("sample_weight") if isinstance(batch, dict)
                  else getattr(batch, "sample_weight", None))
        if weight is None:
            raise ValueError("patient_equal requires fixed dataset sample weights")
    return {"sample_weight": weight,
            "reference_batch_size": getattr(cfg.train, "reference_batch_size", None)}


def symmetric_bernoulli_kl(first, second):
    """Per-sample symmetric Bernoulli KL, with gradients to both FP32 logits.

    The logit identity avoids log(0), probability division, and cancellation of
    separate KL terms: half (p-q) times (log_odds(p)-log_odds(q)).
    """
    if first.shape != second.shape:
        raise ValueError("Consistency logits must have the same shape")
    with torch.autocast(device_type=first.device.type, enabled=False):
        first, second = first.float(), second.float()
        return .5 * (first.sigmoid() - second.sigmoid()) * (first - second)


def off_diagonal(x):
    n, m = x.shape
    if n != m:
        raise ValueError("Covariance must be square")
    return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()


def variance_covariance(x, floor=1.0):
    """VICReg-style statistics over PHYSICAL patients, not accumulated microbatches.
    Independent implementation of the standard variance/covariance formulas.
    """
    # .float() alone is NOT sufficient: autocast would recast the covariance matmul.
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        if len(x) < 2:
            return x.sum()*0, x.sum()*0
        centered = x - x.mean(0)
        std = (centered.var(0, unbiased=False) + 1e-4).sqrt()
        variance = F.relu(floor - std).mean()
        covariance = centered.T @ centered / (len(x)-1)
        covariance = off_diagonal(covariance).square().sum() / x.shape[1]
        return variance, covariance


def patient_average(errors, mask, *, sample_weight=None, reference_batch_size=None):
    """First mean over a patient's targets. No-target patients contribute ZERO.
    Then mean over ALL patients in the physical batch (stable T3 loss scaling).
    """
    counts = mask.sum(1)
    values = errors.masked_fill(~mask, 0).sum(1) / counts.clamp_min(1)
    return additive_average(values, sample_weight, reference_batch_size), values, counts


def bce(logits, labels, mask, *, sample_weight=None, reference_batch_size=None):
    if logits is None or not mask.any():
        return labels.sum()*0
    per = F.binary_cross_entropy_with_logits(logits.float(), labels.float(), reduction="none")
    return additive_average(per.masked_fill(~mask, 0), sample_weight, reference_batch_size)


def representation_loss(model, batch, cfg, *, augment=True):
    """A predicts masked SAME-VISIT features. It never forecasts a later visit."""
    x = batch["image"]
    reduction = _reduction(batch, cfg)
    if augment:
        # Same spatial transform for full target and masked online input; all DCE channels together.
        flip = torch.rand(len(x), device=x.device) < .5
        x = torch.where(flip[:, None, None, None, None], x.flip(-1), x)
    b, _, d, h, w = x.shape
    grid = cfg.model.token_grid
    n = math.prod(grid)
    count = max(1, min(n-1, round(n * cfg.loss.mask_ratio)))
    order = torch.rand(b, n, device=x.device).argsort(1)
    mask = torch.zeros(b, n, device=x.device, dtype=torch.bool)
    mask.scatter_(1, order[:, :count], True)
    voxels = F.interpolate(mask.float().reshape(b, 1, *grid), size=(d, h, w), mode="nearest").bool()
    online_image = x.masked_fill(voxels, 0)
    if augment:
        # Training augmentation only; it is not a second normalization of stored data.
        scale = 1 + .03 * torch.randn(b, 1, 1, 1, 1, device=x.device)
        online_image = online_image * scale
    online = model.encoder(online_image)
    with torch.no_grad():
        target = model.target_encoder(x)
    pred = model.mask_predictor(online)
    error = (pred.float() - target.float()).abs().mean(-1)
    per_masked = error.masked_fill(~mask, 0).sum(1) / mask.sum(1)
    masked = additive_average(per_masked, **reduction)
    projected = model.repr_projector(online.mean(1))
    # Per-stage statistics. The sampler normally gives same-stage physical batches.
    var_terms, cov_terms, token_terms = [], [], []
    for s in batch["stage"].unique():
        sel = batch["stage"] == s
        if sel.sum() >= 2:
            va, co = variance_covariance(projected[sel], cfg.loss.variance_floor)
            var_terms.append(va)
            cov_terms.append(co)
            std = (online[sel].float().var(0, unbiased=False) + 1e-4).sqrt()
            token_terms.append(F.relu(cfg.loss.token_variance_floor - std).mean())
    zero = online.float().sum()*0
    avg = lambda vals: torch.stack(vals).mean() if vals else zero
    var, cov, tv = avg(var_terms), avg(cov_terms), avg(token_terms)
    label_loss = (bce(model.repr_head(online.mean(1)).squeeze(-1), batch["label"], batch["label_mask"],
                      **reduction)
                  if cfg.loss.repr_pcr else zero)
    l = cfg.loss
    loss = l.masked*masked + l.variance*var + l.covariance*cov + l.token_variance*tv + l.repr_pcr*label_loss
    return loss, {"masked": masked, "variance": var, "covariance": cov,
                  "token_variance": tv, "repr_pcr": label_loss,
                  "regularized_stages": float(len(var_terms)), "actual_mask_fraction": mask.float().mean()}


def _classification_terms(out, sup, readout, reduction, zero):
    terms = {"fused": bce(out["logit"], sup.label, sup.label_mask, **reduction),
             "direct": zero, "forecast": zero}
    # Avoid counting the identical direct BCE twice in direct-only configurations.
    if readout != "direct":
        terms["direct"] = bce(out["direct_logit"], sup.label, sup.label_mask, **reduction)
        terms["forecast"] = bce(out["forecast_logit"], sup.label, sup.label_mask, **reduction)
    return terms


def task_world_loss(model, batch, cfg, *, updates=0):
    inp, sup = batch.inputs, batch.targets
    sup.validate(inp)
    l, phase = cfg.loss, cfg.train.phase
    reduction = _reduction(batch, cfg)
    need_free = bool(l.rollout or (phase in {"C", "joint"} and model.cfg.readout != "direct"))
    # This forward call receives ONLY Inputs. No teacher-forced states can reach pCR.
    out = model(inp, need_world=need_free)
    zero = out["direct_logit"].float().sum()*0
    terms = {name: zero for name in ("tf", "rollout", "direct", "forecast", "fused", "align")}
    if l.tf or l.rollout or l.align:
        if l.tf:
            predicted, states, known, pair_mask = model.teacher_forcing(inp, sup, out["transitions"])
            err = (predicted.float() - states[:, 1:].float()).abs().mean((-1, -2))
            terms["tf"] = patient_average(err, pair_mask, **reduction)[0]
        else:
            states, known = model.teacher_states(inp, sup)
        if l.rollout:
            err = (out["states"].float() - states.float()).abs().mean((-1, -2))
            terms["rollout"] = patient_average(err, sup.future_mask, **reduction)[0]
        if l.align:
            err = (out["online_states"].float() - states.float()).square().mean((-1, -2))
            terms["align"] = patient_average(err, inp.observed, **reduction)[0]
    if phase in {"C", "joint"}:
        terms.update(_classification_terms(out, sup, model.cfg.readout, reduction, zero))
        if getattr(l, "consistency", 0):
            # Same Inputs object, independent dropout. Neither view sees Targets.
            # The second view needs no teacher-forced, alignment or A branch.
            second = model(inp, need_world=model.cfg.readout != "direct")
            second_terms = _classification_terms(second, sup, model.cfg.readout, reduction, zero)
            for name, value in second_terms.items():
                terms[name] = .5 * (terms[name] + value)
            per_consistency = symmetric_bernoulli_kl(out["logit"], second["logit"])
            consistency = additive_average(per_consistency.masked_fill(~sup.label_mask, 0), **reduction)
            # Drop graph references not required by classification, particularly
            # the detached forecast path's unused second world rollout.
            del second
        else:
            consistency = zero
    else:
        consistency = zero
    loss = sum(getattr(l, name)*term for name, term in terms.items())
    if getattr(l, "consistency", 0) and phase in {"C", "joint"}:
        warmup = getattr(l, "consistency_warmup_updates", 500)
        weight = l.consistency * (min(max(updates, 0) / warmup, 1.) if warmup else 1.)
        terms["consistency"] = consistency
        terms["consistency_weight"] = weight
        loss = loss + weight * consistency
    return loss, terms, out


def joint_loss(model, batch, cfg, *, augment=True, updates=0):
    """One update combines same-visit representation, future-state and pCR losses.

    The representation branch uses one latest LEGAL observed visit per patient.
    Future targets never become encoder inputs for the task or representation branch.
    """
    loss, terms, out = task_world_loss(model, batch, cfg, updates=updates)
    anchors = batch.inputs.anchor()
    rows = torch.arange(len(anchors), device=anchors.device)
    representation_batch = {
        "image": batch.inputs.images[rows, anchors],
        "stage": anchors,
        "label": batch.targets.label,
        "label_mask": batch.targets.label_mask,
        "sample_weight": getattr(batch, "sample_weight", None),
    }
    auxiliary, representation_terms = representation_loss(model, representation_batch, cfg, augment=augment)
    terms.update(representation_terms)
    return loss + auxiliary, terms, out

#!/usr/bin/env python3
"""Walk one real MRI batch through the existing model, without saving a run."""
from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import torch

from mri_vla_jepa.data import RawMRIStore
from mri_vla_jepa.io import autocast, seed_all
from mri_vla_jepa.model import RawVLAJEPA
from mri_vla_jepa.train_config import load_config
from mri_vla_jepa.training import (
    _configure, _encoder_training_state, _learning_rates, _make_optimizer, load_trained,
)


ROOT = Path(__file__).resolve().parents[1]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    source = result.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path, help="Defaults to the original full-loss T0 configuration")
    source.add_argument("--checkpoint", type=Path, help="Use trained weights and their saved config/scaling")
    result.add_argument("--manifest", type=Path, default=ROOT / "data/registered_roi32/manifest.json")
    result.add_argument("--split", choices=("train", "val", "test"), default="train")
    result.add_argument("--landmark", type=int, choices=range(4), default=0)
    result.add_argument("--patient-offset", type=int, default=0,
                        help="Offset in split patients with a real MRI available at the requested stage")
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--mode", choices=("predict", "loss", "train-step"), default="loss",
                        help="predict: eval forward; loss: training forward; train-step: also backward/AdamW/EMA")
    result.add_argument("--device", default="cpu")
    result.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    result.add_argument("--break", dest="pause", action="store_true", help="Enter pdb before data assembly")
    return result


def describe(value):
    if isinstance(value, torch.Tensor):
        return f"shape={list(value.shape)} dtype={value.dtype} device={value.device}"
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(describe(item) for item in value) + ")"
    return type(value).__name__


def attach_shape_trace(model):
    if model.cfg.encoder_type in {"vq_prequant", "vq_encoder"}:
        encoder_names = ("frontend.encoder.input", "frontend.encoder.downsamples.0",
                         "frontend.encoder.downsamples.1", "frontend.encoder")
        if model.encoder.frontend.feature_source == "pre_quant":
            encoder_names += ("frontend.pre_quant",)
        encoder_names += ("frontend", "pool", "projection", "spatial", "norm")
    else:
        encoder_names = ("stem.0", "stem.3", "stem.6", "spatial", "norm")
    names = (tuple(f"encoder.{name}" for name in encoder_names)
             + ("clinical", "arm", "fusion", "pcr_head")
             + tuple(f"teacher_encoder.{name}" for name in encoder_names)
             + ("world_predictor", "reconstruction"))
    handles = []
    for name in names:
        def trace(module, inputs, output, name=name):
            print(f"  {name}: {describe(inputs[0])} -> {describe(output)}")
        handles.append(model.get_submodule(name).register_forward_hook(trace))
    return handles


def load_debug_model(args):
    """Reuse saved weights, or initialize the fresh model exactly as configured."""
    if args.checkpoint:
        model, cfg, checkpoint_state = load_trained(args.checkpoint, args.device)
        seed_all(cfg.seed, threads=cfg.threads, deterministic=cfg.deterministic)
        print(f"weights: trained checkpoint ({args.checkpoint})")
        return model, cfg, checkpoint_state["normalization"]
    cfg = load_config(args.config or ROOT / "configs/registered_roi32_epoch200_t0_seed17.yaml")
    if cfg.initialize_from_checkpoint:
        raise ValueError(
            "This one-batch script does not initialize a fine-tuning stage from a frozen-stage "
            "config. Use --checkpoint PATH to inspect saved weights; fresh fine-tuning debug "
            "configs may set encoder_checkpoint and clear initialize_from_checkpoint."
        )
    seed_all(cfg.seed, threads=cfg.threads, deterministic=cfg.deterministic)
    model = RawVLAJEPA(cfg.model).to(args.device)
    if cfg.model.encoder_type in {"vq_prequant", "vq_encoder"}:
        if not cfg.encoder_checkpoint:
            raise ValueError("Fresh VQ debugging requires encoder_checkpoint")
        metadata = model.initialize_pretrained_encoder(cfg.encoder_checkpoint)
        print(f"weights: pretrained VQ encoder ({cfg.encoder_checkpoint}); "
              "adapter and backbone use random initialization")
        print("pretrained encoder:", metadata)
    else:
        print("weights: random initialization; this run only explains computation")
    return model, cfg, None


def configure_debug_optimizer(model, cfg, mode):
    """Apply production freezing and differential learning rates for one batch."""
    _configure(model, cfg, cfg.phase(0))
    print("encoder state:", _encoder_training_state(model, cfg))
    optimizer = _make_optimizer(model, cfg) if mode == "train-step" else None
    if optimizer is not None:
        print("learning rates:", _learning_rates(optimizer))
    return optimizer


def main(argv=None):
    args = parser().parse_args(argv)
    if args.batch_size < 1 or args.patient_offset < 0:
        raise ValueError("batch-size must be positive and patient-offset nonnegative")
    if args.precision == "bf16" and not args.device.startswith("cuda"):
        raise ValueError("This project's bf16 autocast requires --device cuda")

    # 1. Config/model: checkpoint mode reuses its own normalization coordinates.
    model, cfg, normalization = load_debug_model(args)
    if cfg.landmarks == "t0" and args.landmark != 0:
        raise ValueError("A fixed-T0 config/checkpoint only supports --landmark 0")
    if args.mode != "predict" and cfg.schedule != "one_stage":
        raise ValueError("Loss/update debugging supports the one-stage V1 configs only")
    if args.mode == "train-step" and cfg.accumulation != 1:
        raise ValueError("A single debug update requires accumulation=1; this script does not simulate accumulation")
    if args.mode == "train-step":
        print("update: one in-memory demonstration with a fresh AdamW; optimizer/RNG/scheduler are not resumed")
    # Saved runs may name a cache on the training machine. A checkpoint supplies
    # its normalization; read the selected manifest's local NPZ files directly.
    image_cache = None if args.checkpoint else cfg.image_cache
    store = RawMRIStore(args.manifest, cfg.image_shape, cfg.allow_synthetic, normalization,
                        image_cache=image_cache,
                        require_lesion_weights=(cfg.model.lesion_world_weighting or cfg.model.lesion_pooling))
    eligible = [index for index in store.by_split[args.split]
                if args.landmark in store.allowed_landmarks(index)]
    indices = eligible[args.patient_offset:args.patient_offset + args.batch_size]
    if len(indices) != args.batch_size:
        raise ValueError("Not enough eligible patients for the requested offset and batch-size")
    tasks = [(index, args.landmark) for index in indices]
    print(f"batch: split={args.split} landmark=T{args.landmark} size={len(tasks)} mode={args.mode}")
    print("data assembly: RawMRIStore.batch (synchronous, same input/supervision contract as training)")
    if args.pause:
        breakpoint()

    # 2. Data: inp contains legal history; sup contains labels/future MRI only for losses.
    supervised = args.mode != "predict"
    inp, sup = store.batch(tasks, supervised=supervised,
                           future_supervision=bool(cfg.jepa_weight or cfg.rollout_weight or cfg.flow_weight))
    inp = inp.to(args.device)
    if sup is not None:
        sup = sup.to(args.device)
    for record_name, record in (("inp", inp), ("sup", sup)):
        if record is not None:
            for field in fields(record):
                print(f"{record_name}.{field.name}: {describe(getattr(record, field.name))}")
    print("observed_mask:", inp.observed_mask.cpu().tolist())
    print("query_mask:", inp.query_mask.cpu().tolist())
    if sup is not None:
        print("future_mask:", sup.future_mask.cpu().tolist())

    # 3. Forward: hooks display shapes; stepping into these calls follows production code.
    handles = attach_shape_trace(model)
    try:
        if args.mode == "predict":
            model.eval()
            with torch.no_grad(), autocast(args.device, args.precision):
                output = model(inp)
            probability = output["pcr_logit"].float().sigmoid()
            print("pcr_logit:", describe(output["pcr_logit"]))
            print("pcr_probability:", describe(probability))
        else:
            optimizer = configure_debug_optimizer(model, cfg, args.mode)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with autocast(args.device, args.precision):
                loss, terms = model.compute_loss(
                    inp, sup, task_weight=cfg.task_weight, jepa_weight=cfg.jepa_weight,
                    rollout_weight=cfg.rollout_weight, flow_weight=cfg.flow_weight,
                    reconstruction_weight=cfg.reconstruction_weight,
                    variance_weight=cfg.variance_weight, variance_definition=cfg.variance_definition,
                    representation_only=cfg.phase(0) == "representation")
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite debug loss")
            print("loss terms:", {name: float(value.detach().float().cpu()) for name, value in terms.items()})

            # 4. Optional one update: same backward -> clipping -> AdamW -> EMA order.
            if optimizer is not None:
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("Non-finite debug gradient norm")
                if any(p.grad is not None for p in model.teacher_encoder.parameters()):
                    raise AssertionError("EMA teacher unexpectedly received a gradient")
                optimizer.step()
                model.update_teacher()
                print("backward/AdamW/EMA completed; gradient_norm:", float(gradient_norm))
    finally:
        for handle in handles:
            handle.remove()
    print("Finished one batch. No checkpoint or prediction file was written.")


if __name__ == "__main__":
    main()

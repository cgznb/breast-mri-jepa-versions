#!/usr/bin/env python3
"""Disposable full-shape B16 readout/gradient/evaluation gates; no weights saved."""
from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
import sys

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from run_branch_ablation import VARIANTS, BASELINE, ablation_config
from check_overfit_cuda import complete_patients
from check_joint_sequence_cuda import Occupancy, memory
from response_jepa.data import Store
from response_jepa.engine import autocast, setup, make_loader, optimizer_for, evaluate, write_json
from response_jepa.model import ResponseJEPA
from response_jepa.objectives import joint_loss


def case(variant):
    cfg = ablation_config(BASELINE / "input_configs/B0_dynamic_seed17.yaml", variant)
    setup(cfg)
    manifest = Path("/root/autodl-tmp/line b/breast-mri-vla-jepa/data/registered_roi32/manifest.json")
    store = Store(manifest, cfg.train.image_shape, image_cache=cfg.train.image_cache)
    train_ids, val_ids = complete_patients(store, "train"), complete_patients(store, "val")
    model = ResponseJEPA(cfg.model).cuda()
    model.configure_trainable("joint", False)
    model.world_trained = True
    if cfg.model.readout == "direct":
        for module in (model.terminal_pool, model.terminal_head, model.gate):
            module.requires_grad_(False)
    optimizer = optimizer_for(model, cfg)
    torch.cuda.reset_peak_memory_stats()
    steps = []
    with Occupancy() as occupancy:
        for stage in (0, 3):
            model.train()
            batch = next(iter(make_loader(store, cfg, [(i, stage) for i in train_ids],
                                          training=True, world=True))).to("cuda")
            assert len(set(batch.keys)) == len(batch.keys) == 16
            assert batch.inputs.anchor().unique().tolist() == [stage]
            optimizer.zero_grad(set_to_none=True)
            with autocast(cfg):
                loss, terms, prediction = joint_loss(model, batch, cfg, augment=False)
            assert torch.isfinite(loss)
            if variant == "D_direct":
                assert prediction["forecast_logit"] is None
                torch.testing.assert_close(prediction["logit"], prediction["direct_logit"])
            elif variant == "F_forecast":
                torch.testing.assert_close(prediction["logit"], prediction["forecast_logit"])
                assert prediction["gate"] is None
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            assert all(p.grad is None for p in model.target_encoder.parameters())
            def norm(module):
                return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
            routes = {"direct": norm(model.direct_head), "forecast": norm(model.terminal_head),
                      "gate": norm(model.gate)}
            assert (routes["direct"] > 0) == (variant != "F_forecast")
            assert (routes["forecast"] > 0) == (variant != "D_direct")
            assert (routes["gate"] > 0) == (variant == "G_gated")
            if stage == 3:
                assert float(terms["tf"]) == float(terms["rollout"]) == 0.
            loss_value = float(loss.detach())
            del loss, terms, prediction, batch
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad, error_if_nonfinite=True)
            optimizer.step()
            model.update_teacher(cfg.train.ema)
            assert all(torch.isfinite(p).all() for p in model.parameters())
            steps.append({"stage": stage, "physical_batch": 16, "loss": loss_value,
                          "gradient_routes": routes, "teacher_no_gradient": True,
                          "loss_gradients_parameters_finite": True, "memory": memory(occupancy)})
            print(variant, "stage", stage, "passed", flush=True)
        metrics, rows, _ = evaluate(model, store, cfg, "val",
                                   tasks=[(i, s) for i in val_ids for s in range(4)])
        assert len(rows) == 64 and math.isfinite(metrics["selection"])
        result = {"variant": variant, "steps": steps, "evaluation_prefixes": 64,
                  "fp32_evaluation_finite": True,
                  "sampled_nvidia_peak_mib": max(occupancy.values, default=None),
                  "memory": memory(occupancy)}
    del model, optimizer, store
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "reports/branch_cuda_gate_20261010.json")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Use a new capacity-report output")
    report = {"formal_training": False, "weights_saved": False, "cases": [], "all_checks_passed": False}
    for variant in VARIANTS:
        report["cases"].append(case(variant))
        write_json(args.output, report)
    report["all_checks_passed"] = True
    write_json(args.output, report)


if __name__ == "__main__":
    main()

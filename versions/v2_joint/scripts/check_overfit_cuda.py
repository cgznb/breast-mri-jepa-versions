"""Disposable real B16 CUDA gates for B0 and the patient-weighted R-Drop B4."""
from pathlib import Path
import argparse
import gc
import json
import math
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from check_joint_sequence_cuda import Occupancy, memory
from response_jepa.config import load_config
from response_jepa.data import Store
from response_jepa.engine import autocast, evaluate, make_loader, optimizer_for, setup, write_json
from response_jepa.model import ResponseJEPA
from response_jepa.objectives import joint_loss
from response_jepa.update_engine import schedule_updates

DEFAULT_DATA = Path("/root/autodl-tmp/line b/breast-mri-vla-jepa/data/registered_roi32")


def gate_config(name, cache):
    cfg = load_config(ROOT / "configs" / "joint_dynamic_all_prefixes_batch16_seed17.yaml")
    cfg.train.image_cache = str(Path(cache).resolve())
    cfg.train.reference_batch_size = 16
    cfg.train.max_updates = 3000
    cfg.train.warmup_updates = 200
    cfg.train.eval_every_updates = 100
    cfg.train.patience_updates = 500
    cfg.train.train_eval_patients = 0
    cfg.train.save_predictions = False
    cfg.train.sample_weighting = "patient_equal" if name == "B4" else "sample"
    cfg.model.forecast_detach = name == "B4"
    cfg.loss.consistency = .1 if name in {"B3", "B4"} else 0.
    cfg.loss.consistency_warmup_updates = 500
    return cfg.validate()


def complete_patients(store, split):
    # Require immediate availability, not only a nominally complete visit list.
    ids = [i for i in store.splits[split]
           if store.allowed(i, "dynamic") == [0, 1, 2, 3]
           and all(store.patients[i]["visits"][j] is not None
                   and store.patients[i]["visits"][j]["available_at"] <= j for j in range(4))][:16]
    if len(ids) != 16:
        raise ValueError(f"CUDA gate requires 16 complete immediately available {split} patients")
    return ids


def case(name, manifest, cache):
    cfg = gate_config(name, cache)
    setup(cfg)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    store = Store(manifest, cfg.train.image_shape, image_cache=cfg.train.image_cache)
    train_ids, val_ids = complete_patients(store, "train"), complete_patients(store, "val")
    model = ResponseJEPA(cfg.model).cuda()
    model.configure_trainable("joint", False)
    model.world_trained = True
    optimizer = optimizer_for(model, cfg)
    steps = []
    with Occupancy() as occupancy:
        for step, stage in enumerate([0, 1, 2, 3, 3], start=1):
            model.train()
            loader = make_loader(store, cfg, [(i, stage) for i in train_ids], training=True, world=True)
            cpu_batch = next(iter(loader))
            batch = cpu_batch.to("cuda", non_blocking=cfg.train.pin_memory)
            batch.inputs.validate()
            batch.targets.validate(batch.inputs)
            assert batch.inputs.anchor().unique().tolist() == [stage]
            assert len(set(batch.keys)) == len(batch.keys) == 16
            assert (batch.sample_weight is not None) == (name == "B4")
            weight_sum = float(batch.sample_weight.sum()) if batch.sample_weight is not None else 16.
            # Explicitly verify each optimizer update leaves the teacher alone,
            # then exactly one EMA update applies the documented interpolation.
            teacher_before = {key: value.detach().clone()
                              for key, value in model.target_encoder.named_parameters()}
            optimizer.zero_grad(set_to_none=True)
            schedule_updates(optimizer, cfg, step)
            with autocast(cfg):
                loss, terms, prediction = joint_loss(model, batch, cfg, augment=False, updates=step)
            assert bool(torch.isfinite(loss))
            loss.backward()
            assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            assert all(p.grad is None for p in model.target_encoder.parameters())
            values = {key: float(value.detach() if isinstance(value, torch.Tensor) else value)
                      for key, value in terms.items()}
            assert all(math.isfinite(value) for value in values.values())
            assert ("consistency" in terms) == (name in {"B3", "B4"})
            if stage == 3:
                assert values["tf"] == values["rollout"] == 0.
            loss_value = float(loss.detach())
            del batch, loss, terms, prediction
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad, error_if_nonfinite=True)
            optimizer.step()
            for key, teacher in model.target_encoder.named_parameters():
                torch.testing.assert_close(teacher, teacher_before[key], atol=0, rtol=0)
            model.update_teacher(cfg.train.ema)
            online = dict(model.encoder.named_parameters())
            for key, teacher in model.target_encoder.named_parameters():
                expected = torch.lerp(teacher_before[key], online[key].detach(), 1-cfg.train.ema)
                torch.testing.assert_close(teacher, expected, atol=0, rtol=0)
            assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
            assert all(p.grad is None for p in model.target_encoder.parameters())
            del teacher_before, online, expected, teacher
            item = {"step": step, "stage": stage, "physical_batch": 16,
                    "sample_weight_sum": weight_sum, "loss": loss_value,
                    "gradient_norm": float(norm), "loss_terms": values,
                    "loss_terms_finite": True, "gradients_finite": True,
                    "parameters_finite": True, "teacher_no_gradient": True,
                    "teacher_unchanged_by_optimizer": True, "teacher_ema_exact": True,
                    "legal_input_and_strict_future": True, "memory": memory(occupancy)}
            steps.append(item)
            print(json.dumps({"case": name, "training_step": item}, allow_nan=False), flush=True)
            del loader, cpu_batch, norm
        training_peak = memory(occupancy)
        # Keep the optimizer and final gradients alive, as in the real runner.
        torch.cuda.reset_peak_memory_stats()
        tasks = [(i, landmark) for i in val_ids for landmark in range(4)]
        metrics, rows, world_rows = evaluate(model, store, cfg, "val", tasks=tasks)
        assert len(rows) == len(tasks) == 64
        assert math.isfinite(metrics["selection"])
        assert not model.training and not model.target_encoder.training
        evaluation = {"precision": "fp32", "eval_batch_size": cfg.train.eval_batch_size,
                      "patients": 16, "prefixes": len(rows), "world_targets": len(world_rows),
                      "selection_finite": True, "teacher_eval": True, "memory": memory(occupancy)}
        result = {"case": name, "profile": "dynamic", "sampling": cfg.train.prefix_sampling,
                  "sample_weighting": cfg.train.sample_weighting, "forecast_detach": cfg.model.forecast_detach,
                  "consistency": cfg.loss.consistency, "max_updates": cfg.train.max_updates,
                  "warmup_updates": cfg.train.warmup_updates, "training_steps": steps,
                  "training_peak": training_peak, "evaluation": evaluation,
                  "nvidia_own_process_sampled_peak_mib": max(occupancy.values, default=None),
                  "nvidia_samples": len(occupancy.values), "seconds": round(time.monotonic()-started, 2)}
        print(json.dumps({"case": name, "evaluation": evaluation}, allow_nan=False), flush=True)
    del model, optimizer, store, metrics, rows, world_rows
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DATA / "manifest.json")
    parser.add_argument("--cache", type=Path, default=DEFAULT_DATA / "image_cache_20261004")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "overfit_cuda_gate_20261009.json")
    parser.add_argument("--cases", nargs="+", choices=("B0", "B3", "B4"), default=["B0", "B4"])
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Capacity report already exists; choose a new --output")
    if not torch.cuda.is_available():
        raise RuntimeError("This real-capacity gate requires CUDA")
    report = {"scope": "real B16 multi-update and partial FP32 validation; disposable random weights",
              "formal_training": False, "weights_saved": False,
              "nvidia_peak_definition": "own process every 0.5 seconds and completed steps; sampled lower bound",
              "verification_overhead": "one encoder teacher clone for exact EMA comparison per update",
              "cases": []}
    try:
        for name in args.cases:
            report["cases"].append(case(name, args.manifest.resolve(), args.cache.resolve()))
            write_json(args.output, {**report, "all_checks_passed": False})
        report["all_checks_passed"] = True
        write_json(args.output, report)
        print("Saved CUDA capacity report:", args.output.resolve(), flush=True)
    except Exception as exc:
        write_json(args.output, {**report, "all_checks_passed": False,
                                 "failure": {"type": type(exc).__name__, "message": str(exc)}})
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

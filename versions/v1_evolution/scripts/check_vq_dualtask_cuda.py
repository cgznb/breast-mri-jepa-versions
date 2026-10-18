#!/usr/bin/env python3
"""Real B16 gates for PCR plus JEPA training and state/copy evaluation."""
from __future__ import annotations

import argparse
import runpy
import gc
from pathlib import Path
import time

import torch

from mri_vla_jepa.data import RawMRIStore
from mri_vla_jepa.diagnostics import aggregate_world_errors, run_training_probe, world_error_rows
from mri_vla_jepa.io import autocast, seed_all, write_json
from mri_vla_jepa.model import RawVLAJEPA
from mri_vla_jepa.train_config import load_config
from mri_vla_jepa.training import _configure, _learning_rates, _make_optimizer

ROOT = Path(__file__).resolve().parents[1]
_release = runpy.run_path(str(ROOT / "scripts/release_runtime.py"))
ORIGINAL = _release["version_directory"](ROOT, "v1")
CODEC = Path("/root/autodl-tmp/MAM/symm-fm-world-v2/data/registered_roi32/codec.pt")


def new_model(cfg):
    model = RawVLAJEPA(cfg.model).to(cfg.device)
    if cfg.model.encoder_type == "vq_prequant":
        model.initialize_pretrained_encoder(CODEC)
    _configure(model, cfg, "joint")
    return model


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def check(path):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required")
    cfg0 = load_config(ROOT / "configs/vq_dualtask_frozen_dynamic_seed17.yaml")
    seed_all(cfg0.seed, threads=cfg0.threads, deterministic=cfg0.deterministic)
    store = RawMRIStore(ORIGINAL / "data/registered_roi32/manifest.json",
                        cfg0.image_shape, image_cache=cfg0.image_cache)
    complete = [i for i in store.by_split["train"]
                if all(v is not None for v in store.patients[i]["visits"])][:32]
    assert len(complete) == 32
    report = {"passed": False, "engineering_only": True, "physical_batch_size": 16,
        "accumulation": 1, "precision": "bf16", "gpu": torch.cuda.get_device_name(),
        "loss_weights": {"pcr": 1., "jepa": .5, "reconstruction": 0., "variance": 0., "flow": 0.},
        "finetune_gate_initialization": "pretrained frontend; complete frozen-to-finetune initialization tested on CPU",
        "patient_rows_saved": False, "cases": [], "diagnostics": []}
    for stage in ("raw", "frozen", "finetune"):
        cfg = load_config(ROOT / f"configs/vq_dualtask_{stage}_dynamic_seed17.yaml")
        for landmark in (0, 2, 3):
            cleanup()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            case = {"stage": stage, "landmark": landmark, "passed": False}
            model = optimizer = inp = sup = loss = terms = gradients = None
            try:
                model = new_model(cfg)
                optimizer = _make_optimizer(model, cfg)
                world_before = {k: p.detach().cpu().clone()
                                for k, p in model.world_predictor.named_parameters()}
                frontend_before = ({k: p.detach().cpu().clone()
                                    for k, p in model.encoder.frontend.named_parameters()}
                                   if stage != "raw" else None)
                head_before = model.pcr_head[-1].weight.detach().clone()
                adapter = model.encoder.projection if stage != "raw" else model.encoder.stem[0]
                adapter_before = adapter.weight.detach().clone()
                inp, sup = store.batch([(i, landmark) for i in complete[:16]],
                                       pin_memory=True, future_supervision=True)
                assert list(inp.images.shape) == [16, 4, 3, 32, 128, 128]
                assert int(inp.observed_mask.sum()) == 16 * (landmark + 1)
                assert int(sup.future_mask.sum()) == 16 * (3 - landmark)
                assert not inp.images[~inp.observed_mask].count_nonzero()
                inp, sup = inp.to(cfg.device, non_blocking=True), sup.to(cfg.device, non_blocking=True)
                with autocast(cfg.device, cfg.precision):
                    loss, terms = model.compute_loss(inp, sup, task_weight=cfg.task_weight,
                        jepa_weight=cfg.jepa_weight, flow_weight=cfg.flow_weight,
                        reconstruction_weight=cfg.reconstruction_weight,
                        variance_weight=cfg.variance_weight, variance_definition=cfg.variance_definition)
                assert all(torch.isfinite(v).all() for v in terms.values())
                assert float(terms["pair_count"]) == 16 * (3 - landmark)
                assert bool(terms["jepa"] > 0) == (landmark < 3)
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                assert gradients and all(torch.isfinite(g).all() for g in gradients)
                assert all(not p.requires_grad and p.grad is None for p in model.teacher_encoder.parameters())
                assert not model.teacher_encoder.training
                for module in (model.fusion, model.pcr_head, adapter):
                    assert any(p.grad is not None and p.grad.count_nonzero() for p in module.parameters())
                world_gradient = any(p.grad is not None and p.grad.count_nonzero()
                                     for p in model.world_predictor.parameters())
                assert bool(world_gradient) == (landmark < 3)
                if stage != "raw":
                    frontend_gradient = any(p.grad is not None and p.grad.count_nonzero()
                                            for p in model.encoder.pretrained_parameters())
                    assert bool(frontend_gradient) == (stage == "finetune")
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                assert torch.isfinite(norm)
                optimizer.step()
                model.update_teacher()
                assert not torch.equal(head_before, model.pcr_head[-1].weight)
                assert not torch.equal(adapter_before, adapter.weight)
                world_changed = any(not torch.equal(world_before[k], p.detach().cpu())
                                    for k, p in model.world_predictor.named_parameters())
                assert world_changed == (landmark < 3)
                if stage != "raw":
                    changed = any(not torch.equal(frontend_before[k], p.detach().cpu())
                                  for k, p in model.encoder.frontend.named_parameters())
                    assert changed == (stage == "finetune")
                    if stage == "frozen":
                        assert all(torch.equal(frontend_before[k], p.detach().cpu())
                                   for k, p in model.teacher_encoder.frontend.named_parameters())
                assert all(torch.isfinite(p).all() for p in model.parameters())
                case.update(losses={k: float(v.detach()) for k, v in terms.items()},
                    gradient_norm=float(norm), learning_rates=_learning_rates(optimizer),
                    pcr_head_updated=True, adapter_updated=True, world_predictor_updated=world_changed,
                    teacher_no_grad=True)
                if landmark == 0:
                    # Full FP32 TF/free 1-3-step and same-teacher copy evaluation.
                    model.eval()
                    rows, teacher, known = world_error_rows(model, inp, sup, complete[:16])
                    world = aggregate_world_errors(rows)
                    assert world["status"] == "evaluated"
                    assert teacher.shape == (16, 4, 32, 128) and known.all()
                    assert all(world["metrics"][f"horizon_{h}"]["patients"] == 16 for h in (1, 2, 3))
                    case["fp32_world_copy_passed"] = True
                    del rows, teacher, known, world
                if landmark == 3:
                    model.eval()
                    with torch.no_grad():
                        logits = model(inp)["pcr_logit"]
                    assert logits.shape == (16,) and torch.isfinite(logits).all()
                    case["fp32_validation_passed"] = True
                    del logits
                case["passed"] = True
                print(f"{stage} T{landmark}: dual loss/backward/update passed", flush=True)
            except Exception as exc:
                case.update(error_type=type(exc).__name__, error=str(exc))
                print(f"{stage} T{landmark}: {type(exc).__name__}: {exc}", flush=True)
            finally:
                torch.cuda.synchronize()
                case.update(seconds=time.monotonic() - started,
                            peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024 ** 2,
                            peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024 ** 2)
                report["cases"].append(case)
                write_json(path, report)
                del model, optimizer, inp, sup, loss, terms, gradients
                if "adapter" in locals():
                    del adapter, adapter_before, head_before
                cleanup()
            if not case["passed"]:
                return report
    for stage in ("raw", "frozen", "finetune"):
        cfg = load_config(ROOT / f"configs/vq_dualtask_{stage}_dynamic_seed17.yaml")
        for landmark in (0, 3):
            cleanup()
            torch.cuda.reset_peak_memory_stats()
            model = new_model(cfg)
            probe, private = run_training_probe(model, store, cfg, 5,
                                               [(i, landmark) for i in complete])
            assert probe["teacher_frozen"] and probe["rng_and_module_modes_preserved"]
            assert all(batch["all_gradient_values_finite"] for batch in probe["gradient_batches"])
            assert all(p.grad is None for p in model.parameters())
            assert not model.teacher_encoder.training
            if stage != "raw":
                assert model.encoder.frontend.training == (stage == "finetune")
            report["diagnostics"].append({"stage": stage, "landmark": landmark, "passed": True,
                "patients": 32, "batch_size": 16, "teacher_no_grad": True,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024 ** 2})
            print(f"{stage} T{landmark}: fixed dual diagnostic passed", flush=True)
            del model, probe, private
            cleanup()
            write_json(path, report)
    report["passed"] = True
    write_json(path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/vq_dualtask_cuda_20261012.json")
    result = check(parser.parse_args().output.resolve())
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

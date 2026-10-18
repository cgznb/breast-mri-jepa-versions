#!/usr/bin/env python3
"""Real ROI32 B16 engineering gates for the separate VQ encoder experiment."""
from __future__ import annotations

import argparse
import runpy
import gc
from pathlib import Path
import time

import torch

from mri_vla_jepa.data import RawMRIStore
from mri_vla_jepa.diagnostics import run_training_probe
from mri_vla_jepa.io import autocast, seed_all, write_json
from mri_vla_jepa.model import RawVLAJEPA
from mri_vla_jepa.train_config import load_config
from mri_vla_jepa.training import _configure, _make_optimizer, _learning_rates

ROOT = Path(__file__).resolve().parents[1]
_release = runpy.run_path(str(ROOT / "scripts/release_runtime.py"))
ORIGINAL = _release["version_directory"](ROOT, "v1")
CODEC = Path("/root/autodl-tmp/MAM/symm-fm-world-v2/data/registered_roi32/codec.pt")


def check(report_path):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("These gates require CUDA BF16")
    cfg0 = load_config(ROOT / "configs/vq_frozen_dynamic_seed17.yaml")
    seed_all(cfg0.seed, threads=cfg0.threads, deterministic=cfg0.deterministic)
    store = RawMRIStore(ORIGINAL / "data/registered_roi32/manifest.json",
                        cfg0.image_shape, image_cache=cfg0.image_cache)
    indices = [i for i in store.by_split["train"]
               if all(visit is not None for visit in store.patients[i]["visits"])][:16]
    if len(indices) != 16:
        raise AssertionError("Need sixteen complete four-visit training patients")
    report = {"engineering_only": True, "passed": False,
              "physical_batch_size": 16, "accumulation": 1,
              "precision": "bf16", "gpu": torch.cuda.get_device_name(),
              "torch": str(torch.__version__), "cases": [], "diagnostics": [],
              "finetune_gate_initialization": "pretrained frontend; complete stage A to B warm-start tested separately on CPU",
              "patient_rows_saved": False}
    for stage in ("frozen", "finetune"):
        cfg = load_config(ROOT / f"configs/vq_{stage}_dynamic_seed17.yaml")
        for landmark in (0, 2, 3):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            model = optimizer = inp = sup = loss = terms = gradient_norm = None
            case = {"stage": stage, "landmark": landmark, "passed": False}
            try:
                model = RawVLAJEPA(cfg.model).to(cfg.device)
                model.initialize_pretrained_encoder(CODEC)
                _configure(model, cfg, "joint")
                optimizer = _make_optimizer(model, cfg)
                before = {name: parameter.detach().cpu().clone()
                          for name, parameter in model.encoder.frontend.named_parameters()}
                projection = model.encoder.projection.weight.detach().clone()
                head = model.pcr_head[-1].weight.detach().clone()
                calls = {"online_visits": [], "teacher_visits": []}
                model.encoder.register_forward_pre_hook(
                    lambda module, args: calls["online_visits"].append(len(args[0])))
                model.teacher_encoder.register_forward_pre_hook(
                    lambda module, args: calls["teacher_visits"].append(len(args[0])))
                inp, sup = store.batch([(i, landmark) for i in indices],
                                       pin_memory=True, future_supervision=False)
                assert list(inp.images.shape) == [16, 4, 3, 32, 128, 128]
                assert int(inp.observed_mask.sum()) == 16 * (landmark + 1)
                assert not sup.future_mask.any()
                assert not inp.images[~inp.observed_mask].count_nonzero()
                inp, sup = inp.to(cfg.device, non_blocking=True), sup.to(cfg.device, non_blocking=True)
                with autocast(cfg.device, cfg.precision):
                    loss, terms = model.compute_loss(inp, sup, task_weight=cfg.task_weight,
                        jepa_weight=cfg.jepa_weight, flow_weight=cfg.flow_weight,
                        reconstruction_weight=cfg.reconstruction_weight,
                        variance_weight=cfg.variance_weight, variance_definition=cfg.variance_definition)
                assert all(torch.isfinite(term).all() for term in terms.values())
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                assert gradients and all(torch.isfinite(g).all() for g in gradients)
                assert all(p.grad is None and not p.requires_grad for p in model.teacher_encoder.parameters())
                assert not calls["teacher_visits"]
                assert all(p.grad is None for p in model.world_predictor.parameters())
                assert all(p.grad is None for p in model.reconstruction.parameters())
                frontend_has_gradient = any(p.grad is not None and p.grad.count_nonzero()
                                            for p in model.encoder.pretrained_parameters())
                assert bool(frontend_has_gradient) == (stage == "finetune")
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                assert torch.isfinite(gradient_norm)
                optimizer.step()
                model.update_teacher()
                assert not torch.equal(projection, model.encoder.projection.weight)
                assert not torch.equal(head, model.pcr_head[-1].weight)
                changed = any(not torch.equal(before[name], p.detach().cpu())
                              for name, p in model.encoder.frontend.named_parameters())
                assert changed == (stage == "finetune")
                if stage == "frozen":
                    assert not model.encoder.frontend.training
                    assert all(torch.equal(before[name], p.detach().cpu())
                               for name, p in model.teacher_encoder.frontend.named_parameters())
                assert all(torch.isfinite(p).all() for p in model.parameters())
                # Validation uses FP32, including the largest legal prefix.
                if landmark == 3:
                    model.eval()
                    with torch.no_grad():
                        logits = model(inp)["pcr_logit"]
                    assert logits.shape == (16,) and torch.isfinite(logits).all()
                    case["fp32_validation_passed"] = True
                torch.cuda.synchronize()
                case.update(passed=True, observed_visits=16 * (landmark + 1),
                            future_visits=0, losses={k: float(v.detach()) for k, v in terms.items()},
                            gradient_norm=float(gradient_norm), learning_rates=_learning_rates(optimizer),
                            frontend_updated=changed, adapter_updated=True, pcr_head_updated=True,
                            teacher_no_grad=True, **calls)
                print(f"{stage} T{landmark}: passed", flush=True)
            except Exception as exc:
                case.update(error_type=type(exc).__name__, error=str(exc))
                print(f"{stage} T{landmark}: {type(exc).__name__}: {exc}", flush=True)
            finally:
                case.update(seconds=time.monotonic() - started,
                            peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024 ** 2,
                            peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024 ** 2)
                report["cases"].append(case)
                write_json(report_path, report)
                # Hooks close over only counters; release all graph references.
                del model, optimizer, inp, sup, loss, terms, gradient_norm
                if "gradients" in locals():
                    del gradients
                if "logits" in locals():
                    del logits
                if "projection" in locals():
                    del projection, head
                gc.collect()
                torch.cuda.empty_cache()
            if not case["passed"]:
                return report
    # Epoch-5 probes use autograd.grad and FP32 teacher features; measure them too.
    probe_indices = [i for i in store.by_split["train"]
                     if all(visit is not None for visit in store.patients[i]["visits"])][:32]
    if len(probe_indices) != 32:
        raise AssertionError("Need thirty-two complete patients for worst-case probes")
    for stage in ("frozen", "finetune"):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cfg = load_config(ROOT / f"configs/vq_{stage}_dynamic_seed17.yaml")
        model = RawVLAJEPA(cfg.model).to(cfg.device)
        model.initialize_pretrained_encoder(CODEC)
        _configure(model, cfg, "joint")
        probe, private = run_training_probe(model, store, cfg, 5,
                                           [(i, 3) for i in probe_indices])
        assert probe["teacher_frozen"] and probe["rng_and_module_modes_preserved"]
        assert all(batch["all_gradient_values_finite"] for batch in probe["gradient_batches"])
        assert all(p.grad is None for p in model.parameters())
        assert not model.teacher_encoder.training
        assert model.encoder.frontend.training == (stage == "finetune")
        report["diagnostics"].append({"stage": stage, "passed": True,
            "patients": 32, "batch_size": 16, "landmark": 3,
            "teacher_no_grad": True, "modes_restored": True,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024 ** 2})
        print(f"{stage} worst-case epoch probe: passed", flush=True)
        del model, probe, private
        gc.collect()
        torch.cuda.empty_cache()
        write_json(report_path, report)
    report["passed"] = True
    write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/vq_encoder_cuda_20261012.json")
    report = check(parser.parse_args().output.resolve())
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

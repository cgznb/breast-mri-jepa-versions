#!/usr/bin/env python3
"""Real B16 engineering gates before the sequential V1 evolution experiment."""
from dataclasses import replace
from pathlib import Path
import argparse
import runpy
import gc
import time

import torch

from mri_vla_jepa.data import RawMRIStore
from mri_vla_jepa.diagnostics import aggregate_world_errors, run_training_probe, world_error_rows
from mri_vla_jepa.io import autocast, seed_all, write_json, read_json
from mri_vla_jepa.model import RawVLAJEPA
from mri_vla_jepa.train_config import load_config
from mri_vla_jepa.training import _configure, _make_optimizer

ROOT = Path(__file__).resolve().parents[1]
_release = runpy.run_path(str(ROOT / "scripts/release_runtime.py"))
ORIGINAL = _release["version_directory"](ROOT, "v1")


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def run(output, diagnostics_only=False):
    paths = sorted(ROOT.glob("configs/evolution_*seed17.yaml"))
    if len(paths) != 10:
        raise ValueError(f"Expected ten seed17 controlled configurations; found {len(paths)}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The real B16 gates require CUDA BF16")
    report = {"schema": "v1_evolution_cuda_gate_v1", "passed": False,
              "engineering_only": True, "physical_batch_size": 16, "accumulation": 1,
              "patient_rows_saved": False, "cases": [], "diagnostics": []}
    if diagnostics_only:
        report = read_json(output)
        if len(report["cases"]) != 30 or not all(case["passed"] for case in report["cases"]):
            raise ValueError("Diagnostic-only verification requires all thirty existing train gates to pass")
        report["diagnostics"] = []
    for path in ([] if diagnostics_only else paths):
        cfg = load_config(path)
        seed_all(cfg.seed, threads=cfg.threads)
        store = RawMRIStore(ORIGINAL / "data/registered_roi32/manifest.json", cfg.image_shape,
                            image_cache=cfg.image_cache,
                            require_lesion_weights=cfg.model.lesion_world_weighting)
        complete = [i for i in store.by_split["train"]
                    if all(visit is not None for visit in store.patients[i]["visits"])][:32]
        assert len(complete) == 32
        for landmark in (0, 2, 3):
            cleanup()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            case = {"config": path.name, "landmark": landmark, "passed": False}
            model = optimizer = inp = sup = loss = terms = None
            try:
                model = RawVLAJEPA(cfg.model).to(cfg.device)
                if cfg.model.encoder_type != "raw":
                    model.initialize_pretrained_encoder(cfg.encoder_checkpoint)
                _configure(model, cfg, "joint")
                optimizer = _make_optimizer(model, cfg)
                frontend = ({name: p.detach().clone() for name, p in model.encoder.frontend.named_parameters()}
                            if cfg.model.encoder_type != "raw" else None)
                inp, sup = store.batch([(i, landmark) for i in complete[:16]], future_supervision=True)
                inp, sup = inp.to(cfg.device), sup.to(cfg.device)
                with autocast(cfg.device, cfg.precision):
                    loss, terms = model.compute_loss(inp, sup, task_weight=cfg.task_weight,
                        jepa_weight=cfg.jepa_weight, rollout_weight=cfg.rollout_weight,
                        flow_weight=0., reconstruction_weight=0., variance_weight=0.,
                        variance_definition="off")
                assert all(torch.isfinite(value).all() for value in terms.values())
                assert int(terms["pair_count"]) == 16 * (3 - landmark)
                assert int(terms["rollout_count"]) == (16 * (3 - landmark) if cfg.rollout_weight else 0)
                task_grads = torch.autograd.grad(terms["task"], tuple(model.world_predictor.parameters()),
                                                retain_graph=True, allow_unused=True)
                task_reaches_world = any(g is not None and bool(g.count_nonzero()) for g in task_grads)
                assert task_reaches_world == (cfg.model.pcr_readout == "forecast" and landmark < 3)
                del task_grads
                loss.backward()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                assert all(not p.requires_grad and p.grad is None for p in model.teacher_encoder.parameters())
                assert any(p.grad is not None and bool(p.grad.count_nonzero()) for p in model.pcr_head.parameters())
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                assert torch.isfinite(norm)
                optimizer.step()
                model.update_teacher()
                if frontend is not None:
                    assert all(torch.equal(frontend[name], p) for name, p in model.encoder.frontend.named_parameters())
                    assert all(torch.equal(frontend[name], p)
                               for name, p in model.teacher_encoder.frontend.named_parameters())
                assert all(torch.isfinite(p).all() for p in model.parameters())
                case.update(losses={key: float(value.detach()) for key, value in terms.items()},
                            teacher_no_grad=True, frozen_frontend_unchanged=frontend is not None,
                            task_gradient_reaches_world=task_reaches_world, gradient_norm=float(norm))
                del frontend
                model.eval()
                if landmark == 0:
                    with torch.no_grad():
                        full = model(inp)["pcr_logit"]
                        only_t3 = torch.zeros_like(inp.query_mask)
                        only_t3[:, 3] = True
                        none = torch.zeros_like(inp.query_mask)
                        assert torch.equal(full, model(replace(inp, query_mask=only_t3))["pcr_logit"])
                        assert torch.equal(full, model(replace(inp, query_mask=none))["pcr_logit"])
                        first = model.forecast_states(inp)[:, 3]
                        assert torch.equal(first, model.forecast_states(replace(inp, query_mask=only_t3))[:, 3])
                    case["request_invariance_passed"] = True
                    rows, teacher, known = world_error_rows(model, inp, sup, complete[:16])
                    world = aggregate_world_errors(rows)
                    assert world["status"] == "evaluated"
                    assert all(world["metrics"][f"horizon_{h}"]["patients"] == 16 for h in (1, 2, 3))
                    case["fp32_tf_free_copy_passed"] = True
                    del full, first, rows, teacher, known, world
                case["passed"] = True
                print(f"{path.name} T{landmark}: passed", flush=True)
            except Exception as exc:
                case.update(error_type=type(exc).__name__, error=str(exc))
                print(f"{path.name} T{landmark}: {type(exc).__name__}: {exc}", flush=True)
            finally:
                torch.cuda.synchronize()
                case.update(seconds=time.monotonic() - started,
                            peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                            peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2)
                report["cases"].append(case)
                write_json(output, report)
                del model, optimizer, inp, sup, loss, terms
                cleanup()
            if not case["passed"]:
                return report
    # The highest-complexity loss paths must also support the normal diagnostic hook.
    for path in paths:
        cfg = load_config(path)
        if not cfg.rollout_weight or not (cfg.model.lesion_world_weighting or cfg.model.encoder_type == "vq_encoder"):
            continue
        cleanup()
        torch.cuda.reset_peak_memory_stats()
        store = RawMRIStore(ORIGINAL / "data/registered_roi32/manifest.json", cfg.image_shape,
                            image_cache=cfg.image_cache,
                            require_lesion_weights=cfg.model.lesion_world_weighting)
        complete = [i for i in store.by_split["train"]
                    if all(v is not None for v in store.patients[i]["visits"])][:32]
        model = RawVLAJEPA(cfg.model).to(cfg.device)
        model.initialize_pretrained_encoder(cfg.encoder_checkpoint)
        _configure(model, cfg, "joint")
        probe, _ = run_training_probe(model, store, cfg, epoch=0,
            tasks=tuple((i, k % 4) for k, i in enumerate(complete)))
        assert probe["rng_and_module_modes_preserved"] and probe["teacher_frozen"]
        assert all(batch["all_gradient_values_finite"] for batch in probe["gradient_batches"])
        report["diagnostics"].append({"config": path.name, "passed": True,
            "peak_reserved_mib": torch.cuda.max_memory_reserved()/1024**2})
        del model, probe
        cleanup()
        write_json(output, report)
    report["passed"] = True
    write_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/evolution_cuda_20261012.json")
    parser.add_argument("--diagnostics-only", action="store_true")
    args = parser.parse_args()
    result = run(args.output, args.diagnostics_only)
    raise SystemExit(0 if result["passed"] else 1)

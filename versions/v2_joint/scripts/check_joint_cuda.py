"""Read real cached training batches for capacity gates, with disposable weights."""
from pathlib import Path
import gc
import json
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from response_jepa.config import load_config
from response_jepa.data import Store
from response_jepa.engine import autocast, make_loader, optimizer_for, setup
from response_jepa.model import ResponseJEPA
from response_jepa.objectives import joint_loss


def case(profile, stage):
    cfg = load_config(ROOT / "configs" / f"joint_{profile}_batch16_seed17.yaml")
    setup(cfg)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    store = Store(
        "/root/autodl-tmp/line b/breast-mri-vla-jepa/data/registered_roi32/manifest.json",
        cfg.train.image_shape, image_cache=cfg.train.image_cache,
    )
    eligible = [i for i in store.splits["train"] if stage in store.allowed(i, profile)]
    tasks = [(i, stage) for i in eligible[:16]]
    assert len(tasks) == 16
    batch = next(iter(make_loader(store, cfg, tasks, training=True, world=True))).to("cuda")
    assert batch.inputs.anchor().unique().numel() == 1 and len(batch.keys) == 16
    model = ResponseJEPA(cfg.model).cuda()
    model.configure_trainable("joint", False)
    optimizer = optimizer_for(model, cfg)
    teacher_before = {key: value.detach().clone() for key, value in model.target_encoder.named_parameters()}
    with autocast(cfg):
        loss, terms, out = joint_loss(model, batch, cfg, augment=cfg.train.augment)
    assert torch.isfinite(loss)
    loss.backward()
    branches = {name: any(p.grad is not None and bool(p.grad.count_nonzero()) for p in module.parameters())
                for name, module in (
                    ("encoder", model.encoder), ("fusion", model.fusion),
                    ("world_output", model.world.output), ("pcr_direct", model.direct_head),
                    ("pcr_forecast", model.terminal_head), ("mask_predictor", model.mask_predictor),
                    ("representation_projector", model.repr_projector),
                )}
    assert all(branches[name] for name in branches if name != "world_output" or stage < 3)
    assert all(p.grad is None for p in model.target_encoder.parameters())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad, error_if_nonfinite=True)
    optimizer.step()
    model.update_teacher(cfg.train.ema)
    online = dict(model.encoder.named_parameters())
    for key, teacher in model.target_encoder.named_parameters():
        expected = torch.lerp(teacher_before[key], online[key].detach(), 1 - cfg.train.ema)
        torch.testing.assert_close(teacher, expected, atol=0, rtol=0)
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
    torch.cuda.synchronize()
    result = {
        "profile": profile, "stage": stage, "physical_batch": 16,
        "legal_input_only": True, "loss_finite": True, "gradient_finite": bool(torch.isfinite(norm)),
        "branch_has_gradient": branches, "teacher_no_gradient": True, "teacher_ema_exact": True,
        "parameters_finite_after_one_update": True,
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "seconds": round(time.monotonic() - started, 2),
    }
    del model, optimizer, batch, loss, terms, out, teacher_before, online
    gc.collect()
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    report = {"scope": "one real training batch per stage with disposable random weights; no run/checkpoint saved", "cases": []}
    for profile, stage in [("t0", 0), ("dynamic", 1), ("dynamic", 2), ("dynamic", 3)]:
        result = case(profile, stage)
        report["cases"].append(result)
        print(json.dumps(result), flush=True)
    target = ROOT / "reports" / "joint_cuda_batch16_20261009.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print("Saved aggregate capacity report:", target, flush=True)

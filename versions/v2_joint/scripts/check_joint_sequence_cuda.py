"""Real multi-update and FP32-evaluation capacity gates; never save weights."""
from pathlib import Path
import gc
import json
import math
import os
import subprocess
import sys
import threading
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from response_jepa.config import load_config
from response_jepa.data import Store, planned_tasks
from response_jepa.engine import (
    autocast, evaluate, make_loader, optimizer_for, schedule_lr, setup,
)
from response_jepa.model import ResponseJEPA
from response_jepa.objectives import joint_loss

MANIFEST = "/root/autodl-tmp/line b/breast-mri-vla-jepa/data/registered_roi32/manifest.json"


def own_gpu_mib():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    for row in result.stdout.splitlines():
        fields = [item.strip() for item in row.split(",")]
        if len(fields) == 2 and fields[0] == str(os.getpid()):
            return float(fields[1]) if fields[1].replace(".", "", 1).isdigit() else None
    return None


class Occupancy:
    """Sample only this process; NVIDIA peak is a sampled lower bound."""
    def __init__(self):
        self.values = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def poll(self):
        while not self.stop.is_set():
            self.read()
            self.stop.wait(.5)

    def read(self):
        value = own_gpu_mib()
        if value is not None:
            self.values.append(value)
        return value

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=15)


def memory(occupancy):
    torch.cuda.synchronize()
    return {
        "allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
        "reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 1),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "nvidia_own_process_mib": occupancy.read(),
    }


def case(profile, stages):
    filename = ("joint_t0_batch16_seed17.yaml" if profile == "t0" else
                "joint_dynamic_all_prefixes_batch16_seed17.yaml")
    cfg = load_config(ROOT / "configs" / filename)
    setup(cfg)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    store = Store(MANIFEST, cfg.train.image_shape, image_cache=cfg.train.image_cache)
    # Complete cases exercise the maximal legal image history/target count.
    complete_train = [i for i in store.splits["train"]
                      if store.allowed(i, "dynamic") == [0, 1, 2, 3]][:16]
    complete_val = [i for i in store.splits["val"]
                    if store.allowed(i, "dynamic") == [0, 1, 2, 3]][:16]
    assert len(complete_train) == len(complete_val) == 16
    for i in complete_train + complete_val:
        assert all(store.patients[i]["visits"][j] is not None
                   and store.patients[i]["visits"][j]["available_at"] <= j for j in range(4))
    model = ResponseJEPA(cfg.model).cuda()
    model.configure_trainable("joint", False)
    model.world_trained = True
    optimizer = optimizer_for(model, cfg)
    actual_loader = make_loader(store, cfg, planned_tasks(store, cfg, "train", 0),
                                training=True, world=True)
    groups = math.ceil(len(actual_loader) / cfg.train.accumulation)
    del actual_loader
    results = []
    with Occupancy() as occupancy:
        for step, stage in enumerate(stages):
            model.train()
            loader = make_loader(store, cfg, [(i, stage) for i in complete_train],
                                 training=True, world=True)
            cpu_batch = next(iter(loader))
            optimizer.zero_grad(set_to_none=True)
            schedule_lr(optimizer, cfg, (step + 1) / groups)
            batch = cpu_batch.to("cuda", non_blocking=cfg.train.pin_memory)
            batch.inputs.validate()
            batch.targets.validate(batch.inputs)
            assert batch.inputs.anchor().unique().tolist() == [stage]
            assert len(set(batch.keys)) == len(batch.keys) == 16
            with autocast(cfg):
                loss, terms, prediction = joint_loss(model, batch, cfg, augment=cfg.train.augment)
            assert bool(torch.isfinite(loss))
            loss.backward()
            for param in model.parameters():
                if param.grad is not None:
                    assert bool(torch.isfinite(param.grad).all())
            assert all(p.grad is None for p in model.target_encoder.parameters())
            loss_value = float(loss.detach())
            term_values = {key: float(value.detach() if isinstance(value, torch.Tensor) else value)
                           for key, value in terms.items()}
            assert all(math.isfinite(value) for value in term_values.values())
            # Match the release point in the actual training engine.
            del batch, loss, terms, prediction
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad,
                                                 error_if_nonfinite=True)
            optimizer.step()
            model.update_teacher(cfg.train.ema)
            assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
            assert all(p.grad is None for p in model.target_encoder.parameters())
            item = {"step": step + 1, "stage": stage, "physical_batch": 16,
                    "loss": loss_value, "gradient_norm": float(norm),
                    "loss_terms_finite": True, "gradients_finite": True,
                    "parameters_finite": True, "teacher_no_gradient": True,
                    "legal_input_and_strict_future": True, "memory": memory(occupancy)}
            results.append(item)
            print(json.dumps({"profile": profile, "training_step": item}), flush=True)
            del loader, cpu_batch, norm
        training_peak = memory(occupancy)
        # Keep optimizer/gradients alive, matching engine transition into eval.
        torch.cuda.reset_peak_memory_stats()
        landmarks = [0] if profile == "t0" else [0, 1, 2, 3]
        val_tasks = [(i, landmark) for i in complete_val for landmark in landmarks]
        metrics, rows, world_rows = evaluate(model, store, cfg, "val", tasks=val_tasks)
        assert len(rows) == len(val_tasks)
        assert math.isfinite(metrics["selection"])
        assert not model.training and not model.target_encoder.training
        evaluation_memory = memory(occupancy)
        evaluation = {"precision": "fp32", "eval_batch_size": cfg.train.eval_batch_size,
                      "patients": 16, "prefixes": len(rows), "world_targets": len(world_rows),
                      "selection_finite": True, "teacher_eval": True,
                      "memory": evaluation_memory}
        result = {"profile": profile, "sampling": cfg.train.prefix_sampling,
                  "training_steps": results, "training_peak": training_peak,
                  "evaluation": evaluation,
                  "nvidia_own_process_sampled_peak_mib": max(occupancy.values, default=None),
                  "nvidia_samples": len(occupancy.values),
                  "seconds": round(time.monotonic() - started, 2)}
        print(json.dumps({"profile": profile, "evaluation": evaluation}), flush=True)
    del model, optimizer, store, metrics, rows, world_rows, complete_train, complete_val
    gc.collect()
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    report = {
        "scope": "real cached MRI multi-update capacity and partial-validation gate; disposable random weights",
        "formal_training": False, "weights_saved": False,
        "nvidia_peak_definition": "own process sampled every 0.5 seconds plus completed-step samples; lower bound",
        "cases": [],
    }
    try:
        for profile, stages in [("t0", [0, 0, 0]), ("dynamic", [0, 1, 2, 3, 3])]:
            report["cases"].append(case(profile, stages))
        report["all_checks_passed"] = True
        target = ROOT / "reports" / "joint_sequence_cuda_20261009.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("Saved aggregate capacity report:", target, flush=True)
    finally:
        gc.collect()
        torch.cuda.empty_cache()

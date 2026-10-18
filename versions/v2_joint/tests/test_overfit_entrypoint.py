"""Exercise the process boundary that previously failed after training ended."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from response_jepa.engine import load_checkpoint
from response_jepa.provenance import source_snapshot
from response_jepa.synthetic import small_config


PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("overfit_entry_queue", PROJECT / "scripts/run_overfit_suite.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


@pytest.mark.parametrize("updates", [True, False], ids=["updates", "epochs"])
def test_external_training_process_finishes_with_matching_summary(cohort, tmp_path, updates):
    cfg = small_config("joint")
    cfg.train.epochs = 1
    cfg.train.train_eval_patients = 0
    cfg.train.save_predictions = True
    if updates:
        cfg.train.prefix_sampling = "all_prefixes"
        cfg.train.sample_weighting = "patient_equal"
        cfg.train.reference_batch_size = cfg.train.batch_size = 3
        cfg.train.max_updates, cfg.train.warmup_updates = 2, 1
        cfg.train.eval_every_updates, cfg.train.patience_updates = 1, 10
        cfg.model.forecast_detach = True
        cfg.model.dropout = .1
        cfg.loss.consistency = .1
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    output = tmp_path / "output"
    source = source_snapshot()
    environment = queue.training_environment()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    command = queue.trainer_command(config, cohort, output)
    assert command[1] == str(PROJECT / "scripts/train_overfit.py")
    result = subprocess.run(command, cwd=PROJECT, env=environment, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout.splitlines()[-1])
    best = load_checkpoint(output / "best.pt")
    last = load_checkpoint(output / "last.pt")
    assert summary["output"] == str(output.resolve())
    assert source_snapshot() == source == best["source_identity"] == last["source_identity"]
    assert (output / "predictions_val.csv").is_file()
    if updates:
        completion = json.loads((output / "completion.json").read_text())
        assert completion["state"] == "completed"
        assert summary["updates"] == completion["updates"] == last["updates"] == 2
        assert summary["best_update"] == completion["best_update"] == best["updates"]
        assert summary["stop_reason"] == "max_updates"
        assert "best_epoch" not in summary
    else:
        assert summary["best_epoch"] == best["epoch"]
        assert "best_update" not in summary


def test_queue_resume_command_keeps_exact_entrypoint_and_identity():
    initial = queue.trainer_command("config", "manifest", "output")
    resumed = queue.trainer_command("config", "manifest", "output", True)
    assert resumed == initial + ["--resume"]
    assert "response_jepa.cli" not in resumed

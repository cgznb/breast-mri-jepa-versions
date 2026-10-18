"""Readout isolation, equal task strength, immutable protocol and strict replay."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from response_jepa.config import load_config
from response_jepa.engine import train
from response_jepa.model import ResponseJEPA
from response_jepa.metrics import task_summary
from response_jepa.objectives import task_world_loss
from response_jepa.provenance import source_snapshot
from response_jepa.synthetic import small_config

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import run_branch_ablation as branch
import analyze_branch_ablation as analysis


def baseline_config():
    cfg = small_config("joint")
    t = cfg.train
    t.seed, t.max_updates, t.warmup_updates = 17, 3000, 200
    t.eval_every_updates, t.patience_updates = 100, 500
    t.prefix_sampling, t.reference_batch_size = "all_prefixes", t.batch_size
    t.train_eval_patients = 0
    t.save_predictions, t.world_eval = True, True
    return cfg


def save_config(path, cfg):
    path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    return path


@pytest.fixture
def references(tmp_path, cohort):
    root = tmp_path / "references"
    root.mkdir()
    runs = []
    for profile in branch.PROFILES:
        for seed in branch.SEEDS:
            cfg = baseline_config()
            cfg.train.profile, cfg.train.seed = profile, seed
            cfg.train.prefix_sampling = "all_prefixes" if profile == "dynamic" else "balanced_cycle"
            name = f"B0_{profile}_seed{seed}"
            path = save_config(root / f"{name}.yaml", cfg)
            runs.append({"name": name, "algorithm": "B0", "profile": profile, "seed": seed,
                         "config": str(path), "manifest": str(cohort), "output": str(root / name),
                         "reservation_mib": 100})
    (root / "experiment_plan.json").write_text(json.dumps({"runs": runs}))
    return root


def mock_reference_audit(monkeypatch):
    def audit(run, *, current_source):
        assert current_source == source_snapshot()
        rows = [{"patient_key": str(label), "landmark": 0, "label": label,
                 **{key: .8 if label else .2 for key in ("direct", "forecast", "fused")}}
                for label in (0, 1)]
        classification = task_summary(rows)
        return {"profile": run["profile"], "seed": run["seed"], "best_update": 100,
                "selection": classification["fused"]["patient_equal"]["nll"],
                "classification": classification, "checks": {"reference_fixture": True}}
    monkeypatch.setattr(analysis.replay, "audit_completed", audit)


@pytest.mark.parametrize("variant", branch.VARIANTS)
def test_readout_is_the_only_structural_change_and_task_strength_is_equal(tmp_path, variant):
    base = baseline_config()
    path = save_config(tmp_path / "base.yaml", base)
    cfg = branch.ablation_config(path, variant)
    assert cfg.model.readout == branch.VARIANTS[variant]
    assert (cfg.loss.fused, cfg.loss.direct, cfg.loss.forecast) == (1, 0, 0)
    actual = cfg.to_dict()
    expected = base.to_dict()
    expected["model"]["readout"] = branch.VARIANTS[variant]
    expected["loss"].update(fused=1., direct=0., forecast=0.)
    assert actual == expected


@pytest.mark.parametrize("variant", branch.VARIANTS)
def test_only_selected_readout_receives_task_gradient(batch, tmp_path, variant):
    cfg = branch.ablation_config(save_config(tmp_path / "base.yaml", baseline_config()), variant)
    cfg.loss.tf = cfg.loss.rollout = cfg.loss.align = 0
    model = ResponseJEPA(cfg.model).eval()
    model.configure_trainable("joint", False)
    loss, terms, out = task_world_loss(model, batch, cfg)
    torch.testing.assert_close(loss, terms["fused"])
    if variant == "D_direct":
        torch.testing.assert_close(out["logit"], out["direct_logit"])
        assert out["forecast_logit"] is None and out["gate"] is None
    elif variant == "F_forecast":
        torch.testing.assert_close(out["logit"], out["forecast_logit"])
        assert out["gate"] is None
    loss.backward()
    def norm(module):
        return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
    assert (norm(model.direct_head) > 0) == (variant != "F_forecast")
    assert (norm(model.terminal_head) > 0) == (variant != "D_direct")
    assert (norm(model.gate) > 0) == (variant == "G_gated")
    assert all(p.grad is None for p in model.target_encoder.parameters())


def test_plan_has_all18_and_rejects_changed_frozen_config(tmp_path, references):
    output = tmp_path / "branch"
    plan = branch.prepare_plan(output, references)
    data = json.loads(plan.read_text())
    assert len(data["runs"]) == len(data["variants"]) == 18
    assert len(data["baseline_references"]) == 6
    assert len({r["name"] for r in data["runs"]}) == 18
    assert branch.prepare_plan(output, references) == plan
    first = Path(data["runs"][0]["config"])
    first.write_text(first.read_text() + "\n# external edit\n")
    with pytest.raises(ValueError, match="differs"):
        branch.prepare_plan(output, references)


@pytest.mark.parametrize("variant", branch.VARIANTS)
def test_actual_two_update_training_replays_and_does_not_change_package(cohort, tmp_path, variant):
    cfg = branch.ablation_config(save_config(tmp_path / "base.yaml", baseline_config()), variant)
    cfg.train.max_updates, cfg.train.warmup_updates = 2, 1
    cfg.train.eval_every_updates, cfg.train.patience_updates = 1, 10
    config = save_config(tmp_path / "train.yaml", cfg)
    source = source_snapshot()
    output = tmp_path / "trained"
    summary = train(cfg, cohort, output)
    assert summary["state"] == "completed" and summary["updates"] == 2
    run = {"name": variant, "config": str(config), "manifest": str(cohort),
           "output": str(output), "algorithm": "B0", "profile": "dynamic", "seed": 17}
    verified = analysis.replay.audit_completed(run, current_source=source)
    assert all(verified["checks"].values())
    assert source_snapshot() == source
    if variant in {"D_direct", "F_forecast"}:
        key = "direct" if variant == "D_direct" else "forecast"
        assert verified["classification"]["fused"] == verified["classification"][key]


def test_partial_report_replays_references_and_rejects_confounded_loss(tmp_path, references, monkeypatch):
    mock_reference_audit(monkeypatch)
    plan = branch.prepare_plan(tmp_path / "plan", references)
    report = analysis.analyze(plan)
    assert not report["complete"] and report["errors"] == []
    assert len(report["original_references"]) == 6
    assert report["verified_completed_runs"] == 0
    dynamic_forecast = next(r for r in report["original_branch_nll"]
                            if r["profile"] == "dynamic" and r["branch"] == "forecast")
    assert dynamic_forecast["seeds"] == 3
    assert dynamic_forecast["mean"] == pytest.approx(-torch.log(torch.tensor(.8)).item())
    first = Path(json.loads(plan.read_text())["runs"][0]["config"])
    cfg = load_config(first)
    cfg.loss.direct = .5
    save_config(first, cfg)
    with pytest.raises(ValueError, match="locked loss controls"):
        analysis.analyze(plan)


def test_branch_finalizer_report_renders_without_unfinished_primary_rows(tmp_path, references, monkeypatch):
    mock_reference_audit(monkeypatch)
    plan = branch.prepare_plan(tmp_path / "plan", references)
    report = analysis.analyze(plan)
    out = tmp_path / "report"
    analysis.write_report(report, out)
    assert json.loads((out / "aggregate.json").read_text())["complete"] is False
    text = (out / "README_ZH.md").read_text()
    assert "0/18" in text and "不是分别重训结果" in text
    assert (out / "original_branch_stage_aggregate.csv").is_file()


def test_controller_rejoins_live_queue_then_analyzes_without_duplicate_launch(tmp_path, monkeypatch):
    output = tmp_path / "suite"
    plan = output / "experiment_plan.json"
    statuses = iter([{"controller": {"pid": 42, "start_ticks": "born"}, "runs": [{"state": "running"}]},
                     {"runs": [{"state": "completed"}]}])
    monkeypatch.setattr(branch, "prepare_plan", lambda output: plan)
    monkeypatch.setattr(branch, "process_info", lambda pid: {"start_ticks": "born"})
    monkeypatch.setattr(branch, "matches_process", lambda *a: True)
    monkeypatch.setattr(branch, "read_json", lambda p: next(statuses) if p.name == "status.json"
                        else {"complete": True, "errors": []} if p.name == "aggregate.json" else None)
    launched, waits = [], []
    monkeypatch.setattr(branch, "child", lambda command, log: launched.append(command))
    monkeypatch.setattr(branch.time, "sleep", lambda seconds: waits.append(seconds))
    assert branch.controller(SimpleNamespace(output=output, budget=31232)) == 0
    assert waits == [10] and len(launched) == 1
    assert launched[0][1].endswith("analyze_branch_ablation.py")
    assert json.loads((output / "research_status.json").read_text())["state"] == "completed"


@pytest.mark.parametrize("failure", ["training", "analysis"])
def test_controller_does_not_claim_success_after_failure(tmp_path, monkeypatch, failure):
    output = tmp_path / "suite"
    monkeypatch.setattr(branch, "prepare_plan", lambda output: output / "experiment_plan.json")
    monkeypatch.setattr(branch, "process_info", lambda pid: {})
    monkeypatch.setattr(branch, "read_json", lambda p: {"runs": [{"state": "failed" if failure == "training" else "completed"}]}
                        if p.name == "status.json" else {"complete": False, "errors": ["bad source"]}
                        if p.name == "aggregate.json" else None)
    monkeypatch.setattr(branch, "child", lambda *a: None)
    with pytest.raises(RuntimeError):
        branch.controller(SimpleNamespace(output=output, budget=31232))
    assert json.loads((output / "research_status.json").read_text())["state"] == "failed"

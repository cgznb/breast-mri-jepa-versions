"""Ordered design gates, fresh-init controls, and non-hardcoded aggregation."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mri_vla_jepa.io import read_json, write_json
from mri_vla_jepa.train_config import from_dict, load_config


ROOT = Path(__file__).resolve().parents[1]


def load_script(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


launcher = load_script("run_evolution_experiment.py", "evolution_launcher_test")
analyzer = load_script("analyze_evolution_experiment.py", "evolution_analyzer_test")


def test_twenty_fresh_runs_keep_matching_protocol_and_isolate_lesion_from_encoder64():
    planned = launcher.jobs(launcher.DEFAULT_OUTPUT)
    assert len(planned) == 20 and len({job["name"] for job in planned}) == 20
    assert [job["stage"] for job in planned] == sorted(job["stage"] for job in planned)
    index = {job["variant"]: load_config(job["config"]) for job in planned if job["seed"] == 17}
    for job in planned:
        cfg = from_dict(launcher._release["portable_config"](load_config(job["config"]).to_dict(), ROOT))
        assert cfg.model.fixed_transition_queries and not cfg.model.lesion_pooling
        assert cfg.jepa_weight == .5 and cfg.task_weight == 1
        assert cfg.variance_weight == cfg.flow_weight == cfg.reconstruction_weight == 0
        assert cfg.variance_definition == "off" and cfg.model.dropout == .3
        assert (cfg.batch_size, cfg.accumulation, cfg.max_epochs, cfg.early_stopping_patience) == (16, 1, 200, 50)
        assert cfg.lr == 1e-4 and cfg.initialize_from_checkpoint is cfg.encoder_lr is None
        assert cfg.seed == job["seed"] and cfg.landmarks == "all_observed"
        assert job["need_world_evaluation"] and job["require_world_head_trained"]
        if job["variant"].startswith("vq_"):
            assert cfg.encoder_training == "frozen" and cfg.encoder_checkpoint == str(launcher.CODEC)
    for name in ("vq_copy", "vq_forecast_detach", "vq_forecast"):
        assert index[name].model.pcr_readout == name.removeprefix("vq_")
    base = index["vq_forecast_history_residual_free"].to_dict()
    lesion = index["vq_forecast_history_residual_free_lesion"].to_dict()
    encoder = index["vq_forecast_history_residual_free_encoder64"].to_dict()
    lesion["model"]["lesion_world_weighting"] = False
    encoder["model"]["encoder_type"] = "vq_prequant"
    assert base == lesion == encoder
    assert base["rollout_weight"] == .5
    # Every dependency is an earlier design step and includes both seeds.
    by_name = {job["name"]: job for job in planned}
    for job in planned:
        for dependency in job["depends_on"] or []:
            assert by_name[dependency]["step"] != job["step"]
            assert planned.index(by_name[dependency]) < planned.index(job)


def test_standalone_training_entrypoint_runs_real_cpu_help():
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="4")
    result = subprocess.run([sys.executable, str(ROOT / "train.py"), "train-vla-jepa", "--help"],
                            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout and "--manifest" in result.stdout and "--output" in result.stdout


def test_missing_training_entrypoint_rejected_before_planning(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing standalone training entry point"):
        launcher.jobs(tmp_path / "runs")


@pytest.mark.parametrize("field", ["rollout_weight", "initialize_from_checkpoint"])
def test_unplanned_loss_or_warm_start_rejected(monkeypatch, field):
    original = launcher.load_config

    def changed(path):
        cfg = original(path)
        if Path(path).name.startswith("evolution_"):
            setattr(cfg, field, .123 if field == "rollout_weight" else "unplanned.pt")
        return cfg

    monkeypatch.setattr(launcher, "load_config", changed)
    with pytest.raises(ValueError, match="fresh-init evolution protocol"):
        launcher.jobs(launcher.DEFAULT_OUTPUT)


@pytest.mark.parametrize("state", ["running", "evaluation_pending", "evaluating", "completed"])
def test_step_waits_for_all_seeds_and_both_world_reports(state):
    parents = [{"name": "a17", "status": "completed", "training_validated": True},
               {"name": "a43", "status": state, "training_validated": True}]
    child = {"name": "b17", "depends_on": ["a17", "a43"]}
    assert launcher.dependency_reason(child, parents + [child]) == (None if state == "completed" else "waiting_for_dependency")
    parents[1].update(status="completed", training_validated=False)
    assert launcher.dependency_reason(child, parents + [child]) == "waiting_for_dependency"
    parents[1].update(status="evaluation_failed", training_validated=True)
    assert launcher.dependency_reason(child, parents + [child]) == "dependency_failed"


def test_invalid_dependencies_rejected():
    child = {"name": "self", "depends_on": ["self"]}
    with pytest.raises(ValueError, match="distinct"):
        launcher.dependency_reason(child, [child])
    child["depends_on"] = ["missing"]
    with pytest.raises(ValueError, match="identify one"):
        launcher.dependency_reason(child, [child])


def test_measured_reservations_admit_only_fitting_three_worker_combinations():
    queue = launcher._load_queue()
    planned = launcher.jobs(launcher.DEFAULT_OUTPUT)
    queue.validate_memory_budget(29696, planned, launcher.SUITE)
    with pytest.raises(ValueError, match="at least 10240"):
        queue.validate_memory_budget(9216, planned, launcher.SUITE)
    assert queue.admission_reason(9216, 2, 20480, 3, 29696) is None
    assert queue.admission_reason(10240, 2, 20480, 3, 29696) == "gpu_memory_budget"
    assert queue.admission_reason(9216, 3, 27648, 3, 29696) == "concurrency_limit"
    assert all(queue.world_memory_mib(job) == 10240 for job in planned)


def mini_jobs(output, suite=launcher.SUITE):
    result = []
    for step, variant in enumerate(("raw_fixed_direct", "vq_copy", "vq_forecast_history")):
        for seed in (17, 43):
            name = f"step{step}_seed{seed}"
            result.append({"name": name, "variant": variant, "stage": step + 1, "step": str(step),
                           "profile": "dynamic", "seed": seed,
                           "config": str(ROOT / f"configs/evolution_{variant}_seed{seed}.yaml"),
                           "output": str(output / name), "status": "queued", "world_device": "cuda",
                           "need_world_evaluation": True, "require_world_head_trained": True,
                           "depends_on": [f"step{step - 1}_seed{s}" for s in (17, 43)] if step else None})
    return result


@pytest.mark.parametrize("fail_world", [False, True])
def test_three_design_steps_complete_in_order_and_resume_never_retrains(tmp_path, monkeypatch, fail_world):
    monkeypatch.setattr(launcher, "source_freeze", lambda output: {})
    monkeypatch.setattr(launcher, "_analyze", lambda *args, **kwargs: {})
    queue = launcher._load_queue()
    output = tmp_path / "runs"
    output.mkdir()
    monkeypatch.setattr(queue, "jobs", mini_jobs)
    monkeypatch.setattr(queue, "validate_completed_run", lambda job, manifest: read_json(Path(job["output"]) / "train_report.json"))
    clock, starts, events = {"tick": 0}, [], []

    class Child:
        def __init__(self, command, **kwargs):
            self.run = Path(command[command.index("--output") + 1])
            self.pid, self.started = 18000 + len(starts), clock["tick"]
            receipt = read_json(output / "controller.json")
            job = next(job for job in receipt["jobs"] if job["name"] == self.run.name)
            assert launcher.dependency_reason(job, receipt["jobs"]) is None
            starts.append(self.run.name)
            events.append(("train", self.run.name))

        def poll(self):
            if clock["tick"] - self.started < 2:
                return None
            write_json(self.run / "train_report.json", {"completed": True})
            for name in ("best.pt", "last.pt"):
                (self.run / name).write_bytes(name.encode())
            return 0

    def evaluate(job, manifest, persist):
        events.append(("world", job["name"]))
        if fail_world and job["name"] == "step0_seed17":
            raise RuntimeError("World gate failed")
        job["world_evaluation"] = {"status": "completed", "results": {
            name: {"status": "evaluated", "world_head_trained": True} for name in ("best", "last")}}

    monkeypatch.setattr(queue.subprocess, "Popen", Child)
    monkeypatch.setattr(queue, "matching_training_process", lambda pid, *args: {"start_ticks": str(pid)} if pid else None)
    monkeypatch.setattr(queue, "evaluate_completed_world", evaluate)
    monkeypatch.setattr(queue.time, "sleep", lambda seconds: clock.update(tick=clock["tick"] + 1))
    assert queue.work(output, tmp_path / "manifest.json", False, 2, launcher.SUITE) == int(fail_world)
    assert len(starts) == (2 if fail_world else 6)
    if not fail_world:
        for step in (1, 2):
            first_start = events.index(("train", f"step{step}_seed17"))
            assert all(events.index(("world", f"step{step-1}_seed{seed}")) < first_start for seed in (17, 43))
    previous_starts = starts.copy()
    assert queue.work(output, tmp_path / "manifest.json", True, 2, launcher.SUITE) == int(fail_world)
    assert starts == previous_starts


def test_source_freeze_uses_text_and_refuses_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "_source_snapshot", lambda: {"model.py": "first source text"})
    launcher.source_freeze(tmp_path)
    saved = read_json(tmp_path / "source_freeze.json")
    assert saved["runtime_source"] == {"model.py": "first source text"}
    assert "scripts/run_evolution_experiment.py" in saved["files"]
    assert "train.py" in saved["files"]
    launcher.source_freeze(tmp_path)
    monkeypatch.setattr(launcher, "_source_snapshot", lambda: {"model.py": "changed source text"})
    with pytest.raises(ValueError, match="changed after queue freeze"):
        launcher.source_freeze(tmp_path)


def test_cli_plan_and_status_are_read_only(tmp_path, monkeypatch, capsys):
    output = tmp_path / "not_started"
    monkeypatch.setattr(sys, "argv", [str(launcher.__file__), "--plan", "--output", str(output)])
    assert launcher.main() == 0 and not output.exists()
    assert '"vq_forecast"' in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", [str(launcher.__file__), "--status", "--output", str(output)])
    assert launcher.main() == 0 and not output.exists()
    assert '"not_started"' in capsys.readouterr().out


@pytest.mark.parametrize("missing_world", [False, True])
def test_analysis_counts_planned_runs_instead_of_six(tmp_path, monkeypatch, missing_world):
    planned = launcher.jobs(tmp_path)[:4]
    cfg = load_config(planned[0]["config"])
    store = SimpleNamespace(manifest={"synthetic": False})
    validators = SimpleNamespace(
        _population=lambda store: ({}, {"anonymous": {0, 1, 2, 3}}),
        _pcr=lambda *args, **kwargs: {kind: {} for kind in ("best", "last")},
        _world=lambda *args, **kwargs: {}, _means=lambda records: [], SnapshotChanged=ValueError)
    wrapper = SimpleNamespace(jobs=lambda output: planned, VARIANTS=launcher.VARIANTS,
                              SUITE=launcher.SUITE,
                              _load_queue=lambda: SimpleNamespace(validate_completed_run=lambda *args: {},
                                  prepared_jobs=lambda *args, **kwargs: planned))
    monkeypatch.setattr(analyzer, "_wrapper", lambda: wrapper)
    monkeypatch.setattr(analyzer, "_validators", lambda: validators)
    monkeypatch.setattr(analyzer, "_store", lambda *args: store)
    monkeypatch.setattr(analyzer, "_world_means", lambda records: [])
    monkeypatch.setattr(analyzer, "_paired", lambda *args: [])
    for job in planned:
        run = Path(job["output"])
        write_json(run / "train_report.json", {"completed": True})
        for name in ("best.pt", "last.pt", "best_validation.json", "epoch_history.json", "world_best.json", "world_last.json"):
            (run / name).write_text("{}")
    if missing_world:
        (Path(planned[-1]["output"]) / "world_last.json").unlink()
        with pytest.raises(ValueError, match="4 completed runs and 8"):
            analyzer.analyze(tmp_path, include_references=False)
    report = analyzer.analyze(tmp_path, require_complete=not missing_world, include_references=False)
    assert report["planned_training_runs"] == report["completed_training_runs"] == 4
    assert report["expected_world_reports"] == 8
    assert report["validated_world_reports"] == (7 if missing_world else 8)
    assert report["status"] == ("partial" if missing_world else "completed")
    text = (tmp_path / "RESULTS_ZH.md").read_text()
    assert "4/4" in text and "/8" in text

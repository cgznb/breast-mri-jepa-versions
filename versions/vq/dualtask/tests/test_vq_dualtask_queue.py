"""Dual-task queue checks without GPU execution or real training processes."""
import importlib.util
from pathlib import Path
import sys

import pytest

from mri_vla_jepa.io import read_json, write_json
from mri_vla_jepa.train_config import from_dict, load_config


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dualtask_launcher_test", ROOT / "scripts/run_vq_dualtask_experiment.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


@pytest.fixture
def queue(monkeypatch):
    # The queue unit tests never replay private artifacts or execute analysis.
    monkeypatch.setattr(wrapper, "_analyze_completed", lambda output: {})
    return wrapper._load_queue()


def test_six_dynamic_jobs_have_fresh_dual_task_configs_and_matching_dependencies():
    jobs = wrapper.jobs(wrapper.DEFAULT_OUTPUT)
    assert len(jobs) == 6
    assert {job["stage"] for job in jobs} == {"raw", "frozen", "finetune"}
    assert {job["seed"] for job in jobs} == {17, 43}
    for job in jobs:
        cfg = from_dict(wrapper._release["portable_config"](load_config(job["config"]).to_dict(), ROOT))
        assert cfg.task_weight == 1 and cfg.jepa_weight == .5
        assert cfg.flow_weight == cfg.reconstruction_weight == cfg.variance_weight == 0
        assert cfg.variance_definition == "off"
        assert cfg.landmarks == "all_observed" and cfg.model.dropout == .3
        assert (cfg.batch_size, cfg.accumulation, cfg.max_epochs, cfg.early_stopping_patience) == (16, 1, 200, 50)
        assert cfg.lr == 1e-4 and cfg.seed == job["seed"]
        assert job["need_world_evaluation"] and job["require_world_head_trained"]
        assert job["world_device"] == "cuda"
        packaged = ROOT / "src/mri_vla_jepa/resources/configs" / Path(job["config"]).name
        assert packaged.read_bytes() == Path(job["config"]).read_bytes()
        if job["stage"] == "raw":
            assert cfg.model.encoder_type == "raw"
            assert cfg.encoder_checkpoint is cfg.initialize_from_checkpoint is cfg.encoder_lr is None
        elif job["stage"] == "frozen":
            assert cfg.encoder_training == "frozen" and cfg.encoder_checkpoint == str(wrapper.CODEC)
            assert cfg.initialize_from_checkpoint is None and cfg.encoder_lr is None
        else:
            assert cfg.encoder_training == "finetune" and cfg.encoder_lr == 1e-5
            assert cfg.encoder_checkpoint is None
            assert job["depends_on"] == f"frozen_dynamic_seed{job['seed']}"
            assert cfg.initialize_from_checkpoint == str(wrapper.DEFAULT_OUTPUT / job["depends_on"] / "best.pt")
            assert "vq_encoder_20261012" not in cfg.initialize_from_checkpoint


@pytest.mark.parametrize("field", ["jepa_weight", "initialize_from_checkpoint"])
def test_launcher_rejects_changed_loss_or_old_pcr_only_warmstart(monkeypatch, field):
    original = wrapper.load_config

    def changed(path):
        cfg = original(path)
        if "finetune" in Path(path).name:
            if field == "jepa_weight":
                cfg.jepa_weight = 0
            else:
                cfg.initialize_from_checkpoint = str(ROOT.parent / "breast-mri-vla-jepa-vq-encoder/runs/vq_encoder_20261012/frozen_dynamic_seed17/best.pt")
        return cfg

    monkeypatch.setattr(wrapper, "load_config", changed)
    with pytest.raises(ValueError, match="L1 plus JEPA protocol"):
        wrapper.jobs(wrapper.DEFAULT_OUTPUT)


@pytest.mark.parametrize("status", ["evaluation_pending", "evaluating", "evaluation_failed", "completed"])
def test_validated_training_unlocks_finetune_independently_of_world_status(queue, status):
    parent = {"name": "frozen", "status": status, "training_validated": True}
    child = {"name": "finetune", "depends_on": "frozen"}
    assert queue.dependency_reason(child, [parent, child]) is None
    parent.update(status="failed", training_validated=False)
    assert queue.dependency_reason(child, [parent, child]) == "dependency_failed"
    parent["status"] = "running"
    assert queue.dependency_reason(child, [parent, child]) == "waiting_for_dependency"


def test_new_suite_budget_includes_automatic_gpu_world_workers(queue, monkeypatch):
    assert wrapper.MEMORY_MIB == {"raw": 10240, "frozen": 9216, "finetune": 13312}
    assert wrapper.WORLD_MEMORY_MIB == 10240
    monkeypatch.setattr(queue, "training_memory_mib", lambda job: 4096)
    planned = [{"need_world_evaluation": True, "world_device": "cuda"}]
    with pytest.raises(ValueError, match="individual worker"):
        queue.validate_memory_budget(queue.WORLD_MEMORY_MIB - 1, planned, wrapper.SUITE)
    queue.validate_memory_budget(29696, planned, wrapper.SUITE)
    assert queue.needs_world_evaluation({}, "losses")
    assert not queue.needs_world_evaluation({}, "baseline")


def fake_jobs(output, suite=wrapper.SUITE):
    return [{"name": f"{stage}_dynamic_seed{seed}", "stage": stage, "profile": "dynamic", "seed": seed,
             "config": str(ROOT / f"configs/vq_dualtask_{stage}_dynamic_seed{seed}.yaml"),
             "output": str(output / f"{stage}_dynamic_seed{seed}"), "status": "queued",
             "world_device": "cuda", "need_world_evaluation": True, "require_world_head_trained": True,
             "depends_on": f"frozen_dynamic_seed{seed}" if stage == "finetune" else None}
            for seed in (17, 43) for stage in ("raw", "frozen", "finetune")]


@pytest.mark.parametrize("world_failure", [False, True])
def test_training_priority_world_failures_do_not_block_b_or_retrain_on_resume(tmp_path, queue, monkeypatch, world_failure):
    output, manifest = tmp_path / "runs", tmp_path / "manifest.json"
    output.mkdir()
    monkeypatch.setattr(queue, "jobs", fake_jobs)
    monkeypatch.setattr(queue, "validate_completed_run", lambda job, manifest: read_json(Path(job["output"]) / "train_report.json"))
    clock, starts, evaluations = {"tick": 0}, [], []

    class Child:
        def __init__(self, command, **kwargs):
            self.run = Path(command[command.index("--output") + 1])
            self.pid, self.started = 9600 + len(starts), clock["tick"]
            if self.run.name.startswith("finetune"):
                seed = self.run.name.rsplit("seed", 1)[1]
                state = read_json(output / "controller.json")
                parent = next(job for job in state["jobs"] if job["name"] == f"frozen_dynamic_seed{seed}")
                assert parent["training_validated"]
                assert (Path(parent["output"]) / "best.pt").exists()
            starts.append(self.run.name)

        def poll(self):
            if clock["tick"] - self.started < 2:
                return None
            write_json(self.run / "train_report.json", {"completed": True})
            for name in ("best.pt", "last.pt"):
                (self.run / name).write_bytes(name.encode())
            return 0

    def evaluate(job, manifest, persist):
        state = read_json(output / "controller.json")
        # If a training slot is available, every remaining queued training job
        # must still be waiting for its checkpoint dependency.
        assert all(queue.dependency_reason(candidate, state["jobs"]) is not None
                   for candidate in state["jobs"] if candidate["status"] == "queued")
        evaluations.append((job["name"], len(starts)))
        if world_failure and job["name"] == "frozen_dynamic_seed17":
            raise RuntimeError("World evaluator failed after valid training")
        job["world_evaluation"] = {"status": "completed", "results": {
            name: {"status": "evaluated", "world_head_trained": True} for name in ("best", "last")}}

    monkeypatch.setattr(queue.subprocess, "Popen", Child)
    monkeypatch.setattr(queue, "matching_training_process", lambda pid, *args: {"start_ticks": str(pid)} if pid else None)
    monkeypatch.setattr(queue, "evaluate_completed_world", evaluate)
    monkeypatch.setattr(queue.time, "sleep", lambda seconds: clock.update(tick=clock["tick"] + 1))
    assert queue.work(output, manifest, False, max_concurrent=2, suite=wrapper.SUITE) == int(world_failure)
    assert len(starts) == 6 and len(set(starts)) == 6
    assert starts.index("finetune_dynamic_seed17") < evaluations[0][1]
    state = read_json(output / "controller.json")
    assert all(job["training_validated"] for job in state["jobs"])
    assert all(job["status"] == "completed" for job in state["jobs"] if job["stage"] == "finetune")
    assert len(evaluations) == 6
    # Re-evaluate completed training artifacts after a controller retry; never retrain.
    assert queue.work(output, manifest, True, max_concurrent=2, suite=wrapper.SUITE) == int(world_failure)
    assert len(starts) == 6 and len(evaluations) == 12


@pytest.mark.parametrize("trained", [True, False])
def test_best_last_world_eval_requires_trained_world_head_and_checkpoint_identity(tmp_path, queue, monkeypatch, trained):
    job = wrapper.jobs(wrapper.DEFAULT_OUTPUT)[0].copy()
    job["output"] = str(tmp_path / "raw")
    run = Path(job["output"])
    run.mkdir()
    for name in ("best.pt", "last.pt"):
        (run / name).write_bytes(name.encode())
    launched = []

    class Evaluator:
        pid = 9700

        def __init__(self, command, **kwargs):
            self.command = command
            launched.append(command)

        def wait(self):
            checkpoint = Path(self.command[self.command.index("--checkpoint") + 1])
            target = Path(self.command[self.command.index("--output") + 1])
            write_json(target, {"status": "evaluated", "world_head_trained": trained,
                               "checkpoint_file_unchanged": True, "runtime_source_matches_checkpoint": True,
                               "checkpoint_identity": {"path": str(checkpoint.resolve()),
                                "size_bytes": checkpoint.stat().st_size, "mtime_ns": checkpoint.stat().st_mtime_ns}})
            return 0

    monkeypatch.setattr(queue.subprocess, "Popen", Evaluator)
    monkeypatch.setattr(queue, "matching_world_process", lambda *args: None)
    if trained:
        result = queue.evaluate_completed_world(job, tmp_path / "manifest.json", lambda: None)
        assert set(result) == {"best", "last"}
        assert all(value["world_head_trained"] for value in result.values())
        assert [Path(cmd[cmd.index("--checkpoint") + 1]).name for cmd in launched] == ["best.pt", "last.pt"]
        assert all(cmd[cmd.index("--device") + 1] == "cuda" for cmd in launched)
    else:
        with pytest.raises(ValueError, match="checkpoint/source identity"):
            queue.evaluate_completed_world(job, tmp_path / "manifest.json", lambda: None)
    assert all("train-vla-jepa" not in cmd and "--resume" not in cmd for cmd in launched)


def test_cli_help_status_and_hidden_worker_suite_reconstruction(tmp_path, queue, monkeypatch, capsys):
    monkeypatch.setattr(wrapper, "_load_queue", lambda: queue)
    monkeypatch.setattr(sys, "argv", [str(spec.origin), "--help"])
    with pytest.raises(SystemExit) as exit_info:
        wrapper.main()
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--profile {dynamic}" in help_text and "--suite" not in help_text
    output = tmp_path / "new_output"
    monkeypatch.setattr(sys, "argv", [str(spec.origin), "--status", "--output", str(output)])
    assert wrapper.main() == 0
    assert not output.exists()
    capsys.readouterr()
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {})
    monkeypatch.setattr(queue, "jobs", lambda output, suite: wrapper.jobs(wrapper.DEFAULT_OUTPUT, suite))
    captured = []
    monkeypatch.setattr(queue, "work", lambda output, manifest, resume, concurrent, suite, budget:
                        captured.append((suite, queue.jobs(output, suite), concurrent, budget)) or 0)
    monkeypatch.setattr(sys, "argv", [str(spec.origin), "--worker", "--suite", wrapper.SUITE,
                                     "--output", str(output), "--manifest", str(manifest),
                                     "--max-concurrent", "2", "--gpu-memory-budget-mib", "29696"])
    monkeypatch.setattr(queue, "validate_memory_budget", lambda *args, **kwargs: None)
    assert wrapper.main() == 0
    assert captured == [(wrapper.SUITE, wrapper.jobs(wrapper.DEFAULT_OUTPUT), 2, 29696)]


@pytest.mark.parametrize("analysis_failure", [False, True])
def test_worker_runs_final_analysis_without_changing_completed_queue(tmp_path, monkeypatch, analysis_failure):
    output = tmp_path / "runs"
    completed = {"status": "completed", "jobs": [{"name": "frozen_dynamic_seed17", "status": "completed"}]}
    calls = []

    def work(target, manifest, resume, *, max_concurrent):
        assert (target, manifest, resume, max_concurrent) == (output, "manifest", False, 3)
        write_json(target / "controller.json", completed)
        return 0

    def analyze(target):
        calls.append(target)
        assert read_json(target / "controller.json") == completed
        if analysis_failure:
            raise ValueError("Invalid world artifact")
        value = {"status": "completed", "summary": {"runs": 6, "world_reports": 12}}
        write_json(target / "analysis.json", value)
        return value

    monkeypatch.setattr(wrapper, "_analyze_completed", analyze)
    assert wrapper._work_with_analysis(work, output, "manifest", False, max_concurrent=3) == int(analysis_failure)
    assert calls == [output]
    assert read_json(output / "controller.json") == completed
    receipt = read_json(output / "analysis.json")
    assert receipt["status"] == ("failed" if analysis_failure else "completed")
    if analysis_failure:
        assert receipt["error_type"] == "ValueError" and receipt["error"] == "Invalid world artifact"
    else:
        assert receipt["summary"] == {"runs": 6, "world_reports": 12}


def test_worker_failure_does_not_run_analysis(tmp_path, monkeypatch):
    def unexpected_analysis(target):
        raise AssertionError("Analysis must wait for a complete queue")

    monkeypatch.setattr(wrapper, "_analyze_completed", unexpected_analysis)
    assert wrapper._work_with_analysis(lambda *args, **kwargs: 7, tmp_path) == 7
    assert not (tmp_path / "analysis.json").exists()

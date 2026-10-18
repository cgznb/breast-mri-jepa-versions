"""Queue admission and wrapper CLI checks without launching training workers."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml

from mri_vla_jepa.data import RawMRIStore, make_raw_synthetic
from mri_vla_jepa.io import read_json, save_checkpoint, write_json
from mri_vla_jepa.train_config import RawVLATrainConfig, from_dict, load_config


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("vq_launcher_for_test", ROOT / "scripts/run_vq_encoder_experiment.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


@pytest.fixture
def queue():
    return wrapper._load_queue()


@pytest.mark.parametrize("suite,profiles,count", [
    ("vq_dynamic", {"dynamic"}, 4), ("vq_t0", {"t0"}, 4),
    ("vq_both", {"t0", "dynamic"}, 8),
])
def test_wrapper_jobs_match_control_configs_and_stage_dependencies(suite, profiles, count):
    jobs = wrapper.jobs(wrapper.DEFAULT_OUTPUT, suite)
    assert len(jobs) == count
    assert {job["profile"] for job in jobs} == profiles
    assert {job["seed"] for job in jobs} == {17, 43}
    by_name = {job["name"]: job for job in jobs}
    for job in jobs:
        cfg = from_dict(wrapper._release["portable_config"](load_config(job["config"]).to_dict(), ROOT))
        assert cfg.model.encoder_type == "vq_prequant"
        assert cfg.model.vq_activation_checkpointing
        assert cfg.seed == job["seed"]
        assert cfg.landmarks == ("t0" if job["profile"] == "t0" else "all_observed")
        assert cfg.jepa_weight == cfg.flow_weight == cfg.reconstruction_weight == cfg.variance_weight == 0
        assert cfg.batch_size == 16 and cfg.training_unit == "epochs"
        assert cfg.max_epochs == 200 and cfg.early_stopping_patience == 50
        if job["stage"] == "frozen":
            assert job["depends_on"] is None
            assert cfg.encoder_checkpoint == str(wrapper.CODEC)
            assert cfg.initialize_from_checkpoint is None
        else:
            parent = by_name[job["depends_on"]]
            assert parent["stage"] == "frozen" and parent["seed"] == job["seed"]
            assert parent["profile"] == job["profile"]
            assert cfg.encoder_checkpoint is None and cfg.encoder_lr == 1e-5
            assert cfg.initialize_from_checkpoint == str(Path(parent["output"]) / "best.pt")


def test_wrapper_rejects_config_drift_and_unmatched_custom_output(tmp_path, monkeypatch):
    original_load = wrapper.load_config

    def changed_config(path):
        cfg = original_load(path)
        if Path(path).name.startswith("vq_frozen_"):
            cfg.model = replace(cfg.model, dropout=.7)
        return cfg

    monkeypatch.setattr(wrapper, "load_config", changed_config)
    with pytest.raises(ValueError, match="matched L1/control protocol"):
        wrapper.jobs(wrapper.DEFAULT_OUTPUT)
    monkeypatch.setattr(wrapper, "load_config", original_load)
    custom = tmp_path / "custom_output"
    prepared = wrapper._load_queue().prepared_jobs(custom, "vq_dynamic", write=True)
    for job in prepared:
        if job["stage"] == "finetune":
            cfg = load_config(job["config"])
            assert cfg.initialize_from_checkpoint == str(custom / job["depends_on"] / "best.pt")


def test_status_before_launch_is_clear_and_read_only(tmp_path, queue, monkeypatch, capsys):
    output = tmp_path / "not_yet_created"
    assert queue.status(output) == {"status": "not_started", "output": str(output.resolve()),
                                    "controller_alive": False, "jobs": []}
    monkeypatch.setattr(wrapper, "_load_queue", lambda: queue)
    argv = [str(spec.origin), "--profile", "t0", "--status", "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    assert wrapper.main() == 0
    assert sys.argv is argv
    assert json.loads(capsys.readouterr().out)["status"] == "not_started"
    assert not output.exists()


def test_wrapper_help_lists_profiles_and_hides_worker_suite(queue, monkeypatch, capsys):
    monkeypatch.setattr(wrapper, "_load_queue", lambda: queue)
    monkeypatch.setattr(sys, "argv", [str(spec.origin), "--help"])
    with pytest.raises(SystemExit) as exit_info:
        wrapper.main()
    assert exit_info.value.code == 0
    text = capsys.readouterr().out
    assert "--profile {dynamic,t0,both}" in text
    assert "default: dynamic" in text
    assert "--detach" in text and "--resume" in text
    assert "--suite" not in text and "--worker" not in text


@pytest.mark.parametrize("profile", ["dynamic", "t0", "both"])
def test_detached_worker_reconstructs_hidden_suite_and_config_set(tmp_path, queue, monkeypatch, profile):
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {})
    expected = wrapper.jobs(wrapper.DEFAULT_OUTPUT, f"vq_{profile}")
    monkeypatch.setattr(wrapper, "_load_queue", lambda: queue)
    # Keep filesystem effects local while checking the actual committed config set.
    monkeypatch.setattr(queue, "jobs", lambda output, suite: wrapper.jobs(wrapper.DEFAULT_OUTPUT, suite))
    captured = {}

    def work(output, manifest, resume, max_concurrent, suite, budget):
        captured.update(suite=suite, jobs=queue.jobs(output, suite), resume=resume,
                        max_concurrent=max_concurrent, budget=budget)
        return 0

    monkeypatch.setattr(queue, "work", work)
    argv = [str(spec.origin), "--worker", "--suite", f"vq_{profile}", "--resume",
            "--output", str(tmp_path / "run"), "--manifest", str(manifest),
            "--max-concurrent", "3"]
    monkeypatch.setattr(sys, "argv", argv)
    assert wrapper.main() == 0
    assert captured["suite"] == f"vq_{profile}"
    assert captured["jobs"] == expected
    assert captured["resume"] and captured["max_concurrent"] == 3
    assert sys.argv is argv


def test_internal_suite_requires_worker(monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(spec.origin), "--suite", "vq_t0"])
    with pytest.raises(ValueError, match="Choose --profile"):
        wrapper.main()


@pytest.mark.parametrize("profile", ["dynamic", "t0", "both"])
def test_detached_command_reenters_wrapper_with_same_profile(tmp_path, queue, monkeypatch, capsys, profile):
    manifest, output = tmp_path / "manifest.json", tmp_path / "runs"
    write_json(manifest, {})
    suite = f"vq_{profile}"
    monkeypatch.setattr(wrapper, "_load_queue", lambda: queue)
    monkeypatch.setattr(queue, "jobs", lambda output, selected: wrapper.jobs(wrapper.DEFAULT_OUTPUT, selected))
    monkeypatch.setattr(queue, "alive_controller", lambda output: None)
    monkeypatch.setattr(queue, "alive_training_children", lambda *args: [])
    monkeypatch.setattr(queue, "alive_world_children", lambda *args: [])
    launched = []

    class Controller:
        pid = 9400

        def __init__(self, command, **kwargs):
            launched.append(command)

    monkeypatch.setattr(queue.subprocess, "Popen", Controller)
    monkeypatch.setattr(sys, "argv", [str(spec.origin), "--profile", profile, "--detach",
                                     "--output", str(output), "--manifest", str(manifest),
                                     "--max-concurrent", "2", "--gpu-memory-budget-mib", "29696"])
    monkeypatch.setattr(queue, "validate_memory_budget", lambda *args, **kwargs: None)
    assert wrapper.main() == 0
    command = launched[0]
    assert command[1] == str(ROOT / "scripts/run_vq_encoder_experiment.py")
    assert "--worker" in command
    assert command[command.index("--suite") + 1] == suite
    assert "--profile" not in command
    expected = queue.prepared_jobs(output, suite)
    assert read_json(output / "launch.json")["jobs"] == expected
    capsys.readouterr()
    reconstructed = []
    monkeypatch.setattr(queue, "work", lambda output, manifest, resume, concurrent, selected, budget:
                        reconstructed.append((selected, queue.prepared_jobs(output, selected), concurrent, budget)) or 0)
    monkeypatch.setattr(sys, "argv", command[1:])
    assert wrapper.main() == 0
    assert reconstructed == [(suite, expected, 2, 29696)]


def test_measured_gpu_reservations_allow_two_stage_workers_with_margin(queue):
    assert wrapper.MEMORY_MIB == {"frozen": 9216, "finetune": 13312}
    budget = 29696
    for stage in ("frozen", "finetune"):
        reservation = queue.training_memory_mib({"stage": stage})
        assert queue.admission_reason(reservation, 1, reservation, 2, budget) is None
        assert queue.admission_reason(reservation, 2, reservation * 2, 2, budget) == "concurrency_limit"


def queue_fixture(tmp_path, queue, monkeypatch):
    output = tmp_path / "queue"
    output.mkdir()
    cfg = RawVLATrainConfig(image_shape=(4, 8, 8), training_unit="epochs", max_epochs=1,
                            batch_size=2, accumulation=1, landmarks="t0",
                            device="cpu", precision="fp32", allow_synthetic=True)
    config = tmp_path / "synthetic_config.yaml"
    config.write_text(yaml.safe_dump(cfg.to_dict()))
    manifest = make_raw_synthetic(tmp_path / "data", n_train=3, n_val=2,
                                  image_shape=cfg.image_shape)
    store = RawMRIStore(manifest, cfg.image_shape, allow_synthetic=True)

    def planned(output, suite="vq_dynamic"):
        return [{"name": stage, "stage": stage, "profile": "t0", "seed": 17,
                 "config": str(config), "output": str(output / stage), "status": "queued",
                 "depends_on": "frozen" if stage == "finetune" else None}
                for stage in ("frozen", "finetune")]

    monkeypatch.setattr(queue, "jobs", planned)
    monkeypatch.setattr(queue, "matching_training_process", lambda *args, **kwargs: None)
    report = {"completed": True, "landmarks": "t0", "selection_protocol": queue._selection_protocol("t0"),
              "stop_reason": "max_epochs", "best_validation_nll": .4,
              "completed_steps": 2, "completed_epochs": 1}
    checkpoint = {"schema": queue.CHECKPOINT_SCHEMA, "config": cfg.to_dict(),
                  "source_snapshot": queue._source_snapshot(), "data_signature": store.signature(),
                  "normalization": store.normalization_state(), "patient_partitions": queue._patient_partitions(store),
                  "protocol": queue._protocol("t0"), "selection_protocol": queue._selection_protocol("t0"),
                  "best_nll": .4, "completed_steps": 2, "epoch_state": {"completed_epochs": 1}}

    def completed(run, *, valid=True):
        write_json(run / "train_report.json", report)
        write_json(run / "best_validation.json", {"selection_score": .4})
        save_checkpoint(run / "last.pt", checkpoint)
        if valid:
            save_checkpoint(run / "best.pt", checkpoint)

    return output, manifest, planned, completed


@pytest.mark.parametrize("outcome", ["success", "exit_failed", "checkpoint_missing"])
def test_finetune_waits_for_validated_frozen_completion_and_propagates_failure(tmp_path, queue, monkeypatch, outcome):
    output, manifest, planned, completed = queue_fixture(tmp_path, queue, monkeypatch)
    clock, starts, snapshots, validated = {"tick": 0}, [], [], []
    real_validation = queue.validate_completed_run

    def validate(job, manifest):
        report = real_validation(job, manifest)
        validated.append((job["name"], clock["tick"]))
        return report

    monkeypatch.setattr(queue, "validate_completed_run", validate)

    class Child:
        def __init__(self, command, **kwargs):
            self.run = Path(command[command.index("--output") + 1])
            self.started = clock["tick"]
            self.pid = 9100 + len(starts)
            if self.run.name == "finetune":
                assert any(name == "frozen" for name, _ in validated)
            starts.append((self.run.name, self.started))

        def poll(self):
            if clock["tick"] - self.started < 2:
                return None
            if self.run.name == "frozen" and outcome == "exit_failed":
                return 1
            completed(self.run, valid=not (self.run.name == "frozen" and outcome == "checkpoint_missing"))
            return 0

    def advance(seconds):
        snapshots.append(read_json(output / "controller.json"))
        clock["tick"] += 1

    monkeypatch.setattr(queue.subprocess, "Popen", Child)
    monkeypatch.setattr(queue.time, "sleep", advance)
    assert queue.work(output, manifest, resume=False, max_concurrent=2, suite="vq_dynamic") == (0 if outcome == "success" else 1)
    waiting = next(job for job in snapshots[0]["jobs"] if job["name"] == "finetune")
    assert waiting["status"] == "queued" and waiting["deferred_reason"] == "waiting_for_dependency"
    state = read_json(output / "controller.json")
    by_name = {job["name"]: job for job in state["jobs"]}
    if outcome == "success":
        assert starts == [("frozen", 0), ("finetune", 2)]
        assert state["status"] == "completed"
    else:
        assert starts == [("frozen", 0)]
        assert state["status"] == "failed"
        assert by_name["finetune"]["status"] == "failed"
        assert by_name["finetune"]["deferred_reason"] == "dependency_failed"
        assert "Prerequisite frozen failed" in by_name["finetune"]["failure_reason"]


@pytest.mark.parametrize("valid", [True, False])
def test_resume_completed_parent_must_revalidate_checkpoint_before_finetune(tmp_path, queue, monkeypatch, valid):
    output, manifest, planned, completed = queue_fixture(tmp_path, queue, monkeypatch)
    old = planned(output)
    old[0]["status"] = "completed"
    write_json(output / "controller.json", {"manifest": str(manifest), "suite": "vq_dynamic", "jobs": old})
    completed(output / "frozen", valid=valid)
    launches = []

    class Child:
        pid = 9300

        def __init__(self, command, **kwargs):
            self.run = Path(command[command.index("--output") + 1])
            launches.append(self.run.name)

        def poll(self):
            completed(self.run)
            return 0

    monkeypatch.setattr(queue.subprocess, "Popen", Child)
    monkeypatch.setattr(queue.time, "sleep", lambda seconds: None)
    if valid:
        assert queue.work(output, manifest, resume=True, suite="vq_dynamic") == 0
        assert launches == ["finetune"]
    else:
        with pytest.raises(FileNotFoundError, match="best.pt"):
            queue.work(output, manifest, resume=True, suite="vq_dynamic")
        assert not launches

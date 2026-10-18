"""Only verified terminal-print failures may be reconciled without retraining."""
import fcntl
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "overfit_finalization_recovery", Path(__file__).parents[1] / "scripts/recover_overfit_finalization.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


@pytest.fixture
def failed_suite(tmp_path, monkeypatch):
    queue = recovery.queue
    suite = tmp_path / "screening"
    config, manifest = tmp_path / "config.yaml", tmp_path / "manifest.json"
    config.write_text("train:\n  seed: 17\n")
    manifest.write_text("{}")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"runs": [{"name": "B0_dynamic_seed17", "config": str(config),
        "manifest": str(manifest), "output": str(tmp_path / "run"), "reservation_mib": 10240}]}))
    status = queue.prepare_suite(recovery.PROJECT, suite, plan)
    run = status["runs"][0]
    run.update(state="failed", failure="Trainer exit 1; completion/artifacts incomplete",
               attempts=[{"attempt": 1, "resume": False, "exit_code": 1}])
    output = Path(run["output"])
    (output / ".queue").mkdir(parents=True)
    queue.atomic_json(output / "completion.json", {"state": "completed", "best_update": 100})
    queue.atomic_json(output / "metrics_best.json", {"selection": .632, "selection_name": "patient_equal_pcr_nll"})
    for name in ("best.pt", "last.pt", "config.json", "history.json", "predictions_val.csv",
                 "world_val.csv", "provenance.json"):
        (output / name).write_bytes(b"fixture artifact\n")
    _, exit_path = queue.attempt_paths(run, run["attempts"][0])
    queue.atomic_json(exit_path, {"exit_code": 1, "ended_at": "original timestamp"})
    Path(run["log"]).write_text(
        "update=600\nTraceback (most recent call last):\n"
        f'  File "{recovery.PROJECT / "src/response_jepa/cli.py"}", line 85, in main\n'
        '    print(json.dumps({"output": result["output"], "best_epoch": result["best_epoch"]}, ensure_ascii=False))\n'
        '                                                                ~~~~~~^^^^^^^^^^^^^^\n'
        "KeyError: 'best_epoch'\n")
    queue.atomic_json(suite / "status.json", status)
    queue.atomic_json(tmp_path / "research_status.json", {"state": "stopped"})
    monkeypatch.setattr(queue, "find_process", lambda *args, **kwargs: None)
    return suite, plan, tmp_path / "report", run, exit_path


def test_verified_recovery_preserves_failure_evidence_and_does_not_repeat_work(failed_suite, monkeypatch):
    suite, plan, report, run, exit_path = failed_suite
    before = (suite / "status.json").read_bytes()
    original_exit = exit_path.read_bytes()
    checkpoints = {name: (Path(run["output"]) / name).read_bytes() for name in ("best.pt", "last.pt")}
    calls = []

    def replay(requested):
        assert requested == plan
        assert (suite / "status.json").read_bytes() == before
        assert not (suite / "post_training_recovery.json").exists()
        calls.append("replay")
        return {"complete": True, "errors": [], "runs": [{"name": run["name"], "verified": True}]}

    def report_writer(audited, path):
        assert calls == ["replay"]
        assert (suite / "status.json").read_bytes() == before
        calls.append("report")
        path.mkdir()
        recovery.queue.atomic_json(path / "aggregate.json", audited)

    monkeypatch.setattr(recovery.analysis, "analyze_suite", replay)
    monkeypatch.setattr(recovery.analysis, "write_report", report_writer)
    result = recovery.recover(suite, plan, report)
    assert result["state"] == "recovered" and result["runs"] == 1
    assert calls == ["replay", "report"]
    assert exit_path.read_bytes() == original_exit
    assert (suite / "status.before_post_training_recovery.json").read_bytes() == before
    saved = recovery.queue.read_json(suite / "status.json")["runs"][0]
    assert saved["state"] == "completed" and saved["completion_basis"] == recovery.BASIS
    assert saved["attempts"][0]["exit_code"] == 1 and saved["original_failure"] == run["failure"]
    receipt = recovery.queue.read_json(suite / "post_training_recovery.json")
    assert receipt["training_repeated"] is False and receipt["original_exits_preserved"] is True
    assert receipt["runs"][0]["original_exit"]["exit_code"] == 1
    assert {name: (Path(run["output"]) / name).read_bytes() for name in checkpoints} == checkpoints
    assert recovery.recover(suite, plan, report)["state"] == "already_recovered"
    assert calls == ["replay", "report"]


@pytest.mark.parametrize("change", ["other_error", "extra_traceback", "wrong_frame", "not_terminal"])
def test_other_failure_signatures_cannot_be_reconciled(failed_suite, monkeypatch, change):
    suite, plan, report, run, _ = failed_suite
    path = Path(run["log"])
    text = path.read_text()
    if change == "other_error":
        text = text.replace("KeyError: 'best_epoch'", "RuntimeError: CUDA out of memory")
    elif change == "extra_traceback":
        text = "Traceback (most recent call last):\n" + text
    elif change == "wrong_frame":
        text = text.replace("line 85, in main", "line 84, in main")
    else:
        text += "subsequent unrelated failure\n"
    path.write_text(text)
    before = (suite / "status.json").read_bytes()
    monkeypatch.setattr(recovery.analysis, "analyze_suite", lambda _: pytest.fail("Unexpected replay"))
    with pytest.raises(ValueError, match="exact post-training"):
        recovery.recover(suite, plan, report)
    assert (suite / "status.json").read_bytes() == before
    assert not (suite / "post_training_recovery.json").exists()


@pytest.mark.parametrize("audit", [
    {"complete": False, "errors": [], "runs": []},
    {"complete": True, "errors": ["prediction replay mismatch"], "runs": []},
    {"complete": True, "errors": [], "runs": [{"name": "wrong_run", "verified": True}]},
    {"complete": True, "errors": [], "runs": [{"name": "B0_dynamic_seed17", "verified": False}]},
])
def test_failed_strict_replay_leaves_queue_and_evidence_unchanged(failed_suite, monkeypatch, audit):
    suite, plan, report, _, exit_path = failed_suite
    before, original_exit = (suite / "status.json").read_bytes(), exit_path.read_bytes()
    monkeypatch.setattr(recovery.analysis, "analyze_suite", lambda _: audit)
    monkeypatch.setattr(recovery.analysis, "write_report", lambda *args: pytest.fail("Unexpected report"))
    with pytest.raises(ValueError, match="Strict artifact"):
        recovery.recover(suite, plan, report)
    assert (suite / "status.json").read_bytes() == before and exit_path.read_bytes() == original_exit
    assert not (suite / "status.before_post_training_recovery.json").exists()
    assert not (suite / "post_training_recovery.json").exists()


@pytest.mark.parametrize("lock_name", ["research", "controller", "worker"])
def test_live_owner_lock_blocks_reconciliation(failed_suite, monkeypatch, lock_name):
    suite, plan, report, run, _ = failed_suite
    locks = {"research": suite.parent / ".research.lock", "controller": suite / ".controller.lock",
             "worker": Path(run["output"]) / ".queue/worker.lock"}
    before = (suite / "status.json").read_bytes()
    monkeypatch.setattr(recovery.analysis, "analyze_suite", lambda _: pytest.fail("Unexpected replay"))
    with locks[lock_name].open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            recovery.recover(suite, plan, report)
    assert (suite / "status.json").read_bytes() == before


def test_live_trainer_blocks_reconciliation_without_lock(failed_suite, monkeypatch):
    suite, plan, report, _, _ = failed_suite
    before = (suite / "status.json").read_bytes()
    monkeypatch.setattr(recovery.queue, "find_process", lambda *args, **kwargs: {"pid": 123})
    monkeypatch.setattr(recovery.analysis, "analyze_suite", lambda _: pytest.fail("Unexpected replay"))
    with pytest.raises(ValueError, match="still alive"):
        recovery.recover(suite, plan, report)
    assert (suite / "status.json").read_bytes() == before


@pytest.mark.parametrize("missing", ["status", "queue_plan", "input_plan"])
def test_missing_queue_metadata_cannot_create_new_recovery_state(failed_suite, monkeypatch, missing):
    suite, plan, report, _, _ = failed_suite
    paths = {"status": suite / "status.json", "queue_plan": suite / "plan.json", "input_plan": plan}
    paths[missing].unlink()
    before = {str(path.relative_to(suite.parent)): path.read_bytes()
              for path in suite.parent.rglob("*") if path.is_file()}
    monkeypatch.setattr(recovery.queue, "prepare_suite", lambda *args: pytest.fail("Unexpected preparation"))
    with pytest.raises(ValueError, match="existing queue"):
        recovery.recover(suite, plan, report)
    after = {str(path.relative_to(suite.parent)): path.read_bytes()
             for path in suite.parent.rglob("*") if path.is_file()}
    assert after == before

"""Watcher completion decisions never conflate progress with final results."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1]/"scripts"/"finalize_joint_suite.py"
spec = importlib.util.spec_from_file_location("joint_finalize", SCRIPT)
finalizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finalizer)


def status(state="completed"):
    return {"runs": [{"name": name, "state": state} for name in sorted(finalizer.EXPECTED)]}


def test_complete_and_failed_are_terminal_but_running_is_not():
    value = status()
    assert finalizer.disposition(value, False) == "terminal"
    value["runs"][0]["state"] = "failed"
    assert finalizer.disposition(value, True) == "terminal"
    value["runs"][1]["state"] = "running"
    assert finalizer.disposition(value, False) == "waiting"


def test_alive_controller_can_wait_unlimited_and_orphan_is_partial_trigger():
    value = status("queued")
    assert finalizer.disposition(value, True) == "waiting"
    assert finalizer.disposition(value, False) == "orphaned"
    value["runs"].pop()
    assert finalizer.disposition(value, False) == "waiting"
    assert finalizer.disposition(None, False) == "waiting"


@pytest.mark.parametrize("complete,expected", [(True, 0), (False, 2)])
def test_analysis_reports_partial_as_nonzero_and_is_cpu_only(tmp_path, monkeypatch, complete, expected):
    output = tmp_path/"output"
    output.mkdir()
    (output/"aggregate.json").write_text(json.dumps({"complete": complete}))
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(finalizer.subprocess, "run", run)
    result = finalizer.run_analysis(tmp_path, output, tmp_path/"baseline.csv", project=tmp_path)
    assert result["returncode"] == expected
    assert result["partial"] is not complete
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert calls[0][1]["env"]["OMP_NUM_THREADS"] == "2"
    assert calls[0][1]["cwd"] == tmp_path
    assert "--baseline-csv" in calls[0][0]


def test_failed_analysis_does_not_report_success(tmp_path, monkeypatch):
    monkeypatch.setattr(finalizer.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=7))
    result = finalizer.run_analysis(tmp_path, tmp_path)
    assert result["returncode"] == 7
    assert result["analysis_returncode"] == 7
    assert result["partial"]


def test_optional_paired_comparison_and_plot_use_cpu_and_separate_outputs(tmp_path, monkeypatch):
    scripts = tmp_path/"scripts"
    scripts.mkdir()
    for name in ("compare_joint_old_predictions.py", "plot_joint_suite.py"):
        (scripts/name).touch()
    output = tmp_path/"report"
    output.mkdir()
    (output/"aggregate.json").write_text(json.dumps({"complete": True}))
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(finalizer.subprocess, "run", run)
    result = finalizer.run_analysis(tmp_path/"suite", output, project=tmp_path)
    assert result["returncode"] == result["paired_returncode"] == result["plot_returncode"] == 0
    assert "compare_joint_old_predictions.py" in calls[1][0][1]
    assert calls[1][0][-1] == str(output/"paired_baseline")
    assert calls[2][0][-1] == str(output/"figures")
    assert all(call[1]["env"]["CUDA_VISIBLE_DEVICES"] == "" for call in calls)


def test_watcher_owns_only_finalizer_status_and_records_failed_partial(tmp_path, monkeypatch):
    original = status()
    original["runs"][0]["state"] = "failed"
    queue = tmp_path/"status.json"
    queue.write_text(json.dumps(original))
    monkeypatch.setattr(finalizer, "controller_alive", lambda *a: False)
    monkeypatch.setattr(finalizer, "run_analysis", lambda *a: {
        "analysis_returncode": 0, "plot_returncode": None, "partial": True,
        "report_path": str(tmp_path/"README_ZH.md"), "error": "Partial report", "returncode": 2})
    assert finalizer.watch(tmp_path, tmp_path) == 2
    result = json.loads((tmp_path/"finalize_status.json").read_text())
    assert result["state"] == "failed" and result["partial"]
    assert json.loads(queue.read_text()) == original

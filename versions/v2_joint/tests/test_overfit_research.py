import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("overfit_research", Path(__file__).parents[1] / "scripts/run_overfit_research.py")
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)


def test_internal_selection_never_promotes_historical_baseline():
    rows = [{"algorithm": f"B{i}", "selection": .55+i*.01} for i in range(5)]
    result = research.choose_candidate({"complete": True, "errors": [], "runs": rows})
    assert result["algorithm"] == "B1"
    assert result["engineering_target_met"] is False
    assert result["delta_nll"] > 0


@pytest.mark.parametrize("complete,errors", [(False, []), (True, ["prediction replay failed"])])
def test_partial_or_unverified_screen_cannot_select(complete, errors):
    with pytest.raises(ValueError, match="verified"):
        research.choose_candidate({"complete": complete, "errors": errors})


def test_plan_is_frozen_and_variants_change_only_intended_algorithm(tmp_path):
    cache = tmp_path / "cache"
    p = research.prepare_plan(tmp_path / "screen", [("B0", "dynamic", 17), ("B4", "dynamic", 17)],
                              tmp_path / "manifest.json", cache)
    import json
    from response_jepa.config import load_config
    plan = json.loads(p.read_text())
    a, b = [load_config(run["config"]) for run in plan["runs"]]
    assert a.model.dim == b.model.dim == 128
    assert a.train.max_updates == b.train.max_updates == 3000
    assert a.train.reference_batch_size == b.train.reference_batch_size == 16
    assert a.loss.consistency == 0 and b.loss.consistency == .1
    assert not a.model.forecast_detach and b.model.forecast_detach
    assert a.train.sample_weighting == "sample" and b.train.sample_weighting == "patient_equal"
    with pytest.raises(ValueError, match="differs"):
        research.prepare_plan(tmp_path / "screen", [("B0", "dynamic", 17)], tmp_path / "manifest.json", cache)


def test_restarted_research_joins_live_queue_without_duplicate_controller(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sys
    directory = tmp_path / "screen"
    directory.mkdir()
    plan = directory / "experiment_plan.json"
    command = [sys.executable, str(tmp_path / "scripts/run_overfit_suite.py"), "--plan", str(plan),
               "--output", str(directory), "--max-concurrent", "3", "--gpu-memory-budget-mib", "31232"]
    info = {"cmdline": command, "cwd": str(tmp_path), "start_ticks": "42"}
    statuses = iter([{"controller": {"pid": 123, "start_ticks": "42"}, "runs": [{"state": "running"}]},
                     {"runs": [{"state": "completed"}]}])
    monkeypatch.setattr(research, "PROJECT", tmp_path)
    monkeypatch.setattr(research, "process_info", lambda pid: info)
    monkeypatch.setattr(research, "read_json", lambda path: next(statuses) if path.name == "status.json"
                        else {"complete": True, "errors": []})
    waits, launched = [], []
    monkeypatch.setattr(research.time, "sleep", lambda seconds: waits.append(seconds))
    monkeypatch.setattr(research, "child", lambda command, log: launched.append(command))
    result, _ = research.run_stage(plan, directory, SimpleNamespace(output=tmp_path / "experiment", budget=31232))
    assert result["complete"]
    assert waits == [10]
    assert len(launched) == 1 and "analyze_overfit_suite.py" in launched[0][1]

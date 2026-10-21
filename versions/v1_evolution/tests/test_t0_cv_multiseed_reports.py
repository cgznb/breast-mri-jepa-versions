"""Report ordering follows validation and cached intervals stay inside each seed."""
from pathlib import Path

import pytest

from mri_vla_jepa import t0_cv_suite as suite
from mri_vla_jepa.io import read_json, write_json
from mri_vla_jepa.training import _scores
from test_t0_cv_suite import spec, _multiseed_spec, _prepare, _write_evaluations


def test_multiseed_ranking_uses_validation_and_caches_per_seed_intervals(spec, monkeypatch):
    state = _prepare(_multiseed_spec(spec, [43, 44]))
    _write_evaluations(state, complete=True)
    winner = "t0_direct_no_jepa"
    for job in state["jobs"]:
        for split in ("val", "test"):
            path = Path(job["output"]) / f"evaluation_best_{split}.json"
            report = read_json(path)
            correct = (job["variant"] == winner) if split == "val" else (job["variant"] != winner)
            for row in report["rows"]:
                row["pcr_probability"] = (.9 if row["pcr"] else .1) if correct else (.1 if row["pcr"] else .9)
            report.update(_scores(report["rows"], "t0", selection_metric="f1", f1_threshold=.5))
            write_json(path, report)
            if split == "val":
                write_json(Path(job["output"]) / "evaluation_last_val.json", dict(report, checkpoint_epoch=61))
    report = suite.write_reports(state["output_dir"], state, bootstrap=3)
    assert report["ranking"][0] == winner
    assert report["aggregate_metrics"][0]["variant"] == winner
    assert next(row for row in report["seed_summary"] if row["variant"] == winner
                and row["population"] == "test_ensemble")["f1_mean"] == 0
    public = Path(state["output_dir"]) / "reports"
    for seed in (43, 44):
        child = read_json(public / f"seed{seed}/report.json")
        assert child["ranking"][0] == winner
        metrics = [row for row in child["aggregate_metrics"] if row["population"] == "test_ensemble"]
        assert all(row["train_seed"] == seed and row["bootstrap"]["repetitions"] == 3 for row in metrics)
        assert all(set(row["bootstrap"]["intervals"]) == set(suite.REPORT_METRICS) for row in metrics)
        assert (public / f"seed{seed}/bootstrap_cache.json").is_file()
    monkeypatch.setattr(suite, "_stratified_bootstrap", lambda *args: pytest.fail("Unchanged reports recomputed bootstrap"))
    assert suite.write_reports(state["output_dir"], state, bootstrap=3) == report


@pytest.mark.parametrize("field,wrong", [("train_seed", 809), ("config_digest", "another-seed-config")])
def test_multiseed_report_rejects_result_from_another_seed(spec, field, wrong):
    state = _prepare(_multiseed_spec(spec, [43, 44]))
    _write_evaluations(state, complete=True)
    job = state["jobs"][0]
    path = Path(job["output"]) / "evaluation_best_val.json"
    report = read_json(path)
    report[field] = wrong
    write_json(path, report)
    with pytest.raises(ValueError, match="belongs to another run"):
        suite.write_reports(state["output_dir"], state, bootstrap=0)

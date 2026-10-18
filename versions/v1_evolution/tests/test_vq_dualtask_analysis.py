"""Saved-artifact analysis must reject stale identities and incorrect ratios."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from mri_vla_jepa.io import load_checkpoint, write_json
import mri_vla_jepa.training as training
from test_vq_dualtask_experiment import codec, dual_setup, single_cpu_thread


@pytest.fixture(scope="module")
def analysis():
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_vq_dualtask_experiment.py"
    spec = importlib.util.spec_from_file_location("dualtask_analysis_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def world_fixture(analysis, checkpoint, stages):
    horizons, adjacent = analysis._world_counts(stages)
    metrics = {}
    for horizon, (patients, count) in horizons.items():
        values = {"patients": patients, "prefix_targets": count}
        for task, error, copy in (("teacher_forcing", .8, 1.), ("autonomous", 1.1, 1.)):
            values.update({f"{task}_l1": error, f"{task}_copy_l1": copy,
                           f"{task}_relative_to_copy": error / copy,
                           f"{task}_skill_vs_copy": 1 - error / copy,
                           f"{task}_near_zero_copy_patients": 0})
        metrics[horizon] = values
    return {"status": "evaluated", "world_head_trained": True,
            "checkpoint_file_unchanged": True, "runtime_source_matches_checkpoint": True,
            "checkpoint_identity": checkpoint["checkpoint_identity"],
            "checkpoint": checkpoint["checkpoint_identity"]["path"],
            "checkpoint_epoch": checkpoint["epoch"], "checkpoint_steps": checkpoint["steps"],
            "split": "val", "independent_test": False, "development_validation": True,
            "synthetic": True, "precision": "fp32", "landmarks": "all_observed",
            "patients": len(stages), "legal_prefixes": sum(map(len, stages.values())),
            "target_coordinates": "same checkpoint EMA teacher for model and both copy baselines",
            "chain_policy": "real continuous visits only; target and all intermediate visits required",
            "reduction": "mean eligible prefix-targets within patient, then mean patients",
            "teacher_forcing_copy": "real immediate predecessor",
            "autonomous_copy": "real landmark state held constant", "metrics": metrics,
            "teacher_forcing_all_adjacent": {"metrics": {
                name: {"patients": patients, "prefix_pairs": count,
                       "teacher_forcing_l1": .75, "teacher_forcing_copy_l1": 1.,
                       "relative_to_copy": .75, "skill_vs_copy": .25, "near_zero_copy_patients": 0}
                for name, (patients, count) in adjacent.items()}}}


@pytest.fixture(scope="module")
def completed_suite(tmp_path_factory, codec, analysis):
    directory = tmp_path_factory.mktemp("dualtask_saved_analysis")
    cfg, store = dual_setup(directory / "data", codec, max_epochs=1)
    output = directory / "runs"
    jobs = []
    for seed in (17, 43):
        for stage in ("raw", "frozen", "finetune"):
            current = replace(cfg, seed=seed)
            if stage == "raw":
                current = replace(current, model=replace(cfg.model, encoder_type="raw", vq_activation_checkpointing=False),
                                  encoder_training="finetune", encoder_checkpoint=None)
            elif stage == "finetune":
                current = replace(current, encoder_training="finetune", encoder_checkpoint=None, encoder_lr=1e-5,
                                  initialize_from_checkpoint=str(output / f"frozen_dynamic_seed{seed}/best.pt"))
            name = f"{stage}_dynamic_seed{seed}"
            config = directory / f"{name}.yaml"
            config.write_text(yaml.safe_dump(current.to_dict()), encoding="utf-8")
            run = output / name
            training.train(store, current, run)
            pcr = analysis._pcr(run, current, store, current_source=True)
            _, stages = analysis._population(store)
            for kind in analysis.KINDS:
                write_json(run / f"world_{kind}.json", world_fixture(analysis, pcr[kind], stages))
            jobs.append({"name": name, "stage": stage, "seed": seed, "config": str(config), "output": str(run)})
    queue = analysis._wrapper()._load_queue()
    queue.jobs = lambda output, suite: jobs
    return output, jobs, store, SimpleNamespace(jobs=lambda output: jobs, SUITE="synthetic_dualtask",
                                               _load_queue=lambda: queue)


def test_final_analysis_validates_twelve_artifacts_and_exports_no_patient_rows(
        completed_suite, analysis, monkeypatch):
    output, jobs, store, wrapper = completed_suite
    monkeypatch.setattr(analysis, "_wrapper", lambda: wrapper)
    result = analysis.analyze(output, manifest=store.path, include_references=False)
    assert result["status"] == "completed" and result["completed_training_runs"] == 6
    assert result["last_pcr_validation"] == (
        "Checkpoint identity and saved checkpoint-epoch aggregates; no last-patient prediction replay.")
    assert result["validated_world_reports"] == 12 and len(result["runs"]) == 6
    assert len(result["pcr_means"]) == len(result["world_means"]) == 6
    assert all(row["complete_seed_pair"] for row in result["pcr_means"])
    assert all(row["all_adjacent_tf_ratio_mean"] == .75 for row in result["world_means"])
    assert {kind for row in result["runs"] for kind in row["world"]} == {"best", "last"}
    text = (output / "analysis.json").read_text()
    assert '"patient_key"' not in text and '"rows"' not in text and "synthetic_" not in text
    assert "sha256" not in text and "config_digest" not in text
    assert json.loads(text) == result
    markdown = (output / "RESULTS_ZH.md").read_text()
    assert "last" in markdown and "未重新推理患者预测" in markdown
    for row in result["runs"]:
        assert row["pcr"]["best"]["validation"] == "saved_prediction_replay"
        assert row["pcr"]["last"]["validation"] == "checkpoint_and_epoch_aggregates"


def test_partial_analysis_marks_pending_world_without_overwriting_on_final_rejection(
        completed_suite, analysis, monkeypatch):
    output, _, store, wrapper = completed_suite
    monkeypatch.setattr(analysis, "_wrapper", lambda: wrapper)
    path = output / "finetune_dynamic_seed43/world_last.json"
    original = path.read_bytes()
    path.unlink()
    try:
        result = analysis.analyze(output, require_complete=False, manifest=store.path, include_references=False)
        assert result["status"] == "partial" and result["validated_world_reports"] == 11
        before = (output / "analysis.json").read_bytes()
        with pytest.raises(ValueError, match="6 completed.*12 valid"):
            analysis.analyze(output, manifest=store.path, include_references=False)
        assert (output / "analysis.json").read_bytes() == before
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize("corruption", ["source", "head", "identity", "epoch", "ratio", "cohort", "copy"])
def test_world_analysis_rejects_invalid_saved_contracts(tmp_path, completed_suite, analysis, corruption):
    output, jobs, store, _ = completed_suite
    _, stages = analysis._population(store)
    run = Path(jobs[0]["output"])
    state = load_checkpoint(run / "best.pt")
    checkpoint = {"checkpoint_identity": analysis._identity(run / "best.pt"),
                  "epoch": state["epoch_state"]["completed_epochs"], "steps": state["completed_steps"]}
    data = world_fixture(analysis, checkpoint, stages)
    if corruption == "source":
        data["runtime_source_matches_checkpoint"] = False
    elif corruption == "head":
        data["world_head_trained"] = False
    elif corruption == "identity":
        data["checkpoint_identity"] = {**data["checkpoint_identity"], "size_bytes": 1}
    elif corruption == "epoch":
        data["checkpoint_epoch"] += 1
    elif corruption == "ratio":
        data["metrics"]["horizon_all"]["autonomous_relative_to_copy"] = .4
    elif corruption == "cohort":
        data["metrics"]["horizon_2"]["prefix_targets"] -= 1
    else:
        data["autonomous_copy"] = "wrong predecessor"
    path = tmp_path / "invalid_world.json"
    write_json(path, data)
    with pytest.raises(ValueError):
        analysis._world(path, checkpoint, stages, synthetic=True)


def test_world_population_includes_late_adjacent_pairs_after_prior_missing_visit(analysis):
    chain, adjacent = analysis._world_counts({"fixture_a": {0, 2, 3}, "fixture_b": {0, 1}})
    assert chain["horizon_1"] == (2, 2) and chain["horizon_2"] == (0, 0)
    assert chain["horizon_all"] == (2, 2)
    assert adjacent["T2_to_T3"] == (1, 2) and adjacent["all"] == (2, 3)


def test_near_zero_copy_and_empty_cohort_keep_null_ratio_guards(analysis):
    empty = {"patients": 0, "teacher_forcing_l1": None, "teacher_forcing_copy_l1": None,
             "relative_to_copy": None, "skill_vs_copy": None, "near_zero_copy_patients": 0}
    assert analysis._ratio(empty, adjacent=True)["teacher_forcing_relative_to_copy"] is None
    zero = {**empty, "patients": 1, "teacher_forcing_l1": .3,
            "teacher_forcing_copy_l1": 0., "near_zero_copy_patients": 1}
    assert analysis._ratio(zero, adjacent=True)["teacher_forcing_relative_to_copy"] is None
    with pytest.raises(ValueError, match="near-zero"):
        analysis._ratio({**zero, "relative_to_copy": 0., "skill_vs_copy": 1.}, adjacent=True)


@pytest.mark.parametrize("field", ["normalization", "patient_partitions", "data_signature"])
def test_historical_reference_requires_matching_data_population_and_coordinates(
        completed_suite, analysis, monkeypatch, field):
    _, jobs, store, _ = completed_suite
    run = Path(jobs[0]["output"])
    original = analysis.load_checkpoint

    def mismatched(path):
        state = original(path)
        if Path(path) == run / "best.pt":
            state[field] = {"different_historical_contract": True}
        return state

    monkeypatch.setattr(analysis, "load_checkpoint", mismatched)
    with pytest.raises(ValueError, match="data/normalization/population"):
        analysis._pcr(run, training.load_config(jobs[0]["config"]), store, current_source=False)


def test_partial_retries_only_certified_concurrent_snapshot_changes(completed_suite, analysis, monkeypatch):
    output, jobs, store, wrapper = completed_suite
    target = Path(jobs[0]["output"])
    monkeypatch.setattr(analysis, "_wrapper", lambda: wrapper)
    read = analysis.read_json
    pcr = analysis._pcr
    attempts = []

    def active_report(path):
        data = read(path)
        return {**data, "completed": False} if Path(path) == target / "train_report.json" else data

    def raced_once(run, *args, **kwargs):
        if run == target:
            attempts.append(run)
            if len(attempts) == 1:
                raise analysis.SnapshotChanged("Concurrent trainer saved")
        return pcr(run, *args, **kwargs)

    monkeypatch.setattr(analysis, "read_json", active_report)
    monkeypatch.setattr(analysis, "_pcr", raced_once)
    result = analysis.analyze(output, require_complete=False, manifest=store.path, include_references=False)
    assert len(attempts) == 2 and result["completed_training_runs"] == 5
    assert result["status"] == "partial"
    attempts.clear()

    def corrupt(run, *args, **kwargs):
        attempts.append(run)
        raise ValueError("Metric corruption")

    monkeypatch.setattr(analysis, "_pcr", corrupt)
    with pytest.raises(ValueError, match="Metric corruption"):
        analysis.analyze(output, require_complete=False, manifest=store.path, include_references=False)
    assert len(attempts) == 1

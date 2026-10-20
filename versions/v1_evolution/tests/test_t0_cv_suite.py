"""Fixed holdout identity, fold-local scaling and probability-level CV reports."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from mri_vla_jepa import t0_cv_suite as suite
from mri_vla_jepa.data import RawMRIStore, make_raw_synthetic
from mri_vla_jepa.io import digest, read_json, write_json
from mri_vla_jepa.train_config import load_config
from mri_vla_jepa.training import _scores


@pytest.fixture
def spec(tmp_path):
    source = make_raw_synthetic(tmp_path / "source", n_train=40, n_val=10, image_shape=(4, 4, 4))
    value = read_json(source)
    for i, patient in enumerate(value["patients"]):
        patient["patient_key"] = f"ISPY2-{10000 + i}"
        # Forty development patients: ten in every pCR x visit stratum.
        if (i // 2) % 2:
            patient["visits"][3] = None
    write_json(source, value)
    codec = tmp_path / "codec.pt"
    codec.write_bytes(b"synthetic-codec-identity-only-no-training")
    reference = deepcopy(value)
    for i, patient in enumerate(reference["patients"]):
        patient["split"] = "test" if i < 10 else "val" if i < 20 else "train"
        for visit in patient["visits"]:
            if visit:
                visit["image"] = str((source.parent / visit["image"]).resolve())
    reference_path = tmp_path / "reference.json"
    write_json(reference_path, reference)
    audit_path = tmp_path / "reference_audit.json"
    write_json(audit_path, {"seed": 17,
        "source_manifest": {"sha256": digest(source)}, "codec": {"sha256": digest(codec)},
        "patients": {s: [p["patient_key"] for p in reference["patients"] if p["split"] == s]
                     for s in ("train", "val", "test")},
        "counts": {s: sum(p["split"] == s for p in reference["patients"]) for s in ("train", "val", "test")}})
    formal = yaml.safe_load((Path(__file__).parents[1] / "configs/t0_fivefold_8gpu.yaml").read_text())
    value = deepcopy(formal)
    value["suite_name"] = "synthetic_cv"
    value["paths"] = {"source_manifest": str(source), "reference_split_manifest": str(reference_path),
        "reference_split_audit": str(audit_path), "codec": str(codec), "output_root": str(tmp_path / "outputs")}
    value.pop("input_sha256")
    value["cv"].update(expected_patients=50, test_size=10, expected_test_positives=5,
        expected_development_positives=20,
        expected_fold_counts=[{"train": 32, "train_positive": 16, "val": 8, "val_positive": 4} for _ in range(5)])
    value["training"].update(allow_synthetic=True, image_shape=[4, 4, 4], encoder_checkpoint=str(codec),
        device="cpu", precision="fp32", pin_memory=False, non_blocking_transfer=False)
    value["report"]["bootstrap"] = 12
    config_path = tmp_path / "suite.yaml"
    config_path.write_text(yaml.safe_dump(value, sort_keys=False))
    return suite.load_suite_config(config_path)


def _prepare(spec):
    return suite.prepare_suite(spec, spec["output_dir"])


def _write_evaluations(state, *, complete=False):
    for job in state["jobs"]:
        manifest = read_json(job["manifest"])
        for split in ("val", "test"):
            rows = []
            for patient in manifest["patients"]:
                if patient["split"] != split:
                    continue
                label = patient["target"]["pcr"]
                # Two good folds and three reversed folds: mean probability
                # predicts everyone wrong, while mean fold F1 is 0.4.
                correct = job["fold_id"] in (1, 2)
                probability = (.9 if label else .1) if correct else (.1 if label else .9)
                rows.append({"patient_key": patient["patient_key"], "landmark": 0,
                    "pcr": label, "pcr_probability": probability})
            report = {"split": split, "landmarks": "t0", "checkpoint_epoch": 11,
                **_scores(rows, "t0", selection_metric="f1", f1_threshold=.5), "rows": rows}
            write_json(Path(job["output"]) / f"evaluation_best_{split}.json", report)
            if split == "val":
                write_json(Path(job["output"]) / "evaluation_last_val.json", dict(report, checkpoint_epoch=61))
        if complete:
            job["status"] = "completed"


def test_reproducible_paired_folds_and_train_only_standardization(spec):
    source_before = Path(spec["paths"]["source_manifest"]).read_bytes()
    state = _prepare(spec)
    assert _prepare(spec) == state
    assert not (Path(spec["output_dir"]) / "matrix.json").exists()
    assert len(state["jobs"]) == 15
    assert {j["seed"] for j in state["jobs"]} == {17}
    test_sets, validation_sets = [], []
    for fold_id in range(1, 6):
        jobs = [j for j in state["jobs"] if j["fold_id"] == fold_id]
        assert len({j["manifest"] for j in jobs}) == 1
        manifest = read_json(jobs[0]["manifest"])
        sets = {s: {p["patient_key"] for p in manifest["patients"] if p["split"] == s} for s in ("train", "val", "test")}
        assert {s: len(x) for s, x in sets.items()} == {"train": 32, "val": 8, "test": 10}
        assert not sets["train"] & sets["val"]
        assert not (sets["train"] | sets["val"]) & sets["test"]
        test_sets.append(sets["test"])
        validation_sets.append(sets["val"])
        store = RawMRIStore(jobs[0]["manifest"], (4, 4, 4), allow_synthetic=True)
        ages = [p["clinical"]["values"][0] for p in manifest["patients"] if p["split"] == "train"]
        normalization = read_json(state["folds"][str(fold_id)]["normalization"])
        assert normalization == store.normalization_state()
        assert normalization["clinical_mean"][0] == pytest.approx(np.mean(ages))
        assert normalization["clinical_std"][0] == pytest.approx(np.std(ages))
        assert normalization["clinical_counts"] == [32] * 4
        assert normalization["clinical_mean"][1:] == [0., 0., 0.]
        assert normalization["clinical_std"][1:] == [1., 1., 1.]
    assert all(ids == test_sets[0] for ids in test_sets)
    assert len(set.union(*validation_sets)) == sum(map(len, validation_sets)) == 40
    assert Path(spec["paths"]["source_manifest"]).read_bytes() == source_before
    for job in state["jobs"]:
        cfg = load_config(job["config"])
        variant = next(v for v in suite.VARIANTS if v["name"] == job["variant"])
        assert cfg.jepa_weight == variant["jepa_weight"]
        assert cfg.model.pcr_readout == variant["pcr_readout"]
        assert cfg.model.spatial_readout == variant["spatial_readout"]
        assert cfg.model.enable_world_model is True
        assert cfg.lr_scheduler == "warmup_cosine" and cfg.weight_decay == .05
        assert cfg.model.ema_decay == .999 and cfg.model.dropout == .3


def test_relative_paths_and_yaml_drift_rejected(spec):
    path = Path(spec["_config_file"])
    raw = yaml.safe_load(path.read_text())
    raw["paths"]["codec"] = "codec.pt"
    raw["training"]["encoder_checkpoint"] = "codec.pt"
    path.write_text(yaml.safe_dump(raw))
    parsed = suite.load_suite_config(path)
    assert parsed["paths"]["codec"] == str(path.parent / "codec.pt")
    assert isinstance(parsed["output_dir"], str)
    json.dumps(parsed)
    state = _prepare(parsed)
    path.write_text(path.read_text() + "\n# changed after preparation\n")
    with pytest.raises(ValueError, match="identity changed"):
        suite.validate_inputs(state)
    with pytest.raises(ValueError, match="changed after loading"):
        _prepare(parsed)


@pytest.mark.parametrize("target", ["codec", "config", "manifest", "data", "fold_assignment", "variant_config"])
def test_input_identity_and_assignment_changes_rejected(spec, target):
    state = _prepare(spec)
    job = state["jobs"][0]
    if target == "codec":
        Path(state["codec"]).write_bytes(b"changed")
    elif target in {"config", "manifest"}:
        path = Path(job[target])
        path.write_text(path.read_text() + "\n")
    elif target == "data":
        path = Path(read_json(state["data_identity"])["assets"][0]["path"])
        path.write_bytes(path.read_bytes() + b"changed")
    elif target == "fold_assignment":
        job["fold_id"] = 2
    else:
        other = state["jobs"][1]
        for field in ("config", "config_sha256", "config_digest"):
            job[field] = other[field]
    with pytest.raises(ValueError):
        suite.validate_inputs(state)


def test_reference_labels_test_ids_and_fold_expectations_cannot_drift(spec):
    changed = deepcopy(spec)
    changed["cv"]["expected_fold_counts"][0]["val_positive"] = 3
    with pytest.raises(ValueError, match="counts differ"):
        _prepare(changed)
    ref_path = Path(spec["paths"]["reference_split_manifest"])
    ref = read_json(ref_path)
    ref["patients"][0]["target"]["pcr"] = 1 - ref["patients"][0]["target"]["pcr"]
    write_json(ref_path, ref)
    with pytest.raises(ValueError, match="changed patient data"):
        _prepare(spec)


def test_test_hidden_until_all_jobs_completed_even_if_test_files_exist(spec, monkeypatch):
    state = _prepare(spec)
    _write_evaluations(state)
    original = suite.validate_classification_report
    def forbid_test(path, **kwargs):
        assert kwargs["split"] != "test", "Test file was read before completion"
        return original(path, **kwargs)
    monkeypatch.setattr(suite, "validate_classification_report", forbid_test)
    report = suite.write_reports(spec["output_dir"], state, bootstrap=0)
    assert report["test_visible"] is False
    assert report["oof_rows"] == 3 * 40
    assert report["test_ensemble_rows"] == 0
    assert all(row["status"] == "withheld" and row["f1"] is None for row in report["fold_metrics"] if row["split"] == "test")
    public = Path(spec["output_dir"]) / "reports"
    assert read_json(public / "test_ensemble_predictions.json")["rows"] == []
    assert read_json(public / "fold_test_predictions.json")["rows"] == []
    assert "evaluation_best_test.json" not in (public / "report.html").read_text()


def test_oof_coverage_probability_ensemble_and_stratified_paired_ci(spec):
    state = _prepare(spec)
    _write_evaluations(state, complete=True)
    report = suite.write_reports(spec["output_dir"], state)
    assert report["test_visible"]
    assert report["oof_rows"] == 120 and report["test_ensemble_rows"] == 30
    public = Path(spec["output_dir"]) / "reports"
    rows = read_json(public / "oof_predictions.json")["rows"]
    for name in [v["name"] for v in suite.VARIANTS]:
        selected = [r for r in rows if r["variant"] == name]
        assert len({r["patient_key"] for r in selected}) == 40
    ensemble = read_json(public / "test_ensemble_predictions.json")["rows"]
    assert all(r["pcr_probability"] == pytest.approx(sum(r["fold_probabilities"]) / 5) for r in ensemble)
    for name in [v["name"] for v in suite.VARIANTS]:
        metric = next(r for r in report["aggregate_metrics"] if r["variant"] == name and r["population"] == "test_ensemble")
        mean_fold = next(r for r in report["fold_summary"] if r["variant"] == name and r["split"] == "test")
        assert metric["f1"] == 0 and mean_fold["f1_mean"] == pytest.approx(.4)
        assert metric["bootstrap"]["repetitions"] == 12
        assert set(metric["bootstrap"]["intervals"]) == set(suite.REPORT_METRICS)
        assert metric["predicted_positive_fraction"] == (metric["tp"] + metric["fp"]) / metric["n"]
        assert mean_fold["predicted_positive_fraction_mean"] == .5
        assert mean_fold["predicted_positive_fraction_std"] == 0
    assert all(r["predicted_positive_fraction"] == (r["tp"] + r["fp"]) / r["n"] for r in report["fold_metrics"])
    assert len(read_json(public / "fold_test_predictions.json")["rows"]) == 3 * 5 * 10
    assert "predicted_positive_fraction" in (public / "aggregate_metrics.csv").read_text()
    assert "predicted_positive_fraction" in (public / "report.html").read_text()
    assert "f1_ci_lower" in (public / "report.html").read_text()
    for paired in report["paired_bootstrap_differences"].values():
        for metrics in paired.values():
            assert all(ci["lower"] == 0 and ci["upper"] == 0 for ci in metrics.values())
    assert suite.write_reports(spec["output_dir"], state) == report


def test_missing_duplicate_or_corrupt_predictions_fail_before_reporting(spec):
    state = _prepare(spec)
    _write_evaluations(state, complete=True)
    path = Path(state["jobs"][0]["output"]) / "evaluation_best_val.json"
    original = read_json(path)
    bad = deepcopy(original)
    bad["rows"].append(bad["rows"][0])
    write_json(path, bad)
    with pytest.raises(ValueError, match="Duplicate"):
        suite.write_reports(spec["output_dir"], state, bootstrap=0)
    write_json(path, original)
    bad = deepcopy(original)
    bad["rows"] = bad["rows"][:-1]
    write_json(path, bad)
    with pytest.raises(ValueError, match="coverage"):
        suite.write_reports(spec["output_dir"], state, bootstrap=0)
    write_json(path, original)
    bad = deepcopy(original)
    bad["metrics"]["f1"] += .01
    write_json(path, bad)
    with pytest.raises(ValueError, match="metric differs"):
        suite.write_reports(spec["output_dir"], state, bootstrap=0)


def test_formal_configuration_is_pinned_and_json_serializable():
    path = Path(__file__).parents[1] / "configs/t0_fivefold_8gpu.yaml"
    spec = suite.load_suite_config(path)
    assert spec["output_dir"] == "/projects/xts0030/public/cxx/t0_cv5_combined_v1"
    json.dumps(spec)
    assert spec["cv"]["expected_fold_counts"] == list(suite.OFFICIAL_FOLDS)
    changed = deepcopy(spec)
    changed["training"]["weight_decay"] = .01
    with pytest.raises(ValueError, match="weight_decay"):
        suite._validate_spec(changed)
    changed = deepcopy(spec)
    changed["cv"]["expected_fold_counts"][0]["val_positive"] = 45
    with pytest.raises(ValueError, match="fold counts"):
        suite._validate_spec(changed)


def test_same_best_checkpoint_required_for_val_and_test(spec):
    state = _prepare(spec)
    _write_evaluations(state, complete=True)
    path = Path(state["jobs"][0]["output"]) / "evaluation_best_test.json"
    report = read_json(path)
    report["checkpoint_epoch"] += 1
    write_json(path, report)
    with pytest.raises(ValueError, match="same best checkpoint"):
        suite.write_reports(spec["output_dir"], state, bootstrap=0)

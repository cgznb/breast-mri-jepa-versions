"""Replay saved results with known probabilities; never start a GPU trainer."""
import csv
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from response_jepa.metrics import task_summary, world_summary
from response_jepa.provenance import source_snapshot
from response_jepa.synthetic import small_config


SCRIPT = Path(__file__).resolve().parents[1]/"scripts"/"analyze_joint_suite.py"
spec = importlib.util.spec_from_file_location("joint_suite_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def dump(path, value):
    path.write_text(json.dumps(value, default=lambda item: item.item()), encoding="utf-8")


def csv_dump(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_run(root, profile="dynamic", seed=17):
    name = f"joint_{profile}_seed{seed}"
    directory = root/name
    directory.mkdir(parents=True)
    cfg = small_config("joint")
    cfg.train.profile, cfg.train.seed = profile, seed
    cfg.train.prefix_sampling = "all_prefixes" if profile == "dynamic" else "balanced_cycle"
    cfg.train.batch_size, cfg.train.accumulation = 16, 1
    cfg.train.epochs, cfg.train.patience, cfg.train.min_delta = 3, 2, .01
    config = cfg.to_dict()
    # Two prefix-rich and two prefix-poor patients, with both labels at T0/T1.
    patients = [{"patient_key": f"PRIVATE_PATIENT_{i}", "split": "val",
                 "target": {"pcr": i%2},
                 "visits": [{"stage": j, "available_at": j} if j < (2 if i < 2 else 1) else None
                            for j in range(4)]} for i in range(4)]
    fingerprint = {"manifest": {"synthetic": True, "patients": patients}}
    normalization = {"mean": [0, 0, 0, 0], "std": [1, 1, 1, 1]}
    source, environment = source_snapshot(), {"runtime": "synthetic"}
    metadata = {"config": config, "source_identity": source, "environment": environment,
                "data_fingerprint": fingerprint, "normalization": normalization}
    history, selected_rows, selected_metrics, selected_world_rows = [], None, None, None
    for epoch, confidence in enumerate((.8, .801, .7), 1):
        rows = []
        for i in range(4):
            for stage in ([0, 1] if profile == "dynamic" and i < 2 else [0]):
                # At T1 use a worse confidence, creating patient-equal vs pooled differences.
                value = confidence-.1*stage
                probability = value if i%2 else 1-value
                rows.append({"patient_key": f"PRIVATE_PATIENT_{i}", "landmark": stage,
                             "label": float(i%2), "fused": probability, "direct": probability,
                             "forecast": probability})
        origin = {"mode": "ema", "run": str(directory), "phase": "joint", "seed": seed,
                  "epoch": epoch, "updates": epoch*2}
        world_rows = [{"patient_key": f"PRIVATE_PATIENT_{i}", "landmark": 0, "target_stage": 1,
                       "horizon": 1, "complete_chain": True, "free_l1": float(i+1), "copy_l1": 2*(i+1.)}
                      for i in range(2)]
        classification = task_summary(rows)
        metrics = {"classification": classification, "world": world_summary(world_rows),
                   "target_coordinate_origin": origin, "world_output_mode": "learned",
                   "task_forecast_source": "predicted_terminal",
                   "selection": classification["fused"]["patient_equal"]["nll"],
                   "selection_name": "patient_equal_pcr_nll"}
        score = metrics["selection"] if epoch == 1 else history[0]["best_score"]
        history.append({"epoch": epoch, "updates": epoch*2, "bad_epochs": epoch-1,
                        "best_score": score, "validation": metrics,
                        "train": {"fused": .4-.1*epoch}, "seconds": 1.0})
        state = {"schema": analysis.SCHEMA, **metadata, "epoch": epoch, "updates": epoch*2,
                 "history": deepcopy(history), "best_score": score, "bad_epochs": epoch-1,
                 "target_coordinate_origin": origin, "model": {"weight": torch.tensor([epoch])}}
        if epoch == 1:
            torch.save(state, directory/"best.pt")
            selected_rows, selected_metrics, selected_world_rows = rows, metrics, world_rows
        if epoch == 3:
            torch.save(state, directory/"last.pt")
    dump(directory/"config.json", config)
    dump(directory/"provenance.json", {key: metadata[key] for key in metadata if key != "config"})
    dump(directory/"history.json", history)
    dump(directory/"metrics_best.json", selected_metrics)
    csv_dump(directory/"predictions_val.csv", selected_rows)
    csv_dump(directory/"world_val.csv", selected_world_rows)
    return directory


def suite_status(root, rows):
    dump(root/"status.json", {"schema": "joint_suite_queue_v1", "runs": rows})


def completed(directory, profile, seed):
    return {"name": directory.name, "output": str(directory), "profile": profile,
            "seed": seed, "state": "completed"}


def test_known_probability_replay_uses_patient_equal_and_honors_min_delta(tmp_path):
    directory = make_run(tmp_path)
    suite_status(tmp_path, [completed(directory, "dynamic", 17)])
    report = analysis.analyze_suite(tmp_path)
    run = next(run for run in report["runs"] if run["verified"])
    # Prefix-rich patients average T0 and T1 before each patient receives equal weight.
    expected = ((-math.log(.8)-math.log(.7))/2-math.log(.8))/2
    assert run["selection"] == pytest.approx(expected)
    pooled = (-4*math.log(.8)-2*math.log(.7))/6
    assert run["selection"] != pytest.approx(pooled)
    assert run["best_epoch"] == 1  # Epoch 2 slightly improves, but less than min_delta.
    assert run["world"]["all"]["ratio"] == .5
    assert run["classification"]["fused"]["per_stage"]["T0"]["auroc"] == 1
    assert run["classification"]["fused"]["per_stage"]["T0"]["auprc"] == 1
    assert run["best_last"]["selection_nll"] > 0
    assert run["best_last"]["train_fused_bce"] < 0
    assert report["partial"] and not report["complete"]


def test_ready_artifacts_while_running_are_excluded_from_final_metrics(tmp_path):
    directory = make_run(tmp_path)
    suite_status(tmp_path, [{**completed(directory, "dynamic", 17), "state": "running"}])
    report = analysis.analyze_suite(tmp_path)
    assert report["verified_completed_runs"] == 0
    assert report["per_seed"] == []
    run = next(r for r in report["runs"] if r["name"] == directory.name)
    assert run["progress"]["latest_epoch"] == 3
    assert report["selection_aggregate"][1]["mean"] is None


@pytest.mark.parametrize("artifact,mutate,reason", [
    ("best.pt", lambda obj: obj["normalization"].update({"std": [2, 2, 2, 2]}), "normalization"),
    ("last.pt", lambda obj: obj.update({"epoch": 2}), "epochs"),
    ("best.pt", lambda obj: obj["source_identity"].update({"schema": "wrong"}), "source_identity"),
    ("last.pt", lambda obj: obj["data_fingerprint"]["manifest"].update({"extra": True}), "data_fingerprint"),
    ("history.json", lambda obj: obj[1].update({"bad_epochs": 0}), "early-stopping"),
    ("metrics_best.json", lambda obj: obj.update({"selection": .001}), "best.validation"),
])
def test_inconsistent_saved_results_are_rejected(tmp_path, artifact, mutate, reason):
    directory = make_run(tmp_path)
    path = directory/artifact
    value = torch.load(path, weights_only=False) if path.suffix == ".pt" else analysis.read_json(path)
    mutate(value)
    if path.suffix == ".pt":
        torch.save(value, path)
    else:
        dump(path, value)
    suite_status(tmp_path, [completed(directory, "dynamic", 17)])
    report = analysis.analyze_suite(tmp_path)
    assert report["verified_completed_runs"] == 0
    assert reason in report["errors"][0]


def test_swapped_probability_rows_fail_independent_replay(tmp_path):
    directory = make_run(tmp_path)
    rows = analysis.read_predictions(directory/"predictions_val.csv")
    for row in rows:
        row["fused"] = .5
    csv_dump(directory/"predictions_val.csv", rows)
    suite_status(tmp_path, [completed(directory, "dynamic", 17)])
    report = analysis.analyze_suite(tmp_path)
    assert "prediction_replay" in report["errors"][0]
    assert report["per_seed"] == []


def test_extra_or_duplicate_patient_prefix_predictions_are_rejected(tmp_path):
    directory = make_run(tmp_path)
    rows = analysis.read_predictions(directory/"predictions_val.csv")
    rows.append(rows[0])
    csv_dump(directory/"predictions_val.csv", rows)
    suite_status(tmp_path, [completed(directory, "dynamic", 17)])
    assert "Duplicate patient-prefix" in analysis.analyze_suite(tmp_path)["errors"][0]


def test_complete_six_run_aggregates_are_sample_std_and_private(tmp_path):
    entries = []
    for profile in analysis.PROFILES:
        for seed in analysis.SEEDS:
            directory = make_run(tmp_path, profile, seed)
            entries.append(completed(directory, profile, seed))
    suite_status(tmp_path, entries)
    report = analysis.analyze_suite(tmp_path)
    assert report["complete"] and not report["partial"]
    assert report["verified_completed_runs"] == 6
    assert all(row["seeds"] == 3 and row["sample_std"] == 0 for row in report["aggregate"])
    assert analysis.summarize_values([1, 2, 3]) == {"seeds": 3, "mean": 2, "sample_std": 1}
    output = tmp_path/"report"
    analysis.write_report(report, output)
    for path in output.iterdir():
        text = path.read_text()
        assert "PRIVATE_PATIENT" not in text
        assert "sha256" not in text and "data_fingerprint" not in text
    assert "全部完成" in (output/"README_ZH.md").read_text()


def test_cross_seed_data_mismatch_withholds_comparison(tmp_path):
    entries = []
    for seed in (17, 43):
        directory = make_run(tmp_path, "dynamic", seed)
        if seed == 43:
            for artifact in ("best.pt", "last.pt"):
                value = torch.load(directory/artifact, weights_only=False)
                value["data_fingerprint"]["manifest"]["note"] = "different cohort identity"
                torch.save(value, directory/artifact)
            provenance = analysis.read_json(directory/"provenance.json")
            provenance["data_fingerprint"]["manifest"]["note"] = "different cohort identity"
            dump(directory/"provenance.json", provenance)
        entries.append(completed(directory, "dynamic", seed))
    suite_status(tmp_path, entries)
    report = analysis.analyze_suite(tmp_path)
    assert report["verified_completed_runs"] == 2
    assert not report["cross_run_identity_consistent"]
    assert report["aggregate"] == [] and report["selection_aggregate"] == []


def test_missing_suite_is_reported_partial_without_fabricated_metrics(tmp_path):
    report = analysis.analyze_suite(tmp_path)
    assert report["partial"] and report["verified_completed_runs"] == 0
    assert report["per_seed"] == [] and report["aggregate"] == []
    analysis.write_report(report, tmp_path/"reports")
    assert "部分结果" in (tmp_path/"reports"/"README_ZH.md").read_text()


def test_world_patient_identity_is_checked_not_only_aggregate_errors(tmp_path):
    directory = make_run(tmp_path)
    with (directory/"world_val.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["patient_key"] = "PRIVATE_PATIENT_OTHER"
    csv_dump(directory/"world_val.csv", rows)
    suite_status(tmp_path, [completed(directory, "dynamic", 17)])
    report = analysis.analyze_suite(tmp_path)
    assert "World target cohort" in report["errors"][0]


def test_same_seed_historical_reference_excludes_seed71_and_stage_average(tmp_path):
    entries = []
    for seed in analysis.SEEDS:
        directory = make_run(tmp_path, "dynamic", seed)
        entries.append(completed(directory, "dynamic", seed))
    suite_status(tmp_path, entries)
    report = analysis.analyze_suite(tmp_path)
    old_rows = []
    for row in report["per_seed"]:
        if row["branch"] != "fused" or row["seed"] == 71:
            continue
        old_rows.append({"experiment": "L1", "profile": row["profile"], "seed": row["seed"],
                         "stage": row["stage"], "n": row["n"], "positives": row["positives"],
                         **{metric: row[metric]+.01 if metric in {"nll", "brier"} else row[metric]-.01
                            for metric in analysis.METRICS}})
    baseline_path = tmp_path/"baseline.csv"
    csv_dump(baseline_path, old_rows)
    baseline = analysis.historical_reference(report, baseline_path)
    assert {r["seed"] for r in baseline["comparisons"]} == {17, 43}
    assert all(r["joint_better"] for r in baseline["comparisons"])
    assert all(row["same_two_seeds_available"] for row in baseline["same_seed_summary"])
    assert all(row["stage"] in {"T0", "T1"} for row in baseline["same_seed_summary"])
    report["historical_reference"] = baseline
    analysis.write_report(report, tmp_path/"report")
    assert "overall patient-equal NLL" in (tmp_path/"report"/"README_ZH.md").read_text()
    assert {r["checkpoint"] for r in report["world_per_seed"]} == {"best", "last"}
    old_rows[0]["positives"] += 1
    csv_dump(baseline_path, old_rows)
    with pytest.raises(ValueError, match="patient/positive counts"):
        analysis.historical_reference(report, baseline_path)

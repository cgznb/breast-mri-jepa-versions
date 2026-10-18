"""Saved update-run metadata and known-probability replay; no MRI inference."""
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml

from response_jepa.data import Store, canonical_key, planned_tasks
from response_jepa.engine import SCHEMA, make_loader, runtime_info, train
from response_jepa.metrics import task_summary, world_summary
from response_jepa.provenance import source_snapshot
from response_jepa.synthetic import small_config

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_overfit_suite.py"
spec = importlib.util.spec_from_file_location("overfit_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def dump(path, value):
    Path(path).write_text(json.dumps(value, default=lambda item: item.item()), encoding="utf-8")


def csv_dump(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_run(root, manifest, *, name="B4_dynamic_seed17", algorithm="B4", seed=17, profile="dynamic"):
    directory = root / name
    directory.mkdir(parents=True)
    cfg = small_config("joint")
    t = cfg.train
    t.profile, t.seed = profile, seed
    t.prefix_sampling = "all_prefixes" if profile == "dynamic" else "balanced_cycle"
    t.batch_size, t.accumulation, t.reference_batch_size = 16, 1, 16
    t.max_updates, t.warmup_updates, t.eval_every_updates, t.patience_updates = 6, 1, 2, 4
    t.min_delta = .01
    t.sample_weighting = "patient_equal" if algorithm in {"B1", "B4"} else "sample"
    cfg.model.forecast_detach = algorithm in {"B2", "B4"}
    cfg.loss.consistency = .1 if algorithm in {"B3", "B4"} else 0.
    cfg.validate()
    config_path = root / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    store = Store(manifest, t.image_shape, allow_synthetic=True)
    metadata = {"config": cfg.to_dict(), "source_identity": source_snapshot(),
                "environment": runtime_info(), "data_fingerprint": store.fingerprint(),
                "normalization": store.normalization}
    history = []
    for update, confidence in zip((0, 2, 4, 6), (.6, .8, .801, .7)):
        remaining, epoch, samples = update, 0, 0
        while True:
            loader = make_loader(store, cfg, planned_tasks(store, cfg, "train", epoch), training=True)
            sizes = [len(indices) for indices in loader.batch_sampler]
            if remaining <= len(sizes):
                samples += sum(sizes[:remaining])
                cursor = remaining
                break
            samples += sum(sizes)
            remaining -= len(sizes)
            epoch += 1
        origin = {"mode": "ema", "run": str(directory), "phase": "joint", "seed": seed,
                  "epoch": epoch, "updates": update}
        predictions, world_rows = [], []
        for index, stage in planned_tasks(store, cfg, "val"):
            patient = store.patients[index]
            key, label = canonical_key(patient["patient_key"]), patient["target"]["pcr"]
            correct = confidence-.05*stage
            probability = correct if label else 1-correct
            predictions.append({"patient_key": key, "landmark": stage, "label": label,
                                "direct": probability, "forecast": probability, "fused": probability})
            for target in range(stage+1, 4):
                world_rows.append({"patient_key": key, "landmark": stage, "target_stage": target,
                                   "horizon": target-stage, "complete_chain": True,
                                   "free_l1": 1+.1*target, "copy_l1": 2+.2*target})
        classification = task_summary(predictions)
        metrics = {"classification": classification, "world": world_summary(world_rows),
                   "selection": classification["fused"]["patient_equal"]["nll"],
                   "selection_name": "patient_equal_pcr_nll", "target_coordinate_origin": origin,
                   "world_output_mode": "learned", "task_forecast_source": "predicted_terminal"}
        selected_update = 0 if update == 0 else 2
        score = metrics["selection"] if update in {0, 2} else history[1]["best_score"]
        history.append({"epoch": epoch, "batch_cursor": cursor, "updates": update, "samples_seen": samples,
                        "training_patients_or_prefixes": samples-(history[-1]["samples_seen"] if history else 0),
                        "train": {} if update == 0 else {"fused": .6-.01*update},
                        "validation": metrics, "best_score": score, "best_update": selected_update,
                        "bad_updates": update-selected_update, "seconds": 1.})
        state = {"schema": SCHEMA, "runner": "optimizer_updates_v1", **metadata,
                 "epoch": epoch, "cursor": {"epoch": epoch, "batch": cursor}, "updates": update,
                 "samples_seen": samples, "best_score": score, "best_update": selected_update,
                 "bad_updates": update-selected_update, "history": deepcopy(history),
                 "target_coordinate_origin": origin, "interval_sums": {}, "interval_samples": 0,
                 "interval_updates": 0, "model": {"weight": torch.ones(1)}}
        if update == 2:
            torch.save(state, directory / "best.pt")
            dump(directory / "metrics_best.json", metrics)
            csv_dump(directory / "predictions_val.csv", predictions)
            csv_dump(directory / "world_val.csv", world_rows)
        if update == 6:
            torch.save(state, directory / "last.pt")
    dump(directory / "config.json", cfg.to_dict())
    dump(directory / "provenance.json", {key: value for key, value in metadata.items() if key != "config"})
    dump(directory / "history.json", history)
    dump(directory / "completion.json", {"state": "completed", "stop_reason": "max_updates", "updates": 6,
        "max_updates": 6, "best_update": 2, "best_score": history[-1]["best_score"],
        "samples_seen": samples, "bad_updates": 4, "output": str(directory)})
    return {"name": name, "config": str(config_path), "manifest": str(manifest),
            "output": str(directory), "reservation_mib": 100, "algorithm": algorithm,
            "profile": profile, "seed": seed}


def plan(root, runs):
    path = root / "plan.json"
    dump(path, {"runs": runs})
    return path


def test_complete_update_run_replays_probabilities_world_and_min_delta(cohort, tmp_path):
    run = make_run(tmp_path, cohort)
    report = analysis.analyze_suite(plan(tmp_path, [run]))
    assert report["complete"] and report["errors"] == []
    audited = report["runs"][0]
    assert audited["best_update"] == 2  # update4 improves less than min_delta
    assert audited["stop_update"] == 6
    assert audited["world"]["all"]["ratio"] == .5
    assert audited["classification"]["fused"]["per_stage"]["T0"]["auroc"] == 1
    assert audited["best_last"]["selection_nll"] > 0
    assert all(audited["checks"].values())


def test_saved_artifacts_without_completed_status_are_excluded(cohort, tmp_path):
    run = make_run(tmp_path, cohort)
    Path(run["output"], "completion.json").unlink()
    report = analysis.analyze_suite(plan(tmp_path, [run]))
    assert not report["complete"] and report["errors"] == []
    assert report["verified_completed_runs"] == 0 and report["per_seed"] == []


def drop_assessment(history):
    removed = history.pop(2)
    history[2]["training_patients_or_prefixes"] += removed["training_patients_or_prefixes"]


@pytest.mark.parametrize("artifact,mutation,reason", [
    ("best.pt", lambda value: value["normalization"]["mean"].__setitem__(0, 123.), "normalization"),
    ("last.pt", lambda value: value.update({"best_update": 4}), "best-update"),
    ("best.pt", lambda value: value["source_identity"].update({"schema": "altered"}), "source"),
    ("last.pt", lambda value: value["environment"].update({"python": "other"}), "environment"),
    ("last.pt", lambda value: value["cursor"].update({"batch": 999}), "cursor"),
    ("history.json", lambda value: value[0].update({"updates": 1}), "initial"),
    ("history.json", lambda value: value[1].update({"samples_seen": 999}), "count"),
    ("history.json", drop_assessment, "assessment"),
    ("metrics_best.json", lambda value: value.update({"selection": .001}), "best.validation"),
    ("completion.json", lambda value: value.update({"updates": 5}), "Completion"),
])
def test_inconsistent_artifacts_never_enter_aggregate(cohort, tmp_path, artifact, mutation, reason):
    run = make_run(tmp_path, cohort)
    path = Path(run["output"], artifact)
    value = torch.load(path, weights_only=False) if path.suffix == ".pt" else analysis.read_json(path)
    mutation(value)
    torch.save(value, path) if path.suffix == ".pt" else dump(path, value)
    report = analysis.analyze_suite(plan(tmp_path, [run]))
    assert report["verified_completed_runs"] == 0 and not report["complete"]
    assert reason in report["errors"][0]
    assert report["per_seed"] == []


def test_changed_prediction_values_fail_metric_replay(cohort, tmp_path):
    run = make_run(tmp_path, cohort)
    path = Path(run["output"], "predictions_val.csv")
    rows = analysis._helpers.read_predictions(path)
    rows[0]["fused"] = .5
    csv_dump(path, rows)
    report = analysis.analyze_suite(plan(tmp_path, [run]))
    assert "prediction_replay" in report["errors"][0]


def test_duplicate_prediction_and_missing_world_rows_are_rejected(cohort, tmp_path):
    run = make_run(tmp_path, cohort)
    path = Path(run["output"], "predictions_val.csv")
    rows = analysis._helpers.read_predictions(path)
    csv_dump(path, rows + rows[:1])
    assert "Duplicate patient-prefix" in analysis.analyze_suite(plan(tmp_path, [run]))["errors"][0]
    csv_dump(path, rows)
    world_path = Path(run["output"], "world_val.csv")
    with world_path.open(newline="") as handle:
        world = list(csv.DictReader(handle))
    csv_dump(world_path, world[1:])
    assert "World target cohort" in analysis.analyze_suite(plan(tmp_path, [run]))["errors"][0]


def test_current_data_metadata_changes_are_rejected_without_pixel_reads(cohort, tmp_path, monkeypatch):
    run = make_run(tmp_path, cohort)
    def forbidden(*args, **kwargs):
        raise AssertionError("Analyzer attempted MRI pixel access")
    monkeypatch.setattr(Store, "read_image", forbidden)
    report = analysis.analyze_suite(plan(tmp_path, [run]))
    assert report["complete"]
    manifest = analysis.read_json(cohort)
    manifest["patients"][0]["clinical"]["values"][0] += 1
    dump(cohort, manifest)
    report = analysis.analyze_suite(plan(tmp_path, [run]))
    assert "data_fingerprint" in report["errors"][0]


def test_aggregate_outputs_are_private_and_partition_aware(cohort, tmp_path):
    one = make_run(tmp_path, cohort)
    two = make_run(tmp_path, cohort, name="B4_dynamic_seed43", seed=43)
    three = make_run(tmp_path, cohort, name="B0_dynamic_seed17", algorithm="B0")
    report = analysis.analyze_suite(plan(tmp_path, [one, two, three]))
    assert report["complete"]
    groups = report["selection_seed_aggregate"]
    assert len(groups) == 2
    assert next(row for row in groups if row["algorithm"] == "B4")["seeds"] == 2
    assert next(row for row in groups if row["algorithm"] == "B4")["sample_std"] == 0
    output = tmp_path / "report"
    analysis.write_report(report, output)
    text = "\n".join(path.read_text() for path in output.iterdir())
    store = Store(cohort, (8, 16, 16), allow_synthetic=True)
    assert not any(patient["patient_key"] in text for patient in store.patients)
    assert "sha256" not in text.lower() and "checksum" not in text.lower()


def test_duplicate_same_partition_seed_is_rejected(cohort, tmp_path):
    first = make_run(tmp_path, cohort)
    second = make_run(tmp_path, cohort, name="duplicate")
    report = analysis.analyze_suite(plan(tmp_path, [first, second]))
    assert not report["complete"] and report["verified_completed_runs"] == 1
    assert "Duplicate algorithm/profile/seed" in report["errors"][0]


def test_actual_cpu_update_runner_artifacts_pass_replay(cohort, tmp_path, monkeypatch):
    # Prevent even CUDA RNG/context initialization during this CPU-only test.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = small_config("joint")
    cfg.train.profile = "dynamic"
    cfg.train.prefix_sampling = "all_prefixes"
    cfg.train.sample_weighting = "patient_equal"
    cfg.train.reference_batch_size = cfg.train.batch_size = 3
    cfg.train.accumulation = 1
    cfg.train.max_updates, cfg.train.warmup_updates = 3, 1
    cfg.train.eval_every_updates, cfg.train.patience_updates = 1, 10
    cfg.train.train_eval_patients = 0
    cfg.train.save_predictions = True
    cfg.model.forecast_detach = True
    cfg.model.dropout = .1
    cfg.loss.consistency = .1
    config_path = tmp_path / "actual.yaml"
    config_path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    output = tmp_path / "actual"
    train(cfg, cohort, output)
    entry = {"name": "actual", "config": str(config_path), "manifest": str(cohort), "output": str(output),
             "reservation_mib": 1, "algorithm": "B4", "profile": "dynamic", "seed": cfg.train.seed}
    report = analysis.analyze_suite(plan(tmp_path, [entry]))
    assert report["complete"], report["errors"]
    assert report["runs"][0]["stop_update"] == 3

"""Patient partitions, complete per-timepoint reporting and destructive retention guards."""
from copy import deepcopy
from pathlib import Path
import statistics

import pytest

from mri_vla_jepa import ablation_suite as suite
from mri_vla_jepa.data import RawMRIStore, make_raw_synthetic
from mri_vla_jepa.io import digest, read_json, write_json
from mri_vla_jepa.model_config import RawVLAJEPAConfig
from mri_vla_jepa.train_config import RawVLATrainConfig
from mri_vla_jepa.training import _scores


@pytest.fixture
def source(tmp_path):
    # Four strata with enough examples to preserve each in all partitions.
    manifest = make_raw_synthetic(tmp_path / "source", n_train=32, n_val=8,
                                  image_shape=(2, 2, 2))
    value = read_json(manifest)
    for i, patient in enumerate(value["patients"]):
        patient["patient_key"] = f"ISPY2-{1000 + i}"
        if (i // 2) % 2:
            patient["visits"][3] = None
    write_json(manifest, value)
    return manifest


def prepare_small(source, tmp_path):
    return suite.prepare_splits(source, tmp_path / "splits", expected_patients=40,
                                val_size=4, test_size=8)


def test_patient_splits_reproducible_disjoint_stratified_and_rebased(source, tmp_path):
    before = source.read_bytes()
    artifacts = prepare_small(source, tmp_path)
    assert source.read_bytes() == before
    assert prepare_small(source, tmp_path) == artifacts
    assignments = []
    for seed in suite.SEEDS:
        manifest = read_json(artifacts[str(seed)]["manifest"])
        audit = read_json(artifacts[str(seed)]["audit"])
        assert audit["counts"] == {"train": 28, "val": 4, "test": 8}
        partitions = {s: set(v) for s, v in audit["patients"].items()}
        assert not partitions["train"] & partitions["val"]
        assert not partitions["train"] & partitions["test"]
        assert not partitions["val"] & partitions["test"]
        assert len(set.union(*partitions.values())) == 40
        assert all(len(v) == 4 for v in audit["stratum_counts"].values())
        assert manifest["provenance"]["independence_scope"] == "downstream_only"
        assert "historical" in manifest["provenance"]["image_standardization"]
        assert all(Path(v["image"]).is_absolute() and Path(v["image"]).exists()
                   for p in manifest["patients"] for v in p["visits"] if v)
        assignments.append(partitions)
        store = RawMRIStore(artifacts[str(seed)]["manifest"], image_shape=(2, 2, 2), allow_synthetic=True)
        assert len(store.by_split["train"]) == 28
        # Clinical normalization is newly fitted, not copied from original train.
        values = [store.patients[i]["clinical"]["values"][0] for i in store.by_split["train"]]
        state = store.normalization_state()
        assert abs(state["clinical_mean"][0] - sum(values) / len(values)) < 1e-5
    assert assignments[0] != assignments[1]


def test_split_rejects_duplicate_ids_size_and_conflicting_reprepare(source, tmp_path):
    with pytest.raises(ValueError, match="cohort"):
        suite.prepare_splits(source, tmp_path / "bad", expected_patients=866)
    prepare_small(source, tmp_path)
    value = read_json(source)
    value["patients"][0]["clinical"]["values"][0] += 1
    write_json(source, value)
    with pytest.raises(ValueError, match="differs"):
        prepare_small(source, tmp_path)
    value["patients"][1]["patient_key"] = value["patients"][0]["patient_key"].replace("ISPY2-", "")
    write_json(source, value)
    with pytest.raises(ValueError, match="duplicate"):
        prepare_small(source, tmp_path)


@pytest.fixture
def base_config():
    model = RawVLAJEPAConfig(model_size="large", encoder_type="vq_encoder", pcr_readout="forecast",
                            fixed_transition_queries=True, world_residual=True, history_delta=True,
                            enable_flow=False)
    return RawVLATrainConfig(model=model).to_dict()


def test_eighteen_configs_keep_seven_ablations_and_add_two_t0_protocols(base_config):
    original = deepcopy(base_config)
    generated = suite.variant_configs(base_config)
    assert base_config == original
    assert len(generated) == len({name for name, _ in generated}) == 18
    assert {(name.rsplit("_seed", 1)[0], c.seed) for name, c in generated} == {
        (variant, seed) for variant in suite.VARIANTS for seed in suite.SEEDS}
    changes = {"warmup_cosine": ("lr_scheduler", "warmup_cosine"),
               "plateau": ("lr_scheduler", "plateau"), "weight_decay": ("weight_decay", .05),
               "dropout": ("model.dropout", .1), "ema": ("model.ema_decay", .999),
               "attention": ("model.spatial_readout", "attention"),
               "t0_delta": ("model.pcr_delta_reference", "t0")}
    normalized = []
    for name, cfg in generated:
        assert cfg.encoder_training == "finetune" and cfg.encoder_lr == cfg.lr == 1e-4
        assert cfg.batch_size == 16 and cfg.checkpoint_every == 38
        assert cfg.training_unit == "epochs" and cfg.max_epochs == 200
        assert cfg.selection_metric == "f1" and cfg.f1_threshold == .5
        assert cfg.model.model_size == "large" and cfg.model.mri_phases == "all"
        assert cfg.model.fixed_transition_queries and cfg.model.world_residual and cfg.model.history_delta
        assert not cfg.model.enable_flow
        value = cfg.to_dict()
        variant = name.rsplit("_seed", 1)[0]
        if variant in suite.T0_VARIANTS:
            assert cfg.landmarks == "t0" and cfg.lr_scheduler == "none"
            assert cfg.model.spatial_readout == "mean" and cfg.model.pcr_delta_reference == "current"
            assert cfg.jepa_weight == (0 if variant == "t0_direct" else .5)
            assert cfg.prediction_training == "teacher_forcing"
            assert cfg.rollout_weight == 0
            assert cfg.model.enable_world_model == (variant != "t0_direct")
            assert cfg.model.pcr_readout == ("zero_state" if variant == "t0_direct" else "forecast")
            value.update(landmarks="all_observed", jepa_weight=.5, seed=17, prediction_training="teacher_forcing")
            value["model"].update(enable_world_model=True, pcr_readout="forecast")
            normalized.append(value)
            continue
        assert cfg.landmarks == "all_observed" and cfg.model.pcr_readout == "forecast"
        assert cfg.model.enable_world_model
        key, expected = changes[variant]
        if key.startswith("model."):
            assert value["model"][key[6:]] == expected
            default = {"dropout": .3, "ema_decay": .995, "spatial_readout": "mean", "pcr_delta_reference": "current"}[key[6:]]
            value["model"][key[6:]] = default
        else:
            assert value[key] == expected
            value[key] = "none" if key == "lr_scheduler" else .01
        value["seed"] = 17
        normalized.append(value)
    assert all(value == normalized[0] for value in normalized)


def test_suite_locks_common_behavior_and_rejects_wrong_architecture(base_config):
    base_config["model"].update(fixed_transition_queries=False, world_residual=False,
                                history_delta=False, enable_flow=True, pcr_readout="direct")
    base_config.update(diagnostics_every_epochs=0, freeze_encoder_after_representation=True)
    configs = suite.variant_configs(base_config)
    assert all(c.model.fixed_transition_queries and c.model.world_residual and c.model.history_delta
               and not c.model.enable_flow and c.model.pcr_readout == ("zero_state" if name.startswith("t0_direct_") else "forecast")
               and c.diagnostics_every_epochs == 5 and not c.freeze_encoder_after_representation
               for name, c in configs)
    base_config["model"]["dim"] = 128
    with pytest.raises(ValueError, match="large template"):
        suite.variant_configs(base_config)


def classification(path, manifest, split, probability=.7, profile="all_observed"):
    rows = [{"patient_key": key, "landmark": stage, "pcr": label,
             "pcr_probability": probability, "pcr_spatial_attention_entropy": 2.,
             "pcr_spatial_effective_tokens": 7.389056}
            for (key, stage), label in suite._expected_rows(manifest, split, profile).items()]
    report = {"split": split, "landmarks": profile, "checkpoint_epoch": 3,
              **_scores(rows, profile, selection_metric="f1", f1_threshold=.5), "rows": rows}
    write_json(path, report)
    return report


@pytest.fixture
def evaluated_run(source, tmp_path):
    split = prepare_small(source, tmp_path)["17"]["manifest"]
    root = tmp_path / "suite"
    run = root / "runs" / "example"
    run.mkdir(parents=True)
    required = {}
    for part in ("val", "test"):
        path = run / f"evaluation_best_{part}.json"
        classification(path, split, part)
        required[part] = {"path": str(path), "kind": "classification", "split": part}
    weight = run / "best.pt"
    weight.write_bytes(b"generated-weight")
    temporary = run / "last.pt.tmp"
    temporary.write_bytes(b"atomic-incomplete-checkpoint")
    return root, run, split, required


def report_jobs(run, manifest, required, *, low_seed43=False):
    jobs = []
    for variant in suite.VARIANTS:
        for seed in suite.SEEDS:
            evaluations = {}
            for split in ("val", "test"):
                if variant in suite.T0_VARIANTS or (seed == 43 and split == "val" and low_seed43):
                    path = run / f"{variant}_seed{seed}_{split}.json"
                    classification(path, manifest, split, probability=.2 if seed == 43 and low_seed43 else .7,
                                   profile=suite.selection_cohort(variant))
                else:
                    path = required[split]["path"]
                evaluations[split] = str(path)
            jobs.append({"variant": variant, "seed": seed, "output": str(run), "manifest": manifest,
                         "status": "completed", "evaluations": evaluations})
    return jobs


def test_public_report_has_120_applicable_rows_and_24_na_and_hides_test(evaluated_run):
    root, run, manifest, required = evaluated_run
    write_json(run / "diagnostics_record.json", {"reports": [{"epoch": 5}]})
    for split in ("val", "test"):
        write_json(run / f"world_best_{split}.json", {"status": "evaluated", "split": split})
    jobs = report_jobs(run, manifest, required)
    report = suite.write_suite_reports(root / "reports", jobs)
    assert len(report["timepoints"]) == 144
    assert report["applicable_timepoint_rows"] == 120
    na = [r for r in report["timepoints"] if not r["applicable"]]
    assert len(na) == 24 and all(r["status"] == "not_applicable" and r["epoch"] is None for r in na)
    assert all(r["status"] == "withheld" and r["f1"] is None
               for r in report["timepoints"] if r["split"] == "test" and r["applicable"])
    document = (root / "reports/report.html").read_text()
    assert "<th>checkpoint</th>" in document and "<th>epoch</th>" in document
    assert "<th>pcr_spatial_attention_entropy_mean</th>" in document
    assert "<th>pcr_spatial_effective_tokens_mean</th>" in document
    assert 'href="../runs/example/diagnostics_record.json"' in document
    assert 'href="../runs/example/evaluation_best_val.json"' in document
    assert 'href="../runs/example/world_best_val.json"' in document
    assert "evaluation_best_test.json" not in document and "world_best_test.json" not in document
    report = suite.write_suite_reports(root / "reports", jobs, suite_complete=True)
    assert report["test_visible"] and all(r["status"] == "evaluated" for r in report["timepoints"] if r["applicable"])
    assert all(r["epoch"] == 3 for r in report["timepoints"] if r["applicable"])
    assert all(r["f1_std"] == 0 for r in report["seed_summary"] if r["status"] != "not_applicable")
    assert set(report["ranking"]["all_observed"]) == set(suite.LONGITUDINAL_VARIANTS)
    assert set(report["ranking"]["t0"]) == set(suite.T0_VARIANTS)
    assert len(report["t0_comparison"]) == 18
    assert report["timepoints"][0]["pcr_spatial_attention_entropy_mean"] == 2.
    document = (root / "reports/report.html").read_text()
    assert 'href="../runs/example/evaluation_best_test.json"' in document
    assert 'href="../runs/example/world_best_test.json"' in document
    assert len((root / "reports/per_config_seed_timepoint.csv").read_text().splitlines()) == 145
    assert len((root / "reports/applicable_timepoints.csv").read_text().splitlines()) == 121
    jobs[0]["status"] = "running"
    assert not suite.write_suite_reports(root / "reports", jobs, suite_complete=True)["test_visible"]


def test_metric_or_prediction_corruption_is_rejected(evaluated_run):
    root, run, manifest, required = evaluated_run
    path = required["val"]["path"]
    report = read_json(path)
    report["metrics"]["f1"] = 0
    write_json(path, report)
    with pytest.raises(ValueError, match="metric differs"):
        suite.validate_classification_report(path, manifest=manifest)
    report = classification(path, manifest, "val")
    report["rows"].pop()
    write_json(path, report)
    with pytest.raises(ValueError, match="coverage"):
        suite.validate_classification_report(path, manifest=manifest)


def test_verified_cleanup_deletes_generated_only_and_is_idempotent(evaluated_run):
    root, run, manifest, required = evaluated_run
    codec = root.parent / "codec.pt"
    codec.write_bytes(b"original-codec")
    receipt = suite.complete_evaluation_receipt(run, root, required, manifest=manifest)
    assert {Path(c["path"]).name for c in receipt["checkpoints"]} == {"best.pt", "last.pt.tmp"}
    assert all(c["sha256"] for c in receipt["checkpoints"])
    result = suite.delete_evaluated_checkpoints(run, root, protected_paths=[codec])
    assert result["all_receipted_weights_absent"] and len(result["deleted"]) == 2
    assert codec.read_bytes() == b"original-codec"
    assert all(Path(v["path"]).exists() for v in required.values())
    assert suite.delete_evaluated_checkpoints(run, root) == result


@pytest.mark.parametrize("status", ["running", "queued", "evaluation_pending", "interrupted"])
def test_active_run_cleanup_and_receipt_forbidden(evaluated_run, status):
    root, run, manifest, required = evaluated_run
    with pytest.raises(ValueError, match="pending"):
        suite.complete_evaluation_receipt(run, root, required, manifest=manifest, status=status)
    with pytest.raises(ValueError, match="pending"):
        suite.delete_evaluated_checkpoints(run, root, status=status)
    assert (run / "best.pt").exists()


def test_cleanup_requires_complete_predictions_and_both_splits(evaluated_run):
    root, run, manifest, required = evaluated_run
    with pytest.raises(FileNotFoundError):
        suite.delete_evaluated_checkpoints(run, root)
    with pytest.raises(ValueError, match="Both validation and test"):
        suite.complete_evaluation_receipt(run, root, {"val": required["val"]}, manifest=manifest)
    report = read_json(required["test"]["path"])
    report["rows"] = report["rows"][:-1]
    write_json(required["test"]["path"], report)
    with pytest.raises(ValueError, match="coverage"):
        suite.complete_evaluation_receipt(run, root, required, manifest=manifest)
    assert (run / "best.pt").exists()


@pytest.mark.parametrize("mutation", ["output", "weight", "new_weight", "protected", "symlink"])
def test_cleanup_checks_all_identities_before_any_deletion(evaluated_run, mutation):
    root, run, manifest, required = evaluated_run
    suite.complete_evaluation_receipt(run, root, required, manifest=manifest)
    kwargs = {}
    if mutation == "output":
        with open(required["val"]["path"], "a") as handle:
            handle.write(" ")
    elif mutation == "weight":
        (run / "best.pt").write_bytes(b"changed")
    elif mutation == "new_weight":
        (run / "unreceipted.pt").write_bytes(b"new")
    elif mutation == "protected":
        kwargs["protected_paths"] = [run / "best.pt"]
    else:
        target = run / "best.pt"
        elsewhere = root.parent / "external.pt"
        target.rename(elsewhere)
        target.symlink_to(elsewhere)
    with pytest.raises(ValueError):
        suite.delete_evaluated_checkpoints(run, root, **kwargs)
    assert (run / "last.pt.tmp").exists()
    assert (run / "best.pt").exists()


def test_suite_boundary_and_original_codec_are_never_receipted(evaluated_run):
    root, run, manifest, required = evaluated_run
    with pytest.raises(ValueError, match="strictly inside"):
        suite.complete_evaluation_receipt(root.parent, root, required, manifest=manifest)
    codec = run / "codec.pt"
    codec.write_bytes(b"must-not-delete")
    with pytest.raises(ValueError, match="original codec"):
        suite.complete_evaluation_receipt(run, root, required, manifest=manifest)


def test_sample_standard_deviation_and_checkpoint_shared_by_all_timepoints(evaluated_run):
    root, run, manifest, required = evaluated_run
    jobs = report_jobs(run, manifest, required, low_seed43=True)
    report = suite.write_suite_reports(root / "reports", jobs)
    summary = next(r for r in report["seed_summary"] if r["variant"] == "warmup_cosine"
                   and r["split"] == "val" and r["timepoint"] == "overall")
    scores = [r["f1"] for r in report["overall"] if r["variant"] == "warmup_cosine" and r["split"] == "val"]
    assert summary["f1_std"] == pytest.approx(statistics.stdev(scores))
    assert summary["f1_std"] != pytest.approx(statistics.pstdev(scores))
    for variant in suite.VARIANTS:
        for seed in suite.SEEDS:
            rows = [r for r in report["timepoints"] if r["variant"] == variant and r["seed"] == seed
                    and r["split"] == "val" and r["applicable"]]
            assert len(rows) == (1 if variant in suite.T0_VARIANTS else 4)
            assert {(r["checkpoint"], r["epoch"]) for r in rows} == {("best", 3)}


def test_wrong_threshold_world_coverage_and_unsupported_kind_prevent_deletion(evaluated_run):
    root, run, manifest, required = evaluated_run
    validation = read_json(required["val"]["path"])
    validation["f1_threshold"] = .6
    write_json(required["val"]["path"], validation)
    with pytest.raises(ValueError, match="threshold"):
        suite.complete_evaluation_receipt(run, root, required, manifest=manifest)
    classification(required["val"]["path"], manifest, "val")
    required["world"] = {"path": str(run / "world.json"), "kind": "world", "split": "val"}
    expected = suite._expected_rows(manifest, "val")
    world = {"status": "evaluated", "split": "val", "patients": len({k for k, _ in expected}),
             "legal_prefixes": len(expected)}
    write_json(run / "world.json", world)
    suite.complete_evaluation_receipt(run, root, required, manifest=manifest)
    world["legal_prefixes"] -= 1
    write_json(run / "world.json", world)
    with pytest.raises(ValueError, match="World evaluation coverage"):
        suite.complete_evaluation_receipt(run, root, required, manifest=manifest)
    required["world"]["kind"] = "jsonl"
    with pytest.raises(ValueError, match="Unsupported"):
        suite.complete_evaluation_receipt(run, root, required, manifest=manifest)
    assert (run / "best.pt").exists()


def test_t0_coverage_excludes_later_visits_and_world_counts_only_t0(evaluated_run):
    root, run, manifest, required = evaluated_run
    for split in ("val", "test"):
        classification(required[split]["path"], manifest, split, profile="t0")
    for split, expected in (("val", 4), ("test", 8)):
        report = suite.validate_classification_report(required[split]["path"], manifest=manifest, profile="t0")
        assert len(report["rows"]) == expected and all(row["landmark"] == 0 for row in report["rows"])
    with pytest.raises(ValueError, match="cohort mismatch"):
        suite.validate_classification_report(required["val"]["path"], manifest=manifest, profile="all_observed")
    world_path = run / "world_best_val.json"
    write_json(world_path, {"status": "evaluated", "split": "val", "landmarks": "t0",
                            "patients": 4, "legal_prefixes": 4})
    required["world"] = {"path": str(world_path), "kind": "world", "split": "val"}
    suite.complete_evaluation_receipt(run, root, required, manifest=manifest)
    classification(required["val"]["path"], manifest, "val", profile="all_observed")
    wrong = read_json(required["val"]["path"])
    wrong["landmarks"] = "t0"
    write_json(required["val"]["path"], wrong)
    with pytest.raises(ValueError, match="later-visit"):
        suite.complete_evaluation_receipt(run, root, required, manifest=manifest)


def test_direct_world_na_must_be_explicit_before_checkpoint_cleanup(evaluated_run):
    root, run, manifest, required = evaluated_run
    for split in ("val", "test"):
        classification(required[split]["path"], manifest, split, profile="t0")
    world_path = run / "world_best_val.json"
    record = {"status": "not_applicable", "split": "val", "world_head_trained": False,
              "reason": "Direct T0 classifier has no world model"}
    write_json(world_path, record)
    required["world"] = {"path": str(world_path), "kind": "world", "split": "val", "applicable": False}
    suite.complete_evaluation_receipt(run, root, required, manifest=manifest)
    record.pop("reason")
    write_json(world_path, record)
    with pytest.raises(ValueError, match="explicit not_applicable"):
        suite.complete_evaluation_receipt(run, root, required, manifest=manifest)
    assert (run / "best.pt").exists()
    record["reason"] = "No world module"
    write_json(world_path, record)
    suite.complete_evaluation_receipt(run, root, required, manifest=manifest)
    assert suite.delete_evaluated_checkpoints(run, root)["all_receipted_weights_absent"]

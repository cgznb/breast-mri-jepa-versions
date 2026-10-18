from dataclasses import replace
from pathlib import Path
import json
import torch
import numpy as np
import pytest
from response_jepa.data import Store, collate_prefix, planned_tasks
from response_jepa.engine import stage_batches
from response_jepa.metrics import paired_bootstrap, world_summary, task_summary
from response_jepa.config import load_config, from_dict


def test_normalization_train_only(cohort):
    s = Store(cohort, [8, 16, 16], allow_synthetic=True)
    before = s.normalization
    m = json.loads(Path(cohort).read_text())
    for p in m["patients"]:
        if p["split"] != "train":
            p["clinical"]["values"][0] += 1000
    Path(cohort).write_text(json.dumps(m))
    after = Store(cohort, [8, 16, 16], allow_synthetic=True).normalization
    assert before == after


def test_inference_never_opens_future_files_or_labels(cohort):
    m = json.loads(Path(cohort).read_text())
    for p in m["patients"]:
        for j, v in enumerate(p["visits"]):
            if j and v is not None:
                v["image"] = f"nonexistent-{p['patient_key']}-{j}.npy"
        p["target"] = {"pcr": "INACCESSIBLE FOR INFERENCE"}
    Path(cohort).write_text(json.dumps(m))
    s = Store(cohort, [8, 16, 16], allow_synthetic=True)
    b = collate_prefix([s.sample(0, 0, future=False, labels=False)])
    b.inputs.validate()
    assert not b.targets.future_mask.any()
    assert not b.targets.label_mask.any()


def test_duplicate_patient_and_asset_rejected(cohort):
    m = json.loads(Path(cohort).read_text())
    m["patients"][1]["patient_key"] = m["patients"][0]["patient_key"]
    Path(cohort).write_text(json.dumps(m))
    with pytest.raises(ValueError, match="Duplicate"):
        Store(cohort, [8, 16, 16], allow_synthetic=True)


def test_missing_visit_not_compressed(cohort):
    s = Store(cohort, [8, 16, 16], allow_synthetic=True)
    assert s.patients[6]["visits"][1] is None
    b = collate_prefix([s.sample(6, 0)])
    assert b.targets.future_mask.tolist() == [[False, False, True, True]]


def test_masked_clinical_and_arm_availability(cohort):
    m = json.loads(Path(cohort).read_text())
    m["patients"][0]["clinical"]["known_at"][1] = 2
    m["patients"][0]["arm"]["known_at_stage"] = 2
    Path(cohort).write_text(json.dumps(m))
    s = Store(cohort, [8, 16, 16], allow_synthetic=True)
    b = collate_prefix([s.sample(0, 0)])
    assert b.inputs.arm_id[0] == 0 and b.inputs.arm_known_at[0] == -1
    assert not b.inputs.clinical_mask[0, 1]
    assert b.inputs.clinical[0, 1] == 0


def test_balanced_cycle_covers_every_landmark(cohort, config):
    s = Store(cohort, [8, 16, 16], allow_synthetic=True)
    seen = set()
    for e in range(4):
        tasks = planned_tasks(s, config, "train", e)
        assert len({i for i, _ in tasks}) == len(tasks)
        seen.update(j for i, j in tasks if i == 0)
    assert seen == {0, 1, 2, 3}


def test_singleton_batch_tail_not_dropped():
    tasks = [(i, 0) for i in range(9)] + [(9+i, 1) for i in range(4)]
    batches = stage_batches(tasks, 4)
    assert sorted(sum(batches, [])) == list(range(13))
    assert all(len({tasks[i][1] for i in b}) == 1 for b in batches)
    assert all(len(b) >= 2 for b in batches)


def test_patient_equal_score_not_visit_pooled():
    rows = [{"patient_key": "a", "landmark": j, "label": 1., "fused": .9, "direct": .9, "forecast": None}
            for j in range(4)]
    rows += [{"patient_key": "b", "landmark": 0, "label": 1., "fused": .1, "direct": .1, "forecast": None}]
    result = task_summary(rows)["fused"]["patient_equal"]["nll"]
    assert abs(result - (-np.log(.9)-np.log(.1))/2) < 1e-10


def test_world_ratio_uses_same_patients_and_ratio_of_means():
    rows = [{"patient_key": "a", "horizon": 1, "complete_chain": True, "free_l1": 1., "copy_l1": 2.},
            {"patient_key": "a", "horizon": 2, "complete_chain": True, "free_l1": 1., "copy_l1": 2.},
            {"patient_key": "b", "horizon": 1, "complete_chain": False, "free_l1": 3., "copy_l1": 3.}]
    r = world_summary(rows)
    assert abs(r["all"]["ratio"] - .8) < 1e-12
    assert r["complete_chain"]["patients"] == 1


def test_paired_metric_intersection_and_directions():
    a = [{"patient_key": str(i), "landmark": 0, "label": i%2, "fused": .5} for i in range(10)]
    b = [{"patient_key": str(i), "landmark": 1, "label": i%2, "fused": .9 if i%2 else .1} for i in range(8)]
    r = paired_bootstrap(a, b, stage_b=1, repetitions=20)
    assert r["common_patients"] == 8
    assert r["deltas"]["auroc"]["point"] == .5
    assert r["deltas"]["nll"]["point"] < 0


def test_unknown_configuration_keys_fail(config):
    d = config.to_dict()
    d["model"]["futur_detach"] = True
    with pytest.raises(ValueError, match="Unknown"):
        from_dict(d)


def test_all_shipped_configs_validate():
    root = Path(__file__).resolve().parents[1]/"configs"
    paths = list(root.rglob("*.yaml"))
    assert len(paths) >= 51
    for path in paths:
        load_config(path)


def test_suite_shares_a_and_b_for_matched_readouts(tmp_path, cohort):
    from response_jepa.experiments import plan_runs
    plan = plan_runs(cohort, tmp_path/"plan", "t0", ["full", "copy_readout", "rollout_aux"], [17], device="cpu")
    assert len(plan) == 5
    assert sum(r["phase"] == "A" for r in plan) == 1
    assert sum(r["phase"] == "B" for r in plan) == 1
    cs = [r for r in plan if r["phase"] == "C"]
    assert len({r["init"] for r in cs}) == 1
    full, copy = cs[0]["config"], cs[1]["config"]
    assert full["loss"] == copy["loss"]


def test_real_roi32_npz_and_original_cache_are_value_preserving(tmp_path):
    from response_jepa.data import CLINICAL, PHASES, ARMS, file_identity
    data = tmp_path/"roi32"
    data.mkdir()
    cache = data/"cache"
    cache.mkdir()
    image = np.zeros((3, 32, 128, 128), dtype=np.float32)
    image[:, 10, 30, 30] = [7.25, -2.5, 3.0]
    patients, entries = [], []
    for i, split in enumerate(("train", "val")):
        source = data/f"p{i}_T0.npz"
        np.savez_compressed(source, image=image)
        cached = cache/f"p{i}.npy"
        np.save(cached, image)
        entries.append({"source": file_identity(source), "cached": file_identity(cached)})
        g = {"source_stage": 0, "source_only": True, "roi_source": "T0", "orientation": "RAS",
             "resampling": "source_grid", "grid_id": f"g{i}"}
        patients.append({"patient_key": str(i), "split": split, "geometry": g,
                         "visits": [{"stage": 0, "available_at": 0, "grid_id": g["grid_id"], "image": source.name}, None, None, None],
                         "clinical": {"values": [40.+i, 0., 1., 1.], "known_at": [0, 0, 0, 0]},
                         "arm": {"label": ARMS[0], "known_at_stage": 0, "semantics": "assigned_arm_scenario"},
                         "target": {"pcr": i}})
    normalization = {"mean": 100., "std": 20., "fit_split": "train", "scope": "fixture",
                     "background_policy": "preserve_zero", "already_normalized": True,
                     "normalization_policy": "synthetic fixture only"}
    manifest = {"schema": "responsewm_raw_mri_v1", "synthetic": True, "phase_order": PHASES,
                "clinical_features": CLINICAL, "canonical_stages": [0, 1, 2, 3], "time_basis": "stage_index",
                "image_preprocessing": {"mode": "preprocessed_roi32"}, "image_normalization": normalization,
                "patients": patients}
    path = data/"manifest.json"
    path.write_text(json.dumps(manifest))
    index = {"schema": "registered_roi32_image_cache_v1", "shape": [3, 32, 128, 128], "dtype": "float32",
             "phase_order": PHASES, "image_preprocessing": {"mode": "preprocessed_roi32"},
             "image_normalization": normalization, "images": entries}
    (cache/"index.json").write_text(json.dumps(index))
    plain = Store(path, allow_synthetic=True)
    cached = Store(path, allow_synthetic=True, image_cache="cache")
    a = plain.sample(0, 0)[0].images[0]
    b = cached.sample(0, 0)[0].images[0]
    torch.testing.assert_close(a, torch.from_numpy(image), atol=0, rtol=0)
    assert torch.equal(a, b)
    assert float(a[0, 10, 30, 30]) == 7.25  # no (7.25-100)/20 second normalization


def test_logistic_baseline_does_not_evaluate_test_by_default(cohort, tmp_path):
    from response_jepa.baselines import tabular
    r = tabular(cohort, output=tmp_path/"tabular.json", include_arm=True,
                allow_synthetic=True, image_shape=(8, 16, 16))
    assert r["test"] is None
    assert (tmp_path/"tabular_val_predictions.csv").exists()
    assert not (tmp_path/"tabular_test_predictions.csv").exists()

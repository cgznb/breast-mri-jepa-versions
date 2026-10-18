import json
import os
from pathlib import Path
import pytest

from response_jepa.data import Store
from response_jepa.provenance import file_identity, record_metadata, source_snapshot


def test_source_snapshot_checks_exact_text_even_when_stat_metadata_is_preserved(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("value = 1\n")
    before, identity = source_snapshot(tmp_path), file_identity(path)
    path.write_text("value = 2\n")
    os.utime(path, ns=(path.stat().st_atime_ns, identity["mtime_ns"]))
    assert file_identity(path) == identity
    assert source_snapshot(tmp_path) != before
    assert before["files"] == {"module.py": "value = 1\n"}


def test_data_identity_retains_semantics_without_checksum_fields(cohort):
    store = Store(cohort, [8, 16, 16], allow_synthetic=True)
    before = store.fingerprint()
    assert before["manifest"] == store.manifest
    assert store.normalization["fit_patient_count"] == 8
    assert store.normalization["fit_patient_keys"] == before["partitions"]["train"]
    assert "fit_patient_hash" not in store.normalization
    manifest = json.loads(Path(cohort).read_text())
    manifest["patients"][0]["clinical"]["values"][0] += 1
    Path(cohort).write_text(json.dumps(manifest))
    changed = Store(cohort, [8, 16, 16], allow_synthetic=True)
    assert changed.fingerprint() != before
    assert changed.normalization != store.normalization


def test_legacy_checksum_fields_are_omitted_only_from_new_records():
    legacy = {"mean": 3., "provenance": {"sha256": "legacy identifier", "scope": "train"},
              "fit_patient_hash": "legacy identifier", "visits": [{"size": 8, "checksum": "legacy identifier"}]}
    recorded = record_metadata(legacy)
    assert recorded == {"mean": 3., "provenance": {"scope": "train"}, "visits": [{"size": 8}]}
    assert "fit_patient_hash" in legacy


def test_observed_stage_uses_actual_visit_not_availability_landmark(cohort):
    manifest = json.loads(Path(cohort).read_text())
    visits = manifest["patients"][0]["visits"]
    visits[1]["available_at"] = 2
    visits[2] = None
    Path(cohort).write_text(json.dumps(manifest))
    store = Store(cohort, [8, 16, 16], allow_synthetic=True)
    assert store.observed_stage(0, 1) == 0
    assert store.observed_stage(0, 2) == 1


def test_inference_accepts_legacy_normalization_metadata_but_rejects_changed_semantics(cohort, tmp_path):
    from response_jepa.engine import train, load_for_inference
    from response_jepa.synthetic import small_config
    manifest = json.loads(cohort.read_text())
    manifest["image_normalization"] = {"scope": "synthetic fixture", "sha256": "legacy identifier"}
    cohort.write_text(json.dumps(manifest))
    cfg = small_config("C")
    cfg.train.freeze_encoder = False
    cfg.train.world_eval = False
    cfg.loss.tf = cfg.loss.rollout = 0
    cfg.model.readout = "direct"
    output = tmp_path / "normalization_metadata"
    train(cfg, cohort, output)
    load_for_inference(output / "best.pt", cohort, allow_synthetic=True)
    manifest["image_normalization"]["scope"] = "different image scale"
    cohort.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="image normalization differs"):
        load_for_inference(output / "best.pt", cohort, allow_synthetic=True)

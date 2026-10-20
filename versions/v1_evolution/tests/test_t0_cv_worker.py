"""CPU contracts for the portable, separately scheduled T0-CV worker."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from mri_vla_jepa import t0_cv_worker as worker
from mri_vla_jepa.contracts import RawMRIInput, RawMRISupervision
from mri_vla_jepa.data import make_raw_synthetic
from mri_vla_jepa.io import digest, load_checkpoint, read_json, save_checkpoint, stable_hash, write_json
from mri_vla_jepa.model import RawVLAJEPA
from mri_vla_jepa.model_config import RawVLAJEPAConfig
from mri_vla_jepa.train_config import RawVLATrainConfig


@pytest.fixture(autouse=True)
def cpu_threads_and_constructor():
    previous_threads = torch.get_num_threads()
    previous_constructor = RawVLAJEPA.__init__
    torch.set_num_threads(1)
    torch.manual_seed(17043)
    yield
    RawVLAJEPA.__init__ = previous_constructor
    torch.set_num_threads(previous_threads)


@pytest.fixture
def model_cfg():
    return RawVLAJEPAConfig(dim=16, heads=2, token_grid=(1, 2, 2),
        encoder_depth=1, fusion_depth=1, predictor_depth=1, state_queries=2,
        pcr_queries=2, fixed_transition_queries=True, pcr_readout="forecast",
        history_delta=True, dropout=0.)


@pytest.fixture
def t0_batch():
    observed = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
    images = torch.zeros(2, 4, 3, 4, 8, 8)
    images[:, 0] = torch.randn(2, 3, 4, 8, 8)
    queries = ~observed
    inp = RawMRIInput(images, observed, torch.zeros(2, dtype=torch.long),
        torch.randn(2, 4), torch.ones(2, 4, dtype=torch.bool),
        torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.bool),
        torch.full((2,), -1, dtype=torch.long), queries)
    future = torch.full_like(images, float("nan"))
    future[queries] = torch.randn(6, 3, 4, 8, 8)
    sup = RawMRISupervision(future, queries, torch.tensor([1., 0.]),
                            torch.ones(2, dtype=torch.bool))
    return inp, sup


def options(jepa=0.):
    return dict(prediction_training="teacher_forcing", task_weight=1.,
        jepa_weight=jepa, rollout_weight=0., flow_weight=0.,
        reconstruction_weight=0., variance_weight=0., variance_definition="off")


def test_paired_shared_initialization_and_rng(model_cfg):
    worker.install_paired_readout_initialization()
    constructor = RawVLAJEPA.__init__
    worker.install_paired_readout_initialization()
    assert RawVLAJEPA.__init__ is constructor
    torch.manual_seed(841)
    forecast = RawVLAJEPA(replace(model_cfg, spatial_readout="attention"))
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(841)
    direct = RawVLAJEPA(replace(model_cfg, pcr_readout="direct"))
    assert torch.equal(expected_rng, torch.get_rng_state())
    shared = direct.state_dict()
    for name, value in forecast.state_dict().items():
        if name.startswith(("pcr_head.", "pcr_spatial_scorer.")):
            continue
        torch.testing.assert_close(value, shared[name], atol=0, rtol=0)
    assert direct.pcr_head[-1].in_features == model_cfg.dim
    assert forecast.pcr_head[-1].in_features == 3 * model_cfg.dim


def test_forecast_attention_is_active_and_normalized(model_cfg, t0_batch):
    worker.install_paired_readout_initialization()
    model = RawVLAJEPA(replace(model_cfg, spatial_readout="attention")).eval()
    inp, sup = t0_batch
    output = model(inp)
    weights = output["pcr_spatial_attention"]
    torch.testing.assert_close(weights, torch.full_like(weights, 1 / weights.shape[1]))
    torch.testing.assert_close(weights.sum(1), torch.ones(2))
    loss, _ = model.compute_loss(inp, sup, **options())
    loss.backward()
    assert model.pcr_spatial_scorer[-1].weight.grad.abs().sum() > 0


def test_direct_classification_does_not_use_future_supervision(model_cfg, t0_batch):
    worker.install_paired_readout_initialization()
    model = RawVLAJEPA(replace(model_cfg, pcr_readout="direct")).eval()
    inp, sup = t0_batch
    changed = sup.future.clone()
    changed[sup.future_mask] = torch.randn_like(changed[sup.future_mask]) * 7
    other = replace(sup, future=changed)
    logits = []
    hook = model.pcr_head.register_forward_hook(lambda module, args, output: logits.append(output.detach()))
    try:
        model.compute_loss(inp, sup, **options(.5))
        model.compute_loss(inp, other, **options(.5))
        model(inp)
    finally:
        hook.remove()
    assert len(logits) == 3
    torch.testing.assert_close(logits[0], logits[1], atol=0, rtol=0)
    torch.testing.assert_close(logits[0], logits[2], atol=0, rtol=0)


def test_direct_zero_jepa_keeps_bce_backbone_gradients(model_cfg, t0_batch):
    worker.install_paired_readout_initialization()
    model = RawVLAJEPA(replace(model_cfg, pcr_readout="direct")).train()
    inp, sup = t0_batch
    loss, _ = model.compute_loss(inp, sup, **options())
    loss.backward()
    assert torch.isfinite(loss) and loss > 0
    for module in (model.encoder, model.fusion, model.pcr_head):
        assert sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in model.world_predictor.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.teacher_encoder.parameters())


@pytest.fixture
def suite_job(tmp_path, model_cfg):
    manifest = make_raw_synthetic(tmp_path / "data", n_train=4, n_val=4, image_shape=(4, 8, 8))
    data = read_json(manifest)
    for patient in data["patients"][-2:]:
        patient["split"] = "test"
    data["provenance"]["independence_scope"] = "downstream_only"
    write_json(manifest, data)
    suite = tmp_path / "suite"
    suite.mkdir()
    codec = tmp_path / "original_codec.pt"
    codec.write_bytes(b"protected synthetic codec identity")
    cfg = RawVLATrainConfig(model=replace(model_cfg, pcr_readout="direct"),
        image_shape=(4, 8, 8), training_unit="epochs", max_epochs=1,
        early_stopping_patience=2, landmarks="t0", batch_size=2,
        accumulation=1, validation_batch_size=2, task_weight=1.,
        jepa_weight=0., reconstruction_weight=0., variance_weight=0.,
        variance_definition="off", prediction_training="teacher_forcing",
        selection_metric="f1", checkpoint_every=1, validation_every=1,
        threads=1, device="cpu", precision="fp32", allow_synthetic=True, seed=17)
    config = suite / "config.yaml"
    config.write_text(yaml.safe_dump(cfg.to_dict()))
    job = {"name": "direct0_fold0", "variant": "t0_direct_jepa0", "fold_id": 0,
        "seed": 17, "config": str(config), "config_digest": cfg.digest,
        "manifest": str(manifest), "manifest_sha256": digest(manifest),
        "output": str(suite / "runs" / "direct0_fold0"), "status": "queued"}
    state = {"jobs": [job], "codec": str(codec), "codec_sha256": digest(codec),
        "runtime": str(suite / "runtime"), "runtime_hashes": {},
        "source_identity": stable_hash(worker._source_snapshot()), "spec": {}}
    write_json(suite / "matrix.json", state)
    return suite, state, job, cfg


def test_missing_test_prediction_prevents_receipt_and_cleanup(suite_job, monkeypatch):
    suite, state, job, cfg = suite_job
    worker.run_job(suite, job["name"], "train")
    actual_evaluate = worker.evaluate

    def incomplete_test(*args, **kwargs):
        result = actual_evaluate(*args, **kwargs)
        if kwargs["split"] == "test":
            result["rows"].pop()
        return result

    monkeypatch.setattr(worker, "evaluate", incomplete_test)
    with pytest.raises(ValueError, match="coverage"):
        worker.run_job(suite, job["name"], "evaluate")
    run = Path(job["output"])
    assert (run / "best.pt").is_file() and (run / "last.pt").is_file()
    assert not (run / "evaluation_complete.json").exists()
    assert not (run / "evaluation_done.json").exists()


@pytest.mark.parametrize("tamper", ["source", "data", "config", "epoch"])
def test_completed_validation_rejects_mismatched_checkpoint(suite_job, tamper):
    suite, state, job, cfg = suite_job
    worker.run_job(suite, job["name"], "train")
    run = Path(job["output"])
    saved = load_checkpoint(run / "best.pt")
    if tamper == "source":
        saved["source_snapshot"]["model.py"] += "\n# different execution source\n"
    elif tamper == "data":
        saved["patient_partitions"]["train"] = []
    elif tamper == "config":
        saved["config"]["jepa_weight"] = .5
    else:
        saved["epoch_state"]["completed_epochs"] += 1
    save_checkpoint(run / "best.pt", saved)
    with pytest.raises(ValueError, match="Completed checkpoint"):
        worker.run_job(suite, job["name"], "evaluate")
    assert (run / "best.pt").is_file() and (run / "last.pt").is_file()
    assert not (run / "evaluation_complete.json").exists()


def test_receipt_allows_partial_cleanup_restart_without_weights(suite_job, monkeypatch):
    suite, state, job, cfg = suite_job
    worker.run_job(suite, job["name"], "train")
    run = Path(job["output"])
    actual_cleanup = worker.delete_evaluated_checkpoints

    def crash_during_cleanup(*args, **kwargs):
        (run / "best.pt").unlink()
        raise RuntimeError("simulated worker crash after one deletion")

    monkeypatch.setattr(worker, "delete_evaluated_checkpoints", crash_during_cleanup)
    with pytest.raises(RuntimeError, match="simulated"):
        worker.run_job(suite, job["name"], "evaluate")
    assert (run / "evaluation_complete.json").is_file()
    assert (run / "last.pt").is_file()
    monkeypatch.setattr(worker, "delete_evaluated_checkpoints", actual_cleanup)
    monkeypatch.setattr(worker, "validate_completed_run", lambda *args: pytest.fail("Receipt retry loaded weights"))
    done = worker.run_job(suite, job["name"], "evaluate")
    assert done["completed"] and done["cleanup"]["all_receipted_weights_absent"]
    assert not list(run.glob("*.pt"))
    assert Path(state["codec"]).read_bytes() == b"protected synthetic codec identity"
    for kind, split in worker.EVALUATIONS:
        assert read_json(run / f"evaluation_{kind}_{split}.json")["rows"]
        world = read_json(run / f"world_{kind}_{split}.json")
        assert world["status"] == "not_applicable" and world["world_head_trained"] is False
    assert worker.run_job(suite, job["name"], "evaluate") == done
    monkeypatch.setattr(worker, "train", lambda *args, **kwargs: pytest.fail("Completed run retrained"))
    assert worker.run_job(suite, job["name"], "train")["status"] == "train_report_completed"


def test_job_lock_excludes_duplicate_worker(tmp_path):
    with worker.job_lock(tmp_path / "run", "train"):
        with pytest.raises(RuntimeError, match="job lock"):
            with worker.job_lock(tmp_path / "run", "evaluate"):
                pytest.fail("A second worker acquired the lock")


def test_profile_final_schema_requires_world_measurement(suite_job, monkeypatch, tmp_path):
    suite, state, job, cfg = suite_job
    cfg = replace(cfg, batch_size=16)
    calls = []

    def capacity(config, manifest, output, **kwargs):
        assert getattr(RawVLAJEPA.__init__, "__paired_readout_initialization__", False)
        calls.append("capacity")
        return {"status": "passed", "batch_size": 16, "trainable_parameters": 50,
            "total_parameters": 100, "training_peak_reserved_mib": 10.,
            "inference_peak_reserved_mib": 12., "diagnostic_peak_reserved_mib": 15.,
            "peak_reserved_mib": 15., "peak_allocated_mib": 14.}

    def world(config, manifest):
        calls.append("world_fp32")
        return {"world_peak_reserved_mib": 20., "world_peak_allocated_mib": 18.,
                "world_profile": {"status": "passed", "precision": "fp32"}}

    monkeypatch.setattr(worker, "_capacity_profiler", lambda runtime: capacity)
    monkeypatch.setattr(worker, "_profile_world", world)
    target = tmp_path / "preflight" / "profile.json"
    result = worker.profile_job(state, job, cfg, target)
    assert calls == ["capacity", "world_fp32"]
    assert result == read_json(target)
    assert result["status"] == "passed" and result["batch_size"] == 16
    assert result["peak_reserved_mib"] == 20.
    assert result["checkpoint_bytes"] == worker.checkpoint_estimate(100, 50)
    assert result["host_peak_mib"] > 0
    assert result["host_reservation_mib"] >= 2 * result["checkpoint_bytes"] / 2**20 + 1024
    assert set(result["hardware"]) == {"gpu_name", "total_mib", "bf16_supported"}
    assert not (Path(job["output"]) / "progress.json").exists()
    assert not list(Path(job["output"]).glob("*.pt"))

    def fail_world(*args):
        raise RuntimeError("FP32 world did not fit")

    monkeypatch.setattr(worker, "_profile_world", fail_world)
    with pytest.raises(RuntimeError, match="did not fit"):
        worker.profile_job(state, job, cfg, target)
    assert read_json(target)["status"] == "profiling"


def test_world_profile_really_evaluates_full_fp32_batch(suite_job, monkeypatch):
    suite, state, job, cfg = suite_job
    worker.install_paired_readout_initialization()
    cfg = replace(cfg, jepa_weight=.5, precision="bf16")
    calls = []
    actual_world = worker.evaluate_world

    def record(model, store, config, **kwargs):
        calls.append((config.precision, kwargs["batch_size"], tuple(store.by_split["val"])))
        return actual_world(model, store, config, **kwargs)

    monkeypatch.setattr(worker, "evaluate_world", record)
    monkeypatch.setattr(worker.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(worker.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(worker.torch.cuda, "max_memory_reserved", lambda: 20 * 2**20)
    monkeypatch.setattr(worker.torch.cuda, "max_memory_allocated", lambda: 18 * 2**20)
    result = worker._profile_world(cfg, job["manifest"])
    assert calls == [("fp32", cfg.validation_batch_size, (0, 1))]
    assert result["world_peak_reserved_mib"] == 20.
    assert result["world_profile"]["patients"] == cfg.validation_batch_size
    assert result["world_profile"]["legal_prefixes"] == cfg.validation_batch_size


def attempt_environment(monkeypatch, suite, identifier="attempt-123"):
    target = suite / "attempts" / f"{identifier}.json"
    monkeypatch.setenv("T0_CV_ATTEMPT_ID", identifier)
    monkeypatch.setenv("T0_CV_ATTEMPT_STATUS", str(target))
    return target


def test_cli_records_successful_attempt(suite_job, monkeypatch):
    suite, state, job, cfg = suite_job
    target = attempt_environment(monkeypatch, suite)
    monkeypatch.setattr(worker, "run_job", lambda *args, **kwargs: {"status": "completed"})
    worker.main(["--suite", str(suite), "--job", job["name"], "--phase", "train"])
    receipt = read_json(target)
    assert receipt["attempt_id"] == "attempt-123" and receipt["exit_code"] == 0
    assert receipt["job"] == job["name"] and receipt["phase"] == "train"
    assert type(receipt["pid"]) is int
    assert receipt["error_type"] is None and receipt["error"] is None


def test_cli_records_load_context_failure_before_training(tmp_path, monkeypatch):
    suite = tmp_path / "missing-suite"
    target = attempt_environment(monkeypatch, suite)
    with pytest.raises(FileNotFoundError):
        worker.main(["--suite", str(suite), "--job", "missing", "--phase", "evaluate"])
    receipt = read_json(target)
    assert receipt["exit_code"] == 1 and receipt["error_type"] == "FileNotFoundError"
    assert "matrix.json" in receipt["error"]


def test_cli_preserves_original_exception_if_receipt_write_fails(suite_job, monkeypatch):
    suite, state, job, cfg = suite_job
    attempt_environment(monkeypatch, suite)
    original = KeyboardInterrupt("interrupted inside worker")

    def failure(*args, **kwargs):
        raise original

    def no_receipt(*args, **kwargs):
        raise OSError("Disk quota exceeded")

    monkeypatch.setattr(worker, "run_job", failure)
    monkeypatch.setattr(worker, "durable_json", no_receipt)
    with pytest.raises(KeyboardInterrupt) as error:
        worker.main(["--suite", str(suite), "--job", job["name"], "--phase", "train"])
    assert error.value is original
    assert "Attempt receipt write failed" in error.value.__notes__[0]


@pytest.mark.parametrize("unsafe", ["identifier", "outside", "overwrite"])
def test_cli_rejects_unsafe_attempt_destination(tmp_path, monkeypatch, unsafe):
    suite = tmp_path / "suite"
    target = attempt_environment(monkeypatch, suite)
    if unsafe == "identifier":
        monkeypatch.setenv("T0_CV_ATTEMPT_ID", "../escape")
    elif unsafe == "outside":
        monkeypatch.setenv("T0_CV_ATTEMPT_STATUS", str(tmp_path / "outside.json"))
    else:
        monkeypatch.setenv("T0_CV_ATTEMPT_STATUS", str(suite / "matrix.json"))
    monkeypatch.setattr(worker, "run_job", lambda *args, **kwargs: pytest.fail("Unsafe attempt was executed"))
    with pytest.raises(ValueError, match="Attempt|Worker output"):
        worker.main(["--suite", str(suite), "--job", "missing", "--phase", "train"])
    assert not target.exists()
    assert not (tmp_path / "outside.json").exists()


def partial_best_only(suite_job, monkeypatch, scheduler="warmup_cosine"):
    suite, state, job, cfg = suite_job
    cfg = replace(cfg, max_epochs=2, lr_scheduler=scheduler)
    Path(job["config"]).write_text(yaml.safe_dump(cfg.to_dict()))
    job["config_digest"] = cfg.digest
    write_json(suite / "matrix.json", state)
    actual_train = worker.train

    def first_epoch(*args, **kwargs):
        return actual_train(*args, **kwargs, stop_after=2)

    monkeypatch.setattr(worker, "train", first_epoch)
    partial = worker.run_job(suite, job["name"], "train")
    assert partial["completed"] is False and partial["completed_epochs"] == 1
    monkeypatch.setattr(worker, "train", actual_train)
    run = Path(job["output"])
    (run / "last.pt").unlink()
    return suite, state, job, cfg, actual_train


@pytest.mark.parametrize("sidecar", ["missing", "wrong_score"])
def test_best_only_recovery_resumes_full_state_and_repairs_validation(suite_job, monkeypatch, sidecar):
    suite, state, job, cfg, actual_train = partial_best_only(suite_job, monkeypatch)
    run = Path(job["output"])
    original_best = digest(run / "best.pt")
    if sidecar == "missing":
        (run / "best_validation.json").unlink()
    else:
        value = read_json(run / "best_validation.json")
        value["selection_score"] = -10.
        write_json(run / "best_validation.json", value)
    calls = []

    def resumed(*args, **kwargs):
        calls.append(kwargs["resume"])
        return actual_train(*args, **kwargs)

    monkeypatch.setattr(worker, "train", resumed)
    report = worker.run_job(suite, job["name"], "train")
    assert calls == [True] and report["completed"] is True
    assert report["completed_epochs"] == 2 and report["completed_steps"] == 4
    recovery = read_json(run / "best_only_recovery.json")
    assert recovery["status"] == "last_restored"
    assert recovery["source"]["sha256"] == recovery["destination_sha256"] == original_best
    assert recovery["completed_epochs"] == 1
    assert read_json(run / "best_validation.json")["selection_score"] == report["best_selection_score"]
    assert (run / "best.pt").is_file() and (run / "last.pt").is_file()
    worker.validate_completed_run(job, cfg)


@pytest.mark.parametrize("corruption", ["identity", "optimizer", "rng", "scheduler", "epoch"])
def test_best_only_recovery_rejects_invalid_state(suite_job, monkeypatch, corruption):
    suite, state, job, cfg, actual_train = partial_best_only(suite_job, monkeypatch)
    run = Path(job["output"])
    saved = load_checkpoint(run / "best.pt")
    if corruption == "identity":
        saved["patient_partitions"]["train"] = []
    elif corruption == "optimizer":
        saved["optimizer"]["state"] = {}
    elif corruption == "rng":
        saved["rng"]["torch"] = torch.zeros(1, dtype=torch.uint8)
    elif corruption == "scheduler":
        saved["scheduler_state"] = None
    else:
        saved["epoch_state"]["position"] = 1
    save_checkpoint(run / "best.pt", saved)
    monkeypatch.setattr(worker, "train", lambda *args, **kwargs: pytest.fail("Invalid best was resumed"))
    with pytest.raises((ValueError, RuntimeError), match="Best-only|RNG"):
        worker.run_job(suite, job["name"], "train")
    assert (run / "best.pt").is_file() and not (run / "last.pt").exists()


def test_evaluate_repairs_sidecar_after_completed_train_interruption(suite_job, monkeypatch):
    suite, state, job, cfg, actual_train = partial_best_only(suite_job, monkeypatch)
    run = Path(job["output"])
    actual_repair = worker._repair_best_validation

    def interrupted_repair(*args, **kwargs):
        (run / "best_validation.json").unlink(missing_ok=True)
        raise RuntimeError("interrupted before validation sidecar repair")

    monkeypatch.setattr(worker, "_repair_best_validation", interrupted_repair)
    with pytest.raises(RuntimeError, match="interrupted before"):
        worker.run_job(suite, job["name"], "train")
    assert read_json(run / "train_report.json")["completed"] is True
    monkeypatch.setattr(worker, "_repair_best_validation", actual_repair)
    result = worker.run_job(suite, job["name"], "evaluate")
    assert result["completed"] is True
    assert read_json(run / "best_validation_recovery.json")["status"] == "restored"
    assert (run / "evaluation_done.json").exists()
    assert not list(run.glob("*.pt"))

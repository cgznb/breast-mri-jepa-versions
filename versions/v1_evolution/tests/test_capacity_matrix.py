"""Experiment matrix and GPU admission checks without starting CUDA workers."""
from copy import deepcopy
import importlib.util
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from mri_vla_jepa.io import digest, read_json, write_json
from mri_vla_jepa.train_config import load_config


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("capacity_matrix_test", ROOT / "scripts/run_capacity_phase_matrix.py")
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


@pytest.fixture
def base():
    return load_config(ROOT / "local_configs/encoder64_seed17.yaml").to_dict()


@pytest.mark.parametrize("batch_size", [16, 8, 4, 2, 1])
def test_48_configs_are_complete_independent_and_share_protocol(base, batch_size):
    original = deepcopy(base)
    configs = matrix.matrix_configs(base, batch_size=batch_size)
    assert base == original
    assert len(configs) == len({name for name, _ in configs}) == 48
    actual = {(cfg.model.model_size, cfg.model.mri_phases, cfg.prediction_training, cfg.seed)
              for _, cfg in configs}
    assert actual == set(product(matrix.SIZES, matrix.PHASES, matrix.MODES, matrix.SEEDS))
    assert [cfg.model.model_size for _, cfg in configs[:16]] == ["large"] * 16
    for _, cfg in configs:
        assert cfg.batch_size == cfg.validation_batch_size == cfg.diagnostics_batch_size == batch_size
        assert cfg.accumulation == 1
        assert cfg.training_unit == "epochs" and cfg.max_epochs == 200
        assert cfg.early_stopping_patience == 50 and cfg.early_stopping_min_delta == 0
        assert cfg.selection_metric == "f1" and cfg.f1_threshold == .5
        assert cfg.landmarks == "all_observed"
        assert cfg.precision == "bf16" and cfg.device == "cuda:0"
        assert cfg.encoder_training == "frozen" and cfg.initialize_from_checkpoint is None
        assert cfg.model.encoder_type == "vq_encoder" and cfg.model.mri_phases in matrix.PHASES
        assert cfg.model.pcr_readout == "forecast" and cfg.model.fixed_transition_queries
        assert cfg.model.history_delta and cfg.model.world_residual
        assert cfg.model.dropout == .3 and not cfg.model.enable_flow
        assert cfg.task_weight == 1 and cfg.jepa_weight == .5
        assert cfg.rollout_weight == (.5 if cfg.prediction_training == "hybrid" else 0)
        assert cfg.flow_weight == cfg.reconstruction_weight == cfg.variance_weight == 0
        assert cfg.variance_definition == "off" and cfg.image_cache is None


def test_prepared_configs_freeze_digest_and_remain_seed_independent(tmp_path):
    codec = tmp_path / "codec.pt"
    codec.touch()
    jobs = matrix.prepare(tmp_path / "matrix", ROOT / "local_configs/encoder64_seed17.yaml", codec, 8)
    assert len(jobs) == 48
    for job in jobs:
        cfg = load_config(job["config"])
        assert cfg.digest == job["config_digest"]
        assert cfg.encoder_checkpoint == str(codec.resolve())
        assert cfg.seed == job["seed"] and cfg.batch_size == 8
        assert job["status"] == "queued" and job["retries"] == 0
    first = Path(jobs[0]["config"])
    second = Path(jobs[1]["config"])
    unchanged = second.read_bytes()
    changed = yaml.safe_load(first.read_text())
    changed["seed"] = 91
    first.write_text(yaml.safe_dump(changed))
    assert second.read_bytes() == unchanged
    assert load_config(first).digest != jobs[0]["config_digest"]


def test_gpu_inventory_resolves_uuid_and_keeps_per_device_margin(monkeypatch):
    raw = "0, GPU-a, NVIDIA A100-SXM4-80GB, 81920, 4\n1, GPU-b, NVIDIA A100-SXM4-80GB, 81920, 4\n"
    monkeypatch.setattr(matrix.subprocess, "check_output", lambda *a, **kw: raw)
    inventory = matrix.gpu_inventory(["0", "GPU-b"])
    assert [gpu["uuid"] for gpu in inventory] == ["GPU-a", "GPU-b"]
    assert all(gpu["budget_mib"] == 73728 for gpu in inventory)
    assert matrix.environment(inventory[1])["CUDA_VISIBLE_DEVICES"] == "GPU-b"
    with pytest.raises(ValueError):
        matrix.gpu_inventory(["0", "GPU-a"])
    with pytest.raises(ValueError):
        matrix.gpu_inventory(["2"])


def devices():
    return [{"uuid": "GPU-a", "index": 0, "budget_mib": 72000},
            {"uuid": "GPU-b", "index": 1, "budget_mib": 72000}]


def admission_fixture():
    job = {"size": "large", "phase": "all", "mode": "teacher_forcing"}
    profiles = {matrix.profile_key(job): {"reservation_mib": 31000}}
    limits = {matrix.benchmark_key(job): 2, "small/single/hybrid": 8}
    return job, profiles, limits


def test_gpu_admission_balances_both_devices_then_fills_memory():
    job, profiles, limits = admission_fixture()
    first = {"gpu_uuid": "GPU-a", "reservation_mib": 31000,
             "benchmark_key": matrix.benchmark_key(job)}
    active = {"large1": first}
    assert matrix.choose_gpu(job, active, devices(), profiles, limits, {})["uuid"] == "GPU-b"
    active["large2"] = {**first, "gpu_uuid": "GPU-b"}
    assert matrix.choose_gpu(job, active, devices(), profiles, limits, {})["uuid"] == "GPU-a"
    active["large3"] = dict(first)
    active["large4"] = {**first, "gpu_uuid": "GPU-b"}
    assert matrix.choose_gpu(job, active, devices(), profiles, limits, {}) is None


def test_admission_obeys_memory_and_oom_cap_on_each_gpu():
    job, profiles, limits = admission_fixture()
    workers = {"worker": {"gpu_uuid": "GPU-a", "reservation_mib": 45000,
                          "benchmark_key": matrix.benchmark_key(job)}}
    assert matrix.choose_gpu(job, workers, devices()[:1], profiles, limits, {}) is None
    workers["worker"]["reservation_mib"] = 31000
    assert matrix.choose_gpu(job, workers, devices()[:1], profiles, limits, {"GPU-a": 1}) is None
    assert matrix.choose_gpu(job, workers, devices(), profiles, limits, {"GPU-a": 1})["uuid"] == "GPU-b"


def test_mixed_workers_use_strictest_measured_limit():
    job, profiles, limits = admission_fixture()
    small = {"size": "small", "phase": "first_post_aqc1", "mode": "hybrid"}
    profiles[matrix.profile_key(small)] = {"reservation_mib": 4000}
    workers = {"large": {"gpu_uuid": "GPU-a", "reservation_mib": 31000,
                         "benchmark_key": matrix.benchmark_key(job)},
               "small": {"gpu_uuid": "GPU-a", "reservation_mib": 4000,
                         "benchmark_key": matrix.benchmark_key(small)}}
    assert matrix.choose_gpu(small, workers, devices()[:1], profiles, limits, {}) is None


def test_mixed_pair_requires_exact_measured_permission_and_never_admits_third():
    job, profiles, limits = admission_fixture()
    small = {"size": "small", "phase": "first_post_aqc1", "mode": "teacher_forcing"}
    other_mode = {**small, "mode": "hybrid"}
    profiles.update({matrix.profile_key(small): {"reservation_mib": 4000},
                     matrix.profile_key(other_mode): {"reservation_mib": 4000}})
    limits[matrix.benchmark_key(small)] = 1
    workers = {"large": {"gpu_uuid": "GPU-a", "reservation_mib": 31000,
                         "benchmark_key": matrix.benchmark_key(job)}}
    allowed = [matrix.mixed_key([matrix.benchmark_key(job), matrix.benchmark_key(small)])]
    assert matrix.choose_gpu(small, workers, devices()[:1], profiles, limits, {}) is None
    assert matrix.choose_gpu(small, workers, devices()[:1], profiles, limits, {}, allowed)["uuid"] == "GPU-a"
    # An exact pairing's result cannot authorize another training mode.
    assert matrix.choose_gpu(other_mode, workers, devices()[:1], profiles, limits, {}, allowed) is None
    assert matrix.choose_gpu(small, workers, devices()[:1], profiles, limits, {"GPU-a": 1}, allowed) is None
    workers["small"] = {"gpu_uuid": "GPU-a", "reservation_mib": 4000,
                        "benchmark_key": matrix.benchmark_key(small)}
    assert matrix.choose_gpu(small, workers, devices()[:1], profiles, limits, {}, allowed) is None


def test_runtime_hashes_detect_changes_and_ignore_python_caches(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/model.py").write_text("model")
    before = matrix.runtime_hashes(tmp_path)
    (tmp_path / "src/__pycache__").mkdir()
    (tmp_path / "src/__pycache__/cache.py").write_text("ignored")
    assert matrix.runtime_hashes(tmp_path) == before
    (tmp_path / "src/model.py").write_text("tampered")
    assert matrix.runtime_hashes(tmp_path) != before


@pytest.mark.parametrize("tamper,match", [("runtime", "scripts"), ("manifest", "manifest"),
                                         ("codec", "codec"), ("config", "config"),
                                         ("data", "data identity")])
def test_worker_rejects_frozen_identity_tampering_before_spawn(tmp_path, monkeypatch, base, tamper, match):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    runtime_file = runtime / "controller.py"
    runtime_file.write_text("frozen controller")
    manifest, codec = tmp_path / "manifest.json", tmp_path / "codec.pt"
    manifest.write_text("{}")
    codec.write_bytes(b"frozen codec")
    config = tmp_path / "config.yaml"
    base["encoder_checkpoint"] = str(codec)
    config.write_text(yaml.safe_dump(base))
    job = {"name": "frozen_job", "config": str(config), "config_digest": load_config(config).digest}
    state = {"source_identity": "frozen-source", "runtime_hashes": matrix.runtime_hashes(runtime),
             "manifest": str(manifest), "manifest_sha256": digest(manifest),
             "codec_sha256": digest(codec), "data_signature": {"identity": "frozen-data"}, "jobs": [job]}
    write_json(tmp_path / "matrix.json", state)

    class Store:
        def signature(self):
            return {"identity": "changed-data" if tamper == "data" else "frozen-data"}

    monkeypatch.setattr(matrix, "ROOT", runtime)
    monkeypatch.setattr(matrix, "source_identity", lambda: "frozen-source")
    monkeypatch.setattr(matrix, "load_queue", lambda: None)
    monkeypatch.setattr(matrix, "RawMRIStore", lambda *_: Store())
    monkeypatch.setattr(matrix, "spawn", lambda *a, **kw: pytest.fail("identity drift must reject before spawn"))
    if tamper == "runtime":
        runtime_file.write_text("altered runtime")
    elif tamper == "manifest":
        manifest.write_text('{"changed": true}')
    elif tamper == "codec":
        codec.write_bytes(b"different codec")
    elif tamper == "config":
        base["seed"] += 1
        config.write_text(yaml.safe_dump(base))
    with pytest.raises(ValueError, match=match):
        matrix.work(tmp_path)


def test_runtime_snapshot_copies_source_and_scripts_independently(tmp_path, monkeypatch):
    source = tmp_path / "project"
    (source / "src/pkg").mkdir(parents=True)
    (source / "scripts").mkdir()
    (source / "src/pkg/model.py").write_text("original model")
    (source / "src/pkg/attention.py").write_text("attention")
    (source / "src/pkg/phases.py").write_text("phases")
    (source / "scripts/run_capacity_phase_matrix.py").write_text("controller")
    (source / "train.py").write_text("train entry")
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(matrix, "ROOT", source)
    runtime = matrix.freeze_runtime(output)
    (source / "src/pkg/model.py").write_text("edited later")
    (source / "scripts/run_capacity_phase_matrix.py").write_text("edited controller")
    assert (runtime / "src/pkg/model.py").read_text() == "original model"
    assert (runtime / "src/pkg/attention.py").read_text() == "attention"
    assert (runtime / "src/pkg/phases.py").read_text() == "phases"
    assert (runtime / "scripts/run_capacity_phase_matrix.py").read_text() == "controller"
    assert (runtime / "train.py").read_text() == "train entry"
    with pytest.raises(FileExistsError):
        matrix.freeze_runtime(output)


def test_worker_rejects_source_drift_before_starting_children(tmp_path, monkeypatch):
    write_json(tmp_path / "matrix.json", {"source_identity": "old-source"})
    monkeypatch.setattr(matrix, "source_identity", lambda: "new-source")
    monkeypatch.setattr(matrix, "load_queue", lambda: None)
    monkeypatch.setattr(matrix, "spawn", lambda *a, **kw: pytest.fail("must reject drift before spawn"))
    with pytest.raises(ValueError, match="source"):
        matrix.work(tmp_path)


def test_preflight_oom_cleans_other_gpu_workers_before_batch_retry(tmp_path, monkeypatch):
    jobs = [{"name": f"trial{index}", "size": "small", "phase": "all",
             "mode": "teacher_forcing", "seed": 17, "config": "unused.yaml"}
            for index in range(2)]
    children = []

    class Child:
        def __init__(self, failure):
            self.pid = 9000 + len(children)
            self.failure = failure
            self.returncode = None
            self.terminated = False

        def poll(self):
            if self.failure:
                self.returncode = 1
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def kill(self):
            self.terminate()

        def wait(self, timeout=None):
            if self.returncode is None:
                raise AssertionError("Must terminate surviving preflight child before waiting")
            return self.returncode

    def spawn(command, log, gpu):
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("CUDA out of memory" if not children else "still running")
        child = Child(failure=not children)
        children.append(child)
        return child

    monkeypatch.setattr(matrix, "spawn", spawn)
    monkeypatch.setattr(matrix, "process_memory", lambda: {})
    monkeypatch.setattr(matrix, "gpu_memory_used", lambda: {})
    monkeypatch.setattr(matrix.time, "sleep", lambda _: None)
    with pytest.raises(matrix.torch_oom):
        matrix.preflight(tmp_path, jobs, Path("manifest.json"), devices())
    assert len(children) == 2
    assert children[1].terminated


def test_preflight_uses_exclusive_gpu_total_when_nvml_has_host_pid(tmp_path, monkeypatch):
    job = {"name": "profile", "size": "small", "phase": "all",
           "mode": "teacher_forcing", "seed": 17, "config": "unused.yaml"}

    class Child:
        pid = 111

        def poll(self):
            return 0

    def spawn(command, log, gpu):
        target = command[command.index("--output") + 1]
        write_json(target, {"peak_reserved_mib": 1000})
        return Child()

    monkeypatch.setattr(matrix, "spawn", spawn)
    monkeypatch.setattr(matrix, "load_config", lambda _: SimpleNamespace(digest="config-identity"))
    monkeypatch.setattr(matrix, "source_identity", lambda: "source-identity")
    monkeypatch.setattr(matrix, "process_memory", lambda: {99111: {"gpu_uuid": "GPU-a", "mib": 2400}})
    monkeypatch.setattr(matrix, "gpu_memory_used", lambda: {"GPU-a": 2600})
    monkeypatch.setattr(matrix.time, "sleep", lambda _: None)
    profiles = matrix.preflight(tmp_path, [job], Path("manifest.json"), devices()[:1])
    result = profiles[matrix.profile_key(job)]
    assert result["nvml_peak_mib"] == 2600
    assert result["reservation_mib"] == 2600 + 512
    assert result["config_digest"] == "config-identity"

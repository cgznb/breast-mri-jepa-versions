"""Fixed historical holdout, paired fivefold T0 preparation and durable reports.

This module never starts training or deletes weights. Test predictions stay out
of public reports until every pre-registered job has completed.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import fcntl
import html
import json
import math
import os
from pathlib import Path
import statistics

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold
import yaml

from .ablation_suite import METRICS, COUNTS, _stratum, _table_csv, validate_classification_report
from .arm import normalize_patient_key
from .data import RawMRIStore
from .io import digest, read_json, stable_hash, write_json
from .metrics import classification_metrics
from .train_config import from_dict, load_config

SCHEMA = "t0_cv5_suite_v1"
STATE_SCHEMA = "t0_cv5_state_v1"
REPORT_METRICS = METRICS + ("predicted_positive_fraction",)
VARIANTS = (
    {"name": "t0_forecast", "pcr_readout": "forecast", "jepa_weight": .5, "spatial_readout": "attention"},
    {"name": "t0_direct_no_jepa", "pcr_readout": "direct", "jepa_weight": 0., "spatial_readout": "mean"},
    {"name": "t0_direct_readout", "pcr_readout": "direct", "jepa_weight": .5, "spatial_readout": "mean"},
)
OFFICIAL_FOLDS = (
    {"train": 554, "train_positive": 177, "val": 139, "val_positive": 44},
    {"train": 554, "train_positive": 176, "val": 139, "val_positive": 45},
    {"train": 554, "train_positive": 176, "val": 139, "val_positive": 45},
    {"train": 555, "train_positive": 178, "val": 138, "val_positive": 43},
    {"train": 555, "train_positive": 177, "val": 138, "val_positive": 44},
)
PATH_FIELDS = ("source_manifest", "reference_split_manifest", "reference_split_audit", "codec", "output_root")


def _identity(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Required input is not a regular file: {path}")
    return {"path": str(path), "size": path.stat().st_size, "sha256": digest(path)}


def _verify_identity(identity):
    if _identity(identity["path"]) != identity:
        raise ValueError(f"Frozen input identity changed: {identity['path']}")


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_spec(spec):
    if not isinstance(spec, dict) or spec.get("schema") != SCHEMA:
        raise ValueError("Unsupported T0 CV suite schema")
    name = spec.get("suite_name")
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("suite_name must be one directory name")
    if set(spec.get("paths", {})) != set(PATH_FIELDS):
        raise ValueError("Suite paths must declare all five input/output roots")
    if spec.get("variants") != list(VARIANTS):
        raise ValueError("Exactly the three pre-registered T0 variants are required")
    cv = spec["cv"]
    for key in ("n_splits", "expected_patients", "test_size"):
        _positive_integer(cv[key], key)
    if cv["n_splits"] != 5 or cv["test_size"] >= cv["expected_patients"]:
        raise ValueError("Expected five folds and a nonempty development population")
    for key in ("split_seed", "train_seed", "holdout_seed"):
        if type(cv[key]) is not int or cv[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if (cv["train_seed"], cv["holdout_seed"], cv["split_seed"]) != (17, 17, 20261020):
        raise ValueError("Holdout/train seeds are 17 and CV split seed is 20261020")
    if not isinstance(cv.get("expected_fold_counts"), list) or len(cv["expected_fold_counts"]) != 5:
        raise ValueError("Explicit expected counts for each of five folds are required")
    resources = spec["resources"]
    for key in ("expected_gpus", "reservation_margin_mib", "threads_per_task", "loader_workers"):
        _positive_integer(resources[key], "resources." + key)
    if resources["expected_gpus"] != 8 or not 1 <= resources["loader_workers"] <= 8:
        raise ValueError("Expected eight GPUs and one to eight loader workers per task")
    for key in ("memory_fraction", "host_memory_fraction"):
        value = resources[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"resources.{key} must be in (0,1]")
    for key in ("poll_seconds", "storage_margin_factor"):
        value = resources[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"resources.{key} must be positive and finite")
    if resources["storage_margin_factor"] < 1:
        raise ValueError("Storage margin factor must be at least one")
    training = deepcopy(spec["training"])
    training["encoder_checkpoint"] = spec["paths"]["codec"]
    cfg = from_dict(training)
    if cfg.seed != 17 or cfg.landmarks != "t0" or cfg.selection_metric != "f1" or cfg.f1_threshold != .5:
        raise ValueError("Suite requires train seed17, T0 and fixed F1 threshold 0.5")
    if not cfg.model.enable_world_model:
        raise ValueError("All three tasks preserve the common enabled world structure")
    synthetic = cfg.allow_synthetic
    if not synthetic:
        if (cv["expected_patients"], cv["test_size"], cv.get("expected_test_positives"),
                cv.get("expected_development_positives")) != (866, 173, 55, 221):
            raise ValueError("Formal cohort expectations must not drift")
        if cv.get("expected_fold_counts") != list(OFFICIAL_FOLDS):
            raise ValueError("Formal fold counts must match the registered split")
        required = {"encoder_training": "finetune", "lr": 1e-4, "encoder_lr": 1e-4,
                    "lr_scheduler": "warmup_cosine", "weight_decay": .05, "batch_size": 16,
                    "training_unit": "epochs", "max_epochs": 200, "early_stopping_patience": 50,
                    "prediction_training": "teacher_forcing", "warmup_epochs": 10,
                    "warmup_start_factor": .1, "cosine_end_epoch": 100, "cosine_min_lr": 1e-6,
                    "task_weight": 1., "jepa_weight": .5, "rollout_weight": 0., "flow_weight": 0.,
                    "reconstruction_weight": 0., "variance_weight": 0., "image_cache": None,
                    "initialize_from_checkpoint": None, "early_stopping_min_delta": 0.,
                    "precision": "bf16", "validation_batch_size": 16, "diagnostics_batch_size": 16,
                    "accumulation": 1, "variance_definition": "off", "validation_every": 1,
                    "schedule": "one_stage", "representation_steps": 0,
                    "freeze_encoder_after_representation": False, "device": "cuda:0"}
        for key, expected in required.items():
            if getattr(cfg, key) != expected:
                raise ValueError(f"Formal combined training setting changed: {key}")
        if (cfg.model.model_size, cfg.model.mri_phases, cfg.model.ema_decay, cfg.model.dropout,
                cfg.model.vq_activation_checkpointing, cfg.model.pcr_readout,
                cfg.model.pcr_delta_reference) != ("large", "all", .999, .3, True, "forecast", "current"):
            raise ValueError("Formal large/all combined model settings changed")
        for key, expected in {"encoder_type": "vq_encoder", "fixed_transition_queries": True,
                "world_residual": True, "history_delta": True, "enable_flow": False,
                "lesion_world_weighting": False, "lesion_pooling": False, "spatial_readout": "attention"}.items():
            if getattr(cfg.model, key) != expected:
                raise ValueError(f"Formal common model setting changed: {key}")
        if tuple(cfg.image_shape) != (32, 128, 128) or tuple(cfg.model.token_grid) != (2, 4, 4):
            raise ValueError("Formal suite requires unchanged ROI32 images and 32 spatial tokens")
        if set(spec.get("input_sha256", {})) != set(PATH_FIELDS) - {"output_root"}:
            raise ValueError("Formal inputs require all four registered SHA256 identities")
    for key in ("bootstrap", "bootstrap_seed"):
        value = spec["report"][key]
        if type(value) is not int or value < 0:
            raise ValueError(f"report.{key} must be a nonnegative integer")
    return cfg


def load_suite_config(path):
    """Resolve YAML paths relative to the YAML, returning JSON-safe values."""
    path = Path(path).expanduser().resolve()
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Suite YAML must contain a mapping")
    value = deepcopy(value)
    for key in PATH_FIELDS:
        p = Path(value["paths"][key]).expanduser()
        value["paths"][key] = str((path.parent / p).resolve())
    for key in ("encoder_checkpoint", "initialize_from_checkpoint", "image_cache"):
        if value["training"].get(key) is not None:
            p = Path(value["training"][key]).expanduser()
            value["training"][key] = str((path.parent / p).resolve())
    if value["training"].get("encoder_checkpoint") not in {None, value["paths"]["codec"]}:
        raise ValueError("Training codec path differs from suite codec")
    value["training"]["encoder_checkpoint"] = value["paths"]["codec"]
    value["output_dir"] = str(Path(value["paths"]["output_root"]) / value["suite_name"])
    value["_config_file"] = str(path)
    value["_config_sha256"] = digest(path)
    _validate_spec(value)
    return value


def _canonical_patient(patient, manifest):
    patient = deepcopy(patient)
    patient.pop("split", None)
    for visit in patient["visits"]:
        if visit is None:
            continue
        for field in ("image", "phases"):
            if field in visit:
                paths = [visit[field]] if field == "image" else visit[field]
                resolved = [str((Path(manifest).parent / p).resolve()) for p in paths]
                visit[field] = resolved[0] if field == "image" else resolved
    return patient


def _immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Existing frozen artifact differs: {path}")
    else:
        write_json(path, value)
        path.chmod(0o600)


def _asset_inventory(patients):
    assets = []
    seen = set()
    for patient in patients:
        for visit in patient["visits"]:
            if visit is None:
                continue
            for raw in RawMRIStore._visit_paths(visit):
                path = Path(raw).resolve()
                if str(path) in seen:
                    raise ValueError("MRI asset belongs to more than one patient/visit")
                seen.add(str(path))
                stat = path.stat()
                assets.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return sorted(assets, key=lambda x: x["path"])


def prepare_suite(spec, output):
    """Freeze five manifests and fifteen configs; leave matrix ownership to runner."""
    cfg = _validate_spec(spec)
    output = Path(output).resolve()
    paths = spec["paths"]
    inputs = {key: _identity(paths[key]) for key in PATH_FIELDS if key != "output_root"}
    if spec.get("_config_file"):
        inputs["suite_config"] = _identity(spec["_config_file"])
        if inputs["suite_config"]["sha256"] != spec["_config_sha256"]:
            raise ValueError("Suite YAML changed after loading")
    for key, expected in spec.get("input_sha256", {}).items():
        if inputs[key]["sha256"] != expected:
            raise ValueError(f"Registered source identity differs: {key}")
    source = read_json(paths["source_manifest"])
    reference = read_json(paths["reference_split_manifest"])
    audit = read_json(paths["reference_split_audit"])
    cv = spec["cv"]
    if bool(source.get("synthetic", False)) != cfg.allow_synthetic:
        raise ValueError("Synthetic permission must match the declared cohort")
    patients = sorted([_canonical_patient(p, paths["source_manifest"]) for p in source["patients"]],
                      key=lambda p: normalize_patient_key(p["patient_key"]))
    keys = [normalize_patient_key(p["patient_key"]) for p in patients]
    reference_records = {_key(p): _canonical_patient(p, paths["reference_split_manifest"])
                         for p in reference["patients"]}
    if len(patients) != cv["expected_patients"] or len(set(keys)) != len(keys):
        raise ValueError("Unexpected cohort size or duplicate normalized patient")
    if len(reference_records) != len(reference["patients"]) or set(reference_records) != set(keys):
        raise ValueError("Reference split cohort differs from source")
    if any(reference_records[_key(p)] != p for p in patients):
        raise ValueError("Reference split changed patient data or labels")
    if audit.get("seed") != cv["holdout_seed"]:
        raise ValueError("Reference holdout seed differs")
    for key in ("source_manifest", "codec"):
        if audit.get(key, {}).get("sha256") != inputs[key]["sha256"]:
            raise ValueError(f"Reference audit source identity differs: {key}")
    ref_ids = {split: {_key(p) for p in reference["patients"] if p["split"] == split}
               for split in ("train", "val", "test")}
    if set.union(*ref_ids.values()) != set(keys) or any(
            ref_ids[a] & ref_ids[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Invalid reference split ownership")
    if any({normalize_patient_key(x) for x in audit.get("patients", {}).get(split, [])} != ids
           or audit.get("counts", {}).get(split) != len(ids) for split, ids in ref_ids.items()):
        raise ValueError("Reference audit patient assignments differ")
    test_ids = ref_ids["test"]
    test = [p for p in patients if _key(p) in test_ids]
    dev = [p for p in patients if _key(p) not in test_ids]
    if len(test) != cv["test_size"] or sum(_label(p) for p in test) != cv["expected_test_positives"]:
        raise ValueError("Fixed test size/positive count differs")
    if sum(_label(p) for p in dev) != cv["expected_development_positives"]:
        raise ValueError("Development positive count differs")
    strata = [_stratum(p) for p in dev]
    if min(Counter(strata).values()) < cv["n_splits"]:
        raise ValueError("Every joint label/visit stratum needs at least five development patients")
    inventory = _asset_inventory(patients)
    data_path = output / "splits" / "data_identity.json"
    _immutable_json(data_path, {"schema": "t0_cv_asset_identity_v1", "assets": inventory})
    folds, jobs, validation_ids = {}, [], set()
    splitter = StratifiedKFold(n_splits=cv["n_splits"], shuffle=True, random_state=cv["split_seed"])
    for fold_id, (tr, va) in enumerate(splitter.split(dev, strata), 1):
        train_ids, val_ids = {_key(dev[i]) for i in tr}, {_key(dev[i]) for i in va}
        if validation_ids & val_ids or train_ids & val_ids or (train_ids | val_ids) & test_ids:
            raise ValueError("Cross-validation patient leakage")
        validation_ids |= val_ids
        counts = {"train": len(tr), "train_positive": sum(_label(dev[i]) for i in tr),
                  "val": len(va), "val_positive": sum(_label(dev[i]) for i in va)}
        if counts != cv["expected_fold_counts"][fold_id - 1]:
            raise ValueError(f"Fold {fold_id} size/class counts differ from registered expectations")
        assignment = {key: "train" if key in train_ids else "val" if key in val_ids else "test" for key in keys}
        manifest = deepcopy(source)
        manifest["patients"] = [dict(deepcopy(p), split=assignment[_key(p)]) for p in patients]
        manifest.setdefault("provenance", {}).update({
            "independence_scope": "downstream_only", "holdout_seed": cv["holdout_seed"],
            "cv_split_seed": cv["split_seed"], "train_seed": cv["train_seed"], "fold_id": fold_id,
            "source_manifest": inputs["source_manifest"], "pretrained_codec": inputs["codec"],
            "reference_holdout_manifest": inputs["reference_split_manifest"],
            "clinical_standardization": "fit_this_fold_training_patients_only",
            "image_standardization": "historical_preprocessed_arrays_unchanged_not_refit",
            "codec_pretraining": "historical_original_codec_population_not_redefined",
            "test_interpretation": "Historical holdout reused; independence only for this downstream round"})
        folder = output / "splits" / f"fold{fold_id}"
        manifest_path = folder / "manifest.json"
        _immutable_json(manifest_path, manifest)
        fold_audit = {"schema": "t0_cv_fold_audit_v1", "fold_id": fold_id, **cv,
            "algorithm": "StratifiedKFold(shuffle=True); joint pCR x four-visit availability",
            "sort": "normalize_patient_key_lexicographic", "sklearn_version": sklearn.__version__,
            "counts": {**counts, "test": len(test), "test_positive": sum(_label(p) for p in test)},
            "patients": {split: [p["patient_key"] for p in manifest["patients"] if p["split"] == split]
                         for split in ("train", "val", "test")},
            "stratum_counts": {split: dict(sorted(Counter(_stratum(p) for p in manifest["patients"]
                if p["split"] == split).items())) for split in ("train", "val", "test")},
            "patient_overlap": False, "source_inputs": inputs}
        audit_path = folder / "split_audit.json"
        _immutable_json(audit_path, fold_audit)
        store = RawMRIStore(manifest_path, cfg.image_shape, allow_synthetic=cfg.allow_synthetic)
        norm_path = folder / "normalization.json"
        _immutable_json(norm_path, store.normalization_state())
        fold = {"fold_id": fold_id, "manifest": str(manifest_path), "manifest_sha256": digest(manifest_path),
            "audit": str(audit_path), "audit_sha256": digest(audit_path),
            "normalization": str(norm_path), "normalization_sha256": digest(norm_path),
            "data_signature": stable_hash(store.signature()), "counts": fold_audit["counts"]}
        folds[str(fold_id)] = fold
        for variant in spec["variants"]:
            value = cfg.to_dict()
            value.update(seed=cv["train_seed"], encoder_checkpoint=paths["codec"], jepa_weight=variant["jepa_weight"])
            value["threads"] = spec["resources"]["threads_per_task"]
            value["loader_workers"] = spec["resources"]["loader_workers"]
            value["model"].update(pcr_readout=variant["pcr_readout"], spatial_readout=variant["spatial_readout"])
            run_cfg = from_dict(value)
            name = f"{variant['name']}_fold{fold_id}"
            config_path = output / "input_configs" / f"{name}.yaml"
            if config_path.exists():
                if load_config(config_path).digest != run_cfg.digest:
                    raise ValueError(f"Existing frozen config differs: {config_path}")
            else:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text(yaml.safe_dump(json.loads(json.dumps(run_cfg.to_dict())), sort_keys=False), encoding="utf-8")
                config_path.chmod(0o600)
            jobs.append({"name": name, "variant": variant["name"], "fold_id": fold_id,
                "seed": cv["train_seed"], "train_seed": cv["train_seed"], "cv_split_seed": cv["split_seed"],
                "config": str(config_path), "config_digest": run_cfg.digest, "config_sha256": digest(config_path),
                "manifest": str(manifest_path), "manifest_sha256": fold["manifest_sha256"],
                "data_signature": fold["data_signature"], "output": str(output / "runs" / name),
                "status": "queued", "retries": 0, "landmarks": "t0", "selection_cohort": "t0",
                "world_applicable": variant["jepa_weight"] > 0})
    if validation_ids != {_key(p) for p in dev}:
        raise ValueError("OOF validation sets do not partition development patients")
    protocol = {"schema": SCHEMA, "suite_name": spec["suite_name"], "spec_digest": stable_hash(spec), "cv": cv,
        "variants": spec["variants"], "inputs": inputs, "folds": folds,
        "selection": "per_fold_validation_patient_equal_F1_at_threshold_0.5",
        "ranking": "validation_fold_F1_arithmetic_mean_descending",
        "test_prediction": "arithmetic_mean_of_five_fold_probabilities_per_patient",
        "test_visibility": "withheld_until_all_15_jobs_completed", "final_refit": False,
        "independence_scope": "downstream_only", "report": spec["report"],
        "initialization": "original_codec_each_run; paired_common_modules_train_seed17",
        "retention": "delete_generated_checkpoints_only_after_verified_evaluations"}
    protocol_path = output / "suite_protocol.json"
    _immutable_json(protocol_path, protocol)
    state = {"schema": STATE_SCHEMA, "output_dir": str(output), "suite_name": spec["suite_name"],
        "spec": deepcopy(spec), "inputs": inputs, "codec": paths["codec"], "codec_sha256": inputs["codec"]["sha256"],
        "data_identity": str(data_path), "data_identity_sha256": digest(data_path),
        "protocol": protocol, "protocol_path": str(protocol_path), "protocol_sha256": digest(protocol_path),
        "folds": folds, "jobs": jobs}
    validate_inputs(state)
    return state


def _key(patient):
    return normalize_patient_key(patient["patient_key"])


def _label(patient):
    return int(patient["target"]["pcr"])


def validate_inputs(state):
    """Reject changed source, YAML, codec, fold configs/manifests or MRI metadata."""
    if state.get("schema") != STATE_SCHEMA:
        raise ValueError("Unsupported T0 CV state schema")
    _validate_spec(state["spec"])
    for identity in state["inputs"].values():
        _verify_identity(identity)
    if digest(state["codec"]) != state["codec_sha256"]:
        raise ValueError("Codec changed")
    for path_key, hash_key in (("protocol_path", "protocol_sha256"), ("data_identity", "data_identity_sha256")):
        if digest(state[path_key]) != state[hash_key]:
            raise ValueError(f"Frozen {path_key} changed")
    if read_json(state["protocol_path"]) != state["protocol"]:
        raise ValueError("Protocol and state differ")
    if state["protocol"]["spec_digest"] != stable_hash(state["spec"]):
        raise ValueError("Frozen suite specification changed")
    if state["inputs"] != state["protocol"]["inputs"] or state["folds"] != state["protocol"]["folds"]:
        raise ValueError("Frozen source/fold ownership changed")
    if state["codec"] != state["inputs"]["codec"]["path"]:
        raise ValueError("Codec path differs from registered source")
    for asset in read_json(state["data_identity"])["assets"]:
        stat = Path(asset["path"]).stat()
        if stat.st_size != asset["size"] or stat.st_mtime_ns != asset["mtime_ns"]:
            raise ValueError(f"MRI data identity changed: {asset['path']}")
    expected = {(v["name"], fold) for v in state["spec"]["variants"] for fold in range(1, 6)}
    jobs = state["jobs"]
    if len(jobs) != len(expected) or {(j["variant"], j["fold_id"]) for j in jobs} != expected:
        raise ValueError("Job identities must cover all three variants and five folds exactly")
    checked_folds = {}
    for job in jobs:
        fold = state["folds"][str(job["fold_id"])]
        if (job["manifest"], job["manifest_sha256"], job["data_signature"]) != (
                fold["manifest"], fold["manifest_sha256"], fold["data_signature"]):
            raise ValueError("Job manifest/fold assignment differs")
        config = load_config(job["config"])
        if digest(job["config"]) != job["config_sha256"] or config.digest != job["config_digest"]:
            raise ValueError("Frozen job config changed")
        if job["seed"] != 17 or config.seed != 17 or config.encoder_checkpoint != state["codec"]:
            raise ValueError("Job initialization or training seed differs")
        variant = next(v for v in state["spec"]["variants"] if v["name"] == job["variant"])
        expected_config = deepcopy(state["spec"]["training"])
        expected_config.update(seed=17, encoder_checkpoint=state["codec"], jepa_weight=variant["jepa_weight"],
            threads=state["spec"]["resources"]["threads_per_task"], loader_workers=state["spec"]["resources"]["loader_workers"])
        expected_config["model"].update(pcr_readout=variant["pcr_readout"], spatial_readout=variant["spatial_readout"])
        if config.digest != from_dict(expected_config).digest:
            raise ValueError("Job config belongs to a different variant")
        expected_name = f"{job['variant']}_fold{job['fold_id']}"
        root = Path(state["output_dir"]).resolve()
        if (job["name"] != expected_name or Path(job["config"]).resolve() != root / "input_configs" / (expected_name + ".yaml")
                or Path(job["output"]).resolve() != root / "runs" / expected_name):
            raise ValueError("Job artifact ownership differs from registered variant/fold")
        if job["fold_id"] in checked_folds:
            continue
        checked_folds[job["fold_id"]] = True
        for name in ("manifest", "audit", "normalization"):
            if digest(fold[name]) != fold[name + "_sha256"]:
                raise ValueError(f"Frozen fold {name} changed")
        store = RawMRIStore(fold["manifest"], config.image_shape, allow_synthetic=config.allow_synthetic)
        if stable_hash(store.signature()) != fold["data_signature"]:
            raise ValueError("Fold data signature changed")
        if store.normalization_state() != read_json(fold["normalization"]):
            raise ValueError("Fold train-only clinical normalization differs")
    return True


def _ci(values):
    finite = [x for x in values if x is not None]
    return {"lower": float(np.quantile(finite, .025)) if finite else None,
            "upper": float(np.quantile(finite, .975)) if finite else None, "valid_replicates": len(finite)}


def _with_positive_fraction(metrics):
    return {**metrics, "predicted_positive_fraction": (metrics["tp"] + metrics["fp"]) / metrics["n"]}


def _stratified_bootstrap(rows_by_variant, repetitions, seed):
    """Paired label-stratified bootstrap with the same patients for every model."""
    if not rows_by_variant:
        return {}, {}
    names = list(rows_by_variant)
    reference = rows_by_variant[names[0]]
    keys = [_key(row) for row in reference]
    labels = np.asarray([row["pcr"] for row in reference])
    probabilities = {}
    for name, rows in rows_by_variant.items():
        if [_key(r) for r in rows] != keys or [r["pcr"] for r in rows] != labels.tolist():
            raise ValueError("Paired bootstrap patient/label alignment differs")
        probabilities[name] = np.asarray([r["pcr_probability"] for r in rows])
    groups = [np.flatnonzero(labels == label) for label in (0, 1)]
    rng = np.random.default_rng(seed)
    draws = {name: {metric: [] for metric in REPORT_METRICS} for name in names}
    for _ in range(repetitions):
        index = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups if len(g)])
        for name in names:
            metrics = _with_positive_fraction(classification_metrics(labels[index], probabilities[name][index], f1_threshold=.5))
            for metric in REPORT_METRICS:
                draws[name][metric].append(metrics[metric])
    intervals = {name: {metric: _ci(values) for metric, values in result.items()} for name, result in draws.items()}
    differences = {}
    for left in names:
        for right in names:
            if names.index(left) >= names.index(right):
                continue
            differences[left + "__minus__" + right] = {metric: _ci([
                a - b if a is not None and b is not None else None
                for a, b in zip(draws[left][metric], draws[right][metric])]) for metric in REPORT_METRICS}
    return intervals, differences


def _prediction_rows(report, job):
    rows = []
    for row in report["rows"]:
        if row["pcr"] not in (0, 1) or not math.isfinite(row["pcr_probability"]) or not 0 <= row["pcr_probability"] <= 1:
            raise ValueError("Invalid T0 labelled probability")
        rows.append({**row, "variant": job["variant"], "fold_id": job["fold_id"], "train_seed": 17,
            "checkpoint": "best", "checkpoint_epoch": report.get("checkpoint_epoch", job.get("best_epoch")),
            "checkpoint_sha256": _checkpoint_sha(report), "evaluation": str(Path(job["output"]) / f"evaluation_best_{report['split']}.json")})
    return sorted(rows, key=_key)


def _checkpoint_sha(report):
    identity = report.get("checkpoint_identity", {})
    return report.get("checkpoint_sha256", identity.get("sha256") if isinstance(identity, dict) else None)


def _validate_report_owner(report, job, checkpoint):
    for field, expected in (("variant", job["variant"]), ("fold_id", job["fold_id"]),
                            ("train_seed", 17), ("checkpoint_kind", checkpoint)):
        if field in report and report[field] != expected:
            raise ValueError(f"Evaluation {field} belongs to another run/checkpoint")


def write_reports(output, state, bootstrap=None):
    """Verify fold reports, publish OOF and five-model mean-probability results."""
    output = Path(output).resolve()
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / ".report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        validate_inputs(state)
        return _write_reports(reports, state, bootstrap)


def _write_reports(output, state, bootstrap):
    jobs = state["jobs"]
    complete = all(j["status"] == "completed" for j in jobs)
    repetitions = state["spec"]["report"]["bootstrap"] if bootstrap is None else bootstrap
    if type(repetitions) is not int or repetitions < 0:
        raise ValueError("Bootstrap count must be a nonnegative integer")
    seed = state["spec"]["report"]["bootstrap_seed"]
    variants = [v["name"] for v in state["spec"]["variants"]]
    fold_rows, last_rows = [], []
    predictions = {split: {name: {} for name in variants} for split in ("val", "test")}
    for job in jobs:
        best_identity = None
        for split in ("val", "test"):
            withheld = split == "test" and not complete
            path = Path(job["output"]) / f"evaluation_best_{split}.json"
            report = None
            if not withheld and path.is_file():
                report = validate_classification_report(path, manifest=job["manifest"], split=split, profile="t0")
                _validate_report_owner(report, job, "best")
                current_identity = (_checkpoint_sha(report), report.get("checkpoint_epoch"))
                if best_identity is not None and any(a is not None and b is not None and a != b
                        for a, b in zip(best_identity, current_identity)):
                    raise ValueError("Validation and test must use the same best checkpoint")
                best_identity = current_identity
                predictions[split][job["variant"]][job["fold_id"]] = _prediction_rows(report, job)
            elif not withheld and job["status"] == "completed":
                raise ValueError(f"Completed fold lacks required evaluation: {path}")
            fold_rows.append({"variant": job["variant"], "fold_id": job["fold_id"], "train_seed": 17,
                "split": split, "timepoint": "T0", "status": "withheld" if withheld else "evaluated" if report else "pending",
                "checkpoint": "best", "epoch": report.get("checkpoint_epoch", job.get("best_epoch")) if report else None,
                "checkpoint_sha256": _checkpoint_sha(report) if report else None,
                **{key: _with_positive_fraction(report["metrics"]).get(key) if report else None for key in REPORT_METRICS + COUNTS}})
        last_path = Path(job["output"]) / "evaluation_last_val.json"
        if last_path.is_file():
            last = validate_classification_report(last_path, manifest=job["manifest"], split="val", profile="t0")
            _validate_report_owner(last, job, "last")
            last_rows.append({"variant": job["variant"], "fold_id": job["fold_id"], "train_seed": 17,
                "split": "val", "checkpoint": "last", "epoch": last.get("checkpoint_epoch"),
                **{key: _with_positive_fraction(last["metrics"]).get(key) for key in REPORT_METRICS + COUNTS}})
        elif job["status"] == "completed":
            raise ValueError(f"Completed fold lacks required last validation evaluation: {last_path}")
    fold_summary = []
    for name in variants:
        for split in ("val", "test"):
            selected = [r for r in fold_rows if r["variant"] == name and r["split"] == split and r["status"] == "evaluated"]
            result = {"variant": name, "split": split, "completed_folds": len(selected)}
            for metric in REPORT_METRICS:
                values = [r[metric] for r in selected if r[metric] is not None]
                result[metric + "_mean"] = statistics.mean(values) if len(values) == 5 else None
                result[metric + "_std"] = statistics.stdev(values) if len(values) == 5 else None
            fold_summary.append(result)
    ranking = sorted(variants, key=lambda name: next((
        (r["f1_mean"] is None, -(r["f1_mean"] or 0), variants.index(name))
        for r in fold_summary if r["variant"] == name and r["split"] == "val")))
    aggregates = {"oof": {}, "test_ensemble": {}}
    for name in variants:
        folds = predictions["val"][name]
        if len(folds) == 5:
            rows = sorted([row for fold in sorted(folds) for row in folds[fold]], key=_key)
            expected = {_key(p): _label(p) for p in read_json(jobs[0]["manifest"])["patients"] if p["split"] != "test"}
            if len(rows) != len(expected) or {_key(r): r["pcr"] for r in rows} != expected:
                raise ValueError("OOF predictions must cover every development patient exactly once")
            aggregates["oof"][name] = rows
        folds = predictions["test"][name]
        if complete:
            if len(folds) != 5:
                raise ValueError("Test ensemble requires all five fold predictions")
            expected = {_key(p): _label(p) for p in read_json(jobs[0]["manifest"])["patients"] if p["split"] == "test"}
            maps = [{_key(r): r for r in folds[fold]} for fold in range(1, 6)]
            if any(set(m) != set(expected) for m in maps):
                raise ValueError("Five fold test patient coverage differs")
            rows = []
            for key in sorted(expected):
                parts = [m[key] for m in maps]
                if any(r["pcr"] != expected[key] for r in parts):
                    raise ValueError("Test ensemble labels differ across folds")
                rows.append({"variant": name, "patient_key": parts[0]["patient_key"], "landmark": 0,
                    "pcr": expected[key], "pcr_probability": math.fsum(r["pcr_probability"] for r in parts) / 5,
                    "fold_probabilities": [r["pcr_probability"] for r in parts], "fold_count": 5,
                    "aggregation": "arithmetic_probability_mean"})
            aggregates["test_ensemble"][name] = rows
    aggregate_metrics, paired_intervals = [], {}
    cache_path = output / "bootstrap_cache.json"
    cache = read_json(cache_path) if cache_path.exists() else {}
    for kind, by_variant in aggregates.items():
        fingerprint = stable_hash({"rows": by_variant, "repetitions": repetitions, "seed": seed})
        if cache.get(kind, {}).get("fingerprint") == fingerprint:
            intervals, paired = cache[kind]["intervals"], cache[kind]["paired"]
        else:
            intervals, paired = _stratified_bootstrap(by_variant, repetitions, seed)
            cache[kind] = {"fingerprint": fingerprint, "intervals": intervals, "paired": paired}
        paired_intervals[kind] = paired
        for name, rows in by_variant.items():
            metrics = _with_positive_fraction(classification_metrics([r["pcr"] for r in rows], [r["pcr_probability"] for r in rows], f1_threshold=.5))
            aggregate_metrics.append({"variant": name, "population": kind, "status": "evaluated", **metrics,
                "bootstrap": {"method": "paired_label_stratified_patient_percentile_95", "repetitions": repetitions,
                              "seed": seed, "intervals": intervals.get(name, {})}})
    write_json(cache_path, cache)
    rank = {name: i for i, name in enumerate(ranking)}
    fold_rows.sort(key=lambda r: (rank[r["variant"]], r["split"] != "val", r["fold_id"]))
    fold_summary.sort(key=lambda r: (rank[r["variant"]], r["split"] != "val"))
    aggregate_metrics.sort(key=lambda r: (rank[r["variant"]], r["population"] != "oof"))
    value = {"schema": "t0_cv5_report_v1", "test_visible": complete, "suite_complete": complete,
        "ranking": ranking, "ranking_rule": state["protocol"]["ranking"],
        "fold_metrics": fold_rows, "fold_summary": fold_summary, "aggregate_metrics": aggregate_metrics,
        "supplementary_last_val": last_rows, "paired_bootstrap_differences": paired_intervals,
        "oof_rows": sum(map(len, aggregates["oof"].values())),
        "test_ensemble_rows": sum(map(len, aggregates["test_ensemble"].values())),
        "notes": ["Fivefold scores share the same historical test patients; fold SD is not an independent-test uncertainty estimate.",
                  "Primary ranking uses five validation F1 scores; OOF pooled metrics are auxiliary.",
                  "Primary test prediction averages five probabilities per patient before computing metrics.",
                  "Historical test has been examined previously; independence is limited to this downstream training round.",
                  "Image preprocessing and codec pretraining retain historical provenance; clinical scaling is fitted per fold.",
                  "Bootstrap intervals condition on these trained models and do not include training uncertainty.",
                  "No final all-development refit; all training seeds are 17, separate from CV split seed 20261020."]}
    write_json(output / "report.json", value)
    _table_csv(output / "fold_metrics.csv", fold_rows)
    _table_csv(output / "fold_summary.csv", fold_summary)
    _table_csv(output / "supplementary_last_val.csv", last_rows)
    _table_csv(output / "aggregate_metrics.csv", [{k: v for k, v in r.items() if k not in {"bootstrap", "reliability_bins"}}
        | {f"{m}_ci_{edge}": r["bootstrap"]["intervals"].get(m, {}).get(edge)
           for m in REPORT_METRICS for edge in ("lower", "upper")} for r in aggregate_metrics])
    for kind, filename in (("oof", "oof_predictions"), ("test_ensemble", "test_ensemble_predictions")):
        rows = [r for name in ranking for r in aggregates[kind].get(name, [])]
        write_json(output / f"{filename}.json", {"schema": "t0_cv_predictions_v1", "population": kind,
            "status": "withheld" if kind == "test_ensemble" and not complete else "available", "rows": rows})
        _table_csv(output / f"{filename}.csv", [dict(r, **({"fold_probabilities": json.dumps(r["fold_probabilities"])}
                   if "fold_probabilities" in r else {})) for r in rows])
    fold_test_rows = [row for name in ranking for fold in sorted(predictions["test"][name])
                     for row in predictions["test"][name][fold]]
    write_json(output / "fold_test_predictions.json", {"schema": "t0_cv_predictions_v1", "population": "fold_test",
        "status": "available" if complete else "withheld", "rows": fold_test_rows})
    _table_csv(output / "fold_test_predictions.csv", fold_test_rows)
    def table(rows, columns):
        return "<table><thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in columns) + "</tr></thead><tbody>" + "".join(
            "<tr>" + "".join("<td>" + html.escape(str(r.get(c, ""))) + "</td>" for c in columns) + "</tr>" for r in rows) + "</tbody></table>"
    aggregate_display = [dict(r, **{f"{metric}_ci_{edge}": r["bootstrap"]["intervals"].get(metric, {}).get(edge)
                         for metric in REPORT_METRICS for edge in ("lower", "upper")}) for r in aggregate_metrics]
    links = []
    for job in jobs:
        for filename in ("train_report.json", "diagnostics.json", "evaluation_best_val.json", "evaluation_last_val.json",
                         "world_best_val.json", "world_last_val.json") + (("evaluation_best_test.json", "world_best_test.json") if complete else ()):
            path = Path(job["output"]) / filename
            if path.is_file():
                links.append(f'<li><a href="{html.escape(os.path.relpath(path, output), quote=True)}">{html.escape(job["name"] + "/" + filename)}</a></li>')
    document = "<!doctype html><html lang='zh'><meta charset='utf-8'><title>T0五折比较</title>" \
        "<style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse;margin:16px 0}th,td{border:1px solid #ddd;padding:6px}</style>" \
        "<h1>3个T0配置 × 5折</h1><p>主排序：五折验证F1均值；测试为五折逐患者概率均值。测试仅在15个任务全部完成后公布。</p>" \
        + "".join("<p>" + html.escape(note) + "</p>" for note in value["notes"]) \
        + "<h2>五折均值与样本标准差</h2>" + table(fold_summary, ["variant", "split", "completed_folds"] + [f"{m}_{s}" for m in REPORT_METRICS for s in ("mean", "std")]) \
        + "<h2>OOF与测试概率集成及95%置信区间</h2>" + table(aggregate_display,
            ["variant", "population", *REPORT_METRICS, *COUNTS] +
            [f"{metric}_ci_{edge}" for metric in REPORT_METRICS for edge in ("lower", "upper")]) \
        + "<h2>各折结果</h2>" + table(fold_rows, ["variant", "fold_id", "split", "status", "epoch", *REPORT_METRICS, *COUNTS]) \
        + "<h2>保留的诊断与逐例结果</h2><ul>" + "".join(links) + "</ul></html>"
    (output / "report.html").write_text(document, encoding="utf-8")
    for path in output.iterdir():
        if path.is_file():
            path.chmod(0o600)
    return value

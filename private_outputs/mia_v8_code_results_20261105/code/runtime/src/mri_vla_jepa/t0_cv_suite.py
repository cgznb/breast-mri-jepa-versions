"""Autonomous medium models, paired fivefold T0-T2 training and T0 reports.

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
from .landmark_profiles import profile_stages
from .metrics import classification_metrics
from .train_config import from_dict, load_config

SCHEMA = "medium_cv5_t0_t2_autonomous_suite_v1"
STATE_SCHEMA = "medium_cv5_t0_t2_autonomous_state_v1"
REPORT_METRICS = METRICS + ("predicted_positive_fraction",)
VARIANTS = (
    {"name": "forecast_attention", "pcr_readout": "forecast", "jepa_weight": 0., "rollout_weight": .5, "spatial_readout": "attention"},
    {"name": "direct_no_jepa", "pcr_readout": "direct", "jepa_weight": 0., "rollout_weight": 0., "spatial_readout": "mean"},
    {"name": "direct_readout", "pcr_readout": "direct", "jepa_weight": 0., "rollout_weight": .5, "spatial_readout": "mean"},
    {"name": "forecast_mean", "pcr_readout": "forecast", "jepa_weight": 0., "rollout_weight": .5, "spatial_readout": "mean"},
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


def train_seeds(cv):
    """Training randomness only; holdout and CV assignments do not use these."""
    seeds = cv.get("train_seeds", [cv["train_seed"]])
    if (not isinstance(seeds, list) or not seeds or
            any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError("cv.train_seeds must be a nonempty list of unique nonnegative integers")
    return list(seeds)


def _job_name(variant, seed, fold, seeds):
    return f"{variant}_seed{seed}_fold{fold}" if len(seeds) > 1 else f"{variant}_fold{fold}"


def _validate_spec(spec):
    if not isinstance(spec, dict) or spec.get("schema") != SCHEMA:
        raise ValueError("Unsupported medium CV suite schema")
    name = spec.get("suite_name")
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("suite_name must be one directory name")
    if set(spec.get("paths", {})) != set(PATH_FIELDS):
        raise ValueError("Suite paths must declare all five input/output roots")
    if spec.get("variants") != list(VARIANTS):
        raise ValueError("Exactly the four registered medium variants are required")
    cv = spec["cv"]
    for key in ("n_splits", "expected_patients", "test_size"):
        _positive_integer(cv[key], key)
    if cv["n_splits"] != 5 or cv["test_size"] >= cv["expected_patients"]:
        raise ValueError("Expected five folds and a nonempty development population")
    for key in ("split_seed", "train_seed", "holdout_seed"):
        if type(cv[key]) is not int or cv[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    train_seeds(cv)
    if (cv["holdout_seed"], cv["split_seed"]) != (17, 20261020):
        raise ValueError("Holdout seed is 17 and CV split seed is 20261020")
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
    if cfg.seed != cv["train_seed"] or cfg.landmarks != "t0_t2" or cfg.selection_metric != "f1" or cfg.f1_threshold != .5:
        raise ValueError("Suite requires matching template train seed, T0-T2 training landmarks and fixed F1 threshold 0.5")
    if (cfg.prediction_training, cfg.jepa_weight, cfg.rollout_weight) != ("hybrid", 0., .5):
        raise ValueError("Autonomous template requires hybrid prediction training, zero teacher-forced JEPA and rollout weight 0.5")
    if not cfg.model.enable_world_model:
        raise ValueError("The four registered variants preserve the enabled world structure")
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
                    "prediction_training": "hybrid", "warmup_epochs": 10,
                    "warmup_start_factor": .1, "cosine_end_epoch": 100, "cosine_min_lr": 1e-6,
                    "task_weight": 1., "jepa_weight": 0., "rollout_weight": .5, "flow_weight": 0.,
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
                cfg.model.pcr_delta_reference) != ("medium", "all", .999, .3, True, "forecast", "current"):
            raise ValueError("Formal medium/all model settings changed")
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
    """Freeze five shared manifests and per-seed configs; runner owns matrix."""
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
    seeds = train_seeds(cv)
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
    _immutable_json(data_path, {"schema": "medium_cv_asset_identity_v1", "assets": inventory})
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
        if "train_seeds" in cv:
            manifest["provenance"].update(train_seeds=seeds,
                training_seed_scope="model_randomness_only; all_training_seeds_share_this_fold")
        folder = output / "splits" / f"fold{fold_id}"
        manifest_path = folder / "manifest.json"
        _immutable_json(manifest_path, manifest)
        fold_audit = {"schema": "medium_cv_fold_audit_v1", "fold_id": fold_id, **cv,
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
        for train_seed in seeds:
            for variant in spec["variants"]:
                value = cfg.to_dict()
                value.update(seed=train_seed, encoder_checkpoint=paths["codec"], jepa_weight=variant["jepa_weight"],
                    rollout_weight=variant["rollout_weight"])
                value["threads"] = spec["resources"]["threads_per_task"]
                value["loader_workers"] = spec["resources"]["loader_workers"]
                value["model"].update(pcr_readout=variant["pcr_readout"], spatial_readout=variant["spatial_readout"])
                run_cfg = from_dict(value)
                name = _job_name(variant["name"], train_seed, fold_id, seeds)
                config_path = output / "input_configs" / f"{name}.yaml"
                if config_path.exists():
                    if load_config(config_path).digest != run_cfg.digest:
                        raise ValueError(f"Existing frozen config differs: {config_path}")
                else:
                    config_path.parent.mkdir(parents=True, exist_ok=True)
                    config_path.write_text(yaml.safe_dump(json.loads(json.dumps(run_cfg.to_dict())), sort_keys=False), encoding="utf-8")
                    config_path.chmod(0o600)
                jobs.append({"name": name, "variant": variant["name"], "fold_id": fold_id,
                    "seed": train_seed, "train_seed": train_seed, "cv_split_seed": cv["split_seed"],
                    "config": str(config_path), "config_digest": run_cfg.digest, "config_sha256": digest(config_path),
                    "manifest": str(manifest_path), "manifest_sha256": fold["manifest_sha256"],
                    "data_signature": fold["data_signature"], "output": str(output / "runs" / name),
                    "status": "queued", "retries": 0, "landmarks": run_cfg.landmarks, "selection_cohort": run_cfg.landmarks,
                    "world_applicable": bool(variant["jepa_weight"] or variant["rollout_weight"])})
    if validation_ids != {_key(p) for p in dev}:
        raise ValueError("OOF validation sets do not partition development patients")
    protocol = {"schema": SCHEMA, "suite_name": spec["suite_name"], "spec_digest": stable_hash(spec), "cv": cv,
        "variants": spec["variants"], "inputs": inputs, "folds": folds,
        "training_landmarks": cfg.landmarks, "checkpoint_selection_cohort": cfg.landmarks,
        "reporting_landmark": 0,
        "selection": ("per_fold_validation_T0_F1_at_threshold_0.5" if cfg.landmarks == "t0" else
                      "per_fold_validation_patient_equal_prefix_F1_at_threshold_0.5"),
        "ranking": "validation_T0_fold_F1_arithmetic_mean_descending",
        "test_prediction": "arithmetic_mean_of_five_fold_T0_probabilities_per_patient",
        "test_visibility": f"withheld_until_all_{len(jobs)}_jobs_completed", "final_refit": False,
        "independence_scope": "downstream_only", "report": spec["report"],
        "initialization": ("original_codec_each_run; paired_common_modules_train_seed17" if seeds == [17]
                           else "original_codec_each_run; paired_common_modules_within_each_train_seed"),
        "retention": "delete_generated_checkpoints_only_after_verified_evaluations"}
    if cfg.landmarks == "t0_t2":
        protocol.update(
            training_landmark_stages=list(profile_stages(cfg.landmarks)),
            checkpoint_selection_landmark_stages=list(profile_stages(cfg.landmarks)),
            auxiliary_reporting_landmark_stages=list(profile_stages(cfg.landmarks)),
            canonical_mri_stages=[0, 1, 2, 3],
            future_supervision_target_stages=[1, 2, 3],
            t3_role="autonomous_rollout_supervision_target_only; never_a_classification_landmark_or_rollout_predecessor",
            selection="per_fold_validation_T0_T1_T2_patient_equal_prefix_F1_at_threshold_0.5")
    if "train_seeds" in cv:
        protocol.update(train_seeds=seeds, training_runs=len(jobs),
                        training_seed_scope="model_randomness_only; identical_holdout_and_five_folds",
                        ranking="validation_T0_mean_of_per_seed_five_fold_F1_means_descending",
                        test_prediction="within_each_seed_arithmetic_mean_of_five_fold_T0_probabilities_per_patient")
    protocol_path = output / "suite_protocol.json"
    _immutable_json(protocol_path, protocol)
    state = {"schema": STATE_SCHEMA, "output_dir": str(output), "suite_name": spec["suite_name"],
        "spec": deepcopy(spec), "inputs": inputs, "codec": paths["codec"], "codec_sha256": inputs["codec"]["sha256"],
        "data_identity": str(data_path), "data_identity_sha256": digest(data_path),
        "protocol": protocol, "protocol_path": str(protocol_path), "protocol_sha256": digest(protocol_path),
        "folds": folds, "jobs": jobs}
    if "train_seeds" in cv:
        state.update(train_seeds=seeds, training_runs=len(jobs))
    validate_inputs(state)
    return state


def _key(patient):
    return normalize_patient_key(patient["patient_key"])


def _label(patient):
    return int(patient["target"]["pcr"])


def validate_inputs(state):
    """Reject changed source, YAML, codec, fold configs/manifests or MRI metadata."""
    if state.get("schema") != STATE_SCHEMA:
        raise ValueError("Unsupported medium CV state schema")
    cfg = _validate_spec(state["spec"])
    if (state["protocol"].get("training_landmarks") != cfg.landmarks
            or state["protocol"].get("checkpoint_selection_cohort") != cfg.landmarks
            or state["protocol"].get("reporting_landmark") != 0):
        raise ValueError("Frozen training, checkpoint-selection or reporting landmarks differ")
    if cfg.landmarks == "t0_t2":
        for field in ("training_landmark_stages", "checkpoint_selection_landmark_stages",
                      "auxiliary_reporting_landmark_stages"):
            if state["protocol"].get(field) != list(profile_stages(cfg.landmarks)):
                raise ValueError(f"Frozen {field} differs from T0-T2 classification scope")
        if (state["protocol"].get("canonical_mri_stages") != [0, 1, 2, 3]
                or state["protocol"].get("future_supervision_target_stages") != [1, 2, 3]):
            raise ValueError("Frozen canonical MRI or future-supervision target stages differ")
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
    seeds = train_seeds(state["spec"]["cv"])
    expected_runs = len(state["spec"]["variants"]) * state["spec"]["cv"]["n_splits"] * len(seeds)
    if state.get("train_seeds", seeds) != seeds or state.get("training_runs", expected_runs) != expected_runs:
        raise ValueError("Registered training seeds or run count changed")
    if (state["protocol"].get("train_seeds", seeds) != seeds or
            state["protocol"].get("training_runs", expected_runs) != expected_runs):
        raise ValueError("Protocol training seeds or run count changed")
    expected = {(v["name"], seed, fold) for v in state["spec"]["variants"] for seed in seeds for fold in range(1, state["spec"]["cv"]["n_splits"] + 1)}
    jobs = state["jobs"]
    if len(jobs) != len(expected) or {(j["variant"], j["train_seed"], j["fold_id"]) for j in jobs} != expected:
        raise ValueError("Job identities must cover all registered variants, training seeds and five folds exactly")
    checked_folds = {}
    for job in jobs:
        fold = state["folds"][str(job["fold_id"])]
        if (job["manifest"], job["manifest_sha256"], job["data_signature"]) != (
                fold["manifest"], fold["manifest_sha256"], fold["data_signature"]):
            raise ValueError("Job manifest/fold assignment differs")
        config = load_config(job["config"])
        if digest(job["config"]) != job["config_sha256"] or config.digest != job["config_digest"]:
            raise ValueError("Frozen job config changed")
        if job["seed"] != job["train_seed"] or config.seed != job["train_seed"] or config.encoder_checkpoint != state["codec"]:
            raise ValueError("Job initialization or training seed differs")
        if job.get("landmarks") != config.landmarks or job.get("selection_cohort") != config.landmarks:
            raise ValueError("Job training landmarks or checkpoint-selection cohort differ")
        variant = next(v for v in state["spec"]["variants"] if v["name"] == job["variant"])
        if job.get("world_applicable") != bool(variant["jepa_weight"] or variant["rollout_weight"]):
            raise ValueError("Job world applicability differs from its autonomous supervision weights")
        expected_config = deepcopy(state["spec"]["training"])
        expected_config.update(seed=job["train_seed"], encoder_checkpoint=state["codec"], jepa_weight=variant["jepa_weight"],
            rollout_weight=variant["rollout_weight"],
            threads=state["spec"]["resources"]["threads_per_task"], loader_workers=state["spec"]["resources"]["loader_workers"])
        expected_config["model"].update(pcr_readout=variant["pcr_readout"], spatial_readout=variant["spatial_readout"])
        if config.digest != from_dict(expected_config).digest:
            raise ValueError("Job config belongs to a different variant")
        expected_name = _job_name(job["variant"], job["train_seed"], job["fold_id"], seeds)
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
        if row["landmark"] != 0:
            continue
        if row["pcr"] not in (0, 1) or not math.isfinite(row["pcr_probability"]) or not 0 <= row["pcr_probability"] <= 1:
            raise ValueError("Invalid T0 labelled probability")
        rows.append({**row, "variant": job["variant"], "fold_id": job["fold_id"], "train_seed": job["train_seed"],
            "checkpoint": "best", "checkpoint_epoch": report.get("checkpoint_epoch", job.get("best_epoch")),
            "checkpoint_sha256": _checkpoint_sha(report), "evaluation": str(Path(job["output"]) / f"evaluation_best_{report['split']}.json")})
    return sorted(rows, key=_key)


def _checkpoint_sha(report):
    identity = report.get("checkpoint_identity", {})
    return report.get("checkpoint_sha256", identity.get("sha256") if isinstance(identity, dict) else None)


def _validate_report_owner(report, job, checkpoint):
    for field, expected in (("variant", job["variant"]), ("fold_id", job["fold_id"]),
                            ("train_seed", job["train_seed"]), ("config_digest", job["config_digest"]),
                            ("checkpoint_kind", checkpoint)):
        if field in report and report[field] != expected:
            raise ValueError(f"Evaluation {field} belongs to another run/checkpoint")


def _t0_metrics(report):
    """Extract the common reporting landmark after full-cohort validation."""
    metrics = report["per_landmark"]["T0"]
    if not metrics.get("n"):
        raise ValueError("A medium comparison report requires labelled T0 patients")
    return metrics


def _evaluation_details(report, job, split, checkpoint, withheld=False):
    """Keep checkpoint-selection scores and each landmark separate from T0 tables."""
    if report:
        expected = "t0_direct_f1" if job["landmarks"] == "t0" else "patient_equal_prefix_f1"
        if (report.get("selection_metric") != expected or report.get("selection_protocol") != expected
                or not isinstance(report.get("selection_score"), (int, float))
                or not math.isclose(report["selection_score"], report["metrics"]["f1"], rel_tol=1e-9, abs_tol=1e-10)):
            raise ValueError("Saved selection score or protocol differs from the validated training cohort F1")
    common = {"variant": job["variant"], "train_seed": job["train_seed"], "fold_id": job["fold_id"],
        "split": split, "checkpoint": checkpoint, "evaluation": f"{checkpoint}_{split}",
        "training_landmarks": job["landmarks"], "selection_cohort": job["selection_cohort"],
        "status": "withheld" if withheld else "evaluated" if report else "pending",
        "epoch": report.get("checkpoint_epoch", job.get("best_epoch") if checkpoint == "best" else None) if report else None,
        "checkpoint_sha256": _checkpoint_sha(report) if report else None}
    selection = {**common, "selection_protocol": report.get("selection_protocol") if report else None,
        "selection_metric": report.get("selection_metric") if report else None,
        "selection_score": report.get("selection_score") if report else None,
        **{key: report["metrics"].get(key) if report else None for key in REPORT_METRICS + COUNTS}}
    if report:
        selection["predicted_positive_fraction"] = _with_positive_fraction(report["metrics"])["predicted_positive_fraction"]
    stages = profile_stages(job["landmarks"])
    landmarks = []
    for stage in stages:
        metrics = report["per_landmark"][f"T{stage}"] if report else None
        row = {**common, "landmark": stage, "timepoint": f"T{stage}",
            **{key: metrics.get(key) if metrics else None for key in REPORT_METRICS + COUNTS}}
        if metrics and metrics.get("n"):
            row["predicted_positive_fraction"] = _with_positive_fraction(metrics)["predicted_positive_fraction"]
        elif report:
            row["status"] = "no_patients"
        landmarks.append(row)
    return selection, landmarks


def _write_auxiliary_reports(output, selection_rows, landmark_rows, complete):
    for name, rows in (("selection_metrics", selection_rows), ("landmark_metrics", landmark_rows)):
        write_json(output / f"{name}.json", {"schema": f"medium_cv5_{name}_v1", "test_visible": complete,
            "reporting_landmark": 0, "rows": rows})
        _table_csv(output / f"{name}.csv", rows)


def _report_scope(state):
    profile = state["spec"]["training"]["landmarks"]
    scope = {"training_landmarks": profile, "checkpoint_selection_cohort": profile, "reporting_landmark": 0}
    if profile == "t0_t2":
        scope.update(training_landmark_stages=list(profile_stages(profile)),
                     checkpoint_selection_landmark_stages=list(profile_stages(profile)),
                     auxiliary_reporting_landmark_stages=list(profile_stages(profile)))
    return scope


def write_reports(output, state, bootstrap=None):
    """Verify fold reports, publish OOF and five-model mean-probability results."""
    output = Path(output).resolve()
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / ".report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        validate_inputs(state)
        if len(train_seeds(state["spec"]["cv"])) > 1:
            return _write_multiseed_reports(reports, state, bootstrap)
        return _write_reports(reports, state, bootstrap)


def _write_multiseed_reports(output, state, bootstrap):
    """Keep five-model ensembles and bootstrap draws inside each seed."""
    seeds = train_seeds(state["spec"]["cv"])
    complete = all(job["status"] == "completed" for job in state["jobs"])
    variants = [variant["name"] for variant in state["spec"]["variants"]]
    per_seed = {}
    for training_seed in seeds:
        seed_state = dict(state, jobs=[job for job in state["jobs"] if job["train_seed"] == training_seed])
        seed_state["protocol"] = dict(state["protocol"], ranking="validation_T0_fold_F1_arithmetic_mean_descending")
        per_seed[training_seed] = _write_reports(output / f"seed{training_seed}", seed_state, bootstrap,
                                               complete_override=complete)
    fold_rows = [row for seed in seeds for row in per_seed[seed]["fold_metrics"]]
    fold_summary = [row for seed in seeds for row in per_seed[seed]["fold_summary"]]
    aggregate_metrics = [row for seed in seeds for row in per_seed[seed]["aggregate_metrics"]]
    last_rows = [row for seed in seeds for row in per_seed[seed]["supplementary_last_val"]]
    selection_rows = [row for seed in seeds for row in per_seed[seed]["selection_metrics"]]
    landmark_rows = [row for seed in seeds for row in per_seed[seed]["landmark_metrics"]]
    seed_metrics = []
    for row in fold_summary:
        if row["split"] == "val":
            seed_metrics.append({"variant": row["variant"], "train_seed": row["train_seed"],
                "population": "validation_fold_mean", "status": "evaluated" if row["completed_folds"] == 5 else "pending",
                **{metric: row[metric + "_mean"] for metric in REPORT_METRICS}})
    seed_metrics.extend(aggregate_metrics)
    seed_summary = []
    for name in variants:
        for population in ("validation_fold_mean", "oof", "test_ensemble"):
            selected = [row for row in seed_metrics if row["variant"] == name and row["population"] == population
                        and row["status"] == "evaluated"]
            result = {"variant": name, "population": population, "expected_seeds": len(seeds),
                      "completed_seeds": len(selected), "sd_scope": "sample_sd_across_training_seed_results"}
            for metric in REPORT_METRICS:
                values = [row[metric] for row in selected if row[metric] is not None]
                result[metric + "_mean"] = statistics.mean(values) if len(values) == len(seeds) else None
                result[metric + "_std"] = statistics.stdev(values) if len(values) == len(seeds) else None
            seed_summary.append(result)
    validation = {row["variant"]: row["f1_mean"] for row in seed_summary if row["population"] == "validation_fold_mean"}
    ranking = sorted(variants, key=lambda name: (validation[name] is None, -(validation[name] or 0), variants.index(name)))
    rank = {name: i for i, name in enumerate(ranking)}
    seed_rank = {seed: i for i, seed in enumerate(seeds)}
    fold_rows.sort(key=lambda row: (rank[row["variant"]], seed_rank[row["train_seed"]], row["split"] != "val", row["fold_id"]))
    fold_summary.sort(key=lambda row: (rank[row["variant"]], seed_rank[row["train_seed"]], row["split"] != "val"))
    aggregate_metrics.sort(key=lambda row: (rank[row["variant"]], seed_rank[row["train_seed"]], row["population"] != "oof"))
    seed_summary.sort(key=lambda row: (rank[row["variant"]], ("validation_fold_mean", "oof", "test_ensemble").index(row["population"])))
    value = {"schema": "medium_cv5_multiseed_report_v1", "test_visible": complete, "suite_complete": complete,
        **_report_scope(state), "selection_metrics": selection_rows, "landmark_metrics": landmark_rows,
        "train_seeds": seeds, "training_runs": len(state["jobs"]), "ranking": ranking,
        "ranking_rule": state["protocol"]["ranking"], "fold_metrics": fold_rows, "fold_summary": fold_summary,
        "aggregate_metrics": aggregate_metrics, "seed_summary": seed_summary, "supplementary_last_val": last_rows,
        "oof_rows": sum(report["oof_rows"] for report in per_seed.values()),
        "test_ensemble_rows": sum(report["test_ensemble_rows"] for report in per_seed.values()),
        "per_seed_reports": {str(seed): f"seed{seed}/report.json" for seed in seeds},
        "notes": ["Training seeds change model randomness only; the historical holdout and five CV folds are identical.",
                  "Each seed produces a separate five-fold probability ensemble; no cross-seed ensemble is used.",
                  "Ranking first averages five T0 validation F1 scores inside each seed, then averages the seed means; checkpoint selection retains the training landmark cohort.",
                  "Fold-summary SD uses five folds inside one seed. Seed-summary SD uses the independent training-seed results.",
                  "Test-summary mean and sample SD are computed across per-seed five-model ensemble metrics, not individual fold scores.",
                  "Bootstrap intervals are computed separately for each seed's probability ensemble and condition on its trained models.",
                  "Seeds share the same historical test patients; training-seed SD is not independent-test uncertainty.",
                  "Historical test has already been examined; independence is limited to this downstream training round.",
                  "Test scores and predictions are withheld until every registered job in the whole suite completes."]}
    write_json(output / "report.json", value)
    _write_auxiliary_reports(output, selection_rows, landmark_rows, complete)
    for filename, rows in (("fold_metrics", fold_rows), ("fold_summary", fold_summary),
                           ("seed_summary", seed_summary), ("supplementary_last_val", last_rows)):
        _table_csv(output / f"{filename}.csv", rows)
    _table_csv(output / "aggregate_metrics.csv", [{key: item for key, item in row.items() if key not in {"bootstrap", "reliability_bins"}}
        | {f"{metric}_ci_{edge}": row["bootstrap"]["intervals"].get(metric, {}).get(edge)
           for metric in REPORT_METRICS for edge in ("lower", "upper")} for row in aggregate_metrics])
    for population, filename in (("oof", "oof_predictions"), ("fold_test", "fold_test_predictions"),
                                  ("test_ensemble", "test_ensemble_predictions")):
        rows = [row for seed in seeds for row in read_json(output / f"seed{seed}" / f"{filename}.json")["rows"]]
        rows.sort(key=lambda row: (rank[row["variant"]], seed_rank[row["train_seed"]], row.get("fold_id", 0), _key(row)))
        write_json(output / f"{filename}.json", {"schema": "medium_cv_predictions_v1", "population": population,
            "status": "withheld" if population != "oof" and not complete else "available", "rows": rows})
        _table_csv(output / f"{filename}.csv", [dict(row, **({"fold_probabilities": json.dumps(row["fold_probabilities"])}
                    if "fold_probabilities" in row else {})) for row in rows])
    def table(rows, columns):
        return "<table><thead><tr>" + "".join(f"<th>{html.escape(column)}</th>" for column in columns) + "</tr></thead><tbody>" + "".join(
            "<tr>" + "".join("<td>" + html.escape(str(row.get(column, ""))) + "</td>" for column in columns) + "</tr>" for row in rows) + "</tbody></table>"
    document = "<!doctype html><html lang='zh'><meta charset='utf-8'><title>Medium五折多种子T0比较</title>" \
        "<style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse;margin:16px 0}th,td{border:1px solid #ddd;padding:6px}</style>" \
        f"<h1>{len(state['spec']['variants'])}个medium变体 × {len(seeds)}个训练种子 × {state['spec']['cv']['n_splits']}折，共{len(state['jobs'])}次训练</h1>" \
        f"<p>训练时间点：{html.escape(value['training_landmarks'])}；checkpoint按该训练口径F1选优。主表和排序统一使用T0；各seed独立汇总五折T0测试概率，跨seed报告指标均值和样本标准差。</p>" \
        + "".join("<p>" + html.escape(note) + "</p>" for note in value["notes"]) \
        + "<h2>跨训练种子均值与样本标准差</h2>" + table(seed_summary, ["variant", "population", "completed_seeds", "expected_seeds"] + [f"{metric}_{part}" for metric in REPORT_METRICS for part in ("mean", "std")]) \
        + "<h2>逐种子OOF及五折测试集成</h2>" + table(aggregate_metrics, ["variant", "train_seed", "population", *REPORT_METRICS, *COUNTS]) \
        + "<h2>逐种子逐折</h2>" + table(fold_rows, ["variant", "train_seed", "fold_id", "split", "status", "epoch", *REPORT_METRICS, *COUNTS]) \
        + "<h2>每个种子的完整报告和置信区间</h2><ul>" + "".join(f'<li><a href="seed{seed}/report.html">seed {seed}</a></li>' for seed in seeds) + "</ul></html>"
    (output / "report.html").write_text(document, encoding="utf-8")
    for path in output.iterdir():
        if path.is_file():
            path.chmod(0o600)
    return value


def _write_reports(output, state, bootstrap, complete_override=None):
    output.mkdir(parents=True, exist_ok=True)
    jobs = state["jobs"]
    complete = all(j["status"] == "completed" for j in jobs) if complete_override is None else complete_override
    training_seeds = {job["train_seed"] for job in jobs}
    if len(training_seeds) != 1:
        raise ValueError("A five-fold report must contain exactly one training seed")
    training_seed = training_seeds.pop()
    repetitions = state["spec"]["report"]["bootstrap"] if bootstrap is None else bootstrap
    if type(repetitions) is not int or repetitions < 0:
        raise ValueError("Bootstrap count must be a nonnegative integer")
    seed = state["spec"]["report"]["bootstrap_seed"]
    variants = [v["name"] for v in state["spec"]["variants"]]
    fold_rows, last_rows, selection_rows, landmark_rows = [], [], [], []
    predictions = {split: {name: {} for name in variants} for split in ("val", "test")}
    for job in jobs:
        best_identity = None
        for split in ("val", "test"):
            withheld = split == "test" and not complete
            path = Path(job["output"]) / f"evaluation_best_{split}.json"
            report = None
            if not withheld and path.is_file():
                report = validate_classification_report(path, manifest=job["manifest"], split=split, profile=job["landmarks"])
                _validate_report_owner(report, job, "best")
                current_identity = (_checkpoint_sha(report), report.get("checkpoint_epoch"))
                if best_identity is not None and any(a is not None and b is not None and a != b
                        for a, b in zip(best_identity, current_identity)):
                    raise ValueError("Validation and test must use the same best checkpoint")
                best_identity = current_identity
                predictions[split][job["variant"]][job["fold_id"]] = _prediction_rows(report, job)
            elif not withheld and job["status"] == "completed":
                raise ValueError(f"Completed fold lacks required evaluation: {path}")
            selection, landmarks = _evaluation_details(report, job, split, "best", withheld)
            selection_rows.append(selection)
            landmark_rows.extend(landmarks)
            fold_rows.append({"variant": job["variant"], "fold_id": job["fold_id"], "train_seed": job["train_seed"],
                "split": split, "timepoint": "T0", "status": "withheld" if withheld else "evaluated" if report else "pending",
                "training_landmarks": job["landmarks"], "selection_cohort": job["selection_cohort"],
                "checkpoint": "best", "epoch": report.get("checkpoint_epoch", job.get("best_epoch")) if report else None,
                "checkpoint_sha256": _checkpoint_sha(report) if report else None,
                **{key: _with_positive_fraction(_t0_metrics(report)).get(key) if report else None for key in REPORT_METRICS + COUNTS}})
        last_path = Path(job["output"]) / "evaluation_last_val.json"
        last = None
        if last_path.is_file():
            last = validate_classification_report(last_path, manifest=job["manifest"], split="val", profile=job["landmarks"])
            _validate_report_owner(last, job, "last")
            last_rows.append({"variant": job["variant"], "fold_id": job["fold_id"], "train_seed": job["train_seed"],
                "split": "val", "checkpoint": "last", "epoch": last.get("checkpoint_epoch"),
                **{key: _with_positive_fraction(_t0_metrics(last)).get(key) for key in REPORT_METRICS + COUNTS}})
        elif job["status"] == "completed":
            raise ValueError(f"Completed fold lacks required last validation evaluation: {last_path}")
        selection, landmarks = _evaluation_details(last, job, "val", "last")
        selection_rows.append(selection)
        landmark_rows.extend(landmarks)
    fold_summary = []
    for name in variants:
        for split in ("val", "test"):
            selected = [r for r in fold_rows if r["variant"] == name and r["split"] == split and r["status"] == "evaluated"]
            result = {"variant": name, "train_seed": training_seed, "split": split, "completed_folds": len(selected)}
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
                rows.append({"variant": name, "train_seed": training_seed, "patient_key": parts[0]["patient_key"], "landmark": 0,
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
            aggregate_metrics.append({"variant": name, "train_seed": training_seed, "population": kind, "status": "evaluated", **metrics,
                "bootstrap": {"method": "paired_label_stratified_patient_percentile_95", "repetitions": repetitions,
                              "seed": seed, "intervals": intervals.get(name, {})}})
    write_json(cache_path, cache)
    rank = {name: i for i, name in enumerate(ranking)}
    fold_rows.sort(key=lambda r: (rank[r["variant"]], r["split"] != "val", r["fold_id"]))
    fold_summary.sort(key=lambda r: (rank[r["variant"]], r["split"] != "val"))
    aggregate_metrics.sort(key=lambda r: (rank[r["variant"]], r["population"] != "oof"))
    value = {"schema": "medium_cv5_report_v1", "test_visible": complete, "suite_complete": complete,
        **_report_scope(state), "selection_metrics": selection_rows, "landmark_metrics": landmark_rows,
        "train_seed": training_seed,
        "ranking": ranking, "ranking_rule": state["protocol"]["ranking"],
        "fold_metrics": fold_rows, "fold_summary": fold_summary, "aggregate_metrics": aggregate_metrics,
        "supplementary_last_val": last_rows, "paired_bootstrap_differences": paired_intervals,
        "oof_rows": sum(map(len, aggregates["oof"].values())),
        "test_ensemble_rows": sum(map(len, aggregates["test_ensemble"].values())),
        "notes": ["Fivefold scores share the same historical test patients; fold SD is not an independent-test uncertainty estimate.",
                  "Primary ranking uses five T0 validation F1 scores; checkpoint selection retains the training landmark cohort, and OOF pooled T0 metrics are auxiliary.",
                  "Primary test prediction averages five T0 probabilities per patient before computing metrics.",
                  "Historical test has been examined previously; independence is limited to this downstream training round.",
                  "Image preprocessing and codec pretraining retain historical provenance; clinical scaling is fitted per fold.",
                  "Bootstrap intervals condition on these trained models and do not include training uncertainty.",
                  f"No final all-development refit; training seed {training_seed}, separate from CV split seed 20261020."]}
    write_json(output / "report.json", value)
    _write_auxiliary_reports(output, selection_rows, landmark_rows, complete)
    _table_csv(output / "fold_metrics.csv", fold_rows)
    _table_csv(output / "fold_summary.csv", fold_summary)
    _table_csv(output / "supplementary_last_val.csv", last_rows)
    _table_csv(output / "aggregate_metrics.csv", [{k: v for k, v in r.items() if k not in {"bootstrap", "reliability_bins"}}
        | {f"{m}_ci_{edge}": r["bootstrap"]["intervals"].get(m, {}).get(edge)
           for m in REPORT_METRICS for edge in ("lower", "upper")} for r in aggregate_metrics])
    for kind, filename in (("oof", "oof_predictions"), ("test_ensemble", "test_ensemble_predictions")):
        rows = [r for name in ranking for r in aggregates[kind].get(name, [])]
        write_json(output / f"{filename}.json", {"schema": "medium_cv_predictions_v1", "population": kind,
            "status": "withheld" if kind == "test_ensemble" and not complete else "available", "rows": rows})
        _table_csv(output / f"{filename}.csv", [dict(r, **({"fold_probabilities": json.dumps(r["fold_probabilities"])}
                   if "fold_probabilities" in r else {})) for r in rows])
    fold_test_rows = [row for name in ranking for fold in sorted(predictions["test"][name])
                     for row in predictions["test"][name][fold]]
    write_json(output / "fold_test_predictions.json", {"schema": "medium_cv_predictions_v1", "population": "fold_test",
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
    document = "<!doctype html><html lang='zh'><meta charset='utf-8'><title>Medium五折T0比较</title>" \
        "<style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse;margin:16px 0}th,td{border:1px solid #ddd;padding:6px}</style>" \
        f"<h1>{len(state['spec']['variants'])}个medium变体 × {state['spec']['cv']['n_splits']}折；训练seed {training_seed}</h1><p>训练时间点：{html.escape(value['training_landmarks'])}；checkpoint按该训练口径F1选优。主表统一报告T0；主排序：五折T0验证F1均值；测试为本种子五折逐患者T0概率均值。测试仅在整个suite全部任务完成后公布。</p>" \
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

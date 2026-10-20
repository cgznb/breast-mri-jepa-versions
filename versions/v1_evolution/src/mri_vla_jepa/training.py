"""VLA-style MRI JEPA training with fixed-T0 or patient-equal dynamic pCR selection."""
from __future__ import annotations

from pathlib import Path
import math
import random
import numpy as np
import torch

from .arm import normalize_patient_key
from .io import (autocast, load_checkpoint, restore_rng, rng_state,
                 save_checkpoint, seed_all, stable_hash, write_json)
from .metrics import classification_metrics, patient_bootstrap
from .data import RawMRIStore
from .data_pipeline import BatchLoader
from .model import RawVLAJEPA
from .phases import selected_phases
from .train_config import RawVLATrainConfig, from_dict, load_config

CHECKPOINT_SCHEMA = "responsewm_raw_vla_jepa_checkpoint_v2"
LEGACY_CHECKPOINT_SCHEMA = "responsewm_raw_vla_jepa_checkpoint_v1"


def _training_rng(device):
    state = rng_state()
    if str(device).startswith("mps"):
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_training_rng(state, device):
    restore_rng(state)
    if str(device).startswith("mps"):
        torch.mps.set_rng_state(state["mps"])


def _configure(model, cfg, phase):
    frozen = phase == "joint" and cfg.schedule == "two_stage" and cfg.freeze_encoder_after_representation
    model.encoder.requires_grad_(not frozen)
    if cfg.model.encoder_type in {"vq_prequant", "vq_encoder"}:
        model.encoder.set_pretrained_trainable(cfg.encoder_training == "finetune" and not frozen)
    if model.teacher_encoder is not None:
        model.teacher_encoder.requires_grad_(False)
    model.train()
    if frozen:
        model.encoder.eval()
    return frozen


def _make_optimizer(model, cfg):
    """Keep the raw optimizer unchanged; isolate the pretrained VQ frontend."""
    if cfg.model.encoder_type == "raw":
        parameters = [p for p in model.parameters() if p.requires_grad]
    else:
        pretrained = list(model.encoder.pretrained_parameters())
        pretrained_ids = {id(p) for p in pretrained}
        parameters = [{"params": [p for p in model.parameters()
                                  if p.requires_grad and id(p) not in pretrained_ids],
                       "lr": cfg.lr, "name": "network"}]
        trainable = [p for p in pretrained if p.requires_grad]
        if trainable:
            parameters.append({"params": trainable,
                               "lr": cfg.encoder_lr if cfg.encoder_lr is not None else cfg.lr,
                               "name": "pretrained_encoder"})
    return torch.optim.AdamW(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)


def _learning_rates(optimizer):
    return {group.get("name", "network"): group["lr"] for group in optimizer.param_groups}


def _warmup_cosine_factor(cfg, batches_per_epoch, completed_steps):
    """LR for the next update; epoch endpoints refer to the last used update."""
    warmup_steps = cfg.warmup_epochs * batches_per_epoch
    end_steps = cfg.cosine_end_epoch * batches_per_epoch
    if warmup_steps < 2:
        raise ValueError("warmup_cosine requires at least two warmup optimizer updates")
    if completed_steps < warmup_steps:
        return (cfg.warmup_start_factor + (1 - cfg.warmup_start_factor)
                * completed_steps / (warmup_steps - 1))
    minimum = cfg.cosine_min_lr / cfg.lr
    fraction = min(1., (completed_steps - (warmup_steps - 1)) / (end_steps - warmup_steps))
    return minimum + (1 - minimum) * (1 + math.cos(math.pi * fraction)) / 2


def _make_scheduler(optimizer, cfg, batches_per_epoch):
    if cfg.lr_scheduler == "warmup_cosine":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda completed: _warmup_cosine_factor(cfg, batches_per_epoch, completed))
    if cfg.lr_scheduler == "plateau":
        minimum_rates = [cfg.plateau_min_lr * group["lr"] / cfg.lr for group in optimizer.param_groups]
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=_selection_mode(cfg.selection_metric), factor=cfg.plateau_factor,
            patience=cfg.plateau_patience, min_lr=minimum_rates, threshold=cfg.plateau_threshold,
            threshold_mode="abs")
    return None


def _encoder_training_state(model, cfg):
    is_vq = cfg.model.encoder_type in {"vq_prequant", "vq_encoder"}
    pretrained_trainable = (any(p.requires_grad for p in model.encoder.pretrained_parameters())
                            if is_vq else None)
    return {"encoder_type": cfg.model.encoder_type,
            "encoder_training": cfg.encoder_training if is_vq else "scratch",
            "encoder_frozen": not any(p.requires_grad for p in model.encoder.parameters()),
            "pretrained_encoder_trainable": pretrained_trainable,
            "pretrained_encoder_frozen": not pretrained_trainable if is_vq else None}


def _checkpoint_file_identity(path):
    checkpoint = Path(path).expanduser().resolve()
    stat = checkpoint.stat()
    if not checkpoint.is_file():
        raise ValueError("Pretrained initialization must identify a checkpoint file")
    return {"path": str(checkpoint), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _score(rows, bootstrap=0, seed=0, f1_threshold=0.5):
    labelled = [row for row in rows if row["pcr"] is not None]
    if not labelled:
        raise ValueError("Direct pCR evaluation needs at least one labelled patient")
    labels = [r["pcr"] for r in labelled]
    probabilities = [r["pcr_probability"] for r in labelled]
    result = classification_metrics(labels, probabilities, f1_threshold=f1_threshold)
    result["missing_labels"] = len(rows) - len(labelled)
    if bootstrap:
        result["bootstrap"] = patient_bootstrap(labels, probabilities,
                            [r["patient_key"] for r in labelled], repetitions=bootstrap, seed=seed,
                            f1_threshold=f1_threshold)
    return result


def _source_snapshot():
    root = Path(__file__).resolve().parent
    return {name: (root / name).read_text(encoding="utf-8") for name in
            ("model.py", "model_config.py", "encoder.py", "vq_encoder.py", "flow.py", "training.py",
             "train_config.py", "cli.py", "data.py", "contracts.py", "constants.py",
             "arm.py", "io.py", "metrics.py", "data_pipeline.py", "diagnostics.py",
             "attention.py", "phases.py")}


def _patient_partitions(store):
    return {split: sorted(normalize_patient_key(store.patients[i]["patient_key"])
                          for i in store.by_split[split]) for split in ("train", "val", "test")}


def _protocol(profile):
    return "direct_pcr_fixed_t0" if profile == "t0" else "direct_pcr_all_observed"


def _selection_protocol(profile, selection_metric="nll"):
    return f"t0_direct_{selection_metric}" if profile == "t0" else f"patient_equal_prefix_{selection_metric}"


def _selection_mode(selection_metric):
    return "max" if selection_metric == "f1" else "min"


def _evaluation_profile(cfg, landmark):
    if landmark is None:
        return cfg.landmarks
    if landmark == "all_observed":
        if cfg.landmarks == "t0":
            raise ValueError("T0 checkpoint only permits landmark=0 evaluation")
        return landmark
    if type(landmark) is not int or not 0 <= landmark <= 3:
        raise ValueError("landmark must be T0..T3 or all_observed")
    if cfg.landmarks == "t0" and landmark != 0:
        raise ValueError("T0 checkpoint only permits landmark=0 evaluation")
    return landmark


def _rows(model, store, *, split, profile, batch_size=1, device="cpu"):
    indices = store.by_split[split]
    if not indices:
        raise ValueError(f"No patients in {split} split")
    tasks = []
    for i in indices:
        stages = store.allowed_landmarks(i) if profile == "all_observed" else [0 if profile == "t0" else profile]
        tasks.extend((i, stage) for stage in stages)
    model.eval()
    rows = []
    with torch.no_grad():
        for offset in range(0, len(tasks), batch_size):
            chunk = tasks[offset:offset + batch_size]
            # Future MRI and target labels never enter the prediction input.
            inp, _ = store.batch(chunk, supervised=False)
            output = model(inp.to(device))
            probabilities = output["pcr_logit"].float().sigmoid().cpu().tolist()
            diagnostics = {name: output[name].float().cpu().tolist() for name in
                           ("pcr_spatial_attention_entropy", "pcr_spatial_effective_tokens") if name in output}
            for position, ((i, stage), probability) in enumerate(zip(chunk, probabilities)):
                patient = store.patients[i]
                label = patient.get("target", {}).get("pcr")
                if label is not None and (type(label) not in {int, float} or label not in (0, 1)):
                    raise ValueError("Evaluation pCR label must be binary or missing")
                rows.append({"patient_key": patient["patient_key"], "landmark": stage,
                             "pcr_probability": probability, "pcr": label,
                             "arm_semantics": patient["arm"]["semantics"],
                             **{name: values[position] for name, values in diagnostics.items()}})
    return rows


def _patient_equal_score(rows, bootstrap=0, seed=0, f1_threshold=0.5):
    """Give each patient unit weight, split equally over their labelled prefixes."""
    groups = {}
    all_ids = {normalize_patient_key(r["patient_key"]) for r in rows}
    for row in rows:
        if row["pcr"] is not None:
            groups.setdefault(normalize_patient_key(row["patient_key"]), []).append(row)
    if not groups:
        raise ValueError("Direct pCR evaluation needs at least one labelled patient")
    patient_nll, patient_brier, patient_confusion = [], [], []
    all_labels, all_probabilities, all_weights = [], [], []
    for patient_rows in groups.values():
        labels = np.asarray([r["pcr"] for r in patient_rows], dtype=np.float64)
        probabilities = np.asarray([r["pcr_probability"] for r in patient_rows], dtype=np.float64)
        if (not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 1
                or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any()):
            raise ValueError("Invalid or inconsistent patient labels/probabilities")
        clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
        patient_nll.append(float(-(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)).mean()))
        patient_brier.append(float(((probabilities - labels) ** 2).mean()))
        metric = classification_metrics(labels, probabilities, f1_threshold=f1_threshold,
                                        sample_weight=np.full(len(labels), 1 / len(labels)))
        patient_confusion.append([metric["tp"], metric["fp"], metric["fn"]])
        all_labels.extend(labels)
        all_probabilities.extend(probabilities)
        all_weights.extend([1 / len(labels)] * len(labels))
    result = classification_metrics(all_labels, all_probabilities, f1_threshold=f1_threshold,
                                    sample_weight=all_weights)
    result.update({"n": len(groups), "patients": len(groups), "prefixes": sum(map(len, groups.values())),
              "positives": sum(patient_rows[0]["pcr"] == 1 for patient_rows in groups.values()),
              "missing_label_patients": len(all_ids) - len(groups),
              "nll": float(np.mean(patient_nll)), "brier": float(np.mean(patient_brier))})
    if bootstrap:
        rng = np.random.default_rng(seed)
        values = {"nll": np.asarray(patient_nll), "brier": np.asarray(patient_brier)}
        sampled = rng.integers(0, len(groups), size=(bootstrap, len(groups)))
        result["bootstrap"] = {
            name: {"lower": float(np.quantile(vector[sampled].mean(1), .025)),
                   "upper": float(np.quantile(vector[sampled].mean(1), .975)),
                   "valid_replicates": bootstrap} for name, vector in values.items()}
        confusion = np.asarray(patient_confusion)[sampled].sum(1)
        denominator = 2 * confusion[:, 0] + confusion[:, 1] + confusion[:, 2]
        f1 = np.divide(2 * confusion[:, 0], denominator, out=np.zeros_like(denominator), where=denominator > 0)
        result["bootstrap"]["f1"] = {"lower": float(np.quantile(f1, .025)),
            "upper": float(np.quantile(f1, .975)), "valid_replicates": bootstrap}
    return result


def _scores(rows, profile, bootstrap=0, seed=0, selection_metric="nll", f1_threshold=0.5):
    if selection_metric not in {"nll", "f1"}:
        raise ValueError("selection_metric must be nll or f1")
    per_landmark = {}
    for stage in range(4):
        selected = [r for r in rows if r["landmark"] == stage]
        labelled = sum(r["pcr"] is not None for r in selected)
        per_landmark[f"T{stage}"] = (_score(selected, bootstrap, seed, f1_threshold) if labelled else
                                      {"n": 0, "missing_labels": len(selected), "nll": None,
                                       "brier": None, "auroc": None, "auprc": None, "f1": None,
                                       "f1_threshold": f1_threshold})
    patient_equal = _patient_equal_score(rows, bootstrap, seed, f1_threshold)
    metrics = patient_equal if profile == "all_observed" else _score(rows, bootstrap, seed, f1_threshold)
    metric = (_selection_protocol(profile, selection_metric) if isinstance(profile, str) else
              (f"t0_direct_{selection_metric}" if profile == 0 else f"landmark_T{profile}_direct_{selection_metric}"))
    return {"selection_metric": metric,
            "metric_name": selection_metric, "selection_mode": _selection_mode(selection_metric),
            "f1_threshold": f1_threshold, "selection_score": metrics[selection_metric], "metrics": metrics,
            "per_landmark": per_landmark, "patient_equal_metrics": patient_equal}


def _checkpoint_config_matches(saved, cfg):
    # Older read-only checkpoints omitted newly introduced default fields.
    normalized = dict(saved)
    defaults = RawVLATrainConfig()
    normalized.setdefault("rollout_weight", defaults.rollout_weight)
    for name in ("training_unit", "max_epochs", "early_stopping_patience", "early_stopping_min_delta",
                 "lr_scheduler", "plateau_factor", "plateau_patience", "plateau_min_lr", "plateau_threshold",
                 "warmup_epochs", "warmup_start_factor", "cosine_end_epoch", "cosine_min_lr"):
        normalized.setdefault(name, getattr(defaults, name))
    for name in ("image_cache", "prefetch_batches", "loader_workers", "pin_memory", "non_blocking_transfer"):
        normalized.setdefault(name, getattr(defaults, name))
    for name in ("variance_definition", "diagnostics_every_epochs", "diagnostics_patients",
                 "diagnostics_batch_size", "diagnostics_seed"):
        normalized.setdefault(name, getattr(defaults, name))
    for name in ("encoder_checkpoint", "encoder_training", "encoder_lr", "initialize_from_checkpoint"):
        normalized.setdefault(name, getattr(defaults, name))
    for name in ("prediction_training", "selection_metric", "f1_threshold"):
        normalized.setdefault(name, getattr(defaults, name))
    model_defaults = defaults.to_dict()["model"]
    normalized["model"] = dict(normalized.get("model", {}))
    for name, value in model_defaults.items():
        normalized["model"].setdefault(name, value)
    return normalized == cfg.to_dict()


def _initialize_from_checkpoint(model, cfg, *, data_signature, normalization,
                                partitions, source_snapshot):
    """Start a new fine-tuning run from a compatible frozen-stage checkpoint."""
    state = load_checkpoint(cfg.initialize_from_checkpoint)
    if state.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Initialization requires a v2 VLA checkpoint schema")
    saved_cfg = from_dict(state["config"])
    if not _checkpoint_config_matches(state["config"], saved_cfg):
        raise ValueError("Initialization checkpoint configuration snapshot mismatch")
    if (saved_cfg.model.encoder_type != "vq_prequant" or saved_cfg.encoder_training != "frozen"
            or cfg.model.encoder_type != "vq_prequant" or cfg.encoder_training != "finetune"):
        raise ValueError("Initialization requires a frozen VQ encoder stage followed by fine-tuning")
    if saved_cfg.to_dict()["model"] != cfg.to_dict()["model"]:
        raise ValueError("Initialization model architecture differs from the frozen stage")
    if state.get("data_signature") != data_signature:
        raise ValueError("Initialization data identity differs from the frozen stage")
    if state.get("normalization") != normalization:
        raise ValueError("Initialization normalization coordinates differ from the frozen stage")
    if state.get("patient_partitions") != partitions:
        raise ValueError("Initialization population differs from the frozen stage")
    if state.get("source_snapshot") != source_snapshot:
        raise ValueError("Initialization implementation source differs from the frozen stage")
    if (state.get("protocol") != _protocol(cfg.landmarks) or state.get("landmarks") != cfg.landmarks
            or state.get("selection_protocol") != _selection_protocol(cfg.landmarks, cfg.selection_metric)):
        raise ValueError("Initialization population/selection protocol differs from the frozen stage")
    # Only initialization, learning-rate, budget and execution options may change.
    mutable = {"encoder_checkpoint", "encoder_training", "encoder_lr", "initialize_from_checkpoint",
               "lr", "lr_scheduler", "plateau_factor", "plateau_patience", "plateau_min_lr",
               "warmup_epochs", "warmup_start_factor", "cosine_end_epoch", "cosine_min_lr",
               "plateau_threshold", "max_epochs", "joint_steps", "early_stopping_patience",
               "early_stopping_min_delta", "seed", "device", "precision", "threads", "deterministic",
               "validation_every", "checkpoint_every", "validation_batch_size", "image_cache",
               "prefetch_batches", "loader_workers", "pin_memory", "non_blocking_transfer",
               "diagnostics_every_epochs", "diagnostics_patients", "diagnostics_batch_size", "diagnostics_seed"}
    original, requested = saved_cfg.to_dict(), cfg.to_dict()
    if {key: value for key, value in original.items() if key not in mutable} != {
            key: value for key, value in requested.items() if key not in mutable}:
        raise ValueError("Initialization loss/sampling configuration differs from the frozen stage")
    model.load_state_dict(state["model"], strict=True)
    return {"type": "frozen_stage_checkpoint",
            "checkpoint": _checkpoint_file_identity(cfg.initialize_from_checkpoint),
            "source_completed_steps": state["completed_steps"],
            "source_completed_epochs": (state.get("epoch_state") or {}).get("completed_epochs"),
            "source_best_validation_nll": state["best_nll"],
            "source_selection_metric": saved_cfg.selection_metric,
            "source_best_selection_score": state.get("best_selection_score", state["best_nll"]),
            "model_weights_loaded": True,
            "optimizer_restored": False, "sampler_restored": False,
            "pretrained_initialization": state.get("initialization")}


def train(store, cfg, output, *, resume=False, stop_after=None):
    cfg.validate()
    with BatchLoader(store, prefetch_batches=cfg.prefetch_batches, workers=cfg.loader_workers,
                     pin_memory=cfg.pin_memory,
                     future_supervision=bool(cfg.jepa_weight or cfg.rollout_weight or cfg.flow_weight or
                         (cfg.prediction_training == "teacher_forcing" and cfg.task_weight and
                          cfg.model.pcr_readout in {"forecast", "forecast_detach"}))) as loader:
        return _train(store, cfg, output, resume=resume, stop_after=stop_after, loader=loader)


def _train(store, cfg, output, *, resume, stop_after, loader):
    cfg.validate()
    if bool(cfg.image_cache) != (store.image_cache is not None):
        raise ValueError("Store image cache differs from the training configuration")
    if cfg.image_cache:
        configured_cache = Path(cfg.image_cache)
        if not configured_cache.is_absolute():
            configured_cache = store.path.parent / configured_cache
        if configured_cache.resolve() != store.image_cache.directory:
            raise ValueError("Store image cache directory differs from the training configuration")
    if store.image_shape != tuple(cfg.image_shape) or store.clinical_dim != cfg.model.clinical_dim:
        raise ValueError("Raw store shape/clinical dimension disagree with the model configuration")
    if store.manifest.get("synthetic", False) and not cfg.allow_synthetic:
        raise ValueError("Synthetic training must be explicitly enabled")
    if (cfg.model.encoder_type in {"vq_prequant", "vq_encoder"} and not store.manifest.get("synthetic", False)
            and not store.preprocessed_roi32):
        raise ValueError("Pretrained VQ training requires the existing preprocessed ROI32 normalization")
    if not store.by_split["train"] or not store.by_split["val"]:
        raise ValueError("Training and profile-matched validation splits are required")
    if not any(store.patients[i].get("target", {}).get("pcr") is not None for i in store.by_split["train"]):
        raise ValueError("Joint pCR training requires labelled training patients")
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError("stop_after must be a positive global optimizer-step limit")
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    last = out / "last.pt"
    if last.exists() and not resume:
        raise FileExistsError("Run already exists; use --resume or a new output directory")
    if resume and not last.exists():
        raise FileNotFoundError("Resume requires last.pt")
    seed_all(cfg.seed, threads=cfg.threads, deterministic=cfg.deterministic)
    model = RawVLAJEPA(cfg.model).to(cfg.device)
    data_signature, source_snapshot = store.signature(), _source_snapshot()
    normalization = store.normalization_state()
    partitions = _patient_partitions(store)
    initialization = None
    if not resume:
        if cfg.initialize_from_checkpoint:
            initialization = _initialize_from_checkpoint(model, cfg, data_signature=data_signature,
                               normalization=normalization, partitions=partitions, source_snapshot=source_snapshot)
        elif cfg.model.encoder_type in {"vq_prequant", "vq_encoder"}:
            if not cfg.encoder_checkpoint:
                raise ValueError("Fresh VQ training requires encoder_checkpoint or initialize_from_checkpoint")
            identity = _checkpoint_file_identity(cfg.encoder_checkpoint)
            metadata = model.initialize_pretrained_encoder(cfg.encoder_checkpoint)
            initialization = {"type": "pretrained_vq_encoder", "checkpoint": identity,
                              "model_weights_loaded": True, "encoder_metadata": metadata}
    _configure(model, cfg, cfg.phase(0))
    optimizer = _make_optimizer(model, cfg)
    batches_per_epoch = math.ceil(len(store.by_split["train"]) / cfg.batch_size)
    scheduler = _make_scheduler(optimizer, cfg, batches_per_epoch)
    initial_scheduler_rates = list(scheduler.base_lrs) if cfg.lr_scheduler == "warmup_cosine" else None
    sampler = random.Random(cfg.seed + 1)
    protocol, selection_protocol = _protocol(cfg.landmarks), _selection_protocol(cfg.landmarks, cfg.selection_metric)
    completed, best_nll, history = 0, float("inf"), []
    best_score = -float("inf") if cfg.selection_metric == "f1" else float("inf")
    best_f1 = None
    best_epoch, best_step = None, None
    epoch_mode = cfg.training_unit == "epochs"
    total_steps = batches_per_epoch * cfg.max_epochs if epoch_mode else cfg.total_steps
    epoch_history = []
    epoch_state = {"completed_epochs": 0, "indices": [], "position": 0,
                   "batches_completed": 0, "examples_seen": 0, "term_sums": {},
                   "gradient_norm_sum": 0.0, "bad_epochs": 0, "stopped_early": False}
    diagnostics_state = None
    if cfg.diagnostics_every_epochs:
        from .diagnostics import make_probe_tasks
        diagnostics_state = {"tasks": make_probe_tasks(store, cfg.landmarks,
                             patients=cfg.diagnostics_patients, seed=cfg.diagnostics_seed),
                             "previous": None, "reports": []}
    if resume:
        state = load_checkpoint(last)
        if state.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("Resume requires a v2 VLA checkpoint schema; v1 checkpoints are read-only")
        if not _checkpoint_config_matches(state["config"], cfg) or state.get("data_signature") != data_signature:
            raise ValueError("Resume config/data identity differs from the saved run")
        if state.get("source_snapshot") != source_snapshot:
            raise ValueError("Resume implementation identity differs from the saved run")
        if state.get("normalization") != normalization:
            raise ValueError("Resume normalization coordinates changed")
        if (state.get("patient_partitions") != partitions or state.get("landmarks") != cfg.landmarks
                or state.get("selection_protocol") != selection_protocol):
            raise ValueError("Resume population/selection identity differs from the saved run")
        model.load_state_dict(state["model"], strict=True)
        initialization = state.get("initialization")
        optimizer.load_state_dict(state["optimizer"])
        completed, best_nll, history = state["completed_steps"], state["best_nll"], state["history"]
        best_score = state.get("best_selection_score", best_nll)
        best_f1 = state.get("best_validation_f1")
        best_epoch, best_step = state.get("best_epoch"), state.get("best_step")
        if epoch_mode:
            epoch_state, epoch_history = state["epoch_state"], state["epoch_history"]
        if cfg.diagnostics_every_epochs:
            saved_diagnostics = state.get("diagnostics_state")
            if (not isinstance(saved_diagnostics, dict)
                    or tuple(saved_diagnostics.get("tasks", ())) != diagnostics_state["tasks"]):
                raise ValueError("Resume fixed diagnostic probe differs from the saved run")
            diagnostics_state = saved_diagnostics
        saved_scheduler = state.get("scheduler_state")
        if scheduler is not None:
            if saved_scheduler is None:
                raise ValueError(f"Resume requires the saved {cfg.lr_scheduler} scheduler state")
            scheduler.load_state_dict(saved_scheduler)
            expected_progress = completed if cfg.lr_scheduler == "warmup_cosine" else epoch_state["completed_epochs"]
            if (scheduler.get_last_lr() != [group["lr"] for group in optimizer.param_groups]
                    or scheduler.last_epoch != expected_progress):
                raise ValueError("Resume scheduler state disagrees with optimizer/epoch progress")
            if cfg.lr_scheduler == "warmup_cosine":
                factor = _warmup_cosine_factor(cfg, batches_per_epoch, completed)
                expected_rates = [rate * factor for rate in initial_scheduler_rates]
                if (scheduler.base_lrs != initial_scheduler_rates or scheduler.get_last_lr() != expected_rates
                        or scheduler._step_count != completed + 1
                        or completed != (epoch_state["completed_epochs"] * batches_per_epoch
                                         + epoch_state["batches_completed"])
                        or saved_scheduler.get("lr_lambdas") != [None] * len(initial_scheduler_rates)):
                    raise ValueError("Resume warmup_cosine state disagrees with the configured update schedule")
        elif saved_scheduler is not None:
            raise ValueError("Resume contains scheduler state but lr_scheduler is none")
        sampler.setstate(state["sampler_state"])
        _restore_training_rng(state["rng"], cfg.device)
        _configure(model, cfg, cfg.phase(completed))
    write_json(out / "config.json", cfg.to_dict())
    write_json(out / "normalization.json", normalization)
    if initialization is not None:
        write_json(out / "initialization.json", initialization)

    def snapshot():
        return {"schema": CHECKPOINT_SCHEMA, "config": cfg.to_dict(),
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "completed_steps": completed, "phase": cfg.phase(completed), "best_nll": best_nll,
                "best_selection_score": best_score, "best_validation_f1": best_f1,
                "best_epoch": best_epoch, "best_step": best_step,
                "selection_metric": cfg.selection_metric, "selection_mode": _selection_mode(cfg.selection_metric),
                "f1_threshold": cfg.f1_threshold, "prediction_training": cfg.prediction_training,
                "history": history, "rng": _training_rng(cfg.device), "sampler_state": sampler.getstate(),
                "source_snapshot": source_snapshot, "patient_partitions": partitions,
                "data_signature": data_signature, "normalization": normalization,
                "protocol": protocol, "landmarks": cfg.landmarks, "selection_protocol": selection_protocol,
                "training_unit": cfg.training_unit,
                "epoch_state": epoch_state if epoch_mode else None, "epoch_history": epoch_history,
                "diagnostics_state": diagnostics_state,
                "initialization": initialization,
                "synthetic": bool(store.manifest.get("synthetic", False))}

    def progress(status, stop_reason=None):
        write_json(out / "progress.json", {
            "status": status, "training_unit": cfg.training_unit,
            "completed_steps": completed, "total_steps": total_steps,
            "completed_epochs": epoch_state["completed_epochs"] if epoch_mode else None,
            "current_epoch": (min(epoch_state["completed_epochs"] + 1, cfg.max_epochs)
                              if epoch_mode else None),
            "max_epochs": cfg.max_epochs if epoch_mode else None,
            "batch_in_epoch": epoch_state["batches_completed"] if epoch_mode else None,
            "batches_per_epoch": batches_per_epoch if epoch_mode else None,
            "patients_seen_in_epoch": epoch_state["examples_seen"] if epoch_mode else None,
            "bad_epochs": epoch_state["bad_epochs"] if epoch_mode else None,
            "early_stopping_patience": cfg.early_stopping_patience if epoch_mode else None,
            "lr_scheduler": cfg.lr_scheduler, "current_lr": optimizer.param_groups[0]["lr"],
            "learning_rates": _learning_rates(optimizer), **_encoder_training_state(model, cfg),
            "best_validation_nll": best_nll if math.isfinite(best_nll) else None,
            "best_selection_score": best_score if math.isfinite(best_score) else None,
            "best_validation_f1": best_f1, "selection_metric": cfg.selection_metric,
            "best_epoch": best_epoch, "best_step": best_step,
            "selection_mode": _selection_mode(cfg.selection_metric), "f1_threshold": cfg.f1_threshold,
            "prediction_training": cfg.prediction_training,
            "last_epoch": epoch_history[-1] if epoch_history else None,
            "last_step": history[-1] if history else None, "stop_reason": stop_reason,
        })

    limit = min(total_steps, stop_after) if stop_after is not None else total_steps
    progress("running")
    while completed < limit and not epoch_state["stopped_early"]:
        phase = cfg.phase(completed)
        if (phase == "joint" and cfg.schedule == "two_stage" and cfg.freeze_encoder_after_representation
                and completed == cfg.representation_steps):
            # Frozen targets must use the same coordinates as the frozen online encoder.
            model.teacher_encoder.load_state_dict(model.encoder.state_dict(), strict=True)
        frozen = _configure(model, cfg, phase)
        lr_used = optimizer.param_groups[0]["lr"]
        learning_rates_used = _learning_rates(optimizer)
        optimizer.zero_grad(set_to_none=True)
        averages = {}
        for _ in range(cfg.accumulation):
            if epoch_mode:
                if not epoch_state["indices"]:
                    epoch_state["indices"] = list(store.by_split["train"])
                    sampler.shuffle(epoch_state["indices"])
                offset = epoch_state["position"]
                indices = epoch_state["indices"][offset:offset + cfg.batch_size]
                if cfg.prefetch_batches and not loader.planned:
                    # Planning uses a clone; only consumed batches advance the checkpointed sampler.
                    preview = random.Random()
                    preview.setstate(sampler.getstate())
                    planned = []
                    for start in range(offset, len(epoch_state["indices"]), cfg.batch_size):
                        chunk = epoch_state["indices"][start:start + cfg.batch_size]
                        planned.append([(i, 0 if cfg.landmarks == "t0" else preview.choice(store.allowed_landmarks(i)))
                                        for i in chunk])
                    loader.plan(planned)
            else:
                indices = [sampler.choice(store.by_split["train"]) for _ in range(cfg.batch_size)]
            tasks = [(i, 0 if cfg.landmarks == "t0" else sampler.choice(store.allowed_landmarks(i))) for i in indices]
            inp, sup = loader.fetch(tasks)
            inp, sup = (inp.to(cfg.device, non_blocking=cfg.non_blocking_transfer),
                        sup.to(cfg.device, non_blocking=cfg.non_blocking_transfer))
            with autocast(cfg.device, cfg.precision):
                loss, terms = model.compute_loss(inp, sup, task_weight=cfg.task_weight,
                    jepa_weight=cfg.jepa_weight, rollout_weight=cfg.rollout_weight, flow_weight=cfg.flow_weight,
                    reconstruction_weight=cfg.reconstruction_weight, variance_weight=cfg.variance_weight,
                    representation_only=phase == "representation", variance_definition=cfg.variance_definition,
                    prediction_training=cfg.prediction_training)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite VLA-JEPA loss at step {completed + 1}")
            (loss / cfg.accumulation).backward()
            names = list(terms)
            values = torch.stack([terms[name].detach().float() for name in names]).cpu().tolist()
            for name, value in zip(names, values):
                averages[name] = averages.get(name, 0.) + value / cfg.accumulation
        gradient_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Non-finite VLA-JEPA gradients")
        sample_update = (cfg.diagnostics_every_epochs and epoch_mode
                         and (epoch_state["completed_epochs"] + 1) % cfg.diagnostics_every_epochs == 0
                         and epoch_state["batches_completed"] == 0)
        update_parameters = ([p for p in model.parameters() if p.requires_grad and p.grad is not None]
                             if sample_update else [])
        before_update = [p.detach().clone() for p in update_parameters]
        optimizer.step()
        update_norm = (float(torch.stack([(p.detach().float() - before.float()).square().sum()
                       for p, before in zip(update_parameters, before_update, strict=True)]).sum().sqrt())
                       if sample_update else None)
        del before_update, update_parameters
        if not frozen and model.teacher_encoder is not None:
            model.update_teacher()
        completed += 1
        if cfg.lr_scheduler == "warmup_cosine":
            scheduler.step()
        entry = {"step": completed, "phase": phase, **averages, "gradient_norm": float(gradient_norm),
                 "lr": lr_used, "learning_rates": learning_rates_used,
                 **_encoder_training_state(model, cfg)}
        if sample_update:
            entry.update(optimizer_update_norm=update_norm,
                         optimizer_update_scope="first_batch_of_diagnostic_epoch")
        epoch_finished = False
        if epoch_mode:
            epoch_state["position"] += len(indices)
            epoch_state["examples_seen"] += len(indices)
            epoch_state["batches_completed"] += 1
            for name, value in averages.items():
                weight = 1 if name.endswith("_count") else len(indices)
                epoch_state["term_sums"][name] = epoch_state["term_sums"].get(name, 0.0) + value * weight
            epoch_state["gradient_norm_sum"] += float(gradient_norm) * len(indices)
            if cfg.diagnostics_every_epochs:
                epoch_state["clipped_batches"] = (epoch_state.get("clipped_batches", 0)
                                                   + int(gradient_norm > cfg.grad_clip))
                if sample_update:
                    epoch_state["sampled_optimizer_update_norm"] = update_norm
            entry.update(epoch=epoch_state["completed_epochs"] + 1,
                         batch_in_epoch=epoch_state["batches_completed"], batch_size=len(indices))
            epoch_finished = epoch_state["position"] == len(epoch_state["indices"])
        validate = (epoch_finished if epoch_mode else phase == "joint" and
                    (completed % cfg.validation_every == 0 or completed == total_steps))
        improved = False
        if validate:
            rows = _rows(model, store, split="val", profile=cfg.landmarks,
                         batch_size=cfg.validation_batch_size, device=cfg.device)
            scores = (_scores(rows, cfg.landmarks) if cfg.selection_metric == "nll" and cfg.f1_threshold == .5 else
                      _scores(rows, cfg.landmarks, selection_metric=cfg.selection_metric, f1_threshold=cfg.f1_threshold))
            entry["validation"] = scores
            if not math.isfinite(scores["selection_score"]):
                raise FloatingPointError("Non-finite profile-matched validation selection score")
            if cfg.lr_scheduler == "plateau":
                scheduler.step(scores["selection_score"])
            min_delta = cfg.early_stopping_min_delta if epoch_mode else 0.0
            value = scores["selection_score"]
            if ((value > best_score + min_delta) if cfg.selection_metric == "f1"
                    else (value < best_score - min_delta)):
                best_score, improved = value, True
                best_nll = scores["metrics"]["nll"]
                best_f1 = scores["metrics"].get("f1")
                best_step = completed
                best_epoch = epoch_state["completed_epochs"] + 1 if epoch_mode else None
        if epoch_finished:
            loader.finish_epoch()
            next_lr = optimizer.param_groups[0]["lr"]
            entry["next_lr"] = next_lr
            entry["next_learning_rates"] = _learning_rates(optimizer)
            epoch_state["completed_epochs"] += 1
            epoch_state["bad_epochs"] = 0 if improved else epoch_state["bad_epochs"] + 1
            epoch_state["stopped_early"] = epoch_state["bad_epochs"] >= cfg.early_stopping_patience
            count = epoch_state["examples_seen"]
            training_terms = {name: value if name.endswith("_count") else value / count
                              for name, value in epoch_state["term_sums"].items()}
            training_terms["gradient_norm"] = epoch_state["gradient_norm_sum"] / count
            if cfg.diagnostics_every_epochs:
                training_terms["gradient_clip_fraction"] = (epoch_state.get("clipped_batches", 0)
                                                             / epoch_state["batches_completed"])
                if "sampled_optimizer_update_norm" in epoch_state:
                    training_terms["sampled_optimizer_update_norm"] = epoch_state["sampled_optimizer_update_norm"]
            epoch_history.append({"epoch": epoch_state["completed_epochs"], "completed_steps": completed,
                                  "patients": count, "batches": epoch_state["batches_completed"],
                                  "lr": lr_used, "next_lr": next_lr,
                                  "learning_rates": learning_rates_used,
                                  "next_learning_rates": _learning_rates(optimizer),
                                  **_encoder_training_state(model, cfg),
                                  "lr_reduced": next_lr < lr_used, "lr_scheduler": cfg.lr_scheduler,
                                  "training": training_terms, "validation": scores,
                                  "best_validation_nll": best_nll, "best_selection_score": best_score,
                                  "best_validation_f1": best_f1, "selection_metric": cfg.selection_metric,
                                  "selection_mode": _selection_mode(cfg.selection_metric),
                                  "f1_threshold": cfg.f1_threshold, "prediction_training": cfg.prediction_training,
                                  "improved": improved,
                                  "bad_epochs": epoch_state["bad_epochs"]})
            if cfg.diagnostics_every_epochs and epoch_state["completed_epochs"] % cfg.diagnostics_every_epochs == 0:
                from .diagnostics import run_training_probe
                probe, previous = run_training_probe(model, store, cfg, epoch_state["completed_epochs"],
                                      diagnostics_state["tasks"], previous=diagnostics_state["previous"])
                diagnostics_state["reports"].append(probe)
                diagnostics_state["previous"] = previous
                write_json(out / "diagnostics.json", diagnostics_state["reports"])
            epoch_state.update(indices=[], position=0, batches_completed=0, examples_seen=0,
                               term_sums={}, gradient_norm_sum=0.0)
            epoch_state.pop("clipped_batches", None)
            epoch_state.pop("sampled_optimizer_update_norm", None)
            write_json(out / "epoch_history.json", epoch_history)
        history.append(entry)
        if improved:
            save_checkpoint(out / "best.pt", snapshot())
            write_json(out / "best_validation.json", {"protocol": protocol, "landmarks": cfg.landmarks,
                       "selection_protocol": selection_protocol, "split": "val", "independent_test": False,
                       **scores, "rows": rows})
        if completed % cfg.checkpoint_every == 0 or completed == limit or epoch_finished:
            save_checkpoint(last, snapshot())
            write_json(out / "history.json", history)
        progress("running")
    finished = completed == total_steps or epoch_state["stopped_early"]
    stop_reason = ("early_stopping" if epoch_state["stopped_early"] else
                   ("max_epochs" if epoch_mode else "max_steps") if finished else "stop_after")
    report = {"completed": finished, "completed_steps": completed,
              "total_steps": total_steps, "schedule": cfg.schedule, "protocol": protocol,
              "lr_scheduler": cfg.lr_scheduler, "final_lr": optimizer.param_groups[0]["lr"],
              "learning_rates": _learning_rates(optimizer), **_encoder_training_state(model, cfg),
              "initialization": initialization,
              "training_unit": cfg.training_unit, "stop_reason": stop_reason,
              "completed_epochs": epoch_state["completed_epochs"] if epoch_mode else None,
              "max_epochs": cfg.max_epochs if epoch_mode else None,
              "batches_per_epoch": batches_per_epoch if epoch_mode else None,
              "early_stopping_patience": cfg.early_stopping_patience if epoch_mode else None,
              "early_stopping_min_delta": cfg.early_stopping_min_delta if epoch_mode else None,
              "bad_epochs": epoch_state["bad_epochs"] if epoch_mode else None,
              "landmarks": cfg.landmarks, "selection_protocol": selection_protocol,
              "selection_split": "val", "independent_test": False,
              "best_validation_nll": best_nll if math.isfinite(best_nll) else None,
              "best_selection_score": best_score if math.isfinite(best_score) else None,
              "best_validation_f1": best_f1, "selection_metric": cfg.selection_metric,
              "best_epoch": best_epoch, "best_step": best_step,
              "selection_mode": _selection_mode(cfg.selection_metric), "f1_threshold": cfg.f1_threshold,
              "prediction_training": cfg.prediction_training,
              "last_checkpoint": str(last), "best_checkpoint": str(out / "best.pt") if (out / "best.pt").exists() else None,
              "synthetic": bool(store.manifest.get("synthetic", False))}
    write_json(out / "train_report.json", report)
    progress("completed" if finished else "paused", stop_reason)
    return report


def load_trained(checkpoint, device="cpu"):
    state = load_checkpoint(checkpoint)
    if state.get("schema") not in {CHECKPOINT_SCHEMA, LEGACY_CHECKPOINT_SCHEMA}:
        raise ValueError("Expected a VLA-JEPA direct-pCR checkpoint")
    cfg = from_dict(state["config"])
    if state["schema"] == LEGACY_CHECKPOINT_SCHEMA and state.get("config_digest") != stable_hash(state["config"]):
        raise ValueError("Checkpoint configuration digest mismatch")
    if state["schema"] == CHECKPOINT_SCHEMA and not _checkpoint_config_matches(state["config"], cfg):
        raise ValueError("Checkpoint configuration snapshot mismatch")
    if (state.get("protocol") != _protocol(cfg.landmarks) or state.get("landmarks") != cfg.landmarks
            or state.get("selection_protocol") != _selection_protocol(cfg.landmarks, cfg.selection_metric)):
        raise ValueError("Checkpoint pCR selection protocol mismatch")
    model = RawVLAJEPA(cfg.model).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, cfg, state


def evaluate(checkpoint, manifest, output, *, split="test", device="cpu", allow_synthetic=False,
             landmark=None, bootstrap=1000):
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val or test")
    if type(bootstrap) is not int or bootstrap < 0:
        raise ValueError("bootstrap must be a nonnegative integer")
    model, cfg, state = load_trained(checkpoint, device)
    profile = _evaluation_profile(cfg, landmark)
    torch.set_num_threads(cfg.threads)
    store = RawMRIStore(manifest, cfg.image_shape, allow_synthetic, state["normalization"],
                        require_lesion_weights=(cfg.model.lesion_world_weighting or cfg.model.lesion_pooling))
    evaluation_ids = {normalize_patient_key(store.patients[i]["patient_key"]) for i in store.by_split[split]}
    if state["schema"] == LEGACY_CHECKPOINT_SCHEMA:
        # Legacy hashes remain a read-only compatibility check, never an output field.
        evaluation_ids = {stable_hash(key) for key in evaluation_ids}
    partitions = state["patient_partitions"]
    if split in {"val", "test"} and evaluation_ids.intersection(partitions["train"]):
        raise ValueError("Evaluation patients overlap checkpoint training patients")
    if split == "test" and evaluation_ids.intersection(partitions["val"]):
        raise ValueError("Independent test patients overlap checkpoint selection patients")
    rows = _rows(model, store, split=split, profile=profile,
                 batch_size=cfg.validation_batch_size, device=device)
    independence_scope = store.manifest.get("provenance", {}).get("independence_scope", "unspecified")
    report = {"protocol": _protocol(profile) if isinstance(profile, str) else
                         ("direct_pcr_fixed_t0" if profile == 0 else "direct_pcr_observed_landmark"),
              "landmarks": profile, "checkpoint_landmarks": cfg.landmarks,
              "selection_protocol": state["selection_protocol"], "split": split,
              "independent_test": split == "test" and independence_scope == "full_pipeline",
              "downstream_holdout": split == "test", "independence_scope": independence_scope,
              "synthetic": bool(store.manifest.get("synthetic", False)),
              "clinical_validation": False if store.manifest.get("synthetic", False) else None,
              **_scores(rows, profile, bootstrap, cfg.seed, cfg.selection_metric, cfg.f1_threshold), "rows": rows}
    write_json(output, report)
    Path(output).chmod(0o600)
    return report


def predict(checkpoint, manifest, patient_key, output, *, landmark=0, device="cpu", allow_synthetic=False,
            forecast_states=False, generate=False, steps=20, seed=0):
    model, cfg, state = load_trained(checkpoint, device)
    if (forecast_states or generate) and cfg.jepa_weight == 0 and cfg.rollout_weight == 0 and cfg.flow_weight == 0:
        raise ValueError("Future prediction requires a checkpoint with a trained JEPA world head or flow loss")
    _evaluation_profile(cfg, landmark)
    if type(landmark) is not int:
        raise ValueError("Prediction landmark must be T0..T3")
    torch.set_num_threads(cfg.threads)
    store = RawMRIStore(manifest, cfg.image_shape, allow_synthetic, state["normalization"],
                        require_lesion_weights=(cfg.model.lesion_world_weighting or cfg.model.lesion_pooling))
    normalized = normalize_patient_key(patient_key)
    matches = [i for i, p in enumerate(store.patients) if normalize_patient_key(p["patient_key"]) == normalized]
    if len(matches) != 1:
        raise ValueError("patient_key must identify exactly one patient")
    inp, _ = store.batch([(matches[0], landmark)], supervised=False)
    inp = inp.to(device)
    with torch.no_grad():
        features = model(inp)
        probability = float(features["pcr_logit"].float().sigmoid()[0])
    report = {"patient_key": store.patients[matches[0]]["patient_key"], "landmark": landmark,
              "pcr_probability": probability, "protocol": "direct_pcr", "checkpoint_landmarks": cfg.landmarks,
              "phase_order": list(selected_phases(cfg.model)),
              "observed_stages": inp.observed_mask[0].nonzero().flatten().cpu().tolist(),
              "requested_stages": inp.query_mask[0].nonzero().flatten().cpu().tolist(),
              "arm_semantics": store.patients[matches[0]]["arm"]["semantics"],
              "arm_visible": bool(inp.arm_mask[0]), "synthetic": bool(store.manifest.get("synthetic", False))}
    if forecast_states:
        with torch.no_grad():
            forecast = model.forecast(inp)
        state_path = Path(output).with_suffix(".states.npz")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(state_path, state_prediction=forecast["state_prediction"].cpu().numpy(),
                            query_mask=forecast["query_mask"].cpu().numpy(),
                            phase_order=np.asarray(report["phase_order"]))
        report["forecast_states"] = str(state_path.resolve())
        report["forecast_shape"] = list(forecast["state_prediction"].shape)
    if generate:
        if not cfg.model.enable_flow:
            raise ValueError("MRI generation requires a checkpoint trained with enable_flow=true")
        with torch.no_grad():
            generated = model.generate(inp, steps=steps, seed=seed)
        image_path = Path(output).with_suffix(".mri.npz")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(image_path, future_images=generated["future_images"].cpu().numpy(),
                            query_mask=generated["query_mask"].cpu().numpy(),
                            dynamics_features=features["dynamics_features"].cpu().numpy(),
                            phase_order=np.asarray(report["phase_order"]),
                            normalization=store.normalization_state()["intensity"])
        report["generated_mri"] = str(image_path.resolve())
        report["generated_shape"] = list(generated["future_images"].shape)
    write_json(output, report)
    Path(output).chmod(0o600)
    return report

"""Dedicated I-SPY2 longitudinal training runtime."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import random
import re
import secrets
import stat
from collections.abc import Mapping
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from torch.utils.data import DataLoader, Sampler
from torch.utils.checkpoint import checkpoint

from .ispy2_conditioning import (
    MEDGEMMA_MODEL_NAME,
    ISPY2Conditioner,
    MedGemmaTextEncoder,
)
from .ispy2_dataset import ISPY2LongitudinalDataset, collate_ispy2_longitudinal
from .ispy2_registered_dataset import (
    RegisteredStrictADataset,
    collate_registered_strict_a,
    is_registered_strict_a_bundle,
)
from .ispy2_preprocessing import TARGET_SPACING_XYZ
from .utils_fm import build_model, sample_with_solver
from scripts.build_ispy2_longitudinal_bundle import verify_bundle as verify_legacy_bundle
from scripts.build_ispy2_registered_strict_a_bundle import (
    verify_bundle as verify_registered_bundle,
)


# Preserve the legacy module-level seam used by existing callers and tests.
verify_bundle = verify_legacy_bundle


CHECKPOINT_SCHEMA = "motfm_ispy2_longitudinal_checkpoint_v2"
CHECKPOINT_METADATA_KEY = "ispy2_longitudinal_metadata"
CHECKPOINT_RNG_KEY = "ispy2_rng_state"
CHECKPOINT_BINDING_KEY = "ispy2_checkpoint_binding"
MEDGEMMA_REVISION = "290cda5eeccbee130f987c4ad74a59ae6f196408"
VALIDATION_ROI_SCHEMA = "motfm_ispy2_validation_roi_v1"


def prepare_validation_values(
    generated: torch.Tensor, *, registered: bool
) -> tuple[torch.Tensor, bool]:
    values = generated.detach().to(device="cpu", dtype=torch.float32)
    if registered:
        return values, False
    return values.clamp(0.0, 1.0), True


def resolve_longitudinal_dataset_backend(bundle_path: str | Path):
    if is_registered_strict_a_bundle(bundle_path):
        return RegisteredStrictADataset, collate_registered_strict_a
    return ISPY2LongitudinalDataset, collate_ispy2_longitudinal


def verify_training_bundle(bundle_path: str | Path) -> dict[str, Any]:
    return (
        verify_registered_bundle(bundle_path)
        if is_registered_strict_a_bundle(bundle_path)
        else verify_bundle(bundle_path)
    )


class CheckpointCompatibilityError(RuntimeError):
    """Raised when a checkpoint cannot safely resume this training contract."""


class EpochShuffleSampler(Sampler[int]):
    """Deterministic train permutation addressed only by seed and epoch."""

    def __init__(self, data_source: Any, *, seed: int) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError("sampler seed must be a nonnegative integer")
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("sampler epoch must be a nonnegative integer")
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed((self.seed + self.epoch) % (2**63 - 1))
        return iter(torch.randperm(len(self), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.data_source)


class EpochSeededDataLoader(DataLoader):
    """Reset the worker generator from the sampler epoch before every iterator."""

    def __init__(self, *args: Any, worker_seed: int, **kwargs: Any) -> None:
        if type(worker_seed) is not int or worker_seed < 0:
            raise ValueError("worker seed must be a nonnegative integer")
        self.worker_seed = worker_seed
        super().__init__(*args, **kwargs)
        if self.generator is None:
            raise ValueError("epoch-seeded DataLoader requires a worker generator")

    def __iter__(self):
        epoch = getattr(self.sampler, "epoch", 0)
        if type(epoch) is not int or epoch < 0:
            raise RuntimeError("DataLoader sampler epoch is invalid")
        self.generator.manual_seed((self.worker_seed + epoch) % (2**63 - 1))
        return super().__iter__()


def immutable_commit(value: Any) -> str:
    """Return a normalized immutable Git commit or reject symbolic revisions."""

    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise ValueError("model revision must be an immutable 40-character commit")
    return value.lower()


def official_medgemma_model_id(value: Any) -> str:
    if value != MEDGEMMA_MODEL_NAME:
        raise ValueError("model id must be the official MedGemma model")
    return MEDGEMMA_MODEL_NAME


def official_medgemma_revision(value: Any) -> str:
    revision = immutable_commit(value)
    if revision != MEDGEMMA_REVISION:
        raise ValueError("model revision must be the fixed official commit")
    return MEDGEMMA_REVISION


def resolve_frozen_base_identity(
    conditioner: Any, *, model_id: Any, configured_revision: Any
) -> dict[str, str]:
    """Bind configured model identity to the commit exposed by the loaded base."""

    if model_id != MEDGEMMA_MODEL_NAME:
        raise CheckpointCompatibilityError("frozen base identity is invalid")
    try:
        revision = official_medgemma_revision(configured_revision)
    except ValueError:
        raise CheckpointCompatibilityError("frozen base identity is invalid") from None
    text_encoder = getattr(conditioner, "text_encoder", None)
    model = getattr(text_encoder, "model", None)
    candidates = [getattr(text_encoder, "resolved_model_commit", None)]
    candidates.append(getattr(getattr(model, "config", None), "_commit_hash", None))
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        try:
            base_model = get_base_model()
        except Exception:
            raise CheckpointCompatibilityError("frozen base identity is unavailable") from None
        candidates.append(
            getattr(getattr(base_model, "config", None), "_commit_hash", None)
        )
    nested_base = getattr(getattr(model, "base_model", None), "model", None)
    candidates.append(
        getattr(getattr(nested_base, "config", None), "_commit_hash", None)
    )
    tokenizer_metadata = getattr(getattr(text_encoder, "tokenizer", None), "init_kwargs", None)
    if isinstance(tokenizer_metadata, Mapping):
        candidates.append(tokenizer_metadata.get("_commit_hash"))

    resolved_candidates: set[str] = set()
    try:
        for candidate in candidates:
            if candidate is not None:
                resolved_candidates.add(immutable_commit(candidate))
    except ValueError:
        raise CheckpointCompatibilityError("frozen base identity is invalid") from None
    if len(resolved_candidates) != 1:
        raise CheckpointCompatibilityError("frozen base identity is unavailable or conflicting")
    resolved_commit = next(iter(resolved_candidates))
    if resolved_commit != revision:
        raise CheckpointCompatibilityError("frozen base identity does not match configuration")
    return {
        "id": model_id,
        "revision": revision,
        "resolved_commit": resolved_commit,
    }


def _finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    normalized = float(value)
    if not math.isfinite(normalized) or (positive and normalized <= 0):
        raise ValueError(f"{name} must be finite")
    return normalized


def _validation_spacing_xyz(config: Mapping[str, Any]) -> tuple[float, float, float]:
    data_args = config.get("data_args")
    if not isinstance(data_args, Mapping):
        raise ValueError("validation sample spacing is invalid")
    spacing = data_args.get("target_spacing_xyz")
    if not isinstance(spacing, (list, tuple)) or len(spacing) != 3:
        raise ValueError("validation sample spacing is invalid")
    try:
        normalized = tuple(
            _finite_number(value, "validation sample spacing", positive=True)
            for value in spacing
        )
    except ValueError:
        raise ValueError("validation sample spacing is invalid") from None
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _resolved_optimizer_settings(train: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
    model_lr = _finite_number(train.get("model_lr", 1e-4), "model_lr", positive=True)
    text_lr = _finite_number(train.get("text_lr", 2e-5), "text_lr", positive=True)
    weight_decay = _finite_number(train.get("weight_decay", 0.01), "weight_decay")
    if weight_decay < 0:
        raise ValueError("weight_decay must be nonnegative and finite")
    betas = train.get("betas", (0.9, 0.999))
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("AdamW betas must contain two values")
    normalized_betas = [
        _finite_number(beta, "AdamW beta") for beta in betas
    ]
    if any(beta < 0 or beta >= 1 for beta in normalized_betas):
        raise ValueError("AdamW betas must be in [0, 1)")
    eps = _finite_number(train.get("eps", 1e-8), "AdamW eps", positive=True)
    paged_switch = train.get("use_paged_adamw_8bit", False)
    if type(paged_switch) is not bool:
        raise ValueError("use_paged_adamw_8bit must be boolean")
    optimizer_name = str(train.get("optimizer", "adamw")).lower()
    if paged_switch or optimizer_name in {"paged_adamw_8bit", "pagedadamw8bit"}:
        optimizer_type = "bitsandbytes.optim.PagedAdamW8bit"
        paged = True
    elif optimizer_name == "adamw":
        optimizer_type = "torch.optim.AdamW"
        paged = False
    else:
        raise ValueError("optimizer must be adamw or paged_adamw_8bit")
    warmup_fraction = _finite_number(
        train.get("warmup_fraction", 0.05), "warmup_fraction", positive=True
    )
    if warmup_fraction >= 1:
        raise ValueError("warmup_fraction must be between zero and one")
    return (
        {
            "type": optimizer_type,
            "paged": paged,
            "model_lr": model_lr,
            "text_lr": text_lr,
            "weight_decay": weight_decay,
            "betas": normalized_betas,
            "eps": eps,
        },
        warmup_fraction,
    )


def resolve_optimization_contract(
    config: Mapping[str, Any], *, train_batches_per_epoch: int
) -> dict[str, Any]:
    """Resolve every setting that controls optimizer and future LR behavior."""

    train = config.get("train_args")
    if not isinstance(train, Mapping):
        raise ValueError("training config requires train_args")
    integer_fields = {
        "batch_size": train.get("batch_size", 1),
        "gradient_accumulation_steps": train.get("gradient_accumulation_steps", 1),
        "max_epochs": train.get("max_epochs", 1),
    }
    if any(type(value) is not int or value <= 0 for value in integer_fields.values()):
        raise ValueError("optimization integer settings must be positive")
    precision = train.get("precision", "bf16-mixed")
    accelerator = train.get("accelerator", "gpu")
    devices = train.get("devices", 1)
    gradient_clip_algorithm = train.get("gradient_clip_algorithm", "norm")
    drop_last = train.get("drop_last", False)
    deterministic = train.get("deterministic", True)
    benchmark = train.get("benchmark", False)
    gradient_clip_val = _finite_number(
        train.get("gradient_clip_val", 1.0), "gradient_clip_val", positive=True
    )
    if (
        not isinstance(precision, str)
        or not precision
        or not isinstance(accelerator, str)
        or not accelerator
        or type(devices) is not int
        or devices <= 0
        or not isinstance(gradient_clip_algorithm, str)
        or not gradient_clip_algorithm
        or type(drop_last) is not bool
        or type(deterministic) is not bool
        or type(benchmark) is not bool
    ):
        raise ValueError("training execution settings are invalid")
    if type(train_batches_per_epoch) is not int or train_batches_per_epoch <= 0:
        raise ValueError("train batches per epoch must be a positive integer")
    seed = train.get("seed", 2026)
    if type(seed) is not int or seed < 0:
        raise ValueError("sampler seed must be a nonnegative integer")
    num_workers = train.get("num_workers", 0)
    prefetch_factor = train.get("prefetch_factor", 2)
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if type(prefetch_factor) is not int or prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be a positive integer")
    optimizer, warmup_fraction = _resolved_optimizer_settings(train)
    total_steps = (
        math.ceil(
            train_batches_per_epoch
            / integer_fields["gradient_accumulation_steps"]
        )
        * integer_fields["max_epochs"]
    )
    warmup_steps = max(1, int(total_steps * warmup_fraction))
    return {
        "optimizer": optimizer,
        "scheduler": {
            "type": "warmup_cosine_lambda",
            "warmup_fraction": warmup_fraction,
            "warmup_steps": warmup_steps,
            "total_optimizer_steps": total_steps,
        },
        **integer_fields,
        "train_batches_per_epoch": train_batches_per_epoch,
        "precision": precision,
        "accelerator": accelerator,
        "devices": devices,
        "gradient_clip_val": gradient_clip_val,
        "gradient_clip_algorithm": gradient_clip_algorithm,
        "drop_last": drop_last,
        "deterministic": deterministic,
        "benchmark": benchmark,
        "sampler": {"type": "epoch_shuffle_v1", "seed": seed},
        "dataloader": {
            "num_workers": num_workers,
            "persistent_workers": False,
            "prefetch_factor": prefetch_factor if num_workers > 0 else None,
            "worker_seed": {"type": "epoch_seed_v1", "seed": seed},
        },
    }


def _validate_optimization_contract(contract: Any) -> None:
    if not isinstance(contract, Mapping) or set(contract) != {
        "optimizer",
        "scheduler",
        "batch_size",
        "gradient_accumulation_steps",
        "max_epochs",
        "train_batches_per_epoch",
        "precision",
        "accelerator",
        "devices",
        "gradient_clip_val",
        "gradient_clip_algorithm",
        "drop_last",
        "deterministic",
        "benchmark",
        "sampler",
        "dataloader",
    }:
        raise CheckpointCompatibilityError("checkpoint optimization metadata is invalid")
    optimizer = contract["optimizer"]
    if not isinstance(optimizer, Mapping) or set(optimizer) != {
        "type",
        "paged",
        "model_lr",
        "text_lr",
        "weight_decay",
        "betas",
        "eps",
    }:
        raise CheckpointCompatibilityError("checkpoint optimizer metadata is invalid")
    expected_type = (
        "bitsandbytes.optim.PagedAdamW8bit"
        if optimizer.get("paged") is True
        else "torch.optim.AdamW"
    )
    if type(optimizer.get("paged")) is not bool or optimizer.get("type") != expected_type:
        raise CheckpointCompatibilityError("checkpoint optimizer metadata is invalid")
    scheduler = contract["scheduler"]
    if not isinstance(scheduler, Mapping) or set(scheduler) != {
        "type",
        "warmup_fraction",
        "warmup_steps",
        "total_optimizer_steps",
    }:
        raise CheckpointCompatibilityError("checkpoint scheduler metadata is invalid")
    try:
        _resolved_optimizer_settings(
            {
                "optimizer": "paged_adamw_8bit" if optimizer["paged"] else "adamw",
                "use_paged_adamw_8bit": optimizer["paged"],
                "model_lr": optimizer["model_lr"],
                "text_lr": optimizer["text_lr"],
                "weight_decay": optimizer["weight_decay"],
                "betas": optimizer["betas"],
                "eps": optimizer["eps"],
                "warmup_fraction": scheduler["warmup_fraction"],
            }
        )
    except (TypeError, ValueError):
        raise CheckpointCompatibilityError("checkpoint optimizer metadata is invalid") from None
    integers = [
        contract["batch_size"],
        contract["gradient_accumulation_steps"],
        contract["max_epochs"],
        contract["train_batches_per_epoch"],
        scheduler["warmup_steps"],
        scheduler["total_optimizer_steps"],
    ]
    if any(type(value) is not int or value <= 0 for value in integers):
        raise CheckpointCompatibilityError("checkpoint optimization metadata is invalid")
    if (
        contract["precision"] != "bf16-mixed"
        or contract["accelerator"] != "gpu"
        or type(contract["devices"]) is not int
        or contract["devices"] != 1
        or type(contract["gradient_clip_val"]) is not float
        or contract["gradient_clip_val"] != 1.0
        or contract["gradient_clip_algorithm"] != "norm"
        or contract["drop_last"] is not False
        or contract["deterministic"] is not True
        or contract["benchmark"] is not False
    ):
        raise CheckpointCompatibilityError("checkpoint execution metadata is invalid")
    try:
        warmup_fraction = _finite_number(
            scheduler["warmup_fraction"], "warmup_fraction", positive=True
        )
    except ValueError:
        raise CheckpointCompatibilityError("checkpoint scheduler metadata is invalid") from None
    expected_total = (
        math.ceil(
            contract["train_batches_per_epoch"]
            / contract["gradient_accumulation_steps"]
        )
        * contract["max_epochs"]
    )
    if (
        scheduler["type"] != "warmup_cosine_lambda"
        or warmup_fraction >= 1
        or scheduler["total_optimizer_steps"] != expected_total
        or scheduler["warmup_steps"]
        != max(1, int(expected_total * warmup_fraction))
    ):
        raise CheckpointCompatibilityError("checkpoint scheduler metadata is invalid")
    sampler = contract["sampler"]
    if (
        not isinstance(sampler, Mapping)
        or set(sampler) != {"type", "seed"}
        or sampler["type"] != "epoch_shuffle_v1"
        or type(sampler["seed"]) is not int
        or sampler["seed"] < 0
    ):
        raise CheckpointCompatibilityError("checkpoint sampler metadata is invalid")
    dataloader = contract["dataloader"]
    if not isinstance(dataloader, Mapping) or set(dataloader) != {
        "num_workers",
        "persistent_workers",
        "prefetch_factor",
        "worker_seed",
    }:
        raise CheckpointCompatibilityError("checkpoint DataLoader metadata is invalid")
    num_workers = dataloader["num_workers"]
    prefetch_factor = dataloader["prefetch_factor"]
    worker_seed = dataloader["worker_seed"]
    if (
        type(num_workers) is not int
        or num_workers < 0
        or dataloader["persistent_workers"] is not False
        or (
            num_workers == 0
            and prefetch_factor is not None
        )
        or (
            num_workers > 0
            and (type(prefetch_factor) is not int or prefetch_factor <= 0)
        )
        or not isinstance(worker_seed, Mapping)
        or set(worker_seed) != {"type", "seed"}
        or worker_seed["type"] != "epoch_seed_v1"
        or worker_seed["seed"] != sampler["seed"]
    ):
        raise CheckpointCompatibilityError("checkpoint DataLoader metadata is invalid")


def _rng_device(device: torch.device | str | int) -> torch.device:
    normalized = torch.device("cuda", device) if type(device) is int else torch.device(device)
    if normalized.type not in {"cpu", "cuda"}:
        raise CheckpointCompatibilityError("training RNG device is unsupported")
    return normalized


def _capture_rng_state(device: torch.device | str | int) -> dict[str, Any]:
    logical_device = _rng_device(device)
    try:
        numpy_state = np.random.get_state()
        return {
            "schema_version": "motfm_ispy2_rng_state_v2",
            "python": random.getstate(),
            "numpy": {
                "bit_generator": numpy_state[0],
                "keys": torch.from_numpy(numpy_state[1].copy()).clone(),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": (
                torch.cuda.get_rng_state(logical_device).clone()
                if logical_device.type == "cuda"
                else None
            ),
        }
    except Exception:
        raise CheckpointCompatibilityError("checkpoint RNG state capture failed") from None


def _restore_rng_state(payload: Any, device: torch.device | str | int) -> None:
    logical_device = _rng_device(device)
    expected = {"schema_version", "python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise CheckpointCompatibilityError("checkpoint RNG state is missing or invalid")
    if payload["schema_version"] != "motfm_ispy2_rng_state_v2":
        raise CheckpointCompatibilityError("checkpoint RNG state schema is unsupported")
    python_state = payload["python"]
    numpy_payload = payload["numpy"]
    torch_cpu = payload["torch_cpu"]
    torch_cuda = payload["torch_cuda"]
    if not isinstance(numpy_payload, Mapping) or set(numpy_payload) != {
        "bit_generator",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise CheckpointCompatibilityError("checkpoint RNG state is missing or invalid")
    numpy_keys = numpy_payload["keys"]
    cached_gaussian = numpy_payload["cached_gaussian"]
    if (
        numpy_payload["bit_generator"] != "MT19937"
        or not isinstance(numpy_keys, torch.Tensor)
        or numpy_keys.device.type != "cpu"
        or numpy_keys.dtype != torch.uint32
        or numpy_keys.ndim != 1
        or numpy_keys.numel() != 624
        or type(numpy_payload["position"]) is not int
        or not 0 <= numpy_payload["position"] <= 624
        or type(numpy_payload["has_gauss"]) is not int
        or numpy_payload["has_gauss"] not in {0, 1}
        or isinstance(cached_gaussian, bool)
        or not isinstance(cached_gaussian, (int, float))
        or not math.isfinite(cached_gaussian)
    ):
        raise CheckpointCompatibilityError("checkpoint RNG state is missing or invalid")
    numpy_state = (
        "MT19937",
        numpy_keys.numpy().copy(),
        numpy_payload["position"],
        numpy_payload["has_gauss"],
        float(cached_gaussian),
    )
    if (
        not isinstance(torch_cpu, torch.Tensor)
        or torch_cpu.dtype != torch.uint8
        or torch_cpu.ndim != 1
        or torch_cpu.numel() == 0
    ):
        raise CheckpointCompatibilityError("checkpoint RNG state is missing or invalid")
    if logical_device.type == "cpu":
        if torch_cuda is not None:
            raise CheckpointCompatibilityError("checkpoint RNG state is missing or invalid")
    elif (
        not isinstance(torch_cuda, torch.Tensor)
        or torch_cuda.dtype != torch.uint8
        or torch_cuda.ndim != 1
        or torch_cuda.numel() == 0
    ):
        raise CheckpointCompatibilityError("checkpoint RNG state is missing or invalid")
    try:
        random.Random().setstate(python_state)
        np.random.RandomState().set_state(numpy_state)
        torch.Generator().set_state(torch_cpu)
    except Exception:
        raise CheckpointCompatibilityError("checkpoint RNG state is missing or invalid") from None
    try:
        if logical_device.type == "cuda":
            torch.cuda.set_rng_state(torch_cuda, logical_device)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_cpu)
    except Exception:
        raise CheckpointCompatibilityError("checkpoint RNG state restore failed") from None


def _validate_checkpoint_metadata(metadata: Any) -> None:
    required = {
        "schema_version",
        "model",
        "source_channels",
        "output_shape_zyx",
        "target_spacing_xyz",
        "contracts",
        "architecture",
        "conditioning",
        "ftv_log1p",
        "optimization",
    }
    if not isinstance(metadata, Mapping) or set(metadata) != required:
        raise CheckpointCompatibilityError("checkpoint metadata fields are invalid")
    if metadata["schema_version"] != CHECKPOINT_SCHEMA:
        raise CheckpointCompatibilityError("checkpoint metadata schema is unsupported")
    model = metadata["model"]
    if (
        not isinstance(model, Mapping)
        or set(model) != {"id", "revision", "resolved_commit"}
        or any(not isinstance(model[key], str) or not model[key] for key in model)
        or model["id"] != MEDGEMMA_MODEL_NAME
    ):
        raise CheckpointCompatibilityError("checkpoint model metadata is invalid")
    try:
        revision = official_medgemma_revision(model["revision"])
        resolved_commit = official_medgemma_revision(model["resolved_commit"])
    except ValueError:
        raise CheckpointCompatibilityError("checkpoint model metadata is invalid") from None
    if revision != resolved_commit:
        raise CheckpointCompatibilityError("checkpoint model metadata is invalid")
    if metadata["source_channels"] != ["dce0", "ser"]:
        raise CheckpointCompatibilityError("checkpoint source channel metadata is invalid")
    shape = metadata["output_shape_zyx"]
    if shape not in ([128, 128, 128], [96, 256, 256]):
        raise CheckpointCompatibilityError("checkpoint output shape metadata is invalid")
    spacing = metadata["target_spacing_xyz"]
    if (
        not isinstance(spacing, list)
        or len(spacing) != 3
        or any(
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in spacing
        )
    ):
        raise CheckpointCompatibilityError("checkpoint spacing metadata is invalid")
    contracts = metadata["contracts"]
    legacy_contracts = {"bundle_sha256", "preprocess_sha256", "qc_sha256"}
    expected_contracts = (
        legacy_contracts | {"normalization_sha256"}
        if shape == [96, 256, 256]
        else legacy_contracts
    )
    if not isinstance(contracts, Mapping) or set(contracts) != expected_contracts:
        raise CheckpointCompatibilityError("checkpoint contract metadata is invalid")
    if any(
        not isinstance(value, str) or len(value) != 64
        for value in contracts.values()
    ):
        raise CheckpointCompatibilityError("checkpoint contract hashes are invalid")
    if not isinstance(metadata["architecture"], Mapping) or not metadata["architecture"]:
        raise CheckpointCompatibilityError("checkpoint architecture metadata is invalid")
    if not isinstance(metadata["conditioning"], Mapping) or not metadata["conditioning"]:
        raise CheckpointCompatibilityError("checkpoint conditioning metadata is invalid")
    ftv = metadata["ftv_log1p"]
    if not isinstance(ftv, Mapping) or set(ftv) != {"mean", "std"}:
        raise CheckpointCompatibilityError("checkpoint FTV metadata is invalid")
    try:
        mean, std = float(ftv["mean"]), float(ftv["std"])
    except (TypeError, ValueError, OverflowError):
        raise CheckpointCompatibilityError("checkpoint FTV metadata is invalid") from None
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise CheckpointCompatibilityError("checkpoint FTV metadata is invalid")
    _validate_optimization_contract(metadata["optimization"])


def _checkpoint_binding(
    metadata: Mapping[str, Any], *, epoch: Any, global_step: Any, nonce: Any
) -> dict[str, Any]:
    if type(epoch) is not int or epoch < 0 or type(global_step) is not int or global_step < 0:
        raise CheckpointCompatibilityError("checkpoint binding fields are invalid")
    if not isinstance(nonce, str) or re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
        raise CheckpointCompatibilityError("checkpoint binding fields are invalid")
    try:
        canonical_metadata = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        raise CheckpointCompatibilityError("checkpoint binding metadata is invalid") from None
    metadata_sha256 = hashlib.sha256(canonical_metadata).hexdigest()
    identity_fields = {
        "schema_version": "motfm_ispy2_checkpoint_binding_v1",
        "metadata_sha256": metadata_sha256,
        "epoch": epoch,
        "global_step": global_step,
        "nonce": nonce,
    }
    binding_id = hashlib.sha256(
        json.dumps(
            identity_fields, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {**identity_fields, "binding_id": binding_id}


def build_checkpoint_metadata(
    config: Mapping[str, Any],
    datamodule: "ISPY2LongitudinalDataModule",
    *,
    conditioner: Any,
) -> dict[str, Any]:
    """Build the exact resume-compatibility contract from verified training inputs."""

    bundle_path = datamodule._bundle_json(config["data_args"]["bundle_path"]).resolve()
    bundle = verify_training_bundle(bundle_path)
    conditioning = config.get("conditioning_args", {})
    model_id = official_medgemma_model_id(
        conditioning.get("model_id", MEDGEMMA_MODEL_NAME)
    )
    model_identity = resolve_frozen_base_identity(
        conditioner,
        model_id=model_id,
        configured_revision=conditioning.get(
            "model_revision", conditioning.get("revision", "")
        ),
    )
    contracts = bundle.get("source_contracts", {})
    qc_hash = str(
        contracts.get("qc_visits_contract_sha256")
        or contracts.get("strict_a_rules_sha256")
        or ""
    )
    try:
        train_batches_per_epoch = len(datamodule.train_dataloader())
    except (TypeError, ValueError):
        raise CheckpointCompatibilityError("training batch count is unavailable") from None
    optimization = resolve_optimization_contract(
        config, train_batches_per_epoch=train_batches_per_epoch
    )
    contract_hashes = {
        "bundle_sha256": str(bundle["bundle_contract_sha256"]),
        "preprocess_sha256": str(bundle["artifacts"]["preprocess"]["sha256"]),
        "qc_sha256": qc_hash,
    }
    if is_registered_strict_a_bundle(bundle_path):
        contract_hashes["normalization_sha256"] = str(
            bundle["artifacts"]["normalization"]["sha256"]
        )
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA,
        "model": model_identity,
        "source_channels": ["dce0", "ser"],
        "output_shape_zyx": list(config["data_args"]["output_shape_zyx"]),
        "target_spacing_xyz": list(
            config["data_args"].get("target_spacing_xyz", TARGET_SPACING_XYZ)
        ),
        "contracts": contract_hashes,
        "architecture": deepcopy(dict(config["model_args"])),
        "conditioning": deepcopy(dict(conditioning)),
        "ftv_log1p": {
            "mean": float(datamodule.ftv_log1p_mean),
            "std": float(datamodule.ftv_log1p_std),
        },
        "optimization": optimization,
    }
    _validate_checkpoint_metadata(metadata)
    return metadata


class ISPY2LongitudinalDataModule(pl.LightningDataModule):
    """Verified longitudinal train/validation data boundary."""


    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        dataset_factory=ISPY2LongitudinalDataset,
    ) -> None:
        super().__init__()
        self.config = config
        self.dataset_factory = dataset_factory
        self.collate_fn = collate_ispy2_longitudinal
        self.train_dataset = None
        self.val_dataset = None
        self.ftv_log1p_mean = float("nan")
        self.ftv_log1p_std = float("nan")

    @staticmethod
    def _bundle_json(value: str | Path) -> Path:
        path = Path(value)
        return path / "bundle.json" if path.is_dir() else path

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit", "validate"):
            return
        if self.train_dataset is not None and self.val_dataset is not None:
            return
        data = self.config["data_args"]
        common = {
            "cache_dir": data["cache_dir"],
            "output_shape_zyx": data.get("output_shape_zyx"),
        }
        if data.get("target_spacing_xyz") is not None:
            common["target_spacing_xyz"] = data["target_spacing_xyz"]
        bundle = self._bundle_json(data["bundle_path"])
        dataset_factory = self.dataset_factory
        if dataset_factory is ISPY2LongitudinalDataset:
            dataset_factory, self.collate_fn = resolve_longitudinal_dataset_backend(bundle)
        self.train_dataset = dataset_factory(
            bundle, fold=data.get("split_train", "train"), **common
        )
        self.val_dataset = dataset_factory(
            bundle, fold=data.get("split_val", "val"), **common
        )
        values = torch.tensor(
            [
                float(row["source_ftv_volume_cc"])
                for row in self.train_dataset._transitions
            ],
            dtype=torch.float64,
        )
        if values.numel() == 0 or not torch.isfinite(values).all() or torch.any(values <= 0):
            raise ValueError("training source FTV values must be positive and finite")
        logged = torch.log1p(values)
        mean = logged.mean().item()
        std = logged.std(unbiased=False).item()
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError("training source FTV log1p statistics require a positive finite std")
        self.ftv_log1p_mean = mean
        self.ftv_log1p_std = std

    def initialize_conditioner(self, conditioner: torch.nn.Module) -> None:
        if not math.isfinite(self.ftv_log1p_mean) or self.ftv_log1p_std <= 0:
            raise RuntimeError("data module must be set up before initializing the conditioner")
        conditioner.source_ftv_encoder.set_statistics(
            self.ftv_log1p_mean, self.ftv_log1p_std
        )

    def _loader(self, dataset: Any, *, shuffle: bool) -> DataLoader:
        train = self.config["train_args"]
        workers = int(train.get("num_workers", 0))
        seed = int(train.get("seed", 2026))
        prefetch_factor = int(train.get("prefetch_factor", 2))
        sampler = EpochShuffleSampler(dataset, seed=seed) if shuffle else None
        worker_generator = torch.Generator()
        return EpochSeededDataLoader(
            dataset,
            worker_seed=seed + (0 if shuffle else 1),
            batch_size=int(train.get("batch_size", 1)),
            shuffle=False,
            sampler=sampler,
            num_workers=workers,
            collate_fn=self.collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
            generator=worker_generator,
            drop_last=bool(train.get("drop_last", False)),
            prefetch_factor=prefetch_factor if workers > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        loader = self._loader(self.train_dataset, shuffle=True)
        trainer = getattr(self, "trainer", None)
        current_epoch = int(getattr(trainer, "current_epoch", 0))
        fit_loop = getattr(trainer, "fit_loop", None)
        epoch_progress = getattr(fit_loop, "epoch_progress", None)
        progress = getattr(epoch_progress, "current", None)
        processed_epoch = getattr(progress, "processed", current_epoch)
        if type(processed_epoch) is not int or processed_epoch < 0:
            raise RuntimeError("Trainer processed epoch is invalid")
        loader.sampler.set_epoch(processed_epoch)
        return loader

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)


class ISPY2FlowMatchingSystem(pl.LightningModule):
    """Joint MOTFM and longitudinal-conditioning training system."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        model: torch.nn.Module | None = None,
        conditioner: torch.nn.Module | None = None,
        model_factory=None,
        conditioner_factory=None,
        device: torch.device | str | int | None = None,
        source_ftv_train_values: torch.Tensor | None = None,
        estimated_optimizer_steps: int | None = None,
        checkpoint_metadata: Mapping[str, Any] | None = None,
        validation_sampler=sample_with_solver,
    ) -> None:
        super().__init__()
        self.config = config
        loads_default_conditioner = conditioner is None and conditioner_factory is None
        uses_default_component = (model is None and model_factory is None) or (
            loads_default_conditioner
        )
        if type(device) is int:
            runtime_device = torch.device("cuda", device)
        elif device is None:
            runtime_device = torch.device(
                "cuda", torch.cuda.current_device()
            ) if uses_default_component and torch.cuda.is_available() else torch.device("cpu")
        else:
            runtime_device = torch.device(device)
        if uses_default_component and (
            runtime_device.type != "cuda" or not torch.cuda.is_available()
        ):
            raise ValueError(
                "real MedGemma longitudinal training requires one explicit CUDA device"
            )
        if model is None:
            if model_factory is not None:
                model = model_factory(config["model_args"], runtime_device)
            else:
                model = build_model(config["model_args"], runtime_device)
        if conditioner is None:
            if conditioner_factory is not None:
                conditioner = conditioner_factory(config, runtime_device)
            else:
                conditioning = config.get("conditioning_args", {})
                model_id = official_medgemma_model_id(
                    conditioning.get("model_id", MEDGEMMA_MODEL_NAME)
                )
                model_revision = official_medgemma_revision(
                    conditioning.get("model_revision", "")
                )
                text_encoder = MedGemmaTextEncoder(
                    model_name=model_id,
                    model_revision=model_revision,
                    device=runtime_device,
                )
                conditioner = ISPY2Conditioner(
                    text_encoder=text_encoder,
                    source_ftv_train_values=source_ftv_train_values,
                )
        if not isinstance(model, torch.nn.Module) or not isinstance(
            conditioner, torch.nn.Module
        ):
            raise TypeError("model and conditioner factories must return torch modules")
        if loads_default_conditioner:
            resolve_frozen_base_identity(
                conditioner,
                model_id=model_id,
                configured_revision=model_revision,
            )
        self.model = model
        self.conditioner = conditioner
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.estimated_optimizer_steps = estimated_optimizer_steps
        self.training_logic_device = runtime_device
        self.validation_sampler = validation_sampler
        self.checkpoint_metadata = None
        self.last_checkpoint_binding: dict[str, Any] | None = None
        if checkpoint_metadata is not None:
            self.bind_checkpoint_metadata(checkpoint_metadata)

        self.activation_checkpoint = bool(
            config.get("train_args", {}).get(
                "activation_checkpoint",
                config.get("model_args", {}).get("activation_checkpoint", False),
            )
        )

    def bind_checkpoint_metadata(self, checkpoint_metadata: Mapping[str, Any]) -> None:
        """Bind validated resume metadata to the already-loaded frozen base."""

        _validate_checkpoint_metadata(checkpoint_metadata)
        expected_model = checkpoint_metadata["model"]
        actual_model = resolve_frozen_base_identity(
            self.conditioner,
            model_id=expected_model["id"],
            configured_revision=expected_model["revision"],
        )
        if actual_model != expected_model:
            raise CheckpointCompatibilityError(
                "frozen base identity does not match checkpoint metadata"
            )
        expected_optimization = resolve_optimization_contract(
            self.config,
            train_batches_per_epoch=checkpoint_metadata["optimization"][
                "train_batches_per_epoch"
            ],
        )
        if expected_optimization != checkpoint_metadata["optimization"]:
            raise CheckpointCompatibilityError(
                "checkpoint optimization metadata is incompatible with the current config"
            )
        expected_steps = expected_optimization["scheduler"]["total_optimizer_steps"]
        if (
            self.estimated_optimizer_steps is not None
            and self.estimated_optimizer_steps != expected_steps
        ):
            raise CheckpointCompatibilityError(
                "configured optimizer step estimate is incompatible with checkpoint metadata"
            )
        self.estimated_optimizer_steps = expected_steps
        self.checkpoint_metadata = deepcopy(dict(checkpoint_metadata))

    def compute_flow_matching_loss(
        self,
        batch: Mapping[str, Any],
        *,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs = batch["model_inputs"]
        target = inputs["target_image"]
        spatial = inputs["spatial_condition"]
        if noise is None:
            noise = torch.randn_like(target)
        if time is None:
            time = torch.rand(target.shape[0], device=target.device, dtype=torch.float32)
        if noise.shape != target.shape:
            raise ValueError("flow noise shape must match target_image")
        if time.shape != (target.shape[0],):
            raise ValueError("flow time must have shape [B]")
        context = self.conditioner(
            clinical_text=inputs["clinical_text"],
            action_text=inputs["action_text"],
            delta_days=inputs["delta_days"],
            stage_id=inputs["stage_id"],
            source_ftv_volume_cc=inputs["source_ftv_volume_cc"],
        )
        sample = self.path.sample(x_0=noise, x_1=target, t=time)
        if self.activation_checkpoint:
            def motfm_forward(x_t, path_time, condition, spatial_condition):
                return self.model(
                    x_t,
                    path_time,
                    condition,
                    spatial_condition=spatial_condition,
                )

            prediction = checkpoint(
                motfm_forward,
                sample.x_t,
                sample.t,
                context,
                spatial,
                use_reentrant=False,
            )
        else:
            prediction = self.model(
                sample.x_t,
                sample.t,
                context,
                spatial_condition=spatial,
            )
        if prediction.shape != sample.dx_t.shape:
            raise ValueError("MOTFM velocity shape must match the CondOT target")
        return F.mse_loss(prediction, sample.dx_t)

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        loss = self.compute_flow_matching_loss(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        loss = self.compute_flow_matching_loss(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def generate_validation_sample(
        self,
        batch: Mapping[str, Any],
        *,
        completed_epoch: int,
        run_dir: str | Path,
    ) -> Path:
        """Generate one deterministic ROI without reading any target-side field."""

        if type(completed_epoch) is not int or completed_epoch <= 0:
            raise ValueError("validation sample epoch must be a positive integer")
        inputs = batch.get("model_inputs")
        if not isinstance(inputs, Mapping):
            raise ValueError("validation sample model inputs are invalid")
        required = {
            "spatial_condition",
            "clinical_text",
            "action_text",
            "delta_days",
            "stage_id",
            "source_ftv_volume_cc",
        }
        if not required.issubset(inputs):
            raise ValueError("validation sample conditions are incomplete")

        def first_text(value: Any) -> list[str]:
            if isinstance(value, str):
                return [value]
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError("validation sample text conditions are invalid")
            return [value[0]]

        def first_tensor(value: Any) -> torch.Tensor:
            if not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[0] < 1:
                raise ValueError("validation sample numeric conditions are invalid")
            return value[:1].to(self.device)

        spatial = first_tensor(inputs["spatial_condition"])
        output_shape = tuple(int(value) for value in self.config["data_args"]["output_shape_zyx"])
        spacing_xyz = _validation_spacing_xyz(self.config)
        out_channels = int(self.config["model_args"].get("out_channels", 1))
        expected_spatial = (1, 2, *output_shape)
        if tuple(spatial.shape) != expected_spatial or out_channels != 1:
            raise ValueError("validation sample spatial condition shape is invalid")
        seed = int(self.config.get("train_args", {}).get("validation_sample_seed", 2026))
        seed += completed_epoch
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        noise = torch.randn(
            (1, out_channels, *output_shape),
            generator=generator,
            device=self.device,
            dtype=spatial.dtype,
        )
        precision_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), precision_context:
            context = self.conditioner(
                clinical_text=first_text(inputs["clinical_text"]),
                action_text=first_text(inputs["action_text"]),
                delta_days=first_tensor(inputs["delta_days"]),
                stage_id=first_tensor(inputs["stage_id"]),
                source_ftv_volume_cc=first_tensor(inputs["source_ftv_volume_cc"]),
            )
            trajectory = self.validation_sampler(
                model=self.model,
                x_init=noise,
                solver_config=self.config.get("solver_args", {}),
                cond=context,
                spatial_condition=spatial,
            )
        if not isinstance(trajectory, torch.Tensor):
            raise RuntimeError("validation sample generation failed")
        generated = trajectory[-1] if trajectory.ndim == len(output_shape) + 3 else trajectory
        expected_generated = (1, out_channels, *output_shape)
        if tuple(generated.shape) != expected_generated or not torch.isfinite(generated).all():
            raise RuntimeError("validation sample generation produced an invalid ROI")
        values, clipped = prepare_validation_values(
            generated,
            registered=(output_shape == (96, 256, 256)),
        )
        volume_xyz = np.ascontiguousarray(values[0, 0].permute(2, 1, 0).numpy())
        affine = np.diag([*spacing_xyz, 1.0]).astype(np.float64)
        artifact = {
            "clipped_to_unit_interval": clipped,
            "dtype": "float32",
            "epoch": completed_epoch,
            "local_ras_affine": affine.tolist(),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "min": float(values.min()),
            "schema": VALIDATION_ROI_SCHEMA,
            "seed": seed,
            "shape_zyx": list(output_shape),
            "spacing_xyz": list(spacing_xyz),
            "std": float(values.std(unbiased=False)),
        }
        destination = (
            Path(run_dir).resolve()
            / "validation_samples"
            / f"epoch_{completed_epoch:04d}.json"
        )
        volume_temporary: Path | None = None
        sidecar_temporary: Path | None = None
        published_volume: Path | None = None
        published_volume_created = False
        lock_descriptor: int | None = None
        lock_acquired = False
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            lock_path = destination.with_name(f".epoch_{completed_epoch:04d}.lock")
            lock_flags = os.O_RDWR | os.O_CREAT
            lock_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            lock_descriptor = os.open(lock_path, lock_flags, 0o600)
            lock_metadata = os.fstat(lock_descriptor)
            if not stat.S_ISREG(lock_metadata.st_mode) or lock_metadata.st_nlink != 1:
                raise OSError("validation sample lock is invalid")
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
            lock_acquired = True
            token = f"{os.getpid()}.{secrets.token_hex(8)}"
            volume_temporary = destination.with_name(
                f".epoch_{completed_epoch:04d}.{token}.tmp.nii.gz"
            )
            image = nib.Nifti1Image(volume_xyz, affine)
            image.set_qform(affine, code=1)
            image.set_sform(affine, code=1)
            nib.save(image, volume_temporary)
            with volume_temporary.open("rb+") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            volume_sha256 = _sha256_file(volume_temporary)
            volume_destination = destination.with_name(
                f"epoch_{completed_epoch:04d}.{volume_sha256}.nii.gz"
            )
            volume_existed = volume_destination.exists()
            os.replace(volume_temporary, volume_destination)
            volume_temporary = None
            published_volume = volume_destination
            published_volume_created = not volume_existed
            _fsync_directory(destination.parent)

            artifact["volume_file"] = volume_destination.name
            artifact["volume_sha256"] = volume_sha256
            sidecar_temporary = destination.with_name(
                f".{destination.name}.{token}.tmp"
            )
            payload = (
                json.dumps(artifact, sort_keys=True, indent=2, allow_nan=False) + "\n"
            )
            with sidecar_temporary.open("x", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(sidecar_temporary, destination)
            sidecar_temporary = None
            _fsync_directory(destination.parent)
        except Exception:
            for temporary in (volume_temporary, sidecar_temporary):
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
            if published_volume_created and published_volume is not None:
                try:
                    committed = json.loads(destination.read_text(encoding="utf-8"))
                    volume_is_committed = (
                        isinstance(committed, Mapping)
                        and committed.get("volume_file") == published_volume.name
                    )
                except Exception:
                    volume_is_committed = False
                if not volume_is_committed:
                    try:
                        published_volume.unlink(missing_ok=True)
                        _fsync_directory(destination.parent)
                    except OSError:
                        pass
            raise RuntimeError("validation sample artifact could not be written") from None
        finally:
            if lock_descriptor is not None:
                if lock_acquired:
                    try:
                        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                    except OSError:
                        pass
                try:
                    os.close(lock_descriptor)
                except OSError:
                    pass
        return destination

    def on_validation_epoch_end(self) -> None:
        completed_epoch = int(self.current_epoch) + 1
        if completed_epoch % 5 != 0:
            return
        if self._trainer is None or not bool(getattr(self.trainer, "is_global_zero", False)):
            return
        try:
            batch = next(iter(self.trainer.datamodule.val_dataloader()))
        except (AttributeError, StopIteration, TypeError):
            raise RuntimeError("validation sample batch is unavailable") from None
        self.generate_validation_sample(
            batch,
            completed_epoch=completed_epoch,
            run_dir=self.trainer.default_root_dir,
        )

    def _optimizer_parameter_groups(self) -> list[dict[str, Any]]:
        model_parameters: list[torch.nn.Parameter] = []
        lora_parameters: list[torch.nn.Parameter] = []
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                model_parameters.append(parameter)
        for name, parameter in self.conditioner.named_parameters():
            if not parameter.requires_grad:
                continue
            in_text_backbone = name.startswith("text_encoder.model.")
            is_adapter = any(
                marker in name.lower()
                for marker in ("lora_", "adapter", "modules_to_save")
            )
            if in_text_backbone and not is_adapter:
                raise ValueError(
                    f"unexpected trainable frozen-base parameter in text encoder: {name}"
                )
            (lora_parameters if in_text_backbone else model_parameters).append(parameter)

        grouped = model_parameters + lora_parameters
        trainable = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if len({id(parameter) for parameter in grouped}) != len(grouped) or {
            id(parameter) for parameter in grouped
        } != {id(parameter) for parameter in trainable}:
            raise RuntimeError(
                "optimizer groups must contain every trainable parameter exactly once"
            )
        if not model_parameters or not lora_parameters:
            raise ValueError("optimizer requires nonempty model and LoRA parameter groups")
        train = self.config["train_args"]
        return [
            {
                "name": "model",
                "params": model_parameters,
                "lr": float(train.get("model_lr", 1e-4)),
            },
            {
                "name": "lora",
                "params": lora_parameters,
                "lr": float(train.get("text_lr", 2e-5)),
            },
        ]

    def _total_optimizer_steps(self) -> int:
        if self.checkpoint_metadata is not None:
            expected = self.checkpoint_metadata["optimization"]["scheduler"][
                "total_optimizer_steps"
            ]
            if self.estimated_optimizer_steps != expected:
                raise CheckpointCompatibilityError(
                    "configured optimizer step estimate is incompatible with checkpoint metadata"
                )
            return expected
        total = self.estimated_optimizer_steps
        if type(total) is not int or total <= 0:
            raise ValueError("estimated optimizer steps must be a positive integer")
        return total

    def on_fit_start(self) -> None:
        if self.checkpoint_metadata is None:
            return
        optimization = self.checkpoint_metadata["optimization"]
        trainer = self.trainer
        if (
            type(trainer.max_epochs) is not int
            or trainer.max_epochs != optimization["max_epochs"]
            or type(trainer.max_steps) is not int
            or trainer.max_steps != -1
            or type(trainer.accumulate_grad_batches) is not int
            or trainer.accumulate_grad_batches
            != optimization["gradient_accumulation_steps"]
            or type(trainer.limit_train_batches) is not float
            or trainer.limit_train_batches != 1.0
        ):
            raise CheckpointCompatibilityError(
                "Trainer optimizer step estimate is incompatible with checkpoint metadata"
            )

    def on_train_start(self) -> None:
        if self.checkpoint_metadata is None:
            return
        expected = self.checkpoint_metadata["optimization"]["scheduler"][
            "total_optimizer_steps"
        ]
        try:
            trainer_estimate = self.trainer.estimated_stepping_batches
        except (TypeError, ValueError, OverflowError):
            raise CheckpointCompatibilityError(
                "Trainer optimizer step estimate is invalid"
            ) from None
        if type(trainer_estimate) is not int or trainer_estimate <= 0:
            raise CheckpointCompatibilityError("Trainer optimizer step estimate is invalid")
        if trainer_estimate != expected:
            raise CheckpointCompatibilityError(
                "Trainer optimizer step estimate is incompatible with checkpoint metadata"
            )

    def configure_optimizers(self) -> dict[str, Any]:
        train = self.config["train_args"]
        groups = self._optimizer_parameter_groups()
        optimizer_contract, configured_warmup_fraction = _resolved_optimizer_settings(train)
        optimizer_kwargs = {
            "weight_decay": optimizer_contract["weight_decay"],
            "betas": tuple(optimizer_contract["betas"]),
            "eps": optimizer_contract["eps"],
        }
        if not optimizer_contract["paged"]:
            optimizer = torch.optim.AdamW(groups, **optimizer_kwargs)
        else:
            try:
                from bitsandbytes.optim import PagedAdamW8bit
            except ImportError:
                raise RuntimeError(
                    "paged_adamw_8bit requires the bitsandbytes optimizer"
                ) from None
            optimizer = PagedAdamW8bit(groups, **optimizer_kwargs)

        total_steps = self._total_optimizer_steps()
        if self.checkpoint_metadata is not None:
            scheduler_contract = self.checkpoint_metadata["optimization"]["scheduler"]
            warmup_fraction = scheduler_contract["warmup_fraction"]
            warmup_steps = scheduler_contract["warmup_steps"]
        else:
            warmup_fraction = configured_warmup_fraction
            warmup_steps = max(1, int(total_steps * warmup_fraction))

        def lr_multiplier(step: int) -> float:
            if total_steps == 1:
                return 1.0
            if step < warmup_steps:
                return float(step) / float(warmup_steps)
            progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _frozen_text_state_keys(self) -> set[str]:
        prefix = "conditioner.text_encoder.model."
        trainable = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        return {
            name
            for name in self.state_dict()
            if name.startswith(prefix) and name not in trainable
        }

    def on_save_checkpoint(self, checkpoint_payload: dict[str, Any]) -> None:
        if self.checkpoint_metadata is None:
            raise CheckpointCompatibilityError(
                "checkpoint metadata must be initialized before saving"
            )
        state = checkpoint_payload.get("state_dict")
        if not isinstance(state, dict):
            raise CheckpointCompatibilityError("checkpoint state_dict is missing")
        for name in self._frozen_text_state_keys():
            state.pop(name, None)
        checkpoint_payload[CHECKPOINT_METADATA_KEY] = deepcopy(self.checkpoint_metadata)
        checkpoint_payload[CHECKPOINT_RNG_KEY] = _capture_rng_state(
            self.training_logic_device
        )
        binding = _checkpoint_binding(
            self.checkpoint_metadata,
            epoch=checkpoint_payload.get("epoch"),
            global_step=checkpoint_payload.get("global_step"),
            nonce=secrets.token_hex(32),
        )
        checkpoint_payload[CHECKPOINT_BINDING_KEY] = binding
        self.last_checkpoint_binding = deepcopy(binding)

    def on_load_checkpoint(self, checkpoint_payload: dict[str, Any]) -> None:
        if self.checkpoint_metadata is None:
            raise CheckpointCompatibilityError(
                "checkpoint metadata must be initialized before loading"
            )
        incoming = checkpoint_payload.get(CHECKPOINT_METADATA_KEY)
        if incoming is None:
            raise CheckpointCompatibilityError(
                "checkpoint metadata is missing; old checkpoints are incompatible"
            )
        _validate_checkpoint_metadata(incoming)
        if incoming != self.checkpoint_metadata:
            raise CheckpointCompatibilityError(
                "checkpoint metadata is incompatible with the current training contract"
            )
        incoming_binding = checkpoint_payload.get(CHECKPOINT_BINDING_KEY)
        if not isinstance(incoming_binding, Mapping) or set(incoming_binding) != {
            "schema_version",
            "metadata_sha256",
            "epoch",
            "global_step",
            "nonce",
            "binding_id",
        }:
            raise CheckpointCompatibilityError("checkpoint binding is missing or invalid")
        expected_binding = _checkpoint_binding(
            incoming,
            epoch=checkpoint_payload.get("epoch"),
            global_step=checkpoint_payload.get("global_step"),
            nonce=incoming_binding.get("nonce"),
        )
        if incoming_binding != expected_binding:
            raise CheckpointCompatibilityError("checkpoint binding is missing or invalid")
        self.last_checkpoint_binding = deepcopy(expected_binding)
        state = checkpoint_payload.get("state_dict")
        if not isinstance(state, dict):
            raise CheckpointCompatibilityError("checkpoint state_dict is missing")
        current_state = self.state_dict()
        frozen_keys = self._frozen_text_state_keys()
        if frozen_keys.intersection(state):
            raise CheckpointCompatibilityError(
                "checkpoint must not contain frozen base state"
            )
        for name in frozen_keys:
            if name not in state:
                state[name] = current_state[name]
        _restore_rng_state(
            checkpoint_payload.get(CHECKPOINT_RNG_KEY), self.training_logic_device
        )

    def export_peft_adapter(self, output_dir: str | Path) -> Path:
        backbone = getattr(getattr(self.conditioner, "text_encoder", None), "model", None)
        save_pretrained = getattr(backbone, "save_pretrained", None)
        if not callable(save_pretrained):
            raise RuntimeError("the active text encoder does not provide PEFT adapter export")
        destination = Path(output_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        save_pretrained(destination, safe_serialization=True)
        return destination

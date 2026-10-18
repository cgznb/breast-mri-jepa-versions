"""Condition encoders for I-SPY2 longitudinal flow matching."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


MEDGEMMA_MODEL_NAME = "google/medgemma-4b-it"
TEXT_EMBEDDING_DIM = 768
CONDITION_DIM = 512


class MedGemmaLoadError(RuntimeError):
    """Raised when the gated quantized text encoder cannot be constructed."""


def _config_hidden_size(config: Any) -> int | None:
    if isinstance(config, Mapping):
        hidden_size = config.get("hidden_size")
    else:
        hidden_size = getattr(config, "hidden_size", None)
    if type(hidden_size) is int and hidden_size > 0:
        return hidden_size
    return None


def _find_text_model(model: nn.Module) -> tuple[str, nn.Module]:
    candidates = []
    for name, module in model.named_modules():
        if name and name.rsplit(".", 1)[-1] in {"language_model", "text_model"}:
            candidates.append((name, module))
    if candidates:
        return min(candidates, key=lambda item: (item[0].count("."), len(item[0])))

    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    if text_config is None and _config_hidden_size(config) is not None:
        return "", model
    raise ValueError("the multimodal model does not expose a text model")


def _text_hidden_size(model: nn.Module, text_model: nn.Module) -> int | None:
    config = getattr(model, "config", None)
    candidates = (
        getattr(config, "text_config", None),
        getattr(text_model, "config", None),
        config,
    )
    for candidate in candidates:
        hidden_size = _config_hidden_size(candidate)
        if hidden_size is not None:
            return hidden_size
    return None


def _text_lora_targets(model: nn.Module) -> list[str]:
    text_path, text_model = _find_text_model(model)
    targets = []
    for child_name, module in text_model.named_modules():
        if not child_name or child_name.rsplit(".", 1)[-1] not in {"q_proj", "v_proj"}:
            continue
        if not hasattr(module, "weight"):
            continue
        targets.append(f"{text_path}.{child_name}" if text_path else child_name)
    if not targets:
        raise ValueError("the text model has no supported LoRA targets")
    return targets


def _cuda_device_index(device: torch.device | str | int | None) -> int:
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required for 4-bit MedGemma training")
    if device is None:
        device_index = torch.cuda.current_device()
    elif type(device) is int:
        device_index = device
    else:
        requested = torch.device(device)
        if requested.type != "cuda":
            raise ValueError("4-bit MedGemma training requires a CUDA device")
        device_index = (
            torch.cuda.current_device() if requested.index is None else requested.index
        )
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise ValueError("the requested CUDA device index is unavailable")
    return device_index


def _injected_model_device(device: torch.device | str | int) -> torch.device:
    if type(device) is int:
        if device < 0:
            raise ValueError("device index must be nonnegative")
        return torch.device("cuda", device)
    return torch.device(device)


def load_quantized_medgemma(
    model_name: str = MEDGEMMA_MODEL_NAME,
    *,
    revision: str | None = None,
    device: torch.device | str | int | None = None,
    transformers_api: Any | None = None,
    peft_api: Any | None = None,
) -> tuple[Any, nn.Module]:
    """Load the pure-text MedGemma backbone with the required NF4 QLoRA setup."""
    try:
        device_index = _cuda_device_index(device)
        if transformers_api is None:
            import transformers as transformers_api
        if peft_api is None:
            import peft as peft_api

        quantization_config = transformers_api.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        revision_kwargs = {"revision": revision} if revision is not None else {}
        tokenizer = transformers_api.AutoTokenizer.from_pretrained(
            model_name, **revision_kwargs
        )
        model = transformers_api.AutoModel.from_pretrained(
            model_name,
            **revision_kwargs,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": device_index},
        )
        target_modules = _text_lora_targets(model)
        model = peft_api.prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
        lora_config = peft_api.LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=target_modules,
            bias="none",
            task_type=peft_api.TaskType.FEATURE_EXTRACTION,
        )
        model = peft_api.get_peft_model(model, lora_config)
        return tokenizer, model
    except Exception:
        raise MedGemmaLoadError(
            "Unable to load the gated MedGemma text encoder. Install the transformers, "
            "peft, and bitsandbytes dependencies, use a CUDA device with 4-bit support, "
            "accept the model license, and authenticate with Hugging Face."
        ) from None


def _normalize_text_batch(texts: str | Sequence[str], name: str) -> list[str]:
    if isinstance(texts, str):
        normalized = [texts]
    elif isinstance(texts, Sequence):
        normalized = list(texts)
    else:
        raise TypeError(f"{name} must contain strings")
    if not normalized or any(not isinstance(text, str) or not text.strip() for text in normalized):
        raise TypeError(f"{name} must contain nonempty strings")
    return normalized


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


class MedGemmaTextEncoder(nn.Module):
    """Shared masked-mean text encoder with a learned 768-dimensional projection."""

    output_dim = TEXT_EMBEDDING_DIM

    def __init__(
        self,
        model_name: str = MEDGEMMA_MODEL_NAME,
        *,
        model_revision: str | None = None,
        output_dim: int = TEXT_EMBEDDING_DIM,
        max_length: int = 512,
        device: torch.device | str | int | None = None,
        tokenizer: Any | None = None,
        model: nn.Module | None = None,
        loader=load_quantized_medgemma,
    ) -> None:
        super().__init__()
        if type(output_dim) is not int or output_dim <= 0:
            raise ValueError("text output_dim must be a positive integer")
        if type(max_length) is not int or max_length <= 0:
            raise ValueError("text max_length must be a positive integer")
        if (tokenizer is None) != (model is None):
            raise ValueError("tokenizer and model must be injected together")
        injected = tokenizer is not None
        if tokenizer is None:
            if model_revision is None:
                tokenizer, model = loader(model_name, device=device)
            else:
                tokenizer, model = loader(
                    model_name, revision=model_revision, device=device
                )
        if not isinstance(model, nn.Module):
            raise TypeError("the text model must be a torch module")
        if injected and device is not None:
            model = model.to(_injected_model_device(device))

        text_model_path, text_model = _find_text_model(model)
        hidden_size = _text_hidden_size(model, text_model)
        if hidden_size is None:
            raise ValueError("the text model must declare a positive hidden_size")

        self.tokenizer = tokenizer
        self.model = model
        self.text_model_path = text_model_path
        self.output_dim = output_dim
        self.max_length = max_length
        self.training_device = _module_device(text_model)
        self.projection = nn.Linear(hidden_size, output_dim, device=self.training_device)

        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_("lora_" in name or "modules_to_save" in name)
        if hasattr(text_model, "gradient_checkpointing_enable"):
            text_model.gradient_checkpointing_enable()
        if hasattr(text_model, "enable_input_require_grads"):
            text_model.enable_input_require_grads()
        elif hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

    def forward(self, texts: str | Sequence[str]) -> torch.Tensor:
        text_batch = _normalize_text_batch(texts, "text input")
        inputs = self.tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if not isinstance(inputs, Mapping) or "attention_mask" not in inputs:
            raise ValueError("the tokenizer must return an attention mask")
        text_model = (
            self.model.get_submodule(self.text_model_path)
            if self.text_model_path
            else self.model
        )
        device = _module_device(text_model)
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        else:
            inputs = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()
            }

        outputs = text_model(**inputs)
        hidden = getattr(outputs, "last_hidden_state", None)
        attention_mask = inputs["attention_mask"]
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise ValueError("the text model returned invalid hidden states")
        if (
            not isinstance(attention_mask, torch.Tensor)
            or attention_mask.shape != hidden.shape[:2]
        ):
            raise ValueError("the tokenizer attention mask shape is invalid")
        mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        token_counts = mask.sum(dim=1)
        if torch.any(token_counts <= 0):
            raise ValueError("the tokenizer produced an empty text sequence")
        pooled = (hidden * mask).sum(dim=1) / token_counts
        projection_device = self.projection.weight.device
        pooled = pooled.to(device=projection_device, dtype=self.projection.weight.dtype)
        projected = self.projection(pooled)
        if not torch.isfinite(projected).all():
            raise ValueError("the projected text condition must be finite")
        norms = projected.norm(p=2, dim=-1, keepdim=True)
        if not torch.isfinite(norms).all() or torch.any(norms <= 0):
            raise ValueError("the projected text condition has an invalid norm")
        return F.normalize(projected, p=2, dim=-1)


def _validate_numeric_vector(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{name} must have shape [B]")
    if value.dtype == torch.bool or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


class FourierDaysEncoder(nn.Module):
    """CLARITY-style fixed Fourier features for positive day intervals."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("frequencies", torch.logspace(0, 4, 32))
        self.projection = nn.Linear(64, 128)

    def forward(self, delta_days: torch.Tensor) -> torch.Tensor:
        _validate_numeric_vector(delta_days, "delta_days")
        if torch.any(delta_days <= 0):
            raise ValueError("delta_days must be positive")
        days = delta_days.to(device=self.frequencies.device, dtype=self.frequencies.dtype)
        if not torch.isfinite(days).all():
            raise ValueError("delta_days must remain finite after casting")
        angles = 2.0 * math.pi * (days[:, None] / 365.0) * self.frequencies[None, :]
        if not torch.isfinite(angles).all():
            raise ValueError("delta_days Fourier angles must be finite")
        features = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        encoded = self.projection(features.to(dtype=self.projection.weight.dtype))
        if not torch.isfinite(encoded).all():
            raise ValueError("delta_days encoding must be finite")
        return encoded


class SourceFTVEncoder(nn.Module):
    """Train-statistic normalization and MLP encoding for source-only FTV."""

    def __init__(self, train_values: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("log_mean", torch.tensor(float("nan")))
        self.register_buffer("log_std", torch.tensor(float("nan")))
        self.mlp = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, 64))
        if train_values is not None:
            self.fit(train_values)

    @staticmethod
    def _logged_values(values: torch.Tensor) -> torch.Tensor:
        _validate_numeric_vector(values, "source FTV")
        if torch.any(values <= 0):
            raise ValueError("source FTV must be positive")
        logged = torch.log1p(values.to(dtype=torch.float64))
        if not torch.isfinite(logged).all():
            raise ValueError("source FTV log values must be finite")
        return logged

    @torch.no_grad()
    def fit(self, train_values: torch.Tensor) -> "SourceFTVEncoder":
        logged = self._logged_values(train_values)
        mean = logged.mean()
        std = logged.std(unbiased=False)
        if not torch.isfinite(mean) or not torch.isfinite(std) or std <= 0:
            raise ValueError("source FTV training statistics require a positive finite std")
        self.log_mean.copy_(mean.to(device=self.log_mean.device, dtype=self.log_mean.dtype))
        self.log_std.copy_(std.to(device=self.log_std.device, dtype=self.log_std.dtype))
        return self

    @torch.no_grad()
    def set_statistics(self, log_mean: float, log_std: float) -> "SourceFTVEncoder":
        mean = torch.as_tensor(log_mean, dtype=self.log_mean.dtype, device=self.log_mean.device)
        std = torch.as_tensor(log_std, dtype=self.log_std.dtype, device=self.log_std.device)
        if mean.ndim != 0 or std.ndim != 0 or not torch.isfinite(mean) or not torch.isfinite(std):
            raise ValueError("source FTV statistics must be finite scalars")
        if std <= 0:
            raise ValueError("source FTV log_std must be positive")
        self.log_mean.copy_(mean)
        self.log_std.copy_(std)
        return self

    def forward(self, source_ftv_volume_cc: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(self.log_mean) or not torch.isfinite(self.log_std):
            raise RuntimeError("source FTV training statistics have not been fitted")
        logged = self._logged_values(source_ftv_volume_cc)
        logged = logged.to(device=self.log_mean.device, dtype=self.log_mean.dtype)
        if not torch.isfinite(logged).all():
            raise ValueError("source FTV must remain finite after casting")
        standardized = ((logged - self.log_mean) / self.log_std).clamp(-5.0, 5.0)
        encoded = self.mlp(standardized[:, None].to(dtype=self.mlp[0].weight.dtype))
        if not torch.isfinite(encoded).all():
            raise ValueError("source FTV encoding must be finite")
        return encoded


class ISPY2Conditioner(nn.Module):
    """Fuse clinical, action, interval, stage, and source FTV into MOTFM context."""

    def __init__(
        self,
        *,
        text_encoder: nn.Module | None = None,
        source_ftv_train_values: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder or MedGemmaTextEncoder()
        if getattr(self.text_encoder, "output_dim", None) != TEXT_EMBEDDING_DIM:
            raise ValueError("the shared text encoder output dimension must be 768")
        self.days_encoder = FourierDaysEncoder()
        self.stage_embedding = nn.Embedding(4, 64, padding_idx=0)
        self.source_ftv_encoder = SourceFTVEncoder(source_ftv_train_values)
        self.fusion = nn.Sequential(
            nn.LayerNorm(1792),
            nn.Linear(1792, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, CONDITION_DIM),
            nn.LayerNorm(CONDITION_DIM),
        )

    @staticmethod
    def _validate_stage(stage_id: torch.Tensor, batch_size: int) -> None:
        _validate_numeric_vector(stage_id, "stage_id")
        if stage_id.shape[0] != batch_size:
            raise ValueError("stage_id batch size does not match text batch")
        if stage_id.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("stage_id must use an integer tensor dtype")
        if torch.any((stage_id < 0) | (stage_id > 3)):
            raise ValueError("stage_id must be in the stage vocabulary [0, 3]")

    @staticmethod
    def _validate_embedding(
        embedding: torch.Tensor, name: str, batch_size: int, width: int
    ) -> None:
        if not isinstance(embedding, torch.Tensor) or embedding.shape != (batch_size, width):
            raise ValueError(f"{name} must have shape [B, {width}]")
        if not torch.isfinite(embedding).all():
            raise ValueError(f"{name} must be finite")

    def forward(
        self,
        clinical_text: str | Sequence[str],
        action_text: str | Sequence[str],
        delta_days: torch.Tensor,
        stage_id: torch.Tensor,
        source_ftv_volume_cc: torch.Tensor,
    ) -> torch.Tensor:
        clinical_batch = _normalize_text_batch(clinical_text, "clinical_text")
        action_batch = _normalize_text_batch(action_text, "action_text")
        batch_size = len(clinical_batch)
        if len(action_batch) != batch_size:
            raise ValueError("action_text batch size does not match clinical_text batch")
        _validate_numeric_vector(delta_days, "delta_days")
        _validate_numeric_vector(source_ftv_volume_cc, "source FTV")
        if delta_days.shape[0] != batch_size or source_ftv_volume_cc.shape[0] != batch_size:
            raise ValueError("numeric condition batch size does not match text batch")
        self._validate_stage(stage_id, batch_size)

        clinical = self.text_encoder(clinical_batch)
        action = self.text_encoder(action_batch)
        self._validate_embedding(clinical, "clinical text encoding", batch_size, 768)
        self._validate_embedding(action, "action text encoding", batch_size, 768)
        days = self.days_encoder(delta_days)
        stage = self.stage_embedding(
            stage_id.to(device=self.stage_embedding.weight.device, dtype=torch.long)
        )
        source_ftv = self.source_ftv_encoder(source_ftv_volume_cc)

        fusion_device = self.fusion[0].weight.device
        fusion_dtype = self.fusion[0].weight.dtype
        parts = [clinical, action, days, stage, source_ftv]
        parts = [part.to(device=fusion_device, dtype=fusion_dtype) for part in parts]
        if any(not torch.isfinite(part).all() for part in parts):
            raise ValueError("condition components must remain finite after casting")
        combined = torch.cat(parts, dim=-1)
        if combined.shape != (batch_size, 1792):
            raise ValueError("the concatenated condition must have shape [B, 1792]")
        output = self.fusion(combined)
        if output.shape != (batch_size, CONDITION_DIM) or not torch.isfinite(output).all():
            raise ValueError("the fused condition must be finite with shape [B, 512]")
        return output.unsqueeze(1)

    def trainable_parameter_groups(self, *, named: bool = False) -> dict[str, Any]:
        """Return disjoint text/non-text groups for optimizers and checkpoints."""
        groups: dict[str, list[Any]] = {"text": [], "non_text": []}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            key = "text" if name.startswith("text_encoder.") else "non_text"
            groups[key].append((name, parameter) if named else parameter)
        if named:
            return {key: dict(values) for key, values in groups.items()}
        return groups

    def optimizer_parameter_groups(
        self, *, model_lr: float, text_lr: float
    ) -> list[dict[str, Any]]:
        if not math.isfinite(model_lr) or model_lr <= 0:
            raise ValueError("model_lr must be positive and finite")
        if not math.isfinite(text_lr) or text_lr <= 0:
            raise ValueError("text_lr must be positive and finite")
        groups = self.trainable_parameter_groups()
        return [
            {"name": "text", "params": groups["text"], "lr": text_lr},
            {"name": "non_text", "params": groups["non_text"], "lr": model_lr},
        ]


# Backward-friendly aliases for callers that use a more explicit class name.
SharedMedGemmaTextEncoder = MedGemmaTextEncoder
LongitudinalConditioner = ISPY2Conditioner


__all__ = [
    "CONDITION_DIM",
    "FourierDaysEncoder",
    "ISPY2Conditioner",
    "LongitudinalConditioner",
    "MEDGEMMA_MODEL_NAME",
    "MedGemmaLoadError",
    "MedGemmaTextEncoder",
    "SharedMedGemmaTextEncoder",
    "SourceFTVEncoder",
    "TEXT_EMBEDDING_DIM",
    "load_quantized_medgemma",
]

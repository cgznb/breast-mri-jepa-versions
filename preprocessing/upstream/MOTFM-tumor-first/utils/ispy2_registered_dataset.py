"""Read-only dataset for the registered Strict-A T0-fixed cache."""

from __future__ import annotations

import math
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from scripts.build_ispy2_registered_strict_a_bundle import verify_bundle
from utils.ispy2_registered import (
    CACHE_SCHEMA,
    COORDINATE_FRAME,
    VALID_FOLDS,
    sha256_json,
)


STAGE_BY_TRANSITION = {"T0->T1": 1, "T1->T2": 2, "T2->T3": 3}


def is_registered_strict_a_bundle(path: str | Path) -> bool:
    candidate = Path(path)
    candidate = candidate / "bundle.json" if candidate.is_dir() else candidate
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return value.get("schema_version") == "motfm_ispy2_registered_strict_a_bundle_v1"


def registered_cache_contract(bundle: Mapping[str, Any]) -> dict[str, str]:
    value = bundle.get("base_cache_contract")
    if value is None:
        artifacts = bundle["artifacts"]
        value = {
            "schema": CACHE_SCHEMA,
            "bundle_contract_sha256": bundle["bundle_contract_sha256"],
            "preprocess_sha256": artifacts["preprocess"]["sha256"],
            "normalization_sha256": artifacts["normalization"]["sha256"],
        }
    required = {
        "schema",
        "bundle_contract_sha256",
        "preprocess_sha256",
        "normalization_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("registered Strict-A base cache contract is invalid")
    resolved = {key: str(value[key]) for key in required}
    if resolved["schema"] != CACHE_SCHEMA or any(
        len(resolved[key]) != 64 for key in required - {"schema"}
    ):
        raise ValueError("registered Strict-A base cache contract is invalid")
    return resolved


def registered_cache_key(bundle: Mapping[str, Any], visit_id: str) -> str:
    contract = registered_cache_contract(bundle)
    return sha256_json(
        {
            "schema": CACHE_SCHEMA,
            "visit_id": str(visit_id),
            "bundle_contract_sha256": contract["bundle_contract_sha256"],
            "preprocess_sha256": contract["preprocess_sha256"],
            "normalization_sha256": contract["normalization_sha256"],
        }
    )


class RegisteredStrictADataset(Dataset):
    """Expose the established MOTFM item boundary from immutable z-score caches."""

    def __init__(
        self,
        bundle_path: str | Path,
        *,
        fold: str | Sequence[str],
        cache_dir: str | Path,
        output_shape_zyx: Sequence[int] | None = None,
        target_spacing_xyz: Sequence[float] | None = None,
        read_only: bool = True,
        include_evaluation_metadata: bool = False,
        source_records: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        if not read_only:
            raise ValueError("registered Strict-A dataset cache is read-only")
        self.bundle_path = Path(bundle_path).expanduser().resolve()
        self.bundle = verify_bundle(self.bundle_path)
        requested = (fold,) if isinstance(fold, str) else tuple(str(value) for value in fold)
        if not requested or set(requested).difference(VALID_FOLDS):
            raise ValueError("registered Strict-A fold is invalid")
        root = self.bundle_path.parent
        preprocess = json.loads(
            (root / self.bundle["artifacts"]["preprocess"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        configured_shape = tuple(int(value) for value in preprocess.get("output_shape_zyx", ()))
        configured_spacing = tuple(
            float(value) for value in preprocess.get("target_spacing_xyz", ())
        )
        if configured_shape:
            if output_shape_zyx is not None and tuple(output_shape_zyx) != configured_shape:
                raise ValueError("registered Strict-A output shape override is incompatible")
            self.output_shape_zyx = configured_shape
        else:
            self.output_shape_zyx = (
                tuple(int(value) for value in output_shape_zyx)
                if output_shape_zyx is not None
                else ()
            )
        if configured_spacing:
            if target_spacing_xyz is not None and tuple(target_spacing_xyz) != configured_spacing:
                raise ValueError("registered Strict-A spacing override is incompatible")
            self.target_spacing_xyz = configured_spacing
        else:
            self.target_spacing_xyz = (
                tuple(float(value) for value in target_spacing_xyz)
                if target_spacing_xyz is not None
                else ()
            )
        transitions = pd.read_csv(
            root / self.bundle["artifacts"]["transitions"]["path"]
        ).to_dict(orient="records")
        if source_records is None:
            self._transitions = [row for row in transitions if str(row["fold"]) in requested]
            self._source_only = False
        else:
            self._transitions = [deepcopy(dict(row)) for row in source_records]
            self._source_only = True
            if any("target" in str(key).lower() for row in self._transitions for key in row):
                raise ValueError("registered source-only records contain target fields")
        if not self._transitions:
            raise ValueError("registered Strict-A selected dataset is empty")
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.include_evaluation_metadata = bool(include_evaluation_metadata)

    def __len__(self) -> int:
        return len(self._transitions)

    def _load(self, visit_id: str) -> dict[str, Any]:
        key = registered_cache_key(self.bundle, visit_id)
        contract = registered_cache_contract(self.bundle)
        path = self.cache_dir / f"{key}.pt"
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, TypeError, ValueError, EOFError):
            raise ValueError(f"registered Strict-A cache is unavailable: {visit_id}") from None
        required = {
            "schema",
            "cache_key",
            "visit_id",
            "bundle_contract_sha256",
            "preprocess_sha256",
            "normalization_sha256",
            "mri",
            "mask",
            "valid_foreground",
            "crop_plan",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("registered Strict-A cache fields are invalid")
        if (
            payload["schema"] != CACHE_SCHEMA
            or payload["cache_key"] != key
            or payload["visit_id"] != visit_id
            or payload["bundle_contract_sha256"] != contract["bundle_contract_sha256"]
            or payload["preprocess_sha256"] != contract["preprocess_sha256"]
            or payload["normalization_sha256"] != contract["normalization_sha256"]
        ):
            raise ValueError("registered Strict-A cache identity is invalid")
        mri, mask, valid = payload["mri"], payload["mask"], payload["valid_foreground"]
        if (
            not isinstance(mri, torch.Tensor)
            or mri.dtype != torch.float16
            or mri.ndim != 4
            or mri.shape[0] != 2
            or not bool(torch.isfinite(mri).all())
        ):
            raise ValueError("registered Strict-A cached MRI is invalid")
        shape = tuple(mri.shape[1:])
        if self.output_shape_zyx and shape != self.output_shape_zyx:
            raise ValueError("registered Strict-A cached MRI shape is invalid")
        if (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.uint8
            or tuple(mask.shape) != (1, *shape)
            or not bool(((mask == 0) | (mask == 1)).all())
            or not isinstance(valid, torch.Tensor)
            or valid.dtype != torch.uint8
            or tuple(valid.shape) != (1, *shape)
            or not bool(((valid == 0) | (valid == 1)).all())
        ):
            raise ValueError("registered Strict-A cached mask is invalid")
        return payload

    @staticmethod
    def _positive(value: Any, *, name: str) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name} must be positive and finite") from None
        if not math.isfinite(parsed) or parsed <= 0:
            raise ValueError(f"{name} must be positive and finite")
        return parsed

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._source_only:
            raise ValueError("registered source-only dataset has no indexed target")
        row = self._transitions[index]
        source = self._load(str(row["source_visit_id"]))
        target = self._load(str(row["target_visit_id"]))
        item: dict[str, Any] = {
            "model_inputs": {
                "target_image": target["mri"][0:1].float(),
                "spatial_condition": source["mri"].float(),
                "clinical_text": str(row["clinical_text"]),
                "action_text": str(row["action_text"]),
                "delta_days": int(row["delta_days"]),
                "stage_id": STAGE_BY_TRANSITION[str(row["transition_type"])],
                "source_ftv_volume_cc": self._positive(
                    row["source_ftv_volume_cc"], name="source FTV"
                ),
            },
            "metadata": {
                "transition_id": str(row["transition_id"]),
                "source_visit_id": str(row["source_visit_id"]),
                "target_visit_id": str(row["target_visit_id"]),
                "fold": str(row["fold"]),
            },
        }
        if self.include_evaluation_metadata:
            item["evaluation_metadata"] = {
                "target_mask": target["mask"].clone(),
                "target_ftv_volume_cc": self._positive(
                    row["target_ftv_volume_cc"], name="target FTV"
                ),
            }
        return item

    def source_condition(
        self, source_visit_id: str, *, source_ftv_volume_cc: float
    ) -> dict[str, Any]:
        matches = [
            row for row in self._transitions if str(row["source_visit_id"]) == source_visit_id
        ]
        if len(matches) != 1:
            raise ValueError("source visit must identify exactly one transition")
        row = matches[0]
        source = self._load(source_visit_id)
        return {
            "model_inputs": {
                "spatial_condition": source["mri"].float(),
                "clinical_text": str(row["clinical_text"]),
                "action_text": str(row["action_text"]),
                "delta_days": int(row["delta_days"]),
                "stage_id": STAGE_BY_TRANSITION[str(row["transition_type"])],
                "source_ftv_volume_cc": self._positive(
                    source_ftv_volume_cc, name="source FTV"
                ),
            },
            "source_geometry": {
                "mask": source["mask"].clone(),
                "valid_foreground": source["valid_foreground"].clone(),
                "preprocess_metadata": deepcopy(source["crop_plan"]),
            },
            "metadata": {
                "transition_id": str(row["transition_id"]),
                "source_visit_id": source_visit_id,
                "fold": str(row["fold"]),
                "transition_type": str(row["transition_type"]),
                "coordinate_frame": COORDINATE_FRAME,
            },
        }


def collate_registered_strict_a(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("registered Strict-A batch must be nonempty")
    has_evaluation = "evaluation_metadata" in batch[0]
    if any(("evaluation_metadata" in item) != has_evaluation for item in batch):
        raise ValueError("registered Strict-A batch mixes evaluation boundaries")
    inputs = [item["model_inputs"] for item in batch]
    target = torch.stack([item["target_image"] for item in inputs])
    spatial = torch.stack([item["spatial_condition"] for item in inputs])
    if not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(spatial).all()):
        raise ValueError("registered Strict-A MRI batch must be finite")
    output: dict[str, Any] = {
        "model_inputs": {
            "target_image": target,
            "spatial_condition": spatial,
            "clinical_text": [str(item["clinical_text"]) for item in inputs],
            "action_text": [str(item["action_text"]) for item in inputs],
            "delta_days": torch.tensor(
                [float(item["delta_days"]) for item in inputs], dtype=torch.float32
            ),
            "stage_id": torch.tensor(
                [int(item["stage_id"]) for item in inputs], dtype=torch.long
            ),
            "source_ftv_volume_cc": torch.tensor(
                [float(item["source_ftv_volume_cc"]) for item in inputs],
                dtype=torch.float32,
            ),
        },
        "metadata": [deepcopy(dict(item["metadata"])) for item in batch],
    }
    if has_evaluation:
        output["evaluation_metadata"] = {
            "target_mask": torch.stack(
                [item["evaluation_metadata"]["target_mask"] for item in batch]
            ),
            "target_ftv_volume_cc": torch.tensor(
                [float(item["evaluation_metadata"]["target_ftv_volume_cc"]) for item in batch],
                dtype=torch.float32,
            ),
        }
    return output

#!/usr/bin/env python3
"""Materialize the immutable registered Strict-A ROI cache."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from scripts.build_ispy2_registered_strict_a_bundle import (
    _crop_visit_images,
    _load_resampled,
    verify_bundle,
)
from utils.ispy2_registered import (
    CACHE_SCHEMA,
    CropPlan,
    ForegroundStatistics,
    apply_global_zscore,
    crop_or_pad_zyx,
)
from utils.ispy2_registered_dataset import registered_cache_contract, registered_cache_key


def build_cache_payload_from_arrays(
    *,
    bundle: Mapping[str, Any],
    visit_id: str,
    cache_key: str,
    dce0: np.ndarray,
    ser: np.ndarray,
    mask: np.ndarray,
    statistics: ForegroundStatistics,
    crop_plan: Mapping[str, Any],
) -> dict[str, Any]:
    normalized, valid = apply_global_zscore(dce0, ser, statistics)
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.shape != tuple(normalized.shape[1:]):
        raise ValueError("registered cache Mask and MRI shape mismatch")
    contract = registered_cache_contract(bundle)
    return {
        "schema": CACHE_SCHEMA,
        "cache_key": str(cache_key),
        "visit_id": str(visit_id),
        "bundle_contract_sha256": contract["bundle_contract_sha256"],
        "preprocess_sha256": contract["preprocess_sha256"],
        "normalization_sha256": contract["normalization_sha256"],
        "mri": torch.from_numpy(normalized).to(dtype=torch.float16).contiguous(),
        "mask": torch.from_numpy(binary[None]).to(dtype=torch.uint8).contiguous(),
        "valid_foreground": torch.from_numpy(valid[None].astype(np.uint8)).contiguous(),
        "crop_plan": dict(crop_plan),
    }


def _plan(value: Mapping[str, Any]) -> CropPlan:
    return CropPlan(
        coordinate_frame=str(value["coordinate_frame"]),
        input_shape_zyx=tuple(int(item) for item in value["input_shape_zyx"]),
        output_shape_zyx=tuple(int(item) for item in value["output_shape_zyx"]),
        bbox_min_zyx=tuple(int(item) for item in value["bbox_min_zyx"]),
        bbox_max_zyx=tuple(int(item) for item in value["bbox_max_zyx"]),
        center_zyx=tuple(int(item) for item in value["center_zyx"]),
        crop_start_zyx=tuple(int(item) for item in value["crop_start_zyx"]),
        t0_mask_voxel_count=int(value["t0_mask_voxel_count"]),
    )


def _load_existing(path: Path, expected: Mapping[str, Any]) -> bool:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError, EOFError, pickle.UnpicklingError):
        return False
    return bool(
        isinstance(payload, dict)
        and payload.get("schema") == CACHE_SCHEMA
        and payload.get("cache_key") == expected["cache_key"]
        and payload.get("visit_id") == expected["visit_id"]
        and payload.get("bundle_contract_sha256")
        == expected["bundle_contract_sha256"]
        and payload.get("preprocess_sha256") == expected["preprocess_sha256"]
        and payload.get("normalization_sha256") == expected["normalization_sha256"]
    )


def storage_estimate_bytes(
    bundle: Mapping[str, Any], shape_zyx: tuple[int, int, int]
) -> dict[str, int]:
    voxels = int(np.prod(shape_zyx, dtype=np.int64))
    visits = int(bundle["counts"]["visit_count"])
    transitions = int(bundle["counts"]["transition_count"])
    val_transitions = int(bundle["counts"]["fold_transition_counts"]["val"])
    components = {
        # Two float16 MRI channels plus uint8 Mask and foreground Mask.
        "roi_cache_bytes": visits * voxels * 6,
        "sdf_cache_bytes": transitions * voxels * 2,
        # Historical sparse Teacher entries peak below 9 MiB per transition.
        "teacher_cache_bytes": transitions * 9 * 1024**2,
        "validation_mask_cache_bytes": val_transitions * 8 * voxels * 2,
        "checkpoint_bytes": 16 * 1024**3,
        "transaction_temp_bytes": 3 * 1024**3,
    }
    subtotal = sum(components.values())
    headroom = int(subtotal * 0.2)
    return {
        **components,
        "subtotal_bytes": subtotal,
        "headroom_bytes": headroom,
        "required_free_bytes": subtotal + headroom,
    }


def required_free_bytes(bundle: Mapping[str, Any], shape_zyx: tuple[int, int, int]) -> int:
    return storage_estimate_bytes(bundle, shape_zyx)["required_free_bytes"]


def cache_build_counter(*, destination_existed: bool) -> str:
    return "rebuilt" if destination_existed else "built"


def build_cache(
    *,
    bundle_path: Path,
    cache_dir: Path,
    visit_ids: set[str] | None = None,
    append_only: bool = False,
    skip_space_check: bool = False,
    progress_every: int = 25,
) -> dict[str, int]:
    bundle = verify_bundle(bundle_path)
    if int(bundle["counts"].get("empty_source_or_target_mask_transitions", 0)) != 0:
        raise RuntimeError("registered Strict-A UOT training does not support empty endpoint Masks")
    root = bundle_path.parent
    preprocess = json.loads(
        (root / bundle["artifacts"]["preprocess"]["path"]).read_text(encoding="utf-8")
    )
    shape = tuple(int(value) for value in preprocess["output_shape_zyx"])
    spacing = tuple(float(value) for value in preprocess["target_spacing_xyz"])
    cache_dir = cache_dir.expanduser().resolve()
    probe = cache_dir.parent
    while not probe.exists():
        probe = probe.parent
    estimate = storage_estimate_bytes(bundle, shape)
    required = estimate["required_free_bytes"]
    estimate_path = cache_dir.parent / "storage_estimate.json"
    estimate_path.parent.mkdir(parents=True, exist_ok=True)
    estimate_path.write_text(
        json.dumps(estimate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not skip_space_check and shutil.disk_usage(probe).free < required:
        raise OSError(
            f"registered Strict-A cache requires at least {required / 1024**3:.1f} GiB free"
        )
    statistics = ForegroundStatistics.from_dict(
        json.loads(
            (root / bundle["artifacts"]["normalization"]["path"]).read_text(
                encoding="utf-8"
            )
        )
    )
    crop_values = json.loads(
        (root / bundle["artifacts"]["crop_plans"]["path"]).read_text(encoding="utf-8")
    )
    plans = {patient: _plan(value) for patient, value in crop_values.items()}
    visits = pd.read_csv(root / bundle["artifacts"]["visits"]["path"]).to_dict(
        orient="records"
    )
    available_ids = {str(visit["visit_id"]) for visit in visits}
    if len(available_ids) != len(visits):
        raise ValueError("registered bundle visit identities are duplicated")
    if visit_ids is not None:
        requested = {str(value) for value in visit_ids}
        if not requested or "" in requested:
            raise ValueError("registered ROI visit filter must be nonempty")
        missing = sorted(requested.difference(available_ids))
        if missing:
            raise ValueError(f"registered ROI visit filter contains unknown IDs: {missing}")
        visits = [visit for visit in visits if str(visit["visit_id"]) in requested]
    cache_dir.mkdir(parents=True, exist_ok=True)
    counts = {"total": len(visits), "built": 0, "reused": 0, "rebuilt": 0}
    contract = registered_cache_contract(bundle)
    for index, visit in enumerate(visits, start=1):
        visit_id = str(visit["visit_id"])
        key = registered_cache_key(bundle, visit_id)
        identity = {
            "cache_key": key,
            "visit_id": visit_id,
            "bundle_contract_sha256": contract["bundle_contract_sha256"],
            "preprocess_sha256": contract["preprocess_sha256"],
            "normalization_sha256": contract["normalization_sha256"],
        }
        destination = cache_dir / f"{key}.pt"
        destination_existed = destination.exists()
        if destination.is_file() and _load_existing(destination, identity):
            counts["reused"] += 1
        else:
            if destination_existed and append_only:
                raise FileExistsError(
                    f"registered ROI append-only collision: {visit_id}"
                )
            plan = plans[str(visit["patient_id"])]
            dce0, ser = _crop_visit_images(
                visit, plan, target_spacing_xyz=spacing
            )
            mask = _load_resampled(
                visit["mask_path"],
                visit["meta_path"],
                is_mask=True,
                target_spacing_xyz=spacing,
            )
            if tuple(mask.shape) != plan.input_shape_zyx:
                raise ValueError("registered cache Mask does not share patient T0 grid")
            mask = crop_or_pad_zyx(mask, plan.crop_start_zyx, plan.output_shape_zyx)
            payload = build_cache_payload_from_arrays(
                bundle=bundle,
                visit_id=visit_id,
                cache_key=key,
                dce0=dce0,
                ser=ser,
                mask=mask,
                statistics=statistics,
                crop_plan=plan.to_dict(),
            )
            with tempfile.NamedTemporaryFile(
                dir=cache_dir, prefix=f".{key}.", suffix=".tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
            try:
                torch.save(payload, temporary)
                if not _load_existing(temporary, identity):
                    raise RuntimeError("registered cache atomic readback failed")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            counts[cache_build_counter(destination_existed=destination_existed)] += 1
        if index % progress_every == 0 or index == len(visits):
            print(json.dumps({"event": "roi_cache", "completed": index, **counts}), flush=True)
    if counts["built"] + counts["rebuilt"] + counts["reused"] != counts["total"]:
        raise RuntimeError("registered cache completion accounting is invalid")
    return counts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--visit-allowlist", type=Path)
    parser.add_argument("--append-only", action="store_true")
    parser.add_argument("--skip-space-check", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    visit_ids = None
    if args.visit_allowlist is not None:
        values = [
            line.strip()
            for line in args.visit_allowlist.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not values or len(values) != len(set(values)):
            raise ValueError("registered ROI visit allowlist must be nonempty and unique")
        visit_ids = set(values)
    report = build_cache(
        bundle_path=args.bundle.expanduser().resolve(),
        cache_dir=args.cache_dir,
        visit_ids=visit_ids,
        append_only=args.append_only,
        skip_space_check=args.skip_space_check,
        progress_every=args.progress_every,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

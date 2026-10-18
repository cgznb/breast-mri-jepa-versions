#!/usr/bin/env python3
"""Compare the final Z96 Strict-A cohort with the documented native-grid 1263 reference."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
import pandas as pd

from scripts.build_ispy2_registered_strict_a_bundle import (
    DEFAULT_LEGACY_BUNDLE,
    DEFAULT_REGISTERED_MANIFEST,
    TARGET_SPACING_XYZ,
    _assess_registered_visits,
    _read_json,
    _registered_rows,
    strict_a_transition_reasons,
    verify_bundle,
)
from utils.ispy2_registered import (
    CropPlan,
    reconstruct_transition_universe,
    sha256_file,
    t0_fixed_crop_plan,
)


REFERENCE_SHAPE_ZYX = (128, 256, 256)
REFERENCE_SELECTED_COUNT = 1263
SCHEMA = "motfm_ispy2_strict_a_reference_difference_v1"


def reference_crop_plan(
    t0_mask_zyx: np.ndarray,
    *,
    output_shape_zyx: Sequence[int] = REFERENCE_SHAPE_ZYX,
) -> CropPlan:
    """Reproduce the bbox-centering convention used by the frozen 1263 audit."""
    plan = t0_fixed_crop_plan(t0_mask_zyx, output_shape_zyx=output_shape_zyx)
    starts = tuple(
        (low + high + 1 - size) // 2
        for low, high, size in zip(
            plan.bbox_min_zyx, plan.bbox_max_zyx, plan.output_shape_zyx
        )
    )
    center = tuple(
        start + size // 2 for start, size in zip(starts, plan.output_shape_zyx)
    )
    return replace(plan, center_zyx=center, crop_start_zyx=starts)


def partition_decisions(
    *, current: Mapping[str, str], reference: Mapping[str, str]
) -> tuple[dict[str, str], dict[str, str]]:
    if not set(reference).issubset(current):
        raise ValueError("current bundle is missing frozen reference transitions")
    return (
        {key: str(current[key]) for key in sorted(reference)},
        {key: str(current[key]) for key in sorted(set(current).difference(reference))},
    )


def compare_decisions(
    *, current: Mapping[str, str], reference: Mapping[str, str]
) -> list[dict[str, str]]:
    frozen, _ = partition_decisions(current=current, reference=reference)
    rows: list[dict[str, str]] = []
    for transition_id in sorted(reference):
        current_reason = frozen[transition_id]
        reference_reason = str(reference[transition_id])
        current_included = current_reason == "strict_a_pass"
        reference_included = reference_reason == "strict_a_pass"
        if current_included == reference_included:
            continue
        rows.append(
            {
                "transition_id": transition_id,
                "reference_decision": "included" if reference_included else "excluded",
                "reference_reason": reference_reason,
                "current_decision": "included" if current_included else "excluded",
                "current_reason": current_reason,
            }
        )
    return rows


def _native_mask(row: Mapping[str, Any]) -> np.ndarray:
    try:
        mask = np.asanyarray(nib.load(str(row["mask_path"])).dataobj)
    except (KeyError, OSError, ValueError):
        raise ValueError("native registered Mask is unreadable") from None
    if mask.ndim != 3 or not np.isfinite(mask).all():
        raise ValueError("native registered Mask is invalid")
    return np.asarray(mask)


def _category(reason: str) -> str:
    codes = set(str(reason).split("+"))
    if "patient_has_no_T0" in codes:
        return "not_evaluable_no_T0"
    if reason == "strict_a_pass":
        return "quality_pass_masks_complete"
    quality = any(code.endswith("quality_failed") for code in codes)
    cropped = any(code.endswith("mask_not_fully_retained") for code in codes)
    if quality and cropped:
        return "quality_review_and_mask_cropped"
    if quality:
        return "quality_review_masks_complete"
    if cropped:
        return "quality_pass_mask_cropped"
    return "other_exclusion"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def run_reference_audit(
    *,
    bundle_path: Path,
    output_dir: Path,
    legacy_bundle_path: Path = DEFAULT_LEGACY_BUNDLE,
    registered_manifest_path: Path = DEFAULT_REGISTERED_MANIFEST,
    progress_every: int = 100,
) -> dict[str, Any]:
    bundle = verify_bundle(bundle_path)
    if sha256_file(legacy_bundle_path) != bundle["source_contracts"]["legacy_bundle_sha256"]:
        raise ValueError("reference audit legacy bundle changed")
    if (
        sha256_file(registered_manifest_path)
        != bundle["source_contracts"]["registered_manifest_sha256"]
    ):
        raise ValueError("reference audit registered manifest changed")

    legacy = _read_json(legacy_bundle_path)
    legacy_root = legacy_bundle_path.parent
    legacy_visits = pd.read_csv(legacy_root / legacy["artifacts"]["visits"]["path"])
    accepted = pd.read_csv(legacy_root / legacy["artifacts"]["transitions"]["path"])
    prior_exclusions = pd.read_csv(legacy_root / legacy["artifacts"]["exclusions"]["path"])
    universe = reconstruct_transition_universe(legacy_visits, accepted, prior_exclusions)
    registered = pd.read_csv(registered_manifest_path)
    registered_visits = _registered_rows(legacy_visits, registered)
    reference_visits, reference_plans = _assess_registered_visits(
        registered_visits,
        target_spacing_xyz=TARGET_SPACING_XYZ,
        output_shape_zyx=REFERENCE_SHAPE_ZYX,
        progress_every=progress_every,
        mask_loader=_native_mask,
        crop_plan_builder=reference_crop_plan,
    )
    by_visit = {
        str(row["visit_id"]): row
        for row in reference_visits.to_dict(orient="records")
    }
    reference_decisions: dict[str, str] = {}
    universe_by_id: dict[str, dict[str, Any]] = {}
    for transition in universe:
        transition_id = str(transition["transition_id"])
        universe_by_id[transition_id] = dict(transition)
        reasons = strict_a_transition_reasons(
            transition,
            by_visit,
            patient_has_t0=str(transition["patient_id"]) in reference_plans,
        )
        reference_decisions[transition_id] = "+".join(reasons) if reasons else "strict_a_pass"

    root = bundle_path.parent
    transitions = pd.read_csv(root / bundle["artifacts"]["transitions"]["path"])
    exclusions = pd.read_csv(root / bundle["artifacts"]["exclusions"]["path"])
    all_current_decisions = {
        str(row["transition_id"]): "strict_a_pass"
        for row in transitions.to_dict(orient="records")
    }
    all_current_decisions.update(
        {
            str(row["transition_id"]): str(row["reason"])
            for row in exclusions.to_dict(orient="records")
        }
    )
    current_decisions, expanded_decisions = partition_decisions(
        current=all_current_decisions, reference=reference_decisions
    )
    differences = compare_decisions(
        current=current_decisions, reference=reference_decisions
    )
    for row in differences:
        transition = universe_by_id[row["transition_id"]]
        row.update(
            {
                "patient_id": str(transition["patient_id"]),
                "transition_type": str(transition["transition_type"]),
                "reference_category": _category(row["reference_reason"]),
                "current_category": _category(row["current_reason"]),
            }
        )
    differences.sort(key=lambda row: row["transition_id"])

    reference_selected = sum(
        reason == "strict_a_pass" for reason in reference_decisions.values()
    )
    if reference_selected != REFERENCE_SELECTED_COUNT:
        raise RuntimeError(
            f"native-grid Strict-A reference changed: {reference_selected} != {REFERENCE_SELECTED_COUNT}"
        )
    current_selected = sum(
        reason == "strict_a_pass" for reason in current_decisions.values()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "reference_1263_transition_differences.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    columns = [
        "transition_id",
        "patient_id",
        "transition_type",
        "reference_decision",
        "reference_reason",
        "reference_category",
        "current_decision",
        "current_reason",
        "current_category",
    ]
    pd.DataFrame(differences, columns=columns).to_csv(temporary, index=False)
    os.replace(temporary, csv_path)
    report = {
        "schema": SCHEMA,
        "bundle_contract_sha256": bundle["bundle_contract_sha256"],
        "reference_shape_zyx": list(REFERENCE_SHAPE_ZYX),
        "current_shape_zyx": [96, 256, 256],
        "universe_transition_count": len(current_decisions),
        "reference_selected_count": reference_selected,
        "current_selected_count": current_selected,
        "selected_count_delta": current_selected - reference_selected,
        "expanded_transition_count": len(expanded_decisions),
        "expanded_selected_count": sum(
            reason == "strict_a_pass" for reason in expanded_decisions.values()
        ),
        "membership_change_count": len(differences),
        "reference_category_counts": dict(
            sorted(Counter(_category(value) for value in reference_decisions.values()).items())
        ),
        "current_category_counts": dict(
            sorted(Counter(_category(value) for value in current_decisions.values()).items())
        ),
        "difference_csv": {
            "path": csv_path.name,
            "sha256": sha256_file(csv_path),
            "row_count": len(differences),
        },
    }
    _atomic_json(output_dir / "reference_1263_summary.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = run_reference_audit(
        bundle_path=args.bundle.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        progress_every=args.progress_every,
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Move high-transition incremental patients into a balanced validation holdout."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.build_ispy2_registered_strict_a_bundle import (
    ARTIFACT_FILENAMES,
    _artifact,
    verify_bundle,
)
from utils.ispy2_registered import sha256_file, sha256_json
from utils.ispy2_registry import load_clinical_table, normalize_patient_id


HOLDOUT_SPLIT_POLICY = "frozen_base_val_new_patient_holdout_v1"
BALANCE_COLUMNS = ("pCR", "HR", "HER2", "MP", "trial_arm")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_patient_ids(path: Path) -> tuple[str, ...]:
    values = tuple(
        sorted(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    )
    if (
        not values
        or len(values) != len(set(values))
        or any(normalize_patient_id(value) != value for value in values)
    ):
        raise ValueError("validation candidate patient identities are invalid")
    return values


def balanced_patient_holdout(
    patient_rows: Sequence[Mapping[str, Any]],
    *,
    holdout_count: int,
    seed: int = 2026,
) -> tuple[str, ...]:
    if type(holdout_count) is not int or holdout_count <= 0:
        raise ValueError("validation holdout count must be a positive integer")
    if seed != 2026:
        raise ValueError("validation holdout requires seed 2026")
    normalized: list[dict[str, str]] = []
    for source in patient_rows:
        patient_id = str(source.get("patient_id", "")).strip()
        if not patient_id:
            raise ValueError("validation holdout patient ID must be nonempty")
        normalized.append(
            {
                "patient_id": patient_id,
                **{
                    column: str(source.get(column, "missing")).strip() or "missing"
                    for column in BALANCE_COLUMNS
                },
            }
        )
    if (
        holdout_count >= len(normalized)
        or len({row["patient_id"] for row in normalized}) != len(normalized)
    ):
        raise ValueError("validation holdout pool or requested count is invalid")

    capacities = {
        "train": len(normalized) - holdout_count,
        "val": holdout_count,
    }
    fractions = {
        fold: capacity / len(normalized) for fold, capacity in capacities.items()
    }
    totals = Counter(
        f"{column}={row[column]}"
        for row in normalized
        for column in BALANCE_COLUMNS
    )
    rng = np.random.default_rng(seed)
    jitter = {row["patient_id"]: float(rng.random()) for row in normalized}
    ordered = sorted(
        normalized,
        key=lambda row: (
            -sum(
                1.0 / totals[f"{column}={row[column]}"]
                for column in BALANCE_COLUMNS
            ),
            jitter[row["patient_id"]],
            row["patient_id"],
        ),
    )
    split: dict[str, list[str]] = {"train": [], "val": []}
    assigned = {fold: Counter() for fold in split}
    for row in ordered:
        features = [f"{column}={row[column]}" for column in BALANCE_COLUMNS]
        choices = [
            fold for fold in split if len(split[fold]) < capacities[fold]
        ]

        def cost(fold: str) -> tuple[float, int, str]:
            score = 0.0
            for feature in features:
                target = totals[feature] * fractions[fold]
                current = assigned[fold][feature]
                score += ((current + 1 - target) ** 2 - (current - target) ** 2) / max(
                    target, 1.0
                )
            return score, len(split[fold]), fold

        selected_fold = min(choices, key=cost)
        split[selected_fold].append(row["patient_id"])
        assigned[selected_fold].update(features)
    return tuple(sorted(split["val"]))


def _rewrite_fold(path: Path, selected: set[str]) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or "patient_id" not in fieldnames or "fold" not in fieldnames:
        raise ValueError(f"bundle fold table is invalid: {path}")
    covered: set[str] = set()
    for row in rows:
        patient_id = str(row["patient_id"])
        if patient_id in selected:
            if row["fold"] != "train":
                raise ValueError("validation holdout patient is not currently train")
            row["fold"] = "val"
            covered.add(patient_id)
    if covered != selected:
        raise ValueError("validation holdout patient table coverage is incomplete")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_validation_holdout_bundle(
    *,
    source_bundle_path: Path,
    output_dir: Path,
    selected_patient_ids: Sequence[str],
    selection_report: Mapping[str, Any],
) -> dict[str, Any]:
    source_bundle_path = source_bundle_path.expanduser().resolve()
    source = verify_bundle(source_bundle_path)
    if source.get("split_policy") != "frozen_base_val_new_train_v1":
        raise ValueError("validation holdout source must be the new-train expansion")
    selected = {str(value) for value in selected_patient_ids}
    if not selected or len(selected) != len(selected_patient_ids):
        raise ValueError("validation holdout selection is empty or duplicated")

    source_root = source_bundle_path.parent
    split = json.loads(
        (source_root / source["artifacts"]["split"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    train = {str(value) for value in split["train"]}
    val = {str(value) for value in split["val"]}
    increment_ids = set(
        (source_root / "increment/new_patient_ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    if not selected <= train or not selected <= increment_ids or selected.intersection(val):
        raise ValueError("validation holdout selection is outside incremental train")

    output = output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError("validation holdout bundle output already exists")
    staging = output.with_name(f".{output.name}.building")
    if staging.exists():
        raise FileExistsError("validation holdout bundle staging already exists")
    shutil.copytree(source_root, staging)
    try:
        _rewrite_fold(staging / ARTIFACT_FILENAMES["visits"], selected)
        _rewrite_fold(staging / ARTIFACT_FILENAMES["transitions"], selected)
        updated_split = {
            "train": sorted(train.difference(selected)),
            "val": sorted(val | selected),
        }
        _write_json(staging / ARTIFACT_FILENAMES["split"], updated_split)

        visits = pd.read_csv(staging / ARTIFACT_FILENAMES["visits"])
        transitions = pd.read_csv(staging / ARTIFACT_FILENAMES["transitions"])
        bundle = dict(source)
        bundle["artifacts"] = dict(source["artifacts"])
        for name in ("visits", "transitions", "split"):
            path = staging / ARTIFACT_FILENAMES[name]
            row_count = len(visits) if name == "visits" else (
                len(transitions) if name == "transitions" else None
            )
            bundle["artifacts"][name] = (
                _artifact(path, row_count=row_count)
                if row_count is not None
                else _artifact(path)
            )
        bundle["counts"] = dict(source["counts"])
        bundle["counts"]["fold_patient_counts"] = {
            "train": len(updated_split["train"]),
            "val": len(updated_split["val"]),
        }
        bundle["counts"]["fold_transition_counts"] = {
            fold: int((transitions["fold"].astype(str) == fold).sum())
            for fold in ("train", "val")
        }
        bundle["split_policy"] = HOLDOUT_SPLIT_POLICY
        unsigned = dict(bundle)
        unsigned.pop("bundle_contract_sha256", None)
        bundle["bundle_contract_sha256"] = sha256_json(unsigned)
        _write_json(staging / "bundle.json", bundle)

        holdout_root = staging / "validation_holdout"
        holdout_root.mkdir()
        (holdout_root / "patient_ids.txt").write_text(
            "".join(f"{value}\n" for value in sorted(selected)), encoding="utf-8"
        )
        report = {
            **dict(selection_report),
            "source_bundle_contract_sha256": source["bundle_contract_sha256"],
            "output_bundle_contract_sha256": bundle["bundle_contract_sha256"],
            "selected_patient_ids_sha256": sha256_file(
                holdout_root / "patient_ids.txt"
            ),
            "fold_patient_counts": bundle["counts"]["fold_patient_counts"],
            "fold_transition_counts": bundle["counts"]["fold_transition_counts"],
        }
        _write_json(holdout_root / "selection.json", report)
        verified = verify_bundle(staging / "bundle.json")
        staging.replace(output)
        return verified
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        column: dict(
            sorted(Counter(str(row[column]) for row in rows).items())
        )
        for column in BALANCE_COLUMNS
    }


def build_validation_holdout_bundle(
    *,
    source_bundle_path: Path,
    clinical_path: Path,
    candidate_patient_path: Path,
    output_dir: Path,
    holdout_count: int = 50,
    seed: int = 2026,
) -> dict[str, Any]:
    source = verify_bundle(source_bundle_path)
    source_root = source_bundle_path.expanduser().resolve().parent
    candidate_ids = set(_read_patient_ids(candidate_patient_path))
    transitions = pd.read_csv(source_root / ARTIFACT_FILENAMES["transitions"])
    candidate_transitions = transitions.loc[
        transitions["patient_id"].astype(str).isin(candidate_ids)
    ]
    transition_counts = candidate_transitions.groupby("patient_id").size()
    if transition_counts.empty:
        raise ValueError("validation holdout has no selected candidate transitions")
    maximum = int(transition_counts.max())
    eligible_ids = set(transition_counts.loc[transition_counts == maximum].index)
    if len(eligible_ids) < holdout_count:
        raise ValueError("validation holdout has too few high-transition patients")

    clinical = load_clinical_table(clinical_path)
    clinical = clinical.loc[clinical["patient_id"].astype(str).isin(eligible_ids)]
    clinical_by_id = {
        str(row["patient_id"]): row for row in clinical.to_dict(orient="records")
    }
    if set(clinical_by_id) != eligible_ids:
        raise ValueError("validation holdout clinical coverage is incomplete")
    patient_rows = [
        {
            "patient_id": patient_id,
            "pCR": clinical_by_id[patient_id]["pCR"],
            "HR": clinical_by_id[patient_id]["HR"],
            "HER2": clinical_by_id[patient_id]["HER2"],
            "MP": clinical_by_id[patient_id]["MP"],
            "trial_arm": clinical_by_id[patient_id]["Arm"],
        }
        for patient_id in sorted(eligible_ids)
    ]
    selected = balanced_patient_holdout(
        patient_rows, holdout_count=holdout_count, seed=seed
    )
    selected_set = set(selected)
    selected_rows = [
        row for row in patient_rows if row["patient_id"] in selected_set
    ]
    report = {
        "schema": "motfm_ispy2_incremental_validation_holdout_v1",
        "seed": seed,
        "selection_policy": "maximum_selected_transition_count_then_balanced_v1",
        "balance_columns": list(BALANCE_COLUMNS),
        "candidate_patient_count": len(candidate_ids),
        "selected_candidate_patient_count": int(transition_counts.size),
        "maximum_transition_count": maximum,
        "eligible_pool_patient_count": len(eligible_ids),
        "requested_holdout_patient_count": holdout_count,
        "selected_holdout_patient_count": len(selected),
        "selected_holdout_transition_count": int(
            transition_counts.loc[list(selected)].sum()
        ),
        "candidate_patient_allowlist_sha256": sha256_file(candidate_patient_path),
        "clinical_sha256": sha256_file(clinical_path),
        "eligible_distribution": _distribution(patient_rows),
        "selected_distribution": _distribution(selected_rows),
    }
    return write_validation_holdout_bundle(
        source_bundle_path=source_bundle_path,
        output_dir=output_dir,
        selected_patient_ids=selected,
        selection_report=report,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--clinical", type=Path, required=True)
    parser.add_argument("--candidate-patients", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-patients", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    bundle = build_validation_holdout_bundle(
        source_bundle_path=args.bundle,
        clinical_path=args.clinical,
        candidate_patient_path=args.candidate_patients,
        output_dir=args.output_dir,
        holdout_count=args.holdout_patients,
        seed=args.seed,
    )
    print(json.dumps({"counts": bundle["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

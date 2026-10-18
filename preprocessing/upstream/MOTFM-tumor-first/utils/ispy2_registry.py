"""Verified I-SPY2 registry, manifest, and clinical-table loading."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np


DATE_REGISTRY_SCHEMA = "meworld_phase3_date_registry_v3"
VOLUME_INDEX_SCHEMA = "meworld_phase3_volume_index_v1"
QC_VISITS_SCHEMA = "meworld_phase3_qc_visits_v1"

DATE_INDEX_FILENAME = "phase3_v2_date_index.csv"
DATE_QC_FILENAME = "phase3_v2_date_index_qc_ok.csv"
DATE_CONTRACT_FILENAME = "phase3_v2_date_registry_contract.json"
VOLUME_INDEX_FILENAME = "phase3_v2_ftv_volume_index.csv"
VOLUME_CONTRACT_FILENAME = "phase3_v2_ftv_volume_contract.json"
QC_VISITS_FILENAME = "phase3_v2_qc_visits.csv"
QC_CONTRACT_FILENAME = "phase3_v2_qc_visits_contract.json"

VISIT_TO_INDEX = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}
IDENTITY_COLUMNS = (
    "patient_id",
    "visit_id",
    "visit",
    "visit_index",
    "study_instance_uid",
)
DATE_COLUMNS = (
    *IDENTITY_COLUMNS,
    "visit_date",
    "visit_date_source",
    "date_status",
    "qc_status",
    "local_dicom_status",
    "local_study_date",
    "tcia_study_date",
    "local_series_dates_json",
    "local_acquisition_dates_json",
    "tcia_series_dates_json",
    "date_warnings_json",
    "local_header_count",
    "local_evidence_sha256",
    "date_row_sha256",
)
VOLUME_COLUMNS = (
    *IDENTITY_COLUMNS,
    "qc_status",
    "volume_status",
    "mask_path",
    "meta_path",
    "ftv_voxel_count_manifest",
    "ftv_voxel_count",
    "row_spacing_mm",
    "column_spacing_mm",
    "slice_spacing_mm",
    "ftv_voxel_volume_mm3",
    "ftv_volume_cc",
    "volume_row_sha256",
)
QC_COLUMNS = (
    *IDENTITY_COLUMNS,
    "visit_date",
    "visit_date_source",
    "qc_status",
    "mask_path",
    "meta_path",
    "ftv_voxel_count",
    "row_spacing_mm",
    "column_spacing_mm",
    "slice_spacing_mm",
    "ftv_voxel_volume_mm3",
    "ftv_volume_cc",
    "qc_visit_row_sha256",
)
CLINICAL_COLUMNS = (
    "Patient_ID",
    "Arm",
    "HR",
    "HER2",
    "MP",
    "pCR",
    "Age_at_Screening",
    "menopausal_status",
)


class ContractError(ValueError):
    """Raised when a versioned registry or identity relationship is invalid."""


@dataclass(frozen=True)
class VerifiedRegistries:
    date: pd.DataFrame
    volume: pd.DataFrame
    qc: pd.DataFrame
    contracts: Mapping[str, Mapping[str, Any]]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ContractError(f"{field} must be non-empty")
    return text


def normalize_patient_key(value: Any) -> str:
    """Normalize an I-SPY2 identifier to its numeric join key."""

    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if math.isfinite(value) and value.is_integer() else ""
    text = _text(value)
    if not text:
        return ""
    numeric = re.fullmatch(r"(\d+)(?:\.0+)?", text)
    if numeric:
        return numeric.group(1)
    suffix = re.search(r"(\d+)$", text)
    return suffix.group(1) if suffix else text


def normalize_patient_id(value: Any) -> str:
    """Normalize a clinical or registry ID to ``ISPY2-<numeric key>``."""

    key = normalize_patient_key(value)
    if not key:
        return ""
    return key if key.upper().startswith("ISPY2-") else f"ISPY2-{key}"


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must be a JSON object: {path}")
    return payload


def _read_csv_exact(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(columns):
                raise ContractError(f"unexpected CSV columns in {path}")
            rows = list(reader)
    except OSError as exc:
        raise ContractError(f"registry artifact is not readable: {path}") from exc
    if any(None in row or set(row) != set(columns) for row in rows):
        raise ContractError(f"malformed CSV row in {path}")
    return rows


def _validate_identity(row: Mapping[str, str], label: str) -> None:
    patient_id = _required_text(row.get("patient_id"), f"{label} patient_id")
    if normalize_patient_id(patient_id) != patient_id:
        raise ContractError(f"{label} patient_id is not canonical")
    visit = _required_text(row.get("visit"), f"{label} visit").upper()
    if visit not in VISIT_TO_INDEX or row.get("visit") != visit:
        raise ContractError(f"{label} visit must be T0, T1, T2, or T3")
    if row.get("visit_id") != f"{patient_id}:{visit}":
        raise ContractError(f"{label} visit_id does not match patient_id and visit")
    if row.get("visit_index") != str(VISIT_TO_INDEX[visit]):
        raise ContractError(f"{label} visit_index does not match visit")
    _required_text(row.get("study_instance_uid"), f"{label} study_instance_uid")


def _ensure_unique_identities(rows: Sequence[Mapping[str, str]], label: str) -> None:
    identities = [tuple(row[column] for column in IDENTITY_COLUMNS) for row in rows]
    patient_visits = [(row["patient_id"], row["visit"]) for row in rows]
    uids = [row["study_instance_uid"] for row in rows]
    if len(identities) != len(set(identities)):
        raise ContractError(f"{label} contains duplicate composite identities")
    if len(patient_visits) != len(set(patient_visits)) or len(uids) != len(set(uids)):
        raise ContractError(f"{label} contains conflicting patient visits or study UIDs")


def _validate_date_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for raw in rows:
        row = {column: _text(raw.get(column)) for column in DATE_COLUMNS}
        _validate_identity(row, "date row")
        try:
            date.fromisoformat(row["visit_date"])
            header_count = int(row["local_header_count"])
        except ValueError as exc:
            raise ContractError("date row has an invalid date or local_header_count") from exc
        if header_count < 0 or str(header_count) != row["local_header_count"]:
            raise ContractError("date row local_header_count must be canonical and nonnegative")
        payload: dict[str, Any] = {
            column: row[column] for column in DATE_COLUMNS if column != "date_row_sha256"
        }
        payload["visit_index"] = VISIT_TO_INDEX[row["visit"]]
        payload["local_header_count"] = header_count
        if row["date_row_sha256"] != sha256_json(payload):
            raise ContractError("date row SHA256 mismatch")
        normalized.append(row)
    _ensure_unique_identities(normalized, "date registry")
    normalized.sort(key=lambda row: (row["patient_id"], int(row["visit_index"])))
    previous: dict[str, date] = {}
    for row in normalized:
        current = date.fromisoformat(row["visit_date"])
        if row["patient_id"] in previous and current <= previous[row["patient_id"]]:
            raise ContractError("date registry patient dates must be strictly increasing")
        previous[row["patient_id"]] = current
    return normalized


def _positive_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{field} must be positive and finite") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ContractError(f"{field} must be positive and finite")
    return result


def _canonical_nonnegative_int(value: Any, field: str) -> int:
    text = _text(value)
    if not re.fullmatch(r"0|[1-9]\d*", text):
        raise ContractError(f"{field} must be a canonical integer")
    return int(text)


def _validate_physical_volume_fields(
    row: Mapping[str, str],
    *,
    require_positive_voxels: bool,
    validate_manifest_count: bool,
) -> int:
    voxel_count = _canonical_nonnegative_int(row["ftv_voxel_count"], "voxel count")
    if require_positive_voxels and voxel_count <= 0:
        raise ContractError("voxel count must be positive")
    if validate_manifest_count:
        manifest_count = _canonical_nonnegative_int(
            row["ftv_voxel_count_manifest"], "manifest voxel count"
        )
        if manifest_count != voxel_count:
            raise ContractError("manifest voxel count does not equal measured voxel count")

    row_spacing = _positive_float(row["row_spacing_mm"], "row_spacing_mm")
    column_spacing = _positive_float(row["column_spacing_mm"], "column_spacing_mm")
    slice_spacing = _positive_float(row["slice_spacing_mm"], "slice_spacing_mm")
    voxel_volume = _positive_float(
        row["ftv_voxel_volume_mm3"], "ftv_voxel_volume_mm3"
    )
    expected_voxel_volume = row_spacing * column_spacing * slice_spacing
    if not math.isclose(
        voxel_volume,
        expected_voxel_volume,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ContractError("voxel volume does not equal spacing product")
    if require_positive_voxels:
        ftv_volume = _positive_float(row["ftv_volume_cc"], "ftv_volume_cc")
        expected_ftv_volume = voxel_count * voxel_volume / 1000.0
        if not math.isclose(
            ftv_volume,
            expected_ftv_volume,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ContractError("FTV volume does not equal voxel count times voxel volume")
    return voxel_count


def _validate_volume_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for raw in rows:
        row = {column: _text(raw.get(column)) for column in VOLUME_COLUMNS}
        _validate_identity(row, "volume row")
        expected = sha256_json(
            {column: row[column] for column in VOLUME_COLUMNS if column != "volume_row_sha256"}
        )
        if row["volume_row_sha256"] != expected:
            raise ContractError("volume row SHA256 mismatch")
        if row["volume_status"] == "ok":
            if row["qc_status"] != "ok":
                raise ContractError("positive volume row must have qc_status=ok")
            _validate_physical_volume_fields(
                row,
                require_positive_voxels=True,
                validate_manifest_count=True,
            )
        elif row["volume_status"] == "excluded_empty_ftv_mask":
            if row["ftv_volume_cc"] or row["ftv_voxel_count"] != "0":
                raise ContractError("excluded empty volume must have zero voxels and no volume")
            _validate_physical_volume_fields(
                row,
                require_positive_voxels=False,
                validate_manifest_count=True,
            )
        else:
            raise ContractError("unsupported volume status")
        normalized.append(row)
    _ensure_unique_identities(normalized, "volume registry")
    normalized.sort(key=lambda row: (row["patient_id"], int(row["visit_index"])))
    return normalized


def _validate_qc_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for raw in rows:
        row = {column: _text(raw.get(column)) for column in QC_COLUMNS}
        _validate_identity(row, "QC row")
        if row["qc_status"] != "ok":
            raise ContractError("QC visits may only contain qc_status=ok rows")
        _validate_physical_volume_fields(
            row,
            require_positive_voxels=True,
            validate_manifest_count=False,
        )
        expected = sha256_json(
            {column: row[column] for column in QC_COLUMNS if column != "qc_visit_row_sha256"}
        )
        if row["qc_visit_row_sha256"] != expected:
            raise ContractError("QC visit row SHA256 mismatch")
        normalized.append(row)
    _ensure_unique_identities(normalized, "QC registry")
    normalized.sort(key=lambda row: (row["patient_id"], int(row["visit_index"])))
    return normalized


def _topology(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    by_patient: dict[str, set[str]] = {}
    for row in rows:
        by_patient.setdefault(row["patient_id"], set()).add(row["visit"])
    pairs = (("T0", "T1"), ("T1", "T2"), ("T2", "T3"), ("T0", "T2"), ("T0", "T3"))
    transitions = 0
    steps = 0
    for visits in by_patient.values():
        for source, target in pairs:
            source_index = VISIT_TO_INDEX[source]
            target_index = VISIT_TO_INDEX[target]
            required = {f"T{index}" for index in range(source_index, target_index + 1)}
            if required.issubset(visits):
                transitions += 1
                steps += target_index - source_index
    return {
        "date_comparison_count": sum(max(0, len(visits) - 1) for visits in by_patient.values()),
        "transition_count": transitions,
        "step_occurrence_count": steps,
    }


def _date_counts(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    qc = [row for row in rows if row["qc_status"] == "ok"]
    return {
        "visit_count": len(rows),
        "patient_count": len({row["patient_id"] for row in rows}),
        "qc_ok_visit_count": len(qc),
        "qc_ok_patient_count": len({row["patient_id"] for row in qc}),
        "visit_distribution": {
            visit: sum(row["visit"] == visit for row in rows) for visit in VISIT_TO_INDEX
        },
        "all_topology": _topology(rows),
        "qc_ok_topology": _topology(qc),
    }


def _volume_counts(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    return {
        "visit_count": len(rows),
        "patient_count": len({row["patient_id"] for row in rows}),
        "positive_volume_count": sum(row["volume_status"] == "ok" for row in rows),
        "excluded_empty_ftv_mask_count": sum(
            row["volume_status"] == "excluded_empty_ftv_mask" for row in rows
        ),
    }


def _qc_counts(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    return {
        "visit_count": len(rows),
        "patient_count": len({row["patient_id"] for row in rows}),
    }


def _validate_self_hash(contract: Mapping[str, Any], field: str, label: str) -> None:
    recorded = contract.get(field)
    expected = sha256_json({key: value for key, value in contract.items() if key != field})
    if recorded != expected:
        raise ContractError(f"{label} contract SHA256 mismatch")


def _validate_artifact(
    root: Path,
    artifact: Mapping[str, Any],
    expected_filename: str,
    expected_rows: int,
    label: str,
) -> Path:
    if artifact.get("path") != expected_filename:
        raise ContractError(f"{label} artifact path mismatch")
    path = root / expected_filename
    if artifact.get("sha256") != sha256_file(path):
        raise ContractError(f"{label} artifact SHA256 mismatch")
    if artifact.get("row_count") != expected_rows:
        raise ContractError(f"{label} artifact row count mismatch")
    return path


def _as_frame(rows: Sequence[Mapping[str, str]], kind: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["visit_index"] = frame["visit_index"].astype("int64")
    if kind == "date":
        frame["local_header_count"] = frame["local_header_count"].astype("int64")
    else:
        count_columns = [
            column
            for column in ("ftv_voxel_count_manifest", "ftv_voxel_count")
            if column in frame.columns
        ]
        for column in count_columns:
            frame[column] = frame[column].astype("int64")
        numeric = [
            column
            for column in (
                "row_spacing_mm",
                "column_spacing_mm",
                "slice_spacing_mm",
                "ftv_voxel_volume_mm3",
                "ftv_volume_cc",
            )
            if column in frame.columns
        ]
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_verified_registries(
    date_contract_dir: str | Path,
    volume_contract_dir: str | Path,
    qc_contract_dir: str | Path,
    manifest_csv: str | Path,
) -> VerifiedRegistries:
    """Load and fully validate the three immutable registry contracts."""

    date_root = Path(date_contract_dir).resolve()
    volume_root = Path(volume_contract_dir).resolve()
    qc_root = Path(qc_contract_dir).resolve()
    manifest_path = Path(manifest_csv).resolve()
    manifest_sha = sha256_file(manifest_path)

    date_contract = _load_json_object(date_root / DATE_CONTRACT_FILENAME, "date contract")
    volume_contract = _load_json_object(volume_root / VOLUME_CONTRACT_FILENAME, "volume contract")
    qc_contract = _load_json_object(qc_root / QC_CONTRACT_FILENAME, "QC contract")
    if date_contract.get("schema_version") != DATE_REGISTRY_SCHEMA:
        raise ContractError("unsupported date registry schema")
    if volume_contract.get("schema_version") != VOLUME_INDEX_SCHEMA:
        raise ContractError("unsupported volume index schema")
    if qc_contract.get("schema_version") != QC_VISITS_SCHEMA:
        raise ContractError("unsupported QC visits schema")
    _validate_self_hash(date_contract, "date_registry_sha256", "date registry")
    _validate_self_hash(volume_contract, "volume_contract_sha256", "volume")
    _validate_self_hash(qc_contract, "qc_visits_contract_sha256", "QC visits")

    manifest_hashes = {
        date_contract.get("manifest_sha256"),
        volume_contract.get("manifest_sha256"),
        qc_contract.get("manifest_sha256"),
    }
    if manifest_hashes != {manifest_sha}:
        raise ContractError("registry manifest SHA256 mismatch")
    if qc_contract.get("date_registry_sha256") != date_contract.get("date_registry_sha256"):
        raise ContractError("QC contract date registry SHA256 mismatch")
    if qc_contract.get("volume_contract_sha256") != volume_contract.get("volume_contract_sha256"):
        raise ContractError("QC contract volume SHA256 mismatch")

    date_rows = _validate_date_rows(_read_csv_exact(date_root / DATE_INDEX_FILENAME, DATE_COLUMNS))
    date_qc_rows = _validate_date_rows(_read_csv_exact(date_root / DATE_QC_FILENAME, DATE_COLUMNS))
    volume_rows = _validate_volume_rows(
        _read_csv_exact(volume_root / VOLUME_INDEX_FILENAME, VOLUME_COLUMNS)
    )
    qc_rows = _validate_qc_rows(_read_csv_exact(qc_root / QC_VISITS_FILENAME, QC_COLUMNS))

    _validate_artifact(
        date_root,
        date_contract["artifacts"]["date_index_csv"],
        DATE_INDEX_FILENAME,
        len(date_rows),
        "date index",
    )
    _validate_artifact(
        date_root,
        date_contract["artifacts"]["qc_ok_date_index_csv"],
        DATE_QC_FILENAME,
        len(date_qc_rows),
        "date QC view",
    )
    _validate_artifact(
        volume_root,
        volume_contract["artifact"],
        VOLUME_INDEX_FILENAME,
        len(volume_rows),
        "volume index",
    )
    _validate_artifact(
        qc_root,
        qc_contract["artifact"],
        QC_VISITS_FILENAME,
        len(qc_rows),
        "QC visits",
    )
    expected_date_qc = [row for row in date_rows if row["qc_status"] == "ok"]
    if date_qc_rows != expected_date_qc:
        raise ContractError("date QC view does not match date index")
    if date_contract.get("counts") != _date_counts(date_rows):
        raise ContractError("date registry counts mismatch")
    if volume_contract.get("counts") != _volume_counts(volume_rows):
        raise ContractError("volume index counts mismatch")
    if qc_contract.get("counts") != _qc_counts(qc_rows):
        raise ContractError("QC visits counts mismatch")

    date_keys = {tuple(row[column] for column in IDENTITY_COLUMNS) for row in date_rows}
    volume_keys = {tuple(row[column] for column in IDENTITY_COLUMNS) for row in volume_rows}
    qc_keys = {tuple(row[column] for column in IDENTITY_COLUMNS) for row in qc_rows}
    expected_qc_keys = {
        tuple(row[column] for column in IDENTITY_COLUMNS)
        for row in volume_rows
        if row["qc_status"] == "ok" and row["volume_status"] == "ok"
    }
    if date_keys != volume_keys or qc_keys != expected_qc_keys:
        raise ContractError("registry identity coverage mismatch")

    return VerifiedRegistries(
        date=_as_frame(date_rows, "date"),
        volume=_as_frame(volume_rows, "volume"),
        qc=_as_frame(qc_rows, "qc"),
        contracts={"date": date_contract, "volume": volume_contract, "qc": qc_contract},
    )


def _resolve_source_path(
    value: Any,
    field: str,
    processed_root: Path | None,
    require_files: bool,
) -> Path:
    text = _required_text(value, f"manifest {field}")
    path = Path(text).expanduser()
    if not path.is_absolute():
        if processed_root is None:
            raise ContractError(f"relative manifest {field} requires processed_root")
        path = processed_root / path
    path = path.resolve()
    if processed_root is not None:
        try:
            path.relative_to(processed_root)
        except ValueError as exc:
            raise ContractError(f"manifest {field} resolves outside processed_root") from exc
    if require_files and not path.is_file():
        raise ContractError(f"manifest {field} does not exist: {path}")
    return path


def _parse_dce_paths(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [_text(item) for item in value if _text(item)]
    text = _text(value)
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ContractError("manifest dce_paths must be valid JSON or semicolon-separated") from exc
        if not isinstance(parsed, list):
            raise ContractError("manifest dce_paths JSON must be a list")
        return [_text(item) for item in parsed if _text(item)]
    return [_text(item) for item in text.split(";") if _text(item)]


def _parse_path_mapping(value: Any, field: str) -> dict[str, str]:
    if isinstance(value, Mapping):
        return {str(key): _text(item) for key, item in value.items() if _text(item)}
    text = _text(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ContractError(f"manifest {field} must contain valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ContractError(f"manifest {field} must contain a JSON object")
    return {str(key): _text(item) for key, item in parsed.items() if _text(item)}


def load_image_manifest(
    manifest_csv: str | Path,
    *,
    processed_root: str | Path | None = None,
    require_files: bool = True,
) -> pd.DataFrame:
    """Resolve DCE0/SER paths and enrich manifest rows with exact meta identities."""

    root = Path(processed_root).resolve() if processed_root is not None else None
    manifest = pd.read_csv(manifest_csv, dtype=str, keep_default_na=False)
    required = {"patient_id", "visit", "mask_path", "meta_path"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ContractError(f"image manifest is missing columns: {missing}")

    rows: list[dict[str, Any]] = []
    for source in manifest.to_dict(orient="records"):
        patient_id = normalize_patient_id(source.get("patient_id"))
        visit = _required_text(source.get("visit"), "manifest visit").upper()
        if not patient_id or visit not in VISIT_TO_INDEX:
            raise ContractError("manifest patient_id/visit is invalid")

        dce_candidates = _parse_dce_paths(source.get("dce_paths"))
        direct_dce = _text(source.get("dce0_path"))
        if direct_dce and dce_candidates and direct_dce != dce_candidates[0]:
            raise ContractError("manifest dce0_path conflicts with first dce_paths entry")
        dce0 = direct_dce or (dce_candidates[0] if dce_candidates else "")

        derived = _parse_path_mapping(
            source.get("derived_paths_json") or source.get("derived_paths"),
            "derived_paths_json",
        )
        direct_ser = _text(source.get("ser_path"))
        if direct_ser and derived.get("ser") and direct_ser != derived["ser"]:
            raise ContractError("manifest ser_path conflicts with derived SER path")
        ser = direct_ser or derived.get("ser", "")

        paths = {
            "dce0_path": _resolve_source_path(dce0, "dce0_path", root, require_files),
            "ser_path": _resolve_source_path(ser, "ser_path", root, require_files),
            "mask_path": _resolve_source_path(source.get("mask_path"), "mask_path", root, require_files),
            "meta_path": _resolve_source_path(source.get("meta_path"), "meta_path", root, True),
        }
        meta = _load_json_object(paths["meta_path"], "manifest meta")
        meta_patient = normalize_patient_id(meta.get("patient_id")) if meta.get("patient_id") is not None else ""
        meta_visit = _text(meta.get("visit")).upper()
        if meta_patient and meta_patient != patient_id:
            raise ContractError("manifest patient_id conflicts with meta patient_id")
        if meta_visit and meta_visit != visit:
            raise ContractError("manifest visit conflicts with meta visit")
        uid_values = {
            _text(source.get(field))
            for field in ("study_instance_uid", "study_uid", "StudyInstanceUID")
            if _text(source.get(field))
        }
        meta_uid = _text(meta.get("study_uid") or meta.get("study_instance_uid"))
        if meta_uid:
            uid_values.add(meta_uid)
        if len(uid_values) != 1:
            raise ContractError("manifest study UID sources are missing or conflict")
        uid = next(iter(uid_values))
        rows.append(
            {
                "patient_id": patient_id,
                "visit_id": f"{patient_id}:{visit}",
                "visit": visit,
                "visit_index": VISIT_TO_INDEX[visit],
                "study_instance_uid": uid,
                **{name: str(path) for name, path in paths.items()},
                "manifest_qc_status": _text(source.get("qc_status")),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ContractError("image manifest contains no rows")
    identity_records = [
        {column: str(row[column]) for column in IDENTITY_COLUMNS}
        for row in frame.to_dict(orient="records")
    ]
    _ensure_unique_identities(identity_records, "image manifest")
    return frame.sort_values(["patient_id", "visit_index"]).reset_index(drop=True)


def _identity_key(row: Mapping[str, Any]) -> tuple[str, str, str, int, str]:
    return (
        normalize_patient_id(row["patient_id"]),
        str(row["visit_id"]),
        str(row["visit"]).upper(),
        int(row["visit_index"]),
        str(row["study_instance_uid"]),
    )


def _frame_map(frame: pd.DataFrame, label: str) -> dict[tuple[str, str, str, int, str], dict[str, Any]]:
    rows = frame.to_dict(orient="records")
    mapping = {_identity_key(row): row for row in rows}
    if len(mapping) != len(rows):
        raise ContractError(f"{label} contains duplicate composite identities")
    return mapping


def _same_path(left: Any, right: Any) -> bool:
    return Path(str(left)).expanduser().resolve() == Path(str(right)).expanduser().resolve()


def _same_number(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def join_registry_sources(
    date_frame: pd.DataFrame,
    volume_frame: pd.DataFrame,
    qc_frame: pd.DataFrame,
    image_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Join all sources exclusively by the full composite visit identity."""

    dates = _frame_map(date_frame, "date registry")
    volumes = _frame_map(volume_frame, "volume registry")
    qc = _frame_map(qc_frame, "QC registry")
    images = _frame_map(image_frame, "image manifest")
    if dates.keys() != volumes.keys() or dates.keys() != images.keys():
        raise ContractError("date, volume, and image identity coverage mismatch")
    expected_qc = {
        key
        for key, row in volumes.items()
        if _text(row.get("qc_status")) == "ok" and _text(row.get("volume_status")) == "ok"
    }
    if qc.keys() != expected_qc:
        raise ContractError("QC registry identity coverage mismatch")

    output: list[dict[str, Any]] = []
    for key in sorted(qc, key=lambda item: (item[0], item[3])):
        date_row = dates[key]
        volume_row = volumes[key]
        qc_row = qc[key]
        image_row = images[key]
        if (
            str(date_row["visit_date"]) != str(qc_row["visit_date"])
            or str(date_row["visit_date_source"]) != str(qc_row["visit_date_source"])
            or _text(date_row["qc_status"]) != "ok"
            or _text(volume_row["qc_status"]) != "ok"
        ):
            raise ContractError("joined date/QC fields conflict")
        for field in ("mask_path", "meta_path"):
            if not _same_path(volume_row[field], qc_row[field]) or not _same_path(
                volume_row[field], image_row[field]
            ):
                raise ContractError(f"joined {field} values conflict")
        for field in (
            "ftv_voxel_count",
            "row_spacing_mm",
            "column_spacing_mm",
            "slice_spacing_mm",
            "ftv_voxel_volume_mm3",
            "ftv_volume_cc",
        ):
            if not _same_number(volume_row[field], qc_row[field]):
                raise ContractError(f"joined {field} values conflict")
        output.append(
            {
                "patient_id": key[0],
                "visit_id": key[1],
                "visit": key[2],
                "visit_index": key[3],
                "study_instance_uid": key[4],
                "visit_date": str(date_row["visit_date"]),
                "visit_date_source": str(date_row["visit_date_source"]),
                "qc_status": "ok",
                "dce0_path": str(image_row["dce0_path"]),
                "ser_path": str(image_row["ser_path"]),
                "mask_path": str(image_row["mask_path"]),
                "meta_path": str(image_row["meta_path"]),
                "ftv_voxel_count": int(volume_row["ftv_voxel_count"]),
                "row_spacing_mm": float(volume_row["row_spacing_mm"]),
                "column_spacing_mm": float(volume_row["column_spacing_mm"]),
                "slice_spacing_mm": float(volume_row["slice_spacing_mm"]),
                "ftv_voxel_volume_mm3": float(volume_row["ftv_voxel_volume_mm3"]),
                "ftv_volume_cc": float(volume_row["ftv_volume_cc"]),
            }
        )
    return pd.DataFrame(output)


def load_clinical_table(path: str | Path) -> pd.DataFrame:
    """Load a clinical CSV/XLSX and add canonical patient join keys."""

    source = Path(path)
    if source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    elif source.suffix.lower() == ".xlsx":
        frame = pd.read_excel(source)
    else:
        raise ContractError("clinical input must be CSV or XLSX")
    missing = sorted(set(CLINICAL_COLUMNS).difference(frame.columns))
    if missing:
        raise ContractError(f"clinical input is missing columns: {missing}")
    output = frame.copy()
    output["patient_key"] = output["Patient_ID"].map(normalize_patient_key)
    output["patient_id"] = output["Patient_ID"].map(normalize_patient_id)
    if output["patient_key"].eq("").any():
        raise ContractError("clinical patient IDs must be non-empty")
    if output["patient_key"].duplicated().any():
        raise ContractError("clinical patient IDs must be unique after normalization")
    return output


def _display_value(value: Any) -> str:
    try:
        if pd.isna(value):
            return "unknown"
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text if text else "unknown"


def _tokenize_balance_value(value: Any) -> str:
    displayed = _display_value(value)
    return "missing" if displayed == "unknown" else displayed


def _allocate_split_counts(
    total: int, fractions: Sequence[float]
) -> dict[str, int]:
    folds = ("train", "val", "test")
    raw = [total * fraction for fraction in fractions]
    counts = [int(np.floor(value)) for value in raw]
    remaining = total - sum(counts)
    order = sorted(
        range(len(fractions)),
        key=lambda index: (raw[index] - counts[index], fractions[index]),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1
    return dict(zip(folds, counts))


def _make_patient_split_v3(joined: pd.DataFrame, seed: int) -> dict[str, Any]:
    folds = ("train", "val", "test")
    train_frac = 0.7
    val_frac = 0.15
    fractions = (train_frac, val_frac, 1.0 - train_frac - val_frac)
    balance_columns = ["pCR", "HR", "HER2", "MP", "trial_arm", "visit_pattern"]
    patient_rows: list[dict[str, str]] = []
    conflicts: dict[str, list[str]] = {}
    for patient_id, group in joined.groupby("patient_id", sort=True):
        visits = sorted(
            {str(value).upper() for value in group["visit"]},
            key=lambda value: VISIT_TO_INDEX[value],
        )
        row = {"patient_id": str(patient_id), "visit_pattern": "+".join(visits)}
        for column in balance_columns[:-1]:
            values = [_tokenize_balance_value(value) for value in group[column].tolist()]
            non_missing = [value for value in values if value != "missing"]
            counts = Counter(non_missing)
            if counts:
                maximum = max(counts.values())
                row[column] = sorted(
                    value for value, count in counts.items() if count == maximum
                )[0]
                if len(counts) > 1:
                    conflicts.setdefault(str(patient_id), []).append(column)
            else:
                row[column] = "missing"
        patient_rows.append(row)
    if not patient_rows:
        raise ValueError("cannot split an empty visit cohort")

    target_counts = _allocate_split_counts(len(patient_rows), fractions)
    feature_totals = Counter(
        f"{column}={row[column]}" for row in patient_rows for column in balance_columns
    )
    rng = np.random.default_rng(seed)
    jitter = {row["patient_id"]: float(rng.random()) for row in patient_rows}
    ordered = sorted(
        patient_rows,
        key=lambda row: (
            -sum(
                1.0 / feature_totals[f"{column}={row[column]}"]
                for column in balance_columns
            ),
            jitter[row["patient_id"]],
            row["patient_id"],
        ),
    )
    split: dict[str, Any] = {fold: [] for fold in folds}
    assigned = {fold: Counter() for fold in folds}
    for row in ordered:
        features = [f"{column}={row[column]}" for column in balance_columns]
        candidates = [fold for fold in folds if len(split[fold]) < target_counts[fold]]

        def assignment_cost(fold: str) -> tuple[float, int, str]:
            fraction = fractions[folds.index(fold)]
            feature_cost = 0.0
            for feature in features:
                target = feature_totals[feature] * fraction
                current = assigned[fold][feature]
                feature_cost += (
                    (current + 1 - target) ** 2 - (current - target) ** 2
                ) / max(target, 1.0)
            capacity_cost = (len(split[fold]) + 1) / max(target_counts[fold], 1)
            return feature_cost + 0.05 * capacity_cost, len(split[fold]), fold

        selected = min(candidates, key=assignment_cost)
        split[selected].append(row["patient_id"])
        assigned[selected].update(features)
    for fold in folds:
        split[fold] = sorted(split[fold])
    all_patients = [patient for fold in folds for patient in split[fold]]
    expected_patients = {row["patient_id"] for row in patient_rows}
    if len(all_patients) != len(set(all_patients)) or set(all_patients) != expected_patients:
        raise RuntimeError("patient split is not an exact disjoint partition")
    split["schema_version"] = "motfm_ispy2_patient_split_v1"
    split["metadata"] = {
        "schema_version": "motfm_ispy2_patient_split_v1",
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": fractions[2],
        "n_patients": len(patient_rows),
        "fold_patients": {fold: len(split[fold]) for fold in folds},
        "balance_columns": balance_columns,
        "patient_field_conflicts": conflicts,
        "visit_cohort_sha256": sha256_json(
            sorted(patient_rows, key=lambda row: row["patient_id"])
        ),
        "pcr_policy": "split_balancing_only_not_published_in_model_records",
    }
    return split


def build_split_and_visit_records(
    visits: pd.DataFrame,
    clinical: pd.DataFrame,
    *,
    seed: int = 2026,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join clinical data transiently, split patients, and publish leakage-safe records."""

    if isinstance(seed, bool) or int(seed) != 2026:
        raise ValueError("I-SPY2 split seed must be exactly 2026")
    required_visits = {
        "patient_id",
        "visit_id",
        "visit",
        "visit_index",
        "study_instance_uid",
        "visit_date",
        "ftv_volume_cc",
    }
    missing_visits = sorted(required_visits.difference(visits.columns))
    if missing_visits:
        raise ValueError(f"visits are missing split fields: {missing_visits}")
    missing_clinical = sorted(set(CLINICAL_COLUMNS).difference(clinical.columns))
    if missing_clinical:
        raise ValueError(f"clinical data is missing fields: {missing_clinical}")

    clinical_rows = clinical.copy()
    clinical_rows["_patient_id"] = clinical_rows["Patient_ID"].map(normalize_patient_id)
    if clinical_rows["_patient_id"].eq("").any() or clinical_rows["_patient_id"].duplicated().any():
        raise ValueError("clinical patient IDs must be non-empty and unique")
    frame = visits.copy()
    frame["patient_id"] = frame["patient_id"].map(normalize_patient_id)
    joined = frame.merge(
        clinical_rows[list(CLINICAL_COLUMNS) + ["_patient_id"]],
        left_on="patient_id",
        right_on="_patient_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        missing_count = int(
            joined.loc[joined["_merge"] != "both", "patient_id"].nunique()
        )
        raise ValueError(
            f"clinical join incomplete: {missing_count} patient(s) missing"
        )
    joined["trial_arm"] = joined["Arm"].map(_display_value)
    split = _make_patient_split_v3(joined, int(seed))
    fold_by_patient = {
        patient: fold for fold in ("train", "val", "test") for patient in split[fold]
    }

    original_columns = [
        column
        for column in visits.columns
        if column not in {"pCR", "Patient_ID", "Arm", "patient_key", "_patient_id"}
    ]
    records = joined[original_columns].copy()
    for column in ("HR", "HER2", "MP", "Age_at_Screening", "menopausal_status"):
        records[column] = joined[column]
    records["trial_arm"] = joined["trial_arm"]
    records["clinical_text"] = joined.apply(
        lambda row: "; ".join(
            (
                f"age at screening {_display_value(row['Age_at_Screening'])}",
                f"HR {_display_value(row['HR'])}",
                f"HER2 {_display_value(row['HER2'])}",
                f"MP {_display_value(row['MP'])}",
                f"menopausal status {_display_value(row['menopausal_status'])}",
            )
        ),
        axis=1,
    )
    records["fold"] = records["patient_id"].map(fold_by_patient)
    records = records.sort_values(["patient_id", "visit_index"]).reset_index(drop=True)
    return records, split

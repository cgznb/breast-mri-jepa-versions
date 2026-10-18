from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


def _rows(path: Path | str) -> list[dict[str, str]]:
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _manifest_paths(row: Mapping[str, str]) -> tuple[Path, ...]:
    values: list[str] = []
    for name in (
        "dce_paths",
        "mask_path",
        "raw_mask_path",
        "ser_path",
        "adc_path",
        "dwi_mask_path",
        "dwi_paths",
        "meta_path",
        "transform_path",
    ):
        value = row.get(name, "")
        if name in {"dce_paths", "dwi_paths"}:
            values.extend(item for item in value.split(";") if item)
        elif value:
            values.append(value)
    pe_value = row.get("pe_paths_json", "")
    if pe_value:
        payload = json.loads(pe_value)
        values.extend(str(value) for value in payload.values() if value)
    return tuple(Path(value) for value in values)


def _atomic_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return path


def generate_final_report(
    registered_manifest: Path | str,
    uid_status_paths: Iterable[Path | str],
    failure_paths: Iterable[Path | str],
    output_root: Path | str,
) -> dict[str, Any]:
    registered = _rows(registered_manifest)
    missing_paths: list[dict[str, str]] = []
    for row in registered:
        for path in _manifest_paths(row):
            if not path.is_file():
                missing_paths.append(
                    {
                        "patient_id": row.get("patient_id", ""),
                        "visit": row.get("visit", ""),
                        "path": str(path),
                    }
                )
    uid_counts: Counter[str] = Counter()
    invalid_uid_states: list[dict[str, str]] = []
    for status_path in uid_status_paths:
        for row in _rows(status_path):
            status = row.get("status", "")
            uid_counts[status] += 1
            if status not in {"present", "excluded_after_retry"}:
                invalid_uid_states.append(row)
    failures: list[dict[str, str]] = []
    for failure_path in failure_paths:
        path = Path(failure_path)
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            failures.extend(dict(value) for value in payload.get("failures", ()))
        else:
            failures.extend(_rows(path))
    report = {
        "registered_manifest": str(Path(registered_manifest)),
        "registered_rows": len(registered),
        "registered_patients": len({row.get("patient_id", "") for row in registered}),
        "rows_by_collection": dict(Counter(row.get("collection", "") for row in registered)),
        "registration_status_counts": dict(
            Counter(row.get("registration_status", "") for row in registered)
        ),
        "dwi_registration_status_counts": dict(
            Counter(row.get("dwi_registration_status", "") for row in registered)
        ),
        "uid_status_counts": dict(uid_counts),
        "invalid_uid_states": invalid_uid_states,
        "missing_manifest_paths": missing_paths,
        "failure_count": len(failures),
        "failures": failures,
    }
    report["acceptance_status"] = (
        "passed"
        if not invalid_uid_states and not missing_paths
        else "failed"
    )
    destination_root = Path(output_root)
    json_path = destination_root / "manifest" / "final_coverage_qc_report.json"
    markdown_path = destination_root / "manifest" / "final_coverage_qc_report.md"
    _atomic_text(json_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# ISPY2 and ACRIN T0 Registration Report",
        "",
        f"- Acceptance: `{report['acceptance_status']}`",
        f"- Registered patients: {report['registered_patients']}",
        f"- Registered manifest rows: {report['registered_rows']}",
        f"- Missing manifest paths: {len(missing_paths)}",
        f"- Invalid UID states: {len(invalid_uid_states)}",
        f"- Traced failures: {len(failures)}",
        "",
        "## Registration Status",
        "",
        "```json",
        json.dumps(report["registration_status_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "## DWI Registration Status",
        "",
        "```json",
        json.dumps(report["dwi_registration_status_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "## UID Status",
        "",
        "```json",
        json.dumps(report["uid_status_counts"], indent=2, sort_keys=True),
        "```",
        "",
    ]
    _atomic_text(markdown_path, "\n".join(lines))
    report["json_path"] = str(json_path)
    report["markdown_path"] = str(markdown_path)
    return report

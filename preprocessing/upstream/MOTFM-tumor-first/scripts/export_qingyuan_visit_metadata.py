#!/usr/bin/env python3
"""Export Study UID and StudyDate for terminal Qingyuan registered visits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO

import pydicom


TERMINAL_STATES = {"fixed_reference", "deformable", "rigid_fallback"}
FIELDS = (
    "source_patient_id",
    "patient_id",
    "visit",
    "Study UID",
    "Study Date",
    "Series UID",
)


def _canonical_patient_id(value: str) -> str:
    suffix = str(value).strip().rsplit("-", 1)[-1]
    if not suffix.isdigit():
        raise ValueError(f"invalid Qingyuan patient ID: {value}")
    return f"ISPY2-{suffix}"


def _first_dicom(series_dir: Path) -> Path:
    try:
        return next(path for path in sorted(series_dir.iterdir()) if path.is_file())
    except (OSError, StopIteration):
        raise ValueError(f"DCE source series contains no DICOM files: {series_dir}") from None


def collect_visit_metadata(registered_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for meta_path in sorted(registered_root.glob("*/T*/meta/*.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ValueError(f"registered metadata is unreadable: {meta_path}") from None
        status = str(meta.get("registration", {}).get("status", ""))
        if status not in TERMINAL_STATES:
            continue
        patient_id = str(meta.get("patient_id", ""))
        visit = str(meta.get("visit", ""))
        dce_dirs = meta.get("source_series_dirs", {}).get("dce", [])
        dce_uids = meta.get("source_series_uids", {}).get("dce", [])
        if len(dce_dirs) != 1 or len(dce_uids) != 1:
            raise ValueError(f"DCE source identity is ambiguous: {patient_id}:{visit}")
        dicom_path = _first_dicom(Path(str(dce_dirs[0])))
        dataset = pydicom.dcmread(
            str(dicom_path),
            stop_before_pixels=True,
            specific_tags=["StudyInstanceUID", "StudyDate", "SeriesInstanceUID"],
        )
        study_uid = str(getattr(dataset, "StudyInstanceUID", "")).strip()
        study_date = str(getattr(dataset, "StudyDate", "")).strip()
        series_uid = str(getattr(dataset, "SeriesInstanceUID", "")).strip()
        if not study_uid or not study_date or not series_uid:
            raise ValueError(f"DICOM study identity is incomplete: {patient_id}:{visit}")
        if series_uid != str(dce_uids[0]):
            raise ValueError(f"DCE Series UID conflicts with metadata: {patient_id}:{visit}")
        parsed_date = datetime.strptime(study_date, "%Y%m%d").strftime("%m-%d-%Y")
        rows.append(
            {
                "source_patient_id": patient_id,
                "patient_id": _canonical_patient_id(patient_id),
                "visit": visit,
                "Study UID": study_uid,
                "Study Date": parsed_date,
                "Series UID": series_uid,
            }
        )
    identities = [(row["patient_id"], row["visit"]) for row in rows]
    if not rows or len(identities) != len(set(identities)):
        raise ValueError("exported Qingyuan visit identities are empty or duplicated")
    return rows


def _write_csv(handle: TextIO, rows: list[dict[str, str]]) -> None:
    writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registered-root",
        type=Path,
        default=Path("/data1/cxx/ispy2_acrin_t0_registration_v1/registered"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    rows = collect_visit_metadata(args.registered_root.expanduser().resolve())
    if args.output is None:
        _write_csv(sys.stdout, rows)
    else:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            _write_csv(handle, rows)
        temporary.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

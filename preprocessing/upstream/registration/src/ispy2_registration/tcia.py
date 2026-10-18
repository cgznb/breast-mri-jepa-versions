from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pydicom


TCIA_SERIES_MARKER = "ListOfSeriesToDownload="


@dataclass(frozen=True)
class TciaManifest:
    header_lines: tuple[str, ...]
    series_uids: tuple[str, ...]


def read_tcia_manifest(path: Path | str) -> TciaManifest:
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read TCIA manifest {source}: {exc}") from exc
    try:
        marker_index = lines.index(TCIA_SERIES_MARKER)
    except ValueError as exc:
        raise ValueError(f"invalid TCIA manifest {source}: missing series marker") from exc
    header = tuple(lines[: marker_index + 1])
    uids = tuple(line.strip() for line in lines[marker_index + 1 :] if line.strip())
    if not uids or len(set(uids)) != len(uids):
        raise ValueError(f"invalid TCIA manifest {source}: empty or duplicate series UIDs")
    return TciaManifest(header_lines=header, series_uids=uids)


def write_tcia_manifest(
    path: Path | str,
    manifest: TciaManifest,
    series_uids: Iterable[str],
) -> Path:
    destination = Path(path)
    values = tuple(str(value).strip() for value in series_uids if str(value).strip())
    if len(set(values)) != len(values):
        raise ValueError("repair manifest contains duplicate SeriesInstanceUID values")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write("\n".join((*manifest.header_lines, *values)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination


def _dicom_files(directory: Path) -> list[Path]:
    dcm = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".dcm")
    return dcm


def _expected_md5(directory: Path) -> dict[str, str]:
    path = directory / "md5hashes.csv"
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            first, second = row[0].strip(), row[1].strip()
            if len(first) == 32:
                values[Path(second).name] = first.lower()
            elif len(second) == 32:
                values[Path(first).name] = second.lower()
    return values


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_dicom_series(
    dicom_root: Path | str | Sequence[Path | str],
    *,
    deep_headers: bool = False,
    verify_md5: bool = False,
) -> list[dict[str, object]]:
    roots = (
        (Path(dicom_root),)
        if isinstance(dicom_root, (str, Path))
        else tuple(Path(value) for value in dicom_root)
    )
    if not roots or any(not root.is_dir() for root in roots):
        raise ValueError(f"one or more DICOM roots do not exist: {roots}")
    rows: list[dict[str, object]] = []
    for root in roots:
        for directory, _, filenames in os.walk(root):
            if not any(name.lower().endswith(".dcm") for name in filenames):
                continue
            series_dir = Path(directory)
            files = _dicom_files(series_dir)
            errors: list[str] = []
            datasets = []
            selected = files if deep_headers else files[:1]
            for path in selected:
                try:
                    datasets.append(
                        pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
                    )
                except Exception as exc:
                    errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            first = datasets[0] if datasets else None
            uid = str(getattr(first, "SeriesInstanceUID", "")) if first is not None else ""
            if deep_headers and uid:
                seen = {str(getattr(dataset, "SeriesInstanceUID", "")) for dataset in datasets}
                if seen != {uid}:
                    errors.append(f"mixed SeriesInstanceUID values: {sorted(seen)}")
            expected = _expected_md5(series_dir)
            md5_missing = sorted(set(expected) - {path.name for path in files})
            md5_mismatch: list[str] = []
            if verify_md5:
                for path in files:
                    if path.name in expected and _md5(path) != expected[path.name]:
                        md5_mismatch.append(path.name)
            rows.append({
                "series_instance_uid": uid,
                "patient_id": str(getattr(first, "PatientID", "")) if first else "",
                "study_instance_uid": str(getattr(first, "StudyInstanceUID", "")) if first else "",
                "series_description": str(getattr(first, "SeriesDescription", "")) if first else "",
                "series_dir": str(series_dir),
                "dicom_file_count": len(files),
                "header_status": "readable" if not errors else "error",
                "header_errors": " | ".join(errors),
                "md5_manifest_status": "present" if expected else "missing",
                "md5_expected_count": len(expected),
                "md5_missing_files": ";".join(md5_missing),
                "md5_mismatches": ";".join(md5_mismatch),
            })
    return sorted(rows, key=lambda row: (str(row["series_instance_uid"]), str(row["series_dir"])))


def write_inventory_csv(path: Path | str, rows: list[dict[str, object]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "series_instance_uid",
        "patient_id",
        "study_instance_uid",
        "series_description",
        "series_dir",
        "dicom_file_count",
        "header_status",
        "header_errors",
        "md5_manifest_status",
        "md5_expected_count",
        "md5_missing_files",
        "md5_mismatches",
    ]
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination


def missing_series_uids(manifest: TciaManifest, rows: Iterable[dict[str, object]]) -> tuple[str, ...]:
    present = {
        str(row.get("series_instance_uid", ""))
        for row in rows
        if row.get("header_status") == "readable"
    }
    return tuple(uid for uid in manifest.series_uids if uid not in present)


def write_uid_status_csv(
    path: Path | str,
    manifest: TciaManifest,
    rows: Iterable[dict[str, object]],
    *,
    mark_missing_excluded: bool,
) -> Path:
    by_uid = {
        str(row.get("series_instance_uid", "")): row
        for row in rows
        if row.get("series_instance_uid")
    }
    status_rows = []
    for uid in manifest.series_uids:
        row = by_uid.get(uid)
        present = row is not None and row.get("header_status") == "readable"
        status_rows.append(
            {
                "series_instance_uid": uid,
                "status": (
                    "present"
                    if present
                    else "excluded_after_retry"
                    if mark_missing_excluded
                    else "missing_pending_retry"
                ),
                "series_dir": str(row.get("series_dir", "")) if row else "",
                "dicom_file_count": row.get("dicom_file_count", "") if row else "",
                "header_status": row.get("header_status", "missing") if row else "missing",
                "exclusion_reason": "unavailable_after_UID_retry"
                if mark_missing_excluded and not present
                else "",
            }
        )
    return write_inventory_csv(path, status_rows)

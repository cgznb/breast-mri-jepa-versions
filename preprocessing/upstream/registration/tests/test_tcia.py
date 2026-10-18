from __future__ import annotations

from pathlib import Path

from ispy2_registration.tcia import (
    TciaManifest,
    missing_series_uids,
    read_tcia_manifest,
    write_tcia_manifest,
)


def test_tcia_repair_manifest_preserves_header_and_expected_order(tmp_path: Path) -> None:
    source = tmp_path / "source.tcia"
    source.write_text(
        "downloadServerUrl=https://example.invalid\n"
        "manifestVersion=3.0\n"
        "ListOfSeriesToDownload=\n"
        "1.2.3\n"
        "1.2.4\n"
        "1.2.5\n",
        encoding="utf-8",
    )
    manifest = read_tcia_manifest(source)
    rows = [
        {"series_instance_uid": "1.2.4", "header_status": "readable"},
        {"series_instance_uid": "1.2.3", "header_status": "readable"},
    ]

    missing = missing_series_uids(manifest, rows)
    destination = write_tcia_manifest(tmp_path / "repair.tcia", manifest, missing)

    assert missing == ("1.2.5",)
    assert destination.read_text(encoding="utf-8").endswith(
        "ListOfSeriesToDownload=\n1.2.5\n"
    )


def test_tcia_manifest_rejects_duplicate_uids(tmp_path: Path) -> None:
    source = tmp_path / "bad.tcia"
    source.write_text("ListOfSeriesToDownload=\n1.2.3\n1.2.3\n", encoding="utf-8")

    try:
        read_tcia_manifest(source)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate SeriesInstanceUIDs were accepted")

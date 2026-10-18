from __future__ import annotations

import csv
import json
from pathlib import Path

from ispy2_registration.report import generate_final_report


def _csv(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_final_report_requires_terminal_uid_states_and_existing_manifest_paths(
    tmp_path: Path,
) -> None:
    image = tmp_path / "registered.nii.gz"
    image.write_bytes(b"test")
    registered = _csv(
        tmp_path / "registered.csv",
        [
            {
                "patient_id": "P1",
                "visit": "T0",
                "collection": "ACRIN-6698",
                "dce_paths": str(image),
                "mask_path": "",
                "raw_mask_path": "",
                "ser_path": "",
                "adc_path": "",
                "dwi_mask_path": "",
                "dwi_paths": "",
                "meta_path": "",
                "transform_path": "",
                "pe_paths_json": "{}",
                "registration_status": "fixed_reference",
                "dwi_registration_status": "not_available",
            }
        ],
    )
    uid_status = _csv(
        tmp_path / "uids.csv",
        [
            {"series_instance_uid": "1.2.3", "status": "present"},
            {"series_instance_uid": "1.2.4", "status": "excluded_after_retry"},
        ],
    )
    failures = tmp_path / "failures.json"
    failures.write_text(json.dumps({"failures": [], "count": 0}), encoding="utf-8")

    report = generate_final_report(registered, [uid_status], [failures], tmp_path)

    assert report["acceptance_status"] == "passed"
    assert report["uid_status_counts"] == {"present": 1, "excluded_after_retry": 1}
    assert Path(report["json_path"]).is_file()
    assert Path(report["markdown_path"]).is_file()

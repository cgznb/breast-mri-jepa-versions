"""Restore missing registered phases after checking the existing phase anchors."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mewm_ispy2.registered_roi32_data import check_disk, file_identity, read_json, verify_identity, write_json
from mewm_ispy2.registered_three_phase_data import load_config

REMOTE = "/data1/cxx/organized_v1/03_registered_nifti"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def relative(path):
    parts = Path(path).parts[-4:]
    value = "/".join(parts)
    if not re.fullmatch(r"[A-Za-z0-9_-]+/T[0-3]/dce/[A-Za-z0-9_-]+_aqc_[0-9]+\.nii\.gz", value):
        raise ValueError("Unexpected registered image path")
    return value


def main():
    config, _ = load_config(REPO / "configs/registered_three_phase_roi32_v1.yaml")
    root = Path(config["output_dir"])
    manifest = read_json(root / "inventory.json")
    if not manifest["missing_sources"]:
        print(json.dumps({"stage": "already_complete", "missing_files": 0}), flush=True)
        return
    visits = {row["visit_id"]: row for row in manifest["visits"]}
    requested, anchors = {}, {}
    for missing in manifest["missing_sources"]:
        requested[relative(missing["path"])] = missing
        row = visits[missing["visit_id"]]
        for source in row["phase_sources"]:
            if source is not None:
                verify_identity(source)
                anchors[relative(source["path"])] = source
    paths = sorted(set(requested) | set(anchors))
    # Content signatures are transient compatibility checks, never report fields.
    program = """import hashlib,json,pathlib
root=pathlib.Path(%r)
result={}
for relative in %r:
    path=root/relative
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024**2),b''):
            value.update(block)
    stat=path.stat()
    result[relative]={'size_bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'content':value.hexdigest()}
print(json.dumps(result))
""" % (REMOTE, paths)
    response = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "qingyuan", "python3 -"],
                              input=program, text=True, capture_output=True, check=True, timeout=600)
    remote = json.loads(response.stdout)
    for name, identity in anchors.items():
        if digest(identity["path"]) != remote[name]["content"]:
            raise ValueError("Remote registered anchor differs from the existing local source")
        verify_identity(identity)
    total = sum(remote[name]["size_bytes"] for name in requested)
    check_disk(config, total)
    target = root / "source_supplement"
    target.mkdir(exist_ok=True)
    print(json.dumps({"stage": "anchors_verified", "anchors": len(anchors), "missing_files": len(requested), "bytes": total}), flush=True)
    subprocess.run(["rsync", "-a", "--relative", "--files-from=-", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=15",
                    "qingyuan:" + REMOTE + "/", str(target) + "/"],
                   input="\n".join(sorted(requested)) + "\n", text=True, check=True, timeout=1800)
    restored = []
    for name, missing in requested.items():
        path = target / name
        if path.stat().st_size != remote[name]["size_bytes"] or digest(path) != remote[name]["content"]:
            raise ValueError("Restored registered phase differs from its remote source")
        restored.append({"original_missing_path": missing["path"], "visit_id": missing["visit_id"], "phase": missing["phase"],
                         "source": file_identity(path), "remote_path": REMOTE + "/" + name,
                         "remote_size_bytes": remote[name]["size_bytes"], "remote_mtime_ns": remote[name]["mtime_ns"]})
    write_json(root / "source_supplement.json", {"schema": "registered_three_phase_source_supplement_v1",
               "status": "verified", "existing_anchors": list(anchors.values()), "existing_anchors_equal": True,
               "restored_sources_equal": True, "restored": restored, "total_bytes": total})
    print(json.dumps({"stage": "complete", "restored_files": len(restored), "bytes": total}), flush=True)


if __name__ == "__main__":
    main()

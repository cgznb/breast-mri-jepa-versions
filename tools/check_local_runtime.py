#!/usr/bin/env python3
"""Check the migrated ROI32 data and local V1 runtime without starting training.

Only aggregate counts and fixed check names are written or printed. NPZ members
are inspected through their NPY headers. Optional full-file SHA256 verification
compares every image byte with the original server's migration checksums.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("data/registered_roi32")
VERSION = Path("versions/v1_evolution")
STEM = "evolution_vq_forecast_history_residual_free_encoder64_seed"
EXPECTED_PATIENTS = 866
EXPECTED_IMAGES = 3039
EXPECTED_SPLITS = {"train": 764, "val": 102}
ARRAYS = {
    "image.npy": ((3, 32, 128, 128), {"<f4", "=f4"}, 4),
    "roi.npy": ((1, 32, 128, 128), {"|b1", "?", "|?"}, 1),
    "support.npy": ((3, 32, 128, 128), {"|b1", "?", "|?"}, 1),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def read_header(archive: zipfile.ZipFile, name: str) -> tuple[dict, int]:
    with archive.open(name) as handle:
        prefix = handle.read(8)
        if len(prefix) != 8 or prefix[:6] != b"\x93NUMPY":
            raise ValueError("invalid NPY magic")
        version = tuple(prefix[6:8])
        if version == (1, 0):
            length_size, length_format, encoding = 2, "<H", "latin1"
        elif version in {(2, 0), (3, 0)}:
            length_size, length_format = 4, "<I"
            encoding = "utf8" if version == (3, 0) else "latin1"
        else:
            raise ValueError("unsupported NPY version")
        raw_length = handle.read(length_size)
        if len(raw_length) != length_size:
            raise ValueError("truncated NPY length")
        length = struct.unpack(length_format, raw_length)[0]
        if length > 65536:
            raise ValueError("oversized NPY header")
        raw_header = handle.read(length)
        if len(raw_header) != length:
            raise ValueError("truncated NPY header")
        header = ast.literal_eval(raw_header.decode(encoding))
        if not isinstance(header, dict):
            raise ValueError("invalid NPY header")
        return header, 8 + length_size + length


def check_npz(path: Path) -> Counter:
    present = Counter()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if "image.npy" not in names or len(names) != len(set(names)):
            raise ValueError("missing image or duplicate ZIP members")
        for name, (shape, descriptors, itemsize) in ARRAYS.items():
            if name not in names:
                continue
            header, header_bytes = read_header(archive, name)
            if (header.get("shape") != shape or header.get("descr") not in descriptors
                    or header.get("fortran_order") is not False):
                raise ValueError("incompatible array header")
            if archive.getinfo(name).file_size != header_bytes + math.prod(shape) * itemsize:
                raise ValueError("incompatible array member size")
            present[name.removesuffix(".npy")] += 1
    return present


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path,
                        help="Aggregate JSON report; default: private_outputs/setup_runtime/runtime_check.json")
    parser.add_argument("--data-only", action="store_true",
                        help="Skip YAML and PyTorch checks; only Python's standard library is required")
    parser.add_argument("--skip-image-headers", action="store_true",
                        help="Only check NPZ presence/sizes; the report explicitly marks headers as skipped")
    parser.add_argument("--full-image-sha256", action="store_true",
                        help="Read all image bytes and compare migration-source-sha256.json")
    args = parser.parse_args(argv)
    root = args.project_root.expanduser().resolve()
    output = (args.output.expanduser().resolve() if args.output else
              root / "private_outputs/setup_runtime/runtime_check.json")
    data = root / DATA
    checks: dict[str, dict] = {}

    def record(name: str, passed: bool, **details) -> None:
        checks[name] = {"passed": bool(passed), **details}

    def attempt(name: str, function):
        try:
            return function()
        except Exception as exc:
            # Exception messages and per-record paths can contain patient IDs.
            record(name, False, error_type=type(exc).__name__)
            return None

    inventory = attempt("source_inventory_readable",
                        lambda: read_json(data / "migration-source-inventory.json"))
    original = attempt("original_manifest_readable", lambda: read_json(data / "manifest.original.json"))
    local = attempt("local_manifest_readable", lambda: read_json(data / "manifest.json"))
    if inventory is not None:
        record("source_inventory_readable", True)
    if original is not None:
        record("original_manifest_readable", True)
    if local is not None:
        record("local_manifest_readable", True)
    image_checksums = None
    if args.full_image_sha256:
        image_checksums = attempt("source_image_checksums_readable",
                                  lambda: read_json(data / "migration-source-sha256.json"))
        if image_checksums is not None:
            record("source_image_checksums_readable", True)

    def check_source_hashes():
        for name, filename, key in (
            ("original_manifest_sha256", data / "manifest.original.json", "source_manifest_sha256"),
            ("codec_sha256", root / "data/pretrained/codec.pt", "codec_sha256"),
        ):
            def verify_hash(path=filename, expected=inventory[key]):
                record(name, isinstance(expected, str) and len(expected) == 64 and sha256(path) == expected)
            attempt(name, verify_hash)
    if inventory is not None:
        attempt("source_hashes_checked", check_source_hashes)

    def check_manifest():
        patients = local["patients"]
        keys = [patient["patient_key"] for patient in patients]
        splits = Counter(patient["split"] for patient in patients)
        record("patient_cohort", len(patients) == EXPECTED_PATIENTS
               and all(isinstance(key, str) and key for key in keys)
               and len(set(keys)) == EXPECTED_PATIENTS and dict(splits) == EXPECTED_SPLITS,
               patients=len(patients), unique_patients=len(set(keys)), splits=dict(splits))
        restored = copy.deepcopy(local)
        paths = []
        stage_counts = [0, 0, 0, 0]
        image_paths_valid = True
        if len(patients) != len(original["patients"]):
            record("manifest_only_image_paths_changed", False)
            return []
        for patient, source_patient, restore_patient in zip(patients, original["patients"], restored["patients"]):
            visits, source_visits = patient["visits"], source_patient["visits"]
            if len(visits) != 4 or len(source_visits) != 4:
                raise ValueError("incompatible visit slots")
            for stage, (visit, source_visit, restore_visit) in enumerate(
                    zip(visits, source_visits, restore_patient["visits"])):
                if visit is None or source_visit is None:
                    if visit != source_visit:
                        image_paths_valid = False
                    continue
                filename = Path(source_visit["image"]).name
                expected = "views/" + filename
                image_paths_valid &= visit.get("image") == expected
                restore_visit["image"] = source_visit["image"]
                paths.append(visit["image"])
                stage_counts[stage] += 1
        canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
        record("manifest_only_image_paths_changed", image_paths_valid
               and canonical(restored) == canonical(original), image_references=len(paths))
        record("manifest_image_population", len(paths) == EXPECTED_IMAGES
               and len(set(paths)) == EXPECTED_IMAGES
               and stage_counts == [866, 836, 727, 610],
               image_references=len(paths), unique_images=len(set(paths)), stages=stage_counts)
        return paths

    paths = None
    if original is not None and local is not None:
        paths = attempt("manifest_structure", check_manifest)

    def check_files():
        image_files = inventory["image_files"]
        inventory_valid = (isinstance(image_files, dict) and len(image_files) == EXPECTED_IMAGES
                           and all(isinstance(name, str) and Path(name).name == name
                                   and isinstance(info, dict) and type(info.get("size")) is int
                                   and info["size"] > 0 for name, info in image_files.items()))
        record("inventory_image_population", inventory_valid, images=len(image_files))
        if not inventory_valid:
            return
        record("manifest_matches_inventory", paths is not None
               and set(paths) == {"views/" + name for name in image_files})
        expected_hashes = None
        if args.full_image_sha256 and isinstance(image_checksums, dict):
            candidate = image_checksums.get("images")
            if (image_checksums.get("algorithm") == "sha256" and isinstance(candidate, dict)
                    and set(candidate) == set(image_files)
                    and all(isinstance(row, dict) and row.get("bytes") == image_files[name]["size"]
                            and isinstance(row.get("sha256"), str) and len(row["sha256"]) == 64
                            and all(ch in "0123456789abcdef" for ch in row["sha256"])
                            for name, row in candidate.items())):
                expected_hashes = candidate
        if args.full_image_sha256:
            record("source_image_checksum_population", expected_hashes is not None)
        counts = Counter()
        members = Counter()
        expected_bytes = sum(info["size"] for info in image_files.values())
        views = (data / "views").resolve()
        for name, info in image_files.items():
            path = data / "views" / name
            try:
                if not path.is_file() or path.resolve().parent != views:
                    counts["missing_or_outside_views"] += 1
                    continue
                counts["present"] += 1
                if path.stat().st_size != info["size"]:
                    counts["size_mismatch"] += 1
                    continue
                counts["size_matched"] += 1
                if expected_hashes is not None:
                    if sha256(path) == expected_hashes[name]["sha256"]:
                        counts["sha256_matched"] += 1
                    else:
                        counts["sha256_mismatch"] += 1
                if not args.skip_image_headers:
                    try:
                        members.update(check_npz(path))
                        counts["header_matched"] += 1
                    except Exception:
                        counts["invalid_archive_or_header"] += 1
            except OSError:
                counts["unreadable"] += 1
        record("image_files_present_and_sized", counts["size_matched"] == EXPECTED_IMAGES,
               expected_files=EXPECTED_IMAGES, expected_bytes=expected_bytes,
               present=counts["present"], size_matched=counts["size_matched"],
               missing_or_outside_views=counts["missing_or_outside_views"],
               size_mismatch=counts["size_mismatch"], unreadable=counts["unreadable"])
        if args.full_image_sha256:
            record("image_sha256", counts["sha256_matched"] == EXPECTED_IMAGES,
                   matched=counts["sha256_matched"], mismatched=counts["sha256_mismatch"],
                   expected_files=EXPECTED_IMAGES)
        else:
            checks["image_sha256"] = {"skipped": True}
        if args.skip_image_headers:
            checks["npz_array_headers"] = {"skipped": True}
        else:
            record("npz_array_headers", counts["header_matched"] == EXPECTED_IMAGES,
                   checked=counts["header_matched"], invalid=counts["invalid_archive_or_header"],
                   array_counts=dict(members), array_contents_and_crc_checked=False)
    if inventory is not None:
        attempt("image_file_checks", check_files)

    def check_configs():
        import yaml

        for seed in (17, 43):
            name = f"local_config_seed{seed}"
            def verify_config(seed=seed, name=name):
                source = yaml.safe_load((root / VERSION / "configs" / f"{STEM}{seed}.yaml").read_text())
                config = yaml.safe_load((root / VERSION / "local_configs" / f"encoder64_seed{seed}.yaml").read_text())
                expected = copy.deepcopy(source)
                expected["encoder_checkpoint"] = str((root / "data/pretrained/codec.pt").resolve())
                expected["image_cache"] = None
                record(name, config == expected and config["seed"] == seed,
                       seed=seed, only_resource_paths_changed=config == expected,
                       image_cache_disabled=config.get("image_cache") is None)
            attempt(name, verify_config)

    def check_cuda():
        import torch

        available = torch.cuda.is_available()
        record("torch_cuda_available", available, torch_version=torch.__version__,
               torch_cuda_version=torch.version.cuda,
               device_count=torch.cuda.device_count() if available else 0)
        if not available:
            return
        # Run a tiny BF16 matmul to verify execution as well as advertised support.
        supported = torch.cuda.is_bf16_supported()
        executed = False
        if supported:
            with torch.inference_mode():
                matrix = torch.ones((8, 8), device="cuda", dtype=torch.bfloat16)
                result = matrix @ matrix
                executed = bool(torch.isfinite(result).all().item()) and result.dtype == torch.bfloat16
                torch.cuda.synchronize()
        properties = torch.cuda.get_device_properties(0)
        record("cuda_bf16_execution", supported and executed,
               supported=supported, executed=executed, device_name=properties.name,
               total_memory_bytes=properties.total_memory)

    if args.data_only:
        checks["local_configs"] = {"skipped": True}
        checks["torch_cuda"] = {"skipped": True}
    else:
        attempt("local_configs_readable", check_configs)
        attempt("torch_cuda_execution", check_cuda)

    passed = all(item.get("passed", True) for item in checks.values())
    summary = {
        "schema": "vla_pcrv1_local_runtime_check_v1",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "mode": "data_only" if args.data_only else "data_and_cuda_runtime",
        "training_started": False,
        "checks": checks,
        "limitations": [
            "NPZ payloads are not decompressed; array values and ZIP CRCs are not independently evaluated.",
            *(["Per-image SHA256 verification was not requested."]
              if not args.full_image_sha256 else []),
        ],
    }
    # Keep reports out of immutable inputs, historical configs, and source files.
    if not output.is_relative_to(root / "private_outputs") or output.suffix != ".json":
        parser.error("--output must be a JSON file inside the project's private_outputs directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    failed = [name for name, result in checks.items() if result.get("passed") is False]
    print(json.dumps({"passed": passed, "mode": summary["mode"], "failed_checks": failed,
                      "report": str(output)}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

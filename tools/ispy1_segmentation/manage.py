#!/usr/bin/env python3
"""Durable one-GPU I-SPY1 inference queue, progress tables and final validation."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import io
import json
import os
import stat
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    def reject_nonfinite(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(path.read_text(), parse_constant=reject_nonfinite)


def write_text(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_json(path, value):
    write_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_receipt(receipt, case, contract, contract_error):
    """Check current case/contract and derived artifacts without statting DICOMs."""
    if not isinstance(receipt, dict) or receipt.get("status") != "complete":
        raise ValueError("Receipt is not a completed object")
    if contract_error:
        raise ValueError("Cannot verify run contract: " + contract_error)
    if receipt.get("contract") != contract:
        raise ValueError("Receipt differs from current run_contract.json")
    if receipt.get("case_id") != case["case_id"]:
        raise ValueError("Receipt case ID differs from inventory")
    selected = case["first_post"]
    phase = receipt.get("source_phase", {})
    expected_files = selected["files"]
    source_files = receipt.get("source_files", [])
    phase_files = phase.get("files", [])
    if phase.get("series_uid") != selected["series_uid"]:
        raise ValueError("Receipt source series differs from current inventory")
    if any(not isinstance(files, list) or len(files) != len(expected_files) or set(files) != set(expected_files) for files in (source_files, phase_files)):
        raise ValueError("Receipt source files differ from current inventory")
    metadata = receipt.get("output_file_metadata")
    if not isinstance(metadata, list) or not metadata:
        raise ValueError("Receipt has no derived output metadata")
    output_paths = receipt.get("output_paths")
    if not isinstance(output_paths, dict) or not output_paths:
        raise ValueError("Receipt has no derived output paths")
    listed_paths = set()
    for item in metadata:
        path = Path(item["path"])
        current = path.stat()
        if not stat.S_ISREG(current.st_mode):
            raise ValueError("Recorded output is not a file: " + str(path))
        if current.st_size != item["size_bytes"] or current.st_mtime_ns != item["mtime_ns"]:
            raise ValueError("Recorded output changed: " + str(path))
        listed_paths.add(str(path.resolve()))
    if not {str(Path(path).resolve()) for path in output_paths.values()}.issubset(listed_paths):
        raise ValueError("Derived output path is absent from output_file_metadata")
    if not isinstance(receipt.get("qc"), dict):
        raise ValueError("Receipt QC is not an object")
    if not isinstance(receipt["qc"].get("review_flags", []), list):
        raise ValueError("Receipt QC review_flags is not a list")


def summarize(inventory, output):
    """Check receipt/output metadata; no DICOM stats, voxel loads or GPU access."""
    cases = read_json(inventory)["cases"]
    contract, contract_error = None, ""
    try:
        contract = read_json(output / "run_contract.json")
        if not isinstance(contract, dict):
            raise ValueError("run_contract.json is not an object")
    except Exception as error:
        contract_error = f"{type(error).__name__}: {error}"
    rows = []
    inventory_counts, result_counts = Counter(), Counter()
    by_visit = defaultdict(Counter)
    for case in sorted(cases, key=lambda c: (c["patient_id"], c.get("normalized_visit", "") or "", c["case_id"])):
        directory = output / "cases" / case["case_id"]
        receipt_path, status_path = directory / "receipt.json", directory / "status.json"
        has_receipt = receipt_path.exists()
        receipt, receipt_error, status_error = None, "", ""
        if has_receipt:
            try:
                receipt = read_json(receipt_path)
                validate_receipt(receipt, case, contract, contract_error)
            except Exception as error:
                receipt_error = f"{type(error).__name__}: {error}"
                receipt = None
        case_status = {}
        if status_path.exists():
            try:
                case_status = read_json(status_path)
                if not isinstance(case_status, dict):
                    raise ValueError("Case status is not an object")
            except Exception as error:
                status_error = f"{type(error).__name__}: {error}"
                case_status = {}
        source_status = case["status"]
        inventory_counts[source_status] += 1
        visit = case.get("normalized_visit") or "unknown"
        by_visit[visit]["inventory_" + source_status] += 1
        if receipt_error or (has_receipt and status_error):
            result = "invalid_receipt"
        elif source_status != "ready":
            result = source_status
        elif receipt is not None and receipt.get("status") == "complete" and case_status.get("status") != "failed":
            result = "complete"
        elif status_error:
            result = "failed"
        elif case_status.get("status") == "failed":
            result = "failed"
        elif case_status.get("status") == "running":
            result = "running"
        else:
            result = "pending"
        result_counts[result] += 1
        by_visit[visit]["result_" + result] += 1
        qc = receipt.get("qc", {}) if receipt else {}
        paths = receipt.get("output_paths", {}) if receipt else {}
        reasons = case.get("qc", {}).get("reasons", [])
        rows.append({
            "case_id": case["case_id"], "patient_id": case["patient_id"], "study_uid": case["study_uid"],
            "normalized_visit": visit, "original_visit": case.get("original_visit"),
            "inventory_status": source_status, "result_status": result,
            "inference_stage": case_status.get("stage", ""),
            "volume_ml": qc.get("volume_ml"), "voxel_count": qc.get("voxel_count"),
            "component_count": qc.get("component_count"), "empty": qc.get("empty"),
            "edge_voxel_count": qc.get("edge_voxel_count"),
            "review_flags": ";".join(str(flag) for flag in qc.get("review_flags", [])),
            "inventory_reasons": ";".join(str(r) for r in reasons),
            "elapsed_seconds": receipt.get("elapsed_seconds") if receipt else None,
            "error": "; ".join(str(error) for error in (receipt_error, status_error, case_status.get("error", "")) if error),
            "receipt_validation_error": receipt_error,
            "native_image": paths.get("native_image", ""),
            "native_mask": paths.get("native_mask", ""), "overlay": paths.get("overlay", ""),
            "receipt": str(receipt_path) if has_receipt else "",
        })
    summary = {
        "schema": "ispy1_mamamia_queue_summary_v1", "updated_at": now(),
        "inventory": str(inventory), "output": str(output),
        "run_contract_error": contract_error,
        "patients": len({c["patient_id"] for c in cases}), "studies": len(cases),
        "inventory_counts": dict(inventory_counts), "result_counts": dict(result_counts),
        "by_visit": {v: dict(count) for v, count in sorted(by_visit.items())},
        "ready_cases_complete": result_counts["complete"] == inventory_counts["ready"],
        "unresolved_inventory_cases": sum(count for status, count in inventory_counts.items() if status not in ("ready", "excluded")),
        "note": "Completed masks are automated predictions. Anatomical QC is separate from geometry validation; no Dice is inferred without matched reference masks.",
        "cases": rows,
    }
    write_json(output / "aggregate.json", summary)
    buffer = io.StringIO()
    if rows:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_text(output / "aggregate.csv", buffer.getvalue())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.04)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=20)
    parser.add_argument("--detach", action="store_true", help="Start durable background manager and print PID/log")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    args.inventory, args.output = args.inventory.resolve(), args.output.resolve()
    if not 1 <= args.poll_seconds <= 60:
        raise ValueError("poll-seconds must be in [1, 60]")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summary = summarize(args.inventory, args.output)
        print(json.dumps({k: v for k, v in summary.items() if k != "cases"}), flush=True)
        return 0
    if args.detach:
        command = [sys.executable, str(Path(__file__).resolve()), *[a for a in sys.argv[1:] if a != "--detach"]]
        with (args.output / "manager.log").open("a", buffering=1) as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        print(json.dumps({"manager_pid": process.pid, "log": str(args.output / "manager.log"), "output": str(args.output)}), flush=True)
        return 0
    with (args.output / ".manager.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another queue manager already owns this output directory") from None
        command = [sys.executable, str(HERE / "infer.py"), "--inventory", str(args.inventory), "--output", str(args.output), "--device", args.device, "--gpu-memory-fraction", str(args.gpu_memory_fraction), "--cpu-threads", str(args.cpu_threads)]
        state = {"schema": "ispy1_mamamia_queue_manager_v1", "manager_pid": os.getpid(), "started_at": now(), "status": "inference_running", "command": command}
        write_json(args.output / "manager_status.json", state)
        print(json.dumps(state), flush=True)
        with (args.output / "inference.log").open("a", buffering=1) as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
            state["inference_pid"] = process.pid
            while True:
                summary = summarize(args.inventory, args.output)
                state.update(updated_at=now(), result_counts=summary["result_counts"], inventory_counts=summary["inventory_counts"])
                write_json(args.output / "manager_status.json", state)
                print(json.dumps({"updated_at": state["updated_at"], "result_counts": state["result_counts"]}), flush=True)
                try:
                    returncode = process.wait(timeout=args.poll_seconds)
                    break
                except subprocess.TimeoutExpired:
                    pass
        state.update(status="independent_validation_running", inference_returncode=returncode, updated_at=now())
        write_json(args.output / "manager_status.json", state)
        validator = [sys.executable, str(HERE / "verify_outputs.py"), "--output", str(args.output)]
        with (args.output / "independent_validation.log").open("a", buffering=1) as log:
            verification = subprocess.Popen(validator, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            state["validation_pid"] = verification.pid
            while True:
                try:
                    validation_returncode = verification.wait(timeout=args.poll_seconds)
                    break
                except subprocess.TimeoutExpired:
                    state["updated_at"] = now()
                    write_json(args.output / "manager_status.json", state)
        summary = summarize(args.inventory, args.output)
        success = returncode == 0 and validation_returncode == 0 and summary["ready_cases_complete"]
        state.update(status="ready_cases_completed_and_verified" if success else "attention_required", completed_at=now(), validation_returncode=validation_returncode, result_counts=summary["result_counts"], inventory_counts=summary["inventory_counts"], unresolved_inventory_cases=summary["unresolved_inventory_cases"], by_visit=summary["by_visit"])
        write_json(args.output / "manager_status.json", state)
        print(json.dumps(state), flush=True)
        return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

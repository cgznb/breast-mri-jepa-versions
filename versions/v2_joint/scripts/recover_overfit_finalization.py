#!/usr/bin/env python3
"""Reconcile the known post-training CLI KeyError after strict artifact replay.

This never trains, resumes, changes checkpoints, or rewrites trainer exit codes.
All runs must have the exact terminal-print failure and pass the frozen-source
CPU analyzer before any queue state is changed.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

import analyze_overfit_suite as analysis
import run_overfit_suite as queue


BASIS = "verified_post_training_cli_failure"


def validate_terminal_failure(run):
    attempts = run["attempts"]
    if run["state"] != "failed" or len(attempts) != 1 or attempts[0].get("resume"):
        raise ValueError("Recovery requires one failed, non-resumed attempt")
    attempt = attempts[0]
    _, exit_path = queue.attempt_paths(run, attempt)
    result = queue.read_json(exit_path)
    if attempt.get("exit_code") != 1 or not result or result.get("exit_code") != 1:
        raise ValueError("Original trainer exit must remain 1")
    if not queue.terminal_training(run["output"]):
        raise ValueError("Training terminal artifacts are incomplete")
    lines = Path(run["log"]).read_text(encoding="utf-8").strip().splitlines()
    frame = f'  File "{PROJECT / "src/response_jepa/cli.py"}", line 85, in main'
    if (not lines or lines[-1] != "KeyError: 'best_epoch'"
            or lines.count("Traceback (most recent call last):") != 1
            or len(lines) < 4 or lines[-4] != frame
            or 'result["best_epoch"]' not in lines[-3]
            or 'print(json.dumps(' not in lines[-3]):
        raise ValueError("Failure is not the exact post-training CLI print KeyError")
    commands = [queue.worker_command(run, attempt)]
    for resume in (False, True):
        commands.extend(factory(run["config"], run["manifest"], run["output"], resume)
                        for factory in (queue.trainer_command, queue._helpers.trainer_command))
    if any(queue.find_process(command, PROJECT) for command in commands):
        raise ValueError("A worker or trainer is still alive")
    return {"name": run["name"], "output": run["output"], "original_failure": run.get("failure"),
            "original_exit": result, "original_exit_file": str(exit_path), "log": run["log"],
            "artifact_files": {name: queue.file_record(Path(run["output"]) / name) for name in
                ("best.pt", "last.pt", "config.json", "completion.json", "history.json",
                 "metrics_best.json", "predictions_val.csv", "world_val.csv", "provenance.json")}}


def recover(suite, plan, report):
    suite, plan, report = [Path(path).resolve() for path in (suite, plan, report)]
    if not all(path.is_file() for path in (suite / "status.json", suite / "plan.json", plan)):
        raise ValueError("Recovery requires an existing queue and immutable plan")
    with ExitStack() as stack:
        for path in (suite.parent / ".research.lock", suite / ".controller.lock"):
            handle = stack.enter_context(path.open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = queue.prepare_suite(PROJECT, suite, plan)
        receipt_path = suite / "post_training_recovery.json"
        if all(run["state"] == "completed" and run.get("completion_basis") == BASIS
               for run in status["runs"]):
            if not receipt_path.is_file():
                raise ValueError("Recovered status has no recovery receipt")
            return {"state": "already_recovered", "receipt": str(receipt_path)}
        for run in status["runs"]:
            handle = stack.enter_context((Path(run["output"]) / ".queue/worker.lock").open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evidence = [validate_terminal_failure(run) for run in status["runs"]]
        audited = analysis.analyze_suite(plan)
        if (not audited.get("complete") or audited.get("errors")
                or {row["name"] for row in audited["runs"] if row.get("verified")}
                != {run["name"] for run in status["runs"]}):
            raise ValueError("Strict artifact and frozen-source replay failed; queue is unchanged")
        backup = suite / "status.before_post_training_recovery.json"
        if backup.exists() or receipt_path.exists():
            raise ValueError("Recovery evidence already exists; inspect before retrying")
        analysis.write_report(audited, report)
        queue.atomic_json(backup, status)
        research = queue.read_json(suite.parent / "research_status.json")
        research_backup = suite.parent / "research_status.before_screen_recovery.json"
        if research is not None and not research_backup.exists():
            queue.atomic_json(research_backup, research)
        receipt = {"schema": "overfit_post_training_recovery_v1", "recovered_at": queue.now(),
                   "basis": BASIS, "plan": str(plan), "verified_report": str(report / "aggregate.json"),
                   "reason": "Training completed; CLI printed best_epoch instead of best_update after saving artifacts",
                   "training_repeated": False, "original_exits_preserved": True, "runs": evidence}
        queue.atomic_json(receipt_path, receipt)
        for run in status["runs"]:
            run.update(state="completed", completion_basis=BASIS, recovery_receipt=str(receipt_path))
            if "failure" in run:
                run["original_failure"] = run.pop("failure")
        status.update(updated_at=queue.now(), post_training_recovery=str(receipt_path))
        queue.atomic_json(suite / "status.json", status)
    return {"state": "recovered", "runs": len(status["runs"]), "receipt": str(receipt_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(recover(args.suite, args.plan, args.report), ensure_ascii=False))


if __name__ == "__main__":
    main()

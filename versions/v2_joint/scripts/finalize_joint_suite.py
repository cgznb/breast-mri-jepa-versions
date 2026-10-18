#!/usr/bin/env python3
"""Wait for the six-run queue, then replay aggregate results on CPU.

The watcher only owns finalize_status.json and its own lock/log. It never
modifies queue state or signals the controller or training processes.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
EXPECTED = {f"joint_{profile}_seed{seed}" for profile in ("t0", "dynamic")
            for seed in (17, 43, 71)}


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def controller_alive(status, project=PROJECT):
    """A recycled PID or an unrelated process must not keep a watcher alive."""
    pid = (status.get("controller") or {}).get("pid")
    if not isinstance(pid, int):
        return False
    try:
        folder = Path("/proc")/str(pid)
        fields = (folder/"stat").read_text().rsplit(")", 1)[1].split()
        command = (folder/"cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
        return (fields[0] != "Z" and str(project/"scripts"/"run_joint_suite.py") in command
                and (folder/"cwd").resolve() == project.resolve())
    except (OSError, ValueError, IndexError):
        return False


def disposition(status, alive):
    """Return waiting, terminal, or orphaned; never infer success from files."""
    if not isinstance(status, dict):
        return "waiting"
    runs = status.get("runs")
    if not isinstance(runs, list) or {r.get("name") for r in runs} != EXPECTED or len(runs) != 6:
        return "waiting"
    states = [r.get("state") for r in runs]
    if all(state in {"completed", "failed"} for state in states):
        return "terminal"
    if alive or "running" in states:
        # Children survive controller restarts and can be adopted later.
        return "waiting"
    return "orphaned"


def analysis_environment(project=PROJECT):
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(project/"src"), PYTHONDONTWRITEBYTECODE="1",
                       PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="",
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    return environment


def run_analysis(suite, output, baseline=None, project=PROJECT):
    command = [sys.executable, str(project/"scripts"/"analyze_joint_suite.py"),
               "--suite-root", str(suite), "--output", str(output)]
    if baseline:
        command += ["--baseline-csv", str(baseline)]
    result = subprocess.run(command, cwd=project, env=analysis_environment(project), check=False)
    report = read_json(output/"aggregate.json")
    complete = bool(isinstance(report, dict) and report.get("complete"))
    error = None if result.returncode == 0 and complete else (
        "Analysis failed" if result.returncode else "Partial report: fewer than six verified completed runs")
    paired = project/"scripts"/"compare_joint_old_predictions.py"
    paired_returncode = None
    paired_report_path = None
    if result.returncode == 0 and report is not None and paired.is_file():
        compared = subprocess.run([sys.executable, str(paired), "--suite-root", str(suite),
                                   "--output", str(output/"paired_baseline")],
                                  cwd=project, env=analysis_environment(project), check=False)
        paired_returncode = compared.returncode
        paired_report_path = str(output/"paired_baseline"/"aggregate.json")
        if compared.returncode:
            error = "Paired comparison with historical predictions failed"
    plot = project/"scripts"/"plot_joint_suite.py"
    plot_returncode = None
    if result.returncode == 0 and report is not None and plot.is_file():
        plotted = subprocess.run([sys.executable, str(plot), "--suite-root", str(suite),
                                  "--output", str(output/"figures")],
                                 cwd=project, env=analysis_environment(project), check=False)
        plot_returncode = plotted.returncode
        if plotted.returncode:
            error = "Optional result plotting failed"
    return {"analysis_returncode": result.returncode, "plot_returncode": plot_returncode,
            "paired_returncode": paired_returncode, "paired_report_path": paired_report_path,
            "partial": not complete, "report_path": str(output/"README_ZH.md"),
            "error": error, "returncode": result.returncode or paired_returncode or plot_returncode or (0 if complete else 2)}


def watch(suite, output, baseline=None, poll_seconds=30, project=PROJECT):
    suite, output = Path(suite).resolve(), Path(output).resolve()
    with (suite/".finalize.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another finalizer owns this suite", flush=True)
            return 3
        own = {"pid": os.getpid(), "state": "waiting"}
        path = suite/"finalize_status.json"
        atomic_json(path, own)
        try:
            while True:
                status = read_json(suite/"status.json")
                decision = disposition(status, controller_alive(status, project) if status else False)
                if decision != "waiting":
                    break
                time.sleep(poll_seconds)
            own.update(state="analyzing", trigger=decision)
            atomic_json(path, own)
            results = run_analysis(suite, output, baseline, project)
            if decision == "orphaned":
                results.update(partial=True, error="Controller disappeared before all runs were terminal")
                results["returncode"] = results["returncode"] or 2
            own.update(results, state="completed" if results["returncode"] == 0 else "failed")
            atomic_json(path, own)
            return results["returncode"]
        except Exception as exc:
            # Store only the exception class; raw checkpoint errors can contain identifiers.
            own.update(state="failed", error=f"Finalizer exception: {type(exc).__name__}", returncode=1)
            atomic_json(path, own)
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-csv")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    suite = Path(args.suite_root).resolve()
    if not suite.is_dir():
        parser.error("--suite-root must be an existing directory")
    if args.detach:
        command = [sys.executable, str(SCRIPT), "--suite-root", str(suite),
                   "--output", str(Path(args.output).resolve()), "--poll-seconds", str(args.poll_seconds)]
        if args.baseline_csv:
            command += ["--baseline-csv", str(Path(args.baseline_csv).resolve())]
        with (suite/"finalize.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=PROJECT, env=analysis_environment(),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        print(json.dumps({"finalizer_pid": child.pid, "suite_root": str(suite)}))
        return 0
    return watch(suite, args.output, args.baseline_csv, args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Collect stage/seed metrics, not best-of-seed scores or false significance claims."""
import argparse
import csv
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rows = []
    for path in sorted(Path(args.workdir).rglob("metrics_best.json")):
        cfg = json.loads((path.parent/"config.json").read_text())
        if cfg["train"]["phase"] not in {"C", "joint"}:
            continue
        result = json.loads(path.read_text())
        world = result.get("world")
        for branch, value in result["classification"].items():
            if value is None:
                continue
            for stage, score in value["per_stage"].items():
                if not score["n"]:
                    continue
                rows.append({"run": str(path.parent.relative_to(args.workdir)),
                             "profile": cfg["train"]["profile"], "seed": cfg["train"]["seed"],
                             "branch": branch, "stage": stage, **score,
                             "world_all_ratio": world["all"]["ratio"] if world else None,
                             "target_coordinate_origin": result.get("target_coordinate_origin")})
    if not rows:
        raise ValueError("No completed C results found")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} seed/stage/branch rows; use patient predictions for paired confidence intervals.")

if __name__ == "__main__":
    main()

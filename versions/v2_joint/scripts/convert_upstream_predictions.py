#!/usr/bin/env python3
"""Convert original patient-level JSON rows; never infer patients from aggregate AUROC."""
import argparse
import csv
import json
from pathlib import Path

def candidates(x):
    out = []
    if isinstance(x, list):
        if x and all(isinstance(r, dict) and {"patient_key", "landmark", "pcr_probability", "pcr"} <= set(r) for r in x):
            return [x]
        for value in x:
            out.extend(candidates(value))
    elif isinstance(x, dict):
        for value in x.values():
            out.extend(candidates(value))
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    found = candidates(json.loads(Path(args.input).read_text()))
    if len(found) != 1:
        raise ValueError("Expected one unambiguous patient prediction list. Aggregates cannot reconstruct predictions.")
    keys, rows = set(), []
    for old in found[0]:
        key = str(old["patient_key"]).removeprefix("ISPY2-")
        if key.endswith(".0") and key[:-2].isdigit():
            key = key[:-2]
        stage, prob = int(old["landmark"]), float(old["pcr_probability"])
        if (key, stage) in keys or stage not in range(4) or not 0 <= prob <= 1:
            raise ValueError("Duplicate or invalid prediction")
        keys.add((key, stage))
        label = old["pcr"]
        if label is not None and label not in (0, 1):
            raise ValueError("Invalid binary label")
        rows.append({"patient_key": key, "landmark": stage, "label": label,
                     "fused": prob, "direct": prob, "forecast": None})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.chmod(0o600)

if __name__ == "__main__":
    main()

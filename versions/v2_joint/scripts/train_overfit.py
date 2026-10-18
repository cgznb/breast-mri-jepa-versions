#!/usr/bin/env python3
"""Train through the frozen runtime and report its update or epoch summary.

The package CLI assumes an epoch-based return value. Keep the package source
unchanged for strict replay of existing experiments and adapt the return value
here, outside the recorded runtime package.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from response_jepa.config import load_config
from response_jepa.engine import train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    result = train(cfg, args.manifest, args.output, resume=args.resume)
    fields = ("best_epoch",) if cfg.train.max_updates is None else (
        "best_update", "updates", "stop_reason")
    summary = {"output": result["output"], **{field: result[field] for field in fields}}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

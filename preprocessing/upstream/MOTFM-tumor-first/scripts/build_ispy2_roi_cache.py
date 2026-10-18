#!/usr/bin/env python3
"""Build or resume the validated MOTFM I-SPY2 per-visit ROI cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.ispy2_dataset import ISPY2LongitudinalDataset, VALID_FOLDS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--fold",
        action="append",
        choices=VALID_FOLDS,
        help="fold to cache; may be repeated (default: all folds)",
    )
    parser.add_argument("--output-shape", nargs=3, type=int, metavar=("D", "H", "W"))
    parser.add_argument(
        "--target-spacing", nargs=3, type=float, metavar=("X", "Y", "Z")
    )
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dataset = ISPY2LongitudinalDataset(
        args.bundle,
        cache_dir=args.cache_dir,
        fold=args.fold or VALID_FOLDS,
        output_shape_zyx=args.output_shape,
        target_spacing_xyz=args.target_spacing,
    )
    counts = dataset.build_cache(progress_every=args.progress_every)
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

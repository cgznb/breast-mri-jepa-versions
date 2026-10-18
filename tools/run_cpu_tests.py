#!/usr/bin/env python3
"""Run one isolated project's existing tests with portable CPU BF16 backends."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    project = args.project.expanduser().resolve(strict=True)
    if not (project / "src").is_dir() or not (project / "tests").is_dir():
        parser.error("--project must contain src/ and tests/")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.chdir(project)
    sys.path.insert(0, str(project / "src"))
    import torch
    import pytest

    # Some CPU builds have BF16 fused-MHA / oneDNN paths that do not support
    # the test operations. Use regular CPU operators, retaining every test.
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.mkldnn.enabled = False
    print(f"CPU validation: torch={torch.__version__}; fused MHA=False; oneDNN=False", flush=True)
    pytest_args = args.pytest_args
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    return int(pytest.main(pytest_args or ["-q"]))


if __name__ == "__main__":
    raise SystemExit(main())

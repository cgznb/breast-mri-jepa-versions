#!/usr/bin/env python3
"""Separate entry point; never overwrites the upstream train.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from response_jepa.cli import main
if __name__ == "__main__":
    main()

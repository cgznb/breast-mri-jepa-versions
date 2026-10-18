#!/usr/bin/env python3
"""Verify copied preprocessing sources without importing or running them."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-syntax", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
    failures = []
    syntax_checked = 0
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            failures.append({"path": entry["path"], "reason": "unsafe manifest path"})
            continue
        path = root / relative
        if not path.is_file():
            failures.append({"path": entry["path"], "reason": "missing"})
            continue
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["archived_sha256"]:
            failures.append({"path": entry["path"], "reason": "checksum mismatch"})
        if args.check_syntax and path.suffix == ".py":
            try:
                compile(data, str(relative), "exec")
                syntax_checked += 1
            except (SyntaxError, UnicodeError) as error:
                failures.append({"path": entry["path"], "reason": str(error)})
    print(json.dumps({
        "passed": not failures,
        "files": len(manifest["files"]),
        "identifier_redacted_files": sum(
            entry["source_sha256"] != entry["archived_sha256"]
            for entry in manifest["files"]
        ),
        "python_syntax_checked": syntax_checked,
        "failures": failures,
    }, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())

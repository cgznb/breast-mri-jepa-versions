#!/usr/bin/env python3
"""Copy a historical YAML, replacing only explicitly requested resource paths."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path)
    parser.add_argument("--codec", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    args = parser.parse_args()
    source = args.config.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if source == output or output.exists():
        parser.error("--output must be a new file; historical configs are never overwritten")
    with source.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        parser.error("config must contain a YAML mapping")

    replacements: dict[str, str] = {}
    for flag, key, is_dir in (
        ("image_cache", "image_cache", True),
        ("codec", "encoder_checkpoint", False),
        ("init_checkpoint", "initialize_from_checkpoint", False),
    ):
        value = getattr(args, flag)
        if value is None:
            continue
        path = value.expanduser().resolve()
        if not (path.is_dir() if is_dir else path.is_file()):
            parser.error(f"--{flag.replace('_', '-')} requires an existing {'directory' if is_dir else 'file'}: {path}")
        replacements[key] = str(path)
    if not replacements:
        parser.error("provide at least one resource path override")

    matches: dict[str, list[tuple[dict[str, Any], str]]] = {key: [] for key in replacements}

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in matches:
                    matches[key].append((node, key))
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(config)
    for key, locations in matches.items():
        if not locations:
            parser.error(f"historical config does not define {key}; refusing to add settings")
        for node, field in locations:
            if key in {"encoder_checkpoint", "initialize_from_checkpoint"} and not node[field]:
                parser.error(f"{key} was disabled; a path override cannot change initialization design")
            if node[field] is not None and not isinstance(node[field], str):
                parser.error(f"{key} is not a path string or null")
    for key, locations in matches.items():
        for node, field in locations:
            node[field] = replacements[key]

    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against a concurrently created output.
    with output.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    print(f"Created {output}")
    for key, value in replacements.items():
        print(f"  {key}: {value}")
    print("All other parsed YAML values are unchanged. Manifest is supplied to the training CLI.")


if __name__ == "__main__":
    main()

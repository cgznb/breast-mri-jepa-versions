"""Exact source records and file metadata without persisted checksums.

Source text is small enough to compare directly. MRI/cache identities follow the
upstream path, size and modification-time contract; pixel data stay in the cache.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def record_metadata(value):
    """Copy legacy metadata while excluding checksum/commit identifier fields.

    Existing source manifests and cache indexes are still read unchanged for
    compatibility checks. Their checksum values must not enter newly written
    training metadata, checkpoints or reports.
    """
    if isinstance(value, dict):
        return {key: record_metadata(item) for key, item in value.items()
                if not any(marker in key.lower() for marker in ("checksum", "sha256", "hash", "commit"))}
    if isinstance(value, list):
        return [record_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [record_metadata(item) for item in value]
    return deepcopy(value)


def source_snapshot(root=None):
    """Return exact package filenames and source text for strict resume checks."""
    root = Path(root) if root is not None else Path(__file__).parent
    return {"schema": "response_jepa_source_snapshot_v1",
            "files": {str(path.relative_to(root)): path.read_text(encoding="utf-8")
                      for path in sorted(root.rglob("*.py"))}}

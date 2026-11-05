#!/usr/bin/env python3
"""Verify the sealed code/results bundle without reading original source paths."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();root=args.bundle_root.resolve()
    manifest=root/'SHA256SUMS'
    checked=0;total=0
    for line in manifest.read_text().splitlines():
        digest,relative=line.split('  ',1)
        path=root/relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('External link or path outside bundle')
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=digest:raise ValueError(f'SHA256 mismatch: {relative}')
        checked+=1;total+=path.stat().st_size
    forbidden={'.pt','.pth','.ckpt','.safetensors','.npz','.npy','.nii','.dcm','.xlsx','.xls','.parquet'}
    for path in root.rglob('*'):
        if path.is_symlink():raise ValueError('Unexpected symlink')
        if path.is_file() and (path.suffix.lower() in forbidden or path.name.endswith('.nii.gz')):
            raise ValueError('Training asset or weight file found in result-only bundle')
    print(json.dumps(dict(completed=True,sha256_files_verified=checked,verified_bytes=total,
                         training_arrays_or_weights_found=False,external_paths_accessed=False),indent=2))

if __name__=='__main__':main()

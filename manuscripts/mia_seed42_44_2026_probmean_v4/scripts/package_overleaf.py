#!/usr/bin/env python3
"""Package generated manuscript assets and aggregate audits without patient data."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'mia_seed42_44_2026_probmean_overleaf.zip'
files = [ROOT / name for name in [
    'main.tex', 'references.bib', 'elsarticle.cls', 'elsarticle-harv.bst',
    'build.sh', 'regenerate.sh', 'requirements_analysis.txt', 'README_ZH.md',
    'highlights.txt',
]]
for folder, suffixes in {
    'sections': {'.tex'}, 'tables': {'.tex'},
    'figures': {'.tex', '.pdf', '.png'}, 'data': {'.csv', '.json'},
    'scripts': {'.py'}, 'notes': {'.md', '.json'},
}.items():
    files.extend(p for p in (ROOT/folder).iterdir() if p.suffix in suffixes
                 and p.name != 'package_manifest.json')
files.extend(p for p in (ROOT/'sources/official').iterdir()
             if p.suffix in {'.json', '.txt'} and not p.name.startswith('mia_'))
files.extend(p for p in (ROOT/'vendor').rglob('*') if p.is_file()
             and p.suffix not in {'.log', '.aux', '.out'})
files = sorted(set(files))
for p in files:
    assert p.is_file() and p.is_relative_to(ROOT)
    assert p.suffix.lower() not in {'.npz', '.npy', '.nii', '.gz', '.pt', '.pth', '.dcm'}

csv_checks = {}
for p in (ROOT/'data').glob('*.csv'):
    with p.open(newline='') as handle:
        reader = csv.DictReader(handle)
        assert not {'patient_key', 'patient_id', 'research_id', 'pcr_probability',
                    'image_path'}.intersection(reader.fieldnames or [])
        csv_checks[p.name] = {'rows': sum(1 for _ in reader),
                              'columns': reader.fieldnames}
manifest = {
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Seeds42/44/2026: within each seed, fivefold probabilities averaged per patient then metrics; equal mean and sample SD over three seed metrics. Macro is equal landmark-metric mean; no cross-seed probability ensemble.',
    'privacy_scope': 'Aggregate CSVs only; rendered anonymous development MRI tiles in framework figure; no volumes, checkpoints, patient IDs or patient-level probability rows.',
    'csv_header_checks': csv_checks,
    'files': [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
               'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
}
manifest_path = ROOT/'notes/package_manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
with zipfile.ZipFile(TARGET, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for p in files + [manifest_path]:
        archive.write(p, str(p.relative_to(ROOT)))
print(f'Packaged {len(files)+1} files: {TARGET.name} ({TARGET.stat().st_size:,} bytes)')

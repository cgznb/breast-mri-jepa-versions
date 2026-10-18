"""Relocate resource paths without changing historical scientific settings."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


LEGACY_ROOT = Path('/root/autodl-tmp/line b')
LEGACY_VERSIONS = {
    'v1': 'breast-mri-vla-jepa',
    'encoder': 'breast-mri-vla-jepa-vq-encoder',
    'dualtask': 'breast-mri-vla-jepa-vq-dualtask',
    'evolution': 'breast-mri-vla-jepa-v1-evolution',
}


def version_directory(root, version):
    root = Path(root).resolve()
    layouts = {
        'v1': ('v1',),
        'encoder': ('vq', 'encoder'),
        'dualtask': ('vq', 'dualtask'),
        'evolution': ('v1_evolution',),
    }
    for parent in root.parents:
        candidate = parent.joinpath(*layouts[version])
        if candidate.is_dir() and (candidate / 'src/mri_vla_jepa').is_dir():
            return candidate
    # Preserve the original four sibling checkout layout.
    return root.parent / LEGACY_VERSIONS[version]


def portable_config(value, root):
    """Canonicalize only known historical locations, retaining path semantics."""
    root = Path(root).resolve()
    result = copy.deepcopy(value)
    cache = result.get('image_cache')
    if cache is not None:
        old = LEGACY_ROOT / LEGACY_VERSIONS['v1']
        try:
            suffix = Path(cache).relative_to(old)
        except ValueError:
            pass
        else:
            result['image_cache'] = str(version_directory(root, 'v1') / suffix)
    initial = result.get('initialize_from_checkpoint')
    if initial is not None:
        for legacy in LEGACY_VERSIONS.values():
            try:
                suffix = Path(initial).relative_to(LEGACY_ROOT / legacy)
            except ValueError:
                continue
            if suffix.parts and suffix.parts[0] == 'runs':
                # Inherited scripts/configs belong to the package running them;
                # the run-suite and parent names still have to match exactly.
                result['initialize_from_checkpoint'] = str(root / suffix)
            break
    return result


def prepare_jobs(planned, output, root, *, write=False, image_cache=None, codec=None):
    """Keep plans read-only; freeze executable path copies before workers start."""
    root, output = Path(root).resolve(), Path(output).resolve()
    result = copy.deepcopy(planned)
    for job in result:
        source = Path(job['config'])
        original = yaml.safe_load(source.read_text(encoding='utf-8'))
        runtime = portable_config(original, root)
        if image_cache is not None:
            runtime['image_cache'] = (None if image_cache.lower() == 'none'
                                      else str(Path(image_cache).expanduser().resolve()))
        checkpoint = runtime.get('encoder_checkpoint')
        if checkpoint is not None:
            if codec is not None:
                runtime['encoder_checkpoint'] = str(Path(codec).expanduser().resolve())
            elif checkpoint == '/root/autodl-tmp/MAM/symm-fm-world-v2/data/registered_roi32/codec.pt':
                runtime['encoder_checkpoint'] = str(root / 'weights/roi32_codec.pt')
        initial = runtime.get('initialize_from_checkpoint')
        if initial is not None:
            try:
                suffix = Path(initial).relative_to(root / 'runs')
            except ValueError:
                pass
            else:
                if len(suffix.parts) >= 3:
                    runtime['initialize_from_checkpoint'] = str(output.joinpath(*suffix.parts[1:]))
        target = output / 'input_configs' / source.name
        if runtime == original and not target.exists():
            continue
        if not write and not target.exists():
            continue
        job['source_config'] = str(source)
        job['config'] = str(target)
        if write:
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if target.exists():
                saved = yaml.safe_load(target.read_text(encoding='utf-8'))
                if saved != runtime:
                    raise ValueError('Queue resource paths changed after input-config freeze; use a new output directory')
            else:
                target.write_text(yaml.safe_dump(runtime, sort_keys=False), encoding='utf-8')
                target.chmod(0o600)
    return result

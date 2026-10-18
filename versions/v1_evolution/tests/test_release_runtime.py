"""Release relocation keeps source configs intact and binds runtime resources."""
from pathlib import Path
import importlib.util
import os
import runpy
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
release = runpy.run_path(str(ROOT / 'scripts/release_runtime.py'))


def test_runtime_copies_relocate_only_resources_and_keep_a_read_only_plan(tmp_path):
    source = ROOT / 'configs/vq_finetune_dynamic_seed17.yaml'
    before = source.read_bytes()
    original = yaml.safe_load(before)
    output = tmp_path / 'new-queue'
    plan = [{'name': 'finetune', 'config': str(source), 'output': str(output / 'finetune')}]
    assert release['prepare_jobs'](plan, output, ROOT) == plan
    assert not output.exists()
    prepared = release['prepare_jobs'](plan, output, ROOT, write=True, image_cache='none')
    runtime = yaml.safe_load(Path(prepared[0]['config']).read_text())
    assert runtime['initialize_from_checkpoint'] == str(output / 'frozen_dynamic_seed17/best.pt')
    assert runtime['image_cache'] is None
    changed = {key for key in original if original[key] != runtime[key]}
    assert changed == {'image_cache', 'initialize_from_checkpoint'}
    assert source.read_bytes() == before
    assert plan[0]['config'] == str(source)
    # A resumed queue cannot silently use another resource configuration.
    with pytest.raises(ValueError, match='resource paths changed'):
        release['prepare_jobs'](plan, output, ROOT, write=True, image_cache='/different/cache')


def test_codec_override_keeps_model_loss_and_training_protocol(tmp_path):
    source = ROOT / 'configs/vq_frozen_dynamic_seed17.yaml'
    original = yaml.safe_load(source.read_text())
    plan = [{'config': str(source), 'output': str(tmp_path / 'frozen')}]
    prepared = release['prepare_jobs'](plan, tmp_path, ROOT, write=True,
                                       codec=tmp_path / 'synthetic-codec.pt')
    runtime = yaml.safe_load(Path(prepared[0]['config']).read_text())
    assert runtime['encoder_checkpoint'] == str(tmp_path / 'synthetic-codec.pt')
    for key in original:
        if key not in {'image_cache', 'encoder_checkpoint'}:
            assert original[key] == runtime[key]


def test_detached_relative_cache_keeps_repository_root_coordinates(tmp_path, monkeypatch):
    """Parent starts at repo root; detached worker changes cwd to its package."""
    from mri_vla_jepa.train_config import RawVLATrainConfig

    spec = importlib.util.spec_from_file_location(
        'release_detach_queue', ROOT / 'scripts/run_registered_roi32_experiments.py')
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    repository = next(parent for parent in ROOT.parents
                      if (parent / 'tools/run_cpu_tests.py').is_file())
    monkeypatch.chdir(repository)
    cache = tmp_path / 'relative-cache'
    cache.mkdir()
    source = tmp_path / 'synthetic-config.yaml'
    source.write_text(yaml.safe_dump(RawVLATrainConfig(device='cpu').to_dict()))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    output = tmp_path / 'detached-run'
    monkeypatch.setattr(queue, 'jobs', lambda output, suite: [
        {'name': 'example', 'profile': 't0', 'seed': 17, 'status': 'queued',
         'config': str(source), 'output': str(output / 'example')}])
    launched = []

    class Controller:
        pid = 123456

        def __init__(self, command, **kwargs):
            assert kwargs['cwd'] == ROOT
            launched.append(command)

    monkeypatch.setattr(queue.subprocess, 'Popen', Controller)
    monkeypatch.setattr(sys, 'argv', [str(spec.origin), '--detach', '--output', str(output),
                                   '--manifest', str(manifest), '--image-cache',
                                   os.path.relpath(cache, repository)])
    assert queue.main() == 0
    command = launched[0]
    assert command[command.index('--image-cache') + 1] == str(cache)
    # Simulate the real detached cwd and verify its frozen config still agrees.
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, 'argv', command[1:])

    def worker(output, manifest, resume, concurrent, suite, budget):
        planned = queue.prepared_jobs(output, suite, write=True)
        runtime = yaml.safe_load(Path(planned[0]['config']).read_text())
        assert runtime['image_cache'] == str(cache)
        return 0

    monkeypatch.setattr(queue, 'work', worker)
    assert queue.main() == 0

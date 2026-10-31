#!/usr/bin/env python3
"""Measure fixed Medium976 workloads without starting formal training jobs.

The module stays Torch-free until apply_allocator_policy or the private worker.
All memory units are MiB, except explicitly named byte fields.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ALLOCATOR_FRACTION = .22
ALLOCATOR_CONF = 'expandable_segments:True,garbage_collection_threshold:0.2'
NON_TORCH_MIB = 1024
MARGIN_MIB = 768
SCHEMA = 'medium976_capacity_profile_v1'
TASK = Path('/projects/xts0030/public/cxx/planned_tasks/medium976_t0_t2_no_tf_cv5_multiseed_v1')
PROFILE_SCRIPT = Path('/projects/xts0030/public/cxx/planned_tasks/medium_t0_t2_no_tf_cv5_multiseed_v1/runtime/scripts/profile_capacity_phase.py')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.chmod(0o600)
    temporary.replace(path)


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def policy_identity():
    return {'schema': 'medium976_allocator_policy_v1', 'source': identity(__file__),
            'fraction': ALLOCATOR_FRACTION, 'backend': 'native',
            'configuration': ALLOCATOR_CONF, 'non_torch_mib': NON_TORCH_MIB,
            'per_job_margin_mib': MARGIN_MIB}


def profile_binding(spec):
    fields = ('schema', 'runtime_src', 'dataset_root', 'variants', 'train_seeds', 'fold_ids',
              'runtime_identities', 'config_identities', 'manifest_identities', 'asset_identities')
    return {**{key: spec[key] for key in fields},
            'allocator_policy': policy_identity(), 'profiler_source': identity(PROFILE_SCRIPT)}


def apply_allocator_policy():
    if os.environ.get('PYTORCH_CUDA_ALLOC_CONF') != ALLOCATOR_CONF:
        raise ValueError('Worker CUDA allocator environment differs from measured policy')
    import torch
    if torch.cuda.memory.get_allocator_backend() != 'native':
        raise ValueError('The capacity policy requires the native allocator')
    torch.cuda.set_per_process_memory_fraction(ALLOCATOR_FRACTION, device=0)
    getter = getattr(torch.cuda, 'get_per_process_memory_fraction', None)
    actual = getter(device=0) if getter else None
    properties = torch.cuda.get_device_properties(0)
    # The allocator stores an integer byte cap and reports cap_bytes/total.
    if actual is not None and abs(actual - ALLOCATOR_FRACTION) > 1 / properties.total_memory + 1e-15:
        raise ValueError('Allocator fraction readback differs from policy')
    return {'status': 'applied', 'policy': policy_identity(), 'actual_fraction': actual,
            'fraction_verification': 'readback_verified' if getter else 'setter_verified',
            'gpu_name': properties.name, 'total_mib': properties.total_memory / 2**20,
            'bf16_supported': bool(torch.cuda.is_bf16_supported()),
            'torch_version': torch.__version__, 'pid': os.getpid(),
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')}


def gpu_facts():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines():
        index, uuid, name, total, used, utilization = [item.strip() for item in line.split(',')]
        rows.append({'index': index, 'uuid': uuid, 'name': name, 'total_mib': float(total),
                     'used_mib': float(used), 'utilization': float(utilization)})
    return rows


def _selected_gpu(selector):
    matches = [gpu for gpu in gpu_facts() if selector in (gpu['index'], gpu['uuid'])]
    if len(matches) != 1:
        raise ValueError(f'GPU selector does not identify one visible device: {selector}')
    return matches[0]


def _verify_inputs(spec):
    grid = {(variant, seed, fold) for variant in spec['variants']
            for seed in spec['train_seeds'] for fold in spec['fold_ids']}
    actual = [(job['variant'], job['seed'], job['fold_id']) for job in spec['jobs']]
    if (spec['schema'] != 'medium976_t0_t2_no_tf_training_spec_v1'
            or not grid or len(actual) != len(grid) or set(actual) != grid
            or spec.get('training_runs') != len(grid)):
        raise ValueError('Expected the full fixed Medium976 plan')
    for key in ('runtime_identities', 'config_identities', 'manifest_identities', 'asset_identities'):
        for expected in spec[key]:
            if identity(expected['path']) != expected:
                raise ValueError(f'Capacity input changed: {expected["path"]}')


def _validate_measurements(spec, profile, binding, policy):
    if (profile.get('schema') != SCHEMA or profile.get('status') != 'passed'
            or profile.get('binding') != binding):
        raise ValueError('Capacity measurement does not match its original complete plan and policy')
    variants = profile.get('variants', {})
    if set(variants) != set(spec['variants']):
        raise ValueError('Capacity profile must cover every configured variant')
    for variant, value in variants.items():
        if (value.get('status') != 'passed' or value.get('batch_size') != 16
                or value.get('validation_batch_size') != 16 or value.get('profile_stages') != [0, 1, 2]
                or value.get('landmarks') != 't0_t2' or value.get('benchmark_seconds') != 0
                or value.get('allocator_startup', {}).get('policy') != policy
                or value.get('allocator_startup', {}).get('status') != 'applied'
                or value.get('binding') != binding):
            raise ValueError(f'Incomplete or stale full-batch profile: {variant}')
        peaks = [value.get(key) for key in ('training_peak_reserved_mib', 'inference_peak_reserved_mib',
                                           'diagnostic_peak_reserved_mib', 'peak_reserved_mib', 'nvml_peak_mib')]
        if any(type(number) not in (float, int) or not math.isfinite(number) or number <= 0 for number in peaks):
            raise ValueError(f'Missing measured memory stage: {variant}')
        world = value.get('world_profile', {})
        expected_world = 'not_applicable' if variant == 'direct_no_jepa' else 'passed'
        if world.get('status') != expected_world:
            raise ValueError(f'Missing world evaluation capacity: {variant}')
        if expected_world == 'passed' and (world.get('precision') != 'fp32' or world.get('batch_size') != 16):
            raise ValueError('World profile did not use the full FP32 validation batch')
        minimum = math.ceil(max(max(peaks), value['allocator_startup']['total_mib'] * ALLOCATOR_FRACTION
                                + NON_TORCH_MIB) + MARGIN_MIB)
        if (value.get('reservation_mib', 0) < minimum
                or value.get('host_reservation_mib', 0) < value.get('host_peak_mib', math.inf)
                or value.get('checkpoint_bytes', 0) <= 0 or value.get('cpu_slots', 0) <= 0):
            raise ValueError(f'Capacity reservation understates measured resources: {variant}')


def _inheritance_helper():
    source = Path(__file__).with_name('capacity_reuse.py')
    module_spec = importlib.util.spec_from_file_location('_medium976_capacity_reuse', source)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _validate_evidence(spec, profile):
    _verify_inputs(spec)
    if (profile.get('schema') != SCHEMA or profile.get('status') != 'passed'
            or profile.get('binding') != profile_binding(spec)):
        raise ValueError('Capacity profile does not match the complete current plan and allocator policy')
    if profile.get('evidence_kind') == 'inherited':
        legacy_spec, legacy_profile = _inheritance_helper().validate_inheritance(
            spec, profile, current_binding=profile_binding(spec), current_policy=policy_identity(),
            current_source=Path(__file__))
        _validate_measurements(legacy_spec, legacy_profile, legacy_profile['binding'],
                               legacy_profile['binding']['allocator_policy'])
    elif profile.get('evidence_kind') in (None, 'measured'):
        _validate_measurements(spec, profile, profile_binding(spec), policy_identity())
    else:
        raise ValueError('Unknown capacity evidence kind')


def validate_profile(spec, profile, gpus):
    _validate_evidence(spec, profile)
    if not gpus:
        raise ValueError('Capacity validation requires at least one selected GPU')
    measured = profile['measured_gpu']
    for selector in gpus:
        observed = _selected_gpu(str(selector))
        if (observed['name'] != measured['name'] or observed['total_mib'] != measured['total_mib']):
            raise ValueError('Capacity was measured on a different GPU model or physical capacity')


def reservation_for(profile, variant, stage):
    if stage not in ('train', 'eval', 'evaluate', 'validation', 'test', 'profile'):
        raise ValueError(f'Unknown capacity stage: {stage}')
    if profile.get('evidence_kind') == 'inherited':
        reference = profile.get('inheritance', {}).get('current_spec', {})
        if identity(reference.get('path', '')) != reference:
            raise ValueError('Inherited capacity belongs to a changed current spec')
        _validate_evidence(read(reference['path']), profile)
    value = profile['variants'][variant]
    if profile.get('status') != 'passed' or value.get('status') != 'passed':
        raise ValueError('Unpassed capacity profile cannot admit workers')
    return {'gpu_mib': value['reservation_mib'], 'host_mib': value['host_reservation_mib'],
            'cpu_slots': value['cpu_slots'], 'checkpoint_bytes': value['checkpoint_bytes'],
            'loader_threads': value['loader_threads']}


def _worker(args):
    spec = read(args.spec)
    _verify_inputs(spec)
    startup = apply_allocator_policy()
    sys.path.insert(0, spec['runtime_src'])
    import torch
    from mri_vla_jepa import data
    from mri_vla_jepa.t0_cv_worker import install_paired_readout_initialization, _profile_world, checkpoint_estimate
    from mri_vla_jepa.train_config import load_config
    if not Path(data.__file__).resolve().is_relative_to(Path(spec['runtime_src']).resolve()):
        raise ValueError('Capacity worker imported the wrong data runtime')
    install_paired_readout_initialization()
    job = next(job for job in spec['jobs'] if job['name'] == args.job)
    cfg = load_config(job['config'])
    if cfg.batch_size != 16 or cfg.validation_batch_size != 16 or cfg.image_cache is not None:
        raise ValueError('Profile requires unchanged full-batch noncached configuration')
    module_spec = importlib.util.spec_from_file_location('_medium976_original_profiler', PROFILE_SCRIPT)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    began = time.monotonic()
    result = module.profile(job['config'], job['manifest'], args.worker_output.with_suffix('.stages.json'),
                            benchmark_seconds=0)
    torch.cuda.empty_cache()
    result.update(_profile_world(cfg, job['manifest']))
    result['peak_reserved_mib'] = max(result['peak_reserved_mib'], result['world_peak_reserved_mib'])
    result['peak_allocated_mib'] = max(result['peak_allocated_mib'], result['world_peak_allocated_mib'])
    checkpoint = checkpoint_estimate(result['total_parameters'], result['trainable_parameters'])
    host_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    # Loader threads prefetch full input and full future tensors. Include both
    # in-flight construction and queued batches beyond the sequential probe RSS.
    batch_mib = cfg.batch_size * 4 * 3 * math.prod(cfg.image_shape) * 4 * 2 / 2**20
    prefetch_mib = (cfg.prefetch_batches + cfg.loader_workers + 1) * batch_mib
    result.update(binding=profile_binding(spec), allocator_startup=startup,
                  schema=SCHEMA, variant=job['variant'], manifest=job['manifest'],
                  validation_batch_size=cfg.validation_batch_size,
                  host_peak_mib=host_peak, checkpoint_bytes=checkpoint,
                  prefetch_headroom_mib=prefetch_mib,
                  host_reservation_mib=math.ceil(max(host_peak, 2 * checkpoint / 2**20 + 1024) + prefetch_mib),
                  cpu_slots=cfg.threads, loader_threads=cfg.loader_workers,
                  host_formula='ceil(max(measured RSS, 2*checkpoint MiB+1024) + (prefetch+loaders+1)*input_and_future_batch MiB)',
                  total_profile_seconds=time.monotonic() - began,
                  measurement_scope='real train patients only; unchanged training/FP32 pCR inference/diagnostics/world; no checkpoint writes')
    write(args.worker_output, result)


def _process_observations(uuid):
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    records = []
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(',')]
        if len(fields) == 3 and fields[0] == uuid:
            records.append({'pid': int(fields[1]), 'used_mib': float(fields[2])})
    return records


def _collect(args):
    spec = read(args.spec)
    _verify_inputs(spec)
    gpu = _selected_gpu(args.gpu)
    output = args.output or Path(spec['task_dir']) / 'capacity_profile.json'
    if Path(output).resolve().is_relative_to(Path(spec['output_root']).resolve()):
        raise ValueError('Capacity output must stay outside formal training output_root')
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    requested = args.variants or spec['variants']
    if not requested or not set(requested) <= set(spec['variants']):
        raise ValueError('Unknown capacity variants')
    profile = {'schema': SCHEMA, 'status': 'profiling', 'binding': profile_binding(spec),
               'measured_gpu': gpu, 'variants': {}, 'created_at_utc': datetime.now(timezone.utc).isoformat()}
    if output.exists():
        previous = read(output)
        if previous.get('evidence_kind') == 'inherited':
            validate_profile(spec, previous, [gpu['uuid']])
            print(str(output), flush=True)
            return
        if previous.get('binding') != profile_binding(spec):
            raise ValueError('Existing capacity output has another binding; choose a new private output')
        if previous['measured_gpu']['uuid'] != gpu['uuid']:
            raise ValueError('Incremental capacity measurements must use the original GPU')
        profile = previous
    lock_path = Path('/tmp') / f'medium976_capacity_{gpu["uuid"]}.lock'
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for variant in requested:
            if profile['variants'].get(variant, {}).get('status') == 'passed':
                continue
            baseline = _selected_gpu(gpu['uuid'])
            if _process_observations(gpu['uuid']) or baseline['used_mib'] > 128 or baseline['utilization'] > 0:
                raise RuntimeError('Capacity measurement requires an idle GPU with no compute processes')
            job = next(job for job in spec['jobs'] if job['variant'] == variant)
            directory = output.parent / (output.stem + '.workers')
            directory.mkdir(exist_ok=True, mode=0o700)
            child_output = directory / f'{variant}.json'
            environment = os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=gpu['uuid'], PYTHONPATH=spec['runtime_src'],
                               PYTORCH_CUDA_ALLOC_CONF=ALLOCATOR_CONF,
                               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            for key in list(environment):
                if key.startswith('MRI_CXX_ALLOCATOR_'):
                    environment.pop(key)
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--spec', str(args.spec),
                       '--job', job['name'], '--worker-output', str(child_output)]
            telemetry = []
            contamination = False
            with (directory / f'{variant}.log').open('a') as log:
                child = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                while True:
                    observed = _selected_gpu(gpu['uuid'])
                    processes = _process_observations(gpu['uuid'])
                    contamination |= len(processes) > 1
                    telemetry.append({'elapsed_seconds': time.monotonic(), 'used_mib': observed['used_mib'],
                                      'utilization': observed['utilization'], 'processes': processes})
                    if child.poll() is not None:
                        break
                    time.sleep(.5)
            write(directory / f'{variant}.telemetry.json', telemetry)
            if child.returncode != 0 or contamination or not child_output.exists():
                failure = {'status': 'failed', 'returncode': child.returncode, 'foreign_process_detected': contamination,
                           'log': str(directory / f'{variant}.log')}
                profile['variants'][variant] = failure
                profile['status'] = 'failed'
                write(output, profile)
                raise RuntimeError(f'{variant} capacity probe failed; inspect {failure["log"]}')
            result = read(child_output)
            result['nvml_peak_mib'] = max(item['used_mib'] for item in telemetry)
            result['nvml_scope'] = 'whole-device memory, conservative upper bound; GPU empty at start and <=1 compute process throughout'
            result['nvml_baseline_mib'] = baseline['used_mib']
            result['telemetry_file'] = str(directory / f'{variant}.telemetry.json')
            result['telemetry_interval_seconds'] = .5
            result['reservation_mib'] = math.ceil(max(result['nvml_peak_mib'], result['peak_reserved_mib'],
                                                      result['allocator_startup']['total_mib'] * ALLOCATOR_FRACTION
                                                      + NON_TORCH_MIB) + MARGIN_MIB)
            result['reservation_formula'] = 'ceil(max(full-phase PyTorch/device-NVML peak, CUDA usable total*.22+1024)+768)'
            profile['variants'][variant] = result
            profile['status'] = 'passed' if set(profile['variants']) == set(spec['variants']) else 'partial'
            write(output, profile)
            print(json.dumps({'variant': variant, 'status': result['status'], 'gpu_mib': result['reservation_mib'],
                              'host_mib': result['host_reservation_mib'],
                              'seconds': result['total_profile_seconds']}), flush=True)
        if profile['status'] == 'passed':
            validate_profile(spec, profile, [gpu['uuid']])
        print(str(output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, default=TASK / 'training_spec.json')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--variants', nargs='+')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--job', help=argparse.SUPPRESS)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _worker(args)
    else:
        _collect(args)


if __name__ == '__main__':
    main()

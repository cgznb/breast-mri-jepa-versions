#!/usr/bin/env python3
"""CPU-only independent checks for the fixed-fold Medium976 training entry.

No training, evaluation, GPU probing or worker launch is allowed by this check.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime, timezone
import importlib.util
import io
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path('/personal_fast_storage/VLA-PCRV1')
TASK = Path('/projects/xts0030/public/cxx/planned_tasks/medium976_t0_t2_no_tf_cv5_multiseed_v1')
OLD_TASK = Path('/projects/xts0030/public/cxx/planned_tasks/medium_t0_t2_no_tf_cv5_multiseed_v1')
AUDIT = ROOT / 'private_outputs/medium976_split_audit_20261028'
VARIANTS = {
    'forecast_attention': ('forecast', 'attention', .5),
    'direct_no_jepa': ('direct', 'mean', 0.),
    'direct_readout': ('direct', 'mean', .5),
    'forecast_mean': ('forecast', 'mean', .5),
}
SEEDS = [21, 44, 203]
JOB_COUNT = 60


def read(path):
    return json.loads(Path(path).read_text())


def info(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def check_identity(item):
    path = Path(item['path']).resolve()
    stat = path.stat()
    assert stat.st_size == item['bytes'], f'File size changed: {path}'
    assert stat.st_mtime_ns == item['mtime_ns'], f'File modification time changed: {path}'


def denied(*args, **kwargs):
    raise AssertionError('CPU preflight attempted a worker, training, evaluation or GPU operation')


def verify_paired_initialization():
    import torch
    from mri_vla_jepa.model import RawVLAJEPA
    from mri_vla_jepa.model_config import RawVLAJEPAConfig
    from mri_vla_jepa.t0_cv_worker import install_paired_readout_initialization
    install_paired_readout_initialization()
    config = RawVLAJEPAConfig(dim=16, heads=2, encoder_depth=1, fusion_depth=1,
                             predictor_depth=1, token_grid=(1, 1, 1), state_queries=2,
                             pcr_queries=1, fm_channels=2, encoder_type='raw',
                             enable_flow=False, enable_world_model=True,
                             fixed_transition_queries=True,
                             pcr_readout='forecast', spatial_readout='mean')
    torch.manual_seed(2026)
    forecast = RawVLAJEPA(config)
    forecast_rng = torch.get_rng_state().clone()
    torch.manual_seed(2026)
    direct = RawVLAJEPA(replace(config, pcr_readout='direct'))
    assert torch.equal(forecast_rng, torch.get_rng_state())
    fparams, dparams = forecast.state_dict(), direct.state_dict()
    shared = sorted(k for k in fparams if not k.startswith('pcr_head.'))
    assert shared and all(k in dparams and torch.equal(fparams[k], dparams[k]) for k in shared)
    assert all(t.device.type == 'cpu' for t in (*fparams.values(), *dparams.values()))
    return {'tiny_CPU_constructor_only': True, 'shared_state_tensors_equal': len(shared),
            'post_constructor_CPU_rng_equal': True, 'forward_or_training_called': False}


def verify_cli_contracts(runner, spec_path):
    """Parse the real CLI while denying all launch and GPU discovery paths."""
    parsed = []
    original_parse = argparse.ArgumentParser.parse_args

    def capture_parse(parser, *args, **kwargs):
        arguments = original_parse(parser, *args, **kwargs)
        parsed.append(arguments)
        return arguments

    with patch.object(argparse.ArgumentParser, 'parse_args', capture_parse), \
         patch.object(runner, 'preflight', lambda spec, selected: {'CPU_CLI_probe': True}), \
         patch.object(runner, 'resolve_gpus', denied), \
         patch.object(runner, 'controller', denied), \
         patch('subprocess.Popen', denied), patch('subprocess.check_output', denied):
        for extra, expected in (([], 60), (['--max-active', '5'], 5), (['--max-active', '120'], 120)):
            result = runner.main(['--spec', str(spec_path), '--preflight-only', '--gpus', '0', *extra])
            assert result == {'CPU_CLI_probe': True}
            assert parsed[-1].max_active == expected and parsed[-1].gpus == ['0']
        for invalid in ('0', '-1'):
            try:
                with redirect_stderr(io.StringIO()):
                    runner.main(['--spec', str(spec_path), '--preflight-only', '--max-active', invalid])
            except (ValueError, SystemExit):
                pass
            else:
                raise AssertionError(f'Nonpositive --max-active accepted: {invalid}')
    return {'default_max_active': 60, 'one_GPU_max_active_5_accepted': True,
            'custom_ceiling_above_planned_job_count_accepted': True,
            'nonpositive_max_active_rejected': True, 'GPU_discovery_or_launch_called': False}


@contextmanager
def CPU_allocator_calls(capacity, torch):
    """Execute the real policy with only physical CUDA operations substituted."""
    calls = []

    def set_fraction(fraction, device):
        assert fraction == capacity.ALLOCATOR_FRACTION and device == 0
        calls.append({'fraction': fraction, 'logical_device': device})

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, PYTORCH_CUDA_ALLOC_CONF=capacity.ALLOCATOR_CONF,
                                       CUDA_VISIBLE_DEVICES='CPU-FAKE-GPU'))
        stack.enter_context(patch.object(torch.cuda.memory, 'get_allocator_backend', lambda: 'native'))
        stack.enter_context(patch.object(torch.cuda, 'set_per_process_memory_fraction', set_fraction))
        stack.enter_context(patch.object(torch.cuda, 'get_per_process_memory_fraction',
                                         lambda device: capacity.ALLOCATOR_FRACTION, create=True))
        stack.enter_context(patch.object(torch.cuda, 'get_device_properties',
                                         lambda device: SimpleNamespace(name='CPU-FAKE-GPU', total_memory=30000 * 2**20)))
        stack.enter_context(patch.object(torch.cuda, 'is_bf16_supported', lambda: True))
        stack.enter_context(patch.object(torch.cuda, 'init', denied))
        stack.enter_context(patch.object(torch.cuda, '_lazy_init', denied))
        yield calls


def capacity_fixture(capacity, spec):
    """Synthetic full-phase measurements: never eligible for formal training."""
    binding = capacity.profile_binding(spec)
    gpu = {'index': '0', 'uuid': 'GPU-CPU-FAKE-0', 'name': 'CPU-FAKE-GPU',
           'total_mib': 30000, 'used_mib': 0, 'utilization': 0}
    variants = {}
    for variant in spec['variants']:
        variants[variant] = {
            'status': 'passed', 'binding': binding, 'batch_size': 16, 'validation_batch_size': 16,
            'profile_stages': [0, 1, 2], 'landmarks': 't0_t2', 'benchmark_seconds': 0,
            'allocator_startup': {'status': 'applied', 'policy': capacity.policy_identity(), 'total_mib': 30000},
            'training_peak_reserved_mib': 7000, 'inference_peak_reserved_mib': 7100,
            'diagnostic_peak_reserved_mib': 7200, 'peak_reserved_mib': 7200, 'nvml_peak_mib': 7300,
            'world_profile': ({'status': 'not_applicable'} if variant == 'direct_no_jepa'
                              else {'status': 'passed', 'precision': 'fp32', 'batch_size': 16}),
            'reservation_mib': 10000, 'host_peak_mib': 256, 'host_reservation_mib': 512,
            'cpu_slots': 4, 'checkpoint_bytes': 1024, 'loader_threads': 2,
        }
    return {'schema': capacity.SCHEMA, 'status': 'passed', 'binding': binding,
            'measured_gpu': gpu, 'variants': variants}


def verify_capacity_contracts(runner, spec):
    capacity = runner.capacity
    profile = capacity_fixture(capacity, spec)
    facts = [profile['measured_gpu']]
    with patch.object(capacity, 'gpu_facts', lambda: facts), patch('subprocess.run', denied), \
         patch('subprocess.check_output', denied):
        capacity.validate_profile(spec, profile, ['0'])
        for variant in spec['variants']:
            for stage in ('train', 'evaluate'):
                assert capacity.reservation_for(profile, variant, stage)['gpu_mib'] == 10000
        for label, mutate in (
            ('stale_plan_binding', lambda p: p['binding'].update(train_seeds=[-1])),
            ('stale_allocator_binding', lambda p: p['binding']['allocator_policy'].update(fraction=.9)),
            ('understated_reservation', lambda p: p['variants']['forecast_mean'].update(reservation_mib=1)),
            ('missing_variant', lambda p: p['variants'].pop('direct_readout')),
            ('missing_diagnostic_phase', lambda p: p['variants']['forecast_attention'].pop('diagnostic_peak_reserved_mib')),
        ):
            malformed = json.loads(json.dumps(profile))
            mutate(malformed)
            try:
                capacity.validate_profile(spec, malformed, ['0'])
            except ValueError:
                pass
            else:
                raise AssertionError(f'Invalid capacity profile accepted: {label}')
        try:
            with patch.object(capacity, 'gpu_facts', lambda: [dict(facts[0], total_mib=40000)]):
                capacity.validate_profile(spec, profile, ['0'])
        except ValueError:
            pass
        else:
            raise AssertionError('A different physical GPU capacity accepted the profile')
    return {'real_profile_validation_and_reservation_functions_called': True,
            'complete_four_variant_full_phase_fixture_accepted': True,
            'stale_plan_or_allocator_binding_rejected': True,
            'understated_peak_or_missing_phase_or_variant_rejected': True,
            'different_GPU_capacity_rejected': True, 'physical_GPU_queried': False}


def verify_capacity_refresh_cli(runner, spec, spec_path):
    """A passed profile for another GPU must be archived and measured anew."""
    fresh = capacity_fixture(runner.capacity, spec)
    stale = json.loads(json.dumps(fresh))
    stale['measured_gpu']['total_mib'] = 40000
    assert stale['status'] == 'passed' and stale['binding'] == fresh['binding']
    measured, dispatched = [], []
    with tempfile.TemporaryDirectory(prefix='medium976_refresh_CPU_') as directory:
        profile_path = Path(directory) / 'capacity_profile.json'
        profile_path.write_text(json.dumps(stale) + '\n')

        def fake_measurement(command, *, check):
            assert check is True and command[2] == str(Path(runner.capacity.__file__).resolve())
            assert command[command.index('--spec') + 1] == str(spec_path.resolve())
            assert command[command.index('--output') + 1] == str(profile_path.resolve())
            assert not profile_path.exists(), 'An invalid passed profile would be reused by the profiler'
            archived = list(profile_path.parent.glob('capacity_profile.superseded.*.json'))
            assert len(archived) == 1 and read(archived[0]) == stale
            profile_path.write_text(json.dumps(fresh) + '\n')
            measured.append(command)
            return SimpleNamespace(returncode=0)

        def fake_controller(actual_spec, selected, gpus, max_active, actual_path, **kwargs):
            assert actual_spec == spec and len(selected) == JOB_COUNT
            assert gpus == [fresh['measured_gpu']['uuid']] and max_active == 5
            assert actual_path == spec_path.resolve() and kwargs['capacity_profile'] == fresh
            dispatched.append(True)

        with patch.object(runner, 'preflight', lambda actual, selected: {'CPU_CLI_probe': True}), \
             patch.object(runner, 'resolve_gpus', lambda selected: [fresh['measured_gpu']['uuid']]), \
             patch.object(runner.capacity, 'gpu_facts', lambda: [fresh['measured_gpu']]), \
             patch.object(runner, 'controller', fake_controller), \
             patch('subprocess.run', fake_measurement), patch('subprocess.check_output', denied), \
             patch('subprocess.Popen', denied), redirect_stdout(io.StringIO()):
            runner.main(['--spec', str(spec_path), '--gpus', '0', '--max-active', '5',
                         '--capacity-profile', str(profile_path)])
        assert len(measured) == len(dispatched) == 1 and read(profile_path) == fresh
    return {'passed_profile_with_same_plan_binding_but_invalid_GPU_archived': True,
            'replacement_profile_revalidated_before_controller': True,
            'one_GPU_max_active_5_reached_controller': True,
            'real_profile_validation_called': True, 'actual_profiling_or_training_called': False}


def verify_allocator_contracts(capacity, torch):
    receipt = capacity.apply_allocator_policy()
    assert receipt['status'] == 'applied' and receipt['policy'] == capacity.policy_identity()
    assert receipt['actual_fraction'] == capacity.ALLOCATOR_FRACTION
    for label in ('wrong_environment', 'wrong_backend', 'wrong_fraction_readback'):
        with ExitStack() as stack:
            if label == 'wrong_environment':
                stack.enter_context(patch.dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='CPU-invalid-policy'))
            elif label == 'wrong_backend':
                stack.enter_context(patch.object(torch.cuda.memory, 'get_allocator_backend', lambda: 'cudaMallocAsync'))
            else:
                stack.enter_context(patch.object(torch.cuda, 'get_per_process_memory_fraction', lambda device: .9))
            try:
                capacity.apply_allocator_policy()
            except ValueError:
                pass
            else:
                raise AssertionError(f'Invalid allocator policy accepted: {label}')
    return {'real_policy_function_called': True, 'fraction': capacity.ALLOCATOR_FRACTION,
            'logical_device': 0, 'readback_checked': True,
            'wrong_environment_backend_or_readback_rejected': True,
            'CUDA_operations_substituted_with_CPU_stubs': True}


def verify_controller_gates(runner, spec):
    """Exercise controller decisions with fake receipts/processes in temporary dirs."""
    results = {}
    with tempfile.TemporaryDirectory(prefix='medium976_entry_CPU_') as directory:
        def isolated(case):
            trial = json.loads(json.dumps(spec))
            output = Path(directory) / case
            trial['output_root'] = str(output)
            for job in trial['jobs']:
                job['output'] = str(output / 'runs' / job['name'])
            return trial, output

        names = [j['name'] for j in spec['jobs']]
        for label, trained, evaluated, expected_visible, expected_status in (
            ('59_trained', set(names[:-1]), set(), False, 'training_pending'),
            ('60_trained_59_evaluated', set(names), set(names[:-1]), False, 'evaluation_pending'),
            ('60_trained_60_evaluated', set(names), set(names), True, 'completed'),
        ):
            trial, output = isolated(label)
            with patch.object(runner, 'training_complete', lambda job, *unused: job['name'] in trained), \
                 patch.object(runner, 'evaluation_complete', lambda job, *unused: job['name'] in evaluated), \
                 patch.object(runner.admission, 'gpu_snapshot', denied), \
                 patch.object(runner.admission, 'effective_host', denied), \
                 patch('subprocess.Popen', denied), redirect_stdout(io.StringIO()):
                # A completed selected job must not shrink the full 60-job gate.
                runner.controller(trial, [trial['jobs'][0]], ['0'], 1, Path(directory) / 'unused_spec.json')
            matrix = read(output / 'matrix.json')
            assert matrix['test_visible'] is expected_visible and matrix['status'] == expected_status
            assert (output / 'results_index.json').exists() is expected_visible
            results[label] = {'test_visible': expected_visible, 'status': expected_status,
                              'filtered_selection_did_not_reduce_60_job_gate': True}
        for gpu_count, max_active, foreign_mib, expected_peak in (
            (1, 5, 0, 2), (2, 5, 0, 4), (2, 3, 0, 3), (2, 120, 12000, 3),
        ):
            label = f'{gpu_count}GPU_global{max_active}_foreign{foreign_mib}'
            trial, output = isolated(label)
            trained, active, peaks = set(), Counter(), Counter()
            peak_total, cross_seed_overlap = [0], [False]
            next_pid = [8000000]
            processes, launches, snapshots = {}, [], []
            gpus = [f'GPU-CPU-FAKE-{i}' for i in range(gpu_count)]
            jobs = [trial['jobs'][i] for i in (0, 5, 10, 15, 20, 25, 30, 35, 40, 45)]
            by_name = {job['name']: job for job in jobs}
            profile = capacity_fixture(runner.capacity, trial)

            class FakeOwnership:
                def __init__(self, spec_path):
                    self.spec_path = spec_path

                def record(self, pid, command, gpu, attempt):
                    assert attempt and command[command.index('--spec') + 1] == str(self.spec_path)
                    return {'pid': pid, 'command': command, 'gpu_uuid': gpu, 'attempt': attempt,
                            'start_ticks': pid}

                def account(self, snapshot, records):
                    assert len(records) == sum(active.values())
                    return {'owned_mib': {gpu: 0 for gpu in snapshot},
                            'foreign_mib': {gpu: item['used_mib'] for gpu, item in snapshot.items()},
                            'evidence': []}

            def fake_snapshot(selected):
                assert selected == gpus
                sample = {gpu: {'total_mib': 30000,
                                'used_mib': foreign_mib if gpu == gpus[0] else 0,
                                'processes': []} for gpu in selected}
                snapshots.append(sample)
                return sample

            class FakeProcess:
                def __init__(self, command, *, env, **kwargs):
                    self.name = command[command.index('--worker') + 1]
                    assert command[command.index('--worker-stage') + 1] == 'train'
                    self.gpu = env['CUDA_VISIBLE_DEVICES']
                    assert env['PYTHONPATH'] == trial['runtime_src']
                    assert env['PYTORCH_CUDA_ALLOC_CONF'] == runner.capacity.ALLOCATOR_CONF
                    assert env['MEDIUM976_ATTEMPT_ID']
                    for process in processes.values():
                        if not process.finished and by_name[process.name]['seed'] != by_name[self.name]['seed']:
                            cross_seed_overlap[0] = True
                    next_pid[0] += 1
                    self.pid, self.finished = next_pid[0], False
                    self.polls_remaining = (2, 4, 6, 8)[len(launches) % 4]
                    processes[self.pid] = self
                    launches.append(self.name)
                    active[self.gpu] += 1
                    peaks[self.gpu] = max(peaks[self.gpu], active[self.gpu])
                    peak_total[0] = max(peak_total[0], sum(active.values()))
                    assert active[self.gpu] <= (1 if foreign_mib and self.gpu == gpus[0] else 2)
                    assert sum(active.values()) <= max_active

                def poll(self):
                    if not self.finished:
                        self.polls_remaining -= 1
                        if self.polls_remaining > 0:
                            return None
                        self.finished = True
                        active[self.gpu] -= 1
                        trained.add(self.name)
                    return 0

            with patch.object(runner, 'training_complete', lambda job, *unused: job['name'] in trained), \
                 patch.object(runner, 'evaluation_complete', lambda job, *unused: False), \
                 patch.object(runner.admission, 'GPUOwnership', FakeOwnership), \
                 patch.object(runner.admission, 'gpu_snapshot', fake_snapshot), \
                 patch.object(runner.admission, 'effective_host', lambda: {
                     'cpu_slots': 120, 'total_mib': 1000000, 'available_mib': 1000000}), \
                 patch('subprocess.check_output', denied), patch('subprocess.run', denied), \
                 patch('subprocess.Popen', FakeProcess), redirect_stdout(io.StringIO()):
                runner.controller(trial, jobs, gpus, max_active, Path(directory) / 'unused_spec.json',
                                  capacity_profile=profile, poll_seconds=0)
            assert len(trained) == len(jobs) and peak_total[0] == expected_peak
            assert max(peaks.values()) == 2 and not sum(active.values()) and cross_seed_overlap[0]
            assert len(launches) == len(set(launches)) == len(jobs) and snapshots
            matrix = read(output / 'matrix.json')
            assert matrix['test_visible'] is False and matrix['status'] == 'training_pending'
            assert matrix['capacity_profile'] == profile
            assert matrix['scheduling']['max_active'] == max_active
            assert matrix['scheduling']['allocator_policy'] == runner.capacity.policy_identity()
            assert not (output / 'results_index.json').exists()
            results[label] = {
                'completed_fake_jobs': len(jobs), 'maximum_total_active': peak_total[0],
                'maximum_active_per_GPU': dict(peaks), 'cross_seed_rolling_refill': True,
                'real_peak_reservation_and_GPU_admission_called': True,
                'capacity_profile_and_allocator_policy_recorded': True,
                'actual_processes_launched': False, 'physical_GPU_queried': False}
    return results


def verify_run_and_evaluation_contracts(runner, spec, training_module):
    """Use disposable marker files and stub training to verify recovery gates."""
    results = {}
    with tempfile.TemporaryDirectory(prefix='medium976_contract_CPU_') as directory:
        def isolated(label):
            trial = json.loads(json.dumps(spec))
            trial['output_root'] = str(Path(directory) / label)
            for job in trial['jobs']:
                job['output'] = str(Path(trial['output_root']) / 'runs' / job['name'])
            job = trial['jobs'][0]
            run = Path(job['output'])
            run.mkdir(parents=True)
            return trial, job, run

        def save(path, obj):
            path.write_text(json.dumps(obj) + '\n')

        def must_reject(call, expected_types=(ValueError, RuntimeError)):
            try:
                with redirect_stderr(io.StringIO()):
                    call()
            except expected_types:
                return
            raise AssertionError('An invalid training/evaluation contract was accepted')

        for label, existing in (('old_checkpoint_without_contract', 'last.pt'),
                                ('best_only_partial_checkpoint', 'best.pt')):
            trial, job, run = isolated(label)
            (run / existing).write_bytes(b'CPU test marker, not a checkpoint')
            if label.startswith('best_only'):
                save(run / 'run_identity.json', runner.run_identity(trial, job))
            with patch.object(training_module, 'train', denied):
                must_reject(lambda: runner.worker(trial, job['name'], 'train'))
            results[label] = 'rejected before training'
        for label in ('changed_allocator_policy', 'changed_admission_source'):
            trial, job, run = isolated(label)
            contract = runner.run_identity(trial, job)
            assert contract['allocator_policy'] == runner.capacity.policy_identity()
            assert contract['admission_identity'] == runner.identity(Path(runner.admission.__file__))
            if label == 'changed_allocator_policy':
                contract['allocator_policy'] = dict(contract['allocator_policy'], CPU_verifier_changed=True)
            else:
                contract['admission_identity']['mtime_ns'] += 1
            save(run / 'run_identity.json', contract)
            (run / 'last.pt').write_bytes(b'CPU test marker, not a checkpoint')
            with patch.object(training_module, 'train', denied):
                must_reject(lambda: runner.worker(trial, job['name'], 'train'))
            results[label] = 'rejected before training'
        for label, resume, completed in (('fresh', False, True), ('resume', True, True), ('paused', False, False)):
            trial, job, run = isolated(label)
            if resume:
                save(run / 'run_identity.json', runner.run_identity(trial, job))
                (run / 'last.pt').write_bytes(b'CPU test marker, not a checkpoint')
            calls = []

            def fake_train(store, cfg, output, *, resume=False):
                calls.append(resume)
                report = {'completed': completed, 'landmarks': 't0_t2', 'best_selection_score': .5}
                save(Path(output) / 'train_report.json', report)
                for name in ('best.pt', 'last.pt'):
                    (Path(output) / name).write_bytes(b'CPU test marker, not a checkpoint')
                return report

            with patch.object(training_module, 'train', fake_train), patch.object(training_module, 'evaluate', denied):
                if completed:
                    runner.worker(trial, job['name'], 'train')
                else:
                    must_reject(lambda: runner.worker(trial, job['name'], 'train'))
            assert calls == [resume]
            assert (run / 'training_done.json').exists() is completed
            assert runner.training_complete(job, trial) is completed
            assert (run / 'allocator_policy_train.json').is_file()
            results[label] = {'resume_flag': resume, 'completed_receipt_written': completed,
                              'allocator_receipt_written': True, 'real_training_called': False}
        trial, job, run = isolated('evaluation_receipts')
        save(run / 'train_report.json', {'completed': True, 'landmarks': 't0_t2', 'best_selection_score': .5})
        save(run / 'run_identity.json', runner.run_identity(trial, job))
        for filename in ('best.pt', 'last.pt'):
            (run / filename).write_bytes(b'CPU test marker, not a checkpoint')
        save(run / 'training_done.json', {'completed': True, 'run_identity': runner.identity(run / 'run_identity.json'),
             'files': [runner.identity(run / filename) for filename in ('train_report.json', 'best.pt', 'last.pt')]})
        patients = read(job['manifest'])['patients']
        report_paths = []
        reports = {}
        for label in ('best_val', 'last_val', 'best_test'):
            split = 'test' if label == 'best_test' else 'val'
            report = {'split': split, 'landmarks': 't0_t2', 'checkpoint_landmarks': 't0_t2',
                      'synthetic': False, 'metric_name': 'f1', 'f1_threshold': .5, 'selection_score': .5,
                      'rows': [{'patient_key': p['patient_key'], 'landmark': stage, 'pcr': p['target']['pcr'],
                                'pcr_probability': .5} for p in patients if p['split'] == split
                               for stage, visit in enumerate(p['visits'][:3]) if visit is not None]}
            reports[label] = report
            target = run / f'evaluation_{label}.json'
            save(target, report)
            report_paths.append(target)
            runner.validate_evaluation_report(job, label, report)
        receipt = {'completed': True, 'job': job['name'], 'manifest': runner.identity(job['manifest']),
                   'reports': [runner.identity(p) for p in report_paths],
                   'best_checkpoint': runner.identity(run / 'best.pt'),
                   'last_checkpoint': runner.identity(run / 'last.pt'),
                   'training_done': runner.identity(run / 'training_done.json'),
                   'run_identity': runner.identity(run / 'run_identity.json')}
        save(run / 'evaluation_done.json', receipt)
        with patch.object(runner, 'training_complete', lambda job, *unused: True):
            runner.EVALUATION_CACHE.clear()
            assert runner.evaluation_artifacts_complete(job)
            assert not runner.evaluation_complete(job, trial), 'Reports alone cannot establish best-only completion'
            runner.finalize_checkpoint_retention(trial, job)
            assert runner.evaluation_complete(job, trial)
            save(run / 'evaluation_done.json', dict(receipt, reports=[]))
            assert not runner.evaluation_complete(job, trial)
        for label, change in (
            ('missing_patient_prefix', lambda r: r['rows'].pop()),
            ('duplicate_patient_prefix', lambda r: r['rows'].append(dict(r['rows'][0]))),
            ('invalid_probability', lambda r: r['rows'][0].update(pcr_probability=1.1)),
            ('wrong_label', lambda r: r['rows'][0].update(pcr=1-r['rows'][0]['pcr'])),
            ('wrong_profile', lambda r: r.update(landmarks='all_observed')),
            ('wrong_best_validation_score', lambda r: r.update(selection_score=.6)),
        ):
            malformed = json.loads(json.dumps(reports['best_val']))
            change(malformed)
            must_reject(lambda: runner.validate_evaluation_report(job, 'best_val', malformed))
            results[label] = 'rejected'
        trial, job, run = isolated('evaluate_before_60_trained')
        with patch.object(runner, 'training_complete', lambda job, *unused: False), \
             patch.object(training_module, 'evaluate', denied):
            must_reject(lambda: runner.worker(trial, job['name'], 'evaluate'))
        results['test_evaluation_before_all60_training'] = 'rejected before evaluation'
        results['complete_three_report_receipt'] = 'accepted with full coverage'
        results['empty_evaluation_receipt_reports'] = 'rejected'
    return results


def verify_checkpoint_retention(runner, spec, training_module):
    """Exercise real worker/receipt/cleanup paths with disposable CPU markers."""
    from mri_vla_jepa import t0_cv_worker as runtime_worker
    actual_training_complete = runner.training_complete
    results = {}
    with tempfile.TemporaryDirectory(prefix='medium976_retention_CPU_') as directory:
        def isolated(label):
            trial = json.loads(json.dumps(spec))
            trial['output_root'] = str(Path(directory) / label)
            for job in trial['jobs']:
                job['output'] = str(Path(trial['output_root']) / 'runs' / job['name'])
            job = trial['jobs'][0]
            run = Path(job['output'])
            return trial, job, run

        def save(path, value):
            Path(path).write_text(json.dumps(value) + '\n')

        def fake_train(store, cfg, output, *, resume=False):
            assert resume is False
            report = {'completed': True, 'landmarks': 't0_t2', 'best_selection_score': .5}
            save(Path(output) / 'train_report.json', report)
            for filename in ('best.pt', 'last.pt'):
                (Path(output) / filename).write_bytes(('CPU fake ' + filename).encode())
            return report

        def fake_evaluate(checkpoint, manifest, output, *, split, **kwargs):
            assert Path(checkpoint).is_file()
            patients = read(manifest)['patients']
            report = {'split': split, 'landmarks': 't0_t2', 'checkpoint_landmarks': 't0_t2',
                      'synthetic': False, 'metric_name': 'f1', 'f1_threshold': .5, 'selection_score': .5,
                      'rows': [{'patient_key': patient['patient_key'], 'landmark': stage,
                                'pcr': patient['target']['pcr'], 'pcr_probability': .5}
                               for patient in patients if patient['split'] == split
                               for stage, visit in enumerate(patient['visits'][:3]) if visit is not None]}
            save(output, report)
            return report

        def worker_call(trial, job, stage, evaluator=fake_evaluate, validator=None):
            # Other jobs are independently covered by the fixed-60 gate tests.
            def completed(item, actual_spec=None):
                return (actual_training_complete(item, actual_spec)
                        if item['name'] == job['name'] else True)

            with patch.object(runner, 'training_complete', completed), \
                 patch.object(training_module, 'train', fake_train if stage == 'train' else denied), \
                 patch.object(training_module, 'evaluate', evaluator), \
                 patch.object(runtime_worker, 'validate_completed_run', validator or (lambda *unused: None)), \
                 patch('subprocess.Popen', denied), redirect_stderr(io.StringIO()):
                runner.worker(trial, job['name'], stage)

        def must_reject(call):
            try:
                with redirect_stderr(io.StringIO()):
                    call()
            except (ValueError, RuntimeError, FileNotFoundError):
                return
            raise AssertionError('An invalid or interrupted retention operation was accepted')

        for label in ('normal', 'evaluation_failure', 'interrupt_before_unlink', 'interrupt_after_unlink',
                      'last_disappears_during_training_gate',
                      'tampered_best', 'missing_best_after_cleanup', 'missing_report_after_cleanup'):
            trial, job, run = isolated(label)
            worker_call(trial, job, 'train')
            best_before, last_before = runner.identity(run / 'best.pt'), runner.identity(run / 'last.pt')
            assert actual_training_complete(job, trial) and (run / 'last.pt').is_file()
            if label == 'evaluation_failure':
                calls = []

                def failing_evaluate(checkpoint, manifest, output, *, split, **kwargs):
                    calls.append(Path(output).name)
                    if len(calls) == 3:
                        raise RuntimeError('CPU injected formal evaluation failure')
                    return fake_evaluate(checkpoint, manifest, output, split=split, **kwargs)

                must_reject(lambda: worker_call(trial, job, 'evaluate', failing_evaluate))
                assert len(calls) == 3 and runner.identity(run / 'last.pt') == last_before
                assert not (run / 'checkpoint_retention.json').exists()
                assert not (run / 'evaluation_done.json').exists()
                assert actual_training_complete(job, trial) and not runner.evaluation_complete(job, trial)
                results[label] = 'last preserved; no completed evaluation or deletion receipt'
                continue
            if label in ('interrupt_before_unlink', 'last_disappears_during_training_gate'):
                original_unlink = Path.unlink

                def interrupted_unlink(path, *args, **kwargs):
                    if path == run / 'last.pt':
                        raise RuntimeError('CPU injected interruption before deleting last')
                    return original_unlink(path, *args, **kwargs)

                with patch.object(Path, 'unlink', interrupted_unlink):
                    must_reject(lambda: worker_call(trial, job, 'evaluate'))
                assert runner.identity(run / 'last.pt') == last_before
            elif label == 'interrupt_after_unlink':
                original_write = runner.write

                def interrupted_write(path, value):
                    if Path(path) == run / 'checkpoint_retention.json' and value.get('status') == 'completed':
                        raise RuntimeError('CPU injected interruption after deleting last')
                    return original_write(path, value)

                with patch.object(runner, 'write', interrupted_write):
                    must_reject(lambda: worker_call(trial, job, 'evaluate'))
                assert not (run / 'last.pt').exists()
            else:
                worker_call(trial, job, 'evaluate')
            if label == 'last_disappears_during_training_gate':
                original_identity, raced = runner.identity, []

                def racing_identity(path):
                    if Path(path) == run / 'last.pt' and not raced:
                        (run / 'last.pt').unlink()
                        raced.append(True)
                        raise FileNotFoundError('CPU simulated concurrent journaled deletion')
                    return original_identity(path)

                with patch.object(runner, 'identity', racing_identity):
                    assert actual_training_complete(job, trial)
                assert raced == [True] and not (run / 'last.pt').exists()
            if label.startswith('interrupt_') or label == 'last_disappears_during_training_gate':
                assert read(run / 'checkpoint_retention.json')['status'] == 'pending_delete'
                assert actual_training_complete(job, trial) and runner.evaluation_artifacts_complete(job)
                assert not runner.evaluation_complete(job, trial)
                worker_call(trial, job, 'evaluate', denied, validator=denied)
            assert not (run / 'last.pt').exists() and runner.identity(run / 'best.pt') == best_before
            assert read(run / 'checkpoint_retention.json')['status'] == 'completed'
            assert actual_training_complete(job, trial) and runner.evaluation_complete(job, trial)
            receipt_before = runner.identity(run / 'checkpoint_retention.json')
            runner.finalize_checkpoint_retention(trial, job)
            assert runner.identity(run / 'checkpoint_retention.json') == receipt_before
            if label == 'tampered_best':
                with (run / 'best.pt').open('ab') as stream:
                    stream.write(b'CPU injected tampering')
                assert not actual_training_complete(job, trial)
                assert not runner.evaluation_artifacts_complete(job)
                assert not runner.evaluation_complete(job, trial)
                must_reject(lambda: runner.finalize_checkpoint_retention(trial, job))
                must_reject(lambda: worker_call(trial, job, 'evaluate', denied))
                with patch.object(training_module, 'train', denied):
                    must_reject(lambda: runner.worker(trial, job['name'], 'train'))
                results[label] = 'changed best rejected; no retraining or evaluation'
            elif label == 'missing_best_after_cleanup':
                report_before = runner.identity(run / 'train_report.json')
                (run / 'best.pt').unlink()
                assert not actual_training_complete(job, trial)
                assert not runner.evaluation_artifacts_complete(job)
                assert not runner.evaluation_complete(job, trial)
                must_reject(lambda: worker_call(trial, job, 'evaluate', denied))
                with patch.object(training_module, 'train', denied):
                    must_reject(lambda: runner.worker(trial, job['name'], 'train'))
                assert not (run / 'best.pt').exists() and not (run / 'last.pt').exists()
                assert runner.identity(run / 'train_report.json') == report_before
                results[label] = 'missing retained best rejected; no fresh training or report overwrite'
            elif label == 'missing_report_after_cleanup':
                (run / 'evaluation_best_test.json').unlink()
                assert not runner.evaluation_artifacts_complete(job)
                must_reject(lambda: actual_training_complete(job, trial))
                must_reject(lambda: worker_call(trial, job, 'evaluate', denied))
                with patch.object(training_module, 'train', denied):
                    must_reject(lambda: runner.worker(trial, job['name'], 'train'))
                assert runner.identity(run / 'best.pt') == best_before
                results[label] = 'missing formal report rejected; retained best preserved; no recomputation'
            else:
                results[label] = {'only_best_checkpoint_remains': True, 'formal_three_reports_verified': True,
                                  'training_and_evaluation_receipts_remain_valid': True,
                                  'cleanup_idempotent': True, 'repeated_evaluation_called': False}
        storage = Path(directory) / 'storage_glob_race'
        storage.mkdir()
        best, disappeared_last = storage / 'best.pt', storage / 'last.pt'
        best.write_bytes(b'CPU retained best')
        disappeared_last.write_bytes(b'CPU deleted last')
        stale_glob = [best, disappeared_last]
        disappeared_last.unlink()
        original_glob = Path.glob

        def racing_glob(path, pattern):
            if path == storage:
                assert pattern == '*.pt'
                return iter(stale_glob)
            return original_glob(path, pattern)

        with patch.object(Path, 'glob', racing_glob):
            assert runner.checkpoint_bytes_on_disk(storage) == best.stat().st_size
        results['checkpoint_storage_glob_stat_race'] = 'vanished last skipped; retained best counted'
    return {'cases': results, 'real_worker_and_retention_functions_called': True,
            'real_training_or_evaluation_called': False, 'physical_GPU_used': False}


def verify_direct_worker_plan_guard(runner, spec, training_module):
    runner.validate_plan(spec)
    subset = json.loads(json.dumps(spec))
    subset['jobs'] = subset['jobs'][:1]
    with patch.object(runner.capacity, 'apply_allocator_policy', denied), \
         patch.object(runner, 'import_runtime', denied), \
         patch.object(training_module, 'train', denied), patch.object(training_module, 'evaluate', denied), \
         patch('subprocess.Popen', denied):
        try:
            runner.worker(subset, subset['jobs'][0]['name'], 'evaluate')
        except ValueError:
            pass
        else:
            raise AssertionError('A subset specification bypassed the direct worker full-plan guard')
    return {'full_60_plan_accepted': True, 'subset_spec_direct_worker_evaluate_rejected': True,
            'rejection_before_allocator_runtime_model_or_evaluation': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, default=TASK)
    parser.add_argument('--runner', type=Path, default=ROOT / 'tools/medium976/run_training.py')
    parser.add_argument('--output', type=Path, default=AUDIT / 'training_entry_cpu_verification.json')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    task = args.task.resolve()
    spec_path = task / 'training_spec.json'
    spec = read(spec_path)
    assert spec['schema'] == 'medium976_t0_t2_no_tf_training_spec_v1'
    assert Path(spec['task_dir']).resolve() == task
    runtime = Path(spec['runtime_src']).resolve()
    assert runtime == task / 'runtime/src'
    dataset = Path(spec['dataset_root']).resolve()
    assert dataset == ROOT / 'data/registered_roi32_medium976_v1'
    output_root = Path(spec['output_root']).resolve()
    assert 'medium976' in output_root.name and output_root != OLD_TASK
    assert spec['train_seeds'] == SEEDS and set(spec['variants']) == set(VARIANTS)
    jobs = spec['jobs']
    assert len(jobs) == JOB_COUNT and len({j['name'] for j in jobs}) == JOB_COUNT
    assert {(j['variant'], j['seed'], j['fold_id']) for j in jobs} == set(itertools.product(VARIANTS, SEEDS, range(1, 6)))
    for field in ('runtime_identities', 'config_identities', 'manifest_identities', 'asset_identities'):
        assert spec[field]
        for identity in spec[field]:
            check_identity(identity)
    assert len(spec['asset_identities']) == 1
    assert Path(spec['asset_identities'][0]['path']).resolve() == ROOT / 'data/pretrained/codec.pt'
    old_src = OLD_TASK / 'runtime/src/mri_vla_jepa'
    new_src = runtime / 'mri_vla_jepa'
    raw_src = ROOT / 'private_outputs/medium976_roi32_v1/runtime/src/mri_vla_jepa'
    source_equality = {}
    for name in ('data.py', 'data_pipeline.py'):
        assert (new_src / name).read_bytes() == (raw_src / name).read_bytes(), name
        source_equality[name] = 'Validated raw-storage runtime'
    for name in ('model.py', 'model_config.py', 'training.py', 'train_config.py', 'diagnostics.py',
                 'landmark_profiles.py', 'cli.py', 't0_cv_worker.py'):
        assert (new_src / name).read_bytes() == (old_src / name).read_bytes(), name
        source_equality[name] = 'Frozen original Medium T0-T2 no-TF runtime'
    sys.path.insert(0, str(runtime))
    import torch
    torch.set_num_threads(1)
    from mri_vla_jepa import data as data_module
    from mri_vla_jepa import training as training_module
    from mri_vla_jepa import train_config as config_module
    from mri_vla_jepa.data import RawMRIStore
    from mri_vla_jepa.landmark_profiles import allowed_landmarks, profile_stages
    from mri_vla_jepa.train_config import load_config
    assert Path(data_module.__file__).resolve().is_relative_to(runtime)
    assert Path(training_module.__file__).resolve().is_relative_to(runtime)
    assert Path(config_module.__file__).resolve().is_relative_to(runtime)
    assert profile_stages('t0_t2') == (0, 1, 2)
    registered_folds = read(AUDIT / 'folds.json')['folds']
    stores, split_counts = {}, {}
    for fold in range(1, 6):
        path = dataset / 'cv_manifests' / f'fold{fold}' / 'manifest.json'
        store = RawMRIStore(path)
        assert store.preprocessed_roi32_raw and store.preprocessed_roi32
        assert store.manifest['image_preprocessing'] == {'mode': 'preprocessed_roi32_raw'}
        assert len(store.patients) == 976
        for split in ('train', 'val', 'test'):
            ids = {store.patients[i]['patient_key'] for i in store.by_split[split]}
            assert ids == set(registered_folds[f'fold{fold}'][split])
        for i, patient in enumerate(store.patients):
            actual = [stage for stage, visit in enumerate(patient['visits']) if visit is not None and stage <= 2]
            assert allowed_landmarks(store, i, 't0_t2') == actual and actual
        assert store.patients[next(i for i, p in enumerate(store.patients) if p['patient_key'] == 'ISPY2-893874')]['visits'][3] is None
        state = store.normalization_state()
        assert state == read(dataset / 'normalization' / f'fold{fold}_clinical_state.json')
        split_counts[f'fold{fold}'] = {s: len(store.by_split[s]) for s in ('train', 'val', 'test')}
        stores[fold] = store
    for job in jobs:
        variant, seed, fold = job['variant'], job['seed'], job['fold_id']
        config = Path(job['config']).resolve()
        assert config == task / 'input_configs' / f"{job['name']}.yaml"
        assert Path(job['manifest']).resolve() == dataset / 'cv_manifests' / f'fold{fold}' / 'manifest.json'
        assert Path(job['output']).resolve() == output_root / 'runs' / job['name']
        assert Path(job['output']).resolve().is_relative_to(output_root)
        cfg = load_config(config)
        assert cfg.seed == seed and cfg.landmarks == 't0_t2'
        assert cfg.prediction_training == 'hybrid' and cfg.jepa_weight == 0
        assert (cfg.model.pcr_readout, cfg.model.spatial_readout, cfg.rollout_weight) == VARIANTS[variant]
        assert cfg.model.enable_world_model is True and cfg.model.model_size == 'medium'
        assert cfg.model.mri_phases == 'all' and cfg.model.enable_flow is False
        assert cfg.model.lesion_world_weighting is False and cfg.model.lesion_pooling is False
        assert cfg.flow_weight == cfg.reconstruction_weight == cfg.variance_weight == 0
        assert cfg.training_unit == 'epochs' and cfg.max_epochs == 200
        assert cfg.early_stopping_patience == 50 and cfg.selection_metric == 'f1' and cfg.f1_threshold == .5
        assert cfg.batch_size == 16 and cfg.accumulation == 1 and cfg.device == 'cuda:0' and cfg.precision == 'bf16'
        assert cfg.lr == cfg.encoder_lr == 1e-4 and cfg.encoder_training == 'finetune'
        assert cfg.initialize_from_checkpoint is None
        assert Path(cfg.encoder_checkpoint).resolve() == ROOT / 'data/pretrained/codec.pt'
        assert cfg.image_cache is None
        original_config = Path('/projects/xts0030/public/cxx/medium_t0_t2_no_tf_cv5_multiseed_v1/input_configs') / f'{variant}_no_tf_seed{seed}_fold{fold}.yaml'
        if not original_config.exists():
            original_config = Path('/projects/xts0030/public/cxx/medium_t0_t2_no_tf_cv5_multiseed_v1/input_configs') / f'{variant}_seed{seed}_fold{fold}.yaml'
        assert original_config.exists(), str(original_config)
        assert cfg.to_dict() == load_config(original_config).to_dict(), job['name']
    paired = verify_paired_initialization()
    runner_source = args.runner.read_text()
    assert 't0_cv_suite' not in runner_source and 'prepare_splits' not in runner_source
    assert 'install_paired_readout_initialization' in runner_source
    sys.path.insert(0, str(args.runner.resolve().parent))
    import_spec = importlib.util.spec_from_file_location('_medium976_training_entry_for_verification', args.runner)
    runner = importlib.util.module_from_spec(import_spec)
    sys.modules[import_spec.name] = runner
    import_spec.loader.exec_module(runner)
    selected = runner.select_jobs(spec, variants=None, seeds=None, folds=None)
    assert {j['name'] for j in selected} == {j['name'] for j in jobs} and len(selected) == JOB_COUNT
    subset = runner.select_jobs(spec, variants=['forecast_mean', 'direct_no_jepa'], seeds=[21, 203], folds=[2, 5])
    assert {(j['variant'], j['seed'], j['fold_id']) for j in subset} == set(itertools.product(['forecast_mean', 'direct_no_jepa'], [21, 203], [2, 5]))
    assert len(subset) == 8 and spec == read(spec_path)
    invalid_cases = ({'variants': ['invalid_variant']}, {'seeds': [999999]}, {'folds': [0]}, {'folds': [6]})
    for invalid in invalid_cases:
        try:
            runner.select_jobs(spec, **invalid)
        except (ValueError, AssertionError, SystemExit):
            pass
        else:
            raise AssertionError(f'Invalid filter accepted: {invalid}')
    for accepted in (['0'], ['0', '1'], ['GPU-1fd96108-151b-fd8e-8c13-9b75a23a90b8']):
        runner.validate_gpus(accepted)
    for rejected in ([], ['0', '0'], ['-1'], ['0,1'], ['invalid']):
        try:
            runner.validate_gpus(rejected)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid GPU selection accepted: {rejected}')
    captured = io.StringIO()
    # Patch aliases on the imported runner as well as their source modules.
    aliases = [(name, value) for name, value in vars(runner).items()
               if callable(value) and getattr(value, '__name__', '') in {'train', 'evaluate'}]
    for name, _ in aliases:
        setattr(runner, name, denied)
    try:
        with patch('subprocess.Popen', denied), patch('subprocess.run', denied), \
             patch('multiprocessing.process.BaseProcess.start', denied), \
             patch.object(training_module, 'train', denied), patch.object(training_module, 'evaluate', denied), \
             patch.object(torch.cuda, 'is_available', denied), patch.object(torch.cuda, 'device_count', denied), \
             patch.object(torch.cuda, 'set_device', denied), redirect_stdout(captured):
            result = runner.main(['--spec', str(spec_path), '--preflight-only', '--gpus', '0', '--max-active', '5', '--variants', 'forecast_mean',
                                  '--seeds', '21', '--folds', '1'])
            assert result in (None, 0) or isinstance(result, dict)
    finally:
        for name, value in aliases:
            setattr(runner, name, value)
    assert spec == read(spec_path), 'Preflight changed the registered 60-job plan'
    with redirect_stdout(io.StringIO()):
        cli = verify_cli_contracts(runner, spec_path)
    capacity_contracts = verify_capacity_contracts(runner, spec)
    capacity_refresh = verify_capacity_refresh_cli(runner, spec, spec_path)
    gates = verify_controller_gates(runner, spec)
    direct_worker_plan_guard = verify_direct_worker_plan_guard(runner, spec, training_module)
    with CPU_allocator_calls(runner.capacity, torch) as allocator_calls:
        allocator_contracts = verify_allocator_contracts(runner.capacity, torch)
        contracts = verify_run_and_evaluation_contracts(runner, spec, training_module)
        retention_contracts = verify_checkpoint_retention(runner, spec, training_module)
    assert allocator_calls and not torch.cuda.is_initialized()
    report = {'schema': 'medium976_training_entry_independent_CPU_verification_v1',
              'status': 'passed', 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'registered_jobs': JOB_COUNT, 'strict_configs_checked': JOB_COUNT, 'fixed_fold_manifests_checked': 5,
              'all_three_seeds_four_variants_five_folds_registered': True,
              'config_settings_equal_original_no_TF_configs': True,
              'source_files_compared_directly': source_equality,
              'profile': 't0_t2', 'T3_input_landmarks_forbidden_actual_T3_future_targets_retained': True,
              'raw_storage_and_fold_clinical_states_agree': True, 'split_counts': split_counts,
              'original_pretrained_codec_asset_identity_checked': True,
              'paired_readout_initialization': paired,
              'default_selection_jobs': JOB_COUNT, 'filtered_selection_jobs': 8,
              'invalid_filters_rejected': True, 'preflight_cpu_only_under_denied_worker_and_GPU_calls': True,
              'GPU_selector_pure_validation_checked_without_GPU_probe': True,
              'multiworker_CLI_contracts': cli,
              'capacity_profile_binding_and_peak_reservations': capacity_contracts,
              'automatic_capacity_refresh_CLI': capacity_refresh,
              'allocator_policy_and_worker_receipts': allocator_contracts,
              'full_plan_unchanged_by_filters_and_preflight': True,
              'controller_gate_and_concurrency_simulations': gates,
              'run_identity_resume_and_evaluation_contract_tests': contracts,
              'best_only_checkpoint_retention': retention_contracts,
              'fixed_60_direct_worker_plan_guard': direct_worker_plan_guard,
              'preflight_stdout': captured.getvalue(), 'real_training_or_evaluation_called': False,
              'inputs': {'spec': info(spec_path), 'runner': info(args.runner), 'verifier': info(Path(__file__)),
                         'capacity': info(Path(runner.capacity.__file__)),
                         'admission': info(Path(runner.admission.__file__))}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': 'passed', 'report': str(args.output), 'registered_jobs': JOB_COUNT,
                      'configs_checked': JOB_COUNT, 'fixed_folds_checked': 5,
                      'CPU_only_preflight': True, 'real_training_or_evaluation_called': False}, indent=2))


if __name__ == '__main__':
    main()

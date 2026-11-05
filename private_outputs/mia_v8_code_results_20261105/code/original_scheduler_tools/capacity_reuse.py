#!/usr/bin/env python3
"""Prove CPU-only reuse of the original Medium976 capacity measurements.

Old measurement dictionaries and allocator startup receipts stay unchanged.
Only the explicit inheritance wrapper carries the new three-seed plan binding.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import yaml

BINDING_FIELDS = ('schema', 'runtime_src', 'dataset_root', 'variants', 'train_seeds', 'fold_ids',
                  'runtime_identities', 'config_identities', 'manifest_identities', 'asset_identities')
UNCHANGED_FIELDS = ('schema', 'runtime_src', 'dataset_root', 'variants', 'fold_ids',
                    'runtime_identities', 'manifest_identities', 'asset_identities')
CRITICAL_FUNCTIONS = ('policy_identity', 'apply_allocator_policy', '_worker',
                      'gpu_facts', '_selected_gpu', '_process_observations')
CRITICAL_CONSTANTS = ('ALLOCATOR_FRACTION', 'ALLOCATOR_CONF', 'NON_TORCH_MIB', 'MARGIN_MIB',
                      'SCHEMA', 'PROFILE_SCRIPT')


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def _checked(reference):
    if identity(reference['path']) != reference:
        raise ValueError(f'Capacity inheritance artifact changed: {reference["path"]}')
    return Path(reference['path'])


def _same_metadata(snapshot, original):
    observed = identity(snapshot)
    if any(observed[key] != original[key] for key in ('bytes', 'mtime_ns')):
        raise ValueError(f'Legacy snapshot metadata differs from original measurement: {snapshot}')
    return observed


def _asts(path):
    return ast.parse(Path(path).read_text())


def _function(tree, name):
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f'Cannot uniquely prove allocator function: {name}')
    return ast.dump(matches[0], include_attributes=False)


def _assignment(tree, name):
    matches = [node.value for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    if len(matches) != 1:
        raise ValueError(f'Cannot uniquely prove allocator constant: {name}')
    return ast.dump(matches[0], include_attributes=False)


def _reservation_assignment(tree):
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                    and target.value.id == 'result' and isinstance(target.slice, ast.Constant)
                    and target.slice.value == 'reservation_mib'):
                matches.append(ast.dump(node.value, include_attributes=False))
    if len(matches) != 1:
        raise ValueError('Cannot uniquely prove GPU reservation formula')
    return matches[0]


def _without_source(policy):
    return {key: value for key, value in policy.items() if key != 'source'}


def build_proof(spec, legacy_root, *, current_binding, current_policy, current_source):
    legacy_root = Path(legacy_root).resolve()
    old_spec_path = legacy_root / 'legacy_training_spec.json'
    old_profile_path = legacy_root / 'legacy_capacity_profile.json'
    old_source_path = legacy_root / 'legacy_capacity.py'
    old_spec, old_profile = read(old_spec_path), read(old_profile_path)
    if (old_spec.get('training_runs') != 120 or len(old_spec.get('jobs', [])) != 120
            or set(old_spec.get('train_seeds', [])) != {43, 44, 809, 203, 30, 21}
            or spec.get('training_runs') != 60 or len(spec.get('jobs', [])) != 60
            or spec.get('train_seeds') != [21, 44, 203]):
        raise ValueError('Capacity inheritance is restricted to the explicit 120-to-60 three-seed subset')
    if any(spec[key] != old_spec[key] for key in UNCHANGED_FIELDS):
        raise ValueError('Runtime, data, folds, variants or pretrained assets changed; capacity cannot be inherited')
    old_binding = old_profile.get('binding', {})
    old_policy = old_binding.get('allocator_policy', {})
    expected_binding = {**{key: old_spec[key] for key in BINDING_FIELDS},
                        'allocator_policy': old_policy, 'profiler_source': current_binding['profiler_source']}
    if (old_profile.get('status') != 'passed' or old_binding != expected_binding
            or old_profile.get('evidence_kind') not in (None, 'measured')):
        raise ValueError('Legacy profile lacks an intact original measurement binding')
    if _without_source(old_policy) != _without_source(current_policy):
        raise ValueError('Allocator policy mathematics or environment changed')
    original_source = old_policy.get('source', {})
    _same_metadata(old_source_path, original_source)
    if original_source['path'] != str(Path(current_source).resolve()):
        raise ValueError('Legacy measurement came from another capacity implementation')
    old_tree, new_tree = _asts(old_source_path), _asts(current_source)
    for name in CRITICAL_FUNCTIONS:
        if _function(old_tree, name) != _function(new_tree, name):
            raise ValueError(f'Capacity execution or allocator function changed: {name}')
    for name in CRITICAL_CONSTANTS:
        if _assignment(old_tree, name) != _assignment(new_tree, name):
            raise ValueError(f'Capacity policy constant changed: {name}')
    if _reservation_assignment(old_tree) != _reservation_assignment(new_tree):
        raise ValueError('GPU reservation formula changed')
    old_configs = {item['path']: item for item in old_spec['config_identities']}
    if len(old_configs) != 120:
        raise ValueError('The original 120 config identities are not intact')
    snapshots = {}
    for path, original in old_configs.items():
        snapshot = legacy_root / 'legacy_configs' / Path(path).name
        snapshots[path] = _same_metadata(snapshot, original)
    old_jobs = {job['name']: job for job in old_spec['jobs']}
    if len(old_jobs) != 120 or len({job['name'] for job in spec['jobs']}) != 60:
        raise ValueError('Duplicate capacity inheritance jobs')
    old_math = {}
    for variant in spec['variants']:
        value = old_profile['variants'][variant]
        representative = value['config']
        if representative not in snapshots:
            raise ValueError('Original measured config has no byte-preserving snapshot')
        parsed = yaml.safe_load(Path(snapshots[representative]['path']).read_text())
        parsed.pop('seed')
        old_math[variant] = parsed
    pairs = []
    for job in spec['jobs']:
        if job != old_jobs.get(job['name']):
            raise ValueError('New jobs are not an exact subset of the original job records')
        original = old_configs.get(job['config'])
        if original is None or identity(job['config']) != original:
            raise ValueError('Selected config metadata differs from original measured plan')
        snapshot = snapshots[job['config']]
        if Path(job['config']).read_bytes() != Path(snapshot['path']).read_bytes():
            raise ValueError('Selected config bytes differ from the original 120-config snapshot')
        parsed = yaml.safe_load(Path(job['config']).read_text())
        if parsed.pop('seed') != job['seed'] or parsed != old_math[job['variant']]:
            raise ValueError('New capacity workload differs from measured representative beyond seed')
        pairs.append({'job': job['name'], 'current_config': original, 'legacy_snapshot': snapshot,
                      'byte_equal': True, 'all_values_equal_after_ignoring_seed': True})
    expected_configs = [old_configs[job['config']] for job in spec['jobs']]
    if spec['config_identities'] != expected_configs:
        raise ValueError('Selected plan config identities are not the ordered original subset')
    return {'schema': 'medium976_capacity_inheritance_proof_v1', 'status': 'passed',
            'legacy_spec': identity(old_spec_path), 'legacy_profile': identity(old_profile_path),
            'legacy_capacity_source': identity(old_source_path), 'helper_source': identity(__file__),
            'legacy_binding': old_binding, 'current_binding': current_binding,
            'legacy_jobs': 120, 'current_jobs': 60, 'current_seeds': spec['train_seeds'],
            'unchanged_fields': list(UNCHANGED_FIELDS),
            'unchanged_execution_functions': list(CRITICAL_FUNCTIONS),
            'unchanged_policy_constants': list(CRITICAL_CONSTANTS),
            'gpu_reservation_formula_ast_equal': True, 'config_pairs': pairs,
            'new_gpu_measurements_performed': False,
            'measurement_scope': 'original four-variant full-batch and four-concurrent-worker measurements; inherited by strict subset proof'}


def validate_inheritance(spec, profile, *, current_binding, current_policy, current_source):
    inheritance = profile.get('inheritance', {})
    if inheritance.get('schema') != 'medium976_capacity_inheritance_v1':
        raise ValueError('Inherited capacity lacks its explicit inheritance receipt')
    current_spec_path = _checked(inheritance['current_spec'])
    if read(current_spec_path) != spec:
        raise ValueError('Inherited capacity does not identify the current full spec')
    proof = read(_checked(inheritance['proof']))
    expected = build_proof(spec, inheritance['legacy_root'], current_binding=current_binding,
                           current_policy=current_policy, current_source=current_source)
    if proof != expected or inheritance.get('new_gpu_measurements_performed') is not False:
        raise ValueError('Capacity inheritance proof is stale or inconsistent')
    old_profile = read(_checked(proof['legacy_profile']))
    if (profile.get('variants') != old_profile['variants']
            or profile.get('measured_gpu') != old_profile['measured_gpu']
            or profile.get('created_at_utc') != old_profile['created_at_utc']):
        raise ValueError('Inherited capacity altered the original measured values or startup receipts')
    return read(_checked(proof['legacy_spec'])), old_profile


def inherit(spec_path, legacy_root, output, concurrency_output):
    import capacity
    spec = read(spec_path)
    capacity._verify_inputs(spec)
    binding, policy = capacity.profile_binding(spec), capacity.policy_identity()
    proof = build_proof(spec, legacy_root, current_binding=binding, current_policy=policy,
                        current_source=Path(capacity.__file__))
    proof_path = Path(output).with_name('capacity_inheritance_proof.json')
    capacity.write(proof_path, proof)
    original = read(proof['legacy_profile']['path'])
    profile = deepcopy(original)
    profile.update(evidence_kind='inherited', binding=binding,
                   inherited_at_utc=datetime.now(timezone.utc).isoformat(),
                   inheritance={'schema': 'medium976_capacity_inheritance_v1',
                                'current_spec': identity(spec_path), 'proof': identity(proof_path),
                                'legacy_root': str(Path(legacy_root).resolve()),
                                'new_gpu_measurements_performed': False})
    capacity.validate_profile(spec, profile, [original['measured_gpu']['uuid']])
    capacity.write(output, profile)
    legacy_concurrency_path = Path(legacy_root) / 'legacy_capacity_concurrency_verification.json'
    concurrency = read(legacy_concurrency_path)
    if (concurrency.get('status') != 'passed' or concurrency.get('binding') != original['binding']
            or concurrency.get('policy') != original['binding']['allocator_policy']
            or any(worker.get('status') != 'passed' or worker.get('exit_code') != 0
                   for worker in concurrency.get('workers', [])) or len(concurrency.get('workers', [])) != 4):
        raise ValueError('Legacy concurrent capacity verification is not intact')
    wrapper = {'schema': 'medium976_inherited_concurrency_verification_v1', 'status': 'passed',
               'evidence_kind': 'inherited', 'binding': binding,
               'new_gpu_measurements_performed': False,
               'capacity_profile': identity(output), 'inheritance_proof': identity(proof_path),
               'legacy_report': identity(legacy_concurrency_path), 'original_measurement': concurrency}
    capacity.write(concurrency_output, wrapper)
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--legacy-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--concurrency-output', type=Path, required=True)
    args = parser.parse_args()
    profile = inherit(args.spec, args.legacy_root, args.output, args.concurrency_output)
    print(json.dumps({'status': profile['status'], 'evidence_kind': profile['evidence_kind'],
                      'jobs': 60, 'seeds': [21, 44, 203], 'new_gpu_measurements_performed': False,
                      'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

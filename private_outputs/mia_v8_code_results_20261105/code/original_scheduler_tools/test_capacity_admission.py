"""CPU-only resource/namespace tests; no model, CUDA, or training calls."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location('medium976_admission_tests', Path(__file__).with_name('admission.py'))
admission = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(admission)
GPU = 'GPU-11111111-2222-3333-4444-555555555555'
OTHER_GPU = 'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'


def snapshot(used=51000, processes=None):
    return {GPU: {'total_mib': 81920, 'used_mib': used, 'processes': processes or []}}


def reservations(count):
    return [{'gpu_uuid': GPU, 'reservation_mib': 19000} for _ in range(count)]


def test_same_gpu_fourth_worker_fits_full_peak_but_fifth_does_not():
    assert admission.choose_gpu(snapshot(), reservations(3), {GPU: 51000}, 19000, .96) == GPU
    assert admission.choose_gpu(snapshot(68000), reservations(4), {GPU: 68000}, 19000, .96) is None


def test_unknown_nvml_usage_remains_foreign_and_cannot_be_subtracted():
    assert admission.choose_gpu(snapshot(34000), reservations(2), {}, 19000, .96) is None
    assert admission.choose_gpu(snapshot(34000), reservations(2), {GPU: 34000}, 19000, .96) == GPU


def test_live_usage_growth_guard_rejects_even_with_small_reservations():
    assert admission.choose_gpu(snapshot(65000), reservations(1), {GPU: 65000}, 13000, .96) is None


def test_freed_slot_refills_without_waiting_for_other_workers():
    assert admission.choose_gpu(snapshot(68000), reservations(4), {GPU: 68000}, 19000, .96) is None
    assert admission.choose_gpu(snapshot(51000), reservations(3), {GPU: 51000}, 19000, .96) == GPU


def test_choose_ignores_unselected_gpu_and_bad_ownership_cannot_create_capacity():
    assert admission.choose_gpu(snapshot(65000), reservations(1), {OTHER_GPU: 65000}, 19000, .96) is None
    assert admission.choose_gpu(snapshot(65000), reservations(1), {GPU: 90000}, 19000, .96) is None


def test_choose_prefers_less_committed_gpu_and_does_not_mutate_inputs():
    data = snapshot(51000)
    data[OTHER_GPU] = {'total_mib': 81920, 'used_mib': 4, 'processes': []}
    before = deepcopy(data)
    assert admission.choose_gpu(data, reservations(3), {GPU: 51000}, 19000, .96) == OTHER_GPU
    assert data == before


@pytest.mark.parametrize('quantity', [0, -1, float('nan'), float('inf'), True])
def test_invalid_candidate_is_rejected(quantity):
    with pytest.raises(ValueError):
        admission.choose_gpu(snapshot(), [], {}, quantity, .96)


def host():
    return {'cpu_slots': 13, 'total_mib': 228000, 'available_mib': 224000}


def worker(**kwargs):
    return {'cpu_slots': 6, 'host_mib': 20000, **kwargs}


def test_thirteen_cpu_allocation_allows_four_shared_workers_but_not_five():
    assert admission.host_admits([worker()] * 3, worker(), host())[0]
    assert not admission.host_admits([worker()] * 4, worker(), host())[0]
    assert not admission.host_admits([worker()] * 2, worker(), host(), cpu_oversubscribe=1)[0]


def test_loader_threads_count_toward_the_shared_cpu_budget():
    profiled = worker(cpu_slots=4, loader_threads=2)
    assert admission.host_admits([profiled] * 3, profiled, host())[0]
    assert not admission.host_admits([profiled] * 4, profiled, host())[0]


def test_host_available_ram_and_future_growth_gate_even_with_free_cpu():
    limited = {**host(), 'available_mib': 40000}
    assert not admission.host_admits([worker()], worker(), limited)[0]
    assert admission.host_admits([worker(rss_mib=20000)], worker(), limited)[0]


def test_host_total_budget_and_legacy_reservation_key_are_supported():
    assert admission.host_admits([], {'cpu_slots': 6, 'host_reservation_mib': 20000}, host())[0]
    assert not admission.host_admits([], worker(host_mib=210000), host())[0]
    assert not admission.host_admits([], {'cpu_slots': 6}, host())[0]


def test_effective_host_uses_online_allocation_instead_of_host_affinity(monkeypatch):
    actual_read = admission._read_text
    monkeypatch.setattr(admission.os, 'cpu_count', lambda: 13)
    monkeypatch.setattr(admission.os, 'sched_getaffinity', lambda _: set(range(128)))
    monkeypatch.setattr(admission, '_cgroup_roots', lambda: [])
    monkeypatch.setattr(admission, '_read_text', lambda p: '0-12' if str(p) == '/sys/devices/system/cpu/online' else actual_read(p))
    result = admission.effective_host()
    assert result['cpu_slots'] == 13
    assert result['available_mib'] <= result['total_mib']


def ownership_setup(monkeypatch):
    command = ['python', '-u', '/tools/run_training.py', '--spec', '/task/training_spec.json', '--worker', 'job']
    record = {'pid': 123, 'start_ticks': 456, 'command': command, 'gpu_uuid': GPU, 'attempt': 'attempt-1'}
    local = {'start_ticks': 456, 'state': 'S', 'command': command,
             'environment': {'CUDA_VISIBLE_DEVICES': GPU, 'MEDIUM976_ATTEMPT_ID': 'attempt-1'}}
    monkeypatch.setattr(admission, '_read_process', lambda pid: deepcopy(local))
    manager = admission.GPUOwnership('/task/training_spec.json')
    manager.pod_name = 'our-pod'
    native = {'Devices': [{'DeviceUUID': GPU, 'ProcessInfos': [
        {'Pid': 90123, 'NSPids': [90123, 123], 'PodName': 'our-pod', 'Cmdlines': command + [''],
         'DeviceUUID': GPU, 'MemUsed': 999999}]}]}
    monkeypatch.setattr(manager, '_platform_snapshot', lambda: deepcopy(native))
    return manager, record, local, native


def test_namespace_ownership_uses_nvml_memory_never_platform_memused(monkeypatch):
    manager, record, _, _ = ownership_setup(monkeypatch)
    result = manager.account(snapshot(20000, [{'pid': 90123, 'used_mib': 17000}]), [record])
    assert result['owned_mib'] == {GPU: 17000}
    assert result['foreign_mib'] == {GPU: 3000}
    assert result['evidence'][0]['verified']


@pytest.mark.parametrize('field,value', [('PodName', 'another-pod'), ('NSPids', [90123, 999]),
                                       ('DeviceUUID', OTHER_GPU), ('Cmdlines', ['wrong'])])
def test_invalid_namespace_claim_is_foreign(monkeypatch, field, value):
    manager, record, _, native = ownership_setup(monkeypatch)
    native['Devices'][0]['ProcessInfos'][0][field] = value
    result = manager.account(snapshot(20000, [{'pid': 90123, 'used_mib': 17000}]), [record])
    assert result['owned_mib'][GPU] == 0
    assert result['foreign_mib'][GPU] == 20000


@pytest.mark.parametrize('field,value', [('start_ticks', 999), ('command', ['wrong'])])
def test_pid_reuse_or_changed_command_cannot_claim_memory(monkeypatch, field, value):
    manager, record, local, _ = ownership_setup(monkeypatch)
    local[field] = value
    assert manager.account(snapshot(20000, [{'pid': 90123, 'used_mib': 17000}]), [record])['owned_mib'][GPU] == 0


@pytest.mark.parametrize('field,value', [('CUDA_VISIBLE_DEVICES', OTHER_GPU), ('MEDIUM976_ATTEMPT_ID', 'other-attempt')])
def test_wrong_gpu_environment_or_attempt_is_foreign(monkeypatch, field, value):
    manager, record, local, _ = ownership_setup(monkeypatch)
    local['environment'][field] = value
    assert manager.account(snapshot(20000, [{'pid': 90123, 'used_mib': 17000}]), [record])['owned_mib'][GPU] == 0


def test_namespace_service_failure_and_duplicate_claims_fail_closed(monkeypatch):
    manager, record, _, native = ownership_setup(monkeypatch)
    native['Devices'][0]['ProcessInfos'] *= 2
    assert manager.account(snapshot(20000, [{'pid': 90123, 'used_mib': 17000}]), [record])['owned_mib'][GPU] == 0
    monkeypatch.setattr(manager, '_platform_snapshot', lambda: (_ for _ in ()).throw(OSError('unavailable')))
    assert manager.account(snapshot(20000, [{'pid': 90123, 'used_mib': 17000}]), [record])['foreign_mib'][GPU] == 20000


def test_duplicate_active_records_cannot_double_subtract_nvml(monkeypatch):
    manager, record, _, _ = ownership_setup(monkeypatch)
    assert manager.account(snapshot(20000, [{'pid': 90123, 'used_mib': 17000}]), [record, record])['owned_mib'][GPU] == 0


def test_worker_starting_or_already_exited_never_breaks_record(monkeypatch):
    monkeypatch.setattr(admission, '_read_process', lambda _: (_ for _ in ()).throw(FileNotFoundError()))
    manager = admission.GPUOwnership()
    record = manager.record(99999, ['expected-future-command'], GPU, 'attempt')
    assert record['start_ticks'] is None
    assert manager.account(snapshot(4), [record])['owned_mib'][GPU] == 0


def test_snapshot_only_returns_allocated_uuid_and_unknown_process_memory_is_foreign(monkeypatch):
    outputs = iter([f'{GPU}, 81920, 20000\n{OTHER_GPU}, 81920, 70000\n',
                    f'{GPU}, 123, N/A\n{OTHER_GPU}, 456, 70000\n'])
    monkeypatch.setattr(admission.subprocess, 'check_output', lambda *a, **k: next(outputs))
    assert admission.gpu_snapshot([GPU]) == snapshot(20000)

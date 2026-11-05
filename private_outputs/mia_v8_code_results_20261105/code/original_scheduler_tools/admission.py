"""Resource admission for isolated Medium976 workers; no Torch/CUDA imports."""
from __future__ import annotations

from collections import Counter
import http.client
import math
import os
from pathlib import Path
import socket
import subprocess

MIB = 2 ** 20
ATTEMPT_ENV = 'MEDIUM976_ATTEMPT_ID'
NAMESPACE_SOCKET = '/run/gpu_state_collector/ns.sock'


def _number(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected a finite resource quantity')
    if value < 0 or (positive and value == 0):
        raise ValueError('Expected a positive resource quantity')
    return value


def gpu_snapshot(gpus):
    """Read original NVML quantities, limited to explicitly selected UUIDs."""
    selected = {g['uuid'] if isinstance(g, dict) else g for g in gpus}
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.total,memory.used',
                                   '--format=csv,noheader,nounits'], text=True, timeout=10)
    result = {}
    for line in raw.splitlines():
        uuid, total, used = [x.strip() for x in line.split(',')]
        if uuid in selected:
            result[uuid] = {'total_mib': int(total), 'used_mib': int(used), 'processes': []}
    if set(result) != selected:
        raise ValueError('NVML did not return every selected physical GPU')
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                   '--format=csv,noheader,nounits'], text=True, timeout=10)
    for line in raw.splitlines():
        fields = [x.strip() for x in line.split(',')]
        if len(fields) != 3 or fields[0] not in result:
            continue
        try:
            pid, used = int(fields[1]), int(fields[2])
        except ValueError:
            continue  # Unmeasured process memory stays in aggregate foreign usage.
        if pid > 0 and used >= 0:
            result[fields[0]]['processes'].append({'pid': pid, 'used_mib': used})
    return result


def choose_gpu(snapshot, reservations, owned_mib, candidate_mib, budget_fraction, growth_guard=1024):
    """Reserve every future peak plus foreign usage and live growth headroom."""
    candidate_mib = _number(candidate_mib, positive=True)
    budget_fraction = _number(budget_fraction, positive=True)
    growth_guard = _number(growth_guard)
    if budget_fraction > 1:
        raise ValueError('GPU budget fraction cannot exceed one')
    choices = []
    for uuid, sample in snapshot.items():
        try:
            total = _number(sample['total_mib'], positive=True)
            used = _number(sample['used_mib'])
            if used > total:
                continue
            reserved = sum(_number(r['reservation_mib'], positive=True)
                           for r in reservations if r['gpu_uuid'] == uuid)
            own = _number(owned_mib.get(uuid, 0))
        except (KeyError, ValueError, TypeError):
            continue
        if own > used:
            own = 0  # Inconsistent telemetry cannot establish ownership.
        foreign = max(0, used - own)
        committed = max(reserved + foreign, used + growth_guard)
        budget = math.floor(total * budget_fraction)
        if committed + candidate_mib <= budget:
            choices.append((committed / budget, uuid))
    return min(choices)[1] if choices else None


def _read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _cpu_list_size(text):
    if not text:
        return None
    cpus = set()
    for section in text.split(','):
        bounds = [int(v) for v in section.split('-')]
        cpus.update(range(bounds[0], bounds[-1] + 1))
    return len(cpus) or None


def _cgroup_roots():
    roots = [Path('/sys/fs/cgroup')]
    for controller in ('cpu', 'cpu,cpuacct', 'memory', 'cpuset'):
        roots.append(Path('/sys/fs/cgroup') / controller)
    for line in (_read_text('/proc/self/cgroup') or '').splitlines():
        _, controllers, membership = line.split(':', 2)
        if not controllers:
            roots.append(Path('/sys/fs/cgroup') / membership.lstrip('/'))
        else:
            for controller in [controllers, *controllers.split(',')]:
                roots.append(Path('/sys/fs/cgroup') / controller / membership.lstrip('/'))
    return list(dict.fromkeys(roots))


def effective_host():
    """Use the narrowest visible allocation; affinity alone may expose the host."""
    os_cpus = os.cpu_count() or 1
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os_cpus
    online = _cpu_list_size(_read_text('/sys/devices/system/cpu/online')) or os_cpus
    cpu_limits = [os_cpus, affinity, online]
    quota_limits, memory_limits, memory_remaining = [], [], []
    cgroup_visible = False
    for root in _cgroup_roots():
        quota = _read_text(root / 'cpu.max')
        if quota:
            cgroup_visible = True
            amount, period = quota.split()
            if amount != 'max' and int(period) > 0:
                quota_limits.append(int(amount) / int(period))
        else:
            amount = _read_text(root / 'cpu.cfs_quota_us')
            period = _read_text(root / 'cpu.cfs_period_us')
            if amount is not None and period is not None:
                cgroup_visible = True
                if int(amount) > 0 and int(period) > 0:
                    quota_limits.append(int(amount) / int(period))
        for name in ('cpuset.cpus.effective', 'cpuset.cpus'):
            value = _cpu_list_size(_read_text(root / name))
            if value:
                cgroup_visible = True
                cpu_limits.append(value)
        for limit_name, current_name in (('memory.max', 'memory.current'),
                                          ('memory.limit_in_bytes', 'memory.usage_in_bytes')):
            value = _read_text(root / limit_name)
            if value is not None:
                cgroup_visible = True
            if value and value != 'max':
                limit = int(value)
                memory_limits.append(limit)
                current = _read_text(root / current_name)
                if current is not None:
                    memory_remaining.append(max(0, limit - int(current)))
    if quota_limits:
        cpu_limits.append(min(quota_limits))
    info = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        info[key] = int(value.strip().split()[0]) * 1024
    total = min([info['MemTotal'], *memory_limits])
    available = min([info['MemAvailable'], total, *memory_remaining])
    return {'cpu_slots': max(1, math.floor(min(cpu_limits))), 'os_cpu_count': os_cpus,
            'affinity_cpus': affinity, 'online_cpus': online,
            'cgroup_cpu_quota': min(quota_limits) if quota_limits else None,
            'total_mib': total / MIB, 'available_mib': available / MIB,
            'swap_mib': info.get('SwapTotal', 0) / MIB,
            'cgroup_visible': cgroup_visible, 'memory_source': 'proc_meminfo_and_visible_cgroup_min',
            'cgroup_note': 'visible limits applied' if cgroup_visible else 'cgroup limits unavailable; proc/sys allocation used'}


def host_admits(active_reservations, candidate, host, cpu_oversubscribe=2.0):
    """Allow CPU sharing while reserving full future RAM peaks and live growth."""
    try:
        share = _number(cpu_oversubscribe, positive=True)
        cpu = lambda r: (_number(r.get('cpu_slots', 6), positive=True)
                         + _number(r.get('loader_threads', 0)))
        ram = lambda r: _number(r.get('host_mib', r.get('host_reservation_mib')), positive=True)
        if sum(cpu(r) for r in active_reservations) + cpu(candidate) > _number(host['cpu_slots'], positive=True) * share:
            return False, 'CPU slots exhausted'
        reserved = sum(ram(r) for r in active_reservations)
        growth = sum(max(0, ram(r) - _number(r.get('rss_mib', 0))) for r in active_reservations)
        peak = ram(candidate)
        total = _number(host['total_mib'], positive=True)
        available = _number(host['available_mib'])
        system_reserve = _number(host.get('system_reserve_mib', max(2048, total * .05)))
        fraction = _number(host.get('memory_fraction', .85), positive=True)
        if fraction > 1:
            return False, 'Invalid host memory fraction'
        if peak + growth > max(0, available - system_reserve):
            return False, 'Insufficient available RAM for reserved worker growth'
        if reserved + peak > total * fraction:
            return False, 'Host RAM reservation budget exhausted'
        return True, 'admitted'
    except (KeyError, ValueError, TypeError):
        return False, 'Missing or invalid host resource reservation'


def process_tree_rss_mib(pid):
    seen = set()

    def visit(current):
        if current in seen:
            return 0
        seen.add(current)
        try:
            root = Path('/proc') / str(current)
            rss = next(int(line.split()[1]) * 1024 for line in (root / 'status').read_text().splitlines()
                       if line.startswith('VmRSS:'))
            children = (root / 'task' / str(current) / 'children').read_text().split()
            return rss + sum(visit(int(child)) for child in children)
        except (OSError, ValueError, StopIteration):
            return 0

    return visit(pid) / MIB


def _read_process(pid):
    root = Path('/proc') / str(pid)
    stat = (root / 'stat').read_text().rsplit(') ', 1)[1].split()
    environment = {}
    for entry in (root / 'environ').read_bytes().split(b'\0'):
        if b'=' in entry:
            key, value = entry.split(b'=', 1)
            if key in (b'CUDA_VISIBLE_DEVICES', ATTEMPT_ENV.encode()):
                environment[key.decode()] = value.decode()
    return {'start_ticks': int(stat[19]), 'state': stat[0],
            'command': [v.decode() for v in (root / 'cmdline').read_bytes().split(b'\0') if v],
            'environment': environment}


class _UnixHTTP(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(NAMESPACE_SOCKET)


class GPUOwnership:
    """Use namespace service only for identity; memory always comes from NVML."""

    def __init__(self, spec_path=None):
        self.spec_path = Path(spec_path).resolve() if spec_path else None
        self.pod_name = os.environ.get('HOSTNAME', '').strip() or (_read_text('/etc/hostname') or '')

    def record(self, pid, cmd, gpu_uuid, attempt):
        # A fresh child may not have exec'd yet, or may have already failed.
        # Missing identity is retained as unverified, never an admission exception.
        try:
            start = _read_process(pid)['start_ticks']
        except (OSError, ValueError, IndexError):
            start = None
        return {'pid': pid, 'start_ticks': start, 'command': list(cmd),
                'gpu_uuid': gpu_uuid, 'attempt': attempt}

    def _local(self, record):
        actual = _read_process(record['pid'])
        expected = record['command']
        if (record.get('start_ticks') is None or actual['start_ticks'] != record['start_ticks']
                or actual['state'] == 'Z' or actual['command'] != expected
                or actual['environment'].get('CUDA_VISIBLE_DEVICES') != record['gpu_uuid']
                or not record.get('attempt')
                or actual['environment'].get(ATTEMPT_ENV) != record['attempt']):
            raise ValueError('Worker command/start/GPU/attempt identity does not match')
        if self.spec_path is not None:
            if expected.count('--spec') != 1 or Path(expected[expected.index('--spec') + 1]).resolve() != self.spec_path:
                raise ValueError('Worker uses a different prepared specification')
        return actual

    def _platform_snapshot(self):
        import json
        connection = _UnixHTTP('localhost', timeout=3)
        try:
            connection.request('GET', '/info?tm=5')
            response = connection.getresponse()
            if response.status != 200:
                raise ValueError('GPU namespace query failed')
            body = response.read(16 * MIB + 1)
            if len(body) > 16 * MIB:
                raise ValueError('GPU namespace response is too large')
            payload = json.loads(body)
            if not isinstance(payload.get('Devices'), list):
                raise ValueError('GPU namespace response has no device list')
            return payload
        finally:
            connection.close()

    def account(self, snapshot, active):
        owned = {uuid: 0 for uuid in snapshot}
        evidence, before = [], {}
        for index, record in enumerate(active):
            try:
                before[index] = self._local(record)
            except (OSError, ValueError, KeyError, TypeError, IndexError):
                pass
        try:
            payload = self._platform_snapshot() if before else {'Devices': []}
            entries = [(d['DeviceUUID'], p) for d in payload['Devices']
                       for p in (d.get('ProcessInfos') or [])]
            host_claims = Counter(p.get('Pid') for _, p in entries)
        except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException):
            entries, host_claims = [], Counter()
        candidates = []
        local_claims = Counter(r.get('pid') for r in active)
        for index, record in enumerate(active):
            row = {'pid': record.get('pid'), 'gpu_uuid': record.get('gpu_uuid'), 'verified': False, 'owned_mib': 0}
            evidence.append(row)
            try:
                if index not in before or self._local(record) != before[index]:
                    raise ValueError('No stable local worker identity across the namespace query')
                uuid = record['gpu_uuid']
                if uuid not in snapshot or not self.pod_name or local_claims[record['pid']] != 1:
                    raise ValueError('Unknown GPU, pod, or duplicate local worker')
                matches = []
                for device, process in entries:
                    host_pid, nspids = process.get('Pid'), process.get('NSPids')
                    cmd = list(process.get('Cmdlines') or [])
                    while cmd and cmd[-1] == '':
                        cmd.pop()
                    if (type(host_pid) is int and host_pid > 0 and isinstance(nspids, list) and nspids
                            and nspids[0] == host_pid and nspids[-1] == record['pid']
                            and device == uuid and process.get('DeviceUUID', uuid) == uuid
                            and process.get('PodName') == self.pod_name and cmd == record['command']):
                        matches.append(host_pid)
                if len(matches) != 1 or host_claims[matches[0]] != 1:
                    raise ValueError('Namespace mapping is missing, ambiguous, or duplicated')
                host_pid = matches[0]
                observed = [p for p in snapshot[uuid]['processes'] if p.get('pid') == host_pid]
                if len(observed) != 1:
                    raise ValueError('Mapped host PID has no unique current NVML allocation')
                value = _number(observed[0]['used_mib'])
                if self._local(record) != before[index]:
                    raise ValueError('Worker identity changed during mapping')
                candidates.append((uuid, value, row))
                row.update(verified=True, host_pid=host_pid, owned_mib=value, reason='verified namespace/command/GPU/attempt')
            except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                row['reason'] = str(error)
        for uuid, value, row in candidates:
            owned[uuid] += value
        for uuid, sample in snapshot.items():
            if owned[uuid] > sample['used_mib']:
                owned[uuid] = 0
                for row in evidence:
                    if row['gpu_uuid'] == uuid:
                        row.update(verified=False, owned_mib=0, reason='Ownership exceeds aggregate NVML usage')
        return {'owned_mib': owned,
                'foreign_mib': {uuid: max(0, sample['used_mib'] - owned[uuid]) for uuid, sample in snapshot.items()},
                'evidence': evidence}

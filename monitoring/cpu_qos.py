"""Keep a launch-bound training process on the fast cores of a hybrid Intel CPU (baolong, i9-13980HX:
8 P-cores = logical 0-15, 16 E-cores = logical 16-31).

11-09 profile of a training run: the env step is one Python thread launching thousands of tiny CUDA kernels
(GPU 1-9 % busy meanwhile), and that thread sat largely on E-cores (P-cores carried 0.81 cores in total
while the main thread alone used 0.95). Windows' scheduler steers below-normal and EcoQoS threads to the
E-cores. `apply` turns EcoQoS off (PROCESS_POWER_THROTTLING_EXECUTION_SPEED, state 0 = high QoS), sets the
priority class and optionally restricts the process to a set of logical CPUs. Child processes started
afterwards inherit priority and affinity. No effect off Windows. Standard library only.

11-09 bench_launch --device cpu on baolong (beside a training): tiny op 8.5 us pinned to E-cores vs 2.7 on P;
a column step (245 envs, ~0.1 s of host work) 147 ms at below-normal, 86 pinned to P-cores, 31-32 ms at
normal priority with EcoQoS off: longer below-normal bursts get moved to the E-cores.
"""
import ctypes, os

PRIORITY = dict(idle=0x40, below=0x4000, normal=0x20, above=0x8000)


def cpu_sets():
    """{'p': mask, 'e': mask, 'all': mask} from GetSystemCpuSetInformation (EfficiencyClass > 0 = P-core).
    None when unknown (not Windows, or a CPU without efficiency classes)."""
    if os.name != 'nt':
        return None
    k = ctypes.windll.kernel32
    need = ctypes.c_ulong(0)
    k.GetSystemCpuSetInformation(None, 0, ctypes.byref(need), None, 0)
    buf = (ctypes.c_ubyte*need.value)()
    if not k.GetSystemCpuSetInformation(buf, need, ctypes.byref(need), None, 0):
        return None
    off, cls = 0, {}
    while off < need.value:
        size = int.from_bytes(bytes(buf[off:off+4]), 'little')
        # SYSTEM_CPU_SET_INFORMATION.CpuSet: Id(4) Group(2) LogicalProcessorIndex(1) CoreIndex(1)
        # LastLevelCacheIndex(1) NumaNodeIndex(1) EfficiencyClass(1) ... after Size(4) and Type(4)
        lp, eff = buf[off+14], buf[off+18]
        cls[lp] = eff
        off += size
    if not cls or len(set(cls.values())) < 2:
        return None
    top = max(cls.values())
    p = sum(1 << i for i, c in cls.items() if c == top)
    a = sum(1 << i for i in cls)
    return dict(p=p, e=a & ~p, all=a)


def apply(priority='normal', cores='all', high_qos=True, pid=None):
    """Set the priority class, EcoQoS opt-out and CPU affinity ('all', 'p', 'e' or an int mask) of this
    process, or of process `pid` (an already running training: no restart, results unchanged).
    Returns what was applied (for run.json)."""
    if os.name != 'nt':
        return dict(applied=False)
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = ctypes.c_void_p      # the pseudo handle -1 must not be truncated to 32 bits
    k.OpenProcess.restype = ctypes.c_void_p
    if pid is None:
        h = ctypes.c_void_p(k.GetCurrentProcess())
    else:   # PROCESS_SET_INFORMATION | PROCESS_QUERY_INFORMATION
        h = ctypes.c_void_p(k.OpenProcess(0x0200 | 0x0400, False, int(pid)))
        if not h.value:
            return dict(pid=pid, error=f'OpenProcess failed ({k.GetLastError()})')
    out = dict(priority=priority, cores=cores, high_qos=high_qos)
    out['priority_ok'] = bool(k.SetPriorityClass(h, PRIORITY[priority]))
    if high_qos:
        class PPTS(ctypes.Structure):
            _fields_ = [('Version', ctypes.c_ulong), ('ControlMask', ctypes.c_ulong), ('StateMask', ctypes.c_ulong)]
        st = PPTS(1, 1, 0)                     # control EXECUTION_SPEED, state off: never throttled
        out['qos_ok'] = bool(k.SetProcessInformation(h, 4, ctypes.byref(st), ctypes.sizeof(st)))
    if cores != 'all':
        sets = cpu_sets()
        mask = cores if isinstance(cores, int) else (sets or {}).get(cores)
        if mask:
            out['mask'] = hex(mask)
            out['affinity_ok'] = bool(k.SetProcessAffinityMask(h, ctypes.c_size_t(mask)))
    if pid is not None:
        k.CloseHandle(h); out['pid'] = pid
    return out


def children(pid):
    """Direct and indirect child pids (Toolhelp snapshot), e.g. the TE worker processes of a training."""
    k = ctypes.windll.kernel32

    class PE(ctypes.Structure):
        _fields_ = [('dwSize', ctypes.c_ulong), ('cntUsage', ctypes.c_ulong), ('th32ProcessID', ctypes.c_ulong),
                    ('th32DefaultHeapID', ctypes.c_size_t), ('th32ModuleID', ctypes.c_ulong), ('cntThreads', ctypes.c_ulong),
                    ('th32ParentProcessID', ctypes.c_ulong), ('pcPriClassBase', ctypes.c_long), ('dwFlags', ctypes.c_ulong),
                    ('szExeFile', ctypes.c_char*260)]
    k.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    snap = ctypes.c_void_p(k.CreateToolhelp32Snapshot(2, 0))
    e = PE(); e.dwSize = ctypes.sizeof(PE)
    parent = {}
    ok = k.Process32First(snap, ctypes.byref(e))
    while ok:
        parent[e.th32ProcessID] = e.th32ParentProcessID
        ok = k.Process32Next(snap, ctypes.byref(e))
    k.CloseHandle(snap)
    out, todo = [], [int(pid)]
    while todo:
        p = todo.pop()
        born = _created(p)
        # Windows reuses pids: a live process whose parent pid was recycled is not a child (created earlier)
        kids = [c for c, q in parent.items() if q == p and c != p and born is not None and (_created(c) or 0) >= born]
        out += kids; todo += kids
    return out


def _created(pid):
    """Creation time of a process (FILETIME as int), None when it cannot be opened."""
    k = ctypes.windll.kernel32
    k.OpenProcess.restype = ctypes.c_void_p
    h = ctypes.c_void_p(k.OpenProcess(0x1000, False, int(pid)))     # PROCESS_QUERY_LIMITED_INFORMATION
    if not h.value:
        return None
    t = [ctypes.c_ulonglong() for _ in range(4)]
    ok = k.GetProcessTimes(h, *[ctypes.byref(x) for x in t])
    k.CloseHandle(h)
    return t[0].value if ok else None


def main():
    import argparse, json
    ap = argparse.ArgumentParser(description='apply a placement to a running process tree (e.g. a training started '
                                             'at below-normal): python -m monitoring.cpu_qos --pid 1234 --tree')
    ap.add_argument('--pid', type=int, required=True)
    ap.add_argument('--tree', action='store_true', help='also its child processes (TE workers)')
    ap.add_argument('--priority', default='normal', choices=list(PRIORITY))
    ap.add_argument('--cores', default='p')
    ap.add_argument('--no-qos', action='store_true')
    a = ap.parse_args()
    pids = [a.pid]+(children(a.pid) if a.tree else [])
    for p in pids:
        print(json.dumps(apply(a.priority, a.cores, not a.no_qos, pid=p)), flush=True)


if __name__ == '__main__':
    main()

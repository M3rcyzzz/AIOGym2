"""24x7 WSL2 local GPU job runner, thermal heartbeat monitor, and seamless checkpoint handoff.

Manages detached training jobs on `guomao` (`Ubuntu-24.04`, `/root/AIOGym2`) and `baolong`
(`Ubuntu`, `/root/AIOGym2`) over either Windows-proxied WSL2 SSH or native Linux Tailscale SSH
(`guomao-wsl` / `baolong-wsl`). Uses fast Linux ext4 caches (`~/.cache/aio/{inductor,triton,val-cache}`),
survives Mac sleep and WAN drops via `systemd --user` or `setsid nohup`, tracks GPU thermals into
`heartbeat.json` (warning at >= 84 C), and handles verified Colab <-> local checkpoint handoffs.

  python -m monitoring.local_job start guomao sq-g1 -- python -m aio.train --out {run} ...
  python -m monitoring.local_job status guomao sq-g1
  python -m monitoring.local_job stop guomao sq-g1
  python -m monitoring.local_job sync guomao sq-g1
  python -m monitoring.local_job handoff --from colab:sq-pure-10 --to local:guomao:sq-pure-10
  python -m monitoring.local_job setup-wsl guomao [--apply]
"""
import argparse
import base64
import json
import math
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

COLAB_ROOT = Path(os.environ.get('AIO_COLAB_ROOT', Path.home() / 'colab-runs'))
LOCAL_SYNC_ROOT = Path(os.environ.get('AIO_LOCAL_SYNC_ROOT', Path.home() / 'local-runs'))
THERMAL_WARN_C = 84.0
DEFAULT_ENV = {
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    'AIO_SEQ_AMP': '1',
    'AIO_SEQ_ROOMS': '1',
    'PYTHONUNBUFFERED': '1',
}
HOSTS = {
    'guomao': dict(
        ssh='guomao', wsl_distro='Ubuntu-24.04', wsl_user='root', native=False,
        repo='/root/AIOGym2', python='/root/AIOGym2/.venv/bin/python', runs_root='/root/aio-runs',
        native_alias='guomao-wsl', default_sizes='6:12x5@72 1:24x10@28 1:48x20@16', default_accum=1,
        default_quiet_hours='00:00-10:30',
    ),
    'guomao-wsl': dict(
        ssh='guomao-wsl', wsl_distro=None, wsl_user=None, native=True,
        repo='/root/AIOGym2', python='/root/AIOGym2/.venv/bin/python', runs_root='/root/aio-runs',
        default_sizes='6:12x5@72 1:24x10@28 1:48x20@16', default_accum=1,
        default_quiet_hours='00:00-10:30',
    ),
    'baolong': dict(
        ssh='baolong', wsl_distro='Ubuntu', wsl_user='root', native=False,
        repo='/root/AIOGym2', python='/root/AIOGym2/.venv/bin/python', runs_root='/root/aio-runs',
        native_alias='baolong-wsl', default_sizes='6:12x5@44 1:24x10@20 1:48x20@8', default_accum=2,
        default_quiet_hours=None,
    ),
    'baolong-wsl': dict(
        ssh='baolong-wsl', wsl_distro=None, wsl_user=None, native=True,
        repo='/root/AIOGym2', python='/root/AIOGym2/.venv/bin/python', runs_root='/root/aio-runs',
        default_sizes='6:12x5@44 1:24x10@20 1:48x20@8', default_accum=2,
        default_quiet_hours=None,
    ),
}

REMOTE_RUNNER = r'''
import fcntl, json, os, signal, subprocess, sys, threading, time
from pathlib import Path

J = Path(sys.argv[1])
held = open(J / 'runner.lock', 'w')
try:
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    sys.exit('another local runner already owns this job')

spec = json.loads((J / 'job.json').read_text())
run_dir = J / 'run'
in_dir = J / 'in'
cache_root = Path.home() / '.cache' / 'aio'
for d in (run_dir, in_dir, cache_root / 'inductor', cache_root / 'triton', cache_root / 'val-cache'):
    d.mkdir(parents=True, exist_ok=True)

env = dict(os.environ)
env.update(spec.get('env') or {})
env.setdefault('TORCHINDUCTOR_CACHE_DIR', str(cache_root / 'inductor'))
env.setdefault('TRITON_CACHE_DIR', str(cache_root / 'triton'))
env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
env.setdefault('AIO_SEQ_AMP', '1')
env.setdefault('AIO_SEQ_ROOMS', '1')
env['PYTHONUNBUFFERED'] = '1'
env['PYTHONPATH'] = spec['repo']

cmd = [
    x.replace('{run}', str(run_dir))
     .replace('{in}', str(in_dir))
     .replace('{val_cache}', str(cache_root / 'val-cache'))
    for x in spec['cmd']
]
warn_temp = float(spec.get('thermal_warn_c', 84.0))
log_lock = threading.Lock()
log = open(J / 'job.log', 'a', buffering=1, errors='replace')


def write_log(msg):
    with log_lock:
        log.write(time.strftime('%Y-%m-%dT%H:%M:%SZ ', time.gmtime()) + msg + '\n')


def parse_smi(line):
    parts = [x.strip() for x in (line or '').split(',')]
    if len(parts) < 6:
        return {}
    def num(s):
        try:
            return float(s)
        except ValueError:
            return None
    throttle = parts[6].lower() in ('active', '1', 'true') if len(parts) > 6 else False
    return {
        'temp_c': num(parts[0]),
        'power_w': num(parts[1]),
        'power_limit_w': num(parts[2]),
        'util_pct': num(parts[3]),
        'mem_used_mb': num(parts[4]),
        'mem_total_mb': num(parts[5]),
        'thermal_throttle': throttle,
    }


def write_heartbeat(pid, running):
    smi_line = ''
    try:
        r = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.total,clocks_throttle_reasons.hw_thermal_slowdown',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            smi_line = (r.stdout.strip().splitlines() or [''])[0]
    except Exception:
        pass
    gpu = parse_smi(smi_line)
    temp = gpu.get('temp_c')
    warn = bool((temp is not None and temp >= warn_temp) or gpu.get('thermal_throttle'))
    msg = f'GPU thermal warning: {temp} C >= {warn_temp} C' if warn else None
    hb = dict(
        host=spec.get('host'), job=spec.get('name'), pid=pid, running=running,
        time=round(time.time(), 3), gpu=gpu, thermal_warning=warn, warning_msg=msg,
    )
    tmp = J / 'heartbeat.json.tmp'
    tmp.write_text(json.dumps(hb, indent=1) + '\n')
    os.replace(tmp, J / 'heartbeat.json')
    if warn:
        write_log(f'[thermal] {msg}')


write_log('start ' + json.dumps(cmd))
proc = subprocess.Popen(
    cmd, cwd=spec['repo'], env=env, stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace',
    start_new_session=True,
)
(J / 'pid').write_text(str(proc.pid))
stop_hb = threading.Event()


def hb_loop():
    while not stop_hb.is_set():
        write_heartbeat(proc.pid, proc.poll() is None)
        stop_hb.wait(30.0)


def handle_term(signum, _frame):
    write_log(f'received signal {signum}, forwarding SIGTERM to child group {proc.pid}')
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        pass


signal.signal(signal.SIGTERM, handle_term)
t_hb = threading.Thread(target=hb_loop, daemon=True)
t_hb.start()
for line in proc.stdout:
    write_log(line.rstrip('\n'))
rc = proc.wait()
stop_hb.set()
write_heartbeat(proc.pid, False)
write_log(f'exit {rc}')
tmp_ex = J / 'exit.tmp'
tmp_ex.write_text(json.dumps(dict(code=rc, time=time.time())))
os.replace(tmp_ex, J / 'exit')
'''


def resolve_host(host, wsl_distro=None, native_ssh=False, repo=None, python_bin=None, runs_root=None):
    """Resolve SSH transport profile for guomao, baolong, native *-wsl aliases, or custom Linux SSH hosts."""
    if host in HOSTS:
        cfg = dict(HOSTS[host])
    else:
        cfg = dict(
            ssh=host, wsl_distro=wsl_distro, wsl_user='root' if wsl_distro else None,
            native=bool(native_ssh or host.endswith('-wsl') or not wsl_distro),
            repo='/root/AIOGym2', python='/root/AIOGym2/.venv/bin/python', runs_root='/root/aio-runs',
            default_sizes='6:12x5@48 1:24x10@24 1:48x20@10', default_accum=2,
        )
    if native_ssh:
        cfg['native'] = True
        cfg['wsl_distro'] = None
    if wsl_distro:
        cfg['wsl_distro'] = wsl_distro
        cfg['wsl_user'] = cfg.get('wsl_user') or 'root'
        cfg['native'] = False
    if repo:
        cfg['repo'] = repo
    if python_bin:
        cfg['python'] = python_bin
    if runs_root:
        cfg['runs_root'] = runs_root
    return cfg


def build_ssh_cmd(host_cfg, remote_cmd=None, connect_timeout=10):
    base = ['ssh', '-o', 'BatchMode=yes', '-o', f'ConnectTimeout={int(connect_timeout)}', host_cfg['ssh']]
    if host_cfg['native'] or not host_cfg.get('wsl_distro'):
        return base + ([remote_cmd] if remote_cmd else ['bash', '-s'])
    user = host_cfg.get('wsl_user') or 'root'
    wsl = f'wsl -d {host_cfg["wsl_distro"]} -u {user} --'
    if remote_cmd:
        return base + [f'{wsl} {remote_cmd}']
    return base + [f'{wsl} bash -s']


def ssh_bash(host_cfg, script, timeout=60, check=True, connect_timeout=10):
    """Execute a bash script over SSH (piped via stdin so Windows ConPTY never mangles quoting)."""
    cmd = build_ssh_cmd(host_cfg, connect_timeout=connect_timeout)
    return subprocess.run(cmd, input=script, text=True, encoding='utf-8', errors='replace',
                          capture_output=True, timeout=timeout, check=check)


def parse_nvidia_smi(line):
    """Parse one CSV line from nvidia-smi into numeric GPU telemetry dict."""
    parts = [x.strip() for x in (line or '').split(',')]
    if len(parts) < 6:
        return {}

    def _num(s):
        try:
            v = float(s)
            return v if math.isfinite(v) else None
        except ValueError:
            return None

    throttle = parts[6].lower() in ('active', '1', 'true') if len(parts) > 6 else False
    return {
        'temp_c': _num(parts[0]),
        'power_w': _num(parts[1]),
        'power_limit_w': _num(parts[2]),
        'util_pct': _num(parts[3]),
        'mem_used_mb': _num(parts[4]),
        'mem_total_mb': _num(parts[5]),
        'thermal_throttle': throttle,
    }


def build_heartbeat(host, job, pid, running, smi_line='', now=None, warn_temp=THERMAL_WARN_C):
    """Build structured heartbeat.json dict with thermal warning threshold (>= 84 C)."""
    gpu = parse_nvidia_smi(smi_line)
    temp = gpu.get('temp_c')
    warn = bool((temp is not None and temp >= warn_temp) or gpu.get('thermal_throttle'))
    msg = None
    if warn:
        if temp is not None and temp >= warn_temp:
            msg = f'GPU temperature {temp:.1f} C >= {warn_temp:.1f} C thermal warning threshold'
        else:
            msg = 'GPU hardware thermal slowdown active'
    return {
        'host': host,
        'job': job,
        'pid': int(pid) if pid is not None else None,
        'running': bool(running),
        'time': round(time.time() if now is None else float(now), 3),
        'gpu': gpu,
        'thermal_warning': warn,
        'warning_msg': msg,
    }


def _safe_torch_meta(path):
    """Inspect lightweight metadata from a .pt checkpoint without moving tensors to GPU."""
    try:
        import torch
    except ImportError:
        return {}
    sys.modules.setdefault('pathlib._local', pathlib)
    saved = pathlib.WindowsPath
    if os.name != 'nt':
        pathlib.WindowsPath = pathlib.PureWindowsPath
    try:
        ck = torch.load(path, map_location='cpu', weights_only=False)
    finally:
        pathlib.WindowsPath = saved
    if not isinstance(ck, dict):
        return {}
    cfg = ck.get('config') if isinstance(ck.get('config'), dict) else {}
    return {
        'iter': ck.get('iter'),
        'updates': ck.get('updates'),
        'samples': ck.get('samples'),
        'hours': ck.get('hours'),
        'guard_units': ck.get('guard_units'),
        'cosine_by': cfg.get('cosine_by'),
        'total_updates': cfg.get('total_updates'),
        'lr': cfg.get('lr'),
        'seq': cfg.get('seq'),
    }


def verify_checkpoint(path, inspect_torch=True):
    """Verify a local last.pt exists, is non-empty, is a valid zip archive, and read key counters."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'checkpoint not found: {p}')
    size = p.stat().st_size
    if size <= 0:
        raise ValueError(f'checkpoint is empty: {p}')
    if not zipfile.is_zipfile(p):
        raise ValueError(f'checkpoint is not a valid PyTorch zip archive: {p}')
    meta = dict(path=str(p), bytes=size, valid_zip=True)
    if inspect_torch:
        try:
            meta.update({k: v for k, v in _safe_torch_meta(p).items() if v is not None})
        except Exception as exc:
            meta['torch_inspect_note'] = f'{type(exc).__name__}: {exc}'
    return meta


def parse_endpoint(spec, colab_root=None):
    """Parse `colab:<run>`, `local:<host>:<run>`, or `path:<dir>` into a structured endpoint dict."""
    croot = Path(colab_root) if colab_root is not None else COLAB_ROOT
    if spec.startswith('colab:'):
        name = spec.split(':', 1)[1].strip()
        if not name:
            raise ValueError(f'invalid colab endpoint: {spec!r}')
        return dict(spec=spec, kind='colab', name=name, run_dir=croot / name / 'run', job_dir=croot / name)
    if spec.startswith('local:'):
        parts = spec.split(':', 2)
        if len(parts) != 3 or not parts[1] or not parts[2]:
            raise ValueError(f'invalid local endpoint (expected local:<host>:<run>): {spec!r}')
        host, name = parts[1].strip(), parts[2].strip()
        hcfg = resolve_host(host)
        return dict(spec=spec, kind='local', host=host, host_cfg=hcfg, name=name,
                    remote_job_dir=f'{hcfg["runs_root"]}/{name}',
                    remote_run_dir=f'{hcfg["runs_root"]}/{name}/run')
    if spec.startswith('path:'):
        p = Path(spec.split(':', 1)[1]).expanduser()
        return dict(spec=spec, kind='path', name=p.name, run_dir=p, job_dir=p.parent)
    raise ValueError(f'endpoint must start with colab:, local:<host>:, or path:, got {spec!r}')


def _append_handoff_audit_local(run_dir, audit):
    metrics = Path(run_dir) / 'metrics.jsonl'
    metrics.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({'handoff': audit}, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n'
    with open(metrics, 'a', encoding='utf-8') as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _append_handoff_audit_remote(host_cfg, remote_run_dir, audit):
    line = json.dumps({'handoff': audit}, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    script = (
        f'mkdir -p {shlex.quote(remote_run_dir)}\n'
        f'printf %s\\\\n {shlex.quote(line)} >> {shlex.quote(remote_run_dir + "/metrics.jsonl")}\n'
    )
    ssh_bash(host_cfg, script, timeout=30)


def _transfer_file_to_remote(local_file, host_cfg, remote_file, timeout=1800):
    local_file = Path(local_file)
    remote_dir = os.path.dirname(remote_file)
    if host_cfg['native'] or not host_cfg.get('wsl_distro'):
        ssh_bash(host_cfg, f'mkdir -p {shlex.quote(remote_dir)}\n', timeout=30)
        subprocess.run(
            ['rsync', '-az', '--partial', '--inplace', str(local_file), f'{host_cfg["ssh"]}:{remote_file}'],
            check=True, capture_output=True, text=True, timeout=timeout,
        )
        return
    # Windows-proxied WSL2: avoid cmd.exe quoting/&& metacharacters by mkdir -> dd -> mv
    tmp = f'{remote_file}.tmp'
    ssh_bash(host_cfg, f'mkdir -p {shlex.quote(remote_dir)}\n', timeout=30)
    cmd = build_ssh_cmd(host_cfg, remote_cmd=f'dd of={tmp} bs=4M status=none')
    with open(local_file, 'rb') as f:
        p = subprocess.run(cmd, stdin=f, capture_output=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f'remote write failed for {remote_file}: {p.stderr.decode("utf-8", "replace")[-400:]}')
    ssh_bash(host_cfg, f'mv -f {shlex.quote(tmp)} {shlex.quote(remote_file)}\n', timeout=30)


def _transfer_file_from_remote(host_cfg, remote_file, local_file, timeout=1800):
    local_file = Path(local_file)
    local_file.parent.mkdir(parents=True, exist_ok=True)
    if host_cfg['native'] or not host_cfg.get('wsl_distro'):
        subprocess.run(
            ['rsync', '-az', '--partial', '--inplace', f'{host_cfg["ssh"]}:{remote_file}', str(local_file)],
            check=True, capture_output=True, text=True, timeout=timeout,
        )
        return
    tmp = local_file.parent / f'.{local_file.name}.tmp.{os.getpid()}'
    cmd = build_ssh_cmd(host_cfg, remote_cmd=f'cat {remote_file}')
    with open(tmp, 'wb') as out:
        p = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, timeout=timeout)
    if p.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f'remote read failed for {remote_file}: {p.stderr.decode("utf-8", "replace")[-400:]}')
    os.replace(tmp, local_file)


def _remote_verify_checkpoint(host_cfg, remote_last):
    script = f'''{shlex.quote(host_cfg["python"])} - <<'PY'
import json, os, pathlib, sys, zipfile
p = {remote_last!r}
if not os.path.isfile(p):
    raise SystemExit('missing: ' + p)
sz = os.path.getsize(p)
if sz <= 0 or not zipfile.is_zipfile(p):
    raise SystemExit('invalid zip: ' + p)
meta = dict(path=p, bytes=sz, valid_zip=True)
try:
    import torch
    sys.modules.setdefault('pathlib._local', pathlib)
    ck = torch.load(p, map_location='cpu', weights_only=False)
    if isinstance(ck, dict):
        cfg = ck.get('config') if isinstance(ck.get('config'), dict) else {{}}
        for k in ('iter', 'updates', 'samples', 'hours', 'guard_units'):
            if ck.get(k) is not None:
                meta[k] = ck[k]
        for k in ('cosine_by', 'total_updates', 'lr', 'seq'):
            if cfg.get(k) is not None:
                meta[k] = cfg[k]
except Exception as exc:
    meta['torch_inspect_note'] = repr(exc)
print(json.dumps(meta))
PY
'''
    p = ssh_bash(host_cfg, script, timeout=180)
    return json.loads(p.stdout.strip().splitlines()[-1])


def recommend_resume_flags(dst_ep, meta, total_updates=None, lr=None, accum=None):
    """Build exact recommended resume CLI flags for the target machine."""
    cur_updates = int(meta.get('updates') or 0)
    target_updates = total_updates or meta.get('total_updates') or max(cur_updates + 3000, 6000)
    use_lr = lr if lr is not None else (meta.get('lr') or 6e-5)
    flags = [
        '--resume',
        '--cosine-by', 'updates',
        '--total-updates', str(int(target_updates)),
        '--lr', f'{float(use_lr):g}',
        '--card-alloc', 'expandable_segments:True,garbage_collection_threshold:0.8',
    ]
    if dst_ep['kind'] == 'colab':
        flags += ['--eval-every', '0', '--eval-snap-every', '10', '--keep-eval-snaps', '6']
    elif dst_ep['kind'] == 'local':
        hcfg = dst_ep['host_cfg']
        acc = accum if accum is not None else hcfg.get('default_accum', 1)
        flags += ['--accum', str(int(acc)), '--val-cache', '$HOME/.cache/aio/val-cache']
        if hcfg.get('default_quiet_hours'):
            flags += ['--quiet-hours', str(hcfg['default_quiet_hours'])]
    return flags


def handoff(src_spec, dst_spec, *, colab_root=None, total_updates=None, lr=None, accum=None,
            verify_only=False, dry_run=False, inspect_torch=True):
    """Safely verify and transfer last.pt + run.json + metrics.jsonl (+ eval.jsonl) across machines."""
    src = parse_endpoint(src_spec, colab_root=colab_root)
    dst = parse_endpoint(dst_spec, colab_root=colab_root)
    files = ('last.pt', 'run.json', 'metrics.jsonl', 'eval.jsonl')

    # 1. Verify source checkpoint integrity
    if src['kind'] in ('colab', 'path'):
        meta = verify_checkpoint(src['run_dir'] / 'last.pt', inspect_torch=inspect_torch)
    else:
        meta = _remote_verify_checkpoint(src['host_cfg'], f'{src["remote_run_dir"]}/last.pt')

    resume_flags = recommend_resume_flags(dst, meta, total_updates=total_updates, lr=lr, accum=accum)
    result = dict(
        src=src_spec, dst=dst_spec, verified=meta, transferred=[],
        resume_flags=resume_flags, dry_run=bool(dry_run or verify_only),
    )
    if verify_only or dry_run:
        return result

    # 2. Transfer files (staging via local tmp dir if remote -> remote)
    transferred = []
    if src['kind'] in ('colab', 'path') and dst['kind'] in ('colab', 'path'):
        dst['run_dir'].mkdir(parents=True, exist_ok=True)
        for name in files:
            s_path = src['run_dir'] / name
            if not s_path.is_file():
                continue
            d_path = dst['run_dir'] / name
            if shutil.which('rsync'):
                subprocess.run(
                    ['rsync', '-a', '--partial', '--inplace', str(s_path), str(d_path)],
                    check=True, capture_output=True, text=True, timeout=600,
                )
            else:
                tmp = d_path.parent / f'.{name}.tmp.{os.getpid()}'
                shutil.copy2(s_path, tmp)
                os.replace(tmp, d_path)
            transferred.append(name)
        dst_sz = (dst['run_dir'] / 'last.pt').stat().st_size
        if dst_sz != meta['bytes']:
            raise RuntimeError(f'handoff size mismatch: src={meta["bytes"]} dst={dst_sz}')
    elif src['kind'] in ('colab', 'path') and dst['kind'] == 'local':
        for name in files:
            s_path = src['run_dir'] / name
            if not s_path.is_file():
                continue
            _transfer_file_to_remote(s_path, dst['host_cfg'], f'{dst["remote_run_dir"]}/{name}')
            transferred.append(name)
        dst_meta = _remote_verify_checkpoint(dst['host_cfg'], f'{dst["remote_run_dir"]}/last.pt')
        if dst_meta['bytes'] != meta['bytes']:
            raise RuntimeError(f'handoff size mismatch: src={meta["bytes"]} dst={dst_meta["bytes"]}')
    elif src['kind'] == 'local' and dst['kind'] in ('colab', 'path'):
        dst['run_dir'].mkdir(parents=True, exist_ok=True)
        for name in files:
            r_path = f'{src["remote_run_dir"]}/{name}'
            chk = ssh_bash(src['host_cfg'], f'test -f {shlex.quote(r_path)} && echo OK || true\n', timeout=30)
            if 'OK' not in chk.stdout:
                continue
            _transfer_file_from_remote(src['host_cfg'], r_path, dst['run_dir'] / name)
            transferred.append(name)
        dst_meta = verify_checkpoint(dst['run_dir'] / 'last.pt', inspect_torch=False)
        if dst_meta['bytes'] != meta['bytes']:
            raise RuntimeError(f'handoff size mismatch: src={meta["bytes"]} dst={dst_meta["bytes"]}')
    else:
        with tempfile.TemporaryDirectory(prefix='aio-handoff-') as td:
            td_path = Path(td)
            for name in files:
                r_src = f'{src["remote_run_dir"]}/{name}'
                chk = ssh_bash(src['host_cfg'], f'test -f {shlex.quote(r_src)} && echo OK || true\n', timeout=30)
                if 'OK' not in chk.stdout:
                    continue
                _transfer_file_from_remote(src['host_cfg'], r_src, td_path / name)
                _transfer_file_to_remote(td_path / name, dst['host_cfg'], f'{dst["remote_run_dir"]}/{name}')
                transferred.append(name)
            dst_meta = _remote_verify_checkpoint(dst['host_cfg'], f'{dst["remote_run_dir"]}/last.pt')
            if dst_meta['bytes'] != meta['bytes']:
                raise RuntimeError(f'handoff size mismatch: src={meta["bytes"]} dst={dst_meta["bytes"]}')

    # 3. Append structured handoff audit line to destination metrics.jsonl
    audit = {
        'from': src_spec,
        'to': dst_spec,
        'time': round(time.time(), 3),
        'bytes': meta['bytes'],
        'iter': meta.get('iter'),
        'updates': meta.get('updates'),
        'samples': meta.get('samples'),
        'hours': meta.get('hours'),
    }
    audit = {k: v for k, v in audit.items() if v is not None}
    if dst['kind'] in ('colab', 'path'):
        _append_handoff_audit_local(dst['run_dir'], audit)
    else:
        _append_handoff_audit_remote(dst['host_cfg'], dst['remote_run_dir'], audit)

    result['transferred'] = transferred
    result['audit'] = audit
    return result


def build_win32_wsl_launch_cmd(host_cfg, job_dir):
    """Build a PowerShell -EncodedCommand invoking Win32_Process.Create for wsl.exe.

    Keeping wsl.exe open as a detached Windows process prevents WSL2's 8-second
    vmIdleTimeout from shutting down the Linux distro after SSH disconnects.
    """
    distro = host_cfg['wsl_distro']
    user = host_cfg.get('wsl_user') or 'root'
    py = host_cfg['python']
    runner = f'{job_dir}/runner.py'
    cmdline = f'wsl.exe -d {distro} -u {user} -- {py} {runner} {job_dir}'
    ps = (
        f"$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
        f"-Arguments @{{CommandLine='{cmdline}'}}; "
        f"Write-Output ('WINPID ' + $r.ProcessId + ' RC ' + $r.ReturnValue)"
    )
    b64 = base64.b64encode(ps.encode('utf-16le')).decode('ascii')
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host_cfg['ssh'],
            f'powershell -NoProfile -EncodedCommand {b64}']


def start_job(host_cfg, host_name, job_name, cmd, *, ref=None, put_files=(), use_systemd=False,
              extra_env=None, thermal_warn_c=THERMAL_WARN_C, quiet_hours=None):
    """Launch a detached 24x7 training job inside WSL2 with ext4 caches, quiet-hours guard, and GPU thermal heartbeat."""
    job_dir = f'{host_cfg["runs_root"]}/{job_name}'
    env = dict(DEFAULT_ENV)
    if extra_env:
        env.update(extra_env)
    cmd = list(cmd)
    qh = host_cfg.get('default_quiet_hours') if quiet_hours is None else quiet_hours
    if qh and any('aio.train' in str(x) for x in cmd) and '--quiet-hours' not in cmd:
        cmd.extend(['--quiet-hours', str(qh)])
    spec = dict(
        name=job_name, host=host_name, repo=host_cfg['repo'], cmd=cmd,
        env=env, thermal_warn_c=float(thermal_warn_c), quiet_hours=qh, started=time.time(),
    )
    setup = [
        'set -euo pipefail',
        f'mkdir -p {shlex.quote(job_dir + "/in")} {shlex.quote(job_dir + "/run")} '
        '"$HOME/.cache/aio/inductor" "$HOME/.cache/aio/triton" "$HOME/.cache/aio/val-cache"',
    ]
    if ref:
        setup += [
            f'cd {shlex.quote(host_cfg["repo"])}',
            f'git fetch origin {shlex.quote(ref)} && git checkout {shlex.quote(ref)}',
        ]
    ssh_bash(host_cfg, '\n'.join(setup) + '\n', timeout=180)

    for pf in put_files:
        p = Path(pf).expanduser()
        _transfer_file_to_remote(p, host_cfg, f'{job_dir}/in/{p.name}')

    spec_json = json.dumps(spec, indent=1)
    unit = f'aio-{job_name}'.replace('/', '-')

    stage_script = f'''set -euo pipefail
cat > {shlex.quote(job_dir + "/job.json")} <<'JOBEOF'
{spec_json}
JOBEOF
cat > {shlex.quote(job_dir + "/runner.py")} <<'RUNEOF'
{REMOTE_RUNNER}
RUNEOF
rm -f {shlex.quote(job_dir + "/exit")} {shlex.quote(job_dir + "/pid")}
'''

    if not host_cfg['native'] and host_cfg.get('wsl_distro'):
        # Windows-proxied WSL2: write files first, then spawn detached wsl.exe via Win32_Process.Create
        # so Windows keeps the WSL2 VM alive for the full duration of runner.py.
        ssh_bash(host_cfg, stage_script, timeout=60)
        wp = subprocess.run(build_win32_wsl_launch_cmd(host_cfg, job_dir),
                            text=True, encoding='utf-8', errors='replace',
                            capture_output=True, timeout=60, check=True)
        time.sleep(2.0)
        p_pid = ssh_bash(host_cfg, f'cat {shlex.quote(job_dir + "/pid")} 2>/dev/null || echo pending\n', timeout=30)
        out = f'STARTED {p_pid.stdout.strip()} ({wp.stdout.strip()})'
        return dict(host=host_name, job=job_name, job_dir=job_dir, output=out)

    if use_systemd:
        launch_line = (
            f'if command -v systemd-run >/dev/null 2>&1 && systemctl --user is-system-running >/dev/null 2>&1; then '
            f'systemd-run --user --unit={shlex.quote(unit)} --property=Restart=on-failure --property=RestartSec=15 '
            f'{shlex.quote(host_cfg["python"])} {shlex.quote(job_dir + "/runner.py")} {shlex.quote(job_dir)}; '
            f'else setsid nohup {shlex.quote(host_cfg["python"])} {shlex.quote(job_dir + "/runner.py")} '
            f'{shlex.quote(job_dir)} </dev/null >/dev/null 2>&1 & fi'
        )
    else:
        launch_line = (
            f'setsid nohup {shlex.quote(host_cfg["python"])} {shlex.quote(job_dir + "/runner.py")} '
            f'{shlex.quote(job_dir)} </dev/null >/dev/null 2>&1 &'
        )

    launch_script = (
        stage_script
        + f'{launch_line}\n'
        + 'sleep 1\n'
        + f'echo "STARTED $(cat {shlex.quote(job_dir + "/pid")} 2>/dev/null || echo pending)"\n'
    )
    p = ssh_bash(host_cfg, launch_script, timeout=60)
    return dict(host=host_name, job=job_name, job_dir=job_dir, output=p.stdout.strip())


def status_job(host_cfg, job_name=None, timeout=15):
    """Query status, heartbeat.json, last metrics row, and latest eval row from remote WSL2 host."""
    runs_root = host_cfg['runs_root']
    script = f'''{shlex.quote(host_cfg["python"])} - <<'PY'
import json, os, time
from pathlib import Path

root = Path({runs_root!r})
target = {job_name!r}
dirs = [root / target] if target else (sorted(p.parent for p in root.glob('*/job.json')) if root.exists() else [])
out = []
for d in dirs:
    if not d.exists():
        continue
    def read_json(p):
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    def tail_jsonl(p, n=3):
        try:
            lines = p.read_text(errors='replace').splitlines()[-n-2:]
        except OSError:
            return []
        rows = []
        for x in lines:
            try:
                r = json.loads(x)
                if isinstance(r, dict):
                    rows.append(r)
            except ValueError:
                pass
        return rows[-n:]
    pid_s = (d / 'pid').read_text().strip() if (d / 'pid').exists() else ''
    alive = False
    if pid_s.isdigit():
        try:
            st = Path(f'/proc/{{pid_s}}/stat').read_text()
            alive = st.rsplit(')', 1)[1].split()[0] not in 'ZX'
        except OSError:
            alive = False
    out.append(dict(
        job=d.name,
        running=alive,
        pid=int(pid_s) if pid_s.isdigit() else None,
        spec=read_json(d / 'job.json'),
        heartbeat=read_json(d / 'heartbeat.json'),
        exit=read_json(d / 'exit'),
        metrics_tail=tail_jsonl(d / 'run' / 'metrics.jsonl', 6),
        eval_tail=tail_jsonl(d / 'run' / 'eval.jsonl', 3),
        best_eval=read_json(d / 'run' / 'best_eval.json'),
        metrics_mtime=(d / 'run' / 'metrics.jsonl').stat().st_mtime if (d / 'run' / 'metrics.jsonl').exists() else None,
    ))
print(json.dumps(dict(now=time.time(), jobs=out)))
PY
'''
    p = ssh_bash(host_cfg, script, timeout=timeout, connect_timeout=min(10, timeout))
    return json.loads(p.stdout.strip().splitlines()[-1])


def stop_job(host_cfg, job_name):
    """Gracefully stop a remote WSL2 job (SIGTERM process group and optional systemd user unit)."""
    job_dir = f'{host_cfg["runs_root"]}/{job_name}'
    unit = f'aio-{job_name}'.replace('/', '-')
    script = f'''set -euo pipefail
if command -v systemctl >/dev/null 2>&1; then
    systemctl --user stop {shlex.quote(unit)} 2>/dev/null || true
fi
if [ -f {shlex.quote(job_dir + "/pid")} ]; then
    PID=$(cat {shlex.quote(job_dir + "/pid")})
    if [ -n "$PID" ]; then
        kill -TERM -"$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true
        sleep 2
    fi
fi
echo STOPPED
'''
    p = ssh_bash(host_cfg, script, timeout=30)
    return dict(job=job_name, output=p.stdout.strip())


def sync_job(host_cfg, host_name, job_name, dest=None, with_last=False, with_eval_snaps=True):
    """Sync logs, metrics, eval sidecars, heartbeat, and optionally last.pt / eval snapshots to Mac."""
    dst = Path(dest).expanduser() if dest else (LOCAL_SYNC_ROOT / host_name / job_name)
    (dst / 'run' / 'eval').mkdir(parents=True, exist_ok=True)
    remote_job = f'{host_cfg["runs_root"]}/{job_name}'

    if host_cfg['native'] or not host_cfg.get('wsl_distro'):
        excludes = ['--exclude=run/row*.pt', '--exclude=*.tmp']
        if not with_last:
            excludes.append('--exclude=run/last.pt')
        if not with_eval_snaps:
            excludes.append('--exclude=run/eval/*.pt')
        subprocess.run(
            ['rsync', '-az', '--partial', '--inplace', *excludes,
             f'{host_cfg["ssh"]}:{remote_job}/', f'{dst}/'],
            check=True, capture_output=True, text=True, timeout=1800,
        )
        return dict(dest=str(dst), transport='rsync')

    # Windows-proxied WSL2: pull small files + optional eval/*.pt & last.pt
    small = ['job.json', 'heartbeat.json', 'job.log', 'exit',
             'run/run.json', 'run/metrics.jsonl', 'run/eval.jsonl', 'run/best_eval.json']
    pulled = []
    for rel in small:
        r_path = f'{remote_job}/{rel}'
        chk = ssh_bash(host_cfg, f'test -f {shlex.quote(r_path)} && echo OK || true\n', timeout=20)
        if 'OK' in chk.stdout:
            _transfer_file_from_remote(host_cfg, r_path, dst / rel, timeout=300)
            pulled.append(rel)
    if with_eval_snaps:
        ls = ssh_bash(host_cfg, f'ls -1 {shlex.quote(remote_job + "/run/eval")} 2>/dev/null || true\n', timeout=20)
        for name in ls.stdout.splitlines():
            name = name.strip()
            if name.endswith('.pt') and not name.endswith('.tmp'):
                _transfer_file_from_remote(host_cfg, f'{remote_job}/run/eval/{name}', dst / 'run' / 'eval' / name)
                pulled.append(f'run/eval/{name}')
    if with_last:
        _transfer_file_from_remote(host_cfg, f'{remote_job}/run/last.pt', dst / 'run' / 'last.pt')
        pulled.append('run/last.pt')
    return dict(dest=str(dst), transport='ssh-stream', pulled=pulled)


def build_setup_wsl_script(host_cfg, hostname=None):
    """Build safe Linux-only WSL2 setup script (systemd=true, ext4 caches, Tailscale SSH check).

    NEVER modifies Windows .venv, D:\\aiogym2-runs, Windows sshd.exe, or guomao rendering port 8765.
    """
    alias = hostname or host_cfg.get('native_alias') or f'{host_cfg["ssh"]}-wsl'
    return f'''set -euo pipefail
# 1. Fast Linux ext4 cache & run directories (never /mnt/c or /mnt/d)
mkdir -p "$HOME/.cache/aio/inductor" "$HOME/.cache/aio/triton" "$HOME/.cache/aio/val-cache" {shlex.quote(host_cfg["runs_root"])}
if [ -d "$HOME/aio-cache/inductor" ] && [ -z "$(ls -A "$HOME/.cache/aio/inductor" 2>/dev/null)" ]; then
    rmdir "$HOME/.cache/aio/inductor"
    ln -s "$HOME/aio-cache/inductor" "$HOME/.cache/aio/inductor"
fi

# 2. Idempotently enable systemd=true in /etc/wsl.conf (Linux-side only)
if [ ! -f /etc/wsl.conf ]; then
    printf '[boot]\\nsystemd=true\\n' > /etc/wsl.conf
elif ! grep -qE '^\\s*systemd\\s*=\\s*true' /etc/wsl.conf; then
    if grep -qE '^\\s*\\[boot\\]' /etc/wsl.conf; then
        sed -i '/^\\s*\\[boot\\]/a systemd=true' /etc/wsl.conf
    else
        printf '\\n[boot]\\nsystemd=true\\n' >> /etc/wsl.conf
    fi
fi

# 3. Report status (Tailscale SSH activation is printed cleanly for one-time approval)
echo "WSL_CONF:"
cat /etc/wsl.conf
echo "SYSTEMD: $(systemctl is-system-running 2>&1 || true)"
echo "DISK_ROOT:"
df -h /
if command -v tailscale >/dev/null 2>&1; then
    echo "TAILSCALE: installed ($(tailscale version | head -n1))"
    echo "NEXT_STEP: sudo tailscale up --ssh --hostname={alias}"
else
    echo "TAILSCALE: not installed yet"
    echo "INSTALL_CMD: curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up --ssh --hostname={alias}"
fi
'''


def setup_wsl(host_cfg, apply=False, hostname=None):
    script = build_setup_wsl_script(host_cfg, hostname=hostname)
    if not apply:
        return dict(applied=False, host=host_cfg['ssh'], script=script)
    p = ssh_bash(host_cfg, script, timeout=60)
    return dict(applied=True, host=host_cfg['ssh'], output=p.stdout)


def build_parser():
    ap = argparse.ArgumentParser(description='24x7 WSL2 local GPU runner & Colab <-> local handoff.')
    sub = ap.add_subparsers(dest='do', required=True)

    p_start = sub.add_parser('start', help='Start a detached WSL2 training job with thermal heartbeat')
    p_start.add_argument('host', help='guomao, baolong, guomao-wsl, baolong-wsl, or SSH host')
    p_start.add_argument('name', help='Job name under /root/aio-runs/<name>')
    p_start.add_argument('--ref', default=None, help='Optional git branch/commit to checkout before starting')
    p_start.add_argument('--put', action='append', default=[], help='Local file to place into {in}/NAME')
    p_start.add_argument('--systemd', action='store_true', help='Use systemd-run --user when available')
    p_start.add_argument('--thermal-warn', type=float, default=THERMAL_WARN_C)
    p_start.add_argument('--quiet-hours', default=None, help='Override quiet window HH:MM-HH:MM (default: host profile, guomao=00:00-10:30)')
    p_start.add_argument('--no-quiet-hours', action='store_true', help='Disable default host quiet-hours pause')
    p_start.add_argument('--native-ssh', action='store_true')
    p_start.add_argument('--wsl-distro', default=None)

    p_st = sub.add_parser('status', help='Show remote job liveness, GPU thermals, and latest rows')
    p_st.add_argument('host')
    p_st.add_argument('name', nargs='?', default=None)
    p_st.add_argument('--json', action='store_true', dest='as_json')
    p_st.add_argument('--native-ssh', action='store_true')
    p_st.add_argument('--wsl-distro', default=None)

    p_stop = sub.add_parser('stop', help='Gracefully stop a remote WSL2 job')
    p_stop.add_argument('host')
    p_stop.add_argument('name')
    p_stop.add_argument('--native-ssh', action='store_true')
    p_stop.add_argument('--wsl-distro', default=None)

    p_sync = sub.add_parser('sync', help='Pull metrics, eval sidecars, heartbeat, and optional checkpoints')
    p_sync.add_argument('host')
    p_sync.add_argument('name')
    p_sync.add_argument('--dest', type=Path, default=None)
    p_sync.add_argument('--with-last', action='store_true')
    p_sync.add_argument('--no-eval-snaps', action='store_true')
    p_sync.add_argument('--native-ssh', action='store_true')
    p_sync.add_argument('--wsl-distro', default=None)

    p_ho = sub.add_parser('handoff', help='Verify and transfer last.pt + run.json + metrics.jsonl across machines')
    p_ho.add_argument('--from', dest='src', required=True, help='colab:<run>, local:<host>:<run>, or path:<dir>')
    p_ho.add_argument('--to', dest='dst', required=True, help='colab:<run>, local:<host>:<run>, or path:<dir>')
    p_ho.add_argument('--total-updates', type=int, default=None)
    p_ho.add_argument('--lr', type=float, default=None)
    p_ho.add_argument('--accum', type=int, default=None)
    p_ho.add_argument('--verify-only', action='store_true')
    p_ho.add_argument('--dry-run', action='store_true')

    p_sw = sub.add_parser('setup-wsl', help='Print or apply safe Linux-only WSL2 systemd/cache/Tailscale SSH setup')
    p_sw.add_argument('host')
    p_sw.add_argument('--apply', action='store_true')
    p_sw.add_argument('--hostname', default=None)
    p_sw.add_argument('--native-ssh', action='store_true')
    p_sw.add_argument('--wsl-distro', default=None)

    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[argv.index('--') + 1:] if '--' in argv else []
    a = build_parser().parse_args(argv[:argv.index('--')] if '--' in argv else argv)

    if a.do == 'start':
        if not cmd:
            raise SystemExit('start requires a command after --')
        hcfg = resolve_host(a.host, wsl_distro=a.wsl_distro, native_ssh=a.native_ssh)
        qh = '' if a.no_quiet_hours else a.quiet_hours
        res = start_job(hcfg, a.host, a.name, cmd, ref=a.ref, put_files=a.put,
                        use_systemd=a.systemd, thermal_warn_c=a.thermal_warn, quiet_hours=qh)
        print(json.dumps(res, indent=1))
    elif a.do == 'status':
        hcfg = resolve_host(a.host, wsl_distro=a.wsl_distro, native_ssh=a.native_ssh)
        res = status_job(hcfg, a.name)
        if a.as_json:
            print(json.dumps(res, indent=1))
        else:
            for j in res['jobs']:
                hb = j.get('heartbeat') or {}
                gpu = hb.get('gpu') or {}
                last_m = next((r for r in reversed(j.get('metrics_tail') or []) if 'iter' in r), {})
                warn = ' [THERMAL WARN]' if hb.get('thermal_warning') else ''
                print(
                    f'{j["job"]}: running={j["running"]} pid={j["pid"]} iter={last_m.get("iter")} '
                    f'gpu={gpu.get("temp_c")}C/{gpu.get("power_w")}W/{gpu.get("util_pct")}% '
                    f'mem={gpu.get("mem_used_mb")}/{gpu.get("mem_total_mb")}MiB{warn}'
                )
    elif a.do == 'stop':
        hcfg = resolve_host(a.host, wsl_distro=a.wsl_distro, native_ssh=a.native_ssh)
        print(json.dumps(stop_job(hcfg, a.name), indent=1))
    elif a.do == 'sync':
        hcfg = resolve_host(a.host, wsl_distro=a.wsl_distro, native_ssh=a.native_ssh)
        print(json.dumps(sync_job(hcfg, a.host, a.name, dest=a.dest,
                                  with_last=a.with_last, with_eval_snaps=not a.no_eval_snaps), indent=1))
    elif a.do == 'handoff':
        res = handoff(a.src, a.dst, total_updates=a.total_updates, lr=a.lr, accum=a.accum,
                      verify_only=a.verify_only, dry_run=a.dry_run)
        print(json.dumps(res, indent=1))
        print('Resume flags:', ' '.join(res['resume_flags']))
    elif a.do == 'setup-wsl':
        hcfg = resolve_host(a.host, wsl_distro=a.wsl_distro, native_ssh=a.native_ssh)
        res = setup_wsl(hcfg, apply=a.apply, hostname=a.hostname)
        if res['applied']:
            print(res['output'])
        else:
            print('# Dry-run (pass --apply to execute safe Linux-side WSL2 setup):')
            print(res['script'])


if __name__ == '__main__':
    main()

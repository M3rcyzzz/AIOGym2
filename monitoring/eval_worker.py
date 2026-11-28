"""Zero-CU asynchronous validation and mechanistic-family exam daemon.

Watches `<run-dir>/run/eval/row*.pt` (and `outer_r*.pt` from `diloco_hub.py`), runs upgraded
`python -m aio.train --score ...` locally or on a remote WSL2 GPU (`guomao`/`baolong`), appends
structured results to sidecar `<run-dir>/run/eval.jsonl` (NEVER mutating `metrics.jsonl`), updates
`<run-dir>/run/best_eval.json` + atomically copies the best snapshot to `<run-dir>/run/eval/best_eval.pt`,
and optionally triggers `aio.sim.famexam` on milestones.

  python -m monitoring.eval_worker --run-dir ~/colab-runs/sq-pure-11 \\
      --eval-host guomao-wsl --val-cache ~/.cache/aio/val-cache \\
      --val-n 256 --val-sizes 24x10@64 48x20@32 --famexam-every 50
"""
import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_FAMEXAM_PLANTS = 'qtank,ph,vdv,excstr,bio,evap,furnace,column'
SNAP_PATTERNS = (
    ('row', re.compile(r'^row(\d+)\.pt$')),
    ('round', re.compile(r'^outer_r(\d+)\.pt$')),
)
REMOTE_PROFILES = {
    'guomao': dict(ssh='guomao', wsl_distro='Ubuntu-24.04', wsl_user='root', native=False,
                   repo='/root/AIOGym2', python='/root/AIOGym2/.venv/bin/python',
                   val_cache='/root/.cache/aio/val-cache', stage='/root/.cache/aio/eval-stage',
                   quiet_hours='00:00-10:30'),
    'guomao-wsl': dict(ssh='guomao-wsl', wsl_distro=None, wsl_user=None, native=True,
                       repo='/root/AIOGym2', python='/root/AIOGym2/.venv/bin/python',
                       val_cache='/root/.cache/aio/val-cache', stage='/root/.cache/aio/eval-stage',
                       quiet_hours='00:00-10:30'),
    'baolong': dict(ssh='baolong', wsl_distro='Ubuntu', wsl_user='root', native=False,
                    repo='/root/AIOGym2', python='/root/AIOGym2/.venv/bin/python',
                    val_cache='/root/.cache/aio/val-cache', stage='/root/.cache/aio/eval-stage',
                    quiet_hours=None),
    'baolong-wsl': dict(ssh='baolong-wsl', wsl_distro=None, wsl_user=None, native=True,
                        repo='/root/AIOGym2', python='/root/AIOGym2/.venv/bin/python',
                        val_cache='/root/.cache/aio/val-cache', stage='/root/.cache/aio/eval-stage',
                        quiet_hours=None),
}


def in_quiet_hours(spec, tz_name='Asia/Shanghai', now=None):
    """Return True if current time in `tz_name` falls inside `HH:MM-HH:MM` quiet window."""
    if not spec or '-' not in str(spec):
        return False
    from datetime import datetime, timedelta, timezone
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone(timedelta(hours=8)) if 'Shanghai' in str(tz_name) else timezone.utc
    s_part, e_part = str(spec).strip().split('-', 1)
    sh, sm = (int(x) for x in s_part.strip().split(':'))
    eh, em = (int(x) for x in e_part.strip().split(':'))
    s_min, e_min = (sh % 24) * 60 + sm, (eh % 24) * 60 + em
    if s_min == e_min:
        return False
    dt = datetime.fromtimestamp(time.time() if now is None else float(now), tz=tz)
    cur = dt.hour * 60 + dt.minute + dt.second / 60.0
    if s_min < e_min:
        return s_min <= cur < e_min
    return cur >= s_min or cur < e_min


def pick_eval_host(eval_host, fallback_host=None, tz_name='Asia/Shanghai', now=None):
    """Avoid waking quiet-hours hosts (e.g. guomao 00:00-10:30 CST) by routing evals to fallback host."""
    prof = REMOTE_PROFILES.get(eval_host)
    if prof and in_quiet_hours(prof.get('quiet_hours'), tz_name=tz_name, now=now):
        return fallback_host or ('baolong' if eval_host.startswith('guomao') else 'local')
    return eval_host


def sanitize_finite(obj):
    """Replace non-finite floats with None recursively so strict JSON output never fails."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_finite(v) for v in obj]
    return obj


def resolve_run_root(run_dir):
    """Resolve the active run directory (`<dir>/run` for Colab job dirs, or `<dir>` if already a run/hub dir)."""
    p = Path(run_dir).expanduser()
    if (p / 'run').is_dir() or (p / 'job.json').exists():
        return p / 'run'
    if (p / 'eval').is_dir() or (p / 'eval.jsonl').exists() or (p / 'hub_last.pt').exists():
        return p
    return p / 'run'


def parse_snap_info(name):
    """Parse snapshot filename -> (kind, step_index) or None for non-eval files."""
    if not name or name.startswith('.') or name == 'best_eval.pt' or name.endswith('.tmp'):
        return None
    for kind, rx in SNAP_PATTERNS:
        m = rx.match(name)
        if m:
            return kind, int(m.group(1))
    return None


def is_snapshot_ready(path, min_age=0.0, now=None):
    """Verify snapshot is non-empty, settled, and a valid zip archive (.pt)."""
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return False
    if st.st_size <= 0:
        return False
    if min_age > 0:
        t = time.time() if now is None else now
        if t - st.st_mtime < min_age:
            return False
    try:
        return zipfile.is_zipfile(p)
    except OSError:
        return False


def snap_signature(path):
    st = Path(path).stat()
    return [int(st.st_size), round(st.st_mtime, 3)]


def load_eval_state(run_root):
    """Load already-evaluated snapshot signatures, best record, and last famexam points from sidecar files."""
    run_root = Path(run_root)
    seen = {}
    best = None
    last_famexam_pts = None

    eval_log = run_root / 'eval.jsonl'
    if eval_log.exists():
        for line in eval_log.read_text(encoding='utf-8', errors='replace').splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict):
                continue
            snap = rec.get('snap')
            sig = rec.get('snap_sig')
            if snap and sig is not None:
                seen[snap] = sig
            pts = rec.get('pts')
            if isinstance(pts, (int, float)) and math.isfinite(pts):
                if best is None or pts > best['pts']:
                    best = dict(pts=float(pts), snap=snap, iter=rec.get('iter'), round=rec.get('round'),
                                updates=rec.get('updates'), samples=rec.get('samples'), hours=rec.get('hours'),
                                val=rec.get('val'), val_sizes=rec.get('val_sizes'), famexam=rec.get('famexam'))
                if rec.get('famexam'):
                    last_famexam_pts = float(pts)

    best_file = run_root / 'best_eval.json'
    if best_file.exists():
        try:
            saved = json.loads(best_file.read_text(encoding='utf-8'))
            pts = saved.get('pts')
            if isinstance(pts, (int, float)) and math.isfinite(pts):
                if best is None or pts > best['pts']:
                    best = saved
        except (OSError, ValueError):
            pass

    return seen, best, last_famexam_pts


def list_pending_snaps(run_root, seen_sigs, min_age=0.0):
    """Return sorted list of newly ready eval snapshots in `<run_root>/eval/`."""
    eval_dir = Path(run_root) / 'eval'
    if not eval_dir.is_dir():
        return []
    pending = []
    for p in eval_dir.iterdir():
        if not p.is_file():
            continue
        info = parse_snap_info(p.name)
        if info is None:
            continue
        if not is_snapshot_ready(p, min_age=min_age):
            continue
        sig = snap_signature(p)
        if seen_sigs.get(p.name) == sig:
            continue
        kind, idx = info
        pending.append(((0 if kind == 'row' else 1, idx, p.name), p))
    pending.sort(key=lambda item: item[0])
    return [p for _, p in pending]


def extract_primary_pts(score_out):
    """Extract primary 12x5 validation points from upgraded or legacy `aio.train --score` JSON."""
    if not isinstance(score_out, dict):
        return None
    val = score_out.get('val')
    if isinstance(val, dict) and isinstance(val.get('pts'), (int, float)) and math.isfinite(val['pts']):
        return float(val['pts'])
    all_block = score_out.get('all')
    if isinstance(all_block, dict) and isinstance(all_block.get('model'), (int, float)) and math.isfinite(all_block['model']):
        return float(all_block['model'])
    pts = score_out.get('pts')
    if isinstance(pts, (int, float)) and math.isfinite(pts):
        return float(pts)
    return None


def parse_last_json(stdout):
    """Parse the last valid JSON dict line emitted to stdout."""
    for line in reversed((stdout or '').splitlines()):
        s = line.strip()
        if not s or not s.startswith('{'):
            continue
        try:
            obj = json.loads(s)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    raise RuntimeError(f'no JSON object found in output: {(stdout or "")[-600:]}')


def build_score_cmd(python_bin, snap_path, val_cache=None, val_n=256, val_seed=1_000_000,
                    val_sizes=None, val_raw=True, meter=True, device=None):
    cmd = [str(python_bin), '-m', 'aio.train', '--score', str(snap_path),
           '--val-n', str(int(val_n)), '--val-seed', str(int(val_seed))]
    if val_cache:
        cmd += ['--val-cache', str(val_cache)]
    if val_sizes:
        cmd += ['--val-sizes', *[str(s) for s in val_sizes]]
    if val_raw:
        cmd.append('--val-raw')
    if meter:
        cmd.append('--meter')
    if device:
        cmd += ['--device', str(device)]
    return cmd


def build_famexam_cmd(python_bin, snap_path, out_json, plants=DEFAULT_FAMEXAM_PLANTS,
                      n=8, seed=3_000_000, device='cpu'):
    return [str(python_bin), '-m', 'aio.sim.famexam',
            '--plants', str(plants),
            '--ctl', 'aio.plugin:policy',
            '--arg', str(snap_path),
            '--n', str(int(n)),
            '--seed', str(int(seed)),
            '--out', str(out_json),
            '--device', str(device)]


def resolve_remote(eval_host, native_ssh=False, wsl_distro=None, remote_repo=None,
                   remote_python=None, val_cache=None):
    base = dict(REMOTE_PROFILES.get(eval_host, dict(
        ssh=eval_host, wsl_distro=None, wsl_user=None,
        native=native_ssh or eval_host.endswith('-wsl') or not wsl_distro,
        repo='/root/AIOGym2', python='/root/AIOGym2/.venv/bin/python',
        val_cache='/root/.cache/aio/val-cache', stage='/root/.cache/aio/eval-stage',
    )))
    if native_ssh:
        base['native'] = True
        base['wsl_distro'] = None
    if wsl_distro:
        base['wsl_distro'] = wsl_distro
        base['native'] = False
    if remote_repo:
        base['repo'] = remote_repo
    if remote_python:
        base['python'] = remote_python
    if val_cache:
        base['val_cache'] = str(val_cache)
    return base


def _ssh_exec(rcfg, bash_script, timeout=1800, stdin_bytes=None):
    ssh_cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', rcfg['ssh']]
    if rcfg['native'] or not rcfg.get('wsl_distro'):
        if stdin_bytes is not None:
            cmd = ssh_cmd + ['bash', '-lc', bash_script]
            return subprocess.run(cmd, input=stdin_bytes, capture_output=True, timeout=timeout, check=True)
        cmd = ssh_cmd + ['bash', '-s']
        return subprocess.run(cmd, input=bash_script, text=True, capture_output=True, timeout=timeout, check=True)
    user = rcfg.get('wsl_user') or 'root'
    wsl_prefix = f'wsl -d {rcfg["wsl_distro"]} -u {user} --'
    if stdin_bytes is not None:
        cmd = ssh_cmd + [f'{wsl_prefix} bash -lc {shlex.quote(bash_script)}']
        return subprocess.run(cmd, input=stdin_bytes, capture_output=True, timeout=timeout, check=True)
    cmd = ssh_cmd + [f'{wsl_prefix} bash -s']
    return subprocess.run(cmd, input=bash_script, text=True, capture_output=True, timeout=timeout, check=True)


def _stage_remote_snap(rcfg, local_snap):
    remote_dir = rcfg['stage']
    remote_path = f'{remote_dir}/{local_snap.name}'
    if rcfg['native'] or not rcfg.get('wsl_distro'):
        _ssh_exec(rcfg, f'mkdir -p {shlex.quote(remote_dir)} {shlex.quote(rcfg["val_cache"])}\n', timeout=60)
        subprocess.run(
            ['rsync', '-az', '--partial', '--inplace', str(local_snap), f'{rcfg["ssh"]}:{remote_path}'],
            check=True, capture_output=True, text=True, timeout=600,
        )
        return remote_path
    # Windows-proxied WSL2: stream atomically over SSH stdin into WSL ext4
    tmp = f'{remote_path}.tmp'
    script = (
        f'mkdir -p {shlex.quote(remote_dir)} {shlex.quote(rcfg["val_cache"])} && '
        f'cat > {shlex.quote(tmp)} && mv {shlex.quote(tmp)} {shlex.quote(remote_path)}'
    )
    _ssh_exec(rcfg, script, timeout=600, stdin_bytes=local_snap.read_bytes())
    return remote_path


def run_score(snap_path, eval_host='local', python_bin=None, val_cache=None, val_n=256,
              val_seed=1_000_000, val_sizes=None, val_raw=True, meter=True, device=None,
              native_ssh=False, wsl_distro=None, remote_repo=None, remote_python=None, timeout=1800):
    """Run `aio.train --score` locally or on a remote WSL2 GPU and return parsed JSON dict."""
    snap_path = Path(snap_path)
    if eval_host == 'local':
        py = python_bin or sys.executable
        if val_cache and str(val_cache).startswith('/root/') and not os.access('/root', os.W_OK):
            val_cache = Path.home() / '.cache' / 'aio' / 'val-cache'
        cmd = build_score_cmd(py, snap_path, val_cache=val_cache, val_n=val_n, val_seed=val_seed,
                              val_sizes=val_sizes, val_raw=val_raw, meter=meter, device=device)
        env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONUNBUFFERED='1')
        p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            raise RuntimeError(f'score failed ({p.returncode}): {(p.stdout + p.stderr)[-1200:]}')
        return parse_last_json(p.stdout)

    rcfg = resolve_remote(eval_host, native_ssh=native_ssh, wsl_distro=wsl_distro,
                          remote_repo=remote_repo, remote_python=remote_python, val_cache=val_cache)
    remote_snap = _stage_remote_snap(rcfg, snap_path)
    dev = device or 'cuda'
    cmd = build_score_cmd(rcfg['python'], remote_snap, val_cache=rcfg['val_cache'], val_n=val_n,
                          val_seed=val_seed, val_sizes=val_sizes, val_raw=val_raw, meter=meter, device=dev)
    script = (
        'set -euo pipefail\n'
        f'cd {shlex.quote(rcfg["repo"])}\n'
        'export AIO_SEQ_AMP=1 AIO_SEQ_ROOMS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1\n'
        f'export PYTHONPATH={shlex.quote(rcfg["repo"])}\n'
        f'{" ".join(shlex.quote(x) for x in cmd)}\n'
    )
    p = _ssh_exec(rcfg, script, timeout=timeout)
    return parse_last_json(p.stdout)


def summarize_famexam(data):
    """Extract per-family model points and macro mean from `aio.sim.famexam` JSON output."""
    if not isinstance(data, dict):
        return None
    per_fam = {}
    for fam, body in data.items():
        if not isinstance(body, dict):
            continue
        pts_map = body.get('points')
        if isinstance(pts_map, dict):
            val = pts_map.get('aio')
            if isinstance(val, (int, float)) and math.isfinite(val):
                per_fam[fam] = round(float(val), 2)
    if not per_fam:
        return None
    mean_pts = round(sum(per_fam.values()) / len(per_fam), 2)
    return dict(mean=mean_pts, families=per_fam)


def run_famexam(snap_path, out_json, eval_host='local', python_bin=None,
                plants=DEFAULT_FAMEXAM_PLANTS, n=8, seed=3_000_000, device='cpu',
                native_ssh=False, wsl_distro=None, remote_repo=None, remote_python=None, timeout=1800):
    """Run `aio.sim.famexam` on a milestone snapshot and return summary dict."""
    snap_path, out_json = Path(snap_path), Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if eval_host == 'local':
        py = python_bin or sys.executable
        cmd = build_famexam_cmd(py, snap_path, out_json, plants=plants, n=n, seed=seed, device=device)
        env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONUNBUFFERED='1')
        p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            raise RuntimeError(f'famexam failed ({p.returncode}): {(p.stdout + p.stderr)[-1200:]}')
        data = json.loads(out_json.read_text(encoding='utf-8'))
        return summarize_famexam(data)

    rcfg = resolve_remote(eval_host, native_ssh=native_ssh, wsl_distro=wsl_distro,
                          remote_repo=remote_repo, remote_python=remote_python)
    remote_snap = _stage_remote_snap(rcfg, snap_path)
    remote_out = f'{rcfg["stage"]}/{out_json.name}'
    cmd = build_famexam_cmd(rcfg['python'], remote_snap, remote_out, plants=plants, n=n, seed=seed, device=device)
    script = (
        'set -euo pipefail\n'
        f'cd {shlex.quote(rcfg["repo"])}\n'
        f'export PYTHONPATH={shlex.quote(rcfg["repo"])} PYTHONUNBUFFERED=1\n'
        f'rm -f {shlex.quote(remote_out)}\n'
        f'{" ".join(shlex.quote(x) for x in cmd)} >&2\n'
        f'cat {shlex.quote(remote_out)}\n'
    )
    p = _ssh_exec(rcfg, script, timeout=timeout)
    data = json.loads(p.stdout)
    atomic_write_json(out_json, data)
    return summarize_famexam(data)


def atomic_copy(src, dst):
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f'.{dst.name}.tmp.{os.getpid()}'
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def atomic_write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    clean = sanitize_finite(data)
    tmp = p.parent / f'.{p.name}.tmp.{os.getpid()}'
    tmp.write_text(json.dumps(clean, ensure_ascii=False, allow_nan=False, indent=1) + '\n', encoding='utf-8')
    os.replace(tmp, p)


def append_jsonl(path, row):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    clean = sanitize_finite(row)
    line = json.dumps(clean, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n'
    with open(p, 'a', encoding='utf-8') as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def should_trigger_famexam(row_iter, pts, is_new_best, famexam_every, famexam_delta, last_famexam_pts):
    if famexam_every <= 0 and famexam_delta <= 0:
        return False
    if famexam_every > 0 and row_iter is not None and row_iter > 0 and (row_iter % famexam_every == 0):
        return True
    if is_new_best and famexam_delta > 0 and pts is not None:
        if last_famexam_pts is None or (pts - last_famexam_pts) >= famexam_delta:
            return True
    return False


def evaluate_snapshot(run_root, snap_path, best_state, last_famexam_pts, *,
                      eval_host='local', fallback_eval_host=None, quiet_tz='Asia/Shanghai', quiet_now=None,
                      python_bin=None, val_cache=None, val_n=256,
                      val_seed=1_000_000, val_sizes=None, val_raw=True, meter=True,
                      device=None, famexam_every=50, famexam_delta=0.5,
                      famexam_plants=DEFAULT_FAMEXAM_PLANTS, famexam_n=8, famexam_device='cpu',
                      native_ssh=False, wsl_distro=None, remote_repo=None, remote_python=None,
                      score_fn=None, famexam_fn=None):
    """Score one snapshot, append sidecar eval.jsonl, update best_eval.json/best_eval.pt, and run milestone famexam."""
    run_root, snap_path = Path(run_root), Path(snap_path)
    eff_host = pick_eval_host(eval_host, fallback_host=fallback_eval_host, tz_name=quiet_tz, now=quiet_now)
    if eff_host != eval_host:
        print(f'eval_worker: quiet hours on {eval_host}, routing {snap_path.name} to {eff_host}', file=sys.stderr, flush=True)
    info = parse_snap_info(snap_path.name)
    kind, idx = info if info else ('row', None)
    sig = snap_signature(snap_path)
    t0 = time.time()

    if score_fn is not None:
        score_out = score_fn(snap_path)
    else:
        score_out = run_score(
            snap_path, eval_host=eff_host, python_bin=python_bin, val_cache=val_cache,
            val_n=val_n, val_seed=val_seed, val_sizes=val_sizes, val_raw=val_raw,
            meter=meter, device=device, native_ssh=native_ssh, wsl_distro=wsl_distro,
            remote_repo=remote_repo, remote_python=remote_python,
        )
    elapsed = time.time() - t0
    pts = extract_primary_pts(score_out)
    row_iter = score_out.get('iter')
    if row_iter is None:
        row_iter = idx
    outer_round = score_out.get('round') if score_out.get('round') is not None else (idx if kind == 'round' else None)

    is_new_best = pts is not None and (best_state is None or pts > best_state['pts'])
    fam_summary = None
    if should_trigger_famexam(row_iter, pts, is_new_best, famexam_every, famexam_delta, last_famexam_pts):
        fam_out = run_root / 'eval' / f'famexam_{snap_path.stem}.json'
        if famexam_fn is not None:
            fam_summary = famexam_fn(snap_path, fam_out)
        else:
            fam_summary = run_famexam(
                snap_path, fam_out, eval_host=eff_host, python_bin=python_bin,
                plants=famexam_plants, n=famexam_n, device=famexam_device,
                native_ssh=native_ssh, wsl_distro=wsl_distro,
                remote_repo=remote_repo, remote_python=remote_python,
            )
        if fam_summary is not None and pts is not None:
            last_famexam_pts = pts

    val_block = score_out.get('val')
    if val_block is None and pts is not None:
        val_block = dict(pts=round(pts, 4))

    entry = dict(
        snap=snap_path.name,
        snap_sig=sig,
        iter=row_iter,
        round=outer_round,
        updates=score_out.get('updates'),
        samples=score_out.get('samples'),
        hours=score_out.get('hours'),
        pts=round(pts, 4) if pts is not None else None,
        val=val_block,
        raw=score_out.get('raw'),
        val_sizes=score_out.get('val_sizes'),
        famexam=fam_summary,
        eval_host=eff_host,
        eval_seconds=round(elapsed, 2),
        evaluated_at=round(time.time(), 3),
    )
    entry = {k: v for k, v in entry.items() if v is not None}

    # Append ONLY to sidecar eval.jsonl — NEVER touches metrics.jsonl!
    append_jsonl(run_root / 'eval.jsonl', entry)

    if is_new_best:
        best_pt = run_root / 'eval' / 'best_eval.pt'
        atomic_copy(snap_path, best_pt)
        best_state = dict(
            pts=round(pts, 4),
            snap=snap_path.name,
            best_pt=str(best_pt.name),
            iter=row_iter,
            round=outer_round,
            updates=score_out.get('updates'),
            samples=score_out.get('samples'),
            hours=score_out.get('hours'),
            val=val_block,
            raw=score_out.get('raw'),
            val_sizes=score_out.get('val_sizes'),
            famexam=fam_summary,
            eval_host=eff_host,
            updated_at=round(time.time(), 3),
        )
        best_state = {k: v for k, v in best_state.items() if v is not None}
        atomic_write_json(run_root / 'best_eval.json', best_state)

    return entry, best_state, last_famexam_pts


def process_once(run_root, **kwargs):
    """Evaluate all pending snapshots once and return the list of appended entries."""
    seen, best_state, last_famexam_pts = load_eval_state(run_root)
    min_age = kwargs.pop('min_age', 0.0)
    snaps = list_pending_snaps(run_root, seen, min_age=min_age)
    results = []
    for snap in snaps:
        entry, best_state, last_famexam_pts = evaluate_snapshot(
            run_root, snap, best_state, last_famexam_pts, **kwargs,
        )
        seen[snap.name] = entry['snap_sig']
        results.append(entry)
    return results


def build_parser():
    ap = argparse.ArgumentParser(description='Zero-CU async validation & exam daemon (watches eval/row*.pt).')
    ap.add_argument('--run-dir', type=Path, required=True, help='Job or run directory containing eval/row*.pt')
    ap.add_argument('--eval-host', default='local', help='local, guomao, baolong, guomao-wsl, baolong-wsl, or SSH host')
    ap.add_argument('--fallback-eval-host', default=None, help='Fallback host used during eval-host quiet hours (default: baolong for guomao, else local)')
    ap.add_argument('--quiet-tz', default='Asia/Shanghai', help='Timezone for quiet-hours check (default: Asia/Shanghai)')
    ap.add_argument('--val-cache', type=Path, default=None, help='Persistent directory for cached validation references')
    ap.add_argument('--val-n', type=int, default=256)
    ap.add_argument('--val-seed', type=int, default=1_000_000)
    ap.add_argument('--val-sizes', nargs='*', default=['24x10@64', '48x20@32'])
    ap.add_argument('--val-raw', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--meter', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--device', default=None, help='Device for --score (default: auto/cuda on remote)')
    ap.add_argument('--famexam-every', type=int, default=50, help='Run famexam every N rows/rounds (0: disable cadence)')
    ap.add_argument('--famexam-delta', type=float, default=0.5, help='Also run famexam on new best improving by >= delta pts (0: off)')
    ap.add_argument('--famexam-plants', default=DEFAULT_FAMEXAM_PLANTS)
    ap.add_argument('--famexam-n', type=int, default=8)
    ap.add_argument('--famexam-device', default='cpu')
    ap.add_argument('--poll', type=float, default=30.0, help='Seconds between directory scans when watching')
    ap.add_argument('--min-age', type=float, default=2.0, help='Minimum file age (s) before reading a snapshot')
    ap.add_argument('--python', dest='python_bin', default=None, help='Local Python interpreter for --score')
    ap.add_argument('--native-ssh', action='store_true', help='Use direct Linux SSH + rsync for --eval-host')
    ap.add_argument('--wsl-distro', default=None, help='Explicit WSL2 distro name when proxying through Windows SSH')
    ap.add_argument('--remote-repo', default=None)
    ap.add_argument('--remote-python', default=None)
    ap.add_argument('--once', action='store_true', help='Process currently available snapshots once and exit')
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    run_root = resolve_run_root(a.run_dir)
    val_cache = a.val_cache
    if val_cache is None and a.eval_host == 'local':
        val_cache = Path.home() / '.cache' / 'aio' / 'val-cache'
    kw = dict(
        eval_host=a.eval_host, fallback_eval_host=a.fallback_eval_host, quiet_tz=a.quiet_tz,
        python_bin=a.python_bin, val_cache=val_cache,
        val_n=a.val_n, val_seed=a.val_seed, val_sizes=a.val_sizes,
        val_raw=a.val_raw, meter=a.meter, device=a.device,
        famexam_every=a.famexam_every, famexam_delta=a.famexam_delta,
        famexam_plants=a.famexam_plants, famexam_n=a.famexam_n, famexam_device=a.famexam_device,
        native_ssh=a.native_ssh, wsl_distro=a.wsl_distro,
        remote_repo=a.remote_repo, remote_python=a.remote_python,
        min_age=0.0 if a.once else a.min_age,
    )
    while True:
        try:
            done = process_once(run_root, **kw)
            for entry in done:
                print(json.dumps(entry, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(f'eval_worker: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
            if a.once:
                raise
        if a.once:
            return
        time.sleep(a.poll)


if __name__ == '__main__':
    main()

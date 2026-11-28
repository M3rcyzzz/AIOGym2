"""The training runs' progress for the research site's progress bars: rows done of the planned rows, time per row, when
each run should end. Standard library only; runs on the Mac next to the Colab watchers (docs/colab.md).

  python3 -m monitoring.train_status                                   # print train-status.json
  python3 -m monitoring.train_status --workers guomao,baolong          # include local WSL2 GPU workers (5 s timeout)
  python3 -m monitoring.train_status --push HOST --every 600           # keep the live site's copy current

Reads every job in ~/colab-runs (AIO_COLAB_ROOT): job.json (the command, or the pack's commands, and the starts), each
run's metrics.jsonl (one line per row: iter, t_row, hours), optional sidecar eval.jsonl / best_eval.json, and the job's
end. When --workers is given, polls each remote WSL2 host with a 5 s timeout and gracefully falls back to cached stale
entries if a host is temporarily unreachable. --push writes the file into the live release on the server (HOST:DEST,
a temporary name, then a rename), so the page shows it without a new release; a deploy carries the live copy into the
new release.
"""
import argparse, json, math, os, re, shlex, subprocess, sys, time
from pathlib import Path

ROOT = Path(os.environ.get('AIO_COLAB_ROOT', Path.home()/'colab-runs'))
DEST = '/home/ubuntu/decks/aio/train-status.json'
TRAIN_HOURS = 2.     # aio.train's default --hours
WORKER_TIMEOUT = 5.0
WORKER_WSL = {
    'guomao': ('Ubuntu-24.04', 'root'),
    'baolong': ('Ubuntu', 'root'),
}
WORKER_RUNS_ROOT = '/root/aio-runs'


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def words(cmd):
    """cmd's words, a `bash -c "..."` script split too (a resumed run's command is one)."""
    out = []
    for x in cmd:
        out += x.split() if ' ' in x else [x]
    return out


def flag(cmd, name, kind=float):
    for i, x in enumerate(cmd):
        if x == name and i+1 < len(cmd):
            return kind(cmd[i+1])
        if x.startswith(name+'='):
            return kind(x.split('=', 1)[1])
    return None


def tail(path, n=6):
    """The last n rows of a metrics.jsonl (a half-written last line is skipped)."""
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell()-64*1024))
            lines = f.read().decode('utf-8', 'replace').splitlines()
    except OSError:
        return []
    rows = []
    for x in lines[-n-1:]:
        try:
            r = json.loads(x)
        except ValueError:
            continue
        if isinstance(r, dict) and 'iter' in r and 'final' not in r:
            rows.append(r)
    return rows[-n:]


def _extract_eval_pts(rec):
    if not isinstance(rec, dict):
        return None
    pts = rec.get('pts')
    if _finite(pts):
        return float(pts)
    val = rec.get('val')
    if isinstance(val, dict) and _finite(val.get('pts')):
        return float(val['pts'])
    return None


def summarize_eval_records(rows, best_doc=None):
    """Summarize parsed eval.jsonl rows and optional best_eval.json into a compact finite dict."""
    latest = None
    best_pts = None
    best_iter = None
    for r in rows or ():
        pts = _extract_eval_pts(r)
        if pts is None:
            continue
        latest = r
        if best_pts is None or pts > best_pts:
            best_pts = pts
            best_iter = r.get('iter')
    if isinstance(best_doc, dict):
        bpts = _extract_eval_pts(best_doc)
        if bpts is not None and (best_pts is None or bpts > best_pts):
            best_pts = bpts
            best_iter = best_doc.get('iter')
            if latest is None:
                latest = best_doc
    if latest is None and best_pts is None:
        return None
    out = {}
    if latest is not None:
        lpts = _extract_eval_pts(latest)
        if lpts is not None:
            out['pts'] = round(lpts, 2)
        if latest.get('iter') is not None:
            out['iter'] = int(latest['iter'])
        if _finite(latest.get('evaluated_at')):
            out['at'] = round(float(latest['evaluated_at']))
        vs = latest.get('val_sizes')
        if isinstance(vs, dict):
            sz = {}
            for k, v in vs.items():
                p = v.get('pts') if isinstance(v, dict) else v
                if _finite(p):
                    sz[str(k)] = round(float(p), 2)
            if sz:
                out['sizes'] = sz
        fam = latest.get('famexam')
        if isinstance(fam, dict) and _finite(fam.get('mean')):
            out['famexam_mean'] = round(float(fam['mean']), 2)
    if best_pts is not None:
        out['best_pts'] = round(best_pts, 2)
        if best_iter is not None:
            out['best_iter'] = int(best_iter)
    return out or None


def eval_summary(run_dir):
    """Read sidecar eval.jsonl and best_eval.json from a run directory without touching metrics.jsonl."""
    d = Path(run_dir)
    rows = []
    eval_log = d/'eval.jsonl'
    if eval_log.exists():
        try:
            for line in eval_log.read_text(encoding='utf-8', errors='replace').splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict):
                    rows.append(r)
        except OSError:
            pass
    best_doc = None
    best_file = d/'best_eval.json'
    if best_file.exists():
        try:
            best_doc = json.loads(best_file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            pass
    return summarize_eval_records(rows, best_doc)


def exits(job):
    """Each command's exit code from the job's job.log ('[name] exit N'; 'exit N' for a single command)."""
    out = {}
    try:
        lines = (ROOT/job/'job.log').read_text(errors='replace').splitlines()
    except OSError:
        return out
    for x in lines:
        m = re.match(r'\S+ (?:\[([^\]]+)\] )?exit (-?\d+)$', x)
        if m:
            out[m.group(1) or ''] = int(m.group(2))
    return out


def run_status(job, sub, now):
    name, cmd = sub.get('name') or job['name'], words(sub['cmd'])
    d = ROOT/job['name']/'run'/(sub.get('name') or '')
    rows = tail(d/'metrics.jsonl')
    last = rows[-1] if rows else {}
    total = flag(cmd, '--rows', int)
    cap_h = flag(cmd, '--hours') if 'aio.train' in cmd else None
    if cap_h is None and 'aio.train' in cmd:
        cap_h = TRAIN_HOURS
    if total is not None and total >= 10**8:
        total = None
    t = [r['t_row'] for r in rows if _finite(r.get('t_row'))]
    t_row = sum(t)/len(t) if t else None
    done_rows = int(last.get('iter', 0))
    hours = last.get('hours')
    seen = (d/'metrics.jsonl').stat().st_mtime if (d/'metrics.jsonl').exists() else None
    end = job.get('done')
    code = exits(job['name']).get(sub.get('name') or '', end.get('code') if end else None)
    if total and done_rows >= total:
        state = 'done'
    elif code is not None or end:
        state = 'done' if code == 0 else 'stopped'
    elif seen and now-seen > max(4*(t_row or 0), 1800)+900:
        state = 'quiet'                                           # no new row for long: the VM, the Mac or the sync
    else:
        state = 'running'
    eta = None
    if state == 'running' and t_row and seen:
        left = (total - done_rows) if total else None
        if cap_h is not None and _finite(hours):
            by_time = max(0., cap_h - hours)*3600/t_row
            left = by_time if left is None else min(left, by_time)
        if left is not None:
            eta = round(seen + left*t_row)
    ev = eval_summary(d)
    out = dict(name=name, job=job['name'], state=state, rows=done_rows, total=total, cap_hours=cap_h,
               hours=hours, t_row=round(t_row, 1) if t_row else None, eta=eta, row_at=round(seen) if seen else None,
               started=round(job['starts'][0]['time']) if job.get('starts') else None,
               ended=round(end['time']) if end and end.get('time') else None,
               eval=ev)
    return {k: v for k, v in out.items() if v is not None and not (isinstance(v, float) and not math.isfinite(v))}


def _worker_probe_script(runs_root=WORKER_RUNS_ROOT):
    return f'''python3 - <<'PY'
import json, os, time
from pathlib import Path

root = Path({runs_root!r})
jobs = []
if root.exists():
    for p in sorted(root.glob('*/job.json')):
        d = p.parent
        def rj(path):
            try:
                return json.loads(path.read_text())
            except Exception:
                return None
        def tjl(path, n=6, require_iter=False):
            try:
                with open(path, 'rb') as f:
                    f.seek(0, 2)
                    f.seek(max(0, f.tell() - 64 * 1024))
                    lines = f.read().decode('utf-8', 'replace').splitlines()
            except OSError:
                return []
            out = []
            for x in lines:
                try:
                    r = json.loads(x)
                except ValueError:
                    continue
                if isinstance(r, dict) and (not require_iter or ('iter' in r and 'final' not in r)):
                    out.append(r)
            return out[-n:]
        pid_s = (d/'pid').read_text().strip() if (d/'pid').exists() else ''
        alive = False
        if pid_s.isdigit():
            try:
                st = Path(f'/proc/{{pid_s}}/stat').read_text()
                alive = st.rsplit(')', 1)[1].split()[0] not in 'ZX'
            except OSError:
                pass
        mp = d/'run'/'metrics.jsonl'
        jobs.append(dict(
            job=d.name,
            spec=rj(p),
            running=alive,
            exit=rj(d/'exit'),
            heartbeat=rj(d/'heartbeat.json'),
            metrics=tjl(mp, 6, require_iter=True),
            metrics_mtime=mp.stat().st_mtime if mp.exists() else None,
            eval_rows=tjl(d/'run'/'eval.jsonl', 64, require_iter=False),
            best_eval=rj(d/'run'/'best_eval.json'),
        ))
print(json.dumps(dict(now=time.time(), jobs=jobs)))
PY
'''


def _worker_run_status(host, rec, keep_h, now):
    spec = rec.get('spec') or {}
    cmd = words(spec.get('cmd') or [])
    if cmd and 'aio.train' not in cmd:
        return None
    end = rec.get('exit')
    if end and _finite(end.get('time')) and now - end['time'] > keep_h * 3600:
        return None
    rows = rec.get('metrics') or []
    last = rows[-1] if rows else {}
    total = flag(cmd, '--rows', int) if cmd else None
    cap_h = flag(cmd, '--hours') if 'aio.train' in cmd else None
    if total is not None and total >= 10**8:
        total = None
    t = [r['t_row'] for r in rows if _finite(r.get('t_row'))]
    t_row = sum(t)/len(t) if t else None
    done_rows = int(last.get('iter', 0))
    hours = last.get('hours')
    seen = rec.get('metrics_mtime')
    code = end.get('code') if isinstance(end, dict) else None
    if total and done_rows >= total:
        state = 'done'
    elif code is not None:
        state = 'done' if code == 0 else 'stopped'
    elif not rec.get('running'):
        state = 'stopped'
    elif seen and now - seen > max(4*(t_row or 0), 1800) + 900:
        state = 'quiet'
    else:
        state = 'running'
    eta = None
    if state == 'running' and t_row and seen:
        left = (total - done_rows) if total else None
        if cap_h is not None and _finite(hours):
            by_time = max(0., cap_h - hours)*3600/t_row
            left = by_time if left is None else min(left, by_time)
        if left is not None:
            eta = round(seen + left*t_row)
    ev = summarize_eval_records(rec.get('eval_rows'), rec.get('best_eval'))
    hb = rec.get('heartbeat') if isinstance(rec.get('heartbeat'), dict) else {}
    gpu_raw = hb.get('gpu') if isinstance(hb.get('gpu'), dict) else {}
    gpu = {k: v for k, v in gpu_raw.items() if v is not None and (not isinstance(v, float) or math.isfinite(v))} or None
    out = dict(
        name=rec['job'], job=f'{host}:{rec["job"]}', host=host, state=state,
        rows=done_rows, total=total, cap_hours=cap_h, hours=hours,
        t_row=round(t_row, 1) if t_row else None, eta=eta,
        row_at=round(seen) if _finite(seen) else None,
        started=round(spec['started']) if _finite(spec.get('started')) else None,
        ended=round(end['time']) if isinstance(end, dict) and _finite(end.get('time')) else None,
        eval=ev,
        gpu=gpu,
        thermal_warning=True if hb.get('thermal_warning') else None,
    )
    return {k: v for k, v in out.items() if v is not None and not (isinstance(v, float) and not math.isfinite(v))}


def poll_worker(host, keep_h=24., now=None, timeout=WORKER_TIMEOUT):
    """Poll one remote WSL2/Linux worker over SSH with strict timeout -> list of TrainRun dicts."""
    now = now or time.time()
    ssh_cmd = ['ssh', '-o', 'BatchMode=yes', '-o', f'ConnectTimeout={max(1, int(timeout))}', host]
    if host in WORKER_WSL:
        distro, user = WORKER_WSL[host]
        ssh_cmd.append(f'wsl -d {distro} -u {user} -- bash -s')
    else:
        ssh_cmd += ['bash', '-s']
    p = subprocess.run(ssh_cmd, input=_worker_probe_script(), text=True, capture_output=True, timeout=timeout, check=True)
    payload = json.loads(p.stdout.strip().splitlines()[-1])
    runs = []
    for rec in payload.get('jobs') or ():
        item = _worker_run_status(host, rec, keep_h, now)
        if item is not None:
            runs.append(item)
    return runs


def _load_worker_cache(path):
    if not path or not Path(path).exists():
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_worker_cache(path, cache):
    if not path:
        return
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix('.tmp')
        tmp.write_text(json.dumps(cache, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        os.replace(tmp, p)
    except OSError:
        pass


def collect(keep_h=24., now=None, workers=None, worker_timeout=WORKER_TIMEOUT, cache_path=None, poll_fn=None):
    now = now or time.time()
    runs = []
    for p in sorted(ROOT.glob('*/job.json')):
        try:
            job = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        end = job.get('done')
        if end and end.get('time') and now-end['time'] > keep_h*3600:
            continue
        subs = job.get('pack') or [dict(name='', cmd=job.get('cmd') or [])]
        for s in subs:
            if 'aio.train' not in words(s.get('cmd', [])):
                continue                                          # rereads and benches are not training
            try:
                runs.append(run_status(job, s, now))
            except Exception as e:                                # one odd job does not stop the rest
                print(f'{p.parent.name}/{s.get("name")}: {type(e).__name__}: {e}', file=sys.stderr)

    worker_list = [w.strip() for w in (workers.split(',') if isinstance(workers, str) else (workers or ())) if w.strip()]
    worker_meta = None
    if worker_list:
        cpath = cache_path if cache_path is not None else (ROOT/'.worker-status-cache.json')
        cache = _load_worker_cache(cpath)
        poller = poll_fn or poll_worker
        worker_meta = {}
        for host in worker_list:
            try:
                wruns = poller(host, keep_h=keep_h, now=now, timeout=worker_timeout)
                runs.extend(wruns)
                cache[host] = dict(updated=round(now), runs=wruns)
                worker_meta[host] = dict(ok=True, stale=False, updated=round(now), runs=len(wruns))
            except Exception as e:
                cached = cache.get(host) or {}
                stale_runs = []
                for r in cached.get('runs') or ():
                    copy = dict(r, stale=True)
                    if copy.get('state') == 'running':
                        copy['state'] = 'quiet'
                    stale_runs.append(copy)
                runs.extend(stale_runs)
                worker_meta[host] = dict(
                    ok=False, stale=True, updated=cached.get('updated'),
                    runs=len(stale_runs), error=f'{type(e).__name__}: {str(e)[-200:]}',
                )
        _save_worker_cache(cpath, cache)

    order = dict(running=0, quiet=0, stopped=1, done=1)
    runs.sort(key=lambda r: (order.get(r['state'], 2), -(r.get('started') or 0), r['name']))
    out = dict(updated=round(now), source='fleet' if worker_list else 'colab', runs=runs)
    if worker_meta is not None:
        out['workers'] = worker_meta
    return out


def push(host, dest, data):
    body = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    tmp = dest+'.tmp'
    subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host,
                    f'cat > {shlex.quote(tmp)} && mv {shlex.quote(tmp)} {shlex.quote(dest)}'],
                   input=body.encode(), check=True, timeout=60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--push', metavar='HOST', help='ssh destination of the site server')
    ap.add_argument('--dest', default=DEST)
    ap.add_argument('--every', type=int, default=0, help='seconds between pushes (0: once)')
    ap.add_argument('--keep', type=float, default=24., help='hours a finished job stays listed')
    ap.add_argument('--workers', default='', help='comma-separated WSL2 worker hosts to poll (e.g. guomao,baolong)')
    ap.add_argument('--worker-timeout', type=float, default=WORKER_TIMEOUT, help='per-worker SSH timeout in seconds (default 5)')
    ap.add_argument('--worker-cache', type=Path, default=None, help='path for graceful stale worker cache fallback')
    a = ap.parse_args()
    while True:
        data = collect(a.keep, workers=a.workers, worker_timeout=a.worker_timeout, cache_path=a.worker_cache)
        if a.push:
            try:
                push(a.push, a.dest, data)
                print(time.strftime('%H:%M:%S'), 'pushed', len(data['runs']), 'runs', flush=True)
            except Exception as e:                                # the network: tried again next round
                print(time.strftime('%H:%M:%S'), f'push failed: {type(e).__name__}: {e}', flush=True)
        else:
            print(json.dumps(data, ensure_ascii=False, indent=1))
        if not a.every:
            return
        time.sleep(a.every)


if __name__ == '__main__':
    main()

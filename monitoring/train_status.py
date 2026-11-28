"""The training runs' progress for the research site's progress bars: rows done of the planned rows, time per row, when
each run should end. Standard library only; runs on the Mac next to the Colab watchers (docs/colab.md).

  python3 -m monitoring.train_status                                   # print train-status.json
  python3 -m monitoring.train_status --push HOST --every 600           # keep the live site's copy current

Reads every job in ~/colab-runs (AIO_COLAB_ROOT): job.json (the command, or the pack's commands, and the starts), each
run's metrics.jsonl (one line per row: iter, t_row, hours) and the job's end. A run's planned rows are its --rows; with
--hours the run may end earlier, so its planned time is shown too. Jobs that ended more than --keep hours ago are left
out. --push writes the file into the live release on the server (HOST:DEST, a temporary name, then a rename), so the
page shows it without a new release; a deploy carries the live copy into the new release.
"""
import argparse, json, math, os, re, shlex, subprocess, sys, time
from pathlib import Path

ROOT = Path(os.environ.get('AIO_COLAB_ROOT', Path.home()/'colab-runs'))
DEST = '/home/ubuntu/decks/aio/train-status.json'
TRAIN_HOURS = 2.     # aio.train's default --hours


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
    t = [r['t_row'] for r in rows if isinstance(r.get('t_row'), (int, float))]
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
        if cap_h is not None and hours is not None:
            by_time = max(0., cap_h - hours)*3600/t_row
            left = by_time if left is None else min(left, by_time)
        if left is not None:
            eta = round(seen + left*t_row)
    out = dict(name=name, job=job['name'], state=state, rows=done_rows, total=total, cap_hours=cap_h,
               hours=hours, t_row=round(t_row, 1) if t_row else None, eta=eta, row_at=round(seen) if seen else None,
               started=round(job['starts'][0]['time']) if job.get('starts') else None,
               ended=round(end['time']) if end and end.get('time') else None)
    return {k: v for k, v in out.items() if v is not None and not (isinstance(v, float) and not math.isfinite(v))}


def collect(keep_h=24., now=None):
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
    order = dict(running=0, quiet=0, stopped=1, done=1)
    runs.sort(key=lambda r: (order[r['state']], -(r.get('started') or 0), r['name']))
    return dict(updated=round(now), source='colab', runs=runs)


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
    a = ap.parse_args()
    while True:
        data = collect(a.keep)
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

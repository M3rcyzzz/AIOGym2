"""Long jobs on Colab, driven from the Mac with the `colab` CLI (docs/colab.md). Standard library only.

  python3 -m monitoring.colab_job start NAME --ref BRANCH [--gpu G4,A100] [--put FILE]... -- \\
      python -m aio.train --out {run} --init {in}/FILE ... --resume
  python3 -m monitoring.colab_job status NAME
  python3 -m monitoring.colab_job stop NAME

start: a VM (--gpu: tried in the order given), the tree of --ref (git archive from here: the repository is private, no
credentials go to the VM), the --put files and, when the Mac has some of the run, its last.pt and small files; then the
command in the background there (setsid: apart from the kernel and from the Mac). {run} in it is the run's folder on the
VM, {in} the folder of the --put files. Then the watcher, in the background here (it keeps the Mac awake): every --every
seconds the run's new and changed files come back to ~/colab-runs/NAME/run; a VM that is gone, or older than
--max-hours, is replaced by a new one that goes on from the last last.pt that came back (aio.train --resume: with no
last.pt it starts from nothing, so one command serves every VM); when the command has ended its files come back and the
VM is stopped.
"""
import argparse, contextlib, fcntl, json, os, re, shutil, subprocess, sys, tarfile, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('AIO_COLAB_ROOT', Path.home()/'colab-runs'))
VM = '/content/aio'
MARK = '@@aio@@'
PART = 40 << 20             # a request to the VM's contents API carries up to ~100 MB of base64: bigger files go in parts
SMALL = 1 << 20             # besides last.pt and metrics.jsonl, the run's files up to this size go to a new VM
UPLOADS = 4                 # parts sent at once: one stream from the Mac is slow
DOWNLOADS = UPLOADS         # parts pulled at once in pull(): matches put()
LAST_EVERY = 1800           # default seconds between heavy last.pt syncs (30 min; always synced on rollover/exit)
POLL = 120                  # seconds between the watcher's looks at whether the command has ended
AHEAD = 3                   # hours before a VM is too old (--max-hours) from which the next one is asked for, while it runs on
PIP = ['numpy>=2', 'scipy']  # pyproject's dependencies (torch is the VM's own)
ROW_CKPT = re.compile(r'^row\d+\.pt$')   # bulk historical snapshots from --save-every; skipped by sync unless --sync-rows
TRANSIENT = (
    'SSLError', 'SSLEOFError', 'UNEXPECTED_EOF_WHILE_READING', 'EOF occurred in violation of protocol',
    'ConnectionError', 'ConnectionResetError', 'RemoteDisconnected', 'ChunkedEncodingError',
    'ProtocolError', 'ReadTimeout', 'ConnectTimeout', 'ProxyError', 'Max retries exceeded',
    'no answer in', 'token refresh failed', 'Bad Gateway', 'Service Unavailable', 'Gateway Timeout',
    '502', '503', '504',
)

RUNNER = r'''
import fcntl, json, os, subprocess, sys, threading, time
J = sys.argv[1]
held = open(J+'/runner.lock', 'w')
try:
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    sys.exit('another runner has the job')
spec = json.load(open(J+'/job.json'))
subs = spec.get('pack') or [dict(name='', src='src', cmd=spec['cmd'])]   # a pack: its commands at once, run/NAME each
env = dict(os.environ, PYTHONUNBUFFERED='1')
if spec.get('threads'):
    env.update(OMP_NUM_THREADS=str(spec['threads']), MKL_NUM_THREADS=str(spec['threads']))
log, held_log = open(J+'/job.log', 'a', buffering=1, errors='replace'), threading.Lock()
def write(s):
    with held_log:
        log.write(time.strftime('%Y-%m-%dT%H:%M:%SZ ', time.gmtime())+s+'\n')
if spec.get('mps'):
    write('mps: %d' % subprocess.run(['nvidia-cuda-mps-control', '-d']).returncode)
codes, procs = {}, []
for s in subs:
    run, tag = J+'/run'+('/'+s['name'] if s['name'] else ''), ('[%s] ' % s['name'] if s['name'] else '')
    os.makedirs(run, exist_ok=True)
    cmd = [x.replace('{run}', run).replace('{in}', J+'/in') for x in s['cmd']]
    write(tag+'start '+json.dumps(cmd))
    try:
        p = subprocess.Popen(cmd, cwd=J+'/'+s['src'], env=dict(env, PYTHONPATH=J+'/'+s['src']), stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
    except OSError as e:
        write(tag+'cannot start: %r' % e)
        codes[s['name']] = 127
        continue
    procs.append((s['name'], tag, p))
open(J+'/pid', 'w').write(' '.join(str(p.pid) for _, _, p in procs))
def pump(name, tag, p):
    for line in p.stdout:
        write(tag+line.rstrip('\n'))
    codes[name] = p.wait()
    write(tag+'exit %d' % codes[name])
threads = [threading.Thread(target=pump, args=x) for x in procs]
for t in threads:
    t.start()
for t in threads:
    t.join()
rc = next((c for c in codes.values() if c), 0)
json.dump(dict(code=rc, codes=codes, time=time.time()), open(J+'/exit', 'w'))
'''

STATUS = r'''
import json, os, shutil, subprocess, time
def read(p):
    try:
        return open(p).read().strip()
    except OSError:
        return None
files = {}
for b, _, fs in os.walk(J+'/run'):
    for f in fs:
        try:
            st = os.stat(os.path.join(b, f))
        except OSError:
            continue
        files[os.path.relpath(os.path.join(b, f), J)] = [st.st_size, st.st_mtime]
pids, ex = (read(J+'/pid') or '').split(), read(J+'/exit')
stats = [read('/proc/%s/stat' % p) for p in pids]
gpu = shutil.which('nvidia-smi')
gpu = subprocess.run([gpu, '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
                     capture_output=True, text=True).stdout.strip() if gpu else ''
print(MARK+json.dumps(dict(now=time.time(), files=files, exit=json.loads(ex) if ex else None, gpu=gpu,
                           running=any(s and s.rsplit(')', 1)[1].split()[0] not in 'ZX' for s in stats))))
'''

STAGE = r'''
import json, os, shutil, subprocess, tarfile, time, zipfile
x = J+'/.xfer'
shutil.rmtree(x, ignore_errors=True)
os.makedirs(x)
def copy(r, dst, loose=False):
    """J/r to dst whole: nothing wrote it while it was copied, and a .pt (a zip) has its end -> [size, mtime], or None."""
    for _ in range(1 if loose else 40):
        a = os.stat(J+'/'+r)
        shutil.copyfile(J+'/'+r, dst)
        b = os.stat(J+'/'+r)
        if loose or ((a.st_size, a.st_mtime_ns) == (b.st_size, b.st_mtime_ns) and os.path.getsize(dst) == b.st_size
                     and (not r.endswith('.pt') or zipfile.is_zipfile(dst))):
            return [b.st_size, b.st_mtime]
        time.sleep(.5)
    return None
got = {}
if TAR:
    with tarfile.open(x+'/f', 'w:gz') as t:
        for r in RELS:
            v = os.path.exists(J+'/'+r) and copy(r, x+'/one', loose=r == 'job.log')
            if v:
                t.add(x+'/one', arcname=r)
                got[r] = v
else:
    v = copy(RELS[0], x+'/f')
    got = {RELS[0]: v} if v else {}
size, parts = os.path.getsize(x+'/f') if os.path.exists(x+'/f') else 0, ['f']
if got and size > PART:
    subprocess.run(['split', '-b', str(PART), '-d', '-a', '3', 'f', 'f.'], cwd=x, check=True)
    os.remove(x+'/f')
    parts = sorted(p for p in os.listdir(x) if p.startswith('f.'))
print(MARK+json.dumps(dict(files=got, size=size, parts=parts)))
'''

LAUNCH = r'''
import fcntl, os, subprocess, tarfile
busy = False
with open(J+'/runner.lock', 'a') as f:                  # a runner holds it while its command runs: not a second one
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(f, fcntl.LOCK_UN)
    except OSError:
        busy = True
if not busy:
    for src in SRCS:                                    # src.tgz -> src/, and a pack's src-COMMIT.tgz -> src-COMMIT/
        if not os.path.exists(J+'/'+src+'/.unpacked'):
            with tarfile.open(J+'/'+src+'.tgz') as t:
                t.extractall(J+'/'+src, filter='data')
            open(J+'/'+src+'/.unpacked', 'w').close()
    if PIPS and not os.path.exists(J+'/.pipped'):
        subprocess.run(['python3', '-m', 'pip', 'install', '-q', *PIPS], check=True)
        open(J+'/.pipped', 'w').close()
    open(J+'/runner.py', 'w').write(RUNNER)
    for f in ('exit', 'pid'):
        if os.path.exists(J+'/'+f):
            os.remove(J+'/'+f)
    subprocess.run(['bash', '-c', 'setsid nohup python3 "$0/runner.py" "$0" > /dev/null 2>&1 < /dev/null &', J], check=True)
print(MARK+str(int(busy)))
'''


def say(name, msg):
    line = time.strftime('%Y-%m-%d %H:%M:%S ')+msg
    print(line, flush=True)
    with open(ROOT/name/'watch.log', 'a') as f:
        f.write(line+'\n')


def load(name):
    return json.loads((ROOT/name/'job.json').read_text())


def save(job):
    p = ROOT/job['name']/'job.json'
    p.with_suffix('.tmp').write_text(json.dumps(job, indent=1))
    os.replace(p.with_suffix('.tmp'), p)


@contextlib.contextmanager
def lock(name):
    """One of start, a round of watch, sync, stop at a time on a job."""
    with open(ROOT/name/'.lock', 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def proxy_env():
    """The environment for the `colab` CLI with the Mac's proxy as it is now (`scutil --proxy`), not as it was when a watcher
    started (11-24: the proxy app was changed, its port with it, and a watcher kept calling the old one for half an hour)."""
    e = dict(os.environ)
    if not shutil.which('scutil'):
        return e
    try:
        out = subprocess.run(['scutil', '--proxy'], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return e
    v = dict(re.findall(r'^\s*(\w+)\s*:\s*(\S+)\s*$', out, re.M))
    if v.get('HTTPSEnable') == '1' and v.get('HTTPSProxy') and v.get('HTTPSPort'):
        url = f'http://{v["HTTPSProxy"]}:{v["HTTPSPort"]}'
        for k in ('HTTPS_PROXY', 'HTTP_PROXY', 'https_proxy', 'http_proxy'):
            e[k] = url
    return e


def colab(*args, inp=None, timeout=900, tries=3):
    """The `colab` CLI; a failure is tried again (the network), then raised."""
    err = ''
    for i in range(tries):
        try:
            p = subprocess.run(['colab', *args], input=inp, stdin=None if inp is not None else subprocess.DEVNULL,
                               capture_output=True, text=True, timeout=timeout, env=proxy_env())
            if p.returncode == 0:
                return p
            err = (p.stdout+p.stderr)[-1500:]
            if 'authorize colab-cli' in err:                    # the CLI asks for a login when one token refresh fails (the network);
                err = 'token refresh failed once (the network); the login is still there'   # the next call works
        except subprocess.TimeoutExpired:
            err = f'no answer in {timeout} s'
        if i+1 < tries:
            time.sleep(15*(i+1))
    raise RuntimeError(f'colab {" ".join(args[:3])}: {err}')


def rexec(s, job, code, timeout=180, **vals):
    """Python on the kernel of VM s, with J (the job's folder there), MARK and vals defined -> what it printed after MARK."""
    head = ''.join(f'{k} = {v!r}\n' for k, v in dict(J=f'{VM}/{job["name"]}', MARK=MARK, PART=PART, **vals).items())
    p = colab('exec', '-s', s, '--timeout', str(int(timeout)), inp=head+code, timeout=timeout+30, tries=2)
    got = [x[len(MARK):] for x in p.stdout.splitlines() if x.startswith(MARK)]
    if not got:
        raise RuntimeError(f'exec on {s}: no answer: {(p.stdout+p.stderr)[-1500:]}')
    return json.loads(got[-1])


def sessions():
    return set(re.findall(r'^\[([^\]]+)\] \S+ \|', colab('sessions', timeout=300).stdout, re.M))


@contextlib.contextmanager
def keepalive(s, job):
    """The kernel of VM s run a line every 4 min: uploads alone do not count as use, and Colab took back VMs being filled (11-21)."""
    stop = threading.Event()
    def beat():
        while not stop.wait(240):
            try:
                rexec(s, job, 'print(MARK+"0")', timeout=120)
            except Exception:
                pass
    t = threading.Thread(target=beat, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()


def put(s, job, src, rel):
    """A local file to J/rel on the VM (in parts above PART, UPLOADS at once); its folder is there."""
    dst, size = f'{VM}/{job["name"]}/{rel}', src.stat().st_size
    if size <= PART:
        colab('upload', '-s', s, str(src), dst)
        return
    n = -(-size//PART)
    with tempfile.TemporaryDirectory() as d, keepalive(s, job):
        def one(i):
            p = Path(d)/f'p{i}'
            with open(src, 'rb') as f:
                f.seek(i*PART)
                p.write_bytes(f.read(PART))
            colab('upload', '-s', s, str(p), f'{dst}.{i:03d}')
            p.unlink()
        with ThreadPoolExecutor(UPLOADS) as ex:
            list(ex.map(one, range(n)))
    got = rexec(s, job, 'import os\nwith open(J+"/"+REL, "wb") as out:\n    for i in range(N):\n'
                '        p = J+"/"+REL+".%03d" % i\n        out.write(open(p, "rb").read())\n        os.remove(p)\n'
                'print(MARK+str(os.path.getsize(J+"/"+REL)))', REL=rel, N=n)
    if got != size:
        raise RuntimeError(f'{dst}: {got} bytes there, {size} here')


def wants_sync(rel, sync_rows=False, sync_last=True):
    """Whether a run file on the VM is pulled back to the Mac. Lightweight eval/row*.pt snapshots always sync;
    bulk historical rowN.pt snapshots (2+ GB each) are skipped by default unless --sync-rows is set; heavy
    last.pt can be throttled via sync_last=False between --last-every intervals (always synced on rollover/exit)."""
    p = Path(rel)
    if p.suffix == '.tmp' or p.name.startswith('.'):
        return False
    if p.parent.name == 'eval':
        return True
    if p.name == 'last.pt' and not sync_last:
        return False
    if sync_rows:
        return True
    return not ROW_CKPT.match(p.name)


def is_transient(err):
    s = str(err)
    return any(tok in s for tok in TRANSIENT)


def download_part(s, remote_path, local_path, expected_size=None, tries=5, backoff=5., sleep=time.sleep):
    """Download one staged chunk from VM s, retrying transient SSL/network errors with exponential backoff."""
    local_path = Path(local_path)
    last_err = None
    for i in range(tries):
        if local_path.exists():
            local_path.unlink()
        try:
            colab('download', '-s', s, remote_path, str(local_path), tries=1)
            if not local_path.exists():
                raise RuntimeError(f'colab download -s {s}: {remote_path} missing after download')
            if expected_size is not None and local_path.stat().st_size != expected_size:
                got_sz = local_path.stat().st_size
                raise RuntimeError(f'colab download -s {s}: {remote_path} truncated ({got_sz} of {expected_size} bytes)')
            return
        except RuntimeError as e:
            last_err = e
            if local_path.exists():
                local_path.unlink()
            transient = is_transient(e) or 'truncated' in str(e) or 'missing after download' in str(e)
            if i + 1 >= tries or (not transient and i >= 1):
                raise
            sleep(backoff * (2 ** i))
    raise last_err


def pull(s, job, rels, tar):
    """J/rels (each written by nobody for 5 s; job.log as it is) to the job's folder here -> {rel: [size, mtime] there}."""
    d = ROOT/job['name']
    st = rexec(s, job, STAGE, timeout=300, RELS=rels, TAR=tar)
    if not st['files']:
        return {}
    parts = st['parts']
    with tempfile.TemporaryDirectory(dir=d) as td:
        whole = Path(td)/'whole'
        if len(parts) == 1:
            download_part(s, f'{VM}/{job["name"]}/.xfer/{parts[0]}', whole, expected_size=st['size'])
        else:
            def one(item):
                idx, p = item
                expected = PART if idx < len(parts) - 1 else st['size'] - PART * (len(parts) - 1)
                part = Path(td)/f'part.{idx:03d}'
                download_part(s, f'{VM}/{job["name"]}/.xfer/{p}', part, expected_size=expected)
                return part
            with keepalive(s, job), ThreadPoolExecutor(min(DOWNLOADS, len(parts))) as ex:
                got_parts = list(ex.map(one, enumerate(parts)))
            with open(whole, 'wb') as out:
                for part in got_parts:
                    with open(part, 'rb') as inp:
                        shutil.copyfileobj(inp, out, 4 << 20)
                    part.unlink()
        if whole.stat().st_size != st['size']:
            raise RuntimeError(f'{rels[0]}...: {whole.stat().st_size} bytes came back, {st["size"]} there')
        if tar:
            with tarfile.open(whole) as t:
                t.extractall(d, filter='data')
        else:
            (d/rels[0]).parent.mkdir(parents=True, exist_ok=True)
            os.replace(whole, d/rels[0])
    for r, (_, mt) in st['files'].items():
        os.utime(d/r, (mt, mt))
    return st['files']


def save_seen(d, seen):
    p = d/'synced.json'
    p.with_suffix('.tmp').write_text(json.dumps(seen))
    os.replace(p.with_suffix('.tmp'), p)


def sync(job, s, sync_rows=None, sync_last=True):
    """The run's new and changed files and job.log from VM s -> (its status, the files that came back). The .pt first:
    metrics.jsonl is not behind the last.pt here (a row more is done again after a resume). Bulk rowN.pt snapshots are
    skipped unless sync_rows (or job['sync_rows']) is set; eval/row*.pt snapshots always sync."""
    if sync_rows is None:
        sync_rows = bool(job.get('sync_rows'))
    d = ROOT/job['name']
    seen = json.loads((d/'synced.json').read_text()) if (d/'synced.json').exists() else {}
    st = rexec(s, job, STATUS, timeout=120)
    new = [r for r, v in st['files'].items() if seen.get(r) != v and wants_sync(r, sync_rows, sync_last=sync_last)]
    got = {}
    for r in sorted((x for x in new if x.endswith('.pt')), key=lambda x: (x.endswith('/last.pt'), x)):   # last.pt last
        one = pull(s, job, [r], False)
        got.update(one)
        if one:
            seen.update(one)
            save_seen(d, seen)
    small = pull(s, job, sorted(x for x in new if not x.endswith('.pt'))+['job.log'], True)
    got.update(small)
    seen.update((r, v) for r, v in small.items() if r != 'job.log')
    save_seen(d, seen)
    return st, got


def shape(g):
    """`colab new` options for one entry of --gpu: a GPU (G4, A100, L4, ...), CPU (none), or a TPU host (V5E1, V6E1: many cores)."""
    g = g.upper()
    return [] if g == 'CPU' else ['--tpu', g] if g in ('V5E1', 'V6E1') else ['--gpu', g]


def new_vm(job, limit=3600):
    """A VM of one of the job's cards -> its name and card. Asked again 10 s after a refusal, the cards in turn (Mercy
    11-24; it was every 5 min), for `limit` seconds. A request is given up after 150 s (11-25: a refusal took minutes
    to come back, 14 requests in the hour); a VM that such a request made after all is taken."""
    s = f'{job["name"]}-{len(job["starts"])+1}'
    t, n = time.time(), 0
    while time.time()-t < limit:
        for g in job['gpu'].split(','):
            try:
                colab('new', '-s', s, *shape(g), *(['--high-mem'] if job['high_mem'] else []),
                      timeout=150, tries=1)
                return s, g
            except RuntimeError as e:
                try:
                    if s in sessions():                         # the request that was given up, or refused late, made the VM
                        return s, g
                except Exception:
                    pass
                n += 1
                if n <= 2 or n % 30 == 0:                       # (the log: the first refusals, then one in thirty)
                    say(job['name'], f'no {g} (asked {n} times): {str(e)[-300:]}')
            time.sleep(10)
    raise RuntimeError(f'no VM of {job["gpu"]} in {limit/60:.0f} min of asking')


def _cmd_words(cmd):
    out = []
    for x in cmd or ():
        out += x.split() if ' ' in x else [x]
    return out


def should_skip_in_pt(job, back, run_dir):
    """Skip uploading in/*.pt weights on rollover when run/last.pt is already in `back` and every command uses
    `--resume` (saves ~6.2 min / 1.16 CU per rollover without breaking fresh starts or non-resume jobs)."""
    if not back:
        return False
    back_rels = {str(Path('run')/f.relative_to(run_dir)) for f in back}
    subs = job.get('pack') or [dict(name='', cmd=job.get('cmd') or [])]
    for sub in subs:
        last_rel = f'run/{sub["name"]}/last.pt' if sub.get('name') else 'run/last.pt'
        if last_rel not in back_rels or '--resume' not in _cmd_words(sub.get('cmd')):
            return False
    return True


def prune_orphaned_tmps(job_dir):
    """Remove orphaned tmp* staging directories left behind inside a job folder by interrupted pulls."""
    removed = []
    d = Path(job_dir)
    if not d.exists():
        return removed
    for p in sorted(d.iterdir()):
        if p.is_dir() and p.name.startswith('tmp'):
            shutil.rmtree(p, ignore_errors=True)
            removed.append(p.name)
    return removed


def launch(job, vm=None):
    """A new VM for the job (vm: the one already there for it, `spare`), all it needs there, its command started."""
    name, d = job['name'], ROOT/job['name']
    s, g = vm or new_vm(job)
    job.pop('spare', None)
    job['session'] = s
    job['starts'].append(dict(time=time.time(), session=s, gpu=g))
    save(job)
    run = d/'run'
    back = [f for f in run.rglob('*') if f.is_file() and not f.name.startswith('.') and
            (f.name in ('last.pt', 'metrics.jsonl') or (f.suffix != '.pt' and f.stat().st_size <= SMALL))] if any(run.rglob('last.pt')) else []
    skip_in_pt = should_skip_in_pt(job, back, run)
    try:
        dirs = sorted({'in', 'run'} | {str(Path('run')/f.relative_to(run).parent) for f in back})
        rexec(s, job, 'import os\nfor p in DIRS:\n    os.makedirs(J+"/"+p, exist_ok=True)\nprint(MARK+"0")', DIRS=dirs)
        for src in srcs(job):
            put(s, job, d/f'{src}.tgz', f'{src}.tgz')
        for f in sorted((d/'in').iterdir()):
            if skip_in_pt and f.suffix == '.pt':
                continue
            put(s, job, f, f'in/{f.name}')
        for f in back:
            put(s, job, f, str(Path('run')/f.relative_to(run)))
        (d/'vm-job.json').write_text(json.dumps({k: job[k] for k in ('cmd', 'pack', 'threads', 'mps') if job.get(k)}))
        put(s, job, d/'vm-job.json', 'job.json')
        rexec(s, job, LAUNCH, timeout=900, RUNNER=RUNNER, PIPS=PIP+job.get('pip', []), SRCS=srcs(job))
        if back:                                                # what went there is what the Mac has: not fetched again
            seen = json.loads((d/'synced.json').read_text()) if (d/'synced.json').exists() else {}
            files = rexec(s, job, STATUS, timeout=120)['files']
            seen.update((r, files[r]) for r in (str(Path('run')/f.relative_to(run)) for f in back) if r in files)
            save_seen(d, seen)
    except Exception:                                           # a VM that runs nothing costs the same
        stop_vm(job, s)
        job['fails'] = job.get('fails', 0)+1
        save(job)
        raise
    job['fails'] = 0
    save(job)
    what = ' '.join(job['cmd']) if job.get('cmd') else 'pack of '+', '.join(f'{x["name"]} ({x["commit"][:9]})' for x in job['pack'])
    say(name, f'started on {s} ({g}), commit {job["commit"] if job.get("pack") else job["commit"][:9]}, {len(back)} files of the run sent: {what}')
    return s


def srcs(job):
    """The source trees a job's VM has: src (from src.tgz) for one command, one per commit for a pack."""
    return sorted({x['src'] for x in job['pack']}) if job.get('pack') else ['src']


def stop_vm(job, s):
    try:
        colab('stop', '-s', s, timeout=300)
    except RuntimeError as e:
        say(job['name'], f'stop {s}: {e}')


def finish(job, s, code):
    time.sleep(10)                                              # what was written as it ended is not 'being written' any more
    try:
        for _ in range(3):
            if not sync(job, s)[1].keys()-{'job.log'}:
                break
    except Exception as e:
        say(job['name'], f'last sync: {type(e).__name__}: {str(e)[-500:]}')
    stop_vm(job, s)
    job['done'] = dict(time=time.time(), code=code)
    save(job)
    log = ROOT/job['name']/'job.log'
    tail = log.read_text(errors='replace').splitlines()[-3:] if log.exists() else []
    say(job['name'], f'ended with exit {code}; {s} stopped; files in {ROOT/job["name"]/"run"}; log ends: {" | ".join(tail)[-600:]}')


def watcher(job):
    pid = job.get('watcher')
    if not pid:
        return None
    p = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True)
    return pid if 'colab_job watch' in p.stdout else None


def watch(name):
    """Until the job is done: sync every --every seconds, a new VM for one that is gone or too old, stop at the end."""
    with lock(name):
        job = load(name)
        pruned = prune_orphaned_tmps(ROOT/name)
        if pruned:
            say(name, f'watch: pruned {len(pruned)} orphaned tmp dir(s): {", ".join(pruned[:5])}')
        job['watcher'] = os.getpid()
        save(job)
    if shutil.which('caffeinate'):
        subprocess.Popen(['caffeinate', '-is', '-w', str(os.getpid())])
    say(name, f'watch: pid {os.getpid()}, every {job["every"]} s, last every {job.get("last_every", LAST_EVERY)} s, a new VM after {job["max_hours"]} h')
    gone = dead = 0
    synced = 0.
    last_synced = 0.
    wait = 0
    while True:                                                 # (the nap first: a `continue` below still waits)
        time.sleep(wait)
        wait = min(job['every'], POLL)
        try:
            with lock(name):
                job = load(name)
                if job.get('done'):
                    say(name, 'watch: the job is done')
                    return
                if job.get('fails', 0) >= 3:                   # never given up: a VM that did not start costs nothing
                    say(name, 'watch: three new VMs in a row did not start; the next in 30 min')
                    job['fails'] = 0
                    save(job)
                    wait = 1800
                    continue
                if not job.get('session'):                     # no VM yet: the watcher gets the first one (start does not wait for it)
                    launch(job)
                    continue
                s, live = job['session'], sessions()
                spare = job.get('spare')                        # the next VM, taken while this one still runs (below)
                if spare and spare['session'] not in live:
                    spare = None
                    job.pop('spare')
                    save(job)
                for x in live:                                  # a VM of this job that is not its current one only costs
                    if x != s and not (spare and x == spare['session']) and re.fullmatch(re.escape(name)+r'-\d+', x):
                        say(name, f'{x} is not the job\'s VM ({s}): stopped')
                        stop_vm(job, x)
                if s not in live:
                    gone += 1
                    wait = 60
                    if gone >= 2:                               # twice, a minute apart: not a hiccup of the listing
                        say(name, f'{s} is gone: a new VM goes on from the last last.pt here')
                        launch(job)
                        gone = 0
                    continue
                gone = 0
                age = (time.time()-job['starts'][-1]['time'])/3600
                if time.time()-synced >= job['every']:
                    last_every = job.get('last_every', LAST_EVERY)
                    need_last = bool(spare) or last_every <= 0 or (time.time()-last_synced >= last_every)
                    st, got = sync(job, s, sync_last=need_last)
                    synced = time.time()
                    if need_last and not spare:
                        last_synced = synced
                else:                                           # between syncs only whether it has ended: a VM idles less
                    st, got = rexec(s, job, STATUS, timeout=120), {}
                if st['exit'] is not None:
                    finish(job, s, st['exit']['code'])
                    return
                if not st['running']:                           # no exit written: the runner itself was ended
                    dead += 1
                    wait = 60
                    if dead >= 3:
                        job['revived'] = job.get('revived', 0)+1
                        save(job)
                        if job['revived'] > 2:
                            finish(job, s, None)
                            return
                        say(name, f'the command on {s} is not running and wrote no exit: started there again')
                        rexec(s, job, LAUNCH, timeout=900, RUNNER=RUNNER, PIPS=[], SRCS=srcs(job))
                        dead = 0
                    continue
                dead = 0
                # The next VM is asked for AHEAD hours before this one is too old, while this one goes on: it is stopped only
                # once the next is there (11-25: stopped first, then no card for over an hour and nothing trained).
                if age > job['max_hours']-AHEAD and not spare:
                    try:
                        vs, vg = new_vm(job, limit=300)
                        spare = job['spare'] = dict(session=vs, gpu=vg, time=time.time())
                        save(job)
                        say(name, f'{vs} ({vg}) is there for when {s} ({age:.1f} h) is stopped: at its next last.pt')
                        synced = 0.                             # the next round syncs
                        wait = 60
                    except RuntimeError as e:
                        if age > job['max_hours']:
                            say(name, f'{s} is {age:.1f} h old and no VM to go on with yet ({str(e)[-80:]}): it runs on')
                elif spare:
                    if any(k == 'run/last.pt' or k.endswith('/last.pt') for k in got) or time.time()-spare['time'] > 1800:
                        say(name, f'{s} is {age:.1f} h old: {spare["session"]} goes on from the last.pt here')
                        stop_vm(job, s)
                        launch(job, (spare['session'], spare['gpu']))
                        last_synced = time.time()
                    else:
                        synced = 0.                             # for the next last.pt
                        wait = 60
        except Exception as e:                                  # the network, the VM: tried again next round
            say(name, f'watch: {type(e).__name__}: {str(e)[-800:]}')
            wait = min(wait, 120)


def resolve(ref):
    git = ['git', '-C', str(REPO)]
    subprocess.run(git+['fetch', '-q', 'origin', ref], capture_output=True)
    for r in (f'origin/{ref}', ref):
        p = subprocess.run(git+['rev-parse', '--verify', '-q', r+'^{commit}'], capture_output=True, text=True)
        if p.returncode == 0:
            return p.stdout.strip()
    raise SystemExit(f'no commit for {ref}')


def spawn_watch(name):
    if watcher(load(name)):
        return
    with open(ROOT/name/'watch.out', 'a') as out:
        p = subprocess.Popen([sys.executable, '-m', 'monitoring.colab_job', 'watch', name], cwd=REPO, stdin=subprocess.DEVNULL,
                             stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    print(f'watcher pid {p.pid}, log {ROOT/name/"watch.log"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='do', required=True)
    p = sub.add_parser('start', help='make the job (with a command) or start it again (without), and its watcher')
    p.add_argument('name')
    p.add_argument('--gpu', default='G4,A100', help='tried in this order: a GPU, CPU (none) or a TPU host V5E1 / V6E1; pick by the work (docs/colab.md)')
    p.add_argument('--high-mem', action='store_true', help='with CPU: 8 cores, 51 GB (0.26 units an hour)')
    p.add_argument('--ref', default='main', help='branch or commit; the job keeps the commit it had at the start')
    p.add_argument('--put', action='append', default=[], help='a local file the command reads as {in}/NAME')
    p.add_argument('--pip', action='append', default=[], help='a package more to install on each VM')
    p.add_argument('--every', type=int, default=600, help='seconds between the watcher\'s rounds')
    p.add_argument('--last-every', type=int, default=LAST_EVERY,
                   help='seconds between heavy last.pt syncs (default 1800 = 30 min; always synced on rollover/exit)')
    p.add_argument('--max-hours', type=float, default=23., help='a VM this old is replaced (Colab ends them at 24 h)')
    p.add_argument('--no-watch', action='store_true')
    p.add_argument('--pack', type=Path, help='instead of a command: a JSON list of {"name", "ref", "cmd"} run at once on the VM, '
                   'each in {run}=run/NAME on its own commit; one run alone leaves the G4 mostly idle (docs/colab.md)')
    p.add_argument('--threads', type=int, help='OMP/MKL threads of each command')
    p.add_argument('--mps', action='store_true', help='NVIDIA MPS on the VM: the commands\' kernels share the GPU at once')
    p.add_argument('--sync-rows', action='store_true', help='also pull historical rowN.pt checkpoints (skipped by default to save Mac disk)')
    for q in ('watch', 'sync', 'status', 'stop'):
        sp = sub.add_parser(q)
        sp.add_argument('name')
        if q == 'sync':
            sp.add_argument('--sync-rows', action='store_const', const=True, default=None,
                            help='also pull historical rowN.pt checkpoints for this sync')
            sp.add_argument('--no-last', action='store_true',
                            help='skip heavy last.pt during this manual sync (eval/row*.pt and logs still sync)')
    argv = sys.argv[1:]
    cmd = argv[argv.index('--')+1:] if '--' in argv else []      # the command: all after --, with {run} and {in}
    a = ap.parse_args(argv[:argv.index('--')] if '--' in argv else argv)
    d = ROOT/a.name
    if a.do == 'start':
        if (d/'job.json').exists():
            if cmd:
                raise SystemExit(f'{a.name} is there ({d}): start it again without a command, or take another name')
            with lock(a.name):                                  # (not beside a watcher's own new VM)
                job = load(a.name)
                if not job.get('done') and job.get('session') and job['session'] in sessions():   # after the Mac was off: its watcher again
                    print(f'{a.name} runs on {job["session"]}')
                else:                                           # its watcher gets it a VM (11-23: a start that waited in the foreground for a G4 died with the shell it ran in)
                    job.pop('done', None)
                    job['fails'], job['session'] = 0, None
                    save(job)
                    print(f'{a.name}: its watcher gets it a VM')
            if not a.no_watch:
                spawn_watch(a.name)
            return
        else:
            if bool(cmd) == bool(a.pack):
                raise SystemExit('a command (after --) or --pack FILE, one of them')
            for x in ('in', 'run'):
                (d/x).mkdir(parents=True, exist_ok=True)
            pack = None
            if a.pack:
                pack = [dict(name=x['name'], commit=resolve(x['ref']), cmd=x['cmd']) for x in json.loads(a.pack.read_text())]
                for x in pack:
                    x['src'] = f'src-{x["commit"][:9]}'
                    if not (d/f'{x["src"]}.tgz').exists():
                        subprocess.run(['git', '-C', str(REPO), 'archive', '--format=tar.gz', '-o', str(d/f'{x["src"]}.tgz'), x['commit']],
                                       check=True)
                commit = ','.join(sorted({x['commit'][:9] for x in pack}))
            else:
                commit = resolve(a.ref)
                subprocess.run(['git', '-C', str(REPO), 'archive', '--format=tar.gz', '-o', str(d/'src.tgz'), commit], check=True)
            for f in a.put:
                shutil.copy2(Path(f).expanduser(), d/'in'/Path(f).name)
            job = dict(name=a.name, ref=a.ref, commit=commit, gpu=a.gpu, high_mem=a.high_mem, pip=a.pip, cmd=cmd, pack=pack,
                       threads=a.threads, mps=a.mps, every=a.every, max_hours=a.max_hours, sync_rows=a.sync_rows, starts=[], made=time.time())
            if a.last_every != LAST_EVERY:
                job['last_every'] = a.last_every
        job['session'] = None
        save(job)
        print(f'{a.name}: its watcher gets it a VM (watch.log)')
        if not a.no_watch:
            spawn_watch(a.name)
    elif a.do == 'watch':
        watch(a.name)
    elif a.do == 'sync':
        job = load(a.name)
        with lock(a.name):
            st, got = sync(job, job['session'], sync_rows=a.sync_rows, sync_last=not a.no_last)
        print(json.dumps(dict(running=st['running'], exit=st['exit'], gpu=st['gpu'], came_back=sorted(got))))
    elif a.do == 'stop':
        job = load(a.name)
        with lock(a.name):
            if job.get('session'):
                try:
                    sync(job, job['session'])
                except Exception as e:
                    print(f'last sync: {e}')
                stop_vm(job, job['session'])
            job['done'] = dict(time=time.time(), code=None, stopped=True)
            save(job)
        say(a.name, f'stopped by hand; files in {d/"run"}')
    else:
        job = load(a.name)
        t = job['starts'][-1]['time'] if job['starts'] else job['made']
        print(f'{a.name}: commit {job["commit"] if job.get("pack") else job["commit"][:9]} ({job["ref"]}), VM {job.get("session")} ({len(job["starts"])} so far, '
              f'this one {(time.time()-t)/3600:.1f} h), watcher {watcher(job) or "none"}, '
              f'{"done "+json.dumps(job["done"]) if job.get("done") else "not done"}')
        if not job.get('done') and job.get('session'):
            try:
                st = rexec(job['session'], job, STATUS, timeout=120)
                print(f'  on the VM: running {st["running"]}, exit {st["exit"]}, GPU (MiB used, total, %) {st["gpu"]}')
            except RuntimeError as e:
                print(f'  VM: {e}')
        for f in ('job.log', 'run/metrics.jsonl'):
            if (d/f).exists():
                print(f'  {f}: '+(d/f).read_text(errors='replace').splitlines()[-1][:400] if (d/f).stat().st_size else f'  {f}: empty')
        print('  '+' | '.join(x.strip() for x in colab('usage', timeout=120).stdout.splitlines() if x.strip()))


if __name__ == '__main__':
    main()

"""progress.json for https://m3rcyzzz.club/aio/ (user request, 2026-11-02: keep the site current).

Two sources, merged at publish time:
  * site_progress_notes.json (next to this file): the bilingual status line, insights,
    progress log, which training runs to show, and the P0 reference tables (p0_refs).
    Edit it at each milestone.
  * training runs on baolong (scenario-*, ph-*, mixed-*, apc-*, sim-*), read over ssh without touching
    them: <runs>/<run>/seedN/{run.json,metrics.jsonl,done.json,train.log} and the p0_eval
    reports, <run>/p0_report*.json or <run>/seed*/p0_report*.json. A variant trained inside a
    run's folder, <run>/<variant>-seedN (apc-v3/anchor-seed1, mixed-*/seq-seed1), is its own run
    with id <run>-<variant>. A run on the unified simulation core (aio.train, sim-*, docs/sim.md) writes
    final.json instead of done.json, and its validations as val {x_pi, ...} and lit {wb, ory, ty, al}.
If baolong cannot be reached, the runs from the previous progress.json are kept and marked
stale. `freshness` records each block's source run and time; publish_bilingual_site refuses
to build when the curated notes are behind the data (a newer run, or a P0 report newer
than the insights), and the page labels stale blocks.

  python -m experiments.shared_rl_view.site_progress --out progress.json [--host baolong]
"""
from decimal import Decimal, ROUND_HALF_UP
import argparse
import datetime as dt
import json
import math
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTES = HERE / 'site_progress_notes.json'
PYTHON = r'D:\codebase\AIOGym2\.venv\Scripts\python.exe'
RUNS = r'D:\aiogym2-runs'
CST = dt.timezone(dt.timedelta(hours=8))
TRAIN_POINTS = 400  # episode-return points kept per seed
NOTES_MAX_LAG_H = 4  # insights older than this while a run is training are labelled stale
PENDING_GRACE_H = 3  # a 'no model yet' notice may run this far past its planned start before publishing refuses
PAUSE_MAX_H = 12  # a 'training paused, resume time to be decided' notice must be revisited within this many hours

# Runs on baolong (Windows, Python 3.12); prints one ASCII JSON line.
REMOTE = r'''
import glob, json, ntpath, os, re, time
ROOT = %r
KEEP = ('it', 'samples', 'hours', 'sps', 'lr', 'pi_updates', 'kl_final', 'ent', 'ep_return', 'ep_fail',
        'val_iqm', 'val_median', 'val_worst', 'val_fail',
        'ep_return_tep', 'ep_fail_tep', 'tep_val_iqm', 'tep_val_median', 'tep_val_fail', 'score',
        'guard', 'guard_ok', 'guard_violation', 'best_pt', 'bc',
        'iter', 'beta', 'id', 'sim', 'student_score', 'sim_score', 'val', 'lit')   # aio.train (sim-*): val {x_pi, ...}, lit {wb, ...}
# run.json keys shown on the page; of the forgetting guards (rsac --anchor/--guard-families) only what is
# guarded, the reference IQMs and each family's teacher as 'init' or <run>/<seed>/<file>, never a full path
CONF = ('seed', 'envs', 'steps', 'minibatches', 'lr', 'lr_max', 'desired_kl', 'max_samples', 'max_hours',
        'eval_every', 'patience', 'tep_envs', 'reward_scale', 'plant', 'family_scale', 'scale_anchor', 'snapshot_hours',
        'tep_frac', 'anchor_families', 'anchor_coef', 'guard_families', 'guard_frac', 'guard_ref_iqm', 'anchor_of',
        'plant_envs', 'train_horizon', 'qnorm', 'anchor_filter', 'critic_family',
        # aio.train (sim-*): its budget, batches, validation batch and the references measured on it at the start
        'trainer', 'hours', 'window', 'seqs', 'epochs', 'beta_iters', 'beta_min', 'sim_envs', 'sim_start', 'chunk', 'val_n', 'val_seed',
        'lit', 'init', 'val_refs', 'lit_formula_pi', 'scale')
# per-family columns of a multi-plant run (rsac --plant a,b,c): val_iqm_qtank, ep_return_shell, ...
FAM = re.compile(r'(val_(iqm|median|fail)|ep_(return|fail))_[a-z0-9]+$')
def load(p):
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
def progress(p, tail=16384):
    # last JSON progress line of a training log (stdout; UTF-16 when PowerShell redirected it)
    try:
        size = os.path.getsize(p)
        with open(p, 'rb') as f:
            utf16 = f.read(2) == b'\xff\xfe'
            start = max(0, size-tail)
            start -= (start & 1) if utf16 else 0
            f.seek(start); raw = f.read()
    except OSError:
        return None
    text = raw.decode('utf-16-le' if utf16 else 'utf-8', errors='ignore')
    for line in reversed(text.splitlines()):
        line = line.strip().lstrip('\ufeff')
        if line.startswith('{'):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and r.get('samples') is not None:
                return {k: r[k] for k in ('it', 'iter', 'samples', 'hours', 'sps') if k in r}
    return None
out = dict(now=time.time(), runs={})
def read(run, var, sdirs):
    seeds = {}
    for sd in sdirs:
        rows, m, rj = [], os.path.join(sd, 'metrics.jsonl'), os.path.join(sd, 'run.json')
        try:
            with open(m, encoding='utf-8') as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue  # a line still being written
                    rows.append({k: v for k, v in r.items() if k in KEEP or FAM.match(k)})
        except OSError:
            pass
        meta = load(rj) or {}
        # metrics.jsonl may be written only at validations (recurrent SAC: about every 30 min);
        # train.log carries progress lines in between, and latest.pt is saved every few minutes
        # (rsac --save-every-min), so liveness uses all three
        lg = os.path.join(sd, 'train.log')
        times = [os.path.getmtime(f) for f in (m, lg, os.path.join(sd, 'latest.pt')) if os.path.exists(f)]
        conf = {k: meta.get(k) for k in CONF}
        if isinstance(conf.get('anchor_of'), dict):   # teacher per family: 'init', or the checkpoint's last three path parts
            conf['anchor_of'] = {q: v if v == 'init' else '/'.join(re.split(r'[\\/]+', str(v))[-3:]) for q, v in conf['anchor_of'].items()}
        init = conf.pop('init', None)
        if init and meta.get('trainer') == 'aio.train':   # aio.train --init: the checkpoint's last two path parts, never a full path
            conf['init'] = '/'.join(re.split(r'[\\/]+', str(init))[-2:])
        seeds[os.path.basename(sd)[len(var)+1:] if var else os.path.basename(sd)] = dict(
            rows=rows, done=load(os.path.join(sd, 'done.json')) or load(os.path.join(sd, 'final.json')), config=conf,
            started=os.path.getctime(rj) if os.path.exists(rj) else None,
            modified=max(times) if times else None, progress=progress(lg))
    # p0_eval reports, run-level (p0_report.json) or per seed (seedN/p0_report_5h.json); the newest per seed wins
    p0, p0_time = {}, None
    files = (glob.glob(os.path.join(run, 'p0_report*.json')) if var is None else []) + [f for d in sdirs for f in glob.glob(os.path.join(d, 'p0_report*.json'))]
    for rp in sorted(files, key=os.path.getmtime):
        rep = load(rp) or {}
        for key, v in rep.items():
            if isinstance(v, dict) and 'all' in v:
                v.pop('scenarios', None)
                v['report'] = os.path.basename(rp)
                name = ntpath.basename(key.rstrip('\\/'))
                p0[name[len(var)+1:] if var and name.startswith(var + '-') else name] = v
                p0_time = max(p0_time or 0, os.path.getmtime(rp))
    starts = [s['started'] for s in seeds.values() if s['started']]
    out['runs'][os.path.basename(run) + ('-' + var if var else '')] = dict(seeds=seeds, p0=p0, p0_time=p0_time, started=min(starts) if starts else None)
for run in sorted(glob.glob(os.path.join(ROOT, 'scenario-*')) + glob.glob(os.path.join(ROOT, 'ph-*')) + glob.glob(os.path.join(ROOT, 'mixed-*')) + glob.glob(os.path.join(ROOT, 'apc-*'))
                 + glob.glob(os.path.join(ROOT, 'sim-*'))):
    if not os.path.isdir(run):
        continue
    # seedN directories, and variants trained in the same folder (anchor-seed1 -> run '<run>-anchor', zoo-tok-seed1 -> '<run>-zoo-tok');
    # a set-aside attempt such as seed1-fp32 is not a seed of anything
    variants = {}
    for d in sorted(glob.glob(os.path.join(run, '*seed*'))):
        mt = re.fullmatch(r'(?:([a-z0-9]+(?:-[a-z0-9]+)*)-)?(seed[0-9]+)', os.path.basename(d))
        if mt and os.path.isdir(d):
            variants.setdefault(mt.group(1), []).append(d)
    for var, sdirs in sorted(variants.items(), key=lambda kv: kv[0] or ''):
        read(run, var, sdirs)
print(json.dumps(out))
''' % RUNS


def fetch(host):
    # host 'local': the publisher runs on the training machine itself (11-13: publishing moved to baolong, which has
    # no ssh to itself), so the reader runs here with this interpreter
    cmd = [sys.executable, '-X', 'utf8', '-'] if host == 'local' else \
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, PYTHON, '-X', 'utf8', '-']
    r = subprocess.run(cmd, input=REMOTE, text=True, capture_output=True, timeout=180, check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def stamp(t):
    return dt.datetime.fromtimestamp(t, CST).isoformat(timespec='minutes') if t else None


def thin(points, n=TRAIN_POINTS):
    """Every k-th point plus the last one, so long runs stay small."""
    if len(points) <= n:
        return points
    k = -(-len(points) // n)
    return points[::k] + ([points[-1]] if (len(points) - 1) % k else [])


def p0_row(rep):
    """The 'all' group of one p0_eval entry, in the shape the page draws."""
    a = rep.get('all') or {}
    keep = ('iqm', 'iqm_ci95', 'win', 'tie', 'loss', 'median')
    return dict(samples=rep.get('samples'), iteration=rep.get('iteration'), n=a.get('n'), iqm=a.get('iqm'),
                median=a.get('median'), worst=a.get('worst'), gate=a.get('gate'),
                vs_dmc={k: (a.get('vs_dmc') or {}).get(k) for k in keep},
                vs_oracle={k: (a.get('vs_oracle') or {}).get(k) for k in keep},
                nominal=[(rep.get('nominal') or {}).get('out1_s'), (rep.get('nominal') or {}).get('out2_s')])


# a seed is running while metrics.jsonl, train.log or latest.pt changed this recently. rsac writes a progress
# line every 250 iterations (about 7 min, 11 under load) and nothing during a validation (about 5 min), so
# 12-17 quiet minutes are normal on a fifteen-plant run (11-07)
LIVE_S = 1800


def finite(r):
    """A dict without its non-finite numbers. A collapsed episode can log -inf (round 3, random flowsheets, it 14000),
    which Python's json writes as -Infinity and browsers refuse: the published progress.json did not parse (11-09)."""
    return {k: v for k, v in r.items() if not (isinstance(v, float) and not math.isfinite(v))}


LIT_MODELS = ('wb', 'ory', 'ty', 'al')   # the literature columns aio.train --lit evaluates (docs/heldout.md section 2)


def sim_summary(rows, done):
    """A run on the unified simulation core (aio.train): every validation as a multiple of the formula PI on the
    same batch (val.x_pi; 1 = formula PI, lower is better), the literature columns' multiples recorded with it,
    best.pt as aio.train writes it (each strictly lower x_pi, the final validation included) and the last one.
    lit is recorded only: the verdict reads best.pt and last.pt, never the row with the best lit."""
    ev = [r for r in rows if isinstance(r.get('val'), dict) and r['val'].get('x_pi') is not None]
    fin = done if isinstance(done, dict) and isinstance(done.get('val'), dict) and done['val'].get('x_pi') is not None else None
    pts = [dict(it=r.get('iter'), samples=r.get('samples'), hours=r.get('hours'), val=finite(r['val']),
                lit=finite(r['lit']) if isinstance(r.get('lit'), dict) else None) for r in ev]
    if fin:
        pts.append(dict(it=fin.get('iter'), samples=fin.get('samples'), hours=fin.get('hours'), val=finite(fin['val']),
                        lit=finite(fin['lit']) if isinstance(fin.get('lit'), dict) else None, final=True))
    best = None
    for q in pts:
        if best is None or q['val']['x_pi'] < best['val']['x_pi']:
            best = q
    pick = lambda q: dict(it=q['it'], samples=q['samples'], hours=q['hours'], x_pi=q['val']['x_pi'], score=q['val'].get('score'),   # noqa: E731
                          cost_x_pi=q['val'].get('cost_x_pi'), win=q['val'].get('win'), failed=q['val'].get('failed'), lit=q['lit'],
                          con=q['val'].get('con'), con_pi=q['val'].get('con_pi'))   # con: share of the settled decisions at constrained points spent outside
    return dict(sim_val=[[q['samples'], q['val']['x_pi'], q['val'].get('cost_x_pi'), q['val'].get('win'), q['val'].get('failed')] for q in pts],
                lit_val={g: [[q['samples'], q['lit'][g]] for q in pts if q['lit'] and q['lit'].get(g) is not None] for g in LIT_MODELS},
                student=thin([[r['samples'], r['student_score']] for r in rows if r.get('student_score') is not None and r.get('samples') is not None]),
                best_sim=pick(best) if best else None, last_sim=pick(pts[-1]) if pts else None)


def seed_summary(name, s, p0, now):
    rows = [finite(r) for r in s.get('rows') or []]
    val = [[r['samples'], r['val_iqm'], r.get('val_median'), r.get('val_worst'), r.get('val_fail')]
           for r in rows if r.get('val_iqm') is not None]
    train = thin([[r['samples'], r['ep_return']] for r in rows if r.get('ep_return') is not None])
    # mixed four-tank + TEP runs: TEP validation, and checkpoints chosen by the geometric mean of both IQMs
    tep_val = [[r['samples'], r['tep_val_iqm'], r.get('tep_val_median'), r.get('tep_val_fail'), r.get('score')]
               for r in rows if r.get('tep_val_iqm') is not None and r.get('val_iqm') is not None]
    train_tep = thin([[r['samples'], r['ep_return_tep']] for r in rows if r.get('ep_return_tep') is not None])
    last = rows[-1] if rows else {}
    sim = sim_summary(rows, s.get('done')) if (s.get('config') or {}).get('trainer') == 'aio.train' else {}
    p = finite(s.get('progress') or {})
    if p.get('samples') is not None and p['samples'] > (last.get('samples') or 0):
        last = dict(last, **p)   # newer progress line from the training log
    best = min(val, key=lambda v: v[1]) if val else None
    best_mixed = min(tep_val, key=lambda v: v[4] if v[4] is not None else float('inf')) if tep_val else None
    # several families in one run: validation IQM per family (TEP as 'tep'), and the checkpoint with the
    # lowest selection score (geometric mean of all families' IQMs, rsac 'score')
    fams = sorted({k[len('val_iqm_'):] for r in rows for k in r if k.startswith('val_iqm_')})
    suite = {}
    if fams:
        evals = [r for r in rows if r.get('score') is not None]
        cols = fams + (['tep'] if any(r.get('tep_val_iqm') is not None for r in evals) else [])
        key = lambda q: 'tep_val_iqm' if q == 'tep' else f'val_iqm_{q}'
        fam_val = {q: [[r['samples'], r[key(q)], r.get('tep_val_median' if q == 'tep' else f'val_median_{q}'),
                        r.get('tep_val_fail' if q == 'tep' else f'val_fail_{q}')] for r in evals if r.get(key(q)) is not None] for q in cols}
        # best.pt: rsac marks the row that wrote it (best_pt, PR #116: feasibility first, so with forgetting
        # guards a checkpoint within the guard beats any outside it); older runs: a score 2% below the best so far
        marked = [r for r in evals if r.get('best_pt')]
        b, low = (marked[-1] if marked else None), float('inf')
        for r in ([] if marked else evals):
            if r['score'] < low*.98 and r.get('guard_ok', True):
                b, low = r, r['score']
        # forgetting guard per validation: samples, {family: IQM of (run - reference) per scenario}, passed
        guard = [[r['samples'], finite(r['guard']), bool(r.get('guard_ok'))] for r in evals if isinstance(r.get('guard'), dict)]
        suite = dict(fam_val=fam_val, score_val=[[r['samples'], r['score']] for r in evals], **(dict(guard=guard) if guard else {}),
                     fam_train={q: thin([[r['samples'], r[f'ep_return_{q}']] for r in rows if r.get(f'ep_return_{q}') is not None]) for q in cols},
                     best_suite=dict(samples=b['samples'], score=b['score'], iqm={q: b.get(key(q)) for q in cols},
                                     **({k: b.get(k) for k in ('guard_ok', 'guard_violation')} if 'guard' in b else {})) if b else None)
        tep_val = []    # the mixed four-tank + TEP fields do not apply
    if s.get('done'):
        state = 'done'
    elif s.get('modified') and now - s['modified'] < LIVE_S:
        state = 'running'
    else:
        state = 'stopped'
    return dict(seed=name, state=state, samples=last.get('samples'), hours=last.get('hours'), sps=last.get('sps'),
                lr=last.get('lr'), started=stamp(s.get('started')), updated=stamp(s.get('modified')),
                best=dict(samples=best[0], val_iqm=best[1]) if best else None, val=val, train=train,
                config=s.get('config'), p0=p0_row(p0[name]) if name in p0 else None, **suite, **sim,
                **(dict(tep_val=tep_val, train_tep=train_tep, best_mixed=dict(
                    samples=best_mixed[0], tep_val_iqm=best_mixed[1], score=best_mixed[4],
                    val_iqm=next((v[1] for v in val if v[0] == best_mixed[0]), None))) if tep_val else {}))


TONES = ('good', 'bad')   # a log entry opens with Good news / Bad news, a clear view first (user 11-09)
AUTO_GAP_S = 1800   # an automatic log entry within this of a hand-written one is left out (the notes already say it)
AUTO_DAYS = 3       # automatic entries older than this are left to the hand-written log


def half_up(x, nd=0):
    """Round half away from zero (site rule 10), as an int for nd=0, else a string with nd decimals."""
    q = Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    return int(q) if nd == 0 else f'{q:.{nd}f}'


def log_name(run, zh=False):
    """Short round name for the log: the label's second part up to its first comma ('第二轮', 'Round 2'),
    or `log_name`/`log_name_zh` from the notes."""
    own = run.get('log_name_zh' if zh else 'log_name')
    if own:
        return own
    parts = ((run.get('label_zh') if zh else None) or run.get('label') or run['id']).split(' · ')
    name = re.split('[，,]', parts[1] if len(parts) > 1 else parts[0])[0].strip()
    return name if zh else name[:1].upper() + name[1:]


def times(x):
    """A multiple of the formula PI to two significant figures, half up (site rule 10): 0.096×, 0.71×, 1.2×, 12×."""
    k = max(0, 1 - math.floor(math.log10(x))) if x > 0 else 2
    return (half_up(x, k).rstrip('0').rstrip('.') if k else f'{half_up(x):,}') + '×'   # 0.5×, not 0.50× (as the page writes it)


SIM_LOG_DROP = .05   # an aio.train best.pt is logged when 5% below the last one logged (the replay is stale from then on too)


def sim_log(rows, en, zh, at, t0):
    """aio.train writes best.pt at every strictly lower validation x_pi; the log takes the first validation and then each
    best.pt at least SIM_LOG_DROP below the last one logged, so a fast first hour does not flood it. Below 1x is good news;
    the first validation above 1x (the start does worse than the formula PI on the new core) is bad news."""
    out, low, logged = [], float('inf'), None
    for r in rows:
        v = (r.get('val') or {}).get('x_pi') if isinstance(r.get('val'), dict) else None
        if v is None or r.get('hours') is None or not v < low:
            continue
        low = v
        if logged is not None and v > logged*(1 - SIM_LOG_DROP):
            continue
        t = t0 + r['hours']*3600
        head_en, head_zh = f"{en}, it {r.get('iter')}: validation {times(v)} formula PI", f"{zh} it {r.get('iter')}：验证 {times(v)} 公式 PI"
        if logged is None:
            tone = 'good' if v < 1 else 'bad'
            why_en = 'the first validation, already better than the formula PI' if v < 1 else 'the first validation, still worse than the formula PI'
            why_zh = '第一次验证，已好于公式 PI' if v < 1 else '第一次验证，还不如公式 PI'
        else:
            drop = half_up((1 - v/logged)*100, 1)
            tone = 'good'
            why_en = f"{drop}% below {times(logged)}, the kept checkpoint" + ('' if v < 1 else '; still worse than the formula PI')
            why_zh = f"比 {times(logged)} 低 {drop}%，换为保留权重" + ('' if v < 1 else '，仍不如公式 PI')
        out.append(dict(date=at(t), t=t, auto=True, tone=tone, text=f'{head_en}, {why_en}.', text_zh=f'{head_zh}，{why_zh}。'))
        logged = v
    return out


def auto_log(spec, raw):
    """Progress-log entries read from the training machine (user 11-09: the log stood still overnight while
    round 2 ran): a seed's start, each validation that wrote best.pt, and its end (done.json; final.json for
    aio.train). Times are the seed's start (run.json) plus the training clock, so a validation may read a few
    minutes early."""
    out, many = [], len(raw) > 1
    for name, s in sorted(raw.items()):
        t0 = s.get('started')
        if not t0:
            continue
        en, zh = log_name(spec), log_name(spec, True)
        if many:
            en, zh = f'{en}, {name}', f"{zh}（{name.replace('seed', '种子 ')}）"
        at = lambda t: dt.datetime.fromtimestamp(t, CST).strftime('%Y-%m-%d %H:%M')
        out.append(dict(date=at(t0), t=t0, auto=True, tone='good', text=f'{en} started.', text_zh=f'{zh}开训。'))
        if (s.get('config') or {}).get('trainer') == 'aio.train':
            out += sim_log(s.get('rows') or [], en, zh, at, t0)
        kept = None   # the row best.pt holds so far
        for r in s.get('rows') or []:
            if r.get('best_pt') and r.get('score') is not None and r.get('hours') is not None:
                # why it was written (rsac, PR #116: feasibility first): outside the forgetting guard only while no
                # checkpoint is within it; the first one within it replaces any score; after that a 2% lower score
                head_en, head_zh = f"{en}, it {r.get('it')}: selection score {half_up(r['score']):,}", f"{zh} it {r.get('it')}：选模分数 {half_up(r['score']):,}"
                guarded, ok = 'guard_ok' in r, bool(r.get('guard_ok', True))
                tone = 'bad' if guarded and not ok else 'good'      # a kept checkpoint that forgets is bad news
                if guarded and not ok:
                    why_en, why_zh = 'outside the forgetting guard, kept for now (none within it yet)', '不满足防遗忘约束，暂作保留权重（还没有满足约束的点）'
                elif guarded and kept is not None and not kept.get('guard_ok', True):
                    why_en, why_zh = 'the first within the forgetting guard, now the kept checkpoint', '第一个满足防遗忘约束的点，成为保留权重'
                elif kept is not None:
                    drop = half_up((1 - r['score']/kept['score'])*100, 1)
                    why_en = f"{drop}% below the kept {half_up(kept['score']):,}, now the kept checkpoint"
                    why_zh = f"比保留的 {half_up(kept['score']):,} 低 {drop}%，换为保留权重"
                else:
                    why_en, why_zh = 'the first kept checkpoint', '第一个保留权重'
                out.append(dict(date=at(t0 + r['hours']*3600), t=t0 + r['hours']*3600, auto=True, tone=tone,
                                text=f'{head_en}, {why_en}.', text_zh=f'{head_zh}，{why_zh}。'))
                kept = r
        d = s.get('done')
        if d and s.get('modified'):
            h, n = d.get('hours'), d.get('samples')
            more = [x for x in [f'{half_up(h, 1)} h' if h else '', f'{half_up(n/1e6, 1)} M samples' if n else ''] if x]
            more_zh = '，'.join(x for x in [f'{half_up(h, 1)} 小时' if h else '', f'{half_up(n/1e4):,} 万样本' if n else ''] if x)
            # ran to its limit: good news; stopped by hand (STOP file) before it: bad news. Results get a hand-written entry.
            halted = bool(d.get('halted'))
            out.append(dict(date=at(s['modified']), t=s['modified'], auto=True, tone='bad' if halted else 'good',
                            text=f"{en} {'was stopped early' if halted else 'ran to the end'}{': ' + ', '.join(more) if more else ''}.",
                            text_zh=f"{zh}{'提前停止' if halted else '跑完'}{'：' + more_zh if more_zh else ''}。"))
    return out


def logged(runs, now):
    """The automatic entries of the last AUTO_DAYS from every round but side runs, so the log moves between
    hand-written milestones and a round's night of new checkpoints stays in it after the next round starts
    (11-09: round 2's entries vanished when round 3 became current). Older rounds add nothing: their entries
    are older than AUTO_DAYS."""
    recent = now - AUTO_DAYS*86400
    return [e for r in runs if r.get('role') != 'side' for e in r.get('_log', []) if e['t'] >= recent]


def merge_log(timeline, auto):
    """The notes' timeline plus the automatic entries no hand-written one already covers, newest first."""
    stamp_of = lambda e: dt.datetime.strptime(e['date'], '%Y-%m-%d %H:%M').replace(tzinfo=CST).timestamp()
    hand = [stamp_of(e) for e in timeline]
    # nothing older than the oldest hand-written entry: a pruned log stays pruned (Mercy 11-12, outdated entries removed)
    first = min(hand) if hand else 0
    keep = [{k: v for k, v in e.items() if k != 't'} for e in auto if e['t'] >= first and all(abs(e['t'] - h) > AUTO_GAP_S for h in hand)]
    return sorted(timeline + keep, key=lambda e: e['date'], reverse=True)


def runs_from(remote, notes):
    out = []
    for spec in notes['runs']:
        r = remote['runs'].get(spec['id'])
        if r is None:
            print(f"warning: run {spec['id']} is not on baolong yet", file=sys.stderr)
            continue
        seeds = [seed_summary(k, v, r.get('p0') or {}, remote['now']) for k, v in sorted(r['seeds'].items())]
        if spec.get('stopped'):  # the notes know a run was stopped before the LIVE_S window has passed
            for x in seeds:
                x['state'] = 'done' if x['state'] == 'done' else 'stopped'
        out.append(dict(spec, seeds=seeds, started=stamp(r.get('started')), p0_time=stamp(r.get('p0_time')),
                        _log=auto_log(spec, r['seeds'])))
    return out


def pretty(iso, zh=False):
    """'2026-11-02T16:20+08:00' -> '2 Nov 16:20' / '11月2日 16:20'."""
    if not iso:
        return '—'
    d = dt.datetime.fromisoformat(iso[:16])
    return f'{d.month}月{d.day}日 {d:%H:%M}' if zh else f"{d.day} {d:%b} {d:%H:%M}"


def zh_cat(a, b):
    """Join Chinese text, with a space only after a Latin letter or digit ('递归 SAC 还' but 'C 轮还')."""
    return a + (' ' if a and a[-1].isascii() and a[-1].isalnum() else '') + b


def family(run_id):
    """Which task a run trains: 'ph' for ph-*, 'mixed' (four tanks + TEP) for mixed-*, 'suite' (several
    APC suite plants, rsac --plant a,b,c) for apc-*, 'sim' (the unified simulation core, aio.train) for sim-*,
    else the four tanks."""
    return ('ph' if run_id.startswith('ph-') else 'mixed' if run_id.startswith('mixed-') else 'suite' if run_id.startswith('apc-')
            else 'sim' if run_id.startswith('sim-') else 'tank')


FAMILIES = ('tank', 'ph', 'mixed', 'suite', 'sim')


def current(runs, fam='tank'):
    return next((r for r in runs if r.get('role') == 'current' and family(r['id']) == fam), None)


def short_name(run):
    return run.get('label', run['id']).split(' · ')[0]


def round_name(run, zh=False):
    """The round a replayed checkpoint comes from: its label up to the seed, at most two parts, so rounds
    that share a first part stay apart ('Fifteen plants · per-channel memory', not just 'Fifteen plants')."""
    label = (run.get('label_zh') if zh else None) or run.get('label') or run['id']
    parts = [x for x in label.split(' · ') if not re.match(r'(种子|seed)\s*\d', x, re.I)]
    return ' · '.join(parts[:2]) or label.split(' · ')[0]


def freshness(out, remote=None, reachable=True):
    """Where each block's data comes from, and whether it is behind the current run.

    Problems with level 'error' make publish_bilingual_site refuse to build (user request,
    2026-11-02: do not publish out-of-date content); 'stale' blocks are published with a
    visible label. Times are UTC+8 ISO minutes."""
    runs, notes_as_of = out.get('runs') or [], out['notes_as_of']
    cur = current(runs)
    blocks, problems = {}, []

    def problem(level, text, text_zh):
        problems.append(dict(level=level, text=text, text_zh=text_zh))
        return dict(text=text, text_zh=text_zh)

    # training curves
    t = dict(run=cur and cur['id'], through=out.get('training_through'))
    if not reachable:
        t['stale'] = problem('stale', f"Training machine unreachable at publish time; curves are as of {pretty(out.get('training_through'))}.",
                             f"发布时连不上训练机，曲线停在 {pretty(out.get('training_through'), True)}。")
    blocks['training'] = t

    # a run newer than the one the notes call current, per task; a run the notes name as a side run
    # (role 'side': a control experiment, not the task's current round) is known and does not count
    side = {r['id'] for r in runs if r.get('role') == 'side'}
    for c in (current(runs, f) for f in FAMILIES):
        if not (remote and c):
            continue
        mine = (remote['runs'].get(c['id']) or {}).get('started') or 0
        newer = sorted(k for k, v in remote['runs'].items()
                       if k != c['id'] and k not in side and family(k) == family(c['id']) and (v.get('started') or 0) > mine and v.get('seeds'))
        if newer:
            problem('error', f"{', '.join(newer)} started after {c['id']}, which the notes still call current. Update site_progress_notes.json.",
                    f"{'、'.join(newer)} 比笔记里的当前轮次 {c['id']} 更新，请先更新 site_progress_notes.json。")

    # a suite run's notes name every family it trains, each with references on its own validation set (site
    # rule 13); on 11-06 the fifteen-plant run had neither, and the page called it "five plants" with no table
    for c in runs:
        if c.get('task') != 'suite' or c.get('role') == 'side' or not c.get('seeds'):
            continue
        trained = set()
        for x in c['seeds']:
            cfg = x.get('config') or {}
            trained |= {q for q in str(cfg.get('plant') or '').split(',') if q} | ({'tep'} if cfg.get('tep_envs') else set())
            trained |= set(x.get('fam_val') or {})
        refs = c.get('fam_refs') or {}
        missing = sorted(q for q in trained if q not in (c.get('families') or []) or q not in refs)
        if missing:
            problem('error', f"{c['id']} trains {', '.join(missing)}, which its notes do not list in families and fam_refs. Update site_progress_notes.json.",
                    f"{c['id']} 训练了 {'、'.join(missing)}，但笔记的 families 和 fam_refs 里没有，请先更新 site_progress_notes.json。")

    # curated text must not predate the newest P0 report of any task's current run
    n = dict(as_of=notes_as_of)
    late = [c for c in (current(runs, f) for f in FAMILIES) if c and c.get('p0_time') and c['p0_time'][:16] > notes_as_of[:16]]
    live = [c for c in (current(runs, f) for f in FAMILIES) if c and any(x.get('state') == 'running' for x in c.get('seeds') or [])]
    if late:
        c = late[0]
        n['stale'] = problem('error', f"The insights were written at {pretty(notes_as_of)}, before {c['id']}'s P0 report ({pretty(c['p0_time'])}). Update site_progress_notes.json.",
                             f"结论写于 {pretty(notes_as_of, True)}，早于 {c['id']} 的 P0 报告（{pretty(c['p0_time'], True)}），请先更新 site_progress_notes.json。")
    elif out.get('training_through') and live:
        lag = (dt.datetime.fromisoformat(out['training_through'][:16]) - dt.datetime.fromisoformat(notes_as_of[:16])).total_seconds()/3600
        if lag > NOTES_MAX_LAG_H:
            n['stale'] = problem('stale', f"Written {pretty(notes_as_of)}; training has run {lag:.0f} h since.",
                                 f"写于 {pretty(notes_as_of, True)}，之后训练又跑了 {lag:.0f} 小时。")
    blocks['notes'] = n

    # a 'no RL model yet' notice must not outlive what it promises (user request, 2026-11-03): the
    # training start while pending, the learner joining the replay while training, a revisit while paused.
    # tep_rl is TEP's notice; suite_rl is shared by the APC suite plants (quadruple tank, Shell, BSM1), zoo_rl by the plant zoo,
    # expand_rl by the families added for the expansion round until a run trains them, heldout_rl by the held-out
    # families (never trained: 'heldout' until the learner's zero-shot run joins their replays), lit_rl by the literature columns.
    for key, en, zh in (('tep_rl', 'TEP', 'TEP'), ('suite_rl', 'new-plant', '新装置'), ('zoo_rl', 'new-environment', '新环境'),
                        ('expand_rl', 'expansion-family', '扩充族'), ('heldout_rl', 'held-out', '留出族'), ('lit_rl', 'literature-column', '文献精馏塔'),
                        ('mimo_rl', 'APC-scale-plant', 'APC 规模装置'), ('sim_rl', 'simulation-core', '新仿真核心')):
        t = out.get(key) or {}
        due = {'pending': ('expected_start', 'RL training starts', 'RL 训练'),
               'training': ('expected_replay', 'the learner joins the replay', '学习器加入回放'),
               'heldout': ('expected_replay', "the learner's zero-shot run joins the replay", '学习器零样本回放加入')}.get(t.get('state'))
        if due and t.get(due[0]) and out.get('generated_at'):
            late = (dt.datetime.fromisoformat(out['generated_at'][:16]) - dt.datetime.fromisoformat(t[due[0]][:16])).total_seconds()/3600
            if late > PENDING_GRACE_H:
                problem('error', f"The {en} notice says {due[1]} {pretty(t[due[0]])}, {late:.0f} h ago. Update {key} in site_progress_notes.json.",
                        f"{zh}提示写的是{due[2]}在 {pretty(t[due[0]], True)}，已过去 {late:.0f} 小时，请更新 site_progress_notes.json 的 {key}。")
        # paused with no resume time yet: the notice says when it paused and must not stand unrevised for long
        if t.get('state') == 'paused' and t.get('paused_since') and out.get('generated_at'):
            held = (dt.datetime.fromisoformat(out['generated_at'][:16]) - dt.datetime.fromisoformat(t['paused_since'][:16])).total_seconds()/3600
            if held > PAUSE_MAX_H:
                problem('error', f"The {en} notice says training paused {pretty(t['paused_since'])}, {held:.0f} h ago, with no resume time. Update {key} in site_progress_notes.json.",
                        f"{zh}提示写的是训练在 {pretty(t['paused_since'], True)} 暂停，已过去 {held:.0f} 小时，还没有恢复时间，请更新 site_progress_notes.json 的 {key}。")
    # every log entry opens with good or bad news (user 11-09: "每次日志都以好消息/坏消息开头，要观点明确")
    for e in out.get('timeline') or []:
        if e.get('tone') not in TONES:
            problem('error', f"log entry {e['date']} needs tone 'good' or 'bad' (the page opens it with Good news / Bad news)",
                    f"日志 {e['date']} 要写 tone（good 或 bad），页面据此以“好消息/坏消息”开头")
    return dict(blocks=blocks, problems=problems)


def build(host='baolong', previous=None, remote=None, now=None):
    notes = json.loads(NOTES.read_text(encoding='utf-8'))
    now = now or dt.datetime.now(CST)
    out = dict(generated_at=now.isoformat(timespec='minutes'), generated_label=now.strftime('%Y-%m-%d %H:%M (UTC+8)'),
               notes_as_of=notes['as_of'], status=notes['status'], insights=notes['insights'],
               insights_zh=notes['insights_zh'], timeline=notes['timeline'], p0_refs=notes.get('p0_refs'), tep_rl=notes.get('tep_rl'), suite_rl=notes.get('suite_rl'), zoo_rl=notes.get('zoo_rl'), expand_rl=notes.get('expand_rl'), heldout_rl=notes.get('heldout_rl'), lit_rl=notes.get('lit_rl'), mimo_rl=notes.get('mimo_rl'), sim_rl=notes.get('sim_rl'), tables=notes.get('tables'), curve=notes.get('curve'), train_labels=notes.get('train_labels'), exam_note=notes.get('exam_note'), replays=notes.get('replays'), model=notes.get('model'),
               superseded=notes.get('superseded'))
    try:
        remote = remote if remote is not None else fetch(host)
        out['runs'] = runs_from(remote, notes)
        out['timeline'] = merge_log(notes['timeline'], logged(out['runs'], now.timestamp()))
        for r in out['runs']:
            r.pop('_log', None)
        out['training_through'] = stamp(max(((s['modified'] or 0) for r in remote['runs'].values() for s in r['seeds'].values()),
                                            default=0) or None)   # no run there (training on Colab): reachable, nothing to read
        reachable = True
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError) as e:
        print(f'warning: could not read the runs on {host} ({e}); keeping the previous training data', file=sys.stderr)
        prev = previous or {}
        out['runs'], out['training_through'] = prev.get('runs', []), prev.get('training_through')
        remote, reachable = None, False
    out['freshness'] = freshness(out, remote, reachable)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--host', default='baolong')
    ap.add_argument('--previous', type=Path)
    a = ap.parse_args()
    prev = json.loads(a.previous.read_text(encoding='utf-8')) if a.previous and a.previous.exists() else None
    out = build(a.host, prev)
    for pr in out['freshness']['problems']:
        print(f"{pr['level']}: {pr['text']}", file=sys.stderr)
    a.out.write_text(json.dumps(out, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')

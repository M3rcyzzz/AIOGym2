"""The standard benchmark of the environment (docs/standard.md): fixed exam batches kept as files, their reference
scores, one scorecard, and the checks an environment of this standard passes.

    python -m aio.sim.bench --ctl MODULE:FACTORY [--arg WEIGHTS] [--exam usual,stock] [--out card.json]
    python -m aio.sim.bench --check                      (conformance: the checks of `conform`)
    python -m aio.sim.bench --make usual,stock           (the maintainer: the packs and reference tables written again)

A batch drawn from a seed is not the same plants on every machine (the draw's linear algebra rounds differently from
one build of the numerical library to another: aio.sim.stay.key). So the plants, tasks, schedules and noise seeds of a
benchmark batch are kept in a pack (`freeze`), and every machine reads the same ones (`thaw`), bit for bit; the
references' per-plant scores are kept beside it, computed once on the plants of the pack. A controller is read as
aio.exam reads a model: it lives its own stay on the plants, window after window, from the packet alone.

A controller: reset(prog), __call__(prog, st, k, obs) -> moves (n, NMV) in [-1, 1] of each MV's slew limit; it may
read `obs` (the packet, aio.interface) and what `prog` publishes of its size (n, nmv, ncv, device, dtype), nothing
else. FACTORY(arg, device=) builds it (aio.sim.plugin).
"""
import argparse
import io
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from aio import scale as SC
from aio.sim import control as CT, core as C, plugin as PG, prior as PR, stay as ST

VERSION = 'H4'                                   # the benchmark: its batches, references and scorecard. Anything that changes a number of a card is another version.
DATA = Path(__file__).resolve().parent/'bench_data'
SUITE = dict(                                    # exam (aio.sim.stay.EXAMS): windows, and the key of its batch (stay.key: the plants, tasks, schedules, the references' version)
    usual=dict(windows=16, key='7112cb3fcc1afd29'),      # the generator as in training, 64 units of 12x5
    stock=dict(windows=8, key='1cd585e1035b3d0c'),       # every unit with a conserving train of vessels
    standard=dict(windows=16, key='88c268fc117a0251'),   # the generator of the standard (prior.STANDARD): a quarter of the units with such a train
    usual24=dict(windows=8, key='a03cef5e7bad038b'),     # 64 units of 24x10, shaped as the controllers of industrial APC are (references of 11-13: no cost, no trips)
)
SUITE.update(                                    # made on a GPU machine (`--make NAME --device cuda --dir DIR`; 11-16, torch 2.14.1+cu130)
    usual48=dict(windows=8, n=32, key='1ffb5fec1947bbad', away=True),     # 32 units of 48x20; its pack is 73 MB: not in the repository, read from $AIO_BENCH_DATA
    mech=dict(windows=8, key='0d2615745b137a1d', built=True),             # plants of the engine, the ten of the first pool: no pack yet (an engine does not freeze), built from the seed;
                                                                          # the references are of the batch a Colab G4 draws (another machine draws another: here 0d0a787c42472d2c) and read there alone
    mech_new=dict(windows=8, key='05f96345d9da52da', built=True),         # and the five since
)
PENDING = {}                                     # batches whose references are yet to be computed
HERE = tuple(q for q, v in SUITE.items() if not v.get('away') and not v.get('built'))      # the batches a card is of by default: their packs are in the repository
BASE = ('hold', 'pi', 'dmc', 'teacher')          # the scale: 0 hold, 100 the best of these per plant (as aio.exam)
OWN = ('dmc_meter', 'teacher_meter')             # references that read the instruments alone, each on its own stay
PUBLIC = ('y', 'prev_y', 'target', 'last_target', 'bad', 'age', 'u', 'prev_u')      # all of the state a packet is made of (`conform`)

_INT = ('nl', 's0', 'nseg', 'n', 'seed', 'ncv', 'nmv', 'nd', 'nq', 'ndq', 'nu', 'nzmax', 'nr', 'nnorm', 'nuni', 'size', 'priv')
_TENS = ('P', 'T', 'hold_k', 'seg_k', 'kinds', 'ramp', 'k_first', 'settled_k')
_DICT = ('c', 'start', 'pi')


def freeze(progs, path, **meta):
    """The windows of a stay (linear plants) as one file: what `step`, `observe` and the score read, the noise seed,
    and the formula PI's tuning. Not the teacher's model (thirty times the size: its scores are in the reference
    table) and not the generator's tail (the windows are all here)."""
    assert all(p.mech is None for p in progs), 'plants of the engine are built from their seed (the same on every machine)'
    arr, head = {}, dict(bench=VERSION, key=ST.key(progs), windows=len(progs), refs=ST.REFS_VERSION, **meta, w=[])
    for w, p in enumerate(progs):
        head['w'].append(dict({q: int(getattr(p, q)) for q in _INT}, dims=list(p.dims), blind=bool(p.blind), lay={q: [a, b, list(s)] for q, (a, b, s) in p.lay.items()},
                              tlay={q: [a, b, list(s)] for q, (a, b, s) in p.tlay.items()}))
        for q in _TENS:
            arr[f'{w}/{q}'] = getattr(p, q).cpu().numpy()
        for d in _DICT:
            for q, v in (getattr(p, d, None) or {}).items():
                arr[f'{w}/{d}/{q}'] = v.cpu().numpy()
    arr['head'] = np.frombuffer(json.dumps(head).encode(), dtype=np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arr)
    tmp = path.with_suffix('.tmp')
    tmp.write_bytes(buf.getvalue())
    tmp.replace(path)
    return head['key']


def thaw(path, device='cpu'):
    """The windows of a pack, as Programs without a teacher: (list, head). The same plants on every machine."""
    z = np.load(path)
    head = json.loads(bytes(z['head']).decode())
    dev, progs = torch.device(device), []
    for w, h in enumerate(head['w']):
        p = C.Program.__new__(C.Program)
        for q in _INT:
            setattr(p, q, h[q])
        p.dims, p.blind, p.device, p.dtype = tuple(h['dims']), h['blind'], dev, torch.float32
        p.lay = {q: (a, b, tuple(s)) for q, (a, b, s) in h['lay'].items()}
        p.tlay = {q: (a, b, tuple(s)) for q, (a, b, s) in h['tlay'].items()}
        for q in _TENS:
            setattr(p, q, torch.from_numpy(z[f'{w}/{q}']).to(dev))
        for d in _DICT:
            setattr(p, d, {f.split('/', 2)[2]: torch.from_numpy(z[f]).to(dev) for f in z.files if f.startswith(f'{w}/{d}/')})
        p.mech, p.mg, p.menu, p.was, p.tail, p.Fr, p.meter, p.gen = None, [], None, None, None, None, None, torch.device('cpu')
        p.ar = torch.arange(p.n, device=dev)
        p.g = torch.Generator(); p.g.manual_seed(p.seed)
        progs.append(p)
    assert ST.key(progs) == head['key'], f"{path}: the pack does not hold the batch it names"
    return progs, head


def pack(name, device='cpu'):
    """The windows of the benchmark exam `name` and its reference table."""
    refs = json.loads((DATA/VERSION/f'{name}.refs.json').read_text())
    if SUITE[name].get('built'):                                   # plants of the engine: from the seed
        progs = list(ST.exam(name, refs.get('n', ST.EXAM_N), ST.EXAM_SEED, SUITE[name]['windows'], teacher=False, device=device, gen_device='cpu'))
        head = dict(key=ST.key(progs))
    else:
        where = [DATA/VERSION/f'{name}.npz']+([Path(os.environ['AIO_BENCH_DATA'])/f'{name}.npz'] if os.environ.get('AIO_BENCH_DATA') else [])
        found = [f for f in where if f.exists()]
        if not found:
            raise FileNotFoundError(f'{name}: its pack is not in the repository; set AIO_BENCH_DATA to the folder that holds {name}.npz')
        progs, head = thaw(found[0], device)
    assert head['key'] == refs['key'] == SUITE[name]['key'], f'{name}: pack {head["key"]}, references {refs["key"]}, suite {SUITE[name]["key"]}'
    return progs, refs


@torch.no_grad()
def stay(progs, ctl, say=None):
    """A controller's own stay on the windows: per window and plant the score (row-decisions outside), whether the unit
    failed physically, and whether it went past a shutdown limit its plant declares."""
    st, out, t0 = None, dict(score=[], failed=[], tripped=[], cost=[]), time.time()
    for w, prog in enumerate(progs):
        r = CT.rollout(prog, ctl, st=st)
        st = r['st']
        for q in out:
            out[q].append(r[q].float().cpu().tolist())
        if say is not None:
            say(w, float(r['score'].mean()), time.time()-t0)
    return out


def _commit():
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=Path(__file__).parent, capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def card(name, ctl=None, device='cpu', say=None, run=None):
    """The scorecard of a controller on the benchmark exam `name` (ctl None: the references alone; run: the output of
    `stay` if it was made elsewhere). Per part of the batch (all, the plants with a level, the rest) and controller:
    points (0 the MVs left where they are, 100 the best reference per plant; by the row-decisions outside), cost_points
    and cost_worse (the same scale and the share of plants left worse than untouched, by the cost of aio.sim.core.step:
    how far the rows were out, not how often), worse (share of the plants left worse
    than untouched), low (points on its worst tenth of the plants), worst, outside (share of the row-decisions outside),
    failed and tripped (plant-windows that failed physically, that went past a shutdown limit)."""
    progs, refs = pack(name, device)
    r0 = ST.EXAMS[name].get('read', 0)
    rows = torch.tensor(refs['rows'])[r0:].sum(0)
    has = 'cost' in refs                                           # (references computed before the cost was kept: points by the count alone)
    zero = [[0.]*len(refs['rows'][0])]*len(refs['rows'])
    runs = {q: dict(score=refs['score'][q], failed=refs['failed'][q], tripped=refs.get('tripped', {}).get(q, zero), cost=refs['cost'][q] if has else zero) for q in refs['score'] if q in BASE+OWN}
    if ctl is not None or run is not None:
        runs['controller'] = run or stay(progs, ctl, say)
    per = {q: torch.tensor(v['score'])[r0:].sum(0).tolist() for q, v in runs.items()}
    best = SC.envelope({q: per[q] for q in BASE})
    cst = {q: torch.tensor(v['cost'])[r0:].sum(0).tolist() for q, v in runs.items()}       # the same scale on the cost itself: how far, not how often
    cbest = SC.envelope({q: cst[q] for q in BASE}) if has else None
    lev = ((progs[0].c['integ'] > 0).sum(1) > 0).tolist()
    table = {}
    for part, m in (('all', [True]*len(lev)), ('levels', lev), ('rest', [not x for x in lev])):
        if not any(m):
            continue
        pick = lambda v: [x for x, k in zip(v, m) if k]           # noqa: E731
        mt = torch.tensor(m)
        table[part] = dict(plants=sum(m))
        for q, v in per.items():
            if q == 'hold':
                continue
            table[part][q] = dict(points=round(SC.points(pick(v), pick(per['hold']), pick(best)), 2),
                                  cost_points=round(SC.points(pick(cst[q]), pick(cst['hold']), pick(cbest)), 2) if has else None,
                                  cost_worse=round(sum(x > y for x, y in zip(pick(cst[q]), pick(cst['hold'])))/sum(m), 4) if has else None,
                                  **{k: round(x, 4) for k, x in SC.tail(pick(v), pick(per['hold']), pick(best)).items()},
                                  outside=round(float(sum(pick(v))/rows[mt].sum().clamp_min(1)), 4),
                                  failed=int(torch.tensor(runs[q]['failed'])[r0:, mt].sum()), tripped=int(torch.tensor(runs[q]['tripped'])[r0:, mt].sum()))
    return dict(bench=VERSION, exam=name, key=refs['key'], refs_version=refs['refs'], plants=len(lev), windows=len(progs), read_from=r0, decisions=PR.HORIZON,
                table=table, per_plant={q: per[q] for q in ('hold', 'controller') if q in per}, torch=torch.__version__, commit=_commit())


def make(name, path=None, say=None, device='cpu'):
    """The pack and the reference table of an exam, from its seed on this machine (the maintainer's; the key has to be
    the suite's, or it is another batch: aio.sim.stay.key)."""
    path = Path(path or DATA/VERSION)
    s = SUITE.get(name) or PENDING[name]
    n = s.get('n', ST.EXAM_N)
    progs = list(ST.exam(name, n, ST.EXAM_SEED, s['windows'], device=device, gen_device='cpu'))
    key = ST.key(progs)
    assert s.get('key') in (None, key), f'{name}: this machine draws {key}, the suite is {SUITE[name]["key"]} (pip torch 2.15.0 draws it)'
    out = ST.references(n, 0, len(progs), BASE, progs=progs, extra={q: ST.REFS[q]() for q in OWN}, say=say)
    refs = dict(bench=VERSION, exam=name, key=key, n=n, refs=ST.REFS_VERSION, torch=torch.__version__, commit=_commit(), rows=out['rows'].tolist(),
                **{k: {q: v.float().tolist() for q, v in out[k].items()} for k in ('score', 'failed', 'tripped', 'cost')})
    path.mkdir(parents=True, exist_ok=True)
    if progs[0].mech is None:
        freeze(progs, path/f'{name}.npz', exam=name)
    tmp = path/f'{name}.refs.tmp'
    tmp.write_text(json.dumps(refs))
    tmp.replace(path/f'{name}.refs.json')
    return key


# ---------------------------------------------------------------- conformance
@torch.no_grad()
def conform(steps=120):
    """The checks of the standard, as (name, passed, detail): the packs hold the suite's batches; a rollout is the same
    twice; the packet is made of the instruments' readings, the targets and the commands alone; a move beyond the slew
    limit is the move at the limit; the packet is finite; a stream of a conserving train takes from one vessel what
    it gives the next."""
    res = []
    for name, s in SUITE.items():
        if s.get('built') or (s.get('away') and not os.environ.get('AIO_BENCH_DATA')):     # (built from the seed: minutes; a pack that is elsewhere)
            continue
        try:
            progs, refs = pack(name)
            ok, why = len(progs) == s['windows'], f"key {refs['key']}, {len(progs)} windows"
        except (AssertionError, FileNotFoundError) as e:
            ok, why = False, str(e)
        res.append((f'pack {name}', ok, why))
    prog = pack('usual')[0][0]

    def run(amp, seed=1):
        g = torch.Generator().manual_seed(seed)
        st, obs, sc = C.reset(prog), [], torch.zeros(prog.n)
        for k in range(steps):
            obs.append(C.observe(prog, st, k))
            a = (2*torch.rand(prog.n, prog.nmv, generator=g)-1)
            st, out = C.step(prog, st, a*amp if amp <= 1 else torch.sign(a)*amp, k)
            sc += out['outside'].sum(-1)
        return torch.stack(obs), sc, st

    o1, s1, st = run(1.)
    o2, s2, _ = run(1.)
    res.append(('deterministic', bool((o1 == o2).all() and (s1 == s2).all()), f'{steps} decisions twice, {prog.n} plants'))
    res.append(('packet finite', bool(torch.isfinite(o1).all()), f'largest entry {float(o1.abs().max()):.1f}'))
    # the packet reads nothing of the state but PUBLIC: every other entry scrambled, the packet is the same
    g = torch.Generator().manual_seed(2)
    scr = {q: (v if q in PUBLIC else (torch.rand(v.shape, generator=g) > .5) if v.dtype == torch.bool else v+torch.randn(v.shape, generator=g).to(v.dtype)) for q, v in st.items()}
    same = bool((C.observe(prog, st, steps) == C.observe(prog, scr, steps)).all())
    res.append(('no privileged state in the packet', same, f'{len(st)-len(PUBLIC)} entries of the state scrambled, the packet unchanged'))
    oa, _, sa = run(1.0001)                                        # full moves, as signs
    ob, _, sb = run(7.)
    gap = float((sa['u']-sb['u']).abs().max())                     # (the straight-through gradient adds and takes away the move asked for: rounding)
    res.append(('moves beyond the limit are the limit', gap < 1e-5, f'moves of 7 slew limits act as moves of 1 (valve positions within {gap:.0e} of the span)'))
    # conservation on the stock exam: with nothing but one valve moved, at most two levels move and what one gains the other loses (in the vessels' own holdups)
    p = C.build(48, 9, teacher=False, quiet=True, shape=PR.STOCK, menu=PR.Menu(rate=(0., 0., 0., 0., 0.)))
    c, st = p.c, C.reset(p)
    lev, v0 = c['integ'] > 0, st['v'].clone()
    a = torch.zeros(p.n, p.nmv, dtype=p.dtype); a[:, 0] = 1.
    for k in range(40):
        st, _ = C.step(p, st, a*(k < 6), k)
    dv = (st['v']-v0)*c['tint']*lev
    moved = dv.abs() > 1e-6
    two = moved.sum(1) == 2
    ok = bool((moved.sum(1) <= 2).all() and two.any() and ((dv > 1e-6).sum(1)[two] == 1).all())
    res.append(('a stream takes from one vessel what it gives the next', ok, f'{int(two.sum())} of {p.n} trains have the valve between two vessels'))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ctl', default='', help='MODULE:FACTORY of the controller (aio.sim.plugin); none: the references alone')
    ap.add_argument('--arg', default=None, help='what FACTORY is given (a weights file)')
    ap.add_argument('--exam', default=','.join(HERE), help='of '+', '.join(SUITE))
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--out', default='')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--make', default='')
    ap.add_argument('--dir', default='', help='--make: where the pack and the reference table go (default: the repository)')
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.check:
        res = conform()
        for name, ok, why in res:
            print(f"{'PASS' if ok else 'FAIL'}  {name}: {why}")
        raise SystemExit(0 if all(ok for _, ok, _ in res) else 1)
    if a.make:
        for name in a.make.split(','):
            print(name, make(name, a.dir or None, device=a.device, say=lambda w, s, t: print(f'window {w}: '+'  '.join(f'{q} {v:.0f}' for q, v in s.items())+f'  ({t:.0f} s)', flush=True)))
        return
    cards = []
    for name in a.exam.split(','):
        ctl = PG.build(a.ctl, a.arg, device=a.device)[0] if a.ctl else None
        cd = card(name, ctl, a.device, say=lambda w, s, t: print(f'{name} window {w}: {s:.0f}  ({t:.0f} s)', flush=True))
        cards.append(cd)
        print(f"{VERSION} {name}  key {cd['key']}  {cd['plants']} plants x {cd['windows']} windows   points / by cost / worse than untouched / worst tenth / failed / tripped")
        for part, t in cd['table'].items():
            print(f"  {part:7s}{t['plants']:4d}  "+'   '.join(f"{q} {v['points']:.1f} / {v['cost_points'] if v['cost_points'] is not None else float('nan'):.1f} / {v['worse']:.0%} / {v['low']:.0f} / {v['failed']} / {v['tripped']}" for q, v in t.items() if q != 'plants'))
    if a.out:
        Path(a.out).write_text(json.dumps(dict(bench=VERSION, controller=a.ctl, arg=a.arg, cards=cards)))


if __name__ == '__main__':
    main()

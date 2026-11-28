"""Faults and a start-up: the exams of what a unit does when something stays wrong (`exam`), and the same faults in training (`train_draw`, aio.train --faults).

  python -m aio.sim.faults --plants usual,tep,column --kinds bad,freeze,drift,stick,startup --out faults.json [--ctl MODULE:FACTORY --arg WEIGHTS]

Every episode of a batch gets one fault of the kind, on one row or valve and for a stretch the generator's own
events never last (FAULT_LEN decisions; its bad-quality readings last 5 to 31):

  bad      a tracked process CV's instrument goes bad-quality: its last value is held and the packet says so
  freeze   the same reading stuck at its last value with no quality flag (a frozen transmitter)
  drift    the reading walks away from the true value, linearly, DRIFT_BANDS bands by the end of the fault, no flag
  stick    a valve stops moving whatever it is told (its friction is never overcome); the command still moves
  startup  no fault: window 0 drives every MV to its low limit and holds it there (a unit on standby or turned
           down), window 1 (no events) starts the controller from there; scored from its first decision

The scale is famexam's: per batch, 100 * (hold - controller) / (hold - teacher) of the decisions outside, on true
values (none when the teacher takes back less than REACH of hold's: BSM1 does not start up within a window);
with every controller the episodes that went past a shutdown limit the plant declares (`trips`, Tennessee
Eastman's interlocks). `key`: the batch's (stay.key of its windows, with the faults drawn on it).
'usual' is the generator as trained (12x5, --n-usual episodes); a plant name is that plant alone (--n episodes),
at the smallest size that holds it.
"""
import argparse, contextlib, hashlib, json, time
from pathlib import Path
import numpy as np
import torch

from aio.sim import core as C, control as CT, mech as ME, plants as PL, plugin as PG, prior as PR, stay as ST

KINDS = ('bad', 'freeze', 'drift', 'stick', 'startup')
FAULT_LEN = (40, 200)            # decisions a fault lasts (uniform)
DRIFT_BANDS = (3., 8.)           # how far a drifting reading is off by the end of its fault, bands
SEED = 4_000_000
REACH = .2                       # no points when the teacher takes back less than this share of the decisions outside of the MVs left alone


def draw(prog, kind, seed):
    """One fault per episode: (n,) row or MV index, first decision, length, and a drift's end in value units."""
    g = np.random.default_rng([seed, KINDS.index(kind)])
    c, n = prog.c, prog.n
    if kind == 'stick':
        ok = (c['mvm'] > 0).cpu().numpy()
    else:
        ok = ((c['trk'] > 0) & (c['rst'] <= 0) & (c['rowm'] > 0) & (c['dvr'] <= 0)).cpu().numpy()
    idx = np.array([g.choice(np.flatnonzero(r)) if r.any() else -1 for r in ok])
    dur = g.integers(FAULT_LEN[0], FAULT_LEN[1]+1, n)
    k1 = prog.k_first.clamp_max(PR.HORIZON).cpu().numpy()
    lo = np.maximum(k1, 40)
    k0 = np.array([g.integers(a, max(a+1, PR.HORIZON-d-60)) for a, d in zip(lo, dur)])
    band = c['bandf'].cpu().numpy()[np.arange(n), np.maximum(idx, 0)]
    off = g.uniform(*DRIFT_BANDS, n)*np.where(g.random(n) < .5, -1., 1.)*band
    t = lambda a, d=torch.long: torch.as_tensor(a, dtype=d, device=prog.device)     # noqa: E731
    return dict(kind=kind, idx=t(idx), k0=t(k0), dur=t(dur), off=t(off, prog.dtype))


TRAIN_KINDS = ('bad', 'freeze', 'drift', 'stick')               # (their order is aio.sim.core.step's)


def train_draw(prog, share, seed):
    """Faults in training (aio.train --faults): this share of the episodes gets one, of a kind drawn evenly among
    TRAIN_KINDS, on a tracked process row or a valve in service, for FAULT_LEN decisions after the first event; put on
    the program as `fault`, which aio.sim.core.step reads (no wrapping: the gradient goes through it). None: no episode."""
    g = np.random.default_rng([int(seed) & 0xffffffff, 23])
    c, n, dev = prog.c, prog.n, prog.device
    kind = g.integers(0, len(TRAIN_KINDS), n)
    okm = (c['mvm'] > 0).cpu().numpy()
    okr = ((c['trk'] > 0) & (c['rst'] <= 0) & (c['rowm'] > 0) & (c['dvr'] <= 0)).cpu().numpy()
    has, mv, row = g.random(n) < share, np.zeros(okm.shape, bool), np.zeros(okr.shape, bool)
    for i in range(n):
        ok = okm[i] if kind[i] == 3 else okr[i]
        if not ok.any():
            has[i] = False
            continue
        j = g.choice(np.flatnonzero(ok))
        (mv if kind[i] == 3 else row)[i, j] = True
    if not has.any():
        return None
    dur = g.integers(FAULT_LEN[0], FAULT_LEN[1]+1, n)
    lo = np.maximum(prog.k_first.clamp_max(PR.HORIZON).cpu().numpy(), 40)
    k0 = np.array([g.integers(a, max(a+1, PR.HORIZON-d-60)) for a, d in zip(lo, dur)])
    band = (c['bandf'].cpu().numpy()*row).sum(1)
    off = g.uniform(*DRIFT_BANDS, n)*np.where(g.random(n) < .5, -1., 1.)*band
    t = lambda a, d=torch.long: torch.as_tensor(a, dtype=d, device=dev)     # noqa: E731
    return dict(has=t(has, torch.bool), kind=t(kind), mv=t(mv, torch.bool), row=t(row, torch.bool), k0=t(k0), dur=t(dur), off=t(off, prog.dtype))


@contextlib.contextmanager
def applied(prog, f):
    """The core's step and the program's parameters with the fault in (for `control.rollout`)."""
    step, params = C.step, prog.params
    ar, idx, k0, dur, kind = prog.ar, f['idx'], f['k0'], f['dur'], f['kind']
    has = idx >= 0
    j = idx.clamp_min(0)
    held = {}

    def on(k):
        return has & (k >= k0) & (k < k0+dur)

    def faulty_params(seg, k=None):
        P = params(seg, k)
        if kind == 'stick' and k is not None:
            P = dict(P)
            m = on(k)
            fs = P['fs'].clone()
            fs[ar[m], j[m]] = 2.                                  # beyond any move: the valve stays where it is
            P['fs'] = fs
        return P

    def faulty_step(prog_, st, a, k, *args, **kw):
        if kind == 'bad':                                         # the core holds the last value and flags it
            st = dict(st)
            start = has & (k == k0)
            if bool(start.any()):
                b = st['bad'].clone()
                b[ar[start], j[start]] = dur[start].to(b.dtype)
                st['bad'] = b
        st1, out = step(prog_, st, a, k, *args, **kw)
        if kind in ('freeze', 'drift'):
            m = on(k)
            if bool(m.any()):
                y, last = st1['y'].clone(), st1['last'].clone()
                e = ar[m]
                if kind == 'freeze':
                    first = m & (k == k0)
                    if bool(first.any()):
                        for i in ar[first].tolist():
                            held[i] = st['y'][i, j[i]].detach().clone()
                    v = torch.stack([held[i] for i in e.tolist()])
                else:
                    v = y[e, j[m]]+f['off'][m]*((k+1-k0[m]).to(y.dtype)/dur[m].to(y.dtype))
                y[e, j[m]] = v
                last[e, j[m]] = v
                st1 = dict(st1, y=y, last=last)
        return st1, out

    C.step, prog.params = faulty_step, faulty_params
    try:
        yield
    finally:
        C.step = step
        del prog.params


class Shutdown:
    """Every MV to its low limit at its slew rate, and held there."""

    def reset(self, prog):
        pass

    def __call__(self, prog, st, k, obs):
        return -torch.ones(prog.n, prog.nmv, dtype=prog.dtype, device=prog.device)*prog.c['mvm']


def _build(plant, n, seed, windows=1, menu=None):
    kw = {}
    if plant != 'usual':
        plants = ME.pool([plant], heldout=plant in PL.HELDOUT)
        nmv = sum(w['kind'] == 'mv' for w in plants[0].ws)
        kw = dict(mech=1., plants=plants, dims=next(k for k, d in PR.SIZES.items() if d[1] >= nmv))
    return list(ST.windows(n, seed, windows, menu, **kw))


def _key(progs, f=None):
    m = hashlib.sha1(ST.key(progs).encode())
    if f is not None:
        m.update(f['kind'].encode())
        for q in ('idx', 'k0', 'dur', 'off'):
            m.update(f[q].detach().cpu().numpy().tobytes())
    return m.hexdigest()[:16]


def exam(plant, kind, ctls, n, seed=SEED):
    """One batch: per controller the episodes' scores and trips; the batch key; for a start-up, the episodes that
    started from standby (a unit that failed, or went past a shutdown limit, on the way down is left out)."""
    if kind == 'startup':
        progs = _build(plant, n, seed, 2, [None, ST.QUIET])
        r0 = CT.rollout(progs[0], Shutdown())
        use = ~r0['failed'] & ~r0['tripped']
        res = {}
        for q, c in ctls.items():
            c.reset(progs[1])
            res[q] = CT.rollout(progs[1], c, st=r0['st'])
        key, extra = _key(progs), dict(from_standby=int(use.sum()))
    else:
        progs = _build(plant, n, seed)
        f = draw(progs[0], kind, seed)
        res = {}
        with applied(progs[0], f):
            for q, c in ctls.items():
                res[q] = CT.rollout(progs[0], c)
        use = f['idx'] >= 0
        key, extra = _key(progs, f), dict(faulted=int(use.sum()))
    per = {q: [float(x) if u else None for x, u in zip(r['score'], use)] for q, r in res.items()}
    tot = {q: sum(x for x in v if x is not None) for q, v in per.items()}
    hold, best = tot['hold'], tot['teacher']
    pts = {q: round(100*(hold-v)/(hold-best), 1) if hold-best > REACH*hold else None for q, v in tot.items()}
    trips = {q: dict(episodes=int((r['tripped'] & use).sum()), first=[int(x) if (x < CT.BIG and u) else None for x, u in zip(r['trip_k'], use)])
             for q, r in res.items()}
    return dict(key=key, n=n, seed=seed, points=pts, per=per, trips=trips, **extra)


def _parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plants', default='usual')
    ap.add_argument('--kinds', default=','.join(KINDS))
    ap.add_argument('--ctl', default='', help='MODULE:FACTORY of an external controller or {name: controller} dict (aio.sim.plugin)')
    ap.add_argument('--arg', default=None, help='argument passed to FACTORY (e.g. a weights path)')
    ap.add_argument('--name', default=None, help='controller name when FACTORY returns one controller (defaults to meta["name"] or "aio")')
    ap.add_argument('--ckpt', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--n-usual', type=int, default=64)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--out', type=Path, required=True)
    return ap


def _load_ctl(ap, a, dev='cpu'):
    if a.ckpt and not a.ctl:
        ap.error('--ckpt has been removed; pass --ctl MODULE:FACTORY --arg PATH instead (aio.sim.plugin)')
    if a.ckpt:
        ap.error('--ckpt has been removed; pass --arg PATH with --ctl MODULE:FACTORY instead')
    c = dict(hold=CT.Hold(), pi=CT.PI(), dmc=CT.Teacher('fixed', meter=True), teacher=CT.Teacher('late'))
    if a.ctl:
        built, meta = PG.build(a.ctl, a.arg, device=dev)
        if isinstance(built, dict):
            c.update(built)
        else:
            cname = a.name or (meta.get('name') if isinstance(meta, dict) else None) or 'aio'
            c[cname] = built
    return c


def main(argv=None):
    ap = _parser()
    a = ap.parse_args(argv)
    if a.ckpt:
        _load_ctl(ap, a)
    out = json.loads(a.out.read_text()) if a.out.exists() else {}
    for p in a.plants.split(','):
        for kind in a.kinds.split(','):
            name = f'{p}/{kind}'
            if name in out:
                continue
            t0 = time.time()
            out[name] = exam(p, kind, _load_ctl(ap, a), a.n_usual if p == 'usual' else a.n, a.seed)
            out[name]['seconds'] = round(time.time()-t0)
            a.out.write_text(json.dumps(out, indent=1))
            print(name, out[name]['key'], out[name]['points'], 'tripped', {q: v['episodes'] for q, v in out[name]['trips'].items()}, out[name]['seconds'], 's', flush=True)


if __name__ == '__main__':
    main()

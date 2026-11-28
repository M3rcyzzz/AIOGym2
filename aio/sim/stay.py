"""A long stay on the same plants: windows of HORIZON decisions one after another (Program.next), and the
reference controllers along them, window by window (docs/sim.md, 'long stay').

    python -m aio.sim.stay --n 64 --seed 7 --windows 8 [--mech 1 --plants qtank,hex] [--rate 4,1,1,1,1 --pace 2 --nseg 40]
                           [--quiet 2] [--ride .8 --exc .3] [--heavy 1] [--shift .15] [--ctl hold,pi,dmc,teacher] [--out stay.json]
    python -m aio.sim.stay --exam pulse          (the fixed exams: EXAMS)

The teacher lives on the plants; every other controller takes each window over from the state the teacher left
(as aio.train.Stay does for a network): the scale of a window is then that window's own, 0 the MVs left where
they stood at its start, 100 the best of the controllers run, episode by episode (aio.scale). With --own every
controller runs its own stay instead, and the 0 is the MVs never moved since the start: those plants drift
further off window after window, the scale stretches and every controller's points rise by themselves.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from aio import scale as SC
from aio.sim import control as CT, core as C, engine as EN, mech as ME, plants as PL, prior as PR

REFS = dict(hold=CT.Hold, pi=CT.PI, dmc=lambda: CT.Teacher('fixed'), teacher=lambda: CT.Teacher('late'), now=lambda: CT.Teacher('now'),
            dmc_meter=lambda: CT.Teacher('fixed', meter=True), teacher_meter=lambda: CT.Teacher('late', loads=False, meter=True))    # the same from the instruments alone (docs/sim.md, section 26)
NAMES = dict(hold='MVs left where they are', pi='formula PI', dmc='step-test DMC (the model of commissioning, never renewed)',
             teacher='teacher (model renewed 10-40 decisions after a process event)', now='exact-model DMC (renewed at the event)',
             dmc_meter='step-test DMC, its prediction corrected from the instruments alone',
             teacher_meter='teacher from the instruments alone, the unmeasured loads not fed forward (what training imitates)')


# ---- the exams of a long stay: fixed batches every validation reads the same way (the same plants in all the
# linear ones, by the seed). The first five are of kinds of change left out of training: `pulse` and plants of
# the engine never occur there, dead-time changes are dropped from it (Menu(drop=('dead',))), and nothing in it
# is as quiet as `quiet`.
EXAM_SEED, EXAM_N, EXAM_WINDOWS = 77_001, 64, 8
QUIET = PR.Menu(rate=(0., 0., 0., 0., 0.))
STILL = PR.Menu(rate=(1., 1., 1., 0., 1.))                       # the usual events, the process as it is
EXAMS = dict(
    pulse=dict(menu=PR.Menu(pulse=.5)),                           # half of the disturbances and process changes come and go: the next event takes them back
    dead=dict(menu=PR.Menu(rate=(1., 1., 1., 3., 1.), proc=2., drop=tuple(q for q in PR.PROCESS if q != 'dead'))),    # the process changes are dead times alone, three times as often
    quiet=dict(menu=lambda w: None if w < 2 else QUIET, read=2),  # two usual windows, then no event at all (the wander and the noise go on): what a controller does when nothing asks; read from window 2 on
    mech=dict(mech=1., plants=PL.FIRST),                          # plants of the engine: the ten of the pool as it was until 11-12 (the same batch as before the pool grew)
    mech_new=dict(mech=1., plants=PL.NEW),                        # and the five in the pool since (building heating, compressor, reactor with column and recycle, fixed bed, fired heater)
    usual=dict(),                                                 # the generator as in training: what the others are read against
    blind=dict(blind=True),                                       # `usual` with the measured disturbances left out of the packet
    usual24=dict(dims='24x10'),                                   # `usual` on the larger sizes, shaped as the controllers of industrial APC are (docs/sim.md, 'sizes'): more zone CVs than MVs, most MVs
    usual48=dict(dims='48x20'),                                   # resting on economic targets, the unit at its constraints much of the time, some MVs not the controller's
    ride=dict(menu=PR.Menu(ride=.8, exc=.3)),                     # the units stay at constrained points while ordinary events go on (a third of the time there)
    heavy=dict(menu=PR.Menu(heavy=1.)),                           # every process change is a heavy one: the gains turn about the operating point, the time constants are others (no dead time moves)
    shift=dict(menu=lambda w: None if w < 2 else STILL._replace(shift=1.) if w == 2 else STILL, read=2),    # two usual windows; at its first event of window 2 every unit becomes another one (a heavy change
                                                                  # that stays), and the process does not change again: how the windows after it go; read from window 2 on
    stock=dict(shape=PR.STOCK),                                   # every plant has a conserving train of vessels: levels that add up the flows between them, most of them buffers (docs/sim.md, 'stock')
    standard=dict(shape=PR.STANDARD),                             # the generator of the standard: `usual` with a conserving train in a quarter of the plants (docs/standard.md)
    sparse=dict(menu=PR.Menu(pace=.5)),                           # events half as often
    dense=dict(menu=PR.Menu(pace=2.), nseg=24),                   # and twice as often
)


def exam(name, n=EXAM_N, seed=EXAM_SEED, count=EXAM_WINDOWS, nseg=None, **kw):
    """The windows of the exam `name` (EXAMS), one at a time; kw: where to put them (device, gen_device, teacher).
    EXAMS[name].get('read', 0) is the first window the exam is read in (the ones before only lead up to it).
    nseg: segments a window has room for, where the exam does not ask for more (None: prior.NSEG, the batches as
    they are; prior.NSEG_STAY: a window that carries on has room for its events to the end, another batch)."""
    e = {q: v for q, v in EXAMS[name].items() if q != 'read'}
    if 'plants' in e:
        e['plants'] = ME.pool(e['plants'])
    return windows(n, seed, count, e.pop('menu', None), max(e.pop('nseg', PR.NSEG), nseg or PR.NSEG), **e, **kw)


REFS_VERSION = 3                 # what the references are: 3 since 11-13, the teacher's prediction worked out again when its model is renewed (docs/sim.md, section 6)
REFS_BIG = 4                     # and at a larger size: 4 since 11-13, the steady layer on true values trades the resting values as the cost does (docs/sim.md, section 15)


def key(progs):
    """What tells the batch of a stay: the plants, the tasks and the schedules of its windows, and the version of
    the references. References kept in a file are those of the batch they were computed on (the same seed has drawn
    other plants on another machine, and draws other events on another version of the generator) by the reference
    controllers as they are now."""
    m = hashlib.sha1()
    if REFS_VERSION > 1:
        m.update(b'refs %d' % REFS_VERSION)
    if progs[0].dims != PR.sizes_of(*PR.DIMS):
        m.update(b'big %d' % REFS_BIG)
    if any(bool((p.c['mech'] > 0).any()) for p in progs):
        m.update(b'guard %d' % EN.GUARD)
    for p in progs:
        for x in (p.P, p.T, p.seg_k, p.kinds, p.ramp):
            m.update(x.detach().cpu().numpy().tobytes())
    return m.hexdigest()[:16]


def kept(progs, path=None, ctl=('hold', 'pi', 'dmc', 'teacher'), say=None):
    """The references of these windows (a list of Programs: `exam`, `windows`), as lists: {'score': {controller:
    [window][episode]}, 'rows': [window][episode], 'failed': {controller: [window][episode]}, 'key'}; the teacher
    lives on the plants, the others take every window over from it. With `path`: read from that file when it
    holds those of this batch, else computed and written there."""
    k = key(progs)
    path = Path(path) if path else None
    if path is not None and path.exists():
        d = json.loads(path.read_text())
        if d.get('key') == k and set(ctl) <= set(d['score']):
            return d
    out = references(progs[0].n, 0, len(progs), ctl, say=say, progs=progs)
    d = dict(key=k, n=progs[0].n, windows=len(progs), device=str(progs[0].device), score={q: v.tolist() for q, v in out['score'].items()},
             rows=out['rows'].tolist(), failed={q: v.int().tolist() for q, v in out['failed'].items()})
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(d))
    return d


def menu_of(menu, w):
    """The menu of window w: one for all, one per window (the last goes on), or a function of w."""
    if callable(menu) and not isinstance(menu, PR.Menu):
        return menu(w)
    if isinstance(menu, (list, tuple)) and not isinstance(menu, PR.Menu):
        return menu[min(w, len(menu)-1)]
    return menu


def windows(n, seed, count, menu=None, nseg=PR.NSEG, **kw):
    """The windows of a stay, one at a time: Programs on the same plants (kw: C.build's). Window w has the
    events and the noise of seed+w, as with Program.next."""
    prog = None
    for w in range(count):
        if prog is None:
            prog = C.build(n, seed, menu=menu_of(menu, w), nseg=nseg, **kw)
        else:                                                     # (the tail says what the plants are; only where to put them is asked again)
            prog = C.build(n, seed+w, after=prog.tail, menu=menu_of(menu, w), nseg=nseg, **{q: v for q, v in kw.items() if q in ('device', 'gen_device', 'teacher', 'dtype', 'blind')})
        yield prog


@torch.no_grad()
def references(n, seed, count, ctl=('hold', 'pi', 'dmc', 'teacher'), menu=None, nseg=PR.NSEG, say=None, lead='teacher', progs=None, extra=None, **kw):
    """The reference controllers along a stay of `count` windows. lead: the controller that lives on the plants;
    the others take every window over from the state it left (None, or a lead that is not run: each its own stay).
    Returns per controller (count, n) scores (row-decisions outside), and 'rows' (count, n) the row-decisions of
    a window, 'failed', 'restarted', 'tripped' and 'cost' (the cost of the window from its first event on) per controller (count, n), 'events' (count, n) the events of a window,
    'proc' those that change the process. progs: the windows, if they are not to be made here (`exam`).
    extra: {name: controller} further ones that each live a stay of their own whatever the lead (one that has to be
    there from the start: aio.sim.steptest); they run after the others in every window."""
    lead = lead if lead in ctl else None
    ctl = ([lead] if lead else [])+[q for q in ctl if q != lead]
    made = {q: REFS[q]() for q in ctl}
    extra = dict(extra or {})
    made.update(extra)
    ctl = ctl+list(extra)
    st = {q: None for q in ctl}
    out = dict(score={q: [] for q in ctl}, failed={q: [] for q in ctl}, restarted={q: [] for q in ctl}, tripped={q: [] for q in ctl}, cost={q: [] for q in ctl}, rows=[], events=[], proc=[])
    t0 = time.time()
    for w, prog in enumerate(windows(n, seed, count, menu, nseg, **kw) if progs is None else progs):
        new = prog.seg_k < PR.BIG
        new[:, :prog.s0+1] = False
        out['events'].append(new.sum(1).cpu())
        out['proc'].append((new & ((prog.kinds == PR.PROC) | (prog.kinds == PR.BOTH))).sum(1).cpu())
        out['rows'].append(((PR.HORIZON-prog.k_first).clamp_min(0)*(prog.c['trk']+prog.c['zon']).sum(-1)).cpu())
        left = st[lead] if lead else None                         # where the lead left the plants in the window before
        for q in ctl:
            own = lead is None or q == lead or q in extra
            r = CT.rollout(prog, made[q] if own else REFS[q](), st=st[q] if own else left)        # (taking over: with nothing kept of its own)
            if hasattr(made[q], 'release'):                       # (one that put its own model in the place of the exact one)
                made[q].release(prog)
            st[q] = r['st']
            out['score'][q].append(r['score'].cpu())
            out['failed'][q].append(r['failed'].cpu())
            out['restarted'][q].append(r['restarted'].cpu())
            out['tripped'][q].append(r['tripped'].cpu())
            out['cost'][q].append(r['cost'].cpu())
        if say is not None:
            say(w, {q: float(v[-1].mean()) for q, v in out['score'].items()}, time.time()-t0)
    for key in ('rows', 'events', 'proc'):
        out[key] = torch.stack(out[key])
    for key in ('score', 'failed', 'restarted', 'tripped', 'cost'):
        out[key] = {q: torch.stack(v) for q, v in out[key].items()}
    return out


def table(out, upto=None):
    """Per window, and over the first 1, 2, 4, 8 ... windows together: points of each controller (0: MVs left
    where they are; 100: the best controller run, episode by episode), the share of row-decisions outside, and its
    tail (aio.scale.tail: the share of the units it leaves worse than untouched, its points on its worst tenth)."""
    score = out['score']
    count = out['rows'].shape[0]
    spans = [(w, w+1) for w in range(count)]+[(0, k) for k in (upto or [2**j for j in range(1, 12)]) if 1 < k <= count]
    rows = []
    for a, b in spans:
        per = {q: v[a:b].sum(0).tolist() for q, v in score.items()}
        best = SC.envelope(per) if len(per) > 1 else per['hold']
        line = dict(decisions=(a*PR.HORIZON, b*PR.HORIZON), events=float(out['events'][a:b].sum(0).float().mean()), proc=float(out['proc'][a:b].sum(0).float().mean()))
        for q in score:
            line[q] = dict(points=SC.points(per[q], per['hold'], best) if 'hold' in per else None,
                           outside=float(score[q][a:b].sum()/out['rows'][a:b].sum().clamp_min(1)),
                           failed=int(out['failed'][q][a:b].sum()), **(SC.tail(per[q], per['hold'], best) if 'hold' in per else {}))
        rows.append(line)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exam', default='', help='one of the fixed exams (%s): its own seed, plants and menu; --n, --seed, --windows where given' % ', '.join(EXAMS))
    ap.add_argument('--n', type=int)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--windows', type=int)
    ap.add_argument('--mech', type=float, default=0.)
    ap.add_argument('--plants', default='')
    ap.add_argument('--ctl', default='hold,pi,dmc,teacher')
    ap.add_argument('--rate', default='')
    ap.add_argument('--pace', type=float, default=1.)
    ap.add_argument('--move', type=float, default=1.)
    ap.add_argument('--step', type=float, default=1.)
    ap.add_argument('--proc', type=float, default=1.)
    ap.add_argument('--wander', type=float, default=1.)
    ap.add_argument('--pulse', type=float, default=0.)
    ap.add_argument('--joint', type=float, default=0.)
    ap.add_argument('--ride', type=float, default=None, help='a unit at a constrained point stays there through the next event with this probability (default: 0, on the larger sizes .9)')
    ap.add_argument('--exc', type=float, default=None, help='share of the events that take the unit to a constrained point')
    ap.add_argument('--heavy', type=float, default=0., help='share of the process events that are heavy ones (the gains turn, the time constants change by up to 3)')
    ap.add_argument('--shift', type=float, default=0., help='probability that a unit has one heavy change in a window that stays')
    ap.add_argument('--turn', action='store_true')
    ap.add_argument('--drop', default='')
    ap.add_argument('--quiet', type=int, default=0, help='the first so many windows have no process changes')
    ap.add_argument('--own', action='store_true', help='every controller on its own stay (default: the others take each window over from the teacher)')
    ap.add_argument('--small', action='store_true', help='plants of two or three valves, none to spare')
    ap.add_argument('--blind', action='store_true', help='the packet without the measured disturbances')
    ap.add_argument('--nseg', type=int, default=PR.NSEG, help='segments a window has room for (%d; %d: a window that carries on has room for its events to the end)' % (PR.NSEG, PR.NSEG_STAY))
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    say = lambda w, s, t: print(f'window {w}: ' + '  '.join(f'{q} {v:.0f}' for q, v in s.items()) + f'  ({t:.0f} s)', flush=True)        # noqa: E731
    if a.exam:
        n, seed, count = a.n or EXAM_N, EXAM_SEED if a.seed is None else a.seed, a.windows or EXAM_WINDOWS
        out = references(n, seed, count, tuple(a.ctl.split(',')), lead=None if a.own else 'teacher', say=say,
                         progs=exam(a.exam, n, seed, count, a.nseg, device=a.device, gen_device='cpu'))
        return report(a, out)
    a.n, a.seed, a.windows = a.n or 64, 7 if a.seed is None else a.seed, a.windows or 8
    given = a.rate or a.drop or a.turn or a.quiet or any(getattr(a, q) != d for q, d in (('pace', 1.), ('move', 1.), ('step', 1.), ('proc', 1.), ('wander', 1.), ('pulse', 0.), ('joint', 0.), ('ride', None), ('exc', None), ('heavy', 0.), ('shift', 0.)))
    rate = tuple(float(x) for x in a.rate.split(',')) if a.rate else (1., 1., 1., 1., 1.)
    menu = PR.Menu(rate=rate, pace=a.pace, move=a.move, step=a.step, proc=a.proc, wander=a.wander, turn=a.turn, joint=a.joint, pulse=a.pulse,
                   drop=tuple(x for x in a.drop.split(',') if x), exc=a.exc, ride=a.ride, heavy=a.heavy, shift=a.shift) if given else None
    first = menu._replace(rate=rate[:3]+(0.,)+rate[4:]) if a.quiet else None
    kw = dict(device=a.device, gen_device='cpu', mech=a.mech, blind=a.blind)
    if a.small:
        kw['shape'] = PR.SMALL
    if a.plants:
        kw['plants'] = ME.pool(a.plants.split(','))
    out = references(a.n, a.seed, a.windows, tuple(a.ctl.split(',')), (lambda w: first if w < a.quiet else menu) if a.quiet else menu, a.nseg, lead=None if a.own else 'teacher', say=say, **kw)
    report(a, out)


def report(a, out):
    rows = table(out)
    names = list(out['score'])
    print('decisions      events  proc   ' + '   '.join(f'{q:>14s}' for q in names) + '      (points, share of row-decisions outside)')
    for r in rows:
        cells = '   '.join(f'{r[q]["points"]:6.0f} {r[q]["outside"]:6.3f}' + ('*' if r[q]['failed'] else ' ') for q in names)
        print(f'{r["decisions"][0]:5d}-{r["decisions"][1]:5d}  {r["events"]:5.1f} {r["proc"]:5.1f}   {cells}')
    r = rows[-1]
    print('all of it: worse than untouched / points on the worst tenth   ' + '   '.join(f'{q} {r[q]["worse"]:.0%} / {r[q]["low"]:.0f}' for q in names if q != 'hold'))
    if any(r[q]['failed'] for r in rows for q in names):
        print('* units failed in the span (charged in full, started again at rest in the next window)')
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(dict(args=vars(a), names={q: NAMES[q] for q in names}, table=rows, per_window={q: v.tolist() for q, v in out['score'].items()},
                           rows=out['rows'].tolist(), failed={q: v.tolist() for q, v in out['failed'].items()}), f)


if __name__ == '__main__':
    main()

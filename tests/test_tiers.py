"""aio.tiers: one controller per time scale, joined through the packet (targets, limits)."""
import math

import numpy as np
import pytest
import torch

from aio import net as N
from aio import tiers as TI
from aio import interface as I

N_SLOW = 20


class PI:
    """A velocity-form PI on the packet alone (error and pairing as published, moves in slew limits). The same
    two numbers serve every tier: each sees its loops at about five of its own decisions."""

    def reset(self):
        self.e = None

    def step(self, packet):
        cv, mv, links = I.split(packet)
        e = -(links*cv[:, :1]).sum(0)*mv[:, I.SIGN]
        e0 = e if getattr(self, 'e', None) is None else self.e
        self.e = e
        return np.clip((e-e0)+e/5, -1, 1)


class Plant:
    """Valve 1 -> flow q (lag 5 ticks, scaled by the header pressure), valve 2 -> pressure p (lag 5 ticks,
    pushed by the flow and by a fast load), flow -> temperature T (lag 5 slow decisions)."""

    def __init__(self, seed=0):
        self.g = np.random.default_rng(seed)
        self.q, self.p, self.T, self.head, self.load = 5., 5., 50., 1., 0.

    def tick(self, v1, v2):
        a, b = math.exp(-1/5), math.exp(-1/(5*N_SLOW))
        self.load = .9*self.load+.15*self.g.standard_normal()
        self.q = a*self.q+(1-a)*10*self.head*v1
        self.p = a*self.p+(1-a)*(10*v2-.5*(self.q-5)+self.load)
        self.T = b*self.T+(1-b)*10*self.q
        return dict(q=self.q, p=self.p, T=self.T)


def row(name, band, hi):
    return dict(name=name, band=band, lo=0., hi=hi)


FAST = dict(every=1, dt=1., rows=[row('q', .1, 10.), row('p', .1, 10.)],
            mvs=[dict(name='v1', lo=0., hi=1., slew=.01, pair='q'), dict(name='v2', lo=0., hi=1., slew=.01, pair='p')])
SLOW = dict(every=N_SLOW, dt=float(N_SLOW), rows=[row('T', 1., 100.)], mvs=[dict(name='q', lo=0., hi=10., slew=.1, pair='T')])
DIRECT = dict(every=N_SLOW, dt=float(N_SLOW), rows=[row('T', 1., 100.), row('p', .1, 10.)],
              mvs=[dict(name='v1', lo=0., hi=1., slew=.01, pair='T'), dict(name='v2', lo=0., hi=1., slew=.01, pair='p')])


def run(tiers, ticks=80*N_SLOW, head=None, make=PI, patch=None):
    """Errors in bands (pressure, temperature) per tick, the runner, the flow target and the flow per tick."""
    ts = TI.Tiers(tiers, make) if tiers else None
    if patch:
        patch(ts)
    pl, u = Plant(), dict(v1=.5, v2=.5)
    if ts:
        ts.reset(u)
    y, out, tgt = dict(q=5., p=5., T=50.), [], []
    for k in range(ticks):
        sp = dict(T=52. if k >= 5*N_SLOW else 50., p=5.)
        if head:
            pl.head = head(k)
        if ts:
            u = ts.step(k, y, sp)
        y = pl.tick(u['v1'], u['v2'])
        out.append((abs(y['p']-sp['p'])/.1, abs(y['T']-sp['T'])/1.))
        tgt.append((ts.cmd.get('q', np.nan) if ts else np.nan, y['q'], y['T']))
    return np.array(out), ts, np.array(tgt)


def test_tiers_hold_the_fast_row_a_single_slow_period_cannot():
    drop = lambda k: .9 if k >= 40*N_SLOW else 1.
    hold, _, _ = run(None, head=drop)
    tiers, ts, _ = run([SLOW, FAST], head=drop)
    direct, _, _ = run([DIRECT], head=drop)
    assert ts.valves == ['v1', 'v2'] and list(ts.writer) == ['q']
    p, T = tiers.mean(0)
    assert p < .5*direct[:, 0].mean() and p < .5*hold[:, 0].mean()        # the pressure is held at the fast period only
    assert T < direct[:, 1].mean() and T < .5*hold[:, 1].mean()           # and the slow row loses nothing by it
    assert tiers[-10*N_SLOW:, 1].mean() < .3                                # back on target after the header pressure dropped


def test_a_saturated_valve_stops_the_target_above_it():
    low = lambda k: .4 if 10*N_SLOW <= k < 50*N_SLOW else 1.               # the header gives 4 at most; the target needs 5.2
    _, ts, tr = run([SLOW, FAST], head=low)
    k = 50*N_SLOW-1
    assert tr[k, 0] <= tr[k, 1]+.1+1e-6 and tr[k, 0] > tr[k, 1]             # the target waits one band above the flow
    _, _, wound = run([SLOW, FAST], head=low, patch=lambda t: setattr(t, '_sat', lambda row: 0))
    assert wound[k, 0] > 9.                                                # without the rule it runs to its own limit
    after = slice(50*N_SLOW, 80*N_SLOW)
    assert tr[after, 2].max() < 52+3 < wound[after, 2].max()               # and the temperature overshoots when the header returns


def test_narrow_numpy_and_torch():
    lo, hi, y, band = np.zeros(3), np.full(3, 10.), np.array([4., 4., 12.]), np.full(3, .1)
    a, b = TI.narrow(lo, hi, y, band, np.array([1, -1, 1]))
    assert np.allclose(a, [0, 3.9, 0]) and np.allclose(b, [4.1, 10, 10])
    a, b = TI.narrow(*(torch.tensor(x) for x in (lo, hi, y, band)), torch.tensor([1, 0, -1]))
    assert torch.allclose(a, torch.tensor([0., 0, 10-TI.ROOM*10]).double()) and torch.allclose(b, torch.tensor([4.1, 10, 10]).double())


def test_same_weights_one_memory_per_tier():
    torch.manual_seed(0)
    net = N.AIO().eval()
    out, ts, _ = run([SLOW, FAST], ticks=3*N_SLOW, make=lambda: N.Controller(net))
    slow, fast = ts.ctl
    assert slow.net is fast.net and slow.h is not fast.h and not torch.equal(slow.h, fast.h)
    assert np.isfinite(out).all() and all(0. <= v <= 1. for v in ts.step(3*N_SLOW, dict(q=5., p=5., T=50.), dict(T=50., p=5.)).values())


def test_a_target_has_one_slower_writer():
    with pytest.raises(ValueError):
        TI.Tiers([dict(SLOW, every=1), FAST], PI)
    with pytest.raises(ValueError):
        TI.Tiers([SLOW, dict(SLOW, every=2*N_SLOW), FAST], PI)
    with pytest.raises(ValueError):
        TI.Tiers([dict(SLOW, every=3), dict(FAST, every=2)], PI)


def test_pair_on_the_two_rate_exam():
    from aio.sim import control as CT, rates as RT
    ex = RT.Exam(6, 3)
    torch.manual_seed(0)
    net = N.AIO().eval()
    pair = TI.Pair(lambda: N.Policy(net))
    res = RT.rollout(ex, pair.slow, pair.fast, steps=6)
    assert pair.slow.ctl.net is pair.fast.ctl.net and pair.slow.ctl is not pair.fast.ctl          # one set of weights, two memories
    assert not torch.equal(pair.slow.ctl.h, pair.fast.ctl.h) and torch.isfinite(res['cost']).all()
    y, band, sat = pair.seen
    assert y.shape == band.shape == sat.shape == (ex.n, ex.slow.nmv)

    class Stuck:                                                  # flow loops that report: valve wide open, flow at .3
        def __init__(self, pair):
            self.pair = pair

        def reset(self, view):
            pass

        def __call__(self, view, fst, k, obs):
            z = torch.zeros(view.n, view.nmv, dtype=view.dtype)
            self.pair.seen = (z+.3, z+RT.BAND, torch.ones(view.n, view.nmv))
            return z

    class Up(CT.Hold):                                            # a slow tier that asks for more flow all the time
        def __call__(self, prog, st, k, obs):
            self.pos = st['u'].clone()
            return torch.ones(prog.n, prog.nmv, dtype=prog.dtype)*prog.c['mvm']

    on = ex.v['mvm'] > 0
    for rule in (True, False):
        pair = TI.Pair(Up, rule)
        RT.rollout(ex, pair.slow, Stuck(pair), steps=30)
        asked = pair.slow.ctl.pos[on]
        if rule:                                                  # the flow asked for comes back to one band above what the valves give
            assert bool((asked <= .3+RT.BAND+1e-6).all()) and bool((asked >= .3).all())
        else:
            assert bool((asked > .9).all())

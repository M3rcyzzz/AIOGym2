"""Multi-rate cascade scheduling and limit-narrowing protocol (docs/sim.md, section 十七).

One controller interface per time scale, joined through the public packet (aio.interface.encode) alone:

  target   an MV of a slower tier may be the target of a row of a faster one (a flow, a pressure): the slow
           tier writes the target, the fast tier writes the valve.
  limits   when the valves paired with that row are at their limits on the side the target asks for, the
           slow tier's limit of that MV comes in to one band beyond the row's measurement (`narrow`): the
           target stops where the plant can follow and comes back at the slew limit.
  loads    what one tier measures and another moves can be a measured-disturbance row (dv) of the first.

`Tiers` runs one plant on the public packet (aio.interface.encode). `Pair` is the same on the two-rate exam
of aio.sim.rates, which holds the loop itself: two copies of a controller and the limit rule.
"""
import numpy as np
import torch

from aio import interface as I

ROOM = .01   # the least a narrowed limit leaves of the MV's own span


def narrow(lo, hi, y, band, sat):
    """Limits (lo, hi) of an MV that is the target of a faster row, given that row's measurement y, its band
    and sat: +1 where the row cannot be raised further (its valves are at the limit on that side), -1 where
    it cannot be lowered, 0 elsewhere. The blocked side comes in to one band beyond the measurement, never
    closer to the other limit than ROOM of the span. numpy arrays or torch tensors."""
    room = ROOM*(hi-lo)
    up, dn = (sat > 0), (sat < 0)
    return lo+dn*((y-band).clip(lo, hi-room)-lo), hi+up*((y+band).clip(lo+room, hi)-hi)


class Tiers:
    """tiers: dicts in any order, each
         every  base ticks per decision; dt: the same in seconds (the packet's period slot)
         rows   dicts: name, band, lo, hi (zone limits; the instrument range of a row that is only tracked),
                and optionally track (1), dv (0), weight (1)
         mvs    dicts: name, lo, hi, slew (per decision of this tier), pair (a row of this tier), sign (+1)
       An MV named like a row of a faster tier writes that row's target; every other MV is a valve.
       make() gives a controller with step(packet) -> moves in [-1, 1] of the slew limit; it is called once
       per tier, so the tiers share weights and nothing else."""

    def __init__(self, tiers, make):
        self.tiers = sorted(tiers, key=lambda t: -t['every'])
        home = {r['name']: t for t in self.tiers for r in t['rows']}
        self.writer = {}
        for t in self.tiers:
            if t['every'] % self.tiers[-1]['every']:
                raise ValueError('every period is a multiple of the fastest')
            for m in t['mvs']:
                if m['name'] in home:
                    if home[m['name']]['every'] >= t['every'] or m['name'] in self.writer:
                        raise ValueError(f"the target of {m['name']} is written by one slower tier")
                    self.writer[m['name']] = t
        self.band = {r['name']: r['band'] for t in self.tiers for r in t['rows']}
        self.pairs = {}                                           # row -> (valve or target, sign) paired with it
        for t in self.tiers:
            for m in t['mvs']:
                self.pairs.setdefault(m['pair'], []).append((m['name'], m.get('sign', 1.)))
        self.valves = [m['name'] for t in self.tiers for m in t['mvs'] if m['name'] not in home]
        self.ctl = [make() for _ in self.tiers]
        self.reset({})

    def reset(self, u):
        """u {valve: position} the plant is at. A target starts where its row is at the tier's first decision."""
        self.cmd, self.at, self.last = dict(u), {}, [None]*len(self.tiers)
        for c in self.ctl:
            if hasattr(c, 'reset'):
                c.reset()

    def _sat(self, row):
        """+1 when every MV paired with the row is at its limit on the side that raises it, -1 lowers, else 0."""
        s = [self.at.get(n, 0)*g for n, g in self.pairs.get(row, ())]
        return 0 if not s else 1 if min(s) > 0 else -1 if max(s) < 0 else 0

    def step(self, k, y, sp):
        """Base tick k. y {row: measurement}, sp {row: target} for the rows no tier writes -> {valve: command}.
        The tiers due at k decide slowest first, so a new target is followed from the same tick."""
        for n, (t, c) in enumerate(zip(self.tiers, self.ctl)):
            if k % t['every']:
                continue
            R, M = t['rows'], t['mvs']
            names = [r['name'] for r in R]
            for m in M:
                if m['name'] in self.writer:
                    self.cmd.setdefault(m['name'], float(y[m['name']]))
            v = np.array([y[q] for q in names], float)
            tg = np.array([self.cmd[q] if q in self.writer else sp[q] for q in names], float)
            u = np.array([self.cmd[m['name']] for m in M], float)
            lo, hi = np.array([m['lo'] for m in M], float), np.array([m['hi'] for m in M], float)
            for j, m in enumerate(M):
                if m['name'] in self.writer:
                    lo[j], hi[j] = narrow(lo[j], hi[j], np.float64(y[m['name']]), self.band[m['name']], self._sat(m['name']))
            slew = np.array([m['slew'] for m in M], float)
            pv, ptg, pu = self.last[n] or (v, tg, u)
            self.last[n] = (v, tg, u)
            packet = I.encode(values=v, previous=pv, target=tg, last_target=ptg, scale=np.array([r['band'] for r in R], float),
                              lower=np.array([r['lo'] for r in R], float), upper=np.array([r['hi'] for r in R], float),
                              tracking=np.array([r.get('track', 1.) for r in R], float), command=u, last_command=pu, u_low=lo, u_high=hi,
                              slew=slew, pairing=[names.index(m['pair']) for m in M], dt=t['dt'], sign=[m.get('sign', 1.) for m in M],
                              dv=[r.get('dv', 0.) for r in R], weight=[r.get('weight', 1.) for r in R])
            a = np.asarray(c.step(packet), float)[:len(M)]
            u = u+np.clip(np.clip(u+a*slew, lo, hi)-u, -slew, slew)   # inside the limits; from outside them, back at the slew limit
            for m, x, l, h in zip(M, u, lo, hi):
                self.cmd[m['name']], self.at[m['name']] = float(x), int(x >= h-1e-9)-int(x <= l+1e-9)
        return {q: self.cmd[q] for q in self.valves}


class Pair:
    """Two tiers of one controller on an exam of aio.sim.rates. make() gives a controller of aio.sim.control
    (reset(prog), ctl(prog, st, k, obs) -> moves) and is called twice: `slow` and `fast` share weights and
    nothing else. rule: the slow tier reads its packet, and moves, inside limits taken in where the valve
    under an MV is at an end of its stroke (`narrow`). fast: another maker for the fast tier (the references
    of the exam: a DMC over flow loops)."""

    def __init__(self, make, rule=True, fast=None):
        self.rule, self.seen = rule, None
        self.slow, self.fast = _Slow(self, make()), _Fast(self, (fast or make)())


class _Fast:
    def __init__(self, pair, ctl):
        self.pair, self.ctl = pair, ctl

    def reset(self, view):
        self.pair.seen = None
        self.ctl.reset(view)

    def __call__(self, view, fst, k, obs):
        row = view.ex.frow                                        # (n, NMV): the row of every valve's flow
        self.pair.seen = (torch.gather(fst['y'], 1, row), torch.gather(fst['band'], 1, row), fst['sat'].clone())
        return self.ctl(view, fst, k, obs)


class _Slow:
    def __init__(self, pair, ctl):
        self.pair, self.ctl = pair, ctl

    def reset(self, prog):
        self.ctl.reset(prog)

    def __call__(self, prog, st, k, obs):
        from aio.sim import core as C, rates as RT
        if not self.pair.rule or self.pair.seen is None:
            return self.ctl(prog, st, k, obs)
        y, band, sat = self.pair.seen
        t = prog.task(prog.seg_at(k))
        px = RT.narrow(prog, *narrow(t['ulo'], t['uhi'], y.to(t['ulo'].dtype), band.to(t['ulo'].dtype), sat))
        return px.move(st, self.ctl(px, st, k, C.observe(px, st, k)), k)

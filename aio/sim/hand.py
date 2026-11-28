"""Hand-written controllers on what the network has: the comparison line "the best hand-written adaptive logic
with the same information" (11-11, Mercy: is the ceiling a piece of logic software one could write by hand?).

They read what a packet holds and nothing else: the published pairing and the sign of every pair's gain, the
targets, the bands, the measurements, the valve positions. No step test is handed over and no model: the formula
PI and the step-test DMC of aio.sim.control are tuned on the exact step responses of the start plant, so they are
not of the same information.

  ZeroPI      one velocity-form PI for every loop of every plant: move = sign * k0 * (de + e/n0), e in bands, the
              move in slew limits (the zero-tuning PI of 11-08 on this core)
  AdaptivePI  the same with a gain per loop that adapts as it runs, by performance monitoring and no model: up
              while the error has stayed on one side outside the band and is not closing, down when it swings
              from outside one edge to outside the other (Hagglund's idle index and oscillation detector)
  RelayPI     every loop runs a relay test of its own from the start of the episode (its valve +-d of span around
              where it stood, switching when the error leaves a hysteresis), reads the ultimate gain and period
              off the swings and tunes its PI by a Ziegler-Nichols rule (Astrom and Hagglund); a loop that does
              not swing in time runs the zero-tuning PI. The test is the controller's own and is scored

Their settings were chosen on a batch of their own (128 episodes, seed 5: 16 settings of ZeroPI, 100 random ones
of AdaptivePI, 60 of RelayPI) and are the defaults here. On the validation batch (256 episodes, seed 10^6; hold
0, the Oracle envelope 100): ZeroPI 22.5, AdaptivePI 27.8, RelayPI 27.7; the formula PI 47.4, the step-test DMC
87.7. Each is worse than leaving the MVs alone in 55-70 of the 256 episodes (the formula PI in 30). The other
hand-written route, identifying the responses on line and solving the teacher's design on them, reads 26-29 of
the teacher's 100 (docs/heldout.md at commit 6ce4fff, 11-11 20:40).

    python -m aio.sim.hand --n 256 --seed 1000000 [--mech 0] [--refs FILE of aio.sim.oracle]
"""
import argparse
import json
import math

import torch

from aio import scale as SC
from aio.sim import control as CT, core as C


class ZeroPI:
    def __init__(self, k0=.1, n0=10.):
        self.k0, self.n0 = k0, n0

    def reset(self, prog):
        self.e = None
        self.sign = torch.sign(prog.pi['kc'])                     # the published sign of every pair's gain (in the packet)

    def err(self, prog, st):
        """(n, NMV) the error of every MV's paired row, in bands."""
        return torch.gather((st['target']-st['y'])/prog.c['bandf'], 1, prog.pi['row'])

    def last(self, e):
        e0, self.e = (e if self.e is None else self.e), e
        return e0

    def __call__(self, prog, st, k, obs):
        e = self.err(prog, st)
        return (self.sign*self.k0*((e-self.last(e))+e/self.n0)).clamp(-1., 1.)*prog.c['mvm']


class AdaptivePI(ZeroPI):
    """up, down: steps of a loop's log gain; hold: decisions an error has to have stayed on one side; band: the
    monitors' dead band (bands); close: what an error has to shrink by in a decision to count as closing."""

    def __init__(self, k0=.1, n0=30., up=.03, down=.1, hold=20, band=.5, close=.02, lim=4.):
        super().__init__(k0, n0)
        self.up, self.down, self.hold, self.band, self.close, self.lim = up, down, hold, band, close, lim

    def reset(self, prog):
        super().reset(prog)
        self.th, self.side, self.run = (torch.zeros(prog.n, prog.nmv, dtype=prog.dtype, device=prog.device) for _ in range(3))

    def __call__(self, prog, st, k, obs):
        e = self.err(prog, st)
        e0 = self.last(e)
        out = e.abs() > self.band
        side = torch.sign(e)*out
        same, swung = (side == self.side) & out, side*self.side < 0
        self.run = torch.where(same, self.run+1, torch.zeros_like(self.run))
        self.side = torch.where(out, side, self.side)
        slow = same & (self.run >= self.hold) & ~(e.abs() < e0.abs()-self.close)
        self.th = (self.th+self.up*slow.to(self.th.dtype)-self.down*swung.to(self.th.dtype)).clamp(-self.lim, self.lim)
        return (self.sign*self.k0*torch.exp(self.th)*((e-e0)+e/self.n0)).clamp(-1., 1.)*prog.c['mvm']


class RelayPI(ZeroPI):
    """d: the relay's amplitude (of span); eps: its hysteresis (bands); nsw: swings before tuning; tmax: decisions
    a loop is given to swing; kf, tf: the rule (Kc = kf*Ku, Ti = tf*Tu; Ziegler-Nichols .45, .83)."""

    def __init__(self, d=.1, eps=1., nsw=4, tmax=400, kf=.45, tf=.83, k0=.1, n0=10.):
        super().__init__(k0, n0)
        self.d, self.eps, self.nsw, self.tmax, self.kf, self.tf = d, eps, nsw, tmax, kf, tf

    def reset(self, prog):
        super().reset(prog)
        z = lambda: torch.zeros(prog.n, prog.nmv, dtype=prog.dtype, device=prog.device)       # noqa: E731
        self.u0, self.s, self.cnt, self.t_sw, self.peak, self.sum_t, self.sum_a, self.kc, self.ti = None, z(), z(), z(), z(), z(), z(), z(), z()
        self.phase = torch.zeros(prog.n, prog.nmv, dtype=torch.long, device=prog.device)      # 0: relay test, 1: tuned, 2: did not swing

    def __call__(self, prog, st, k, obs):
        c, u = prog.c, st['u']
        e = self.err(prog, st)
        e0 = self.last(e)
        if self.u0 is None:
            self.u0 = u.clone()
        test = (self.phase == 0) & (c['mvm'] > 0)
        one = torch.ones_like(e)
        want = torch.where(e > self.eps, one, torch.where(e < -self.eps, -one, torch.where(self.s == 0, one, self.s)))
        sw = test & (want != self.s) & (self.s != 0)
        self.peak = torch.maximum(self.peak, e.abs())
        count = sw & (self.cnt >= 1)                              # the first half swing starts anywhere: not counted
        self.sum_t = self.sum_t+torch.where(count, k-self.t_sw, torch.zeros_like(e))
        self.sum_a = self.sum_a+torch.where(count, self.peak, torch.zeros_like(e))
        self.cnt = self.cnt+sw.to(e.dtype)
        self.t_sw = torch.where(sw, torch.full_like(e, float(k)), self.t_sw)
        self.peak = torch.where(sw, torch.zeros_like(e), self.peak)
        self.s = torch.where(test, want, self.s)
        relay = (((self.u0+self.sign*self.s*self.d).clamp(0., 1.)-u)/c['slew']).clamp(-1., 1.)
        ready = test & (self.cnt >= self.nsw)
        half = (self.cnt-1).clamp_min(1.)
        ku = 4*self.d/(math.pi*(self.sum_a/half).clamp_min(self.eps))     # ultimate gain, span per band
        self.kc = torch.where(ready, self.kf*ku, self.kc)
        self.ti = torch.where(ready, (self.tf*2*self.sum_t/half).clamp_min(1.), self.ti)
        late = test & ~ready & (k >= self.tmax)
        self.phase = torch.where(ready, torch.ones_like(self.phase), torch.where(late, torch.full_like(self.phase, 2), self.phase))
        tuned = (self.sign*self.kc*((e-e0)+e/self.ti.clamp_min(1.))/c['slew']).clamp(-1., 1.)
        zero = (self.sign*self.k0*((e-e0)+e/self.n0)).clamp(-1., 1.)
        return torch.where(self.phase == 1, tuned, torch.where(self.phase == 2, zero, relay))*c['mvm']


HAND = dict(zero_pi=ZeroPI, adaptive_pi=AdaptivePI, relay_pi=RelayPI)


@torch.no_grad()
def table(prog, refs=None):
    """Scores and points of the hand-written controllers beside the references on a program. Points: hold 0, and
    100 the envelope of the references run here and, with refs (a file of aio.sim.oracle for this batch), of the
    Oracle; before / after: on the parts of the scored window up to an episode's first process change and from
    it on (0 and 100 of the references run here on the same part)."""
    ref = dict(hold=CT.Hold(), pi=CT.PI(), dmc=CT.Teacher('fixed'), teacher=CT.Teacher('late'), now=CT.Teacher('now'))
    run = {q: CT.rollout(prog, c) for q, c in dict(ref, **{q: f() for q, f in HAND.items()}).items()}
    per = {q: run[q]['score'].tolist() for q in ref}
    scale = SC.Scale.load(refs, per['pi']) if refs else None
    used = scale is not None
    scale = scale.join(per) if used else SC.Scale.of(per)
    out = dict(n=prog.n, refs_used=used, scale=scale.meta(), worse_than_hold={})
    for q, r in run.items():
        out[q] = dict(score=round(float(r['score'].mean()), 1), pts=round(scale(r['score'].tolist()), 1))
        for part in ('before', 'after'):
            best = torch.stack([run[x][part] for x in ref if x != 'hold']).amin(0)
            out[q]['pts_'+part] = round(SC.points(r[part].tolist(), run['hold'][part].tolist(), best.tolist()), 1)
        out['worse_than_hold'][q] = int((r['score'] > run['hold']['score']).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--seed', type=int, default=10**6)
    ap.add_argument('--mech', type=float, default=0.)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--refs')
    a = ap.parse_args()
    print(json.dumps(dict(seed=a.seed, mech=a.mech, **table(C.build(a.n, a.seed, a.device, 'cpu', mech=a.mech), a.refs))), flush=True)


if __name__ == '__main__':
    main()

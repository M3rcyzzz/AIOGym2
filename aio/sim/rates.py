"""One plant at two rates: the exam of a controller that runs in tiers (docs/sim.md, section 十八; the scheme and
its criterion: section 十七, aio/tiers.py).

A mechanistic plant whose MVs are flows is met with the valves themselves in the loop. The plant is advanced
on a base tick (FAST seconds); its own decision interval is m ticks. Between a flow that is asked for and the
flow the plant gets stands a control valve on the tick: stroking rate, positioner lag, friction, the trim's
characteristic as installed (its share of the pressure drop), a supply pressure that wanders and steps (the
fast disturbance, which no slow measurement shows before the levels have moved), the flow's own lag, a noisy
meter.

    ex = Exam(n, seed, plant='qtank')
    ex.slow     a core.Program as any other: the rows are the plant's CVs, the MVs are the flows asked for;
                period ex.m ticks; events, instruments and the score are the core's, on the true values
    ex.fast     what a controller reads of a Program, for the flow loops: the rows are the flows (tracked; the
                target of a flow is where the slow MV stands, held over the slow period; on the episodes with
                one header the supply pressure is a DV row), the MVs are the valves; period one tick
    res = rollout(ex, slow, fast, mode)

Both controllers are of aio.sim.control: reset(prog) and ctl(prog, st, k, obs) -> moves in [-1, 1] of the slew
limit. `slow` is called every m ticks with ex.slow and the core's packet, `fast` every tick with ex.fast, the
state of the flow loops `fst` (y: the meters, band, u: the valves, sat: -1/0/+1 a valve at an end of its
stroke) and their packet, C.observe(ex.fast, fst, kf): the same layout and size as the slow one. A scheduler
of two tiers is one object that gives the two; `narrow` gives it the slow packet and the slow move under limits
it has taken in because valves below are at an end of their stroke.

  mode 'tiers'    the fast controller writes the valves
       'direct'   one slow period writes the valves: the slow MV is the valve position, linearised by the
                  valve's nominal curve (at nominal pressure the flow is the MV), and nothing looks at the
                  flows in between
       'ideal'    no valves: the flow asked for is the flow (the plant as the core has it; a bound)

The score of the slow rows is the core's. `fscore` counts the flows on the tick, on the true flow: outside its
band of the target; while the valve can still be on its way after the target moved (REACH time constants),
anywhere between the old and the new target counts as inside, and so does a flow short of a target with its
valve at the end of the stroke.

`references` runs, on the same episodes: the valves left where they are (0 points); the formula PI, a step-test
DMC and the re-identified DMC writing the valves at the slow period; the first two over flow PIs (the industrial
practice); an exact-model controller in each tier (the re-identified DMC over flow loops that know their valves).
100 points is the best of them per episode, on the easy side: no long-horizon controller with the full state at
the tick was run. The same slow controllers on the plant without valves are run beside them as a bound.

Exam(sag=p): on that share of the episodes the supply falls for a stretch until a valve wide open passes only
a share of the flow it had at the start (a pump lost: the header where the valves share one, else the supply of
one valve), so that valves reach the end of their stroke short of the flow asked for; what a slow tier does
then (aio.tiers, the limit rule) shows in the slow rows during the sag and after it. `references` then also
runs the two tiers of references with that rule.

An exam: no training batch draws from it and no fixed batch changes with it. The input transport delay of the
shell is left out (it would stand between the slow MV and the valve).
"""
import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import torch

from aio import scale as SC
from aio.sim import core as C, control as CT, mech as ME, model as MD, prior as PR
from aio.sim.engine import Mech
from aio.sim.prior import HORIZON

FAST = 1.                                     # the base tick (s): the period of the fast tier
STROKE = (4., 20.)                            # a full stroke of a valve takes this long (s)
TAU_V, TAU_Q = (1., 6.), (.5, 3.)             # time constants (s) of the positioner and actuator, and of the flow behind the valve (the liquid's inertia, the meter)
EQ_P, RANGE = .5, (30., 50.)                  # share of equal-percentage trims (the others linear) and their rangeability
AUTH = (.2, .8)                               # the valve's share of the pressure drop at full opening (the line has the rest)
OVER = (1.15, 1.5)                            # the flow at full opening and nominal supply pressure, over the top of the MV's range
STICK_P, STICK = .4, (.001, .005)             # share of the valves with friction, and its band (of the stroke); cut so that a slip moves the flow by half a band at most
P_TAU, P_SIG = (5., 60.), (.03, .10)          # the supply pressure wanders around its level: correlation time (s), standing deviation (of nominal)
P_STEPS, P_STEP, P_MAX = 4, .15, .4           # the level changes up to this many times in an episode (another user of the header) to within P_STEP of nominal; the deviation stays within P_MAX
HEADER_P = .5                                 # share of the episodes with all valves on one header; its pressure is measured (a DV row of the fast packet)
SAG, SAG_AT, SAG_FOR = (.5, .9), (.2, .5), (.08, .2)       # a sag (Exam(sag=share of the episodes)): the supply falls until a valve wide open passes this share of its flow at the start (a pump lost), from this point of the episode, for this long (of the horizon)
NOISE = (.001, .004)                          # noise of the flow meters (standing deviation, of the span)
BAND = .01                                    # the band of a flow (of the span)
REACH = 3.                                    # after its target moved, a flow has this many time constants of its valve to get there
MODES = ('tiers', 'direct', 'ideal')
STATE = ('c', 'a', 'xs', 'qd', 'dou', 'ym', 'ymp', 'cp', 'tg', 'tprev', 'since')     # per valve: command, actuator, stem, flow, pressure wander, meter now and before, command before, target before, target before it moved, ticks since


@contextlib.contextmanager
def _shell():
    """The shell of a mechanistic episode without a valve path of its own (the exam has the valve: no curve, lag
    or friction, and no transport delay between the MV and the valve) and at the plant's own clock."""
    keep = ME.PACE, ME.DEAD_P, dict(ME.VALVE)
    ME.PACE, ME.DEAD_P = (1., 1.), 0.
    ME.VALVE.update({q: 0. for q in keep[2]})
    try:
        yield
    finally:
        ME.PACE, ME.DEAD_P = keep[:2]
        ME.VALVE.update(keep[2])


class Exam:
    """n episodes of one plant with its valves on the tick. All of it is fixed by (n, seed, plant, fast, sag);
    the episodes, valves and pressures of an exam with sags are those of the exam without, the sags apart."""

    def __init__(self, n, seed=0, plant='qtank', fast=FAST, device='cpu', dtype=torch.float32, sag=0.):
        with _shell():
            self.slow = prog = C.build(n, seed, device, mech=1., plants=ME.pool([plant]), dtype=dtype)
        assert len(prog.mg) == 1 and prog.nl == 0, 'one batch of plants'
        NMV = prog.nmv
        t = prog.mech.groups[0].t
        dt = float(t['dt'][0])
        self.m = int(round(dt/fast))
        assert self.m >= 2 and abs(self.m*fast-dt) < 1e-6*dt, f'the decision of {plant} ({dt:g} s) is not a whole number of ticks of {fast:g} s'
        self.n, self.seed, self.plant, self.tick, self.device, self.dtype = n, int(seed), plant, float(fast), prog.device, prog.dtype
        p0 = prog.params(torch.zeros(n, dtype=torch.long, device=prog.device))
        assert bool((prog.c['aa'] == 0).all()) and bool((p0['gam'] == 1).all()) and bool((p0['fs'] == 0).all()) and not bool(prog.mg[0]['din'].any())
        # ---- the valves, drawn once per episode (CPU, float64: the same exam on every device)
        g = torch.Generator(); g.manual_seed(self.seed*7919+20261111)
        F = torch.float64
        U = lambda lo, hi, *s: lo+(hi-lo)*torch.rand(n, *s, generator=g, dtype=F)                # noqa: E731
        LU = lambda lo, hi, *s: torch.exp(U(math.log(lo), math.log(hi), *s))                     # noqa: E731
        mvm = prog.c['mvm'].cpu().to(F)
        v = dict(eq=(U(0., 1., NMV) < EQ_P).to(F), R=U(*RANGE, NMV), N=U(*AUTH, NMV), S=U(*OVER, NMV))
        tv, tq = LU(*TAU_V, NMV), LU(*TAU_Q, NMV)
        v.update(av=torch.exp(-fast/tv), bq=torch.exp(-fast/tq), slew=(fast/LU(*STROKE, NMV)).clamp_max(.5),
                 reach=torch.ceil(REACH*(tv+tq)/fast).clamp_min(2.), noise=LU(*NOISE, NMV))
        self.v = v
        grid = torch.linspace(0., 1., 65, dtype=F)[:, None, None]
        fs = torch.where(U(0., 1., NMV) < STICK_P, LU(*STICK, NMV), torch.zeros(n, NMV, dtype=F))
        v['fs'] = torch.minimum(fs, .5*BAND/self.slope(grid).amax(0))*mvm
        v['fd'] = v['fs']*U(.3, 1., NMV)
        common = U(0., 1.) < HEADER_P
        one = lambda a: torch.where(common.view(-1, *([1]*(a.dim()-1))), a[:, :1].expand_as(a), a)       # noqa: E731  (one header: every valve has the first one's)
        v['rho'] = one(torch.exp(-fast/LU(*P_TAU, NMV)))
        v['sig'] = one(U(*P_SIG, NMV))*torch.sqrt(1-v['rho']**2)
        self.pk = one((U(0., 1.3, NMV, P_STEPS)*HORIZON*self.m).sort(-1).values.long())
        self.pv = torch.cat((torch.zeros(n, NMV, 1, dtype=F), one(U(-P_STEP, P_STEP, NMV, P_STEPS))), -1)
        # ---- where the flows stand in the fast packet: the flow of valve j in row frow[:, j], a random one of the first rows; the header pressure after them
        key = U(0., 1., NMV)+10.*(1-mvm)
        self.frow = torch.argsort(torch.argsort(key, -1), -1)
        self.dvrow = mvm.sum(-1).long()
        self.common = common
        # ---- the sags, from a stream of their own: when (ticks from, to) and how deep per valve (0: none)
        self.sag = float(sag)
        gs = torch.Generator(); gs.manual_seed(self.seed*7919+20261112)
        us = torch.rand(n, 4+NMV, generator=gs, dtype=F)
        at = SAG_AT[0]+(SAG_AT[1]-SAG_AT[0])*us[:, 1]
        self.sk = (torch.stack((at, at+SAG_FOR[0]+(SAG_FOR[1]-SAG_FOR[0])*us[:, 2]), 1)*HORIZON*self.m).long()
        hit = torch.nn.functional.one_hot((us[:, 4:]+10.*(1-mvm)).argmin(-1), NMV).to(F)        # the supply of one valve, where every valve has its own
        need = prog.start['ue'].cpu().to(F)/v['S']                # the flow of the start over the flow wide open, at nominal pressure
        need = torch.where(common[:, None], ((need*mvm).sum(-1)/mvm.sum(-1).clamp_min(1.))[:, None].expand_as(need), need)      # one header: one pressure, set by the valves' mean
        left = (((SAG[0]+(SAG[1]-SAG[0])*us[:, 3])[:, None]*need)**2).clamp(0., 1.)             # the pressure that is left, of nominal
        self.sd = (1-left)*(us[:, 0] < self.sag)[:, None]*torch.where(common[:, None], torch.ones_like(hit), hit)*mvm
        to = lambda a: a.to(prog.device, prog.dtype)                                            # noqa: E731
        self.v = {q: to(a) for q, a in v.items()}
        self.v['mvm'] = prog.c['mvm']
        self.pk, self.pv, self.frow, self.dvrow, self.common = self.pk.to(prog.device), to(self.pv), self.frow.to(prog.device), self.dvrow.to(prog.device), common.to(prog.device)
        self.sk, self.sd = self.sk.to(prog.device), to(self.sd)
        self.x0 = self.opening(prog.start['ue'])                  # the valves at the start: where the flows of the start point are, at nominal pressure
        self.fast = View(self)
        self.scale = self.fscale = None

    # ---- the valve as installed (flows in spans of the MV's range)
    def trim(self, x):
        v = self.v
        return torch.where(v['eq'] > 0, (v['R']**x-1)/(v['R']-1), x)

    def flow(self, x, dp=0.):
        """The flow through a valve at opening x with the supply pressure dp off its nominal value."""
        v = self.v
        f = self.trim(x)
        return v['S']*f/torch.sqrt(v['N']+(1-v['N'])*f*f)*(1+dp)**.5

    def slope(self, x):
        """d flow / d opening at nominal pressure."""
        v = self.v
        f = self.trim(x)
        df = torch.where(v['eq'] > 0, torch.log(v['R'])*v['R']**x/(v['R']-1), torch.ones_like(f))
        return v['S']*v['N']/(v['N']+(1-v['N'])*f*f)**1.5*df

    def opening(self, q):
        """The opening that gives the flow q at nominal pressure (the valve's nominal curve, inverted)."""
        v = self.v
        r = (q/v['S']).clamp(0., 1.)
        f = r*torch.sqrt(v['N']/(1-r*r*(1-v['N'])).clamp_min(1e-12))
        return torch.where(v['eq'] > 0, torch.log1p(f*(v['R']-1))/torch.log(v['R']), f).clamp(0., 1.)

    @contextlib.contextmanager
    def attach(self, layer):
        """The valves stand where the engine stood, for the rollouts inside."""
        g, s = self.slow.mg[0], self.slow.start
        keep = g['m'], s['mx']
        g['m'], s['mx'] = layer, layer.start(s['mx'])
        try:
            yield layer
        finally:
            g['m'], s['mx'] = keep

    def points(self, res):
        """(slow, fast) points of a rollout on the scales of `references`."""
        return self.scale(res['score'].tolist()), (self.fscale(res['fscore'].tolist()) if 'fscore' in res else None)


class View:
    """What a controller reads of a Program (aio.sim.control, aio.net.Policy), for the flow loops of an exam:
    n, sizes, c, task, seg_at, pi. The packet is C.observe(view, fst, kf)."""

    def __init__(self, ex):
        prog, v = ex.slow, ex.v
        self.ex, self.n, self.device, self.dtype = ex, ex.n, ex.device, ex.dtype
        self.dims, self.size = prog.dims, prog.size
        NCV, NMV, ND, NQ, NDQ, NU = prog.dims
        self.ncv, self.nmv, self.nd, self.nq, self.ndq, self.nu = prog.dims
        n, mvm = ex.n, v['mvm']
        z = lambda *s, f=0.: torch.full((n, *s), f, dtype=ex.dtype, device=ex.device)            # noqa: E731
        trk = z(NCV).scatter(1, ex.frow, mvm)
        dvr = z(NCV).scatter(1, ex.dvrow[:, None], ex.common[:, None].to(ex.dtype))
        link = z(NCV, NMV)
        link[prog.ar[:, None], ex.frow, torch.arange(NMV, device=ex.device)[None]] = mvm
        self.c = dict(trk=trk, dvr=dvr, rowm=trk+dvr, an_T=z(NCV), dt=z(f=ex.tick), mvm=mvm, slew=v['slew'], psign=mvm, link=link, integ=z(NCV), bandf=z(NCV, f=BAND))
        self._task = dict(scale=torch.where(trk > 0, z(NCV, f=BAND), torch.where(dvr > 0, z(NCV, f=1.), z(NCV, f=.5))), plo=-dvr, phi=z(NCV, f=1.),
                          cvw=z(NCV, f=1.), ulo=z(NMV), uhi=z(NMV, f=1.))
        self._seg = torch.zeros(n, dtype=torch.long, device=ex.device)
        # the formula PI of every flow loop, from the step response of its valve at the start (a commissioning step test)
        L = 60
        g0, a, q, s = ex.slope(ex.x0), z(NMV), z(NMV), [z(NMV)]
        for _ in range(L):
            a = v['av']*a+(1-v['av'])
            q = v['bq']*q+(1-v['bq'])*g0*a
            s.append(q)
        s = torch.stack(s, 1)                                     # (n, L+1, NMV)
        S = z(L+1, NCV, NMV)
        S[prog.ar[:, None, None], torch.arange(L+1, device=ex.device)[None, :, None], ex.frow[:, None, :], torch.arange(NMV, device=ex.device)[None, None, :]] = s
        kc, ti, row = MD.pi_tuning(S, dict(K=z(NCV, NU), dlev=z(NDQ), link=link, integ=z(NCV), mvm=mvm))
        r = s/g0.clamp_min(1e-9)[:, None]
        first = lambda c: c.to(torch.int64).argmax(1).to(ex.dtype)                              # noqa: E731
        theta = (first(r >= .05)-.5).clamp_min(0.)
        self.pi = dict(kc=kc, ti=ti, row=row, theta=theta, tau=(first(r >= .632)-theta).clamp_min(1.))

    def seg_at(self, k):
        return self._seg

    def task(self, seg):
        return self._task

    def slope(self, x):
        return self.ex.slope(x)


class Flow:
    """A flow loop that knows its valve: the rule of the formula PI with the gain of the installed curve where
    the valve stands, and the shortest closed-loop time the rule allows (3 dead times, a tick at least)."""

    def reset(self, view):
        self.e = None

    def __call__(self, view, fst, k, obs):
        pi, c = view.pi, view.c
        e = torch.gather(fst['target']-fst['y'], 1, pi['row'])
        e0 = e if self.e is None else self.e
        self.e = e
        lam = (3*pi['theta']).clamp_min(1.)
        kc = pi['tau']/(view.slope(fst['u']).clamp_min(1e-3)*(lam+pi['theta']))
        return (kc*((e-e0)+e/pi['tau'])/c['slew']).clamp(-1., 1.)*c['mvm']


class Valves:
    """The valves and the plant over one slow decision, standing where the engine stands in core._mech (advance,
    y, nx). Its states follow the plant's in the Program's `mx`: the tick, then STATE per valve, then the
    header pressure as measured now and before. ctl: the fast controller (None: 'direct')."""

    def __init__(self, ex, ctl=None, trace=None):
        prog = ex.slow
        g, m0 = prog.mg[0], prog.mg[0]['m']
        self.ex, self.ctl, self.g = ex, ctl, g
        self.M = Mech(m0.plants, m0.pid, device=m0.device, dtype=m0.dtype, _base=m0._base, pace=m0.pace/ex.m)
        self.trips, self.tripped = self.M.trips, self.M.tripped
        self.nxp, self.nmv = m0.nx, prog.nmv
        self.nx = self.nxp+1+len(STATE)*self.nmv+2
        t = prog.mech.groups[0].t
        self.ismv, self.fidx = (t['ikind'] == 0).to(ex.device), t['fidx'].to(ex.device)
        self.slot = g['chan'].clamp_max(self.nmv-1)
        self.gen = torch.Generator()
        self.first = prog.k_first*ex.m                            # the tick from which the flows are scored
        self.trace, self.tr = trace, []
        self._zero()

    def _zero(self):
        z = torch.zeros(self.ex.n, dtype=torch.float64, device=self.ex.device)
        self.out, self.rows, self.sq, self.sat = z, z.clone(), z.clone(), z.clone()

    def start(self, mx):
        ex = self.ex
        x0, q0 = ex.x0, ex.slow.start['ue']
        s = dict(c=x0, a=x0, xs=x0, qd=q0, dou=torch.zeros_like(x0), ym=q0, ymp=q0, cp=x0, tg=q0, tprev=q0, since=torch.full_like(x0, 1e6))
        return torch.cat([mx[:, :self.nxp], mx.new_zeros(ex.n, 1)]+[s[q] for q in STATE]+[mx.new_zeros(ex.n, 2)], 1)

    def _split(self, x):
        a, nm = self.nxp+1, self.nmv
        return x[:, :self.nxp], {q: x[:, a+i*nm:a+(i+1)*nm] for i, q in enumerate(STATE)}, x[:, a+len(STATE)*nm:]

    def _inputs(self, win, q):
        """The plant's inputs with the flows q (n, NMV) in place of the flows asked for."""
        return torch.where(self.ismv, self.g['wbase']+self.g['wscale']*torch.gather(q, 1, self.slot), win)

    def _rows(self, val, dv=None):
        ex = self.ex
        out = torch.zeros(ex.n, ex.slow.ncv, dtype=val.dtype, device=val.device).scatter(1, ex.frow, val*ex.v['mvm'])
        return out if dv is None else out.scatter(1, ex.dvrow[:, None], (dv*ex.common)[:, None])

    def y(self, x, win, pb):
        xp, s, _ = self._split(x)
        return self.M.y(xp, self._inputs(win, s['qd']), pb)

    def advance(self, x, win, pb):
        ex, v, view = self.ex, self.ex.v, self.ex.fast
        xp, s, pm = self._split(x)
        k0 = int(round(float(x[0, self.nxp])))
        mvm, band = v['mvm'], BAND
        tgt = ((win-self.g['wbase'])/self.g['wscale'].clamp_min(1e-30)).gather(1, self.fidx)*mvm      # the flows asked for, held over the decision
        c, a, xs, qd, dou, ym, ymp, cp, tg, tprev, since = (s[q] for q in STATE)
        pmy, pmp = pm[:, 0], pm[:, 1]
        pos = ex.opening(tgt) if self.ctl is None else None       # 'direct': the slow MV is the opening, through the valve's nominal curve
        for j in range(ex.m):
            kf = k0+j
            if kf == 0:
                self._zero(); self.tr = []
                if self.ctl is not None:
                    self.ctl.reset(view)
            moved = (tgt-tg).abs() > 1e-7
            tprev, since = torch.where(moved, tg, tprev), torch.where(moved, torch.zeros_like(since), since)
            sat = (c >= 1-1e-6).to(c.dtype)-(c <= 1e-6).to(c.dtype)
            if self.ctl is not None:
                fst = dict(target=self._rows(tgt), y=self._rows(ym, pmy), prev_y=self._rows(ymp, pmp), last_target=self._rows(tg), u=c, prev_u=cp, sat=sat*mvm,
                           band=view.c['bandf'], bad=view.c['an_T'], age=view.c['an_T'])
                mv = self.ctl(view, fst, kf, C.observe(view, fst, kf)).to(c.dtype).clamp(-1., 1.)
                c1 = (c+mv*v['slew']*mvm).clamp(0., 1.)
            else:
                c1 = c+torch.minimum(torch.maximum(pos-c, -v['slew']), v['slew'])*mvm
            a1 = v['av']*a+(1-v['av'])*c1
            d = a1-xs
            xs1 = torch.where(d.abs() > v['fs'], a1-torch.sign(d)*v['fd'], xs).clamp(0., 1.)
            self.gen.manual_seed(ex.seed*4_000_037+kf)            # the draws of a tick are the tick's own: the same for every controller, from any state
            z = torch.randn(ex.n, 2*self.nmv+1, generator=self.gen, dtype=torch.float32).to(c.device, c.dtype)
            zp = torch.where(ex.common[:, None], z[:, :1], z[:, :self.nmv])
            dou1 = v['rho']*dou+v['sig']*zp
            dp = (dou1+torch.gather(ex.pv, 2, (ex.pk <= kf).sum(-1, keepdim=True))[:, :, 0]).clamp(-P_MAX, P_MAX)
            if ex.sag > 0:                                        # a sag: what is left of the pressure, the wander on top of it
                dp = (1+dp)*(1-ex.sd*((kf >= ex.sk[:, :1]) & (kf < ex.sk[:, 1:])))-1
            qd1 = (v['bq']*qd+(1-v['bq'])*ex.flow(xs1, dp))*mvm
            xp = self.M.advance(xp, self._inputs(win, (qd+qd1)/2), pb, substeps=1, guard=True)
            ym1 = qd1+v['noise']*z[:, self.nmv:2*self.nmv]*mvm
            pmy, pmp = dp[:, 0]/P_MAX+.01*z[:, -1], pmy
            if ex.sag > 0:
                pmy = pmy.clamp(-1.25, 1.25)                      # the transmitter's range ends there
            # ---- the flows on the true values, against what the valve can have reached
            on = (kf >= self.first).double()
            way = since < v['reach']
            lo = torch.where(way, torch.minimum(tgt, tprev), tgt)-band
            hi = torch.where(way, torch.maximum(tgt, tprev), tgt)+band
            short = ((xs1 >= 1-1e-4) & (qd1 < tgt)) | ((xs1 <= 1e-4) & (qd1 > tgt))
            outside = ((qd1 < lo) | (qd1 > hi)) & ~short & (mvm > 0)
            self.out += outside.sum(-1)*on
            self.rows += mvm.sum(-1)*on
            self.sq += (((qd1-tgt)/band)**2*mvm).sum(-1)*on
            self.sat += (sat.abs()*mvm).sum(-1)*on
            if self.trace is not None:
                i = torch.as_tensor(self.trace, device=c.device)
                self.tr.append(dict(k=kf+1, flow=qd1[i].tolist(), target=tgt[i].tolist(), valve=c1[i].tolist(), stem=xs1[i].tolist(), dp=dp[i].tolist(),
                                    outside=outside[i].int().tolist()))
            c, a, xs, qd, dou, ym, ymp, cp, tg, since = c1, a1, xs1, qd1, dou1, ym1, ym, c, tgt, since+1
        kk = x.new_full((ex.n, 1), float(k0+ex.m))
        return torch.cat([xp, kk, c, a, xs, qd, dou, ym, ymp, cp, tg, tprev, since, pmy[:, None], pmp[:, None]], 1)

    def result(self, failed):
        rows = self.rows.clamp_min(1.)
        res = dict(fscore=torch.where(failed, self.rows, self.out), frows=self.rows, frms=torch.sqrt(self.sq/rows), fsat=self.sat/rows)
        if self.trace is not None:
            res['ftrace'] = self.tr
        return res


class Narrow:
    """A Program read with MV limits inside the task's own (a tier whose valves below are at an end of their
    stroke): C.observe(narrow, st, k) gives the positions and the slew inside them, `move` the move the core
    makes of `a` under them (the core itself keeps the task's limits). At least the room of a pinned MV is
    left."""

    def __init__(self, prog, ulo=None, uhi=None):
        self._p, self._lo, self._hi = prog, ulo, uhi

    def __getattr__(self, q):
        return getattr(self._p, q)

    def task(self, seg):
        t = dict(self._p.task(seg))
        lo, hi = t['ulo'], t['uhi']
        room = torch.minimum(torch.full_like(lo, 2*PR.PIN), hi-lo)
        if self._lo is not None:
            lo = torch.minimum(torch.maximum(lo, self._lo), hi-room)
        if self._hi is not None:
            hi = torch.maximum(torch.minimum(hi, self._hi), lo+room)
        t['ulo'], t['uhi'] = lo, hi
        return t

    def move(self, st, a, k):
        t, c = self.task(self.seg_at(k)), self._p.c
        u0, sl = st['u'], c['slew']
        ut = torch.minimum(torch.maximum(u0+a*sl*c['mvm'], t['ulo']), t['uhi'])
        return torch.minimum(torch.maximum(ut-u0, -sl), sl)/sl*c['mvm']


def narrow(prog, ulo=None, uhi=None):
    return Narrow(prog, ulo, uhi)


@torch.no_grad()
def rollout(ex, slow, fast=None, mode='tiers', steps=HORIZON, trace=None):
    """One controller pair on every episode. The core's result of the slow rows (aio.sim.control.rollout), and of
    the flows: fscore (row-ticks outside, from the first event; a failed episode: all of them), frows (the
    row-ticks of that window), frms (the error in bands), fsat (share of valve-ticks at an end of the stroke),
    ftrace with trace. fast: the flow controller of 'tiers' (the formula PI of every loop without one)."""
    assert mode in MODES, mode
    if mode == 'ideal':
        return CT.rollout(ex.slow, slow, steps, trace)
    lay = Valves(ex, (fast or CT.PI()) if mode == 'tiers' else None, trace)
    with ex.attach(lay):
        res = CT.rollout(ex.slow, slow, steps, trace)
    res.update(lay.result(res['failed']))
    return res


REFS = dict(hold=('hold', None, 'direct'), pi_direct=('pi', None, 'direct'), dmc_direct=('fixed', None, 'direct'), exact_direct=('now', None, 'direct'),
            flow_pi=('hold', 'pi', 'tiers'), pi=('pi', 'pi', 'tiers'), dmc_pi=('fixed', 'pi', 'tiers'), exact=('now', 'flow', 'tiers'))
RULED = dict(dmc_pi_rule=('fixed', 'pi', 'tiers'), exact_rule=('now', 'flow', 'tiers'))   # with the limit rule of aio.tiers between the two (an exam with sags)
BOUND = dict(ideal_dmc=('fixed', None, 'ideal'), ideal_exact=('now', None, 'ideal'))     # the plant without valves: not on the scale's basis


def _ctl(q):
    return {'hold': CT.Hold, 'pi': CT.PI, 'flow': Flow, None: lambda: None}.get(q, lambda: CT.Teacher(q))()


def ruled(slow, fast):
    """The two controllers joined by the limit rule of aio.sim.cascade: the slow one reads its packet, and moves,
    inside limits taken in where the valve under an MV is at an end of its stroke."""
    from aio.sim import cascade as CA
    p = CA.Pair(lambda: slow, fast=lambda: fast)
    return p.slow, p.fast


def basis(ex):
    """The references the exam's scales are of: with sags, also the tiers of references with the limit rule."""
    return {**REFS, **RULED} if ex.sag > 0 else REFS


def references(ex, names=None, log=None):
    """The references on the exam's episodes ({name: rollout result}) and its two scales (ex.scale of the slow
    rows, ex.fscale of the flows): 0 the valves left where they are, 100 per episode the best of the others."""
    out, refs = {}, basis(ex)
    for q, (s, f, mode) in {**refs, **BOUND}.items():
        if names is None or q in names or q in ('hold', 'pi'):
            t0 = time.time()
            a, b = _ctl(s), _ctl(f)
            out[q] = rollout(ex, *(ruled(a, b) if q in RULED else (a, b)), mode)
            if log:
                log(f'{q:11s} {time.time()-t0:6.1f} s')
    per = {q: r['score'].tolist() for q, r in out.items() if q in refs}
    ex.scale = SC.Scale.of(per)
    ex.fscale = SC.Scale.of({q: r['fscore'].tolist() for q, r in out.items() if q in refs})
    return out


def table(ex, out):
    """Rows of the references: name, mode, mean score of the slow rows, points, share of the flow row-ticks outside, points of the flows, failures."""
    rows, refs = [], basis(ex)
    for q, r in out.items():
        sp, fp = ex.points(r) if q in refs else (ex.scale(r['score'].tolist()), None)
        rows.append(dict(name=q, mode={**refs, **BOUND}[q][2], score=float(r['score'].mean()), points=sp, failed=int(r['failed'].sum()),
                         fshare=float(r['fscore'].sum()/r['frows'].sum().clamp_min(1.)) if 'fscore' in r else None, fpoints=fp,
                         at_end=float(r['fsat'].mean()) if 'fsat' in r else None))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--plant', default='qtank')
    ap.add_argument('--fast', type=float, default=FAST)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--sag', type=float, default=0., help='share of the episodes with a sag of the supply pressure (valves reach the end of their stroke)')
    ap.add_argument('--out', default=None, help='json with the table and the per-episode scores (the scale of a later run on the same exam)')
    a = ap.parse_args()
    ex = Exam(a.n, a.seed, a.plant, a.fast, a.device, sag=a.sag)
    out = references(ex, log=print)
    rows, refs = table(ex, out), basis(ex)
    print(f'{a.plant}: {a.n} episodes, seed {a.seed}, tick {a.fast:g} s, {ex.m} ticks a decision, sags on {int((ex.sd > 0).any(-1).sum())} episodes; '
          f'100 = best of {", ".join(ex.scale.basis)} per episode (on the easy side)')
    print(f'{"":12s} {"mode":7s} {"outside":>8s} {"points":>7s} {"flows out":>10s} {"points":>7s} {"at end":>7s} {"failed":>6s}')
    for r in rows:
        f = f'{100*r["fshare"]:9.1f}% {r["fpoints"]:7.1f} {100*r["at_end"]:6.1f}%' if r['fshare'] is not None and r['fpoints'] is not None else f'{"":>10s} {"":>7s} {"":>7s}'
        print(f'{r["name"]:12s} {r["mode"]:7s} {r["score"]:8.1f} {r["points"]:7.1f} {f} {r["failed"]:6d}')
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(dict(plant=a.plant, n=a.n, seed=a.seed, tick=a.fast, m=ex.m, sag=a.sag, scale=ex.scale.meta(), easy=True, table=rows,
                                               per_episode={q: [round(float(x), 1) for x in r['score'].tolist()] for q, r in out.items() if q in refs},
                                               per_episode_flows={q: [round(float(x), 1) for x in r['fscore'].tolist()] for q, r in out.items() if q in refs},
                                               bound={q: [round(float(x), 1) for x in r['score'].tolist()] for q, r in out.items() if q in BOUND})))


if __name__ == '__main__':
    main()

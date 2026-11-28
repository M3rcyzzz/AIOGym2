"""The Oracle: a predictive controller that knows the plant (docs/sim.md, 'Oracle').

  python -m aio.sim.oracle --n 32 --seed 7 --out reports/sim/oracle.json

What it knows: everything about now. The true state, the plant as it is in the segment in force (where a
gradual change is heading included), the loads' levels and their periodic part. What it does not know: any
event before it happens and any random draw to come (the wander of the loads, drift, instrument noise). So it
is causal, and it is what "the best a controller could do if it knew the plant" means here: the 100 of the
scale a learner is read on, and the reference that also exists for a plant with no linear model.

How: nonlinear model-predictive control by shooting through the core itself, in the real-time-iteration
manner (Diehl et al. 2005: one or two improvement steps per decision on a plan that is carried along). At
every `every`-th decision (in between the carried plan is followed; 1 by default: on 256 validation episodes
replanning at every decision scored 6% better than at every second, 222 episodes to 26, for 3.4 times the
time, 11-11) the plan is the MV positions asked for
at the next H decisions. Candidates are the plan
carried from the last decision and the plan of the re-identified teacher (the DMC on the exact local model:
the right answer to first order on fresh information, but tuned with caution), as it is and stretched;
each is run through the plant with the future frozen (`Frozen`: the segment in force stays, the random draws
are at their means; the candidates of every episode run side by side as one batch, `tile`), the best per
episode is kept and improved by a few gradient steps on the same run, and its first move is made. What is minimised is the score itself: the count of the row-decisions outside,
softened over `tau` bands, plus a small share of the Huber cost (a direction far outside, where the count is
flat); the last decisions of the horizon weigh for the stretch behind them.

It is a reference and a source of labels on a share of the states, not a teacher at every state: one episode
costs a few hundred episodes of the plant alone.

The 100 of the scale is not this controller alone. Its plan is the best for a frozen, calm future, and on some
plants (large relative gains) that plan does not hold up in the real one: on 6 of 32 episodes a reference with
less knowledge scored better (baolong, 11-10). The scale's 100 is, per episode, the best of every causal
controller that was run: the Oracle and the references (aio.scale.envelope). The file written here has all of
them per episode, and is what the trainer reads (aio.train --refs).
"""
import argparse, json, time
from pathlib import Path
import torch

from aio.sim import core as C, control as CT, model as MD
from aio.sim.prior import HORIZON


class Frozen:
    """A program as it is known at decision k0: the segment in force stays in force (a gradual change goes
    on to its end), so nothing of a later event is in it. Stepped with the draws at their means (`calm`)."""
    Fr = None                                                     # no teacher prediction along a plan

    def __init__(self, prog, k0):
        self._p, self._seg = prog, prog.seg_at(k0)

    def __getattr__(self, q):
        return getattr(self._p, q)

    def seg_at(self, k):
        return self._seg

    def allowance(self, k):
        p, seg = self._p, self._seg
        t = p.task(seg)
        held = (k < p.hold_k[p.ar, seg])[:, None]
        return torch.where(held, t['blo'], t['alo']), torch.where(held, t['bhi'], t['ahi'])


_MODEL = ('Fr', 'Az', 'wz', 'Sw', 'HX', 'wy', 'Gs', 'Gz', 'zs', 'lab', 'labw', 'phis', 'S0', 'pi', '_teach', 'eff_k', 'start')


def tile(prog, r):
    """The same episodes r times over, as one program of r*n episodes (copy c of episode i at c*n+i): what
    lets the candidates of a decision run as one batch instead of one after the other. The loop is bound by
    the host launching kernels, not by their size, so r candidates then cost about one."""
    n = prog.n

    def rep(x):
        if torch.is_tensor(x) and x.dim() and x.shape[0] == n:
            return x.repeat(r, *[1]*(x.dim()-1))
        if isinstance(x, dict):
            return {q: rep(v) for q, v in x.items()}
        return x
    t = object.__new__(type(prog))
    for q, v in vars(prog).items():                               # the teacher's model is not copied: a plan is run without it
        setattr(t, q, None if q in _MODEL else v if q in ('ffm', 'g', 'mg', 'mech', 'tail') else rep(v))
    t.n, t.ar = r*n, torch.arange(r*n, device=prog.device)
    t.mg = [dict(g, idx=g['idx']+c*n) for c in range(r) for g in prog.mg]      # a mechanistic group once per copy
    return t


def tile_state(st, n, r):
    return {q: v.repeat(r, *[1]*(v.dim()-1)) for q, v in st.items()}


def calm(prog):
    """The draws of a step at their means: no wander, no instrument noise, no spike, no bad value."""
    return (torch.zeros(prog.n, prog.nnorm, dtype=prog.dtype, device=prog.device),
            torch.ones(prog.n, prog.nuni, dtype=prog.dtype, device=prog.device))


def objective(out, st, c, tau=.25, cost_w=.1, log_w=0.):
    """(n,) the score of one decision made smooth, per row: the count of the rows outside softened over `tau`
    bands, plus a direction far outside, where the count is flat: cost_w of the Huber cost (what the Oracle
    plans on) and log_w of log(1 + bands outside). The Huber cost grows with the size of the deviation and so
    weighs a decision 10 bands out five times one just outside, where the score counts both once; the log
    grows 40 times slower (11-11, the trainer's far term: aio.probe). out, st: what core.step returned. What
    the Oracle minimises along a plan and the trainer along a window."""
    ex = torch.maximum(out['alo']-1-out['err'], out['err']-out['ahi']-1)                 # > 0: a tracked row outside
    yt = st['yt']
    zx = torch.maximum(out['zlo']-yt, yt-out['zhi'])/c['bandf']-out['ahi']                # > 0: a zone row outside
    rows = (c['trk']+c['zon']).sum(-1).clamp_min(1.)
    per = (lambda x: torch.sigmoid(x/tau)+log_w*torch.log1p(x.clamp_min(0.))) if log_w else (lambda x: torch.sigmoid(x/tau))
    return ((per(ex)*c['trk']+per(zx)*c['zon']).sum(-1)+cost_w*out['soft'])/rows


def plan_cost(fz, st, k, U, w, tau, cost_w, rnd):
    """(n,) what the plan U (H, n, NMV: the positions asked for at the next H decisions) costs from the state
    st at decision k, with the future frozen: `objective` weighted along the horizon by w (H,)."""
    c = fz.c
    J = torch.zeros(fz.n, dtype=U.dtype, device=U.device)
    for j in range(U.shape[0]):
        a = ((U[j]-st['u'])/c['slew']).clamp(-1., 1.)*c['mvm']
        st, out = C.step(fz, st, a, k+j, rnd=rnd)
        J = J+w[j]*objective(out, st, c, tau, cost_w)
    return J/w.sum()+st['fail'].to(U.dtype)*1e3


class Oracle:
    """ctl(prog, st, k, obs) as the other controllers of aio.sim.control.
    H: decisions planned ahead; iters: gradient steps per decision; tail: (m, d) the last m decisions of the
    horizon stand for d more; gains: how far the teacher's plan is stretched in the candidates."""

    def __init__(self, H=120, iters=1, every=1, tau=.25, cost_w=.1, tail=(12, 60.), gains=(1., 2., 4.), step=1.):
        self.H, self.iters, self.every, self.tau, self.cost_w, self.gains, self.step = H, iters, every, tau, cost_w, gains, step
        self.tail = tail

    def reset(self, prog):
        prog.set_teacher('now')
        self.plan, self.eta = None, torch.full((prog.n,), self.step, dtype=prog.dtype, device=prog.device)
        w = torch.ones(self.H, dtype=prog.dtype, device=prog.device)
        w[-self.tail[0]:] += self.tail[1]/self.tail[0]
        self.w, self.rnd = w, calm(prog)
        self.wide = tile(prog, 1+len(self.gains))                 # every candidate of every episode, side by side
        self.wide_rnd = calm(self.wide)
        mt = torch.tensor(MD.MT, device=prog.device)
        self.idx = (torch.searchsorted(mt, torch.arange(self.H, device=prog.device), right=True)-1).clamp_min(0)
        self.stat = dict(replans=0, accepted=0., tried=0., seed=torch.zeros(1+len(self.gains)))

    def _improve(self, fz, st, k, U, s):
        """Trust-region steepest descent on the plan, per episode: a step is kept where the run got cheaper."""
        def at(V):
            V = V.clone().requires_grad_(True)
            J = plan_cost(fz, st, k, V, self.w, self.tau, self.cost_w, self.rnd)
            g, = torch.autograd.grad(J.sum(), V)
            return J.detach(), g.nan_to_num(0., 0., 0.)
        with torch.enable_grad():
            J, g = at(U)
            for _ in range(self.iters):
                gs = g*s[None]
                d = s[None]*gs/gs.abs().amax((0, 2)).clamp_min(1e-12)[None, :, None]
                Un = (U-self.eta[None, :, None]*d).clamp(0., 1.)
                Jn, gn = at(Un)
                ok = Jn < J
                U, J, g = torch.where(ok[None, :, None], Un, U), torch.where(ok, Jn, J), torch.where(ok[None, :, None], gn, g)
                self.eta = torch.where(ok, self.eta*1.5, self.eta*.5).clamp(1/64, 8.)
                self.stat['accepted'] += float(ok.float().mean()); self.stat['tried'] += 1
        return U

    def __call__(self, prog, st, k, obs):
        c = prog.c
        if self.plan is None:
            self.plan = st['u'][None].repeat(self.H, 1, 1)
        if k % self.every == 0:
            fz = Frozen(prog, k)
            s0 = {q: v.detach() for q, v in st.items() if q != 'tF'}
            with torch.no_grad():
                tp = C.teacher(prog, st, k, plan=True)[:, self.idx].transpose(0, 1)      # (H, n, NMV)
                u0 = st['u'][None]
                cands = [self.plan]+[(u0+g*(tp-u0)).clamp(0., 1.) for g in self.gains]
                R = len(cands)
                J = plan_cost(Frozen(self.wide, k), tile_state(s0, prog.n, R), k, torch.cat(cands, 1), self.w, self.tau, self.cost_w,
                              self.wide_rnd).view(R, prog.n)
                best = J.argmin(0)
                U = torch.stack(cands)[best, :, prog.ar].transpose(0, 1)
                self.stat['seed'] += torch.bincount(best.cpu(), minlength=len(cands)).float()
                K = prog.params(fz._seg)['K'][:, :, :prog.nmv]
                s = 1/(K.abs()/c['bandf'][:, :, None]*c['prc'][:, :, None]).amax(1).clamp_min(1.)*c['mvm']
            self.plan = self._improve(fz, s0, k, U, s) if self.iters else U
            self.stat['replans'] += 1
        a = ((self.plan[0]-st['u'])/c['slew']).clamp(-1., 1.)*c['mvm']
        self.plan = torch.cat((self.plan[1:], self.plan[-1:]))
        return a.detach()


def main():
    from aio import scale as SC
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--mech', type=float, default=0., help='share of the episodes on plants of the engine (as core.build)')
    ap.add_argument('--steps', type=int, default=HORIZON, help='decisions run (fewer: a quick look; the scores are then of that stretch)')
    ap.add_argument('--H', type=int, default=120)
    ap.add_argument('--iters', type=int, default=1)
    ap.add_argument('--every', type=int, default=1, help='decisions between two plans (in between the carried plan is followed; 2: a third of the time, 6%% worse)')
    ap.add_argument('--tau', type=float, default=.25)
    ap.add_argument('--cost-w', type=float, default=.1)
    ap.add_argument('--gains', type=float, nargs='*', default=[1., 2., 4.])
    ap.add_argument('--no-oracle', action='store_true', help='the references alone (minutes): a scale whose 100 is the best reference')
    ap.add_argument('--merge', type=Path, nargs='*', default=[], help='earlier files of the same batch: their controllers join the envelope')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    prog = C.build(a.n, a.seed, a.device, mech=a.mech)
    res = dict(n=a.n, seed=a.seed, mech=a.mech, steps=a.steps, H=a.H, iters=a.iters, every=a.every, tau=a.tau, cost_w=a.cost_w, gains=a.gains,
               device=str(a.device), torch=torch.__version__, score={}, minutes={})
    per = {}
    ora = Oracle(a.H, a.iters, a.every, a.tau, a.cost_w, gains=tuple(a.gains))
    ctls = [('hold', CT.Hold()), ('pi', CT.PI()), ('dmc', CT.Teacher('fixed')), ('teacher', CT.Teacher('late')), ('now', CT.Teacher('now'))]
    for name, ctl in ctls+([] if a.no_oracle else [('oracle', ora)]):
        t0 = time.time()
        r = CT.rollout(prog, ctl, a.steps)
        per[name] = [round(float(x)) for x in r['score']]
        res['score'][name] = round(float(r['score'].mean()), 1); res['minutes'][name] = round((time.time()-t0)/60, 2)
        print(json.dumps({name: res['score'][name], 'minutes': res['minutes'][name], 'failed': int(r['failed'].sum())}), flush=True)
    for f in a.merge:                                             # another run of the same episodes (checked on the formula PI's scores)
        old = json.loads(f.read_text())
        if not SC.same_batch(old['per_episode']['pi'], per['pi']):
            raise SystemExit(f'{f}: not the same episodes (the formula PI scores differ)')
        per.update({f'{q}@{f.stem}': v for q, v in old['per_episode'].items() if q not in SC.NOT_CAUSAL and q not in ('hold', 'pi', 'best')})
    per['best'] = SC.envelope(per)
    res['score']['best'] = round(sum(per['best'])/a.n, 1)
    res['points'] = {q: round(SC.points(v, per['hold'], per['best']), 1) for q, v in per.items()}
    if not a.no_oracle:
        res['oracle_share_of_now'] = round(sum(per['oracle'])/max(sum(per['now']), 1), 3)
        res['oracle_beaten_on'] = sum(o > min(v[i] for q, v in per.items() if q not in ('oracle', 'best', 'hold')) for i, o in enumerate(per['oracle']))
        res['stat'] = dict(accepted=round(ora.stat['accepted']/max(ora.stat['tried'], 1), 3), seed=[round(float(x), 3) for x in ora.stat['seed']/ora.stat['seed'].sum()])
    res['per_episode'] = per
    print(json.dumps({q: v for q, v in res.items() if q != 'per_episode'}))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

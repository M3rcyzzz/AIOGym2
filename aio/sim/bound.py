"""A floor under the score, by trajectory optimisation through the core (diagnostic, docs/sim.md).

  python -m aio.sim.bound --n 32 --seed 7 --iters 300 --out reports/sim/bound.json
  python -m aio.sim.bound --n 32 --seed 7 --iters 300 --causal --out reports/sim/bound-causal.json

The whole MV trajectory of every episode is optimised by gradient through the plant (the step is a pure
function of the state and the moves), starting from the trajectory of the re-identified teacher. The
variables are the MV positions the plan asks for at every decision; the move made is the one towards that
position inside the slew limit, so a change of one variable stays local in time. The optimiser
knows what no controller can: the plant, every event before it happens, and the noise to come. Its score is
therefore not a target for a controller; it is what is left when everything knowable is known, so the part
of the score below it belongs to the task (dead time, slew limits, the reference moving at once) and not
to any controller. The share of the teacher's score above it is what a better controller could still win.

With --causal no gradient passes an event: the plan before an event answers for what happens before it
alone, so it cannot prepare for the event (it still knows the plant and the noise to come, so it stays
optimistic for a controller). The stretch between the two floors is what only foresight of the events
buys; the stretch between the causal floor and the teacher is the most a controller could still win.

What is minimised: a smooth count of the row-decisions outside (the score, with the step of the count
softened over `tau` bands) plus the Huber cost relative to its start value (it gives a direction far outside,
where the smooth count is flat). The best hard score of every episode over the iterations is kept.
"""
import argparse, json, time
from pathlib import Path
import torch

from aio.sim import core as C, control as CT
from aio.sim.prior import HORIZON


def _positions(prog, ctl):
    """The MV positions a controller goes through on every episode, (HORIZON, n, NMV), and its score."""
    st = C.reset(prog); ctl.reset(prog)
    U, score = [], torch.zeros(prog.n)
    with torch.no_grad():
        for k in range(HORIZON):
            a = ctl(prog, st, k, C.observe(prog, st, k))
            st, out = C.step(prog, st, a, k)
            score += out['outside'].sum(-1)*(k >= prog.k_first)
            U.append(st['u'])
    prog.set_teacher()
    return torch.stack(U), score


def run(prog, U, tau, causal=False):
    """One differentiable episode following the planned positions U (HORIZON, n, NMV). Returns the smooth
    count, the Huber cost and the hard score per episode (from the first event), and the failure flag.
    causal: no gradient passes an event, so the plan before an event is shaped by what happens before it
    alone (it cannot prepare for the event)."""
    c = prog.c
    st = C.reset(prog)
    smooth = torch.zeros(prog.n, dtype=U.dtype); soft = torch.zeros(prog.n, dtype=U.dtype); score = torch.zeros(prog.n)
    for k in range(HORIZON):
        cut = (prog.seg_k == k).any(1) if causal else None
        if cut is not None and bool(cut.any()):
            st = {q: torch.where(cut.view(-1, *[1]*(v.dim()-1)), v.detach(), v) if torch.is_tensor(v) and v.requires_grad else v for q, v in st.items()}
        a = ((U[k]-st['u'])/c['slew']).clamp(-1., 1.)*c['mvm']
        st, out = C.step(prog, st, a, k)
        on = (k >= prog.k_first).to(U.dtype)
        ex = torch.maximum(out['alo']-1-out['err'], out['err']-out['ahi']-1)             # > 0: a tracked row outside
        yt = st['yt']
        zx = torch.maximum(out['zlo']-yt, yt-out['zhi'])/c['bandf']-out['ahi']            # > 0: a zone row outside
        smooth = smooth+on*(torch.sigmoid(ex/tau)*c['trk']+torch.sigmoid(zx/tau)*c['zon']).sum(-1)
        soft = soft+on*out['soft']
        score = score+out['outside'].sum(-1).detach()*on
    return smooth, soft, score, st['fail']


def optimise(prog, U0, iters=200, lr=(.3, .03), tau=(.5, .1), cost_w=.1, log=None, causal=False):
    """Optimise the planned positions from U0. The variables are scaled by the plant's gain, so a step of
    the optimiser (Adam; the step and the softening of the count both shrink geometrically) moves the CVs
    by about `lr` bands whatever the plant. Returns the best hard score per episode and its plan."""
    c = prog.c
    K = prog.params(torch.zeros(prog.n, dtype=torch.long, device=prog.device))['K'][:, :, :prog.nmv]
    s = 1/(K.abs()/c['bandf'][:, :, None]*c['prc'][:, :, None]).amax(1).clamp_min(1.)*c['mvm']     # MV span per band of its strongest row
    V = torch.zeros_like(U0).requires_grad_(True)
    opt = torch.optim.Adam([V], lr=lr[0])
    best, bestU, soft0 = None, U0.clone(), None
    rows = (c['trk']+c['zon']).sum(-1).clamp_min(1.)
    for it in range(iters):
        w = it/max(iters-1, 1)
        tk = tau[0]*(tau[1]/tau[0])**w
        for g in opt.param_groups:
            g['lr'] = lr[0]*(lr[1]/lr[0])**w
        U = U0+V*s[None]
        smooth, soft, score, fail = run(prog, U, tk, causal)
        score = torch.where(fail, torch.full_like(score, float('inf')), score)
        if soft0 is None:
            soft0 = soft.detach().clamp_min(1e-6)
        if best is None:
            best = score.clone()
        better = score < best
        best = torch.where(better, score, best)
        bestU = torch.where(better[None, :, None], U.detach(), bestU)
        loss = (smooth/rows/HORIZON+cost_w*soft/soft0).sum()
        opt.zero_grad(); loss.backward()
        V.grad.nan_to_num_(0., 0., 0.)
        opt.step()
        if log and (it % 10 == 0 or it == iters-1):
            log(dict(iter=it, tau=round(tk, 3), score=round(float(score[torch.isfinite(score)].mean()), 1), best=round(float(best.mean()), 1)))
    return best, bestU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--quiet', action='store_true', help='no field layer: the floor of the events alone')
    ap.add_argument('--causal', action='store_true', help='no gradient passes an event: the plan cannot prepare for one')
    ap.add_argument('--threads', type=int)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.time()
    prog = C.build(a.n, a.seed, quiet=a.quiet)
    ref = {}
    for name, ctl in (('pi', CT.PI()), ('dmc', CT.Teacher('fixed')), ('teacher', CT.Teacher('late')), ('now', CT.Teacher('now'))):
        U, s = _positions(prog, ctl)
        ref[name] = s
    prog.Fr = None                                                # the teacher's prediction is not needed from here on
    best, _ = optimise(prog, U, a.iters, log=lambda d: print(json.dumps(d), flush=True), causal=a.causal)
    rows = (prog.c['trk']+prog.c['zon']).sum(-1)
    window = (HORIZON-prog.k_first).clamp_min(0)*rows
    res = dict(n=a.n, seed=a.seed, iters=a.iters, quiet=a.quiet, causal=a.causal, minutes=round((time.time()-t0)/60, 1),
               score={q: round(float(v.mean()), 1) for q, v in ref.items()}, bound=round(float(best.mean()), 1),
               share_of_window={q: round(float((v/window).mean()), 4) for q, v in dict(ref, bound=best).items()},
               floor_share_of_teacher=round(float(best.sum()/ref['now'].sum()), 3),
               per_episode=dict(now=[round(float(v)) for v in ref['now']], bound=[round(float(v)) for v in best],
                                nm=[int(v) for v in prog.c['nm']], T=[round(float(v), 1) for v in prog.c['T']]))
    print(json.dumps({q: v for q, v in res.items() if q != 'per_episode'}))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

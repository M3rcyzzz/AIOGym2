"""The plain law inside the teacher (11-11).

The teacher (aio.sim.core.teacher) plans eight moves over 240 decisions under the MV limits and the zone
limits. Measured on a validation batch (64 episodes, the MVs left alone 0, the teacher 100): without the zone
limits in the plan 98.9, without the MV limits either (the move clamped afterwards) 98.8, and with the
predicted errors known at the nine label lags alone 98.6. So nearly all of the teacher is

    move = K . E / V

E (NLAB, NMV): for every tracked row, how far it will be from where it can settle at the label lags if the
               MVs stop now, in bands, weighted as the teacher weighs it: the after-effect of the moves made
               so far and of the disturbances, on the teacher's model (`heading`);
K (NMV, NLAB, NMV): the first change of every MV's flow per unit of that error, of the unconstrained plan
               (`table`): the model's gains solved for all MVs at once, with the teacher's move suppression.
               It is of a segment's model: it stays as it is until the plant changes;
V (NMV,):      the flow a move of one slew limit makes at the valve's position now (`reach`): the valve curve.

The move is then held to the slew limit and the operator limits (`held`). Its difference from the teacher's own
move at the same state is not small (11-11: rms .13 against the teacher's .12; the plan's limits make the
teacher's first move smaller), yet in closed loop it costs 1.4 points: many moves do the same job. The law's
move is a smooth function of E, K and V, which a network that has them can make exactly, so it is the better
thing to imitate.

A student that imitates the teacher's moves has to come by both inside itself; here they are worked out so
that it can be shown them (told, on the episodes that are told the plant) and taught to read them (its
readouts' targets on every episode). Neither is given to it when it runs.
"""
import torch

from aio.sim import core as C, model as MD

NLAB = MD.NLAB


def spread(P=MD.P_DMC, dtype=torch.float64, device='cpu'):
    """(P, NLAB): errors at the label lags -> at every lag 1..P, linear in between."""
    ll = torch.tensor(MD.LAB_LAGS, device=device)
    j = torch.arange(1, P+1, device=device)
    hi = torch.searchsorted(ll, j).clamp(1, NLAB-1)
    a, b = ll[hi-1], ll[hi]
    w = ((j-a).to(dtype)/(b-a).to(dtype)).clamp(0., 1.)
    B = torch.zeros(P, NLAB, dtype=dtype, device=device)
    B[j-1, hi-1] += 1-w
    B[j-1, hi] += w
    return B


@torch.no_grad()
def table(Sw, HX, chunk=1024):
    """(m, NMV, NLAB, NMV): the first planned change of every MV's flow per unit of the weighted error of every
    tracked row at every label lag, of the plan of aio.sim.model.dmc_move without limits (Sw, HX of dmc_design)."""
    m, P, NMV = Sw.shape[:3]
    B = spread(P, device=Sw.device)
    out = torch.zeros(m, NMV, NLAB, NMV, dtype=Sw.dtype, device=Sw.device)
    for a in range(0, m, chunk):
        s = Sw[a:a+chunk].double()
        g = torch.stack([torch.einsum('btri,tl->bilr', s[:, :P-t], B[t:]) for t in MD.MT], 1)      # (b, M, NMV, NLAB, rows)
        g = (g-torch.cat((g[:, 1:], torch.zeros_like(g[:, :1])), 1)).reshape(s.shape[0], -1, NLAB*NMV)
        H = HX[a:a+chunk].double()
        idle = H.diagonal(dim1=1, dim2=2).abs().sum(-1) == 0      # a segment not in use has no model: its table stays 0
        H = H+idle[:, None, None]*torch.eye(H.shape[-1], dtype=H.dtype, device=H.device)
        X = MD.spd_solve(H, g)                                    # as the teacher's own solve (LU at the default size, Cholesky above it); HX is kept in single
                                                                   # precision, and what rounding leaves of a stiff model need not be positive definite to the last bit (no table: 0)
        out[a:a+chunk] = X[:, :NMV].reshape(-1, NMV, NLAB, NMV).to(Sw.dtype)
    return out


def tables(prog):
    """(n, NSEG, NMV, NLAB, NMV) of a Program: `table` of every segment's model (kept on the program)."""
    if getattr(prog, 'law_K', None) is None:
        n, S = prog.Sw.shape[:2]
        prog.law_K = table(prog.Sw.flatten(0, 1), prog.HX.flatten(0, 1)).unflatten(0, (n, S))
    return prog.law_K


@torch.no_grad()
def heading(prog, st, k, view=None):
    """E (n, NLAB, NMV rows) at decision k and the teacher's view it is of (core.teacher_view)."""
    v = C.teacher_view(prog, st, k) if view is None else view
    lags = torch.tensor(MD.LAB_LAGS, device=prog.device)-1
    return v['E'][:, lags]*v['r'].clamp_max(1.)[:, None, None], v


@torch.no_grad()
def reach(prog, v):
    """V (n, NMV): the flow a move of one slew limit makes at the valve positions now, taken across the travel
    within reach either way (a valve curve is steep at one end); 1 on the slots not in use."""
    c, u, g, sl = prog.c, v['u'], v['g'], v['sl']
    per = (C._flow((u+sl).clamp(0., 1.), g, c['mvm'])-C._flow((u-sl).clamp(0., 1.), g, c['mvm']))/2
    return torch.where(c['mvm'] > 0, per.clamp_min(1e-3), torch.ones_like(per))


@torch.no_grad()
def move(prog, st, k, view=None):
    """-> the law's move (n, NMV) in slew limits, not held to them; E; K (n, NMV, NLAB, NMV) of the model the
    teacher has now; V. view: core.teacher_view of this decision where the caller has it."""
    E, v = heading(prog, st, k, view)
    K, V = tables(prog)[prog.ar, v['sg']]*prog.c['mvm'][:, :, None, None], reach(prog, v)
    return torch.einsum('njlr,nlr->nj', K, E)/V, E, K, V


def held(prog, st, k, m):
    """The move m (n, NMV; slew limits) held to the slew limit and the operator limits, as the plant takes it."""
    u, sl, t = st['u'], prog.c['slew'], prog.task(prog.seg_at(k))
    un = torch.minimum(torch.maximum(u+m.clamp(-1., 1.)*sl, torch.minimum(t['ulo'], u)), torch.maximum(t['uhi'], u))
    return ((un-u)/sl).clamp(-1., 1.)*prog.c['mvm']


class Plain:
    """The law alone as a controller (aio.sim.control): its move clamped to the slew limit and to the operator
    limits. How much of the teacher it is: python -m aio.law."""

    def __init__(self, mode='late', loads=None):
        self.mode, self.loads = mode, (mode != 'fixed') if loads is None else loads

    def reset(self, prog):
        prog.set_teacher(self.mode, self.loads)

    def __call__(self, prog, st, k, obs):
        return held(prog, st, k, move(prog, st, k)[0])


if __name__ == '__main__':
    import argparse, json, time
    from aio.sim import control as CT
    from aio import scale as SC
    ap = argparse.ArgumentParser(description='How much of the teacher the plain law is, on one batch (the MVs left alone 0, the teacher 100)')
    ap.add_argument('--n', type=int, default=64); ap.add_argument('--seed', type=int, default=1_000_000)
    ap.add_argument('--device', default='cpu'); ap.add_argument('--mech', type=float, default=0.)
    a = ap.parse_args()
    prog = C.build(a.n, a.seed, a.device, mech=a.mech)
    per = {}
    for q, ctl in dict(hold=CT.Hold(), pi=CT.PI(), dmc=CT.Teacher('fixed'), teacher=CT.Teacher('late'), law=Plain()).items():
        t0 = time.time()
        per[q] = CT.rollout(prog, ctl)['score'].tolist()
        print(json.dumps(dict(ctl=q, score=round(sum(per[q])/a.n, 1), s=round(time.time()-t0, 1))), flush=True)
    print(json.dumps({q: round(SC.points(v, per['hold'], per['teacher']), 1) for q, v in per.items()}))

"""Reference controllers and rollouts on the simulation core.

A controller is called as ctl(prog, st, k, obs) and returns moves (n, NMV) in [-1, 1] of the slew limit in
the plant's frame. `rollout` runs one on every episode of a program and scores it by the
decisions a tracked CV is outside its band of the reachable
reference (at a constrained operating point: beyond a band past the stretch between the target and the
best reachable steady value) plus decisions a zone CV is outside its limits, on the true values, from the
first event on; a physical failure counts the whole window on every such row. `after` is the part of that score
from the episode's first process change on (`changed_k`), `before` the rest: whether a controller still holds
the plant once the plant is no longer the one it was commissioned on.

  Hold      no moves
  PI        lambda-tuned PI on the published pairing, tuned once on the exact step response of the start
            plant (the formula reference: an auto-tuner after a commissioning step test)
  Teacher   step-response DMC with a steady-state target layer on the exact local model; mode 'late' (the
            training teacher: its model follows a process event 10-40 decisions late and it sees the
            unmeasured loads move), 'now' (the same, re-identified at every event: upper reference),
            'fixed' (the commissioning model and no sight of the loads: a step-test DMC, the engineering
            reference)
  Policy    the earlier actor alone (aio.actor.Actor), deterministic, direct-acting frame;
            the network with the pair pathway is driven by aio.net.Policy
"""
import torch

from aio.sim import core as C
from aio.sim.prior import BIG, BOTH, HORIZON, PROC


class Hold:
    def reset(self, prog):
        pass

    def __call__(self, prog, st, k, obs):
        return torch.zeros(prog.n, prog.nmv, dtype=prog.dtype, device=prog.device)


class PI:
    e = None

    def reset(self, prog):
        self.e = None

    def resume(self, prog):                                        # the next window on the same plants: the loops stay closed
        pass

    def restart(self, f):                                          # units started again at rest: their loops start afresh
        if self.e is not None:
            self.e = torch.where(f[:, None], torch.full_like(self.e, float('nan')), self.e)

    def __call__(self, prog, st, k, obs):
        pi, c = prog.pi, prog.c
        e = torch.gather(st['target']-st['y'], 1, pi['row'])
        e0 = e if self.e is None else torch.where(torch.isnan(self.e), e, self.e)
        self.e = e
        du = pi['kc']*((e-e0)+e/pi['ti'])
        return (du/c['slew']).clamp(-1., 1.)*c['mvm']


class Teacher:
    def __init__(self, mode='late', loads=None, meter=None):
        self.mode, self.loads, self.meter = mode, (mode != 'fixed') if loads is None else loads, meter      # meter: core.Program.set_teacher (the instruments alone)

    def reset(self, prog):
        prog.set_teacher(self.mode, self.loads, self.meter)

    resume = reset

    def __call__(self, prog, st, k, obs):
        return C.teacher(prog, st, k)


class Policy:
    """A trained actor on the packet (deterministic mean action; `canonical`: direct-acting frame)."""

    def __init__(self, actor, canonical=True):
        self.actor, self.canonical = actor, canonical

    def reset(self, prog):
        from aio.actor import STATE
        self.h = torch.zeros(1, prog.n, STATE, device=prog.device)
        self.prev = torch.zeros(prog.n, prog.nmv, device=prog.device)

    @torch.no_grad()
    def __call__(self, prog, st, k, obs):
        from aio.frame import canonical
        o = obs.float()
        o, s = canonical(o) if self.canonical else (o, torch.ones(prog.n, prog.nmv, device=prog.device))
        joint, mem, mask, self.h = self.actor.history(o[:, None], self.prev[:, None], self.h)
        mean, _, _ = self.actor.distribution(joint, mem, mask)
        a = torch.tanh(mean[:, 0])*mask[:, 0]
        self.prev = a
        return (a*s).to(prog.dtype)


@torch.no_grad()
def changed_k(prog):
    """(n,) the decision of an episode's first process change (a 'process' or 'both' event: gains, an actuator's
    capacity, time constants, dead times, a valve's curve or friction; from there on the plant is not the one a
    commissioning model describes); BIG where the episode has none."""
    proc = (prog.kinds == PROC) | (prog.kinds == BOTH)
    return torch.where(proc, prog.seg_k, torch.full_like(prog.seg_k, BIG)).min(1).values


def rollout(prog, ctl, steps=HORIZON, trace=None, st=None):
    """Run a controller on every episode. Returns per-episode tensors: score (decisions outside, from the
    first event), cost (same window), failed; after and before, the score's parts from the first process change
    on and up to it (an episode without one: all of it before); per episode and segment: seg_settled, the row-decisions
    outside once the unit can have lined out after the event that opened the segment (prog.settled_k: the
    event's ramp, the valve travel and the unit's settling time), and seg_rows, the row-decisions of that
    window (none in a segment the next event cuts short); seg_rest and seg_rest_rows, the same of the
    resting rows alone (the spare MVs of a fat plant); trace: list of env indices whose trajectories to
    record. st: the state the window before ended in (prog made by `Program.next`; a controller with `resume`
    keeps what it holds); the state this one ends in comes back as `st`, and `restarted` says which units
    had failed in the window before and start this one at rest (a controller with `restart` is told).
    tripped, trip_k: whether an output of a plant of the engine went past a shutdown limit the plant declares
    (Tennessee Eastman's interlocks), and after which decision first (BIG: never); reported, not scored."""
    go_on = st is not None and hasattr(ctl, 'resume')
    restarted = torch.zeros(prog.n, dtype=torch.bool, device=prog.device) if st is None else st['fail'].clone()
    st = C.reset(prog) if st is None else C.resume(prog, st)
    (ctl.resume if go_on else ctl.reset)(prog)
    if go_on and hasattr(ctl, 'restart') and bool(restarted.any()):
        ctl.restart(restarted)
    n, dev = prog.n, prog.device
    score = torch.zeros(n, device=dev); cost = torch.zeros(n, dtype=torch.float64, device=dev)
    after, kc = torch.zeros(n, device=dev), changed_k(prog)       # (a process change is an event: never before the first one)
    rows = (prog.c['trk']+prog.c['zon']).sum(-1)
    rst = prog.c['rst'] > 0
    nseg = prog.seg_k.shape[1]
    seg_settled, seg_rows = torch.zeros(n, nseg, device=dev), torch.zeros(n, nseg, device=dev)
    seg_rest, seg_rest_rows = torch.zeros(n, nseg, device=dev), torch.zeros(n, nseg, device=dev)
    tr = []
    trip_k = torch.full((n,), BIG, dtype=torch.long, device=dev)
    for k in range(steps):
        obs = C.observe(prog, st, k)
        a = ctl(prog, st, k, obs)
        st, out = C.step(prog, st, a, k)
        on = (k >= prog.k_first)
        outside = out['outside'].sum(-1)
        score += outside*on
        trip_k = torch.where(out['trip'] & (trip_k == BIG), torch.full_like(trip_k, k+1), trip_k)
        after += outside*(k >= kc)
        cost += out['cost'].double()*on
        seg = prog.seg_at(k)
        late = k >= prog.settled_k[prog.ar, seg]
        seg_settled[prog.ar, seg] += outside*late
        seg_rows[prog.ar, seg] += rows*late
        seg_rest[prog.ar, seg] += (out['outside'] & rst).sum(-1)*late
        seg_rest_rows[prog.ar, seg] += rst.sum(-1)*late
        if trace is not None:
            i = torch.as_tensor(trace, device=dev)
            band = prog.c['bandf'][i]
            tr.append(dict(k=k+1, y=st['yt'][i].tolist(), ym=st['y'][i].tolist(), target=st['target'][i].tolist(),
                           ref=out['ref'][i].tolist(), u=st['u'][i].tolist(), outside=out['outside'][i].int().tolist(),
                           ok_lo=(out['ref'][i]+(out['alo'][i]-1)*band).tolist(), ok_hi=(out['ref'][i]+(out['ahi'][i]+1)*band).tolist()))
    window = (HORIZON-prog.k_first).clamp_min(0)*rows
    score = torch.where(st['fail'], window.to(score.dtype), score)
    after = torch.where(st['fail'], ((HORIZON-kc).clamp_min(0)*rows).to(after.dtype), after)
    res = dict(score=score, after=after, before=score-after, cost=cost, failed=st['fail'], tripped=trip_k < BIG, trip_k=trip_k, seg_settled=seg_settled, seg_rows=seg_rows, seg_rest=seg_rest, seg_rest_rows=seg_rest_rows,
               st=C.detach(st), restarted=restarted)
    if trace is not None:
        res['trace'] = tr
    if prog.Fr is not None:                                       # (a batch built without the teacher has no model to set back)
        prog.set_teacher()
    return res

"""Mechanistic episodes of the simulation core (docs/sim.md).

A share of the episodes of a Program runs on plants of the mechanism engine (aio.sim.engine) in place of the
linear pair bank. The shell around the plant is the same for both: MV limits and slew, positioner, friction
and valve curve, loads that wander, instruments, events, the cost against a reachable reference, the
teacher, the labels. A mechanistic episode takes its row values from the engine's states, and everything
the scenario generator and the teacher need to know about the plant comes from generic tools on its table
(steady states, the linearisation, the decision map), never from code of the plant.

In shell units an MV is 0..1 of its physical range (the valve flow), a load is its nominal value plus the
disturbance level (-1..1) times a third of its range, a row is 0..1 of the output's range. At every
operating point of an episode the plant is described to the scenario generator by its local model there,
in the generator's own terms (prior.py): steady gains K per unit of valve flow and of load level, and a
bias that makes the linear steady equations exact at that point. Events are proposed on the local model
with the generator's own functions, then the true steady state of the proposal is solved on the plant, the
local model is taken again there and the one check of every operating point (prior.operable, or the
verified constrained optimum) is applied to it. A proposal that fails is redrawn smaller, as for every
other plant: no operating point is accepted on the strength of a linear extrapolation.

A constrained point is the optimum of the true plant: the generator's optimum on the local model and the
plant's steady state with the valves there are iterated until they agree (Gauss-Newton on the plant).

What is added around the plant, because lumped models have none: transport delay on the inputs (a share of
the episodes), on top of the instruments' sampling and lags that every episode has.
"""
import math
import torch

from aio.sim import prior as PR, plants as PL
from aio.sim.engine import Mech, one
from aio.sim.prior import DMAX, HORIZON, NSEG, BIG, CARRIED, F

TINT = 100.                      # an integrating row is published as a level filled by its rate over 100 of the plant's own decision intervals (t['tint'] decisions of the episode)
BUCKET = 300                     # a plant with more table nodes than this runs in a batch of its own
DEAD_P, DEAD_MAX = .5, 8         # share of the episodes with transport delay on the inputs; the longest (decisions)
PACE = (.5, 1.)                  # an episode's decision interval over the plant's own: the same plant is met at more than one clock. Shorter
                                 # costs nothing; with an upper end above 1 every episode takes a decision in twice the substeps (engine.Mech `more`)
PACE_LEVEL = .8                  # (no longer than leaves the plant's fastest level, at its nominal point, this share of prior.GAIN_MAX per 100 decisions)
VALVE = dict(curve=.3, lag=.5, stick=.4, wear=.25)              # share of an episode's MVs with a curved valve, a positioner lag, friction; of its process events that wear the
                                 # valves (an exam that models the valve itself has none of them in the shell: aio.sim.rates)
START_ROUNDS = 40                # redraws of a start point (of the episodes still without one) before the nominal one is taken
TASK_ROUNDS = 8                  # an episode whose task has no operable start (the outputs to hold, the valves' curves, the clock: not even at the nominal point) is given another task, this many times
EXC_ROUNDS, EXC_TOL, EXC_BAND = 8, 1e-5, 5e-4                    # rounds of the constrained optimum on the true plant; agreement asked of the valve flows, and of the plant's tracked rows with the reachable targets (bands)
LAGS = 240
WALK = dict(iters=12, minstep=1/8)                               # effort on the way to a proposed operating point: one out of reach is redrawn, not chased
MARGIN_MOVE, MARGIN_STEP, MARGIN_RATIO = 5., .1, 4.              # the finite move of a valve the local model has to hold over (bands of its row; share of its span), and how far off it may be
REACH_WAIT, REACH_SETTLE, REACH_MAX = .5, 3., 300                # the plain operator of the reach check: waits (unit time scales), is given to settle, the longest run (decisions)
REACH_TOL, REACH_SHARE, REACH_EVERY = 2., .25, 4                 # arrived: within this many bands, or this share of the largest distance on the way; looked at every so many decisions
REACH_PACK = .8                  # the runs still going are packed together once they are this share of the ones stepped, or fewer
FOLLOW = 1e-6                    # an output follows a level of its plant: under a step it keeps moving by more than this (bands a decision per full span of an MV)
_POOL, _PACE = {}, {}


def pool(names=None, heldout=False):
    """The plants an episode can be on (built once per process). By default the plants of `PLANTS`; a large one
    (`LARGE`) is in only when named; a held-out one (`HELDOUT`) only when named with heldout=True (evaluation)."""
    names = tuple(names or PL.PLANTS)
    for q in names:
        assert heldout or q not in PL.HELDOUT, f'{q} is held out: no run trains on it (heldout=True to evaluate)'
        if q not in _POOL:
            _POOL[q] = (PL.PLANTS.get(q) or PL.LARGE.get(q) or PL.HELDOUT[q])()
    return [_POOL[q] for q in names]


def _levels(q):
    """Of a plant at its nominal point: the longest decision its fastest integrating output allows (in the
    plant's own intervals), and whether an output that does not integrate follows one that does (it keeps
    moving, by more than FOLLOW bands a decision per full span of an MV, for as long as a level does)."""
    if q.name not in _PACE:
        M = Mech([q], n=1, dtype=F)
        Gr = M.gains(M.x0, M.wnom, M.pnom)[5][0]
        rate = lambda integ, per: max([per*abs(float(Gr[j, i]))*(w['hi']-w['lo'])/o['band'] for j, o in enumerate(q.ys) if bool(o.get('integ')) == integ and not o.get('dv')   # noqa: E731
                                       for i, w in enumerate(q.ws) if w['kind'] == 'mv'], default=0.)
        g = rate(True, 100)
        _PACE[q.name] = (PACE_LEVEL*PR.GAIN_MAX/g if g > 0 else math.inf, g > 0 and rate(False, 1) > FOLLOW)
    return _PACE[q.name]


def pace_max(plants):
    """(P,) the longest decision of an episode on each plant, in the plant's own intervals: PACE[1], or what the
    plant's fastest integrating output allows (its rate per full span of an MV, in bands per 100 decisions)."""
    return torch.tensor([min(PACE[1], _levels(q)[0]) for q in plants], dtype=F)


def _meta(plants, nw, ny):
    """Per plant, padded: what every input and output is to the shell."""
    P = len(plants)
    z = lambda m, v=0.: torch.full((P, m), v, dtype=F)            # noqa: E731
    t = dict(ikind=torch.full((P, nw), 3), ilo=z(nw), ihi=z(nw, 1.), inom=z(nw), islew=z(nw, 1.),
             ouse=torch.zeros(P, ny, dtype=torch.bool), otrk=torch.zeros(P, ny, dtype=torch.bool), oint=torch.zeros(P, ny, dtype=torch.bool),
             oby=torch.full((P, ny), -1),
             rlo=z(ny), rhi=z(ny, 1.), oband=z(ny, 1.), tlo=z(ny), thi=z(ny, 1.), olo=z(ny, -math.inf), ohi=z(ny, math.inf),
             plo=z(max(max(q.np for q in plants), 1)), phi=z(max(max(q.np for q in plants), 1)), pev=z(max(max(q.np for q in plants), 1)),
             dt=torch.tensor([q.dt for q in plants], dtype=F), olev=torch.tensor([_levels(q)[1] for q in plants]))
    for i, q in enumerate(plants):
        for j, w in enumerate(q.ws):
            t['ikind'][i, j] = 0 if w['kind'] == 'mv' else (1 if w.get('measured') else 2)
            t['ilo'][i, j], t['ihi'][i, j], t['inom'][i, j] = w['lo'], w['hi'], w['nom']
            t['islew'][i, j] = w.get('slew', 1.)
        for j, o in enumerate(q.ys):
            if o.get('dv'):                                       # a measured load is published by the shell
                continue
            t['ouse'][i, j], t['otrk'][i, j], t['oint'][i, j] = True, bool(o.get('track')), bool(o.get('integ'))
            t['rlo'][i, j], t['rhi'][i, j] = o['range']
            t['oband'][i, j] = o['band']
            t['tlo'][i, j], t['thi'][i, j] = o.get('target', o['range'])
            t['olo'][i, j], t['ohi'][i, j] = o.get('lo', -math.inf), o.get('hi', math.inf)
            if o.get('by'):                                       # a level the unit was built to hold with this MV
                t['oby'][i, j] = [w['name'] for w in q.ws].index(o['by'])
        for j, s in enumerate(q.ps):
            t['plo'][i, j], t['phi'][i, j], t['pev'][i, j] = s['lo'], s['hi'], float(s.get('event', True))
    return t


class Group:
    """The episodes of one batch of plants: how each plant sits in the shell (drawn once per episode), and
    the generator's tools on it. All float64 on the CPU."""

    def __init__(self, M, t, dims=PR.DIMS):
        self.M, self.t, self.k = M, t, M.n
        self.size, self.dims = tuple(dims), PR.sizes_of(*dims)    # (NCV, NMV, ND) of the Program; and all six sizes

    @classmethod
    def draw(cls, r, plants, pid, dims=PR.DIMS):
        """The task of every episode: which slot each MV and load takes, which outputs are tracked (as many as
        there are MVs, integrating ones first), which are zone rows; in random order, so that no slot or row
        number tells the plant."""
        NCV, NMV, ND, NQ, NDQ, NU = PR.sizes_of(*dims)
        pace = PACE[0]*(pace_max(plants)[pid].clamp_min(PACE[0])/PACE[0])**r.u(pid.numel())
        M = Mech(plants, pid, dtype=F, pace=pace, more=2 if PACE[1] > 1 else 1)
        k, nw, ny = M.n, M.nw, M.ny
        t = {q: v[pid] for q, v in _meta(plants, nw, ny).items()}
        t['pid'], t['pace'], t['tint'] = pid.clone(), pace, TINT/pace
        t['dt'], t['islew'] = t['dt']*pace, t['islew']*pace[:, None]          # the interval the packet reports; a valve's travel in one decision
        rank = lambda mask, first=None: torch.argsort(torch.argsort(r.u(k, mask.shape[1])+10.*(~mask)-(0 if first is None else 5.*first), -1), -1)   # noqa: E731
        ismv, ism, isq = t['ikind'] == 0, t['ikind'] == 1, t['ikind'] == 2
        smv, sm, sq = rank(ismv), rank(ism), rank(isq)
        nm = ismv.sum(1)
        assert int(nm.max()) <= NMV and int(nm.min()) >= 1, f'a plant with {int(nm.max())} MVs does not fit a Program of {NMV}'
        chan = torch.full((k, nw), NU)
        chan = torch.where(ismv, smv, chan)
        chan = torch.where(ism & (sm < ND), NMV+sm, chan)
        chan = torch.where(isq & (sq < NQ), NMV+ND+sq, chan)
        nd, nq = (ism & (sm < ND)).sum(1), (isq & (sq < NQ)).sum(1)
        strk = rank(t['otrk'], t['oint'])
        chosen = t['otrk'] & (strk < nm[:, None])
        assert bool((chosen.sum(1) == nm).all()), 'a plant needs a tracked output for every MV'
        assert not bool((t['oint'] & ~chosen).any()), 'an integrating output has to be tracked'
        row = torch.argsort(torch.argsort(r.u(k, ny)+10.*(~chosen), -1), -1)        # tracked rows in random order
        zc = t['ouse'] & ~chosen & (torch.isfinite(t['olo']) | torch.isfinite(t['ohi']))
        sz = rank(zc)
        zsel = zc & (sz < torch.minimum(torch.full_like(nm, PR.nzmax(*dims)), NCV-nm-nd)[:, None])
        nz = zsel.sum(1)
        orow = torch.full((k, ny), NCV)
        orow = torch.where(chosen, row, orow)
        orow = torch.where(zsel, nm[:, None]+sz, orow)
        span = t['ihi']-t['ilo']
        half = torch.minimum(t['ihi']-t['inom'], t['inom']-t['ilo'])/1.5
        used = chan < NU
        t.update(chan=chan, orow=orow, nm=nm, nz=nz, nd=nd, nq=nq,
                 wbase=torch.where(ismv, t['ilo'], t['inom']), wscale=torch.where(ismv, span, half)*used, rspan=t['rhi']-t['rlo'])
        ar = torch.arange(k)
        fidx, ridx = torch.zeros(k, NMV, dtype=torch.long), torch.zeros(k, NMV, dtype=torch.long)   # plant input of MV slot j, plant output of tracked row j
        for j in range(nw):
            m = ismv[:, j]
            fidx[ar[m], smv[m, j]] = j
        for j in range(ny):
            m = chosen[:, j]
            ridx[ar[m], row[m, j]] = j
        t.update(fidx=fidx, ridx=ridx, fm=torch.arange(NMV)[None] < nm[:, None])
        slot = torch.where(t['oby'] >= 0, torch.gather(chan, 1, t['oby'].clamp_min(0)), torch.full_like(t['oby'], -1))
        t['rby'] = torch.full((k, NCV+1), -1).scatter(1, orow, slot)[:, :NMV]      # the MV slot a tracked row names (a level: plant.out(by=))
        # transport delay on the inputs: none on half the episodes, else up to DEAD_MAX decisions per MV
        dmax = torch.where(r.bern(DEAD_P, k), r.logu(k, lo=1., hi=DEAD_MAX+.99), torch.zeros(k, dtype=F))
        t['din'] = torch.where(ismv, (r.u(k, nw)*(dmax[:, None]+1)).floor().clamp(0, DMAX), torch.zeros(k, nw, dtype=F)).long()
        return cls(M, t, dims)

    def view(self, idx):
        return Group(self.M.sub(idx), {q: v[idx] for q, v in self.t.items()}, self.size)

    # ---- between the plant's and the shell's units
    def rows(self, val, fill=0.):
        """(k, ny) values of the outputs as (k, NCV) values of the rows."""
        NCV, NMV, ND, NQ, NDQ, NU = self.dims
        out = torch.full((self.k, NCV+1), fill, dtype=F)
        return out.scatter(1, self.t['orow'], val.to(F))[:, :NCV]

    def chans(self, val, fill=0.):
        """(k, nw) values of the inputs as (k, NU) values of the shell's channels."""
        NCV, NMV, ND, NQ, NDQ, NU = self.dims
        out = torch.full((self.k, NU+1), fill, dtype=F)
        return out.scatter(1, self.t['chan'], val.to(F))[:, :NU]

    def win(self, ue, dlev):
        """Physical inputs at valve flows ue (k, NMV) and disturbance levels dlev (k, NDQ)."""
        t = self.t
        src = torch.cat((ue, dlev, torch.zeros(self.k, 1, dtype=F)), 1)
        return t['wbase']+t['wscale']*torch.gather(src, 1, t['chan'])

    def flows(self, w):
        """Valve flows (k, NMV) of the physical inputs."""
        NCV, NMV, ND, NQ, NDQ, NU = self.dims
        t = self.t
        return self.chans((w-t['wbase'])/t['wscale'].clamp_min(1e-300))[:, :NMV]

    def masks(self):
        NCV, NMV, ND, NQ, NDQ, NU = self.dims
        t, k = self.t, self.k
        ar, am = torch.arange(NCV)[None], torch.arange(NMV)[None]
        nm, nz, nd, nq = (t[q][:, None] for q in ('nm', 'nz', 'nd', 'nq'))
        trk = (ar < nm).to(F)
        zon = ((ar >= nm) & (ar < nm+nz)).to(F)
        dvr = ((ar >= nm+nz) & (ar < nm+nz+nd)).to(F)
        mvm = (am < nm).to(F)
        dact = torch.cat((torch.arange(ND)[None] < nd, torch.arange(NQ)[None] < nq), 1).to(F)
        return trk, zon, dvr, mvm, dact, (ar-(nm+nz)).clamp(0, ND-1)

    # ---- the local model
    def model(self, q, x, w, mp):
        """q with the plant's local model at the steady point (x, w, mp): steady gains per unit of valve flow
        and of load level (an integrating row: its rate over t['tint'] decisions; a row that follows a level:
        with the level where it was), and the bias that makes the generator's linear steady equations exact
        there; the rows' values; and whether the point is stable (no mode of the decision map grows: left
        alone the plant stays, a level aside)."""
        NCV, NMV, ND, NQ, NDQ, NU = self.dims
        t, k = self.t, self.k
        C, D, Phi, Gam, Gv, Gr = self.M.gains(x, w, mp, levels=t['oint'] & t['olev'][:, None])    # rows that follow a level: with the level where it was
        G = torch.where(t['oint'][:, :, None], t['tint'][:, None, None]*Gr, Gv)*t['wscale'][:, None, :]/t['rspan'][:, :, None]
        K = torch.zeros(k, NCV+1, NU+1, dtype=F)
        K[torch.arange(k)[:, None, None], t['orow'][:, :, None], t['chan'][:, None, :]] = G
        K = K[:, :NCV, :NU]
        y = self.M.linearise(x, w, mp)[1]
        v = self.rows((y-t['rlo'])/t['rspan'])
        base = (K*torch.cat((self.flows(w), q['dlev']), 1)[:, None, :]).sum(-1)
        q = dict(q)
        q['K'] = K
        q['bias'] = torch.where(q['integ'] > 0, -base, v-base)*q['prc']
        fin = torch.isfinite(Phi).flatten(1).all(-1)
        with one(Phi):
            rho = torch.linalg.eigvals(torch.where(fin[:, None, None], Phi, 2*torch.eye(Phi.shape[-1], dtype=F)[None])).abs().amax(-1)
        return q, v, rho <= 1+1e-6

    def settle(self, q, tgt, mp, start):
        """The true steady point at which the plant q (its loads, limits and task; parameters mp) holds the
        targets, walked to from the known steady point `start`; q with the local model there, the
        generator's steady point, the plant's state and inputs, and whether it passes the one check of every
        operating point (prior.operable) with the targets inside the plant's own range, the point stable,
        the published gain signs still true and the local model good for a finite move (`margin`)."""
        NCV, NMV, ND, NQ, NDQ, NU = self.dims
        t = self.t
        g = lambda a, i: torch.gather(a, 1, i)                    # noqa: E731
        tphys = g(t['rlo'], t['ridx'])+g(t['rspan'], t['ridx'])*tgt[:, :NMV]
        w1 = self.win(self.flows(start[1]) if start is not None else torch.zeros(self.k, NMV, dtype=F), q['dlev'])
        x, w, conv = self.M.steady(w1, mp, free=t['fidx'], rows=t['ridx'], fmask=t['fm'], targets=tphys, band=g(t['oband'], t['ridx']),
                                   wlo=g(t['ilo'], t['fidx']), whi=g(t['ihi'], t['fidx']), start=start, **WALK)
        x, w = x.nan_to_num(0., 0., 0.), torch.where(torch.isfinite(w), w, w1)
        q, v, stable = self.model(q, x, w, mp)
        ok, info = PR.operable(q, tgt)
        trk = q['trk'] > 0
        ok = ok & conv & stable & (((tgt >= q['tlo']-1e-9) & (tgt <= q['thi']+1e-9)) | ~trk).all(-1)
        if 'psign' in q:
            ok = self.margin(q, x, w, mp, v, ok & self.signs(q))
        return q, info, x, w, ok

    @staticmethod
    def signs(q):
        """The gain of every MV on its published CV has the sign it was commissioned with."""
        NMV = q['link'].shape[2]
        s = torch.sign((q['K'][:, :, :NMV]*q['link']).sum(1))
        return ((s == q['psign']) | (q['mvm'] == 0)).all(-1)

    def margin(self, q, x, w, mp, v, ok, ratios=False):
        """`ok` and the local model holds over a finite move of every MV: with the valve moved either way by
        what the model says shifts its paired row MARGIN_MOVE bands (no more than MARGIN_STEP of its span,
        nor beyond its range), every other valve where it is and every level held by its own MV, the plant
        settles and the paired row has moved the way the model says, by no less than 1/MARGIN_RATIO and no
        more than MARGIN_RATIO times as far. A point that fails sits by a fold of the plant (a yield
        optimum, washout, a valve about to lose its grip): linear advice about it is wrong within one
        move. Only the episodes still `ok` are tried."""
        NCV, NMV, ND, NQ, NDQ, NU = self.dims
        t = self.t
        i = ok.nonzero().flatten()
        if i.numel() == 0:
            return (ok, None) if ratios else ok
        sub = lambda a: a[i]                                      # noqa: E731
        link, mvm, K, integ, bandf = (sub(q[key]) for key in ('link', 'mvm', 'K', 'integ', 'bandf'))
        m = i.numel()
        prow = link.argmax(1)                                     # (m, NMV) paired row of every MV
        mvm = mvm > 0
        Kp = torch.gather(K[:, :, :NMV], 1, prow[:, :, None].expand(m, NMV, NMV))        # row of MV a, MV b
        ig = (torch.gather(integ, 1, prow) > 0) & mvm             # the MVs that hold a level
        band = torch.gather(bandf, 1, prow)
        ue = self.flows(w)[i]
        d = torch.minimum(torch.full_like(band, MARGIN_STEP), MARGIN_MOVE*band/Kp.diagonal(dim1=1, dim2=2).abs().clamp_min(1e-12))
        e, j = (mvm & ~ig).nonzero(as_tuple=True)                 # one trial per valve and direction
        if e.numel() == 0:
            return (ok, None) if ratios else ok
        h = e.numel()
        e, j = e.repeat(2), j.repeat(2)
        sg = torch.cat((torch.ones(h, dtype=F), -torch.ones(h, dtype=F)))
        ar = torch.arange(2*h)
        # what the model says of a unit move, the levels held: the level rows stay, the valve tried moves, the rest stay
        eye = torch.eye(NMV, dtype=F)[None]
        A = torch.where(ig[:, :, None], Kp, eye)[e]
        rhs = torch.where(ig[e], torch.zeros(2*h, NMV, dtype=F), torch.nn.functional.one_hot(j, NMV).to(F))
        du = torch.linalg.solve_ex(A, rhs[..., None])[0][..., 0].nan_to_num(0., 0., 0.)
        # the move: inside the valve's own range, and no further than the valves that hold the levels can follow
        room = torch.where(du*sg[:, None] > 0, 1-ue[e], ue[e])/du.abs().clamp_min(1e-12)
        de = sg*torch.minimum(d[e, j], .9*torch.where(ig[e], room, torch.full_like(room, math.inf)).amin(-1))
        u1 = ue[e].clone()
        u1[ar, j] = (u1[ar, j]+de).clamp(0., 1.)
        de = u1[ar, j]-ue[e, j]
        pred = (Kp[e]@du[..., None])[ar, j, 0]*de
        rows = i[e]
        g = self.view(rows)
        tt = g.t
        lrow = torch.gather(tt['ridx'], 1, prow[e])               # plant output of every MV's row
        y0 = self.M.y(x, w, mp)[i]
        gt = lambda a, b: torch.gather(a, 1, b)                   # noqa: E731
        w1 = g.win(u1, q['dlev'][rows])
        if bool(ig.any()):
            x1, w1, conv = g.M.steady(w1, mp[rows], free=tt['fidx'], rows=lrow, fmask=ig[e], targets=gt(y0[e], lrow), band=gt(tt['oband'], lrow),
                                      wlo=gt(tt['ilo'], tt['fidx']), whi=gt(tt['ihi'], tt['fidx']), start=(x[rows], w[rows], mp[rows]))
        else:
            x1, w1, conv = g.M.steady(w1, mp[rows], start=(x[rows], w[rows], mp[rows]))
        v1 = g.rows((g.M.y(x1.nan_to_num(0., 0., 0.), w1, mp[rows])-tt['rlo'])/tt['rspan'])
        actual = (v1-v[rows])[ar, prow[e, j]]
        rr = actual/torch.where(pred.abs() > 1e-300, pred, torch.full_like(pred, 1e-300))
        fine = (conv & torch.isfinite(rr) & (rr >= 1/MARGIN_RATIO) & (rr <= MARGIN_RATIO)) | (de.abs() < 1e-9)
        bad = torch.zeros(m, dtype=F).index_add_(0, e, (~fine).to(F)) > 0
        out = ok.clone()
        out[i] = ~bad
        if ratios:
            R = torch.full((self.k, NMV, 2), float('nan'), dtype=F)
            R[rows, j, (sg < 0).long()] = torch.where(conv, rr, torch.full_like(rr, float('nan')))
            return out, R
        return out

    def reach(self, p, mp, x0, w0, x1, w1, ok):
        """`ok` and the plant gets from the steady point (x0, w0) to (x1, w1) when the event strikes all at
        once (loads and equipment as after it, from the first decision) and a plain operator runs it: the
        MVs wait REACH_WAIT of the unit's time scale where they were, then travel straight to their new
        steady flows at the slew limit. Asked at the end (REACH_SETTLE time scales after the travel): the
        outputs the shell reads (levels aside) are within REACH_TOL bands of their new steady values, or
        REACH_SHARE of the largest distance on the way. An event that fails throws the plant somewhere it
        does not come back from by itself (washout, ignition): no steady-state check sees that. Only the
        episodes still `ok` are run."""
        NCV, NMV, ND, NQ, NDQ, NU = self.dims
        i = ok.nonzero().flatten()
        if i.numel() == 0:
            return ok
        g = self.view(i)
        t = g.t
        sub = lambda a: a[i]                                      # noqa: E731
        slew, mvm, T, dl = (sub(p[key]) for key in ('slew', 'mvm', 'T', 'dlev'))
        mq, xa, xb, wa, wb = (sub(a) for a in (mp, x0, x1, w0, w1))
        ua, ub = g.flows(wa), g.flows(wb)
        wait = PR.up(REACH_WAIT*T)
        travel = PR.up(((ub-ua).abs()/slew*mvm).amax(-1))
        H = PR.up(wait+travel+REACH_SETTLE*T).clamp_max(REACH_MAX)
        y1 = g.M.y(xb, wb, mq)
        use = ((t['orow'] < NCV) & ~t['oint']).to(F)
        x = xa.clone()
        peak, end = torch.zeros(i.numel(), dtype=F), torch.full((i.numel(),), math.inf, dtype=F)
        act = torch.arange(i.numel())                             # the episodes whose run is not over (the others are no longer stepped: most runs are far shorter than the longest)
        for s in range(1, int(H.max())+1):
            on = H >= s
            if int(on.sum()) <= REACH_PACK*act.numel():
                keep = on.nonzero().flatten()
                g, t, act, x = g.view(keep), {key: v[keep] for key, v in t.items()}, act[keep], x[keep]
                slew, dl, mq, ua, ub, wait, H, y1, use = (a[keep] for a in (slew, dl, mq, ua, ub, wait, H, y1, use))
            tau = (s-wait).clamp_min(0.)[:, None]*slew
            w = g.win(ua+torch.minimum(torch.maximum(ub-ua, -tau), tau), dl)
            x = g.M.advance(x, w, mq)
            last = s >= H
            if s % REACH_EVERY == 0 or bool((last & ~torch.isfinite(end[act])).any()):
                dist = (((g.M.y(x, w, mq)-y1).abs()/t['oband'])*use).amax(-1).nan_to_num(1e30, 1e30, 1e30)
                peak[act] = torch.where(last, peak[act], torch.maximum(peak[act], dist))
                end[act] = torch.where(last & ~torch.isfinite(end[act]), dist, end[act])
        out = ok.clone()
        out[i] = end <= torch.maximum(torch.full_like(peak, REACH_TOL), REACH_SHARE*peak)
        return out

    def settle_constrained(self, q, tgt, mp, start, info):
        """The best steady point of the true plant when the targets cannot all be held. From the optimum on
        the local model (`info`, prior.optimum): the plant's steady state with the valves there, the local
        model again at that state, the optimum again, until the valve flows agree and the plant's tracked rows sit at
        the reachable targets. Plants with an integrating row are left out (no steady state with every valve fixed)."""
        ue, cur = info['ue'], start
        ok = ~(q['integ'] > 0).any(-1)
        good = lambda u: torch.isfinite(u).all(-1)                # noqa: E731
        for _ in range(EXC_ROUNDS):
            ue = torch.where((ok & good(ue))[:, None], ue, self.flows(cur[1])).clamp(0., 1.)
            x, w, conv = self.M.steady(self.win(ue, q['dlev']), mp, start=cur)
            ok = ok & conv
            x = torch.where(ok[:, None], x.nan_to_num(0., 0., 0.), cur[0])
            w = torch.where(ok[:, None], w, cur[1])
            q, v, stable = self.model(q, x, w, mp)
            okc, info, rt, zal = PR._constrained_ok(q, tgt)
            d = ((info['ue']-ue).abs()*q['mvm']).amax(-1).nan_to_num(1., 1., 1.)
            dv = torch.where(q['trk'] > 0, (rt-v).abs()/q['bandf'], torch.zeros_like(v)).amax(-1).nan_to_num(1., 1., 1.)    # a large gain makes a small valve difference many bands
            done = (d < EXC_TOL) & (dv < EXC_BAND)
            if bool((done | ~ok).all()):
                break
            ue, cur = info['ue'], (x, w, mp)
        ok = self.margin(q, x, w, mp, v, ok & okc & stable & done & self.signs(q))
        q['rtgt'], q['zal'] = rt, zal
        return q, info, x, w, ok

    # ---- the start of an episode
    def start(self, r):
        """Every episode at an operable start point: equipment, loads and targets drawn inside the plant's
        ranges, redrawn where the point fails the check (the plant's nominal point after START_ROUNDS). An
        episode whose task has no operable start even there (the outputs it was given to hold do not pair with
        its valves, at its clock) is given another task of the same plant, from its own stream."""
        t, k = self.t, self.k
        r.bind(k)
        best = self._start(r)
        for _ in range(TASK_ROUNDS):
            bad = (~best[-1]).nonzero().flatten()
            if bad.numel() == 0:
                break
            rs = r.sub(bad)
            g = Group.draw(rs, self.M.plants, t['pid'][bad], self.size)
            new = g._start(rs)
            self.t = t = {key: v.index_copy(0, bad, g.t[key]) for key, v in t.items()}
            self.M = Mech(self.M.plants, t['pid'], dtype=F, _base=self.M._base, pace=t['pace'], more=self.M.more)
            best = _put(best, new, bad, torch.ones_like(new[-1]))
        if not bool(best[-1].all()):
            names = sorted({self.M.plants[int(i)].name for i in t['pid'][~best[-1]]})
            raise RuntimeError(f'no operable start for {int((~best[-1]).sum())} of {k} episodes on {names}')
        p, tgt, info, mp, x, w, _ = best
        p = PR.derive(p)
        T, slow = self.timescale(p, x, w, mp)
        p['T'], p['rth'] = T, (self.chans(t['din'].to(F))[:, :p['mvm'].shape[1]]*p['mvm']).sum(-1)/p['mvm'].sum(-1).clamp_min(1)/T
        self.slow = slow
        return p, tgt, info, mp, x, w

    def _start(self, r):
        """A start point for every episode with its task as it is: (p, targets, steady point, parameters, state,
        inputs, whether it is operable)."""
        NCV, NMV, ND, NQ, NDQ, NU = self.dims
        t, k = self.t, self.k
        U = r.u
        r.bind(k)
        trk, zon, dvr, mvm, dact, dvsel = self.masks()
        prc, rstf = trk+zon, torch.zeros(k, NCV, dtype=F)
        z = lambda *s, v=0.: torch.full((k, *s), v, dtype=F)      # noqa: E731
        band = self.rows(t['oband']/t['rspan'], .02)
        nrm = lambda a, fill: self.rows((a-t['rlo'])/t['rspan'], fill)          # noqa: E731
        lo, hi = nrm(t['olo'], -math.inf), nrm(t['ohi'], math.inf)
        inf = z(NCV, v=math.inf)
        ismv = t['ikind'] == 0
        slew = self.chans(torch.where(ismv, t['islew']/(t['ihi']-t['ilo']), torch.ones_like(t['ilo'])), 1.)[:, :NMV].clamp(.002, .5)
        dead = self.chans(t['din'].to(F))[:, None, :].expand(k, NCV, NU)*prc[:, :, None]
        p = dict(nm=t['nm'].to(F), trk=trk, zon=zon, dvr=dvr, prc=prc, mvm=mvm, dact=dact, uact=torch.cat((mvm, dact), 1), dvsel=dvsel.to(F),
                 link=z(NCV, NMV), rst=rstf, spr=z(NMV), stock=z(NCV), K=z(NCV, NU), p1r=z(NCV, NU, v=-.2), p1i=z(NCV, NU), p2r=z(NCV, NU, v=-.4), p2i=z(NCV, NU),
                 beta=z(NCV, NU), first=z(NCV, NU, v=1.), dead=dead.clone(), integ=self.rows(t['oint'].to(F))*trk, tint=t['tint'][:, None].expand(k, NCV).clone(), lam=z(NCV),
                 vc=z(NCV, v=.5), nlag=z(NCV), tlag=z(NCV, v=1.), wa=z(NCV), bias=z(NCV), bandf=band, sband=band.clone(),
                 gam=torch.where(r.bern(VALVE['curve'], k, NMV), U(k, NMV, lo=.6, hi=1.7), z(NMV, v=1.)),
                 aa=torch.where(r.bern(VALVE['lag'], k, NMV), torch.exp(-1/r.logu(k, NMV, lo=.3, hi=3.)), z(NMV))*mvm, slew=slew,
                 warp=z(), jload=z(), uref=z(v=.5), kap=z(v=1.), dlev=z(NDQ),
                 lo=torch.where(zon > 0, lo, -inf), hi=torch.where(zon > 0, hi, inf), cvw=PR.draw_priority(r, trk, rstf), dt=t['dt'].clone(),
                 T=z(v=10.), rth=z(v=.1), store=z(), arch=z(v=float(PR.ARCH.index('mechanism'))), linear=z(), mech=(t['pid']+1).to(F),
                 fcap=z(NMV, v=1.), fs=z(NMV), fd=z(NMV), ulo=z(NMV, v=PR.ULO), uhi=z(NMV, v=PR.UHI), zal=z(NCV),
                 tlo=torch.where(trk > 0, nrm(t['tlo'], 0.), z(NCV)), thi=torch.where(trk > 0, nrm(t['thi'], 1.), z(NCV, v=1.)))
        if self.size != PR.DIMS:                                  # in a batch of a larger size: one unit (prior.draw_plant gives every row and input its unit)
            p.update(unit=z(NCV), unitc=z(NU))
        ar = torch.arange(k)
        for j in range(NMV):                                      # until the gains are known: MV j with row j
            p['link'][ar, j, j] = mvm[:, j]
        pnom = self.M.pnom
        xn, _, okn = self.M.steady(self.M.wnom, pnom)             # the plant's nominal point, where it has one (no level to balance)
        vn = self.rows((self.M.linearise(xn, self.M.wnom, pnom)[1]-t['rlo'])/t['rspan'])
        tnom = torch.where(okn[:, None], torch.minimum(torch.maximum(vn, p['tlo']), p['thi']), (p['tlo']+p['thi'])/2)*trk

        def cand(g, r, p, tnom, nominal=False):                   # one draw of a start point for the episodes of g, from their streams
            t, k, U = g.t, g.k, r.u
            z = lambda *s, v=0.: torch.full((k, *s), v, dtype=F)  # noqa: E731
            trk, mvm, dact, pnom = p['trk'], p['mvm'], p['dact'], g.M.pnom
            ar = torch.arange(k)
            mp = pnom.clone() if nominal else t['plo']+(t['phi']-t['plo'])*U(k, pnom.shape[1])
            dl = z(NDQ) if nominal else torch.cat((U(k, ND, lo=-1., hi=1.), U(k, NQ, lo=-.5, hi=.5)), 1)*dact
            tg = tnom if nominal else (p['tlo']+(p['thi']-p['tlo'])*U(k, NCV))*trk
            q = dict(p); q['dlev'] = dl
            q['rtgt'] = tg.clone()
            q, info, x, w, ok = g.settle(q, tg, mp, None)
            # the engineer's pairing at this point, the gain signs as commissioned, friction no coarser than half a band
            order, lam = pairing(torch.where(torch.isfinite(info['G']), info['G'], torch.zeros_like(info['G'])), q['integ'], mvm, t['rby'])
            link = z(NCV, NMV)
            for j in range(NMV):
                link[ar, order[:, j], j] = mvm[:, j]
            q['link'] = link
            q['psign'] = torch.sign((q['K'][:, :, :NMV]*link).sum(1))*mvm
            both = (trk[:, :NMV, None] > 0) & (mvm > 0)[:, None, :]
            q['fcap'] = .5*PR.STICK_MAX/(info['G'].abs()*both).amax(1).clamp_min(1e-9)
            fs = torch.where(r.bern(VALVE['stick'], k, NMV), U(k, NMV, lo=.002, hi=.02), z(NMV))*mvm
            q['fs'] = torch.minimum(fs, q['fcap'])
            q['fd'] = q['fs']*U(k, NMV, lo=.3, hi=1.)
            ok = ok & ((lam >= PR.RGA_MIN).all(-1) | r.bern(.03, k))    # positive relative gains, as an engineer pairs (3 % as they come)
            ok = g.margin(q, x, w, mp, g.rows((g.M.y(x, w, mp)-t['rlo'])/t['rspan']), ok)
            return q, tg, info, mp, x, w, ok

        best = cand(self, r, p, tnom)
        for it in range(START_ROUNDS+1):
            bad = (~best[-1]).nonzero().flatten()
            if bad.numel() == 0:
                break
            new = cand(self.view(bad), r.sub(bad), {key: v[bad] for key, v in p.items()}, tnom[bad], nominal=it == START_ROUNDS)
            best = _put(best, new, bad, new[-1])
        return best

    # ---- responses
    def response(self, x, w, mp, aa, mvm, dact, lags=LAGS):
        """Fr (k, lags+1, NCV, NU) as prior.model.response: the unit step response of every row (a level's
        rate over t['tint'] decisions, the value of any other row) to a unit of valve flow command of every MV
        (through the positioner lag and the input's transport delay) and to a unit of every load level, of
        the plant linearised at the steady point and stepped as the engine steps it. A row that follows a
        level (it would run away with it under a step of one valve) is given with the level where it was:
        less what the level's displacement accounts for (engine.follow), so that it settles on the gain of
        `model`. What the level then does to it is not in the response: the teacher meets it as a
        disturbance."""
        NCV, NMV, ND, NQ, NDQ, NU = self.dims
        t, k = self.t, self.k
        _, _, A, B, C, D = self.M.linearise(x, w, mp)
        Phi, Gam = self.M.discrete(A, B)
        if bool(t['olev'].any()):                                 # rows that follow a level: their answer with the level where it was (less what its displacement accounts for)
            lev = (t['oint'] & t['olev'][:, None])[:, :, None]
            S = self.M.follow(self.M.lasting(Phi), C, lev[:, :, 0])
            C, D = torch.where(t['olev'][:, None, None] & ~lev, C-S@(C*lev), C), torch.where(t['olev'][:, None, None] & ~lev, D-S@(D*lev), D)
        ar = torch.arange(k)
        Gc = torch.zeros(k, self.M.nx, NU+1, dtype=F)
        Gc[ar[:, None, None], torch.arange(self.M.nx)[None, :, None], t['chan'][:, None, :]] = Gam*t['wscale'][:, None, :]
        Cr = torch.zeros(k, NCV+1, self.M.nx, dtype=F)
        Cr[ar[:, None], t['orow']] = C/t['rspan'][:, :, None]
        Dr = torch.zeros(k, NCV+1, NU+1, dtype=F)
        Dr[ar[:, None, None], t['orow'][:, :, None], t['chan'][:, None, :]] = D*t['wscale'][:, None, :]/t['rspan'][:, :, None]
        Gc, Cr, Dr = Gc[:, :, :NU], Cr[:, :NCV], Dr[:, :NCV, :NU]
        slope = torch.cat((mvm, dact), 1)
        a = torch.cat((aa, torch.zeros(k, NDQ, dtype=F)), 1)
        din = self.chans(t['din'].to(F))
        integ = (self.rows(t['oint'].to(F)) > 0)[:, :, None]
        X = torch.zeros(k, self.M.nx, NU, dtype=F)
        Fr = torch.zeros(k, lags+1, NCV, NU, dtype=F)
        prev = torch.zeros(k, NCV, NU, dtype=F)
        for s in range(1, lags+1):
            u = slope*torch.where(s-din >= 1, 1-a**(s-din).clamp_min(0.), torch.zeros_like(a))
            X = Phi@X+Gc*u[:, None, :]
            Y = Cr@X+Dr*u[:, None, :]
            Fr[:, s] = torch.where(integ, t['tint'][:, None, None]*(Y-prev), Y)
            prev = Y
        return Fr

    def timescale(self, p, x, w, mp):
        """(k,) decisions the slowest published pair takes to 63 % of its answer at the start point (a
        level: 1), and whether some pair is still moving at the end of the response window."""
        NCV, NMV, ND, NQ, NDQ, NU = self.dims
        k = self.k
        Fr = self.response(x, w, mp, torch.zeros(k, NMV, dtype=F), p['mvm'], p['dact'])
        row = p['link'].argmax(1)                                                 # (k, NMV) paired row of each MV
        s = torch.gather(Fr[:, :, :, :NMV], 2, row[:, None, None, :].expand(k, Fr.shape[1], 1, NMV))[:, :, 0, :]
        end = s[:, -1]
        r = s/torch.where(end.abs() > 1e-12, end, torch.ones_like(end))[:, None]
        t63 = torch.where((r >= .632).any(1), (r >= .632).to(torch.int64).argmax(1), torch.full((k, NMV), s.shape[1]-1)).to(F)
        ig = torch.gather(p['integ'][:, :NMV], 1, row) > 0
        t63 = torch.where(ig | (p['mvm'] == 0), torch.ones_like(t63), t63)
        slow = (((s[:, -1]-s[:, -41]).abs() > .05*s.abs().amax(1)) & ~ig & (p['mvm'] > 0)).any(-1)
        return t63.amax(-1).clamp(PR.T_RANGE[0], 100.), slow

    # ---- events
    def event(self, r, p, info, tgt, mp, kind, alpha, base=None, menu=None, turn=None):
        """One candidate event per episode, as prior.draw_event: the moves of targets, loads, limits and
        priorities are proposed on the local model; a process event changes the equipment (1-3 of the
        plant's parameters towards a new draw inside their ranges) and the valves (curve, friction).
        base: (plant dict, parameters) as commissioned, on a long stay: a share PR.RESTORE_P of the process
        events is maintenance, the parameters picked and the valves go back to it. menu: prior.Menu (proc: the
        parameters move inside ranges that many times as wide about the plant's nominal values, a positive one
        never below a quarter of its low end; drop: 'curve' and 'friction' are the valves' two kinds)."""
        NCV, NMV, ND, NQ, NDQ, NU = self.dims
        t, k = self.t, self.k
        U = r.u
        k1 = kind[:, None]
        is_pr = (kind == PR.PROC) | (kind == PR.BOTH)
        q = dict(p)
        a1 = alpha[:, None]
        npar = mp.shape[1]
        has = (t['phi'] > t['plo']).to(F)*t['pev']                # the parameters an event may move
        pick = r.bern(.3, k, npar) | r.one_of(has > 0)
        plo, phi = t['plo'], t['phi']
        if menu is not None and menu.proc != 1.:
            nom = self.M.pnom
            plo = torch.where(plo > 0, torch.maximum(nom+(plo-nom)*menu.proc, .25*plo), nom+(plo-nom)*menu.proc)
            phi = nom+(phi-nom)*menu.proc
        mpc = torch.where(is_pr[:, None] & pick & (has > 0), mp+a1*(plo+(phi-plo)*U(k, npar)-mp), mp)
        sh = r.bern(VALVE['wear'], k, 2) & is_pr[:, None]
        if menu is not None and menu.drop:
            sh = sh & torch.tensor([name not in menu.drop for name in ('curve', 'friction')])[None]
        q['gam'] = torch.where(sh[:, 0, None] & (p['gam'] != 1), (p['gam']*torch.exp(a1*U(k, NMV, lo=-.22, hi=.22))).clamp(.6, 1.7), p['gam'])
        one = torch.nn.functional.one_hot(r.randint(0, NMV-1, k), NMV).to(F)*p['mvm']*sh[:, 1, None].to(F)
        fs = torch.minimum(a1*U(k, NMV, lo=.002, hi=.02), p['fcap'])
        q['fs'] = torch.where(one > 0, torch.maximum(p['fs'], fs), p['fs'])
        q['fd'] = torch.where(one > 0, torch.minimum(torch.maximum(p['fd'], fs*U(k, NMV, lo=.3, hi=1.)), q['fs']), p['fd'])
        if base is not None:
            back = (r.bern(PR.RESTORE_P, k) & is_pr)[:, None]
            mpc = torch.where(back, torch.where(pick & (has > 0), base[1], mp), mpc)
            valves = r.bern(.5, k)[:, None]
            for key in ('gam', 'fs', 'fd'):
                q[key] = torch.where(back, torch.where(valves, base[0][key], p[key]), q[key])
        tn, _ = PR.draw_move(r, p, info, tgt, menu, turn)
        trk = p['trk'] > 0
        tn = torch.where(trk, torch.minimum(torch.maximum(tn, p['tlo']), p['thi']), tn)
        is_sp = (kind == PR.SP) | (kind == PR.BOTH)
        big = ~is_sp | (((tn-tgt).abs()/p['bandf']*p['trk']).amax(-1) >= PR.MOVE_MIN*(1. if menu is None else min(1., menu.move)))
        tn = torch.where(is_sp[:, None], tn, tgt)
        is_ld = (kind == PR.LOAD) | (kind == PR.DV)
        dl, eff = PR.draw_load(r, p, info, tgt, kind == PR.DV, menu)
        big = big & (~is_ld | (eff >= PR.LOAD_MIN*(1. if menu is None else min(1., menu.step))))
        q['dlev'] = torch.where(is_ld[:, None], dl, p['dlev'])
        q['cvw'] = torch.where(k1 == PR.PRIO, PR.draw_priority(r, p['trk'], p['rst'], prob=1.), p['cvw'])
        y, b = info['y'], p['bandf']
        lo = y-b*r.logu(k, NCV, lo=1.5, hi=40.)
        hi = y+b*r.logu(k, NCV, lo=1.5, hi=40.)
        q['lo'] = torch.where((k1 == PR.LIMIT) & torch.isfinite(p['lo']), lo, p['lo'])
        q['hi'] = torch.where((k1 == PR.LIMIT) & torch.isfinite(p['hi']), hi, p['hi'])
        return q, tn, big, mpc

    def episode(self, r, exc_p, tail=None, keep=None, menu=None, nseg=NSEG):
        """Start points and the event schedule, as prior.draw_episode, every operating point settled on the
        true plant. Returns its seven results and, per segment, the plant's parameters, steady state and
        inputs. tail, keep: the next window on the same plants; menu, nseg: as there."""
        k = self.k
        seg_k = torch.full((k, nseg), BIG, dtype=torch.long)
        kinds = torch.full((k, nseg), -1, dtype=torch.long)
        delay = torch.zeros(k, nseg, dtype=torch.long)
        ramp = torch.zeros(k, nseg, dtype=torch.long)
        if menu is not None and menu.exc is not None:
            exc_p = menu.exc
        lz = torch.zeros(k, dtype=torch.long)
        if tail is None:
            cur = self.start(r)
            hist = [cur]
            seg_k[:, 0] = 0
            before, before_m, last_m = cur, (lz-1, lz-1, lz, lz), (lz, lz-1, lz, lz)
        else:
            r.bind(k)
            before, cur = tail['before'], tail['cur']
            hist = [tail['base'], before, cur]
            before_m, last_m = tuple(x.clone() for x in tail['before_m']), tuple(x.clone() for x in tail['last_m'])
            before_m[0].sub_(HORIZON); last_m[0].sub_(HORIZON)
            for j, m in ((1, before_m), (2, last_m)):
                seg_k[:, j], kinds[:, j], delay[:, j], ramp[:, j] = m
            seg_k[:, 0] = seg_k[:, 1]-1
        p = cur[0]
        mix = PR.event_kinds(r, p, menu=menu)
        if menu is not None:
            mix, speed, none = PR.pacing(mix, menu)
        ts = PR.settling(p)
        glo, ghi = ts.clamp(40., 150.), (2.5*ts).clamp(80., 300.)
        if tail is None:
            kk = r.randint(PR.FIRST_EVENT[0], PR.FIRST_EVENT[1], k)
            away = torch.zeros(k, dtype=torch.bool)
            home = cur
            nsp = lz.clone()
        else:
            kk, away, home, nsp = (tail['k']-HORIZON).clamp_min(1), tail['away'], tail['home'], tail['turn']
        merge = lambda c, new, old: tuple({key: _pick(c, a[key], b[key]) for key in b} if isinstance(b, dict) else _pick(c, a, b)   # noqa: E731
                                          for a, b in zip(new, old))
        for e in range(len(hist), nseg):
            p, tgt, info, mp, x, w = cur
            exists = kk < HORIZON-30
            if menu is not None:
                exists = exists & ~none
            kind = r.cat_rows(mix)
            ok = torch.zeros(k, dtype=torch.bool)
            nxt = cur
            for a in PR.ALPHA:
                idx = (exists & ~ok & ~away).nonzero().flatten()  # the episodes still without their event
                if idx.numel() == 0:
                    break
                g = self.view(idx)
                sub = lambda a: {key: v[idx] for key, v in a.items()} if isinstance(a, dict) else a[idx]    # noqa: E731
                pc, tc, big, mpc = g.event(r.sub(idx), sub(p), sub(info), tgt[idx], mp[idx], kind[idx], torch.full((idx.numel(),), a, dtype=F),
                                           None if tail is None else (sub(hist[0][0]), hist[0][3][idx]),
                                           menu, nsp[idx] if menu is not None and menu.turn else None)
                pc, infc, xc, wc, okc = g.settle(pc, tc, mpc, (x[idx], w[idx], mp[idx]))
                okc = g.reach(pc, mpc, x[idx], w[idx], xc, wc, okc & big)
                nxt = _put(nxt, (pc, tc, infc, mpc, xc, wc), idx, okc)
                ok[idx[okc]] = True
            pn = nxt[0]
            pn['rtgt'] = torch.where(ok[:, None] & (pn['trk'] > 0), nxt[1], pn['rtgt'])
            nsp = nsp+(ok & ((kind == PR.SP) | (kind == PR.BOTH))).long()
            out = torch.zeros(k, dtype=torch.bool)
            gone = out                                            # a disturbance or process event the next event takes back
            if menu is not None and menu.pulse > 0:
                gone = ok & ((kind == PR.LOAD) | (kind == PR.DV) | (kind == PR.PROC)) & r.bern(menu.pulse, k)
            idx = (exists & ~away & ~gone & r.bern(exc_p, k)).nonzero().flatten() if e < nseg-1 else out.nonzero().flatten()
            for _ in range(PR.EXC_TRIES):
                if idx.numel() == 0:
                    break
                sub = lambda a: {key: v[idx] for key, v in a.items()} if isinstance(a, dict) else a[idx]    # noqa: E731
                px, tx, infx, kx, okx = PR.draw_excursion(r.sub(idx), sub(p), sub(info), tgt[idx])
                g = self.view(idx)
                px, infx, xx, wx, okc = g.settle_constrained(px, tx, mp[idx], (x[idx], w[idx], mp[idx]), infx)
                okx = g.reach(px, mp[idx], x[idx], w[idx], xx, wx, okx & okc)
                nxt = _put(nxt, (px, tx, infx, mp[idx], xx, wx), idx, okx)
                kind = _put((kind,), (kx,), idx, okx)[0]
                out[idx[okx]] = True
                idx = idx[~okx]
            ok = ok | out
            home = merge(out | gone, cur, home)
            back = away & exists
            nxt = merge(back, home, nxt)
            kind, ok = torch.where(back, torch.full_like(kind, PR.RETURN), kind), ok | back
            away = (away & ~exists) | out | gone
            before, before_m = merge(exists, cur, before), merge(exists, last_m, before_m)      # (the last segment in use so far is no longer the last)
            prev, cur = cur, nxt
            p, tgt, info, mp, x, w = cur
            hist.append(cur)
            seg_k[:, e] = torch.where(exists, kk, torch.full_like(kk, BIG))
            kinds[:, e] = torch.where(ok, kind, torch.full_like(kind, -1))
            delay[:, e] = r.randint(PR.RENEW[0], PR.RENEW[1], k)
            gap = (glo+(ghi-glo)*r.u(k)).floor().long()
            if menu is not None:
                gap = PR.down(gap/speed).long().clamp_min(PR.MIN_GAP)
            ramp[:, e] = torch.where(r.bern(PR.RAMP_P, k), torch.minimum(r.logu(k, lo=5., hi=120.), .8*gap).long(), torch.zeros_like(gap))
            travel = ((info['un']-prev[2]['un']).abs()/p['slew']*p['mvm']).amax(-1)
            gap = torch.maximum(gap, PR.down(PR.TRAVEL*travel+.5*glo/(1. if menu is None else speed)).long())
            stay = PR.down(PR.lineout(p, prev[2], info, ramp[:, e])+PR.STAY*ts).long()
            gap = torch.where(out, torch.maximum(gap, stay), gap)
            last_m = merge(exists, (kk, kinds[:, e], delay[:, e], ramp[:, e]), last_m)
            kk = torch.where(exists, kk+gap, kk)                    # (an event past the window: the next window's first)
        if keep is not None:
            keep.update(base=hist[0], before=before, before_m=before_m, last_m=last_m, cur=cur, home=home, away=away, k=kk, turn=nsp)
        segs, tgts, infos, mps, xs, ws = ([h[i] for h in hist] for i in range(6))
        return (segs, tgts, infos, seg_k, kinds, delay, ramp), (torch.stack(mps, 1), torch.stack(xs, 1), torch.stack(ws, 1))


def pairing(G, integ, mvm, by=None):
    """(k, NMV) row of every MV, as an engineer configures a unit, and the relative gain of each pair. The
    levels first: every integrating row, the one that moves fastest first, takes the free MV that moves it
    most (the levels already paired held), or the MV slot it names (by (k, NMV), -1 for none: a vessel
    built with its level valve). Then the other rows by relative gain with the levels held
    (prior.best_pairing)."""
    NMV = mvm.shape[1]
    k = G.shape[0]
    ar = torch.arange(k)
    act = mvm > 0
    G = torch.where(act[:, :, None] & act[:, None, :], G, torch.zeros_like(G))
    todo, free = (integ[:, :NMV] > 0) & act, act.clone()
    held = torch.zeros(k, NMV, NMV, dtype=torch.bool)             # (row, MV) pairs of the levels
    named = None if by is None or not bool((by >= 0).any()) else (by[:, :, None] < 0) | (by[:, :, None] == torch.arange(NMV)[None, None])
    for _ in range(NMV):
        size = G.abs()*todo[:, :, None]*free[:, None, :]
        if named is not None:
            size = size*named
        flat = size.flatten(1).argmax(1)
        rr, mm = flat//NMV, flat % NMV
        has = todo.any(1) & (size.flatten(1).amax(1) > 0)
        if not bool(has.any()):
            break
        i, rr, mm = ar[has], rr[has], mm[has]
        piv = G[i, rr, mm]
        Gi = G[i]-G[i, :, mm][:, :, None]*G[i, rr, :][:, None, :]/piv[:, None, None]      # the level held by its MV
        Gi[torch.arange(i.numel()), rr, :] = 0.
        Gi[torch.arange(i.numel()), :, mm] = 0.
        G[i] = Gi
        held[i, rr, mm] = True
        todo[i, rr] = False
        free[i, mm] = False
    B = torch.where(held, torch.ones_like(G), G)
    order = PR.best_pairing(B, mvm)
    Bm = torch.where(act[:, :, None] & act[:, None, :], B, torch.eye(NMV, dtype=F)[None])
    Bi, bad = torch.linalg.inv_ex(Bm)
    lam = torch.gather(Bm*Bi.transpose(1, 2), 1, order[:, None, :])[:, 0]         # relative gain of MV j with its row, the levels held
    return order, torch.where(act & (bad == 0)[:, None], lam, torch.ones_like(lam))


def _put(old, new, idx, ok):
    """Tuples of tensors and dicts of tensors: `old` with its rows idx[ok] taken from the rows ok of `new`
    (keys of `new` that `old` lacks are dropped)."""
    got = idx[ok]

    def put(a, b):
        a = a.clone(); a[got] = b[ok].to(a.dtype)
        return a
    return tuple({key: put(a[key], b[key]) for key in a} if isinstance(a, dict) else put(a, b) for a, b in zip(old, new))


def _pick(c, a, b):
    return torch.where(c.view(-1, *([1]*(a.dim()-1))), a, b)


class Set:
    """The mechanistic episodes of a Program: their groups (one per batch of plants), what the plant step
    needs of each, and the responses of their segments."""

    def __init__(self, groups, parts, n, dims=PR.DIMS, tails=None, start=0):
        self.groups, self.parts, self.n = groups, parts, n        # parts: per group (first episode, P, xs, ws)
        self.dims = PR.sizes_of(*dims)
        self.tails, self.s0 = tails, start                        # what the next window on these plants starts from; the segment in force at decision 0

    def names(self):
        out = []
        for g in self.groups:
            out += [g.M.plants[int(i)].name for i in g.t['pid']]
        return out

    def response(self, e, s, pa):
        """Fr of the segments (episode e, segment s) (indices among the mechanistic episodes); pa: their aa,
        mvm, dact."""
        NCV, NMV, ND, NQ, NDQ, NU = self.dims
        Fr = torch.zeros(e.numel(), LAGS+1, NCV, NU, dtype=F)
        for g, (a, P, xs, ws) in zip(self.groups, self.parts):
            m = ((e >= a) & (e < a+g.k)).nonzero().flatten()
            if m.numel():
                i, j = e[m]-a, s[m]
                Fr[m] = g.view(i).response(xs[i, j], ws[i, j], P[i, j], pa['aa'][m], pa['mvm'][m], pa['dact'][m])
        return Fr

    def runtime(self, first, device, dtype):
        """Per group, what core.step uses (on device): the engine, the episodes' indices in the Program
        (first: index of the first mechanistic episode), the maps between shell and plant, the parameters of
        every segment, and the start states."""
        out = []
        for g, (a, P, xs, ws) in zip(self.groups, self.parts):
            t = g.t
            to = lambda v: v.to(device, dtype)                    # noqa: E731
            out.append(dict(m=Mech(g.M.plants, t['pid'], device=device, dtype=dtype, pace=t['pace'], more=g.M.more), idx=(first+a+torch.arange(g.k)).to(device),
                            ar=torch.arange(g.k, device=device), chan=t['chan'].to(device), din=t['din'].to(device), orow=t['orow'].to(device),
                            wbase=to(t['wbase']), wscale=to(t['wscale']), rlo=to(t['rlo']), rspan=to(t['rspan']), P=to(P), x0=to(xs[:, self.s0])))
        return out


def draw(r, n, plants=None, exc_p=PR.EXC_P, weights=None, dims=PR.DIMS, after=None, menu=None, nseg=NSEG):
    """n episodes on mechanistic plants: the seven results of prior.draw_episode (float64, on r's device)
    and the Set. Plants are drawn by `weights` (equal without them); the episodes are ordered by plant.
    after: the Set of the window before; this is the next window on its plants (prior.draw_episode, `tail`)."""
    rc = r.fork('cpu')
    groups, parts, res, tails, a = [], [], [], [], 0
    if after is None:
        plants = plants or pool()
        pid = torch.sort(rc.cat(n, weights or (1.,)*len(plants))).values
        size = [int(q.table('fj').count.sum()) for q in plants]
        bucket = torch.tensor([0 if s <= BUCKET else i+1 for i, s in enumerate(size)])[pid]
        todo = []
        for b in bucket.unique().tolist():
            ids = pid[bucket == b]
            use = ids.unique().tolist()
            todo.append((ids.numel(), [plants[i] for i in use], torch.searchsorted(torch.tensor(use), ids)))
    else:
        assert after.n == n and after.tails is not None
        rc.bind(n)
        todo = list(zip(after.groups, after.tails))
    for item in todo:
        k = item[0].k if after is not None else item[0]
        rg = rc.sub(a+torch.arange(k))                           # the episodes are ordered by plant: this group's are a..a+k
        g = item[0] if after is not None else Group.draw(rg, item[1], item[2], dims)
        tl = {}
        out, (P, xs, ws) = g.episode(rg, exc_p, tail=item[1] if after is not None else None, keep=tl, menu=menu, nseg=nseg)
        groups.append(g); parts.append((a, P, xs, ws)); res.append(out); tails.append(tl)
        a += g.k
    cat = lambda xs: {q: torch.cat([x[q] for x in xs]) for q in xs[0]} if isinstance(xs[0], dict) else torch.cat(xs)   # noqa: E731
    dev = r.dev
    mv = lambda x: {q: v.to(dev) for q, v in x.items()} if isinstance(x, dict) else x.to(dev)                         # noqa: E731
    segs, tgts, infos = ([mv(cat([o[i][e] for o in res])) for e in range(nseg)] for i in range(3))
    seg_k, kinds, delay, ramp = (mv(cat([o[i] for o in res])) for i in range(3, 7))
    return (segs, tgts, infos, seg_k, kinds, delay, ramp), Set(groups, parts, n, dims, tails, 0 if after is None else CARRIED-1)


def join(lin, mech):
    """The episodes of the pair bank followed by the mechanistic ones, as one draw."""
    if lin is None:
        return mech
    cat = lambda a, b: {q: torch.cat((a[q], b[q])) for q in a} if isinstance(a, dict) else torch.cat((a, b))   # noqa: E731
    return tuple([cat(x, y) for x, y in zip(a, b)] if isinstance(a, list) else cat(a, b) for a, b in zip(lin, mech))

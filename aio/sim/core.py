"""The simulation core: one batch of episodes, stepped as a pure function (docs/sim.md).

A `Program` is everything about n episodes that is fixed before the first step: the plants (aio.sim.prior),
their parameters in every segment between events, the published targets, limits and priorities, the field
layer, and what only the simulator knows (aio.sim.model: exact local step responses, the teacher's gains,
identification labels). `reset` gives the state at decision 0 and `step` the next state:

    prog = build(n, seed, device)
    st = reset(prog)
    for k in range(HORIZON):
        obs = observe(prog, st, k)             # the 234-number packet the controller sees
        a = ...                                # moves in [-1, 1] of each MV's slew limit
        st, out = step(prog, st, a, k)         # out: cost and its parts on the true values, failure flag

Nothing is written in place and nothing branches on data, so the loop can be differentiated end to end
(the cost with respect to every earlier move) and every environment of every kind of plant runs in the same
few dozen kernels. All episodes of a batch are at the same decision k.

Signals for training, beside the packet and the cost: `teacher` (the move of a dynamic-matrix controller
that knows the plant's exact model, with the model following a process event only 10-40 decisions late),
`prog.labels(k)` (the local model in the packet's units), `privileged` (model, true state and loads for a
critic).

Conventions: cost = priority x squared tracking error in bands + 10 x squared zone violation in bands
+ 0.001 x squared move in slew units, on the true values; a tracked CV is compared with its target as it was
one dead time ago (the shortest from any MV in service that moves it noticeably), so the reference is
reachable.

MV limits are the operator's (prior.ULO..UHI of the span unless an event narrows them, down to an MV out
of service): the packet gives every MV's position inside them, and the plant keeps the command inside
them (a command outside newly narrowed limits comes back at the slew rate). At a constrained operating
point (prior.optimum) the targets cannot all be held; a tracked CV then counts as outside only beyond a
band past the stretch between its target and its value at the best reachable steady point
(`Program.allowance`), and a zone CV that sits at its limit there only beyond a band past where the best
point has it, both widened by what the wandering loads do there that no valve is left to take out.
Zone limits, like targets, are charged one dead time after they are published.

A resting row (prior: the flow of a spare MV, a tracked row with a wide band) is counted like any tracked
row. The cost charges it against the reachable resting value (where the best steady point has it: the
resting value itself unless the point is constrained), so that what it costs to be away from there is the
band's and no standing pull on the process CVs, which rank above it.
"""
import math
import torch

from aio import interface as IF
from aio.sim import prior as PR, model as MD, mech as ME
from aio.sim.prior import DMAX, HORIZON, NSEG, BIG, SEG_FIELDS, TASK_FIELDS

NZ, NUNI = PR.NDQ+3*IF.NCV, 4*IF.NCV      # normal and uniform draws of one step (the default size; a Program's own: prog.nnorm, prog.nuni)
HUBER = 3.                       # bands beyond which the training cost grows linearly
ZONE_BACK = .3                   # the teacher keeps a zone row this far inside its limit (bands; a quarter of the zone at most): held on the limit itself it is beyond it half the time
WEAK = 4.                        # the steady layer does not chase what takes the valves more than about a span for 2 bands (`_reachable`; bands^2 per span^2)
ROUNDS_BIG = 16                  # a larger size: the steady layer's rounds over the zone rows beyond, each step as far as the cost falls (`_line`)
LEV_STOP = (-.5, 1.5)            # a level's physical stops (its value: 0 and 1 are the ends of its instrument's span)
METER = dict(levels=True, rest=True, q_rest=1., q_lev=1e-3, floor=.02, gate=3., lag=True, lev_model=False, q_min=1e-4)     # the teacher's prediction corrected from the instruments alone (`Program.set_teacher`, `_meter`); chosen on the tuning batch (docs/sim.md, section 26)
METER_BIG = dict(lev_model=True)  # a larger size: a level's inflow walk by the plant's own model, q_min at the least (docs/sim.md, section 15)
METER_KEYS = ('eo', 'ev', 'es', 'eh', 'ep', 'eg')              # what it keeps in the state


def huber(x, d=HUBER):
    """x^2 up to d, 2 d |x| - d^2 beyond (Huber): the cost a gradient is taken of, so that one large
    transient does not outweigh everything else. Exactly the cost inside d, so the trade-off between the
    CVs at a constrained point stays the cost's own (d is widened there to hold the reachable optimum)."""
    ax = x.abs()
    return torch.where(ax <= d, x*x, 2*d*ax-d*d)


def _layout(d, fields):
    lay, a = {}, 0
    for q in fields:
        s = tuple(d[q].shape[1:])
        lay[q] = (a, a+math.prod(s), s); a += math.prod(s)
    return lay, a


def _flat(d, fields):
    return torch.cat([d[q].flatten(1) for q in fields], 1)


def _view(flat, lay, fields=None):
    return {q: flat[:, a:b].reshape(flat.shape[0], *s) for q, (a, b, s) in lay.items() if fields is None or q in fields}


def task_arrays(p, tgt):
    """What the packet publishes about the task in one segment: the target of every row (a zone row's is the
    middle of its limits, a disturbance row's 0), the packet range, the row scale (a band for tracked rows,
    half the range otherwise), the priorities, the limits the cost uses, the MV limits, and the reachable
    targets (the targets, unless the point is constrained). The scoring allowance (alo, ahi, blo, bhi) is
    filled in by the Program, which knows the field layer."""
    trk, zon, dvr = p['trk'] > 0, p['zon'] > 0, p['dvr'] > 0
    one, zero = torch.ones_like(tgt), torch.zeros_like(tgt)
    plo = torch.where(zon & torch.isfinite(p['lo']), p['lo'], zero)
    phi = torch.where(zon & torch.isfinite(p['hi']), p['hi'], one)
    plo = torch.where(dvr, -one, plo)
    full = torch.where(trk, tgt, (plo+phi)/2)
    scale = torch.where(trk, p['bandf'], (phi-plo)/2)
    lo = torch.where(zon & torch.isfinite(p['lo']), p['lo'], -1e6*one)
    hi = torch.where(zon & torch.isfinite(p['hi']), p['hi'], 1e6*one)
    return dict(tgt=full, plo=plo, phi=phi, scale=scale, cvw=p['cvw'], lo=lo, hi=hi, ulo=p['ulo'], uhi=p['uhi'],
                rtgt=torch.where(trk, p['rtgt'], full))


class Program:
    """n episodes, fixed before the first step. Tensors are on `device`, float32 (float64 on request)."""

    def __init__(self, r, n, device=None, dtype=torch.float32, teacher=True, seed=0, quiet=False, exc_p=PR.EXC_P, mech=0., plants=None, dims=PR.DIMS,
                 after=None, menu=None, nseg=NSEG, shape=None, blind=False, event_seed=None):
        """after: the `tail` of the Program of the window before (or that Program); this one is the next HORIZON
        decisions on the same plants (`Program.next`, `build(n, seed, after=tail)`): the plants, their
        instruments and what wanders around them stay, the last two segments are carried with the plant as
        commissioned (prior.draw_episode, `tail`), new events follow at the unit's pace, and the state a
        rollout ended in goes on with `resume`. A tail stays where it was drawn (the CPU, float64) and
        pickles: the next window can be drawn in another process while this one runs.
        menu: what changes in this window (prior.Menu; None: the generator as it was); nseg: segments a window
        can have (events that come faster than usual need more than NSEG). shape: the linear plants' roots kept to
        a range (prior.draw_roots; prior.SMALL). blind: the packet leaves out the measured disturbances (the
        same seed with and without: the same plants, events and noise, once with the disturbances measured and
        once not; the flag can also be set on a Program that is there).
        event_seed: optional separate seed for event schedule, initial off-steady draw, and step noise while
        `seed` draws plant dynamics (None: bit-identical single-seed RNG)."""
        dims = PR.SIZES[dims] if isinstance(dims, str) else tuple(dims)
        after = after.tail if isinstance(after, Program) else after
        if after is not None:
            assert n == after['n'], f"the next window of {after['n']} episodes, asked for {n}"
            mech, dims, exc_p, quiet = (after['n']-after['nl'])/after['n'], after['dims'], after['exc_p'], after['quiet']
        self.dims = NCV, NMV, ND, NQ, NDQ, NU = PR.sizes_of(*dims)          # the size of this batch: rows, MVs, measured disturbances, unmeasured loads, both, input channels
        self.ncv, self.nmv, self.nd, self.nq, self.ndq, self.nu = self.dims
        self.nzmax = PR.nzmax(*dims)                                 # zone CVs a plant can have, and the rows the teacher predicts
        self.nr = NMV+self.nzmax
        self.nnorm, self.nuni = NDQ+3*NCV, 4*NCV                     # normal and uniform draws of one step
        self.size = NCV*IF.CF+NMV*IF.MF+NCV*NMV                      # length of the packet
        self.priv = sum(math.prod(v) for v in priv_fields(*dims).values())       # and of the privileged vector
        nmech = int(round(n*mech))                                # the last nmech episodes run on plants of the engine (aio.sim.mech)
        self.nl, self.mech = n-nmech, None
        rl, rm, r = r.fork(), r.fork(), r.fork()                  # streams of the linear episodes, of the mechanistic ones, of the field layer and the start (all n)
        self.gen, lin = r.dev, {}
        self.event_seed = event_seed
        if event_seed is not None:
            re = PR.Rng(event_seed, self.gen)
            rel_ev, r_ev = re.fork(), re.fork()
        else:
            rel_ev, r_ev = None, r
        self.s0 = s0 = 0 if after is None else PR.CARRIED-1         # the segment in force at decision 0
        self.nseg, self.menu, self.blind = nseg, menu, blind
        draw = PR.draw_episode(rl, n-nmech, exc_p, dims, tail=after['lin'] if after is not None else None, keep=lin, menu=menu, nseg=nseg, shape=shape, event_r=rel_ev) if n > nmech else None
        if nmech:
            md, self.mech = ME.draw(rm, nmech, plants, exc_p, dims=dims, after=after['mech'] if after is not None else None, menu=menu, nseg=nseg)
            draw = ME.join(draw, md)
        segs, tgts, infos, seg_k, kinds, delay, ramp = draw
        if after is None:
            field = PR.draw_field(r, segs[0], infos[0])
            if quiet:                                             # no field layer, start at rest (tests, step tests)
                field = {q: torch.zeros_like(v) for q, v in field.items()}
        else:                                                     # the same instruments and wander, HORIZON decisions on
            field = dict(after['field'])
            field['sin_ph'] = torch.remainder(field['sin_ph']+field['sin_w']*HORIZON, 2*math.pi)
            field['an_ph'] = torch.remainder(field['an_ph']+HORIZON, field['an_T'].clamp_min(1.))
        # whether the process is no longer the commissioned one when this window ends (a process event in this window or in one before it)
        changed = ((kinds == PR.PROC) | (kinds == PR.BOTH))[:, s0+1:].any(1).cpu()
        self.was = None if after is None else torch.as_tensor(after.get('changed', True)).expand(n)       # and when it started (a tail without it: taken as changed)
        if after is not None:
            changed = changed | self.was
        # what the next window on these plants is drawn from (on the generator's device, float64; not moved with the Program)
        self.tail = dict(n=n, nl=n-nmech, dims=tuple(dims), exc_p=exc_p, quiet=quiet, lin=lin, mech=self.mech, field=field, window=0 if after is None else after['window']+1,
                         changed=changed)
        if menu is not None and menu.wander != 1.:                # this window's wander (the tail keeps the field as drawn)
            field = dict(field)
            field['ou_s'], field['sin_amp'] = field['ou_s']*menu.wander, field['sin_amp']*menu.wander
        dev = torch.device(device) if device is not None else r.dev
        self.n, self.device, self.dtype = n, dev, dtype
        to = lambda a: a.to(dev, dtype)                           # noqa: E731
        p0 = segs[0]
        hid = PR.held(p0)                                         # MVs that are not the controller's (prior.THIN): they stand where they are, and the packet, the
        ctl = segs                                                # teacher and the references have no such MV (the plant has more setpoints than MVs)
        if bool(hid.any()):
            ctl = [dict(p, mvm=p['mvm']*~hid, link=p['link']*~hid[:, None, :]) for p in segs]
        pc = ctl[0]
        self.lay, _ = _layout(p0, SEG_FIELDS)
        self.P = to(torch.stack([_flat(p, SEG_FIELDS) for p in segs], 1))
        tasks = [task_arrays(p, t) for p, t in zip(segs, tgts)]
        self.hold_k = (seg_k+self._allow(segs, infos, tasks, field, seg_k, kinds, ramp)).to(dev)
        self.tlay, _ = _layout(tasks[0], TASK_FIELDS)
        self.T = to(torch.stack([_flat(t, TASK_FIELDS) for t in tasks], 1))
        self.seg_k, self.kinds, self.ramp = seg_k.to(dev), kinds.to(dev), to(ramp)
        self.k_first = self.seg_k[:, 1].clamp(0, HORIZON)         # scoring starts at the first event (a window that carries on: at once)
        # from when a segment counts as settled: the unit can have lined out after the event that opened it
        line = [PR.lineout(p0, infos[max(e-1, 0)], infos[e], ramp[:, e].to(PR.F)) for e in range(nseg)]
        self.settled_k = (seg_k+PR.up(torch.stack(line, 1)).long()).to(dev)
        keep = ('trk', 'zon', 'dvr', 'prc', 'mvm', 'dact', 'uact', 'link', 'rst', 'spr', 'integ', 'tint', 'lam', 'vc', 'nlag', 'bias',
                'bandf', 'sband', 'aa', 'slew', 'warp', 'uref', 'kap', 'dt', 'nm', 'T', 'rth', 'store', 'arch', 'linear', 'mech')
        c = {q: to(pc[q]) for q in keep}
        c.update({q: to(v) for q, v in field.items()})
        c['rowm'] = to(((p0['prc']+p0['dvr']) > 0).to(PR.F))
        c['jload'] = p0['jload'].long().to(dev)
        c['dvsel'] = p0['dvsel'].long().to(dev)
        c['psign'] = to(torch.sign((pc['K'][:, :, :NMV]*pc['link']).sum(1))*pc['mvm'])
        if bool(hid.any()):
            c['mvf'] = to(p0['mvm'])                              # the MVs whose flow enters the plant: the controller's, and those held
            c['thin'] = hid.any(-1).to(dev)
        c['lrho'] = to(torch.log1p(p0['lam']/p0['tint'])[:, :NMV])
        c['zrow'], zon = MD.zone_rows(p0)
        c['zrow'], c['zslot'] = c['zrow'].to(dev), zon.to(dev)
        self.c = c
        self.ar = torch.arange(n, device=dev)
        self._ks = torch.arange(HORIZON+4, device=dev, dtype=torch.long)
        # start: the steady point of segment 0, half the episodes off it (inputs, and integrating levels)
        un, v = infos[s0]['un'], infos[s0]['v']
        if after is None:
            off = r_ev.bern(0. if quiet else .5, n)[:, None]
            u0 = torch.where(off, torch.minimum(torch.maximum(un+r_ev.u(n, NMV, lo=-.15, hi=.15), p0['ulo']), p0['uhi']), un)*p0['mvm']
            v0 = torch.where(off & (p0['integ'] > 0) & (p0['mech'] == 0)[:, None], v+p0['bandf']*r_ev.u(n, NCV, lo=-5., hi=5.), v)
        else:                                                     # (run on its own, without the state of the window before: from rest)
            u0, v0 = un*p0['mvm'], v
        self.start = dict(un=to(un), u=to(u0), v=to(v0), ue=to(infos[s0]['ue']), tgt=to(tgts[s0]))
        # the mechanistic episodes: per batch of plants the engine and its maps, the rows it feeds, the start states
        self.mg = self.mech.runtime(self.nl, dev, dtype) if self.mech is not None else []
        c['mrow'] = to(p0['prc']*(p0['mech'] > 0)[:, None])
        mx = torch.zeros(n, max([g['m'].nx for g in self.mg]+[1]), dtype=dtype, device=dev)
        for g in self.mg:
            mx[g['idx'], :g['m'].nx] = g['x0']
        self.start['mx'] = mx
        self.seed = int(seed if event_seed is None else event_seed)+90_001
        self.g = torch.Generator(); self.g.manual_seed(self.seed)                  # on the CPU: the same episodes on every device
        self.Fr, self.meter = None, None
        if teacher:
            self._model(ctl, infos, tasks, seg_k, kinds, delay, to)

    @staticmethod
    def _allow(segs, infos, tasks, field, seg_k, kinds, ramp):
        """The scoring allowance of every segment. Tracked rows, in bands relative to the target
        (alo <= 0 <= ahi): the stretch to the reachable optimum, widened by twice the standing deviation of
        what the wandering loads do to the CV with the valves at a limit staying there (plus the sinusoid's
        amplitude). Zone rows (ahi >= 0, alo 0): the bands beyond its limit a CV may be before it counts,
        0 unless the best steady point has it at the limit (then where it sits there plus a band, widened
        the same way). blo, bhi: the same joined with the previous segment's, in force while a load that
        had the unit constrained is still on its way back (the event's ramp plus the unit's settling
        time)."""
        sig = field['ou_s']/torch.sqrt((1-field['ou_a']**2).clamp_min(1e-12))
        for p, info, t in zip(segs, infos, tasks):
            E = PR.unremovable(p, info)*(p['integ'] == 0)[:, :, None]
            wide = 2*torch.sqrt(((E*sig[:, None, :])**2).sum(-1))+(E.abs()*field['sin_amp'][:, None, :]).sum(-1)
            es = (t['rtgt']-t['tgt'])/p['bandf']*p['trk']
            t['alo'] = (es-wide*p['trk']).clamp_max(0.)
            t['ahi'] = (es+wide*p['trk']).clamp_min(0.)+torch.where(p['zal'] > 0, p['zal']+wide, torch.zeros_like(wide))*p['zon']
        p0 = segs[0]
        settle = PR.settling(p0)
        hold = torch.zeros_like(seg_k)
        for e in range(1, len(tasks)):
            was = PR.constrained(segs[e-1], tasks[e-1]['tgt'])    # from a constrained point, a load or the process on its way: an over_load taken back, or (Menu.ride) an ordinary event there
            back = was & ((segs[e]['dlev'] != segs[e-1]['dlev']).any(-1) | (kinds[:, e] == PR.PROC) | (kinds[:, e] == PR.BOTH))
            hold[:, e] = torch.where(back, PR.down(ramp[:, e]+settle).long(), hold[:, e])
            tasks[e]['blo'] = torch.minimum(tasks[e]['alo'], tasks[e-1]['alo'])
            tasks[e]['bhi'] = torch.maximum(tasks[e]['ahi'], tasks[e-1]['ahi'])
        tasks[0]['blo'], tasks[0]['bhi'] = tasks[0]['alo'], tasks[0]['ahi']
        return hold

    def _model(self, segs, infos, tasks, seg_k, kinds, delay, to, chunk=1024):
        """Exact model of every segment in use: its linear core for the teacher, the teacher's gains and
        schedule, the identification labels, and the formula PI of the start plant."""
        NCV, NMV, ND, NQ, NDQ, NU = self.dims
        n, dev, f = self.n, self.device, self.dtype
        L, P = MD.LAGS, MD.P_DMC
        NR, NZMAX = self.nr, self.nzmax
        NZP = NZMAX*len(MD.ZL)
        chunk = max(8, chunk*IF.NCV*PR.NU//(NCV*NU))               # segments at a time: the work arrays are chunk x lags x rows x inputs (1024 at the default size)
        self.Fr = torch.zeros(n, self.nseg, L+1, NR, NU, dtype=f, device=dev)
        self.Az = torch.zeros(n, self.nseg, NZP, MD.M_DMC*NMV, dtype=f, device=dev)
        self.wz = torch.zeros(n, self.nseg, NZMAX, dtype=f, device=dev)
        self.Sw = torch.zeros(n, self.nseg, P, NMV, NMV, dtype=f, device=dev)
        self.HX = torch.zeros(n, self.nseg, MD.M_DMC*NMV, MD.M_DMC*NMV, dtype=f, device=dev)
        self.wy = torch.zeros(n, self.nseg, NMV, dtype=f, device=dev)
        self.Gs = torch.zeros(n, self.nseg, NMV, NMV, dtype=f, device=dev)       # steady gains of the tracked rows per unit of valve flow
        self.Gz = torch.zeros(n, self.nseg, NZMAX, NMV, dtype=f, device=dev)     # and of the zone rows
        self.zs = torch.ones(n, self.nseg, NZMAX, len(MD.ZL), dtype=f, device=dev)      # share of a zone row's steady answer in place at the lags ZL
        self.trade = torch.ones(n, self.nseg, dtype=torch.bool, device=dev)    # whether the steady-state layer's best steady point takes the targets' place (not on gains that are only roughly known: `install`)
        self.lab = torch.zeros(n, self.nseg, NMV, NCV*MD.NLAB, dtype=f, device=dev)
        wide = MD.KNOTS[max(MD.KNOTS)]                             # the labels at the widest set of lags too (it holds LAB_LAGS)
        nine = [wide.index(t) for t in MD.LAB_LAGS]
        self.labw = torch.zeros(n, self.nseg, NMV, NCV*len(wide), dtype=f, device=dev)
        self.phis = torch.ones(n, self.nseg, dtype=f, device=dev)
        keys = ('K', 'p1r', 'p1i', 'p2r', 'p2i', 'c0r', 'c0i', 'c1r', 'c1i', 'dead', 'gam', 'mvm', 'dact', 'aa', 'tlag', 'nlag',
                'integ', 'lam', 'tint', 'wa', 'prc', 'jload', 'uref', 'kap', 'warp', 'trk', 'zon', 'bandf', 'cvw', 'dlev')
        ei, si = (seg_k < BIG).nonzero(as_tuple=True)             # episode and segment of every segment in use
        order = torch.argsort(si, stable=True)                    # segment 0 of every episode first
        ei, si = ei[order], si[order]
        stack = lambda ds, qs: {q: torch.stack([d[q] for d in ds], 1) for q in qs}    # noqa: E731
        ps, ins, sc = stack(segs, keys), stack(infos, ('un', 'v')), stack(tasks, ('scale',))['scale']
        mvf = self.c['mvf'].to(ps['mvm']) if 'mvf' in self.c else None
        S0 = []
        for a in range(0, ei.numel(), chunk):
            e, g = ei[a:a+chunk], si[a:a+chunk]
            pa = {q: v[e, g] for q, v in ps.items()}
            ia = {q: v[e, g] for q, v in ins.items()}
            Fr = MD.response(pa if mvf is None else dict(pa, mvm=mvf[e]), phi=PR.phi_of(pa, ia['un'], pa['dlev']))     # (an MV held moves the plant too: when its valve's characteristic changes)
            if self.mech is not None:                             # a mechanistic segment: the response of the plant linearised at its steady point
                sel = (e >= self.nl).nonzero().flatten()
                if sel.numel():
                    Fr[sel] = self.mech.response((e[sel]-self.nl).cpu(), g[sel].cpu(), {q: pa[q][sel].cpu() for q in ('aa', 'mvm', 'dact')}).to(Fr.device)
            Sf, Sz, S, Sl, phi = MD.local(Fr, pa, ia, wide)
            # the teacher's errors are taken before the output curve and weighted to bands at the steady point,
            # tracked rows by their priority and zone rows by the cost's factor 10
            band = PR.hprime(ia['v'], pa['wa'])/pa['bandf']
            wy = torch.sqrt(pa['cvw'])*pa['trk']*band
            zrow, zon = MD.zone_rows(pa)
            wz = math.sqrt(10.)*torch.gather(band, 1, zrow)*zon
            ed, gd = e.to(dev), g.to(dev)
            self.Gz[ed, gd] = to(torch.gather(pa['K'][:, :, :NMV], 1, zrow[:, :, None].expand(-1, -1, NMV))*zon[:, :, None]*pa['mvm'][:, None, :])
            self.Fr[ed, gd], self.wy[ed, gd], self.phis[ed, gd] = to(Fr[:, :, :NR]), to(wy[:, :NMV]), to(phi)
            self.Gs[ed, gd] = to(pa['K'][:, :NMV, :NMV]*pa['mvm'][:, None, :])
            isload = (torch.arange(NMV, device=wz.device)[None] == pa['jload'].long()[:, None]) & (pa['warp'] > 0)[:, None]
            Sw, HX, Az = MD.dmc_design(Sf, wy[:, :NMV], pa['mvm'], Sz, wz, 1+(MD.LOAD_SUPP-1)*isload)
            self.Sw[ed, gd], self.HX[ed, gd], self.Az[ed, gd], self.wz[ed, gd] = to(Sw), to(HX), to(Az), to(wz)
            self.lab[ed, gd], self.labw[ed, gd] = to(MD.labels(Sl[:, nine], sc[e, g])), to(MD.labels(Sl, sc[e, g]))
            end = Sz[:, -1]                                       # how much of what the MVs do to a zone row in the end is there by each lag
            self.zs[ed, gd] = to(((Sz[:, list(MD.ZL)]*end[:, None]).sum(-1)/(end**2).sum(-1).clamp_min(1e-30)[:, None]).clamp(0., 1.).transpose(1, 2))
            if a < n:
                S0.append(S[:max(0, n-a)])
        S0 = torch.cat(S0, 0)
        self.S0 = to(S0)                                          # step responses of the start plant (what a step test finds)
        kc, ti, row = MD.pi_tuning(S0, segs[0])
        self.pi = dict(kc=to(kc), ti=to(ti), row=row.to(dev))
        # the teacher's model follows a process event `delay` decisions late (or at the next other event)
        proc = (kinds == PR.PROC) | (kinds == PR.BOTH)
        late = (seg_k+delay*proc).flip(1).cummin(1).values.flip(1)
        mark = proc.clone()                                       # the segment whose process is in force at each segment. In a window that carries on, the segments
        if self.was is not None:                                  # carried are of the commissioned process (segment 0) only where no process event came in the windows
            mark[:, 1] |= self.was.to(mark.device)                # before: the kinds of the two carried say nothing of the events before them
        self.plant = torch.where(mark, torch.arange(proc.shape[1], device=proc.device)[None], 0).cummax(1).values.to(dev)
        self.exact = torch.ones(n, self.nseg, dtype=torch.bool, device=dev)      # the model of the segment is the plant's own (not one put in its place: `install`)
        fixed = seg_k.clone(); fixed[:, 1:] = BIG
        self._teach = dict(late=late.to(dev), now=seg_k.to(dev), fixed=fixed.to(dev))
        self._meters, self._ren, self._installs = {}, None, 0
        self.set_teacher()

    def set_teacher(self, mode='late', loads=True, meter=None):
        """When the teacher's model follows an event: 'late' (10-40 decisions after a process event; the
        training teacher), 'now' (at the event: the re-identified upper reference), 'fixed' (never: the
        commissioning model, as a step-test DMC). loads: whether its prediction sees the unmeasured loads
        move (the training teacher does; a controller in the field cannot).
        meter: what its prediction is corrected with at every decision. None: the plant's true values before
        the instruments (every row before its output curve, a level's true height and true net inflow). True,
        or a dict of METER's fields: what the instruments report alone (`_meter`); levels=False or rest=False
        leaves the integrating rows, or the others, on the true values."""
        NCV, NMV, ND, NQ, NDQ, NU = self.dims
        self.eff_k = self._teach[mode]
        self.ffm = torch.tensor([1.]*ND+[float(loads)]*NQ, dtype=self.dtype, device=self.device)
        self.meter = None
        if meter is not None and meter is not False:              # (a dict with nothing in it: METER as it is)
            m = {**METER, **(METER_BIG if self.dims != PR.sizes_of(*PR.DIMS) else {}), **(meter if isinstance(meter, dict) else {})}     # (a field named in meter overrides METER_BIG's)
            key = tuple(sorted(m.items()))+(bool(loads),)          # (lev_model: what the model leaves unexplained depends on the loads fed forward)
            if key not in self._meters:
                self._meters[key] = _meter_gains(self, m)
            self.meter = self._meters[key]
        return self

    @torch.no_grad()
    def install(self, Fr, phi, seg=0, lam=None, end=None, trade=True):
        """A model that is not the exact one in the place of segment `seg`'s (a commissioning model found by a test on
        the plant: aio.sim.steptest), and the teacher's design redone on it. Fr (n, LAGS+1, nr, nu) as model.response gives
        it (every row before its integrator and output curve, per unit of valve flow and of disturbance, at time scale 1);
        phi (n,) the time scale of the point it was found at; lam: the move suppressions times this (model.dmc_design);
        end (n, nr, NMV): where the responses end, if that is known beyond their last lag (the target layer's steady gains).
        trade False: the steady-state layer keeps the zone limits backed off but does not put its best steady point in the
        targets' place (gains known to some tens of per cent do not give the plant's trade-off).
        The weights of the rows, the valve and output curves and the levels' integration stay the plant's own: what differs
        from the exact-model teacher is the response alone."""
        c, NMV, NR, f, dev = self.c, self.nmv, self.nr, self.dtype, self.device
        Fr, phi = Fr.to(dev, torch.float64), phi.to(dev, torch.float64)
        v = MD.values(Fr[..., :NMV], phi, {q: c[q][:, :NR].double() for q in ('lam', 'tint', 'integ')})
        zr = c['zrow'].clamp_max(NR-1)
        Sz = torch.gather(v, 2, zr[:, None, :, None].expand(-1, v.shape[1], -1, NMV))*c['zslot'][:, None, :, None]
        isload = (torch.arange(NMV, device=dev)[None] == c['jload'][:, None]) & (c['warp'] > 0)[:, None]
        Sw, HX, Az = MD.dmc_design(v[:, :, :NMV], self.wy[:, seg].double(), c['mvm'].double(), Sz, self.wz[:, seg].double(), 1+(MD.LOAD_SUPP-1)*isload, lam=lam)
        self.Fr[:, seg], self.phis[:, seg] = Fr.to(f), phi.to(f)
        self.trade[:, seg], self.exact[:, seg] = trade, False
        self._installs += 1
        self.Sw[:, seg], self.HX[:, seg], self.Az[:, seg] = Sw.to(f), HX.to(f), Az.to(f)
        end = Fr[:, -1, :, :NMV] if end is None else end.to(dev, torch.float64)       # the steady gains the target layer works on
        self.Gs[:, seg] = (end[:, :NMV]*c['mvm'][:, None, :]).to(f)
        self.Gz[:, seg] = (torch.gather(end, 1, zr[:, :, None].expand(-1, -1, NMV))*c['zslot'][:, :, None]*c['mvm'][:, None, :]).to(f)
        end = Sz[:, -1]
        self.zs[:, seg] = ((Sz[:, list(MD.ZL)]*end[:, None]).sum(-1)/(end**2).sum(-1).clamp_min(1e-30)[:, None]).clamp(0., 1.).transpose(1, 2).to(f)
        return self

    def next(self, seed=None, menu=None, nseg=None):
        """The next HORIZON decisions on the same plants: a Program whose rollout starts with `resume(prog, st)`
        from the state this one's ended in. Windows chained this way are one stay of any length on a plant:
        the events keep coming at the unit's pace, nothing is reset. menu, nseg: this window's (None: as here)."""
        seed = self.seed-90_001+1 if seed is None else seed
        return Program(PR.Rng(seed, self.gen), self.n, device=self.device, dtype=self.dtype, teacher=self.Fr is not None, seed=seed, after=self.tail,
                       menu=self.menu if menu is None else menu, nseg=self.nseg if nseg is None else nseg, blind=self.blind)

    # ---- what is in force at decision k
    def seg_at(self, k):
        return (self.seg_k <= k).sum(1)-1

    def since(self, k):
        """(n,) decisions since the last event that changed something, at decision k (BIG: none so far). In a
        window that goes on, the events it carries count from where they were (negative decisions)."""
        real = (self.seg_k <= k) & (self.kinds >= 0)
        last = torch.where(real, self.seg_k, torch.full_like(self.seg_k, -BIG)).max(1).values
        return torch.where(last > -BIG, k-last, torch.full_like(last, BIG))

    def params(self, seg, k=None):
        """Plant parameters of segment `seg`; with k, as they are during decision k (a gradual change is
        still on its way from the previous segment's values)."""
        P1 = self.P[self.ar, seg]
        if k is not None:
            w = ((k+1-self.seg_k[self.ar, seg])/self.ramp[self.ar, seg].clamp_min(1.)).clamp(0., 1.)[:, None]
            P0 = self.P[self.ar, (seg-1).clamp_min(0)]
            P1 = P0+w*(P1-P0)
        return _view(P1, self.lay)

    def task(self, seg):
        return _view(self.T[self.ar, seg], self.tlay)

    @property
    def con(self):
        """(n, nseg) the segments in use whose operating point is a constrained one (its reachable targets are not
        its targets): where an event took the unit, and with Menu.ride where ordinary events found or left it."""
        t = _view(self.T.flatten(0, 1), self.tlay)
        off = (t['rtgt']-t['tgt']).abs().view(self.n, self.T.shape[1], -1)*self.c['trk'][:, None, :]
        return (off.amax(-1) > 1e-9) & (self.seg_k < PR.BIG)

    def allowance(self, k):
        """(n, NCV) twice: how far below and above its target (bands, <= 0 and >= 0) a tracked CV may sit
        at decision k before the band starts: 0 except at and just after a constrained operating point."""
        seg = self.seg_at(k)
        t = self.task(seg)
        held = (k < self.hold_k[self.ar, seg])[:, None]
        return torch.where(held, t['blo'], t['alo']), torch.where(held, t['bhi'], t['ahi'])

    def teacher_seg(self, k):
        return (self.eff_k <= k).sum(1)-1

    def renewals(self):
        """{k: episodes} the decisions after which `step` works the teacher's prediction out again (`_renew`), and of
        which episodes: its model has just been renewed, or (while that model is of the process as it now is, and is
        the plant's own) a gradual change it has in full has just arrived, or the plant's clock moves and REBASE
        decisions have gone by. All of it follows from the schedules, so it is worked out once for the window (in
        `step` it was a dozen operations and a wait for the device at every decision)."""
        key = (self.eff_k.data_ptr(), self._installs, REBASE)
        if self._ren is None or self._ren[0] != key:
            k1 = torch.arange(1, HORIZON+1, device=self.device)[None, :, None]
            sg0 = (self.eff_k[:, None, :] <= k1-1).sum(-1)-1                         # (n, HORIZON) the teacher's segment at k and at k+1
            sg1 = (self.eff_k[:, None, :] <= k1).sum(-1)-1
            ramp, at = torch.gather(self.ramp, 1, sg1), torch.gather(self.seg_k, 1, sg1)
            due = (ramp > 0) & (at+ramp.ceil().long() == k1[..., 0])
            if REBASE:
                due = due | ((self.c['warp'] > 0)[:, None] & (k1[..., 0] % REBASE == 0))
            now = (self.seg_k[:, None, :] <= k1).sum(-1)-1
            due = due & (torch.gather(self.plant, 1, sg1) == torch.gather(self.plant, 1, now))
            due = (due | (sg1 != sg0)) & torch.gather(self.exact, 1, sg1) & (self.c['mech'] == 0)[:, None]
            ks = due.any(0).nonzero().flatten().tolist()
            self._ren = (key, {k: due[:, k].nonzero().flatten() for k in ks})
        return self._ren[1]

    def labels(self, k, knots=MD.NLAB):
        """(n, NMV, NCV*knots) identification target at decision k: the true local response in packet units at
        LAB_LAGS, or at the wider set of model.KNOTS with that many lags."""
        assert knots in (MD.NLAB, max(MD.KNOTS))
        return (self.lab if knots == MD.NLAB else self.labw)[self.ar, self.seg_at(k)]

    def _k_tensor(self, k):
        """0-D int64 tensor for decision `k` on `self.P.device` (O(1) slice, zero H2D copy for k <= HORIZON+3)."""
        dev = self.P.device
        ks = getattr(self, '_ks', None)
        if ks is None or ks.device != dev:
            ks = torch.arange(HORIZON+4, device=dev, dtype=torch.long)
            self._ks = ks
        ki = int(k)
        if 0 <= ki < ks.shape[0]:
            return ks[ki]
        return torch.tensor(ki, device=dev, dtype=torch.long)

    def noise(self):
        """Random draws of one step (own generator, so an episode is reproducible; drawn on the CPU, so it is
        the same episode on every device)."""
        z = torch.randn(self.n, self.nnorm, generator=self.g, dtype=self.dtype)
        u = torch.rand(self.n, self.nuni, generator=self.g, dtype=self.dtype)
        return z.to(self.device), u.to(self.device)

    def prefetch_noise(self, k0=0, H=None, dev=None):
        """Pre-generate `H` steps of `(z, u)` noise on CPU in one pinned buffer and transfer asynchronously to `dev`.
        100% bit-identical to calling `noise()` `H` times in sequence (leaving `self.g` in the exact same state).
        Accepts `(k0, H, dev=None)` or `(H, dev=None)` or `(H=...)`."""
        if H is None:
            H = int(k0)
        elif isinstance(H, (str, torch.device)):
            dev, H = H, int(k0)
        else:
            H = int(H)
        dev = self.device if dev is None else torch.device(dev)
        if H <= 0:
            return (torch.empty((0, self.n, self.nnorm), dtype=self.dtype, device=dev),
                    torch.empty((0, self.n, self.nuni), dtype=self.dtype, device=dev))
        nz, nu = H*self.n*self.nnorm, H*self.n*self.nuni
        pin = dev.type == 'cuda' and torch.cuda.is_available()
        flat = torch.empty(nz+nu, dtype=self.dtype, pin_memory=pin)
        z_cpu = flat[:nz].view(H, self.n, self.nnorm)
        u_cpu = flat[nz:].view(H, self.n, self.nuni)
        for j in range(H):
            torch.randn((self.n, self.nnorm), generator=self.g, dtype=self.dtype, out=z_cpu[j])
            torch.rand((self.n, self.nuni), generator=self.g, dtype=self.dtype, out=u_cpu[j])
        flat_d = flat.to(dev, non_blocking=pin)
        return flat_d[:nz].view(H, self.n, self.nnorm), flat_d[nz:].view(H, self.n, self.nuni)


def build(n, seed=0, device='cpu', gen_device='cpu', event_seed=None, **kw):
    """A fresh batch of n episodes. The draw runs on gen_device (float64) and is moved to device.
    after=tail (a Program's `tail`): the next window on the same plants instead (the seed gives its events and noise).
    event_seed: optional separate seed for events/initial offset/step noise while `seed` draws plant dynamics."""
    return Program(PR.Rng(seed, gen_device), n, device=device, seed=seed, event_seed=event_seed, **kw)


# ---------------------------------------------------------------- state
def _ytrue(v, wa, d, c):
    ND = d.shape[1]//2
    y = PR.h(v, wa)
    y = torch.where(c['dvr'] > 0, torch.gather(d[:, :ND], 1, c['dvsel']), y)
    return y*c['rowm']


def _flow(u, gam, mvm):
    return u.clamp_min(1e-6)**gam*mvm


def reset(prog):
    """State at decision 0: the plant at rest at its start point (half the episodes with the inputs and the
    integrating levels moved off it), instruments settled, the teacher's prediction flat. Where the inputs are
    off it, the valves have just been moved there from their steady positions and have not travelled yet: the
    first packet shows that move as the last one made (11-11; it showed none, while the teacher's prediction
    knew of it: told the true responses, the formula of aio.formula lost 4 points of 100 to it)."""
    NCV, NMV, ND, NQ, NDQ, NU = prog.dims
    c, s, n, dev, f = prog.c, prog.start, prog.n, prog.device, prog.dtype
    prog.g.manual_seed(prog.seed)                                 # every rollout of a program sees the same noise
    s0 = torch.full((n,), prog.s0, dtype=torch.long, device=dev)
    P = prog.params(s0)
    w = torch.cat((s['ue'], P['dlev']), 1)
    fast = (P['K']*w[:, None, :]).sum(-1)+c['bias']
    ig = c['integ'] > 0
    rest = torch.where(ig, -c['lam']*(s['v']-c['vc']), fast)       # what the row lags hold at rest
    yt = _ytrue(s['v'], P['wa'], P['dlev'], c)
    z, _ = prog.noise()
    y = yt+c['nw']*z[:, NDQ:NDQ+NCV]
    t0 = prog.task(s0)
    tgt = t0['tgt']
    zc = torch.zeros(n, NCV, dtype=f, device=dev)
    st = dict(u=s['u'], prev_u=s['un']*c['mvm'], ua=s['un'], xv=s['un'],
              buf=w[:, None, :].expand(n, DMAX+1, NU).contiguous(),
              sr=w[:, None, :, None].expand(n, NCV, NU, 2).contiguous(), si=torch.zeros(n, NCV, NU, 2, dtype=f, device=dev),
              l1=rest, l2=rest, v=s['v'], ou=torch.zeros(n, NDQ, dtype=f, device=dev), d=P['dlev'], odr=zc, od=zc,
              ys=yt, cn=zc, held=yt, pend=yt, age=c['an_T'], bad=zc, last=y, y=y, prev_y=y, yt=yt, fl=rest,
              target=tgt, last_target=tgt, thist=tgt[:, None, :].expand(n, DMAX+1, NCV).contiguous(),
              rhist=t0['rtgt'][:, None, :].expand(n, DMAX+1, NCV).contiguous(),
              alo=torch.zeros(n, DMAX+1, NCV, dtype=f, device=dev), ahi=torch.zeros(n, DMAX+1, NCV, dtype=f, device=dev),
              lo=t0['lo'][:, None, :].expand(n, DMAX+1, NCV).contiguous(), hi=t0['hi'][:, None, :].expand(n, DMAX+1, NCV).contiguous(),
              fail=torch.zeros(n, dtype=torch.bool, device=dev), mx=s['mx'])
    if prog.Fr is not None:                                       # the teacher's prediction: at rest, plus the start offset of the MVs
        dq = _flow(s['u'], P['gam'], c['mvm'])-_flow(s['un'], P['gam'], c['mvm'])
        st['tF'] = rest[:, None, :prog.nr]+torch.einsum('bnij,bj->bni', prog.Fr[:, 0, :, :, :NMV], dq)
        st['qm'] = _flow(s['u'], P['gam'], c.get('mvf', c['mvm']))  # the valve flows its prediction has taken in: those of the commands
        kw = P['K']*w[:, None, :]                                  # what the row lags hold of the inputs it can know of: the valves and measured disturbances, and the loads
        st['k1'] = torch.stack((kw[:, :, :NMV+ND].sum(-1), kw[:, :, NMV+ND:].sum(-1)), -1)
        st['k2'] = st['k1']
        if prog.meter is not None:
            st.update(meter_start(prog, st, P['wa']))
    return st


def detach(st):
    return {q: v.detach() for q, v in st.items()}


def resume(prog, st):
    """The state the window before ended in, as the start of this one (a Program made with `after`, or
    `Program.next`): the plant, the valves, the instruments and every history go on from where they were."""
    st = detach(st)
    f = st['fail']
    if bool(f.any()):                                              # a unit that failed (charged in full in its window) is started again at rest where its plant stands
        rest = reset(prog)
        if 'eo' in st and 'eo' not in rest:                       # (the estimator of a teacher that sees the instruments alone: it is set after this)
            rest.update(meter_start(prog, rest, prog.params(torch.full((prog.n,), prog.s0, dtype=torch.long, device=prog.device))['wa']))
        st = {q: torch.where(f.view(-1, *([1]*(v.dim()-1))), rest[q], v) for q, v in st.items()}
    prog.g.manual_seed(prog.seed)
    return st


# ---------------------------------------------------------------- packet & compiled pure-tensor simulator
_OBS_T_KEYS = ('scale', 'plo', 'phi', 'cvw', 'ulo', 'uhi')
_OBS_C_KEYS = ('rowm', 'dvr', 'trk', 'an_T', 'dt', 'mvm', 'slew', 'psign', 'link')
_OBS_ST_KEYS = ('target', 'y', 'bad', 'prev_y', 'age', 'last_target', 'u', 'prev_u')

_STEP_C_KEYS = ('mvm', 'mvf', 'slew', 'aa', 'ou_a', 'ou_s', 'sin_amp', 'sin_w', 'sin_ph', 'dact',
                'jload', 'uref', 'kap', 'warp', 'od_a', 'od_s', 'od_l', 'bias', 'nlag', 'lam', 'vc',
                'tint', 'integ', 'dvr', 'dvsel', 'rowm', 'as', 'cn_a', 'cn_s', 'nw', 'quant', 'an_T',
                'an_ph', 'spike', 'sband', 'bad', 'prc', 'bandf', 'trk', 'rst', 'zon')
_STEP_ST_KEYS = ('u', 'ua', 'xv', 'ou', 'buf', 'sr', 'si', 'odr', 'od', 'l1', 'l2', 'v', 'mx', 'fail',
                 'ys', 'cn', 'held', 'pend', 'age', 'bad', 'last', 'y', 'target',
                 'thist', 'rhist', 'alo', 'ahi', 'lo', 'hi')
_NEW_ST_KEYS = ('u', 'prev_u', 'ua', 'xv', 'buf', 'sr', 'si', 'l1', 'l2', 'v', 'ou', 'd', 'odr', 'od',
                'ys', 'cn', 'held', 'pend', 'age', 'bad', 'last', 'y', 'prev_y', 'yt', 'fl',
                'target', 'last_target', 'thist', 'rhist', 'alo', 'ahi', 'lo', 'hi', 'fail', 'mx')
_OUT_KEYS = ('cost', 'soft', 'track', 'zone', 'move', 'err', 'erc', 'viol', 'ref', 'fail',
             'alo', 'ahi', 'zlo', 'zhi', 'outside', 'trip')

_SIM_COMPILE = False
_SIM_COMPILE_CPU = False
_SIM_COMPILE_KW = {}
_OBS_COMPILED = None
_STEP_COMPILED = None


def _observe_pure(k_t, seg_k, ar, T_flat, tlay_obs, blind, c_obs, st_obs):
    n = ar.shape[0]
    seg = (seg_k <= k_t).sum(1)-1
    Tf = T_flat[ar, seg]
    s, plo, phi, cvw, ulo, uhi = (Tf[:, a:b].reshape(n, *sh) for a, b, sh in tlay_obs)
    rowm, dvr, trk, an_T, dt, mvm, slew, psign, link = c_obs
    tgt, y, bad_v, prev_y, age, last_target, u, prev_u = st_obs
    bad = bad_v > 0
    zero, one = torch.zeros_like(y), torch.ones_like(y)
    rm = rowm*(1-dvr) if blind else rowm
    cv = torch.stack((torch.where(bad, zero, (y-tgt)/s), torch.where(bad, zero, (y-prev_y)/s),
                      (plo-tgt)/s, (phi-tgt)/s, 1-dvr, cvw, trk, torch.where(bad, zero, one),
                      torch.where(an_T > 0, torch.log1p(age), zero), (torch.log1p(dt)/10)[:, None].expand_as(y),
                      (tgt-last_target)/s, one), -1)*rm[..., None]
    span = (uhi-ulo).clamp_min(1e-9)
    mv = torch.stack((2*(u-ulo)/span-1, 2*(u-prev_u)/span, 2*torch.minimum(slew, span)/span, -torch.ones_like(u), psign,
                      torch.ones_like(u)), -1)*mvm[..., None]
    links = link*rm[:, :, None]
    return torch.cat((cv.flatten(1), mv.flatten(1), links.flatten(1)), -1)


def _step_pure(k_t, a, z, uu, through, stops, dims, lay_items, tlay_items, P_flat, T_flat, seg_k, ramp, hold_k, ar, c_step, st_step):
    NCV, NMV, ND, NQ, NDQ, NU = dims
    n = ar.shape[0]
    (mvm, mvf, sl, aa, ou_a, ou_s, sin_amp, sin_w, sin_ph, dact,
     jload, uref, kap, warp, od_a, od_s, od_l, bias, nlag, lam, vc,
     tint, integ, dvr, dvsel, rowm, a_s, cn_a, cn_s, nw, q_step, an_T,
     an_ph, spike, sband, bad_p, prc, band, trk, rst_f, zon) = c_step
    (u0, ua0, xv0, ou0, buf0, sr0, si0, odr0, od0, l1_0, l2_0, v0, mx, fail0,
     ys0, cn0, held0, pend0, age0, sbad, last0, y0, tgt0,
     thist0, rhist0, alo0, ahi0, lo0, hi0) = st_step

    seg = (seg_k <= k_t).sum(1)-1
    P1 = P_flat[ar, seg]
    w_r = ((k_t+1-seg_k[ar, seg])/ramp[ar, seg].clamp_min(1.)).clamp(0., 1.)[:, None]
    P0 = P_flat[ar, (seg-1).clamp_min(0)]
    Pf = P0+w_r*(P1-P0)
    P = {q: Pf[:, a0:b0].reshape(n, *sh) for q, a0, b0, sh in lay_items}
    Tf = T_flat[ar, seg]
    t = {q: Tf[:, a0:b0].reshape(n, *sh) for q, a0, b0, sh in tlay_items}

    zq, zw, zc, zo = z[:, :NDQ], z[:, NDQ:NDQ+NCV], z[:, NDQ+NCV:NDQ+2*NCV], z[:, NDQ+2*NCV:]
    u1, u2, u3, u4 = uu[:, :NCV], uu[:, NCV:2*NCV], uu[:, 2*NCV:3*NCV], uu[:, 3*NCV:]

    uc = u0+a*sl*mvm
    ut = torch.minimum(torch.maximum(uc, t['ulo']), t['uhi'])
    u = u0+torch.minimum(torch.maximum(ut-u0, -sl), sl)*mvm
    if through:
        u = uc+(u-uc).detach()
    ua = aa*ua0+(1-aa)*u
    dl = ua-xv0
    xv = torch.where(dl.abs() > P['fs'], ua-torch.sign(dl)*P['fd'], xv0).clamp(0., 1.)
    if through:
        xv = ua+(xv-ua).detach()
    ue = xv.clamp_min(1e-6)**P['gam']*mvf

    ou = ou_a*ou0+ou_s*zq
    d = (P['dlev']+ou+sin_amp*torch.sin(sin_w*(k_t+1)+sin_ph)).clamp(-1.5, 1.5)*dact
    w = torch.cat((ue, d), 1)
    buf = torch.cat((w[:, None], buf0[:, :-1]), 1)

    load = torch.gather(torch.cat((xv, (d[:, :ND]+1)/2), 1), 1, jload[:, None])[:, 0].clamp(0., 1.)
    w_phi = ((load+.25)/(uref+.25))**kap
    phi = torch.where(warp > 0, w_phi, torch.ones_like(w_phi)).clamp(.4, 2.5)
    e = phi[:, None, None]
    dd = (P['dead']/e).round().clamp(0, DMAX).long()
    ud = torch.gather(buf.transpose(1, 2)[:, None].expand(n, NCV, NU, DMAX+1), 3, dd[..., None])
    m0, m1 = torch.exp(P['p1r']*e), torch.exp(P['p2r']*e)
    lr = torch.stack((m0*torch.cos(P['p1i']*e), m1*torch.cos(P['p2i']*e)), -1)
    li = torch.stack((m0*torch.sin(P['p1i']*e), m1*torch.sin(P['p2i']*e)), -1)
    sr = lr*sr0-li*si0+(1-lr)*ud
    si = lr*si0+li*sr0-li*ud
    cr = torch.stack((P['c0r'], P['c1r']), -1); ci = torch.stack((P['c0i'], P['c1i']), -1)
    mode = (cr*sr-ci*si).sum(-1)

    odr = od_a*odr0+od_s*zo
    od = od_l*od0+(1-od_l)*odr
    km = P['K']*mode
    fast = km.sum(-1)+bias+od
    al = torch.exp(-phi[:, None]/P['tlag'])
    bl = 1-al
    l1 = al*l1_0+bl*fast
    l2 = al*l2_0+bl*l1
    fl = torch.where(nlag == 0, fast, torch.where(nlag == 1, l1, l2))
    vr = v0+(fl+lam*(v0-vc))/tint
    vs = vr.clamp(*LEV_STOP)
    if stops:
        vs = vr+(vs-vr).detach()
    v = torch.where(integ > 0, vs, fl)
    bad_x = ~torch.isfinite(v).all(-1) | (v.abs() > 50).any(-1) | ~torch.isfinite(sr).flatten(1).all(-1) | ~torch.isfinite(mx).all(-1)
    mx = mx.nan_to_num(0., 0., 0.)
    fail = fail0 | bad_x
    v = v.clamp(-60., 60.).nan_to_num(0., 0., 0.)
    sr = sr.nan_to_num(0., 0., 0.)
    si = si.nan_to_num(0., 0., 0.)
    l1 = l1.nan_to_num(0., 0., 0.)
    l2 = l2.nan_to_num(0., 0., 0.)
    fl = fl.nan_to_num(0., 0., 0.)

    yt_h = PR.h(v, P['wa'])
    yt = torch.where(dvr > 0, torch.gather(d[:, :ND], 1, dvsel), yt_h)*rowm

    f = yt.dtype
    ys = a_s*ys0+(1-a_s)*yt
    cn = cn_a*cn0+cn_s*zc
    ym = ys+nw*zw+cn
    ym = torch.where(q_step > 0, ym+(torch.round(ym/q_step.clamp_min(1e-9))*q_step-ym).detach(), ym)
    an = an_T > 0
    tick = an & (torch.remainder(k_t+1+an_ph, an_T.clamp_min(1.)) == 0)
    held = torch.where(tick, pend0, held0)
    pend = torch.where(tick, ym, pend0)
    age = torch.where(tick, an_T, age0+an.to(f))
    ym = torch.where(an, held, ym)
    ym = torch.where(u1 < spike, ym+sband*(3+5*u2)*torch.sign(u3-.5), ym)
    bad = torch.where((sbad <= 0) & (u4 < bad_p), 5+torch.floor(26*u3), (sbad-1).clamp_min(0.))
    isbad = bad > 0
    last = torch.where(isbad, last0, ym)
    y = torch.where(isbad, last, ym)

    K_abs = P['K'][:, :, :NMV].abs()
    can = ((mvm > 0) & PR.in_service(t['ulo'], t['uhi']))[:, None, :] & (K_abs > 0) & (K_abs >= .05*K_abs.amax(-1, keepdim=True))
    d_ref = (P['dead'][:, :, :NMV]/phi[:, None, None]).round().clamp(0, DMAX)
    d_ref = torch.where(can, d_ref, torch.full_like(d_ref, DMAX)).amin(-1)
    d_ref = torch.where(prc > 0, d_ref, torch.zeros_like(d_ref)).long()[:, None, :]
    ref = torch.gather(thist0, 1, d_ref)[:, 0]
    alo = torch.gather(alo0, 1, d_ref)[:, 0]
    ahi = torch.gather(ahi0, 1, d_ref)[:, 0]
    zlo = torch.gather(lo0, 1, d_ref)[:, 0]
    zhi = torch.gather(hi0, 1, d_ref)[:, 0]
    rref = torch.gather(rhist0, 1, d_ref)[:, 0]

    err = (yt-ref)/band*trk
    rst = rst_f > 0
    erc = torch.where(rst, (yt-rref)/band, err)
    viol = ((zlo-yt).clamp_min(0)+(yt-zhi).clamp_min(0))/band*zon
    mv = (u-u0)/sl*mvm
    track, zone, move = (t['cvw']*erc**2).sum(-1), 10*(viol**2).sum(-1), .001*(mv**2).sum(-1)

    k_next = k_t+1
    seg_n = (seg_k <= k_next).sum(1)-1
    Tn_f = T_flat[ar, seg_n]
    tn = {q: Tn_f[:, a0:b0].reshape(n, *sh) for q, a0, b0, sh in tlay_items}
    target = tn['tgt']
    held_n = (k_next < hold_k[ar, seg_n])[:, None]
    nlo = torch.where(held_n, tn['blo'], tn['alo'])
    nhi = torch.where(held_n, tn['bhi'], tn['ahi'])

    thist = torch.cat((target[:, None], thist0[:, :-1]), 1)
    rhist = torch.cat((tn['rtgt'][:, None], rhist0[:, :-1]), 1)
    alo_n = torch.cat((nlo[:, None], alo0[:, :-1]), 1)
    ahi_n = torch.cat((nhi[:, None], ahi0[:, :-1]), 1)
    lo_n = torch.cat((tn['lo'][:, None], lo0[:, :-1]), 1)
    hi_n = torch.cat((tn['hi'][:, None], hi0[:, :-1]), 1)

    soft = (t['cvw']*huber(erc, HUBER+torch.where(rst, torch.zeros_like(alo), torch.maximum(-alo, ahi)))).sum(-1)+10*huber(viol).sum(-1)+move
    outside = (((err < alo-1) | (err > ahi+1)) & (trk > 0)) | ((viol > ahi) & (zon > 0))
    trip = torch.zeros(n, dtype=torch.bool, device=u.device)

    new_vals = (u, u0, ua, xv, buf, sr, si, l1, l2, v, ou, d, odr, od,
                ys, cn, held, pend, age, bad, last, y, y0, yt, fl,
                target, tgt0, thist, rhist, alo_n, ahi_n, lo_n, hi_n, fail, mx)
    out_vals = (track+zone+move, soft, track, zone, move, err, erc, viol, ref, fail,
                alo, ahi, zlo, zhi, outside, trip)
    return new_vals, out_vals


def enable_sim_compile(enabled=True, dynamic=False, cpu=False, backend='inductor'):
    """Enable/disable `torch.compile` fusion for pure-tensor (`--mech 0`, `Fr=None`, no fault) `observe()` and `step()`.
    Safe on CPU/MPS: compiled path activates only on CUDA unless `cpu=True` is explicitly requested (e.g. unit tests)."""
    global _SIM_COMPILE, _SIM_COMPILE_CPU, _SIM_COMPILE_KW, _OBS_COMPILED, _STEP_COMPILED
    _SIM_COMPILE = bool(enabled)
    _SIM_COMPILE_CPU = bool(cpu)
    _SIM_COMPILE_KW = dict(dynamic=dynamic, backend=backend)
    if _SIM_COMPILE:
        torch._dynamo.config.cache_size_limit = max(getattr(torch._dynamo.config, 'cache_size_limit', 8), 64)
        _OBS_COMPILED = torch.compile(_observe_pure, dynamic=dynamic, backend=backend)
        _STEP_COMPILED = torch.compile(_step_pure, dynamic=dynamic, backend=backend)
    else:
        _OBS_COMPILED = None
        _STEP_COMPILED = None
    return _SIM_COMPILE


def is_sim_compiled():
    """Return True when pure-tensor simulator compilation (`enable_sim_compile`) is active."""
    return _SIM_COMPILE


def step_ctx(prog, k):
    """Compute segment parameters and task arrays `(P, t)` once at decision `k` for reuse across observe/priv_features/step."""
    k_cl = min(int(k), HORIZON-1)
    seg = prog.seg_at(k_cl)
    return prog.params(seg, k_cl), prog.task(seg)


def observe(prog, st, k, *, t=None):
    """The packet at decision k (aio.interface layout: 234 numbers at the default size, prog.size at any)."""
    if _SIM_COMPILE and _OBS_COMPILED is not None and t is None and (prog.P.device.type == 'cuda' or _SIM_COMPILE_CPU):
        c = prog.c
        tlay_obs = tuple(prog.tlay[q] for q in _OBS_T_KEYS)
        c_obs = tuple(c[q] for q in _OBS_C_KEYS)
        st_obs = tuple(st[q] for q in _OBS_ST_KEYS)
        return _OBS_COMPILED(prog._k_tensor(k), prog.seg_k, prog.ar, prog.T, tlay_obs, bool(getattr(prog, 'blind', False)), c_obs, st_obs)
    c = prog.c
    if t is None:
        t = prog.task(prog.seg_at(k))
    s, tgt, y = t['scale'], st['target'], st['y']
    bad = st['bad'] > 0
    zero, one = torch.zeros_like(y), torch.ones_like(y)
    rowm = c['rowm']*(1-c['dvr']) if getattr(prog, 'blind', False) else c['rowm']    # blind: the rows that read a measured disturbance are not there
    cv = torch.stack((torch.where(bad, zero, (y-tgt)/s), torch.where(bad, zero, (y-st['prev_y'])/s),
                      (t['plo']-tgt)/s, (t['phi']-tgt)/s, 1-c['dvr'], t['cvw'], c['trk'], torch.where(bad, zero, one),
                      torch.where(c['an_T'] > 0, torch.log1p(st['age']), zero), (torch.log1p(c['dt'])/10)[:, None].expand_as(y),
                      (tgt-st['last_target'])/s, one), -1)*rowm[..., None]
    u, m = st['u'], c['mvm']
    span = (t['uhi']-t['ulo']).clamp_min(1e-9)                    # the MV limits in force: position, move and slew are given inside them (an MV that is not the controller's: no room, and not in the packet)
    mv = torch.stack((2*(u-t['ulo'])/span-1, 2*(u-st['prev_u'])/span, 2*torch.minimum(c['slew'], span)/span, -torch.ones_like(u), c['psign'],
                      torch.ones_like(u)), -1)*m[..., None]
    links = c['link']*rowm[:, :, None]
    return torch.cat((cv.flatten(1), mv.flatten(1), links.flatten(1)), -1)


# ---------------------------------------------------------------- step
def _mech(prog, st, buf, seg, k):
    """The mechanistic episodes over one decision: the engine's states advanced with the inputs the shell
    gives them (valve flows and load levels in physical units, each after its transport delay; the
    equipment's parameters as they are during decision k). Returns the row values (n, NCV), 0 on rows the
    engine does not feed, and the new states."""
    NCV, NMV, ND, NQ, NDQ, NU = prog.dims
    n, mx = prog.n, st['mx']
    vm = torch.zeros(n, NCV+1, dtype=buf.dtype, device=buf.device)
    trip = torch.zeros(n, dtype=torch.bool, device=buf.device)
    for g in prog.mg:
        i, ar, m = g['idx'], g['ar'], g['m']
        hist = torch.cat((buf[i], buf.new_zeros(i.numel(), DMAX+1, 1)), 2)
        win = g['wbase']+g['wscale']*hist[ar[:, None], g['din'], g['chan']]
        sg = seg[i]
        w = ((k+1-prog.seg_k[i, sg])/prog.ramp[i, sg].clamp_min(1.)).clamp(0., 1.)[:, None]
        P0, P1 = g['P'][ar, (sg-1).clamp_min(0)], g['P'][ar, sg]
        pb = P0+w*(P1-P0)
        x = m.advance(mx[i, :m.nx], win, pb)
        yb = m.y(x, win, pb)
        if m.trips:
            with torch.no_grad():
                trip = trip.index_put((i,), m.tripped(yb))
        vb = (yb-g['rlo'])/g['rspan']
        vm = vm.index_put((i[:, None].expand_as(g['orow']), g['orow']), vb)
        mx = mx.index_copy(0, i, torch.nn.functional.pad(x, (0, mx.shape[1]-m.nx)))
    return vm[:, :NCV], mx, trip


def _phi(xv, d, c):
    ND = d.shape[1]//2
    load = torch.gather(torch.cat((xv, (d[:, :ND]+1)/2), 1), 1, c['jload'][:, None])[:, 0].clamp(0., 1.)
    w = ((load+.25)/(c['uref']+.25))**c['kap']
    return torch.where(c['warp'] > 0, w, torch.ones_like(w)).clamp(.4, 2.5)


def _reference(st, P, phi, c, t):
    """What every process row is charged against at this decision: its target, scoring allowance and zone
    limits as they were one dead time ago (the shortest from any MV in service that moves the row by a
    twentieth or more of what its strongest MV does, at the current time scale): what a controller can have
    reached by now (task rigor rule 5)."""
    NMV = c['mvm'].shape[1]
    K = P['K'][:, :, :NMV].abs()
    can = ((c['mvm'] > 0) & PR.in_service(t['ulo'], t['uhi']))[:, None, :] & (K > 0) & (K >= .05*K.amax(-1, keepdim=True))
    d = (P['dead'][:, :, :NMV]/phi[:, None, None]).round().clamp(0, DMAX)
    d = torch.where(can, d, torch.full_like(d, DMAX)).amin(-1)
    d = torch.where(c['prc'] > 0, d, torch.zeros_like(d)).long()[:, None, :]
    return (torch.gather(st[q], 1, d)[:, 0] for q in ('thist', 'alo', 'ahi', 'lo', 'hi', 'rhist'))


def step(prog, st, a, k, rnd=None, through=True, stops=False, *, P=None, t=None):
    """One decision: moves a (n, NMV) in [-1, 1] of the slew limit (plant frame). Returns the next state and
    `out`: cost (n,), its parts, `soft` (the same cost with Huber tails, for gradients), the per-row errors
    in bands, and whether a row is outside its band or limit.
    stops: straight-through gradients at a level's physical stops too (aio.train --stops; the forward values are the same).
    through: straight-through gradients at the MV limits and through valve friction (the forward values are
    exact; without it a move into a limit or inside the friction band has zero gradient)."""
    if (_SIM_COMPILE and _STEP_COMPILED is not None and not prog.mg and prog.Fr is None
            and getattr(prog, 'fault', None) is None and P is None and t is None
            and (prog.P.device.type == 'cuda' or _SIM_COMPILE_CPU)):
        z, uu = prog.noise() if rnd is None else rnd
        c = prog.c
        c_step = tuple(c.get('mvf', c['mvm']) if q == 'mvf' else c[q] for q in _STEP_C_KEYS)
        st_step = tuple(st[q] for q in _STEP_ST_KEYS)
        lay_items = tuple((q, a0, b0, sh) for q, (a0, b0, sh) in prog.lay.items())
        tlay_items = tuple((q, a0, b0, sh) for q, (a0, b0, sh) in prog.tlay.items())
        new_vals, out_vals = _STEP_COMPILED(
            prog._k_tensor(k), a, z, uu, bool(through), bool(stops), prog.dims,
            lay_items, tlay_items, prog.P, prog.T, prog.seg_k, prog.ramp, prog.hold_k, prog.ar, c_step, st_step,
        )
        return dict(zip(_NEW_ST_KEYS, new_vals)), dict(zip(_OUT_KEYS, out_vals))
    NCV, NMV, ND, NQ, NDQ, NU = prog.dims
    c, n = prog.c, prog.n
    if P is None or t is None:
        seg = prog.seg_at(k)
        if P is None:
            P = prog.params(seg, k)
        if t is None:
            t = prog.task(seg)
    z, uu = prog.noise() if rnd is None else rnd
    zq, zw, zc, zo = z[:, :NDQ], z[:, NDQ:NDQ+NCV], z[:, NDQ+NCV:NDQ+2*NCV], z[:, NDQ+2*NCV:]
    u1, u2, u3, u4 = uu[:, :NCV], uu[:, NCV:2*NCV], uu[:, 2*NCV:3*NCV], uu[:, 3*NCV:]
    mvm = c['mvm']
    # ---- MVs: slew and limits, positioner lag, friction, flow characteristic
    u0, sl = st['u'], c['slew']
    uc = u0+a*sl*mvm
    ut = torch.minimum(torch.maximum(uc, t['ulo']), t['uhi'])     # inside the limits in force; from outside them, back at the slew rate
    u = u0+torch.minimum(torch.maximum(ut-u0, -sl), sl)*mvm
    if through:
        u = uc+(u-uc).detach()
    ua = c['aa']*st['ua']+(1-c['aa'])*u
    dl = ua-st['xv']
    xv = torch.where(dl.abs() > P['fs'], ua-torch.sign(dl)*P['fd'], st['xv']).clamp(0., 1.)
    if through:
        xv = ua+(xv-ua).detach()
    fault = getattr(prog, 'fault', None)                          # faults in training (aio.sim.faults.train_draw): a valve stuck, an instrument gone, frozen or drifting
    if fault is not None:
        fon = (fault['has'] & (k >= fault['k0']) & (k < fault['k0']+fault['dur']))[:, None]
        xv = torch.where(fon & (fault['kind'] == 3)[:, None] & fault['mv'], st['xv'], xv)          # stick: the valve stays whatever it is told
    ue = _flow(xv, P['gam'], c.get('mvf', c['mvm']))
    # ---- disturbances keep moving around their scheduled levels
    ou = c['ou_a']*st['ou']+c['ou_s']*zq
    d = (P['dlev']+ou+c['sin_amp']*torch.sin(c['sin_w']*(k+1)+c['sin_ph'])).clamp(-1.5, 1.5)*c['dact']
    w = torch.cat((ue, d), 1)
    buf = torch.cat((w[:, None], st['buf'][:, :-1]), 1)
    # ---- pair dynamics at the current time scale
    phi = _phi(xv, d, c)
    e = phi[:, None, None]
    dd = (P['dead']/e).round().clamp(0, DMAX).long()
    ud = torch.gather(buf.transpose(1, 2)[:, None].expand(n, NCV, NU, DMAX+1), 3, dd[..., None])
    m0, m1 = torch.exp(P['p1r']*e), torch.exp(P['p2r']*e)
    lr = torch.stack((m0*torch.cos(P['p1i']*e), m1*torch.cos(P['p2i']*e)), -1)
    li = torch.stack((m0*torch.sin(P['p1i']*e), m1*torch.sin(P['p2i']*e)), -1)
    sr = lr*st['sr']-li*st['si']+(1-lr)*ud
    si = lr*st['si']+li*st['sr']-li*ud
    cr = torch.stack((P['c0r'], P['c1r']), -1); ci = torch.stack((P['c0i'], P['c1i']), -1)
    mode = (cr*sr-ci*si).sum(-1)
    # ---- rows: drift, lags, integrating or unstable, output curve
    odr = c['od_a']*st['odr']+c['od_s']*zo
    od = c['od_l']*st['od']+(1-c['od_l'])*odr
    km = P['K']*mode
    fast = km.sum(-1)+c['bias']+od
    al = torch.exp(-phi[:, None]/P['tlag'])
    bl = 1-al
    l1 = al*st['l1']+bl*fast
    l2 = al*st['l2']+bl*l1
    fl = torch.where(c['nlag'] == 0, fast, torch.where(c['nlag'] == 1, l1, l2))
    v0 = st['v']
    vr = v0+(fl+c['lam']*(v0-c['vc']))/c['tint']
    vs = vr.clamp(*LEV_STOP)
    if stops:                                                     # (a level held at its stop answers no valve: no gradient says which way brings it back)
        vs = vr+(vs-vr).detach()
    v = torch.where(c['integ'] > 0, vs, fl)
    mx, trip = st['mx'], None
    if prog.mg:                                                   # rows of the engine: the value, and for the teacher a level's rate
        vm, mx, trip = _mech(prog, st, buf, seg, k)
        mr = c['mrow'] > 0
        v = torch.where(mr, vm, v)
        fl = torch.where(mr, torch.where(c['integ'] > 0, (vm-v0)*c['tint'], vm), fl)
    bad_x = ~torch.isfinite(v).all(-1) | (v.abs() > 50).any(-1) | ~torch.isfinite(sr).flatten(1).all(-1) | ~torch.isfinite(mx).all(-1)
    mx = mx.nan_to_num(0., 0., 0.)
    fail = st['fail'] | bad_x
    v, sr, si, l1, l2, fl = (x.nan_to_num(0., 0., 0.) for x in (v.clamp(-60., 60.), sr, si, l1, l2, fl))
    yt = _ytrue(v, P['wa'], d, c)
    # ---- instruments
    f = yt.dtype
    ys = c['as']*st['ys']+(1-c['as'])*yt
    cn = c['cn_a']*st['cn']+c['cn_s']*zc
    ym = ys+c['nw']*zw+cn
    q = c['quant']
    ym = torch.where(q > 0, ym+(torch.round(ym/q.clamp_min(1e-9))*q-ym).detach(), ym)
    T = c['an_T']
    an = T > 0
    tick = an & (torch.remainder(k+1+c['an_ph'], T.clamp_min(1.)) == 0)
    held = torch.where(tick, st['pend'], st['held'])              # a sample is reported one cycle after it is taken
    pend = torch.where(tick, ym, st['pend'])
    age = torch.where(tick, T, st['age']+an.to(f))
    ym = torch.where(an, held, ym)
    ym = torch.where(u1 < c['spike'], ym+c['sband']*(3+5*u2)*torch.sign(u3-.5), ym)
    sbad = st['bad']
    if fault is not None:                                         # bad: the instrument goes bad-quality for the fault's length (the packet says so)
        sbad = torch.where((fault['has'] & (fault['kind'] == 0) & (k == fault['k0']))[:, None] & fault['row'], fault['dur'][:, None].to(sbad.dtype), sbad)
    bad = torch.where((sbad <= 0) & (u4 < c['bad']), 5+torch.floor(26*u3), (sbad-1).clamp_min(0.))
    isbad = bad > 0
    last = torch.where(isbad, st['last'], ym)
    y = torch.where(isbad, last, ym)
    if fault is not None:                                         # freeze: the reading stays where it was, no flag; drift: it walks off, no flag
        fz = fon & fault['row'] & (fault['kind'] == 1)[:, None]
        dr = fon & fault['row'] & (fault['kind'] == 2)[:, None]
        y = torch.where(fz, st['y'], torch.where(dr, y+(fault['off']*(k+1-fault['k0']).to(y.dtype)/fault['dur'].to(y.dtype))[:, None], y))
        last = torch.where(fz | dr, y, last)
    # ---- cost on the true values against the reachable reference
    ref, alo, ahi, zlo, zhi, rref = _reference(st, P, phi, c, t)
    band = c['bandf']
    err = (yt-ref)/band*c['trk']
    rst = c['rst'] > 0
    erc = torch.where(rst, (yt-rref)/band, err)                   # a resting row is charged against the reachable resting value
    viol = ((zlo-yt).clamp_min(0)+(yt-zhi).clamp_min(0))/band*c['zon']
    mv = (u-u0)/c['slew']*mvm
    track, zone, move = (t['cvw']*erc**2).sum(-1), 10*(viol**2).sum(-1), .001*(mv**2).sum(-1)
    # ---- what is published for decision k+1
    tn = prog.task(prog.seg_at(k+1))
    target = tn['tgt']
    nlo, nhi = prog.allowance(k+1)
    new = dict(u=u, prev_u=u0, ua=ua, xv=xv, buf=buf, sr=sr, si=si, l1=l1, l2=l2, v=v, ou=ou, d=d, odr=odr, od=od,
               ys=ys, cn=cn, held=held, pend=pend, age=age, bad=bad, last=last, y=y, prev_y=st['y'], yt=yt, fl=fl,
               target=target, last_target=st['target'], thist=torch.cat((target[:, None], st['thist'][:, :-1]), 1),
               rhist=torch.cat((tn['rtgt'][:, None], st['rhist'][:, :-1]), 1),
               alo=torch.cat((nlo[:, None], st['alo'][:, :-1]), 1), ahi=torch.cat((nhi[:, None], st['ahi'][:, :-1]), 1),
               lo=torch.cat((tn['lo'][:, None], st['lo'][:, :-1]), 1), hi=torch.cat((tn['hi'][:, None], st['hi'][:, :-1]), 1), fail=fail, mx=mx)
    if prog.Fr is not None:                                       # the teacher's prediction follows what was applied
        with torch.no_grad():
            Fr = prog.Fr[prog.ar, prog.teacher_seg(k)]
            qm = _flow(u, P['gam'], c.get('mvf', mvm))            # (a change of a valve's characteristic moves its flow with no move made)
            dw = torch.cat((qm-st['qm'], (d-st['d'])*prog.ffm), 1)
            new['qm'] = qm
            lag = torch.arange(MD.LAGS+1, device=u.device, dtype=f)[None]+phi[:, None]
            new['tF'] = MD.lerp_at(st['tF']+torch.einsum('bnij,bj->bni', Fr, dw), lag)
            fk = torch.stack((km[:, :, :NMV+ND].sum(-1), km[:, :, NMV+ND:].sum(-1)), -1)
            new['k1'] = al[..., None]*st['k1']+bl[..., None]*fk
            new['k2'] = al[..., None]*st['k2']+bl[..., None]*new['k1']
            who = prog.renewals().get(k) if k < HORIZON else _renewed(prog, k)
            if who is not None:                                   # what the inputs have yet to do, by the model as it now is
                new['tF'] = new['tF'].index_copy(0, who, _renew(prog, new, fk, prog.teacher_seg(k+1), k, phi, P, who))
            if prog.meter is not None:                            # and what the instruments say of it (the teacher that sees them alone)
                new.update(_meter(prog, st, new, P['wa'], tick, an, isbad))
    soft = (t['cvw']*huber(erc, HUBER+torch.where(rst, torch.zeros_like(alo), torch.maximum(-alo, ahi)))).sum(-1)+10*huber(viol).sum(-1)+move
    out = dict(cost=track+zone+move, soft=soft, track=track, zone=zone, move=move, err=err, erc=erc, viol=viol, ref=ref, fail=fail,
               alo=alo, ahi=ahi, zlo=zlo, zhi=zhi,
               outside=(((err < alo-1) | (err > ahi+1)) & (c['trk'] > 0)) | ((viol > ahi) & (c['zon'] > 0)),
               trip=torch.zeros(n, dtype=torch.bool, device=u.device) if trip is None else trip)
    return new, out


# ---------------------------------------------------------------- the teacher's prediction when its model is renewed
REBASE = 30                      # decisions between two renewals of the prediction of a plant whose clock moves with its load: shifted by its rate decision after decision, it blurs
RENEW_FIELDS = ('K', 'p1r', 'p1i', 'p2r', 'p2i', 'c0r', 'c0i', 'c1r', 'c1i', 'dead', 'tlag')     # the plant parameters `_renew` reads
RENEW_WORK = 1 << 23             # numbers in a work array of `_renew` (the episodes renewed at one decision are taken a few at a time)


def _renewed(prog, k):
    """The episodes whose prediction `step` works out again after decision k (`Program.renewals`, one decision: past
    the window's HORIZON), or None."""
    c = prog.c
    sg1 = prog.teacher_seg(k+1)
    ramp = prog.ramp[prog.ar, sg1]
    due = (ramp > 0) & (prog.seg_k[prog.ar, sg1]+ramp.ceil().long() == k+1)
    if REBASE:
        due = due | ((c['warp'] > 0) & ((k+1) % REBASE == 0))
    due = due & (prog.plant[prog.ar, sg1] == prog.plant[prog.ar, prog.seg_at(k+1)])
    due = (due | (sg1 != prog.teacher_seg(k))) & prog.exact[prog.ar, sg1] & (c['mech'] == 0)
    who = due.nonzero().flatten()
    return who if who.numel() else None


@torch.no_grad()
def _renew(prog, st, fk, sg, k, phi, P, who):
    """The prediction (m, LAGS+1, nr) of the episodes `who`, whose teacher has segment sg's model from decision k+1 on.
    `step` asks for it when that model has just been renewed, again when a gradual change it has in full has arrived
    (until then the time constants on their way were taken at their end), and every REBASE decisions on a plant whose
    clock moves with its load (the prediction is shifted by the clock's rate at each decision, and blurs).

    A sum of step responses is the prediction of one model: after a renewal, what the moves made so far have yet to do
    would stay as the old model had it, and what a changed gain is still sending through the row lags would not be in
    it at all. So it is worked out again from the plant's own state `st` (after decision k), in closed form: the delay
    lines run out, the modes settle on what arrives, the row lags follow, all by the new model at time scale 1 (a
    gain on its way there gradually: along its ramp; time constants: the final ones). Only the inputs the teacher
    knows of count (the valve flows, the measured disturbances, the loads if it sees them: st['k1'], st['k2'] hold
    what the row lags have of them), each staying where it is but for the travel the valves have left to their
    commands (a valve held by friction counted as getting there, as the prediction has it between renewals), which is a
    step made now. The value now stays the prediction's own: the offset to the plant (or the estimate of it from the
    instruments) is untouched. fk: what those inputs send into the row lags at decision k; P: the plant's parameters
    then."""
    NCV, NMV, ND, NQ, NDQ, NU = prog.dims
    c, nr, L, f = prog.c, prog.nr, MD.LAGS, st['tF'].dtype
    out = []
    per = nr*max(2*NU*DMAX*DMAX, L*L)
    step_ = max(1, RENEW_WORK//per)
    for a in range(0, who.numel(), step_):
        i = who[a:a+step_]
        m, s1 = i.numel(), sg[i]
        Pn, Pp = _view(prog.P[i, s1], prog.lay, RENEW_FIELDS), _view(prog.P[i, (s1-1).clamp_min(0)], prog.lay, ('K',))
        r = lambda x: x[:, :nr]                                   # noqa: E731
        e = phi[i]
        grp = torch.tensor([1., float(prog.ffm[-1]) if NQ else 0.], dtype=f, device=e.device)
        known = torch.cat((torch.ones(NMV+ND, dtype=f, device=e.device), grp[1].expand(NQ)))
        buf, w0 = st['buf'][i], st['buf'][i, 0]
        # ---- the delay lines: what arrives at each pair at the model steps 1..D from now
        dd = (r(Pn['dead'])/e[:, None, None]).round().clamp(0, DMAX)
        D = min(L, int(torch.ceil((dd*e[:, None, None]).max()).item())+1)
        js = torch.arange(1, L+1, dtype=f, device=e.device)
        at = (dd[..., None]-js[:D]/e[:, None, None, None]).round().clamp_min(0).long()
        arr = torch.gather(buf.transpose(1, 2)[:, None].expand(m, nr, NU, DMAX+1), 3, at)-w0[:, None, :, None]
        # ---- the modes: from where they are, towards what arrives
        pr, pi = torch.stack((r(Pn['p1r']), r(Pn['p2r'])), -1), torch.stack((r(Pn['p1i']), r(Pn['p2i'])), -1)
        kk = torch.arange(L+1, dtype=f, device=e.device)
        pw = torch.polar(torch.exp(pr[..., None]*kk), pi[..., None]*kk)             # (m, nr, NU, 2, L+1) the poles to every power
        lam = pw[..., 1]
        z0 = torch.complex(st['sr'][i, :nr], st['si'][i, :nr])-w0[:, None, :, None]
        ji = torch.arange(D, device=e.device)
        lower = (ji[:, None] >= ji[None, :])
        conv = (pw[..., (ji[:, None]-ji[None, :]).clamp_min(0)]*lower*arr[:, :, :, None, None, :]).sum(-1)      # (m, nr, NU, 2, D)
        tail = pw[..., 1:L-D+1]*conv[..., -1:]
        z = pw[..., 1:]*z0[..., None]+(1-lam)[..., None]*torch.cat((conv, tail), -1)
        cr, ci = torch.stack((r(Pn['c0r']), r(Pn['c1r'])), -1), torch.stack((r(Pn['c0i']), r(Pn['c1i'])), -1)
        mode = (cr[..., None]*z.real-ci[..., None]*z.imag).sum(3)+(cr.sum(-1)*w0[:, None, :])[..., None]      # (m, nr, NU, L)
        # ---- into the row lags: the gains as they are at every step (a gradual change is still on its way)
        Fp, Fn = (torch.einsum('mrc,mrcl->mrl', r(x['K'])*known, mode) for x in (Pp, Pn))
        wr = ((k+1+js[None]/e[:, None]-prog.seg_k[i, s1][:, None])/prog.ramp[i, s1].clamp_min(1.)[:, None]).clamp(0., 1.)
        fin = Fp+wr[:, None, :]*(Fn-Fp)
        la = -1/r(Pn['tlag'])
        jl = torch.arange(L, device=e.device)
        G = torch.exp(la[..., None, None]*(jl[:, None]-jl[None, :]).clamp_min(0))*(jl[:, None] >= jl[None, :])*(1-torch.exp(la))[..., None, None]
        dec = torch.exp(la[..., None]*js)
        k1, k2, f0 = ((x[i, :nr]*grp).sum(-1) for x in (st['k1'], st['k2'], fk))
        l1 = dec*k1[..., None]+torch.einsum('mrts,mrs->mrt', G, fin)
        l2 = dec*k2[..., None]+torch.einsum('mrts,mrs->mrt', G, l1)
        nl = r(c['nlag'][i])
        now = torch.where(nl == 0, f0, torch.where(nl == 1, k1, k2))
        fut = torch.where(nl[..., None] == 0, fin, torch.where(nl[..., None] == 1, l1, l2))
        # ---- the travel the valves have left to their commands
        xv, mvf = st['xv'][i], c.get('mvf', c['mvm'])[i]
        dq = st['qm'][i]-_flow(xv, P['gam'][i], mvf)
        tF = torch.cat((torch.zeros_like(now)[:, None], (fut-now[..., None]).transpose(1, 2)), 1)
        out.append(st['tF'][i, :1]+tF+torch.einsum('mlrc,mc->mlr', prog.Fr[i, s1][..., :NMV], dq))
    return torch.cat(out, 0)


# ---------------------------------------------------------------- the teacher's prediction from the instruments alone
@torch.no_grad()
def _meter_gains(prog, m):
    """The gains of `_meter` for the settings m (METER's fields), per row (n, nr): steady-state Kalman filters with an
    integrating disturbance, the standard of offset-free MPC. A row that is not a level: its offset from the model is
    a random walk (q_rest bands a decision) read through the instrument's noise; gain f. A level: its height and the
    net inflow the model does not explain, the inflow a random walk (q_lev bands a decision, each decision); gains L1
    (height) and L2 (inflow). The instrument's noise is its white noise and quantisation, `floor` bands at the least;
    an analyser's reading comes every T decisions. gate: a reading further than this many bands from what was
    expected is believed when the next one confirms it (a spike is 3 to 8; not an analyser's, whose next sample is
    3 to 20 decisions off). lag: the sensor's own lag is in the model."""
    c, nr = prog.c, prog.nr
    r = lambda x: x[:, :nr].double()                              # noqa: E731
    band = r(c['sband']).clamp_min(1e-12)
    R = (r(c['nw'])**2+r(c['quant'])**2/12)/band**2+m['floor']**2
    T = r(c['an_T']).clamp_min(1.)
    rho = T*m['q_rest']**2/R
    f = 2*rho/(rho+torch.sqrt(rho*rho+4*rho))
    ql = _lev_walk(prog, nr).clamp_min(m['q_min']) if m['lev_model'] else m['q_lev']
    q = ql**2*T**3                                                # (between two samples the inflow walks T steps and the height sums it)
    p11, p12, p22 = R.clone(), torch.zeros_like(R), q.clone()
    for _ in range(4000):
        a11, a12, a22 = p11+2*p12+p22, p12+p22, p22+q
        k1, k2 = a11/(a11+R), a12/(a11+R)
        p11, p12, p22 = (1-k1)*a11, (1-k1)*a12, a22-k2*a12
    a11, a12 = p11+2*p12+p22, p12+p22
    L1, L2 = a11/(a11+R), a12/(a11+R)/T                           # (L2: per decision)
    ig = c['integ'][:, :nr] > 0
    f32 = lambda x: x.to(prog.dtype)                             # noqa: E731
    return dict(use=torch.where(ig, torch.full_like(ig, bool(m['levels'])), torch.full_like(ig, bool(m['rest']))) & (c['prc'][:, :nr] > 0), f=f32(f), L1=f32(L1), L2=f32(L2),
                gate=torch.where((c['an_T'][:, :nr] > 0) | (m['gate'] <= 0), float('inf'), f32(band*m['gate'])),      # (an analyser's next sample is too far off to wait for)
                lag=c['as'][:, :nr]*float(bool(m['lag'])))


@torch.no_grad()
def _lev_walk(prog, nr):
    """(n, nr) how fast a level's net inflow the model does not explain walks, by the plant's own model (bands a
    decision, each decision): the wandering loads it does not feed forward through the row's gains, the sinusoid's
    rate on its load, the row's own drift; over the level's time constant and band. The meter's q_lev when lev_model."""
    c, nmv = prog.c, prog.nmv
    K = prog.params(torch.zeros(prog.n, dtype=torch.long, device=prog.device))['K'][:, :nr, nmv:].double()
    blind = (1-prog.ffm.double())[None, None]                     # (the loads the prediction does not see move)
    wan = ((K*c['ou_s'][:, None].double()*blind)**2).sum(-1)
    sin = ((K*(c['sin_amp']*c['sin_w'])[:, None].double()*blind)**2).sum(-1)/2
    return torch.sqrt(wan+sin+c['od_s'][:, :nr].double()**2)/(c['tint'][:, :nr].double()*c['sband'][:, :nr].double()).clamp_min(1e-12)


@torch.no_grad()
def meter_start(prog, st, wa):
    """The estimator's state where it has none yet: the plant taken to be at rest at what its instruments show now
    (every offset is the reading less the model, a level stands still). wa: the output curves in force."""
    c, nr = prog.c, prog.nr
    z = PR.hinv(st['y'][:, :nr], wa[:, :nr])[0]
    ig = c['integ'][:, :nr] > 0
    m = st['tF'][:, 0]
    es = torch.where(ig, z, m)
    return dict(eo=torch.where(ig, -c['lam'][:, :nr]*(z-c['vc'][:, :nr])-m, z-m), ev=torch.where(ig, z, torch.zeros_like(z)),
                es=es, eh=es, ep=es, eg=torch.zeros_like(z))


@torch.no_grad()
def _meter(prog, st, new, wa, tick, an, isbad):
    """The estimator over one decision (prog.meter): `new`'s readings against what the model says the instruments show.
    eo: of a row that is not a level, its offset from the model (before the output curve); of a level, the net inflow
    the model does not explain. ev: a level's height. es: what the sensor shows of the model's value (its lag); eh, ep:
    that at the time an analyser took the sample it reports now, and the one it reports next; eg: the last new
    reading was set aside as a spike. A reading that is not new (an analyser between two samples, a bad value held) changes
    nothing: the model runs on alone."""
    g, c, nr = prog.meter, prog.c, prog.nr
    e0 = {q: st[q] for q in METER_KEYS} if 'eo' in st else meter_start(prog, st, wa)
    r = lambda x: x[:, :nr]                                       # noqa: E731
    ig, tint = r(c['integ']) > 0, r(c['tint'])
    z = PR.hinv(r(new['y']), r(wa))[0]                           # the reading before the output curve
    m = new['tF'][:, 0]                                           # the model's value of the row now (a level: its net inflow)
    vr = e0['ev']+(m+e0['eo']+r(c['lam'])*(e0['ev']-r(c['vc'])))/tint      # a level a decision on, by the model and the inflow not explained so far
    vp = vr.clamp(*LEV_STOP)                                      # (held at its physical stop, as the plant's is)
    held = ig & (vr != vp)                                        # pushed against the stop: the reading says nothing of the inflow
    a = g['lag']
    es = a*e0['es']+(1-a)*torch.where(ig, vp, m)
    tk, anr = r(tick), r(an)
    eh = torch.where(tk, e0['ep'], e0['eh'])
    ep = torch.where(tk, es, e0['ep'])
    shown = torch.where(anr, eh, es)
    e = z-torch.where(ig, shown, shown+e0['eo'])
    fresh = (~anr | tk) & ~r(isbad)
    spike = fresh & (e.abs() > g['gate']) & (e0['eg'] <= 0)
    upd = (fresh & ~spike).to(z.dtype)
    dv = upd*g['L1']*e*ig
    return dict(eo=e0['eo']+upd*torch.where(ig, g['L2']*tint*~held, g['f'])*e, ev=torch.where(ig, (vp+dv).clamp(*LEV_STOP), torch.zeros_like(z)),
                es=es+dv, eh=eh+dv, ep=ep+dv, eg=torch.where(fresh, spike.to(z.dtype), e0['eg']))


@torch.no_grad()
def seen(prog, st, k):
    """`st` as the teacher's prediction takes it: as it is (the true values before the instruments), or with prog.meter
    the rows' values 'fl' and the levels' heights 'v' as estimated from the instruments (`_meter`)."""
    g = prog.meter
    if g is None:
        return st
    nr = prog.nr
    e = st if 'eo' in st else meter_start(prog, st, prog.params(prog.seg_at(k), k)['wa'])
    fl, v = st['fl'].clone(), st['v'].clone()
    fl[:, :nr] = torch.where(g['use'], st['tF'][:, 0]+e['eo'], fl[:, :nr])
    v[:, :nr] = torch.where(g['use'] & (prog.c['integ'][:, :nr] > 0), e['ev'], v[:, :nr])
    return dict(st, fl=fl, v=v)


# ---------------------------------------------------------------- what only the simulator knows
@torch.no_grad()
def free_response(prog, st, k):
    """Where the plant goes if the MVs stop now, by the teacher's model, on the grid its gains were designed
    for (decisions at the steady time scale of its segment): (n, P, NMV) values of the tracked rows before
    the output curve, (n, NZMAX, len(ZL)) of the zone rows at the lags ZL, and r (n,), how much faster the
    plant runs now than at that steady point.

    The model is the plant's linear core (aio.sim.model.response): the superposition of the step responses
    to every flow and disturbance change so far, kept in model time (the plant's own clock, which runs
    with the load), shifted to the true value now. A level is integrated from its true height, so an
    imbalance the model does not explain counts as a lasting rate, not as an offset."""
    NCV, NMV, ND, NQ, NDQ, NU = prog.dims
    c = prog.c
    sg = prog.teacher_seg(k)
    P = prog.Sw.shape[2]
    j = torch.arange(1, P+1, device=prog.device, dtype=prog.dtype)
    tF, phis = st['tF'], prog.phis[prog.ar, sg][:, None]
    off = st['fl'][:, :prog.nr]-tF[:, 0]
    f = MD.lerp_at(tF[:, :, :NMV], j[None]*phis)+off[:, None, :NMV]
    r = _phi(st['xv'], st['d'], c)/phis[:, 0]
    r3 = r[:, None, None]
    vc = c['vc'][:, None, :NMV]
    pw = torch.exp(j[None, :, None]*c['lrho'][:, None]/r3)
    lev = vc+pw*((st['v'][:, None, :NMV]-vc)+torch.cumsum(f/pw, 1)/(r3*c['tint'][:, None, :NMV]))
    zl = torch.tensor(MD.ZL, device=prog.device, dtype=prog.dtype)
    fz = torch.gather(MD.lerp_at(tF, zl[None]*phis)+off[:, None], 2, c['zrow'][:, None, :].expand(-1, len(MD.ZL), -1))
    return torch.where((c['integ'][:, None, :NMV] > 0), lev, f), fz.transpose(1, 2), r


@torch.no_grad()
def _steady_targets(prog, st, sg, vt, wy, lo, hi, vlo, vhi, rounds=4):
    """The steady-state layer of the teacher (the target calculation of an industrial MPC): where the rows
    can settle when the MV limits (lo..hi, flow from now) or the zone limits (vlo..vhi of the zone slots,
    before the output curve) keep the tracked rows from their targets vt. The same trade-off as
    prior.optimum, on the model's steady gains: the CVs weighted as the cost weighs them (a zone row while
    it is beyond its limit), resting values after all of them (a larger size, true values: weighted as the
    cost weighs them, see `teacher_view`), every level in balance first, the MV limits exactly (every active set is tried). It starts from the model's own prediction of where the plant settles, so it follows the loads
    as they move. The zone limits are taken ZONE_BACK inside (the back-off of an industrial controller: the
    score counts a zone row beyond its limit by any amount, and a row held on the limit is beyond it half
    the time). Returns the tracked targets and the zone limits for the dynamic layer: the limits backed off,
    and where something is in the way the values of the best steady point (so that the dynamic layer, whose
    own trade-off depends on how fast each row answers, has nothing left to trade)."""
    NCV, NMV, ND, NQ, NDQ, NU = prog.dims
    c = prog.c
    m = c['mvm'].double()
    tF = st['tF']
    fall = (tF[:, -1]+st['fl'][:, :prog.nr]-tF[:, 0]).double()                 # where the rows settle with the MVs left as they are
    fss, fz = fall[:, :NMV], torch.gather(fall, 1, c['zrow'])
    G, Gz = prog.Gs[prog.ar, sg].double(), prog.Gz[prog.ar, sg].double()
    ig = c['integ'][:, :NMV] > 0
    big = prog.dims != PR.sizes_of(*PR.DIMS)
    w, wz = wy.double()*torch.where(c['rst'][:, :NMV] > 0, 1. if _rest_as_cost(prog) else math.sqrt(PR.REST_EPS), 1.), prog.wz[prog.ar, sg].double()   # resting values last (or as the cost)
    wl = torch.where(ig, w*1e4/c['tint'][:, :NMV], w)                          # a level drifting a band in 100 decisions counts as 100 bands off
    want = torch.where(ig, -c['lam'][:, :NMV]*(vt-c['vc'][:, :NMV]), vt).double()
    e, J = wl*(want-fss), wl[:, :, None]*G
    e0, e = e, _reachable(J, e, c['rst'][:, :NMV] > 0)
    H = J.transpose(1, 2)@J
    ridge = torch.diag_embed(1-m)+1e-12*H.diagonal(dim1=1, dim2=2).amax(-1)[:, None, None]*torch.eye(NMV, dtype=H.dtype, device=H.device)
    H = H+ridge
    g = (J.transpose(1, 2)@e[..., None])[..., 0]
    lo, hi, zlo, zhi = lo.double(), hi.double(), vlo.double(), vhi.double()
    zlo0, zhi0 = zlo, zhi
    back = torch.minimum(ZONE_BACK*math.sqrt(10.)/wz.clamp_min(1e-30), .25*(zhi-zlo).clamp_min(0.))*(wz > 0)      # wz: sqrt(10) per band
    zlo, zhi = torch.where(zlo > -1e8, zlo+back, zlo), torch.where(zhi < 1e8, zhi-back, zhi)
    vlo, vhi = zlo.to(vlo.dtype), zhi.to(vhi.dtype)
    x = torch.linalg.solve(H, g)
    zs = fz+(Gz@x[..., None])[..., 0]
    out = ((x < lo) | (x > hi)).any(-1) | (((zs < zlo) | (zs > zhi)) & c['zslot']).any(-1)
    if 'thin' in c:                                                            # more setpoints than MVs: the targets cannot all be held, whatever the limits
        out = out | (c['thin'] & (((J@x[..., None])[..., 0]-e0).abs().amax(-1) > 1e-6))
    if not bool(out.any()):
        return vt, vlo, vhi
    i = out.nonzero().flatten()
    Hi, gi, loi, hii, Gzi, fzi, wzi = H[i], g[i], lo[i], hi[i], Gz[i], fz[i], wz[i]
    zl, zh, slot = zlo[i], zhi[i], c['zslot'][i]
    xi = torch.minimum(torch.maximum(x[i], loi), hii)
    Ei = eE = Ji = ei = None
    if big:                                                                    # a larger size: the levels are rows to hold, not weights (as prior.optimum has them).
        wq = torch.where(ig, torch.zeros_like(w), w)[i]                         # Weighted 1e4/tint among resting values weighted 1e-4, the solve kept a few digits of
        Ji, ei = wq[:, :, None]*G[i], wq*(want-fss)[i]                          # what the resting values do: on a plant with more setpoints than MVs they went bands astray
        ei = _reachable(Ji, ei, (c['rst'][:, :NMV] > 0)[i])
        Hi = Ji.transpose(1, 2)@Ji
        Hi = Hi+torch.diag_embed(1-m[i])+1e-12*Hi.diagonal(dim1=1, dim2=2).amax(-1)[:, None, None]*torch.eye(NMV, dtype=Hi.dtype, device=Hi.device)
        gi = (Ji.transpose(1, 2)@ei[..., None])[..., 0]
        rate = torch.where(ig, w*100/c['tint'][:, :NMV], torch.zeros_like(w))[i]   # a level's drift in bands per 100 decisions
        Ei, eE = rate[:, :, None]*G[i], rate*(want-fss)[i]
        zl, zh = zlo0[i], zhi0[i]                                               # and the trade-off is made at the zone limits themselves, as the score's reachable
                                                                                # point is: kept inside them by the back-off, the resting values (weighted 1e-4) went
                                                                                # bands astray and setpoints gave; the back-off stays where the point leaves room for it
    mu = None

    def worth(x):                                                              # the steady cost itself (convex): levels off balance, tracked rows, zone rows by how far beyond
        r = (Ji@x[..., None])[..., 0]-ei
        zs = fzi+(Gzi@x[..., None])[..., 0]
        over = ((zs-zh).clamp_min(0)+(zl-zs).clamp_min(0))*slot
        return ((((Ei@x[..., None])[..., 0]-eE)**2).sum(-1)/PR.EQ_SOFT+(r**2).sum(-1)+((wzi*over)**2).sum(-1))
    for _ in range(rounds if Ei is None else ROUNDS_BIG):                      # zone rows count while they are beyond a limit
        zs = fzi+(Gzi@xi[..., None])[..., 0]
        lim = torch.where(zs > zh, zh, zl)
        act = (((zs > zh) | (zs < zl)) & slot).double()
        Jz = (wzi*act)[:, :, None]*Gzi
        Hz = Hi+Jz.transpose(1, 2)@Jz
        gz = gi+(Jz.transpose(1, 2)@(wzi*act*(lim-fzi))[..., None])[..., 0]
        if Ei is None:
            xb, found = PR._box_qp(Hz, -gz, loi, hii, xi)
            xi = torch.where(found[:, None], xb, xi)
            continue
        xb, found, mu = PR._box_qp(Hz, -gz, loi, hii, xi, Ei, eE, mu)
        # (a larger size) a new set of zone rows can be worse than the last: with a dozen zone rows the set went round in
        # circles and the plan stopped where three of them were bands beyond. The step goes as far along as the cost
        # itself falls (the solve's direction is downhill on it: the same gradient here, a convex model), until it no longer does.
        d = torch.where(found[:, None], xb-xi, torch.zeros_like(xi))
        f0 = worth(xi)
        a = _line(Ji, ei, Ei, eE, Gzi, fzi, wzi, zl, zh, slot, xi, d)
        xn = xi+a[:, None]*d
        go = worth(xn) < f0-1e-12*f0.abs()
        xi = torch.where(go[:, None], xn, xi)
        if not bool(go.any()):
            break
    x = x.index_copy(0, i, xi)
    reach = fss+(G@x[..., None])[..., 0]
    zs = fz+(Gz@x[..., None])[..., 0]
    o1 = out[:, None] & prog.trade[prog.ar, sg][:, None]
    vt = torch.where(o1 & ~ig & (w > 0), reach, vt.double()).to(vt.dtype)
    vlo = torch.where(o1 & c['zslot'], torch.minimum(zlo, zs), zlo).to(vlo.dtype)
    vhi = torch.where(o1 & c['zslot'], torch.maximum(zhi, zs), zhi).to(vhi.dtype)
    return vt, vlo, vhi


def _reachable(J, e, rst):
    """The errors e (m, rows; weighted, bands) the steady-state layer acts on, with the part of the process rows' errors
    that the MVs can hardly reach taken down: along a direction in which the rows J (m, rows, NMV; bands per span of
    flow) answer s bands to a span of valve travel, s^2/(s^2+WEAK) of it is kept. On a plant whose setpoints are nearly
    collinear (two or three bands a span in some direction) the prediction's own error of a fraction of a band, taken
    at its word, asked for half a span of travel at every decision, ran a valve into its limit and had the spare MV
    give its resting value up for it. What is left out is asked for again at the next decision from where the valves
    then are, so the layer still ends where it did. The resting rows' errors stay as they are."""
    if WEAK <= 0:
        return e
    Jp = J*(~rst)[:, :, None]
    Hp = Jp.transpose(1, 2)@Jp+WEAK*torch.eye(J.shape[2], dtype=J.dtype, device=J.device)
    x = torch.linalg.solve(Hp, (Jp.transpose(1, 2)@(e*~rst)[..., None]))
    return torch.where(rst, e, (Jp@x)[..., 0])


def _line(J, e, E, eE, Gz, fz, wz, zl, zh, slot, x, d):
    """(m,) the step along d (0..1) that minimises the steady cost of `_steady_targets` (a larger size) from x: the cost is
    convex and piecewise quadratic along the line (a zone row's term starts where it crosses a limit), so its slope is
    piecewise linear and only grows. The slope at 0, at 1 and at every crossing in between, all at once; the root lies on
    the piece where it changes sign, by the line through that piece's ends: exact, in a few dozen operations (sixty
    bisections of the same slope were two thirds of the teacher's operations at a larger size). A few fixed steps to
    try (1, 1/2 ... 1/32) stalled where a zone row that was inside crossed its limit within the first thirty-second
    of the way."""
    r0, r1 = (J@x[..., None])[..., 0]-e, (J@d[..., None])[..., 0]
    l0, l1 = (E@x[..., None])[..., 0]-eE, (E@d[..., None])[..., 0]
    z0, z1 = fz+(Gz@x[..., None])[..., 0], (Gz@d[..., None])[..., 0]
    w2 = wz**2*slot
    s0, s1 = (r0*r1).sum(-1)+(l0*l1).sum(-1)/PR.EQ_SOFT, (r1*r1).sum(-1)+(l1*l1).sum(-1)/PR.EQ_SOFT
    moves = z1 != 0
    den = torch.where(moves, z1, torch.ones_like(z1))
    cross = torch.cat(((zh-z0)/den, (zl-z0)/den), 1)*moves.repeat(1, 2)       # where each zone row meets a limit
    t = torch.cat((torch.zeros_like(s0)[:, None], cross.clamp(0., 1.), torch.ones_like(s0)[:, None]), 1).sort(1).values
    z = z0[:, None]+t[:, :, None]*z1[:, None]
    over = (z-zh[:, None]).clamp_min(0)-(zl[:, None]-z).clamp_min(0)
    s = s0[:, None]+t*s1[:, None]+(over*(w2*z1)[:, None]).sum(-1)             # (m, points) the slope there
    up = s >= 0
    j = up.int().argmax(1, keepdim=True).clamp_min(1)                          # the first point where the cost no longer falls
    ta, tb, sa, sb = (torch.gather(v, 1, i)[:, 0] for v in (t, s) for i in (j-1, j))
    a = ta+(tb-ta)*(sa/(sa-sb).clamp_max(-1e-300)).clamp(0., 1.)
    return torch.where(s[:, -1] <= 0, torch.ones_like(a), torch.where(s[:, 0] >= 0, torch.zeros_like(a), a))


def _rest_as_cost(prog):
    """Whether the teacher's steady layer weighs the resting values as the cost does, at the reachable resting value
    (`teacher_view`): at a larger size, on true values."""
    return prog.dims != PR.sizes_of(*PR.DIMS) and prog.meter is None


@torch.no_grad()
def teacher_view(prog, st, k):
    """What the teacher's plan at decision k starts from (aio.law reads the same): sg, the segment its model is
    of; t, the task; E (n, P, NMV), the weighted errors of the tracked rows if the MVs stop now, against where
    the steady-state layer says they can settle; fz, r of `free_response`; q, the valve flows, and qlo, qhi, the
    MV limits in flow from now; vlo, vhi, the zone limits before the output curve; g, u, sl, wy.
    A larger size, true values (`_rest_as_cost`): a resting row is aimed at the reachable resting value the
    cost charges it against (rhist) and weighed in the steady layer as the cost weighs it. Ranked after every
    zone limit, the spare valves held the zone rows exactly on their limits while the unmeasured loads wandered
    and went bands from a resting value the score does not move with the loads (usual48 plant 2; docs/sim.md,
    section 15). From the instruments alone the teacher cannot tell the loads' design level from their wander
    (that value is not in the packet either): it keeps the published values, ranked last."""
    NCV, NMV, ND, NQ, NDQ, NU = prog.dims
    c, n = prog.c, prog.n
    sg, seg = prog.teacher_seg(k), prog.seg_at(k)
    P, t = prog.params(seg, k), prog.task(seg)
    st = seen(prog, st, k)
    free, fz, r = free_response(prog, st, k)
    g, u, sl = P['gam'], st['u'], c['slew']
    q = _flow(u, g, c['mvm'])
    qlo, qhi = _flow(t['ulo'], g, c['mvm'])-q, _flow(t['uhi'], g, c['mvm'])-q
    wy = prog.wy[prog.ar, sg]
    tg = st['target'][:, :NMV]
    if _rest_as_cost(prog):
        tg = torch.where(c['rst'][:, :NMV] > 0, st['rhist'][:, 0, :NMV], tg)
    vt, _ = PR.hinv(tg, P['wa'][:, :NMV])
    # zone limits before the output curve
    zr = c['zrow']
    wa, zlo, zhi = (torch.gather(x, 1, zr) for x in (P['wa'], t['lo'], t['hi']))
    big = torch.full_like(zlo, 1e9)
    vlo = torch.where(c['zslot'] & (zlo > -1e5), PR.hinv(zlo.clamp(-1., 2.), wa)[0], -big)
    vhi = torch.where(c['zslot'] & (zhi < 1e5), PR.hinv(zhi.clamp(-1., 2.), wa)[0], big)
    vt, vlo, vhi = _steady_targets(prog, st, sg, vt, wy, qlo, qhi, vlo, vhi)
    E = (vt[:, None]-free)*wy[:, None]
    return dict(sg=sg, t=t, E=E, fz=fz, r=r, q=q, qlo=qlo, qhi=qhi, vlo=vlo, vhi=vhi, g=g, u=u, sl=sl, wy=wy)


@torch.no_grad()
def teacher(prog, st, k, plan=False, view=None):
    """(n, NMV) move of the teacher in [-1, 1] of the slew limit, plant frame: step-response DMC
    on the plant's exact model, with the MV limits and the zone limits
    (weighted as the cost weighs them). It works on the linear core and handles the rest exactly: errors
    are taken before the output curve, moves are computed in valve flow and turned into positions through
    the valve curve, levels are integrated from their true height, and the load-dependent time scale is
    followed. Its prediction is shifted at every decision to the plant's true values before the instruments
    (or, Program.set_teacher(meter=), to what the instruments show: `seen`). It feeds forward the measured
    disturbances and (training teacher) the unmeasured loads. Its
    model follows a process event late (prog.eff_k), which a learner that has to identify the change from
    its own history can match. Where the MV limits or the zone limits keep the targets out of reach it steers
    to the best steady point its model finds inside them, and it keeps a zone row a little inside its limit
    (`_steady_targets`).
    plan: instead of the move, the whole plan: (n, M, NMV) the positions it means the MVs to have at the
    decisions model.MT from now (the first of them is where its move heads). view: `teacher_view` of this
    decision where the caller has it already."""
    NCV, NMV, ND, NQ, NDQ, NU = prog.dims
    c, n = prog.c, prog.n
    v = teacher_view(prog, st, k) if view is None else view
    sg, t, E, fz, r, q, vlo, vhi, g, u, sl = (v[x] for x in ('sg', 't', 'E', 'fz', 'r', 'q', 'vlo', 'vhi', 'g', 'u', 'sl'))
    # every planned position is inside the MV limits as the plant treats them and no further from the position
    # now than the slew limit lets the valve travel by then (a plan that counts on a jump later, to use an
    # inverse response against a zone limit, would start the wrong way)
    j = (torch.tensor(MD.MT, device=u.device, dtype=u.dtype)+1)[None, :, None]
    a, b = u[:, None]-j*sl[:, None], u[:, None]+j*sl[:, None]
    g1, m1 = g[:, None], c['mvm'][:, None]
    lo = _flow(torch.maximum(a, torch.minimum(t['ulo'][:, None], b)).clamp(0., 1.), g1, m1)-q[:, None]
    hi = _flow(torch.minimum(b, torch.maximum(t['uhi'][:, None], a)).clamp(0., 1.), g1, m1)-q[:, None]
    # the zone limits as weighted distances from the free response. A row heading beyond a limit is asked back
    # along a funnel: at every lag by the share of its steady answer to the MVs that can be there by then
    # (nothing inside its dead time or the wrong-way stretch of an inverse response), so the plan gains nothing
    # by buying a short relief at one lag with moves it has to take back
    w = prog.wz[prog.ar, sg][:, :, None]*torch.tensor(MD.ZDT, device=prog.device, dtype=prog.dtype).sqrt()
    dl, dh, zs = vlo[:, :, None]-fz, vhi[:, :, None]-fz, prog.zs[prog.ar, sg]
    zl, zh = (torch.where(dl > 0, dl*zs, dl)*w).reshape(n, -1), (torch.where(dh < 0, dh*zs, dh)*w).reshape(n, -1)
    # a plant running slower than the model it was designed on gets smaller moves: the errors are scaled, not
    # the result, so a move at the slew limit stays there
    rc = r.clamp_max(1.)
    dq = MD.dmc_move(prog.Sw[prog.ar, sg], prog.HX[prog.ar, sg], E*rc[:, None, None], lo, hi, prog.Az[prog.ar, sg], zl*rc[:, None], zh*rc[:, None],
                     full=plan)
    if plan:
        un = (q[:, None]+dq.view(n, -1, NMV)).clamp(0., 1.)**(1/g[:, None])
        return torch.where(c['mvm'][:, None] > 0, un, u[:, None])
    un = (q+dq).clamp(0., 1.)**(1/g)
    return ((un-u)/sl).clamp(-1., 1.)*c['mvm']


@torch.no_grad()
def privileged(prog, st, k):
    """What a critic may know beside the packet: the true local model (labels), the teacher's free response
    of the tracked rows at the label lags (where the plant is heading if the MVs stop), the true values,
    the disturbance levels and the true row states, all in packet units."""
    NCV, NMV, ND, NQ, NDQ, NU = prog.dims
    c = prog.c
    lab = prog.labels(k).flatten(1)
    band = c['bandf']
    lags = torch.tensor(MD.LAB_LAGS, device=prog.device)-1
    wa = prog.params(prog.seg_at(k), k)['wa'][:, None, :NMV]
    free = (PR.h(free_response(prog, st, k)[0][:, lags], wa)-st['target'][:, None, :NMV])/band[:, None, :NMV]*c['trk'][:, None, :NMV]
    t = prog.task(prog.seg_at(k))
    err = (st['yt']-st['target'])/t['scale']*c['rowm']
    return torch.cat((lab, torch.asinh(free.flatten(1)), torch.asinh(err), st['d'], st['od']/band*c['prc'],
                      (st['v']-.5)*c['prc'], st['xv']*c['mvm']), 1)


def priv_fields(ncv=IF.NCV, nmv=IF.NMV, nd=PR.ND):
    """The parts of `privileged`, in order, with the shape of each per episode: the labels (model.labels), the
    free response of the tracked rows at the label lags, the true errors, the disturbance levels, the rows'
    offsets, their states and the valve flows."""
    return dict(lab=(nmv, ncv*MD.NLAB), free=(MD.NLAB, nmv), err=(ncv,), d=(2*nd,), od=(ncv,), v=(ncv,), xv=(nmv,))


PRIV = sum(math.prod(v) for v in priv_fields().values())         # at the default size; a Program's own: prog.priv

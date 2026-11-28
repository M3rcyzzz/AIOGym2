"""The mechanism engine: one batched, differentiable simulator for every plant (docs/sim.md, 'Engine').

A plant is a `Plant`: states, inputs (MVs and loads), parameters, right-hand sides and outputs written as
expressions (aio.sim.expr). `Mech` packs the tables of the plants of a batch (one per episode, of any
structure, padded to a common shape) and gives, for the whole batch at once:

  f, fj, y      right-hand sides, with the Jacobian df/dx, and the outputs
  advance       one decision: ROS2 (Verwer et al. 1999), a linearly implicit two-stage method of order two
                that is L-stable, so stiff plants need no small steps, and of order two for any Jacobian,
                so one Jacobian and one factorisation serve the whole decision. A mass vector marks
                algebraic equations (0 = f). The backward pass treats the Jacobian as a constant.
                Every substep carries its own error estimate (the first-order solution inside the method);
                an episode whose estimate is too large, in bands of the outputs, takes that decision again
                in more substeps with a fresh Jacobian at each (a runaway, a tray running dry).
  steady        a steady state by damped Newton with the exact Jacobian, alone or together with the inputs
                that hold chosen outputs at targets, continued from the nominal point
  linearise     A, B, C, D at a point; `responses`: the step responses of the outputs over decisions, of the
                scheme as it is run (what the labels, the PI tuning and the linear references are made of)

The engine has no code of any plant: four tanks, a pH tank, a distillation column or Tennessee Eastman are
tables.
"""
import math
import numpy as np
import torch

from aio.sim.expr import Graph, E, CLS, TINY, compile_table

GAMMA = 1+1/math.sqrt(2)
INF = float('inf')
BIG = 96                         # a batched factorisation of matrices above this order runs on one thread (`one`)
ERR_TOL, REFINE, AHEAD, RICH_OWN = 1., 4, 8, 2.    # the accuracy guard of `advance` (see there and `atol`)
GUARD = 2                        # its version (in the key of a batch with plants of the engine, aio.sim.stay.key): 2 since 11-13, a refined decision judged by Richardson's estimate too, episode by episode
STATS = None                     # set to a list to collect, per call of `advance`, the error estimates (n,) of the plain decision


class one:
    """Some torch builds (2.15 on Linux with MKL) return wrong pivots or hang in a threaded batched LU of matrices
    of order above about 128. One thread is right everywhere, and these batches are small."""

    def __init__(self, A):
        self.on = A.device.type == 'cpu' and A.dim() > 2 and A.shape[-1] > BIG and A.shape[0] > 1

    def __enter__(self):
        if self.on:
            self.n = torch.get_num_threads()
            torch.set_num_threads(1)

    def __exit__(self, *a):
        if self.on:
            torch.set_num_threads(self.n)


class Plant:
    """One plant as data. Time in `time_unit` seconds; a decision lasts `dt` seconds, integrated in `substeps`."""

    def __init__(self, name, dt, time_unit=1., substeps=1, refresh=False):
        self.name, self.dt, self.time_unit, self.substeps, self.refresh = name, float(dt), float(time_unit), int(substeps), bool(refresh)
        self.g = Graph()
        self.xs, self.ws, self.ps, self.ys = [], [], [], []
        self.rhs = {}
        self._tab = {}

    # ---- declarations
    def state(self, name, x0, lo=-INF, hi=INF, scale=None, algebraic=False, maxstep=INF):
        """A state, its nominal value (the Newton start), physical bounds and typical size. algebraic: its
        equation is 0 = f (scale: the size of that residual; maxstep: the most one Newton step may move it)."""
        self.xs.append(dict(name=name, x0=float(x0), lo=float(lo), hi=float(hi), scale=float(scale if scale is not None else max(abs(x0), 1e-3)),
                            mass=0. if algebraic else 1., maxstep=float(maxstep)))
        return E(self.g, self.g.leaf('x', len(self.xs)-1))

    def mv(self, name, lo, hi, slew, nom=None):
        """A manipulated input in physical units: range, largest move per decision, nominal value."""
        self.ws.append(dict(name=name, kind='mv', lo=float(lo), hi=float(hi), slew=float(slew), nom=float((lo+hi)/2 if nom is None else nom)))
        return E(self.g, self.g.leaf('w', len(self.ws)-1))

    def load(self, name, nom, lo, hi, measured=False):
        """A load: an input that is not manipulated (feed rate, composition, ambient): nominal value and range."""
        self.ws.append(dict(name=name, kind='load', lo=float(lo), hi=float(hi), nom=float(nom), measured=bool(measured)))
        return E(self.g, self.g.leaf('w', len(self.ws)-1))

    def param(self, name, nom, lo=None, hi=None, event=True):
        """A parameter of the equipment (fouling, activity, a valve coefficient): nominal value and range.
        event=False: it differs from one plant to the next and holds for an episode (how far the real equipment
        is from its model: nothing that changes while the unit runs)."""
        self.ps.append(dict(name=name, nom=float(nom), lo=float(nom if lo is None else lo), hi=float(nom if hi is None else hi), event=bool(event)))
        return E(self.g, self.g.leaf('p', len(self.ps)-1))

    def const(self, v):
        return E(self.g, self.g.const(v))

    def der(self, x, expr):
        """dx/dt = expr (for an algebraic state: 0 = expr)."""
        nd = self.g.nodes[x.i]
        assert nd[0] == 'in' and nd[1] == 'x' and nd[2] not in self.rhs
        self.rhs[nd[2]] = expr.i if isinstance(expr, E) else self.g.const(expr)

    def out(self, name, expr, **meta):
        """An output row: a measured or computed quantity (meta: range, band, limits, how it is used)."""
        self.ys.append(dict(name=name, node=expr.i if isinstance(expr, E) else self.g.const(expr), **meta))

    # ---- sizes and tables
    @property
    def nx(self):
        return len(self.xs)

    @property
    def nw(self):
        return len(self.ws)

    @property
    def np(self):
        return len(self.ps)

    @property
    def ny(self):
        return len(self.ys)

    def table(self, kind):
        """'f': right-hand sides; 'fj': with df/dx; 'y': outputs; 'lin': f, y and every first derivative."""
        if kind not in self._tab:
            g = self.g
            assert sorted(self.rhs) == list(range(self.nx)), f'{self.name}: every state needs a right-hand side'
            f, y = [self.rhs[i] for i in range(self.nx)], [o['node'] for o in self.ys]
            outs, idx = {}, {}
            if kind in ('f', 'fj', 'lin'):
                outs['f'] = f
            if kind in ('y', 'lin'):
                outs['y'] = y
            if kind in ('fj', 'lin'):
                xl = [g.leaf('x', i) for i in range(self.nx)]
                wl = [g.leaf('w', i) for i in range(self.nw)] if kind == 'lin' else []
                D = g.diff(f+(y if kind == 'lin' else []), xl+wl)
                for q, rows, cols in (('fx', f, xl), ('fw', f, wl), ('yx', y, xl), ('yw', y, wl)):
                    if q != 'fx' and kind != 'lin':
                        continue
                    ent = [(a, b, D[r][c]) for a, r in enumerate(rows) for b, c in enumerate(cols) if c in D[r]]
                    outs[q] = [e[2] for e in ent]
                    idx[q] = np.array([(a, b) for a, b, _ in ent], dtype=np.int64).reshape(-1, 2)
            t = compile_table(g, outs)
            t.idx = idx
            self._tab[kind] = t
        return self._tab[kind]

    def nominal(self):
        """x0, w nominal, p nominal as float64 rows."""
        t = lambda v: torch.tensor([v], dtype=torch.float64)      # noqa: E731
        return t([s['x0'] for s in self.xs]), t([w['nom'] for w in self.ws]), t([p['nom'] for p in self.ps])


_FN = dict(exp=lambda v: torch.exp(v.clamp(-80., 80.)), log=lambda v: torch.log(v.clamp_min(TINY)), sqrt=lambda v: torch.sqrt(v.clamp_min(0.)),
           abs=torch.abs, sign=torch.sign, relu=torch.relu, step=lambda v: (v > 0).to(v.dtype), tanh=torch.tanh)


class Packed:
    """The tables of P plants in one layout: per level and class, operand registers (P, K, ...)."""

    def __init__(self, tabs, nx, nw, npar, dims, device, dtype):
        L = max(t.count.shape[0] for t in tabs)
        count = np.zeros((L, len(CLS)), dtype=np.int64)
        for t in tabs:
            count[:t.count.shape[0]] = np.maximum(count[:t.count.shape[0]], t.count)
        base = 2+nx+nw+npar+np.concatenate(([0], np.cumsum(count.reshape(-1))[:-1]))
        self.nreg = int(2+nx+nw+npar+count.sum())
        lead = np.array([0, 2, 2+nx, 2+nx+nw])                    # spaces -1 (fixed), -2 x, -3 w, -4 p

        def reg(a):
            sp, ix = a[..., 0], a[..., 1]
            return np.where(sp < 0, lead[np.clip(-sp-1, 0, 3)], base[np.clip(sp, 0, None)])+ix
        P, M = len(tabs), tabs[0].M
        to = lambda a, f=False: torch.as_tensor(a, device=device, dtype=dtype if f else torch.long)   # noqa: E731
        self.levels = []
        for l in range(1, L):
            lv = []
            for c, name in enumerate(CLS):
                K = int(count[l, c])
                if not K:
                    continue
                if name == 'lin':
                    idx, coef, c0 = np.ones((P, K, M), dtype=np.int64), np.zeros((P, K, M)), np.zeros((P, K))
                    for i, t in enumerate(tabs):
                        if (l, c) in t.ops:
                            r, cf, z = t.ops[(l, c)]
                            idx[i, :len(r)], coef[i, :len(r)], c0[i, :len(r)] = reg(r), cf, z
                    lv.append((name, (to(idx.reshape(P, K*M)), to(coef, True), to(c0, True))))
                else:
                    na = 1 if name in _FN else 2
                    idx = np.zeros((P, K, na), dtype=np.int64)        # padding: f(1), 0*0 and 0/1
                    if name == 'mul':
                        idx[:] = 1
                    elif name == 'div':
                        idx[:, :, 0] = 1
                    for i, t in enumerate(tabs):
                        if (l, c) in t.ops:
                            r = t.ops[(l, c)][0]
                            idx[i, :len(r)] = reg(r)
                    lv.append((name, tuple(to(idx[:, :, j]) for j in range(na))))
            self.levels.append(lv)
        self.out, self.pos = {}, {}
        for q in tabs[0].outs:
            n = max(len(t.outs[q]) for t in tabs) if q not in dims or isinstance(dims[q], tuple) else dims[q]
            idx = np.ones((P, n), dtype=np.int64)                 # padding: the zero register
            pos = np.zeros((P, n), dtype=np.int64)
            for i, t in enumerate(tabs):
                idx[i, :len(t.outs[q])] = reg(t.outs[q])
                if q in t.idx:
                    pos[i, :len(t.outs[q])] = t.idx[q][:, 0]*dims[q][1]+t.idx[q][:, 1]
            self.out[q] = to(idx)
            if q in tabs[0].idx:
                self.pos[q] = to(pos)

    def take(self, pid):
        """The tables per episode (pid: the plant of every episode); with one plant, shared by all."""
        if self.out[next(iter(self.out))].shape[0] == 1:
            return self
        s = object.__new__(Packed)
        s.nreg = self.nreg
        s.levels = [[(name, tuple(a[pid] for a in arr)) for name, arr in lv] for lv in self.levels]
        s.out, s.pos = {q: v[pid] for q, v in self.out.items()}, {q: v[pid] for q, v in self.pos.items()}
        return s


def run(pk, x, w, p):
    """All registers (n, nreg) of the packed tables at the states, inputs and parameters given."""
    n = x.shape[0]
    one = x.new_ones(n, 1)
    R = torch.cat((one, torch.zeros_like(one), x, w, p), 1)
    G = lambda i: torch.gather(R, 1, i.expand(n, -1))             # noqa: E731
    for lv in pk.levels:
        parts = []
        for name, a in lv:
            if name == 'lin':
                idx, coef, c0 = a
                parts.append((G(idx).view(n, *coef.shape[1:])*coef).sum(-1)+c0)
            elif name == 'mul':
                parts.append(G(a[0])*G(a[1]))
            elif name == 'div':
                b = G(a[1])
                parts.append(G(a[0])/torch.where(b.abs() < TINY, torch.full_like(b, TINY), b))
            else:
                parts.append(_FN[name](G(a[0])))
        R = torch.cat([R]+parts, 1)
    return R


def _out(pk, R, q):
    return torch.gather(R, 1, pk.out[q].expand(R.shape[0], -1))


def _mat(pk, R, q, rows, cols):
    v = _out(pk, R, q)
    return torch.zeros(R.shape[0], rows*cols, dtype=R.dtype, device=R.device).scatter_add(1, pk.pos[q].expand(R.shape[0], -1), v).view(-1, rows, cols)


class Mech:
    """The plants of a batch. plants: the distinct plants; pid (n,): the plant of every episode (None: one
    plant for all). pace (n,): how many of the plant's own decision intervals a decision of the episode lasts
    (1 without it); more: the substeps of a decision are this many times the plants' own (2 for a batch
    with decisions longer than the plants' own: the cost of a step is the number of substeps of the batch,
    so every episode takes them). Runs in `dtype` on `device`; the build tools (steady, linearise,
    responses) run in float64 on the CPU and return float64."""

    def __init__(self, plants, pid=None, n=None, device='cpu', dtype=torch.float32, _base=None, pace=None, more=1):
        self.plants = plants
        self._base = {} if _base is None else _base              # packed tables before they are spread over the episodes
        self.pid = torch.zeros(n, dtype=torch.long) if pid is None else torch.as_tensor(pid, dtype=torch.long)
        self.n, self.device, self.dtype = self.pid.numel(), torch.device(device), dtype
        self.nx, self.nw, self.np, self.ny = (max(getattr(q, a) for q in plants) for a in ('nx', 'nw', 'np', 'ny'))
        self.np = max(self.np, 1)
        self._pk = {}

        def tab(attr, key, pad, size):
            return torch.tensor([[r[key] for r in getattr(q, attr)]+[pad]*(size-len(getattr(q, attr))) for q in plants], dtype=torch.float64)[self.pid]
        self.x0, self.xlo, self.xhi = tab('xs', 'x0', 0., self.nx), tab('xs', 'lo', -INF, self.nx), tab('xs', 'hi', INF, self.nx)
        self.xscale, self.mass = tab('xs', 'scale', 1., self.nx), tab('xs', 'mass', 1., self.nx)
        self.xmask = (torch.arange(self.nx)[None] < torch.tensor([q.nx for q in plants])[self.pid][:, None]).double()
        self.wnom, self.wlo, self.whi = tab('ws', 'nom', 0., self.nw), tab('ws', 'lo', 0., self.nw), tab('ws', 'hi', 0., self.nw)
        self.pnom = tab('ps', 'nom', 0., self.np)
        self.pace = torch.ones(self.n, dtype=torch.float64) if pace is None else torch.as_tensor(pace, dtype=torch.float64).cpu()
        self.h = torch.tensor([q.dt/q.time_unit for q in plants], dtype=torch.float64)[self.pid]*self.pace      # one decision in model time
        self.more = int(more)
        self.substeps = max(q.substeps for q in plants)*self.more
        self.refresh = any(q.refresh for q in plants)
        to = lambda a: a.to(self.device, dtype)                   # noqa: E731
        self.zstep = torch.where(self.mass > 0, torch.full_like(self.mass, INF), tab('xs', 'maxstep', INF, self.nx))
        self.alg = bool((self.mass == 0).any())
        trip = lambda j, pad: torch.tensor([[float(o.get('trip', (-INF, INF))[j]) for o in q.ys]+[pad]*(self.ny-q.ny) for q in plants], dtype=torch.float64)[self.pid]    # noqa: E731
        self.r = dict(xlo=to(self.xlo), xhi=to(self.xhi), mass=to(self.mass), h=to(self.h)[:, None], zstep=to(self.zstep), am=to(1-self.mass),
                      ascale=to(self.xscale), trlo=trip(0, -INF).to(self.device), trhi=trip(1, INF).to(self.device))
        self.trips = any('trip' in o for q in plants for o in q.ys)

    def pack(self, kind, build=False):
        key = (kind, build)
        if key not in self._pk:
            dims = dict(f=self.nx, y=self.ny, fx=(self.nx, self.nx), fw=(self.nx, self.nw), yx=(self.ny, self.nx), yw=(self.ny, self.nw))
            dev, dt = (torch.device('cpu'), torch.float64) if build else (self.device, self.dtype)
            bk = (kind, str(dev), dt)
            if bk not in self._base:
                self._base[bk] = Packed([q.table(kind) for q in self.plants], self.nx, self.nw, self.np, dims, dev, dt)
            self._pk[key] = self._base[bk].take(self.pid.to(dev))
        return self._pk[key]

    def sub(self, idx):
        """The same plants for a part of the episodes (idx: their indices)."""
        return Mech(self.plants, self.pid[idx.cpu()], device=self.device, dtype=self.dtype, _base=self._base, pace=self.pace[idx.cpu()], more=self.more)

    # ---- evaluation
    def f(self, x, w, p):
        pk = self.pack('f')
        return _out(pk, run(pk, x, w, p), 'f')

    def fj(self, x, w, p):
        pk = self.pack('fj')
        R = run(pk, x, w, p)
        return _out(pk, R, 'f'), _mat(pk, R, 'fx', self.nx, self.nx)

    def y(self, x, w, p):
        pk = self.pack('y')
        return _out(pk, run(pk, x, w, p), 'y')

    def linearise(self, x, w, p, rows=None):
        """f, y, A = df/dx, B = df/dw, C = dy/dx, D = dy/dw (float64, CPU); rows: the episodes to do."""
        pk = self.pack('lin', True)
        if rows is not None:
            pk = _rows(pk, rows)
        R = run(pk, x.double().cpu(), w.double().cpu(), p.double().cpu())
        m = lambda q, a, b: _mat(pk, R, q, a, b)                  # noqa: E731
        return _out(pk, R, 'f'), _out(pk, R, 'y'), m('fx', self.nx, self.nx), m('fw', self.nx, self.nw), m('yx', self.ny, self.nx), m('yw', self.ny, self.nw)

    def tripped(self, y):
        """(n,) whether an output y (n, ny, the plant's own units) is past a shutdown limit the plant declares
        (`out(..., trip=(lo, hi))`: the interlock that stops the real unit, e.g. Tennessee Eastman's 3,000 kPa)."""
        return ((y.double() < self.r['trlo']) | (y.double() > self.r['trhi'])).any(-1)

    # ---- one decision
    def atol(self):
        """What an error is measured in. (n, nx): the change of a state that moves a tracked or limited output by
        one band, now or within the next AHEAD decisions (through the plant's own decision map at the nominal
        point; the smallest over the outputs); a state no such output answers to is not looked at. (n, ny): the
        band of those outputs (the others: not looked at), for the error read through the output map where
        the plant is (a log purity far from its nominal value answers many times more than there)."""
        if 'atol' not in self.r:
            if 'atol' not in self._base:
                P = len(self.plants)
                Mp = Mech(self.plants, torch.arange(P), dtype=torch.float64, _base=self._base)
                _, _, A, B, C, _ = Mp.linearise(Mp.x0, Mp.wnom, Mp.pnom)
                Phi, _ = Mp.discrete(A, B)
                S, Ck = C.abs(), C
                for _ in range(AHEAD):
                    Ck = Ck@Phi
                    S = torch.maximum(S, Ck.abs().nan_to_num(INF, INF, INF))
                task = lambda o: o.get('track') or o.get('integ') or 'lo' in o or 'hi' in o      # noqa: E731  (the outputs a task can score)
                band = torch.tensor([[float(o.get('band', INF)) if task(o) else INF for o in q.ys]+[INF]*(self.ny-q.ny) for q in self.plants], dtype=torch.float64)
                a = (band[:, :, None]/S.clamp_min(1e-300)).amin(1)
                self._base['atol'] = torch.where(Mp.mass > 0, a, torch.full_like(a, INF))
                self._base['oband'] = band
            self.r['atol'] = self._base['atol'][self.pid].to(self.device, self.dtype)
            self.r['oband'] = self._base['oband'][self.pid].to(self.device, self.dtype)
        return self.r['atol'], self.r['oband']

    def advance(self, x, w, p, substeps=None, refresh=None, guard=None):
        """The states after one decision with the inputs held. refresh: a new Jacobian at every substep (for
        plants whose Jacobian jumps inside a decision: a saturating inner loop). guard: an episode whose error
        estimate over the decision is above ERR_TOL bands takes the decision again in REFINE times the substeps
        with a new Jacobian at each, and again (up to REFINE^3 times the substeps) until it is right by both of two
        estimates: Richardson's (the scheme is of second order: the finer decision's error is its difference to the
        coarser one over REFINE^2-1, in the states by `atol` and through the output map, the larger) within
        ERR_TOL, and `_run`'s own (of first order and pessimistic) within RICH_OWN times it. The others are left as
        they are. (Until 11-13 a refined decision was judged by `_run`'s estimate alone and every episode of the
        batch was taken on while any of them was above: the exothermic CSTR spent 16 % of its decisions there for
        nothing. Richardson's alone let 149 of 2,048 valve moves of a fifth of the span through at up to 2 bands
        on the building heating.) On unless said otherwise at the plant's own substeps; a caller that names the
        substeps gets exactly those (a convergence study, a reference)."""
        S = substeps or self.substeps
        guard = substeps is None if guard is None else guard
        x1, err = self._run(x, w, p, S, self.refresh if refresh is None else refresh)
        if STATS is not None:
            STATS.append(err.detach().cpu())
        if guard:
            bad = ((err > ERR_TOL) & torch.isfinite(x).all(-1)).nonzero().flatten()     # an episode already lost is not taken again
            k, xc = REFINE, x1
            while bad.numel():
                sub = self if bad.numel() == self.n else self.sub(bad)
                xb, wb, pb = x[bad], w[bad], p[bad]
                xr, e = sub._run(xb, wb, pb, S*k, True)
                x1 = xr if bad.numel() == self.n else x1.index_copy(0, bad, xr)
                if k >= REFINE**3:
                    break
                with torch.no_grad():
                    atol, band = self.atol()
                    xo, xn = xc[bad].detach(), xr.detach()
                    d = ((xn-xo).abs()/atol[bad]).nan_to_num(INF, INF, INF).amax(-1)
                    dy = ((sub.y(xn, wb, pb)-sub.y(xo, wb, pb)).abs()/band[bad]).nan_to_num(INF, INF, INF).amax(-1)
                    keep = (torch.maximum(d, dy)/(REFINE**2-1) > ERR_TOL) | (e > RICH_OWN*ERR_TOL)
                bad, xc, k = bad[keep], x1, k*REFINE
        return x1

    def _run(self, x, w, p, S, refresh):
        """S substeps of ROS2. Returns the states and, per episode, the error estimate of the decision in bands
        (`atol`). Two errors are estimated at every substep. The method's own: the difference to the
        first-order solution of the same stages, with the stiff part taken out (Shampine's filter; without it
        every valve move on a pressure-flow network reads as an error, because the method damps the pressure
        transient in one step, which is what it should do). And the Jacobian's: a fast mode that collapses
        inside the step lands where the Jacobian in use says its balance is, and the filter hides how far that
        is from where it is; what the right-hand side did over the step beyond what that Jacobian says,
        f(x1) - f(x0) - J (x1 - x0), taken through the same solve, is that distance (the Newton correction of a
        collapsed mode, the curvature term of a slow one; zero on a linear plant). Read in the states, the
        substeps added up; and through the output map at the state reached, once for the decision. An episode
        whose matrix is singular (states far outside the plant's domain) comes back not finite, the others
        untouched."""
        r = self.r
        atol, band = self.atol()
        h = r['h']/S
        dif = r['mass'] > 0
        err = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        esum, last = torch.zeros_like(x).detach(), None

        def judge(f1):
            """The errors of the substep before, now that the right-hand side where it ended is known."""
            nonlocal err, esum
            with torch.no_grad():
                solve, J, x0, x1, f0, fm, k1, k2, dx = last
                jdx = r['mass']*dx/(GAMMA*h)-(1.5*f0+.5*fm-r['mass']*k1/h)/GAMMA      # J dx, from what the stages solved
                cut = (x1 != x0+dx).any(-1).nonzero().flatten()                   # a state stopped at its bound, an algebraic one solved again
                if cut.numel():
                    jdx = jdx.index_copy(0, cut, (J[cut]@(x1-x0)[cut][..., None])[..., 0])
                off = (f1.detach()-f0-jdx)*dif                                    # the algebraic equations are solved where the step ends
                e = solve(torch.stack((r['mass']*(k1+k2)/(2*GAMMA*h), off), -1), cols=True)
                err = err+(e.abs().sum(-1)/atol).nan_to_num(INF, INF, INF).amax(-1)
                esum = esum+e.sum(-1)

        for s in range(S):
            if s == 0 or refresh:
                f, J = self.fj(x, w, p)
                J = J.detach()
                if last is not None:
                    judge(f)
                W = torch.diag_embed(r['mass']/(GAMMA*h))-J
                with one(W):
                    LU, piv, _ = torch.linalg.lu_factor_ex(W)

                def solve(b, cols=False, LU=LU, piv=piv):
                    with one(LU):
                        return torch.linalg.lu_solve(LU, piv, b if cols else b[..., None])[..., slice(None) if cols else 0]
            else:
                f = self.f(x, w, p)
                judge(f)
            k1 = solve(f/GAMMA)
            fm = self.f(x+k1, w, p)
            k2 = solve((fm-2*r['mass']*k1/h)/GAMMA)
            dx = 1.5*k1+.5*k2
            if self.alg:
                dx = torch.minimum(torch.maximum(dx, -r['zstep']), r['zstep'])
            x1 = torch.minimum(torch.maximum(x+dx, r['xlo']), r['xhi'])
            if self.alg:
                x1 = self._algebraic(x1, w, p)
            last = (solve, J, x.detach(), x1.detach(), f.detach(), fm.detach(), k1.detach(), k2.detach(), dx.detach())
            x = x1
        with torch.no_grad():
            xd = x.detach()
            judge(self.f(xd, w, p))
            ey = (self.y(xd-esum, w, p)-self.y(xd, w, p)).abs()
            err = torch.maximum(err, (ey/band).nan_to_num(INF, INF, INF).amax(-1))
        return x, err

    def _algebraic(self, x, w, p, iters=8, tol=1e-6):
        """Damped Newton on the algebraic equations with the other states held, until every episode's residual
        (in units of the state's scale) is under tol: the full step (no further than maxstep) where it lowers
        the residual, a quarter of it elsewhere. The Jacobian is a constant in the backward pass, which gives
        the gradient of the implicit function at the root."""
        r = self.r
        am = r['am']
        clip = lambda v: torch.minimum(torch.maximum(v, r['xlo']), r['xhi'])      # noqa: E731
        res = lambda v: (v*am/r['ascale']).abs().amax(-1, keepdim=True)          # noqa: E731
        r0 = None
        for _ in range(iters):
            f, J = self.fj(x, w, p)
            r0 = res(f) if r0 is None else r0
            A = (J*am[:, :, None]*am[:, None, :]).detach()+torch.diag_embed(1-am)
            with one(A):
                dz = torch.linalg.solve(A, (-f*am)[..., None])[..., 0]
            dz = torch.minimum(torch.maximum(dz, -r['zstep']), r['zstep'])
            x1, x2 = clip(x+dz), clip(x+dz/4)
            r1 = res(self.f(x1, w, p))
            better = r1 < r0
            if not bool(better.all()):
                r1 = torch.where(better, r1, res(self.f(x2, w, p)))
            move = r0 > tol
            x = torch.where(move, torch.where(better, x1, x2), x)
            r0 = torch.where(move, r1, r0)
            if bool((r0 <= tol).all()):
                break
        return x

    # ---- build tools (float64, CPU)
    def steady(self, w, p, x0=None, free=None, rows=None, targets=None, band=None, wlo=None, whi=None, homotopy=1, iters=40, tol=1e-8,
               start=None, fmask=None, minstep=1/256):
        """A steady state of every episode: f(x, w, p) = 0. With `free` (indices of inputs) and `rows` (indices
        of outputs), those inputs are solved for as well so that the outputs sit at `targets` (errors scaled
        by `band`); the problem is square. Both may be lists (the same for every episode) or (n, m) index
        tensors with `fmask` (n, m) marking the entries in use. Continued from a known steady point (`start`:
        x, w, p; the nominal point without it): every episode walks its own way from there to its inputs,
        parameters and targets, first in `homotopy` stages, in shorter ones where a stage fails (down to
        `minstep` of the way) and longer ones where it works. Returns x, w, converged."""
        n, nx = self.n, self.nx
        f64 = lambda a: a.double().cpu()                          # noqa: E731
        w1, p1 = f64(w), f64(p)
        nf = 0 if free is None else (len(free) if isinstance(free, (list, tuple)) else free.shape[1])
        if nf:
            idx = lambda a: (torch.tensor(a, dtype=torch.long)[None].expand(n, -1) if isinstance(a, (list, tuple)) else a.long().cpu())   # noqa: E731
            free, rows = idx(free), idx(rows)
            fm = torch.ones(n, nf, dtype=torch.bool) if fmask is None else fmask.bool().cpu()
            t1, bd = f64(targets), f64(band)
            wlo = torch.gather(self.wlo, 1, free) if wlo is None else f64(wlo)
            whi = torch.gather(self.whi, 1, free) if whi is None else f64(whi)
        if start is not None:
            xs, w0, p0 = (f64(a) for a in start)
            x = xs.clone() if x0 is None else f64(x0).clone()
        else:
            w0, p0 = self.wnom, self.pnom
            x = f64(self.x0 if x0 is None else x0).clone()
            if nf:
                xs, _, okn = self.steady(w0, p0, iters=iters, tol=tol)
                x = torch.where(okn[:, None], xs, x)
        if nf:
            t0 = torch.gather(self.linearise(x, w0, p0)[1], 1, rows)
        lo = torch.cat((self.xlo, torch.where(fm, wlo, torch.full_like(wlo, -INF))), 1) if nf else self.xlo
        hi = torch.cat((self.xhi, torch.where(fm, whi, torch.full_like(whi, INF))), 1) if nf else self.xhi
        walk = homotopy > 1 or nf or start is not None
        z = torch.cat((x, torch.where(fm, torch.gather(w0, 1, free), torch.zeros_like(wlo))), 1) if nf else x
        lam = torch.zeros(n, 1, dtype=torch.float64) if walk else torch.ones(n, 1, dtype=torch.float64)
        step = torch.full((n, 1), 1./homotopy, dtype=torch.float64)
        ok = torch.zeros(n, dtype=torch.bool)
        eye = torch.eye(nf, dtype=torch.float64)[None] if nf else None

        def inputs(ws, z):                                        # the free inputs written into ws (entries not in use go to a spare column)
            if not nf:
                return ws
            return torch.cat((ws, ws[:, :1]), 1).scatter(1, torch.where(fm, free, torch.full_like(free, ws.shape[1])), z[:, nx:])[:, :ws.shape[1]]
        xsc, xm = self.xscale, self.xmask
        ms = torch.cat((self.zstep, torch.full_like(wlo, INF)), 1) if nf else self.zstep       # the furthest a state may go in one step (algebraic ones)
        zsc = torch.cat((xsc, torch.where(fm, (whi-wlo).clamp_min(1e-12), torch.ones_like(wlo))), 1) if nf else xsc       # the size of a unit of every unknown
        for _ in range(64 if walk else 1):
            live = (~ok & (step[:, 0] >= minstep)).nonzero().flatten()        # the episodes still on their way
            if live.numel() == 0:
                break
            lt = (lam+step).clamp_max(1.)
            ws, ps = w0+lt*(w1-w0), p0+lt*(p1-p0)
            ts = t0+lt*(t1-t0) if nf else None

            def res(z, sel=None):                                 # residual and Jacobian of the live episodes (sel: some of them)
                r = live if sel is None else live[sel]
                wr = ws[r]
                if nf:
                    fr, rw, fmr, bdr = free[r], rows[r], fm[r], bd[r]
                    wr = torch.cat((wr, wr[:, :1]), 1).scatter(1, torch.where(fmr, fr, torch.full_like(fr, wr.shape[1])), z[:, nx:])[:, :wr.shape[1]]
                f, y, A, B, C, D = self.linearise(z[:, :nx], wr, ps[r], rows=None if r.numel() == n else r)
                F = f/xsc[r]
                J = A/xsc[r][:, :, None]+torch.diag_embed(1-xm[r])
                if nf:
                    cols = fr[:, None, :]
                    Fy = torch.where(fmr, (torch.gather(y, 1, rw)-ts[r])/bdr, z[:, nx:])
                    Jxz = torch.gather(B, 2, cols.expand(-1, nx, -1))/xsc[r][:, :, None]*fmr[:, None, :]
                    Jyx = torch.gather(C, 1, rw[:, :, None].expand(-1, -1, nx))/bdr[:, :, None]*fmr[:, :, None]
                    Dr = torch.gather(torch.gather(D, 1, rw[:, :, None].expand(-1, -1, D.shape[2])), 2, cols.expand(-1, nf, -1))
                    Jyz = torch.where(fmr[:, :, None] & fmr[:, None, :], Dr/bdr[:, :, None], eye*(~fmr)[:, :, None])
                    F = torch.cat((F, Fy), 1)
                    J = torch.cat((torch.cat((J, Jxz), 2), torch.cat((Jyx, Jyz), 2)), 1)
                return F, J
            zn, oks = newton(res, z[live], lo[live], hi[live], iters, tol, ms[live], zsc[live])
            g = live[oks]
            z[g] = zn[oks]
            lam[g] = lt[g]
            ok[g] = lam[g, 0] >= 1.
            step[g] = 2*step[g]
            bad = live[~oks]
            step[bad] = step[bad]/2
        return z[:, :nx], inputs(w0+lam*(w1-w0) if walk else w1.clone(), z), ok

    def discrete(self, A, B, substeps=None):
        """The decision map of `advance` on the linearised plant: x+ = Phi x + Gam w (deviations)."""
        S = substeps or self.substeps
        h = (self.h/S)[:, None, None]
        I = torch.eye(self.nx, dtype=torch.float64)[None]
        M = torch.diag_embed(self.mass)
        with one(A):
            W = torch.linalg.inv(M-GAMMA*h*A)
        P, Q = W@(h*A), W@(h*B)
        T = 2*I+.5*P-W@M
        Ph, Gh = I+T@P, T@Q
        Phi, Gam = Ph, Gh
        for _ in range(S-1):
            Phi, Gam = Ph@Phi, Ph@Gam+Gh
        return Phi, Gam

    def gains(self, x, w, p, doublings=14, levels=None):
        """At a steady point: C, D, the decision map (Phi, Gam), the steady gains of the outputs to the inputs
        (n, ny, nw) and the rate at which each output keeps moving per decision under a unit step (0 unless
        the plant integrates), both of the decision map itself: summed over 2^doublings decisions. On a plant
        that integrates, an output that follows a level never settles under a step of one input. With
        `levels` (n, ny: the outputs that integrate) the gain of every other output of such a plant is taken
        with the levels where they were: the answer less the ramp it ends in, less what the displacement of
        the levels accounts for (`follow`). It is the steady gain over every move that leaves the levels at
        rest and puts them back."""
        _, _, A, B, C, D = self.linearise(x, w, p)
        Phi, Gam = self.discrete(A, B)
        Pk, X = Phi, Gam
        for _ in range(doublings):                               # X = sum_{j < N} Phi^j Gam, Pk = Phi^N
            X, Pk = X+Pk@X, Pk@Pk
        Gv, Gr = C@X+D, C@(Pk@Gam)
        if levels is not None and bool(levels.any()):
            lev = levels[:, :, None]
            off = Gv-2**doublings*Gr                              # where the ramp every output ends in starts from
            Gv = torch.where(levels.any(-1)[:, None, None] & ~lev, off-self.follow(Pk, C, levels)@(off*lev), Gv)
        return C, D, Phi, Gam, Gv, Gr

    @staticmethod
    def lasting(Phi, doublings=14):
        """Phi^(2^doublings): what of a state is still there after that many decisions. Of a stable plant
        nothing; of one that integrates, the level every state has run up (the projector on the modes that
        stay)."""
        for _ in range(doublings):
            Phi = Phi@Phi
        return Phi

    @staticmethod
    def follow(Pk, C, levels):
        """S (n, ny, ny): by how much every output ends up moved for good per unit by which each integrating
        output (`levels` (n, ny)) is displaced, on the modes that stay (Pk = `lasting`). An output y that
        follows the levels h is y - S h plus a part that settles; the columns of the other outputs are 0."""
        return (C@Pk)@torch.linalg.pinv((C*levels[:, :, None])@Pk, rtol=1e-9)

    def responses(self, x, w, p, L, substeps=None):
        """(n, L+1, ny, nw): what a unit step of every input, held from decision 0, has done to every output
        after 0..L decisions, by the plant linearised at the point given (a steady state)."""
        _, _, A, B, C, D = self.linearise(x, w, p)
        Phi, Gam = self.discrete(A, B, substeps)
        out, X = [torch.zeros_like(D)], torch.zeros_like(Gam)
        for _ in range(L):
            X = Phi@X+Gam
            out.append(C@X+D)
        return torch.stack(out, 1)


def _rows(pk, rows):
    s = object.__new__(Packed)
    s.nreg = pk.nreg
    pick = lambda a: a if a.shape[0] == 1 else a[rows]            # noqa: E731
    s.levels = [[(name, tuple(pick(a) for a in arr)) for name, arr in lv] for lv in pk.levels]
    s.out, s.pos = {q: pick(v) for q, v in pk.out.items()}, {q: pick(v) for q, v in pk.pos.items()}
    return s


STEPS = (1., .5, .2, .05)


def newton(res, z, lo, hi, iters=40, tol=1e-8, maxstep=None, scale=None):
    """Damped Newton for a batch of square systems. res(z, sel) -> residual (m, d) and Jacobian (m, d, d) of the
    rows sel (all of them without it). Every row takes its first step of 1, 1/2, 1/5, 1/20 that passes the
    natural test: the Newton correction it would take next with the same Jacobian is smaller than the one
    it has just taken (in units of `scale`), which does not depend on how the equations are scaled against
    each other. z stays inside lo..hi and no entry moves further than `maxstep` at once. Only the rows
    still moving are evaluated: a row is left alone once it has converged or no step passes. Returns z and
    which rows converged (largest residual below 1e3 tol at the end, so that a row within rounding of its
    solution counts)."""
    z = torch.minimum(torch.maximum(z.clone(), lo), hi)
    F, J = res(z)
    nrm = F.abs().amax(-1).nan_to_num(INF, INF, INF)
    stuck = torch.zeros_like(nrm, dtype=torch.bool)
    eye = torch.eye(J.shape[-1], dtype=J.dtype)[None]
    short = torch.tensor(STEPS[1:], dtype=z.dtype)
    ns = short.numel()
    sc = torch.ones_like(z) if scale is None else scale
    big = lambda v: v.abs().amax(-1).nan_to_num(INF, INF, INF)    # noqa: E731
    for _ in range(iters):
        act = ((nrm >= tol) & ~stuck).nonzero().flatten()
        if act.numel() == 0:
            break
        Ja = J[act]
        Ja = torch.where(torch.isfinite(Ja).flatten(1).all(-1)[:, None, None], Ja, eye.expand_as(Ja))
        with one(Ja):
            lu, piv, info = torch.linalg.lu_factor_ex(Ja)
        sing = (info != 0)[:, None]

        def step(Fv, rows=None):                                  # the Newton correction of the residual Fv with this Jacobian
            a, b = (lu, piv) if rows is None else (lu[rows], piv[rows])
            with one(a):
                dz = torch.linalg.lu_solve(a, b, -Fv.nan_to_num(0., 0., 0.)[..., None])[..., 0]
            return torch.where(sing if rows is None else sing[rows], torch.zeros_like(dz), dz).nan_to_num(0., 0., 0.)
        dz = step(F[act])
        if maxstep is not None:
            dz = torch.minimum(torch.maximum(dz, -maxstep[act]), maxstep[act])
        size = big(dz/sc[act])
        zn = torch.minimum(torch.maximum(z[act]+dz, lo[act]), hi[act])            # the full step first
        Fn, Jn = res(zn, act)
        nn = big(Fn)
        better = (big(step(Fn)/sc[act]) < size) | (nn < tol)
        b = act[better]
        z[b], F[b], J[b], nrm[b] = zn[better], Fn[better], Jn[better], nn[better]
        rest = (~better).nonzero().flatten()
        if rest.numel():                                          # the shorter steps of the others, in one evaluation
            i = act[rest]
            m = i.numel()
            zc = torch.minimum(torch.maximum(z[i][:, None]+short[None, :, None]*dz[rest][:, None], lo[i][:, None]), hi[i][:, None]).reshape(m*ns, -1)
            rep = rest.repeat_interleave(ns)
            Fc, Jc = res(zc, i.repeat_interleave(ns))
            nc = big(Fc)
            good = ((big(step(Fc, rep)/sc[act][rep]) < size[rep]) | (nc < tol)).view(m, ns)
            pick = torch.arange(m)*ns+good.to(torch.int64).argmax(1)
            ok = good.any(1)
            b = i[ok]
            z[b], F[b], J[b], nrm[b] = zc[pick][ok], Fc[pick][ok], Jc[pick][ok], nc[pick][ok]
            stuck[i[~ok]] = True                                  # no step helped: no solution nearby
    return z, nrm < 1e3*tol

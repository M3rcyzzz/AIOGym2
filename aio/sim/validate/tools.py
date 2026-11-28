"""Tools of the validation scripts.

`Unit` is one plant of the engine as plain numpy functions in float64. `Unit.flow` is the reference every
trajectory is compared with: the same equations integrated by LSODA (ODEPACK, through scipy) to a relative
tolerance of 1e-10, with the algebraic states solved to rounding inside every evaluation. It shares no code
with the engine's integrator (ROS2, one Jacobian per decision). `Tap` records what the shell feeds a plant
during a rollout, so that the comparison is made on the inputs of training episodes and not on a test
signal chosen here.
"""
import numpy as np
import torch
from scipy.integrate import solve_ivp

from aio.sim import plants as PL
from aio.sim.engine import Mech

F = torch.float64


def _t(a):
    return torch.as_tensor(np.atleast_2d(np.asarray(a, dtype=np.float64)), dtype=F)


class Unit:
    """One plant of the engine (by name, or a Plant), one episode, float64."""

    def __init__(self, plant):
        self.P = (PL.PLANTS.get(plant) or PL.LARGE[plant])() if isinstance(plant, str) else plant
        self.m = m = Mech([self.P], n=1, dtype=F)
        P = self.P
        self.nx, self.nw, self.npar, self.ny = P.nx, P.nw, max(P.np, 1), P.ny
        self.alg = (m.mass[0] == 0).numpy()
        self.dif = ~self.alg
        self.h = float(m.h[0])                                    # one decision in model time
        self.xscale = m.xscale[0].numpy()
        self.zstep = m.zstep[0].numpy()
        self.xlo, self.xhi = m.xlo[0].numpy(), m.xhi[0].numpy()
        self.x0, self.w0, self.p0 = m.x0[0].numpy(), m.wnom[0].numpy(), m.pnom[0].numpy()
        self.band = np.array([o.get('band', 1.) for o in P.ys])
        self.xname, self.wname, self.yname = [s['name'] for s in P.xs], [w['name'] for w in P.ws], [o['name'] for o in P.ys]
        self.pname = [s['name'] for s in P.ps]
        self.mv = [i for i, w in enumerate(P.ws) if w['kind'] == 'mv']
        self.shown = [i for i, o in enumerate(P.ys) if not o.get('dv')]      # outputs that are states of the plant (a measured load is an input)

    # ---- the plant's own functions
    def f(self, x, w, p):
        return self.m.f(_t(x), _t(w), _t(p))[0].numpy()

    def fj(self, x, w, p):
        f, J = self.m.fj(_t(x), _t(w), _t(p))
        return f[0].numpy(), J[0].numpy()

    def y(self, x, w, p):
        return self.m.y(_t(x), _t(w), _t(p))[0].numpy()

    def lin(self, x, w, p):
        """f, y, A, B, C, D at a point."""
        return tuple(a[0].numpy() for a in self.m.linearise(_t(x), _t(w), _t(p)))

    def steady(self, w=None, p=None, x0=None):
        w, p = self.w0 if w is None else w, self.p0 if p is None else p
        x, _, ok = self.m.steady(_t(w), _t(p), x0=None if x0 is None else _t(x0))
        return x[0].numpy(), bool(ok[0])

    def inputs(self, **kw):
        """The nominal inputs with some of them (by name) set."""
        w = self.w0.copy()
        for k, v in kw.items():
            w[self.wname.index(k)] = v
        return w

    def params(self, **kw):
        p = self.p0.copy()
        for k, v in kw.items():
            p[self.pname.index(k)] = v
        return p

    # ---- the reference
    def close(self, x, w, p, tol=1e-13):
        """x with its algebraic states at the root of their equations (Newton to rounding, the others held)."""
        if not self.alg.any():
            return x
        x, a = np.array(x, dtype=np.float64), self.alg
        for _ in range(60):
            f, J = self.fj(x, w, p)
            r = f[a]
            if np.abs(r/self.xscale[a]).max() < tol:
                break
            dz = np.linalg.solve(J[np.ix_(a, a)], -r)
            x[a] += np.clip(dz, -self.zstep[a], self.zstep[a])
        return x

    def flow(self, x0, w, p, T=None, rtol=1e-10, grid=None):
        """The states after model time T (one decision without it) with the inputs held, by LSODA; with
        `grid`, the states at those times as well (rows)."""
        T = self.h if T is None else T
        d, a = self.dif, self.alg
        z = [self.close(x0, w, p)]

        def full(xd):
            x = z[0].copy()
            x[d] = xd
            z[0] = x = self.close(x, w, p)
            return x

        def rhs(t, xd):
            return self.f(full(xd), w, p)[d]

        def jac(t, xd):
            J = self.fj(full(xd), w, p)[1]
            if a.any():
                return J[np.ix_(d, d)]-J[np.ix_(d, a)]@np.linalg.solve(J[np.ix_(a, a)], J[np.ix_(a, d)])
            return J
        sol = solve_ivp(rhs, (0., T), z[0][d], method='LSODA', jac=jac, rtol=rtol, atol=rtol*1e-2*self.xscale[d], t_eval=grid)
        assert sol.success, sol.message
        if grid is not None:
            return np.stack([full(c) for c in sol.y.T])
        return full(sol.y[:, -1])

    def run(self, x0, W, Pm, rtol=1e-10):
        """The reference over len(W) decisions, the inputs W[k] and parameters Pm[k] held over decision k:
        the states after every decision (K, nx)."""
        x, out = np.array(x0, dtype=np.float64), []
        for w, p in zip(W, Pm):
            x = self.flow(x, w, p, rtol=rtol)
            out.append(x)
        return np.stack(out)

    def engine(self, x0, W, Pm, substeps=None, refresh=None, dtype=F):
        """The engine's own integrator over the same decisions (as shipped without arguments)."""
        m = self.m if dtype == F else Mech([self.P], n=1, dtype=dtype)
        x, out = torch.as_tensor(np.atleast_2d(x0), dtype=dtype), []
        for w, p in zip(W, Pm):
            x = m.advance(x, torch.as_tensor(np.atleast_2d(w), dtype=dtype), torch.as_tensor(np.atleast_2d(p), dtype=dtype), substeps=substeps, refresh=refresh)
            out.append(x[0].double().numpy())
        return np.stack(out)

    def outputs(self, X, W, Pm):
        return self.m.y(_t(X), _t(W), _t(Pm)).numpy()

    # ---- linear dynamics at a steady point
    def modes(self, x, w, p):
        """Eigenvalues of the differential states' dynamics at a point (the algebraic states eliminated),
        per unit of model time."""
        _, _, A, *_ = self.lin(x, w, p)
        d, a = self.dif, self.alg
        Ad = A[np.ix_(d, d)]
        if a.any():
            Ad = Ad-A[np.ix_(d, a)]@np.linalg.solve(A[np.ix_(a, a)], A[np.ix_(a, d)])
        return np.linalg.eigvals(Ad)

    def dcgain(self, x, w, p):
        """Steady gains of the outputs to the inputs (ny, nw) of the continuous plant: D - C A^-1 B."""
        _, _, A, B, C, D = self.lin(x, w, p)
        return D-C@np.linalg.solve(A, B)

    def zeros(self, x, w, p, rows, cols):
        """Transmission zeros of the square system from inputs `cols` to outputs `rows`."""
        from scipy.linalg import eig
        _, _, A, B, C, D = self.lin(x, w, p)
        B, C, D = B[:, cols], C[rows], D[np.ix_(rows, cols)]
        n, m = A.shape[0], len(cols)
        M = np.block([[A, B], [C, D]])
        N = np.block([[np.eye(n), np.zeros((n, m))], [np.zeros((m, n)), np.zeros((m, m))]])
        v = eig(M, N, right=False)
        return v[np.isfinite(v) & (np.abs(v) < 1e6)]


class Tap:
    """While open, records every call of Mech.advance: (mech, states before, inputs, parameters, states after),
    as the runtime made it (its dtype and device)."""

    def __enter__(self):
        self.calls, self._orig = [], Mech.advance
        tap = self

        def advance(m, x, w, p, substeps=None, refresh=None):
            out = tap._orig(m, x, w, p, substeps=substeps, refresh=refresh)
            tap.calls.append((m, x.detach().cpu(), w.detach().cpu(), p.detach().cpu(), out.detach().cpu()))
            return out
        Mech.advance = advance
        return self

    def __exit__(self, *exc):
        Mech.advance = self._orig


def episodes(name, n, seed, controller='pi', steps=None):
    """n training episodes on one plant, run by a reference controller: per decision the plant's states
    before, its inputs, its parameters and its states after, each (K, n, .), as the runtime computed them;
    the runtime's dtype; and the rollout's result."""
    from aio.sim import core as C, control as CT, mech as ME
    from aio.sim.prior import HORIZON
    prog = C.build(n, seed, mech=1., plants=ME.pool([name]), quiet=True)
    ctl = dict(pi=CT.PI, hold=CT.Hold, teacher=lambda: CT.Teacher('late'))[controller]()
    with Tap() as tap:
        res = CT.rollout(prog, ctl, steps=steps or HORIZON)
    assert len({id(c[0]) for c in tap.calls}) == 1, 'one plant, one group'
    X0, W, Pm, X1 = (torch.stack([c[i] for c in tap.calls]).double().numpy() for i in (1, 2, 3, 4))
    return dict(x0=X0, w=W, p=Pm, x1=X1, dtype=tap.calls[0][1].dtype, failed=res['failed'].cpu().numpy(), score=res['score'].cpu().numpy())

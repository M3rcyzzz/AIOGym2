"""The designed columns of aio.sim.flow against an independent stage-by-stage solution.

The engine's column is built of units on fitted local property models. Here the same column (the same number
of theoretical stages, feed, reflux, bottoms rate and stage pressures, read from the engine's steady state) is
solved again as the textbook equilibrium-stage model (material balances, equilibrium and energy balance on
every stage; Naphtali-Sandholm unknowns, Newton), by a solver that shares no code with the engine, on

  fitted      the rigorous model the local models were fitted to (aio.sim.props_fit.source).
              What differs from the engine is then the local fit and the unit equations.
  reference   property models that owe nothing to that package: the multi-fluid Helmholtz model of CoolProp
              (reference equations of the pure fluids, GERG-2008 mixing) for the hydrocarbons; for methanol
              and water the activity coefficients of modified UNIFAC (Dortmund) with vapour pressures and
              enthalpies of the pure fluids from their reference equations.
              What differs is then the choice of property model: how much of the column's answer hangs on it.

    python -m aio.sim.validate.columns btcol meoh debut     # needs `thermo` and `CoolProp`; writes reports/validation/columns.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[3]/'reports'/'validation'
NAMES = {'propane': 'Propane', 'isobutane': 'IsoButane', 'n-butane': 'n-Butane', 'isopentane': 'Isopentane', 'n-pentane': 'n-Pentane',
         'benzene': 'Benzene', 'toluene': 'Toluene', 'methanol': 'Methanol', 'water': 'Water'}
SHIFT = dict(F=1.05, zF=.02, reflux=.03, steam=.03)                # the second operating point: feed rate times; light share, reflux and steam valves plus


def engine_state(name, shift=None):
    """A steady state of a designed column in the engine (the design point, or with the inputs shifted): per
    stage (sump, trays from the bottom, drum) composition, temperature and pressure; the internal flows; feed,
    products and duties."""
    import torch
    from aio.sim import flow, plants as PL
    from aio.sim.validate.tools import Unit, _t
    P = PL.LARGE[name]()
    u = Unit(P)
    st = P.units['stages']
    pkg = st[0].pkg
    nc, ns = pkg.nc, len(st)
    x0, w0 = u.x0, u.w0.copy()
    if shift:
        k = {n: i for i, n in enumerate(u.wname)}
        w0[k['F']] *= shift['F']
        for q in ('zF', 'reflux', 'steam'):
            w0[k[q]] += shift[q]
        x, _, ok = u.m.steady(_t(w0), _t(u.p0), homotopy=8)
        assert bool(ok[0]), f'{name}: no steady state at the shifted inputs'
        x0 = x[0].numpy()
    ix = {n: i for i, n in enumerate(u.xname)}
    n = np.array([[x0[ix[f'{s.name}.{sp}']] for sp in pkg.species] for s in st])
    T = np.array([x0[ix[f'{s.name}.T']] for s in st])
    x = n/n.sum(1, keepdims=True)
    Pj, y = np.zeros(ns), np.zeros((ns, nc))
    for j in range(ns):
        pj, yj = pkg.bubble(list(x[j]), T[j])
        Pj[j], y[j] = float(pj), [float(v) for v in yj]
    Fl, zf, _ = P.units['feed']
    ex = list(P.units['V'])+list(P.units['L'])+[Fl]+list(zf)
    val = flow.probe(P, ex, torch.as_tensor(x0)[None], torch.as_tensor(w0)[None])[0].numpy()
    nV, nL = len(P.units['V']), len(P.units['L'])
    out = dict(zip(u.yname, u.y(x0, w0, u.p0)))
    w = dict(zip(u.wname, w0))
    return dict(species=list(pkg.species), model=pkg.d['model'], nstage=P.design['nstage'], nfeed=P.design['nfeed'], light=list(P.design['light']),
                x=x, y=y, T=T, P=Pj, V=val[:nV], L=val[nV:nV+nL], F=float(val[nV+nL]), z=val[nV+nL+1:], TF=float(w['TF']),
                reflux=float(out['L']), D=float(out['D']), B=float(out['B']), Qr=float(out['Qr']), Qc=float(out['Qc']), yD=float(out['yD']), yB=float(out['yB']),
                band=dict(zip(u.yname, u.band)))


# ---- property models: K values and molar enthalpies (J/mol) of a liquid x and a vapour y at T (K), P (Pa)
class Fitted:
    """The rigorous model the engine's local models were fitted to (aio.sim.props_fit.source; pressures there in kPa)."""

    def __init__(self, species, model):
        from aio.sim.props_fit import source
        self.src = source(species, model)

    def kh(self, T, P, x, y):
        return self.src.kh(T, P/1e3, x, y)

    def hl(self, T, P, x):
        return self.src.hl(T, P/1e3, x)


class Helmholtz:
    """CoolProp's multi-fluid Helmholtz model: both phases from one equation of state."""

    def __init__(self, species):
        import CoolProp
        self.cp, self.n = CoolProp, len(species)
        mix = '&'.join(NAMES[s] for s in species)
        self.L, self.G = CoolProp.AbstractState('HEOS', mix), CoolProp.AbstractState('HEOS', mix)
        self.L.specify_phase(CoolProp.iphase_liquid)
        self.G.specify_phase(CoolProp.iphase_gas)

    def at(self, S, T, P, z):
        S.set_mole_fractions([float(v) for v in z])
        S.update(self.cp.PT_INPUTS, float(P), float(T))
        return S

    def kh(self, T, P, x, y):
        L, G = self.at(self.L, T, P, x), self.at(self.G, T, P, y)
        return np.array([L.fugacity_coefficient(i)/G.fugacity_coefficient(i) for i in range(self.n)]), L.hmolar(), G.hmolar()

    def hl(self, T, P, x):
        return self.at(self.L, T, P, x).hmolar()


class Dortmund:
    """Activity coefficients and heat of mixing of modified UNIFAC (Dortmund) over an ideal vapour (as its
    parameters were fitted); vapour pressures and the enthalpies of the pure liquids and vapours from their
    reference equations (the vapour at the stage pressure, or saturated where it could not exist there)."""
    GROUPS = {'methanol': {15: 1}, 'water': {16: 1}, 'benzene': {9: 6}, 'toluene': {9: 5, 11: 1}}

    def __init__(self, species):
        import CoolProp
        from thermo.unifac import DOUFIP2016, DOUFSG, UNIFAC
        self.cp, self.n = CoolProp, len(species)
        self.ge = UNIFAC.from_subgroups(T=350., xs=[1./self.n]*self.n, chemgroups=[self.GROUPS[s] for s in species], version=1,
                                        interaction_data=DOUFIP2016, subgroups=DOUFSG)
        self.S = [CoolProp.AbstractState('HEOS', NAMES[s]) for s in species]
        self.V = [CoolProp.AbstractState('HEOS', NAMES[s]) for s in species]
        for v in self.V:
            v.specify_phase(CoolProp.iphase_gas)

    def pure(self, i, T, P):
        S, cp = self.S[i], self.cp
        S.update(cp.QT_INPUTS, 0., float(T))
        ps, hl = S.p(), S.hmolar()
        if P < ps:
            self.V[i].update(cp.PT_INPUTS, float(P), float(T))
            return ps, hl, self.V[i].hmolar()
        S.update(cp.QT_INPUTS, 1., float(T))
        return ps, hl, S.hmolar()

    def kh(self, T, P, x, y):
        g = self.ge.to_T_xs(float(T), [float(v) for v in x])
        ps, hl, hv = (np.array(v) for v in zip(*[self.pure(i, T, P) for i in range(self.n)]))
        return np.array(g.gammas())*ps/P, float((x*hl).sum()+g.HE()), float((y*hv).sum())

    def hl(self, T, P, x):
        return self.kh(T, P, np.asarray(x), np.asarray(x))[1]


def models(species, model):
    """The fitted package, and the reference model for these species."""
    return dict(fitted=lambda: Fitted(species, model), reference=lambda: Dortmund(species) if 'water' in species else Helmholtz(species))


class Rigorous:
    """Equilibrium stages 0 (reboiler) .. N (top tray) under a total condenser with boiling reflux.
    Unknowns per stage: the component flows of the liquid and of the vapour leaving it (kmol/h), and T."""

    def __init__(self, e, pk):
        self.e, self.pk = e, pk
        self.nc, self.N = len(e['species']), e['nstage']
        self.Pst, self.Pd = e['P'][:-1]*1e3, e['P'][-1]*1e3       # Pa: stages, drum
        self.F, self.z, self.nf = e['F'], np.asarray(e['z']), e['nfeed']
        self.hF = pk.hl(e['TF'], self.Pst[self.nf], self.z)       # J/mol = kJ/kmol
        _, hl, hv = pk.kh(e['T'][0], self.Pst[0], e['x'][0], e['y'][0])
        self.lat = abs(hv-hl)
        self.m = 2*self.nc+1
        self.Td, self.yd = float(e['T'][-1]), np.asarray(e['y'][-1])
        self.h = np.tile(np.r_[np.full(2*self.nc, 1e-6*self.F), 1e-4], self.N+1)
        self.evals = 0

    def start(self):
        e, nc = self.e, self.nc
        X = np.zeros((self.N+1, self.m))
        for j in range(self.N+1):
            Lj = e['B'] if j == 0 else e['L'][j-1]                # liquid leaving stage j = liquid entering stage j-1
            X[j, :nc], X[j, nc:2*nc], X[j, -1] = Lj*e['x'][j], e['V'][j]*e['y'][j], e['T'][j]
        return X.ravel()

    def boiling(self, x):
        """Temperature and enthalpy of the liquid x boiling at the drum pressure (secant on ln sum K x; the
        vapour over it by substitution)."""
        def g(T, y):
            for _ in range(200):
                K, hl, _ = self.pk.kh(T, self.Pd, x, y)
                s = float((K*x).sum())
                yn = K*x/s
                if np.abs(yn-y).max() < 1e-14:
                    break
                y = yn
            return math.log(s), yn, hl
        T0, y = self.Td, self.yd
        f0, y, hl = g(T0, y)
        T1 = T0+.2
        for _ in range(60):
            f1, y, hl = g(T1, y)
            if abs(f1) < 1e-13 or f1 == f0:
                break
            T0, T1, f0 = T1, T1-f1*(T1-T0)/(f1-f0), f1
        self.Td, self.yd = T1, y
        return T1, hl

    def stage(self, T, P, l, v):
        x, y = l/l.sum(), v/v.sum()
        K, hl, hv = self.pk.kh(T, P, x, y)
        return x, y, K, hl, hv

    def residual(self, X, reflux, B):
        nc, N, m = self.nc, self.N, self.m
        self.evals += 1
        X = X.reshape(N+1, m)
        l, v, T = np.maximum(X[:, :nc], 1e-14), np.maximum(X[:, nc:2*nc], 1e-14), X[:, -1]
        prop = [self.stage(T[j], self.Pst[j], l[j], v[j]) for j in range(N+1)]
        xd = v[N]/v[N].sum()
        _, hR = self.boiling(xd)
        R = np.zeros((N+1, m))
        for j in range(N+1):
            x, y, K, hL, hV = prop[j]
            lin, hin = (reflux*xd, hR) if j == N else (l[j+1], prop[j+1][3])
            vin, hvin = (np.zeros(nc), 0.) if j == 0 else (v[j-1], prop[j-1][4])
            f = self.F*self.z if j == self.nf else 0.
            R[j, :nc] = (lin+vin+f-l[j]-v[j])/self.F
            R[j, nc:2*nc] = K*x-y
            if j == 0:                                            # the reboiler: its duty is whatever leaves B at the bottom
                R[j, -1] = (l[0].sum()-B)/self.F
            else:
                R[j, -1] = (lin.sum()*hin+vin.sum()*hvin+(self.F*self.hF if j == self.nf else 0.)-l[j].sum()*hL-v[j].sum()*hV)/(self.F*self.lat)
        return R.ravel()

    def jacobian(self, X, R, reflux, B):
        """Finite differences; a stage's equations see its neighbours only, so every third stage is moved at once."""
        N1, m = self.N+1, self.m
        J = np.zeros((N1*m, N1*m))
        for g in range(3):
            for k in range(m):
                idx = np.arange(g, N1, 3)*m+k
                Xp = X.copy()
                Xp[idx] += self.h[idx]
                d = (self.residual(Xp, reflux, B)-R)/self.h[k]
                for j in range(g, N1, 3):
                    r0, r1 = max(j-1, 0)*m, min(j+2, N1)*m
                    J[r0:r1, j*m+k] = d[r0:r1]
        return J

    def solve(self, reflux, B, X0=None, tol=1e-9, iters=60):
        """Newton with a backtracking step; flows stay positive, temperatures move by 8 K at most."""
        X = self.start() if X0 is None else X0.copy()
        flow = np.tile(np.r_[np.ones(2*self.nc, bool), False], self.N+1)
        R = self.residual(X, reflux, B)
        for _ in range(iters):
            r = float(np.abs(R).max())
            if r < tol:
                break
            d = np.linalg.solve(self.jacobian(X, R, reflux, B), -R)
            a = min(1., 8./max(float(np.abs(d[~flow]).max()), 1e-9))
            for _ in range(12):
                Xn = X+a*d
                Xn[flow] = np.maximum(Xn[flow], .1*X[flow])
                try:
                    Rn = self.residual(Xn, reflux, B)
                    ok = np.isfinite(Rn).all() and np.abs(Rn).max() < (1-1e-4*a)*r
                except Exception:                                 # a phase the equation of state cannot find at the trial point
                    ok = False
                if ok:
                    break
                a *= .5
            if not ok:
                break
            X, R = Xn, Rn
        return X, float(np.abs(R).max())

    def read(self, X, reflux):
        nc, N, m, e = self.nc, self.N, self.m, self.e
        X = X.reshape(N+1, m)
        l, v, T = X[:, :nc], X[:, nc:2*nc], X[:, -1]
        xd, xb = v[N]/v[N].sum(), l[0]/l[0].sum()
        light = np.array(e['light'])
        Td, hR = self.boiling(xd)
        p0, p1, pN = self.stage(T[0], self.Pst[0], l[0], v[0]), self.stage(T[1], self.Pst[1], l[1], v[1]), self.stage(T[N], self.Pst[N], l[N], v[N])
        Qr = l[0].sum()*p0[3]+v[0].sum()*p0[4]-l[1].sum()*p1[3]
        Qc = v[N].sum()*(pN[4]-hR)
        return dict(yD=float(-math.log10(xd[~light].sum())), yB=float(-math.log10(xb[light].sum())), T=T.tolist(), Td=float(Td), xd=xd.tolist(), xb=xb.tolist(),
                    V=v.sum(1).tolist(), L=l.sum(1).tolist(), Qr=float(Qr), Qc=float(Qc))

    def specs(self, yD, yB, X, reflux, B, tol=2e-4):
        """The reflux and bottoms rate that give the purities yD, yB (Newton on the two numbers)."""
        q, d = np.array([reflux, B]), np.array([1., 1.])
        for _ in range(12):
            X, res = self.solve(q[0], q[1], X)
            g = self.read(X, q[0])
            d = np.array([g['yD']-yD, g['yB']-yB])
            if np.abs(d).max() < tol or res > 1e-6:
                break
            J = np.zeros((2, 2))
            for k in range(2):
                dq = np.zeros(2)
                dq[k] = 2e-3*q[k]
                Xk, _ = self.solve(q[0]+dq[0], q[1]+dq[1], X)
                gk = self.read(Xk, q[0]+dq[0])
                J[:, k] = (np.array([gk['yD']-yD, gk['yB']-yB])-d)/dq[k]
            step = np.linalg.solve(J, d)
            q = q-step*min(1., .2/float(np.abs(step/q).max()))    # a fifth of either flow at most per pass
        return X, q, g, bool(np.abs(d).max() < tol and res < 1e-6)


def against(e, pk, specs=True):
    """The engine's steady state against the stage-by-stage solution on one property model: at the same
    reflux and bottoms rate; and (specs) what that column needs for the engine's product purities."""
    r = Rigorous(e, pk)
    r0 = float(np.abs(r.residual(r.start(), e['reflux'], e['B'])).max())
    X, res = r.solve(e['reflux'], e['B'])
    g = r.read(X, e['reflux'])
    dT = np.array(g['T'])-e['T'][:-1]
    out = dict(residual=res, residual_at_engine_state=r0,
               solution=dict(yD=g['yD'], yB=g['yB'], Qr=g['Qr'], Qc=g['Qc'], Vb=g['V'][0], T_bottom=g['T'][0], T_top=g['T'][-1], T_drum=g['Td'], T=g['T']),
               same_flows=dict(yD_bands=(g['yD']-e['yD'])/e['band']['yD'], yB_bands=(g['yB']-e['yB'])/e['band']['yB'], T_max=float(np.abs(dT).max()),
                               T_drum=g['Td']-float(e['T'][-1]), Qr=g['Qr']/e['Qr']-1, Qc=g['Qc']/e['Qc']-1, Vb=g['V'][0]/float(e['V'][0])-1))
    if specs:
        X, q, g, ok = r.specs(e['yD'], e['yB'], X, e['reflux'], e['B'])
        out['same_purity'] = dict(converged=ok, reflux=float(q[0]/e['reflux']-1), bottoms=float(q[1]/e['B']-1), Qr=g['Qr']/e['Qr']-1, Qc=g['Qc']/e['Qc']-1,
                                  Vb=g['V'][0]/float(e['V'][0])-1, T_max=float(np.abs(np.array(g['T'])-e['T'][:-1]).max()), yD=g['yD'], yB=g['yB'])
    out['evaluations'] = r.evals
    return out


def compare(name, which=('fitted', 'reference')):
    """A column at its design point (both property models) and at shifted inputs (the fitted one)."""
    e = engine_state(name)
    mk = models(e['species'], e['model'])
    eng = lambda e: dict(yD=e['yD'], yB=e['yB'], reflux=e['reflux'], distillate=e['D'], bottoms=e['B'], feed=e['F'], Qr=e['Qr'], Qc=e['Qc'], Vb=float(e['V'][0]),     # noqa: E731
                         T_bottom=float(e['T'][0]), T_top=float(e['T'][-2]), T_drum=float(e['T'][-1]), P_top=float(e['P'][-2]), P_bottom=float(e['P'][0]), T=e['T'][:-1].tolist())
    out = dict(plant=name, species=e['species'], stages=e['nstage'], feed_stage=e['nfeed'], engine=eng(e))
    for k in which:
        out[k] = against(e, mk[k]())
    if 'fitted' in which:
        s = engine_state(name, SHIFT)
        out['shifted'] = dict(inputs=SHIFT, engine=eng(s), fitted=against(s, mk['fitted'](), specs=False))
    return out


def table(res=None):
    res = json.loads((OUT/'columns.json').read_text()) if res is None else res
    rows = ['plant   model      yD     yB (bands)  T max K  Qr     Qc    | same purity: reflux  Qr']
    for name, r in res.items():
        for k, d in (('fitted', r.get('fitted')), ('reference', r.get('reference')), ('shifted', r.get('shifted', {}).get('fitted'))):
            if d:
                s, q = d['same_flows'], d.get('same_purity')
                rows.append('%-7s %-9s %+6.2f %+6.2f      %5.2f  %+5.1f%% %+5.1f%% | %s' % (name, k, s['yD_bands'], s['yB_bands'], s['T_max'], 100*s['Qr'], 100*s['Qc'],
                            '%+5.1f%% %+5.1f%%%s' % (100*q['reflux'], 100*q['Qr'], '' if q['converged'] else ' (not converged)') if q else ''))
    return '\n'.join(rows)


if __name__ == '__main__':
    f = OUT/'columns.json'
    res = json.loads(f.read_text()) if f.is_file() else {}
    for name in sys.argv[1:] or ('btcol', 'meoh', 'debut'):
        res[name] = compare(name)
        f.write_text(json.dumps(res, indent=1))
        print(table({name: res[name]}), flush=True)

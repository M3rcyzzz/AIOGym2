"""Physical properties of the flowsheet units (docs/sim.md, 'Properties').

A `Package` is a set of species over a window of pressure, with local models fitted to a rigorous property
package (aio/sim/props_fit.py; coefficients and their errors in props.json). Every function takes numbers,
arrays or expressions of a plant (aio.sim.expr), so that one formula serves the plant's table, the sizing
of the equipment and the tests. Units: kmol, kJ, K, kPa, m3.

  psat(T)          per species: the effective vapour pressure (the pure one with NRTL)
  gamma(x, T)      activity coefficients (NRTL, or the fitted composition term)
  bubble(x, T)     pressure of a boiling liquid and the vapour in equilibrium with it: explicit, as K_i P is
                   linear in P
  hl, hv, cpl      molar enthalpies of liquid and vapour; vl: liquid molar volume; z: vapour compressibility; mw
  he, he_i, cpe    heat of mixing of the liquid where the package has one (Redlich-Kister in every pair of
                   species, linear in temperature), its partial molar parts and its temperature derivative
"""
import functools
import json
import pathlib
import numpy as np

from aio.sim import expr as X

R = 8.314462618                  # kJ/(kmol K) = kPa m3/(kmol K)
PATH = pathlib.Path(__file__).with_name('props.json')


def _exp(a):
    return X.exp(a) if isinstance(a, X.E) else np.exp(a)


def dot(c, v):
    """sum c_i v_i (as one sum where the terms are expressions)."""
    r = 0.
    for a, b in zip(c, v):
        r = r+a*b
    return r


class Package:
    def __init__(self, name, d):
        self.name, self.d = name, d
        self.species, self.nc, self.model = d['species'], len(d['species']), d['model']
        self.mw, self.T0, self.P0 = d['mw'], d['T0'], d.get('P0', 100.)
        self.ant, self.hlc, self.hvc, self.vlc, self.zc = d['ant'], d['hl'], d['hv'], d['vl'], d['z']
        self.g, self.ep, self.hec = d.get('g'), d.get('ep'), d.get('he')
        if self.g is not None and not any(v for r in self.g for v in r):
            self.g = None
        if self.ep is not None and not any(self.ep):
            self.ep = None

    def index(self, name):
        return self.species.index(name)

    # ---- equilibrium
    def psat(self, T, k=None):
        ps = [_exp(a-b/(T+c)) for a, b, c in self.ant]
        return ps if k is None else [p*q for p, q in zip(ps, k)]

    def gamma(self, x, T):
        if self.model != 'nrtl':
            return [1.]*self.nc if self.g is None else [_exp(dot(r, x)) for r in self.g]
        a, b, al, n = self.d['a'], self.d['b'], self.d['alpha'], self.nc
        it = 1./T
        tau = [[a[i][j]+b[i][j]*it for j in range(n)] for i in range(n)]
        G = [[_exp(-al[i][j]*tau[i][j]) if i != j else 1. for j in range(n)] for i in range(n)]
        S = [dot(x, [G[k][j] for k in range(n)]) for j in range(n)]
        C = [dot([x[m] for m in range(n) if m != j], [tau[m][j]*G[m][j] for m in range(n) if m != j]) for j in range(n)]
        out = []
        for i in range(n):
            lg = C[i]/S[i]
            for j in range(n):
                lg = lg+x[j]*G[i][j]/S[j]*(tau[i][j]-C[j]/S[j])
            out.append(_exp(lg))
        return out

    def bubble(self, x, T, k=None):
        """P, y of a liquid x boiling at T (k: per species factors on the volatility, a plant's mismatch)."""
        ps, g = self.psat(T, k), self.gamma(x, T)
        pp = [x[i]*g[i]*ps[i] for i in range(self.nc)]            # x_i K_i P at the reference pressure
        P = sum(pp[1:], pp[0])
        if self.ep is not None:                                   # K_i P = (.)(1 + E_i (P - P0)/P0): P solves a linear equation
            b = dot(self.ep, pp)
            P = (P-b)/(1.-b/self.P0)
        P = P.held if isinstance(P, X.E) else P
        if self.ep is not None:
            r = (P-self.P0)/self.P0
            pp = [q*(1.+e*r) for q, e in zip(pp, self.ep)]
        return P, [q/P for q in pp]

    def bubble_T(self, x, P, T=None, k=None):
        """The boiling temperature of x at P (numbers): Newton on ln P."""
        x = [float(v) for v in x]
        T = float(self.T0 if T is None else T)
        for _ in range(60):
            f = np.log(self.bubble(x, T, k)[0]/P)
            df = (np.log(self.bubble(x, T+.01, k)[0]/P)-f)/.01
            dT = float(np.clip(-f/df, -30., 30.))
            T += dT
            if abs(dT) < 1e-10:
                break
        return T

    # ---- enthalpy, volume
    def _poly(self, c, T):
        t = T-self.T0
        return [a+b*t+q*t*t for a, b, q in c]

    def hl_i(self, T, x=None):
        """Per species: the enthalpy of the pure liquid; with x, its partial molar enthalpy in that mixture."""
        h = self._poly(self.hlc, T)
        return h if x is None or self.hec is None else [a+b for a, b in zip(h, self.he_i(x, T))]

    def _pairs(self, T, slope=False):
        """Per pair of species (i, j, coefficients c_k): the heat of mixing is x_i x_j sum_k c_k (x_i - x_j)^k."""
        t = T-self.T0
        return [(i, j, [b if slope else a+b*t for a, b in ck]) for i, j, ck in self.hec]

    def he(self, x, T, slope=False):
        """Heat of mixing of the liquid (slope: its derivative in temperature)."""
        if self.hec is None:
            return 0.
        r = 0.
        for i, j, ck in self._pairs(T, slope):
            d, pw, q = x[i]-x[j], 1., 0.
            for c in ck:
                q, pw = q+c*pw, pw*d
            r = r+x[i]*x[j]*q
        return r

    def cpe(self, x, T):
        return self.he(x, T, slope=True)

    def he_i(self, x, T):
        """Partial molar heats of mixing: of a term homogeneous of degree d in x, (1 - d) x the term plus its
        derivative in x_m."""
        out = [0.]*self.nc
        for i, j, ck in self._pairs(T):
            d, pw, low = x[i]-x[j], 1., 0.                         # pw = d^k, low = d^(k-1)
            for k, c in enumerate(ck):
                term = x[i]*x[j]*pw
                out = [o-(k+1)*c*term for o in out]
                out[i] = out[i]+c*(x[j]*pw+k*x[i]*x[j]*low)
                out[j] = out[j]+c*(x[i]*pw-k*x[i]*x[j]*low)
                low, pw = pw, pw*d
        return out

    def hv_i(self, T, P=None):
        h = self._poly([c[:3] for c in self.hvc], T)
        if P is None or len(self.hvc[0]) < 4:
            return h
        r = (P-self.P0)/self.P0
        return [a+c[3]*r for a, c in zip(h, self.hvc)]

    def cpl_i(self, T):
        t = T-self.T0
        return [b+2*q*t for _, b, q in self.hlc]

    def vl_i(self, T):
        t = T-self.T0
        return [a+b*t for a, b in self.vlc]

    def hl(self, x, T):
        return dot(x, self.hl_i(T))+self.he(x, T)

    def cpl(self, x, T):
        return dot(x, self.cpl_i(T))+self.cpe(x, T)

    def hv(self, y, T, P=None):
        return dot(y, self.hv_i(T, P))

    def vl(self, x, T):
        return dot(x, self.vl_i(T))

    def z(self, y):
        return dot(y, self.zc)

    def mwm(self, x):
        return dot(x, self.mw)


@functools.lru_cache(None)
def _data():
    return json.loads(PATH.read_text())


@functools.lru_cache(None)
def package(name):
    return Package(name, _data()[name])


def synthetic(name, alpha, T0=350., P0=100., B=3000., C=-40., latent=3e4, cpl=1., cpv=1., vl=.1, mw=50.):
    """Ideal species of constant relative volatilities `alpha` (to the last one), equal heats of vaporisation and
    heat capacities: the textbook column (constant molar overflow) as a limit of the same units."""
    n = len(alpha)
    A = np.log(P0)+B/(T0+C)
    d = dict(species=[f's{i}' for i in range(n)], model='ideal', mw=[mw]*n, T0=T0, ant=[[float(A+np.log(a)), B, C] for a in alpha],
             hl=[[0., cpl, 0.]]*n, hv=[[latent, cpv, 0.]]*n, vl=[[vl, 0.]]*n, z=[1.]*n)
    return Package(name, d)

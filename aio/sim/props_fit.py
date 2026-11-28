"""Local property models fitted to a rigorous one (docs/sim.md, 'Properties').

Offline. The rigorous model of a package (`source`):

  pr      Peng-Robinson with ChemSep's interaction parameters, both phases (hydrocarbons; `thermo`)
  nrtl    NRTL with ChemSep's parameters for the activity coefficients, and every pure fluid by its reference
          equation of state (`CoolProp`: IAPWS-95 for water, de Reuck and Craven for methanol): vapour
          pressure, enthalpy of the saturated liquid, enthalpy and compressibility of the vapour. The heat of
          mixing is modified UNIFAC's (Dortmund), which is fitted to heats of mixing; NRTL parameters fitted
          to phase equilibria give it the wrong sign for methanol-water (+0.3 against -0.8 kJ/mol at 25 C).
          With an ideal gas and the liquid enthalpy taken from the vapour pressure curve instead (the first
          version) the heat of vaporisation of methanol was 6 % high and the liquid heat capacities 13 and
          20 % low (validation thread, 11-10).
  ref     one fluid by its reference equation of state (steam: IAPWS-95)

The engine reads the fitted coefficients (aio/sim/props.json) and never imports these packages.

    python -m aio.sim.props_fit            # refit every package and rewrite props.json
    python -m aio.sim.props_fit c3c5       # one package

A package is a set of species over a window of pressure. Bubble points of mixtures across the whole
composition simplex are computed with the rigorous model; fitted to them, per species i:

  equilibrium   ln(K_i P) = A_i - B_i/(T + C_i) + sum_j g_ij x_j + ln(1 + E_i (P - P0)/P0)
                (with NRTL: the pure vapour pressure, and gamma_i from the published parameters). Linear in
                P, so the pressure of a boiling liquid stays explicit.
  enthalpy      h_L = sum x_i (a_i + b_i t + c_i t^2) + h_E, h_V = sum y_i (a_i + b_i t + c_i t^2 + e_i (P - P0)/P0),
                t = T - T0; liquid also at subcooled states, vapour also superheated. h_E, where the rigorous
                model has a heat of mixing of its own: sum over pairs x_i x_j sum_k (A_k + B_k t) (x_i - x_j)^k
  volume        v_L = sum x_i (v0_i + v1_i t)  (saturated liquid),  Z = sum y_i Z_i

and the errors of the fitted model on fresh mixtures are stored with it (bubble temperature, vapour
composition, enthalpies as a share of the heat of vaporisation).
Units: kmol, kJ, K, kPa, m3. The enthalpy zero is the ideal gas at 298.15 K, per species.
"""
import json
import pathlib
import sys
import numpy as np

PATH = pathlib.Path(__file__).with_name('props.json')
PACKAGES = dict(
    c3c5=dict(species=['propane', 'isobutane', 'n-butane', 'isopentane', 'n-pentane'], model='pr', P=(850., 1350.)),
    bt=dict(species=['benzene', 'toluene'], model='pr', P=(90., 220.)),
    meoh=dict(species=['methanol', 'water'], model='nrtl', P=(80., 180.)),
    steam=dict(species=['water'], model='ref', P=(250., 1600.)),
)
SOURCE = dict(pr='Peng-Robinson, ChemSep kij (thermo)',
              nrtl='NRTL, ChemSep parameters; heat of mixing of modified UNIFAC (Dortmund) (thermo); pure fluids by their reference equations of state (CoolProp)',
              ref='reference equation of state (CoolProp: IAPWS-95)')
FLUID = {'methanol': 'Methanol', 'water': 'Water'}                 # CoolProp's names
GROUPS = {'methanol': {15: 1}, 'water': {16: 1}}                   # modified UNIFAC (Dortmund) subgroups
HE_ORDER = 3                                                       # terms of the heat of mixing per pair
NS, SUB, SUP = 600, 40., 25.                                       # mixtures per package; the most subcooling and superheat sampled (K)


class Cubic:
    """Peng-Robinson for both phases (thermo). Pressures in kPa."""

    def __init__(self, species):
        from thermo import ChemicalConstantsPackage, CEOSGas, CEOSLiquid, PRMIX, FlashVL, FlashPureVLS
        from thermo.interaction_parameters import IPDB
        self.c, self.p = c, p = ChemicalConstantsPackage.from_IDs(species)
        eos = dict(Tcs=c.Tcs, Pcs=c.Pcs, omegas=c.omegas)
        if c.N > 1:
            eos['kijs'] = IPDB.get_ip_asymmetric_matrix('ChemSep PR', c.CASs, 'kij')
        self.gas = CEOSGas(PRMIX, eos_kwargs=eos, HeatCapacityGases=p.HeatCapacityGases)
        self.liq = CEOSLiquid(PRMIX, eos_kwargs=eos, HeatCapacityGases=p.HeatCapacityGases)
        self.fl = FlashVL(c, p, liquid=self.liq, gas=self.gas) if c.N > 1 else FlashPureVLS(c, p, gas=self.gas, liquids=[self.liq], solids=[])
        self.extra = {}

    def bubble(self, P, x):
        """T, y, and the compressibility of the vapour, of the liquid x boiling at P."""
        r = self.fl.flash(P=P*1e3, VF=0., zs=[float(v) for v in x])
        return r.T, (np.array(r.gas.zs) if self.c.N > 1 else np.ones(1)), r.gas.Z()

    def hl(self, T, P, x):
        return self.liq.to(T=T, P=P*1e3, zs=[float(v) for v in x]).H()

    def hv(self, T, P, y):
        return self.gas.to(T=T, P=P*1e3, zs=[float(v) for v in y]).H()

    def kh(self, T, P, x, y):
        """K values and the enthalpies of both phases at given compositions."""
        L, G = self.liq.to(T=T, P=P*1e3, zs=[float(v) for v in x]), self.gas.to(T=T, P=P*1e3, zs=[float(v) for v in y])
        return np.exp(np.array(L.lnphis())-np.array(G.lnphis())), L.H(), G.H()

    def psat(self, i, T):
        return self.p.VaporPressures[i].T_dependent_property(T)/1e3

    def vl(self, i, T):
        return self.p.VolumeLiquids[i].T_dependent_property(T)*1e3

    he = None                                                     # the heat of mixing is in the equation of state


class Fluids:
    """Every pure fluid by its reference equation of state, the liquid mixture by NRTL (one species: none).
    Vapour: the species mix ideally, each as its own gas at (T, P) where that exists (P below its vapour
    pressure); above, its departure from the ideal gas grows in proportion to the pressure, from the saturated
    vapour's (a second virial coefficient). Pressures in kPa."""

    def __init__(self, species):
        import CoolProp
        from thermo import ChemicalConstantsPackage
        self.cp = CoolProp
        self.c = ChemicalConstantsPackage.from_IDs(species)[0]
        self.n = len(species)
        self.S = [CoolProp.AbstractState('HEOS', FLUID[s]) for s in species]
        self.G = [CoolProp.AbstractState('HEOS', FLUID[s]) for s in species]
        for g in self.G:
            g.specify_phase(CoolProp.iphase_gas)
        self.h0 = [self._ideal(i, 298.15) for i in range(self.n)]
        self.ge, self.extra = None, {}
        if self.n > 1:
            from thermo import NRTL
            from thermo.interaction_parameters import IPDB
            b = IPDB.get_ip_asymmetric_matrix('ChemSep NRTL', self.c.CASs, 'bij')     # tau_ij = b_ij/T
            a = [[0.]*self.n for _ in range(self.n)]
            al = IPDB.get_ip_asymmetric_matrix('ChemSep NRTL', self.c.CASs, 'alphaij')
            self.ge = NRTL(T=350., xs=[1./self.n]*self.n, tau_as=a, tau_bs=b, alpha_cs=al)
            self.extra = dict(a=np.array(a).tolist(), b=np.array(b).tolist(), alpha=np.array(al).tolist())
            from thermo.unifac import DOUFIP2016, DOUFSG, UNIFAC
            self.do = UNIFAC.from_subgroups(T=350., xs=[1./self.n]*self.n, chemgroups=[GROUPS[s] for s in species], version=1,
                                            interaction_data=DOUFIP2016, subgroups=DOUFSG)

    def _ideal(self, i, T):
        self.G[i].update(self.cp.PT_INPUTS, 1., float(T))          # 1 Pa
        return self.G[i].hmolar()

    def sat(self, i, T):
        """Vapour pressure, enthalpy and molar volume of the saturated liquid, enthalpy and compressibility of the saturated vapour."""
        S = self.S[i]
        S.update(self.cp.QT_INPUTS, 0., float(T))
        ps, hl, vl = S.p()/1e3, S.hmolar()-self.h0[i], 1e3/S.rhomolar()
        S.update(self.cp.QT_INPUTS, 1., float(T))
        return ps, hl, vl, S.hmolar()-self.h0[i], S.compressibility_factor()

    def psat(self, i, T):
        return self.sat(i, T)[0]

    def vl(self, i, T):
        return self.sat(i, T)[2]

    def vapour(self, i, T, P):
        """Enthalpy and compressibility of species i as a gas at (T, P)."""
        ps, _, _, hs, zs = self.sat(i, T)
        if P < ps:
            G = self.G[i]
            G.update(self.cp.PT_INPUTS, P*1e3, float(T))
            return G.hmolar()-self.h0[i], G.compressibility_factor()
        hi = self._ideal(i, T)-self.h0[i]
        return hi+(hs-hi)*P/ps, 1+(zs-1)*P/ps

    def gamma(self, T, x):
        return np.ones(1) if self.ge is None else np.array(self.ge.to_T_xs(float(T), [float(v) for v in x]).gammas())

    def he(self, T, x):
        """Heat of mixing of the liquid."""
        return 0. if self.ge is None else self.do.to_T_xs(float(T), [float(v) for v in x]).HE()

    def hl(self, T, P, x):
        return float(sum(x[i]*self.sat(i, T)[1] for i in range(self.n))+self.he(T, x))

    def hv(self, T, P, y):
        return float(sum(y[i]*self.vapour(i, T, P)[0] for i in range(self.n)))

    def kh(self, T, P, x, y):
        sat = [self.sat(i, T) for i in range(self.n)]
        return self.gamma(T, x)*np.array([s[0] for s in sat])/P, float(sum(x[i]*sat[i][1] for i in range(self.n))+self.he(T, x)), self.hv(T, P, y)

    def bubble(self, P, x):
        from scipy.optimize import brentq
        x = np.asarray(x, dtype=float)
        tc = min(self.c.Tcs)
        pp = lambda T: x*self.gamma(T, x)*np.array([self.psat(i, T) for i in range(self.n)])      # noqa: E731
        T = brentq(lambda T: float(pp(T).sum())-P, 260., .98*tc, xtol=1e-10)
        y = pp(T)/P
        return T, y, float(sum(y[i]*self.vapour(i, T, P)[1] for i in range(self.n)))


def source(species, model):
    """The rigorous model a package is fitted to."""
    return Cubic(species) if model == 'pr' else Fluids(species)


def mixtures(rng, nc, n):
    """Compositions over the whole simplex: spread ones, ones near an edge, and ones with one species almost pure."""
    if nc == 1:
        return np.ones((n, 1))
    x = np.concatenate((rng.dirichlet(np.full(nc, 1.), n//3), rng.dirichlet(np.full(nc, .3), n//3), rng.dirichlet(np.full(nc, .08), n-2*(n//3))))
    x = np.maximum(x, 1e-7)
    return x/x.sum(1, keepdims=True)


def sample(src, nc, P, n, seed):
    """Bubble points: T, x, y, the enthalpies of both phases there, and of subcooled liquid and superheated vapour."""
    rng = np.random.default_rng(seed)
    X, Pk = mixtures(rng, nc, n), rng.uniform(P[0], P[1], n)
    out = dict(T=[], P=[], x=[], y=[], HL=[], HV=[], Z=[], Ts=[], HLs=[], Th=[], HVh=[])
    for x, pk in zip(X, Pk):
        try:
            T, y, Z = src.bubble(pk, x)
            ts, th = T-rng.uniform(0., SUB), T+rng.uniform(0., SUP)
            row = (T, pk, x, y, src.hl(T, pk, x), src.hv(T, pk, y), Z, ts, src.hl(ts, pk, x), th, src.hv(th, pk, y))
        except Exception:                                         # a mixture past its critical point in the window: not a point of the model
            continue
        for q, v in zip(out, row):
            out[q].append(v)
    return {q: np.array(v) for q, v in out.items()}


def antoine(T, lnp, more=None, cs=np.linspace(-120., 40., 161)):
    """ln p = A - B/(T + C) (+ linear terms in the columns of `more`) by least squares: linear for every C of a
    grid, the best C kept. Returns A, B, C and the coefficients of `more`."""
    best = None
    for c in cs:
        if T.min()+c < 20.:
            continue
        M = np.stack((np.ones_like(T), -1./(T+c)), 1)
        M = M if more is None else np.concatenate((M, more), 1)
        ab, *_ = np.linalg.lstsq(M, lnp, rcond=None)
        e = float(((M@ab-lnp)**2).sum())
        if best is None or e < best[0]:
            best = (e, [float(ab[0]), float(ab[1]), float(c)]+[float(v) for v in ab[2:]])
    return best[1]


def fit(name, spec):
    species, model, P = spec['species'], spec['model'], spec['P']
    src = source(species, model)
    c, nc = src.c, len(species)
    s = sample(src, nc, P, NS, 1)
    T0 = float(round(np.median(s['T'])))
    Tlo, Thi = float(s['T'].min()), float(s['T'].max())
    P0 = float(np.mean(P))
    ant, g, ep = [], [[0.]*nc for _ in range(nc)], [0.]*nc
    for i in range(nc):
        if model == 'nrtl' or nc == 1:                            # the pure vapour pressure
            hi = min(Thi+SUP, .97*c.Tcs[i])
            T = np.linspace(min(Tlo-SUB, hi-40.), hi, 80)
            lnp = np.log([src.psat(i, t) for t in T])
            ant.append(antoine(T, lnp)[:3])
        else:                                                     # K_i P of the mixtures: the fugacity corrections of the window are in the fit
            a = antoine(s['T'], np.log(s['y'][:, i]/s['x'][:, i]*s['P']), np.concatenate((s['x'][:, :-1], ((s['P']-P0)/P0)[:, None]), 1))
            ant.append(a[:3])
            g[i][:nc-1], ep[i] = a[3:3+nc-1], a[-1]
    quad = lambda t, w: np.concatenate((w, w*t[:, None], w*t[:, None]**2), 1)      # noqa: E731
    tl, xl, he = np.concatenate((s['T'], s['Ts']))-T0, np.concatenate((s['x'], s['x'])), None
    hl, *_ = np.linalg.lstsq(quad(tl, xl), np.concatenate((s['HL'], s['HLs'])), rcond=None)
    if src.he is not None and nc > 1:                             # the pure liquids on their own, the heat of mixing by pairs
        Tg = np.linspace(Tlo-SUB, Thi, 60)
        hl = np.array([np.polyfit(Tg-T0, [src.hl(t, P0, np.eye(nc)[i]) for t in Tg], 2)[::-1] for i in range(nc)]).T.ravel()
        pairs = [(i, j) for i in range(nc) for j in range(i+1, nc)]
        M = np.concatenate([np.stack([xl[:, i]*xl[:, j]*(xl[:, i]-xl[:, j])**k*tp for k in range(HE_ORDER) for tp in (np.ones_like(tl), tl)], 1) for i, j in pairs], 1)
        co, *_ = np.linalg.lstsq(M, np.array([src.he(t+T0, x) for t, x in zip(tl, xl)]), rcond=None)
        he = [[i, j, co[q*2*HE_ORDER:(q+1)*2*HE_ORDER].reshape(HE_ORDER, 2).tolist()] for q, (i, j) in enumerate(pairs)]
    y2, pr = np.concatenate((s['y'], s['y'])), ((np.concatenate((s['P'], s['P']))-P0)/P0)[:, None]
    hv, *_ = np.linalg.lstsq(np.concatenate((quad(np.concatenate((s['T'], s['Th']))-T0, y2), y2*pr), 1), np.concatenate((s['HV'], s['HVh'])), rcond=None)
    z, *_ = np.linalg.lstsq(s['y'], s['Z'], rcond=None)
    vl = []
    for i in range(nc):
        hi = min(Thi, .93*c.Tcs[i])
        T = np.linspace(min(Tlo-SUB, hi-30.), hi, 40)
        v = np.array([src.vl(i, t) for t in T])
        v1, v0 = np.polyfit(T-T0, v, 1)
        vl.append([float(v0), float(v1)])
    d = dict(species=species, model=model, source=SOURCE[model],
             mw=[float(v) for v in c.MWs], Tc=[float(v) for v in c.Tcs], hf=[float(v or 0.) for v in c.Hfgs],
             T0=T0, P0=P0, Tlo=Tlo, Thi=Thi, Plo=float(P[0]), Phi=float(P[1]), ant=ant, g=g, ep=ep,
             hl=hl.reshape(3, nc).T.tolist(), hv=hv.reshape(4, nc).T.tolist(), vl=vl, z=[float(v) for v in z], **src.extra)
    if he is not None:
        d['he'] = he
    d['fit'] = errors(d, sample(src, nc, P, NS//2, 2))
    return d


def errors(d, s):
    """The fitted model against fresh mixtures of the rigorous one."""
    from aio.sim.props import Package
    pk = Package('fit', d)
    lat = float(np.mean(s['HV']-s['HL']))
    T = np.array([pk.bubble_T(x, pr) for x, pr in zip(s['x'], s['P'])])
    Pb, y = pk.bubble(list(s['x'].T), T)
    y = np.stack(np.broadcast_arrays(*y), 1)
    hl, hv = pk.hl(list(s['x'].T), s['Ts']), pk.hv(list(s['y'].T), s['Th'], s['P'])
    hl0, hv0 = pk.hl(list(s['x'].T), s['T']), pk.hv(list(s['y'].T), s['T'], s['P'])
    q = lambda v: float(np.round(v, 6))                           # noqa: E731
    return dict(n=int(len(T)), latent=q(lat), dT_max=q(np.abs(T-s['T']).max()), dT_rms=q(np.sqrt(((T-s['T'])**2).mean())),
                dy_max=q(np.abs(y-s['y']).max()), dy_rms=q(np.sqrt(((y-s['y'])**2).mean())),
                dhl_max=q(max(np.abs(hl-s['HLs']).max(), np.abs(hl0-s['HL']).max())/lat),
                dhv_max=q(max(np.abs(hv-s['HVh']).max(), np.abs(hv0-s['HV']).max())/lat),
                dlat_max=q(np.abs((hv0-hl0)-(s['HV']-s['HL'])).max()/lat))


def main(names):
    data = json.loads(PATH.read_text()) if PATH.exists() else {}
    for q in names or PACKAGES:
        data[q] = fit(q, PACKAGES[q])
        print(q, json.dumps(data[q]['fit']))
    PATH.write_text(json.dumps(data, indent=1)+'\n')


if __name__ == '__main__':
    main(sys.argv[1:])

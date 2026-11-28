"""The engine's property packages (aio.sim.props) against property models that owe nothing to the package
they were fitted to.

  pure species   saturation temperature, heat of vaporisation, liquid heat capacity, liquid volume and vapour
                 compressibility against the reference equation of state of each fluid (CoolProp: IAPWS-95
                 for water, the Helmholtz equations of the others), over the package's pressure window
  mixtures       bubble temperature, vapour composition and the relative volatility of the key species against
                 the multi-fluid Helmholtz model (mixing parameters of GERG-2008 for the alkanes, of Bell and
                 Lemmon 2016 for benzene-toluene), and against modified UNIFAC (Dortmund) over reference
                 vapour pressures (methanol-water, benzene-toluene)
  c3c5           where the relative volatility of the keys comes from: Peng-Robinson with and without the
                 interaction parameters, the Helmholtz model, the pure vapour pressures

    python -m aio.sim.validate.refprops      # needs `CoolProp` and `thermo`; writes reports/validation/refprops.json
"""
import json

import numpy as np

from aio.sim import props as PP
from aio.sim.validate.columns import NAMES, OUT, Dortmund

KEYS = dict(c3c5=('n-butane', 'isopentane'), bt=('benzene', 'toluene'), meoh=('methanol', 'water'))
AT = dict(c3c5=1100., bt=105., meoh=105., steam=1000.)             # kPa: where the plants run


def _f(v):
    return float(v)


def pure(pkg, at):
    """Every species alone: engine minus reference (K) or over reference minus one, at the plant's pressure
    and the largest over the window."""
    import CoolProp.CoolProp as CP
    sp, nc = pkg.d['species'], len(pkg.d['species'])
    out = {}
    for i, s in enumerate(sp):
        e, f = [1. if j == i else 0. for j in range(nc)], NAMES[s]

        def row(P):
            T = _f(pkg.bubble_T(e, P))
            sat = lambda k, q: CP.PropsSI(k, 'P', P*1e3, 'Q', q, f)                          # noqa: E731
            Tr = sat('T', 0)
            lat, latr = _f(pkg.hv(e, T, P))-_f(pkg.hl(e, T)), sat('Hmolar', 1)-sat('Hmolar', 0)
            cpl = (_f(pkg.hl(e, T))-_f(pkg.hl(e, T-20.)))/20.
            cplr = (CP.PropsSI('Hmolar', 'P', P*1e3, 'T', Tr-.01, f)-CP.PropsSI('Hmolar', 'P', P*1e3, 'T', Tr-20., f))/19.99
            return dict(Tsat=T-Tr, latent=lat/latr-1, liquid_cp=cpl/cplr-1, liquid_volume=_f(pkg.vl(e, T))*sat('Dmolar', 0)/1e3-1, Z=_f(pkg.z(e))/sat('Z', 1)-1,
                        latent_engine=lat, latent_reference=latr)
        rows = [row(P) for P in np.linspace(pkg.d['Plo'], pkg.d['Phi'], 5)]
        out[s] = dict(at=row(at), window={k: max(abs(r[k]) for r in rows) for k in ('Tsat', 'latent', 'liquid_cp', 'liquid_volume', 'Z')})
    return out


def _stats(rows):
    r = np.array(rows)
    rms = lambda v: float(np.sqrt((v**2).mean()))                  # noqa: E731
    return dict(points=len(r), T_rms=rms(r[:, 0]), T_max=float(np.abs(r[:, 0]).max()), T_mean=float(r[:, 0].mean()), y_rms=rms(r[:, 1]), y_max=float(np.abs(r[:, 1]).max()),
                volatility_mean=float(r[:, 2].mean()), volatility_rms=rms(r[:, 2]))


def _engine(pkg, x, P, i, j):
    T = _f(pkg.bubble_T(list(x), P))
    y = np.array([_f(v) for v in pkg.bubble(list(x), T)[1]])
    return T, y, (y[i]/x[i])/(y[j]/x[j])


def helmholtz(name, n=300, seed=0):
    """Boiling mixtures over the composition simplex and the pressure window: engine against the Helmholtz model."""
    import CoolProp
    pkg = PP.package(name)
    sp = pkg.d['species']
    i, j = (sp.index(k) for k in KEYS[name])
    S = CoolProp.AbstractState('HEOS', '&'.join(NAMES[s] for s in sp))
    rng, rows, failed = np.random.default_rng(seed), [], 0
    for _ in range(n):
        x, P = rng.dirichlet(np.ones(len(sp))), rng.uniform(pkg.d['Plo'], pkg.d['Phi'])
        try:
            S.set_mole_fractions(list(x))
            S.update(CoolProp.PQ_INPUTS, P*1e3, 0.)
            Tr, yr = S.T(), np.array(S.mole_fractions_vapor())
        except Exception:
            failed += 1
            continue
        T, y, a = _engine(pkg, x, P, i, j)
        rows.append((T-Tr, np.abs(y-yr).max(), a/((yr[i]/x[i])/(yr[j]/x[j]))-1))
    return dict(_stats(rows), failed=failed)


def dortmund(name, n=25):
    """A binary over its composition range at three pressures: engine against modified UNIFAC (Dortmund)."""
    from scipy.optimize import brentq
    pkg = PP.package(name)
    D, rows = Dortmund(pkg.d['species']), []
    for P in (pkg.d['Plo'], AT[name], pkg.d['Phi']):
        for x1 in np.linspace(.02, .98, n):
            x = np.array([x1, 1-x1])
            T, y, a = _engine(pkg, x, P, 0, 1)
            Tr = brentq(lambda t: float((D.kh(t, P*1e3, x, x)[0]*x).sum())-1, T-15., T+15., xtol=1e-10)
            yr = D.kh(Tr, P*1e3, x, x)[0]*x
            rows.append((T-Tr, abs(y[0]-yr[0]), a/((yr[0]/x[0])/(yr[1]/x[1]))-1))
    return _stats(rows)


def keys_c3c5(n=60, seed=1):
    """The relative volatility of n-butane to isopentane in boiling five-species mixtures by four models, and
    how far Peng-Robinson's pure vapour pressures are from the reference equations."""
    import CoolProp
    import CoolProp.CoolProp as CP
    from thermo import CEOSGas, CEOSLiquid, ChemicalConstantsPackage, FlashPureVLS, FlashVL, PRMIX
    from thermo.interaction_parameters import IPDB
    pkg = PP.package('c3c5')
    sp = pkg.d['species']
    c, p = ChemicalConstantsPackage.from_IDs(sp)
    i, j = (sp.index(k) for k in KEYS['c3c5'])

    def pr(k):
        eos = dict(Tcs=c.Tcs, Pcs=c.Pcs, omegas=c.omegas, kijs=k)
        return FlashVL(c, p, liquid=CEOSLiquid(PRMIX, eos_kwargs=eos, HeatCapacityGases=p.HeatCapacityGases), gas=CEOSGas(PRMIX, eos_kwargs=eos, HeatCapacityGases=p.HeatCapacityGases))
    f1, f0 = pr(IPDB.get_ip_asymmetric_matrix('ChemSep PR', c.CASs, 'kij')), pr([[0.]*len(sp) for _ in sp])
    S = CoolProp.AbstractState('HEOS', '&'.join(NAMES[s] for s in sp))
    rng, rows = np.random.default_rng(seed), []
    for _ in range(n):
        x, P = rng.dirichlet(np.ones(len(sp))), rng.uniform(pkg.d['Plo'], pkg.d['Phi'])
        al = lambda y: (y[i]/x[i])/(y[j]/x[j])                     # noqa: E731
        S.set_mole_fractions(list(x))
        S.update(CoolProp.PQ_INPUTS, P*1e3, 0.)
        rows.append((_engine(pkg, x, P, i, j)[2], al(f1.flash(P=P*1e3, VF=0., zs=list(x)).gas.zs), al(f0.flash(P=P*1e3, VF=0., zs=list(x)).gas.zs), al(S.mole_fractions_vapor())))
    r = np.array(rows).mean(0)
    ratio = []
    for T in (340., 360., 380., 400.):                             # the two keys alone: ratio of vapour pressures, Peng-Robinson over reference
        ps = []
        for s in KEYS['c3c5']:
            ci, pi = ChemicalConstantsPackage.from_IDs([s])
            eos = dict(Tcs=ci.Tcs, Pcs=ci.Pcs, omegas=ci.omegas)
            fl = FlashPureVLS(ci, pi, gas=CEOSGas(PRMIX, eos_kwargs=eos, HeatCapacityGases=pi.HeatCapacityGases),
                              liquids=[CEOSLiquid(PRMIX, eos_kwargs=eos, HeatCapacityGases=pi.HeatCapacityGases)], solids=[])
            ps.append(fl.flash(T=T, VF=0.).P/CP.PropsSI('P', 'T', T, 'Q', 0, NAMES[s]))
        ratio.append(ps[0]/ps[1]-1)
    return dict(points=n, engine=float(r[0]), peng_robinson_chemsep=float(r[1]), peng_robinson_no_kij=float(r[2]), helmholtz_gerg2008=float(r[3]),
                engine_over_helmholtz=float(r[0]/r[3]-1), pure_pressure_ratio_pr_over_reference=float(np.max(np.abs(ratio))))


def run():
    out = {}
    for name in ('c3c5', 'bt', 'meoh', 'steam'):
        pkg = PP.package(name)
        out[name] = dict(model=pkg.d['model'], window=[pkg.d['Plo'], pkg.d['Phi']], at=AT[name], pure=pure(pkg, AT[name]))
    out['c3c5']['helmholtz'], out['bt']['helmholtz'] = helmholtz('c3c5'), helmholtz('bt')
    out['bt']['dortmund'], out['meoh']['dortmund'] = dortmund('bt'), dortmund('meoh')
    out['c3c5']['keys'] = keys_c3c5()
    (OUT/'refprops.json').write_text(json.dumps(out, indent=1))
    return out


def table(res=None):
    res = json.loads((OUT/'refprops.json').read_text()) if res is None else res
    rows = ['package species      kPa     Tsat K  latent  liquid cp  liquid volume  Z     (engine against the reference equation; in brackets the largest over the window)']
    for name, r in res.items():
        for s, d in r['pure'].items():
            a, w = d['at'], d['window']
            rows.append('%-7s %-11s %6.0f  %+5.2f (%.2f)  %+5.1f%% (%.1f)  %+5.1f%% (%.1f)  %+5.1f%% (%.1f)  %+5.1f%% (%.1f)' % (
                name, s, r['at'], a['Tsat'], w['Tsat'], 100*a['latent'], 100*w['latent'], 100*a['liquid_cp'], 100*w['liquid_cp'], 100*a['liquid_volume'], 100*w['liquid_volume'],
                100*a['Z'], 100*w['Z']))
    rows.append('mixtures: bubble T rms / max K, vapour mole fraction rms / max, relative volatility of the keys mean / rms')
    for name, r in res.items():
        for k in ('helmholtz', 'dortmund'):
            if k in r:
                m = r[k]
                rows.append('%-7s %-9s %4d points  T %.2f / %.2f  y %.4f / %.4f  volatility %+.1f%% / %.1f%%' % (
                    name, k, m['points'], m['T_rms'], m['T_max'], m['y_rms'], m['y_max'], 100*m['volatility_mean'], 100*m['volatility_rms']))
    if 'keys' in res.get('c3c5', {}):
        k = res['c3c5']['keys']
        rows.append('c3c5 n-butane over isopentane: engine %.3f, Peng-Robinson %.3f (no kij %.3f), Helmholtz %.3f; pure vapour pressure ratio, PR over reference, within %.1f%%' % (
            k['engine'], k['peng_robinson_chemsep'], k['peng_robinson_no_kij'], k['helmholtz_gerg2008'], 100*k['pure_pressure_ratio_pr_over_reference']))
    return '\n'.join(rows)


if __name__ == '__main__':
    print(table(run()))

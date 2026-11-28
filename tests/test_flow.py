"""Properties and flowsheet units (aio.sim.props, aio.sim.flow): the fitted property models, the column built of
units against its limit and its own balances, the designed columns, the accuracy guard of the engine."""
import json
import math

import pytest
import torch

from aio.sim import engine as EN, flow as FL, mech as ME, plants as PL, props as PP
from aio.sim.engine import Mech

F = torch.float64


def outputs(P, M, x=None, w=None):
    y = M.y(M.x0 if x is None else x, M.wnom if w is None else w, M.pnom)[0]
    return dict(zip([o['name'] for o in P.ys], y.tolist()))


# ---------------------------------------------------------------- properties
def test_fitted_properties_are_close_to_the_rigorous_package_on_fresh_mixtures():
    data = json.loads(PP.PATH.read_text())
    assert set(data) >= {'c3c5', 'bt', 'meoh', 'steam'}
    for name, d in data.items():
        e = d['fit']
        assert e['n'] >= 250, name
        assert e['dT_rms'] < .5 and e['dT_max'] < 2., (name, e)           # boiling temperature, K
        assert e['dy_rms'] < 2e-3 and e['dy_max'] < .015, (name, e)      # vapour mole fractions
        assert max(e['dhl_max'], e['dhv_max'], e['dlat_max']) < .02, (name, e)     # enthalpies, as a share of the heat of vaporisation


def test_bubble_pressure_is_explicit_and_consistent():
    for name in ('c3c5', 'bt', 'meoh'):
        pk = PP.package(name)
        x = [1./pk.nc]*pk.nc
        for P in (pk.d['Plo'], pk.P0, pk.d['Phi']):
            T = pk.bubble_T(x, P)
            Pb, y = pk.bubble(x, T)
            assert abs(Pb/P-1) < 1e-9 and abs(sum(y)-1) < 1e-12 and pk.d['Tlo']-1 < T < pk.d['Thi']+1
            assert pk.hv(y, T, P)-pk.hl(x, T) > 0
        lo, hi = pk.bubble_T(x, pk.d['Plo']), pk.bubble_T(x, pk.d['Phi'])
        assert hi > lo                                                # boils hotter under pressure
    pk = PP.package('meoh')                                           # NRTL: both activity coefficients above 1, each 1 where its species is pure
    g = pk.gamma([.3, .7], 350.)
    assert g[0] > 1 and g[1] > 1 and abs(pk.gamma([1., 0.], 350.)[0]-1) < 1e-12


def test_properties_against_the_model_they_were_fitted_to():
    pytest.importorskip('thermo')
    pytest.importorskip('CoolProp')
    from aio.sim import props_fit as PF
    for name in ('bt', 'meoh', 'steam'):
        spec, pk = PF.PACKAGES[name], PP.package(name)
        src, nc = PF.source(spec['species'], spec['model']), len(spec['species'])
        for x0 in (.1, .5, .9) if nc > 1 else (1.,):
            x = [x0, 1-x0][:nc]
            for P in spec['P']:
                Tr, yr, _ = src.bubble(P, x)
                T = pk.bubble_T(x, P)
                _, y = pk.bubble(x, T)
                lat = src.hv(Tr, P, yr)-src.hl(Tr, P, x)
                assert abs(T-Tr) < .15 and abs(y[0]-yr[0]) < 3e-3, (name, x0, P, T-Tr)
                assert abs((pk.hv(y, T, P)-pk.hl(x, T))/lat-1) < .01
                cp = (src.hl(Tr, P, x)-src.hl(Tr-20., P, x))/20.
                assert abs((pk.hl(x, T)-pk.hl(x, T-20.))/20./cp-1) < .02, (name, x0, P)      # the liquid's heat capacity, with the heat of mixing


def test_the_heat_of_mixing_is_consistent_and_has_the_sign_of_the_measurements():
    pk = PP.package('meoh')
    x, T = [.3, .7], 350.
    assert -900 < pk.he(x, 298.15) < -500 and pk.he(x, T) < 0 and pk.cpe(x, T) > 0      # methanol-water mixes exothermically (about -0.8 kJ/mol at 25 C), less so when hot
    assert abs(sum(a*b for a, b in zip(x, pk.he_i(x, T)))-pk.he(x, T)) < 1e-9             # the partial molar parts add up to it
    nh = lambda n: sum(n)*pk.he([v/sum(n) for v in n], T)                                # noqa: E731
    for i in range(2):
        up, dn = [v+1e-6*(j == i) for j, v in enumerate(x)], [v-1e-6*(j == i) for j, v in enumerate(x)]
        assert abs((nh(up)-nh(dn))/2e-6-pk.he_i(x, T)[i]) < 1e-4
    assert abs((pk.he(x, T+.5)-pk.he(x, T-.5))-pk.cpe(x, T)) < 1e-6
    assert PP.package('bt').he(x, T) == 0.                                                # Peng-Robinson packages: in the fitted enthalpies


# ---------------------------------------------------------------- the column of units
def test_with_constant_volatility_and_equal_latent_heats_the_column_of_units_is_skogestads_column_a():
    pk = PP.synthetic('cmo', [1.5, 1.], cpl=5., cpv=5.)
    P = FL.column('skoge', pk, 39, 20, 60., [.5, .5], 355., 100., 30., 162.3774, 's0', 's1', eff=1., design=False)
    x0, res = FL.settle(P)
    assert res < 1e-6
    M = Mech([P], n=1, dtype=F)
    names, inputs = [o['name'] for o in P.ys], [q['name'] for q in P.ws]
    iL, iV, iD = names.index('L'), names.index('Vb'), names.index('D')
    tg = torch.tensor([[162.3774, 192.3774, 30.]], dtype=F)          # column A's flows (kmol/min x 60)
    x, w, ok = M.steady(M.wnom, M.pnom, free=[0, 1, inputs.index('TF')], rows=[iL, iV, iD], targets=tg, band=torch.ones(1, 3, dtype=F),
                        start=(torch.tensor([x0], dtype=F), M.wnom, M.pnom))
    assert bool(ok.all())
    xs = FL.probe(P, [s.x[0] for s in P.units['stages']], x, w)[0]
    A = PL.column()
    MA = Mech([A], n=1, dtype=F)
    xa, _, oka = MA.steady(MA.wnom, MA.pnom)
    assert bool(oka.all())
    assert float((xs-xa[0, :41]).abs().max()) < 3e-4                 # the composition on every stage
    KA = MA.gains(xa, MA.wnom, MA.pnom)[4][0][:2, :2]
    K = M.gains(x, w, M.pnom)[4][0]
    G = K[[names.index('yD'), names.index('yB')]][:, :2]@torch.linalg.inv(K[[iL, iV]][:, :2])*60.
    assert float(((G-KA)/KA).abs().max()) < .02                      # purity gains to reflux and boilup


@pytest.fixture(scope='module', params=['btcol', 'meoh', 'debut'])
def designed(request):
    def no(*a, **k):
        raise AssertionError('the stored design point does not belong to this table: the column would be designed again')
    with pytest.MonkeyPatch.context() as mp:                            # the stored point is used: nothing is computed when the plant is built
        mp.setattr(FL, 'settle', no)
        P = PL.LARGE[request.param]()
    return request.param, P, Mech([P], n=1, dtype=F)


def test_a_designed_column_rests_on_specification_at_its_stored_point(designed):
    name, P, M = designed
    got = FL._stored()[name]
    assert len(got['x']) == P.nx and [s['x0'] for s in P.xs] == got['x'] and [q['nom'] for q in P.ws] == got['w']
    assert float((M.f(M.x0, M.wnom, M.pnom)/M.xscale).abs().max()) < 1e-6
    y, d = outputs(P, M), P.design
    assert abs(y['yD']-2) < 1e-3 and abs(y['yB']-2) < 1e-3 and abs(y['P']/d['P']-1) < 1e-4
    assert abs(y['L']/d['L']-1) < .05 and abs(y['D']/d['D']-1) < .05     # sized for the flows it runs at
    assert all(.3 < q['nom'] < .8 for q in P.ws[:3])                    # no valve near an end at the design point
    assert .25 < y['LD'] < .75 and .25 < y['LB'] < .75 and y['dP'] < P.ys[[o['name'] for o in P.ys].index('dP')]['hi']
    Phi = M.gains(M.x0, M.wnom, M.pnom)[2][0]
    assert float(torch.linalg.eigvals(Phi).abs().max()) < 1.            # stable with the level loops closed


def test_a_designed_column_closes_its_balances_and_its_hydraulics_are_physical(designed):
    name, P, M = designed
    u, pk = P.units, PP.package(dict(btcol='bt', meoh='meoh', debut='c3c5')[name])
    st, nc = u['stages'], pk.nc
    Fl, zf, hf = u['feed']
    ex = [Fl, u['D'], u['B'], u['Qr'], u['Qc'], hf, st[-1].hl, st[0].hl]+list(zf)+list(st[-1].x)+list(st[0].x)+[s.P for s in st]+[s.T for s in st]+u['V']+u['L']
    v = FL.probe(P, ex, M.x0)[0]
    Ff, D, B, Qr, Qc, hF, hD, hB = v[:8].tolist()
    z, xd, xb = v[8:8+nc], v[8+nc:8+2*nc], v[8+2*nc:8+3*nc]
    n = len(st)
    Pj, Tj = v[8+3*nc:8+3*nc+n], v[8+3*nc+n:8+3*nc+2*n]
    V, L = v[8+3*nc+2*n:8+3*nc+3*n-1], v[8+3*nc+3*n-1:]
    assert float((Ff*z-D*xd-B*xb).abs().max())/Ff < 1e-6                # every species
    assert abs(Ff*hF+Qr-Qc-D*hD-B*hB)/Qr < 1e-6                         # energy
    assert bool((Pj[:-1] > Pj[1:]).all()) and bool((Tj[:-2] > Tj[1:-1]).all())      # pressure and temperature fall from the sump to the top tray
    assert bool((V > 0).all()) and bool((L > 0).all())
    assert float(V.max()/V.min()) < 1.6                                 # no stage dry or flooded at the design point
    for j in (0, n//2, n-2):                                            # a stage is a boiling liquid: its pressure is the bubble pressure of what is on it
        xj = FL.probe(P, list(st[j].x), M.x0)[0].tolist()
        kf = [M.pnom[0, [q['name'] for q in P.ps].index('volatility')].item() if i == P.design['lk'] else 1. for i in range(nc)]
        assert abs(pk.bubble(xj, float(Tj[j]), kf)[0]/float(Pj[j])-1) < 1e-9


def test_reflux_and_steam_move_the_products_the_way_a_column_does(designed):
    name, P, M = designed
    K = M.gains(M.x0, M.wnom, M.pnom)[4][0]
    r = {o['name']: i for i, o in enumerate(P.ys)}
    assert K[r['yD'], 0] > 0 and K[r['yB'], 0] < 0                      # more reflux: purer top, less pure bottom
    assert K[r['yD'], 1] < 0 and K[r['yB'], 1] > 0                      # more steam: the reverse
    assert K[r['P'], 2] < 0                                             # cooling lowers the pressure
    w = M.wnom.clone()                                                  # steam: more boilup at once, so more pressure and more pressure drop
    w[0, 1] += .02                                                      # (later the top gets heavier and both may fall again: the steady gains have either sign)
    x = M.x0
    for _ in range(3):
        x = M.advance(x, w, M.pnom)
    d = (M.y(x, w, M.pnom)-M.y(M.x0, M.wnom, M.pnom))[0]
    assert float(d[r['P']]) > 0 and float(d[r['dP']]) > 0 and float(d[r['Vb']]) > 0


# ---------------------------------------------------------------- the engine's guard
def test_the_guard_takes_a_violent_decision_again_in_finer_steps():
    P = PL.LARGE['btcol']()
    M = Mech([P], n=3, dtype=F)
    x, w, p = M.x0.clone(), M.wnom.clone(), M.pnom.clone()
    w[1, 1], w[1, 0] = 1., .15                                          # the steam valve thrown open and the reflux nearly shut in one decision
    w[2, 2] = .05                                                       # the cooling water all but lost
    band = torch.tensor([float(o['band']) for o in P.ys], dtype=F)
    err = lambda a, b: ((M.y(a, w, p)-M.y(b, w, p)).abs()/band)[:, :7].amax(-1)      # noqa: E731
    EN.STATS = []
    try:
        plain, guarded = M.advance(x, w, p, guard=False), M.advance(x, w, p)
        est = EN.STATS[-1]
    finally:
        EN.STATS = None
    ref = M.advance(x, w, p, substeps=512, refresh=True, guard=False)
    assert float(est[0]) < 1e-3 and bool(torch.equal(plain[0], guarded[0]))       # at rest: nothing estimated, nothing redone
    assert bool((est[1:] > EN.ERR_TOL).all())
    e0, e1 = err(plain, ref), err(guarded, ref)
    assert bool((e1[1:] < .25).all()) and bool((e1[1:] < .5*e0[1:]).all()), (e0.tolist(), e1.tolist())


def test_an_episode_outside_the_plants_domain_comes_back_not_finite_and_leaves_the_others_alone():
    P = PL.LARGE['meoh']()
    M = Mech([P], n=3, dtype=F)
    x, w, p = M.x0.clone(), M.wnom.clone(), M.pnom.clone()
    good = M.sub(torch.tensor([0])).advance(x[:1], w[:1], p[:1])
    x[1] = 0.                                                           # no liquid anywhere, at 0 K
    x[2, 5] = math.nan
    out = M.advance(x, w, p)                                            # no exception from the factorisation
    assert bool(torch.allclose(out[0], good[0], rtol=0, atol=1e-10))
    assert not bool(torch.isfinite(out[2]).all())


def test_parameters_that_hold_for_an_episode_are_not_moved_by_events():
    P = PL.LARGE['meoh']()
    t = ME._meta([P], P.nw, P.ny)
    ev = {q['name']: bool(t['pev'][0, j]) for j, q in enumerate(P.ps)}
    assert ev == dict(ua_reb=True, ua_cond=True, tray_dp=False, volatility=False, cv_reflux=True)
    for q in (PL.PLANTS[k]() for k in ('qtank', 'evap', 'hex')):       # the small plants as before
        assert bool((ME._meta([q], q.nw, q.ny)['pev'][0, :q.np] > 0).all())

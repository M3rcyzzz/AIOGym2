"""The building on radiators (aio.sim.plants_bldg): the radiator, the hydraulics, the pairing, the heat balance, the
equations against a separate integration, the walls against the BOPTEST radiator case, and episodes on it."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp

from aio.sim import core as C, control as CT, mech as ME, plants_bldg as PB
from aio.sim.engine import Mech

F = torch.float64
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def plant():
    P = PB.bldg()
    return P, Mech([P], n=1, dtype=F)


def pw(d, n=PB.RAD_N, eps=.5):
    return d*(d*d+eps*eps)**(.5*(n-1))


def rhs(x, u, w, p):
    """The same building written out in numpy (per hour): x in the plant's order, u = (Tset, valve 1, valve 2),
    w = (Tout, sun, gains), p = (ach, rad, pump, ce, ci, ue)."""
    z = PB.ZONE
    ta, te, ti, tr, tb = x[0:2], x[2:4], x[4:6], x[6:12].reshape(2, 3), x[12]
    tset, tout, sol, gin = u[0], w[0], w[1], w[2]
    ach, fr, ph, ce, ci, ue = p
    M = np.array(PB.QNOM)/(PB.CP*PB.DT_DES)
    g = np.array([k/np.sqrt(.5+.5*k*k) for k in u[1:]])
    s = (M*g).sum()/M.sum()
    m = M*g*np.sqrt(PB.HEAD*ph/(1+(PB.PIPE+PB.DROOP*ph)*s*s))
    tret = (m*tr[:, -1]).sum()/m.sum()
    qb = min(max(m.sum()*PB.CP*(tset-tret)+PB.KB*(tset-tb), 0.), PB.QMAX)
    d = np.zeros(13)
    d[12] = 3600*(qb-m.sum()*PB.CP*(tb-tret))/PB.CB
    ua = PB.rad_ua()
    for i in range(2):
        A, Q = PB.AREAS[i], PB.QNOM[i]
        qel = fr*Q*ua/3*pw(tr[i]-ta[i])
        tin = np.concatenate(([tb], tr[i, :-1]))
        d[6+3*i:9+3*i] = 3600*(m[i]*PB.CP*(tin-tr[i])-qel)/(PB.RAD_C*Q/3)
        qr, qs, qg = qel.sum(), sol*z['win']*A, gin*A
        rad = PB.RAD_FRAC*qr+.5*qg+qs
        hw = (z['Uw']*ue+.335*ach*z['vol'])*A
        hae, hai, heo = z['Hae']*A, z['Hai']*A, z['Heo']*ue*A
        d[i] = 3600*(hae*(te[i]-ta[i])+hai*(ti[i]-ta[i])+hw*(tout-ta[i])+PB.H12*(ta[1-i]-ta[i])+(1-PB.RAD_FRAC)*qr+.5*qg)/(z['Ca']*A)
        d[2+i] = 3600*(hae*(ta[i]-te[i])+heo*(tout-te[i])+z['share_e']*rad)/(z['Ce']*ce*A)
        d[4+i] = 3600*(hai*(ta[i]-ti[i])+(1-z['share_e'])*rad)/(z['Ci']*ci*A)
    return d, qb, tret


def test_a_radiator_gives_its_nominal_output_at_its_nominal_temperatures():
    ts, tr, ta = PB.NOM
    ua, mcp = PB.rad_ua(), 1./(ts-tr)
    t = np.full(3, .5*(ts+tr))
    for _ in range(4000):                                         # the three elements at rest, by relaxation
        tin = np.concatenate(([ts], t[:-1]))
        t = t+.02*(mcp*(tin-t)-ua/3*(t-ta)**PB.RAD_N)/mcp
    out = (ua/3*(t-ta)**PB.RAD_N).sum()
    assert abs(out-1.) < 1e-6 and abs(t[-1]-tr) < 1e-4            # the nominal output, and the nominal return temperature with it
    assert abs(pw(30.)/pw(15.)-2**PB.RAD_N) < 2e-3                # EN 442: the output goes with the excess temperature to the power n
    assert pw(0.) == 0. and pw(-3.) == -pw(3.)                    # smooth through a cold radiator
    assert abs(PB.RAD_C-(5.8e-6*1000*PB.CP+.0263*500)) < 1e-9


def test_opening_one_valve_takes_flow_from_the_other():
    assert np.allclose(PB.flows((1., 1.)), 1.)                    # the design flows with both valves open and the nominal pump
    a, b = PB.flows((.5, .5)), PB.flows((1., .5))
    assert b[1] < a[1] and b[0] > a[0]
    assert PB.flows((1., .03))[0] > 1.1                           # alone on the pump a branch passes more than its design flow
    assert PB.flows((1., 1.), .85)[0] < .95
    lo = PB.flows((.03, .03))
    assert 0 < lo[0] < .06


def test_the_spread_pairs_where_the_return_temperature_does_not(plant):
    """Relative gains at rest: every loop of rooms by valves and spread by supply temperature is positive; with the
    return temperature in the place of the spread the supply temperature moves it one way with the valves where
    they are and the other way with the rooms held, so its relative gain is negative."""
    P, M = plant

    def gains(w0, rows):
        x0 = M.steady(w0, M.pnom)[0]
        y0 = M.y(x0, w0, M.pnom)[0]
        G = torch.zeros(3, 3, dtype=F)
        for j in range(3):
            w = w0.clone()
            d = 1e-3*(P.ws[j]['hi']-P.ws[j]['lo'])
            w[0, j] += d
            G[:, j] = (M.y(M.steady(w, M.pnom)[0], w, M.pnom)[0]-y0)[rows]/d
        return G.numpy()
    for tset, k1, k2 in ((52., .2, .2), (60., .15, .1), (45., .5, .4), (70., .1, .1)):
        w = M.wnom.clone()
        w[0, 0], w[0, 1], w[0, 2] = tset, k1, k2
        G = gains(w, [0, 1, 2])                                   # rooms and spread by (supply temperature, valve 1, valve 2)
        L = G*np.linalg.inv(G).T
        assert L[0, 1] > .3 and L[1, 2] > .3 and L[2, 0] > .1, L
        G = gains(w, [0, 1, 3])                                   # the return temperature in the place of the spread
        L = G*np.linalg.inv(G).T
        assert G[2, 0] > 0 and L[2, 0] < 0, L


def test_the_heat_balance_closes_at_rest(plant):
    P, M = plant
    x = M.steady(M.wnom, M.pnom)[0]
    xs, w, p = x[0].numpy(), M.wnom[0].numpy(), M.pnom[0].numpy()
    d, qb, tret = rhs(xs, w[:3], w[3:], p)
    assert np.abs(d).max() < 1e-6                                 # the engine's steady point is one of the numpy building
    z, ue, ach = PB.ZONE, p[5], p[0]
    A = np.array(PB.AREAS)
    loss = (((z['Uw']*ue+.335*ach*z['vol'])*(xs[0:2]-w[3])+z['Heo']*ue*(xs[2:4]-w[3]))*A).sum()
    gains = ((w[4]*z['win']+w[5])*A).sum()
    assert abs(qb+gains-loss) < 1e-6*loss                         # boiler, sun and occupants in; windows, air changes and walls out
    y = M.y(x, M.wnom, M.pnom)[0]
    assert abs(float(y[2])-(xs[12]-tret)) < 1e-9 and abs(float(y[3])-tret) < 1e-9 and abs(float(y[4])-qb/PB.QMAX) < 1e-9
    assert 19 < float(y[0]) < 23.5 and 19 < float(y[1]) < 23.5 and 9 < float(y[2]) < 22       # the nominal point is inside the targets
    assert float(y[3]) < PB.TRET_HI-10 and float(y[4]) < .5                                    # and well inside the limits


def test_the_table_is_the_building_written_out(plant):
    """The engine at the plant's own substeps against a stiff solver on the numpy building, the inputs moving."""
    P, M = plant
    g = torch.Generator().manual_seed(0)
    x = M.steady(M.wnom, M.pnom)[0]
    xr, w, p = x[0].numpy().copy(), M.wnom.clone(), M.pnom.clone()
    p[0, :3] = torch.tensor([.8, .9, .95])
    lo, hi = np.array([q['lo'] for q in P.ws]), np.array([q['hi'] for q in P.ws])
    worst = np.zeros(5)
    for k in range(96):                                           # a day
        if k % 8 == 0:                                            # every two hours the inputs move: valves and setpoint anywhere, the loads over half their range
            u = torch.rand(6, generator=g, dtype=F).numpy()
            u[3:] = .25+.5*u[3:]
            w = torch.tensor(lo+u*(hi-lo))[None]
        x = M.advance(x, w, p)
        wn, pn = w[0].numpy(), p[0].numpy()
        xr = solve_ivp(lambda t, s: rhs(s, wn[:3], wn[3:], pn)[0], (0., P.dt/3600.), xr, method='Radau', rtol=1e-9, atol=1e-9).y[:, -1]
        _, qb, tret = rhs(xr, wn[:3], wn[3:], pn)
        y = M.y(x, w, p)[0].numpy()
        worst = np.maximum(worst, np.abs(y[:5]-np.array([xr[0], xr[1], xr[12]-tret, tret, qb/PB.QMAX])))
    band = np.array([o['band'] for o in P.ys[:4]])
    assert (worst[:4] < .25*band).all(), worst[:4]/band           # room temperatures, spread and return temperature within a quarter of a band all day
    assert worst[4] < .05                                         # the boiler load moves in a step with the setpoint: within 5 % of the capacity


def test_the_walls_answer_as_the_boptest_radiator_case(plant):
    """The nominal walls were fitted to it: supply temperature to room temperature over ten days, the shape
    (BESTEST 900 with a radiator, an IDEAS model, operative temperature; here the air of room 1)."""
    P, M = plant
    ref = json.loads((ROOT/'reports/validation/boptest.json').read_text())['steps']['bestest_hydronic']
    assert ref['dt'] == P.dt
    R = np.array([r[0][0] for r in ref['response']])
    R = R/R[-1]
    x = M.steady(M.wnom, M.pnom)[0]
    S = M.responses(x, M.wnom, M.pnom, 960)[0, :, 0, 0].numpy()
    r = (S/S[-1])[::8]
    assert len(r) == len(R)
    assert np.sqrt(((r-R)**2).mean()) < .02 and np.abs(r-R).max() < .05, (np.sqrt(((r-R)**2).mean()), np.abs(r-R).max())
    # a fast part and a tail of days: a fifth of the answer in two hours, an eighth of it still to come after sixty
    assert .2 < r[1] < .35 and .8 < r[30] < .92


def test_the_rooms_answer_where_the_prior_does_not_reach(plant):
    """The room loops over the response window of the prior (240 decisions): next to no dead time against the time to
    63 %, and that time many times the time to 20 %, as the BOPTEST radiator loop (ratio 0.014, spread 33); light and
    heavy buildings on both sides of the nominal one."""
    from aio.sim.validate import realdata as RD
    P, M = plant
    names = [q['name'] for q in P.ps]
    seen = []
    for ce, ci in ((1., 1.), (.6, .6), (1.5, 1.5)):
        p = M.pnom.clone()
        p[0, names.index('ce')], p[0, names.index('ci')] = ce, ci
        x = M.steady(M.wnom, p)[0]
        S = M.responses(x, M.wnom, p, 240)[0]
        for i, j in ((0, 0), (0, 1), (1, 2)):                     # setpoint and own valve to a room
            f = RD.loop(S[:, i, j].numpy())
            assert f['ratio'] < .03 and f['spread'] > 12 and f['theta'] <= 2, f
            seen.append(f['tau'])
        f = RD.loop(S[:, 3, 0].numpy())                           # the return temperature answers in a few decisions
        assert f['tau'] <= 4
    assert seen[3] < seen[0] < seen[6] and 35 < min(seen) and max(seen) < 80


def test_episodes_run_on_the_building():
    prog = C.build(6, 3, mech=1., plants=ME.pool(['bldg']))
    assert bool((prog.c['mvm'].sum(-1) == 3).all()) and bool((prog.c['trk'].sum(-1) == 3).all())
    assert int((prog.kinds[:, 1:] >= 0).sum()) >= 3*prog.n         # events are found on it: a slow plant is not thrown out by the generator's checks
    score = {}
    for name, ctl in (('hold', CT.Hold()), ('pi', CT.PI()), ('teacher', CT.Teacher('late'))):
        r = CT.rollout(prog, ctl)
        prog.set_teacher()
        assert not bool(r['failed'].any())
        score[name] = float(r['score'].mean())
    assert score['teacher'] < .6*score['pi'] < .6*score['hold'], score

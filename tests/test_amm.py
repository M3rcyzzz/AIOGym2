"""The ammonia converter (aio.sim.plants_amm) against what its two papers print, an independent numpy model of the
published equations, the record of the plant it describes, and in episodes. Morud and Skogestad, AIChE J. 44(4),
1998 (M); Straus and Skogestad, "Economic NMPC for heat-integrated chemical reactors", 2017 (S)."""
import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp

from aio.sim import core as C, control as CT, mech as ME, plants as PL, plants_amm as PA
from aio.sim.engine import Mech

F = torch.float64
MC = np.array([6.69, 9.63, 15.2])*2200.          # kg of catalyst a bed (M: bed volumes, 2200 kg/m3)
N = 10
CPG, CPC, DH, UA = 3500., 1100., 2.7e6, 536.*283.
U_NOM = np.array([58., 35., 32.])/252.          # M: 58, 35, 32 t/h of 252
U_OPT = np.array([.2124, .3079, .2958])         # S, Table III


def rate(T, w, p, act=1.):
    """kg NH3/(kg catalyst s). S, Eqs. 1-3: Temkin-Pyzhev in kmol N2/(m3 catalyst h) with pressures in bar, times
    4.75 and 2*17/2200; the gas is hydrogen and nitrogen three to one and w (mass) ammonia."""
    TK = T+273.15
    k1, k2 = 1.79e4*np.exp(-87090./(8.314*TK)), 2.57e16*np.exp(-198464./(8.314*TK))
    ya = w/(2-w)
    yn = (1-ya)/4
    pa, pn, ph = ya*p, yn*p, 3*yn*p
    return act*4.75*(k1*pn*ph**1.5/pa-k2*pa/ph**1.5)*2*17./2200./3600.


def preheat(T0, mh, m, tf, ua=1.):
    ntu, c = UA*ua/(mh*CPG), mh/m
    e = np.exp(-ntu*(1-c))
    return tf+(1-e)/(1-c*e)*(T0-tf)


def beds(u, m, tf, win, Ti):
    """Per bed: flow through it, and what the stream ahead of it is mixed with (share of the quench, ...)."""
    q = np.asarray(u)*m
    flow = m-q.sum()
    out = []
    for b in range(3):
        out.append((flow, q[b]))
        flow = flow+q[b]
    return out


def residual(x, u, load, par):
    """The plant's right-hand side: 30 temperature derivatives and those of the ammonia of the 30 cells (the cell's
    balance over the gas it holds, plants_amm.GAS: not in the two papers), at any temperatures and mass fractions
    x. load: p, feed, Tf, win; par: act, ua, mass."""
    T, W = x[:3*N], x[3*N:]
    p, m, tf, win = load
    act, ua, mass = par
    tin, wprev = preheat(T[-1], max(m*(1-sum(u)), .01*m), m, tf, ua), win
    dT, g = np.zeros(3*N), np.zeros(3*N)
    for b, (flow, q) in enumerate(beds(u, m, tf, win, None)):
        tin, wprev, flow = (flow*tin+q*tf)/(flow+q), (flow*wprev+q*win)/(flow+q), flow+q
        mc = MC[b]/N
        for j in range(N):
            i = b*N+j
            r = rate(T[i], W[i], p, act)
            dT[i] = (CPG*flow*(tin-T[i])+mc*r*DH)/(mc*CPC*mass)
            g[i] = (flow*(wprev-W[i])+mc*r)/(.33*50.*MC[b]/2200./N)
            tin, wprev = T[i], W[i]
    return np.concatenate((dT, g))


def model(T, u, p=200., m=70., tf=250., win=.08, act=1., ua=1., Ti=None, upstream=False):
    """The published model (the gas holds nothing): temperature derivatives with the ammonia of every cell at its balance; the preheated stream's temperature;
    the mass fractions; the bed inlet temperatures. Ti given: the loop cut at the exchanger. upstream: the rate
    of a cell at the ammonia entering it (the reading that gives M's numbers), else at the ammonia in the cell
    (S's statement of the cells: the plant of the engine)."""
    Ti = preheat(T[-1], m*(1-sum(u)), m, tf, ua) if Ti is None else Ti
    tin, wprev = Ti, win
    dT, W, tins = np.zeros(3*N), np.zeros(3*N), np.zeros(3)
    for b, (flow, q) in enumerate(beds(u, m, tf, win, None)):
        tin, wprev, flow = (flow*tin+q*tf)/(flow+q), (flow*wprev+q*win)/(flow+q), flow+q
        tins[b] = tin
        mc = MC[b]/N
        for j in range(N):
            i = b*N+j
            if upstream:
                r = rate(T[i], wprev, p, act)
                w = wprev+mc*r/flow
            else:
                w = wprev
                for _ in range(80):                                 # Newton; the balance falls in w
                    g = flow*(wprev-w)+mc*rate(T[i], w, p, act)
                    dg = -flow+mc*(rate(T[i], w+1e-7, p, act)-rate(T[i], w-1e-7, p, act))/2e-7
                    wn = w-g/dg
                    wn = .5*w if wn <= 0 else (.5*(w+1) if wn >= 1 else wn)
                    done, w = abs(wn-w) < 1e-14, wn
                    if done:
                        break
                r = rate(T[i], w, p, act)
            dT[i] = (CPG*flow*(tin-T[i])+mc*r*DH)/(mc*CPC)
            W[i] = w
            tin, wprev = T[i], w
    return dT, Ti, W, tins


def jacobian(T, u, h=1e-5, **kw):
    J = np.zeros((3*N, 3*N))
    for i in range(3*N):
        a, b = T.copy(), T.copy()
        a[i] += h
        b[i] -= h
        J[:, i] = (model(a, u, **kw)[0]-model(b, u, **kw)[0])/(2*h)
    return J


def steady(u, T=None, **kw):
    """The ignited steady state: from a hot profile at the published point, else from the one given (Newton: a steady
    state the unit would cycle around is found like any other)."""
    if T is None:
        T = solve_ivp(lambda t, z: model(z, u, **kw)[0], (0., 2e4), np.linspace(420., 480., 3*N), method='LSODA', rtol=1e-8, atol=1e-8).y[:, -1]
    for _ in range(40):
        f = model(T, u, **kw)[0]
        if np.abs(f).max() < 1e-11:
            break
        T = T-np.linalg.solve(jacobian(T, u, **kw), f)
    assert np.abs(model(T, u, **kw)[0]).max() < 1e-9
    return T


def rightmost(T, u, **kw):
    ev = np.linalg.eigvals(jacobian(T, u, **kw))
    return ev[np.argmax(ev.real+1e-9*np.abs(ev.imag))]


@pytest.fixture(scope='module')
def points():
    """Steady states down the pressure, both readings: {upstream: {p: T}}."""
    out = {}
    for up in (True, False):
        T, out[up] = None, {}
        for p in (200., 190., 180., 176., 172., 170.):
            T = out[up][p] = steady(U_NOM, T, p=p, upstream=up)
    return out


def test_the_exchanger_and_the_temperature_wave_are_the_papers():
    mh = 127/3.6
    assert abs(UA/(mh*CPG)-1.23) < .005                          # M: NTU 1.23
    assert abs((preheat(1., mh, 70., 0.))-.629) < .0006          # M: efficiency 0.629
    flows = np.cumsum([127., 58., 35., 32.])[1:]/3.6
    speed = flows*CPG/(MC*CPC)                                   # bed lengths a second
    assert np.abs(speed-np.array([.0111, .0092, .0067])).max() < 6e-5      # M: migration velocities
    assert abs((1/speed).sum()-348.) < 1.5                       # M: 348 s through the three beds (from its rounded speeds)
    assert np.abs(speed/(2*N)-np.array([5.6e-4, 4.6e-4, 3.3e-4])).max() < 6e-6    # M: the dispersion ten upwind cells stand for
    assert abs(PA.effectiveness(mh, 70.)-.629) < .0006 and tuple(PA.VOL) == (6.69, 9.63, 15.2)


def test_the_ammonia_made_is_the_later_papers_table(points):
    """S, Table III: 16.2147 kg/s at the published split, 18.1797 at its best one (12 % more): 16.19 and 18.17 here."""
    T = points[False][200.]
    _, Ti, W, tins = model(T, U_NOM)
    made, best = 70.*(W[-1]-.08), 70.*(model(steady(U_OPT), U_OPT)[2][-1]-.08)
    assert abs(made/16.2147-1) < 2e-3 and abs(best/18.1797-1) < 1e-3 and abs(best/made-18.1797/16.2147) < 2e-3
    assert abs(Ti-PA.NOM['Ti']) < .01 and np.abs(tins-np.array(PA.NOM['Tin'])).max() < .01 and np.abs(T[[9, 19, 29]]-np.array(PA.NOM['Tout'])).max() < .01
    assert np.abs(W[[9, 19, 29]]-np.array(PA.NOM['w'])).max() < 1e-4 and abs(made-PA.NOM['made']) < 1e-3
    assert np.abs(T-np.array(PA.T_NOM)).max() < .06 and np.abs(W-np.array(PA.ammonia(PA.T_NOM))).max() < 3e-4


def test_the_instability_is_the_first_papers(points):
    """M prints the pair of eigenvalues furthest right at 200 and at 170 bar, the pressure where it crosses (172 bar),
    the steady gain of the outlet to the preheated stream with the loop cut (0.218), the exchanger gain that
    brings the loop to its limit (0.629 at 172 bar, 0.591 at 170), where the poles are with the loop cut (about
    -0.05 to -0.1 1/s), the period of the cycle (about 420 s). The rate of a cell at the ammonia entering it gives
    them; with the cells as S states them (the plant) the crossing is 4 bar higher."""
    up = points[True]
    e200, e170 = rightmost(up[200.], U_NOM, upstream=True), rightmost(up[170.], U_NOM, p=170., upstream=True)
    assert abs(e200.real+.0017) < 1.5e-4 and abs(abs(e200.imag)-.0183) < 1.5e-4
    assert abs(e170.real-.0002) < 1.5e-4 and abs(abs(e170.imag)-.0148) < 1.5e-4
    e172 = rightmost(up[172.], U_NOM, p=172., upstream=True)
    assert -1.5e-4 < e172.real < 0 < e170.real and abs(2*np.pi/abs(e172.imag)-420.) < 8.      # the crossing: 171 bar, 419 s
    cut = {}
    for p in (172., 170.):
        T, kw = up[p], dict(p=p, upstream=True)
        Ti = model(T, U_NOM, **kw)[1]
        g0 = (steady(U_NOM, T, Ti=Ti+.5, **kw)[-1]-T[-1])/.5
        J = jacobian(T, U_NOM, Ti=Ti, **kw)
        b = (model(T, U_NOM, Ti=Ti+1e-5, **kw)[0]-model(T, U_NOM, Ti=Ti-1e-5, **kw)[0])/2e-5
        ws = np.linspace(.0005, .03, 1476)
        g = np.array([np.linalg.solve(1j*w*np.eye(3*N)-J, b)[-1] for w in ws])
        i = np.argmin(np.abs(np.unwrap(np.angle(g))+2*np.pi))     # the phase back at a full turn: the loop closes on itself
        cut[p] = g0, 1/abs(g[i]), np.linalg.eigvals(J).real
    assert cut[172.][0] < .218 < cut[170.][0] and cut[170.][0]-cut[172.][0] < .02
    assert abs(cut[172.][1]/.629-1) < .03 and abs(cut[170.][1]/.591-1) < .03
    assert -.105 < cut[172.][2].min() and cut[172.][2].max() < -.05
    cell = points[False]
    lam = {p: rightmost(cell[p], U_NOM, p=p) for p in (180., 176., 172.)}
    assert lam[180.].real < lam[176.].real < 0 < lam[172.].real and abs(2*np.pi/abs(lam[176.].imag)-420.) < 8.     # the plant: 175 bar, 417 s


def test_why_the_plant_is_not_the_first_papers_program():
    """With the rate at the ammonia entering a cell, a cell can convert past equilibrium: at S's best split the
    program's reading has a steady state with a cell at 600 C, hotter than the gas can get by reacting. With the
    cells as stated the same start ends on the one profile, under 530 C."""
    hot = np.linspace(420., 480., 3*N)
    wrong, right = steady(U_OPT, upstream=True), steady(U_OPT)
    assert wrong.max() > 590. and right.max() < 530.
    assert np.abs(model(right, U_OPT)[0]).max() < 1e-9 and hot.max() < 500.


def test_a_hotter_inlet_first_cools_the_outlet(points):
    """M, Fig. 11: with the loop cut, a step of the preheated stream's temperature first lowers the outlet (the gas
    arrives depleted before the temperature wave does) and then raises it by the steady gain."""
    T = points[False][200.]
    Ti = model(T, U_NOM)[1]
    sol = solve_ivp(lambda t, z: model(z, U_NOM, Ti=Ti+1.)[0], (0., 900.), T, method='LSODA', rtol=1e-9, atol=1e-9, t_eval=np.arange(0., 901., 20.))
    y = sol.y[-1]-T[-1]
    assert y.min() < -.4 and 150. < sol.t[y.argmin()] < 348. and .18 < y[-1] < .22


def test_it_cycles_as_the_plant_did(points):
    """The plant (1989, M's Fig. 1 and abstract): the pressure fell, the temperatures swung over about 200 C (between
    about 300 and 500 C) every six minutes until it was restored. The model at 170 bar: the outlet between 356 and
    539 C every 6.8 minutes (the pressure the plant fell to is not given; M's own simulation: about 7 minutes)."""
    T = points[False][200.]
    sol = solve_ivp(lambda t, z: model(z, U_NOM, p=170.)[0], (0., 12600.), T, method='LSODA', rtol=1e-7, atol=1e-7, t_eval=np.arange(0., 12601., 5.))
    y, t = sol.y[-1][sol.t > 9000.], sol.t[sol.t > 9000.]
    cross = t[1:][(y[1:] > y.mean()) & (y[:-1] <= y.mean())]
    assert 160. < y.max()-y.min() < 220. and 380. < np.diff(cross).mean() < 440.
    assert 340. < y.min() < 375. and 520. < y.max() < 545.
    back = solve_ivp(lambda t, z: model(z, U_NOM)[0], (0., 7200.), sol.y[:, -1], method='LSODA', rtol=1e-7, atol=1e-7).y[:, -1]
    assert np.abs(back-T).max() < 1.                             # the pressure restored: it lines out where it was


@pytest.fixture(scope='module')
def plant():
    P = PA.amm()
    M = Mech([P], n=1, dtype=F)
    x, w, ok = M.steady(M.wnom, M.pnom)
    assert bool(ok)
    return P, M, x, w


def test_the_table_is_the_equations(plant):
    P, M = plant[:2]
    g = torch.Generator().manual_seed(3)
    for _ in range(20):
        u = torch.rand(10, generator=g, dtype=F)
        x = torch.cat((320.+230*torch.rand(30, generator=g, dtype=F), .05+.3*torch.rand(30, generator=g, dtype=F)))[None]
        w = torch.stack((.08+.28*u[0], .04+.28*u[1], .04+.26*u[2], 176.+39*u[3], 60.+18*u[4], 240.+22*u[5], .065+.03*u[6]))[None]
        p = torch.stack((.75+.35*u[7], .85+.25*u[8], .8+.45*u[9]))[None]
        ref = residual(x[0].numpy(), w[0, :3].numpy(), w[0, 3:].numpy(), p[0].numpy())
        assert np.abs(M.f(x, w, p)[0].numpy()-ref).max() < 1e-9*max(1., np.abs(ref).max())


def test_the_engine_settles_where_the_model_does(plant, points):
    P, M, x, w = plant
    T = points[False][200.]
    _, Ti, W, tins = model(T, U_NOM)
    assert np.abs(x[0, :30].numpy()-T).max() < 1e-6 and np.abs(x[0, 30:].numpy()-W).max() < 1e-9
    y = M.y(x, w, M.pnom)[0].numpy()
    assert np.abs(y[:3]-tins).max() < 1e-6 and np.abs(y[3:6]-T[[9, 19, 29]]).max() < 1e-6 and abs(y[6]-W[-1]) < 1e-9
    w2 = w.clone()
    w2[0, :3] = torch.tensor(U_OPT, dtype=F)                      # S's best split, from the published point
    x2, _, ok = M.steady(w2, M.pnom, start=(x, w, M.pnom))
    assert bool(ok) and np.abs(x2[0, :30].numpy()-steady(U_OPT)).max() < 1e-6 and abs(70.*(float(x2[0, -1])-.08)/18.1797-1) < 1e-3


def test_the_plants_own_steps_follow_the_fine_integration(plant, points):
    """A decision of 30 s against LSODA on the numpy right-hand side, through steps of the three quenches, the pressure,
    the feed and its temperature that swing the bed temperatures by up to 90 C: from the reference's own state every
    decision, and left alone for all 80 (the unit rings, so a difference in phase adds up)."""
    P, M, x, w = plant
    x, w = x.clone(), w.clone()
    xr, par = x[0].numpy().copy(), M.pnom[0].numpy()
    worst, local = np.zeros(6), np.zeros(6)
    for k in range(80):
        for at, i, to in ((2, 0, U_NOM[0]+.03), (14, 3, 186.), (26, 1, U_NOM[1]-.03), (38, 5, 244.), (50, 4, 75.), (62, 2, U_NOM[2]+.04)):
            if k == at:
                w[0, i] = to
        u, load = w[0, :3].numpy().copy(), w[0, 3:].numpy().copy()
        x, xl = M.advance(x, w, M.pnom), M.advance(torch.tensor(xr)[None], w, M.pnom)
        xr = solve_ivp(lambda t, z: residual(z, u, load, par), (0., 30.), xr, method='LSODA', rtol=1e-9, atol=1e-11).y[:, -1]
        ref = np.concatenate((model(xr[:30], u, p=load[0], m=load[1], tf=load[2], win=load[3])[3], xr[[9, 19, 29]]))
        worst, local = np.maximum(worst, np.abs(M.y(x, w, M.pnom)[0, :6].numpy()-ref)), np.maximum(local, np.abs(M.y(xl, w, M.pnom)[0, :6].numpy()-ref))
    assert local.max() < .12 and worst.max() < 1.6, (local, worst)  # 0.03 and 0.6 of a band (2 C)


def test_the_engine_loses_stability_where_the_model_does(plant, points):
    """The plant's linearisation with the ammonia at its balance has the published model's eigenvalues; the gas it
    holds moves the pair by less than 2e-4 1/s and the pressure of instability by half a bar; its map over a
    decision carries the pair; left alone at 170 bar it cycles as the model does."""
    P, M, x, w = plant
    rho = {}
    for p in (200., 180., 176., 172.):
        w2 = w.clone()
        w2[0, 3] = p
        xs, _, ok = M.steady(w2, M.pnom, start=(x, w, M.pnom))
        assert bool(ok) and np.abs(xs[0, :30].numpy()-points[False][p]).max() < 1e-5
        A = M.linearise(xs, w2, M.pnom)[2][0].numpy()
        ev = np.linalg.eigvals(A[:30, :30]-A[:30, 30:]@np.linalg.solve(A[30:, 30:], A[30:, :30]))
        lam = rightmost(points[False][p], U_NOM, p=p)
        assert abs(ev[np.argmax(ev.real+1e-9*np.abs(ev.imag))]-lam) < 1e-6
        ev = np.linalg.eigvals(A)
        own = ev[np.argmax(np.where(np.abs(ev.imag) > 1e-6, ev.real, -1.)+1e-9*np.abs(ev.imag))]
        assert abs(own-lam) < 2e-4
        ev = torch.linalg.eigvals(M.gains(xs, w2, M.pnom)[2][0])
        ev = ev[ev.abs().argmax()]
        rho[p] = float(ev.abs())
        assert abs(np.log(rho[p])/30.-own.real) < 6e-5 and abs(abs(float(torch.angle(ev)))/30.-abs(own.imag)) < 1.5e-4
    assert rho[200.] < rho[180.] < rho[176.] < 1. < rho[172.]
    w2 = w.clone()
    w2[0, 3] = 170.
    y = []
    for k in range(360):
        x = M.advance(x, w2, M.pnom)
        y.append(float(x[0, 29]))
    y = np.array(y[240:])
    cross = np.nonzero((y[1:] > y.mean()) & (y[:-1] <= y.mean()))[0]
    assert 160. < y.max()-y.min() < 220. and 380. < 30.*np.diff(cross).mean() < 440.


def test_it_is_in_the_default_pool():
    assert 'amm' in PL.NEW and 'amm' in PL.PLANTS and 'amm' not in PL.LARGE and 'amm' not in PL.FIRST


@pytest.fixture(scope='module')
def prog():
    torch.manual_seed(0)
    return C.build(6, 5, mech=1., plants=ME.pool(['amm']))


def test_the_shell_rests_only_where_the_unit_does(prog):
    """Every operating point of an episode is one the unit stays at when left alone: the generator turns down the
    pressures (and feed temperatures, activities) at which it cycles."""
    g = prog.mech.groups[0]
    a, P, xs, ws = prog.mech.parts[0]
    for s in range(xs.shape[1]):
        Phi = g.M.gains(xs[:, s], ws[:, s], P[:, s])[2]
        assert float(torch.linalg.eigvals(Phi).abs().amax(-1).max()) < 1.
    assert float(ws[:, :, 3].min()) > 175. and float(prog.c['T'].max()) < 60


def test_episodes_on_the_converter(prog):
    hold = CT.rollout(prog, CT.Hold())
    pi = CT.rollout(prog, CT.PI())
    tch = CT.rollout(prog, CT.Teacher('late'))
    prog.set_teacher()
    assert not bool(hold['failed'].any() | pi['failed'].any() | tch['failed'].any())
    h, f, t = (float(r['score'].mean()) for r in (hold, pi, tch))
    print('hold %.0f formula PI %.0f teacher %.0f' % (h, f, t))
    assert t < .5*f and f < h and bool((tch['score'] <= hold['score']).all())

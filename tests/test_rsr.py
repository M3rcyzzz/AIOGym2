"""The reactor, column and recycle (aio.sim.plants_rsr): the steady state against the paper's Table 3, its file of the
largest feed and what its text says; the equations against a separate integration; the snowball; the rows that
follow the reactor's holdup in the shell; episodes on it.

The paper: Larsson, Govatsmark, Skogestad, Yu, Ind. Eng. Chem. Res. 42 (2003) 1225. It is a steady-state study:
it prints no holdups and no dynamics, so the dynamic checks here are of the engine against the same equations."""
import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares, minimize_scalar

from aio.sim import core as C, control as CT, mech as ME, plants as PL, plants_rsr as PR
from aio.sim.engine import Mech

F = torch.float64
K, X0, AL, NT, NF = .341, .9, 2., 22, 13
_WARM = {}


def column(V, D, feed, z):
    """Mole fractions on the stages (the reboiler first, the total condenser last) at boilup V, distillate D and a liquid feed of z on
    stage 13 from the bottom: the stage balances at constant molar flows, solved as they stand."""
    B, L = feed-D, V-D
    st = np.arange(1, NT+1)
    lin = np.where(st < NF, L+feed, L)*(st < NT)
    lout = np.where(st == 1, B, np.where(st <= NF, L+feed, L))
    vin, vout = V*(st > 1), V*(st < NT)

    def res(x):
        y = AL*x/(1+(AL-1)*x)
        r = lin*np.append(x[1:], 0.)-lout*x+vin*np.append(0., y[:-1])-vout*y+feed*z*(st == NF)
        r[-1] = V*y[-2]-(L+D)*x[-1]
        return r/feed
    for x0 in (_WARM.get(0), np.linspace(.02, .9, NT), np.linspace(.001, .999, NT), np.full(NT, z)):
        if x0 is None:
            continue
        s = least_squares(res, x0, bounds=(1e-12, 1-1e-12), xtol=1e-15, ftol=1e-15, gtol=1e-15)
        if np.abs(s.fun).max() < 1e-9:
            break
    assert np.abs(s.fun).max() < 1e-9
    _WARM[0] = s.x
    return s.x


def boilup(D, f0, mr, xb):
    """The boilup that leaves xb in the bottoms at the recycle D (the reactor follows from the balance of the whole plant)."""
    xr = f0*(X0-xb)/(K*mr)
    if not xr < ((f0+D)*xr-f0*xb)/D < 1:
        return 1e9
    f = lambda V: column(V, D, f0+D, xr)[0]-xb                   # noqa: E731
    lo, hi = D*1.02, D*1.02+200.
    while f(hi) > 0:
        hi += 400.
        if hi > 2e4:
            return 1e9
    return brentq(f, lo, hi, xtol=1e-9) if f(lo) > 0 else lo


def least_boilup(f0, mr, xb):
    """The paper's case I: the recycle that takes the least boilup."""
    r = minimize_scalar(lambda D: boilup(D, f0, mr, xb), bounds=(f0*.3, f0*4.), method='bounded', options=dict(xatol=1e-7))
    xr = f0*(X0-xb)/(K*mr)
    return dict(F=f0+r.x, V=r.fun, L=r.fun-r.x, D=r.x, xD=((f0+r.x)*xr-f0*xb)/r.x, xr=xr)


def largest_feed(vmax, mr, xb):
    """The paper's case II: the feed at which the least boilup is all the reboiler has."""
    return brentq(lambda f0: least_boilup(f0, mr, xb)['V']-vmax, 20., 800., xtol=1e-6)


def rhs(x, u, w, p):
    """The plant in numpy: x = (Mr, xr, x1..x22, M1..M22), u = (F, L, V) kmol/h, w = (F0, x0), p = (k, alpha, holdup factor)."""
    mr, xr, xs, m = x[0], x[1], x[2:24], x[24:46]
    (f, l, v), (f0, x0), (k, al, hold) = u, w, p
    y = al*xs/(1+(al-1)*xs)
    d = max(586.72+30.*(m[21]/hold-150.), 0.)                    # the two drums: proportional level controllers
    b = max(460.+30.*(m[0]/hold-150.), 0.)
    st = np.arange(1, 23)
    down = np.where(st <= 13, 791.91+1046.72, 791.91)+(m/hold-8.)/(.063/60)     # liquid off a tray: linearised hydraulics
    down[0], down[21] = b, l+d
    lin = np.append(down[1:], 0.)
    lin[20] = l                                                  # the reflux onto the top tray
    vin, vout, feed = v*(st > 1), v*(st < 22), f*(st == 13)
    dm = lin-down+vin-vout+feed
    dmx = lin*np.append(xs[1:], 0.)-down*xs+vin*np.append(0., y[:-1])-vout*y+feed*xr
    return np.concatenate(([f0+d-f, (f0*(x0-xr)+d*(xs[21]-xr))/max(mr, 1.)-k*xr], (dmx-xs*dm)/np.maximum(m, 1e-2), dm))


BAND = torch.tensor([[28., .005, .04]], dtype=F)


def settle(M, mr, xd, xb, w=None, start=None):
    """The plant's steady state with the holdup, the distillate and the bottoms at the values given: x, w."""
    tg = torch.tensor([[mr, xd, -np.log10(xb)]], dtype=F)
    x, w, ok = M.steady(M.wnom if w is None else w, M.pnom, free=[0, 1, 2], rows=[0, 1, 2], targets=tg, band=BAND, start=start)
    assert bool(ok)
    return x, w


@pytest.fixture(scope='module')
def plant():
    P = PR.rsr()
    return P, Mech([P], n=1, dtype=F)


@pytest.fixture(scope='module')
def nominal(plant):
    return settle(plant[1], PR.NOM['Mr'], PR.NOM['xD'], PR.XB_MAX)


def test_the_steady_state_is_table_3_of_the_paper():
    """Case I (least boilup at the nominal feed): F 958, V 1276, L 778, D 497, xD 0.82, xr 0.43. Case II (largest feed at a boilup of
    1500): F0 497.8, F 1113, L 885, D 615, xD 0.83, xr 0.46. Both with the holdup at 2800 kmol and 0.0105 in the bottoms."""
    c = least_boilup(460., 2800., .0105)
    assert abs(c['F']-958) < 1 and abs(c['V']-1276) < 1 and abs(c['L']-778) < 1 and abs(c['D']-497) < 1, c
    assert abs(c['xD']-.82) < .006 and abs(c['xr']-.43) < .005, c
    assert all(abs(c[q]-PR.TABLE3[q]) < (.3 if q in 'FLVD' else 2e-4) for q in PR.TABLE3), c     # the numbers the plant file carries
    f0 = largest_feed(1500., 2800., .0105)
    c = least_boilup(f0, 2800., .0105)
    assert abs(f0-497.8) < .3 and abs(c['F']-1113) < 1.5 and abs(c['L']-885) < 1.5 and abs(c['D']-615) < 1.5, (f0, c)
    assert abs(c['xD']-.83) < .006 and abs(c['xr']-.46) < .006, c


@pytest.mark.parametrize('vmax,f0', ((500., 278.8079), (2000., 576.4256)))
def test_the_largest_feed_is_the_papers_file(vmax, f0):
    """tab_F0Vmax.txt of the paper's files (holdup 2800, 0.0150 in the bottoms): all 21 rows are within 0.05 %; two of them here."""
    assert abs(largest_feed(vmax, 2800., .015)/f0-1) < 6e-4


def test_what_the_text_says_of_back_off_and_of_what_to_hold():
    c0 = least_boilup(460., 2800., .0105)
    assert .025 < least_boilup(460., 2800., .0085)['V']/c0['V']-1 < .035       # the bottoms backed off to 0.0085: 'about 3 %' more energy
    assert .4 < least_boilup(460., 2000., .0105)['V']/c0['V']-1 < .65          # the holdup backed off (Luyben's structure): 'about 50 %' more boilup
    c1 = least_boilup(414., 2800., .0105)                                      # 10 % less feed, run at its own optimum
    xr = 414.*(X0-.0105)/(K*2800.)
    D = 414.*(xr-.0105)/(c0['xD']-xr)
    assert 0 <= boilup(D, 414., 2800., .0105)/c1['V']-1 < 1e-3                 # xD held where it was: 'less than 0.1 %'

    def held(reflux):                                            # the boilup with the reflux a given function of the recycle
        f = lambda D: column(reflux(D)+D, D, 414.+D, xr)[0]-.0105               # noqa: E731
        Ds = np.linspace(150., 900., 31)
        v = [f(D) for D in Ds]
        (a, b), = [(a, b) for a, b, fa, fb in zip(Ds[:-1], Ds[1:], v[:-1], v[1:]) if fa*fb < 0]
        D = brentq(f, a, b)
        return (reflux(D)+D)/c1['V']-1
    assert 0 <= held(lambda D: c0['L']/c0['F']*(414.+D)) < 1e-4                # L/F held: 'less than 0.01 %'
    assert .02 < held(lambda D: c0['L']) < .06                                 # L held: 2.8 % here; the paper says 'about 5 %' (not reproduced closer)


def test_the_table_is_the_equations(plant):
    P, M = plant
    g = torch.Generator().manual_seed(3)
    for _ in range(20):
        u = torch.rand(10, generator=g, dtype=F)
        x = torch.cat((torch.stack((1500.+1500*u[0], .2+.6*u[1])), torch.sort(.01+.9*torch.rand(22, generator=g, dtype=F)).values,
                       torch.tensor([150.]+[8.]*20+[150.], dtype=F)*(.7+.6*torch.rand(22, generator=g, dtype=F))))[None]
        w = torch.stack((500.+1200*u[2], 300.+1200*u[3], 600.+1400*u[4], 390.+130*u[5], .82+.15*u[6]))[None]
        p = torch.stack((.30+.07*u[7], 1.92+.16*u[8], .6+u[9]))[None]
        ref = rhs(x[0].numpy(), w[0, :3].numpy(), w[0, 3:].numpy(), p[0].numpy())
        assert np.abs(M.f(x, w, p)[0].numpy()-ref).max() < 1e-9*max(1., np.abs(ref).max())


def test_the_engine_settles_where_the_steady_model_does(plant, nominal):
    P, M = plant
    for mr, xd, (x, w) in ((2600., PR.NOM['xD'], nominal), (2800., PR.TABLE3['xD'], settle(M, 2800., PR.TABLE3['xD'], PR.XB_MAX))):
        xr = 460.*(X0-PR.XB_MAX)/(K*mr)
        D = 460.*(xr-PR.XB_MAX)/(xd-xr)
        V = boilup(D, 460., mr, PR.XB_MAX)
        got = [float(a) for a in w[0, :3]]
        assert np.allclose(got, [460.+D, V-D, V], rtol=1e-7), (got, D, V)
        assert abs(float(M.y(x, w, M.pnom)[0, 3])-xr) < 1e-9
    assert all(abs(float(nominal[1][0, i])-PR.NOM[q]) < .01 for i, q in enumerate('FLV'))       # the nominal point of the plant file


def test_the_recycle_snowballs(plant, nominal):
    """With the holdup and both products held, 10 % more feed takes 39 % more recycle and 23 % more boilup."""
    P, M = plant
    x, w = nominal
    wn = M.wnom.clone()
    wn[0, 3] = 506.
    x1, w1 = settle(M, PR.NOM['Mr'], PR.NOM['xD'], PR.XB_MAX, wn, (x, w, M.pnom))
    d0, d1 = float(M.y(x, w, M.pnom)[0, 4]), float(M.y(x1, w1, M.pnom)[0, 4])
    xr = 506.*(X0-PR.XB_MAX)/(K*PR.NOM['Mr'])
    assert abs(d1-506.*(xr-PR.XB_MAX)/(PR.NOM['xD']-xr)) < 1e-4
    assert 1.37 < d1/d0 < 1.42 and 1.24 < float(w1[0, 0]/w[0, 0]) < 1.29 and 1.21 < float(w1[0, 2]/w[0, 2]) < 1.26


def test_the_plants_own_steps_follow_the_fine_integration(plant, nominal):
    """A decision of 3 min in two substeps against LSODA through steps of the feed, of the three valves and of the feed composition."""
    P, M = plant
    x, w = nominal[0].clone(), nominal[1].clone()
    xr, pn = x[0].numpy().copy(), M.pnom[0].numpy()
    worst = np.zeros(4)
    for k in range(60):
        for at, i, to in ((2, 3, 500.), (12, 0, float(nominal[1][0, 0])+60.), (20, 2, float(nominal[1][0, 2])+80.), (30, 1, float(nominal[1][0, 1])+50.), (40, 4, .86)):
            if k == at:
                w[0, i] = to
        x = M.advance(x, w, M.pnom)
        xr = solve_ivp(lambda t, z: rhs(z, w[0, :3].numpy(), w[0, 3:].numpy(), pn), (0., .05), xr, method='LSODA', rtol=1e-9, atol=1e-11).y[:, -1]
        worst = np.maximum(worst, np.abs(M.y(x, w, M.pnom)[0, :4].numpy()-np.array([xr[0], xr[23], -np.log10(xr[2]), xr[1]])))
    assert (worst < .25*np.array([28., .005, .04, .01])).all(), worst          # a quarter of a band at every decision (0.11 at most, on the bottoms)
    assert abs(float(x[0, 0])-xr[0]) < 1. and float(x[0, 0]) > PR.NOM['Mr']+40    # the holdup has run up: nobody held it


def test_the_rows_that_follow_the_holdup_have_their_gains_with_it_held(plant, nominal):
    """Under a step of one valve the holdup runs away and the compositions with it: summed over the decisions that is no gain. With the
    levels named, the gains are those of the true steady state with the holdup where it was (here held by the effluent)."""
    P, M = plant
    x, w = nominal
    assert ME._levels(P)[1] and not any(ME._levels(q)[1] for q in ME.pool(PL.FIRST))
    lev = torch.tensor([[True]+[False]*5])
    Gv, Gr = M.gains(x, w, M.pnom, levels=lev)[4:]
    raw = M.gains(x, w, M.pnom)[4]
    assert np.allclose(Gr[0, 0, :3].numpy(), [-.05, -.05, .05], atol=1e-9)     # kmol a decision per kmol/h: the holdup's balance
    assert float(raw[0, 1, :3].abs().min()) > 100*float(Gv[0, 1, :3].abs().max())
    Kc, r = Gv[0, 1:3, :3], Gr[0, 0, :3]
    held = Kc[:, 1:]-Kc[:, :1]*r[None, 1:]/r[0]                                # d(xD, xB)/d(L, V), the effluent following
    fd = torch.zeros(2, 2, dtype=F)
    for j in (1, 2):
        y = []
        for sg in (1., -1.):
            wq = w.clone()
            wq[0, j] += 2*sg
            xq, wq, ok = M.steady(wq, M.pnom, free=[0], rows=[0], targets=torch.tensor([[PR.NOM['Mr']]], dtype=F), band=BAND[:, :1], start=(x, w, M.pnom))
            assert bool(ok)
            y.append(M.y(xq, wq, M.pnom)[0, 1:3])
        fd[:, j-1] = (y[0]-y[1])/4
    assert float(((held-fd).abs()/fd.abs().amax(1, keepdim=True)).max()) < 1e-3, (held, fd)
    assert held[0, 0] > 0 > held[0, 1] and held[1, 1] > 0                      # reflux purifies the distillate, boilup the bottoms


def test_the_holdup_takes_the_valve_the_plant_names(plant, nominal):
    """All three valves move the holdup alike; by its span the boilup would take it, and then the effluent's own gain on the row left to
    it changes sign once the holdup is held: a loop commissioned on it runs away. The plant names the effluent."""
    P, M = plant
    x, w = nominal
    Gv, Gr = M.gains(x, w, M.pnom, levels=torch.tensor([[True]+[False]*5]))[4:]
    span = torch.tensor([1200., 1200., 1400.], dtype=F)
    G = torch.stack((100*Gr[0, 0, :3]*span/28., Gv[0, 1, :3]*span/.005, Gv[0, 2, :3]*span/.04))[None]
    integ, mvm = torch.tensor([[1., 0., 0.]], dtype=F), torch.ones(1, 3, dtype=F)
    order, lam = ME.pairing(G.clone(), integ, mvm, torch.tensor([[0, -1, -1]]))
    assert order[0].tolist() == [0, 1, 2] and float(lam.min()) > .5            # effluent - holdup, reflux - distillate, boilup - bottoms
    order, lam = ME.pairing(G.clone(), integ, mvm)
    assert order[0].tolist() == [1, 2, 0]
    g = G[0]
    assert g[1, 0] > 0 > g[1, 0]-g[1, 2]*g[0, 0]/g[0, 2]


@pytest.fixture(scope='module')
def prog():
    torch.manual_seed(0)
    return C.build(6, 5, mech=1., plants=ME.pool(['rsr']))


def test_the_shell_publishes_the_rows_with_the_holdup_held(prog):
    c, t = prog.c, prog.mech.groups[0].t
    ar = torch.arange(prog.n)
    assert bool((c['link'][ar, t['orow'][:, 0], t['chan'][:, 0]] == 1).all())  # the holdup with the effluent
    Fr, Gs = prog.Fr[:, 0, -1, :3, :3].double(), prog.Gs[:, 0, :3, :3].double()
    val = (c['integ'][:, :3] == 0)[:, :, None]
    assert float((((Fr-Gs).abs()*val).amax((1, 2))/(Gs.abs()*val).amax((1, 2))).max()) < .02      # the response settles on the gain: no ramp in it
    assert float(c['T'].max()) < 60                              # and the unit's time scale is the column's and the reactor's, not the end of the window


def test_it_is_in_the_default_pool():
    assert 'rsr' in PL.NEW and 'rsr' in PL.PLANTS and 'rsr' not in PL.LARGE and 'rsr' not in PL.FIRST


def test_episodes_on_the_recycle_plant(prog):
    hold = CT.rollout(prog, CT.Hold())
    pi = CT.rollout(prog, CT.PI())
    tch = CT.rollout(prog, CT.Teacher('late'))
    prog.set_teacher()
    assert not bool(hold['failed'].any() | pi['failed'].any() | tch['failed'].any())
    assert float(tch['score'].mean()) < .5*float(pi['score'].mean()) < .5*float(hold['score'].mean())

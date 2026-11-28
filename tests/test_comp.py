"""The compressor (aio.sim.plants_comp): the characteristic against what the paper prints, the equations against a
separate integration, the surge cycle, the pairing, and episodes on it.

The paper: Gravdahl, Willems, de Jager, Egeland, J. Propulsion and Power 20(5), 2004; the same rig in their CDC 2000
paper. Their map (Fig. 6) could not be read from here; what is compared is what the text and the axes print."""
import math

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp

from aio.sim import core as C, control as CT, mech as ME, plants as PL, plants_comp as PC
from aio.sim.engine import Mech

F = torch.float64
# the steady points of the two set-point tests (speed 1000 rpm, kt, flow kg/s, pressure bar), read off the axes of CDC Figs. 5 and 6 with
# mt = kt sqrt(pp - p01): the fuel valve test at kt 0.0015, the blow-off test from kt 0.0015 to 0.0022
POINTS = ((22.5, .0015, .300, 1.41), (21.5, .0015, .283, 1.37), (17.8, .0015, .230, 1.245), (15.5, .0022, .276, 1.17))


@pytest.fixture(scope='module')
def plant():
    P = PC.comp()
    return P, Mech([P], n=1, dtype=F)


def char(m, n, t01=293., p01=1.013, foul=1.):
    """The paper's characteristic written out again (its Eqs. 9-21), with the two fitted numbers."""
    rho, w = p01*1e5/(287.*t01), n*1000*2*np.pi/60
    if m < 0:
        return 10.*m*m+char(0., n, t01, p01, foul)
    d1 = np.sqrt((.106**2+.054**2)/2)
    u1, u2 = w*d1/2, w*.09
    c1, c2 = m/(rho*.00956), m/(rho*np.pi*.18*.007)
    inc_i = .5*abs(u1-c1/np.tan(np.radians(40.)))**2.04
    inc_d = .5*abs(.78*.9*u2-c2/np.tan(np.radians(28.)))**2.04
    f = (-1.8*np.log10(6.9/(u2*.007/1.5e-5)))**-2
    fric = foul*4.8e6*f*(m/rho)**2
    mix = (1/(1+(c2/(.9*u2))**2))*((1-.25-1)/(1-.25))**2*(c2**2+(.9*u2)**2)/2
    w2, w1t = np.hypot(c2, .1*u2), np.hypot(c1, w*.053)
    df = 1-w2/w1t+.75*.9/((w1t/w2)*(20/np.pi*(1-.106/.18)+2*.106/.18))
    h = .9*u2**2-inc_i-inc_d-fric-mix-.05*df**2*u2**2
    return (1+h/(1005.*t01))**3.5


def rhs(x, u, w, p, vp=.0203, lc=2.45, inertia=PC.J):
    """The paper's Eq. 23 and the turbine lag in numpy: x = (pp bar, m kg/s, N 1000 rpm, Pt kW), u = (power kW, 1000 kt),
    w = (T01 K, p01 bar), p = (drive, foul, J, lag)."""
    pp, m, n, pt = x
    t01, p01 = w
    om = n*1000*2*np.pi/60
    mt = u[1]*1e-3*np.sqrt(max(pp-p01, 0.)*1e5)
    return np.array([1.4*287.*t01/vp*(m-mt)/1e5, .00956/lc*(char(m, n, t01, p01, p[1])*p01-pp)*1e5,
                     (1000*pt/om-abs(m)*.09*.9*om*.09)/(inertia*p[2])*60/(2*np.pi*1000), (p[0]*u[0]-pt)/(8.*p[3])])


def cycle(n, share, lc=2.45, vol=1., t1=5.):
    """The surge cycle at the speed n with the valve at `share` of the one through the peak and the turbine power held: frequency Hz, the
    range of the plenum pressure (bar) and of the flow, the swing of the speed (rpm, half of peak to peak, the drift taken out)."""
    ms = PC.surge_flow(n)
    kt = share*ms/math.sqrt((PC.psi_c(ms, n)-1)*PC.P01*1e5)
    m0, p0 = PC.point(n, kt)
    pw = m0*.9*(n*PC.KRPM*.09)**2/1e3
    s = solve_ivp(lambda t, x: rhs(x, (pw, kt*1e3), (293., 1.013), (1., 1., 1., 1.), vp=.0203*vol, lc=lc), (0., t1), [p0*1.003, m0, n, pw],
                  method='LSODA', rtol=1e-8, atol=1e-10, max_step=2e-4, dense_output=True)
    t = np.linspace(t1-1.5, t1, 30001)
    x = s.sol(t)
    q = x[0]-x[0].mean()
    up = np.where((q[:-1] < 0) & (q[1:] >= 0))[0]
    swing = np.mean([np.ptp(x[2][a:b]-np.linspace(x[2][a], x[2][b], b-a)) for a, b in zip(up[:-1], up[1:])])*500
    return (len(up)-1)/(t[up[-1]]-t[up[0]]), x[0].min(), x[0].max(), x[1].min(), x[1].max(), swing


def test_the_characteristic_passes_through_the_test_points_of_the_paper_on_its_falling_side():
    for n, kt, m, p in POINTS:
        mm, pp = PC.point(n, kt)
        assert abs(pp-p) < .006 and abs(mm-m) < .004, (n, kt, mm, pp)
        assert PC.psi_c(mm+1e-3, n) < PC.psi_c(mm-1e-3, n)                      # the rig ran there without surging
        assert mm > PC.surge_flow(n)
    assert all(abs(PC.psi_c(m, n)-char(m, n)) < 1e-12 for n in (14., 21., 26.) for m in (-.1, 0., .05, .2, .35, .5))


def test_the_first_tenth_of_flow_raises_the_pressure_ratio_by_a_tenth():
    """The paper takes cn = 10 for 'approximately the same pressure rise' at +0.1 and -0.1 kg/s."""
    for n in (18., 21., 23.):
        assert .07 < PC.psi_c(.1, n)-PC.psi_c(0., n) < .12, n


def test_the_surge_line_is_inside_the_papers_map_and_the_fit_of_it_holds():
    q, e = PC.surge_line()
    rho = PC.P01*1e5/(PC.RG*PC.T01)
    for n in np.linspace(12., 28., 9):
        assert abs(rho*q*(n/20.)**e/PC.surge_flow(n)-1) < 3e-3
    for n in (18., 21., 23.):                                                    # the three speed lines of the map: flow axis to 0.5 kg/s, ratio 1 to 1.6
        ms = PC.surge_flow(n)
        assert .15 < ms < .35 and 1.15 < PC.psi_c(ms, n) < 1.5
        assert PC.psi_c(1.4*ms, n) > 1.                                          # 'within 2 % up to about 40 % above the surge flow'
    assert abs(PC.surge_flow(21., t01=305.)/PC.surge_flow(21.)-293./305.) < 1e-3  # the surge line is one line in volume flow


def test_the_table_is_the_equations(plant):
    P, M = plant
    g = torch.Generator().manual_seed(3)
    for _ in range(40):
        u = torch.rand(12, generator=g, dtype=F)
        x = torch.tensor([[1.05+.5*u[0], -.15+.7*u[1], 12.+14*u[2], 4.+14*u[3]]], dtype=F)
        w = torch.tensor([[3.+15*u[4], 1.+2.2*u[5], 278.+30*u[6], .99+.04*u[7]]], dtype=F)
        p = torch.tensor([[.85+.23*u[8], 1.+.35*u[9], .7+1.3*u[10], .7+.7*u[11]]], dtype=F)
        ref = rhs(x[0].numpy(), w[0, :2].numpy(), w[0, 2:].numpy(), p[0].numpy())
        assert np.allclose(M.f(x, w, p)[0].numpy(), ref, rtol=1e-9, atol=1e-9)
    y = M.y(M.x0, M.wnom, M.pnom)[0]
    assert abs(float(y[2])-(float(M.x0[0, 1])/PC.surge_flow(float(M.x0[0, 2]))-1)) < 3e-3


def test_it_surges_at_the_frequency_the_paper_measured():
    f, plo, phi, mlo, mhi, swing = cycle(22., .85)
    assert abs(f-19.53) < .6, f                                 # measured 19.53 Hz; the duct length 2.45 m is the paper's fit to it on its own characteristic
    assert mlo < 0 < .4 < mhi and 1.1 < plo < phi < 1.5        # deep surge: the flow reverses
    assert abs(swing-40.) < 6., swing                           # the rotor inertia is set by this one
    assert all(abs(cycle(22., s)[0]-19.53) < 1.6 for s in (.95, .7))
    assert 20. < cycle(22., .85, lc=1.8)[0] < 23.5              # CDC 2000, an earlier loss model with Lc 1.8 m: 23 Hz simulated, 20 Hz measured
    assert 150. < cycle(22., .85, vol=10., t1=10.)[5] < 300.    # the paper: 280 rpm with ten times the plenum volume


def test_the_engine_makes_the_same_cycle_at_small_steps(plant):
    P, _ = plant
    n, h = 22., 2.5e-3
    M = Mech([P], torch.zeros(1, dtype=torch.long), dtype=F, pace=torch.tensor([h], dtype=F))
    ms = PC.surge_flow(n)
    kt = .85*ms/math.sqrt((PC.psi_c(ms, n)-1)*PC.P01*1e5)
    m0, p0 = PC.point(n, kt)
    pw = m0*.9*(n*PC.KRPM*.09)**2/1e3
    x = torch.tensor([[p0*1.003, m0, n, pw]], dtype=F)
    w = torch.tensor([[pw, kt*1e3, 293., 1.013]], dtype=F)
    out = []
    for _ in range(int(3./h)):
        x = M.advance(x, w, M.pnom, substeps=10)
        out.append(float(x[0, 0]))
    q = np.array(out[-int(1.5/h):])
    q = q-q.mean()
    up = np.where((q[:-1] < 0) & (q[1:] >= 0))[0]
    tz = (up-q[up]/(q[up+1]-q[up]))*h
    f = (len(tz)-1)/(tz[-1]-tz[0])
    assert abs(f/cycle(22., .85)[0]-1) < .02, f


def test_the_plants_own_steps_follow_the_fine_integration_right_of_the_peak(plant):
    """A decision of 1 s in two substeps against LSODA: a step of the turbine power, then the valve closing at its full rate to a margin of 10 %."""
    P, M = plant
    x = M.steady(M.wnom, M.pnom)
    x = x[0] if isinstance(x, tuple) else x
    xr = x[0].numpy().copy()
    w = M.wnom.clone()
    worst = np.zeros(4)
    for k in range(60):
        w[0, 0] = 14. if k < 30 else 11.
        w[0, 1] = max(1.9-.085*max(k-30, 0), 1.55)
        x = M.advance(x, w, M.pnom)
        xr = solve_ivp(lambda t, z: rhs(z, w[0, :2].numpy(), (293., 1.013), (1., 1., 1., 1.)), (0., 1.), xr, method='LSODA', rtol=1e-9, atol=1e-11).y[:, -1]
        worst = np.maximum(worst, np.abs(x[0].numpy()-xr))
    assert worst[0] < .005 and worst[1] < .004 and worst[2] < .1, worst     # within one band of pressure, flow and speed at every decision
    assert .08 < float(M.y(x, w, M.pnom)[0, 2]) < .2


def test_either_pairing_has_positive_relative_gains(plant):
    P, M = plant
    x = M.steady(M.wnom, M.pnom)
    x = x[0] if isinstance(x, tuple) else x
    G = M.gains(x, M.wnom, M.pnom)[4][0, :2, :2]
    lam = float(G[0, 0]*G[1, 1]/(G[0, 0]*G[1, 1]-G[0, 1]*G[1, 0]))
    assert G[0, 0] > 0 and G[1, 0] > 0 and G[0, 1] < 0 and G[1, 1] > 0        # more power: pressure and flow up; the valve open: pressure down, flow up
    assert .2 < lam < .8, lam


def test_it_is_in_the_default_pool():
    assert 'comp' in PL.NEW and 'comp' in PL.PLANTS and 'comp' not in PL.LARGE and 'comp' not in PL.FIRST


def test_episodes_on_the_compressor():
    torch.manual_seed(0)
    prog = C.build(6, 5, mech=1., plants=ME.pool(['comp']))
    hold = CT.rollout(prog, CT.Hold())
    pi = CT.rollout(prog, CT.PI())
    tch = CT.rollout(prog, CT.Teacher('late'))
    prog.set_teacher()
    assert not bool(hold['failed'].any() | pi['failed'].any() | tch['failed'].any())
    assert float(tch['score'].mean()) < float(pi['score'].mean()) < float(hold['score'].mean())

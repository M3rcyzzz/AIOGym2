"""A turbine-driven centrifugal compressor as data for the mechanism engine (docs/sim.md, 'Compressor').

The rig of Gravdahl, Willems, de Jager and Egeland, "Modeling of surge in free-spool centrifugal compressors:
experimental validation", J. Propulsion and Power 20(5), 2004 (and their CDC 2000 paper on the same rig): a
BBC VTR 160L turbocharger whose compressor blows through one valve into the atmosphere and whose turbine is
fed on its own. What it adds to the plants of the engine: a unit that answers in seconds, a characteristic
with a peak (to its left the machine surges), and a limit that is a line in the plane of two measurements.

Dynamics (their Eq. 23): plenum pressure, compressor flow, speed,
    dpp/dt = a01^2/Vp (m - mt),   dm/dt = A1/Lc (psi_c(m, w) p01 - pp),   dw/dt = (Pt/w - |m| r2 sigma U2)/J,
the blow-off valve mt = kt sqrt(pp - p01), and the turbine power behind a lag of 8 s after the fuel valve.

Characteristic (their Eqs. 9-21): the ideal work sigma U2^2 less the incidence losses of the impeller and
of the diffuser (to the power 2 + delta), friction (Haaland's factor at Re = U2 b2/nu), mixing and blade
loading, as printed; isentropic to the pressure ratio; in reversed flow cn m^2 + psi_c(0, w). The paper
does not print everything it computes with. Read from it here: Ca3 as the radial velocity at the impeller
exit, m/(rho01 pi D2 b2). Fitted to what it does print (CHI, KF below): the share of the swirl sigma U2 that
meets the diffuser vanes, and the friction number. The facts of the fit and the ones left to check it are in
tests/test_comp.py and docs/sim.md; the paper's map (its Fig. 6) could not be read and is not among them.
Inlet air at 293 K and 1.013 bar (not printed); the rotor inertia from the speed swing in surge.

At the plant's own steps (a decision of 1 s) the pressure-flow mode (24 Hz) is collapsed by the integrator: to
the right of the peak that is what it does in the machine, to the left of it the engine follows the
equilibrium where the machine would cycle. The surge margin is a limited row; nothing here scores or trains
what happens in surge, and there is no anti-surge law (Mercy, 11-11). The cycle itself is in the equations:
the tests run them at steps of a fraction of a millisecond against the paper's surge frequency.

Not here: choke, rotating stall, recirculation at low flow, the reversed-flow torque beyond the paper's
idealisation, the large-vessel configuration of the rig.
"""
import math

from aio.sim.engine import Plant
from aio.sim.expr import E, exp, log, sqrt, abs_, where

# Table 1 of the paper (VTR 160L)
Z, SIG = 20, .9                                # impeller blades, slip factor
DT1, DH1, D2, B2 = .106, .054, .180, .007      # inducer tip and hub, impeller exit diameter and width, m
TB1, TA2 = math.tan(math.radians(40.)), math.tan(math.radians(28.))     # blade inlet angle, diffuser vane inlet angle (from the tangent)
VP, A1, LC = .0203, .00956, 2.45               # plenum volume m3, duct area m2, duct length m (the paper's fit to its surge frequency)
D1 = math.sqrt((DT1**2+DH1**2)/2)              # the diameter that halves the inducer area
DEL, CN = .04, 10.                             # the exponent above 2 of the incidence losses; reversed flow
BLD = Z/math.pi*(1-DT1/D2)+2*DT1/D2            # of the diffusion factor
TLAG = 8.                                      # fuel valve to turbine torque, s
# not printed in the paper
T01, P01 = 293., 1.013                         # inlet air, K and bar
CP, KAP, RG, NU = 1005., 1.4, 287., 1.5e-5     # air
CHI, KF = .78, 4.8e6                           # fitted: share of the swirl at the diffuser vanes; friction number, 1/m4 (loss = KF f (m/rho)^2)
J = .0075                                      # rotor inertia, kg m2: a speed swing of 40 rpm in surge at 22,000 rpm
KRPM = 1000*2*math.pi/60                       # rad/s per 1000 rpm
QS_AT = (12., 28.)                             # the speeds the surge line is fitted over, 1000 rpm


def _is(a):
    return isinstance(a, E)


def _pw(a, e):
    """a^e of a positive number or expression."""
    return exp(e*log(a)) if _is(a) else max(a, 1e-30)**e


def _root(a):
    return sqrt(a) if _is(a) else math.sqrt(max(a, 0.))


def head(m, om, rho, foul=1., chi=CHI, kf=KF):
    """The specific enthalpy the gas keeps (J/kg) at the flow m (kg/s, forward), the speed om (rad/s) and the
    inlet density rho. Numbers or expressions."""
    u1, u2 = om*(D1/2), om*(D2/2)
    ca1, cr2 = m/(rho*A1), m/(rho*(math.pi*D2*B2))
    v1, v3 = u1-ca1/TB1, chi*SIG*u2-cr2/TA2                     # the velocity lost at the inducer and at the diffuser vanes
    lf = log(u2*(B2/(6.9*NU))) if _is(u2) else math.log(u2*B2/(6.9*NU))
    fr = (math.log(10.)/1.8)**2/(lf*lf)                         # Haaland, smooth walls
    q = m/rho
    w2, w1t = _root(cr2*cr2+((1-SIG)*u2)*((1-SIG)*u2)), _root(ca1*ca1+(om*(DT1/2))*(om*(DT1/2)))
    df = 1-w2/w1t+.75*SIG*w2/(w1t*BLD)
    return (SIG*u2*u2-.5*_pw(v1*v1+1e-12, 1+DEL/2)-.5*_pw(v3*v3+1e-12, 1+DEL/2)-foul*kf*fr*q*q
            - (SIG*u2)*(SIG*u2)/18-.05*df*df*u2*u2)


def psi_c(m, n, t01=T01, p01=P01, foul=1., **kw):
    """The pressure ratio at the flow m (kg/s, either way) and the speed n (1000 rpm). Numbers."""
    rho, om = p01*1e5/(RG*t01), n*KRPM
    if m < 0:
        return CN*m*m+psi_c(0., n, t01, p01, foul, **kw)
    return max(1+head(m, om, rho, foul, **kw)/(CP*t01), .05)**(KAP/(KAP-1))


def surge_flow(n, t01=T01, p01=P01, foul=1., **kw):
    """The flow at the peak of the speed line n (1000 rpm), kg/s: the paper's surge line. Golden section."""
    a, b, g = 1e-3, 1.5, (math.sqrt(5)-1)/2
    c, d = b-g*(b-a), a+g*(b-a)
    for _ in range(60):
        if psi_c(c, n, t01, p01, foul, **kw) > psi_c(d, n, t01, p01, foul, **kw):
            b, d = d, c
            c = b-g*(b-a)
        else:
            a, c = c, d
            d = a+g*(b-a)
    return .5*(a+b)


def point(n, kt, t01=T01, p01=P01, foul=1., **kw):
    """The steady flow (kg/s) and plenum pressure (bar) at the speed n (1000 rpm) behind the valve kt."""
    f = lambda m: (psi_c(m, n, t01, p01, foul, **kw)-1)*p01*1e5-(m/kt)**2      # noqa: E731
    a, b = 1e-5, 1.5
    for _ in range(80):
        c = .5*(a+b)
        a, b = (c, b) if f(c) > 0 else (a, c)
    return c, psi_c(c, n, t01, p01, foul, **kw)*p01


def surge_line():
    """(q, e): the volume flow at the peak, m3/s, is q (n/20)^e over QS_AT (the losses are close to homogeneous
    in flow and speed; the fit is within 0.3 %, tests/test_comp.py)."""
    rho = P01*1e5/(RG*T01)
    a, b = (surge_flow(n)/rho for n in QS_AT)
    e = math.log(b/a)/math.log(QS_AT[1]/QS_AT[0])
    return a/(QS_AT[0]/20.)**e, e


def comp(vp=VP, lc=LC, inertia=J):
    """The compressor of the rig: plenum pressure and flow by the turbine power (the fuel valve) and the blow-off
    valve; the margin to the surge line and the speed under limits. Loads: the inlet temperature (measured)
    and pressure. Time in s, pressure in bar, speed in 1000 rpm, power in kW, the valve as 1000 kt."""
    P = Plant('comp', dt=1., substeps=2)
    pw = P.mv('power', 3., 18., 1.5, 11.)                       # 18 kW: the rig reaches 25,000 rpm on its air supply
    kv = P.mv('blowoff', 1., 3.2, .085, 1.9)                    # the valve travels 0.11 V/s; 3.485 to 4.395 V is kt 0.0015 to 0.0022
    t01 = P.load('T01', T01, 278., 308., measured=True)
    p01 = P.load('p01', P01, .99, 1.03)
    drive = P.param('drive', 1., .85, 1.08)                     # turbine power per fuel valve position (the air and the gas it is fed)
    foul = P.param('foul', 1., 1., 1.35)                        # friction (deposits)
    jj = P.param('J', 1., .7, 2., event=False)                  # what is mounted on the shaft
    lag = P.param('lag', 1., .7, 1.4, event=False)
    pp = P.state('pp', 1.33, lo=.5, hi=3.)
    m = P.state('m', .33, lo=-1., hi=2.)
    n = P.state('N', 21., lo=3., hi=40.)
    pt = P.state('Pt', 11., lo=0., hi=40.)
    rho, om = p01*(1e5/RG)/t01, n*KRPM
    fwd = exp((KAP/(KAP-1))*log(1+head(m, om, rho, foul)/(CP*t01)))
    rev = CN*m*m+exp((KAP/(KAP-1))*log(1+head(0., om, rho, foul)/(CP*t01)))
    psi = where(m, fwd, rev)
    mt = kv*1e-3*sqrt((pp-p01)*1e5)
    P.der(pp, (KAP*RG/vp/1e5)*t01*(m-mt))
    P.der(m, (A1/lc*1e5)*(psi*p01-pp))
    P.der(n, (1000.*pt/om-abs_(m)*(SIG*D2*D2/4)*om)/(inertia*KRPM)/jj)
    P.der(pt, (drive*pw-pt)/(TLAG*lag))
    q, e = surge_line()
    P.out('p', pp, range=(1., 1.7), band=.005, track=True, target=(1.18, 1.38))
    P.out('flow', m, range=(0., .6), band=.004, track=True, target=(.24, .40))
    P.out('margin', m/(rho*q*exp(e*log(n/20.)))-1, range=(-.5, 2.), band=.02, lo=.10)     # flow over the surge flow at this speed and inlet state, less one
    P.out('speed', n, range=(8., 30.), band=.1, hi=25.)
    P.out('T01', t01, range=(270., 315.), band=1., dv=True)
    return P

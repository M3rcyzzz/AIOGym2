"""Building heating as data for the mechanism engine (docs/sim.md, 'Buildings').

Two rooms heated by radiators off one boiler loop. What it adds to the plants of the engine: a response with a
fast part and a tail of days behind it (the air and the radiators answer in minutes, the walls in days), which
the linear prior does not reach (docs/sim.md, the clock), on a unit whose loads (weather, sun, occupants) are
as large as its inputs.

Zones: the two-element reduced-order zone of VDI 6007 (the structure of IBPSA ThermalZones.ReducedOrder.RC.
TwoElements): air, exterior walls with their mass, interior mass that sees no ambient, windows and
infiltration as a conductance to the outside. The massless surface nodes are eliminated (each surface in
series with its wall) and the long-wave exchange between the inner surfaces is left out: radiant gains (the
sun through the windows, the radiant share of the radiators and of the occupants) go to the masses by area,
convective gains to the air. The nominal walls are fitted to the step response of the BOPTEST test case
bestest_hydronic (BESTEST 900 with a radiator, an IDEAS model: reports/validation/boptest.json), ten days of
it; tests/test_bldg.py holds the fit.

Radiators: EN 442-2 as in Modelica Buildings Fluid.HeatExchangers.Radiators.RadiatorEN442_2: elements along the
flow, each giving UA/N |T - Tair|^(n-1) (T - Tair), 35 % of it radiant, water and metal capacity 5.8e-6 m3 and
0.0263 kg (500 J/kg K) per W of nominal output; UA from the nominal output at 75/60/20 degC.

Hydraulics: the two branches (valve and radiator in series) between one pair of headers, the pump curve and
the common pipe in front of them; the branch flows follow from the valve conductances in closed form, so
opening one valve takes flow from the other. The boiler holds the supply temperature up to its capacity.

Not here: floor heating (the slab in the path of the heat), cooling, air systems, CO2, occupancy schedules and
weather files (the loads wander and swing as the shell makes every load do).
"""
from aio.sim.engine import Plant
from aio.sim.expr import exp, log, sqrt, clamp

CP = 4186.                                     # water, J/kg K
RAD_N, RAD_EL, RAD_FRAC = 1.3, 3, .35          # EN 442 exponent, elements along the flow, radiant share
RAD_C = 5.8e-6*1000.*CP+.0263*500.             # J/K per W of nominal output: water and metal
NOM = (75., 60., 20.)                          # nominal supply, return and room temperature of a radiator, degC


def _pw(d, n=RAD_N, eps=.5):
    """d |d|^(n-1), smooth through 0 (eps K)."""
    return d*exp(.5*(n-1)*log(d*d+eps*eps))


def rad_ua(q=1., n=RAD_N, nel=RAD_EL, nom=NOM):
    """UA (W/K^n per W of nominal output) of the radiator that gives q at the nominal temperatures, its elements
    in series at the nominal flow (bisection on the steady element balances)."""
    ts, tr, ta = nom
    mcp = q/(ts-tr)

    def out(ua):
        t, tot = ts, 0.
        for _ in range(nel):                                      # each element: mcp (t_in - t) = ua/nel (t - ta)^n, solved by bisection on t
            lo, hi = ta, t
            for _ in range(80):
                mid = .5*(lo+hi)
                if mcp*(t-mid) > ua/nel*(mid-ta)**n:
                    lo = mid
                else:
                    hi = mid
            tot += mcp*(t-lo)
            t = lo
        return tot
    lo, hi = 0., 1.
    for _ in range(80):
        mid = .5*(lo+hi)
        lo, hi = (mid, hi) if out(mid) < q else (lo, mid)
    return .5*(lo+hi)


# per m2 of floor: conductances W/K, capacities J/K; the walls as fitted to BOPTEST bestest_hydronic (tests/test_bldg.py)
ZONE = dict(Hae=4.15, Hai=2.2, Heo=1., Uw=.6, vol=2.7, Ca=1.0e4, Ce=8.9e4, Ci=1.57e5, win=.25, share_e=.5)
AREAS = (48., 32.)                              # floor area of the two rooms, m2
QNOM = (3500., 2400.)                           # nominal output of their radiators, W
H12 = 25.                                       # conductance between the rooms (a partition and a door), W/K
QMAX = 6000.                                    # boiler capacity, W
TRET_HI = 55.                                   # return temperature limit, degC (a condensing boiler: below the dew point of its flue gas)
CB, KB = 2.5e5, 2000.                           # boiler and piping water, J/K; gain of the boiler's own temperature loop, W/K
DT_DES = NOM[0]-NOM[1]
HEAD, PIPE, DROOP = 1.5, .2, .3                 # pump head at no flow, loss of the common pipe and droop of the pump curve at the design flow (the headers then hold 1)


def flows(kv, pump=1.):
    """The flow of each branch over its design flow, from the valve conductances (over their full-open value)
    and the pump head (over its nominal): a branch passes g(k) sqrt(dp), g = k/sqrt(Rv + Rr k^2) (valve and
    radiator in series, authority one half), and the headers hold dp = H/(1 + (pipe + droop) s^2) with s the
    total flow at dp = 1 over the design total. Numbers or expressions."""
    M = [q/(CP*DT_DES) for q in QNOM]
    g = [k/sqrt(.5+.5*k*k) if not isinstance(k, float) else k/(.5+.5*k*k)**.5 for k in kv]
    s = (M[0]*g[0]+M[1]*g[1])/(M[0]+M[1])
    dp = HEAD*pump/(1+(PIPE+DROOP*pump)*s*s)
    root = sqrt(dp) if not isinstance(dp, float) else dp**.5
    return [gi*root for gi in g]


def bldg(zone=None):
    """Two rooms on radiators, one boiler: the two room temperatures and the spread of the loop (supply over
    return) by the supply temperature setpoint and the two radiator valves; the return temperature and the
    boiler load under limits. Loads: outdoor temperature (measured), sun and internal gains. Time in h,
    temperatures in degC.

    The spread and not the return temperature is the third target: with the rooms held by their valves a
    higher supply temperature lowers the return (less flow for the same heat), the opposite of what it does
    with the valves where they are, so no pairing of the return temperature has positive relative gains; the
    spread moves the same way in both cases."""
    z = dict(ZONE, **(zone or {}))
    P = Plant('bldg', dt=900., time_unit=3600., substeps=2)
    tset = P.mv('Tset', 25., 80., 2.5, 52.)
    kv = [P.mv(f'valve{i+1}', .03, 1., .15, .2) for i in range(2)]                    # valve conductance over its full-open value
    tout = P.load('Tout', 2., -10., 14., measured=True)
    sol = P.load('sol', 40., 0., 160.)                                                # sun through the windows, W per m2 of window
    gin = P.load('gain', 4., 0., 10.)                                                 # occupants and appliances, W per m2 of floor
    ach = P.param('ach', .5, .3, 1.2)                                                 # air changes per hour (a window ajar)
    fr = P.param('rad', 1., .8, 1.05)                                                 # radiator output (air in it, a cover)
    ph = P.param('pump', 1., .85, 1.05)                                               # pump head
    ce, ci = P.param('ce', 1., .6, 1.5, event=False), P.param('ci', 1., .6, 1.5, event=False)     # heavy or light walls and floors
    ue = P.param('ue', 1., .7, 1.4, event=False)                                      # insulation
    ta = [P.state(f'Ta{i+1}', 21., lo=-20., hi=60.) for i in range(2)]
    te = [P.state(f'Te{i+1}', 18., lo=-30., hi=60.) for i in range(2)]
    ti = [P.state(f'Ti{i+1}', 21., lo=-20., hi=60.) for i in range(2)]
    tr = [[P.state(f'Tr{i+1}{j+1}', 45.-5.*j, lo=0., hi=110.) for j in range(RAD_EL)] for i in range(2)]
    tb = P.state('Tb', 52., lo=0., hi=110.)
    M = [q/(CP*DT_DES) for q in QNOM]
    m = [M[i]*f for i, f in enumerate(flows(kv, ph))]
    mt = m[0]+m[1]
    tret = (m[0]*tr[0][-1]+m[1]*tr[1][-1])/mt
    qb = clamp(mt*CP*(tset-tret)+KB*(tset-tb), 0., QMAX)
    P.der(tb, 3600.*(qb-mt*CP*(tb-tret))/CB)
    ua = rad_ua()
    for i in range(2):
        A = AREAS[i]
        qel = [fr*QNOM[i]*ua/RAD_EL*_pw(t-ta[i]) for t in tr[i]]
        tin = tb
        for j in range(RAD_EL):
            P.der(tr[i][j], 3600.*(m[i]*CP*(tin-tr[i][j])-qel[j])/(RAD_C*QNOM[i]/RAD_EL))
            tin = tr[i][j]
        qr = qel[0]+qel[1]+qel[2]
        qs, qg = sol*z['win']*A, gin*A
        rad = RAD_FRAC*qr+.5*qg+qs                                                    # radiant gains: to the masses by area
        hw = (z['Uw']*ue+.335*ach*z['vol'])*A                                         # windows and infiltration
        hae, hai, heo = z['Hae']*A, z['Hai']*A, z['Heo']*ue*A
        P.der(ta[i], 3600.*(hae*(te[i]-ta[i])+hai*(ti[i]-ta[i])+hw*(tout-ta[i])+H12*(ta[1-i]-ta[i])+(1-RAD_FRAC)*qr+.5*qg)/(z['Ca']*A))
        P.der(te[i], 3600.*(hae*(ta[i]-te[i])+heo*(tout-te[i])+z['share_e']*rad)/(z['Ce']*ce*A))
        P.der(ti[i], 3600.*(hai*(ta[i]-ti[i])+(1-z['share_e'])*rad)/(z['Ci']*ci*A))
    for i in range(2):
        P.out(f'T{i+1}', ta[i], range=(10., 30.), band=.4, track=True, target=(19., 23.5))
    P.out('dT', tb-tret, range=(0., 40.), band=1., track=True, target=(9., 22.))            # the spread of the loop, supply over return
    P.out('Tret', tret, range=(15., 75.), band=1., hi=TRET_HI)
    P.out('load', qb/QMAX, range=(0., 1.2), band=.02, hi=.92)
    P.out('Tout', tout, range=(-20., 30.), band=1., dv=True)
    return P

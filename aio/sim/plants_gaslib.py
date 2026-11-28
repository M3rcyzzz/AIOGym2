"""GasLib-24 gas network as data for the mechanism engine (docs/sim.md, 'GasLib').

The network GasLib-24 (gaslib.zib.de, CC BY 3.0; Pfetsch et al. 2015) in the edge list of morgen (Himpe, Grundel,
Benner 2021; networks/GasLib24.net, BSD-2): 20 pipes of 10 m to 100 km, three compressor stations, one valve,
three entries at a pressure, five exits drawing a mass flow. The isothermal Euler equations discretised as morgen's
'ode_end' model does, term for term: every pipe cut into pieces of a nominal length (the last one the rest), a
pressure per node and a mass flux per piece; each node holds the volume of the pieces that end in it (at the
nominal length, as morgen does for every edge), the friction of a piece is Schifrinson's in proportion to its
real length and acts at the density of its end node, compressibility AGA88 at the steady mean pressure (one
constant RTz), compressors and valves are edges of the nominal length and the mean pipe diameter without
friction. A compressor holds its outlet pressure at its setpoint (morgen's model of a station under pressure
control); its speed is read off the station's characteristic map of GasLib (n-isolines of the adiabatic head
over volumetric flow, the flow shared by the station's units).

`gaslib_raw()`: setpoints, valve and boundary values as inputs (compared with morgen, aio/sim/validate/gaslib.py).
`gaslib()`: the plant an APC sees; MVs the three compressor outlet pressures and the valve, exits' flows loads.
Units: bar, kg/s, s.
"""
import math

import torch

from aio.sim.engine import Plant, Mech
from aio.sim.expr import E, sqrt, abs_, relu, clamp, maximum, minimum

# morgen networks/GasLib24.net: type, from, to, length (m), diameter (m), height difference (m), roughness (m)
NET = (('P', 1, 3, 50000, 1.1, 0, 1e-5), ('P', 16, 8, 50000, 1.1, 0, 1e-5), ('P', 2, 9, 50000, 1.1, 0, 1e-5), ('S', 9, 10), ('P', 10, 11, 10, 2.1, 0, 1e-5),
       ('P', 12, 13, 30000, .9, 0, 1e-5), ('P', 13, 15, 30000, .9, 0, 1e-5), ('P', 15, 14, 30000, .9, 0, 1e-5), ('P', 13, 16, 30000, .9, 0, 1e-5),
       ('P', 16, 14, 30000, .9, 0, 1e-5), ('P', 14, 17, 30000, .9, 0, 1e-5), ('P', 17, 4, 30000, .9, 0, 1e-5), ('P', 17, 18, 30000, .9, 0, 1e-5),
       ('P', 19, 20, 100000, .9, 0, 1.2e-5), ('P', 20, 23, 50000, .9, 0, 1.2e-5), ('P', 19, 21, 100000, .85, 0, 1.2e-5), ('P', 21, 23, 50000, .85, 0, 1.2e-5),
       ('P', 22, 5, 30000, .5, 0, 1e-5), ('P', 24, 6, 50000, 1.1, 0, 1e-5), ('P', 24, 7, 50000, 1.1, 0, 1e-5), ('S', 3, 10), ('C', 11, 12), ('C', 18, 19),
       ('C', 23, 24), ('V', 21, 22), ('S', 25, 1), ('S', 26, 2), ('S', 27, 3), ('S', 4, 28), ('S', 5, 29), ('S', 6, 30), ('S', 7, 31), ('S', 8, 32))
NOM_LEN = 12000.                   # m: the pieces of the exam plant (morgen's nominal length dt vmax / cfl at cfl 0.1)
NOM_MORGEN = 2400.                 # m: morgen's default (dt 60 s, vmax 20 m/s, cfl 0.5), the comparison
T0, RS = 293.15, 530.              # K, J/(kg K): morgen's GasLib24 scenario (20 C)
TC, PCR = 273.15-82.595, 45.988    # AGA88: critical temperature (K) and pressure (bar)
# GasLib-24 compressor stations (GasLib-24-v1-20211130.cs), in the order of the network's compressor edges: units
# in parallel, speed range (1/min) and the n-isoline coefficients of the first unit: H_ad (kJ/kg) =
# sum c[3i+j] Q^i n^j over i, j = 0..2, Q the unit's inlet volume flow (m3/s)
STATIONS = (('CS1', 3, 5760., 11600., (-9.12494, .00210704, 6.98108e-08, 10.566, -.00011885, 1.33716e-07, -10.5652, .00066214, -3.50928e-08)),
            ('CS2', 1, 3500., 6500., (62.0918, -.0292953, 4.29877e-06, -58.9579, .0265692, -2.57876e-06, 4.32767, -.00240606, 2.45649e-07)),
            ('CS3', 2, 5760., 11600., (-9.12322, .00210668, 6.98297e-08, 10.5641, -.000118454, 1.33695e-07, -10.5647, .00066203, -3.50873e-08)))
KAPPA = 1.3                        # isentropic exponent of the natural gas (adiabatic head)
ZETA0 = 2e4                         # valve: loss coefficient at half open (on the mean pipe area; about 2 bar at the nominal flow); open adds nothing (morgen)


def refine(nom=NOM_LEN):
    """morgen's format_network: pieces of the pipes, supply and demand nodes, the incidence. Returns the edges
    (type, from, to, length, diameter, roughness) and the node lists, in morgen's order."""
    pipes = [e for e in NET if e[0] == 'P']
    mean_d = sum(e[4] for e in pipes)/len(pipes)
    froms, tos = [e[1] for e in NET], [e[2] for e in NET]
    nmax = max(froms+tos)
    supply = sorted(n for n in set(froms)-set(tos) if froms.count(n) == 1)
    demand = sorted(n for n in set(tos)-set(froms) if tos.count(n) == 1)
    edges, extra_edges, nxt = [], [], nmax+1
    for e in NET:
        if e[0] == 'P':
            n = int(math.floor(e[3]/nom))
            if n > 0:
                nodes = [e[1]]+list(range(nxt, nxt+n))+[e[2]]
                nxt += n
                lens = [nom]+[nom]*(n-1)+[math.fmod(e[3], nom)]
                pieces = [('P', nodes[i], nodes[i+1], lens[i], e[4], e[6]) for i in range(n+1)]
                edges.append(pieces[0])
                extra_edges += pieces[1:]
                continue
            edges.append(('P', e[1], e[2], float(e[3]), e[4], e[6]))
        else:
            edges.append((e[0], e[1], e[2], nom, mean_d, 0.))
    edges += extra_edges
    nodes = sorted(set(n for e in edges for n in e[1:3])-set(supply))
    return edges, nodes, supply, demand


def schifrinson(D, k):
    return .11*(k/D)**.25


def _network(P, cp, valve, ps_in, qd, rtz, nom=NOM_LEN, fric=1.):
    """Declare the pressure of every node and the flux of every piece and their equations. cp: compressor outlet
    pressures (bar); valve: opening (0..1, 1 open: morgen's valve); ps_in: entry pressures (bar); qd: exit flows (kg/s).
    Returns the quantities the outputs are made of."""
    edges, nodes, supply, demand = refine(nom)
    ps0, qs0 = steady_guess(edges, nodes, supply, demand)
    p = {n: P.state(f'p{n}', ps0[n], lo=1., scale=1.) for n in nodes}
    q = [P.state(f'q{i}', qs0[i], scale=5.) for i in range(len(edges))]
    cap = {n: 0. for n in nodes}
    for (t, a, b, L, D, k) in edges:
        cap[b] += .25*math.pi*D*D*nom
    comp = [i for i, e in enumerate(edges) if e[0] == 'C']
    flow = {n: 0. for n in nodes}
    for i, (t, a, b, L, D, k) in enumerate(edges):
        A = .25*math.pi*D*D
        pa = ps_in[supply.index(a)] if a in supply else p[a]
        if t == 'C':
            pa = cp[comp.index(i)]
        rho = 1e5*p[b]/rtz
        dq = A/nom*1e5*(pa-p[b])
        lam = schifrinson(D, k) if k > 0 else 0.
        if lam:
            dq = dq-fric*(L/nom)*lam*q[i]*abs_(q[i])/(2.*D*A*rho)
        if t == 'V':
            zeta = ZETA0*(1./(valve*valve)-1.)/3.                 # 0 open, ZETA0 at half open
            dq = dq-A/nom*zeta*q[i]*abs_(q[i])/(2.*rho*A*A)
        P.der(q[i], dq)
        if a in flow:
            flow[a] = flow[a]-q[i]
        flow[b] = flow[b]+q[i]
    for j, n in enumerate(demand):
        flow[n] = flow[n]-qd[j]
    for n in nodes:
        P.der(p[n], 1e-5*rtz/cap[n]*flow[n])
    sup_flow = [q[[i for i, e in enumerate(edges) if e[1] == s][0]] for s in supply]
    comp_in = [p[edges[i][1]] for i in comp]
    return dict(p=p, q=q, edges=edges, demand=demand, supply=supply, sup_flow=sup_flow, comp=comp, comp_in=comp_in)


def steady_guess(edges, nodes, supply, demand):
    """A starting point for the steady state: every pressure 50 bar, fluxes from a spanning flow of the demand."""
    return {n: 50. for n in nodes}, [20.]*len(edges)


def z_aga88(p):
    return 1.+(.257-.533*TC/T0)*p/PCR


def speed(st, m, p_in, p_out, rtz):
    """The speed (1/min) of a station's units on its map: the head the pressure ratio asks, the unit's inlet volume flow."""
    name, units, nmin, nmax, c = st
    zrt = rtz
    head = zrt*KAPPA/(KAPPA-1.)*((p_out/p_in)**((KAPPA-1.)/KAPPA)-1.)/1000.        # kJ/kg
    Q = m/units/(1e5*p_in/zrt)                                                       # m3/s per unit
    a2 = c[2]+c[5]*Q+c[8]*Q*Q
    a1 = c[1]+c[4]*Q+c[7]*Q*Q
    a0 = c[0]+c[3]*Q+c[6]*Q*Q-head
    d = a1*a1-4.*a2*a0
    return (-a1+(sqrt(d) if isinstance(d, E) else math.sqrt(max(d, 0.))))/(2.*a2)


# TRR154 GasLib-24 transient boundary (github.com/TRR154/InstationaryGasNetworks, GasLib-24-no-resistor-sinus, Apache-2.0):
# entry02 at 49 bar, the five exits at these flows at t = 0 (kg/s), in morgen's demand order
QD0 = (21.805555555555557, 21.805555555555557, 26.638102777777778, 26.638102777777778, 21.805555555555557)
PIN0 = 49.


def gaslib_raw(dt=60., substeps=4, rtz=None, nom=NOM_LEN):
    """The network with every boundary value an input: compressor outlet setpoints and valve opening as MVs, entry
    pressures and exit flows as loads. rtz: R T z (J/kg), morgen's constant; by default at this plant's steady mean pressure."""
    P = Plant('gaslib_raw', dt=dt, substeps=substeps)
    edges, nodes, supply, demand = refine(nom)
    cp = [P.mv(f'cp{i+1}', 30., 80., 2., 50.) for i in range(3)]
    valve = P.mv('valve', .05, 1., .05, 1.)
    ps_in = [P.load(f'pin{i+1}', PIN0, 40., 60.) for i in range(len(supply))]
    qd = [P.load(f'qd{i+1}', QD0[i], 10., 40.) for i in range(len(demand))]
    r = P.param('rtz', RTZ if rtz is None else rtz)
    g = _network(P, cp, valve, ps_in, qd, r, nom)
    for j, n in enumerate(g['demand']):
        P.out(f'p_exit{j+1}', g['p'][n], range=(0., 1.), band=1.)
    for j, v in enumerate(g['sup_flow']):
        P.out(f'q_entry{j+1}', v, range=(0., 1.), band=1.)
    return P


RTZ = T0*RS


CP_NOM = (60., 66., 72.)           # bar: compressor outlet setpoints of the exam (each station boosts: 11, 8, 6 bar)
VALVE_NOM = .5
_RTZ = {}


def _steady_rtz():
    """RTz of the exam's nominal point (AGA88 at its mean node pressure, as morgen takes it) and that point."""
    if not _RTZ:
        rtz = RTZ*.9
        for _ in range(3):
            P = gaslib_raw(rtz=rtz)
            M = Mech([P], n=1, dtype=torch.float64)
            x, w, p = P.nominal()
            names = [q['name'] for q in P.ws]
            for j, v in enumerate(CP_NOM):
                w[0, names.index(f'cp{j+1}')] = v
            w[0, names.index('valve')] = VALVE_NOM
            xs, _, ok = M.steady(w, p)
            assert bool(ok.all())
            nodes = refine()[1]
            rtz = RTZ*sum(z_aga88(float(v)) for v in xs[0, :len(nodes)])/len(nodes)
        _RTZ.update(rtz=rtz, x=xs[0].tolist())
    return _RTZ['rtz'], _RTZ['x']


def gaslib():
    """GasLib-24 under its stations' pressure control: the three outlet setpoints and the valve opening are the MVs,
    the five exits' flows (measured, TRR154's values at t = 0) and the entry pressure loads, friction a parameter.
    Tracked: the pressure at exit 1 (behind CS1), exit 2 (behind the valve), exit 3 (behind CS3) and at the valve's
    inlet (the transit pressure behind CS2); limits: every exit at 40-75 bar, each station's speed inside its map's
    range. A decision is 10 min."""
    rtz, x0 = _steady_rtz()
    P = Plant('gaslib', dt=600., substeps=4)
    edges, nodes, supply, demand = refine()
    cp = [P.mv(f'cp{i+1}', v-8., v+8., .5, v) for i, v in enumerate(CP_NOM)]
    valve = P.mv('valve', .2, 1., .03, VALVE_NOM)
    pin = P.load('pin', PIN0, 46., 52.)
    qd = [P.load(f'qd{i+1}', QD0[i], .8*QD0[i], 1.2*QD0[i], measured=True) for i in range(len(demand))]
    fr = P.param('fric', 1., .85, 1.15)
    g = _network(P, cp, valve, [pin]*len(supply), qd, rtz, fric=fr)
    for st, v in zip(P.xs, x0):
        st['x0'] = v
    pe = [g['p'][n] for n in g['demand']]
    P.out('p_exit1', pe[0], range=(40., 75.), band=.3, track=True, target=(52., 64.), lo=40., hi=75.)
    P.out('p_exit2', pe[1], range=(40., 75.), band=.3, track=True, target=(56., 68.), lo=40., hi=75.)
    P.out('p_exit3', pe[2], range=(40., 80.), band=.3, track=True, target=(66., 76.), lo=40., hi=78.)
    P.out('p_transit', g['p'][21], range=(40., 80.), band=.3, track=True, target=(58., 70.))
    P.out('p_exit5', pe[4], range=(40., 75.), band=.3, lo=40., hi=75.)
    for k, i in enumerate(g['comp']):
        st = STATIONS[k]
        n = speed(st, g['q'][i], g['p'][edges[i][1]], cp[k], rtz)
        P.out(f'n_{st[0]}', n, range=(0., st[3]*1.1), band=50., lo=st[2], hi=st[3])
    P.out('supply', sum(g['sup_flow'][1:], g['sup_flow'][0]), range=(80., 160.), band=1.)
    return P

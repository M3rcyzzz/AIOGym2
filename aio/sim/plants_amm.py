"""A quench-cooled ammonia synthesis converter with its feed-effluent exchanger, as data for the mechanism engine
(docs/sim.md, 'Fixed bed').

The reactor of Morud and Skogestad, "Analysis of instability in an industrial ammonia reactor", AIChE J. 44(4),
1998, 888: three catalyst beds in series, fresh feed mixed in before each (the quench), and the rest of the feed
preheated by the effluent of the last bed. The plant it describes lost its pressure in 1989 and its
temperatures swung over about 200 C every six minutes for two hours. What it adds to the plants of the
engine: a distributed unit (a temperature wave takes six minutes through the beds while the gas takes none), a
wrong-way answer (a hotter inlet first cools the outlet: the gas arrives depleted before the wave arrives), and
heat fed back from the outlet to the inlet, which makes the unit oscillate on its own when the pressure or the
feed temperature is too low.

As published: beds of 6.69, 9.63 and 15.2 m3, ten cells each (so many that the numerical dispersion is the
physical one), catalyst 2200 kg/m3 and 1100 J/kg K, gas 3500 J/kg K, 2.7 MJ per kg of ammonia, 252 t/h of
stoichiometric gas with 8 % (mass) ammonia at 250 C and 200 bar, of which 58, 35 and 32 t/h are the quenches;
an exchanger of 283 m2 at 536 W/m2 K without dynamics. The rate is not printed there; it is taken from Straus
and Skogestad, "Economic NMPC for heat-integrated chemical reactors" (2017), who print it for this reactor
(Temkin-Pyzhev, constants below, times 4.75 to match the plant) and state the cells: each a stirred volume
with the rate at its own temperature and ammonia. That is the plant here.

The first paper's own numbers (the eigenvalues at 200 and 170 bar, instability at 172 bar, the loop gains, a
period of 420 s) come out to the last digit printed when the rate of a cell is taken at the ammonia entering
it, which is evidently what its program does; with the cells as stated in the second paper the instability
is at 175 bar and the period 417 s. The plant is not built that way: away from the published point a cell
then converts past equilibrium (at the second paper's own best split it has a steady state with a cell at
600 C). tests/test_amm.py holds both readings against what each paper prints.

Not in the papers and taken here: the gas a cell holds. Both papers leave it out (the ammonia is then the
root of an equation per cell); here a third of the bed (the void fraction of Araujo and Skogestad's note
on the same kinetics, 2008) at the 50 kg/m3 the first paper calls typical. The gas then takes 8 s through
the beds, the pressure of instability moves by half a bar, and a decision costs the engine a quarter of
what the roots cost. Also taken: the valves' travel, the ranges of the loads, the catalyst limit of 530 C.
The beds have no pressure drop and no heat loss; the gas between the beds, in the exchanger and in the
quench lines takes no time.

The published operating point is 25 bar above the pressure where the unit starts to cycle. An operating
point that cycles when left alone is turned down by the shell (mech.Group.settle), so the episodes go towards
that pressure and not below it; the cycle itself is in the equations and in the tests.
"""
import math

from aio.sim.engine import Plant
from aio.sim.expr import E, exp, maximum

NB, NC = 3, 10                                 # beds, cells a bed
VOL = (6.69, 9.63, 15.2)                       # m3
RHO, CPC = 2200., 1100.                        # catalyst: kg/m3, J/kg K
CPG, DH = 3500., 2.7e6                         # gas J/kg K; J per kg of ammonia made
GAS = .33*50.                                  # kg of gas a m3 of bed (not in the two papers)
UA = 536.*283.                                 # W/K
K1, E1, K2, E2, ACT, RG = 1.79e4, 87090., 2.57e16, 198464., 4.75, 8.314    # Temkin-Pyzhev: kmol N2/(m3 catalyst h), bar, J/mol
MNH3 = 17.
M0, P0, TF0, W0 = 70., 200., 250., .08         # kg/s, bar, C, kg ammonia a kg
U0 = (58/252, 35/252, 32/252)                  # share of the feed to each quench
T_HI = 530.                                    # C, the catalyst
NOM = dict(Ti=414.16, Tin=(362.69, 414.74, 479.86), Tout=(445.91, 513.30, 511.18), w=(.1879, .2985, .3113), made=16.192)   # the published point as this model gives it (w: out of each bed; made: kg/s of ammonia)
T_NOM = (368.5, 374.7, 381.2, 388.2, 395.8, 403.9, 412.9, 422.7, 433.7, 445.9,
         424.8, 436.0, 448.5, 462.2, 476.2, 489.2, 499.5, 506.6, 510.9, 513.3,
         493.0, 501.6, 506.4, 508.9, 510.1, 510.7, 511.0, 511.1, 511.2, 511.2)       # its cell temperatures (where the steady solve starts)


def _exp(a):
    return exp(a) if isinstance(a, E) else math.exp(a)


def rate(t, w, p, act=1.):
    """kg of ammonia a kg of catalyst a second at t C, w kg ammonia a kg of gas, p bar (the gas is hydrogen and
    nitrogen three to one and ammonia)."""
    tk = t+273.15
    pa, pn, ph = p*w/(2-w), p*(1-w)/(2*(2-w)), p*3*(1-w)/(2*(2-w))
    ph15 = ph**1.5 if isinstance(ph, E) else ph*math.sqrt(ph)
    return act*ACT*(K1*_exp(-E1/RG/tk)*pn*ph15/pa-K2*_exp(-E2/RG/tk)*pa/ph15)*(2*MNH3/RHO/3600.)


def effectiveness(mh, m, ua=UA):
    """Of the counter-current exchanger: the preheated stream mh against the whole effluent m."""
    ntu, c = ua/(mh*CPG), mh/m
    e = _exp(-ntu*(1-c))
    return (1-e)/(1-c*e)


def ammonia(T, u=U0, m=M0, p=P0, win=W0):
    """The ammonia in every cell (kg a kg of gas) at rest at the cell temperatures T: numbers. The balance of a cell
    falls in its ammonia, so it has one root."""
    flow, w, out = m*(1-sum(u)), win, []
    for b in range(NB):
        w, flow = (flow*w+m*u[b]*win)/(flow+m*u[b]), flow+m*u[b]
        for j in range(NC):
            lo, hi = 1e-9, 1-1e-9
            for _ in range(60):
                mid = .5*(lo+hi)
                lo, hi = (mid, hi) if flow*(w-mid)+VOL[b]*RHO/NC*rate(T[b*NC+j], mid, p) > 0 else (lo, mid)
            w = .5*(lo+hi)
            out.append(w)
    return out


def amm():
    """The three bed inlet temperatures by the three quench shares; the bed outlet temperatures under the catalyst
    limit. Loads: the loop pressure and the feed rate (measured), the feed temperature and its ammonia. Time in s,
    temperatures in C, flows in kg/s."""
    P = Plant('amm', dt=30., substeps=12)
    u = [P.mv(f'q{b+1}', lo, hi, slew, U0[b]) for b, (lo, hi, slew) in enumerate(((.08, .36, .02), (.04, .32, .02), (.04, .30, .02)))]
    p = P.load('p', P0, 176., 215., measured=True)
    m = P.load('feed', M0, 60., 78., measured=True)
    tf = P.load('Tf', TF0, 240., 262.)
    win = P.load('win', W0, .065, .095)
    act = P.param('act', 1., .75, 1.1)                           # the catalyst (ageing, poisoning)
    ua = P.param('ua', 1., .85, 1.1)                             # the exchanger (fouling)
    mass = P.param('mass', 1., .8, 1.25, event=False)            # the catalyst's heat capacity
    T = [P.state(f'T{b+1}_{j+1}', T_NOM[b*NC+j], lo=100., hi=900., scale=100.) for b in range(NB) for j in range(NC)]
    W = [P.state(f'w{i//NC+1}_{i%NC+1}', v, lo=1e-4, hi=.95, scale=.1) for i, v in enumerate(ammonia(T_NOM))]
    mh = maximum(m*(1-u[0]-u[1]-u[2]), .01*m)
    ti = tf+effectiveness(mh, m, UA*ua)*(T[-1]-tf)               # the preheated stream as it leaves the exchanger
    flow, tin, w, tins = mh, ti, win, []
    for b in range(NB):
        q = m*u[b]
        tin, w, flow = (flow*tin+q*tf)/(flow+q), (flow*w+q*win)/(flow+q), flow+q
        tins.append(tin)
        mc = VOL[b]*RHO/NC
        for j in range(NC):
            i = b*NC+j
            r = rate(T[i], W[i], p, act)
            P.der(T[i], (CPG*flow*(tin-T[i])+mc*r*DH)/(mc*CPC*mass))
            P.der(W[i], (flow*(w-W[i])+mc*r)/(GAS*VOL[b]/NC))
            tin, w = T[i], W[i]
    for b in range(NB):
        P.out(f'Tin{b+1}', tins[b], range=(300., 540.), band=2., track=True, target=((345., 385.), (395., 435.), (460., 500.))[b])
    for b in range(NB):
        P.out(f'Tout{b+1}', T[(b+1)*NC-1], range=(380., 580.), band=2., hi=T_HI)
    P.out('w', W[-1], range=(.1, .4), band=.003)
    P.out('p', p, range=(150., 230.), band=1., dv=True)
    P.out('feed', m, range=(50., 90.), band=.5, dv=True)
    return P

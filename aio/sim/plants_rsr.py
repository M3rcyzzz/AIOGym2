"""A reactor, a distillation column and the recycle between them, as data for the mechanism engine (docs/sim.md,
'Reactor, separator, recycle').

The plant of Larsson, Govatsmark, Skogestad and Yu, "Control structure selection for reactor, separator, and
recycle processes", Ind. Eng. Chem. Res. 42 (2003) 1225 (after Wu and Yu 1996, and Luyben's snowball papers): A
reacts to B in an isothermal stirred tank, first order; the column takes B out at the bottom and sends the
unreacted A back to the reactor as its distillate. What it adds to the plants of the engine: a material
recycle. With the reactor holdup held, the recycle and the boilup grow much faster than the feed (the
snowball), the unit as a whole answers over hours while its column answers in minutes, and the holdup of
the reactor is a level the controller holds itself, whose height the products follow (the shell gives
the gains and responses of such rows with the level where it was: mech.Group.model).

As published: feed 460 kmol/h of 0.9 A, rate constant 0.341 1/h, reactor holdup up to 2800 kmol, 22 stages
with the reboiler and a total condenser, liquid feed on stage 13 from the bottom, relative volatility 2,
constant molar flows, bottoms no more than 0.0105 A. tests/test_rsr.py holds the steady state against their
Table 3, their file of the largest feed against the boilup limit, and what the text says of back-off
(one figure, the cost of a constant reflux, is not reproduced: 2.8 % here, about 5 % there).

Not in the paper (it is a steady-state study) and taken as in Skogestad's column A (plants.column): tray
holdups of half a minute of the column feed, linearised tray hydraulics (0.063 min a tray), proportional
level controllers on the condenser drum (by the distillate, which is the recycle) and on the reboiler (by
the bottoms). The drums hold about five minutes of the flow through them. The reactor's own level valve is
its effluent (the conventional structure): all three valves move the holdup alike, and with it on
another one the loops left over change sign once it is held.
"""
import math

from aio.sim.engine import Plant
from aio.sim.expr import log, clamp, maximum

NT, NF = 22, 13                                # stages (1 the reboiler, NT the total condenser); the feed stage
K, X0, F0, ALPHA = .341, .9, 460., 2.          # 1/h; feed composition; kmol/h; relative volatility
MR_MAX, XB_MAX = 2800., .0105                  # kmol; mol A/mol
TABLE3 = dict(F=957.5, L=778.2, V=1275.7, D=497.5, xD=.8151, xr=.4285)    # their Table 3, case I (the least boilup at the nominal feed, holdup 2800), as this model gives it
NOM = dict(Mr=2600., F=1046.72, L=791.91, V=1378.64, D=586.72, xD=.8151, xr=.4615)     # the plant's nominal point: the same products at a holdup of 2600 kmol
M0, TAUL = 8., .063/60                         # kmol a tray; h
MDRUM, KC = 150., 30.                          # kmol in each drum at the nominal flows; level controller gain, 1/h
LN10 = math.log(10.)


def rsr(nt=NT, nf=NF):
    """Reactor holdup, distillate (recycle) and bottoms composition by the reactor effluent, the reflux and the
    boilup. Loads: the feed rate (measured) and its composition. Time in h, flows in kmol/h. The nominal
    holdup is below the vessel's 2800 kmol so that a target has room on both sides."""
    P = Plant('rsr', dt=180., time_unit=3600., substeps=2)
    F = P.mv('F', 500., 1700., 80., NOM['F'])
    L = P.mv('L', 300., 1500., 80., NOM['L'])
    V = P.mv('V', 600., 2000., 80., NOM['V'])
    f0 = P.load('F0', F0, 390., 520., measured=True)
    x0 = P.load('x0', X0, .82, .97)
    k = P.param('k', K, .30, .37)                                # the catalyst, the reactor temperature
    alpha = P.param('alpha', ALPHA, 1.92, 2.08)                  # the column pressure
    hold = P.param('hold', 1., .6, 1.6, event=False)             # tray and drum holdups
    Mr = P.state('Mr', NOM['Mr'], lo=50., hi=4000., scale=2800.)
    xr = P.state('xr', NOM['xr'], lo=0., hi=1., scale=1.)
    x = [P.state(f'x{i+1}', XB_MAX+(NOM['xD']-XB_MAX)*i/(nt-1), lo=0., hi=1., scale=1.) for i in range(nt)]
    M = [P.state(f'M{i+1}', MDRUM if i in (0, nt-1) else M0, lo=.05, scale=MDRUM if i in (0, nt-1) else M0) for i in range(nt)]
    y = [alpha*x[i]/(1+(alpha-1)*x[i]) for i in range(nt)]
    D = maximum(NOM['D']+KC*(M[-1]/hold-MDRUM), 0.)              # the recycle: the condenser drum's level controller
    B = maximum(F0+KC*(M[0]/hold-MDRUM), 0.)
    Lq = [None]+[(NOM['L']+NOM['F'] if i+1 <= nf else NOM['L'])+(M[i]/hold-M0)/TAUL for i in range(1, nt-1)]+[L]     # liquid leaving stage i+1 downwards
    P.der(Mr, f0+D-F)
    P.der(xr, (f0*(x0-xr)+D*(x[-1]-xr))/maximum(Mr, 1.)-k*xr)
    for i in range(nt):
        lin, xin = (Lq[i+1], x[i+1]) if i < nt-1 else (0., 0.)
        vin, yin = (V, y[i-1]) if i else (0., 0.)
        lout = B if i == 0 else (L+D if i == nt-1 else Lq[i])
        vout = V if i < nt-1 else 0.
        dM = lin-lout+vin-vout
        dMx = lin*xin-lout*x[i]+vin*yin-vout*y[i]
        if i == nf-1:
            dM, dMx = dM+F, dMx+F*xr
        P.der(M[i], dM)
        P.der(x[i], (dMx-x[i]*dM)/maximum(M[i], 1e-2))
    xb = clamp(x[0], 1e-9, 1-1e-9)
    P.out('Mr', Mr, range=(1500., 3000.), band=28., track=True, target=(2400., 2750.), lo=1800., hi=MR_MAX, integ=True, by='F')     # the reactor's level valve is its effluent
    P.out('xD', x[-1], range=(.6, 1.), band=.005, track=True, target=(.78, .86))
    P.out('xB', -log(xb)/LN10, range=(1.3, 2.7), band=.04, track=True, target=(1.8, 2.2))       # log10 of the A left in the product: 0.0105 is 1.98
    P.out('xr', xr, range=(.2, .8), band=.01)
    P.out('D', D, range=(0., 1500.), band=15.)
    P.out('F0', f0, range=(300., 600.), band=5., dv=True)
    return P

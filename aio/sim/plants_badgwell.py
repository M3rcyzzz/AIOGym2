"""Badgwell's fired heater as data for the mechanism engine (docs/sim.md, 'Fired heater (Badgwell)').

The 5x5 linear fired heater of the 1998 MPC workshop as Badgwell's mpcsim case gives it (make_heater_cont.m of the
Georgia Tech ChBE 4412 fired heater package): pass flows 1 and 2 and fuel gas (setpoints of their flow
controllers) and the two pass inlet temperatures (measured disturbances) to the combined outlet temperature, the
outlet flow, the pass temperature difference and the two tubeskin temperatures; every element a gain over one or
two first-order lags, in minutes. Operating point and limits as heater_mpcsim.m sets them.
"""
from aio.sim.engine import Plant

U0, D0 = (100., 100., 95.), (540., 540.)
Y0 = (750., 200., 0., 882., 882.)
# (output, input) -> gain, time constants (min); inputs: FC1, FC2, FG, TI1, TI2
TF = {(0, 0): (-.8, (1., 2.)), (0, 1): (-.8, (1., 2.)), (0, 2): (6., (1., 3.)), (0, 3): (.35, (1., 2.)), (0, 4): (.35, (1., 2.)),
      (1, 0): (1., (2.,)), (1, 1): (1., (2.,)),
      (2, 0): (-1., (10., 15.)), (2, 1): (1., (10., 15.)), (2, 3): (.4, (1., 2.)), (2, 4): (-.4, (1., 2.)),
      (3, 0): (-1.2, (1.,)), (3, 2): (10., (1., 2.)), (3, 3): (.8, (1.,)),
      (4, 1): (-1.2, (1.,)), (4, 2): (10., (1., 2.)), (4, 4): (.8, (1.,))}
OUTS = ('TO', 'FO', 'DT', 'TS1', 'TS2')


def heater(dt=1.):
    """The fired heater as an APC sees it: MVs the two pass flows and the fuel gas, the two inlet temperatures
    measured loads. Outlet temperature, outlet flow and pass balance tracked; tubeskins at most 918 F (heater_mpcsim.m:
    MVs 85-135 BPH and 92-103 MSCFH, TO 740-760 F, FO 190-260 BPH, DT -5..5 F). Time in min, a decision a minute."""
    P = Plant('heater', dt=dt*60., time_unit=60., substeps=2)
    fc1, fc2 = (P.mv(f'FC{i+1}', 85., 135., 5., U0[i]) for i in range(2))
    fg = P.mv('FG', 92., 103., 1., U0[2])
    ti = [P.load(f'TI{i+1}', D0[i], 530., 560., measured=True) for i in range(2)]
    gain = P.param('fuel', 1., .9, 1.08)                          # heating value of the fuel gas
    u = (fc1, fc2, fg, ti[0], ti[1])
    u0 = U0+D0
    y = [Y0[j] for j in range(5)]
    for (j, i), (k, taus) in TF.items():
        z = u[i]-u0[i]
        for s, tau in enumerate(taus):
            x = P.state(f'x{j}{i}_{s}', 0., scale=1.)
            P.der(x, (z-x)/tau)
            z = x
        y[j] = y[j]+k*(gain if i == 2 else 1.)*z
    P.out('TO', y[0], range=(730., 770.), band=.5, track=True, target=(742., 758.), lo=740., hi=760.)
    P.out('FO', y[1], range=(180., 270.), band=1., track=True, target=(195., 255.), lo=190., hi=260.)
    P.out('DT', y[2], range=(-10., 10.), band=.3, track=True, target=(-2., 2.), lo=-5., hi=5.)
    P.out('TS1', y[3], range=(860., 930.), band=1., hi=918.)
    P.out('TS2', y[4], range=(860., 930.), band=1., hi=918.)
    return P

"""Literature multiloop benchmarks: the strongly coupled, long-dead-time column models on which multiloop PID
tuning is classically tested. Never trained on, never used to select or tune; aio.sim.stats reads where they sit
in the prior.

Models (time in minutes; G_ij = K e^{-theta s} / (tau s + 1), or the higher-order form given):
- wb  Wood & Berry (1973) methanol-water column, 2x2 (reflux, steam -> top, bottom composition);
- ory Ogunnaike & Ray (1979) ethanol-water column, 3x3;
- ty  Tyreus (1979) stabiliser, 3x3, as reproduced by Huang & Lin, Ind. Eng. Chem. Res. 45 (2006) 3161;
- al  the Alatiqi & Luyben (1986) side-stream column, the 3x3 subsystem in Huang & Lin (2006).
Sources checked 11-08: Wood-Berry in Huang & Lin (2006) and Vu & Lee, ISA Trans. 49 (2010) 79; Ogunnaike-Ray
magnitudes in Edet & Katebi (IFAC PID 2018), signs in Vu & Lee; Tyreus and Alatiqi-Luyben from Huang & Lin.
Decision interval: 30 s (wb, ory), 120 s (ty: one 60 min dead time), 60 s (al)."""
import numpy as np

from aio.interface import NCV, NMV

# (K, theta, tau1, tau2, lead): tau2 None = first order; lead = numerator (lead s + 1)
_F = lambda K, th, tau: (K, th, tau, None, 0.)          # noqa: E731
MODELS = dict(
    wb=dict(dt=30., G=[[_F(12.8, 1., 16.7), _F(-18.9, 3., 21.)],
                       [_F(6.6, 7., 10.9), _F(-19.4, 3., 14.4)]]),
    ory=dict(dt=30., G=[[_F(.66, 2.6, 6.7), _F(-.61, 3.5, 8.64), _F(-.0049, 1., 9.06)],
                        [_F(1.11, 6.5, 3.25), _F(-2.36, 3., 5.), _F(-.012, 1.2, 7.09)],
                        [_F(-34.68, 9.2, 8.15), _F(46.2, 9.4, 10.9), (.87, 1., 3.89, 18.8, 11.61)]]),
    ty=dict(dt=120., G=[[_F(1.986, .71, 66.7), _F(-5.24, 60., 400.), _F(-5.984, 2.24, 14.29)],
                        [(-.0204, .59, 7.14, 7.14, 0.), (.33, .68, 2.38, 2.38, 0.), (-2.38, .42, 1.43, 1.43, 0.)],
                        [_F(-.374, 7.75, 22.22), (11.3, 3.79, 21.74, 21.74, 0.), _F(9.811, 1.59, 11.36)]]),
    al=dict(dt=60., G=[[(4.09, 1.3, 33., 8.3, 0.), (-6.36, .2, 31.6, 20., 0.), _F(-.25, .4, 21.)],
                       [_F(-4.17, 4., 45.), _F(6.93, 1.01, 44.6), _F(-.05, 5., 34.5)],
                       [_F(-1.73, 17., 13.), _F(5.11, 11., 13.3), _F(4.61, 1.02, 18.5)]]),
)
NAMES = tuple(MODELS)


def _tables():
    """Per model: dt (s), K, theta (s), tau1 (s), tau2 (s, 0 = first order), lead (s) as (NCV, NMV) arrays."""
    out = []
    for name in NAMES:
        md = MODELS[name]; G = md['G']; n = len(G)
        T = {k: np.zeros((NCV, NMV)) for k in ('K', 'th', 't1', 't2', 'lead')}
        for i in range(n):
            for j in range(n):
                K, th, t1, t2, lead = G[i][j]
                T['K'][i, j], T['th'][i, j], T['t1'][i, j] = K, 60*th, 60*t1
                T['t2'][i, j] = 0. if t2 is None else 60*t2
                T['lead'][i, j] = 60*lead
        out.append(dict(name=name, n=n, dt=md['dt'], **T))
    return out


TABLES = _tables()

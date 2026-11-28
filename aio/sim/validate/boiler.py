"""The drum boiler of Astrom and Bell (plants_heldout.boiler2) against the plant data of the paper.

    python -m aio.sim.validate.boiler

The data are the six open-loop experiments on the 160 MW unit P16/G16 (Oresundsverket, 1969) as the paper shows them
(Automatica 36 (2000) 363, Figs. 8-13: PRBS-like perturbations of fuel, feedwater or steam flow at medium and high
load; drum pressure and level, 3500 s, 10 s samples), read off the figures (data/boiler_plant.json; the logged
files themselves are not available). Only the perturbed input is shown there: the paper says the steam flow moved
with the pressure in every run and other inputs were corrected by hand. So per run three constants are fitted by
least squares (a gain on the shown input, as the paper corrected its transducers, and the levels of the two inputs not
shown: the heat or feedwater held, the steam flow with its valve held, in proportion to pressure); the rest is the model as published. Reported: the error
of drum pressure and drum level over the run (RMS and largest), and as a share of the measured swing.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from aio.sim import plants_heldout as H
from aio.sim.engine import Mech

DATA = Path(__file__).parent/'data'/'boiler_plant.json'
F = torch.float64
TF = dict(medium=250., high=290.)                # C: feedwater temperature (not given for the runs; 300 C at rated load, TFRT-3192)
SUB = 2                                         # 5 s substeps (ROS2)
P_GAP, L_GAP = .2, .3                           # what the tests hold the model to: RMS error over the measured swing (worst run: 20 %, 25 %)


def load(path=DATA):
    return json.loads(Path(path).read_text())


def simulate(c, thetas):
    """The run c under the model, one episode per row of thetas (the run's constants, k x 3); returns drum pressure (MPa)
    and level (m) at the samples, (k, samples) each."""
    P = H.boiler2(raw=True)
    th = torch.as_tensor(np.atleast_2d(thetas), dtype=F)
    k = th.shape[0]
    M = Mech([P], n=k, dtype=F)
    t, u, pm, lm = (np.array(c[q]) for q in ('t', 'u', 'p', 'level'))
    kind = c['input']
    a, b, s = th[:, 0], th[:, 1], th[:, 2]
    p0 = float(pm[0])
    names = [q['name'] for q in P.ws]
    iQ, iqf, iqs, itf = (names.index(q) for q in ('Q', 'qf', 'qs', 'tf'))

    def inputs(i, p):
        g, c1, c2 = a, b, s                                       # gain on the shown input; the two inputs not shown
        if kind == 'Q':
            Q, qf, qs = g*u[i], c1, c2*p/p0-4.8*(p-8.5)
        elif kind == 'qf':
            Q, qf, qs = c1, g*u[i], c2*p/p0-4.8*(p-8.5)
        else:
            Q, qf, qs = c1, c2, g*u[i]+0.*p
        return Q+0.*p, qf+0.*p, qs                                # a steam valve held: flow in proportion to pressure (the plant
                                                                  # model adds the code's transducer correction to its qs input)
    Q, qf, qs = inputs(0, torch.full((k,), p0, dtype=F))
    x = torch.zeros(k, 4, dtype=F)
    for e in range(k):
        Vwt, ar, Vsd, _ = H.boiler2_point(p0, float(qs[e])+4.8*(p0-8.5), TF[c['load']])
        x[e] = torch.tensor([Vwt, p0, ar, Vsd], dtype=F)
    _, wn, pn = P.nominal()
    w, pn = wn.expand(k, -1).clone(), pn.expand(k, -1)
    w[:, itf] = TF[c['load']]
    out_p, out_l = [], []
    for i in range(len(t)):
        Q, qf, qs = inputs(i, x[:, 1])
        w[:, iQ], w[:, iqf], w[:, iqs] = Q, qf, qs
        y = M.y(x, w, pn)
        out_p.append(y[:, 0]); out_l.append(y[:, 1])
        if i < len(t)-1:
            x = M.advance(x, w, pn, substeps=SUB*int(round(t[i+1]-t[i]))//10)
    p, lv = torch.stack(out_p, 1).numpy(), torch.stack(out_l, 1).numpy()
    return p, lv-lv[:, :1]+lm[0]                                  # the model's level from where the measured one starts


def _res(c, p, l):
    return np.nan_to_num(np.concatenate([(p-np.array(c['p']))/.02, (l-np.array(c['level']))/.01], -1), nan=1e3, posinf=1e3, neginf=-1e3)


def fit(c, iters=12):
    """The run's three constants by Levenberg-Marquardt on pressure (in 0.02 MPa) and level (in 1 cm), the
    finite-difference columns run as one batch with the point."""
    u = np.array(c['u'])
    qs0 = 50. if c['load'] == 'medium' else 105.
    um = float(np.mean(u))
    Q0 = 88. if c['load'] == 'medium' else 175.
    th = np.array([1., qs0, qs0] if c['input'] == 'Q' else [1., Q0, um] if c['input'] == 'qs' else [1., Q0, qs0])
    lam, best = 1e-2, None
    for _ in range(iters):
        h = 1e-4*np.abs(th)
        T = np.vstack([th]+[th+np.eye(3)[j]*h[j] for j in range(3)])
        p, l = simulate(c, T)
        R = _res(c, p, l)
        r0 = R[0]
        if best is None or r0 @ r0 < best[0]:
            best = (r0 @ r0, th.copy())
        J = ((R[1:]-r0)/h[:, None]).T
        A = J.T@J
        th_new = th-np.linalg.solve(A+lam*np.diag(np.diag(A))+1e-9*np.eye(3), J.T@r0)
        pn, ln = simulate(c, th_new)
        rn = _res(c, pn, ln)[0]
        if rn @ rn < r0 @ r0:
            th, lam = th_new, lam/3
        else:
            lam *= 4
    return best[1] if best[0] < (rn @ rn) else th


def summary(data):
    rows = []
    for f, c in sorted(data.items(), key=lambda kv: int(kv[0])):
        th = fit(c)
        p, l = (v[0] for v in simulate(c, th))
        pm, lm = np.array(c['p']), np.array(c['level'])
        ep, el = p-pm, l-lm
        rows.append(dict(fig=int(f), input=c['input'], load=c['load'], theta=[float(v) for v in th],
                         p_rms=float(np.sqrt(np.mean(ep**2))), p_max=float(np.abs(ep).max()), p_swing=float(pm.max()-pm.min()),
                         l_rms=float(np.sqrt(np.mean(el**2))), l_max=float(np.abs(el).max()), l_swing=float(lm.max()-lm.min())))
    return rows


def table(rows):
    lines = ['plant data of Astrom & Bell (2000), Figs. 8-13, against plants_heldout.boiler2 (three constants fitted per run)',
             f'{"fig":>4}{"input":>6}{"load":>8}{"p RMS":>9}{"max":>7}{"/swing":>8}{"level RMS":>11}{"max":>7}{"/swing":>8}  fitted']
    for r in rows:
        lines.append(f'{r["fig"]:>4}{r["input"]:>6}{r["load"]:>8}{r["p_rms"]:>9.3f}{r["p_max"]:>7.3f}{100*r["p_rms"]/r["p_swing"]:>7.1f}%'
                     f'{r["l_rms"]:>11.4f}{r["l_max"]:>7.3f}{100*r["l_rms"]/r["l_swing"]:>7.1f}%  {[round(v, 3) for v in r["theta"]]}')
    lines.append('pressure MPa, level m; swing: the measured range over the run')
    return '\n'.join(lines)


if __name__ == '__main__':
    print(table(summary(load(sys.argv[1] if len(sys.argv) > 1 else DATA))))

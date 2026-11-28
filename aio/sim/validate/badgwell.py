"""Badgwell's fired heater of the engine (plants_badgwell) against the case's own model and run.

    python -m aio.sim.validate.badgwell

data/badgwell_heater.npz: A, B, C, D of HeaterModelCont.mat (the case's minimal continuous realisation of its
transfer functions, minutes, deviation variables) and the logged run heaterData.mat of its MPC (inputs and outputs a
minute). Compared: the step response coefficients of all 25 pairs over 100 min at the case's 1 min sample, and the
outputs of the logged run replayed open loop through the engine (inlet temperatures at their nominal values).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import expm

from aio.sim import plants_badgwell as H
from aio.sim.engine import Mech

DATA = Path(__file__).parent/'data'/'badgwell_heater.npz'
F = torch.float64
NSTEP = 100


def reference_steps(z, n=NSTEP):
    """(n, 5 outputs, 5 inputs): unit step responses of the case's state space at 1 min (exact zero-order hold)."""
    A, B, C, D = (z[k] for k in 'ABCD')
    nx = A.shape[0]
    M = expm(np.block([[A, B], [np.zeros((B.shape[1], nx+B.shape[1]))]]))
    Ad, Bd = M[:nx, :nx], M[:nx, nx:]
    out, x = [], np.zeros((nx, B.shape[1]))
    for _ in range(n):
        x = Ad@x+Bd
        out.append(C@x+D)
    return np.array(out)


def engine_steps(n=NSTEP):
    P = H.heater()
    M = Mech([P], n=5, dtype=F)
    x, w, p = P.nominal()
    x, p = x.expand(5, -1).clone(), p.expand(5, -1)
    w0 = w.expand(5, -1)
    w1 = w0.clone()
    order = ['FC1', 'FC2', 'FG', 'TI1', 'TI2']
    names = [q['name'] for q in P.ws]
    for i, q in enumerate(order):
        w1[i, names.index(q)] += 1.
    y0 = M.y(x, w0, p)
    out = []
    for _ in range(n):
        x = M.advance(x, w1, p, substeps=20)
        out.append((M.y(x, w1, p)-y0).T.numpy())                   # (outputs, inputs)
    return np.array(out)


def replay(z):
    P = H.heater()
    M = Mech([P], n=1, dtype=F)
    x, w, p = P.nominal()
    names = [q['name'] for q in P.ws]
    U = z['udata']
    Y = [M.y(x, w, p)[0].numpy()]
    for k in range(1, U.shape[1]):
        v = w.clone()
        for i, q in enumerate(('FC1', 'FC2', 'FG')):
            v[0, names.index(q)] = U[i, k-1]                       # the MPC's move at k-1 holds over the minute
        x = M.advance(x, v, p, substeps=20)
        Y.append(M.y(x, v, p)[0].numpy())
    return np.array(Y)[:, :5].T


def summary(z):
    ref, eng = reference_steps(z), engine_steps()
    gain = np.abs(ref[-1]).max()
    dev = np.abs(eng-ref).max()
    run = replay(z)
    return dict(step_dev=float(dev), step_gain=float(gain), step_rel=float(np.abs(eng-ref).max()/max(np.abs(ref).max(), 1e-12)),
                run_dev=[float(v) for v in np.abs(run-z['ydata']).max(1)], run_move=[float(v) for v in np.ptp(z['ydata'], 1)])


def table(r):
    return '\n'.join([f'step response coefficients, 25 pairs x {NSTEP} min: largest gap {r["step_dev"]:.2e} (largest coefficient {r["step_gain"]:.1f}, '
                      f'relative {r["step_rel"]:.1e})',
                      'logged MPC run replayed open loop, largest gap per output (output range): ' +
                      ', '.join(f'{q} {d:.2e} ({m:.1f})' for q, d, m in zip(H.OUTS, r['run_dev'], r['run_move']))])


if __name__ == '__main__':
    print(table(summary(np.load(sys.argv[1] if len(sys.argv) > 1 else DATA))))

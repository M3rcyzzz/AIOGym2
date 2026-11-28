"""The refrigeration cycle of the engine (plants_refrig) against the IFAC PID 2018 benchmark's reference runs.

    python -m aio.sim.validate.refrig

The benchmark (www.dia.uned.es/~fmorilla/benchmarkPID2018) ships its model compiled; what it ships in the open are
two runs of its standard 20 min test (controllers 1 and 2: the same reference and disturbance steps, different
moves), every input and output logged a second (data/refrig_reference.npz holds the inputs and the outputs compared).
The engine's raw plant is driven open loop by the logged speed, valve and disturbances (the logged inputs act one
sample later in the benchmark's logs) from its steady state at the first sample. Run A identified the components
(compressor and valve maps, the evaporator's two zones, the condenser's wall); run B was not used for that.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from aio.sim import plants_refrig as R
from aio.sim.engine import Mech

DATA = Path(__file__).parent/'data'/'refrig_reference.npz'
F = torch.float64
LOADS = ('Tsec_condens_in', 'msec_condens', 'Psec_condens', 'Tsec_evap_in', 'msec_evap', 'Psec_evap', 'Tsurr')
ROWS = (('Tsec_out', 'Tsec_evap_out', 'C'), ('TSH', 'TSH', 'K'), ('Pe', 'Pevap', 'bar'), ('Pc', 'Pcondens_out', 'bar'), ('m', 'm', 'g/s'), ('COP', 'COP', '-'))
GAP = dict(Tsec_out=.05, TSH=.12)                                  # what the tests hold the engine to: mean error over the run's range


def load(path=DATA):
    z = np.load(path)
    out = {}
    for k in z.files:
        r, f = k.split('/')
        out.setdefault(r, {})[f] = z[k]
    return out


def simulate(c):
    P = R.refrig_raw()
    M = Mech([P], n=1, dtype=F)
    names = [q['name'] for q in P.ws]
    x, w, p = P.nominal()
    K = len(c['N'])

    def wk(k):
        k = max(k-1, 0)                                            # the benchmark's logged input acts one sample later
        v = w.clone()
        v[0, names.index('N')], v[0, names.index('Av')] = c['N'][k], c['Av'][k]
        for (q, *_), src in zip(R.DIST, LOADS):
            v[0, names.index(q)] = c[src][k]
        return v
    x, _, ok = M.steady(wk(0), p, start=(x, wk(0), p))
    assert bool(ok.all())
    Y = []
    for k in range(K):
        Y.append(M.y(x, wk(k), p)[0])
        x = M.advance(x, wk(k), p)
    Y = torch.stack(Y).numpy()
    yn = [o['name'] for o in P.ys]
    return {q: Y[:, yn.index(q)] for q, *_ in ROWS}


def summary(data):
    rows = []
    for run, c in data.items():
        e = simulate(c)
        for q, ref, unit in ROWS:
            d = np.abs(e[q]-c[ref])
            rows.append(dict(run=run, row=q, unit=unit, mean=float(d.mean()), max=float(d.max()), range=float(np.ptp(c[ref]))))
    return rows


def table(rows):
    lines = ['IFAC PID 2018 refrigeration benchmark: engine (open loop on the logged inputs) against the reference runs (A fitted, B not)',
             f'{"run":<4}{"row":<10}{"mean error":>12}{"largest":>10}{"run range":>11}']
    for r in rows:
        lines.append(f'{r["run"]:<4}{r["row"]:<10}{r["mean"]:>12.3f}{r["max"]:>10.3f}{r["range"]:>11.3f}  {r["unit"]}')
    return '\n'.join(lines)


if __name__ == '__main__':
    print(table(summary(load(sys.argv[1] if len(sys.argv) > 1 else DATA))))

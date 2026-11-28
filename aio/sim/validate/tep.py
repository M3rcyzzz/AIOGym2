"""Tennessee Eastman of the engine against the native code (docs/sim.md, 'TEP and BSM1').

    python -m aio.sim.validate.tep [data/tep_native.npz]

The data are trajectories of the native library recorded by tep_native_record.py (the routine's own 50
states, no measurement noise). Open loop: the engine's raw plant starts from the same state with the same
valves and disturbances; compared are XMEAS 1-41 computed from both state trajectories. Closed loop: the
engine's plant with its regulatory layer (continuous PI) against the native code under the same layer run
once a second; compared are the plant's output rows, in bands, at the production step and at a fine one.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from aio.sim import plants_tep as T
from aio.sim.engine import Mech

DATA = Path(__file__).parent/'data'/'tep_native.npz'
F = torch.float64
# the routine's step disturbances as the plant's loads (IDV 2 keeps the A/C ratio: it takes 0.00243719 off A as it adds 0.005 to B)
IDV = {0: {}, 1: dict(xA4=.455), 2: dict(xB4=.010, xA4=.485-2.43719e-3), 3: dict(tD=50.), 4: dict(tcwr=40.), 5: dict(tcws=45.), 6: dict(a_avail=0.), 7: dict(c_avail=.8)}
EVENT = 3600                                                         # closed loop: the change comes an hour in
FINE = 120                                                           # substeps of the fine closed-loop run (ten seconds)
OPEN_GAP, CLOSED_GAP, FINE_GAP = .002, 2., .5                        # what the tests hold the engine to: share of the base value; bands at the production step and at the fine one


def load(path=DATA):
    z = np.load(path, allow_pickle=False)
    out = {}
    for key in z.files:
        if '/' in key:
            name, field = key.split('/')
            out.setdefault(name, {})[field] = z[key]
    return out, [str(q) for q in z['mv_names']]


def _w(P, values):
    w = P.nominal()[1]
    names = [q['name'] for q in P.ws]
    for q, v in values.items():
        w[0, names.index(q)] = float(v)
    return w


def open_loop(c):
    """One open-loop case: the engine from the native start, one-second steps. Returns XMEAS 1-41 of the native
    states and of the engine's (both through the engine's output equations), and the native window means."""
    t = c['t'].astype(int)
    every, N = int(t[1]-t[0]), len(t)
    P = T.tep_raw(dt=float(every), substeps=every)
    w = _w(P, dict({f'xmv{j+1}': v for j, v in enumerate(c['u'])}, **IDV[int(c['fault'])]))
    p = P.nominal()[2]
    xn = T.from_routine(c['y'])
    M1, MN = Mech([P], n=1, dtype=F), Mech([P], n=N, dtype=F)
    x, X = xn[:1], [xn[:1]]
    for _ in range(N-1):
        x = M1.advance(x, w, p, refresh=True)
        X.append(x)
    out = lambda v: MN.y(v, w.expand(N, -1), p.expand(N, -1))[:, :41].numpy()         # noqa: E731
    return dict(t=t, native=out(xn), engine=out(torch.cat(X)), means=c['xm'].astype(float), stop=int(c['stop']))


def closed_loop(c, names, substeps=None):
    """One closed-loop case: the engine's nine-MV plant from the native start, a decision every twenty minutes.
    Returns the output rows of the native states and of the engine's at the decisions."""
    P = T.tep9()
    k = int(round(P.dt/float(c['t'][1]-c['t'][0])))
    yn = c['y'][::k].astype(float)
    N = len(yn)
    mv = dict(zip(names, c['mv']))
    w0, w1 = _w(P, {}), _w(P, dict({q: mv[q] for q in T.MVS9}, **IDV[int(c['fault'])]))
    p = P.nominal()[2]
    M1, MN = Mech([P], n=1, dtype=F), Mech([P], n=N, dtype=F)
    pad = lambda v: torch.cat([v, torch.zeros(len(v), P.nx-50, dtype=F)], -1)         # noqa: E731
    x = pad(T.from_routine(yn[:1]))
    X = [x]
    for i in range(1, N):
        x = M1.advance(x, w1 if i*P.dt > EVENT else w0, p, substeps=substeps)
        X.append(x)
    out = lambda v: MN.y(v, w1.expand(N, -1), p.expand(N, -1)).numpy()                # noqa: E731
    return dict(t=np.arange(N)*P.dt, native=out(pad(T.from_routine(yn))), engine=out(torch.cat(X)), stop=c['stop'], plant=P)


def summary(data, names, fine=True):
    """The comparison as rows of numbers (what the table and the tests read)."""
    base = np.array(T.base()['xm'])
    rows = dict(open=[], closed=[])
    for name, c in data.items():
        if name.startswith('ol_'):
            r = open_loop(c)
            move = np.abs(r['native']-r['native'][0]).max(0)/base
            dev = np.abs(r['engine']-r['native']).max(0)/base
            i, j = int(move.argmax()), int(dev.argmax())
            mid = .5*(r['native'][1:]+r['native'][:-1])[:, :22]                        # window means against the states at both ends of the window
            bias = np.abs((r['means'][1:, :22]-mid).mean(0))/base[:22]
            rows['open'].append(dict(case=name, seconds=int(r['t'][-1]), stop=r['stop'], move=float(move[i]), move_at=i+1, dev=float(dev[j]), dev_at=j+1,
                                     bias=float(bias.max()), bias_at=int(bias.argmax())+1))
        elif name.startswith('cl_'):
            r = closed_loop(c, names)
            band = np.array([o['band'] for o in r['plant'].ys])
            onames = [o['name'] for o in r['plant'].ys]
            move = np.abs(r['native']-r['native'][0]).max(0)/band
            dev = np.abs(r['engine']-r['native'])/band
            row = dict(case=name, hours=float(r['t'][-1]/3600.), stop=[int(s) for s in r['stop']], move=float(move.max()), move_at=onames[int(move.argmax())],
                       dev=float(dev.max()), dev_at=onames[int(dev.max(0).argmax())], end=float(dev[-1].max()))
            if fine:
                rf = closed_loop(c, names, substeps=FINE)
                row['dev_fine'] = float((np.abs(rf['engine']-rf['native'])/band).max())
            rows['closed'].append(row)
    return rows


def table(rows):
    lines = ['open loop (XMEAS 1-41, % of the base value; bias: native window means of XMEAS 1-22 against the states)',
             f'{"case":<10}{"seconds":>8}{"trip":>6}{"largest move":>16}{"largest gap":>16}{"bias":>10}']
    for r in rows['open']:
        lines.append(f'{r["case"]:<10}{r["seconds"]:>8}{r["stop"]:>6}{100*r["move"]:>10.2f} ({r["move_at"]:>2}){100*r["dev"]:>11.4f} ({r["dev_at"]:>2}){100*r["bias"]:>10.3f}')
    lines += ['', 'closed loop (output rows, in bands; gap at the production step and at ten-second steps; cl_base is the noise floor of the native runs:',
              'measurement noise reaches the valves through the layer, mean of three seeds)',
              f'{"case":<16}{"hours":>6}{"largest move":>20}{"gap":>16}{"at the end":>12}{"fine":>8}  trips (s)']
    for r in rows['closed']:
        lines.append(f'{r["case"]:<16}{r["hours"]:>6.1f}{r["move"]:>10.1f} ({r["move_at"]:>7}){r["dev"]:>7.2f} ({r["dev_at"]:>6}){r["end"]:>12.2f}{r.get("dev_fine", float("nan")):>8.2f}  {r["stop"]}')
    return '\n'.join(lines)


if __name__ == '__main__':
    d, mv_names = load(sys.argv[1] if len(sys.argv) > 1 else DATA)
    print(table(summary(d, mv_names)))

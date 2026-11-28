"""The vinyl acetate plant of the engine against the native code (docs/sim.md, 'VAM').

    python -m aio.sim.validate.vam [data/vam_native.npz]

The data are trajectories of the native C code recorded by vam_native_record.py (explicit Euler at 1/3 s, no
noise). Base: the engine's 43 measurements at the code's base against the paper's Appendix 3. Open loop: the
engine's raw plant from the same state with the same MVs and disturbances; compared are the 43 measurements of both
state trajectories (through the engine's output equations), in % of the base value. Closed loop: the engine's plant
with its regulatory layer against the native code under the same layer run once a second; compared are the
plant's output rows, in bands.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from aio.sim import plants_vam as V
from aio.sim.engine import Mech

DATA = Path(__file__).parent/'data'/'vam_native.npz'
F = torch.float64
EVENT = 10.                                                          # min: closed-loop changes come ten minutes in
OPEN_GAP, CLOSED_GAP, PAPER_GAP = .01, 1., .005                     # what the tests hold the engine to (paper: the native base has the same gaps)


def load(path=DATA):
    z = np.load(path, allow_pickle=False)
    out = {}
    for key in z.files:
        name, field = key.split('/')
        out.setdefault(name, {})[field] = z[key]
    return out


def _w(P, values):
    w = P.nominal()[1]
    names = [q['name'] for q in P.ws]
    for q, v in values.items():
        w[0, names.index(q)] = float(v)
    return w


def _event(name):
    """The case's disturbance: loads during it (and how long, min)."""
    return dict(c2h6=(dict(xC2H6=.003), None), colfeed=(dict(colfeed=0.), 5.), hac=(dict(F_HAc=0.), 5.), o2=(dict(F_O2=0.), 5.)).get(name.split('_', 1)[1], ({}, None))


def paper():
    """Largest gap of the engine's base measurements to the paper's Appendix 3, as a share of the value (the
    compositions printed to six decimals: values under 1e-3 against 1e-3)."""
    b, p = np.array(V.base()['meas']), np.array(V.PAPER)
    g = np.abs(b-p)/np.maximum(np.abs(p), 1e-3)
    return float(g.max()), V.MEAS[int(g.argmax())]


def open_loop(c, name, substeps=12):
    P = V.vam_raw(dt=60., substeps=substeps)
    every = int(c['t'][1]-c['t'][0])
    u = dict(zip(V.MVS_RAW, c['u']))
    loads, until = _event(name)
    w1 = _w(P, {**u, **loads})
    w0 = _w(P, u)
    p = P.nominal()[2]
    n = len(c['t'])
    xn = torch.cat([torch.tensor(c['y'], dtype=F), torch.tensor(c['t'], dtype=F)[:, None]], 1)
    M = Mech([P], n=1, dtype=F)
    x = torch.tensor([list(V.X0)+[0.]], dtype=F)                       # the code's base, exactly
    X = [x]
    for k in range(1, n):
        for s in range(every):
            m = (k-1)*every+s
            x = M.advance(x, w1 if until is None or m < until else w0, p)
        X.append(x)
    MN = Mech([P], n=n, dtype=F)
    out = lambda v: MN.y(v, w0.expand(n, -1), p.expand(n, -1))[:, :43].numpy()        # noqa: E731
    return out(xn), out(torch.cat(X))


def closed_loop(c, name):
    P = V.vam_all()
    P.dt, P.substeps = 300., 5
    every = int(round(P.dt/60.))
    k = int(round(every/(c['t'][1]-c['t'][0]))) or 1
    yn = torch.tensor(c['y'][::k], dtype=F)
    n = len(yn)
    case = name.split('_', 1)[1]
    loads, until = _event(name)
    loads = {('o2_avail' if q == 'F_O2' else 'hac_avail' if q == 'F_HAc' else q): v for q, v in loads.items()}
    change = {}
    for q in V.APC:
        if case.startswith(q) and case[len(q):].replace('.', '', 1).isdigit():
            change[q] = float(case[len(q):])
    w0, ws, wl = _w(P, {}), _w(P, change), _w(P, {**change, **loads})
    p = P.nominal()[2]
    M = Mech([P], n=1, dtype=F)
    x = P.nominal()[0]
    X = [x]
    for i in range(1, n):
        m = (i-1)*every
        w = w0 if m < EVENT else wl if until is None or m < EVENT+until else ws
        x = M.advance(x, w, p)
        X.append(x)
    MN = Mech([P], n=n, dtype=F)
    out = lambda v: MN.y(v, ws.expand(n, -1), p.expand(n, -1)).numpy()                  # noqa: E731
    return out(yn), out(torch.cat(X)), P


def summary(data):
    scale = np.maximum(np.abs(np.array(V.base()['meas'])), 1e-4)
    rows = dict(paper=paper(), open=[], closed=[], failed=[])
    for name, c in data.items():
        if not np.isfinite(c['y']).all():                            # the native code broke down (Column.c's decanter matrix)
            rows['failed'].append(name)
        elif name.startswith('ol_'):
            nat, eng = open_loop(c, name)
            move = np.abs(nat-nat[0]).max(0)/scale
            dev = np.abs(eng-nat).max(0)/scale
            i, j = int(move.argmax()), int(dev.argmax())
            rows['open'].append(dict(case=name, minutes=int(c['t'][-1]), move=float(move[i]), move_at=V.MEAS[i], dev=float(dev[j]), dev_at=V.MEAS[j],
                                     rel=float(dev[j]/move[i])))
        elif name.startswith('cl_'):
            nat, eng, P = closed_loop(c, name)
            band = np.array([o['band'] for o in P.ys])
            names = [o['name'] for o in P.ys]
            move = np.abs(nat-nat[0]).max(0)/band
            dev = np.abs(eng-nat)/band
            rows['closed'].append(dict(case=name, hours=float(c['t'][-1]/60.), move=float(move.max()), move_at=names[int(move.argmax())],
                                       dev=float(dev.max()), dev_at=names[int(dev.max(0).argmax())], end=float(dev[-1].max())))
    return rows


def table(rows):
    g, at = rows['paper']
    lines = [f'base against the paper (Appendix 3; under 1e-3 against 1e-3): largest gap {100*g:.3f} % ({at})', '',
             'open loop (43 measurements, % of the base value; values under 1e-4 against 1e-4)',
             f'{"case":<14}{"min":>5}{"largest move":>26}{"largest gap":>26}{"gap/move":>10}']
    for r in rows['open']:
        lines.append(f'{r["case"]:<14}{r["minutes"]:>5}{100*r["move"]:>12.3f} ({r["move_at"]:>11}){100*r["dev"]:>12.4f} ({r["dev_at"]:>11}){100*r["rel"]:>9.2f}%')
    lines += ['', 'closed loop (output rows, in bands)', f'{"case":<14}{"h":>5}{"largest move":>24}{"gap":>22}{"at the end":>12}']
    for r in rows['closed']:
        lines.append(f'{r["case"]:<14}{r["hours"]:>5.1f}{r["move"]:>12.1f} ({r["move_at"]:>9}){r["dev"]:>10.2f} ({r["dev_at"]:>9}){r["end"]:>12.2f}')
    if rows['failed']:
        lines += ['', 'the native code broke down (not compared): '+', '.join(rows['failed'])]
    return '\n'.join(lines)


if __name__ == '__main__':
    print(table(summary(load(sys.argv[1] if len(sys.argv) > 1 else DATA))))

"""BSM1 of the engine against the published benchmark and an independent implementation (docs/sim.md, 'TEP and BSM1').

    python -m aio.sim.validate.bsm            # the table
    python -m aio.sim.validate.bsm record     # write data/bsm1_dry.npz again (needs `pip install bsm2-python`)

Steady state: the open-loop case under the constant influent against the published table. Dynamics, by the
benchmark's protocol: from that steady state the dry-weather influent file for fourteen days, then the same
file again, evaluated on days 7-14 of the second pass. Against (a) the averages printed in the benchmark
report (Alex et al., Benchmark Simulation Model no. 1, 2008, appendix 4, the Matlab-Simulink results) and
(b) the trajectories of bsm2-python 0.0.16 (BSD-3; an independent port of the IWA reference that integrates
the units one after another at one-minute steps), run by the same protocol.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from aio.sim import plants_bsm as B
from aio.sim.engine import Mech

ROOT = Path(__file__).resolve().parents[3]
INFLUENT = ROOT/'data'/'bsm1'/'dryinfluent.csv'
DATA = Path(__file__).parent/'data'/'bsm1_dry.npz'
F = torch.float64
EVAL = 7.                                                            # the evaluation week starts here (d)
# what is compared: name -> (stream, column among the 13 components, or 13 for the solids)
SERIES = dict(SNH_eff=('eff', B.SNH), SNO_eff=('eff', B.SNO), TSS_eff=('eff', 13), SS_eff=('eff', B.SS), SO_eff=('eff', B.SO), SALK_eff=('eff', B.SALK),
              SO5=('r5', B.SO), SNH5=('r5', B.SNH), XBA5=('r5', B.XBA), XBH5=('r5', B.XBH), SNO2=('r2', B.SNO), TSS_under=('under', 13))
# the report's effluent averages 'based on load', dry weather, open loop, days 7-14
REPORT = dict(SNH_eff=4.7589, SNO_eff=8.8238, TSS_eff=12.9917, SS_eff=.97352, SO_eff=.74639, SALK_eff=4.4562)
STREAMS = ('y_in1', 'y_out1', 'y_out2', 'y_out3', 'y_out4', 'y_out5', 'y_out5_r', 'ys_in', 'ys_out', 'ys_eff')


def steady():
    """Largest relative gap to the published open-loop steady state: reactors, settler solids."""
    P = B.bsm1(influent=True)
    x = P.nominal()[0][0]
    pub = torch.tensor(B.PUBLISHED['reactors'], dtype=F)
    r = ((x[:65].reshape(5, 13)-pub).abs()/pub.abs().clamp_min(1e-3)).max()
    tss = torch.tensor(B.PUBLISHED['tss'], dtype=F)
    s = ((x[65:].reshape(B.NL, 8)[:, 7]-tss).abs()/tss).max()
    return dict(reactors=float(r), settler=float(s), SNH5=float(x[4*13+B.SNH]), TSS_eff=float(x[65+7]), TSS_under=float(x[65+8*(B.NL-1)+7]))


def influent():
    a = np.loadtxt(INFLUENT, delimiter=',')
    return a[:, 0], a[:, 1:14], a[:, 15]                             # time (d), the 13 components, flow


def engine(passes=2, days=None, substeps=None):
    """The dry-weather file on the engine from the open-loop steady state, a decision per influent sample (15 min),
    `passes` times over (days: only so much of every pass). Returns the series at the end of every decision: [pass, time, series]."""
    t, cin, q = influent()
    n = len(t) if days is None else int(round(days*96))
    P = B.bsm1(influent=True)
    M = Mech([P], n=1, dtype=F)
    x, w, p = P.nominal()
    names = [v['name'] for v in P.ws]
    cols = [names.index(f'in_{B.NAMES[c]}') for c in range(13)]+[names.index('Qin')]
    X = []
    for _ in range(passes):
        for k in range(n):
            w = w.clone()
            w[0, cols] = torch.tensor(np.r_[cin[k], q[k]])
            x = M.advance(x, w, p, substeps=substeps)
            X.append(x[0].numpy().copy())
    X = np.array(X)
    R, S = X[:, :65].reshape(-1, 5, 13), X[:, 65:].reshape(-1, B.NL, 8)
    solids = B.X2TSS*R[:, 4, [B.XI, B.XS, B.XBH, B.XBA, B.XP]].sum(-1)

    def stream(j):                                                   # a settler layer as a stream: its solubles, the particulates of reactor 5 by its solids
        v = R[:, 4, :]*(S[:, j, 7]/solids)[:, None]
        for a, c in enumerate(B.SOL):
            v[:, c] = S[:, j, a]
        return np.c_[v, S[:, j, 7]]
    st = dict(eff=stream(0), under=stream(B.NL-1), r5=np.c_[R[:, 4], solids], r2=np.c_[R[:, 1], solids])
    Y = np.stack([st[where][:, col] for where, col in SERIES.values()], -1).reshape(passes, n, len(SERIES))
    return dict(t=t[:n]+(t[1]-t[0]), q=q[:n]-B.QW, y=Y)


def record(out=DATA, settle=200., passes=2):
    """The same on bsm2-python: to steady state under the constant influent first (bsm2-python starts from a plant
    of ones), then the dry-weather file at one-minute steps, `passes` times. Kept at the influent samples."""
    from bsm2_python.bsm1_ol import BSM1OL
    row = np.r_[B.CONST_IN, B.X2TSS*sum(B.CONST_IN[c] for c in (B.XI, B.XS, B.XBH, B.XBA, B.XP)), B.QIN0, 15., np.zeros(5)]
    const = np.c_[np.arange(0., settle+1., 1.), np.tile(row, (int(settle)+1, 1))]
    a = BSM1OL(data_in=const, timestep=15/1440.)
    for i in range(len(a.timesteps)):
        a.step(i)
    start = np.array([dict(eff=a.ys_eff, under=a.ys_out, r5=a.y_out5, r2=a.y_out2)[where][col] for where, col in SERIES.values()])
    Y = []
    for _ in range(passes):
        m = BSM1OL(data_in=str(INFLUENT), timestep=1/1440., evaltime=np.array([EVAL, 13.9]))
        for k in range(1, 6):
            getattr(m, f'reactor{k}').y0 = getattr(a, f'reactor{k}').y0.copy()
        m.settler.ys0 = a.settler.ys0.copy()
        for q in STREAMS:
            setattr(m, q, getattr(a, q).copy())
        n = len(m.timesteps)
        for i in range(n):
            m.step(i)
        at = np.arange(14, n, 15)                                    # the state after the 15th minute of every influent sample
        st = dict(eff=m.ys_eff_all, under=m.ys_out_all, r5=m.y_out5_all, r2=m.y_out2_all)
        Y.append(np.stack([st[where][at, col] for where, col in SERIES.values()], -1))
        a = m
    np.savez_compressed(out, t=(m.simtime[at]+1/1440.).astype(np.float32), y=np.array(Y, dtype=np.float32), start=start, names=np.array(list(SERIES)))
    return out


def summary(run=None, path=DATA):
    """Per series over the evaluation week of the last pass: load-weighted means of the engine and of bsm2-python (and
    the report's where it prints one), the largest and the mean gap between the two trajectories as shares of the swing."""
    ref = np.load(path)
    assert [str(q) for q in ref['names']] == list(SERIES)
    run = run or engine(passes=len(ref['y']))
    n = min(len(ref['t']), len(run['t']))
    assert np.allclose(ref['t'][:n], run['t'][:n], atol=1e-3)
    on = run['t'][:n] >= EVAL
    q = run['q'][:n][on]
    rows = []
    for i, name in enumerate(SERIES):
        a, b = run['y'][-1, :n, i][on], ref['y'][-1, :n, i][on].astype(float)
        swing = b.max()-b.min()
        rows.append(dict(name=name, engine=float((a*q).sum()/q.sum()), ref=float((b*q).sum()/q.sum()), report=REPORT.get(name), swing=float(swing),
                         gap=float(np.abs(a-b).max()/swing), mean_gap=float(np.abs(a-b).mean()/swing)))
    return rows


def table(rows, st):
    lines = [f'steady state against the published table: reactors {100*st["reactors"]:.2f} %, settler solids {100*st["settler"]:.2f} % (largest relative gap)',
             '', 'dry weather, days 7-14 of the second pass (means weighted by the effluent flow; gaps between the trajectories as shares of the swing)',
             f'{"series":<12}{"engine":>10}{"report":>10}{"bsm2-python":>13}{"swing":>10}{"largest gap":>13}{"mean gap":>10}']
    for r in rows:
        rep = f'{r["report"]:>10.4f}' if r['report'] is not None else f'{"":>10}'
        lines.append(f'{r["name"]:<12}{r["engine"]:>10.4f}{rep}{r["ref"]:>13.4f}{r["swing"]:>10.3f}{100*r["gap"]:>12.2f}%{100*r["mean_gap"]:>9.2f}%')
    return '\n'.join(lines)


if __name__ == '__main__':
    if sys.argv[1:] == ['record']:
        print('wrote', record())
    else:
        print(table(summary(), steady()))

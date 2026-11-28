"""BSM1 as engine data (aio.sim.plants_bsm): the published steady state, the benchmark's dry-weather dynamics
against an independent implementation, the APC plant and the episodes on it."""
import numpy as np
import pytest
import torch

from aio.sim import core as C, control as CT, mech as ME, plants_bsm as B
from aio.sim.engine import Mech
from aio.sim.validate import bsm as V

F = torch.float64


@pytest.fixture(scope='module')
def apc():
    P = B.bsm1()
    return P, Mech([P], n=1, dtype=F)


def test_the_open_loop_steady_state_is_the_published_one():
    st = V.steady()
    assert st['reactors'] < .005 and st['settler'] < .005, st        # the table prints three to four digits
    assert abs(st['SNH5']-1.7333) < 2e-3 and abs(st['TSS_eff']-12.497) < 2e-2 and abs(st['TSS_under']-6394.) < 1.
    P = B.bsm1(influent=True)
    f = Mech([P], n=1, dtype=F).f(*P.nominal())[0]
    scale = torch.tensor([s['scale'] for s in P.xs], dtype=F)
    assert float((f/scale).abs().max()) < 1e-6                       # per day


def test_two_days_of_dry_weather_follow_the_reference():
    ref = np.load(V.DATA)
    run = V.engine(passes=1, days=2.)
    n = run['y'].shape[1]
    a, b = run['y'][0], ref['y'][0, :n].astype(float)
    swing = b.max(0)-b.min(0)
    gap = np.abs(a-b).max(0)/swing
    assert gap.max() < .06 and (np.abs(a-b).mean(0)/swing).max() < .02, dict(zip(V.SERIES, gap.round(3)))


def test_the_apc_plant_rests_at_its_targets_and_is_stable(apc):
    P, M = apc
    x, w, p = P.nominal()
    y = M.y(x, w, p)[0]
    assert float((y[:4]-torch.tensor(B.TARGET, dtype=F)).abs().max()) < 1e-6
    scale = torch.tensor([s['scale'] for s in P.xs], dtype=F)
    assert float((M.f(x, w, p)[0]/scale).abs().max()) < 1e-6
    assert float(torch.linalg.eigvals(M.linearise(x, w, p)[2][0]).real.max()) < -.05     # per day
    rows = {o['name']: i for i, o in enumerate(P.ys)}
    assert all(float(y[rows[q]]) < P.ys[rows[q]]['hi'] for q in ('SNHe', 'TNe', 'TSSe'))  # inside the effluent limits at the nominal point


def test_the_production_step_is_within_half_a_band_of_a_fine_one(apc):
    P, M = apc
    x, w, p = P.nominal()
    slew = [v['slew'] for v in P.ws[:4]]
    band = torch.tensor([o['band'] for o in P.ys], dtype=F)
    xa, xb, worst = x, x, torch.zeros(P.ny, dtype=F)
    for k in range(16):                                              # the MVs at half their largest move, turning round every four decisions; the flow up 10 %
        w = w.clone()
        for j in range(4):
            w[0, j] += .5*(1 if (k//4) % 2 == 0 else -1)*(1 if j % 2 == 0 else -1)*slew[j]
        if k == 8:
            w[0, 4] *= 1.1
        xa, xb = M.advance(xa, w, p), M.advance(xb, w, p, substeps=192)
        worst = torch.maximum(worst, ((M.y(xa, w, p)-M.y(xb, w, p)).abs()/band)[0])
    assert float(worst.max()) < .5, worst.tolist()


def test_episodes_on_the_plant_rank_the_references():
    prog = C.build(4, 5, mech=1., plants=ME.pool(['bsm1']))
    res = {q: CT.rollout(prog, c, 240) for q, c in dict(hold=CT.Hold(), teacher=CT.Teacher()).items()}       # two and a half days
    assert not any(bool(v['failed'].any()) for v in res.values())
    assert float(res['teacher']['score'].mean()) < .2*float(res['hold']['score'].mean())

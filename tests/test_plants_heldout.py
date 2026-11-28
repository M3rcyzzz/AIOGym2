"""The held-out plants as engine data (aio.sim.plants_heldout): each against the older environment it is written
from, its step size, the pool's guard, and the episodes on them.

The older environments are no longer in the repository. What they gave at the points of the first test is in
tests/data/older.npz, written at the last commit that had them (tests/data/freeze_older.py there)."""
from pathlib import Path

import numpy as np
import pytest
import torch

from aio.sim import core as C, control as CT, mech as ME, plants as PL, plants_heldout as H
from aio.sim.engine import Mech

F = torch.float64
OLD = ('boiler', 'wo', 'mma', 'cryst')
OLDER = np.load(Path(__file__).parent/'data'/'older.npz')


@pytest.mark.parametrize('name', OLD)
def test_equations_match_older_environment(name):
    """At 48 random points of states, moves and parameters."""
    x, u, fo, yo = (torch.as_tensor(OLDER[f'{name}/eq/{q}']) for q in ('x', 'u', 'f', 'y'))
    p = dict(zip(OLDER[name+'/params'].tolist(), torch.as_tensor(OLDER[name+'/eq/p']).unbind(-1)))
    n = len(x)
    P = H.HELDOUT[name]()
    m = Mech([P], n=n, dtype=F)
    w, pp = torch.zeros(n, P.nw, dtype=F), torch.zeros(n, max(P.np, 1), dtype=F)
    w[:, [i for i, q in enumerate(P.ws) if q['kind'] == 'mv']] = u
    for i, q in enumerate(P.ws):
        if q['kind'] == 'load':
            w[:, i] = p[q['name']]
    for i, q in enumerate(P.ps):
        pp[:, i] = p[q['name']]
    fn, yn = m.f(x, w, pp), m.y(x, w, pp)
    assert float(((fn-fo).abs()/fo.abs().amax(0, keepdim=True).clamp_min(1e-12)).max()) < 1e-12
    assert float(((yn-yo).abs()/yo.abs().amax(0, keepdim=True).clamp_min(1e-12)).max()) < 1e-12
    assert [o['name'] for o in P.ys] == OLDER[name+'/cv_names'].tolist() and [o['band'] for o in P.ys] == OLDER[name+'/band'].tolist()
    x0, w0, p0 = P.nominal()
    assert float((Mech([P], n=1, dtype=F).f(x0, w0, p0)[0]/torch.as_tensor(OLDER[name+'/xscale'])).abs().max()) < 5e-3      # declared at the nominal point (as printed, four digits)


@pytest.mark.parametrize('name', OLD)
def test_the_production_step_is_within_half_a_band_of_a_fine_one(name):
    P = H.HELDOUT[name]()
    m = Mech([P], n=1, dtype=F)
    x, w, p = P.nominal()
    mv = [i for i, q in enumerate(P.ws) if q['kind'] == 'mv']
    band = torch.tensor([o['band'] for o in P.ys], dtype=F)
    xa, xb, worst = x, x, 0.
    for k in range(40):                                              # every MV at half its largest move, turning round every five decisions
        w = w.clone()
        for j in mv:
            w[0, j] += .5*(1 if (k//5) % 2 == 0 else -1)*(1 if j % 2 == 0 else -1)*P.ws[j]['slew']
        xa, xb = m.advance(xa, w, p), m.advance(xb, w, p, substeps=32)
        worst = max(worst, float(((m.y(xa, w, p)-m.y(xb, w, p)).abs()/band).max()))
    assert worst < .5, worst


def test_held_out_plants_are_given_only_for_evaluation():
    assert set(PL.HELDOUT) == set(H.HELDOUT) | {'vam', 'gaslib', 'refrig', 'badgwell'} and set(H.HELDOUT) == set(OLD) | {'boiler2'} and not set(PL.HELDOUT) & (set(PL.PLANTS) | set(PL.LARGE))
    with pytest.raises(AssertionError):
        ME.pool(['boiler'])
    assert [P.name for P in ME.pool(['wo', 'cryst'], heldout=True)] == ['wo', 'cryst']
    assert {P.name for P in ME.pool()} == set(PL.PLANTS)             # the default pool is the training plants


def test_episodes_on_the_plants_rank_the_references():
    prog = C.build(24, 5, mech=1., plants=ME.pool(list(OLD), heldout=True))
    res = {q: CT.rollout(prog, c) for q, c in dict(hold=CT.Hold(), teacher=CT.Teacher()).items()}
    assert not bool(res['teacher']['failed'].any())
    names = prog.mech.names()
    assert set(names) == set(OLD)
    for name in OLD:
        i = torch.tensor([j for j, q in enumerate(names) if q == name])
        assert float(res['teacher']['score'][i].mean()) < .4*float(res['hold']['score'][i].mean()), name

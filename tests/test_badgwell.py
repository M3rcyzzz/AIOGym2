"""Badgwell's fired heater as engine data (aio.sim.plants_badgwell): the case's own model and run, held out."""
import numpy as np
import pytest

from aio.sim import mech as ME, plants as PL, plants_badgwell as H
from aio.sim.validate import badgwell as V


def test_the_case_model_and_run():
    r = V.summary(np.load(V.DATA))
    assert r['step_rel'] < 1e-3                                       # step response coefficients of all 25 pairs
    assert max(r['run_dev']) < .02                                     # the logged MPC run replayed, deg F / BPH


def test_held_out_with_the_case_limits():
    P = H.heater()
    o = {q['name']: q for q in P.ys}
    assert [q['name'] for q in P.ys if q.get('track')] == ['TO', 'FO', 'DT'] and o['TS1']['hi'] == o['TS2']['hi'] == 918.
    assert [(w['lo'], w['hi']) for w in P.ws if w['kind'] == 'mv'] == [(85., 135.), (85., 135.), (92., 103.)]
    assert 'badgwell' in PL.HELDOUT and 'badgwell' not in PL.PLANTS
    with pytest.raises(AssertionError):
        ME.pool(['badgwell'])

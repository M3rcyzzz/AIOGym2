"""The IFAC PID 2018 refrigeration benchmark as engine data (aio.sim.plants_refrig): the benchmark's operating point,
held out, and the reference runs."""
import pytest
import torch

from aio.sim import mech as ME, plants as PL, plants_refrig as R
from aio.sim.engine import Mech
from aio.sim.validate import refrig as V

F = torch.float64


def test_the_benchmarks_operating_point():
    P = R.refrig_raw()
    M = Mech([P], n=1, dtype=F)
    y = dict(zip([o['name'] for o in P.ys], M.y(*P.nominal())[0].tolist()))
    assert abs(y['Tsec_out']-(-22.15)) < .05 and abs(y['TSH']-14.65) < .5        # main document, Table 4


def test_held_out_and_tracked():
    P = R.refrig()
    assert [o['name'] for o in P.ys if o.get('track')] == ['Tsec_out', 'TSH'] and len([w for w in P.ws if w['kind'] == 'load']) == 7
    assert 'refrig' in PL.HELDOUT and 'refrig' not in PL.PLANTS
    with pytest.raises(AssertionError):
        ME.pool(['refrig'])


def test_the_reference_runs():
    rows = {(r['run'], r['row']): r for r in V.summary(V.load())}
    for run in ('A', 'B'):                                           # B was not used to identify the components
        for q, g in V.GAP.items():
            r = rows[run, q]
            assert r['mean'] < g*r['range'], r

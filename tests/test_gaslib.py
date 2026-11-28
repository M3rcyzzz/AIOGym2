"""GasLib-24 as engine data (aio.sim.plants_gaslib): morgen's discretisation, the exam plant, held out, and the
comparison with morgen's recorded runs."""
import pytest
import torch

from aio.sim import mech as ME, plants as PL, plants_gaslib as G
from aio.sim.engine import Mech
from aio.sim.validate import gaslib as V

F = torch.float64


def test_morgens_pieces():
    edges, nodes, supply, demand = G.refine(G.NOM_MORGEN)
    assert len(edges) == 363 and supply == [25, 26, 27] and demand == [28, 29, 30, 31, 32]      # morgen's own count at its default
    edges, nodes, *_ = G.refine()
    assert len(edges) == 95 and len(nodes) == 91 and [e[0] for e in edges].count('C') == 3


def test_the_exam_plant_rests_and_is_held_out():
    P = G.gaslib()
    M = Mech([P], n=1, dtype=F)
    x, w, p = P.nominal()
    assert P.nx == 186 and float(M.f(x, w, p).abs().max()) < 1e-5            # its nominal point is the steady state
    y = dict(zip([o['name'] for o in P.ys], M.y(x, w, p)[0].tolist()))
    for k, st in enumerate(G.STATIONS):
        assert st[2] < y[f'n_{st[0]}'] < st[3]                                # every station inside its map's speed range
    assert [o['name'] for o in P.ys if o.get('track')] == ['p_exit1', 'p_exit2', 'p_exit3', 'p_transit']
    assert 'gaslib' in PL.HELDOUT and 'gaslib' not in PL.PLANTS
    with pytest.raises(AssertionError):
        ME.pool(['gaslib'])


@pytest.mark.skipif(not V.DATA.exists(), reason='morgen runs are recorded where Octave is (aio/sim/validate/gaslib.py record)')
def test_the_engine_follows_morgen():
    c = V.load()['trr154']
    c = {k: (v[:361] if getattr(v, 'ndim', 0) and len(v) == 1441 else v) for k, v in c.items()}     # six hours of the day
    nat, eng = V.compare(c)
    scale = abs(nat[0])
    assert float((abs(eng-nat)[:, 3:]/scale[3:]).max()) < V.GAP                                  # exit pressures
    assert float(abs(eng-nat)[:, :3].max()/nat[0, :3].sum()) < .01                               # entry flows, against the supply

"""The vinyl acetate plant as engine data (aio.sim.plants_vam): the paper's base, the regulatory layer, held out,
and the comparison with the native code's recorded trajectories."""
import pytest
import torch

from aio.sim import mech as ME, plants as PL, plants_vam as V
from aio.sim.engine import Mech
from aio.sim.validate import vam as VV, vam_native_record as R

F = torch.float64


def test_the_base_is_the_papers():
    g, at = VV.paper()
    assert g < VV.PAPER_GAP, (g, at)
    P = V.vam_raw()
    assert P.nx == 247 and P.nw == 30 and len(V.MEAS) == 43
    y = Mech([P], n=1, dtype=F).y(*P.nominal())[0]
    assert abs(float(y[43])-161.473) < .01                            # peak reactor temperature at the base, below the 200 C interlock


def test_the_layer_holds_the_base_and_it_is_held_out():
    P = V.vam()
    M = Mech([P], n=1, dtype=F)
    x, w, p = P.nominal()
    y0 = M.y(x, w, p)[0]
    for _ in range(3):
        x = M.advance(x, w, p)
    band = torch.tensor([o['band'] for o in P.ys], dtype=F)
    assert float(((M.y(x, w, p)[0]-y0).abs()/band).max()) < 1.      # half an hour at the base: within a band
    assert [v['name'] for v in P.ws if v['kind'] == 'mv'] == list(V.MVS) and P.dt == 600.
    assert 'vam' in PL.HELDOUT and 'vam' not in PL.PLANTS and 'vam' not in PL.LARGE
    with pytest.raises(AssertionError):
        ME.pool(['vam'])


def test_the_recorders_layer_is_the_plants():
    assert R.LOOPS == V.LOOPS and R.FIXED == V.FIXED and R.SP0 == {q: v[3] for q, v in V.APC.items()}
    assert (R.ANALYSED, R.STAGES, R.DEAD, R.LAG) == (V.ANALYSED, V.AN_STAGES, V.AN_DEAD, V.LAG)


@pytest.mark.skipif(not VV.DATA.exists(), reason='the native trajectories are recorded where the C code is (aio/sim/validate/vam_native_record.py)')
def test_the_engine_follows_the_native_code():
    data = VV.load()
    some = {q: data[q] for q in ('ol_base', 'ol_u2', 'ol_u7', 'ol_c2h6', 'cl_Trct165', 'cl_hac') if q in data}
    rows = VV.summary(some)
    for r in rows['open']:
        assert r['dev'] < VV.OPEN_GAP, r
    for r in rows['closed']:
        assert r['dev'] < VV.CLOSED_GAP, r

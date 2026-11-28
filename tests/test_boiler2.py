"""The drum boiler of Astrom and Bell 2000 (plants_heldout.boiler2): saturation properties, the paper's step
responses, held out, and the plant data of the paper."""
import pytest
import torch

from aio.sim import mech as ME, plants as PL, plants_heldout as H
from aio.sim.engine import Mech
from aio.sim.validate import boiler as B

F = torch.float64


def test_saturation_is_iapws():
    T, rw, rs, hw, hs = H.sat(10.)                                    # IAPWS-IF97 at 10 MPa: 584.149 K, 688.4 / 55.46 kg/m3, 1407.9 / 2725.5 kJ/kg
    assert abs(T-584.149) < .01 and abs(rw/688.42-1) < 1e-3 and abs(rs/55.46-1) < 1e-3
    assert abs(hw/1407.87e3-1) < 1e-3 and abs(hs/2725.47e3-1) < 1e-3


def test_the_papers_fuel_step():
    """Fig. 6 of the paper (medium load, +10 MW at 50 s): the level swells about 1 cm within 15 s, the pressure climbs."""
    P = H.boiler2(raw=True)
    P.dt, P.substeps = 1., 1
    M = Mech([P], n=1, dtype=F)
    x, w, p = P.nominal()
    assert float(M.f(x, w, p).abs().max()) < 1e-9                    # the equilibrium of boiler2_point
    w2 = w.clone()
    w2[0, [q['name'] for q in P.ws].index('Q')] += 10.
    l0 = float(M.y(x, w, p)[0, 1])
    L = []
    for _ in range(150):
        x = M.advance(x, w2, p)
        L.append(float(M.y(x, w2, p)[0, 1])-l0)
    k = max(range(150), key=lambda i: L[i])
    assert .008 < L[k] < .014 and k < 20 and float(x[0, 1]) > 8.7


def test_it_is_held_out_and_runs():
    assert 'boiler2' in PL.HELDOUT and 'boiler2' not in PL.PLANTS
    with pytest.raises(AssertionError):
        ME.pool(['boiler2'])
    P = ME.pool(['boiler2'], heldout=True)[0]
    assert [o['name'] for o in P.ys if o.get('track')] == ['p', 'qs', 'level']


def test_the_plant_data():
    rows = B.summary({'9': B.load()['9']})                            # feedwater perturbations at medium load (one run: the fit takes minutes)
    assert rows[0]['p_rms'] < B.P_GAP*rows[0]['p_swing'] and rows[0]['l_rms'] < B.L_GAP*rows[0]['l_swing'], rows

"""Tennessee Eastman as engine data (aio.sim.plants_tep): the published base case, the regulatory layer, the
episodes on it, and the comparison with the native code where its recorded trajectories are in the tree."""
import pytest
import torch

from aio.sim import core as C, control as CT, mech as ME, plants as PL, plants_tep as T
from aio.sim.engine import Mech
from aio.sim.validate import tep as V, tep_native_record as R

F = torch.float64
# Downs & Vogel (1993), the base case: XMEAS 1-41 as printed
PUBLISHED = ('0.25052 3664.0 4509.3 9.3477 26.902 42.339 2705.0 75.000 120.40 0.33712 80.109 50.000 2633.7 25.160 50.000 3102.2 22.949 65.731 230.31 '
             '341.43 94.599 77.297 32.188 8.8933 26.383 6.8820 18.776 1.6567 32.958 13.823 23.978 1.2565 18.579 2.2633 4.8436 2.2986 0.01787 0.83570 '
             '0.09858 53.724 43.828').split()


@pytest.fixture(scope='module')
def apc():
    P = T.tep()
    return P, Mech([P], n=1, dtype=F)


def test_the_base_case_is_the_published_one():
    b = T.base()
    for i, (v, s) in enumerate(zip(b['xm'], PUBLISHED)):
        digits = len(s.split('.')[1])
        assert abs(v-float(s)) <= .51*10**-digits, (i+1, v, s)       # every printed digit
    assert max(abs(a-e) for a, e in zip(b['energy'], (T.YY[8], T.YY[17], T.YY[26], T.YY[35]))) < 1e-9      # the temperatures carry the routine's energies
    assert abs(b['cost']-170.6) < .5                                 # $/h, the paper's operating cost of the base case
    P = T.tep_raw()
    M = Mech([P], n=1, dtype=F)
    f = M.f(*P.nominal())[0]
    assert float(f.abs().max()) < 5e-3                               # a steady state as far as the routine's printed state goes (per hour)
    assert torch.equal(T.from_routine([T.YY])[0], P.nominal()[0][0])


def test_the_layer_holds_the_base_case_and_is_stable(apc):
    P, M = apc
    x, w, p = P.nominal()
    xs, _, ok = M.steady(w, p)
    assert bool(ok.all())
    y0, ys = M.y(x, w, p)[0], M.y(xs, w, p)[0]
    band = torch.tensor([o['band'] for o in P.ys], dtype=F)
    assert float(((ys-y0).abs()/band).max()) < .05                   # the layer's steady state is the base case
    A = M.linearise(xs, w, p)[2][0]
    assert float(torch.linalg.eigvals(A).real.max()) < -.05          # per hour: the slowest mode settles in about half a day
    names = [o['name'] for o in P.ys]
    assert [names[i] for i, o in enumerate(P.ys) if o.get('track')] == ['prod', 'G', 'yA', 'yAC'] and [v['name'] for v in P.ws[:4]] == list(T.MVS4)


def test_the_layer_rejects_the_published_disturbances(apc):
    P, M = apc
    x, w0, p = P.nominal()
    names = [v['name'] for v in P.ws]
    rows = {o['name']: i for i, o in enumerate(P.ys)}
    for idv in (2, 3, 4, 5, 7):                                      # IDV 1 and 6 need the A feed ratio moved: the APC's work
        w = w0.clone()
        for q, v in V.IDV[idv].items():
            w[0, names.index(q)] = v
        xk = x
        for _ in range(72):                                          # a day
            xk = M.advance(xk, w, p)
        y = M.y(xk, w, p)[0]
        assert bool(torch.isfinite(y).all())
        for q in ('P', 'T', 'L', 'Ls', 'Lc'):
            o = P.ys[rows[q]]
            assert o.get('lo', -1e9) < float(y[rows[q]]) < o.get('hi', 1e9), (idv, q, float(y[rows[q]]))
        assert abs(float(y[rows['T']])-T.base()['xm'][8]) < .2, idv   # production and composition drift with the feed: the APC's rows


def test_the_production_step_is_within_half_a_band_of_a_fine_one(apc):
    P, M = apc
    x, w, p = P.nominal()
    slew = [v['slew'] for v in P.ws[:4]]
    band = torch.tensor([o['band'] for o in P.ys], dtype=F)
    xa, xb, worst = x, x, torch.zeros(P.ny, dtype=F)
    for k in range(36):                                              # every MV at its largest move, turning round every six decisions
        w = w.clone()
        for j in range(4):
            w[0, j] += (1 if (k//6) % 2 == 0 else -1)*(1 if j % 2 == 0 else -1)*slew[j]
        xa, xb = M.advance(xa, w, p), M.advance(xb, w, p, substeps=120)
        worst = torch.maximum(worst, ((M.y(xa, w, p)-M.y(xb, w, p)).abs()/band)[0])
    assert float(worst.max()) < .5 and float(worst[:4].max()) < .1, worst.tolist()      # the tracked rows: a tenth of a band


def test_episodes_on_the_plant_rank_the_references():
    assert set(PL.LARGE) >= {'tep', 'bsm1'} and not set(PL.LARGE) & set(PL.PLANTS)      # named, never in the default pool
    prog = C.build(8, 5, mech=1., plants=ME.pool(['tep']))
    res = {q: CT.rollout(prog, c) for q, c in dict(hold=CT.Hold(), pi=CT.PI(), teacher=CT.Teacher()).items()}
    assert not any(bool(v['failed'].any()) for v in res.values())
    sc = {q: float(v['score'].mean()) for q, v in res.items()}
    assert sc['teacher'] < .5*sc['pi'] < .5*sc['hold'], sc


def test_the_recorders_layer_is_the_plants():
    b = T.base()['xm']
    assert R.TUNE == T.TUNE and tuple(R.U0) == tuple(T.YY[38:50]) and max(abs(a-e) for a, e in zip(R.XM0, b)) < 1e-9
    assert R.MV0 == dict(Fp=100., Eadj=0., rA=b[0], rC=b[3], Psp=b[6], Tsp=b[8], Lsp=b[7], recycle=T.YY[42], steam=T.YY[46], agitator=T.YY[49])


@pytest.mark.skipif(not V.DATA.exists(), reason='the native trajectories are recorded on the training machine (aio/sim/validate/tep_native_record.py)')
def test_the_engine_follows_the_native_code():
    data, names = V.load()
    rows = V.summary(data, names, fine=False)
    assert len(rows['open']) == 20 and len(rows['closed']) == 16
    for r in rows['open']:                                           # open loop: valve steps and IDV 1-7 from the native start, to the trip where there is one
        assert r['dev'] < V.OPEN_GAP, r
    for r in rows['closed']:                                         # under the layer, a day each, at the production step (the steps are 12-20 times the largest move)
        assert r['dev'] < V.CLOSED_GAP, r
    fine = V.summary({q: data[q] for q in ('cl_Fp-15', 'cl_idv2')}, names)['closed']
    assert all(r['dev_fine'] < V.FINE_GAP for r in fine), fine      # at ten-second steps what is left is the noise of the native runs

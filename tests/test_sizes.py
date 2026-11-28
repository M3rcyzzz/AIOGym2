"""A batch has its own size (prior.SIZES): the packet tells its shape, the larger sizes hold plants of several
units in a chain, and everything that runs on the default size runs on them."""
import numpy as np
import pytest
import torch

from aio import interface as IF
from aio.sim import core as C, control as CT, mech as ME, model as MD, prior as PR

F = torch.float64


def test_a_packet_tells_its_shape():
    assert IF.size() == IF.SIZE == 234 and [IF.size(*s) for s in IF.SHAPES] == [234, 588, 1656]
    assert [v[:2] for v in PR.SIZES.values()] == list(IF.SHAPES) and PR.SIZES['12x5'] == PR.DIMS
    for ncv, nmv in IF.SHAPES:
        assert IF.shape(IF.size(ncv, nmv)) == (ncv, nmv) == IF.shape(IF.size(ncv, nmv), nmv)
    assert IF.shape(234, 6) == (11, 6)                            # a length alone does not tell: the action's width does
    with pytest.raises(ValueError):
        IF.shape(235)
    with pytest.raises(ValueError):
        IF.shape(234, 4)
    for lib in (np, torch):
        obs = lib.arange(2*3*588).reshape(2, 3, 588)
        cv, mv, link = IF.split(obs)
        assert tuple(cv.shape) == (2, 3, 24, IF.CF) and tuple(mv.shape) == (2, 3, 10, IF.MF) and tuple(link.shape) == (2, 3, 24, 10)
        assert int(cv[1, 2, 5, 3]) == int(obs[1, 2, 5*IF.CF+3]) and int(mv[0, 1, 9, 5]) == int(obs[0, 1, 24*IF.CF+9*IF.MF+5])
        assert int(link[1, 0, 23, 9]) == int(obs[1, 0, -1])


def test_the_sizes_of_a_batch_come_from_its_tensors():
    for name, dims in PR.SIZES.items():
        ncv, nmv, nd = dims
        p, tgt, _ = PR.draw_plant(PR.Rng(1), 4, dims=dims)
        assert PR.sizes(p) == PR.sizes_of(*dims) == (ncv, nmv, nd, nd, 2*nd, nmv+2*nd)
        assert tuple(p['K'].shape[1:]) == (ncv, nmv+2*nd) and tuple(p['link'].shape[1:]) == (ncv, nmv)
    assert PR.nzmax(*PR.DIMS) == MD.NZMAX and MD.NR == IF.NMV+MD.NZMAX
    assert C.PRIV == 630 and sum(int(np.prod(v)) for v in C.priv_fields(24, 10, 4).values()) == 2340


def test_a_larger_plant_is_several_units_in_a_chain():
    """The units a larger size holds do not fit the next smaller one; most pass the check of every operating
    point as drawn; a unit feels the units upstream more than the ones downstream; and the gains of the MVs
    are sparser than a single unit's."""
    r = PR.Rng(3)
    p, tgt, ro = PR.draw_plant(r, 1500, dims=PR.SIZES['24x10'])
    nm, nz = p['nm'], p['zon'].sum(1)
    assert int(nm.min()) == 6 and int(nm.max()) == 10 and int(nz.min()) >= 6 and int(nz.max()) == PR.nzmax(*PR.SIZES['24x10']) == 14
    assert 1.3 < float((nz/nm).mean()) < 1.7                      # more zone CVs than MVs (tests/test_apc.py)
    assert bool((p['arch'] == PR.PLANT).all())                    # six MVs or more: two units at least
    ok, _ = PR.operable(p, tgt)
    assert float(ok.double().mean()) > .8
    K = p['K'][:, :, :10]
    proc = (p['prc'][:, :, None]*p['mvm'][:, None, :]) > 0
    dens = float(((K != 0) & proc).double().sum()/proc.double().sum())
    q, _, _ = PR.draw_plant(PR.Rng(3), 1500)
    one = (q['prc'][:, :, None]*q['mvm'][:, None, :]) > 0
    assert .35 < dens < .65 < float(((q['K'][:, :, :5] != 0) & one).double().sum()/one.double().sum())
    own = (K.abs()*p['link']).sum(1)                              # an MV published with a CV it does not move: as rare as on a single unit (the badly paired 3 %, prior._paired)
    assert float(((own == 0) & (p['mvm'] > 0)).any(1)[ok].double().mean()) < .01
    # a plant settles more slowly than its units: the decision interval is chosen for the plant
    assert float(p['T'].median()) > float(q['T'].median()) and float(p['T'].max()) <= PR.TAU_MAX


def test_the_points_of_the_largest_size_pass_when_checked_again():
    """A constrained point is published once and checked wherever the episode is used: its best steady point is
    the same whichever batch computes it (the levels are rows of the solve, not weights: tests/test_sim.py)."""
    segs, tgts, infos, seg_k, kinds, delay, ramp = PR.draw_episode(PR.Rng(9), 40, exc_p=.6, dims=PR.SIZES['48x20'])
    con = 0
    for e, (p, t) in enumerate(zip(segs, tgts)):
        live = seg_k[:, e] < PR.HORIZON
        assert bool(PR.point_ok(p, t)[live].all())
        con += int((PR.constrained(p, t) & live).sum())
    assert con > 30


@pytest.mark.parametrize('size,n', [('24x10', 6), ('48x20', 3)])
def test_the_larger_sizes_run(size, n):
    """Draw, packet, privileged vector, labels, the formula PI and the teacher on a larger batch: the shapes
    are the size's, nothing is lost, and the teacher is ahead of holding; at 24x10 well ahead of the formula
    PI too (at 48x20 not on every plant yet: docs/sim.md section 15)."""
    torch.manual_seed(0)
    prog = C.build(n, 7, dims=size)
    ncv, nmv, nd = PR.SIZES[size]
    assert (prog.ncv, prog.nmv, prog.nd, prog.nq) == (ncv, nmv, nd, nd) and prog.size == IF.size(ncv, nmv)
    st = C.reset(prog)
    obs = C.observe(prog, st, 0)
    assert tuple(obs.shape) == (n, prog.size) and bool(torch.isfinite(obs).all())
    cv, mv, link = IF.split(obs)
    assert torch.equal(link.sum(1), prog.c['mvm']) and torch.equal(mv[:, :, 5], prog.c['mvm'])       # every MV in use has its CV
    assert torch.equal(cv[:, :, 11], prog.c['rowm'])
    priv = C.privileged(prog, st, 0)
    assert tuple(priv.shape) == (n, prog.priv) and bool(torch.isfinite(priv).all())
    a = 0
    for q, shp in C.priv_fields(ncv, nmv, nd).items():
        k = int(np.prod(shp))
        if q == 'lab':
            assert torch.equal(priv[:, a:a+k].view(n, *shp), prog.labels(0))
        a += k
    assert a == prog.priv and tuple(prog.labels(0).shape) == (n, nmv, ncv*MD.NLAB)
    assert tuple(C.teacher(prog, st, 0).shape) == (n, nmv)
    with torch.no_grad():
        s = {q: float(CT.rollout(prog, c, steps=360)['score'].sum()) for q, c in (('hold', CT.Hold()), ('pi', CT.PI()), ('teacher', CT.Teacher()))}
    assert np.isfinite(list(s.values())).all() and s['teacher'] < .7*s['hold'] and (size == '48x20' or s['teacher'] < .7*s['pi']), s


def test_a_mechanistic_plant_sits_in_a_larger_batch():
    """The plants of the engine in a batch of a larger size: the same plants, more of their limits published
    (a larger size has room for more zone rows); the nine-MV Tennessee Eastman plant fits where five MVs did not."""
    prog = C.build(3, 5, mech=1., plants=ME.pool(['qtank', 'vdv']), dims='24x10')
    assert tuple(prog.c['link'].shape[1:]) == (24, 10) and int(prog.c['nm'].max()) <= 2
    st = C.reset(prog)
    obs = C.observe(prog, st, 0)
    with torch.no_grad():
        for k in range(30):
            st, out = C.step(prog, st, C.teacher(prog, st, k), k)
    assert tuple(obs.shape) == (3, 588) and bool(torch.isfinite(st['y']).all()) and not bool(st['fail'].any())
    with pytest.raises(AssertionError, match='does not fit'):
        C.build(2, 5, mech=1., plants=ME.pool(['tep9']), teacher=False)

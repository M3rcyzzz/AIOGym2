"""Heavy process changes (prior.Menu.heavy): what a valve move does becomes something else while the steady point
stays where it is; the new gains and time constants are drawn about those of the plant as commissioned."""
import math

import pytest
import torch

from aio.sim import control as CT, core as C, prior as PR

M, BIG = PR.Menu, PR.BIG
ONLY = M(rate=(0., 0., 0., 1., 0.), heavy=1., exc=0.)            # process events alone, every one a heavy change


def gains(p):
    NMV = PR.sizes(p)[1]
    on = (p['trk'][:, :, None] > 0) & (p['mvm'] > 0)[:, None, :]
    return p['K'][:, :, :NMV]*on, on


def own(p):
    NMV = PR.sizes(p)[1]
    return (p['K'][:, :, :NMV]*(p['link'][:, :, :NMV] > 0)).sum(-1)


@pytest.fixture(scope='module')
def drawn():
    return PR.draw_episode(PR.Rng(3), 192, menu=ONLY)


def test_without_it_nothing_changes():
    a, b = C.build(24, 3, teacher=False, menu=M()), C.build(24, 3, teacher=False, menu=M(heavy=0.))
    assert torch.equal(a.P, b.P) and torch.equal(a.T, b.T) and torch.equal(a.seg_k, b.seg_k)


def test_the_steady_point_stays_and_the_gains_are_others(drawn):
    segs, tgts, infos, seg_k, kinds, delay, ramp = drawn
    rel, flips, n = [], [], 0
    for e in range(1, len(segs)):
        m = (seg_k[:, e] < BIG) & (kinds[:, e] == PR.PROC)
        if not bool(m.any()):
            continue
        a, b = segs[e-1], segs[e]
        assert bool(PR.point_ok(b, tgts[e])[m].all())
        assert float(((infos[e]['un']-infos[e-1]['un']).abs()*a['mvm'])[m].max()) < 1e-9      # every valve where it stood
        assert torch.equal(tgts[e][m], tgts[e-1][m]) and float((infos[e]['y']-infos[e-1]['y']).abs()[m].max()) < 1e-9
        assert torch.equal(a['dead'][m], b['dead'][m]) and torch.equal(a['dlev'][m], b['dlev'][m])     # no dead time moves (the kind kept for the exams)
        (Ka, on), (Kb, _) = gains(a), gains(b)
        rel.append(((Kb-Ka).flatten(1).norm(dim=1)/Ka.flatten(1).norm(dim=1))[m])
        cross = on & ~(a['link'][:, :, :PR.sizes(a)[1]] > 0) & (Ka != 0)
        flips.append(((torch.sign(Ka) != torch.sign(Kb)) & cross)[m].float().sum()/cross[m].float().sum().clamp_min(1))
        n += int(m.sum())
    rel = torch.cat(rel)
    assert n > 300 and float(rel.median()) > .3 and float(torch.quantile(rel, .9)) > .7     # (the usual kinds: median 0.1 where the gains change at all)
    assert float(torch.stack(flips).mean()) > .08                 # cross effects change sign


def test_heavy_changes_stay_about_the_plant_as_commissioned():
    a = C.build(96, 5, teacher=False, menu=M(heavy=1.))
    tails = [a.tail['lin']]
    for w in range(1, 4):
        a = a.next()
        tails.append(a.tail['lin'])
    first = tails[0]['base'][0]
    k0, far = own(first), []
    for t in tails:
        p = t['cur'][0]
        ratio = (own(p)/torch.where(k0 != 0, k0, torch.ones_like(k0)))[(k0 != 0) & (p['trk'] > 0) & (p['rst'] == 0)]
        assert bool((ratio > .2).all()) and bool((ratio < 4.).all())                # the paired gain keeps its sign and its size (the usual kinds are not there to drift it)
        (K0, _), (K1, _) = gains(first), gains(p)
        far.append(float(((K1-K0).flatten(1).norm(dim=1)/K0.flatten(1).norm(dim=1)).median()))
        tau = (torch.log(p['p1r'].abs().clamp_min(1e-30)/first['p1r'].abs().clamp_min(1e-30)).abs()*(p['trk'][:, :, None] > 0)*(p['rst'][:, :, None] == 0)).amax((1, 2))
        assert float(tau.max()) <= math.log(PR.HEAVY_TIME)+1e-9                      # every time constant within 3 of the commissioned one
    assert far[-1] < 1.5*far[0]+.1 and far[0] > .2                # as far from it after four windows as after one


def test_the_references_on_heavy_changes():
    a = C.build(32, 3, menu=M(heavy=1.))
    late, fixed, hold = (CT.rollout(a, c) for c in (CT.Teacher('late'), CT.Teacher('fixed'), CT.Hold()))
    assert not bool(late['failed'].any()) and late['score'].mean() < .4*hold['score'].mean()
    assert fixed['score'].mean() > 1.3*late['score'].mean()       # the model of commissioning is worth less

"""A unit at a constrained point stays there while ordinary events go on (prior.Menu.ride): what took it there
holds, the events are drawn on the unit as it would be without it, and every point they give is checked."""
import pytest
import torch

from aio.sim import control as CT, core as C, prior as PR

M, BIG = PR.Menu, PR.BIG
RIDE = M(ride=.8, exc=.3)


def new(p):
    m = p.seg_k < BIG
    m[:, :p.s0+1] = False
    return m


def ordinary(k):
    return (k >= 0) & (k <= PR.PRIO)


@pytest.fixture(scope='module')
def drawn():
    keep = {}
    return PR.draw_episode(PR.Rng(3), 192, menu=RIDE, keep=keep), keep


def test_without_it_the_next_event_brings_the_unit_back():
    a, b = C.build(24, 3, teacher=False, menu=M()), C.build(24, 3, teacher=False, menu=M(ride=0.))
    assert torch.equal(a.P, b.P) and torch.equal(a.T, b.T) and torch.equal(a.seg_k, b.seg_k)
    assert torch.equal(a.con, PR.is_excursion(a.kinds) & (a.seg_k < BIG))        # a constrained point is where an event took the unit, and nowhere else
    old = (a.kinds[:, 1:] == PR.RETURN) & (a.kinds[:, :-1] == PR.OVER_LOAD)       # the allowance of the point left holds while its load is on the way back
    assert torch.equal((a.hold_k > a.seg_k)[:, 1:] & (a.seg_k[:, 1:] < BIG), old & (a.seg_k[:, 1:] < BIG))


def test_ordinary_events_at_a_constrained_point(drawn):
    (segs, tgts, infos, seg_k, kinds, delay, ramp), keep = drawn
    live = seg_k < BIG
    con = torch.stack([PR.constrained(p, t) for p, t in zip(segs, tgts)], 1) & live
    ok = torch.stack([PR.point_ok(p, t) for p, t in zip(segs, tgts)], 1)
    assert bool(ok[live].all())                                   # every point passes the one check, operable or constrained
    there = con & ordinary(kinds)
    assert int(there.sum()) > 150 and all(int((there & (kinds == q)).sum()) > 0 for q in (PR.SP, PR.PROC, PR.LOAD, PR.DV, PR.PRIO))
    back = kinds == PR.RETURN
    assert not bool((con & back).any())                           # taken back, the unit is at an operable point
    usual = PR.draw_episode(PR.Rng(3), 192, menu=M(exc=.3))
    assert int(con.sum()) > 1.5*int((PR.is_excursion(usual[4]) & (usual[3] < BIG)).sum())
    okh, _ = PR.operable(*keep['home'][:2])
    assert bool((okh | ~keep['away']).all()) and int(keep['riding'].sum()) > 20   # the unit without what holds it is an operable point all along


def test_what_took_the_unit_there_holds(drawn):
    (segs, tgts, infos, seg_k, kinds, delay, ramp), _ = drawn
    seen = free = 0
    for e in range(1, len(segs)-1):
        a, b = segs[e], segs[e+1]
        stays = PR.is_excursion(kinds[:, e]) & ordinary(kinds[:, e+1]) & (seg_k[:, e+1] < BIG)
        m = stays & (kinds[:, e] == PR.MV_LIMIT)
        assert torch.equal(a['ulo'][m], b['ulo'][m]) and torch.equal(a['uhi'][m], b['uhi'][m])      # the narrowed limit
        m = stays & (kinds[:, e] == PR.MV_OUT)
        assert torch.equal(a['ulo'][m], b['ulo'][m]) and torch.equal(a['uhi'][m], b['uhi'][m])      # the valve out of service
        m = stays & (kinds[:, e] == PR.ZONE_LIMIT) & (kinds[:, e+1] != PR.LIMIT)
        assert torch.equal(a['lo'][m], b['lo'][m]) and torch.equal(a['hi'][m], b['hi'][m])
        m = stays & (kinds[:, e] == PR.OVER_LOAD) & (kinds[:, e+1] != PR.LOAD) & (kinds[:, e+1] != PR.DV)
        assert torch.equal(a['dlev'][m], b['dlev'][m])                                              # the load that had moved
        seen += int(stays.sum())
        free += int((stays & ~PR.constrained(b, tgts[e+1])).sum())
    assert seen > 60 and 0 < free < seen                          # some events leave it free while it holds (the limit no longer binds), most do not


def test_the_stay_goes_on_into_the_next_window():
    a = C.build(96, 5, teacher=False, menu=M(ride=.9, exc=.4))
    riding = a.tail['lin']['riding']
    assert int(riding.sum()) > 15
    b, c = a.next(7), a.next(7, menu=M())
    first = lambda p: p.kinds[:, 3]                               # noqa: E731
    had = riding & (b.seg_k[:, 3] < BIG)
    assert float((first(b)[had] != PR.RETURN).float().mean()) > .6
    assert bool((first(c)[riding & (c.seg_k[:, 3] < BIG)] == PR.RETURN).all())    # a window without it: the next event brings them back


def test_the_allowance_of_the_point_left_holds_while_a_load_is_on_its_way():
    p = C.build(96, 5, teacher=False, menu=RIDE)
    con, k = p.con, p.kinds
    dlev = p.params(torch.zeros(p.n, dtype=torch.long))['dlev']
    seen = 0
    for e in range(1, p.nseg):
        s = torch.full((p.n,), e, dtype=torch.long)
        moved = (p.params(s)['dlev'] != p.params(s-1)['dlev']).any(-1) | (k[:, e] == PR.PROC) | (k[:, e] == PR.BOTH)
        m = con[:, e-1] & (p.seg_k[:, e] < BIG)
        assert torch.equal((p.hold_k[:, e] > p.seg_k[:, e])[m], moved[m])
        seen += int((m & moved & ordinary(k[:, e])).sum())
    assert seen > 10 and dlev.shape[0] == p.n


def test_the_references_on_such_a_stay():
    a = C.build(24, 3, menu=RIDE)
    r, hold = CT.rollout(a, CT.Teacher('late')), CT.rollout(a, CT.Hold())
    assert not bool(r['failed'].any()) and bool(torch.isfinite(r['score']).all())
    assert r['score'].mean() < .35*hold['score'].mean()
    b = a.next()
    r2 = CT.rollout(b, CT.Teacher('late'), st=r['st'])
    assert not bool(r2['failed'].any()) and r2['score'].mean() < .35*CT.rollout(b, CT.Hold(), st=r['st'])['score'].mean()

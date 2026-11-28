"""What changes during a window is a menu (prior.Menu): how often each kind of event comes, how large, in what order."""
import pytest
import torch

from aio.sim import control as CT, core as C, mech as ME, prior as PR

M, BIG = PR.Menu, PR.BIG
N = 48


def build(menu=None, nseg=PR.NSEG, n=N, seed=3):
    return C.build(n, seed, teacher=False, menu=menu, nseg=nseg)


def new(p):
    """The events of this window: (n, nseg)."""
    m = p.seg_k < BIG
    m[:, :p.s0+1] = False
    return m


def count(p, *kinds):
    return float(sum(((p.kinds == q) & new(p)).sum() for q in kinds))/p.n


def at(p, e):
    s = torch.full((p.n,), e, dtype=torch.long)
    return p.params(s), p.task(s)


def across(p, kinds, what):
    """what(before, after) (n,) over the events of the given kinds, in one vector."""
    out = []
    for e in range(p.s0+1, p.nseg):
        m = new(p)[:, e] & torch.isin(p.kinds[:, e], torch.tensor(kinds))
        out.append(what(at(p, e-1), at(p, e))[m])
    return torch.cat(out)


@pytest.fixture(scope='module')
def plain():
    return build(M())


def test_a_plain_menu_is_the_usual_mix(plain):
    usual = build()
    assert abs(float(new(plain).sum())-float(new(usual).sum()))/N < 1.
    assert all(abs(count(plain, k)-count(usual, k)) < .6 for k in (PR.SP, PR.DV, PR.LOAD, PR.PROC))


def test_each_kind_comes_at_its_own_rate(plain):
    assert count(build(M(rate=(4, 1, 1, 1, 1)), 40), PR.SP) > 2.5*count(plain, PR.SP)
    assert count(build(M(rate=(1, 4, 1, 1, 1)), 40), PR.DV) > 2.5*count(plain, PR.DV)
    assert count(build(M(rate=(1, 1, 4, 1, 1)), 40), PR.LOAD) > 2.5*count(plain, PR.LOAD)
    assert count(build(M(rate=(1, 1, 1, 4, 1)), 40), PR.PROC) > 2.5*count(plain, PR.PROC)
    assert count(build(M(rate=(1, 1, 1, 0, 1))), PR.PROC, PR.BOTH) == 0
    only = build(M(rate=(0, 1, 0, 0, 0), exc=0.))
    assert count(only, PR.DV) > 0 and count(only, PR.SP, PR.LOAD, PR.PROC, PR.BOTH, PR.LIMIT, PR.PRIO) == 0


def test_pace_and_a_window_that_holds_more(plain):
    fast = build(M(pace=2.), 24)
    assert float(new(fast).sum()) > 1.6*float(new(plain).sum())
    gap = fast.seg_k[:, 1:].clamp_max(BIG)-fast.seg_k[:, :-1]
    assert bool((gap[fast.seg_k[:, 1:] < BIG] >= PR.MIN_GAP).all())
    t = C.build(12, 3, menu=M(pace=2.), nseg=24)                  # the teacher and the rollout take a window of any number of segments
    r, hold = CT.rollout(t, CT.Teacher('late')), CT.rollout(t, CT.Hold())
    assert r['seg_settled'].shape[1] == 24 and not bool(r['failed'].any())
    assert r['score'].mean() < .5*hold['score'].mean()


def test_a_quiet_window_between_two_busy_ones():
    a = build(n=24)
    b = a.next(menu=M(rate=(0, 0, 0, 0, 0), wander=0.))
    assert not bool(new(b).any())
    assert float(b.c['ou_s'].abs().sum()) == 0 and float(b.c['sin_amp'].abs().sum()) == 0
    assert bool((b.kinds[:, 3:] == -1).all())
    c = b.next(menu=M())                                          # the events come back, and the wander with them
    assert float(new(c).sum())/24 > 4 and float(c.c['ou_s'].abs().sum()) > 0
    assert torch.equal(c.P[:, 0], a.P[:, 0])


def test_setpoints_move_one_at_a_time_in_turn():
    p = build(M(rate=(1, 0, 0, 0, 0), turn=True, exc=0.))
    can = (p.c['trk'] > 0) & (p.c['rst'] == 0)
    seen = 0
    for i in range(p.n):
        rows, order = can[i].nonzero().flatten().tolist(), []
        for e in range(1, p.nseg):
            if p.seg_k[i, e] < BIG and p.kinds[i, e] == PR.SP:
                d = ((at(p, e)[1]['tgt'][i]-at(p, e-1)[1]['tgt'][i]).abs() > 1e-9).nonzero().flatten().tolist()
                assert len(d) == 1
                order.append(d[0])
        assert order == [rows[j % len(rows)] for j in range(len(order))]
        seen += len(order) > len(rows) > 1
    assert seen > 5                                               # units that went round once and started again
    half = build(M(rate=(1, 0, 0, 0, 0), turn=.5, exc=0.))
    moved = across(half, [PR.SP], lambda a, b: ((b[1]['tgt']-a[1]['tgt']).abs() > 1e-9).float().sum(-1))
    many = (half.c['trk'] > 0).sum(1) > 1
    assert .3 < float((moved == 1).float().mean()) < .9 and bool(many.any())     # half of the moves alone (and those of a unit with one CV)


def test_a_pulse_is_taken_back_by_the_next_event():
    p = build(M(rate=(0, 1, 1, 1, 0), pulse=1., exc=0.), 24)
    k, seen = p.kinds, 0
    for e in range(1, p.nseg-1):
        m = new(p)[:, e] & torch.isin(k[:, e], torch.tensor([PR.DV, PR.LOAD, PR.PROC])) & (p.seg_k[:, e+1] < BIG)
        assert bool((k[m, e+1] == PR.RETURN).all())
        assert torch.equal(p.P[m, e+1], p.P[m, e-1]) and torch.equal(p.T[m, e+1], p.T[m, e-1])
        assert not torch.equal(p.P[m, e], p.P[m, e-1]) or not bool(m.any())
        seen += int(m.sum())
    assert seen > N
    assert count(build(M(rate=(0, 1, 1, 1, 0), exc=0.), 24), PR.RETURN) == 0


def test_loads_move_together():
    def share(p):
        return across(p, [PR.DV, PR.LOAD], lambda a, b: ((b[0]['dlev'] != a[0]['dlev']).float().sum(-1)/(p.c['dact'] > 0).float().sum(-1).clamp_min(1.)))
    assert share(build(M(rate=(0, 1, 1, 0, 0), joint=1., exc=0.))).min() == 1.
    assert share(build(M(rate=(0, 1, 1, 0, 0), exc=0.))).mean() < .8


def test_kinds_can_be_left_out():
    rate = (1, 1, 1, 3, 1)

    def dead(p):
        return across(p, [PR.PROC, PR.BOTH], lambda a, b: (b[0]['dead'] != a[0]['dead']).flatten(1).any(1).float())
    assert dead(build(M(rate=rate), 24)).mean() > .1
    left = build(M(rate=rate, drop=('dead', 'priority')), 24)
    assert dead(left).sum() == 0 and count(left, PR.PRIO) == 0 and count(left, PR.PROC) > 2


def test_sizes(plain):
    rate = (1, 1, 1, 0, 0)
    usual, small = build(M(rate=rate, exc=0.)), build(M(rate=rate, move=.3, step=.3, exc=0.))

    def move(p):
        return across(p, [PR.SP], lambda a, b: ((b[1]['tgt']-a[1]['tgt']).abs()/p.c['bandf']*(p.c['trk'] > 0)).amax(-1)).median()

    def step(p):
        return across(p, [PR.DV, PR.LOAD], lambda a, b: (b[0]['dlev']-a[0]['dlev']).abs().amax(-1)).median()
    assert move(small) < .6*move(usual) and step(small) < .5*step(usual)
    rate = (1, 1, 1, 3, 1)

    def gain(p):
        g = across(p, [PR.PROC], lambda a, b: (b[0]['K']-a[0]['K']).flatten(1).norm(dim=1)/a[0]['K'].flatten(1).norm(dim=1).clamp_min(1e-30))
        return g[g > 1e-6].median()
    assert gain(build(M(rate=rate, proc=3.), 24)) > 1.4*gain(build(M(rate=rate), 24))
    assert float(build(M(wander=2.)).c['ou_s'].abs().sum()) == pytest.approx(2*float(plain.c['ou_s'].abs().sum()))


def test_plants_of_the_engine_take_a_menu():
    plants = ME.pool(['qtank', 'hex'])
    a = C.build(8, 7, mech=1., plants=plants, teacher=False, menu=M(rate=(1, 1, 1, 3, 1), proc=2., pulse=.3, turn=True), nseg=20)
    usual = C.build(8, 7, mech=1., plants=plants, teacher=False)
    assert count(a, PR.PROC) > count(usual, PR.PROC)
    r = CT.rollout(a, CT.Hold(), steps=240)
    assert bool(torch.isfinite(r['score']).all())
    b = a.next(menu=M(rate=(0, 0, 0, 0, 0)))
    assert not bool(new(b).any()) and b.mech.names() == a.mech.names()
    r = CT.rollout(b, CT.Hold(), steps=120, st=r['st'])
    assert bool(torch.isfinite(r['score']).all())


def test_small_plants():
    """Two or three valves and none to spare; everything else as usual."""
    p = C.build(64, 3, teacher=False, shape=PR.SMALL)
    nm = (p.c['mvm'] > 0).sum(1)
    assert set(nm.tolist()) == {2, 3} and float(p.c['rst'].sum()) == 0
    assert bool(((p.c['trk'] > 0).sum(1) == nm).all())            # as many tracked CVs as valves
    assert float((p.c['dvr'].sum(1) > 0).float().mean()) > .3 and float((p.c['zon'].sum(1) > 0).float().mean()) > .3
    assert float(new(p).sum())/64 > 5
    q = p.next()
    assert torch.equal(q.c['mvm'], p.c['mvm'])


def test_the_same_disturbances_measured_and_not():
    """A pair: the same plants, events and noise; in one the packet has the measured disturbances, in the
    other their rows are not there."""
    a, b = C.build(24, 3, teacher=False), C.build(24, 3, teacher=False, blind=True)
    assert torch.equal(a.P, b.P) and torch.equal(a.seg_k, b.seg_k)
    ra, rb = CT.rollout(a, CT.Hold(), steps=200), CT.rollout(b, CT.Hold(), steps=200)
    assert torch.equal(ra['st']['yt'], rb['st']['yt'])            # the plants do the same
    oa, ob = C.observe(a, ra['st'], 200), C.observe(b, rb['st'], 200)
    NCV, NMV = a.ncv, a.nmv
    from aio import interface as IF
    dv = a.c['dvr'] > 0
    ca, cb = oa[:, :NCV*IF.CF].view(-1, NCV, IF.CF), ob[:, :NCV*IF.CF].view(-1, NCV, IF.CF)
    assert bool(dv.any()) and float(ca[dv].abs().sum()) > 0 and float(cb[dv].abs().sum()) == 0
    assert torch.equal(ca[~dv], cb[~dv])
    la, lb = oa[:, -NCV*NMV:].view(-1, NCV, NMV), ob[:, -NCV*NMV:].view(-1, NCV, NMV)
    assert float(lb[dv].abs().sum()) == 0 and torch.equal(la[~dv], lb[~dv])
    assert b.next().blind and not a.next().blind

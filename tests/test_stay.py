"""A unit run on over several episodes, on the side of the network and the trainer (the plants' side: tests/test_long.py):
aio.net.Policy `stay` keeps its memory from one window to the next and its control forgets; aio.train `Stay` reads what
staying is worth; `Lane.onto` takes a lane to the next window of its plants with everything carried."""
import argparse

import torch

from aio import net as N, scale as SC, train as T
from aio.sim import core as C, control as CT, prior as PR


def test_the_network_keeps_its_memory_over_the_windows_and_the_control_forgets():
    torch.manual_seed(0)
    net = N.AIO(slow=2000.).eval()
    a = C.build(6, 3)
    b = C.build(6, 4, after=a.tail)
    own = N.Policy(net, stay=True, read=True, shadow=T.SHADOW)
    r = CT.rollout(a, own)
    h, hb = own.h.clone(), own.hb.clone()
    assert h.shape[-1] == N.STATE+N.SSLOW and float(h.abs().sum()) > 0 and not torch.equal(h, hb)   # (the shadow ran the last decisions alone)
    again = torch.tensor([False, True, False, False, False, False])
    own.restart(again)
    fresh = own.forgetful()
    own.resume(b)
    assert torch.equal(own.h, h*(~again).float()[None, :, None])           # nothing cleared but what started again
    fresh.resume(b)
    assert torch.equal(fresh.h, hb*(~again).float()[None, :, None])
    assert set(own.readout()) == {'id_own', 'id_cross', 'id_own_near', 'id_cross_near', 'id_own_quiet', 'id_cross_quiet'}
    rb = CT.rollout(b, own, st=r['st'])                                     # the rollout resumes it: the memory it had goes into the first decision
    read = own.readout()                                                    # all of the window; near an event; in the quiet after
    part = [read[q] for q in ('id_own_near', 'id_own_quiet') if read[q] is not None]
    assert part and min(part)-1e-9 <= read['id_own'] <= max(part)+1e-9
    plain = N.Policy(net)
    rp = CT.rollout(b, plain, st=r['st'])                                   # without `stay`: cleared, another rollout
    assert not torch.equal(rb['st']['u'], rp['st']['u'])
    own.reset(b)
    assert float(own.h.abs().sum()) == 0                                    # a rollout from a program's start clears it


def test_what_staying_is_worth_is_read_per_episode():
    torch.manual_seed(0)
    net = N.AIO()
    stay = T.Stay(4, 11, 3, 'cpu')
    m = stay.meta()
    assert m['links'] == 3 and all(len(m[q]) == 3 for q in ('hold', 'best', 'pi', 'dmc', 'teacher'))
    assert all(x > 60. for x in m['teacher'])                              # the teacher goes on holding the units it took over
    out = stay(net)
    assert len(out['pts']) == 3 and out['fresh'][0] is None and all(x is not None for x in out['fresh'][1:])
    assert net.training and out['gain'] is not None and len(out['id_cross']) == 3 and len(out['id_cross_near']) == 3
    # the moves are watched without the network being taken for one that starts anew: the points are those of the network on its own
    own, st, pts = N.Policy(net, stay=True), None, []
    net.eval()
    with torch.no_grad():
        for j, prog in enumerate(stay.progs):
            if j:
                own.restart(st['fail'])
            r = CT.rollout(prog, own, st=st)
            st = r['st']
            pts.append(round(SC.points(r['score'].tolist(), stay.hold[j], stay.best[j]), 2))
    net.train()
    assert pts == out['pts'] and len(out['flip']) == 3 and all(0. <= x <= 1. for x in out['flip']) and len(out['move']) == 3
    # the plants with a level among their scored rows, and the others, each on its own scale
    lev = T.levels(stay.progs[0])
    assert out['n_level'] == int(lev.sum()) and len(out['pts_level']) == 3 and len(out['pts_other']) == 3
    assert (out['pts_level'][0] is None) == (out['n_level'] == 0) and (out['pts_other'][0] is None) == (out['n_level'] == 4)
    c = stay.progs[0].c
    assert torch.equal(lev, ((c['integ'] > 0) & ((c['trk'] > 0) | (c['zon'] > 0))).any(-1))
    assert not bool(T.levels(C.build(2, 5, mech=1.)).any())                 # an episode on a plant of the engine: none


def test_the_exams_are_read_like_a_stay(tmp_path):
    torch.manual_seed(0)
    net = N.AIO()
    exams = T.exams_of(['pulse', 'quiet'], 4, 3, 'cpu', tmp_path)
    assert exams['quiet'].first == 2 and exams['pulse'].first == 0 and not exams['pulse'].fresh
    m = exams['pulse'].meta()
    assert T.Stay(4, 0, 3, 'cpu', progs=exams['pulse'].progs, fresh=False).meta() == m       # the references kept are those the stay runs itself
    assert T.exams_of(['pulse'], 4, 3, 'cpu', tmp_path)['pulse'].meta() == m and (tmp_path/'pulse-4x3-cpu.json').exists()   # (read back)
    assert m['links'] == 3 and m['shadow'] == 0 and all(len(m[q]) == 3 for q in ('hold', 'best', 'pi', 'dmc', 'teacher'))
    out = T.read_exams(exams, net)
    assert set(out) == {'pulse', 'quiet'} and all(len(o['pts']) == 3 and 'fresh' not in o and o['mean'] is not None for o in out.values())
    assert out['quiet']['mean'] == out['quiet']['pts'][2]                  # the windows that lead up to the quiet are not in its mean
    per = {}                                                               # --per: the scores themselves, window by window and plant by plant,
    assert T.read_exams(exams, net, per) == out and set(per) == set(out)   # beside every reference's: the points are theirs
    e = exams['pulse']
    assert set(e.per[0]) == {'hold', 'pi', 'dmc', 'teacher'} and len(per['pulse']) == 3 and len(per['pulse'][0]) == 4 and len(e.level) == 4
    assert T.kinds(e.progs[0]) == ['lin']*len(e.level)                     # (the kind of plant of every episode: an exam of the engine's plants is read by kind)
    assert [round(SC.points(per['pulse'][j], e.per[j]['hold'], SC.envelope(e.per[j])), 2) for j in range(3)] == out['pulse']['pts']


def test_a_lane_goes_on_to_the_next_window_of_its_plants():
    gen = torch.Generator().manual_seed(1)
    a = argparse.Namespace(known=.25, critic_known=.75, slow=2000.)
    p0 = C.build(5, 3)
    lane = T.Lane(p0, a, gen)
    lane.left = 1
    lane.st = CT.rollout(p0, CT.Hold())['st']
    lane.k, lane.ha, lane.hc, lane.prev = PR.HORIZON, lane.ha+1., lane.hc+2., lane.prev+.5
    lane.hs = torch.ones(5, N.SSTAT+N.SSLOW)
    assert lane.done
    lane.st['fail'][2] = True
    u = lane.st['u'].clone()
    p1 = C.build(5, 4, after=p0.tail)
    known = lane.known.clone()
    lane.onto(p1)
    keep = torch.tensor([1., 1., 0., 1., 1.])
    assert lane.prog is p1 and lane.k == 0 and lane.age == 1 and lane.left == 0 and not lane.done
    assert torch.equal(lane.ha[0, :, 0], keep) and torch.equal(lane.hc[0, :, 0], 2*keep) and torch.equal(lane.hs[:, 0], keep)
    assert torch.equal(lane.prev[:, 0], .5*keep) and torch.equal(lane.known, known) and float(lane.score.sum()) == 0
    assert torch.equal(lane.st['u'][keep > 0], u[keep > 0]) and not bool(lane.st['fail'].any())   # the plants where they were; the failed one started again


def test_the_menu_of_the_training_plants():
    assert T.menu_kw(None) == {} and T.menu_kw([]) == {}                   # none: the generator as it is
    kw = T.menu_kw(['rate=3,2,1,0,1', 'pace=2', 'turn=.5', 'drop=dead,limits'], nseg=30)
    m = kw['menu']
    assert kw['nseg'] == 30 and m.rate == (3., 2., 1., 0., 1.) and m.pace == 2. and m.turn == .5 and m.drop == ('dead', 'limits') and m.move == 1.
    kw = T.menu_kw(['nd=1,2', 'blind=1'], nseg=30)                         # what plants they are, with no menu: no nseg either
    assert kw == dict(shape=dict(nd=(1, 2)), blind=True)
    p = C.build(8, 5, **T.menu_kw(['nd=1,2', 'rate=1,3,1,1,1']))
    assert int(p.c['dvr'].sum(-1).min()) >= 1                              # every plant has a measured disturbance
    a = C.build(4, 3, **T.menu_kw(['rate=3,1,1,0,1', 'pace=2'], nseg=30))
    b = C.build(4, 4, after=a.tail, **T.menu_kw(['rate=3,1,1,0,1', 'pace=2'], nseg=30))
    plain = C.build(4, 3)
    events = lambda p: int((p.kinds >= 0).sum())                           # noqa: E731
    assert events(a) > events(plain) and b.n == 4
    r = CT.rollout(a, CT.Hold())
    assert torch.isfinite(CT.rollout(b, CT.Hold(), st=r['st'])['score']).all()


def test_a_new_batch_of_plants_takes_each_group_of_the_mix_by_its_own_chance():
    mix = T.mix_of(['2/3:nd=1,2+rate=1,3,1,1,1', '1/2:ride=.8+exc=.3', '1/3:shift=.2'])
    assert [round(p, 3) for p, _ in mix] == [.667, .5, .333] and mix[1][1] == ('ride=.8', 'exc=.3')
    base = ['pulse=0', 'drop=dead']
    draws = [T.mixed(base, mix, (7 << 32)+j) for j in range(3000)]
    assert draws[5] == T.mixed(base, mix, (7 << 32)+5)                     # the draw is the seed's own
    share = [sum(on[i] for _, on in draws)/len(draws) for i in range(3)]
    assert all(abs(x-p) < .04 for x, (p, _) in zip(share, mix))
    assert abs(sum(on[0]*on[1] for _, on in draws)/len(draws)-share[0]*share[1]) < .03    # one group independently of the other
    items, _ = next(d for d in draws if d[1] == (1, 1, 0))
    kw = T.menu_kw(items, 16, True)
    m = kw['menu']
    assert m.pulse == 0. and m.drop == ('dead',) and m.rate == (1., 3., 1., 1., 1.) and m.ride == .8 and m.exc == .3 and m.shift == 0.
    assert kw['shape'] == dict(nd=(1, 2)) and kw['nseg'] == 16
    assert next(d for d in draws if d[1] == (0, 0, 0))[0] == base          # no group came up: --menu alone
    assert T.menu_kw(None, 16, True) == dict(nseg=16) and T.menu_kw(None, 16) == {}    # the lanes that train get the room without a menu; a --plain lane stays the generator as it is
    try:
        T.mix_of(['1/2:warp=3'])
        assert False
    except AssertionError as e:
        assert 'warp' in str(e)


def test_a_batch_has_the_mechanistic_plants_of_one_group_and_a_builder_builds_them_by_name():
    """--mech-groups: without it every batch has --mech of its episodes on plants of every kind; with it a batch
    has any with the chance --mech-batches and then those of one group, taken by the groups' weights, the draw
    the seed's own. A batch drawn with the names (by a builder: the plants are built in its process) has plants
    of those kinds alone and is the batch drawn in place."""
    assert T.mech_of(.083, 1., (), 5) == (.083, None, None)
    groups = T.groups_of(['10:qtank,ph,vdv', 'column', '1/2:hex'])
    assert groups == ((10., ('qtank', 'ph', 'vdv')), (1., ('column',)), (.5, ('hex',)))
    draws = [T.mech_of(.33, .25, groups, (7 << 32)+j) for j in range(4000)]
    assert draws[9] == T.mech_of(.33, .25, groups, (7 << 32)+9)
    has = [d for d in draws if d[0] > 0]
    assert abs(len(has)/len(draws)-.25) < .03 and all(d == (0., None, None) for d in draws if d[0] == 0)
    assert all(d[0] == .33 and d[1] == groups[d[2]][1] for d in has)
    share = [sum(d[2] == g for d in has)/len(has) for g in range(3)]
    assert all(abs(x-w/11.5) < .04 for x, (w, _) in zip(share, groups))
    # drawn by an even sequence instead (a run: per kind of lane, `evenly`): the shares hold in every stretch of the run
    v = [T.evenly(k, 1, 'x') for k in range(400)]
    assert v == [T.evenly(k, 1, 'x') for k in range(400)] and v != [T.evenly(k, 1, 'y') for k in range(400)] and all(0 <= x < 1 for x in v)
    even = [T.mech_of(.33, .25, groups, None, x) for x in v]
    for part in (even[:200], even[200:], even[:40]):              # a share of .25*.5/11.5 (1 in 92): about twice in each half of 400
        n = [sum(d[2] == g for d in part) for g in range(3)]
        assert all(abs(x-len(part)*.25*w/11.5) < 2.5 for x, (w, _) in zip(n, groups))   # (drawn independently: give or take 6.5 for the first)
    assert all(d[0] == .33 and d[1] == groups[d[2]][1] for d in even if d[0] > 0) and all(d == (0., None, None) for d in even if d[0] == 0)
    assert T.mech_of(.33, 0., groups, None, 0.) == (0., None, None) and [T.pick([2., 1., 1.], u) for u in (0., .49, .5, .74, .75, .999)] == [0, 0, 1, 1, 2, 2]
    u = [T.evenly(k, 0, 'x') for k in range(400)]                 # the two coordinates even together: a share of v within a share of u
    assert abs(sum(a < 4/6 and b < .02 for a, b in zip(u, v))-400*4/6*.02) < 2
    try:
        T.groups_of(['1:qtank,nosuch'])
        assert False
    except AssertionError as e:
        assert 'nosuch' in str(e)
    # --bench: a batch with the kinds of the first group given, as a batch of a run that took that group
    assert T.bench_kinds(argparse.Namespace(mech=.25, mech_groups=['1:heater,bldg,amm', 'column'])) == ('heater', 'bldg', 'amm')
    assert T.bench_kinds(argparse.Namespace(mech=0., mech_groups=['column'])) is None and T.bench_kinds(argparse.Namespace(mech=.25, mech_groups=None)) is None
    seeds = iter(range(300, 310))
    ahead = T.Ahead(1, 6, lambda: next(seeds), threads=1)
    try:
        prog, kw, tag = ahead.own('k', lambda sd: (dict(mech=.5, names=('qtank', 'hex')), sd), 0)
    finally:
        ahead.close()
    here = T.drawn(6, 300, mech=.5, names=('qtank', 'hex'))
    assert tag == 300 and kw['names'] == ('qtank', 'hex') and prog.n == 6
    assert len(prog.mech.groups) == 1 and set(prog.mech.names()) <= {'qtank', 'hex'}      # one group of engines runs: the kinds named
    assert all(torch.equal(v, here.c[q]) for q, v in prog.c.items())
    plain = C.build(6, 300, mech=.5)                                       # the pool's kinds: other plants for the same seed
    assert any(not torch.equal(v, plain.c[q]) for q, v in prog.c.items() if v.shape == plain.c[q].shape)

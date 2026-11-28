"""Mechanistic episodes of the simulation core (aio.sim.mech): the generator on the engine's plants, the plant
step inside the core, the references on them."""
import pytest
import torch

from aio.interface import NCV, NMV
from aio.sim import core as C, control as CT, mech as ME, plants as PL, prior as PR, stay as ST

F = torch.float64


@pytest.fixture(scope='module')
def drawn():
    return ME.draw(PR.Rng(3, 'cpu'), 30, ME.pool(PL.FIRST))        # (the ten of the pool as it was: the draw these tests were written on)


@pytest.fixture(scope='module')
def prog():
    return C.build(24, 5, mech=1., plants=ME.pool(['qtank', 'vdv', 'bio', 'evap', 'furnace', 'hex']))


def groups(drawn):
    (segs, tgts, infos, seg_k, kinds, delay, ramp), S = drawn
    for g, (a, P, xs, ws) in zip(S.groups, S.parts):
        rows = torch.arange(a, a+g.k)
        yield g, rows, P, xs, ws


def test_every_operating_point_is_a_steady_state_of_the_plant_and_passes_the_generators_check(drawn):
    (segs, tgts, infos, seg_k, kinds, delay, ramp), S = drawn
    assert len(set(S.names())) >= 8                                  # the plants come in the proportions asked (equal)
    for e, (p, t) in enumerate(zip(segs, tgts)):
        assert bool(PR.point_ok(p, t).all()), e
    for g, rows, P, xs, ws in groups(drawn):
        tt = g.t
        for e in range(PR.NSEG):
            x, w, mp = xs[:, e], ws[:, e], P[:, e]
            y0 = g.M.y(x, w, mp)
            x1 = x
            for _ in range(3):
                x1 = g.M.advance(x1, w, mp)
            move = ((g.M.y(x1, w, mp)-y0).abs()/tt['oband'])[tt['orow'] < NCV]
            assert float(move.max()) < 1e-3, (g.M.plants[0].name, e, float(move.max()))   # a thousandth of a band in three decisions
            # the row values the generator holds are the plant's outputs there, and the tracked ones sit at the reachable targets
            v = g.rows((y0-tt['rlo'])/tt['rspan'])
            trk = segs[e]['trk'][rows] > 0
            assert float(((v-segs[e]['rtgt'][rows]).abs()/segs[e]['bandf'][rows])[trk].max()) < 1e-3


def test_the_local_model_is_the_plants_answer_to_a_small_move(drawn):
    (segs, tgts, infos, seg_k, kinds, delay, ramp), S = drawn
    p = segs[0]
    for g, rows, P, xs, ws in groups(drawn):
        tt = g.t
        x, w, mp = xs[:, 0], ws[:, 0], P[:, 0]
        still = ~(p['integ'][rows] > 0).any(-1)                      # a plant with a level has no steady state with every valve held
        if not bool(still.any()):
            continue
        ue = g.flows(w)
        x0, w0, _ = g.M.steady(g.win(ue, p['dlev'][rows]), mp, start=(x, w, mp))       # the same point, to the solver's full precision
        for j in range(NMV):
            on = still & (p['mvm'][rows, j] > 0)
            if not bool(on.any()):
                continue
            v, ok = [], on.clone()
            for h in (1e-4, -1e-4):                                  # a central difference: a one-sided one is off by the curvature (.5% at this size on the column and the tanks)
                u1 = ue.clone(); u1[:, j] += h
                x1, w1, conv = g.M.steady(g.win(u1, p['dlev'][rows]), mp, start=(x0, w0, mp))
                v.append(g.rows((g.M.y(x1, w1, mp)-tt['rlo'])/tt['rspan']))
                ok = ok & conv
            K = p['K'][rows][:, :, j]
            err = ((v[0]-v[1])/2e-4-K).abs()/(K.abs().amax(-1, keepdim=True)+1e-9)
            assert bool(ok[on].all()) and float(err[on].max()) < 5e-4, (g.M.plants[0].name, j, float(err[on].max()))


def test_published_signs_hold_and_the_model_holds_over_a_finite_move_at_every_point(drawn):
    (segs, tgts, infos, seg_k, kinds, delay, ramp), S = drawn
    for g, rows, P, xs, ws in groups(drawn):
        tt = g.t
        for e in range(0, PR.NSEG, 3):
            q = {key: v[rows] for key, v in segs[e].items()}
            assert bool(ME.Group.signs(q).all())
            x, w, mp = xs[:, e], ws[:, e], P[:, e]
            v = g.rows((g.M.y(x, w, mp)-tt['rlo'])/tt['rspan'])
            ok, R = g.margin(q, x, w, mp, v, torch.ones(g.k, dtype=torch.bool), ratios=True)
            assert bool(ok.all()), (g.M.plants[0].name, e)
            fin = torch.isfinite(R)
            assert float(R[fin].min()) >= 1/ME.MARGIN_RATIO-1e-9 and float(R[fin].max()) <= ME.MARGIN_RATIO+1e-9


def test_a_level_is_paired_with_the_valve_that_moves_it_most():
    # rows L2 (rate), X2, P2 against P100, F2, F200 of the evaporator: X2 answers to F2 alone at steady state, so pairing by
    # relative gain on the open levels would give the level to the cooling water
    G = torch.tensor([[[-11.9, -22.9, -3.25, 0, 0], [1.07, .56, -.127, 0, 0], [0., -2.4, 0., 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]], dtype=F)
    integ = torch.zeros(1, NCV, dtype=F); integ[0, 0] = 1.
    mvm = torch.tensor([[1., 1., 1., 0., 0.]], dtype=F)
    order, lam = ME.pairing(G, integ, mvm)
    assert order[0, :3].tolist() == [1, 0, 2]                        # P100 with P2, F2 with the level, F200 with X2 (through the level loop)
    assert bool((lam[0, :3] > PR.RGA_MIN).all())


def test_reach_refuses_a_point_the_plant_does_not_go_to(drawn, monkeypatch):
    (segs, tgts, infos, seg_k, kinds, delay, ramp), S = drawn
    done = 0
    for g, rows, P, xs, ws in groups(drawn):
        ev = (kinds[rows, 1] >= 0) & ~(segs[0]['integ'][rows] > 0).any(-1)
        if not bool(ev.any()):
            continue
        p = {key: v[rows] for key, v in segs[1].items()}
        ok = g.reach(p, P[:, 1], xs[:, 0], ws[:, 0], xs[:, 1], ws[:, 1], ev)
        assert bool(ok[ev].all())                                    # every accepted event was reached when it was drawn
        moved = ev & ((g.M.y(xs[:, 0], ws[:, 0], P[:, 0])-g.M.y(xs[:, 1], ws[:, 1], P[:, 1])).abs()/g.t['oband']*(g.t['orow'] < NCV)).amax(-1).gt(4.)
        if bool(moved.any()):                                        # told to end where it started, the same run has not arrived
            monkeypatch.setattr(ME, 'REACH_SHARE', 0.)               # by the distance at the end alone (a run that overshoots fourfold on its way would be let off by the share rule)
            no = g.reach(p, P[:, 1], xs[:, 0], ws[:, 0], xs[:, 0], ws[:, 1], moved)
            monkeypatch.undo()
            assert not bool(no[moved].any())
            done += int(moved.sum())
    assert done >= 3


def test_a_plant_is_met_at_more_than_one_clock(drawn, monkeypatch):
    """An episode's decision lasts 0.5 to 1 of the plant's own interval (no longer than its fastest level allows):
    the engine steps that much time, the packet reports that interval, a valve travels that share of its slew in a
    decision, and a level fills over as many more decisions. The plant at half its interval, taken twice, is where
    the plant at its own interval is after one decision."""
    from aio.sim.engine import Mech
    (segs, tgts, infos, seg_k, kinds, delay, ramp), S = drawn
    p = segs[0]
    seen = []
    for g, rows, P, xs, ws in groups(drawn):
        tt = g.t
        pace = tt['pace']
        seen.append(pace)
        cap = ME.pace_max(g.M.plants)[tt['pid']]
        assert bool((pace >= ME.PACE[0]-1e-12).all()) and bool((pace <= cap+1e-12).all())
        own = torch.tensor([q.dt for q in g.M.plants], dtype=F)[tt['pid']]
        hown = torch.tensor([q.dt/q.time_unit for q in g.M.plants], dtype=F)[tt['pid']]
        assert torch.equal(p['dt'][rows], own*pace) and torch.equal(g.M.h, hown*pace) and torch.equal(g.M.pace, pace)
        assert torch.equal(p['tint'][rows], (ME.TINT/pace)[:, None].expand(-1, NCV))
        slew = torch.tensor([[w.get('slew', 1.)/(w['hi']-w['lo']) if w['kind'] == 'mv' else 1. for w in q.ws]+[1.]*(g.M.nw-q.nw) for q in g.M.plants], dtype=F)[tt['pid']]
        want = g.chans(slew*pace[:, None], 1.)[:, :NMV].clamp(.002, .5)
        assert float((p['slew'][rows]-want).abs().max()) < 1e-12
    pace = torch.cat(seen)
    assert float(pace.min()) < .6 and float(pace.max()) > .9
    assert ME.pace_max([ME.pool(['evap'])[0]])[0] < 1. and ME.pace_max([ME.pool(['qtank'])[0]])[0] == ME.PACE[1]     # the evaporator's level is near the limit at its own interval
    # two half decisions are one whole one (the same substep size, the Jacobian of the start kept for both)
    q = PL.PLANTS['hex']()
    one, half = Mech([q], n=1, dtype=F), Mech([q], n=1, dtype=F, pace=torch.tensor([.5]), more=2)
    x, w, mp = one.x0.clone(), one.wnom.clone(), one.pnom.clone()
    w = w*1.05
    a = one.advance(x, w, mp, substeps=2*q.substeps, refresh=False)
    b = half.advance(half.advance(x, w, mp, refresh=False), w, mp, refresh=False)
    assert half.substeps == 2*q.substeps and float((a-b).abs().max()/a.abs().max()) < 1e-3
    c = Mech([q], n=1, dtype=F, pace=torch.tensor([.5])).advance(x, w, mp, refresh=False)        # half the time in the plant's own substeps
    d = one.advance(x, w, mp, substeps=2*q.substeps, guard=False, refresh=True)
    assert float((c-x).abs().max()) < float((d-x).abs().max())
    # a longer decision than the plant's own: every episode of the batch takes it in twice the substeps
    monkeypatch.setattr(ME, 'PACE', (.5, 2.))
    g = ME.Group.draw(PR.Rng(1, 'cpu'), [q], torch.zeros(6, dtype=torch.long))
    assert g.M.more == 2 and g.M.substeps == 2*q.substeps and g.view(torch.arange(2)).M.substeps == 2*q.substeps and float(g.t['pace'].max()) > 1.


def test_program_runs_the_plants_and_rests_at_its_start(prog):
    assert prog.nl == 0 and prog.g.device.type == 'cpu'              # the noise is the same on every device
    q = C.build(12, 7, mech=1., quiet=True, plants=ME.pool(['qtank', 'vdv', 'evap']))
    st = C.reset(q)
    z = torch.zeros(q.n, NMV, dtype=q.dtype)
    kmin = int(q.k_first.min())
    for k in range(min(kmin, 25)):
        st, out = C.step(q, st, z, k)
        assert int(out['outside'].sum()) == 0
    assert bool(torch.isfinite(st['mx']).all())


def test_references_rank_on_the_plants(prog):
    res = {q: CT.rollout(prog, c) for q, c in dict(hold=CT.Hold(), pi=CT.PI(), teacher=CT.Teacher()).items()}
    assert not any(bool(v['failed'].any()) for v in res.values())
    sc = {q: v['score'] for q, v in res.items()}
    assert float(sc['teacher'].mean()) < .6*float(sc['pi'].mean()) < .6*float(sc['hold'].mean())
    assert float((sc['teacher'] < sc['hold']).double().mean()) > .9
    names = prog.mech.names()
    for name in set(names):
        i = torch.tensor([j for j, q in enumerate(names) if q == name])
        assert float(sc['teacher'][i].mean()) < float(sc['hold'][i].mean()), name


def test_mixed_program_keeps_the_linear_episodes_as_they_are():
    a = C.build(12, 9)
    b = C.build(16, 9, mech=.25, plants=ME.pool(['qtank']))
    assert b.nl == 12 and int((b.c['mech'] > 0).sum()) == 4
    ra, rb = CT.rollout(a, CT.PI()), CT.rollout(b, CT.PI())
    assert torch.isfinite(rb['score']).all()
    assert a.P.shape[1:] == b.P.shape[1:]


def test_gradients_pass_through_the_plant_step():
    q = C.build(6, 7, mech=1., quiet=True, dtype=F, plants=ME.pool(['qtank', 'vdv', 'evap']))
    base = .3*q.c['mvm']

    def run(first):
        st, tot = C.reset(q), 0.
        for k in range(10):
            st, out = C.step(q, st, first if k == 0 else base, k)
            tot = tot+out['cost'].sum()
        return tot
    a = base.clone().requires_grad_(True)
    g, = torch.autograd.grad(run(a), a)
    assert bool(torch.isfinite(g).all()) and bool((g.abs().sum(-1) > 0).all())
    d = torch.zeros_like(base); d[0, 0] = 1e-6
    fd = float(run(base+d)-run(base-d))/2e-6
    assert abs(float(g[0, 0])-fd) < .01*abs(fd)+1e-9


def test_the_default_pool_has_the_five_new_plants_and_the_exam_of_the_first_ten_is_as_it_was():
    """What a batch with a share of mechanistic episodes is drawn from (and a run with --mech trains on): the ten
    plants of before and the five added 11-11 and 11-12. The exam `mech` names the ten, in the order the pool had
    them, so it is the batch it was; `mech_new` is the five. The large and the held-out plants stay out."""
    assert PL.FIRST == ('qtank', 'ph', 'vdv', 'excstr', 'bio', 'evap', 'furnace', 'gasnet', 'hex', 'column')
    assert PL.NEW == ('bldg', 'comp', 'rsr', 'amm', 'heater') and tuple(PL.PLANTS) == PL.FIRST+PL.NEW
    assert not set(PL.PLANTS) & (set(PL.LARGE) | set(PL.HELDOUT)) and {'tep', 'bsm1', 'btcol', 'meoh', 'debut'} <= set(PL.LARGE)
    assert ST.EXAMS['mech']['plants'] == PL.FIRST and ST.EXAMS['mech_new']['plants'] == PL.NEW
    assert [q.name for q in ME.pool()] == list(PL.PLANTS) and [q.name for q in ME.pool(PL.FIRST)] == list(PL.FIRST)


def test_an_episode_whose_task_has_no_operable_start_is_given_another_task(monkeypatch):
    """A task is drawn before its start point (the outputs to hold, the slots, the valves' curves, the clock, the
    transport delay). Where no start point passes the check with it, not even the plant's nominal one, the
    episode gets another task of the same plant from its own stream; the others keep theirs, and the batch no
    longer fails as a whole (one such episode in a hundred was enough to stop a draw of 128)."""
    q = ME.pool(['qtank', 'vdv'])
    pid = torch.tensor([0, 0, 0, 1, 1, 1])
    r = PR.Rng(11, 'cpu').bind(6)
    g = ME.Group.draw(r, q, pid)
    before = {key: v.clone() for key, v in g.t.items()}
    real, calls = ME.Group._start, []

    def start(self, rr):                                             # the first draw: episodes 1 and 4 have no operable start
        best = real(self, rr)
        calls.append(self.k)
        if len(calls) == 1:
            ok = best[-1].clone(); ok[[1, 4]] = False
            best = best[:-1]+(ok,)
        return best
    monkeypatch.setattr(ME.Group, '_start', start)
    p, tgt, info, mp, x, w = g.start(r)
    assert calls == [6, 2]
    same = torch.tensor([True, False, True, True, False, True])
    for key in ('chan', 'orow', 'pace', 'din'):
        assert torch.equal(g.t[key][same], before[key][same])
    assert not torch.equal(g.t['pace'][~same], before['pace'][~same]) and torch.equal(g.t['pid'], pid)
    assert torch.equal(g.M.pace, g.t['pace']) and g.M.n == 6
    ok, _ = PR.operable(p, tgt)
    assert bool(ok.all())
    y = g.M.linearise(x, w, mp)[1]                                  # the start of the two is a steady point of their plant with their new task
    v = g.rows((y-g.t['rlo'])/g.t['rspan'])
    assert float(((v-tgt).abs()*p['trk']/p['bandf']).max()) < 1e-3
    assert float((g.M.f(x, w, mp)*g.M.xmask).abs().max()) < 1e-6
    monkeypatch.setattr(ME, 'TASK_ROUNDS', 0)                        # without the redraw the batch fails as before
    calls.clear()
    with pytest.raises(RuntimeError):
        ME.Group.draw(PR.Rng(11, 'cpu').bind(6), q, pid).start(PR.Rng(11, 'cpu').bind(6))

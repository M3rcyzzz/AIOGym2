"""aio.train (second trainer): the guard holds one episode's gradient and leaves the others alone, a cut stops the
gradient of the episodes it is for, the critic's targets are what TD(lambda) says, every line written is strict
JSON, and a window, the critic's fit and the horizons' adaptation run end to end on a small batch."""
import faulthandler
import json
import math
import time
from types import SimpleNamespace

import pytest
import torch

from aio import net as N, train as T
from aio.sim import core as C


def args(**kw):
    a = dict(window=6, window_min=1, guard='cut', cap=10., gamma=.99, lam=.95, tau=.25, cost_w=.1, log_w=0., smooth=0., floor=.1, fail_cost=10., noise=.05,
             relax=4, known=.25, critic_known=.75, id_coef=.1, bc_coef=10., bc_iters=30, value_warm=3, critic_iters=2, clip=1., lean=False, fit_chunk=0,
             drive=0., drive_noise=.2, bc_floor=0., bc_hold=0, sim_window=0, replay=0, batch=256, keep=12, probe=0., law=False, law_coef=1.)
    a.update(kw)
    return SimpleNamespace(**a)


def test_guard_holds_nothing_until_it_knows_what_is_usual():
    g = torch.randn(8, 5)*torch.logspace(-3, 3, 8)[:, None]
    gd = T.Guard(8, 10., g.device)
    assert torch.equal(gd(g), g) and not gd.held().any()
    assert gd.median() == pytest.approx(float(g.norm(dim=-1).median()))


def test_guard_holds_one_episode_to_the_cap_and_drops_what_is_not_finite():
    g = torch.ones(6, 5)
    g[1] *= 1e4
    g[2, 0] = float('nan')
    g[3, 1] = float('inf')
    one = float(g[0].norm())
    gd = T.Guard(6, 10., g.device, usual=one)
    out = gd(g)
    assert torch.equal(out[0], g[0]) and torch.equal(out[4], g[4])                       # the others keep theirs
    assert torch.allclose(out[1].norm(), torch.tensor(10*one))                           # held to cap times the usual level
    assert not out[2].any() and not out[3].any()                                         # dropped
    assert gd.capped.tolist() == [False, True, False, False, False, False] and gd.broken.tolist() == [False, False, True, True, False, False]
    assert gd.held().tolist() == [False, True, True, True, False, False]
    assert gd.peak() == pytest.approx(1e4, rel=1e-4) and gd.median() == pytest.approx(one)
    assert torch.equal(gd(.5*g[:, :])[4], .5*g[4]) and gd.top[4] == pytest.approx(one)   # top: the largest over the window's moves


def test_guard_measures_in_units_of_the_objective():
    """While the simulator gradient is faded in, the hook sees it times its weight: the same episodes are held."""
    g = torch.ones(4, 5)
    g[1] *= 50.
    one = float(g[0].norm())
    full, faint = T.Guard(4, 10., g.device, usual=one), T.Guard(4, 10., g.device, usual=one, weight=.03)
    a, b = full(g), faint(.03*g)
    assert torch.allclose(b, .03*a) and torch.equal(full.capped, faint.capped) and torch.allclose(full.top, faint.top)


def test_running_reference():
    assert T.running(None, 2.) == 2. and T.running(2., None) == 2. and T.running(2., 4.) == pytest.approx(2.2)


def test_guard_hook_keeps_the_gradient_of_the_other_episodes():
    """A toy plant whose backward pass overflows in float32 for the second episode only."""
    gain = torch.tensor([[1.], [3e38]])
    for hook, finite in ((False, False), (True, True)):
        w = torch.ones(2, requires_grad=True)
        act = w[None].expand(2, 2)*1.
        if hook:
            act.register_hook(T.Guard(2, 10., act.device))
        (act*gain*10.).sum().backward()
        assert bool(torch.isfinite(w.grad).all()) is finite
    assert torch.equal(w.grad, torch.full((2,), 10.))


def test_cut_rows_stops_the_gradient_of_the_episodes_it_is_for():
    x = torch.ones(3, 2, requires_grad=True)
    y = T.cut_rows(x*2., torch.tensor([False, True, False]))
    y.sum().backward()
    assert torch.equal(y, torch.full((3, 2), 2.))
    assert torch.equal(x.grad, torch.tensor([[2., 2.], [0., 0.], [2., 2.]]))


def test_td_targets():
    cost, v = torch.tensor([[1., 2., 3.]]), torch.tensor([[10., 20., 30., 40.]])
    g = .9
    full = T.td_targets(cost, v, g, 1.)                                                  # lambda 1: the costs, then the value after the last
    assert torch.allclose(full, torch.tensor([[1+g*(2+g*(3+g*40)), 2+g*(3+g*40), 3+g*40]]))
    one = T.td_targets(cost, v, g, 0.)                                                   # lambda 0: one step, then the next value
    assert torch.allclose(one, torch.tensor([[1+g*20, 2+g*30, 3+g*40]]))


def test_dumps_is_strict_json():
    s = T.dumps(dict(a=float('inf'), b=[float('nan'), 1.5], c=dict(d=-float('inf')), e='x', f=(1, 2.)))
    assert 'Infinity' not in s and 'NaN' not in s
    assert json.loads(s) == dict(a=None, b=[None, 1.5], c=dict(d=None), e='x', f=[1, 2.])


def test_weights_fade_the_teacher_out_and_the_value_in():
    a = args()
    assert T.weights(0, a) == dict(bc=1., sim=0., value=0.)
    w = T.weights(15, a)
    assert w['bc'] == pytest.approx(.5) and w['sim'] == pytest.approx(.5) and w['value'] == 1.
    assert T.weights(30, a)['bc'] == 0. and T.weights(99, a)['sim'] == 1.
    a.bc_iters = 0                                                                       # from earlier weights: no teacher at all
    assert T.weights(0, a)['bc'] == 0. and T.weights(0, a)['sim'] == 1.
    a = args(bc_floor=.1)                                                                # the teacher stays in at a tenth
    assert T.weights(15, a)['bc'] == pytest.approx(.5) and T.weights(99, a) == dict(bc=.1, sim=.9, value=1.)
    a = args(bc_hold=10, bc_iters=20, bc_floor=.1)                                       # the teacher alone for ten rows first
    assert T.weights(0, a)['sim'] == 0. and T.weights(10, a) == dict(bc=1., sim=0., value=1.)
    assert T.weights(20, a)['bc'] == pytest.approx(.5) and T.weights(30, a)['bc'] == .1
    a = args(value=0.)                                                                   # --value: the critic's weight once it is in
    assert T.weights(0, a)['value'] == 0. and T.weights(99, a)['value'] == 0.
    assert T.weights(1, args(value=.6))['value'] == pytest.approx(.2) and T.weights(99, args(value=.6))['value'] == .6


def test_the_rate_comes_down_a_cosine_and_a_run_carried_on_over_more_hours_goes_back_up_it():
    assert T.lr_factor(0., 2.) == pytest.approx(1.) and T.lr_factor(1., 2.) == pytest.approx(.55)
    assert T.lr_factor(2., 2.) == pytest.approx(.1) and T.lr_factor(9., 2.) == pytest.approx(.1)
    assert T.lr_factor(4.3, 10.)/T.lr_factor(4.3, 4.3) > 6                               # a run at the end of its hours, resumed with --hours 10 and the same --lr (refused: --resume)


def test_a_run_and_the_bench_build_the_networks_the_flags_ask_for():
    a = args(no_stats=False, tied=0, knots=N.NLAB, formula=False, solve=False, given=False, slow=0., arx=True, dv=True, diff='S', heads=2, law=True, replay=2)
    net, critic = T.nets(a, torch.device('cpu'))
    sd = net.pairs.state_dict()
    assert N.law_of(sd) and N.arx_of(sd) and N.dv_of(sd) and N.diff_of(sd) == 'S' and net.probe == 0. and len(critic.heads) == 2
    assert not N.unseen_of(sd)
    a.unseen = True
    assert N.unseen_of(T.nets(a, torch.device('cpu'))[0].pairs.state_dict()) and not N.told_of(sd)
    a.told = True                                                                        # --told: the share told, into the pairs (the critic has none)
    net, critic = T.nets(a, torch.device('cpu'))
    assert N.told_of(net.pairs.state_dict()) and not N.told_of(critic.pairs.state_dict())
    a.formula = True
    with pytest.raises(AssertionError):                                                  # --unseen: beside the E read-out that learns
        T.nets(a, torch.device('cpu'))
    a.formula = False
    a.replay = 0
    with pytest.raises(AssertionError):                                                  # --law is for --replay
        T.nets(a, torch.device('cpu'))


def test_adapt_shortens_what_was_held_and_gives_a_halving_back_after_quiet_windows():
    a = args(window=32, window_min=4, relax=2)
    assert T.top_level(a) == 3
    lane = SimpleNamespace(level=torch.tensor([0, 1, 3, 2]), quiet=torch.tensor([0, 0, 0, 1]))
    gd = SimpleNamespace(held=lambda: torch.tensor([True, False, True, False]))
    T.adapt(lane, gd, a)
    assert lane.level.tolist() == [1, 1, 3, 1] and lane.quiet.tolist() == [0, 1, 0, 0]   # up; waits; at the top; quiet for two windows: one back
    T.adapt(lane, SimpleNamespace(held=lambda: torch.zeros(4, dtype=torch.bool)), a)
    assert lane.level.tolist() == [1, 0, 3, 1] and lane.quiet.tolist() == [1, 0, 1, 1]
    a.guard = 'cap'
    T.adapt(lane, gd, a)
    assert not lane.level.any()


def _setup(a, n=6, seed=5, stats=True):
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    net, critic = N.AIO(False), N.Critic(stats=False)
    torch.nn.init.normal_(net.pairs.out.weight, std=.05)                                 # the pair pathway in play
    if stats:                                                                            # the same networks grown by the running sums, read
        old, net, critic = (net, critic), N.AIO(), N.Critic()
        net.load_state_dict(old[0].state_dict()); critic.load_state_dict(old[1].state_dict())
        torch.nn.init.normal_(net.pairs.stat[-1].weight, std=.3)
    lane = T.Lane(C.build(n, seed), a, gen)
    return net, critic, lane, gen


@pytest.mark.parametrize('guard,lean', [('cut', False), ('cut', True), ('open', True), ('cap', False)])
def test_an_update_runs_end_to_end(guard, lean):
    a = args(guard=guard, lean=lean, fit_chunk=4 if lean else 0)
    net, critic, lane, gen = _setup(a)
    lane.level[:3] = 1                                                                   # some episodes on short stretches from the start
    opt, copt = torch.optim.Adam(net.parameters(), lr=1e-3), torch.optim.Adam(critic.parameters(), lr=1e-3)
    before = [q.detach().clone() for q in list(net.parameters())+list(critic.parameters())]
    temp = dict(usual=None)
    for w, skip in ((dict(bc=1., sim=0., value=0.), 0), (dict(bc=.5, sim=.5, value=1.), 1), (dict(bc=0., sim=1., value=1.), 0)):
        log = T.update(net, critic, opt, copt, lane, a, w, gen, temp)                    # (the first window with the simulator gradient measures the guard's level: no step)
        assert log['skip'] == skip and all(math.isfinite(log[q]) for q in ('sim', 'cost', 'id', 'critic', 'ev', 'value', 'far' if skip else 'gn'))
    assert lane.k == 18 and int(lane.level.max()) <= T.top_level(a)
    assert all(not v.requires_grad for v in lane.st.values()) and not lane.ha.requires_grad and not lane.hc.requires_grad
    after = list(net.parameters())+list(critic.parameters())
    moved = [bool((x != y).any()) for x, y in zip(before, after)]
    assert sum(moved) > .9*len(moved)                                                    # actor, pair pathway and critic all learn
    assert all(q.requires_grad for q in critic.parameters())
    assert temp['usual'] > 0 and log['peak'] > 0                                         # the guard has its reference from the first window with the gradient in


def test_accum_steps_once_on_the_mean_of_the_windows_gradients():
    a = args(accum=2)
    net, critic, lane, gen = _setup(a)
    opt, copt = torch.optim.SGD(net.parameters(), lr=1.), torch.optim.Adam(critic.parameters(), lr=1e-3)
    before = [q.detach().clone() for q in net.parameters()]
    temp, w, gs, differ = dict(usual=None), dict(bc=1., sim=0., value=0.), [], []
    T.update(net, critic, opt, copt, lane, a, w, gen, temp)
    gs.append([None if q.grad is None else q.grad.clone() for q in net.parameters()])
    assert all(bool((x == y).all()) for x, y in zip(before, net.parameters())) and temp['acc']['k'] == 1   # the first window: no step yet
    T.update(net, critic, opt, copt, lane, a, w, gen, temp)
    assert temp['acc'] == dict(g=None, n=0, k=0)
    for x, y, g in zip(before, net.parameters(), gs[0]):                                 # what is stepped on is in .grad: the mean of the two
        if g is not None:
            assert torch.allclose(x-y, y.grad, atol=1e-6)
            differ.append(not torch.allclose(y.grad, g))
    assert sum(differ) > .5*len(differ)
    T.gathered(net, opt, temp, 2, False); T.gathered(net, opt, temp, 2, False)           # two windows left out: nothing to step on
    assert temp['acc'] == dict(g=None, n=0, k=0)


def test_the_actor_step_leaves_the_critic_alone_and_the_value_reaches_the_moves():
    """The actor's objective passes through the critic into the plant and back to the moves; the critic's weights
    get nothing from it. With the costs of the window taken out (gamma 0 keeps the first decision's only), what is
    left of the gradient is the critic's."""
    a = args(noise=0., id_coef=0.)
    net, critic, lane, gen = _setup(a)
    for q in critic.parameters():
        q.requires_grad_(False)
    loss, rec, guard, _ = T.window(net, critic, lane, a, dict(bc=0., sim=1., value=1.), gen)
    loss.backward()
    assert all(q.grad is None for q in critic.parameters())
    assert sum(float(q.grad.abs().sum()) for q in net.parameters() if q.grad is not None) > 0
    assert rec['obs'].shape[:2] == (6, 7) and rec['cost'].shape == (6, 6) and rec['z'].shape == (6, 7, 2)
    assert guard.median() > 0 and guard.peak() is None                                   # no reference yet


def _first_move_gradient(window, cut):
    """Per episode, the gradient that reaches the first move of a window of the cost terms alone."""
    a = args(window=window, noise=0., id_coef=0.)
    net, critic, lane, gen = _setup(a)
    if cut:
        lane.level[:] = T.top_level(a)                                                   # a cut at every decision
    seen, orig = [], T.Guard.__call__

    def spy(self, g):
        seen.append(g.clone())
        return orig(self, g)
    T.Guard.__call__ = spy
    try:
        for q in critic.parameters():
            q.requires_grad_(False)
        T.window(net, critic, lane, a, dict(bc=0., sim=1., value=0.), gen)[0].backward()
    finally:
        T.Guard.__call__ = orig
    assert len(seen) == window
    return seen[-1]                                                                      # the hooks fire last decision first


def test_a_cut_keeps_later_costs_from_earlier_moves():
    """With every episode cut at every decision (and no critic value) a move is charged for its own decision only:
    the gradient at the first move points where it does in a window of one decision. Uncut, the later decisions
    are charged to it as well."""
    cos = torch.nn.functional.cosine_similarity
    one, cut, whole = _first_move_gradient(1, False), _first_move_gradient(4, True), _first_move_gradient(4, False)
    live = one.norm(dim=-1) > 0
    assert bool(live.any())
    assert bool((cos(one[live], cut[live]) > .9999).all())
    assert bool((cos(one[live], whole[live]) < .999).any())


def test_a_step_down_an_episode_s_gradient_lowers_its_cost():
    """The gradient through plant and network points downhill, episode by episode: after a small step against the
    gradient of one episode alone, the same window (same noise) costs that episode less. Deterministic actor, no
    critic value, no teacher. Measured over four initialisations of the network (11-11): 10 to 12 of 12 episodes
    at this step; the others are an episode a sticking valve or a coarse transmitter holds flat (no change at
    all) and one near rest whose cost turns within the step. So three in four is asked, not all."""
    n = 12
    a = args(window=12, noise=0., id_coef=0.)
    net, critic, lane, gen = _setup(a, n=n)
    start = dict(st=lane.st, ha=lane.ha, hc=lane.hc, prev=lane.prev, noise=lane.prog.g.get_state())
    theta = [q.detach().clone() for q in net.parameters()]
    for q in critic.parameters():
        q.requires_grad_(False)
    disc = a.gamma**torch.arange(12.)
    orig = T.Guard.__call__

    def run():
        lane.st, lane.ha, lane.hc, lane.prev, lane.k, lane.hs = start['st'], start['ha'], start['hc'], start['prev'], 0, None
        lane.prog.g.set_state(start['noise'])
        loss, rec, _, _ = T.window(net, critic, lane, a, dict(bc=0., sim=1., value=0.), gen)
        return loss, (rec['cost']*disc).sum(1)
    better = worse = 0
    for i in range(n):
        only = torch.nn.functional.one_hot(torch.tensor(i), n)[:, None]
        T.Guard.__call__ = lambda self, g: orig(self, g)*only                           # the hooks fire in the backward pass
        try:
            net.zero_grad(set_to_none=True)
            loss, before = run()
            loss.backward()
        finally:
            T.Guard.__call__ = orig
        gs = [(q, q.grad) for q in net.parameters() if q.grad is not None]
        gn = torch.sqrt(sum(g.square().sum() for _, g in gs))
        assert float(gn) > 0
        with torch.no_grad():
            for q, g in gs:
                q -= 1e-5*g/gn
        after = run()[1][i]
        better, worse = better+bool(after < before[i]), worse+bool(after > before[i])
        with torch.no_grad():
            for q, t in zip(net.parameters(), theta):
                q.copy_(t)
    assert better >= 9 and worse <= 2


def test_the_noise_on_the_moves_is_fixed_and_the_spread_head_is_left_alone():
    """Every move gets noise of the given spread before the tanh; the network's own spread head has no part in it
    (steering its mean entropy ran away from SAC's weights, 11-11)."""
    a = args(noise=.2, id_coef=0.)
    net, critic, lane, gen = _setup(a, n=64)
    with torch.no_grad():
        net.actor.head.out.bias[1] = 1.                                                  # a spread head far off: it must not matter
    obs, s = T.canonical(C.observe(lane.prog, lane.st, 0).float())
    u, mean, mask, _, _, _ = T.decide(net, lane, obs, lane.prev, lane.ha, s, 0, gen, a.noise)
    z = torch.atanh(u.clamp(-.999, .999))-mean
    assert float(z[mask > 0].std()) == pytest.approx(.2, rel=.2) and not u[mask == 0].any()
    quiet, *_ = T.decide(net, lane, obs, lane.prev, lane.ha, s, 0, gen, 0.)
    assert torch.allclose(quiet, torch.tanh(mean)*mask, atol=1e-6)                       # (two calls: sums over pairs, not bit for bit)
    for q in critic.parameters():
        q.requires_grad_(False)
    loss, *_ = T.window(net, critic, lane, a, dict(bc=0., sim=1., value=1.), gen)
    loss.backward()
    g = net.actor.head.out.weight.grad
    assert float(g[0].abs().sum()) > 0 and not g[1].any()                                # the mean learns, the spread head gets nothing


def test_the_running_sums_are_kept_once_for_actor_and_critic():
    a = args()
    net, critic, lane, gen = _setup(a)
    T.advance(net, critic, lane, 3, gen, .05)
    hs = lane.hs.clone()
    for q in critic.parameters():
        q.requires_grad_(False)
    _, rec, _, _ = T.window(net, critic, lane, a, dict(bc=0., sim=1., value=1.), gen)
    F, after = net.pairs.stats(rec['obs'][:, :6], hs)                                    # what the network itself would have kept
    assert torch.allclose(rec['feats'], F, atol=1e-6) and torch.allclose(lane.hs, after, atol=1e-6) and float(F.abs().sum()) > 0
    sums = slice(N.BASE+N.SPAIR, N.BASE+N.SPAIR+N.SSTAT)
    assert not lane.ha[..., sums].any() and not lane.hc[..., sums].any()
    net, critic, lane, gen = _setup(a, stats=False)                                      # --no-stats: the network of before, nothing kept
    assert not N.has_stats(net.state_dict()) and not N.has_stats(critic.state_dict())
    opt, copt = torch.optim.Adam(net.parameters(), lr=1e-3), torch.optim.Adam(critic.parameters(), lr=1e-3)
    T.advance(net, critic, lane, 3, gen, .05)
    log = T.update(net, critic, opt, copt, lane, a, dict(bc=0., sim=1., value=1.), gen, dict(usual=1e-3))
    assert lane.hs is None and log['skip'] == 0 and math.isfinite(log['critic'])


def test_lanes_start_at_different_points():
    a = args()
    net, critic, lane, gen = _setup(a)
    T.advance(net, critic, lane, 12, gen, .05)
    assert lane.k == 12 and float(lane.ha.abs().sum()) > 0 and float(lane.hc.abs().sum()) > 0


def test_rebuilding_the_graphs_at_the_backward_pass_changes_nothing():
    """--lean: the networks' graphs are rebuilt at the backward pass. The loss, the gradient into every weight and
    what the guard measured are those of the kept graph; the critic's fit in parts is its fit at once."""
    got = []
    for lean, chunk in ((False, 0), (True, 4)):
        a = args(lean=lean, fit_chunk=chunk)
        net, critic, lane, gen = _setup(a)
        lane.level[:2] = 1                                                               # cuts too
        lane.label[-2:] = True                                                           # and two episodes for imitation
        for q in critic.parameters():
            q.requires_grad_(False)
        loss, rec, guard, _ = T.window(net, critic, lane, a, dict(bc=.5, sim=.5, value=1.), gen, usual=1e-3)
        loss.backward()
        for q in critic.parameters():
            q.requires_grad_(True)
        copt = torch.optim.SGD(critic.parameters(), lr=1e-2)
        fit = T.fit_critic(critic, copt, lane, rec, a)
        got.append((float(loss), [q.grad.clone() for q in net.parameters() if q.grad is not None], guard.top.clone(), guard.capped.clone(),
                    fit, [q.detach().clone() for q in critic.parameters()]))
    (l0, g0, t0, c0, f0, w0), (l1, g1, t1, c1, f1, w1) = got
    assert l1 == pytest.approx(l0, rel=1e-5) and len(g0) == len(g1) > 0
    scale = max(float(x.abs().max()) for x in g0)
    assert all(torch.allclose(x, y, rtol=1e-3, atol=1e-5*scale) for x, y in zip(g0, g1))
    assert torch.allclose(t0, t1, rtol=1e-3) and torch.equal(c0, c1) and bool(c0.any())  # the guard held the same episodes
    assert f1['critic'] == pytest.approx(f0['critic'], rel=1e-4) and f1['ev'] == pytest.approx(f0['ev'], abs=1e-4)
    assert all(torch.allclose(x, y, rtol=1e-3, atol=1e-6) for x, y in zip(w0, w1))


def test_a_move_that_turns_back_costs_and_a_ramp_does_not():
    """--smooth: every decision of a window costs the squared change of the move (before its noise) from the
    decision before, per MV, on top of the objective: nothing else of the window changes. With the term the
    whole of the objective, a small step against the gradient makes the same window's moves change less."""
    n, w = 6, dict(bc=0., sim=1., value=0.)
    a, plain = args(smooth=5., noise=0., id_coef=0.), args(noise=0., id_coef=0.)
    net, critic, lane, gen = _setup(a, n=n)
    with torch.no_grad():
        net.actor.head.out.weight[0] *= 30                                               # moves of a size that shows
    start = dict(st=lane.st, ha=lane.ha, hc=lane.hc, prev=lane.prev, pm=lane.pm, noise=lane.prog.g.get_state())

    def run(a):
        lane.st, lane.ha, lane.hc, lane.prev, lane.pm, lane.k, lane.hs = start['st'], start['ha'], start['hc'], start['prev'], start['pm'], 0, None
        lane.prog.g.set_state(start['noise'])
        loss, rec, _, _ = T.window(net, critic, lane, a, w, gen)
        u = torch.cat((start['pm'][:, None], rec['act'][:, :a.window]), 1)              # no noise: the moves are the means
        return loss, rec['cost'], (u[:, 1:]-u[:, :-1]).square().sum(-1)/(lane.prog.c['mvm'] > 0).sum(-1).clamp_min(1)[:, None]
    (_, c0, r0), (_, c1, r1) = run(plain), run(a)
    assert float(r0.mean()) > 1e-4 and torch.equal(r0, r1)
    assert torch.allclose(c1-c0, 5.*r0, rtol=1e-3, atol=1e-5)
    assert torch.equal(lane.pm, lane.prev)                                               # carried to the next window
    a.smooth = 1e4
    loss, _, before = run(a)
    net.zero_grad(set_to_none=True)
    loss.backward()
    gs = [(q, q.grad) for q in net.parameters() if q.grad is not None]
    gn = torch.sqrt(sum(g.square().sum() for _, g in gs))
    with torch.no_grad():
        for q, g in gs:
            q -= 1e-4*g/gn
    after = run(a)[2]
    assert float(after.sum()) < float(before.sum())


def test_the_teacher_takes_the_plant_over_after_a_window_and_the_episode_goes_on_as_it_was():
    """--after: what the teacher pays over the decisions after a window, from the state the window ended in, is
    added to the window's own cost, discounted on from it, and the window's moves answer for it. The lane, what
    the window recorded and the plant's random draws are those of the window without it; past the episode's
    last decision there is nothing to add."""
    n, w = 6, dict(bc=0., sim=1., value=0.)
    plain, a = args(noise=0., id_coef=0.), args(noise=0., id_coef=0., after=8)
    net, critic, lane, gen = _setup(a, n=n)
    prog = lane.prog
    start = dict(st=lane.st, ha=lane.ha, hc=lane.hc, prev=lane.prev, pm=lane.pm, noise=prog.g.get_state())

    def run(a, k=0):
        lane.st, lane.ha, lane.hc, lane.prev, lane.pm, lane.k, lane.hs = start['st'], start['ha'], start['hc'], start['prev'], start['pm'], k, None
        prog.g.set_state(start['noise'])
        net.zero_grad(set_to_none=True)
        loss, rec, _, _ = T.window(net, critic, lane, a, w, gen)
        loss.backward()
        grad = torch.cat([q.grad.flatten() for q in net.parameters() if q.grad is not None])
        return rec, lane.st, prog.g.get_state(), grad
    (r0, s0, g0, d0), (r1, s1, g1, d1) = run(plain), run(a)
    assert all(torch.equal(r0[q], r1[q]) for q in ('obs', 'act', 'cost', 'z')) and lane.k == 6
    assert all(torch.equal(s0[q], s1[q]) for q in s0) and torch.equal(g0, g1)            # the episode goes on from the same state and draws
    assert bool(torch.isfinite(d1).all()) and float((d1-d0).abs().max()) > 1e-6          # the moves answer for what follows
    last = T.HORIZON-a.window
    assert torch.equal(run(plain, last)[3], run(a, last)[3])                             # nothing after the episode's last decision
    # the sum itself: the teacher's moves from that state, the window's discount carried on
    zero, one = torch.zeros(n), torch.ones(n)
    st, want, d = s1, zero, .5*one
    for j in range(6, 9):
        st, out = C.step(prog, st, C.teacher(prog, st, j).to(prog.dtype), j)
        cost = torch.where(out['fail'], torch.full_like(zero, a.fail_cost), T.OR.objective(out, st, prog.c, a.tau, a.cost_w, a.log_w).float())
        want, d = want+d*cost, d*a.gamma
    prog.g.set_state(g1)
    acc, wsum, disc = T.handed(prog, s1, 6, 3, a, one, zero, .5*one)
    assert torch.allclose(acc, 1.+want, rtol=1e-5) and float(want.min()) > 0
    assert torch.allclose(wsum, .5*(1+a.gamma+a.gamma**2)*one) and torch.allclose(disc, .5*a.gamma**3*one)
    assert torch.equal(prog.g.get_state(), g1)
    # --after-by hold: the MVs left where they are; the teacher's model is not kept up along the way and is back after
    st, want, d = s1, zero, .5*one
    for j in range(6, 9):
        st, out = C.step(prog, st, torch.zeros(n, prog.nmv, dtype=prog.dtype), j)
        want, d = want+d*T.OR.objective(out, st, prog.c, a.tau, a.cost_w, a.log_w).float(), d*a.gamma
    prog.g.set_state(g1)
    Fr = prog.Fr
    acc, _, _ = T.handed(prog, s1, 6, 3, args(noise=0., id_coef=0., after=8, after_by='hold'), one, zero, .5*one)
    assert torch.allclose(acc, 1.+want, rtol=1e-5) and prog.Fr is Fr and torch.equal(prog.g.get_state(), g1)


def test_an_episode_is_for_imitation_or_for_the_simulator_gradient():
    """The imitation episodes (a share equal to the teacher's weight): in those drawn for --drive the plant gets
    the teacher's move (with its noise), the student's in the others; the imitation term is taken on them alone
    and none of it goes through the plant or the last move; the simulator's gradient and the guard are for the
    other episodes alone, in the same units whatever their share."""
    n = 8
    a = args(noise=0., drive_noise=0., id_coef=0.)
    net, critic, _, gen = _setup(a, n=n)
    lane = T.Lane(C.build(n, 5), a, gen, teach=.5, drive=.5)
    assert lane.label.dtype == torch.bool and not bool((lane.drive & ~lane.label).any())
    full, none = T.Lane(C.build(n, 5), a, gen, teach=1., drive=1.), T.Lane(C.build(n, 5), a, gen)
    assert bool(full.label.all()) and bool(full.drive.all()) and not bool(none.label.any()) and none.drive is None
    lane.label, lane.drive = torch.arange(n) < 4, torch.arange(n) < 2                    # 0, 1: the teacher drives; 2, 3: labelled; 4-7: simulator
    for q in critic.parameters():
        q.requires_grad_(False)
    start = dict(st=lane.st, ha=lane.ha, hc=lane.hc, prev=lane.prev, pm=lane.pm, noise=lane.prog.g.get_state())
    seen, raw = [], []
    step, call = C.step, T.Guard.__call__

    def back():
        lane.st, lane.ha, lane.hc, lane.prev, lane.pm, lane.k, lane.hs = start['st'], start['ha'], start['hc'], start['prev'], start['pm'], 0, None
        lane.prog.g.set_state(start['noise'])

    def spy(prog, st, act, k, *args, **kw):
        seen.append((act.detach().clone(), (C.teacher(prog, st, k).float()).clone()))
        return step(prog, st, act, k, *args, **kw)
    C.step = spy
    try:
        loss, rec, guard, log = T.window(net, critic, lane, a, dict(bc=.5, sim=.5, value=0.), gen)
    finally:
        C.step = step
    d, lab = lane.drive, lane.label
    for act, tch in seen:
        mask = (lane.prog.c['mvm'] > 0).float()
        assert torch.allclose(act[d], (tch*mask)[d].clamp(-.999, .999), atol=2e-3)                          # the teacher's own move, plant frame
    assert not any(torch.allclose(act[~d], (tch*(lane.prog.c['mvm'] > 0))[~d], atol=1e-3) for act, tch in seen)
    loss.backward()
    assert float(guard.top[lab].max()) == 0. and float(guard.top[~lab].min()) > 0.                         # the guard sees the simulator's episodes
    assert torch.isfinite(loss) and log['bc'] > 0
    top = guard.top.clone()
    # the lane goes on as the teacher left it
    assert torch.equal(lane.prev, rec['act'][:, a.window-1])
    # the imitation term alone: the weights get its gradient, the moves none of it through the plant
    net.zero_grad(set_to_none=True)
    back()
    T.Guard.__call__ = lambda self, g: (raw.append(g.detach().clone()), g)[1]
    try:
        loss, *_ = T.window(net, critic, lane, a, dict(bc=1., sim=1e-30, value=0.), gen)
        loss.backward()
    finally:
        T.Guard.__call__ = call
    assert raw and max(float(g.abs().max()) for g in raw) < 1e-20
    assert sum(float(q.grad.abs().sum()) for q in net.parameters() if q.grad is not None) > 1e-6
    # the same episodes with no imitation episode beside them: what the guard measures of each is the same
    net.zero_grad(set_to_none=True)
    back()
    lane.label, lane.drive = torch.zeros(n, dtype=torch.bool), None
    loss, _, guard, log = T.window(net, critic, lane, a, dict(bc=.5, sim=.5, value=0.), gen)
    loss.backward()
    assert log['bc'] is None and torch.allclose(guard.top[~lab], top[~lab], rtol=1e-3)
    back()
    T.advance(net, critic, lane, 3, gen, 0., 0.)
    assert lane.k == 3


def test_a_window_gathered_goes_through_the_network_at_once_as_it_did_a_decision_at_a_time():
    """--replay: a window is run without a graph and kept as data; drawn again, it goes through `steps` in one
    call from the state it was gathered in and gives the moves the student made a decision at a time (the
    running sums included: the second window starts from sums that are not zero). A window that reaches the
    episode's end is padded, and the padding counts for nothing. Steps on the buffer bring the teacher nearer
    on those windows; the lane has moved on and the critic was fitted."""
    n = 6
    a = args(noise=0., replay=3, batch=4, keep=3, fit_chunk=2)
    net, critic, _, gen = _setup(a, n=n)
    opt, copt = torch.optim.Adam(net.parameters(), lr=3e-4), torch.optim.Adam(critic.parameters(), lr=3e-4)
    lane = T.Lane(C.build(n, 5), a, gen, teach=1.)
    buf = T.Buffer(a.keep*n)
    T.imitate(net, critic, opt, copt, lane, buf, SimpleNamespace(**dict(vars(a), replay=0)), gen)    # a first window, gathered only
    ha, hs, prev = lane.ha, lane.hs, lane.prev
    with torch.no_grad():
        _, rec, _, _ = T.window(net, critic, lane, a, dict(bc=1., sim=0., value=0.), gen)
    assert lane.k == 12 and float(hs.abs().sum()) > 0
    h0 = torch.cat((ha[0, :, :N.STATE-N.SSTAT], hs), -1)
    mean, _, mask, _, _ = net.steps(rec['obs'][:, :6], rec['prev'][:, :6], h0[None], (rec['lab'][:, :6], lane.known), None)
    assert torch.equal(rec['prev'][:, 0], prev)
    assert torch.allclose(torch.tanh(mean)*mask, rec['act'][:, :6], atol=2e-5) and float(rec['act'][:, :6].abs().sum()) > 0
    assert rec['tch'].shape == rec['m'].shape == (n, 6, mask.shape[-1]) and float(rec['m'].sum()) > 0
    kept = buf.x['feats'][:n]                                     # the running sums' features are kept with the window, in half precision
    assert kept.dtype == torch.float16 and kept.shape == (n, 6)+rec['feats'].shape[2:] and float(kept.abs().sum()) > 0
    with torch.no_grad():                                         # a window whose critic goes through it at once, after its last decision:
        _, once, _, _ = T.window(net, critic, lane, a, dict(bc=1., sim=0., value=0.), gen, later=4)       # 4 and 2 of the 6 episodes at a time
        hc, zs = once['hc'], []                                   # the values and the state of the critic run a decision at a time
        for j in range(6):
            z, hc = critic.steps(*(once[q][:, j:j+1] for q in ('obs', 'prev', 'act')), hc, (once['lab'][:, j:j+1], lane.cknown), once['feats'][:, j:j+1])
            zs.append(z[:, 0])
    assert once['z'].shape == rec['z'].shape and float(once['z'].abs().sum()) > 0 and not torch.equal(once['hc'], lane.hc)
    assert torch.allclose(torch.stack(zs, 1), once['z'][:, :6], atol=1e-4) and torch.allclose(hc, lane.hc, atol=1e-4)
    off = [float((net.steps(rec['obs'][:, :6], rec['prev'][:, :6], h0[None], (rec['lab'][:, :6], lane.known), f)[0]-mean).abs().max())
           for f in (rec['feats'], rec['feats'].half().float(), torch.zeros_like(rec['feats']))]
    assert off[0] < 2e-5 and off[1] < 1e-4 and off[2] > 20*off[1]                        # given as kept, the same moves; without them, others
    before = [q.detach().clone() for q in net.parameters()]
    lane.k = T.HORIZON-4                                          # a window cut short by the episode's end
    log = T.imitate(net, critic, opt, copt, lane, buf, a, gen)
    assert buf.n == 2*n and lane.done and log['updates'] == 3 and log['skip'] == 0 and math.isfinite(log['fit']) and log['bc'] is not None
    last = {q: v[n:2*n] for q, v in buf.x.items()}
    assert last['obs'].shape[:2] == (n, 6) and not last['m'][:, 4:].any() and not last['obs'][:, 4:].any() and float(last['m'][:, :4].sum()) > 0
    assert not last['feats'][:, 4:].any() and float(last['feats'][:, :4].abs().sum()) > 0
    assert any(not torch.equal(x, q) for x, q in zip(before, net.parameters())) and 'critic' in log
    fits = [T.learn(net, opt, buf, a, gen)['fit'] for _ in range(40)]
    assert sum(fits[-5:]) < sum(fits[:5])


def test_the_average_of_the_weights_follows_the_steps_and_never_from_far_behind():
    """--ema: after every step of the optimizer the average moves towards the weights; early on it is their
    plain mean over the last few steps (a start at the untrained weights is forgotten), later the decay given."""
    torch.manual_seed(0)
    net = torch.nn.Linear(3, 2)
    opt = torch.optim.SGD(net.parameters(), lr=1.)
    avg = T.Average(net, .9)
    opt.register_step_post_hook(avg)
    start = net.weight.detach().clone()
    for i in range(60):
        opt.zero_grad()
        (net.weight-1.).square().sum().backward()                 # one step puts the weights on the other side of 1, the next back
        opt.step()
        assert not avg.avg.weight.requires_grad
        if i == 0:                                                # the first step: a tenth of the way is kept of the start
            assert torch.allclose(avg.avg.weight, .1*start+.9*net.weight.detach(), atol=1e-6)
    assert avg.n == 60 and torch.allclose(avg.avg.weight, torch.ones(2, 3), atol=.12*float((start-1).abs().max()))
    assert float((net.weight-1).abs().max()) == pytest.approx(float((start-1).abs().max()), rel=1e-4)      # while the weights still swing as far as at first


def test_the_e_read_out_s_error_is_read_on_the_plants_of_the_prior_and_of_the_engines_apart():
    """`e_by_kind`: `head` over the episodes on plants of the prior and over those on mechanistic plants, each a column of
    its own (head_lin, head_mech) where a step has episodes of the kind; a gathered window of a batch that has both
    gives both, and `head` is between them."""
    err = torch.tensor([[[1., 3.]], [[5., 7.]], [[9., 11.]]])                           # (episode, decision, row)
    wt = torch.ones_like(err)
    wt[2, 0, 1] = 0.
    assert T.e_by_kind(err, wt, torch.tensor([1., 1., 0.])) == dict(head_lin=pytest.approx(4.), head_mech=pytest.approx(9.))
    assert T.e_by_kind(err, wt, torch.ones(3)) == dict(head_lin=pytest.approx(5.))       # no episode of the engines: no column, not a 0
    n = 8
    a = args(noise=0., replay=1, batch=8, keep=2, fit_chunk=4, law=True, lr=1e-3, known=.5, clip=10., builders=0)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    net, critic = N.AIO(law=True), N.Critic(2)
    opt, copt = torch.optim.Adam(net.parameters(), lr=1e-3), torch.optim.Adam(critic.parameters(), lr=3e-4)
    prog = T.drawn(n, 5, 'cpu', 'cpu', ('qtank',), mech=.5)
    assert 0 < prog.nl < n
    lane, buf = T.Lane(prog, a, gen, teach=1.), T.Buffer(a.keep*n)
    log = T.imitate(net, critic, opt, copt, lane, buf, a, gen)
    assert torch.equal(buf.x['nl'][:n], (torch.arange(n) < prog.nl).float())
    lo, hi = sorted((log['head_lin'], log['head_mech']))
    assert lo <= log['head'] <= hi and lo < hi


def test_plants_drawn_and_not_run_are_problems_of_design_the_k_read_out_learns_from():
    """--design: of a plant drawn and not run, the packet of its first decision, its true responses and the teacher's
    K of its first model are what the first decision of an episode on it gives a gathered window. A pool of them
    is filled by sets (by a builder: the set drawn in place), each read once before it is learned from
    (`design_new`); a step of `learn` then has their term too (`design`), whose gradient is the K read-out's, and
    steps bring the read-out towards the teacher's K on plants it has not been given."""
    n = 8
    a = args(noise=0., replay=1, batch=8, keep=4, fit_chunk=4, law=True, lr=1e-3, known=.5, clip=10., design=8, design_set=n, design_keep=64, design_coef=1., builders=0)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    net, critic = N.AIO(law=True), N.Critic(2)
    opt, copt = torch.optim.Adam(net.parameters(), lr=1e-3), torch.optim.Adam(critic.parameters(), lr=3e-4)
    lane, buf = T.Lane(C.build(n, 5), a, gen, teach=1.), T.Buffer(a.keep*n)
    T.imitate(net, critic, opt, copt, lane, buf, a, gen)
    x = T.design_set(n, 5, net.pairs.nk)
    assert x['K'].shape == (n, N.NMV, N.NMV, N.NLAB) and x['pairs'].shape == (n, N.NMV, N.NMV) and float(x['pairs'].sum()) > 0
    assert torch.allclose(x['obs'].float(), buf.x['obs'][:n, 0], atol=2e-3, rtol=2e-3) and torch.allclose(x['lab'].float(), buf.x['lab'][:n, 0].float(), atol=2e-3, rtol=2e-3)
    assert torch.equal(x['K'], buf.x['K'][:n, 0]) and torch.equal(x['pairs'].amax(2) > 0, buf.x['trk'][:n] > 0)
    seeds = iter(range(200, 300))
    ahead = T.Ahead(1, n, lambda: 0, threads=1)
    try:                                                                                 # a builder's sets, in the order asked for
        ds = T.Designs(a, ahead.pool, lambda: next(seeds), 'cpu', net.pairs.nk)
        ds.more(net)
        first = T.design_set(n, 200, net.pairs.nk)
        assert ds.buf.n >= n and all(torch.equal(ds.buf.x[q][:n], v) for q, v in first.items())
    finally:
        ahead.close()
    assert ds.log()['design_new'] > 0 and ds.log()['design_new'] is None                 # read as it came in, once
    ds = T.Designs(a, None, lambda: next(seeds), 'cpu', net.pairs.nk, every=1)           # drawn in the run's own process: a set a call
    for _ in range(4):
        ds.more(net)
    assert ds.buf.n == 4*n and 'feats' in ds.buf.x
    one = {q: v[:n] for q, v in ds.buf.x.items()}
    with torch.no_grad():                                                                # the sums worked out once: the same error as at every step
        assert float(T.design_error(net, one)) == pytest.approx(float(T.design_error(net, {q: v for q, v in one.items() if q != 'feats'})), rel=2e-3, abs=1e-4)
    net.zero_grad()
    ds.loss(net, gen).backward()
    assert net.pairs.gain[-1].weight.grad.abs().sum() > 0 and net.pairs.out.weight.grad is None      # the K read-out's, not the moves'
    steps = [T.learn(net, opt, buf, a, gen, ds) for _ in range(150)]
    mean = lambda rows, q: sum(r[q] for r in rows)/len(rows)                             # noqa: E731
    assert mean(steps[-20:], 'design') < .8*mean(steps[:20], 'design')
    ds.more(net)                                                                         # (the next set: plants not learned from)
    assert T.learn(net, opt, buf, a, gen)['design'] is None                              # without a pool: the step as it was
    a.design_every = 3                                                                   # --design-every: one step of three takes them
    every = T.Designs(a, None, lambda: next(seeds), 'cpu', net.pairs.nk, every=1)
    every.more(net)
    took = [T.learn(net, opt, buf, a, gen, every)['design'] is not None for _ in range(7)]
    assert took == [False, False, True, False, False, True, False] and ds.stride == 1
    a.replay, a.unseen = 4, True                                                         # --unseen's ranks: of a window's last step alone
    calls, real = [], T.unseen_log
    T.unseen_log = lambda *k: (calls.append(1), real(*k))[1]
    try:
        unet = N.AIO(law=True, unseen=True)
        uopt = torch.optim.Adam(unet.parameters(), lr=1e-3)
        ulane, ubuf = T.Lane(C.build(n, 5), a, gen, teach=1.), T.Buffer(a.keep*n)
        log = T.imitate(unet, critic, uopt, copt, ulane, ubuf, a, gen)
        assert len(calls) == 1 and math.isfinite(log['unseen']) and math.isfinite(log['gap_un'])
        T.learn(unet, uopt, ubuf, a, gen)
        assert len(calls) == 2                                                           # a step on its own: logged
    finally:
        T.unseen_log = real


def test_the_teacher_of_the_instruments_alone_is_what_is_imitated_and_the_unseen_is_not_told():
    """--meter: a lane's teacher corrects its prediction from the instruments alone and does not see the unmeasured
    loads move; the E gathered is that teacher's (K, the design of the model, is the same), and an episode that is
    told is not told what the instruments do not show: the E read-out is the same whatever the true one is.
    Without --meter it is told. The references of the instruments alone are run beside the others and are no
    part of the scale."""
    n = 8
    kw = dict(noise=0., replay=1, batch=8, keep=4, fit_chunk=4, law=True, lr=1e-3, known=1., clip=10.)
    bufs, lanes = {}, {}
    for meter in (False, True):
        a = args(meter=meter, **kw)
        torch.manual_seed(0)
        gen = torch.Generator().manual_seed(0)
        net, critic = N.AIO(law=True, unseen=True), N.Critic(2)
        torch.nn.init.normal_(net.pairs.unseen_in.weight, std=.5)                         # (the reading in use by the E read-out)
        opt, copt = torch.optim.SGD(net.parameters(), lr=0.), torch.optim.SGD(critic.parameters(), lr=0.)
        lanes[meter], bufs[meter] = T.Lane(C.build(n, 5), a, gen, teach=1.), T.Buffer(a.keep*n)
        for _ in range(3):
            T.imitate(net, critic, opt, copt, lanes[meter], bufs[meter], a, gen)
        heads = []
        for shift in (0., 3.):                                                           # the same windows, the true `unseen` moved
            x = {q: v.clone() for q, v in bufs[meter].x.items()}
            bufs[meter].x['U'] = x['U']+shift
            heads.append(T.learn(net, opt, bufs[meter], a, torch.Generator().manual_seed(1))['head'])
            bufs[meter].x = x
        assert (heads[0] == heads[1]) == meter                                           # told it, the E read-out follows it; not told, it cannot
    plain, metered = lanes[False], lanes[True]
    assert plain.prog.meter is None and float(plain.prog.ffm[-1]) == 1. and 'eo' not in plain.st
    assert metered.prog.meter is not None and float(metered.prog.ffm[-1]) == 0. and 'eo' in metered.st
    assert torch.equal(bufs[True].x['K'][:n, 0], bufs[False].x['K'][:n, 0])               # the design of the first model: one and the same
    assert not torch.allclose(bufs[True].x['E'].float(), bufs[False].x['E'].float(), atol=1e-3)
    prog = C.build(6, 9)
    ref = T.references(prog, True)
    assert set(T.METERED) < set(ref) and set(T.references(prog)) == set(ref)-set(T.METERED)
    scale, _ = T.scale_of(ref)
    assert scale.best == T.scale_of({q: r for q, r in ref.items() if q not in T.METERED})[0].best
    assert T.before_after(ref['pi'], ref) == T.before_after(ref['pi'], {q: r for q, r in ref.items() if q not in T.METERED})


def test_a_batch_drawn_ahead_by_a_builder_is_the_batch_drawn_in_place():
    """--builders: a batch drawn by another process and moved to the run's device is, tensor for tensor, the
    batch `core.build` draws in place for the same seed, the batches come in the order asked for, and the
    teacher's first rollout on it scores the same."""
    from aio.sim import control as CT
    seeds = iter(range(100, 110))
    ahead = T.Ahead(1, 6, lambda: next(seeds), threads=1)
    try:
        got = [T.moved(ahead(), 'cpu') for _ in range(2)]
        assert len(ahead.pids()) == 1
    finally:
        ahead.close()
    for s, p in zip((100, 101), got):
        ref = C.build(6, s)
        same = lambda x, y: torch.equal(x, y) if torch.is_tensor(x) else True    # noqa: E731
        assert all(same(v, getattr(p, q)) for q, v in vars(ref).items()) and float(ref.Fr.abs().sum()) > 0
        assert all(torch.equal(v, p.c[q]) for q, v in ref.c.items()) and all(torch.equal(v, p.start[q]) for q, v in ref.start.items())
        assert p.seed == ref.seed and p.device == ref.device
    assert torch.equal(CT.rollout(got[0], CT.Teacher(), 40)['score'], CT.rollout(C.build(6, 100), CT.Teacher(), 40)['score'])


def test_a_unit_is_probed_before_its_first_event_and_the_network_learns_to_say_when():
    """--probe: in a gathered window the moves made before an episode's first event carry the probe's noise,
    the others --noise; the buffer keeps which decisions those were, and steps on it bring the head's second
    output towards saying so. As a controller the network adds that noise only where it says so, the same
    at every rollout; a network without --probe runs as before."""
    from aio.sim import control as CT
    n = 8
    a = args(noise=0., replay=40, batch=8, keep=4, fit_chunk=4, probe=.8, lr=1e-3)
    net, critic, _, gen = _setup(a, n=n)
    opt, copt = torch.optim.Adam(net.parameters(), lr=1e-3), torch.optim.Adam(critic.parameters(), lr=3e-4)
    lane = T.Lane(C.build(n, 5), a, gen, teach=1.)
    buf = T.Buffer(a.keep*n)
    with torch.no_grad():
        _, rec, _, _ = T.window(net, critic, T.Lane(C.build(n, 5), a, gen, teach=1.), a, dict(bc=1., sim=0., value=0.), gen, later=4)
        still = T.window(net, critic, T.Lane(C.build(n, 5), SimpleNamespace(**dict(vars(a), probe=0.)), gen, teach=1.),
                         SimpleNamespace(**dict(vars(a), probe=0.)), dict(bc=1., sim=0., value=0.), gen, later=4)[1]
    k_first = lane.prog.k_first
    assert rec['pre'].shape == (n, 6) and torch.equal(rec['pre'], (torch.arange(6)[None] < k_first[:, None]).float()) and float(rec['pre'].sum()) > 0
    assert 'pre' not in still and float((rec['act'][:, :6]-still['act'][:, :6]).abs().amax()) > .05      # other moves where a unit is probed
    first = T.imitate(net, critic, opt, copt, lane, buf, a, gen)
    for _ in range(3):
        lane.k = 0
        last = T.imitate(net, critic, opt, copt, lane, buf, a, gen)
    assert torch.equal(buf.x['pre'][:n], rec['pre']) and last['pre'] < .5*first['pre']
    prog = C.build(n, 5)
    net.probe = .8
    moves = lambda: [r['u'] for r in CT.rollout(prog, N.Policy(net), 12, trace=list(range(n)))['trace']]    # noqa: E731
    one, two = moves(), moves()
    net.probe = 0.
    assert one == two and one != moves()


def test_the_law_is_the_teacher_but_for_its_limits_and_a_network_is_taught_its_two_factors():
    """aio.law: the product K . E / V of the teacher's view is its move wherever no limit of the plan binds, and
    as a controller it loses little to it. --law: a gathered window keeps the teacher's E and K of the tracked
    rows, V and the law's move, and the move to imitate is the law's; steps on it bring the readouts towards
    them; a network told them multiplies to the law's move exactly; one without --law computes what it did,
    whatever the features' width."""
    from aio import law as L
    from aio.sim import control as CT
    n = 8
    prog = C.build(n, 5)
    st = C.reset(prog); prog.set_teacher('late')
    for k in range(40):
        st, _ = C.step(prog, st, C.teacher(prog, st, k), k)
    m, E, K, V = L.move(prog, st, 40)
    free = MD_move_without_limits(prog, st, 40)
    assert torch.allclose(m.clamp(-1., 1.)*prog.c['mvm'], free, atol=.05)                # the unconstrained plan's first move, through the valve curve
    per = {q: CT.rollout(prog, c, 240)['score'] for q, c in dict(hold=CT.Hold(), teacher=CT.Teacher('late'), law=L.Plain()).items()}
    assert float(per['law'].sum()) < float(per['teacher'].sum())+.1*float((per['hold']-per['teacher']).sum())
    a = args(noise=0., replay=1, batch=8, keep=4, fit_chunk=4, law=True, lr=1e-3, known=.5, clip=10.)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    net, critic = N.AIO(law=True), N.Critic(2)
    opt, copt = torch.optim.Adam(net.parameters(), lr=1e-3), torch.optim.Adam(critic.parameters(), lr=3e-4)
    lane = T.Lane(C.build(n, 5), a, gen, teach=1.)
    buf = T.Buffer(a.keep*n)
    for _ in range(4):
        T.imitate(net, critic, opt, copt, lane, buf, a, gen)
    assert buf.x['E'].shape == (a.keep*n, 6, N.NMV, N.NLAB) and buf.x['K'].shape == (a.keep*n, 6, N.NMV, N.NMV, N.NLAB)
    on = buf.x['m'] > 0
    gap = (buf.x['tch']-buf.x['lm'].clamp(-1., 1.))[on].abs()                            # what is imitated: the law's move, held to the limits
    assert bool((buf.x['tch'].abs() <= buf.x['lm'].abs()+1e-5)[on].all()) and float(gap.median()) < 1e-5
    steps = [T.learn(net, opt, buf, a, gen) for _ in range(200)]                         # on what is in the buffer now
    mean = lambda rows, q: sum(r[q] for r in rows)/len(rows)                             # noqa: E731
    assert mean(steps[-20:], 'head') < .5*mean(steps[:20], 'head') and mean(steps[-20:], 'gain') < .9*mean(steps[:20], 'gain')
    assert all(r['law'] == r['law'] for r in steps)
    untold = [r for r in steps if r['k_own'] is not None]                                # K on the episodes not told, own row and the others: no number of a step that drew none
    assert untold and all(r['k_own'] > 0 and r['k_cross'] > 0 for r in untold)
    b = args(noise=0., law=True, law_keep='EK', known=.5, clip=10.)                    # and beside the simulator gradient (--law-keep K)
    loss, _, _, log = T.window(net, critic, T.Lane(C.build(n, 6), b, gen, teach=0.), b, dict(bc=0., sim=1., value=1.), gen)
    assert log['k_own'] >= 0 and log['k_cross'] >= 0 and log['gain'] > 0 and bool(torch.isfinite(loss))
    err, w = torch.rand(4, 3, 5, 5), torch.ones(4, 3, 5, 5)
    known = torch.tensor([1., 0., 1., 0.])
    link = torch.zeros(4, 5, 5)
    link[:, [0, 1, 2, 3, 4], [2, 0, 1, 4, 3]] = 1.                                       # the pairs: a permutation of the rows, not the diagonal
    pair = link[0].bool()
    got = T.k_untold(err, w, known, link)
    assert got['k_own'] == pytest.approx(float(err[[1, 3]][..., pair].mean()), rel=1e-5) and got['k_cross'] == pytest.approx(float(err[[1, 3]][..., ~pair].mean()), rel=1e-5)
    assert T.k_untold(err, w, torch.ones_like(known), link) == {}                         # every episode told: no column, not a 0
    # told: the product of the teacher's own E and K is the law's move
    x = {q: v[:n] for q, v in buf.x.items()}
    grow = N.NCV-N.NMV
    code = (x['lab'], torch.ones(n), torch.nn.functional.pad(x['E'].float(), (0, 0, 0, grow)), torch.nn.functional.pad(x['K'].float(), (0, 0, 0, 0, 0, grow)), x['V'])
    with torch.no_grad():
        net.steps(x['obs'], x['prev'], x['h0'][None], code, x['feats'].float())
    mv, on = net.pairs.read_law[3], x['m'] > 0
    assert torch.allclose(torch.asinh(mv)[on], torch.asinh(x['lm'])[on], atol=.02)
    # a network without the law: the wider features of a trainer that has it change nothing
    plain = N.AIO()
    obs, prev = x['obs'][:, :3], x['prev'][:, :3]
    f = plain.pairs.stats(obs)[0]
    wide = torch.cat((f, torch.randn(*f.shape[:-1], N.NF)), -1)
    with torch.no_grad():
        assert torch.equal(plain.steps(obs, prev, feats=f)[0], plain.steps(obs, prev, feats=wide)[0])


def MD_move_without_limits(prog, st, k):
    """The first move of the teacher's plan with no limit in it, in slew limits, through the valve curve."""
    from aio.sim import model as MD
    v = C.teacher_view(prog, st, k)
    big = torch.full((prog.n, MD.M_DMC, prog.dims[1]), 1e9, dtype=prog.dtype)
    dq = MD.dmc_move(prog.Sw[prog.ar, v['sg']], prog.HX[prog.ar, v['sg']], v['E']*v['r'].clamp_max(1.)[:, None, None], -big, big)
    un = (v['q']+dq).clamp(0., 1.)**(1/v['g'])
    return ((un-v['u'])/v['sl']).clamp(-1., 1.)*prog.c['mvm']


def test_the_hold_on_the_actor_s_output_measures_nothing():
    g = torch.ones(4, 5)
    g[1] *= 1e4
    g[2, 0] = float('nan')
    one = float(g[0].norm())
    gd = T.Guard(4, 10., g.device, usual=one)
    out = gd.hold(g)
    assert torch.equal(out[0], g[0]) and torch.allclose(out[1].norm(), torch.tensor(10*one)) and not out[2].any()
    assert not gd.top.any() and gd.median() is None and gd.held().tolist() == [False, True, True, False]
    assert torch.equal(T.Guard(4, 10., g.device).hold(g)[1], g[1])                       # no level yet: nothing is held


def test_the_first_window_with_the_simulator_gradient_measures_and_a_spike_is_not_stepped_on():
    """Nothing is held while the guard has no level, so that window's step would be whatever its wildest episode
    says: it is measured and not taken. Later a step far above the recent ones is not taken either; the critic is
    fitted and the lane moves on all the same."""
    a = args(spike=10.)
    net, critic, lane, gen = _setup(a)
    opt, copt = torch.optim.Adam(net.parameters(), lr=1e-3), torch.optim.Adam(critic.parameters(), lr=1e-3)
    w, temp = dict(bc=0., sim=1., value=1.), dict(usual=None)
    same = lambda old: all(torch.equal(x, y) for x, y in zip(old, net.parameters()))
    keep = lambda: [q.detach().clone() for q in net.parameters()]
    calls, orig = [], T.Guard.hold

    def spy(self, g):
        calls.append(self.limit)
        return orig(self, g)
    T.Guard.hold = spy
    try:
        old, c0 = keep(), [q.detach().clone() for q in critic.parameters()]
        log = T.update(net, critic, opt, copt, lane, a, w, gen, temp)
        assert log['skip'] == 1 and same(old) and temp['usual'] > 0 and temp['gns'] == [log['far']] and 'gn' not in log and lane.k == 6
        assert any(not torch.equal(x, y) for x, y in zip(c0, critic.parameters()))       # the critic had its fit
        assert len(calls) == 6 and calls[0] is None                                      # the hold sits at every decision's output
        log = T.update(net, critic, opt, copt, lane, a, w, gen, temp)                    # the level is known: a step
        assert log['skip'] == 0 and not same(old) and len(calls) == 12 and calls[-1] > 0
        temp['gns'] = [1e-9]*8                                                           # recent steps far below this one
        old = keep()
        log = T.update(net, critic, opt, copt, lane, a, w, gen, temp)
        assert log['skip'] == 1 and same(old) and len(temp['gns']) == 9 and temp['gns'][-1] == pytest.approx(10*1e-9)   # a step not taken counts at the most a step may be: no ratchet
        assert log['far'] > 10*1e-9
        a.spike = 0.                                                                     # --spike 0: every step is taken
        assert T.update(net, critic, opt, copt, lane, a, w, gen, temp)['skip'] == 0 and not same(old)
        n = len(calls)
        T.update(net, critic, opt, copt, lane, a, dict(bc=1., sim=0., value=0.), gen, temp)
        assert len(calls) == n                                                           # imitation alone: the plant is run without the graph, nothing to hold
    finally:
        T.Guard.hold = orig


def test_heads_of_fixed_horizons_are_fitted_on_costs_and_on_the_head_before_never_on_themselves():
    """--fixed S: head m is the discounted cost of the next S*(m+1) decisions. Its target is the cost of the first
    S of them and head m-1's value S decisions on; head 0's has costs alone in it. What a head says is in no
    target of its own, and what the longest says is in none at all."""
    S, g, M = 4, .9, 3
    true = torch.tensor([2.*(1-g**(S*(m+1)))/(1-g) for m in range(M)])                   # a cost of 2 at every decision
    cost, z = torch.full((2, 6+S), 2.), T.symlog(true).expand(2, 6+S, M).clone()
    tgt = T.fixed_targets(cost, z, S, g)
    assert tgt.shape == (2, 6, M) and torch.allclose(T.symexp(tgt), true.expand(2, 6, M), rtol=1e-5)
    more = z.clone(); more[..., -1] += 3.
    assert torch.equal(T.fixed_targets(cost, more, S, g), tgt)
    more = z.clone(); more[..., 0] += 1.
    t = T.fixed_targets(cost, more, S, g)
    assert torch.equal(t[..., 0], tgt[..., 0]) and bool((t[..., 1] > tgt[..., 1]).all()) and torch.equal(t[..., 2], tgt[..., 2])
    cost = torch.rand(3, 5+S)
    t = T.fixed_targets(cost, torch.zeros(3, 5+S, 1), S, g)
    assert torch.allclose(T.symexp(t[..., 0]), sum(g**i*cost[:, i:i+5] for i in range(S)), atol=1e-5)
    a = SimpleNamespace(fixed=S, gamma=g, heads=M)
    assert T.reach(a) == pytest.approx(float(true[-1])/2) and T.reach(a, 0) == pytest.approx(float(true[0])/2)
    assert T.reach(SimpleNamespace(gamma=g)) == pytest.approx(10.)
    assert torch.equal(T.value_of(z, a), T.symexp(z[..., -1])) and torch.equal(T.value_of(z, SimpleNamespace()), T.symexp(z.mean(-1)))


def test_a_window_waits_for_the_decisions_after_it_and_a_run_that_ends_goes_on_as_it_was():
    """--fixed: a window is fitted once the S decisions after its last have been run. Where a unit's run ends (an
    episode that failed and starts again; every unit when the lane is released) the targets of the decisions
    before the end take the cost to go on as over the run's last S decisions: not the next run's costs, and no
    value of the critic's. The next run's own windows are fitted on their own costs."""
    n, S, g = 3, 4, .9
    due = T.Due(SimpleNamespace(fixed=S, gamma=g, heads=2))
    r0 = (1-g**S)/(1-g)

    def rec(H, c):                                                                       # a window of H decisions that cost c each; the heads say 0
        return dict(obs=torch.arange(n*(H+1.)).reshape(n, H+1, 1), prev=torch.zeros(n, H+1, 1), act=torch.zeros(n, H+1, 1), lab=torch.zeros(n, H+1, 1),
                    hc=torch.zeros(1, n, 2), feats=None, cost=torch.full((n, H), c), z=torch.zeros(n, H+1, 2), tch=torch.zeros(n, H, 1))
    head0 = lambda *c: sum(g**i*x for i, x in enumerate(c))                              # noqa: E731
    due.add(rec(4, 1.))
    assert not due.ready()
    due.add(rec(2, 2.))
    assert not due.ready()                                                               # two of the four decisions the first waits for
    due.gone(torch.tensor([True, False, False]))                                         # unit 0 failed: its next episode is another run
    due.add(rec(4, 5.), slice(0, 2))                                                     # (a window fitted on two of its episodes)
    assert due.ready()
    r, part, tgt, said = due.pop()
    assert part is None and set(r) == {'obs', 'prev', 'act', 'lab', 'hc', 'feats'} and tgt.shape == (n, 4, 2) and said == 0.
    v = T.symexp(tgt)
    assert v[1, 0, 0] == pytest.approx(head0(1, 1, 1, 1)) and v[1, 3, 0] == pytest.approx(head0(1, 2, 2, 5)) and v[0, 3, 0] == pytest.approx(head0(1, 2, 2, 1.5))
    assert v[1, 3, 1] == pytest.approx(head0(1, 2, 2, 5)) and v[0, 3, 1] == pytest.approx(head0(1, 2, 2, 1.5)+g**S*1.5*r0)   # after its end: 1.5 a decision, as over its last four
    assert v[0, 1, 1] == pytest.approx(head0(1, 1, 1, 2))                                # (the head before said 0 while the run was on)
    assert due.ready()
    _, _, tgt, _ = due.pop()
    v = T.symexp(tgt)
    assert tgt.shape == (n, 2, 2) and v[1, 0, 0] == pytest.approx(head0(2, 2, 5, 5)) and v[0, 0, 0] == pytest.approx(head0(2, 2, 1.5, 1.5))
    assert not due.ready() and len(due.recs) == 1 and not due.ends                       # the next run's first window waits; the end is done with
    due.last()                                                                           # the lane is released
    r, part, tgt, _ = due.pop()
    assert part == slice(0, 2) and r['obs'].shape[0] == 2 and r['hc'].shape[1] == 2 and tgt.shape == (2, 4, 2)
    assert torch.allclose(T.symexp(tgt[..., 0]), torch.full((2, 4), 5*r0), rtol=1e-5)    # unit 0's new run on its own costs
    assert torch.allclose(T.symexp(tgt[..., 1]), torch.full((2, 4), 5*r0*(1+g**S)), rtol=1e-5)
    assert not due.recs and not due.ready()


def test_the_critic_with_heads_of_fixed_horizons_is_fitted_a_window_late_and_on_all_windows_in_the_end():
    """--fixed in the trainer: the first window of a lane leaves no fit behind, every later one fits the window
    it completes; when the lane is released the last ones are fitted too. The actor's objective has the longest head."""
    a = args(fixed=6, heads=2, critic_iters=1)
    net, critic, lane, gen = _setup(a)
    opt, copt = torch.optim.Adam(net.parameters(), lr=1e-3), torch.optim.Adam(critic.parameters(), lr=1e-3)
    temp, w = dict(usual=None), dict(bc=0., sim=1., value=1.)
    logs = [T.update(net, critic, opt, copt, lane, a, w, gen, temp) for _ in range(3)]
    assert 'critic' not in logs[0] and all('critic' in x and math.isfinite(x['critic']) and 'value' in x for x in logs[1:])
    assert len(lane.due.recs) == 1 and lane.due.cost.shape[1] == a.window
    before = [q.detach().clone() for q in critic.parameters()]
    out = T.fit_due(critic, copt, lane, a, last=True)
    assert math.isfinite(out['critic']) and not lane.due.recs and any(not torch.equal(x, y) for x, y in zip(before, critic.parameters()))
    with torch.no_grad():                                                                # the longest head alone moves the objective
        s = dict(st={q: x.clone() for q, x in lane.st.items()}, ha=lane.ha.clone(), hc=lane.hc.clone(), hs=lane.hs.clone(), prev=lane.prev.clone(), pm=lane.pm.clone(), k=lane.k, g=lane.prog.g.get_state())
        got = []
        for bias in (0., 0., 2.):
            lane.st, lane.ha, lane.hc, lane.hs, lane.prev, lane.pm, lane.k = {q: x.clone() for q, x in s['st'].items()}, s['ha'].clone(), s['hc'].clone(), s['hs'].clone(), s['prev'].clone(), s['pm'].clone(), s['k']
            lane.prog.g.set_state(s['g'])
            last = [q for q in critic.heads[0].parameters()][-1]
            last += bias
            got.append(float(T.window(net, critic, lane, a, w, torch.Generator().manual_seed(3), usual=1e-3)[3]['sim']))
            last -= bias
        assert got[0] == got[1]


def test_a_batch_of_a_larger_size_trains_as_one_of_the_default_size_does():
    """--sizes: a batch is of one size, taken by its seed's own draw, with as many pairs as one of --envs plants of the
    default size. A lane of 24x10 carries the state of that size (the network's own count of it, with the slow sums and
    the rows' own past), a window gathered there goes through the network at once from the state it was kept in and
    gives the moves made a decision at a time, and its buffer, its problems of design and a window with the simulator
    gradient run as they do at 12x5, the K read-out's error on the episodes not told included."""
    sh = T.sizes_of(['4:12x5', '1:24x10', '1:48x20@7'], 768)
    assert sh == ((4., '12x5', 768), (1., '24x10', 192), (1., '48x20', 7)) and T.sizes_of(None, 768) is None
    got = [T.size_of(sh, s) for s in range(600)]
    even = [T.size_of(sh, s, T.evenly(s, 0, 'z')) for s in range(600)]  # by an even sequence: the shares within one batch in every stretch
    assert all(abs(even[i:i+120].count(x)-120*w/sum(y[0] for y in sh)) <= 2.5 for x, w in ((tuple(q[1:]), q[0]) for q in sh) for i in (0, 240, 480))
    assert got == [T.size_of(sh, s) for s in range(600)] and 330 < got.count(('12x5', 768)) < 470 and ('48x20', 7) in got
    assert T.per_pairs(256, '24x10') == 64 and T.per_pairs(256, '48x20') == 16 and T.per_pairs(256, '12x5') == 256
    n, size = 4, '24x10'
    b = args(noise=0., replay=0, batch=4, keep=3, fit_chunk=2, slow=600., arx=True, dv=True)
    torch.manual_seed(0)
    net, critic = N.AIO(slow=600., arx=True, dv=True), N.Critic(2)
    for m in (net.pairs.stat[-1], net.pairs.out):                                        # the pair pathway and the sums in play
        torch.nn.init.normal_(m.weight, std=.05)
    opt, copt = torch.optim.SGD(net.parameters(), lr=0.), torch.optim.SGD(critic.parameters(), lr=0.)
    lanes = [T.Lane(C.build(n, 5, dims=size), b, torch.Generator().manual_seed(0), teach=1.) for _ in range(2)]
    lane = lanes[0]
    assert lane.size == size and lane.ha.shape[-1] == net.state_size(24, 10) and lane.hc.shape[-1] == critic.state_size(24, 10) and lane.prev.shape == (n, 10)
    buf = T.Buffer(b.keep*n, T.per_pairs(b.batch, size))
    for x in lanes:                                                                      # a first window, gathered only
        T.imitate(net, critic, opt, copt, x, T.Buffer(b.keep*n), b, torch.Generator())
    with torch.no_grad():
        _, rec, _, _ = T.window(net, critic, lanes[0], b, dict(bc=1., sim=0., value=0.), torch.Generator())
    T.imitate(net, critic, opt, copt, lanes[1], buf, b, torch.Generator())               # the same window, kept
    x = {q: v[:n] for q, v in buf.x.items()}
    assert buf.batch == 1 and x['obs'].shape[-1] == 24*12+10*6+24*10 and x['h0'].shape[-1] == lane.ha.shape[-1] and float(lanes[1].hs.abs().sum()) > 0
    with torch.no_grad():
        mean, _, mask, _, _ = net.steps(x['obs'], x['prev'], x['h0'][None], (x['lab'], x['known']), x['feats'].float())
    assert torch.allclose(torch.tanh(mean)*mask, rec['act'][:, :b.window], atol=1e-6) and float(rec['act'].abs().amax()) > 1e-3
    a = args(noise=0., replay=1, batch=8, keep=3, fit_chunk=2, law=True, lr=1e-3, known=.5, clip=10., unseen=True, meter=True, unpaired=.5,
             design=8, design_set=8, design_keep=64, design_coef=1., builders=0, law_keep='EK')
    gen = torch.Generator().manual_seed(0)
    net, critic = N.AIO(law=True, unseen=True), N.Critic(2)
    opt, copt = torch.optim.Adam(net.parameters(), lr=1e-3), torch.optim.Adam(critic.parameters(), lr=3e-4)
    lane, buf = T.Lane(C.build(n, 5, dims=size), a, gen, teach=1.), T.Buffer(a.keep*n, T.per_pairs(a.batch, size))
    ds = T.Designs(a, None, iter(range(300, 400)).__next__, 'cpu', net.pairs.nk, dict(dims=size), every=1, b=T.per_pairs(a.design, size), n=4, keep=16)
    log = T.imitate(net, critic, opt, copt, lane, buf, a, gen, ds)
    assert buf.n == n and buf.batch == 2 and ds.b == 2 and ds.plants == 4 and ds.buf.x['K'].shape[1:3] == (10, 10) and buf.x['K'].shape[2:4] == (10, 10)
    assert all(math.isfinite(log[q]) for q in ('fit', 'gain', 'head', 'design', 'unseen')) and log['skip'] == 0
    more = [T.learn(net, opt, buf, a, gen, ds) for _ in range(6)]                         # (a step that draws told windows alone has no number for the untold)
    assert float(lane.known.sum()) < n and any(r['k_own'] is not None and r['k_own'] > 0 and r['k_cross'] > 0 for r in more)
    sim = T.Lane(C.build(n, 6, dims=size), a, gen)                                       # the simulator gradient on a lane of this size, its own guard
    temp = dict(usual=None)
    logs = [T.update(net, critic, opt, copt, sim, a, dict(bc=0., sim=1., value=0.), gen, temp) for _ in range(2)]
    assert temp['usual'] is not None and all(math.isfinite(r['sim']) for r in logs) and logs[1]['skip'] == 0 and 'k_cross' in logs[1]


def test_batches_read_now_and_then_wait_off_the_card_and_are_on_it_for_a_use(monkeypatch):
    """`parked`, `visit`: a batch built on the card is kept on the CPU and is on the card while it is read (the validation
    batches of a run; a lane's plants in --bench --sizes), back where it was whatever the use did; `Buffer.to`: the
    windows of a size no lane is of."""
    moves = []
    monkeypatch.setattr(T, 'kind', lambda d: 'cuda' if str(d).startswith('card') else 'cpu')
    monkeypatch.setattr(T, 'moved', lambda p, d: (moves.append((p.name, str(d))), setattr(p, 'device', d), p)[-1])
    a, b = SimpleNamespace(name='a', device='card'), SimpleNamespace(name='b', device='cpu')
    T.parked(a, b)
    assert moves == [('a', 'cpu')] and a.device == 'cpu'
    with T.visit('card', a, b):
        assert a.device == b.device == 'card'
    assert a.device == b.device == 'cpu' and moves[1:] == [('a', 'card'), ('b', 'card'), ('a', 'cpu'), ('b', 'cpu')]
    c = SimpleNamespace(name='c', device='card')
    with T.visit('card', c):                                                             # one that is there is left there
        pass
    assert c.device == 'card' and len(moves) == 5
    with pytest.raises(RuntimeError):
        with T.visit('card', a):
            raise RuntimeError
    assert a.device == 'cpu'
    monkeypatch.undo()
    buf = T.Buffer(4)
    assert buf.to('cpu') is buf                                                          # nothing kept yet
    buf.add(dict(x=torch.ones(2, 3)))
    assert buf.to('cpu').x['x'].device.type == 'cpu' and buf.n == 2


def test_no_cap_on_a_cpu():
    assert T.cap_card(torch.device('cpu')) is None and T.cap_card(torch.device('cpu'), 3.) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='a card')
def test_the_allocator_gives_back_what_it_keeps_before_it_holds_more_than_its_cap():
    """Blocks kept from tensors of one size fit none of another: uncapped, the allocator takes more of the card for them
    (on Windows past the card, into shared memory); capped, it gives back what it keeps and tries again first, and an
    allocation that does not fit even then is out of memory."""
    dev, mb = torch.device('cuda'), 2**20
    torch.cuda.empty_cache()
    assert T.cap_card(dev, 2.) == 2.
    try:
        torch.cuda.reset_peak_memory_stats(dev)
        a = [torch.empty(300*mb, dtype=torch.uint8, device=dev) for _ in range(5)]
        del a                                                   # 1.5 GB kept
        b = [torch.empty(700*mb, dtype=torch.uint8, device=dev) for _ in range(2)]
        assert torch.cuda.max_memory_reserved(dev) <= 2*2**30 and len(b) == 2
        del b
        with pytest.raises(torch.cuda.OutOfMemoryError):
            torch.empty(2500*mb, dtype=torch.uint8, device=dev)
    finally:
        torch.cuda.set_per_process_memory_fraction(1., torch.cuda.current_device())
        torch.cuda.empty_cache()


def test_a_run_that_gives_no_sign_of_life_is_ended_and_started_again_from_its_last_row(tmp_path):
    """`watched` (--restarts): a child that stops giving signs of life is ended once twice --stall has passed and started
    again with --resume; one that ends by itself is not, and its exit code is the watcher's; one that stalls again
    after the restarts allowed ends the watch with 3. `alive` writes the sign the watcher goes by, `timed` adds up."""
    import sys
    child = ("import sys, time, pathlib\n"
             "out = pathlib.Path(sys.argv[1])\n"
             "(out/'runs.txt').open('a').write(' '.join(sys.argv[2:])+'\\n')\n"
             "if '--resume' in sys.argv or 'ends' in sys.argv:\n"
             "    sys.exit(7 if 'ends' in sys.argv else 0)\n"
             "(out/'alive').write_text('a window')\n"
             "time.sleep(60)\n")
    cmd = [sys.executable, '-c', child, str(tmp_path)]
    assert T.watched(cmd, tmp_path, .2, 2, grace=.2, poll=.05) == 0                        # stalls, is started again with --resume, ends
    assert (tmp_path/'runs.txt').read_text().split('\n')[:2] == ['', '--resume']
    note = json.loads((tmp_path/'restarts.jsonl').read_text())
    assert note['restart'] == 1 and note['last'] == 'a window' and note['idle_s'] >= 0
    assert T.watched(cmd+['ends'], tmp_path, .2, 2, grace=.2, poll=.05) == 7                # an end of its own (an error too) is no stall
    assert len((tmp_path/'restarts.jsonl').read_text().splitlines()) == 1
    stuck = [sys.executable, '-c', "import time; time.sleep(60)", '--resume']
    assert T.watched(stuck, tmp_path, .2, 1, grace=.2, poll=.05) == 3                      # stalls whatever it is started with: twice, then no more
    rows = [json.loads(x) for x in (tmp_path/'restarts.jsonl').read_text().splitlines()]
    assert [r['restart'] for r in rows] == [1, 1, None]
    was = dict(T.SIGN)
    try:
        T.SIGN.update(stall=0., path=tmp_path/'alive')
        T.alive(row=3, windows=70)
        assert {q: v for q, v in json.loads((tmp_path/'alive').read_text()).items() if q != 'time'} == dict(row=3, windows=70)
    finally:
        T.SIGN.update(was)
    took = {}
    assert T.timed(took, 'val', lambda x: x+1, 1) == 2 and T.timed(took, 'val', lambda: None) is None and took['val'] >= 0
    import subprocess                                                                    # `ends_with`: a watched run ends as any when it is done, and at once when its watcher is gone
    run = lambda body: subprocess.Popen([sys.executable, '-c', 'import sys, time\nfrom aio import train as T\nT.ends_with(sys.stdin.fileno())\n'+body],     # noqa: E731
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, start_new_session=True)
    done, left = run('time.sleep(.5)'), run('print("up", flush=True)\ntime.sleep(300)')
    assert done.wait(timeout=120) == 0
    assert left.stdout.readline().strip() == b'up'
    left.stdin.close()
    assert left.wait(timeout=30) != 0


def test_the_critic_s_fit_takes_as_many_pairs_at_a_time_at_every_size():
    """--fit-chunk counts episodes of the default size. A lane of a larger size takes as many pairs at a time: the
    chunks the critic goes through a gathered window in, the share of that window's episodes it is fitted on, and the
    chunks of its fit after a window with the graph kept. At the default size nothing changes."""
    sized = lambda q: SimpleNamespace(size=q)                     # noqa: E731
    assert [T.chunk_of(args(fit_chunk=128), sized(q)) for q in ('12x5', '24x10', '48x20')] == [128, 32, 8]
    assert T.chunk_of(args(fit_chunk=0), sized('48x20')) == 0 and T.chunk_of(args(fit_chunk=2), sized('48x20')) == 1
    n, a = 4, args(noise=0., replay=1, batch=4, keep=3, fit_chunk=8)
    for size, at_once in (('24x10', 2), ('12x5', 4)):
        torch.manual_seed(0)
        gen, net, critic = torch.Generator().manual_seed(0), N.AIO(), N.Critic(2)
        opt, copt = torch.optim.SGD(net.parameters(), lr=0.), torch.optim.SGD(critic.parameters(), lr=0.)
        lane = T.Lane(C.build(n, 5, dims=size), a, gen, teach=1.)
        seen, steps = [], critic.steps
        critic.steps = lambda obs, *k, **kw: (seen.append(len(obs)), steps(obs, *k, **kw))[1]
        T.imitate(net, critic, opt, copt, lane, T.Buffer(a.keep*n), a, gen)
        # the window in chunks, the decision after it on all, then --critic-iters fits on the first chunk alone
        assert seen == [at_once]*(n//at_once)+[n]+[at_once]*a.critic_iters
        with torch.no_grad():
            _, rec, _, _ = T.window(net, critic, lane, a, dict(bc=1., sim=0., value=0.), gen)
        del seen[:]
        log = T.fit_critic(critic, copt, lane, rec, a)             # after a window run a decision at a time: every episode, a chunk at a time
        assert seen == [at_once]*(n//at_once)*a.critic_iters and math.isfinite(log['critic'])


def test_the_systems_of_the_largest_size_are_not_solved_by_a_batched_lu(monkeypatch):
    """48x20 has 160 unknowns in the law's table and 132 in the network's fit with the measured disturbances. The
    batched LU of 160 does not return on a threaded MKL (the builder that draws problems of design at that size
    hung on it), so both go by Cholesky (aio.sim.model.spd_solve): the same numbers as each system by LU on
    its own, one that is not positive definite still solved, one with no solution 0, the default size by LU bit
    for bit; and the law there is the teacher's unconstrained move, as at the default size."""
    from aio import law as L
    from aio.sim import model as MD
    lu, seen, f = torch.linalg.solve_ex, [], torch.float64
    monkeypatch.setattr(torch.linalg, 'solve_ex', lambda A, B, **kw: (seen.append((A.shape[-1], A.dim())), lu(A, B, **kw))[1])
    batched = lambda: [k for k, dim in seen if k > MD.LU_MAX and dim > 2]                # noqa: E731
    g = torch.Generator().manual_seed(0)
    rand = lambda *shape: torch.randn(*shape, generator=g, dtype=f)                      # noqa: E731

    def gram(*shape):                                                                    # (.., k, k) positive definite
        A = rand(*shape, 2*shape[-1])
        return A@A.transpose(-1, -2)/(2*shape[-1])
    H, G = gram(5, 40), rand(5, 40, 7)
    assert torch.equal(MD.spd_solve(H, G), lu(H, G)[0])
    H, G = gram(5, 160), rand(5, 160, 7)
    H[1] -= 2*H[1].diagonal().max()*torch.eye(160, dtype=f)                              # not positive definite: it has a solution
    H[3] = 0.                                                                            # none
    seen.clear()
    X, ok = MD.spd_solve(H, G), [0, 1, 2, 4]
    assert seen == [(160, 2), (160, 2)]                                                  # the two Cholesky fails on, each on its own
    assert torch.allclose(H[ok]@X[ok], G[ok], atol=1e-9) and float(X[3].abs().max()) == 0.
    n = 3
    prog = C.build(n, 7, dims='48x20')
    st = C.reset(prog); prog.set_teacher('late')
    for k in range(40):
        st, _ = C.step(prog, st, C.teacher(prog, st, k), k)
    seen.clear()
    m, E, K, V = L.move(prog, st, 40)
    assert tuple(K.shape) == (n, 20, MD.NLAB, 20) and float(K.abs().amax()) > 0 and not batched()
    assert torch.allclose(m.clamp(-1., 1.)*prog.c['mvm'], MD_move_without_limits(prog, st, 40), atol=.05)
    b, ncv, NF = 2, 48, N.NF                                                             # the network's fit: 20 MVs' moves and NDV measured disturbances, NF sums each
    nx, nxd = 20*NF, N.NXD
    J = gram(b, nx+nxd)
    sums = (J[:, :nx, :nx], rand(b, ncv, nx), rand(b, ncv, NF, nx), gram(b, ncv, NF)+torch.eye(NF, dtype=f), rand(b, ncv, NF),
            J[:, :nx, nx:], J[:, nx:, nx:], rand(b, ncv, nxd), rand(b, ncv, NF, nxd))
    seen.clear()
    out = N.Stats.solve_own(*sums)
    assert tuple(out[0].shape) == (b, ncv, 20, NF) and tuple(out[2].shape) == (b, ncv, N.NDV, NF) and not batched()
    monkeypatch.setattr(MD, 'LU_MAX', 10**9)                                             # by LU, a system at a time (which returns)
    for i in range(b):
        ref = N.Stats.solve_own(*(x[i:i+1] for x in sums))
        assert all(torch.allclose(x[i:i+1], y, atol=1e-7, rtol=1e-6) for x, y in zip(out, ref))


def test_a_run_waiting_for_its_plants_gives_signs_of_life_but_not_for_ever(tmp_path):
    import concurrent.futures as cf
    was = dict(T.SIGN)
    try:
        T.SIGN.update(stall=.1, path=tmp_path/'alive', file=open(tmp_path/'stalls.txt', 'w'))
        with cf.ThreadPoolExecutor(1) as pool:
            assert T.waited(pool.submit(lambda: (time.sleep(.3), 5)[1]), poll=.05) == 5
        assert 'build' in json.loads((tmp_path/'alive').read_text())          # while a batch is in the making the run is alive
        (tmp_path/'alive').unlink()
        with cf.ThreadPoolExecutor(1) as pool:
            assert T.waited(pool.submit(lambda: (time.sleep(.8), 6)[1]), poll=.05) == 6
        assert json.loads((tmp_path/'alive').read_text())['build'] <= .4      # past four times --stall it says nothing more: the watch can end it
    finally:
        faulthandler.cancel_dump_traceback_later()
        T.SIGN['file'].close()
        T.SIGN.update(was)


def test_lead_hands_the_plant_to_the_teacher_first():
    """--lead: up to the decision drawn for it an episode's moves are the teacher's (the plant is where the teacher alone
    takes it, the network is given them as its last moves) and nothing of the window's cost reaches the network from them."""
    a = args(window=4, noise=0., id_coef=0., lead=1., lead_to=3)
    net, critic, lane, gen = _setup(a)
    n = lane.prog.n
    assert lane.lead is not None and bool((lane.lead > lane.prog.k_first).all())
    lane.lead[:] = 0
    lane.lead[: n//2] = 10**6                                                            # the teacher's all the window, the network's own
    for q in critic.parameters():
        q.requires_grad_(False)
    st, noise = {q: (x.clone() if torch.is_tensor(x) else x) for q, x in lane.st.items()}, lane.prog.g.get_state()
    seen, orig = [], T.Guard.__call__
    T.Guard.__call__ = lambda self, g: (seen.append(g.detach().clone()), orig(self, g))[1]
    try:
        loss, rec, _, _ = T.window(net, critic, lane, a, dict(bc=0., sim=1., value=0.), gen)
        loss.backward()
    finally:
        T.Guard.__call__ = orig
    lane.prog.g.set_state(noise)
    for k in range(4):                                                                   # the teacher alone on the same plants and noise
        tm = C.teacher(lane.prog, st, k)
        assert torch.allclose(rec['act'][: n//2, k].abs(), (tm.float()[: n//2]*(rec['act'][: n//2, k] != 0)).abs(), atol=1e-6)
        st, _ = C.step(lane.prog, st, tm.to(lane.prog.dtype), k)
    assert all(not g[: n//2].any() for g in seen) and any(bool(g[n//2:].any()) for g in seen)


def test_id_sign_teaches_whether_and_which_way():
    """--id-sign: the read-out's first two values are logits of which way a row answers an MV and whether it does; a pair that
    does not answer is taught nothing about the way. A window runs with it."""
    a = args(id_sign=True)
    true = torch.zeros(2, 3, 2, 5)
    true[0, 0, 0], true[0, 1, 1] = 3., -3.                                               # one pair each way, the others silent
    read = torch.zeros(2, 3, 2, 5, requires_grad=True)
    err = T.read_error(None, read, true, a)
    assert err.shape == (2, 3, 2) and torch.allclose(err[0, 0, 0], torch.tensor(2*math.log(2.))) and torch.allclose(err[1], torch.full((3, 2), math.log(2.)))
    g, = torch.autograd.grad(err.sum(), read)
    assert float(g[0, 0, 0, 0]) < 0 < float(g[0, 1, 1, 0]) and float(g[0, 0, 0, 1]) < 0 < float(g[1, 0, 0, 1])
    assert not g[1, ..., 0].any() and not g[..., 2:].any()                               # silent pairs: nothing about the way; the other values: nothing
    net, critic, lane, gen = _setup(a)
    loss, _, _, log = T.window(net, critic, lane, a, dict(bc=0., sim=1., value=0.), gen)
    assert math.isfinite(float(loss)) and math.isfinite(log['id'])


def test_after_and_critic_value_do_not_double_count_and_keep_td_bootstrap():
    """--after > 0 replaces the critic's value at k1 in the actor's objective (no double-counting of [k1, k1+after)),
    while rec['z'][:, -1] still holds the critic's real k1 value for TD targets. With --after 0, z at k1 enters tot."""
    n = 6
    a_after = args(window=6, noise=0., id_coef=0., after=8, after_by='hold')
    a_plain = args(window=6, noise=0., id_coef=0., after=0)
    net, critic, lane, gen = _setup(a_after, n=n)
    prog = lane.prog
    start = dict(st=lane.st, ha=lane.ha, hc=lane.hc, prev=lane.prev, pm=lane.pm, noise=prog.g.get_state())

    def run(a, bias_shift):
        lane.st, lane.ha, lane.hc, lane.prev, lane.pm, lane.k, lane.hs = start['st'], start['ha'], start['hc'], start['prev'], start['pm'], 0, None
        prog.g.set_state(start['noise'])
        with torch.no_grad():
            for h in critic.heads:
                list(h.parameters())[-1].add_(bias_shift)
        for q in critic.parameters():
            q.requires_grad_(False)
        net.zero_grad(set_to_none=True)
        loss, rec, _, _ = T.window(net, critic, lane, a, dict(bc=0., sim=1., value=1.), gen)
        loss.backward()
        grad = torch.cat([q.grad.flatten() for q in net.parameters() if q.grad is not None])
        with torch.no_grad():
            for h in critic.heads:
                list(h.parameters())[-1].sub_(bias_shift)
        for q in critic.parameters():
            q.requires_grad_(True)
        return float(loss), rec['z'][:, -1].clone(), grad

    l0, z0, g0 = run(a_after, 0.)
    l1, z1, g1 = run(a_after, 5.)
    assert l0 == pytest.approx(l1, rel=0, abs=1e-7) and torch.equal(g0, g1)              # critic at k1 does not double-count into tot
    assert torch.allclose(z1-z0, torch.full_like(z0, 5.), atol=1e-5)                      # rec['z'][:, -1] still keeps the real k1 value for TD
    lp0, _, gp0 = run(a_plain, 0.)
    lp1, _, gp1 = run(a_plain, 5.)
    assert abs(lp1-lp0) > 1e-4 and float((gp1-gp0).abs().max()) > 1e-6                    # without --after, critic z at k1 does drive the actor


def test_probe_unpaired_and_pre_cost_under_pure_real_cost():
    """Under pure real-cost (taught=False): an untrained `when` head injects zero probe noise even if --probe > 0;
    when `when` is trained, probe noise hits only blind episodes (--unpaired > 0); --pre-cost scales cost (including
    --smooth) only on blind episodes before k_first and passes post-event gradient back into pre-event moves."""
    from aio import seqnet as SQ
    n = 6
    torch.manual_seed(0)
    net = SQ.SeqAIO('d64,l2,h4,m8,g1,q1', w=16)
    for q, p in net.named_parameters():
        if p.dim() > 1 and float(p.detach().abs().sum()) == 0 and not q.endswith('when.weight'):
            torch.nn.init.normal_(p, std=.05)
    critic = N.Critic(2, stats=True)
    a0 = args(window=16, noise=0., id_coef=0., unpaired=.5, probe=0., pre_cost=1., smooth=5., seq='d64,l2,h4,m8,g1,q1', seq_window=16)
    prog = C.build(n, 5)
    prog.k_first[:] = 8                                                                   # first event mid-window: 0..7 pre, 8..15 post
    lane = T.Lane(prog, a0, torch.Generator().manual_seed(0))
    lane.unpaired = torch.tensor([True, False, True, False, True, False])
    start = dict(st=lane.st, ha=lane.ha, hc=lane.hc, prev=lane.prev, pm=lane.pm, noise=prog.g.get_state())

    def run(a):
        net.forget()
        lane.st, lane.ha, lane.hc, lane.prev, lane.pm, lane.k, lane.hs = start['st'], start['ha'], start['hc'], start['prev'], start['pm'], 0, None
        prog.g.set_state(start['noise'])
        for q in critic.parameters():
            q.requires_grad_(False)
        net.zero_grad(set_to_none=True)
        loss, rec, _, _ = T.window(net, critic, lane, a, dict(bc=0., sim=1., value=0.), torch.Generator().manual_seed(7))
        return loss, rec

    _, r_base = run(a0)
    a_probe = SimpleNamespace(**dict(vars(a0), probe=.5))
    _, r_untrained = run(a_probe)                                                         # `when` is zero -> no ghost probe noise
    assert torch.equal(r_base['act'][:, :16], r_untrained['act'][:, :16])
    with torch.no_grad():
        net.actor.when.bias.fill_(3.)                                                     # trained `when`: only blind episodes probed before k_first
    _, r_trained = run(a_probe)
    assert torch.equal(r_base['act'][~lane.unpaired, :16], r_trained['act'][~lane.unpaired, :16])
    assert float((r_base['act'][lane.unpaired, :8]-r_trained['act'][lane.unpaired, :8]).abs().max()) > .05
    prog.k_first[:] = 0                                                                   # once k >= k_first: no probe noise even on blind episodes
    assert torch.equal(run(a0)[1]['act'][:, :16], run(a_probe)[1]['act'][:, :16])
    prog.k_first[:] = 8

    a_pre = SimpleNamespace(**dict(vars(a0), pre_cost=.25))                               # --pre-cost .25: blind pre-event cost scaled by .25
    loss_pre, r_pre = run(a_pre)
    assert torch.allclose(r_pre['cost'][lane.unpaired, :8], .25*r_base['cost'][lane.unpaired, :8], atol=1e-6)
    assert torch.equal(r_pre['cost'][~lane.unpaired, :16], r_base['cost'][~lane.unpaired, :16])
    assert torch.equal(r_pre['cost'][lane.unpaired, 8:16], r_base['cost'][lane.unpaired, 8:16])
    loss_pre.backward()
    assert float(net.actor.act[-1].weight.grad.abs().sum()) > 0


def test_accum_counts_true_updates_ema_warm_n_and_shared_acc():
    """--accum: update() reports updates=1 only when opt.step() actually fires (0 on accumulating/skipped windows),
    EMA and warm_n advance once per optimizer step, and sharing `acc` across size temp dicts avoids stale per-size buffers."""
    a = args(accum=2, wd=.05, beta2=.95, lr=1e-3, warm_steps=10)
    net, critic, lane, gen = _setup(a)
    opt = T.optimizer(net, a)
    copt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    ema = T.Average(net, .9)
    opt.register_step_post_hook(ema)
    for g in opt.param_groups:
        g['lr'] = g['set_lr'] = a.lr
    acc = dict(g=None, n=0, k=0)
    t_small, t_large = dict(usual=None, acc=acc), dict(usual=None, acc=acc)               # two size temps sharing one acc
    w = dict(bc=1., sim=0., value=0.)
    l1 = T.update(net, critic, opt, copt, lane, a, w, gen, t_small)
    assert l1['updates'] == 0 and ema.n == 0 and opt.warm_n == 0 and acc['k'] == 1
    l2 = T.update(net, critic, opt, copt, lane, a, w, gen, t_large)
    assert l2['updates'] == 1 and ema.n == 1 and opt.warm_n == 1 and acc == dict(g=None, n=0, k=0)
    # set_lr stays at the scheduled LR while lr is warmed (prevents false --resume lr-jump crash mid-warmup)
    pg = opt.state_dict()['param_groups'][0]
    assert pg['set_lr'] == pytest.approx(a.lr) and pg['lr'] == pytest.approx(a.lr/a.warm_steps)


def test_prune_saves_keeps_only_newest_row_snapshots(tmp_path):
    """--keep-saves N: keeps only the N newest row<int>.pt files, never deletes last.pt/best.pt, and 0 keeps all."""
    for name in ('last.pt', 'best.pt', 'row5.pt', 'row10.pt', 'row20.pt', 'row100.pt', 'row_note.pt'):
        (tmp_path/name).write_bytes(b'x')
    assert T.prune_saves(tmp_path, 0) == []
    assert len(list(tmp_path.glob('*.pt'))) == 7
    dropped = T.prune_saves(tmp_path, 2)
    assert dropped == ['row5.pt', 'row10.pt']
    assert sorted(p.name for p in tmp_path.glob('*.pt')) == ['best.pt', 'last.pt', 'row100.pt', 'row20.pt', 'row_note.pt']


def test_optimizer_excludes_2d_lag_b_from_weight_decay_and_logs_x_norm():
    """optimizer(): per-head lag_b (dim 2) goes into the zero-decay group; parts() reports x_norm after a forward pass."""
    from aio import interface as IF, seqnet as SQ
    net = SQ.SeqAIO('d64,l2,h4,m8,g1,q1,u1', w=16)
    opt = T.optimizer(net, SimpleNamespace(wd=.05, lr=1e-4, beta2=.95, warm_steps=0))
    dec_ids = {id(p) for p in opt.param_groups[0]['params']}
    rest_ids = {id(p) for p in opt.param_groups[1]['params']}
    assert opt.param_groups[0]['weight_decay'] == .05 and opt.param_groups[1]['weight_decay'] == 0.
    for lay in net.actor.layers:
        assert id(lay.lag_b) in rest_ids and id(lay.lag_b) not in dec_ids
        assert id(lay.out_l.weight) in dec_ids
    obs = torch.randn(2, 4, IF.SIZE)
    prev = torch.zeros(2, 4, IF.NMV)
    net.steps(obs, prev)[0].square().mean().backward()
    g = T.parts(net)
    assert 'x_norm' in g and g['x_norm'] > 0 and 'x_norm' in T.GAUGES


def test_muon_adamw_optimizer_routing_groups_ema_and_state_roundtrip():
    """--opt muon puts 2D hidden matrices into Muon (Newton-Schulz), routing biases into bias_lr_mult AdamW,
    embeddings/readout/norms into AdamW, supports EMA + warm_steps, and round-trips state_dict cleanly."""
    from aio import interface as IF, seqnet as SQ
    torch.manual_seed(0)
    net = SQ.SeqAIO('d64,l2,h4,m8,g1,q1,u1', w=16)
    a = SimpleNamespace(opt='muon', wd=.05, lr=2e-4, beta2=.95, warm_steps=4, bias_lr_mult=10., muon_mom=.95)
    opt = T.optimizer(net, a)
    assert isinstance(opt, T.MuonAdamW)
    by_kind = {}
    for g in opt.param_groups:
        k = (g['kind'], g.get('lr_mult', 1.), g['weight_decay'])
        by_kind[k] = {id(p) for p in g['params']}
    mu_ids = by_kind[('muon', 1., .05)]
    rb_ids = by_kind[('adamw', 10., 0.)]
    ad_dec = by_kind[('adamw', 1., .05)]
    assert id(net.actor.layers[0].qkv_t.weight) in mu_ids
    assert id(net.actor.layers[0].ff[0].weight) in mu_ids
    assert id(net.actor.act[0].weight) in mu_ids
    assert id(net.actor.act[2].weight) in ad_dec and id(net.actor.act[2].weight) not in mu_ids
    assert id(net.actor.row.weight) in ad_dec
    for lay in net.actor.layers:
        for p in (lay.pair, lay.lag_pair, lay.lag_own, lay.lag_b):
            assert id(p) in rb_ids
    # Check Newton-Schulz compresses singular values near 1 (spectral normalization)
    G = torch.randn(32, 16)
    O = T.zeropower_newtonschulz5(G)
    sg, so = torch.linalg.svdvals(G), torch.linalg.svdvals(O)
    assert float(so.min()) > .5 and float(so.max()) < 1.3
    assert float(so.max()/so.min()) < float(sg.max()/sg.min())
    # Run two steps with EMA & warm_steps (non-zero init on zero-init residual/readout layers so grads flow)
    with torch.no_grad():
        net.actor.act[-1].weight.normal_(0, .2)
        net.actor.layers[0].out_t.weight.normal_(0, .2)
    ema = T.Average(net, .9)
    opt.register_step_post_hook(ema)
    obs = torch.randn(2, 4, IF.SIZE)
    prev = torch.zeros(2, 4, IF.NMV)
    w0 = net.actor.layers[0].qkv_t.weight.detach().clone()
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        (net.steps(obs, prev)[0]-.5).square().mean().backward()
        opt.step()
    assert ema.n == 2 and float((net.actor.layers[0].qkv_t.weight-w0).abs().max()) > 1e-5
    sd = opt.state_dict()
    opt2 = T.optimizer(net, a)
    opt2.load_state_dict(sd)
    assert len(opt2.state) == len(opt.state)


def test_priv_critic_seq_ckpt_and_muon_update_end_to_end(tmp_path):
    """End-to-end training update with SeqAIO + --seq-ckpt + --priv-critic + --opt muon + cached_refs."""
    from aio.sim import core as C
    torch.manual_seed(1)
    a = SimpleNamespace(
        seq='d64,l2,h4,m8,g1,q1,u1', seq_window=16, seq_ckpt=True, priv_critic=True,
        opt='muon', bias_lr_mult=5., muon_mom=.95, replay=1, no_stats=True,
        law=False, formula=False, solve=False, given=False, reader=False, slow=0., arx=False, dv=False, tied=0, knots=N.NLAB,
        probe=0., unpaired=.25, pre_cost=.25, meter=False, heads=2, fixed=0, bc_iters=0, bc_floor=0.,
        window=8, window_min=4, guard='cut', cap=10., spike=10., relax=4, lean=False,
        gamma=.99, lam=.95, tau=.25, cost_w=.1, log_w=0., smooth=3., floor=.1, fail_cost=10.,
        stops=True, hard=False, noise=.05, known=.25, critic_known=.75, id_coef=0., id_sign=False,
        bc_coef=10., bc_own=False, drive_noise=.2, faults=0., wd=.05, lr=2e-4, beta2=.95,
        warm_steps=0, clip=1., accum=1, fit_chunk=4, critic_iters=2, after=0, ppo=False, value=1.,
    )
    net, critic = T.nets(a, torch.device('cpu'))
    assert net.actor.grad_ckpt is True and getattr(critic, 'privileged', False) is True
    opt, copt = T.optimizer(net, a), torch.optim.Adam(critic.parameters(), lr=5e-4)
    prog = C.build(4, 99, 'cpu', 'cpu')
    gen = torch.Generator().manual_seed(3)
    lane = T.Lane(prog, a, gen, teach=0.)
    assert lane.prog.Fr is None                                         # teacher prediction skipped under pure real cost
    T.advance(net, critic, lane, 4, gen, a.noise)
    temp = dict(usual=None, acc=dict(g=None, n=0, k=0))
    log1 = T.update(net, critic, opt, copt, lane, a, dict(bc=0., sim=1., value=1.), gen, temp)
    log2 = T.update(net, critic, opt, copt, lane, a, dict(bc=0., sim=1., value=1.), gen, temp)
    assert log1['bad'] == 0. and log2['bad'] == 0. and log2['updates'] == 1
    assert log2['critic'] is not None and math.isfinite(log2['critic'])
    # cached_refs writes once and reloads cleanly
    prog_v = C.build(2, 100, 'cpu', 'cpu')
    r1 = T.cached_refs(tmp_path, 'test-val', prog_v, False)
    r2 = T.cached_refs(tmp_path, 'test-val', prog_v, False)
    assert (tmp_path/'test-val.pt').exists()
    assert torch.allclose(r1['pi']['score'], r2['pi']['score'])
    # PrivCritic + fixed-horizon TD (--fixed 8 --heads 3): Due / fit_due / Due.last all work end-to-end without self-bootstrap
    af = SimpleNamespace(**{**vars(a), 'fixed': 8, 'heads': 3})
    net_f, crit_f = T.nets(af, torch.device('cpu'))
    opt_f, copt_f = T.optimizer(net_f, af), torch.optim.Adam(crit_f.parameters(), lr=5e-4)
    lane_f = T.Lane(C.build(4, 101, 'cpu', 'cpu'), af, torch.Generator().manual_seed(4), teach=0.)
    tf = dict(usual=None, acc=dict(g=None, n=0, k=0))
    l1 = T.update(net_f, crit_f, opt_f, copt_f, lane_f, af, dict(bc=0., sim=1., value=1.), torch.Generator().manual_seed(5), tf)
    l2 = T.update(net_f, crit_f, opt_f, copt_f, lane_f, af, dict(bc=0., sim=1., value=1.), torch.Generator().manual_seed(6), tf)
    assert 'critic' not in l1 and 'critic' in l2 and math.isfinite(l2['critic'])
    last_out = T.fit_due(crit_f, copt_f, lane_f, af, last=True)
    assert 'critic' in last_out and math.isfinite(last_out['critic'])
    # SeqAIO v2 (g2 + f1 + e8) end-to-end with MuonAdamW (--bias-lr-mult puts edge_bias into rbias) and PrivCritic
    av2 = SimpleNamespace(**{**vars(a), 'seq': 'd64,l2,h4,m8,g2,q1,u1,f1,e8', 'fixed': 8, 'heads': 4, 'rev_smooth': 1.0})
    net_v2, crit_v2 = T.nets(av2, torch.device('cpu'))
    opt_v2, copt_v2 = T.optimizer(net_v2, av2), torch.optim.Adam(crit_v2.parameters(), lr=5e-4)
    rbias_ids = {id(p) for g in opt_v2.param_groups if g.get('lr_mult', 1.) == 5. for p in g['params']}
    assert id(net_v2.actor.layers[0].edge_bias.weight) in rbias_ids and id(net_v2.actor.layers[0].lag_b) in rbias_ids
    lane_v2 = T.Lane(C.build(4, 102, 'cpu', 'cpu'), av2, torch.Generator().manual_seed(7), teach=0.)
    tv2 = dict(usual=None, acc=dict(g=None, n=0, k=0))
    lv1 = T.update(net_v2, crit_v2, opt_v2, copt_v2, lane_v2, av2, dict(bc=0., sim=1., value=1.), torch.Generator().manual_seed(8), tv2)
    lv2 = T.update(net_v2, crit_v2, opt_v2, copt_v2, lane_v2, av2, dict(bc=0., sim=1., value=1.), torch.Generator().manual_seed(9), tv2)
    assert lv1['bad'] == 0. and lv2['bad'] == 0. and lv2['updates'] == 1
    reloaded = N.of_state(net_v2.actor.state_dict(), net_v2.pairs.state_dict())
    assert reloaded.edge == 8 and reloaded.fb is True and reloaded.actor.layers[0].lags == 2
    # Log-linear g1 -> g2 lag_b expansion preserves trained slots and interpolates new slots without cold-start starvation
    from aio import seqnet as SQ
    v1 = torch.tensor([[0.1, -0.2, 0.3, -0.1, 0.5]])
    v2 = T.expand_lag_b(v1, SQ.LAGS_BY[1], SQ.LAGS_BY[2])
    assert v2.shape == (1, 8)
    for i1, tau in enumerate(SQ.LAGS_BY[1]):
        assert torch.allclose(v2[:, SQ.LAGS_BY[2].index(tau)], v1[:, i1])
    assert float(v2[0, 3]) > -0.5 and float(v2[0, 5]) > -0.5 and float(v2[0, 6]) > -0.5
    # SeqAIO v3 (g2 + f1 + r1) end-to-end with MuonAdamW, PrivCritic, and training-only --rel-id-coef probe
    av3 = SimpleNamespace(**{**vars(a), 'seq': 'd64,l2,h4,m8,g2,q1,u1,f1,r1', 'fixed': 8, 'heads': 4, 'rev_smooth': 1.0, 'rel_id_coef': 0.1, 'unpaired': 0.5, 'pre_cost': 0.0})
    net_v3, crit_v3 = T.nets(av3, torch.device('cpu'))
    opt_v3, copt_v3 = T.optimizer(net_v3, av3), torch.optim.Adam(crit_v3.parameters(), lr=5e-4)
    rb3 = {id(p) for g in opt_v3.param_groups if g.get('lr_mult', 1.) == 5. for p in g['params']}
    assert id(net_v3.actor.layers[0].rel_bias.weight) in rb3 and id(net_v3.actor.layers[0].rel_lag_bias.weight) in rb3 and id(net_v3.actor.rel_p) in rb3
    rel_probe = torch.nn.Linear(SQ.REL_F, 2)
    rel_popt = torch.optim.AdamW(rel_probe.parameters(), lr=3e-3)
    lane_v3 = T.Lane(C.build(4, 103, 'cpu', 'cpu'), av3, torch.Generator().manual_seed(10), teach=0.)
    lane_v3.k = 16
    tv3 = dict(usual=None, acc=dict(g=None, n=0, k=0), rel_probe=rel_probe, rel_popt=rel_popt)
    lr1 = T.update(net_v3, crit_v3, opt_v3, copt_v3, lane_v3, av3, dict(bc=0., sim=1., value=1.), torch.Generator().manual_seed(11), tv3)
    lr2 = T.update(net_v3, crit_v3, opt_v3, copt_v3, lane_v3, av3, dict(bc=0., sim=1., value=1.), torch.Generator().manual_seed(12), tv3)
    assert lr1['bad'] == 0. and lr2['bad'] == 0. and lr2['updates'] == 1 and 'rel_id' in lr2 and math.isfinite(lr2['rel_id'])
    reloaded3 = N.of_state(net_v3.actor.state_dict(), net_v3.pairs.state_dict())
    assert reloaded3.rel == 1 and reloaded3.edge == 0 and reloaded3.fb is True and reloaded3.actor.layers[0].lags == 2
    # SeqAIO v4 (g2 + f1 + r2) end-to-end with MuonAdamW, PrivCritic, 3-output physical-gain rel_probe, and sensitivity-scaled noise
    av4 = SimpleNamespace(**{**vars(a), 'seq': 'd64,l2,h4,m8,g2,q1,u1,f1,r2', 'fixed': 8, 'heads': 4, 'rev_smooth': 1.0, 'rel_id_coef': 0.1, 'unpaired': 0.5, 'pre_cost': 0.0, 'noise': 0.02})
    net_v4, crit_v4 = T.nets(av4, torch.device('cpu'))
    opt_v4, copt_v4 = T.optimizer(net_v4, av4), torch.optim.Adam(crit_v4.parameters(), lr=5e-4)
    rb4 = {id(p) for g in opt_v4.param_groups if g.get('lr_mult', 1.) == 5. for p in g['params']}
    assert id(net_v4.actor.act_gate.weight) in rb4 and id(net_v4.actor.act_gate.bias) in rb4
    rel_probe4 = torch.nn.Linear(SQ.REL_F2, 3)
    rel_popt4 = torch.optim.AdamW(rel_probe4.parameters(), lr=3e-3)
    lane_v4 = T.Lane(C.build(4, 104, 'cpu', 'cpu'), av4, torch.Generator().manual_seed(13), teach=0.)
    lane_v4.k = 16
    tv4 = dict(usual=None, acc=dict(g=None, n=0, k=0), rel_probe=rel_probe4, rel_popt=rel_popt4)
    l4a = T.update(net_v4, crit_v4, opt_v4, copt_v4, lane_v4, av4, dict(bc=0., sim=1., value=1.), torch.Generator().manual_seed(14), tv4)
    l4b = T.update(net_v4, crit_v4, opt_v4, copt_v4, lane_v4, av4, dict(bc=0., sim=1., value=1.), torch.Generator().manual_seed(15), tv4)
    assert l4a['bad'] == 0. and l4b['bad'] == 0. and l4b['updates'] == 1 and 'rel_id' in l4b and math.isfinite(l4b['rel_id'])
    reloaded4 = N.of_state(net_v4.actor.state_dict(), net_v4.pairs.state_dict())
    assert reloaded4.rel == 2 and reloaded4.edge == 0 and reloaded4.fb is True and reloaded4.actor.layers[0].lags == 2
    # Gradient accumulation (--accum > 1) must step both actor and training-only rel_probe once per `accum` windows
    av4_acc = SimpleNamespace(**{**vars(av4), 'accum': 2})
    w_probe_before = rel_probe4.weight.detach().clone()
    tv4_acc = dict(usual=tv4['usual'], rel_probe=rel_probe4, rel_popt=rel_popt4)
    la1 = T.update(net_v4, crit_v4, opt_v4, copt_v4, lane_v4, av4_acc, dict(bc=0., sim=1., value=1.), torch.Generator().manual_seed(16), tv4_acc)
    assert la1['updates'] == 0 and torch.equal(rel_probe4.weight, w_probe_before)
    la2 = T.update(net_v4, crit_v4, opt_v4, copt_v4, lane_v4, av4_acc, dict(bc=0., sim=1., value=1.), torch.Generator().manual_seed(17), tv4_acc)
    assert la2['updates'] == 1 and not torch.equal(rel_probe4.weight, w_probe_before)

"""aio.net: with the pair pathway at its start the network computes what the earlier actor did (so every earlier
checkpoint loads and behaves the same), a sequence and single decisions give the same, the true responses take the
estimates' place only where asked, the running sums put a row's changes down to the MV that made them, and the
critic and the controller wrappers hold their contracts."""
import math

import pytest
import torch

from aio import net as N
from aio.sim import core as C, control as CT
from aio.interface import NCV, NMV, CF, MF, SIZE
from aio.actor import Actor, STATE as BASE
from aio.frame import canonical
from aio.sim.model import NLAB


@pytest.fixture(scope='module')
def packets():
    """Packets of a short closed-loop run (8 episodes, 6 decisions), the last actions, and the true responses."""
    torch.manual_seed(3)
    prog = C.build(8, 11)
    st = C.reset(prog)
    obs, prev, lab = [], [], []
    a = torch.zeros(8, NMV)
    for k in range(6):
        o, s = canonical(C.observe(prog, st, k).float())
        obs.append(o); prev.append(a); lab.append(N.pair_labels(prog.labels(k).float()*s[:, :, None]))
        a = (.3*torch.randn(8, NMV)).clamp(-1, 1)*prog.c['mvm'].float()
        st, _ = C.step(prog, st, (a*s).to(prog.dtype), k)
    return torch.stack(obs, 1), torch.stack(prev, 1), torch.stack(lab, 1)


def lively(net):
    torch.nn.init.normal_(net.pairs.out.weight, std=.05)                                 # the pair pathway in play
    if net.pairs.stats is not None:
        torch.nn.init.normal_(net.pairs.stat[-1].weight, std=.3)                         # and what it reads of the running sums
    return net


def test_at_its_start_the_network_is_the_earlier_actor(packets):
    obs, prev, _ = packets
    torch.manual_seed(0)
    net = N.AIO()
    joint, mem, mask, h = net.actor.history(obs, prev, torch.zeros(1, 8, BASE))
    mean0, logstd0, _ = net.actor.distribution(joint, mem, mask)
    mean, logstd, mask1, lab, state = net.steps(obs, prev)
    assert torch.equal(mean, mean0) and torch.equal(logstd, logstd0) and torch.equal(mask1, mask)
    assert state.shape == (1, 8, N.STATE) and torch.equal(state[..., :BASE], h)
    assert lab.shape == (8, 6, NCV, NMV, NLAB)


def test_an_earlier_state_is_taken_as_it_is(packets):
    obs, prev, _ = packets
    net = lively(N.AIO())
    h = .1*torch.randn(1, 8, BASE)
    a = net.steps(obs, prev, h)
    b = net.steps(obs, prev, torch.cat((h, torch.zeros(1, 8, N.STATE-BASE)), -1))
    assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(a, b))                    # (sums over pairs: not bit for bit between two calls)


def test_a_sequence_is_its_decisions_one_by_one(packets):
    obs, prev, _ = packets
    net = lively(N.AIO())
    mean, _, _, lab, state = net.steps(obs, prev)
    h, ms, ls = None, [], []
    for t in range(obs.shape[1]):
        m, _, _, l, h = net.steps(obs[:, t:t+1], prev[:, t:t+1], h)
        ms.append(m); ls.append(l)
    assert torch.allclose(torch.cat(ms, 1), mean, atol=1e-5) and torch.allclose(torch.cat(ls, 1), lab, atol=1e-5)
    assert torch.allclose(h, state, atol=1e-5)


def test_the_running_sums_can_be_kept_by_the_caller(packets):
    """The trainer keeps them once for actor and critic: the same moves and read-out as when the network does."""
    obs, prev, _ = packets
    net = lively(N.AIO())
    F, hs = net.pairs.stats(obs)
    assert F.shape == (8, 6, NCV, NMV, N.NSTAT) and hs.shape == (8, N.SSTAT) and bool(torch.isfinite(F).all())
    a, b = net.steps(obs, prev), net.steps(obs, prev, feats=F)
    assert torch.allclose(a[0], b[0], atol=1e-6) and torch.allclose(a[3], b[3], atol=1e-6)
    assert torch.equal(a[4][..., N.BASE+N.SPAIR:], hs[None]) and not b[4][..., N.BASE+N.SPAIR:].any()
    F2, hs2 = net.pairs.stats(obs[:, 3:], net.pairs.stats(obs[:, :3])[1])                # carried on from a state
    assert torch.allclose(F2, F[:, 3:], atol=1e-6) and torch.allclose(hs2, hs, atol=1e-6)
    F0 = torch.zeros_like(F)
    assert float((net.steps(obs, prev, feats=F0)[3]-a[3]).detach().abs().max()) > 1e-4       # and they are read


def test_weights_from_before_the_running_sums_compute_what_they_did(packets):
    obs, prev, _ = packets
    old = lively(N.AIO(stats=False))
    new = N.AIO()
    new.actor.load_state_dict(old.actor.state_dict()); new.pairs.load_state_dict(old.pairs.state_dict())
    assert not N.has_stats(old.pairs.state_dict()) and N.has_stats(new.pairs.state_dict())
    a, b = old.steps(obs, prev), new.steps(obs, prev)
    assert torch.allclose(a[0], b[0], atol=1e-6) and torch.allclose(a[3], b[3], atol=1e-6)
    with pytest.raises(RuntimeError):
        N.AIO(stats=False).pairs.load_state_dict(new.pairs.state_dict())                 # the other way round is refused


def packet_of(dy, du):
    """Packets (b,T,SIZE) of row 0's changes dy (b,T) and the moves du (b,T,m) of the first m MVs, nothing else in service."""
    b, T, m = du.shape
    cv, mv = torch.zeros(b, T, NCV, CF), torch.zeros(b, T, NMV, MF)
    cv[:, :, 0, 1], cv[:, :, 0, -1] = dy, 1.
    mv[:, :, :m, 1], mv[:, :, :m, -1] = du, 1.
    obs = torch.cat((cv.flatten(2), mv.flatten(2), torch.zeros(b, T, NCV*NMV)), -1)
    assert obs.shape[-1] == SIZE
    return obs


def test_the_running_sums_put_a_change_down_to_the_mv_that_made_it():
    """Row 0 answers MV 0 alone (gain 2, a lag of 5 decisions); MV 1 moves against MV 0 nearly move for move
    and does nothing. Solved per MV, MV 1 looks as if it had the response with the other sign; solved jointly,
    MV 0 has it and MV 1 has none."""
    g = torch.Generator().manual_seed(0)
    T = 400
    u0 = .1*torch.randn(4, T, generator=g)
    u = torch.stack((u0, -u0+.03*torch.randn(4, T, generator=g)), -1)
    y, dy = torch.zeros(4), []
    for t in range(T):
        dy.append(.2*(2*u0[:, :t+1].sum(1)-y))
        y = y+dy[-1]
    stats = N.Stats()
    F, _ = stats(packet_of(torch.stack(dy, 1), u))
    h, step = None, torch.zeros(N.NF)                             # the filters' answers to a step of MV 0, 8 decisions on
    for t in range(8):
        _, h = stats(packet_of(torch.zeros(1, 1), torch.full((1, 1, 1), float(t == 0))), h)
        step = step+h[0, :N.NF]
    at8 = lambda f: (.45*torch.sinh(f[:, -1, 0, :2])*step).sum(-1)                           # noqa: E731  (4, 2 MVs): row 0's response 8 decisions on
    joint, alone, true = at8(F[..., N.NF:2*N.NF]), at8(F[..., 2*N.NF:3*N.NF]), 2*(1-.8**8)
    assert torch.allclose(joint[:, 0], torch.full((4,), true), atol=.1) and bool((joint[:, 1].abs() < .1).all())
    assert bool((alone[:, 1] < -1.).all())
    assert bool((F[:, -1, 0, 0, 0] > .7).all()) and bool((F[:, -1, 0, 1, 0] < -.7).all())  # the correlations alone cannot tell the two apart
    assert not F[:, :, 1:, :, :3*N.NF].any() and not F[:, :, :, 2:, :3*N.NF].any()          # rows and MVs out of service: nothing
    bad = packet_of(torch.full((1, 3), float('nan')), torch.full((1, 3, 2), float('inf')))
    F, h = stats(bad)
    assert bool(torch.isfinite(F).all()) and bool(torch.isfinite(h).all())                  # a packet that is not finite does not stay in the sums


def test_tied_rounds_are_one_set_of_weights_and_the_response_enters_as_a_factor(tmp_path, packets):
    obs, prev, true = packets
    torch.manual_seed(2)
    net = N.AIO(tied=6)
    joint, mem, mask, _ = net.actor.history(obs, prev, torch.zeros(1, 8, BASE))
    assert torch.equal(net.steps(obs, prev)[0], net.actor.distribution(joint, mem, mask)[0])    # at its start: the actor alone, as with the two rounds
    count = lambda q: sum(p.numel() for p in q.parameters())                             # noqa: E731
    assert count(N.PairNet(tied=3)) == count(N.PairNet(tied=8))                          # more rounds, the same weights
    lively(net)
    mean, _, _, lab, state = net.steps(obs, prev)
    h, ms = None, []
    for t in range(obs.shape[1]):
        m, _, _, _, h = net.steps(obs[:, t:t+1], prev[:, t:t+1], h)
        ms.append(m)
    assert torch.allclose(torch.cat(ms, 1), mean, atol=1e-5) and torch.allclose(h, state, atol=1e-5)
    told = net.steps(obs, prev, code=(true, torch.ones(8)))[0]
    assert float((told-mean).detach().abs().max()) > 1e-4                                # told the true responses it moves differently
    # a pair's message is the sender's state times a gate that is linear in the pair's response: with the
    # gates shut nothing passes between rows and MVs, whatever the responses are
    shut = N.AIO(tied=6); shut.load_state_dict(net.state_dict())
    for q in (shut.pairs.gate_row, shut.pairs.gate_mv):
        torch.nn.init.zeros_(q.weight); torch.nn.init.zeros_(q.bias)
    a, b = shut.steps(obs, prev, code=(true, torch.ones(8)))[0], shut.steps(obs, prev, code=(3*true, torch.ones(8)))[0]
    assert torch.allclose(a, b, atol=1e-6)
    torch.save(dict(model=dict(actor=net.actor.state_dict(), pairs=net.pairs.state_dict())), tmp_path/'tied.pt')
    back, _ = N.load(tmp_path/'tied.pt')                                                 # the rounds are read from the weights
    assert back.pairs.tied == 6 and torch.allclose(back.steps(obs, prev)[0], mean, atol=1e-6)
    critic = N.Critic(tied=6)
    assert critic.steps(obs, prev, torch.zeros(8, 6, NMV))[0].shape == (8, 6, 2)


def test_true_responses_take_the_place_of_the_estimates_only_where_asked(packets):
    obs, prev, true = packets
    net = lively(N.AIO())
    mean, _, _, lab, _ = net.steps(obs, prev)
    share = torch.tensor([1., 0.]*4)
    told, _, _, lab_t, _ = net.steps(obs, prev, code=(true, share))
    assert torch.equal(lab_t, lab)                                                       # what is read out stays the network's own estimate
    assert torch.allclose(told[1::2], mean[1::2], atol=1e-6)                             # episodes not told: unchanged
    assert float((told[0::2]-mean[0::2]).abs().max()) > 1e-4                             # episodes told: the moves change
    none, _, _, _, _ = net.steps(obs, prev, code=(true, torch.zeros(8)))
    assert torch.allclose(none, mean, atol=1e-6)


def test_episodes_do_not_see_each_other(packets):
    obs, prev, _ = packets
    net = lively(N.AIO())
    perm = torch.randperm(8)
    a = net.steps(obs, prev)
    b = net.steps(obs[perm], prev[perm])
    assert torch.allclose(a[0][perm], b[0], atol=1e-5) and torch.allclose(a[3][perm], b[3], atol=1e-5)


def test_the_readout_learns_only_through_the_pair_memories(packets):
    """The identification has its own memory: its loss sends nothing into the actor the earlier checkpoints trained."""
    obs, prev, true = packets
    net = N.AIO()
    _, _, _, lab, _ = net.steps(obs, prev)
    (lab-true).square().mean().backward()
    assert all(q.grad is None or not q.grad.any() for q in net.actor.parameters())
    assert sum(float(q.grad.abs().sum()) for q in net.pairs.parameters() if q.grad is not None) > 0


def test_critic(packets):
    obs, prev, true = packets
    critic = N.Critic(heads=3)
    act = (.2*torch.randn(8, 6, NMV)).requires_grad_(True)
    z, state = critic.steps(obs, prev, act, code=(true, torch.ones(8)))
    assert z.shape == (8, 6, 3) and state.shape == (1, 8, N.STATE)
    g, = torch.autograd.grad(z.sum(), act)
    mask = obs.new_tensor(1.)*(N.unpack(obs)[1][..., -1] > 0)
    assert float((g*mask).abs().sum()) > 0 and not (g*(1-mask)).any()                    # it reads the moves of the MVs in service, no others
    z1, _ = critic.steps(obs[:, :3], prev[:, :3], act[:, :3], code=(true[:, :3], torch.ones(8)))
    z2, _ = critic.steps(obs[:, 3:], prev[:, 3:], act[:, 3:], _, (true[:, 3:], torch.ones(8)))
    assert z1.shape == z2.shape


def test_the_controller_wrappers_agree(packets):
    net = lively(N.AIO()).eval()
    prog = C.build(6, 21)
    steps = int(prog.seg_k[:, 1].sort().values[2])+15                                    # past the first event of three of the episodes: before it nothing is charged
    a = CT.rollout(prog, N.Policy(net), steps)
    b = CT.rollout(prog, CT.Policy(net.legacy()), steps)                                 # the earlier actor's interface
    assert torch.allclose(a['score'], b['score']) and torch.allclose(a['cost'], b['cost'], rtol=1e-3)
    k = CT.rollout(prog, N.Policy(net, known=True), steps)
    assert not torch.allclose(k['cost'], a['cost'])                                      # told the true responses it moves differently


def test_checkpoints_load_with_and_without_the_pair_pathway(tmp_path, packets):
    obs, prev, _ = packets
    torch.manual_seed(1)
    old = Actor()
    torch.save(dict(model=dict(actor=old.state_dict(), critic={})), tmp_path/'old.pt')   # a checkpoint from before 11-10
    net, _ = N.load(tmp_path/'old.pt')
    joint, mem, mask, _ = old.history(obs, prev, torch.zeros(1, 8, BASE))
    assert torch.equal(net.steps(obs, prev)[0], old.distribution(joint, mem, mask)[0])
    new = lively(N.AIO())
    torch.save(dict(model=dict(actor=new.actor.state_dict(), pairs=new.pairs.state_dict())), tmp_path/'new.pt')
    net, _ = N.load(tmp_path/'new.pt')
    assert torch.allclose(net.steps(obs, prev)[0], new.steps(obs, prev)[0], atol=1e-6)
    leg = net.legacy()                                                                   # the earlier actor's interface on the same weights
    j, m, k, _ = leg.history(obs, prev, torch.zeros(1, 8, BASE))
    assert leg.canonical and torch.allclose(leg.distribution(j, m, k)[0], new.steps(obs, prev)[0], atol=1e-6)


def test_a_slow_set_of_the_sums_keeps_what_the_first_forgets():
    """Row 0 answers MV 0 (gain 2, a lag of 5 decisions). 400 decisions of moves that show it, then 1,200 of
    moves a fifth the size under a row noise that swamps them. The first set of sums (forgetting in 200) is
    left with the noisy stretch; the slow set (2,000) still has the response."""
    g = torch.Generator().manual_seed(1)
    T1, T2 = 400, 1200
    u = torch.cat((.1*torch.randn(4, T1, generator=g), .02*torch.randn(4, T2, generator=g)), 1)
    y, dy = torch.zeros(4), []
    for t in range(T1+T2):
        dy.append(.2*(2*u[:, :t+1].sum(1)-y))
        y = y+dy[-1]
    dy = torch.stack(dy, 1)
    dy[:, T1:] += .05*torch.randn(4, T2, generator=g)
    stats = N.Stats(slow=2000.)
    assert stats.size == N.SSTAT+N.SSLOW and N.Stats().size == N.SSTAT
    obs = packet_of(dy, u[..., None])
    F, hs = stats(obs)
    assert F.shape[-1] == 2*N.NSTAT and hs.shape == (4, N.SSTAT+N.SSLOW)
    F0, h0 = N.Stats()(obs)
    assert torch.equal(F[..., :N.NSTAT], F0) and torch.equal(hs[:, :N.SSTAT], h0)             # the first set is as it is without the second
    Fa, ha = stats(obs[:, :700])
    Fb, hb = stats(obs[:, 700:], ha)
    assert torch.allclose(torch.cat((Fa, Fb), 1), F, atol=1e-5) and torch.allclose(hb, hs, rtol=1e-4, atol=1e-6)   # the state carries both
    h, step = None, torch.zeros(N.NF)
    for t in range(8):
        _, h = N.Stats()(packet_of(torch.zeros(1, 1), torch.full((1, 1, 1), float(t == 0))), h)
        step = step+h[0, :N.NF]
    at8 = lambda f: (.45*torch.sinh(f[:, -1, 0, 0])*step).sum(-1)                           # noqa: E731  (4,): row 0's response to MV 0, 8 decisions on
    fast, slow, true = at8(F[..., N.NF:2*N.NF]), at8(F[..., N.NSTAT+N.NF:N.NSTAT+2*N.NF]), 2*(1-.8**8)
    assert bool(((slow-true).abs() < .25).all()) and float((fast-true).abs().mean()) > 2*float((slow-true).abs().mean())
    assert bool((F[:, -1, 0, 0, N.NSTAT+3*N.NF:N.NSTAT+4*N.NF] > F[:, -1, 0, 0, 3*N.NF:4*N.NF]).all())      # and how much the MV has moved, the larger moves included


def test_weights_from_before_the_slow_sums_compute_what_they_did(packets):
    obs, prev, _ = packets
    old = lively(N.AIO())
    new = N.AIO(slow=2000.)
    assert 'slow' not in old.pairs.state_dict() and not any(k.startswith('stat2') for k in old.pairs.state_dict())
    new.actor.load_state_dict(old.actor.state_dict()); new.pairs.load_state_dict(old.pairs.state_dict())
    assert N.slow_of(old.pairs.state_dict()) == 0. and N.slow_of(new.pairs.state_dict()) == 2000.
    a, b = old.steps(obs, prev), new.steps(obs, prev)
    assert torch.equal(a[0], b[0]) and torch.equal(a[3], b[3])                           # bit for bit: the new layer adds zero
    assert a[4].shape == (1, 8, N.STATE) and b[4].shape == (1, 8, N.STATE+N.SSLOW) and torch.equal(b[4][..., :N.STATE], a[4])
    with pytest.raises(RuntimeError):
        N.AIO().pairs.load_state_dict(new.pairs.state_dict())                            # the other way round is refused
    torch.nn.init.normal_(new.pairs.stat2[-1].weight, std=.3)
    c = new.steps(obs, prev)
    assert float((c[3]-b[3]).abs().max()) > 1e-4                                          # in play, the slow sums are read
    whole = new.steps(obs, prev)
    first = new.steps(obs[:, :3], prev[:, :3])
    rest = new.steps(obs[:, 3:], prev[:, 3:], first[4])
    assert torch.allclose(torch.cat((first[0], rest[0]), 1), whole[0], atol=1e-5) and torch.allclose(rest[4], whole[4], atol=1e-5)


def test_the_slow_sums_sit_before_the_moves_the_formula_keeps(packets):
    from aio import formula as FM
    obs, prev, _ = packets
    torch.manual_seed(4)
    net = lively(N.AIO(law=True, knots=16, formula=True, solve=True, slow=2000.))
    torch.nn.init.normal_(net.pairs.stat2[-1].weight, std=.3)
    whole = net.steps(obs, prev)
    assert whole[4].shape == (1, 8, N.STATE+N.SSLOW+FM.SMEM) and net.pairs.size == N.SPAIR+N.SSTAT+N.SSLOW+FM.SMEM
    first = net.steps(obs[:, :3], prev[:, :3])
    rest = net.steps(obs[:, 3:], prev[:, 3:], first[4])
    assert torch.allclose(torch.cat((first[0], rest[0]), 1), whole[0], atol=1e-5) and torch.allclose(rest[4], whole[4], atol=1e-5)
    same = N.AIO(law=True, knots=16, formula=True, solve=True)                           # the same weights without the slow sums: the same state before and behind them
    same.actor.load_state_dict(net.actor.state_dict())
    same.pairs.load_state_dict({k: v for k, v in net.pairs.state_dict().items() if k != 'slow' and not k.startswith('stat2')})
    s0 = same.steps(obs, prev)[4]
    assert torch.equal(s0[..., N.STATE:], whole[4][..., N.STATE+N.SSLOW:]) and torch.equal(s0[..., N.STATE-N.SSTAT:N.STATE], whole[4][..., N.STATE-N.SSTAT:N.STATE])


def test_the_trainer_gathers_and_steps_with_the_slow_sums():
    from aio import train as T
    from tests.test_train_guard import args
    n = 8
    a = args(noise=0., replay=1, batch=8, keep=4, fit_chunk=4, law=True, lr=1e-3, known=.5, clip=10., slow=2000.)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    net, critic = N.AIO(law=True, slow=2000.), N.Critic(2)
    opt, copt = torch.optim.Adam(net.parameters(), lr=1e-3), torch.optim.Adam(critic.parameters(), lr=3e-4)
    lane, buf = T.Lane(C.build(n, 5), a, gen, teach=1.), T.Buffer(a.keep*n)
    for _ in range(3):
        log = T.imitate(net, critic, opt, copt, lane, buf, a, gen)
    assert buf.x['h0'].shape == (a.keep*n, N.STATE+N.SSLOW) and buf.x['feats'].shape[-1] == 2*N.NSTAT+N.NF
    assert lane.hs.shape == (n, N.SSTAT+N.SSLOW) and float(buf.x['h0'][2*n:3*n, N.STATE:].abs().sum()) > 0   # the slow sums of the window's first decision are kept with it
    assert log['head'] == log['head'] and log['law'] == log['law']
    assert float(net.pairs.stat2[-1].weight.abs().sum()) > 0                              # a step moved the layer that reads them


def test_the_rows_own_past_takes_the_loop_out_of_the_sums():
    """Row 0 answers MV 0 (gain 2, a lag of 5 decisions) under a load that drifts; a PI loop moves MV 0 against
    the row, with a small test on top. The MV then moves with the load, and the sums fitted on the moves alone
    put part of the load's work down to the MV: the response comes out a third too small in every run. Fitted
    with the row's own past beside the moves, it comes out right on average."""
    g = torch.Generator().manual_seed(2)
    b, T = 4, 3000
    y, pos, w, integ, ym = (torch.zeros(b) for _ in range(5))
    test = .1*torch.sign(torch.randn(b, T, generator=g))*(torch.rand(b, T, generator=g) < .1)
    dy, du = [], []
    for t in range(T):
        u = -.3*ym-.02*integ+test[:, t]                                                  # decided on what was seen up to the decision before
        pos = pos+u
        w = .95*w+.03*torch.randn(b, generator=g)
        y = y+.2*(2*pos-y)+w
        dy.append(y-ym); du.append(u)
        ym = y.clone(); integ = integ+ym
    obs = packet_of(torch.stack(dy, 1), torch.stack(du, 1)[..., None])
    stats = N.Stats(slow=1e6, arx=True)
    assert stats.size == N.SSTAT+N.SSLOW+N.SROW+2*N.SARX and N.Stats(arx=True).size == N.SSTAT+N.SROW+N.SARX
    F, hs = stats(obs)
    assert F.shape[-1] == 2*N.NSTAT+2*N.NARX and hs.shape == (b, stats.size)
    F0, h0 = N.Stats(slow=1e6)(obs)
    assert torch.equal(F[..., :2*N.NSTAT], F0) and torch.equal(hs[:, :N.SSTAT+N.SSLOW], h0)   # the sums and their features are as they are without it
    Fa, ha = stats(obs[:, :700])
    Fb, hb = stats(obs[:, 700:], ha)
    assert torch.allclose(torch.cat((Fa, Fb), 1), F, atol=2e-4) and torch.allclose(hb, hs, rtol=1e-4, atol=1e-6)   # the state carries all of it
    o = N.SSTAT
    R, c = hs[:, o:o+N.NX*N.NX].reshape(b, N.NX, N.NX), hs[:, o+N.NX*N.NX:o+N.SSLOW-NCV].reshape(b, NCV, N.NX)
    o = N.SSTAT+N.SSLOW+N.SROW+N.SARX                                                    # the slow set's P, Q, d
    P, Q = hs[:, o:o+NCV*N.NF*N.NX].reshape(b, NCV, N.NF, N.NX), hs[:, o+NCV*N.NF*N.NX:o+N.SARX-NCV*N.NF].reshape(b, NCV, N.NF, N.NF)
    d = hs[:, o+N.SARX-NCV*N.NF:o+N.SARX].reshape(b, NCV, N.NF)
    tb, ta = N.Stats.solve_own(R, c, P, Q, d)
    f = torch.float64                                                                    # the same as the whole system of row 0 solved at once
    lam, lam2 = 1e-3*R.diagonal(dim1=1, dim2=2).mean(-1).to(f)+1e-10, 1e-3*Q[:, 0].diagonal(dim1=1, dim2=2).mean(-1).to(f)+1e-10
    top = torch.cat((R.to(f)+lam[:, None, None]*torch.eye(N.NX, dtype=f), P[:, 0].to(f).transpose(1, 2)), -1)
    bot = torch.cat((P[:, 0].to(f), Q[:, 0].to(f)+lam2[:, None, None]*torch.eye(N.NF, dtype=f)), -1)
    th = torch.linalg.solve(torch.cat((top, bot), 1), torch.cat((c[:, 0], d[:, 0]), -1).to(f)[..., None])[..., 0]
    assert torch.allclose(th[:, :N.NX], tb[:, 0].reshape(b, N.NX), rtol=1e-5, atol=1e-7) and torch.allclose(th[:, N.NX:], ta[:, 0], rtol=1e-5, atol=1e-7)
    fa = F[:, -1, 0, 0, stats.at_arx][:, N.NARX:]                                        # the slow set's: the weights on MV 0's moves, then on the row's past
    assert torch.allclose(.45*torch.sinh(fa[:, :N.NF]), tb[:, 0, 0].float(), atol=1e-5) and torch.allclose(fa[:, N.NF:], ta[:, 0].float().clamp(-N.A_CLIP, N.A_CLIP), atol=1e-6)
    h, step = None, torch.zeros(N.NF)
    for t in range(8):
        _, h = N.Stats()(packet_of(torch.zeros(1, 1), torch.full((1, 1, 1), float(t == 0))), h)
        step = step+h[0, :N.NF]
    true = 2*(1-.8**8)                                                                   # row 0's response to MV 0, 8 decisions on
    plain = (.45*torch.sinh(F[:, -1, 0, 0, stats.at_slow][:, N.NF:2*N.NF])*step).sum(-1)
    own = .45*N.Stats.response(tb, ta, (8,))[:, 0, 0, 0]
    assert bool((plain < .8*true).all())
    assert abs(float(own.mean())-true) < .1*true and float((own-true).abs().mean()) < .5*float((plain-true).abs().mean())
    zero = N.Stats.response(torch.zeros_like(tb), ta, (1, 8))
    assert float(zero.abs().max()) == 0.                                                 # no weight on the moves: no response


def test_weights_from_before_the_rows_own_past_compute_what_they_did(packets):
    from aio import formula as FM
    obs, prev, _ = packets
    old = lively(N.AIO(slow=2000.))
    torch.nn.init.normal_(old.pairs.stat2[-1].weight, std=.3)
    new = N.AIO(slow=2000., arx=True)
    assert 'arx' not in old.pairs.state_dict() and not any(k.startswith('stat3') for k in old.pairs.state_dict())
    new.actor.load_state_dict(old.actor.state_dict()); new.pairs.load_state_dict(old.pairs.state_dict())
    assert not N.arx_of(old.pairs.state_dict()) and N.arx_of(new.pairs.state_dict())
    a, b = old.steps(obs, prev), new.steps(obs, prev)
    assert torch.equal(a[0], b[0]) and torch.equal(a[3], b[3])                           # bit for bit: the new layer adds zero
    more = N.SROW+2*N.SARX
    assert b[4].shape == (1, 8, N.STATE+N.SSLOW+more) and torch.equal(b[4][..., :N.STATE+N.SSLOW], a[4])
    with pytest.raises(RuntimeError):
        N.AIO(slow=2000.).pairs.load_state_dict(new.pairs.state_dict())                  # the other way round is refused
    torch.nn.init.normal_(new.pairs.stat3[-1].weight, std=.3)
    c = new.steps(obs, prev)
    assert float((c[3]-b[3]).abs().max()) > 1e-4                                          # in play, it is read
    first = new.steps(obs[:, :3], prev[:, :3])
    rest = new.steps(obs[:, 3:], prev[:, 3:], first[4])
    assert torch.allclose(torch.cat((first[0], rest[0]), 1), c[0], atol=1e-5) and torch.allclose(rest[4], c[4], atol=1e-5)
    lone = N.AIO(arx=True)                                                               # without a slow set: the same layer, the slow half of what it reads at zero
    assert lone.pairs.stat3[0].weight.shape == new.pairs.stat3[0].weight.shape and lone.steps(obs, prev)[4].shape == (1, 8, N.STATE+N.SROW+N.SARX)
    lone.pairs.load_state_dict({k: v for k, v in new.pairs.state_dict().items() if k != 'slow' and not k.startswith('stat2')})
    both = N.AIO(law=True, knots=16, formula=True, solve=True, slow=2000., arx=True)     # the formula's moves stay last
    assert both.pairs.size == N.SPAIR+N.SSTAT+N.SSLOW+more+FM.SMEM and both.steps(obs, prev)[4].shape == (1, 8, N.STATE+N.SSLOW+more+FM.SMEM)
    crit = N.Critic(2)                                                                   # a critic without it reads an actor's features that have it
    F = new.pairs.stats(obs)[0]
    assert torch.equal(crit.steps(obs, prev, prev, feats=F)[0], crit.steps(obs, prev, prev, feats=F[..., :N.NSTAT])[0])


def test_a_measured_disturbance_stands_beside_the_mvs_in_the_sums(packets):
    """Row 0 answers MV 0 (gain 2, a lag of 5 decisions) and a measured disturbance (gain 1.5, the same lag) that
    row 1 reads and that steps now and then; a PI loop moves MV 0 against row 0, with a small test on top. Fitted
    on the moves and the row's own past, what the disturbance does is left to those two to explain. With the
    disturbance row among the inputs the MV's response is closer, and the disturbance's own comes out of the fit."""
    from aio.interface import DV
    g = torch.Generator().manual_seed(3)
    b, T = 4, 3000
    y, pos, w, integ, ym, dist = (torch.zeros(b) for _ in range(6))
    test = .1*torch.sign(torch.randn(b, T, generator=g))*(torch.rand(b, T, generator=g) < .1)
    jump = torch.sign(torch.randn(b, T, generator=g))*(torch.rand(b, T, generator=g) < .02)
    dy, du, dd = [], [], []
    for t in range(T):
        u = -.3*ym-.02*integ+test[:, t]
        pos = pos+u
        dist = dist+jump[:, t]
        w = .95*w+.01*torch.randn(b, generator=g)
        y = y+.2*(2*pos+1.5*dist-y)+w
        dy.append(y-ym); du.append(u); dd.append(jump[:, t])
        ym = y.clone(); integ = integ+ym
    obs = packet_of(torch.stack(dy, 1), torch.stack(du, 1)[..., None])
    cv = obs[..., :NCV*CF].reshape(b, T, NCV, CF).clone()
    cv[:, :, 0, DV] = 1.                                                                 # row 0 is the MVs' to move; row 1 reads the disturbance
    cv[:, :, 1, 1], cv[:, :, 1, -1] = torch.stack(dd, 1), 1.
    seen = torch.cat((cv.flatten(2), obs[..., NCV*CF:]), -1)
    cv[:, :, 1] = 0.
    blind = torch.cat((cv.flatten(2), obs[..., NCV*CF:]), -1)                            # the same without that row
    with pytest.raises(AssertionError):
        N.Stats(dv=True)
    stats, plain = N.Stats(slow=1e6, arx=True, dv=True), N.Stats(slow=1e6, arx=True)
    assert stats.size == plain.size+N.SDVF+2*N.SDV and N.Stats(arx=True, dv=True).size == N.SSTAT+N.SROW+N.SARX+N.SDVF+N.SDV
    F, hs = stats(seen)
    F0, h0 = plain(seen)
    assert F.shape == F0.shape and hs.shape == (b, stats.size)
    assert torch.equal(F[..., :2*N.NSTAT], F0[..., :2*N.NSTAT]) and torch.equal(hs[:, :plain.size], h0)   # what was there is as it was
    Fn, hn = stats(blind)                                                                # no row reads a disturbance: the fits are the ones without
    assert torch.allclose(Fn, plain(blind)[0], atol=1e-5) and float(hn[:, plain.size:].abs().max()) == 0.
    Fa, ha = stats(seen[:, :700])
    Fb, hb = stats(seen[:, 700:], ha)
    assert torch.allclose(torch.cat((Fa, Fb), 1), F, atol=2e-4) and torch.allclose(hb, hs, rtol=1e-4, atol=1e-6)   # the state carries all of it
    o = N.SSTAT
    R, c = hs[:, o:o+N.NX*N.NX].reshape(b, N.NX, N.NX), hs[:, o+N.NX*N.NX:o+N.SSLOW-NCV].reshape(b, NCV, N.NX)
    o = N.SSTAT+N.SSLOW+N.SROW+N.SARX                                                    # the slow set's P, Q, d, then (behind the filters) its sums with the disturbances
    P, Q = hs[:, o:o+NCV*N.NF*N.NX].reshape(b, NCV, N.NF, N.NX), hs[:, o+NCV*N.NF*N.NX:o+N.SARX-NCV*N.NF].reshape(b, NCV, N.NF, N.NF)
    d = hs[:, o+N.SARX-NCV*N.NF:o+N.SARX].reshape(b, NCV, N.NF)
    o, cut = plain.size+N.SDVF+N.SDV, (0, N.NX*N.NXD, N.NX*N.NXD+N.NXD*N.NXD, N.NX*N.NXD+N.NXD*N.NXD+NCV*N.NXD, N.SDV)
    ins = [hs[:, o+i:o+j].reshape(b, *shape) for i, j, shape in zip(cut, cut[1:], ((N.NX, N.NXD), (N.NXD, N.NXD), (NCV, N.NXD), (NCV, N.NF, N.NXD)))]
    tb, ta, tz = N.Stats.solve_own(R, c, P, Q, d, *ins)
    tb0, ta0 = N.Stats.solve_own(R, c, P, Q, d)
    fa = F[:, -1, 0, 0, stats.at_arx][:, N.NARX:]                                        # the features are the wider fit's
    assert torch.allclose(.45*torch.sinh(fa[:, :N.NF]), tb[:, 0, 0].float(), atol=1e-5) and torch.allclose(fa[:, N.NF:], ta[:, 0].float().clamp(-N.A_CLIP, N.A_CLIP), atol=1e-6)
    true = 2*(1-.8**8)                                                                   # row 0's response to MV 0, 8 decisions on; to the disturbance 1.5 in place of 2
    wide, narrow = .45*N.Stats.response(tb, ta, (8,))[:, 0, 0, 0], .45*N.Stats.response(tb0, ta0, (8,))[:, 0, 0, 0]
    assert float((wide-true).abs().mean()) < .7*float((narrow-true).abs().mean()) and float((wide-true).abs().max()) < .1*true
    dist = .45*N.Stats.response(tz, ta, (8,))[:, 0, 0, 0]
    assert float((dist-.75*true).abs().max()) < .1*.75*true and float(tz[:, 0, 1].abs().max()) == 0.   # the first of them; there is no second
    assert float(tb[:, 1].abs().max()) < 5e-3                                            # the disturbance row itself: next to nothing of its change is the MVs'
    obs_, prev, _ = packets                                                              # the network: the same weights, the same layers
    old, new = N.AIO(slow=2000., arx=True), N.AIO(slow=2000., arx=True, dv=True)
    assert sorted(k for k in new.state_dict() if k not in old.state_dict()) == ['pairs.dvs']
    new.actor.load_state_dict(old.actor.state_dict()); new.pairs.load_state_dict(old.pairs.state_dict())
    assert N.dv_of(new.pairs.state_dict()) and not N.dv_of(old.pairs.state_dict())
    a_, b_ = old.steps(obs_, prev), new.steps(obs_, prev)
    more = N.SSLOW+N.SROW+2*N.SARX
    assert torch.equal(a_[0], b_[0]) and b_[4].shape == (1, 8, N.STATE+more+N.SDVF+2*N.SDV) and torch.equal(b_[4][..., :N.STATE+more], a_[4])
    with pytest.raises(AssertionError):
        N.AIO(slow=2000., dv=True)


def test_tied_rounds_take_the_mean_so_a_plant_written_twice_is_moved_as_the_plant():
    torch.manual_seed(0)
    net = lively(N.AIO(tied=6)).eval()
    prog = C.build(6, 5)
    o, s = canonical(C.observe(prog, C.reset(prog), 0).float())
    cv, mv, link = o[:, :NCV*CF].reshape(-1, NCV, CF), o[:, NCV*CF:NCV*CF+NMV*MF].reshape(-1, NMV, MF), o[:, NCV*CF+NMV*MF:].reshape(-1, NCV, NMV)
    twice = torch.cat((cv.repeat(1, 2, 1).flatten(1), mv.repeat(1, 2, 1).flatten(1), link.repeat(1, 2, 2).flatten(1)), 1)   # every row and every MV twice: 24x10
    lab = N.pair_labels(prog.labels(0).float()*s[:, :, None])
    with torch.no_grad():
        a = net.steps(o[:, None], torch.zeros(6, 1, NMV), None, (lab[:, None], torch.ones(6)))[0][:, 0]
        b = net.steps(twice[:, None], torch.zeros(6, 1, 2*NMV), None, (lab.repeat(1, 2, 2, 1)[:, None], torch.ones(6)))[0][:, 0]
    assert float(a.abs().max()) > 1e-3
    assert torch.allclose(b[:, :NMV], a, atol=1e-6) and torch.allclose(b[:, NMV:], a, atol=1e-6)   # a sum over the pairs would grow with their number, round after round


def test_the_trainer_gathers_and_steps_with_the_measured_disturbances_in_the_sums():
    from aio import train as T
    from tests.test_train_guard import args
    n = 8
    a = args(noise=0., replay=1, batch=8, keep=4, fit_chunk=4, law=True, lr=1e-3, known=.5, clip=10., slow=2000., arx=True, dv=True)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    net, critic = N.AIO(law=True, slow=2000., arx=True, dv=True), N.Critic(2)
    opt, copt = torch.optim.Adam(net.parameters(), lr=1e-3), torch.optim.Adam(critic.parameters(), lr=3e-4)
    prog = C.build(n, 5)
    assert bool((prog.c['dvr'] > 0).any())                                               # the batch has measured disturbances
    lane, buf = T.Lane(prog, a, gen, teach=1.), T.Buffer(a.keep*n)
    for _ in range(3):
        log = T.imitate(net, critic, opt, copt, lane, buf, a, gen)
    more = N.SSLOW+N.SROW+2*N.SARX+N.SDVF+2*N.SDV
    assert buf.x['h0'].shape == (a.keep*n, N.STATE+more) and buf.x['feats'].shape[-1] == 2*N.NSTAT+N.NF+2*N.NARX
    assert lane.hs.shape == (n, N.SSTAT+more) and float(buf.x['h0'][2*n:3*n, -2*N.SDV:].abs().sum()) > 0
    assert log['head'] == log['head'] and log['law'] == log['law']


def test_the_trainer_gathers_and_steps_with_the_rows_own_past():
    from aio import train as T
    from tests.test_train_guard import args
    n = 8
    a = args(noise=0., replay=1, batch=8, keep=4, fit_chunk=4, law=True, lr=1e-3, known=.5, clip=10., slow=2000., arx=True)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    net, critic = N.AIO(law=True, slow=2000., arx=True), N.Critic(2)
    opt, copt = torch.optim.Adam(net.parameters(), lr=1e-3), torch.optim.Adam(critic.parameters(), lr=3e-4)
    lane, buf = T.Lane(C.build(n, 5), a, gen, teach=1.), T.Buffer(a.keep*n)
    for _ in range(3):
        log = T.imitate(net, critic, opt, copt, lane, buf, a, gen)
    more = N.SSLOW+N.SROW+2*N.SARX
    assert buf.x['h0'].shape == (a.keep*n, N.STATE+more) and buf.x['feats'].shape[-1] == 2*N.NSTAT+N.NF+2*N.NARX
    assert lane.hs.shape == (n, N.SSTAT+more) and float(buf.x['h0'][2*n:3*n, N.STATE+N.SSLOW:].abs().sum()) > 0
    assert log['head'] == log['head'] and log['law'] == log['law']
    assert float(net.pairs.stat3[-1].weight.abs().sum()) > 0                              # a step moved the layer that reads it


def test_readouts_that_give_the_change_from_lag_to_lag(tmp_path, packets):
    from aio import formula as FM
    obs, prev, lab = packets
    torch.manual_seed(0)
    x = torch.randn(4, 6, 9).cumsum(-1)
    assert torch.allclose(N.values_of(N.steps_of(x, 12.), 12.), x, atol=1e-3)            # there and back
    assert torch.allclose(torch.sinh(N.steps_of(x, 12.)[..., 1:]), torch.diff(torch.sinh(x), dim=-1), atol=1e-2, rtol=1e-4)
    plain, net = lively(N.AIO(law=True)), N.AIO(law=True, diff='SE')
    assert N.diff_of(plain.pairs.state_dict()) == '' and N.diff_of(net.pairs.state_dict()) == 'SE'
    assert [q for q, _ in plain.named_parameters()] == [q for q, _ in net.named_parameters()]    # no weights of its own
    for a, b in ((net, plain), (plain, net)):                     # a read-out means the one or the other: neither takes the other's weights
        with pytest.raises(RuntimeError):
            a.pairs.load_state_dict(b.pairs.state_dict())
    net.load_state_dict(plain.state_dict(), strict=False)
    a, b = plain.steps(obs, prev), net.steps(obs, prev)
    # the same layers: what they gave as values is now the change from knot to knot, and the responses handed on are those added up
    assert torch.allclose(b[3], N.values_of(a[3], FM.S_CLIP), atol=1e-5) and float((b[3]-a[3]).detach().abs().max()) > 1e-3
    assert torch.allclose(N.taught(net, b[3], 'S'), a[3], atol=1e-4) and N.taught(plain, a[3], 'S') is a[3]
    E = net.pairs.read_law[0]
    assert E.shape == plain.pairs.read_law[0].shape and bool(torch.isfinite(N.taught(net, E, 'E')).all())
    told = net.steps(obs, prev, None, (lab, torch.ones(8)))       # told the responses: they take the place of the values, as before
    assert torch.allclose(told[3], b[3]) and float((told[0]-b[0]).detach().abs().max()) > 0
    only = N.AIO(law=True, diff='E')
    only.load_state_dict(plain.state_dict(), strict=False)
    assert torch.equal(only.steps(obs, prev)[3], a[3]) and N.diff_of(only.pairs.state_dict()) == 'E'
    with pytest.raises(AssertionError):
        N.AIO(diff='E')                                           # E: of a network with the law
    torch.save(dict(model=dict(actor=net.actor.state_dict(), pairs=net.pairs.state_dict()), config={}), tmp_path/'d.pt')
    back = N.load(tmp_path/'d.pt')[0]
    assert back.pairs.diff == 'SE' and torch.allclose(back.steps(obs, prev)[0], net.eval().steps(obs, prev)[0], atol=1e-6)


def test_the_trainer_teaches_those_readouts_the_changes():
    from aio import train as T
    from tests.test_train_guard import args
    n = 8
    a = args(noise=0., replay=1, batch=8, keep=4, fit_chunk=4, law=True, lr=1e-3, known=.5, clip=10., slow=2000.)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    net, critic = N.AIO(law=True, slow=2000., diff='SE'), N.Critic(2)
    opt, copt = torch.optim.Adam(net.parameters(), lr=1e-3), torch.optim.Adam(critic.parameters(), lr=3e-4)
    lane, buf = T.Lane(C.build(n, 5), a, gen, teach=1.), T.Buffer(a.keep*n)
    w0 = [net.pairs.read[2].weight.detach().clone(), net.pairs.heading[2].weight.detach().clone()]
    for _ in range(3):
        log = T.imitate(net, critic, opt, copt, lane, buf, a, gen)
    assert log['head'] == log['head'] and log['law'] == log['law'] and log['id'] == log['id']
    assert float((net.pairs.read[2].weight-w0[0]).abs().sum()) > 0 and float((net.pairs.heading[2].weight-w0[1]).abs().sum()) > 0


def test_readout_of_what_the_instruments_do_not_show(tmp_path, packets):
    obs, prev, lab = packets
    torch.manual_seed(0)
    plain, net = lively(N.AIO(law=True)), N.AIO(law=True, unseen=True)
    assert not N.unseen_of(plain.pairs.state_dict()) and N.unseen_of(net.pairs.state_dict())
    net.actor.load_state_dict(plain.actor.state_dict()); net.pairs.load_state_dict(plain.pairs.state_dict())   # weights from before it
    a, b = plain.steps(obs, prev), net.steps(obs, prev)
    assert torch.equal(a[0], b[0]) and torch.equal(plain.pairs.read_law[0], net.pairs.read_law[0])   # what gives it to E's read-out starts at zero
    U = net.pairs.read_unseen
    assert U.shape == obs.shape[:2]+(NCV, N.UNSEEN) and plain.pairs.read_unseen is None
    E, K, V = (x.detach() for x in net.pairs.read_law[:3])
    torch.nn.init.normal_(net.pairs.unseen_in.weight, std=.5)
    code = (lab, torch.ones(8), E, K, V)
    own = (net.steps(obs, prev, None, code), net.pairs.read_law[0].detach().clone())[1]        # told the law but not this: its own reading goes in
    told = [(net.steps(obs, prev, None, code+(x*torch.ones_like(U),)), net.pairs.read_law[0].detach().clone())[1] for x in (1., 2.)]
    assert float((told[0]-own).abs().max()) > 1e-3 and float((told[1]-told[0]).abs().max()) > 1e-3   # told: the true ones take its place
    none = (net.steps(obs, prev, None, (lab, torch.zeros(8), E, K, V, 2*torch.ones_like(U))), net.pairs.read_law[0].detach().clone())[1]
    assert torch.allclose(none, (net.steps(obs, prev), net.pairs.read_law[0].detach().clone())[1], atol=1e-6)   # an episode told nothing: its own
    assert net.pairs.read_unseen.requires_grad                    # and it is read in every episode: what it is taught on
    net.zero_grad()
    net.pairs.read_law[0].square().sum().backward()
    assert net.pairs.unseen_in.weight.grad.any() and not net.pairs.unseen.weight.grad    # E's loss teaches how the reading is used, not the reading
    for kw in (dict(), dict(law=True, formula=True)):
        with pytest.raises(AssertionError):
            N.AIO(unseen=True, **kw)                              # beside the E read-out that learns
    torch.save(dict(model=dict(actor=net.actor.state_dict(), pairs=net.pairs.state_dict()), config={}), tmp_path/'u.pt')
    back = N.load(tmp_path/'u.pt')[0]
    assert back.pairs.unseen is not None and torch.allclose(back.steps(obs, prev)[0], net.eval().steps(obs, prev)[0], atol=1e-6)


def test_the_pairs_are_given_the_share_of_the_responses_that_was_told(tmp_path, packets):
    obs, prev, lab = packets
    torch.manual_seed(0)
    plain, net = lively(N.AIO(law=True)), N.AIO(law=True, told=True)
    assert not N.told_of(plain.pairs.state_dict()) and N.told_of(net.pairs.state_dict())
    net.actor.load_state_dict(plain.actor.state_dict()); net.pairs.load_state_dict(plain.pairs.state_dict())   # weights from before it
    one, none = (lab, torch.ones(8)), (lab, torch.zeros(8))
    assert torch.equal(plain.steps(obs, prev, None, one)[0], net.steps(obs, prev, None, one)[0])   # the layer starts at zero
    torch.nn.init.normal_(net.pairs.told_in.weight, std=.5)
    K = lambda n, code: (n.steps(obs, prev, None, code), n.pairs.read_law[1].detach().clone())[1]   # noqa: E731
    assert float((K(net, one)-K(plain, one)).abs().max()) > 1e-3   # told the responses: the read-outs know it
    assert torch.equal(K(net, none), K(plain, none)) and torch.equal(K(net, None), K(plain, None))   # its own reading: nothing is added
    assert torch.equal(net.steps(obs, prev)[0], plain.steps(obs, prev)[0])     # as it runs
    half = (lab, torch.tensor([1., 1., 1., 1., 0., 0., 0., 0.]))
    k, p = K(net, half), K(plain, half)
    assert float((k[:4]-p[:4]).abs().max()) > 1e-3 and torch.equal(k[4:], p[4:])   # episode by episode
    net.zero_grad()
    net.steps(obs, prev, None, one)
    net.pairs.read_law[1].square().sum().backward()
    assert net.pairs.told_in.weight.grad.any()
    torch.save(dict(model=dict(actor=net.actor.state_dict(), pairs=net.pairs.state_dict()), config={}), tmp_path/'t.pt')
    back = N.load(tmp_path/'t.pt')[0]
    assert back.pairs.told_in is not None and torch.allclose(K(back, one), K(net.eval(), one), atol=1e-6)


def test_the_labels_of_what_the_instruments_do_not_show():
    torch.manual_seed(1)
    prog = C.build(16, 7)
    st = C.reset(prog)
    for k in range(30):
        st, _ = C.step(prog, st, (.2*torch.randn(16, NMV)*prog.c['mvm']).to(prog.dtype).clamp(-1, 1), k)
    U, c = N.unseen_code(prog, st, 30), prog.c
    trk, lev = (c['trk'] > 0)[:, :NMV], ((c['integ'] > 0) & (c['trk'] > 0))[:, :NMV]
    assert U.shape == (16, NCV, N.UNSEEN) and bool(torch.isfinite(U).all())
    assert not U[:, NMV:].any() and not U[:, :NMV][~trk].any()    # of the tracked rows
    assert not U[:, :NMV, 1][~lev].any()                          # a row that does not integrate stays where it is
    assert bool(lev.any()) and float(U[:, :NMV, 1][lev].abs().max()) > 0 and float(U[:, :NMV, 0][trk].abs().max()) > 0
    # a level's next change is its rate (aio.sim.core.step), whatever the MVs do at this decision
    v0, wy = st['v'][:, :NMV].clone(), prog.wy[prog.ar, prog.teacher_seg(30)]
    st1, _ = C.step(prog, st, torch.zeros(16, NMV, dtype=prog.dtype), 30)
    rate1 = (st1['fl'][:, :NMV]+c['lam'][:, :NMV]*(v0-c['vc'][:, :NMV]))/c['tint'][:, :NMV]
    assert torch.allclose((st1['v'][:, :NMV]-v0)[lev], rate1[lev], atol=1e-5)
    U1 = N.unseen_code(prog, st1, 31)
    now = (st1['fl'][:, :NMV]+c['lam'][:, :NMV]*(st1['v'][:, :NMV]-c['vc'][:, :NMV]))/c['tint'][:, :NMV]
    assert torch.allclose(U1[:, :NMV, 1][lev], (now*wy)[lev].float()/N.RATE_UNIT, atol=1e-4)


def test_the_trainer_teaches_what_the_instruments_do_not_show():
    from aio import train as T
    from tests.test_train_guard import args
    n = 8
    a = args(noise=0., replay=1, batch=8, keep=4, fit_chunk=4, law=True, lr=1e-3, known=.5, clip=10., slow=2000.)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    net, critic = N.AIO(law=True, slow=2000., unseen=True), N.Critic(2)
    opt, copt = torch.optim.Adam(net.parameters(), lr=1e-3), torch.optim.Adam(critic.parameters(), lr=3e-4)
    lane, buf = T.Lane(C.build(n, 5), a, gen, teach=1.), T.Buffer(a.keep*n)
    w0 = net.pairs.unseen.weight.detach().clone()
    for _ in range(3):
        log = T.imitate(net, critic, opt, copt, lane, buf, a, gen)
    assert log['unseen'] == log['unseen'] and log['gap_un'] == log['gap_un'] and log['head'] == log['head']
    assert buf.x['U'].shape[1:] == (a.window, NMV, N.UNSEEN) and buf.x['since'].shape[1:] == (a.window,) and buf.x['lev'].shape[1:] == (NMV,)
    assert float((net.pairs.unseen.weight-w0).abs().sum()) > 0 and float(net.pairs.unseen_in.weight.abs().sum()) > 0   # both moved by a step
    # once the simulator gradient is in the objective it stays taught with E (--law-keep)
    a = args(noise=0., law=True, law_keep='E', known=.5, clip=10., slow=2000.)
    lane = T.Lane(C.build(n, 6), a, gen, teach=0.)
    loss, _, _, log = T.window(net, critic, lane, a, dict(bc=0., sim=1., value=1.), gen)
    assert log['unseen'] == log['unseen'] and log['head'] == log['head'] and bool(torch.isfinite(loss))
    # the log: a read-out that is right explains all of it, in every band of decisions since the last event
    U = torch.randn(4, 6, NMV, 2)
    since, lev, wt = torch.tensor([3., 20., 70., 200., 5., 5.]).expand(4, 6), torch.ones(4, NMV), torch.ones(4, 6, NMV)
    out = T.unseen_log(U, U, since, lev, wt)
    assert out['gap_un'] == 0 and all(out['rate_un'+q] == 0 and out['rate_rk'+q] == pytest.approx(1.) for q in ('', '0', '16', '64', '128'))
    out = T.unseen_log(torch.zeros_like(U), U, since, lev, wt)
    assert out['gap_un'] == pytest.approx(1.) and out['rate_un'] == pytest.approx(1.)
    assert set(out) <= set(T.UNSEEN)


def test_the_same_weights_run_plants_of_the_larger_sizes():
    """No weight has a shape that depends on how many rows or MVs a packet has: every function takes the size from the
    packets it is given, and a network made at the default size runs a plant of 24 rows by 10 MVs as it is."""
    from aio import interface as I
    torch.manual_seed(0)
    net, critic = lively(N.AIO(law=True, slow=2000., arx=True, unseen=True)).eval(), N.Critic().eval()
    assert N.state_size() == N.STATE and net.state_size() == BASE+net.pairs.size and N.sizes()[:3] == (N.NPAIR, N.NX, N.SSTAT)
    prog = C.build(3, 5, dims='24x10')
    ncv, nmv = prog.ncv, prog.nmv
    assert (ncv, nmv) == (24, 10) and I.rows(nmv) == ncv and I.shape(I.size(ncv, nmv)) == (ncv, nmv)
    st = C.reset(prog)
    o, s = canonical(C.observe(prog, st, 0).float())
    assert o.shape == (3, I.size(ncv, nmv)) and s.shape == (3, nmv)
    lab = N.pair_labels(prog.labels(0).float()*s[:, :, None])
    E, K, V = N.law_code(prog, st, 0, s)[:3]
    U = N.unseen_code(prog, st, 0)
    assert lab.shape[:3] == (3, ncv, nmv) and E.shape == (3, ncv, NLAB) and K.shape == (3, ncv, nmv, NLAB) and V.shape == (3, nmv) and U.shape == (3, ncv, N.UNSEEN)
    for known in (False, True):                                   # on its own readings, and told the responses and the law
        ctl, st = N.Policy(net, known=known), C.reset(prog)
        ctl.reset(prog)
        with torch.no_grad():
            for k in range(3):
                a = ctl(prog, st, k, C.observe(prog, st, k))
                st, _ = C.step(prog, st, a, k)
        assert a.shape == (3, nmv) and bool(torch.isfinite(a).all()) and float(a.abs().sum()) > 0
        assert ctl.h.shape[-1] == net.state_size(ncv, nmv) > N.state_size(ncv, nmv) > N.STATE
    with torch.no_grad():
        q, h = critic.steps(o[:, None], torch.zeros(3, 1, nmv), torch.zeros(3, 1, nmv))
    assert q.shape[:2] == (3, 1) and h.shape[-1] == critic.state_size(ncv, nmv) and bool(torch.isfinite(q).all())
    # a sequence and single decisions give the same at this size too
    obs = torch.stack([o, o], 1)
    with torch.no_grad():
        whole = net.steps(obs, torch.zeros(3, 2, nmv))
        h1 = net.steps(obs[:, :1], torch.zeros(3, 1, nmv))[4]
        two = net.steps(obs[:, 1:], torch.zeros(3, 1, nmv), h1)
    assert torch.allclose(whole[0][:, 1], two[0][:, 0], atol=1e-5) and torch.allclose(whole[4], two[4], atol=1e-4)
    # the controller on one plant's packets takes its size from the first one
    one = N.Controller(net)
    assert one.step(C.observe(prog, C.reset(prog), 0)[0].float().numpy()).shape == (nmv,)


def test_reader_in_place_of_the_sums():
    """A network with the learned reader (aio.reader): no running sums, the same moves whether a sequence is run
    in one piece or in two with the state carried, and it comes back from its own weights."""
    import torch
    from aio import net as N
    from aio.interface import SIZE
    torch.manual_seed(0)
    net = N.AIO(stats=False, reader=True).eval()
    assert net.pairs.stats is None and net.pairs.reader is not None
    obs, prev = torch.rand(2, 6, SIZE), torch.zeros(2, 6, N.NMV)
    with torch.no_grad():
        whole = net.steps(obs, prev)
        a = net.steps(obs[:, :4], prev[:, :4])
        b = net.steps(obs[:, 4:], prev[:, 4:], a[4])
        again = N.of_state(net.actor.state_dict(), net.pairs.state_dict()).steps(obs, prev)
    assert whole[4].shape[-1] == net.state_size()
    assert torch.allclose(whole[0][:, 4:], b[0], atol=1e-5) and torch.allclose(whole[4], b[4], atol=1e-5)
    assert torch.equal(whole[0], again[0])
    sd = net.pairs.state_dict()
    assert N.reader_of(sd) and not N.has_stats(sd)


def test_reader_law_readouts_are_labels_alone():
    """With the learned reader the law's read-outs (E, K, V, their product) are taught and never run: the moves do
    not depend on their weights, and the read-outs are there to be taught."""
    import torch
    from aio import net as N
    from aio.interface import SIZE
    torch.manual_seed(0)
    net = N.AIO(stats=False, law=True, reader=True).eval()
    assert net.pairs.apart and not hasattr(net.pairs, 'law_in')
    obs, prev = torch.rand(2, 6, SIZE), torch.zeros(2, 6, N.NMV)
    heads = [p for q in ('kern', 'heading', 'gain', 'valve') for p in getattr(net.pairs, q).parameters()]
    mean = net.steps(obs, prev)[0]
    eh, kh, vh, mv = net.pairs.read_law
    assert all(g is None for g in torch.autograd.grad(mean.sum(), heads, allow_unused=True, retain_graph=True))
    assert any(g is not None and float(g.abs().sum()) > 0 for g in torch.autograd.grad(eh.sum()+kh.sum()+vh.sum(), heads, allow_unused=True))
    with torch.no_grad():
        for p in heads:
            p.normal_()
        assert torch.equal(net.steps(obs, prev)[0], mean)
    again = N.of_state(net.actor.state_dict(), net.pairs.state_dict())
    assert again.pairs.apart and again.pairs.law


def test_reader_all_decisions_at_once_is_the_loop():
    """aio.reader.Reader.forward (the sums of a window by cumulative sums) gives what one decision at a time gives,
    whole or in two pieces with the state carried."""
    import torch
    from aio import reader as RD
    from aio.interface import SIZE
    torch.manual_seed(0)
    r = RD.Reader(out=32)
    with torch.no_grad():
        for p in r.parameters():
            p.add_(.05*torch.randn_like(p))
        obs, prev = torch.rand(3, 150, SIZE), torch.rand(3, 150, 5)-.5
        a, sa = r(obs, prev)
        b, sb = r.forward_loop(obs, prev)
        a1, s1 = r(obs[:, :70], prev[:, :70])
        a2, s2 = r(obs[:, 70:], prev[:, 70:], s1)
    assert torch.allclose(a, b, atol=1e-4) and torch.allclose(sa, sb, rtol=1e-4, atol=1e-6)
    assert torch.allclose(a[:, 70:], a2, atol=1e-4) and torch.allclose(sa, s2, rtol=1e-4, atol=1e-6)


def test_reader_constant_key_and_shared_rounds():
    """The switches of aio.reader (`const`, `roundsN`): the window at once is the loop, the state has their size, and the
    weights say what was asked (aio.net.reader_of), so a network is rebuilt as it was."""
    import torch
    from aio import net as N, reader as RD
    from aio.interface import SIZE
    torch.manual_seed(0)
    for asked in ('const', 'rounds8', 'const,rounds8'):
        r = RD.Reader(out=32, **RD.spec(asked))
        with torch.no_grad():
            for p in r.parameters():
                p.add_(.02*torch.randn_like(p))
            obs, prev = torch.rand(2, 100, SIZE), torch.rand(2, 100, 5)-.5
            a, sa = r(obs, prev)
            b, sb = r.forward_loop(obs, prev)
            a1, s1 = r(obs[:, :40], prev[:, :40])
            a2, s2 = r(obs[:, 40:], prev[:, 40:], s1)
        assert torch.allclose(a, b, atol=1e-4) and torch.allclose(sa, sb, rtol=1e-4, atol=1e-6), asked
        assert torch.allclose(a[:, 40:], a2, atol=1e-4) and torch.allclose(sa, s2, rtol=1e-4, atol=1e-6), asked
        assert sa.shape[1] == r.size_of(12, 5) == RD.size_of(12, 5, const='const' in asked)
        net = N.PairNet(stats=False, law=True, reader=asked)
        assert N.reader_of(net.state_dict()) == asked
    assert N.reader_of(N.PairNet(stats=False, reader=True).state_dict()) is True
    assert RD.spec(True) == dict(const=False, rounds=RD.L, shared=False)



def test_law_read_or_labels_with_either():
    """`apart` says whether the law's product is read by what runs, with the running sums or the learned reader alike:
    the sums with the read-outs as labels alone, the reader with the product read; the weights say which."""
    import torch
    from aio import net as N
    from aio.interface import SIZE
    torch.manual_seed(0)
    obs, prev = torch.rand(2, 6, SIZE), torch.zeros(2, 6, N.NMV)
    for kw, apart in ((dict(stats=True, apart=True), True), (dict(stats=False, reader=True, apart=False), False), (dict(stats=True), False)):
        net = N.AIO(law=True, **kw).eval()
        assert net.pairs.apart is apart and hasattr(net.pairs, 'law_in') is not apart
        if not apart:
            torch.nn.init.normal_(net.pairs.law_in.weight)
        heads = [p for q in ('kern', 'heading', 'gain', 'valve') for p in getattr(net.pairs, q).parameters()]
        grads = torch.autograd.grad(net.steps(obs, prev)[0].sum(), heads, allow_unused=True)
        assert all(g is None for g in grads) if apart else any(g is not None and float(g.abs().sum()) > 0 for g in grads)
        again = N.of_state(net.actor.state_dict(), net.pairs.state_dict())
        assert again.pairs.apart is apart and (again.pairs.reader is None) == (net.pairs.reader is None)


def test_wide_pairs_by_the_environment():
    """AIO_WIDE=2: the pair pathway twice as wide, every size following; its weights say so and do not load at another width."""
    import os, subprocess, sys
    code = ('import torch\nfrom aio import net as N\nfrom aio.interface import SIZE, NMV\n'
            'net = N.AIO(stats=True, law=True)\nassert N.P == 64 and N.D == 96 and net.state_size() == N.STATE\n'
            'out = net.steps(torch.zeros(2, 3, SIZE), torch.zeros(2, 3, NMV))\nassert out[4].shape == (1, 2, N.STATE)\n'
            'assert int(net.pairs.state_dict()["wide"]) == 2\n')
    subprocess.run([sys.executable, '-c', code], check=True, env=dict(os.environ, AIO_WIDE='2'))
    net = N.AIO(stats=True, law=True)
    assert N.WIDE == 1 and 'wide' not in net.pairs.state_dict()
    with pytest.raises(AssertionError):
        N.of_state(net.actor.state_dict(), dict(net.pairs.state_dict(), wide=torch.tensor(2)))


def test_reader_sizes_are_read_back_from_the_weights():
    """The learned reader's sizes (aio.reader.spec: k keys per MV, v values per row, w the width of its layers) are in
    its weights: a network built from them alone is the same one, and its state has room for the larger sums."""
    from aio import reader as RD
    for asked in ('k16,w96', 'const,rounds8,k16,v8,w96'):
        net = N.PairNet(stats=False, reader=asked, law=True)
        assert N.reader_of(net.state_dict()) == asked
        sp = RD.said(asked)
        assert net.reader.size_of(N.NCV, N.NMV) == RD.size_of(N.NCV, N.NMV, **sp) > RD.size_of(N.NCV, N.NMV)
    assert N.reader_of(N.PairNet(stats=False, reader=True).state_dict()) is True


SEQ_SPECS = ('d64,l2,h4', 'd64,l2,h4,m8,g1,q1')     # as before; with the 11-23 parts (memory, lag routing, QK-norm)


@pytest.mark.parametrize('asked', SEQ_SPECS)
def test_seq_network_steps_alike_alone_and_in_windows(asked):
    """aio.seqnet.SeqAIO behind AIO's interface: while the history fits in the state, a decision at a time from the
    carried state gives what windows give; the weights say what network they are of."""
    from aio import seqnet as SQ
    torch.manual_seed(0)
    net = SQ.SeqAIO(asked, w=32)
    for p in net.parameters():
        if p.dim() > 1 and float(p.abs().sum()) == 0:
            torch.nn.init.normal_(p, std=.05)
    b, T = 2, 24
    o = torch.randn(b, T, SIZE)
    o[..., :NCV*CF].reshape(b, T, NCV, CF)[..., -1] = 1
    o[..., NCV*CF:NCV*CF+NMV*MF].reshape(b, T, NMV, MF)[..., -1] = 1
    o[..., -NCV*NMV:] = 0
    pv = torch.randn(b, T, NMV)

    def run(step, which=0):
        h, out = None, []
        for t in range(0, T, step):
            got = net.steps(o[:, t:t+step], pv[:, t:t+step], h)
            h = got[4]
            out.append(got[which])
        return torch.cat(out, 1), h

    (a, h), (c, _) = run(1), run(6)
    assert h.shape[-1] == net.state_size() and float(a.abs().mean()) > 1e-3
    assert torch.allclose(a, c, atol=1e-5)
    # --probe: the head's second output (`when`), alike alone and in windows; weights from before it take it at zero
    (a2, _), (c2, _) = run(1, 1), run(6, 1)
    assert float(a2.abs().mean()) > 1e-4 and torch.allclose(a2, c2, atol=1e-5)
    old = {q: x for q, x in net.actor.state_dict().items() if not q.startswith('when.')}
    was = N.of_state(old, net.pairs.state_dict(), config=dict(probe=.3))
    assert was.probe == .3 and float(was.steps(o[:, :4], pv[:, :4])[1].abs().sum()) == 0
    assert torch.allclose(was.steps(o[:, :4], pv[:, :4])[0], net.eval().steps(o[:, :4], pv[:, :4])[0], atol=1e-6)
    again = N.of_state(net.actor.state_dict(), net.pairs.state_dict())
    assert torch.allclose(again.steps(o[:, :4], pv[:, :4])[0], net.eval().steps(o[:, :4], pv[:, :4])[0], atol=1e-6)


@pytest.mark.parametrize('asked', SEQ_SPECS)
def test_seq_network_computes_only_the_new_decision(asked):
    """Without a graph a single decision is computed on what is kept of the ones before it (aio.seqnet.Room): the
    same moves as the whole history gives while it fits; a batch whose kept decisions are gone, two batches taking
    turns and a plant whose state is cleared on the way give them too."""
    from aio import seqnet as SQ
    torch.manual_seed(0)
    net = SQ.SeqAIO(asked, w=32).eval()
    for p in net.parameters():
        if p.dim() > 1 and float(p.abs().sum()) == 0:
            torch.nn.init.normal_(p, std=.05)
    b, T = 3, 24
    o = torch.randn(2, b, T, SIZE)
    o[..., :NCV*CF].reshape(2, b, T, NCV, CF)[..., -1] = 1
    o[..., NCV*CF:NCV*CF+NMV*MF].reshape(2, b, T, NMV, MF)[..., -1] = 1
    o[..., -NCV*NMV:] = 0
    pv = torch.randn(2, b, T, NMV)
    whole = [net.steps(o[i], pv[i])[0].detach() for i in range(2)]
    with torch.no_grad():
        h, out = [None, None], [[], []]
        for t in range(T):
            for i in range(2):                                    # two batches of a shape, a decision each in turn
                mean, _, _, lab, h[i] = net.steps(o[i][:, t:t+1], pv[i][:, t:t+1], h[i])
                out[i].append(mean)
            if t == 9:
                net.forget()                                      # built again from the packets in the state
        assert len(net.rooms) == 1 and len(next(iter(net.rooms.values()))) == 2
        for i in range(2):
            assert torch.allclose(torch.cat(out[i], 1), whole[i], atol=1e-5)
        h, out = None, []
        for t in range(T):                                        # the first plant starts again at decision 10
            if t == 10:
                h[0, 0] = 0
            mean, _, _, _, h = net.steps(o[0][:, t:t+1], pv[0][:, t:t+1], h)
            out.append(mean)
        out = torch.cat(out, 1)
        assert torch.allclose(out[1:], whole[0][1:], atol=1e-5)
        assert torch.allclose(out[0, 10:], net.steps(o[0][:1, 10:], pv[0][:1, 10:])[0], atol=1e-5)
        net.most = 0                                              # nothing kept: every decision over its whole history, as before
        h, out = None, []
        for t in range(T):
            mean, _, _, _, h = net.steps(o[1][:, t:t+1], pv[1][:, t:t+1], h)
            out.append(mean)
        assert torch.allclose(torch.cat(out, 1), whole[1], atol=1e-5)


@pytest.mark.parametrize('asked', SEQ_SPECS)
def test_seq_network_single_decisions_with_a_graph(asked):
    """A decision at a time with a graph (the plant's gradient), from a detached state: the moves and the gradient
    of a window from that state, the graph going through the decisions since; then on without a graph as before."""
    from aio import seqnet as SQ
    torch.manual_seed(0)
    net = SQ.SeqAIO(asked, w=32).eval()
    for p in net.parameters():
        if p.dim() > 1 and float(p.detach().abs().sum()) == 0:
            torch.nn.init.normal_(p, std=.05)
    b, T = 3, 26
    o = torch.randn(b, T, SIZE)
    o[..., :NCV*CF].reshape(b, T, NCV, CF)[..., -1] = 1
    o[..., NCV*CF:NCV*CF+NMV*MF].reshape(b, T, NMV, MF)[..., -1] = 1
    o[..., -NCV*NMV:] = 0
    pv = torch.randn(b, T, NMV)
    whole = net.steps(o, pv)[0].detach()
    with torch.no_grad():
        h = None
        for t in range(10):
            _, _, _, _, h = net.steps(o[:, t:t+1], pv[:, t:t+1], h)
    ps = [p for p in net.parameters()]
    want = torch.autograd.grad(net.steps(o[:, 10:18], pv[:, 10:18], h)[0].square().sum(), ps, allow_unused=True)
    for again in range(2):                                        # two windows: the second from the detached state of the first
        out, g = [], h
        for t in range(10+8*again, 18+8*again):
            mean, _, _, _, g = net.steps(o[:, t:t+1], pv[:, t:t+1], g)
            out.append(mean)
        out = torch.cat(out, 1)
        assert torch.allclose(out, whole[:, 10+8*again:18+8*again], atol=1e-5)
        got = torch.autograd.grad(out.square().sum(), ps, allow_unused=True)
        if not again:
            for x, y in zip(got, want):
                assert (x is None) == (y is None) and (x is None or torch.allclose(x, y, atol=1e-4, rtol=1e-3))
        h = g.detach()
    with torch.no_grad():
        assert torch.allclose(net.steps(o[:, 26-1:26], pv[:, 26-1:26], net.steps(o[:, :25], pv[:, :25])[4])[0], whole[:, 25:], atol=1e-5)
        assert torch.allclose(net.steps(o[:, 25:26], pv[:, 25:26], net.steps(o[:, 24:25], pv[:, 24:25], net.steps(o[:, :24], pv[:, :24])[4])[4])[0], whole[:, 25:], atol=1e-5)


def _seq_inputs(b, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    o = torch.randn(b, T, SIZE, generator=g)
    o[..., :NCV*CF].reshape(b, T, NCV, CF)[..., -1] = 1
    o[..., NCV*CF:NCV*CF+NMV*MF].reshape(b, T, NMV, MF)[..., -1] = 1
    o[..., -NCV*NMV:] = 0
    return o, torch.randn(b, T, NMV, generator=g)


def _seq_net(asked):
    from aio import seqnet as SQ
    torch.manual_seed(0)
    net = SQ.SeqAIO(asked, w=32).eval()
    for p in net.parameters():
        if p.dim() > 1 and float(p.detach().abs().sum()) == 0:
            torch.nn.init.normal_(p, std=.05)
    return net


@pytest.mark.parametrize('asked', SEQ_SPECS)
def test_seq_peek_leaves_the_room_as_it_was(asked):
    """The trainer's decision after a window's last (no graph, under `peek`) is not kept: the next window, from the state
    of the window's last decision, takes the same Room (none built again) and gives what it would have."""
    net = _seq_net(asked)
    o, pv = _seq_inputs(2, 30)
    whole = net.steps(o, pv)[0].detach()
    with torch.no_grad():
        h = None
        for t in range(8):
            _, _, _, _, h = net.steps(o[:, t:t+1], pv[:, t:t+1], h)
    for w0 in (8, 16):                                            # two windows with a graph, each followed by a decision under peek
        g = h
        for t in range(w0, w0+8):
            mean, _, _, _, g = net.steps(o[:, t:t+1], pv[:, t:t+1], g)
            assert torch.allclose(mean, whole[:, t:t+1], atol=1e-5)
        built = net.built
        with torch.no_grad(), net.peek():
            peeked = net.steps(o[:, w0+8:w0+9], pv[:, w0+8:w0+9], g)[0]
        assert torch.allclose(peeked, whole[:, w0+8:w0+9], atol=1e-5)
        h = g.detach()
        with torch.no_grad():
            again = net.steps(o[:, w0+8:w0+9], pv[:, w0+8:w0+9], h)[0]
        assert net.built == built and torch.allclose(again, whole[:, w0+8:w0+9], atol=1e-5)


@pytest.mark.parametrize('asked', SEQ_SPECS)
def test_seq_cut_stops_the_gradient_through_the_kept_decisions(asked):
    """--guard cut: after net.cut(stop) an episode's later moves pass no gradient to its earlier inputs, through the
    time attention's kept decisions as through the state; the others' still do."""
    net = _seq_net(asked)
    o, pv = _seq_inputs(2, 20)
    with torch.no_grad():
        h = None
        for t in range(6):
            _, _, _, _, h = net.steps(o[:, t:t+1], pv[:, t:t+1], h)
    x = o[:, 6:7].clone().requires_grad_(True)
    mean, _, _, _, g = net.steps(x, pv[:, 6:7], h)
    g = g+0.*x.sum()                                              # (the state says that a graph goes with it, as the trainer's does)
    for t in range(7, 10):
        if t == 8:
            stop = torch.tensor([True, False])
            net.cut(stop)
            g = torch.cat((g[:, :1].detach(), g[:, 1:]), 1)        # (the state is cut too: train.cut_rows)
        mean, _, _, _, g = net.steps(o[:, t:t+1], pv[:, t:t+1], g)
    gx, = torch.autograd.grad(mean.square().sum(), x)
    assert float(gx[0].abs().sum()) == 0 and float(gx[1].abs().sum()) > 0


def test_gauges_of_a_step_of_the_seq_network():
    """aio.train.parts: the gradient's norm by part of the --seq network and the size of its move head; none of another."""
    from aio import seqnet as SQ, train as TR
    torch.manual_seed(0)
    net = SQ.SeqAIO('d64,l3,h4', w=32)
    o = torch.randn(2, 4, SIZE)
    o[..., :NCV*CF].reshape(2, 4, NCV, CF)[..., -1] = 1
    o[..., NCV*CF:NCV*CF+NMV*MF].reshape(2, 4, NMV, MF)[..., -1] = 1
    (net.steps(o, torch.randn(2, 4, NMV))[0]-1.).square().sum().backward()
    g = TR.parts(net)
    assert set(g) == set(TR.GAUGES)-{'act_rms', 'act_sat', 'act_step'} and g['w_out'] == 0 and g['gn_out'] > 0
    assert g['gn_lo'] == 0 == g['gn_in']                              # (the head's last layer is zero at the start: nothing reaches below it)
    assert TR.parts(N.AIO()) == {}


def test_level_stops_pass_the_gradient_only_when_asked():
    """aio.sim.core.step(stops=True): the same plant, and a level held at its physical stop still says which way a valve moves it."""
    from aio.sim import stay as ST, core as C
    prog = ST.C.build(8, ST.EXAM_SEED)
    c = prog.c
    lev = (c['integ'] > 0).any(1)
    assert bool(lev.any())
    st = C.reset(prog)
    st['v'] = torch.where(c['integ'] > 0, torch.full_like(st['v'], 5.), st['v'])          # beyond the upper stop
    out, noise = {}, prog.g.get_state()
    for stops in (False, True):
        prog.g.set_state(noise)                                      # (the same noise both times)
        a = torch.zeros(prog.n, prog.nmv, dtype=prog.dtype, requires_grad=True)
        s, _ = C.step(prog, {q: (x.clone() if torch.is_tensor(x) else x) for q, x in st.items()}, a, 0, stops=stops)
        s2, _ = C.step(prog, s, a, 1, stops=stops)
        g, = torch.autograd.grad((s2['v']*(c['integ'] > 0)).sum(), a, allow_unused=True)
        out[stops] = (s2['v'].detach(), torch.zeros_like(a) if g is None else g)
    assert torch.equal(out[False][0], out[True][0])
    assert float(out[False][1][lev].abs().sum()) == 0 and float(out[True][1][lev].abs().sum()) > 0


def test_seq_grown_from_older_weights_moves_as_they_did():
    """--init of a seqnet without m, g, p into one with them: the added parts start where they change nothing, so the moves are
    the older network's, over a whole sequence and step by step (the memory and the lags go through the state)."""
    old = _seq_net('d64,l2,h4')
    from aio import seqnet as SQ
    new = SQ.SeqAIO('d64,l2,h4,m8,g1', w=32).eval()
    got = new.actor.load_state_dict(old.actor.state_dict(), strict=False)
    assert not got.unexpected_keys
    o, pv = _seq_inputs(2, 24)
    with torch.no_grad():
        want = old.steps(o, pv)[0]
        assert torch.allclose(new.steps(o, pv)[0], want, atol=1e-5)
        h = None
        for t in range(24):
            mean, _, _, _, h = new.steps(o[:, t:t+1], pv[:, t:t+1], h)
            assert torch.allclose(mean, want[:, t:t+1], atol=1e-5)


def test_policy_probe_skips_paired_when_unpaired_trained():
    """With net.unpaired > 0 and net.probe > 0, Policy/Controller inject probe noise only on blind rollouts,
    leaving paired rollouts deterministic; legacy net.unpaired == 0 still probes as before."""
    from aio.sim import core as C, control as CT
    net = _seq_net('d64,l2,h4,m8,g1,q1')
    with torch.no_grad():
        net.actor.when.bias.fill_(5.)                                                     # probing(second) > 0 everywhere
    prog = C.build(4, 5)
    net.unpaired, net.probe = .25, 0.
    paired_clean = CT.rollout(prog, N.Policy(net, paired=True), 8, trace=[0])['trace']
    blind_clean = CT.rollout(prog, N.Policy(net, paired=False), 8, trace=[0])['trace']
    net.probe = .5
    paired_probed = CT.rollout(prog, N.Policy(net, paired=True), 8, trace=[0])['trace']
    blind_probed = CT.rollout(prog, N.Policy(net, paired=False), 8, trace=[0])['trace']
    assert [r['u'] for r in paired_clean] == [r['u'] for r in paired_probed]               # paired eval: never probed
    assert [r['u'] for r in blind_clean] != [r['u'] for r in blind_probed]                # blind eval: probed
    net.unpaired = 0.                                                                     # legacy model: probes as before
    legacy_probed = CT.rollout(prog, N.Policy(net), 8, trace=[0])['trace']
    assert [r['u'] for r in paired_clean] != [r['u'] for r in legacy_probed]


def test_seq_forget_releases_both_rooms_and_last_room():
    """SeqAIO.forget() must drop both `rooms` and `last_room` so validation Rooms (~9.6 GB at 48x20@32) are not pinned on GPU."""
    import gc, weakref
    net = _seq_net('d64,l2,h4,m8,g1,q1')
    o, pv = _seq_inputs(2, 4)
    with torch.no_grad():
        net.steps(o[:, :1], pv[:, :1])
    assert net.rooms and net.last_room is not None
    ref = weakref.ref(net.last_room)
    net.forget()
    gc.collect()
    assert net.rooms == {} and net.last_room is None and ref() is None
    net.cut(torch.tensor([True, False]))                                                  # safe no-op after forget()


def test_seq_mup_readout_spec_compat_and_exact_migration():
    """u1 scales act[-1] by 1/sqrt(d), round-trips through Held.seq/of_buffer while leaving u0 legacy buffer lengths untouched,
    and multiplying act[-1] weights/bias by sqrt(d) preserves forward outputs identically while cutting act[-1] grad norm by sqrt(d)."""
    import math
    from aio import seqnet as SQ
    u0 = SQ.SeqAIO('d64,l2,h4,m8,g1,q1', w=16)
    u1 = SQ.SeqAIO('d64,l2,h4,m8,g1,q1,u1', w=16)
    assert len(u0.pairs.seq) == 7 and SQ.of_buffer(u0.pairs.seq) == ('d64,l2,h4,m8,g1,q1', 16)
    assert len(u1.pairs.seq) == 8 and SQ.of_buffer(u1.pairs.seq) == ('d64,l2,h4,m8,g1,q1,u1', 16)
    plain = SQ.SeqAIO('d64,l2,h4', w=16)
    assert len(plain.pairs.seq) == 4 and SQ.of_buffer(plain.pairs.seq) == ('d64,l2,h4', 16)

    torch.manual_seed(0)
    with torch.no_grad():
        u0.actor.act[-1].weight.normal_(0, .05)
        u0.actor.act[-1].bias.fill_(.02)
    u1.actor.load_state_dict(u0.actor.state_dict())
    with torch.no_grad():
        u1.actor.act[-1].weight.mul_(math.sqrt(64))
        u1.actor.act[-1].bias.mul_(math.sqrt(64))
    o, pv = _seq_inputs(2, 6)
    m0 = u0.steps(o, pv)[0]
    m1 = u1.steps(o, pv)[0]
    assert torch.allclose(m0, m1, atol=1e-6)
    assert u1.actor.x_norm_last is not None and float(u1.actor.x_norm_last) > 0
    u0.zero_grad(); u1.zero_grad()
    (m0 - .1).square().mean().backward()
    (m1 - .1).square().mean().backward()
    assert float(u1.actor.act[-1].weight.grad.norm()) == pytest.approx(float(u0.actor.act[-1].weight.grad.norm())/8., rel=1e-5)
    assert torch.allclose(u1.actor.act[0].weight.grad, u0.actor.act[0].weight.grad, atol=1e-6)


def test_seq_step_graph_grad_ckpt_matches_exact_forward_and_gradients():
    """Vectorized Layer.step_graph with grad_ckpt=True must match grad_ckpt=False bit-for-bit on outputs and gradients,
    and still respect SeqAIO.cut() along the graph."""
    torch.manual_seed(7)
    n0 = _seq_net('d64,l2,h4,m8,g1,q1,u1')
    n1 = _seq_net('d64,l2,h4,m8,g1,q1,u1')
    with torch.no_grad():
        for p in n0.parameters():
            if p.dim() >= 2 and float(p.abs().max()) == 0:
                p.normal_(0, .05)
    n1.load_state_dict(n0.state_dict())
    n1.actor.grad_ckpt = True
    o, pv = _seq_inputs(3, 10)

    def run_seq(net):
        h = None
        ys = []
        for t in range(10):
            if t == 5:
                net.cut(torch.tensor([True, False, False]))
            y, _, _, _, h = net.steps(o[:, t:t+1], pv[:, t:t+1], h)
            ys.append(y)
        out = torch.cat(ys, 1)
        out.square().mean().backward()
        return out.detach()

    y0 = run_seq(n0)
    y1 = run_seq(n1)
    assert torch.allclose(y0, y1, atol=1e-5)
    for (q, p0), (_, p1) in zip(n0.named_parameters(), n1.named_parameters()):
        if p0.grad is None:
            assert p1.grad is None, q
        else:
            assert p1.grad is not None and torch.allclose(p0.grad, p1.grad, atol=1e-4, rtol=1e-4), q


def test_priv_critic_size_agnostic_and_differentiable_through_delay_line():
    """PrivCritic runs across 12x5/24x10/48x20 with shared weights, indexes the delay pipeline strictly by each
    pair's dead time dd (zero leak from expired slots s >= dd into cv_feat), and sends non-zero exact gradients
    into an action made on the very last step (still inside dead time / buf pipeline) as well as true level states."""
    from aio.sim import core as C
    crit = N.PrivCritic(heads=4, d=64, layers=2, nheads=4)
    for dims in ('12x5', '24x10'):
        prog = C.build(2, 11, 'cpu', 'cpu', dims=dims, teacher=False)
        st = C.reset(prog)
        u = torch.zeros(2, prog.nmv, requires_grad=True)
        st1, _ = C.step(prog, st, u, 0, through=True, stops=True)
        z, pf = crit.eval_state(prog, st1, 1)
        assert z.shape == (2, 4) and torch.isfinite(z).all()
        assert pf[0].shape[-1] == N.PRIV_CV and pf[2].shape[-1] == N.PRIV_EDGE
        z.sum().backward()
        assert u.grad is not None and float(u.grad.abs().sum()) > 0
        # Batched parallel eval over (B, T) matches single-step eval
        pf_seq = tuple(x[:, None].expand(-1, 3, *x.shape[1:]) for x in pf)
        z_seq = crit(*pf_seq)
        assert z_seq.shape == (2, 3, 4) and torch.allclose(z_seq[:, 0], z.detach(), atol=1e-5)
    # Verify dead-time masking: expired buf slots (s > max_dd) do not leak into cv_feat
    prog = C.build(4, 11, 'cpu', 'cpu', dims='12x5', teacher=False)
    st = C.detach(C.reset(prog))
    buf_var = st['buf'].clone().detach().requires_grad_(True)
    st_test = dict(st, buf=buf_var)
    cv_f, _, _, _, _, _ = N.priv_features(prog, st_test, 0)
    cv_f.sum().backward()
    P0 = prog.params(torch.zeros(4, dtype=torch.long), 0)
    phi0 = C._phi(st['xv'], st['d'], prog.c)
    dd0 = (P0['dead'][:, :, :prog.nmv] / phi0[:, None, None]).round().clamp(0, 60).long()
    for b in range(4):
        for j in range(prog.nmv):
            m_dd = int(dd0[b, :, j].max())
            if m_dd + 1 < buf_var.shape[1]:
                assert float(buf_var.grad[b, m_dd+1:, j].abs().max()) == 0.0


def test_seq_v2_dense_lags_mem_feedback_and_edge_memory():
    """g2 (dense lags 1,3,8,16,24,36,48,64), f1 (mem0->Layer0 feedback), and eK (persistent pairwise bilinear
    edge memory injected into every layer's channel attention) preserve exact step-0 identity when grown from
    g1 weights (for f1+eK), match T>1 vs T=1 bit-for-bit, round-trip through Held.seq/of_buffer, remain
    size-agnostic across 12x5/48x20, and pass finite gradients through grad_ckpt=True step_graph."""
    from aio import seqnet as SQ
    from aio.sim import core as C
    old = _seq_net('d32,l2,h4,m8,g1,q1,u1')
    with torch.no_grad():
        old.actor.mem_out.weight.normal_(0, .05)
        old.actor.act[-1].weight.normal_(0, .1)
    # 1. Spec & Held.seq round-trip (leaving legacy g1 buffer lengths untouched)
    n_fe = SQ.SeqAIO('d32,l2,h4,m8,g1,q1,u1,f1,e8', w=32)
    n_v2 = SQ.SeqAIO('d32,l2,h4,m8,g2,q1,u1,f1,e8', w=32)
    assert SQ.of_buffer(n_fe.pairs.seq) == ('d32,l2,h4,m8,g1,q1,u1,f1,e8', 32)
    assert SQ.of_buffer(n_v2.pairs.seq) == ('d32,l2,h4,m8,g2,q1,u1,f1,e8', 32)
    assert len(old.pairs.seq) == 8 and SQ.of_buffer(old.pairs.seq) == ('d32,l2,h4,m8,g1,q1,u1', 32)
    # 2. Exact step-0 forward identity when growing f1 + e8 onto g1 weights
    got = n_fe.actor.load_state_dict(old.actor.state_dict(), strict=False)
    assert not got.unexpected_keys
    o, pv = _seq_inputs(2, 12)
    old.eval(); n_fe.eval()
    with torch.no_grad():
        want = old.steps(o, pv)[0]
        assert torch.allclose(n_fe.steps(o, pv)[0], want, atol=1e-5)
    # 3. T>1 vs step-by-step T=1 exact equivalence and grad_ckpt=True BPTT across 12x5 and 48x20
    with torch.no_grad():
        n_v2.actor.mem_fb.weight.normal_(0, .05)
        for lay in n_v2.actor.layers:
            lay.edge_bias.weight.normal_(0, .1)
            for lin in (lay.out_t, lay.out_l, lay.out_c, lay.ff[-1]):
                lin.weight.normal_(0, .05)
        n_v2.actor.act[-1].weight.normal_(0, .1)
    for dims, nmv in (('12x5', 5), ('48x20', 20)):
        prog = C.build(2, 13, 'cpu', 'cpu', dims=dims, teacher=False)
        st = C.reset(prog)
        os_, pvs = [], []
        a = torch.zeros(2, nmv)
        for k in range(5):
            ok_, s = canonical(C.observe(prog, st, k).float())
            os_.append(ok_); pvs.append(a)
            a = (.2*torch.randn(2, nmv)).clamp(-1, 1)*prog.c['mvm'].float()
            st, _ = C.step(prog, st, (a*s).to(prog.dtype), k)
        obs_t, prev_t = torch.stack(os_, 1), torch.stack(pvs, 1)
        n_v2.eval(); n_v2.forget()
        m_seq, _, _, _, h_seq = n_v2.steps(obs_t, prev_t)
        n_v2.forget(); h = None; ms = []
        for t in range(5):
            mt, _, _, _, h = n_v2.steps(obs_t[:, t:t+1], prev_t[:, t:t+1], h)
            ms.append(mt)
        assert torch.allclose(torch.cat(ms, 1), m_seq, atol=1e-5)
        assert torch.allclose(h, h_seq, atol=1e-5)
        # Grad checkpointing BPTT reaches mem_fb, edge_bias, edge_r, edge_m, edge_a, and all 8 g2 lags
        n_v2.actor.grad_ckpt = True
        n_v2.train(); n_v2.zero_grad(); n_v2.forget()
        h = None; loss = 0.
        for t in range(5):
            mt, _, _, _, h = n_v2.steps(obs_t[:, t:t+1], prev_t[:, t:t+1], h)
            loss = loss+mt.square().sum()
        loss.backward()
        for q in ('mem_fb.weight', 'layers.0.edge_bias.weight', 'layers.1.edge_bias.weight',
                  'layers.0.lag_b', 'edge_r.weight', 'edge_m.weight', 'edge_a'):
            g = dict(n_v2.actor.named_parameters())[q].grad
            assert g is not None and float(g.norm()) > 0 and bool(torch.isfinite(g).all()), f'{dims} {q}'


def test_seq_rel_deltanet_identity_equivariance_and_bptt():
    """r1 (multi-scale Erlang-8 SSM + symmetric Gated DeltaNet + 1-hop edge message passing + lag bias):
    exact step-0 identity on --init upgrade from g2+f1, exact T>1 vs T=1 and Room rebuild equivalence,
    exact sign equivariance under MV negation, and grad_ckpt=True BPTT across 12x5 and 48x20."""
    from aio import seqnet as SQ
    from aio.frame import canonical
    from aio.sim import core as C
    torch.manual_seed(7)
    base = SQ.SeqAIO('d32,l2,h4,m8,g2,q1,u1,f1', w=32)
    with torch.no_grad():
        base.actor.mem_fb.weight.normal_(0, .05)
        for lay in base.actor.layers:
            lay.out_t.weight.normal_(0, .05); lay.out_c.weight.normal_(0, .05)
            lay.out_l.weight.normal_(0, .05); lay.ff[-1].weight.normal_(0, .05)
        base.actor.act[-1].weight.normal_(0, .1)
    n_r1 = SQ.SeqAIO('d32,l2,h4,m8,g2,q1,u1,f1,r1', w=32)
    assert SQ.of_buffer(n_r1.pairs.seq) == ('d32,l2,h4,m8,g2,q1,u1,f1,r1', 32)
    got = n_r1.actor.load_state_dict(base.actor.state_dict(), strict=False)
    assert not got.unexpected_keys
    # 1. Exact step-0 forward identity when growing r1 onto g2+f1 weights
    o, pv = _seq_inputs(2, 10)
    base.eval(); n_r1.eval()
    with torch.no_grad():
        assert torch.allclose(n_r1.steps(o, pv)[0], base.steps(o, pv)[0], atol=1e-5)
    # 2. Exact odd sign equivariance of DeltaNet E_ij under MV negation (u, du -> -u, -du)
    with torch.no_grad():
        o_flip = o.clone()
        mv0, mv1 = 12*12, 12*12+5*6
        mv_blk = o_flip[..., mv0:mv1].unflatten(-1, (5, 6)).clone()
        mv_blk[..., 0] = -mv_blk[..., 0]
        mv_blk[..., 1] = -mv_blk[..., 1]
        o_flip[..., mv0:mv1] = mv_blk.flatten(-2)
        h1 = h2 = None
        for t in range(10):
            _, _, _, _, h1 = n_r1.steps(o[:, t:t+1], pv[:, t:t+1], h1)
            f1 = n_r1.actor.rel_feat_last.clone()
            _, _, _, _, h2 = n_r1.steps(o_flip[:, t:t+1], -pv[:, t:t+1], h2)
            f2 = n_r1.actor.rel_feat_last.clone()
        assert torch.allclose(f1[..., :18], -f2[..., :18], atol=1e-5)      # signed E channels flip exactly
        assert torch.allclose(f1[..., 18:], f2[..., 18:], atol=1e-5)       # |E| & excitation energy invariant
    # 3. T>1 vs T=1 exact equivalence, Room rebuild equivalence, and grad_ckpt=True BPTT on 12x5 & 48x20
    with torch.no_grad():
        n_r1.actor.rel_mv_out.weight.normal_(0, .05)
        n_r1.actor.rel_cv_out.weight.normal_(0, .05)
        for lay in n_r1.actor.layers:
            lay.rel_bias.weight.normal_(0, .1)
            lay.rel_lag_bias.weight.normal_(0, .05)
    for dims, nmv in (('12x5', 5), ('48x20', 20)):
        prog = C.build(2, 17, 'cpu', 'cpu', dims=dims, teacher=False)
        st = C.reset(prog)
        os_, pvs = [], []
        a = torch.zeros(2, nmv)
        for k in range(6):
            ok_, s = canonical(C.observe(prog, st, k).float())
            os_.append(ok_); pvs.append(a)
            a = (.25*torch.randn(2, nmv)).clamp(-1, 1)*prog.c['mvm'].float()
            st, _ = C.step(prog, st, (a*s).to(prog.dtype), k)
        obs_t, prev_t = torch.stack(os_, 1), torch.stack(pvs, 1)
        n_r1.eval(); n_r1.forget()
        m_seq, _, _, _, h_seq = n_r1.steps(obs_t, prev_t)
        n_r1.forget(); h = None; ms = []
        for t in range(6):
            mt, _, _, _, h = n_r1.steps(obs_t[:, t:t+1], prev_t[:, t:t+1], h)
            ms.append(mt)
            if t == 2:
                n_r1.forget()                                     # verify Room rebuild from hidden state matches exactly
        assert torch.allclose(torch.cat(ms, 1), m_seq, atol=1e-5)
        assert torch.allclose(h, h_seq, atol=1e-5)
        n_r1.actor.grad_ckpt = True
        n_r1.train(); n_r1.zero_grad(); n_r1.forget()
        h = None; loss = 0.
        for t in range(6):
            mt, _, _, _, h = n_r1.steps(obs_t[:, t:t+1], prev_t[:, t:t+1], h)
            loss = loss+mt.square().sum()
        loss.backward()
        for q in ('rel_p', 'rel_a', 'rel_eta', 'rel_c', 'layers.0.rel_bias.weight', 'layers.0.rel_lag_bias.weight',
                  'layers.1.rel_bias.weight', 'layers.1.rel_lag_bias.weight', 'rel_mv_out.weight', 'rel_cv_out.weight',
                  'rel_e_mv.weight', 'rel_c_in.weight', 'rel_e_cv.weight', 'rel_m_in.weight'):
            g = dict(n_r1.actor.named_parameters())[q].grad
            assert g is not None and float(g.norm()) > 0 and bool(torch.isfinite(g).all()), f'{dims} {q}'


def test_seq_r2_smith_gain_gate_identity_equivariance_and_bptt():
    """r2 (gain-aware multi-scale DeltaNet + 8-scale in-flight pipe Neural Smith Predictor + log-gain precision gate):
    zero extra state bytes vs r1, bit-exact step-0 identity on r1 -> r2 warm-start, exact odd/even MV negation
    equivariance, T>1 vs T=1 + Room rebuild equivalence, and grad_ckpt=True BPTT across 12x5 and 48x20."""
    from aio import seqnet as SQ, train as T
    from aio.frame import canonical
    from aio.sim import core as C
    torch.manual_seed(11)
    assert SQ.ctx_size(12, 5, 128, 64, 0, 2) == SQ.ctx_size(12, 5, 128, 64, 0, 1)   # zero extra state bytes
    n_r1 = SQ.SeqAIO('d32,l2,h4,m8,g2,q1,u1,f1,r1', w=32)
    with torch.no_grad():
        n_r1.actor.mem_fb.weight.normal_(0, .05)
        n_r1.actor.rel_mv_out.weight.normal_(0, .05)
        n_r1.actor.rel_cv_out.weight.normal_(0, .05)
        for lay in n_r1.actor.layers:
            lay.out_t.weight.normal_(0, .05); lay.out_c.weight.normal_(0, .05)
            lay.out_l.weight.normal_(0, .05); lay.ff[-1].weight.normal_(0, .05)
            lay.rel_bias.weight.normal_(0, .1); lay.rel_lag_bias.weight.normal_(0, .05)
        n_r1.actor.act[-1].weight.normal_(0, .1)
    n_r2 = SQ.SeqAIO('d32,l2,h4,m8,g2,q1,u1,f1,r2', w=32)
    assert SQ.of_buffer(n_r2.pairs.seq) == ('d32,l2,h4,m8,g2,q1,u1,f1,r2', 32)
    src = dict(n_r1.actor.state_dict())
    own = n_r2.actor.state_dict()
    for k, v in list(src.items()):
        if k in own and v.dim() == 2 and v.shape[0] == own[k].shape[0] and v.shape[1] == SQ.REL_F and own[k].shape[1] == SQ.REL_F2:
            src[k] = T.expand_rel_in(v, SQ.REL_F2)
    got = n_r2.actor.load_state_dict(src, strict=False)
    assert not got.unexpected_keys and set(got.missing_keys) == {'rel_pipe_in.weight', 'rel_pipe_in.bias', 'act_gate.weight', 'act_gate.bias'}
    # 1. Exact step-0 forward identity when upgrading r1 -> r2
    o, pv = _seq_inputs(2, 10)
    n_r1.eval(); n_r2.eval()
    with torch.no_grad():
        m1, _, _, _, h1 = n_r1.steps(o, pv)
        m2, _, _, _, h2 = n_r2.steps(o, pv)
        assert torch.allclose(m2, m1, atol=1e-5) and torch.allclose(h2, h1, atol=1e-5)
    # 2. Exact odd/even equivariance of 24-dim rf and 16-dim in-flight pipe features under MV negation
    with torch.no_grad():
        o_flip = o.clone()
        mv0, mv1 = 12*12, 12*12+5*6
        mv_blk = o_flip[..., mv0:mv1].unflatten(-1, (5, 6)).clone()
        mv_blk[..., 0] = -mv_blk[..., 0]
        mv_blk[..., 1] = -mv_blk[..., 1]
        o_flip[..., mv0:mv1] = mv_blk.flatten(-2)
        ha = hb = None
        for t in range(10):
            _, _, _, _, ha = n_r2.steps(o[:, t:t+1], pv[:, t:t+1], ha)
            f1, p1 = n_r2.actor.rel_feat_last.clone(), n_r2.actor.rel_pipe_last.clone()
            _, _, _, _, hb = n_r2.steps(o_flip[:, t:t+1], -pv[:, t:t+1], hb)
            f2, p2 = n_r2.actor.rel_feat_last.clone(), n_r2.actor.rel_pipe_last.clone()
        assert f1.shape[-1] == SQ.REL_F2 and p1.shape[-1] == SQ.REL_E
        assert torch.allclose(f1[..., :18], -f2[..., :18], atol=1e-5)       # signed DeltaNet channels flip
        assert torch.allclose(f1[..., 18:], f2[..., 18:], atol=1e-5)        # |E|, mv_act, +4 physical log-gain/activity invariant
        assert torch.allclose(p1, -p2, atol=1e-5)                           # in-flight pipe displacement & impulse wave flip
    # 3. T>1 vs T=1, Room rebuild equivalence, and grad_ckpt=True BPTT into every new r2 parameter (even from zero-init!)
    for dims, nmv in (('12x5', 5), ('48x20', 20)):
        prog = C.build(2, 19, 'cpu', 'cpu', dims=dims, teacher=False)
        st = C.reset(prog)
        os_, pvs = [], []
        a = torch.zeros(2, nmv)
        for k in range(6):
            ok_, s = canonical(C.observe(prog, st, k).float())
            os_.append(ok_); pvs.append(a)
            a = (.25*torch.randn(2, nmv)).clamp(-1, 1)*prog.c['mvm'].float()
            st, _ = C.step(prog, st, (a*s).to(prog.dtype), k)
        obs_t, prev_t = torch.stack(os_, 1), torch.stack(pvs, 1)
        # Non-zero r2 modules for T>1 vs T=1 + Room rebuild check
        with torch.no_grad():
            n_r2.actor.rel_pipe_in.weight.normal_(0, .05)
            n_r2.actor.act_gate.weight.normal_(0, .05)
        n_r2.eval(); n_r2.forget()
        m_seq, _, _, _, h_seq = n_r2.steps(obs_t, prev_t)
        n_r2.forget(); h = None; ms = []
        for t in range(6):
            mt, _, _, _, h = n_r2.steps(obs_t[:, t:t+1], prev_t[:, t:t+1], h)
            ms.append(mt)
            if t == 2:
                n_r2.forget()
        assert torch.allclose(torch.cat(ms, 1), m_seq, atol=1e-5)
        assert torch.allclose(h, h_seq, atol=1e-5)
        # Verify zero-init step-0 gradients flow cleanly (no saddle point!)
        with torch.no_grad():
            n_r2.actor.rel_pipe_in.weight.zero_(); n_r2.actor.rel_pipe_in.bias.zero_()
            n_r2.actor.act_gate.weight.zero_(); n_r2.actor.act_gate.bias.zero_()
        n_r2.actor.grad_ckpt = True
        n_r2.train(); n_r2.zero_grad(); n_r2.forget()
        h = None; loss = 0.
        for t in range(6):
            mt, _, _, _, h = n_r2.steps(obs_t[:, t:t+1], prev_t[:, t:t+1], h)
            loss = loss+mt.square().sum()
        loss.backward()
        named = dict(n_r2.actor.named_parameters())
        for q in ('rel_pipe_in.weight', 'act_gate.weight', 'act_gate.bias',
                  'layers.0.rel_bias.weight', 'layers.0.rel_lag_bias.weight', 'rel_e_mv.weight', 'rel_e_cv.weight'):
            g = named[q].grad
            assert g is not None and float(g.norm()) > 0 and bool(torch.isfinite(g).all()), f'{dims} {q}'
        for q in ('layers.0.rel_bias.weight', 'layers.0.rel_lag_bias.weight', 'rel_e_mv.weight', 'rel_e_cv.weight'):
            assert float(named[q].grad[:, SQ.REL_F:].norm()) > 0, f'{dims} {q} new log-gain cols dead'




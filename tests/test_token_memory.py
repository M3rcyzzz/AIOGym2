"""Per-channel memory and MV->CV attention (11-05): grown onto trained networks without changing them."""
import torch
from aio.interface import NCV, NMV, CF, MF
from aio.actor import Actor, History, MEMORY, STATE, NTOK, TMEM


def packets(b, T, n=4, m=3, seed=0):
    """Random public packets with n active CVs, m active MVs and a one-hot pairing per MV."""
    g = torch.Generator().manual_seed(seed)
    cv = torch.zeros(b, T, NCV, CF); mv = torch.zeros(b, T, NMV, MF); links = torch.zeros(b, T, NCV, NMV)
    cv[..., :n, :] = torch.randn(b, T, n, CF, generator=g); cv[..., :n, -1] = 1
    mv[..., :m, :] = torch.randn(b, T, m, MF, generator=g); mv[..., :m, -1] = 1
    pair = torch.randint(0, n, (m,), generator=g)
    links[..., pair, torch.arange(m)] = 1
    prev = torch.zeros(b, T, NMV); prev[..., :m] = torch.randn(b, T, m, generator=g).tanh()
    return torch.cat((cv.flatten(-2), mv.flatten(-2), links.flatten(-2)), -1), prev


def wake(net, seed=1):
    """Move the zero-initialised outputs, so the new modules change what the network computes."""
    g = torch.Generator().manual_seed(seed)
    t = (net.history if hasattr(net, 'history') else net).tokens
    with torch.no_grad():
        for layer in (t.cv_out, t.mv_out, t.att_out):
            layer.weight.copy_(torch.randn(layer.weight.shape, generator=g)*.3)
            layer.bias.copy_(torch.randn(layer.bias.shape, generator=g)*.1)
    return net


def old_history(h, obs, prev, hidden=None):
    """The computation before 11-05: tokens, pooled features, one global GRU."""
    joint, x, mask = h.channels(obs, prev)
    memory, hidden = h.gru(x, hidden)
    return joint, memory, mask, hidden


def test_checkpoint_from_before_loads_strictly_and_computes_the_same():
    torch.manual_seed(0)
    src = Actor()
    old = {k: v for k, v in src.state_dict().items() if '.tokens.' not in k}       # what a pre-11-05 checkpoint holds
    net = Actor(); net.load_state_dict(old)                                       # strict: no missing-key error
    obs, prev = packets(3, 7)
    h0 = torch.randn(1, 3, MEMORY)
    j0, m0, k0, s0 = old_history(net.history, obs, prev, h0)
    j1, m1, k1, s1 = net.history(obs, prev, h0)
    assert torch.equal(j0, j1) and torch.equal(m0, m1) and torch.equal(k0, k1)
    assert s1.shape == (1, 3, STATE) and torch.equal(s1[..., :MEMORY], s0)
    bad = {k: v for k, v in old.items() if not k.startswith('history.gru.')}
    try:
        Actor().load_state_dict(bad)
    except RuntimeError as e:
        assert 'history.gru' in str(e)
    else:
        raise AssertionError('a missing key outside the grown module must still fail')


def test_saved_and_reloaded_with_the_new_modules():
    torch.manual_seed(0); a = wake(Actor()); b = Actor(); b.load_state_dict(a.state_dict())
    obs, prev = packets(2, 5)
    for x, y in zip(a(obs, prev)[:4], b(obs, prev)[:4]):
        assert torch.equal(x, y)


def test_global_sized_start_is_a_zero_channel_state():
    torch.manual_seed(0); h = wake(History())
    obs, prev = packets(2, 6)
    hg = torch.randn(1, 2, MEMORY)
    a = h(obs, prev, hg)
    b = h(obs, prev, torch.cat((hg, torch.zeros(1, 2, NTOK*TMEM)), -1))
    for x, y in zip(a, b):
        assert torch.equal(x, y)
    assert torch.equal(h(obs, prev)[3], h(obs, prev, torch.zeros(1, 2, MEMORY))[3])


def test_state_carries_across_calls():
    torch.manual_seed(0); h = wake(History())
    obs, prev = packets(3, 9)
    j, m, k, s = h(obs, prev)
    j1, m1, _, s1 = h(obs[:, :4], prev[:, :4])
    j2, m2, _, s2 = h(obs[:, 4:], prev[:, 4:], s1)
    torch.testing.assert_close(torch.cat((j1, j2), 1), j, rtol=0, atol=1e-5)
    torch.testing.assert_close(torch.cat((m1, m2), 1), m, rtol=0, atol=1e-5)
    torch.testing.assert_close(s2, s, rtol=0, atol=1e-5)


def test_burn_in_is_the_state_after_each_rows_own_prefix():
    torch.manual_seed(0); h = wake(History())
    P = 6; obs, prev = packets(5, P, seed=3)
    lengths = torch.tensor([6, 3, 0, 1, 5])
    pre = h.prefix(obs, prev, lengths)
    assert pre.shape == (1, 5, STATE) and torch.all(pre[0, 2] == 0)
    for i, n in enumerate(lengths.tolist()):
        if n:
            s = h(obs[i:i+1, :n], prev[i:i+1, :n])[3]
            torch.testing.assert_close(pre[:, i:i+1], s, rtol=0, atol=1e-5)
    assert h.prefix(obs[:, :0], prev[:, :0], lengths*0).shape == (1, 5, STATE)
    assert torch.equal(h.prefix_steps(obs, prev, lengths), pre)


def test_channel_order_does_not_matter():
    """Any order of the CV and MV slots gives the same outputs, permuted: no channel identity."""
    torch.manual_seed(0); net = wake(Actor())
    obs, prev = packets(2, 5, n=5, m=4, seed=4)
    cvo, mvo = torch.randperm(NCV), torch.randperm(NMV)
    cv = obs[..., :NCV*CF].reshape(2, 5, NCV, CF)[..., cvo, :]
    mv = obs[..., NCV*CF:NCV*CF+NMV*MF].reshape(2, 5, NMV, MF)[..., mvo, :]
    links = obs[..., -NCV*NMV:].reshape(2, 5, NCV, NMV)[..., cvo, :][..., mvo]
    obs2 = torch.cat((cv.flatten(-2), mv.flatten(-2), links.flatten(-2)), -1)
    m1, s1 = net(obs, prev)[:2]
    m2, s2 = net(obs2, prev[..., mvo])[:2]
    torch.testing.assert_close(m2, m1[..., mvo], rtol=0, atol=2e-5)
    torch.testing.assert_close(s2, s1[..., mvo], rtol=0, atol=2e-5)


def test_inactive_channels_are_ignored():
    """What an inactive slot holds (padding) cannot reach the active outputs."""
    torch.manual_seed(0); net = wake(Actor())
    obs, prev = packets(2, 5, n=3, m=2, seed=5)
    noisy = obs.clone()
    cv = noisy[..., :NCV*CF].view(2, 5, NCV, CF); cv[..., 3:, :-1] = torch.randn(2, 5, NCV-3, CF-1)
    mv = noisy[..., NCV*CF:NCV*CF+NMV*MF].view(2, 5, NMV, MF); mv[..., 2:, :-1] = torch.randn(2, 5, NMV-2, MF-1)
    a, b = net(obs, prev)[0], net(noisy, prev)[0]
    torch.testing.assert_close(a[..., :2], b[..., :2], rtol=0, atol=1e-6)


def test_gradient_reaches_the_grown_outputs_first():
    """Zero outputs: the first gradient moves cv_out, mv_out and att_out (their inputs are not zero),
    and once they have moved the memories and the attention learn too."""
    torch.manual_seed(0); net = Actor()
    obs, prev = packets(3, 6)
    net(obs, prev)[0].square().sum().backward()
    t = net.history.tokens
    for layer in (t.cv_out, t.mv_out, t.att_out):
        assert layer.weight.grad.abs().sum() > 0
    net.zero_grad(); wake(net)
    net(obs, prev)[0].square().sum().backward()
    for p in (t.cv_gru.weight_ih_l0, t.mv_gru.weight_ih_l0, t.q.weight, t.k.weight, t.v.weight, t.link_bias):
        assert p.grad is not None and p.grad.abs().sum() > 0

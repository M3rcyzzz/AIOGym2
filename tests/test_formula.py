"""aio.formula: the law's two steps as arithmetic on the responses a network reads. E is what a plant whose
responses are straight lines between the knots does (levels and analyzers' old values included), the move is the
teacher's own design solved, a sequence is its decisions one by one, and a network with the two switches computes
exactly that, keeps the moves made in its state, loads back as it was saved and trains end to end."""
import torch

from aio import formula as FM, law as L, net as N, train as T
from aio.sim import core as C, model as MD
from aio.interface import NCV, NMV, CF, MF, SIZE
from aio.actor import STATE as BASE
from aio.frame import canonical


def packets(y, du, dtype=torch.float64):
    """Packets (b,T,SIZE) of a plant with every row and MV in service: rows y (b,T,NCV) in bands off a target of
    0, the moves du (b,T,NMV) in full travel that the packets show, MV limits at their widest."""
    b, T = y.shape[:2]
    obs = torch.zeros(b, T, SIZE, dtype=dtype)
    cv, mv = obs[..., :NCV*CF].view(b, T, NCV, CF), obs[..., NCV*CF:NCV*CF+NMV*MF].view(b, T, NMV, MF)
    cv[..., 0] = y; cv[:, 1:, :, 1] = y[:, 1:]-y[:, :-1]; cv[..., 5] = 1.; cv[..., 6] = 1.; cv[..., -1] = 1.
    mv[..., 1] = du/FM.SPAN; mv[..., 2] = .1; mv[..., -1] = 1.
    return obs


def plant(nk, b=3, T=420, seed=0):
    """A linear plant stepped by brute force: responses that are straight lines between the knots, settled by
    the last three (row 0: a level, which goes on at its last slope); moves over the last 160 decisions.
    -> the knots' values (b,NCV,NMV,nk), the rows (b,T+240,NCV), the moves (b,T+240,NMV)."""
    torch.manual_seed(seed)
    kn = MD.KNOTS[nk]
    Sk = torch.randn(b, NCV, NMV, nk).double()*torch.linspace(.2, 1, nk)
    Sk[..., -3:] = Sk[..., -4:-3]
    Sk[:, 0] = .03*torch.tensor(kn).double()*torch.randn(b, NMV, 1).double()
    full = torch.einsum('tk,baik->bait', FM.hat(torch.arange(0, T+241), kn), Sk)
    slope = (Sk[:, 0, :, -1]-Sk[:, 0, :, -2])/(kn[-1]-kn[-2])
    full[:, 0, :, kn[-1]:] = Sk[:, 0, :, -1:]+slope[..., None]*torch.arange(0, T+241-kn[-1])
    du = torch.zeros(b, T+240, NMV).double()
    du[:, T-160:T] = .01*torch.randn(b, 160, NMV).double()*(torch.rand(b, 160, NMV) < .3)
    y = torch.zeros(b, T+240, NCV).double()
    for t in range(T+240):                                         # y(t) = sum over the moves so far of S(their age), the packet's own move of age 1
        y[:, t] = torch.einsum('bais,bsi->ba', full[..., t-torch.arange(0, t+1)+1], du[:, :t+1])
    return Sk, y, du


def test_e_is_what_a_plant_of_straight_line_responses_does():
    for nk in MD.KNOTS:
        f = FM.Formula(MD.KNOTS[nk])
        Sk, y, du = plant(nk)
        T = y.shape[1]-240
        obs, S = packets(y, du)[:, :T], Sk[:, None].expand(-1, T, -1, -1, -1)
        mem, state = f.memory(obs)
        E, rate, lev = f.heading(obs, S, mem)
        assert torch.equal(lev[:, -1], torch.eye(NCV)[0].double().expand(3, -1))             # the level is told from the rows that settle
        true = -torch.stack([y[:, T-1+lag] for lag in MD.LAB_LAGS], -1)                      # no move after T: the error at every label lag if the MVs stop
        assert float(true.abs().amax()) > .1 and torch.allclose(E[:, -1], true, atol=2e-5)   # (the moves are kept in single precision)
        assert float(rate[:, -1].abs().amax()) < 1e-5                                        # nothing but the moves made it change
        h, one = None, []
        for t in range(T-30, T):                                                             # decision by decision from the state kept
            if h is None:
                h = f.memory(obs[:, :T-30])[1]
            m1, h = f.memory(obs[:, t:t+1], h)
            one.append(f.heading(obs[:, t:t+1], S[:, t:t+1], m1)[0])
        assert torch.allclose(torch.cat(one, 1), E[:, -30:], atol=1e-6) and torch.allclose(h, state, atol=1e-6)


def test_an_analyzers_old_value_is_brought_up_to_now():
    f = FM.Formula(MD.KNOTS[16])
    Sk, y, du = plant(16, seed=1)
    T, old = y.shape[1]-240, 7
    obs = packets(y, du)[:, :T]
    late = obs.clone()
    cv = late[..., :NCV*CF].view(3, T, NCV, CF)
    cv[:, old:, 1:4, 0] = y[:, :T-old, 1:4]                        # rows 1-3 report what they were seven decisions ago, and say so
    cv[:, :, 1:4, 8] = torch.log1p(torch.tensor(float(old)))
    S = Sk[:, None].expand(-1, T, -1, -1, -1)
    E = f.heading(obs, S, f.memory(obs)[0])[0]
    mem = f.memory(late)[0]
    El = f.heading(late, S, mem)[0]
    assert torch.allclose(El[:, -1, 1:], E[:, -1, 1:], atol=2e-5)
    stale = f.heading(late, S, mem[:5]+(None,))[0]                 # without it the moves of the last seven decisions are counted twice
    assert float((stale[:, -1, 1:4]-E[:, -1, 1:4]).abs().amax()) > 1e-3


def test_the_move_is_the_teachers_design_solved():
    """Against model.dmc_design and aio.law.table on the responses at every lag, for a slew limit of the full travel."""
    torch.manual_seed(1)
    for nk in MD.KNOTS:
        f, m = FM.Formula(MD.KNOTS[nk]), 4
        Sk = 3*torch.randn(m, NMV, NMV, nk).double()*torch.linspace(.2, 1, nk)
        Sk[..., -3:] = Sk[..., -4:-3]
        full = torch.einsum('tk,baik->btai', FM.hat(torch.arange(0, MD.LAGS+1), MD.KNOTS[nk]), Sk)
        w, mvm = torch.rand(m, NMV).double()+.5, torch.ones(m, NMV).double()
        mvm[0, -1] = 0.                                            # an MV out of service
        Sw, HX, _ = MD.dmc_design(full, w, mvm, torch.zeros(m, MD.LAGS+1, MD.NZMAX, NMV).double(), torch.zeros(m, MD.NZMAX).double())
        E = torch.randn(m, MD.NLAB, NMV).double()
        ref = torch.einsum('njlr,nlr->nj', L.table(Sw, HX), E)
        obs = torch.zeros(m, 1, SIZE).double()
        cv, mv = obs[..., :NCV*CF].view(m, 1, NCV, CF), obs[..., NCV*CF:NCV*CF+NMV*MF].view(m, 1, NMV, MF)
        cv[:, 0, :NMV, 5] = w**2; cv[:, 0, :NMV, 6] = 1.; cv[:, 0, :NMV, -1] = 1.
        mv[:, 0, :, 2] = 1/FM.SPAN; mv[:, 0, :, -1] = mvm
        S, Ef = torch.zeros(m, 1, NCV, NMV, nk).double(), torch.zeros(m, 1, NCV, MD.NLAB).double()
        S[:, 0, :NMV], Ef[:, 0, :NMV] = Sk*mvm[:, None, :, None], E.transpose(1, 2)
        got = f.move(obs, S, Ef, f.memory(obs)[0])[:, 0]
        assert float(ref.abs().amax()) > .01 and torch.allclose(got.double(), ref, atol=1e-6)
        Ef.requires_grad_(True)                                    # a readout that corrects E is taught through the solve
        g, = torch.autograd.grad(f.move(obs, S, Ef, f.memory(obs)[0]).sum(), Ef)
        assert float(g[:, 0, :NMV].abs().amax()) > 0 and not g[:, 0, NMV:].any()


def run(n=6, seed=21, steps=40):
    """Packets, last actions and the true responses at 16 knots of a short run under the formula told them."""
    prog, ctl = C.build(n, seed), FM.Told(16)
    st = C.reset(prog); ctl.reset(prog)
    obs, prev, lab, a = [], [], [], torch.zeros(n, NMV)
    for k in range(steps):
        raw = C.observe(prog, st, k)
        o, s = canonical(raw.float())
        obs.append(o); prev.append(a); lab.append(N.pair_labels(prog.labels(k, 16).float()*s[:, :, None]))
        u = ctl(prog, st, k, raw)
        a = u.float()*s
        st, _ = C.step(prog, st, u, k)
    return torch.stack(obs, 1), torch.stack(prev, 1), torch.stack(lab, 1)


def test_a_network_with_the_switches_computes_the_formula_and_keeps_the_moves_in_its_state(tmp_path):
    obs, prev, true = run()
    n, T = obs.shape[:2]
    torch.manual_seed(0)
    net = N.AIO(law=True, knots=16, formula=True, solve=True).eval()
    assert not any(k.startswith(('kern.', 'gain.', 'valve.')) for k in net.pairs.state_dict()) and net.pairs.stats.moves is False
    told = (true, torch.ones(n))
    with torch.no_grad():
        mean, _, mask, lab, state = net.steps(obs, prev, code=told)
        eh, kh, vh, mv = net.pairs.read_law
        assert lab.shape == (n, T, NCV, NMV, 16) and state.shape == (1, n, N.STATE+FM.SMEM) and kh is None and vh is None
        # told the true responses, with nothing learned: the module's E and move
        f = FM.Formula(MD.KNOTS[16])
        S = torch.sinh(true)*(N.unpack(obs)[0][..., -1][..., None]*N.unpack(obs)[1][..., -1][:, :, None])[..., None]
        mem, hf = f.memory(obs)
        E = f.heading(obs, S, mem)[0]
        assert torch.allclose(torch.sinh(eh), E.clamp(-200., 200.), atol=1e-3, rtol=1e-3)
        want = f.move(obs, S, E.clamp(-200., 200.), mem)
        assert float(want.abs().amax()) > .01 and torch.allclose(mv, want, atol=1e-3, rtol=1e-3)
        assert torch.equal(state[0, :, N.STATE:], hf)
        # a sequence is its decisions one by one; a state from before the formula starts its part at zero
        h, ms = torch.zeros(1, n, BASE), []
        for t in range(T):
            h = net.steps(obs[:, t:t+1], prev[:, t:t+1], h, (true[:, t:t+1], told[1]))[4]
            ms.append(net.pairs.read_law[3])
        assert torch.allclose(torch.cat(ms, 1), mv, atol=1e-4, rtol=1e-4) and torch.allclose(h, state, atol=1e-4)
        # each switch by itself, and the weights say which are on
        for kw in (dict(formula=True), dict(solve=True), dict(formula=True, solve=True), dict()):
            one = N.AIO(law=True, knots=16, **kw)
            torch.save(dict(model=dict(actor=one.actor.state_dict(), pairs=one.pairs.state_dict())), tmp_path/'net.pt')
            back, _ = N.load(tmp_path/'net.pt')
            assert (back.pairs.nk, back.pairs.worked, back.pairs.solve) == (16, bool(kw.get('formula')), bool(kw.get('solve')))
            assert torch.allclose(back.steps(obs, prev)[0], one.eval().steps(obs, prev)[0], atol=1e-6)
            assert (back.pairs.read_law[1] is None) == bool(kw.get('solve'))
    # not told, the move comes from the responses the network reads: their readout is taught through E and the solve's right-hand side
    net.train()
    net.steps(obs, prev)
    net.pairs.read_law[3].square().mean().backward()
    assert float(net.pairs.read[-1].weight.grad.abs().sum()) > 0


def test_the_trainer_gathers_and_steps_with_the_switches():
    """--law --knots 16 --formula --solve: the told episodes are told the responses at 16 knots (nothing else is
    there to tell), a window drawn from the buffer starts from the moves made before it, and a step runs."""
    from tests.test_train_guard import args
    n = 8
    a = args(noise=0., replay=1, batch=8, keep=4, fit_chunk=4, law=True, lr=1e-3, known=.5, clip=10., knots=16, formula=True, solve=True)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    net, critic = N.AIO(law=True, knots=16, formula=True, solve=True), N.Critic(2, knots=16)
    opt, copt = torch.optim.Adam(net.parameters(), lr=1e-3), torch.optim.Adam(critic.parameters(), lr=3e-4)
    lane, buf = T.Lane(C.build(n, 5), a, gen, teach=1.), T.Buffer(a.keep*n)
    for _ in range(3):
        log = T.imitate(net, critic, opt, copt, lane, buf, a, gen)
    assert buf.x['lab'].shape[-1] == 16 and buf.x['h0'].shape == (a.keep*n, N.STATE+FM.SMEM)
    assert log['gain'] is None and log['valve'] is None and log['head'] == log['head'] and log['law'] == log['law']
    x = {q: v[:3*n] for q, v in buf.x.items()}
    code = (x['lab'], torch.ones(n))
    with torch.no_grad():
        net.eval()
        net.steps(*(torch.cat([x[q][i*n:(i+1)*n] for i in range(3)], 1) for q in ('obs', 'prev')), x['h0'][None, :n], (torch.cat([x['lab'][i*n:(i+1)*n] for i in range(3)], 1), code[1]))
        whole = net.pairs.read_law[3][:, -6:]
        net.steps(x['obs'][2*n:], x['prev'][2*n:], x['h0'][None, 2*n:], (x['lab'][2*n:], code[1]))
        assert float(whole.abs().amax()) > 1e-3 and torch.allclose(net.pairs.read_law[3], whole, atol=1e-4, rtol=1e-3)
        net.steps(x['obs'][2*n:], x['prev'][2*n:], x['h0'][None, 2*n:, :N.STATE], (x['lab'][2*n:], code[1]))       # the moves before the window forgotten
        assert not torch.allclose(net.pairs.read_law[3], whole, atol=1e-4, rtol=1e-3)


def test_given_the_responses_it_works_the_law_out_and_reading_them_it_uses_its_learned_readouts(tmp_path):
    """`given`: an episode told the responses goes through the formula and the solve (what the network above does
    everywhere), one that is not through the readouts of E, K and V (what a network with the law alone does)."""
    obs, prev, true = run()
    n, nt = obs.shape[:2]
    torch.manual_seed(0)
    net = N.AIO(law=True, knots=16, given=True).eval()
    for q in (net.pairs.heading, net.pairs.gain, net.pairs.valve, net.pairs.kern):         # the learned readouts in play
        torch.nn.init.normal_(q[-1].weight, std=.05)
    with torch.no_grad():
        net.steps(obs, prev)
        e0, k0, v0, m0 = net.pairs.read_law                                               # not told
        net.steps(obs, prev, code=(true, torch.zeros(n)))
        assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(net.pairs.read_law, (e0, k0, v0, m0)))         # (sums over pairs: not bit for bit between two calls)
        net.steps(obs, prev, code=(true, torch.ones(n)))
        (e1, k1, v1, m1), ef1 = net.pairs.read_law, net.pairs.read_given                  # all told: the module's E and move
        f = FM.Formula(MD.KNOTS[16])
        S = torch.sinh(true)*(N.unpack(obs)[0][..., -1][..., None]*N.unpack(obs)[1][..., -1][:, :, None])[..., None]
        mem = f.memory(obs)[0]
        E = f.heading(obs, S, mem)[0].clamp(-200., 200.)
        assert torch.allclose(torch.sinh(ef1), E, atol=1e-3, rtol=1e-3) and torch.allclose(m1, f.move(obs, S, E, mem), atol=1e-3, rtol=1e-3)
        assert float((m1-m0).abs().amax()) > 1e-3 and k1 is not None and v1 is not None
        assert e1.shape == ef1.shape and float((e1-ef1).abs().amax()) > 1e-3             # the readouts that learn E, K and V are there to be taught all the same
        share = torch.tensor([1., 0.]*(n//2))
        state = net.steps(obs, prev, code=(true, share))[4]
        (e2, _, _, m2), ef2 = net.pairs.read_law, net.pairs.read_given                    # every other episode told
        assert torch.allclose(m2[0::2], m1[0::2], atol=1e-5) and torch.allclose(ef2[0::2], ef1[0::2], atol=1e-5) and not ef2[1::2].any()
        told = net.steps(obs[1::2], prev[1::2])                                           # the episodes not told, by themselves
        assert torch.allclose(m2[1::2], net.pairs.read_law[3], atol=1e-5) and torch.allclose(e2[1::2], net.pairs.read_law[0], atol=1e-5)
        assert torch.allclose(state[:, 1::2], told[4], atol=1e-5) and net.pairs.read_given is None
        # told E, K and V (a network with the law alone takes them) are not taken: only the responses are given
        net.steps(obs, prev, code=(true, torch.ones(n), torch.randn(n, nt, NCV, MD.NLAB), torch.randn(n, nt, NCV, NMV, MD.NLAB), torch.randn(n, nt, NMV)))
        assert torch.allclose(net.pairs.read_law[3], m1, atol=1e-6)
        torch.save(dict(model=dict(actor=net.actor.state_dict(), pairs=net.pairs.state_dict())), tmp_path/'given.pt')
        back, _ = N.load(tmp_path/'given.pt')
        assert back.pairs.given and back.pairs.nk == 16
        back.steps(obs, prev, code=(true, share))
        assert torch.allclose(back.pairs.read_law[3], m2, atol=1e-6)
        # decision by decision, some told
        h, ms = None, []
        for t in range(nt):
            h = net.steps(obs[:, t:t+1], prev[:, t:t+1], h, (true[:, t:t+1], share))[4]
            ms.append(net.pairs.read_law[3])
        assert torch.allclose(torch.cat(ms, 1), m2, atol=1e-4, rtol=1e-4)
    # the trainer: every reading is taught, the told episodes through the formula
    from tests.test_train_guard import args
    a = args(noise=0., replay=1, batch=8, keep=4, fit_chunk=4, law=True, lr=1e-3, known=.5, clip=10., knots=16, given=True)
    gen = torch.Generator().manual_seed(0)
    net, critic = N.AIO(law=True, knots=16, given=True), N.Critic(2, knots=16)
    opt, copt = torch.optim.Adam(net.parameters(), lr=1e-3), torch.optim.Adam(critic.parameters(), lr=3e-4)
    lane, buf = T.Lane(C.build(8, 5), a, gen, teach=1.), T.Buffer(a.keep*8)
    for _ in range(2):
        log = T.imitate(net, critic, opt, copt, lane, buf, a, gen)
    assert buf.x['h0'].shape == (a.keep*8, N.STATE+FM.SMEM) and all(log[q] == log[q] for q in ('head', 'gain', 'valve', 'law', 'headf', 'hbias'))
    assert float(net.pairs.adjust[-1].weight.abs().sum()) > 0                             # the formula's factor and correction have been stepped (half the episodes are told)


def test_hybrid_starts_at_the_formula():
    """--hybrid: the learned kernel's pending effects go into the heading read-out beside the formula's E; untrained, E is
    the formula's (the read-out's last layer starts at zero), and the weights say what they are."""
    from aio.interface import SIZE
    torch.manual_seed(0)
    obs = torch.randn(3, 6, SIZE)*.2
    obs[..., 11:NCV*12:12], obs[..., 6:NCV*12:12] = 1., 1.
    obs[..., NCV*12+5:NCV*12+NMV*6:6], obs[..., NCV*12+2:NCV*12+NMV*6:6] = 1., .1
    prev = torch.zeros(3, 6, NMV)
    a, b = N.AIO(law=True, knots=16, formula=True).eval(), N.AIO(law=True, knots=16, formula=True, hybrid=True).eval()
    b.load_state_dict({q: v for q, v in a.state_dict().items() if not q.startswith('pairs.heading.0')}, strict=False)
    with torch.no_grad():
        a.steps(obs, prev); b.steps(obs, prev)
    assert torch.allclose(a.pairs.read_law[0], b.pairs.read_law[0], atol=1e-6)
    sd = b.pairs.state_dict()
    assert N.hybrid_of(sd) and N.worked_of(sd) == (True, False) and not N.hybrid_of(a.pairs.state_dict())
    assert N.of_state(b.actor.state_dict(), sd).pairs.hybrid

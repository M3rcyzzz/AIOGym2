"""The step-test DMC that pays for its model (aio.sim.steptest): the estimate reads a made-up record, the exact response put
in by Program.install is the teacher's own design, the controller lives a stay beside the other references without
touching them, a test on plants where nothing happens reads their responses, and the two acceptances: a plant whose
model is too far off on the stretch of the test it was not found on, or does worse than the formula PI in a trial, stays
with the formula PI."""
import torch

from aio.sim import control as CT, core as C, model as MD, prior as PR, stay as ST, steptest as T


def _record(n=3, R=900, seed=0):
    """Two outputs, two inputs in service and one out of it, first-order responses with dead time, coloured noise."""
    g = torch.Generator().manual_seed(seed)
    x = .05*torch.where(torch.rand(n, R, 3, generator=g) < 1/40, -1., 1.).cumprod(1).double()
    x[:, :, 2] = 0
    L = 240
    j = torch.arange(1, L+1).double()
    step = lambda K, tau, dead: K*(1-torch.exp(-(j-dead).clamp_min(0)/tau))          # noqa: E731
    S = torch.zeros(n, L, 2, 3, dtype=torch.float64)
    for i in range(n):
        S[i, :, 0, 0], S[i, :, 0, 1] = step(1.+i, 8.+4*i, 2+i), step(-.5, 30., 10)
        S[i, :, 1, 0], S[i, :, 1, 1] = step(.3, 60., 0), step(2., 15., 5+2*i)
    h = torch.cat((S[:, :1], S[:, 1:]-S[:, :-1]), 1)
    xd = x-x[:, :1]
    z = torch.zeros(n, R, 2, dtype=torch.float64)
    for l in range(L):
        z[:, l:] += torch.einsum('noi,nti->nto', h[:, l], xd[:, :R-l])
    e, w = torch.zeros(n, R, 2, dtype=torch.float64), .002*torch.randn(n, R, 2, generator=g, dtype=torch.float64)
    for t in range(1, R):
        e[:, t] = .97*e[:, t-1]+w[:, t]
    return x, z+e+.001*torch.randn(n, R, 2, generator=g, dtype=torch.float64)+3., S


def test_the_estimate_reads_a_made_up_record():
    x, z, S = _record()
    ok = torch.ones(z.shape, dtype=torch.bool)
    ok[:, 100:130, 0] = False                                               # a stretch of bad values, and a spike
    z[0, 400, 1] += 5.
    kn = tuple(q for q in T.KNOTS if q <= 240)
    th, info = T.identify(x, z, ok, None, kn, T.DECAY[:-1])
    fir = T.curve(th, kn)[:, 1:]
    j = torch.arange(1, 241).double()
    cur = (info['K'][..., None]*T.shape(j, info['par'])).permute(0, 3, 1, 2)
    left = lambda est: float(((est-S)**2).sum()/(S**2).sum())               # noqa: E731
    assert left(fir) < .03 and left(cur) < .03
    assert float(th[..., 2, :].abs().max()) == 0 and float(info['K'][..., 2].abs().max()) == 0      # the input out of service has no response
    true = torch.tensor([[1., -.5], [.3, 2.]]).double()[None].repeat(3, 1, 1)
    true[:, 0, 0] += torch.arange(3)
    big = true.abs() >= 1
    for est in (fir, cur):                                                  # the pairs that matter: half-way up, and where they end (the least sure)
        assert float(((est[:, 59, :, :2]-S[:, 59, :, :2]).abs()/true.abs())[big].max()) < .15
        assert float(((est[:, -1, :, :2]-true).abs()/true.abs())[big].max()) < .3
    assert float(info['used'][0, 0]) <= 900-30 and float(info['used'][0, 1]) <= 899               # what was left out: the bad stretch, the spike
    assert float(info['used'].min()) >= 900-30-9                                                 # and nothing much else
    assert float(info['ar'][:, :, 0].min()) > .5                                                 # the noise is seen to be slow


def test_the_record_on_the_plants_own_clock():
    R = 40
    x = torch.arange(R, dtype=torch.float64)[None, :, None].repeat(2, 1, 1)
    Z = torch.stack((x[..., 0], x[..., 0]**2), -1)
    ok = torch.ones(2, R, 2, dtype=torch.bool)
    ok[:, 10, 1] = False
    phi = torch.ones(2, R)
    phi[1] = 2.                                                             # the second plant runs twice as fast as the decisions
    sparse = torch.tensor([[False, False], [False, True]])
    xm, Zm, okm = T._clock(phi, x, Z, ok, sparse)
    assert xm.shape[1] == 2*(R-1)+1
    assert torch.equal(xm[0, :R], x[0]) and torch.equal(Zm[0, :R], Z[0]) and torch.equal(okm[0, :R], ok[0]) and not bool(okm[0, R:].any())
    assert torch.allclose(xm[1, :, 0], torch.arange(2*R-1, dtype=torch.float64)/2)              # between the two decisions either side
    assert not bool(okm[1, 19:22, 0].all()) or bool(okm[1, :, 0].all())                          # (the first row has no bad sample)
    assert float(okm[1, :, 1].sum()) == R-1 and bool(okm[1, ::2, 1][[0, 1, 9, 11]].all()) and not bool(okm[1, 20, 1])     # a sample at a time: each once, at its own unit


def test_the_exact_response_installed_is_the_teachers_design():
    p = C.build(8, 5)
    kept = {q: getattr(p, q)[:, 0].clone() for q in ('Fr', 'phis', 'Sw', 'HX', 'Az', 'zs')}
    assert bool(p.trade.all())
    p.install(p.Fr[:, 0].clone(), p.phis[:, 0].clone())
    for q, v in kept.items():
        scale = float(v.abs().max())
        assert float((getattr(p, q)[:, 0]-v).abs().max()) < 1e-4*scale, q
    hx = p.HX[:, 0].diagonal(dim1=1, dim2=2).sum()
    p.install(p.Fr[:, 0].clone(), p.phis[:, 0].clone(), lam=10., trade=False)
    assert float(p.HX[:, 0].diagonal(dim1=1, dim2=2).sum()) > float(hx)     # more move suppression: a move costs more
    assert not bool(p.trade[:, 0].any()) and bool(p.trade[:, 1:].all())


def test_the_steady_state_layer_can_be_kept_from_the_targets():
    p = C.build(16, 5, menu=PR.Menu(ride=.8, exc=.3))                       # units at constrained points: the layer has something to trade
    a = CT.rollout(p, CT.Teacher('fixed'))
    p.trade[:, 0] = False
    b = CT.rollout(p, CT.Teacher('fixed'))
    p.trade[:, 0] = True
    c = CT.rollout(p, CT.Teacher('fixed'))
    assert torch.equal(a['score'], c['score']) and not torch.equal(a['score'], b['score'])


def test_the_step_test_lives_a_stay_of_its_own_beside_the_references():
    a = ST.references(4, 5, 2, lead=None)
    ctl = T.StepTest(length=300, hold=20, design='open')
    seen = {}

    def progs():
        for w, p in enumerate(ST.windows(4, 5, 2)):
            seen[w] = p, p.Fr[:, 0].clone(), p.Sw[:, 0].clone()
            yield p
    b = ST.references(4, 5, 2, lead=None, progs=progs(), extra={'steptest': ctl})
    for q in ('hold', 'pi', 'dmc', 'teacher'):
        assert torch.equal(a['score'][q], b['score'][q]), q                 # the others never see its model
    assert b['score']['steptest'].shape == (2, 4)
    for p, Fr, Sw in seen.values():
        assert torch.equal(p.Fr[:, 0], Fr) and torch.equal(p.Sw[:, 0], Sw)  # the exact model is back in its place
    assert ctl.t == 2*720 and ctl.rec is None and set(ctl.model) == {'phi', 'noise', 'fir', 'curve'}
    m = ctl.model['fir']
    assert m['Fr'].shape == seen[0][1].shape and bool(torch.isfinite(m['Fr']).all())
    r = T.ruler(ctl.w0, m['S'])
    assert 0 <= r['own_all'] and r['rise_after8'] is not None


def test_the_reading_the_suppression_and_the_trade_are_chosen_on_one_test():
    progs = list(ST.windows(4, 6, 2))
    got = T.tune(progs, T.StepTest(length=300, hold=20), lams=(1., 10.))
    assert set(got) == {(r, l, t) for r in T.READS for l in (1., 10.) for t in (True, False)} and all(v.shape == (4,) for v in got.values())
    assert not torch.equal(got['fir', 1., True], got['fir', 10., True])
    for p in progs:
        assert bool(p.trade.all())                                          # every setting gave the plants back as they were


def test_who_else_moves_the_valves_during_the_test():
    prog = C.build(12, 5)
    lev = (torch.gather(prog.c['integ'], 1, prog.pi['row']) > 0) & (prog.c['mvm'] > 0)
    assert bool(lev.any()) and bool((~lev.any(1)).any())
    end = {}
    for design in T.DESIGNS:
        ctl = T.StepTest(length=2000, hold=20, design=design)
        end[design] = CT.rollout(prog, ctl, steps=200)['st']['u']
    none = ~lev.any(1)
    assert torch.equal(end['open'][none], end['level'][none])               # without a level to hold, 'level' is 'open'
    assert not torch.equal(end['open'][lev], end['level'][lev])             # a valve whose loop holds a level stays with it
    assert not torch.equal(end['pi'], end['level'])
    ctl = T.StepTest(length=2000, hold=20, sp=.75)
    assert not torch.equal(CT.rollout(prog, ctl, steps=200)['st']['u'], end['pi'])      # the targets moved as well


def test_a_test_on_plants_where_nothing_happens_reads_their_responses():
    prog = next(iter(ST.windows(6, 5, 1, menu=ST.QUIET, quiet=True)))
    ctl = T.StepTest(length=720, design='open')
    CT.rollout(prog, ctl)
    m = ctl.find(prog)
    w = ctl.w0
    j = torch.arange(MD.LAGS+1, dtype=torch.float64)[None]
    exact = T.ruler(w, MD.lerp_at(w['Fr'], j*w['phi'][:, None]))
    assert exact['own_all'] < 1e-6 and exact['cross_all'] < 1e-6            # the ruler reads 0 on the plant's own response
    r = T.ruler(w, m['fir']['S'])
    assert r['own_all'] < .05 and r['own_to16'] < .1, r


def test_a_model_is_tried_on_a_stretch_it_was_not_found_on():
    x, z, S = _record()
    ok = torch.ones(z.shape, dtype=torch.bool)
    kn = tuple(q for q in T.KNOTS if q <= 240)
    late = (torch.arange(900) >= 600)[None].repeat(3, 1)
    th, _ = T.identify(x, z, ok & ~late[:, :, None], None, kn, T.DECAY[:-1])
    wt = torch.ones(3, 2, len(T.SPANS), dtype=torch.float64)
    far = lambda t: T._unseen(x, z, ok, t, kn, late, wt)                    # noqa: E731
    tau = torch.tensor((0,)+kn)
    true = (S[:, tau[1:]-1]-torch.cat((torch.zeros_like(S[:, :1]), S[:, tau[1:-1]-1]), 1)).permute(0, 2, 3, 1)
    assert float(far(true).max()) < .25 and float(far(th).max()) < .4       # the plant's own response, and the one found on the rest
    assert torch.allclose(far(torch.zeros_like(th)), torch.ones(3, len(T.SPANS), dtype=torch.float64))      # no model: 1
    assert float(far(-th).min()) > 1.5                                      # the wrong way round: worse than none
    assert torch.allclose(far(th), T._unseen(x, z+7., ok, th, kn, late, wt))        # where a row stands does not count


def test_the_threshold_of_the_acceptance_is_the_one_that_loses_least_on_the_tuning_batch():
    got = {('fir', 1., True): torch.tensor([10., 10., 90., 80.]), ('curve', 1., True): torch.tensor([30., 30., 30., 30.]), 'pi': torch.tensor([40., 40., 40., 40.]),
           'unseen': dict(fir=torch.tensor([.1, .2, 1.4, 3.]), curve=torch.tensor([.1, .2, .3, .4]))}
    key, th, per = T.accept(got)
    assert key == ('fir', 1., True) and th == .8 and per.tolist() == [10., 10., 40., 40.]      # of .5 to 1.25 (the same plants either side) the middle one
    got['fir', 1., True][2:] = 5.
    assert T.accept(got)[1] == float('inf')                                 # no model worth keeping out


def test_a_plant_whose_model_is_not_accepted_stays_with_the_formula_pi():
    progs = list(ST.windows(4, 6, 2))
    ctl = T.StepTest(length=300, hold=20, unseen=100)
    got = T.tune(progs, ctl, lams=(1.,), reads=('fir',), trades=(True,))
    assert got['unseen']['fir'].shape == (4,) and got['spans']['fir'].shape == (4, len(T.SPANS)) and got['pi'].shape == (4,)
    assert not torch.equal(got['pi'], got['fir', 1., True])

    def stay(accept):
        c, st, tot = T.StepTest(length=300, hold=20, read='fir', unseen=100, accept=accept), None, 0.
        for p in progs:
            r = CT.rollout(p, c, st=st)
            c.release(p)
            st, tot = r['st'], tot+r['score']
        return c, tot
    c, tot = stay(float(got['unseen']['fir'].median()))
    assert int(c.back.sum()) == 2
    for accept, back in ((float('inf'), 0), (-1., 4)):
        c2, t2 = stay(accept)
        assert int(c2.back.sum()) == back
        assert torch.equal(tot[c.back == (back > 0)], t2[c.back == (back > 0)])     # each plant as if all had gone its way


def test_after_a_trial_each_plant_keeps_the_better_of_the_formula_pi_and_the_model():
    progs, shared = list(ST.windows(4, 6, 4)), {}
    kw = dict(length=720, hold=20, read='fir', trial=720, model=shared)
    ctl = {q: T.StepTest(keep=k, **kw) for q, k in (('rule', None), ('model', 'model'), ('alone', 'pi'))}
    out = ST.references(4, 0, 4, (), progs=iter(progs), extra=ctl)['score']
    assert torch.equal(out['rule'][:3], out['model'][:3]) and torch.equal(out['rule'][:3], out['alone'][:3])      # the test and the trial: the same for the three
    rule = ctl['rule']
    b = rule.back.cpu()
    assert torch.equal(rule.back, rule.seen[1] > rule.seen[0]) and bool(ctl['alone'].back.all()) and not bool(ctl['model'].back.any())
    assert float(rule.seen[0].min()) > 0                                    # rows outside as the packet shows them, counted over the trial
    assert torch.equal(out['rule'][3][b], out['alone'][3][b]) and torch.equal(out['rule'][3][~b], out['model'][3][~b])     # each plant goes on as the trial said
    assert ctl['model'].model is shared and {'fir', 'curve', 'phi'} <= set(shared)      # one model, found once
    assert not torch.equal(out['model'][3], out['alone'][3])

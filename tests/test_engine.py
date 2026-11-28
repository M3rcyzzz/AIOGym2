"""The mechanism engine (aio.sim.expr, aio.sim.engine) and the named plants written as tables (aio.sim.plants):
every plant against the older per-family environment it replaced (equations, trajectories, steady states),
and the engine against itself (derivatives, order of the integrator, linearisation, mixed batches).

The older environments are no longer in the repository. What they gave at the points of these tests is in
tests/data/older.npz, written at the last commit that had them (tests/data/freeze_older.py there)."""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from aio.sim import plants as PL
from aio.sim.engine import Mech, Plant, GAMMA
from aio.sim.expr import exp, log, sqrt, tanh, abs_, clamp, where, minimum

F = torch.float64
PK2 = 10.25                      # second dissociation of carbonic acid in the pH benchmark (Henson & Seborg 1994)
OLD = ('vdv', 'excstr', 'bio', 'evap', 'furnace', 'gasnet', 'hex', 'column')
OLDER = np.load(Path(__file__).parent/'data'/'older.npz')


STEP = dict(vdv=(10., 1), excstr=(5., 2), bio=(300., 1), evap=(30., 1), furnace=(5., 1), gasnet=(5., 1), hex=(10., 1), column=(60., 2))


def plant(name):
    """The plant at the older environment's decision interval (the table's own is an APC interval)."""
    P = PL.PLANTS[name]()
    P.dt, P.substeps = STEP[name]
    return P


def older(key):
    """One array of what the older environments gave."""
    return torch.as_tensor(OLDER[key])


def params(name, part):
    """The older environment's parameters at the points of `part`, by name."""
    return dict(zip(OLDER[name+'/params'].tolist(), older(f'{name}/{part}/p').unbind(-1)))


def solve_ph(wa, wb, pk1, pk2=PK2, iters=52):
    """The pH at the root of the charge balance by bisection on [0, 14] (it rises with pH for Wb >= 0)."""
    lo, hi = torch.zeros_like(wa), torch.full_like(wa, 14.)
    for _ in range(iters):
        mid = .5*(lo+hi)
        up = wa+10**(mid-14)-10**(-mid)+wb*(1+2*10**(mid-pk2))/(1+10**(pk1-mid)+10**(mid-pk2)) > 0
        hi, lo = torch.where(up, mid, hi), torch.where(up, lo, mid)
    return .5*(lo+hi)


def split(P, u, p, n):
    """The older (u, p dict) as the plant's inputs w and parameters."""
    w, pp = torch.zeros(n, P.nw, dtype=F), torch.zeros(n, max(P.np, 1), dtype=F)
    w[:, [i for i, q in enumerate(P.ws) if q['kind'] == 'mv']] = u
    for i, q in enumerate(P.ws):
        if q['kind'] == 'load':
            w[:, i] = p[q['name']]
    for i, q in enumerate(P.ps):
        pp[:, i] = p[q['name']]
    return w, pp


def mvs(P):
    return [i for i, q in enumerate(P.ws) if q['kind'] == 'mv']


# ---------------------------------------------------------------- expressions
def test_derivatives_match_autograd():
    P = Plant('t', dt=1.)
    a, b = P.state('a', .7), P.state('b', 1.3)
    u = P.mv('u', 0., 1., .1)
    e = exp(-a*b)+log(a+2.)*sqrt(b)+tanh(a-b)*abs_(a-1.)+clamp(a*u, .2, .9)/(1.+b*b)+where(a-.5, a**3, b**1.7)+minimum(a, b)**2
    P.der(a, e); P.der(b, e*a-u/b)
    P.out('y', e*u+b)
    m = Mech([P], n=7, dtype=F)
    torch.manual_seed(0)
    x = (.3+1.5*torch.rand(7, 2, dtype=F)).requires_grad_(True)
    w = torch.rand(7, 1, dtype=F).requires_grad_(True)
    p = torch.zeros(7, 1, dtype=F)
    f, y, A, B, C, D = m.linearise(x.detach(), w.detach(), p)
    fa, ya = m.f(x, w, p), m.y(x, w, p)
    for i in range(2):
        gx, gw = torch.autograd.grad(fa[:, i].sum(), (x, w), retain_graph=True)
        assert torch.allclose(A[:, i], gx, atol=1e-12) and torch.allclose(B[:, i], gw, atol=1e-12)
    gx, gw = torch.autograd.grad(ya[:, 0].sum(), (x, w))
    assert torch.allclose(C[:, 0], gx, atol=1e-12) and torch.allclose(D[:, 0], gw, atol=1e-12)
    assert torch.allclose(m.fj(x.detach(), w.detach(), p)[1], A)


def test_constants_fold_and_sums_merge():
    P = Plant('t', dt=1.)
    a = P.state('a', 1.)
    e = (2*a+3)-(a+a)+0*a
    assert P.g.value(e.i) == 3.
    assert P.g.value((P.const(2.)**3).i) == 8.
    assert (a*1.).i == a.i and (a+0.).i == a.i and ((a*2.)/2.).i == a.i


# ---------------------------------------------------------------- the plants against the environments they replace
@pytest.mark.parametrize('name', OLD)
def test_equations_match_older_environment(name):
    """At 48 random points of states, moves and parameters (for excstr the inner loop's integral visits both
    saturations)."""
    x, u, p = older(name+'/eq/x'), older(name+'/eq/u'), params(name, 'eq')
    n = len(x)
    P = PL.PLANTS[name]()
    m = Mech([P], n=n, dtype=F)
    w, pp = split(P, u, p, n)
    fo, yo = older(name+'/eq/f'), older(name+'/eq/y')
    fn, yn = m.f(x, w, pp), m.y(x, w, pp)
    tol = 1e-7 if name == 'column' else 1e-11                    # the older column keeps two constants in single precision
    assert float(((fn-fo).abs()/fo.abs().amax(0, keepdim=True).clamp_min(1e-12)).max()) < tol
    assert float(((yn-yo).abs()/yo.abs().amax(0, keepdim=True).clamp_min(1e-12)).max()) < 1e-12


def test_four_tank_and_ph_match_older_environments():
    x, u, p = older('qtank/eq/x'), older('qtank/eq/u'), params('qtank', 'eq')
    n = len(x)
    P = PL.qtank(); m = Mech([P], n=n, dtype=F)
    fo, fn = older('qtank/eq/f'), m.f(x, u, torch.stack([p[q['name']] for q in P.ps], -1))
    assert float(((fn-fo).abs()/fo.abs().clamp_min(1e-9)).max()) < 1e-11
    x, u, p = older('ph/eq/x'), older('ph/eq/u'), params('ph', 'eq')
    P = PL.ph(); m = Mech([P], n=n, dtype=F)
    ph_true = older('ph/eq/ph')                                  # the older solver with the constant of the benchmark
    assert float((ph_true-solve_ph(x[:, 1], x[:, 2], p['pk1'])).abs().max()) < 1e-12
    w, pp = split(P, u, p, n)
    fn = m.f(torch.cat((x, ph_true[:, None]), 1), w, pp)
    fo = older('ph/eq/f')
    assert float(((fn[:, :3]-fo).abs()/fo.abs().clamp_min(1e-12)).max()) < 1e-11
    assert float(fn[:, 3].abs().max()) < 1e-15                   # the charge balance holds at the older solver's pH


def test_ph_is_followed_as_an_algebraic_state():
    """Along a run with both flows moving, the pH state stays at the root of the charge balance."""
    n = 16
    torch.manual_seed(1)
    P = PL.ph(); m = Mech([P], n=n, dtype=F)
    x, w, ok = m.steady(m.wnom, m.pnom)
    assert bool(ok.all()) and abs(float(x[0, 3])-7.) < .05 and abs(float(x[0, 0])-14.) < .2      # the published operating point
    worst, span, w0 = 0., torch.zeros(n, dtype=F), w.clone()
    for k in range(300):
        if k % 15 == 0:
            d = (torch.rand(n, 2, dtype=F)-.5)*2
        w[:, :2] = (w[:, :2]+d-.1*(w[:, :2]-w0[:, :2])).clamp(2., 38.)      # both flows wander around the operating point
        x = m.advance(x, w, m.pnom)
        worst = max(worst, float((x[:, 3]-solve_ph(x[:, 1], x[:, 2], m.pnom[:, 0])).abs().max()))
        span = torch.maximum(span, (x[:, 3]-7.).abs())
    assert float(x[:, 0].min()) > 3. and float(span.max()) > 2.   # the level stays in range and the pH crosses the steep part
    assert worst < .02, worst                                     # a tenth of the band
    # from an empty tank the composition flips in one decision: the state stays finite and comes back to the root
    x = m.advance(torch.tensor([[.5, -.0069, .0038, 10.67]], dtype=F).expand(n, -1), w, m.pnom)
    for _ in range(4):
        x = m.advance(x, w, m.pnom)
    assert float((x[:, 3]-solve_ph(x[:, 1], x[:, 2], m.pnom[:, 0])).abs().max()) < .02


@pytest.mark.parametrize('name', OLD)
def test_trajectories_converge_to_the_older_rk4(name):
    """The same input sequence through the engine (ROS2) and the older RK4, from the steady state at the middle
    of the targets: the difference in the outputs is small against the excursion and falls as the order says
    when the step is cut."""
    xs, ws, us, yo, lev = (older(f'{name}/run/{q}') for q in ('xs', 'ws', 'u', 'y', 'x0'))
    n, K = len(xs), len(us)
    P = plant(name)
    m = Mech([P], n=n, dtype=F)
    mv = mvs(P)
    band = torch.tensor([[P.ys[i]['band'] for i in range(P.ny)]], dtype=F)
    rows = [i for i, o in enumerate(P.ys) if o.get('track')]
    tgt = torch.tensor([[sum(P.ys[i]['target'])/2 for i in rows]], dtype=F).expand(n, -1)
    x0, w0, ok = m.steady(m.wnom, m.pnom, free=mv, rows=rows, targets=tgt, band=band[:, rows].expand(n, -1), homotopy=int(OLDER[name+'/homotopy']))
    assert bool(ok.all()) and float(((x0-xs).abs()/m.xscale).max()) < 1e-6 and float(((w0-ws).abs()/(m.whi-m.wlo)).max()) < 1e-6     # where the older run started
    dev = {}
    for S in (P.substeps, 4*P.substeps):
        xa, w = xs.clone(), ws.clone()
        worst, same = torch.zeros(n, dtype=F), torch.ones(n, dtype=torch.bool)
        for k in range(K):
            w[:, mv] = us[k]
            xa = m.advance(xa, w, m.pnom, S, guard=False)         # the integrator itself: no decision is taken again
            if name == 'evap':                                    # the same model while the separator holds liquid: the older one pumps on from an empty vessel
                same = same & (lev[k] > .1) & (xa[:, 0] > .1)
            worst = torch.where(same, torch.maximum(worst, ((m.y(xa, w, m.pnom)-yo[k]).abs()/band).amax(-1)), worst)
        dev[S] = worst
    a, b = dev[P.substeps], dev[4*P.substeps]
    if name == 'excstr':                                          # open-loop unstable: the runs through ignition are sensitive to any difference
        assert float(a.median()) < 2. and float(b.median()) < float(a.median())/4
    else:
        assert float(a.max()) < 1., float(a.max())                # within a band at the plant's own step
        assert float(b.max()) < max(float(a.max())/8, .01)        # second order (16 times at a quarter of the step), down to the older RK4's own error


def test_the_evaporator_does_not_pump_from_an_empty_separator():
    P = PL.evap(); m = Mech([P], n=1, dtype=F)
    x, w, ok = m.steady(m.wnom, m.pnom, free=[0], rows=[0], targets=torch.tensor([[1.]], dtype=F), band=torch.tensor([[.05]], dtype=F))
    w = w.clone(); w[0, 0] = 5.                                   # the product valve wide open
    low = 1.
    for _ in range(60):
        x = m.advance(x, w, m.pnom)
        low = min(low, float(x[0, 0]))
    assert bool(ok.all()) and 0. <= low < .1 and float(x[0, 0]) < .1      # it runs down to where the pump loses suction and stays there
    assert float(m.f(x, w, m.pnom)[0, 0].abs()) < 1e-3            # what leaves is what the evaporation leaves over


@pytest.mark.parametrize('name', OLD)
def test_steady_states_match_older_solver(name):
    """At the older environment's own draw of parameters and targets. (The equations are the older ones:
    test_equations_match_older_environment.)"""
    p, tgt, wo, oko = params(name, 'ss'), older(name+'/ss/tgt'), older(name+'/ss/w'), older(name+'/ss/ok')
    n = len(tgt)
    P = PL.PLANTS[name]()
    m = Mech([P], n=n, dtype=F)
    w, pp = split(P, m.wnom[:, mvs(P)], p, n)
    rows = [i for i, o in enumerate(P.ys) if o.get('track')]
    band = torch.tensor([[P.ys[i]['band'] for i in rows]], dtype=F).expand(n, -1)
    big = torch.full((n, len(mvs(P))), 1e9, dtype=F)
    x, ws, ok = m.steady(w, pp, free=mvs(P), rows=rows, targets=tgt[:, rows], band=band, homotopy=int(OLDER[name+'/homotopy']), wlo=-big, whi=big)
    assert int(ok.sum()) >= int(oko.sum()) and int(ok.sum()) >= 3
    u = ws[:, mvs(P)]
    assert float((m.f(x, ws, pp).abs()/m.xscale)[ok].max()) < 1e-6            # a steady state
    assert float(((m.y(x, ws, pp)[:, rows]-tgt[:, rows]).abs()/band)[ok].max()) < 1e-6
    same = ok & oko & (((u-wo[:, P.nx:]).abs()/(m.whi-m.wlo)[:, mvs(P)]).amax(-1) < 1e-5)      # the same branch where there are several (vdv)
    assert float(same.sum()) >= .8*float((ok & oko).sum())
    assert float(((x-wo[:, :P.nx]).abs()/m.xscale)[same].max()) < 1e-5


# ---------------------------------------------------------------- the engine against itself
def test_ros2_is_second_order_and_matches_its_linear_map():
    P = PL.vdv(); n = 4
    m = Mech([P], n=n, dtype=F)
    x0, w, ok = m.steady(m.wnom, m.pnom)
    w = w.clone(); w[:, 0] += 3.; w[:, 1] -= 500.

    def run(S):
        x = x0.clone()
        for _ in range(20):
            x = m.advance(x, w, m.pnom, S, guard=False)
        return x
    ref = run(128)
    e = [float((run(S)-ref).abs().max()) for S in (1, 2, 4, 8)]
    for a, b in zip(e, e[1:]):
        assert 3.2 < a/b < 5., e
    # a linear plant: one decision is exactly the map of `discrete`, for any step size (stiff included)
    L = Plant('lin', dt=7., substeps=3)
    a, b = L.state('a', 0.), L.state('b', 0.)
    z = L.state('z', 0., algebraic=True)
    u = L.mv('u', -1., 1., .1, 0.)
    L.der(a, -.3*a+2.*b+u); L.der(b, -40.*b+z); L.der(z, a+u-2.*z)
    L.out('y', a+z)
    ml = Mech([L], n=1, dtype=F)
    _, _, A, B, C, D = ml.linearise(ml.x0, ml.wnom, ml.pnom)
    Phi, Gam = ml.discrete(A, B)
    x, w1 = torch.tensor([[.4, -.2, .3]], dtype=F), torch.tensor([[.7]], dtype=F)
    assert torch.allclose(ml.advance(x, w1, ml.pnom), (Phi@x[..., None])[..., 0]+(Gam@w1[..., None])[..., 0], atol=1e-13)
    assert abs(float(ml.advance(x, w1, ml.pnom)[0, 0]+w1[0, 0]-2*ml.advance(x, w1, ml.pnom)[0, 2])) < 1e-12      # the algebraic equation holds after a step
    assert abs(GAMMA-(1+1/math.sqrt(2))) < 1e-15


@pytest.mark.parametrize('name', ['gasnet', 'column', 'furnace'])
def test_step_responses_are_those_of_the_plant(name):
    P = PL.PLANTS[name](); n = 2
    m = Mech([P], n=n, dtype=F)
    mv = mvs(P)
    rows = [i for i, o in enumerate(P.ys) if o.get('track')]
    band = torch.tensor([[P.ys[i]['band'] for i in rows]], dtype=F).expand(n, -1)
    tgt = torch.tensor([[sum(P.ys[i]['target'])/2 for i in rows]], dtype=F).expand(n, -1)
    x, w, ok = m.steady(m.wnom, m.pnom, free=mv, rows=rows, targets=tgt, band=band)
    assert bool(ok.all())
    L = 25
    S = m.responses(x, w, m.pnom, L)
    span = (m.whi-m.wlo).clamp_min(1e-3)
    for j in range(P.nw):
        eps = 1e-4*span[:, j]
        w1 = w.clone(); w1[:, j] += eps
        xa, xb = x.clone(), x.clone()
        for _ in range(L):
            xa, xb = m.advance(xa, w1, m.pnom), m.advance(xb, w, m.pnom)
        num = (m.y(xa, w1, m.pnom)-m.y(xb, w, m.pnom))/eps[:, None]
        assert float(((num-S[:, L, :, j]).abs()/S[:, L].abs().amax(-1).clamp_min(1e-9)).max()) < 5e-3


def test_plants_of_different_structure_share_a_batch():
    names = ['vdv', 'column', 'gasnet', 'ph']
    plants = [PL.PLANTS[q]() for q in names]
    pid = torch.tensor([0, 1, 2, 3, 1, 0, 3, 2])
    m = Mech(plants, pid=pid, dtype=F)
    torch.manual_seed(0)
    x = m.x0*(1+.05*torch.rand(8, m.nx, dtype=F))
    f, J = m.fj(x, m.wnom, m.pnom)
    y = m.y(x, m.wnom, m.pnom)
    x1 = m.advance(x, m.wnom, m.pnom, guard=False)                # the step itself: the accuracy guard takes a state 5 % off the column's profile again
    for i, k in enumerate(pid.tolist()):
        P = plants[k]
        one = Mech([P], n=1, dtype=F)
        xi, wi, pi = x[i:i+1, :P.nx], m.wnom[i:i+1, :P.nw], m.pnom[i:i+1, :max(P.np, 1)]
        fi, Ji = one.fj(xi, wi, pi)
        assert torch.allclose(f[i, :P.nx], fi[0], atol=1e-12) and torch.allclose(J[i, :P.nx, :P.nx], Ji[0], atol=1e-12)
        assert float(f[i, P.nx:].abs().max() if P.nx < m.nx else 0.) == 0.
        assert torch.allclose(y[i, :P.ny], one.y(xi, wi, pi)[0], atol=1e-12)
        assert torch.allclose(x1[i, :P.nx], one.advance(xi, wi, pi, m.substeps)[0], atol=1e-10)


def test_single_precision_and_gradient():
    """float32 follows float64 on the stiffest plant here, and the backward pass (Jacobian held constant)
    gives the sensitivity of a run to its inputs."""
    P = PL.column(); n = 3
    m64, m32 = Mech([P], n=n, dtype=F), Mech([P], n=n, dtype=torch.float32)
    x, w = m64.x0.clone(), m64.wnom.clone()
    for _ in range(60):
        x = m64.advance(x, w, m64.pnom)
    xa, xb = x.clone(), x.float()
    w2 = w.clone(); w2[:, 0] += .02; w2[:, 1] -= .01
    for _ in range(40):
        xa, xb = m64.advance(xa, w2, m64.pnom), m32.advance(xb, w2.float(), m32.pnom.float() if False else m64.pnom.float())
    ya, yb = m64.y(xa, w2, m64.pnom), m32.y(xb, w2.float(), m64.pnom.float())
    assert float((ya-yb.double()).abs()[:, :2].max()) < .01        # a fifth of the band of the purities
    V = PL.vdv(); mv_ = Mech([V], n=1, dtype=F)
    x0, w0, _ = mv_.steady(mv_.wnom, mv_.pnom)

    def loss(d):
        x, tot = x0.clone(), 0.
        for k in range(15):
            wk = w0.clone(); wk[:, 0] = w0[:, 0]+d[k]
            x = mv_.advance(x, wk, mv_.pnom)
            tot = tot+mv_.y(x, wk, mv_.pnom)[0, 0]
        return tot
    d = torch.zeros(15, dtype=F, requires_grad=True)
    g, = torch.autograd.grad(loss(d), d)
    num = torch.zeros(15, dtype=F)
    for k in range(15):
        e = torch.zeros(15, dtype=F); e[k] = 1e-5
        num[k] = (loss(e)-loss(-e))/2e-5
    assert float((g-num).abs().max()/num.abs().max()) < .05, (g, num)


def test_the_guard_refines_each_episode_only_as_far_as_its_own_estimates_ask():
    """An episode the guard takes again ends at the first refinement (4, 16 or 64 times the substeps, a new Jacobian at
    each) that both estimates find right: Richardson's (its difference to the one before, over 15) within a band, its
    own within two; the others of the batch do not go further with it."""
    from aio.sim import engine as E
    P = PL.PLANTS['excstr']()
    n = 96
    m = Mech([P], n=n, dtype=F)
    g = torch.Generator().manual_seed(1)
    mv = torch.tensor([i for i, q in enumerate(P.ws) if q['kind'] == 'mv'])
    lo, span = m.wlo[:, mv], (m.whi-m.wlo)[:, mv]
    x, w, p = m.x0.clone(), m.wnom.clone(), m.pnom
    for _ in range(6):                                                # valves wandering inside their limits, then thrown anywhere in one decision
        w[:, mv] = torch.minimum(torch.maximum(w[:, mv]+.1*span*(2*torch.rand(n, len(mv), generator=g, dtype=F)-1), lo), lo+span)
        x = m.advance(x, w, p, 4*P.substeps, refresh=True)
    w[:, mv] = lo+span*torch.rand(n, len(mv), generator=g, dtype=F)
    guarded = m.advance(x, w, p)
    lv = [m._run(x, w, p, k*P.substeps, k > 1 or m.refresh) for k in (1, 4, 16, 64)]
    atol, band = m.atol()
    rich = lambda a, b: torch.maximum(((a-b).abs()/atol).nan_to_num(math.inf).amax(-1), ((m.y(a, w, p)-m.y(b, w, p)).abs()/band).nan_to_num(math.inf).amax(-1))/15     # noqa: E731
    want = torch.full((n,), 3)
    for j in (2, 1):
        ok = (rich(lv[j][0], lv[j-1][0]) <= E.ERR_TOL) & (lv[j][1] <= E.RICH_OWN*E.ERR_TOL)
        want = torch.where(ok, torch.full_like(want, j), want)
    want = torch.where(lv[0][1] > E.ERR_TOL, want, torch.zeros_like(want))
    got = torch.stack([(guarded-v).abs().amax(-1) <= 1e-9*(1+v.abs().amax(-1)) for v, _ in lv], -1)
    assert bool(got[torch.arange(n), want].all())
    bc = want.bincount(minlength=4)
    assert int((bc[1:] > 5).sum()) >= 2, bc.tolist()                # the episodes end at different refinements


def test_the_accuracy_guard_takes_again_what_the_old_jacobian_and_the_output_curve_hide():
    """Valves thrown to any position in one decision. The bioreactor: a fast mode collapses inside the decision
    to a balance the decision's first Jacobian misplaces (the method's own estimate, filtered of stiff parts,
    sees none of it). Column A: a log purity far from its nominal value answers many times more than there."""
    from aio.sim import engine as E
    for name, n, plain_min in (('bio', 192, 3.), ('column', 32, 2.)):
        P = PL.PLANTS[name]()
        m = Mech([P], n=n, dtype=F)
        g = torch.Generator().manual_seed(1)
        mv = torch.tensor([i for i, q in enumerate(P.ws) if q['kind'] == 'mv'])
        lo, span = m.wlo[:, mv], (m.whi-m.wlo)[:, mv]
        x, w, p = m.x0.clone(), m.wnom.clone(), m.pnom
        for _ in range(6):                                            # states a controller could have left: valves wandering inside their limits
            w[:, mv] = torch.minimum(torch.maximum(w[:, mv]+.1*span*(2*torch.rand(n, len(mv), generator=g, dtype=F)-1), lo), lo+span)
            x = m.advance(x, w, p, 4*P.substeps, refresh=True)
        w[:, mv] = lo+span*torch.rand(n, len(mv), generator=g, dtype=F)
        scored = [i for i, o in enumerate(P.ys) if o.get('track') or 'lo' in o or 'hi' in o]
        band = torch.tensor([P.ys[i]['band'] for i in scored], dtype=F)
        ref = m.y(m.advance(x, w, p, substeps=64*P.substeps, refresh=True), w, p)[:, scored]
        off = lambda x1: ((m.y(x1, w, p)[:, scored]-ref).abs()/band).amax(-1)      # noqa: E731
        E.STATS = []
        plain, guarded = off(m.advance(x, w, p, guard=False)), off(m.advance(x, w, p))
        est, E.STATS = E.STATS[1], None
        assert float(plain.max()) > plain_min, (name, float(plain.max()))          # the case is one
        assert float(guarded.max()) < 1.5 and int((guarded > 1).sum()) <= 1, (name, float(guarded.max()), int((guarded > 1).sum()))
        assert int(((plain > 1) & (est <= E.ERR_TOL)).sum()) <= 1, name            # what the estimate lets through is within a band
    m = Mech([PL.PLANTS['bio']()], n=1, dtype=F)
    assert float(m._run(m.x0, m.wnom, m.pnom, m.substeps, m.refresh)[1]) < 1e-2     # next to nothing to estimate at the nominal point

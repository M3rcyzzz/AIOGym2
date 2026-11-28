"""The simulation core (aio.sim): generator, plant step, analytic model, teacher, gradients."""
import itertools
import math

import numpy as np
import pytest
import torch

from aio import interface as IF
from aio.interface import NCV, NMV
from aio.sim import core as C, control as CT, prior as PR

F = torch.float64


def test_every_operating_point_is_operable_and_roots_keep_their_prior():
    r = PR.Rng(1)
    n = 768
    segs, tgts, infos, seg_k, kinds, delay, ramp = PR.draw_episode(r, n)
    for p, t, info in zip(segs, tgts, infos):
        assert bool(PR.point_ok(p, t).all())
        free = (p['mvm'] > 0) & ~PR.constrained(p, t)[:, None]         # away from a constrained point every valve keeps its margin
        un = info['un']
        assert bool((((un >= p['ulo']+PR.U_MARGIN-1e-9) & (un <= p['uhi']-PR.U_MARGIN+1e-9)) | ~free).all())
        assert bool((((un >= p['ulo']-1e-9) & (un <= p['uhi']+1e-9)) | (p['mvm'] == 0)).all())
    nm = torch.stack([(segs[0]['nm'] == i).double().mean() for i in range(1, 6)])
    assert float((nm-torch.tensor(PR.NM_P)).abs().max()) < .06        # rejection does not thin out the large units
    arch = torch.stack([(segs[0]['arch'] == i).double().mean() for i in range(6)])
    assert float((arch-torch.tensor(PR.ARCH_P)).abs().max()) < .06
    exists, accepted = (seg_k < PR.BIG)[:, 1:], (kinds >= 0)[:, 1:]
    assert float(accepted.sum()/exists.sum()) > .95                    # events are operable by construction
    assert float(accepted.sum(1).double().mean()) > 6
    exc = PR.is_excursion(kinds)
    assert .1 < float(exc.any(1).double().mean()) < .5                 # constrained points are a share of the episodes, not the rule
    later = torch.cat((kinds[:, 1:], torch.full_like(kinds[:, :1], -1)), 1)
    assert bool(((later == PR.RETURN) | (later == -1))[exc].all())      # the next event brings the unit back
    fat = segs[0]['spr'].sum(-1) > 0
    assert abs(float(fat[segs[0]['nm'] > 1].double().mean())-PR.FAT_P) < .06 and not bool(fat[segs[0]['nm'] == 1].any())


def test_an_episode_draws_from_its_own_stream():
    """What an episode draws depends on the seed, its number and its own history: not on how many episodes
    there are, nor on which others are drawn again."""
    a, b = PR.Rng(3), PR.Rng(3)
    assert torch.equal(a.u(8, 5), b.u(32, 5)[:8])
    idx, keep = torch.tensor([2, 5]), torch.tensor([0, 1, 3, 4, 6, 7])
    ya = a.sub(idx).u(2, 3)                                       # two episodes of a drawn again
    za, zb = a.u(8, 4), b.u(32, 4)
    assert torch.equal(za[keep], zb[keep]) and not torch.equal(za[idx], zb[idx])
    c = PR.Rng(3); c.u(8, 5)
    cand = c.sub(idx, repeat=4).u(8, 3)                           # four candidates of each: different numbers, one move of the counter
    assert torch.equal(cand[0], ya[0]) and torch.equal(cand[4], ya[1]) and not torch.equal(cand[0], cand[1]) and torch.equal(c.u(8, 4), za)
    d = PR.Rng(3); d.u(8, 5)
    with d.only(torch.zeros(8, dtype=torch.bool).index_fill(0, idx, True)):
        w = d.u(8, 3)
    assert torch.equal(w[idx], ya) and torch.equal(d.u(8, 4), za)                  # the others stood still
    assert torch.equal(PR.Rng(3).fork().u(4, 2), PR.Rng(3).fork().u(4, 2)) and not torch.equal(PR.Rng(3).fork().u(4, 2), PR.Rng(3).u(4, 2))
    with pytest.raises(AssertionError):
        a.u(5, 2)                                                 # eight episodes: a draw for five is a mistake
    u = PR.Rng(7).u(20000, 16)
    hist = torch.histc(u.flatten(), 32, 0, 1)
    assert abs(float(u.mean())-.5) < 2e-3 and float(((hist-u.numel()/32)**2/(u.numel()/32)).sum()/31) < 2.
    assert abs(float(torch.corrcoef(torch.stack((u[:-1, 0], u[1:, 0])))[0, 1])) < .02 and abs(float(torch.corrcoef(u[:, :2].T)[0, 1])) < .02
    g = PR.Rng(7).normal(20000, 4)
    assert abs(float(g.mean())) < .01 and abs(float(g.std())-1) < .01
    # the whole generator: the first episodes of a larger batch are the episodes of a smaller one
    small, large = PR.draw_episode(PR.Rng(5), 48), PR.draw_episode(PR.Rng(5), 160)
    assert torch.equal(small[3], large[3][:48]) and torch.equal(small[4], large[4][:48])
    assert all(torch.equal(x[q], y[q][:48]) for x, y in zip(small[0], large[0]) for q in ('K', 'ulo', 'uhi', 'lo', 'hi', 'dlev', 'rtgt'))


def test_pairing_is_the_best_assignment_and_the_box_solve_is_exact():
    """Against every permutation and every active set at the size where they can be listed, and at a size where
    they cannot against what a minimiser has to satisfy."""
    g = torch.Generator().manual_seed(0)
    m, n = 4000, 5
    B = torch.randn(m, n, n, generator=g, dtype=F)*(torch.rand(m, n, n, generator=g) > .3)+2*torch.eye(n, dtype=F)*(torch.rand(m, 1, 1, generator=g) < .5)
    mvm = (torch.arange(n)[None] < torch.randint(1, n+1, (m,), generator=g)[:, None]).to(F)
    act = mvm > 0
    Bm = torch.where(act[:, :, None] & act[:, None, :], B, torch.eye(n, dtype=F))
    Bi, bad = torch.linalg.inv_ex(Bm)
    lam = Bm*Bi.transpose(1, 2)
    perms = torch.tensor(list(itertools.permutations(range(n))))
    lp = lam[:, perms, torch.arange(n)]                           # relative gain of MV j with row perms[q, j]
    score = torch.where(act[:, None, :], lp.clamp_min(1e-3).log().abs()+100.*(lp <= 0), torch.zeros_like(lp)).sum(-1)
    keep = ((perms[None] == torch.arange(n)) | act[:, None, :]).all(-1)
    score = torch.where(keep, score, torch.full_like(score, math.inf))
    order = PR.best_pairing(B, mvm)
    mine = torch.where(act, torch.gather(lam, 1, order[:, None, :])[:, 0].clamp_min(1e-3).log().abs()+100.*(torch.gather(lam, 1, order[:, None, :])[:, 0] <= 0),
                       torch.zeros(m, n, dtype=F)).sum(-1)
    good = bad == 0
    assert bool((order.sort(1).values == torch.arange(n)).all()) and bool(((order == torch.arange(n)) | act).all())
    assert float((mine-score.amin(1))[good].abs().max()) < n*2/PR.PAIR_Q            # the costs are compared in whole units of 1/PAIR_Q a pair
    other = PR.best_pairing(B, mvm, torch.ones(m, dtype=torch.bool), torch.rand(m, n, n, generator=g, dtype=F))
    lo_ = torch.gather(lam, 1, other[:, None, :])[:, 0]
    moved = (other != order).any(1) & good
    assert float(moved.double().mean()) > .1 and bool((((lo_ > .05) | ~act).all(1))[moved].all())     # another pairing, every relative gain above 0.05
    C = torch.rand(64, 24, 24, generator=g, dtype=F)              # a size no table reaches: no swap of two columns lowers the total
    rows = PR.assign(C)
    tot = torch.gather(C, 1, rows[:, None, :]).sum((1, 2))
    assert bool((rows.sort(1).values == torch.arange(24)).all())
    for a in range(24):
        for b in range(a+1, 24):
            sw = rows.clone(); sw[:, a], sw[:, b] = rows[:, b], rows[:, a]
            assert bool((torch.gather(C, 1, sw[:, None, :]).sum((1, 2)) >= tot-1e-12).all())

    def problems(m, n, k):
        J = torch.randn(m, k, n, generator=g, dtype=F)*(torch.rand(m, k, n, generator=g) > .4)*10**(4*torch.rand(m, k, 1, generator=g, dtype=F)-2)
        mv = (torch.arange(n)[None] < torch.randint(1, n+1, (m,), generator=g)[:, None]).to(F)
        J = J*mv[:, None, :]
        H = J.transpose(1, 2)@J+torch.diag_embed(1-mv)+1e-12*torch.eye(n, dtype=F)
        q = torch.randn(m, n, generator=g, dtype=F)*10**(5*torch.rand(m, 1, generator=g, dtype=F)-2)*mv
        lo = .4*torch.rand(m, n, generator=g, dtype=F)*mv
        return H, q, lo, (lo+.6*torch.rand(m, n, generator=g, dtype=F))*mv
    H, q, lo, hi = problems(3000, 5, 7)
    x, ok = PR._box_qp(H, q, lo, hi)
    cost = lambda z: (z*(.5*(H[:, None]@z[..., None])[..., 0]+q[:, None])).sum(-1)       # noqa: E731
    pat = torch.tensor(list(itertools.product(range(3), repeat=5)))[None]              # every variable free, at its low or at its high limit
    free = pat == 0
    xb = torch.where(pat == 1, lo[:, None], torch.where(pat == 2, hi[:, None], torch.zeros_like(lo[:, None])))
    A = torch.where(free[..., :, None] & free[..., None, :], H[:, None], torch.eye(5, dtype=F)*(~free)[..., None])
    z = torch.linalg.solve_ex(A, torch.where(free, -(q[:, None]+(H[:, None]@xb[..., None])[..., 0]), xb)[..., None])[0][..., 0]
    inside = ((z >= lo[:, None]-1e-12) & (z <= hi[:, None]+1e-12)).all(-1) & torch.isfinite(z).all(-1)
    best = torch.where(inside, cost(z), torch.full_like(inside, math.inf, dtype=F)).amin(1)      # the least cost over the vertices of every active set that are inside
    mine = cost(x[:, None])[:, 0]
    assert bool(ok.all()) and float(((mine-best)/(best.abs()+1)).max()) < 1e-9
    H, q, lo, hi = problems(500, 30, 40)                          # thirty variables: the first-order conditions
    x, ok = PR._box_qp(H, q, lo, hi)
    grad = q+(H@x[..., None])[..., 0]
    tol = 1e-9*(q.abs()+(H.abs()@x.abs()[..., None])[..., 0])+1e-9
    assert bool(ok.all()) and bool(((x >= lo) & (x <= hi)).all())
    assert bool(((grad.abs() <= tol) | ((x <= lo) & (grad >= -tol)) | ((x >= hi) & (grad <= tol))).all())


def test_what_an_episode_is_does_not_depend_on_the_last_bits_of_a_solve(monkeypatch):
    """Two machines round differently, and the episodes have to be the same on both. Equally good pairings are
    settled in whole numbers (a separation unit with three or more MVs and negative diagonal relative gains has
    all its off-diagonal relative gains equal in exact arithmetic: the inverse's last bits used to choose, and
    1 % of the episodes were another plant on the training machine), and a count of decisions that is a whole
    number stays that number. Checked on the pairing alone and on a whole batch with noise on every solve."""
    g = torch.Generator().manual_seed(0)
    m, n = 600, 5
    mvm = (torch.arange(n)[None] < torch.randint(3, n+1, (m,), generator=g)[:, None]).to(F)
    g1 = torch.sign(torch.randn(m, n, generator=g, dtype=F))*10**(.6*torch.rand(m, n, generator=g, dtype=F)-.3)
    g2 = 10**(.6*torch.rand(m, n, generator=g, dtype=F)-.3)
    B = g1[:, :, None]*g2[:, None, :]*(1-torch.eye(n, dtype=F)*(.03+.57*torch.rand(m, 1, 1, generator=g, dtype=F)))
    noise = lambda x: x*(1+1e-13*(2*torch.rand(x.shape, generator=g, dtype=x.dtype)-1))      # noqa: E731
    order = PR.best_pairing(B, mvm)
    assert bool(((order != torch.arange(n)) | (mvm == 0)).all())                   # no MV keeps its own row: every such pairing costs the same
    assert all(torch.equal(PR.best_pairing(noise(B), mvm), order) for _ in range(6))
    whole = torch.randint(0, 50, (200, 9, 9), generator=g)                         # whole costs, many ties: exact, and as good as any
    rows = PR.assign(whole)
    tot = lambda c, r: torch.gather(c, 1, r[:, None, :]).sum((1, 2))               # noqa: E731
    assert rows.dtype == torch.long and bool((rows.sort(1).values == torch.arange(9)).all())
    assert torch.equal(tot(whole, rows), tot(whole, PR.assign(whole.to(F)+1e-7*torch.rand(200, 9, 9, generator=g, dtype=F))))
    x = torch.tensor([45., 45+1e-12, 45-1e-12, 45.2, 44.8], dtype=F)
    assert PR.up(x).tolist() == [45., 45., 45., 46., 45.] and PR.down(x).tolist() == [45., 45., 45., 45., 44.]

    def batch():
        p = C.build(64, 7, teacher=False, dtype=F)
        return p, {q: v for q, v in list(vars(p).items())+list(p.c.items()) if torch.is_tensor(v) and not v.is_floating_point()}
    p0, plain = batch()
    L = torch.linalg
    for name in ('inv', 'solve', 'svdvals'):
        monkeypatch.setattr(L, name, lambda *a, _f=getattr(L, name), **k: noise(_f(*a, **k)))
    for name in ('inv_ex', 'solve_ex'):
        monkeypatch.setattr(L, name, lambda *a, _f=getattr(L, name), **k: (lambda r: (noise(r[0]), r[1]))(_f(*a, **k)))
    p1, other = batch()
    assert len(plain) > 8 and all(torch.equal(plain[q], other[q]) for q in plain), [q for q in plain if not torch.equal(plain[q], other[q])]
    assert float((p0.P-p1.P).abs().max()) < 1e-6 and not torch.equal(p0.P, p1.P)   # the same plants, to the solves' rounding


def test_reachable_optimum_is_the_best_steady_point_inside_the_limits():
    r = PR.Rng(11)
    segs, tgts, infos, seg_k, kinds, delay, ramp = PR.draw_episode(r, 1024)
    found = 0
    g = torch.Generator().manual_seed(0)
    for p, t, info in zip(segs, tgts, infos):
        sel = PR.constrained(p, t) & ((p['integ']*p['trk']).sum(-1) == 0)     # levels have to stay in balance: left out here
        if not bool(sel.any()):
            continue
        q = {k: v[sel] for k, v in p.items()}
        tq, ue = t[sel], info['ue'][sel]
        found += int(sel.sum())
        lo, hi = q['ulo']**q['gam']*q['mvm'], q['uhi']**q['gam']*q['mvm']

        def cost(x):                                                   # the steady cost as the simulator charges it, resting values last
            v = (q['K'][:, :, :PR.NMV]*x[:, None, :]).sum(-1)+(q['K'][:, :, PR.NMV:]*q['dlev'][:, None, :]).sum(-1)+q['bias']
            y = PR.h(v, q['wa'])
            viol = ((q['lo']-y).clamp_min(0)+(y-q['hi']).clamp_min(0))/q['bandf']*q['zon']
            return (PR.steady_weight(q)*((y-tq)/q['bandf'])**2*q['trk']).sum(-1)+10*(viol**2).sum(-1)

        best = cost(ue)
        assert float((((PR.h(info['v'][sel], q['wa'])-q['rtgt']).abs()/q['bandf'])*q['trk']).max()) < 1e-6
        for scale in (1e-4, 1e-2, .3):
            for _ in range(8):
                x = torch.minimum(torch.maximum(ue+scale*(hi-lo)*torch.randn(ue.shape, generator=g, dtype=ue.dtype), lo), hi)
                assert bool((cost(x) >= best*(1-1e-9)-1e-12).all())
    assert found > 50


def test_levels_stay_in_balance_at_the_best_steady_point():
    """A level in balance ranks above every CV (a row of the solve, prior.EQ_SOFT): the box solve with such rows
    against every active set at 40 digits, the best steady point of constrained plants with levels against
    Newton on the true cost at 40 digits, and the part of a load the free valves cannot take out against the
    same trade-off at 40 digits. The weight of the rows does not cost the digits of the rest: the same point
    alone, in its batch and with gains a rounding apart is the same to a millionth of a band."""
    mp = pytest.importorskip('mpmath')
    mp.mp.dps = 40
    f = lambda a: mp.mpf(float(a))                                 # noqa: E731
    col = lambda A, js: mp.matrix([[A[a, j] for j in js] for a in range(A.rows)])     # noqa: E731
    g = torch.Generator().manual_seed(3)
    m, n, nq = 12, 4, 2
    J = torch.randn(m, 6, n, generator=g, dtype=F)*10**(2*torch.rand(m, 6, 1, generator=g, dtype=F)-1)
    H, gq = J.transpose(1, 2)@J+1e-12*torch.eye(n, dtype=F), 10*torch.randn(m, n, generator=g, dtype=F)
    E = 30*torch.randn(m, nq, n, generator=g, dtype=F)*(torch.rand(m, nq, n, generator=g) > .3)
    lo = .4*torch.rand(m, n, generator=g, dtype=F)
    hi = lo+.6*torch.rand(m, n, generator=g, dtype=F)
    e = (E@(lo+(hi-lo)*torch.rand(m, n, generator=g, dtype=F))[..., None])[..., 0]
    e[m//2:] += 20*torch.randn(m-m//2, nq, generator=g, dtype=F)   # half of them: rows the box leaves no way to hold
    x, ok, mu = PR._box_qp(H, gq, lo, hi, None, E, e)
    assert bool(ok.all()) and bool(((x >= lo) & (x <= hi)).all())
    assert float(((E@x[..., None])[..., 0]-e-PR.EQ_SOFT*mu).abs().max()) < 1e-9
    w = 1/mp.mpf(str(PR.EQ_SOFT))
    for i in range(m):
        Hm = mp.matrix(n, n); Em = mp.matrix(nq, n)
        for a in range(n):
            for b in range(n):
                Hm[a, b] = f(H[i, a, b])
            for b in range(nq):
                Em[b, a] = f(E[i, b, a])
        A, gm, em = Hm+w*Em.T*Em, mp.matrix([f(a) for a in gq[i]]), mp.matrix([f(a) for a in e[i]])
        c0 = gm-w*Em.T*em                                          # the cost is z'Az/2 + c0'z + const
        best = None
        for pat in itertools.product(range(3), repeat=n):
            z = mp.matrix([f(lo[i, j]) if pat[j] == 1 else f(hi[i, j]) if pat[j] == 2 else 0 for j in range(n)])
            fr = [j for j in range(n) if pat[j] == 0]
            if fr:
                rhs = -(c0+A*z)
                zf = mp.lu_solve(mp.matrix([[A[a, b] for b in fr] for a in fr]), mp.matrix([rhs[a] for a in fr]))
                for a, j in enumerate(fr):
                    z[j] = zf[a]
            if all(f(lo[i, j]) <= z[j] <= f(hi[i, j]) for j in range(n)):
                c = (z.T*A*z)[0]/2+(c0.T*z)[0]
                best = (c, z) if best is None or c < best[0] else best
        assert max(abs(f(x[i, j])-best[1][j]) for j in range(n)) < 1e-9, i
    # ---- constrained plants with levels
    segs, tgts, infos, seg_k, kinds, delay, ramp = PR.draw_episode(PR.Rng(3), 192, exc_p=1.)
    sel = [(p, t, PR.constrained(p, t) & (seg_k[:, e] < PR.HORIZON) & ((p['integ']*p['trk']).sum(-1) > 0)) for e, (p, t) in enumerate(zip(segs, tgts))]
    p = {k: torch.cat([q[k][on] for q, _, on in sel]) for k in segs[0]}
    t = torch.cat([tq[on] for _, tq, on in sel])
    assert t.shape[0] > 80
    one = lambda q, i: {k: v[i:i+1] for k, v in q.items()}        # noqa: E731
    ue, v, y, sat, ok = PR.optimum(p, t)
    E, e, rate = PR.levels(p, PR.hinv(t, p['wa'])[0])
    assert bool(ok.all()) and float(((E@ue[..., None])[..., 0]-e).abs().max()) < 1e-4      # bands of drift per 100 decisions
    band, trk = p['bandf'], p['trk']
    dev = lambda ya, yb, i=slice(None): float(((ya-yb).abs()/band[i]*trk[i]).max())    # noqa: E731
    near = dict(p, K=p['K']*(1+1e-14*(2*torch.rand(p['K'].shape, generator=g, dtype=F)-1)))
    assert dev(PR.optimum(near, t)[2], y) < 1e-6
    info = dict(ue=ue, un=(ue.clamp_min(0)**(1/p['gam']))*p['mvm'], v=v, y=y)
    R = PR.unremovable(p, info)
    sw = torch.sqrt(PR.steady_weight(p))
    zact = ((y > p['hi']) | (y < p['lo'])) & (p['zon'] > 0)
    worst = 0.
    for i in torch.randperm(t.shape[0], generator=g)[:10].tolist():
        assert dev(PR.optimum(one(p, i), t[i:i+1])[2], y[i:i+1], slice(i, i+1)) < 1e-6
        K = [[f(a) for a in row] for row in p['K'][i]]
        dss = [sum(K[r][NMV+j]*f(p['dlev'][i, j]) for j in range(PR.NDQ))+f(p['bias'][i, r]) for r in range(NCV)]
        wa, bd, tg = [f(a) for a in p['wa'][i]], [f(a) for a in band[i]], [f(a) for a in t[i]]
        gam, on = [f(a) for a in p['gam'][i]], [bool(a > 0) for a in p['mvm'][i]]
        lo = [f(p['ulo'][i, j])**gam[j] if on[j] else mp.mpf(0) for j in range(NMV)]
        hi = [f(p['uhi'][i, j])**gam[j] if on[j] else mp.mpf(0) for j in range(NMV)]
        h = lambda a, c: mp.mpf('.5')+(a-.5)+c*(a-.5)*abs(a-.5)   # noqa: E731
        hp = lambda a, c: 1+2*c*abs(a-.5)                          # noqa: E731
        hinv = lambda a, c: mp.mpf('.5')+mp.sign(a-.5)*2*abs(a-.5)/(1+mp.sqrt(1+4*c*abs(a-.5)))   # noqa: E731
        lev = [bool(a > 0) for a in (p['integ'][i]*trk[i])]
        swi = [f(a) for a in sw[i]]

        def rows(z, i=i, K=K, dss=dss, wa=wa, bd=bd, tg=tg, lev=lev, swi=swi, h=h, hp=hp, hinv=hinv):
            Rr, Jr = [], []
            for r in range(NCV):
                fl = sum(K[r][j]*z[j] for j in range(NMV))+dss[r]
                if lev[r]:
                    vt = hinv(tg[r], wa[r])
                    c = mp.sqrt(w)*100/(f(p['tint'][i, r])*bd[r])*hp(vt, wa[r])
                    Rr.append(c*(fl+f(p['lam'][i, r])*(vt-f(p['vc'][i, r])))); Jr.append([c*K[r][j] for j in range(NMV)])
                elif trk[i, r] > 0:
                    Rr.append(swi[r]/bd[r]*(h(fl, wa[r])-tg[r])); Jr.append([swi[r]/bd[r]*hp(fl, wa[r])*K[r][j] for j in range(NMV)])
                elif p['zon'][i, r] > 0 and not f(p['lo'][i, r]) <= h(fl, wa[r]) <= f(p['hi'][i, r]):
                    lim = f(p['hi'][i, r]) if h(fl, wa[r]) > f(p['hi'][i, r]) else f(p['lo'][i, r])
                    Rr.append(mp.sqrt(10)/bd[r]*(h(fl, wa[r])-lim)); Jr.append([mp.sqrt(10)/bd[r]*hp(fl, wa[r])*K[r][j] for j in range(NMV)])
            return mp.matrix(Rr), mp.matrix(Jr)

        z = [lo[j] if sat[i, j] and ue[i, j] <= float(lo[j])+1e-12 else hi[j] if sat[i, j] else f(ue[i, j]) for j in range(NMV)]
        fr = [j for j in range(NMV) if on[j] and not sat[i, j]]
        for _ in range(60 if fr else 0):                           # Newton on the true cost, the valves at a limit staying there
            Rr, Jr = rows(z)
            Jf = col(Jr, fr)
            s = mp.lu_solve(Jf.T*Jf+mp.mpf('1e-12')*mp.eye(len(fr)), -(Jf.T*Rr+mp.mpf('1e-12')*mp.matrix([z[j] for j in fr])))
            for a, j in enumerate(fr):
                z[j] += s[a]
            if max(abs(a) for a in s) < mp.mpf('1e-28'):
                break
        Rr, Jr = rows(z)
        grad = Jr.T*Rr
        assert all(lo[j] <= z[j] <= hi[j] for j in fr) and all(grad[j] >= 0 if z[j] == lo[j] else grad[j] <= 0 for j in range(NMV) if on[j] and sat[i, j])
        yr = [h(sum(K[r][j]*z[j] for j in range(NMV))+dss[r], wa[r]) for r in range(NCV)]
        worst = max(worst, max(abs(f(y[i, r])-yr[r])/bd[r] for r in range(NCV) if trk[i, r] > 0 and not lev[r]))
        # what a unit of every load leaves on the CVs, the free valves set again: the same rows, linear at this point
        per = [mp.mpf(1) if lev[r] else (mp.sqrt(10) if zact[i, r] else swi[r]) for r in range(NCV)]
        cw = [(mp.sqrt(w)*100/f(p['tint'][i, r]) if lev[r] else per[r])*hp(f(v[i, r]), wa[r])/bd[r]*((1 if trk[i, r] > 0 else 0)+(1 if zact[i, r] else 0)) for r in range(NCV)]
        Jm = mp.matrix([[cw[r]*K[r][j] for j in fr] for r in range(NCV)]) if fr else None
        for d in range(PR.NDQ):
            Jd = mp.matrix([cw[r]*K[r][NMV+d]*f(p['dact'][i, d]) for r in range(NCV)])
            if fr:
                Ja = mp.matrix([[0 if lev[r] else Jm[r, a] for a in range(len(fr))] for r in range(NCV)])
                Ha = Ja.T*Ja
                Jd = Jd-Jm*mp.lu_solve(Jm.T*Jm+mp.mpf('1e-12')*mp.diag([Ha[a, a] for a in range(len(fr))]), Jm.T*Jd)
            ref = [Jd[r]/(mp.sqrt(w) if lev[r] else per[r]) for r in range(NCV)]
            assert max(abs(f(R[i, r, d])-ref[r]) for r in range(NCV)) < 1e-6, (i, d)
    assert worst < 1e-7, worst


def test_a_plant_is_met_at_many_clocks_and_a_storage_answers_in_two_parts():
    """The unit's time scale runs from a decision or two to about ninety; a quarter of the units have a storage behind
    them, and a row that feels it answers every input with a fast part and a slow one (the storage's time constant,
    shares that add up to one). The heating loops of BOPTEST have their like among the paired responses."""
    from aio.sim import model as MD, stats as ST
    n = 1536
    p, tgt, info = PR.draw_start(PR.Rng(5), n)
    p = PR.derive(p)
    T, store = p['T'], p['store']
    assert float(T.min()) >= 1. and float(T.max()) <= PR.TAU_MAX and float((T > 45).double().mean()) > .1 and float((T < 5).double().mean()) > .1
    assert abs(float((store > 0).double().mean())-PR.STORE_P) < .04 and float(store[store > 0].min()) >= PR.STORE_T[0] and float(store.max()) <= PR.STORE_T[1]
    assert float(T[store > 0].max()) <= PR.STORE_FAST[1]             # such a unit's decision interval is chosen for its fast part
    # the two parts: real poles, the slow one the row's own, weights between 0 and 1 that add up to 1
    act = (p['prc'][:, :, None]*p['uact'][:, None, :]) > 0
    tail = act & (p['first'] == 0) & (p['p1i'] == 0) & (p['beta'] < 0) & ((-1/p['p2r']) >= PR.TAIL_MIN*(-1/p['p1r'])*(1-1e-9))
    assert float(tail[store > 0].double().mean()) > .3*float(act[store > 0].double().mean())
    c0, c1 = p['c0r'][tail], p['c1r'][tail]
    assert float((c0+c1-1).abs().max()) < 1e-9 and float(c1.min()) >= .1-1e-9 and float(c1.max()) <= .95+1e-9
    rows = tail.any(-1)
    slow = torch.where(tail, -1/p['p2r'], torch.zeros_like(p['p2r']))
    lo = torch.where(tail, -1/p['p2r'], torch.full_like(p['p2r'], 1e9)).amin(-1)
    assert float((slow.amax(-1)-lo)[rows].abs().max()) < 1e-9          # one storage per row: every input meets the same slow time constant
    # the response is the sum of the two lags
    Fr = MD.response(p)
    i, r, j = (tail & (p['dead'] == 0) & (p['nlag'][:, :, None] == 0)).nonzero(as_tuple=True)
    i, r, j = i[:200], r[:200], j[:200]
    k = torch.arange(Fr.shape[1], dtype=F)[None]
    tf, ts, a = (-1/p['p1r'])[i, r, j][:, None], (-1/p['p2r'])[i, r, j][:, None], p['c1r'][i, r, j][:, None]
    mv = j < NMV
    aa = torch.where(mv, p['aa'][i, j.clamp_max(NMV-1)], torch.zeros(i.numel(), dtype=F))
    want = p['K'][i, r, j][:, None]*((1-a)*(1-torch.exp(-k/tf))+a*(1-torch.exp(-k/ts)))
    got = Fr[i, :, r, j]
    plain = (aa == 0) & (p['integ'][i, r] == 0)                      # no positioner lag in front of the pair
    assert int(plain.sum()) > 20 and float((got-want)[plain].abs().max()) < 1e-9
    # the like of the BOPTEST loops
    S = MD.local(Fr, p, info)[2]
    pair = p['link'].argmax(1)
    ratio, spread = ST.shapes(S, pair, (p['mvm'] > 0) & (torch.gather(p['integ'], 1, pair) == 0))
    for q in ST.LOOPS:
        assert ST.alike(ratio, spread, q) > .005, q


def test_plant_rests_at_its_steady_point():
    prog = C.build(48, 2, dtype=torch.float64, quiet=True)
    st = C.reset(prog)
    y0 = st['yt'].clone()
    a = torch.zeros(prog.n, NMV, dtype=torch.float64)
    for k in range(19):
        st, out = C.step(prog, st, a, k)
        assert float(out['cost'].max()) < 1e-18
    assert float((st['yt']-y0).abs().max()) < 1e-12


def test_local_model_is_the_step_response_of_the_plant():
    prog = C.build(96, 3, dtype=torch.float64, quiet=True)
    c = prog.c
    lin = (c['linear'] > 0).nonzero().flatten()
    assert lin.numel() > 10
    steps = 19                                                         # the first event is at decision 20 or later
    for j in range(2):
        st = C.reset(prog)
        y0 = st['yt'].clone()
        a = torch.zeros(prog.n, NMV, dtype=torch.float64); a[:, j] = .25
        du = a[:, j]*c['slew'][:, j]*c['mvm'][:, j]
        zero = torch.zeros_like(a)
        for k in range(steps):
            st, _ = C.step(prog, st, a if k == 0 else zero, k)
            want = prog.S0[:, k+1, :, j]*du[:, None]
            got = (st['yt']-y0)[:, :NMV]
            err = (got-want)[lin].abs().amax()
            assert float(err) < 1e-9*max(1., float(want[lin].abs().amax())), (j, k, float(err))


@pytest.mark.parametrize('dims,n', [('12x5', 32), ('24x10', 24)])
def test_the_teachers_prediction_takes_in_a_change_of_a_valves_characteristic(dims, n):
    """A valve's characteristic that changes moves its flow with no move made (the flow of an MV the operator holds
    too): the teacher's prediction takes the change in, so where it says the rows go with the valves left where they
    are is where they go. Plants whose clock does not follow a load (a valve that sets the clock changes the plant's pace
    with its characteristic)."""
    from aio.sim import model as MD
    prog = C.build(n, 5, dtype=torch.float64, quiet=True, dims=dims)
    c, nr = prog.c, prog.nr
    for q in ('P', 'T', 'Fr', 'phis'):                                 # no event: segment 0 throughout
        x = getattr(prog, q)
        x[:, 1:] = x[:, :1]
    st = C.reset(prog)
    a = torch.zeros(prog.n, prog.nmv, dtype=torch.float64)
    for k in range(5):
        st, _ = C.step(prog, st, a, k)
    before = st['fl'][:, :nr].clone()
    x, y, _ = prog.lay['gam']
    prog.P[:, :, x:y] = (prog.P[:, :, x:y]*1.2).clamp(.6, 1.7)
    for k in range(5, 45):                                             # (the flow jumps at the valve, past its positioner: the model's response has it)
        st, _ = C.step(prog, st, a, k)
    L = 150
    pred = MD.lerp_at(st['tF'], torch.arange(L+1, dtype=torch.float64)[None]*prog.phis[:, :1])+(st['fl'][:, :nr]-st['tF'][:, 0])[:, None]
    got = []
    for k in range(45, 45+L):
        st, _ = C.step(prog, st, a, k)
        got.append(st['fl'][:, :nr])
    got = torch.stack(got, 1)
    rows = (c['prc'][:, :nr] > 0) & (c['mech'] == 0)[:, None] & (c['warp'] == 0)[:, None]
    band = c['sband'][:, :nr]
    moved = ((got[:, -1]-before)/band).abs()*rows
    err = ((got-pred[:, 1:])/band[:, None]).abs().amax(1)*rows
    big = moved > 1
    assert int(big.sum()) > 20
    assert float((err/moved.clamp_min(1e-9))[big].max()) < .15
    if dims != '12x5':
        assert bool((c['thin'] & big.any(1)).any())                    # (some of them with an MV held)


@pytest.mark.parametrize('kind', ['gain', 'actuator', 'time', 'dead', 'lag'])
def test_the_teachers_prediction_is_the_plants_own_when_its_model_is_renewed(kind, monkeypatch):
    """One process change of one kind as the generator draws it, nothing else moving the plant; at the decision the
    teacher's model follows it, the valves are left alone: the rows go where its prediction says, whether the valves
    stood still until then (exactly) or were being moved, the delay lines and lags loaded (as closely as with no event
    at all). Without the renewal of the prediction (core._renew) the old model's account of the moves made stays in
    it. A gradual change of a time constant or a dead time is taken at its end (audit.RAMPED): the changes at once.
    Valves without friction: the prediction takes a valve to follow its command."""
    from aio.sim import audit as A

    def read(moving):
        r = A.prediction(40, 5, kind=kind, moving=moving, dtype=torch.float64, friction=False)
        on = r['rows'] & (r['has'] & ~r['slow'] & (~r['gradual'] if kind in A.RAMPED else r['has']))[:, None]
        assert int(on.sum()) > 15
        return r['err'][on]
    assert float(read(False).max()) < 1e-3
    moved = read(True)
    assert float(moved.quantile(.99)) < .5
    if kind in ('gain', 'time'):
        monkeypatch.setattr(C, '_renew', lambda prog, st, fk, sg, k, phi, P, who: st['tF'][who])
        assert float(read(True).quantile(.99)) > 4*max(float(moved.quantile(.99)), .5)


@pytest.mark.parametrize('kind', ['time', 'dead'])
def test_a_gradual_change_is_in_the_prediction_in_full_once_it_has_arrived(kind, monkeypatch):
    """While time constants or a dead time are on their way the prediction has them at their end. When they have
    arrived it is worked out again (core.step), and from there the rows go where it says as closely as with no event
    at all, the valves having been moved all along; renewed at the events alone it keeps what the way there left."""
    from aio.sim import audit as A

    def read():
        r = A.prediction(40, 5, kind=kind, moving=True, dtype=torch.float64, arrived=True, friction=False)
        on = r['rows'] & (r['has'] & r['gradual'] & ~r['slow'])[:, None]
        assert int(on.sum()) > 50
        return r['err'][on]
    e = read()
    assert float(e.quantile(.9)) < .05 and float(e.max()) < .25
    renew = C._renew
    monkeypatch.setattr(C, '_renew', lambda prog, st, fk, sg, k, phi, P, who: torch.where(
        (sg != prog.teacher_seg(k))[who][:, None, None], renew(prog, st, fk, sg, k, phi, P, who), st['tF'][who]))
    late = read()
    assert float(late.quantile(.9)) > 5*float(e.quantile(.9)) and float(late.max()) > 4*float(e.max())


def test_the_prediction_of_a_plant_whose_clock_moves_is_worked_out_again_every_so_often(monkeypatch):
    """A plant whose clock follows a load: the prediction is shifted by the clock's rate at every decision, and what the
    valves' moves have left in it blurs. Worked out again every core.REBASE decisions it stays several times closer to
    where the rows go (read half way between two renewals, no event, the valves moved until then)."""
    from aio.sim import audit as A

    def read():
        r = A.prediction(64, 5, moving=True, dtype=torch.float64, friction=False)
        on = r['rows'] & r['slow'][:, None]
        assert int(on.sum()) > 50
        return r['err'][on]
    e = read()
    monkeypatch.setattr(C, 'REBASE', 0)
    left = read()
    assert float(left.quantile(.9)) > 2*float(e.quantile(.9)) and float(left.quantile(.99)) > 4*float(e.quantile(.99))


def test_the_renewals_worked_out_for_the_window_are_those_of_every_decision(monkeypatch):
    """Program.renewals (the decisions after which `step` works the prediction out again, and of which episodes) is the
    rule `step` applied at every decision (core._renewed): every teacher mode, a model installed in a segment's place,
    a window that carries on, plants of the engine (never renewed), with and without REBASE."""
    progs = [C.build(48, 3, mech=.25), C.build(32, 11, dims='24x10'), C.build(48, 3).next()]
    p = progs[0]
    p.install(p.Fr[:, 1].clone(), p.phis[:, 1].clone(), seg=1)
    for rebase in (C.REBASE, 0):
        monkeypatch.setattr(C, 'REBASE', rebase)
        for prog in progs:
            for mode in ('late', 'now', 'fixed'):
                prog.set_teacher(mode)
                table = prog.renewals()
                for k in range(C.HORIZON):
                    one = C._renewed(prog, k)
                    assert (one is None and k not in table) or torch.equal(table[k], one), (mode, rebase, k)
                assert mode != 'late' or len(table) > 10
    assert bool((progs[0].c['mech'] > 0).any())


def test_the_steady_layers_step_along_a_line_is_the_exact_minimiser():
    """core._line (a larger size): the cost along the line is convex and piecewise quadratic, its slope piecewise linear;
    the step returned is where the slope changes sign (0 or 1 where it does not within the line), as sixty bisections
    of the slope find it."""
    g = torch.Generator().manual_seed(0)
    f = torch.float64
    R = lambda *s: torch.randn(*s, generator=g, dtype=f)             # noqa: E731
    U = lambda *s: torch.rand(*s, generator=g, dtype=f)              # noqa: E731
    m, nr, nq, nz, nmv = 256, 10, 3, 14, 6
    J, e, E, eE, Gz, fz, wz = R(m, nr, nmv), R(m, nr), .01*R(m, nq, nmv), .01*R(m, nq), R(m, nz, nmv), R(m, nz), 3*U(m, nz)
    zl, zh = torch.where(U(m, nz) < .3, torch.full((m, nz), -1e9, dtype=f), fz-2*U(m, nz)), torch.where(U(m, nz) < .3, torch.full((m, nz), 1e9, dtype=f), fz+2*U(m, nz))
    slot = U(m, nz) < .8
    x, d = .3*R(m, nmv), R(m, nmv)*10**(4*U(m, 1)-3)
    d[:8] = 0; Gz[8:16] = 0                                         # (no way to go; no zone row that answers)

    def slope(a):
        xa = x+a[:, None]*d
        z = fz+(Gz@xa[..., None])[..., 0]
        over = ((z-zh).clamp_min(0)-(zl-z).clamp_min(0))*slot
        r, l = (J@xa[..., None])[..., 0]-e, (E@xa[..., None])[..., 0]-eE
        return (r*(J@d[..., None])[..., 0]).sum(-1)+(l*(E@d[..., None])[..., 0]).sum(-1)/PR.EQ_SOFT+(wz**2*over*(Gz@d[..., None])[..., 0]).sum(-1)
    a = C._line(J, e, E, eE, Gz, fz, wz, zl, zh, slot, x, d)
    lo, hi = torch.zeros(m, dtype=f), torch.ones(m, dtype=f)
    for _ in range(60):
        mid = .5*(lo+hi)
        down = slope(mid) < 0
        lo, hi = torch.where(down, mid, lo), torch.where(down, hi, mid)
    want = torch.where(slope(torch.ones(m, dtype=f)) <= 0, torch.ones(m, dtype=f), torch.where(slope(torch.zeros(m, dtype=f)) >= 0, torch.zeros(m, dtype=f), lo))
    assert float((a-want).abs().max()) < 1e-12
    inside = (a > 0) & (a < 1)
    assert int(inside.sum()) > 20 and int((a == 0).sum()) > 20 and int((a == 1).sum()) > 20 and bool((a[:8] == 1).all())


def test_a_dead_time_the_plant_cannot_hold_is_not_in_the_model_either():
    """The plant keeps prior.DMAX decisions of its inputs: on a plant that runs slowly a dead time longer than that in
    decisions is DMAX decisions there. The model has the same (model.response, phi): a valve moved on such a plant, the
    rows behind the capped pairs go when and where the prediction says."""
    from aio.sim import model as MD
    prog = C.build(256, 3, dtype=torch.float64, quiet=True)
    c, nr, n = prog.c, prog.nr, prog.n
    for q in ('P', 'T', 'Fr', 'phis'):                                 # no event: segment 0 throughout
        x = getattr(prog, q)
        x[:, 1:] = x[:, :1]
    P = prog.params(torch.zeros(n, dtype=torch.long))
    phis = prog.phis[:, 0]
    far = (P['dead'][:, :nr, :NMV]/phis[:, None, None] > PR.DMAX+2) & (c['prc'][:, :nr, None] > 0) & (c['mvm'][:, None, :] > 0) & (P['K'][:, :nr, :NMV].abs() > 0)
    far &= (torch.arange(NMV)[None] != c['jload'][:, None])[:, None, :]                 # (not the valve the clock follows: the pace stays)
    ep = far.flatten(1).any(1).nonzero().flatten()
    assert ep.numel() >= 4
    j = far[ep].any(1).double().argmax(1)                              # one such valve per plant
    a = torch.zeros(n, NMV, dtype=torch.float64)
    a[ep, j] = 1-2*(prog.start['un'][ep, j] > .5).double()
    st = C.reset(prog)
    before = st['fl'][:, :nr].clone()
    for k in range(3):
        st, _ = C.step(prog, st, a, k)
    L, zero = 150, torch.zeros_like(a)
    pred = MD.lerp_at(st['tF'], torch.arange(L+1, dtype=torch.float64)[None]*phis[:, None])+(st['fl'][:, :nr]-st['tF'][:, 0])[:, None]
    got = []
    for k in range(3, 3+L):
        st, _ = C.step(prog, st, zero, k)
        got.append(st['fl'][:, :nr])
    got = torch.stack(got, 1)
    rows = far[ep, :, j]
    band = c['sband'][ep, :nr]
    moved = ((got[ep]-before[ep, None])/band[:, None]).abs().amax(1)[rows]
    err = ((got[ep]-pred[ep, 1:])/band[:, None]).abs().amax(1)[rows]
    assert float(moved.max()) > 1 and float((err/moved.clamp_min(.5)).max()) < .1


def test_the_steady_layer_does_not_chase_what_the_valves_can_hardly_reach(monkeypatch):
    """core._reachable: of a process row's error along a direction the valves answer s bands a span in, s^2/(s^2+WEAK)
    is kept; what the valves reach well stays, a resting row's error stays as it is. On the plant it was found on (the
    default tuning batch, plant 11: setpoints nearly collinear, a spare MV) the teacher no longer loses a window to the
    formula PI."""
    g = torch.Generator().manual_seed(0)
    U, _ = torch.linalg.qr(torch.randn(3, 6, 6, generator=g, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(3, 4, 4, generator=g, dtype=torch.float64))
    s = torch.tensor([80., 20., 1., .5], dtype=torch.float64)
    J = U[:, :, :4]*s@V.transpose(1, 2)
    none = torch.zeros(3, 6, dtype=torch.bool)
    for i in range(4):
        e = U[:, :, i]
        assert torch.allclose(C._reachable(J, e, none), e*s[i]**2/(s[i]**2+C.WEAK), atol=1e-9)
    assert float(C._reachable(J, U[:, :, 5], none).abs().max()) < 1e-9          # (what no valve moves is not asked for)
    rst = none.clone(); rst[:, 2] = True
    e = torch.randn(3, 6, generator=g, dtype=torch.float64)
    out = C._reachable(J, e, rst)
    assert torch.equal(out[:, 2], e[:, 2]) and not torch.allclose(out[:, :2], e[:, :2])
    monkeypatch.setattr(C, 'WEAK', 0.)
    assert C._reachable(J, e, rst) is e
    monkeypatch.undo()
    prog = C.build(12, 78_001)
    pi, teacher = CT.rollout(prog, CT.PI())['score'], CT.rollout(prog, CT.Teacher('late'))['score']
    assert float(teacher[11]) < .9*float(pi[11]) and bool((teacher < pi).all())
    monkeypatch.setattr(C, 'WEAK', 0.)
    assert float(CT.rollout(prog, CT.Teacher('late'))['score'][11]) > float(teacher[11])+50


def test_packet_is_the_public_encoding():
    prog = C.build(24, 4, dtype=torch.float64)
    c = prog.c
    st = C.reset(prog)
    ctl = CT.PI(); ctl.reset(prog)
    for k in range(40):
        obs = C.observe(prog, st, k)
        if k in (0, 17, 39):
            t = prog.task(prog.seg_at(k))
            for i in range(prog.n):
                rows, m = int(c['rowm'][i].sum()), int(c['mvm'][i].sum())
                g = lambda x: x[i, :rows].numpy()                      # noqa: E731
                h = lambda x: x[i, :m].numpy()                         # noqa: E731
                dt = float(c['dt'][i])
                want = IF.encode(g(st['y']), g(st['prev_y']), g(st['target']), g(st['last_target']), g(t['scale']), g(t['plo']), g(t['phi']),
                                 g(c['trk']), h(st['u']), h(st['prev_u']), h(t['ulo']), h(t['uhi']), np.minimum(h(c['slew']), h(t['uhi'])-h(t['ulo'])),
                                 [int(c['link'][i, :, b].argmax()) for b in range(m)], dt, known=1-(g(st['bad']) > 0),
                                 age=np.where(g(c['an_T']) > 0, g(st['age'])*dt, 0.), sign=h(c['psign']), dv=g(c['dvr']), weight=g(t['cvw']))
                # (a bad-quality row publishes zeros for its value features: the encoder does it)
                assert (np.abs(obs[i].numpy()-want)/(1+np.abs(want))).max() < 2e-6, (k, i)
        st, _ = C.step(prog, st, ctl(prog, st, k, obs), k)


def test_the_first_packet_shows_the_move_that_took_the_valves_off_their_steady_positions(monkeypatch):
    prog = C.build(8, 4)
    c, s = prog.c, prog.start
    st = C.reset(prog)
    t = prog.task(prog.seg_at(0))
    off = ((s['u']-s['un'])*c['mvm']).abs().amax(1) > 0
    assert bool(off.any()) and not bool(off.all())                 # half the episodes start with the inputs off the steady point
    move = IF.split(C.observe(prog, st, 0), NMV)[1][..., 1]         # (n, NMV): the last move, inside the MV limits in force
    want = 2*(s['u']-s['un'])/(t['uhi']-t['ulo'])*c['mvm']
    assert torch.allclose(move, want, atol=1e-6) and not bool(move[~off].any()) and bool((move[off].abs().amax(1) > 0).all())
    st1, _ = C.step(prog, st, torch.zeros(prog.n, NMV), 0)         # from the next packet on: the move made, as ever
    assert torch.equal(st1['prev_u'], st['u'])
    # the plant, the score and the references do not see it: with the first packet showing no move, the same episodes
    ctls = dict(hold=CT.Hold, pi=CT.PI, dmc=lambda: CT.Teacher('fixed'), teacher=lambda: CT.Teacher('late'))
    now = {q: CT.rollout(prog, f()) for q, f in ctls.items()}
    reset = C.reset

    def none(prog):
        st = reset(prog)
        st['prev_u'] = st['u']
        return st

    monkeypatch.setattr(C, 'reset', none)
    for q, f in ctls.items():
        was = CT.rollout(prog, f())
        assert torch.equal(was['score'], now[q]['score']) and torch.equal(was['cost'], now[q]['cost']), q


def test_cost_gradient_matches_finite_differences():
    torch.manual_seed(0)
    prog = C.build(12, 5, dtype=torch.float64, quiet=True, teacher=False)
    n, steps = prog.n, 30
    a = (.3*torch.randn(steps, n, NMV, dtype=torch.float64)).requires_grad_(True)

    def total(act):
        st = C.reset(prog)
        tot = 0.
        for k in range(steps):
            st, out = C.step(prog, st, act[k], k, through=False)
            tot = tot+out['cost']
        return tot

    cost = total(a)
    grad, = torch.autograd.grad(cost.sum(), a)
    lin = (prog.c['linear'] > 0).nonzero().flatten().tolist()         # smooth plants: no friction, no limits hit
    assert lin
    eps = 1e-6
    for i in lin[:4]:
        for (k, j) in ((0, 0), (7, 0), (15, 1)):
            if prog.c['mvm'][i, j] == 0:
                continue
            d = torch.zeros_like(a); d[k, i, j] = eps
            with torch.no_grad():
                fd = (total(a+d)[i]-total(a-d)[i])/(2*eps)
            assert abs(float(fd-grad[k, i, j])) <= 1e-5*max(1., abs(float(fd))), (i, k, j, float(fd), float(grad[k, i, j]))


def test_teacher_beats_the_formula_pi_and_rollouts_repeat():
    prog = C.build(64, 6)
    pi = CT.rollout(prog, CT.PI())
    te = CT.rollout(prog, CT.Teacher('late'))
    assert not bool(te['failed'].any())
    assert float(te['cost'].mean()) < .5*float(pi['cost'].mean())
    assert float(te['score'].mean()) < .8*float(pi['score'].mean())
    again = CT.rollout(prog, CT.Teacher('late'))
    assert torch.equal(te['cost'], again['cost'])


def test_a_score_is_read_before_and_after_the_first_process_change():
    prog = C.build(32, 6)
    kc = CT.changed_k(prog)
    proc = (prog.kinds == PR.PROC) | (prog.kinds == PR.BOTH)
    has = proc.any(1)
    assert has.any() and (~has).any()
    first = proc.float().argmax(1)
    assert torch.equal(kc[has], prog.seg_k[torch.arange(prog.n), first][has]) and (kc[~has] == PR.BIG).all()
    assert (kc >= prog.k_first).all()                             # a process change is an event
    r = CT.rollout(prog, CT.PI())
    assert torch.equal(r['before']+r['after'], r['score'])
    assert (r['after'][~has] == 0).all() and (r['after'] >= 0).all() and (r['before'] >= 0).all()
    assert r['after'][has].sum() > 0 and r['before'].sum() > 0


def test_the_teacher_keeps_a_zone_row_inside_its_limit(monkeypatch):
    prog = C.build(64, 9)
    c = prog.c
    zon = c['zon'] > 0
    orig, seen = C._steady_targets, []

    def spy(prog, st, sg, vt, wy, lo, hi, vlo, vhi, rounds=4):
        out = orig(prog, st, sg, vt, wy, lo, hi, vlo, vhi, rounds)
        if len(seen) < 3:
            seen.append((vt, vlo, vhi, out, prog.wz[prog.ar, sg]))
        return out
    monkeypatch.setattr(C, '_steady_targets', spy)

    def run(back):
        monkeypatch.setattr(C, 'ZONE_BACK', back)
        ctl = CT.Teacher('now')
        st = C.reset(prog)
        ctl.reset(prog)
        z = t = 0.
        for k in range(PR.HORIZON):
            st, out = C.step(prog, st, ctl(prog, st, k, C.observe(prog, st, k)), k)
            on = (k >= prog.k_first)[:, None]
            z += float((out['outside'] & zon & on).sum())
            t += float((out['outside'] & ~zon & on).sum())
        prog.set_teacher()
        return z, t
    z1, t1 = run(.3)
    # the limits the layers work with: ZONE_BACK bands inside the published ones (a quarter of the zone at most), an open side left open
    checked = 0
    for vt, vlo, vhi, (ot, olo, ohi), wz in seen:
        free = (ot == vt).all(-1)                                  # nothing in the way: the limits come back as they are used
        band = math.sqrt(10.)/wz.clamp_min(1e-30)
        for a, b, sign in ((vlo, olo, 1.), (vhi, ohi, -1.)):
            fin = c['zslot'] & (a.abs() < 1e8) & free[:, None]
            d = (sign*(b-a))[fin]
            want = torch.minimum(.3*band, .25*(vhi-vlo).clamp_min(0.))[fin]
            assert torch.allclose(d, want, rtol=1e-4, atol=1e-7) and bool((d > 0).all())
            assert torch.equal(b[c['zslot'] & (a.abs() >= 1e8)], a[c['zslot'] & (a.abs() >= 1e8)])
            checked += int(fin.sum())
    assert checked > 50
    seen.clear()
    z0, t0 = run(0.)
    assert z1 < .7*z0 and z1+t1 < z0+t0, (z0, t0, z1, t1)         # a row held on its limit is beyond it half the time


def test_constrained_points_are_scored_against_the_reachable_optimum():
    prog = C.build(40, 12, quiet=True, exc_p=1.)
    c = prog.c
    exc = PR.is_excursion(prog.kinds)
    assert int(exc.sum()) >= 10
    trk, zon = c['trk'] > 0, c['zon'] > 0
    zone_at_limit = 0
    # the allowance of a constrained segment is the stretch from the target to the reachable optimum (tracked rows)
    # and at least a band beyond the limit for a zone CV the optimum has there; 0 elsewhere
    for e in range(1, PR.NSEG):
        t = prog.task(torch.full((prog.n,), e, dtype=torch.long))
        es = (t['rtgt']-t['tgt'])/c['bandf']*c['trk']
        on = prog.seg_k[:, e] < PR.HORIZON-1
        k = prog.seg_k[:, e].clamp_max(PR.HORIZON-1)
        for i in on.nonzero().flatten().tolist():
            lo, hi = prog.allowance(int(k[i]))
            if prog.kinds[i, e] != PR.RETURN:
                assert torch.allclose(lo[i][trk[i]], es[i][trk[i]].clamp_max(0.), atol=1e-3)
                assert torch.allclose(hi[i][trk[i]], es[i][trk[i]].clamp_min(0.), atol=1e-3)
                z = hi[i][zon[i]]
                assert bool(((z == 0) | ((z >= 1) & (z <= 1+PR.ZONE_GIVE+1e-6))).all()) and bool((lo[i][zon[i]] == 0).all())
                assert bool(exc[i, e]) or not bool((z > 0).any())
                zone_at_limit += int((z > 0).any())
            assert bool((es[i].abs() > 1e-2).any()) == bool(exc[i, e])
    assert zone_at_limit >= 3
    # a controller that finds the optimum settles inside it; the formula PI, which has no way to trade CVs, does not
    te, pi = CT.rollout(prog, CT.Teacher('now')), CT.rollout(prog, CT.PI())
    share = lambda r: float(r['seg_settled'][exc].sum()/r['seg_rows'][exc].sum())     # noqa: E731
    assert not bool(te['failed'].any())
    assert share(te) < .15 and share(te) < .5*share(pi), (share(te), share(pi))


def test_at_a_larger_size_the_true_teachers_steady_layer_aims_a_resting_row_where_the_cost_charges_it(monkeypatch):
    from aio.sim import stay as ST
    for dims, meter, big in (('12x5', None, False), ('24x10', None, True), ('24x10', True, False)):    # (from the instruments alone: the published values)
        prog = C.build(16, 4, dims=dims)
        nm = prog.dims[1]
        rst = prog.c['rst'][:, :nm] > 0
        assert bool(rst.any())
        st, te = C.reset(prog), CT.Teacher('late', meter=meter)
        te.reset(prog)
        for k in range(30):
            st, _ = C.step(prog, st, te(prog, st, k, None), k)
        E = C.teacher_view(prog, st, 30)['E']
        d = torch.nn.functional.pad(rst.to(st['target'].dtype), (0, st['target'].shape[-1]-nm))*.05     # every resting row 5 % of its flow up
        moved = {}
        for q in ('target', 'rhist'):
            s2 = dict(st, **{q: st[q]+(d[:, None] if q == 'rhist' else d)})
            moved[q] = bool((C.teacher_view(prog, s2, 30)['E']-E).abs().amax() > 1e-6)
        assert moved == dict(target=not big, rhist=big), (dims, meter, moved)
    # the references of a larger size are of another version; the default size's keep their key
    small, large = C.build(4, 1), C.build(4, 1, dims='24x10')
    k0 = (ST.key([small]), ST.key([large]))
    monkeypatch.setattr(ST, 'REFS_BIG', ST.REFS_BIG+1)
    assert ST.key([small]) == k0[0] and ST.key([large]) != k0[1]


def test_spare_mvs_have_resting_values_that_rank_below_every_process_cv():
    r = PR.Rng(21)
    segs, tgts, infos, seg_k, kinds, delay, ramp = PR.draw_episode(r, 768)
    p = segs[0]
    rst, spr = p['rst'] > 0, p['spr'] > 0
    assert int(rst.sum()) > 100
    K, link = p['K'][:, :, :NMV], p['link'][:, :, :NMV]
    # a resting row is a tracked row that reads the flow of its own spare MV and nothing else; wide band, priority 1, no level
    assert bool(((K != 0).sum(-1)[rst] == 1).all()) and bool(((K*link).sum(-1)[rst] == 1).all())
    assert bool((((link*p['rst'][:, :, None]).sum(1) > 0) == spr).all()) and bool((p['trk'][rst] == 1).all())
    assert bool((p['K'][:, :, NMV:][rst] == 0).all()) and bool((p['integ'][rst] == 0).all()) and bool((p['cvw'][rst] == 1).all())
    assert float(p['bandf'][rst].min()) >= PR.REST_BAND[0] and float(p['bandf'][~rst & (p['trk'] > 0)].max()) <= .03
    assert bool(((p['trk']*(1-p['rst'])).sum(-1) >= 1).all())            # a process CV is left on every plant
    # the events leave it that way (an actuator loss shows in the flow, nothing else does)
    for s in segs[1:]:
        Ks = s['K'][:, :, :NMV]
        assert bool(((Ks != 0).sum(-1)[rst] == 1).all()) and bool(((Ks*link).sum(-1)[rst] <= 1).all()) and bool((s['wa'][rst] == 0).all())
        assert bool((s['cvw'][rst] == 1).all())
    # at a constrained point of a fat plant the spare MVs give first: where a process CV is off its target, moving the
    # spare MVs alone inside their limits does not bring the process CVs and the zone limits closer
    fat = spr.any(-1)
    g = torch.Generator().manual_seed(0)
    seen, push, none_gives = 0, 0, 0
    for e, (s, t, info) in enumerate(zip(segs, tgts, infos)):
        sel = PR.is_excursion(kinds[:, e]) & fat
        if not bool(sel.any()):
            continue
        q = {k: v[sel] for k, v in s.items()}
        tq, ue = t[sel], info['ue'][sel]
        lo, hi = q['ulo']**q['gam']*q['mvm'], q['uhi']**q['gam']*q['mvm']
        cv = q['trk']*(1-q['rst'])

        def process(x):                                                # what ranks above the resting values
            v = (q['K'][:, :, :NMV]*x[:, None, :]).sum(-1)+(q['K'][:, :, NMV:]*q['dlev'][:, None, :]).sum(-1)+q['bias']
            y = PR.h(v, q['wa'])
            viol = ((q['lo']-y).clamp_min(0)+(y-q['hi']).clamp_min(0))/q['bandf']*q['zon']
            lvl = (q['integ'] > 0) & (cv > 0)                           # a level has to stay in balance
            bal = (((v+q['lam']*(PR.hinv(tq, q['wa'])[0]-q['vc']))*1e5/(q['tint']*q['bandf']))**2*lvl).sum(-1)
            return (q['cvw']*((y-tq)/q['bandf'])**2*cv*~lvl).sum(-1)+10*(viol**2).sum(-1)+bal, y

        best, y = process(ue)
        lvl = (q['integ'] > 0) & (cv > 0)                                   # a level's row is its rate there, not a value to hold against its target
        off = ((y-tq).abs()/q['bandf']*cv*~lvl).amax(-1)
        none_gives += int((off < .01).sum()); seen += int(sel.sum())
        for scale in (1e-3, 3e-2, .3):
            for _ in range(6):
                x = torch.where(q['spr'] > 0, torch.minimum(torch.maximum(ue+scale*(hi-lo)*torch.randn(ue.shape, generator=g, dtype=ue.dtype), lo), hi), ue)
                assert bool((process(x)[0] >= best-1e-3*(1+best)).all())
        isp = kinds[sel, e] == PR.PUSH
        push += int(isp.sum())
        # a push asks for a resting value at its valve's limit and takes no process CV off its target
        assert float(off[isp].max() if bool(isp.any()) else 0.) < .05
    assert seen > 40 and push >= 8 and none_gives > .4*seen


def test_a_resting_value_is_charged_where_it_can_be_and_the_teacher_uses_the_spare_mvs():
    prog = C.build(96, 22, exc_p=1.)
    c = prog.c
    rst = c['rst'] > 0
    fat = rst.any(-1)
    exc = PR.is_excursion(prog.kinds) & fat[:, None]
    assert int(exc.sum()) >= 12
    # the cost of a resting row is taken from the reachable resting value, the count from its target with the stretch to it
    st = C.reset(prog)
    z = torch.zeros(prog.n, NMV)
    checked = 0
    for k in range(PR.HORIZON):
        seg = prog.seg_at(k)
        st, out = C.step(prog, st, z, k)
        on = exc[prog.ar, seg] & (k >= prog.seg_k[prog.ar, seg]+PR.DMAX+1)
        if bool(on.any()) and k % 20 == 0:
            t = prog.task(seg)
            shift = ((t['rtgt']-t['tgt'])/c['bandf'])[on][rst[on]]
            assert torch.allclose((out['err']-out['erc'])[on][rst[on]], shift, atol=1e-3)
            assert bool(((out['err'] == out['erc']) | rst)[on].all())
            checked += 1
    assert checked > 5
    te, pi = CT.rollout(prog, CT.Teacher('now')), CT.rollout(prog, CT.PI())
    assert not bool(te['failed'].any())
    cv = lambda r: float((r['seg_settled']-r['seg_rest'])[exc].sum()/(r['seg_rows']-r['seg_rest_rows'])[exc].sum())     # noqa: E731
    rest = lambda r: float(r['seg_rest'][exc].sum()/r['seg_rest_rows'][exc].sum())                                      # noqa: E731
    # the teacher has the spare MVs where the best steady point has them and keeps the process CVs; the formula PI holds
    # every resting value and loses the CVs
    assert rest(te) < .12 and cv(te) < .2 and cv(te) < .5*cv(pi), (rest(te), cv(te), cv(pi))


def test_the_teacher_from_the_instruments_does_not_explain_a_level_at_its_stop_away():
    """A level pushed against its physical stop reads still while its net inflow goes on: the estimator does not
    take the inflow it cannot see to be gone (it keeps the one it had)."""
    prog = C.build(32, 7)
    prog.set_teacher('late', True, True)

    class Up:
        def reset(self, prog):
            pass

        def __call__(self, prog, st, k, obs):
            return torch.ones(prog.n, prog.nmv, dtype=prog.dtype)*prog.c['mvm']

    st = CT.rollout(prog, Up())['st']
    c, nr = prog.c, prog.nr
    ig = (c['integ'][:, :nr] > 0) & (c['prc'][:, :nr] > 0)
    v, fl = st['v'][:, :nr], st['fl'][:, :nr]
    at = ig & (((v >= C.LEV_STOP[1]-1e-6) & (fl > 0)) | ((v <= C.LEV_STOP[0]+1e-6) & (fl < 0))) & ~st['fail'][:, None]
    assert int(at.sum()) >= 3
    seen = st['tF'][:, 0, :nr]+st['eo']
    assert (seen[at]*fl[at] > 0).float().mean() > .9
    assert ((st['ev'] >= C.LEV_STOP[0]) & (st['ev'] <= C.LEV_STOP[1])).all()

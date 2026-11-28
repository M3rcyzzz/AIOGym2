"""What the prior generates, and where the literature test models (aio.sim.lit, held out) sit in it.

  python -m aio.sim.stats                 # 4096 episodes, seed 7
  python -m aio.sim.stats --n 1024 --out reports/sim/stats.json

Part 1: the generator. Raw acceptance of a plant draw by size and archetype, events per episode by kind,
sizes of setpoint moves, steady valve positions, every operating point re-checked with `point_ok`, the
share of episodes that visit a constrained point, how far its reachable optimum is from the targets, by the
way the unit got there how often a zone CV or an MV is at a limit there, and how long the point is held;
the share of fat plants (spare MVs with a resting value) and, at their constrained points, how far the
process CVs and the resting values are from their targets at the best steady point.
Part 2: coverage. The same features are read off the step responses of the start plants (the local model a
step test would find, Program.S0) and off the literature models at their decision interval; for each
literature model the percentile of each feature among the prior's plants with two or more MVs and no spare
one (the literature models are square) is printed. A percentile outside 1-99 is a feature the prior does
not generate.

Features (per plant, tracked block at the published pairing; times in decisions):
  rga_max, rga_min   largest and smallest relative gain of a published pair
  cond               condition number of the steady gains after column then row scaling
  theta, tau         median over the pairs of the apparent dead time (5 % of the final value) and of the
                     time from there to 63 %
  ratio              largest dead time / time constant ratio of a pair
  late               how much later than a row's own MV its slowest significant interaction arrives
                     (dead time of the other MV / dead time of the paired MV, at least a fifth of its gain)
  speed              slowest / fastest paired response (time to 63 %)
  gain               median paired gain (CV range per full MV span)
Per paired response (`shapes`): the dead time over the time constant, and the time to 63 % over the time to 20 %
of the last value; the share of them like each heating loop of BOPTEST (`LOOPS`, `alike`).
"""
import argparse, json, time
from pathlib import Path
import torch

from aio.interface import NMV
from aio.sim import core as C, prior as PR

FEATS = ('rga_max', 'rga_min', 'cond', 'theta', 'tau', 'ratio', 'late', 'speed', 'gain')
Q = (.01, .05, .25, .5, .75, .95, .99)
# The heating loops of BOPTEST as a step test of 240 decisions reads them (docs/validation.md, section 5): apparent dead time
# over time constant, and the time to 63 % over the time to 20 % of the last value (one lag: 4.5; a fast part and a slow one: more)
LOOPS = dict(radiator=(1/70, 32.6), heat_pump=(3/51, 4.5), commercial=(1/25, 10.), apartment=(5/89, 8.8))


def shapes(S, pair, ok):
    """Of every paired step response of S (m, L+1, NMV, NMV; pair (m, NMV) the row of every MV; ok (m, NMV) the pairs
    to read: in use, not integrating): the apparent dead time over the time constant (as `features` reads them) and the
    time to 63 % over the time to 20 % of the last value (as aio.sim.validate.boptest.shape). Both (pairs,)."""
    m, L = S.shape[0], S.shape[1]-1
    d = torch.gather(S, 2, pair[:, None, :, None].expand(m, L+1, -1, S.shape[3])).diagonal(dim1=2, dim2=3)          # (m, L+1, MVs)
    end = d[:, -1]
    ok = ok & (end.abs() > 1e-9)
    r = (d/torch.where(ok, end, torch.ones_like(end))[:, None]).transpose(1, 2)[ok]                          # (pairs, L+1)
    first = lambda c: torch.where(c.any(1), c.to(torch.int64).argmax(1), torch.full_like(c[:, 0], L, dtype=torch.int64))   # noqa: E731
    th, t63 = first(r.abs() >= .05).to(S.dtype), first(r.abs() >= .632).to(S.dtype)

    def cross(level):
        k = first(r >= level)
        a, b = torch.gather(r, 1, (k-1).clamp_min(0)[:, None])[:, 0], torch.gather(r, 1, k[:, None])[:, 0]
        return torch.where(k == 0, torch.zeros_like(a), (k-1)+(level-a)/torch.where(b != a, b-a, torch.ones_like(a)))
    return th/(t63-th).clamp_min(.5), cross(.632)/cross(.2).clamp_min(1e-9)


def alike(ratio, spread, loop, factor=2.):
    """Share of the pairs within `factor` of a loop of LOOPS in both measures."""
    a, b = LOOPS[loop]
    return float(((ratio >= a/factor) & (ratio <= a*factor) & (spread >= b/factor) & (spread <= b*factor)).double().mean())


def features(S, pair, act, ok):
    """S (m, L+1, NMV, NMV) step responses (row, MV), pair (m, NMV) the row of every MV, act (m, NMV) active
    MVs, ok (m, NMV) rows whose response settles (not integrating). Returns {feature: (m,)} (nan: undefined)."""
    m, L = S.shape[0], S.shape[1]-1
    f = S.dtype
    Sp = torch.gather(S, 2, pair[:, None, :, None].expand(m, L+1, NMV, NMV))         # row of MV i on axis 2
    fin = Sp[:, -1]
    nan = torch.full((m,), float('nan'), dtype=f)
    both = act[:, :, None] & act[:, None, :]
    eye = torch.eye(NMV, dtype=f).expand(m, -1, -1)
    G = torch.where(both, fin, eye)
    lam = torch.diagonal(G*torch.linalg.pinv(G).transpose(1, 2), dim1=1, dim2=2)
    lam = torch.where(act, lam, torch.ones_like(lam))
    Gs = G/G.abs().amax(1, keepdim=True).clamp_min(1e-12)
    Gs = Gs/Gs.abs().amax(2, keepdim=True).clamp_min(1e-12)
    sv = torch.linalg.svdvals(Gs)
    rel = Sp.abs()/fin.abs().clamp_min(1e-12)[:, None]
    first = lambda x: torch.where(x.any(1), x.to(torch.int64).argmax(1), torch.full_like(x[:, 0], L, dtype=torch.int64)).to(f)   # noqa: E731
    th, t63 = first(rel >= .05), first(rel >= .632)              # (m, row, MV)
    d = lambda x: torch.diagonal(x, dim1=1, dim2=2)              # noqa: E731
    on = act & torch.gather(ok, 1, pair)                         # pairs with a settled response
    med = lambda x: torch.where(on, x, torch.full_like(x, float('nan'))).nanmedian(1).values   # noqa: E731
    big = lambda x: torch.where(on, x, torch.full_like(x, -1e30)).amax(1)                      # noqa: E731
    sml = lambda x: torch.where(on, x, torch.full_like(x, 1e30)).amin(1)                       # noqa: E731
    thp, taup = d(th), (d(t63)-d(th)).clamp_min(.5)
    sig = both & on[:, :, None] & (fin.abs() >= .2*d(fin).abs()[:, :, None]) & ~torch.eye(NMV, dtype=torch.bool)
    late = torch.where(sig, th/thp.clamp_min(1.)[:, :, None], torch.zeros_like(th)).amax((1, 2))
    any_on = on.any(1)
    out = dict(rga_max=torch.where(act, lam, torch.full_like(lam, -1e30)).amax(1), rga_min=torch.where(act, lam, torch.full_like(lam, 1e30)).amin(1),
               cond=sv[:, 0]/sv[:, -1].clamp_min(1e-12), theta=med(thp), tau=med(taup), ratio=big(thp/taup), late=late,
               speed=big(d(t63).clamp_min(1.))/sml(d(t63).clamp_min(1.)).clamp_min(1.), gain=med(d(fin).abs()))
    for q in ('theta', 'tau', 'ratio', 'late', 'speed', 'gain'):
        out[q] = torch.where(any_on, out[q], nan)
    return out


def lit_responses(L=1500):
    """Step responses of the literature models at their decision interval, gains scaled as lit_batch does
    (column, then row, to 0.375 of the CV range per full MV span)."""
    from aio.sim import lit as LB
    out = {}
    for t in LB.TABLES:
        n, dt = t['n'], t['dt']
        K = torch.tensor(t['K'][:n, :n]); K = K/K.abs().amax(0, keepdim=True); K = .375*K/K.abs().amax(1, keepdim=True)
        th, t1, t2, ld = (torch.tensor(t[q][:n, :n])/dt for q in ('th', 't1', 't2', 'lead'))
        j = torch.arange(L+1, dtype=torch.float64)[:, None, None]
        x = (j-th[None]).clamp_min(0.)
        fo = 1-torch.exp(-x/t1)
        same = (t2-t1).abs() < 1e-9
        t2s = torch.where(same | (t2 == 0), t1*.5, t2)
        so = 1-(t1-ld)/(t1-t2s)*torch.exp(-x/t1)-(t2s-ld)/(t2s-t1)*torch.exp(-x/t2s)
        crit = 1-(1+x*(t1-ld)/t1**2)*torch.exp(-x/t1)
        y = torch.where(t2 == 0, fo, torch.where(same, crit, so))*(j >= th[None])
        S = torch.zeros(L+1, NMV, NMV, dtype=torch.float64); S[:, :n, :n] = K[None]*y
        out[t['name']] = (S, n)
    return out


def generator(n, seed):
    r = PR.Rng(seed+1)
    m = 20000
    p, tgt, ro = PR.draw_plant(r, m)
    ok, info = PR.operable(p, tgt)
    ok = ok & PR._paired(r, p, info, ok)
    fat0 = p['spr'].sum(-1) > 0
    out = dict(raw_acceptance=round(float(ok.double().mean()), 3), raw_acceptance_fat=round(float(ok[fat0].double().mean()), 3),
               by_nm=[round(float(ok[p['nm'] == i].double().mean()), 3) for i in range(1, NMV+1)],
               by_arch={PR.ARCH[i]: round(float(ok[p['arch'] == i].double().mean()), 3) for i in range(len(PR.ARCH_P))})
    t0 = time.time()
    segs, tgts, infos, seg_k, kinds, delay, ramp = PR.draw_episode(PR.Rng(seed), n)
    out['draw_s'] = round(time.time()-t0, 2)
    ex, acc = (seg_k < PR.BIG)[:, 1:], (kinds >= 0)[:, 1:]
    out['events_per_episode'] = round(float(acc.double().sum(1).mean()), 2)
    out['events_accepted'] = round(float(acc.double().sum()/ex.double().sum()), 3)
    out['events_by_kind'] = {PR.EVENTS[i]: round(float((kinds == i).double().sum()/n), 2) for i in range(len(PR.EVENTS))}
    grad = (kinds == PR.PROC) | (kinds == PR.BOTH) | (kinds == PR.LOAD) | (kinds == PR.DV)
    out['gradual'] = round(float(((ramp > 0) & grad).double().sum()/grad.double().sum()), 2)     # of the process and disturbance changes
    mv = []
    for e in range(1, PR.NSEG):
        is_sp = (kinds[:, e] == PR.SP) | (kinds[:, e] == PR.BOTH)
        mv.append(((tgts[e]-tgts[e-1]).abs()/segs[e]['bandf']*segs[e]['trk']).amax(-1)[is_sp])
    qq = torch.tensor(Q, dtype=PR.F)
    out['setpoint_move_bands'] = [round(float(x), 1) for x in torch.quantile(torch.cat(mv), qq)]
    out['not_operable'] = sum(int((~PR.point_ok(segs[e], tgts[e])).sum()) for e in range(PR.NSEG))
    un = torch.cat([i['un'][s['mvm'] > 0] for i, s in zip(infos, segs)])
    out['steady_valve'] = [round(float(x), 2) for x in torch.quantile(un, qq)]
    # constrained operating points: how many episodes visit one, and how far the reachable optimum is from the targets
    exc = PR.is_excursion(kinds)
    out['episodes_constrained'] = round(float(exc.any(1).double().mean()), 3)
    give = torch.cat([((s['rtgt']-t).abs()/s['bandf']*s['trk']).amax(-1)[exc[:, e]] for e, (s, t) in enumerate(zip(segs, tgts))])
    out['constrained_give_bands'] = [round(float(x), 1) for x in torch.quantile(give, qq)] if give.numel() else []
    # by the way the unit got there: how often a zone CV is at its limit at the best steady point, an MV at a limit, and the give
    cat = lambda f: torch.cat([f(e)[exc[:, e]] for e in range(PR.NSEG)])                       # noqa: E731
    kd = cat(lambda e: kinds[:, e])
    zone = cat(lambda e: (segs[e]['zal'] > 0).any(-1))
    sat = cat(lambda e: (((infos[e]['un'] <= segs[e]['ulo']+1e-6) | (infos[e]['un'] >= segs[e]['uhi']-1e-6)) & (segs[e]['mvm'] > 0)).any(-1))
    beyond = cat(lambda e: (segs[e]['zal']-1).clamp_min(0.).amax(-1))
    out['constrained_by_kind'] = {PR.EVENTS[q]: dict(per_episode=round(float((kd == q).double().sum()/n), 3), zone_at_limit=round(float(zone[kd == q].double().mean()), 2),
                                                    mv_at_limit=round(float(sat[kd == q].double().mean()), 2),
                                                    give_bands_median=round(float(give[kd == q].median()), 1)) for q in PR.EXCURSIONS if bool((kd == q).any())}
    out['constrained_zone_at_limit'] = round(float(zone.double().mean()), 3)
    out['zone_beyond_limit_bands'] = [round(float(x), 2) for x in torch.quantile(beyond[zone], qq)] if bool(zone.any()) else []
    # how long a constrained point is held, and how much of that is left once the unit can have lined out
    end = torch.cat((seg_k[:, 1:], torch.full_like(seg_k[:, :1], PR.BIG)), 1).clamp_max(PR.HORIZON)
    held = cat(lambda e: (end[:, e]-seg_k[:, e]).to(PR.F))
    line = cat(lambda e: PR.lineout(segs[0], infos[max(e-1, 0)], infos[e], ramp[:, e].to(PR.F)))
    out['constrained_held_decisions'] = [round(float(x)) for x in torch.quantile(held, qq)]
    out['constrained_lined_out_decisions'] = [round(float(x)) for x in torch.quantile((held-line).clamp_min(0.), qq)]
    # fat plants: how many, how many spare MVs, how often they reach a constrained point, and who gives there
    p0 = segs[0]
    fat = p0['spr'].sum(-1) > 0
    out['fat'] = dict(share=round(float(fat.double().mean()), 3), share_of_two_or_more_mvs=round(float(fat[p0['nm'] > 1].double().mean()), 3),
                      spare_mvs=[round(float((p0['spr'].sum(-1) == i).double().mean()), 3) for i in range(NMV)],
                      episodes_constrained=round(float(exc[fat].any(1).double().mean()), 3),
                      by_kind={PR.EVENTS[q]: round(float((kinds[fat] == q).double().sum()/fat.sum().clamp_min(1)), 3) for q in PR.EXCURSIONS})
    if bool((exc & fat[:, None]).any()):
        off = lambda e: (segs[e]['rtgt']-tgts[e]).abs()/segs[e]['bandf']*segs[e]['trk']          # noqa: E731
        fsel = lambda f: torch.cat([f(e)[exc[:, e] & fat] for e in range(PR.NSEG)])              # noqa: E731
        gcv, grs = fsel(lambda e: (off(e)*(1-segs[e]['rst'])).amax(-1)), fsel(lambda e: (off(e)*segs[e]['rst']).amax(-1))
        out['fat']['constrained_points'] = dict(n=int(gcv.numel()), no_process_cv_gives=round(float((gcv < .01).double().mean()), 2),
                                                process_cv_give_bands=[round(float(x), 2) for x in torch.quantile(gcv, qq)],
                                                resting_give_bands=[round(float(x), 2) for x in torch.quantile(grs, qq)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    res = dict(n=a.n, seed=a.seed, quantiles=Q, generator=generator(min(a.n, 1024), a.seed))
    print(json.dumps(res['generator'], indent=1))
    prog = C.build(a.n, a.seed, teacher=True, dtype=torch.float64)
    c = prog.c
    pair = c['link'][:, :, :NMV].argmax(1)
    act = c['mvm'] > 0
    fe = features(prog.S0.double(), pair, act, c['integ'][:, :NMV] == 0)
    multi = (act.sum(1) >= 2) & (c['spr'].sum(1) == 0)
    lit = lit_responses()
    fl = {}
    for name, (S, n) in lit.items():
        am = torch.arange(NMV)
        fl[name] = features(S[None], am[None], (am < n)[None], torch.ones(1, NMV, dtype=torch.bool))
    res['prior'], res['lit'] = {}, {q: {} for q in lit}
    print(f"\n{'feature':8s} " + ' '.join(f'{f"q{int(100*q):02d}":>7s}' for q in Q) + ' | ' + ' '.join(f'{q:>14s}' for q in lit) + '   (value, percentile in the prior)')
    for q in FEATS:
        x = fe[q][multi]; x = x[torch.isfinite(x)].sort().values
        qs = torch.quantile(x, torch.tensor(Q, dtype=x.dtype))
        res['prior'][q] = [round(float(v), 3) for v in qs]
        cells = []
        for name in lit:
            v = float(fl[name][q][0])
            pct = 100*float(torch.searchsorted(x, torch.tensor(v, dtype=x.dtype)))/max(len(x), 1)
            res['lit'][name][q] = dict(value=round(v, 3), percentile=round(pct, 1))
            cells.append(f'{v:8.2f} {pct:4.0f}%')
        print(f'{q:8s} ' + ' '.join(f'{float(v):7.2f}' for v in qs) + ' | ' + ' '.join(f'{s:>14s}' for s in cells))
    ratio, spread = shapes(prog.S0.double(), pair, act & (torch.gather(c['integ'], 1, pair) == 0))
    res['loops'] = {q: dict(ratio=round(LOOPS[q][0], 3), spread=LOOPS[q][1], alike=round(alike(ratio, spread, q), 4)) for q in LOOPS}
    print('\nBOPTEST heating loops: share of the paired responses within a factor 2 of each in dead time over time constant and in time to 63 % over time to 20 %')
    print('  ' + ', '.join(f"{q} {100*v['alike']:.1f}%" for q, v in res['loops'].items()))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

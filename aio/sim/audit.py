"""Task-rigor checks of the simulation core (docs/task-rigor-checklist.md; docs/sim.md lists what holds by
construction).

  python -m aio.sim.audit [--n 128] [--size 12x5]

A larger size (prior.SIZES) is checked the same way on its own draws; the mechanistic plants are checked at
the default size only (they are the same plants in a larger batch).
"""
import argparse, sys
import torch

from aio.sim import core as C, control as CT, model as MD, prior as PR, stats as ST


def prediction(n, seed, size='12x5', kind=None, moving=True, mech=0., L=150, dtype=torch.float32, amp=.6, arrived=False, friction=True):
    """The teacher's prediction at the decision its model is renewed, against where the rows then go: the largest distance
    over the next L decisions (bands), the valves left alone from that decision on. One process change per episode, of one
    kind of prior.PROCESS as the generator draws it (None: no event; read at decision 75, half way between two renewals of a plant whose clock moves), on a quiet batch: nothing else
    moves the plant. moving: the valves are moved at random until then (every fourth decision six in ten of them get a new
    move, up to amp of the slew limit), so the delay lines, the modes and the row lags are loaded when the model changes;
    otherwise they stand still. arrived: read when a gradual change has come in full and the model has it (the later of the
    two), the valves moved until then. friction False: the valves without friction (the prediction takes a valve to
    follow its command: one held by friction is a load it does not know of).
    Returns err (n, nr), and what to split it by: rows (process rows), has (the episode had the event), slow (its clock
    follows a load), gradual (the change came over some decisions), mech."""
    menu = PR.Menu(rate=(0., 0., 0., float(kind is not None), 0.), drop=tuple(q for q in PR.PROCESS if q != kind), exc=0.)
    prog = C.build(n, seed, quiet=True, dims=size, menu=menu, mech=mech, dtype=dtype)
    prog.seg_k[:, 2:] = PR.BIG; prog.eff_k[:, 2:] = PR.BIG     # the first event alone
    if not friction:
        for q in ('fs', 'fd'):
            prog.P[:, :, prog.lay[q][0]:prog.lay[q][1]] = 0.
    c, nr, f = prog.c, prog.nr, prog.dtype
    has = prog.seg_k[:, 1] < PR.BIG
    read = prog.eff_k[:, 1].clamp_max(200)                                                                # the step before renews the model
    if arrived:
        read = torch.maximum(prog.eff_k[:, 1], prog.seg_k[:, 1]+prog.ramp[:, 1].ceil().long()).clamp_max(400)
    read = torch.where(has, read-1, torch.full_like(read, 75))
    g = torch.Generator().manual_seed(seed+7)
    st = C.reset(prog)
    a = torch.zeros(n, prog.nmv, dtype=f)
    pred, err = torch.zeros(n, L+1, nr, dtype=f), torch.zeros(n, nr, dtype=f)
    js, ar = torch.arange(L+1, dtype=f)[None], torch.arange(n)
    for k in range(int(read.max())+L+1):
        if moving and k % 4 == 0:
            a = (2*torch.rand(n, prog.nmv, generator=g, dtype=f)-1)*amp*(torch.rand(n, prog.nmv, generator=g, dtype=f) < .6)
        st, _ = C.step(prog, st, a*(k <= read)[:, None], k)
        now = k == read
        if bool(now.any()):                                      # at the time scale the plant runs at, from the value now
            p = MD.lerp_at(st['tF'], js*C._phi(st['xv'], st['d'], c)[:, None])+(st['fl'][:, :nr]-st['tF'][:, 0])[:, None]
            pred[now] = p[now]
        e = (st['fl'][:, :nr]-pred[ar, (k-read).clamp(0, L)]).abs()/c['sband'][:, :nr]
        err = torch.where(((k > read) & (k <= read+L))[:, None], torch.maximum(err, e), err)
    return dict(err=err, rows=c['prc'][:, :nr] > 0, has=has, slow=c['warp'] > 0, gradual=prog.ramp[:, 1] > 0, mech=c['mech'] > 0)


RAMPED = ('time', 'dead', 'lag', 'curve')    # a gradual change of these is not followed along its ramp by the prediction (docs/sim.md section 6)


def run(check, n=128, seed=424242, size='12x5'):
    dims = PR.SIZES[size]
    NCV, NMV, ND = dims
    default = dims == PR.DIMS
    kw = {} if default else dict(dims=dims)
    r = PR.Rng(seed)
    N = 1024*PR.NMV//NMV                                         # episodes drawn for the generator's checks
    segs, tgts, infos, seg_k, kinds, delay, ramp = PR.draw_episode(r, N, **kw)
    # 1, 2 every operating point of every episode passes the generator's own check, with margins
    bad, room, ncon, nzone = 0, 1., 0, 0
    for p, t, i in zip(segs, tgts, infos):
        bad += int((~PR.point_ok(p, t)).sum())
        con = PR.constrained(p, t)
        ncon += int(con.sum())
        nzone += int((p['zal'] > 0).any(-1).sum())
        on = (p['mvm'] > 0) & ~con[:, None] & ~PR.held(p)
        room = min(room, float(torch.minimum(i['un']-p['ulo'], p['uhi']-i['un'])[on].min()))
    check('sim operable', bad == 0 and room >= PR.U_MARGIN-1e-9,
          f'{bad} of {N*PR.NSEG} operating points fail the generator\'s check (`point_ok`); every steady valve is at least {room:.3f} of span inside its '
          f'limits (margin {PR.U_MARGIN}) except at the {ncon} constrained points, which are verified optima ({nzone} of them with a zone CV at its limit)')
    # 2 rejected plants are redrawn with the same roots: sizes and archetypes keep their prior
    nm = torch.stack([(segs[0]['nm'] == i).double().mean() for i in range(1, NMV+1)])
    arch = torch.stack([(segs[0]['arch'] == i).double().mean() for i in range(len(PR.ARCH_P))])
    if default:
        dn, da = float((nm-torch.tensor(PR.NM_P)).abs().max()), float((arch-torch.tensor(PR.ARCH_P)).abs().max())
    else:                                                        # a larger size: the sizes above half of it, equally; every plant several units
        dn, da = float((nm-(torch.arange(1, NMV+1) > NMV//2)/(NMV-NMV//2)).abs().max()), float(1-(segs[0]['arch'] == PR.PLANT).double().mean())
    fat = segs[0]['spr'].sum(-1) > 0
    df = abs(float(fat[segs[0]['nm'] > 1].double().mean())-(PR.FAT_P if default else PR.ECON_P))
    tol = .05*(1024/N)**.5                                       # (fewer episodes are drawn of a larger size)
    ds = abs(float((segs[0]['store'] > 0).double().mean())-PR.STORE_P)
    check('sim prior kept', dn < tol and da < tol and df < tol and ds < tol,
          f'largest deviation of the size shares {dn:.3f}, of the archetype shares {da:.3f}, of the share of fat plants {df:.3f}, of the units with a storage {ds:.3f} '
          f'(rejection does not thin out any kind)')
    ev = (kinds >= 0)[:, 1:].double().sum(1)
    check('sim events', float(ev.mean()) >= 5 and float((ev >= 2).double().mean()) > .97,
          f'{float(ev.mean()):.1f} events per episode, {100*float((ev >= 2).double().mean()):.0f}% of the episodes have two or more')
    # the clock: a plant is met at any decision interval an engineer may have chosen for it. The start plants answer in a few
    # decisions or in a hundred, and the heating loops of BOPTEST (slow against their decision interval, no dead time to speak
    # of, a fast part and a slow one: docs/validation.md section 5) have their like among the paired responses
    p0 = segs[0]
    S = MD.local(MD.response(p0), p0, infos[0])[2]
    pair = p0['link'].argmax(1)
    proc = (torch.gather(p0['integ'], 1, pair) == 0) & (torch.gather(p0['rst'], 1, pair) == 0)    # (a resting row is the flow of its own MV: no process to compare;
    ratio, spread = ST.shapes(S, pair, (p0['mvm'] > 0) & proc)                                   # on the larger sizes six pairs in ten are such)
    like = {q: ST.alike(ratio, spread, q) for q in ST.LOOPS}
    T = p0['T']
    check('sim time scales', min(like.values()) >= .005 and float((T < 5).double().mean()) > .05 and float((T > 45).double().mean()) > .05,
          f"the unit's time scale is {float(T.quantile(.01)):.1f} to {float(T.quantile(.99)):.0f} decisions (1 % and 99 %); {100*float((p0['store'] > 0).double().mean()):.0f}% of the units have a "
          f'storage behind them; paired process responses within a factor 2 of a BOPTEST heating loop in dead time over time constant and in time to 63 % over time to 20 %: '
          + ', '.join(f'{q} {100*v:.1f}%' for q, v in like.items()))
    # 3 constraints are priced above tracking, in the same physical unit (a band)
    wmax = max(float(s['cvw'].max()) for s in segs)
    check('sim limits priced', 10. > wmax, f'one band over a limit costs 10 a decision, one band off a target at most {wmax:.1f}')
    # 7 required actuator resolution: 0.1 % of span moves no tracked CV by more than a band; a sticking valve neither
    g = max(float((i['G'].abs()*((s['trk'][:, :NMV, None] > 0) & (s['mvm'] > 0)[:, None, :])).max()) for i, s in zip(infos, segs))
    st = max(float((i['G'].abs()*((s['trk'][:, :NMV, None] > 0) & (s['mvm'] > 0)[:, None, :])*s['fs'][:, None, :]).max()) for i, s in zip(infos, segs))
    check('sim actuator resolution', g*1e-3 <= 1. and st <= PR.STICK_MAX+1e-9, f'largest gain {g:.0f} bands per span (0.1 % = {g*1e-3:.2f} band); largest stick {st:.2f} band')
    # 4 the score is not an actuator floor: the references are far apart on the same episodes, none fails
    prog = C.build(n, seed+1, **kw)
    res = {q: CT.rollout(prog, c) for q, c in dict(hold=CT.Hold(), pi=CT.PI(), dmc=CT.Teacher('fixed'), now=CT.Teacher('now')).items()}
    sc = {q: v['score'] for q, v in res.items()}
    gap = float((sc['pi']-sc['now']).median()/sc['now'].median().clamp_min(1.))
    fails = {q: int(v['failed'].sum()) for q, v in res.items() if q != 'hold'}
    check('sim not a floor', gap > .1 and float((sc['now'] < sc['pi']).double().mean()) > .7,
          f"formula PI minus exact-model DMC, paired median {100*gap:.0f}% of the DMC's median; DMC better on {100*float((sc['now'] < sc['pi']).double().mean()):.0f}% of the episodes")
    check('sim references qualify', float(sc['pi'].mean()) < float(sc['hold'].mean()) and float(sc['dmc'].mean()) < float(sc['pi'].mean()) and not any(fails.values()),
          'decisions outside, mean: ' + ', '.join(f'{q} {float(v.mean()):.0f}' for q, v in sc.items()) + f'; physical failures {fails}')
    # the scale a learner is read on (aio.scale): 0 the MVs left where they are, 100 the best causal controller per episode.
    # It has room, and the references sit on it in the order of what they know
    from aio import scale as SC
    per = {q: v.tolist() for q, v in sc.items()}
    scl = SC.Scale.of(per)
    room, rp, rd, rn = 1-sum(scl.best)/max(sum(scl.hold), 1.), scl(per['pi']), scl(per['dmc']), scl(per['now'])
    check('sim scale', room > .3 and 0 < rp < rd < rn <= 100.,
          f"from the MVs left where they are (0) to the best reference per episode (100; that way is {100*room:.0f}% of hold's score): the formula PI "
          f"{rp:.0f}, the step-test DMC {rd:.0f}, the exact-model DMC {rn:.0f}. The Oracle (aio.sim.oracle) lowers the 100 further")
    # 4, 5 at a constrained operating point the reference is the reachable optimum, not the targets: with a controller that
    # finds it, a settled constrained segment scores like any other settled segment; one that does not find it scores worse
    cprog = C.build(n, seed+3, exc_p=1., **kw)
    exc = PR.is_excursion(cprog.kinds)
    ordn = (cprog.kinds >= 0) & (cprog.kinds < PR.OVER_SP)
    cres = {q: CT.rollout(cprog, c) for q, c in dict(pi=CT.PI(), now=CT.Teacher('now')).items()}
    share = lambda q, g: float(cres[q]['seg_settled'][g].sum()/cres[q]['seg_rows'][g].sum().clamp_min(1))    # noqa: E731
    tc, to, pc = share('now', exc), share('now', ordn), share('pi', exc)
    by = {PR.EVENTS[kd]: share('now', cprog.kinds == kd) for kd in PR.EXCURSIONS}          # every way of getting there
    check('sim constrained points', int(exc.sum()) >= 20 and tc <= to+.08 and tc < .6*pc and max(by.values()) <= to+.12
          and not bool(cres['now']['failed'].any()),
          f'{int(exc.sum())} constrained segments in {n} episodes: row-decisions outside once the unit can have lined out, exact-model DMC '
          f'{100*tc:.0f}% (its other segments {100*to:.0f}%; ' + ', '.join(f'{q} {100*v:.0f}%' for q, v in by.items()) + f'), formula PI {100*pc:.0f}%')
    # fat plants: a resting value ranks below every process CV. At a constrained point where the spare MVs have the travel,
    # no process CV gives; a controller that finds the point keeps the process CVs as it does elsewhere and has the spare MVs
    # where they belong; the formula PI, which holds every resting value, loses the CVs
    gcv = torch.cat([(((s['rtgt']-t).abs()/s['bandf']*s['trk']*(1-s['rst'])).amax(-1))[PR.is_excursion(kinds[:, e]) & fat] for e, (s, t) in enumerate(zip(segs, tgts))])
    cfat = (cprog.c['spr'].sum(-1) > 0)[:, None] & exc
    part = lambda q, a, b, g: float(cres[q][a][g].sum()/cres[q][b][g].sum().clamp_min(1))             # noqa: E731
    cv = lambda q, g: float((cres[q]['seg_settled']-cres[q]['seg_rest'])[g].sum()/(cres[q]['seg_rows']-cres[q]['seg_rest_rows'])[g].sum().clamp_min(1))   # noqa: E731
    tr, tf, pf, tpush = part('now', 'seg_rest', 'seg_rest_rows', cfat), cv('now', cfat), cv('pi', cfat), cv('now', cfat & (cprog.kinds == PR.PUSH))
    check('sim spare MVs', gcv.numel() >= 20 and float((gcv < .01).double().mean()) > .4 and int(cfat.sum()) >= 10 and tr < .1 and tf < .6*pf and tpush <= to+.08,
          f'{gcv.numel()} constrained points on fat plants, no process CV gives at {100*float((gcv < .01).double().mean()):.0f}% of them (the spare MVs take it); '
          f'{int(cfat.sum())} such segments in the {n} episodes, once lined out: exact-model DMC has its resting values outside {100*tr:.0f}% and its process CVs '
          f'{100*tf:.0f}% of the decisions (a resting value pushed to a limit: {100*tpush:.0f}%), formula PI its process CVs {100*pf:.0f}%')
    # 6 scored on the true values: a plant at rest scores 0 whatever its instruments report
    quiet = C.build(64, seed+2, **kw)
    quiet.seg_k[:, 1:] = PR.BIG; quiet.k_first[:] = 0
    for q in ('ou_s', 'od_s', 'sin_amp'):                       # the process stands still, the instruments do not
        quiet.c[q].zero_()
    st0 = C.reset(quiet)
    z = torch.zeros(64, NMV, dtype=quiet.dtype)
    rest = (quiet.start['u'] == quiet.start['un']).all(1)       # episodes that start on their steady point
    out_n, moved = 0., 0.
    for k in range(60):
        st0, out = C.step(quiet, st0, z, k)
        out_n += float((out['outside'].sum(-1)*rest).sum())
        moved = max(moved, float((((st0['y']-st0['yt']).abs()/quiet.c['bandf'])*quiet.c['trk']*rest[:, None]).max()))
    check('sim scored on true values', out_n == 0 and moved > 1.,
          f'{out_n:.0f} row-decisions outside in 60 decisions at rest ({int(rest.sum())} episodes) while a measurement is off by up to {moved:.1f} bands')
    # the teacher's prediction is the plant's own when its model is renewed: with the valves left alone from there, the rows
    # go where it says, whatever kind of process change it was, the valves standing still until then or being moved. Plants
    # whose clock stands (one that follows a load runs at another pace once the valves have moved: reported, not judged),
    # valves without friction (the prediction takes a valve to follow its command; with friction: reported)
    m = max(12, n//2*PR.NMV//NMV)
    qs = torch.tensor([.9, .99], dtype=torch.float64)
    floor, exact, ramped, landed, slow = None, {}, {}, {}, []
    for kind in (None,)+tuple(q for q in PR.PROCESS if q != 'friction'):
        for moving in (False, True):
            r = prediction(m, seed+5, size, kind, moving, friction=False)
            on = r['rows'] & (r['has'] if kind is not None else ~r['has'])[:, None] & ~r['mech'][:, None]
            e = r['err'].double()
            slow.append(e[on & r['slow'][:, None]])
            fixed = on & ~r['slow'][:, None]
            if kind is None:
                floor = floor if not moving else torch.quantile(e[fixed], qs)
                continue
            follows = ~r['gradual'] if kind in RAMPED else torch.ones_like(r['gradual'])
            exact[kind, moving] = torch.quantile(e[fixed & follows[:, None]], qs)
            if kind in RAMPED and bool((fixed & ~follows[:, None]).any()):
                ramped[kind, moving] = torch.quantile(e[fixed & ~follows[:, None]], qs)
                r = prediction(m, seed+5, size, kind, moving, arrived=True, friction=False)
                landed[kind, moving] = torch.quantile(r['err'].double()[r['rows'] & (r['has'] & r['gradual'] & ~r['slow'] & ~r['mech'])[:, None]], qs)
    worst = {mv: max(((q, v) for (q, w), v in exact.items() if w == mv), key=lambda x: float(x[1][1])) for mv in (False, True)}
    sl = torch.quantile(torch.cat(slow), qs)
    say = lambda v: f'{float(v[0]):.2f} / {float(v[1]):.2f}'      # noqa: E731
    late = max(landed.items(), key=lambda x: float(x[1][1]))
    stuck = {}
    for kind in (None, 'gain', 'time'):                           # with the valves' friction
        r = prediction(m, seed+5, size, kind, True)
        on = r['rows'] & (r['has'] & ~r['gradual'] if kind is not None else ~r['has'])[:, None] & ~r['mech'][:, None] & ~r['slow'][:, None]
        stuck[kind] = torch.quantile(r['err'].double()[on], qs)
    check("sim teacher's prediction", all(float(v[0]) < .1 and float(v[1]) < .5 for v in list(exact.values())+list(landed.values())) and all(float(v[0]) < 2.5 for v in ramped.values()),
          f'{m} episodes per kind of process change, largest distance over the 150 decisions after the renewal, bands, 90 % / 99 % of the process rows: no event and '
          f'the valves moved {say(floor)}, valves without friction; a change at once (and a gradual one of a gain, an actuator, an output curve), worst kind: the valves still '
          f'{say(worst[False][1])} ({worst[False][0]}), moved {say(worst[True][1])} ({worst[True][0]}); a gradual change of a time constant, a dead time, a row lag or a valve '
          f'curve, read while it is on its way (taken at its end or as it comes): '
          + ', '.join(f"{q} {say(ramped[q, False])} still, {say(ramped[q, True])} moved" for q in RAMPED if (q, True) in ramped)
          + f'; read once it has arrived, worst kind {say(late[1])} ({late[0][0]}, {"moved" if late[0][1] else "still"}); plants whose clock follows a load, all of the above {say(sl)}; '
          f'with the valves\' friction (one held by it is taken to follow its command), the valves moved: no event {say(stuck[None])}, a gain at once {say(stuck["gain"])}, '
          f'time constants at once {say(stuck["time"])}')
    if not default:
        return
    # 1, 4 mechanistic episodes (aio.sim.mech): every operating point is a steady state of the plant itself, and the references
    # rank on every plant as they do on the pair bank
    mprog = C.build(max(n//2, 40), seed+4, mech=1.)
    drift, names = 0., mprog.mech.names()
    for g, (a, P, xs, ws) in zip(mprog.mech.groups, mprog.mech.parts):
        for e in range(PR.NSEG):
            x, w, mp = xs[:, e], ws[:, e], P[:, e]
            y0 = g.M.y(x, w, mp)
            drift = max(drift, float(((g.M.y(g.M.advance(x, w, mp), w, mp)-y0).abs()/g.t['oband'])[g.t['orow'] < NCV].max()))
    mres = {q: CT.rollout(mprog, c) for q, c in dict(hold=CT.Hold(), pi=CT.PI(), now=CT.Teacher('now')).items()}
    ms = {q: v['score'] for q, v in mres.items()}
    mfail = {q: int(v['failed'].sum()) for q, v in mres.items()}
    per = {q: [float(ms[c][torch.tensor([j for j, w in enumerate(names) if w == q])].mean()) for c in ('hold', 'now')] for q in sorted(set(names))}
    check('sim mechanistic plants', drift < 1e-3 and not any(mfail.values()) and all(b < a for a, b in per.values())
          and float(ms['now'].mean()) < float(ms['pi'].mean()) < float(ms['hold'].mean()),
          f'{mprog.n} episodes on {len(per)} plants of the engine: an operating point moves at most {drift:.1e} band in a decision when left alone; decisions outside, mean: '
          + ', '.join(f'{q} {float(v.mean()):.0f}' for q, v in ms.items()) + f'; exact-model DMC better than no control on every plant; physical failures {mfail}')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=128); ap.add_argument('--size', default='12x5', choices=list(PR.SIZES))
    a = ap.parse_args()
    rows = []

    def check(name, ok, detail):
        rows.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    run(check, a.n, size=a.size)
    print(f'{sum(rows)}/{len(rows)} checks passed')
    sys.exit(0 if all(rows) else 1)


if __name__ == '__main__':
    main()

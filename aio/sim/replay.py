"""One episode of the core under every reference controller, as JSON (for the research site's replay and for
looking at what the plants and the controllers do).

  python -m aio.sim.replay --out replay.json                       # hold, formula PI, accurate-model DMC (instruments alone), teacher
  python -m aio.sim.replay --out replay.json --ckpt RUN/best.pt    # and the network

All controllers run the same episode: same plant, same events, same noise (AGENTS.md, site rule 2). The
episode is the one of the batch (--n episodes of --seed, a seed below 2^32: never a training batch) whose
step-test DMC score is the median among those that reach a constrained operating point (about a quarter;
--pick median: among all of them); --index picks another. Every plant of the core is a new unit, so the file
carries its structure (rows, pairing, limits, events) for the screen to be drawn from. Per segment: the zone
limits, priorities, MV limits and, at a constrained operating point, the reachable targets and `allow`, the
bands a zone CV may be beyond its limit before it counts (0 unless the best steady point has it there); per
decision and tracked CV: ok_lo..ok_hi, the stretch inside which it does not count as outside (the band
around the reachable reference, wider at a constrained point). A tracked row with `rest` true is the
resting value of a spare MV (its `mv`): the row reads that MV's flow (a fat plant, docs/sim.md).
"""
import argparse, json
from pathlib import Path
import torch

from aio.sim import core as C, control as CT, mech as ME, prior as PR

KIND_ZH = dict(setpoint='设定值移动', process='过程变化', both='过程变化并移动设定值', load='不可测负荷阶跃', dv='可测扰动阶跃',
               limit='限值移动', priority='优先级变化', over_setpoint='设定值超出可达范围', over_load='负荷超出阀门能力',
               mv_limit='操作员收窄 MV 限值', zone_limit='区间限值收紧到现值之外', mv_out='MV 退出控制', push='静置值推到阀门限值',
               **{'return': '回到原工况'})
GRADUAL = (PR.PROC, PR.BOTH, PR.LOAD, PR.DV, PR.OVER_LOAD)     # kinds whose change can come gradually (ramp, decisions); and the way back from over_load
MIN_MV = 3                    # replay picks a unit with at least this many MVs and coupled loops
NAME_ZH = dict(hold='不动', pi='公式 PI', dmc='开头拿到准确模型的 DMC（只看仪表）', teacher='老师（精确模型 DMC，仅训练用）', aio='AIO')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=2_000_000)
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--index', type=int)
    ap.add_argument('--pick', choices=('constrained', 'median', 'level'), default='constrained',
                    help='median DMC score among the episodes with a constrained operating point, among those with a level (an integrating CV), or among all')
    ap.add_argument('--ckpt', type=Path)
    ap.add_argument('--plant', help='a mechanistic plant (aio.sim.plants: tep, bsm1, column, ...): every episode on it')
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    dev = torch.device(a.device)
    prog = C.build(a.n, a.seed, dev, **(dict(gen_device='cpu', mech=1., plants=ME.pool([a.plant])) if a.plant else {}))
    # dmc: the page's upper reference, the DMC handed the exact model once at the start, corrected from the instruments alone (stay.REFS dmc_meter)
    ctl = dict(hold=CT.Hold(), pi=CT.PI(), dmc=CT.Teacher('fixed', meter=True), teacher=CT.Teacher('late'))
    source = None
    if a.ckpt:
        from aio import net as N                                  # any AIO checkpoint: with the pair pathway or from before it
        net, ck = N.load(a.ckpt, dev)
        ctl['aio'] = N.Policy(net)
        # which checkpoint (its last three path parts) and, for an aio.train one, its validation on the core (x_pi selects best.pt)
        val = ck.get('val') if isinstance(ck.get('val'), dict) else {}
        source = dict(file='/'.join(a.ckpt.resolve().parts[-3:]), trainer=ck['config'].get('trainer') if isinstance(ck.get('config'), dict) else None,
                      iter=ck.get('iter', ck.get('it', ck.get('iteration'))), samples=ck.get('samples'), hours=ck.get('hours'),
                      x_pi=val.get('x_pi'), val={q: float(v) for q, v in val.items() if isinstance(v, (int, float))})
    i, pick, pick_zh = a.index, 'chosen by index', '按序号指定'
    if i is None:
        sc = CT.rollout(prog, ctl['dmc'])['score']
        pool = torch.arange(a.n, device=sc.device)
        exc = PR.is_excursion(prog.kinds).any(1)
        # the page shows a representative APC unit (Mercy 11-10: "怎么只有一对mvcv"): at least MIN_MV MVs, coupled loops
        mimo = (prog.c['mvm'].sum(1) >= MIN_MV) & (prog.c['arch'].long() != PR.ARCH.index('decoupled'))
        if bool(mimo.any()):
            pool = pool[mimo]
        level = (prog.c['integ'] * prog.c['rowm']).sum(1) > 0
        if a.pick == 'level' and bool(level[pool].any()):
            pool = pool[level[pool]]
            pick, pick_zh = 'median DMC score of the coupled units with 3+ MVs and a level (an integrating CV)', '至少 3 个 MV、回路耦合、带液位（积分量）的回合里 DMC 成绩居中的一个'
        elif a.pick == 'constrained' and bool(exc[pool].any()):
            pool = pool[exc[pool]]
            pick, pick_zh = 'median DMC score of the coupled units with 3+ MVs that reach a constrained operating point', '至少 3 个 MV、回路耦合、到过约束工况的回合里 DMC 成绩居中的一个'
        else:
            pick, pick_zh = 'median DMC score of the batch', '该批回合中 DMC 成绩居中的一个'
        i = int(pool[torch.argsort(sc[pool])[len(pool)//2]])
    c = prog.c
    rows, nm = int(c['rowm'][i].sum()), int(c['mvm'][i].sum())
    rnd = lambda x, d=5: [round(float(v), d) for v in x]         # noqa: E731
    role = lambda r: 'tracked' if c['trk'][i, r] > 0 else 'zone' if c['zon'][i, r] > 0 else 'dv'   # noqa: E731
    pair = c['link'][i, :, :prog.nmv].argmax(0)
    gradual = lambda e: int(prog.kinds[i, e]) in GRADUAL or (int(prog.kinds[i, e]) == PR.RETURN and int(prog.kinds[i, e-1]) == PR.OVER_LOAD)   # noqa: E731
    out = dict(seed=a.seed, n=a.n, index=i, horizon=PR.HORIZON, dt_s=float(c['dt'][i]), k_first=int(prog.k_first[i]),
               pick=pick, pick_zh=pick_zh,
               struct=dict(nm=nm, nz=int(c['zon'][i].sum()), nd=int(c['dvr'][i].sum()), nq=int(c['dact'][i, PR.ND:].sum()),
                           arch=PR.ARCH[int(c['arch'][i])], time_scale=round(float(c['T'][i]), 1), dead_ratio=round(float(c['rth'][i]), 2),
                           linear=bool(c['linear'][i] > 0), load_dependent=bool(c['warp'][i] > 0),
                           spare=int(c['spr'][i].sum()),
                           rows=[dict(role=role(r), integrating=bool(c['integ'][i, r] > 0), band=round(float(c['bandf'][i, r]), 4),
                                      mv=next((j for j in range(nm) if int(pair[j]) == r), None), rest=bool(c['rst'][i, r] > 0)) for r in range(rows)],
                           mvs=[dict(slew=round(float(c['slew'][i, j]), 4), sign=int(c['psign'][i, j]), row=int(pair[j])) for j in range(nm)]),
               events=[dict(k=int(prog.seg_k[i, e]), kind=PR.EVENTS[int(prog.kinds[i, e])], kind_zh=KIND_ZH[PR.EVENTS[int(prog.kinds[i, e])]],
                            ramp=int(prog.ramp[i, e]) if gradual(e) else 0) for e in range(1, prog.seg_k.shape[1]) if int(prog.kinds[i, e]) >= 0],
               segments=[], controllers={}, source=source, plant=a.plant)
    if a.plant:                                                   # the plant's own names of the MV slots and rows (the episode puts them in random order)
        g, k = _group(prog, i)
        P = g.M.plants[int(g.t['pid'][k])]
        chan, orow = g.t['chan'][k], g.t['orow'][k]
        mv_name = {int(chan[w]): P.ws[w]['name'] for w in range(P.nw) if P.ws[w]['kind'] == 'mv'}
        row_name = {int(orow[y]): P.ys[y]['name'] for y in range(P.ny)}
        dv_name = {int(chan[w]) - prog.nmv: P.ws[w]['name'] for w in range(P.nw) if P.ws[w]['kind'] == 'load'}
        for r in range(rows):
            if c['dvr'][i, r] > 0:
                row_name[r] = dv_name.get(int(c['dvsel'][i, r]))
        for j, m in enumerate(out['struct']['mvs']):
            m['name'] = mv_name.get(j)
        for r, w in enumerate(out['struct']['rows']):
            w['name'] = row_name.get(r)
    for e in range(prog.seg_k.shape[1]):
        if int(prog.seg_k[i, e]) < PR.BIG:
            t = prog.task(torch.full((prog.n,), e, dtype=torch.long, device=dev))
            lim = lambda x: [None if abs(float(v)) > 1e5 else round(float(v), 5) for v in x[i, :rows]]   # noqa: E731
            allow = [round(float(t['ahi'][i, r]), 3) if c['zon'][i, r] > 0 else 0. for r in range(rows)]
            out['segments'].append(dict(k=int(prog.seg_k[i, e]), lo=lim(t['lo']), hi=lim(t['hi']), weight=rnd(t['cvw'][i, :rows], 3),
                                        u_lo=rnd(t['ulo'][i, :nm], 4), u_hi=rnd(t['uhi'][i, :nm], 4), reachable=rnd(t['rtgt'][i, :rows]), allow=allow,
                                        constrained=bool(((t['rtgt']-t['tgt']).abs()[i] > 1e-9).any()) or any(allow)))
    for name, ct in ctl.items():
        r = CT.rollout(prog, ct, trace=[i])
        tr = r['trace']
        out['controllers'][name] = dict(name_zh=NAME_ZH[name], score=round(float(r['score'][i])), cost=round(float(r['cost'][i]), 1), failed=bool(r['failed'][i]),
                                        y=[rnd(s['y'][0][:rows]) for s in tr], ym=[rnd(s['ym'][0][:rows]) for s in tr],
                                        target=[rnd(s['target'][0][:rows]) for s in tr], ref=[rnd(s['ref'][0][:rows]) for s in tr],
                                        ok_lo=[rnd(s['ok_lo'][0][:nm]) for s in tr], ok_hi=[rnd(s['ok_hi'][0][:nm]) for s in tr],
                                        u=[rnd(s['u'][0][:nm], 4) for s in tr],
                                        outside=[s['outside'][0][:rows] for s in tr])   # each row outside its band or limit, as the score counts
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out))
    print(json.dumps(dict(index=i, struct={q: v for q, v in out['struct'].items() if q not in ('rows', 'mvs')}, events=len(out['events']),
                          scores={q: v['score'] for q, v in out['controllers'].items()})))


def _group(prog, i):
    """The mechanistic group of episode i and its index there."""
    e = i - prog.nl
    for g, (first, *_) in zip(prog.mech.groups, prog.mech.parts):
        if first <= e < first + g.k:
            return g, e - first
    raise IndexError(i)


if __name__ == '__main__':
    main()

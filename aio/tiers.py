"""One set of weights, one copy per time scale (docs/sim.md, section 十七).

One decision period covers time constants of 2.5 to 90 decisions. A plant whose loops are further apart than
that (flows and pressures answer in seconds, temperatures and compositions in minutes to hours) is run as
tiers: every tier is the same network with its own memory and its own period, on the rows and MVs of its
time scale. The tiers are joined through the packet as it is; nothing is added to it and nothing is retrained:

  target   an MV of a slower tier may be the target of a row of a faster one (a flow, a pressure): the slow
           tier writes the target, the fast tier writes the valve. There is no PID anywhere.
  limits   when the valves paired with that row are at their limits on the side the target asks for, the
           slow tier's limit of that MV comes in to one band beyond the row's measurement (`narrow`): the
           target stops where the plant can follow and comes back at the slew limit, which is what an
           industrial MPC does with a saturated loop under it.
  loads    what one tier measures and another moves can be a measured-disturbance row (dv) of the first.

`Tiers` runs one plant on the public packet (aio.interface.encode). `Pair` is the same
on the exam of aio.sim.rates, which holds the loop itself: two copies of one controller and the limit rule.

    python -m aio.tiers WEIGHTS --n 128 --seed 7 --out reports/sim/tiers-qtank.json
runs a checkpoint on that exam: in tiers (with the limit rule and without), one slow period writing the valves,
and with the flows delivered as asked; points of the slow rows and of the flows, and the share of the way from
"writes the valves" to "flows as asked" that the tiers make up (the criterion of docs/sim.md, section 十七).
"""
import torch

from aio.sim.cascade import ROOM, narrow, Tiers, Pair, _Fast, _Slow  # noqa: F401


def exam(net, n=128, seed=7, plant='qtank', log=print, sag=0.):
    """A network on the two-rate exam, on this machine's own scale. Returns the rows (run, slow points, flow
    points, share of the flow row-ticks outside) of the references and of the network, and `made_up`: of the
    points between the network writing the valves at the slow period and the network with flows as asked,
    the share its tiers make up (None when the two are within 3 points: nothing to tell apart)."""
    import time
    from aio import net as N
    from aio.sim import rates as RT
    ex = RT.Exam(n, seed, plant, sag=sag)                         # sag: the share of the episodes whose valves are taken to the end of their stroke (the limit rule has something to tell)
    refs = RT.references(ex, log=log)
    rows = [dict(name=r['name'], mode=r['mode'], points=r['points'], flows=r['fpoints'], flows_out=r['fshare']) for r in RT.table(ex, refs)]
    make = lambda: N.Policy(net)                                  # noqa: E731
    got = {}
    for name, mode, pair in (('net tiers', 'tiers', Pair(make)), ('net tiers, limits as given', 'tiers', Pair(make, rule=False)),
                             ('net direct', 'direct', None), ('net ideal', 'ideal', None)):
        t0 = time.time()
        res = RT.rollout(ex, pair.slow, pair.fast, mode) if pair else RT.rollout(ex, make(), mode=mode)
        sp, fp = ex.points(res) if 'fscore' in res else (ex.scale(res['score'].tolist()), None)
        got[name] = sp
        rows.append(dict(name=name, mode=mode, points=sp, flows=fp, failed=int(res['failed'].sum()),
                         flows_out=float(res['fscore'].sum()/res['frows'].sum().clamp_min(1.)) if 'fscore' in res else None,
                         valves_at_end=float(res['fsat'].mean()) if 'fsat' in res else None))
        if log:
            log(f'{name:28s} {time.time()-t0:6.1f} s  slow {sp:6.1f}'+(f'  flows {fp:6.1f}' if fp is not None else ''))
    gap = got['net ideal']-got['net direct']
    return dict(plant=plant, n=n, seed=seed, m=ex.m, sag=sag, rows=rows, gap=gap, made_up=(got['net tiers']-got['net direct'])/gap if gap >= 3 else None,
                rule=got['net tiers']-got['net tiers, limits as given'])


def main():
    import argparse
    import json
    from pathlib import Path
    from aio import net as N
    ap = argparse.ArgumentParser(description='A checkpoint on the two-rate exam (aio.sim.rates): tiers, one slow period, flows as asked.')
    ap.add_argument('weights')
    ap.add_argument('--n', type=int, default=128)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--plant', default='qtank')
    ap.add_argument('--sag', type=float, default=0., help='share of the episodes with a sag of the supply (aio.sim.rates): valves reach the end of their stroke')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    net, _ = N.load(a.weights)
    out = exam(net, a.n, a.seed, a.plant, sag=a.sag)
    out['weights'] = a.weights
    print(f"{'':30s} {'mode':7s} {'slow':>6s} {'flows':>6s} {'flows out':>10s}")
    for r in out['rows']:
        print(f"{r['name']:30s} {r['mode']:7s} {r['points']:6.1f} "+(f"{r['flows']:6.1f} {100*r['flows_out']:9.1f}%" if r['flows'] is not None else ''))
    print(f"direct to ideal: {out['gap']:.1f} points; the tiers make up "+('nothing to tell apart (under 3 points)' if out['made_up'] is None else f"{100*out['made_up']:.0f}%"))
    print(f"the limit rule: {out['rule']:+.1f} points of the slow rows")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

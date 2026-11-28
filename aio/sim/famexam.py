"""Scores per mechanistic plant family, for the research site's chart (one batch of fixed episodes per family).

  python -m aio.sim.famexam --plants tep,bsm1,column --ctl MODULE:FACTORY --arg WEIGHTS --out fam.json [--n 8 --seed 3000000]

Every family gets its own batch (--n episodes of --seed on that plant alone, aio.sim.mech, one window of the core's
horizon); every controller runs the same episodes. Points per family = 100 * (hold - controller) / (hold - teacher),
decisions outside summed over the batch: 0 is the MVs left alone, 100 the controller that knows the true plant (the
training teacher). The network never saw the large plants (plants.LARGE: TEP, BSM1, ...) in training; a held-out
plant (plants.HELDOUT: vam, boiler, ...) is named the same way. A plant with more MVs than the default size gets the
smallest size that holds it (prior.SIZES). `trips`: per controller, the episodes that went past a shutdown limit the
plant declares (Tennessee Eastman's interlocks; reported, not scored) and the decision each first did (null: never).

A --ctl factory (aio.sim.plugin) may return either a single controller or a dict `{name: controller}` (e.g. a set of
ablated variants); all entries are merged into the exam batch.
"""
import argparse, json, time
from pathlib import Path
import torch

from aio.sim import core as C, control as CT, mech as ME, plants as PL, plugin as PG, prior as PR


def _parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plants', required=True)
    ap.add_argument('--ctl', default='', help='MODULE:FACTORY of an external controller or {name: controller} dict (aio.sim.plugin)')
    ap.add_argument('--arg', default=None, help='argument passed to FACTORY (e.g. a weights path)')
    ap.add_argument('--name', default=None, help='controller name when FACTORY returns one controller (defaults to meta["name"] or "aio")')
    ap.add_argument('--ckpt', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--ablate', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--seed', type=int, default=3_000_000)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cpu')
    return ap


def _load_ctl(ap, a, dev):
    if a.ckpt and not a.ctl:
        ap.error('--ckpt has been removed; pass --ctl MODULE:FACTORY --arg PATH instead (aio.sim.plugin)')
    if a.ckpt:
        ap.error('--ckpt has been removed; pass --arg PATH with --ctl MODULE:FACTORY instead')
    if a.ablate:
        ap.error('--ablate has been removed; pass a --ctl MODULE:FACTORY returning a dict of controllers instead')
    ctl = dict(hold=CT.Hold(), pi=CT.PI(), dmc=CT.Teacher('fixed', meter=True), teacher=CT.Teacher('late'))
    if a.ctl:
        built, meta = PG.build(a.ctl, a.arg, device=dev)
        if isinstance(built, dict):
            ctl.update(built)
        else:
            cname = a.name or (meta.get('name') if isinstance(meta, dict) else None) or 'aio'
            ctl[cname] = built
    return ctl


def main(argv=None):
    ap = _parser()
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    ctl = _load_ctl(ap, a, dev)
    out = json.loads(a.out.read_text()) if a.out.exists() else {}
    for q in a.plants.split(','):
        if q in out:
            continue
        t0 = time.time()
        plants = ME.pool([q], heldout=q in PL.HELDOUT)
        nmv = sum(w['kind'] == 'mv' for w in plants[0].ws)
        dims = next(k for k, d in PR.SIZES.items() if d[1] >= nmv)
        prog = C.build(a.n, a.seed, dev, gen_device='cpu', mech=1., plants=plants, dims=dims)
        res = {k: CT.rollout(prog, c) for k, c in ctl.items()}
        per = {k: [float(x) for x in r['score']] for k, r in res.items()}
        trips = {k: dict(episodes=int(r['tripped'].sum()), first=[int(x) if x < CT.BIG else None for x in r['trip_k']]) for k, r in res.items()}
        hold, best = sum(per['hold']), sum(per['teacher'])
        pts = {k: round(100*(hold-sum(v))/(hold-best), 1) if hold > best else None for k, v in per.items()}
        out[q] = dict(n=a.n, seed=a.seed, points=pts, per=per, trips=trips, seconds=round(time.time()-t0))
        a.out.write_text(json.dumps(out, indent=1))
        print(q, pts, 'tripped', {k: v['episodes'] for k, v in trips.items()}, out[q]['seconds'], 's', flush=True)


if __name__ == '__main__':
    main()

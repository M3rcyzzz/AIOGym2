"""Scores per mechanistic plant family, for the research site's chart (one batch of fixed episodes per family).

  python -m aio.sim.famexam --plants tep,bsm1,column --ckpt RUN/SEED/last.pt --out fam.json [--n 8 --seed 3000000]

Every family gets its own batch (--n episodes of --seed on that plant alone, aio.sim.mech, one window of the core's
horizon); every controller runs the same episodes. Points per family = 100 * (hold - controller) / (hold - teacher),
decisions outside summed over the batch: 0 is the MVs left alone, 100 the controller that knows the true plant (the
training teacher). The network never saw the large plants (plants.LARGE: TEP, BSM1, ...) in training; a held-out
plant (plants.HELDOUT: vam, boiler, ...) is named the same way. A plant with more MVs than the default size gets the
smallest size that holds it (prior.SIZES). `trips`: per controller, the episodes that went past a shutdown limit the
plant declares (Tennessee Eastman's interlocks; reported, not scored) and the decision each first did (null: never).

--ablate (11-15: what the fixed operators beside the network are worth): the same weights also run as `aio_law` (the
move is the product of the law's read-outs alone, K.E/V, the network's own output left out), `aio_nolaw` (the product
not given to the MVs), `aio_nostats` (the running sums' features not given to the pairs) and `aio_neither` (both). The
weights were trained with both, so the last three say how much the network leans on them, not what one trained
without them would do.
"""
import argparse, json, time
from pathlib import Path
import torch

from aio.sim import core as C, control as CT, mech as ME, plants as PL, prior as PR


def ablated(ckpt, dev):
    """The controllers of --ablate, each on its own copy of the weights."""
    from torch import nn
    from aio import net as N
    from aio.actor import TOKEN

    class Zero(nn.Module):
        def __init__(self, out):
            super().__init__()
            self.o = out

        def forward(self, x):
            return x.new_zeros(x.shape[:-1]+(self.o,))

    class LawOnly(N.Policy):
        @torch.no_grad()
        def __call__(self, prog, st, k, obs):
            from aio.frame import canonical
            o, s = canonical(obs.float(), torch.ones(prog.n, dtype=torch.bool, device=obs.device) if self.unpaired else None)
            mask, self.h = self.net.steps(o[:, None], self.prev[:, None], self.h, None)[2::2]
            self.prev = self.net.pairs.read_law[3][:, 0].clamp(-1., 1.).float()*mask[:, 0]
            return (self.prev*s).to(prog.dtype)

    out = {}
    for how in ('law', 'nolaw', 'nostats', 'neither'):
        net = N.load(ckpt, dev)[0]
        p = net.pairs
        assert p.law and p.stats is not None and not p.apart, '--ablate: weights with the running sums and the law\'s product read'
        if how in ('nostats', 'neither'):
            p.stat = Zero(N.P)
            if hasattr(p, 'stat2'):
                p.stat2 = Zero(N.P)
        if how in ('nolaw', 'neither'):
            p.law_in = Zero(TOKEN)
        out['aio_'+how] = LawOnly(net) if how == 'law' else N.Policy(net)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plants', required=True)
    ap.add_argument('--ckpt', type=Path)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--seed', type=int, default=3_000_000)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--ablate', action='store_true', help='also the same weights with the law\'s move alone, and with the law\'s product, the running sums or both taken from the network')
    a = ap.parse_args()
    dev = torch.device(a.device)
    ctl = dict(hold=CT.Hold(), pi=CT.PI(), dmc=CT.Teacher('fixed', meter=True), teacher=CT.Teacher('late'))
    if a.ckpt:
        from aio import net as N
        net, _ = N.load(a.ckpt, dev)
        ctl['aio'] = N.Policy(net)
        if a.ablate:
            ctl.update(ablated(a.ckpt, dev))
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

"""Where the simulator gradient of the trainer blows up (a diagnosis, nothing is trained).

  python -m aio.gradprobe --init RUN/last.pt --n 192 --seed 3 --device cuda --out probe.json

The trainer's differentiated chunk (aio.train.simulate), on the same kind of batch, with the weights left
alone. For every chunk it measures the norm of the cost gradient on the weights (the trainer's sim_gn) and,
per episode, the gradient arriving at the network's moves, in several variants of the backward pass:

  as_trained    straight-through gradients at the MV limits and through valve friction
  exact         none of that (step(through=False)): the gradient of what the plant really does
  open          as trained, but the packet carries no gradient (no loop through plant -> packet -> network)
  decay         as trained, the gradient through the plant state shrinks by --decay per decision
  cap           as trained, each episode's gradient at each move held to --cap times the batch median

and what the worst episodes have in common (MVs at a limit, moves saturated, unstable rows, constrained
segments). The forward pass is the same in every variant, so the variants differ in the gradient alone.
"""
import argparse, json, math
from pathlib import Path
import torch

from aio.actor import Actor, STATE
from aio.interface import NMV
from aio.frame import canonical
from aio.sim import core as C, prior as PR
from aio.sim.prior import HORIZON

VARIANTS = ('as_trained', 'exact', 'open', 'decay', 'cap')


def chunk(actor, prog, st, h, prev, c0, a, variant):
    """One chunk from (st, h, prev). Returns the weight-gradient norm, the per-move gradients (steps, n, NMV),
    the per-episode features, and the state after the chunk (detached)."""
    n, dev = prog.n, prog.device
    params = [p for p in actor.parameters()]
    grads, keep = {}, []
    csum = torch.zeros(n, device=dev)
    lim = torch.zeros(n, device=dev); sat = torch.zeros(n, device=dev)
    c1 = min(c0+a.chunk, HORIZON)

    def hook(j):
        def f(g):
            g = torch.where(torch.isfinite(g).all(-1, keepdim=True), g, torch.zeros_like(g))
            grads[j] = g.detach()
            if variant == 'cap':
                nrm = g.norm(dim=-1, keepdim=True)
                top = a.cap*nrm[nrm > 0].median() if bool((nrm > 0).any()) else nrm.new_tensor(math.inf)
                g = g*(top/nrm.clamp_min(1e-30)).clamp_max(1.)
            return g
        return f
    for k in range(c0, c1):
        obs, s = canonical(C.observe(prog, st, k).float())
        if variant == 'open':
            obs = obs.detach()
        joint, mem, mask, h = actor.history(obs[:, None], prev[:, None], h)
        mean, _, _ = actor.distribution(joint, mem, mask)
        mask = mask[:, 0]
        u = torch.tanh(mean[:, 0])*mask
        act = (u if k < c1-a.tail else u.detach())*s
        if act.requires_grad:
            act.register_hook(hook(k-c0))
        t = prog.task(prog.seg_at(k))
        st, out = C.step(prog, st, act.to(prog.dtype), k, through=variant != 'exact')
        if variant == 'decay':
            st = {q: (a.decay*v+(1-a.decay)*v.detach()) if v.requires_grad else v for q, v in st.items()}
        csum = csum+out['soft'].float()*(~st['fail']).float()
        m = prog.c['mvm']
        lim += (((st['u'] <= t['ulo']+1e-6) | (st['u'] >= t['uhi']-1e-6)).float()*m).sum(-1)/m.sum(-1).clamp_min(1)
        sat += ((u.abs() > .99).float()*mask).sum(-1)/mask.sum(-1).clamp_min(1)
        prev = u
    steps = c1-c0
    l_sim = (csum/(csum.detach()+a.sim_floor*steps)).mean()
    gs = torch.autograd.grad(l_sim, params, allow_unused=True)
    gn = torch.sqrt(sum(g.square().sum() for g in gs if g is not None))
    G = torch.stack([grads.get(j, torch.zeros(n, NMV, device=dev)) for j in range(steps)])
    flat = torch.cat([g.flatten() for g in gs if g is not None])
    return float(gn), G, dict(cost=csum.detach()/steps, lim=lim/steps, sat=sat/steps), (C.detach(st), h.detach(), prev.detach()), flat


def q(x, ps=(.1, .5, .9, .99, 1.)):
    x = x[torch.isfinite(x)].float()
    return [float(f'{v:.3g}') for v in torch.quantile(x, torch.tensor(ps, device=x.device))] if x.numel() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', type=Path)
    ap.add_argument('--n', type=int, default=192)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--chunk', type=int, default=120)
    ap.add_argument('--tail', type=int, default=30)
    ap.add_argument('--chunks', type=int, default=6)
    ap.add_argument('--sim-floor', type=float, default=1.)
    ap.add_argument('--decay', type=float, default=.95)
    ap.add_argument('--cap', type=float, default=10.)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--gen-device', default='cpu')
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    dev = torch.device(a.device)
    actor = Actor().to(dev); actor.canonical = True
    if a.init:
        actor.load_state_dict(torch.load(a.init, map_location=dev, weights_only=False)['model']['actor'])
    prog = C.build(a.n, a.seed, dev, a.gen_device)
    prog.set_teacher('late')
    c = prog.c
    unst = ((c['integ'] > 0) & (c['lam'] > 0) & (c['prc'] > 0)).any(-1)
    state = (C.reset(prog), torch.zeros(1, a.n, STATE, device=dev), torch.zeros(a.n, NMV, device=dev))
    res = dict(args={k: str(v) for k, v in vars(a).items()}, chunks=[])
    for ci in range(min(a.chunks, math.ceil(HORIZON/a.chunk))):
        c0 = ci*a.chunk
        row, ref, nxt = dict(c0=c0), None, None
        exc = PR.is_excursion(prog.kinds[prog.ar, prog.seg_at(c0)])
        for v in VARIANTS:
            gn, G, feat, after, flat = chunk(actor, prog, *state, c0, a, v)
            per = G.square().sum((0, 2)).sqrt()                                   # (n,) per episode
            prof = G.square().sum(-1).sqrt()                                      # (steps, n)
            r = dict(gn=float(f'{gn:.4g}'), per_episode=q(per), nonfinite=int((~torch.isfinite(per)).sum()))
            if v == 'as_trained':
                ref, nxt = flat, after
                med = per.median()
                big = per > 100*med
                worst = per.argsort(descending=True)[:5]
                r.update(share_over_100x_median=float(big.float().mean()),
                         share_of_sq_norm_in_top_1pct=float(per.square().topk(max(1, a.n//100)).values.sum()/per.square().sum().clamp_min(1e-30)),
                         big=dict(lim=float(feat['lim'][big].mean()) if bool(big.any()) else None, sat=float(feat['sat'][big].mean()) if bool(big.any()) else None,
                                  unstable=float(unst[big].float().mean()) if bool(big.any()) else None,
                                  constrained=float(exc[big].float().mean()) if bool(big.any()) else None, cost=float(feat['cost'][big].mean()) if bool(big.any()) else None),
                         rest=dict(lim=float(feat['lim'][~big].mean()), sat=float(feat['sat'][~big].mean()), unstable=float(unst[~big].float().mean()),
                                   constrained=float(exc[~big].float().mean()), cost=float(feat['cost'][~big].mean())),
                         worst=[dict(ep=int(i), norm=float(f'{float(per[i]):.3g}'), lim=round(float(feat['lim'][i]), 2), sat=round(float(feat['sat'][i]), 2),
                                     unstable=bool(unst[i]), constrained=bool(exc[i]), cost=round(float(feat['cost'][i]), 1),
                                     log10_by_step=[round(math.log10(float(prof[j, i])+1e-30), 1) for j in range(0, prof.shape[0], 10)]) for i in worst.tolist()])
            else:
                r['cos_to_as_trained'] = float(torch.nn.functional.cosine_similarity(flat.nan_to_num(0., 0., 0.), ref.nan_to_num(0., 0., 0.), 0))
            row[v] = r
            del G, flat
        res['chunks'].append(row)
        print(json.dumps(row), flush=True)
        state = nxt
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

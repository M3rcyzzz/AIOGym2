"""Does a step along the trainer's simulator gradient lower what the plant really charges? (a measurement, nothing is trained)

  python -m aio.stprobe --ckpt RUN/last.pt --n 256 --seed 5 --device cuda --out stprobe.json

The trainer differentiates through an MV limit and through a valve's friction band as if they were not there
(aio.sim.core.step, `through`): where the plant's own derivative is 0 the gradient is not. Here, on one batch,
from decision --start for --window decisions: the gradient of the window's objective on the weights both ways
(`through` and `hard`), then the weights moved a little along each, and the objective of the same window run
again with no graph (the plant as it is). Reported: how far each step lowers it on the batch the gradient is
of and on another batch (--other), over all episodes and over those with an MV on a limit for a fifth of
the window or more.
"""
import argparse, json
from pathlib import Path
import torch

from aio import net as N
from aio.frame import canonical
from aio.sim import core as C, oracle as OR


def run(net, prog, st, h, prev, k0, W, through, unpaired, cap=10., fail_cost=10.):
    """The window from (st, h, prev) -> the objective per decision of every episode (n,), the share of its decisions with an MV on a limit."""
    n, c = prog.n, prog.c
    tot, lim = torch.zeros(n, device=prog.device), torch.zeros(n, device=prog.device)
    g = prog.g.get_state()

    def hold(x):                                                  # an episode's gradient at a move held to `cap` times the batch median (as the trainer's guard)
        nrm = x.norm(dim=-1, keepdim=True)
        med = nrm[nrm > 0].median() if bool((nrm > 0).any()) else nrm.new_tensor(1.)
        return x*(cap*med/nrm.clamp_min(1e-30)).clamp_max(1.)
    for k in range(k0, k0+W):
        o, s = canonical(C.observe(prog, st, k).float(), torch.ones(n, dtype=torch.bool, device=prog.device) if unpaired else None)
        mean, _, mask, _, h = net.steps(o[:, None], prev[:, None], h, None)
        u = torch.tanh(mean[:, 0])*mask[:, 0]
        act = u*s
        if act.requires_grad:
            act.register_hook(hold)
        st, out = C.step(prog, st, act.to(prog.dtype), k, through=through)
        tot = tot+torch.where(out['fail'], torch.full_like(tot, fail_cost), OR.objective(out, st, c).float())
        t = prog.task(prog.seg_at(k))
        on = ((st['u'] <= t['ulo']+1e-6) | (st['u'] >= t['uhi']-1e-6)) & (c['mvm'] > 0)
        lim = lim+on.any(-1).float()
        prev = u
    prog.g.set_state(g)
    return tot/W, lim/W


def start(net, prog, k0, unpaired):
    pol = N.Policy(net)
    st = C.reset(prog)
    pol.reset(prog)
    with torch.no_grad():
        for k in range(k0):
            st, _ = C.step(prog, st, pol(prog, st, k, C.observe(prog, st, k)), k)
    return C.detach(st), pol.h.clone(), pol.prev.clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--other', type=int, default=6)
    ap.add_argument('--start', type=int, nargs='+', default=[200, 400])
    ap.add_argument('--window', type=int, default=64)
    ap.add_argument('--eps', type=float, nargs='+', default=[1e-5, 3e-5, 1e-4, 3e-4], help='length of the step, as a share of the norm of the weights')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.backends.cudnn.enabled = False                         # (a recurrent layer in eval mode has no backward pass there)
    net, _ = N.load(a.ckpt, dev)
    unpaired = N.blind_main(net)
    ps = [p for p in net.parameters() if p.requires_grad] or list(net.parameters())
    for p in ps:
        p.requires_grad_(True)
    wn = float(torch.sqrt(sum(p.detach().square().sum() for p in ps)))
    rows = []
    for k0 in a.start:
        B = [(prog, *start(net, prog, k0, unpaired)) for prog in (C.build(a.n, a.seed, dev), C.build(a.n, a.other, dev))]
        grads, base = {}, []
        for name, through in (('through', True), ('hard', False)):
            prog, st, h, prev = B[0]
            J, lim = run(net, prog, st, h, prev, k0, a.window, through, unpaired)
            (J/(J.detach()+.1)).mean().backward()                 # every episode relative to its own level, as the trainer
            grads[name] = [torch.nan_to_num(p.grad.detach().clone()) if p.grad is not None else torch.zeros_like(p) for p in ps]
            net.zero_grad(set_to_none=True)
        with torch.no_grad():
            for prog, st, h, prev in B:
                J, lim = run(net, prog, st, h, prev, k0, a.window, True, unpaired)
                base.append((J, lim >= .2))
        gn = {q: float(torch.sqrt(sum(x.square().sum() for x in g))) for q, g in grads.items()}
        cos = float(sum((x*y).sum() for x, y in zip(grads['through'], grads['hard']))/(gn['through']*gn['hard']+1e-30))
        row = dict(start=k0, norm=gn, cos=cos, on_limit=[float(b[1].float().mean()) for b in base], base=[float(b[0].mean()) for b in base], steps=[])
        for q, g in grads.items():
            for e in a.eps:
                d = [-e*wn*x/(gn[q]+1e-30) for x in g]
                with torch.no_grad():
                    for p, x in zip(ps, d):
                        p.add_(x)
                    r = dict(grad=q, eps=e)
                    for b, (prog, st, h, prev), where in zip(base, B, ('same', 'other')):
                        J, _ = run(net, prog, st, h, prev, k0, a.window, True, unpaired)
                        rel = (J-b[0])/(b[0]+.1)                  # per episode, relative to its own level
                        r[where] = float(rel.mean())
                        r[where+'_limit'] = float(rel[b[1]].mean()) if bool(b[1].any()) else None
                        r[where+'_free'] = float(rel[~b[1]].mean()) if bool((~b[1]).any()) else None
                    for p, x in zip(ps, d):
                        p.sub_(x)
                row['steps'].append(r)
        rows.append(row)
        print(json.dumps(row), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(ckpt=str(a.ckpt), n=a.n, window=a.window, rows=rows), indent=1))


if __name__ == '__main__':
    main()

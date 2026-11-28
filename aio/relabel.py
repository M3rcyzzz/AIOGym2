"""Weights whose read-outs give values, made into weights whose read-outs give the change from lag to lag
(aio.net, `diff`; --diff of aio.train).

The read-out's own layers are fitted again to say what they said, in the other form: on what the network's trunk
hands them in its own closed loop (plants of the usual menu, memory carried from window to window), towards the
changes from lag to lag of the values the read-out gave there. Nothing is learnt from the simulator's labels here, so
the network that comes out controls as the one that went in (printed: how far the values it now reads are from
the ones it read, and with --check both in closed loop on plants neither has seen); a run from these weights
(--init, with or without --diff: they keep theirs) then teaches the read-out its new labels.

    python -m aio.relabel WEIGHTS OUT.pt --diff S|E|SE [--plants 64] [--windows 2] [--iters 4000] [--check 2]
"""
import argparse, copy, json, time
import torch

from aio import net as N, formula as FM
from aio.sim import control as CT, stay as ST

CLIP = dict(S=FM.S_CLIP, E=N.E_CLIP)
HEAD = dict(S='read', E='heading')


def gather(net, what, plants, windows, seed, every, device):
    """What the read-outs `what` are handed and give in the network's own closed loop: {q: (inputs (N, .), values (N, lags))}."""
    got = {q: ([], []) for q in what}
    step = [0]

    def hook(q):
        def f(mod, inp, out):
            if step[0] % every == 0:
                x, y = inp[0].detach().flatten(0, -2), out.detach().flatten(0, -2)
                on = x.abs().amax(-1) > 0                         # the slots in use (an empty row's token is 0)
                got[q][0].append(x[on].float().cpu()); got[q][1].append(y[on].float().cpu())
            step[0] += q == what[-1]
        return f
    hooks = [getattr(net.pairs, HEAD[q]).register_forward_hook(hook(q)) for q in what]
    ctl, st = N.Policy(net, stay=True), None
    with torch.no_grad():
        for prog in ST.exam('usual', plants, seed, windows, device=device, gen_device='cpu'):
            st = CT.rollout(prog, ctl, st=st)['st']
    for h in hooks:
        h.remove()
    return {q: (torch.cat(x), torch.cat(y)) for q, (x, y) in got.items()}


def fit(head, x, y, clip, iters, bs=4096, lr=1e-3, seed=0):
    """A copy of `head` fitted to give the changes from lag to lag of the values y it gave on x; the last tenth of the
    samples is left out of the fit -> the copy, and on those: the unexplained share of the values (asinh units) it now
    reads against the ones it read, per lag."""
    new, g = copy.deepcopy(head).train(), torch.Generator().manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g)
    tr, te = idx[x.shape[0]//10:], idx[:x.shape[0]//10]
    y = y.clamp(-clip, clip)
    t = N.steps_of(y, clip)
    opt = torch.optim.Adam(new.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=.1)
    for i in range(iters):
        j = tr[torch.randint(0, tr.numel(), (bs,), generator=g)]
        out = new(x[j])
        loss = (out-t[j]).square().mean()+(N.values_of(out, clip)-y[j]).square().mean()     # the changes, and the values they add up to
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(new.parameters(), 1.)
        opt.step(); sched.step()
    with torch.no_grad():
        d = N.values_of(new(x[te]), clip)-y[te]
        left = (d.square().sum(0)/y[te].square().sum(0).clamp_min(1e-12)).tolist()
    return new.eval(), left


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('weights'); ap.add_argument('out')
    ap.add_argument('--diff', required=True, help='S (the responses), E (the E read-out), SE')
    ap.add_argument('--plants', type=int, default=64); ap.add_argument('--windows', type=int, default=2)
    ap.add_argument('--seed', type=int, default=78_201, help='of the plants the fit is made on')
    ap.add_argument('--every', type=int, default=4, help='one decision in so many is kept')
    ap.add_argument('--iters', type=int, default=4000); ap.add_argument('--device', default='cpu')
    ap.add_argument('--check', type=int, default=0, help='windows of closed loop, before and after, on plants of another seed (0: none)')
    ap.add_argument('--threads', type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    net, ck = N.load(a.weights, a.device)
    what = [q for q in 'SE' if q in a.diff and q not in net.pairs.diff]
    assert what and not set(a.diff)-set('SE'), 'nothing to do: --diff S, E or SE, on weights that do not have it'
    assert 'E' not in what or (net.pairs.law and not net.pairs.worked), 'E: the read-out that learns E (--law without --formula)'
    t0 = time.time()
    data = gather(net, what, a.plants, a.windows, a.seed, a.every, a.device)
    pairs, rep = dict(ck['model']['pairs']), {}
    for q in what:
        x, y = data[q]
        head, left = fit(getattr(net.pairs, HEAD[q]), x, y, CLIP[q], a.iters)
        for k, v in head.state_dict().items():
            pairs[f'{HEAD[q]}.{k}'] = v.to(pairs[f'{HEAD[q]}.{k}'].dtype).clone()
        pairs['diff_'+q] = torch.tensor(1.)
        rep[q] = dict(samples=int(x.shape[0]), unexplained=[round(v, 4) for v in left])
        print(json.dumps(dict(readout=q, **rep[q], minutes=round((time.time()-t0)/60, 1))), flush=True)
    out = {k: v for k, v in ck.items() if k not in ('raw', 'opt', 'copt')}           # weights to start a run from (--init), not to resume one
    out['model'] = dict(ck['model'], pairs=pairs)
    out['config'] = dict(ck.get('config') or {}, diff=''.join(q for q in 'SE' if q in net.pairs.diff or q in what))
    out['relabel'] = dict(source=str(a.weights), fit=rep, plants=a.plants, windows=a.windows, seed=a.seed)
    torch.save(out, a.out)
    if a.check:
        new, score = N.load(a.out, a.device)[0], {}
        for name, m in (('before', net), ('after', new)):
            ctl, st, per = N.Policy(m, stay=True), None, []
            with torch.no_grad():
                for prog in ST.exam('usual', a.plants, a.seed+1, a.check, device=a.device, gen_device='cpu'):
                    r = CT.rollout(prog, ctl, st=st)
                    st = r['st']
                    per.append(round(float(r['score'].mean()), 2))
            score[name] = per
        print(json.dumps(dict(check='row-decisions outside the band per plant, by window (lower is better)', **score, minutes=round((time.time()-t0)/60, 1))), flush=True)


if __name__ == '__main__':
    main()

"""Training-side controller factories for the environment's plugin hook (aio.sim.plugin).

Used via `--ctl MODULE:FACTORY --arg PATH` in aio.sim.{bench,replay,famexam,faults} and aio.sim.validate.boptest:

  aio.plugin:policy   a checkpoint (.pt) as aio.net.Policy, with replay source metadata
  aio.plugin:ablated  a checkpoint (.pt) as {aio, aio_law, aio_nolaw, aio_nostats, aio_neither}
  aio.plugin:packet   a released folder or checkpoint (.pt) as aio.net.Controller (.step(packet))
"""
from pathlib import Path

import torch
from torch import nn

from aio import net as N


def _source_meta(path, ck):
    val = ck.get('val') if isinstance(ck.get('val'), dict) else {}
    return dict(
        file='/'.join(Path(path).resolve().parts[-3:]),
        trainer=ck['config'].get('trainer') if isinstance(ck.get('config'), dict) else None,
        iter=ck.get('iter', ck.get('it', ck.get('iteration'))),
        samples=ck.get('samples'),
        hours=ck.get('hours'),
        x_pi=val.get('x_pi'),
        val={q: float(v) for q, v in val.items() if isinstance(v, (int, float))},
    )


def policy(arg, device='cpu', **kw):
    """Load an AIO checkpoint (.pt) and return (aio.net.Policy(net), dict(name='aio', source=...))."""
    if not arg:
        raise ValueError('aio.plugin:policy requires --arg PATH to a checkpoint')
    p = Path(arg)
    net, ck = N.load(p, device)
    return N.Policy(net, **kw), dict(name='aio', source=_source_meta(p, ck))


def _ablated_variants(ckpt, dev):
    """The ablated controllers (law-only, nolaw, nostats, neither), each on its own copy of the weights."""
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


def ablated(arg, device='cpu', **kw):
    """Load an AIO checkpoint (.pt) and return ({aio, aio_law, aio_nolaw, aio_nostats, aio_neither}, dict(name='aio'))."""
    if not arg:
        raise ValueError('aio.plugin:ablated requires --arg PATH to a checkpoint')
    p = Path(arg)
    net, _ = N.load(p, device)
    ctls = dict(aio=N.Policy(net, **kw), **_ablated_variants(p, device))
    return ctls, dict(name='aio')


def packet(arg, device='cpu', **kw):
    """Load a released folder or checkpoint (.pt) and return aio.net.Controller(net) (.step(packet))."""
    if not arg:
        raise ValueError('aio.plugin:packet requires --arg PATH to a released folder or checkpoint')
    p = Path(arg)
    net = N.released(p, device) if p.is_dir() else N.load(p, device)[0]
    return N.Controller(net)

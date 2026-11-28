"""Trajectory check of a --seq checkpoint before a long run (11-23): look at what it does, decision by decision.

  python -m aio.trajcheck CKPT OUT.json [--n 128] [--handed 240] [--device cuda]

On the exam batch (aio.sim.stay.EXAM_SEED), beside nothing moved (0) and the teacher (100):
  told     told which valve goes with which row and which way it acts. Points, and whether it swings: per valve the power of
           its position at periods of 12 to 48 decisions after the first event (`swing`, model over teacher, median over the
           valves that move), the same of the tracked rows' error in bands (`row_swing`), and how often a move turns back (`flip`).
  blind    told nothing of the pairing. Points; `probe`: valve moves before the first event (rms of the step, model; 0 is still);
           `handed`: the teacher drives the first K decisions, then the model has the plant: points after K (does it use a
           history that holds the answer) and `same_side`: share of the valves whose net move over the 30 decisions after K
           goes the teacher's way (where the teacher's is 0.02 of the span or more).
The traces of the plants with a level, told and blind, go to OUT.json.pt (u, y, target, band limits) for plots."""
import argparse, json, time
import torch

from aio import net as N
from aio.frame import canonical
from aio.sim import control as CT, core as C, stay as ST

LO, HI = 12, 48                                                   # periods of the swing, decisions


class Handed(N.Policy):
    """The teacher has the plant for the first K decisions while the network runs alongside on what happens; then the
    network (then='model') or the teacher (then='teacher')."""

    def __init__(self, net, K, then, **kw):
        super().__init__(net, **kw)
        self.K, self.then = K, then

    def __call__(self, prog, st, k, obs):
        a = super().__call__(prog, st, k, obs)
        if k >= self.K and self.then == 'model':
            return a
        t = torch.zeros_like(a) if k >= self.K and self.then == 'hold' else C.teacher(prog, st, k)
        o, s = canonical(obs.float(), torch.ones(prog.n, dtype=torch.bool, device=obs.device) if self.unpaired else None)
        self.prev = (t.float()*s)*N.unpack(o)[1][..., -1]
        return t.to(prog.dtype)


def run(prog, ctl, name=''):
    t = time.time()
    r = CT.rollout(prog, ctl, trace=list(range(prog.n)))
    print(json.dumps(dict(arm=name, s=round(time.time()-t))), flush=True)
    tr = r['trace']
    f = lambda q: torch.tensor([s[q] for s in tr], dtype=torch.float32)       # noqa: E731  (T, n, ...)
    return dict(score=r['score'].float().cpu(), u=f('u'), y=f('y'), target=f('target'), ref=f('ref'), lo=f('ok_lo'), hi=f('ok_hi'), outside=f('outside'))


def band_power(x, k0):
    """x (T, n, m): per (plant, channel) the power at periods LO..HI after decision k0 (n,), the mean removed."""
    T = x.shape[0]
    t = torch.arange(T)[:, None, None]
    m = (t >= k0[None, :, None]).float()
    xm = (x*m).sum(0)/m.sum(0).clamp_min(1)
    z = (x-xm)*m
    P = torch.fft.rfft(z, dim=0).abs().square()
    per = T/torch.arange(1, P.shape[0]).float()
    sel = torch.zeros(P.shape[0], dtype=torch.bool)
    sel[1:] = (per >= LO) & (per <= HI)
    return P[sel].sum(0)/m.sum(0).clamp_min(1)


def pts(hold, ctl, best, m=None):
    m = torch.ones_like(hold, dtype=torch.bool) if m is None else m
    return round(100*float((hold[m]-ctl[m]).sum()/(hold[m]-best[m]).sum().clamp_min(1e-9)), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('out')
    ap.add_argument('--n', type=int, default=128)
    ap.add_argument('--handed', type=int, default=240)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    t0 = time.time()
    dev = torch.device(a.device)
    net, _ = N.load(a.ckpt, dev)
    prog = ST.C.build(a.n, ST.EXAM_SEED, dev)
    c = prog.c
    k0 = prog.k_first.cpu().clamp_max(719)
    mvm, trk = (c['mvm'] > 0).cpu(), ((c['trk'] > 0) & (c['prc'] > 0) & (c['rowm'] > 0)).cpu()
    lev = ((c['integ'] > 0) & (c['prc'] > 0) & (c['rowm'] > 0)).any(1).cpu()
    R = dict(hold=run(prog, CT.Hold(), 'hold'), teacher=run(prog, CT.Teacher('late'), 'teacher'))
    R['told'], R['blind'] = run(prog, N.Policy(net, paired=True), 'told'), run(prog, N.Policy(net, paired=False), 'blind')
    hold, best = R['hold']['score'], R['teacher']['score']
    out = dict(ckpt=a.ckpt, n=a.n, levels=int(lev.sum()))
    bu = {q: band_power(R[q]['u'], k0) for q in ('teacher', 'told', 'blind')}
    err = {q: (R[q]['y']-R[q]['ref'])/((R[q]['hi']-R[q]['lo'])/2).clamp_min(1e-6) for q in ('teacher', 'told', 'blind')}
    be = {q: band_power(err[q], k0) for q in err}
    moving = mvm & (bu['teacher'] > 0)
    for q in ('told', 'blind'):
        du = R[q]['u'].diff(dim=0)
        flip = ((du[1:]*du[:-1] < 0) & (du[1:].abs() > 1e-4)).float().sum()/(du[1:].abs() > 1e-4).float().sum().clamp_min(1)
        out[q] = dict(pts=pts(hold, R[q]['score'], best), pts_level=pts(hold, R[q]['score'], best, lev), pts_other=pts(hold, R[q]['score'], best, ~lev),
                      swing=round(float((bu[q][moving]/bu['teacher'][moving]).median()), 2),
                      row_swing=round(float((be[q][trk]/be['teacher'][trk].clamp_min(1e-9)).median()), 2), flip=round(float(flip), 3))
        pre = torch.arange(du.shape[0])[:, None] < k0[None, :]
        out[q]['probe'] = round(float((du.square()*pre[..., None]*mvm[None]).sum()/(pre[..., None]*mvm[None]).sum().clamp_min(1))**.5, 4)
    K = a.handed
    if K:
        hm, ht = run(prog, Handed(net, K, 'model', paired=False)), run(prog, Handed(net, K, 'teacher', paired=False))
        hh = run(prog, Handed(net, K, 'hold', paired=False))                                       # (the teacher to K, then nothing moved)
        after = lambda r: r['outside'][K:].sum((0, 2))                                            # noqa: E731
        dm, dt = hm['u'][K+30]-hm['u'][K], ht['u'][K+30]-ht['u'][K]
        big = mvm & (dt.abs() >= .02)
        out['handed'] = dict(K=K, pts=pts(after(hh), after(hm), after(ht)), pts_level=pts(after(hh), after(hm), after(ht), lev),
                             same_side=round(float(((dm*dt > 0) & big).sum()/big.sum().clamp_min(1)), 3), valves=int(big.sum()))
    out['seconds'] = round(time.time()-t0)
    print(json.dumps(out), flush=True)
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)
    li = lev.nonzero()[:, 0]
    torch.save({q: {f: R[q][f][:, li] for f in ('u', 'y', 'target', 'lo', 'hi')} for q in ('teacher', 'told', 'blind')} | dict(ids=li, k0=k0[li]), a.out+'.pt')


if __name__ == '__main__':
    main()

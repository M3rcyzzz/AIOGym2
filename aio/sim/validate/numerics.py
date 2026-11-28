"""The integrator as it is run, on training episodes.

The shell's inputs to the plant are recorded during rollouts of training episodes (tools.episodes) and
replayed through references of the same equations:

  local    every decision restarted from the runtime's own state: the error one decision adds, in bands of
           each output, and as a share of how far the output moved in that decision
  global   the whole episode from its start state on the same inputs (open loop): the drift a rollout
           accumulates, in bands
  exact    on a few episodes the reference is LSODA at 1e-10 (tools.Unit.flow); on all of them it is the
           engine's own scheme with 64 times the substeps and a fresh Jacobian at each, checked against
           LSODA on those few
  float32  the same scheme in float64 on the same inputs, which separates rounding from the scheme
"""
import numpy as np
import torch

from aio.sim.engine import Mech
from aio.sim.validate.tools import Unit, episodes, F

REFINE = 64


def _T(a):
    return torch.as_tensor(a, dtype=F)


def numerics(name, n=16, seed=0, controller='pi', exact=2, steps=None, refine=REFINE):
    u = Unit(name)
    ep = episodes(name, n, seed, controller, steps)
    X0, W, Pm, X1 = ep['x0'], ep['w'], ep['p'], ep['x1']
    K = X0.shape[0]
    m = Mech([u.P], n=n, dtype=F)
    band = _T(u.band)[None]
    sh = u.shown
    S = u.P.substeps*refine
    y = lambda x, k: m.y(x, _T(W[k]), _T(Pm[k]))      # noqa: E731
    loc, loc64, move, glob = [], [], [], []
    xg = _T(X0[0])
    clipped, bound = 0, np.zeros((n, u.nx), dtype=bool)          # decisions an episode spends with a state at its bound (the engine clamps; the references below do too, LSODA does not)
    for k in range(K):
        w, p = _T(W[k]), _T(Pm[k])
        x0 = _T(X0[k])
        xr = m.advance(x0, w, p, substeps=S, refresh=True)                    # one decision, converged
        yr = y(xr, k)
        loc.append(((y(_T(X1[k]), k)-yr).abs()/band)[:, sh])
        loc64.append(((y(m.advance(x0, w, p), k)-yr).abs()/band)[:, sh])
        move.append(((yr-y(x0, k)).abs()/band)[:, sh])
        xg = m.advance(xg, w, p, substeps=S, refresh=True)                    # the whole episode, converged
        glob.append(((y(_T(X1[k]), k)-y(xg, k)).abs()/band)[:, sh])
        hit = (_T(X1[k]) <= m.xlo) | (_T(X1[k]) >= m.xhi)
        clipped += int(hit.any(1).sum())
        bound |= hit.numpy()
    loc, loc64, move, glob = (torch.stack(a).numpy() for a in (loc, loc64, move, glob))       # (K, n, outputs)
    # the converged reference itself against LSODA, on a few episodes
    ex_loc, ex_glob = (0., 0.) if exact else (None, None)      # None: not run
    for e in [i for i in range(n) if not bound[i].any()][:exact]:
        xe = X0[0, e]
        for k in range(K):
            ref = u.flow(X0[k, e], W[k, e], Pm[k, e])
            eng = m.sub(torch.tensor([e])).advance(_T(X0[k, e:e+1]), _T(W[k, e:e+1]), _T(Pm[k, e:e+1]), substeps=S, refresh=True)[0].numpy()
            d = np.abs(u.y(ref, W[k, e], Pm[k, e])-u.y(eng, W[k, e], Pm[k, e]))/u.band
            ex_loc = max(ex_loc, float(d[sh].max()))
            xe = u.flow(xe, W[k, e], Pm[k, e])
            if k % 20 == 19 or k == K-1:
                d = np.abs(u.y(xe, W[k, e], Pm[k, e])-u.y(X1[k, e], W[k, e], Pm[k, e]))/u.band
                ex_glob = max(ex_glob, float(np.abs(d[sh]-glob[k, e]).max()))
    worst = loc.max((0, 2))                                       # per episode
    names = [u.yname[i] for i in sh]
    task = np.array([bool(u.P.ys[i].get('track') or u.P.ys[i].get('integ') or 'lo' in u.P.ys[i] or 'hi' in u.P.ys[i]) for i in sh])     # rows of the task: tracked or limited (the rest is shown only)
    return dict(plant=name, controller=controller, episodes=n, decisions=K, dtype=str(ep['dtype']).replace('torch.', ''),
                dt=u.P.dt, substeps=u.P.substeps, reference_substeps=S, failed=int(ep['failed'].sum()), at_state_bound=clipped,
                states_at_bound=[u.xname[i] for i in np.nonzero(bound.any(0))[0]], episodes_at_bound=int(bound.any(1).sum()),
                local_max=float(loc.max()), local_p99=float(np.quantile(loc, .99)), local_median_episode_max=float(np.median(worst)),
                local_max_task=float(loc[:, :, task].max()), global_max_task=float(glob[:, :, task].max()), local_over_one_band_task=int((loc[:, :, task].max(-1) > 1).sum()),
                local_by_output={q: float(v) for q, v in zip(names, loc.max((0, 1)))}, task_outputs=[q for q, t in zip(names, task) if t],
                local_max_float64=float(loc64.max()),
                local_share=float(loc.sum()/max(move.sum(), 1e-300)),                 # error added per decision over distance moved
                global_max=float(glob.max()), global_median_episode_max=float(np.median(glob.max((0, 2)))),
                global_final_median=float(np.median(glob[-1].max(-1))),
                worst_output=names[int(loc.max((0, 1)).argmax())],
                reference_vs_lsoda_local=ex_loc, reference_vs_lsoda_global=ex_glob)


def stress(name, n=16, seed=0, draws=2048, jump=1., steps=None, refine=REFINE):
    """States the training episodes reached (formula PI and the teacher), each with its manipulated inputs
    moved in one decision, every valve at once: by its own slew limit, up or down (jump='slew': the largest
    move the runtime can make, the edge of the training distribution); or, off the distribution, by up to
    `jump` of the span (0.2: several times the slew limit of every plant; 1: any position inside the limits).
    The runtime's step (guard on) and the same step without the guard against the converged reference, in
    bands; how many of the decisions the guard took again; and the reference against 4 times its own substeps
    on the worst of them."""
    from aio.sim import engine as E
    u = Unit(name)
    rng = np.random.default_rng(seed)
    X0, W, Pm = [], [], []
    for c in ('pi', 'teacher'):
        ep = episodes(name, n, seed, c, steps)
        K = ep['x0'].shape[0]
        k, e = rng.integers(0, K, draws//2), rng.integers(0, n, draws//2)
        ok = np.isfinite(ep['x0'][k, e]).all(-1)
        X0.append(ep['x0'][k, e][ok]), W.append(ep['w'][k, e][ok]), Pm.append(ep['p'][k, e][ok])
    x0, w, p = _T(np.concatenate(X0)), _T(np.concatenate(W)), _T(np.concatenate(Pm))
    N = x0.shape[0]
    m = Mech([u.P], n=N, dtype=F)
    mv = torch.as_tensor(u.mv)
    lo, span = m.wlo[:, mv], (m.whi-m.wlo)[:, mv]
    r = _T(rng.random((N, len(u.mv))))
    if jump == 'slew':
        step = _T([u.P.ws[i]['slew'] for i in u.mv])[None]*torch.sign(2*r-1)
    else:
        step = jump*span*(2*r-1)
    w[:, mv] = lo+span*r if jump != 'slew' and jump >= 1 else torch.minimum(torch.maximum(w[:, mv]+step, lo), lo+span)
    band, sh, S = _T(u.band)[None], u.shown, u.P.substeps*refine
    y = lambda x: m.y(x, w, p)      # noqa: E731
    yr = y(m.advance(x0, w, p, substeps=S, refresh=True))
    names = [u.yname[i] for i in sh]
    task = np.array([bool(u.P.ys[i].get('track') or u.P.ys[i].get('integ') or 'lo' in u.P.ys[i] or 'hi' in u.P.ys[i]) for i in sh])
    E.STATS = []
    err = ((y(m.advance(x0, w, p))-yr).abs()/band)[:, sh].numpy()
    guarded, gtask = err.max(-1), err[:, task].max(-1)
    again = int((E.STATS[0] > E.ERR_TOL).sum())
    E.STATS = None
    plain = ((y(m.advance(x0, w, p, guard=False))-yr).abs()/band)[:, sh].amax(-1).numpy()
    moved = ((yr-y(x0)).abs()/band)[:, sh].amax(-1).numpy()
    worst = torch.as_tensor(np.argsort(-np.nan_to_num(guarded, nan=np.inf))[:32].copy())
    ms = m.sub(worst)
    fine = ms.y(ms.advance(x0[worst], w[worst], p[worst], substeps=4*S, refresh=True), w[worst], p[worst])
    return dict(plant=name, decisions=N, jump=jump, taken_again=again,
                guarded_max=float(np.nanmax(guarded)), guarded_over_one_band=int((guarded > 1).sum()), guarded_not_finite=int((~np.isfinite(guarded)).sum()),
                guarded_max_task=float(np.nanmax(gtask)), guarded_over_one_band_task=int((gtask > 1).sum()),
                guarded_by_output={q: float(v) for q, v in zip(names, np.nanmax(err, 0))},
                plain_max=float(np.nanmax(plain)), plain_over_one_band=int((plain > 1).sum()),
                moved_median=float(np.nanmedian(moved)), moved_max=float(np.nanmax(moved)),
                reference_vs_finer=float(((fine-yr[worst]).abs()/band)[:, sh].amax()))


if __name__ == '__main__':
    import json
    import sys
    torch.set_num_threads(1)
    name = sys.argv[1]
    out = [numerics(name, n=int(sys.argv[2]) if len(sys.argv) > 2 else 16, controller=c, exact=int(sys.argv[3]) if len(sys.argv) > 3 else 2,
                    refine=int(sys.argv[4]) if len(sys.argv) > 4 else REFINE) for c in ('pi', 'teacher')]
    print(json.dumps(out))

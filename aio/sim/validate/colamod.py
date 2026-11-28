"""Skogestad's column A: the original MATLAB code against the engine's column, whole trajectories.

The original `colamod.m` (S. Skogestad, NTNU; not in this repository, see fetch.py) is run unchanged in Octave
with lsode (stiff, 1e-10; cross-checked by ode45 on the first decisions). The level loops are the four lines of the original `cola_lv.m` with the inputs passed
in instead of written into the file; the script first checks that the wrapper returns what `cola_lv.m`
returns at the nominal inputs. The same input schedules (reflux, boilup, feed rate, feed composition, feed
liquid fraction, held over each decision) are given to

  the engine's equations integrated by LSODA    what the transcription is worth along a trajectory
  the engine as it is run (ROS2, float64/32)    what a rollout sees

Schedules: the steps of the original page (feed rate, feed composition, reflux, boilup), and staircases
that move reflux and boilup at the plant's slew limits while the feed wanders over its range.
"""
import os
import subprocess
import tempfile

import numpy as np
import torch

from aio.sim.validate.tools import Unit

WRAP = """function xprime = aio_cola_lv(t, X, u)
% the level loops of cola_lv.m, the inputs u = [LT VB F zF qF] passed in
NT=41; KcB=10; KcD=10; MDs=0.5; MBs=0.5; Ds=0.5; Bs=0.5;
MB=X(NT+1); MD=X(2*NT);
D=Ds+(MD-MDs)*KcD; B=Bs+(MB-MBs)*KcB;
U(1)=u(1); U(2)=u(2); U(3)=D; U(4)=B; U(5)=u(3); U(6)=u(4); U(7)=u(5);
xprime=colamod(t,X,U);
"""

DRIVER = """addpath('{cola}');
S = load('schedule.txt'); x = load('x0.txt'); x = x(:);
% the wrapper against the original cola_lv.m at its own (nominal) inputs
rand('seed', 1); worst = 0;
for i = 1:20
  X = [0.01+0.98*rand(41,1); 0.4+0.2*rand(41,1)];
  a = cola_lv(0, X); b = aio_cola_lv(0, X, [2.70629 3.20629 1.0 0.5 1.0]);
  worst = max(worst, max(abs(a(:)-b(:))));
end
printf('wrapper_vs_cola_lv %.3e\\n', worst);
lsode_options('relative tolerance', 1e-10); lsode_options('absolute tolerance', 1e-12); lsode_options('integration method', 'stiff');
out = zeros(rows(S), 82); x1 = x;
for k = 1:rows(S)
  u = S(k, :);
  [xx, istate, msg] = lsode(@(X, t) aio_cola_lv(t, X, u), x, [0 {dt}/2 {dt}]);
  if istate != 2, error(msg); end
  x = xx(end, :)'; out(k, :) = x';
end
save('-ascii', '-double', 'out.txt', 'out');
% a second integrator of another family on the first decisions (explicit Runge-Kutta)
opts = odeset('RelTol', 1e-9, 'AbsTol', 1e-11); worst = 0;
for k = 1:min(rows(S), 12)
  u = S(k, :);
  [t, xx] = ode45(@(t, X) aio_cola_lv(t, X, u), [0 {dt}/2 {dt}], x1, opts);
  x1 = xx(end, :)'; worst = max(worst, max(abs(x1' - out(k, :))));
end
printf('lsode_vs_ode45 %.3e\\n', worst);
% the steady state and the linearisation of the original code at the nominal inputs (central differences)
u0 = [2.70629 3.20629 1.0 0.5 1.0];
xs = load('x0.txt'); xs = xs(:);
xx = lsode(@(X, t) aio_cola_lv(t, X, u0), xs, [0 10000 20000]); xs = xx(end, :)';
A = zeros(82); h = 1e-6;
for j = 1:82
  e = zeros(82, 1); e(j) = h;
  A(:, j) = (aio_cola_lv(0, xs+e, u0) - aio_cola_lv(0, xs-e, u0))/(2*h);
end
B = zeros(82, 5);
for j = 1:5
  e = zeros(1, 5); e(j) = h;
  B(:, j) = (aio_cola_lv(0, xs, u0+e) - aio_cola_lv(0, xs, u0-e))/(2*h);
end
save('-ascii', '-double', 'xs.txt', 'xs'); save('-ascii', '-double', 'A.txt', 'A'); save('-ascii', '-double', 'B.txt', 'B');
"""

NAMES = ('LT', 'VB', 'F', 'zF', 'qF')


def octave(cola_dir, x0, schedule, dt_min):
    """The original code over a schedule (K, 5) of [LT VB F zF qF], each row held dt_min minutes, from x0 (82):
    the states after every decision, the original's steady state and linearisation at the nominal inputs, and
    the wrapper's largest difference from cola_lv.m."""
    with tempfile.TemporaryDirectory() as d:
        np.savetxt(os.path.join(d, 'schedule.txt'), schedule, fmt='%.17g')
        np.savetxt(os.path.join(d, 'x0.txt'), x0, fmt='%.17g')
        open(os.path.join(d, 'aio_cola_lv.m'), 'w').write(WRAP)
        open(os.path.join(d, 'driver.m'), 'w').write(DRIVER.format(cola=os.path.abspath(cola_dir), dt=repr(float(dt_min))))
        r = subprocess.run(['octave-cli', '--no-gui', '--quiet', 'driver.m'], cwd=d, capture_output=True, text=True, timeout=7200)
        assert r.returncode == 0, r.stderr[-2000:]
        num = lambda k: float(r.stdout.split(k)[1].split()[0])      # noqa: E731
        ld = lambda f: np.loadtxt(os.path.join(d, f))      # noqa: E731
        return dict(x=ld('out.txt'), xs=ld('xs.txt'), A=ld('A.txt'), B=ld('B.txt'), wrapper=num('wrapper_vs_cola_lv'), ode45=num('lsode_vs_ode45'))


def schedules(u, seed=0, K=240):
    """Named input schedules (K, 5) for the column."""
    w0 = np.array([u.w0[u.wname.index(q)] for q in NAMES])
    out = {}

    def step(k, v, at=4):
        s = np.tile(w0, (K, 1))
        s[at:, NAMES.index(k)] = v
        return s
    out['F +1 %'] = step('F', 1.01)
    out['zF 0.50 to 0.51'] = step('zF', .51)
    out['L +1 %'] = step('LT', w0[0]*1.01)
    out['V +1 %'] = step('VB', w0[1]*1.01)
    slew = u.P.ws[u.wname.index('LT')]['slew']
    out['L and V +10 % together, at the slew limit'] = np.tile(w0, (K, 1))
    out['L and V +10 % together, at the slew limit'][4:, :2] = w0[:2]+np.minimum(.1*w0[:2], slew*np.arange(1, K-3)[:, None])
    out['qF 1.0 to 0.9'] = step('qF', .9)
    rng = np.random.default_rng(seed)
    P = u.P
    lim = {q: (P.ws[u.wname.index(q)]['lo'], P.ws[u.wname.index(q)]['hi']) for q in NAMES}
    for j in range(3):                                            # staircases: reflux and the distillate split move, the feed wanders
        s, cur, tgt, load = np.zeros((K, 5)), w0.copy(), w0.copy(), w0[2:].copy()
        nxt = 0
        for k in range(K):
            if k >= nxt:
                nxt = k+int(rng.integers(5, 40))
                if rng.random() < .5:
                    i = int(rng.integers(0, 3))
                    q = NAMES[2+i]
                    load[i] = rng.uniform(lim[q][0]+.15*(lim[q][1]-lim[q][0]), lim[q][1]-.15*(lim[q][1]-lim[q][0]))
                LT = rng.uniform(2.4, 3.0)
                D = load[0]*load[1]*(1+rng.uniform(-.03, .03))                      # distillate near the light component fed
                tgt[:2] = LT, LT+D-(1-load[2])*load[0]
                tgt[:2] = np.clip(tgt[:2], [lim['LT'][0], lim['VB'][0]], [lim['LT'][1], lim['VB'][1]])
            cur[:2] += np.clip(tgt[:2]-cur[:2], -slew, slew)
            cur[2:] += np.clip(load-cur[2:], -.01, .01)                             # the feed ramps
            s[k] = cur
        out[f'staircase {j+1}'] = s
    return out


def compare(cola_dir, seed=0, K=240):
    u = Unit('column')
    xs, ok = u.steady()
    assert ok
    idx = [u.wname.index(q) for q in NAMES]
    h = u.h                                                       # minutes
    rows, first = [], None
    for name, S in schedules(u, seed, K).items():
        W = np.tile(u.w0, (len(S), 1))
        W[:, idx] = S
        Pm = np.tile(u.p0, (len(S), 1))
        o = octave(cola_dir, xs, S, h)
        first = first or o
        y = lambda X: u.outputs(X, W, Pm)      # noqa: E731
        yo = y(o['x'])
        ref = u.run(xs, W, Pm)
        e64 = u.engine(xs, W, Pm)
        e32 = u.engine(xs, W, Pm, dtype=torch.float32)
        d = lambda Y: np.abs(Y-yo)[:, u.shown]/u.band[u.shown]      # noqa: E731
        span = (yo.max(0)-yo.min(0))[u.shown]/u.band[u.shown]
        rows.append(dict(schedule=name, decisions=len(S), moved_bands=float(span[:2].max()),
                         equations_lsoda=float(d(y(ref)).max()), engine_float64=float(d(y(e64)).max()), engine_float32=float(d(y(e32)).max()),
                         engine_float64_purities=float(d(y(e64))[:, :2].max()), worst_output=[u.yname[i] for i in u.shown][int(d(y(e64)).max(0).argmax())],
                         states_lsoda=float(np.abs(ref-o['x']).max())))
    # steady state and linearisation of the original code against the engine's
    _, _, A, B, C, D = u.lin(xs, u.w0, u.p0)
    lam_o, lam_e = np.linalg.eigvals(first['A']).real.max(), np.linalg.eigvals(A).real.max()
    Go, Ge = -np.linalg.solve(first['A'], first['B']), -np.linalg.solve(A, B[:, idx])
    g = lambda G: np.array([[G[40, 0], G[40, 1]], [G[0, 0], G[0, 1]]])*100.      # noqa: E731
    return dict(wrapper_vs_cola_lv=first['wrapper'], lsode_vs_ode45=first['ode45'], steady_state_diff=float(np.abs(first['xs']-xs).max()),
                slowest_eig_original=float(lam_o), slowest_eig_engine=float(lam_e),
                gain_original=g(Go).round(3).tolist(), gain_engine=g(Ge).round(3).tolist(), runs=rows)


if __name__ == '__main__':
    import json
    import sys
    torch.set_num_threads(1)
    print(json.dumps(compare(sys.argv[1]), indent=1))

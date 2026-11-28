"""Real plant records against what the simulator generates.

  csth(dir)             the stirred tank heater pilot plant (Thornhill et al. 2008): the measured valve curves
                        against the shell's family of valve characteristics (flow = opening^gamma, gamma
                        0.6-1.7), and the measured noise and disturbance records against the field layer
  tanks(dir)            the cascaded-tanks benchmark (a real rig, 4 s): a two-tank plant written with the
                        engine's own level equations, fitted on the estimation record and scored on the
                        validation record, next to the best linear model of the same records
  steps(dir)            the open-loop step tests of a pilot distillation column (two tray temperatures against
                        heater and reflux): relative gains and dead time / time constant of a real column
  records(dir)          step-response features (aio.sim.stats.features) of models identified from the
                        input-output records of real plants (DaISy), and where they sit among the prior's plants

The records are not in the repository (fetch.py).
"""
import gzip
import re
from pathlib import Path

import numpy as np

GAMMA = (.6, 1.7)                # prior.py: the shell's valve characteristic, flow = opening^gamma
SBAND = (.01, .03)               # prior.py: the instrument band, share of the range
WHITE, COLOURED, AR = (0., .3), (.05, .3), (.9, .99)             # prior.draw_field: noise sigma in instrument bands; AR(1) coefficient per decision


# ---------------------------------------------------------------- CSTH
def mdl_tables(path):
    """Lookup tables of a Simulink .mdl file: {block name: (inputs, outputs)}."""
    txt = Path(path).read_text(errors='replace')
    out = {}
    for blk in re.split(r'\n\s*Block \{', txt):
        if not re.search(r'BlockType\s+Lookup\b', blk):
            continue
        name = re.search(r'Name\s+"([^"]*)"', blk)

        def vec(key):
            m = re.search(key+r'\s+((?:"[^"]*"\s*)+)', blk)
            s = ''.join(re.findall(r'"([^"]*)"', m.group(1)))
            try:
                return np.array([float(v) for v in re.sub(r'[\[\];,]', ' ', s).split()])
            except ValueError:                                    # an expression, not numbers
                return None
        a, b = vec('InputValues'), vec('OutputValues')
        if name and a is not None and b is not None and len(a) == len(b):
            out[name.group(1).replace('\\n', ' ')] = (a, b)
    return out


def valve_fit(x, f):
    """A measured valve curve (signal, flow) against flow = opening^gamma between the last closed point and
    the first saturated one: best gamma, the largest gap (share of the full flow) at the best gamma and at
    the best gamma inside the shell's range, the first open point (share of travel, share of flow), and the
    ratio of the steepest to the flattest segment (how much the valve gain moves over the travel)."""
    x, f = np.asarray(x, float), np.asarray(f, float)
    lo = np.nonzero(f <= f.min()+1e-12)[0].max()
    hi = np.nonzero(f >= f.max()-1e-12)[0].min()
    u, q = (x[lo:hi+1]-x[lo])/(x[hi]-x[lo]), (f[lo:hi+1]-f[lo])/(f[hi]-f[lo])
    gs = np.linspace(.3, 3., 541)
    gap = np.array([np.abs(u**g-q).max() for g in gs])
    inside = (gs >= GAMMA[0]) & (gs <= GAMMA[1])
    slope = np.diff(q)/np.diff(u)
    return dict(gamma=float(gs[gap.argmin()]), gap=float(gap.min()), gap_in_family=float(gap[inside].min()), first_open=(float(u[1]), float(q[1])),
                gain_ratio=float(slope.max()/slope.min()), turns=int((np.diff(np.sign(np.diff(slope))) != 0).sum()), points=int(len(u)))


def sequence_stats(v, span):
    """A measured record: sigma as a share of the instrument span, lag-one correlation, and the log-log slope
    of its spectrum (0: white; -2: a random walk)."""
    v = np.asarray(v, float)-np.mean(v)
    r1 = float(np.corrcoef(v[:-1], v[1:])[0, 1])
    P, fr = np.abs(np.fft.rfft(v))**2, np.fft.rfftfreq(len(v))
    m = (fr > .005) & (fr < .4)
    return dict(sigma=float(v.std()/span), lag1=r1, slope=float(np.polyfit(np.log(fr[m]), np.log(P[m]), 1)[0]))


def csth(d):
    from scipy.io import loadmat
    d = Path(d)
    t = mdl_tables(d/'CSTHMaster.mdl')
    key = lambda s: next(k for k in t if s in k)      # noqa: E731
    valves = {n: valve_fit(*t[key(k)]) for n, k in (('cold water valve', 'CWValve/mA 2flow/(cu.m/sec)'), ('hot water valve', 'HWValve'), ('steam valve', 'Steam enth'))}
    seq = {n: sequence_stats(loadmat(d/f)[k][1], 16.) for n, f, k in (('cold water flow disturbance', 'CWFlowDisturbance.mat', 'CWFlowDisturbance'),
                                                                     ('level disturbance', 'LevelDisturbance.mat', 'LevelDisturbance'),
                                                                     ('temperature noise', 'TemperatureNoise.mat', 'TemperatureNoise'))}
    return dict(valves=valves, sequences=seq,
                shell=dict(gamma=GAMMA, noise_sigma=(0., WHITE[1]*SBAND[1]+COLOURED[1]*SBAND[1])))


# ---------------------------------------------------------------- cascaded tanks
def tanks_data(d):
    """uEst, yEst, uVal, yVal (V) and the sampling time (s) of the cascaded-tanks benchmark."""
    from scipy.io import loadmat
    d = Path(d)
    f = next(iter(sorted(d.rglob('dataBenchmark.mat'))), None)
    if f is not None:
        m = loadmat(f)
        g = lambda k: np.asarray(m[k], float).ravel()      # noqa: E731
        return g('uEst'), g('yEst'), g('uVal'), g('yVal'), float(np.ravel(m['Ts'])[0])
    f = next(iter(sorted(d.rglob('dataBenchmark.csv'))))
    a = np.genfromtxt(f, delimiter=',', names=True)
    return a['uEst'], a['yEst'], a['uVal'], a['yVal'], float(a['Ts'][0])


def tanks_plant(ts):
    """Two tanks in cascade with the engine's level equations (the outflow of plants.qtank: c sqrt(h)), a
    pump, and an upper tank that overflows at the top of its range, part of the overflow reaching the lower
    tank. Levels in volts of the level sensor; the parameters are fitted."""
    from aio.sim.engine import Plant
    from aio.sim.expr import sqrt, relu, minimum
    P = Plant('tanks', dt=ts, substeps=2)
    u = P.mv('u', 0., 10., 10., 3.)
    k1, k2, k3, k4 = P.param('k1', .05, 0., 1.), P.param('k2', .05, 0., 1.), P.param('k3', .05, 0., 1.), P.param('k4', .04, 0., 1.)
    k5, off = P.param('k5', .3, 0., 1.), P.param('off', 0., -2., 2.)
    top = 10.
    h1, h2 = P.state('h1', 5., lo=0., hi=20.), P.state('h2', 5., lo=0., hi=20.)
    full = relu(h1-top)                                           # the upper tank above its rim spills at once
    P.der(h1, -k1*sqrt(relu(h1))+k4*u-5.*full)
    P.der(h2, k2*sqrt(relu(minimum(h1, top)))-k3*sqrt(relu(h2))+k5*5.*full)
    P.out('y', minimum(h2, top)+off, range=(0., 10.), band=.1, track=True)
    return P


def tanks(d, iters=80, starts=(.03, .06, .12)):
    """Fit on the estimation record, score on the validation record (free run, the initial levels fitted on
    each record's first samples as the benchmark allows). Returns the rms errors (V) of the plant and of the
    best linear output-error model of orders 1-4 found on the same record."""
    import torch
    from scipy.optimize import least_squares
    from scipy.signal import lfilter
    from aio.sim.engine import Mech
    uE, yE, uV, yV, ts = tanks_data(d)
    P = tanks_plant(ts)
    nb = 9                                                        # the point and one perturbation per parameter, in one pass
    m = Mech([P], n=nb, dtype=torch.float64)

    def sim(TH, u):
        TH = np.atleast_2d(TH)
        k = len(TH)
        T = torch.tensor(np.concatenate((TH, np.repeat(TH[:1], nb-k, 0))), dtype=torch.float64)
        p, x = T[:, :6].contiguous(), T[:, 6:8].contiguous()
        out = []
        for uk in u:
            w = torch.full((nb, 1), float(uk), dtype=torch.float64)
            out.append(m.y(x, w, p)[:, 0])
            x = m.advance(x, w, p)
        return torch.stack(out, 1).numpy()[:k]

    def jac(free, base, u, n):
        def J(z):
            th = base.copy()
            th[free] = z
            h = 1e-5*np.maximum(np.abs(z), 1e-2)
            T = np.tile(th, (len(free)+1, 1))
            T[1:, free] += np.diag(h)
            Y = sim(T, u[:n])
            return ((Y[1:]-Y[:1])/h[:, None]).T
        return J
    lo, hi = np.array([1e-4, 1e-4, 1e-4, 1e-4, 0., -2., 0., 0.]), np.array([1., 1., 1., 1., 1., 2., 12., 12.])
    allp = np.arange(8)
    fit = None
    for k in starts:                                              # the outflow coefficient of both tanks; the pump gain follows from the mean level
        th0 = np.array([k, k, k, k*np.sqrt(yE.mean())/uE.mean(), .3, 0., yE[0], yE[0]])
        f = least_squares(lambda th: sim(th, uE)[0]-yE, th0, jac=jac(allp, th0, uE, len(uE)), bounds=(lo, hi), max_nfev=iters, x_scale='jac')
        if fit is None or f.cost < fit.cost:
            fit = f
    th = fit.x
    # validation: the same parameters, the initial levels alone re-fitted on the first 50 samples
    lv = np.array([6, 7])

    def val(x0):
        t = th.copy()
        t[6:] = x0
        return sim(t, uV[:50])[0]-yV[:50]
    x0 = least_squares(val, np.clip([yV[0], yV[0]], lo[6:], hi[6:]), jac=jac(lv, th, uV, 50), bounds=(lo[6:], hi[6:]), max_nfev=30).x
    tv = th.copy()
    tv[6:] = x0
    rms = lambda e: float(np.sqrt(np.mean(e**2)))      # noqa: E731
    yv = sim(tv, uV)[0]
    res = dict(ts=ts, samples=len(yE), params=dict(zip(('k1', 'k2', 'k3', 'k4', 'k5', 'off'), th[:6].round(5).tolist())),
               plant_est=rms(sim(th, uE)[0]-yE), plant_val=rms(yv-yV), output_span=float(yV.max()-yV.min()),
               overflow_samples=int((yV >= 10.-1e-6).sum()))

    # the best linear output-error model on the same records (orders 1-4, free run, fitted by least squares on the simulation error)
    best = None
    for n in (1, 2, 3, 4):
        def lin(c, u, y):
            b, a = c[:n], np.concatenate(([1.], c[n:2*n]))
            return lfilter(np.concatenate(([0.], b)), a, u-c[2*n])+c[2*n+1]

        c0 = np.concatenate((np.full(n, .05/n), np.zeros(n), [uE.mean(), yE.mean()]))
        c0[n] = -.9
        f = least_squares(lambda c: np.nan_to_num(lin(c, uE, yE)-yE, nan=1e3, posinf=1e3, neginf=-1e3), c0, max_nfev=400)
        e, v = rms(lin(f.x, uE, yE)-yE), rms(lin(f.x, uV, yV)-yV)
        if np.isfinite(v) and (best is None or v < best[1]):
            best = (e, v, n)
    res.update(linear_est=best[0], linear_val=best[1], linear_order=best[2])
    return res


# ---------------------------------------------------------------- input-output records of real plants
def daisy(path, skip=1):
    """A DaISy record (.dat or .dat.gz): the columns after the first `skip` (a sample counter or time)."""
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt') as f:
        a = np.loadtxt(f)
    return a[:, skip:]


def step_model(U, Y, L, weights=(1e-3, 1e-2, 1e-1, 1., 10.), hold=.3):
    """Step responses (L+1, outputs, inputs) of a finite-impulse-response model of an input-output record
    (means removed, every signal divided by its standard deviation), fitted by least squares with a penalty
    on the second difference of every impulse response. The weight of the penalty is the one that simulates
    the last `hold` of the record best after a fit on the rest; the fit there (R2 of the simulated output,
    per output) is returned with the responses, which are then fitted on the whole record."""
    U, Y = np.asarray(U, float), np.asarray(Y, float)
    U, Y = (U-U.mean(0))/U.std(0), (Y-Y.mean(0))/Y.std(0)
    N, m = U.shape
    X = np.zeros((N-L, L*m))
    for k in range(L):                                            # lag k+1: the output cannot answer inside the sample
        X[:, k*m:(k+1)*m] = U[L-1-k:N-1-k]
    T = Y[L:]
    D = np.diff(np.eye(L), 2, axis=0)
    Pn = np.kron(D.T@D+1e-3*np.eye(L), np.eye(m))
    cut = int((1-hold)*len(X))

    def solve(rows, lam):
        return np.linalg.solve(X[rows].T@X[rows]+lam*len(X[rows])*Pn, X[rows].T@T[rows])
    tr, te = slice(0, cut), slice(cut, None)
    r2 = lambda H: 1-((T[te]-X[te]@H)-(T[te]-X[te]@H).mean(0)).var(0)/T[te].var(0)      # noqa: E731
    fits = [r2(solve(tr, lam)) for lam in weights]
    best = int(np.argmax([f.mean() for f in fits]))
    h = solve(slice(None), weights[best]).reshape(L, m, -1).transpose(0, 2, 1)       # (lag, output, input)
    return np.concatenate((np.zeros((1,)+h.shape[1:]), np.cumsum(h, 0))), fits[best]


def pairing(G):
    """One output for every input (inputs <= outputs): the choice whose relative gains are closest to one."""
    from itertools import permutations
    ny, m = G.shape
    best = None
    for rows in permutations(range(ny), m):
        g = G[list(rows)]
        if abs(np.linalg.det(g)) < 1e-9:
            continue
        lam = np.diag(g*np.linalg.inv(g).T)
        sc = np.abs(lam-1).sum()
        if best is None or sc < best[0]:
            best = (sc, rows)
    return list(best[1])


# name: (file, first data column, inputs, outputs, sampling time in s or None, what)
RECORDS = dict(
    dryer=('process_industry/dryer2.dat.gz', 1, 3, 3, 10., 'industrial dryer: fuel, exhaust fan speed, raw material rate -> dry bulb, wet bulb, moisture'),
    evaporator=('process_industry/evaporator.dat.gz', 0, 3, 3, None, 'four-stage evaporator: feed, steam, cooling water -> dry matter, product flow, product temperature'),
    glassfurnace=('process_industry/glassfurnace.dat.gz', 1, 3, 6, None, 'glass furnace: two heating inputs, one cooling input -> six temperatures'),
    exchanger=('process_industry/exchanger.dat.gz', 1, 1, 1, 1., 'steam-heated exchanger: liquid flow -> outlet temperature'),
    heating=('thermic/heating_system.dat.gz', 1, 1, 1, 2., 'halogen lamp over a steel plate: drive voltage -> plate temperature'),
)
LAGS = dict(dryer=40, evaporator=80, glassfurnace=40, exchanger=120, heating=150)
SHOWN = ('rga_max', 'rga_min', 'cond', 'theta', 'tau', 'ratio', 'late', 'speed')


QS = (.01, .05, .5, .95, .99)
BATCH = (4096, 7)                # the batch aio.sim.stats reads: episodes, seed


def _shapes(r):
    """Of step responses over their last value (pairs, L+1), each on its own: the apparent dead time (first decision at
    5 %), the time from there to 63 % (at least half a decision), their ratio (as aio.sim.stats.features reads a pair),
    the time to 63 % over the time to 20 % (`spread`; one lag: 4.5, a fast part with a slow tail behind it: much more)
    and how far from the last value the response still is at 0.8 L (`settled`)."""
    L, i = r.shape[1]-1, np.arange(len(r))
    first = lambda c: np.where(c.any(1), c.argmax(1), L)          # noqa: E731
    th, t63 = first(np.abs(r) >= .05), first(np.abs(r) >= .632)
    tau = np.maximum(t63-th, .5)

    def cross(level):
        k = first(r >= level)
        lo, hi = r[i, np.maximum(k-1, 0)], r[i, k]
        return np.where(k == 0, 0., k-1+(level-lo)/np.where(hi != lo, hi-lo, 1.))
    return dict(theta=th.astype(float), tau=tau, ratio=th/tau, spread=cross(.632)/np.maximum(cross(.2), 1e-9), settled=np.abs(r[:, int(.8*L)]-1))


def loop(d):
    """`_shapes` of one paired step response (decisions 0..L)."""
    d = np.asarray(d, float)
    return {q: float(v[0]) for q, v in _shapes((d/d[-1])[None]).items()}


_BATCH = {}


def prior_batch(n=BATCH[0], seed=BATCH[1]):
    """The start plants of the prior (the batch aio.sim.stats reads): their step responses (n, L+1, row, MV), the row
    of every MV, the MVs in use, the integrating rows, the plants with a spare MV. The draw is in float64 whatever the
    batch is held in; it is held in single precision, without which 4096 episodes do not fit in 16 GB (at 1024
    episodes every number read off it is the same in double)."""
    import torch
    from aio.interface import NMV
    from aio.sim import core as C
    if (n, seed) not in _BATCH:
        _BATCH.clear()
        prog = C.build(n, seed, teacher=True, dtype=torch.float32, quiet=True)
        c = prog.c
        _BATCH[(n, seed)] = dict(S=prog.S0.double(), pair=c['link'][:, :, :NMV].argmax(1), act=c['mvm'] > 0, integ=c['integ'][:, :NMV] > 0, spare=c['spr'].sum(1) > 0)
    return _BATCH[(n, seed)]


def prior_features(n=BATCH[0], seed=BATCH[1]):
    """The features of the prior's start plants (as aio.sim.stats prints them): sorted values per feature,
    for the plants with two or more MVs and no spare one, and for all plants."""
    import torch
    from aio.sim import stats as ST
    b = prior_batch(n, seed)
    fe = ST.features(b['S'], b['pair'], b['act'], ~b['integ'])
    multi = (b['act'].sum(1) >= 2) & ~b['spare']
    srt = lambda x: np.sort(x[torch.isfinite(x)].numpy())      # noqa: E731
    return {q: (srt(fe[q][multi]), srt(fe[q])) for q in SHOWN}


def prior_pairs(n=BATCH[0], seed=BATCH[1]):
    """`_shapes` of every paired, settling response of the prior's start plants (one loop is compared with these; a
    plant with `prior_features`, whose dead time over time constant is the largest of the plant's pairs)."""
    b = prior_batch(n, seed)
    S, pair, m = b['S'].numpy(), b['pair'].numpy(), len(b['S'])
    mv = np.arange(S.shape[3])[None]
    d = S[np.arange(m)[:, None], :, pair, mv]                     # (m, MV, L+1): the row of every MV
    use = b['act'].numpy() & ~np.take_along_axis(b['integ'].numpy(), pair, 1) & (np.abs(d[:, :, -1]) > 1e-9)
    return _shapes(d[use]/d[use][:, -1:])


def pct(x, v):
    """Percentile of v among x (ties: the middle of the tied block)."""
    x = np.sort(x)
    return round(100*float(.5*(np.searchsorted(x, v, 'left')+np.searchsorted(x, v, 'right')))/max(len(x), 1), 1)


def alike(P, v, factor=2.):
    """Share of the prior's pairs P within `factor` of one loop v in both its dead time over time constant and its spread."""
    return float(((P['ratio'] >= v['ratio']/factor) & (P['ratio'] <= v['ratio']*factor) & (P['spread'] >= v['spread']/factor) & (P['spread'] <= v['spread']*factor)).mean())


def place(features, loops, n=BATCH[0], seed=BATCH[1]):
    """Where a real plant sits in the prior. features {name: value} of the plant (aio.sim.stats.features; with one loop
    only its dead time and time constant), loops: `loop` of every paired response. Returns the features and the loops
    with percentiles (outside 1-99: the prior does not generate it; a plant against the prior's plants with two or more
    MVs and no spare one, a loop against the prior's paired responses), and per loop the share of the prior's pairs
    like it (`alike`)."""
    F, P = prior_features(n, seed), prior_pairs(n, seed)
    one = len(loops) == 1
    fe = {q: dict(value=round(v, 3), percentile=pct(P[q] if one else F[q][0], v)) for q, v in features.items() if not one or q in ('theta', 'tau', 'ratio')}
    pairs = [dict({q: dict(value=round(v[q], 3), percentile=pct(P[q], v[q])) for q in P}, alike=round(alike(P, v), 4)) for v in loops]
    return fe, pairs


def prior_quantiles(n=BATCH[0], seed=BATCH[1]):
    F, P = prior_features(n, seed), prior_pairs(n, seed)
    return dict(batch=dict(episodes=n, seed=seed, plants=len(F['rga_max'][0]), pairs=len(P['ratio'])), quantiles=QS,
                plants={q: [round(float(v), 3) for v in np.quantile(F[q][0], QS)] for q in SHOWN}, pairs={q: [round(float(v), 3) for v in np.quantile(x, QS)] for q, x in P.items()})


def replace(rec, n=BATCH[0], seed=BATCH[1]):
    """The stored records (`records` of reports/validation/realdata.json, or the `steps` of boptest.json) against the
    prior as it is now: every percentile from the stored values, no data needed."""
    for k, r in rec.items():
        if isinstance(r, dict) and 'loops' in r:
            r['features'], r['pairs'] = place({q: v['value'] for q, v in r['features'].items()}, r['loops'], n, seed)
    rec['prior'] = prior_quantiles(n, seed)
    return rec


def place_column(st, n=BATCH[0], seed=BATCH[1]):
    """The pilot column of `steps` among the prior's plants with two or more MVs: its relative gain and how much slower
    one loop is than the other."""
    F = prior_features(n, seed)
    st['percentile'] = dict(relative_gain=pct(F['rga_max'][0], st['relative_gain']), speed=pct(F['speed'][0], st['speed']))
    return st


def records(d, n=BATCH[0], seed=BATCH[1], minfit=.5):
    """Every record of RECORDS in the DaISy folder d: the fit of its model on the held-out end of the record,
    its features (times in samples of the record), every paired response on its own, and where they sit in the
    prior (`place`). Outputs the model does not simulate (R2 below minfit on the held-out part) are left out
    before pairing."""
    import torch
    from aio.interface import NMV
    from aio.sim import stats as ST
    out = {}
    for name, (f, skip, nu, ny, ts, what) in RECORDS.items():
        a = daisy(Path(d)/f, skip)
        U, Y = a[:, :nu], a[:, nu:nu+ny]
        L = LAGS[name]
        S, fit = step_model(U, Y, L)
        keep = [i for i in range(ny) if fit[i] >= minfit]
        r = dict(what=what, samples=len(a), sampling_s=ts, inputs=nu, outputs=ny, lags=L, fit=[round(float(v), 2) for v in fit], outputs_kept=len(keep))
        m = min(nu, len(keep))
        if m >= 1:
            G = S[-1][keep]                                       # (kept outputs, inputs)
            if len(keep) >= nu:
                rows, cols = (pairing(G) if nu > 1 else [0]), list(range(nu))
            else:                                                 # fewer outputs than inputs: one input for every output
                cols, rows = pairing(G.T), list(range(m))
            Sq = S[:, [keep[i] for i in rows]][:, :, cols]        # (L+1, m, m): row i is the output of input cols[i]
            r['settled'] = round(float(np.abs(np.diagonal(Sq[int(.8*L)])/np.diagonal(Sq[-1])-1).max()), 2)      # how far from the final value at 0.8 L
            Sp = torch.zeros(1, L+1, NMV, NMV, dtype=torch.float64)
            Sp[0, :, :m, :m] = torch.tensor(Sq)
            am = torch.arange(NMV)
            fe = ST.features(Sp, am[None], (am < m)[None], torch.ones(1, NMV, dtype=torch.bool))
            r['pairing'] = [(int(c), int(keep[i])) for c, i in zip(cols, rows)]      # (input, output)
            r['features'] = {q: dict(value=float(fe[q][0])) for q in SHOWN}
            r['loops'] = [loop(Sq[:, i, i]) for i in range(m)]
        out[name] = r
    return replace(out, n, seed)


# ---------------------------------------------------------------- open-loop step tests of a pilot column
def steps(d, smooth=200):
    """The four step tests of the Manipal pilot column (T1Heater, T5Heater, T1Reflux, T5Reflux: one
    temperature record each; the files carry no time stamps, no input values and not the instant of the
    step, so dead times cannot be read). Relative gains do not depend on the size of the steps; the rise
    (5 % to 63 % of the change) is in samples."""
    import openpyxl
    out, G, rise = {}, np.zeros((2, 2)), np.zeros((2, 2))
    for i, y in enumerate(('T1', 'T5')):
        for j, u in enumerate(('Heater', 'Reflux')):
            ws = openpyxl.load_workbook(next(Path(d).rglob(f'{y}{u}.xlsx')), read_only=True, data_only=True).worksheets[0]
            v = np.array([r[0] for r in ws.iter_rows(values_only=True) if r and isinstance(r[0], (int, float))], float)
            v = np.convolve(v, np.ones(smooth)/smooth, mode='valid')
            a, b = v[:len(v)//100].mean(), v[-len(v)//100:].mean()
            rel = (v-a)/(b-a)
            G[i, j], rise[i, j] = b-a, int(np.argmax(rel >= .632))-int(np.argmax(rel >= .05))
            out[f'{y} to {u}'] = dict(samples=len(v), start=round(float(a), 2), end=round(float(b), 2), rise=int(rise[i, j]), overshoot=round(float(rel.max()-1), 2))
    lam = G*np.linalg.inv(G).T
    k = 0 if lam[0, 0] > 0 else 1                                 # the pairing with positive relative gains: T1 with input k
    out['pairing'] = f"T1 with {('Heater', 'Reflux')[k]}, T5 with {('Heater', 'Reflux')[1-k]}"
    out['relative_gain'] = round(float(lam[0, k]), 2)
    out['speed'] = round(float(max(rise[0, k], rise[1, 1-k])/min(rise[0, k], rise[1, 1-k])), 2)
    return out

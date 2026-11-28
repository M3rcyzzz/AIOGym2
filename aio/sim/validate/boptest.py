"""BOPTEST as an outside exam: building emulators that carry their own baseline controller and their own
scoring (IBPSA Project 1, https://github.com/ibpsa/project1-boptest).

Everything here goes through BOPTEST's REST interface, so it runs against the project's service
(`docker compose up web worker provision`) or against `boptest_serve.py` in the worker image.

  reference   two days of every standard period under the test case's own controller, as the project's
              tests run them, against the KPI files the project keeps: is this deployment the published one
  baseline    the whole two-week period under the test case's own controller: the score to beat
  steps       step tests on the actuators around the baseline operating day, as differences from the same
              days without the step (a building has no steady state: weather and occupancy keep moving),
              and the step-response features of aio.sim.stats against the training prior
  exam        a controller on the public packet in place of the test case's zone-temperature loops, scored
              by BOPTEST's KPIs: the network zero-shot, and a PI tuned by formula from the step tests
"""
import csv
import json
import os
import queue
import threading
import urllib.error
import urllib.request

PRICES = ('constant', 'dynamic', 'highly_dynamic')
DAY = 86400.


class Client:
    def __init__(self, url, timeout=7200):
        self.url, self.timeout = url.rstrip('/'), timeout

    def call(self, method, path, body=None):
        r = urllib.request.Request(f'{self.url}/{path}', data=None if body is None else json.dumps(body).encode(), method=method,
                                   headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(r, timeout=self.timeout) as f:
            return json.loads(f.read())

    def cases(self):
        return [d['testcaseid'] for d in self.call('GET', 'testcases')]


class Test:
    """One running test (a selected test case). Requests answer with BOPTEST's payload and raise on its errors."""

    def __init__(self, client, case):
        self.c, self.case = client, case
        self.id = client.call('POST', f'testcases/{case}/select')['testid']

    def _payload(self, r, what):
        if r.get('status') != 200:
            raise RuntimeError(f'{self.case} {what}: {r.get("message")}')
        return r['payload']

    def get(self, what):
        return self._payload(self.c.call('GET', f'{what}/{self.id}'), what)

    def put(self, what, **q):
        return self._payload(self.c.call('PUT', f'{what}/{self.id}', q), what)

    def advance(self, u=None):
        return self._payload(self.c.call('POST', f'advance/{self.id}', u or {}), 'advance')

    def stop(self):
        self.c.call('PUT', f'stop/{self.id}')

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.stop()


def spread(urls, jobs, fn):
    """fn(client, job) for every job, one job at a time on each server. Results in the order of jobs. A
    server that stops answering hands its job back to the others."""
    q, out = queue.Queue(), [None]*len(jobs)
    for i, j in enumerate(jobs):
        q.put((i, j))

    def work(url):
        c = Client(url)
        while True:
            try:
                i, j = q.get_nowait()
            except queue.Empty:
                return
            try:
                out[i] = fn(c, j)
            except (urllib.error.URLError, ConnectionError) as e:
                out[i] = dict(job=j, error=repr(e)[:400])
                try:
                    c.cases()
                except Exception:
                    q.put((i, j))
                    return
            except Exception as e:
                out[i] = dict(job=j, error=repr(e)[:400])

    for _ in urls:                                                  # jobs handed back are picked up in the next round
        th = [threading.Thread(target=work, args=(u,)) for u in urls]
        [t.start() for t in th]
        [t.join() for t in th]
        if q.empty():
            break
    return out


def periods(root, case):
    """The standard two-week periods of a test case (the day each is centred on)."""
    with open(os.path.join(root, 'testcases', case, 'models', 'days.json')) as f:
        return json.load(f)


def kpis(test):
    """BOPTEST's KPIs of the test so far under each electricity price (the price only changes the cost)."""
    out = {}
    for p in PRICES:
        test.put('scenario', electricity_price=p)
        out[p] = test.get('kpi')
    test.put('scenario', electricity_price='constant')
    return out


def reference(urls, root, cases=None):
    """Per test case: the largest relative difference between this deployment's KPIs and the project's
    reference files (testing/references/<case>/kpis_<period>_<price>.csv: two days of each period under the
    test case's own controller, testing/utilities.py run_time_period). The time ratio is the machine's."""
    cases = cases or Client(urls[0]).cases()
    jobs = [(c, p) for c in cases for p in periods(root, c)]

    def one(client, job):
        case, period = job
        with Test(client, case) as t:
            t.put('scenario', time_period=period)
            t.put('step', step=DAY)
            for _ in range(2):
                t.advance()
            got, worst, n = kpis(t), 0., 0
            for price in PRICES:
                path = os.path.join(root, 'testing', 'references', case, f'kpis_{period}_{price}.csv')
                if not os.path.isfile(path):
                    continue
                with open(path) as f:
                    ref = {r['keys']: r['value'] for r in csv.DictReader(f)}
                for k, v in ref.items():
                    if k == 'time_rat' or v == '':
                        continue
                    a, b = got[price][k], float(v)
                    worst, n = max(worst, abs(a-b)/max(abs(b), 1e-12)), n+1
        return dict(case=case, period=period, values=n, worst=worst)

    res = spread(urls, jobs, one)
    out = {}
    for c in cases:
        rs = [r for r in res if r.get('case') == c]
        bad = [r for r in res if r.get('job', (None,))[0] == c]
        out[c] = dict(periods=len(rs), values=sum(r['values'] for r in rs), worst=max([r['worst'] for r in rs], default=None),
                      errors=[r['error'] for r in bad])
    return out


def baseline(urls, root, cases=None):
    """The whole of every standard period under the test case's own controller: BOPTEST's KPIs."""
    cases = cases or Client(urls[0]).cases()
    jobs = [(c, p) for c in cases for p in periods(root, c)]

    def one(client, job):
        case, period = job
        with Test(client, case) as t:
            t.put('scenario', time_period=period)
            t.put('step', step=DAY)
            days = 0
            while t.advance():
                days += 1
            return dict(case=case, period=period, days=days, kpi=kpis(t))

    return spread(urls, jobs, one)


# ---------------------------------------------------------------- the zone loops taken over
# Per test case: the decision interval, the actuators moved (one MV may drive several points), the points
# held, and the controlled values with the signal that carries the setpoint the test case's own loop follows
# (read from the test case every decision; it is not overwritten). `lo`, `hi` of a CV: the instrument range
# published in the packet; `band`: 2 % of it, the middle of what the prior draws.
LOOPS = {
    'bestest_hydronic': dict(                      # radiator: supply temperature of the boiler -> operative temperature
        dt=900., hold={'ovePum_u': 1.},
        mv=[dict(points=('oveTSetSup_u',), lo=293.15, hi=353.15)],
        cv=[dict(y='reaTRoo_y', sp='oveTSetHea_u', lo=283.15, hi=303.15, band=.4)]),
    'bestest_hydronic_heat_pump': dict(            # floor heating: compressor speed -> operative temperature; fan and pump run with the compressor
        dt=3600., hold={'oveFan_u': lambda u: float(u[0] > 0), 'ovePum_u': lambda u: float(u[0] > 0)},
        mv=[dict(points=('oveHeaPumY_u',), lo=0., hi=1.)],
        cv=[dict(y='reaTZon_y', sp='oveTSet_u', lo=283.15, hi=303.15, band=.4)]),
    'singlezone_commercial_hydronic': dict(        # radiator valve -> zone temperature; ventilation stays with the test case
        dt=900., hold={},
        mv=[dict(points=('oveValRad_u',), lo=0., hi=1.)],
        cv=[dict(y='reaTZon_y', sp='oveTZonSet_u', lo=283.15, hi=303.15, band=.4)]),
    'twozone_apartment_hydronic': dict(            # two floor-heating valves -> two zone temperatures; the pump gives the flow the open valves ask for
        dt=900., hold={'hydronicSystem_oveMpumCon_u': lambda u: .172*(u[0]+u[1])},
        mv=[dict(points=('hydronicSystem_oveMDayZ_u',), lo=0., hi=1.), dict(points=('hydronicSystem_oveMNigZ_u',), lo=0., hi=1.)],
        cv=[dict(y='dayZon_reaTRooAir_y', sp='thermostatDayZon_oveTsetZon_u', lo=283.15, hi=303.15, band=.4),
            dict(y='nigZon_reaTRooAir_y', sp='thermostatNigZon_oveTsetZon_u', lo=283.15, hi=303.15, band=.4)]),
}
SLEW = .2          # largest move per decision, of the MV's range (the prior's largest)


def overwrite(loop, u):
    q = {}
    for m, v in zip(loop['mv'], u):
        for p in m['points']:
            q[p], q[p[:-2]+'_activate'] = float(v), 1
    for p, v in loop['hold'].items():
        q[p], q[p[:-2]+'_activate'] = float(v(u) if callable(v) else v), 1
    return q


def step_test(client, case, day, lags=240, size=.1, loop=None):
    """Open-loop step tests of the loop's MVs on the days from `day` (day of year). The test case's own
    controller runs the week before and the first day, each MV is then held at its mean of that day (kept
    `size` of its range inside its limits), and stepped up by `size` of its range in a run of its own. The
    response is the difference from the run without a step, under the same weather and occupancy.
    Returns S (lags+1, CVs, MVs) in CV range per full MV range, and the operating values."""
    import numpy as np
    L = loop or LOOPS[case]
    dt, nm = L['dt'], len(L['mv'])
    lo, hi = np.array([m['lo'] for m in L['mv']]), np.array([m['hi'] for m in L['mv']])
    span = hi-lo

    def run(j, u0=None):
        with Test(client, case) as t:
            t.put('initialize', start_time=(day-1)*DAY, warmup_period=7*DAY)
            t.put('step', step=dt)
            us = []
            for _ in range(int(DAY/dt)):
                y = t.advance()
                us.append([y[m['points'][0]] for m in L['mv']])
            if u0 is None:
                u0 = np.clip(np.mean(us, 0), lo+size*span, hi-size*span)
            u = u0.copy()
            if j is not None:
                u[j] += size*span[j]
            ys = [[y[c['y']] for c in L['cv']]]
            for _ in range(lags):
                y = t.advance(overwrite(L, u))
                ys.append([y[c['y']] for c in L['cv']])
        return np.array(ys), u0

    base, u0 = run(None)
    S = np.stack([run(j, u0)[0]-base for j in range(nm)], -1)                       # (lags+1, CVs, MVs)
    S = S/np.array([c['hi']-c['lo'] for c in L['cv']])[None, :, None]/size
    return S, dict(u0=u0.tolist(), base=base[::max(1, lags//60)].tolist())


def step_features(S):
    """aio.sim.stats.features of one square step-response block (paired on the diagonal)."""
    import torch
    from aio.interface import NMV
    from aio.sim import stats as ST
    m = S.shape[1]
    Sp = torch.zeros(1, S.shape[0], NMV, NMV, dtype=torch.float64)
    Sp[0, :, :m, :m] = torch.tensor(S)
    am = torch.arange(NMV)
    fe = ST.features(Sp, am[None], (am < m)[None], torch.ones(1, NMV, dtype=torch.bool))
    return {q: float(v[0]) for q, v in fe.items()}


def lambda_pi(S):
    """The formula PI of aio.sim.model.pi_tuning on a settling step response (lambda tuning on the pairing:
    dead time from the 5 % point, time constant from there to 63 %, closed-loop time the larger of the time
    constant and three dead times; with interacting pairs, relative gain off one by more than 1.5 times, all
    but the fastest loop three times slower). Gains per full MV range and CV range, times in decisions."""
    import numpy as np
    m = S.shape[1]
    s = np.stack([S[:, i, i] for i in range(m)], 1)
    K = s[-1]
    r = s/K
    first = lambda c: np.where(c.any(0), c.argmax(0), len(s)-1).astype(float)      # noqa: E731
    theta = np.maximum(first(r >= .05)-.5, 0.)
    tau = np.maximum(first(r >= .632)-theta, 1.)
    lam = np.maximum(np.maximum(tau, 3*theta), 1.)
    G = S[-1]
    rel = np.diagonal(G*np.linalg.inv(G).T)
    inter = (rel <= 0) | (np.abs(np.log(np.clip(rel, 1e-12, None))) > np.log(1.5))
    if inter.sum() > 1:
        slow = inter & (np.arange(m) != np.where(inter, lam, np.inf).argmin())
        lam = np.where(slow, 3*lam, lam)
    return dict(kc=(tau/(K*(lam+theta))).tolist(), ti=tau.tolist(), theta=theta.tolist(), tau=tau.tolist(), lam=lam.tolist(), gain=K.tolist(), rga=rel.tolist())


def shape(d):
    """Of one paired step response (decisions 0..240): how far from its last value it still is at decision
    192 (a response that has lined out: 0), and the time to 63 % over the time to 20 % of the last value
    (one lag: 4.5; a fast part with a slow tail behind it: much more)."""
    import numpy as np
    r = np.asarray(d, float)/d[-1]

    def cross(level):
        k = int(np.argmax(r >= level))
        return k if k == 0 else k-1+(level-r[k-1])/(r[k]-r[k-1])

    return dict(settled=float(abs(r[int(.8*(len(r)-1))]-1)), spread=float(cross(.632)/max(cross(.2), 1e-9)))


def steps(urls, root, cases=None, period='peak_heat_day', days=10):
    """Step tests of every loop from the start of a period over `days`. The features of the first 240
    decisions (what the prior's features are read from), every paired response on its own, where they sit in
    the prior (realdata.place: percentiles, and the share of the prior's paired responses like each loop), the
    share of the last value reached inside 240 decisions, and the formula PI."""
    import numpy as np
    from aio.sim.validate import realdata as RD
    cases = cases or list(LOOPS)

    def one(client, case):
        dt = LOOPS[case]['dt']
        Sl, op = step_test(client, case, periods(root, case)[period]-7, lags=max(240, int(days*DAY/dt)))
        S = Sl[:241]
        m = S.shape[1]
        fe = step_features(S)
        return dict(case=case, period=period, dt=dt, mvs=m, operating=op, pi=lambda_pi(S), gains=S[-1].tolist(), gains_last=Sl[-1].tolist(),
                    reached=(np.diagonal(S[-1])/np.diagonal(Sl[-1])).tolist(), response=Sl[::max(1, (len(Sl)-1)//120)].tolist(),
                    features={q: dict(value=fe[q]) for q in RD.SHOWN}, loops=[RD.loop(S[:, i, i]) for i in range(m)])

    return RD.replace({r.get('case', str(r.get('job'))): r for r in spread(urls, cases, one)})


class FormulaPI:
    """The formula PI on the pairing (velocity form, moves inside the slew limit), gains from `lambda_pi`."""

    def __init__(self, loop, tune):
        import numpy as np
        self.kc, self.ti = np.array(tune['kc']), np.array(tune['ti'])
        self.rng = np.array([c['hi']-c['lo'] for c in loop['cv']])
        self.e = None

    def __call__(self, y, sp, u, lo, hi):
        import numpy as np
        e = (sp-y)/self.rng
        e0 = e if self.e is None else self.e
        self.e = e
        du = np.clip(self.kc*((e-e0)+e/self.ti), -SLEW, SLEW)
        return np.clip(u+du*(hi-lo), lo, hi)


class Network:
    """The network on the public packet (aio.net.Controller; `weights`: a released folder of
    release/huggingface, or a run's .pt file): every CV tracked to the setpoint of the test case's own loop,
    its band as the scale, the instrument range as the packet range, the pairing on the diagonal with the
    sign of the step test."""

    def __init__(self, loop, weights, sign):
        import sys
        import numpy as np
        from pathlib import Path
        from aio import net as N
        self.encode = sys.modules[N.shape_of.__module__].encode
        self.ctl = N.Controller(N.released(weights) if Path(weights).is_dir() else N.load(weights)[0])
        cv = loop['cv']
        self.k = dict(scale=np.array([c['band'] for c in cv]), lower=np.array([c['lo'] for c in cv]), upper=np.array([c['hi'] for c in cv]),
                      tracking=np.ones(len(cv)), pairing=list(range(len(loop['mv']))), dt=loop.get('packet_dt', loop['dt']), sign=np.sign(sign))
        self.last = None

    def __call__(self, y, sp, u, lo, hi):
        import numpy as np
        py, psp, pu = self.last or (y, sp, u)
        self.last = (y, sp, u)
        slew = SLEW*(hi-lo)
        packet = self.encode(values=y, previous=py, target=sp, last_target=psp, command=u, last_command=pu, u_low=lo, u_high=hi, slew=slew, **self.k)
        return np.clip(u+self.ctl.step(packet)[:len(u)]*slew, lo, hi)


def thin(Y, SP, U, n=336):
    """About n evenly spaced decisions of a run, to three decimals: what the report keeps of a trajectory."""
    k = max(1, len(Y)//n)
    return dict(y=Y[::k].round(3).tolist(), sp=SP[::k].round(3).tolist(), u=U[::k].round(3).tolist())


def exam(client, case, period, make, loop=None):
    """One standard period with a controller in place of the test case's zone loops. `make(loop)` gives
    ctl(y, sp, u, lo, hi) -> u. The controller takes over from the test case's own actuator values, reads
    the setpoints the test case's own loops follow, and decides once per `dt`. Returns BOPTEST's KPIs and how
    the CVs sat against those setpoints (mean absolute error, share of decisions outside the band, both over
    all decisions and over those where the setpoint is the day's highest: occupied hours; how far above the
    setpoint on average, and the share of decisions an MV is at its lower limit)."""
    import numpy as np
    L = loop or LOOPS[case]
    lo, hi = np.array([m['lo'] for m in L['mv']]), np.array([m['hi'] for m in L['mv']])
    band = np.array([c['band'] for c in L['cv']])
    with Test(client, case) as t:
        y = t.put('scenario', time_period=period, electricity_price='constant')['time_period']
        t.put('step', step=L['dt'])
        ctl = make(L)
        u = np.clip([y[m['points'][0]] for m in L['mv']], lo, hi)
        Y, SP, U = [], [], []
        while True:
            yy, sp = np.array([y[c['y']] for c in L['cv']]), np.array([y[c['sp']] for c in L['cv']])
            Y.append(yy); SP.append(sp)
            u = ctl(yy, sp, u, lo, hi)
            U.append(u)
            y = t.advance(overwrite(L, u))
            if not y:
                break
        k = kpis(t)
    Y, SP, U = np.array(Y), np.array(SP), np.array(U)
    err = np.abs(Y-SP)
    occ = SP >= SP.max(0)-1e-6
    return dict(case=case, period=period, decisions=len(Y), kpi=k,
                error=err.mean(0).tolist(), outside=(err > band).mean(0).tolist(),
                error_occupied=[float(err[occ[:, i], i].mean()) for i in range(Y.shape[1])],
                outside_occupied=[float((err[occ[:, i], i] > band[i]).mean()) for i in range(Y.shape[1])],
                moves=(np.abs(np.diff(U, axis=0)).mean(0)/(hi-lo)).tolist(),
                mean_above_setpoint=(Y-SP).mean(0).tolist(), highest=Y.max(0).tolist(), at_low_limit=((U-lo)/(hi-lo) < .01).mean(0).tolist(),
                trace=thin(Y, SP, U))


def own(client, case, period, loop=None):
    """The same period under the test case's own controller, read at the same instants: KPIs and the same
    tracking numbers as `exam`."""
    import numpy as np
    L = loop or LOOPS[case]
    band = np.array([c['band'] for c in L['cv']])
    with Test(client, case) as t:
        y = t.put('scenario', time_period=period, electricity_price='constant')['time_period']
        t.put('step', step=L['dt'])
        Y, SP, U = [], [], []
        while y:
            Y.append([y[c['y']] for c in L['cv']]); SP.append([y[c['sp']] for c in L['cv']]); U.append([y[m['points'][0]] for m in L['mv']])
            y = t.advance()
        k = kpis(t)
    Y, SP, U = np.array(Y), np.array(SP), np.array(U)
    err = np.abs(Y-SP)
    occ = SP >= SP.max(0)-1e-6
    span = np.array([m['hi']-m['lo'] for m in L['mv']])
    return dict(case=case, period=period, decisions=len(Y), kpi=k,
                error=err.mean(0).tolist(), outside=(err > band).mean(0).tolist(),
                error_occupied=[float(err[occ[:, i], i].mean()) for i in range(Y.shape[1])],
                outside_occupied=[float((err[occ[:, i], i] > band[i]).mean()) for i in range(Y.shape[1])],
                moves=(np.abs(np.diff(U, axis=0)).mean(0)/span).tolist(),
                mean_above_setpoint=(Y-SP).mean(0).tolist(), highest=Y.max(0).tolist(),
                at_low_limit=((U-np.array([m['lo'] for m in L['mv']]))/span < .01).mean(0).tolist(),
                trace=thin(Y, SP, U))


def exams(urls, weights, tests, cases=None, which=('peak_heat_day', 'typical_heat_day')):
    """Every loop over its standard heating periods under: the test case's own controller, the formula PI
    tuned from the step test, the network, and the network with half the band (1 % of the range, the
    narrowest the prior draws: how much the result depends on that commissioning number). `tests`: the
    result of `steps`."""
    import copy
    import numpy as np
    cases = cases or [c for c in LOOPS if c in tests]
    jobs = [(c, p, k) for c in cases for p in which for k in ('own', 'formula_pi', 'network', 'network_half_band')]

    def one(client, job):
        case, period, kind = job
        sign = np.sign(np.diagonal(np.array(tests[case]['gains'])))
        loop = LOOPS[case]
        if kind == 'network_half_band':
            loop = copy.deepcopy(loop)
            for c in loop['cv']:
                c['band'] /= 2
        if kind == 'own':
            r = own(client, case, period)
        elif kind == 'formula_pi':
            r = exam(client, case, period, lambda L: FormulaPI(L, tests[case]['pi']))
        else:
            r = exam(client, case, period, lambda L: Network(L, weights, sign), loop=loop)
        r['controller'] = kind
        return r

    return spread(urls, jobs, one)

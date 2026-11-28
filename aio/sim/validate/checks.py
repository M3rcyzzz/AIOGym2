"""Per plant: the engine's table against the papers and against physics.

  equations(name)     the engine's right-hand sides against the second transcription (papers.py) at random
                      points: largest difference per equation, as a share of that equation's size
  published(name)     rows of (what, published, engine, tolerance, ok): operating points, time constants,
                      gains, zeros, linearised models, as the papers print them
  conservation(name)  mass, component and energy ledgers written here from the flowsheet, against the sum of
                      the engine's derivatives, at random points

The tolerance of a published row is the precision the paper prints (half a unit of the last digit, or more
where the entry says why), never a number chosen to pass.
"""
import math
import numpy as np

from aio.sim.validate import papers as PP
from aio.sim.validate.tools import Unit

LN10 = math.log(10.)


def row(what, pub, eng, tol, note=''):
    pub, eng = np.asarray(pub, dtype=float), np.asarray(eng, dtype=float)
    dev = float(np.abs(eng-pub).max())
    return dict(what=what, published=pub.round(6).tolist(), engine=eng.round(6).tolist(), dev=dev, tol=float(tol), ok=bool(dev <= tol), note=note)


# ---------------------------------------------------------------- equations
def _points(u, n, rng, name):
    """Random points of a plant: states around the nominal ones, inputs over their ranges."""
    X = u.x0[None]*(1+.3*rng.uniform(-1, 1, (n, u.nx)))
    W = np.array([[rng.uniform(w['lo'], w['hi']) for w in u.P.ws] for _ in range(n)])
    if name == 'column':
        X[:, :41] = np.clip(X[:, :41], 1e-4, 1-1e-4)
    if name == 'ph':
        X[:, u.xname.index('pH')] = rng.uniform(3., 11., n)
        X[:, u.xname.index('wa')] = rng.uniform(-1e-3, 1e-3, n)
        X[:, u.xname.index('wb')] = rng.uniform(0., 1e-3, n)
        for q in ('sa', 'sb'):                                    # the engine's stream strengths: not in the paper
            W[:, u.wname.index(q)] = 1.
    if name == 'excstr':
        X[:, u.xname.index('I')] = 0.
    if name == 'evap':                                            # the engine floors the cooling water flow at 1 kg/min (the paper divides by it)
        W[:, u.wname.index('F200')] = np.maximum(W[:, u.wname.index('F200')], 1.)
    return X, W


OWN = dict(qtank=('g1', 'g2', 'k1', 'k2'), ph=('pk1',), bio=('mum', 'Km', 'K1', 'Y', 'alpha'), column=('alpha',))      # parameters the paper has too
SKIP = dict(excstr=('Tj', 'I'))                                   # states the engine adds around the paper's plant (an inner loop)


def equations(name, n=200, seed=0, **over):
    """Largest difference between the engine's equations and the paper's, per equation, over n random points
    (as a share of the largest value that equation takes over the points). `over`: constants of the paper's
    transcription to override (to locate a difference)."""
    if name not in PP.PAPERS:
        return None
    u, rng = Unit(name), np.random.default_rng(seed)
    X, W = _points(u, n, rng, name)
    worst, size = {}, {}
    for x, w in zip(X, W):
        p = u.p0.copy()
        for q in OWN.get(name, ()):
            s = u.P.ps[u.pname.index(q)]
            p[u.pname.index(q)] = rng.uniform(s['lo'], s['hi'])
        v = dict(zip(u.wname, w)) | dict(zip(u.pname, p)) | over
        if name == 'column':
            v |= dict(x=x[:41], M=x[41:])
            d = PP.column(v)
            ref = dict(zip(u.xname, np.concatenate((d['x'], d['M']))))
        else:
            ref = PP.PAPERS[name](v | dict(zip(u.xname, x)))
        f = dict(zip(u.xname, u.f(x, w, p)))
        for k, r in ref.items():
            if k.startswith('_') or k in SKIP.get(name, ()):
                continue
            key = k.rstrip('0123456789') if name == 'column' else k
            worst[key] = max(worst.get(key, 0.), abs(f[k]-r))
            size[key] = max(size.get(key, 0.), abs(r))
    return {k: worst[k]/max(size[k], 1e-300) for k in worst}


# ---------------------------------------------------------------- published numbers
def _qtank():
    u, Q, out = Unit('qtank'), PP.QTANK, []
    area, hole, g = np.array([28., 32., 28., 32.]), np.array([.071, .057, .071, .057]), 981.
    for tag in ('minus', 'plus'):
        d = Q[tag]
        p = u.params(g1=d['g'][0], g2=d['g'][1], k1=d['k'][0], k2=d['k'][1])
        w = u.inputs(v1=d['v'][0], v2=d['v'][1])
        xs, ok = u.steady(w, p, x0=np.array(d['h']))
        # the paper's operating levels are measured values printed to 0.1 cm; its parameters put the steady levels up to
        # 0.2 cm from them (the paper's own equations, written here, give the same levels as the engine)
        own = np.array([((1-d['g'][1])*d['k'][1]*d['v'][1]/hole[2])**2/(2*g), ((1-d['g'][0])*d['k'][0]*d['v'][0]/hole[3])**2/(2*g)])
        own = np.concatenate((((np.array([d['g'][0]*d['k'][0]*d['v'][0], d['g'][1]*d['k'][1]*d['v'][1]])+hole[2:]*np.sqrt(2*g*own))/hole[:2])**2/(2*g), own))
        out.append(row(f'P{"-" if tag == "minus" else "+"} steady levels at the published parameters (cm)', d['h'], xs, .5,
                       'the paper prints measured levels; its own equations give %s' % np.round(own, 2).tolist()))
        out.append(row(f'P{"-" if tag == "minus" else "+"} steady levels against the paper\'s equations solved here (cm)', own, xs, 1e-6))
        h = np.array(d['h'])                                      # the paper linearises at the printed levels
        lam = np.sort(u.modes(h, w, p).real)
        T = np.sort(-1/lam)
        out.append(row(f'P{"-" if tag == "minus" else "+"} time constants (s)', np.sort(d['T']), T, 1., 'printed as whole seconds'))
        G = u.dcgain_at(h, w, p)[:2][:, u.mv]*Q['kc']
        out.append(row(f'P{"-" if tag == "minus" else "+"} steady gains (V/V)', d['gain'], G, .06, 'printed to two digits'))
        z = np.sort(u.zeros(h, w, p, [0, 1], u.mv).real)
        out.append(row(f'P{"-" if tag == "minus" else "+"} multivariable zeros (1/s)', np.sort(d['zeros']), z, 2e-3,
                       'zeros of the printed transfer matrix; it prints T3 = 23 s where its formula at the printed level gives 23.9 s' if tag == 'minus' else 'zeros of the printed transfer matrix'))
        rga = G[0, 0]*G[1, 1]/(G[0, 0]*G[1, 1]-G[0, 1]*G[1, 0])
        out.append(row(f'P{"-" if tag == "minus" else "+"} relative gain', d['rga'], rga, .01))
    return out


def _ph():
    u, d, out = Unit('ph'), PP.PH, []
    xs, ok = u.steady()
    g = dict(zip(u.xname, xs))
    out.append(row('nominal level (cm)', d['h'], g['h'], .05))
    out.append(row('nominal pH', d['pH'], g['pH'], .05))
    out.append(row('effluent invariant Wa4 (mol/l)', d['wa4'], g['wa'], .05e-4, 'the paper prints -4.32e-4; mixing its three streams gives -4.36e-4'))
    out.append(row('effluent invariant Wb4 (mol/l)', d['wb4'], g['wb'], .005e-4))
    # the second dissociation constant: read off the engine's charge balance at high pH
    x = xs.copy()
    x[u.xname.index('pH')], x[u.xname.index('wa')], x[u.xname.index('wb')] = 10.25, 0., 1e-3
    f = u.f(x, u.w0, u.p0)[u.xname.index('pH')]
    res = lambda pk2: PP.ph(dict(zip(u.wname, u.w0)) | dict(zip(u.xname, x)) | dict(pk2=pk2))['pH']-f      # noqa: E731
    lo, hi = 9., 12.
    for _ in range(60):
        mid = (lo+hi)/2
        lo, hi = (mid, hi) if res(lo)*res(mid) > 0 else (lo, mid)
    out.append(row('second dissociation constant pK2', PP.PH_PK2, (lo+hi)/2, .005))
    return out


def _vdv():
    u, d = Unit('vdv'), PP.VDV
    xs, ok = u.steady(u.inputs(r=d['r'], QK=d['QK'], cA0=d['cA0'], T0=d['T0']))
    return [row('operating point cA, cB (mol/l)', (d['cA'], d['cB']), xs[:2], .005), row('operating point T, TK (degC)', (d['T'], d['TK']), xs[2:], .05)]


def _excstr():
    u, d = Unit('excstr'), PP.CSTR
    xs, ok = u.steady(u.inputs(Tsp=d['T'], q=d['q'], CAf=d['CAf'], Tf=d['Tf']))
    g = dict(zip(u.xname, xs))
    return [row('operating point CA (mol/l)', d['CA'], g['CA'], .005), row('coolant temperature that holds 350 K (K)', d['Tc'], g['Tj'], .05)]


def _bio():
    u, d, out = Unit('bio'), PP.BIO, []
    xs, ok = u.steady(u.inputs(D=d['D'], Sf=d['Sf']))
    out.append(row('stable steady state X, S (g/l)', (d['X'], d['S']), xs[:2], 5e-5))
    for X, S in d['other']:
        x = np.array([X, S, 0.])
        f = u.f(x, u.w0, u.p0)[:2]
        out.append(row(f'the paper\'s other steady state ({X}, {S}): residual (g/l/h)', (0., 0.), f, 2e-4, 'printed to four decimals'))
    return out


def _evap():
    u, d, out = Unit('evap'), PP.EVAP, []
    n = d['nominal']
    x = np.array([n['L2'], n['X2'], n['P2']])
    w = u.inputs(F2=n['F2'], P100=n['P100'], F200=n['F200'], F1=n['F1'], X1=n['X1'], T1=n['T1'], T200=n['T200'])
    f, y, A, B, C, D = u.lin(x, w, u.p0)
    out.append(row('derivatives at the published operating point (per min)', (0., 0., 0.), f, 2e-3, 'the table is printed to three or four digits'))
    out.append(row('cooling water outlet T201 (degC)', n['T201'], y[u.yname.index('T201')], .06))
    ref = PP.evap(dict(zip(u.wname, w)) | dict(zip(u.xname, x)))
    out.append(row('vapour flow F4 and duties Q100, Q200 from the paper\'s equations (table values)', (n['F4'], n['Q100'], n['Q200']), (ref['_F4'], ref['_Q100'], ref['_Q200']), .5,
                   'the paper\'s equations against its own table (not the engine)'))
    return out


def _column():
    u, d, out = Unit('column'), PP.COLUMN, []
    xs, ok = u.steady()
    out.append(row('steady compositions xD, xB at the published flows', (d['xD'], d['xB']), (xs[40], xs[0]), 5e-5))
    G = u.dcgain(xs, u.w0, u.p0)
    i, j = [u.yname.index('yD'), u.yname.index('yB')], [u.wname.index('LT'), u.wname.index('VB')]
    Gx = np.array([[(1-xs[40])*LN10], [-xs[0]*LN10]])*G[np.ix_(i, j)]*100.      # from log impurities to 0.01 mole fraction
    out.append(row('steady gains d(yD, xB)/d(L, V) (0.01 mole fraction per kmol/min)', d['gain'], Gx, .5, 'printed to one decimal'))
    rga = Gx[0, 0]*Gx[1, 1]/(Gx[0, 0]*Gx[1, 1]-Gx[0, 1]*Gx[1, 0])
    out.append(row('relative gain of the LV configuration', d['rga'], rga, 1., 'the paper computes it from the rounded gains (35.1); the unrounded model gives about 36'))
    lam = u.modes(xs, u.w0, u.p0)
    out.append(row('slowest eigenvalue (1/min)', d['eig1'], lam.real.max(), 1e-6, 'the page linearises the code numerically'))
    return out


PUBLISHED = dict(qtank=_qtank, ph=_ph, vdv=_vdv, excstr=_excstr, bio=_bio, evap=_evap, column=_column)


def published(name):
    return PUBLISHED[name]() if name in PUBLISHED else None


def _dcgain_at(self, x, w, p):
    """Steady gains of the model linearised at a point that need not be steady (as a paper does at printed values)."""
    _, _, A, B, C, D = self.lin(x, w, p)
    return D-C@np.linalg.solve(A, B)


Unit.dcgain_at = _dcgain_at


# ---------------------------------------------------------------- conservation
def _ledgers(name, u, x, w, p):
    """[(what, rate of change of the inventory by the engine's derivatives, in - out written from the flowsheet, throughput)]"""
    f = dict(zip(u.xname, u.f(x, w, p)))
    s, i, q = dict(zip(u.xname, x)), dict(zip(u.wname, w)), dict(zip(u.pname, p))
    if name == 'qtank':
        A, a, g = (28., 32., 28., 32.), (.071, .057, .071, .057), 981.
        acc = sum(A[k]*f[f'h{k+1}'] for k in range(4))
        fin = q['k1']*i['v1']+q['k2']*i['v2']
        out = sum(q[f'c{k+1}']*a[k]*math.sqrt(2*g*s[f'h{k+1}']) for k in range(2))
        return [('water', acc, fin-out, fin)]
    if name == 'ph':
        A = 207.
        q4 = 4.59*q['cv']*(s['h']+11.5)**.607
        fin = i['q1']+i['q2']+i['q3']
        wa = i['q1']*3e-3*i['sa']+i['q2']*-3e-2+i['q3']*-3.05e-3*i['sb']
        wb = i['q2']*3e-2+i['q3']*5e-5*i['sb']
        return [('volume', A*f['h'], fin-q4, fin),
                ('charge invariant Wa', A*(f['h']*s['wa']+s['h']*f['wa']), wa-q4*s['wa'], fin*3e-3),
                ('carbonate invariant Wb', A*(f['h']*s['wb']+s['h']*f['wb']), wb-q4*s['wb'], fin*3e-3)]
    if name == 'gasnet':
        C1, RV = 6.65e-4, 30.
        valve = lambda op, hi, lo, cv: cv*RV**(op-1)*math.sqrt(max(hi-lo, 0.)*(hi+lo)/2+1e-6)      # noqa: E731
        qin, qout = valve(i['u1'], i['P0'], s['P1'], 4.17*q['cv1']), valve(i['u2'], s['P2'], i['Plp'], 1.395)
        return [('gas in the two volumes', (f['P1']+f['P2'])/C1, qin-i['qd']-qout, max(qin, 1.))]
    if name == 'hex':
        n = sum(k.startswith('Th') and k != 'Thc' for k in u.xname)
        CPH, CPC, CPW, MH, MC, MH2, MW = 2.5, 4.18, 4.18, 150., 200., 400., 300.
        acc = sum(MH*CPH*f[f'Th{k}']+MC*CPC*f[f'Tc{k}'] for k in range(n))+MH2*CPH*f['Thc']+MW*CPW*f['Tw']
        b, Fw = min(max(i['bypass'], 0.), 1.), max(i['Fw'], .1)
        fin = i['Fh']*CPH*i['Th_in']+i['Fc']*CPC*i['Tc_in']+Fw*CPW*i['Tw_in']
        out = i['Fh']*CPH*s['Thc']+i['Fc']*CPC*s['Tc0']+Fw*CPW*s['Tw']
        del b
        return [('heat', acc, fin-out, i['Fh']*CPH*(i['Th_in']-i['Tc_in']))]
    if name == 'furnace':
        n = sum(k.startswith('TA') for k in u.xname)
        CG, CPG, PHI, LHV, AFR, CPF, MCELL = 3000., 1.15, .4, 48000., 17.2, 2.8, 400.
        acc = CG*f['Tg']+MCELL*CPF*sum(f[f'TA{k}']+f[f'TB{k}'] for k in range(n))
        fuel, air = i['fuel'], i['air']
        lam = air/(max(fuel, 1e-3)*AFR*i['lhv'])
        fire = fuel*LHV*i['lhv']*min(lam, 1.)
        FA, FB = i['split']*i['Ft'], (1-i['split'])*i['Ft']
        fin = fire+i['Ft']*CPF*i['Tin']
        out = (fuel+air)*CPG*(s['Tg']-25.)*PHI+FA*CPF*s[f'TA{n-1}']+FB*CPF*s[f'TB{n-1}']
        return [('heat', acc, fin-out, fire)]
    if name == 'evap':
        T2 = .5616*s['P2']+.3126*s['X2']+48.43
        F4 = (.16*q['ua1']*(i['F1']+50.)*(.1538*i['P100']+90.-T2)-.07*i['F1']*(T2-i['T1']))/38.5      # the vapour boiled off, by the heat balance
        return [('liquid in the separator', 20.*f['L2'], i['F1']-F4-i['F2'], i['F1']),
                ('solute (the paper holds the mass at 20 kg whatever the level)', 20.*f['X2'], i['F1']*i['X1']-i['F2']*s['X2'], i['F1']*i['X1'])]
    if name == 'column':
        xx, M = x[:41], x[41:]
        fx, fM = np.array([f[f'x{k}'] for k in range(41)]), np.array([f[f'M{k}'] for k in range(41)])
        D, B = .5+10.*(M[40]-.5), .5+10.*(M[0]-.5)
        return [('total moles', fM.sum(), i['F']-D-B, i['F']),
                ('light component', (M*fx+xx*fM).sum(), i['F']*i['zF']-D*xx[40]-B*xx[0], i['F']*i['zF'])]
    return []


NO_LEDGER = dict(vdv='the by-products C and D are not states', excstr='the product is not a state', bio='yield model: no element balance')


def conservation(name, n=200, seed=1):
    """Per ledger: the largest gap between the inventory's rate by the engine and in - out by the flowsheet,
    as a share of the throughput, over n random points (engine parameters and loads included)."""
    if name in NO_LEDGER:
        return dict(none=NO_LEDGER[name])
    u, rng = Unit(name), np.random.default_rng(seed)
    X, W = _points(u, n, rng, name)
    if name == 'ph':
        for q in ('sa', 'sb'):
            W[:, u.wname.index(q)] = rng.uniform(.8, 1.2, n)
    worst = {}
    for x, w in zip(X, W):
        p = np.array([rng.uniform(s['lo'], s['hi']) for s in u.P.ps]) if u.P.ps else u.p0
        for what, acc, net, through in _ledgers(name, u, x, w, p):
            worst[what] = max(worst.get(what, 0.), abs(acc-net)/abs(through))
    return worst

"""Named plants as data for the mechanism engine (docs/sim.md, 'Plants').

Each function writes a plant's published equations once and returns the `Plant` (a table). Sources are in
the docstrings; the older per-family environments these replace are compared equation by equation in
tests/test_engine.py.

The decision interval `dt` is an APC interval for the unit: the slowest published pair takes 10-40 decisions
to 63 % of its answer. `substeps` keeps the outputs of a run with the MVs moving within half a band of a
32-substep run (the median episode: within a third). Outputs carry what the task layer (aio.sim.mech) needs:
range, band, whether tracked and inside which targets, limits, `integ` for a level that does not regulate
itself.
"""
import math

from aio.sim.engine import Plant
from aio.sim.expr import exp, log, sqrt, abs_, step, relu, minimum, maximum, clamp

LN10 = math.log(10.)
INF = float('inf')


def pow10(a):
    return exp(LN10*a)


def qtank():
    """Johansson's quadruple-tank process (IEEE TCST 8(3), 2000): pump 1 feeds tanks 1 and 4, pump 2 tanks 2
    and 3, the upper tanks drain into the lower ones. gamma1 + gamma2 < 1 puts the multivariable zero in
    the right half plane. Units cm, s, V."""
    P = Plant('qtank', dt=10., substeps=2)
    A, a, G = (28., 32., 28., 32.), (.071, .057, .071, .057), 981.
    v1, v2 = P.mv('v1', 0., 10., .5, 3.), P.mv('v2', 0., 10., .5, 3.)
    g1, g2 = P.param('g1', .70, .2, .9), P.param('g2', .60, .2, .9)
    k1, k2 = P.param('k1', 3.33, 2.66, 3.66), P.param('k2', 3.35, 2.68, 3.69)
    c = [P.param(f'c{i+1}', 1., .85, 1.1) for i in range(4)]
    h = [P.state(f'h{i+1}', x0, lo=0., hi=20.) for i, x0 in enumerate((12.4, 12.7, 1.8, 1.4))]
    q = [c[i]*a[i]*sqrt(2*G*h[i]) for i in range(4)]
    f1, f2 = k1*v1, k2*v2
    P.der(h[0], (-q[0]+q[2]+g1*f1)/A[0])
    P.der(h[1], (-q[1]+q[3]+g2*f2)/A[1])
    P.der(h[2], (-q[2]+(1-g2)*f2)/A[2])
    P.der(h[3], (-q[3]+(1-g1)*f1)/A[3])
    for i in range(4):
        P.out(f'h{i+1}', h[i], range=((2., 19.) if i < 2 else (.2, 19.)), band=.5, track=i < 2, lo=2. if i < 2 else .2, hi=19., target=(6., 17.))
    return P


def ph():
    """pH neutralisation (Henson & Seborg, IEEE TCST 2(3), 1994): acid q1, buffer q2 and base q3 into a
    stirred tank with gravity outflow; states are the level and the reaction invariants Wa, Wb, and the pH
    is the root of the charge balance (an algebraic state). Units cm, ml/s, mol/l."""
    P = Plant('ph', dt=10., substeps=2)
    AREA, Z, NEXP, CV = 207., 11.5, .607, 4.59
    WA1, WA2, WA3, WB1, WB2, WB3, PK2 = 3e-3, -3e-2, -3.05e-3, 0., 3e-2, 5e-5, 10.25
    q3, q1 = P.mv('q3', 0., 40., 1., 15.6), P.mv('q1', 0., 40., 1., 16.6)
    q2 = P.load('q2', .55, .3, .8)
    sa, sb = P.load('sa', 1., .8, 1.2), P.load('sb', 1., .8, 1.2)
    pk1, cv = P.param('pk1', 6.35, 6.0, 6.7), P.param('cv', 1., .85, 1.1)
    h = P.state('h', 14., lo=.5, hi=40.)
    wa, wb = P.state('wa', -4.36e-4, scale=1e-3), P.state('wb', 5.28e-4, lo=0., scale=1e-3)     # carbonate cannot be negative (the charge balance is monotone in pH for Wb >= 0)
    pH = P.state('pH', 7., lo=0., hi=14., scale=1e-3, algebraic=True, maxstep=1.)
    hv = maximum(h, 1e-3)
    q4 = CV*cv*exp(NEXP*log(h+Z))
    vol = AREA*hv
    P.der(h, (q1+q2+q3-q4)/AREA)
    P.der(wa, (q1*(WA1*sa-wa)+q2*(WA2-wa)+q3*(WA3*sb-wa))/vol)
    P.der(wb, (q1*(WB1-wb)+q2*(WB2-wb)+q3*(WB3*sb-wb))/vol)
    a, b = pow10(pk1-pH), pow10(pH-PK2)
    P.der(pH, wa+pow10(pH-14.)-pow10(-pH)+wb*(1+2*b)/(1+a+b))
    P.out('pH', pH, range=(4., 10.), band=.2, track=True, target=(5., 9.))
    P.out('h', h, range=(3., 30.), band=1., track=True, target=(8., 22.), lo=3., hi=30.)
    return P


def vdv():
    """Van de Vusse reactor with cooling jacket (Klatt & Engell 1998; Chen et al. 1995): A -> B -> C, 2A -> D.
    The yield of B has a maximum in the dilution rate, so the gain changes sign. Time in h."""
    P = Plant('vdv', dt=20., time_unit=3600.)
    K10, K20, K30, E1, E2, E3 = 1.287e12, 1.287e12, 9.043e9, -9758.3, -9758.3, -8560.
    DHAB, DHBC, DHAD = 4.2, -11.0, -41.85
    RHO, CP, KW, AR, VR, MK, CPK = .9342, 3.01, 4032., .215, 10., 5., 2.
    r, QK = P.mv('r', 3., 35., 1., 14.19), P.mv('QK', -9000., 0., 300., -1113.5)
    cA0, T0 = P.load('cA0', 5.1, 4.6, 5.6), P.load('T0', 104.9, 98., 110.)
    kw, act = P.param('kw', 1., .75, 1.05), P.param('act', 1., .85, 1.1)
    cA, cB = P.state('cA', 2.14, lo=0.), P.state('cB', 1.09, lo=0.)
    T, TK = P.state('T', 114.2), P.state('TK', 112.9)
    Tk = T+273.15
    k1, k2, k3 = act*K10*exp(E1/Tk), act*K20*exp(E2/Tk), act*K30*exp(E3/Tk)
    P.der(cA, r*(cA0-cA)-k1*cA-k3*cA*cA)
    P.der(cB, -r*cB+k1*cA-k2*cB)
    P.der(T, r*(T0-T)-(k1*cA*DHAB+k2*cB*DHBC+k3*cA*cA*DHAD)/(RHO*CP)+kw*KW*AR*(TK-T)/(RHO*CP*VR))
    P.der(TK, (QK+kw*KW*AR*(T-TK))/(MK*CPK))
    P.out('cB', cB, range=(.5, 1.2), band=.02, track=True, target=(.8, 1.12))
    P.out('T', T, range=(90., 140.), band=.5, track=True, target=(105., 125.), hi=140.)
    P.out('cA', cA, range=(.8, 4.), band=.05, lo=.8)
    P.out('TK', TK, range=(80., 140.), band=.5)
    return P


def excstr():
    """Exothermic CSTR (Seborg et al.; Hicks & Ray) with a jacket-temperature PI below the MVs: the MVs are the
    reactor temperature setpoint and the feed rate, the inner loop saturates at the coolant limits (with
    anti-windup). Open-loop unstable without the inner loop. Time in min."""
    P = Plant('excstr', dt=5., time_unit=60., substeps=4, refresh=True)
    KB, TIB, TC0, TAUJ = 5., 1., 300., .1
    Tsp, q = P.mv('Tsp', 330., 375., .25, 350.), P.mv('q', 60., 140., 1., 100.)
    CAf, Tf = P.load('CAf', 1., .9, 1.1), P.load('Tf', 350., 340., 360.)
    ua, act = P.param('ua', 1., .8, 1.05), P.param('act', 1., .85, 1.15)
    CA, T = P.state('CA', .5, lo=0., hi=2.), P.state('T', 350.)
    Tj, I = P.state('Tj', 300.), P.state('I', 0., scale=10.)
    raw = TC0+KB*(Tsp-T)+KB/TIB*I
    Tc = clamp(raw, 260., 340.)
    k = act*7.2e10*exp(-8750./T)
    e = Tsp-T
    P.der(CA, q/100.*(CAf-CA)-k*CA)
    P.der(T, q/100.*(Tf-T)+5e4/(1000.*.239)*k*CA+ua*5e4/(100.*1000.*.239)*(Tj-T))
    P.der(Tj, (Tc-Tj)/TAUJ)
    P.der(I, e*(1-step(raw-340.)*step(e)-step(260.-raw)*step(-e)))
    P.out('CA', CA, range=(0., 1.), band=.01, track=True, target=(.25, .65))
    P.out('prod', q*(CAf-CA), range=(20., 80.), band=.5, track=True, target=(35., 70.))
    P.out('Tc', Tc, range=(270., 330.), band=.5, lo=270., hi=330.)
    P.out('T', T, range=(320., 380.), band=.5)
    return P


def bio():
    """Continuous fermenter with substrate inhibition (Henson & Seborg 1997, after Jobses): biomass X,
    substrate S, product P; the growth rate has a maximum in S. Time in h."""
    P = Plant('bio', dt=900., time_unit=3600., substeps=2)
    D, Sf = P.mv('D', .05, .6, .03, .3), P.mv('Sf', 1.5, 8., .3, 4.)
    mum, Km, K1 = P.param('mum', .53, .45, .6), P.param('Km', .12, .1, .15), P.param('K1', .4545, .35, .55)
    Y, alpha = P.param('Y', .4, .36, .44), P.param('alpha', 2.2, 1.8, 2.6)
    X, S, Pr = P.state('X', 1.53, lo=0.), P.state('S', .174, lo=0.), P.state('P', 4.386, lo=0., scale=5.)
    Sp = relu(S)
    mu = mum*Sp/(Km+Sp+K1*Sp*Sp)
    P.der(X, (mu-D)*X)
    P.der(S, D*(Sf-S)-mu*X/Y)
    P.der(Pr, -D*Pr+(alpha*mu+.2)*X)
    P.out('X', X, range=(0., 3.), band=.02, track=True, target=(1., 2.6))
    P.out('P', Pr, range=(0., 12.), band=.1, track=True, target=(4., 10.))
    P.out('S', S, range=(0., 1.), band=.02, hi=.4)
    return P


def evap():
    """Forced-circulation evaporator (Newell & Lee 1989): separator level (integrating), product composition
    and operating pressure by product flow, steam pressure and cooling water. Time in min."""
    P = Plant('evap', dt=60., time_unit=60., substeps=2)
    F2, P100, F200 = P.mv('F2', 0., 5., .5, 2.), P.mv('P100', 100., 400., 20., 194.7), P.mv('F200', 0., 400., 40., 208.)
    F1 = P.load('F1', 10., 8.5, 11.5, measured=True)
    X1, T1, T200 = P.load('X1', 5., 4.5, 5.5), P.load('T1', 40., 35., 45.), P.load('T200', 25., 20., 30.)
    ua1, ua2 = P.param('ua1', 1., .85, 1.05), P.param('ua2', 1., .8, 1.05)
    L2 = P.state('L2', 1., lo=0., hi=3.)
    X2, P2 = P.state('X2', 25., lo=.1, hi=100.), P.state('P2', 50.5, lo=1., hi=300.)
    F2 = F2*clamp(L2/.1, 0., 1.)                                  # the product pump loses suction as the separator runs empty (the published model lets the level go negative)
    Fc = maximum(F200, 1.)
    T2 = .5616*P2+.3126*X2+48.43
    T3 = .507*P2+55.
    T100 = .1538*P100+90.
    Q100 = .16*ua1*(F1+50.)*(T100-T2)
    F4 = (Q100-.07*F1*(T2-T1))/38.5
    UA2 = 6.84*ua2
    Q200 = UA2*(T3-T200)/(1+UA2/(.14*Fc))
    F5 = Q200/38.5
    P.der(L2, (F1-F4-F2)/20.)
    P.der(X2, (F1*X1-F2*X2)/20.)
    P.der(P2, (F4-F5)/4.)
    P.out('L2', L2, range=(.3, 1.8), band=.05, track=True, target=(.7, 1.3), lo=.3, hi=1.8, integ=True)     # a level with a pumped outflow: it integrates
    P.out('X2', X2, range=(10., 40.), band=.5, track=True, target=(20., 32.))
    P.out('P2', P2, range=(30., 80.), band=1., track=True, target=(42., 62.))
    P.out('T201', T200+Q200/(.07*Fc), range=(25., 50.), band=1., hi=50.)
    P.out('F1', F1, range=(6., 14.), band=1., dv=True)
    return P


def furnace(ncell=4):
    """Two-pass fired heater: fuel, air and the pass split hold the outlet temperature, the pass balance and
    the excess oxygen; radiant transfer to both coils (T^4), a firebox energy balance, and a lagged oxygen
    analyser. Time in s."""
    P = Plant('furnace', dt=5.)
    CG, CPG, PHI, LHV, AFR, CPF, MCELL, KAPPA = 3000., 1.15, .4, 48000., 17.2, 2.8, 400., 1.42e-9
    f, a, s = P.mv('fuel', .15, .8, .005, .42), P.mv('air', 3., 18., .1, 8.4), P.mv('split', .3, .7, .003, .5)
    Ft = P.load('Ft', 50., 42., 58., measured=True)
    Tin, lhv = P.load('Tin', 250., 232., 268.), P.load('lhv', 1., .92, 1.06)
    kA, kB = P.param('kA', 1., .8, 1.05), P.param('kB', 1., .8, 1.05)
    Tg = P.state('Tg', 850., scale=800.)
    TA = [P.state(f'TA{i}', x0, scale=300.) for i, x0 in enumerate((280., 310., 340., 370.)[:ncell])]
    TB = [P.state(f'TB{i}', x0, scale=300.) for i, x0 in enumerate((280., 310., 340., 370.)[:ncell])]
    o2m = P.state('o2', 3.)
    lam = a/(maximum(f, 1e-3)*AFR*lhv)
    Tg4 = (Tg+273.15)**4
    QA = [KAPPA*kA*(Tg4-(t+273.15)**4) for t in TA]
    QB = [KAPPA*kB*(Tg4-(t+273.15)**4) for t in TB]
    fire = f*LHV*lhv*minimum(lam, 1.)
    P.der(Tg, (fire-(f+a)*CPG*(Tg-25.)*PHI-sum(QA[1:], QA[0])-sum(QB[1:], QB[0]))/CG)
    FA, FB = s*Ft, (1-s)*Ft
    for i in range(ncell):
        P.der(TA[i], (FA*CPF*((TA[i-1] if i else Tin)-TA[i])+QA[i])/(MCELL*CPF))
        P.der(TB[i], (FB*CPF*((TB[i-1] if i else Tin)-TB[i])+QB[i])/(MCELL*CPF))
    P.der(o2m, (20.9*(lam-1)/lam-o2m)/20.)
    P.out('COT', s*TA[-1]+(1-s)*TB[-1], range=(330., 410.), band=1., track=True, target=(355., 385.))
    P.out('dT', TA[-1]-TB[-1], range=(-20., 20.), band=1., track=True, target=(-6., 6.))
    P.out('O2', o2m, range=(0., 8.), band=.2, track=True, target=(1.5, 4.5))
    P.out('TA', TA[-1], range=(300., 420.), band=1., hi=400.)
    P.out('TB', TB[-1], range=(300., 420.), band=1., hi=400.)
    P.out('Ft', Ft, range=(35., 65.), band=.5, dv=True)
    return P


def gasnet():
    """Gas pressure-reducing station with a line: inlet valve from a high-pressure header, a line with
    inertia and friction, a delivery volume with a demand and a relief valve to a low-pressure system.
    Equal-percentage valves, compressible flow. Units bar, kg/s, s."""
    P = Plant('gasnet', dt=60., substeps=3)
    C1, CA, RV, CV1, CV2, KF = 6.65e-4, 1.6, 30., 4.17, 1.395, 1./1600.
    u1, u2 = P.mv('u1', 0., 1., .1, .6), P.mv('u2', 0., 1., .1, .5)
    P0, Plp = P.load('P0', 80., 74., 86.), P.load('Plp', 20., 16., 24.)
    qd = P.load('qd', 30., 24., 36., measured=True)
    kf, cv1 = P.param('kf', 1., .5, 2.), P.param('cv1', 1., .85, 1.05)
    P1, q, P2 = P.state('P1', 60.), P.state('q', 40.), P.state('P2', 59.)

    def valve(u, hi, lo, cv):
        return cv*exp(math.log(RV)*(u-1))*sqrt(relu(hi-lo)*(hi+lo)/2+1e-6)
    qin, qout = valve(u1, P0, P1, CV1*cv1), valve(u2, P2, Plp, CV2)
    P.der(P1, C1*(qin-q))
    P.der(q, CA*(P1-P2-KF*kf*q*abs_(q)))
    P.der(P2, C1*(q-qd-qout))
    P.out('P2', P2, range=(50., 66.), band=.2, track=True, target=(56., 62.))     # 0.2 bar: a band of 0.1 asks the valves for 0.08 % of their travel
    P.out('q_in', qin, range=(20., 60.), band=.5, track=True, target=(32., 48.))
    P.out('P1', P1, range=(50., 66.), band=.2, hi=66.)
    P.out('qd', qd, range=(20., 40.), band=.5, dv=True)
    return P


def hexnet(n=8):
    """Heat-exchanger train with a bypass: a counter-current exchanger in n cells, a hot-side bypass that
    mixes around it, and a trim cooler on cooling water. Units kW, kg/s, s."""
    P = Plant('hex', dt=10., substeps=2)
    CPH, CPC, CPW, MH, MC, MH2, MW = 2.5, 4.18, 4.18, 150., 200., 400., 300.
    b, Fw = P.mv('bypass', 0., .6, .01, .2), P.mv('Fw', 1., 40., .5, 15.)
    Th_in = P.load('Th_in', 150., 135., 165., measured=True)
    Fh, Fc = P.load('Fh', 10., 8., 12.), P.load('Fc', 12., 10., 14.)
    Tc_in, Tw_in = P.load('Tc_in', 30., 22., 38.), P.load('Tw_in', 25., 18., 30.)
    ua = P.param('ua', 1., .6, 1.05)
    Th = [P.state(f'Th{i}', x0, scale=100.) for i, x0 in enumerate((128.3, 110.1, 94.9, 82.2, 71.5, 62.7, 55.2, 49.0)[:n])]
    Tc = [P.state(f'Tc{i}', x0, scale=100.) for i, x0 in enumerate((70.3, 61.6, 54.3, 48.3, 43.2, 39., 35.4, 32.5)[:n])]
    Thc, Tw = P.state('Thc', 47.4, scale=100.), P.state('Tw', 33.7, scale=100.)
    bb, Fww = clamp(b, 0., 1.), maximum(Fw, .1)
    UAk = 60.*ua/n
    for i in range(n):
        qx = UAk*(Th[i]-Tc[i])
        P.der(Th[i], (Fh*(1-bb)*CPH*((Th[i-1] if i else Th_in)-Th[i])-qx)/(MH*CPH))
        P.der(Tc[i], (Fc*CPC*((Tc[i+1] if i < n-1 else Tc_in)-Tc[i])+qx)/(MC*CPC))
    Tmix = bb*Th_in+(1-bb)*Th[-1]
    qc = 40.*exp(.8*log(Fww/15.))*(Thc-Tw)
    P.der(Thc, (Fh*CPH*(Tmix-Thc)-qc)/(MH2*CPH))
    P.der(Tw, (Fww*CPW*(Tw_in-Tw)+qc)/(MW*CPW))
    P.out('Thc', Thc, range=(30., 120.), band=.5, track=True, target=(40., 62.))
    P.out('Tc_out', Tc[0], range=(30., 120.), band=.5, track=True, target=(60., 77.))
    P.out('Tw', Tw, range=(15., 50.), band=.5, hi=45.)
    P.out('Th_in', Th_in, range=(120., 180.), band=.5, dv=True)
    return P


def column(nt=41, nf=21):
    """Skogestad's distillation column A (Skogestad & Morari 1988; colamod.m): binary, 41 stages with reboiler
    and total condenser, constant relative volatility, linearised tray hydraulics, level controllers on
    both ends (LV configuration). The purities are log impurities. Time in min."""
    P = Plant('column', dt=300., time_unit=60., substeps=2)
    M0, TAUL, KC, L0, D0, B0 = .5, .063, 10., 2.70629, .5, .5
    L0b = L0+1.
    LT, VB = P.mv('LT', 2.2, 3.2, .05, 2.70629), P.mv('VB', 2.7, 3.7, .05, 3.20629)
    F = P.load('F', 1., .85, 1.15, measured=True)
    zF, qF = P.load('zF', .5, .4, .6), P.load('qF', 1., .85, 1.05)
    alpha = P.param('alpha', 1.5, 1.42, 1.6)
    x = [P.state(f'x{i}', .01+.98*i/(nt-1), lo=0., hi=1., scale=1.) for i in range(nt)]
    M = [P.state(f'M{i}', M0, lo=.01, scale=.5) for i in range(nt)]
    y = [alpha*x[i]/(1+(alpha-1)*x[i]) for i in range(nt)]
    L = [(L0b if i <= nf-1 else L0)+(M[i]-M0)/TAUL for i in range(nt-1)]+[LT]      # liquid leaving stage i
    V = [VB if i < nf-1 else VB+(1-qF)*F for i in range(nt-1)]+[0.]                # vapour leaving stage i (none from the condenser)
    D, B = D0+KC*(M[-1]-M0), B0+KC*(M[0]-M0)
    for i in range(nt):
        Lin, xin = (L[i+1], x[i+1]) if i < nt-1 else (0., 0.)
        Vin, yin = (V[i-1], y[i-1]) if i else (0., 0.)
        Lout = B if i == 0 else (LT+D if i == nt-1 else L[i])
        dM = Lin-Lout+Vin-V[i]
        dMx = Lin*xin-Lout*x[i]+Vin*yin-V[i]*y[i]
        if i == nf-1:
            dM, dMx = dM+F, dMx+F*zF
        P.der(M[i], dM)
        P.der(x[i], (dMx-x[i]*dM)/maximum(M[i], 1e-3))
    xd, xb = clamp(x[-1], 1e-9, 1-1e-9), clamp(x[0], 1e-9, 1-1e-9)
    P.out('yD', -log(1-xd)/LN10, range=(1., 3.5), band=.1, track=True, target=(1.6, 2.6))     # 0.1 in log10 of the impurity: 0.05 asks the flows for 0.05 % of their range
    P.out('yB', -log(xb)/LN10, range=(1., 3.5), band=.1, track=True, target=(1.6, 2.6))
    P.out('x31', clamp(x[30], 1e-9, 1-1e-9), range=(0., 1.), band=.01)
    P.out('x10', clamp(x[9], 1e-9, 1-1e-9), range=(0., 1.), band=.01)
    P.out('F', F, range=(.8, 1.2), band=.01, dv=True)
    return P


def btcol():
    """Benzene-toluene column at atmospheric pressure, 20 theoretical stages: built of the flowsheet units
    (aio.sim.flow.column), Peng-Robinson properties."""
    from aio.sim import flow, props
    pk = props.package('bt')
    return flow.column('btcol', pk, 20, 10, 100., [.5, .5], pk.bubble_T([.5, .5], 115.)-5., 105., 50., 110., 'benzene', 'toluene')


def meoh():
    """Methanol-water column at atmospheric pressure, 18 theoretical stages, NRTL: a non-ideal mixture (the
    volatility falls towards the methanol end), low-pressure steam."""
    from aio.sim import flow, props
    pk = props.package('meoh')
    z = [.4, .6]
    return flow.column('meoh', pk, 18, 8, 200., z, pk.bubble_T(z, 115.)-5., 105., 80., 90., 'methanol', 'water', Ph=500.)


def debut():
    """Debutaniser: C3-C5 feed split between n-butane and isopentane at 1.1 MPa, 30 theoretical stages, five
    species, Peng-Robinson with pressure-dependent volatilities and vapour enthalpies."""
    from aio.sim import flow, props
    pk = props.package('c3c5')
    z = [.05, .15, .25, .20, .35]
    return flow.column('debut', pk, 30, 15, 500., z, pk.bubble_T(z, 1150.)-10., 1100., 225., 450., 'n-butane', 'isopentane')


def _late(module, name):
    """A plant of another file, built when it is asked for."""
    def build():
        from importlib import import_module
        return getattr(import_module(module), name)()
    return build


# The default pool (mech.pool()): what a batch with a share of mechanistic episodes is drawn from, and so what a run can train on.
PLANTS = dict(qtank=qtank, ph=ph, vdv=vdv, excstr=excstr, bio=bio, evap=evap, furnace=furnace, gasnet=gasnet, hex=hexnet, column=column)
FIRST = tuple(PLANTS)            # the pool as it was until 11-12: the exam `mech` (aio.sim.stay) and aio.sim.validate are of these ten
# In the pool since 11-12, the five together (one change of the fixed batches): building heating, centrifugal compressor, reactor with column and
# recycle, ammonia converter, fired heater. The exam `mech_new` is of these five.
PLANTS.update(bldg=_late('aio.sim.plants_bldg', 'bldg'), comp=_late('aio.sim.plants_comp', 'comp'), rsr=_late('aio.sim.plants_rsr', 'rsr'), amm=_late('aio.sim.plants_amm', 'amm'),
              heater=_late('aio.sim.plants_htr', 'heater'))
NEW = tuple(q for q in PLANTS if q not in FIRST)
# Large plants: asked for by name (mech.pool(['tep'])), never in the default pool.
LARGE = dict(tep=_late('aio.sim.plants_tep', 'tep'), tep9=_late('aio.sim.plants_tep', 'tep9'), bsm1=_late('aio.sim.plants_bsm', 'bsm1'), btcol=btcol, meoh=meoh, debut=debut)   # tep9: nine MVs, for a batch of a larger size (prior.SIZES)
# Held-out plants (docs/heldout.md): never trained on; mech.pool gives them only with heldout=True.
HELDOUT = {q: _late('aio.sim.plants_heldout', q) for q in ('boiler', 'wo', 'mma', 'cryst', 'boiler2')}
HELDOUT['vam'] = _late('aio.sim.plants_vam', 'vam')         # vinyl acetate plant: large MIMO exam (eight setpoints of Luyben's layer)
HELDOUT['gaslib'] = _late('aio.sim.plants_gaslib', 'gaslib')   # GasLib-24 gas network: three compressor stations and a valve
HELDOUT['refrig'] = _late('aio.sim.plants_refrig', 'refrig')   # IFAC PID 2018 refrigeration benchmark: 2x2 with seven disturbances
HELDOUT['badgwell'] = _late('aio.sim.plants_badgwell', 'heater')   # Badgwell's 5x5 linear fired heater (1998 MPC workshop)

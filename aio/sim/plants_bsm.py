"""IWA Benchmark Simulation Model No. 1 as data for the mechanism engine (docs/sim.md, 'TEP and BSM1').

BSM1 (Alex et al. 2008; Gernaey et al. 2014): five activated-sludge reactors in series (two anoxic of
1000 m3, three aerated of 1333 m3; ASM1, thirteen components each) and a ten-layer Takacs secondary settler
(1500 m2 x 4 m, feed into the sixth layer from the bottom; seven soluble components and the suspended solids
per layer), return sludge 18446 m3/d, waste sludge 385 m3/d, internal recycle from reactor 5 to reactor 1.
145 states; time in days. Kinetic and settling parameters are BSM1's (15 C); the temperature corrections of
the rates, of the oxygen saturation and of KLa are BSM2's and vanish at 15 C.

`bsm1()` is the plant an APC sees: KLa of the three aerated reactors and the internal recycle hold the
dissolved oxygen in reactors 3-5 and the nitrate at the end of the anoxic zone; the effluent ammonia, total
nitrogen and suspended solids are limits; influent flow (measured), organic and nitrogen load are loads;
temperature, aeration efficiency and settling velocity are parameters. Its nominal point is the steady
state with oxygen at 2 g/m3 and nitrate at 1 g N/m3 under the constant BSM1 influent.
`bsm1(influent=True)` has every influent component as a load and the BSM1 open-loop inputs as nominal: it
is what the published steady state and the dynamic influent files are run on (aio/sim/validate).
"""
import math

import torch

from aio.sim.engine import Plant, Mech
from aio.sim.expr import exp, relu, step, minimum, maximum

SI, SS, XI, XS, XBH, XBA, XP, SO, SNO, SNH, SND, XND, SALK = range(13)
NAMES = ('SI', 'SS', 'XI', 'XS', 'XBH', 'XBA', 'XP', 'SO', 'SNO', 'SNH', 'SND', 'XND', 'SALK')
SOL = (SI, SS, SO, SNO, SNH, SND, SALK)             # solubles carried by the settler layers
PART = (XI, XS, XBH, XBA, XP, XND)                  # particulates: in the settler they follow the solids of its feed
ORG, NIT = (SS, XI, XS, XBH, XBA, XP), (SNH, SND, XND)
VOL = (1000., 1000., 1333., 1333., 1333.)
QR, QW = 18446., 385.
AREA, HEIGHT, NL, FEED = 1500., 4., 10, 4           # layers from the top; the feed enters index 4 (sixth from the bottom)
V0MAX, V0, RH, RP, FNS, XT = 250., 474., .000576, .00286, .00228, 3000.
MU_H, K_S, K_OH, K_NO, B_H, MU_A, K_NH, K_OA, B_A, NY_G, K_A, K_H, K_X, NY_H = 4., 10., .2, .5, .3, .5, 1., .4, .05, .8, .05, 3., .1, .8
Y_H, Y_A, F_P, I_XB, I_XP, X2TSS = .67, .24, .08, .08, .06, .75
# the constant (stabilisation) influent of BSM1: flow-weighted dry-weather average
CONST_IN = (30., 69.5, 51.2, 202.32, 28.17, 0., 0., 0., 0., 31.56, 6.95, 10.59, 7.)
QIN0 = 18446.
OPEN_LOOP = (240., 240., 84., 55338.)               # KLa3, KLa4, KLa5 (1/d), internal recycle (m3/d) of the BSM1 open-loop case
KLA_MAX, QA_MAX = 360., 5*18446.
# published open-loop steady state (Alex et al. 2008, constant influent): reactors 1-5, then the solids of the settler layers from the top
PUBLISHED = dict(
    reactors=((30., 2.81, 1149., 82.1, 2552., 148., 449., .0043, 5.37, 7.92, 1.22, 5.28, 4.93),
              (30., 1.46, 1149., 76.4, 2553., 148., 450., .0000631, 3.66, 8.34, .882, 5.03, 5.08),
              (30., 1.15, 1149., 64.9, 2557., 149., 450., 1.72, 6.54, 5.55, .829, 4.39, 4.67),
              (30., .995, 1149., 55.7, 2559., 150., 451., 2.43, 9.30, 2.97, .767, 3.88, 4.29),
              (30., .889, 1149., 49.3, 2559., 150., 452., .491, 10.4, 1.73, .688, 3.53, 4.13)),
    tss=(12.5, 18.1, 29.5, 69.0, 356., 356., 356., 356., 356., 6394.))
TARGET = (2., 2., 2., 1.)                           # the nominal point of the APC plant: SO3, SO4, SO5, SNO2


def _rates(c, T):
    """ASM1 conversion rates of the thirteen components at concentrations c (a list) and temperature T."""
    d = T-15.
    arr = lambda k, k10: k*exp(math.log(k/k10)/5.*d)              # noqa: E731
    mu_h, b_h, mu_a, b_a, k_h, k_a = arr(MU_H, 3.), arr(B_H, .2), arr(MU_A, .3), arr(B_A, .03), arr(K_H, 2.5), arr(K_A, .04)
    c = [relu(v) for v in c]
    ss, so, sno, snh, snd, xnd, xs, xbh, xba = c[SS], c[SO], c[SNO], c[SNH], c[SND], c[XND], c[XS], c[XBH], c[XBA]
    xbh_, xs_ = maximum(xbh, 1e-9), maximum(xs, 1e-9)
    mon_s, o_h, i_h, n_o = ss/(K_S+ss), so/(K_OH+so), K_OH/(K_OH+so), sno/(K_NO+sno)
    p1 = mu_h*mon_s*o_h*xbh
    p2 = mu_h*mon_s*i_h*n_o*NY_G*xbh
    p3 = mu_a*(snh/(K_NH+snh))*(so/(K_OA+so))*xba
    p4, p5 = b_h*xbh, b_a*xba
    p6 = k_a*snd*xbh
    rat = xs/xbh_
    p7 = k_h*(rat/(K_X+rat))*(o_h+NY_H*i_h*n_o)*xbh
    p8 = p7*(xnd/xs_)
    return [0., (-p1-p2)/Y_H+p7, 0., (1-F_P)*(p4+p5)-p7, p1+p2-p4, p3-p5, F_P*(p4+p5),
            -(1-Y_H)/Y_H*p1-(4.57-Y_A)/Y_A*p3, -((1-Y_H)/(2.86*Y_H))*p2+p3/Y_A,
            -I_XB*(p1+p2)-(I_XB+1/Y_A)*p3+p6, -p6+p8, (I_XB-F_P*I_XP)*(p4+p5)-p8,
            -I_XB/14*p1+((1-Y_H)/(14*2.86*Y_H)-I_XB/14)*p2-(I_XB/14+1/(7*Y_A))*p3+p6/14]


def _sosat(T):
    """Oxygen saturation (g/m3), BSM2: 8 at 15 C."""
    tk = (T+273.15)/100.
    lntk = type(tk)(tk.g, tk.g.un('log', tk.i)) if hasattr(tk, 'g') else math.log(tk)
    e = exp if hasattr(tk, 'g') else math.exp
    return .9997743214*8./10.5*(56.12*6791.5*e(-66.7354+87.4755/tk+24.4526*lntk))


def _guess():
    """A starting point near the open-loop steady state: the published table, the settler's solubles as in reactor 5."""
    R = [list(r) for r in PUBLISHED['reactors']]
    S = [[R[4][c] for c in SOL]+[t] for t in PUBLISHED['tss']]
    return R, S


def _build(name, influent, x0, mv0, dt=900., substeps=48):
    P = Plant(name, dt=dt, time_unit=86400., substeps=substeps)
    kla = [P.mv(f'KLa{i}', 0., KLA_MAX, 60., mv0[i-3]) for i in (3, 4, 5)]
    qa = P.mv('Qa', 0., QA_MAX, 18446., mv0[3])
    if influent:
        cin = [P.load(f'in_{NAMES[c]}', CONST_IN[c], 0., max(4*CONST_IN[c], 1.)) for c in range(13)]
        qin = P.load('Qin', QIN0, 5000., 60000., measured=True)
    else:
        qin = P.load('Qin', QIN0, .85*QIN0, 1.15*QIN0, measured=True)
        cod, tkn = P.load('cod', 1., .85, 1.15), P.load('tkn', 1., .9, 1.1)
        cin = [CONST_IN[c]*(cod if c in ORG else (tkn if c in NIT else 1.)) for c in range(13)]
    temp, alpha, v0s = P.param('temp', 15., 12., 18.), P.param('alpha', 1., .85, 1.), P.param('v0s', 1., .75, 1.)
    R0, S0 = x0
    scale = (30., 2., 1000., 50., 2500., 150., 450., 1., 5., 5., 1., 5., 5.)
    R = [[P.state(f'{NAMES[c]}{k+1}', R0[k][c], lo=0., scale=scale[c]) for c in range(13)] for k in range(5)]
    S = [[P.state(f'{NAMES[c]}s{j+1}', S0[j][a], lo=0., scale=scale[c]) for a, c in enumerate(SOL)]+[P.state(f'TSSs{j+1}', S0[j][7], lo=0., scale=max(S0[j][7], 10.))]
         for j in range(NL)]

    q1 = (qin+qa+QR).held
    r5 = R[4]
    tss_f = (X2TSS*(r5[XI]+r5[XS]+r5[XBH]+r5[XBA]+r5[XP])).held
    frac = lambda j: S[j][7]/maximum(tss_f, 1e-6)                 # noqa: E731
    fb = frac(NL-1).held
    cr = [None]*13                                                # return sludge: the bottom layer's solubles, particulates by the feed's fractions
    for a, c in enumerate(SOL):
        cr[c] = S[NL-1][a]
    for c in PART:
        cr[c] = r5[c]*fb
    c1 = [(qin*cin[c]+qa*r5[c]+QR*cr[c])/q1 for c in range(13)]
    sosat, kfac = _sosat(temp), exp(math.log(1.024)*(temp-15.))
    air = [None, None]+[k*alpha*kfac for k in kla]
    for k in range(5):
        up = c1 if k == 0 else R[k-1]
        rate = _rates(R[k], temp)
        for c in range(13):
            d = q1/VOL[k]*(up[c]-R[k][c])+rate[c]
            if c == SO and air[k] is not None:
                d = d+air[k]*(sosat-R[k][SO])
            P.der(R[k][c], d)

    # settler (Takacs): flows per area, m/d
    qf, qu = qin+QR, QR+QW
    v_in, v_up, v_dn = qf/AREA, (qf-qu)/AREA, qu/AREA
    h = HEIGHT/NL
    X = [relu(S[j][7]) for j in range(NL)]
    xmin = FNS*tss_f
    vs = [minimum(relu(V0*v0s*(exp(-RH*(X[j]-xmin))-exp(-RP*(X[j]-xmin)))), V0MAX*v0s) for j in range(NL)]
    jt = [vs[j]*X[j] for j in range(NL)]
    js = [0.]                                                     # js[j]: solids settling from layer j-1 into layer j
    for j in range(NL-1):
        both = minimum(jt[j], jt[j+1])
        js.append(both+(1.-step(X[j+1]-XT))*(jt[j]-both) if j < FEED else both)      # above the feed: unhindered while the layer below is thin
    js.append(0.)
    feed = [r5[c] for c in SOL]+[tss_f]
    for j in range(NL):
        for a in range(8):
            if j < FEED:
                adv = v_up*(S[j+1][a]-S[j][a])
            elif j == FEED:
                adv = v_in*feed[a]-(v_up+v_dn)*S[j][a]
            else:
                adv = v_dn*(S[j-1][a]-S[j][a])
            if a == 7:
                adv = adv+js[j]-js[j+1]
            P.der(S[j][a], adv/h)

    ft = frac(0).held
    e = {c: S[0][a] for a, c in enumerate(SOL)}
    e.update({c: r5[c]*ft for c in PART})
    tn = e[SNO]+e[SNH]+e[SND]+e[XND]+I_XB*(e[XBH]+e[XBA])+I_XP*(e[XP]+e[XI])
    P.out('SO3', R[2][SO], range=(0., 8.), band=.3, track=True, target=(1., 2.5))
    P.out('SO4', R[3][SO], range=(0., 8.), band=.3, track=True, target=(1., 2.5))
    P.out('SO5', R[4][SO], range=(0., 8.), band=.3, track=True, target=(1., 2.5))
    P.out('SNO2', R[1][SNO], range=(0., 10.), band=.4, track=True, target=(.5, 3.))
    P.out('SNHe', e[SNH], range=(0., 12.), band=.5, hi=4.)
    P.out('TNe', tn, range=(0., 30.), band=1., hi=18.)
    P.out('TSSe', S[0][7], range=(0., 40.), band=1., hi=30.)
    P.out('Qin', qin, range=(10000., 26000.), band=500., dv=True)
    return P


_NOM = {}


def nominal(influent):
    """The steady state the plant is declared at (float64): the open-loop case for the validation plant, the
    point with the tracked outputs at `TARGET` for the APC plant. Solved once per process on the plant's own
    table, starting from the published open-loop table."""
    if influent not in _NOM:
        P = _build('bsm1_guess', True, _guess(), OPEN_LOOP)
        M = Mech([P], n=1, dtype=torch.float64)
        x, w, p = P.nominal()
        xo, _, ok = M.steady(w, p, iters=80)
        assert bool(ok.all()), 'BSM1: the open-loop steady state was not found'
        mv = list(OPEN_LOOP)
        if not influent:
            xo, wo, ok = M.steady(w, p, free=[0, 1, 2, 3], rows=[0, 1, 2, 3], targets=torch.tensor([TARGET], dtype=torch.float64),
                                  band=torch.tensor([[.3, .3, .3, .4]], dtype=torch.float64), start=(xo, w, p), iters=80)
            assert bool(ok.all()), 'BSM1: the nominal point of the APC plant was not found'
            mv = [float(v) for v in wo[0, :4]]
        v = [float(a) for a in xo[0]]
        _NOM[influent] = (([v[13*k:13*k+13] for k in range(5)], [v[65+8*j:73+8*j] for j in range(NL)]), mv)
    return _NOM[influent]


def bsm1(influent=False):
    x0, mv0 = nominal(influent)
    return _build('bsm1_in' if influent else 'bsm1', influent, x0, mv0)

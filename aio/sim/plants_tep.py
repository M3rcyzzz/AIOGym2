"""Tennessee Eastman as data for the mechanism engine (docs/sim.md, 'TEP and BSM1').

The process of Downs & Vogel (Comput. Chem. Eng. 17(3), 1993) written once as expressions, equation by
equation after their routine TEFUNC: reactor, product separator with condenser, stripper and the vapour
header with the recycle compressor; eight components (A, B, C gases; D, E, F, G, H condensable), four
reactions, ideal-gas and Antoine vapour-liquid equilibrium, the property constants of TEINIT. Units are
the routine's: lb-mol, hours, deg C, mmHg absolute inside, the published measurement units outside.

Two things differ from the routine, neither changes a trajectory:
  - a vessel's temperature is a state in place of its internal energy (dT/dt from the energy balance, the
    same index-one system), so there is no inner iteration; the energies are outputs of the raw plant;
  - what the routine draws or switches inside (IDV steps, random walks, measurement noise, analyser
    sampling, valve sticking, the shutdown latch) is left to the shell: feed composition and temperatures,
    cooling-water temperatures and the feed availabilities are loads, the kinetic and heat-transfer
    factors are parameters, and outputs are true values. The limits behind the shutdown are output limits.

`tep_raw()` is the open-loop process with its twelve valves (compared with the native code,
aio/sim/validate). `tep()` is the plant an APC sees: the regulatory layer of a DCS is part of the plant
(feed flow controllers, reactor and separator temperature, the three levels, reactor pressure on the
purge; the structure of Ricker, J. Proc. Cont. 6(4), 1996, as continuous PI with anti-windup), and the MVs
are what Ricker's outer loops moved: production index, D/E shift, A and C feed ratios; further setpoints
and the three free valves are MVs when asked for.
"""
import math

import torch

from aio.sim.engine import Plant, Mech
from aio.sim.expr import exp, sqrt, relu, step, maximum, clamp, where

NC = 8
XMW = (2., 25.4, 28., 32., 46., 48., 62., 76.)
AVP = (0., 0., 0., 15.92, 16.35, 16.35, 16.43, 17.21)
BVP = (0., 0., 0., -1444., -2114., -2114., -2748., -3318.)
CVP = (0., 0., 0., 259., 265.5, 265.5, 232.9, 249.6)
AD = (1., 1., 1., 23.3, 33.9, 32.8, 49.9, 50.5)
BD = (0., 0., 0., -.0700, -.0957, -.0995, -.0191, -.0541)
CD = (0., 0., 0., -.0002, -.000152, -.000233, -.000425, -.000150)
AH = (1e-6, 1e-6, 1e-6, .960e-6, .573e-6, .652e-6, .515e-6, .471e-6)
BH = (0., 0., 0., 8.70e-9, 2.41e-9, 2.18e-9, 5.65e-10, 8.70e-10)
CH = (0., 0., 0., 4.81e-11, 1.82e-11, 1.94e-11, 3.82e-12, 2.62e-12)
AV = (1e-6, 1e-6, 1e-6, 86.7e-6, 160e-6, 160e-6, 225e-6, 209e-6)
AG = (3.411e-6, .3799e-6, .2491e-6, .3567e-6, .3463e-6, .3930e-6, .170e-6, .150e-6)
BG = (7.18e-10, 1.08e-9, 1.36e-11, 8.51e-10, 8.96e-10, 1.02e-9, 0., 0.)
CG = (6.0e-13, -3.98e-13, -3.93e-14, -3.12e-13, -3.27e-13, -3.12e-13, 0., 0.)
RG, RU = 998.9, 3.57696e-6                          # gas constant for pressures (mmHg ft3 / lb-mol K); for the internal energy of a gas
VTR, VTS, VTC, VTV = 1300., 3500., 156.5, 5000.     # vessel volumes, ft3
HTR = (.06899381054, .05)                           # heats of reaction 1 and 2
HWR, HWS = 7060., 11138.                            # cooling-water heat capacities
CPFLMX, CPPRMX = 280275., 1.3                       # compressor
VRNG = (400., 400., 100., 1500., None, None, 1500., 1000., .03, 1000., 1200., None)
VTAU = tuple(t/3600. for t in (8., 8., 6., 9., 7., 5., 5., 5., 120., 5., 5., 5.))
SFR0 = (.995, .991, .99, .916, .936, .938, .058, .0301)
KSTR = (None, None, None, 8.5010, 11.402, 11.795, .0480, .0242)
FEED = ((0., .0001, 0., .9999, 0., 0., 0., 0.), (0., 0., 0., 0., .9999, .0001, 0., 0.), (.9999, .0001, 0., 0., 0., 0., 0., 0.))   # streams 1 (D), 2 (E), 3 (A)
# the routine's initial state (YY of TEINIT): holdups, internal energies, cooling-water temperatures, valves
YY = (10.40491389, 4.363996017, 7.570059737, .4230042431, 24.15513437, 2.942597645, 154.3770655, 159.1865960, 2.808522723,
      63.75581199, 26.74026066, 46.38532432, .2464521543, 15.20484404, 1.852266172, 52.44639459, 41.20394008, .5699317760,
      .4306056376, 7.9906200783e-3, .9056036089, 1.6054258216e-2, .7509759687, 8.8582855955e-2, 48.27726193, 39.38459028, .3755297257,
      107.7562698, 29.77250546, 88.32481135, 23.03929507, 62.85848794, 5.546318688, 11.92244772, 5.555448243, .9218489762,
      94.59927549, 77.29698353,
      63.05263039, 53.97970677, 24.64355755, 61.30192144, 22.21, 40.06374673, 38.10034370, 46.53415582, 47.44573456, 41.10581288, 18.11349055, 50.)
COMP = 'ABCDEFGH'
# operating cost of Downs & Vogel: $/kg-mol of each component lost, $/kWh of compressor work, $/kg of steam
COST = (2.206, 0., 6.177, 22.06, 14.56, 17.89, 30.44, 22.94)
COST_KWH, COST_STEAM = .0536, .0318
MEAS = ('A feed', 'D feed', 'E feed', 'AC feed', 'recycle', 'reactor feed', 'reactor P', 'reactor level', 'reactor T', 'purge', 'sep T', 'sep level',
        'sep P', 'sep underflow', 'stripper level', 'stripper P', 'product flow', 'stripper T', 'steam', 'compressor work', 'reactor CW T', 'condenser CW T')


def _poly(c1, c2, c3, T):
    """c1 T + c2 T^2 + c3 T^3 with the powers shared."""
    T2 = T*T
    return c1*T+c2*T2+c3*(T2*T)


def _hl(i, T):
    """Liquid molar enthalpy of component i (TESUB1, ITY 0) and its slope (TESUB3)."""
    return _poly(1.8*XMW[i]*AH[i], .9*XMW[i]*BH[i], .6*XMW[i]*CH[i], T)


def _cl(i, T):
    return 1.8*XMW[i]*(AH[i]+BH[i]*T+CH[i]*(T*T))


def _hg(i, T):
    """Vapour molar enthalpy (ITY 1): the liquid form with the gas constants plus the heat of vaporisation."""
    return _poly(1.8*XMW[i]*AG[i], .9*XMW[i]*BG[i], .6*XMW[i]*CG[i], T)+XMW[i]*AV[i]


def _cg(i, T):
    return 1.8*XMW[i]*(AG[i]+BG[i]*T+CG[i]*(T*T))


def _dens(i, T):
    """Liquid molar density term of TESUB4: XMW / this is the molar volume."""
    return AD[i]+(BD[i]+CD[i]*T)*T


def _newton(fun, slope, T):
    for _ in range(60):
        T = T-fun(T)/slope(T)
    return T


def initial():
    """The routine's initial state in this plant's states: the four temperatures that carry the energies of YY."""
    liq = lambda n, idx, E: _newton(lambda T: sum(n[i]*_hl(i, T) for i in idx)-E, lambda T: sum(n[i]*_cl(i, T) for i in idx), 100.)   # noqa: E731
    nR, nS, nC, nV = YY[0:8], YY[9:17], YY[18:26], YY[27:35]
    heavy = range(3, 8)
    TR, TS, TC = liq(nR, heavy, YY[8]), liq(nS, heavy, YY[17]), liq(nC, range(8), YY[26])
    TV = _newton(lambda T: sum(nV[i]*_hg(i, T) for i in range(8))-RU*(T+273.15)*sum(nV)-YY[35], lambda T: sum(nV[i]*_cg(i, T) for i in range(8))-RU*sum(nV), 100.)
    return nR, TR, nS, TS, nC, TC, nV, TV


def from_routine(y):
    """The routine's 50 states (rows of y) as this plant's: the same, with the four temperatures in place of the internal energies."""
    y = torch.as_tensor(y, dtype=torch.float64).clone()
    start = torch.full_like(y[..., 0], 100.)
    liq = lambda n, idx, E: _newton(lambda T: sum(n[..., i]*_hl(i, T) for i in idx)-E, lambda T: sum(n[..., i]*_cl(i, T) for i in idx), start)   # noqa: E731
    nV, heavy = y[..., 27:35], range(3, 8)
    TV = _newton(lambda T: sum(nV[..., i]*_hg(i, T) for i in range(8))-RU*(T+273.15)*nV.sum(-1)-y[..., 35],
                 lambda T: sum(nV[..., i]*_cg(i, T) for i in range(8))-RU*nV.sum(-1), start)
    TR, TS, TC = liq(y[..., 0:8], heavy, y[..., 8]), liq(y[..., 9:17], heavy, y[..., 17]), liq(y[..., 18:26], range(8), y[..., 26])
    y[..., 8], y[..., 17], y[..., 26], y[..., 35] = TR, TS, TC, TV
    return y


def _sum(v):
    v = list(v)
    out = v[0]
    for a in v[1:]:
        out = out+a
    return out


def _process(P, L):
    """Declare the fifty process states and the equations of the first thirty-eight. L: loads and parameters
    by name. Returns the quantities the outputs and the layer are made of; the valve positions `vp` get
    their commands from the caller (`_valves`), which may compute them from the measurements."""
    nR0, TR0, nS0, TS0, nC0, TC0, nV0, TV0 = initial()
    nR = [P.state(f'nR{c}', v, lo=0.) for c, v in zip(COMP, nR0)]
    TR = P.state('TR', TR0, scale=10.)
    nS = [P.state(f'nS{c}', v, lo=0.) for c, v in zip(COMP, nS0)]
    TS = P.state('TS', TS0, scale=10.)
    nC = [P.state(f'nC{c}', v, lo=0.) for c, v in zip(COMP, nC0)]
    TC = P.state('TC', TC0, scale=10.)
    nV = [P.state(f'nV{c}', v, lo=0.) for c, v in zip(COMP, nV0)]
    TV = P.state('TV', TV0, scale=10.)
    TWR, TWS = P.state('TWR', YY[36], scale=10.), P.state('TWS', YY[37], scale=10.)
    vp = [P.state(f'v{i+1}', YY[38+i], lo=0., hi=100., scale=10.) for i in range(12)]
    heavy = range(3, 8)

    # ---- holdups to compositions, volumes, pressures
    utlr, utls, utlc, utvv = _sum(nR[i] for i in heavy).held, _sum(nS[i] for i in heavy).held, _sum(nC).held, _sum(nV).held
    xlr = [None]*3+[nR[i]/utlr for i in heavy]
    xls = [None]*3+[nS[i]/utls for i in heavy]
    xlc = [n/utlc for n in nC]
    xvv = [n/utvv for n in nV]
    vlr = _sum(nR[i]*(XMW[i]/_dens(i, TR)) for i in heavy).held
    vls = _sum(nS[i]*(XMW[i]/_dens(i, TS)) for i in heavy).held
    vlc = _sum(nC[i]*(XMW[i]/_dens(i, TC)) for i in range(8)).held
    vvr, vvs = VTR-vlr, VTS-vls
    tkr, tks, tkv = TR+273.15, TS+273.15, TV+273.15
    ppr = [nR[i]*RG*tkr/vvr for i in range(3)]+[exp(AVP[i]+BVP[i]/(TR+CVP[i]))*xlr[i] for i in heavy]
    pps = [nS[i]*RG*tks/vvs for i in range(3)]+[exp(AVP[i]+BVP[i]/(TS+CVP[i]))*xls[i] for i in heavy]
    ptr, pts = _sum(ppr).held, _sum(pps).held
    ptv = utvv*RG*tkv/VTV
    xvr, xvs = [q/ptr for q in ppr], [q/pts for q in pps]

    # ---- reactions (lb-mol/h): A + C + D -> G, A + C + E -> H, A + E -> F, 3D -> 2F
    pa, pc = maximum(ppr[0], 1e-30), maximum(ppr[2], 1e-30)
    drive = exp(1.1544*_ln(pa)+.3735*_ln(pc))
    rr1 = exp(31.5859536-40000./1.987/tkr)*L['r1f']*drive*ppr[3]*vvr
    rr2 = exp(3.00094014-20000./1.987/tkr)*L['r2f']*drive*ppr[4]*vvr
    k3 = exp(53.4060443-60000./1.987/tkr)
    rr3 = k3*ppr[0]*ppr[4]*vvr
    rr4 = k3*.767488334*ppr[0]*ppr[3]*vvr
    crx = (-rr1-rr2-rr3, 0., -rr1-rr2, -rr1-1.5*rr4, -rr2-rr3, rr3+rr4, rr1, rr2)
    rh = rr1*HTR[0]+rr2*HTR[1]

    # ---- flows (lb-mol/h)
    xmws6, xmws8, xmws9 = (_sum(z[i]*XMW[i] for i in range(8)).held for z in (xvv, xvr, xvs))
    f1, f2 = vp[0]*VRNG[0]/100., vp[1]*VRNG[1]/100.
    f3 = vp[2]*L['a_avail']*VRNG[2]/100.
    f4 = vp[3]*L['c_avail']*VRNG[3]/100.+1e-10
    f11, f13 = vp[6]*VRNG[6]/100., vp[7]*VRNG[7]/100.
    uac = vp[8]*VRNG[8]*L['uac']/100.
    fwr, fws = vp[9]*VRNG[9]/100., vp[10]*VRNG[10]/100.
    agsp = (vp[11]+150.)/100.
    f6 = 1937.6*sqrt(relu(ptv-ptr))/xmws6
    f8 = 4574.21*sqrt(relu(ptr-pts))*L['cv8']/xmws8
    f10 = vp[5]*.151169*sqrt(relu(pts-760.))/xmws9
    pr = clamp(ptv/pts, 1., CPPRMX)
    flms = CPFLMX+CPFLMX/1.197*(1.-pr*pr*pr)
    cpdh = flms*tks*1.8e-6*1.9872*(ptv-pts)/(xmws9*pts)
    f9 = maximum(flms-vp[4]*53.349*sqrt(relu(ptv-pts)), 1e-3)/xmws9
    x4 = (L['xA4'], L['xB4'], 1.-L['xA4']-L['xB4'], 0., 0., 0., 0., 0.)
    t1, t4 = L['tD'], L['tC']

    # ---- stripper split (streams 4 and 11 in; 5 overhead, 12 down)
    tmp = where(TC-170., TC-120.262, where(5.292-TC, .1, 363.744/(177.-clamp(TC, 5.292, 170.))-2.22579488))
    vov = f4/maximum(f11, .1)*tmp
    on = step(f11-.1)
    sfr = list(SFR0[:3])+[SFR_OFF[i]+on*(KSTR[i]*vov/(1.+KSTR[i]*vov)-SFR_OFF[i]) for i in heavy]
    fin = [x4[i]*f4+(xls[i]*f11 if i >= 3 else 0.) for i in range(8)]
    fc5 = [sfr[i]*fin[i] for i in range(8)]
    fc12 = [fin[i]-fc5[i] for i in range(8)]

    # ---- heat transfer
    uar = clamp(.025*vlr/7.8-.25, 0., 1.)*(-.5*agsp*agsp+2.75*agsp-2.5)*855490e-6
    qur = uar*(TWR-TR)*L['uar']
    r8 = f8/3528.73
    r84 = (r8*r8)*(r8*r8)
    uas = .404655*(1.-1./(1.+r84))
    qus = uas*(TWS-TR)*L['uas']
    quc = uac*relu(100.-TC)

    # ---- component balances
    fc6 = [xvv[i]*f6 for i in range(8)]
    fc8 = [xvr[i]*f8 for i in range(8)]
    out_s = f9+f10
    dR = [fc6[i]-fc8[i]+crx[i] for i in range(8)]
    dS = [fc8[i]-xvs[i]*out_s-(xls[i]*f11 if i >= 3 else 0.) for i in range(8)]
    dC = [fc12[i]-xlc[i]*f13 for i in range(8)]
    dV = [FEED[0][i]*f1+FEED[1][i]*f2+FEED[2][i]*f3+fc5[i]+xvs[i]*f9-fc6[i] for i in range(8)]

    # ---- energy balances (stream enthalpy times flow, written per component flow)
    hgR, hgS, hgC, hgV = ([_hg(i, T) for i in range(8)] for T in (TR, TS, TC, TV))
    hlR, hlS, hlC = ([_hl(i, T) for i in range(8)] for T in (TR, TS, TC))
    H6 = _sum(fc6[i]*hgV[i] for i in range(8)).held
    H8 = _sum(fc8[i]*hgR[i] for i in range(8)).held
    hs = _sum(xvs[i]*hgS[i] for i in range(8)).held
    H9, H10 = f9*hs+cpdh, f10*hs
    H11 = f11*_sum(xls[i]*hlS[i] for i in heavy)
    H13 = f13*_sum(xlc[i]*hlC[i] for i in range(8))
    H5 = _sum(fc5[i]*hgC[i] for i in range(8)).held
    H4 = f4*_sum(x4[i]*_hg(i, t4) for i in range(3))
    H1 = f1*_sum(FEED[0][i]*_hg(i, t1) for i in range(8) if FEED[0][i])
    H2 = f2*sum(FEED[1][i]*_hg(i, 45.) for i in range(8))
    H3 = f3*sum(FEED[2][i]*_hg(i, 45.) for i in range(8))
    dER = H6-H8+rh+qur
    dES = H8-H9-H10-H11+qus
    dEC = H4+H11-H5-H13+quc
    dEV = H1+H2+H3+H5+H9-H6
    for i in range(8):
        P.der(nR[i], dR[i]); P.der(nS[i], dS[i]); P.der(nC[i], dC[i]); P.der(nV[i], dV[i])
    P.der(TR, (dER-_sum(dR[i]*hlR[i] for i in heavy))/_sum(nR[i]*_cl(i, TR) for i in heavy))
    P.der(TS, (dES-_sum(dS[i]*hlS[i] for i in heavy))/_sum(nS[i]*_cl(i, TS) for i in heavy))
    P.der(TC, (dEC-_sum(dC[i]*hlC[i] for i in range(8)))/_sum(nC[i]*_cl(i, TC) for i in range(8)))
    P.der(TV, (dEV-_sum(dV[i]*(hgV[i]-RU*tkv) for i in range(8)))/_sum(nV[i]*(_cg(i, TV)-RU) for i in range(8)))
    P.der(TWR, (fwr*500.53*(L['tcwr']-TWR)-qur*1e6/1.8)/HWR)
    P.der(TWS, (fws*500.53*(L['tcws']-TWS)-qus*1e6/1.8)/HWS)

    # ---- the measurements of the routine (XMEAS 1-41), true values
    xmws1, xmws2 = sum(FEED[0][i]*XMW[i] for i in range(8)), sum(FEED[1][i]*XMW[i] for i in range(8))
    scm = .359/35.3145
    xm = [f3*scm, f1*xmws1*.454, f2*xmws2*.454, f4*scm, f9*scm, f6*scm, (ptr-760.)/760.*101.325, (vlr-84.6)/666.7*100., TR, f10*scm, TS,
          (vls-27.5)/290.*100., (pts-760.)/760.*101.325, f11*vls/utls/35.3145, (vlc-78.25)/VTC*100., (ptv-760.)/760.*101.325, f13*vlc/utlc/35.3145, TC,
          quc*1.04e3*.454, cpdh*.29307e3, TWR, TWS]
    xm += [100.*xvv[i] for i in range(6)]+[100.*xvs[i] for i in range(8)]+[100.*xlc[i] for i in heavy]
    purge, prod = f10*.454, f13*.454                                 # kg-mol/h
    cost = purge*_sum(COST[i]*xvs[i] for i in range(8) if COST[i])+prod*_sum(COST[i]*xlc[i] for i in (3, 4, 5))+COST_KWH*xm[19]+COST_STEAM*xm[18]
    energy = (_sum(nR[i]*hlR[i] for i in heavy), _sum(nS[i]*hlS[i] for i in heavy), _sum(nC[i]*hlC[i] for i in range(8)),
              _sum(nV[i]*(hgV[i]-RU*tkv) for i in range(8)))
    return dict(xm=xm, vp=vp, cost=cost, energy=energy, vlr=vlr, vls=vls, vlc=vlc)


SFR_OFF = (None, None, None, .9999, .999, .999, .99, .98)            # stripping factors with no liquid feed to the stripper


def _ln(a):
    """log of an expression that is already positive."""
    return type(a)(a.g, a.g.un('log', a.i))


def _inputs(P, ranges=True):
    """The loads and parameters every variant has, at the routine's values."""
    r = (lambda lo, hi: (lo, hi)) if ranges else (lambda lo, hi: (None, None))
    L = dict(xA4=P.load('xA4', .485, .455, .515), xB4=P.load('xB4', .005, .002, .010),
             tD=P.load('tD', 45., 35., 55.), tC=P.load('tC', 45., 35., 55.),
             tcwr=P.load('tcwr', 35., 28., 42.), tcws=P.load('tcws', 40., 33., 47.),
             a_avail=P.load('a_avail', 1., 0., 1.), c_avail=P.load('c_avail', 1., .8, 1.))
    for q, (lo, hi) in dict(r1f=(.9, 1.1), r2f=(.9, 1.1), uar=(.85, 1.05), uas=(.85, 1.05), uac=(.85, 1.05), cv8=(.9, 1.05)).items():
        L[q] = P.param(q, 1., *r(lo, hi))
    return L


def _valves(P, vp, cmd):
    """First-order valves: the positions follow the commands (0-100)."""
    for i in range(12):
        c = clamp(cmd[i], 0., 100.) if hasattr(cmd[i], 'g') else min(max(float(cmd[i]), 0.), 100.)
        P.der(vp[i], (c-vp[i])/VTAU[i])


def tep_raw(dt=60., substeps=4):
    """The open-loop process: the twelve valves are the MVs, the outputs are XMEAS 1-41, the operating cost
    and the four internal energies (the routine's states 9, 18, 27, 36). Unstable; for comparison with the
    native code and for the tools on the table, not for episodes."""
    P = Plant('tep_raw', dt=dt, time_unit=3600., substeps=substeps)
    xmv = [P.mv(f'xmv{i+1}', 0., 100., 2., YY[38+i]) for i in range(12)]
    q = _process(P, _inputs(P))
    _valves(P, q['vp'], xmv)
    for i, v in enumerate(q['xm']):
        P.out(f'xmeas{i+1}', v, range=(0., 1.), band=1.)
    P.out('cost', q['cost'], range=(0., 400.), band=1.)
    for nm, e in zip(('ETR', 'ETS', 'ETC', 'ETV'), q['energy']):
        P.out(nm, e, range=(0., 4.), band=1.)
    return P


_BASE = {}


def base():
    """XMEAS 1-41, the cost and the energies at the routine's initial state (float64), from the raw plant's table."""
    if not _BASE:
        P = tep_raw()
        M = Mech([P], n=1, dtype=torch.float64)
        x, w, p = P.nominal()
        y = M.y(x, w, p)[0]
        _BASE.update(xm=[float(v) for v in y[:41]], cost=float(y[41]), energy=[float(v) for v in y[42:46]])
    return _BASE


def _pi(P, name, e, K, Ti, bias, lo, hi, scale):
    """A PI controller of the regulatory layer: out = bias + K (e + integral of e / Ti), limited to lo..hi; the
    integral is held while the output is at a limit and the error pushes further (as in plants.excstr)."""
    I = P.state(name, 0., scale=scale)
    raw = bias+K*e+(K/Ti)*I
    P.der(I, e*(1.-step(raw-hi)*step(K*e)-step(lo-raw)*step(-1.*K*e)))
    return clamp(raw, lo, hi)


# what an APC may move: name -> (low, high, largest move per decision, nominal; None: the base value)
APC = dict(Fp=(70., 108., 1., 100.), Eadj=(-15., 15., .5, 0.), rA=(.05, .60, .01, None), rC=(8.2, 10.3, .03, None),
           Psp=(2600., 2850., 5., None), Tsp=(116., 126., .2, None), Lsp=(60., 90., .5, None),
           recycle=(0., 60., 2., YY[42]), steam=(0., 100., 2., YY[46]), agitator=(0., 100., 5., YY[49]))
MVS4 = ('Fp', 'Eadj', 'rA', 'rC')
MVS9 = MVS4+('Psp', 'Tsp', 'Lsp', 'recycle', 'steam')
MIN = 1/60.                                         # a minute, in hours
# the shutdown interlocks of Downs & Vogel (1993, table 3): reactor pressure above 3,000 kPa, reactor temperature above
# 175 C, liquid volumes out of 2-24 m3 (reactor), 1-12 m3 (separator), 1-8 m3 (stripper base), as the level outputs
# (per cent, `_process`: (V ft3 - offset)/span) read them; reported by a rollout, not scored (control.rollout)
_pct = lambda m3, off, span: (m3*35.3145-off)/span*100.             # noqa: E731
TRIP = dict(P=(-math.inf, 3000.), T=(-math.inf, 175.), L=(_pct(2., 84.6, 666.7), _pct(24., 84.6, 666.7)),
            Ls=(_pct(1., 27.5, 290.), _pct(12., 27.5, 290.)), Lc=(_pct(1., 78.25, VTC), _pct(8., 78.25, VTC)))
# the regulatory loops: gain (output per unit of error) and integral time (h)
TUNE = dict(fc=(.5, 10/3600.), tr=(-8., 7.5*MIN), ts=(-4., 15*MIN), lr=(.8, 60*MIN), ls=(-.4, 120*MIN), lc=(-.3, 120*MIN), pr=(-1., 20*MIN))


def tep(mvs=MVS4, name='tep'):
    """Tennessee Eastman under its regulatory layer. mvs: which of `APC` are MVs (the others stay at their
    nominal values). Time in h; a decision is twenty minutes: the slowest pair (A feed ratio to A in the reactor feed) takes 41 decisions to 63 %.

    The layer (continuous PI, anti-windup): flow controllers on the four feeds; reactor temperature on the
    reactor cooling water; separator temperature on the condenser cooling water, its setpoint from the
    reactor level; separator and stripper levels on their outflows; reactor pressure on the purge valve.
    Feed setpoints: D = base Fp/100 - 32 Eadj, E = base Fp/100 + 46 Eadj (kg/h; Eadj kg-mol/h moved from D
    to E), A = rA Fp/100, A+C = rC Fp/100 (kscmh)."""
    b = base()['xm']
    P = Plant(name, dt=1200., time_unit=3600., substeps=4, refresh=True)
    nom = dict(rA=b[0], rC=b[3], Psp=b[6], Tsp=b[8], Lsp=b[7])
    u = {}
    for q, (lo, hi, slew, n0) in APC.items():
        n0 = nom[q] if n0 is None else n0
        u[q] = P.mv(q, lo, hi, slew, n0) if q in mvs else n0
    q = _process(P, _inputs(P))
    xm, vp = q['xm'], q['vp']
    fp = u['Fp']/100.
    sp = (b[1]*fp-32.*u['Eadj'], b[2]*fp+46.*u['Eadj'], u['rA']*fp, u['rC']*fp)          # D, E (kg/h), A, A+C (kscmh)
    cmd = [None]*12
    kf, tf = TUNE['fc']
    for j, (valve, meas) in enumerate(((0, 1), (1, 2), (2, 0), (3, 3))):
        gain = b[meas]/YY[38+valve]                                                      # flow per % of valve at the base
        cmd[valve] = _pi(P, f'I_f{valve+1}', sp[j]-xm[meas], kf/gain, tf, YY[38+valve], 0., 100., gain*tf)
    cmd[9] = _pi(P, 'I_tr', u['Tsp']-xm[8], *TUNE['tr'], YY[47], 0., 100., 1.)
    tsp = _pi(P, 'I_lr', u['Lsp']-xm[7], *TUNE['lr'], b[10], 0., 120., 10.)
    cmd[10] = _pi(P, 'I_ts', tsp-xm[10], *TUNE['ts'], YY[48], 0., 100., 1.)
    cmd[6] = _pi(P, 'I_ls', 50.-xm[11], *TUNE['ls'], YY[44], 0., 100., 10.)
    cmd[7] = _pi(P, 'I_lc', 50.-xm[14], *TUNE['lc'], YY[45], 0., 100., 10.)
    cmd[5] = _pi(P, 'I_pr', u['Psp']-xm[6], *TUNE['pr'], YY[43], 0., 100., 10.)
    cmd[4], cmd[8], cmd[11] = u['recycle'], u['steam'], u['agitator']
    _valves(P, vp, cmd)

    ya = 100.*xm[22]/(xm[22]+xm[24])
    yac = xm[22]+xm[24]
    mv = lambda k: k in mvs                                                              # noqa: E731
    P.out('prod', xm[16], range=(14., 27.), band=.25, track=True, target=(17.5, 24.5))
    P.out('G', xm[39], range=(35., 70.), band=.5, track=True, target=(46., 61.))
    P.out('yA', ya, range=(45., 65.), band=.5, track=True, target=(51., 59.))
    P.out('yAC', yac, range=(50., 67.), band=.5, track=True, target=(55., 62.))
    P.out('P', xm[6], range=(2500., 2950.), band=5., track=mv('Psp'), target=(2620., 2830.), hi=2895., trip=TRIP['P'])
    P.out('T', xm[8], range=(110., 135.), band=.2, track=mv('Tsp'), target=(117., 125.), hi=150., trip=TRIP['T'])
    P.out('L', xm[7], range=(40., 110.), band=1., track=mv('Lsp'), target=(62., 88.), lo=50., hi=100., trip=TRIP['L'])
    P.out('Frec', xm[4], range=(15., 35.), band=.3, track=mv('recycle'), target=(22., 31.))
    P.out('Tstr', xm[17], range=(40., 90.), band=.5, track=mv('steam'), target=(55., 75.))
    P.out('Ls', xm[11], range=(0., 100.), band=2., lo=30., hi=100., trip=TRIP['Ls'])
    P.out('Lc', xm[14], range=(0., 100.), band=2., lo=30., hi=100., trip=TRIP['Lc'])
    for nm, i in (('vA', 2), ('vC', 3), ('vPurge', 5), ('vCWR', 9), ('vCWS', 10)):
        P.out(nm, vp[i], range=(0., 100.), band=2., lo=5., hi=95.)
    P.out('cost', q['cost'], range=(60., 320.), band=5.)                                   # $/h: what two points of the purge valve cost
    return P


def tep9():
    """The nine-MV plant (for when the shell takes more than five MVs)."""
    return tep(MVS9, name='tep9')

"""The held-out plants as data for the mechanism engine (docs/heldout.md, docs/sim.md 'TEP and BSM1').

The four families no run trains on, written once as tables: equation by equation the plants of the
previous generation's held-out environments (compared with what those gave in tests/test_plants_heldout.py), at
an APC decision interval.
They are in `HELDOUT`, not in `plants.PLANTS`: `mech.pool` gives them only when asked with heldout=True.
"""
import math

from aio.sim.engine import Plant
from aio.sim.expr import E, exp, log, sqrt, relu, maximum


def boiler():
    """Drum boiler-turbine unit (Bell & Astrom 1987, the 160 MW unit; operating point 4): drum pressure, electric
    output and drum level by fuel, steam valve and feedwater. The level integrates and first rises when the
    steam valve opens (swell). Time in s."""
    P = Plant('boiler', dt=10., substeps=4)
    u1, u2, u3 = P.mv('fuel', 0., 1., .07, .34), P.mv('steam', 0., 1., .05, .69), P.mv('feedwater', 0., 1., .1, .436)    # fuel: 0.007/s as published
    fq, fw = P.load('fq', 1., .9, 1.1), P.load('fw', 1., .8, 1.2)                     # fuel heating value, feedwater subcooling
    eta, kv = P.param('eta', 1., .93, 1.03), P.param('kv', 1., .92, 1.05)             # turbine efficiency, steam valve coefficient
    p, E, rho = P.state('P', 108., lo=1.), P.state('E', 66.65), P.state('rho', 428., lo=50.)
    v = kv*u2
    q = p**1.125
    P.der(p, -.0018*v*q+.9*fq*u1-.15*fw*u3)
    P.der(E, eta*(.073*v-.016)*q-.1*E)
    P.der(rho, (141.*u3-(1.1*v-.19)*p)/85.)
    acs = (1.-.001538*rho)*(.8*p-25.6)/(rho*(1.0394-.0012304*p))
    qe = (.854*v-.147)*p+45.59*fq*u1-2.514*fw*u3-2.096
    P.out('P', p, range=(70., 150.), band=1., track=True, target=(95., 125.))
    P.out('E', E, range=(10., 130.), band=1., track=True, target=(45., 95.))
    P.out('L', .05*(.13073*rho+100.*acs+qe/9.-67.975), range=(-.2, .2), band=.02, track=True, target=(-.05, .05), lo=-.1, hi=.1, integ=True)
    P.out('qs', (1.1*v-.19)*p, range=(20., 160.), band=1.)
    return P


def wo():
    """Williams-Otto reactor with separation and recycle (Williams & Otto 1960; the separation and the recycle
    drum are the older environment's simplification): P production, by-product fraction and recycle flow by
    feed B, reactor temperature and purge share. The recycle grows several fold as the purge closes. Time in s."""
    W = 2105.
    P = Plant('wo', dt=120., substeps=2)
    FB, TR, purge = P.mv('FB', 1.5, 6., .2, 3.), P.mv('TR', 70., 100., 1.5, 85.), P.mv('eta', .1, .6, .025, .3)
    FA = P.load('FA', 1.8275, 1.65, 2.)
    a, s3, tauR = P.param('a', 1., .85, 1.1), P.param('s3', 1., .8, 1.25), P.param('tauR', 600., 400., 900.)
    x0 = (.17048, .23542, .03266, .46509, .03987, .05647, 1.56789, 2.16508, .30038, 4.2773)
    XA, XB, XC, XE, XG, XP, rA, rB, rC, rE = [P.state(q, v, lo=0.) for q, v in zip(('XA', 'XB', 'XC', 'XE', 'XG', 'XP', 'rA', 'rB', 'rC', 'rE'), x0)]
    T = TR+273.15
    k1, k2, k3 = a*1.6599e6*exp(-6666.7/T), a*7.2117e8*exp(-8333.3/T), a*s3*2.6745e12*exp(-11111./T)
    r1, r2, r3 = k1*XA*XB, k2*XB*XC, k3*XC*XP
    R = rA+rB+rC+rE
    F = FA+FB+R
    P.der(XA, (FA+rA-F*XA)/W-r1)
    P.der(XB, (FB+rB-F*XB)/W-r1-r2)
    P.der(XC, (rC-F*XC)/W+2*r1-2*r2-r3)
    P.der(XE, (rE-F*XE)/W+2*r2)
    P.der(XG, -F*XG/W+1.5*r3)
    P.der(XP, -F*XP/W+r2-.5*r3)
    for X, r in ((XA, rA), (XB, rB), (XC, rC), (XE, rE)):
        P.der(r, ((1-purge)*F*X-r)/tauR)
    P.out('P', F*XP, range=(.4, 1.1), band=.01, track=True, target=(.6, .9))
    P.out('XG', XG, range=(0., .12), band=.002, track=True, target=(.025, .06))
    P.out('R', R, range=(2., 16.), band=.2, track=True, target=(6., 11.))
    P.out('F', F, range=(6., 16.), band=.2, hi=16.)
    P.out('XE', XE, range=(.3, .6), band=.01)
    return P


def mma():
    """MMA solution polymerisation CSTR (Maner, Doyle, Ogunnaike & Pearson 1996, with the older environment's
    Arrhenius terms and energy balance): number-average molecular weight and reactor temperature by initiator
    flow and jacket temperature. Time in h."""
    V, CMIN, MM, RG, RHOCP, DH, UA = .1, 6., 100.12, 8.314, 1800., 57800., 1500.
    K335 = dict(kI=(.10225, 1.2541e5), kp=(2.4952e6, 1.8283e4), kfm=(2.4522e3, 7.4478e4), ktc=(1.3281e10, 2.9442e3), ktd=(1.0930e11, 2.9442e3))
    P = Plant('mma', dt=60., time_unit=3600.)
    FI, Tj = P.mv('FI', .004, .05, .002, .016783), P.mv('Tj', 300., 345., 1., 328.)
    F = P.load('F', 1., .85, 1.15, measured=True)
    CIin, Tin = P.load('CIin', 8., 7.2, 8.8), P.load('Tin', 325., 320., 329.)
    fe, ua = P.param('fe', .58, .5, .62), P.param('ua', 1., .75, 1.05)
    Cm, CI = P.state('Cm', 5.5068, lo=1e-9), P.state('CI', .1329, lo=1e-9)
    D0, D1, T = P.state('D0', .0019752, lo=1e-9), P.state('D1', 49.382, lo=1e-9), P.state('T', 335.)
    kI, kp, kfm, ktc, ktd = (k0*exp(-E/RG*(1./T-1./335.)) for k0, E in (K335[q] for q in ('kI', 'kp', 'kfm', 'ktc', 'ktd')))
    P0 = sqrt(2.*fe*relu(CI)*kI/(ktd+ktc), 1e-30)                                    # about 1e-13 at the nominal point: below sqrt's default floor
    P.der(Cm, -(kp+kfm)*Cm*P0+F*(CMIN-Cm)/V)
    P.der(CI, -kI*CI+(FI*CIin-F*CI)/V)
    P.der(D0, (.5*ktc+ktd)*P0*P0+kfm*Cm*P0-F*D0/V)
    P.der(D1, MM*(kp+kfm)*Cm*P0-F*D1/V)
    P.der(T, (F*RHOCP*(Tin-T)+DH*kp*Cm*P0*V-ua*UA*(T-Tj))/(V*RHOCP))
    P.out('NAMW', D1/maximum(D0, 1e-9), range=(15000., 40000.), band=500., track=True, target=(20000., 31000.))
    P.out('T', T, range=(320., 350.), band=.5, track=True, target=(330., 340.))
    P.out('X', 1.-Cm/CMIN, range=(0., .13), band=.005, hi=.13)
    P.out('Pr', F*D1, range=(30., 80.), band=1.)
    P.out('F', F, range=(.7, 1.3), band=.01, dv=True)
    return P


def cryst():
    """Continuous MSMPR cooling crystalliser of potassium sulphate, moments of the population balance (Randolph
    & Larson; kinetic constants are the older environment's choice): mean size L43 and production by fines
    destruction and crystalliser temperature. Time in h."""
    RHOC_KV, VOL = 2660.*.5, 2.
    P = Plant('cryst', dt=600., time_unit=3600., substeps=2)
    Tc, phi = P.mv('T', 10., 45., 1., 30.), P.mv('phi', 0., .9, .03, .5)
    Q, Cin = P.load('Q', 2., 1.6, 2.4, measured=True), P.load('Cin', 164., 156., 170.)
    kg, kb = P.param('kg', 72., 60., 85.), P.param('kb', 3150., 2400., 4000.)
    x0 = (132.16, 53.02, 22.38, 18.9, 23.94, 40.43)
    C, m0, m1, m2, m3, m4 = [P.state(q, v, lo=0.) for q, v in zip(('C', 'm0', 'm1', 'm2', 'm3', 'm4'), x0)]
    cs = 74.+1.8*Tc
    S = (C-cs)/cs
    Sp = relu(S)
    MT = RHOC_KV*m3*1e-3
    G, B = kg*Sp*sqrt(Sp), kb*Sp*Sp*relu(MT)
    D = Q/VOL
    P.der(C, (Cin-C)*D-3.*RHOC_KV*G*m2*1e-3)
    P.der(m0, (1.-phi)*B-m0*D)
    P.der(m1, G*m0-m1*D)
    P.der(m2, 2.*G*m1-m2*D)
    P.der(m3, 3.*G*m2-m3*D)
    P.der(m4, 4.*G*m3-m4*D)
    P.out('L43', 100.*m4/maximum(m3, 1e-6), range=(100., 260.), band=3., track=True, target=(150., 200.))
    P.out('prod', Q*MT, range=(20., 140.), band=2., track=True, target=(45., 85.))
    P.out('S', S, range=(0., .06), band=.003, hi=.06)
    P.out('MT', MT, range=(10., 60.), band=1.)
    P.out('Q', Q, range=(1.4, 2.6), band=.02, dv=True)
    return P




# ---------------------------------------------------------------- drum boiler of Astrom and Bell (2000)
# Saturated water and steam (IAPWS): the temperature from the pressure by IAPWS-IF97's region 4 equation; the
# densities, the enthalpies and the vapour pressure's slope from the IAPWS supplementary release on saturation
# properties (Wagner and Pruss 1993). Pressure in MPa.
_TC, _PC, _RC = 647.096, 22.064, 322.
_IF97 = (0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2, 0.12020824702470e5, -0.32325550322333e7, 0.14915108613530e2,
         -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849, 0.65017534844798e3)
_PSAT = ((-7.85951783, 1.), (1.84408259, 1.5), (-11.7866497, 3.), (22.6807411, 3.5), (-15.9618719, 4.), (1.80122502, 7.5))
_RHOL = ((1.99274064, 1/3), (1.09965342, 2/3), (-.510839303, 5/3), (-1.75493479, 16/3), (-45.5170352, 43/3), (-6.74694450e5, 110/3))
_RHOV = ((-2.03150240, 2/6), (-2.68302940, 4/6), (-5.38626492, 8/6), (-17.2991605, 18/6), (-44.7586581, 37/6), (-63.9201063, 71/6))
_ALPHA = (-1135.905627715, ((-5.65134998e-8, -19.), (2690.66631, 1.), (127.287297, 4.5), (-135.003439, 5.), (.981825814, 54.5)))


def _pw(v, e):
    return v**e if not isinstance(v, E) else (v*v if e == 2. else exp(e*log(v)))


def _lnx(v):
    return log(v) if isinstance(v, E) else math.log(v)


def _expx(v):
    return exp(v) if isinstance(v, E) else math.exp(v)


def _sqx(v):
    return sqrt(v) if isinstance(v, E) else math.sqrt(v)


def sat(p):
    """Saturation at p (MPa): T (K), and the liquid's and vapour's density (kg/m3) and enthalpy (J/kg)."""
    n = _IF97
    b = _pw(p, .25)
    Ee, Fe, Ge = b*b+n[2]*b+n[5], n[0]*b*b+n[3]*b+n[6], n[1]*b*b+n[4]*b+n[7]
    D = 2*Ge/(-Fe-_sqx(Fe*Fe-4*Ee*Ge))
    T = (n[9]+D-_sqx((n[9]+D)*(n[9]+D)-4*(n[8]+n[9]*D)))/2
    t = 1.-T/_TC
    S = sum(a*_pw(t, g) for a, g in _PSAT)
    dS = sum(a*g*_pw(t, g-1.) for a, g in _PSAT)                # dS/dt
    ps = _PC*1e6*_expx(_TC/T*S)                                  # Pa (equal to p within the two equations' consistency)
    dpdT = -ps*(_TC*S/T+dS)/T
    rw = _RC*(1.+sum(c*_pw(t, e) for c, e in _RHOL))
    rs = _RC*_expx(sum(c*_pw(t, e) for c, e in _RHOV))
    th = T/_TC
    al = 1000.*(_ALPHA[0]+sum(d*_pw(th, e) for d, e in _ALPHA[1]))
    return T, rw, rs, al+T/rw*dpdT, al+T/rs*dpdT


DP = 1e-3                                                         # MPa: the step of the central differences in pressure


def _satd(p):
    """sat(p) and its slopes in p (MPa) by central differences (error about 1e-7 of the value)."""
    v, lo, hi = sat(p), sat(p-DP), sat(p+DP)
    return v, tuple((b-a)/(2*DP) for a, b in zip(lo, hi))


# P16/G16, 160 MW, oil fired (Sydkraft, Malmo): the parameters of the paper's Matlab model (TFRT-7577, appendix B,
# with which Figs. 8-13 of the paper compare the model to the plant). The text's values (Sec. 5.1) differ: Ad 20 m2, Vd 40 m3, mr 160 t.
B2 = dict(Vd=37., Vr=37., Vdc=11., Ad=23., mt=300000., mr=100000., md=20000., Cp=650., Cfw=4.18, ke=200., Vsd0=8., beta=.3)
B2['Vt'] = B2['Vd']+B2['Vr']+B2['Vdc']


def _drum(P, Vwt, p, ar, Vsd, Q, qf, tf, qs, k=B2):
    """dVwt, dp, dar, dVsd of the four-state model (Automatica 36 (2000) 363, eqs. 24-25; Q in W, flows kg/s, p MPa)
    and the drum level (m, from the normal level) and the circulation."""
    (T, rw, rs, hw, hs), (dT, drw, drs, dhw, dhs) = _satd(p)
    Vr, Vt, Cp, beta = k['Vr'], k['Vt'], k['Cp'], k['beta']
    hf = (k['Cfw']*tf+p*1e3/rw)*1e3
    hc = hs-hw
    z = ar*(rw-rs)/rs
    lz = _lnx(1.+z)
    av = rw/(rw-rs)*(1.-lz/z)
    davdar = rw/rs*(lz/z-1./(1.+z))/z
    zp = (rs*drw-rw*drs)*ar/(rs*rs)
    z1 = (rw*drs-rs*drw)/((rw-rs)*(rw-rs))
    davdp = z1*(1.-lz/z)+rw/(rw-rs)*(lz/z-1./(1.+z))/z*zp
    qdc = _sqx(k['ke']*(rw-rs)*Vr*av)
    Td = 600./qs
    Vst, Vwd = Vt-Vwt, Vwt-k['Vdc']-(1.-av)*Vr
    e11, e12 = rw-rs, Vst*drs+Vwt*drw
    e21 = hw*rw-hs*rs
    e22 = Vst*(hs*drs+rs*dhs)+Vwt*(hw*drw+rw*dhw)-Vt*1e6+k['mt']*Cp*dT
    e32 = ((1.-ar)*hc*drs+rs*dhs)*av*Vr+(rw*dhw-ar*hc*drw)*(1.-av)*Vr+(rs+(rw-rs)*ar)*hc*Vr*davdp-Vr*1e6+k['mr']*Cp*dT
    e33 = (rs+(rw-rs)*ar)*hc*Vr*davdar
    e42 = (rs*Vsd*dhs+rw*Vwd*dhw-(Vsd+Vwd)*1e6+k['md']*Cp*dT)/hc+(1.+beta)*ar*Vr*((1.-av)*drw+av*drs+(rs-rw)*davdp)+Vsd*drs
    e43 = (1.+beta)*ar*Vr*(rs-rw)*davdar
    b1, b2 = qf-qs, Q+qf*hf-qs*hs
    det = e11*e22-e12*e21
    dV = (b1*e22-e12*b2)/det
    dp = (e11*b2-e21*b1)/det
    da = (Q-qdc*ar*hc-e32*dp)/e33
    dS = ((hf-hw)*qf/hc+rs*(k['Vsd0']-Vsd)/Td-e42*dp-e43*da)/rs
    level = (Vwd+Vsd)/k['Ad']-1.
    return (dV, dp, da, dS), dict(level=level, qdc=qdc, av=av, hs=hs, hf=hf)


def boiler2_point(p, qs, tf, k=B2):
    """The equilibrium at drum pressure p (MPa) with steam flow qs (kg/s) and feedwater temperature tf (C), the drum at
    its normal level: Vwt, ar, Vsd and the heat Q (W) (the paper's eq. 27, the code's Vsd0 and Td = 600/qs)."""
    T, rw, rs, hw, hs = sat(p)
    hf = (k['Cfw']*tf+p*1e3/rw)*1e3
    hc = hs-hw
    Q = qs*(hs-hf)
    lo, hi = 1e-4, .5
    for _ in range(80):                                           # Q = qdc(ar) ar hc: bisection on the steam quality
        ar = (lo+hi)/2
        z = ar*(rw-rs)/rs
        av = rw/(rw-rs)*(1.-math.log(1.+z)/z)
        lo, hi = (ar, hi) if math.sqrt(k['ke']*(rw-rs)*k['Vr']*av)*ar*hc < Q else (lo, ar)
    Vsd = k['Vsd0']-600./qs*(hw-hf)*qs/(rs*hc)
    Vwt = k['Ad']-Vsd+k['Vdc']+(1.-av)*k['Vr']
    return Vwt, ar, Vsd, Q


def boiler2(raw=False):
    """Drum boiler of Astrom and Bell (Automatica 36 (2000) 363; the 160 MW unit P16/G16): total water volume,
    drum pressure, steam quality at the riser outlet and steam volume under the level, saturated properties of
    IAPWS. Shrink and swell come from the steam under the level. Time in s.

    raw=True: the inputs of the paper (heat MW, feedwater and steam flow kg/s, feedwater temperature C) as MVs,
    for the comparison with the plant data (aio/sim/validate/boiler.py). Otherwise the plant an APC sees at medium
    load: heat, feedwater and the steam valve (choked: steam flow in proportion to pressure and opening) as MVs;
    drum pressure, steam flow and drum level tracked (the level integrates)."""
    p0, qs0, tf0 = 8.5, 50., 250.
    Vwt0, ar0, Vsd0, Q0 = boiler2_point(p0, qs0, tf0)
    P = Plant('boiler2', dt=10., substeps=4)
    Vwt, p, ar, Vsd = (P.state(q, v, scale=s) for q, v, s in (('Vwt', Vwt0, 1.), ('p', p0, .1), ('ar', ar0, .005), ('Vsd', Vsd0, .5)))
    if raw:
        Q = P.mv('Q', 0., 200., 5., Q0/1e6)*1e6
        qf, qs, tf = P.mv('qf', 0., 160., 5., qs0), P.mv('qs', 10., 160., 5., qs0), P.mv('tf', 150., 300., 5., tf0)
    else:
        Q, qf = P.mv('Q', 40., 160., 2., Q0/1e6)*1e6, P.mv('qf', 20., 110., 3., qs0)
        u = P.mv('valve', .2, 1., .02, .5)
        tf = P.load('tf', tf0, tf0-15., tf0+15.)
        kv = P.param('kv', 1., .93, 1.05)
        qs = kv*qs0/(.5*p0)*u*p
    if raw:
        qs_true = qs+4.8*(p-8.5)                                  # the code's correction of the plant's steam flow transducer
    else:
        qs_true = qs
    eta = P.param('eta', 1., .95, 1.03)                           # heat release to the risers per unit of firing (the engine needs one parameter)
    d, o = _drum(P, Vwt, p, ar, Vsd, eta*Q, qf, tf, qs_true)
    for x, dx in zip((Vwt, p, ar, Vsd), d):
        P.der(x, dx)
    if raw:
        for nm, v in (('p', p), ('level', o['level']), ('Vwt', Vwt), ('ar', ar), ('Vsd', Vsd), ('qdc', o['qdc']), ('av', o['av'])):
            P.out(nm, v, range=(0., 1.), band=1.)
        return P
    P.out('p', p, range=(7., 10.), band=.02, track=True, target=(7.8, 9.2), hi=9.8)
    P.out('qs', qs, range=(25., 80.), band=.5, track=True, target=(38., 62.))
    P.out('level', o['level'], range=(-.4, .4), band=.01, track=True, target=(-.08, .08), lo=-.25, hi=.25, integ=True)
    P.out('ar', ar, range=(0., .15), band=.002)
    return P


HELDOUT = dict(boiler=boiler, wo=wo, mma=mma, cryst=cryst, boiler2=boiler2)

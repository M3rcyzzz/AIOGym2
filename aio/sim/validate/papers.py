"""The literature plants written a second time, from the papers, in plain numpy, and the numbers the papers
publish.

Nothing here is taken from aio.sim.plants: checks.py compares the two transcriptions at random points, and
the engine's plant with the published operating points, time constants, gains and zeros. Each function takes
a dict of named values (the engine's names for states and inputs; the paper's constants are written here)
and returns the derivatives by state name (for an algebraic state: the residual).

A published number is given as the paper prints it. Where a paper's own table does not satisfy its own
equations (rounded operating points), the entry says so and carries the tolerance that follows from the
rounding, not a tolerance chosen to pass.
"""
import numpy as np


# ---------------------------------------------------------------- Johansson's quadruple-tank process
def qtank(v):
    """K. H. Johansson, 'The quadruple-tank process', IEEE Trans. Control Syst. Technol. 8(3), 2000, eq. (1).
    cm, s, V. Cross-sections A1 = A3 = 28, A2 = A4 = 32 cm2; outlets a1 = a3 = 0.071, a2 = a4 = 0.057 cm2."""
    A1, A2, A3, A4 = 28., 32., 28., 32.
    a1, a2, a3, a4 = .071, .057, .071, .057
    g = 981.
    k1, k2, g1, g2 = v.get('k1', 3.33), v.get('k2', 3.35), v.get('g1', .70), v.get('g2', .60)
    r = lambda h: np.sqrt(2*g*h)      # noqa: E731
    return dict(h1=-a1/A1*r(v['h1'])+a3/A1*r(v['h3'])+g1*k1/A1*v['v1'],
                h2=-a2/A2*r(v['h2'])+a4/A2*r(v['h4'])+g2*k2/A2*v['v2'],
                h3=-a3/A3*r(v['h3'])+(1-g2)*k2/A3*v['v2'],
                h4=-a4/A4*r(v['h4'])+(1-g1)*k1/A4*v['v1'])


def _zeros2(K, T1, T2, T3, T4):
    """Zeros of the printed transfer matrix [[K11/(1+T1 s), K12/((1+T3 s)(1+T1 s))], [K21/((1+T4 s)(1+T2 s)), K22/(1+T2 s)]]:
    the roots of K11 K22 (1+T3 s)(1+T4 s) = K12 K21."""
    (k11, k12), (k21, k22) = K
    return tuple(sorted(np.roots([k11*k22*T3*T4, k11*k22*(T3+T4), k11*k22-k12*k21]).real))


# read in this session in Johansson & Nunes, ACC 1998 (the same rig and tables as the 2000 paper): the parameter table, both
# operating points, and the two printed transfer matrices. The zeros and relative gains follow from the printed matrices and gammas.
QTANK = dict(
    source='Johansson 2000 / Johansson & Nunes, ACC 1998: parameter table, operating points, transfer matrices', read=True,
    kc=.5,                                                        # V/cm, the level sensors: the published gains are in V/V
    minus=dict(h=(12.4, 12.7, 1.8, 1.4), v=(3.00, 3.00), k=(3.33, 3.35), g=(.70, .60), T=(62., 90., 23., 30.), gain=((2.6, 1.5), (1.4, 2.8))),
    plus=dict(h=(12.6, 13.0, 4.8, 4.9), v=(3.15, 3.15), k=(3.14, 3.29), g=(.43, .34), T=(63., 91., 39., 56.), gain=((1.5, 2.5), (2.5, 1.6))))
for _d in (QTANK['minus'], QTANK['plus']):
    _d['zeros'] = _zeros2(_d['gain'], *_d['T'])
    _d['rga'] = _d['g'][0]*_d['g'][1]/(_d['g'][0]+_d['g'][1]-1)


# ---------------------------------------------------------------- pH neutralisation
PH_PK2 = 10.25      # the nominal table of the Henson-Seborg process as reprinted by Hu & Rangaiah (Chem. Eng. Sci. 54, 1999) and Medi & Asadbeigi (Heliyon 7, 2021, Table 3); the 1994 paper itself was not read here


def ph(v):
    """M. A. Henson, D. E. Seborg, 'Adaptive nonlinear control of a pH neutralization process', IEEE Trans.
    Control Syst. Technol. 2(3), 1994, eqs. (1)-(6) and Table I. cm, ml/s, mol/l."""
    A, z, n, Cv = 207., 11.5, .607, 4.59
    Wa1, Wa2, Wa3, Wb1, Wb2, Wb3 = 3e-3, -3e-2, -3.05e-3, 0., 3e-2, 5e-5
    pK1, pK2 = v.get('pk1', 6.35), v.get('pk2', PH_PK2)
    h, wa, wb, pH = v['h'], v['wa'], v['wb'], v['pH']
    q1, q2, q3 = v['q1'], v['q2'], v['q3']
    return dict(h=(q1+q2+q3-Cv*(h+z)**n)/A,
                wa=(q1*(Wa1-wa)+q2*(Wa2-wa)+q3*(Wa3-wa))/(A*h),
                wb=(q1*(Wb1-wb)+q2*(Wb2-wb)+q3*(Wb3-wb))/(A*h),
                pH=wa+10.**(pH-14.)+wb*(1+2*10.**(pH-pK2))/(1+10.**(pK1-pH)+10.**(pH-pK2))-10.**(-pH))


# read in this session in the reprint of the nominal table (Medi & Asadbeigi 2021, Table 3, after Hu & Rangaiah 1999): streams, h, pH, Wa4, Wb4,
# pK1, pK2. The outflow law Cv (h + z)^n with 4.59, 0.607, 11.5 is quoted from memory of the 1994 paper (the reprint has Cv sqrt(h), 8.75:
# the same flow at the nominal level).
PH = dict(source='Henson & Seborg 1994, nominal table (read in the reprint of Medi & Asadbeigi 2021, Table 3)', read=True,
          q=(16.6, .55, 15.6), h=14.0, pH=7.0, wa4=-4.32e-4, wb4=5.28e-4)


# ---------------------------------------------------------------- Van de Vusse reactor
def vdv(v):
    """K.-U. Klatt, S. Engell, 'Gain-scheduling trajectory control of a continuous stirred tank reactor',
    Comput. Chem. Eng. 22(4/5), 1998, eqs. (1)-(4) and Table 1 (the benchmark of Chen, Kremling, Allgoewer,
    ECC 1995). mol/l, degC, h; r = Vdot/VR in 1/h, QK in kJ/h."""
    k10, k20, k30 = 1.287e12, 1.287e12, 9.043e9
    E1, E2, E3 = -9758.3, -9758.3, -8560.
    dHab, dHbc, dHad = 4.2, -11.0, -41.85
    rho, Cp, kw, AR, VR, mK, CpK = .9342, 3.01, 4032., .215, 10., 5.0, 2.0
    cA, cB, T, TK = v['cA'], v['cB'], v['T'], v['TK']
    k = lambda k0, E: k0*np.exp(E/(T+273.15))      # noqa: E731
    k1, k2, k3 = k(k10, E1), k(k20, E2), k(k30, E3)
    return dict(cA=v['r']*(v['cA0']-cA)-k1*cA-k3*cA**2,
                cB=-v['r']*cB+k1*cA-k2*cB,
                T=v['r']*(v['T0']-T)-(k1*cA*dHab+k2*cB*dHbc+k3*cA**2*dHad)/(rho*Cp)+kw*AR/(rho*Cp*VR)*(TK-T),
                TK=(v['QK']+kw*AR*(T-TK))/(mK*CpK))


VDV = dict(source='Klatt & Engell 1998, Table 1 and operating point (quoted from memory)', read=False, r=14.19, QK=-1113.5, cA0=5.10, T0=104.9, cA=2.14, cB=1.09, T=114.2, TK=112.9)


# ---------------------------------------------------------------- exothermic CSTR
def cstr(v):
    """D. E. Seborg, T. F. Edgar, D. A. Mellichamp, F. J. Doyle, 'Process Dynamics and Control', the CSTR of
    section 2.4.6 (eqs. 2-66, 2-68, Table 2.3). mol/l, K, min. Tc is the coolant temperature seen by the
    reactor (the engine's jacket state)."""
    q, V, rho, C, mdH, ER, k0, UA = v['q'], 100., 1000., .239, 5e4, 8750., 7.2e10, 5e4
    cA, T = v['CA'], v['T']
    k = k0*np.exp(-ER/T)
    return dict(CA=q/V*(v['CAf']-cA)-k*cA,
                T=q/V*(v['Tf']-T)+mdH/(rho*C)*k*cA+UA/(V*rho*C)*(v['Tj']-T))


CSTR = dict(source='Seborg et al., Table 2.3 (quoted from memory)', read=False, q=100., CAf=1., Tf=350., Tc=300., CA=.5, T=350.)


# ---------------------------------------------------------------- continuous fermenter
def bio(v):
    """B. W. Bequette, 'Process Dynamics: Modeling, Analysis and Simulation', 1998, Module 8 (biochemical
    reactor, substrate inhibition), with the product equation of Henson & Seborg, 'Nonlinear Process
    Control', 1997, ch. 4. g/l, h."""
    mum, Km, K1, Y = v.get('mum', .53), v.get('Km', .12), v.get('K1', .4545), v.get('Y', .4)
    alpha, beta = v.get('alpha', 2.2), .2
    X, S, P = v['X'], v['S'], v['P']
    mu = mum*S/(Km+S+K1*S**2)
    return dict(X=(mu-v['D'])*X, S=v['D']*(v['Sf']-S)-mu*X/Y, P=-v['D']*P+(alpha*mu+beta)*X)


BIO = dict(source='Bequette 1998, Module 8 (quoted from memory)', read=False, D=.3, Sf=4., X=1.5302, S=.1746, other=((.9951, 1.5122), (0., 4.)))


# ---------------------------------------------------------------- forced-circulation evaporator
def evap(v):
    """R. B. Newell, P. L. Lee, 'Applied Process Control: A Case Study', Prentice Hall 1989, ch. 2 (model
    equations and Table of steady-state values). kg/min, %, kPa, degC, m, min."""
    F1, X1, T1, T200 = v['F1'], v['X1'], v['T1'], v['T200']
    F2, P100, F200 = v['F2'], v['P100'], v['F200']
    L2, X2, P2 = v['L2'], v['X2'], v['P2']
    F3, rhoA, M, C, lam, Cp = 50., 20., 20., 4., 38.5, .07
    T2 = .5616*P2+.3126*X2+48.43
    T3 = .507*P2+55.
    T100 = .1538*P100+90.
    Q100 = .16*(F1+F3)*(T100-T2)
    F4 = (Q100-F1*Cp*(T2-T1))/lam
    UA2 = 6.84
    Q200 = UA2*(T3-T200)/(1+UA2/(2*Cp*F200))
    F5 = Q200/lam
    return dict(L2=(F1-F4-F2)/rhoA, X2=(F1*X1-F2*X2)/M, P2=(F4-F5)/C,
                _T201=T200+Q200/(F200*Cp), _T2=T2, _F4=F4, _Q100=Q100, _Q200=Q200, _F100=Q100/36.6)


# the table of steady values was read in this session as reprinted by Switalski, Control and Cybernetics 28(2), 1999, Table 1
EVAP = dict(source='Newell & Lee 1989, table of steady values (read in the reprint of Switalski 1999, Table 1)', read=True,
            nominal=dict(F1=10., F2=2., F4=8., F5=8., X1=5., X2=25., T1=40., T2=84.6, T3=80.6, L2=1., P2=50.5, F100=9.3, T100=119.9,
                         P100=194.7, Q100=339., F200=208., T200=25., T201=46.1, Q200=307.9))


# ---------------------------------------------------------------- distillation column A
def column(v, NT=41, NF=21):
    """S. Skogestad, M. Morari, 'Understanding the dynamic behavior of distillation columns', Ind. Eng. Chem.
    Res. 27, 1988 (column A), as written out in colamod.m with the level loops of cola_lv.m (S. Skogestad,
    NTNU). Stages are numbered from the reboiler (1) to the condenser (NT). kmol, min."""
    alpha = v.get('alpha', 1.5)
    M0, taul, F0, qF0, L0 = .5, .063, 1., 1., 2.70629
    L0b = L0+qF0*F0
    x, M = np.asarray(v['x'], dtype=float), np.asarray(v['M'], dtype=float)
    LT, VB, F, zF, qF = v['LT'], v['VB'], v['F'], v['zF'], v['qF']
    KcB, KcD, MDs, MBs, Ds, Bs = 10., 10., .5, .5, .5, .5
    D, B = Ds+(M[NT-1]-MDs)*KcD, Bs+(M[0]-MBs)*KcB
    y = alpha*x[:NT-1]/(1+(alpha-1)*x[:NT-1])
    V = np.full(NT-1, VB)
    V[NF-1:] += (1-qF)*F
    L = np.zeros(NT)
    L[NT-1] = LT
    L[1:NF] = L0b+(M[1:NF]-M0)/taul
    L[NF:NT-1] = L0+(M[NF:NT-1]-M0)/taul
    dM, dMx = np.zeros(NT), np.zeros(NT)
    i = np.arange(1, NT-1)
    dM[i] = L[i+1]-L[i]+V[i-1]-V[i]
    dMx[i] = L[i+1]*x[i+1]-L[i]*x[i]+V[i-1]*y[i-1]-V[i]*y[i]
    dM[NF-1] += F
    dMx[NF-1] += F*zF
    dM[0] = L[1]-V[0]-B
    dMx[0] = L[1]*x[1]-V[0]*y[0]-B*x[0]
    dM[NT-1] = V[NT-2]-LT-D
    dMx[NT-1] = V[NT-2]*y[NT-2]-LT*x[NT-1]-D*x[NT-1]
    return dict(x=(dMx-x*dM)/M, M=dM)


# read in this session on Skogestad's page of the model (skoge.folk.ntnu.no/book/matlab_m/cola/cola.html): the column data and the
# slowest eigenvalue, -5.1562e-03 per min. The gain matrix and its relative gain are the distillation example of Skogestad &
# Postlethwaite, 'Multivariable Feedback Control' (a model 'of column A'), quoted from memory.
COLUMN = dict(source="Skogestad's column A page (data, slowest eigenvalue); Skogestad & Postlethwaite (gains, quoted from memory)", read=True,
              LT=2.70629, VB=3.20629, xD=.99, xB=.01, eig1=-5.1562e-3,
              gain=((87.8, -86.4), (108.2, -109.6)),              # d(yD, xB)/d(L, V), in units of 0.01 mole fraction per kmol/min, LV configuration
              rga=35.1)

PAPERS = dict(qtank=qtank, ph=ph, vdv=vdv, excstr=cstr, bio=bio, evap=evap, column=column)

"""Vinyl acetate monomer plant as data for the mechanism engine (docs/sim.md, 'VAM').

The plant of Luyben and Tyreus (Comput. Chem. Eng. 22, 1998) as coded in C by Chen, Dave, McAvoy and Luyben
(Ind. Eng. Chem. Res. 42, 2003), routine by routine: vaporizer, plug-flow reactor (ten sections), feed-effluent
exchanger, separator with flash, compressor, absorber (eight trays and base), CO2 removal, purge, azeotropic
column (twenty trays, reboiler, decanter) and HAc tank. Seven components (O2, CO2, C2H4, C2H6, VAc, H2O, HAc),
Wilson activities, Antoine pressures. Units are the code's: kmol, min, degC, psia, kcal.

Where the code iterates (bubble points of the column trays and the reboiler, the separator flash, the decanter
split) the table carries the converged values: Newton steps unrolled from the code's own starting points,
the decanter split in closed form.

`vam_raw()` is the open-loop process: 246 states and a clock (the catalyst ages with time, as in the code), the 26
MVs, the 43 measurements (compared with the native code, aio/sim/validate/vam.py). `vam()` is the plant an APC
sees: Luyben's regulatory layer (the paper's Appendix 2: 22 loops and four fixed flows, the three analysers with their 10 min dead
time as ten-stage lags) and the MVs are its setpoints and fixed flows; a decision is 10 min, the analysers' cycle.
"""
import math

import torch

from aio.sim.engine import Plant, Mech
from aio.sim.expr import E, exp, log, sqrt, step, relu, clamp, maximum, where

COMP = ('O2', 'CO2', 'C2H4', 'C2H6', 'VAc', 'H2O', 'HAc')
ST = (0, 1, 2, 4, 5, 6)                      # the components carried as states (C2H6 is the rest)
VI = (64.178, 37.400, 49.347, 52.866, 101.564, 18.01, 61.445)
AIJ = {(4, 5): 1384.6, (4, 6): -136.1, (5, 4): 2266.4, (5, 6): 670.7, (6, 4): 726.7, (6, 5): 230.6}
MW = (32., 44.01, 28.052, 30.068, 86.088, 18.008, 60.052)
SPG = (.5, 1.18, .57, .57, .85, 1., .98)
HVAP = (2300., 2429., 1260., 1260., 8600., 10684., 5486.)
CLA, CLB = (.3, .6, .6, .6, .44, .99, .46), (0., 0., 0., 0., .0011, .0002, .0012)
CVA, CVB = (.218, .23, .37, .37, .29, .56, .52), (.0001, 0., .0007, .0007, .0006, -.0016, .0007)
AA = (9.2, 7.937, 9.497, 9.497, 12.6564, 14.6394, 14.5236)
BB = (0., 0., -313., -313., -2984.45, -3984.92, -4457.83)
CC = (273.16, 273.16, 273.16, 273.16, 226.66, 233.43, 258.45)
RG, DW, TREF = 1.987, 999., 273.16
Y_O2 = (1., 0., 0., 0., 0., 0., 0.)
X_HAC = (0., 0., 0., 0., 0., 0., 1.)
T_FEED = 30.                                 # O2, C2H4 and HAc feeds
# equipment
VW_VAP, VW_SEP, V_LOOP, UA_SEP = 4., 8., 170., 20007.*.252*9/5
NR, TUBE_AREA, UA_RCT, RHO_CAT, CP_CAT, EPS = 11, .67240058914293, 269.839144893182, 385., .23, .8
E_R1, E_R2 = -42100., -316000.
STO1, STO2 = (-.5, 0., -1., 0., 1., 1., -1.), (-3., 2., -1., 0., 0., 2., 0.)
FRICTION = .000794745091749963
UA_FEHE, REF_HOT, REF_COLD = 6483./60., 589.670, 498.952
COMP_COEF = 15000.
NT_ABS, NF_ABS, VW_ABS, HYDTAU = 8, 2, 8.5, .1
M0_ABS = 30/2.20462
L0_ABS = (16.327, 16.311, .911, .885, .853, .844, .829, .815)
NMT = 60/2.20462
QMT = (400*.252, 400*.252)+(200*.252,)*6
CO2_F_REF, CO2_X_REF = 6.41360260517487, .0134241598608995
NT_COL, NF_COL, P_BOT = 20, 15, 30.
K_DEC = (395., .05, 1.)
M0_COL, L0_COL = 2.3, (8.7947,)*15+(4.9741,)*5
VW_BOT, VW_ORG, VW_AQ, VW_TANK = 5.66, 1.7, 1.7, 2.83
T_GUESS = (131, 121, 106, 96, 93, 92, 91, 90, 89, 88, 88, 87, 86, 85, 84, 80, 78, 77, 76, 75)
YEAR = 60*24*365.
BUBBLE_ITERS, FLASH_OUTER, FLASH_INNER = 6, 4, 2
MVS_RAW = ('F_O2', 'F_C2H4', 'F_HAc', 'Q_vap', 'F_vap', 'Q_heater', 'T_shell', 'F_sepliq', 'T_sepshell', 'F_sepvap', 'Q_comp', 'F_absliq', 'F_circ',
           'Q_circ', 'F_scrub', 'Q_scrub', 'F_CO2', 'F_purge', 'bypass', 'F_reflux', 'Q_reb', 'Q_cond', 'F_org', 'F_aq', 'F_bot', 'F_vapliq')
MEAS = ('P_vap', 'L_vap', 'T_vap', 'T_heater', 'T_rct', 'F_rct', 'T_fehe_cold', 'T_fehe_hot', 'L_sep', 'T_sep', 'T_comp', 'P_rec', 'L_abs', 'T_circ',
        'T_scrub', 'F_rec', 'F_org', 'L_org', 'L_aq', 'T_dec', 'L_bot', 'T_tray5', 'L_tank')+tuple(f'xorg_{c}' for c in COMP[4:])+tuple(
        f'xbot_{c}' for c in COMP[4:])+tuple(f'yrec_{c}' for c in COMP)+tuple(f'yrct_{c}' for c in COMP)
X0 = (
    0.000663659407133395, 0.000332732450155642, 0.0134912199479539, 0.00162751551202943, 0.0271065270204093, 0.951882997844851,
    0.699999996824111, 119.145219885782, 0.0175589716398125, 0.0162929429097801, 0.0150834006635812, 0.0139332286281951,
    0.0128463158160867, 0.0118263519039695, 0.0108758835733143, 0.00999566895667261, 0.00918441933033151, 0.0084389570853562,
    0.00159553696160461, 0.00162458880807157, 0.00166274684754579, 0.0017062209092524, 0.00175126306173113, 0.00179431965646144,
    0.0018322211755343, 0.00186236754739096, 0.00188285688293816, 0.00189252049871014, 0.140819033783875, 0.134754781991664,
    0.12903327158763, 0.12360615867642, 0.118440144936224, 0.113509157452373, 0.108788990644953, 0.104254050243748,
    0.099876188529604, 0.0956254359469808, 0.00136784321300813, 0.00234999379462041, 0.00327584967176358, 0.00413290171322466,
    0.00491174993955765, 0.00560570677851114, 0.00621051389698821, 0.00672409205292511, 0.00714627791095438, 0.00747853616589494,
    0.00318437552973624, 0.00418774557402648, 0.0051443768726401, 0.0060378731408602, 0.00685501757742005, 0.00758551558109247,
    0.00822189285800267, 0.00875942531335696, 0.0091960024956991, 0.00953187545645922, 0.0255550841888396, 0.0236187475024977,
    0.021793498578692, 0.0200800315537184, 0.0184793537862666, 0.0169916497340827, 0.0156155197925629, 0.0143475877037275,
    0.0131825069898176, 0.0121133356791918, 152.515501748242, 155.641592023486, 158.015939863408, 159.73262963317,
    160.85183921479, 161.417694492952, 161.472975701556, 161.069917928174, 160.275450900194, 159.169995766986,
    0.00132380406401621, 0.000719612261903781, 0.0236372995127054, 0.174008547825989, 0.334038451182119, 0.456451002498642,
    0.500046419718972, 39.9999988383099, 0.0569465492045698, 0.0126985233423726, 0.643808436634138, 0.0209017186689456,
    0.00743940828211637, 0.00398954930049825, 84.2519845772682, 41.0012287591771, 0.00126828905962982, 0.00128403831066931,
    0.0010136884642688, 0.000950312133226393, 0.000931584631849887, 0.000918819492877675, 0.00090593863273761, 0.000891015112493328,
    0.000917400271826095, 0.000924586355608594, 0.000799401296569979, 0.000767258962839511, 0.000753759239015543, 0.000743904564497274,
    0.000734233796173805, 0.000722036192308671, 0.0259513974177929, 0.0272833807330475, 0.0223547022666141, 0.0212170812606176,
    0.0207925863280729, 0.020482612444623, 0.0201776448585294, 0.0199095449822689, 0.255554217009127, 0.254570175421162,
    0.142827718835525, 0.111241714052836, 0.0876375405894624, 0.0694647126836213, 0.0524258536179032, 0.0325036209994564,
    0.129127357377499, 0.125762009872873, 0.0741286897275229, 0.0594192297171773, 0.0604303190638807, 0.0622188868900966,
    0.0636812502036909, 0.065395390490694, 0.576778642679934, 0.579234099392294, 0.750018257267289, 0.798023710798955,
    0.821243597976687, 0.838083598390124, 0.854109180620964, 0.872719450601011, 13.6084339670806, 13.6061145578899,
    13.6068981796877, 13.6066292515065, 13.6079016059848, 13.6073496310857, 13.6076729839204, 13.6075910770526,
    47.8987514441403, 35.1279984080967, 39.0133484644232, 39.2355888214155, 39.70693699704, 40.2031532587549,
    40.6647299086819, 39.6960239415672, 0.00126828905855437, 0.000917400290598044, 0.0259513993773912, 0.255554230775313,
    0.129127360768626, 0.576778624920724, 0.499946538488202, 47.8987516921869, 5.55839747640732e-05, 0.000276708140287114,
    0.00144598318785804, 0.00837754598891844, 0.0505364673088945, 0.226382222105162, 0.447991466512184, 0.530995397736738,
    0.555910888115674, 0.564809573155954, 0.56882919324755, 0.57114460340736, 0.572803782917262, 0.574200214532656,
    0.575495385413331, 0.805783827190703, 0.880403477943325, 0.906875683571773, 0.9184060942439, 0.92599255852643,
    0.806269830806122, 0.672900931578798, 0.550686777995853, 0.470472677932738, 0.419202664992037, 0.347968715736377,
    0.269611659810616, 0.243074248379238, 0.237878819499972, 0.237261167498344, 0.237433332635326, 0.237685764504895,
    0.237917513711896, 0.238129770517882, 0.238333033291516, 0.0775672581141022, 0.0268791043104891, 0.00966301925493509,
    0.00347298767319165, 0.0011971531531783, 2.64109552188728, 2.55082805050048, 2.48144065400843, 2.43735186419207,
    2.40256883580744, 2.39176741421249, 2.42649648114181, 2.44573720385547, 2.45121159989488, 2.45231329683537,
    2.45185115757585, 2.45074398149757, 2.44934331576837, 2.44779485561834, 2.44616086713232, 2.37268484694175,
    2.37511835481015, 2.37516769486192, 2.37504695877361, 2.37522269207844, 1.08213392363876e-05, 0.906549438617103,
    0.500007499269263, 0.949785850539446, 0.000351974228354218, 0.499985183304765, 0.00240452114142709, 0.000351973887173725,
    0.499986567556951, 7.92174932346166e-06, 0.931589602930866, 0.500018753056283, 109.773605988319, 150.0,
    80.0, 25.0, 25.0, 45.8449999779885, 97.1009365439832, 133.999999835572)
U0 = (
    0.52343003710111, 0.835217268934129, 0.790026413554696, 21876.9877721852, 18.728267009014, 9008.5406935041,
    135.018391707178, 2.75438428905024, 36.0012295162535, 16.1026, 27192.4219484591, 1.21366689616643,
    15.1198, 10729.8900873211, 0.756, 2018.43004751657, 6.55307587098636, 0.00315687443512887,
    0.313026840552122, 4.9848844779243, 67178.8579112547, 60367.3972613543, 0.829040050877669, 0.836077056930357,
    2.15837365008457, 2.1924)

# the paper's Appendix 3: the 43 measurements at the base (the code's own base reproduces them)
PAPER = (128., .7, 119.145, 150., 159.17, 18.857, 97.1, 134., .5, 40., 80., 128., .5, 25., 25., 16.5359, .829, .5, .5, 45.845, .5, 110., .5,
         .949786, .049862, .000352, .000010, .093440, .906550, .055664, .007304, .681208, .249191, .001597, .000894, .004142, .075000, .006273,
         .585110, .214038, .001373, .008558, .109648)


# ---------------------------------------------------------------- arithmetic on expressions or numbers
def _ex(v):
    return exp(v) if isinstance(v, E) else math.exp(min(max(v, -80.), 80.))


def _ln(v):
    return log(v) if isinstance(v, E) else math.log(v)


def _sq(v):
    return sqrt(v) if isinstance(v, E) else math.sqrt(max(v, 0.))


def _h(v):
    return v.held if isinstance(v, E) else v


def _s(terms):
    """A sum, held; terms that are the number zero are left out."""
    tot = 0.
    for t in terms:
        if isinstance(t, E) or t:
            tot = tot+t
    return _h(tot)


def _full(xs, rest):
    """Seven mole fractions from the six carried (ST) with component `rest` the remainder."""
    x = [0.]*7
    for i, v in zip(ST, xs):
        x[i] = v
    x[rest] = _h(1.-_s(x[i] for i in range(7) if i != rest))
    return x


def _lam(T, idx=range(7)):
    """Wilson's Lambda at T (gamma_wilson_7)."""
    tk = RG*(T+TREF)
    return {(i, j): VI[j]/VI[i]*(_ex(-AIJ[i, j]/tk) if (i, j) in AIJ else 1.) for i in idx for j in idx}


def _gamma(L, x, idx=range(7)):
    s = {i: _s(L[i, j]*x[j] for j in idx if isinstance(x[j], E) or x[j]) for i in idx}
    return {i: _h(_ex(1.-_ln(s[i])-_s(L[j, i]/s[j]*x[j] for j in idx if isinstance(x[j], E) or x[j]))) for i in idx}


def _ps(j, T):
    return _ex(AA[j]+BB[j]/(T+CC[j])) if BB[j] else math.exp(AA[j])


def _hl(j, T):
    return MW[j]*T*(CLA[j]+.5*CLB[j]*T)


def _hv(j, T):
    return MW[j]*T*(CVA[j]+.5*CVB[j]*T)+HVAP[j]


def _HL(x, T, idx=range(7)):
    return _s(x[j]*_hl(j, T) for j in idx if isinstance(x[j], E) or x[j])


def _HV(y, T, idx=range(7)):
    return _s(y[j]*_hv(j, T) for j in idx if isinstance(y[j], E) or y[j])


def _cpl(x, T, idx=range(7)):
    """sum x MW (cp_l(T)): the heat capacity of a liquid holdup per mole."""
    return _s(x[j]*MW[j]*(CLA[j]+CLB[j]*T) for j in idx if isinstance(x[j], E) or x[j])


def _quad(A, B, Cq):
    return (-B+_sq(B*B-4*A*Cq))/(2*A)


def _t_gas(y, Hm, idx=range(7)):
    """The temperature of a vapour of composition y with molar enthalpy Hm."""
    nz = [j for j in idx if isinstance(y[j], E) or y[j]]
    return _quad(_s(.5*CVB[j]*MW[j]*y[j] for j in nz), _s(CVA[j]*MW[j]*y[j] for j in nz), _s(HVAP[j]*y[j] for j in nz)-Hm)


def _t_liq(x, Hm, idx=range(7)):
    nz = [j for j in idx if isinstance(x[j], E) or x[j]]
    return _quad(_s(.5*CLB[j]*MW[j]*x[j] for j in nz), _s(CLA[j]*MW[j]*x[j] for j in nz), -Hm)


def _mix(F1, x1, T1, F2, x2, T2, gas):
    """mixer.c: flow, composition and temperature of two streams joined."""
    F = _h(F1+F2)
    x = [_h((F1*x1[i]+F2*x2[i])/F) if any(isinstance(v, E) or v for v in (x1[i], x2[i])) else 0. for i in range(7)]
    if gas:
        return F, x, _t_gas(x, _h((F1*_HV(x1, T1)+F2*_HV(x2, T2))/F))
    return F, x, _t_liq(x, _h((F1*_HL(x1, T1)+F2*_HL(x2, T2))/F))


def _conc(x, idx=range(7)):
    """Liquid molar concentration (kmol/m3) from the specific gravities."""
    return _h(DW/_s(x[j]*MW[j]/SPG[j] for j in idx if isinstance(x[j], E) or x[j]))


def _bubble(T, P, x):
    """bubpt_column: the bubble point of the VAc-H2O-HAc liquid x (indices 4-6) at P, Newton from T as the code
    does (its slope leaves out the activities' temperature dependence). Returns T and the vapour."""
    idx = (4, 5, 6)
    for _ in range(BUBBLE_ITERS):
        g = _gamma(_lam(T, idx), x, idx)
        k = {j: _h(_ps(j, T)*x[j]*g[j]) for j in idx}
        T = _h(T-(_s(k.values())-P)/_s(-BB[j]*k[j]/((T+CC[j])*(T+CC[j])) for j in idx))
    g = _gamma(_lam(T, idx), x, idx)
    y = [0.]*7
    for j in idx:
        y[j] = _h(_ps(j, T)*x[j]*g[j]/P)
    return T, y


def _names():
    n = [f'vap_x{COMP[c]}' for c in ST]+['vap_L', 'vap_T']
    n += [f'rct_{COMP[c]}{s}' for c in ST for s in range(1, NR)]+[f'rct_T{s}' for s in range(1, NR)]
    n += [f'sep_x{COMP[c]}' for c in ST]+['sep_L', 'sep_TL']+[f'sep_y{COMP[c]}' for c in ST]+['sep_P', 'sep_TV']
    n += [f'abs_x{COMP[c]}{i}' for c in ST for i in range(NT_ABS)]+[f'abs_M{i}' for i in range(NT_ABS)]+[f'abs_T{i}' for i in range(NT_ABS)]
    n += [f'abs_xb{COMP[c]}' for c in ST]+['abs_L', 'abs_Tb']
    n += [f'col_xVAc{i}' for i in range(NT_COL)]+[f'col_xHAc{i}' for i in range(NT_COL)]+[f'col_M{i}' for i in range(NT_COL)]
    n += ['bot_xVAc', 'bot_xHAc', 'bot_L', 'org_xVAc', 'org_xHAc', 'org_L', 'aq_xVAc', 'aq_xHAc', 'aq_L', 'tank_xVAc', 'tank_xHAc', 'tank_L', 'tank_T']
    n += ['T_heater', 'T_comp', 'T_circ', 'T_scrub', 'T_dec', 'T_fehe_cold', 'T_fehe_hot']
    return tuple(n)


NAMES = _names()


def _states(P):
    X = []
    for i, nm in enumerate(NAMES):
        temp = nm.startswith(('T_', 'tank_T', 'abs_Tb', 'vap_T', 'sep_T', 'rct_T', 'abs_T'))
        X.append(P.state(nm, X0[i], scale=10. if temp else max(abs(X0[i]), 1e-3)))
    return X


def _vaporizer(X, F_out, Q, Fg, yg, Tg, Fl, xl, Tl):
    """Vaporizer.c: liquid holdup with gas and liquid feeds, vapour leaves at the bubble point."""
    x = _full(X[0:6], 3)
    L, T = X[6], X[7]
    conc = _conc(x)
    vol = L*VW_VAP
    g = _gamma(_lam(T), x)
    k = [_h(x[i]*_ps(i, T)*g[i]) for i in range(7)]
    Pv = _s(k)
    y = [_h(v/Pv) for v in k]
    HLv, HVv = _HL(x, T), _HV(y, T)
    Md = _h(Fg+Fl-F_out)
    xd = [(Fg*yg[i]+Fl*xl[i]-F_out*y[i]-x[i]*Md)/conc/vol for i in ST]
    Td = (Fg*(_HV(yg, Tg)-HLv)+Fl*(_HL(xl, Tl)-HLv)-F_out*(HVv-HLv)+Q)/(conc*vol)/_cpl(x, T)
    return xd+[Md/conc/VW_VAP, Td], dict(P=Pv, y=y, T=T, L=L)


def _rates(Pp, T, act, age):
    """reaction_rate.c: kmol/min per kg catalyst of VAc formation and ethylene combustion, Pp partial pressures (psia)."""
    tk = T+273.15
    decay = act*(_ex(-age/YEAR/.621) if isinstance(age, E) else math.exp(-age/YEAR/.621))
    alpha = decay*where(T-180., _ex(-(T-180.)/50.), 1.)
    w = 1.+1.7*Pp[5]
    r1 = .1036*_ex(-3674./tk)*Pp[0]*Pp[2]*Pp[6]*w/((1.+.583*Pp[0]*w)*(1.+6.8*Pp[6]))
    w2 = 1.+.68*Pp[5]
    r2 = 1.9365e5*_ex(-10116./tk)*Pp[0]*w2/(1.+.76*Pp[0]*w2)
    return _h(alpha*r1*RHO_CAT), _h(alpha*r2*RHO_CAT)


def _reactor(X, Fin, yin, Tin, Pin, Cin, dP, Tsh, act, age, uar):
    """Reactor.c: ten sections, backward differences, gas and bed at one temperature."""
    Pp = [Pin-dP*(i/(NR-1)) for i in range(NR)]
    Gm = _h(Fin*_s(MW[j]*yin[j] for j in range(7))/TUBE_AREA)
    C = [[_h(Cin*yin[j]) for j in ST]]+[[X[8+10*k+s] for k in range(6)] for s in range(NR-1)]
    Tg = [Tin]+[X[68+s] for s in range(NR-1)]
    v, fcp, ccp, r = [], [], [], []
    for i in range(NR):
        ct = _h(Pp[i]/8.314/(Tg[i]+TREF)/14.696*101.325)
        c7 = [0.]*7
        for k, j in enumerate(ST):
            c7[j] = C[i][k]
        c7[3] = _h(ct-_s(C[i]))
        y = [_h(c/ct) for c in c7]
        mw = _s(MW[j]*y[j] for j in range(7))
        cp = [(CVA[j]+CVB[j]*Tg[i])*MW[j] for j in range(7)]
        fcp.append(_s(cp[j]*Gm/mw*y[j] for j in range(7)))
        ccp.append(_s(cp[j]*ct*y[j] for j in range(7)))
        v.append(_h(Gm/(ct*mw)))
        r.append(_rates([y[j]*Pp[i] for j in range(7)], Tg[i], act, age) if i else None)
    dC = [[None]*(NR-1) for _ in range(6)]
    dT = []
    for n in range(1, NR):
        for k, j in enumerate(ST):
            dC[k][n-1] = (-(C[n][k]*v[n]-C[n-1][k]*v[n-1])+STO1[j]*r[n][0]+STO2[j]*r[n][1])/EPS
        q = UA_RCT*uar*(Tg[n]-Tsh)
        dT.append((-fcp[n]*(Tg[n]-Tg[n-1])-r[n][0]*E_R1-r[n][1]*E_R2-q)/(EPS*ccp[n]+CP_CAT*RHO_CAT))
    return [q for row in dC for q in row]+dT, Tg


def _separator(X, Feed, z, Tf, F_liq, Tsh, F_gas):
    """Separator.c: flash of the reactor effluent at the jacket temperature + 5 and the vapour space pressure."""
    x = _full(X[78:84], 3)
    L, TL = X[84], X[85]
    yv = _full(X[86:92], 3)
    Pv, TV = X[92], X[93]
    conc = _conc(x)
    vol = L*VW_SEP
    hl, hv, hf = _HL(x, TL), _HV(yv, TV), _HV(z, Tf)
    T = Tsh+5.
    ps = [_ps(i, T) for i in range(7)]
    lam = _lam(T)
    beta, xf = _h(F_gas/Feed), list(x)
    for _ in range(FLASH_OUTER):
        g = _gamma(lam, xf)
        K = [_h(ps[i]*g[i]/Pv) for i in range(7)]
        for _ in range(FLASH_INNER):
            den = [_h(1.+(K[i]-1.)*beta) for i in range(7)]
            beta = _h(beta+_s(z[i]*(K[i]-1.)/den[i] for i in range(7))/_s(z[i]*(K[i]-1.)*(K[i]-1.)/(den[i]*den[i]) for i in range(7)))
        xf = [_h(z[i]/(1.+(K[i]-1.)*beta)) for i in range(7)]
    FV = _h(Feed*beta)
    FLq = _h(Feed-FV)
    yf = [_h((Feed*z[i]-FLq*xf[i])/FV) for i in range(7)]
    hvf = _HV(yf, T)
    Md = _h(FLq-F_liq)
    xd = [(FLq*xf[i]-F_liq*x[i]-x[i]*Md)/conc/vol for i in ST]
    TLd = (Feed*(hf-hl)-FV*(hvf-hl)-UA_SEP*(T-Tsh))/(conc*vol)/_cpl(x, TL)
    vrt = _h(V_LOOP/8.314/(TV+273.15))
    Pd = _h((FV-F_gas)/vrt*14.696/101.325)
    yd = [(FV*yf[i]-F_gas*yv[i]-vrt*yv[i]*Pd)/vrt/(Pv/14.696*101.325) for i in ST]
    TVd = FV*(hvf-hv)/(vrt*Pv)/_cpl(yv, TV)
    return xd+[Md/conc/VW_SEP, TLd]+yd+[Pd, TVd], dict(x=x, TL=TL, y=yv, P=Pv, TV=TV, L=L)


def _absorber(X, Pin, F_liq, Fin, yin, Tin, F_scrub, x_scrub, T_scrub_in, F_circ, Q_circ, Q_scrub, nmt):
    """Absorber.c: eight trays with rate-limited mass transfer and a base; circulation to tray 2, scrub to the top."""
    NT = NT_ABS
    x = [_full([X[94+k*NT+i] for k in range(6)], 3) for i in range(NT)]
    M = X[142:150]
    TL = X[150:158]
    xb = _full(X[158:164], 3)
    Lb, Tb = X[164], X[165]
    conc = _conc(xb)
    volb = Lb*VW_ABS
    T_circ, T_scrub = X[241], X[242]
    HVin, HLb = _HV(yin, Tin), _HL(xb, Tb)
    dTcirc = (_t_liq(xb, (F_circ*HLb-Q_circ)/F_circ)-T_circ)/2.
    HLcirc = _HL(xb, T_circ)
    dTscrub = (_t_liq(x_scrub, (F_scrub*_HL(x_scrub, T_scrub_in)-Q_scrub)/F_scrub)-T_scrub)/2.
    HLscrub = _HL(x_scrub, T_scrub)
    FL = [_h(L0_ABS[i]+(M[i]-M0_ABS)/HYDTAU) for i in range(NT)]
    N, sumN, HN, Q = [], [], [], []
    Fv, yv, Tv, Hv = Fin, yin, Tin, HVin
    for i in range(NT):
        g = _gamma(_lam(TL[i]), x[i])
        yi = [_ps(j, TL[i])*x[i][j]*g[j]/Pin for j in range(7)]
        n = [_h(clamp(nmt*NMT*(yv[j]-yi[j]), -.5*Fv*yv[j], .5*Fv*yv[j])) if isinstance(yv[j], E) or yv[j] else 0. for j in range(7)]
        sn = _s(n)
        hn = _s(n[j]*_hv(j, TL[i]) for j in range(7) if isinstance(n[j], E))   # sumN H_N of the code
        q = _h(QMT[i]*(Tv-TL[i])*9/5)
        F2 = _h(Fv-sn)
        y2 = [_h((Fv*yv[j]-n[j])/F2) for j in range(7)]
        T2 = _h(_t_gas(y2, (Fv*Hv-hn-q)/F2))
        Fv, yv, Tv, Hv = F2, y2, T2, _HV(y2, T2)
        N.append(n); sumN.append(sn); HN.append(hn); Q.append(q)
    HL = [_HL(x[i], TL[i]) for i in range(NT)]
    Md_b = _h(FL[0]-F_liq-F_circ)
    dxb = [(FL[0]*x[0][j]-(F_liq+F_circ)*xb[j]-xb[j]*Md_b)/conc/volb for j in ST]
    dTb = FL[0]*(HL[0]-HLb)/(conc*volb)/_cpl(xb, Tb)
    dx, dM, dT = [[None]*NT for _ in range(6)], [], []
    for i in range(NT):
        if i == NT-1:
            md = _h(F_scrub+sumN[i]-FL[i])
            comp = [F_scrub*x_scrub[j]+N[i][j]-FL[i]*x[i][j] for j in range(7)]
            heat = F_scrub*(HLscrub-HL[i])
        else:
            md = FL[i+1]+sumN[i]-FL[i]
            comp = [FL[i+1]*x[i+1][j]+N[i][j]-FL[i]*x[i][j] for j in range(7)]
            heat = FL[i+1]*(HL[i+1]-HL[i])
            if i == NF_ABS-1:
                md = md+F_circ
                comp = [comp[j]+F_circ*xb[j] for j in range(7)]
                heat = heat+F_circ*(HLcirc-HL[i])
            md = _h(md)
        for k, j in enumerate(ST):
            dx[k][i] = (comp[j]-x[i][j]*md)/M[i]
        dM.append(md)
        dT.append((heat+HN[i]-sumN[i]*HL[i]+Q[i])/M[i]/_cpl(x[i], TL[i]))
    der = [q for row in dx for q in row]+dM+dT+dxb+[Md_b/conc/VW_ABS, dTb]
    return der, dict(F=Fv, y=yv, T=Tv, L=Lb, dTcirc=dTcirc, dTscrub=dTscrub)


def _fehe(F_c, y_c, T_c, F_h, y_h, T_h, X, uaf):
    """FEHE.c: effectiveness of a counter-current exchanger, outlet temperatures lagged by 5 min."""
    mw_h, mw_c = _s(y_h[i]*MW[i] for i in range(7)), _s(y_c[i]*MW[i] for i in range(7))
    ua = uaf*UA_FEHE*(((F_h*mw_h/REF_HOT)**.8)+((F_c*mw_c/REF_COLD)**.8))/2.
    Cc = _h(F_c*_s(y_c[i]*(CVA[i]+CVB[i]*T_c)*MW[i] for i in range(7) if isinstance(y_c[i], E) or y_c[i]))
    Ch = _h(F_h*_s(y_h[i]*(CVA[i]+CVB[i]*T_h)*MW[i] for i in range(7) if isinstance(y_h[i], E) or y_h[i]))
    ntu, cr = _h(ua/Cc), _h(Cc/Ch)
    ee = _h(_ex(-ntu*(1.-cr)))
    eff = (1.-ee)/(1.-cr*ee)
    Tco = _h(eff*(T_h-T_c)+T_c)
    Tho = T_h-cr*(Tco-T_c)
    return (Tco-X[244])/5., (Tho-X[245])/5.


def _column(X, FL_in, xF, TF, LR, QR, LO, LA, LB, F_HAc, F_scrub, F_vl, Q_cond):
    """Column.c: twenty trays (bubble points from the code's guesses), reboiler, decanter, HAc tank."""
    NT, NF = NT_COL, NF_COL
    I3 = (4, 5, 6)

    def x3(v, h):
        x = [0.]*7
        x[4], x[6] = v, h
        x[5] = _h(1.-v-h)
        return x
    x = [x3(X[166+i], X[186+i]) for i in range(NT)]
    M = X[206:226]
    xb, xo, xa = x3(X[226], X[227]), x3(X[229], X[230]), x3(X[232], X[233])
    xt = x3(X[235], X[236])
    Lt, Tt = X[237], X[238]
    Tdec = X[243]
    cb, co, ca, ct = _conc(xb, I3), _conc(xo, I3), _conc(xa, I3), _conc(xt)
    volb, volo, vola, volt = X[228]*VW_BOT, X[231]*VW_ORG, X[234]*VW_AQ, Lt*VW_TANK
    HLF = _HL(xF, TF, I3)
    HLO = _HL(xo, Tdec, I3)
    TB, yB = _bubble(130., P_BOT, xb)
    HLB, HVB = _HL(xb, TB, I3), _HV(yB, TB, I3)
    T, y, HL, HV = [], [], [], []
    for i in range(NT):
        t, yy = _bubble(float(T_GUESS[i]), 29.4-.6*i, x[i])
        T.append(t); y.append(yy); HL.append(_HL(x[i], t, I3)); HV.append(_HV(yy, t, I3))
    HLt = _HL(xt, Tt)
    L = [_h(L0_COL[i]+(M[i]-M0_COL)/HYDTAU) for i in range(NT)]
    VB = _h((QR+L[0]*HL[0]-LB*HLB)/HVB)
    V = [_h((HL[1]*L[1]+HVB*VB-HL[0]*L[0])/HV[0])]
    for n in range(1, NT-1):
        V.append(_h((HL[n+1]*L[n+1]+HV[n-1]*V[n-1]+(HLF*FL_in if n == NF-1 else 0.)-HL[n]*L[n])/HV[n]))
    V.append(_h((HLO*LR+HV[NT-2]*V[NT-2]-HL[NT-1]*L[NT-1])/HV[NT-1]))
    yt = y[NT-1]
    Vt = V[NT-1]
    dTdec = (_t_liq(yt, (Vt*_HV(yt, T[NT-1], I3)-Q_cond)/Vt, I3)-Tdec)/2.
    # decanter: the split of the code's Rachford-Rice iteration and its linear system, in closed form
    a, b = K_DEC[0]-1., K_DEC[1]-1.
    beta = _h((a*yt[4]+b*yt[5])/(-a*b*(yt[4]+yt[5])))
    aq = [0.]*7
    org = [0.]*7
    for k, j in enumerate(I3):
        aq[j] = _h(yt[j]/(1.+beta*(K_DEC[k]-1.)))
        org[j] = K_DEC[k]*aq[j]
    F_org = _h(beta*Vt)
    F_aq = _h(Vt-F_org)
    dx, dM = [[None]*NT, [None]*NT], [None]*NT
    for i in range(NT):
        up = (L[i+1], x[i+1]) if i < NT-1 else (LR, xo)
        dn = (VB, yB) if i == 0 else (V[i-1], y[i-1])
        md = _h(up[0]-L[i]+dn[0]-V[i]+(FL_in if i == NF-1 else 0.))
        for k, j in enumerate((4, 6)):
            mx = up[0]*up[1][j]-L[i]*x[i][j]+dn[0]*dn[1][j]-V[i]*y[i][j]+(FL_in*xF[j] if i == NF-1 else 0.)
            dx[k][i] = (mx-x[i][j]*md)/M[i]
        dM[i] = md
    mb = _h(L[0]-LB-VB)
    mo = _h(F_org-LR-LO)
    ma = _h(F_aq-LA)
    bot = [(L[0]*x[0][j]-LB*xb[j]-VB*yB[j]-xb[j]*mb)/cb/volb for j in (4, 6)]
    orgd = [(F_org*org[j]-(LR+LO)*xo[j]-xo[j]*mo)/co/volo for j in (4, 6)]
    aqd = [(F_aq*aq[j]-LA*xa[j]-xa[j]*ma)/ca/vola for j in (4, 6)]
    # HAc tank: fresh HAc and the column bottoms mixed (the mixer's enthalpy balance)
    Fin = _h(F_HAc+LB)
    xin = [_h((F_HAc*X_HAC[j]+LB*xb[j])/Fin) if j in I3 else 0. for j in range(7)]
    Hin = _h((F_HAc*_HL(X_HAC, T_FEED)+LB*_HL(xb, TB, I3))/Fin)
    out = F_scrub+F_vl
    mt = _h(Fin-out)
    tank = [(Fin*xin[j]-out*xt[j]-xt[j]*mt)/ct/volt for j in (4, 6)]
    dTt = Fin*(Hin-HLt)/(ct*volt)/_cpl(xt, Tt)
    der = dx[0]+dx[1]+dM+bot+[mb/cb/VW_BOT]+orgd+[mo/co/VW_ORG]+aqd+[ma/ca/VW_AQ]+tank+[mt/ct/VW_TANK, dTt]
    return der, dict(T=T, xo=xo, xb=xb, xt=xt, Lt=Lt, Tt=Tt, dTdec=dTdec, Lb=X[228], Lo=X[231], La=X[234], F_org=F_org)


def _process(P, u, L, clock, X=None):
    """The flowsheet of VAModel.c. u: the 26 MVs (expressions or numbers); L: loads and parameters by name;
    clock: a 247th state, the catalyst's age in min (the code's time). Declares the states and their equations;
    returns what outputs are made of."""
    X = _states(P) if X is None else X
    age = P.state('age', 0., scale=60.) if clock else 0.
    (F_O2, F_C2H4, F_HAc, Q_vap, F_vap, Q_heater, T_shell, F_sepliq, T_sepshell, F_sepvap, Q_comp, F_absliq, F_circ, Q_circ, F_scrub,
     Q_scrub, F_CO2, F_purge, bypass, F_reflux, Q_reb, Q_cond, F_org, F_aq, F_bot, F_vl) = u
    F_O2, F_HAc = F_O2*L['o2_avail'], F_HAc*L['hac_avail']
    y_c2h4 = (0., 0., 1.-L['xC2H6'], L['xC2H6'], 0., 0., 0.)
    T_heater, T_comp, T_circ, T_scrub, T_dec, T_cold, T_hot = X[239:246]
    # separator liquid and absorber base to the gas removal; its liquid is the column feed
    xs, xb = _full(X[78:84], 3), _full(X[158:164], 3)
    Fgr, xgr, Tgr = _mix(F_sepliq, xs, X[85], F_absliq, xb, X[165], gas=False)
    F_gas = _h(Fgr*_s(xgr[i] for i in range(4)))
    y_gas = [_h(Fgr*xgr[i]/F_gas) if i < 4 else 0. for i in range(7)]
    F_col = _h(Fgr*_s(xgr[i] for i in range(4, 7)))
    x_col = [0.]*4+[_h(Fgr*xgr[i]/F_col) for i in range(4, 7)]
    F_col = F_col*L['colfeed']
    # separator vapour and the removed gas to the compressor
    ys = _full(X[86:92], 3)
    Fc, yc, _ = _mix(F_sepvap, ys, X[93], F_gas, y_gas, Tgr, gas=True)
    TV, Ps = X[93], X[92]
    P_rec = _h(Ps/14.696*101.325*_s(yc[i]*MW[i] for i in range(7))/8.314/(273.16+TV)*2.20462/(3.2808**3)/144*COMP_COEF+Ps)
    T_cco = _h(P_rec/Ps*(TV+273.15)-273.15)
    d_comp = (_t_gas(yc, (Fc*_HV(yc, T_cco)-Q_comp)/Fc)-T_comp)/2.
    # absorber
    xt = [0.]*4+[X[235], _h(1.-X[235]-X[236]), X[236]]
    d_abs, ab = _absorber(X, P_rec, F_absliq, Fc, yc, T_comp, F_scrub, xt, X[238], F_circ, Q_circ, Q_scrub, L['nmt'])
    # CO2 removal, purge, fresh ethylene, bypass split
    ya, Fa = ab['y'], ab['F']
    eff = minimum1(.995-3.14e-3*(F_CO2-CO2_F_REF)-32.5*(ya[1]-CO2_X_REF))
    F_co2p = _h(F_CO2*ya[1]*eff)
    F_ex = _h(F_CO2-F_co2p)
    y_pg = [_h((F_CO2*ya[i]-(F_co2p if i == 1 else 0.))/F_ex) for i in range(7)]
    F_gr = _h(Fa-F_co2p-F_purge)
    y_gr = [_h((Fa*ya[i]-(F_co2p if i == 1 else 0.)-F_purge*y_pg[i])/F_gr) for i in range(7)]
    Fm, ym, Tm = _mix(F_gr, y_gr, ab['T'], F_C2H4, y_c2h4, T_FEED, gas=True)
    F_tovap = _h(Fm*bypass)
    F_cold = _h(Fm-F_tovap)
    # vaporizer vapour, heater, O2: the reactor feed
    xv = _full(X[0:6], 3)
    gv = _gamma(_lam(X[7]), xv)
    kv = [_h(xv[i]*_ps(i, X[7])*gv[i]) for i in range(7)]
    P_vap = _s(kv)
    yv = [_h(v/P_vap) for v in kv]
    d_heat = (_t_gas(yv, (F_vap*_HV(yv, X[7])+Q_heater)/F_vap)-T_heater)/2.
    Fr, yr, Tr = _mix(F_vap, yv, T_heater, F_O2, Y_O2, T_FEED, gas=True)
    pk = P_vap/14.696*101.325
    Cr = _h(pk/(8.314*(Tr+TREF)))
    G = _h(Fr*_s(MW[i]*yr[i] for i in range(7)))
    Vr = _h(Fr*8.314*(Tr+273.15)/pk)
    dP = _h(FRICTION*(G/Vr)*Vr*Vr)
    # reactor and exchanger
    d_rct, Tg = _reactor(X, Fr, yr, Tr, P_vap, Cr, dP, T_shell, L['act'], age, L['uar'])
    Ch = _h((P_vap-dP)/14.696*101.325/(8.314*(Tg[-1]+TREF)))
    yh = [0.]*7
    for k, j in enumerate(ST):
        yh[j] = _h(X[8+10*k+NR-2]/Ch)
    yh[3] = _h(1.-_s(yh))
    Fh = _h(G/_s(MW[i]*yh[i] for i in range(7)))
    d_cold, d_hot = _fehe(F_cold, ym, Tm, Fh, yh, Tg[-1], X, L['uafehe'])
    Fvv, yvv, Tvv = _mix(F_cold, ym, T_cold, F_tovap, ym, Tm, gas=True)
    # units
    d_vap, _ = _vaporizer(X, F_vap, Q_vap, Fvv, yvv, Tvv, F_vl, xt, X[238])
    d_sep, _ = _separator(X, Fh, yh, T_hot, F_sepliq, T_sepshell, F_sepvap)
    d_col, co = _column(X, F_col, x_col, Tgr, F_reflux, Q_reb, F_org, F_aq, F_bot, F_HAc, F_scrub, F_vl, Q_cond)
    der = d_vap+d_rct+d_sep+d_abs+d_col+[d_heat, d_comp, ab['dTcirc'], ab['dTscrub'], co['dTdec'], d_cold, d_hot]
    for x, d in zip(X, der):
        P.der(x, d)
    if clock:
        P.der(age, 1.)
    meas = [P_vap, X[6], X[7], T_heater, Tg[-1], Fh, T_cold, T_hot, X[84], X[85], T_comp, P_rec, X[164], T_circ, T_scrub, F_tovap+F_cold, F_org,
            X[231], X[234], T_dec, X[228], co['T'][4], X[237]]+[co['xo'][j] for j in (4, 5, 6)]+[co['xb'][j] for j in (4, 5, 6)]+list(ym)+list(yr)
    peak = Tg[1]
    for t in Tg[2:]:
        peak = maximum(peak, t)
    return dict(X=X, meas=meas, peak=peak, co2=F_co2p, prod=F_org*co['xo'][4])


def minimum1(v):
    """min(v, 1) (ComponentSeparator.c)."""
    return v-relu(v-1.)


# the MVs' ranges (test_VAcPlant.m's controller ranges; the fixed flows at about three times their base)
RANGE = ((0., 2.268), (0., 7.56), (0., 4.536), (0., 1433400.), (0., 50.), (0., 15000.), (110., 150.), (0., 4.536), (0., 80.), (0., 50.),
         (0., 50000.), (0., 4.536), (0., 50.), (0., 30000.), (0., 2.4), (0., 5000.), (0., 22.68), (0., .02268), (0., 1.), (0., 7.56),
         (0., 100000.), (0., 150000.), (0., 2.4), (0., 2.4), (0., 4.536), (0., 4.536))


def _inputs(P):
    """Loads (the paper's disturbances: C2H6 in the ethylene feed, column feed, O2 and HAc feeds lost) and parameters."""
    L = dict(xC2H6=P.load('xC2H6', .001, .001, .005), colfeed=P.load('colfeed', 1., 0., 1.),
             o2_avail=P.load('o2_avail', 1., 0., 1.), hac_avail=P.load('hac_avail', 1., 0., 1.))
    for q, (lo, hi) in dict(act=(.8, 1.), uar=(.9, 1.05), nmt=(.85, 1.1), uafehe=(.85, 1.05)).items():
        L[q] = P.param(q, 1., lo, hi)
    return L


def vam_raw(dt=60., substeps=4):
    """The open-loop process: the 26 MVs of the code, the 43 measurements, the peak reactor temperature. A clock
    (state 247, min) ages the catalyst as the code's time does. Unstable (levels integrate); for comparison with
    the native code and for the tools, not for episodes."""
    P = Plant('vam_raw', dt=dt, time_unit=60., substeps=substeps)
    u = [P.mv(nm, lo, hi, (hi-lo)/20., U0[i]) for i, (nm, (lo, hi)) in enumerate(zip(MVS_RAW, RANGE))]
    q = _process(P, u, _inputs(P), clock=True)
    for nm, v in zip(MEAS, q['meas']):
        P.out(nm, v, range=(0., 1.), band=1.)
    P.out('T_peak', q['peak'], range=(0., 1.), band=1.)
    return P


_BASE = {}


def base():
    """The 43 measurements, the peak temperature, the product and the CO2 removed at the code's base (float64)."""
    if not _BASE:
        P = vam_raw()
        M = Mech([P], n=1, dtype=torch.float64)
        x, w, p = P.nominal()
        y = M.y(x, w, p)[0]
        _BASE.update(meas=[float(v) for v in y[:43]], peak=float(y[43]))
    return _BASE


# ---------------------------------------------------------------- the plant an APC sees
# Luyben's regulatory layer (the paper's Appendix 2, tuned as in test_VAcPlant.m): name, MV of the code (index),
# measurement (index of the 43, or an analyser), setpoint (an APC input by name, or a number), gain (scaled, as the
# code's controller), reset time (min; 0: proportional only), action, measurement range, MV range.
LOOPS = (('O2', 0, 36, 'O2', 5., 10., 1, (0., .2), (0., 2.268)),
         ('Prec', 1, 11, 'Prec', .3, 20., 1, (0., 200.), (0., 7.56)),
         ('Ltank', 2, 22, .5, 2., 0., 1, (0., 1.), (0., 4.536)),
         ('Lvap', 3, 1, .7, .1, 30., -1, (0., 1.), (0., 1433400.)),
         ('Pvap', 4, 0, 'P_rec', 5., 10., -1, (0., 200.), (0., 50.)),
         ('Theater', 5, 3, 150., 1., 5., 1, (120., 170.), (0., 15000.)),
         ('Trct', 6, 4, 'Trct', 3., 10., 1, (0., 200.), (110., 150.)),
         ('Lsep', 7, 8, .5, 5., 0., -1, (0., 1.), (0., 4.536)),
         ('Tsep', 8, 9, 'Tsep', 5., 20., -1, (0., 80.), (0., 80.)),
         ('Tcomp', 10, 10, 80., 1., 5., -1, (70., 90.), (0., 50000.)),
         ('Labs', 11, 12, .5, 5., 0., -1, (0., 1.), (0., 4.536)),
         ('Tcirc', 13, 13, 25., 1., 5., -1, (10., 40.), (0., 30000.)),
         ('Tscrub', 15, 14, 25., 1., 5., -1, (10., 40.), (0., 5000.)),
         ('CO2', 16, 30, 'CO2', 1., 0., -1, (0., .5), (0., 22.68)),
         ('C2H6', 17, 32, 'C2H6', 1., 0., -1, (0., 1.), (0., .02268)),
         ('Tfehe', 18, 7, 'Tfehe', 5., 10., 1, (0., 200.), (0., 1.)),
         ('H2Ob', 19, 27, 'H2Ob', .5, 60., -1, (0., .2), (0., 7.56)),
         ('Ttray', 20, 21, 'Tcol', 20., 30., 1, (0., 120.), (0., 100000.)),
         ('Tdec', 21, 19, 45.845, 1., 5., -1, (40., 50.), (0., 150000.)),
         ('Lorg', 22, 17, .5, 1., 0., -1, (0., 1.), (0., 2.4)),
         ('Laq', 23, 18, .5, 1., 0., -1, (0., 1.), (0., 2.4)),
         ('Lbot', 24, 20, .5, 1., 0., -1, (0., 1.), (0., 4.536)))
FIXED = {9: 'Fgas', 12: 'Fcirc', 14: 'Fscrub', 25: 'Fvl'}               # flows the layer leaves where they are set
ANALYSED = (30, 32, 27)                                                  # CO2 and C2H6 in the gas recycle, H2O in the column bottoms
AN_STAGES, AN_DEAD, LAG = 10, 10., .05                                   # analyser dead time (min) as ten first-order stages; transmitter lag (min)
# what an APC may move: name -> (low, high, largest move per decision, nominal)
APC = dict(Fvl=(1.7, 2.7, .05, 2.1924), Trct=(150., 168., .5, 159.17), O2=(.05, .08, .002, .075), Prec=(115., 138., 1., 128.),
           Fgas=(13., 19., .2, 16.1026), H2Ob=(.05, .18, .005, .09344), Tcol=(105., 114., .5, 110.), C2H6=(.15, .35, .01, .25),
           CO2=(.004, .015, .0005, .0076393), Tsep=(30., 50., .5, 40.), Tfehe=(130., 145., .5, 134.), Fcirc=(10., 20., .3, 15.1198),
           Fscrub=(.5, 1., .02, .756))
MVS = ('Trct', 'O2', 'Prec', 'H2Ob', 'Tcol', 'C2H6')                    # product and CO2 made are free rows: reactor temperature and pressure hold them
MVS_ALL = tuple(q for q in APC if q not in ('Fcirc', 'Fscrub'))           # every setpoint with a tracked output of its own


def _ctl(P, name, y, sp, K, Ti, act, yr, ur, u0, y0, sp0):
    """The code's controller (controller.c) in continuous time: scaled error, full-position PI (or P) with the bias
    that starts it at u0, output limited to its range, integral held while the output is at a limit and the error
    pushes further. y is the transmitted measurement, already limited to its range."""
    ylo, yhi = yr
    ulo, uhi = ur
    kk = K*act
    e = (sp-y)/(yhi-ylo)
    b = (u0-ulo)/(uhi-ulo)-kk*(sp0-y0)/(yhi-ylo)
    if Ti:
        I = P.state(f'I_{name}', 0., scale=.01)
        us = b+kk*(e+I/Ti)
        P.der(I, e*(1.-step(us-1.)*step(kk*e)-step(-us)*step(-kk*e)))
    else:
        us = b+kk*e
    return ulo+(uhi-ulo)*clamp(us, 0., 1.)


def vam(mvs=MVS, name='vam'):
    """The vinyl acetate plant under Luyben's regulatory layer. mvs: which of `APC` are MVs (the others stay at their
    nominal values). Time in min; a decision is 10 min, the analysers' cycle.

    States: the 246 of the process, then the transmitters (a 3 s lag each, as in test_VAcPlant.m), the analysers'
    dead time (ten stages each) and the integrals of the PI loops. Outputs: VAc product (organic product times its
    VAc fraction, kmol/min) and CO2 made (the yield loss), both tracked, the peak reactor temperature (interlock 200 C), O2 at the
    reactor inlet (8 %), recycle pressure (140 psia), reactor feed and FEHE hot exit temperatures (above 130 C), the
    product's impurities, the analyser readings and the levels."""
    b = base()['meas']
    P = Plant(name, dt=600., time_unit=60., substeps=2)                 # ROS2 with the accuracy guard: 0.4 band of ten substeps with a Jacobian at each, six times faster
    sp = {}
    for q, (lo, hi, slew, n0) in APC.items():
        sp[q] = P.mv(q, lo, hi, slew, n0) if q in mvs else n0
    L = _inputs(P)
    X = _states(P)
    tx, an = {}, {}
    for nm, k, j, *_r, yr, ur in LOOPS:
        tx[nm] = P.state(f'tx_{nm}', min(max(b[j], yr[0]), yr[1]), scale=max(abs(b[j]), 1e-3))
    for j in ANALYSED:
        an[j] = [P.state(f'an{j}_{s+1}', b[j], scale=max(abs(b[j]), 1e-4)) for s in range(AN_STAGES)]
    u = [None]*26
    for i, q in FIXED.items():
        u[i] = sp[q]
    for nm, k, j, spq, K, Ti, act, yr, ur in LOOPS:
        if spq == 'P_rec':                                       # vaporizer pressure follows the recycle pressure as transmitted
            s, s0 = tx['Prec'], b[11]
        elif isinstance(spq, str):
            s, s0 = sp[spq], APC[spq][3]
        else:
            s, s0 = spq, spq
        u[k] = _ctl(P, nm, tx[nm], s, K, Ti, act, yr, ur, U0[k], min(max(b[j], yr[0]), yr[1]), s0)
    q = _process(P, u, L, clock=False, X=X)
    m = q['meas']
    for j in ANALYSED:
        z = m[j]
        for zs in an[j]:
            P.der(zs, (z-zs)/(AN_DEAD/AN_STAGES))
            z = zs
    for nm, k, j, *_r, yr, ur in LOOPS:
        P.der(tx[nm], (clamp(an[j][-1] if j in an else m[j], yr[0], yr[1])-tx[nm])/LAG)
    mv = lambda k: k in mvs                                                              # noqa: E731
    P.out('prod', q['prod'], range=(.5, 1.), band=.005, track=True, target=(.70, .81))
    P.out('co2', q['co2'], range=(0., .15), band=.002, track=True, target=(.068, .098))
    P.out('Tpeak', q['peak'], range=(140., 210.), band=.5, hi=200.)
    P.out('O2', m[36], range=(0., .1), band=.001, track=mv('O2'), target=(.055, .078), hi=.08)
    P.out('Prec', m[11], range=(100., 145.), band=.5, hi=140.)
    P.out('Trct', m[4], range=(140., 175.), band=.3)
    P.out('Tfeed', m[3], range=(120., 170.), band=.5, lo=130.)
    P.out('Tfehe', m[7], range=(120., 150.), band=.5, track=mv('Tfehe'), target=(131., 143.), lo=130.)
    P.out('Tsep', m[9], range=(25., 55.), band=.3, track=mv('Tsep'), target=(32., 48.))
    P.out('xHAc_org', m[25], range=(0., .002), band=2e-5, hi=6e-4)
    P.out('xVAc_bot', m[26], range=(0., .0005), band=5e-6, hi=1e-4)
    P.out('H2Ob', an[27][-1], range=(0., .2), band=.002, track=mv('H2Ob'), target=(.06, .16))
    P.out('C2H6', an[32][-1], range=(0., .5), band=.005, track=mv('C2H6'), target=(.17, .33))
    P.out('CO2', an[30][-1], range=(0., .03), band=2e-4, track=mv('CO2'), target=(.005, .013))
    P.out('Ttray5', m[21], range=(90., 120.), band=.3, track=mv('Tcol'), target=(106., 113.))
    for nm, j in (('Lvap', 1), ('Lsep', 8), ('Labs', 12), ('Lorg', 17), ('Laq', 18), ('Lbot', 20), ('Ltank', 22)):
        P.out(nm, m[j], range=(0., 1.), band=.02, lo=.1, hi=.9)
    return P


def vam_all():
    """Eleven setpoints of `APC` as MVs: for the comparison with the native code (aio/sim/validate/vam.py), not for episodes
    (HAc recycle and recycle gas flow have no tracked row of their own)."""
    return vam(MVS_ALL, name='vam_all')

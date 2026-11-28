"""A two-pass crude heater with natural draft as data for the mechanism engine (docs/sim.md, 'Fired heater').

The heater 21 H-1 of the crude unit at Homs, whose data sheet is printed in Ibrahim and Al-Qassimi, "Simulation
of heat transfer in the convection section of fired process heaters", Periodica Polytechnica Chem. Eng. 54(1),
2010, 33 (and worked through in their paper of 2008, 52(2), 61): a cabin of 20 m, 60 radiant tubes of 219 mm
along the walls and 40 of 168 mm in the convection bank above (the lowest 8 are the shield), two passes,
225.7 t/h of heavy crude from 210 through 250 to 355 C, 23.6 MW absorbed at 75.78 %, natural gas with 25 %
excess air, 298 t of refractory. What it adds to the plants of the engine: heat put in along the whole way of
the oil (the outlet answers the fuel as a ramp over the eleven minutes the oil takes, and the inlet after a
dead time), a fire side that answers in seconds beside it, a slow tail from the walls, and the limits a heater
is run against (tube skin, bridgewall, draft, burner pressure).

Built on the data sheet: the tubes, the flows, the oil's three temperatures and two duties (its enthalpy is
taken from them: 13 % under the API correlation at this gravity), the heat release, the excess air and the
5 % lost. The fire is by the composition: heats of formation and ideal gas heat capacities of the species.
The paper's heating value (927,844 kJ/kmol) is 5 % above what its composition gives (883,381), so the same heat
release takes 126 and not 120 kmol/h of fuel and makes 5 % more flue gas; its air and its flue gas analysis
follow from the composition to the last digit, and its two gas temperatures (800 and 400 C) follow from its
heating value (791 and 408 C with these enthalpies).

Model: each pass is cells of oil and of tube wall, in the order the oil passes them (bank, shield, radiant).
Radiant section as Lobo and Evans: gas at one temperature (a state), sigma alpha Acp F (Tg^4 - Tw^4) to the
cold plane of every cell and 2 Btu/h ft2 F of convection, the shield in the same radiation. Bank: the gas
row by row, what leaves the rows of a cell a fast state of its own (the gas held between them) that goes to the
outlet past a wall at one temperature, outside coefficient as the paper has it (Monrad plus gas radiation, plus 10 %),
inside Sieder and Tate with the viscosity through the data sheet's one point. The walls lose the 5 % through
one lump of the refractory's weight. Flows are resistances between pressures: each pass a valve and its
coil's friction between the charge header and the column; the fuel a valve (IEC 60534, compressible) into
the burner header (a state) and the tips (a nozzle, choked at the design point) into the box; the air
through the registers by the draft of box, bank and stack against bank, stack and damper.

Fitted, to the data sheet: the exchange factor (0.684, to the oil leaving the bank at 250 C) and a factor on
the bank's outside coefficient (0.879, to 355 C at the printed heat release). The bridgewall is then at 720 C
and the stack at 391 C. Not on the data sheet and taken here: fouling inside the tubes, the height of the
box and the stack, the pressures of the two headers, the burner pressure, an arch draft of 25 Pa with the
damper half open, the valves' travel, the ranges of the loads and the limits. The stack's lining takes no
heat; the oil's expansion does not add to the flow; the exchange factor does not follow the gas.

Dynamics have no outside reference: no step test of a crude heater was found in the open literature.
tests/test_htr.py holds the plant against its own equations written again in numpy (and against one cell a
tube), not against a heater.
"""
import math

from aio.sim import flow as FL
from aio.sim.engine import Plant
from aio.sim.expr import exp, log, sqrt, relu, minimum, maximum, clamp

# ---------------------------------------------------------------- the data sheet
OIL = 225700./3600.                                   # kg/s, both passes
TIN, TX, TOUT = 210., 250., 355.                      # the oil in, leaving the bank, out: C
QRAD, QCONV = 6.182e7/3600., 2.30e7/3600.             # kW
DUTY, EFF, LOSS, EXCESS, HUM = QRAD+QCONV, .7578, .05, .25, .015
RELEASE = DUTY/EFF                                    # 1.1193e8 kJ/h as printed
SG, KW = .9148, 12.1                                  # the crude: specific gravity, UOP K
LEFF, LTOT = 20.024, 20.824                           # tube: in the box, overall, m
RAD = dict(n=30, do=.219, di=.203, pitch=.394)        # tubes a pass
CNV = dict(n=20, do=.168, di=.152, pitch=.250, shield=4, row=4)
POUT, REFR = 910., 298000.                            # kPa at the outlet; kg of refractory
X = dict(c1=.8043, c2=.0902, c3=.0454, ic4=.0020, nc4=.0032, ic5=.0004, nc5=.0002, co2=.0352, h2s=.0009, n2=.0182)   # the fuel (n2: 0.01735 printed, and the 0.085 % the list lacks)
NCV_PRINTED = 927844.41                               # kJ/kmol
# ---------------------------------------------------------------- the species (thermo 0.6.1)
LHV_I = dict(c1=802.6e3, c2=1428.6e3, c3=2043.3e3, ic4=2647.6e3, nc4=2657.2e3, ic5=3264.7e3, nc5=3271.4e3, h2s=518.0e3)     # kJ/kmol to CO2, SO2 and steam
O2_I = dict(c1=2., c2=3.5, c3=5., ic4=6.5, nc4=6.5, ic5=8., nc5=8., h2s=1.5)
C_I = dict(c1=1, c2=2, c3=3, ic4=4, nc4=4, ic5=5, nc5=5, h2s=1)                    # carbon dioxide made (the sulfur dioxide with it)
H_I = dict(c1=2, c2=3, c3=4, ic4=5, nc4=5, ic5=6, nc5=6, h2s=1)                    # water made
MW_I = dict(c1=16.042, c2=30.069, c3=44.096, ic4=58.122, nc4=58.122, ic5=72.149, nc5=72.149, h2s=34.081, co2=44.010, n2=28.013)
# sensible enthalpy above 15 C, kJ/kmol, cubic in (T - 15): within 0.4 % of the integral of cp to 1200 C
HN2 = (-2.3104e-07, 0.00326382, 28.3044, 40.8894)
HO2 = (-1.53904e-06, 0.00584925, 28.9047, 8.34445)
HCO2 = (-4.6861e-06, 0.0164422, 38.1943, -72.9126)
HH2O = (2.79153e-07, 0.00555028, 32.8724, 30.5243)
HC1 = (-1.11226e-07, 0.0292641, 34.2175, 21.1136)
HC3 = (-4.92795e-06, 0.0805236, 77.8756, -348.059)
MWS = (44.010, 18.015, 31.999, 28.013)                # CO2, H2O, O2, N2
MWA = (1-HUM)*(.21*31.999+.79*28.013)+HUM*18.015      # the air, moist
SIG, RG, PATM, GRAV = 5.670374e-11, 8.314, 101.325, 9.80665      # kW/m2 K4, kJ/kmol K, kPa, m/s2
# ---------------------------------------------------------------- taken (not on the data sheet)
HRC = 11.36e-3                                        # kW/m2 K: convection in the radiant section (Lobo and Evans: 2 Btu/h ft2 F)
ALPHA = .915                                          # Hottel: one row against a wall at 1.8 diameters
KMET, CPMET, RHOMET = 19e-3, .55, 7900.               # AISI 304 at 350 C: kW/m K, kJ/kg K, kg/m3
RFOUL, KCOKE = .5, 3e-3                               # m2 K/kW inside the tubes (TEMA, crude above 260 C: 0.5 to 1.2); coke, kW/m K
FRIC, BEND = .0165, 50.                               # Darcy factor (commercial steel at Re 2.5e5); a return bend, in diameters
PIN = 1400.                                           # kPa before the pass valves
PH, PB0, VB, TFUEL, GAM, XT = 450., 250., .5, 25., 1.3, .65      # fuel header and burners kPa abs, burner header m3, fuel C, cp/cv, the valve's choking ratio
RC = (2/(GAM+1))**(GAM/(GAM-1))                       # pressure ratio where the tips choke
HFB, HCV, HST, DST = 11.3, 2.5, 20., 2.2              # floor to arch, the bank, the stack and its diameter: m
DRAFT0, XD0 = 25., .5                                 # Pa at the arch with the damper at XD0
KSTACK = .02*HST/DST+1.5                              # velocity heads: friction, entry, exit
VFB, TAU_O2, CPREF, TAMB = 900., 20., 1., 25.         # m3 of box; the analyser, s; refractory kJ/kg K; C
# ---------------------------------------------------------------- fitted, and the design point at the fit (tests/test_htr.py finds them again)
FEX, FCV = .684319, .878976                           # exchange factor; on the bank's outside coefficient
TG0, TS0, DP0 = 720.1891, 390.7141, 174.1359          # bridgewall and stack C, coil pressure drop kPa


def poly(c, t):
    tau = t-15.
    return ((c[0]*tau+c[1])*tau+c[2])*tau+c[3]


def fuel(c3=X['c3']):
    """A kmol of the fuel with c3 of propane in place of methane: heating value kJ, oxygen needed, carbon dioxide
    and water made, molar mass, and what stands in for its sensible heat (methane, propane, nitrogen). Numbers
    or expressions (every one is linear in c3)."""
    d = c3-X['c3']
    f = lambda t: sum(X[k]*t[k] for k in t)          # noqa: E731
    return dict(lhv=f(LHV_I)+(LHV_I['c3']-LHV_I['c1'])*d, o2=f(O2_I)+3.*d, co2=f(C_I)+2.*d, h2o=f(H_I)+2.*d, mw=f(MW_I)+(MW_I['c3']-MW_I['c1'])*d,
                c1=X['c1']+X['c2']-d, c3=X['c3']+X['ic4']+X['nc4']+X['ic5']+X['nc5']+d, inert=X['co2']+X['n2']+X['h2s'])


FU0 = fuel()
# ---------------------------------------------------------------- the oil
CP1 = (QRAD/(OIL*(TOUT-TX))-QCONV/(OIL*(TX-TIN)))/(.5*(TOUT-TIN))        # cp = CP0 + CP1 T kJ/kg K, from the two duties
CP0 = QCONV/(OIL*(TX-TIN))-CP1*.5*(TX+TIN)


def hoil(t):
    return CP0*t+(.5*CP1)*t*t


def _rho_api(t):
    """API MPMS 11.1 (crude oils), beyond its 150 C."""
    r15 = SG*999.016
    a, d = 613.9723/r15**2, t-15.
    return r15*math.exp(-a*d*(1+.8*a*d))


RHO_B = (_rho_api(TIN)-_rho_api(TOUT))/(TOUT-TIN)     # a line through the two ends (0.02 % between them)
RHO_A = _rho_api(TIN)+RHO_B*TIN


def rho(t):
    return RHO_A-RHO_B*t


def visc(t):
    """Kinematic viscosity, m2/s: ASTM D341 through the data sheet's 30 cSt at 70 C, with the slope between the two
    viscosities of the API correlation (Abbott) at this gravity and K (which gives 29.1 cSt at 70 C)."""
    api = 141.5/SG-131.5
    l100 = (4.39371-1.94733*KW+.12769*KW**2+3.2629e-4*api**2-1.18246e-2*KW*api
            + (.171617*KW**2+10.9943*api+9.50663e-2*api**2-.860218*KW*api)/(api+50.3642-4.78231*KW))
    l210 = -.463634-.166532*api+5.13447e-4*api**2-8.48995e-3*KW*api+(8.0325e-2*KW+1.24899*api+.19768*api**2)/(api+26.786-2.6296*KW)
    z = lambda v: math.log10(math.log10(v+.7))        # noqa: E731
    t1, t2 = (100-32)/1.8+273.15, (210-32)/1.8+273.15
    slope = (z(10**l100)-z(10**l210))/math.log10(t2/t1)
    return (10**(10**(z(30.)-slope*math.log10((t+273.15)/343.15)))-.7)*1e-6


def film(F, di, t):
    """Sieder and Tate (without the wall's viscosity), kW/m2 K, at the mass flow F of a pass; the oil's conductivity
    is the paper's line."""
    nu, r, k, cp = visc(t), rho(t), (.49744-29.4604e-5*t)/3600., CP0+CP1*t
    v = F/(r*math.pi/4*di*di)
    return .027*(v*di/nu)**.8*(cp*nu*r/k)**(1/3)*k/di


def cells(nc, ns, nr):
    """The cells of a pass in the order the oil passes them: nc in the bank, ns in the shield, nr radiant."""
    out, F0 = [], OIL/2
    acp = (2*RAD['n']*RAD['pitch']*LEFF*ALPHA, CNV['row']*CNV['pitch']*LEFF)      # cold plane of the radiant tubes, of the bank's opening: m2, both passes
    for sec, m, tubes, (ta, tb) in ((0, nc, CNV['n']-CNV['shield'], (TIN, TX-8.)), (1, ns, CNV['shield'], (TX-8., TX)), (2, nr, RAD['n'], (TX, TOUT))):
        g = CNV if sec < 2 else RAD
        do, di, k, tm = g['do'], g['di'], tubes/m, .5*(ta+tb)
        hold = LTOT+math.pi*g['pitch']/2                          # with the return bend
        for j in range(m):
            out.append(dict(sec=sec, do=do, di=di, A=k*math.pi*do*LEFF, V=k*math.pi/4*di*di*hold, Cw=k*math.pi/4*(do*do-di*di)*hold*RHOMET*CPMET,
                            kf=k*FRIC*(LTOT+BEND*di)/(2*di*(math.pi/4*di*di)**2)/1e3,       # kPa per (kg/s)^2 (m3/kg)
                            rw=do*math.log(do/di)/(2*KMET),                              # the wall, m2 K/kW on the outside
                            acp=(0., acp[1]/2/ns, acp[0]/2/nr)[sec],
                            hi0=film(F0, di, tm), hib=math.log(film(F0, di, tm+40.)/film(F0, di, tm-40.))/80., hit=tm,
                            t0=(ta+(tb-ta)*(j+1)/m) if sec != 1 else tb))
    return out


# ---------------------------------------------------------------- sized at the design point
FREE = LEFF*(CNV['row']-1)*(CNV['pitch']-CNV['do'])   # m2 between the tubes of a row
AIR_PER = (1+EXCESS)*FU0['o2']/(.21*(1-HUM))          # kmol of air a kmol of fuel


def hair(t):
    return (1-HUM)*(.21*poly(HO2, t)+.79*poly(HN2, t))+HUM*poly(HH2O, t)


def hfuel(fu, t):
    return fu['c1']*poly(HC1, t)+fu['c3']*poly(HC3, t)+fu['inert']*poly(HN2, t)


NF0 = RELEASE/(FU0['lhv']+hfuel(FU0, TFUEL)+AIR_PER*hair(TAMB))         # kmol/s of fuel: the printed heat release
NA0 = AIR_PER*NF0
_SP = (NF0*FU0['co2']+NF0*X['co2'], NF0*FU0['h2o']+HUM*NA0, .21*(1-HUM)*NA0-NF0*FU0['o2'], .79*(1-HUM)*NA0+NF0*X['n2'])
WG0 = sum(a*b for a, b in zip(_SP, MWS))              # kg/s of flue gas
MWG = WG0/sum(_SP)
UR = 2*LOSS*RELEASE/(TG0-TAMB)                        # kW/K, gas to refractory and refractory to the air
CR = REFR*CPREF
CFB = PATM*VFB/(RG*1013.)*34.                         # the gas in the box, kJ/K
CS = PATM*(math.pi/4*DST*DST*HST+20.)/(RG*673.)*33.   # the gas above the bank
CB = PATM*(LEFF*CNV['row']*CNV['pitch']*HCV-2*CNV['n']*math.pi/4*CNV['do']**2*LEFF)/(RG*850.)*33./(CNV['n']*math.pi*CNV['do']*LEFF)      # the gas between the tubes of the bank, kJ/K per m2 of a pass's tubes
_root = lambda dp, d0=1.: dp/math.sqrt(dp+d0)         # noqa: E731
CVO = OIL/2/(.5*math.sqrt(_root(PIN-POUT)**2-DP0))     # the pass valves half open: the coil takes DP0 of the header's pressure


def _valve(x, ph, pb):
    xr = min(max((ph-pb)/ph, 0.), XT)
    return x*ph*(1-xr/(3*XT))*xr/math.sqrt(xr+1e-4)


def _tips(pb):
    r = max(min(PATM/pb, 1.), RC)
    q = max(r**(2/GAM)-r**((GAM+1)/GAM), 0.)
    return pb*q/math.sqrt(q+1e-6)


CT, CVF = NF0/_tips(PB0), NF0/_valve(.5, PH, PB0)     # the tips at the burner pressure, the fuel valve half open
_RA, _RG = PATM*MWA/(RG*(TAMB+273.15)), lambda t: PATM*MWG/(RG*(t+273.15))       # densities, kg/m3
_BFB = GRAV*HFB*(_RA-_RG(TG0))                        # the box's own draft, Pa
KREG = (DRAFT0+_BFB)*_RA/(MWA*NA0)**2                 # the registers: the arch at DRAFT0
_GM = WG0/FREE
_DPCV = 4*(.25+.118/((CNV['pitch']-CNV['do'])/CNV['do'])**1.08)*(_GM*CNV['do']/3.9e-5)**-.16*(2*CNV['n']/CNV['row'])*_GM**2/(2*_RG(.5*(TG0+TS0)))    # Jakob, staggered rows
KCV = _DPCV*_RG(.5*(TG0+TS0))/WG0**2
KST = KSTACK/(2*(math.pi/4*DST*DST)**2)
KD = (_BFB+GRAV*(HCV*(_RA-_RG(.5*(TG0+TS0)))+HST*(_RA-_RG(TS0)))-(DRAFT0+_BFB)-_DPCV-KST*WG0**2/_RG(TS0))*_RG(TS0)/WG0**2*XD0**2    # the damper takes what is left


def heater(nc=4, ns=1, nr=10):
    """The outlet temperature, the feed, the balance of the two passes and the oxygen by the two pass valves, the
    fuel valve and the stack damper; tube skins, bridgewall, arch draft and burner pressure under limits. Loads:
    the oil's inlet temperature and the fuel header pressure (measured), the fuel's propane and the air's
    temperature. Time in s, temperatures in C, the oil in kg/s, gas in kmol/s, pressures in kPa (the draft in Pa)."""
    std = (nc, ns, nr) == (4, 1, 10)
    P = Plant('heater' if std else f'heater{nc}_{ns}_{nr}', dt=30., substeps=4)
    xv = [P.mv('passA', .15, .95, .03, .5), P.mv('passB', .15, .95, .03, .5)]
    xf = P.mv('fuel', .15, .95, .02, .5)
    xd = P.mv('damper', .15, 1., .04, XD0)
    tin = P.load('Tin', TIN, 195., 225., measured=True)
    ph = P.load('Ph', PH, 410., 490., measured=True)
    c3 = P.load('c3', X['c3'], .005, .105)
    tamb = P.load('Tamb', TAMB, 5., 40.)
    foul = [P.param('cokeA', RFOUL, .2, 1.5), P.param('cokeB', RFOUL, .2, 1.5)]          # inside the radiant tubes of a pass, m2 K/kW
    soot = P.param('soot', 1., .8, 1.05)                         # the bank's outside coefficient
    reg = P.param('reg', 1., .85, 1.15)                          # the registers as the operator has set them
    pin = P.param('pin', 1., .95, 1.05)                          # the pressure before the pass valves
    emis = P.param('emis', 1., .92, 1.08, event=False)           # the exchange factor
    metal = P.param('metal', 1., .8, 1.25, event=False)          # the tubes' heat capacity
    c, n = cells(nc, ns, nr), nc+ns+nr
    To = [[P.state(f'T{q}{j+1}', c[j]['t0'], lo=50., hi=700., scale=100.) for j in range(n)] for q in 'AB']
    Tw = [[P.state(f'W{q}{j+1}', c[j]['t0']+35., lo=50., hi=900., scale=100.) for j in range(n)] for q in 'AB']
    Tb = [P.state(f'G{j+1}', TS0+(TG0-TS0)*(j+.5)/(nc+ns), lo=50., hi=1600., scale=100.) for j in range(nc+ns)]      # the gas leaving the rows of a cell
    tg = P.state('Tg', TG0, lo=100., hi=1600., scale=100.)
    tr = P.state('Tr', .5*(TG0+TAMB), lo=0., hi=1200., scale=100.)
    ts = P.state('Ts', TS0, lo=50., hi=1200., scale=100.)
    o2m = P.state('O2', 3.82, lo=0., hi=21., scale=1.)
    pb = P.state('Pb', PB0, lo=PATM, hi=600., scale=100.)
    fu = fuel(c3)
    # the oil: a valve and the coil between the header and the column
    F = []
    for q in range(2):
        narrow = exp(5.*log(RAD['di']/(RAD['di']-2*KCOKE*foul[q])))          # coke takes of the bore
        kq = sum(((narrow if c[j]['sec'] == 2 else 1.)*c[j]['kf'])/rho(To[q][j]) for j in range(n))
        F.append(FL.pumped(CVO, xv[q], POUT, pin*PIN, kq))
    # the fuel: the valve, the burner header, the tips
    wob = sqrt(fu['mw']*(1/FU0['mw']))
    xr = clamp((ph-pb)/ph, 0., XT)
    nv = CVF*xf*ph*(1.-xr*(1/(3*XT)))*xr/sqrt(xr+1e-4)/wob
    lr = log(clamp(PATM/pb, RC, 1.))
    q = relu(exp((2/GAM)*lr)-exp(((GAM+1)/GAM)*lr))
    nt = CT*pb*q/sqrt(q+1e-6)/wob
    # the air the draft draws
    ra = (PATM*MWA/RG)/(tamb+273.15)
    rg = lambda t: (PATM*MWG/RG)/(t+273.15)                      # noqa: E731
    rfb, rcv, rst = rg(tg), rg(.5*(tg+ts)), rg(ts)
    buoy = GRAV*(HFB*(ra-rfb)+HCV*(ra-rcv)+HST*(ra-rst))
    a = (KREG*MWA*MWA)/(ra*reg*reg)
    cc = (MWA*MWA)*(KCV/rcv+(KST+KD/(xd*xd))/rst)
    mu = fu['mw']*nt*(1/MWA)
    na = (sqrt(relu((a+cc)*buoy-a*cc*mu*mu))-cc*mu)/(a+cc)
    draft = a*na*na-(GRAV*HFB)*(ra-rfb)
    # the fire
    o2in = (.21*(1-HUM))*na
    phi = minimum(o2in/maximum(nt*fu['o2'], 1e-12), 1.)          # short of air, the rest of the fuel leaves as it came
    sp = [nt*(X['co2']+phi*fu['co2']), nt*phi*fu['h2o']+HUM*na, o2in-phi*nt*fu['o2'], (.79*(1-HUM))*na+X['n2']*nt]
    left = (1.-phi)*nt
    ng = sp[0]+sp[1]+sp[2]+sp[3]+left
    wg = sp[0]*MWS[0]+sp[1]*MWS[1]+sp[2]*MWS[2]+sp[3]*MWS[3]+left*fu['mw']
    co = [sp[0]*HCO2[k]+sp[1]*HH2O[k]+sp[2]*HO2[k]+sp[3]*HN2[k]+left*(fu['c1']*HC1[k]+fu['c3']*HC3[k]+fu['inert']*HN2[k]) for k in range(4)]
    hg = lambda t: ((co[0]*(t-15.)+co[1])*(t-15.)+co[2])*(t-15.)+co[3]       # noqa: E731  the gas at t, kW
    cg = lambda t: ((3.*co[0])*(t-15.)+2.*co[1])*(t-15.)+co[2]             # noqa: E731  kW/K
    heat = phi*nt*fu['lhv']+na*hair(tamb)+nt*hfuel(fu, TFUEL)
    # radiation from the gas to every cell that sees it; Lobo and Evans' convection on the radiant tubes
    tg4 = (tg+273.15)**4
    Q = [[(SIG*FEX*c[j]['acp'])*emis*(tg4-(Tw[q][j]+273.15)**4)+(HRC*c[j]['A'])*(tg-Tw[q][j]) if c[j]['sec'] == 2 else
          (SIG*FEX*c[j]['acp'])*emis*(tg4-(Tw[q][j]+273.15)**4) if c[j]['sec'] == 1 else 0. for j in range(n)] for q in range(2)]
    qrad = sum(Q[q][j] for q in range(2) for j in range(nc, n))
    # the bank, row by row as the gas passes it: across the rows of a cell as past a wall at one temperature
    g23 = exp((2/3)*log(wg*(3600./FREE)))
    for j in range(nc+ns-1, -1, -1):
        t = tg if j == nc+ns-1 else Tb[j+1]
        cpm, tk = cg(t), t+273.15
        ho = soot*(FCV*1.1/3600.)*((.018/CNV['do']**(1/3))*(cpm/wg)*g23*exp(.3*log(tk))+9.2e-2*tk-34.)      # kW/m2 K: Monrad and gas radiation as the paper has them
        At = 2*c[j]['A']
        twm = .5*(Tw[0][j]+Tw[1][j])
        hout = hg(twm+(t-twm)*exp(-(ho*At)/cpm))
        dq, skew = .5*(hg(t)-hout), (.25*At)*ho*(Tw[1][j]-Tw[0][j])
        Q[0][j], Q[1][j] = Q[0][j]+dq+skew, Q[1][j]+dq-skew
        P.der(Tb[j], (hout-hg(Tb[j]))/(CB*c[j]['A']))
    # balances
    for q in range(2):
        f8 = exp(.8*log(F[q]*(2/OIL)))
        for j in range(n):
            T, W, k = To[q][j], Tw[q][j], c[j]
            hi = k['hi0']*f8*exp(k['hib']*(T-k['hit']))
            qi = k['A']*(W-T)/(k['do']/k['di']/hi+k['rw']+(foul[q] if k['sec'] == 2 else RFOUL))
            P.der(T, (F[q]*(hoil(tin if j == 0 else To[q][j-1])-hoil(T))+qi)/(k['V']*rho(T)*(CP0+CP1*T)))
            P.der(W, (Q[q][j]-qi)/(k['Cw']*metal))
    P.der(tg, (heat-hg(tg)-qrad-UR*(tg-tr))/CFB)
    P.der(tr, (UR/CR)*((tg-tr)-(tr-tamb)))
    P.der(ts, (hg(Tb[0])-hg(ts))/CS)
    P.der(o2m, (100.*sp[2]/ng-o2m)/TAU_O2)
    P.der(pb, (RG*(TFUEL+273.15)/VB)*(nv-nt))
    hm = (F[0]*hoil(To[0][-1])+F[1]*hoil(To[1][-1]))/(F[0]+F[1])
    P.out('COT', (sqrt(CP0*CP0+(2*CP1)*hm)-CP0)/CP1, range=(320., 390.), band=1., track=True, target=(340., 365.))
    P.out('feed', F[0]+F[1], range=(40., 85.), band=.3, track=True, target=(54., 68.))
    P.out('dT', To[0][-1]-To[1][-1], range=(-25., 25.), band=1., track=True, target=(-3., 3.))
    P.out('O2', o2m, range=(0., 10.), band=.2, track=True, target=(2.2, 5.))
    P.out('skinA', Tw[0][-1], range=(330., 480.), band=2., hi=430.)
    P.out('skinB', Tw[1][-1], range=(330., 480.), band=2., hi=430.)
    P.out('BWT', tg, range=(550., 900.), band=5., hi=800.)
    P.out('draft', draft, range=(-20., 90.), band=2., lo=8.)
    P.out('Pb', pb, range=(100., 400.), band=5., lo=125., hi=330.)
    P.out('Tin', tin, range=(180., 240.), band=1., dv=True)
    P.out('Ph', ph, range=(380., 520.), band=5., dv=True)
    if std:
        FL.nominal(P)
    else:
        FL._apply(P, dict(x=FL.settle(P)[0]))
    return P

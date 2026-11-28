"""IFAC PID 2018 refrigeration benchmark as data for the mechanism engine (docs/sim.md, 'Refrigeration').

The one-stage vapour-compression cycle of Bejarano, Alfaya, Rodriguez, Ortega and Morilla (Benchmark PID 2018, UNED;
R404A, evaporator secondary 60 % propylene glycol, condenser secondary air): compressor speed and expansion valve
opening hold the evaporator's secondary outlet temperature and the superheat, under seven disturbances.

The benchmark's model ships as compiled S-functions only (no source, no parameters published). Written here as its
documentation describes it: the evaporator static (moving boundary: a two-phase zone and a superheated zone,
counterflow), one mass flow through the cycle, the condenser the cycle's dynamics (in its mode 2, two phase at the
outlet: the condensing temperature on a wall that heats the air; the outlet enthalpy follows the pressure, the
charge being fixed), compressor and valve static. R404A from CoolProp fitted over the operating range. The
component parameters are identified from the benchmark's reference run RSBenchmark_20170524_13_48 (its logged
pressures, enthalpies and flow); the other reference run tests them (aio/sim/validate/refrig.py).
"""
import torch

from aio.sim.engine import Plant, Mech
from aio.sim.expr import exp, log, sqrt, relu, clamp, maximum

# R404A (CoolProp 7, pseudo-pure) fits; P in bar, h kJ/kg, T C, v m3/kg, s kJ/(kg K); lp = ln P on the low side
TE_DEW = (0.020455708821130947, 0.23083196572120454, 2.2262257024089123, 20.66614981661417, -45.74411891636084)
HE_V = (-0.02232598701583063, 0.02103622072837374, 1.1598682588377587, 12.494063278971119, 340.02485934877717)
HE_L = (0.05574881155805602, 0.44013059696921103, 3.2071155354185863, 26.072656271615866, 138.94434209969492)
TC_SAT = (0.0019968826191551006, -0.15587294899307788, 5.96722434889058, -29.062614356785886)
HC_L = (0.002743028067717678, -0.19653909403750366, 8.231002803930474, 158.55157203155633)
HC_V = (0.0008630418325708198, -0.09746367276902007, 3.043507399845214, 352.16159330120337)
VC_L = (9.87458482416198e-09, -3.491526147152859e-07, 1.7360213585417337e-05, 0.0007754833058862794)
VC_V = (-2.8567977310151088e-06, 0.00018847040694445103, -0.004685967761736088, 0.04989095366238743)
TSUC = (-45.744451758081354, 20.66358123859054, 1.2742702288566563, -0.11535989409346267, 2.2184739951917005, -0.0008615607353905652, -0.00959117533644105, 0.0006429605578234798, -4.403376536288306e-06, 0.19487775043221173)
LN_RHOSUC = (1.6893976174397831, 0.9414599741963655, -0.006716696408375582, 0.0004640422536265433, 0.00529666212685828, 3.52836234726768e-05, -8.424378103805308e-05, -2.7171176757295745e-06, -1.5755253024440918e-07, 0.0018652219787614964)
S_SUC = (1.6431294474900686, -0.026291879421295742, 0.004394402661950685, -0.00038276256795573293, 0.003910251508887206, -1.1962437463840345e-05, -9.747190457743608e-06, 2.1624706512432767e-06, 3.705608838527885e-08, 5.177538507433023e-05)
H_DIS = (-674.3888253617669, -4.51582775634992, 1567.2908431558867, 5.81144935356605, -0.030048234559503673, -921.2541780213629, -0.057087880284900416, -0.389437448193355, 212.7580178832428, 0.0016641681007776299)
PC_OF_T = (1.7221460718599857e-05, 0.0019436295644037099, 0.20104829076534833, 5.9686306275119625)
# identified from the benchmark reference run RSBenchmark_20170524_13_48 (aio/sim/validate/refrig.py)
CV = (0.06369285213609119, -0.0023318631845366247, 1.2518518269458156e-05)
CS = (0.6046067326933656, 0.010887719523678093, -0.002251049828265971)
CPW = (0.9612317868385372, 0.42641405156182755, 38.24365340623253)
KV = (0.10508813237702011, 1.7538701349777497e-05, -1.2157931842983665e-05)
GH = (0.5615831834710123, -6.132124962328064, 208.25194406648407)
KDP = 0.003744461456288517
CP_SEC = 3.3703710405581893
UATP = 49.52
UASH = 4.97
KAIR = 153.52
C_WALL = 5381.8

# the benchmark's input ranges (main document, Table 3) and initial operating point (Table 4)
N0, AV0 = 36.45, 48.79
DIST = (('Tc_air', 30., 27., 33.), ('m_air', 150., 125., 175.), ('P_air', 1., 1., 1.), ('Te_sec', -20., -22., -18.), ('m_sec', 64.503, 55., 75.),
        ('P_sec', 1., 1., 1.), ('T_surr', 25., 20., 30.))


def _feat(lp, dh):
    return (1., lp, dh, lp*dh, lp*lp, dh*dh, lp*lp*dh, lp*dh*dh, dh*dh*dh, lp*lp*lp)


def _hfeat(P, s):
    return (1., P, s, P*s, P*P, s*s, P*P*s, P*s*s, s*s*s, P*P*P)


def _dot(f, c):
    out = 0.
    for a, b in zip(f, c):
        out = out+a*b
    return out


def _poly(c, x):
    out = 0.
    for a in c:
        out = out*x+a
    return out


def _cycle(P, N, Av, d, ua=1.):
    """The cycle: the condensing temperature a state; the evaporator pressure, the suction enthalpy and the share of
    the evaporator in two phase algebraic (the evaporator is static, one mass flow through the cycle)."""
    Tc = P.state('Tc', 35.9, scale=1.)
    Pe = P.state('Pe', .97, lo=.3, hi=3., scale=.05, algebraic=True, maxstep=.05)
    heo = P.state('heo', 351., lo=330., hi=400., scale=1., algebraic=True, maxstep=5.)
    z = P.state('z', .5, lo=1e-3, hi=1., scale=.05, algebraic=True, maxstep=.1)
    Pco = _poly(PC_OF_T, Tc)
    hco = _poly(GH, Pco)
    hl, hv = _poly(HC_L, Pco), _poly(HC_V, Pco)
    vl, vv = _poly(VC_L, Pco), _poly(VC_V, Pco)
    x = clamp((hco-hl)/(hv-hl), 0., 1.)
    a = Av/100.
    mv = (KV[0]*a+KV[1]*a*a+KV[2]*a*a*a)*sqrt(relu(Pco-Pe)/(vl+x*(vv-vl)))
    Pci = Pco+KDP*mv*mv
    lp = log(Pe)
    hve = _poly(HE_V, lp)
    f = _feat(lp, heo-hve)
    rho = exp(_dot(f, LN_RHOSUC))
    r = Pci/Pe
    mc = N*rho*(CV[0]+CV[1]*r+CV[2]*r*r)
    Te = _poly(TE_DEW, lp)
    cpv = 10./(_dot(_feat(lp, 10.), TSUC)-_dot(_feat(lp, 0.), TSUC))
    Cs = d['m_sec']*CP_SEC
    mcp = maximum(mv, 1e-3)*cpv
    Qsh = mcp*(d['Te_sec']-Te)*(1.-exp(-ua*UASH*(1.-z)/mcp))
    T1 = d['Te_sec']-Qsh/Cs
    P.der(Pe, (mc-mv)/5.)
    P.der(z, (Cs*(1.-exp(-ua*UATP*z/Cs))*(T1-Te)-mv*(hve-hco))/500.)
    P.der(heo, heo-hve-Qsh/maximum(mv, 1e-3))
    s = _dot(f, S_SUC)
    hci = heo+(_dot(_hfeat(Pci, s), H_DIS)-heo)/(CS[0]+CS[1]*r+CS[2]*N)
    Kair = KAIR*d['m_air']/150.
    P.der(Tc, (mv*(hci-hco)-Kair*(Tc-d['Tc_air']))/C_WALL)
    Qe = mv*(heo-hco)
    Qcomp = mv*(hci-heo)
    W = CPW[0]*Qcomp+CPW[1]*N+CPW[2]
    return dict(Tsec_out=d['Te_sec']-Qe/Cs, TSH=_dot(f, TSUC)-Te, Pe=Pe, Pc=Pco, m=mv, Qe=Qe, W=W, COP=Qe/Qcomp, hco=hco, hci=hci, heo=heo)


def _settle(P):
    """The plant's nominal point: its steady state at the nominal inputs (states' x0 rewritten)."""
    M = Mech([P], n=1, dtype=torch.float64)
    x, w, p = P.nominal()
    xs, _, ok = M.steady(w, p)
    assert bool(ok.all()), P.name
    for st, v in zip(P.xs, xs[0].tolist()):
        st['x0'] = v
    return P


def refrig_raw(dt=1.):
    """The cycle with the benchmark's inputs: compressor speed and valve opening as MVs, the seven disturbances of
    Table 1 as loads (the two secondary pressures and the surroundings temperature act on nothing, as in the
    benchmark's model for the first two). For the comparison with the reference runs."""
    P = Plant('refrig_raw', dt=dt, substeps=1)
    N, Av = P.mv('N', 30., 50., 2., N0), P.mv('Av', 10., 100., 5., AV0)
    d = {q: P.load(q, v, lo, hi) for q, v, lo, hi in DIST}
    P.param('k', 1., 1., 1.)
    o = _cycle(P, N, Av, d)
    for q in ('Tsec_out', 'TSH', 'Pe', 'Pc', 'm', 'Qe', 'W', 'COP', 'hco', 'hci', 'heo'):
        P.out(q, o[q], range=(0., 1.), band=1.)
    return _settle(P)


def refrig():
    """The benchmark as an APC sees it: compressor speed and valve opening hold the evaporator's secondary outlet
    temperature (the cooling demand) and the superheat; the seven disturbances as loads (the two secondary inlet
    temperatures and mass flows measured, as in the benchmark). A decision is 5 s."""
    P = Plant('refrig', dt=5., substeps=2)
    N, Av = P.mv('N', 30., 50., 1., N0), P.mv('Av', 10., 100., 3., AV0)
    d = {q: P.load(q, v, lo, hi, measured=q in ('Tc_air', 'm_air', 'Te_sec', 'm_sec')) for q, v, lo, hi in DIST}
    ua = P.param('ua', 1., .9, 1.05)                               # the evaporator's heat transfer (frost, fouling)
    o = _cycle(P, N, Av, d, ua)
    P.out('Tsec_out', o['Tsec_out'], range=(-26., -19.), band=.05, track=True, target=(-23.5, -21.))
    P.out('TSH', o['TSH'], range=(0., 30.), band=.3, track=True, target=(6., 22.), lo=2.)
    P.out('COP', o['COP'], range=(.5, 1.6), band=.02)
    P.out('Pc', o['Pc'], range=(13., 20.), band=.05)
    P.out('Pe', o['Pe'], range=(.5, 1.6), band=.01)
    return _settle(P)

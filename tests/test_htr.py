"""The crude heater (aio.sim.plants_htr) against the data sheet it is built on and against its own equations written
again in numpy: the design point found again from the data sheet, the table against the equations, the
plant's own steps against a fine integration, fifteen cells a pass against one cell a tube; and in episodes.
Ibrahim and Al-Qassimi, Periodica Polytechnica Chem. Eng. 54(1), 2010, 33 (the data sheet of heater 21 H-1) and
52(2), 2008, 61. No step test of a crude heater was found to hold the dynamics against."""
import math

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, fsolve

from aio.sim import core as C, control as CT, mech as ME, plants as PL, plants_htr as PH
from aio.sim.engine import Mech

F64 = torch.float64
# ---------------------------------------------------------------- the data sheet
OIL = 225700./3600.                                   # kg/s
TIN, TX, TOUT = 210., 250., 355.
QRAD, QCONV = 6.182e7/3600., 2.30e7/3600.             # kW
DUTY = QRAD+QCONV
EFF, LOSS, EXCESS, HUM = .7578, .05, .25, .015
RELEASE = DUTY/EFF
SG, KW = .9148, 12.1
LEFF, LTOT = 20.024, 20.824
RAD = dict(n=30, do=.219, di=.203, pitch=.394)
CNV = dict(n=20, do=.168, di=.152, pitch=.250, shield=4, row=4)
POUT, REFR = 910., 298000.
FUEL = dict(c1=.8043, c2=.0902, c3=.0454, ic4=.0020, nc4=.0032, ic5=.0004, nc5=.0002, co2=.0352, h2s=.0009, n2=.01735)
PRINTED = dict(ncv=927844.41, fuel=120., air=1589.014, flue=1720.9, co2=8.234, h2o=15.968, o2=3.82, release=1.1193e8, bridgewall=800., stack=400., G=2.677, visc70=30.)
# ---------------------------------------------------------------- species: heats of combustion to CO2, SO2 and steam (kJ/kmol), oxygen, products, molar mass
SPEC = dict(c1=(802.6e3, 2., 1, 2, 16.042), c2=(1428.6e3, 3.5, 2, 3, 30.069), c3=(2043.3e3, 5., 3, 4, 44.096), ic4=(2647.6e3, 6.5, 4, 5, 58.122), nc4=(2657.2e3, 6.5, 4, 5, 58.122),
            ic5=(3264.7e3, 8., 5, 6, 72.149), nc5=(3271.4e3, 8., 5, 6, 72.149), h2s=(518.0e3, 1.5, 1, 1, 34.081), co2=(0., 0., 1, 0, 44.010), n2=(0., 0., 0, 0, 28.013))
HN2 = (-2.3104e-07, 0.00326382, 28.3044, 40.8894)     # kJ/kmol above 15 C, cubic in (T - 15)
HO2 = (-1.53904e-06, 0.00584925, 28.9047, 8.34445)
HCO2 = (-4.6861e-06, 0.0164422, 38.1943, -72.9126)
HH2O = (2.79153e-07, 0.00555028, 32.8724, 30.5243)
HC1 = (-1.11226e-07, 0.0292641, 34.2175, 21.1136)
HC3 = (-4.92795e-06, 0.0805236, 77.8756, -348.059)
MWA = (1-HUM)*(.21*31.999+.79*28.013)+HUM*18.015
SIG, RG, PATM, GRAV = 5.670374e-11, 8.314, 101.325, 9.80665
# ---------------------------------------------------------------- taken (as the plant's file says)
HRC, ALPHA = 11.36e-3, .915
KMET, CPMET, RHOMET = 19e-3, .55, 7900.
RFOUL, KCOKE, FRIC, BEND = .5, 3e-3, .0165, 50.
PIN, PHD, PB0, VB, TFUEL, GAM, XT = 1400., 450., 250., .5, 25., 1.3, .65
HFB, HCV, HST, DST = 11.3, 2.5, 20., 2.2
DRAFT0, XD0 = 25., .5
KSTACK = .02*HST/DST+1.5
VFB, TAU_O2, CPREF, TAMB = 900., 20., 1., 25.
CB = PATM*(LEFF*CNV['row']*CNV['pitch']*HCV-2*CNV['n']*math.pi/4*CNV['do']**2*LEFF)/(RG*850.)*33./(CNV['n']*math.pi*CNV['do']*LEFF)


def poly(c, t):
    tau = t-15.
    return ((c[0]*tau+c[1])*tau+c[2])*tau+c[3]


def dpoly(c, t):
    tau = t-15.
    return (3*c[0]*tau+2*c[1])*tau+c[2]


def fuel(c3=None):
    """A kmol of the fuel (propane in place of methane when c3 is given)."""
    x = dict(FUEL)
    x['n2'] += 1-sum(x.values())                       # the 0.085 % the printed composition lacks
    if c3 is not None:
        x['c1'], x['c3'] = x['c1']-(c3-x['c3']), c3
    f = lambda i: sum(x[k]*SPEC[k][i] for k in x)     # noqa: E731
    return dict(lhv=f(0), o2=f(1), co2=f(2), h2o=f(3), mw=f(4), n2=x['n2'], xco2=x['co2'], c1=x['c1']+x['c2'], c3=1-x['c1']-x['c2']-x['co2']-x['n2']-x['h2s'], inert=x['co2']+x['n2']+x['h2s'])


def hfuel(fu, t):
    return fu['c1']*poly(HC1, t)+fu['c3']*poly(HC3, t)+fu['inert']*poly(HN2, t)


def hair(t):
    return (1-HUM)*(.21*poly(HO2, t)+.79*poly(HN2, t))+HUM*poly(HH2O, t)


def burn(nf, na, c3=None, tamb=TAMB, lhv=None):
    """Flue gas (kmol/s of CO2, H2O, O2, N2, fuel left), heat released and the enthalpy coming in, kW."""
    fu = fuel(c3)
    o2in = .21*(1-HUM)*na
    phi = min(1., o2in/max(nf*fu['o2'], 1e-12))
    sp = np.array([nf*(fu['xco2']+phi*(fu['co2']-fu['xco2'])), nf*phi*fu['h2o']+HUM*na, o2in-phi*nf*fu['o2'], .79*(1-HUM)*na+nf*fu['n2'], (1-phi)*nf])
    return sp, phi*nf*(lhv or fu['lhv']), na*hair(tamb)+nf*hfuel(fu, TFUEL), fu


def hgas(sp, fu, t):
    return sp[0]*poly(HCO2, t)+sp[1]*poly(HH2O, t)+sp[2]*poly(HO2, t)+sp[3]*poly(HN2, t)+sp[4]*hfuel(fu, t)


def cgas(sp, fu, t):
    return sp[0]*dpoly(HCO2, t)+sp[1]*dpoly(HH2O, t)+sp[2]*dpoly(HO2, t)+sp[3]*dpoly(HN2, t)+sp[4]*(fu['c1']*dpoly(HC1, t)+fu['c3']*dpoly(HC3, t)+fu['inert']*dpoly(HN2, t))


# ---------------------------------------------------------------- the oil
CP1 = (QRAD/(OIL*(TOUT-TX))-QCONV/(OIL*(TX-TIN)))/(.5*(TOUT+TX)-.5*(TX+TIN))
CP0 = QCONV/(OIL*(TX-TIN))-CP1*.5*(TX+TIN)


def hoil(t):
    return CP0*t+.5*CP1*t*t


def rho_api(t):
    r15 = SG*999.016
    a, d = 613.9723/r15**2, t-15.
    return r15*math.exp(-a*d*(1+.8*a*d))


RHO_B = (rho_api(210.)-rho_api(355.))/145.
RHO_A = rho_api(210.)+RHO_B*210.


def rho(t):
    return RHO_A-RHO_B*t


def abbott():
    """API (Abbott): kinematic viscosity at 100 and 210 F from the gravity and K, cSt."""
    api = 141.5/SG-131.5
    l100 = 4.39371-1.94733*KW+.12769*KW**2+3.2629e-4*api**2-1.18246e-2*KW*api+(.171617*KW**2+10.9943*api+9.50663e-2*api**2-.860218*KW*api)/(api+50.3642-4.78231*KW)
    l210 = -.463634-.166532*api+5.13447e-4*api**2-8.48995e-3*KW*api+(8.0325e-2*KW+1.24899*api+.19768*api**2)/(api+26.786-2.6296*KW)
    return 10**l100, 10**l210


def walther(t, t0, v0, slope):
    z = math.log10(math.log10(v0+.7))-slope*math.log10((t+273.15)/(t0+273.15))
    return 10**(10**z)-.7


def visc(t):
    v1, v2 = abbott()
    t1, t2 = (100-32)/1.8, (210-32)/1.8
    slope = (math.log10(math.log10(v1+.7))-math.log10(math.log10(v2+.7)))/math.log10((t2+273.15)/(t1+273.15))
    return walther(t, 70., 30., slope)*1e-6


def film(F, di, t):
    nu, r, k, cp = visc(t), rho(t), (.49744-29.4604e-5*t)/3600., CP0+CP1*t
    v = F/(r*math.pi/4*di*di)
    return .027*(v*di/nu)**.8*(cp*nu*r/k)**(1/3)*k/di


class Heater:
    """The heater at a discretisation (cells a pass in the bank, the shield and the radiant section, in the order
    the oil passes them). States: oil and wall of pass A, of pass B, the gas after each cell of the bank, then
    box, refractory, stack, analyser, burner header."""

    def __init__(self, nc=4, ns=1, nr=10, fex=.6, fcv=1.):
        self.nc, self.ns, self.nr, self.n, self.nb = nc, ns, nr, nc+ns+nr, nc+ns
        self.fex, self.fcv, self.fu0, self.F0 = fex, fcv, fuel(), OIL/2
        tubes = np.array([(CNV['n']-CNV['shield'])/nc]*nc+[CNV['shield']/ns]*ns+[RAD['n']/nr]*nr)
        do, di = np.array([CNV['do']]*self.nb+[RAD['do']]*nr), np.array([CNV['di']]*self.nb+[RAD['di']]*nr)
        pitch = np.array([CNV['pitch']]*self.nb+[RAD['pitch']]*nr)
        self.sec = np.array([0]*nc+[1]*ns+[2]*nr)
        self.do, self.di = do, di
        self.A = tubes*math.pi*do*LEFF
        self.V = tubes*math.pi/4*di*di*(LTOT+math.pi*pitch/2)
        self.Cw = tubes*math.pi/4*(do*do-di*di)*(LTOT+math.pi*pitch/2)*RHOMET*CPMET
        self.kf = tubes*FRIC*(LTOT+BEND*di)/(2*di*(math.pi/4*di*di)**2)/1e3
        self.rw = do*np.log(do/di)/(2*KMET)
        self.acp_rad, self.acp_sh = 2*RAD['n']*RAD['pitch']*LEFF*ALPHA, CNV['row']*CNV['pitch']*LEFF
        self.acp = np.where(self.sec == 2, self.acp_rad/2/nr, np.where(self.sec == 1, self.acp_sh/2/ns, 0.))
        self.free = LEFF*(CNV['row']-1)*(CNV['pitch']-CNV['do'])
        self.hi0, self.hib, self.hit = np.zeros(self.n), np.zeros(self.n), np.zeros(self.n)
        for s, (ta, tb) in enumerate(((TIN, TX-8), (TX-8, TX), (TX, TOUT))):
            d, tm, m = (CNV['di'] if s < 2 else RAD['di']), .5*(ta+tb), self.sec == s
            self.hi0[m], self.hib[m], self.hit[m] = film(self.F0, d, tm), math.log(film(self.F0, d, tm+40)/film(self.F0, d, tm-40))/80., tm
        self.Cfb, self.Cr = PATM*VFB/(RG*1013.)*34., REFR*CPREF
        self.Cs = PATM*(math.pi/4*DST*DST*HST+20.)/(RG*673.)*33.

    def valve(self, xf, ph, pb, mw):
        xr = min(max((ph-pb)/ph, 0.), XT)
        return self.cvf*xf*ph*(1-xr/(3*XT))*xr/math.sqrt(xr+1e-4)/math.sqrt(mw/self.fu0['mw'])

    def tips(self, pb, mw):
        r = max(min(PATM/pb, 1.), (2/(GAM+1))**(GAM/(GAM-1)))
        q = max(r**(2/GAM)-r**((GAM+1)/GAM), 0.)
        return self.ct*pb*q/math.sqrt(q+1e-6)/math.sqrt(mw/self.fu0['mw'])

    def air(self, nf, mw, tg, ts, xd, tamb, reg):
        """Air drawn, kmol/s, and the draft at the arch, Pa."""
        ra = PATM*MWA/(RG*(tamb+273.15))
        rg = lambda t: PATM*self.mwg/(RG*(t+273.15))  # noqa: E731
        rfb, rcv, rst = rg(tg), rg(.5*(tg+ts)), rg(ts)
        B = GRAV*(HFB*(ra-rfb)+HCV*(ra-rcv)+HST*(ra-rst))
        a = self.kreg*MWA*MWA/(ra*reg*reg)
        c = MWA*MWA*(self.kcv/rcv+(self.kst+self.kd/(xd*xd))/rst)
        mu = mw*nf/MWA
        na = (-c*mu+math.sqrt(max((a+c)*B-a*c*mu*mu, 0.)))/(a+c)
        return na, a*na*na-GRAV*HFB*(ra-rfb)

    def rhs(self, x, u, d, p, fixed=None):
        """dx/dt and everything on the way. fixed: the flows given (the design stage, before the valves are sized)."""
        n, nb = self.n, self.nb
        To, Tw = [x[0:n], x[2*n:3*n]], [x[n:2*n], x[3*n:4*n]]
        Tb = x[4*n:4*n+nb]
        m = 4*n+nb
        tg, tr, ts, o2m, pb = x[m:m+5]
        tin, ph, c3, tamb = d
        fu = fuel(c3)
        F, kq = [], []
        for q in range(2):
            k = self.kf/rho(To[q])
            k = np.where(self.sec == 2, k*(self.di/(self.di-2*p['foul'][q]*KCOKE))**5, k)
            kq.append(k.sum())
            dp, cvx = max(p['pin']*PIN-POUT, 0.), (0. if fixed else self.cvo*u[q])
            F.append(fixed['F'][q] if fixed else cvx*dp/math.sqrt(dp+1.)/math.sqrt(1+kq[q]*cvx*cvx))
        if fixed:
            nf = nt = fixed['nf']
            na, draft = fixed['na'], DRAFT0
        else:
            nf, nt = self.valve(u[2], ph, pb, fu['mw']), self.tips(pb, fu['mw'])
            na, draft = self.air(nt, fu['mw'], tg, ts, u[3], tamb, p['reg'])
        sp, rel, hin, fu = burn(nt, na, c3, tamb)
        ng = sp.sum()
        wg = sp[0]*44.01+sp[1]*18.015+sp[2]*31.999+sp[3]*28.013+sp[4]*fu['mw']
        Q = [SIG*self.acp*self.fex*p.get('emis', 1.)*((tg+273.15)**4-(Tw[q]+273.15)**4)+np.where(self.sec == 2, HRC*self.A*(tg-Tw[q]), 0.) for q in range(2)]
        qrad = Q[0].sum()+Q[1].sum()
        G, hout = wg/self.free*3600., np.zeros(nb)
        for j in range(nb-1, -1, -1):                      # the bank, in the order the gas passes it
            t = tg if j == nb-1 else Tb[j+1]
            cpm = cgas(sp, fu, t)
            ho = p['soot']*self.fcv*1.1*(.018*(cpm/wg)*G**(2/3)*(t+273.15)**.3/CNV['do']**(1/3)+9.2e-2*(t+273.15)-34.)/3600.
            At, twm = 2*self.A[j], .5*(Tw[0][j]+Tw[1][j])
            hout[j] = hgas(sp, fu, twm+(t-twm)*math.exp(-ho*At/cpm))
            q = hgas(sp, fu, t)-hout[j]
            Q[0][j] += .5*q+.25*ho*At*(Tw[1][j]-Tw[0][j])
            Q[1][j] += .5*q-.25*ho*At*(Tw[1][j]-Tw[0][j])
        dx = np.zeros_like(x)
        for q in range(2):
            rf = np.where(self.sec == 2, p['foul'][q], RFOUL)
            hi = self.hi0*(F[q]/self.F0)**.8*np.exp(self.hib*(To[q]-self.hit))
            qi = self.A*(Tw[q]-To[q])/(self.do/(self.di*hi)+self.rw+rf)
            prev = np.concatenate(([tin], To[q][:-1]))
            dx[2*n*q:2*n*q+n] = (F[q]*(hoil(prev)-hoil(To[q]))+qi)/(self.V*rho(To[q])*(CP0+CP1*To[q]))
            dx[2*n*q+n:2*n*q+2*n] = (Q[q]-qi)/(self.Cw*p.get('metal', 1.))
        loss = fixed['loss'] if fixed else self.ur*(tg-tr)
        dx[4*n:m] = (hout-hgas(sp, fu, Tb))/(CB*self.A[:nb])
        dx[m] = (rel+hin-hgas(sp, fu, tg)-qrad-loss)/self.Cfb
        dx[m+1] = ((tg-tr)-(tr-tamb))/3.4e4 if fixed else self.ur*((tg-tr)-(tr-tamb))/self.Cr
        dx[m+2] = (hgas(sp, fu, Tb[0])-hgas(sp, fu, ts))/self.Cs
        dx[m+3] = (100*sp[2]/ng-o2m)/TAU_O2
        dx[m+4] = 0. if fixed else RG*(TFUEL+273.15)/VB*(nf-nt)
        ta, tb = To[0][-1], To[1][-1]
        hm = (F[0]*hoil(ta)+F[1]*hoil(tb))/(F[0]+F[1])
        y = dict(COT=(-CP0+math.sqrt(CP0*CP0+2*CP1*hm))/CP1, feed=F[0]+F[1], dT=ta-tb, O2=o2m, skinA=Tw[0][-1], skinB=Tw[1][-1], BWT=tg, draft=draft, Pb=pb,
                 TX=.5*(To[0][nb-1]+To[1][nb-1]), stack=ts, nf=nt, na=na, ng=ng, wg=wg, rel=rel, hin=hin, qrad=qrad, loss=loss, kq=kq, F=F, sp=sp,
                 qcoil=sum(float(Q[q][self.sec == 2].sum()) for q in range(2)), o2=100*sp[2]/ng)
        return dx, y

    def steady(self, x, u, d, p, fixed=None):
        sc = np.concatenate((np.full(4*self.n, 100.), np.full(self.nb, .1), [10., 1e4, 1., 10., .01]))      # seconds: the residual as a move
        return fsolve(lambda z: self.rhs(z, u, d, p, fixed)[0]*sc, x, xtol=1e-13)

    def y(self, x, u, d, p):
        o = self.rhs(x, u, d, p)[1]
        return np.array([o[k] for k in ROWS])

    def size(self):
        """Sized at the design point, as the plant's file says: the printed heat release with 25 % excess air, the
        oil at its three temperatures; the exchange factor and the bank's factor fitted to the two the heater
        has to give; valves and damper half open, the burners at their pressure, the arch at its draft."""
        fu = self.fu0
        air_per = (1+EXCESS)*fu['o2']/(.21*(1-HUM))
        self.nf0 = RELEASE/(fu['lhv']+hfuel(fu, TFUEL)+air_per*hair(TAMB))
        self.na0 = air_per*self.nf0
        sp = burn(self.nf0, self.na0)[0]
        self.mwg = (sp[0]*44.01+sp[1]*18.015+sp[2]*31.999+sp[3]*28.013)/sp.sum()
        u, d, p = U0, D0, P0
        fixed = dict(F=[self.F0, self.F0], nf=self.nf0, na=self.na0, loss=LOSS*RELEASE)
        prof = np.concatenate((np.linspace(TIN, TX, self.nb+1)[1:], np.linspace(TX, TOUT, self.nr+1)[1:]))
        x = [np.concatenate((prof, prof+35., prof, prof+35., 395.+345.*(np.arange(self.nb)+.5)/self.nb, [740., 380., 395., 3.82, PB0]))]

        def res(z):
            self.fex, self.fcv = z
            x[0] = self.steady(x[0], u, d, p, fixed)
            o = self.rhs(x[0], u, d, p, fixed)[1]
            return np.array([o['TX']-TX, o['COT']-TOUT])
        z = np.array([self.fex, self.fcv])
        for _ in range(30):
            r = res(z)
            if np.abs(r).max() < 1e-9:
                break
            J = np.zeros((2, 2))
            for i in range(2):
                zz = z.copy()
                zz[i] *= 1.0005
                J[:, i] = (res(zz)-r)/(zz[i]-z[i])
            z = z-np.linalg.solve(J, r)
        res(z)
        o = self.rhs(x[0], u, d, p, fixed)[1]
        tg, ts, wg = o['BWT'], o['stack'], o['wg']
        self.ur = 2*LOSS*RELEASE/(tg-TAMB)
        dp = PIN-POUT
        self.dp0 = o['kq'][0]*self.F0**2
        self.cvo = self.F0/(.5*math.sqrt(dp*dp/(dp+1.)-self.dp0))
        self.cvf = self.ct = 1.
        self.ct, self.cvf = self.nf0/self.tips(PB0, fu['mw']), self.nf0/self.valve(.5, PHD, PB0, fu['mw'])
        ra, rg = PATM*MWA/(RG*(TAMB+273.15)), lambda t: PATM*self.mwg/(RG*(t+273.15))
        rfb, rcv, rst = rg(tg), rg(.5*(tg+ts)), rg(ts)
        B = GRAV*(HFB*(ra-rfb)+HCV*(ra-rcv)+HST*(ra-rst))
        self.kreg = (DRAFT0+GRAV*HFB*(ra-rfb))*ra/(MWA*self.na0)**2
        gm = wg/self.free
        fj = (.25+.118/((CNV['pitch']-CNV['do'])/CNV['do'])**1.08)*(gm*CNV['do']/3.9e-5)**-.16      # Jakob, staggered rows
        dpcv = 4*fj*(2*CNV['n']/CNV['row'])*gm**2/(2*rcv)
        self.kcv = dpcv*rcv/wg**2
        self.kst = KSTACK/(2*(math.pi/4*DST*DST)**2)
        self.kd = (B-(DRAFT0+GRAV*HFB*(ra-rfb))-dpcv-self.kst*wg**2/rst)*rst/wg**2*XD0**2
        x[0][4*self.n+self.nb+1] = .5*(tg+TAMB)
        self.xs = self.steady(x[0], u, d, p)
        self.design = dict(tg=tg, ts=ts, dp0=self.dp0, fex=z[0], fcv=z[1], dpcv=dpcv, dpd=self.kd/XD0**2*wg**2/rst, B=B)
        return self


ROWS = ('COT', 'feed', 'dT', 'O2', 'skinA', 'skinB', 'BWT', 'draft', 'Pb')
U0, D0 = np.array([.5, .5, .5, XD0]), np.array([TIN, PHD, FUEL['c3'], TAMB])
P0 = dict(foul=[RFOUL, RFOUL], soot=1., reg=1., pin=1.)


@pytest.fixture(scope='module')
def ref():
    return Heater().size()


@pytest.fixture(scope='module')
def plant():
    P = PH.heater()
    return P, Mech([P], n=1, dtype=F64)


def perm(P, n=15, nb=5):
    """The plant's states in the order of the numpy model."""
    nm = [s['name'] for s in P.xs]
    return [nm.index(f'{k}{q}{j+1}') for q in 'AB' for k in 'TW' for j in range(n)]+[nm.index(f'G{j+1}') for j in range(nb)]+[nm.index(k) for k in ('Tg', 'Tr', 'Ts', 'O2', 'Pb')]


def inputs(P, M, u, d, p):
    w = torch.tensor([[*u, *d]], dtype=F64)
    assert [q['name'] for q in P.ws] == ['passA', 'passB', 'fuel', 'damper', 'Tin', 'Ph', 'c3', 'Tamb']
    mp, pn = M.pnom.clone(), [q['name'] for q in P.ps]
    for k, v in (('cokeA', p['foul'][0]), ('cokeB', p['foul'][1]), ('soot', p['soot']), ('reg', p['reg']), ('pin', p['pin']), ('emis', p.get('emis', 1.)), ('metal', p.get('metal', 1.))):
        mp[0, pn.index(k)] = v
    return w, mp


# ---------------------------------------------------------------- the data sheet and the paper
def test_the_air_and_the_flue_gas_are_the_papers():
    """At the paper's 120 kmol/h of fuel and 25 % excess air the composition gives its air, its flue gas and its
    analysis to the digits printed (its nitrogen, printed as 71.79 %, is 71.98 %: the four printed do not add up)."""
    fu = fuel()
    air = (1+EXCESS)*fu['o2']/(.21*(1-HUM))*PRINTED['fuel']
    sp = burn(PRINTED['fuel'], air)[0]
    assert abs(air-PRINTED['air']) < .05 and abs(sp.sum()-PRINTED['flue']) < .15
    pct = 100*sp/sp.sum()
    assert abs(pct[0]-PRINTED['co2']) < 2e-3 and abs(pct[1]-PRINTED['h2o']) < 2e-3 and abs(pct[2]-PRINTED['o2']) < 2e-3 and abs(pct[3]-71.98) < .01
    assert abs(fu['mw']-19.99) < .07


def test_the_papers_heating_value_is_not_its_compositions():
    """The composition's lower heating value is 883,381 kJ/kmol; the paper prints 927,844 (5 % more). With the
    printed value and real enthalpies the paper's own balance gives its heat release and, within 10 C, its two
    gas temperatures; by the composition the same heat release takes 5 % more fuel and makes 5 % more flue gas,
    which leaves the radiant section and the bank cooler."""
    fu = fuel()
    assert abs(fu['lhv']-883381.) < 5. and abs(PRINTED['ncv']/fu['lhv']-1.0503) < 2e-4
    out = {}
    for name, ncv, nf in (('printed', PRINTED['ncv'], PRINTED['fuel']/3600.), ('composition', fu['lhv'], None)):
        per = (1+EXCESS)*fu['o2']/(.21*(1-HUM))
        nf = nf or RELEASE/(ncv+hfuel(fu, TFUEL)+per*hair(TAMB))
        sp, _, hin, _ = burn(nf, per*nf)
        rel = nf*ncv+hin
        out[name] = (rel*3600., nf*3600., sp.sum()*3600., brentq(lambda t: rel*(1-LOSS)-QRAD-hgas(sp, fu, t), 300., 1500.), brentq(lambda t: rel*(1-LOSS)-DUTY-hgas(sp, fu, t), 100., 1500.))
    rel, nf, ng, tb, ts = out['printed']
    assert abs(rel/PRINTED['release']-1) < 1e-3 and abs(tb-PRINTED['bridgewall']) < 10. and abs(ts-PRINTED['stack']) < 10.
    rel, nf, ng, tb, ts = out['composition']
    assert abs(rel/PRINTED['release']-1) < 1e-3 and abs(nf-126.05) < .1 and abs(ng-1807.7) < .5 and abs(tb-757.) < 2. and abs(ts-391.) < 2.


def test_the_oil_and_the_bank_are_the_data_sheets():
    """The oil's heat capacity carries the two printed duties over the printed temperatures; the API correlation
    gives the data sheet's one viscosity (29.1 against 30 cSt at 70 C); the bank's free area gives the printed
    mass velocity at the printed flue gas (2.70 against 2.677 kg/m2 s)."""
    assert abs(OIL*(hoil(TX)-hoil(TIN))-QCONV) < 1e-6 and abs(OIL*(hoil(TOUT)-hoil(TX))-QRAD) < 1e-6
    assert abs(CP0-PH.CP0) < 1e-12 and abs(CP1-PH.CP1) < 1e-15 and 2.5 < CP0+CP1*230. < 2.56 and 2.6 < CP0+CP1*302.5 < 2.62
    v1, v2 = abbott()
    slope = (math.log10(math.log10(v1+.7))-math.log10(math.log10(v2+.7)))/math.log10(372.04/310.93)
    assert abs(walther(70., (100-32)/1.8, v1, slope)-29.1) < .1
    assert abs(visc(70.)*1e6-30.) < 1e-9 and abs(visc(300.)-PH.visc(300.)) < 1e-15 and .7e-6 < visc(355.) < .8e-6
    free = LEFF*(CNV['row']-1)*(CNV['pitch']-CNV['do'])
    assert abs(PRINTED['flue']*27.82/3600./free-2.70) < .01 and abs(2.70/PRINTED['G']-1) < .01
    for t in (210., 282.5, 355.):
        assert abs(rho(t)/rho_api(t)-1) < 3e-4 and abs(rho(t)-PH.rho(t)) < 1e-9


def test_the_design_point_is_found_again(ref):
    """From the data sheet alone the numpy model finds the two fitted numbers and the design point the plant's file
    holds, and at it the heater is the data sheet's: feed, three oil temperatures, heat release, excess air,
    efficiency; the valves half open, the burners at their pressure, the arch at its draft."""
    d = ref.design
    assert abs(d['fex']-PH.FEX) < 2e-6 and abs(d['fcv']-PH.FCV) < 2e-6
    assert abs(d['tg']-PH.TG0) < 2e-3 and abs(d['ts']-PH.TS0) < 2e-3 and abs(d['dp0']-PH.DP0) < 2e-3
    for a, b in ((ref.cvo, PH.CVO), (ref.cvf, PH.CVF), (ref.ct, PH.CT), (ref.kreg, PH.KREG), (ref.kcv, PH.KCV), (ref.kst, PH.KST), (ref.kd, PH.KD), (ref.ur, PH.UR), (ref.nf0, PH.NF0)):
        assert abs(a/b-1) < 2e-5
    o = ref.rhs(ref.xs, U0, D0, P0)[1]
    assert abs(o['feed']*3.6-225.7) < 1e-3 and abs(o['TX']-TX) < 2e-3 and abs(o['COT']-TOUT) < 2e-3 and abs(o['dT']) < 1e-9
    assert abs((o['rel']+o['hin'])*3600./PRINTED['release']-1) < 1e-3 and abs(o['o2']-PRINTED['o2']) < 2e-3
    duty = o['feed']*(hoil(o['COT'])-hoil(TIN))
    assert abs(duty/DUTY-1) < 1e-4 and abs(duty/(o['rel']+o['hin'])-EFF) < 1e-4 and abs(o['loss']/(o['rel']+o['hin'])-LOSS) < 1e-4
    assert abs(o['draft']-DRAFT0) < .01 and abs(o['Pb']-PB0) < 1e-3
    # what follows without having been asked for: a bridgewall of 720 C (the paper computes 800 with its heating value), a stack of 391 (400), 20.8 kW/m2 on
    # the radiant tubes, 174 kPa through a coil, eleven minutes through a pass, the damper taking a quarter of the draft
    assert abs(o['BWT']-720.2) < .2 and abs(o['stack']-390.7) < .2 and abs(o['skinA']-394.1) < .2
    assert abs(o['qcoil']/(2*ref.A[ref.sec == 2].sum())-20.8) < .1 and abs(d['dp0']-174.1) < .1
    n = ref.n
    assert 640. < float((ref.V*rho(ref.xs[:n])).sum())/ref.F0 < 680.
    assert 40. < d['dpcv'] < 48. and 50. < d['dpd'] < 62. and 230. < d['B'] < 250.
    assert 9. < ref.Cr/(2*ref.ur)/3600. < 9.6                      # hours: the walls


# ---------------------------------------------------------------- the engine against the equations
def test_the_table_is_the_equations(plant, ref):
    P, M = plant
    pm = perm(P)
    assert P.nx == 70 and [o['name'] for o in P.ys[:9]] == list(ROWS)
    x0 = M.x0[0].numpy()[pm]
    assert np.abs(x0-ref.xs).max() < 1e-3 and float((M.f(M.x0, M.wnom, M.pnom)/M.xscale).abs().max()) < 1e-9
    rng = np.random.default_rng(0)
    for k in range(12):
        x = x0+rng.normal(0, 8., x0.size)
        x[-2], x[-1] = max(3.8+rng.normal(), .1), 250.+rng.uniform(-80., 80.)
        u = np.array([rng.uniform(.2, .9), rng.uniform(.2, .9), rng.uniform(.2, .9), rng.uniform(.2, 1.)])
        d = np.array([rng.uniform(195., 225.), rng.uniform(410., 490.), rng.uniform(.005, .105), rng.uniform(5., 40.)])
        p = dict(foul=[rng.uniform(.2, 1.5), rng.uniform(.2, 1.5)], soot=rng.uniform(.8, 1.05), reg=rng.uniform(.85, 1.15), pin=rng.uniform(.95, 1.05), emis=rng.uniform(.92, 1.08),
                 metal=rng.uniform(.8, 1.25))
        w, mp = inputs(P, M, u, d, p)
        xt = torch.zeros(1, x0.size, dtype=F64)
        xt[0, pm] = torch.tensor(x)
        fr, o = ref.rhs(x, u, d, p)
        fe, ye = M.f(xt, w, mp)[0].numpy()[pm], M.y(xt, w, mp)[0, :9].numpy()
        assert (np.abs(fe-fr)/(np.abs(fr)+1e-6)).max() < 5e-4
        assert (np.abs(ye-np.array([o[q] for q in ROWS]))/(np.abs(ye)+1e-3)).max() < 5e-4


STEPS = (('fuel', 2, .04), ('passA', 0, .06), ('damper', 3, .08), ('Tin', 4, 10.), ('c3', 6, .03))


def moved(i, dv):
    u, d = U0.copy(), D0.copy()
    (u if i < 4 else d)[i % 4] += dv
    return u, d


@pytest.fixture(scope='module')
def runs(plant, ref):
    """An hour after a step of each kind: the plant at its own substeps, the numpy model at the same cells (LSODA)."""
    P, M = plant
    out = {}
    for name, i, dv in STEPS:
        u, d = moved(i, dv)
        w, mp = inputs(P, M, u, d, P0)
        x, ye = M.x0.clone(), []
        for k in range(120):
            x = M.advance(x, w, mp)
            ye.append(M.y(x, w, mp)[0, :9].numpy())
        sol = solve_ivp(lambda t, z: ref.rhs(z, u, d, P0)[0], (0., 3600.), ref.xs, method='LSODA', t_eval=np.arange(30., 3601., 30.), rtol=1e-9, atol=1e-9)
        out[name] = (np.array(ye), np.array([ref.y(sol.y[:, k], u, d, P0) for k in range(120)]))
    return out


def test_the_plants_own_steps_follow_the_fine_integration(plant, runs):
    """Within 0.2 of a band on every row (the analyser's 20 s lag is the worst: 0.13)."""
    P, M = plant
    band = np.array([o['band'] for o in P.ys[:9]])
    for name, (e, r) in runs.items():
        assert (np.abs(e-r)/band).max() < .2, name


def test_what_the_heater_does(plant, runs, ref):
    y0 = ref.y(ref.xs, U0, D0, P0)
    cot = {k: v[1][:, 0]-y0[0] for k, v in runs.items()}
    # the fuel: a ramp over the time the oil takes, with no dead time and no overshoot: a tenth after two minutes, two thirds after ten, all but 5 % after twenty
    f = cot['fuel']
    assert f[-1] > 9.5 and .003 < f[0]/f[-1] < .015 and .06 < f[3]/f[-1] < .12 and .55 < f[19]/f[-1] < .75 and f[39]/f[-1] > .93 and bool((np.diff(f[:60]) > 0).all())
    # the inlet temperature: nothing for five minutes, then the wave
    t = cot['Tin']
    assert t[-1] > 8.5 and abs(t[9]/t[-1]) < .02 and t[19]/t[-1] < .2 and t[39]/t[-1] > .6
    # a pass valve: more oil in that pass, colder; the other pass barely
    e, r = runs['passA']
    assert r[-1, 1]-y0[1] > 2. and r[-1, 2]-y0[2] < -7. and r[-1, 4]-y0[4] < -8. and abs(r[-1, 5]-y0[5]) < 1.
    # the damper: more air at once (oxygen, draft), the oil a little colder later
    e, r = runs['damper']
    assert r[3, 3]-y0[3] > .4 and r[0, 7]-y0[7] > 6. and -1.5 < r[-1, 0]-y0[0] < -.8
    # richer fuel through the same valve: more heat and less oxygen (the Wobbe number: heating value over the root of the molar mass)
    e, r = runs['c3']
    fu, fr = fuel(), fuel(FUEL['c3']+.03)
    wobbe = fr['lhv']/math.sqrt(fr['mw'])/(fu['lhv']/math.sqrt(fu['mw']))
    assert 1.015 < wobbe < 1.025 and r[-1, 0]-y0[0] > 3. and r[-1, 3]-y0[3] < -.25 and abs(r[-1, 8]-y0[8]) < .5
    # coke in one pass: its skin hotter, its flow a little smaller; soot on the bank: the stack hotter and the oil colder
    P, M = plant
    p = dict(P0, foul=[1.5, RFOUL])
    xs = ref.steady(ref.xs, U0, D0, p)
    o = ref.rhs(xs, U0, D0, p)[1]
    assert o['skinA']-y0[4] > 15. and abs(o['skinB']-y0[5]) < 3. and -.6 < o['F'][0]-ref.F0 < -.1
    p = dict(P0, soot=.8)
    o = ref.rhs(ref.steady(ref.xs, U0, D0, p), U0, D0, p)[1]
    assert o['stack']-390.7 > 15. and o['COT']-355. < -3.


def test_the_walls_keep_the_outlet_moving_for_hours(ref):
    """After the fuel valve moves, 97 % of the outlet's answer is there in an hour; the rest follows the refractory
    (nine hours)."""
    u, d = moved(2, .04)
    y0 = ref.y(ref.xs, U0, D0, P0)[0]
    sol = solve_ivp(lambda t, z: ref.rhs(z, u, d, P0)[0], (0., 40*3600.), ref.xs, method='LSODA', t_eval=[3600., 2*3600., 10*3600., 40*3600.], rtol=1e-9, atol=1e-9)
    c = np.array([ref.y(sol.y[:, k], u, d, P0)[0] for k in range(4)])-y0
    assert .96 < c[0]/c[3] < .985 and c[1]/c[3] < .99 and .98 < c[2]/c[3] < .998


def test_fifteen_cells_against_one_a_tube(runs):
    """The plant's 15 cells a pass against 50 (one a tube): the same steady state (0.2 C at the skin), a step of a
    valve within 0.3 of a band (4 % of the move); the dead time after the inlet temperature is smeared (0.9 of a
    band, 9 % of the move)."""
    fine = Heater(16, 4, 30).size()
    y0, yf = runs['fuel'][1][0]*0+Heater.y(fine, fine.xs, U0, D0, P0), None
    coarse0 = np.array([355.0008, 62.6944, 0., 3.8196, 394.1184, 394.1184, 720.1917, 24.9943, 250.])
    assert np.abs(y0-coarse0).max() < .25 and abs(y0[0]-coarse0[0]) < 2e-3
    band = np.array([1., .3, 1., .2, 2., 2., 5., 2., 5.])
    for name, i, dv, tol, share in (('fuel', 2, .04, .35, .04), ('Tin', 4, 10., 1., .11)):
        u, d = moved(i, dv)
        sol = solve_ivp(lambda t, z: fine.rhs(z, u, d, P0)[0], (0., 3600.), fine.xs, method='LSODA', t_eval=np.arange(30., 3601., 30.), rtol=1e-8, atol=1e-8)
        yf = np.array([fine.y(sol.y[:, k], u, d, P0) for k in range(120)])-y0
        e = runs[name][0]-coarse0
        assert (np.abs(e-yf)/band).max() < tol and np.abs(e-yf)[:, 0].max() < share*np.abs(yf[:, 0]).max(), name


# ---------------------------------------------------------------- in the shell
def test_it_is_in_the_default_pool():
    assert 'heater' in PL.NEW and 'heater' in PL.PLANTS and 'heater' not in PL.LARGE and 'heater' not in PL.FIRST


@pytest.fixture(scope='module')
def prog():
    torch.manual_seed(0)
    return C.build(6, 5, mech=1., plants=ME.pool(['heater']))


def test_episodes_on_the_heater(prog):
    hold = CT.rollout(prog, CT.Hold())
    pi = CT.rollout(prog, CT.PI())
    tch = CT.rollout(prog, CT.Teacher('late'))
    prog.set_teacher()
    assert not bool(hold['failed'].any() | pi['failed'].any() | tch['failed'].any())
    h, f, t = (float(r['score'].mean()) for r in (hold, pi, tch))
    print('hold %.0f formula PI %.0f teacher %.0f' % (h, f, t))
    assert t < .5*f and f < h and bool((tch['score'] <= hold['score']).all())

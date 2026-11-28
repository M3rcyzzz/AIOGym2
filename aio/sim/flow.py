"""Units and their connections (docs/sim.md, 'Flowsheets'): a plant written the way dynamic flowsheet
simulators build one, and compiled to one table of the mechanism engine.

  holdups      every vessel, tray and shell is a holdup with its own material and energy balance, and a
               pressure that follows from its state: a boiling liquid has its bubble pressure (`Liquid.P`),
               a steam shell its own pressure state
  flows        nothing sets a flow but a resistance between two pressures: tray holes (orifice, against the
               liquid on the tray above), a weir (Francis), a valve, a pumped line (pump curve and valve
               together). A valve that is wide open or a pump at the end of its curve is the limit of the
               plant, not a number written beside it. Every square root law is regular at no flow (`root`)
  properties   of a `Package` (aio.sim.props): fitted to a rigorous property package over the window the
               flowsheet works in
  sizing       the equipment is sized from a design point with the usual rules (flooding velocity, weir
               load, residence times, valves half open), then run as rated equipment
  steady state the design point is rough: the flowsheet is run to rest from it once (`settle`) and the
               result kept with the code (aio/sim/nominal.json, checked against the table it belongs to)

Units: kmol, kJ, K, kPa, m, model time in `time_unit` seconds (an hour by default).
"""
import copy
import hashlib
import json
import math
import pathlib
import torch

from aio.sim import props as PP
from aio.sim.engine import Plant, Mech
from aio.sim.expr import E, exp, log, sqrt, relu, minimum, maximum, clamp
from aio.sim.props import R, dot

GRAV = 9.80665
NOMINAL = pathlib.Path(__file__).with_name('nominal.json')
CPW = 75.4                       # liquid water, kJ/(kmol K)


class Stream:
    """A flow of material: F (kmol per time unit), composition, molar enthalpy."""

    def __init__(self, F, z, h):
        self.F, self.z, self.h = F, z, h


class Liquid:
    """A liquid holdup: moles of every species and temperature (states). Read as a boiling liquid it has a
    bubble pressure `P` and an equilibrium vapour `y` (computed when asked for)."""

    def __init__(self, fs, name, pkg, n0, T0, k=None):
        P, M0 = fs.P, float(sum(n0))
        self.fs, self.name, self.pkg, self.k = fs, name, pkg, k
        self.n = [P.state(f'{name}.{s}', v, lo=0., scale=M0) for s, v in zip(pkg.species, n0)]
        self.T = P.state(f'{name}.T', T0, lo=pkg.d.get('Tlo', 200.)-80., hi=pkg.d.get('Thi', 600.)+80., scale=10.)
        self.M = maximum(sum(self.n[1:], self.n[0]), 1e-9*M0).held if pkg.nc > 1 else self.n[0]
        self.x = [v/self.M for v in self.n] if pkg.nc > 1 else [1.]
        self.hli = pkg.hl_i(self.T, self.x)                         # partial molar enthalpies (with the heat of mixing, where the package has one)
        self.hl = dot(self.x, self.hli).held if pkg.nc > 1 else self.hli[0]
        self.vol = dot(self.n, pkg.vl_i(self.T)).held
        self.mass = dot(self.n, pkg.mw)
        self.dn, self.dH = [0.]*pkg.nc, 0.
        self._vle = None
        fs.nodes.append(self)

    # ---- as a boiling liquid
    def _eq(self):
        if self._vle is None:
            pk = self.pkg
            P, y = pk.bubble(self.x, self.T, self.k)
            hv = dot(y, pk.hv_i(self.T, P)).held
            self._vle = dict(P=P, y=y, hv=hv, rhov=P/(pk.z(y)*R*self.T), mwv=pk.mwm(y))
        return self._vle

    P = property(lambda self: self._eq()['P'])
    y = property(lambda self: self._eq()['y'])
    hv = property(lambda self: self._eq()['hv'])
    rhov = property(lambda self: self._eq()['rhov'])               # vapour molar density, kmol/m3
    mwv = property(lambda self: self._eq()['mwv'])

    def liquid(self, F):
        return Stream(F, self.x, self.hl)

    def vapour(self, F):
        return Stream(F, self.y, self.hv)

    # ---- balances
    def add(self, s, sign=1.):
        for i in range(self.pkg.nc):
            self.dn[i] = self.dn[i]+sign*(s.F*s.z[i])
        self.dH = self.dH+sign*(s.F*s.h)

    def heat(self, Q):
        self.dH = self.dH+Q

    def close(self):
        P, pk = self.fs.P, self.pkg
        for v, d in zip(self.n, self.dn):
            P.der(v, d)
        P.der(self.T, (self.dH-dot(self.hli, self.dn))/(dot(self.n, pk.cpl_i(self.T))+self.M*pk.cpe(self.x, self.T)))


class Flowsheet:
    def __init__(self, name, dt, time_unit=3600., substeps=1, refresh=False):
        self.P = Plant(name, dt, time_unit=time_unit, substeps=substeps, refresh=refresh)
        self.tu, self.nodes = float(time_unit), []
        self.mv, self.load, self.param, self.out, self.state = self.P.mv, self.P.load, self.P.param, self.P.out, self.P.state

    def liquid(self, pkg, name, n0, T0, k=None):
        return Liquid(self, name, pkg, n0, T0, k)

    def move(self, s, src=None, dst=None):
        """The stream s leaves src and enters dst (None: the world outside)."""
        if src is not None:
            src.add(s, -1.)
        if dst is not None:
            dst.add(s)
        return s

    def close(self, settle=True):
        for nd in self.nodes:
            nd.close()
        if settle:
            nominal(self.P)
        return self.P


# ---------------------------------------------------------------- resistances
def root(dp, d0):
    """The square root law of a turbulent resistance, regular at no flow: dp/sqrt(dp + d0) for a driving pressure
    difference dp > 0 (sqrt(dp) well above d0, linear below it, as any real resistance is at its laminar end),
    nothing against it. A bare sqrt has an infinite slope where the flow starts, and a plant that passes through
    no flow (a tray that seals, a valve between equal pressures) would not be integrable."""
    q = relu(dp)
    return q/sqrt(q+d0)


def weir(fs, t, area, lw, hw, c=1.84):
    """Liquid over the weir of a tray (Francis: m3/s = c lw h^1.5), kmol per time unit."""
    how = relu(t.vol/area-hw)
    return (c*lw*fs.tu)*how*sqrt(how)*t.M/t.vol


def holes(fs, lo, up, ah, xi=1.9, head=0., cf=1., d0=.01):
    """Vapour from a boiling holdup up through the holes of the tray above (orifice: dp = xi rho u^2/2), against
    the pressure there and the liquid standing on it (head, kPa)."""
    return (ah*math.sqrt(2e3/xi)*fs.tu)*cf*sqrt(lo.rhov/lo.mwv)*root(lo.P-up.P-head, d0)


def wet(t, area, beta=.6):
    """The pressure the liquid on a tray adds for the vapour coming up through it (aerated: beta), kPa."""
    return (beta*GRAV/(1e3*area))*t.mass


def valve(cv, x, dp, d0=1.):
    """A valve of linear trim: flow = cv x sqrt(dp)."""
    return cv*x*root(dp, d0)


def pumped(cv, x, dp, head, kq, d0=1.):
    """A pump (head - kq F^2) and a valve in one line, against the pressure difference dp it has to overcome:
    the flow where the pump curve meets the valve."""
    return cv*x*root(head-dp, d0)/sqrt(1.+kq*(cv*x)*(cv*x))


def tsat(pkg, P, i=0):
    """Boiling temperature of the pure species i at P: Antoine inverted."""
    a, b, c = pkg.ant[i]
    return b/(a-log(P))-c


def cooler(F, Th, Tc, ua0, F0, share=.5, cp=CPW):
    """Duty of a condenser or cooler with the hot side at one temperature Th: effectiveness of the coolant flow F
    entering at Tc; the coolant film carries `share` of the resistance at the design flow F0 and falls with F^0.8."""
    ua = ua0/((1.-share)+share*exp(-.8*log(F/F0)))
    return F*cp*(Th-Tc)*(1.-exp(-ua/(F*cp)))


def pi(fs, name, e, kc, ti, lo=0., hi=1., bias=.5):
    """A PI controller of the base layer on the error e (output lo..hi), integrating only while its output is
    inside the limits or the error leads back into them."""
    I = fs.state(f'{name}.I', 0., scale=1.)
    raw = bias+kc*(e+I/ti)
    out = clamp(raw, lo, hi)
    from aio.sim.expr import step
    fs.P.der(I, e*(1.-step(raw-hi)*step(e*kc)-step(lo-raw)*step(-e*kc)))
    return out


# ---------------------------------------------------------------- the nominal steady state
def signature(P):
    """What the stored steady state belongs to: the plant's equations and declarations."""
    h = hashlib.sha1()
    h.update(repr(P.g.nodes).encode())
    h.update(repr([(q, P.rhs[q]) for q in sorted(P.rhs)]).encode())
    h.update(repr([(w['kind'], w['nom']) for w in P.ws]+[p['nom'] for p in P.ps]).encode())
    return h.hexdigest()[:16]


def settle(P, hours=60., tol=1e-7, verbose=False):
    """Run the plant to rest from its rough design point at the nominal inputs, then finish with Newton. The
    design point is not a state the plant could be in (a cold feed on a tray that was drawn as boiling), so
    the first decisions are taken in many short substeps: twice as many wherever a decision fails or moves a
    state by more than half its scale, half as many once it moves little. Returns the steady state (a list)
    and the largest scaled residual."""
    M = Mech([P], n=1, dtype=torch.float64)
    x, w, p = M.x0.clone(), M.wnom.clone(), M.pnom.clone()
    h = M.h[0].item()*P.time_unit/3600.
    S, t, res = 64, 0., math.inf
    while t < hours:
        xn = M.advance(x, w, p, substeps=S, refresh=True)
        move = float(((xn-x)/M.xscale).abs().max()) if bool(torch.isfinite(xn).all()) else math.inf
        if move > .5 and S < 4096:
            S *= 2
            continue
        x, t = xn, t+h
        if move < .05 and S > 1:
            S //= 2
        res = float((M.f(x, w, p)/M.xscale).abs().max())
        if verbose and round(t/h) % 60 == 0:
            print(P.name, f'{t:.1f} h, substeps {S}, residual {res:.2e}')
        if res < tol:
            break
    if not math.isfinite(res):
        raise RuntimeError(f'{P.name}: the design point does not settle')
    xs, _, ok = M.steady(w, p, x0=x, tol=1e-9)
    x = xs if bool(ok.all()) else x
    return x[0].tolist(), float((M.f(x, w, p)/M.xscale).abs().max())


def _stored():
    return json.loads(NOMINAL.read_text()) if NOMINAL.exists() else {}


def _store(name, got):
    try:
        data = _stored()
        data[name] = got
        NOMINAL.write_text(json.dumps(data, indent=0, sort_keys=True)+'\n')
    except OSError:                                              # an installed copy that cannot be written: computed again next time
        pass


def _apply(P, got):
    for s, v in zip(P.xs, got['x']):
        s['x0'] = float(v)
    for q, v in zip(P.ws, got.get('w', ())):
        q['nom'] = float(v)


def nominal(P):
    """Put the plant's nominal steady state into its states' x0: the stored one where it belongs to this very
    table, else by `settle` (written back where the file can be written)."""
    sig, got = signature(P), _stored().get(P.name)
    if got is None or got['sig'] != sig or len(got['x']) != P.nx:
        x, res = settle(P)
        got = dict(sig=sig, residual=res, x=x)
        _store(P.name, got)
    _apply(P, got)
    return got


def trim(P, x, free, rows, targets):
    """From the steady state x at the nominal inputs: the inputs `free` (names) that put the outputs `rows` at
    `targets`, and the steady state there (the design point, on specification)."""
    M = Mech([P], n=1, dtype=torch.float64)
    wi, ri = [[q['name'] for q in P.ws].index(v) for v in free], [[o['name'] for o in P.ys].index(v) for v in rows]
    t = lambda v: torch.tensor([v], dtype=torch.float64)          # noqa: E731
    xs, w, ok = M.steady(M.wnom, M.pnom, free=wi, rows=ri, targets=t(list(targets)), band=t([P.ys[i]['band'] for i in ri]), start=(t(x), M.wnom, M.pnom))
    return xs, w, M, bool(ok.all())                              # not ok: as far towards the targets as the equipment goes


def probe(P, exprs, x, w=None, p=None):
    """Values (n, len(exprs)) of any expressions of the plant at the states x (n, nx): for checks and plots."""
    Q = copy.copy(P)
    Q.ys, Q._tab = [dict(name=str(i), node=(e.i if isinstance(e, E) else P.g.const(e))) for i, e in enumerate(exprs)], {}
    M = Mech([Q], n=x.shape[0], dtype=torch.float64)
    return M.y(x.double(), M.wnom if w is None else w.double(), M.pnom if p is None else p.double())


# ---------------------------------------------------------------- distillation
def column(name, pkg, nstage, nfeed, F, z, TF, P, D, L, lk, hk, spec=(2., 2.), design=True, **kw):
    """The column of `_column`, designed: sized for the reflux L and distillate D given, run to rest, its three
    valves trimmed until both products are on specification (`spec`: -log10 of the impurity at the top and at
    the bottom) at the pressure P; if the flows that takes are more than 5 % from the ones it was sized for,
    it is sized again for them. The nominal point of the plant is that steady state."""
    if not design:
        return _column(name, pkg, nstage, nfeed, F, z, TF, P, D, L, lk, hk, **kw)
    key = hashlib.sha1(repr((json.dumps(pkg.d, sort_keys=True), nstage, nfeed, F, z, TF, P, D, L, lk, hk, spec, sorted(kw.items()))).encode()).hexdigest()[:16]
    got = _stored().get(name)
    if got is not None and got.get('key') == key:
        Pl = _column(name, pkg, nstage, nfeed, F, z, TF, P, got['D'], got['L'], lk, hk, **kw)
        if signature(Pl) == got['sig']:
            _apply(Pl, got)
            return Pl
    for _ in range(8):
        Pl = _column(name, pkg, nstage, nfeed, F, z, TF, P, D, L, lk, hk, **kw)
        sig = signature(Pl)
        x, _ = settle(Pl)
        xs, w, M, ok = trim(Pl, x, ('reflux', 'steam', 'cooling'), ('yD', 'yB', 'P'), (spec[0], spec[1], P))
        y = dict(zip([o['name'] for o in Pl.ys], M.y(xs, w, M.pnom)[0].tolist()))
        if ok and abs(y['L']/L-1) < .05 and abs(y['D']/D-1) < .05:
            break
        L, D = (y['L'] if ok else max(y['L'], 1.15*L)), y['D']    # a valve or an exchanger ran out before the specification: sized for more
    else:
        raise RuntimeError(f'{name}: no design puts both products on specification')
    got = dict(key=key, sig=sig, D=D, L=L, x=xs[0].tolist(), w=w[0].tolist(), residual=float((M.f(xs, w, M.pnom)/M.xscale).abs().max()))
    _store(name, got)
    _apply(Pl, got)
    return Pl


def _column(name, pkg, nstage, nfeed, F, z, TF, P, D, L, lk, hk, dt=60., substeps=2, eff=.7, imp=.01, steam=None, Ph=1000., Tcw=303., bands=None,
            hydraulic=None, tight=False):
    """A distillation column with a total condenser and a reboiler, built of the units above.

    Stages 1..nstage are trays (theoretical ones: each stands for 1/eff real trays in holdup and pressure
    drop), below them the reboiler with the sump, above them the condenser with the reflux drum. Feed F
    (kmol/h, composition z, temperature TF) on stage nfeed; distillate D and reflux L at a top pressure P
    are the design point, lk and hk the key species (the lighter ones leave at the top).

    What moves what: the reflux valve (pumped from the drum), the steam valve of the reboiler (a shell that
    condenses at its own pressure), the cooling water valve of the condenser; the drum and sump levels are
    held by proportional controllers on the product valves. The column pressure is nobody's setpoint: it is
    where condenser, reboiler and tray hydraulics meet.
    """
    st = PP.package('steam') if steam is None else steam
    nc, tu = pkg.nc, 3600.
    fs = Flowsheet(name, dt, time_unit=tu, substeps=substeps, refresh=True)       # a pressure-flow network: its Jacobian moves with every flow
    ilk, ihk = pkg.index(lk), pkg.index(hk)
    # ---- design point (numbers): a sharp split with `imp` of the wrong key in each product
    B = F-D
    light = [i <= ilk for i in range(nc)]
    zl = sum(v for v, q in zip(z, light) if q)
    xd = [(v/zl*(1-imp) if q else 0.) for v, q in zip(z, light)]
    xb = [(0. if q else v/(1-zl)*(1-imp)) for v, q in zip(z, light)]
    xd[ihk], xb[ilk] = imp, imp
    nreal = 1./eff
    V = L+D
    Td = pkg.bubble_T(xd, P)
    yt = xd                                                       # total condenser: the top vapour is the distillate
    rhov = P*pkg.mwm(yt)/(pkg.z(yt)*R*(Td+5.))
    rhol = pkg.mwm(xd)/pkg.vl(xd, Td)
    # tray: active area from the flooding velocity (Souders-Brown, 75 % of flood), holes a tenth of it, weir .73 of the diameter
    uf = .07*math.sqrt((rhol-rhov)/rhov)*(1. if hydraulic is None else hydraulic.get('csb', 1.))
    area = V*pkg.mwm(yt)/rhov/3600./(.75*uf)
    dia = math.sqrt(area/.8*4/math.pi)
    lw, hw, ah = .73*dia, .05, .1*area
    xi = 1.9 if hydraulic is None else hydraulic.get('xi', 1.9)
    dpd = xi*rhov*(V*pkg.mwm(yt)/rhov/3600./ah)**2/2/1e3         # dry drop per real tray at the design load, kPa
    how = lambda q: (q/(1.84*lw))**(2/3)                          # noqa: E731
    Ls, Lr = L+F, L
    xf = list(z)
    dpt = lambda x, Lq, T: nreal*(dpd+.6*GRAV*(hw+how(Lq*pkg.vl(x, T)/3600.))*pkg.mwm(x)/pkg.vl(x, T)/1e3)     # noqa: E731
    dp_r, dp_s = dpt(xd, Lr, Td), dpt(xb, Ls, Td+40.)
    dp_line = 5. if hydraulic is None else hydraulic.get('line', 5.)
    Pt = P+dp_line                                                # top tray
    Pb = Pt+dp_r*(nstage-nfeed)+dp_s*nfeed
    Tb = pkg.bubble_T(xb, Pb)
    hF = pkg.hl(xf, TF)
    Qc = V*(pkg.hv(yt, Td+5., P)-pkg.hl(xd, Td))
    Qr = Qc+D*pkg.hl(xd, Td)+B*pkg.hl(xb, Tb)-F*hF
    Vb = Qr/(pkg.hv(xb, Tb, Pb)-pkg.hl(xb, Tb))

    # ---- inputs
    xl = fs.mv('reflux', 0., 1., .02, .55)
    xs = fs.mv('steam', 0., 1., .02, .6)
    xc = fs.mv('cooling', 0., 1., .02, .6)
    Fl = fs.load('F', F, .8*F, 1.2*F, measured=True)
    dz = fs.load('zF', 0., -.06, .06)                             # light key and lighter for heavy key and heavier
    Tf = fs.load('TF', TF, TF-12., TF+12.)
    Tw = fs.load('Tcw', Tcw, Tcw-8., Tcw+6.)
    Phd = fs.load('Psteam', Ph, .9*Ph, 1.08*Ph)
    uar, uac = fs.param('ua_reb', 1., .7, 1.05), fs.param('ua_cond', 1., .7, 1.05)
    ktr = fs.param('tray_dp', 1., .9, 1.4, event=False)           # fouled or damaged trays: more pressure drop for the same load
    kv = fs.param('volatility', 1., .94, 1.06, event=False)       # the light key's volatility against the model's: property models differ by this much (docs/validation.md)
    cvl = fs.param('cv_reflux', 1., .9, 1.1)
    kfac = [kv if i == ilk else 1. for i in range(nc)]

    # ---- holdups: sump, trays (from the bottom), drum
    def prof(j):                                                  # the rough profile: compositions between the products, boiling at a straight pressure line
        a = j/(nstage+1.)
        lx = [math.log(max(u, 1e-6))*(1-a)+math.log(max(v, 1e-6))*a for u, v in zip(xb, xd)]
        x = [math.exp(v) for v in lx]
        s = sum(x)
        x = [v/s for v in x]
        Pj = Pb-(Pb-Pt)*(j-0.)/nstage if j <= nstage else P
        return x, pkg.bubble_T(x, Pj), Pj
    Md, Mb = (L+D)*5./60., (B+Vb)*4./60.                          # five minutes of what leaves the drum, four of the sump, at half level
    vd, vb = 2*Md*pkg.vl(xd, Td), 2*Mb*pkg.vl(xb, Tb)
    x0, T0, _ = prof(0)
    sump = fs.liquid(pkg, 'sump', [Mb*v for v in x0], T0, kfac)
    trays = []
    for j in range(1, nstage+1):
        x0, T0, _ = prof(j)
        Lq = Ls if j <= nfeed else Lr
        m = nreal*area*(hw+how(Lq*pkg.vl(x0, T0)/3600.))/pkg.vl(x0, T0)
        trays.append(fs.liquid(pkg, f't{j}', [m*v for v in x0], T0, kfac))
    drum = fs.liquid(pkg, 'drum', [Md*v for v in xd], Td, kfac)
    area_s = nreal*area                                           # a stage holds the liquid of 1/eff trays; its weir passes the liquid of one

    # ---- flows
    cf = 1./(nreal*ktr)**.5 if isinstance(ktr, float) else exp(-.5*log(nreal*ktr))      # 1/eff trays in series per stage: the same flow at 1/eff times the drop
    d0 = .02*nreal*dpd                                            # the orifice law is linear below 2 % of the design drop of a stage
    lows = [sump]+trays
    Vs, Lq = [], []
    for j in range(nstage):                                       # vapour from stage j up to tray j+1
        lo, up = lows[j], trays[j]
        Vs.append(fs.move(lo.vapour(holes(fs, lo, up, ah, xi, head=wet(up, area_s)*nreal, cf=cf, d0=d0)), lo, up).F)
    top = trays[-1]
    cline = V/math.sqrt(dp_line*P/(pkg.z(yt)*R*(Td+5.))/pkg.mwm(yt))
    Vs.append(fs.move(top.vapour(cline*sqrt(top.rhov/top.mwv)*root(top.P-drum.P, .02*dp_line)), top, drum).F)
    for j in range(nstage):                                       # liquid from tray j+1 down
        t = trays[j]
        Lq.append(fs.move(t.liquid(weir(fs, t, area_s, lw, hw)), t, lows[j]).F)
    # reflux: pumped from the drum to the top tray (static lift in the pump head)
    head_r = 300.
    cv_r = L/.55/math.sqrt((head_r-(Pt-P))/(1.+.25))             # the pump loses a fifth of its head at the design flow
    kq_r = .25/(cv_r*.55)**2
    Lf = pumped(cv_r*cvl, xl, top.P-drum.P, head_r, kq_r, d0=.01*head_r)
    fs.move(drum.liquid(Lf), drum, top)
    Lq.append(Lf)
    # products on level: proportional, valve half open at the design flow, full travel over the level range
    lvd, lvb = drum.vol/vd, sump.vol/vb
    kl = 4. if tight else 2.
    Df = (2*D)*clamp(.5+kl*(lvd-.5), 0., 1.)
    Bf = (2*B)*clamp(.5+kl*(lvb-.5), 0., 1.)
    fs.move(drum.liquid(Df), drum)
    fs.move(sump.liquid(Bf), sump)
    # feed: z shifted between the keys' sides by the load
    zl_, zh_ = zl, 1-zl
    zf = [(v*(1+dz/zl_) if q else v*(1-dz/zh_)) for v, q in zip(z, light)]
    fs.move(Stream(Fl, zf, pkg.hl(zf, Tf)), None, trays[nfeed-1])
    # reboiler: steam condensing in a shell at its own pressure
    dTr = 25.
    Ts0 = Tb+dTr
    Ps0 = float(st.bubble([1.], Ts0)[0])
    Ps0 = min(Ps0, .8*Ph)
    Ts0 = float(st.bubble_T([1.], Ps0))
    ua_r = Qr/(Ts0-Tb)
    latent = lambda T, P: st.hv([1.], T, P)-st.hl([1.], T)       # noqa: E731  (of steam condensing at the shell's own pressure)
    Fs0 = Qr/float(latent(Ts0, Ps0))
    cv_s = Fs0/.6/math.sqrt(Ph-Ps0)
    Ps = fs.state('shell.P', Ps0, lo=1., hi=3000., scale=Ps0)
    Tsh = tsat(st, Ps)
    Qreb = (ua_r*uar)*relu(Tsh-sump.T)
    fs.P.der(Ps, (R*Ts0/2.)*(valve(cv_s, xs, Phd-Ps, d0=.01*Ph)-Qreb/latent(Tsh, Ps)))     # a shell of 2 m3 of steam
    sump.heat(Qreb)
    # condenser on cooling water
    dTw = 10.
    Fw0 = Qc/(CPW*dTw)
    eps = dTw/(Td-Tcw)
    ua_c = -math.log(1-eps)*Fw0*CPW
    Fcw = (Fw0/.6)*maximum(xc, .02)
    Qcond = cooler(Fcw, drum.T, Tw, ua_c*uac, Fw0)
    drum.heat(-Qcond)

    # ---- outputs
    b = dict(y=.1, P=.01*P, dp=.5, T=1., L=.03)
    b.update(bands or {})
    LN10 = math.log(10.)
    heavy_top = sum([drum.x[i] for i in range(nc) if not light[i]][1:], [drum.x[i] for i in range(nc) if not light[i]][0])
    light_bot = sum([sump.x[i] for i in range(nc) if light[i]][1:], [sump.x[i] for i in range(nc) if light[i]][0])
    fs.out('yD', -log(clamp(heavy_top, 1e-9, 1.))/LN10, range=(1., 3.5), band=b['y'], track=True, target=(1.5, 2.6))
    fs.out('yB', -log(clamp(light_bot, 1e-9, 1.))/LN10, range=(1., 3.5), band=b['y'], track=True, target=(1.5, 2.6))
    fs.out('P', top.P, range=(.8*P, 1.2*P), band=b['P'], track=True, target=(.92*P, 1.08*P))
    dpn = Pb-Pt
    fs.out('dP', sump.P-top.P, range=(0., 2.5*dpn), band=.02*dpn, hi=1.5*dpn)       # flooding shows as pressure drop
    fs.out('TB', sump.T, range=(Tb-30., Tb+30.), band=b['T'], hi=Tb+12.)
    fs.out('LD', lvd, range=(0., 1.), band=b['L'], lo=.2, hi=.8)
    fs.out('LB', lvb, range=(0., 1.), band=b['L'], lo=.2, hi=.8)
    fs.out('F', Fl, range=(.7*F, 1.3*F), band=.01*F, dv=True)
    # what a replay or a check wants to see (not rows of the task: no limits)
    fs.out('L', Lf, range=(0., 2.5*L), band=.01*L)
    fs.out('D', Df, range=(0., 2*D), band=.01*D)
    fs.out('B', Bf, range=(0., 2*B), band=.01*B)
    fs.out('Qr', Qreb, range=(0., 2.5*Qr), band=.01*Qr)
    fs.out('Qc', Qcond, range=(0., 2.5*Qc), band=.01*Qc)
    fs.out('Vb', Vs[0], range=(0., 2.5*Vb), band=.01*Vb)
    for j in (nstage//4, nfeed, 3*nstage//4):
        fs.out(f'T{j}', trays[j-1].T, range=(Td-20., Tb+20.), band=b['T'])
    P_ = fs.close(settle=False)
    P_.design = dict(F=F, D=D, B=B, L=L, V=V, Vb=Vb, Qc=Qc, Qr=Qr, P=P, Pt=Pt, Pb=Pb, Td=Td, Tb=Tb, area=area, dia=dia, lw=lw, hw=hw, ah=ah, dp_tray=(dp_r+dp_s)/2/nreal,
                     drum=vd, sump=vb, ua_reb=ua_r, ua_cond=ua_c, cv_steam=cv_s, cv_reflux=cv_r, cw=Fw0/.6, light=light, lk=ilk, hk=ihk, nstage=nstage, nfeed=nfeed)
    P_.units = dict(stages=[sump]+trays+[drum], V=Vs, L=Lq, D=Df, B=Bf, feed=(Fl, zf, pkg.hl(zf, Tf)), Qr=Qreb, Qc=Qcond)       # V[j]: vapour leaving stage j; L[j]: liquid entering stage j from above
    return P_

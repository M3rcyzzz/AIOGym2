"""Record trajectories of the native vinyl acetate plant for aio.sim.validate.vam.

Standalone (numpy and the native library only):

    python vam_native_record.py build <dir with the C routines> <new build dir>     # MSVC, Windows
    python vam_native_record.py record <vam.dll> <out.npz>

The C routines are those of Chen, Dave, McAvoy and Luyben (terpconnect.umd.edu/~mcavoy/VAC Material/C routines,
no licence stated: kept out of the repository). The build copies them unchanged, cuts the MEX gateway off
VAModel.c and adds vam_native_adapter.c. Integration as in their test_VAcPlant.m: explicit Euler at 1/3 s.

Recorded, no noise, a sample a minute: the 246 states and the 43 measurements.
Open loop (MVs held): the base, each MV up 2 % (the jacket temperatures 1 C), the paper's feed disturbances (C2H6 in the ethylene feed,
column feed lost 5 min, HAc and O2 feeds lost 5 min). Closed loop: the regulatory layer of
aio.sim.plants_vam.vam written again in plain Python and run once a second (Luyben's loops as continuous PI,
the three analysers as ten-stage lags of 10 min in all), setpoint steps and the disturbances ten minutes in.
"""
import ctypes as ct
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

H = 1/180.                                   # min: the step of test_VAcPlant.m
NX, NU, NY = 246, 26, 43
SKIP = ('controller.c', 'Transmit.c')        # MATLAB-side controller and transmitter, not the plant


def build(src, out):
    src, out = Path(src).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for p in src.glob('*.c'):
        if p.name in SKIP:
            continue
        t = p.read_text(encoding='latin1')
        if p.name == 'VAModel.c':
            t = t.replace('#include "mex.h"', '')
            t = t[:t.index('void mexFunction')]
        (out/p.name).write_text(t, encoding='latin1')
    shutil.copy(Path(__file__).with_name('vam_native_adapter.c'), out)
    vswhere = 'C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe'
    vs = subprocess.check_output([vswhere, '-latest', '-products', '*', '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
                                  '-property', 'installationPath'], text=True).strip()
    setup = Path(vs)/'VC/Auxiliary/Build/vcvars64.bat'
    (out/'compile.bat').write_text(f'@echo off\ncall "{setup}" >nul\nif errorlevel 1 exit /b 1\n'
                                   'cl /nologo /LD /O2 /fp:strict /FIstdlib.h *.c /link /OUT:vam.dll\n')
    r = subprocess.run(['cmd.exe', '/d', '/c', str(out/'compile.bat')], cwd=out, capture_output=True)
    (out/'compiler.log').write_bytes(r.stdout+r.stderr)
    r.check_returncode()
    print('built', out/'vam.dll')


class Native:
    def __init__(self, dll):
        self.lib = lib = ct.CDLL(str(dll))
        d = ct.POINTER(ct.c_double)
        lib.vam_init.argtypes = [d, d]
        lib.vam_f.argtypes = [d, d, ct.c_double, ct.c_int, d, d]
        lib.vam_euler.argtypes = [d, d, ct.c_longlong, ct.c_int, ct.c_double, ct.c_int, d]

    @staticmethod
    def _p(a):
        return a.ctypes.data_as(ct.POINTER(ct.c_double))

    def init(self):
        x, u = np.zeros(NX), np.zeros(NU)
        self.lib.vam_init(self._p(x), self._p(u))
        return x, u

    def f(self, x, u, t=0., dist=0):
        x, u = np.ascontiguousarray(x, dtype=float), np.ascontiguousarray(u, dtype=float)
        dx, y = np.zeros(NX), np.zeros(NY)
        self.lib.vam_f(self._p(x), self._p(u), float(t), int(dist), self._p(dx), self._p(y))
        return dx, y

    def euler(self, x, u, k0, n, dist=0):
        """x advanced in place by n steps from step k0; the measurements of the last evaluation."""
        u, y = np.ascontiguousarray(u, dtype=float), np.zeros(NY)
        self.lib.vam_euler(self._p(x), self._p(u), int(k0), int(n), H, int(dist), self._p(y))
        return y


# ---------------------------------------------------------------- open loop
STEP = {6: 1., 8: 1.}                       # the two jacket temperatures move by 1 C, every other MV by 2 %
OPEN = [('ol_base', {}, {}, 60)]
OPEN += [(f'ol_u{j+1}', {j: STEP.get(j)}, {}, 20) for j in range(NU)]
OPEN += [('ol_c2h6', {}, dict(dist=1), 30), ('ol_colfeed', {}, dict(dist=2, until=5), 30),
         ('ol_hac', {}, dict(zero=2, until=5), 30), ('ol_o2', {}, dict(zero=0, until=5), 30)]
EVERY_OL = 2                                # min between samples


def open_loop(N, moves, ev, minutes):
    x, u = N.init()
    for j, d in moves.items():
        u[j] = u[j]+d if d is not None else u[j]*1.02
    X, Y = [x.copy()], [N.f(x, u)[1]]
    for m in range(minutes):
        on = m < ev.get('until', minutes)
        uu = u.copy()
        if on and 'zero' in ev:
            uu[ev['zero']] = 0.
        N.euler(x, uu, m*180, 180, ev.get('dist', 0) if on else 0)
        if (m+1) % EVERY_OL == 0:
            X.append(x.copy()); Y.append(N.f(x, uu, (m+1)*1., ev.get('dist', 0) if on else 0)[1])
    return dict(y=np.array(X, dtype=np.float32), meas=np.array(Y, dtype=np.float32), u=u, t=np.arange(len(X))*EVERY_OL)


# ---------------------------------------------------------------- closed loop: the layer of aio.sim.plants_vam.vam at one second
# name, MV index, measurement, setpoint (name or number), K, Ti (0: P), action, y range, u range (plants_vam.LOOPS)
LOOPS = (('O2', 0, 36, 'O2', 5., 10., 1, (0., .2), (0., 2.268)), ('Prec', 1, 11, 'Prec', .3, 20., 1, (0., 200.), (0., 7.56)),
         ('Ltank', 2, 22, .5, 2., 0., 1, (0., 1.), (0., 4.536)), ('Lvap', 3, 1, .7, .1, 30., -1, (0., 1.), (0., 1433400.)),
         ('Pvap', 4, 0, 'P_rec', 5., 10., -1, (0., 200.), (0., 50.)), ('Theater', 5, 3, 150., 1., 5., 1, (120., 170.), (0., 15000.)),
         ('Trct', 6, 4, 'Trct', 3., 10., 1, (0., 200.), (110., 150.)), ('Lsep', 7, 8, .5, 5., 0., -1, (0., 1.), (0., 4.536)),
         ('Tsep', 8, 9, 'Tsep', 5., 20., -1, (0., 80.), (0., 80.)), ('Tcomp', 10, 10, 80., 1., 5., -1, (70., 90.), (0., 50000.)),
         ('Labs', 11, 12, .5, 5., 0., -1, (0., 1.), (0., 4.536)), ('Tcirc', 13, 13, 25., 1., 5., -1, (10., 40.), (0., 30000.)),
         ('Tscrub', 15, 14, 25., 1., 5., -1, (10., 40.), (0., 5000.)), ('CO2', 16, 30, 'CO2', 1., 0., -1, (0., .5), (0., 22.68)),
         ('C2H6', 17, 32, 'C2H6', 1., 0., -1, (0., 1.), (0., .02268)), ('Tfehe', 18, 7, 'Tfehe', 5., 10., 1, (0., 200.), (0., 1.)),
         ('H2Ob', 19, 27, 'H2Ob', .5, 60., -1, (0., .2), (0., 7.56)), ('Ttray', 20, 21, 'Tcol', 20., 30., 1, (0., 120.), (0., 100000.)),
         ('Tdec', 21, 19, 45.845, 1., 5., -1, (40., 50.), (0., 150000.)), ('Lorg', 22, 17, .5, 1., 0., -1, (0., 1.), (0., 2.4)),
         ('Laq', 23, 18, .5, 1., 0., -1, (0., 1.), (0., 2.4)), ('Lbot', 24, 20, .5, 1., 0., -1, (0., 1.), (0., 4.536)))
FIXED = {9: 'Fgas', 12: 'Fcirc', 14: 'Fscrub', 25: 'Fvl'}
ANALYSED, STAGES, DEAD, LAG = (30, 32, 27), 10, 10., .05
SP0 = dict(Fvl=2.1924, Trct=159.17, O2=.075, Prec=128., Fgas=16.1026, H2Ob=.09344, Tcol=110., C2H6=.25, CO2=.0076393, Tsep=40., Tfehe=134.,
           Fcirc=15.1198, Fscrub=.756)
EVENT, HOURS, EVERY_CL = 10, 6, 5           # min; h; min between samples
CLOSED = [('cl_base', {}, {}), ('cl_Trct151', dict(Trct=151.), {}), ('cl_Trct165', dict(Trct=165.), {}), ('cl_H2Ob.18', dict(H2Ob=.18), {}),
          ('cl_Fvl2.64', dict(Fvl=2.64), {}), ('cl_O2.07', dict(O2=.07), {}), ('cl_Prec120', dict(Prec=120.), {}), ('cl_Fgas17', dict(Fgas=17.), {}),
          ('cl_Tcol108', dict(Tcol=108.), {}), ('cl_C2H6.30', dict(C2H6=.30), {}),
          ('cl_c2h6', {}, dict(dist=1)), ('cl_colfeed', {}, dict(dist=2, until=5)), ('cl_hac', {}, dict(zero=2, until=5)), ('cl_o2', {}, dict(zero=0, until=5))]


class Layer:
    """Transmitters (3 s lag), analysers (ten stages, 10 min), the loops as continuous PI with a held integral: explicit at h min."""

    def __init__(self, y0, u0):
        clip = lambda v, r: min(max(v, r[0]), r[1])                                       # noqa: E731
        self.y0, self.u0 = y0, u0
        self.tx = {nm: clip(y0[j], yr) for nm, k, j, *_r, yr, ur in LOOPS}
        self.an = {j: [y0[j]]*STAGES for j in ANALYSED}
        self.I = {nm: 0. for nm, k, j, sp, K, Ti, *_r in LOOPS if Ti}
        self.clip = clip

    def u(self, sp):
        u = np.array(self.u0, dtype=float)
        for i, q in FIXED.items():
            u[i] = sp[q]
        self.e = {}
        for nm, k, j, spq, K, Ti, act, yr, ur in LOOPS:
            s, s0 = (self.tx['Prec'], self.y0[11]) if spq == 'P_rec' else (sp[spq], SP0[spq]) if isinstance(spq, str) else (spq, spq)
            kk = K*act
            e = (s-self.tx[nm])/(yr[1]-yr[0])
            us = (self.u0[k]-ur[0])/(ur[1]-ur[0])-kk*(s0-self.clip(self.y0[j], yr))/(yr[1]-yr[0])+kk*(e+(self.I[nm]/Ti if Ti else 0.))
            if Ti:
                self.e[nm] = 0. if (us > 1 and kk*e > 0) or (us < 0 and kk*e < 0) else e
            u[k] = ur[0]+(ur[1]-ur[0])*min(max(us, 0.), 1.)
        return u

    def step(self, y, h):
        for nm in self.I:
            self.I[nm] += self.e[nm]*h
        for j, z in self.an.items():
            prev = [y[j]]+z[:-1]
            self.an[j] = [a+(p-a)*h/(DEAD/STAGES) for a, p in zip(z, prev)]
        for nm, k, j, *_r, yr, ur in LOOPS:
            v = self.an[j][-1] if j in self.an else y[j]
            self.tx[nm] += (self.clip(v, yr)-self.tx[nm])*h/LAG

    def state(self):
        """In the order of plants_vam.vam's states after the process: transmitters, analysers, integrals."""
        return list(self.tx.values())+[a for j in ANALYSED for a in self.an[j]]+list(self.I.values())


def closed_loop(N, change, ev):
    x, u0 = N.init()
    y0 = N.f(x, u0)[1]
    lay, sp = Layer(y0, u0), dict(SP0)
    X, U = [np.r_[x, lay.state()]], [lay.u(sp)]
    for s in range(HOURS*3600):
        m = s/60.
        if m >= EVENT:
            sp.update(change)
        on = EVENT <= m < EVENT+ev.get('until', 1e9)
        u = lay.u(sp)
        uu = u.copy()
        if on and 'zero' in ev:
            uu[ev['zero']] = 0.
        y = N.euler(x, uu, 0, 3, ev.get("dist", 0) if on else 0)       # one second; the catalyst does not age: vam() has its activity as a parameter
        lay.step(y, 1/60.)
        if (s+1) % (EVERY_CL*60) == 0:
            X.append(np.r_[x, lay.state()]); U.append(u)
    return dict(y=np.array(X, dtype=np.float32), u=np.array(U, dtype=np.float32), t=np.arange(len(X))*EVERY_CL)


def _case(dll, kind, spec, out):
    N = Native(dll)
    r = open_loop(N, *spec) if kind == 'ol' else closed_loop(N, *spec)
    np.savez(out, **r)


def record(dll, out, timeout=120):
    """Each case in a process of its own: the code's flash loop (Separator.c) never ends on some states far from
    the base; such a case is left out and named."""
    import multiprocessing as mp
    import tempfile
    data, tmp = {}, Path(tempfile.mkdtemp())
    for kind, cases in (('ol', OPEN), ('cl', CLOSED)):
        for name, *spec in cases:
            f = tmp/f'{name}.npz'
            p = mp.Process(target=_case, args=(dll, kind, spec, str(f)))
            p.start(); p.join(timeout)
            if p.is_alive():
                p.terminate(); p.join()
            if not f.exists():
                print(name, 'left out: the native code did not finish', flush=True)
                continue
            z = np.load(f)
            data.update({f'{name}/{k}': z[k] for k in z.files})
            print(name, flush=True)
    np.savez_compressed(out, **data)
    print('wrote', out)


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build(sys.argv[2], sys.argv[3])
    else:
        record(sys.argv[2], sys.argv[3])

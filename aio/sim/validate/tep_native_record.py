"""Record trajectories of the native Tennessee Eastman code for aio.sim.validate.tep.

Standalone (numpy and the native library only), to run on the machine that has the build:

    python tep_native_record.py <te.dll> <out.npz>

The library is the project's wrapper around the revised Tennessee Eastman C code
(tep_native_adapter.c, built by tep_native_build.py: RK4 at one second, measurements with their noise once a
second). What is recorded is the routine's own 50 states, read from the saved handle, so the comparison
carries no measurement noise; the noisy XMEAS are kept as window means for the measurement equations.

Open loop (valves held, one seed: the states do not depend on it): the base case until the plant trips,
each valve moved by five points, IDV 1-7. Closed loop (three seeds, mean): the regulatory layer of
aio.sim.plants_tep.tep written again in plain Python, run once a second on the noisy measurements; steps
in the APC inputs and IDV 1-5, 7 an hour into a day.
"""
import ctypes as ct
import sys

import numpy as np

# the base case: valves (the routine's states 39-50) and XMEAS 1-41 without noise (aio.sim.plants_tep.base)
U0 = (63.05263039, 53.97970677, 24.64355755, 61.30192144, 22.21, 40.06374673, 38.1003437, 46.53415582, 47.44573456, 41.10581288, 18.11349055, 50.)
XM0 = (0.25052137678432373, 3664.038884863, 4509.268390168219, 9.34774234250622, 26.902440399615, 42.33884885634592, 2705.0000000464083,
       74.99999998088552, 120.40000000169141, 0.337117051302681, 80.10940225933267, 49.99999999568061, 2633.727883126596, 25.160173384912884,
       50.00000000449063, 3102.2131372370486, 22.94873008984307, 65.73102903414427, 230.30782040479613, 341.43365084541847, 94.59927549,
       77.29698353, 32.18761309449962, 8.893272645564029, 26.383289433228583, 6.882011757270133, 18.776305947424234, 1.656727356649244,
       32.957537290841856, 13.82294586738243, 23.978144239837704, 1.2565424846624529, 18.579269702499033, 2.2633413851447473,
       4.8436409545196595, 2.2985780751121134, 0.017865524957181182, 0.8357022623245169, 0.09857691352349168, 53.72397878932856,
       43.82802189355887)
MIN = 1/60.
TUNE = dict(fc=(.5, 10/3600.), tr=(-8., 7.5*MIN), ts=(-4., 15*MIN), lr=(.8, 60*MIN), ls=(-.4, 120*MIN), lc=(-.3, 120*MIN), pr=(-1., 20*MIN))
MV0 = dict(Fp=100., Eadj=0., rA=XM0[0], rC=XM0[3], Psp=XM0[6], Tsp=XM0[8], Lsp=XM0[7], recycle=U0[4], steam=U0[8], agitator=U0[11])

OPEN = [('ol_base', {}, 0, 3600, 10)]
OPEN += [(f'ol_v{j+1}', {j: 5.}, 0, 900, 15) for j in range(12)]
OPEN += [(f'ol_idv{k}', {}, k, 1800, 30) for k in range(1, 8)]
EVENT, DAY, EVERY, SEEDS = 3600, 86400, 600, (1, 2, 3)
CLOSED = [('cl_base', {}, 0), ('cl_Fp-15', dict(Fp=85.), 0), ('cl_Eadj+10', dict(Eadj=10.), 0), ('cl_rA+.05', dict(rA=XM0[0]+.05), 0),
          ('cl_rC-.3', dict(rC=XM0[3]-.3), 0), ('cl_Psp-60', dict(Psp=XM0[6]-60.), 0), ('cl_Tsp+2', dict(Tsp=XM0[8]+2.), 0),
          ('cl_Lsp-5', dict(Lsp=XM0[7]-5.), 0), ('cl_recycle+10', dict(recycle=U0[4]+10.), 0), ('cl_steam+10', dict(steam=U0[8]+10.), 0),
          ('cl_idv1', {}, 1), ('cl_idv2', {}, 2), ('cl_idv3', {}, 3), ('cl_idv4', {}, 4), ('cl_idv5', {}, 5), ('cl_idv7', {}, 7)]


class Native:
    def __init__(self, dll, seed):
        self.lib = lib = ct.CDLL(str(dll))
        lib.te_new.argtypes = [ct.c_double]; lib.te_new.restype = ct.c_void_p
        lib.te_free.argtypes = [ct.c_void_p]
        lib.te_save.argtypes = [ct.c_void_p, ct.c_void_p]
        lib.te_read.argtypes = [ct.c_void_p, ct.c_void_p]
        lib.te_advance.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int]
        self.size = lib.te_size()
        self.h = lib.te_new(float(seed))
        self.buf = ct.create_string_buffer(self.size)
        self.out = np.empty(72)
        self.at = self._offset()

    def raw(self):
        self.lib.te_save(self.h, self.buf)
        return np.frombuffer(self.buf.raw[:self.size-self.size % 8], dtype=np.float64)

    def _offset(self):
        """Where the 50 states sit in the saved handle: followed by the clock, and their last twelve are the valve positions te_read gives."""
        r, o = self.raw(), self.read()
        hits = [i for i in range(len(r)-51) if np.array_equal(r[i+38:i+50], o[41:53]) and r[i+50] == o[65] and abs(r[i]-10.40491389) < 1e-6]
        assert len(hits) == 1, hits
        return hits[0]

    def read(self):
        self.lib.te_read(self.h, self.out.ctypes.data)
        return self.out.copy()

    def state(self):
        return self.raw()[self.at:self.at+50].copy()

    def advance(self, u, seconds=1, fault=0):
        a = np.ascontiguousarray(u, dtype=np.float64)
        return self.lib.te_advance(self.h, a.ctypes.data, 0, int(seconds), 1, int(fault))

    def close(self):
        self.lib.te_free(self.h)


class Layer:
    """aio.sim.plants_tep.tep's regulatory layer, explicit at one second."""
    def __init__(self):
        self.I = {}

    def pi(self, k, e, K, Ti, bias, lo, hi, dt):
        i = self.I.get(k, 0.)
        raw = bias+K*e+K/Ti*i
        if not ((raw > hi and K*e > 0) or (raw < lo and K*e < 0)):
            self.I[k] = i+e*dt
        return min(max(raw, lo), hi)

    def __call__(self, xm, mv, dt=1/3600.):
        fp = mv['Fp']/100.
        sp = (XM0[1]*fp-32.*mv['Eadj'], XM0[2]*fp+46.*mv['Eadj'], mv['rA']*fp, mv['rC']*fp)
        u = [0.]*12
        kf, tf = TUNE['fc']
        for j, (valve, meas) in enumerate(((0, 1), (1, 2), (2, 0), (3, 3))):
            gain = XM0[meas]/U0[valve]
            u[valve] = self.pi(f'f{valve+1}', sp[j]-xm[meas], kf/gain, tf, U0[valve], 0., 100., dt)
        u[9] = self.pi('tr', mv['Tsp']-xm[8], *TUNE['tr'], U0[9], 0., 100., dt)
        tsp = self.pi('lr', mv['Lsp']-xm[7], *TUNE['lr'], XM0[10], 0., 120., dt)
        u[10] = self.pi('ts', tsp-xm[10], *TUNE['ts'], U0[10], 0., 100., dt)
        u[6] = self.pi('ls', 50.-xm[11], *TUNE['ls'], U0[6], 0., 100., dt)
        u[7] = self.pi('lc', 50.-xm[14], *TUNE['lc'], U0[7], 0., 100., dt)
        u[5] = self.pi('pr', mv['Psp']-xm[6], *TUNE['pr'], U0[5], 0., 100., dt)
        u[4], u[8], u[11] = mv['recycle'], mv['steam'], mv['agitator']
        return u


def open_loop(dll, moves, fault, seconds, every):
    p = Native(dll, 1)
    u = np.array(U0)
    for j, d in moves.items():
        u[j] += d
    t, Y, XM, acc, stop = [0], [p.state()], [p.read()[:41]], [], 0
    for k in range(1, seconds+1):
        code = p.advance(u, 1, fault)
        if code:
            stop = k
            break
        acc.append(p.read()[:41])
        if k % every == 0:
            t.append(k); Y.append(p.state()); XM.append(np.mean(acc, 0)); acc = []
    p.close()
    return dict(t=np.array(t, dtype=np.int32), y=np.array(Y), xm=np.array(XM, dtype=np.float32), u=u, stop=np.int32(stop), fault=np.int32(fault))


def closed_loop(dll, change, fault):
    runs, stops = [], []
    for seed in SEEDS:
        p, lay, mv = Native(dll, seed), Layer(), dict(MV0)
        xm, Y, stop = p.read()[:41], [p.state()], 0
        for k in range(1, DAY+1):
            if k == EVENT+1:
                mv.update(change)
            code = p.advance(lay(xm, mv), 1, fault if k > EVENT else 0)
            if code:
                stop = k
                break
            xm = p.read()[:41]
            if k % EVERY == 0:
                Y.append(p.state())
        p.close()
        runs.append(np.array(Y)); stops.append(stop)
    n = min(len(r) for r in runs)
    R = np.stack([r[:n] for r in runs])
    mv = dict(MV0); mv.update(change)
    return dict(t=np.arange(n, dtype=np.int32)*EVERY, y=R.mean(0).astype(np.float32), spread=R.std(0).max(0).astype(np.float32),
                stop=np.array(stops, dtype=np.int32), fault=np.int32(fault), mv=np.array([mv[q] for q in MV0]))


def main(dll, out):
    data = {}
    for name, moves, fault, seconds, every in OPEN:
        r = open_loop(dll, moves, fault, seconds, every)
        print(name, 'samples', len(r['t']), 'stop', int(r['stop']), flush=True)
        data.update({f'{name}/{k}': v for k, v in r.items()})
    for name, change, fault in CLOSED:
        r = closed_loop(dll, change, fault)
        print(name, 'samples', len(r['t']), 'stop', r['stop'].tolist(), 'spread', float(r['spread'].max()), flush=True)
        data.update({f'{name}/{k}': v for k, v in r.items()})
    data['mv_names'] = np.array(list(MV0))
    np.savez_compressed(out, **data)
    print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

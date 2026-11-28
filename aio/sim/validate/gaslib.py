"""The GasLib-24 network of the engine against morgen (docs/sim.md, 'GasLib').

    python -m aio.sim.validate.gaslib record MORGEN_DIR OCTAVE_CLI TRR154_BCD [out.npz]   # runs morgen, writes the npz
    python -m aio.sim.validate.gaslib [data/gaslib_morgen.npz]                           # the comparison

morgen (github.com/gramian/morgen, BSD-2, Octave) is run with its own network file, model (ode_end), friction,
compressibility, steady state and IMEX2 scheme (30 s: at its 60 s default the explicit friction of the 0.5 m pipe is unstable above 20 kg/s,
and so is its steady state), pieces of 2.4 km (its default); the driver written here only sets the scenario: the five
exits' flows of the TRR154 GasLib-24 transient data (24 h, a minute each), the entry pressures and the compressor
setpoints (morgen holds them constant; the driver steps them where a case does). Compared: the three entry flows
and the five exit pressures, the engine's raw plant from morgen's steady state with morgen's RTz.
"""
import gzip
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

from aio.sim import plants_gaslib as G
from aio.sim.engine import Mech

DATA = Path(__file__).parent/'data'/'gaslib_morgen.npz'
F = torch.float64
HOURS = 24
CP0 = (50., 50., 50.)
# case: compressor setpoint changes (hour, station, bar), entry pressure changes (hour, bar)
CASES = dict(trr154=((), ()), cp_ramps=(((1, 0, 2.), (6, 2, -1.5), (12, 1, 1.)), ()), pin_ramps=((), ((1, -1.), (12, .5))))
RAMP = 30                                                          # min: a case's changes are ramps (morgen's explicit friction does not take steps)
GAP = .005                                                        # what the tests hold the engine to: share of the base value

DRIVER = r'''
cd('%(morgen)s'); addpath(genpath('utils')); addpath('models'); addpath('solvers'); addpath('%(tmp)s');
cfg.dt = %(dt)d; cfg.vmax = 20; cfg.cfl = %(cfl)g;
network = format_network('%(net)s', cfg);
fr = @friction_schifrinson; mc.tuning = 1.0; mc.reynolds = 1e5; mc.friction = @(D,k) fr(1e5,D,k); mc.gravity = 'static';
discrete = ode_end(network, mc);
sc.dt = %(dt)d; sc.maxiter_lin = 20; sc.maxiter_non = 2000; sc.maxerror = 1e-6;
sc.Tc = celsius2kelvin(-82.595); sc.pc = 45.988; sc.pn = 1.01325;
cf = @compressibility_aga88; sc.compressibility = @(p,T) cf(p,T,sc.pc,sc.Tc);
scenario = format_scenario('%(scen)s', network);
cpt = dlmread('%(cpt)s');                       %% compressor setpoints per step (the driver's addition)
st = steadystate(discrete, scenario, sc);
rtz = scenario.T0 * scenario.Rs * st.z0;
LAMBDA = 0.5; dt = %(dt)d;
[EL,EU,EP] = lu(discrete.E(rtz),'vector');
[AL,AU,AP] = lu(discrete.E(rtz) - (dt * LAMBDA) * discrete.A,'vector');
t = 0:dt:scenario.tH; K = numel(t);
y = repmat(st.ys,[1,K]); xk = discrete.x0;
for k = 2:K
    u = scenario.us + scenario.ut(t(k));
    fp = @(x) discrete.F * cpt(:,k) + discrete.B * u + discrete.f(st.xs,x,scenario.us,u,rtz);
    zk = discrete.E(rtz) * xk;
    z1 = AU \ (AL \ zk(AP));
    f1 = dt * fp(xk);
    a1 = dt * discrete.A * z1;
    z2 = zk + f1 + (1.0 - 2.0 * LAMBDA) * a1;
    z2 = AU \ (AL \ z2(AP));
    f2 = dt * fp(z1);
    a2 = dt * discrete.A * z2;
    zk = 0.5 * (f1 + f2 + a1 + a2);
    xk = xk + EU \ (EL \ zk(EP));
    y(:,k) = y(:,k) + discrete.C * xk;
end
z0 = st.z0; err = st.err;
save('-v7', '%(out)s', 't', 'y', 'z0', 'err');
'''

READER = r"col = textscan(fileread(network_path),'%c %f %f %f %f %f %f','HeaderLines',1,'Delimiter',',','EndOfLine','\n','CommentStyle','#');"
PARSER = r"""L = strsplit(fileread(network_path), "\n"); L = L(2:end); L = L(~cellfun(@isempty, strtrim(L)));
        n = numel(L); col = {repmat(' ',n,1), nan(n,1), nan(n,1), nan(n,1), nan(n,1), nan(n,1), nan(n,1)};
        for i = 1:n, f = strsplit(strtrim(L{i}), ','); col{1}(i) = f{1}(1); for j = 2:numel(f), col{j}(i) = str2double(f{j}); end, end"""


def trr154_demand(bcd):
    """The five exits' flows of the TRR154 file (kg/s, a row per minute), in morgen's demand order (exit01..exit05)."""
    r = ET.parse(gzip.open(bcd)).getroot()
    out = {}
    for node in r.iter():
        if node.tag.endswith('node') and node.get('id', '').startswith('exit'):
            out[node.get('id')] = [-float(v.get('value')) for v in node.iter() if v.tag.endswith('massflow')]
    return np.array([out[f'exit0{i}'] for i in range(1, 6)]).T


def boundaries(qd, case):
    """Per minute: exit flows (5), entry pressures (3), compressor setpoints (3); the case's changes as ramps."""
    steps_cp, steps_pin = CASES[case]
    K = HOURS*60+1
    pin = np.full((K, 3), G.PIN0)
    cp = np.tile(CP0, (K, 1))
    ramp = np.clip((np.arange(K)[:, None]-np.array([h*60 for h, *_ in steps_cp+steps_pin])[None])/RAMP, 0., 1.)
    for i, (h, j, d) in enumerate(steps_cp):
        cp[:, j] += d*ramp[:, i]
    for i, (h, d) in enumerate(steps_pin):
        pin += d*ramp[:, len(steps_cp)+i][:, None]
    return qd[:K], pin, cp


def record(morgen, octave, bcd, out=DATA, dt=30, cases=None):
    dt = int(dt)
    qd = trr154_demand(bcd)
    data = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for case in (cases.split(',') if cases else CASES):
            q, pin, cp = boundaries(qd, case)
            ut = '|'.join(str(60*k) for k in range(len(q)))
            up = '|'.join(';'.join(f'{v:.10g}' for v in row) for row in pin)
            uq = '|'.join(';'.join(f'{v:.10g}' for v in row) for row in q)
            scen = tmp/f'{case}.ini'
            scen.write_text(f'T0 = 20.0\nRs = 530.0\ntH = {HOURS*3600}.0\ncp = {";".join(str(v) for v in CP0)}\nup = {up}\nuq = {uq}\nut = {ut}\n', newline='\n')
            cpt = tmp/f'{case}_cp.txt'
            np.savetxt(cpt, np.repeat(cp, 60//dt, 0)[:-(60//dt-1)].T, delimiter=',')      # morgen's steps are dt s
            res = tmp/f'{case}.mat'
            net = Path(morgen)/'networks'/'GasLib24.net'
            fmt = (Path(morgen)/'utils'/'format'/'format_network.m').read_text()
            assert READER in fmt
            (tmp/'format_network.m').write_text(fmt.replace(READER, PARSER), newline='\n')   # Octave's textscan misreads the type column (morgen's FIXME)
            drv = tmp/f'{case}.m'
            drv.write_text(DRIVER % dict(dt=dt, cfl=dt*20/G.NOM_MORGEN, morgen=Path(morgen).as_posix(), tmp=tmp.as_posix(), net=net.as_posix(), scen=scen.as_posix(), cpt=cpt.as_posix(), out=res.as_posix()), newline='\n')
            subprocess.run([octave, '--no-gui', '--quiet', drv.as_posix()], check=True)
            from scipy.io import loadmat
            m = loadmat(res)
            data.update({f'{case}/y': m['y'].T[::60//dt], f'{case}/t': m['t'].ravel()[::60//dt], f'{case}/z0': float(m['z0'].ravel()[0]), f'{case}/err': float(m['err'].ravel()[0]),
                         f'{case}/qd': q, f'{case}/pin': pin, f'{case}/cp': cp})
            print(case, 'z0', float(m['z0'].ravel()[0]), 'steady error', float(m['err'].ravel()[0]), flush=True)
    np.savez_compressed(out, **data)
    print('wrote', out)


def load(path=DATA):
    z = np.load(path)
    out = {}
    for k in z.files:
        c, f = k.split('/')
        out.setdefault(c, {})[f] = z[k]
    return out


def compare(c, substeps=4, nom=G.NOM_MORGEN):
    """The engine's raw plant over the case from its steady state (morgen's RTz). Returns morgen's and the engine's
    outputs (entry flows, exit pressures), a row a minute."""
    z0 = float(c['z0'])
    P = G.gaslib_raw(dt=60., substeps=substeps, rtz=G.T0*G.RS*z0, nom=nom)
    M = Mech([P], n=1, dtype=F)
    names = [q['name'] for q in P.ws]
    x, w, p = P.nominal()

    def wk(k):
        v = w.clone()
        for j in range(3):
            v[0, names.index(f'cp{j+1}')] = c['cp'][k, j]
            v[0, names.index(f'pin{j+1}')] = c['pin'][k, j]
        for j in range(5):
            v[0, names.index(f'qd{j+1}')] = c['qd'][k, j]
        return v
    x, _, ok = M.steady(wk(0), p, start=(x, wk(0), p))
    assert bool(ok.all())
    Y = [M.y(x, wk(0), p)[0]]
    for k in range(1, len(c['t'])):
        x = M.advance(x, wk(k), p)                                # morgen takes the input at the end of the step (implicit)
        Y.append(M.y(x, wk(k), p)[0])
    eng = torch.stack(Y).numpy()
    eng = np.concatenate([eng[:, 5:8], eng[:, :5]], 1)            # morgen's order: entry flows, then exit pressures
    return np.asarray(c['y']), eng


def summary(data):
    rows = []
    for case, c in data.items():
        nat, eng = compare(c)
        _, coarse = compare(c, nom=G.NOM_LEN)
        base = np.abs(nat[0])
        scale = np.r_[np.full(3, nat[0, :3].sum()), base[3:]]      # entry flows against the total supply, pressures against their own
        dev, devc = np.abs(eng-nat).max(0)/scale, np.abs(coarse-nat).max(0)/scale
        move = np.abs(nat-nat[0]).max(0)/scale
        rows.append(dict(case=case, dev_q=float(dev[:3].max()), dev_p=float(dev[3:].max()), move_q=float(move[:3].max()), move_p=float(move[3:].max()),
                         dev_p12=float(devc[3:].max())))
    return rows


def table(rows):
    lines = ['GasLib-24, TRR154 exit flows, 24 h: engine against morgen (ode_end, 2.4 km pieces, IMEX2 at 30 s)',
             f'{"case":<10}{"exit pressures: move":>22}{"gap":>8}{"gap, 12 km":>12}{"entry flows: move":>20}{"gap":>9}   (% of the base pressure / of the total supply)']
    for r in rows:
        lines.append(f'{r["case"]:<10}{100*r["move_p"]:>21.2f}%{100*r["dev_p"]:>7.3f}%{100*r["dev_p12"]:>11.3f}%{100*r["move_q"]:>19.2f}%{100*r["dev_q"]:>8.2f}%')
    lines.append('entry flows under the ramps: morgen is not converged in its step (minute 724 of pin_ramps, entry 3: 171 / 158 / 144 kg/s at 30 / 15 / 5 s;')
    lines.append('the engine 128 at 15 s and at 2 s substeps); the exit pressures do not depend on it')
    return '\n'.join(lines)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'record':
        record(*sys.argv[2:])
    else:
        print(table(summary(load(sys.argv[1] if len(sys.argv) > 1 else DATA))))

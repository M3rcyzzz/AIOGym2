"""Plugin factories (aio.plugin), environment plugin CLI wiring (replay, famexam, faults), and boptest.Network."""
import json
import sys
import types

import numpy as np
import pytest
import torch

from aio import interface as IF, net as N, plugin as AP
from aio.sim import control as CT, core as C, famexam, faults, plugin as PG, replay
from aio.sim.validate import boptest as BP


def _save_ckpt(path, *, law=False, slow=0.):
    torch.manual_seed(0)
    net = N.AIO(law=law, slow=slow).eval()
    ck = dict(
        model=dict(actor=net.actor.state_dict(), pairs=net.pairs.state_dict()),
        config=dict(trainer='aio.train', unpaired=0.),
        iter=42,
        samples=12_345_678,
        hours=1.5,
        val=dict(x_pi=0.85, points=62.5, note='ok'),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, path)
    return net


# ---------------------------------------------------------------- aio.plugin factories

def test_plugin_policy_builds_policy_and_exact_source_meta(tmp_path):
    ckpt = tmp_path/'run-1'/'seed1'/'best.pt'
    _save_ckpt(ckpt)
    ctl, meta = PG.build('aio.plugin:policy', str(ckpt), device='cpu')
    assert isinstance(ctl, N.Policy)
    assert meta['name'] == 'aio'
    src = meta['source']
    assert src == dict(
        file='run-1/seed1/best.pt',
        trainer='aio.train',
        iter=42,
        samples=12_345_678,
        hours=1.5,
        x_pi=0.85,
        val=dict(x_pi=0.85, points=62.5),
    )
    prog = C.build(2, 5)
    res = CT.rollout(prog, ctl, steps=10)
    assert torch.isfinite(res['score']).all()
    with pytest.raises(ValueError):
        AP.policy(None)


def test_plugin_packet_loads_pt_and_released_dir(tmp_path):
    ckpt = tmp_path/'run-1'/'seed1'/'best.pt'
    net = _save_ckpt(ckpt)
    ctl, meta = PG.build('aio.plugin:packet', str(ckpt))
    assert isinstance(ctl, N.Controller) and meta is None
    pkt = np.zeros(IF.SIZE, dtype=np.float32)
    mv = ctl.step(pkt)
    assert mv.shape == (IF.NMV,) and np.isfinite(mv).all()

    # released folder layout (model.safetensors + config.json)
    pytest.importorskip('safetensors')
    from safetensors.torch import save_file
    rel = tmp_path/'released'
    rel.mkdir()
    tensors = {f'actor.{k}': v for k, v in net.actor.state_dict().items()}
    tensors.update({f'pairs.{k}': v for k, v in net.pairs.state_dict().items()})
    save_file(tensors, str(rel/'model.safetensors'))
    (rel/'config.json').write_text(json.dumps(dict(probe=0., unpaired=0.)), encoding='utf-8')
    ctl2, _ = PG.build('aio.plugin:packet', str(rel))
    assert isinstance(ctl2, N.Controller)
    assert np.allclose(ctl2.step(pkt), mv)
    with pytest.raises(ValueError):
        AP.packet('')


def test_plugin_ablated_builds_all_variants(tmp_path):
    ckpt = tmp_path/'run-law'/'seed1'/'last.pt'
    _save_ckpt(ckpt, law=True, slow=1000.)
    ctls, meta = PG.build('aio.plugin:ablated', str(ckpt), device='cpu')
    assert meta == dict(name='aio')
    assert set(ctls) == {'aio', 'aio_law', 'aio_nolaw', 'aio_nostats', 'aio_neither'}
    assert all(isinstance(v, N.Policy) for v in ctls.values())
    # nolaw & neither zero out law_in; nostats & neither zero out stat/stat2
    assert type(ctls['aio_nolaw'].net.pairs.law_in).__name__ == 'Zero'
    assert type(ctls['aio_nostats'].net.pairs.stat).__name__ == 'Zero'
    assert type(ctls['aio_nostats'].net.pairs.stat2).__name__ == 'Zero'
    assert type(ctls['aio_neither'].net.pairs.law_in).__name__ == 'Zero'
    assert type(ctls['aio_neither'].net.pairs.stat).__name__ == 'Zero'
    assert type(ctls['aio_law']).__name__ == 'LawOnly'
    prog = C.build(2, 7)
    for name, c in ctls.items():
        r = CT.rollout(prog, c, steps=6)
        assert torch.isfinite(r['score']).all(), name


# ---------------------------------------------------------------- replay / famexam / faults plugin CLI

@pytest.fixture
def fake_plugin_mod(monkeypatch):
    mod_name = '_test_fake_sim_plugin'
    mod = types.ModuleType(mod_name)

    def single(arg, device='cpu'):
        return CT.Hold(), dict(name='custom_meta', source=dict(file='fake/seed1/best.pt', trainer='aio.train', iter=1, samples=100, hours=0.1, x_pi=0.5, val={}))

    def bare(arg, device='cpu'):
        return CT.Hold()

    def multi(arg, device='cpu'):
        return dict(m1=CT.Hold(), m2=CT.PI()), dict(name='ignored_for_dict')

    class FakePacket:
        def __init__(self, gain=0.5):
            self.gain = float(gain)
            self.packets = []

        def step(self, packet):
            self.packets.append(np.asarray(packet, dtype=float).copy())
            return np.full(IF.NMV, self.gain, dtype=float)

    def pkt_factory(arg, device='cpu'):
        return FakePacket(float(arg) if arg is not None else 0.5)

    mod.single = single
    mod.bare = bare
    mod.multi = multi
    mod.FakePacket = FakePacket
    mod.pkt_factory = pkt_factory
    monkeypatch.setitem(sys.modules, mod_name, mod)
    return mod_name, mod


@pytest.mark.parametrize('mod,extra', [
    (replay, ['--out', 'x.json']),
    (famexam, ['--plants', 'qtank', '--out', 'x.json']),
    (faults, ['--out', 'x.json']),
])
def test_deprecated_ckpt_flag_errors_clearly(mod, extra, capsys):
    with pytest.raises(SystemExit) as exc:
        mod.main(extra + ['--ckpt', 'old.pt'])
    assert exc.value.code != 0
    err = capsys.readouterr().err
    assert '--ctl' in err and '--arg' in err


def test_famexam_deprecated_ablate_flag_errors_clearly(capsys):
    with pytest.raises(SystemExit) as exc:
        famexam.main(['--plants', 'qtank', '--out', 'x.json', '--ablate'])
    assert exc.value.code != 0
    assert '--ctl' in capsys.readouterr().err


def test_replay_famexam_faults_load_single_named_and_dict_plugins(fake_plugin_mod, tmp_path):
    mod_name, _ = fake_plugin_mod
    for mod, base in (
        (replay, ['--out', str(tmp_path/'r.json')]),
        (famexam, ['--plants', 'qtank', '--out', str(tmp_path/'f.json')]),
        (faults, ['--out', str(tmp_path/'fl.json')]),
    ):
        ap = mod._parser()
        # 1. single controller using meta['name']
        a = ap.parse_args(base + ['--ctl', f'{mod_name}:single', '--arg', 'foo'])
        got = mod._load_ctl(ap, a, 'cpu')
        ctls, src = got if mod is replay else (got, None)
        assert 'custom_meta' in ctls and isinstance(ctls['custom_meta'], CT.Hold)
        if mod is replay:
            assert src['file'] == 'fake/seed1/best.pt'
        # 2. single controller with --name override
        a = ap.parse_args(base + ['--ctl', f'{mod_name}:single', '--name', 'override'])
        got = mod._load_ctl(ap, a, 'cpu')
        ctls = got[0] if mod is replay else got
        assert 'override' in ctls and 'custom_meta' not in ctls
        # 3. bare controller (meta is None) defaults to 'aio'
        a = ap.parse_args(base + ['--ctl', f'{mod_name}:bare'])
        got = mod._load_ctl(ap, a, 'cpu')
        ctls = got[0] if mod is replay else got
        assert 'aio' in ctls and isinstance(ctls['aio'], CT.Hold)
        # 4. dict factory merges all returned controllers
        a = ap.parse_args(base + ['--ctl', f'{mod_name}:multi'])
        got = mod._load_ctl(ap, a, 'cpu')
        ctls = got[0] if mod is replay else got
        assert {'m1', 'm2'} <= set(ctls) and isinstance(ctls['m1'], CT.Hold) and isinstance(ctls['m2'], CT.PI)


def test_replay_cli_end_to_end_compatible_with_site_converter(tmp_path):
    from experiments.shared_rl_view import research_site_sim_replay as R
    ckpt = tmp_path/'sim-test'/'seed1'/'best.pt'
    _save_ckpt(ckpt)
    out = tmp_path/'raw.json'
    replay.main(['--out', str(out), '--n', '4', '--seed', '1234', '--index', '0',
                 '--ctl', 'aio.plugin:policy', '--arg', str(ckpt)])
    raw = json.loads(out.read_text(encoding='utf-8'))
    assert 'aio' in raw['controllers'] and raw['controllers']['aio']['name_zh'] == 'AIO'
    assert raw['source']['file'] == 'sim-test/seed1/best.pt' and raw['source']['trainer'] == 'aio.train'
    conv = R.convert(raw)
    assert conv['runs'][0]['controller'] == 'rl' and conv['runs'][0]['source']['run'] == 'sim-test'


# ---------------------------------------------------------------- boptest.Network & exams

def test_boptest_network_with_object_factory_and_plugin_spec(fake_plugin_mod):
    mod_name, mod = fake_plugin_mod
    loop = BP.LOOPS['twozone_apartment_hydronic']
    sign = np.array([1., 1.])
    lo = np.array([m['lo'] for m in loop['mv']])
    hi = np.array([m['hi'] for m in loop['mv']])
    y = np.array([293.15, 294.15])
    sp = np.array([295.15, 295.15])
    u = np.array([0.4, 0.5])

    # 1. instantiated .step(packet) controller object
    fake = mod.FakePacket(gain=0.5)
    net_obj = BP.Network(loop, fake, sign)
    u1 = net_obj(y, sp, u, lo, hi)
    u2 = net_obj(y + 0.1, sp, u1, lo, hi)
    assert len(fake.packets) == 2 and fake.packets[0].shape == (IF.SIZE,)
    assert np.all(np.isfinite(fake.packets[0])) and np.all(np.isfinite(fake.packets[1]))
    # slew = 0.2 * (1 - 0) = 0.2; gain 0.5 -> +0.1 per step
    assert np.allclose(u1, [0.5, 0.6]) and np.allclose(u2, [0.6, 0.7])

    # 2. MODULE:FACTORY plugin spec + arg
    net_spec = BP.Network(loop, f'{mod_name}:pkt_factory', sign, arg='1.0')
    u_hi = net_spec(y, sp, np.array([0.95, 0.95]), lo, hi)
    assert np.allclose(u_hi, hi)  # clipped at upper bound

    # 3. zero-arg callable factory
    net_fn = BP.Network(loop, lambda: mod.FakePacket(gain=-0.5), sign)
    assert np.allclose(net_fn(y, sp, u, lo, hi), [0.3, 0.4])

    # 4. invalid ctl type raises TypeError
    with pytest.raises(TypeError):
        BP.Network(loop, 12345, sign)


def test_boptest_exams_passes_ctl_and_arg_to_network(fake_plugin_mod, monkeypatch):
    mod_name, _ = fake_plugin_mod
    seen = []

    def fake_own(client, case, period, loop=None):
        return dict(case=case, period=period, tag='own')

    def fake_exam(client, case, period, make, loop=None):
        c = make(loop or BP.LOOPS[case])
        seen.append((case, type(c).__name__, getattr(c, 'ctl', None)))
        return dict(case=case, period=period, tag=type(c).__name__)

    monkeypatch.setattr(BP, 'own', fake_own)
    monkeypatch.setattr(BP, 'exam', fake_exam)
    monkeypatch.setattr(BP, 'spread', lambda urls, jobs, fn: [fn(None, j) for j in jobs])

    tests = dict(bestest_hydronic=dict(gains=[[1.]], pi=dict(kc=[1.], ti=[10.])))
    res = BP.exams(['http://unused'], f'{mod_name}:pkt_factory', tests, cases=['bestest_hydronic'], which=('peak_heat_day',), arg='0.25')
    assert [r['controller'] for r in res] == ['own', 'formula_pi', 'network', 'network_half_band']
    net_ctls = [ctl for _, cls, ctl in seen if cls == 'Network']
    assert len(net_ctls) == 2 and net_ctls[0] is not net_ctls[1] and all(c.gain == 0.25 for c in net_ctls)

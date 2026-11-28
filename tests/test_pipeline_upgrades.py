"""Comprehensive tests for aio.train pipeline upgrades:
- Cross-machine & --accum parity fixes (safe_torch_load + Path sanitization, --cosine-by updates/samples,
  GUARD_UNITS=3 batch invariance & gns clearing, post-accum clipping + deferred critic fit + --priv-critic fit_chunk,
  auto-restore resume config, cap_card allocator merge, perf glue: Watch/EMA/empty_cache/need_read/settle_rooms/prefetch_noise).
- Zero-CU async eval (--eval-every 0, atomic save & eval snapshots <275 MB BF16+FP32-1D reloadable via N.load,
  upgraded --score with --val-cache/--val-sizes/--val-raw/--meter).
- Curriculum & focal weighting (--focal-alpha, menu_kw cls_p/sep_p/integ_p, --hard-seeds/--hard-share).
- Elastic MuLoCo worker client (--diloco-hub binary HTTP /sync round-trip, blockwise INT8 g=128 pack/unpack,
  optimizer state preservation, local finiteness/min-steps guard, unreachable Hub fault tolerance).
"""
import http.server
import io
import json
import math
import os
import pathlib
import sys
import threading
import types
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from aio import interface as IF, net as N, seqnet as SQ, train as T
from aio.sim import core as C


def _seq_args(**kw):
    a = dict(
        seq='d32,l2,h4,m4,g1,q1,u1', seq_window=8, seq_ckpt=False, priv_critic=True,
        opt='muon', bias_lr_mult=4., muon_mom=.95, replay=1, no_stats=True,
        law=False, formula=False, solve=False, given=False, reader=False, slow=0., arx=False, dv=False, tied=0, knots=N.NLAB,
        probe=0., unpaired=.25, pre_cost=1., meter=False, heads=2, fixed=0, bc_iters=0, bc_floor=0.,
        window=4, window_min=2, guard='cut', cap=10., spike=10., relax=4, lean=False,
        gamma=.99, lam=.95, tau=.25, cost_w=.1, log_w=0., smooth=0., rev_smooth=0., floor=.1, fail_cost=10.,
        stops=True, hard=False, noise=0., known=.25, critic_known=.75, id_coef=0., id_sign=False,
        bc_coef=10., bc_own=False, drive_noise=.2, faults=0., wd=.05, lr=2e-4, beta2=.95,
        warm_steps=0, clip=1., accum=1, fit_chunk=0, critic_iters=2, after=0, ppo=False, value=1.,
        focal_alpha=0., rel_id_coef=0.,
    )
    a.update(kw)
    return SimpleNamespace(**a)


def test_safe_torch_load_and_path_sanitization(tmp_path):
    """safe_torch_load handles Python 3.13 pathlib._local and WindowsPath pickles; sanitize_paths removes all Path objects."""
    raw_meta = dict(
        out=tmp_path/'run',
        val_cache=pathlib.PurePosixPath('/content/aio/val-cache'),
        win=pathlib.PureWindowsPath('D:\\aiogym2-runs\\last.pt'),
        nested=[tmp_path/'a', (tmp_path/'b', dict(p=tmp_path/'c'))],
        num=42,
    )
    clean = T.sanitize_paths(raw_meta)
    assert isinstance(clean['out'], str) and isinstance(clean['val_cache'], str) and isinstance(clean['win'], str)
    assert isinstance(clean['nested'][0], str) and isinstance(clean['nested'][1][1]['p'], str)
    assert clean['num'] == 42

    # Simulate a Python 3.13 checkpoint pickled with a `pathlib._local` PosixPath reference
    fake_mod = types.ModuleType('pathlib._local')
    fake_mod.PosixPath = pathlib.PurePosixPath
    fake_mod.WindowsPath = pathlib.PureWindowsPath
    old_mod = sys.modules.get('pathlib._local')
    sys.modules['pathlib._local'] = fake_mod
    p_file = tmp_path/'py313_win.pt'
    try:
        # Pickle a PureWindowsPath directly (safe on POSIX) and tensor payload
        torch.save(dict(t=torch.tensor([1., 2.]), win_path=pathlib.PureWindowsPath('C:\\tmp\\x.pt'),
                        clean_cfg=clean), p_file)
    finally:
        if old_mod is None:
            sys.modules.pop('pathlib._local', None)
        else:
            sys.modules['pathlib._local'] = old_mod

    # Remove pathlib._local to simulate Python <=3.12 without it
    sys.modules.pop('pathlib._local', None)
    loaded = T.safe_torch_load(p_file, map_location='cpu')
    assert torch.equal(loaded['t'], torch.tensor([1., 2.]))
    assert 'x.pt' in str(loaded['win_path'])
    assert loaded['clean_cfg'] == clean


def test_cap_card_alloc_merge_and_maybe_empty_cache(monkeypatch):
    """cap_card preserves expandable_segments:True from PYTORCH_CUDA_ALLOC_CONF; maybe_empty_cache gates on watermark."""
    assert T.merge_alloc_conf('garbage_collection_threshold:0.8', 'expandable_segments:True') == (
        'expandable_segments:True,garbage_collection_threshold:0.8'
    )
    assert T.merge_alloc_conf('garbage_collection_threshold:0.8', 'expandable_segments:True,garbage_collection_threshold:0.5') == (
        'expandable_segments:True,garbage_collection_threshold:0.8'
    )
    assert T.merge_alloc_conf('garbage_collection_threshold:0.8', '') == 'garbage_collection_threshold:0.8'

    # Verify cap_card passes merged allocator string when CUDA is active
    seen_conf = []
    monkeypatch.setenv('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    monkeypatch.setattr(torch.cuda.memory, '_set_allocator_settings', lambda s: seen_conf.append(s))
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda i=0: (10*2**30, 24*2**30))
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda i=0: 2*2**30)
    monkeypatch.setattr(torch.cuda, 'set_per_process_memory_fraction', lambda frac, i=0: None)
    T.cap_card(torch.device('cuda:0'), 16., 'garbage_collection_threshold:0.8')
    assert seen_conf == ['expandable_segments:True,garbage_collection_threshold:0.8']

    # Verify maybe_empty_cache gating
    emptied = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: emptied.append(1))
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda i=0: int(0.50*24*2**30))
    assert T.maybe_empty_cache(torch.device('cuda:0'), 0.85, expandable=True) is False
    assert len(emptied) == 0
    assert T.maybe_empty_cache(torch.device('cuda:0'), 0.85, expandable=False) is True
    assert len(emptied) == 1
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda i=0: int(0.90*24*2**30))
    assert T.maybe_empty_cache(torch.device('cuda:0'), 0.85, expandable=True) is True
    assert len(emptied) == 2


def test_guard_units_3_batch_size_invariance():
    """GUARD_UNITS == 3 measures per-episode objective gradients independently of batch size n (n=4 vs n=8 identical)."""
    assert T.GUARD_UNITS == 3
    torch.manual_seed(0)
    net, critic = N.AIO(False), N.Critic(stats=False)
    for q in critic.parameters():
        q.requires_grad_(False)
    a = _seq_args(seq=None, priv_critic=False, window=4, noise=0., id_coef=0., unpaired=0., known=0., guard='cap', cap=10.)

    # Generate deterministic shared step noise for episodes 0..7 so episodes 0..3 see identical trajectories at n=4 and n=8
    prog8_ref = C.build(8, 77, 'cpu', 'cpu')
    C.reset(prog8_ref)
    noise8 = [prog8_ref.noise() for _ in range(a.window)]

    tops, capped = {}, {}
    for n in (4, 8):
        prog = C.build(n, 77, 'cpu', 'cpu')
        lane = T.Lane(prog, a, torch.Generator().manual_seed(11), teach=0.)
        # Ensure identical reset state on episodes 0..n-1 (since reset() draws 1 noise vector of shape (n, ...))
        if n == 4:
            st8 = C.reset(C.build(8, 77, 'cpu', 'cpu'))
            lane.st = {k: v[:4].clone() for k, v in st8.items()}
        prog.prefetch_noise = lambda k0, H, dev, _n=n: [(z[:_n], u[:_n]) for z, u in noise8[:H]]
        net.zero_grad(set_to_none=True)
        loss, _, gd, _ = T.window(net, critic, lane, a, dict(bc=0., sim=1., value=0.), torch.Generator().manual_seed(12), usual=None)
        loss.backward()
        tops[n] = gd.top.clone()

    # Episodes 0..3 are bit-identical across n=4 and n=8; Guard.top must match to machine precision!
    assert float(tops[4].max()) > 0
    assert torch.allclose(tops[4], tops[8][:4], rtol=1e-4, atol=1e-6)

    # With a shared `usual` threshold, capping triggers on the exact same episodes regardless of n!
    thresh = float(tops[4].median()) * 0.05
    for n in (4, 8):
        prog = C.build(n, 77, 'cpu', 'cpu')
        lane = T.Lane(prog, a, torch.Generator().manual_seed(11), teach=0.)
        if n == 4:
            st8 = C.reset(C.build(8, 77, 'cpu', 'cpu'))
            lane.st = {k: v[:4].clone() for k, v in st8.items()}
        prog.prefetch_noise = lambda k0, H, dev, _n=n: [(z[:_n], u[:_n]) for z, u in noise8[:H]]
        net.zero_grad(set_to_none=True)
        loss, _, gd, _ = T.window(net, critic, lane, a, dict(bc=0., sim=1., value=0.), torch.Generator().manual_seed(12), usual=thresh)
        loss.backward()
        capped[n] = gd.capped.clone()
    assert torch.equal(capped[4], capped[8][:4]) and bool(capped[4].any())


def test_post_accum_clipping_and_deferred_critic_fit_and_priv_fit_chunk():
    """Post-accum clipping averages unclipped grads before clipping once; deferred critic steps copt ONLY when actor steps;
    --priv-critic defaults chunk_of() to full batch unless explicitly overridden smaller."""
    # 1. Post-accum clipping vs pre-accum clipping mathematical proof
    net = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        net.weight.fill_(1.0)
    opt = torch.optim.SGD(net.parameters(), lr=1.0)
    probe = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        probe.weight.fill_(1.0)
    popt = torch.optim.SGD(probe.parameters(), lr=1.0)
    temp = {}
    # Window 1: grad = [10, 0] (norm 10 >> clip=1.0); Window 2: grad = [-9, 0] (norm 9 >> clip=1.0)
    # Post-accum mean = [0.5, 0] (norm 0.5 <= 1.0 -> unclipped step of 0.5!)
    # (Old pre-accum clip would have averaged [1, 0] and [-1, 0] -> 0.0!)
    net.weight.grad = torch.tensor([[10.0, 0.0]])
    probe.weight.grad = torch.tensor([[10.0, 0.0]])
    assert T.gathered(net, opt, temp, 2, True, rel_probe=probe, rel_popt=popt, clip=1.0) is False
    net.weight.grad = torch.tensor([[-9.0, 0.0]])
    probe.weight.grad = torch.tensor([[-9.0, 0.0]])
    assert T.gathered(net, opt, temp, 2, True, rel_probe=probe, rel_popt=popt, clip=1.0) is True
    assert temp['acc'] == dict(g=None, n=0, k=0)
    assert torch.allclose(net.weight.detach(), torch.tensor([[0.5, 1.0]]), atol=1e-6)
    assert torch.allclose(probe.weight.detach(), torch.tensor([[0.5, 1.0]]), atol=1e-6)

    # 2. Deferred fit_critic when accum == 3: critic weights stay frozen across windows 1 & 2, step ONCE at window 3
    torch.manual_seed(2)
    a = _seq_args(accum=3, critic_iters=2, fixed=0, heads=2, window=4, noise=0.05)
    net, critic = T.nets(a, torch.device('cpu'))
    opt = T.optimizer(net, a)
    copt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    prog = C.build(4, 55, 'cpu', 'cpu')
    gen = torch.Generator().manual_seed(3)
    lane = T.Lane(prog, a, gen, teach=0.)
    c_steps = []
    orig_cstep = copt.step
    copt.step = lambda *args, **kw: (c_steps.append(1), orig_cstep(*args, **kw))[1]
    c_before = [p.detach().clone() for p in critic.parameters()]
    temp = dict(usual=1.0, acc=dict(g=None, n=0, k=0))
    w = dict(bc=0., sim=1., value=1.)

    l1 = T.update(net, critic, opt, copt, lane, a, w, gen, temp)
    assert l1['updates'] == 0 and len(c_steps) == 0
    assert all(torch.equal(x, y) for x, y in zip(c_before, critic.parameters()))

    l2 = T.update(net, critic, opt, copt, lane, a, w, gen, temp)
    assert l2['updates'] == 0 and len(c_steps) == 0
    assert all(torch.equal(x, y) for x, y in zip(c_before, critic.parameters()))

    l3 = T.update(net, critic, opt, copt, lane, a, w, gen, temp)
    assert l3['updates'] == 1 and len(c_steps) == a.critic_iters
    assert temp['acc'] == dict(g=None, n=0, k=0)
    assert any(not torch.equal(x, y) for x, y in zip(c_before, critic.parameters()))
    assert 'critic' in l3 and math.isfinite(l3['critic'])

    # 3. chunk_of with --priv-critic defaults to full batch (0) on 48x20@18 instead of legacy 2
    lane_big = SimpleNamespace(size='48x20', prog=SimpleNamespace(n=18))
    assert T.chunk_of(SimpleNamespace(priv_critic=True, fit_chunk=32), lane_big) == 0
    assert T.chunk_of(SimpleNamespace(priv_critic=True, fit_chunk=256), lane_big) == 0
    assert T.chunk_of(SimpleNamespace(priv_critic=True, fit_chunk=4, _fit_chunk_explicit=True), lane_big) == 4
    assert T.chunk_of(SimpleNamespace(priv_critic=False, fit_chunk=32), lane_big) == 2


def test_restore_resume_config_and_cosine_by_updates():
    """restore_resume_config restores structural/optimizer/schedule flags unless explicitly overridden; scheduled_lr supports updates/samples/hub_lr."""
    net = SQ.SeqAIO('d32,l2,h4,m4,g1,q1,u1', w=16)
    ck = dict(
        model=dict(pairs=net.pairs.state_dict()),
        cosine_by='updates', total_updates=5000, total_samples=0,
        config=dict(
            seq='d32,l2,h4,m4,g1,q1,u1', seq_window=16, replay=1, seq_ckpt=True,
            ema=0.97, wd=0.05, muon_mom=0.95, beta2=0.95, rel_id_coef=0.1,
            unpaired=0.5, meter=True, stops=True, fixed=64, heads=4, priv_critic=True,
            opt='muon', bias_lr_mult=4.0, knots=N.NLAB, focal_alpha=0.5,
        ),
    )
    bare = SimpleNamespace(
        seq=None, seq_window=128, replay=0, seq_ckpt=False, ema=0., wd=0., muon_mom=0.9, beta2=0.99,
        rel_id_coef=0., unpaired=0., meter=False, stops=False, fixed=0, heads=2, priv_critic=False,
        opt='adamw', bias_lr_mult=1., knots=N.NLAB, cosine_by='hours', total_updates=0, total_samples=0, focal_alpha=0.,
    )
    T.restore_resume_config(bare, ck, explicit=set())
    assert bare.seq == 'd32,l2,h4,m4,g1,q1,u1' and bare.seq_window == 16 and bare.replay == 1 and bare.seq_ckpt is True
    assert bare.ema == 0.97 and bare.wd == 0.05 and bare.muon_mom == 0.95 and bare.beta2 == 0.95
    assert bare.rel_id_coef == 0.1 and bare.unpaired == 0.5 and bare.meter is True and bare.stops is True
    assert bare.fixed == 64 and bare.heads == 4 and bare.priv_critic is True and bare.opt == 'muon'
    assert bare.cosine_by == 'updates' and bare.total_updates == 5000 and bare.focal_alpha == 0.5

    # Explicit overrides win (e.g. disabling EMA on DiLoCo worker)
    over = SimpleNamespace(**vars(bare))
    over.ema = 0.0
    over.total_updates = 10000
    T.restore_resume_config(over, ck, explicit={'ema', 'total_updates'})
    assert over.ema == 0.0 and over.total_updates == 10000

    # scheduled_lr across updates / samples / hub_lr
    a_u = SimpleNamespace(lr=1e-4, warmup=1, hours=2., cosine_by='updates', total_updates=100, total_samples=0)
    assert T.scheduled_lr(a_u, dict(row=5, updates=0, samples=0), hours_now=99.) == pytest.approx(1e-4)
    assert T.scheduled_lr(a_u, dict(row=5, updates=50, samples=0), hours_now=99.) == pytest.approx(5.5e-5)
    assert T.scheduled_lr(a_u, dict(row=5, updates=100, samples=0), hours_now=0.) == pytest.approx(1e-5)
    assert T.scheduled_lr(a_u, dict(row=5, updates=50, samples=0), hours_now=0., hub_lr=7.5e-5) == pytest.approx(7.5e-5)
    a_s = SimpleNamespace(lr=1e-4, warmup=1, hours=2., cosine_by='samples', total_updates=0, total_samples=1000)
    assert T.scheduled_lr(a_s, dict(row=5, updates=0, samples=500), hours_now=0.) == pytest.approx(5.5e-5)


def test_eval_every_0_atomic_snapshots_resume_and_upgraded_score(tmp_path, monkeypatch):
    """End-to-end CLI test: --eval-every 0 (0 eval overhead), --eval-snap-every (BF16+FP32-1D <275 MB layout,
    N.load reloadable), decoupled --save-every, --cosine-by updates + bare --resume, and upgraded --score with --val-cache."""
    out_dir = tmp_path/'run'
    vcache = tmp_path/'vcache'

    # 1. Train 3 rows with --eval-every 0 --eval-snap-every 1 --keep-eval-snaps 2 --eval-snap-raw --save-every 2 --cosine-by updates
    argv1 = [
        'aio.train', '--out', str(out_dir), '--device', 'cpu', '--cpu-qos', 'off', '--stall', '0',
        '--seq', 'd32,l2,h4,m4,g1,q1,u1', '--seq-window', '8', '--replay', '1', '--no-stats',
        '--priv-critic', '--opt', 'muon', '--wd', '0.05', '--ema', '0.9', '--ema-device', 'cpu',
        '--bc-iters', '0', '--envs', '2', '--lanes', '1', '--window', '4', '--row', '2', '--rows', '3',
        '--eval-every', '0', '--eval-snap-every', '1', '--keep-eval-snaps', '2', '--eval-snap-raw',
        '--save-every', '2', '--keep-saves', '2', '--cosine-by', 'updates', '--total-updates', '10',
        '--hours', '1', '--val-cache', str(vcache), '--meter', '--stops', '--unpaired', '0.5',
    ]
    monkeypatch.setattr(sys, 'argv', argv1)
    T.main()

    run_meta = json.loads((out_dir/'run.json').read_text())
    final_meta = json.loads((out_dir/'final.json').read_text())
    rows = [json.loads(line) for line in (out_dir/'metrics.jsonl').read_text().splitlines() if line.strip()]
    assert len(rows) == 3
    assert run_meta['val_refs'] is None and run_meta['scale'] is None and run_meta['start'] is None
    assert final_meta['val'] is None and all('val' not in r for r in rows)
    # Decoupled --save-every saved row2.pt & final row3.pt despite --eval-every 0
    assert (out_dir/'last.pt').exists() and (out_dir/'row2.pt').exists() and (out_dir/'row3.pt').exists()
    # --eval-snap-every 1 with --keep-eval-snaps 2 pruned row1.pt and kept row2.pt, row3.pt (no .tmp files!)
    eval_files = sorted(p.name for p in (out_dir/'eval').iterdir())
    assert eval_files == ['row2.pt', 'row3.pt']

    # Verify last.pt has ZERO Path objects in config and persists cosine_by/total_updates
    last_ck = T.safe_torch_load(out_dir/'last.pt', map_location='cpu')
    assert last_ck['cosine_by'] == 'updates' and last_ck['total_updates'] == 10 and last_ck['guard_units'] == 3
    assert not any(isinstance(v, os.PathLike) for v in last_ck['config'].values())

    # Verify BF16 >=2D + FP32 <=1D snapshot layout & N.load reloadability
    snap_path = out_dir/'eval'/'row3.pt'
    snap = T.safe_torch_load(snap_path, map_location='cpu')
    assert snap['eval_snap'] is True and snap['iter'] == 3 and 'raw' in snap and 'opt' not in snap
    assert not any(isinstance(v, os.PathLike) for v in snap['config'].values())
    for k, v in snap['model']['actor'].items():
        if v.is_floating_point():
            assert v.dtype == (torch.bfloat16 if v.ndim >= 2 else torch.float32), f'{k}: {v.shape} {v.dtype}'
    reloaded, _ = N.load(snap_path, 'cpu')
    obs = torch.zeros(1, 2, IF.SIZE)
    prev = torch.zeros(1, 2, IF.NMV)
    with torch.no_grad():
        mean = reloaded.steps(obs, prev)[0]
    assert mean.shape == (1, 2, IF.NMV) and bool(torch.isfinite(mean).all())

    # Verify 131M-parameter pack_eval_state byte budget is strictly < 275 MB (262-268 MB)
    fake_131m = dict(w2d=torch.empty(130_800_000 // 1024, 1024, dtype=torch.float32),
                     b1d=torch.empty(200_000, dtype=torch.float32))
    packed_131m = T.pack_eval_state(fake_131m)
    total_bytes = sum(t.numel()*t.element_size() for t in packed_131m.values())
    assert total_bytes < 275 * 1024 * 1024, f'131M eval snap too big: {total_bytes/1e6:.1f} MB'

    # 2. Bare --resume (omitting --seq, --priv-critic, --opt, --wd, --ema, --cosine-by, --total-updates!) + envs change clears gns
    argv2 = [
        'aio.train', '--out', str(out_dir), '--resume', '--device', 'cpu', '--cpu-qos', 'off', '--stall', '0',
        '--envs', '3', '--lanes', '1', '--window', '4', '--row', '1', '--rows', '4',
        '--eval-every', '0', '--hours', '1', '--lr-jump',
    ]
    monkeypatch.setattr(sys, 'argv', argv2)
    T.main()
    rows2 = [json.loads(line) for line in (out_dir/'metrics.jsonl').read_text().splitlines() if line.strip()]
    assert len(rows2) == 4 and rows2[-1]['iter'] == 4
    last_ck2 = T.safe_torch_load(out_dir/'last.pt', map_location='cpu')
    assert len(last_ck2['gns']) == 1                              # cleared old gns on envs 2 -> 3 change, kept GUARD_UNITS=3 usual!
    assert last_ck2['usual'] is not None and last_ck2['cosine_by'] == 'updates' and last_ck2['total_updates'] == 10

    # 3. Upgraded --score on eval snapshot with --val-cache, --val-sizes, --val-raw, --meter
    ref_calls = []
    orig_refs = T.references
    monkeypatch.setattr(T, 'references', lambda *args, **kw: (ref_calls.append(1), orig_refs(*args, **kw))[1])
    score_argv = [
        'aio.train', '--score', str(snap_path), '--device', 'cpu', '--cpu-qos', 'off',
        '--val-n', '2', '--val-seed', '1234', '--val-cache', str(vcache),
        '--val-sizes', '24x10@2', '--val-raw', '--meter',
    ]
    monkeypatch.setattr(sys, 'argv', score_argv)
    res1 = T.main()
    assert len(ref_calls) == 2                                    # 12x5 + 24x10 cached on first call
    assert 'val' in res1 and 'raw' in res1 and 'val_sizes' in res1 and '24x10' in res1['val_sizes']
    assert all(k in res1['val'] for k in ('pts', 'con', 'move', 'flip', 'win', 'x_pi', 'pts_before', 'pts_after'))
    assert res1['iter'] == 3 and res1['updates'] == snap['updates'] and res1['samples'] == snap['samples']
    assert 'all' in res1 and 'score' in res1                      # legacy keys preserved

    # Second --score call must hit disk cache with ZERO references() recomputation!
    monkeypatch.setattr(sys, 'argv', score_argv)
    res2 = T.main()
    assert len(ref_calls) == 2
    assert res2['val']['pts'] == res1['val']['pts'] and res2['val_sizes']['24x10']['pts'] == res1['val_sizes']['24x10']['pts']


def test_focal_alpha_menu_kw_hard_seeds_and_perf_glue(tmp_path):
    """Test --focal-alpha weighting, menu_kw cls_p/sep_p/integ_p, HardSeeds exact anchor mix, Watch GPU stats,
    Average _foreach_lerp_ + CPU EMA, and Track A defensive hooks (need_read=False, lane_tag, settle_rooms, prefetch_noise)."""
    # 1. menu_kw & mix_of support cls_p, sep_p, integ_p
    kw = T.menu_kw(['cls_p=0.10,0.30,0.60', 'sep_p=0.45', 'integ_p=0.25', 'nd=1,2'])
    assert kw['shape'] == dict(cls_p=(0.1, 0.3, 0.6), sep_p=0.45, integ_p=0.25, nd=(1, 2))
    mx = T.mix_of(['0.35:cls_p=0.10,0.30,0.60+sep_p=0.45+integ_p=0.25'])
    assert len(mx) == 1 and mx[0][0] == pytest.approx(0.35)

    # 2. HardSeeds preserves exact 50 / 35 / 15 ratio and never touches plain lanes or non-12x5 sizes
    hpath = tmp_path/'hard.json'
    hpath.write_text(json.dumps(dict(seeds=[10_000_001, 10_000_002, 10_000_003], meta=dict(n=512))))
    hs = T.HardSeeds(hpath, share=0.15)
    assert hs.pick(1, plain=True) is None
    assert hs.pick(1, is_default_size=False) is None
    assert hs.pick(1, menu_active=True) is None
    c_uni = c_menu = c_hard = 0
    for sd in range(2000):
        _, on = T.mixed(None, mx, sd)
        if any(on):
            c_menu += 1
        elif hs.pick(sd, plain=False, is_default_size=True, menu_active=False, p_menu=0.35) is not None:
            c_hard += 1
        else:
            c_uni += 1
    assert 0.46 < c_uni/2000 < 0.54 and 0.31 < c_menu/2000 < 0.39 and 0.12 < c_hard/2000 < 0.18
    # drawn() works cleanly with plant_seed & event_seed (even standalone before Track A lands)
    p_hard = T.drawn(2, 999, 'cpu', 'cpu', plant_seed=10_000_001, event_seed=999)
    assert p_hard.n == 2

    # 3. --focal-alpha upweights higher-cost episodes vs uniform focal_alpha=0
    torch.manual_seed(4)
    net, critic = N.AIO(False), N.Critic(stats=False)
    for q in critic.parameters():
        q.requires_grad_(False)
    grads = {}
    for fa in (0.0, 0.7):
        a = _seq_args(seq=None, priv_critic=False, window=4, noise=0., id_coef=0., focal_alpha=fa)
        prog = C.build(6, 33, 'cpu', 'cpu')
        lane = T.Lane(prog, a, torch.Generator().manual_seed(5), teach=0.)
        net.zero_grad(set_to_none=True)
        seen = []
        orig_g = T.Guard.__call__
        T.Guard.__call__ = lambda self, g: (seen.append(g.detach().clone()), orig_g(self, g))[1]
        try:
            loss, rec, _, _ = T.window(net, critic, lane, a, dict(bc=0., sim=1., value=0.), torch.Generator().manual_seed(6))
            loss.backward()
        finally:
            T.Guard.__call__ = orig_g
        grads[fa] = (seen[-1].norm(dim=-1), rec['cost'].sum(1))
    g0, costs = grads[0.0]
    g1, _ = grads[0.7]
    ratio = g1 / g0.clamp_min(1e-12)
    hi_i, lo_i = int(costs.argmax()), int(costs.argmin())
    assert float(ratio[hi_i]) > float(ratio[lo_i])

    # 4. Track A defensive hooks (need_read=False, lane_tag, settle_rooms, prefetch_noise) + zero UserWarning!
    a = _seq_args(window=4, id_coef=0., noise=0.)
    net_s, crit_s = T.nets(a, torch.device('cpu'))
    opt_s, copt_s = T.optimizer(net_s, a), torch.optim.Adam(crit_s.parameters(), lr=1e-3)
    prog_s = C.build(2, 41, 'cpu', 'cpu')
    lane_s = T.Lane(prog_s, a, torch.Generator().manual_seed(7), teach=0.)
    hook_log = dict(need_read=[], lane_tag=[], settled=0, prefetched=0)
    orig_steps = net_s.steps

    def fast_steps(obs, prev, hidden=None, code=None, feats=None, need_read=True, lane_tag=None):
        hook_log['need_read'].append(need_read)
        hook_log['lane_tag'].append(lane_tag)
        mean, sec, mask, lab, h = orig_steps(obs, prev, hidden, code, feats)
        return mean, sec, mask, (None if not need_read else lab), h

    net_s.steps = fast_steps
    net_s.settle_rooms = lambda: hook_log.__setitem__('settled', hook_log['settled'] + 1)

    def prefetch_noise(k0, H, dev):
        hook_log['prefetched'] += 1
        return [prog_s.noise() for _ in range(H)]

    prog_s.prefetch_noise = prefetch_noise
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        log = T.update(net_s, crit_s, opt_s, copt_s, lane_s, a, dict(bc=0., sim=1., value=1.),
                       torch.Generator().manual_seed(8), dict(usual=1.0, acc=dict(g=None, n=0, k=0)))
    req_grad_warns = [w for w in caught if 'requires_grad=True' in str(w.message)]
    assert not req_grad_warns
    assert log['id'] == 0.0 and log['id_own'] == 0.0 and log['id_cross'] == 0.0
    assert hook_log['need_read'] == [False]*4
    assert len(set(hook_log['lane_tag'])) == 1 and hook_log['lane_tag'][0] is not None
    assert hook_log['settled'] >= 1 and hook_log['prefetched'] == 1

    # 5. Average.update() with _foreach_lerp_ and device='cpu' (must never use non_blocking=True on GPU->CPU!)
    import inspect
    assert 'non_blocking=True' not in inspect.getsource(T.Average.update)
    lin = torch.nn.Linear(4, 3)
    avg = T.Average(lin, 0.9, device='cpu')
    w0 = lin.weight.detach().clone()
    with torch.no_grad():
        lin.weight.add_(1.0)
    avg.update()
    assert avg.n == 1 and torch.allclose(avg.avg.weight, 0.1*w0 + 0.9*lin.weight.detach(), atol=1e-6)


def test_diloco_client_int8_roundtrip_mock_hub_and_fault_tolerance():
    """INT8 blockwise g=128 pack/unpack (<1e-2 error), live HTTP /sync round-trip preserving optimizer states,
    local min_steps & non-finite guards, and zero-crash resilience when Hub is unreachable."""
    # 1. Blockwise symmetric INT8 (g=128) pack/unpack across odd & multi-block shapes
    torch.manual_seed(0)
    for shape in ((5,), (17, 19), (128, 256), (3, 7, 29)):
        x = torch.randn(*shape) * 0.03
        packed = T.pack_delta_int8(x, g=128)
        assert set(packed) == {'q', 's', 'shape', 'numel', 'g'}
        assert packed['q'].dtype == torch.int8 and packed['s'].dtype == torch.float16 and packed['g'] == 128
        y = T.unpack_delta_int8(packed)
        assert y.shape == x.shape and y.dtype == torch.float32
        rel_err = float((x - y).norm() / x.norm().clamp_min(1e-8))
        assert rel_err < 1e-2, f'INT8 rel_err too high for {shape}: {rel_err}'

    # 2. Live mock HTTP Hub server verifying exact binary wire contract
    received = []
    a = _seq_args(seq='d32,l2,h4,m4,g1,q1,u1', rel_id_coef=0.1)
    net, critic = T.nets(a, torch.device('cpu'))
    opt = T.optimizer(net, a)
    copt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    rel_probe = torch.nn.Linear(4, 2)

    # Take one optimizer step so MuonAdamW and PrivCritic optimizer states are populated
    obs, prev = torch.randn(2, 2, IF.SIZE), torch.zeros(2, 2, IF.NMV)
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        net.actor.act[-1].weight.normal_(0, 0.1)
    net.steps(obs, prev)[0].square().mean().backward()
    opt.step()
    copt.zero_grad(set_to_none=True)
    prog = C.build(2, 15, 'cpu', 'cpu')
    st = C.reset(prog)
    critic.eval_state(prog, st, 0)[0].square().mean().backward()
    copt.step()

    client = T.DiLoCoClient('http://127.0.0.1:0', H=4, worker='test-worker', compress='int8', timeout=5.)
    client.snapshot_anchor(net)
    anchor_copy = {k: v.clone() for k, v in client.anchor.items()}

    # Simulate local inner updates moving actor weights away from anchor
    with torch.no_grad():
        for p in net.actor.parameters():
            p.add_(0.01)
    expected_delta = {k: anchor_copy[k] - v.detach().float().cpu() for k, v in net.actor.state_dict().items() if v.is_floating_point()}

    # Build target merged Hub weights (actor BF16 >=2D / FP32 <=1D, shifted critic & rel_probe)
    hub_actor = T.pack_eval_state({k: v + 0.05 for k, v in net.actor.state_dict().items()})
    hub_critic = {k: v.detach().clone() + 0.02 for k, v in critic.state_dict().items()}
    hub_probe = {k: v.detach().clone() + 0.03 for k, v in rel_probe.state_dict().items()}

    class MockHub(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == '/sync'
            length = int(self.headers['Content-Length'])
            req = T.safe_torch_load(io.BytesIO(self.rfile.read(length)), map_location='cpu')
            received.append(req)
            resp_dict = dict(round=req['round'] + 1, global_steps=32, inner_lr=7.5e-5, accepted=True,
                             actor=hub_actor, critic=hub_critic, rel_probe=hub_probe)
            buf = io.BytesIO()
            torch.save(resp_dict, buf)
            data = buf.getvalue()
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), MockHub)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        client.hub_url = f'http://127.0.0.1:{port}'
        # Local min_steps guard: 2 < int(0.7 * 4) = 2? Test 1 < 2
        client.record(1, 100)
        assert client.maybe_sync(net, critic, rel_probe) is None
        assert client.maybe_sync(net, critic, rel_probe, force=True)['skipped'] == 'min_steps'
        assert len(received) == 0

        # Reach H=4 updates -> triggers real HTTP /sync
        opt_state_before = {id(p): {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st_d.items()} for p, st_d in opt.state.items()}
        copt_state_before = {id(p): {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st_d.items()} for p, st_d in copt.state.items()}
        client.record(3, 300)
        res = client.maybe_sync(net, critic, rel_probe)
        assert res is not None and res['accepted'] is True and res['round'] == 1 and res['inner_lr'] == pytest.approx(7.5e-5)
        assert client.hub_lr == pytest.approx(7.5e-5) and client.updates_since_sync == 0 and client.env_steps_since_sync == 0
        assert len(received) == 1
        req = received[0]
        assert set(req) == {'worker', 'round', 'steps', 'env_steps', 'compress', 'actor_delta', 'critic', 'rel_probe', 'norm'}
        assert req['worker'] == 'test-worker' and req['steps'] == 4 and req['env_steps'] == 400 and req['compress'] == 'int8'
        decoded = T.unpack_actor_delta(req['actor_delta'])
        for k, exp in expected_delta.items():
            assert torch.allclose(decoded[k], exp, atol=1e-3, rtol=2e-2)
        # Verify Hub weights applied cleanly (BF16 upcast to FP32!)
        for k, v in net.actor.state_dict().items():
            assert v.dtype == torch.float32
            assert torch.allclose(v, hub_actor[k].float(), atol=1e-6)
        for k, v in critic.state_dict().items():
            assert torch.allclose(v, hub_critic[k], atol=1e-6)
        for k, v in rel_probe.state_dict().items():
            assert torch.allclose(v, hub_probe[k], atol=1e-6)
        # Verify local MuonAdamW and PrivCritic optimizer states are 100% preserved!
        for p, st_d in opt.state.items():
            for k, v in st_d.items():
                if torch.is_tensor(v):
                    assert torch.equal(v, opt_state_before[id(p)][k])
        for p, st_d in copt.state.items():
            for k, v in st_d.items():
                if torch.is_tensor(v):
                    assert torch.equal(v, copt_state_before[id(p)][k])

        # 3. Non-finite delta guard prevents poisoning Hub
        saved_w = net.actor.row.weight.detach().clone()
        with torch.no_grad():
            net.actor.row.weight[0, 0] = float('nan')
        client.record(4, 400)
        bad_res = client.maybe_sync(net, critic, rel_probe)
        assert bad_res['skipped'] == 'non_finite' and len(received) == 1
        with torch.no_grad():
            net.actor.row.weight.copy_(saved_w)
    finally:
        srv.shutdown()
        srv.server_close()

    # 4. Unreachable Hub degrades gracefully with ZERO crash!
    dead_client = T.DiLoCoClient('http://127.0.0.1:1', H=2, worker='offline-test', timeout=0.5)
    dead_client.snapshot_anchor(net)
    dead_client.record(2, 200)
    err_res = dead_client.maybe_sync(net, critic, rel_probe)
    assert err_res['accepted'] is False and 'error' in err_res


def test_quiet_sleep_seconds_enforces_guomao_quiet_hours():
    """Verify --quiet-hours parser handles 00:00-10:30 CST, boundary times, wrap-around windows, and disabled specs."""
    from datetime import datetime, timedelta, timezone
    cst = timezone(timedelta(hours=8))
    ts = lambda h, m, s=0: datetime(2026, 11, 29, h, m, s, tzinfo=cst).timestamp()

    assert T.quiet_sleep_seconds('', 'Asia/Shanghai', now=ts(1, 0)) == 0.0
    assert T.quiet_sleep_seconds('00:00-10:30', 'Asia/Shanghai', now=ts(0, 0, 0)) == pytest.approx(10.5 * 3600)
    assert T.quiet_sleep_seconds('00:00-10:30', 'Asia/Shanghai', now=ts(1, 0, 0)) == pytest.approx(9.5 * 3600)
    assert T.quiet_sleep_seconds('00:00-10:30', 'Asia/Shanghai', now=ts(10, 29, 30)) == pytest.approx(30.0)
    assert T.quiet_sleep_seconds('00:00-10:30', 'Asia/Shanghai', now=ts(10, 30, 0)) == 0.0
    assert T.quiet_sleep_seconds('00:00-10:30', 'Asia/Shanghai', now=ts(15, 0, 0)) == 0.0
    assert T.quiet_sleep_seconds('00:00-10:30', 'Asia/Shanghai', now=ts(23, 59, 0)) == 0.0
    # Overnight wrap-around window (23:30-10:30)
    assert T.quiet_sleep_seconds('23:30-10:30', 'Asia/Shanghai', now=ts(23, 45, 0)) == pytest.approx(10.75 * 3600)
    assert T.quiet_sleep_seconds('23:30-10:30', 'Asia/Shanghai', now=ts(2, 0, 0)) == pytest.approx(8.5 * 3600)
    assert T.quiet_sleep_seconds('23:30-10:30', 'Asia/Shanghai', now=ts(12, 0, 0)) == 0.0

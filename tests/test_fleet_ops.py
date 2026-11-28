"""Comprehensive unit and live HTTP integration tests for Track C hybrid fleet operations:
- monitoring.colab_job (parallel 4-thread pull, eval/row*.pt sync, --last-every, skip in/*.pt on resume, tmp pruning)
- monitoring.diloco_hub (live ThreadingHTTPServer 2-worker round-trip, INT8 pack/unpack <1e-2, MVUE weights,
  Outer Nesterov SGD math, PrivCritic & rel_probe FedAvg, Outer EMA, WSD LR, anomaly norm rejection,
  quorum=1 elastic detach/rejoin, atomic hub_last.pt + eval/outer_r*.pt snapshot export)
- monitoring.eval_worker (sidecar eval.jsonl, metrics.jsonl untouched, best_eval.json + best_eval.pt, famexam milestone)
- monitoring.local_job (thermal heartbeat >= 84 C, verified checkpoint handoff + audit line + resume flags, setup-wsl)
- monitoring.train_status (sidecar eval.jsonl merge, --workers polling + stale fallback, strict finite JSON)
- monitoring.mine_hard_seeds (VectorRng bit-exact parity with C.build(1, seed), causal regret ranking, JSON output)
"""
import io
import json
import math
import os
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from aio.sim import control as CT, core as C
from monitoring import colab_job as cj
from monitoring import diloco_hub as dh
from monitoring import eval_worker as ew
from monitoring import local_job as lj
from monitoring import mine_hard_seeds as mh
from monitoring import train_status as ts


# ---------------------------------------------------------------------------
# 1. monitoring.colab_job upgrades
# ---------------------------------------------------------------------------

def test_colab_job_parallel_pull_downloads_chunks_concurrently_and_reassembles(tmp_path, monkeypatch):
    monkeypatch.setattr(cj, 'ROOT', tmp_path)
    monkeypatch.setattr(cj, 'PART', 4)
    monkeypatch.setattr(cj, 'DOWNLOADS', 4)
    d = tmp_path / 'sq-pure-11'
    (d / 'run' / 'eval').mkdir(parents=True)
    job = {'name': 'sq-pure-11'}
    chunks = {'f.000': b'AAAA', 'f.001': b'BBBB', 'f.002': b'CCCC', 'f.003': b'DD'}

    monkeypatch.setattr(
        cj, 'rexec',
        lambda *a, **kw: {
            'files': {'run/eval/row10.pt': [14, 1700.0]},
            'size': 14,
            'parts': ['f.000', 'f.001', 'f.002', 'f.003'],
        },
    )

    barrier = threading.Barrier(4, timeout=5.0)
    active_lock = threading.Lock()
    active = 0
    max_active = 0
    f002_fail = {'done': False}

    def fake_colab(*args, **kw):
        nonlocal active, max_active
        assert args[0] == 'download'
        name = Path(args[3]).name
        dst = Path(args[4])
        if not (name == 'f.002' and f002_fail['done']):
            with active_lock:
                active += 1
                max_active = max(max_active, active)
            try:
                barrier.wait()
            finally:
                with active_lock:
                    active -= 1
        if name == 'f.002' and not f002_fail['done']:
            f002_fail['done'] = True
            raise RuntimeError('SSLError: UNEXPECTED_EOF_WHILE_READING')
        dst.write_bytes(chunks[name])

    monkeypatch.setattr(cj, 'colab', fake_colab)
    monkeypatch.setattr(cj.time, 'sleep', lambda _: None)

    got = cj.pull('sq-pure-11-1', job, ['run/eval/row10.pt'], tar=False)
    assert got == {'run/eval/row10.pt': [14, 1700.0]}
    assert (d / 'run' / 'eval' / 'row10.pt').read_bytes() == b'AAAABBBBCCCCDD'
    assert max_active == 4


def test_colab_job_wants_sync_eval_snaps_and_last_every_throttling(tmp_path, monkeypatch):
    # eval/row*.pt ALWAYS syncs even when sync_rows=False and sync_last=False
    assert cj.wants_sync('run/eval/row10.pt', sync_rows=False, sync_last=False)
    assert cj.wants_sync('run/sub-a/eval/row20.pt', sync_rows=False, sync_last=False)
    assert cj.wants_sync('run/eval/outer_r3.pt', sync_rows=False, sync_last=False)
    # Heavy bulk row*.pt outside eval/ stays skipped by default
    assert not cj.wants_sync('run/row10.pt', sync_rows=False, sync_last=True)
    assert not cj.wants_sync('run/sub-a/row20.pt', sync_rows=False, sync_last=True)
    # Atomic .tmp files skipped
    assert not cj.wants_sync('run/eval/row10.pt.tmp', sync_rows=True, sync_last=True)
    # last.pt throttled when sync_last=False, synced when sync_last=True
    assert not cj.wants_sync('run/last.pt', sync_rows=False, sync_last=False)
    assert cj.wants_sync('run/last.pt', sync_rows=False, sync_last=True)

    monkeypatch.setattr(cj, 'ROOT', tmp_path)
    d = tmp_path / 'sq-pure-11'
    (d / 'run').mkdir(parents=True)
    job = {'name': 'sq-pure-11', 'session': 'sq-pure-11-1'}
    vm_files = {
        'run/row10.pt': [2_000_000_000, 100.0],
        'run/eval/row10.pt': [268_000_000, 100.0],
        'run/last.pt': [1_608_000_000, 105.0],
        'run/metrics.jsonl': [5000, 105.0],
    }
    pulled = []
    monkeypatch.setattr(cj, 'rexec', lambda *a, **kw: {'files': vm_files, 'running': True, 'exit': None, 'gpu': ''})
    monkeypatch.setattr(cj, 'pull', lambda s, j, rels, tar: (pulled.append((list(rels), tar)) or {r: vm_files[r] for r in rels if r in vm_files}))

    # Intermediate cycle (sync_last=False): pulls eval/row10.pt + metrics.jsonl, skips last.pt and bulk row10.pt
    _, got1 = cj.sync(job, 'sq-pure-11-1', sync_last=False)
    assert 'run/eval/row10.pt' in got1
    assert 'run/metrics.jsonl' in got1
    assert 'run/last.pt' not in got1
    assert 'run/row10.pt' not in got1

    # Full cycle (sync_last=True, e.g. at --last-every or rollover/exit): pulls last.pt now
    _, got2 = cj.sync(job, 'sq-pure-11-1', sync_last=True)
    assert 'run/last.pt' in got2
    assert 'run/eval/row10.pt' not in got2  # already recorded in synced.json


def test_colab_job_launch_skips_in_pt_on_resume_and_prunes_tmp_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(cj, 'ROOT', tmp_path)
    d = tmp_path / 'sq-pure-11'
    (d / 'in').mkdir(parents=True)
    (d / 'run').mkdir(parents=True)
    (d / 'src.tgz').write_bytes(b'src')
    (d / 'in' / 'sq10_last.pt').write_bytes(b'init-1.6gb')
    (d / 'in' / 'hard_seeds.json').write_text('{"seeds":[1]}')
    (d / 'run' / 'last.pt').write_bytes(b'resumed-1.6gb')
    (d / 'run' / 'metrics.jsonl').write_text('{"iter":1}\n')

    # Orphaned tmp* dirs pruned cleanly without touching real job dirs/files
    (d / 'tmp_leaked_1').mkdir()
    (d / 'tmp_leaked_1' / 'part').write_bytes(b'junk')
    (d / 'tmp_leaked_2').mkdir()
    removed = cj.prune_orphaned_tmps(d)
    assert sorted(removed) == ['tmp_leaked_1', 'tmp_leaked_2']
    assert not (d / 'tmp_leaked_1').exists()
    assert (d / 'run' / 'last.pt').exists()

    # Rollover launch with --resume skips in/*.pt while keeping non-.pt in/* and run/last.pt
    job = {
        'name': 'sq-pure-11', 'ref': 'main', 'commit': 'abcdef123456',
        'gpu': 'G4', 'high_mem': False, 'pip': [],
        'cmd': ['python', '-m', 'aio.train', '--out', '{run}', '--init', '{in}/sq10_last.pt', '--resume'],
        'pack': None, 'threads': None, 'mps': False, 'every': 600, 'max_hours': 23.0,
        'starts': [], 'made': 0.0,
    }
    uploaded = []
    monkeypatch.setattr(cj, 'put', lambda s, j, src, rel: uploaded.append(rel))
    monkeypatch.setattr(cj, 'rexec', lambda s, j, code, **kw: {'files': {'run/last.pt': [13, 1.0], 'run/metrics.jsonl': [11, 1.0]}} if code is cj.STATUS else 0)

    cj.launch(job, vm=('sq-pure-11-2', 'G4'))
    assert 'in/sq10_last.pt' not in uploaded
    assert 'in/hard_seeds.json' in uploaded
    assert 'run/last.pt' in uploaded
    assert 'run/metrics.jsonl' in uploaded

    # Fresh start (no run/last.pt): MUST upload in/sq10_last.pt
    (d / 'run' / 'last.pt').unlink()
    uploaded.clear()
    cj.launch(job, vm=('sq-pure-11-1', 'G4'))
    assert 'in/sq10_last.pt' in uploaded
    assert 'in/hard_seeds.json' in uploaded


# ---------------------------------------------------------------------------
# 2. monitoring.diloco_hub live ThreadingHTTPServer + full MuLoCo math
# ---------------------------------------------------------------------------

def _post_sync(url, payload):
    buf = io.BytesIO()
    torch.save(payload, buf)
    req = urllib.request.Request(
        f'{url}/sync',
        data=buf.getvalue(),
        headers={'Content-Type': 'application/octet-stream'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        assert resp.status == 200
        return dh.safe_torch_load(io.BytesIO(resp.read()), map_location='cpu')


def _get_status(url):
    with urllib.request.urlopen(f'{url}/status', timeout=10) as resp:
        assert resp.status == 200
        return json.loads(resp.read().decode('utf-8'))


def test_diloco_int8_pack_unpack_accuracy():
    torch.manual_seed(42)
    for shape in ((16, 33), (257,), (8, 128), (3, 5, 7)):
        t = torch.randn(*shape, dtype=torch.float32) * 0.1
        packed = dh.pack_int8(t, g=128)
        assert packed['g'] == 128
        assert packed['shape'] == shape
        assert packed['numel'] == t.numel()
        assert packed['q'].dtype == torch.int8
        assert packed['s'].dtype == torch.float16
        restored = dh.unpack_int8(packed)
        assert restored.shape == t.shape
        assert restored.dtype == torch.float32
        rel_err = float((restored - t).norm() / t.norm().clamp_min(1e-8))
        assert rel_err < 1e-2


def test_diloco_wsd_schedule():
    assert dh.wsd_lr(0, 6e-5, warmup_steps=0, stable_steps=100, decay_steps=50) == pytest.approx(6e-5)
    assert dh.wsd_lr(0, 6e-5, warmup_steps=10, stable_steps=100, decay_steps=50) == pytest.approx(6e-6)
    assert dh.wsd_lr(9, 6e-5, warmup_steps=10, stable_steps=100, decay_steps=50) == pytest.approx(6e-5)
    assert dh.wsd_lr(110, 6e-5, warmup_steps=10, stable_steps=100, decay_steps=50) == pytest.approx(6e-5)
    # Halfway through cosine decay: 0.55 * base_lr
    assert dh.wsd_lr(135, 6e-5, warmup_steps=10, stable_steps=100, decay_steps=50, min_lr_frac=0.1) == pytest.approx(6e-5 * 0.55)
    # End of decay: 0.1 * base_lr
    assert dh.wsd_lr(160, 6e-5, warmup_steps=10, stable_steps=100, decay_steps=50, min_lr_frac=0.1) == pytest.approx(6e-6)


def test_diloco_hub_live_two_worker_roundtrip_math_anomaly_and_elastic_quorum(tmp_path, monkeypatch):
    monkeypatch.setenv('AIO_HUB_QUIET', '1')
    torch.manual_seed(7)

    actor0 = {
        'layer.weight': torch.randn(16, 20, dtype=torch.float32),
        'layer.bias': torch.randn(16, dtype=torch.float32),
    }
    critic0 = {'v.weight': torch.ones(4, 8, dtype=torch.float32)}
    probe0 = {'weight': torch.zeros(3, 6, dtype=torch.float32)}
    pairs0 = {'seq': torch.tensor([1, 2, 3], dtype=torch.long)}

    hub = dh.DiLoCoHub(
        actor0, critic_sd=critic0, rel_probe_sd=probe0, pairs=pairs0,
        config={'out': tmp_path / 'orig', 'seq': 'd192,l4,h6'},
        out_dir=tmp_path / 'hub_out',
        outer_lr=0.7, outer_mom=0.6, outer_ema_max=0.55,
        base_lr=6e-5, warmup_steps=0, stable_steps=1000, decay_steps=0,
        min_quorum=1, target_quorum=2, grace_s=0.25,
        norm_mult=3.0, norm_warmup=2,
    )
    srv = dh.DiLoCoServer(hub, host='127.0.0.1', port=0).start()
    try:
        st0 = _get_status(srv.url)
        assert st0['round'] == 0 and st0['global_steps'] == 0 and st0['inner_lr'] == pytest.approx(6e-5)

        # ---- Round 1: 2 concurrent workers with unequal env_steps (MVUE 0.75 vs 0.25) ----
        d1 = {'layer.weight': torch.full((16, 20), 0.20), 'layer.bias': torch.full((16,), 0.40)}
        d2 = {'layer.weight': torch.full((16, 20), 0.60), 'layer.bias': torch.full((16,), -0.40)}
        c1 = {'v.weight': torch.full((4, 8), 2.0)}
        c2 = {'v.weight': torch.full((4, 8), 6.0)}
        p1 = {'weight': torch.full((3, 6), 1.0)}
        p2 = {'weight': torch.full((3, 6), 5.0)}

        req1 = {
            'worker': 'guomao', 'round': 0, 'steps': 32, 'env_steps': 3000, 'compress': 'int8',
            'actor_delta': {k: dh.pack_int8(v) for k, v in d1.items()},
            'critic': c1, 'rel_probe': p1, 'norm': 1.0,
        }
        req2 = {
            'worker': 'baolong', 'round': 0, 'steps': 32, 'env_steps': 1000, 'compress': 'int8',
            'actor_delta': {k: dh.pack_int8(v) for k, v in d2.items()},
            'critic': c2, 'rel_probe': p2, 'norm': 1.0,
        }
        with ThreadPoolExecutor(2) as ex:
            f1 = ex.submit(_post_sync, srv.url, req1)
            f2 = ex.submit(_post_sync, srv.url, req2)
            r1, r2 = f1.result(), f2.result()

        assert r1['accepted'] is True and r2['accepted'] is True
        assert r1['round'] == 1 and r2['round'] == 1
        assert r1['global_steps'] == 64 and r2['global_steps'] == 64
        # Broadcast dtype contract: >=2D BF16, <=1D FP32
        assert r1['actor']['layer.weight'].dtype == torch.bfloat16
        assert r1['actor']['layer.bias'].dtype == torch.float32
        # Verify exact MVUE + Outer Nesterov math:
        # w1=0.75, w2=0.25 -> avg_w = 0.75*0.20 + 0.25*0.60 = 0.30; avg_b = 0.75*0.40 + 0.25*(-0.40) = 0.20
        # buf_1 = avg_d; step_1 = avg_d + 0.6*buf_1 = 1.6*avg_d; theta_1 = theta_0 - 0.7 * 1.6 * avg_d
        exp_w1 = actor0['layer.weight'] - 0.7 * 1.6 * 0.30
        exp_b1 = actor0['layer.bias'] - 0.7 * 1.6 * 0.20
        assert torch.allclose(hub.actor['layer.weight'], exp_w1, atol=2e-3)
        assert torch.allclose(hub.actor['layer.bias'], exp_b1, atol=2e-3)
        assert torch.allclose(r1['actor']['layer.weight'].float(), exp_w1, atol=2e-2)
        assert torch.allclose(r1['actor']['layer.bias'], exp_b1, atol=2e-3)
        # Plain sample-weighted FedAvg on PrivCritic & rel_probe (NO Nesterov!)
        # 0.75*2 + 0.25*6 = 3.0; 0.75*1 + 0.25*5 = 2.0
        assert torch.allclose(r1['critic']['v.weight'], torch.full((4, 8), 3.0), atol=1e-5)
        assert torch.allclose(r1['rel_probe']['weight'], torch.full((3, 6), 2.0), atol=1e-5)
        # Outer EMA check at t=1: d_out = min(0.55, (1+1)/(5+1)) = 2/6 = 1/3
        d_out_1 = 2.0 / 6.0
        exp_ema_w1 = d_out_1 * actor0['layer.weight'] + (1.0 - d_out_1) * hub.actor['layer.weight']
        assert torch.allclose(hub.outer_ema['layer.weight'], exp_ema_w1, atol=1e-5)
        # Snapshot export & hub_last.pt verified
        snap1 = tmp_path / 'hub_out' / 'eval' / 'outer_r1.pt'
        assert snap1.is_file() and (tmp_path / 'hub_out' / 'hub_last.pt').is_file()
        sck = dh.safe_torch_load(snap1)
        assert sck['round'] == 1 and sck['updates'] == 64
        assert sck['model']['actor']['layer.weight'].dtype == torch.bfloat16
        assert sck['model']['actor']['layer.bias'].dtype == torch.float32
        assert isinstance(sck['config']['out'], str)

        # ---- Round 2: 2-worker Nesterov momentum accumulation check ----
        req1_r2 = dict(req1, round=1, actor_delta={k: v.to(torch.bfloat16) for k, v in d1.items()}, compress='bf16')
        req2_r2 = dict(req2, round=1, actor_delta={k: v.to(torch.bfloat16) for k, v in d2.items()}, compress='bf16')
        with ThreadPoolExecutor(2) as ex:
            f1 = ex.submit(_post_sync, srv.url, req1_r2)
            f2 = ex.submit(_post_sync, srv.url, req2_r2)
            r1_2, r2_2 = f1.result(), f2.result()
        assert r1_2['accepted'] is True and r1_2['round'] == 2
        # buf_2 = 0.6*0.30 + 0.30 = 0.48; step_2 = 0.30 + 0.6*0.48 = 0.588; theta_2 = theta_1 - 0.7*0.588
        exp_w2 = exp_w1 - 0.7 * (0.30 + 0.6 * (0.6 * 0.30 + 0.30))
        assert torch.allclose(hub.actor['layer.weight'], exp_w2, atol=5e-3)

        # ---- Anomaly norm gate: norm_updates == 2 >= norm_warmup(2), ema_norm == 1.0 -> norm=5.0 REJECTED ----
        before_w = hub.actor['layer.weight'].clone()
        bad_req = dict(req2, round=2, norm=5.0)
        r_bad = _post_sync(srv.url, bad_req)
        assert r_bad['accepted'] is False
        assert r_bad['round'] == 2
        assert torch.allclose(hub.actor['layer.weight'], before_w)
        assert _get_status(srv.url)['rejected_count'] == 1

        # ---- Elastic quorum=1 detach: guomao runs alone, grace_s expires -> steps to round 3! ----
        t0 = time.monotonic()
        solo_req = dict(req1, round=2, norm=1.0)
        r_solo = _post_sync(srv.url, solo_req)
        dt = time.monotonic() - t0
        assert r_solo['accepted'] is True
        assert r_solo['round'] == 3
        assert dt >= 0.20  # waited grace_s=0.25 before stepping solo
        assert (tmp_path / 'hub_out' / 'eval' / 'outer_r3.pt').is_file()

        # ---- Elastic rejoin: baolong arrives stale (round=2 < hub.round=3) -> rejected with fresh round=3 weights ----
        stale_req = dict(req2, round=2, norm=1.0)
        r_stale = _post_sync(srv.url, stale_req)
        assert r_stale['accepted'] is False
        assert r_stale['round'] == 3

        # baolong updates to round 3 and rejoins guomao cleanly for round 3 -> 4!
        with ThreadPoolExecutor(2) as ex:
            f1 = ex.submit(_post_sync, srv.url, dict(req1, round=3, norm=1.0))
            f2 = ex.submit(_post_sync, srv.url, dict(req2, round=3, norm=1.0))
            r1_4, r2_4 = f1.result(), f2.result()
        assert r1_4['accepted'] is True and r2_4['accepted'] is True
        assert r1_4['round'] == 4 and r2_4['round'] == 4

        # Hub resume from hub_last.pt preserves exact state
        restored_hub = dh.DiLoCoHub.from_hub_state(tmp_path / 'hub_out' / 'hub_last.pt')
        assert restored_hub.round == 4
        assert restored_hub.global_steps == hub.global_steps
        assert torch.allclose(restored_hub.actor['layer.weight'], hub.actor['layer.weight'])
        assert torch.allclose(restored_hub.outer_ema['layer.weight'], hub.outer_ema['layer.weight'])
    finally:
        srv.shutdown()


# ---------------------------------------------------------------------------
# 3. monitoring.eval_worker sidecar eval.jsonl + best_eval.json + famexam
# ---------------------------------------------------------------------------

def _write_fake_pt(path, payload=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w') as zf:
        zf.writestr('data.json', json.dumps(payload or {'ok': True}))


def test_eval_worker_sidecar_jsonl_best_tracking_and_famexam(tmp_path):
    run_root = tmp_path / 'sq-pure-11' / 'run'
    (run_root / 'eval').mkdir(parents=True)
    metrics_path = run_root / 'metrics.jsonl'
    metrics_orig = '{"iter":10,"t_row":82.5,"hours":0.23}\n'
    metrics_path.write_text(metrics_orig, encoding='utf-8')
    metrics_mtime_before = metrics_path.stat().st_mtime_ns

    # Verify build_score_cmd passes all upgraded zero-CU eval flags
    cmd = ew.build_score_cmd(
        'python', run_root / 'eval' / 'row10.pt', val_cache='/tmp/vc',
        val_n=256, val_sizes=['24x10@64', '48x20@32'], val_raw=True, meter=True,
    )
    assert '--score' in cmd and '--val-cache' in cmd and '--val-sizes' in cmd
    assert '--val-raw' in cmd and '--meter' in cmd

    # Create 3 snapshots + 1 corrupt non-zip file + 1 .tmp file (both must be ignored)
    _write_fake_pt(run_root / 'eval' / 'row10.pt', {'row': 10})
    _write_fake_pt(run_root / 'eval' / 'row20.pt', {'row': 20})
    _write_fake_pt(run_root / 'eval' / 'row50.pt', {'row': 50})
    (run_root / 'eval' / 'row60.pt').write_bytes(b'not-a-zip')
    _write_fake_pt(run_root / 'eval' / 'row70.pt.tmp', {'row': 70})

    scores_by_name = {
        'row10.pt': {'iter': 10, 'updates': 120, 'samples': 10000, 'hours': 0.25, 'val': {'pts': 49.2}, 'val_sizes': {'24x10': {'pts': 60.1}}},
        'row20.pt': {'iter': 20, 'updates': 240, 'samples': 20000, 'hours': 0.50, 'val': {'pts': 48.7}, 'val_sizes': {'24x10': {'pts': 59.8}}},
        'row50.pt': {'iter': 50, 'updates': 600, 'samples': 50000, 'hours': 1.25, 'val': {'pts': 51.4}, 'val_sizes': {'24x10': {'pts': 62.0}, '48x20': {'pts': 66.5}}},
    }
    fam_calls = []

    def fake_score(snap):
        return scores_by_name[snap.name]

    def fake_famexam(snap, out_json):
        fam_calls.append(snap.name)
        return {'mean': 71.2, 'families': {'qtank': 75.0, 'ph': 67.4}}

    res = ew.process_once(
        run_root, eval_host='local', famexam_every=50, famexam_delta=0.5,
        score_fn=fake_score, famexam_fn=fake_famexam,
    )
    assert [r['snap'] for r in res] == ['row10.pt', 'row20.pt', 'row50.pt']
    # metrics.jsonl MUST NEVER BE TOUCHED
    assert metrics_path.read_text(encoding='utf-8') == metrics_orig
    assert metrics_path.stat().st_mtime_ns == metrics_mtime_before

    # Sidecar eval.jsonl has 3 lines with strict JSON
    eval_lines = [json.loads(x) for x in (run_root / 'eval.jsonl').read_text().splitlines()]
    assert len(eval_lines) == 3
    assert [x['pts'] for x in eval_lines] == [49.2, 48.7, 51.4]

    # best_eval.json and best_eval.pt track row50.pt (51.4 > 49.2 > 48.7)
    best_doc = json.loads((run_root / 'best_eval.json').read_text())
    assert best_doc['snap'] == 'row50.pt' and best_doc['pts'] == 51.4
    assert best_doc['famexam']['mean'] == 71.2
    assert (run_root / 'eval' / 'best_eval.pt').read_bytes() == (run_root / 'eval' / 'row50.pt').read_bytes()
    # famexam triggered on row10 (first best) and row50 (every 50 + +2.2 pts), NOT row20
    assert fam_calls == ['row10.pt', 'row50.pt']

    # Second pass processes zero duplicate snapshots
    res2 = ew.process_once(run_root, eval_host='local', score_fn=fake_score, famexam_fn=fake_famexam)
    assert res2 == []


# ---------------------------------------------------------------------------
# 4. monitoring.local_job handoff, thermal heartbeat, setup-wsl
# ---------------------------------------------------------------------------

def test_local_job_heartbeat_thermal_warning_and_setup_wsl_safety():
    hb_ok = lj.build_heartbeat('guomao', 'sq-g1', 1234, True, '76, 145.5, 175.0, 88, 14200, 24576, Not Active', now=1000.0)
    assert hb_ok['thermal_warning'] is False
    assert hb_ok['gpu']['temp_c'] == 76.0 and hb_ok['gpu']['util_pct'] == 88.0

    hb_hot = lj.build_heartbeat('guomao', 'sq-g1', 1234, True, '84, 165.0, 175.0, 95, 15000, 24576, Not Active', now=1001.0)
    assert hb_hot['thermal_warning'] is True
    assert '84.0' in hb_hot['warning_msg']

    hb_thr = lj.build_heartbeat('baolong', 'sq-b1', 2345, True, '81, 130.0, 150.0, 90, 11000, 16384, Active', now=1002.0)
    assert hb_thr['thermal_warning'] is True

    # Setup script is Linux-side only and never modifies Windows venv / D:\aiogym2-runs / port 8765
    script = lj.build_setup_wsl_script(lj.resolve_host('guomao'))
    assert 'systemd=true' in script
    assert '$HOME/.cache/aio/inductor' in script
    assert 'tailscale up --ssh --hostname=guomao-wsl' in script
    assert 'D:\\aiogym2-runs' not in script
    assert '8765' not in script


def test_local_job_handoff_verifies_transfers_audits_and_recommends_flags(tmp_path):
    colab_root = tmp_path / 'colab-runs'
    src_run = colab_root / 'sq-pure-10' / 'run'
    src_run.mkdir(parents=True)
    dst_run = tmp_path / 'local-stage' / 'sq-pure-10'

    # Write real torch checkpoint containing a Path object to verify safe inspection
    ck = {
        'iter': 295, 'updates': 3540, 'samples': 283_200, 'hours': 9.41, 'guard_units': 3,
        'config': {'lr': 6e-5, 'val_cache': Path('/content/aio/val-cache'), 'seq': 'd768,l12,h12,r2'},
        'model': {'actor': {'w': torch.zeros(2, 2)}},
    }
    torch.save(ck, src_run / 'last.pt')
    (src_run / 'run.json').write_text(json.dumps({'trainer': 'aio.train', 'seq': 'd768,l12,h12,r2'}))
    (src_run / 'metrics.jsonl').write_text('{"iter":295,"t_row":83.0,"hours":9.41}\n')
    (src_run / 'eval.jsonl').write_text('{"snap":"row290.pt","iter":290,"pts":48.9}\n')

    res = lj.handoff(
        'colab:sq-pure-10', f'path:{dst_run}',
        colab_root=colab_root, total_updates=8000,
    )
    assert set(res['transferred']) == {'last.pt', 'run.json', 'metrics.jsonl', 'eval.jsonl'}
    assert (dst_run / 'last.pt').stat().st_size == (src_run / 'last.pt').stat().st_size
    assert res['verified']['iter'] == 295 and res['verified']['updates'] == 3540
    assert '--resume' in res['resume_flags']
    assert '--cosine-by' in res['resume_flags'] and 'updates' in res['resume_flags']
    assert '8000' in res['resume_flags']

    # Verify audit line appended cleanly to metrics.jsonl without corrupting prior rows
    mlines = [json.loads(x) for x in (dst_run / 'metrics.jsonl').read_text().splitlines()]
    assert len(mlines) == 2
    assert mlines[0]['iter'] == 295
    assert mlines[1]['handoff']['from'] == 'colab:sq-pure-10'
    assert mlines[1]['handoff']['updates'] == 3540

    # Corrupt last.pt MUST be rejected before any transfer
    bad_src = tmp_path / 'bad-src'
    bad_src.mkdir()
    (bad_src / 'last.pt').write_bytes(b'corrupted-bytes')
    with pytest.raises(ValueError, match='not a valid PyTorch zip'):
        lj.handoff(f'path:{bad_src}', f'path:{tmp_path / "never"}')


# ---------------------------------------------------------------------------
# 5. monitoring.train_status eval.jsonl merge + --workers stale fallback
# ---------------------------------------------------------------------------

def test_train_status_merges_eval_jsonl_and_handles_worker_stale_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, 'ROOT', tmp_path)
    now = 1_700_000_000.0

    # Colab job with sidecar eval.jsonl (including a bad NaN line to verify strict finite filtering)
    jd = tmp_path / 'sq-pure-11'
    (jd / 'run').mkdir(parents=True)
    (jd / 'job.json').write_text(json.dumps({
        'name': 'sq-pure-11',
        'cmd': ['python', '-m', 'aio.train', '--out', '{run}', '--rows', '600', '--hours', '16'],
        'starts': [{'time': now - 3600}],
    }))
    mp = jd / 'run' / 'metrics.jsonl'
    mp.write_text(
        json.dumps({'iter': 19, 't_row': 80.0, 'hours': 0.5}) + '\n'
        + json.dumps({'iter': 20, 't_row': 82.0, 'hours': 0.55}) + '\n'
        + json.dumps({'handoff': {'from': 'x', 'to': 'y'}}) + '\n'
    )
    os.utime(mp, (now - 30, now - 30))
    (jd / 'run' / 'eval.jsonl').write_text(
        json.dumps({'snap': 'row10.pt', 'iter': 10, 'pts': 49.5, 'evaluated_at': now - 600}) + '\n'
        + '{"snap":"bad.pt","pts":NaN}\n'
        + json.dumps({
            'snap': 'row20.pt', 'iter': 20, 'pts': 50.8, 'evaluated_at': now - 60,
            'val_sizes': {'24x10': {'pts': 61.2}, '48x20': {'pts': 66.4}},
            'famexam': {'mean': 70.9},
        }) + '\n'
    )

    # First collect without workers: 100% backward compat (source='colab'), eval merged cleanly
    c0 = ts.collect(now=now)
    assert c0['source'] == 'colab'
    assert 'workers' not in c0
    assert len(c0['runs']) == 1
    r0 = c0['runs'][0]
    assert r0['rows'] == 20 and r0['state'] == 'running'
    assert r0['eval']['pts'] == 50.8 and r0['eval']['best_pts'] == 50.8
    assert r0['eval']['sizes'] == {'24x10': 61.2, '48x20': 66.4}
    assert r0['eval']['famexam_mean'] == 70.9
    json.dumps(c0, allow_nan=False)

    # Collect with two workers: both healthy on pass 1
    cache_file = tmp_path / 'worker-cache.json'
    worker_state = {'baolong_down': False}

    def fake_poll(host, keep_h=24., now=None, timeout=5.0):
        assert timeout == 5.0
        if host == 'baolong' and worker_state['baolong_down']:
            raise TimeoutError('SSH timed out after 5s')
        return [{
            'name': f'run-{host}', 'job': f'{host}:run-{host}', 'host': host,
            'state': 'running', 'rows': 42, 'total': 300, 'hours': 1.2,
            'started': round(now - 4000), 'eval': {'pts': 52.1, 'best_pts': 52.1},
        }]

    c1 = ts.collect(now=now, workers='guomao,baolong', worker_timeout=5.0, cache_path=cache_file, poll_fn=fake_poll)
    assert c1['source'] == 'fleet'
    assert len(c1['runs']) == 3
    assert c1['workers']['guomao']['ok'] is True and c1['workers']['baolong']['ok'] is True
    json.dumps(c1, allow_nan=False)

    # Pass 2: baolong drops off WAN -> graceful stale fallback (marked stale=True, state='quiet')
    worker_state['baolong_down'] = True
    c2 = ts.collect(now=now + 60, workers='guomao,baolong', worker_timeout=5.0, cache_path=cache_file, poll_fn=fake_poll)
    assert c2['workers']['guomao']['ok'] is True
    assert c2['workers']['baolong']['ok'] is False and c2['workers']['baolong']['stale'] is True
    b_run = next(r for r in c2['runs'] if r.get('host') == 'baolong')
    assert b_run['stale'] is True and b_run['state'] == 'quiet'
    json.dumps(c2, allow_nan=False)


# ---------------------------------------------------------------------------
# 6. monitoring.mine_hard_seeds VectorRng parity + regret mining
# ---------------------------------------------------------------------------

def test_mine_hard_seeds_vector_rng_parity_and_regret_ranking(tmp_path):
    seeds = [10_000_000, 10_000_007, 10_000_042]
    batched = mh.build_from_seeds(seeds, device='cpu', gen_device='cpu', dims='12x5', teacher=False)
    for idx, s in enumerate(seeds):
        single = C.build(1, s, device='cpu', gen_device='cpu', dims='12x5', teacher=False)
        assert torch.equal(batched.P[idx:idx + 1], single.P)
        assert torch.equal(batched.T[idx:idx + 1], single.T)
        assert torch.equal(batched.seg_k[idx:idx + 1], single.seg_k)

    # Verify exact causal regret formula: (model - teacher) / max(hold - teacher, 100.0)
    top_seeds, details, stats = mh.rank_regret(
        [100, 101, 102, 103],
        model_scores=[500.0, 400.0, 250.0, 300.0],
        teacher_scores=[100.0, 100.0, 100.0, 200.0],
        hold_scores=[500.0, 300.0, 150.0, 220.0],   # 102 & 103 hit min_denom=100 floor!
        top_frac=0.5,
        min_denom=100.0,
    )
    # Regrets: 100 -> 400/400=1.0, 101 -> 300/200=1.5, 102 -> 150/100=1.5, 103 -> 100/100=1.0
    assert top_seeds == [101, 102]
    assert details[0]['regret'] == 1.5 and stats['max_regret_top'] == 1.5

    # End-to-end fast mining rollout writing JSON (default batch mode + plant mode)
    out_batch = tmp_path / 'hard_batch.json'
    payload_b = mh.mine_hard_seeds(
        None, out_batch, n=8, batch=2, start_seed=10_000_000,
        top_frac=0.25, dims='12x5', device='cpu', steps=12, mode='batch', ctl=CT.PI(),
    )
    saved_b = json.loads(out_batch.read_text(encoding='utf-8'))
    assert saved_b == payload_b
    assert saved_b['meta']['mode'] == 'batch' and saved_b['meta']['n_batches'] == 4
    assert len(saved_b['seeds']) == 1 and saved_b['seeds'][0] in range(10_000_000, 10_000_004)

    out_plant = tmp_path / 'hard_plant.json'
    payload_p = mh.mine_hard_seeds(
        None, out_plant, n=4, batch=2, start_seed=10_000_000,
        top_frac=0.25, dims='12x5', device='cpu', steps=12, mode='plant', ctl=CT.PI(),
    )
    saved_p = json.loads(out_plant.read_text(encoding='utf-8'))
    assert saved_p == payload_p and saved_p['meta']['mode'] == 'plant'
    assert len(saved_p['seeds']) == 1 and saved_p['seeds'][0] in range(10_000_000, 10_000_004)


def test_guomao_quiet_hours_enforced_in_eval_worker_and_local_job(tmp_path):
    """guomao enforces 00:00-10:30 CST quiet hours in both eval_worker routing and local_job defaults."""
    from datetime import datetime, timedelta, timezone
    cst = timezone(timedelta(hours=8))
    night = datetime(2026, 11, 29, 1, 15, tzinfo=cst).timestamp()
    day = datetime(2026, 11, 29, 11, 0, tzinfo=cst).timestamp()

    assert ew.in_quiet_hours('00:00-10:30', 'Asia/Shanghai', now=night) is True
    assert ew.in_quiet_hours('00:00-10:30', 'Asia/Shanghai', now=day) is False
    # During quiet hours guomao routes to baolong (or custom fallback); baolong runs 24x7
    assert ew.pick_eval_host('guomao', now=night) == 'baolong'
    assert ew.pick_eval_host('guomao-wsl', fallback_host='local', now=night) == 'local'
    assert ew.pick_eval_host('guomao', now=day) == 'guomao'
    assert ew.pick_eval_host('baolong', now=night) == 'baolong'

    # evaluate_snapshot records the effective rerouted eval_host during quiet hours
    run_root = tmp_path / 'qtest' / 'run'
    _write_fake_pt(run_root / 'eval' / 'row10.pt', {'row': 10})
    entries = ew.process_once(
        run_root, eval_host='guomao', quiet_now=night, famexam_every=0, famexam_delta=0,
        score_fn=lambda p: {'iter': 10, 'val': {'pts': 50.0}},
    )
    assert len(entries) == 1 and entries[0]['eval_host'] == 'baolong'

    # local_job recommends & defaults --quiet-hours 00:00-10:30 for guomao, None for baolong
    ep_g = lj.parse_endpoint('local:guomao:sq-11')
    ep_b = lj.parse_endpoint('local:baolong:sq-11')
    flags_g = lj.recommend_resume_flags(ep_g, {'updates': 4000, 'lr': 6e-5})
    flags_b = lj.recommend_resume_flags(ep_b, {'updates': 4000, 'lr': 6e-5})
    assert '--quiet-hours' in flags_g and '00:00-10:30' in flags_g
    assert '--quiet-hours' not in flags_b
    assert lj.resolve_host('guomao')['default_quiet_hours'] == '00:00-10:30'
    assert lj.resolve_host('baolong')['default_quiet_hours'] is None
    # Windows-proxied WSL2 launch uses Win32_Process.Create via -EncodedCommand to avoid vmIdleTimeout shutdown
    import base64
    wcmd = lj.build_win32_wsl_launch_cmd(lj.resolve_host('baolong'), '/root/aio-runs/sq-11')
    assert wcmd[-1].startswith('powershell -NoProfile -EncodedCommand ')
    decoded = base64.b64decode(wcmd[-1].split()[-1]).decode('utf-16le')
    assert 'Win32_Process' in decoded and 'wsl.exe -d Ubuntu -u root -- /root/AIOGym2/.venv/bin/python /root/aio-runs/sq-11/runner.py /root/aio-runs/sq-11' in decoded

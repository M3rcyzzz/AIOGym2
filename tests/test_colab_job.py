"""Unit tests for monitoring.colab_job: row*.pt sync filtering, chunk SSL retry/backoff, and rollover upload."""
import json
from pathlib import Path
import pytest

from monitoring import colab_job as cj


def test_wants_sync_skips_historical_row_snapshots_by_default():
    assert cj.wants_sync('run/last.pt')
    assert cj.wants_sync('run/best.pt')
    assert cj.wants_sync('run/sub-a/last.pt')
    assert cj.wants_sync('run/sub-a/best.pt')
    assert cj.wants_sync('run/metrics.jsonl')
    assert cj.wants_sync('run/run.json')
    assert cj.wants_sync('run/alive')
    assert cj.wants_sync('run/stalls.txt')
    assert cj.wants_sync('run/check1.json')
    assert cj.wants_sync('run/check1.json.pt')

    assert not cj.wants_sync('run/row5.pt')
    assert not cj.wants_sync('run/row720.pt')
    assert not cj.wants_sync('run/sub-a/row100.pt')

    assert cj.wants_sync('run/row5.pt', sync_rows=True)
    assert cj.wants_sync('run/sub-a/row100.pt', sync_rows=True)


def test_sync_filters_row_checkpoints_and_persists_seen_incrementally(tmp_path, monkeypatch):
    monkeypatch.setattr(cj, 'ROOT', tmp_path)
    d = tmp_path / 'sq-pure-2'
    (d / 'run').mkdir(parents=True)
    job = {'name': 'sq-pure-2', 'session': 'sq-pure-2-1'}
    vm_files = {
        'run/row5.pt': [2140000000, 100.0],
        'run/row10.pt': [2140000000, 105.0],
        'run/best.pt': [2140000000, 105.0],
        'run/last.pt': [2140000000, 110.0],
        'run/metrics.jsonl': [12000, 110.0],
        'run/run.json': [9000, 100.0],
        'run/alive': [51, 110.0],
        'run/stalls.txt': [0, 100.0],
    }

    pulled = []

    def fake_rexec(s, j, code, timeout=180, **vals):
        assert code is cj.STATUS
        return {'files': vm_files, 'running': True, 'exit': None, 'gpu': '10,96,90'}

    def fake_pull(s, j, rels, tar):
        pulled.append((list(rels), tar))
        return {r: vm_files[r] for r in rels if r in vm_files} | ({'job.log': [100, 110.0]} if 'job.log' in rels else {})

    monkeypatch.setattr(cj, 'rexec', fake_rexec)
    monkeypatch.setattr(cj, 'pull', fake_pull)

    st, got = cj.sync(job, 'sq-pure-2-1')
    assert st['running'] is True
    assert pulled == [
        (['run/best.pt'], False),
        (['run/last.pt'], False),
        (['run/alive', 'run/metrics.jsonl', 'run/run.json', 'run/stalls.txt', 'job.log'], True),
    ]
    assert 'run/row5.pt' not in got and 'run/row10.pt' not in got
    assert 'run/last.pt' in got and 'run/best.pt' in got
    seen = json.loads((d / 'synced.json').read_text())
    assert set(seen) == {'run/best.pt', 'run/last.pt', 'run/alive', 'run/metrics.jsonl', 'run/run.json', 'run/stalls.txt'}

    # Opt-in --sync-rows pulls only the previously skipped rowN.pt snapshots
    pulled.clear()
    _, got_rows = cj.sync(job, 'sq-pure-2-1', sync_rows=True)
    assert pulled == [
        (['run/row10.pt'], False),
        (['run/row5.pt'], False),
        (['job.log'], True),
    ]
    assert 'run/row5.pt' in got_rows and 'run/row10.pt' in got_rows


def test_sync_incremental_seen_survives_second_pt_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(cj, 'ROOT', tmp_path)
    d = tmp_path / 'sq-pure-2'
    (d / 'run').mkdir(parents=True)
    job = {'name': 'sq-pure-2', 'session': 'sq-pure-2-1'}
    vm_files = {
        'run/best.pt': [100, 10.0],
        'run/last.pt': [100, 11.0],
        'run/metrics.jsonl': [50, 11.0],
    }

    monkeypatch.setattr(cj, 'rexec', lambda *a, **kw: {'files': vm_files, 'running': True, 'exit': None, 'gpu': ''})

    def fake_pull(s, j, rels, tar):
        if rels == ['run/last.pt']:
            raise RuntimeError('transient failure on last.pt')
        return {r: vm_files[r] for r in rels if r in vm_files}

    monkeypatch.setattr(cj, 'pull', fake_pull)
    with pytest.raises(RuntimeError, match='last.pt'):
        cj.sync(job, 'sq-pure-2-1')

    # best.pt was already persisted to synced.json so the retry won't re-download it
    seen = json.loads((d / 'synced.json').read_text())
    assert seen == {'run/best.pt': [100, 10.0]}


def test_download_part_retries_ssl_eof_and_truncation_with_backoff(tmp_path, monkeypatch):
    part = tmp_path / 'part'
    sleeps = []
    calls = []

    def fake_colab(*args, **kw):
        calls.append((args, kw))
        n = len(calls)
        if n == 1:
            part.write_bytes(b'partial')
            raise RuntimeError(
                "colab download -s sq-pure-1-2: [colab] Download failed: "
                "SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol'))"
            )
        if n == 2:
            # Silent truncation: exit 0 but short file
            part.write_bytes(b'short')
            return None
        part.write_bytes(b'0123456789')
        return None

    monkeypatch.setattr(cj, 'colab', fake_colab)
    cj.download_part('sq-pure-1-2', '/content/aio/sq-pure-1/.xfer/f.025', part,
                     expected_size=10, tries=5, backoff=5., sleep=sleeps.append)
    assert len(calls) == 3
    assert all(kw.get('tries') == 1 for _, kw in calls)
    assert sleeps == [5.0, 10.0]
    assert part.read_bytes() == b'0123456789'


def test_download_part_fails_fast_on_non_transient_error(tmp_path, monkeypatch):
    part = tmp_path / 'part'
    sleeps = []
    calls = []

    def fake_colab(*args, **kw):
        calls.append(args)
        raise RuntimeError('colab download -s s1: 404 Not Found')

    monkeypatch.setattr(cj, 'colab', fake_colab)
    with pytest.raises(RuntimeError, match='404 Not Found'):
        cj.download_part('s1', '/content/aio/x/.xfer/f.000', part, expected_size=10, tries=5, backoff=5., sleep=sleeps.append)
    assert len(calls) == 2
    assert sleeps == [5.0]
    assert not part.exists()


def test_pull_recovers_single_dropped_chunk_without_losing_prior_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(cj, 'ROOT', tmp_path)
    monkeypatch.setattr(cj, 'PART', 4)
    d = tmp_path / 'sq-pure-2'
    (d / 'run').mkdir(parents=True)
    job = {'name': 'sq-pure-2'}
    chunks = {'f.000': b'ABCD', 'f.001': b'EFGH', 'f.002': b'IJ'}
    f001_attempts = 0

    monkeypatch.setattr(
        cj, 'rexec',
        lambda *a, **kw: {'files': {'run/last.pt': [10, 1234.0]}, 'size': 10, 'parts': ['f.000', 'f.001', 'f.002']},
    )

    def fake_colab(*args, **kw):
        nonlocal f001_attempts
        assert args[0] == 'download'
        name = Path(args[3]).name
        dst = Path(args[4])
        if name == 'f.001' and f001_attempts == 0:
            f001_attempts += 1
            raise RuntimeError('SSLError: [SSL: UNEXPECTED_EOF_WHILE_READING]')
        dst.write_bytes(chunks[name])

    monkeypatch.setattr(cj, 'colab', fake_colab)
    monkeypatch.setattr(cj.time, 'sleep', lambda _: None)

    got = cj.pull('sq-pure-2-1', job, ['run/last.pt'], tar=False)
    assert got == {'run/last.pt': [10, 1234.0]}
    assert (d / 'run' / 'last.pt').read_bytes() == b'ABCDEFGHIJ'
    assert f001_attempts == 1


def test_launch_uploads_large_metrics_jsonl_and_last_pt_not_row_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr(cj, 'ROOT', tmp_path)
    monkeypatch.setattr(cj, 'SMALL', 100)
    d = tmp_path / 'sq-pure-2'
    (d / 'in').mkdir(parents=True)
    (d / 'run').mkdir(parents=True)
    (d / 'src.tgz').write_bytes(b'src')
    (d / 'in' / 'last.pt').write_bytes(b'init-ckpt')
    (d / 'run' / 'last.pt').write_bytes(b'resumed-ckpt')
    (d / 'run' / 'best.pt').write_bytes(b'best-ckpt')
    (d / 'run' / 'row5.pt').write_bytes(b'row5-ckpt')
    (d / 'run' / 'metrics.jsonl').write_bytes(b'x' * 500)   # > SMALL
    (d / 'run' / 'run.json').write_bytes(b'{}')
    job = {
        'name': 'sq-pure-2', 'ref': 'main', 'commit': 'abcdef123456',
        'gpu': 'G4', 'high_mem': False, 'pip': [], 'cmd': ['python', '-m', 'aio.train'],
        'pack': None, 'threads': None, 'mps': False, 'every': 600, 'max_hours': 23.0,
        'starts': [], 'made': 0.0,
    }
    uploaded = []
    monkeypatch.setattr(cj, 'put', lambda s, j, src, rel: uploaded.append(rel))
    monkeypatch.setattr(cj, 'rexec', lambda s, j, code, **kw: {'files': {'run/last.pt': [12, 1.0], 'run/metrics.jsonl': [500, 1.0], 'run/run.json': [2, 1.0]}} if code is cj.STATUS else 0)

    cj.launch(job, vm=('sq-pure-2-2', 'G4'))
    assert 'in/last.pt' in uploaded
    assert 'run/last.pt' in uploaded
    assert 'run/metrics.jsonl' in uploaded
    assert 'run/run.json' in uploaded
    assert 'run/row5.pt' not in uploaded
    assert 'run/best.pt' not in uploaded

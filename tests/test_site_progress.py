"""site_progress: the baolong reader and the progress.json it feeds (no ssh)."""
import json
import os
import subprocess
import sys
import time

import pytest

from experiments.shared_rl_view import site_progress as sp


@pytest.fixture(autouse=True)
def undated_notices(tmp_path_factory, monkeypatch):
    """The live notes' training notice carries dates (paused_since, expected_*) that go stale with the wall clock;
    the reader's tests must not fail because the live site is due an update (11-13), so they read a copy without them.
    The notice checks themselves are tested on notices each test writes."""
    n = json.loads(sp.NOTES.read_text(encoding='utf-8'))
    for v in n.values():
        if isinstance(v, dict) and 'state' in v:
            for k in ('paused_since', 'expected_start', 'expected_replay'):
                v.pop(k, None)
    p = tmp_path_factory.mktemp('notes') / 'site_progress_notes.json'
    p.write_text(json.dumps(n, ensure_ascii=False), encoding='utf-8')
    monkeypatch.setattr(sp, 'NOTES', p)


def fake_runs(root, now):
    for run, n, done in (('scenario-ppo-20261102', 5, False), ('scenario-ppo-20261102b', 90, False),
                         ('scenario-ppo-20261102c', 30, False)):
        for s in (1, 2, 3):
            d = root / run / f'seed{s}'
            d.mkdir(parents=True)
            (d / 'run.json').write_text(json.dumps(dict(seed=s, envs=2048, steps=72, out=str(d))))
            rows = []
            for it in range(1, n + 1):
                r = dict(it=it, samples=it * 147456, hours=it / 60, sps=2500., lr=5e-5, pl=.1, ep_return=-900 + it)
                if it % 25 == 0:
                    r.update(val_iqm=5000 / it + s, val_median=600., val_worst=900, val_fail=0)
                rows.append(json.dumps(r))
            (d / 'metrics.jsonl').write_text('\n'.join(rows) + '\n{"it": 9')  # last line half written
            if run.endswith('b'):
                os.utime(d / 'metrics.jsonl', (now - 7200, now - 7200))
    rep = {f'D:\\aiogym2-runs\\scenario-ppo-20261102b\\seed{s}': dict(
        checkpoint='best.pt', samples=7e6, iteration=50, scenarios={'x': {}}, nominal=dict(out1_s=130, out2_s=120),
        all=dict(n=58, median=230., iqm=225., worst=700, gate=.6,
                 vs_dmc=dict(median=-5., iqm=-12., iqm_ci95=[-30., 4.], win=33, tie=1, loss=24),
                 vs_oracle=dict(median=10., iqm=14., iqm_ci95=[2., 30.], win=20, tie=2, loss=36))) for s in (1, 2, 3)}
    (root / 'scenario-ppo-20261102b' / 'p0_report.json').write_text(json.dumps(rep))


def test_a_seed_in_a_long_validation_still_reads_as_running(tmp_path):
    """A fifteen-plant validation writes nothing for about 17 minutes; latest.pt counts as a sign of life."""
    now = time.time()
    for name, age, latest in (('seed1', 1200, None), ('seed2', 2400, 300), ('seed3', 2400, None)):
        d = tmp_path / 'apc-v3' / f'zoo-rescale-{name}'
        d.mkdir(parents=True)
        (d / 'metrics.jsonl').write_text(json.dumps(dict(it=2000, samples=8704000, hours=1.)) + '\n')
        os.utime(d / 'metrics.jsonl', (now - age, now - age))
        if latest:
            (d / 'latest.pt').write_bytes(b'')
            os.utime(d / 'latest.pt', (now - latest, now - latest))
    code = sp.REMOTE.replace(repr(sp.RUNS), repr(str(tmp_path)))
    remote = json.loads(subprocess.run([sys.executable, '-'], input=code, text=True, capture_output=True, check=True).stdout)
    seeds = remote['runs']['apc-v3-zoo-rescale']['seeds']     # quiet 20 min; quiet 40 min but latest.pt 5 min ago; quiet 40 min
    assert {k: sp.seed_summary(k, v, {}, now)['state'] for k, v in seeds.items()} == dict(seed1='running', seed2='running', seed3='stopped')


def notes_fixture(tmp_path, monkeypatch):
    """The live notes with run C current and run B previous, as these tests expect."""
    n = json.loads(sp.NOTES.read_text(encoding='utf-8'))
    n['as_of'] = '2026-11-02T20:28:00+08:00'
    n['runs'] = [dict(id='scenario-ppo-20261102c', role='current', label='Run C · x', label_zh='C 轮 · x', note='', note_zh=''),
                 dict(id='scenario-ppo-20261102b', role='previous', label='Run B · x', label_zh='B 轮 · x', note='', note_zh='')]
    f = tmp_path / 'notes.json'
    f.write_text(json.dumps(n), encoding='utf-8')
    monkeypatch.setattr(sp, 'NOTES', f)


def test_reader_and_build(tmp_path, monkeypatch):
    now = time.time()
    notes_fixture(tmp_path, monkeypatch)
    tmp_path = tmp_path / 'runs'
    fake_runs(tmp_path, now)
    code = sp.REMOTE.replace(repr(sp.RUNS), repr(str(tmp_path)))
    remote = json.loads(subprocess.run([sys.executable, '-'], input=code, text=True, capture_output=True, check=True).stdout)
    assert set(remote['runs']) == {'scenario-ppo-20261102', 'scenario-ppo-20261102b', 'scenario-ppo-20261102c'}
    assert 'scenarios' not in remote['runs']['scenario-ppo-20261102b']['p0']['seed1']
    out = sp.build(remote=remote)
    assert [r['id'] for r in out['runs']] == ['scenario-ppo-20261102c', 'scenario-ppo-20261102b']
    c, b = out['runs']
    assert [s['state'] for s in c['seeds']] == ['running'] * 3 and [s['state'] for s in b['seeds']] == ['stopped'] * 3
    s1 = b['seeds'][0]
    assert len(s1['val']) == 3 and s1['best']['val_iqm'] == min(v[1] for v in s1['val'])
    assert s1['p0']['vs_dmc']['iqm'] == -12. and s1['p0']['nominal'] == [130, 120] and c['seeds'][0]['p0'] is None
    assert out['p0_refs'] == json.loads(sp.NOTES.read_text(encoding='utf-8')).get('p0_refs')  # reference tables pass through (none now)

    # a run the notes mark as stopped is shown stopped even while its metrics are fresh
    monkeypatch.setattr(sp, 'NOTES', sp.NOTES)
    n = json.loads(sp.NOTES.read_text(encoding='utf-8')); n['runs'][0]['stopped'] = True
    sp.NOTES.write_text(json.dumps(n), encoding='utf-8')
    assert {s['state'] for s in sp.build(remote=remote)['runs'][0]['seeds']} == {'stopped'}
    n['runs'][0].pop('stopped'); sp.NOTES.write_text(json.dumps(n), encoding='utf-8')

    # run B's P0 report predates the notes, so nothing blocks publishing
    f = out['freshness']
    assert b['p0_time'] and not f['blocks']['training'].get('stale') and not [p for p in f['problems'] if p['level'] == 'error']

    # a P0 report on the current run that is newer than the notes blocks the build
    rc = remote['runs']['scenario-ppo-20261102c']
    rc['p0'] = {k: v for k, v in remote['runs']['scenario-ppo-20261102b']['p0'].items()}
    rc['p0_time'] = now + 10 * 365 * 86400
    f = sp.build(remote=remote)['freshness']
    assert f['blocks']['notes']['stale'] and [p for p in f['problems'] if p['level'] == 'error']

    # so does a run that started after the one the notes call current
    rc['p0'], rc['p0_time'] = {}, None
    remote['runs']['scenario-ppo-20261103d'] = dict(seeds={'seed1': dict(modified=None)}, p0={}, p0_time=None, started=rc['started'] + 60)
    errors = [p['text'] for p in sp.build(remote=remote)['freshness']['problems'] if p['level'] == 'error']
    assert len(errors) == 1 and 'scenario-ppo-20261103d' in errors[0]


def test_unreachable_keeps_previous(capsys):
    prev = dict(runs=[dict(id='x', seeds=[])], training_through='2026-11-02T16:00+08:00')
    out = sp.build(host='no-such-host.invalid', previous=prev)
    assert out['runs'] == prev['runs'] and out['training_through'] == prev['training_through']
    assert 'warning' in capsys.readouterr().err
    assert out['freshness']['blocks']['training']['stale']


def test_thin_keeps_last():
    pts = [[i, i] for i in range(1001)]
    t = sp.thin(pts, 100)
    assert len(t) <= 101 and t[-1] == pts[-1] and t[0] == pts[0]


def test_i18n_checks_find_missing_chinese(tmp_path):
    from experiments.shared_rl_view import site_i18n
    data = dict(runs=[dict(label='A', label_zh='甲', note='n'), dict(label='B', label_zh='乙', source=dict(note='x'))],
                freshness=dict(problems=[dict(text='t')]))
    assert site_i18n.missing_zh(data, 'x.json') == ['.runs[0].note']
    (tmp_path / 'a.tsx').write_text('tr("Ok", "好") + tr(`Run ${n}`, ``) + tr("x", "")')
    assert len(site_i18n.missing_tr(tmp_path)) == 2


def test_reader_takes_per_seed_reports_and_sac_runs(tmp_path):
    now = time.time()
    fake_runs(tmp_path, now)
    rep = {'D:\\aiogym2-runs\\scenario-ppo-20261102c\\seed2': dict(
        checkpoint='best_5h.pt', samples=7e7, iteration=500, scenarios={'x': {}}, nominal=dict(out1_s=100, out2_s=90),
        all=dict(n=58, median=360., iqm=358., worst=900, gate=.4,
                 vs_dmc=dict(median=70., iqm=77.7, iqm_ci95=[50., 100.], win=14, tie=1, loss=43),
                 vs_oracle=dict(median=90., iqm=100., iqm_ci95=[80., 120.], win=2, tie=0, loss=56)))}
    (tmp_path / 'scenario-ppo-20261102c' / 'seed2' / 'p0_report_5h.json').write_text(json.dumps(rep))
    sac = tmp_path / 'scenario-rsac-20261102' / 'seed1'
    sac.mkdir(parents=True)
    (sac / 'run.json').write_text(json.dumps(dict(seed=1, max_hours=10)))
    (sac / 'metrics.jsonl').write_text(json.dumps(dict(it=1, samples=1024, hours=.01, sps=900., ep_return=-2000.)) + '\n')
    old = sac.parent / 'seed1-fp32'  # a set-aside attempt, renamed
    old.mkdir()
    (old / 'run.json').write_text(json.dumps(dict(seed=1)))
    (old / 'metrics.jsonl').write_text(json.dumps(dict(it=1, samples=512, hours=.01, sps=1155., ep_return=-2500.)) + '\n')
    code = sp.REMOTE.replace(repr(sp.RUNS), repr(str(tmp_path)))
    remote = json.loads(subprocess.run([sys.executable, '-'], input=code, text=True, capture_output=True, check=True).stdout)
    c = remote['runs']['scenario-ppo-20261102c']
    assert set(c['p0']) == {'seed2'} and c['p0']['seed2']['report'] == 'p0_report_5h.json' and c['p0_time']
    assert 'scenario-rsac-20261102' in remote['runs'] and remote['runs']['scenario-rsac-20261102']['seeds']['seed1']['rows']
    assert set(remote['runs']['scenario-rsac-20261102']['seeds']) == {'seed1'}
    # metrics.jsonl only at validations: a newer progress line in train.log (UTF-16, as PowerShell writes it)
    # keeps the seed running and gives the latest count
    os.utime(sac / 'metrics.jsonl', (now - 1800, now - 1800))
    (sac / 'train.log').write_bytes('\ufeff'.encode('utf-16-le') + ('warmup\r\n' + json.dumps(dict(it=40, samples=3584000, hours=.98, sps=1008.)) + '\r\n{"it": 41, "samp').encode('utf-16-le'))
    remote = json.loads(subprocess.run([sys.executable, '-'], input=code, text=True, capture_output=True, check=True).stdout)
    s1 = remote['runs']['scenario-rsac-20261102']['seeds']['seed1']
    assert s1['progress'] == dict(it=40, samples=3584000, hours=.98, sps=1008.)
    row = sp.seed_summary('seed1', s1, {}, remote['now'])
    assert row['state'] == 'running' and row['samples'] == 3584000 and row['sps'] == 1008.


def test_ph_runs_are_read_but_kept_out_of_the_four_tank_blocks(tmp_path, monkeypatch):
    now = time.time()
    notes_fixture(tmp_path, monkeypatch)
    n = json.loads(sp.NOTES.read_text(encoding='utf-8'))
    n['runs'].append(dict(id='ph-rsac-20261103', role='current', label='pH · x', label_zh='pH · x', note='', note_zh='',
                          refs=[dict(id='dmc', label='DMC', label_zh='DMC', iqm=194)]))
    sp.NOTES.write_text(json.dumps(n), encoding='utf-8')
    root = tmp_path / 'runs'
    fake_runs(root, now)
    d = root / 'ph-rsac-20261103' / 'seed1'
    d.mkdir(parents=True)
    (d / 'run.json').write_text(json.dumps(dict(seed=1, envs=1024, max_hours=10)))
    (d / 'metrics.jsonl').write_text(json.dumps(dict(it=2000, samples=2048000, hours=.5, sps=1080., val_iqm=900., val_fail=0)) + '\n')
    (root / 'ph-rsac-20261103' / 'p0_report.json').write_text(json.dumps({'D:\\x\\seed1': dict(samples=1, all=dict(n=60, iqm=150.))}))
    old_report = time.mktime((2026, 11, 1, 12, 0, 0, 0, 0, -1))  # before the notes
    os.utime(root / 'ph-rsac-20261103' / 'p0_report.json', (old_report, old_report))
    code = sp.REMOTE.replace(repr(sp.RUNS), repr(str(root)))
    remote = json.loads(subprocess.run([sys.executable, '-'], input=code, text=True, capture_output=True, check=True).stdout)
    remote['runs']['ph-rsac-20261103']['started'] = now + 3600   # started after run C
    out = sp.build(remote=remote)
    ph = next(r for r in out['runs'] if r['id'] == 'ph-rsac-20261103')
    assert ph['seeds'][0]['state'] == 'running' and ph['seeds'][0]['best']['val_iqm'] == 900. and ph['refs'][0]['iqm'] == 194
    f = out['freshness']
    assert not [p for p in f['problems'] if p['level'] == 'error']
    # a pH P0 report newer than the notes blocks the build too
    remote['runs']['ph-rsac-20261103']['p0_time'] = now + 10 * 365 * 86400
    errors = [p['text'] for p in sp.build(remote=remote)['freshness']['problems'] if p['level'] == 'error']
    assert len(errors) == 1 and "ph-rsac-20261103's P0 report" in errors[0]
    remote['runs']['ph-rsac-20261103']['p0_time'] = None
    # a newer pH run than the current one is caught, and only for pH
    remote['runs']['ph-rsac-20261104'] = dict(seeds={'seed1': dict(modified=None)}, p0={}, p0_time=None, started=now + 7200)
    errors = [p['text'] for p in sp.build(remote=remote)['freshness']['problems'] if p['level'] == 'error']
    assert len(errors) == 1 and errors[0].startswith('ph-rsac-20261104 started after ph-rsac-20261103')


def test_pending_tep_notice_refuses_once_its_start_has_long_passed():
    out = dict(runs=[], notes_as_of='2026-11-03T13:25:00+08:00', training_through=None,
               tep_rl=dict(state='pending', expected_start='2026-11-03T18:05:00+08:00'))
    out['generated_at'] = '2026-11-03T20:00+08:00'
    assert not [p for p in sp.freshness(out)['problems'] if p['level'] == 'error']
    out['generated_at'] = '2026-11-03T21:30+08:00'
    errors = [p for p in sp.freshness(out)['problems'] if p['level'] == 'error']
    assert len(errors) == 1 and 'tep_rl' in errors[0]['text']
    out['tep_rl']['state'] = 'training'
    assert not [p for p in sp.freshness(out)['problems'] if p['level'] == 'error']
    # while training, the promise is the learner joining the TEP replay
    out['tep_rl']['expected_replay'] = '2026-11-03T18:00:00+08:00'
    errors = [p for p in sp.freshness(out)['problems'] if p['level'] == 'error']
    assert len(errors) == 1 and 'joins the replay' in errors[0]['text'] and 'TEP notice' in errors[0]['text']
    # paused with no resume time: fine for a while, refused once it stands unrevised too long
    out['tep_rl'] = dict(state='paused', paused_since='2026-11-03T22:17:00+08:00')
    out['generated_at'] = '2026-11-04T08:00+08:00'
    assert not [p for p in sp.freshness(out)['problems'] if p['level'] == 'error']
    out['generated_at'] = '2026-11-04T10:30+08:00'
    errors = [p for p in sp.freshness(out)['problems'] if p['level'] == 'error']
    assert len(errors) == 1 and 'paused' in errors[0]['text']


def test_mixed_runs_carry_tep_validation_and_pick_the_best_by_score(tmp_path, monkeypatch):
    now = time.time()
    notes_fixture(tmp_path, monkeypatch)
    n = json.loads(sp.NOTES.read_text(encoding='utf-8'))
    n['runs'].append(dict(id='mixed-rsac-20261103', role='current', task='mixed', label='Mixed · x', label_zh='混合 · x', note='', note_zh=''))
    sp.NOTES.write_text(json.dumps(n), encoding='utf-8')
    root = tmp_path / 'runs'
    fake_runs(root, now)
    d = root / 'mixed-rsac-20261103' / 'seed1'
    d.mkdir(parents=True)
    (root / 'mixed-rsac-20261103' / 'seed1-nolib').mkdir()   # a failed first start, not a seed
    (d / 'run.json').write_text(json.dumps(dict(seed=1, envs=1024, tep_envs=256, max_hours=10, reward_scale=[1., 20.39])))
    rows = [dict(it=2000, samples=1000, hours=.3, val_iqm=200., tep_val_iqm=9000., score=1341.6, ep_return=-5., ep_return_tep=-900.),
            dict(it=4000, samples=2000, hours=.6, val_iqm=240., tep_val_iqm=6000., score=1200., ep_return=-6., ep_return_tep=-700.)]
    (d / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    code = sp.REMOTE.replace(repr(sp.RUNS), repr(str(root)))
    remote = json.loads(subprocess.run([sys.executable, '-'], input=code, text=True, capture_output=True, check=True).stdout)
    assert list(remote['runs']['mixed-rsac-20261103']['seeds']) == ['seed1']
    remote['runs']['mixed-rsac-20261103']['started'] = now + 3600
    out = sp.build(remote=remote)
    s = next(r for r in out['runs'] if r['id'] == 'mixed-rsac-20261103')['seeds'][0]
    assert s['config']['reward_scale'] == [1., 20.39] and s['train_tep'] == [[1000, -900.], [2000, -700.]]
    assert s['best']['val_iqm'] == 200. and s['best_mixed'] == dict(samples=2000, tep_val_iqm=6000., score=1200., val_iqm=240.)
    f = out['freshness']
    assert not [p for p in f['problems'] if p['level'] == 'error']


def test_suite_notice_refuses_once_its_start_has_long_passed():
    out = dict(runs=[], notes_as_of='2026-11-04T05:00:00+08:00', training_through=None,
               suite_rl=dict(state='pending', expected_start='2026-11-04T08:32:00+08:00'))
    out['generated_at'] = '2026-11-04T11:00+08:00'
    assert not [p for p in sp.freshness(out)['problems'] if p['level'] == 'error']
    out['generated_at'] = '2026-11-04T11:40+08:00'
    errors = [p for p in sp.freshness(out)['problems'] if p['level'] == 'error']
    assert len(errors) == 1 and 'suite_rl' in errors[0]['text'] and 'suite_rl' in errors[0]['text_zh']


def test_zoo_notice_refuses_once_its_start_has_long_passed():
    out = dict(runs=[], notes_as_of='2026-11-06T17:00:00+08:00', training_through=None,
               zoo_rl=dict(state='pending', expected_start='2026-11-06T18:30:00+08:00'))
    out['generated_at'] = '2026-11-06T21:00+08:00'
    assert not [p for p in sp.freshness(out)['problems'] if p['level'] == 'error']
    out['generated_at'] = '2026-11-06T21:40+08:00'
    errors = [p for p in sp.freshness(out)['problems'] if p['level'] == 'error']
    assert len(errors) == 1 and 'zoo_rl' in errors[0]['text'] and 'zoo_rl' in errors[0]['text_zh']


def test_heldout_notice_refuses_once_the_learner_replay_is_long_overdue():
    out = dict(runs=[], notes_as_of='2026-11-07T17:00:00+08:00', training_through=None,
               heldout_rl=dict(state='heldout', expected_replay='2026-11-07T19:30:00+08:00'))
    out['generated_at'] = '2026-11-07T22:00+08:00'
    assert not [p for p in sp.freshness(out)['problems'] if p['level'] == 'error']
    out['generated_at'] = '2026-11-07T22:40+08:00'
    errors = [p for p in sp.freshness(out)['problems'] if p['level'] == 'error']
    assert len(errors) == 1 and 'heldout_rl' in errors[0]['text'] and '留出族' in errors[0]['text_zh']
    out['heldout_rl'] = dict(state='zeroshot')    # the learner is in the replay: nothing is due
    assert not [p for p in sp.freshness(out)['problems'] if p['level'] == 'error']


def test_literature_column_notice_refuses_once_overdue():
    out = dict(runs=[], notes_as_of='2026-11-08T19:00:00+08:00', training_through=None,
               lit_rl=dict(state='heldout', expected_replay='2026-11-08T23:30:00+08:00'))
    out['generated_at'] = '2026-11-09T02:00+08:00'
    assert not [p for p in sp.freshness(out)['problems'] if p['level'] == 'error']
    out['generated_at'] = '2026-11-09T02:40+08:00'
    errors = [p for p in sp.freshness(out)['problems'] if p['level'] == 'error']
    assert len(errors) == 1 and 'lit_rl' in errors[0]['text'] and '文献精馏塔' in errors[0]['text_zh']


def test_a_hyphenated_variant_is_its_own_run(tmp_path):
    root = tmp_path / 'runs'
    for name in ('seed1', 'zoo-seed1', 'zoo-tok-seed1', 'zoo-tok-seed1-old'):
        d = root / 'apc-v3' / name
        d.mkdir(parents=True)
        (d / 'run.json').write_text(json.dumps(dict(seed=1, plant='qtank')))
        (d / 'metrics.jsonl').write_text(json.dumps(dict(it=0, samples=0, score=9000.)) + '\n')
    code = sp.REMOTE.replace(repr(sp.RUNS), repr(str(root)))
    remote = json.loads(subprocess.run([sys.executable, '-'], input=code, text=True, capture_output=True, check=True).stdout)
    assert {k for k in remote['runs'] if k.startswith('apc-v3')} == {'apc-v3', 'apc-v3-zoo', 'apc-v3-zoo-tok'}
    assert list(remote['runs']['apc-v3-zoo-tok']['seeds']) == ['seed1']


FAMS = ['tank', 'qtank', 'shell', 'tep']


def test_suite_runs_carry_per_family_validation_and_pick_the_best_by_score(tmp_path, monkeypatch):
    now = time.time()
    notes_fixture(tmp_path, monkeypatch)
    n = json.loads(sp.NOTES.read_text(encoding='utf-8'))
    n['runs'].append(dict(id='apc-v3', role='current', task='suite', label='Suite · x', label_zh='五类 · x', note='', note_zh='',
                          families=FAMS, fam_refs={q: dict(refs=[]) for q in FAMS}))
    sp.NOTES.write_text(json.dumps(n), encoding='utf-8')
    root = tmp_path / 'runs'
    fake_runs(root, now)
    (root / 'apc-v3' / 'zeroshot').mkdir(parents=True)      # evaluation files, not a seed
    d = root / 'apc-v3' / 'seed1'
    d.mkdir()
    (d / 'run.json').write_text(json.dumps(dict(seed=1, envs=1024, tep_envs=256, max_hours=10, plant='qtank,shell', snapshot_hours=5,
                                                family_scale=dict(qtank=6000., shell=140000., tep=20.764), scale_anchor='tep', init='D:\\x\\best.pt')))
    rows = [dict(it=0, samples=0, hours=0., val_iqm=500., val_iqm_qtank=25000., val_median_qtank=24000., val_fail_qtank=0,
                 val_iqm_shell=290000., val_fail_shell=0, tep_val_iqm=9900., score=40000., ep_return_qtank=-9e5, ep_return_tep=-900.),
            dict(it=2000, samples=1000, hours=.5, val_iqm=400., val_iqm_qtank=900., val_fail_qtank=0, val_iqm_shell=60000., val_fail_shell=1,
                 tep_val_iqm=9500., score=12000., ep_return_qtank=-2e5, ep_return_shell=-4e6, ep_return_tep=-800.)]
    (d / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    code = sp.REMOTE.replace(repr(sp.RUNS), repr(str(root)))
    remote = json.loads(subprocess.run([sys.executable, '-'], input=code, text=True, capture_output=True, check=True).stdout)
    assert list(remote['runs']['apc-v3']['seeds']) == ['seed1']
    assert 'init' not in remote['runs']['apc-v3']['seeds']['seed1']['config']       # no training-machine paths
    out = sp.build(remote=remote)
    s = next(r for r in out['runs'] if r['id'] == 'apc-v3')['seeds'][0]
    assert s['config']['family_scale']['tep'] == 20.764 and s['config']['plant'] == 'qtank,shell'
    assert s['fam_val']['qtank'] == [[0, 25000., 24000., 0], [1000, 900., None, 0]] and s['fam_val']['tep'][1] == [1000, 9500., None, None]
    assert s['fam_train']['shell'] == [[1000, -4e6]] and s['score_val'] == [[0, 40000.], [1000, 12000.]]
    assert s['best_suite'] == dict(samples=1000, score=12000., iqm=dict(qtank=900., shell=60000., tep=9500.))
    assert 'tep_val' not in s and 'best_mixed' not in s
    assert sp.family('apc-v3') == 'suite' and sp.current(out['runs'], 'suite')['id'] == 'apc-v3'
    assert not [p for p in out['freshness']['problems'] if p['level'] == 'error']


def test_a_variant_in_a_run_folder_is_its_own_run_with_forgetting_guards(tmp_path, monkeypatch):
    now = time.time()
    notes_fixture(tmp_path, monkeypatch)
    n = json.loads(sp.NOTES.read_text(encoding='utf-8'))
    n['runs'].append(dict(id='apc-v3', role='current', task='suite', label='Suite · x', label_zh='五类 · x', note='', note_zh='',
                          families=FAMS, fam_refs={q: dict(refs=[]) for q in FAMS}))
    sp.NOTES.write_text(json.dumps(n), encoding='utf-8')
    root = tmp_path / 'runs'
    fake_runs(root, now)
    base = root / 'apc-v3' / 'seed1'
    base.mkdir(parents=True)
    (base / 'run.json').write_text(json.dumps(dict(seed=1, plant='qtank,shell')))
    (base / 'metrics.jsonl').write_text(json.dumps(dict(it=0, samples=0, val_iqm_qtank=900., score=9000.)) + '\n')
    (base.parent / 'p0_report.json').write_text(json.dumps({'seed1': dict(all=dict(n=4, iqm=1.))}))
    d = base.parent / 'anchor-seed1'
    d.mkdir()
    (d / 'run.json').write_text(json.dumps(dict(seed=1, plant='tank,qtank', tep_envs=256, tep_frac=1/3, anchor='D:\\aiogym2-runs\\m\\best.pt',
                                                anchor_families=['tank', 'qtank'], anchor_coef=2.5, guard_families='tank,tep', guard_frac=.1,
                                                anchor_of=dict(tank='D:\\aiogym2-runs\\mixed-rsac-20261103\\seed1\\best.pt', qtank='init'),
                                                guard_ref_iqm=dict(tank=730., tep=9957.))))
    # rsac marks the rows that wrote best.pt (feasibility first, PR #116)
    rows = [dict(it=0, samples=0, val_iqm_tank=7000., val_iqm_qtank=1500., tep_val_iqm=20000., score=5000., bc=.4,
                 guard=dict(tank=6000., tep=9000.), guard_ok=False, guard_violation=8.1, best_pt=True),
            dict(it=2000, samples=1000, val_iqm_tank=760., val_iqm_qtank=1400., tep_val_iqm=10200., score=4000., bc=.2,
                 guard=dict(tank=40., tep=300.), guard_ok=True, guard_violation=0., best_pt=True),
            dict(it=4000, samples=2000, val_iqm_tank=800., val_iqm_qtank=1300., tep_val_iqm=10100., score=3990., bc=.1,
                 guard=dict(tank=50., tep=200.), guard_ok=True, guard_violation=0.),    # not 2% better: rsac keeps best.pt
            dict(it=6000, samples=3000, val_iqm_tank=2000., val_iqm_qtank=900., tep_val_iqm=15000., score=3000., bc=.1,
                 guard=dict(tank=1200., tep=4000.), guard_ok=False, guard_violation=1.5)]  # better score, outside the guard
    (d / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    old = d.parent / 'anchor-seed1-old'                                     # a set-aside attempt is no seed
    old.mkdir()
    code = sp.REMOTE.replace(repr(sp.RUNS), repr(str(root)))
    remote = json.loads(subprocess.run([sys.executable, '-'], input=code, text=True, capture_output=True, check=True).stdout)
    assert list(remote['runs']['apc-v3']['seeds']) == ['seed1'] and set(remote['runs']['apc-v3']['p0']) == {'seed1'}
    v = remote['runs']['apc-v3-anchor']
    assert list(v['seeds']) == ['seed1'] and v['p0'] == {}                  # the run-level P0 report is the base run's
    cfg = v['seeds']['seed1']['config']
    assert 'anchor' not in cfg and 'D:' not in json.dumps(cfg)               # no training-machine paths
    assert cfg['guard_ref_iqm'] == dict(tank=730., tep=9957.) and cfg['anchor_families'] == ['tank', 'qtank'] and cfg['guard_families'] == 'tank,tep'
    assert cfg['anchor_of'] == dict(tank='mixed-rsac-20261103/seed1/best.pt', qtank='init')   # teachers, without the machine path
    # until the notes name the variant, a newer run in the same task refuses the build
    out = sp.build(remote=remote)
    assert any('apc-v3-anchor' in p['text'] for p in out['freshness']['problems'] if p['level'] == 'error')
    n['runs'].insert(len(n['runs']) - 1, dict(id='apc-v3-anchor', role='current', task='suite', init_run='apc-v3',
                                              label='Six plants · x', label_zh='六类 · x', note='', note_zh='',
                                              families=FAMS, fam_refs={q: dict(refs=[]) for q in FAMS}))
    sp.NOTES.write_text(json.dumps(n), encoding='utf-8')
    out = sp.build(remote=remote)
    assert sp.current(out['runs'], 'suite')['id'] == 'apc-v3-anchor'
    assert not [p for p in out['freshness']['problems'] if p['level'] == 'error']
    # a side run (a control experiment the notes name, not the current round) does not refuse the build
    n2 = json.loads(sp.NOTES.read_text(encoding='utf-8'))
    n2['runs'] = [r for r in n2['runs'] if r['id'] != 'apc-v3-anchor']
    n2['runs'].append(dict(id='apc-v3-anchor', role='side', task='suite', label='Control · x', label_zh='对照 · x', note='', note_zh=''))
    sp.NOTES.write_text(json.dumps(n2), encoding='utf-8')
    side = sp.build(remote=remote)
    assert sp.current(side['runs'], 'suite')['id'] == 'apc-v3'
    assert not [p for p in side['freshness']['problems'] if 'started after' in p['text']]
    sp.NOTES.write_text(json.dumps(n), encoding='utf-8')
    s = next(r for r in out['runs'] if r['id'] == 'apc-v3-anchor')['seeds'][0]
    assert s['fam_val']['tank'][1][:2] == [1000, 760.] and set(s['fam_val']) == {'tank', 'qtank', 'tep'}
    assert s['best_suite']['samples'] == 1000 and s['best_suite']['guard_ok'] is True   # the last row that wrote best.pt
    # no checkpoint within the guard yet: best.pt is the one closest to it
    seed = dict(remote['runs']['apc-v3-anchor']['seeds']['seed1'], rows=[dict(r) for r in rows[:1]])
    one = sp.seed_summary('seed1', seed, {}, remote['now'])['best_suite']
    assert one['samples'] == 0 and one['guard_ok'] is False and one['guard_violation'] == 8.1
    # rows from before PR #116 carry no best_pt: a score 2% below the best so far, within the guard
    seed = dict(seed, rows=[{k: v for k, v in r.items() if k != 'best_pt'} for r in rows])
    assert sp.seed_summary('seed1', seed, {}, remote['now'])['best_suite']['samples'] == 1000
    assert s['guard'][0] == [0, dict(tank=6000., tep=9000.), False] and s['guard'][3][2] is False


def test_live_check_retries_a_stalled_response():
    import subprocess
    from types import SimpleNamespace
    from experiments.shared_rl_view import publish_bilingual_site as pub
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(28, cmd)   # curl timeout, as on 11-03
        return SimpleNamespace(stdout=b'<meta name="aio-release" content="r1">')
    pub.live_check('r1', run=run, sleep=lambda s: None)
    assert len(calls) == 2 and calls[0][-1].endswith('index.html?release=r1')


def test_live_check_gives_up_after_its_tries():
    import pytest
    from types import SimpleNamespace
    from experiments.shared_rl_view import publish_bilingual_site as pub
    with pytest.raises(AssertionError):
        pub.live_check('r1', tries=2, run=lambda cmd, **kw: SimpleNamespace(stdout=b'<meta name="aio-release" content="r0">'), sleep=lambda s: None)


def test_a_suite_run_must_list_every_family_it_trains():
    seed = dict(seed='seed1', config=dict(plant='tank,qtank,proc', tep_envs=256), fam_val={'tank': [], 'qtank': [], 'proc': [], 'tep': []})
    run = dict(id='apc-v3-zoo', role='current', task='suite', seeds=[seed])
    out = dict(runs=[run], notes_as_of='2026-11-06T18:00:00+08:00', training_through=None, generated_at='2026-11-06T19:00+08:00')
    errors = [p for p in sp.freshness(out)['problems'] if p['level'] == 'error']
    assert len(errors) == 1 and 'proc, qtank, tank, tep' in errors[0]['text'] and errors[0]['text_zh']
    run.update(families=['tank', 'qtank', 'proc', 'tep'], fam_refs={q: dict(refs=[]) for q in ('tank', 'qtank', 'proc')})
    errors = [p for p in sp.freshness(out)['problems'] if p['level'] == 'error']
    assert len(errors) == 1 and 'tep' in errors[0]['text'] and 'proc' not in errors[0]['text']
    run['fam_refs']['tep'] = dict(refs=[])
    assert not [p for p in sp.freshness(out)['problems'] if p['level'] == 'error']
    run['role'] = 'side'; run.pop('families')
    assert not [p for p in sp.freshness(out)['problems'] if p['level'] == 'error']


def test_progress_log_adds_training_events_the_notes_lack():
    import datetime as dt
    from experiments.shared_rl_view import site_progress as S
    spec = dict(id='apc-v3-exp2', role='current', label='Eighteen plants · round 2, APC-scale plants · seed 1',
                label_zh='十八类装置 · 第二轮，APC 规模装置 · 种子 1')
    t0 = dt.datetime(2026, 11, 8, 23, 38, tzinfo=S.CST).timestamp()
    raw = {'seed1': dict(started=t0, modified=t0 + 12.18 * 3600, done=dict(hours=12.0, samples=100056832),
                         rows=[dict(it=2000, hours=1.0, score=6100.5, best_pt=True), dict(it=4000, hours=2.0, score=6300.),
                               dict(it=12000, hours=6.1, score=5562.9, best_pt=True)])}
    auto = S.auto_log(spec, raw)
    assert [e['date'] for e in auto] == ['2026-11-08 23:38', '2026-11-09 00:38', '2026-11-09 05:44', '2026-11-09 11:48']
    assert auto[1]['text_zh'] == '第二轮 it 2000：选模分数 6,101，第一个保留权重。'      # rounded half up, rule 10
    assert auto[2]['text_zh'] == '第二轮 it 12000：选模分数 5,563，比保留的 6,101 低 8.8%，换为保留权重。'
    # with forgetting guards, as round 2 ran: it 2000 outside them, it 10000 the first inside, it 12000 2% lower
    g = S.auto_log(spec, {'seed1': dict(raw['seed1'], done=None, rows=[
        dict(it=2000, hours=1.0, score=5421., best_pt=True, guard_ok=False), dict(it=10000, hours=5., score=5767., best_pt=True, guard_ok=True),
        dict(it=12000, hours=6., score=5562.9, best_pt=True, guard_ok=True)])})
    assert [e['text_zh'] for e in g[1:]] == ['第二轮 it 2000：选模分数 5,421，不满足防遗忘约束，暂作保留权重（还没有满足约束的点）。',
                                             '第二轮 it 10000：选模分数 5,767，第一个满足防遗忘约束的点，成为保留权重。',
                                             '第二轮 it 12000：选模分数 5,563，比保留的 5,767 低 3.5%，换为保留权重。']
    assert g[2]['text'] == 'Round 2, it 10000: selection score 5,767, the first within the forgetting guard, now the kept checkpoint.'
    assert auto[3]['text'] == 'Round 2 ran to the end: 12.0 h, 100.1 M samples.' and auto[3]['text_zh'] == '第二轮跑完：12.0 小时，10,006 万样本。'
    # each opens with good or bad news (11-09): a kept checkpoint that forgets is bad news, a run stopped by hand too
    assert [e['tone'] for e in auto] == ['good'] * 4 and [e['tone'] for e in g[1:]] == ['bad', 'good', 'good']
    stopped = S.auto_log(spec, {'seed1': dict(raw['seed1'], done=dict(hours=3.2, halted='STOP file'))})
    assert stopped[-1]['tone'] == 'bad' and stopped[-1]['text_zh'].startswith('第二轮提前停止')
    notes = [dict(date='2026-11-09 11:49', text='Round 2 ended.', text_zh='第二轮结束。'),
             dict(date='2026-11-08 23:38', text='Round 2 started.', text_zh='第二轮开训。')]
    log = S.merge_log(notes, auto)
    # the start and the end are in the notes already; the two new kept checkpoints are not
    assert [(e['date'], bool(e.get('auto'))) for e in log] == [('2026-11-09 11:49', False), ('2026-11-09 05:44', True),
                                                              ('2026-11-09 00:38', True), ('2026-11-08 23:38', False)]
    assert all('t' not in e for e in log)
    # several seeds name theirs; a label override wins
    two = S.auto_log(dict(spec, log_name='R2', log_name_zh='二'), dict(raw, seed2=dict(raw['seed1'], done=None, rows=[])))
    assert two[0]['text'] == 'R2, seed1 started.' and any(e['text_zh'] == '二（种子 2）开训。' for e in two)


def test_a_log_entry_without_good_or_bad_news_stops_the_build(tmp_path, monkeypatch):
    notes_fixture(tmp_path, monkeypatch)
    n = json.loads(sp.NOTES.read_text(encoding='utf-8'))
    n['timeline'].insert(0, dict(date='2026-11-09 22:00', text='Something happened.', text_zh='发生了一件事。'))
    sp.NOTES.write_text(json.dumps(n), encoding='utf-8')
    out = sp.build(remote=dict(runs={}))
    errors = [p['text'] for p in out['freshness']['problems'] if p['level'] == 'error']
    assert [e for e in errors if 'tone' in e] == [e for e in errors if '2026-11-09 22:00' in e] != []   # the notes' own entries all have one


def test_progress_log_keeps_the_last_round_after_the_next_starts():
    """11-09: round 2's night of new checkpoints left the log when round 3 became current."""
    import datetime as dt
    from experiments.shared_rl_view import site_progress as S
    now = dt.datetime(2026, 11, 9, 16, 20, tzinfo=S.CST).timestamp()
    e = lambda h, text: dict(t=now - h*3600, date='', text=text, text_zh=text, auto=True)
    runs = [dict(id='apc-v3-exp3', role='current', _log=[e(.3, 'r3 start')]),
            dict(id='apc-v3-exp2', role='current', _log=[e(16, 'r2 start'), e(11, 'r2 it 10000'), e(4.5, 'r2 end')]),
            dict(id='apc-v3-zoo', role='current', _log=[e(80, 'old round')]),           # older than AUTO_DAYS
            dict(id='mixed-rsac-20261103-seq', role='side', _log=[e(1, 'side run')])]   # side runs stay out
    assert [x['text'] for x in S.logged(runs, now)] == ['r3 start', 'r2 start', 'r2 it 10000', 'r2 end']


def test_non_finite_metrics_are_left_out_so_browsers_can_parse_the_page_data(tmp_path, monkeypatch):
    # 11-09: round 3 logged ep_return_net = -inf; json wrote -Infinity and the live page could not parse progress.json
    now = time.time()
    notes_fixture(tmp_path, monkeypatch)
    n = json.loads(sp.NOTES.read_text(encoding='utf-8'))
    n['runs'].append(dict(id='apc-v3', role='current', task='suite', label='Suite · x', label_zh='五类 · x', note='', note_zh='',
                          families=FAMS, fam_refs={q: dict(refs=[]) for q in FAMS}))
    sp.NOTES.write_text(json.dumps(n), encoding='utf-8')
    root = tmp_path / 'runs'
    fake_runs(root, now)
    d = root / 'apc-v3' / 'seed1'
    d.mkdir(parents=True)
    (d / 'run.json').write_text(json.dumps(dict(seed=1, envs=1024, max_hours=10, plant='qtank,shell')))
    rows = [dict(it=0, samples=0, hours=0., val_iqm_qtank=900., val_iqm_shell=6e4, score=7000., ep_return_qtank=-2e5, ep_return_shell=-4e6,
                 guard=dict(qtank=1., shell=float('nan')), guard_ok=True),
            dict(it=250, samples=500, hours=.1, ep_return_qtank=-1e5, ep_return_shell=float('-inf')),
            dict(it=500, samples=1000, hours=.2, ep_return_qtank=-1e5, ep_return_shell=-3e6)]
    (d / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    code = sp.REMOTE.replace(repr(sp.RUNS), repr(str(root)))
    remote = json.loads(subprocess.run([sys.executable, '-'], input=code, text=True, capture_output=True, check=True).stdout)
    out = sp.build(remote=remote)
    s = next(r for r in out['runs'] if r['id'] == 'apc-v3')['seeds'][0]
    assert s['fam_train']['shell'] == [[0, -4e6], [1000, -3e6]] and s['fam_train']['qtank'][1] == [500, -1e5]
    assert s['guard'] == [[0, dict(qtank=1.), True]]
    json.dumps(out, allow_nan=False)          # raises on any NaN or Infinity left


def test_the_publisher_refuses_to_write_non_finite_json(tmp_path):
    from experiments.shared_rl_view import publish_bilingual_site as pub
    with pytest.raises(ValueError):
        pub.write(tmp_path / 'x.json', dict(a=[1, float('-inf')]))
    assert not (tmp_path / 'x.json').exists()


def test_the_build_check_rejects_a_release_with_non_finite_json(tmp_path):
    from experiments.shared_rl_view import publish_bilingual_site as pub
    (tmp_path / 'ok.json').write_text('{"a":[1,2.5]}')
    pub.strict_json(tmp_path)
    (tmp_path / 'tank-replay.json').write_text('{"a":[1,-Infinity]}')
    with pytest.raises(SystemExit, match='tank-replay.json'):
        pub.strict_json(tmp_path)


def fake_sim_run(root, now, final=False):
    """An aio.train round on the unified simulation core (sim-*, docs/sim.md): run.json with the references measured on its
    validation batch, a metrics row per iteration with val and lit every 5, final.json at the end (no done.json)."""
    d = root / 'sim-r1' / 'seed1'
    d.mkdir(parents=True)
    refs = dict(hold=dict(score=1862., cost=1e5, x_pi=1.636, con=.6), pi=dict(score=1138., cost=3e4, x_pi=1., con=.33),
                dmc=dict(score=563., cost=2e4, x_pi=.4947, con=.12), teacher=dict(score=474., cost=1.5e4, x_pi=.4165, con=.08))
    (d / 'run.json').write_text(json.dumps(dict(trainer='aio.train', envs=1024, sim_envs=192, window=240, hours=12., val_n=256, val_seed=10**6,
                                                lit=16, init='D:\\aiogym2-runs\\starts\\exp4-last.pt', val_refs=refs,
                                                lit_formula_pi=dict(wb=900., ory=1200., ty=1500., al=1100.), out=str(d))))
    rows, xs = [], {5: 1.20, 10: .80, 15: .78, 20: .74, 25: .76, 30: .70}
    for it in range(1, 31):
        r = dict(iter=it, beta=1., lr=3e-4, bc=.02, id=.3, student_score=900. - it, failed=0., samples=it*737280, hours=it/60)
        if it in xs:
            r['val'] = dict(score=1138.*xs[it], cost=2e4, failed=0, x_pi=xs[it], cost_x_pi=xs[it], win=.5, loss=.5, con=round(.4 - it/100, 4), con_pi=.33)
            r['lit'] = dict(wb=1.1, ory=.9, ty=.7, al=1.0 + it/100)
        rows.append(json.dumps(r))
    (d / 'metrics.jsonl').write_text('\n'.join(rows) + '\n')
    if final:
        (d / 'final.json').write_text(json.dumps(dict(iter=30, samples=30*737280, hours=12.0, val=dict(score=700., x_pi=.62, cost_x_pi=.6, win=.6, loss=.4, failed=0),
                                                      val_refs=refs, best=.62, lit=dict(wb=1., ory=.8, ty=.6, al=.9))))
    return d


def test_a_simulation_core_round_is_read_as_multiples_of_the_formula_pi(tmp_path, monkeypatch):
    """aio.train writes val.x_pi (the formula PI on the same batch = 1), lit per literature model and final.json; best.pt is
    every strictly lower x_pi (docs/sim.md 8). The page draws the multiples with the batch's references from run.json."""
    now = time.time()
    notes_fixture(tmp_path, monkeypatch)
    n = json.loads(sp.NOTES.read_text(encoding='utf-8'))
    n['runs'].insert(0, dict(id='sim-r1', role='current', task='sim', label='Simulation core · round 1 · seed 1', label_zh='新仿真核心 · 第一轮 · 种子 1',
                             log_name='Simulation core round 1', log_name_zh='新核心第一轮'))
    n['timeline'] = []   # a hand-written entry within 30 minutes of now would hide the automatic ones
    sp.NOTES.write_text(json.dumps(n), encoding='utf-8')
    fake_sim_run(tmp_path / 'runs', now)
    code = sp.REMOTE.replace(repr(sp.RUNS), repr(str(tmp_path / 'runs')))
    remote = json.loads(subprocess.run([sys.executable, '-'], input=code, text=True, capture_output=True, check=True).stdout)
    out = sp.build(remote=remote)
    r = next(x for x in out['runs'] if x['id'] == 'sim-r1')
    s = r['seeds'][0]
    assert sp.family('sim-r1') == 'sim' and sp.current(out['runs'], 'sim') is r and s['state'] == 'running'
    assert s['config']['init'] == 'starts/exp4-last.pt' and s['config']['val_refs']['dmc']['x_pi'] == .4947 and s['config']['hours'] == 12.
    assert [v[1] for v in s['sim_val']] == [1.2, .8, .78, .74, .76, .7] and s['best_sim']['x_pi'] == .7 and s['best_sim']['it'] == 30
    assert s['lit_val']['al'][0] == [5*737280, 1.05] and s['best_sim']['lit']['ty'] == .7 and s['last_sim']['it'] == 30
    assert s['samples'] == 30*737280 and len(s['student']) == 30 and s['val'] == []
    assert s['best_sim']['con'] == .1 and s['best_sim']['con_pi'] == .33 and s['config']['val_refs']['teacher']['con'] == .08
    # the log: the first validation (worse than formula PI: bad news), then each best.pt 5% below the last logged one
    log = [e for e in out['timeline'] if e.get('auto') and '新核心第一轮' in e['text_zh']]
    assert [e['text_zh'] for e in log][::-1] == [
        '新核心第一轮开训。', '新核心第一轮 it 5：验证 1.2× 公式 PI，第一次验证，还不如公式 PI。', '新核心第一轮 it 10：验证 0.8× 公式 PI，比 1.2× 低 33.3%，换为保留权重。',
        '新核心第一轮 it 20：验证 0.74× 公式 PI，比 0.8× 低 7.5%，换为保留权重。', '新核心第一轮 it 30：验证 0.7× 公式 PI，比 0.74× 低 5.4%，换为保留权重。']
    assert [e['tone'] for e in log][::-1] == ['good', 'bad', 'good', 'good', 'good']
    # final.json ends it: done, and its final validation counts for best.pt
    d = tmp_path / 'runs2'
    fake_sim_run(d, now, final=True)
    remote = json.loads(subprocess.run([sys.executable, '-'], input=sp.REMOTE.replace(repr(sp.RUNS), repr(str(d))), text=True, capture_output=True, check=True).stdout)
    out = sp.build(remote=remote)
    s = next(x for x in out['runs'] if x['id'] == 'sim-r1')['seeds'][0]
    assert s['state'] == 'done' and s['best_sim']['x_pi'] == .62 and s['best_sim']['lit']['ory'] == .8
    assert any(e.get('auto') and e['text_zh'] == '新核心第一轮跑完：12.0 小时，2,212 万样本。' for e in out['timeline'])


def test_simulation_core_notice_refuses_once_its_start_has_long_passed():
    import datetime as dt
    now = dt.datetime(2026, 11, 10, 22, 0, tzinfo=sp.CST)
    out = dict(runs=[], notes_as_of='2026-11-10T15:00+08:00', generated_at=now.isoformat(timespec='minutes'),
               sim_rl=dict(state='pending', expected_start='2026-11-10T17:45+08:00'))
    errors = [p['text_zh'] for p in sp.freshness(out)['problems'] if p['level'] == 'error']
    assert len(errors) == 1 and '新仿真核心' in errors[0] and 'sim_rl' in errors[0]


def test_simulation_core_replay_is_the_cores_own_episode_in_the_page_format(tmp_path):
    """research_site_sim_replay reshapes aio.sim.replay: every row's outside flags add up to each controller's score,
    a zone row's limits move with its limit events, the start of a round is marked as its starting weights."""
    from experiments.shared_rl_view import research_site_sim_replay as R
    H, k0, dt = 720, 30, 30.
    rows = [dict(role='tracked', integrating=False, band=.02, mv=1), dict(role='tracked', integrating=True, band=.03, mv=0),
            dict(role='zone', integrating=False, band=.02, mv=None), dict(role='dv', integrating=False, band=.02, mv=None)]
    seg = lambda k, hi, w: dict(k=k, lo=[None, None, .1, None], hi=[None, None, hi, None], weight=[w, 1., 1., 1.])   # noqa: E731
    def ctl(off):
        y = [[.5 + off*(k > 400), .4, .8 + .2*(k > 500), .3] for k in range(H)]
        tgt = [[.5 if k < 299 else .55, .4, .5, .5] for k in range(H)]
        out = [[int(abs(y[k][0]-tgt[k][0]) > .02), 0, int(y[k][2] > (.9 if k >= 450 else .95)), 0] for k in range(H)]
        return dict(name_zh='', score=sum(sum(o) for k, o in enumerate(out) if k >= k0), cost=1., failed=False, y=y, ym=y, target=tgt, ref=tgt,
                    u=[[.5, .5] for _ in range(H)], outside=out)
    raw = dict(seed=2_000_000, n=64, index=7, horizon=H, dt_s=dt, k_first=k0, pick='median step-test DMC score of the batch', pick_zh='',
               struct=dict(nm=2, nz=1, nd=1, nq=1, arch='separation', time_scale=20., dead_ratio=.6, linear=False, load_dependent=True, rows=rows,
                           mvs=[dict(slew=.08, sign=1, row=1), dict(slew=.05, sign=-1, row=0)]),
               events=[dict(k=30, kind='priority', ramp=0), dict(k=300, kind='setpoint', ramp=0), dict(k=450, kind='limit', ramp=0), dict(k=600, kind='load', ramp=40)],
               segments=[seg(0, .95, 1.), seg(30, .95, 2.), seg(300, .95, 2.), seg(450, .9, 2.), seg(600, .9, 2.)],
               controllers=dict(hold=ctl(.1), pi=ctl(.05), dmc=ctl(0.), teacher=ctl(0.), aio=ctl(.01)),
               source=dict(file='sim-r1/seed1/best.pt', trainer='aio.train', iter=40, samples=29491200, hours=.8, x_pi=.9))
    d = R.convert(raw)
    assert d['version'] == 1 and d['process'] == 'sim'
    assert [r['controller'] for r in d['runs']] == ['rl', 'fpi', 'dmc', 'teacher', 'hold']
    for r, (q, c) in zip(d['runs'], [(q, raw['controllers'][q]) for q in R.ORDER]):
        assert sum(r['cols']['out']) == c['score'] and r['total_s'] == c['score']*dt and len(r['cols']['y0']) == H + 1
        assert r['track_s'] + r['viol_s'] == r['total_s'] and r['cv0_s'] + r['cv2_viol_s'] == r['total_s']
    z = next(c for c in d['cvs'] if c['zone'])
    assert z['lims'] == [[0., 10., 95.], [450*dt, 10., 90.]] and z['label_zh'] == '区间量 Z1'
    assert [c['label_zh'] for c in d['cvs']] == ['CV1（与 MV2 配对）', 'CV2（与 MV1 配对），积分型', '区间量 Z1', '实测扰动 D1']
    assert d['mvs'][1]['label_zh'] == 'MV2，反作用' and d['start']['w_cv0'] == 1. and d['start']['hi_cv2'] == 95.
    ev = d['events']
    assert [e['kind'] for e in ev] == ['other', 'setpoint', 'other', 'disturbance'] and ev[0]['state']['w_cv0'] == 2.
    assert ev[1]['label_zh'] == '目标 CV1 55.00%' and ev[2]['label_zh'] == '限值 Z1 10.0–90.0%' and ev[3]['label_zh'].startswith('不可测负荷变化，40 个决策')
    rl = d['runs'][0]
    assert rl['source']['run'] == 'sim-r1' and rl['source']['x_pi'] == .9 and rl['label_zh'] == 'AIO 模型（现行权重）' and rl['source']['iter'] == 40
    start = R.convert(dict(raw, source=dict(file='aiogym2-runs/starts/exp4-last.pt')), start_of='sim-r1/seed1')['runs'][0]
    assert start['source']['which'] == 'init' and start['source']['x_pi'] is None and '起点权重（exp4-last.pt）' in start['label_zh']
    with pytest.raises(AssertionError):
        R.convert(dict(raw, source=dict(file='aiogym2-runs/starts/exp4-last.pt')))   # starting weights must be marked as such
    from experiments.shared_rl_view import site_i18n
    assert not site_i18n.missing_zh(d, 'sim-replay.json')


def test_simulation_core_replay_names_the_constrained_operating_points():
    """Events that take the unit to a constrained point (aio.sim.prior: a target beyond reach, a load the valves cannot
    follow, an MV limit narrowed) and the one that brings it back are shown, with the MV limit that moved."""
    from experiments.shared_rl_view import research_site_sim_replay as R
    H, k0, dt = 720, 30, 30.
    rows = [dict(role='tracked', integrating=False, band=.02, mv=0), dict(role='tracked', integrating=False, band=.03, mv=1)]
    seg = lambda k, hi, con: dict(k=k, lo=[None, None], hi=[None, None], weight=[1., 1.], u_lo=[.05, .05], u_hi=[.95, hi],   # noqa: E731
                                  reachable=[.5, .4], constrained=con)
    tgt = [[.7 if 99 <= k < 199 else .5, .4] for k in range(H)]
    y = [[.5, .4] for _ in range(H)]
    out = [[0, 0] for _ in range(H)]
    ctl = lambda: dict(name_zh='', score=0, cost=1., failed=False, y=y, ym=y, target=tgt, ref=tgt, u=[[.5, .5] for _ in range(H)],   # noqa: E731
                       outside=out, ok_lo=y, ok_hi=y)
    raw = dict(seed=2_000_000, n=64, index=7, horizon=H, dt_s=dt, k_first=k0, pick='median step-test DMC score of the batch', pick_zh='',
               struct=dict(nm=2, nz=0, nd=0, nq=1, arch='interacting', time_scale=20., dead_ratio=.6, linear=True, load_dependent=False, rows=rows,
                           mvs=[dict(slew=.08, sign=1, row=0), dict(slew=.05, sign=1, row=1)]),
               events=[dict(k=100, kind='over_setpoint', ramp=0), dict(k=200, kind='return', ramp=0), dict(k=300, kind='mv_limit', ramp=0),
                       dict(k=400, kind='return', ramp=0), dict(k=500, kind='over_load', ramp=30), dict(k=600, kind='return', ramp=20)],
               segments=[seg(0, .95, False), seg(100, .95, True), seg(200, .95, False), seg(300, .6, True), seg(400, .95, False), seg(500, .95, True),
                         seg(600, .95, False)],
               controllers=dict(hold=ctl(), pi=ctl(), dmc=ctl(), teacher=ctl()))
    ev = R.convert(raw)['events']
    assert [e['kind'] for e in ev] == ['setpoint', 'setpoint', 'other', 'other', 'disturbance', 'other']
    assert ev[0]['label_zh'] == '目标超出可达范围：CV1 70.00%' and ev[1]['target'] == [50., 40.] and ev[1]['label_zh'] == '回到原工况'
    assert ev[2]['label_zh'] == '操作员收窄 MV2 5.0–60.0%' and ev[4]['label_zh'].startswith('负荷超出阀门能力，30 个决策')
    assert ev[5]['label_zh'].startswith('回到原工况，20 个决策')
    from experiments.shared_rl_view import site_i18n
    assert not site_i18n.missing_zh(R.convert(raw), 'sim-replay.json')


def test_simulation_core_replay_draws_mv_limits_and_the_wider_stretch_at_a_constrained_point():
    """The page draws what the core scores: an MV's operator limits as steps, and where a constrained operating point
    widens what a tracked CV may do without counting as outside (ok_lo..ok_hi), that stretch per run; the recount of
    a file without the core's flags uses it too."""
    from experiments.shared_rl_view import research_site_sim_replay as R
    H, k0, dt = 720, 30, 30.
    rows = [dict(role='tracked', integrating=False, band=.02, mv=0), dict(role='tracked', integrating=False, band=.03, mv=1)]
    seg = lambda k, lo1, con: dict(k=k, lo=[None, None], hi=[None, None], weight=[1., 1.], u_lo=[.05, lo1], u_hi=[.95, .95],   # noqa: E731
                                   reachable=[.5, .36 if con else .4], constrained=con)
    ref = [[.5, .4] for _ in range(H)]
    y = [[.5, .36 if 300 <= k < 400 else .4] for k in range(H)]            # held at the reachable optimum while MV2 is narrowed
    wide = lambda k: 300 <= k < 400                                          # noqa: E731
    ok_lo = [[.48, (.33 if wide(k) else .37)] for k in range(H)]
    ok_hi = [[.52, .43] for k in range(H)]
    ctl = lambda out: dict(name_zh='', score=0, cost=1., failed=False, y=y, ym=y, target=ref, ref=ref, u=[[.5, .5] for _ in range(H)],   # noqa: E731
                           outside=out, ok_lo=ok_lo, ok_hi=ok_hi)
    raw = dict(seed=2_000_000, n=64, index=21, horizon=H, dt_s=dt, k_first=k0,
               pick='median step-test DMC score of the episodes that reach a constrained operating point', pick_zh='',
               struct=dict(nm=2, nz=0, nd=0, nq=0, arch='interacting', time_scale=20., dead_ratio=.6, linear=True, load_dependent=False, rows=rows,
                           mvs=[dict(slew=.08, sign=1, row=0), dict(slew=.05, sign=1, row=1)]),
               events=[dict(k=300, kind='mv_limit', ramp=0), dict(k=400, kind='return', ramp=0)],
               segments=[seg(0, .05, False), seg(300, .6, True), seg(400, .05, False)],
               controllers=dict(pi=ctl([[0, 0] for _ in range(H)]), hold=ctl(None)))
    d = R.convert(raw)
    m = d['mvs'][1]
    assert (m['lower'], m['upper']) == (5., 95.) and m['lims'] == [[0., 5., 95.], [300*dt, 60., 95.], [400*dt, 5., 95.]] and 'lims' not in d['mvs'][0]
    assert d['start']['ulo_u1'] == 5. and d['events'][0]['state']['ulo_u1'] == 60. and d['events'][1]['state']['ulo_u1'] == 5.
    for r in d['runs']:                     # CV1's stretch is the band everywhere: not carried; CV2's is wider for a while
        assert list(r['ok']) == ['y1'] and len(r['ok']['y1'][0]) == H + 1
        assert r['ok']['y1'][0][301] == 33. and r['ok']['y1'][0][1] == 37. and r['ok']['y1'][1][350] == 43.
    assert d['runs'][1]['controller'] == 'hold' and d['runs'][1]['total_s'] == 0      # recounted: 36 % is inside 33-43 %
    assert '约束工况' in d['footnote_zh'] and '在到过约束工况的回合里' in d['footnote_zh']
    raw['controllers']['hold']['ok_lo'] = None                                          # without ok: 36 % is 2 bands off 40 %
    assert R.recount(dict(raw['controllers']['hold'], ok_lo=None), rows, lambda k: raw['segments'][0])[350] == [0, 1]


def test_simulation_core_replay_shows_a_zone_limit_in_force_and_an_mv_out_of_service():
    """A zone limit moved past where the unit runs and an MV taken out of service are named; the zone CV carries how far
    beyond its limit it may be before it counts where the best steady point has it there (`allow`), and the recount of
    a file without the core's flags uses it."""
    from experiments.shared_rl_view import research_site_sim_replay as R
    H, k0, dt = 720, 30, 30.
    rows = [dict(role='tracked', integrating=False, band=.02, mv=0), dict(role='tracked', integrating=False, band=.03, mv=1),
            dict(role='zone', integrating=False, band=.02, mv=None)]
    seg = lambda k, zlo, allow=0., u1=(.05, .95): dict(k=k, lo=[None, None, zlo], hi=[None, None, None], weight=[1., 1., 1.],   # noqa: E731
                                                       u_lo=[.05, u1[0]], u_hi=[.95, u1[1]], reachable=[.5, .4, .5], allow=[0., 0., allow],
                                                       constrained=allow > 0 or u1 != (.05, .95))
    tgt = [[.6 if 299 <= k < 399 else .5, .4, .5] for k in range(H)]
    y = [[.5, .4, .425 if 100 <= k < 200 else .5] for k in range(H)]        # the zone CV sits 1.25 bands below its new limit: inside the allowance
    ctl = lambda out: dict(name_zh='', score=0, cost=1., failed=False, y=y, ym=y, target=tgt, ref=tgt, u=[[.5, .5] for _ in range(H)],   # noqa: E731
                           outside=out, ok_lo=[[.48, .37] for _ in range(H)], ok_hi=[[.62, .43] for _ in range(H)])
    raw = dict(seed=2_000_000, n=64, index=3, horizon=H, dt_s=dt, k_first=k0,
               pick='median step-test DMC score of the episodes that reach a constrained operating point', pick_zh='',
               struct=dict(nm=2, nz=1, nd=0, nq=0, arch='interacting', time_scale=20., dead_ratio=.6, linear=True, load_dependent=False, rows=rows,
                           mvs=[dict(slew=.08, sign=1, row=0), dict(slew=.05, sign=1, row=1)]),
               events=[dict(k=100, kind='zone_limit', ramp=0), dict(k=200, kind='return', ramp=0), dict(k=300, kind='mv_out', ramp=0),
                       dict(k=400, kind='return', ramp=0)],
               segments=[seg(0, .3), seg(100, .45, 1.5), seg(200, .3), seg(300, .3, u1=(.48, .52)), seg(400, .3)],
               controllers=dict(pi=ctl([[0, 0, 0] for _ in range(H)]), hold=ctl(None)))
    d = R.convert(raw)
    ev = d['events']
    assert [e['kind'] for e in ev] == ['other', 'other', 'setpoint', 'setpoint']
    assert ev[0]['label_zh'] == '区间限值收紧到现值之外：Z1 不低于 45.0%' and ev[2]['label_zh'] == '退出控制：MV2 停在 50.0%；目标 CV1 60.00%'
    z = next(c for c in d['cvs'] if c['zone'])
    assert z['allow'] == [[0., 0.], [100*dt, 3.], [200*dt, 0.]] and z['lims'][1] == [100*dt, 45., None]
    assert d['mvs'][1]['lims'][1] == [300*dt, 48., 52.]
    assert d['runs'][1]['controller'] == 'hold' and d['runs'][1]['total_s'] == 0      # recounted: 42.5 % is inside 45 % less 1.5 bands
    assert R.recount(dict(raw['controllers']['hold']), rows, lambda k: dict(raw['segments'][1], allow=None))[150] == [0, 0, 1]
    from experiments.shared_rl_view import site_i18n
    assert not site_i18n.missing_zh(d, 'sim-replay.json')


def test_simulation_core_replay_names_a_spare_mvs_resting_value_and_the_push_to_its_limit():
    """A fat plant: the resting value of a spare MV is a tracked channel of its own (R<MV>, `rest`), the process CVs keep
    their numbering, a push names the resting value it moves and carries the wider stretch, and an event kind the
    converter does not know is refused, not skipped."""
    import pytest
    from experiments.shared_rl_view import research_site_sim_replay as R
    H, k0, dt = 720, 30, 30.
    rows = [dict(role='tracked', integrating=False, band=.02, mv=1, rest=False), dict(role='tracked', integrating=False, band=.08, mv=0, rest=True),
            dict(role='zone', integrating=False, band=.02, mv=None, rest=False)]
    seg = lambda k, con=False: dict(k=k, lo=[None, None, .3], hi=[None, None, None], weight=[1., 1., 1.], u_lo=[.05, .05], u_hi=[.95, .95],   # noqa: E731
                                    reachable=[.5, .8 if con else .4, .5], allow=[0., 0., 0.], constrained=con)
    push = lambda k: 199 <= k < 399                                                     # noqa: E731
    tgt = [[.5, .92 if push(k) else .4, .5] for k in range(H)]
    y = [[.5, .8 if push(k) else .4, .5] for k in range(H)]                             # the zone CV stops the MV at 80 %
    ctl = dict(name_zh='', score=0, cost=1., failed=False, y=y, ym=y, target=tgt, ref=tgt, u=[[.8 if push(k) else .4, .5] for k in range(H)],
               outside=[[0, 0, 0] for _ in range(H)], ok_lo=[[.48, .72 if push(k) else .32] for k in range(H)],
               ok_hi=[[.52, 1. if push(k) else .48] for k in range(H)])
    raw = dict(seed=2_000_000, n=64, index=5, horizon=H, dt_s=dt, k_first=k0,
               pick='median step-test DMC score of the episodes that reach a constrained operating point', pick_zh='',
               struct=dict(nm=2, nz=1, nd=0, nq=0, spare=1, arch='interacting', time_scale=20., dead_ratio=.6, linear=True, load_dependent=False, rows=rows,
                           mvs=[dict(slew=.08, sign=1, row=1), dict(slew=.05, sign=1, row=0)]),
               events=[dict(k=200, kind='push', ramp=0), dict(k=400, kind='return', ramp=0)],
               segments=[seg(0), seg(200, True), seg(400)], controllers=dict(pi=ctl))
    d = R.convert(raw)
    cv, rest, zone = d['cvs']
    assert [c['tag'] for c in d['cvs']] == ['CV1', 'R1', 'Z1'] and rest.get('rest') is True and 'rest' not in cv
    assert cv['label'] == 'CV1 (paired with MV2)' and rest['label_zh'] == 'MV1 的静置值（它的流量）' and rest['band'] == 8.
    ev = d['events']
    assert [e['kind'] for e in ev] == ['setpoint', 'setpoint'] and ev[0]['label_zh'].startswith('静置值推到阀门限值：R1 92')
    assert ev[0]['target'] == [50., 92., ] and ev[1]['target'] == [50., 40.]
    assert d['start']['nr'] == 1 and d['start']['nt'] == 2
    assert '1 个富余' in d['footnote_zh'] and '1 个跟踪 CV' in d['footnote_zh'] and '1 spare' in d['footnote']
    ok = d['runs'][0]['ok']
    assert list(ok) == ['y1'] and ok['y1'][0][250] == 72. and ok['y1'][1][250] == 100.          # target to the reachable value, plus its wide band
    from experiments.shared_rl_view import site_i18n
    assert not site_i18n.missing_zh(d, 'sim-replay.json')
    raw['events'][0]['kind'] = 'something_new'
    with pytest.raises(ValueError):
        R.convert(raw)

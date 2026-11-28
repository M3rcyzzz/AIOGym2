"""A long stay on the same plants: windows of HORIZON decisions chained with Program.next and resume."""
import pytest
import torch

from aio.sim import control as CT, core as C, mech as ME, prior as PR

HORIZON, BIG = PR.HORIZON, PR.BIG


@pytest.fixture(scope='module')
def lin():
    a = C.build(24, 7)
    return a, a.next()


@pytest.fixture(scope='module')
def eng():
    a = C.build(8, 7, mech=1., plants=ME.pool(['qtank', 'hex']))
    return a, a.next()


def test_a_fresh_batch_is_what_it_was():
    """Keeping the tail draws nothing: the episodes of a batch are those of the generator alone."""
    segs, tgts, infos, seg_k, kinds, delay, ramp = PR.draw_episode(PR.Rng(3), 16)
    keep = {}
    again = PR.draw_episode(PR.Rng(3), 16, keep=keep)
    assert torch.equal(seg_k, again[3]) and torch.equal(kinds, again[4]) and torch.equal(ramp, again[6])
    assert all(torch.equal(segs[-1][q], again[0][-1][q]) for q in ('K', 'dead', 'dlev', 'fs'))
    assert set(keep) == {'base', 'before', 'before_m', 'last_m', 'cur', 'home', 'away', 'k', 'turn', 'riding', 'cause', 'mend'}


def test_the_next_window_carries_the_last_two_segments(lin):
    a, b = lin
    n = a.n
    last = (a.seg_k < BIG).sum(1)-1                               # the last segment in use
    ar = torch.arange(n)
    assert torch.equal(b.P[:, 0], a.P[:, 0])                      # the plant as commissioned
    assert torch.equal(b.P[:, 2], a.P[ar, last])
    assert torch.equal(b.P[:, 1], a.P[ar, (last-1).clamp_min(0)])
    assert torch.equal(b.T[:, 2], a.T[ar, last])
    assert torch.equal(b.seg_k[:, 2], a.seg_k[ar, last]-HORIZON)
    assert torch.equal(b.kinds[:, 2], a.kinds[ar, last]) and torch.equal(b.ramp[:, 2], a.ramp[ar, last])
    assert bool((b.seg_k[:, :3] <= 0).all()) and bool((b.seg_k[:, 3:] >= 1).all())
    assert bool((b.seg_at(0) == 2).all()) and bool((b.k_first == 0).all())
    # how long since the last event: counted across the seam
    had = a.kinds[ar, last] >= 0
    assert torch.equal(b.since(0)[had], (HORIZON-a.seg_k[ar, last])[had]) and torch.equal(a.since(HORIZON)[had], b.since(0)[had])
    assert bool((a.since(0) == BIG).all())


def test_a_change_on_its_way_goes_on_across_the_seam():
    """The plant during the first decision of the next window is the one the window before would have had
    at decision HORIZON."""
    a = C.build(192, 7, teacher=False)
    b = a.next()
    ar = torch.arange(a.n)
    last = (a.seg_k < BIG).sum(1)-1
    w = ((HORIZON+1-a.seg_k[ar, last])/a.ramp[ar, last].clamp_min(1.)).clamp(0., 1.)[:, None]
    P0, P1 = a.P[ar, (last-1).clamp_min(0)], a.P[ar, last]
    want = P0+w*(P1-P0)
    got = b.params(b.seg_at(0), 0)
    flat = torch.cat([got[q].flatten(1) for q in C.SEG_FIELDS], 1)
    assert torch.allclose(flat, want, atol=1e-6)
    assert bool(((w > 0) & (w < 1)).any()), 'no episode of this batch has a ramp across the seam: take another seed'


def test_events_keep_coming_and_the_teacher_follows(lin):
    a, b = lin
    new = (b.seg_k[:, 3:] < BIG).sum(1).float()
    assert new.mean() > 4 and bool((b.seg_k[:, 3] < 200).all())
    # the teacher's model of a carried process event is renewed when it would have been
    proc = (b.kinds[:, 2] == PR.PROC) | (b.kinds[:, 2] == PR.BOTH)
    assert bool((b._teach['late'][:, 2] <= b.seg_k[:, 2]+40*proc).all())
    assert bool((b._teach['fixed'][:, 1:] == BIG).all())          # the commissioning model stays the commissioning model
    assert torch.equal(b.pi['kc'], a.pi['kc']) and torch.equal(b.S0, a.S0)


def test_the_state_goes_on_without_a_jump(lin):
    a, b = lin
    ra = CT.rollout(a, CT.Teacher('late'))
    st = ra['st']
    y0 = st['yt'].clone()
    rb = CT.rollout(b, CT.Teacher('late'), steps=1, st=st)
    move = ((rb['st']['yt']-y0).abs()/a.c['bandf']*(a.c['trk']+a.c['zon'])).amax(-1)
    assert move.median() < .2 and move.max() < 3.                 # one decision's worth, in bands
    for q in ('u', 'xv', 'v', 'ou', 'target'):
        assert rb['st'][q].shape == st[q].shape
    full = CT.rollout(b, CT.Teacher('late'), st=st)
    hold = CT.rollout(b, CT.Hold(), st=st)
    assert not bool(full['failed'].any())
    assert full['score'].mean() < .5*hold['score'].mean()         # the teacher holds the plants it is handed mid-way


def test_a_failed_unit_is_started_again(lin):
    a, b = lin
    st = C.reset(a)
    st['fail'] = st['fail'].clone(); st['fail'][3] = True
    st['v'] = st['v'].clone(); st['v'][3] = 40.
    new = C.resume(b, st)
    rest = C.reset(b)
    assert not bool(new['fail'].any())
    assert torch.equal(new['v'][3], rest['v'][3]) and torch.equal(new['v'][4], st['v'][4])
    r = CT.rollout(b, CT.Hold(), steps=1, st=st)
    assert r['restarted'].nonzero().flatten().tolist() == [3]


def test_wear_does_not_pile_up():
    """Process events alone only wear a plant down (actuators lose capacity, valves stick); on a long stay
    maintenance puts things back, and after eight windows the plants are no worse than after one."""
    prog = C.build(96, 3, teacher=False)

    def weakest(p):
        b, c = p.tail['lin']['base'][0], p.tail['lin']['cur'][0]
        NMV = b['mvm'].shape[1]
        kb = (b['K'][:, :, :NMV]*b['trk'][:, :, None]).norm(dim=1)
        kc = (c['K'][:, :, :NMV]*b['trk'][:, :, None]).norm(dim=1)
        return torch.where(b['mvm'] > 0, kc/kb.clamp_min(1e-30), torch.ones_like(kb)).amin(1)
    first = weakest(prog)
    for _ in range(7):
        prog = prog.next()
    late = weakest(prog)
    assert late.median() > .9 and late.quantile(.1) > first.quantile(.1)-.15


def test_plants_of_the_engine_go_on(eng):
    a, b = eng
    assert b.mech.names() == a.mech.names() and b.mech.s0 == 2
    assert torch.equal(b.seg_k[:, 2], a.seg_k[torch.arange(a.n), (a.seg_k < BIG).sum(1)-1]-HORIZON)
    ga, gb = a.mg[0], b.mg[0]
    last = (a.seg_k[ga['idx']] < BIG).sum(1)-1
    assert torch.equal(gb['P'][:, 2], ga['P'][ga['ar'], last])    # the equipment as the window before left it
    ra = CT.rollout(a, CT.Teacher('late'))
    rb = CT.rollout(b, CT.Teacher('late'), st=ra['st'])
    hold = CT.rollout(b, CT.Hold(), st=ra['st'])
    assert not bool(rb['failed'].any())
    assert rb['score'].mean() < .6*hold['score'].mean()
    c = b.next()
    assert c.mech.names() == a.mech.names()


def test_the_tail_is_all_the_next_window_needs(lin):
    """Drawn in another process from the pickled tail and a seed: the same window."""
    import pickle
    a, b = lin
    tail = pickle.loads(pickle.dumps(a.tail))
    c = C.build(a.n, a.seed-90_001+1, after=tail)
    assert torch.equal(c.P, b.P) and torch.equal(c.T, b.T) and torch.equal(c.seg_k, b.seg_k)
    assert c.tail['window'] == 1 and a.tail['window'] == 0
    d = C.build(a.n, 12345, after=tail)                          # another seed: other events on the same plants
    assert torch.equal(d.P[:, :3], b.P[:, :3]) and not torch.equal(d.seg_k, b.seg_k)
    with pytest.raises(AssertionError):
        C.build(a.n+1, 0, after=tail)


def test_the_references_along_a_stay(lin):
    """aio.sim.stay: every controller on its own stay, read window by window on the scale of that window."""
    from aio.sim import stay as ST
    a, b = lin
    out = ST.references(a.n, 7, 2, ctl=('hold', 'pi'))            # no teacher among them: each its own stay
    pi = CT.PI()
    ra = CT.rollout(a, pi)
    rb = CT.rollout(b, pi, st=ra['st'])                           # (the same controller: its loops stay closed)
    assert torch.equal(out['score']['pi'][0], ra['score']) and torch.equal(out['score']['pi'][1], rb['score'])
    led = ST.references(a.n, 7, 2, ctl=('hold', 'pi'), lead='pi')  # the others take each window over from the lead
    assert torch.equal(led['score']['pi'], out['score']['pi']) and torch.equal(led['score']['hold'][0], out['score']['hold'][0])
    assert torch.equal(led['score']['hold'][1], CT.rollout(b, CT.Hold(), st=ra['st'])['score'])
    assert led['score']['hold'][1].mean() < out['score']['hold'][1].mean()        # held from where a controller left them, not from the start
    rows = ST.table(out)
    assert [r['decisions'] for r in rows] == [(0, HORIZON), (HORIZON, 2*HORIZON), (0, 2*HORIZON)]
    assert all(r['hold']['points'] == 0 and r['pi']['points'] == pytest.approx(100.) for r in rows)   # two controllers: the best is the PI wherever it beats holding
    quiet = list(ST.windows(8, 7, 3, menu=[None, PR.Menu(rate=(0, 0, 0, 0, 0)), None], teacher=False))
    assert [int((p.seg_k[:, p.s0+1:] < BIG).any()) for p in quiet] == [1, 0, 1]


def test_room_for_the_events_of_a_window_that_carries_on():
    """prior.NSEG_STAY: such a window has room for its events to the end. With NSEG (the batches as they are) more
    than half of the units have used theirs up before it ends."""
    from aio.sim import stay as ST

    def stayed(nseg):
        tail, out, events = None, [], []
        for w in range(3):
            keep = {}
            seg_k = PR.draw_episode(PR.Rng(7+w), 96, tail=tail, keep=keep, nseg=nseg)[3]
            if tail is not None:
                new = (seg_k[:, PR.CARRIED:] < BIG).sum(1)
                assert int(new.max()) <= nseg-PR.CARRIED
                out.append(float((keep['k'] < HORIZON-30).float().mean()))        # the next event would still have been in this window
                events.append(float(new.float().mean()))
            tail = keep
        return sum(out)/len(out), sum(events)/len(events)
    (a, ea), (b, eb) = stayed(PR.NSEG), stayed(PR.NSEG_STAY)
    assert a > .4 and b < .05 and eb > ea+.8
    kw = dict(n=8, count=2, teacher=False)
    usual, same, roomy, dense = (list(ST.exam(q, nseg=ns, **kw)) for q, ns in (('usual', None), ('usual', PR.NSEG), ('usual', PR.NSEG_STAY), ('dense', PR.NSEG_STAY)))
    assert all(torch.equal(x.P, y.P) and torch.equal(x.seg_k, y.seg_k) for x, y in zip(usual, same))
    assert [p.nseg for p in roomy] == [PR.NSEG_STAY]*2 and [p.nseg for p in dense] == [24]*2      # (an exam that asks for more keeps its own)
    assert torch.equal(roomy[0].P[:, 0], usual[0].P[:, 0])        # the same plants as commissioned


def test_the_exams_are_fixed_batches():
    """aio.sim.stay.EXAMS: the same plants in every linear exam, each with its own kind of change."""
    from aio.sim import stay as ST
    kw = dict(n=12, count=3, teacher=False)
    new = lambda p: (p.seg_k[:, p.s0+1:] < BIG, p.kinds[:, p.s0+1:])                                 # noqa: E731
    usual, again = list(ST.exam('usual', **kw)), list(ST.exam('usual', **kw))
    assert all(torch.equal(a.P, b.P) and torch.equal(a.seg_k, b.seg_k) for a, b in zip(usual, again))
    for name in ('pulse', 'dead', 'quiet', 'blind', 'sparse', 'dense', 'ride', 'heavy', 'shift'):
        ws = list(ST.exam(name, **kw))
        if name == 'shift':                                       # two usual windows, then one process event a unit: the change that stays
            assert torch.equal(ws[1].seg_k, usual[1].seg_k) and torch.equal(ws[1].P, usual[1].P) and ST.EXAMS['shift']['read'] == 2
            u, k = new(ws[2])
            one = (u & ((k == PR.PROC) | (k == PR.BOTH))).sum(1)
            assert bool((one <= 1).all()) and int(one.sum()) >= ws[2].n-1
        if name == 'heavy':                                       # the gains are others after a process event, and no dead time moves
            turned = 0
            for p in ws:
                for e in range(p.s0+1, p.nseg):
                    m = (p.seg_k[:, e] < BIG) & (p.kinds[:, e] == PR.PROC)
                    a, b = p.params(torch.full((p.n,), e-1)), p.params(torch.full((p.n,), e))
                    assert torch.equal(a['dead'][m], b['dead'][m])
                    turned += int((a['K'][m] != b['K'][m]).flatten(1).any(1).sum())
            assert turned > 10
        if name == 'ride':
            assert sum(int((p.con & (p.kinds >= 0) & (p.kinds <= PR.PRIO)).sum()) for p in ws) > 10
        assert torch.equal(ws[0].P[:, 0], usual[0].P[:, 0]) and torch.equal(ws[-1].P[:, 0], usual[0].P[:, 0])      # the plants as commissioned
        used, kinds = zip(*(new(p) for p in ws))
        count = lambda *ks: sum(int((u & torch.isin(k, torch.tensor(ks))).sum()) for u, k in zip(used, kinds))    # noqa: E731
        if name == 'pulse':
            assert count(PR.RETURN) > 2*sum(int((u & (k == PR.RETURN)).sum()) for u, k in (new(p) for p in usual))
        if name == 'dead':
            for p in ws:
                for e in range(p.s0+1, p.nseg):
                    m = (p.seg_k[:, e] < BIG) & (p.kinds[:, e] == PR.PROC)
                    s0, s1 = torch.full((p.n,), e-1), torch.full((p.n,), e)
                    a, b = p.params(s0), p.params(s1)
                    assert all(torch.equal(a[q][m], b[q][m]) for q in ('K', 'p1r', 'gam', 'wa', 'fs', 'tlag'))
            assert count(PR.PROC) > 12
        if name == 'quiet':
            assert bool(used[0].any()) and bool(used[1].any()) and not bool(used[2].any())
        if name == 'blind':
            assert all(p.blind for p in ws) and torch.equal(ws[1].seg_k, usual[1].seg_k)
        if name in ('sparse', 'dense'):
            assert (count(*range(14)) > 1.4*sum(int(u.sum()) for u, _ in (new(p) for p in usual))) == (name == 'dense')
    m = list(ST.exam('mech', n=4, count=2, teacher=False))
    assert m[0].mech is not None and m[0].nl == 0 and m[1].mech.names() == m[0].mech.names()
    assert ST.EXAMS['quiet']['read'] == 2
    m = list(ST.exam('mech_new', n=5, count=2, teacher=False))
    assert set(m[0].mech.names()) <= {'amm', 'bldg', 'comp', 'heater', 'rsr'} and m[1].mech.names() == m[0].mech.names()


def test_the_references_of_an_exam_are_kept(tmp_path):
    from aio.sim import stay as ST
    progs = list(ST.exam('usual', n=8, count=2))
    f = tmp_path/'refs'/'usual.json'
    a = ST.kept(progs, f, ctl=('hold', 'pi', 'teacher'))
    assert f.exists() and len(a['score']['pi']) == 2 and len(a['score']['pi'][0]) == 8
    said = []
    b = ST.kept(progs, f, ctl=('hold', 'pi', 'teacher'), say=lambda *x: said.append(x))
    assert b['score'] == a['score'] and not said                 # read, not run again
    other = list(ST.exam('usual', n=8, count=2, seed=5))
    c = ST.kept(other, f, ctl=('hold', 'pi', 'teacher'), say=lambda *x: said.append(x))
    assert c['key'] != a['key'] and len(said) == 2               # another batch: computed anew, the file is its own now

"""The teacher whose prediction is corrected from the instruments alone (core.Program.set_teacher(meter=), core._meter):
off by default; with perfect instruments its estimates are the true values; a reading that is not new changes nothing;
a spike is set aside; the estimator goes through the windows of a stay and a restart; the step-test DMC starts it at
its hand-over."""
import torch

from aio.sim import control as CT, core as C, prior as PR, stay as ST, steptest as T

IDEAL = ('nw', 'cn_s', 'as', 'quant', 'an_T', 'spike', 'bad')
SHARP = dict(q_rest=1e3, q_lev=1e3, floor=1e-6, gate=0.)          # gains of 1: what the instruments show is taken as it is


def _ideal(prog, keep=()):
    for q in IDEAL:
        if q not in keep:
            prog.c[q] = torch.zeros_like(prog.c[q])
    return prog


def _run(prog, ctl, steps, st=None, each=None):
    st = C.reset(prog) if st is None else C.resume(prog, st)
    ctl.reset(prog)
    for k in range(steps):
        st, _ = C.step(prog, st, ctl(prog, st, k, C.observe(prog, st, k)), k)
        if each:
            each(k, st)
    return st


def test_by_default_the_teacher_takes_the_true_values_and_the_state_has_no_estimator():
    prog = C.build(6, 3)
    assert prog.meter is None
    st = _run(prog, CT.Teacher('fixed'), 5)
    assert not set(C.METER_KEYS) & set(st)
    assert C.seen(prog, st, 5) is st
    prog.set_teacher('fixed', False, True)
    assert prog.meter is not None and set(prog.meter) == {'use', 'f', 'L1', 'L2', 'gate', 'lag'}
    g = prog.meter
    prog.set_teacher('fixed', False, True)
    assert prog.meter is g                                        # the gains are worked out once per setting
    prog.set_teacher('fixed', False, {})
    assert prog.meter is g                                        # (a dict with nothing in it: the settings as they are)
    prog.set_teacher()
    assert prog.meter is None


def test_with_perfect_instruments_the_estimates_are_the_true_values():
    prog = _ideal(C.build(24, 11))
    nr, prc = prog.nr, prog.c['prc'][:, :prog.nr] > 0
    ig, rest = (prog.c['integ'][:, :nr] > 0) & prc, (prog.c['integ'][:, :nr] <= 0) & prc
    assert bool(ig.any()) and bool(rest.any())
    worst = dict(rest=0., level=0., inflow=0.)
    band = prog.c['sband'][:, :nr].clamp_min(1e-9)
    last = {}

    def each(k, st):
        off = st['fl'][:, :nr]-st['tF'][:, 0]
        worst['rest'] = max(worst['rest'], float((((st['eo']-off)/band).abs()*rest).max()))
        worst['level'] = max(worst['level'], float((((st['ev']-st['v'][:, :nr])/band).abs()*ig).max()))
        if 'off' in last and k > 2:                               # the inflow not explained: known one decision late
            worst['inflow'] = max(worst['inflow'], float((((st['eo']-last['off'])/prog.c['tint'][:, :nr]/band).abs()*ig).max()))
        last['off'] = off
    st = _run(prog, CT.Teacher('fixed', meter=SHARP), 240, each=each)
    assert worst['rest'] < 1e-3 and worst['level'] < 1e-3, worst
    assert worst['inflow'] < .05, worst                           # (bands of level a decision)
    a = CT.rollout(prog, CT.Teacher('fixed'))['score']
    b = CT.rollout(prog, CT.Teacher('fixed', meter=SHARP))['score']
    assert abs(float(b.mean())-float(a.mean())) < .05*float(a.mean())
    assert set(C.METER_KEYS) <= set(st)


def test_only_the_rows_asked_for_are_taken_from_the_instruments():
    prog = C.build(12, 11)
    nr, prc = prog.nr, prog.c['prc'][:, :prog.nr] > 0
    ig, rest = (prog.c['integ'][:, :nr] > 0) & prc, (prog.c['integ'][:, :nr] <= 0) & prc
    st = _run(prog, CT.Teacher('fixed', meter=dict(rest=False)), 30)
    prog.set_teacher('fixed', False, dict(rest=False))
    s = C.seen(prog, st, 30)
    assert torch.equal(s['fl'][:, :nr][~ig], st['fl'][:, :nr][~ig]) and not torch.equal(s['fl'][:, :nr][ig], st['fl'][:, :nr][ig])
    assert torch.equal(s['fl'][:, nr:], st['fl'][:, nr:]) and torch.equal(s['v'][:, :nr][~ig], st['v'][:, :nr][~ig])
    prog.set_teacher('fixed', False, dict(levels=False))
    s = C.seen(prog, st, 30)
    assert torch.equal(s['fl'][:, :nr][~rest], st['fl'][:, :nr][~rest]) and torch.equal(s['v'], st['v'])      # (a row that reads a measured disturbance is not the plant's)
    assert not torch.equal(s['fl'][:, :nr][rest], st['fl'][:, :nr][rest])


def test_a_reading_that_is_not_new_changes_nothing():
    prog = _ideal(C.build(12, 11))
    prog.c['bad'] = torch.ones_like(prog.c['bad'])                # a bad value held for 5 to 30 decisions, one good reading, the next bad value
    ig = prog.c['integ'][:, :prog.nr] > 0
    seen = []
    _run(prog, CT.Teacher('fixed', meter=True), 40, each=lambda k, st: seen.append((st['eo'].clone(), st['bad'][:, :prog.nr] > 0)))
    for k in range(39):
        same = seen[k+1][0] == seen[k][0]
        assert bool(same[seen[k+1][1] & ~ig].all())               # no offset moves on a held value
    assert sum(int((seen[k+1][0] != seen[k][0])[~ig].any()) for k in range(39)) >= 1
    prog = _ideal(C.build(12, 11))
    prog.c['an_T'] = torch.full_like(prog.c['an_T'], 7.)*prog.c['prc']       # every row an analyser's: a sample every 7 decisions
    seen = []
    _run(prog, CT.Teacher('fixed', meter=True), 40, each=lambda k, st: seen.append(st['eo'].clone()))
    rest = ~ig & (prog.c['prc'][:, :prog.nr] > 0)
    moved = torch.stack([seen[k+1] != seen[k] for k in range(39)]).sum(0)[rest]
    assert int(moved.max()) <= 7 and float(moved.float().median()) >= 3     # an offset moves when its sample comes (if at all: a row at rest on its model has none)


def test_a_spike_is_set_aside():
    out = {}
    for q, gate in (('gated', 2.), ('taken', 0.)):
        prog = _ideal(C.build(24, 11))
        prog.c['spike'] = .1*prog.c['prc']                        # one reading in ten jumps by 3 to 8 bands
        out[q] = float(CT.rollout(prog, CT.Teacher('fixed', meter=dict(gate=gate)))['score'].mean())
    prog = _ideal(C.build(24, 11))
    clean = float(CT.rollout(prog, CT.Teacher('fixed', meter=True))['score'].mean())
    assert out['gated'] < out['taken']-50 and out['gated'] < 1.15*clean, (out, clean)


def test_the_estimator_goes_through_the_windows_of_a_stay_and_a_restart():
    a = C.build(8, 7)
    ctl = CT.Teacher('fixed', meter=True)
    r1 = CT.rollout(a, ctl)
    assert set(C.METER_KEYS) <= set(r1['st'])
    b = C.build(8, 8, after=a.tail)
    st = dict(r1['st'])
    st['fail'] = st['fail'].clone(); st['fail'][0] = True         # a unit that failed is started again at rest, its estimator with it
    got = C.resume(b, st)
    assert set(C.METER_KEYS) <= set(got) and all(bool(torch.isfinite(got[q]).all()) for q in C.METER_KEYS)
    assert torch.equal(got['eo'][1:], r1['st']['eo'][1:]) and float(got['eg'][0].abs().sum()) == 0
    r2 = CT.rollout(b, ctl, st=r1['st'])
    assert bool(torch.isfinite(r2['score']).all()) and set(C.METER_KEYS) <= set(r2['st'])
    r3 = CT.rollout(b, CT.Teacher('fixed'), st=r1['st'])          # another controller on the same state: the estimator is dropped
    assert not set(C.METER_KEYS) & set(r3['st'])


def test_the_step_test_dmc_starts_the_estimator_at_its_hand_over():
    ctl = T.StepTest(length=300, hold=20, design='open', meter=True)
    b = ST.references(4, 5, 2, lead=None, extra={'steptest': ctl})
    assert b['score']['steptest'].shape == (2, 4) and bool(torch.isfinite(b['score']['steptest']).all())
    assert ctl.meter is True


def test_at_a_larger_size_a_levels_inflow_walks_by_the_plants_own_model():
    """core.METER_BIG: at the larger sizes the walk of a level's net inflow is the plant's own (core._lev_walk, q_min at
    the least); at the default size it is METER's q_lev, as it was."""
    small = C.build(8, 3)
    small.set_teacher('late', False, True)
    want = C._meter_gains(small, dict(C.METER))
    assert all(torch.equal(small.meter[q], want[q]) for q in want)
    big = C.build(8, 3, dims='24x10')
    big.set_teacher('late', False, True)
    ig = (big.c['integ'][:, :big.nr] > 0) & (big.c['prc'][:, :big.nr] > 0)
    assert bool(ig.any())
    walk = C._lev_walk(big, big.nr)
    assert bool(torch.isfinite(walk).all()) and float(walk[ig].max()) > C.METER['q_min']
    own = C._meter_gains(big, dict(C.METER, lev_model=True))
    assert torch.equal(big.meter['L2'], own['L2'])
    assert not torch.equal(C._meter_gains(big, dict(C.METER))['L2'][ig], own['L2'][ig])
    big.set_teacher('late', True, True)                            # (what the model leaves unexplained depends on the loads it feeds forward)
    assert not torch.equal(big.meter['L2'][ig], own['L2'][ig])
    big.set_teacher('late', False, dict(lev_model=False))          # a field named in meter overrides METER_BIG's
    assert torch.equal(big.meter['L2'], C._meter_gains(big, dict(C.METER))['L2'])


def _moves(prog, steps, seed=1):
    g = torch.Generator().manual_seed(seed)
    out, a = [], None
    for k in range(steps):
        if k % 4 == 0:
            a = (2*torch.rand(prog.n, prog.nmv, generator=g, dtype=prog.dtype)-1)*.6*(torch.rand(prog.n, prog.nmv, generator=g, dtype=prog.dtype) < .6)
        out.append(a)
    return out


def _renewed(prog, k):
    """Who has the prediction worked out again at decision k, by the occasion (as core.step has it): the model renewed,
    a gradual change arrived in full, the turn of a plant whose clock moves."""
    sg = prog.teacher_seg(k+1)
    ramp = prog.ramp[prog.ar, sg]
    event = sg != prog.teacher_seg(k)
    own = prog.plant[prog.ar, sg] == prog.plant[prog.ar, prog.seg_at(k+1)]         # (its model is of the process as it now is)
    arrived = (ramp > 0) & (prog.seg_k[prog.ar, sg]+ramp.ceil().long() == k+1) & own & ~event
    turn = (prog.c['warp'] > 0) & ((k+1) % C.REBASE == 0) & own & ~event & ~arrived
    return dict(event=event, arrived=arrived, turn=turn)


def test_a_renewal_of_the_prediction_gives_the_instruments_teacher_nothing_it_cannot_know(monkeypatch):
    """core._renew works the prediction out again from the plant's own state: when the model is renewed, when a gradual
    change has arrived, and every core.REBASE decisions on a plant whose clock moves. The teacher that sees the
    instruments alone gets nothing of the true values that way, on any of the three occasions: the same plants, valve
    moves and measured disturbances with other random numbers for the unmeasured loads, the output drift and the
    instruments, and the prediction is the same number for number, what every renewal changed in it included. The
    teacher that is told the loads has them in its prediction, and there the two differ."""
    prog = C.build(24, 3, dtype=torch.float64, menu=PR.Menu(rate=(0., 0., 0., 1., 0.), exc=0.))
    ND, steps = prog.nd, 260
    acts = _moves(prog, steps)
    g = torch.Generator().manual_seed(2)
    noise = [(torch.randn(prog.n, prog.nnorm, generator=g, dtype=prog.dtype), torch.rand(prog.n, prog.nuni, generator=g, dtype=prog.dtype)) for _ in range(steps)]
    other = []
    for z, u in noise:                                             # all but the measured disturbances' own draws
        z2 = torch.randn(prog.n, prog.nnorm, generator=g, dtype=prog.dtype)
        z2[:, :ND] = z[:, :ND]
        other.append((z2, torch.rand(prog.n, prog.nuni, generator=g, dtype=prog.dtype)))
    renew, log = C._renew, []

    def spy(prog, st, fk, sg, k, phi, P, who):
        out = renew(prog, st, fk, sg, k, phi, P, who)
        log.append((k, who.clone(), out-st['tF'][who]))
        return out
    monkeypatch.setattr(C, '_renew', spy)

    def run(rnd):
        log.clear()
        st = C.reset(prog)
        for k in range(steps):
            st, _ = C.step(prog, st, acts[k], k, rnd=rnd[k])
        return st, list(log)
    prog.set_teacher('late', False, True)
    a, la = run(noise)
    b, lb = run(other)
    seen = {q: [0, 0.] for q in ('event', 'arrived', 'turn')}       # of every occasion: how many, and the largest change it made
    for k, who, d in la:
        for q, on in _renewed(prog, k).items():
            if bool(on[who].any()):
                seen[q] = [seen[q][0]+int(on[who].sum()), max(seen[q][1], float(d[on[who]].abs().max()))]
    assert all(m >= 4 and d > 1e-3 for m, d in seen.values()), seen
    assert len(la) == len(lb) and all(x[0] == y[0] and torch.equal(x[1], y[1]) and torch.equal(x[2], y[2]) for x, y in zip(la, lb))
    assert torch.equal(a['tF'], b['tF']) and torch.equal(a['d'][:, :ND], b['d'][:, :ND])
    assert not torch.equal(a['d'][:, ND:], b['d'][:, ND:]) and not torch.equal(a['eo'], b['eo'])                  # (the plants did not go the same way)
    prog.set_teacher('late', True)
    a, la = run(noise)
    b, lb = run(other)
    assert not all(torch.equal(x[2], y[2]) for x, y in zip(la, lb)) and not torch.equal(a['tF'], b['tF'])


def test_the_estimates_go_through_a_renewal_of_the_prediction_unbroken(monkeypatch):
    """A renewal leaves the prediction's value now as it is, so what the instruments' teacher takes for the value of a
    row now and for a level's net inflow now (core.seen: that value plus the estimated offset) is, at the decision of a
    renewal, what it is without it, bit for bit; the estimator's own state too. On each of the three occasions (the
    model renewed, a gradual change arrived, the turn of a plant whose clock moves)."""
    prog = C.build(24, 3, dtype=torch.float64, menu=PR.Menu(rate=(0., 0., 0., 1., 0.), exc=0.))
    prog.set_teacher('late', False, True)
    steps = 260
    acts = _moves(prog, steps)
    renew, off = C._renew, [False]
    monkeypatch.setattr(C, '_renew', lambda prog, st, fk, sg, k, phi, P, who: st['tF'][who] if off[0] else renew(prog, st, fk, sg, k, phi, P, who))
    st, g = C.reset(prog), torch.Generator().manual_seed(2)
    seen = {q: [0, 0.] for q in ('event', 'arrived', 'turn')}
    for k in range(steps):
        occ = _renewed(prog, k)
        rnd = (torch.randn(prog.n, prog.nnorm, generator=g, dtype=prog.dtype), torch.rand(prog.n, prog.nuni, generator=g, dtype=prog.dtype))
        off[0] = True
        bare, _ = C.step(prog, st, acts[k], k, rnd=rnd)            # the same decision without the renewal
        off[0] = False
        st, _ = C.step(prog, st, acts[k], k, rnd=rnd)
        x, y = C.seen(prog, st, k+1), C.seen(prog, bare, k+1)
        for q, on in occ.items():
            if bool(on.any()):
                assert all(torch.equal(st[m][on], bare[m][on]) for m in C.METER_KEYS)
                assert torch.equal(x['fl'][on], y['fl'][on]) and torch.equal(x['v'][on], y['v'][on]) and torch.equal(st['tF'][on, 0], bare['tF'][on, 0])
                seen[q] = [seen[q][0]+int(on.sum()), max(seen[q][1], float((st['tF'][on]-bare['tF'][on]).abs().max()))]
    assert all(m >= 4 and d > 1e-3 for m, d in seen.values()), seen     # (while what lies ahead in the prediction was changed)


def test_a_model_that_is_not_the_plants_own_is_not_renewed_from_the_plants_state(monkeypatch):
    """core._renew works the prediction out from the plant's state by the plant's parameters. A commissioning model the
    process has since left (Teacher('fixed') after a process change) is of another plant than that state: its prediction
    is renewed while the process is the commissioning one (a plant whose clock moves, every core.REBASE decisions) and
    never after. A model put in the plant's place (Program.install: a step test's) is never renewed: the renewal would
    hand it the plant's own responses."""
    prog = C.build(24, 3, dtype=torch.float64, menu=PR.Menu(rate=(0., 0., 0., 1., 0.), exc=0.))
    steps = 260
    acts = _moves(prog, steps)
    renew, log = C._renew, []

    def spy(prog, st, fk, sg, k, phi, P, who):
        log.append((k, who.clone()))
        return renew(prog, st, fk, sg, k, phi, P, who)
    monkeypatch.setattr(C, '_renew', spy)

    def run():
        log.clear()
        st = C.reset(prog)
        for k in range(steps):
            st, _ = C.step(prog, st, acts[k], k)
        return list(log)
    prog.set_teacher('fixed')
    got = run()
    changed = lambda k: prog.plant[prog.ar, prog.seg_at(k+1)] > 0                     # noqa: E731  (the process is no longer the commissioning one)
    assert sum(who.numel() for _, who in got) >= 3 and all(not bool(changed(k)[who].any()) and bool((prog.c['warp'][who] > 0).all()) for k, who in got)
    slow = prog.c['warp'] > 0
    assert bool((slow & changed(steps-1)).any()) and bool((slow & ~changed(29)).any())  # (plants it stopped on, plants it was done on)
    prog.install(prog.Fr[:, 0].clone(), prog.phis[:, 0].clone())
    assert run() == []


def test_the_commissioning_model_is_not_renewed_in_a_later_window_once_the_process_has_changed(monkeypatch):
    """A window that carries on holds the plant as commissioned and the last two segments in use; the kinds of those two
    say nothing of the events before them. Where a process event came in a window before, the process is not the
    commissioned one even when the last two events were setpoint moves, and Teacher('fixed') is not renewed from the
    plant's state (it was: on the exam of 16 windows the accurate-model DMC of one plant whose clock moves went from
    8,967 row-decisions outside to 20,601). Where none came, it is renewed as in the first window."""
    renew, log = C._renew, []

    def spy(prog, st, fk, sg, k, phi, P, who):
        log.append(who.clone())
        return renew(prog, st, fk, sg, k, phi, P, who)
    monkeypatch.setattr(C, '_renew', spy)
    moves = PR.Menu(rate=(1., 0., 0., 0., 0.), exc=0.)                    # setpoint moves alone
    for first, changed in ((PR.Menu(rate=(0., 0., 0., 1., 0.), exc=0.), True), (moves, False)):
        prog = C.build(32, 5, dtype=torch.float64, menu=first).next(menu=moves).next(menu=moves)
        kinds = prog.kinds[:, 1:prog.s0+1]
        plain = ~((kinds == PR.PROC) | (kinds == PR.BOTH)).any(1)           # neither of the two segments carried was opened by a process event
        slow = prog.c['warp'] > 0
        assert bool((plain & slow).any()) and bool((prog.was == changed).all()) and bool((prog.tail['changed'] == changed).all())
        assert bool(((prog.plant[:, prog.s0] > 0) == changed).all())
        prog.set_teacher('fixed')
        log.clear()
        st, acts = C.reset(prog), _moves(prog, 2*C.REBASE)
        for k in range(2*C.REBASE):
            st, _ = C.step(prog, st, acts[k], k)
        assert sum(who.numel() for who in log) == (0 if changed else 2*int(slow.sum()))

"""A heavy change that stays (prior.Menu.shift): once in a window a chosen unit becomes another one, and that is
what it is from then on; the exam `shift` has one such change and no other process change after it."""
import torch

from aio.sim import core as C, prior as PR, stay as ST

M, BIG = PR.Menu, PR.BIG


def gains(p):
    NMV = PR.sizes(p)[1]
    on = (p['trk'][:, :, None] > 0) & (p['mvm'] > 0)[:, None, :]
    return p['K'][:, :, :NMV]*on


def far(a, b):
    return (gains(a)-gains(b)).flatten(1).norm(dim=1)/gains(b).flatten(1).norm(dim=1).clamp_min(1e-30)


def stay(menus, n, seed):
    """The windows of a stay drawn by the generator alone: (draw, tail) per window."""
    out, tail = [], None
    for w, m in enumerate(menus):
        keep = {}
        out.append((PR.draw_episode(PR.Rng(seed+w), n, tail=tail, keep=keep, menu=m), keep))
        tail = keep
    return out


def test_without_it_nothing_changes():
    a, b = C.build(24, 3, teacher=False, menu=M()), C.build(24, 3, teacher=False, menu=M(shift=0.))
    assert torch.equal(a.P, b.P) and torch.equal(a.T, b.T) and torch.equal(a.seg_k, b.seg_k)
    assert a.tail['lin']['mend'] is a.tail['lin']['base'][0]      # maintenance puts back the plant as commissioned


def test_one_change_that_stays():
    """The exam's menus: two usual windows, one change at the first ordinary event of the third, no process
    change after it."""
    n = 96
    ws = stay([ST.menu_of(ST.EXAMS['shift']['menu'], w) for w in range(5)], n, 7)
    (segs, tgts, infos, seg_k, kinds, delay, ramp), keep = ws[2]
    proc = (seg_k < BIG) & ((kinds == PR.PROC) | (kinds == PR.BOTH))
    proc[:, :3] = False
    assert bool((proc.sum(1) <= 1).all()) and int(proc.sum()) >= n-2          # one each (a unit held at a constrained point all window long has none)
    assert not bool(((kinds == PR.BOTH) & proc).any())
    rel, at = torch.zeros(n, dtype=PR.F), torch.full((n,), BIG)
    for e in range(3, len(segs)):
        m = proc[:, e]
        if not bool(m.any()):
            continue
        a, b = segs[e-1], segs[e]
        assert bool(PR.point_ok(b, tgts[e])[m].all())
        assert float(((infos[e]['un']-infos[e-1]['un']).abs()*a['mvm'])[m].max()) < 1e-9      # every valve where it stood
        assert torch.equal(tgts[e][m], tgts[e-1][m]) and torch.equal(a['dead'][m], b['dead'][m])
        rel[m], at[m] = far(b, a)[m], seg_k[m, e]
        assert all(torch.equal(keep['mend'][q][m], b[q][m]) for q in PR.LASTING)        # what maintenance puts back from here on
    multi = (segs[0]['mvm'].sum(-1) > 1) & proc.any(1)
    assert float(rel[multi].median()) > .3 and float(torch.quantile(rel[multi], .9)) > .7
    assert float((at[proc.any(1)] <= 1).float().mean()) > .4       # most of them as the window opens
    shifted = keep['cur'][0]
    for (segs, tgts, infos, seg_k, kinds, delay, ramp), keep in ws[3:]:
        used = seg_k[:, 3:] < BIG
        assert bool(used.any()) and not bool((used & ((kinds[:, 3:] == PR.PROC) | (kinds[:, 3:] == PR.BOTH))).any())
        assert all(torch.equal(keep['cur'][0][q], shifted[q]) for q in PR.LASTING+('dead', 'gam', 'fs', 'tlag'))   # the unit it became
        assert all(torch.equal(keep['mend'][q], shifted[q]) for q in PR.LASTING)
        assert torch.equal(keep['base'][0]['K'], ws[0][1]['base'][0]['K'])              # the plant as commissioned is still the first one


def test_maintenance_puts_back_what_the_change_made():
    """Usual windows after a change: the wear and the maintenance go on about the unit it became."""
    ws = stay([None, M(shift=1.), None, None, None], 96, 5)
    first, mend = ws[0][1]['base'][0], ws[1][1]['mend']
    multi = first['mvm'].sum(-1) > 1
    assert float(far(mend, first)[multi].median()) > .3
    for _, keep in ws[2:]:
        assert all(torch.equal(keep['mend'][q], mend[q]) for q in PR.LASTING)           # no further change: the same unit to put back
        cur = keep['cur'][0]
        assert float(far(cur, mend)[multi].median()) < .1 and float(torch.quantile(far(cur, mend)[multi], .9)) < .4
        assert float(far(cur, first)[multi].median()) > .3                               # and not the plant as commissioned


def test_few_of_them():
    n, count, prev, tail = 96, torch.zeros(96), None, None
    for w in range(8):
        keep = {}
        PR.draw_episode(PR.Rng(11+w), n, tail=tail, keep=keep, menu=M(shift=.2, drop=('dead',)))
        prev = keep['base'][0] if prev is None else prev
        count += ((keep['mend']['K'] != prev['K']).flatten(1).any(1) | (keep['mend']['p1r'] != prev['p1r']).flatten(1).any(1)).float()
        prev, tail = keep['mend'], keep
    assert .9 < float(count.mean()) < 1.8 and float((count > 3).float().mean()) < .1    # one or two in eight windows


def test_with_the_other_knobs():
    """Staying at constraints, heavy changes, changes that come and go, all at once: every point passes the one
    check, and a change that stays is never taken back."""
    ws = stay([M(shift=.5, ride=.8, exc=.3, heavy=.5, pulse=.5, drop=('dead',))]*2, 64, 9)
    for (segs, tgts, infos, seg_k, kinds, delay, ramp), keep in ws:
        for e in range(1, len(segs)):
            live = seg_k[:, e] < BIG
            assert bool(PR.point_ok(segs[e], tgts[e])[live].all())
            assert torch.equal(segs[e]['dead'][live], segs[e-1]['dead'][live]) or bool((kinds[live, e] != PR.PROC).any())


def test_the_references_after_the_change():
    """The exam: the teacher goes on as before; the model of commissioning is worth less from the change on."""
    out = ST.references(24, ST.EXAM_SEED, 4, ('hold', 'dmc', 'teacher'), progs=ST.exam('shift', n=24, count=4))
    s, rows = out['score'], out['rows'].sum(1).clamp_min(1)
    assert not bool(out['failed']['teacher'].any())
    assert bool((out['proc'][2] <= 1).all()) and int(out['proc'][3].sum()) == 0
    late, fixed, hold = (s[q].sum(1)/rows for q in ('teacher', 'dmc', 'hold'))
    assert bool((late < .45*hold).all())
    assert float(fixed[2:].sum()) > 1.5*float(late[2:].sum())      # after the change
    assert float(fixed[2:].sum()/late[2:].sum()) > 1.2*float(fixed[:2].sum()/late[:2].sum())       # and by more than before it

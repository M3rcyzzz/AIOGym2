"""The larger sizes are shaped as the controllers of industrial APC are (docs/sim.md, 'sizes'): more zone CVs than
MVs, most MVs resting on economic targets, the unit at its constraints much of the time, and on some plants MVs
that are not the controller's, so that there are more setpoints than MVs to hold them with. The default size
is as it was."""
import torch

from aio.sim import core as C, control as CT, prior as PR, stay as ST

N, SEED = 32, 7


def _batch(**kw):
    return C.build(N, SEED, 'cpu', 'cpu', dims='24x10', **kw)


def test_the_default_size_has_none_of_it():
    prog = C.build(16, 3, 'cpu', 'cpu', mech=0.)
    assert 'thin' not in prog.c and 'mvf' not in prog.c
    assert PR.Menu().ride is None and PR.Menu().exc is None
    p, _, _ = PR.draw_plant(PR.Rng(1), 8)
    assert 'hid' not in p and 'unit' not in p


def test_a_larger_plant_has_the_shape_of_an_apc_controller():
    prog = _batch(menu=PR.Menu(pace=0.), nseg=3)
    c = prog.c
    nm, nz = c['mvf'].sum(1), c['zon'].sum(1)
    rest = c['rst'].sum(1)
    assert float((nz/nm).mean()) > 1.2 and float((nz/nm).max()) <= PR.ZONE_PER[1]+.1     # more zone CVs than MVs
    fat = rest > 0
    assert abs(float(fat.float().mean())-PR.ECON_P) < .2                                  # most plants have MVs resting on a target of their own
    assert float((rest/nm)[fat].mean()) > PR.ECON_SHARE[0]
    thin = c['thin']
    assert 0 < int(thin.sum()) < N and abs(float(thin.float().mean())-PR.THIN_P) < .25
    held = (c['mvf'] > 0) & (c['mvm'] == 0)                                               # MVs whose flow is in the plant and that the controller does not have
    assert bool((held.any(1) == thin).all())
    sp = (c['trk']-c['rst']).sum(1)
    more = sp-(c['mvm'].sum(1)-rest)                                                      # setpoints beyond the MVs there are for them
    assert bool((more[thin] > 0).all()) and bool((more[~thin] == 0).all())
    assert bool((c['link'].sum(1)[held] == 0).all()) and bool((c['psign'][held] == 0).all())      # nothing published pairs a row with one


def test_an_mv_that_is_not_the_controllers_stands_where_it_is():
    prog = _batch()
    c = prog.c
    held = (c['mvf'] > 0) & (c['mvm'] == 0)
    for s in range(prog.seg_k.shape[1]):                          # its limits are its position, in every segment
        t = prog.task(torch.full((N,), s, dtype=torch.long))
        assert bool((t['ulo'][held] == t['uhi'][held]).all())
    for ctl in (CT.PI(), CT.Teacher('late')):
        st = C.reset(prog)
        ctl.reset(prog)
        u0 = st['u'].clone()
        for k in range(120):
            a = ctl(prog, st, k, C.observe(prog, st, k))
            st, _ = C.step(prog, st, a+held*.7, k)                # (a move asked of one is not made)
        assert bool((st['u'][held] == u0[held]).all()) and float((st['u']-u0).abs()[c['mvm'] > 0].max()) > 0
    prog.set_teacher()


def test_a_larger_plant_runs_at_its_constraints_unless_a_menu_says_otherwise():
    usual, plain = _batch(), _batch(menu=PR.Menu(ride=0., exc=PR.EXC_P))
    share = lambda p: float(p.con.float().sum()/(p.seg_k < PR.BIG).float().sum())        # noqa: E731
    assert share(usual) > .25 and share(usual) > share(plain)+.1
    thin = usual.c['thin']
    used = (usual.seg_k < PR.BIG)
    assert float(usual.con[thin].float().sum()/used[thin].float().sum()) > .4            # where setpoints outnumber the MVs, most points are trade-offs


def test_a_plant_alone_is_drawn_fast_and_is_the_same_plant():
    a, b = _batch(), _batch(menu=PR.Menu(pace=0.), nseg=3)
    assert bool((a.P[:, 0] == b.P[:, 0]).all()) and bool((a.T[:, 0] == b.T[:, 0]).all())
    assert bool((C.observe(a, C.reset(a), 0) == C.observe(b, C.reset(b), 0)).all())
    assert int((b.seg_k < PR.BIG).sum(1).max()) == 1


def test_the_teacher_holds_a_plant_with_more_setpoints_than_mvs():
    prog = next(iter(ST.windows(N, SEED, 1, dims='24x10')))
    thin = prog.c['thin']
    got = {q: CT.rollout(prog, ST.REFS[q]()) for q in ('hold', 'pi', 'teacher', 'teacher_meter')}
    assert not any(bool(r['failed'].any()) for r in got.values())
    sc = {q: r['score'][thin].double().mean() for q, r in got.items()}
    assert sc['teacher'] < .5*sc['pi'] and sc['teacher'] < .25*sc['hold']
    assert sc['teacher_meter'] < 1.3*sc['teacher']+20


def test_the_larger_exams_are_the_larger_sizes():
    for name, ncv in (('usual24', 24), ('usual48', 48)):
        ws = list(ST.exam(name, 2, ST.EXAM_SEED, 2))
        assert ws[0].ncv == ncv == ws[1].ncv and ST.key(ws) == ST.key(list(ST.exam(name, 2, ST.EXAM_SEED, 2)))


def test_the_steady_layer_steps_to_the_lowest_cost_along_its_line():
    g = torch.Generator().manual_seed(5)
    m, n, q, nz = 64, 10, 2, 14
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)                    # noqa: E731
    J, e, E, eE, Gz, fz = r(m, n, n), r(m, n), r(m, q, n)*1e-3, r(m, q)*1e-3, r(m, nz, n), r(m, nz)
    wz, slot = 3.16*torch.ones(m, nz, dtype=torch.float64), torch.rand(m, nz, generator=g) < .7
    zl, zh = fz-1.-torch.rand(m, nz, generator=g, dtype=torch.float64), fz+.2
    x, d = torch.zeros(m, n, dtype=torch.float64), r(m, n)
    a = C._line(J, e, E, eE, Gz, fz, wz, zl, zh, slot, x, d)

    def cost(s):                                                                         # (m, k) the steady cost at steps s along d
        y = x[:, None]+s[..., None]*d[:, None]
        res = ((y@J.transpose(1, 2))-e[:, None])**2
        lev = ((y@E.transpose(1, 2))-eE[:, None])**2/PR.EQ_SOFT
        z = fz[:, None]+y@Gz.transpose(1, 2)
        over = ((z-zh[:, None]).clamp_min(0)+(zl[:, None]-z).clamp_min(0))*slot[:, None]
        return res.sum(-1)+lev.sum(-1)+((wz[:, None]*over)**2).sum(-1)
    grid = torch.linspace(0., 1., 20001, dtype=torch.float64)[None].expand(m, -1)
    assert bool((cost(a[:, None])[:, 0] <= cost(grid).amin(-1)+1e-9*cost(grid).amin(-1).abs()+1e-12).all())

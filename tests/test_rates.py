"""One plant at two rates (aio.sim.rates): the valves on the tick, the two packets, the modes, the scores."""
import pytest
import torch

from aio import interface as IF
from aio.sim import core as C, control as CT, mech as ME, prior as PR, rates as RT


@pytest.fixture(scope='module')
def ex():
    return RT.Exam(12, 3)


class Rec:
    """A fast controller that moves nothing and keeps what it was shown."""

    def reset(self, view):
        self.obs, self.fst = [], []

    def __call__(self, view, fst, k, obs):
        self.obs.append(obs); self.fst.append(fst)
        return torch.zeros(view.n, view.nmv, dtype=view.dtype)


def test_the_exam_leaves_the_shell_and_the_program_as_they_were(ex):
    assert ME.PACE == (.5, 1.) and ME.DEAD_P == .5 and ME.VALVE == dict(curve=.3, lag=.5, stick=.4, wear=.25)
    g, s = ex.slow.mg[0], ex.slow.start
    m0, mx0 = g['m'], s['mx']
    a = CT.rollout(ex.slow, CT.PI(), 40)
    RT.rollout(ex, CT.PI(), steps=40)
    assert g['m'] is m0 and s['mx'] is mx0
    b = RT.rollout(ex, CT.PI(), mode='ideal', steps=40)
    assert torch.equal(a['cost'], b['cost']) and 'fscore' not in b


def test_the_shell_of_the_exam_has_no_valve_of_its_own_and_runs_at_the_plants_clock(ex):
    prog = ex.slow
    assert ex.m == 10 and bool((prog.c['dt'] == 10.).all())
    assert bool((prog.c['aa'] == 0).all()) and not bool(prog.mg[0]['din'].any())
    for e in range(PR.NSEG):
        p = prog.params(torch.full((ex.n,), e))
        assert bool((p['gam'] == 1).all()) and bool((p['fs'] == 0).all()) and bool((p['fd'] == 0).all())


def test_the_nominal_curve_inverts_the_installed_valve(ex):
    g = torch.Generator(); g.manual_seed(0)
    x = torch.rand(ex.n, ex.slow.nmv, generator=g)
    q = ex.flow(x)
    assert torch.allclose(ex.opening(q), x, atol=2e-4)
    assert torch.allclose(ex.flow(torch.ones_like(x)), ex.v['S']) and bool((ex.flow(torch.zeros_like(x)) == 0).all())
    h = 1e-3
    xm = x.clamp(.05, .95).double()
    fd = (ex.flow(xm+h)-ex.flow(xm-h))/(2*h)
    assert torch.allclose(ex.slope(xm), fd, rtol=2e-3)
    assert bool((fd > 0).all())
    mvm = ex.v['mvm'] > 0
    assert torch.allclose(ex.flow(ex.x0)[mvm], ex.slow.start['ue'][mvm], atol=1e-5)          # the valves start where the flows of the start point are
    # a slip of a sticking valve moves the flow by half a band at most
    grid = torch.linspace(0., 1., 65)[:, None, None]
    assert bool((ex.v['fs']*ex.slope(grid).amax(0) <= .5*RT.BAND+1e-9).all())


def test_the_fast_packet_is_laid_out_as_the_slow_one(ex):
    rec = Rec()
    RT.rollout(ex, CT.Hold(), rec, steps=3)
    assert len(rec.obs) == 3*ex.m
    obs, fst = rec.obs[-1], rec.fst[-1]
    assert obs.shape == (ex.n, ex.slow.size) == C.observe(ex.slow, C.reset(ex.slow), 0).shape
    cv, mv, links = IF.split(obs)
    nm = ex.v['mvm'].sum(-1)
    trk, dv = cv[:, :, 6], (cv[:, :, 4] == 0) & (cv[:, :, -1] == 1)
    assert torch.equal(trk.sum(-1), nm) and torch.equal(links.sum((1, 2)), nm)
    assert torch.equal(dv.sum(-1), ex.common.to(dv.dtype).long()) and 0 < int(ex.common.sum()) < ex.n          # one header: its pressure is a DV row
    assert torch.allclose(cv[:, :, 9][cv[:, :, -1] == 1], torch.log1p(torch.tensor(RT.FAST))/10)             # the period slot says one tick
    # a flow's row is paired with its valve and reads its meter against the slow MV
    ar = torch.arange(ex.n)[:, None]
    assert torch.equal(links[ar, ex.frow, torch.arange(ex.slow.nmv)[None]], ex.v['mvm'])
    assert set(('y', 'band', 'u', 'sat', 'target')) <= set(fst)
    st = C.reset(ex.slow)
    assert torch.allclose(torch.gather(fst['target'], 1, ex.frow)*ex.v['mvm'], st['u'], atol=1e-6)            # the MVs were not moved: the targets are where they stand
    assert torch.allclose(mv[:, :, 0], (2*fst['u']-1)*ex.v['mvm'])                                             # the valve inside its stroke


def test_a_quiet_header_and_valves_at_rest_leave_the_plant_at_rest(ex):
    keep = {q: ex.v[q] for q in ('sig', 'fs', 'fd', 'noise')}, ex.pv
    for q in keep[0]:
        ex.v[q] = torch.zeros_like(ex.v[q])
    ex.pv = torch.zeros_like(ex.pv)
    try:
        s = ex.slow.start
        rest = ((s['u']-s['un']).abs().amax(-1) < 1e-9).nonzero().flatten().tolist()       # the episodes that start on their steady point
        assert len(rest) >= 2
        r = RT.rollout(ex, CT.Hold(), mode='direct', steps=15, trace=rest)
        mvm = ex.v['mvm'][rest] > 0
        for row in r['ftrace']:
            assert torch.allclose(torch.tensor(row['flow'])[mvm], s['ue'][rest][mvm], atol=1e-5)
        y0 = torch.tensor(r['trace'][0]['y'])
        assert torch.allclose(torch.tensor(r['trace'][-1]['y']), y0, atol=2e-3)
    finally:
        ex.v.update(keep[0]); ex.pv = keep[1]


def test_every_controller_meets_the_same_supply_pressure(ex):
    a = RT.rollout(ex, CT.Hold(), mode='direct', steps=6, trace=[0, 5])
    b = RT.rollout(ex, CT.PI(), CT.PI(), steps=6, trace=[0, 5])
    c = RT.rollout(ex, CT.PI(), CT.PI(), steps=6, trace=[0, 5])
    assert [r['dp'] for r in a['ftrace']] == [r['dp'] for r in b['ftrace']]
    assert [r['flow'] for r in b['ftrace']] == [r['flow'] for r in c['ftrace']]            # and a rollout is repeatable
    dp = torch.tensor([r['dp'] for r in a['ftrace']])
    assert float(dp.abs().max()) > 0 and float(dp.abs().max()) <= RT.P_MAX


def test_flow_loops_hold_the_flows_and_one_slow_period_cannot(ex):
    steps = 300
    hold = RT.rollout(ex, CT.Hold(), mode='direct', steps=steps)
    direct = RT.rollout(ex, CT.Teacher('fixed'), mode='direct', steps=steps)
    tiers = RT.rollout(ex, CT.Teacher('fixed'), CT.PI(), steps=steps)
    exact = RT.rollout(ex, CT.Teacher('now'), RT.Flow(), steps=steps)
    ideal = RT.rollout(ex, CT.Teacher('fixed'), mode='ideal', steps=steps)
    share = lambda r: float(r['fscore'].sum()/r['frows'].sum())                             # noqa: E731
    assert float(hold['frows'].sum()) > 0 and not bool(tiers['failed'].any())
    assert share(tiers) < .5*share(direct) and share(exact) < .5*share(direct)
    assert float(tiers['frms'].mean()) < float(direct['frms'].mean())
    s = lambda r: float(r['score'].sum())                                                   # noqa: E731
    assert s(tiers) < s(direct) < s(hold)
    assert s(tiers)-s(ideal) < .25*(s(direct)-s(ideal))                                     # over flow loops the slow tier does nearly as with flows as asked


def test_narrowed_limits_show_in_the_packet_and_in_the_move(ex):
    prog = ex.slow
    st = C.reset(prog)
    mvm, sl = prog.c['mvm'] > 0, prog.c['slew']
    t = prog.task(prog.seg_at(0))
    u = st['u']
    nar = RT.narrow(prog, uhi=u+.2*sl)
    tn = nar.task(prog.seg_at(0))
    assert bool((tn['uhi'] <= t['uhi']).all()) and torch.equal(tn['ulo'], t['ulo'])
    assert bool(((tn['uhi']-tn['ulo'])[mvm] >= 2*PR.PIN-1e-6).all())
    up = torch.ones_like(u)
    assert bool(((u+nar.move(st, up, 0)*sl)[mvm] <= torch.maximum(tn['uhi'], u)[mvm]+1e-6).all())
    assert bool((nar.move(st, up, 0)[mvm] < 1).any())
    for a in (up, -up, .3*up):                                                              # without narrower limits: the move the core makes
        assert torch.allclose(u+RT.narrow(prog).move(st, a, 0)*sl, C.step(prog, st, a, 0, through=False)[0]['u'], atol=1e-6)
    out = nar.move(dict(st, u=tn['uhi']+3*sl), 0*up, 0)                                     # from beyond a limit: back at the slew limit
    assert torch.allclose(out[mvm], -torch.ones_like(out)[mvm])
    cv, mv, _ = IF.split(C.observe(nar, st, 0))
    assert torch.allclose(mv[:, :, 0][mvm], (2*(u-tn['ulo'])/(tn['uhi']-tn['ulo'])-1)[mvm], atol=1e-5)
    cv0, mv0, _ = IF.split(C.observe(prog, st, 0))
    assert torch.equal(cv, cv0) and not torch.equal(mv, mv0)


def test_the_references_give_the_two_scales(ex):
    out = RT.references(ex, names=('dmc_direct', 'dmc_pi'))
    assert set(out) == {'hold', 'pi', 'dmc_direct', 'dmc_pi'}
    assert ex.scale(out['hold']['score'].tolist()) == 0 and ex.fscale(out['hold']['fscore'].tolist()) == 0
    sp, fp = ex.points(out['dmc_pi'])
    assert 50 < sp <= 100 and 50 < fp <= 100
    assert ex.points(out['dmc_direct'])[0] < sp
    rows = RT.table(ex, out)
    assert [r['name'] for r in rows] == list(out) and rows[0]['mode'] == 'direct'


def test_a_sag_takes_valves_to_the_end_of_their_stroke_and_the_limit_rule_tells():
    """Exam(sag=1): the same exam with a stretch in which a valve wide open passes less than its flow at the start.
    Valves reach the end of their stroke; a slow tier under the limit rule of aio.tiers loses fewer slow rows to
    it than one that reads its limits as given."""
    e0, ex = RT.Exam(12, 3), RT.Exam(12, 3, sag=1.)
    assert e0.sag == 0 and not bool(e0.sd.any())
    for q in e0.v:
        assert torch.equal(e0.v[q], ex.v[q])                                                # the same valves, and the same supply but for the sags
    assert torch.equal(e0.pk, ex.pk) and torch.equal(e0.pv, ex.pv) and torch.equal(e0.x0, ex.x0)
    assert set(RT.basis(ex))-set(RT.basis(e0)) == set(RT.RULED)
    mvm, hit, one = ex.v['mvm'] > 0, ex.sd > 0, ex.common
    assert bool(hit.any(-1).all()) and bool((ex.sd < 1).all()) and not bool((hit & ~mvm).any())
    assert bool((hit.sum(-1)[~one] == 1).all()) and bool((hit == mvm)[one].all())           # the supply of one valve, or the header of all
    assert bool((ex.sk[:, 0] < ex.sk[:, 1]).all()) and 0 < int(ex.sk.min()) and int(ex.sk.max()) < PR.HORIZON*ex.m
    q0, S = ex.slow.start['ue'], ex.v['S']
    wide = ex.flow(torch.ones_like(q0), -ex.sd)                                             # wide open under the sag
    r = (wide/q0)[hit & ~one[:, None]]
    assert bool((r > RT.SAG[0]-1e-4).all()) and bool((r < RT.SAG[1]+1e-4).all())
    m = mvm.to(q0.dtype)
    r = ((wide/S*m).sum(-1)/((q0/S*m).sum(-1)))[one]                                        # one header: by the mean of its valves
    assert bool((r > RT.SAG[0]-1e-4).all()) and bool((r < RT.SAG[1]+1e-4).all())
    a = RT.rollout(e0, CT.Teacher('fixed'), CT.PI())
    b = RT.rollout(ex, CT.Teacher('fixed'), CT.PI())
    c = RT.rollout(ex, *RT.ruled(CT.Teacher('fixed'), CT.PI()))
    assert not bool(b['failed'].any()) and not bool(c['failed'].any())
    assert float(a['fsat'].mean()) < .01 and float(b['fsat'].mean()) > .03                  # valves at the end of their stroke
    s = lambda r: float(r['score'].sum())                                                   # noqa: E731
    assert s(a) < s(c) < s(b), (s(a), s(c), s(b))                                           # the sag costs slow rows; the rule gives part of them back
    assert float(c['fsat'].mean()) < float(b['fsat'].mean())                                # and the valves are at the end of their stroke for less of the time
    assert float(b['fscore'].sum()/b['frows'].sum()) < .25                                  # a flow short of its target at the end of the stroke is not counted

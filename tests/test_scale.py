"""aio.scale and the Oracle's batching: 0 is the MVs left where they are and 100 the best causal controller per episode, a file
of another batch is refused, and the candidates of a decision run side by side cost what they cost one by one."""
import json

import torch

from aio import scale as SC
from aio.sim import core as C, oracle as OR

PER = dict(hold=[90., 80., 70.], pi=[50., 60., 40.], dmc=[30., 70., 20.], oracle=[10., 40., 30.], bound=[1., 1., 1.])


def test_the_envelope_is_the_best_causal_controller_per_episode():
    assert SC.envelope(PER) == [10., 40., 20.]                                           # never hold, never one that knows the noise to come
    sc = SC.Scale.of(PER)
    assert sc(PER['hold']) == 0. and sc(sc.best) == 100.
    assert abs(sc(PER['pi'])-100*(240-150)/(240-70)) < 1e-9 and abs(sc(PER['dmc'])-100*(240-120)/(240-70)) < 1e-9
    assert sc([95., 90., 80.]) < 0 and sc([5., 30., 10.]) > 100                          # worse than doing nothing; better than everything run
    assert sc.meta()['oracle'] and sc.meta()['zero'] == 'hold' and set(sc.basis) == {'pi', 'dmc', 'oracle'}
    assert not SC.Scale.of({q: v for q, v in PER.items() if q != 'oracle'}).meta()['oracle']


def test_a_number_of_the_old_scale_is_read_on_this_one():
    """Until 11-11 the 0 was the formula PI: (pi - score)/(pi - best). The same score, in points."""
    sc = SC.Scale.of(PER)
    for score in (PER['dmc'], PER['hold'], PER['pi'], [5., 30., 10.]):
        old = (150-sum(score))/(150-70)
        assert abs(sc.from_recovered(old)-sc(score)) < 1e-9


def test_further_controllers_lower_the_envelope():
    sc = SC.Scale.of(PER).join(dict(now=[20., 30., 25.], pi=PER['pi'], hold=PER['hold']))
    assert sc.best == [10., 30., 20.] and 'now' in sc.basis and 'hold' not in sc.basis and sc.hold == PER['hold']


def test_a_file_of_another_batch_is_refused(tmp_path):
    f = tmp_path/'o.json'
    f.write_text(json.dumps(dict(per_episode=dict(PER, best=[0., 0., 0.]))))
    sc = SC.Scale.load(f, [50., 61., 40.])                                               # the same episodes up to rounding through the loop
    assert sc is not None and sc.best == [10., 40., 20.]                                 # the file's own 'best' is not taken on trust
    assert SC.Scale.load(f, [50., 90., 40.]) is None and SC.Scale.load(f, [50., 60.]) is None
    assert SC.same_batch([100.]*10, [101.]*10) and not SC.same_batch([100.]*10, [110.]*10)


def test_candidates_side_by_side_cost_what_they_cost_one_by_one():
    torch.manual_seed(0)
    prog = C.build(5, 31, mech=.4)
    st = C.reset(prog)
    H, R, k = 8, 3, 0
    s0 = {q: v for q, v in st.items() if q != 'tF'}
    plans = [(s0['u'][None]+.2*torch.randn(H, 5, s0['u'].shape[1], dtype=prog.dtype)).clamp(0., 1.) for _ in range(R)]
    w = torch.ones(H, dtype=prog.dtype)
    one = torch.stack([OR.plan_cost(OR.Frozen(prog, k), s0, k, U, w, .25, .1, OR.calm(prog)) for U in plans])
    wide = OR.tile(prog, R)
    assert wide.n == 15 and wide.Fr is None and len(wide.mg) == R*len(prog.mg)
    side = OR.plan_cost(OR.Frozen(wide, k), OR.tile_state(s0, 5, R), k, torch.cat(plans, 1), w, .25, .1, OR.calm(wide)).view(R, 5)
    assert torch.allclose(side, one, rtol=1e-4, atol=1e-5)
    assert float(one.std(0).max()) > 0                                                   # the plans do differ


def test_the_objective_is_the_score_made_smooth():
    """Every row inside: near 0. Rows outside: their share of the rows (softened) plus the Huber share; more the further out."""
    prog = C.build(6, 7)
    st = C.reset(prog)
    z = torch.zeros(6, st['u'].shape[1], dtype=prog.dtype)
    st1, out = C.step(prog, st, z, 0)
    rows = (prog.c['trk']+prog.c['zon']).sum(-1).clamp_min(1.)
    j = OR.objective(out, st1, prog.c)
    assert j.shape == (6,) and bool((j >= 0).all())
    count = out['outside'].sum(-1)/rows
    assert bool((j[count == 0] < .1).all()) and bool((j[count > 0] > .5*count[count > 0]).all())   # the count, softened; little else inside
    far = dict(out, err=out['err']+50*prog.c['trk'])
    trk = prog.c['trk'].sum(-1) > 0
    assert bool((OR.objective(far, st1, prog.c)[trk] > j[trk]).all())
    count, log = OR.objective(far, st1, prog.c, cost_w=0.), OR.objective(far, st1, prog.c, cost_w=0., log_w=.2)
    assert bool((log[trk] > count[trk]).all()) and bool((log-count < .2*4.5).all())   # the log far term: a direction that stays small 50 bands out

"""A conserving train of vessels in the pair bank (prior.STOCK; docs/sim.md, 'stock')."""
import torch

from aio.sim import core as C, prior as PR

NMV = PR.NMV


def test_without_it_no_plant_has_one():
    p, _, _ = PR.draw_start(PR.Rng(3), 64)
    assert not bool(p['stock'].any())


def test_a_stream_takes_from_one_vessel_what_it_adds_to_the_next_through_every_event():
    segs, tgts, infos, seg_k, kinds, delay, ramp = PR.draw_episode(PR.Rng(5), 256, shape=PR.STOCK)
    p0 = segs[0]
    lev = p0['stock'] > 0
    n = lev.sum(1)
    assert bool(((n >= 1) & (n <= PR.STOCK_MAX)).all()) and bool(((p0['integ'] > 0) == lev).all())
    assert bool((lev <= (p0['trk'] > 0)).all()) and not bool((lev & (p0['rst'] > 0)).any())
    assert bool(((p0['K'][:, :, :NMV]*p0['link']).sum(-1)[lev] != 0).all())           # every vessel has its handle
    ratio0 = None
    for p, t in zip(segs, tgts):
        assert bool(PR.point_ok(p, t).all())
        Kl = p['K'][:, :, :NMV]*lev[:, :, None]
        cnt, pos = (Kl != 0).sum(1), (Kl > 0).sum(1)
        assert int(cnt.max()) <= 2 and bool((pos[cnt == 2] == 1).all())               # a valve moves two levels at the most, and those opposite ways
        two = (cnt == 2)[:, None, :] & (Kl != 0)
        for q in ('p1r', 'dead'):                                                     # one flow: the same lag and delay at both ends
            a = p[q][:, :, :NMV]
            hi, lo = torch.where(two, a, torch.full_like(a, -1e9)).amax(1), torch.where(two, a, torch.full_like(a, 1e9)).amin(1)
            assert bool(((hi == lo) | (cnt != 2)).all())
        ratio = -Kl.clamp_min(0).sum(1)/Kl.clamp_max(0).sum(1).clamp_max(-1e-12)      # what the stream adds over what it takes: the holdups' ratio
        ratio = torch.where(cnt == 2, ratio, torch.zeros_like(ratio))
        ratio0 = ratio if ratio0 is None else ratio0
        assert torch.allclose(ratio, ratio0, rtol=1e-9, atol=0)                       # a process event changes a stream with its valve, never one end of it
    assert .4 < float((p0['bandf'][lev] >= PR.STOCK_BAND[0]).double().mean()) < .8    # most vessels are buffers
    assert bool(((kinds == PR.PROC) | (kinds == PR.BOTH)).any())


def test_a_level_adds_up_the_flows_at_its_vessel():
    prog = C.build(48, 9, teacher=False, quiet=True, shape=PR.STOCK, menu=PR.Menu(rate=(0., 0., 0., 0., 0.)))
    c = prog.c
    st = C.reset(prog)
    lev = c['integ'] > 0
    v0 = st['v'].clone()
    j = 0                                                                              # every plant's first MV full speed up for a while
    a = torch.zeros(prog.n, prog.nmv, dtype=prog.dtype)
    a[:, j] = 1.
    for k in range(40):
        st, _ = C.step(prog, st, a*(k < 6), k)
    dv = (st['v']-v0)*c['tint']*lev                                                    # what each level took in, in the row's own units
    moved = dv.abs() > 1e-6
    assert bool((moved.sum(1) <= 2).all()) and bool(moved.any())                      # the valve's own vessel and the one at the stream's other end, nothing else
    two = moved.sum(1) == 2
    assert bool(two.any()) and bool(((dv > 1e-6).sum(1)[two] == 1).all())             # one loses what the other gains

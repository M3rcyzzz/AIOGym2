"""aio.sim.hand: the hand-written controllers on the packet's information (the comparison line)."""
import torch

from aio.sim import control as CT, core as C, hand as H


def test_hand_written_controllers_move_only_the_mvs_in_use_and_repeat():
    prog = C.build(24, 6)
    hold = CT.rollout(prog, CT.Hold())['score']
    for name, make in H.HAND.items():
        ctl = make()
        st = C.reset(prog)
        ctl.reset(prog)
        for k in range(30):
            a = ctl(prog, st, k, C.observe(prog, st, k))
            assert a.shape == (prog.n, prog.nmv) and bool((a.abs() <= 1).all()), name
            assert bool((a[prog.c['mvm'] == 0] == 0).all()), name
            st, _ = C.step(prog, st, a, k)
        r, again = CT.rollout(prog, make()), CT.rollout(prog, make())
        assert torch.equal(r['score'], again['score']), name
        assert not bool(r['failed'].any()), name
        assert float(r['score'].mean()) < float(hold.mean()), name      # better than leaving the MVs alone, on average


def test_the_adaptive_gain_goes_down_on_a_swing_and_up_on_a_standing_error():
    prog = C.build(4, 6)
    ctl = H.AdaptivePI(hold=3)
    ctl.reset(prog)
    st = C.reset(prog)
    row, band = prog.pi['row'], prog.c['bandf']

    def at(e):                                                    # the controller with every paired row e bands off its target
        s = dict(st)
        s['y'] = st['target']-e*band
        return ctl(prog, s, 0, None)
    for _ in range(6):
        at(2.)
    assert bool((ctl.th > 0).all())                               # stood two bands off on one side without closing
    up = ctl.th.clone()
    at(-2.)
    assert bool((ctl.th < up).all())                              # swung to the other side
    assert row.shape == ctl.th.shape

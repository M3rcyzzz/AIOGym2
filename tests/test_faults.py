"""The exam-only faults and start-up (aio.sim.faults) and the shutdown limits (`trip`)."""
import torch

from aio.sim import control as CT, core as C, faults as F, mech as ME, prior as PR


class Push:
    def __init__(self, sign):
        self.sign = sign

    def reset(self, prog):
        pass

    def __call__(self, prog, st, k, obs):
        return self.sign*torch.ones(prog.n, prog.nmv, dtype=prog.dtype, device=prog.device)*prog.c['mvm']


def test_a_fault_changes_only_the_faulted_episodes_and_is_taken_out_after():
    base = F._build('usual', 8, 7)[0]
    pi0 = CT.rollout(base, CT.PI())['score']
    step = C.step
    for kind in ('bad', 'freeze', 'drift', 'stick'):
        prog = F._build('usual', 8, 7)[0]
        f = F.draw(prog, kind, 7)
        assert (f['idx'] >= 0).all() and (f['k0'] >= prog.k_first.clamp_max(PR.HORIZON)).all()
        assert (f['k0']+f['dur'] <= PR.HORIZON).all()
        with F.applied(prog, f):
            hold = CT.rollout(prog, CT.Hold())
            pi = CT.rollout(prog, CT.PI())['score']
        assert C.step is step and 'params' not in prog.__dict__
        if kind != 'stick':                                       # a reading's fault does not move the plant
            assert torch.equal(hold['score'], CT.rollout(base, CT.Hold())['score'])
        assert not torch.equal(pi, pi0), kind
        assert torch.equal(CT.rollout(prog, CT.PI())['score'], pi0)


def test_a_start_up_runs_from_the_standby_state():
    out = F.exam('usual', 'startup', dict(hold=CT.Hold(), pi=CT.PI(), teacher=CT.Teacher('late')), 4, 7)
    assert out['from_standby'] == 4 and out['points']['hold'] == 0. and out['points']['teacher'] == 100.
    assert all(v['episodes'] == 0 for v in out['trips'].values())


def test_a_plant_with_no_shutdown_limits_never_trips_and_tep_does_when_pushed():
    r = CT.rollout(C.build(4, 7), Push(1.))
    assert not r['tripped'].any() and (r['trip_k'] == CT.BIG).all()
    plants = ME.pool(['tep'])
    prog = C.build(2, 3_000_000, mech=1., plants=plants, dims='12x5')
    assert not CT.rollout(prog, CT.Hold())['tripped'].any()
    up = CT.rollout(prog, Push(1.))
    assert up['tripped'].all() and (up['trip_k'] < 200).all() and not up['failed'].any()


def test_training_faults_in_the_core_are_the_exams_faults():
    """aio.train --faults: the core's own faults (prog.fault, train_draw) do what the exam's wrapped ones do, episode for
    episode: the same PI rollout as `applied` with each episode's fault; no fault: the plant as it was."""
    prog = F._build('usual', 8, 7)[0]
    f = F.train_draw(prog, 1., 7)
    assert f is not None and bool(f['has'].all())
    ref = CT.rollout(prog, CT.PI())['score']
    prog.fault = f
    got = CT.rollout(prog, CT.PI())['score']
    del prog.fault
    assert not torch.equal(got, ref)
    for kind, name in enumerate(F.TRAIN_KINDS):
        on = (f['kind'] == kind)
        if not bool(on.any()):
            continue
        idx = torch.where(on, (f['mv'] if name == 'stick' else f['row']).float().argmax(1), torch.full_like(f['k0'], -1))
        ex = dict(kind=name, idx=idx, k0=f['k0'], dur=f['dur'], off=f['off'])
        with F.applied(prog, ex):
            want = CT.rollout(prog, CT.PI())['score']
        assert torch.allclose(got[on], want[on]), name
    assert F.train_draw(prog, 0., 7) is None

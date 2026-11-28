"""The standard benchmark (aio.sim.bench, docs/standard.md): packs, the scorecard, conformance, the Gymnasium shell."""
import numpy as np
import pytest
import torch

from aio.sim import bench as B, control as CT, core as C, prior as PR, stay as ST


def test_the_environment_conforms():
    res = B.conform(steps=60)
    assert all(ok for _, ok, _ in res), [r for r in res if not r[1]]


def test_a_pack_is_the_batch_its_seed_draws_and_runs_as_it_does(tmp_path):
    progs = list(ST.exam('usual', 8, 5, 2))
    key = B.freeze(progs, tmp_path/'p.npz')
    back, head = B.thaw(tmp_path/'p.npz')
    assert head['key'] == key == ST.key(progs)
    a, b = B.stay(progs, CT.PI()), B.stay(back, CT.PI())           # (the formula PI's tuning is in the pack)
    assert a['score'] == b['score'] and a['failed'] == b['failed']


def test_the_card_of_a_reference_is_its_row_of_the_table():
    progs, refs = B.pack('stock')
    cd = B.card('stock', run={k: refs[k]['teacher_meter'] for k in ('score', 'failed', 'tripped', 'cost')})
    t = cd['table']['all']
    assert t['controller'] == t['teacher_meter'] and cd['key'] == B.SUITE['stock']['key']
    assert t['dmc_meter']['points'] < t['teacher_meter']['points'] <= 100.5 and t['pi']['points'] > 0
    assert set(t['controller']) == {'points', 'cost_points', 'cost_worse', 'worse', 'low', 'worst', 'outside', 'failed', 'tripped'}


def test_holding_scores_zero_on_the_first_window():
    progs, refs = B.pack('usual')
    run = B.stay(progs[:1], CT.Hold())
    assert run['score'][0] == refs['score']['hold'][0]             # hold's own stay and hold after the teacher start the first window alike


def test_the_gymnasium_shell():
    pytest.importorskip('gymnasium')
    from aio.sim.gym import VecEnv
    env = VecEnv(4, seed=3)
    obs, info = env.reset()
    assert obs.shape == (4, env.single_observation_space.shape[0]) and info['mask'].shape == (4, env.nmv)
    for k in range(PR.HORIZON):
        obs, r, term, trunc, info = env.step(np.zeros((4, env.nmv), np.float32))
    assert trunc.all() and (r <= 0).all() and env.seed == 4
    env = VecEnv(4, seed=5, tensors=True, grad=True, reward='soft')
    env.reset()
    w = torch.zeros(env.nmv, requires_grad=True)
    total = 0.
    for k in range(90):                                            # (the cost counts from the first event: decisions 20-80)
        total = total-env.step(torch.tanh(w).expand(4, -1))[1].sum()
    total.backward()
    assert torch.isfinite(w.grad).all() and w.grad.abs().sum() > 0


def test_a_batch_whose_references_came_without_the_cost_reads_by_the_count_alone():
    t = B.card('usual24')['table']['all']
    assert t['teacher_meter']['cost_points'] is None and 90 < t['dmc_meter']['points'] < t['teacher_meter']['points'] < 100


def test_the_default_card_is_of_the_batches_whose_packs_are_here():
    assert set(B.HERE) == {'usual', 'stock', 'standard', 'usual24'} and all((B.DATA/B.VERSION/f'{q}.refs.json').exists() for q in B.SUITE)

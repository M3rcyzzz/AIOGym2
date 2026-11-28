"""Performance, VRAM, BF16 AMP, compile, simulator batching, curriculum, and safe-load tests (Track A)."""
import io
import os
import pathlib
import pickle
import sys

import pytest
import torch

from aio import net as N, seqnet as SQ
from aio.interface import CF, MF, NCV, NMV, SIZE
from aio.sim import core as C, prior as PR


def _seq_inputs(b, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    o = torch.randn(b, T, SIZE, generator=g)
    o[..., :NCV*CF].reshape(b, T, NCV, CF)[..., -1] = 1
    o[..., NCV*CF:NCV*CF+NMV*MF].reshape(b, T, NMV, MF)[..., -1] = 1
    o[..., -NCV*NMV:] = 0
    return o, torch.randn(b, T, NMV, generator=g)


def _lively_seq(asked='d32,l2,h4,m8,g2,q1,u1,f1,r2', w=32, seed=0):
    torch.manual_seed(seed)
    net = SQ.SeqAIO(asked, w=w)
    with torch.no_grad():
        for p in net.parameters():
            if p.dim() >= 2 and float(p.abs().max()) == 0:
                p.normal_(0, .05)
    return net


def _old_cut(net, stop):
    """Reference implementation of the old O(H^2) Room.late copy-based `cut()`."""
    r = net.last_room
    if r is None or not r.late_pos:
        return
    N_tok = r.K[0].shape[0]//stop.shape[0]
    sb = stop.repeat_interleave(N_tok)[:, None, None]
    for late in r.late:
        for j, (k, v) in enumerate(late):
            late[j] = (torch.where(sb, k.detach(), k), torch.where(sb, v.detach(), v))


@pytest.mark.parametrize('grad_ckpt', (False, True))
def test_cut_mask_exact_forward_and_backward_parity_zero_late_copies(grad_ckpt):
    """New Room.cut_pos causal mask inside `_step_core_late` matches old copy-based `cut()` on both forward outputs
    and input+parameter gradients across multiple staggered cuts (including g2 lags crossing cuts), while allocating
    ZERO copies of `r.late` tensors and keeping `v` contiguous (never pinning the 3x qkv_t buffer)."""
    net_ref = _lively_seq(seed=3).train()
    net_new = _lively_seq(seed=3).train()
    net_new.load_state_dict(net_ref.state_dict())
    net_ref.actor.grad_ckpt = grad_ckpt
    net_new.actor.grad_ckpt = grad_ckpt

    b, T = 3, 14
    o_base, pv = _seq_inputs(b, T, seed=19)
    cuts = {
        4: torch.tensor([True, False, False]),
        8: torch.tensor([True, True, False]),
        11: torch.tensor([False, True, False]),
    }

    def run(net, use_old):
        net.forget()
        net.zero_grad()
        with torch.no_grad():
            h = None
            for t in range(4):
                _, _, _, _, h = net.steps(o_base[:, t:t+1], pv[:, t:t+1], h)
        xs = [o_base[:, t:t+1].clone().requires_grad_(True) for t in range(4, T)]
        g = h.detach()
        ys = []
        ptrs_before = None
        for idx, t in enumerate(range(4, T)):
            if t in cuts:
                stop = cuts[t]
                if not use_old and t == 8:
                    ptrs_before = [(k.data_ptr(), v.data_ptr()) for k, v in net.last_room.late[0]]
                if use_old:
                    _old_cut(net, stop)
                else:
                    net.cut(stop)
                if not use_old and t == 8:
                    ptrs_after = [(k.data_ptr(), v.data_ptr()) for k, v in net.last_room.late[0]]
                    assert ptrs_after == ptrs_before, 'new cut() must NOT copy/replace Room.late tensors'
                g = torch.where(stop[None, :, None], g.detach(), g)
            mean, _, _, _, g = net.steps(xs[idx], pv[:, t:t+1], g)
            ys.append(mean)
        # Verify v in r.late is contiguous and 1x storage (does NOT pin 3x qkv_t buffer)
        if not use_old:
            for layer_late in net.last_room.late:
                for k_t, v_t in layer_late:
                    assert v_t.is_contiguous()
                    assert v_t.untyped_storage().nbytes() == v_t.numel()*v_t.element_size()
        out = torch.cat(ys, 1)
        # Weight only the final step heavily so gradients test multi-cut cutoff across all earlier steps & lags
        loss = (out[:, -1].square().sum() + 0.1*out.square().sum())
        loss.backward()
        return out.detach(), [x.grad.detach().clone() for x in xs], {q: (None if p.grad is None else p.grad.detach().clone()) for q, p in net.named_parameters()}

    y_ref, gx_ref, gp_ref = run(net_ref, use_old=True)
    y_new, gx_new, gp_new = run(net_new, use_old=False)

    assert torch.allclose(y_new, y_ref, atol=1e-6)
    for t_i, (gr, gn) in enumerate(zip(gx_ref, gx_new)):
        assert torch.allclose(gn, gr, atol=1e-5, rtol=1e-4), f'input grad mismatch at step {t_i}'
    for q in gp_ref:
        if gp_ref[q] is None:
            assert gp_new[q] is None, q
        else:
            assert gp_new[q] is not None and torch.allclose(gp_new[q], gp_ref[q], atol=1e-5, rtol=1e-4), f'param grad mismatch: {q}'

    # Strict zero check across cut boundary: loss ONLY on final step t=13 -> env 0 (cut at 8) and env 1 (cut at 11)
    # must have exact 0 gradient on input at t=5 (before both cuts), while env 2 (never cut) has > 0 gradient!
    net_new.forget()
    with torch.no_grad():
        h = None
        for t in range(4):
            _, _, _, _, h = net_new.steps(o_base[:, t:t+1], pv[:, t:t+1], h)
    x5 = o_base[:, 5:6].clone().requires_grad_(True)
    g = h.detach()
    for t in range(4, T):
        if t in cuts:
            net_new.cut(cuts[t])
            g = torch.where(cuts[t][None, :, None], g.detach(), g)
        inp = x5 if t == 5 else o_base[:, t:t+1]
        mean, _, _, _, g = net_new.steps(inp, pv[:, t:t+1], g)
    gx5, = torch.autograd.grad(mean.square().sum(), x5)
    assert float(gx5[0].abs().sum()) == 0.0
    assert float(gx5[1].abs().sum()) == 0.0
    assert float(gx5[2].abs().sum()) > 0.0


def test_need_read_false_settle_rooms_and_mem_rel_no_hidden_pin():
    """need_read=False skips read(pair) completely (0 grad on read head, exact action parity),
    settle_rooms(inplace=True) frees late graphs without cloning K/V, and mem0/rel0 clones do not pin full hidden."""
    net = _lively_seq(seed=5).train()
    net.actor.grad_ckpt = True
    b, T = 2, 8
    o, pv = _seq_inputs(b, T, seed=21)

    # 1. need_read=False matches need_read=True on mean/when/mask/hidden and skips read(pair) completely
    net.forget(); h1 = None; m1s = []
    for t in range(T):
        m, w, mk, lab, h1 = net.steps(o[:, t:t+1], pv[:, t:t+1], h1, need_read=True)
        assert lab is not None and lab.shape == (b, 1, NCV, NMV, net.actor.nk)
        m1s.append(m)
    net.zero_grad(); net.forget(); h2 = None; m2s = []
    saved_ptrs = set()
    hidden_ptrs = set()
    with torch.autograd.graph.saved_tensors_hooks(lambda t: (saved_ptrs.add(t.untyped_storage().data_ptr()), t)[1], lambda t: t):
        for t in range(T):
            m, w, mk, lab, h2 = net.steps(o[:, t:t+1], pv[:, t:t+1], h2, need_read=False)
            assert lab is None
            hidden_ptrs.add(h2.untyped_storage().data_ptr())
            m2s.append(m)
        loss = torch.cat(m2s, 1).square().sum()
        loss.backward()
    assert torch.allclose(torch.cat(m1s, 1), torch.cat(m2s, 1), atol=1e-6)
    assert torch.allclose(h1.detach(), h2.detach(), atol=1e-6)
    assert all(p.grad is None for p in net.actor.read.parameters())
    assert hidden_ptrs and not (hidden_ptrs & saved_ptrs), 'saved tensors must not pin full hidden storage via mem0/rel0 views'

    # Also verify T > 1 path with need_read=False
    net.eval(); net.forget()
    with torch.no_grad():
        mt, _, _, lt, ht = net.steps(o, pv, need_read=True)
        mf, _, _, lf, hf = net.steps(o, pv, need_read=False)
        assert lf is None and torch.allclose(mt, mf, atol=1e-6) and torch.allclose(ht, hf, atol=1e-6)

    # 2. settle_rooms(inplace=True) after backward avoids cloning K/V and produces exact subsequent predictions
    r = net.last_room
    assert len(r.late_pos) == 0  # eval T>1 settled it; re-run a graph window
    net.train(); net.forget(); h = None
    for t in range(5):
        m, _, _, _, h = net.steps(o[:, t:t+1], pv[:, t:t+1], h, need_read=False)
    r = net.last_room
    assert len(r.late_pos) == 5 and r.read is True
    k_ptrs = [K.data_ptr() for K in r.K]
    v_ptrs = [V.data_ptr() for V in r.V]
    m.square().sum().backward()
    net.settle_rooms(inplace=True)
    assert len(r.late_pos) == 0 and all(len(l) == 0 for l in r.late) and r.read is False and r.cut_pos is None
    assert [K.data_ptr() for K in r.K] == k_ptrs and [V.data_ptr() for V in r.V] == v_ptrs
    with torch.no_grad():
        m_next = net.steps(o[:, 5:6], pv[:, 5:6], h.detach(), need_read=False)[0]
    assert torch.allclose(m_next, mt[:, 5:6], atol=1e-5)


def test_room_fast_paths_skip_raw_compare_and_respect_inplace_reset():
    """SeqAIO.room() hits zero-sync fast paths on unmodified hidden and on lane_tag across detach()/cut_rows(),
    while still catching in-place mutations (h[0, 0] = 0) and hidden=None resets."""
    net = _lively_seq(seed=9).eval()
    assert getattr(net, 'supports_fast_steps', False) is True
    b, T = 2, 8
    o, pv = _seq_inputs(b, T, seed=29)
    with torch.no_grad():
        whole = net.steps(o, pv, need_read=False)[0]

    # 1. Automatic fast path (lane_tag=None): poison r.raw after step 0; unmodified hidden still hits without reading r.raw!
    net.forget()
    with torch.no_grad():
        m0, _, _, _, h = net.steps(o[:, 0:1], pv[:, 0:1], None, need_read=False)
        r0 = net.last_room
        built0 = net.built
        saved_raw = r0.raw.clone()
        r0.raw.fill_(-999.0)  # if slow equality check ran, it would miss and rebuild a new Room!
        m1, _, _, _, h = net.steps(o[:, 1:2], pv[:, 1:2], h, need_read=False)
        assert net.built == built0 and net.last_room is r0
        assert torch.allclose(m1, whole[:, 1:2], atol=1e-5)
        # In-place mutation h[0, 0] = 0 MUST invalidate the fast path and reset env 0 cleanly!
        r0.raw = saved_raw  # restore valid raw for env 1
        h[0, 0] = 0.0       # bumps h._version -> forces slow check!
        m2, _, _, _, h = net.steps(o[:, 2:3], pv[:, 2:3], h, need_read=False)
        m2_fresh = net.steps(o[:1, 2:3], pv[:1, 2:3], None, need_read=False)[0]
        assert torch.allclose(m2[:1], m2_fresh, atol=1e-5)

    # 2. Explicit lane_tag fast path survives h.detach() and torch.where (cut_rows) without reading r.raw!
    net.forget()
    tag = ('lane-A', 12345)
    m0, _, _, _, h = net.steps(o[:, 0:1], pv[:, 0:1], None, need_read=False, lane_tag=tag)
    r1 = net.last_room
    built1 = net.built
    r1.raw.fill_(-999.0)
    h_det = h.detach().clone()  # new tensor object (simulates cut_rows / detach)
    m1, _, _, _, h2 = net.steps(o[:, 1:2], pv[:, 1:2], h_det, need_read=False, lane_tag=tag)
    assert net.built == built1 and net.last_room is r1
    assert torch.allclose(m1.detach(), whole[:, 1:2], atol=1e-5)
    # hidden=None with same tag builds a fresh room (zero stale reuse)
    m0b, _, _, _, _ = net.steps(o[:, 0:1], pv[:, 0:1], None, need_read=False, lane_tag=tag)
    assert net.built == built1 + 1 and torch.allclose(m0b.detach(), m0.detach(), atol=1e-6)


def test_bf16_amp_embed_and_late_stay_bf16_without_fp32_promotion():
    """Under BF16 autocast, embed() keeps x/on/bias in bfloat16 (no FP32 promotion from x * on),
    k/v stored in Room.late stay bfloat16 with contiguous 1x v, and update_rel remains float32."""
    net = _lively_seq(seed=11).train()
    o, pv = _seq_inputs(2, 4, seed=31)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        x, on, bias, link, ncv = net.actor.embed(o[:, :1], pv[:, :1])
        assert x.dtype == torch.bfloat16, f'embed() leaked to {x.dtype}'
        assert on.dtype == torch.bfloat16
        assert bias.dtype == torch.bfloat16
        h = None
        for t in range(3):
            m, _, _, _, h = net.steps(o[:, t:t+1], pv[:, t:t+1], h, need_read=False)
        assert net.actor.rel_last.dtype == torch.float32
        assert net.actor.rel_feat_last.dtype == torch.float32
        for layer_late in net.last_room.late:
            for k_t, v_t in layer_late:
                assert k_t.dtype == torch.bfloat16, f'late k leaked to {k_t.dtype}'
                assert v_t.dtype == torch.bfloat16, f'late v leaked to {v_t.dtype}'
                assert v_t.is_contiguous()
                assert v_t.untyped_storage().nbytes() == v_t.numel()*2
    # _amp_ctx returns nullcontext on non-CUDA (and when already inside autocast)
    assert type(SQ._amp_ctx(True, o)).__name__ == 'nullcontext'


def test_enable_compile_has_zero_orig_mod_keys_and_supports_need_read_false():
    """enable_compile(dynamic=True) compiles outer hot helpers & layer cores while keeping ZERO `_orig_mod.` keys
    in state_dict() and preserving exact parity for need_read=True and need_read=False."""
    net_eager = _lively_seq(seed=13).train()
    net_ind = _lively_seq(seed=13).train()
    net_ind.load_state_dict(net_eager.state_dict())
    net_ind.enable_compile(dynamic=True)
    assert not any('_orig_mod' in k for k in net_ind.state_dict())
    assert not any('_orig_mod' in k for k in net_ind.actor.state_dict())
    assert list(net_ind.state_dict().keys()) == list(net_eager.state_dict().keys())

    net_comp = _lively_seq(seed=13).train()
    net_comp.load_state_dict(net_eager.state_dict())
    net_comp.enable_compile(dynamic=True, backend='eager')
    assert not any('_orig_mod' in k for k in net_comp.state_dict())
    assert list(net_comp.state_dict().keys()) == list(net_eager.state_dict().keys())

    o, pv = _seq_inputs(2, 4, seed=37)
    for need_read in (False, True):
        net_eager.forget(); net_comp.forget()
        net_eager.zero_grad(); net_comp.zero_grad()
        he = hc = None
        me_list, mc_list = [], []
        for t in range(3):
            me, _, _, le, he = net_eager.steps(o[:, t:t+1], pv[:, t:t+1], he, need_read=need_read)
            mc, _, _, lc, hc = net_comp.steps(o[:, t:t+1], pv[:, t:t+1], hc, need_read=need_read)
            assert (le is None) == (lc is None) == (not need_read)
            me_list.append(me); mc_list.append(mc)
        ye, yc = torch.cat(me_list, 1), torch.cat(mc_list, 1)
        assert torch.allclose(yc, ye, atol=1e-5)
        ye.square().sum().backward()
        yc.square().sum().backward()
        for (q, pe), (_, pc) in zip(net_eager.named_parameters(), net_comp.named_parameters()):
            if pe.grad is None:
                assert pc.grad is None, q
            else:
                assert pc.grad is not None and torch.allclose(pc.grad, pe.grad, atol=1e-4, rtol=1e-4), q


def test_event_seed_splits_plant_dynamics_from_events_and_noise():
    """C.build(..., event_seed=None) is bit-identical to baseline; distinct event_seeds keep bit-identical plant
    dynamics (segs[0]) while generating distinct event schedules, initial offsets, and step noise."""
    p_def = C.build(8, seed=42, teacher=False)
    p_none = C.build(8, seed=42, event_seed=None, teacher=False)
    assert torch.equal(p_def.P, p_none.P) and torch.equal(p_def.T, p_none.T)
    assert torch.equal(p_def.seg_k, p_none.seg_k) and torch.equal(p_def.kinds, p_none.kinds)
    assert torch.equal(p_def.start['u'], p_none.start['u'])

    p_e1 = C.build(8, seed=42, event_seed=1001, teacher=False)
    p_e2 = C.build(8, seed=42, event_seed=2002, teacher=False)
    # Plant commissioning dynamics (segment 0 & physical plant constants) are BIT-IDENTICAL across all three!
    assert torch.equal(p_e1.P[:, 0], p_def.P[:, 0])
    assert torch.equal(p_e2.P[:, 0], p_def.P[:, 0])
    for key in ('integ', 'arch', 'bandf', 'slew', 'T', 'rth', 'aa', 'nw', 'an_T'):
        assert torch.equal(p_e1.c[key], p_def.c[key]) and torch.equal(p_e2.c[key], p_def.c[key]), key
    # Events, initial off-steady draw, and step noise differ between event_seed=1001 and 2002!
    assert not torch.equal(p_e1.seg_k, p_e2.seg_k)
    assert not torch.equal(p_e1.kinds, p_e2.kinds)
    assert not torch.equal(p_e1.start['u'], p_e2.start['u'])
    z1, u1 = p_e1.noise()
    z2, u2 = p_e2.noise()
    assert not torch.equal(z1, z2) and not torch.equal(u1, u2)


def test_prefetch_noise_bit_identical_to_per_step_noise():
    """Program.prefetch_noise generates bit-identical (z, u) and rollout states as per-step noise()."""
    p1 = C.build(6, seed=77, teacher=False)
    p2 = C.build(6, seed=77, teacher=False)
    st1, st2 = C.reset(p1), C.reset(p2)
    H = 16
    zs, us = p2.prefetch_noise(0, H, 'cpu')
    assert zs.shape == (H, 6, p2.nnorm) and us.shape == (H, 6, p2.nuni)
    a = torch.zeros(6, p1.nmv)
    for j in range(H):
        st1, out1 = C.step(p1, st1, a, j)
        st2, out2 = C.step(p2, st2, a, j, rnd=(zs[j], us[j]))
        assert torch.equal(out1['cost'], out2['cost'])
        assert torch.equal(st1['y'], st2['y'])
    # Generator state after H steps is bit-identical
    z1_next, u1_next = p1.noise()
    z2_next, u2_next = p2.noise()
    assert torch.equal(z1_next, z2_next) and torch.equal(u1_next, u2_next)
    # Also test (H,) and (H=..., dev=...) call conventions
    z_a, u_a = p1.prefetch_noise(4)
    z_b, u_b = p2.prefetch_noise(H=4, dev='cpu')
    assert torch.equal(z_a, z_b) and torch.equal(u_a, u_b)


@pytest.mark.parametrize('dims', ('12x5', '24x10'))
def test_enable_sim_compile_and_priv_features_reuse_exact_parity(dims):
    """Compiled pure-tensor observe(), step(), and priv_features() match eager bit-for-bit on forward & gradients."""
    prog = C.build(4, seed=55, dims=dims, teacher=False)
    st0 = C.reset(prog)
    zs, us = prog.prefetch_noise(0, 6, 'cpu')
    torch.manual_seed(0)
    acts = [(.2*torch.randn(4, prog.nmv)).clamp(-1, 1) for _ in range(6)]

    def rollout(use_compile):
        C.enable_sim_compile(use_compile, dynamic=False, cpu=use_compile, backend='eager')
        N.enable_priv_compile(None)  # follow C.is_sim_compiled()
        try:
            st = {q: v.clone() for q, v in st0.items()}
            a_vars = [x.clone().requires_grad_(True) for x in acts]
            obs_l, cost_l, pf_l = [], [], []
            for k in range(6):
                obs_l.append(C.observe(prog, st, k))
                pf_l.append(N.priv_features(prog, st, k))
                st, out = C.step(prog, st, a_vars[k], k, rnd=(zs[k], us[k]), through=True, stops=True)
                cost_l.append(out['soft'])
            pf_end = N.priv_features(prog, st, 6)
            loss = sum(c.sum() for c in cost_l) + pf_end[0].sum() + pf_end[1].sum()
            loss.backward()
            return (torch.stack(obs_l), torch.stack(cost_l).detach(),
                    tuple(x.detach() for x in pf_end), [av.grad.detach().clone() for av in a_vars])
        finally:
            C.enable_sim_compile(False)
            N.enable_priv_compile(False)

    obs_e, cost_e, pf_e, g_e = rollout(False)
    obs_c, cost_c, pf_c, g_c = rollout(True)
    assert torch.allclose(obs_c, obs_e, atol=1e-6)
    assert torch.allclose(cost_c, cost_e, atol=1e-6)
    for xe, xc in zip(pf_e, pf_c):
        assert torch.allclose(xc, xe, atol=1e-6)
    for ge, gc in zip(g_e, g_c):
        assert torch.allclose(gc, ge, atol=1e-5)

    # Explicit P/t/phi reuse in priv_features matches default call bit-for-bit
    P0, t0 = C.step_ctx(prog, 0)
    phi0 = C._phi(st0['xv'], st0['d'], prog.c)
    pf_default = N.priv_features(prog, st0, 0)
    pf_reused = N.priv_features(prog, st0, 0, P=P0, t=t0, phi=phi0)
    for a_, b_ in zip(pf_default, pf_reused):
        assert torch.equal(a_, b_)


def test_prior_curriculum_probs_shift_distributions_and_keep_defaults_bit_identical():
    """prior.py cls_p/sep_p/integ_p smoothly shift dead-time class, SEP archetype, and integrating-level shares
    while default absent keys stay 100% bit-identical and all generated points pass `point_ok`."""
    # 1. Bit-identical defaults (shape=None vs shape={})
    s_none = PR.draw_episode(PR.Rng(314), 16, shape=None)
    s_empty = PR.draw_episode(PR.Rng(314), 16, shape={})
    assert torch.equal(s_none[0][0]['K'], s_empty[0][0]['K'])
    assert torch.equal(s_none[0][0]['integ'], s_empty[0][0]['integ'])
    assert torch.equal(s_none[0][0]['arch'], s_empty[0][0]['arch'])

    # 2. Shifted hard-regime probabilities on n=512 plants
    n = 512
    ro_def = PR.draw_roots(PR.Rng(99), n)
    shape_hard = dict(cls_p=(.10, .30, .60), sep_p=.55, integ_p=.35)
    ro_hard = PR.draw_roots(PR.Rng(99), n, shape=shape_hard)
    assert float((ro_def['cls'] == 2).float().mean()) == pytest.approx(.20, abs=.06)
    assert float((ro_hard['cls'] == 2).float().mean()) == pytest.approx(.60, abs=.06)
    assert float((ro_def['arch'] == PR.SEP).float().mean()) == pytest.approx(.25, abs=.06)
    assert float((ro_hard['arch'] == PR.SEP).float().mean()) == pytest.approx(.55, abs=.06)

    p_def, t_def, _ = PR.draw_start(PR.Rng(123), 256)
    p_hard, t_hard, _ = PR.draw_start(PR.Rng(123), 256, shape=shape_hard)
    assert bool(PR.point_ok(p_hard, t_hard).all())
    lev_def = float((p_def['integ'] > 0).any(-1).float().mean())
    lev_hard = float((p_hard['integ'] > 0).any(-1).float().mean())
    assert lev_hard > lev_def + 0.15, f'expected higher level share: {lev_hard:.3f} vs {lev_def:.3f}'
    assert float((p_hard['arch'] == PR.SEP).float().mean()) > 0.45

    # 3. Also works on 24x10 multi-unit plants cleanly
    p_big, t_big, _ = PR.draw_start(PR.Rng(7), 8, dims=PR.SIZES['24x10'], shape=shape_hard)
    assert bool(PR.point_ok(p_big, t_big).all())


def test_safe_torch_load_handles_pathlib_local_and_windows_path_and_of_state_meter(tmp_path):
    """safe_torch_load cleanly loads checkpoints carrying Python 3.13 `pathlib._local` and `WindowsPath`,
    restoring pathlib state afterwards; of_state preserves `net.meter` on both SeqAIO and AIO."""
    # Construct authentic pickle payloads referencing pathlib._local.PosixPath, pathlib._local.WindowsPath, and pathlib.WindowsPath
    fake_local = type(sys)('pathlib._local')
    _P313Posix = type('PosixPath', (), {
        '__module__': 'pathlib._local',
        '__qualname__': 'PosixPath',
        '__reduce__': lambda self: (fake_local.PosixPath, ('/content/aio/val-cache',)),
    })
    _P313Win = type('WindowsPath', (), {
        '__module__': 'pathlib._local',
        '__qualname__': 'WindowsPath',
        '__reduce__': lambda self: (fake_local.WindowsPath, ('D:\\aiogym2-runs\\exp\\last.pt',)),
    })
    fake_local.PosixPath = _P313Posix
    fake_local.WindowsPath = _P313Win

    had_local = 'pathlib._local' in sys.modules
    orig_local = sys.modules.get('pathlib._local')
    orig_win = pathlib.WindowsPath
    ck_path = tmp_path / 'cross_py_os.pt'
    try:
        sys.modules['pathlib._local'] = fake_local
        payload = {
            'p313_posix': _P313Posix(),
            'p313_win': _P313Win(),
            'win_path': pathlib.PureWindowsPath('D:/runs/a.pt'),
            't': torch.tensor([1.0, 2.0]),
        }
        torch.save(payload, ck_path)
    finally:
        if had_local:
            sys.modules['pathlib._local'] = orig_local
        else:
            sys.modules.pop('pathlib._local', None)
        pathlib.WindowsPath = orig_win

    # Also write a checkpoint whose pickle references pathlib.WindowsPath directly
    _FakeWinStd = type('WindowsPath', (), {
        '__module__': 'pathlib',
        '__qualname__': 'WindowsPath',
        '__reduce__': lambda self: (pathlib.WindowsPath, ('C:\\runs\\b.pt',)),
    })
    ck_win_path = tmp_path / 'win_std.pt'
    orig_win = pathlib.WindowsPath
    try:
        pathlib.WindowsPath = _FakeWinStd
        torch.save({'win': _FakeWinStd(), 'x': torch.ones(2)}, ck_win_path)
    finally:
        pathlib.WindowsPath = orig_win

    loaded = N.safe_torch_load(ck_path, map_location='cpu')
    assert torch.equal(loaded['t'], torch.tensor([1.0, 2.0]))
    assert isinstance(loaded['p313_posix'], pathlib.Path) and 'val-cache' in str(loaded['p313_posix'])
    assert isinstance(loaded['p313_win'], pathlib.Path) and 'last.pt' in str(loaded['p313_win'])

    loaded_win = N.safe_torch_load(ck_win_path, map_location='cpu')
    assert isinstance(loaded_win['win'], pathlib.Path) and 'b.pt' in str(loaded_win['win'])
    assert pathlib.WindowsPath is orig_win
    if not had_local:
        assert 'pathlib._local' not in sys.modules

    # Verify of_state preserves net.meter cleanly on both SeqAIO and AIO
    seq = _lively_seq('d32,l1,h4,m4,g1,q1', w=16)
    s_on = N.of_state(seq.actor.state_dict(), seq.pairs.state_dict(), config={'meter': True, 'probe': 0.1, 'unpaired': 0.25})
    s_off = N.of_state(seq.actor.state_dict(), seq.pairs.state_dict(), config={})
    assert s_on.meter is True and s_off.meter is False
    assert s_on.probe == pytest.approx(0.1) and s_on.unpaired == pytest.approx(0.25)

    aio_net = N.AIO()
    a_on = N.of_state(aio_net.actor.state_dict(), aio_net.pairs.state_dict(), config={'meter': True})
    a_off = N.of_state(aio_net.actor.state_dict(), aio_net.pairs.state_dict(), config=None)
    assert a_on.meter is True and a_off.meter is False

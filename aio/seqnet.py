"""A plain attention network over the packets (11-16; nothing uses it unless asked): every row and every MV is a token
at every decision, and the layers alternate between two kinds of attention.

  time      a token over its own past (causal, rotary positions): what this row, this MV did before
  channels  the tokens of one decision over each other (rows and MVs alike; a learned bias per head where a pair is
            published as paired): what the others are doing now

Nothing in it has the shape of a least-squares fit: no running sums of products, no normalising, no rounds of an
iteration, no product of read-outs (aio.net.Stats, aio.reader, the law). What it reads of the plant it has to find by
attending over the history. No weight depends on the number of rows or MVs.

  read      per pair, from the row's token, the MV's and whether they are published as paired: the responses
            (aio.net's knots, asinh units), the label the others' read-outs are taught on
  act       per MV, from its token: the mean of the move before the tanh

`spec`: 'd192,l4,h6' (width, layers of each kind, heads); 11-23 additions, all off unless asked (weights from before
are the same network):
  q1   QK-norm: queries and keys of every attention normalised per head (RMSNorm) before the positions are turned in
  g1   lag routing: a token also attends to every token of the decisions LAGS before (rows to MVs, MVs to rows, at lags):
       which valve moved how long before what this row now shows, in one hop (the keys and values of the time attention)
  mK   a memory of K numbers per token that forgets over 4 to 2048 decisions (a learned linear recurrence after the
       layers, read by the heads): what is older than the w decisions the attention sees

What a decision costs (SeqAIO): without a graph a single decision is the only one computed, on the keys and values kept
of the ones before it (Room; not in the state, built again from the state's packets where it is missing); a window with
a graph attends to the state's decisions without a graph through them. A single decision with a graph (the plant's
gradient) is the only one computed too: its graph goes through the decisions since the state last came detached, their
keys and values held with their graphs (Room.late; the graphs are kept, not rebuilt: no --lean)."""
import contextlib
import math
import os
import weakref
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from aio.actor import symlog, unpack
from aio.interface import CF, MF

D, LAYERS, HEADS = 192, 4, 6
LAGS = (1, 3, 8, 24, 64)                  # g1: the decisions back a token attends to every token of
LAGS_BY = {1: LAGS, 2: (1, 3, 8, 16, 24, 36, 48, 64)}   # g2: dense dead-time coverage closing the 16/36/48 lag holes
LAG_CHUNK = 16                       # decisions whose lag scores are held at once (Layer.lagged)
REL_K = 8                            # r1/r2: 8-stage strictly-positive Erlang/Gamma SSM cascade per MV
REL_E = 2*REL_K                      # r1/r2: 16 pairwise DeltaNet channels (8 du impulse + 8 u_dev level-rate)
REL_F = REL_E+4                      # r1: 20 dimensionless pairwise features fed to attention & message passing
REL_F2 = REL_F+4                     # r2: +4 dimensionally-exact physical gain & activity features (|K*slew|, |K_norm|, |K_int*slew|, RMS(dy))
REL_MP = 64                          # r1/r2: edge-conditioned 1-hop message passing hidden width
REL_TAUS = (1.5, 2.5, 4.5, 7., 10., 13., 16., 20.)   # r1/r2: initial Erlang time constants (peaks ~[0,1,4,9,17,28,42,60])
FIELDS = ('d', 'layers', 'heads', 'mem', 'lags', 'qk', 'mup', 'fb', 'edge', 'rel')


def rel_feat_dim(rel):
    """Pairwise feature width fed to attention & message passing (r1: 20, r2: 24)."""
    return REL_F2 if rel >= 2 else (REL_F if rel else 0)


def rel_state_size(ncv, nmv):
    """Per-plant flat float32 state length for r1/r2 (MV Erlang states + CV variance + NCVxNMVx16 DeltaNet matrix)."""
    return nmv*(2*REL_K+2+REL_E)+ncv*2+ncv*nmv*REL_E


def spec(asked):
    """What `asked` ('d192,l4,h6', any part left out, or True) says as Tokens' arguments."""
    out = dict(d=D, layers=LAYERS, heads=HEADS, mem=0, lags=0, qk=0, mup=0, fb=0, edge=0, rel=0)
    for x in ([] if asked is True or not asked else str(asked).split(',')):
        q = {'d': 'd', 'l': 'layers', 'h': 'heads', 'm': 'mem', 'g': 'lags', 'q': 'qk', 'u': 'mup', 'f': 'fb', 'e': 'edge', 'r': 'rel'}.get(x[:1])
        assert q and (x[1:].isdigit() or not x[1:]), asked
        out[q] = int(x[1:] or 1)                                  # ('q' alone: on)
    assert out['d'] % out['heads'] == 0 and (out['d']//out['heads']) % 2 == 0, 'the width over the heads is even (rotary positions)'
    assert out['lags'] in (0, 1, 2), f'unsupported lag mode g{out["lags"]}'
    assert not out['fb'] or out['mem'] > 0, 'f1 (memory feedback) requires mK > 0'
    assert out['rel'] in (0, 1, 2), f'unsupported rel mode r{out["rel"]}'
    return out


def said(sp):
    """The spec string of Tokens' arguments (`spec`'s inverse; the parts that are off left out)."""
    out = 'd%d,l%d,h%d' % (sp['d'], sp['layers'], sp['heads'])
    return out+''.join(',%s%d' % (k, sp[q]) for k, q in (('m', 'mem'), ('g', 'lags'), ('q', 'qk'), ('u', 'mup'), ('f', 'fb'), ('e', 'edge'), ('r', 'rel')) if sp[q])


_ROTARY_FREQ = {}
_ROTARY_TBL = {}
_EYE_CACHE = {}
_LAG_TAUS = {}
_ARANGE_CACHE = {}


def _is_compiling():
    return hasattr(torch, 'compiler') and torch.compiler.is_compiling()


def _eye_cached(n, dev, dtype):
    if _is_compiling():
        return torch.eye(n, device=dev, dtype=dtype)
    key = (n, dev, dtype)
    e = _EYE_CACHE.get(key)
    if e is None:
        if len(_EYE_CACHE) >= 32:
            _EYE_CACHE.clear()
        e = torch.eye(n, device=dev, dtype=dtype)
        _EYE_CACHE[key] = e
    return e


def _lag_taus_cached(lag_list, dev):
    if _is_compiling():
        return torch.tensor(lag_list, device=dev, dtype=torch.long)
    key = (lag_list, dev)
    t = _LAG_TAUS.get(key)
    if t is None:
        if len(_LAG_TAUS) >= 32:
            _LAG_TAUS.clear()
        t = torch.tensor(lag_list, device=dev, dtype=torch.long)
        _LAG_TAUS[key] = t
    return t


def _arange_cached(n, dev):
    if _is_compiling():
        return torch.arange(n, device=dev, dtype=torch.long)
    tbl = _ARANGE_CACHE.get(dev)
    if tbl is None or tbl.shape[0] < n:
        cap = max(256, 1 << (max(n, 1) - 1).bit_length())
        if len(_ARANGE_CACHE) >= 16:
            _ARANGE_CACHE.clear()
        tbl = torch.arange(cap, device=dev, dtype=torch.long)
        _ARANGE_CACHE[dev] = tbl
    return tbl[:n]


def _pos_tensor(ps, dev):
    """(len(ps),) long tensor on `dev`, avoiding CPU->GPU H2D copy whenever `ps` is consecutive integers."""
    m = len(ps)
    if m == 0:
        return _arange_cached(0, dev)
    if ps[-1] - ps[0] == m - 1:
        return _arange_cached(m, dev) + ps[0]
    return torch.tensor(ps, device=dev, dtype=torch.long)


def _amp_ctx(amp, x):
    """Enter CUDA bfloat16 autocast only if `amp` is enabled and we are not already inside a hoisted autocast."""
    if amp and x.is_cuda and not torch.is_autocast_enabled('cuda'):
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


def rotary_cs(h, start, T, dev, dtype):
    """Cached (cos, sin) slices of shape (T, h//2) for rotary position embeddings (O(1) slice, zero per-step trig)."""
    dt = torch.float32 if dev.type == 'mps' else torch.float64
    if isinstance(start, int) and start >= 0:
        tkey = (h, dev, dtype)
        tbl = _ROTARY_TBL.get(tkey)
        need = start + T
        if tbl is None or tbl[0].shape[0] < need:
            cap = max(1024, 1 << (need - 1).bit_length())
            fkey = (h, dev, dt)
            f = _ROTARY_FREQ.get(fkey)
            if f is None:
                f = torch.exp(torch.arange(0, h, 2, device=dev, dtype=dt)*(-math.log(10000.)/h))
                _ROTARY_FREQ[fkey] = f
            a = torch.arange(cap, device=dev, dtype=dt)[:, None]*f
            tbl = (a.cos().to(dtype), a.sin().to(dtype))
            if len(_ROTARY_TBL) >= 32:
                _ROTARY_TBL.clear()
            _ROTARY_TBL[tkey] = tbl
        return tbl[0][start:need], tbl[1][start:need]
    fkey = (h, dev, dt)
    f = _ROTARY_FREQ.get(fkey)
    if f is None:
        f = torch.exp(torch.arange(0, h, 2, device=dev, dtype=dt)*(-math.log(10000.)/h))
        _ROTARY_FREQ[fkey] = f
    a = (start+torch.arange(T, device=dev, dtype=dt))[:, None]*f
    return a.cos().to(dtype), a.sin().to(dtype)


def apply_rotary(x, c, s):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1*c-x2*s, x1*s+x2*c), -1).flatten(-2)


def rotary(x, start=0):
    """x (..., T, h) with h even, turned by its position start, start+1, ...: attention then sees only how far apart two decisions are."""
    T, h = x.shape[-2], x.shape[-1]
    c, s = rotary_cs(h, start, T, x.device, x.dtype)
    return apply_rotary(x, c, s)


class Layer(nn.Module):
    def __init__(self, d, heads, qk=0, lags=0, edge=0, rel=0):
        super().__init__()
        self.h = heads
        dh = d//heads
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv_t, self.out_t = nn.Linear(d, 3*d), nn.Linear(d, d)
        self.qkv_c, self.out_c = nn.Linear(d, 3*d), nn.Linear(d, d)
        self.pair = nn.Parameter(torch.zeros(heads))              # the bias of a head between a row and an MV published as paired
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.SiLU(), nn.Linear(4*d, d))
        self.qk, self.lags, self.edge, self.rel = bool(qk), int(lags), int(edge), int(rel)
        self.lag_list = LAGS_BY[self.lags] if self.lags else ()
        if self.qk:                                               # q1: queries and keys normalised per head (a key cannot grow past bfloat16: sq-big-s1 11-20)
            self.qn_t, self.kn_t, self.qn_c, self.kn_c = (nn.RMSNorm(dh) for _ in range(4))
        if self.lags:                                             # g1/g2: every token of the decisions lag_list before, through the time attention's keys and values
            self.n4, self.q_l, self.out_l = nn.LayerNorm(d), nn.Linear(d, d), nn.Linear(d, d)
            self.lag_b = nn.Parameter(torch.zeros(heads, len(self.lag_list)))     # per head: how much each lag counts
            self.lag_pair = nn.Parameter(torch.zeros(heads))              # a row and an MV published as paired
            self.lag_own = nn.Parameter(torch.zeros(heads))               # a token's own channel
            if self.qk:
                self.qn_l = nn.RMSNorm(dh)
        if self.edge:                                             # eK: persistent pairwise (CV, MV) edge memory injected into channel attention bias (zero-init)
            self.edge_bias = nn.Linear(self.edge, heads, bias=False)
            nn.init.zeros_(self.edge_bias.weight)
        if self.rel:                                              # r1/r2: multi-scale DeltaNet pairwise bias into BOTH channel & lag attention (zero-init)
            rf_dim = rel_feat_dim(self.rel)
            self.rel_bias = nn.Linear(rf_dim, heads, bias=False)
            nn.init.zeros_(self.rel_bias.weight)
            if self.lags:
                self.rel_lag_bias = nn.Linear(rf_dim, heads*len(self.lag_list), bias=False)
                nn.init.zeros_(self.rel_lag_bias.weight)
        for lin in (self.out_t, self.out_c, self.ff[-1])+((self.out_l,) if self.lags else ()):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def norm_qk(self, q, k, which):
        if not self.qk:
            return q, k
        return getattr(self, 'qn_'+which)(q), getattr(self, 'kn_'+which)(k)

    def forward(self, x, on, bias, seen=None, before=None, keep=None, start=0, edge0=None, ncv=None):
        """x (b,T,N,d) tokens, on (b,T,N) which are in service, bias (b,T,N,N) 1 where two are published as paired,
        seen (b,T,T') bool or None: which decisions a decision attends to (None: all before it and itself).
        before: the keys and values (b*N,h,P,d/h) of P decisions ahead of these (as `keep` took them: T' = P+T), which
        are not computed here; keep: a list this layer's own are put in; start: the position of the first decision."""
        b, T, N, d = x.shape
        h = self.h
        # ---- time: every token over its own past
        q, k, v = self.qkv_t(self.n1(x)).transpose(1, 2).reshape(b*N, T, 3, h, d//h).permute(2, 0, 3, 1, 4)     # (b*N,h,T,d/h) each
        q, k = self.norm_qk(q, k, 't')
        q, k = rotary(q, start), rotary(k, start)
        if keep is not None:
            keep.append((k, v))
        if before is not None:
            k, v = torch.cat((before[0].to(k.dtype), k), 2), torch.cat((before[1].to(v.dtype), v), 2)
        if seen is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=seen[:, None, None].expand(b, N, 1, T, k.shape[2]).reshape(b*N, 1, T, k.shape[2]))
        x = x+self.out_t(y.transpose(1, 2).reshape(b, N, T, d).transpose(1, 2))*on[..., None]
        if self.lags:
            x = self.lagged(x, on, bias, *self.lag_seq(k, v, b, N, T, seen), start)
        return self.across(x, on, bias, edge0, ncv)

    def lag_seq(self, k, v, b, N, T, seen):
        """The keys and values (b,N,h,T,d/h) of the decisions lag_list before each of the T last of k, v (b*N,h,S,d/h), and
        which are there (b,T,L)."""
        S, h, dh, Lm = k.shape[2], k.shape[1], k.shape[3], max(self.lag_list)
        P = S-T
        kp, vp = F.pad(k, (0, 0, Lm, 0)), F.pad(v, (0, 0, Lm, 0))
        t = torch.arange(T, device=k.device)
        KL, VL, ok = [], [], []
        for tau in self.lag_list:
            a0 = Lm+P-tau
            KL.append(kp[:, :, a0:a0+T].reshape(b, N, h, T, dh))
            VL.append(vp[:, :, a0:a0+T].reshape(b, N, h, T, dh))
            idx = P+t-tau
            there = (idx >= 0)[None].expand(b, T)
            if seen is not None:
                there = there & seen.gather(2, idx.clamp_min(0)[None, :, None].expand(b, T, 1))[..., 0]
            ok.append(there)
        return KL, VL, torch.stack(ok, -1)

    def lag_ring(self, K, V, there, where, at, pos, b, N, idx=None, ok=None):
        """The same for one new decision at slot `at` of the ring K, V (b*N,h,S,d/h): stacked (b,h,N*L,dh) each, (b,1,L)."""
        S, h, dh = K.shape[2], K.shape[1], K.shape[3]
        L = len(self.lag_list)
        if idx is None:
            taus = _lag_taus_cached(self.lag_list, K.device)
            idx = (at-taus) % S
            ok = (there.index_select(1, idx) & (where.index_select(0, idx) == (pos-taus))[None])[:, None]
        K_all = K.index_select(2, idx).reshape(b, N, h, L, dh).permute(0, 2, 1, 3, 4).reshape(b, h, N*L, dh)
        V_all = V.index_select(2, idx).reshape(b, N, h, L, dh).permute(0, 2, 1, 3, 4).reshape(b, h, N*L, dh)
        return K_all, V_all, ok

    def lagged(self, x, on, bias, KL, VL, ok, start, rel_feat=None, ncv=None, cs=None):
        """The lag routing (g1/g2): every token over every token of the decisions LAGS before (KL, VL: list of (b,N,h,T,d/h)
        or pre-stacked (b*T,h,N*L,d/h), ok (b,T,L) which are there), with a sink that says nothing (where none is there). Its
        scores are (b,T,h,N,N,L): over LAG_CHUNK decisions at a time, and with a graph each chunk's are computed again at the
        backward pass (11-23: 64 windows of 128 decisions of a 48x20 plant held 90 GB at once)."""
        T = x.shape[1]
        if T <= LAG_CHUNK:
            return self.lagged_part(x, on, bias, KL, VL, ok, start, rel_feat, ncv, cs)
        out = []
        for t0 in range(0, T, LAG_CHUNK):
            sl = slice(t0, t0+LAG_CHUNK)
            rf_sl = None if rel_feat is None else (rel_feat if rel_feat.dim() == 4 else rel_feat[:, sl])
            part = (x[:, sl], on[:, sl], bias[:, sl], [K[:, :, :, sl] for K in KL], [V[:, :, :, sl] for V in VL], ok[:, sl], start+t0, rf_sl, ncv)
            if torch.is_grad_enabled():
                out.append(checkpoint(self.lagged_part, *part, use_reentrant=False))
            else:
                out.append(self.lagged_part(*part))
        return torch.cat(out, 1)

    def lagged_part(self, x, on, bias, KL, VL, ok, start, rel_feat=None, ncv=None, cs=None):
        b, T, N, d = x.shape
        h, L = self.h, len(self.lag_list)
        dh = d//h
        q = self.q_l(self.n4(x)).reshape(b, T, N, h, dh).permute(0, 2, 3, 1, 4)                                 # (b,N,h,T,dh)
        if self.qk:
            q = self.qn_l(q)
        q = apply_rotary(q, cs[0], cs[1]) if cs is not None else rotary(q, start)
        qf = q.permute(0, 3, 2, 1, 4).reshape(b*T, h, N, dh)                                                    # (b*T,h,N,dh)
        if torch.is_tensor(KL):
            K_all, V_all = KL, VL
        else:
            K_all = torch.stack(KL, 2).permute(0, 4, 3, 1, 2, 5).reshape(b*T, h, N*L, dh)
            V_all = torch.stack(VL, 2).permute(0, 4, 3, 1, 2, 5).reshape(b*T, h, N*L, dh)
        sc = (torch.matmul(qf, K_all.to(qf.dtype).transpose(-1, -2)).float()/math.sqrt(dh)).reshape(b, T, h, N, N, L)
        eye = _eye_cached(N, x.device, sc.dtype)
        sc = sc+self.lag_b.float()[None, None, :, None, None, :]+(eye*self.lag_own.float()[:, None, None])[None, None, ..., None] \
            + (bias.to(sc.dtype)[:, :, None]*self.lag_pair.float()[None, None, :, None, None])[..., None]
        if self.rel and rel_feat is not None:
            r_in = rel_feat[:, None] if rel_feat.dim() == 4 else rel_feat
            rlb = self.rel_lag_bias(r_in.to(self.rel_lag_bias.weight.dtype)).float().unflatten(-1, (h, L)).permute(0, 1, 4, 2, 3, 5)
            sc_c = torch.cat((sc[:, :, :, :ncv, :ncv], sc[:, :, :, :ncv, ncv:]+rlb), 4)
            sc_m = torch.cat((sc[:, :, :, ncv:, :ncv]+rlb.transpose(3, 4), sc[:, :, :, ncv:, ncv:]), 4)
            sc = torch.cat((sc_c, sc_m), 3)
        m = (on[:, :, None, None, :, None] > 0) & ok[:, :, None, None, None, :]
        sc = sc.masked_fill(~m, float('-inf')).flatten(-2)
        a = torch.softmax(torch.cat((sc, torch.zeros_like(sc[..., :1])), -1), -1)[..., :-1]                     # (b,T,h,N,N*L)
        y = torch.matmul(a.reshape(b*T, h, N, N*L).to(V_all.dtype), V_all)                                      # (b*T,h,N,dh)
        y = y.reshape(b, T, h, N, dh).permute(0, 1, 3, 2, 4).reshape(b, T, N, d)
        return x+self.out_l(y.to(x.dtype))*on[..., None]

    def _step_infer_core(self, x, on, bias, K, V, there, cos_p, sin_p, lag_idx, lag_ok, edge0, ncv, rel_feat):
        """Fixed-arity single-decision inference core (compilable via torch.compile without pos specialization)."""
        b, _, N, d = x.shape
        h, dh = self.h, d//self.h
        q, k, v = self.qkv_t(self.n1(x)).reshape(b*N, 3, h, 1, dh).unbind(1)                                    # (b*N,h,1,d/h) each
        q, k = self.norm_qk(q, k, 't')
        q, k = apply_rotary(q, cos_p, sin_p), apply_rotary(k, cos_p, sin_p)
        k0, v0 = k[:, :, 0].to(K.dtype), v[:, :, 0].to(V.dtype)
        return q, k0, v0

    def _step_infer_post(self, x, on, bias, q, K, V, there, cos_p, sin_p, lag_idx, lag_ok, edge0, ncv, rel_feat):
        b, _, N, d = x.shape
        h, dh = self.h, d//self.h
        y = F.scaled_dot_product_attention(q.to(K.dtype), K, V, attn_mask=there[:, None, None, None].expand(b, N, 1, 1, -1).reshape(b*N, 1, 1, -1))
        x = x+self.out_t(y.to(x.dtype).transpose(1, 2).reshape(b, N, 1, d).transpose(1, 2))*on[..., None]
        if self.lags:
            K_all, V_all, ok = self.lag_ring(K, V, there, None, 0, 0, b, N, lag_idx, lag_ok)
            x = self.lagged_part(x, on, bias, K_all, V_all, ok, 0, rel_feat, ncv, (cos_p, sin_p))
        return self.across(x, on, bias, edge0, ncv, rel_feat)

    def step(self, x, on, bias, K, V, there, at, pos, where=None, edge0=None, ncv=None, rel_feat=None, cs=None, lag_idx=None, lag_ok=None):
        """forward for one new decision, its past not computed again: x (b,1,N,d); K, V (b*N,h,S,d/h) the keys (turned)
        and values of the decisions kept, the new one's written at slot `at`; there (b,S) which slots a plant has
        (the new one among them); pos: the new decision's position; where (S,): the position of the decision in a slot (g1/g2)."""
        b, _, N, d = x.shape
        dh = d//self.h
        cos_p, sin_p = rotary_cs(dh, pos, 1, x.device, x.dtype) if cs is None else cs
        if self.lags and lag_idx is None:
            S = K.shape[2]
            taus = _lag_taus_cached(self.lag_list, K.device)
            lag_idx = (at-taus) % S
            lag_ok = (there.index_select(1, lag_idx) & (where.index_select(0, lag_idx) == (pos-taus))[None])[:, None]
        q, k0, v0 = self._step_infer_core(x, on, bias, K, V, there, cos_p, sin_p, lag_idx, lag_ok, edge0, ncv, rel_feat)
        K[:, :, at], V[:, :, at] = k0, v0
        return self._step_infer_post(x, on, bias, q, K, V, there, cos_p, sin_p, lag_idx, lag_ok, edge0, ncv, rel_feat)

    def _step_core_late(self, x, on, bias, K, V, see, Kl, Vl, live_all, K_lag, V_lag, ok_lag, cos_p, sin_p, edge0, ncv, rel_feat, cut_late=None, cut_lag=None):
        """Fixed-arity differentiable layer core (with late graph tokens; directly torch.compile-able with dynamic=True)."""
        b, _, N, d = x.shape
        h, S, dh = self.h, K.shape[2], d//self.h
        q, k, v = self.qkv_t(self.n1(x)).reshape(b*N, 3, h, 1, dh).unbind(1)                                    # (b*N,h,1,d/h) each
        q, k = self.norm_qk(q, k, 't')
        q, k, v = apply_rotary(q, cos_p, sin_p)[:, :, 0], apply_rotary(k, cos_p, sin_p)[:, :, 0], v[:, :, 0].contiguous()
        ring = torch.matmul(q[:, :, None].to(K.dtype), K.transpose(-1, -2))[:, :, 0].to(q.dtype)                # (b*N,h,S)
        ring = ring.masked_fill(~see[:, None, None].expand(b, N, 1, S).reshape(b*N, 1, S), float('-inf'))
        if Kl is not None:
            if cut_late is not None:
                m_cut = cut_late[:, None, None, :, None].expand(b, N, 1, Kl.shape[2], 1).reshape(b*N, 1, Kl.shape[2], 1)
                Kl = torch.where(m_cut, Kl.detach(), Kl)
                Vl = torch.where(m_cut, Vl.detach(), Vl)
            Kl_all = torch.cat((Kl, k[:, :, None]), 2)                                                          # (b*N,h,n_late+1,dh)
            Vl_all = torch.cat((Vl, v[:, :, None]), 2)
            since = torch.matmul(q[:, :, None], Kl_all.transpose(-1, -2))[:, :, 0]                              # (b*N,h,n_late+1)
            since = since.masked_fill(~live_all[:, None, None, :].expand(b, N, 1, Kl_all.shape[2]).reshape(b*N, 1, Kl_all.shape[2]), float('-inf'))
            a = torch.softmax(torch.cat((ring, since), -1).float()/math.sqrt(dh), -1)
            y = torch.matmul(a[:, :, None, :S].to(V.dtype), V)[:, :, 0].to(q.dtype)+torch.matmul(a[:, :, None, S:].to(Vl_all.dtype), Vl_all)[:, :, 0].to(q.dtype)
        else:
            a = torch.softmax(torch.cat((ring, (q*k).sum(-1, keepdim=True)), -1).float()/math.sqrt(dh), -1)
            y = torch.matmul(a[:, :, None, :S].to(V.dtype), V)[:, :, 0].to(q.dtype)+a[..., -1:].to(v.dtype)*v
        x = x+self.out_t(y.reshape(b, N, d)[:, None])*on[..., None]
        if self.lags and K_lag is not None:
            if cut_lag is not None:
                L = len(self.lag_list)
                m_lag = cut_lag[:, None, None, :, None].expand(b, 1, N, L, 1).reshape(b, 1, N*L, 1)
                K_lag = torch.where(m_lag, K_lag.detach(), K_lag)
                V_lag = torch.where(m_lag, V_lag.detach(), V_lag)
            x = self.lagged_part(x, on, bias, K_lag, V_lag, ok_lag, 0, rel_feat, ncv, (cos_p, sin_p))
        return self.across(x, on, bias, edge0, ncv, rel_feat), k, v

    def _step_graph_body(self, x, on, bias, K, V, see, cos_p, sin_p, n_late, n_lags, ncv, edge0, rel_feat,
                         lag_plan, ring_idx, ring_oks, cut_late, cut_lag, *tensors):
        """Pure functional body of `step_graph` (compatible with activation checkpointing, O(H) saved memory)."""
        b, _, N, d = x.shape
        h, dh = self.h, d//self.h
        if n_late:
            Kl_l = tensors[:n_late]
            Vl_l = tensors[n_late:2*n_late]
            lv_l = tensors[2*n_late:3*n_late]
            Kl = torch.stack(Kl_l, 2)
            Vl = torch.stack(Vl_l, 2)
            live_all = torch.stack(lv_l+(torch.ones_like(see[:, 0]),), 1)
        else:
            Kl_l = Vl_l = lv_l = ()
            Kl = Vl = live_all = None
        if lag_plan is not None:
            L = len(lag_plan)
            if ring_idx is not None:
                rK = K.index_select(2, ring_idx).unbind(2)
                rV = V.index_select(2, ring_idx).unbind(2)
                rOk = ring_oks.unbind(1)
            K_lag = torch.stack([Kl_l[idx] if is_late else rK[idx] for is_late, idx in lag_plan], 2).to(x.dtype).reshape(b, N, h, L, dh).permute(0, 2, 1, 3, 4).reshape(b, h, N*L, dh)
            V_lag = torch.stack([Vl_l[idx] if is_late else rV[idx] for is_late, idx in lag_plan], 2).to(x.dtype).reshape(b, N, h, L, dh).permute(0, 2, 1, 3, 4).reshape(b, h, N*L, dh)
            ok_lag = torch.stack([lv_l[idx] if is_late else rOk[idx] for is_late, idx in lag_plan], -1)[:, None]
        elif n_lags:
            off = 3*n_late
            K_lag = torch.stack(tensors[off:off+n_lags], 2).to(x.dtype).reshape(b, N, h, n_lags, dh).permute(0, 2, 1, 3, 4).reshape(b, h, N*n_lags, dh)
            V_lag = torch.stack(tensors[off+n_lags:off+2*n_lags], 2).to(x.dtype).reshape(b, N, h, n_lags, dh).permute(0, 2, 1, 3, 4).reshape(b, h, N*n_lags, dh)
            ok_lag = torch.stack(tensors[off+2*n_lags:off+3*n_lags], -1)[:, None]
        else:
            K_lag = V_lag = ok_lag = None
        return self._step_core_late(x, on, bias, K, V, see, Kl, Vl, live_all, K_lag, V_lag, ok_lag, cos_p, sin_p, edge0, ncv, rel_feat, cut_late, cut_lag)

    def step_graph(self, x, on, bias, K, V, see, late, lives, pos, lagkv=None, grad_ckpt=False, edge0=None, ncv=None, rel_feat=None, cs=None,
                   lag_plan=None, ring_idx=None, ring_oks=None, cut_late=None, cut_lag=None):
        """step with a graph: nothing is written into K and V (the graph holds them as they are), the decisions since the
        graph began are in `late`, a list of their keys and values (b*N,h,d/h) with their own graphs, lives (b,) bool
        each: which plants they are decisions of; this one's are put last. see (b,S): the slots a plant attends to.
        lag_plan/ring_idx/ring_oks (g1/g2): lag indexing resolved inside checkpointed body (zero saved ring-lag copies);
        lagkv: legacy fallback list of (k, v, ok) per lag.
        cut_late/cut_lag: optional causal gradient cutoff masks from Room.cut_pos (applied inside checkpointed core).
        grad_ckpt: checkpoint the layer body (passes existing late tensor refs without O(H^2) copies)."""
        dh = x.shape[-1]//self.h
        cos_p, sin_p = rotary_cs(dh, pos, 1, x.device, x.dtype) if cs is None else cs
        n_late = len(late)
        use_plan = lag_plan if self.lags else None
        n_lags = 0 if use_plan is not None else (len(lagkv) if self.lags and lagkv is not None else 0)
        tensors = tuple(kj for kj, _ in late)+tuple(vj for _, vj in late)+tuple(lives)
        if n_lags:
            tensors = tensors+tuple(kj for kj, _, _ in lagkv)+tuple(vj for _, vj, _ in lagkv)+tuple(oj for _, _, oj in lagkv)
        if grad_ckpt and torch.is_grad_enabled():
            x, k, v = checkpoint(self._step_graph_body, x, on, bias, K, V, see, cos_p, sin_p, n_late, n_lags, ncv, edge0, rel_feat,
                                 use_plan, ring_idx, ring_oks, cut_late, cut_lag, *tensors, use_reentrant=False)
        else:
            x, k, v = self._step_graph_body(x, on, bias, K, V, see, cos_p, sin_p, n_late, n_lags, ncv, edge0, rel_feat,
                                            use_plan, ring_idx, ring_oks, cut_late, cut_lag, *tensors)
        late.append((k, v))
        return x

    def across(self, x, on, bias, edge0=None, ncv=None, rel_feat=None):
        """The channel attention and the feed-forward of forward, on what the time attention gives."""
        b, T, N, d = x.shape
        h = self.h
        # ---- channels: the tokens of a decision over each other
        q, k, v = self.qkv_c(self.n2(x)).reshape(b*T, N, 3, h, d//h).permute(2, 0, 3, 1, 4)                    # (b*T,h,N,d/h) each
        q, k = self.norm_qk(q, k, 'c')
        m = (bias.reshape(b*T, 1, N, N)*self.pair[None, :, None, None]).to(x.dtype)
        off_b = None
        if self.edge and edge0 is not None:
            e_in = edge0[:, None] if edge0.dim() == 4 else edge0
            off_b = self.edge_bias(e_in.to(x.dtype)).permute(0, 1, 4, 2, 3).reshape(b*T, h, ncv, N-ncv)
        if self.rel and rel_feat is not None:
            r_in = rel_feat[:, None] if rel_feat.dim() == 4 else rel_feat
            rb = self.rel_bias(r_in.to(x.dtype)).permute(0, 1, 4, 2, 3).reshape(b*T, h, ncv, N-ncv)
            off_b = rb if off_b is None else off_b+rb
        if off_b is not None:
            mc = torch.cat((m[:, :, :ncv, :ncv], m[:, :, :ncv, ncv:]+off_b), 3)
            mm = torch.cat((m[:, :, ncv:, :ncv]+off_b.transpose(-1, -2), m[:, :, ncv:, ncv:]), 3)
            m = torch.cat((mc, mm), 2)
        m = m.masked_fill(~(on.reshape(b*T, 1, 1, N) > 0), float('-inf'))                                      # nothing attends to a token out of service
        m = torch.where(torch.isinf(m).all(-1, keepdim=True), torch.zeros_like(m), m)                         # (a decision with none in service: no NaN)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        x = x+self.out_c(y.transpose(1, 2).reshape(b, T, N, d))*on[..., None]
        return x+self.ff(self.n3(x))*on[..., None]


class Tokens(nn.Module):
    def __init__(self, d=D, layers=LAYERS, heads=HEADS, knots=5, mem=0, lags=0, qk=0, mup=0, fb=0, edge=0, rel=0):
        super().__init__()
        self.d, self.nk, self.nmem, self.mup, self.fb, self.edge, self.rel = d, knots, mem, bool(mup), bool(fb), int(edge), int(rel)
        self.row, self.mv = nn.Linear(CF, d), nn.Linear(MF+1, d)
        self.layers = nn.ModuleList([Layer(d, heads, qk, lags, edge, rel) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.read = nn.Sequential(nn.Linear(2*d+1, 2*d), nn.SiLU(), nn.Linear(2*d, knots))
        self.act = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        nn.init.zeros_(self.act[-1].weight)
        nn.init.zeros_(self.act[-1].bias)
        self.when = nn.Linear(d, 1)                               # --probe: "this unit has not been disturbed yet" (aio.net.probing), per MV; 0 at its start
        nn.init.zeros_(self.when.weight)
        nn.init.zeros_(self.when.bias)
        if mem:                                                   # mK: a memory per token that forgets over 4 to 2048 decisions (a = 1-1/tau), read by the heads
            self.mem_n, self.mem_in, self.mem_out = nn.LayerNorm(d), nn.Linear(d, mem), nn.Linear(mem, d)
            self.mem_a = nn.Parameter(torch.log(torch.logspace(math.log10(4.), math.log10(2048.), mem)-1.))
            nn.init.zeros_(self.mem_out.weight)
            nn.init.zeros_(self.mem_out.bias)
        if self.fb:                                               # f1: feed persistent token memory back into Layer 0 (zero-init: exact step-0 identity)
            self.mem_fb = nn.Linear(mem, d)
            nn.init.zeros_(self.mem_fb.weight)
            nn.init.zeros_(self.mem_fb.bias)
        if self.edge:                                             # eK: persistent pairwise bilinear cross-correlator (CV i, MV j) over 8..2048 decisions
            self.edge_rn, self.edge_mn = nn.LayerNorm(d), nn.LayerNorm(d)
            self.edge_r, self.edge_m = nn.Linear(d, self.edge), nn.Linear(d, self.edge)
            self.edge_a = nn.Parameter(torch.log(torch.logspace(math.log10(8.), math.log10(2048.), self.edge)-1.))
        if self.rel:                                              # r1/r2: multi-scale Erlang-8 SSM + symmetric Gated DeltaNet + 1-hop edge-conditioned message passing
            rf_dim = rel_feat_dim(self.rel)
            self.rel_p = nn.Parameter(torch.log(torch.expm1(torch.tensor(REL_TAUS, dtype=torch.float32))))
            self.rel_a = nn.Parameter(torch.tensor(5.5))          # sigmoid(5.5) ~= 0.996 decay
            self.rel_eta = nn.Parameter(torch.tensor(0.0))        # sigmoid(0.0) = 0.5 Delta-rule step size
            self.rel_c = nn.Parameter(torch.tensor(math.log(math.expm1(8.))))   # softplus ~= 8.0 NLMS damping
            self.rel_rn, self.rel_mn = nn.LayerNorm(d), nn.LayerNorm(d)
            self.rel_e_mv, self.rel_c_in, self.rel_mv_out = nn.Linear(rf_dim, REL_MP), nn.Linear(d, REL_MP), nn.Linear(REL_MP, d)
            self.rel_e_cv, self.rel_m_in, self.rel_cv_out = nn.Linear(rf_dim, REL_MP), nn.Linear(d, REL_MP), nn.Linear(REL_MP, d)
            for lin in (self.rel_mv_out, self.rel_cv_out):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
            if self.rel >= 2:                                     # r2: 8-scale Neural Smith Predictor in-flight pipe displacement + log-gain precision gate (zero-init identity)
                self.rel_pipe_in = nn.Linear(REL_E, d)
                self.act_gate = nn.Linear(d, 1)
                for lin in (self.rel_pipe_in, self.act_gate):
                    nn.init.zeros_(lin.weight)
                    nn.init.zeros_(lin.bias)
        self.mem_last = None                                      # (the memory after the last decision)
        self.edge_last = None                                     # (the pairwise edge memory after the last decision)
        self.rel_last = None                                      # (r1/r2 flat DeltaNet + SSM state after the last decision)
        self.rel_feat_last = None                                 # (r1/r2 (b,ncv,nmv,rf_dim) pairwise features after the last decision)
        self.rel_pipe_last = None                                 # (r2 (b,nmv,16) multi-scale in-flight pipe displacement & impulse wave)
        self.act_scale_last = None                                # (r2 (b,T,nmv) multiplicative log-gain precision gate scale)
        self.x_norm_last = None                                   # (mean active residual norm before final LayerNorm, on device)
        self.grad_ckpt = False                                    # (checkpoint each layer's body inside step_graph: O(H) VRAM)

    def load_state_dict(self, sd, strict=True, **kw):
        """Weights from before `when` take it at its zero start."""
        own = self.state_dict()
        return super().load_state_dict({**{q: own[q] for q in own if q.startswith('when.') and q not in sd}, **sd}, strict, **kw)

    def tokens(self, obs, previous, seen=None, ahead=0, mem0=None):
        """obs (b,T,SIZE) packets in the frame of aio.frame.canonical, previous (b,T,NMV) the last actions,
        seen: as Layer.forward -> rows' tokens (b,T,NCV,d), MVs' (b,T,NMV,d), the pairing (b,T,NCV,NMV).
        ahead: the first so many decisions are only there to be attended to (seen is given): the same tokens of the
        others, no graph through those and none returned of them. mem0 (b,N,mem): the memory before the first
        decision returned (None: empty); the one after the last is left in `mem_last`."""
        if ahead:                                                 # the decisions ahead: no graph, their keys and values to the others
            with torch.no_grad():
                x, on, bias, _, _ = self.embed(obs[:, :ahead], previous[:, :ahead])
                kept = []
                for layer in self.layers:
                    x = layer(x, on, bias, seen[:, :ahead, :ahead], keep=kept)
            x, on, bias, link, ncv = self.embed(obs[:, ahead:], previous[:, ahead:])
            for layer, kv in zip(self.layers, kept):
                x = layer(x, on, bias, seen[:, ahead:], before=kv, start=ahead)
        else:
            x, on, bias, link, ncv = self.embed(obs, previous)
            for layer in self.layers:
                x = layer(x, on, bias, seen)
        self.x_norm_last = (x.detach().float().norm(dim=-1)*on.float()).sum()/on.float().sum().clamp_min(1.)
        x = self.remember(self.norm(x), on, mem0)
        return x[:, :, :ncv], x[:, :, ncv:], link

    def _remember_step(self, x, on, mem0):
        """Single-step mK recurrence core (compilable via torch.compile, zero Python loop)."""
        b, _, N, _ = x.shape
        a = torch.sigmoid(self.mem_a.float())
        u = self.mem_in(self.mem_n(x)).float()*(1-a)
        m = x.new_zeros(b, N, self.nmem, dtype=torch.float32) if mem0 is None else mem0.float()
        m = torch.where(on[:, 0, :, None] > 0, a*m+u[:, 0], m)
        return x+self.mem_out(m[:, None].to(x.dtype))*on[..., None], m

    def remember(self, x, on, mem0):
        """mK: the tokens after the layers with the memory read in, the memory carried on decision by decision."""
        if not self.nmem:
            return x
        if x.shape[1] == 1:
            out, m = self._remember_step(x, on, mem0)
            self.mem_last = m
            return out
        b, T, N, _ = x.shape
        a = torch.sigmoid(self.mem_a.float())
        u = self.mem_in(self.mem_n(x)).float()*(1-a)
        m = x.new_zeros(b, N, self.nmem, dtype=torch.float32) if mem0 is None else mem0.float()
        ms = []
        for t in range(T):
            m = torch.where(on[:, t, :, None] > 0, a*m+u[:, t], m)
            ms.append(m)
        self.mem_last = m
        return x+self.mem_out(torch.stack(ms, 1).to(x.dtype))*on[..., None]

    def update_edge(self, r, m, on, ncv, edge0):
        """eK: update persistent pairwise bilinear cross-correlator E_ij(t) = a*E_ij(t-1) + (1-a)*tanh(r_e)*tanh(m_e)."""
        if not self.edge:
            return
        b, T, _, _ = r.shape
        nmv = m.shape[2]
        a = torch.sigmoid(self.edge_a.float())
        re = torch.tanh(self.edge_r(self.edge_rn(r)))                                         # (b,T,ncv,edge)
        me = torch.tanh(self.edge_m(self.edge_mn(m)))                                         # (b,T,nmv,edge)
        u = (re[:, :, :, None, :]*me[:, :, None, :, :]).float()*(1.-a)                        # (b,T,ncv,nmv,edge)
        e = r.new_zeros(b, ncv, nmv, self.edge, dtype=torch.float32) if edge0 is None else edge0.float()
        for t in range(T):
            pair_on = (on[:, t, :ncv, None] > 0) & (on[:, t, None, ncv:] > 0)
            e = torch.where(pair_on[..., None], a*e+u[:, t], e)
        self.edge_last = e

    def _update_rel_core(self, cv, mv, s, ncv, nmv):
        """Pure functional float32 Erlang-8 + Gated DeltaNet core (compilable via torch.compile with zero graph breaks)."""
        b = cv.shape[0]
        n_mv_s = nmv*(2*REL_K+2+REL_E)
        n_cv_s = ncv*2
        mv_s = s[:, :n_mv_s].reshape(b, nmv, 2*REL_K+2+REL_E)
        cv_s = s[:, n_mv_s:n_mv_s+n_cv_s].reshape(b, ncv, 2)
        E = s[:, n_mv_s+n_cv_s:].reshape(b, ncv, nmv, REL_E)
        z_du, z_u = mv_s[..., :REL_K], mv_s[..., REL_K:2*REL_K]
        u_ema, u_init = mv_s[..., 2*REL_K], mv_s[..., 2*REL_K+1]
        k_var = mv_s[..., 2*REL_K+2:]
        dy_s, v_var = cv_s[..., 0], cv_s[..., 1]

        cv_on = cv[..., 11] > .5
        cv_kn = (cv[..., 7] > .5) & cv_on
        mv_on = mv[..., 5] > .5
        dy = cv[..., 1]*cv_kn.float()
        u = mv[..., 0]*mv_on.float()
        du = mv[..., 1]*mv_on.float()

        # Causal slow valve baseline u_ema (initialized cleanly on first active step so u_dev=0 at t=0)
        u_ema_n = torch.where(mv_on, torch.where(u_init > .5, .985*u_ema+.015*u, u), u_ema)
        u_init_n = torch.where(mv_on, torch.ones_like(u_init), u_init)
        u_dev = (u-u_ema_n)*mv_on.float()

        # 8-stage strictly-positive Erlang/Gamma IIR cascade with learnable poles
        taus = F.softplus(self.rel_p.float()).clamp_min(.5)
        alphas = torch.exp(-1./taus)
        xd, xu = du, u_dev
        zdn, zun = [], []
        for m in range(REL_K):
            a = alphas[m]
            zd = a*z_du[..., m]+(1.-a)*xd
            zu = a*z_u[..., m]+(1.-a)*xu
            zdn.append(zd); zun.append(zu)
            xd, xu = zd, zu
        z_du_n = torch.where(mv_on[..., None], torch.stack(zdn, -1), z_du)
        z_u_n = torch.where(mv_on[..., None], torch.stack(zun, -1), z_u)
        kt = torch.cat((z_du_n, z_u_n), -1)                                                       # (b,nmv,16)
        k_var_n = torch.where(mv_on[..., None], (.99*k_var+.01*kt.square()).clamp_min(0.), k_var)
        kn = (kt/(k_var_n+1e-4).sqrt())*mv_on[..., None].float()

        # Symmetric CV normalization (updated ONLY when cv_kn so analyzer CVs never decay between samples)
        dy_s_n = torch.where(cv_kn, .5*dy_s+.5*dy, dy_s)
        v_var_n = torch.where(cv_kn, (.99*v_var+.01*dy_s_n.square()).clamp_min(0.), v_var)
        vn = (dy_s_n/(v_var_n+1e-4).sqrt())*cv_kn.float()

        # Multi-scale Gated DeltaNet fast-weight update (masked by cv_kn & mv_on)
        dec = torch.sigmoid(self.rel_a.float())
        eta = torch.sigmoid(self.rel_eta.float())
        damp = F.softplus(self.rel_c.float()).clamp_min(1.)
        kn2 = kn.square().sum(dim=(1, 2), keepdim=True).unsqueeze(1)+damp                         # (b,1,1,1)
        v_hat = (E*kn[:, None, :, :]).sum(dim=(2, 3))                                             # (b,ncv)
        err = (vn-v_hat).clamp(-10., 10.)
        upd = cv_kn[:, :, None] & mv_on[:, None, :]
        E_upd = dec*E+eta*(err[:, :, None, None]*kn[:, None, :, :])/kn2
        E_n = torch.where(upd[..., None], E_upd, E)

        mv_out = torch.cat((z_du_n, z_u_n, u_ema_n[..., None], u_init_n[..., None], k_var_n), -1).reshape(b, n_mv_s)
        cv_out = torch.stack((dy_s_n, v_var_n), -1).reshape(b, n_cv_s)
        rel_out = torch.cat((mv_out, cv_out, E_n.reshape(b, ncv*nmv*REL_E)), -1)

        # 20-dim scale-bounded pairwise feature tensor (exact odd sign symmetry in E channels + excitation energy)
        rf = torch.cat((
            torch.asinh(2.*E_n),
            torch.asinh(2.*E_n[..., :REL_K].sum(-1, keepdim=True)),
            torch.asinh(2.*E_n[..., REL_K:].sum(-1, keepdim=True)),
            torch.asinh(2.*E_n.abs().sum(-1, keepdim=True)),
            torch.log1p(k_var_n.sum(-1, keepdim=True).clamp_min(0.))[:, None].expand(-1, ncv, -1, -1),
        ), -1)
        pipe = None
        if self.rel >= 2:
            # r2: +4 strictly-even dimensionally-exact physical gain & activity features (zero extra state bytes)
            slew_obs = mv[..., 2].clamp_min(1e-4)                                                 # normalized full-rate step (2*slew/span)
            ratio = ((v_var_n[:, :, None, None]+1e-4)/(k_var_n[:, None, :, :]+1e-4)).sqrt()
            H_phys = E_n*ratio                                                                    # exact unnormalized per-stage filter weights
            k_imp_abs = H_phys[..., :REL_K].abs().sum(-1)                                         # steady-state step gain |K_norm| (bands/span)
            k_lev_abs = H_phys[..., REL_K:].abs().sum(-1)                                         # integrator gain |K_int| (bands/step/span)
            k_slew = torch.asinh(k_imp_abs*slew_obs[:, None, :])                                  # bands per full-rate step |K*slew|
            k_norm = torch.asinh(k_imp_abs)                                                       # steady-state gain in asinh bands
            k_int_slew = torch.asinh(k_lev_abs*slew_obs[:, None, :])                              # level-rate bands/step per full-rate step
            cv_act = torch.asinh(4.*v_var_n.clamp_min(0.).sqrt())[:, :, None].expand(-1, -1, nmv)
            rf = torch.cat((rf, k_slew[..., None], k_norm[..., None], k_int_slew[..., None], cv_act[..., None]), -1)
            # r2: 8-scale slew-normalized in-flight pipe displacement + impulse wave for Neural Smith Predictor
            s_div = mv[..., 2].clamp_min(1e-3)[..., None]
            pipe = torch.cat((
                torch.asinh((u_dev[..., None]-z_u_n)/s_div),
                torch.asinh(4.*z_du_n/s_div),
            ), -1)*mv_on[..., None].float()
        return rel_out, rf, pipe

    def update_rel(self, obs, rel0):
        """r1: causal strictly-positive Erlang-8 SSM + symmetric-normalized Gated DeltaNet fast-weight step (float32).
        obs: (b, 1, SIZE), rel0: (b, rel_state_size(ncv, nmv)) or None -> rel_feat (b, ncv, nmv, REL_F)."""
        if not self.rel:
            return None
        dev_type = obs.device.type
        ac = torch.autocast(dev_type, enabled=False) if torch.is_autocast_enabled(dev_type) else contextlib.nullcontext()
        with ac:
            cv, mv, _ = unpack(obs[:, 0].float())
            b, ncv, nmv = cv.shape[0], cv.shape[1], mv.shape[1]
            st_len = rel_state_size(ncv, nmv)
            s = cv.new_zeros(b, st_len, dtype=torch.float32) if rel0 is None else rel0.float()
            rel_out, rf, pipe = self._update_rel_core(cv, mv, s, ncv, nmv)
        self.rel_last = rel_out
        if pipe is not None:
            self.rel_pipe_last = pipe
        self.rel_feat_last = rf
        return rf

    def _rel_cv_core(self, x, on, ncv, rel_feat, rel_pipe):
        on_c, on_m = on[:, :, :ncv], on[:, :, ncv:]
        x_m = x[:, :, ncv:]
        if self.rel >= 2 and rel_pipe is not None:
            x_m = x_m+self.rel_pipe_in(rel_pipe[:, None].to(x.dtype))*on_m[..., None]
        rf = rel_feat[:, None].to(x.dtype)
        m_msg = self.rel_m_in(self.rel_mn(x_m))*on_m[..., None]
        agg_cv = (self.rel_e_cv(rf)*m_msg[:, :, None, :, :]).sum(3)/on_m.float().sum(-1, keepdim=True).clamp_min(1.).sqrt().to(x.dtype)[..., None]
        dx_c = self.rel_cv_out(agg_cv)*on_c[..., None]
        return torch.cat((x[:, :, :ncv]+dx_c, x_m), 2)

    def rel_cv_message(self, x, on, ncv, rel_feat, rel_pipe=None):
        """r1/r2: Layer-0 in-flight pipe injection (r2) + edge-conditioned CV<-MV predictor message (zero-init output; feeds all layers)."""
        if not self.rel or rel_feat is None:
            return x
        return self._rel_cv_core(x, on, ncv, rel_feat, rel_pipe)

    def _rel_mv_core(self, r, m, on, ncv, rel_feat):
        on_c, on_m = on[:, :, :ncv], on[:, :, ncv:]
        rf = rel_feat[:, None].to(r.dtype)
        c_msg = self.rel_c_in(self.rel_rn(r))*on_c[..., None]
        agg_mv = (self.rel_e_mv(rf)*c_msg[:, :, :, None, :]).sum(2)/on_c.float().sum(-1, keepdim=True).clamp_min(1.).sqrt().to(r.dtype)[..., None]
        return m+self.rel_mv_out(agg_mv)*on_m[..., None]

    def rel_mv_message(self, r, m, on, ncv, rel_feat):
        """r1: Post-layer 1-hop edge-conditioned MV<-CV decoupler message directly into `act` (zero-init output)."""
        if not self.rel or rel_feat is None:
            return m
        return self._rel_mv_core(r, m, on, ncv, rel_feat)

    def _embed_core(self, cv, mv, link, previous):
        ncv, nmv = cv.shape[-2], mv.shape[-2]
        cm, mm = cv[..., -1:], mv[..., -1:]
        x = torch.cat((self.row(symlog(cv*cm)), self.mv(torch.cat((symlog(mv*mm), previous[..., None]*mm), -1))), 2)
        on = torch.cat((cm[..., 0], mm[..., 0]), 2).to(x.dtype)
        x = x*on[..., None]
        link_d = link.to(x.dtype)
        z_cc = x.new_zeros(*x.shape[:2], ncv, ncv)
        z_mm = x.new_zeros(*x.shape[:2], nmv, nmv)
        bias = torch.cat((torch.cat((z_cc, link_d), -1), torch.cat((link_d.transpose(-1, -2), z_mm), -1)), -2)
        return x, on, bias

    def embed(self, obs, previous):
        """-> the tokens before the layers (b,T,N,d), which are in service (b,T,N), the pairing among them (b,T,N,N)
        and as published (b,T,NCV,NMV), the number of rows."""
        cv, mv, link = unpack(obs)
        ncv = cv.shape[-2]
        x, on, bias = self._embed_core(cv, mv, link, previous)
        return x, on, bias, link, ncv

    def forward(self, obs, previous, seen=None, ahead=0, mem0=None, need_read=True):
        """-> the responses read per pair (b,T,NCV,NMV,knots) or None if need_read=False, the moves' means per MV (b,T,NMV),
        the logit of `when` per MV (b,T,NMV) (ahead: of the decisions after those)."""
        return self.heads(*self.tokens(obs, previous, seen, ahead, mem0), need_read=need_read)

    def enable_compile(self, dynamic=True, backend='inductor'):
        """Compile fixed-arity per-layer step cores, Erlang/DeltaNet update, and outer hot step helpers via torch.compile (zero `_orig_mod.` keys in state_dict)."""
        if getattr(self, '_compiled', False):
            return self
        torch._dynamo.config.cache_size_limit = max(getattr(torch._dynamo.config, 'cache_size_limit', 8), 64)
        self._embed_core = torch.compile(self._embed_core, dynamic=dynamic, backend=backend)
        if self.rel:
            self._update_rel_core = torch.compile(self._update_rel_core, dynamic=dynamic, backend=backend)
            self._rel_cv_core = torch.compile(self._rel_cv_core, dynamic=dynamic, backend=backend)
            self._rel_mv_core = torch.compile(self._rel_mv_core, dynamic=dynamic, backend=backend)
        if self.nmem:
            self._remember_step = torch.compile(self._remember_step, dynamic=dynamic, backend=backend)
        self._heads_act = torch.compile(self._heads_act, dynamic=dynamic, backend=backend)
        self._heads_read = torch.compile(self._heads_read, dynamic=dynamic, backend=backend)
        for layer in self.layers:
            layer._step_core_late = torch.compile(layer._step_core_late, dynamic=dynamic, backend=backend)
            layer._step_infer_core = torch.compile(layer._step_infer_core, dynamic=dynamic, backend=backend)
            layer._step_infer_post = torch.compile(layer._step_infer_post, dynamic=dynamic, backend=backend)
        self._compiled = True
        return self

    def step(self, obs, previous, room, mem0=None, edge0=None, rel0=None, need_read=True):
        """forward for one new decision (b,1,...) on the keys and values kept in `room` (Room), where its own are put."""
        rel_feat = self.update_rel(obs, rel0) if self.rel else None
        x, on, bias, link, ncv = self.embed(obs, previous)
        if self.fb and mem0 is not None:
            x = x+self.mem_fb(mem0.to(x.dtype))[:, None]*on[..., None]
        if self.rel:
            x = self.rel_cv_message(x, on, ncv, rel_feat, self.rel_pipe_last if self.rel >= 2 else None)
        at = room.at
        room.there[:, at] = True
        dh = self.d//self.layers[0].h
        cs = rotary_cs(dh, room.pos, 1, x.device, x.dtype)
        lag_idx = lag_ok = None
        if self.layers[0].lags:
            S = room.there.shape[1]
            taus = _lag_taus_cached(self.layers[0].lag_list, x.device)
            lag_idx = (at-taus) % S
            lag_ok = (room.there.index_select(1, lag_idx) & (room.where.index_select(0, lag_idx) == (room.pos-taus))[None])[:, None]
        for i, layer in enumerate(self.layers):
            x = layer.step(x, on, bias, room.K[i], room.V[i], room.there, at, room.pos, room.where, edge0, ncv, rel_feat, cs, lag_idx, lag_ok)
        self.x_norm_last = (x.detach().float().norm(dim=-1)*on.float()).sum()/on.float().sum().clamp_min(1.)
        x = self.remember(self.norm(x), on, mem0)
        r, m = x[:, :, :ncv], x[:, :, ncv:]
        if self.edge:
            self.update_edge(r, m, on, ncv, edge0)
        if self.rel:
            m = self.rel_mv_message(r, m, on, ncv, rel_feat)
        return self.heads(r, m, link, need_read=need_read)

    def step_graph(self, obs, previous, room, mem0=None, edge0=None, rel0=None, need_read=True):
        """step with a graph through the decisions since the graph began (Room.late): what the plant's gradient goes through."""
        rel_feat = self.update_rel(obs, rel0) if self.rel else None
        x, on, bias, link, ncv = self.embed(obs, previous)
        if self.fb and mem0 is not None:
            x = x+self.mem_fb(mem0.to(x.dtype))[:, None]*on[..., None]
        if self.rel:
            x = self.rel_cv_message(x, on, ncv, rel_feat, self.rel_pipe_last if self.rel >= 2 else None)
        w = room.there.shape[1]-1
        S = room.there.shape[1]
        see = room.there & (room.pos-room.where <= w)[None]
        ok = [room.pos-p <= w for p in room.late_pos]
        lives = [l for l, o in zip(room.late_live, ok) if o]
        lp = [p for p, o in zip(room.late_pos, ok) if o]
        last = (room.at-1) % S                                    # the ring's newest slot (nothing is written into it while a graph is made)
        dh = self.d//self.layers[0].h
        cs = rotary_cs(dh, room.pos, 1, x.device, x.dtype)
        # Hoist lag slot indexing ONCE outside the layer loop on GPU (zero H2D copies, ring lookup deferred into checkpointed body)
        lag_plan = None
        ring_idx = ring_oks = None
        has_late_lag = False
        if self.layers[0].lags:
            ll = self.layers[0].lag_list
            taus_gpu = _lag_taus_cached(ll, x.device)
            plan, ring_pos_idx = [], []
            for j_l, tau in enumerate(ll):
                p = room.pos-tau
                if lp and p >= lp[0]:
                    plan.append((True, p-lp[0]))
                    has_late_lag = True
                else:
                    plan.append((False, len(ring_pos_idx)))
                    ring_pos_idx.append(j_l)
            lag_plan = tuple(plan)
            if ring_pos_idx:
                if ring_pos_idx[0] + len(ring_pos_idx) == len(ll):
                    taus_ring = taus_gpu[ring_pos_idx[0]:]
                else:
                    taus_ring = taus_gpu[ring_pos_idx]
                rp_t = room.pos-taus_ring
                ring_idx = (last-(room.where[last]-rp_t)) % S
                ring_oks = room.there.index_select(1, ring_idx) & (room.where.index_select(0, ring_idx) == rp_t)[None]
        cut_late = cut_lag = None
        if room.cut_pos is not None:
            if lp:
                cut_late = _pos_tensor(lp, x.device)[None, :] < room.cut_pos[:, None]
            if has_late_lag:
                cut_lag = (room.pos-_lag_taus_cached(self.layers[0].lag_list, x.device))[None, :] < room.cut_pos[:, None]
        for i, layer in enumerate(self.layers):
            late = [kv for kv, o in zip(room.late[i], ok) if o]
            x = layer.step_graph(x, on, bias, room.K[i], room.V[i], see, late, lives, room.pos, None, self.grad_ckpt,
                                 edge0, ncv, rel_feat, cs, lag_plan, ring_idx, ring_oks, cut_late, cut_lag)
            room.late[i].append(late[-1])
        self.x_norm_last = (x.detach().float().norm(dim=-1)*on.float()).sum()/on.float().sum().clamp_min(1.)
        x = self.remember(self.norm(x), on, mem0)
        r, m = x[:, :, :ncv], x[:, :, ncv:]
        if self.edge:
            self.update_edge(r, m, on, ncv, edge0)
        if self.rel:
            m = self.rel_mv_message(r, m, on, ncv, rel_feat)
        return self.heads(r, m, link, need_read=need_read)

    def _heads_act(self, m):
        act = self.act(m)[..., 0]
        if self.mup:
            act = act/math.sqrt(self.d)
        scale = None
        if self.rel >= 2:
            scale = torch.exp(self.act_gate(m)[..., 0].clamp(-5., .5))
            act = act*scale
        return act, scale

    def _heads_read(self, r, m, link):
        ncv, nmv = r.shape[2], m.shape[2]
        pair = torch.cat((r[:, :, :, None].expand(-1, -1, -1, nmv, -1), m[:, :, None].expand(-1, -1, ncv, -1, -1), link[..., None].to(r.dtype)), -1)
        return self.read(pair)

    def heads(self, r, m, link, need_read=True):
        act, scale = self._heads_act(m)
        if scale is not None:
            self.act_scale_last = scale
        when = self.when(m.detach())[..., 0]
        lab = self._heads_read(r, m, link) if need_read else None
        return lab, act, when


# ---- the same network behind the interface of aio.net.AIO (aio.train, aio.net.Policy): `steps` with a state

IDLE = 4096                               # decisions (of any batch) after which what is kept of a batch that made none is dropped
W = 128                                   # decisions of history a decision sees (kept in the state as they came: packets and last actions)


def ctx_size(ncv, nmv, w=W, mem=0, edge=0, rel=0):
    """Length of the history kept for packets of ncv rows and nmv MVs: w decisions of packet and last actions, how many
    of them are there, (mK) the memory of every token, (eK) edge memory, and (r1) multi-scale DeltaNet state."""
    from aio.interface import size
    return w*(size(ncv, nmv)+nmv)+1+(ncv+nmv)*mem+ncv*nmv*edge+(rel_state_size(ncv, nmv) if rel else 0)


class Room:
    """What the time attention needs of the last decisions of a batch of plants, kept so that a new decision is the only
    one computed: per layer the keys (turned to their positions) and values of the w+1 last, in a ring (bfloat16
    on a card: l8, d256, w128 take 1 MB per token, 13.8 GB for 768 plants of 12x5). Not part of the state: `raw` and
    `n` say which state it belongs to, and a state it does not belong to has it built again from the packets it holds."""

    def __init__(self, layers, b, N, h, S, dh, dev):
        dt = torch.bfloat16 if dev.type == 'cuda' else torch.float32                       # (not float16: past 65504 a key turns inf, the moves NaN; sq-big-s1 11-20)
        self.K = [torch.zeros(b*N, h, S, dh, device=dev, dtype=dt) for _ in range(layers)]
        self.V = [torch.zeros(b*N, h, S, dh, device=dev, dtype=dt) for _ in range(layers)]
        self.there = torch.zeros(b, S, dtype=torch.bool, device=dev)
        self.where = torch.full((S,), -10**9, dtype=torch.long, device=dev)                  # the position of the decision in a slot
        self.at, self.pos, self.raw, self.n, self.used = 0, 0, None, None, 0
        self.read = False                                                                   # a graph was made on the ring since it was last written
        self.late, self.late_pos, self.late_live = [[] for _ in range(layers)], [], []       # the decisions made with a graph (Layer.step_graph): in the ring once it is done with (`settle`)
        self.cut_pos = None                                                                 # (b,) int64 latest cut pos per env (None: no cut since last settle)
        self.lane_tag = None                                                                # optional caller lane identity for zero-sync room matching
        self._last_h_ref, self._last_h_ver = None, -1                                       # weakref & _version of last hidden tensor returned for this room

    def on(self, live=None):
        """The decision just written is the newest one; live (b,): the plants it is a decision of (None: all)."""
        if live is not None:
            self.there[:, self.at] = live
        self.where[self.at] = self.pos
        self.at, self.pos = (self.at+1) % self.there.shape[1], self.pos+1

    def on_graph(self):
        """The decision just made with a graph (in `late`) is the newest one."""
        self.late_pos.append(self.pos)
        self.read = True
        self.late_live.append(torch.ones_like(self.there[:, 0]))
        self.pos += 1

    def settle(self, inplace=False):
        """The decisions made with a graph, into the ring without it: when a state comes that carries no graph.
        inplace: True when called after `loss.backward()` so no active autograd graph references K/V (avoids full-ring clone)."""
        S = self.there.shape[1]
        m = len(self.late_pos)
        if not m:
            self.read = False
            self.cut_pos = None
            return
        if self.read and not inplace:                             # a graph may still hold the ring as it was (a window's backward pass comes after the next one began): written in a copy
            self.K, self.V = [K.clone() for K in self.K], [V.clone() for V in self.V]
        self.read = False
        if m > S:
            skip = m-S
            self.at = (self.at+skip) % S
            late_pos, late_live = self.late_pos[skip:], self.late_live[skip:]
            lates = [late[skip:] for late in self.late]
            m = S
        else:
            late_pos, late_live, lates = self.late_pos, self.late_live, self.late
        end = min(self.at+m, S)
        n1 = end-self.at
        for K, V, late in zip(self.K, self.V, lates):
            kl = torch.stack([kv[0].detach().to(K.dtype) for kv in late], 2)
            vl = torch.stack([kv[1].detach().to(V.dtype) for kv in late], 2)
            K[:, :, self.at:end] = kl[:, :, :n1]
            V[:, :, self.at:end] = vl[:, :, :n1]
            if n1 < m:
                K[:, :, :m-n1] = kl[:, :, n1:]
                V[:, :, :m-n1] = vl[:, :, n1:]
        lv = torch.stack(late_live, 1)
        ps = _pos_tensor(late_pos, self.where.device)
        self.there[:, self.at:end] = lv[:, :n1]
        self.where[self.at:end] = ps[:n1]
        if n1 < m:
            self.there[:, :m-n1] = lv[:, n1:]
            self.where[:m-n1] = ps[n1:]
        self.at = (self.at+m) % S
        self.late, self.late_pos, self.late_live, self.cut_pos = [[] for _ in self.K], [], [], None


class Held(nn.Module):
    """What aio.train asks of a network's pair pathway, for one that has none: sizes, and that nothing of the law, the
    running sums or the learned reader is there. `seq` says the network's spec to whoever loads the weights."""
    law, stats, fm, unseen, given, apart, read_law, diff = False, None, None, None, False, False, None, ''
    reader = True                                                 # (aio.train.imitate: the memory is in the actor's state, where a learned reader's is)

    def __init__(self, sp, w, knots):
        super().__init__()
        extra = [sp[q] for q in FIELDS[3:]]
        while extra and not extra[-1]:
            extra.pop()
        self.register_buffer('seq', torch.tensor([sp['d'], sp['layers'], sp['heads'], w]+extra))
        self.w, self.nk, self.mem, self.edge, self.rel = w, knots, sp['mem'], sp['edge'], sp['rel']

    def size_of(self, ncv=None, nmv=None):
        from aio import net as N
        ncv, nmv = (N.NCV, N.NMV) if ncv is None else (ncv, nmv)
        return N.sizes(ncv, nmv)[7]+ctx_size(ncv, nmv, self.w, self.mem, self.edge, self.rel)


def of_buffer(seq):
    """Held.seq -> the spec string and w."""
    v = [int(x) for x in seq]
    sp = dict(zip(FIELDS, v[:3]+v[4:]+[0]*(len(FIELDS)-3-len(v[4:]))))
    return said(sp), v[3]


class SeqAIO(nn.Module):
    """aio.seqnet.Tokens as aio.net.AIO is used: `steps` over sequences or single decisions, the last w decisions kept in
    the state (behind the room of AIO's own memories, which stays zero: the trainer's layout of a state is AIO's)."""
    canonical = True
    supports_fast_steps = True

    def __init__(self, asked=True, w=W, knots=None):
        super().__init__()
        from aio import net as N
        sp = spec(asked)
        self.actor = Tokens(knots=N.NLAB if knots is None else knots, **sp)
        self.pairs = Held(sp, w, self.actor.nk)
        self.w, self.heads, self.mem, self.fb, self.edge, self.rel = w, sp['heads'], sp['mem'], bool(sp['fb']), int(sp['edge']), int(sp['rel'])
        self.peeking, self.last_room, self.built = False, None, 0     # (`peek`: a decision that is not kept; the Room of the last decision; Rooms built from packets)
        self.amp = os.environ.get('AIO_SEQ_AMP', '0') == '1'                                  # AIO_SEQ_AMP=1, on a card: windows, and single decisions with a graph, in bfloat16 (half the memory of the graph; the moves differ by a few percent of their size)
        self.rooms, self.most, self.clock = {}, int(os.environ.get('AIO_SEQ_ROOMS', 4)), 0    # the batches whose last decisions are kept (Room): at most so many (0: none, every decision computes its whole history)

    def enable_compile(self, dynamic=True, backend='inductor'):
        """Compile `self.actor` step cores and outer hot helpers (zero `_orig_mod.` keys in state_dict)."""
        self.actor.enable_compile(dynamic=dynamic, backend=backend)
        return self

    def forget(self):
        """Drop what is kept of the batches' last decisions (the memory; the next decision of a batch builds it again)."""
        self.rooms = {}
        self.last_room = None

    def settle_rooms(self, inplace=True):
        """Settle all rooms holding active `late` autograd graphs into their detached KV rings.
        Call right after `loss.backward()` (`inplace=True`, zero transient K/V clone) so inactive lanes do not pin graphs."""
        for rs in self.rooms.values():
            for r in rs:
                if r.late_pos or r.read:
                    r.settle(inplace=inplace)

    def room(self, past, n, ncv, nmv, hidden=None, lane_tag=None):
        """The Room of the batch whose state holds the packets `past` (b,w,f), n (b,) of them there: the one kept for it,
        else a new one, built from the packets (the oldest batch's dropped past `most`).
        Zero-sync fast paths: (1) unmodified live `hidden` tensor object returned on the previous step (exact weakref +
        _version check); (2) caller-provided `lane_tag` (e.g. `(id(lane), id(prog))`) across `detach()`/`cut_rows()`."""
        b, w, key = past.shape[0], self.w, (past.shape[0], ncv, nmv, past.device)
        self.clock += 1
        rs = self.rooms.get(key, ())
        if hidden is not None:
            for r in rs:
                ref = r._last_h_ref
                if ref is not None and ref() is hidden:
                    if hidden._version == r._last_h_ver:
                        r.used = self.clock
                        if lane_tag is not None and not self.peeking:
                            r.lane_tag = lane_tag
                        return r
                    break                                                                                # in-place mutation on `hidden`: must run full equality check
            if lane_tag is not None:
                for r in rs:
                    if r.lane_tag == lane_tag:
                        ref = r._last_h_ref
                        if ref is None or ref() is not hidden or hidden._version == r._last_h_ver:
                            r.used = self.clock
                            return r
        flat, some = past.reshape(b, -1), n > 0
        if hidden is not None and bool(some.any()):
            for r in rs:
                if bool(((~some) | ((n == r.n) & (flat == r.raw).all(-1))).all()):
                    r.there[~some] = False                                                                   # a plant whose state was cleared starts again
                    r.late_live = [l & some for l in r.late_live]
                    if r.cut_pos is not None:
                        r.cut_pos = torch.where(some, r.cut_pos, torch.full_like(r.cut_pos, -10**9))
                    r.used = self.clock
                    if lane_tag is not None and not self.peeking:
                        r.lane_tag = lane_tag
                    return r
        if lane_tag is not None:
            for r in rs:
                if r.lane_tag == lane_tag:
                    r.lane_tag = None
        every = sorted(((r.used, k, r) for k, rs_ in self.rooms.items() for r in rs_), key=lambda x: x[0])
        for i, (used, k, old) in enumerate(every):                                                           # room for it: the batches not seen for long go, and the oldest past `most`
            if len(every)-i >= self.most or self.clock-used > IDLE:
                self.rooms[k].remove(old)
        del every
        d = self.actor.d
        r = Room(len(self.actor.layers), b, ncv+nmv, self.heads, w+1, d//self.heads, past.device)
        self.built += 1
        r.used = self.clock
        r.lane_tag = lane_tag
        self.rooms.setdefault(key, []).append(r)
        if hidden is None:
            return r
        size = past.shape[-1]-nmv
        mem_r = past.new_zeros(b, ncv+nmv, self.mem, dtype=torch.float32) if self.mem else None
        edge_r = past.new_zeros(b, ncv, nmv, self.edge, dtype=torch.float32) if self.edge else None
        rel_r = past.new_zeros(b, rel_state_size(ncv, nmv), dtype=torch.float32) if self.rel else None
        with torch.no_grad(), _amp_ctx(self.amp, past):
            for j in range(w):                                                                               # the packets the state holds, one decision after the other
                live = j >= w-n
                if bool(live.any()):
                    self.actor.step(past[:, j:j+1, :size], past[:, j:j+1, size:], r, mem_r, edge_r, rel_r, need_read=False)
                    if self.mem:
                        mem_r = torch.where(live[:, None, None], self.actor.mem_last, mem_r)
                    if self.edge:
                        edge_r = torch.where(live[:, None, None, None], self.actor.edge_last, edge_r)
                    if self.rel:
                        rel_r = torch.where(live[:, None], self.actor.rel_last, rel_r)
                    r.on(live)
        return r

    def state_size(self, ncv=None, nmv=None):
        from aio import net as N
        ncv, nmv = (N.NCV, N.NMV) if ncv is None else (ncv, nmv)
        return N.base_size(ncv, nmv)+self.pairs.size_of(ncv, nmv)

    def peek(self):
        """A context in which a single decision without a graph is not kept: the Room is left as it was (the trainer's
        decision after a window's last, for the critic, is made again as the next window's first; 11-23: kept, it made
        every window build its Room again from the 128 packets of its state)."""
        net = self

        class _Peek:
            def __enter__(self):
                net.peeking = True

            def __exit__(self, *_):
                net.peeking = False
        return _Peek()

    def cut(self, stop):
        """--guard cut on the decisions made with a graph since the state last came without one: the episodes where stop
        (b,) is set pass no gradient back past this point through what the time attention keeps of them (Room.late).
        Stores an immutable per-env `cut_pos` snapshot on Room and masks inside checkpointed `_step_core_late` (0 extra VRAM)."""
        r = self.last_room
        if r is None or not r.late_pos:
            return
        pos_t = torch.full((stop.shape[0],), r.pos, dtype=torch.long, device=stop.device)
        if r.cut_pos is None:
            r.cut_pos = torch.where(stop, pos_t, torch.full_like(pos_t, -10**9))
        else:
            r.cut_pos = torch.where(stop, pos_t, r.cut_pos)

    def steps(self, obs, previous, hidden=None, code=None, feats=None, need_read=True, lane_tag=None):
        """-> mean (b,T,NMV) before the tanh, the head's second output (--probe: aio.net.probing), the MV mask, the responses read
        (b,T,NCV,NMV,knots) or None if need_read=False, the state (1,b,state_size). code and feats (the told responses, the sums' features) are not used."""
        from aio import net as N
        from aio.interface import shape as shape_of
        ncv, nmv = shape_of(obs.shape[-1])
        b, T, w, f = obs.shape[0], obs.shape[1], self.w, obs.shape[-1]+nmv
        size, at = self.state_size(ncv, nmv), N.base_size(ncv, nmv)+N.sizes(ncv, nmv)[7]
        graph = torch.is_grad_enabled() and hidden is not None and hidden.requires_grad                       # the state carries the graph of the decisions before
        hidden_in = hidden
        if hidden is None:
            hidden = obs.new_zeros(1, b, size, dtype=torch.float32)
        elif hidden.shape[-1] < size:
            hidden = torch.cat((hidden, hidden.new_zeros(1, b, size-hidden.shape[-1])), -1)
            hidden_in = hidden
        past, n = hidden[0, :, at:at+w*f].reshape(b, w, f), hidden[0, :, at+w*f].round().long().clamp(0, w)      # right-aligned: the last n are there
        ma, NM = at+w*f+1, (ncv+nmv)*self.mem                                                              # mK: the tokens' memory behind them
        ea, NE = ma+NM, ncv*nmv*self.edge                                                                  # eK: persistent pairwise edge memory behind mK
        ra, NR = ea+NE, (rel_state_size(ncv, nmv) if self.rel else 0)                                      # r1: multi-scale DeltaNet state behind eK
        mem0 = hidden[0, :, ma:ma+NM].reshape(b, ncv+nmv, self.mem).clone() if self.mem else None          # clone tiny slices so autograd does not pin full `hidden`
        edge0 = hidden[0, :, ea:ea+NE].reshape(b, ncv, nmv, self.edge).clone() if self.edge else None
        rel0 = hidden[0, :, ra:ra+NR].clone() if self.rel else None
        if T > 1 and (self.fb or self.edge or self.rel):                                                   # recurrent cross-step feedback: exact step-by-step equivalence with T=1
            h_cur = hidden_in
            ms, ws, mks, ls = [], [], [], []
            for t in range(T):
                mt, wt, mkt, lt, h_cur = self.steps(obs[:, t:t+1], previous[:, t:t+1], h_cur, code, feats, need_read=need_read, lane_tag=lane_tag)
                ms.append(mt); ws.append(wt); mks.append(mkt); ls.append(lt)
            return torch.cat(ms, 1), torch.cat(ws, 1), torch.cat(mks, 1), (torch.cat(ls, 1) if need_read else None), h_cur

        def state(keep, det=False):
            mem = (self.actor.mem_last.reshape(b, NM).float(),) if self.mem else ()
            edg = (self.actor.edge_last.reshape(b, NE).float(),) if self.edge else ()
            rel_s = (self.actor.rel_last.reshape(b, NR).float(),) if self.rel else ()
            h0 = hidden[0].detach() if det else hidden[0]
            return torch.cat((h0[:, :at].float(), keep)+mem+edg+rel_s+(h0[:, ra+NR:].float(),), -1)[None]
        if T == 1 and self.most > 0:                                                                         # one decision: only it is computed (Room)
            r = self.room(past.detach().to(obs.dtype), n, ncv, nmv, hidden=hidden_in, lane_tag=lane_tag)
            self.last_room = r
            if not graph:
                r.settle()
            if torch.is_grad_enabled():                                                                      # with a graph through the decisions since the state last came without one
                with _amp_ctx(self.amp, obs):                                                                # (AIO_SEQ_AMP=1: hoisted-friendly, no re-entry if already active)
                    lab, mean, when = self.actor.step_graph(obs, previous.to(obs.dtype), r, mem0, edge0, rel0, need_read=need_read)
                r.on_graph()
            elif self.peeking:                                                                               # a decision that is not kept: the Room as it was
                snap = (r.at, r.pos, r.there[:, r.at].clone(), r.where[r.at].clone(), [K[:, :, r.at].clone() for K in r.K], [V[:, :, r.at].clone() for V in r.V])
                with _amp_ctx(self.amp, obs):
                    lab, mean, when = self.actor.step(obs, previous.to(obs.dtype), r, mem0, edge0, rel0, need_read=need_read)
                r.at, r.pos = snap[0], snap[1]
                r.there[:, r.at], r.where[r.at] = snap[2], snap[3]
                for K, V, k0, v0 in zip(r.K, r.V, snap[4], snap[5]):
                    K[:, :, r.at], V[:, :, r.at] = k0, v0
            else:
                with _amp_ctx(self.amp, obs):
                    lab, mean, when = self.actor.step(obs, previous.to(obs.dtype), r, mem0, edge0, rel0, need_read=need_read)
                r.on()
            mask = unpack(obs)[1][..., -1]
            past_next = torch.cat((past.detach()[:, 1:].to(obs.dtype), torch.cat((obs.detach(), previous.detach().to(obs.dtype)), -1)), 1)
            n_next = (n+1).clamp(max=w)
            keep = torch.cat((past_next.reshape(b, w*f).float(), n_next.float()[:, None]), -1)
            hidden = state(keep, det=True)
            mean = mean.float()*mask
            if mean.requires_grad:
                hidden = hidden+0.*mean.sum()                                                                # (the state says that a graph goes with it)
            if not self.peeking or torch.is_grad_enabled():
                r.raw, r.n = past_next.reshape(b, w*f), n_next
                r._last_h_ref, r._last_h_ver = weakref.ref(hidden), hidden._version
                if lane_tag is not None:
                    r.lane_tag = lane_tag
            return mean, when.float()*mask, mask, (None if lab is None else lab.float()), hidden
        x = torch.cat((past.to(obs.dtype), torch.cat((obs, previous.to(obs.dtype)), -1)), 1)                  # (b,w+T,f)
        j = torch.arange(w+T, device=obs.device)
        there = j[None, :] >= (w-n)[:, None]                                                               # (b,w+T)
        back = j[:, None]-j[None, :]                                                                        # a decision sees itself and the w before it: what it has when it runs alone
        seen = ((back >= 0) & (back <= w))[None] & (there[:, None, :] | (back == 0)[None])
        with _amp_ctx(self.amp, obs):
            lab, mean, when = self.actor(x[..., :obs.shape[-1]], x[..., obs.shape[-1]:], seen, w, mem0, need_read=need_read)  # the w decisions of the state: attended to, no graph through them
        mask = unpack(obs)[1][..., -1]
        keep = torch.cat((x[:, -w:].reshape(b, w*f).float(), (n+T).clamp(max=w).float()[:, None]), -1)
        hidden = state(keep)
        mean = mean.float()*mask
        return mean, when.float()*mask, mask, (None if lab is None else lab.float()), hidden

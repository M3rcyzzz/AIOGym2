"""Permutation-equivariant public channels with per-channel memory and MV-CV attention (11-05), full-prefix GRU,
residual MLP heads (dense since 11-04; earlier MoE)."""
import contextlib
import math
import torch
from torch import nn
from aio.interface import NCV, NMV, CF, MF, shape as shape_of

TOKEN = 128
MEMORY = 256
WIDTH = 512
EXPERTS = 1      # 11-04: dense heads. Runs up to 11-04 used 4 gated experts; their checkpoints still load
DEPTH = 3
TMEM = 64        # 11-05: recurrent memory of each CV and MV channel
ATT = 64         # MV->CV attention width
NTOK = NCV+NMV
STATE = MEMORY+NTOK*TMEM   # recurrent state carried between calls: the global GRU, then one state per channel


def symlog(x):
    return x.sign() * torch.log1p(x.abs())


def symexp(x):
    return x.sign() * torch.expm1(x.abs())


def state_size(ncv=NCV, nmv=NMV):
    """Length of the recurrent state for packets of ncv rows and nmv MVs (STATE: the default size's)."""
    return MEMORY+(ncv+nmv)*TMEM


def unpack(obs):
    NCV, NMV = shape_of(obs.shape[-1])                            # the packets' size (the module's names are the default one)
    cv = obs[..., :NCV*CF].reshape(*obs.shape[:-1], NCV, CF)
    mv = obs[..., NCV*CF:NCV*CF+NMV*MF].reshape(*obs.shape[:-1], NMV, MF)
    link = obs[..., -NCV*NMV:].reshape(*obs.shape[:-1], NCV, NMV)
    return cv, mv, link


def fp32_rnn():
    """GRUs stay in float32 (cuDNN, TF32) when the caller runs under bf16 autocast: cuDNN's
    RNNs take float and half only, and these GRUs are small and latency-bound. Outside autocast a no-op."""
    return torch.autocast('cuda', enabled=False) if torch.is_autocast_enabled() else contextlib.nullcontext()


def _f32(h):
    return None if h is None else h.float()


def pool(x, mask):
    return (x*mask[..., None]).sum(-2) / mask.sum(-1, keepdim=True).clamp_min(1)


class PublicChannels(nn.Module):
    def __init__(self):
        super().__init__()
        self.cv = nn.Sequential(nn.Linear(CF, TOKEN), nn.LayerNorm(TOKEN), nn.SiLU())
        self.mv = nn.Sequential(nn.Linear(MF+1, TOKEN), nn.LayerNorm(TOKEN), nn.SiLU())
        self.join = nn.Sequential(nn.Linear(TOKEN*3, TOKEN), nn.LayerNorm(TOKEN), nn.SiLU())

    def embed(self, obs, previous_action):
        """Channel tokens (masked), masks and the masked pairing matrix of a packet sequence."""
        cv, mv, links = unpack(obs)
        cm, mm = cv[..., -1], mv[..., -1]
        ce = self.cv(symlog(cv*cm[..., None])) * cm[..., None]
        me = self.mv(torch.cat((symlog(mv*mm[..., None]),
                               previous_action[..., None]*mm[..., None]), -1))
        links = links*cm[..., :, None]*mm[..., None, :]
        return ce, me, cm, mm, links

    def combine(self, ce, me, cm, mm, links):
        linked = torch.einsum('...ij,...id->...jd', links, ce)
        linked = linked/links.sum(-2)[..., None].clamp_min(1)
        global_cv = pool(ce, cm)
        joint = self.join(torch.cat((me, linked,
                        global_cv[..., None, :].expand_as(me)), -1))*mm[..., None]
        return joint, torch.cat((global_cv, pool(joint, mm)), -1), mm

    def forward(self, obs, previous_action):
        return self.combine(*self.embed(obs, previous_action))


def split_tokens(hidden, b, NCV=NCV, NMV=NMV):
    """(1, b, NTOK*TMEM) or None -> per-channel GRU states (1, b*NCV, TMEM), (1, b*NMV, TMEM)."""
    if hidden is None:
        return None, None
    hc = hidden[..., :NCV*TMEM].reshape(1, b*NCV, TMEM).contiguous()
    hm = hidden[..., NCV*TMEM:].reshape(1, b*NMV, TMEM).contiguous()
    return hc, hm


class TokenMemory(nn.Module):
    """Per-channel memory and MV->CV attention (11-05).

    Before, the GRU only saw the average over all channels, so the network could not tell which MV
    moved and which CV answered: the per-pair gain and time scale a cross-condition controller has to
    identify online (docs/generalist-forgetting-diagnosis-20261104.md, 7). Now every CV and every MV
    keeps its own GRU state (weights shared across channels, so any number and order of channels),
    fed with its own token and its partner's (the paired MV's token for a CV, the paired CVs' for an
    MV). The memories are added to the tokens, and each MV then attends over all CVs, with the pairing
    as a learned additive bias, so couplings outside the pairing can be learned (XHugWBC, ICML 2026;
    Suzuki et al. 2026: per-joint temporal tokens and attention across a morphology).

    Grown onto a trained network: cv_out, mv_out and att_out start at zero, so with these weights at
    their start the network computes exactly what it did before. History's load hook leaves them at
    that start when a checkpoint predates them."""
    def __init__(self):
        super().__init__()
        self.cv_gru = nn.GRU(TOKEN*2, TMEM, batch_first=True)
        self.mv_gru = nn.GRU(TOKEN*2, TMEM, batch_first=True)
        self.cv_out = nn.Linear(TMEM, TOKEN)
        self.mv_out = nn.Linear(TMEM, TOKEN)
        self.q = nn.Linear(TOKEN, ATT)
        self.k = nn.Linear(TOKEN, ATT)
        self.v = nn.Linear(TOKEN, ATT)
        self.att_out = nn.Linear(ATT, TOKEN)
        self.link_bias = nn.Parameter(torch.tensor(2.))
        for layer in (self.cv_out, self.mv_out, self.att_out):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, ce, me, cm, mm, links, hidden=None):
        """Tokens (b,T,N,TOKEN) -> tokens with memory and attention, per-step channel memories
        (b,T,NTOK,TMEM: CVs then MVs) and the channel states (1, b, NTOK*TMEM)."""
        b, T, NCV, NMV = ce.shape[0], ce.shape[1], ce.shape[2], me.shape[2]
        to_mv = torch.einsum('...ij,...id->...jd', links, ce)/links.sum(-2)[..., None].clamp_min(1)
        to_cv = torch.einsum('...ij,...jd->...id', links, me)/links.sum(-1)[..., None].clamp_min(1)
        xc = torch.cat((ce, to_cv), -1).transpose(1, 2).reshape(b*NCV, T, TOKEN*2)
        xm = torch.cat((me, to_mv), -1).transpose(1, 2).reshape(b*NMV, T, TOKEN*2)
        hc, hm = split_tokens(hidden, b, NCV, NMV)
        with fp32_rnn():
            oc, hc = self.cv_gru(xc.float(), _f32(hc))
            om, hm = self.mv_gru(xm.float(), _f32(hm))
        tc = oc.reshape(b, NCV, T, TMEM).transpose(1, 2)
        tm = om.reshape(b, NMV, T, TMEM).transpose(1, 2)
        ce = ce+self.cv_out(tc)*cm[..., None]
        me = me+self.mv_out(tm)*mm[..., None]
        score = torch.einsum('...jd,...id->...ji', self.q(me), self.k(ce))/math.sqrt(ATT)
        score = (score+self.link_bias*links.transpose(-1, -2)).masked_fill(cm[..., None, :] <= 0, -1e9)
        me = me+self.att_out(torch.einsum('...ji,...id->...jd', score.softmax(-1), self.v(ce)))*mm[..., None]
        state = torch.cat((hc.reshape(1, b, NCV*TMEM), hm.reshape(1, b, NMV*TMEM)), -1)
        return ce, me, torch.cat((tc, tm), 2), state


def _grown(module, keys):
    """Load hook: a checkpoint from before a grown module (TokenMemory) leaves it at its start, which
    computes what the checkpoint did. Any other missing key still fails a strict load."""
    keys.missing_keys[:] = [k for k in keys.missing_keys if 'tokens' not in k.split('.')]


class History(nn.Module):
    """Channel tokens, per-channel memories and a global GRU over the pooled channels. The recurrent
    state is one tensor (1, b, STATE): the global GRU's, then each channel's. A caller may start from
    zeros of the global size MEMORY (as written before 11-05); the channel states then start at zero."""
    def __init__(self):
        super().__init__()
        self.channels = PublicChannels()
        self.gru = nn.GRU(TOKEN*2, MEMORY, batch_first=True)
        self.tokens = TokenMemory()
        self.register_load_state_dict_post_hook(_grown)

    def flatten_parameters(self):
        for gru in (self.gru, self.tokens.cv_gru, self.tokens.mv_gru):
            gru.flatten_parameters()

    def steps(self, obs, previous, hidden=None):
        """-> joint, global memory (b,T,MEMORY), MV mask, channel memories (b,T,NTOK,TMEM), state."""
        hg = ht = None
        if hidden is not None:
            hg = hidden[..., :MEMORY].contiguous()
            ht = hidden[..., MEMORY:] if hidden.shape[-1] > MEMORY else None
        ce, me, cm, mm, links = self.channels.embed(obs, previous)
        ce, me, tok, ht = self.tokens(ce, me, cm, mm, links, ht)
        joint, x, mask = self.channels.combine(ce, me, cm, mm, links)
        with fp32_rnn():
            memory, hg = self.gru(x.float(), _f32(hg))
        return joint, memory, mask, tok, torch.cat((hg, ht), -1)

    def forward(self, obs, previous, hidden=None):
        joint, memory, mask, _, hidden = self.steps(obs, previous, hidden)
        return joint, memory, mask, hidden

    @torch.no_grad()
    def prefix(self, obs, previous, lengths, host_lengths=None, init=None):
        """State after each row's own prefix (its first `lengths` positions; zeros when 0). Since 11-05
        one pass over the whole padded prefix, read at each row's last position (a GRU's output is its
        state, and later padding cannot change earlier outputs): fixed shapes, no host lengths, so it
        also runs inside a CUDA graph. host_lengths is accepted for older callers and not needed.
        init: (1, b, STATE) state before the prefix (R2D2 stored state; None: zeros)."""
        b = obs.shape[0]
        NTOK = sum(shape_of(obs.shape[-1]))
        if obs.shape[1] == 0:
            return obs.new_zeros(1, b, MEMORY+NTOK*TMEM) if init is None else init
        _, memory, _, tok, _ = self.steps(obs, previous, init)
        rows = torch.arange(b, device=obs.device)
        last = (lengths-1).clamp_min(0).long()
        state = torch.cat((memory[rows, last], tok[rows, last].reshape(b, NTOK*TMEM)), -1)
        return (state*(lengths > 0).to(state.dtype)[:, None])[None]

    prefix_steps = prefix


class Residual(nn.Module):
    """Pre-LN residual block x + MLP(LN(x))/sqrt(DEPTH). hidden > WIDTH is an inverted bottleneck. The
    divisor stays sqrt(DEPTH) whatever the block count, so a head that grows a block still computes what it did."""
    def __init__(self, hidden=WIDTH):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, hidden),
                                 nn.SiLU(), nn.Linear(hidden, WIDTH))

    def forward(self, x):
        return x + self.net(x)/math.sqrt(DEPTH)


class ExpertHead(nn.Module):
    """Residual MLP head: project to WIDTH, DEPTH residual blocks, LayerNorm, linear readout.

    One expert (the default since 11-04) is a plain dense head; forward returns all-ones gate weights
    of width 1. Several experts is the earlier mixture: a softmax gate over parallel residual stacks
    (its actor gate had collapsed to one expert on every family).
    Loading a state dict rebuilds the head to the checkpoint's expert count, so MoE checkpoints
    (earlier runs) and dense ones both load into a fresh Actor();
    keep(k) turns a loaded MoE head into a dense one made of expert k."""
    def __init__(self, inputs, outputs, experts=None, hidden=WIDTH, depth=DEPTH):
        super().__init__()
        self.hidden, self.depth = hidden, depth
        self.project = nn.Linear(inputs, WIDTH)
        self.gate, self.experts = self.stacks(EXPERTS if experts is None else experts, hidden, depth)
        self.norm = nn.LayerNorm(WIDTH)
        self.out = nn.Linear(WIDTH, outputs)
        nn.init.normal_(self.out.weight, std=.001)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def stacks(experts, hidden=WIDTH, depth=DEPTH):
        gate = nn.Linear(WIDTH, experts) if experts > 1 else None
        return gate, nn.ModuleList([nn.Sequential(*[Residual(hidden) for _ in range(depth)]) for _ in range(experts)])

    def keep(self, k):
        """MoE -> dense: keep expert k's residual stack, drop the gate and the other experts."""
        self.experts = nn.ModuleList([self.experts[k]])
        self.gate = None
        return self

    def _load_from_state_dict(self, state_dict, prefix, *args):
        n = len({k[len(prefix):].split('.')[1] for k in state_dict if k.startswith(prefix+'experts.')})
        if n and n != len(self.experts):             # rebuild to the checkpoint's expert count before loading
            gate, experts = self.stacks(n, self.hidden, self.depth)
            like = self.project.weight
            self.experts = experts.to(like)
            self.gate = None if gate is None else gate.to(like)
        self._grow(state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args)

    @torch.no_grad()
    def _grow(self, state_dict, prefix):
        """Function-preserving growth (11-07): a checkpoint with narrower residual MLPs or fewer blocks is
        written into this head's shapes. The checkpoint's weights fill the leading slice; new hidden units keep their
        fresh input weights and get zero output weights; a block the checkpoint lacks gets a zero last layer
        (identity). The head then computes what the checkpoint did, and the new units learn from there."""
        own = self.state_dict()
        blocks = {tuple(k[len(prefix):].split('.')[:3]) for k in state_dict if k.startswith(prefix+'experts.')}
        if not blocks:
            return
        for k, v in own.items():
            parts = k.split('.')                     # experts.<e>.<block>.net.<1|3>.<weight|bias>
            if parts[0] != 'experts':
                continue
            key = prefix+k
            last = parts[-2] == '3'
            if key not in state_dict:
                if tuple(parts[:3]) in blocks:
                    continue                         # a block the checkpoint has: leave a genuine missing key to fail
                state_dict[key] = torch.zeros_like(v) if last else v.clone()
                continue
            old = state_dict[key]
            if old.shape == v.shape:
                continue
            assert old.dim() == v.dim() and all(o <= n for o, n in zip(old.shape, v.shape)), f'{key}: {tuple(old.shape)} does not grow into {tuple(v.shape)}'
            new = torch.zeros_like(v) if last else v.clone()
            new[tuple(slice(0, o) for o in old.shape)] = old.to(new)
            state_dict[key] = new

    def forward(self, x):
        x = self.project(x)
        if self.gate is None:
            return self.out(self.norm(self.experts[0](x))), x.new_ones(*x.shape[:-1], 1)
        gate = self.gate(x).softmax(-1)
        features = torch.stack([expert(x) for expert in self.experts], -2)
        return self.out(self.norm((features*gate[..., None]).sum(-2))), gate


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.history = History()
        self.head = ExpertHead(TOKEN+MEMORY, 2)
        with torch.no_grad():
            self.head.out.bias[1] = -1.

    def features(self, batch):
        h = self.history.prefix(batch['prefix_obs'], batch['prefix_previous'], batch['prefix_lengths'])
        return self.history(batch['obs'], batch['previous'], h)

    @staticmethod
    def head_input(joint, memory):
        return torch.cat((joint, memory[..., None, :].expand(*joint.shape[:-1], MEMORY)), -1)

    def distribution(self, joint, memory, mask):
        output, gates = self.head(self.head_input(joint, memory))
        output = output.float()      # bf16 autocast: sample and log-prob in float32 (no-op otherwise)
        return output[..., 0]*mask, output[..., 1].clamp(-5., 1.), gates

    def forward(self, obs, previous, hidden=None):
        joint, memory, mask, hidden = self.history(obs, previous, hidden)
        mean, logstd, gates = self.distribution(joint, memory, mask)
        return mean, logstd, mask, hidden, gates


def count(module):
    return sum(p.numel() for p in module.parameters())

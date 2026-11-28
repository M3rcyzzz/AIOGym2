"""A learned reader of the plant from the packets, in place of aio.net.Stats (11-14; nothing uses it unless asked).

Stats is written by hand: filters with fixed time constants, forgetting sums of products, a regularised solve.
Here the same job is left to weights. What is kept is only that the layer can multiply, as attention does:

  keys     every MV's packet features and last action through a layer, then through leaky memories whose time
           constants are learned: what that MV did lately, K numbers
  values   every row's packet features through a layer: what that row just did, V numbers
  memory   per set (each with a forgetting rate that is learned): C = sum value x key (per row, per MV),
           G = sum key x key (all MVs together). Linear attention over the whole past, kept as running sums: the
           cost of a decision does not grow with the history
  rounds   L learned steps W <- W + (C - W G) A_l, A_l a learned K x K matrix shared by all MVs: what a stack of
           linear attention layers does in context (von Oswald et al. 2023); nothing is solved
  out      per pair, a layer on W, C and the keys' energies -> what the pair memory takes in beside its packet features

No weight depends on the number of rows or MVs. The state is the memories and the sums (`size_of`)."""
import torch
from torch import nn

from aio.actor import symlog, unpack
from aio.interface import CF, MF

BLOCK, FLOOR = 64, 16.                    # decisions summed at once; the shortest forgetting (decisions): rho^-BLOCK stays small
K, V, L, SETS, NOUT = 8, 4, 4, 2, 32      # NOUT = aio.net.P: added to the pair memory's input


def size_of(ncv, nmv, k=K, v=V, sets=SETS, const=False):
    """Length of a Reader's state for packets of ncv rows and nmv MVs: the keys' memories, then per set C, G, E
    (`const`: with the constant key, one more beside the MVs')."""
    nx = nmv*k
    n1 = nx+bool(const)
    return nx+sets*(ncv*v*n1+n1*n1+ncv)


def spec(reader):
    """What `reader` asks for (aio.net.PairNet: True, or comma-separated 'const', 'rounds8', and the sizes 'k16' keys
    per MV, 'v8' values per row, 'w192' width of its layers) as Reader's arguments."""
    asked = [] if reader is True else [x for x in str(reader).split(',') if x]
    num = lambda q: [int(x[len(q):]) for x in asked if x.startswith(q) and x[len(q):].isdigit()]     # noqa: E731
    n, size = num('rounds'), {q: num(c) for q, c in (('k', 'k'), ('v', 'v'), ('width', 'w'))}
    assert all(x == 'const' or (x.lstrip('roundskvw') == '' or x.lstrip('roundskvw').isdigit()) and (num('rounds') or x[0] in 'kvw') for x in asked), reader
    out = dict(const='const' in asked, rounds=n[0] if n else L, shared=bool(n))
    return out | {q: z[0] for q, z in size.items() if z}


def said(reader):
    """The sizes a spec asks for that are not the usual ones, as the state's size needs them: dict(k=, v=)."""
    sp = spec(reader)
    return dict(k=sp.get('k', K), v=sp.get('v', V), const=sp['const'])


class Reader(nn.Module):
    def __init__(self, k=K, v=V, rounds=L, sets=SETS, out=NOUT, width=48, const=False, shared=False):
        """const (11-15; off unless asked for): one more key that is always 1, so the sums also hold every value's
        forgetting mean and the rounds give the part of it the MVs' keys do not explain (a row that goes on rising
        whatever the MVs do: a level's net inflow); read per row. shared (with rounds): every round the same A, so
        more rounds are more steps of one iteration and no more weights."""
        super().__init__()
        self.k, self.v, self.rounds, self.sets, self.const, self.shared = k, v, rounds, sets, bool(const), bool(shared)
        self.key = nn.Sequential(nn.Linear(MF+1, width), nn.SiLU(), nn.Linear(width, k))
        self.val = nn.Sequential(nn.Linear(CF, width), nn.SiLU(), nn.Linear(width, v))
        self.pole = nn.Parameter(torch.linspace(-1., 5., k))                    # the memories' time constants: 1+exp(.) decisions
        self.mix = nn.Parameter(torch.ones(k))                                  # how much of the key is its memory, how much its present
        self.forget = nn.Parameter(torch.tensor([5.3, 9.])[:sets])              # the sets' forgetting: exp(.) decisions (200 and 8,000 to start)
        self.A = nn.Parameter(torch.eye(k).repeat(1 if shared else rounds, 1, 1)/(2*k))
        if shared:
            self.register_buffer('nround', torch.tensor(rounds))
        if const:
            self.A0 = nn.Parameter(torch.full((1 if shared else rounds,), .5))
        self.unit = nn.Parameter(torch.tensor(.45))
        # where it starts: every key the MV's move (feature 1), the first value the row's change (feature 1); the layers
        # beside them start at nothing. All of it is trained.
        self.key0, self.val0 = nn.Linear(MF+1, k, bias=False), nn.Linear(CF, v, bias=False)
        for lin, n in ((self.key0, k), (self.val0, 1)):
            nn.init.zeros_(lin.weight)
            lin.weight.data[:n, 1] = 1.
        for seq in (self.key, self.val):
            nn.init.zeros_(seq[-1].weight)
            nn.init.zeros_(seq[-1].bias)
        self.out = nn.Sequential(nn.Linear(sets*(3*v*k+k+1+(3*v if const else 0)), 2*width), nn.SiLU(), nn.Linear(2*width, out))

    def size_of(self, ncv, nmv):
        return size_of(ncv, nmv, self.k, self.v, self.sets, self.const)

    def steps_of(self, nmv, device):
        """The rounds' matrices over all keys: the same K x K for every MV, the constant key's own number last."""
        eye, A = torch.eye(nmv, device=device), []
        for i in range(self.rounds):
            j = 0 if self.shared else i
            Ai = torch.kron(eye, self.A[j])
            A.append(torch.block_diag(Ai, self.A0[j].reshape(1, 1)) if self.const else Ai)
        return A

    def forward(self, obs, previous, state=None):
        """obs (b,T,SIZE) packets, previous (b,T,NMV) the last actions, state (b,size) or None
        -> (b,T,NCV,NMV,NOUT), the state."""
        dtype, obs, previous = obs.dtype, obs.float(), previous.float()             # the sums in single precision whatever the caller runs in
        state = None if state is None else state.float()
        cv, mv, _ = unpack(obs)
        b, T, ncv, nmv = cv.shape[0], cv.shape[1], cv.shape[-2], mv.shape[-2]
        k, v, nx = self.k, self.v, mv.shape[-2]*self.k
        cm, mm = cv[..., -1:], mv[..., -1:]
        qi, yi = torch.cat((symlog(mv*mm), previous[..., None]*mm), -1), symlog(cv*cm)
        q = (self.key(qi)+self.key0(qi))*mm                                              # (b,T,NMV,k)
        y = (self.val(yi)+self.val0(yi))*cm                                              # (b,T,NCV,v)
        if state is None:
            state = obs.new_zeros(b, self.size_of(ncv, nmv))
        x, o, C, G, E, n1 = state[:, :nx].reshape(b, nmv, k), nx, [], [], [], nx+self.const
        for _ in range(self.sets):
            C.append(state[:, o:o+ncv*v*n1].reshape(b, ncv*v, n1)); o += ncv*v*n1
            G.append(state[:, o:o+n1*n1].reshape(b, n1, n1)); o += n1*n1
            E.append(state[:, o:o+ncv]); o += ncv
        a = torch.exp(-1/(1+torch.exp(self.pole)))
        rho = torch.exp(-1/torch.exp(self.forget).clamp_min(FLOOR))
        A = self.steps_of(nmv, obs.device)
        keys = []
        for t in range(T):                                        # the keys' memories, a decision at a time (small); everything after it for all decisions at once
            x = a*x+torch.sqrt(1-a*a)*q[:, t]
            keys.append(self.mix*x+(1-self.mix)*q[:, t])
        keys = torch.stack(keys, 1)                               # (b,T,NMV,k)
        X, Y, y2 = keys.reshape(b, T, nx), y.reshape(b, T, ncv*v), y.square().sum(-1)
        if self.const:
            X = torch.cat((X, X.new_ones(b, T, 1)), -1)
        pair = lambda z: z[..., :nx].reshape(b, T, ncv, v, nmv, k).permute(0, 1, 2, 4, 3, 5).reshape(b, T, ncv, nmv, v*k)     # noqa: E731
        row = lambda z: z[..., nx].reshape(b, T, ncv, 1, v).expand(b, T, ncv, nmv, v)     # noqa: E731  (the constant key's: per row)
        f, last = [], []
        for s in range(self.sets):
            Cs, Gs, Es, c0, g0, e0 = [], [], [], C[s], G[s], E[s]
            for t0 in range(0, T, BLOCK):                         # the forgetting sums of a block by one cumulative sum: sum_j rho^(t-j) p_j = rho^t sum_j rho^-j p_j
                j = torch.arange(1, min(BLOCK, T-t0)+1, device=obs.device)
                up, dn = rho[s]**(-j), rho[s]**j
                sl = slice(t0, t0+len(j))
                c = (c0[:, None]+torch.cumsum(Y[:, sl, :, None]*X[:, sl, None, :]*up[None, :, None, None], 1))*dn[None, :, None, None]
                g = (g0[:, None]+torch.cumsum(X[:, sl, :, None]*X[:, sl, None, :]*up[None, :, None, None], 1))*dn[None, :, None, None]
                e = (e0[:, None]+torch.cumsum(y2[:, sl]*up[None, :, None], 1))*dn[None, :, None]
                Cs.append(c); Gs.append(g); Es.append(e)
                c0, g0, e0 = c[:, -1], g[:, -1], e[:, -1]
            c, g, e = torch.cat(Cs, 1), torch.cat(Gs, 1), torch.cat(Es, 1)+1e-10          # (b,T,NCV*v,nx), (b,T,nx,nx), (b,T,NCV)
            last += [c0.reshape(b, -1), g0.reshape(b, -1), e0]
            d = g.diagonal(dim1=2, dim2=3)+1e-10                                         # every key's energy, every row's
            si, se = d.rsqrt(), e.rsqrt().repeat_interleave(v, 2)
            Cn, Gn = c*se[..., :, None]*si[..., None, :], g*si[..., :, None]*si[..., None, :]      # as correlations: on no scale of the plant's
            W = torch.zeros_like(Cn)
            for i in range(self.rounds):
                W = W+(Cn-W@Gn)@A[i]
            unit = (e.sqrt().repeat_interleave(v, 2)[..., :, None]*si[..., None, :])/self.unit      # back on the plant's scale: change per move
            eg = (torch.log(d[..., :nx])/8).reshape(b, T, 1, nmv, k).expand(b, T, ncv, nmv, k)
            ee = (torch.log(e)/8)[..., None, None].expand(b, T, ncv, nmv, 1)
            f += [pair(W), pair(Cn), torch.asinh(pair(W*unit)), eg, ee]
            if self.const:
                f += [row(W), row(Cn), torch.asinh(row(W*unit))]
        self.keys = keys                                          # (b,T,NMV,k): what every MV did lately, as the sums took it (the labels' read-outs of aio.net use it)
        return self.out(torch.cat(f, -1)), torch.cat([x.reshape(b, nx)]+last, -1)

    def forward_loop(self, obs, previous, state=None):
        """obs (b,T,SIZE) packets, previous (b,T,NMV) the last actions, state (b,size) or None
        -> (b,T,NCV,NMV,NOUT), the state. One decision at a time: what `forward` computes, kept to check it against."""
        dtype, obs, previous = obs.dtype, obs.float(), previous.float()             # the sums in single precision whatever the caller runs in
        state = None if state is None else state.float()
        cv, mv, _ = unpack(obs)
        b, T, ncv, nmv = cv.shape[0], cv.shape[1], cv.shape[-2], mv.shape[-2]
        k, v, nx = self.k, self.v, mv.shape[-2]*self.k
        cm, mm = cv[..., -1:], mv[..., -1:]
        qi, yi = torch.cat((symlog(mv*mm), previous[..., None]*mm), -1), symlog(cv*cm)
        q = (self.key(qi)+self.key0(qi))*mm                                              # (b,T,NMV,k)
        y = (self.val(yi)+self.val0(yi))*cm                                              # (b,T,NCV,v)
        if state is None:
            state = obs.new_zeros(b, self.size_of(ncv, nmv))
        x, o, C, G, E, n1 = state[:, :nx].reshape(b, nmv, k), nx, [], [], [], nx+self.const
        for _ in range(self.sets):
            C.append(state[:, o:o+ncv*v*n1].reshape(b, ncv*v, n1)); o += ncv*v*n1
            G.append(state[:, o:o+n1*n1].reshape(b, n1, n1)); o += n1*n1
            E.append(state[:, o:o+ncv]); o += ncv
        a = torch.exp(-1/(1+torch.exp(self.pole)))
        rho = torch.exp(-1/torch.exp(self.forget).clamp_min(FLOOR))
        A = self.steps_of(nmv, obs.device)
        out, keys = [], []
        for t in range(T):
            x = a*x+torch.sqrt(1-a*a)*q[:, t]
            X = (self.mix*x+(1-self.mix)*q[:, t]).reshape(b, nx)
            keys.append(X.reshape(b, nmv, k))
            if self.const:
                X = torch.cat((X, X.new_ones(b, 1)), -1)
            Y = y[:, t].reshape(b, ncv*v)
            f = []
            for s in range(self.sets):
                C[s] = rho[s]*C[s]+Y[:, :, None]*X[:, None, :]
                G[s] = rho[s]*G[s]+X[:, :, None]*X[:, None, :]
                E[s] = rho[s]*E[s]+y[:, t].square().sum(-1)
                d = G[s].diagonal(dim1=1, dim2=2)+1e-10                                  # every key's energy, every row's
                e = E[s]+1e-10
                si, se = d.rsqrt(), e.rsqrt().repeat_interleave(v, 1)
                Cn, Gn = C[s]*se[:, :, None]*si[:, None, :], G[s]*si[:, :, None]*si[:, None, :]      # as correlations: on no scale of the plant's
                W = torch.zeros_like(Cn)
                for i in range(self.rounds):
                    W = W+(Cn-W@Gn)@A[i]
                pair = lambda z: z[..., :nx].reshape(b, ncv, v, nmv, k).permute(0, 1, 3, 2, 4).reshape(b, ncv, nmv, v*k)     # noqa: E731
                row = lambda z: z[..., nx].reshape(b, ncv, 1, v).expand(b, ncv, nmv, v)     # noqa: E731
                unit = (e.sqrt().repeat_interleave(v, 1)[:, :, None]*si[:, None, :])/self.unit      # back on the plant's scale: change per move
                eg = (torch.log(d[..., :nx])/8).reshape(b, 1, nmv, k).expand(b, ncv, nmv, k)
                ee = (torch.log(e)/8)[:, :, None, None].expand(b, ncv, nmv, 1)
                f += [pair(W), pair(Cn), torch.asinh(pair(W*unit)), eg, ee]
                if self.const:
                    f += [row(W), row(Cn), torch.asinh(row(W*unit))]
            out.append(self.out(torch.cat(f, -1)))
        state = torch.cat([x.reshape(b, nx)]+[z.reshape(b, -1) for s in range(self.sets) for z in (C[s], G[s], E[s])], -1)
        self.keys = torch.stack(keys, 1)                          # (b,T,NMV,k): what every MV did lately, as the sums took it (the labels' read-outs of aio.net use it)
        return torch.stack(out, 1), state

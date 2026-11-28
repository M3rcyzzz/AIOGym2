"""The law worked out, not learned (11-11): its two steps as arithmetic on the responses a network reads.

aio.law: nearly all of the teacher is move = K . E / V. A network taught to read E from its history through a
learned product (a kernel read from a pair's response times six filtered moves) could not: with the teacher's
own K, 31 points (the MVs left alone 0, the teacher 100), told the true responses or not. Yet E is arithmetic
on the responses and the moves made, and so is the move:

  E     E[l, a] = e_now[a] - sum_i sum_j (S_ai(L_l + 1 + j) - S_ai(1 + j)) du_i(t - j)
        the packet's error of row a now (target less measurement, bands), less what the moves made so far have
        yet to do to it by the label lag L_l. du_i(t - j): the move of MV i the packet of j decisions ago
        showed (made the decision before it), full travel 1; S_ai: the pair's response, read at a few lags
        (model.KNOTS), a straight line in between; j < J.
        An analyzer's value is some decisions old (the packet says how many): what the moves have done since
        is added to it first.
        A row that goes on rising (a level: its responses as steep over the last two knots as before them)
        is taken to go on at its last slope beyond the last knot, and the part of its rate that the moves do
        not explain (the packet's change of the row less what the moves did in the last decision, smoothed)
        to last.
  move  the teacher's design (model.dmc_design: eight moves at the decisions model.MT over 240 decisions, its
        move suppression) solved on those responses for E; its first move, over the slew limit.

No weights: `Formula` keeps the last J moves of every MV (a state of the network), the rest are constants.
Told the true responses at the knots, with nothing learned (python -m aio.formula; 64 episodes of the batch of
seed 10^6, the MVs left alone 0, the teacher 100; the law with the teacher's own E, K and V 98.4): 16 knots 83.1
(through an untrained network, aio.net with both switches: the same), 9 knots 72.9, the published pairs alone
53.9; E by the formula with the teacher's own gains 83.0, and the teacher's E with the move solved here 98.3: the
solve loses nothing, what is missing is in E. What the packet does not carry, a network's heading readout has to
add (taking the formula without the analyzers' ages, 80.2, as the base): the analyzers' delay +2.9 (now in the
formula), the error on true values +0.7 more (noise, sensor lags), where the teacher's steady-state layer and
the output curve put the error now +1.8 more, the teacher's own pending effects +11.2 (of the MVs alone +6.5:
levels integrated from their true rate +2.4, the other rows +4.1, with every lag of the responses and the MVs'
offset at the episode's start; the unmeasured loads' after-effects +1.8, the measured disturbances' +2.9), the
valve curve +0.5, the plant's clock 0.
Those numbers are of the packet before 11-11, whose first one showed no move where an episode starts with its
valves just moved off their steady positions (half of them; the teacher's prediction knew). With that move in
the first packet: 16 knots 87.3 (through the network the same; the teacher's own gains 87.1), 9 knots 76.9, the
published pairs alone 58.3. What arithmetic could still add, each on its own: the steady-state layer of the
teacher (where the rows can settle inside the limits) +2.4, the measured disturbances' responses given as well
and their pending effects by the same formula +1.5; together 92.2, where the teacher's own bookkeeping with the
16 knots is 94.6 without the unmeasured loads (97.2 with them). Neither is done.
It needs the responses to a step test's accuracy. On the ones a network reads from closed-loop data (weights of
v2-law, nine knots) it gave 26 points where the same network's learned readouts gave 66 and the true responses
73; with a tenth of the read-out's error left 64, a quarter 56. So a network uses it where the responses are
given and its learned readouts where it reads them (aio.net: `given`).
"""
import torch
from torch import nn

from aio.interface import NCV, NMV
from aio.actor import unpack
from aio.sim.model import KNOTS, LAB_LAGS, NLAB, MT, P_DMC, LAM_DMC, LAM_REL

J = 240                          # decisions of moves kept
SPAN = .45                       # the packet's move, per full travel, when the MV limits in force are at their widest (.9 of the travel)
RATE = .95                       # smoothing of a level's unexplained rate, per decision
LEVEL = .95                      # a row goes on rising: the slope of its responses over the last two knots is at least this much of the slope before them
S_CLIP = 12.                     # the responses read, asinh, bounded before they are used
AGE = 40                         # decisions an analyzer's value can be old (prior: a cycle of up to 19 decisions, reported one cycle late)
SMEM = NMV*J+NMV+NCV             # the state: every MV's moves, its widest limits so far, every row's smoothed change


def hat(lags, knots):
    """(len(lags), len(knots)): a response's values at the knots -> at the lags: 0 at lag 0, a straight line
    in between, flat beyond the last knot."""
    t = torch.as_tensor(lags, dtype=torch.float64)
    L = torch.tensor((0,)+tuple(knots), dtype=torch.float64)
    one = torch.ones_like(t)
    return torch.stack([torch.minimum(((t-L[l-1])/(L[l]-L[l-1])).clamp(0, 1), ((L[l+1]-t)/(L[l+1]-L[l])).clamp(0, 1) if l+1 < len(L) else one)
                        for l in range(1, len(L))], 1)


class Formula(nn.Module):
    """knots: the lags the responses are given at (model.KNOTS)."""

    def __init__(self, knots=LAB_LAGS):
        super().__init__()
        self.knots, last = tuple(knots), knots[-1]
        age, lags = 1+torch.arange(J), torch.tensor(LAB_LAGS)
        B = lambda t: hat(t, knots)                               # noqa: E731
        W = torch.stack([B(age+L)-B(age) for L in LAB_LAGS], 0)    # (NLAB, J, NK): what a move of that age has yet to do by each label lag, per knot
        V = ((age[None]+lags[:, None]-last).clamp_min(0)-(age[None]-last).clamp_min(0)).double()/(last-knots[-2])     # (NLAB, J): the part beyond the last knot, per unit of the last slope
        W1 = list((1-RATE)*(B(age)-B(age-1)))                      # (J, NK): what it did in the last decision, smoothed as the rows' change is
        for j in range(1, J):
            W1[j] = W1[j]+RATE*W1[j-1]
        self.ends = (float(last-knots[-2]), float(knots[-2]-knots[-3]))
        WA = torch.stack([B(age)-B((age-a).clamp_min(0)) for a in range(AGE+1)], 0)       # (AGE+1, J, NK): what it has done since a value that old was taken
        q = torch.arange(P_DMC)
        A = torch.stack([B((q-t+1).clamp_min(0)) for t in MT], 0)                 # (M, P, NK): a move at MT[m] at the horizon's lags, per knot
        for name, x in (('W', W.flip(1).reshape(NLAB, J, -1).permute(1, 0, 2).reshape(J, -1)), ('V', V.flip(1).T), ('W1', torch.stack(W1, 0).flip(0)),
                        ('WA', WA.flip(1).permute(1, 0, 2).reshape(J, -1))):
            self.register_buffer(name, x.float().contiguous(), persistent=False)                  # oldest move first, as the moves are kept
        self.register_buffer('lags', lags.float(), persistent=False)
        self.register_buffer('C', torch.einsum('mqk,nql->mnkl', A, A), persistent=False)        # the design's sums over the horizon
        self.register_buffer('F', torch.einsum('mqk,ql->mlk', A, hat(q+1, LAB_LAGS)), persistent=False)

    @torch.no_grad()
    def memory(self, obs, state=None):
        """obs (b,T,SIZE) packets, state (b,SMEM) or None -> per decision, what the moves made so far have yet
        to do by every label lag per unit of every knot's response Z (b,T,NMV,NLAB,NK), its part beyond the
        last knot per unit of the last slope (b,T,NMV,NLAB), what they did in the last decision (b,T,NMV,NK)
        and the rows' change (b,T,NCV), both smoothed, the slew limit in full travel (b,T,NMV), what they
        have done since every row's value was taken (b,T,NCV,NMV,NK; None: no analyzer); the state."""
        cv, mv, _ = unpack(obs)
        b, T = obs.shape[:2]
        state = obs.new_zeros(b, SMEM) if state is None else state
        hist, wide, dys = state[:, :NMV*J].reshape(b, NMV, J).float(), state[:, NMV*J:NMV*J+NMV].float(), state[:, NMV*J+NMV:].float()
        mm = mv[..., -1].float()
        # the packet gives position, move and slew limit inside the MV limits in force: where the slew feature is
        # smallest the limits are at their widest, taken as .9 of the travel
        inv = (mm/mv[..., 2].float().clamp_min(1e-6)).nan_to_num(0., 0., 0.)
        wide = torch.maximum(wide[:, None], inv.cummax(1).values)                     # (b,T,NMV)
        du = (mv[..., 1].float()*mm).nan_to_num(0., 0., 0.).clamp(-1e3, 1e3)*SPAN*inv/wide.clamp_min(1e-9)
        win = torch.cat((hist, du.transpose(1, 2)), -1).unfold(-1, J, 1)[:, :, 1:]    # (b,NMV,T,J): the last J moves at every decision, oldest first
        Z = (win@self.W).reshape(b, NMV, T, NLAB, -1).transpose(1, 2)
        tail, Z1 = (win@self.V).transpose(1, 2), (win@self.W1).transpose(1, 2)
        dy, out = (cv[..., 1]*cv[..., -1]).float().nan_to_num(0., 0., 0.).clamp(-1e3, 1e3), []
        for t in range(T):
            dys = RATE*dys+(1-RATE)*dy[:, t]
            out.append(dys)
        old, since = (torch.expm1(cv[..., 8].float().clamp(0., 6.))*cv[..., -1].float()).round().clamp(0, AGE).long(), None     # (b,T,NCV) decisions
        has = (old > 0).flatten(1).any(1)
        if bool(has.any()):                                       # an analyzer's value is that old: what the moves did since is not in it
            G = (win[has]@self.WA).reshape(-1, NMV, T, AGE+1, self.WA.shape[1]//(AGE+1)).permute(0, 2, 1, 3, 4)       # (b',T,NMV,AGE+1,NK)
            G = torch.gather(G[:, :, None].expand(-1, -1, NCV, -1, -1, -1), 4, old[has][:, :, :, None, None, None].expand(-1, -1, -1, NMV, 1, G.shape[-1]))[:, :, :, :, 0]
            since = G.new_zeros(b, T, NCV, NMV, G.shape[-1]).index_copy(0, has.nonzero()[:, 0], G)
        state = torch.cat((win[:, :, -1].reshape(b, NMV*J), wide[:, -1], dys), -1).to(state.dtype)
        return (Z, tail, Z1, torch.stack(out, 1), SPAN/wide.clamp_min(1e-9)*mm, since), state

    @staticmethod
    def weight(obs):
        """(b,T,NCV): how the teacher weighs a band of error of every row: the root of its priority; 0 on rows not tracked."""
        cv = unpack(obs)[0].float()
        return cv[..., 5].clamp_min(0.).sqrt()*cv[..., 6]*cv[..., -1]

    def rising(self, S):
        """(b,T,NCV): 1 on the rows that go on rising (levels) by their responses S (b,T,NCV,NMV,NK)."""
        a, b = (S[..., -1]-S[..., -2]).abs().sum(-1)/self.ends[0], (S[..., -2]-S[..., -3]).abs().sum(-1)/self.ends[1]
        return ((a > LEVEL*b) & (a > 1e-9)).to(S.dtype)

    def heading(self, obs, S, mem):
        """E (b,T,NCV,NLAB): how far every tracked row will be from its target at the label lags if the MVs stop
        now, in bands as the teacher weighs them (`weight`); then the part of every row's change per decision
        that its MVs do not explain (b,T,NCV; weighted the same) and which rows go on rising (b,T,NCV).
        S (b,T,NCV,NMV,NK): the responses, bands per full travel, 0 on the pairs not in service; mem: of
        `memory`."""
        Z, tail, Z1, dys = (x.to(S.dtype) for x in mem[:4])
        cv = unpack(obs)[0]
        lev = self.rising(S.detach())
        pend = torch.einsum('btaik,btilk->btal', S, Z)
        more = torch.einsum('btai,btil->btal', S[..., -1]-S[..., -2], tail)           # a level goes on at its last slope
        rate = dys-torch.einsum('btaik,btik->bta', S, Z1)                             # and at the rate its MVs do not explain
        now = -cv[..., 0].to(S.dtype)
        if mem[5] is not None:                                    # an analyzer's row: brought up to now
            now = now-torch.einsum('btaik,btaik->bta', S, mem[5].to(S.dtype))
        E, w = now[..., None]-pend-lev[..., None]*(more+rate[..., None]*self.lags.to(S.dtype)), self.weight(obs).to(S.dtype)
        return E*w[..., None], rate*w, lev

    def move(self, obs, S, E, mem, chunk=2048):
        """(b,T,NMV): the first move of the teacher's plan without its limits, in slew limits (not held to
        them): the design solved on the responses S (b,T,NCV,NMV,NK) for the weighted errors E (b,T,NCV,NLAB)
        of the tracked rows (the first NMV row slots). The teacher suppresses moves per unit of valve flow;
        here per unit of travel (the valve curve is not in the packet). Gradient goes to E and S through the
        right-hand side; the system's matrix is taken as given."""
        b, T, M = S.shape[0], S.shape[1], len(MT)
        w = self.weight(obs)[:, :, :NMV].flatten(0, 1).double()
        s = S[:, :, :NMV].flatten(0, 1).double()*w[:, :, None, None]                  # (x, rows, NMV, NK)
        e, out = E[:, :, :NMV].flatten(0, 1).double(), []
        first = torch.eye(M*NMV, NMV, dtype=torch.float64, device=S.device)
        for a in range(0, s.shape[0], chunk):
            sa = s[a:a+chunk]
            with torch.no_grad():                                 # H d = g in the moves; the first NMV rows of H^-1 give the first of them
                G = torch.einsum('xijkl,mnkl->xminj', torch.einsum('xaik,xajl->xijkl', sa, sa), self.C).reshape(-1, M*NMV, M*NMV)
                H = G+torch.diag_embed(LAM_DMC*P_DMC+LAM_REL*G.diagonal(dim1=1, dim2=2))
                Y = torch.linalg.solve_ex(H, first.expand(H.shape[0], -1, -1))[0].nan_to_num(0., 0., 0.)
            g = torch.einsum('xaik,mlk,xal->xmi', sa, self.F, e[a:a+chunk]).reshape(-1, M*NMV)
            out.append(torch.einsum('xqi,xq->xi', Y, g))
        slew = mem[4]
        return (torch.cat(out, 0).reshape(b, T, NMV)/slew.double().clamp_min(1e-9)).float()*(slew > 0)


class Told:
    """The formula as a controller of aio.sim.control, told the true responses at the knots (never how a network
    runs: what the two steps are worth before anything is learned). solve False: E by the formula, the
    teacher's own gains and valve curve (aio.law). own: the published pairs' responses alone."""

    def __init__(self, knots=16, solve=True, own=False):
        self.nk, self.solve, self.own = knots, solve, own

    def reset(self, prog):
        prog.set_teacher('late')
        self.f, self.h = Formula(KNOTS[self.nk]).to(prog.device), None

    @torch.no_grad()
    def __call__(self, prog, st, k, obs):
        from aio.frame import canonical
        from aio import law as L
        o, s = canonical(obs.float())
        S = torch.sinh((prog.labels(k, self.nk).float()*s[:, :, None]).unflatten(-1, (NCV, self.nk)).transpose(1, 2))[:, None]     # (n,1,NCV,NMV,NK)
        if self.own:
            S = S*prog.c['link'][:, None, :, :, None].float()
        mem, self.h = self.f.memory(o[:, None], self.h)
        E = self.f.heading(o[:, None], S, mem)[0]
        if self.solve:
            return (self.f.move(o[:, None], S, E, mem)[:, 0].clamp(-1., 1.)*s).to(prog.dtype)
        _, v = L.heading(prog, st, k)
        K = L.tables(prog)[prog.ar, v['sg']]*prog.c['mvm'][:, :, None, None]
        m = torch.einsum('njlr,nlr->nj', K, E[:, 0, :NMV].transpose(1, 2).to(K.dtype))/L.reach(prog, v)
        return L.held(prog, st, k, m)


class Wired:
    """The same through a network (aio.net with both steps worked out, nothing trained, told the true responses
    at the knots): the move its law gives, before its head has learned what to do with it."""

    def __init__(self, knots=16):
        self.nk = knots

    def reset(self, prog):
        from aio import net as N
        prog.set_teacher('late')
        torch.manual_seed(0)
        self.net = N.AIO(law=True, knots=self.nk, formula=True, solve=True).to(prog.device).eval()
        self.h, self.prev = None, torch.zeros(prog.n, NMV, device=prog.device)

    @torch.no_grad()
    def __call__(self, prog, st, k, obs):
        from aio.frame import canonical
        from aio.net import pair_labels
        o, s = canonical(obs.float())
        code = (pair_labels(prog.labels(k, self.nk).float()*s[:, :, None])[:, None], torch.ones(prog.n, device=prog.device))
        mask, self.h = self.net.steps(o[:, None], self.prev[:, None], self.h, code)[2::2]
        self.prev = self.net.pairs.read_law[3][:, 0].clamp(-1., 1.)*mask[:, 0]
        return (self.prev*s).to(prog.dtype)


if __name__ == '__main__':
    import argparse, json, time
    from aio.sim import control as CT, core as C
    from aio import law as L, scale as SC
    ap = argparse.ArgumentParser(description='What the formula is worth told the true responses at the knots, on one batch (the MVs left alone 0, the teacher 100)')
    ap.add_argument('--n', type=int, default=64); ap.add_argument('--seed', type=int, default=1_000_000)
    ap.add_argument('--device', default='cpu'); ap.add_argument('--mech', type=float, default=0.)
    ap.add_argument('--arms', default='law,gains16,solve16,solve9,own16')
    a = ap.parse_args()
    prog = C.build(a.n, a.seed, a.device, mech=a.mech)
    arms = dict(law=L.Plain(), gains16=Told(16, False), gains9=Told(9, False), solve16=Told(16), solve9=Told(9), own16=Told(16, own=True), net16=Wired(16))
    per = {}
    for q, ctl in dict(hold=CT.Hold(), pi=CT.PI(), teacher=CT.Teacher('late'), **{q: arms[q] for q in a.arms.split(',') if q}).items():
        t0 = time.time()
        per[q] = CT.rollout(prog, ctl)['score'].tolist()
        print(json.dumps(dict(ctl=q, score=round(sum(per[q])/a.n, 1), s=round(time.time()-t0, 1))), flush=True)
    print(json.dumps({q: round(SC.points(v, per['hold'], per['teacher']), 1) for q, v in per.items()}))

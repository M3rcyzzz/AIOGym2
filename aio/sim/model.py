"""What the simulator knows about a plant and the learner never sees: its exact local model.

For every segment of an episode (a plant between two events, at the steady point the targets ask for) this
module gives, in closed form from the plant's parameters and without stepping the plant:

- `response`: the unit step response of every row to every input, before the row's integrator and output
  curve, per unit valve flow, at the reference time scale: the linear core of the plant, which the teacher
  uses with the valve curve, the output curve, the levels and the load-dependent time scale handled exactly;
- `local`: from it, the small-signal response of the CV values to the MV positions at the steady point, in
  decisions (what a step test there would identify);
- `dmc_design`, `dmc_move`: the dynamic-matrix controller of a response with the MV limits (the teacher's and the
  references' law);
- `labels`: the response at a few lags in the packet's units, the supervised target of the identification
  readout;
- `pi_tuning`: the lambda-tuned PI of the published pairing (the formula reference).

The response follows the plant step of aio.sim.core operation by operation (positioner lag, dead times,
modes, row lags), so a small step on the real plant away from valve friction reproduces `local`
(tests/test_sim.py).
"""
import math
import torch

from aio import interface as IF
from aio.sim.prior import hprime, phi_of, sizes, nzmax, DMAX

LAGS = 240                       # length of the step responses (decisions)
P_DMC, M_DMC, LAM_DMC = 240, 8, 3.
MT = (0, 1, 3, 6, 10, 16, 26, 42)           # decisions from now at which the plan moves the MVs (M_DMC of them): spread out, so
                                            # that it can hold a level off balance long enough to bring it back, or wait out a dead time
LAM_REL = .02                    # move suppression relative to each move's own effect (diagonal of the Hessian)
LOAD_SUPP = 30.                  # the throughput MV of a load-dependent plant is moved this much more reluctantly
LAB_LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 240)
NLAB = len(LAB_LAGS)
# lags a response can be read out at (a straight line in between). Nine are too few to work out what the moves made
# so far have yet to do (aio.formula; the teacher's own bookkeeping with the responses known at the knots alone, 64
# episodes, the MVs left alone 0 and the teacher 100: 9 knots 85.7, 16 96.4, 24 97.6, every lag 97.6)
KNOTS = {NLAB: LAB_LAGS, 16: (1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128, 181, 240)}
NZMAX = 4                        # zone CVs a plant can have (prior.NZ_P) at the default size; prior.nzmax at any
NR = IF.NMV+NZMAX                # rows the teacher predicts: the tracked and the zone CVs
ZL = (4, 8, 16, 32, 64, 128, 240)            # lags at which the teacher checks the zone limits
ZDT = (6., 6., 12., 24., 48., 88., 56.)      # decisions each of them stands for


def lerp_at(x, pos):
    """x (m, L+1, ...) read at fractional lags pos (m, J), flat beyond the last entry."""
    L = x.shape[1]-1
    pos = pos.clamp(0, L)
    i0 = pos.floor().long().clamp_max(L-1)
    tail = [1]*(x.dim()-2)
    idx = i0.view(*i0.shape, *tail).expand(-1, -1, *x.shape[2:])
    a, b = torch.gather(x, 1, idx), torch.gather(x, 1, idx+1)
    return a+(pos-i0).view(*pos.shape, *tail).to(x.dtype)*(b-a)


@torch.no_grad()
def response(p, lags=LAGS, phi=None):
    """Fr (m, lags+1, NCV, NU): unit step response of every row before its integrator and output curve (a
    level's rate, the value of any other row) to a unit of valve flow of every MV and a unit of every
    disturbance and load, at time scale 1. phi (m,): the time scale the plant runs at where the model is
    for (None: 1). The plant keeps DMAX decisions of its inputs: a dead time that is longer than that in
    decisions on a plant running slowly is DMAX decisions there, and so it is here."""
    NCV, NMV, ND, NQ, NDQ, NU = sizes(p)
    m, dev, f = p['K'].shape[0], p['K'].device, p['K'].dtype
    slope = torch.cat((p['mvm'], p['dact']), 1)
    aa = torch.cat((p['aa'], torch.zeros(m, NDQ, dtype=f, device=dev)), 1)
    k = torch.arange(lags+1, dtype=f, device=dev)
    uin = slope[:, None, :]*(1-aa[:, None, :]**k[None, :, None])          # input after the positioner lag, uin[0] = 0
    d = (p['dead'] if phi is None else torch.minimum(p['dead'], (DMAX*phi).round()[:, None, None])).long()
    m0, m1 = torch.exp(p['p1r']), torch.exp(p['p2r'])
    lr = torch.stack((m0*torch.cos(p['p1i']), m1*torch.cos(p['p2i'])), -1)
    li = torch.stack((m0*torch.sin(p['p1i']), m1*torch.sin(p['p2i'])), -1)
    cr = torch.stack((p['c0r'], p['c1r']), -1); ci = torch.stack((p['c0i'], p['c1i']), -1)
    al = torch.exp(-1/p['tlag'])[:, :, None]
    nlag = p['nlag'][:, :, None]
    sr = torch.zeros(m, NCV, NU, 2, dtype=f, device=dev); si = torch.zeros_like(sr)
    l1 = torch.zeros(m, NCV, NU, dtype=f, device=dev); l2 = torch.zeros_like(l1)
    Fr = torch.zeros(m, lags+1, NCV, NU, dtype=f, device=dev)
    ut = uin.transpose(1, 2)[:, None].expand(m, NCV, NU, lags+1)
    for t in range(1, lags+1):
        ud = torch.gather(ut, 3, (t-d).clamp_min(0)[..., None])
        sr, si = lr*sr-li*si+(1-lr)*ud, lr*si+li*sr-li*ud
        fast = p['K']*(cr*sr-ci*si).sum(-1)
        l1 = al*l1+(1-al)*fast
        l2 = al*l2+(1-al)*l1
        Fr[:, t] = torch.where(nlag == 0, fast, torch.where(nlag == 1, l1, l2))
    return Fr


def zone_rows(p):
    """(m, NZMAX) row of every zone slot (the rows after the tracked ones) and whether it is in use."""
    NCV, NMV, ND, NQ, NDQ, NU = sizes(p); NZMAX = nzmax(NCV, NMV, ND)
    row = (p['trk'].sum(1).long()[:, None]+torch.arange(NZMAX, device=p['trk'].device)[None]).clamp_max(NCV-1)
    return row, torch.gather(p['zon'], 1, row) > 0


def values(Fr, phi, p):
    """The response Fr (m, lags+1, rows, inputs; `response`) in decisions at the time scale phi (m,), as the value
    of every row before its output curve: a level (p['integ'], with p['tint'] and p['lam'] of those rows) adds up its rate."""
    L, f, dev = Fr.shape[1]-1, Fr.dtype, Fr.device
    j = torch.arange(L+1, dtype=f, device=dev)
    g = lerp_at(Fr, j[None]*phi[:, None])                        # the plant runs phi times faster than the model
    rho = (1+p['lam']/p['tint'])[:, None, :, None]
    pw = rho**j[None, :, None, None]
    lev = pw*torch.cumsum(g/pw, 1)/p['tint'][:, None, :, None]   # a level adds up its rate
    return torch.where((p['integ'] > 0)[:, None, :, None], lev, g)


@torch.no_grad()
def local(Fr, p, info, lags=LAB_LAGS):
    """Small-signal response at the steady point `info` in decisions, to the MVs:
    Sf (m, lags+1, NMV, NMV)   the value of rows 0..NMV-1 before the output curve, per unit of valve flow
                               (the teacher's design model), Sz (m, lags+1, NZMAX, NMV) the same of the
                               zone rows;
    S (m, lags+1, NMV, NMV), Sl (m, len(lags), NCV, NMV)   the normalised CV value of rows 0..NMV-1 and of
                               every row at `lags`, per full span of MV position (what a step test identifies);
    phi (m,)                   the steady time scale."""
    NCV, NMV, ND, NQ, NDQ, NU = sizes(p); NZMAX = nzmax(NCV, NMV, ND)
    m, L = Fr.shape[0], Fr.shape[1]-1
    phi = phi_of(p, info['un'], p['dlev'])
    v = values(Fr[..., :NMV], phi, p)
    slope = (p['gam']*info['un'].clamp_min(1e-6)**(p['gam']-1)*p['mvm'])[:, None, None, :]
    y = v*(hprime(info['v'], p['wa'])*p['prc'])[:, None, :, None]*slope
    row, on = zone_rows(p)
    Sz = torch.gather(v, 2, row[:, None, :, None].expand(m, L+1, NZMAX, NMV))*on[:, None, :, None]
    return v[:, :, :NMV], Sz, y[:, :, :NMV], y[:, list(lags)], phi


@torch.no_grad()
def dmc_design(S, wy, mvm, Sz, wz, supp=None, P=P_DMC, M=M_DMC, chunk=None, lam=None):
    """The dynamic-matrix controller of the response S (m, lags+1, NMV, NMV) (Cutler & Ramaker 1980:
    errors over P decisions weighted by wy, M moves at the decisions MT
    from now, move suppression LAM_DMC*P times supp (m, NMV) plus LAM_REL of each move's own effect), in
    the form the teacher solves with limits:
    Sw (m, P, NMV, NMV)        the weighted response;
    HX (m, M*NMV, M*NMV)       the Hessian in the planned MV positions X_j = u(MT_j) - u_now (the moves
                               are X_j - X_{j-1}), which the MV limits bound directly;
    Az (m, NZMAX*len(ZL), M*NMV)  the weighted value of every zone row at the lags ZL per unit of X
                               (Sz, wz: their response and weight).
    lam: both move suppressions times this (a model that is not exact asks for more; None: as they are)."""
    NMV, NZMAX = mvm.shape[1], wz.shape[1]
    m, dev = S.shape[0], S.device
    chunk = chunk or max(1, 6400//NMV**2)                         # 256 at five MVs: the work array is chunk x P x NMV x M x NMV
    Sw = S[:, 1:P+1]*wy[:, None, :, None]*mvm[:, None, None, :]
    D = torch.eye(M, dtype=torch.float64, device=dev)-torch.diag(torch.ones(M-1, dtype=torch.float64, device=dev), -1)
    D = torch.kron(D, torch.eye(NMV, dtype=torch.float64, device=dev))
    sup = torch.ones(m, NMV, dtype=torch.float64, device=dev) if supp is None else supp.double()
    sup = LAM_DMC*P*sup.repeat(1, M)
    rel = LAM_REL
    if lam is not None:
        sup, rel = sup*lam, rel*lam
    HX = torch.zeros(m, M*NMV, M*NMV, dtype=S.dtype, device=dev)
    for a in range(0, m, chunk):
        s = Sw[a:a+chunk].double()
        b = s.shape[0]
        A = torch.zeros(b, P, NMV, M, NMV, dtype=torch.float64, device=dev)
        for j, t in enumerate(MT):
            A[:, t:, :, j, :] = s[:, :P-t]
        A = A.reshape(b, P*NMV, M*NMV)
        H = A.transpose(1, 2)@A
        H = H+torch.diag_embed(sup[a:a+chunk]+rel*H.diagonal(dim1=1, dim2=2))
        HX[a:a+chunk] = (D.T@H@D).to(S.dtype)
    zw = torch.tensor(ZDT, dtype=S.dtype, device=dev).sqrt()
    Az = torch.zeros(m, NZMAX, len(ZL), M, NMV, dtype=S.dtype, device=dev)
    for i, t in enumerate(ZL):
        for j, tj in enumerate(MT):
            if tj < t:
                Az[:, :, i, j, :] = Sz[:, t-tj]*(wz[:, :, None]*zw[i])*mvm[:, None, :]
    Az = (Az.reshape(m, NZMAX*len(ZL), M*NMV).double()@D).to(S.dtype)
    return Sw, HX, Az


LU_MAX = 64                      # the teacher's systems up to this many unknowns are solved by LU (the default size has 40: its moves stay bit for bit), larger ones by Cholesky


def _spd(H, g):
    """x of H x = g, H (m, k, k) symmetric positive definite (the move suppression makes it so). The
    larger sizes by Cholesky: the right method for it, and the batched LU of 160 unknowns does not return
    on a threaded MKL (torch 2.15, CPU)."""
    if H.shape[-1] <= LU_MAX:
        return torch.linalg.solve(H, g)
    return torch.cholesky_solve(g[..., None], torch.linalg.cholesky(H))[..., 0]


def spd_solve(H, G):
    """X of H X = G for a batch of symmetric positive definite H (m, k, k) and right sides G (m, k, r), 0 where a
    system has no solution: the law's tables of the teacher's Hessians (aio.law.table), the fits of the network's
    running sums (aio.net.Stats). As `_spd`: up to LU_MAX unknowns by LU, which the default size's tables and
    features were made with; the larger ones by Cholesky. At 48x20 a table has 160 of them (a fit 120 to 132), and
    on the CPU the batched LU of 160 does not return once there are two threads: a builder that draws problems of
    design hung on it, as a reading on the cloud would. A system that is not positive definite to the last bit (its
    sums are kept in single precision) is solved by LU on its own, which does return."""
    if H.shape[-1] <= LU_MAX:
        return torch.linalg.solve_ex(H, G)[0].nan_to_num(0., 0., 0.)
    L, info = torch.linalg.cholesky_ex(H)
    X = torch.cholesky_solve(G, L)
    for i in map(tuple, info.nonzero().tolist()):
        X[i] = torch.linalg.solve_ex(H[i], G[i])[0]
    return X.nan_to_num(0., 0., 0.)


@torch.no_grad()
def dmc_move(Sw, HX, E, lo, hi, Az=None, zl=None, zh=None, iters=8, full=False):
    """First planned position change (m, NMV) of the dynamic-matrix controller (full: all of them, (m, M*NMV),
    the changes planned for the decisions MT from now) for the weighted predicted
    errors E (m, P, NMV), with every planned position inside lo..hi (m, M, NMV; relative to the position
    now) and the zone rows kept inside their limits as far as the tracked errors allow (Az X between zl and
    zh, both (m, NZMAX*len(ZL)), weighted like the cost). Active sets by block pivoting, on the episodes
    that need it: planned positions that leave their limits are held there, held ones whose multiplier has
    the wrong sign are let go again, zone values that leave their limits are pulled back to the limit, and
    the rest is solved again."""
    NMV = E.shape[2]
    m, P = E.shape[0], E.shape[1]
    g = torch.stack([torch.einsum('btai,bta->bi', Sw[:, :P-t], E[:, t:]) for t in MT], 1)
    g = (g-torch.cat((g[:, 1:], torch.zeros_like(g[:, :1])), 1)).reshape(m, -1).double()
    H = HX.double()
    lo, hi = lo.reshape(m, -1).double(), hi.reshape(m, -1).double()
    X = _spd(H, g)
    need = ((X < lo) | (X > hi)).any(-1)
    if Az is not None:
        pz = (Az.double()@X[..., None])[..., 0]
        need = need | ((pz < zl) | (pz > zh)).any(-1)
    i = need.nonzero().flatten()
    if i.numel():
        H, g, lo, hi, x = H[i], g[i], lo[i], hi[i], X[i]
        if Az is not None:
            Az, zl, zh = Az[i].double(), zl[i].double(), zh[i].double()
        fixed = torch.zeros_like(x, dtype=torch.bool)
        act = None
        tol = 1e-9*(g.abs().amax(-1, keepdim=True)+1)
        for _ in range(iters):
            Hc, gc = H, g
            if Az is not None:
                pz = (Az@x[..., None])[..., 0]
                new = (pz < zl) | (pz > zh)
                same = act is not None and bool((new == act).all())
                act = new
                Aa = Az*act[:, :, None]
                Hc = H+Aa.transpose(1, 2)@Aa
                gc = g+(Aa.transpose(1, 2)@torch.where(pz < zl, zl, zh)[..., None])[..., 0]
            else:
                same = True
            out = ((x < lo) | (x > hi)) & ~fixed
            grad = (Hc@x[..., None])[..., 0]-gc                   # on a held position: its multiplier
            back = fixed & (hi-lo > 1e-12) & (((x >= hi) & (grad > tol)) | ((x <= lo) & (grad < -tol)))
            if same and not bool(out.any()) and not bool(back.any()):
                break
            fixed = (fixed | out) & ~back
            xc = torch.where(fixed, torch.minimum(torch.maximum(x, lo), hi), torch.zeros_like(x))
            free = (~fixed).double()
            A = Hc*free[:, :, None]*free[:, None, :]+torch.diag_embed(fixed.double())
            x = _spd(A, torch.where(fixed, xc, gc-(Hc@xc[..., None])[..., 0]))
        X = X.index_copy(0, i, torch.minimum(torch.maximum(x, lo), hi))
    return (X if full else X[:, :NMV]).to(E.dtype)


def labels(Sl, scale):
    """Identification target (m, NMV, NCV*lags): each MV's response on every row at the lags of Sl (LAB_LAGS,
    or a wider set of KNOTS) in packet units (the row's scale per full MV span), asinh-compressed."""
    NCV, NMV = Sl.shape[2:]
    y = Sl/scale[:, None, :, None]
    return torch.asinh(y).permute(0, 3, 2, 1).reshape(y.shape[0], NMV, NCV*Sl.shape[1])


@torch.no_grad()
def pi_tuning(S, p):
    """Lambda (IMC) PI of each MV on its published CV from that pair's step response (FOPDT by the 5 % and 63 % points; lambda = max(tau, 3 theta, 1);
    integrating loops by slope and arrest time max(3 theta, 5); loops whose relative gain is outside
    [1/1.5, 1.5] get three times the lambda, the fastest kept). Returns kc, ti (m, NMV) per decision in
    normalised units and the paired row of each MV."""
    NCV, NMV, ND, NQ, NDQ, NU = sizes(p)
    m, dev, f = S.shape[0], S.device, S.dtype
    row = p['link'][:, :NMV, :].argmax(1)                                   # (m, NMV) paired row of each MV
    s = torch.gather(S, 2, row[:, None, None, :].expand(m, S.shape[1], 1, NMV))[:, :, 0, :]   # (m, lags+1, NMV)
    n = s.shape[1]
    Kss = s[:, -1]
    r = s/torch.where(Kss.abs() > 1e-9, Kss, torch.ones_like(Kss))[:, None]
    first = lambda c: torch.where(c.any(1), c.to(torch.int64).argmax(1), torch.full((m, NMV), n-1, device=dev)).to(f)   # noqa: E731
    t5, t63 = first(r >= .05), first(r >= .632)
    theta = (t5-.5).clamp_min(0.)
    tau = (t63-theta).clamp_min(1.)
    ig = torch.gather(p['integ'][:, :NMV], 1, row) > 0
    kp = (s[:, -1]-s[:, -41])/40
    th_i = (first(s.abs() > .02*s.abs().amax(1, keepdim=True))-1).clamp_min(0.)
    theta = torch.where(ig, th_i, theta)
    lam = torch.where(ig, torch.maximum(3*theta, torch.full_like(theta, 5.)), torch.maximum(torch.maximum(tau, 3*theta), torch.ones_like(tau)))
    # relative gains of the pairs from the final values of the tracked block
    G = torch.gather(S[:, -1], 1, row[:, :, None].expand(-1, -1, NMV)).double()    # row a = paired row of MV a
    act = p['mvm'] > 0
    eye = torch.eye(NMV, dtype=torch.float64, device=dev)
    G = torch.where(act[:, :, None] & act[:, None, :], G, eye)
    Gi, bad = torch.linalg.inv_ex(G)
    lamr = torch.where(bad[:, None] > 0, torch.ones(m, NMV, dtype=torch.float64, device=dev), (G*Gi.transpose(1, 2)).diagonal(dim1=1, dim2=2)).to(f)
    inter = act & ~ig & ((lamr <= 0) | (lamr.clamp_min(1e-12).log().abs() > math.log(1.5)))
    many = inter.sum(1, keepdim=True) > 1
    fastest = torch.where(inter, lam, torch.full_like(lam, math.inf)).argmin(1, keepdim=True)
    slow = inter & many & (torch.arange(NMV, device=dev)[None] != fastest)
    lam = torch.where(slow, 3*lam, lam)
    ok = Kss.abs() > 1e-9
    kc = torch.where(ok, tau/(Kss*(lam+theta)), torch.zeros_like(tau))
    kci = torch.where(kp.abs() > 1e-12, (2*lam+theta)/(kp*(lam+theta)**2), torch.zeros_like(tau))
    kc = torch.where(ig, kci, kc)*p['mvm']
    ti = torch.where(ig, 2*lam+theta, tau)
    return kc, ti, row

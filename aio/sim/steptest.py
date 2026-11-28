"""The step-test DMC as a plant gets one: a test on the plant itself, a model identified from what the test showed,
and the teacher's design on that model (docs/sim.md, 'the step test paid for').

    python -m aio.sim.steptest --length 5760 [--design pi|level|open] [--amp .05] [--calm 8] [--unseen 1440 | --trial 1440]
                               [--windows 16] [--out x.json]

The reference `dmc` of aio.sim.stay (control.Teacher('fixed')) has the exact model of commissioning and has paid
nothing for it. This one starts where a project starts:

  the test    `length` decisions at the start of the stay: every valve in service moves between two positions `amp`
              either side of where it stood, switching at random (a generalised binary noise signal, HOLD decisions
              on average between switches, the valves independent). 'pi': the formula PI stays in service and the
              signal is added to its output (--sp: its targets are moved as well); 'level': only the loops that hold a
              level stay; 'open': nothing else moves the valves. The events of the stay go on (--calm: not during the
              first windows), and the test is scored like any other decision.
  the model   a finite step response of every row to every valve and measured disturbance, from the measurements and
              the valve positions of the test alone (`identify`: regularised least squares on the response's rises
              between KNOTS, kernels that die out at the rates DECAY weighted by marginal likelihood, the noise an
              autoregression found from the residuals), then the same record as low-order curves. Bad values are left
              out by their flag, spikes by their neighbours, an analyser's samples are put back to when they were
              taken, a level is read as its rate, and the record is read on the plant's own clock.
  the design  that response in the place of the exact one of commissioning (Program.install); which reading (the rises
              or the curves), the move suppression and whether the steady-state layer puts its best steady point in
              the targets' place are chosen on another batch (`tune`); from there the teacher's law as `dmc` runs it.
  acceptance  (--unseen) the last decisions of the test are kept from a model of the rest, and that model is tried on
              them: what it leaves of the rows' changes there over the changes themselves (`_unseen`; 1: no better
              than no model). A plant whose model is further off than a threshold stays with the formula PI; the
              threshold is the one that loses least on the other batch (`accept`). The model that goes into service
              is the whole record's.
              (--trial) after the test the formula PI runs alone for `trial` decisions, then the model for as many,
              and each plant keeps the one under which fewer of its rows were outside their published targets and
              limits, as the controller itself can see them (`_outside`). Both stretches are scored.

What it still shares with `dmc` and a controller in the field would not have: the valve and output curves, the
levels' integration constants and the load's time scale are the plant's own, and its prediction starts from the
true value of every row (the measurement noise does not reach it). What `dmc` has beyond that: the response itself,
the steady gains beyond the last lag, and a test that cost nothing.
"""
import argparse
import copy
import json
import math
import time

import torch

from aio import scale as SC
from aio.sim import control as CT, core as C, model as MD, prior as PR, stay as ST

HOLD = 40                        # decisions between two switches of a valve, on average (a third of what the slower pairs take to settle)
KNOTS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 27, 33, 40, 48, 58, 70, 84, 100, 120, 145, 175, 205, 240, 300, 380, 480, 600, 720)
DECAY = (.8, .95, .985, .995)    # per decision: how fast a response's rises die out, the kinds the estimate mixes
AR = 8                           # order of the noise's autoregression
SPIKE = .05                      # a spike is at least this far from its neighbours (of the row's spread over the test)
STARTS = ((1., 1.), (.4, 1.), (2.5, .5))     # the curves' fit is started from the rises' own rate and dead time, and from these multiples of them
KEEP = 4.                        # a pair's curve is kept when its gain is this many of its standard deviations from 0
LAMS = (1., 3., 10., 30.)        # the move suppressions tried (times the teacher's)
READS = ('curve', 'fir')         # the readings of the record a design can be made on
DESIGNS = ('open', 'level', 'pi')            # who else moves the valves during the test: nobody; the formula PI on the levels alone; the formula PI
TUNE_SEED = 78_001               # the batch the suppression is chosen on (not an exam's)
SPANS = (8, 16, 32, 64, 120, 240)            # decisions a row's change is taken over when a model is tried on a stretch it was not found on
ACCEPT = (.5, .6, .7, .8, .9, 1., 1.25, 1.5, 2., math.inf)     # how far off there a model may be and still go into service: the thresholds tried
F = torch.float64


# ---------------------------------------------------------------- the estimate
def _kernels(knots, decay):
    """(len(decay), nb, nb) covariances of a response's rises between the knots, for impulse responses whose
    covariance at lags j, l is rate**max(j, l) (the 'TC' kernel), each scaled to a mean variance of 1."""
    tau = (0,)+tuple(knots)
    A = torch.zeros(len(knots), knots[-1], dtype=F)
    for b in range(len(knots)):
        A[b, tau[b]:tau[b+1]] = 1.
    lag = torch.arange(1, knots[-1]+1, dtype=F)
    out = []
    for lam in decay:
        K = A@torch.pow(torch.tensor(lam, dtype=F), torch.maximum(lag[:, None], lag[None, :]))@A.T
        out.append(K/K.diagonal().mean())
    return torch.stack(out)


def _regressors(x, knots):
    """(n, R, NI*nb): for every input and every stretch between knots, the input's mean over the lags of that
    stretch (lag 1 is the decision the value is seen at), counted from where it stood at the start."""
    n, R, NI = x.shape
    Cp = torch.cat((torch.zeros(n, 1, NI, dtype=F), (x-x[:, :1]).cumsum(1)), 1)        # Cp[m+1] = sum up to m
    t = torch.arange(R)
    tau = (0,)+tuple(knots)
    cols = [(Cp[:, (t-tau[b]+1).clamp_min(0)]-Cp[:, (t-tau[b+1]+1).clamp_min(0)])/(tau[b+1]-tau[b]) for b in range(len(knots))]
    return torch.stack(cols, -1).reshape(n, R, NI*len(knots))


def _em(G, r, zz, N, c, s2, Ks, iters, flat=1e4):
    """Regularised least squares with the kernels' weights c (m, NI, M) and the noise variance s2 (m,) moved towards
    the maximum of the marginal likelihood (EM on the variance components). G = X'X (m, p, p), r = X'z, zz = z'z,
    N the samples; the last parameter is the offset (flat prior). Returns the estimate (m, p), c, s2.
    (By Cholesky throughout: the batched LU of this many unknowns does not return on a threaded MKL.)"""
    m, p = r.shape
    NI, M = c.shape[1:]
    nb = Ks.shape[1]
    eye, eb = torch.eye(p, dtype=F), torch.eye(nb, dtype=F)
    blk = [slice(i*nb, (i+1)*nb) for i in range(NI)]
    for _ in range(max(iters, 1)):
        Kb = torch.einsum('nim,mab->niab', c, Ks)
        Lb = torch.linalg.cholesky(Kb+1e-12*eb)                   # the prior covariance K = L L', block by block
        L = torch.zeros(m, p, p, dtype=F)
        for i, b in enumerate(blk):
            L[:, b, b] = Lb[:, i]
        L[:, -1, -1] = math.sqrt(flat)
        B = L.transpose(1, 2)@G
        A = torch.linalg.cholesky(s2[:, None, None]*eye+B@L)
        mu = (L@torch.cholesky_solve(L.transpose(1, 2)@r[..., None], A))[..., 0]
        if iters:
            W = torch.cholesky_solve(B, A)
            gam = (r-(G@mu[..., None])[..., 0])/s2[:, None]       # K^-1 mu, and K^-1 - K^-1 Sigma K^-1, without the inverse of K
            Gam = (G-B.transpose(1, 2)@W)/s2[:, None, None]
            gi = torch.stack([gam[:, b] for b in blk], 1)
            Gii = torch.stack([Gam[:, b, b] for b in blk], 1)     # (m, NI, nb, nb)
            c = (c+c**2*(torch.einsum('nia,mab,nib->nim', gi, Ks, gi)-torch.einsum('niab,mba->nim', Gii, Ks))/nb).clamp_min(0.)
            res = zz-2*(r*mu).sum(-1)+torch.einsum('na,nab,nb->n', mu, G, mu)+s2*torch.einsum('nab,nba->n', W, L)
            s2 = (res/N.clamp_min(1.)).clamp_min(1e-8)
    return mu, c, s2


def _filter(a, x):
    """x (m, R, ...) through 1 - a_1 q^-1 - ... (a (m, AR)); the first AR entries are left as they are (not used)."""
    y = x.clone()
    for lag in range(1, a.shape[1]+1):
        y[:, lag:] -= a[:, lag-1].view(-1, *([1]*(x.dim()-1)))*x[:, :-lag]
    return y


def _window(ok, order):
    """ok (m, R): whether the sample and the `order` before it all count."""
    w = ok.clone()
    for lag in range(1, order+1):
        w[:, lag:] &= ok[:, :-lag]
    w[:, :order] = False
    return w


def _clock(phi, x, Z, ok, sparse):
    """The record on the plant's own clock (model time: the decisions' time scales phi (n, R) added up; a plant whose
    pace follows its load answers the same way on it at every load): x (n, R, NI), Z and ok (n, R, NO) at whole units of
    it, between the two decisions either side; a row whose samples come one at a time (sparse (n, NO)): every sample
    at the unit nearest to it."""
    n, R = phi.shape
    tau = phi.to(F).cumsum(1)
    tau = tau-tau[:, :1]
    L = int(tau[:, -1].max().floor())+1
    grid = torch.arange(L, dtype=F)[None].expand(n, L).contiguous()
    i1 = torch.searchsorted(tau.contiguous(), grid, right=True).clamp(1, R-1)
    i0 = i1-1
    t0, t1 = tau.gather(1, i0), tau.gather(1, i1)
    w = ((grid-t0)/(t1-t0).clamp_min(1e-9)).clamp(0., 1.)[..., None]
    inside = (grid <= tau[:, -1:])[..., None]
    at = lambda v, i: torch.gather(v, 1, i[:, :, None].expand(-1, -1, v.shape[2]))      # noqa: E731
    xm = at(x, i0)*(1-w)+at(x, i1)*w
    Zm, okm = at(Z, i0)*(1-w)+at(Z, i1)*w, (at(ok, i0) | (w >= 1)) & (at(ok, i1) | (w <= 0)) & inside     # (a decision that carries no weight need not count)
    idx = tau.round().long().clamp(0, L-1)[:, :, None].expand(-1, -1, Z.shape[2])
    cnt = torch.zeros(n, L, Z.shape[2], dtype=F).scatter_add(1, idx, ok.to(F))
    Zs = torch.zeros(n, L, Z.shape[2], dtype=F).scatter_add(1, idx, Z*ok.to(F))/cnt.clamp_min(1.)
    sp = sparse[:, None, :]
    return xm, torch.where(sp, Zs, Zm), torch.where(sp, cnt > 0, okm)


def shape(t, par):
    """A unit step response of second order with a zero and a dead time, at the lags t (..., J): par (..., 4) =
    (log of the rate sig, b, z, dead time). The poles are sig (1 +- sqrt(rho)), rho = 1 - exp(b) (below 0: a complex
    pair); z is the slope at the start over sig (0: none, above: a lead, below: the wrong way first)."""
    sig, b, z, dead = (par[..., i:i+1] for i in range(4))
    sig = torch.exp(sig)
    rho = 1-torch.exp(b)
    rho = torch.where(rho.abs() < 1e-6, torch.full_like(rho, 1e-6), rho)
    sq = torch.sqrt(torch.complex(rho, torch.zeros_like(rho)))*sig
    tp = (t-dead).clamp_min(0.)
    e1, e2 = torch.exp((-sig+sq)*tp), torch.exp((-sig-sq)*tp)
    return 1-((e1+e2)/2+sig*(1-z)*(e1-e2)/(2*sq)).real


def _guess(s1, knots):
    """Where a fit of `shape` to the rises found starts: s1 (m, NI, nb) the response at the knots."""
    tau = torch.tensor(knots, dtype=F)
    K = s1[..., -1:]
    rel = s1/torch.where(K.abs() > 1e-12, K, torch.ones_like(K))
    first = lambda c: torch.where(c.any(-1), tau[c.to(torch.int64).argmax(-1)], tau[-1])      # noqa: E731
    dead = (first(rel >= .1)-1).clamp_min(0.)
    t63 = first(rel >= .63)
    sig = 2.67/(t63-dead).clamp_min(1.)
    return torch.stack((torch.log(sig), torch.full_like(sig, math.log(.75)), torch.zeros_like(sig), dead), -1)


def _curves(G, r, zz, N, par, knots, act, P, s2, iters=300, keep=KEEP):
    """A low-order curve (`shape`) for every input at once in the place of the rises: the same criterion the rises were
    found by (the normal equations of the whitened record G (m, p, p), r (m, p), zz, N, the last parameter the offset;
    plus the prior of the rises, P (m, NI, nb, nb) its precision and s2 (m,) the noise variance), over curves only. The
    shapes by gradient steps from `par` (m, NI, 4), the gains solved at every step; then the pairs whose gain is not
    `keep` standard deviations from 0 are dropped and the others' gains solved again. Returns the gains (m, NI), par,
    and the criterion reached."""
    m, NI = par.shape[:2]
    nb = len(knots)
    q = NI*nb
    tau = torch.tensor(knots, dtype=F)
    Gb, go, goo = G[:, :q, :q].reshape(m, NI, nb, NI, nb), G[:, :q, -1].reshape(m, NI, nb), G[:, -1, -1]
    rb, ro = r[:, :q].reshape(m, NI, nb), r[:, -1]
    slow = 10.7/float(N.max())                                   # no curve slower than the record can show: its time constants within a quarter of it
    lo = torch.tensor([math.log(slow), -9., -8., 0.], dtype=F)
    hi = torch.tensor([math.log(2.), 3.2, 8., 2.*PR.DMAX], dtype=F)

    def solve(par, on):
        g = shape(tau, par)*on[..., None]
        u = torch.cat((g[..., :1], g[..., 1:]-g[..., :-1]), -1)   # the unit curve's rises between the knots
        A = torch.zeros(m, NI+1, NI+1, dtype=F)
        A[:, :NI, :NI] = torch.einsum('mia,miajb,mjb->mij', u, Gb, u)+torch.diag_embed(s2[:, None]*torch.einsum('mia,miab,mib->mi', u, P, u))
        A[:, :NI, -1] = A[:, -1, :NI] = (u*go).sum(-1)
        A[:, -1, -1] = goo
        A = A+torch.diag_embed(torch.cat((1e-9*goo[:, None]+(1-on)*goo[:, None], torch.zeros(m, 1, dtype=F)), 1))
        b = torch.cat(((u*rb).sum(-1), ro[:, None]), 1)
        K = torch.linalg.solve(A, b)
        return K, A, zz-(b*K).sum(-1)

    par = par.clone().requires_grad_(True)
    opt = torch.optim.Adam([par], lr=.05)
    on = act.to(F)
    with torch.enable_grad():
        for _ in range(iters):
            K, A, left = solve(par, on)
            opt.zero_grad()
            (left/zz.clamp_min(1e-12)).sum().backward()
            opt.step()
            with torch.no_grad():
                par.copy_(torch.minimum(torch.maximum(par, lo), hi))
    par = par.detach()
    K, A, left = solve(par, on)
    sd = torch.sqrt(s2[:, None]*torch.linalg.inv(A).diagonal(dim1=1, dim2=2)[:, :NI].clamp_min(0.))
    on = on*(K[:, :NI].abs() >= keep*sd)
    K, A, left = solve(par, on)
    return K[:, :NI]*on, par, left


@torch.no_grad()
def identify(x, z, ok, avg=None, knots=KNOTS, decay=DECAY, ar=AR, em=(6, 14, 6), chunk=16, say=None):
    """The response of every output to every input from a record of R decisions. x (n, R, NI) the inputs as they stood
    when the outputs z (n, R, NO) were seen; ok (n, R, NO) the samples that count; avg (n, NO): where above 1, the
    output is a mean over that many decisions up to its own (an analyser on a level: the regressors are averaged the
    same way). Returns th (n, NO, NI, nb), the response's rise over every stretch between knots, and what else was
    found: of the noise, 'sd' (n, NO) of the innovation in the outputs' units, 'ar' (n, NO, AR), 'used' (n, NO) samples;
    and the same record read as low-order curves (the industrial step after the rises: `shape` up to the last knot and
    flat from there, by the same criterion): 'K' (n, NO, NI) the gains (0: no curve kept for the pair), 'par' (n, NO, NI, 4)."""
    x, z = x.to(F), z.to(F)
    n, R, NI = x.shape
    NO, nb, M = z.shape[2], len(knots), len(decay)
    Ks = _kernels(knots, decay)
    q = NI*nb
    th = torch.zeros(n, NO, NI, nb, dtype=F)
    info = dict(sd=torch.zeros(n, NO, dtype=F), ar=torch.zeros(n, NO, ar, dtype=F), used=torch.zeros(n, NO, dtype=F),
                K=torch.zeros(n, NO, NI, dtype=F), par=torch.zeros(n, NO, NI, 4, dtype=F))
    for a0 in range(0, n, chunk):
        sl = slice(a0, a0+chunk)
        xs = (x[sl]-x[sl, :1]).std(1)
        act = xs > 1e-9
        xs = torch.where(act, xs, torch.ones_like(xs))
        X = _regressors(x[sl]/xs[:, None, :], knots)
        m = X.shape[0]
        X1 = torch.cat((X, torch.ones(m, R, 1, dtype=F)), -1)
        for o in range(NO):
            w0 = ok[sl, :, o].clone()
            rows = (w0.sum(1) > 4*ar+8).nonzero().flatten()
            if not rows.numel():
                continue
            Xo = X1[rows]
            if avg is not None and bool((avg[sl, o][rows] > 1).any()):      # the mean of the regressors over the decisions the sample spans
                D = avg[sl, o][rows].clamp_min(1)
                cs = torch.cat((torch.zeros(len(rows), 1, q+1, dtype=F), Xo.cumsum(1)), 1)
                t = torch.arange(R)[None, :]
                lo = (t+1-D[:, None]).clamp_min(0)
                Xo = (cs[:, 1:]-torch.gather(cs, 1, lo[:, :, None].expand(-1, -1, q+1)))/D[:, None, None]
            w, zo = w0[rows], z[sl, :, o][rows]
            wf = w.to(F)
            mean = (zo*wf).sum(1)/wf.sum(1)
            sd = torch.sqrt((((zo-mean[:, None])**2)*wf).sum(1)/wf.sum(1)).clamp_min(1e-12)
            zn = (zo-mean[:, None])/sd[:, None]
            # spikes: a sample far from the middle one of itself and its two neighbours, against what the row's noise does there
            # (a step or a response stays; a spike is gone again)
            near = torch.stack((torch.cat((zn[:, :1], zn[:, :-1]), 1), zn, torch.cat((zn[:, 1:], zn[:, -1:]), 1)), -1).median(-1).values
            three = w & torch.cat((w[:, :1], w[:, :-1]), 1) & torch.cat((w[:, 1:], w[:, -1:]), 1)
            dev = torch.where(three, (zn-near).abs(), torch.full_like(zn, float('nan')))
            q9 = torch.nanquantile(dev, torch.tensor([.9, .99], dtype=F), dim=1).nan_to_num(0.)      # never more than 1 in 100: a spike is rare
            w = w & ~(dev > torch.maximum(6*q9[0], q9[1]).clamp_min(SPIKE)[:, None])
            c = None
            s2 = torch.full((len(rows),), .25, dtype=F)
            # the noise: white to start, then an autoregression of what is left; not for samples that come one at a time
            # (an analyser's): those stay as they are
            dense = _window(w, ar).sum(1) > .5*w.sum(1)
            a = torch.zeros(len(rows), ar, dtype=F)
            for rnd, iters in enumerate(em):
                Xf, zf, wq = _filter(a, Xo), _filter(a, zn), torch.where(dense[:, None], _window(w, ar), w)
                wv = wq.to(F)
                Xw = Xf*wv[:, :, None]
                G, r, zz, N = Xw.transpose(1, 2)@Xf, (Xw*zf[:, :, None]).sum(1), (zf**2*wv).sum(1), wv.sum(1)
                if c is None:                                     # to start: every input in service the same share of the output's variance, every kind of kernel the same
                    g = torch.stack([torch.einsum('nab,mba->nm', G[:, i*nb:(i+1)*nb, i*nb:(i+1)*nb], Ks) for i in range(NI)], 1)/N[:, None, None]
                    c = torch.where(g > 1e-12, 1/(M*act[rows].sum(1).clamp_min(1)[:, None, None]*g.clamp_min(1e-12)), torch.zeros_like(g))
                mu, c, s2 = _em(G, r, zz, N, c, s2, Ks, iters)
                if rnd < len(em)-1:
                    e = zn-(Xo*mu[:, None, :]).sum(-1)            # what is left, as measured
                    wa = _window(w, ar).to(F)
                    E = torch.stack([torch.cat((torch.zeros(len(rows), lag, dtype=F), e[:, :-lag]), 1) for lag in range(1, ar+1)], -1)*wa[:, :, None]
                    H = E.transpose(1, 2)@E
                    H = H+1e-6*H.diagonal(dim1=1, dim2=2).mean(-1).clamp_min(1e-12)[:, None, None]*torch.eye(ar, dtype=F)
                    a = torch.linalg.solve(H, (E*(e*wa)[:, :, None]).sum(1))*dense[:, None]
            # the same record as low-order curves, from several starts
            g0 = _guess(mu[:, :q].view(-1, NI, nb).cumsum(-1), knots)
            P = torch.linalg.inv(torch.einsum('nim,mab->niab', c, Ks)+1e-9*torch.eye(nb, dtype=F))
            best = None
            for fr, fd in STARTS:
                g1 = g0.clone()
                g1[..., 0], g1[..., 3] = g1[..., 0]+math.log(fr), g1[..., 3]*fd
                got = _curves(G, r, zz, N, g1, knots, act[rows], P, s2)
                best = got if best is None else tuple(torch.where((got[2] < best[2]).view(-1, *([1]*(u.dim()-1))), u, v) for u, v in zip(got, best))
            idx = torch.arange(a0, min(a0+chunk, n))[rows]
            info['K'][idx, o], info['par'][idx, o] = best[0]*sd[:, None]/xs[rows], best[1]
            th[idx, o] = mu[:, :q].view(-1, NI, nb)*sd[:, None, None]/xs[rows][:, :, None]
            info['sd'][idx, o], info['ar'][idx, o], info['used'][idx, o] = torch.sqrt(s2)*sd, a, w.sum(1).to(F)
        if say is not None:
            say(min(a0+chunk, n))
    return th, info


def curve(th, knots=KNOTS):
    """(n, knots[-1]+1, NO, NI) the step response at every lag from its rises between the knots (a straight line in between)."""
    tau = torch.tensor((0,)+tuple(knots), dtype=F)
    j = torch.arange(knots[-1]+1, dtype=F)
    B = ((j[:, None]-tau[None, :-1])/(tau[1:]-tau[:-1])[None, :]).clamp(0., 1.)       # (lags+1, nb)
    return torch.einsum('noib,jb->njoi', th, B)


def _unseen(x, Z, ok, th, knots, late, wt, avg=None, spans=SPANS, chunk=16):
    """How far a model is off on a stretch of the record it was not found on, against predicting nothing (the
    acceptance of commissioning). Over `late` (n, R) and for every span h: the change of every row from its mean over h
    decisions to its mean over the next h, as measured and as the model has it from the inputs; the squares of what
    the model leaves of that change over the squares of the change itself, every row in the units wt (n, NO, spans)
    puts it in. th (n, NO, NI, nb) the model's rises between the knots. Returns (n, spans); 1: no better than no model."""
    n, R, NI = x.shape
    NO, nb = Z.shape[2], len(knots)
    out = torch.zeros(n, len(spans), dtype=F)
    front = lambda f: torch.cat((torch.zeros_like(f[:, :1]), f.cumsum(1)), 1)        # noqa: E731
    for a0 in range(0, n, chunk):
        sl = slice(a0, a0+chunk)
        yh = torch.einsum('nrq,noq->nro', _regressors(x[sl], knots), th[sl].reshape(-1, NO, NI*nb))
        if avg is not None:                                       # a mean over the decisions a sample spans, as it was found
            D = avg[sl].clamp_min(1)[:, None, :]
            lo = (torch.arange(R)[None, :, None]+1-D).clamp_min(0)
            cs = front(yh)
            yh = (cs[:, 1:]-torch.gather(cs, 1, lo.expand(-1, -1, NO)))/D
        v = (ok[sl] & late[sl, :, None]).to(F)
        Ce, Cz, Cv = front((Z[sl]-yh)*v), front(Z[sl]*v), front(v)
        for i, h in enumerate(spans):
            if R < 2*h+1:
                continue
            se, sz, sv = (C[:, h:]-C[:, :-h] for C in (Ce, Cz, Cv))        # sums over the h decisions up to each
            w2 = wt[sl, :, i][:, None, :]**2*((sv[:, h:] > 0) & (sv[:, :-h] > 0))
            d = lambda q: q[:, h:]/sv[:, h:].clamp_min(1.)-q[:, :-h]/sv[:, :-h].clamp_min(1.)      # noqa: E731
            out[sl, i] = (d(se)**2*w2).sum((1, 2))/(d(sz)**2*w2).sum((1, 2)).clamp_min(1e-30)
    return out


# ---------------------------------------------------------------- the controller
class StepTest:
    """A controller for control.rollout that lives a stay of its own (stay.references, `extra`): the test for the first
    `length` decisions, then the teacher's law on the model the test gave. design: who else moves the valves during the
    test (DESIGNS); lam: the move suppression times this; model: one found before on these plants (`self.model` after
    the test), not to be found again."""

    def __init__(self, length=5760, amp=.05, design='pi', hold=HOLD, lam=None, seed=3, model=None, say=None, read='curve', sp=0., trade=True,
                 unseen=0, accept=None, trial=0, keep=None, meter=None):
        """hold: decisions between two switches of a valve on average. sp ('pi'): the loops' targets are moved as well, this
        many bands either side, switching half as often (a loop takes a move of its valve back; a move of its target stays).
        read: which reading of the record goes into the design: 'curve', the low-order curves (what goes into an
        industrial controller), or 'fir', the rises as found. trade: whether the steady-state layer puts its best steady
        point in the targets' place (Program.install). unseen: the last so many decisions of the test are kept from a
        model of the rest, which is tried on them (`_unseen`); accept: a plant whose model is further off there than this
        stays with the formula PI (None: every model goes into service; the model in service is the whole record's).
        trial: the acceptance as a plant does it: after the test the formula PI alone for so many decisions, then the model
        for as many, and a plant on which more rows were outside their band or limit under the model (as the packet
        shows them: `_outside`) goes back to the formula PI for good. keep: 'model' or 'pi', all the plants one way
        whatever the trial showed (what the others would have had). model: a dict that is empty is filled by the first
        controller that finds the model (several controllers on the same plants and the same test share one). meter: the
        prediction of the controller in service is corrected from the instruments alone (core.Program.set_teacher)."""
        assert design in DESIGNS and read in READS and (accept is None or unseen) and not (trial and accept is not None) and keep in (None, 'model', 'pi')
        self.length, self.amp, self.design, self.hold, self.lam, self.seed, self.model, self.say = length, amp, design, hold, lam, seed, model, say
        self.read, self.sp, self.trade, self.unseen, self.accept, self.trial, self.keep = read, sp, trade, unseen, accept, trial, keep
        self.t, self.kept, self.back, self.meter = 0, None, None, meter

    def reset(self, prog):
        n, NMV = prog.n, prog.nmv
        g = torch.Generator().manual_seed(self.seed)
        flip = torch.rand(n, self.length, NMV, generator=g) < 1/self.hold
        first = torch.where(torch.rand(n, 1, NMV, generator=g) < .5, -1., 1.)
        self.levels = (first*torch.where(flip, -1., 1.).cumprod(1)).to(prog.device, prog.dtype)
        flip = torch.rand(n, self.length, NMV, generator=g) < .5/self.hold
        first = torch.where(torch.rand(n, 1, NMV, generator=g) < .5, -1., 1.)
        self.targets = (first*torch.where(flip, -1., 1.).cumprod(1)).to(prog.device, prog.dtype)
        self.t, self.rec, self.d, self.q, self.center, self.back, self.seen = 0, [], None, None, None, None, [0., 0.]
        self.off = torch.zeros(n, NMV, dtype=prog.dtype, device=prog.device)
        self.pi = CT.PI()
        self.pi.reset(prog)
        # the plant as commissioned, for `ruler`: its labels, and what turns a response into their units
        c, z = prog.c, torch.zeros(n, dtype=torch.long, device=prog.device)
        P = prog.params(z)
        vs = torch.where(c['integ'] > 0, PR.hinv(prog.start['tgt'], P['wa'])[0], prog.start['v'])
        self.w0 = {q: v.detach().cpu().to(F) for q, v in dict(
            lab=prog.lab[:, 0], gam=P['gam'], un=prog.start['un'], scale=prog.task(z)['scale'], hp=PR.hprime(vs, P['wa']), prc=c['prc'], mvm=c['mvm'], link=c['link'],
            integ=c['integ'], tint=c['tint'], lam=c['lam'], Fr=prog.Fr[:, 0], phi=prog.phis[:, 0]).items()}

    def resume(self, prog):
        if self.t > self.length+self.trial and self.model:
            self._install(prog)

    def restart(self, f):
        self.pi.restart(f)

    def release(self, prog):
        """The exact model of commissioning back in its place (the other references of these windows run on it)."""
        if self.kept is not None:
            for q, v in self.kept.items():
                getattr(prog, q)[:, 0] = v
            self.kept = None

    def _install(self, prog):
        if self.kept is None:
            self.kept = {q: getattr(prog, q)[:, 0].clone() for q in ('Fr', 'phis', 'Sw', 'HX', 'Az', 'Gs', 'Gz', 'zs', 'trade')}
        m = self.model[self.read]
        prog.install(m['Fr'], self.model['phi'], 0, self.lam, m['end'], self.trade)
        prog.set_teacher('fixed', False, self.meter)

    @torch.no_grad()
    def __call__(self, prog, st, k, obs):
        t, L, T = self.t, self.length, self.trial
        self.t += 1
        if t <= L+T:
            self._record(prog, st, k)
        if t < L:
            return self._move(prog, st, k, obs, t)
        if t < L+T:                                               # the trial's first half: the formula PI alone
            self.seen[0] = self.seen[0]+self._outside(prog, st)
            return self.pi(prog, st, k, obs)
        if t == L+T:
            self._handover(prog, st, k)
        if T and t < L+2*T:                                       # its second half: the model
            self.seen[1] = self.seen[1]+self._outside(prog, st)
        elif T and t == L+2*T:
            self.back = self.seen[1] > self.seen[0] if self.keep is None else torch.full_like(self.seen[0], self.keep == 'pi', dtype=torch.bool)
        a = C.teacher(prog, st, k)
        if self.accept is not None or T:                          # the plants whose model was not accepted: the formula PI goes on alone
            api = self.pi(prog, st, k, obs)                       # (its loops are kept up to date on every plant)
            if self.back is not None:
                a = torch.where(self.back[:, None], api, a)
        return a

    @staticmethod
    def _outside(prog, st):
        """(n,) rows outside their band or limit as the packet shows them: the measurements against the published targets and limits."""
        c, y = prog.c, st['y']
        e = (y-st['target']).abs() > c['bandf']
        z = (y < st['lo'][:, 0]) | (y > st['hi'][:, 0])
        return ((e & (c['trk'] > 0)) | (z & (c['zon'] > 0))).sum(1).to(F)

    def _record(self, prog, st, k):
        """What the packet shows at this decision (the measurements with their flags, the valve positions), and what
        the prediction's bookkeeping follows (core.step): the flows and disturbances as they changed in the decision before."""
        c = prog.c
        P = prog.params(prog.seg_at(k), k)
        g = P['gam']
        d = st['d']
        q = torch.where(c['mvm'] > 0, st['qm'], torch.zeros_like(st['qm']))     # the flows as the prediction took them in (a valve's characteristic that changed: with no move)
        self.rec.append(dict(y=st['y'], bad=st['bad'] > 0, age=st['age'], wa=P['wa'],
                             dq=q-(C._flow(st['prev_u'], g, c['mvm']) if self.q is None else self.q),
                             dd=torch.zeros_like(d) if self.d is None else d-self.d, phi=C._phi(st['xv'], d, c)))
        self.d, self.q = d, q

    def _move(self, prog, st, k, obs, t):
        c = prog.c
        if self.center is None:
            self.center = st['u'].clone()
        want = self.amp*self.levels[:, t]
        free = ((self.center+want-st['u'])/c['slew']).clamp(-1., 1.)*c['mvm']
        if self.design == 'open':
            return free
        if self.sp:
            row = prog.pi['row']
            st = dict(st, target=st['target'].scatter_add(1, row, self.sp*torch.gather(c['bandf'], 1, row)*self.targets[:, t]*c['mvm']))
        api = self.pi(prog, st, k, obs)
        a = (api+(want-self.off)/c['slew']).clamp(-1., 1.)*c['mvm']
        if self.design == 'level':                                # only the valves whose loop holds a level stay with it
            a = torch.where(torch.gather(c['integ'], 1, prog.pi['row']) > 0, a, free)
        self.off = self.off+(a-api)*c['slew']
        return a

    def _handover(self, prog, st, k):
        if not self.model:
            found = self.find(prog)
            if self.model is None:
                self.model = found
            else:
                self.model.update(found)
        if self.accept is not None:
            self.back = (self.model['unseen'][self.read] > self.accept).to(prog.device)
        self._install(prog)
        # what the moves made so far have yet to do, by the model that is now in place
        NCV, NMV, ND, NQ, NDQ, NU = prog.dims
        f, dev = prog.dtype, prog.device
        Fr = prog.Fr[:, 0]
        lag = torch.arange(MD.LAGS+1, device=dev, dtype=f)[None]
        ffm = torch.tensor([1.]*ND+[0.]*NQ, dtype=f, device=dev)
        tF = torch.zeros(prog.n, MD.LAGS+1, prog.nr, dtype=f, device=dev)
        for r in self.rec[-int(MD.LAGS/.4)-1:]:
            tF = MD.lerp_at(tF+torch.einsum('bnij,bj->bni', Fr, torch.cat((r['dq'], r['dd']*ffm), 1)), lag+r['phi'][:, None])
        st['tF'] = tF
        if prog.meter is not None:                                # the estimator starts on the prediction that is now in place
            st.update(C.meter_start(prog, st, prog.params(prog.seg_at(k), k)['wa']))
        self.rec = None

    @torch.no_grad()
    def find(self, prog):
        """The model from the record of the test, in both readings: {'phi' (n,) the test's mean time scale, 'noise', and
        under 'fir' and 'curve': 'Fr' (n, LAGS+1, nr, nu) in the place of model.response's (on the plant's own clock),
        'end' (n, nr, NMV) where the responses end (the last knot), 'S' (n, LAGS+1, NCV, NI) the response in decisions
        of the plant as commissioned; with `unseen`, 'unseen': per reading (n,) how far the model of the record without
        its last stretch is off on that stretch (`_unseen`, the mean over SPANS), and 'spans' the same per span}."""
        NCV, NMV, ND, NQ, NDQ, NU = prog.dims
        c, n = prog.c, prog.n
        cpu = lambda q: torch.stack([r[q] for r in self.rec[:self.length+1]], 1).cpu()        # noqa: E731  (the test; a trial's first half is not read)
        Y, bad, age, wa = cpu('y').to(F), cpu('bad'), cpu('age'), cpu('wa').to(F)
        R = Y.shape[1]
        cc = {q: c[q].cpu() for q in ('prc', 'dvr', 'dvsel', 'integ', 'tint', 'lam', 'vc', 'an_T', 'mvm')}
        V = PR.hinv(Y, wa)[0]
        # an analyser's value is the sample of T decisions before the tick that shows it: put back to when it was taken
        T = cc['an_T'].long()
        t = torch.arange(R)[None, :, None]
        at = t+T[:, None, :]
        seen = (at < R) & ~torch.gather(bad, 1, at.clamp_max(R-1)) & ((T[:, None, :] == 0) | (torch.gather(age, 1, at.clamp_max(R-1)) == T[:, None, :]))
        Vs = torch.gather(V, 1, at.clamp_max(R-1))
        # a level is read as its rate (over the decisions between two samples)
        ig = (cc['integ'] > 0)[:, None, :]
        D = T.clamp_min(1)
        back = (t-D[:, None, :]).clamp_min(0)
        prev, had = torch.gather(Vs, 1, back), torch.gather(seen, 1, back) & (t >= D[:, None, :])
        mid = torch.where(D[:, None, :] > 1, (Vs+prev)/2, prev)
        rate = cc['tint'][:, None, :]*(Vs-prev)/D[:, None, :]-cc['lam'][:, None, :]*(mid-cc['vc'][:, None, :])
        Z = torch.where(ig, rate, Vs)
        ok = seen & torch.where(ig, had & (Vs > -.45) & (Vs < 1.45) & (prev > -.45) & (prev < 1.45), torch.ones_like(seen)) & (cc['prc'] > 0)[:, None, :]
        avg = torch.where((cc['integ'] > 0) & (T > 1), D, torch.ones_like(D))
        # the inputs: the valve flows as the moves left them, and the measured disturbances as their rows show them
        dv = [(Y*((cc['dvr'] > 0) & (cc['dvsel'] == j))[:, None, :]).sum(-1) for j in range(ND)]
        x = torch.cat((cpu('dq').to(F).cumsum(1), torch.stack(dv, -1)), -1)
        phis = cpu('phi').to(F)
        x, Z, ok = _clock(phis, x, Z, ok, T > 0)                    # on the plant's own clock
        kn = tuple(q for q in KNOTS if q <= max(MD.LAGS, R//4))    # a response no longer than a quarter of the test (longer ones take the noise's slow part for the plant's)
        dk = DECAY if kn[-1] > MD.LAGS else DECAY[:-1]
        th, noise = identify(x, Z, ok, avg, kn, dk, say=self.say)
        nr, out = prog.nr, dict(phi=phis.mean(1), noise=noise)
        if self.unseen:                                           # the acceptance: a model of the record without its last stretch, tried on that stretch
            tau = phis.cumsum(1)
            late = torch.arange(x.shape[1], dtype=F)[None] >= (tau[:, R-1-self.unseen]-tau[:, 0])[:, None]
            tha, na = identify(x, Z, ok & ~late[:, :, None], avg, kn, dk, say=self.say)
            sh = shape(torch.tensor(kn, dtype=F), na['par'])
            thc = na['K'][..., None]*torch.cat((sh[..., :1], sh[..., 1:]-sh[..., :-1]), -1)
            w0 = self.w0                                           # every row in its band: a level's rate as what it does to the level over the span
            wt = (w0['hp']/w0['scale'])[:, :, None]*torch.where((w0['integ'] > 0)[:, :, None], torch.tensor(SPANS, dtype=F)/w0['tint'][:, :, None], torch.ones(1, dtype=F))
            sp = {q: _unseen(x, Z, ok, t, kn, late, wt, avg) for q, t in (('fir', tha), ('curve', thc))}
            out.update(spans=sp, unseen={q: v.mean(1) for q, v in sp.items()})
        j = torch.arange(MD.LAGS+1, dtype=F)
        dec = j[None]*self.w0['phi'][:, None]                      # the decisions of the plant as commissioned, on the clock (for `ruler`)
        S = curve(th, kn)                                         # (n, lags+1, NCV, NMV+ND)
        Fr = torch.zeros(n, MD.LAGS+1, nr, NU, dtype=F)
        Fr[..., :NMV+ND] = S[:, :MD.LAGS+1, :nr]
        out['fir'] = dict(Fr=Fr, end=S[:, -1, :nr, :NMV].clone(), S=MD.lerp_at(S, dec))
        K, par = noise['K'], noise['par']
        Fr = torch.zeros(n, MD.LAGS+1, nr, NU, dtype=F)
        Fr[..., :NMV+ND] = (K[:, :nr, :, None]*shape(j, par[:, :nr])).permute(0, 3, 1, 2)
        out['curve'] = dict(Fr=Fr, end=(K*shape(torch.tensor([float(kn[-1])], dtype=F), par)[..., 0])[:, :nr, :NMV],
                            S=(K[..., None]*shape(dec[:, None, None, :].clamp_max(kn[-1]), par)).permute(0, 3, 1, 2))
        return out


# ---------------------------------------------------------------- how good the model is, how much suppression it wants
@torch.no_grad()
def ruler(w, S):
    """The unexplained share of the identification labels of the plant as commissioned (asinh of the response in
    packet units at model.LAB_LAGS: the measure of the network's readout), own pairs and cross pairs, over the batch
    and for the lags up to 16, 32-64 and 128-240, and its quartiles over the plants; 'rise_*': the errors of the own
    pairs' rises between the labels' lags. w: StepTest.w0; S (n, lags+1,
    rows, NMV+): a response in decisions, every row before its integrator and output curve per unit of valve flow
    (a model's 'S'; the exact one, model.lerp_at(w['Fr'], j*w['phi']), reads 0)."""
    n, NMV = w['un'].shape
    NCV = w['scale'].shape[1]
    lab = w['lab'].view(n, NMV, NCV, MD.NLAB).permute(0, 2, 1, 3)                   # (n, NCV, NMV, lags)
    ll, nr = list(MD.LAB_LAGS), S.shape[2]
    g = S[..., :NMV].to(F)
    j = torch.arange(g.shape[1], dtype=F)[None, :, None, None]
    rho = (1+w['lam']/w['tint'])[:, None, :nr, None]
    lev = rho**j*torch.cumsum(g/rho**j, 1)/w['tint'][:, None, :nr, None]             # a level adds up its rate
    v = torch.where((w['integ'] > 0)[:, None, :nr, None], lev, g)[:, ll].permute(0, 2, 3, 1)
    slope = (w['gam']*w['un'].clamp_min(1e-6)**(w['gam']-1)*w['mvm'])[:, None, :, None]
    est = torch.zeros_like(lab)
    est[:, :nr] = torch.asinh(v*(w['hp']*w['prc'])[:, :nr, None, None]*slope/w['scale'][:, :nr, None, None])
    own = (w['link'] > 0) & (w['prc'] > 0)[:, :, None] & (w['mvm'] > 0)[:, None, :]
    cross = ~(w['link'] > 0) & (w['prc'] > 0)[:, :, None] & (w['mvm'] > 0)[:, None, :]
    out = {}
    for q, mk in (('own', own), ('cross', cross)):
        for name, s in (('all', slice(0, 9)), ('to16', slice(0, 5)), ('32_64', slice(5, 7)), ('128_240', slice(7, 9))):
            out[f'{q}_{name}'] = round(float((((est-lab)[..., s]**2)*mk[..., None]).sum()/((lab[..., s]**2)*mk[..., None]).sum().clamp_min(1e-30)), 4)
        e, en = (((est-lab)**2)*mk[..., None]).sum((1, 2, 3)), ((lab**2)*mk[..., None]).sum((1, 2, 3))
        out[f'{q}_plants'] = [round(float(x), 3) for x in torch.quantile((e/en.clamp_min(1e-30))[en > 0], torch.tensor([.25, .5, .75, .9], dtype=F))]
    # the rises between the labels' lags, own pairs: how far off, as a share of the largest value the pair's response takes (median, 9 in 10)
    raw, got = torch.sinh(lab), torch.sinh(est)
    rise = lambda x: torch.diff(x, dim=-1, prepend=torch.zeros_like(x[..., :1]))      # noqa: E731
    top = raw.abs().amax(-1, keepdim=True)
    rel = ((rise(got)-rise(raw)).abs()/top.clamp_min(1e-12))[own & (top[..., 0] > 0)]
    for name, sl in (('to8', slice(0, 4)), ('after8', slice(4, 9))):
        out[f'rise_{name}'] = [round(float(x), 3) for x in torch.quantile(rel[:, sl].flatten(), torch.tensor([.5, .9], dtype=F))] if rel.numel() else None
    return out


def tune(progs, ctl, lams=LAMS, reads=READS, trades=(True, False), say=None):
    """Which reading of the record, how much move suppression and whether the steady-state layer trades: `progs`, the
    windows of a stay on a batch that is not an exam's (a list); ctl the controller. The test is run once; from the
    window of the handover on, every (read, lam, trade) goes on from the same state with the same record. Returns
    {(read, lam, trade): (n,) decisions outside from that window on}; with ctl.unseen also 'pi', the same of the
    formula PI going on alone from the handover (a plant whose model is not accepted), and 'unseen', 'spans' (`find`):
    what `accept` chooses the threshold on."""
    w0 = ctl.length//PR.HORIZON
    st = None
    for prog in progs[:w0]:
        st = CT.rollout(prog, ctl, st=st)['st']
        ctl.release(prog)
    out, model = {}, None
    for key in ((r, l, t) for r in reads for l in lams for t in trades):
        c2 = copy.deepcopy(ctl)
        (c2.read, c2.lam, c2.trade), c2.model = key, model
        s, tot = st, 0.
        for prog in progs[w0:]:
            r = CT.rollout(prog, c2, st=s)
            c2.release(prog)
            s, tot = r['st'], tot+r['score'].cpu()
        model, out[key] = c2.model, tot
        if say is not None:
            say(key, float(tot.mean()))
    if ctl.unseen:
        c2 = copy.deepcopy(ctl)
        c2.model, c2.accept = model, -1.
        s, tot = st, 0.
        for prog in progs[w0:]:
            r = CT.rollout(prog, c2, st=s)
            c2.release(prog)
            s, tot = r['st'], tot+r['score'].cpu()
        out.update(pi=tot, unseen=model['unseen'], spans=model['spans'])
    return out


def accept(got, grid=ACCEPT):
    """The setting and the acceptance threshold that lose the least on the tuning batch (`tune`'s result with
    ctl.unseen), a plant whose model is further off than the threshold on the unseen stretch taking the formula PI's
    count: ((read, lam, trade), threshold, the count per plant). Of thresholds that part the plants the same way, the
    middle one."""
    best = None
    for q, v in got.items():
        if not isinstance(q, tuple):
            continue
        for th in grid:
            per = torch.where(got['unseen'][q[0]] > th, got['pi'], v)
            tot = float(per.sum())
            if best is None or tot < best[0]-1e-9:
                best = [tot, q, [th], per]
            elif abs(tot-best[0]) <= 1e-9 and q == best[1]:
                best[2].append(th)
    return best[1], best[2][(len(best[2])-1)//2], best[3]


def _windows(a, n, seed, count):
    """The windows of the exam; --calm: the first so many of them without an event."""
    if not a.calm:
        return ST.exam(a.exam, n=n, seed=seed, count=count)
    e = {q: v for q, v in ST.EXAMS[a.exam].items() if q != 'read'}
    assert 'plants' not in e
    menu = e.pop('menu', None)
    return ST.windows(n, seed, count, lambda w: ST.QUIET if w < a.calm else ST.menu_of(menu, w), e.pop('nseg', PR.NSEG), **e)


def _tap(progs, m):
    for p in progs:
        for x in (p.P, p.T, p.seg_k, p.kinds, p.ramp):
            m.update(x.detach().cpu().numpy().tobytes())
        yield p


def main():
    import hashlib
    from pathlib import Path
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exam', default='usual')
    ap.add_argument('--n', type=int, default=ST.EXAM_N)
    ap.add_argument('--windows', type=int, default=16)
    ap.add_argument('--length', type=int, default=5760, help='decisions of the test')
    ap.add_argument('--amp', type=float, default=.05, help='how far either side of its position a valve is moved (of its full travel)')
    ap.add_argument('--design', default='pi', choices=DESIGNS)
    ap.add_argument('--sp', type=float, default=0., help="'pi': the loops' targets are moved too, this many bands either side")
    ap.add_argument('--calm', type=int, default=0, help='the first so many windows have no event (the wander and the noise go on): a test in a calm stretch; another batch than the exam\'s')
    ap.add_argument('--hold', type=float, default=HOLD, help='decisions between two switches of a valve, on average')
    ap.add_argument('--read', default=','.join(READS), help='the reading of the record the design is made on: one, or several to choose from on the tuning batch')
    ap.add_argument('--lam', default=','.join('%g' % x for x in LAMS), help='move suppressions (times the teacher\'s): one, or several to choose from on the tuning batch')
    ap.add_argument('--unseen', type=int, default=0, help='the acceptance: the last so many decisions of the test are kept from a model that is tried on them, and a plant '
                    'whose model is too far off there stays with the formula PI (the threshold chosen on the tuning batch); 0: every model goes into service')
    ap.add_argument('--trial', type=int, default=0, help='the acceptance as a plant does it: after the test the formula PI alone for so many decisions, then the model for as many; '
                    'a plant with more rows outside under the model (as the packet shows them) goes back to the formula PI. Also run: all the plants kept either way')
    ap.add_argument('--meter', action='store_true', help='the controller in service corrects its prediction from the instruments alone (core.Program.set_teacher; its model was read '
                    'through the sensors, so their lag is not put in again)')
    ap.add_argument('--tune-n', type=int, default=32)
    ap.add_argument('--tune-windows', type=int, default=4, help='windows after the one of the handover the choice is made on')
    ap.add_argument('--refs', default='', help='file of the other references of these windows (hold, pi, dmc, teacher; the teacher lives on the plants): read if it is of this batch, else computed and written')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.time()
    say = lambda *x: print(*x, f'({time.time()-t0:.0f} s)', flush=True)        # noqa: E731
    lams, reads = [float(x) for x in a.lam.split(',')], a.read.split(',')
    res = dict(args=vars(a))
    # the reading, the suppression and the steady-state layer's part: on another batch
    after = math.ceil((a.length+2*a.trial)/PR.HORIZON)             # the first window after the test and the trial
    progs = list(_windows(a, a.tune_n, TUNE_SEED, after+1+a.tune_windows))
    got = tune(progs, StepTest(a.length, a.amp, a.design, a.hold, sp=a.sp, unseen=a.unseen, trial=a.trial, meter=dict(lag=False) if a.meter else None), lams, reads,
               say=lambda q, v: say('tuning batch: %s, suppression x%g, %s: %.0f decisions outside a plant' % (q[0], q[1], 'trading' if q[2] else 'not trading', v)))
    sets = {q: v for q, v in got.items() if isinstance(q, tuple)}
    (read, lam, trade), acc = min(sets, key=lambda q: float(sets[q].sum())), None
    res['tune'] = {'%s x%g %s' % (q[0], q[1], 'trade' if q[2] else 'keep'): float(v.mean()) for q, v in sets.items()}
    if a.unseen:
        (read, lam, trade), acc, per = accept(got)
        res['tune_accept'] = dict(threshold=acc if math.isfinite(acc) else 'all', outside=float(per.mean()), back=int((got['unseen'][read] > acc).sum()), pi=got['pi'].tolist(),
                                  unseen={q: v.tolist() for q, v in got['unseen'].items()}, spans={q: v.tolist() for q, v in got['spans'].items()},
                                  per={'%s x%g %s' % (q[0], q[1], 'trade' if q[2] else 'keep'): v.tolist() for q, v in sets.items()})
        say('tuning batch: models further off than %g on the unseen stretch stay with the formula PI (%d of %d plants): %.0f decisions outside a plant (the formula PI alone %.0f)' % (
            acc, res['tune_accept']['back'], a.tune_n, float(per.mean()), float(got['pi'].mean())))
    del progs
    res.update(read=read, lam=lam, trade=trade, accept=acc if acc is None or math.isfinite(acc) else 'all')
    # the exam
    kw = dict(length=a.length, amp=a.amp, design=a.design, hold=a.hold, lam=lam, read=read, sp=a.sp, trade=trade, unseen=a.unseen, accept=acc, trial=a.trial, model={},
              meter=dict(lag=False) if a.meter else None)
    ctl, key = StepTest(**kw), hashlib.sha1()
    extra = {'steptest': ctl}
    if a.trial:                                                   # what every plant would have had either way (the same test, the same model)
        extra.update(model=StepTest(keep='model', **kw), alone=StepTest(keep='pi', **kw))
    refs = Path(a.refs) if a.refs else None
    have = json.loads(refs.read_text()) if refs is not None and refs.exists() else None
    names = () if have is not None else ('hold', 'pi', 'dmc', 'teacher')
    out = ST.references(a.n, 0, a.windows, names, progs=_tap(_windows(a, a.n, ST.EXAM_SEED, a.windows), key), extra=extra,
                        say=lambda w, s, t: say(f'window {w}: ' + '  '.join(f'{q} {v:.0f}' for q, v in s.items())))
    key = key.hexdigest()[:16]
    if have is None:
        have = dict(key=key, score={q: out['score'][q].tolist() for q in names}, rows=out['rows'].tolist())
        if refs is not None:
            refs.parent.mkdir(parents=True, exist_ok=True)
            refs.write_text(json.dumps(have))
    assert have['key'] == key, 'the references in %s are of another batch' % a.refs
    score = {q: torch.tensor(v) for q, v in have['score'].items()}
    score['steptest'] = out['score']['steptest']
    res.update(key=key, per_window={q: v.tolist() for q, v in score.items()}, rows=out['rows'].tolist(), failed=out['failed']['steptest'].int().tolist(),
               ruler={q: ruler(ctl.w0, ctl.model[q]['S']) for q in READS}, noise_sd=ctl.model['noise']['sd'].tolist())
    if a.unseen:
        res.update(back=ctl.back.int().tolist(), unseen={q: v.tolist() for q, v in ctl.model['unseen'].items()}, spans={q: v.tolist() for q, v in ctl.model['spans'].items()})
    if a.trial:
        m, p, b = out['score']['model'][after:].sum(0), out['score']['alone'][after:].sum(0), ctl.back.cpu()
        res.update(back=b.int().tolist(), seen=[x.tolist() for x in ctl.seen], either={q: out['score'][q].tolist() for q in ('model', 'alone')},
                   trial=dict(back=int(b.sum()), back_right=int((b & (p < m)).sum()), kept_right=int((~b & (m <= p)).sum()), model_worse=int((p < m).sum())))
    rows = []
    print('decisions      ' + '   '.join(f'{q:>14s}' for q in ('pi', 'dmc', 'teacher', 'steptest')) + '      (points: 0 the MVs left alone, 100 the best of the four references per episode; share of row-decisions outside)')
    spans = [(w, w+1) for w in range(a.windows)]+[(b, a.windows) for b in sorted({math.ceil(a.length/PR.HORIZON), a.windows//2, after}) if 0 < b < a.windows]+[(0, a.windows)]
    for b0, b1 in spans:
        per = {q: v[b0:b1].sum(0).tolist() for q, v in score.items()}
        best = SC.envelope({q: per[q] for q in ('hold', 'pi', 'dmc', 'teacher')})
        line = dict(decisions=(b0*PR.HORIZON, b1*PR.HORIZON))
        for q in ('pi', 'dmc', 'teacher', 'steptest'):
            line[q] = dict(points=SC.points(per[q], per['hold'], best), outside=float(score[q][b0:b1].sum()/out['rows'][b0:b1].sum().clamp_min(1)))
        rows.append(line)
        print(f'{b0*PR.HORIZON:5d}-{b1*PR.HORIZON:5d}    ' + '   '.join(f'{line[q]["points"]:6.0f} {line[q]["outside"]:6.3f} ' for q in ('pi', 'dmc', 'teacher', 'steptest')))
    res['table'] = rows
    ru = res['ruler'][read]
    print('test of %d decisions, +-%g, %s, a switch every %g; the %s reading, suppression x%g, the steady-state layer %s; units failed %d; unexplained share of the labels: own %.3f, cross %.3f' % (
        a.length, a.amp, a.design, a.hold, read, lam, 'trading' if trade else 'not trading', int(out['failed']['steptest'].sum()), ru['own_all'], ru['cross_all']))
    if a.unseen:
        print('acceptance on the last %d decisions of the test: %d of %d plants stay with the formula PI (further off than %g)' % (a.unseen, int(ctl.back.sum()), a.n, acc))
    if a.trial:
        t = res['trial']
        print('trial of %d decisions each way after the test: %d of %d plants go back to the formula PI, %d of them rightly (fewer decisions outside with it from decision %d on); '
              'the model is the worse of the two on %d plants; %d of the plants kept are rightly kept' % (a.trial, t['back'], a.n, t['back_right'], after*PR.HORIZON, t['model_worse'], t['kept_right']))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res))


if __name__ == '__main__':
    main()

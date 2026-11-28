"""The AIO network of the rebuilt pipeline (docs/sim.md, 'Network').

The actor of aio.actor, unchanged, so every earlier checkpoint loads into it, with a pair
pathway beside it:

  identify   For every pair of a row and an MV in service, a small recurrent memory that sees that row's and
             that MV's packet features. Two GRU layers; between them every pair reads the mean over the MVs of
             its row and over the rows of its MV, so that a row's move can be put down to the right MV when
             several moved. It is read out as the pair's step response at model.LAB_LAGS: the identification,
             trained against the simulator's labels.
  products   What the memories above cannot do themselves is multiply, and identification is a correlation:
             a row's change times the moves that came before it. `Stats` (no weights) keeps, for every pair,
             the forgetting sum of the row's change times the MV's recent moves at a few time scales, and
             solves the sums of a row for all its MVs at once (recursive least squares: a change is put down
             to the MV that made it when several moved). The pair memory reads the result as further inputs,
             through a layer that starts at zero. Measured 11-10 on logged closed-loop runs, the read-out
             trained alone (experiments.diagnostics.id_probe; share of the labels' variance explained on
             held-out episodes, published pairs / the others): 480 steps .08 / .00 without, .39 / .23 with;
             three times the steps: .36 / .04 without (the published pairs catch up, the others do not),
             .39 / .13 with (the others .22 at 600 steps, then overfitted on the 384 logged episodes).
             `slow` S (11-11; 0: none): a second set of the same sums that forgets over S decisions, read
             through a layer of its own that starts at zero. The first set forgets in FORGET = 200, less than
             a response lasts (240), so what a unit shows of itself does not add up beyond that. Measured on
             the packets of 32 plants run on for 5,760 decisions under the formula PI with a test of its own
             on top (every MV +-5% of travel, a random telegraph; the same sums solved by plain least squares,
             unexplained share of the responses, own pairs / the others): forgetting in 200, .27 / .65 from
             1,440 decisions on and no better after; in 2,000, .25 / .49 at 1,440 and .16 / .30 at 5,760.
             Without a test the others stay at .83 however long it runs: in closed loop most of a row's
             change is the loads', and the MVs move with them. It is for a unit the network has run for a
             while, its state carried on; it changes nothing by itself in an episode from a cold start.
             `arx` (11-12; off unless asked for): every set of the sums is also fitted with the row's own
             earlier changes beside the MVs' moves, read through a further layer that starts at zero. A row's
             change is partly its own noise and the loads', and in closed loop the MVs move with exactly
             that: sums fitted on the moves alone put it down to the MVs. With the row's past in the fit it
             has a place of its own (the direct method of closed-loop identification: a high-order ARX, here
             in the same filters). Measured on 192 plants with nothing learned (the sums solved as they are;
             unexplained share, own pairs / the others; without / with): under the formula PI with a test on
             top (+-5%), 500, 1,000 and 2,000 decisions .33 / 1.11, .22 / .77, .19 / .57 against .14 / .58,
             .09 / .36, .06 / .25, the median error of a response's shape .26 against .10; the test alone,
             no loops, 2,000 decisions .12 / .39 against .06 / .17; never forgetting, 18,000 quiet decisions
             under the loops later .41 / .63 against .13 / .40; on the validation chain of 8 episodes on
             the same plants with the network (v2-law) at the controls and no test .28 / .44 against
             .17 / .38. A read-out trained from zero on such stays (768 plants; 256 held out) reads .041 / .31
             at the end of the test and .070 / .38 at 20,000 decisions from the plain sums, .032 / .21 and
             .056 / .30 given this fit's weights.
             `dv` (11-12; off unless asked for; with `arx`): the rows that read a measured disturbance are
             inputs of those fits too, beside the MVs: what a measured disturbance does to a row is then not
             left to the MVs and the row's past to explain. No new input and no new layer: the same features,
             of the wider fit. Measured with nothing learned on 64 plants that have one or two, 8 episodes on
             the same plants, no test (own pairs / the others on the rows the MVs drive; without / with): the
             network (v2-law) at the controls .18 / .44 against .17 / .44, with three times the usual steps
             of the measured disturbances .23 / .48 against .20 / .46; under the teacher no difference in the
             others. What the disturbances do to the rows comes out of the same fit (unexplained .84 with the
             usual steps, .48 with three times as many; what feedforward would go by) and is not read yet.
             `reader` (11-14; off unless asked for; in place of the sums, without the law): aio.reader, the same job
             left to weights: learned keys of the MVs and values of the rows, learned time constants and forgetting,
             their products summed over the whole past (linear attention as running sums) and a few learned rounds
             on them; nothing is solved. Read-out trained alone on 4,096 logged episodes (experiments.diagnostics.
             reader_probe; share explained on held-out quiet runs, published pairs / the others): .47 / .24 against
             the sums' .46 / .29, still falling; on 256 episodes .09 / .00 against .22 / .20.
             `apart` (11-15): says it for either, the sums with the read-outs as labels alone or the reader with the
             product read (aio.train --law-path): the two by two that tells a difference of the reader from one of the law.
             `reader` may name what to add (11-15, both untried; aio.reader.spec): 'const', a key that is always 1
             (the rows' own drift, a level's net inflow); 'rounds8', eight rounds of one shared step.
             With `law` beside it the law's read-outs (E, K, V, their product) are labels alone (`apart`): they are
             taught against the teacher's and shape what the rows, MVs and pairs hold, and nothing that runs reads
             what they give: without their weights the moves are bit for bit the same.
  control    Rounds of messages between rows and MVs along the pairs, each pair's message shaped by its
             response and its memory. This is what one iteration of a predictive controller's solve does with
             the model: every row's residual under the moves planned so far, then every MV's correction from
             the rows it moves. The result is added to each MV's token before the policy head. That last layer
             starts at zero, so with earlier weights the network first computes exactly what it did.
             `tied` K (11-11; 0: the two rounds above, each with its own weights, a pair's message a layer on
             its response and the sender's state put side by side): K rounds with one set of weights, a
             pair's message the sender's state times a gate that is linear in the pair's response, averaged
             over the pairs of the receiver (11-12; it was their sum times a constant: a K half trained on
             12x5 plants with 8 rounds left 211 times the teacher's K unexplained on 48x20 ones, the sum
             growing with the number of pairs round after round; the mean leaves .81, and .20 against .19
             where it was trained). A solve multiplies the model by the plan (S du)
             and by the residual (S' e) and repeats one step; a layer on two things side by side has to
             approximate the product, and two rounds are two steps.

  law        (`law`, 11-11; aio.law) Nearly all of the teacher is one product: move = K . E / V, E how far
             every tracked row will be from where it can settle at the label lags if the MVs stop now, K
             every MV's first change of flow per unit of it, V the flow a move of one slew limit makes. A
             student shown only the teacher's moves has to come by them unaided (11-11: told the responses
             it still stood 31 points under the teacher, on the slow plants, the ones with dead time, with
             levels, with five MVs: where the next move hangs on what the moves made so far have yet to do).
             Here they are read out and trained against the teacher's own (E from a row's state after the
             rounds and from the pairs' pending effects: a kernel read from the pair's response times the
             MV's filtered moves, a product again; K from the pair and its two ends; V from the MV's state),
             multiplied (no weights), and the product goes to the MV's token through a layer that starts at
             zero. Told episodes get the true E, K and V in place of the readouts.
             That learned product could not work E out (11-11: with the teacher's own K, 31 points told the
             responses or not). Two switches put arithmetic in its place (aio.formula, no weights; both off
             unless asked for). `formula`: E is worked out from the responses read and the moves made (the
             last 240 of every MV, kept in the state); the heading readout then learns only a factor per
             row and a correction (both start at nothing), from the row's state and what the formula found.
             `solve`: the move is the teacher's design solved on the responses read for that E, in place of
             the K and V readouts. `knots`: the lags the responses are read out at (model.KNOTS: nine, or
             sixteen, which the formula needs). Told episodes are then told the responses alone.
             `given` (11-11): both, but only where the responses are given (a told episode; a plant whose
             step tests are handed over); where the network reads them itself, its learned readouts of E,
             K and V as before. The formula needs the responses to a step test's accuracy: on the ones a
             network reads from closed-loop data it gave 26 points where the learned readouts gave 66
             (told the true ones 73; nine knots, weights of v2-law), a tenth of the read-out's error
             already cost 9.

Before 11-10 a row's memory saw its own token and its paired MV's, and an MV's its paired rows': one CV to one
MV, the way PID loops are paired. No unit saw an MV move and an unpaired row answer, so coupling had nowhere to
be identified and the policy could be an adaptive multi-loop controller at most.

Running, the network has the packet alone: it estimates the responses and acts on its estimates (`code=None`).
In training, `code=(labels, share)` puts the true responses in place of the estimates on the episodes where
share is 1. "Can it control a plant it knows" and "can it get to know the plant" are then trained, and read,
apart.

`Critic` is the same trunk with its own weights and a value head: what the moves made so far and the one
about to be made will cost from here on, Q(history, move). The trainer (aio.train) uses it beyond the horizon
it differentiates through; it is kept in every checkpoint because it reads nothing but the packet history
and the move, which is what learning on from a real unit's own data needs. In training most episodes tell it
the true responses (`code`), the rest leave it to its own estimates, so it is usable both ways.

No weight has a shape that depends on how many rows or MVs the packet has (interface.NCV, NMV): channels,
pairs and heads share their weights across rows and MVs, and the pair pathway works on the pairs in service
only. A larger packet changes the size of the recurrent state, not the network.
"""
import math
import os
import torch
from torch import nn

from aio.interface import NCV, NMV, CF, MF, DV as DRIVEN, shape as shape_of, rows as rows_of
from aio.actor import Actor, ExpertHead, History, STATE as BASE, TOKEN, MEMORY, pool, symlog, symexp, unpack, state_size as base_size  # noqa: F401
from aio.sim import model as MD
from aio.sim.model import NLAB, KNOTS
from aio import formula as FM
from aio import reader as RD


def SQ_ctx(ncv, nmv, w, asked=None):
    """aio.seqnet.ctx_size of a network of that spec (imported late: that module is not needed otherwise)."""
    from aio import seqnet as SQ
    sp = SQ.spec(asked) if asked else None
    return SQ.ctx_size(ncv, nmv, w, sp['mem'] if sp else 0, sp['edge'] if sp else 0, sp['rel'] if sp else 0)

WIDE = int(os.environ.get('AIO_WIDE', '1'))     # the pair pathway's widths times this (11-15; the environment says it before the import: every size below follows)
P = 32*WIDE                      # memory of a pair, per layer
D = 48*WIDE                      # width of the messages
ROUNDS = 2
NPAIR = NCV*NMV
POLES = (1.5, 4., 10., 25., 60., 150.)       # time constants (decisions) of the filters every MV's moves go through
NF = len(POLES)
FORGET = 200.                    # decisions over which the running sums forget
NEAR = 200                       # decisions after an event that count as near it (Policy, `readout`): about one response
NX = NMV*NF
NSTAT = 4*NF+1                   # what `Stats` gives the memory of a pair
E_CLIP, K_CLIP, K_UNIT, V_CLIP, V_UNIT = 6., 5., 1e-3, 2., .06     # the law's readouts: asinh(E), asinh(K/K_UNIT), log(V/V_UNIT), bounded before they are multiplied
UNSEEN, GAP_UNIT, RATE_UNIT = 2, 1., .05   # what the instruments do not show of a row (`unseen_code`): two numbers, in weighted bands and weighted bands a decision
SSTAT = 3*NX+NX*NX+NCV*NX+NCV    # its state: the filters', then the sums R (moves x moves), c (row x moves), E (row x row)
SSLOW = NX*NX+NCV*NX+NCV         # a second set of the sums behind it, forgetting slowly (`Stats`, slow)
NARX = 2*NF                      # what a set of sums fitted with the rows' own past adds for a pair (`Stats`, arx)
SROW = 3*NCV*NF                  # its state: the filters every row's changes go through,
SARX = NCV*NF*NX+NCV*NF*NF+NCV*NF      # and per set of sums P (row's past x moves), Q (row's past x itself), d (row's change x its past)
A_CLIP = 5.                      # bound of a row's weights on its own past as a feature
NDV = 2                          # rows that read a measured disturbance taken beside the MVs in those fits (`Stats`, dv): a plant's first NDV
NXD = NDV*NF
SDVF = 3*NXD                     # its state: the filters their changes go through,
SDV = NX*NXD+NXD*NXD+NCV*NXD+NCV*NF*NXD      # and per set of sums their products: moves x theirs, theirs x theirs, row's change x theirs, row's past x theirs
SPAIR = NPAIR*2*P
STATE = BASE+SPAIR+SSTAT         # recurrent state: the actor's, two layers per pair, the running sums; a network with the slow sums
                                 # carries SSLOW more behind it, one with the rows' own past in the sums SROW and SARX per set more,
                                 # one with the measured disturbances in them SDVF and SDV per set more, then one with the formula
                                 # FM.SMEM (the moves made): `PairNet.size`


def sizes(ncv=NCV, nmv=NMV):
    """(NPAIR, NX, SSTAT, SSLOW, SROW, SARX, SDV, SPAIR) for packets of ncv rows and nmv MVs. The module's names are those of
    the default size; a packet of another size in use (aio.interface.SHAPES) goes through the same
    weights, and every function here takes these from the packets it is given (as aio.sim.prior does: a batch has one size)."""
    nx = nmv*NF
    return (ncv*nmv, nx, 3*nx+nx*nx+ncv*nx+ncv, nx*nx+ncv*nx+ncv, 3*ncv*NF, ncv*NF*nx+ncv*NF*NF+ncv*NF,
            nx*NXD+NXD*NXD+ncv*NXD+ncv*NF*NXD, ncv*nmv*2*P)


def state_size(ncv=NCV, nmv=NMV):
    """STATE for packets of ncv rows and nmv MVs: the least a network carries (AIO.state_size: all a given one does)."""
    s = sizes(ncv, nmv)
    return base_size(ncv, nmv)+s[7]+s[2]


def _in(i, o):
    return nn.Sequential(nn.Linear(i, o), nn.LayerNorm(o), nn.SiLU())


def _up(i, o):
    return nn.Sequential(nn.Linear(i, o), nn.SiLU(), nn.Linear(o, o))


class Stats(nn.Module):
    """Running least squares of every row's changes on every MV's recent moves, from the packets alone (no
    weights, no gradient; causal).

    Every MV's move (packet feature 1) goes through NF orthonormal filters (Takenaka-Malmquist sections with
    the time constants POLES: the first a lag of 1.5 decisions, each further one behind an all-pass of the
    one before), which hold its recent moves at several time scales: X (NMV*NF). Summed with forgetting
    (FORGET decisions): R = sum X X', c = sum (row's change) X', E = sum (row's change)^2. For every pair:

      corr    c / sqrt(R_jj E_i) per filter: how the row's changes went with that MV's moves
      joint   (R + lam I)^-1 c, the row's sums solved for all MVs at once, per filter, per full MV span,
              asinh: recursive least squares in the filter coordinates
      alone   the same solved for that MV by itself (what it would be were the other MVs still)
      energy  log R_jj per filter (how much that MV has moved: how far the above can be trusted), log E_i

    A linear attention layer over the pair's past adds up the same products (von Oswald et al. 2023: such a
    layer does a step of least squares in context); here they are kept as running sums, so the cost per
    decision does not grow with the history and the history is not cut at the trained window.

    slow S (0: none; inf: never forgets): a second set of R, c, E that forgets over S decisions, kept behind the
    first in the state (SSLOW more); its features, the same four of its own sums, come last.

    arx: every set is also solved with the row's own earlier changes beside the MVs' moves (the changes up to
    the decision before, through the same NF filters: Y). A row's change is partly its own noise and the loads',
    and in closed loop the MVs move with exactly that; fitted on the moves alone, it is put down to them. With
    the row's past in the fit it has a place of its own: the direct method of closed-loop identification (a
    high-order ARX in orthonormal filters; Ljung 1999 ch. 13, Zhu's ASYM). Kept per set behind everything
    else in the state: the rows' filters once (SROW), then P = sum Y X', Q = sum Y Y', d = sum (row's change) Y'
    (SARX per set). Features, NARX per set, last of all: the fit's weights on the pair's MV's moves (per filter,
    per full MV span, asinh) and the row's weights on its own past (per filter, the same for all its MVs). The
    response of a pair is that fit run forward from a step of the MV; a read-out learns it from the two.

    dv (with arx): the rows that read a measured disturbance (the packet's flag; a plant's first NDV) are inputs
    of those fits too: their changes, this decision's with them, through the same NF filters, beside the MVs'
    moves. A row's change that a measured disturbance made then has its own place and is not left to the MVs
    and the row's past to explain. The features are the same ones, of the wider fit; the weights it puts on
    the disturbances (`solve_own`'s third result: what each does to every row) are not read yet. Kept last in
    the state: their filters (SDVF), then per set their sums with the moves, with themselves, with every row's
    change and with every row's past (SDV)."""

    def __init__(self, moves=False, slow=0., arx=False, dv=False):
        super().__init__()
        assert arx or not dv, 'the measured disturbances go into the fits with the rows\' own past'
        self.moves = moves                                        # the features are followed by the MV's NF filtered moves (the law's pending effects are products with them)
        self.slow = float(slow)
        self.arx, self.dv = bool(arx), bool(dv)
        self.rho2 = math.exp(-1/self.slow) if self.slow > 0 else 0.
        self.sets = 2 if self.slow > 0 else 1
        self.size = self.size_of()                                # of its state, at the default size
        o = NSTAT+(NF if moves else 0)
        self.at_slow = slice(o, o+NSTAT)                          # where the slow set's features are, and those of the fits with the rows' own past
        o += NSTAT if self.slow > 0 else 0
        self.at_arx = slice(o, o+self.sets*NARX)
        a = torch.exp(-1/torch.tensor(POLES))
        self.register_buffer('a', a, persistent=False)
        self.register_buffer('g', torch.sqrt(1-a*a), persistent=False)
        self.rho = math.exp(-1/FORGET)

    def size_of(self, ncv=NCV, nmv=NMV):
        """Length of its state for packets of ncv rows and nmv MVs."""
        _, _, SSTAT, SSLOW, SROW, SARX, SDV, _ = sizes(ncv, nmv)
        return SSTAT+(SSLOW if self.slow > 0 else 0)+(SROW+self.sets*SARX if self.arx else 0)+(SDVF+self.sets*SDV if self.dv else 0)

    def bank(self, v, x, w, vp):
        """The filters moved on by a decision: v (..., ) what goes in, x, w, vp (..., NF) their state."""
        xs, ws, vs = [], [], []
        for m in range(NF):                                       # the lag of section m, then its all-pass feeds the next
            xs.append(self.a[m]*x[..., m]+self.g[m]*v)
            ws.append(self.a[m]*w[..., m]+vp[..., m]-self.a[m]*v)
            vs.append(v)
            v = ws[-1]
        return torch.stack(xs, -1), torch.stack(ws, -1), torch.stack(vs, -1)

    @torch.no_grad()
    def forward(self, obs, state=None):
        """obs (b,T,SIZE) packets, state (b,self.size) or None -> the features (b,T,NCV,NMV,NSTAT; with `moves`
        NF more, with `slow` NSTAT more after those, with `arx` NARX per set after those), the state."""
        cv, mv, _ = unpack(obs)
        dy, dp = ((x[..., 1]*x[..., -1]).float().nan_to_num(0., 0., 0.).clamp(-1e3, 1e3) for x in (cv, mv))   # the sums stay finite whatever a packet holds
        b, T = dy.shape[:2]
        NCV, NMV = cv.shape[-2], mv.shape[-2]                     # the packets' size, and what goes with it
        _, NX, SSTAT, SSLOW, SROW, SARX, SDV, _ = sizes(NCV, NMV)
        if state is None:
            state = dy.new_zeros(b, self.size_of(NCV, NMV))
        state = state.float()
        x, w, vp = (state[:, i*NX:(i+1)*NX].reshape(b, NMV, NF) for i in range(3))
        R = state[:, 3*NX:3*NX+NX*NX].reshape(b, NX, NX)
        c = state[:, 3*NX+NX*NX:SSTAT-NCV].reshape(b, NCV, NX)
        E = state[:, SSTAT-NCV:SSTAT]
        o = SSTAT
        if self.slow > 0:
            R2, c2, E2 = state[:, o:o+NX*NX].reshape(b, NX, NX), state[:, o+NX*NX:o+SSLOW-NCV].reshape(b, NCV, NX), state[:, o+SSLOW-NCV:o+SSLOW]
            o += SSLOW
        if self.arx:
            yx, yw, yvp = (state[:, o+i*NCV*NF:o+(i+1)*NCV*NF].reshape(b, NCV, NF) for i in range(3))
            o, own = o+SROW, []
            for _ in range(self.sets):                            # per set: P (row's past x moves), Q (row's past x itself), d (row's change x its past)
                own.append([state[:, o:o+NCV*NF*NX].reshape(b, NCV, NF, NX), state[:, o+NCV*NF*NX:o+SARX-NCV*NF].reshape(b, NCV, NF, NF),
                            state[:, o+SARX-NCV*NF:o+SARX].reshape(b, NCV, NF)])
                o += SARX
        ins = [(), ()]
        if self.dv:
            zx, zw, zvp = (state[:, o+i*NXD:o+(i+1)*NXD].reshape(b, NDV, NF) for i in range(3))
            o, ins = o+SDVF, []
            for _ in range(self.sets):                            # per set: moves x theirs, theirs x theirs, row's change x theirs, row's past x theirs
                cut = (0, NX*NXD, NX*NXD+NXD*NXD, NX*NXD+NXD*NXD+NCV*NXD, SDV)
                ins.append([state[:, o+i:o+j].reshape(b, *shape) for i, j, shape in zip(cut, cut[1:], ((NX, NXD), (NXD, NXD), (NCV, NXD), (NCV, NF, NXD)))])
                o += SDV
            on, at = torch.sort(((cv[..., -1] > 0) & (cv[..., DRIVEN] < .5)).float(), dim=-1, descending=True, stable=True)
            dz = torch.gather(dy, 2, at[..., :NDV])*on[..., :NDV]      # (b,T,NDV): the changes of the first NDV rows that read a measured disturbance
        out = []
        for t in range(T):
            x, w, vp = self.bank(dp[:, t], x, w, vp)
            X = x.reshape(b, NX)
            R = self.rho*R+X[:, :, None]*X[:, None, :]
            c = self.rho*c+dy[:, t][:, :, None]*X[:, None, :]
            E = self.rho*E+dy[:, t]**2
            f = [self.features(R, c, E)]
            if self.moves:
                f.append(x[:, None].expand(b, NCV, NMV, NF))
            if self.slow > 0:
                R2 = self.rho2*R2+X[:, :, None]*X[:, None, :]
                c2 = self.rho2*c2+dy[:, t][:, :, None]*X[:, None, :]
                E2 = self.rho2*E2+dy[:, t]**2
                f.append(self.features(R2, c2, E2))
            if self.arx:
                for q, rho in zip(own, (self.rho, self.rho2)):    # yx: the rows' changes up to the decision before
                    q[0] = rho*q[0]+yx[:, :, :, None]*X[:, None, None, :]
                    q[1] = rho*q[1]+yx[:, :, :, None]*yx[:, :, None, :]
                    q[2] = rho*q[2]+dy[:, t][:, :, None]*yx
                if self.dv:
                    zx, zw, zvp = self.bank(dz[:, t], zx, zw, zvp)
                    Z = zx.reshape(b, NXD)
                    for q, rho in zip(ins, (self.rho, self.rho2)):
                        q[0] = rho*q[0]+X[:, :, None]*Z[:, None, :]
                        q[1] = rho*q[1]+Z[:, :, None]*Z[:, None, :]
                        q[2] = rho*q[2]+dy[:, t][:, :, None]*Z[:, None, :]
                        q[3] = rho*q[3]+yx[:, :, :, None]*Z[:, None, None, :]
                yx, yw, yvp = self.bank(dy[:, t], yx, yw, yvp)
                f.append(self.fit(R, c, *own[0], *ins[0]))
                if self.slow > 0:
                    f.append(self.fit(R2, c2, *own[1], *ins[1]))
            out.append(torch.cat(f, -1) if len(f) > 1 else f[0])
        state = [x.reshape(b, NX), w.reshape(b, NX), vp.reshape(b, NX), R.reshape(b, NX*NX), c.reshape(b, NCV*NX), E]
        if self.slow > 0:
            state += [R2.reshape(b, NX*NX), c2.reshape(b, NCV*NX), E2]
        if self.arx:
            state += [yx.reshape(b, NCV*NF), yw.reshape(b, NCV*NF), yvp.reshape(b, NCV*NF)]
            for P_, Q_, d_ in own:
                state += [P_.reshape(b, NCV*NF*NX), Q_.reshape(b, NCV*NF*NF), d_.reshape(b, NCV*NF)]
        if self.dv:
            state += [zx.reshape(b, NXD), zw.reshape(b, NXD), zvp.reshape(b, NXD)]
            for q in ins:
                state += [x_.reshape(b, -1) for x_ in q]
        return torch.stack(out, 1), torch.cat(state, -1)

    @staticmethod
    def features(R, c, E):
        b, NCV, NX = c.shape
        NMV = NX//NF
        d = R.diagonal(dim1=1, dim2=2)
        corr = (c/torch.sqrt(d[:, None, :]*E[:, :, None]+1e-12)).reshape(b, NCV, NMV, NF)
        lam = (1e-3*d.mean(-1)+1e-10).double()[:, None, None]
        R64, c64 = R.double(), c.double()
        solve = MD.spd_solve                                      # R + lam I is positive definite; the blocks of NF by LU, 48x20's 120 moves together by Cholesky
        joint = solve(R64+lam*torch.eye(NX, dtype=torch.float64, device=R.device), c64.transpose(1, 2)).transpose(1, 2).reshape(b, NCV, NMV, NF)
        Rb = R64.reshape(b, NMV, NF, NMV, NF).diagonal(dim1=1, dim2=3).permute(0, 3, 1, 2)       # (b,NMV,NF,NF): each MV's own block
        alone = solve(Rb+lam[..., None]*torch.eye(NF, dtype=torch.float64, device=R.device), c64.reshape(b, NCV, NMV, NF).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        ld = (torch.log(d+1e-8)/8).reshape(b, 1, NMV, NF).expand(b, NCV, NMV, NF)
        le = (torch.log(E+1e-8)/8)[:, :, None, None].expand(b, NCV, NMV, 1)
        # the move feature spans 2 over the MV limits in force (.9 of the span unless an event narrowed them)
        return torch.cat((corr, torch.asinh(joint/.45).float(), torch.asinh(alone/.45).float(), ld, le), -1)

    @staticmethod
    def solve_own(R, c, P, Q, d, Rz=None, Rzz=None, cz=None, Pz=None):
        """A set of sums solved with every row's own past beside the moves: for row i, [R P_i'; P_i Q_i] [tb; ta] =
        [c_i; d_i] (a little of each block's mean diagonal added to it, as `features` does). R is the same for all
        rows, so it is solved once, for c and P together, and each row is left a system of NF.
        -> tb (b,NCV,NMV,NF) the weights on the MVs' moves, ta (b,NCV,NF) on the row's past; float64.
        With the sums of the measured disturbances (Rz moves x theirs (b,NX,NXD), Rzz theirs x theirs, cz every
        row's change x theirs (b,NCV,NXD), Pz every row's past x theirs (b,NCV,NF,NXD)) they stand beside the
        moves in the same system, and a third result is the weights on them, tz (b,NCV,NDV,NF)."""
        (b, NCV, NX), f = c.shape, torch.float64
        NMV, nx = NX//NF, NX
        solve = MD.spd_solve                                      # (the moves' sums with the disturbances' are 132 unknowns at 48x20: by Cholesky, as above)
        lam = (1e-3*R.diagonal(dim1=1, dim2=2).mean(-1)+1e-10).to(f)[:, None, None]
        lam2 = (1e-3*Q.diagonal(dim1=2, dim2=3).mean(-1)+1e-10).to(f)[:, :, None, None]
        P64, c64 = P.to(f), c.to(f)
        A = R.to(f)+lam*torch.eye(NX, dtype=f, device=R.device)
        if Rz is not None:
            lamz = (1e-3*Rzz.diagonal(dim1=1, dim2=2).mean(-1)+1e-10).to(f)[:, None, None]
            A = torch.cat((torch.cat((A, Rz.to(f)), -1), torch.cat((Rz.to(f).transpose(1, 2), Rzz.to(f)+lamz*torch.eye(NXD, dtype=f, device=R.device)), -1)), 1)
            P64, c64, nx = torch.cat((P64, Pz.to(f)), -1), torch.cat((c64, cz.to(f)), -1), NX+NXD
        W = solve(A, torch.cat((c64.transpose(1, 2), P64.reshape(b, NCV*NF, nx).transpose(1, 2)), -1))
        Wc, Wp = W[..., :NCV].transpose(1, 2), W[..., NCV:].transpose(1, 2).reshape(b, NCV, NF, nx)       # R^-1 c_i (b,NCV,NX), R^-1 P_i' by rows (b,NCV,NF,NX)
        left = Q.to(f)+lam2*torch.eye(NF, dtype=f, device=R.device)-torch.einsum('bimx,binx->bimn', P64, Wp)
        ta = solve(left, (d.to(f)-torch.einsum('bimx,bix->bim', P64, Wc))[..., None])[..., 0]
        tb = Wc-torch.einsum('bim,bimx->bix', ta, Wp)
        if Rz is not None:
            return tb[..., :NX].reshape(b, NCV, NMV, NF), ta, tb[..., NX:].reshape(b, NCV, NDV, NF)
        return tb.reshape(b, NCV, NMV, NF), ta

    @staticmethod
    def fit(R, c, P, Q, d, *dv):
        """-> (b,NCV,NMV,NARX): `solve_own`'s weights on the pair's MV's moves (per full MV span, asinh) and the
        row's weights on its own past (bounded; the same for all its MVs). dv: the measured disturbances' sums."""
        tb, ta = Stats.solve_own(R, c, P, Q, d, *dv)[:2]
        NMV = tb.shape[2]
        return torch.cat((torch.asinh(tb/.45).float(), ta.clamp(-A_CLIP, A_CLIP).float()[:, :, None, :].expand(-1, -1, NMV, -1)), -1)

    @staticmethod
    def response(tb, ta, lags):
        """The fit run forward from a unit step of the MV: what it has done to the row after each of `lags`
        decisions, per full MV span -> (b,NCV,NMV,len(lags)). (Not on the network's path: for checks and plots.)
        Of the weights on the measured disturbances (tz (b,NCV,NDV,NF)): .45 times it is what a unit of the
        disturbance row's packet value has done to the row."""
        b, f = tb.shape[0], torch.float64
        a = torch.exp(-1/torch.tensor(POLES, dtype=f, device=tb.device))
        g = torch.sqrt(1-a*a)

        def bank(v, x, w, vp):
            xs, ws, vs = [], [], []
            for m in range(NF):
                xs.append(a[m]*x[..., m]+g[m]*v); ws.append(a[m]*w[..., m]+vp[..., m]-a[m]*v); vs.append(v); v = ws[-1]
            return torch.stack(xs, -1), torch.stack(ws, -1), torch.stack(vs, -1)
        mvs = [torch.zeros(NF, dtype=f, device=tb.device) for _ in range(3)]
        own = [torch.zeros(tb.shape, dtype=f, device=tb.device) for _ in range(3)]
        y, out = torch.zeros(tb.shape[:-1], dtype=f, device=tb.device), []
        for t in range(max(lags)):
            mvs = bank(torch.tensor(1. if t == 0 else 0., dtype=f, device=tb.device), *mvs)
            dy = ((tb.to(f)*mvs[0]).sum(-1)+(own[0]*ta.to(f)[:, :, None, :]).sum(-1)).clamp(-1e3, 1e3)
            own = bank(dy, *own)
            y = (y+dy).clamp(-1e3, 1e3)
            if t+1 in lags:
                out.append(y)
        return torch.stack(out, -1).float()/.45


def _grown_stat(module, keys):
    """Load hook: weights from before the running sums (11-10), from before their slow set (11-11), from
    before the rows' own past in them, from before the read-out of what the instruments do not show or from
    before the share told (11-12), leave the layer that reads them at its zero start, where the network computes what those weights did."""
    keys.missing_keys[:] = [k for k in keys.missing_keys if not {'stat', 'stat2', 'slow', 'stat3', 'arx', 'dvs', 'unseen', 'unseen_in', 'told_in'} & set(k.split('.'))]


def has_stats(pairs):
    """Whether a PairNet's weights (its state dict, or a module's with the pairs under `pairs.`) read the
    running sums."""
    return any('stat' in k.split('.') for k in pairs)


def slow_of(pairs):
    """The decisions over which the slow set of the running sums of a PairNet's weights (its state dict) forgets;
    0: it has none."""
    return float(pairs['slow']) if 'slow' in pairs else 0.


def arx_of(pairs):
    """Whether a PairNet's weights (its state dict) read the sums fitted with the rows' own past."""
    return 'arx' in pairs


def dv_of(pairs):
    """Whether those fits of a PairNet's weights (its state dict) take the measured disturbances beside the MVs."""
    return 'dvs' in pairs


def unseen_of(pairs):
    """Whether a PairNet's weights (its state dict) read out what the instruments do not show (`unseen`)."""
    return 'unseen.weight' in pairs


def apart_of(pairs):
    """Whether the law's read-outs of a PairNet's weights (its state dict) are labels alone: nothing takes their product in."""
    return law_of(pairs) and 'law_in.weight' not in pairs


def reader_of(pairs):
    """Whether a PairNet's weights (its state dict) have the learned reader (aio.reader) in place of the running sums:
    False, True, or what was asked of it beside ('const', 'rounds8', 'const,rounds8': aio.reader.spec)."""
    if not any(k.startswith('reader.') for k in pairs):
        return False
    asked = ['const']*('reader.A0' in pairs)+(['rounds%d' % int(pairs['reader.nround'])] if 'reader.nround' in pairs else [])
    k, v, w = pairs['reader.pole'].shape[0], pairs['reader.val0.weight'].shape[0], pairs['reader.key.0.weight'].shape[0]
    asked += ['k%d' % k]*(k != RD.K)+['v%d' % v]*(v != RD.V)+['w%d' % w]*(w != 48)
    return ','.join(asked) or True


def told_of(pairs):
    """Whether a PairNet's weights (its state dict) are given the share of the responses that was told (`told`)."""
    return 'told_in.weight' in pairs


def law_of(pairs):
    """Whether a PairNet's weights (its state dict) have the law's readouts."""
    return any(k.startswith('heading.') for k in pairs)


def knots_of(pairs):
    """The number of lags a PairNet's weights (its state dict) read the responses out at."""
    return int(pairs['read.2.weight'].shape[0])


def worked_of(pairs):
    """Which of the law's steps a PairNet's weights (its state dict) leave to arithmetic: (E by the formula,
    the move by the design solved)."""
    return (law_of(pairs) and not given_of(pairs) and pairs['heading.2.weight'].shape[0] == NLAB+1, law_of(pairs) and not any(k.startswith('gain.') for k in pairs))


def hybrid_of(pairs):
    """Whether a PairNet's weights (its state dict) have the learned kernel beside the formula's E (`hybrid`)."""
    return worked_of(pairs)[0] and any(k.startswith('kern.') for k in pairs)


def given_of(pairs):
    """Whether a PairNet's weights (its state dict) work both steps out where the responses are given and read
    them out where they are not."""
    return any(k.startswith('adjust.') for k in pairs)


def tied_of(pairs):
    """The number of tied rounds of a PairNet's weights (its state dict); 0: the two rounds of their own."""
    return int(pairs['rounds']) if 'rounds' in pairs else 0


def diff_of(pairs):
    """Which read-outs of a PairNet's weights (its state dict) give the change from one lag to the next: 'S', 'E', 'SE'; '': none."""
    return ''.join(q for q in 'SE' if 'diff_'+q in pairs)


def steps_of(x, clip):
    """asinh values at successive lags (..., n) -> asinh of the change from each lag to the next (the first: the value itself)."""
    v = torch.sinh(x.float().clamp(-clip, clip))
    return torch.asinh(torch.diff(v, dim=-1, prepend=torch.zeros_like(v[..., :1])))


def values_of(d, clip):
    """The other way: asinh of the changes added up (a change between two values within the bound is within twice it)."""
    top = math.asinh(2*math.sinh(clip))
    return torch.asinh(torch.cumsum(torch.sinh(d.float().clamp(-top, top)), -1))


def taught(net, x, what):
    """The responses ('S': (..., knots)) or E ('E': (..., NLAB)), asinh values, as the read-out of `net` (an AIO) is taught
    them: as they are, or (a network with `diff`) as the changes from lag to lag. Both sides of the read-out's loss go
    through it: read and label."""
    return steps_of(x, FM.S_CLIP if what == 'S' else E_CLIP) if what in net.pairs.diff else x


class PairNet(nn.Module):
    """stats False: without the running sums and the layer that reads them (the network of PR #299).
    tied K: the control half as K rounds of one set of weights with products in the messages (0: two rounds
    of their own weights, as before). law: with the law's readouts (needs the running sums: their filters).
    knots: the lags the responses are read out at (model.KNOTS). formula, solve (with law): E worked out from
    the responses and the moves made, the move by the design solved on them (aio.formula), in place of the
    readouts that learn them. given (with law): both worked out in the episodes whose responses are given
    (`code`), the learned readouts in the others; it has the weights of both. slow S (with stats): a second set
    of the running sums that forgets over S decisions (`Stats`), read through a layer of its own that starts at
    zero: weights from before it compute what they did. arx (with stats): every set of the sums is also fitted
    with the rows' own past beside the moves (`Stats`), read through a further layer that starts at zero. dv
    (with arx): those fits take the measured disturbances beside the MVs too (`Stats`): no layer of its own.
    diff (11-12; 'S', 'E' or both; none unless asked for): the read-out of the responses ('S') or of E ('E'; the learned
    one, with law) gives the change from one lag to the next, the first lag's value first, and is taught that (`taught`);
    the values are the changes added up, and everything after it gets the values as before. A read-out taught asinh
    values by squared error reads what is large and uncertain low: a level's response 240 decisions on came out at a
    third (11-12); its changes from knot to knot do not. No weights of its own: the same layers, another meaning of
    what they give (weights of a network without it are of no use to one with it until their read-out is fitted again).
    unseen (11-12; with law, the E read-out that learns): a read-out of what the instruments do not show of every row
    (`unseen_code`: the true value less the reading; the rate the row goes on at with nothing still to come, a level's net
    inflow), one layer on the row's token, and the E read-out's first layer gets what it gives beside the row's token and
    the moves' effect still to come (the told episodes: the true ones, as E, K and V are told). The effect still to come
    is counted from now, so what E at the far lags lacks is the rate now: with the true model a slow filter of the
    readings gets four fifths of it, the network's trunk had a third of its size (11-12). The layer into the E read-out
    starts at zero: weights from before compute what they did.
    told (11-12): the pairs are given the share of the responses they have that was told (`code`'s share: 1 in a told
    episode and in a problem of design, 0 where the network has its own reading, which is all it has when it runs), through
    a layer without a bias that starts at zero. The same read-outs then learn two things that otherwise pull against each
    other: what to make of the true responses, and what to make of its own reading of them, which is wrong in ways that
    go together from pair to pair. Nothing is added to what a network that is told nothing computes.
    hybrid (11-15; with formula): the heading read-out gets the learned kernel's pending effects beside the formula's E. The
    formula is exact where the responses are (told, or read well after a long stay) and useless on a poor reading; the
    kernel is the other way round (11-15, weights of 60 rows from zero: E's error .82 learned, 1.28 by the formula on its
    own reading, .47 by the formula on the true responses; told the true responses the learned read-out gains nothing)."""

    def __init__(self, stats=True, tied=0, law=False, knots=NLAB, formula=False, solve=False, given=False, slow=0., arx=False, dv=False, diff='', unseen=False, told=False, reader=False, apart=None, hybrid=False):
        super().__init__()
        assert stats or not (slow or arx), 'the slow sums are a second set of the running sums, the rows\' own past a second fit of them'
        assert not reader or not (stats or formula or solve or given), 'the learned reader (aio.reader) is in place of the running sums and of every formula'
        self.apart = bool(law and (reader if apart is None else apart))    # the law's read-outs as labels alone: taught, never read by what runs
                                                                  # (as it is with the reader unless said; `apart` says it for either: the two by two of sums or reader, law read or not)
        assert arx or not dv, 'the measured disturbances go into the fits with the rows\' own past'
        given = law and given
        self.reader = None
        formula, solve = formula and not given, solve and not given
        self.hybrid = hybrid = bool(law and formula and hybrid)   # the formula's E and the learned kernel's, both before the heading read-out
        self.nk = NK = len(KNOTS[knots])
        self.inp = _in(CF+MF+2, P)
        self.gru1 = nn.GRU(P, P, batch_first=True)
        self.gru2 = nn.GRU(3*P, P, batch_first=True)
        self.read = nn.Sequential(nn.Linear(P, 2*P), nn.SiLU(), nn.Linear(2*P, NK))
        self.edge = _in(NK+1+P, D)
        self.row0, self.mv0 = _in(TOKEN, D), _in(TOKEN, D)
        self.tied = tied
        if tied:
            self.register_buffer('rounds', torch.tensor(tied))
            self.gate_row, self.gate_mv = nn.Linear(D+NK+1, D), nn.Linear(D+NK+1, D)        # no norm: the response's size goes through
            self.say_row, self.say_mv = nn.Linear(D, D), nn.Linear(D, D)
            self.up_row, self.up_mv = _up(2*D, D), _up(2*D, D)
            for q in (self.up_row, self.up_mv):                   # a round starts close to leaving the states as they are
                q[-1].weight.data.mul_(.1); q[-1].bias.data.zero_()
        else:
            self.to_row = nn.ModuleList([_in(2*D, D) for _ in range(ROUNDS)])
            self.to_mv = nn.ModuleList([_in(2*D, D) for _ in range(ROUNDS)])
            self.up_row = nn.ModuleList([_up(2*D, D) for _ in range(ROUNDS)])
            self.up_mv = nn.ModuleList([_up(2*D, D) for _ in range(ROUNDS)])
        self.out = nn.Linear(D, TOKEN)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.stats = None
        if stats:                                                 # last: the layers above start as they did without
            self.stats, self.stat = Stats(moves=law and (not formula or hybrid), slow=slow, arx=arx, dv=dv), _up(NSTAT, P)
            nn.init.zeros_(self.stat[-1].weight)
            nn.init.zeros_(self.stat[-1].bias)
            self.register_load_state_dict_post_hook(_grown_stat)
        self.law, self.read_law, self.read_given = law, None, None
        self.worked, self.solve, self.given = law and (formula or given), law and (solve or given), given
        self.fm = FM.Formula(KNOTS[knots]) if self.worked or self.solve else None      # no weights
        self.size = self.size_of()                                # of its state, at the default size
        if law:
            assert stats or formula or reader, 'the law reads the filtered moves of the running sums'
            if formula:                                           # a factor per row and a correction of what the formula finds: both start at nothing
                self.heading = nn.Sequential(nn.Linear(D+NLAB+2+(NLAB if hybrid else 0), 2*D), nn.SiLU(), nn.Linear(2*D, NLAB+1))
                nn.init.zeros_(self.heading[-1].weight)
                nn.init.zeros_(self.heading[-1].bias)
                if hybrid:                                        # the learned kernel beside it: what the moves so far have yet to do, as the read-out that learns has it
                    assert stats, 'the learned kernel reads the filtered moves of the running sums'
                    self.nf = NF
                    self.kern = nn.Sequential(nn.Linear(NK+1+P, 2*P), nn.SiLU(), nn.Linear(2*P, NLAB*self.nf))
            else:
                self.nf = RD.spec(reader).get('k', RD.K) if reader else NF                  # the moves so far as the sums' filters hold them, or as the learned reader's keys do
                self.kern = nn.Sequential(nn.Linear(NK+1+P, 2*P), nn.SiLU(), nn.Linear(2*P, NLAB*self.nf))
                self.heading = nn.Sequential(nn.Linear(D+NLAB, 2*D), nn.SiLU(), nn.Linear(2*D, NLAB))
            if given:                                             # the formula's factor and correction, beside the learned readout
                self.adjust = nn.Sequential(nn.Linear(D+NLAB+2, 2*D), nn.SiLU(), nn.Linear(2*D, NLAB+1))
                nn.init.zeros_(self.adjust[-1].weight)
                nn.init.zeros_(self.adjust[-1].bias)
            if not solve:
                self.gain = nn.Sequential(nn.Linear(3*D, 2*D), nn.SiLU(), nn.Linear(2*D, NLAB))
                self.valve = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, 1))
            if not self.apart:
                self.law_in = nn.Linear(2, TOKEN)
                nn.init.zeros_(self.law_in.weight)
                nn.init.zeros_(self.law_in.bias)
        if slow:                                                  # last of all: every layer above starts as it does without
            self.register_buffer('slow', torch.tensor(float(slow)))
            self.stat2 = _up(NSTAT, P)
            nn.init.zeros_(self.stat2[-1].weight)
            nn.init.zeros_(self.stat2[-1].bias)
        if arx:                                                   # and after that
            self.register_buffer('arx', torch.tensor(1.))
            self.stat3 = _up(2*NARX, P)                           # of both sets (zeros where there is no slow set: adding it later leaves the layer as it is)
            nn.init.zeros_(self.stat3[-1].weight)
            nn.init.zeros_(self.stat3[-1].bias)
        if dv:
            self.register_buffer('dvs', torch.tensor(1.))
        assert not set(diff)-set('SE') and ('E' not in diff or (law and not self.worked)), 'diff: S (the responses), E (the E read-out that learns, with law)'
        self.diff = ''.join(q for q in 'SE' if q in diff)
        for q in self.diff:
            self.register_buffer('diff_'+q, torch.tensor(1.))
        self.unseen, self.read_unseen = None, None
        if unseen:                                                # last of all
            assert law and not self.worked, 'unseen: beside the E read-out that learns (with law)'
            self.unseen, self.unseen_in = nn.Linear(D, UNSEEN), nn.Linear(UNSEEN, 2*D, bias=False)
            nn.init.zeros_(self.unseen_in.weight)
        if reader:                                                # last of all: learned, in place of the running sums (aio.reader); its state is where theirs is
            self.reader = RD.Reader(out=P, **RD.spec(reader))
            self.size = self.size_of()
        if WIDE > 1:                                              # weights of another width do not load: said in the file
            self.register_buffer('wide', torch.tensor(WIDE))
        self.told_in = None
        if told:                                                  # after that
            self.told_in = nn.Linear(1, D, bias=False)
            nn.init.zeros_(self.told_in.weight)

    def size_of(self, ncv=NCV, nmv=NMV):
        """Length of its state for packets of ncv rows and nmv MVs."""
        s = sizes(ncv, nmv)
        return s[7]+(self.reader.size_of(ncv, nmv) if self.reader is not None else s[2] if self.stats is None else self.stats.size_of(ncv, nmv))+(FM.SMEM if self.fm is not None else 0)

    def forward(self, obs, previous, ce, me, state, code=None, feats=None):
        """obs (b,T,SIZE) packets, previous (b,T,NMV) the last actions, ce (b,T,NCV,TOKEN) and me (b,T,NMV,TOKEN)
        the channel tokens, state (b, self.size): the pair memories, then the running sums (their slow set
        behind the first), then (a network with the formula) the moves made (aio.formula);
        code: (labels (b,T,NCV,NMV,knots), share (b,)) or None, with the law (labels, share, E (b,T,NCV,NLAB),
        K (b,T,NCV,NMV,NLAB), V (b,T,NMV)[, what the instruments do not show (b,T,NCV,UNSEEN): `unseen_code`]),
        as the readouts are (`law_code`: each is put in place of the
        readout that learns it, where the network has that readout; a `given` network is told the responses
        alone, and works E and the move out in the episodes of that share); feats: the running sums' features
        (b,T,NCV,NMV,NSTAT, or more: `Stats`; a network without the slow set reads the first NSTAT of an
        actor's that has it) where the caller keeps them itself (the trainer: once for actor and critic; the
        state's sums are then left as they are), None: kept here.
        -> change of the MV tokens (b,T,NMV,TOKEN), the responses read (b,T,NCV,NMV,knots), state. With the
        law, self.read_law is then (E read (b,T,NCV,NLAB), K read (b,T,NCV,NMV,NLAB), V read (b,T,NMV), their
        product, the law's move (b,T,NMV) in slew limits); K and V are None where the design is solved in
        every episode. A `given` network: E, K and V as it reads them in every episode, the move of the path
        each episode takes; self.read_given is the formula's E (b,T,NCV,NLAB; 0 in the episodes not given
        the responses), None where no episode is. With `unseen`, self.read_unseen is what that read-out gives
        (b,T,NCV,UNSEEN)."""
        cv, mv, links = unpack(obs)
        b, T, NK = obs.shape[0], obs.shape[1], self.nk
        NCV, NMV = cv.shape[-2], mv.shape[-2]                     # the packets' size, and what goes with it
        NPAIR, _, SSTAT, _, _, _, _, SPAIR = sizes(NCV, NMV)
        assert self.fm is None or NPAIR == sizes()[0], 'the formula (aio.formula) is written for the default size alone'
        ns = self.reader.size_of(NCV, NMV) if self.reader is not None else SSTAT if self.stats is None else self.stats.size_of(NCV, NMV)
        hp, hs, hf = state[:, :SPAIR].reshape(b, NPAIR, 2*P), state[:, SPAIR:SPAIR+ns], state[:, SPAIR+ns:]
        seen = None
        if self.reader is not None:                               # what the learned reader makes of the packets so far, per pair (b,T,NCV,NMV,P)
            seen, hs = self.reader(obs, previous, hs)
            hs = hs.to(state.dtype)
        if self.stats is None:
            feats = None
        elif feats is None:
            feats, hs = self.stats(obs, hs)
            hs = hs.to(state.dtype)
        cm, mm = cv[..., -1], mv[..., -1]
        alive = cm[..., :, None]*mm[..., None, :]                             # (b,T,NCV,NMV)
        there = alive.amax(1) > 0                                            # pairs in service somewhere in the window
        bi, ci, mi = there.nonzero(as_tuple=True)
        if bi.numel() == 0:
            return obs.new_zeros(b, T, NMV, TOKEN), obs.new_zeros(b, T, NCV, NMV, NK), torch.cat((hp.reshape(b, SPAIR), hs, hf), -1)
        on = alive[bi, :, ci, mi][..., None]                                  # (A,T,1)
        link = links[bi, :, ci, mi][..., None]
        x = torch.cat((symlog(cv*cm[..., None])[bi, :, ci], symlog(mv*mm[..., None])[bi, :, mi], previous[bi, :, mi][..., None], link), -1)
        pi = ci*NMV+mi
        h = hp[bi, pi]
        # ---- identify
        xin = self.inp(x)
        if seen is not None:
            xin = xin+seen[bi, :, ci, mi].to(xin.dtype)
        if feats is not None:
            xin = xin+self.stat(feats[bi, :, ci, mi][..., :NSTAT].to(xin.dtype))
            if self.stats.slow > 0:
                xin = xin+self.stat2(feats[bi, :, ci, mi][..., self.stats.at_slow].to(xin.dtype))
            if self.stats.arx:
                fa = feats[bi, :, ci, mi][..., self.stats.at_arx].to(xin.dtype)
                xin = xin+self.stat3(fa if self.stats.sets == 2 else torch.cat((fa, torch.zeros_like(fa)), -1))
        o1, h1 = self.gru1(xin, h[None, :, :P].contiguous())
        o1 = o1*on
        dense = o1.new_zeros(b, NCV, NMV, T, P).index_put((bi, ci, mi), o1)
        cnt = there.to(o1.dtype)
        rmean = dense.sum(2)/cnt.sum(2).clamp_min(1)[..., None, None]          # (b,NCV,T,P) over the MVs of a row
        cmean = dense.sum(1)/cnt.sum(1).clamp_min(1)[..., None, None]          # (b,NMV,T,P) over the rows of an MV
        o2, h2 = self.gru2(torch.cat((o1, rmean[bi, ci], cmean[bi, mi]), -1), h[None, :, P:].contiguous())
        est = self.read(o2)                                                  # (A,T,NK)
        if 'S' in self.diff:                                      # read as the change from knot to knot: the responses by adding them up
            est = values_of(est, FM.S_CLIP).to(est.dtype)
        lab = est.new_zeros(b, NCV, NMV, T, NK).index_put((bi, ci, mi), est*on).permute(0, 3, 1, 2, 4)
        if code is not None:
            w = code[1].to(est.dtype)[bi][:, None, None]
            est = w*code[0].permute(0, 2, 3, 1, 4)[bi, ci, mi].to(est.dtype)+(1-w)*est
        # ---- control
        e = self.edge(torch.cat((est, link, o2), -1))*on                      # (A,T,D)
        if self.told_in is not None and code is not None:         # how much of the responses it has here was told
            e = e+self.told_in(w)*on
        cmT, mmT = cm.transpose(1, 2)[..., None], mm.transpose(1, 2)[..., None]
        r, m = self.row0(ce).transpose(1, 2)*cmT, self.mv0(me).transpose(1, 2)*mmT     # (b,NCV,T,D), (b,NMV,T,D)
        nr, nc = cnt.sum(2).clamp_min(1).sqrt()[..., None, None], cnt.sum(1).clamp_min(1).sqrt()[..., None, None]
        if self.tied:
            ge = torch.cat((e, est, link), -1)
            gr, gm = self.gate_row(ge)*on, self.gate_mv(ge)*on
            for k in range(self.tied):
                agg = torch.zeros_like(r).index_put((bi, ci), gr*self.say_mv(m)[bi, mi], accumulate=True)/nr**2     # the mean over the row's MVs in service,
                r = r+self.up_row(torch.cat((r, agg), -1))*cmT
                agg = torch.zeros_like(m).index_put((bi, mi), gm*self.say_row(r)[bi, ci], accumulate=True)/nc**2     # over the MV's rows: the same scale at any size
                m = m+self.up_mv(torch.cat((m, agg), -1))*mmT
        for k in range(0 if self.tied else ROUNDS):
            msg = self.to_row[k](torch.cat((e, m[bi, mi]), -1))*on
            agg = torch.zeros_like(r).index_put((bi, ci), msg, accumulate=True)/nr
            r = r+self.up_row[k](torch.cat((r, agg), -1))*cmT
            msg = self.to_mv[k](torch.cat((e, r[bi, ci]), -1))*on
            agg = torch.zeros_like(m).index_put((bi, mi), msg, accumulate=True)/nc
            m = m+self.up_mv[k](torch.cat((m, agg), -1))*mmT
        hp = hp.index_put((bi, pi), torch.cat((h1[0], h2[0]), -1))
        dj = self.out(m)
        if self.law:
            told = code is not None and len(code) > 2 and not self.given
            wb = code[1].to(dj.dtype)[:, None, None] if code is not None else None
            # a `given` network works the law out in the episodes whose responses are given (sel; g: their share), no others
            sel = (code[1] > 0).nonzero()[:, 0] if self.given and code is not None else None
            sel = None if sel is None or sel.numel() == 0 else sel
            g = wb.float() if sel is not None else None
            work = self.fm is not None and (sel is not None or not self.given)
            if self.fm is not None:                               # the moves made go on being kept in every episode
                mem, hf = self.fm.memory(obs, hf)
                hf = hf.to(state.dtype)
            if work:                                              # the responses as numbers (bands per full travel), of the episodes worked out
                Sd = torch.sinh(est.float().clamp(-FM.S_CLIP, FM.S_CLIP))*on
                Sd = Sd.new_zeros(b, NCV, NMV, T, NK).index_put((bi, ci, mi), Sd).permute(0, 3, 1, 2, 4)      # (b,T,NCV,NMV,NK)
                of = (lambda x: x) if sel is None else (lambda x: None if x is None else x[sel])              # noqa: E731
                back = (lambda x: x) if sel is None else (lambda x: x.new_zeros((b,)+x.shape[1:]).index_copy(0, sel, x))   # noqa: E731
                ob, Sd, mem = of(obs), of(Sd), tuple(of(x) for x in mem)
            Ef = El = None
            if self.worked and work:
                # E worked out; the readout sees it, the rate the moves do not explain and which rows are levels, and
                # gives a factor (0..2, 1 at the start) and a correction (0 at the start)
                Ew, rate, lev = (x.transpose(1, 2) for x in self.fm.heading(ob, Sd, mem))                   # (b',NCV,T,NLAB), (b',NCV,T) twice
                rs, also = of(r), ()
                if self.hybrid:                                   # the learned kernel's pending effects (as below), beside the formula's E
                    xf = feats[bi, :, ci, mi][..., NSTAT:NSTAT+NF].to(e.dtype)
                    pk = (self.kern(torch.cat((est, link, o2), -1)).unflatten(-1, (NLAB, self.nf))*xf[..., None, :]).sum(-1)*on
                    also = (torch.asinh(of(torch.zeros_like(r[..., :NLAB]).index_put((bi, ci), pk, accumulate=True))),)
                h = (self.adjust if self.given else self.heading)(torch.cat((rs,)+also+(torch.asinh(Ew).to(r.dtype), torch.asinh(rate)[..., None].to(r.dtype), lev[..., None].to(r.dtype)), -1)).float()
                Ef = back(torch.asinh(2*torch.sigmoid(h[..., -1:])*Ew+torch.sinh(h[..., :-1].clamp(-E_CLIP, E_CLIP)))*of(cmT))
            if not self.worked or self.given:
                # what the MV's moves so far have yet to do to the row: a kernel read from the pair's response, times
                # the moves as the filters hold them (the responses are per full MV span, the move feature spans 2)
                xf = (self.reader.keys[bi, :, mi] if self.reader is not None else feats[bi, :, ci, mi][..., NSTAT:NSTAT+NF]).to(e.dtype)
                pend = (self.kern(torch.cat((est, link, o2), -1)).unflatten(-1, (NLAB, self.nf))*xf[..., None, :]).sum(-1)*on
                pend = torch.zeros_like(r[..., :NLAB]).index_put((bi, ci), pend, accumulate=True)
                if self.unseen is None:
                    El = self.heading(torch.cat((r, torch.asinh(pend)), -1))                  # (b,NCV,T,NLAB)
                else:                                             # what the instruments do not show, read from the row's token; E is read with it
                    uh = self.unseen(r)*cmT                                                   # (b,NCV,T,UNSEEN)
                    U = uh.detach()                               # E's read-out takes it as a number: its own labels alone teach the reading, not E's loss
                    if told and len(code) > 5:                    # (the told episodes: with the true one)
                        U = wb[..., None]*code[5].transpose(1, 2).to(uh.dtype)+(1-wb[..., None])*U
                    El = self.heading[2](self.heading[1](self.heading[0](torch.cat((r, torch.asinh(pend)), -1))+self.unseen_in(torch.asinh(U))))
                    self.read_unseen = uh.transpose(1, 2)
                if 'E' in self.diff:                              # read as the change from lag to lag
                    El = values_of(El, E_CLIP).to(El.dtype)
                El = El*cmT
            eh = E = Ef if El is None else El                     # what the readout that learns E is taught on: every episode
            if told and not self.worked:
                E = wb[..., None]*code[2].transpose(1, 2).to(eh.dtype)+(1-wb[..., None])*eh
            elif Ef is not None and El is not None:               # the E in use: the formula's in the episodes given the responses
                E = g[..., None]*Ef+(1-g[..., None])*El.float()
            Es = torch.sinh(E.float().clamp(-E_CLIP, E_CLIP))
            kh = vh = ms = ml = None
            if self.solve and work:                               # the design solved on the responses for that E
                ms = back(self.fm.move(ob, Sd, of(Es).transpose(1, 2), mem).transpose(1, 2))  # (b,NMV,T) in slew limits
            if not self.solve or self.given:
                kh = self.gain(torch.cat((e, r[bi, ci], m[bi, mi]), -1))*on                  # (A,T,NLAB)
                vh = self.valve(m)[..., 0]*mmT[..., 0]                                       # (b,NMV,T)
                K, V = kh, vh
                if told:
                    K = w*code[3].permute(0, 2, 3, 1, 4)[bi, ci, mi].to(kh.dtype)+(1-w)*kh
                    V = wb*code[4].transpose(1, 2).to(vh.dtype)+(1-wb)*vh
                prod = (K_UNIT*torch.sinh(K.float().clamp(-K_CLIP, K_CLIP))*Es[bi, ci]).sum(-1)*on[..., 0]
                ml = torch.zeros_like(m[..., 0], dtype=prod.dtype).index_put((bi, mi), prod, accumulate=True)   # (b,NMV,T): the change of flow
                ml = ml/(V_UNIT*torch.exp(V.float().clamp(-V_CLIP, V_CLIP)))                 # in slew limits
                kh, vh = kh.new_zeros(b, NCV, NMV, T, NLAB).index_put((bi, ci, mi), kh).permute(0, 3, 1, 2, 4), vh.transpose(1, 2)
            mv = ml if ms is None else ms if ml is None else g*ms+(1-g)*ml
            self.read_given = Ef.transpose(1, 2) if self.given and Ef is not None else None
            if not self.apart:                                    # (apart: E, K, V and their product are read out to be taught; the MVs' tokens get nothing of them)
                dj = dj+self.law_in(torch.stack((torch.asinh(mv), mv.clamp(-1., 1.)), -1).to(dj.dtype))*mmT
            self.read_law = (eh.transpose(1, 2), kh, vh, mv.transpose(1, 2))
        return dj.transpose(1, 2), lab, torch.cat((hp.reshape(b, SPAIR), hs, hf), -1)


def trunk(hist, pairs, obs, previous, hidden=None, code=None, feats=None):
    """A History and a PairNet over a packet sequence -> the MV tokens the heads read (b,T,NMV,TOKEN: with the
    pair pathway added), the global memory (b,T,MEMORY), the MV mask (b,T,NMV), the responses read
    (b,T,NCV,NMV,knots), the state (1,b,BASE+pairs.size: STATE, and what the formula keeps behind it). hidden
    None, or shorter (a state from before the pair pathway, from before its running sums, or of STATE for a
    network with the formula): the rest starts at zero. feats: as PairNet.forward."""
    ncv, nmv = shape_of(obs.shape[-1])                            # the packets' size
    BASE = base_size(ncv, nmv)
    b, size = obs.shape[0], BASE+pairs.size_of(ncv, nmv)
    if hidden is None:
        hidden = obs.new_zeros(1, b, size)
    elif hidden.shape[-1] < size:
        hidden = torch.cat((hidden, hidden.new_zeros(1, b, size-hidden.shape[-1])), -1)
    hg, ht = hidden[..., :MEMORY].contiguous(), hidden[..., MEMORY:BASE].contiguous()
    ce, me, cm, mm, links = hist.channels.embed(obs, previous)
    ce, me, _, ht = hist.tokens(ce, me, cm, mm, links, ht)
    joint, x, mask = hist.channels.combine(ce, me, cm, mm, links)
    memory, hg = hist.gru(x, hg)
    dj, lab, hp = pairs(obs, previous, ce, me, hidden[0, :, BASE:], code, feats)
    return joint+dj*mm[..., None], memory, mask, lab, torch.cat((hg, ht, hp[None]), -1)


class AIO(nn.Module):
    """The actor with the pair pathway. `steps` is the one entry: sequences or single decisions alike."""
    canonical = True                                              # the direct-acting frame (aio.frame.canonical)

    def __init__(self, stats=True, tied=0, law=False, knots=NLAB, formula=False, solve=False, given=False, slow=0., arx=False, dv=False, diff='', unseen=False, told=False, reader=False, apart=None, hybrid=False):
        super().__init__()
        self.actor = Actor()
        self.pairs = PairNet(stats, tied, law, knots, formula, solve, given, slow, arx, dv, diff, unseen, told, reader, apart, hybrid)

    def steps(self, obs, previous, hidden=None, code=None, feats=None):
        """-> mean (b,T,NMV) before the tanh, the head's second output (b,T,NMV; `probing`), the MV mask
        (b,T,NMV), the responses read (b,T,NCV,NMV,knots), the state (1,b,STATE; more with the formula: `trunk`).
        feats: as PairNet.forward."""
        joint, memory, mask, lab, hidden = trunk(self.actor.history, self.pairs, obs, previous, hidden, code, feats)
        out = self.actor.head(self.actor.head_input(joint, memory))[0].float()
        return out[..., 0]*mask, out[..., 1], mask, lab, hidden

    def state_size(self, ncv=NCV, nmv=NMV):
        """Length of the whole state `steps` carries for packets of ncv rows and nmv MVs (with what --slow, --arx, --dv and
        the formula add): what a caller that keeps the state itself starts at zero."""
        return base_size(ncv, nmv)+self.pairs.size_of(ncv, nmv)

    def legacy(self):
        """The interface of the earlier actor (history, distribution), for code that drives one
        (aio.sim.control.Policy)."""
        return _Legacy(self)


class _Legacy:
    canonical = True

    def __init__(self, net):
        self.net = net

    def history(self, obs, previous, hidden):
        mean, _, mask, _, hidden = self.net.steps(obs, previous, hidden)
        return mean[..., None], None, mask, hidden

    def distribution(self, joint, memory, mask):
        return joint[..., 0]*mask, torch.zeros_like(joint[..., 0]), None

    def eval(self):
        self.net.eval(); return self

    def train(self, mode=True):
        self.net.train(mode); return self


class Critic(nn.Module):
    """Q(history, move): the symlog of the cost to come, from `heads` heads on one trunk (History and PairNet of
    its own). The move enters at each MV's token."""

    def __init__(self, heads=2, stats=True, tied=0, knots=NLAB):
        super().__init__()
        self.history = History()
        self.pairs = PairNet(stats, tied, knots=knots)
        self.action = _in(TOKEN+1, TOKEN)
        self.heads = nn.ModuleList([ExpertHead(TOKEN+MEMORY, 1) for _ in range(heads)])

    def steps(self, obs, previous, action, hidden=None, code=None, feats=None):
        """action (b,T,NMV): the move made at each decision -> (b,T,heads) symlog values, the state (1,b,STATE)."""
        joint, memory, mask, _, hidden = trunk(self.history, self.pairs, obs, previous, hidden, code, feats)
        acted = self.action(torch.cat((joint, (action*mask)[..., None]), -1))
        x = torch.cat((pool(acted, mask), memory), -1)
        return torch.cat([h(x)[0] for h in self.heads], -1), hidden

    def state_size(self, ncv=NCV, nmv=NMV):
        """As AIO.state_size, of the critic's own trunk."""
        return base_size(ncv, nmv)+self.pairs.size_of(ncv, nmv)


PRIV_CV, PRIV_MV, PRIV_EDGE, PRIV_G = 25, 10, 4, 2


def priv_features(prog, st, k):
    """Differentiable privileged simulator state features at decision k (training-only Critic; never deployed).
    Preserves exact autograd paths through true noiseless yt, v, fl, l1, l2, sr, si, dead-time-indexed delay
    pipeline buf[0..dd], Smith-predictor ultimate error err_inf, integrating-level in-flight pipe drift, and
    valve states u, ua, xv so moves made inside long dead times (dd up to 60) near the window boundary get
    exact, aligned gradients with zero leak to expired slots s >= dd.
    -> (cv_feat (n,NCV,PRIV_CV), mv_feat (n,NMV,PRIV_MV), edge (n,NCV,NMV,PRIV_EDGE), cm (n,NCV), mm (n,NMV), g (n,PRIV_G))."""
    from aio.sim import core as C
    from aio.sim.prior import DMAX, HORIZON
    NCV, NMV = prog.ncv, prog.nmv
    c, n, dev = prog.c, prog.n, prog.device
    k_cl = min(int(k), HORIZON-1)
    seg = prog.seg_at(k_cl)
    P, t = prog.params(seg, k_cl), prog.task(seg)
    phi = C._phi(st['xv'], st['d'], c)
    ref, alo, ahi, zlo, zhi, rref = C._reference(st, P, phi.detach(), c, t)
    band = c['bandf'].float().clamp_min(1e-6)
    rowm, prc, trk, zon = c['rowm'].float(), c['prc'].float(), c['trk'].float(), c['zon'].float()
    mvm = c['mvm'].float()
    ig = (c['integ'] > 0).float()
    rst = (c['rst'] > 0).float()
    yt, v, fl = st['yt'].float(), st['v'].float(), st['fl'].float()
    err_b = (yt-ref.float())/band
    ex_trk = torch.maximum(alo.float()-1.-err_b, err_b-ahi.float()-1.)*trk
    ex_zon = (torch.maximum(zlo.float()-yt, yt-zhi.float())/band-ahi.float())*zon
    tint = c['tint'].float().clamp_min(1e-6)
    lev_rate = (fl+c['lam'].float()*(v-c['vc'].float()))/tint/band*ig
    # Exact local gain, dead time dd, and pole time constant per (CV, MV) pair
    K_mv = P['K'][:, :, :NMV].float()/band[:, :, None]*prc[:, :, None]*mvm[:, None, :]
    e = phi[:, None, None]
    dd = (P['dead'][:, :, :NMV]/e).round().clamp(0, DMAX).long()                   # (n, NCV, NMV)
    dd_f = dd.float()
    tau_p = (-1./(P['p1r'][:, :, :NMV].float()*e.float()).clamp_max(-1e-3)).clamp(0., 256.)
    cr = torch.stack((P['c0r'][:, :, :NMV], P['c1r'][:, :, :NMV]), -1).float()
    ci = torch.stack((P['c0i'][:, :, :NMV], P['c1i'][:, :, :NMV]), -1).float()
    mode_mv = (cr*st['sr'][:, :, :NMV].float()-ci*st['si'][:, :, :NMV].float()).sum(-1)
    # Exact dead-time-indexed slices of the delay pipe buf (0 = newest move, dd = exiting now, >= dd expired)
    buf_mv = st['buf'][:, :, :NMV].float()                                        # (n, DMAX+1, NMV)
    buf_now = buf_mv[:, 0]
    s_idx = torch.arange(buf_mv.shape[1], device=dev)[None, None, None, :]        # (1, 1, 1, D+1)
    dd_e = dd[..., None]                                                          # (n, NCV, NMV, 1)
    b_exp = buf_mv.transpose(1, 2)[:, None, :, :]                                 # (n, 1, NMV, D+1)
    buf_exit = torch.gather(b_exp.expand(n, NCV, NMV, -1), 3, dd_e)[..., 0]
    m_next16 = (s_idx >= (dd_e-16).clamp_min(0)) & (s_idx < dd_e)
    cnt_n16 = m_next16.float().sum(-1)
    buf_n16 = torch.where(cnt_n16 > 0, (b_exp*m_next16.float()).sum(-1)/cnt_n16.clamp_min(1.), buf_now[:, None, :])
    m_pipe = s_idx < dd_e
    cnt_pipe = m_pipe.float().sum(-1)
    buf_pipe = torch.where(cnt_pipe > 0, (b_exp*m_pipe.float()).sum(-1)/cnt_pipe.clamp_min(1.), buf_now[:, None, :])
    m_new16 = s_idx < dd_e.clamp_max(16)
    cnt_new16 = m_new16.float().sum(-1)
    buf_new16 = torch.where(cnt_new16 > 0, (b_exp*m_new16.float()).sum(-1)/cnt_new16.clamp_min(1.), buf_now[:, None, :])
    # Differentiable Smith-predictor features & in-flight integrating-level pipe drift
    pend_exit = (K_mv*(buf_exit-mode_mv)).sum(-1)
    pend_n16 = (K_mv*(buf_n16-mode_mv)).sum(-1)
    pend_pipe = (K_mv*(buf_pipe-mode_mv)).sum(-1)
    pend_new16 = (K_mv*(buf_new16-mode_mv)).sum(-1)
    pend_inf = (K_mv*(buf_now[:, None, :]-mode_mv)).sum(-1)
    err_inf = (err_b+pend_inf)*(1.-ig)*trk
    lev_pipe_drift = ((K_mv*((b_exp-mode_mv[..., None])*m_pipe.float()).sum(-1)).sum(-1)/tint)*ig
    lev_inf_rate = (lev_rate+pend_inf/tint)*ig
    dist_eff = ((P['K'][:, :, NMV:].float()*st['d'][:, None, :].float()).sum(-1)+st['od'].float())/band*prc
    act_pair = (K_mv.abs() > 1e-4).float()
    max_dd_row = (dd_f*act_pair).amax(-1)/64.
    max_tau_row = (tau_p*act_pair).amax(-1)/64.
    cv_feat = torch.stack((
        symlog(err_b*trk),
        symlog((yt-st['target'].float())/band*trk),
        symlog((yt-rref.float())/band*rst),
        symlog(ex_trk.clamp_min(-2.)),
        symlog(ex_zon.clamp(-5., 50.)),
        symlog((v-c['vc'].float())/band*prc),
        symlog(lev_rate),
        symlog(fl/band*prc),
        symlog((st['l1'].float()-fl)/band*prc),
        symlog((st['l2'].float()-fl)/band*prc),
        symlog(pend_exit),
        symlog(pend_n16),
        symlog(pend_pipe),
        symlog(pend_new16),
        symlog(pend_inf),
        symlog(err_inf),
        symlog(lev_pipe_drift),
        symlog(lev_inf_rate),
        symlog(dist_eff),
        max_dd_row,
        max_tau_row,
        trk,
        zon,
        ig,
        symlog(t['cvw'].float()*trk),
    ), -1)*rowm[..., None]
    span = (t['uhi']-t['ulo']).float().clamp_min(1e-6)
    slew = c['slew'].float().clamp_min(1e-6)
    u, ua, xv = st['u'].float(), st['ua'].float(), st['xv'].float()
    buf_4 = buf_mv[:, :4].mean(1)
    buf_16 = buf_mv[:, 4:16].mean(1)
    mv_feat = torch.stack((
        2*(u-t['ulo'].float())/span-1.,
        2*(ua-t['ulo'].float())/span-1.,
        2*(xv-t['ulo'].float())/span-1.,
        ((u-st['prev_u'].float())/slew).clamp(-2., 2.),
        ((u-t['ulo'].float())/slew).clamp(-5., 50.)/10.,
        ((t['uhi'].float()-u)/slew).clamp(-5., 50.)/10.,
        (slew/span).clamp(0., 2.),
        symlog((buf_now-buf_4)*5.),
        symlog((buf_4-buf_16)*5.),
        mvm,
    ), -1)*mvm[..., None]
    edge = torch.stack((
        symlog(K_mv),
        c['link'][:, :, :NMV].float(),
        dd_f/64.,
        tau_p/64.,
    ), -1)*rowm[:, :, None, None]*mvm[:, None, :, None]
    # Pure Markov state globals (NO time-of-episode k_rem: prevents episode-end bootstrap self-amplification!)
    g = torch.stack(((trk+zon).sum(-1)/float(max(NCV, 1)), st['fail'].float()), -1)
    return cv_feat, mv_feat, edge, rowm, mvm, g


class _PrivBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3*d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2*d), nn.SiLU(), nn.Linear(2*d, d))
        self.eb = nn.Linear(PRIV_EDGE, heads, bias=False)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.ff[-1].weight); nn.init.zeros_(self.ff[-1].bias)

    def forward(self, x, on, edge, ncv):
        b, N, d = x.shape
        h, dh = self.h, d//self.h
        q, k, v = self.qkv(self.n1(x)).reshape(b, N, 3, h, dh).permute(2, 0, 3, 1, 4)
        bias = x.new_zeros(b, h, N, N)
        eb = self.eb(edge).permute(0, 3, 1, 2)                                       # (b, h, NCV, NMV)
        bias[:, :, :ncv, ncv:] = eb
        bias[:, :, ncv:, :ncv] = eb.transpose(-1, -2)
        bias = bias.masked_fill(~(on[:, None, None, :] > 0), float('-inf'))
        bias = torch.where(torch.isinf(bias).all(-1, keepdim=True), torch.zeros_like(bias), bias)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x+self.out(y.transpose(1, 2).reshape(b, N, d))*on[..., None]
        return x+self.ff(self.n2(x))*on[..., None]


class PrivCritic(nn.Module):
    """Training-only differentiable Markov state-value critic V(st) on exact simulator state (`priv_features`).
    Size-agnostic across 12x5, 24x10, 48x20; zero recurrent state (one call at window boundary, parallel batched fit)."""
    privileged = True

    def __init__(self, heads=2, d=256, layers=2, nheads=4):
        super().__init__()
        self.d = d
        self.cv_in = nn.Sequential(nn.Linear(PRIV_CV, d), nn.LayerNorm(d), nn.SiLU())
        self.mv_in = nn.Sequential(nn.Linear(PRIV_MV, d), nn.LayerNorm(d), nn.SiLU())
        self.g_in = nn.Linear(PRIV_G, d)
        self.blocks = nn.ModuleList([_PrivBlock(d, nheads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head_norm = nn.LayerNorm(3*d+PRIV_G)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(3*d+PRIV_G, d), nn.SiLU(), nn.Linear(d, d//2), nn.SiLU(), nn.Linear(d//2, 1))
            for _ in range(heads)
        ])
        for hd in self.heads:
            nn.init.constant_(hd[-1].bias, 3.5)

    def forward(self, cv_feat, mv_feat, edge, cm, mm, g):
        """Leading shape (..., NCV, PRIV_CV) etc. -> (..., heads) symlog cost-to-come."""
        lead = cv_feat.shape[:-2]
        ncv, nmv = cv_feat.shape[-2], mv_feat.shape[-2]
        cv_f = cv_feat.reshape(-1, ncv, PRIV_CV)
        mv_f = mv_feat.reshape(-1, nmv, PRIV_MV)
        ed = edge.reshape(-1, ncv, nmv, PRIV_EDGE)
        cm_f, mm_f, g_f = cm.reshape(-1, ncv), mm.reshape(-1, nmv), g.reshape(-1, PRIV_G)
        on = torch.cat((cm_f, mm_f), 1)
        x = torch.cat((self.cv_in(cv_f), self.mv_in(mv_f)), 1)+self.g_in(g_f)[:, None, :]
        x = x*on[..., None]
        for blk in self.blocks:
            x = blk(x, on, ed, ncv)
        x = self.norm(x)*on[..., None]
        cv_h, mv_h = x[:, :ncv], x[:, ncv:]
        cv_mean = (cv_h*cm_f[..., None]).sum(1)/cm_f.sum(1, keepdim=True).clamp_min(1.)
        cv_max = cv_h.masked_fill(~(cm_f[..., None] > 0), -1e9).amax(1)
        cv_max = torch.where((cm_f.sum(1, keepdim=True) > 0), cv_max, torch.zeros_like(cv_max))
        mv_mean = (mv_h*mm_f[..., None]).sum(1)/mm_f.sum(1, keepdim=True).clamp_min(1.)
        feat = self.head_norm(torch.cat((cv_mean, cv_max, mv_mean, g_f), -1))
        return torch.cat([hd(feat) for hd in self.heads], -1).reshape(*lead, len(self.heads))

    def eval_state(self, prog, st, k):
        """Differentiable V(st_k) -> (n, heads) symlog values, and detached feature tuple for `fit_critic`."""
        feats = priv_features(prog, st, k)
        return self(*feats), tuple(x.detach() for x in feats)


def probing(second):
    """Logit of "this unit has not been disturbed yet: move its valves to see how it answers", from the second
    output of the actor's head (`steps`). A network trained without --probe never has it above 0 by training;
    it is read only where `net.probe` is set."""
    return second-2.


def law_code(prog, st, k, s, view=None):
    """The teacher's E, K and V at decision k (aio.law) as the network's readouts are: (n, NCV, NLAB) asinh(E)
    of the tracked rows (the first NMV row slots; 0 on the others), (n, NCV, NMV, NLAB) asinh(K/K_UNIT) in the
    direct-acting frame (s (n, NMV): the frame's signs), (n, NMV) log(V/V_UNIT); then the law's move (n, NMV)
    in slew limits as it comes out of them, and held to the limits (aio.law.held: what is imitated), both in
    that frame."""
    from aio import law as L
    m, E, K, V = L.move(prog, st, k, view)
    n, nmv = E.shape[0], E.shape[2]
    NCV, NMV = prog.ncv, prog.nmv                                 # the program's size
    K = K.float()*s[:, :, None, None]
    Ec, Kc = E.new_zeros(n, NCV, NLAB, dtype=torch.float32), E.new_zeros(n, NCV, NMV, NLAB, dtype=torch.float32)
    Ec[:, :nmv] = torch.asinh(E.float().transpose(1, 2))
    Kc[:, :nmv] = torch.asinh(K.permute(0, 3, 1, 2)/K_UNIT)
    return Ec, Kc, torch.log(V.float()/V_UNIT), m.float()*s, L.held(prog, st, k, m).float()*s


def unseen_code(prog, st, k):
    """What the instruments do not show of the tracked rows at decision k, as the read-out `unseen` gives it:
    (n, NCV, UNSEEN), 0 on the other row slots. Before the output curve and weighted as E is (weighted bands):
    [0] the row's true value less its reading (noise, the sensor's lag, an analyzer's last sample, a reading that
        is not there), in GAP_UNIT;
    [1] the rate the row goes on at with nothing of the moves made still to come, a decision: a level's net inflow
        and leak (what its level does next: aio.sim.core.step), 0 of a row that does not integrate, in RATE_UNIT.
    Which rows integrate is known here alone, where the labels are made; nothing in the packets says it."""
    from aio.sim import prior as PR
    c, nmv = prog.c, prog.wy.shape[2]
    f = lambda x: x[:, :nmv]                                      # noqa: E731
    wy, lev = prog.wy[prog.ar, prog.teacher_seg(k)], f(c['integ']) > 0
    read = PR.hinv(f(st['y']).clamp(-1., 2.), f(prog.params(prog.seg_at(k), k)['wa']))[0]
    rate = (f(st['fl'])+f(c['lam'])*(f(st['v'])-f(c['vc'])))/f(c['tint'])
    out = wy.new_zeros(wy.shape[0], prog.ncv, UNSEEN, dtype=torch.float32)
    out[:, :nmv, 0] = ((torch.where(lev, f(st['v']), f(st['fl']))-read)*wy).float()/GAP_UNIT
    out[:, :nmv, 1] = (torch.where(lev, rate, torch.zeros_like(rate))*wy).float()/RATE_UNIT
    return out.nan_to_num(0., 0., 0.)


class Policy:
    """The network as a controller of aio.sim.control. known: it is told the true responses instead of reading
    them from the history (how well it controls a plant it knows; never how it runs); a network with the law
    is then told the teacher's E, K and V as well (a `given` network the responses alone: it works the rest out).
    A network trained with
    --probe (net.probe > 0) adds noise of that spread before the tanh to its moves while it says the unit has
    not been disturbed yet (`probing`): its own plant test, before the first event, where nothing is scored.
    The noise is drawn from the program's seed: the same rollout every time. Without it, deterministic.
    stay: its memory is kept from one program to the next on the same plants (`resume`: a rollout that takes the
    plant over in the state the one before ended in; aio.sim.core.Program, `after`); `restart` clears the
    episodes that start again.
    read: the error of the responses it reads against the true ones is summed (`readout`: id_own, id_cross, as the
    trainer's rows; the same over the decisions within NEAR of the last event that changed something and over
    the quiet ones after, `_near` and `_quiet`: in a quiet closed loop the sums that forget lose the plant). shadow S: for the last S decisions of a program a second memory is run beside its own, from
    zero, on the same packets and moves; `forgetful` is this network with that memory: what it would be had it
    forgotten everything older than S decisions (the control for what the longer memory is worth).
    A network of a run with --unpaired (net.unpaired > 0) gets the packets with nothing published about the pairing
    (which row an MV is published with, which way it acts on it; aio.frame.canonical) and works in the plant's own frame;
    paired=True: with both published all the same (the control for what they are worth to it)."""

    def __init__(self, net, known=False, stay=False, read=False, shadow=0, paired=None):
        self.net, self.known, self.probe = net, known, float(getattr(net, 'probe', 0.))
        # a network of a run with --unpaired runs with nothing published about the pairing (aio.frame.canonical); paired=True: told all the same
        self.unpaired = blind_main(net) if paired is None else not paired
        self.stay, self.read, self.shadow, self.h, self.hb = stay, read, shadow, None, None

    def reset(self, prog, keep=False):
        if not (keep and self.stay and self.h is not None):
            self.h = torch.zeros(1, prog.n, state_size(prog.ncv, prog.nmv), device=prog.device)
            self.prev = torch.zeros(prog.n, prog.nmv, device=prog.device)
        self.g = torch.Generator(device=prog.device).manual_seed(int(getattr(prog, 'seed', 0)))
        self.err = torch.zeros(2, 2, 2, dtype=torch.float64, device=prog.device)   # near an event / quiet, published pairs / the others, sum / count

    def resume(self, prog):
        """The next window on the same plants (aio.sim.control.rollout, `st`): with `stay` the memory goes on."""
        self.reset(prog, keep=True)

    def restart(self, again):
        """again (n,): the episodes whose units start again at rest; their memory starts again too."""
        keep = (~again).float()
        self.h, self.prev = self.h*keep[None, :, None], self.prev*keep[:, None]
        if self.hb is not None:
            self.hb = self.hb*keep[None, :, None]

    def forgetful(self):
        """This network with the shadow memory and the same last moves, as a controller that keeps them."""
        other = Policy(self.net, self.known, stay=True)
        other.h, other.prev = self.hb.clone(), self.prev.clone()
        return other

    def readout(self):
        """Mean squared error of the responses read over the rollout: published pairs, the others; then both within
        NEAR decisions of the last event and in the quiet after (None where there were no such decisions)."""
        e, a = self.err, self.err.sum(0)
        out = dict(id_own=float(a[0, 0]/a[0, 1].clamp_min(1.)), id_cross=float(a[1, 0]/a[1, 1].clamp_min(1.)))
        for i, when in enumerate(('near', 'quiet')):
            for j, q in enumerate(('id_own', 'id_cross')):
                out[f'{q}_{when}'] = float(e[i, j, 0]/e[i, j, 1]) if float(e[i, j, 1]) > 0 else None
        return out

    @torch.no_grad()
    def __call__(self, prog, st, k, obs):
        from aio.frame import canonical
        from aio.sim.prior import HORIZON
        o, s = canonical(obs.float(), torch.ones(prog.n, dtype=torch.bool, device=obs.device) if self.unpaired else None)
        true = pair_labels(prog.labels(k, self.net.pairs.nk).float()*s[:, :, None]) if self.known or self.read else None
        code = (true[:, None], torch.ones(prog.n, device=prog.device)) if self.known else None
        if self.known and self.net.pairs.law:
            code = code+tuple(x[:, None] for x in law_code(prog, st, k, s)[:3])
        if self.shadow and k >= HORIZON-self.shadow:              # the memory of the last decisions alone, on what this one saw and did
            hb = None if k == HORIZON-self.shadow else self.hb
            self.hb = self.net.steps(o[:, None], self.prev[:, None], hb, code)[4]
        mean, second, mask, lab, self.h = self.net.steps(o[:, None], self.prev[:, None], self.h, code)
        if self.read:
            e = (lab[:, 0]-true).square().mean(-1)
            w = prog.c['prc'].float()[:, :, None]*mask[:, 0][:, None, :]*(~st['fail']).float()[:, None, None]
            link = prog.c['link'][:, :, :prog.nmv].float()
            near = (prog.since(k) < NEAR).float()[:, None, None] if hasattr(prog, 'since') else torch.ones_like(w[:, :1, :1])
            for i, t in enumerate((near, 1-near)):
                self.err[i] += torch.stack((torch.stack(((e*w*link*t).sum(), (w*link*t).sum())),
                                            torch.stack(((e*w*(1-link)*t).sum(), (w*(1-link)*t).sum())))).double()
        z = mean[:, 0]
        if self.probe and (self.unpaired or not getattr(self.net, 'unpaired', 0.)):
            z = z+self.probe*(probing(second[:, 0]) > 0)*torch.randn(z.shape, generator=self.g, device=z.device)
        a = torch.tanh(z)*mask[:, 0]
        self.prev = a
        return (a*s).to(prog.dtype)


class Controller:
    """The network on the public packets of one plant (aio.interface.encode), as it
    runs: it has the packet alone and keeps its own memory. step(packet) -> the move of every MV slot in
    [-1, 1] of its slew limit (0 on slots not in use). One per tier of aio.tiers."""

    def __init__(self, net):
        self.net, self.probe, self.unpaired = net, float(getattr(net, 'probe', 0.)), blind_main(net)
        self.reset()

    def reset(self):
        self.h, self.prev = None, None                            # (the last move: as wide as the first packet's MVs)
        self.g = torch.Generator().manual_seed(0)

    @torch.no_grad()
    def step(self, packet):
        from aio.frame import canonical
        o, s = canonical(torch.as_tensor(packet, dtype=torch.float32)[None], torch.ones(1, dtype=torch.bool) if self.unpaired else None)
        if self.prev is None:
            self.prev = torch.zeros(1, shape_of(o.shape[-1])[1])
        mean, second, mask, _, self.h = self.net.steps(o[:, None], self.prev[:, None], self.h)
        z = mean[:, 0]
        if self.probe and (self.unpaired or not getattr(self.net, 'unpaired', 0.)):      # its own plant test, as in Policy
            z = z+self.probe*(probing(second[:, 0]) > 0)*torch.randn(z.shape, generator=self.g)
        self.prev = torch.tanh(z)*mask[:, 0]
        return (self.prev*s)[0].numpy()


def blind_main(net):
    """Whether a network runs and is validated with nothing published about the pairing: a run where at least half the episodes
    were blind (--unpaired .5 and up). One trained mostly told (11-23: the pairing and the signs are commissioning knowledge) runs told."""
    return float(getattr(net, 'unpaired', 0.)) >= .5


def of_state(actor, pairs=None, config=None, device='cpu'):
    """AIO in eval mode from the state dicts of a run (`load`) or of a released file (release/huggingface): what
    the pair pathway has is read from its own entries. One from before the pair pathway (pairs None) leaves it
    at its start, where the network computes what the actor did; one without the running sums (from before
    them, or trained with --no-stats) is run without them. config: the run's (`probe`, `unpaired`)."""
    if pairs is not None and 'seq' in pairs:                      # the plain attention network (aio.seqnet): no pair pathway, its spec in `seq`
        from aio import seqnet as SQ
        asked, w = SQ.of_buffer(pairs['seq'])
        net = SQ.SeqAIO(asked, w, actor['read.2.weight'].shape[0])
        net.actor.load_state_dict(actor)
        net.probe, net.unpaired = float((config or {}).get('probe') or 0.), float((config or {}).get('unpaired') or 0.)
        return net.to(device).eval()
    kw = dict(zip(('formula', 'solve'), worked_of(pairs)), knots=knots_of(pairs), law=law_of(pairs), tied=tied_of(pairs), given=given_of(pairs),
              slow=slow_of(pairs), arx=arx_of(pairs), dv=dv_of(pairs), diff=diff_of(pairs), unseen=unseen_of(pairs), told=told_of(pairs), reader=reader_of(pairs), apart=apart_of(pairs), hybrid=hybrid_of(pairs)) if pairs is not None else {}
    wide = int(pairs['wide']) if pairs is not None and 'wide' in pairs else 1
    assert wide == WIDE, f'these weights have the pair pathway {wide} times as wide: run with AIO_WIDE={wide} in the environment'
    net = AIO(stats=pairs is not None and has_stats(pairs), **kw)
    net.actor.load_state_dict(actor)
    if pairs is not None:
        net.pairs.load_state_dict(pairs)
    net.probe = float((config or {}).get('probe') or 0.)          # a run with --probe: it runs with its own plant test (Policy)
    net.unpaired = float((config or {}).get('unpaired') or 0.)    # a run with --unpaired: it runs without the published pairing
    return net.to(device).eval()


def load(path, device='cpu'):
    """(AIO in eval mode, checkpoint dict) of a .pt file of any AIO run (`of_state`)."""
    import os
    import pathlib
    import sys
    sys.modules.setdefault('pathlib._local', pathlib)             # Python 3.13+ checkpoints loaded on Python <=3.12
    saved = pathlib.WindowsPath
    if os.name != 'nt':                                           # a checkpoint written on Windows carries a WindowsPath
        pathlib.WindowsPath = pathlib.PureWindowsPath
    try:
        ck = torch.load(path, map_location='cpu', weights_only=False)
    finally:
        pathlib.WindowsPath = saved
    return of_state(ck['model']['actor'], ck['model'].get('pairs'), ck.get('config'), device), ck


def released(path, device='cpu'):
    """AIO in eval mode from a folder written by release/huggingface/export.py (model.safetensors, config.json)."""
    import json
    from pathlib import Path
    from safetensors.torch import load_file
    state = load_file(str(Path(path)/'model.safetensors'))

    def part(prefix):
        return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    return of_state(part('actor.'), part('pairs.') or None, json.loads((Path(path)/'config.json').read_text(encoding='utf-8')), device)


def pair_labels(lab):
    """Labels (..., NMV, NCV*knots) as the simulator stores them -> (..., NCV, NMV, knots)."""
    return lab.unflatten(-1, (rows_of(lab.shape[-2]), -1)).transpose(-3, -2)

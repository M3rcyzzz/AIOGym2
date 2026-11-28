"""Plant and scenario prior of the simulation core (docs/sim.md).

One draw is one process unit as an APC engineer meets it, and one 720-decision episode on it. The draw is
ancestral: a few physical roots first (size, the unit's time scale, how dead-time dominant it is, how its
loops interact), then everything that depends on them, so the plants are coherent instead of a matrix of
independent entries. A plant that fails the operability check is redrawn with the same roots, so the check
does not thin out the large and the strongly interacting units.

Every operating point of the episode passes `operable` when it is generated (docs/task-rigor-checklist.md
rules 1 and 2). Setpoint moves and disturbance steps are operable by construction: the steady valve
positions are linear in the targets and the disturbance levels, so the largest feasible step along a drawn
direction is known in closed form (`reach`) and the step is cut to it. The whole event schedule is fixed
before the first step, so the plant step has no data-dependent branches, can be captured as one graph and
can be differentiated.

A share of the events takes the unit to a constrained point: a target that would need a valve beyond its
limit, a load the valves cannot follow all the way, an MV limit the operator narrows past the valve's
steady position, a zone limit the operator moves past where the CV sits, or an MV taken out of service
before a setpoint move. The targets cannot all be held there; `optimum` finds the best steady point (the
cost's own trade-off between the tracked CVs and the zone limits, with every valve inside its limits) and
the episode is scored against it (p['rtgt'], p['zal']), so the reference stays reachable (rule 4). The
next event brings the unit back.

The plant (all times in decisions, all values normalised: an MV is 0..1 of its span, a CV 0..1 of its range):

  inputs    w = [valve flows of the MVs (NMV), measured disturbances (ND), unmeasured loads (NQ)]
  pair      every CV row i and input j: gain K, dead time, two modes (first order, overdamped, inverse
            response, lead, underdamped), exact zero-order-hold discretisation
  row       sum over inputs + bias + a slow output drift -> 0-2 extra lags -> integrating or slowly
            unstable or neither -> a monotone output curve (Wiener)
  MV        slew limit -> positioner / slave-loop lag -> valve friction (stick-slip) -> flow characteristic
  load      a measured feed rate or one MV can be the throughput: every pole, zero and dead time scales with it
  sensor    lag, white and coloured noise, quantisation, sampled analysers (sample-and-hold, age published),
            spikes, bad-quality intervals (aio.sim.core)

Disturbances enter through the same pair dynamics as the MVs (own gains, lags and dead times), so a measured
disturbance is worth feeding forward only as far as its path is slower than the MVs', and an unmeasured load
is seen only through the CVs. Between events they keep moving (mean-reverting noise, sometimes a sinusoid).

Rows 0..nm-1 are the tracked CVs (one per MV, the published pairing), then the zone CVs (limits only), then
the measured-disturbance rows.

A unit can have more MVs than CVs to hold (a fat plant; FAT_P of the units with two or more MVs): its spare
MVs have no process CV of their own. Each has a resting value instead, published the way an APC engineer
configures one: as a tracked row that reads the MV's own flow, with the resting value as its target and a
wide band (REST_BAND of the flow's range against 1-3 % of the range for a process CV). `rst` marks the
row, `spr` the MV. In the plant, the packet and the score it is a tracked row like any other. What is its
own is its rank, as in the steady-state layer of an industrial MPC: a resting value comes after every
process CV and every zone limit. Where they can all be held the spare MV rests at its value; where a valve
runs out or a zone limit is in the way, the best steady point (`optimum`) has the spare MV leave its
resting value as far as that takes, before any process CV gives (REST_EPS), and the cost charges the row
against where that leaves it (the reachable resting value, p['rtgt']), so the dynamic urgency of coming
back is the band's and does not pull on the CVs. In a transient the spare MV may help at that price.
`push` is the event that makes the unit ride a constraint: the operator moves a resting value to the
valve's limit (most feed, least steam), and the spare MV goes there as far as the other valves and the
zone limits let it.
"""
import contextlib
import math
import typing
import torch

from aio.interface import NCV, NMV

ND, NQ = 2, 2                    # measured disturbances, unmeasured load inputs
NDQ = ND+NQ
NU = NMV+NDQ                     # input channels of the pair dynamics
# These are the default size. A batch has one size (rows, MVs, measured disturbances and as many unmeasured loads), chosen when it is
# drawn; every function reads the size off the tensors it is given (`sizes`), so the names above inside a function are that batch's.
DIMS = (NCV, NMV, ND)
SIZES = {'12x5': DIMS, '24x10': (24, 10, 4), '48x20': (48, 20, 8)}
DMAX = 60                        # longest dead time (decisions)
HORIZON = 720
NSEG = 12                        # the start and up to 11 events
CARRIED = 3                      # segments the next window on the same plants takes over: the plant as commissioned and the last two in use
NSEG_STAY = 16                   # segments with which a window that carries on has room for its events to the end (13 new ones; with NSEG, 9: more than half of the units
                                 # have used them up some 220 decisions before it ends, and nothing happens there). Not the default: the batches of a stay are as they were
BIG = 10**9
DTS = (1., 2., 5., 10., 30., 60., 120.)
ULO, UHI = .05, .95              # MV limits given to the controller at commissioning (of the valve's span)
U_MARGIN = .03                   # steady valve positions stay this far inside the MV limits (8-92 %), unless the point is constrained
ON = 1e-9                        # a point built to sit on a limit of the check (a step cut to it by `reach`) is inside it, whichever way it rounds
SNAP = 1e-6                      # a count of decisions computed from a solved point: a value this near a whole number is that number (`up`, `down`)
SIGMA_MIN, GAIN_MAX = .5, 800.   # scaled steady gains (bands per full MV span): not singular (relative gains to ~20), no sub-0.1 % actuator
WANDER = .03                     # valve travel (1 sigma, of span) that holding the targets against the wandering loads may take
ZONE_MARGIN = 1.                 # steady zone CVs stay this many bands inside their limits
STICK_MAX = 1.                   # a sticking valve moves no tracked CV by more than this many bands per stick
F = torch.float64

# ---- structure priors
NM_P = (.08, .22, .30, .22, .18)            # P(nm = 1..5): mostly APC scale
NZ_P = (.35, .25, .2, .12, .08)             # zone CVs
ND_P = (.4, .35, .25)                       # measured disturbances
NQ_P = (.3, .4, .3)                         # unmeasured load inputs
T_RANGE = (2.5, 90.)                        # the unit's time scale (decisions): from a unit that answers within a few decisions to one whose answer just fits model.LAGS
TAU_MAX = 150.
RTH = ((.01, .3), (.3, 1.2), (1.2, 4.))     # dead time over time constant: lag dominant, balanced, dead-time dominant
STORE_P = .25                               # share of the units with a storage behind them (thermal mass, a large holdup, a recycle): their rows answer with a fast part and a slow one
STORE_T = (40., 200.)                       # the storage's time constant (decisions); the decision interval of such a unit is chosen for its fast part
STORE_FAST = (1., 20.)                      # (the time scale of such a unit's fast part)
STORE_ROW_P, TAIL_ROW_P = .8, .05           # share of such a unit's rows that feel it; in any other unit, of the rows with a slow part of their own
TAIL_T = (6., 60.)                          # (its time constant over the row's)
TAIL_SHARE = (.3, .9)                       # how much of a row's answer is the slow part
TAIL_MIN = 4.                               # a pair has the two parts where the slow one is this many times slower than its own time constant
DEC, SER, INT, SEP, BLE, SPA = range(6)
ARCH = ('decoupled', 'series', 'interacting', 'separation', 'blending', 'sparse', 'mechanism', 'plant')   # 'mechanism': a plant of the engine (aio.sim.mech); 'plant': several units in a chain (the larger sizes)
PLANT = 7
ARCH_P = (.10, .15, .25, .25, .10, .15)
KIND_P = (.35, .25, .1, .1, .2)             # first order, overdamped, inverse response, lead, underdamped
RGA_MIN = .03                               # published pairings have positive relative gains (97 % of the plants)
ALT_PAIR_P = .1                             # the published pairing is another positive one, not the best (chosen for speed)
GEO_P, GEO = .7, (2., 30.)                  # distributed units: dead time grows up to 3-31x with the distance input -> row
GEO_NEAR = .15                              # distance at which a pair has the unit's own dead-time ratio
FAT_P = .3                                  # share of the units with two or more MVs that have spare ones
NR_P = (.7, .2, .1)                         # how many of its MVs are spare (never all: one process CV stays)
REST_BAND = (.04, .12)                      # band of a resting value (of the flow's range; a process CV's is 1-3 % of its range)
REST_EPS = 1e-4                             # weight of a resting value in the best steady point, against 1 in the cost: it ranks below every CV
EQ_SOFT = 1e-6                              # a level in balance ranks above every CV there: a drift of d bands per 100 decisions counts as d^2/EQ_SOFT (0.001 band as a band off a priority-1 target)
REST_TAU = (.5, 3.)                         # lag of an MV's flow reading (decisions)
# a plant of a larger size (SIZES) is several units in a chain, each a unit as above with its own kind and time scale
DOWN_P, DOWN_FADE = .7, .5                  # a unit feels the unit d places upstream with this probability, halved (as the gain is) with every place more
DOWN_GAIN = (.15, .7)                       # how much of an upstream MV's move reaches the next unit's feed (its rate and its composition: two directions)
BACK_P, BACK_GAIN = .15, (.05, .25)         # a downstream unit acts on an upstream one (recycle, heat integration): seldom and weakly
UNIT_T = .3                                 # the units' time scales differ by up to 10**UNIT_T either way
LOCAL_P = .7                                # share of the setpoint, process and limit events that strike one unit only (the others hold their targets and limits)
# the larger sizes are shaped as the controllers of industrial APC are (docs/sim.md, 'sizes'): about twice as many CVs as MVs, most of them
# held inside limits; few setpoints, most MVs resting on a target of their own that economics moves to a limit; the unit runs where limits stop them
ZONE_PER = (1., 2.5)                        # zone CVs per MV
ECON_P, ECON_SHARE = .85, (.35, .8)         # share of the plants with spare MVs, and how many of their MVs are spare
APC_EXC, APC_RIDE = .5, .9                  # where no menu says otherwise: share of the events that take the unit to a constrained point, and how likely it stays there through the next
PUSH_MORE = .5                              # a push takes each of the other spare MVs along with this probability (each to a limit of its own)
THIN_P, THIN_SHARE = .35, (.15, .5)         # share of the plants with MVs that are not the controller's (a hand valve, an MV the operators keep), and how many of the MVs with a process CV:
                                            # they stay where they stand, the packet has no such MV, and the plant has more setpoints than MVs (the ones that rank lowest give)
# a conserving train of vessels (shape={'stock': share of the plants}; docs/sim.md, 'stock'): levels that add up the flows between them
STOCK_MAX = 4                               # vessels at the most (one handle each: a valve that is not a spare one)
STOCK_CAP = .4                              # a stream moves the vessel it leaves and the one it enters by amounts whose ratio (their holdups') is within 10**STOCK_CAP
STOCK_AGAINST = .5                          # share of the trains whose levels are held by their inflows (the throughput is set at the far end) instead of their outflows
STOCK_FEED = (.1, .5)                       # gain of the throughput (a disturbance: the feed, or the demand) on the vessel it enters or leaves
STOCK_MEASURED = .7                         # that disturbance is a measured one where the plant has one
STOCK_REC = .35                             # share of the trains of two or more vessels with a recycle from the last to the first, its flow another valve of the plant
STOCK_SIDE = .4                             # share of the vessels with an unmeasured side stream
STOCK_WIDE = .6                             # share of the vessels that are buffers: the level may be anywhere in a wide band
STOCK_BAND = (.08, .25)                     # (that band, of the level's range)
STOCK_FILL = (15., 100.)                    # decisions a vessel's handle takes, a quarter of its span open or shut, to move the level a quarter of its range (the vessel's size)
STOCK = dict(stock=1.)                      # a shape: every plant has such a train
STANDARD = dict(stock=.25)                  # the generator of the standard (docs/standard.md): a quarter of the plants have one; what training and the `standard` exam draw

ROOTS = ('nm', 'nd', 'nq', 'nz', 'nr', 'T', 'cls', 'rth', 'arch', 'linear', 'rich', 'geo', 'dt', 'store')
# ---- events
SP, PROC, BOTH, LOAD, DV, LIMIT, PRIO, OVER_SP, OVER_LOAD, MV_LIMIT, RETURN, ZONE_LIMIT, MV_OUT, PUSH = range(14)
EVENTS = ('setpoint', 'process', 'both', 'load', 'dv', 'limit', 'priority', 'over_setpoint', 'over_load', 'mv_limit', 'return',
          'zone_limit', 'mv_out', 'push')
EXCURSIONS = (OVER_SP, OVER_LOAD, MV_LIMIT, ZONE_LIMIT, MV_OUT, PUSH)  # the kinds that take the unit to a constrained point
EVENT_P = (.30, .20, .12, .15, .10, .08, .05)     # the ordinary kinds
EXC_P = .15                                 # share of the events that take the unit to a constrained point (the next one brings it back)
EXC_TRIES = 3                               # candidates drawn for one
EXC_KIND_P = (.3, .2, .2, .15, .15)         # by a target, by a disturbance, by an MV limit, by a zone limit, by an MV out of service
PUSH_P = .5                                 # on a fat plant, share of them that are a resting value pushed to the valve's limit instead
ZONE_PAST = (2., 15.)                       # bands a zone limit is moved past where its CV sits
ZONE_GIVE = 1.5                             # at the best steady point a zone CV is at most this many bands beyond its limit
PIN = .02                                   # an MV out of service keeps this much room either side of where it stands (of span)
GIVE = (1.5, 30.)                           # bands the worst tracked CV is off its target at a constrained point (a resting value: bands of its instrument)
LEVEL_DRIFT = .05                           # a level at a constrained point drifts less than this (bands per 100 decisions) per unit of any disturbance
ALPHA = (1., .6, .35, .2, .1, .05)          # a rejected process change is retried smaller
TRAVEL = 1.5                                # the next event waits this many times the valve travel at full slew (plus half the shortest gap)
SETTLE, STAY = 1.5, .5                      # a constrained point is held the valve travel plus this many settling times of the unit: to line out, and to stay
FIRST_EVENT = (20, 80)
RENEW = (10, 40)                            # decisions after an event before the teacher's model follows it
RAMP_P = .65                                # share of process and disturbance changes that come gradually
RESTORE_P = .35                             # on a long stay (a window that carries on), share of the process events that are maintenance: some of what has changed goes back to as commissioned
MOVE_MIN, LOAD_MIN = 1.5, .5                # smallest setpoint move and disturbance effect worth an event (bands)
MIN_GAP = 5                                 # decisions between two events at the least, however fast a menu has them come
PROCESS = ('gain', 'actuator', 'time', 'dead', 'curve', 'output', 'friction', 'lag')    # the kinds of process change (change_process)


class Menu(typing.NamedTuple):
    """What changes on a plant during a window: how often, how large, in what order. Without one (None) the
    generator is as it was, bit for bit. A menu belongs to a window: the next window on the same plants can
    have another (a quiet one, a denser one, one without process changes)."""
    rate: tuple = (1., 1., 1., 1., 1.)     # how often, each against its usual share: setpoint moves, measured disturbances, unmeasured loads, process changes, the rest (limits, priorities). The events come as much faster as the shares add up to; all 0: none
    pace: float = 1.                       # events per unit time, all kinds (the gaps between events are divided by it)
    move: float = 1.                       # size of a setpoint move
    step: float = 1.                       # size of a disturbance step, measured or not
    proc: float = 1.                       # size of a process change (its factors to this power; a mechanistic plant's parameters over ranges this many times as wide)
    wander: float = 1.                     # what keeps moving between events (the loads' wander and the sinusoid); 0: nothing
    turn: float = 0.                       # share of the setpoint moves that take one tracked CV alone, each in its turn (True: all of them)
    joint: float = 0.                      # share of the disturbance events that move every load at once, measured and unmeasured
    pulse: float = 0.                      # share of the disturbance and process events that the next event takes back
    drop: tuple = ()                       # left out: event kinds (EVENTS) and kinds of process change (PROCESS), by name
    exc: typing.Optional[float] = None     # share of the events that take the unit to a constrained point (None: as the generator is asked)
    ride: typing.Optional[float] = None    # a unit at a constrained point stays there through the next event with this probability: what took it there holds (the narrowed limit, the
                                           # load, the targets asked for) while ordinary events go on, drawn on the unit as it would be without it; 0: the next event brings it back
                                           # (None: 0 on the default size, APC_RIDE on the larger ones)
    heavy: float = 0.                      # share of the process events that are heavy ones instead of the usual kinds: what a valve move does becomes something else (the gains turn about the
                                           # operating point: cross effects as large as the row's gains, either sign; the paired gain keeps its sign) and the time constants change by up to 3 either way
    shift: float = 0.                      # probability that the unit has one heavy change in this window that stays: at its first ordinary event it becomes another unit (as `heavy` makes one)
                                           # and that is what it is from then on (maintenance puts it back to that, not to as commissioned). Few of them: 0.15 is one or two in eight windows

SEG_FIELDS = ('K', 'p1r', 'p1i', 'p2r', 'p2i', 'c0r', 'c0i', 'c1r', 'c1i', 'dead', 'gam', 'fs', 'fd', 'tlag', 'wa', 'dlev')
TASK_FIELDS = ('tgt', 'plo', 'phi', 'scale', 'cvw', 'lo', 'hi', 'ulo', 'uhi', 'rtgt', 'alo', 'ahi', 'blo', 'bhi')


def sizes_of(ncv, nmv, nd):
    """(NCV, NMV, ND, NQ, NDQ, NU) of a size: rows, MVs, measured disturbances, unmeasured loads (as many), both, input channels."""
    return ncv, nmv, nd, nd, 2*nd, nmv+2*nd


def sizes(p):
    """The same of a batch of plants (or of the parameters of a segment), read off its tensors."""
    ncv, nu = p['K'].shape[-2:]
    ndq = p['dlev'].shape[-1]
    return ncv, nu-ndq, ndq//2, ndq-ndq//2, ndq, nu


def nzmax(ncv, nmv, nd):
    """Zone CVs a plant of this size can have. At the default size: the rows left by its MVs' own rows and the measured
    disturbances, less one. At a larger size: the rows the MVs of the size leave (a plant with fewer MVs and
    disturbances than the size has room for has the more of them: ZONE_PER)."""
    return ncv-nmv-nd-1 if (ncv, nmv, nd) == DIMS else ncv-nmv


# ---------------------------------------------------------------- random numbers: one stream per episode
_M64 = 1 << 64


def _s64(c):                                                     # a 64-bit constant as a signed integer
    return c-_M64 if c >= (1 << 63) else c


_C1, _C2, _C3 = _s64(0x9E3779B97F4A7C15), _s64(0xBF58476D1CE4E5B9), _s64(0x94D049BB133111EB)


def _shr(x, k):                                                  # logical shift of a signed 64-bit tensor
    return (x >> k) & ((1 << (64-k))-1)


def _mix(x):                                                     # splitmix64's finaliser: wrapping integer arithmetic, the same on every device
    x = x+_C1
    x = (x ^ _shr(x, 30))*_C2
    x = (x ^ _shr(x, 27))*_C3
    return x ^ _shr(x, 31)


class Rng:
    """Random numbers, one stream per episode: number c of episode i is a hash of (seed, i, c), the same on
    every device. A draw of shape (m, ...) is for the m episodes of this view and gives each the next numbers
    of its own stream, so what an episode draws does not depend on how many others there are, nor on which of
    them are drawn again. `sub(idx)` is the view on some of the episodes (the counters are shared), with
    `repeat` several candidates of each; inside `only(mask)` the others' streams stand still (their numbers
    are not to be used); `fork()` is a new set of streams (another purpose, another number of episodes)."""

    def __init__(self, seed, device='cpu'):
        self.dev = torch.device(device)
        self.seed = int(seed) % _M64
        self.rows = self.lane = self.ctr = self.live = None
        self.forks = 0

    # ---- views
    def bind(self, n):
        if self.rows is None:
            self.rows = torch.arange(n, device=self.dev)
            self.lane = torch.zeros(n, dtype=torch.long, device=self.dev)
            self.ctr = torch.zeros(n, dtype=torch.long, device=self.dev)
        assert self.rows.numel() == n, f'a draw for {n} episodes from the streams of {self.rows.numel()} (take .sub(idx) for some of them, .fork() for another set)'
        return self

    def sub(self, idx, repeat=1):
        v = Rng.__new__(Rng)
        v.dev, v.seed, v.ctr, v.forks, v.live = self.dev, self.seed, self.ctr, 0, None
        v.rows, lane = self.rows[idx].repeat_interleave(repeat), self.lane[idx].repeat_interleave(repeat)
        v.lane = lane*repeat+torch.arange(repeat, device=self.dev).repeat(idx.numel()) if repeat > 1 else lane
        return v

    def fork(self, device=None):
        self.forks += 1
        return Rng(int(_mix(torch.tensor(_s64(self.seed))+_mix(torch.tensor(-self.forks)))) % _M64, device or self.dev)

    @contextlib.contextmanager
    def only(self, mask):
        old, self.live = self.live, mask if self.live is None else self.live & mask
        try:
            yield self
        finally:
            self.live = old

    # ---- the numbers
    def _next(self, shape):
        """Uniforms in [0, 1) of the shape asked: for each episode of the view the next numbers of its stream."""
        m = shape[0]
        self.bind(m)
        k = math.prod(shape[1:])
        key = _mix(_mix(_mix(torch.full_like(self.rows, _s64(self.seed)))+self.rows)+self.lane)
        j = self.ctr[self.rows][:, None]+torch.arange(k, device=self.dev)[None]
        z = _mix(key[:, None]+_mix(j))
        step = torch.full_like(self.rows, k) if self.live is None else k*self.live.long()
        self.ctr[self.rows] = self.ctr[self.rows]+step             # candidates of one episode share its counter: all moved by the same count
        return (_shr(z, 11).to(F)*(1./(1 << 53))).view(*shape)

    def u(self, *shape, lo=0., hi=1.):
        return lo+(hi-lo)*self._next(shape)

    def logu(self, *shape, lo, hi):
        return torch.exp(self.u(*shape, lo=math.log(lo), hi=math.log(hi)))

    def bern(self, p, *shape):
        return self.u(*shape) < p

    def sign(self, *shape):
        return torch.where(self.u(*shape) < .5, -1., 1.).to(F)

    def cat(self, m, probs, *more):
        c = torch.tensor(probs, dtype=F, device=self.dev).cumsum(0)
        return torch.searchsorted(c/c[-1], self.u(m, *more).contiguous(), right=True).clamp_max(len(probs)-1)

    def cat_rows(self, probs):
        """One category per row of a (m, k) probability table."""
        c = probs.cumsum(-1)
        return torch.searchsorted((c/c[:, -1:]).contiguous(), self.u(probs.shape[0], 1), right=True)[:, 0].clamp_max(probs.shape[1]-1)

    def randint(self, lo, hi, *shape):
        return (lo+(self.u(*shape)*(hi-lo+1)).floor().long()).clamp_max(hi)

    def normal(self, *shape):
        a = self._next((*shape, 2))                               # Box-Muller
        return torch.sqrt(-2*torch.log1p(-a[..., 0]))*torch.cos(2*math.pi*a[..., 1])

    def one_of(self, mask):
        """One-hot (bool) of one random true entry per row of a (m, k) mask (all false where a row has none)."""
        j = (self.u(*mask.shape)+mask.to(F)).argmax(-1)
        return torch.nn.functional.one_hot(j, mask.shape[-1]).bool() & mask


# ---------------------------------------------------------------- output curve (monotone quadratic Wiener map)
def h(v, a):
    w = v-.5
    return .5+w+a*w*w.abs()


def hprime(v, a):
    return 1+2*a*(v-.5).abs()


def hinv(y, a):
    z = y-.5
    disc = 1+4*a*z.abs()
    w = torch.sign(z)*2*z.abs()/(1+torch.sqrt(disc.clamp_min(0)))
    return .5+w, disc > 0


def _cdiv(ar, ai, br, bi):
    d = br*br+bi*bi
    return (ar*br+ai*bi)/d, (ai*br-ar*bi)/d


def derive(p):
    """Mode weights of every pair from its poles (1/decision) and zero beta: the pair output is
    Re(c0 s0 + c1 s1) of two unit-gain modes s <- lam s + (1 - lam) u, lam = exp(pole x time scale);
    c0 = (1 - beta p1) p2/(p2 - p1), c1 = (1 - beta p2) p1/(p1 - p2), c0 + c1 = 1."""
    p1r, p1i, p2r, p2i, beta = p['p1r'], p['p1i'], p['p2r'], p['p2i'], p['beta']
    nr, ni = (1-beta*p1r)*p2r+beta*p1i*p2i, (1-beta*p1r)*p2i-beta*p1i*p2r
    c1r, c1i = _cdiv(nr, ni, p2r-p1r, p2i-p1i)
    nr, ni = (1-beta*p2r)*p1r+beta*p2i*p1i, (1-beta*p2r)*p1i-beta*p2i*p1r
    c2r, c2i = _cdiv(nr, ni, p1r-p2r, p1i-p2i)
    first = p['first'] > 0
    p['c0r'] = torch.where(first, torch.ones_like(c1r), c1r); p['c0i'] = torch.where(first, torch.zeros_like(c1i), c1i)
    p['c1r'] = torch.where(first, torch.zeros_like(c2r), c2r); p['c1i'] = torch.where(first, torch.zeros_like(c2i), c2i)
    return p


# ---------------------------------------------------------------- one plant
SMALL = dict(nm=(2, 3), nr=(0, 0))            # a shape: two or three valves, none to spare


def draw_roots(r, m, dims=DIMS, shape=None):
    """What everything else about a unit depends on. shape: {root: (lo, hi)}, the roots kept to a range (counts
    outside it are folded in, the others cut to it); the draws are the usual ones."""
    NCV, NMV, ND, NQ, NDQ, NU = sizes_of(*dims)
    if tuple(dims) == DIMS:
        nm = r.cat(m, NM_P)+1
        nd = r.cat(m, ND_P)
        nq = r.cat(m, NQ_P)
        nz = torch.minimum(r.cat(m, NZ_P), NCV-nm-nd)
        nr = torch.where(r.bern(FAT_P, m) & (nm > 1), torch.minimum(r.cat(m, NR_P)+1, nm-1), torch.zeros_like(nm))     # spare MVs
    else:                                                        # a larger size: the plants that do not fit the next smaller one, with more limits than a single unit has
        nm = r.randint(NMV//2+1, NMV, m)
        nd = r.randint(0, ND, m)
        nq = r.randint(0, NQ, m)
        nz = torch.minimum((r.u(m, lo=ZONE_PER[0], hi=ZONE_PER[1])*nm).round().long(), torch.clamp(NCV-nm-nd-1, max=nzmax(*dims)))
        nr = torch.where(r.bern(ECON_P, m), torch.minimum((r.u(m, lo=ECON_SHARE[0], hi=ECON_SHARE[1])*nm).ceil().long(), nm-1), torch.zeros_like(nm))
        hs = torch.where(r.bern(THIN_P, m), r.u(m, lo=THIN_SHARE[0], hi=THIN_SHARE[1]), torch.zeros(m, dtype=F, device=r.dev))
    T = r.logu(m, lo=T_RANGE[0], hi=T_RANGE[1])                  # the unit's time scale (decisions)
    cls = r.cat(m, (.4, .4, .2))                                 # lag dominant, balanced, dead-time dominant
    rth = torch.where(cls == 0, r.logu(m, lo=RTH[0][0], hi=RTH[0][1]), torch.where(cls == 1, r.logu(m, lo=RTH[1][0], hi=RTH[1][1]), r.logu(m, lo=RTH[2][0], hi=RTH[2][1])))
    arch = r.cat(m, ARCH_P)
    linear = r.bern(.3, m)                                       # no valve, output or friction nonlinearity (identified-model-like)
    rich = r.bern(.6, m)                                         # pair shapes beyond first order
    geo = torch.where(r.bern(GEO_P, m), r.logu(m, lo=GEO[0], hi=GEO[1]), torch.zeros(m, dtype=F, device=r.dev))
    dt = torch.tensor(DTS, dtype=F, device=r.dev)[r.cat(m, (1.,)*len(DTS))]
    store = torch.where(r.bern(STORE_P, m), r.logu(m, lo=STORE_T[0], hi=STORE_T[1]), torch.zeros(m, dtype=F, device=r.dev))    # the storage's time constant (0: none)
    T = torch.where(store > 0, r.logu(m, lo=STORE_FAST[0], hi=STORE_FAST[1]), T)
    ro = dict(nm=nm, nd=nd, nq=nq, nz=nz, nr=nr, T=T, cls=cls, rth=rth, arch=arch, linear=linear, rich=rich, geo=geo, dt=dt, store=store)
    if tuple(dims) != DIMS:
        ro['hs'] = hs
    if shape and 'stock' in shape:                              # (drawn only where asked: the other batches are as they were, bit for bit)
        ro['stock'] = r.bern(float(shape['stock']), m)
        shape = {q: v for q, v in shape.items() if q != 'stock'}
    if shape:
        for q, (lo, hi) in shape.items():
            ro[q] = lo+(ro[q]-lo) % (hi-lo+1) if ro[q].dtype == torch.long else ro[q].clamp(lo, hi)
        ro['nr'] = torch.minimum(ro['nr'], ro['nm']-1)
        ro['nz'] = torch.minimum(ro['nz'], NCV-ro['nm']-ro['nd'])
    return ro


def assign(cost):
    """(m, n) the row that takes every column in the assignment of least total cost (m, n, n: rows, columns):
    the Hungarian method by shortest augmenting paths, the same arithmetic for every episode whatever the
    batch. Among columns equally near, the first. Integer costs are handled exactly (no rounding anywhere:
    the same answer on every machine, ties included)."""
    m, n, dev, dt = cost.shape[0], cost.shape[1], cost.device, cost.dtype
    inf = math.inf if dt.is_floating_point else 2**62
    a = torch.zeros(m, n+1, n+1, dtype=dt, device=dev)
    a[:, 1:, 1:] = cost
    u, v = torch.zeros(m, n+1, dtype=dt, device=dev), torch.zeros(m, n+1, dtype=dt, device=dev)      # potentials of the rows and the columns
    row = torch.zeros(m, n+1, dtype=torch.long, device=dev)       # the row of every column (0: none yet); column 0 holds the row being placed
    way = torch.zeros(m, n+1, dtype=torch.long, device=dev)
    ar, cols = torch.arange(m, device=dev), torch.arange(n+1, device=dev)[None]
    real = cols > 0
    for i in range(1, n+1):
        row[:, 0] = i
        j0 = torch.zeros(m, dtype=torch.long, device=dev)
        near = torch.full((m, n+1), inf, dtype=dt, device=dev)
        used = torch.zeros(m, n+1, dtype=torch.bool, device=dev)
        live = torch.ones(m, dtype=torch.bool, device=dev)
        for _ in range(n+1):                                      # grow the tree until it reaches a column without a row
            used[ar, j0] = used[ar, j0] | live
            i0 = row[ar, j0]
            cur = a[ar, i0]-u[ar, i0][:, None]-v
            free = ~used & real
            better = free & (cur < near) & live[:, None]
            near = torch.where(better, cur, near)
            way = torch.where(better, j0[:, None], way)
            reach = torch.where(free, near, torch.full_like(near, inf))
            delta = reach.amin(1)
            j1 = torch.where(reach == delta[:, None], cols, n+1).amin(1)             # the first of the nearest, whatever `min` does with ties
            delta = torch.where(live, delta, torch.zeros_like(delta))[:, None]
            u = u.scatter_add(1, row, torch.where(used, delta, torch.zeros_like(near)))
            v = v-torch.where(used, delta, torch.zeros_like(near))
            near = torch.where(used, near, near-delta)
            j0 = torch.where(live, j1, j0)
            live = live & (row[ar, j0] != 0)
            if not bool(live.any()):
                break
        go = torch.ones(m, dtype=torch.bool, device=dev)
        for _ in range(n+1):                                      # move the rows along the path found
            j1 = way[ar, j0]
            row[ar, j0] = torch.where(go, row[ar, j1], row[ar, j0])
            j0 = torch.where(go, j1, j0)
            go = go & (j0 != 0)
            if not bool(go.any()):
                break
    return row[:, 1:]-1


FAR = 1e6                                   # the cost of a pair that is not allowed
PAIR_Q, PAIR_TIE = 2.**20, 4096             # pairing costs are compared in units of 2^-20, each unit PAIR_TIE of the tie-break (the distance from the order given: below PAIR_TIE in total up to 64 MVs)


def best_pairing(B, mvm, alt=None, u=None):
    """(m, n) row order of the tracked block B (m, n, n) that an engineer would publish: the pairing whose
    relative gains are positive and closest to 1 (row out[:, j] goes with MV j): an assignment problem, the
    sum over the pairs of |log relative gain| (100 more for a negative one). Where `alt`, a pairing with
    every relative gain above 0.05 instead, picked by the uniform draws u (m, n, n) (the assignment of least
    total draw): the pairing chosen for speed rather than for steady-state interaction. Equally good
    pairings: the one nearest the order given, never the one rounding prefers: the costs are rounded to
    integers (units of 2^-20) and assigned in exact arithmetic. A separation unit with three or more MVs
    whose diagonal relative gains are negative has all its off-diagonal relative gains equal in exact
    arithmetic, so several pairings cost exactly the same and the inverse's last bits used to choose,
    differently on different machines."""
    m, n, dev = B.shape[0], B.shape[1], B.device
    act = mvm > 0
    eye = torch.eye(n, dtype=F, device=dev)
    both = act[:, :, None] & act[:, None, :]
    Bm = torch.where(both, B, eye)
    Bi, bad = torch.linalg.inv_ex(Bm)
    lam = Bm*Bi.transpose(1, 2)                                   # relative gain of row i with MV j
    an = torch.arange(n, device=dev)
    stay = torch.where(eye > 0, torch.zeros_like(lam), torch.full_like(lam, FAR))       # rows without an MV stay where they are
    cost = torch.where(both, lam.clamp_min(1e-3).log().abs()+100.*(lam <= 0), stay).nan_to_num(FAR, FAR, FAR)
    best = assign((cost*PAIR_Q).round().long()*PAIR_TIE+(an[:, None]-an[None, :]).abs())
    if alt is not None:
        cost = torch.where(both, torch.where(lam > .05, u, torch.full_like(lam, FAR)), stay)
        other = assign(cost)
        fine = torch.gather(cost, 1, other[:, None, :]).sum((1, 2)) < FAR/2
        best = torch.where((alt & fine)[:, None], other, best)
    return torch.where((bad > 0)[:, None], an[None].expand(m, -1), best)


def _stock(r, m, on, dims, K, p1r, p1i, p2r, p2i, beta, first, dead, integ, tint, lam, nlag, wa, bandf, mvm, spr, nd, nq, inv, coldead):
    """A conserving train of vessels in the plants `on` (m,): 1-STOCK_MAX of the tracked rows become the levels of
    vessels in a line, each with one valve as its handle, a stream of the line. A stream leaves one vessel and
    enters the next: what it takes from the one it adds to the other, in the ratio of their holdups, with one lag
    and one transport delay (it is one flow). The levels are held by their outflows, and the feed of the first
    vessel is a disturbance (measured where the plant has one), or by their inflows, and the disturbance is the
    demand on the last. Some trains have a recycle from the last vessel to the first, its flow another valve of
    the plant (one with a CV of its own, or a spare one); some vessels an unmeasured side stream. Nothing else
    moves a level: a level's rate is the sum of the flows at its vessel, which the valves and the measured feed
    show. Most vessels are buffers: the level is asked to stay within a wide band, not on a point. The plant's
    other rows (and every row's answer to these valves) are as drawn. Returns the fields it changed and the
    (m, NCV) mask of the levels."""
    NCV, NMV, ND, NQ, NDQ, NU = sizes_of(*dims)
    dev, U, LU = r.dev, r.u, r.logu
    rows = torch.arange(m, device=dev)
    cand = (mvm > 0) & ~spr
    L = torch.minimum(r.randint(1, STOCK_MAX, m), cand.sum(1))
    rank = torch.argsort(torch.argsort(U(m, NMV)+10*(~cand), -1), -1)           # place in the line of every valve that can be a handle
    hand = cand & (rank < L[:, None]) & on[:, None]
    against = r.bern(STOCK_AGAINST, m)
    cap = 10**U(m, NMV, lo=-STOCK_CAP, hi=STOCK_CAP)
    taus = LU(m, NU, lo=REST_TAU[0], hi=REST_TAU[1])
    measured = r.bern(STOCK_MEASURED, m) & (nd > 0)
    f = torch.where(measured, torch.full_like(nd, NMV), torch.where(nq > 0, torch.full_like(nd, NMV+ND), torch.full_like(nd, -1)))
    gf = U(m, lo=STOCK_FEED[0], hi=STOCK_FEED[1])
    rec = r.bern(STOCK_REC, m) & (L > 1)
    x = r.one_of((mvm > 0) & ~hand)                                             # the recycle's valve
    xs = U(m, lo=.3, hi=1.)
    side = r.bern(STOCK_SIDE, m, NCV)
    wide, wband = r.bern(STOCK_WIDE, m, NCV), U(m, NCV, lo=STOCK_BAND[0], hi=STOCK_BAND[1])
    fill = LU(m, NCV, lo=STOCK_FILL[0], hi=STOCK_FILL[1])
    own = torch.ones(m, NCV, dtype=F, device=dev)
    js = [((rank == k) & hand).to(F).argmax(1) for k in range(STOCK_MAX)]       # the handle of vessel k, its row, and whether the plant has that vessel
    rs = [inv[rows, j] for j in js]
    has = [on & (L > k) for k in range(STOCK_MAX)]
    Kn = torch.zeros_like(K)
    lev = torch.zeros(m, NCV, dtype=torch.bool, device=dev)
    sg = torch.where(against, 1., -1.).to(F)

    def add(row, col, v, ok):                                    # (rows without it add 0 to an entry of their own)
        Kn[rows, row, col] = Kn[rows, row, col]+torch.where(ok, v, torch.zeros_like(v))

    for k in range(STOCK_MAX):
        a = K[rows, rs[k], js[k]].abs().clamp_min(.1)
        lev[rows, rs[k]] = lev[rows, rs[k]] | has[k]
        own[rows, rs[k]] = torch.where(has[k], a, own[rows, rs[k]])
        add(rs[k], js[k], sg*a, has[k])                                         # its own vessel: an outflow lowers it, an inflow raises it
        for kn, ok in ((k+1, ~against), (k-1, against)):                        # and the vessel at the stream's other end
            if 0 <= kn < STOCK_MAX:
                add(rs[kn], js[k], -sg*a*cap[rows, js[k]], has[k] & has[kn] & ok)
    last = torch.stack(rs, 1)[rows, (L-1).clamp_min(0)]                         # the row of the last vessel
    ends = torch.where(against, last, rs[0])                                    # where the throughput enters (the feed) or leaves (the demand)
    add(ends, f.clamp_min(0), -sg*gf, on & (f >= 0))
    xj = x.to(F).argmax(1)
    okx = on & rec & x.any(1)
    b = K[rows, last, xj].abs().clamp_min(.1)*xs
    add(last, xj, -b, okx)
    add(rs[0], xj, b*cap[rows, xj], okx)
    q = (torch.arange(NU, device=dev)[None] >= NMV+ND) & (torch.arange(NU, device=dev)[None] != f[:, None])    # unmeasured side streams: the loads as drawn, on some vessels
    Kn = torch.where(q[:, None, :] & side[:, :, None], K, Kn)
    l3, o1 = lev[:, :, None], on[:, None]
    K = torch.where(l3, Kn, K)
    p1r, p2r = torch.where(l3, (-1/taus)[:, None, :], p1r), torch.where(l3, (-2/taus)[:, None, :], p2r)
    p1i, p2i, beta = (torch.where(l3, torch.zeros_like(t), t) for t in (p1i, p2i, beta))
    first = torch.where(l3, torch.ones_like(first), first)
    dead = torch.where(l3, coldead[:, None, :]*(K != 0), dead)
    integ = torch.where(o1, lev.to(F), integ)                                   # (such a plant has no level of the other kind)
    tint = torch.where(lev, own*fill, tint)                                     # the vessel's size goes with its handle: it can be filled and emptied in the time the events leave
    lam, nlag, wa = (torch.where(lev, torch.zeros_like(t), t) for t in (lam, nlag, wa))
    bandf = torch.where(lev & wide, wband, bandf)
    return K, p1r, p1i, p2r, p2i, beta, first, dead, integ, tint, lam, nlag, wa, bandf, lev.to(F)


def draw_plant(r, m, roots=None, dims=DIMS, shape=None):
    """m plants at their commissioning operating point: dict of (m, ...) float64 tensors."""
    NCV, NMV, ND, NQ, NDQ, NU = sizes_of(*dims)
    dev, U, LU = r.dev, r.u, r.logu
    ar, am, au = torch.arange(NCV, device=dev), torch.arange(NMV, device=dev), torch.arange(NU, device=dev)
    ro = draw_roots(r, m, dims, shape) if roots is None else roots
    nm, nd, nq, nz, nr, T, rth, arch, linear, rich, geo, store = (ro[q] for q in ('nm', 'nd', 'nq', 'nz', 'nr', 'T', 'rth', 'arch', 'linear', 'rich', 'geo', 'store'))
    trk = (ar[None] < nm[:, None]).to(F)
    zon = ((ar[None] >= nm[:, None]) & (ar[None] < (nm+nz)[:, None])).to(F)
    dvr = ((ar[None] >= (nm+nz)[:, None]) & (ar[None] < (nm+nz+nd)[:, None])).to(F)
    prc = trk+zon
    mvm = (am[None] < nm[:, None]).to(F)
    # spare MVs (a fat plant): nr of the MVs, at random; the row of the pattern that goes with a spare MV reads its flow
    spr = (torch.argsort(torch.argsort(U(m, NMV)+10*(1-mvm), -1), -1) < nr[:, None]) & (mvm > 0)
    rs0 = torch.cat((spr, torch.zeros(m, NCV-NMV, dtype=torch.bool, device=dev)), 1)                # resting rows, before the shuffle
    dact = torch.cat(((torch.arange(ND, device=dev)[None] < nd[:, None]), (torch.arange(NQ, device=dev)[None] < nq[:, None])), 1).to(F)
    uact = torch.cat((mvm, dact), 1)
    dvsel = (ar[None]-(nm+nz)[:, None]).clamp(0, ND-1)
    # ---- steady gains of the MVs (change of the row value per full MV span), tracked block by archetype
    mag = 10**U(m, NCV, NMV, lo=-.65, hi=.5)
    sgn = r.sign(m, NCV, NMV)
    eye = (ar[:, None] == am[None]).to(F)[None]
    low = (ar[:, None] > am[None]).to(F)[None]                   # row downstream of the MV (series units)
    a3 = arch[:, None, None]
    V = lambda lo, hi: U(m, NCV, NMV, lo=lo, hi=hi)              # noqa: E731
    multi = tuple(dims) != DIMS
    if multi:
        # several units in a chain. The MVs are dealt to the units in order (1-5 each, as a unit on its own has);
        # a tracked row is in its MV's unit, a zone row and a disturbance in any; the feed (the first measured
        # disturbance and the first unmeasured one) enters the first unit. Inside a unit everything is as in a
        # single one, of the unit's own kind. A unit feels the units upstream through its feed, which has two
        # directions (rate and composition): the rows' gains to them, times how much each upstream MV moves them.
        ab = torch.arange(NMV, device=dev)
        blk = ((r.cat(m, NM_P, NMV)+1).cumsum(1)[:, None, :] <= am[None, :, None]).sum(-1)       # (m, NMV) unit of every MV
        nb = torch.gather(blk, 1, (nm-1)[:, None])+1                                             # (m, 1) units
        anyb = lambda k: torch.minimum((U(m, k)*nb).floor().long(), nb-1)                        # noqa: E731
        rb = torch.where(ar[None] < nm[:, None], torch.cat((blk, blk.new_zeros(m, NCV-NMV)), 1), anyb(NCV))
        feed = (torch.arange(NDQ, device=dev) == 0) | (torch.arange(NDQ, device=dev) == ND)
        cb = torch.cat((blk, torch.where(feed[None], torch.zeros_like(nb), anyb(NDQ))), 1)       # (m, NU) unit of every input
        a3 = torch.gather(r.cat(m, ARCH_P, NMV), 1, blk)[:, None, :]
        gap = ab[:, None]-ab[None]                                                               # places the row's unit is downstream of the input's
        fade = DOWN_FADE**(gap.clamp_min(1)-1).to(F)
        pu = U(m, NMV, NMV)
        feels = torch.where(gap > 0, pu < DOWN_P*fade, (gap < 0) & (pu < BACK_P)).to(F)
        reach_ = feels*U(m, NMV, NMV, lo=DOWN_GAIN[0], hi=DOWN_GAIN[1])*fade
        byunit = lambda t, rbi: torch.gather(t.flatten(1), 1, (rbi[:, :, None]*NMV+cb[:, None, :]).flatten(1)).view(m, NCV, NU)   # noqa: E731
        dist = rb[:, :, None]-cb[:, None, :]
        same = dist == 0
        fdir = [r.sign(m, NCV)*10**U(m, NCV, lo=-.65, hi=.5) for _ in range(2)]
        smv = [U(m, NMV, lo=-1., hi=1.) for _ in range(2)]
        cross = torch.where(dist[:, :, :NMV] > 0, (fdir[0][:, :, None]*smv[0][:, None, :]+fdir[1][:, :, None]*smv[1][:, None, :])*byunit(reach_, rb)[:, :, :NMV],
                            r.sign(m, NCV, NMV)*10**V(-.65, .5)*V(*BACK_GAIN)*r.bern(.5, m, NCV, NMV).to(F)*byunit(feels, rb)[:, :, :NMV])
    off = torch.where(a3 == DEC, V(0., .3),
          torch.where(a3 == SER, low*V(.3, 1.2)+(1-low)*V(0., .15),
          torch.where(a3 == SPA, r.bern(.35, m, NCV, NMV).to(F)*V(.2, 1.),
          torch.where(a3 == BLE, V(.6, 1.), V(.3, 1.)))))
    dg = torch.where(a3 == BLE, V(.15, .5), torch.ones(m, NCV, NMV, dtype=F, device=dev))
    Kt = sgn*mag*(eye*dg+(1-eye)*off)
    # separation: one strong direction plus a small spread (two products of one column: large relative gains)
    g1 = sgn[:, :, 0]*10**U(m, NCV, lo=-.3, hi=.3); g2 = 10**U(m, NMV, lo=-.3, hi=.3)
    eps = (LU(m, lo=.03, hi=.6)*r.sign(m))[:, None, None]
    if multi:
        eps = torch.gather(LU(m, NMV, lo=.03, hi=.6)*r.sign(m, NMV), 1, blk)[:, None, :]
    Kt = torch.where(a3 == SEP, g1[:, :, None]*g2[:, None, :]*(1+eye*eps), Kt)
    if multi:
        Kt = torch.where(same[:, :, :NMV], Kt, cross)
    Kt = torch.where(rs0[:, :, None], eye.expand(m, -1, -1), Kt)   # a resting row: the flow of its own MV and nothing else
    order = best_pairing(Kt[:, :NMV], mvm, r.bern(ALT_PAIR_P, m), U(m, NMV, NMV))   # the engineer's pairing goes on the diagonal
    Kt = torch.cat((torch.gather(Kt[:, :NMV], 1, order[:, :, None].expand(-1, -1, NMV)), Kt[:, NMV:]), 1)
    Kz = sgn*mag*V(.2, 1.)*r.bern(.7, m, NCV, NMV).to(F)         # zone rows: any MV may move them
    if multi:
        Kz = torch.where(same[:, :, :NMV], Kz, cross)                # (of its unit; the others as the tracked rows feel them)
        rb = torch.cat((torch.gather(rb[:, :NMV], 1, order), rb[:, NMV:]), 1)                    # the tracked rows as paired
        dist = rb[:, :, None]-cb[:, None, :]
        same = dist == 0
    Km = torch.where(trk[:, :, None] > 0, Kt, Kz)*prc[:, :, None]*mvm[:, None, :]
    Kd = r.sign(m, NCV, NDQ)*U(m, NCV, NDQ, lo=.05, hi=.4)*prc[:, :, None]*dact[:, None, :]*(~rs0)[:, :, None]
    if multi:                                                    # a disturbance: its own unit, and the units downstream that feel that one
        Kd = Kd*torch.where(same[:, :, NMV:], torch.ones_like(Kd), (dist[:, :, NMV:] > 0)*byunit(reach_, rb)[:, :, NMV:]/DOWN_GAIN[1])
    K = torch.cat((Km, Kd), 2)
    # ---- dynamics: rows share the unit's time scale (a few are fast: a flow or a pressure among compositions),
    # pairs spread around their row; in a distributed unit every input acts at its own place, the paired
    # row mostly near its MV, and dead time (less so the lag) grows with the distance between input and row
    ser = (a3 == SER).to(F)
    xc = U(m, NU)
    xr = U(m, NCV)
    xr = torch.cat((torch.where(r.bern(.7, m, NMV), (xc[:, :NMV]+U(m, NMV, lo=-.1, hi=.1)).clamp(0., 1.), xr[:, :NMV]), xr[:, NMV:]), 1)
    far = (1+geo[:, None, None]*(xr[:, :, None]-xc[:, None, :]).abs())/(1+geo[:, None, None]*GEO_NEAR)
    depth = torch.cat(((ar[:, None]-am[None]).clamp_min(0).to(F)[None].expand(m, -1, -1)*ser,
                       torch.zeros(m, NCV, NDQ, dtype=F, device=dev)), 2)         # stages between the MV and the row
    tau_row = T[:, None]*10**U(m, NCV, lo=-.35, hi=.35)*torch.where(r.bern(.12, m, NCV), 10**U(m, NCV, lo=-1., hi=-.4), 1.)
    Tp = T
    if multi:                                                    # between units: the places inside a unit do not count, every unit between is two stages
        far = torch.where(same, far, torch.ones_like(far))
        depth = torch.where(same, depth, 2.*dist.abs())
        tu = torch.gather(10**U(m, NMV, lo=-UNIT_T, hi=UNIT_T), 1, rb)               # every unit has its own time scale
        slow = (tu[:, :, None]*(1+dist.abs())*(K != 0)).amax((1, 2)).clamp_min(1.).sqrt()            # the slowest way through the plant, in unit time scales (square root)
        Tp, T = (T*slow).clamp(max=TAU_MAX), T/slow              # the decision interval is chosen for the plant: its units are faster in decisions than one on its own, the plant slower
        tau_row = tau_row*tu/slow[:, None]
        arch = torch.where(nb[:, 0] > 1, torch.full_like(arch, PLANT), a3[:, 0, 0])
    spread = torch.where(r.bern(.12, m, NCV, NU), U(m, NCV, NU, lo=-.6, hi=.6), U(m, NCV, NU, lo=-.2, hi=.2))
    tau = (tau_row[:, :, None]*10**spread*far**.3*(1+.5*depth)).clamp(.5, TAU_MAX)
    tau = torch.where(rs0[:, :, None], LU(m, NCV, lo=REST_TAU[0], hi=REST_TAU[1])[:, :, None], tau)
    kind = torch.where(rich[:, None, None], r.cat(m, KIND_P, NCV, NU), torch.zeros(m, NCV, NU, dtype=torch.long, device=dev))
    kind = torch.where(rs0[:, :, None], torch.zeros_like(kind), kind)
    W = lambda lo, hi: U(m, NCV, NU, lo=lo, hi=hi)               # noqa: E731
    # a storage behind a row: every answer of the row is a fast part (the pair's own time constant) and a slow one (the
    # storage's, the same for the whole row), in shares of the pair's own; the two modes with a zero between their poles
    has = store[:, None] > 0
    feel = torch.where(has, r.bern(STORE_ROW_P, m, NCV), r.bern(TAIL_ROW_P, m, NCV) & rich[:, None]) & ~rs0
    slow_t = (torch.where(has, store[:, None]*10**U(m, NCV, lo=-.1, hi=.1), tau_row*LU(m, NCV, lo=TAIL_T[0], hi=TAIL_T[1]))).clamp(max=STORE_T[1])[:, :, None]
    share = (U(m, NCV, lo=TAIL_SHARE[0], hi=TAIL_SHARE[1])[:, :, None]+W(-.15, .15)).clamp(.1, .95)
    tail = feel[:, :, None] & (slow_t >= TAIL_MIN*tau)
    kind = torch.where(tail, torch.full_like(kind, 5), kind)
    p1r, p1i = -1/tau, torch.zeros_like(tau)
    p2r, p2i = -1/torch.where(tail, slow_t, tau*W(.1, .95)), torch.zeros_like(tau)
    beta = torch.zeros_like(tau)
    beta = torch.where(kind == 2, tau*W(.2, 1.), beta)
    beta = torch.where(kind == 3, -tau*W(.5, 2.), beta)
    beta = torch.where(tail, -((1-share)*slow_t+share*tau), beta)
    zeta, wn = W(.2, .7), 1/tau
    osc = kind == 4
    p1r = torch.where(osc, -zeta*wn, p1r); p1i = torch.where(osc, wn*torch.sqrt(1-zeta**2), p1i)
    p2r = torch.where(osc, -zeta*wn, p2r); p2i = torch.where(osc, -wn*torch.sqrt(1-zeta**2), p2i)
    first = (kind == 0).to(F)
    ismv = (au[None, None] < NMV).to(F)
    coldead = torch.cat(((U(m, NMV)**2*3).floor(), torch.zeros(m, NDQ, dtype=F, device=dev)), 1)     # actuator transport
    dead = rth[:, None, None]*tau_row[:, :, None]*torch.where(ismv > 0, W(.4, 1.6), W(.2, 2.5))*far
    dead = (dead+depth*(rth*T)[:, None, None]*W(.3, 1.)+coldead[:, None, :]).floor().clamp(0, DMAX)
    dead = torch.where(rs0[:, :, None], coldead[:, None, :].expand(-1, NCV, -1), dead)             # a flow answers its valve at once
    dead = dead*prc[:, :, None]*uact[:, None, :]
    # ---- published pairing: the tracked rows are shuffled among themselves (row perm[r] of the pattern is row r)
    perm = torch.argsort(U(m, NCV)+(1-trk)*(10+ar[None]), -1)
    gat = lambda a: torch.gather(a, 1, perm[:, :, None].expand(-1, -1, a.shape[2]))   # noqa: E731
    K, p1r, p1i, p2r, p2i, beta, first, dead = (gat(a) for a in (K, p1r, p1i, p2r, p2i, beta, first, dead))
    inv = torch.argsort(perm, -1)
    link = torch.zeros(m, NCV, NMV, dtype=F, device=dev)
    rows = torch.arange(m, device=dev)
    for j in range(NMV):
        link[rows, inv[:, j], j] = mvm[:, j]
    rst = torch.gather(rs0, 1, perm)                             # resting rows, as published
    rstf = rst.to(F)
    units = dict(unit=torch.gather(rb, 1, perm).to(F), unitc=cb.to(F)) if multi else {}             # the unit of every row and of every input
    # ---- rows
    integ = trk*r.bern(.15, m, NCV).to(F)*(1-rstf)
    tint = LU(m, NCV, lo=20., hi=200.)
    lam = torch.where(r.bern(.3, m, NCV), U(m, NCV, lo=.05, hi=.3), torch.zeros(m, NCV, dtype=F, device=dev))*integ
    vc = U(m, NCV, lo=.3, hi=.7)
    nlag = r.cat(m, (.5, .3, .2), NCV).to(F)*(1-rstf)
    tlag = (torch.gather(tau_row, 1, perm)*10**U(m, NCV, lo=-.7, hi=-.1)).clamp(.5, 30.)
    lin1 = linear[:, None]
    wa = torch.where(r.bern(.4, m, NCV) | lin1 | rst, torch.zeros(m, NCV, dtype=F, device=dev), U(m, NCV, lo=-.6, hi=.6))
    sband = U(m, NCV, lo=.01, hi=.03)                             # what the instruments are scaled by
    bandf = torch.where(rst, U(m, NCV, lo=REST_BAND[0], hi=REST_BAND[1]), sband)
    # ---- MVs
    gam = torch.where(r.bern(.5, m, NMV) | lin1, torch.ones(m, NMV, dtype=F, device=dev), U(m, NMV, lo=.6, hi=1.7))
    aa = torch.where(r.bern(.5, m, NMV), torch.exp(-1/LU(m, NMV, lo=.3, hi=3.)), torch.zeros(m, NMV, dtype=F, device=dev))*mvm
    slew = LU(m, NMV, lo=.02, hi=.2)
    warp = (r.bern(.35, m) & ~linear).to(F)
    bydv = r.bern(.6, m) & (nd > 0)                              # the throughput is a measured feed rate, or else an MV
    jload = torch.where(bydv, NMV+(U(m)*nd).floor().clamp(max=ND-1), (U(m)*nm).floor().clamp(max=NMV-1))
    if multi:                                                    # the feed's rate, or an MV of the first unit
        jload = torch.where(bydv, torch.full_like(jload, NMV), (U(m)*(blk == 0).sum(1).clamp(max=nm)).floor())
    uref, kap = U(m, lo=.3, hi=.7), U(m, lo=.5, hi=1.)
    stock = torch.zeros(m, NCV, dtype=F, device=dev)             # the levels of a conserving train
    if 'stock' in ro and bool(ro['stock'].any()):
        K, p1r, p1i, p2r, p2i, beta, first, dead, integ, tint, lam, nlag, wa, bandf, stock = _stock(
            r, m, ro['stock'], dims, K, p1r, p1i, p2r, p2i, beta, first, dead, integ, tint, lam, nlag, wa, bandf, mvm, spr, nd, nq, inv, coldead)
    # ---- commissioning point: the bias puts a random input point at a random level, levels in balance there
    un0 = U(m, NMV, lo=.3, hi=.7)
    ue0 = un0**gam*mvm
    dlev = torch.cat((U(m, ND, lo=-1., hi=1.), U(m, NQ, lo=-.5, hi=.5)), 1)*dact
    base = (K*torch.cat((ue0, dlev), 1)[:, None, :]).sum(-1)
    v0 = torch.where(rst, base, U(m, NCV, lo=.3, hi=.7))         # a resting row reads the flow itself (no bias)
    bias = torch.where(integ > 0, -base-lam*(v0-vc), v0-base)*prc
    yc = h(v0, wa)
    gap = bandf*LU(m, NCV, lo=3., hi=40.)
    side = U(m, NCV)
    inf = torch.full((m, NCV), math.inf, dtype=F, device=dev)
    lo = torch.where((zon > 0) & (side < .65), yc-gap, -inf)
    hi = torch.where((zon > 0) & (side >= .35), yc+gap*U(m, NCV, lo=.8, hi=1.2), inf)
    p = dict(nm=nm.to(F), trk=trk, zon=zon, dvr=dvr, prc=prc, mvm=mvm, dact=dact, uact=uact, dvsel=dvsel.to(F), link=link,
             rst=rstf, spr=spr.to(F), stock=stock,
             K=K, p1r=p1r, p1i=p1i, p2r=p2r, p2i=p2i, beta=beta, first=first, dead=dead,
             integ=integ, tint=tint, lam=lam, vc=vc, nlag=nlag, tlag=tlag, wa=wa, bias=bias, bandf=bandf, sband=sband,
             gam=gam, aa=aa, slew=slew, warp=warp, jload=jload, uref=uref, kap=kap,
             dlev=dlev, lo=lo, hi=hi, cvw=draw_priority(r, trk, rstf), dt=ro['dt'],
             T=Tp, rth=rth, store=store, arch=arch.to(F), linear=linear.to(F), **units)
    # ---- valve friction, never coarser than half a band on any tracked CV at the commissioning point
    G, both = scaled_gains(p, un0*mvm, v0)
    p['fcap'] = .5*STICK_MAX/(G.abs()*both).amax(1).clamp_min(1e-9)
    fs = torch.where(r.bern(.4, m, NMV) & ~lin1, U(m, NMV, lo=.002, hi=.02), torch.zeros(m, NMV, dtype=F, device=dev))*mvm
    p['fs'] = torch.minimum(fs, p['fcap'])
    p['fd'] = p['fs']*U(m, NMV, lo=.3, hi=1.)
    tgt = h(v0, wa)*trk
    p['ulo'], p['uhi'] = torch.full_like(mvm, ULO), torch.full_like(mvm, UHI)
    p['rtgt'] = tgt.clone()                                      # what the plant can reach: the targets, unless the point is constrained
    p['mech'] = torch.zeros(m, dtype=F, device=dev)              # 0: a plant of the pair bank (else aio.sim.mech: the plant's number + 1)
    p['tlo'], p['thi'] = .03*trk, 1-.03*trk                      # where a target may be (a mechanistic plant: its own range)
    p['psign'] = torch.sign((K[:, :, :NMV]*link).sum(1))*mvm     # sign of every MV's gain on its published CV, as commissioned
    p['zal'] = torch.zeros_like(tgt)                             # bands a zone CV may sit beyond its limit: 0, unless the point is constrained
    if multi:                                                    # MVs that are not the controller's (THIN): among those with a process CV that settles, never all of them
        can = (mvm > 0) & ~spr & (torch.gather(integ, 1, inv[:, :NMV]) == 0)
        nc = can.sum(1)
        nh = torch.minimum((ro['hs']*nc).ceil().long(), (nc-1).clamp_min(0))
        p['hid'] = ((torch.argsort(torch.argsort(U(m, NMV)+10*(~can), -1), -1) < nh[:, None]) & can).to(F)
    return derive(p), tgt, ro


def draw_priority(r, trk, rst, prob=.5, ratio=3.):
    """CV priority weights (the equal-concern errors an APC engineer sets): log-uniform in [1/ratio, ratio]
    on the tracked process CVs of half the plants, geometric mean 1 so the cost scale does not move. A
    resting row keeps 1: its wide band is its priority."""
    NCV = trk.shape[1]
    m = trk.shape[0]
    cv = trk*(1-rst)
    lw = (2*r.u(m, NCV)-1)*math.log(ratio)*cv
    lw = (lw-lw.sum(1, keepdim=True)/cv.sum(1, keepdim=True).clamp_min(1))*cv
    w = torch.where(r.bern(prob, m)[:, None], lw.exp(), torch.ones_like(lw))
    return torch.where(rst > 0, torch.ones_like(w), w)


# ---------------------------------------------------------------- steady state and operability
def _system(p):
    """A (m, NMV, NMV): the steady equations of the tracked rows in the valve flows (identity elsewhere)."""
    NCV, NMV, ND, NQ, NDQ, NU = sizes(p)
    eye = torch.eye(NMV, dtype=F, device=p['K'].device).expand(p['K'].shape[0], -1, -1)
    t5 = p['trk'][:, :NMV] > 0
    return torch.where(t5[:, :, None], p['K'][:, :NMV, :NMV], eye), t5


def solve(p, tgt):
    """Steady valve flows ue (m, NMV) that hold the tracked targets under the current disturbance levels,
    the steady value v of every row, and whether the targets are on the output curves. A level (integrating
    row) is steady where its inflows balance, whatever its height; a slowly unstable one where they balance
    its own drift."""
    NCV, NMV, ND, NQ, NDQ, NU = sizes(p)
    vt, okh = hinv(tgt, p['wa'])
    dss = (p['K'][:, :, NMV:]*p['dlev'][:, None, :]).sum(-1)+p['bias']
    A, t5 = _system(p)
    ig = p['integ'][:, :NMV] > 0
    need = torch.where(ig, -dss[:, :NMV]-(p['lam']*(vt-p['vc']))[:, :NMV], vt[:, :NMV]-dss[:, :NMV])
    rhs = torch.where(t5, need, torch.zeros_like(need))
    ue, bad = torch.linalg.solve_ex(A, rhs[..., None])
    ue = torch.where(bad[:, None] > 0, torch.full_like(rhs, math.nan), ue[..., 0])*p['mvm']
    v = (p['K'][:, :, :NMV]*ue[:, None, :]).sum(-1)+dss
    v = torch.where(p['integ'] > 0, vt, v)
    return ue, v, (okh | (p['trk'] == 0)).all(-1)


def scaled_gains(p, un, v):
    """Steady gains of the tracked block in bands per full MV span at this operating point (integrating
    rows: the change over 100 decisions), and the mask of its entries (identity x 1e3 elsewhere)."""
    NCV, NMV, ND, NQ, NDQ, NU = sizes(p)
    dv = hprime(v, p['wa'])[:, :NMV, None]
    rate = torch.where(p['integ'][:, :NMV, None] > 0, 100/p['tint'][:, :NMV, None], torch.ones_like(dv))
    G = dv*rate*p['K'][:, :NMV, :NMV]*(p['gam']*un.clamp_min(1e-6)**(p['gam']-1))[:, None, :]/p['bandf'][:, :NMV, None]
    both = (p['trk'][:, :NMV, None] > 0) & (p['mvm'] > 0)[:, None, :]
    return torch.where(both, G, torch.eye(NMV, dtype=F, device=G.device)*1e3), both


def rga(p, G):
    """Relative gain of every MV with its published CV (1 on inactive MVs)."""
    NCV, NMV, ND, NQ, NDQ, NU = sizes(p)
    L = G*torch.linalg.inv(G).transpose(1, 2)
    return torch.where(p['mvm'] > 0, (L*p['link'][:, :NMV, :]).sum(1), torch.ones_like(p['mvm']))


def held(p):
    """(m, NMV) the MVs that are not the controller's (THIN): their limits are one position, where they stand for good."""
    return (p['mvm'] > 0) & (p['uhi'] <= p['ulo'])


def operable(p, tgt):
    """(m,) bool and the steady point: the tracked targets can be held with every valve U_MARGIN inside its
    limits (8-92 % as commissioned), every zone CV a band inside its limits, the targets inside the published range, gains neither singular nor
    needing sub-0.1 % actuator resolution, and a sticking valve moving no tracked CV by more than a band
    per stick (task rigor rules 1, 7). The one check of every operating point of every episode."""
    ue, v, ok = solve(p, tgt)
    mvm = p['mvm'] > 0
    un = ue.clamp_min(0)**(1/p['gam'])
    ok = ok & torch.isfinite(ue).all(-1) & ((ue > 0) | ~mvm).all(-1)
    pin = held(p)                                                # (an MV that is not the controller's stands where it is held)
    ok = ok & (((un >= p['ulo']+U_MARGIN-ON) & (un <= p['uhi']-U_MARGIN+ON)) | ~mvm | pin).all(-1) & (((un-p['ulo']).abs() <= 1e-9) | ~pin).all(-1)
    y = h(v, p['wa'])
    m = ZONE_MARGIN*p['bandf']
    ok = ok & (((y >= p['lo']+m-ON) & (y <= p['hi']-m+ON)) | (p['zon'] == 0)).all(-1)
    ok = ok & (((tgt >= .03-ON) & (tgt <= .97+ON)) | (p['trk'] == 0)).all(-1)
    ok = ok & (((v > -.3) & (v < 1.3)) | (p['prc'] == 0)).all(-1)
    G, both = scaled_gains(p, un, v)
    G = torch.where(torch.isfinite(G), G, torch.zeros_like(G))
    sv = torch.linalg.svdvals(G)
    ok = ok & (sv[:, -1] >= SIGMA_MIN) & ((G.abs()*both).amax((1, 2)) <= GAIN_MAX)
    ok = ok & ((G.abs()*both*p['fs'][:, None, :]).amax((1, 2)) <= STICK_MAX)
    return ok, dict(ue=ue, un=un*p['mvm'], v=v, y=y, G=G)


def phi_of(p, xv, d):
    """Time scale of the dynamics at valve positions xv and disturbance levels d (load-dependent plants;
    1 otherwise): the load is the position of the throughput valve or a measured feed rate (-1..1)."""
    NCV, NMV, ND, NQ, NDQ, NU = sizes(p)
    load = torch.gather(torch.cat((xv, (d[:, :ND]+1)/2), 1), 1, p['jload'].long()[:, None])[:, 0].clamp(0., 1.)
    w = ((load+.25)/(p['uref']+.25))**p['kap']
    return torch.where(p['warp'] > 0, w, torch.ones_like(w)).clamp(.4, 2.5)


def _vbounds(p):
    """Where the value of every process row (before its output curve) may sit at a steady point: tracked
    rows inside 4-96 % of the published range, zone rows 1.2 margins inside their limits, both on the
    invertible part of the curve."""
    a, b = p['wa'], p['bandf']
    zinv = torch.where(a < 0, .9/(4*(-a).clamp_min(1e-9)), torch.full_like(a, 9.))
    zt, zz = zinv.clamp_max(.46), zinv.clamp_max(.7)
    trk = p['trk'] > 0
    mg = 1.2*ZONE_MARGIN*b
    ylo = torch.where(trk, .5-zt, torch.maximum(.5-zz, torch.where(torch.isfinite(p['lo']), p['lo']+mg, -torch.ones_like(a))))
    yhi = torch.where(trk, .5+zt, torch.minimum(.5+zz, torch.where(torch.isfinite(p['hi']), p['hi']-mg, 2*torch.ones_like(a))))
    return hinv(ylo, a)[0], hinv(yhi, a)[0]


def reach(p, info, tgt, dvt, dd):
    """(m,) the largest s in [0, 1] for which the targets moved by s x dvt (in row value, before the output
    curve) and the disturbance levels moved by s x dd leave every valve inside its margins, every target
    inside the range and every zone CV inside its limits. The steady valve flows are linear in both, so the
    answer is exact; the gain checks of `operable` are left to it."""
    NCV, NMV, ND, NQ, NDQ, NU = sizes(p)
    K, mvm = p['K'], p['mvm']
    kd = (K[:, :, NMV:]*dd[:, None, :]).sum(-1)
    A, t5 = _system(p)
    ig = p['integ'][:, :NMV] > 0
    rhs = torch.where(ig, -kd[:, :NMV]-p['lam'][:, :NMV]*dvt[:, :NMV], dvt[:, :NMV]-kd[:, :NMV])
    du, bad = torch.linalg.solve_ex(A, torch.where(t5, rhs, torch.zeros_like(rhs))[..., None])
    du = du[..., 0]*mvm
    big = torch.full_like(kd[:, :1], 1e9)

    def room(x, dx, lo, hi, on):
        s = torch.where(dx > 1e-12, (hi-x)/dx.clamp_min(1e-12), torch.where(dx < -1e-12, (lo-x)/dx.clamp_max(-1e-12), big))
        return torch.where(on, s, big).amin(-1)

    g = p['gam']
    s = room(info['ue'], du, (p['ulo']+U_MARGIN+.01)**g, (p['uhi']-U_MARGIN-.01)**g, mvm > 0)
    trk = p['trk'] > 0
    vt, _ = hinv(tgt, p['wa'])
    v = torch.where(trk, vt, info['v'])
    dv = torch.where(trk, dvt, (K[:, :, :NMV]*du[:, None, :]).sum(-1)+kd)
    vlo, vhi = _vbounds(p)
    s = torch.minimum(s, room(v, dv, vlo, vhi, p['prc'] > 0))
    s = torch.where((bad > 0) | ~torch.isfinite(s), torch.zeros_like(s), s)
    return s.clamp(0., 1.)


# ---------------------------------------------------------------- constrained operating points
def _box_qp(H, g, lo, hi, x0=None, E=None, e=None, mu0=None):
    """Minimiser of x'Hx/2 + g'x inside lo..hi (m, n), H positive definite: the primal active-set method
    (Nocedal & Wright, algorithm 16.3) from x0 put inside the box. The variables of the working set sit on a
    limit, the others take the Newton step: all of it where it stays inside, else as far as the first limit
    met, whose variable joins the set; at the minimiser of a working set the variables whose gradient points
    into the box leave it. The result is the exact minimiser of the set it ends on, in a number of solves
    that grows with the number of limits that change, not with 3^n. Returns x and whether the first-order
    conditions hold there (the gradient to a part in 1e13 of its own terms: a thousand times their rounding).

    With E (m, q, n) and e (m, q) the rows Ex = e are to hold before anything in H is traded (levels in
    balance): the cost has |Ex - e|^2/(2 EQ_SOFT) more. The rows are not added to H, where a weight of
    1/EQ_SOFT buries the digits of everything else (the method of weighting on the normal equations: Bjorck,
    Numerical Methods for Least Squares Problems, section 5.1). They are carried as multipliers
    mu = (Ex - e)/EQ_SOFT, and a working set is solved in the augmented form
        [ H   E'       ] [x ]   [-g]
        [ E  -EQ_SOFT  ] [mu] = [ e]
    (the quasi-definite system of the regularised QP solvers), whose conditioning is that of H on the
    directions the rows leave free, whatever the weight. x, mu and the rows' residual move by increments,
    so a solve refines the one before. Where the limits leave no way to hold a row it gives as little as it
    can, and the caller sees that in Ex - e. Returns x, ok and mu; mu0 is the mu of x0 from an earlier call
    with the same rows."""
    dev, n = H.device, H.shape[1]
    eq = E is not None
    eye = torch.eye(n, dtype=H.dtype, device=dev)
    inf = torch.full_like(g, math.inf)

    def first_order(H, g, x, E=None, mu=None):
        grad, size = g+(H@x[..., None])[..., 0], g.abs()+(H.abs()@x.abs()[..., None])[..., 0]
        if mu is not None:
            grad, size = grad+(E.transpose(1, 2)@mu[..., None])[..., 0], size+(E.abs().transpose(1, 2)@mu.abs()[..., None])[..., 0]
        return grad, 1e-13*size+1e-13

    x = torch.zeros_like(g) if x0 is None else torch.where(torch.isfinite(x0), x0, torch.zeros_like(g))
    x = torch.minimum(torch.maximum(x, lo), hi)
    mu = rho = None
    if eq:
        soft = EQ_SOFT*torch.eye(E.shape[1], dtype=H.dtype, device=dev)
        c = (E@x[..., None])[..., 0]-e
        mu = c/EQ_SOFT if mu0 is None else mu0
        rho = c-EQ_SOFT*mu                                         # what the rows are off by, beyond what mu accounts for
    grad, tg = first_order(H, g, x, E, mu)
    wl, wh = (x <= lo) & (grad > 0), (x >= hi) & (grad < 0)       # the working set: at a limit, the gradient pointing out of the box
    live = torch.ones(g.shape[0], dtype=torch.bool, device=dev)
    for _ in range(6*n+20):
        i = live.nonzero().flatten()
        if i.numel() == 0:
            break
        Hi, gi, xi, loi, hii, wli, whi = H[i], g[i], x[i], lo[i], hi[i], wl[i], wh[i]
        Ei, mui, rhoi = (E[i], mu[i], rho[i]) if eq else (None, None, None)
        grad, _ = first_order(Hi, gi, xi, Ei, mui)
        free = ~(wli | whi)
        A = torch.where(free[:, :, None] & free[:, None, :], Hi, eye*(~free)[:, :, None].to(H.dtype))
        rhs = torch.where(free, -grad, torch.zeros_like(grad))
        if eq:
            Ef = Ei*free[:, None, :]
            A = torch.cat((torch.cat((A, Ef.transpose(1, 2)), 2), torch.cat((Ef, -soft.expand(i.numel(), -1, -1)), 2)), 1)
            rhs = torch.cat((rhs, -rhoi), 1)
        sol, bad = torch.linalg.solve_ex(A, rhs[..., None])
        good = (bad == 0) & torch.isfinite(sol[..., 0]).all(-1)
        step = torch.where(good[:, None] & free, sol[:, :n, 0], torch.zeros_like(grad))
        room = torch.where(step > 0, (hii-xi)/step, torch.where(step < 0, (loi-xi)/step, inf[i]))    # share of the step to the limit ahead
        alpha, jb = room.min(1)
        hit = alpha < 1
        alpha = alpha.clamp(0., 1.)[:, None]
        at = torch.nn.functional.one_hot(jb, n).bool() & hit[:, None]                   # the variable that meets its limit first
        up = torch.gather(step, 1, jb[:, None]) > 0
        xi = torch.where(at, torch.where(up, hii, loi), torch.minimum(torch.maximum(xi+alpha*step, loi), hii))
        wli, whi = wli | (at & ~up), whi | (at & up)
        if eq:
            dmu = torch.where(good[:, None], sol[:, n:, 0], torch.zeros_like(mui))
            mui, rhoi = mui+alpha*dmu, rhoi+alpha*((Ef@step[..., None])[..., 0]-EQ_SOFT*dmu)
        grad, tg = first_order(Hi, gi, xi, Ei, mui)
        out_lo, out_hi = wli & (grad < -tg) & ~hit[:, None], whi & (grad > tg) & ~hit[:, None]     # at the minimiser of the set: these leave it
        wli, whi = wli & ~out_lo, whi & ~out_hi
        x, wl, wh = x.index_copy(0, i, xi), wl.index_copy(0, i, wli), wh.index_copy(0, i, whi)
        rough = ((grad.abs() > tg) & ~(wli | whi)).any(-1)        # the solve left more than rounding: one more step refines it
        if eq:
            mu, rho = mu.index_copy(0, i, mui), rho.index_copy(0, i, rhoi)
            rough = rough | (rhoi.abs() > _row_tol(Ei, xi, e[i], mui)).any(-1)
        live = live.index_copy(0, i, hit | out_lo.any(-1) | out_hi.any(-1) | rough)
    grad, tg = first_order(H, g, x, E, mu)
    at_lo, at_hi = x <= lo, x >= hi
    ok = ~live & torch.isfinite(x).all(-1) & (lo <= hi).all(-1) & ((grad.abs() <= tg) | (at_lo & (grad >= -tg)) | (at_hi & (grad <= tg))).all(-1)
    if eq:
        return x, ok & (rho.abs() <= _row_tol(E, x, e, mu)).all(-1), mu
    return x, ok


def _row_tol(E, x, e, mu):
    """A part in 1e13 of the terms of Ex - e - EQ_SOFT mu."""
    return 1e-13*((E.abs()@x.abs()[..., None])[..., 0]+e.abs()+EQ_SOFT*mu.abs())+1e-13


def steady_weight(p):
    """(m, NCV) what a tracked row weighs in the best steady point: its priority, and REST_EPS of it for a
    resting value (it gives before any process CV does)."""
    return p['cvw']*torch.where(p['rst'] > 0, REST_EPS, 1.)


def levels(p, vt):
    """The rows that are levels to keep in balance, as equalities of a steady point: E (m, NMV, NMV) and
    e (m, NMV) with Ex - e the drift of the level at the valve flows x, in bands per 100 decisions, when it
    stands at the value vt (before the output curve); and what turns the level's inflow into that drift
    (m, NMV; 0 on the rows that are no level). Tracked rows are among the first NMV."""
    NCV, NMV, ND, NQ, NDQ, NU = sizes(p)
    rate = (100/(p['tint']*p['bandf'])*hprime(vt, p['wa'])*(p['integ'] > 0)*p['trk'])[:, :NMV]
    dss = (p['K'][:, :NMV, NMV:]*p['dlev'][:, None, :]).sum(-1)+p['bias'][:, :NMV]
    return rate[:, :, None]*p['K'][:, :NMV, :NMV], -rate*(dss+(p['lam']*(vt-p['vc']))[:, :NMV]), rate


def optimum(p, tgt, iters=12):
    """The best steady point when the targets cannot all be held: the valve flows inside the MV limits that
    minimise the steady cost as the simulator charges it (priority x squared tracking error in bands plus
    10 x squared zone violation in bands, resting values last: `steady_weight`) with every level in balance
    (`levels`: rows that hold before any of that is traded, EQ_SOFT). Gauss-Newton on the output curves
    with an exact box solve each time; a zone row counts while it is beyond its limit (the cost is smooth
    there, so the set settles in a few rounds). Returns the valve flows ue (m, NMV), the steady value v and
    the CV value y of every row, the valves at a limit, and whether the point is verified (first-order
    conditions of the true cost, levels balanced)."""
    NCV, NMV, ND, NQ, NDQ, NU = sizes(p)
    mvm, g = p['mvm'], p['gam']
    Km = p['K'][:, :, :NMV]
    dss = (p['K'][:, :, NMV:]*p['dlev'][:, None, :]).sum(-1)+p['bias']
    trk, ig, zon = p['trk'], p['integ'] > 0, p['zon']
    wa, band, sw = p['wa'], p['bandf'], torch.sqrt(steady_weight(p))
    vt = hinv(tgt, wa)[0]
    lo, hi = p['ulo']**g*mvm, p['uhi']**g*mvm
    E, e, _ = levels(p, vt)
    ridge = torch.diag_embed(1-mvm)+1e-12*torch.eye(NMV, dtype=F, device=tgt.device)
    sz, zero = math.sqrt(10.), torch.zeros_like(tgt)

    def parts(x):
        fl = (Km*x[:, None, :]).sum(-1)+dss
        y, slope = h(fl, wa), hprime(fl, wa)
        over = torch.where(y > p['hi'], y-p['hi'], torch.where(y < p['lo'], y-p['lo'], zero))*zon      # beyond a zone limit, signed
        r = torch.where(ig, zero, sw*(y-tgt)/band)*trk+sz*over/band
        J = (torch.where(ig, zero, sw*slope/band)*trk+sz*slope/band*(over != 0))[:, :, None]*Km
        return r, J

    x = solve(p, tgt)[0]
    x = torch.minimum(torch.maximum(torch.where(torch.isfinite(x), x, .5*(lo+hi)), lo), hi)
    found, mu = torch.ones(tgt.shape[0], dtype=torch.bool, device=tgt.device), None
    for _ in range(iters):
        r, J = parts(x)
        H = J.transpose(1, 2)@J+ridge
        gq = (J.transpose(1, 2)@(r-(J@x[..., None])[..., 0])[..., None])[..., 0]
        x, found, mu = _box_qp(H, gq, lo, hi, x, E, e, mu)
    r, J = parts(x)
    grad = (J.transpose(1, 2)@r[..., None])[..., 0]+(E.transpose(1, 2)@mu[..., None])[..., 0]
    tol = (1e-6*((J.abs()*r.abs()[:, :, None]).sum(1)+(E.abs()*mu.abs()[:, :, None]).sum(1))+1e-9
           +1e-11*((J.transpose(1, 2)@J).abs()@x.abs()[..., None])[..., 0])       # of the terms of the gradient; at a zero residual what is left is the rounding of J'J x
    at_lo, at_hi = (x <= lo+1e-12) & (mvm > 0), (x >= hi-1e-12) & (mvm > 0)
    kkt = (((grad.abs() <= tol) | at_lo | at_hi) & (~at_lo | (grad >= -tol)) & (~at_hi | (grad <= tol)) | (mvm == 0) | (hi <= lo)).all(-1)   # (an MV held at one position: whatever its gradient)
    bal = (((E@x[..., None])[..., 0]-e).abs() <= .02).all(-1)     # a level drifts by less than 0.02 band in 100 decisions
    v = (Km*x[:, None, :]).sum(-1)+dss
    v = torch.where(ig, vt, v)
    return x, v, h(v, wa), at_lo | at_hi, found & kkt & bal & torch.isfinite(x).all(-1)


def unremovable(p, info):
    """(m, NCV, NDQ) what a unit of every disturbance level does to the CVs at the steady point `info` once
    the valves at a limit stay there and the free ones are set again for the best trade-off: bands for a
    tracked CV that settles and for a zone CV beyond its limit, bands of drift per 100 decisions for a
    level. 0 at an operable point (every valve free, no zone limit in the way); at a constrained one it is
    the part of a wandering load no controller can take out. The trade-off is `optimum`'s, levels first
    (the same augmented system)."""
    NCV, NMV, ND, NQ, NDQ, NU = sizes(p)
    mvm = p['mvm'] > 0
    un = info['un']
    sat = ((un <= p['ulo']+1e-9) | (un >= p['uhi']-1e-9)) & mvm
    free = (mvm & ~sat).to(F)
    ig = p['integ'] > 0
    zact = ((info['y'] > p['hi']) | (info['y'] < p['lo'])) & (p['zon'] > 0)
    per = torch.where(zact, torch.full_like(info['y'], math.sqrt(10.)), torch.sqrt(steady_weight(p)).clamp_min(1e-9))
    w = per*hprime(info['v'], p['wa'])/p['bandf']*(p['trk']*~ig+zact.to(F))            # the residuals of `optimum`
    J = w[:, :, None]*p['K'][:, :, :NMV]*free[:, None, :]
    Jd = w[:, :, None]*p['K'][:, :, NMV:]*p['dact'][:, None, :]
    E, _, rate = levels(p, info['v'])
    E, Ed = E*free[:, None, :], rate[:, :, None]*p['K'][:, :NMV, NMV:]*p['dact'][:, None, :]
    H = J.transpose(1, 2)@J+torch.diag_embed(1-free)
    H = H+1e-12*torch.diag_embed(H.diagonal(dim1=1, dim2=2))
    soft = EQ_SOFT*torch.eye(NMV, dtype=F, device=H.device).expand(H.shape[0], -1, -1)
    A = torch.cat((torch.cat((H, E.transpose(1, 2)), 2), torch.cat((E, -soft), 2)), 1)
    X, bad = torch.linalg.solve_ex(A, torch.cat((J.transpose(1, 2)@Jd, Ed), 1))
    X = X[:, :NMV]
    R = (Jd-J@X)/per[:, :, None]
    R = torch.cat((torch.where(rate[:, :, None] > 0, Ed-E@X, R[:, :NMV]), R[:, NMV:]), 1)
    on = (bad == 0) & (sat.any(-1) | zact.any(-1))
    return torch.where(on[:, None, None] & torch.isfinite(R), R, torch.zeros_like(R))


def is_excursion(kinds):
    """Whether an event kind takes the unit to a constrained point."""
    return torch.isin(kinds, torch.tensor(EXCURSIONS, device=kinds.device))


def constrained(p, tgt):
    """(m,) whether the operating point is a constrained one (its reachable targets are not its targets)."""
    return ((p['rtgt']-tgt).abs()*p['trk']).amax(-1) > 1e-9


def _constrained_ok(p, tgt):
    """The checks of a constrained point, as `operable` for the others: `optimum` verified with a valve at
    its limit or a zone CV at its limit, no zone CV more than ZONE_GIVE bands beyond its limit, targets
    inside the range, sane gains, the worst tracked CV off its target by GIVE bands (a resting value: by
    that many bands of its instrument), and no level left to drift when a load moves. Returns (ok, steady point, reachable targets, zone allowance): the allowance
    is, for a zone CV less than ZONE_MARGIN inside its limit there, how far beyond the limit it sits plus a
    band (0 for the others)."""
    ue, v, y, sat, ok = optimum(p, tgt)
    un = (ue.clamp_min(0)**(1/p['gam']))*p['mvm']
    trk, zon = p['trk'] > 0, p['zon'] > 0
    inside = torch.minimum(y-p['lo'], p['hi']-y)/p['bandf']       # bands inside the nearer limit
    near = zon & (inside < ZONE_MARGIN)
    ok = ok & (sat.any(-1) | near.any(-1))
    ok = ok & ((inside >= -ZONE_GIVE) | ~zon).all(-1)
    ok = ok & (((tgt >= .03-ON) & (tgt <= .97+ON)) | ~trk).all(-1) & (((v > -.3) & (v < 1.3)) | (p['prc'] == 0)).all(-1)
    off = (y-tgt).abs()/p['sband']*p['trk']
    give = off.amax(-1)
    ok = ok & ((give >= GIVE[0]) | held(p).any(-1)) & ((off*(1-p['rst'])).amax(-1) <= GIVE[1] if 'unit' in p else give <= GIVE[1])   # (a plant with more setpoints than MVs: any trade-off, however
                                                                 # small; a larger plant: a spare MV may stop anywhere short of its limit)
    G, both = scaled_gains(p, un, v)
    G = torch.where(torch.isfinite(G), G, torch.zeros_like(G))
    ok = ok & (torch.linalg.svdvals(G)[:, -1] >= SIGMA_MIN) & ((G.abs()*both).amax((1, 2)) <= GAIN_MAX)
    ok = ok & ((G.abs()*both*p['fs'][:, None, :]).amax((1, 2)) <= STICK_MAX)
    info = dict(ue=ue, un=un, v=v, y=y, G=G)
    drift = (unremovable(p, info).abs()*(p['integ'] > 0)[:, :, None]).amax((1, 2))
    ok = ok & (drift <= LEVEL_DRIFT)                              # the free valves keep every level in balance when a load moves
    zal = torch.where(near, (-inside).clamp_min(0.)+1., torch.zeros_like(y))
    return ok, info, torch.where(trk, y, tgt), zal


def point_ok(p, tgt):
    """(m,) the one check of every operating point of every episode: operable, or a verified constrained
    point whose published reachable targets and zone allowance are the optimum's."""
    ok, _ = operable(p, tgt)
    okc, _, rt, zal = _constrained_ok(p, tgt)
    okc = okc & (((rt-p['rtgt']).abs()/p['bandf']*p['trk']).amax(-1) < 1e-6) & ((zal-p['zal']).abs().amax(-1) < 1e-6)
    return torch.where(constrained(p, tgt), okc, ok & (p['zal'] == 0).all(-1))


def point(p, tgt):
    """The operating point of a plant and its targets, whichever it is: (m,) whether it passes the one check
    (`operable`, or a verified constrained point), the steady point, the reachable targets and the zone
    allowance to publish."""
    ok1, inf1 = operable(p, tgt)
    ok2, inf2, rt, zal = _constrained_ok(p, tgt)
    w = lambda x, y: torch.where(ok1.view(-1, *([1]*(x.dim()-1))), x, y)             # noqa: E731
    return ok1 | ok2, {key: w(inf1[key], inf2[key]) for key in inf1}, w(torch.where(p['trk'] > 0, tgt, p['rtgt']), rt), w(torch.zeros_like(zal), zal)


def cause_of(p, tgt, q, tq):
    """What took a unit from the operable point (p, tgt) to the constrained point (q, tq), as something that
    can be put on the unit again after it has changed (`with_cause`): the shift of the disturbance levels
    and of the targets, the MV limits and the zone limits that were moved."""
    mu = (q['ulo'] != p['ulo']) | (q['uhi'] != p['uhi'])
    mz = (q['lo'] != p['lo']) | (q['hi'] != p['hi'])
    return dict(dlev=q['dlev']-p['dlev'], tgt=tq-tgt, mu=mu, ulo=q['ulo'], uhi=q['uhi'], mz=mz, lo=q['lo'], hi=q['hi'])


def with_cause(p, tgt, c):
    """The plant and the targets with what took the unit to a constrained point (`cause_of`) put on them."""
    q = dict(p)
    q['dlev'] = p['dlev']+c['dlev']
    q['ulo'], q['uhi'] = torch.where(c['mu'], c['ulo'], p['ulo']), torch.where(c['mu'], c['uhi'], p['uhi'])
    q['lo'], q['hi'] = torch.where(c['mz'], c['lo'], p['lo']), torch.where(c['mz'], c['hi'], p['hi'])
    return q, tgt+c['tgt']


def draw_excursion(r, p, info, tgt):
    """A candidate constrained point from the operable steady point `info`. Six ways, as they happen on a
    unit; in the first three one valve is asked for more than its limit gives, by a few bands' worth of
    travel:
      over_setpoint  the operator asks for targets whose steady point has that valve beyond its limit
                     (resting values stay as they are);
      over_load      a disturbance or load moves until the valve would have to be beyond its limit;
      mv_limit       the operator narrows the valve's limit past its steady position;
      zone_limit     the operator moves a zone limit past where its CV sits (the setpoints have to give);
      mv_out         one MV is taken out of service where it stands (PIN of room either side), and the
                     operator moves the setpoints of the CVs that settle (levels keep theirs);
      push           (fat plants) the operator moves the resting value of a spare MV to the valve's limit,
                     as far as it can be asked for (U_MARGIN inside): the unit is to run with the most or
                     the least of it that the other valves and the zone limits allow. Valid when one of
                     them stops it, MOVE_MIN bands or more from where the MV stands.
    On a fat plant the first five are where a spare MV earns its place: it leaves its resting value before
    a process CV gives. Returns the plant, the targets, the steady point, the event kind and whether the
    candidate is valid."""
    NCV, NMV, ND, NQ, NDQ, NU = sizes(p)
    m = tgt.shape[0]
    mvm, g = p['mvm'], p['gam']
    trk, ig = p['trk'] > 0, p['integ'] > 0
    j1 = r.one_of(mvm > 0).to(F)                                  # the valve that runs out
    up = r.bern(.5, m)                                           # at its high limit
    both = (p['trk'][:, :NMV, None] > 0) & (mvm > 0)[:, None, :]
    gj = ((info['G'].abs()*both).amax(1)*j1).sum(-1)             # its strongest effect on a tracked CV (bands per span)
    delta = (r.logu(m, lo=3., hi=45.)/gj.clamp_min(1e-6)).clamp(.02, .3)
    pick = lambda a: (a*j1).sum(-1)                              # noqa: E731
    lim = torch.where(up, pick(p['uhi']), pick(p['ulo']))
    unj = torch.where(up, lim+delta, (lim-delta).clamp_min(.005))                    # where the valve would have to be
    zcan = (p['zon'] > 0) & (torch.isfinite(p['lo']) | torch.isfinite(p['hi'])) & (p['K'][:, :, :NMV] != 0).any(-1)
    kind = r.cat(m, EXC_KIND_P)
    kind = torch.where((kind == 1) & (p['dact'].sum(-1) == 0), torch.zeros_like(kind), kind)
    kind = torch.where((kind == 3) & ~zcan.any(-1), torch.zeros_like(kind), kind)
    spr, rst = p['spr'] > 0, p['rst'] > 0
    kind = torch.where(spr.any(-1) & r.bern(PUSH_P, m), torch.full_like(kind, 5), kind)
    q = dict(p)
    # ---- targets of the steady point with that valve there and the others where they are
    ue_c = torch.where(j1 > 0, unj[:, None]**g, info['ue'])*mvm
    dss = (p['K'][:, :, NMV:]*p['dlev'][:, None, :]).sum(-1)+p['bias']
    vt = hinv(tgt, p['wa'])[0]
    v_c = torch.where(trk & ~ig & ~rst, (p['K'][:, :, :NMV]*ue_c[:, None, :]).sum(-1)+dss, vt)
    vlo, vhi = _vbounds(p)
    ok_a = ((v_c >= vlo) & (v_c <= vhi) | ~trk).all(-1)
    t_a = torch.where(trk, h(v_c, p['wa']), torch.zeros_like(vt))
    # ---- level of one disturbance at which that valve would have to be there
    ch = r.one_of(p['dact'] > 0).to(F)
    A, t5 = _system(p)
    kd = (p['K'][:, :NMV, NMV:]*ch[:, None, :]).sum(-1)
    du, bad = torch.linalg.solve_ex(A, torch.where(t5, -kd, torch.zeros_like(kd))[..., None])
    duj = pick(du[..., 0]*mvm)                                   # flow that valve needs per unit of the disturbance
    s = (pick(unj[:, None]**g)-pick(info['ue']))/torch.where(duj >= 0, duj.clamp_min(1e-12), duj.clamp_max(-1e-12))
    dl_b = p['dlev']+s[:, None]*ch
    ok_b = (bad == 0) & torch.isfinite(s) & (dl_b.abs() <= 1.).all(-1) & (s.abs() > 1e-6)
    # ---- its limit, moved past its steady position
    unow = pick(info['un'])
    uhi_c = torch.where((j1 > 0) & up[:, None], (unow-delta)[:, None], p['uhi'])
    ulo_c = torch.where((j1 > 0) & ~up[:, None], (unow+delta)[:, None], p['ulo'])
    ok_c = ((uhi_c-ulo_c) >= .25).all(-1)
    # ---- a zone limit, moved past where its CV sits
    z1 = r.one_of(zcan)
    top = torch.where(torch.isfinite(p['hi']) & torch.isfinite(p['lo']), r.bern(.5, m, NCV), torch.isfinite(p['hi']))
    past = p['bandf']*r.logu(m, NCV, lo=ZONE_PAST[0], hi=ZONE_PAST[1])
    hi_d = torch.where(z1 & top, info['y']-past, p['hi'])
    lo_d = torch.where(z1 & ~top, info['y']+past, p['lo'])
    ok_d = z1.any(-1) & ((hi_d-lo_d >= 4*p['bandf']) | ~z1).all(-1)                  # the zone keeps some width
    # ---- that valve out of service where it stands, and a setpoint move (levels stay: with a valve short,
    # a level moves only as fast as the other CVs are given up for it, which no segment is long enough for)
    t_e, _ = draw_move(r, p, info, tgt)
    t_e = torch.where(ig, tgt, t_e)
    ulo_e = torch.where(j1 > 0, torch.maximum(p['ulo'], (unow-PIN)[:, None]), p['ulo'])
    uhi_e = torch.where(j1 > 0, torch.minimum(p['uhi'], (unow+PIN)[:, None]), p['uhi'])
    ok_e = ((t_e-tgt).abs()/p['bandf']*p['trk']).amax(-1) >= MOVE_MIN
    # ---- the resting value of a spare MV, moved to its valve's limit
    s1 = r.one_of(spr).to(F)
    far = torch.where(r.bern(.5, m)[:, None], p['uhi']-U_MARGIN, p['ulo']+U_MARGIN)**g
    if 'unit' in p:                                              # a larger plant: several at once, each to a limit of its own
        s1 = ((s1 > 0) | (spr & r.bern(PUSH_MORE, m, NMV))).to(F)
        far = torch.where(r.bern(.5, m, NMV), p['uhi']-U_MARGIN, p['ulo']+U_MARGIN)**g
    row = (p['link'][:, :, :NMV]*s1[:, None, :]).sum(-1) > 0     # its resting row
    v_f = info['v']+(p['K'][:, :, :NMV]*(s1*(far-info['ue']))[:, None, :]).sum(-1)
    t_f = torch.where(row, h(v_f, p['wa']), tgt)
    ok_f = spr.any(-1) & ((v_f >= vlo) & (v_f <= vhi) | ~row).all(-1)
    k1 = kind[:, None]
    tc = torch.where(k1 == 0, t_a, torch.where(k1 == 4, t_e, torch.where(k1 == 5, t_f, tgt)))
    q['dlev'] = torch.where(k1 == 1, dl_b, p['dlev'])
    q['ulo'] = torch.where(k1 == 2, ulo_c, torch.where(k1 == 4, ulo_e, p['ulo']))
    q['uhi'] = torch.where(k1 == 2, uhi_c, torch.where(k1 == 4, uhi_e, p['uhi']))
    q['lo'], q['hi'] = torch.where(k1 == 3, lo_d, p['lo']), torch.where(k1 == 3, hi_d, p['hi'])
    ok = torch.stack((ok_a, ok_b, ok_c, ok_d, ok_e, ok_f), 1).gather(1, k1)[:, 0]
    okc, infc, rt, zal = _constrained_ok(q, tc)
    gone = ((infc['y']-info['y']).abs()/p['bandf']*row).amax(-1) >= MOVE_MIN             # the spare MV has somewhere to go
    ok = ok & (gone | (kind != 5))
    q['rtgt'], q['zal'] = rt, zal
    code = torch.tensor(EXCURSIONS, device=kind.device)[kind]
    return q, tc, infc, code, ok & okc


# ---------------------------------------------------------------- events
RESTORED = (('K',), ('p1r', 'p1i', 'p2r', 'p2i', 'beta', 'dead', 'tlag'), ('gam', 'fs', 'fd'), ('wa',))   # what maintenance puts back together: gains, dynamics, valves, output curves


HEAVY_TURN = (.5, 1.5)                      # a heavy process change: the change of a row's gains over their root mean square
HEAVY_OWN = (.4, 2.5)                       # (the paired gain stays within this of what it was, its sign with it)
HEAVY_TIME = 3.                             # and its time constants change by up to this factor either way
HEAVY_TRIES = 6                             # candidates for the change of the gains
LASTING = ('K', 'p1r', 'p1i', 'p2r', 'p2i', 'beta')   # what a heavy change that stays (Menu.shift) leaves for maintenance to put back: the gains and time constants it made


def turn_gains(r, p, first, at, alpha, on, rowm):
    """A heavy change of the gains: (m, NCV, NMV) to add to the MV gains of the plants `on`, rows `rowm`. The
    new gains are drawn about those of the plant as commissioned (`first`: off them by HEAVY_TURN x alpha of
    the row's root mean square, cross effects of either sign), so heavy changes do not add up over a stay;
    and every row's change is at right angles to the valve flows of the steady point `at`, so that point
    stays where it is (targets held, every valve where it stands, levels in balance) while what a move of
    the valves does is something else. The paired gain keeps its sign and stays within HEAVY_OWN of the
    commissioned one. The first of HEAVY_TRIES candidates is taken that leaves the gains of the tracked block
    sane (as `operable` asks) and the published pairing one with positive relative gains; where none does,
    the first one halved until it does, at most three times, else nothing."""
    NCV, NMV, ND, NQ, NDQ, NU = sizes(p)
    m = p['K'].shape[0]
    Km, K0 = p['K'][:, :, :NMV], first['K'][:, :, :NMV]
    sup = (K0 != 0) & (p['mvm'] > 0)[:, None, :] & rowm           # the valves the row answers to
    ue = at['ue'][:, None, :]*sup
    rms = torch.sqrt((K0**2*sup).sum(-1, keepdim=True)/sup.sum(-1, keepdim=True).clamp_min(1))
    own = p['link'][:, :, :NMV] > 0
    sg = torch.sign((K0*own).sum(-1))
    a, a0 = sg*(Km*own).sum(-1), (K0*own).sum(-1).abs()           # the paired gain now and as commissioned, as positive numbers

    def candidate(size, z):
        D = (K0+rms*alpha[:, None, None]*size*z-Km)*sup
        D = D-(D*ue).sum(-1, keepdim=True)/(ue**2).sum(-1, keepdim=True).clamp_min(1e-30)*ue
        d = sg*(D*own).sum(-1)
        c = torch.where(d < 0, (a-HEAVY_OWN[0]*a0)/(-d).clamp_min(1e-12), (HEAVY_OWN[1]*a0-a)/d.clamp_min(1e-12)).clamp(0., 1.)
        return D*torch.where(a0 > 0, c, torch.ones_like(c))[:, :, None]*on[:, None, None]

    def fine(D):
        q = dict(p)
        q['K'] = torch.cat((Km+D, p['K'][:, :, NMV:]), -1)
        G, both = scaled_gains(q, at['un'], at['v'])
        G = torch.where(torch.isfinite(G), G, torch.zeros_like(G))
        Gi, bad = torch.linalg.inv_ex(G)
        lam = torch.where(p['mvm'] > 0, (G*Gi.transpose(1, 2)*p['link'][:, :NMV, :]).sum(1), torch.ones_like(p['mvm']))
        ok = (bad == 0) & (torch.linalg.svdvals(G)[:, -1] >= SIGMA_MIN) & ((G.abs()*both).amax((1, 2)) <= GAIN_MAX)
        return ok & ((G.abs()*both*p['fs'][:, None, :]).amax((1, 2)) <= STICK_MAX), lam

    size, z = r.u(m, HEAVY_TRIES, NCV, 1, lo=HEAVY_TURN[0], hi=HEAVY_TURN[1]), r.normal(m, HEAVY_TRIES, NCV, NMV)
    zero = torch.zeros_like(Km)
    _, lam0 = fine(zero)
    head, out, done = None, zero, ~on
    for t in range(HEAVY_TRIES):
        D = candidate(size[:, t], z[:, t])
        head = D if head is None else head
        ok, lam = fine(D)
        take = ~done & ok & ((lam >= RGA_MIN) | (lam >= lam0-1e-12)).all(-1)
        out, done = torch.where(take[:, None, None], D, out), done | take
    for f in (.5, .25, .125):
        if bool(done.all()):
            break
        ok, lam = fine(f*head)
        take = ~done & ok & ((lam >= RGA_MIN) | (lam >= lam0-1e-12)).all(-1)
        out, done = torch.where(take[:, None, None], f*head, out), done | take
    return out


def change_process(r, p, alpha, base=None, menu=None, at=None, first=None, shift=None):
    """A process event: 1-3 of gain drift, actuator loss, time-constant change, dead-time change, valve
    characteristic, output curve, friction onset, lag and disturbance-path drift, each scaled by alpha
    (m,) in (0, 1]. Factors are positive, so the published gain signs hold. A resting row (the reading of
    a spare MV's flow) changes only with its valve: an actuator loss shows in it, the rest passes it by.
    base: the plant as commissioned (a long stay): a share RESTORE_P of the events is maintenance instead,
    one or more of RESTORED go back to it (wear only ever adds up: actuators lose capacity, valves stick,
    dead times grow). at: the steady point the unit is at, first: the plant as commissioned; with Menu.heavy
    a share of the events is a heavy change instead of all that (`turn_gains`, and every row's time constant
    anywhere within HEAVY_TIME of the commissioned one; no dead time moves, no maintenance). shift: (m,) the
    plants whose event is a heavy change whatever the share (Menu.shift: the caller makes it stay)."""
    NCV, NMV, ND, NQ, NDQ, NU = sizes(p)
    m, dev = p['K'].shape[0], p['K'].device
    U = r.u
    q = dict(p)
    lin = p['linear'] > 0
    pick = r.bern(.2, m, 8)
    rows = torch.arange(m, device=dev)
    pick[rows, r.randint(0, 7, m)] = True
    pw = 1. if menu is None else float(menu.proc)
    powr = (lambda x: x) if pw == 1. else (lambda x: x**pw)       # noqa: E731
    if menu is not None and menu.drop:                            # kinds left out; an event that had only those takes the first kind allowed
        allowed = torch.tensor([name not in menu.drop for name in PROCESS], device=dev)
        pick = pick & allowed[None]
        if bool(allowed.any()):
            pick[~pick.any(1), int(allowed.to(F).argmax())] = True
    hv = None
    if menu is not None and menu.heavy > 0 and at is not None and first is not None:    # a heavy change instead of the usual kinds
        hv = r.bern(float(menu.heavy), m)
    if shift is not None:
        hv = shift if hv is None else hv | shift
    if hv is not None:
        pick = pick & ~hv[:, None]
    act = p['prc'][:, :, None]*p['uact'][:, None, :]
    ismv = (torch.arange(NU, device=dev) < NMV)[None, None]
    cv = ((p['rst'] == 0) & (p['stock'] == 0))[:, :, None]       # not a flow reading, nor the level of a conserving train (its streams change with their valves alone)
    rowh, mvh = torch.ones(m, NCV, 1, dtype=torch.bool, device=dev), torch.ones(m, NMV, dtype=torch.bool, device=dev)
    if 'unit' in p:                                              # several units: most events strike one of them (a loss of actuator or friction: one valve anyway)
        loc, u = r.bern(LOCAL_P, m)[:, None], (r.one_of(p['prc'] > 0).to(F)*p['unit']).sum(1, keepdim=True)
        rowh, mvh = (~(loc & (p['unit'] != u)))[:, :, None], ~(loc & (p['unitc'][:, :NMV] != u))
        cv = cv & rowh
    a1, a2, a3 = alpha[:, None], alpha[:, None, None], alpha[:, None, None]
    drift = powr(1+a3*U(m, NCV, NU, lo=-.35, hi=.35)*r.bern(.5, m, NCV, NU).to(F))
    K = torch.where(pick[:, 0, None, None] & ismv & cv, p['K']*drift, p['K'])
    loss = torch.ones(m, NU, dtype=F, device=dev)
    loss[rows, r.randint(0, NMV-1, m)] = powr(1-alpha*U(m, lo=.2, hi=.55))
    K = torch.where(pick[:, 1, None, None], K*loss[:, None, :], K)
    q['K'] = torch.where(pick[:, 7, None, None] & ~ismv & rowh & (p['stock'] == 0)[:, :, None], K*powr(1+a3*U(m, NCV, NU, lo=-.3, hi=.3)), K)
    fac = powr(torch.exp(a1*U(m, NCV, lo=math.log(.6), hi=math.log(1.7))))[:, :, None]
    for key in ('p1r', 'p1i', 'p2r', 'p2i'):
        q[key] = torch.where(pick[:, 2, None, None] & cv, p[key]/fac, p[key])
    q['beta'] = torch.where(pick[:, 2, None, None] & cv, p['beta']*fac, p['beta'])
    nd = (p['dead']*powr(1+a2*U(m, NCV, NU, lo=-.3, hi=.5))+a2*r.randint(-2, 3, m, NCV, NU).to(F)).round().clamp(0, DMAX)
    q['dead'] = torch.where(pick[:, 3, None, None] & (act > 0) & cv, nd, p['dead'])
    q['gam'] = torch.where(pick[:, 4, None] & ~lin[:, None] & mvh, (p['gam']*powr(torch.exp(a1*U(m, NMV, lo=-.22, hi=.22)))).clamp(.6, 1.7), p['gam'])
    q['wa'] = torch.where(pick[:, 5, None] & ~lin[:, None] & cv[:, :, 0], (p['wa']+pw*a1*U(m, NCV, lo=-.25, hi=.25)).clamp(-.6, .6), p['wa'])
    one = torch.nn.functional.one_hot(r.randint(0, NMV-1, m), NMV).to(F)*p['mvm']*(pick[:, 6] & ~lin)[:, None].to(F)
    fs = torch.minimum(pw*a1*U(m, NMV, lo=.002, hi=.02), p['fcap'])
    q['fs'] = torch.where(one > 0, torch.maximum(p['fs'], fs), p['fs'])
    q['fd'] = torch.where(one > 0, torch.minimum(torch.maximum(p['fd'], fs*U(m, NMV, lo=.3, hi=1.)), q['fs']), p['fd'])
    q['tlag'] = torch.where(pick[:, 7, None] & rowh[:, :, 0], (p['tlag']*powr(torch.exp(a1*U(m, NCV, lo=math.log(.6), hi=math.log(1.7))))).clamp(.5, 30.), p['tlag'])
    if hv is not None:
        if 'gain' not in menu.drop:
            q['K'] = torch.cat((q['K'][:, :, :NMV]+turn_gains(r, p, first, at, alpha, hv, cv & rowh & (p['prc'] > 0)[:, :, None]), q['K'][:, :, NMV:]), -1)
        if 'time' not in menu.drop:
            fac = torch.exp(a1*U(m, NCV, lo=-math.log(HEAVY_TIME), hi=math.log(HEAVY_TIME)))[:, :, None]
            for key in ('p1r', 'p1i', 'p2r', 'p2i'):
                q[key] = torch.where(hv[:, None, None] & cv, first[key]/fac, q[key])
            q['beta'] = torch.where(hv[:, None, None] & cv, first['beta']*fac, q['beta'])
    if base is not None:
        back, what = r.bern(RESTORE_P, m), r.bern(.5, m, len(RESTORED))
        if hv is not None:
            back = back & ~hv
        what[rows, r.randint(0, len(RESTORED)-1, m)] = True
        for j, keys in enumerate(RESTORED):
            for key in keys:
                v = torch.where((what[:, j]).view(-1, *([1]*(p[key].dim()-1))), base[key], p[key])
                q[key] = torch.where(back.view(-1, *([1]*(p[key].dim()-1))), v, q[key])
    return derive(q)


def draw_move(r, p, info, tgt, menu=None, turn=None):
    """A setpoint move from the steady point `info`, operable by construction: half the time towards
    another operating point of the plant (some or all valves somewhere else, levels at a new height: the
    feasible directions of an ill-conditioned plant), half the time steps of 2-25 bands on all or a few
    tracked CVs. A resting value moves with a quarter of them (an operator sets it and leaves it). Returns
    the new targets and the largest tracked change in bands."""
    NCV, NMV, ND, NQ, NDQ, NU = sizes(p)
    m, dev = tgt.shape[0], tgt.device
    trk, ig = p['trk'] > 0, p['integ'] > 0
    vt, _ = hinv(tgt, p['wa'])
    vt = torch.where(trk, vt, torch.zeros_like(vt))
    hp, b = hprime(vt, p['wa']), p['bandf']
    zero = torch.zeros_like(vt)
    rst = p['rst'] > 0
    stay = (r.bern(.4, m)[:, None] & r.bern(.5, m, NMV)) | ((p['spr'] > 0) & r.bern(.75, m, NMV))
    away = torch.zeros_like(trk)
    if 'unit' in p:                                              # several units: most moves are one unit's, the others hold their targets
        loc, u = r.bern(LOCAL_P, m)[:, None], (r.one_of(trk).to(F)*p['unit']).sum(1, keepdim=True)
        away, stay = loc & (p['unit'] != u), stay | (loc & (p['unitc'][:, :NMV] != u))
    d0 = torch.where(stay, torch.zeros_like(p['mvm']), r.u(m, NMV, lo=.1, hi=.9)**p['gam']-info['ue'])*p['mvm']
    dA = (p['K'][:, :, :NMV]*d0[:, None, :]).sum(-1)
    lev = hinv(r.u(m, NCV, lo=.15, hi=.85), p['wa'])[0]-vt
    dA = torch.where(ig, torch.where(r.bern(.5, m, NCV), lev, zero), dA)
    size = 1. if menu is None else float(menu.move)
    step = size*torch.where(p['stock'] > 0, p['sband'], b)*torch.where(rst, r.u(m, NCV, lo=1., hi=4.), r.u(m, NCV, lo=2., hi=25.))*r.sign(m, NCV)/hp     # (a buffer's level: steps as an instrument's, not as wide as its band)
    few = ((r.bern(.4, m)[:, None] & r.bern(.6, m, NCV)) | (rst & r.bern(.75, m, NCV))) & ~r.one_of(trk & ~away)
    dB = torch.where(few, zero, step)
    isA = r.bern(.5, m)
    if turn is not None:                                         # one tracked CV alone: the turn-th of them, counted round
        can = trk & ~rst & ~away
        nth = can.long().cumsum(1)-1
        alone = r.bern(float(menu.turn), m) if float(menu.turn) < 1. else torch.ones_like(isA)
        dB = torch.where(alone[:, None], torch.where(can & (nth == (turn % can.sum(1).clamp_min(1))[:, None]), step, zero), dB)
        isA = isA & ~alone
    mv_size = size
    dvt = torch.where(isA[:, None], dA, dB)*p['trk']*(~away)
    s = reach(p, info, tgt, dvt, torch.zeros(m, NDQ, dtype=F, device=dev))
    size = (dvt.abs()*hp/b).amax(-1)
    s = torch.where(isA, torch.minimum(s*r.u(m, lo=.3, hi=1.), mv_size*r.logu(m, lo=6., hi=40.)/size.clamp_min(1e-9)), s).clamp_max(1.)*.98
    return torch.where(trk, h(vt+s[:, None]*dvt, p['wa']), zero), s*size


def draw_load(r, p, info, tgt, measured, menu=None):
    """A step of one or all measured disturbances (or unmeasured loads) to new levels, cut so that the
    valves can still hold the targets and so that its open-loop effect on the worst tracked CV is 3-25 bands
    (a level: over 100 decisions). Returns the new levels and that effect in bands."""
    NCV, NMV, ND, NQ, NDQ, NU = sizes(p)
    m, dev = tgt.shape[0], tgt.device
    isd = (torch.arange(NDQ, device=dev) < ND)[None]
    typ = torch.where(measured[:, None], isd, ~isd) & (p['dact'] > 0)
    sel = torch.where(r.bern(.6, m)[:, None], r.one_of(typ), typ)
    if menu is not None and menu.joint > 0:                       # every load at once, measured and unmeasured
        sel = torch.where(r.bern(menu.joint, m)[:, None], p['dact'] > 0, sel)
    dd = torch.where(sel, r.u(m, NDQ, lo=-1., hi=1.)-p['dlev'], torch.zeros_like(p['dlev']))
    s = reach(p, info, tgt, torch.zeros_like(tgt), dd)
    kd = (p['K'][:, :, NMV:]*dd[:, None, :]).sum(-1)
    rate = torch.where(p['integ'] > 0, 100/p['tint'], torch.ones_like(kd))
    eff = (kd.abs()*rate*hprime(info['v'], p['wa'])/p['bandf']*p['trk']).amax(-1)
    s = torch.minimum(s*.98, (1. if menu is None else float(menu.step))*r.u(m, lo=3., hi=25.)/eff.clamp_min(1e-9)).clamp_max(1.)
    return p['dlev']+s[:, None]*dd, s*eff


def event_kinds(r, p, spread=.8, menu=None):
    """Per-episode event mix (a regime: mostly regulation, mostly transitions, mostly process changes ...):
    EVENT_P with log-normal weights; kinds a plant cannot have are folded into setpoint events. With a menu:
    each kind times its rate (not normalised: the row's sum is how much faster the events come)."""
    NCV, NMV, ND, NQ, NDQ, NU = sizes(p)
    m, dev = p['K'].shape[0], p['K'].device
    q = torch.tensor(EVENT_P, dtype=F, device=dev)[None]*torch.exp(spread*r.normal(m, len(EVENT_P)))
    q = q/q.sum(-1, keepdim=True)
    has = torch.ones(m, len(EVENT_P), dtype=F, device=dev)
    has[:, LOAD] = (p['dact'][:, ND:].sum(-1) > 0).to(F)
    has[:, DV] = (p['dact'][:, :ND].sum(-1) > 0).to(F)
    has[:, LIMIT] = (p['zon'].sum(-1) > 0).to(F)
    has[:, PRIO] = ((p['trk']*(1-p['rst'])).sum(-1) > 1).to(F)
    lost = (q*(1-has)).sum(-1)
    q = q*has
    q[:, SP] += lost
    if menu is not None:
        w = torch.ones(len(EVENT_P), dtype=F, device=dev)
        w[SP], w[DV], w[LOAD], w[PROC], w[BOTH] = menu.rate[0], menu.rate[1], menu.rate[2], menu.rate[3], min(menu.rate[0], menu.rate[3])
        w[LIMIT] = w[PRIO] = menu.rate[4]
        for name in menu.drop:
            if name in EVENTS[:len(EVENT_P)]:
                w[EVENTS.index(name)] = 0.
        q = q*w[None]
    return q


def draw_event(r, p, info, tgt, kind, alpha, base=None, menu=None, turn=None, first=None, shift=None):
    """One candidate event of the given kind per plant (process changes scaled by alpha): the plant and the
    targets after it, and whether it is large enough to count. Operability is checked by the caller.
    base: what maintenance puts back, on a long stay; first, shift: change_process's."""
    NCV, NMV, ND, NQ, NDQ, NU = sizes(p)
    m = tgt.shape[0]
    k1 = kind[:, None]
    is_pr = (kind == PROC) | (kind == BOTH)
    pc = change_process(r, p, alpha, base, menu, info, first, shift)
    q = {key: (torch.where(is_pr.view(-1, *([1]*(v.dim()-1))), pc[key], v) if pc[key] is not v else v) for key, v in p.items()}
    ok1, inf1 = operable(q, tgt)                                 # the changed plant has to hold the old targets
    if menu is not None and menu.proc > 1.:
        # a large change: where the old targets cannot be held the targets give way, half or all the way to
        # what the valves where they stand now give on the changed plant (levels keep theirs)
        dss = (q['K'][:, :, NMV:]*q['dlev'][:, None, :]).sum(-1)+q['bias']
        yh = h((q['K'][:, :, :NMV]*info['ue'][:, None, :]).sum(-1)+dss, q['wa'])
        free = (q['trk'] > 0) & (q['integ'] == 0) & (q['rst'] == 0)
        tgt0 = tgt
        for lam in (.5, 1.):
            t2 = torch.where(free, (tgt0+lam*(yh-tgt0)).clamp(.04, .96), tgt0)
            ok2, inf2 = operable(q, t2)
            take = is_pr & ~ok1 & ok2
            tgt = torch.where(take[:, None], t2, tgt)
            inf1 = {key: torch.where(take.view(-1, *([1]*(v.dim()-1))), inf2[key], v) for key, v in inf1.items()}
            ok1 = ok1 | take
    inf1 = {key: torch.where(ok1.view(-1, *([1]*(v.dim()-1))), v, info[key]) for key, v in inf1.items()}
    tn, size = draw_move(r, q, inf1, tgt, menu, turn)
    is_sp = (kind == SP) | (kind == BOTH)
    big = ok1 & (~is_sp | (size >= MOVE_MIN*(1. if menu is None else min(1., menu.move))))
    tn = torch.where(is_sp[:, None], tn, tgt)
    is_ld = (kind == LOAD) | (kind == DV)
    dl, eff = draw_load(r, q, inf1, tgt, kind == DV, menu)
    big = big & (~is_ld | (eff >= LOAD_MIN*(1. if menu is None else min(1., menu.step))))
    q['dlev'] = torch.where(is_ld[:, None], dl, q['dlev'])
    q['cvw'] = torch.where(k1 == PRIO, draw_priority(r, p['trk'], p['rst'], prob=1.), p['cvw'])
    # limit move: every limit goes to 1.5-40 bands from the steady value
    y, b = inf1['y'], q['bandf']
    lo = y-b*r.logu(m, NCV, lo=1.5, hi=40.)
    hi = y+b*r.logu(m, NCV, lo=1.5, hi=40.)
    here = k1 == LIMIT
    if 'unit' in p:                                              # several units: most limit moves are one unit's (an operator moves the limits of the unit in front of them)
        u = (r.one_of(p['zon'] > 0).to(F)*p['unit']).sum(1, keepdim=True)
        here = here & (~r.bern(LOCAL_P, m)[:, None] | (p['unit'] == u))
    q['lo'] = torch.where(here & torch.isfinite(p['lo']), lo, p['lo'])
    q['hi'] = torch.where(here & torch.isfinite(p['hi']), hi, p['hi'])
    return q, tn, big


# ---------------------------------------------------------------- one episode
def _paired(r, p, info, ok):
    """Published pairings have positive relative gains, as an engineer pairs (3 % are left as drawn)."""
    NCV, NMV, ND, NQ, NDQ, NU = sizes(p)
    eye = torch.eye(NMV, dtype=F, device=r.dev)
    lam = rga(p, torch.where(ok[:, None, None], info['G'], eye))
    return (lam >= RGA_MIN).all(-1) | r.bern(.03, lam.shape[0])


def draw_start(r, n, rounds=60, cand=4, dims=DIMS, shape=None):
    """n operable plants at their commissioning points with their targets and steady points. A plant that
    fails is redrawn with the same roots (rejection, never pruning, and the roots keep their distribution)."""
    p, tgt, ro = draw_plant(r, n, dims=dims, shape=shape)
    ok, info = operable(p, tgt)
    ok = ok & _paired(r, p, info, ok)
    for _ in range(rounds):
        bad = (~ok).nonzero().flatten()
        if bad.numel() == 0:
            break
        nb, rb = bad.numel(), r.sub(bad, repeat=cand)
        pn, tn, _ = draw_plant(rb, nb*cand, {q: v[bad].repeat_interleave(cand) for q, v in ro.items()}, dims)
        okc, infc = operable(pn, tn)
        okc = (okc & _paired(rb, pn, infc, okc)).view(nb, cand)
        pick = torch.arange(nb, device=r.dev)*cand+okc.to(F).argmax(1)
        for key in p:
            p[key][bad] = pn[key][pick]
        tgt[bad] = tn[pick]
        for key in info:
            info[key][bad] = infc[key][pick]
        ok[bad] = okc.any(1)
    if not bool(ok.all()):
        raise RuntimeError(f'{int((~ok).sum())} of {n} plants found no operable start')
    return p, tgt, info


def settling(p):
    """(m,) decisions a unit takes to settle after a move: twice its time constant and dead time."""
    return (2*p['T']*(1+p['rth'])).clamp(30., 200.)


def up(x):
    """x rounded up to whole decisions; a whole number plus rounding noise stays that number (a valve travel
    of zero comes out as 1e-13 on one machine and 0 on another)."""
    return (x-SNAP).ceil()


def down(x):
    """x rounded down to whole decisions, the same way."""
    return (x+SNAP).floor()


def lineout(p, prev, info, ramp):
    """(m,) decisions after an event before the unit can have lined out at the new steady point: the
    event's own ramp, the valve travel at full slew (times TRAVEL) and SETTLE settling times."""
    travel = ((info['un']-prev['un']).abs()/p['slew']*p['mvm']).amax(-1)
    return ramp+TRAVEL*travel+SETTLE*settling(p)


def in_service(ulo, uhi):
    """An MV whose limits leave it room to move (not pinned where it stands, `mv_out`)."""
    return (uhi-ulo) > 2.5*PIN


def pacing(mix, menu):
    """(mix, speed, none) of a menu's event mix: the mix normalised, how much faster than usual the events come,
    and the episodes that have none."""
    dens = mix.sum(-1)
    none = dens <= 0
    mix = torch.where(none[:, None], torch.nn.functional.one_hot(torch.tensor(SP), mix.shape[1]).to(mix)[None], mix)
    return mix, (menu.pace*dens).clamp_min(1e-9), none | (menu.pace <= 0)


def draw_episode(r, n, exc_p=EXC_P, dims=DIMS, tail=None, keep=None, menu=None, nseg=NSEG, shape=None):
    """Start plants, targets and the event schedule of n episodes: lists over segments of the plant dicts,
    targets and steady points, the start decision of each segment (BIG when unused), the event kind that
    opened it (-1: none), the teacher's renewal delay, and over how many decisions the plant and the
    disturbance levels move to their new values (0: at once). An event that takes the unit to a
    constrained point (a share exc_p of them tries to) is followed by one that brings it back to where it
    was.
    tail: the next window on the same plants (what `keep` took from the window before). Its first three
    segments are carried: 0 the plant as commissioned, 1 and 2 the last two segments the window before had
    in use, at their own decisions counted from this window's start (negative), so a change still on its way
    at the boundary goes on as it was, the teacher's model follows it when it would have, and the unit comes
    back from a constrained point it was left at. The events go on at the unit's own pace: up to 9 more.
    keep: a dict that takes this window's tail. menu: what changes in this window (Menu); nseg: segments a
    window can have (a menu with events that come faster needs more than NSEG); shape: the plants' roots kept
    to a range (draw_roots; SMALL: two or three valves, none to spare).
    Menu.shift: a unit chosen for it has, at its first ordinary event of the window, a heavy change that stays:
    the event is a process event of that kind alone (never taken back, never replaced by an excursion), and
    the tail's `mend` (what maintenance puts back) takes the gains and time constants it made."""
    dev = r.dev
    seg_k = torch.full((n, nseg), BIG, dtype=torch.long, device=dev)
    kinds = torch.full((n, nseg), -1, dtype=torch.long, device=dev)
    delay = torch.zeros(n, nseg, dtype=torch.long, device=dev)
    ramp = torch.zeros(n, nseg, dtype=torch.long, device=dev)
    big = tuple(dims) != DIMS
    if menu is not None and menu.exc is not None:
        exc_p = menu.exc
    elif big and exc_p == EXC_P:                                 # a larger size, where nothing else is asked: it runs where limits stop it
        exc_p = APC_EXC
    lz = torch.zeros(n, dtype=torch.long, device=dev)
    free = None
    if tail is None:
        p, tgt, info = draw_start(r, n, dims=dims, shape=shape)
        hid = p.pop('hid', None)
        if hid is not None and bool(hid.any()):                  # MVs that are not the controller's (THIN): held where they stand from here on
            free, p = (p, tgt, info), dict(p)
            p['ulo'], p['uhi'] = torch.where(hid > 0, info['un'], p['ulo']), torch.where(hid > 0, info['un'], p['uhi'])
        segs, tgts, infos = [p], [tgt], [info]
        seg_k[:, 0] = 0
        before, before_m, last_m = (p, tgt, info), (lz-1, lz-1, lz, lz), (lz, lz-1, lz, lz)
    else:
        r.bind(n)
        base, before, (p, tgt, info) = tail['base'], tail['before'], tail['cur']
        segs, tgts, infos = [base[0], before[0], p], [base[1], before[1], tgt], [base[2], before[2], info]
        before_m, last_m = tuple(x.clone() for x in tail['before_m']), tuple(x.clone() for x in tail['last_m'])
        before_m[0].sub_(HORIZON); last_m[0].sub_(HORIZON)
        for j, m in ((1, before_m), (2, last_m)):
            seg_k[:, j], kinds[:, j], delay[:, j], ramp[:, j] = m
        seg_k[:, 0] = seg_k[:, 1]-1
    mix = event_kinds(r, p, menu=menu)
    if menu is not None:
        mix, speed, none = pacing(mix, menu)
    ts = settling(p)                                             # how long this unit takes to settle
    glo, ghi = ts.clamp(40., 150.), (2.5*ts).clamp(80., 300.)
    thin = held(p).any(-1)                                       # plants with more setpoints than MVs: every event is drawn on the plant with those MVs free (`home`), then they are held again
    anythin = bool(thin.any())                                   # (as Menu.ride does at a constrained point; here for good: such a plant has no other home)
    if tail is None:
        k = r.randint(FIRST_EVENT[0], FIRST_EVENT[1], n)
        away = torch.zeros(n, dtype=torch.bool, device=dev)      # the unit is at a constrained point
        home = (p, tgt, info)                                    # where it comes back to
        nsp = lz.clone()                                         # setpoint moves so far (whose turn it is)
        riding, cause = torch.zeros(n, dtype=torch.bool, device=dev), None
        if free is not None:
            home, away, riding, cause = free, thin.clone(), thin.clone(), cause_of(free[0], tgt, p, tgt)
    else:
        k, away, home, nsp = (tail['k']-HORIZON).clamp_min(1), tail['away'], tail['home'], tail['turn']
        riding, cause = tail.get('riding'), tail.get('cause')
        riding = torch.zeros(n, dtype=torch.bool, device=dev) if riding is None or cause is None else riding
    ride = float(menu.ride) if menu is not None and menu.ride is not None else (APC_RIDE if big else 0.)     # a unit at a constrained point stays there (Menu.ride)
    if ride > 0 and cause is None:
        cause = cause_of(p, tgt, p, tgt)
    mend = segs[0] if tail is None else tail.get('mend') or segs[0]     # what maintenance puts back: the plant as commissioned, or what a heavy change that stayed made of it
    todo = r.bern(float(menu.shift), n) if menu is not None and menu.shift > 0 else None    # units that have such a change in this window (Menu.shift), at their first ordinary event
    heavy = menu is not None and (menu.heavy > 0 or menu.shift > 0)
    pick = lambda c, x, y: torch.where(c.view(-1, *([1]*(x.dim()-1))), x, y)          # noqa: E731
    picks = lambda c, xs, ys: tuple({key: pick(c, x[key], y[key]) for key in x} if isinstance(x, dict) else pick(c, x, y) for x, y in zip(xs, ys))   # noqa: E731
    for e in range(len(segs), nseg):
        exists = k < HORIZON-30
        if menu is not None:
            exists = exists & ~none
        kind = r.cat_rows(mix)
        ok = torch.zeros(n, dtype=torch.bool, device=dev)
        hit = torch.zeros(n, dtype=torch.bool, device=dev)      # this event is the heavy change that stays
        if todo is not None:
            hit = todo & exists & ~away
            kind = torch.where(hit, torch.full_like(kind, PROC), kind)
        pn, tn, inf_n = p, tgt, info
        on = ok                                                  # at a constrained point to stay: this event is an ordinary one too
        src, hn = (p, tgt, info), home
        if ride > 0:
            on = away & riding & exists & r.bern(ride, n)
        if anythin:
            on = on | (thin & exists)
        if ride > 0 or anythin:
            src = picks(on, home, src)                           # drawn on the unit as it would be without what holds it there
        for a in ALPHA:
            need = exists & ~ok & (~away | on)
            if not bool(need.any()):
                break
            with r.only(need):                                    # the streams of the others stand still
                pc, tc, big = draw_event(r, src[0], src[2], src[1], kind, torch.full((n,), a, dtype=F, device=dev), None if tail is None else mend,
                                         menu, nsp if menu is not None and menu.turn else None, segs[0] if heavy else None, hit if todo is not None else None)
            okc, infc = operable(pc, tc)
            take = need & okc & big
            if bool((take & on).any()):                           # what holds the unit put on again: an operable point (it no longer binds) or a verified constrained one
                idx = (take & on).nonzero().flatten()
                pc['rtgt'] = torch.where(pc['trk'] > 0, tc, pc['rtgt'])
                qa, ta = with_cause({key: v[idx] for key, v in pc.items()}, tc[idx], {key: v[idx] for key, v in cause.items()})
                oka, infa, qa['rtgt'], qa['zal'] = point(qa, ta)
                oka = oka & (qa['dlev'].abs() <= 1.).all(-1)      # (a load that holds the unit there stays inside its range)
                take[idx[~oka]] = False
                good, sel = idx[oka], take & on

                def put(x, y):                                    # y's valid rows into a copy of x
                    x = x.clone(); x[good] = y[oka]
                    return x
                hn = ({key: pick(sel, pc[key], hn[0][key]) for key in pc}, pick(sel, tc, hn[1]), {key: pick(sel, infc[key], hn[2][key]) for key in infc})
                pc, tc, infc = {key: put(pc[key], qa[key]) for key in pc}, put(tc, ta), {key: put(infc[key], infa[key]) for key in infc}
            pn = {key: pick(take, pc[key], pn[key]) for key in pn}
            tn = pick(take, tc, tn)
            inf_n = {key: pick(take, infc[key], inf_n[key]) for key in inf_n}
            ok = ok | take
        rode = on & ok
        if todo is not None:                                     # the unit is another one from here on
            hit = hit & ok
            todo = todo & ~hit
            mend = {**mend, **{key: pick(hit, pn[key], mend[key]) for key in LASTING}}
        pn['rtgt'] = torch.where((ok & ~rode)[:, None] & (pn['trk'] > 0), tn, pn['rtgt'])
        nsp = nsp+(ok & ((kind == SP) | (kind == BOTH))).long()
        out = torch.zeros(n, dtype=torch.bool, device=dev)
        gone = out                                               # a disturbance or process event the next event takes back
        if menu is not None and menu.pulse > 0:
            gone = ok & ((kind == LOAD) | (kind == DV) | (kind == PROC)) & r.bern(menu.pulse, n) & ~rode & ~hit
        idx = (exists & ~away & ~gone & ~hit & r.bern(exc_p, n)).nonzero().flatten() if e < nseg-1 else out.nonzero().flatten()   # a slot is left to come back in
        for _ in range(EXC_TRIES):
            if idx.numel() == 0:
                break
            px, tx, infx, kx, okx = draw_excursion(r.sub(idx), {key: v[idx] for key, v in p.items()}, {key: v[idx] for key, v in info.items()}, tgt[idx])
            got = idx[okx]

            def put(x, y):                                        # y's valid rows into a copy of x
                x = x.clone(); x[got] = y[okx]
                return x
            pn = {key: put(pn[key], px[key]) for key in pn}
            tn, kind = put(tn, tx), put(kind, kx)
            inf_n = {key: put(inf_n[key], infx[key]) for key in inf_n}
            out[got] = True
            idx = idx[~okx]
        ok = ok | out
        home = picks(out | gone, (p, tgt, info), home)
        if ride > 0:
            if bool(out.any()):                                   # what took the unit there, to hold while it stays
                new = cause_of(p, tgt, pn, tn)
                cause = {key: pick(out, new[key], cause[key]) for key in cause}
        if ride > 0 or anythin:
            home = picks(rode, hn, home)                          # (the unit without it has had this event too)
        riding = (riding & ~exists) | ((out | rode) if ride > 0 else out & False)
        before, before_m = picks(exists, (p, tgt, info), before), picks(exists, last_m, before_m)   # (the last segment in use so far is no longer the last)
        back = away & exists & ~rode
        if anythin:                                              # (a plant with MVs that are not the controller's stays as it is when no event is found for it)
            riding, back = riding | thin, back & ~thin
        pn = {key: pick(back, home[0][key], pn[key]) for key in pn}
        tn = pick(back, home[1], tn)
        inf_n = {key: pick(back, home[2][key], inf_n[key]) for key in inf_n}
        kind, ok = torch.where(back, torch.full_like(kind, RETURN), kind), ok | back
        away = (away & ~exists) | out | gone | rode
        if anythin:
            away = away | thin
        p, tgt, info = pn, tn, inf_n
        segs.append(p); tgts.append(tgt); infos.append(info)
        seg_k[:, e] = torch.where(exists, k, torch.full_like(k, BIG))
        kinds[:, e] = torch.where(ok, kind, torch.full_like(kind, -1))
        delay[:, e] = r.randint(RENEW[0], RENEW[1], n)
        gap = (glo+(ghi-glo)*r.u(n)).floor().long()
        if menu is not None:
            gap = down(gap/speed).long().clamp_min(MIN_GAP)
        ramp[:, e] = torch.where(r.bern(RAMP_P, n), torch.minimum(r.logu(n, lo=5., hi=120.), .8*gap).long(), torch.zeros_like(gap))
        travel = ((info['un']-infos[-2]['un']).abs()/p['slew']*p['mvm']).amax(-1)      # decisions the valves need at full slew
        gap = torch.maximum(gap, down(TRAVEL*travel+.5*glo/(1. if menu is None else speed)).long())     # an event is not cut short by the next one (rule 3)
        stay = down(lineout(p, infos[-2], info, ramp[:, e])+STAY*ts).long()           # a constrained point is reached and held
        gap = torch.where(out | (rode & constrained(p, tgt)), torch.maximum(gap, stay), gap)
        last_m = picks(exists, (k, kinds[:, e], delay[:, e], ramp[:, e]), last_m)
        k = torch.where(exists, k+gap, k)                           # (an event past the window: the next window's first)
    if keep is not None:
        keep.update(base=(segs[0], tgts[0], infos[0]), before=before, before_m=before_m, last_m=last_m, cur=(p, tgt, info), home=home, away=away, k=k, turn=nsp,
                    riding=riding, cause=cause, mend=mend)
    return segs, tgts, infos, seg_k, kinds, delay, ramp


def draw_field(r, p, info):
    """What sits around the process for one episode: how the disturbances keep moving between events and
    what the instruments do to the measurements (docs/sim.md, 'field layer'). Per-episode constants.
    The wander of every disturbance is cut to what the valves can hold the targets against inside their
    margin (WANDER of span at 1 sigma, at the start point `info`), as the steps are by `reach`: an
    ill-conditioned unit gets small wander in its weak direction, not a floor no controller can remove."""
    NCV, NMV, ND, NQ, NDQ, NU = sizes(p)
    m, dev = p['K'].shape[0], p['K'].device
    U, LU = r.u, r.logu
    z = lambda *s: torch.zeros(m, *s, dtype=F, device=dev)       # noqa: E731
    inten = torch.where(r.bern(.25, m), z(), U(m))               # a quarter of the episodes are quiet between events
    c = {}
    # input disturbances: Ornstein-Uhlenbeck around the scheduled level, sometimes a sinusoid on one load
    a = torch.exp(-1/LU(m, NDQ, lo=20., hi=600.))
    A, t5 = _system(p)
    du, bad = torch.linalg.solve_ex(A, torch.where(t5[:, :, None], p['K'][:, :NMV, NMV:], torch.zeros_like(p['K'][:, :NMV, NMV:])))
    travel = (du.abs()/(p['gam']*info['un'].clamp_min(1e-6)**(p['gam']-1)).clamp_min(1e-6)[:, :, None]*p['mvm'][:, :, None]).amax(1)
    cap = torch.where(bad[:, None] > 0, torch.zeros_like(travel), WANDER/travel.clamp_min(1e-9))    # (m, NDQ) level change per WANDER of valve
    sig = torch.minimum(inten[:, None]*U(m, NDQ, hi=.25), cap)*p['dact']
    c['ou_a'], c['ou_s'] = a, sig*torch.sqrt(1-a*a)
    one = torch.nn.functional.one_hot(ND+r.randint(0, NQ-1, m), NDQ).to(F)*p['dact']
    c['sin_amp'] = one*(r.bern(.2, m) & (inten > 0))[:, None].to(F)*torch.minimum(U(m, 1, lo=.05, hi=.3), 1.5*cap)
    c['sin_w'] = 2*math.pi/LU(m, 1, lo=20., hi=400.).expand(-1, NDQ)
    c['sin_ph'] = U(m, 1, hi=2*math.pi).expand(-1, NDQ)
    # output drift on the process rows (value units; a band is bandf)
    a = torch.exp(-1/LU(m, NCV, lo=60., hi=600.))
    sig = inten[:, None]*U(m, NCV, lo=.1, hi=.6)*p['sband']*p['prc']*r.bern(.8, m, NCV).to(F)
    c['od_a'], c['od_s'], c['od_l'] = a, sig*torch.sqrt(1-a*a), torch.exp(-1/LU(m, NCV, lo=2., hi=20.))
    # instruments
    real = r.bern(.6, m)[:, None] & (p['prc'] > 0)
    c['nw'] = torch.where(r.bern(.5, m)[:, None], U(m, 1, hi=.3)*p['sband'], z(NCV))*p['prc']
    a = U(m, NCV, lo=.9, hi=.99)
    on = real & r.bern(.3, m, NCV)
    c['cn_a'], c['cn_s'] = a, torch.where(on, U(m, NCV, lo=.05, hi=.3)*p['sband']*torch.sqrt(1-a*a), z(NCV))
    c['as'] = torch.where(real & r.bern(.3, m, NCV), torch.exp(-1/LU(m, NCV, lo=.5, hi=4.)), z(NCV))
    c['quant'] = torch.where(real & r.bern(.15, m, NCV), U(m, NCV, lo=.05, hi=.25)*p['sband'], z(NCV))
    c['an_T'] = torch.where(real & r.bern(.15, m, NCV) & (p['rst'] == 0) & (p['stock'] == 0), r.randint(3, 20, m, NCV).to(F), z(NCV))
    c['an_ph'] = r.randint(0, 10**6, m, NCV).to(F)
    c['spike'] = real.to(F)*1e-3
    c['bad'] = real.to(F)*5e-4
    return c

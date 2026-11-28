"""Training of the AIO network on the simulation core (docs/sim.md, 'Training'): the second trainer (11-10).

What is learned from is the simulator's own gradient. The first trainer cloned a teacher and added that gradient
as a small part; it stopped at 0.91 of the formula PI, a tenth of the way to a controller that knows the plant.
Here the plants run in closed loop with the graph kept and the score, made smooth, is differentiated through
plant and network back to every move (Xu et al. 2022 SHAC; Georgiev et al. 2024 AHAC; Xing et al. 2025 SAPO):

  window    --window decisions of a batch of episodes at a time (a lane; --lanes of them take turns, each at
            another point of its episodes). What is minimised is what the Oracle minimises
            (aio.sim.oracle.objective: the rows outside, softened, plus a small share of the Huber cost), per
            episode relative to its own level (plus --floor), so an upset of 100 bands weighs as one of 3.
  critic    Beyond the window: Q(history, move) of aio.net.Critic, its own network, fitted to TD(lambda)
            targets on the windows just run (two heads; the targets take the more pessimistic). It is in the
            graph at every decision, so a move's effect past the window's end is charged to it. The actor's
            gradient reaches it through the plant alone: the critic's weights are held and the move it is given
            is detached. Most episodes tell it the true responses (--critic-known), the rest do not, so it is
            kept for learning on from a real unit's data (packet history and moves are all it needs).
  guard     On a few plants the gradient grows along the window (measured 11-10: 1.35x a decision, the batch
            gradient 1e5 from one or two episodes). Where the plant's gradient enters the network, at every
            move, an episode's gradient is held to --cap times what is usual: the median over the episodes of
            their largest gradient in a window, carried as a running mean over the windows (per decision the
            batch has no usual level: the last moves of a window have next to no effect inside it, and on
            most plants none); one that is not finite is dropped. An episode that was held is then given
            shorter stretches, --guard cut: its graph is cut every window/2, /4 .. --window-min decisions,
            with the critic's value at each cut; after --relax windows without being held it gets one halving
            back. --guard open: such an episode keeps the full window but the policy's reaction to the packet
            is left out of its gradient (the open-loop gradient of Song et al. 2025). --guard cap: the hold
            alone.
  explore   In training every move gets noise of a fixed spread (--noise, before the tanh; the plant's hard
            limits and sticking valves pass a gradient only where the moves are spread over them). The
            spread is not learned and there is no entropy term: the first version steered the mean entropy
            of the network's own spread head to a target, and from SAC's weights it ran away within 15 rows
            (11-11: the head is clamped at both ends and passes no gradient there, so the MVs at the lower
            end kept the mean entropy low while the weight pushed the others to the upper end; mean spread
            0.7 for a target of 0.05, validation and literature columns collapsed). Validation is
            deterministic.
  teacher   Warm start only: for the first --bc-iters rows the teacher's move at the visited states is
            imitated, fading out as the simulator gradient fades in. None with --init or --resume.
  identify  The pair pathway (aio.net) is read out as every pair's step response and trained on the
            simulator's labels (--id-coef; `id_own`, `id_cross` in the log: published pairs and the others).
            A share --known of the episodes is told the true responses instead of its own estimates.
            The pair memories also read running least squares of every row's changes on every MV's moves
            (aio.net.Stats: no weights; kept here once per decision for actor and critic alike).
            --no-stats: the network without them (PR #299's), for the comparison.
            --slow S: a second set of those sums that forgets over S decisions in place of 200 (aio.net,
            `slow`): for a unit the network has run for a while, its state carried on.
            --arx: every set of the sums is also fitted with the rows' own past beside the MVs' moves (aio.net,
            `arx`): what the loop feeds back of a row's noise is then not read as the MVs' effect.
            --dv (with --arx): those fits take the rows that read a measured disturbance beside the MVs too
            (aio.net, `dv`): what a measured disturbance does to a row is then not put down to the MVs.
            --tied K: the control half as K rounds of one set of weights with products in the messages
            (aio.net, `tied`; 0, the default: two rounds of their own weights).
  law       --law (with --replay; aio.law, aio.net `law`): the teacher's move is, but for 1.4 points of its
            score, the product of things a student shown only the move has to come by unaided: E, how far
            every tracked row will be from where it can settle if the MVs stop now, K, every MV's first
            change of flow per unit of it, and V, the valve curve. They are worked out at every decision
            gathered, kept with the window, and the network's readouts of them are trained towards them on
            every episode (--law-coef; rows: `head`, `gain`, `valve`, and `law`, their product against the
            law's move); the told episodes get the teacher's own in place of the readouts, as they get the
            true responses. The move imitated is the law's (held to the limits), not the teacher's.
            --meter: the teacher is the one that corrects its prediction from the instruments alone and
            does not see the unmeasured loads move (`set_teacher`): its move and its E are what a controller in
            the field can know.
            --design B: K is the design of a plant's model and has nothing of where the plant is, so plants
            that are drawn and not run teach it too: every step of learning also takes B of them from a pool
            the builders keep filling (`Designs`; rows: `design`, and `design_new` on plants never seen).
  memory    98% of a window's memory is the two networks' graphs of every decision (measured 11-10: actor
            0.42, critic 0.34 MB per episode and decision; the plant 2%). They are not kept: each decision's is
            rebuilt at the backward pass (torch's checkpoint; the same gradient, a window some 30% slower),
            and the critic is fitted --fit-chunk episodes at a time. Memory then grows with the episodes, not
            with episodes x decisions, so a batch can be several times larger. --no-lean keeps the graphs.

Selection: the deterministic actor on a fixed validation batch (--val-n episodes, --val-seed), read on the
scale of aio.scale: `val.pts`, where it stands from the MVs left where they are (0) to the best causal
controller per episode (100); `val.pts_known` the same when it is told the true responses (control alone,
identification taken out). The 100 is the envelope of the references run here (formula PI, step-test DMC,
teacher, re-identified teacher) and, with --refs FILE, of the Oracle and whatever else aio.sim.oracle wrote for
the same batch; without the Oracle the scale's 100 is too easy and run.json says so (`scale.oracle`).
`val.pts_before` and `val.pts_after` are the points on the two parts of every episode's scored window: up to its
first process change and from it on (`before_after`: what a controller is worth once the plant is not the one it started
on; written beside `pts` in every row and for every reference). The
formula PI is one of the references (`val_refs.pi.pts`), not the criterion; `x_pi` (the multiple of it) is
still written for the pages that read it, and `con`.

Staying on a unit (--stay L, 11-12): the plants of every other lane (lanes 0, 2, ..) are run for L episodes in
a row instead of one: when an episode ends the next one of the same units is drawn (aio.sim.core.Program,
`after`: the plant as the events left it, a schedule of events of its own), the plant's state is carried over
and so are the memories of actor and critic and the running sums; nothing is cleared. The other lanes take new
plants every episode, as before (half the decisions on units the network has run for a while, half on new ones).
With --mech the share of plants of the engine (aio.sim.mech: the fifteen of the default pool) is on every lane, those that keep their plants too.
Until then every unit was run for one episode from a cleared memory, and with --life for its first 256 decisions
alone, so nothing a network came to know about a unit could pay: 8 events seen once each. --val-stay L reads
what staying is worth (`Stay`): a fixed batch run on for L episodes; per episode the points with the memory
carried (`stay.pts`), with everything older than SHADOW decisions forgotten (`stay.fresh`), and the error of the
responses read (`stay.id_own`, `stay.id_cross`; `_near` and `_quiet`: within 200 decisions of the last event, and
in the quiet after it).
What changes on the training plants (--menu, 11-12): the fields of aio.sim.prior.Menu, how often each kind of
event comes and how large, which kinds are left out (`drop`: met in the exams alone), and what plants they are
(nd=1,2: one or two measured disturbances on every plant). --menu-stay: on the lanes that keep their plants (more
steps where the network has the time to learn a unit from them); --menu-plain with --plain K: on every K-th lane
(a share of the plants as the generator makes them, or with a menu of their own). The validation batches keep
the generator's own: the exam does not move with the training ground.

One row of metrics.jsonl is --row windows (default: one episode's worth).

  python -m aio.train --out RUN --hours 2 --refs reports/sim/oracle-val.json            # from scratch, GPU defaults
  python -m aio.train --out RUN --hours 2 --init OLD/last.pt --lr 1e-4 --warmup 10      # from earlier weights
  python -m aio.train --out RUN --hours 2 --resume                                      # continue RUN/last.pt
  python -m aio.train --out RUN --device cpu --envs 32 --lanes 2 --val-n 32 --hours .2
  python -m aio.train --bench --envs 256                                                # speed and memory
"""
import argparse, collections, contextlib, copy, faulthandler, functools, inspect, io, itertools, json, math, os, pathlib, random, signal, socket, subprocess, sys, threading, time, types
import urllib.error, urllib.request
from fractions import Fraction
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint

from aio import interface as IF
from aio.frame import canonical
from aio import net as N, scale as SC
from aio.net import symlog, symexp, pair_labels
from aio.sim import core as C, control as CT, oracle as OR, prior as PR, stay as ST
from aio.sim.prior import HORIZON



GUARD_UNITS = 3     # of `usual` in a checkpoint. 3: batch-size-invariant per-episode objective units (2: divided by n; 1: with imitation term)


def safe_torch_load(path, map_location='cpu'):
    """Cross-Python (3.13 pathlib._local -> <=3.12) and cross-OS (WindowsPath -> PosixPath) checkpoint loader."""
    fn = getattr(N, 'safe_torch_load', None)
    if fn is not None:
        return fn(path, map_location=map_location)
    sys.modules.setdefault('pathlib._local', pathlib)
    saved = pathlib.WindowsPath
    if os.name != 'nt':
        pathlib.WindowsPath = pathlib.PureWindowsPath
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    finally:
        pathlib.WindowsPath = saved


def sanitize_paths(o):
    """Recursively convert any Path / PathLike values to plain str so checkpoints never pickle pathlib._local or WindowsPath."""
    if isinstance(o, os.PathLike):
        return os.fspath(o)
    if isinstance(o, dict):
        return {k: sanitize_paths(v) for k, v in o.items()}
    if isinstance(o, list):
        return [sanitize_paths(v) for v in o]
    if isinstance(o, tuple):
        return tuple(sanitize_paths(v) for v in o)
    return o


def _scalar(x):
    return float(x.detach()) if torch.is_tensor(x) else float(x)


def _steps_caps(net):
    """Cached capability check for optional Track A SeqAIO.steps(need_read=..., lane_tag=...) kwargs."""
    caps = getattr(net, '_steps_caps_cache', None)
    if caps is None:
        try:
            params = inspect.signature(net.steps).parameters
            has_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            caps = (has_kw or 'need_read' in params, has_kw or 'lane_tag' in params)
        except (TypeError, ValueError):
            caps = (False, False)
        try:
            net._steps_caps_cache = caps
        except Exception:
            pass
    return caps


def _supports_event_seed():
    """Whether aio.sim.core.Program / build accepts optional `event_seed` (Track A anti-memorization split)."""
    try:
        bp = inspect.signature(C.build).parameters
        pp = inspect.signature(C.Program.__init__).parameters
        return ('event_seed' in bp) or ('event_seed' in pp)
    except (TypeError, ValueError):
        return False


def quiet_sleep_seconds(spec, tz_name='Asia/Shanghai', now=None):
    """Return seconds remaining in a quiet-hours window (`HH:MM-HH:MM`, e.g. `00:00-10:30`), or 0.0 if outside."""
    if not spec or '-' not in str(spec):
        return 0.0
    from datetime import datetime, timedelta, timezone
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone(timedelta(hours=8)) if 'Shanghai' in str(tz_name) or 'Beijing' in str(tz_name) else timezone.utc
    s_part, e_part = str(spec).strip().split('-', 1)
    sh, sm = (int(x) for x in s_part.strip().split(':'))
    eh, em = (int(x) for x in e_part.strip().split(':'))
    s_min, e_min = (sh % 24) * 60 + sm, (eh % 24) * 60 + em
    if s_min == e_min:
        return 0.0
    dt = datetime.fromtimestamp(time.time() if now is None else float(now), tz=tz)
    cur = dt.hour * 60 + dt.minute + dt.second / 60.0
    if s_min < e_min:
        return (e_min - cur) * 60.0 if s_min <= cur < e_min else 0.0
    if cur >= s_min:
        return ((1440.0 - cur) + e_min) * 60.0
    if cur < e_min:
        return (e_min - cur) * 60.0
    return 0.0


def _prefetch_noise(prog, k0, H, dev):
    """Defensively invoke Track A's optional `prog.prefetch_noise(...)` once per window -> list of H (z, u) pairs or None."""
    fn = getattr(prog, 'prefetch_noise', None)
    if fn is None:
        return None
    try:
        params = [p for p in inspect.signature(fn).parameters.values()
                  if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        names = [p.name for p in params]
        if names and names[0].startswith('k') and len(params) >= 3:
            res = fn(k0, H, dev)
        elif names and names[0].startswith('k') and len(params) == 2:
            res = fn(k0, H)
        elif len(params) >= 2:
            res = fn(H, dev)
        else:
            res = fn(H)
    except Exception:
        return None
    if res is None:
        return None
    if isinstance(res, tuple) and len(res) == 2 and torch.is_tensor(res[0]) and torch.is_tensor(res[1]) and res[0].shape[0] == H:
        return [(res[0][j], res[1][j]) for j in range(H)]
    if isinstance(res, (list, tuple)) and len(res) == H:
        return res
    return None


def _fused_kw(mod):
    """Enable CUDA fused Adam/AdamW when all trainable parameters are floating-point CUDA tensors."""
    ps = [p for p in mod.parameters() if p.requires_grad]
    if ps and all(p.is_cuda and p.is_floating_point() for p in ps):
        return dict(fused=True)
    return {}


def merge_alloc_conf(conf, env_conf=None):
    """Merge `--card-alloc` with `PYTORCH_CUDA_ALLOC_CONF` so env settings like `expandable_segments:True` are preserved."""
    env_str = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '') if env_conf is None else (env_conf or '')
    parts = {}
    for src in (env_str, conf or ''):
        for tok in src.split(','):
            tok = tok.strip()
            if not tok:
                continue
            if ':' in tok:
                k, v = tok.split(':', 1)
                parts[k.strip()] = v.strip()
            else:
                parts[tok] = None
    return ','.join(f'{k}:{v}' if v is not None else k for k, v in parts.items())


def _has_expandable_segments(alloc_conf=''):
    s = merge_alloc_conf(alloc_conf)
    return any(tok.strip().lower() == 'expandable_segments:true' for tok in s.split(','))


def maybe_empty_cache(dev, threshold=.85, expandable=None):
    """Gate `torch.cuda.empty_cache()` behind high-watermark VRAM pressure when `expandable_segments:True` is active."""
    if dev.type != 'cuda' or not torch.cuda.is_available():
        return False
    if expandable is None:
        expandable = _has_expandable_segments()
    idx = torch.cuda.current_device() if dev.index is None else dev.index
    _, total = torch.cuda.mem_get_info(idx)
    reserved = torch.cuda.memory_reserved(idx)
    if (not expandable) or (reserved > threshold*total):
        torch.cuda.empty_cache()
        return True
    return False


@contextlib.contextmanager
def on_device(mod, dev):
    """Temporarily move a module (e.g. CPU EMA weights) onto `dev` for validation, restoring its original device on exit."""
    p = next(mod.parameters(), None)
    was = None if p is None else p.device
    target = torch.device(dev)
    if was is not None and was != target:
        mod.to(target)
    try:
        yield mod
    finally:
        if was is not None and was != target:
            mod.to(was)


def atomic_save(obj, path):
    """Atomically save a PyTorch file via a sibling temp file + `os.replace` so SIGKILL/preemption never corrupts checkpoints."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f'.{dst.name}.{os.getpid()}.tmp')
    try:
        torch.save(obj, tmp)
        os.replace(tmp, dst)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def explicit_args(ap, argv):
    """Return the set of argparse `dest` names explicitly provided on the CLI."""
    opt_to_dest = {}
    for act in ap._actions:
        for opt in act.option_strings:
            opt_to_dest[opt] = act.dest
    seen = set()
    for tok in argv:
        if not tok.startswith('-'):
            continue
        opt = tok.split('=', 1)[0]
        if opt in opt_to_dest:
            seen.add(opt_to_dest[opt])
    return seen


def pack_eval_state(sd):
    """Pack actor state dict for lightweight eval snapshots (~268 MB): >=2D floats in BF16, <=1D floats in FP32."""
    out = {}
    for k, v in sd.items():
        t = v.detach().cpu()
        if t.is_floating_point():
            out[k] = t.to(torch.bfloat16) if t.ndim >= 2 else t.to(torch.float32)
        else:
            out[k] = t
    return out


def scheduled_lr(a, c, hours_now, hub_lr=None):
    """Compute base actor LR from Hub override (if active) or cosine schedule indexed by `hours`, `updates`, or `samples`."""
    if hub_lr is not None:
        return float(hub_lr)
    warm = min(1., (c['row']+1)/max(getattr(a, 'warmup', 1), 1))
    basis = getattr(a, 'cosine_by', 'hours')
    if basis == 'updates':
        tot = getattr(a, 'total_updates', 0)
        fac = lr_factor(c['updates'], tot) if tot > 0 else 1.
    elif basis == 'samples':
        tot = getattr(a, 'total_samples', 0)
        fac = lr_factor(c['samples'], tot) if tot > 0 else 1.
    else:
        fac = lr_factor(hours_now, a.hours)
    return a.lr*warm*fac


def masked_mean(x, m):
    return (x*m).sum()/m.sum().clamp_min(1.)


def k_untold(err, w, known, link):
    """The K read-out's error (err: per episode, decision, row and MV, as `gain`; w: where it counts) on the episodes that
    are not told the responses, an MV's own row and the other rows apart: what K is when it has the responses the
    network reads itself (`gain` counts the told ones too, and the problems of design of --design are told). link
    (episode, row, MV): the pairs, the plant's own (prog.c['link']: a permutation of the tracked rows, not the
    diagonal; 11-12, usual 64 plants: 57 of 214 pairs on it; told or not, as --unpaired leaves them out of the
    packet). Floats, for the rows; not a loss. A column is left out where nothing counts for it (every episode
    told: a mean of nothing is not a 0)."""
    w = w*(1-known.float()).reshape(-1, *(1,)*(w.dim()-1))
    own = (link > 0).to(w.dtype).reshape(link.shape[0], *(1,)*(w.dim()-3), *link.shape[1:])
    return {q: float(masked_mean(err, m)) for q, m in (('k_own', w*own), ('k_cross', w*(1-own))) if float(m.sum()) > 0}


def e_by_kind(err, wt, nl):
    """The E read-out's error (err: per episode, decision and row, as `head`; wt: where it counts) on the plants of the
    prior and on those of the engines apart (nl: per episode, 1 on a plant of the prior). The teacher's prediction,
    whose E the read-out is taught, is far off on a mechanistic plant away from its steady point (11-13, the
    simulation thread: 19 bands at the 90th percentile with no event and a valve moved far; the fix of the references'
    version 3 is of the prior's plants), so the two are read apart. Floats, for the rows; not a loss. A column is left
    out where nothing counts for it (no episode of the kind: a mean of nothing is not a 0)."""
    lin = nl.to(wt.dtype).reshape(-1, *(1,)*(wt.dim()-1))
    return {q: float(masked_mean(err, m)) for q, m in (('head_lin', wt*lin), ('head_mech', wt*(1-lin))) if float(m.sum()) > 0}


def clean(o):
    """Floats that are not finite become None: every file written is strict JSON (the site parses it in a browser)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {q: clean(v) for q, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def dumps(o, **kw):
    return json.dumps(clean(o), allow_nan=False, **kw)


def rnd(x, n):
    return None if x is None else round(x, n)


SIGN = dict(stall=0., file=None, path=None)                       # of this process (--stall): the seconds, where the stacks go (None: stderr), the file of its signs of life
KEPT = 'AIO_TRAIN_KEPT'                                           # in the environment of a run that is watched (--restarts)


def cap_card(dev, gb=0., conf='', spare=.5):
    """The most this process's allocator may hold of the card, in GB: `gb`, or (0) what is free when it starts less
    `spare`; below 0, no cap. On Windows the driver lets a card's memory run over into the computer's (shared GPU memory,
    from driver 536.40 on) rather than fail an allocation, so the allocator is never made to give back the blocks it
    keeps: it holds more than the card has and the rows get several times slower (11-13, 16 GB card, two lanes: 3.1 to
    3.7 GB over, a row of 24x10 four to eight times as long as alone). Capped (torch.cuda.set_per_process_memory_fraction),
    an allocation past the cap first has the blocks kept freed and is tried again, and is out of memory only if it still
    does not fit: what is in use at once has to fit the card, what is kept no longer pushes it over. None on a CPU.
    conf: the allocator's own settings (as PYTORCH_CUDA_ALLOC_CONF, over it; garbage_collection_threshold:.8 gives back
    the oldest blocks kept once it holds that share of the cap, before an allocation has to wait for all of them to be
    given back: it acts only under a cap). Measured by the allocator's counts (`retries`)."""
    if dev.type != 'cuda':
        return None
    if conf:
        merged = merge_alloc_conf(conf)
        if merged:
            torch.cuda.memory._set_allocator_settings(merged)
    if gb < 0:
        return None
    i = torch.cuda.current_device() if dev.index is None else dev.index     # (the fraction takes a card's number, not 'cuda')
    free, total = torch.cuda.mem_get_info(i)
    cap = gb*2**30 if gb > 0 else free+torch.cuda.memory_reserved(i)-spare*2**30
    torch.cuda.set_per_process_memory_fraction(max(.05, min(1., cap/total)), i)
    return round(cap/2**30, 2)


RETRIES = ('num_alloc_retries', 'num_device_alloc', 'num_device_free')


def retries(dev, since=None):
    """The allocator's counts so far (torch.cuda.memory_stats): allocations that did not fit and had what it kept freed
    before being tried again, the blocks it asked the card for, the blocks it gave back. since: the counts of an earlier
    call, and what is returned is what was added since. Near a cap (--card-gb) an allocator that keeps asking for and
    giving back blocks a window has these by the thousand and its rows go slow without being out of memory or over the
    card (11-13: two lanes of 24x10 and 48x20 under a cap, 45 minutes in the critic's fit)."""
    if dev.type != 'cuda':
        return dict.fromkeys(RETRIES, 0)
    s = torch.cuda.memory_stats()
    now = {q: int(s.get(q, 0)) for q in RETRIES}
    return now if since is None else {q: now[q]-since[q] for q in RETRIES}


def alive(**note):
    """A sign of life of a run: a window, or a batch of a validation or of the references, is done. The clock of --stall
    starts again (that long without a sign: where every thread stands is written to OUT/stalls.txt, and again after
    as long; the timer is not Python's, so a call that never returns is caught too), and OUT/alive is written anew:
    what the process watching a run of --restarts goes by (`watched`). Nothing where neither is set."""
    if SIGN['stall']:
        faulthandler.dump_traceback_later(SIGN['stall'], repeat=True, file=SIGN['file'] or sys.stderr)
    if SIGN['path'] is not None:
        try:
            SIGN['path'].write_text(dumps(dict(time=round(time.time(), 1), **note)))
        except OSError:                                           # (a file held by another process for a moment is no reason to end a run)
            pass


def waited(f, poll=60.):
    """The result of a builder's future (`Ahead`, the next episode of a lane's units, a set of problems of design), with a
    sign of life every `poll` seconds while it is in the making, for up to four times --stall: a batch of the slowest
    mechanistic kinds takes many minutes to draw on a builder's threads (more on a CPU the run shares), and the run
    waiting for it is not stalled (11-13: a run of the whole recipe on the CPU was stopped twice so); one that does not
    come in that long is, and the watch goes on as before."""
    import concurrent.futures as cf
    t = time.time()
    while True:
        try:
            return f.result(timeout=poll)
        except cf.TimeoutError:
            if time.time()-t < 4*SIGN['stall']:
                alive(build=round(time.time()-t))


def timed(into, q, f, *args, **kw):
    """f(*args, **kw), its seconds added to into[q]."""
    t = time.time()
    out = f(*args, **kw)
    into[q] = round(into.get(q, 0.)+time.time()-t, 1)
    return out


def end(pid):
    """A process and all it started (a run and its builders), at once."""
    if sys.platform == 'win32':
        subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)], capture_output=True)
    else:
        os.killpg(pid, signal.SIGKILL)                            # (a watched run is the head of its own group: `watched`)


def ends_with(fd):
    """A watched run ends when the process watching it does (its end of the pipe `fd` closes): at once, builders and all.
    (Read as a descriptor: a thread waiting in the buffered reader of stdin holds its lock, and the interpreter ending
    a run that is done then aborts on it.)"""
    def wait():
        try:
            while os.read(fd, 1024):
                pass
        finally:
            try:
                end(os.getpid())
            finally:
                os._exit(3)
    threading.Thread(target=wait, daemon=True).start()


def watched(cmd, out, stall, restarts, grace=30., poll=5.):
    """--restarts N: the run `cmd` as a child of this process, which watches its signs of life (OUT/alive: `alive`). A run
    that has given none for twice --stall (its stacks are in OUT/stalls.txt twice by then) is ended with all it started
    and started again with --resume: from OUT/last.pt, the last row (what it had of the row it stopped in is done
    again, the buffers fill anew), up to N times; each time a line in OUT/restarts.jsonl and on stderr. A run that ends
    by itself, in an error too, or that someone ends, is not started again. -> its exit code (3: it stalled once more
    than N times). 11-13: a --bench of three sizes on the training machine gave no line for 40 minutes, with no error
    and nothing to say where it stood."""
    sign, n = out/'alive', 0
    while True:
        sign.write_text(dumps(dict(time=round(time.time(), 1), started=n)))
        child = subprocess.Popen(cmd+['--resume']*(n > 0 and '--resume' not in cmd), stdin=subprocess.PIPE, env=dict(os.environ, **{KEPT: '1'}),
                                 start_new_session=sys.platform != 'win32')
        idle = 0.
        try:
            while child.poll() is None and idle <= 2*stall+grace:
                time.sleep(min(poll, stall/4))
                idle = time.time()-sign.stat().st_mtime
        finally:                                                  # (and when this process is interrupted: the run ends with it)
            stalled = child.poll() is None
            if stalled:
                end(child.pid); child.wait()
        if not stalled:
            return child.returncode
        line = dumps(dict(stalled=time.strftime('%Y-%m-%d %H:%M:%S'), idle_s=round(idle), last=sign.read_text(), restart=n+1 if n < restarts else None))
        print(line, file=sys.stderr, flush=True)
        with open(out/'restarts.jsonl', 'a') as f:
            f.write(line+'\n')
        if n >= restarts:
            return 3
        n += 1


def cut_rows(x, cut):
    """x (n, ...) with the gradient stopped on the episodes where cut (n,) is set."""
    return torch.where(cut.view(-1, *[1]*(x.dim()-1)), x.detach(), x)


class Guard:
    """Gradient hook on the moves of a window (n, NMV), where the plant's gradient enters the network. Per
    episode: a gradient that is not finite is dropped (broken), one above cap times `usual` is scaled down to
    it (capped). usual: the trainer's running median of the episodes' largest gradient norm in a window, in
    units of the objective's own gradient (the hook sees it times `weight`, the objective's weight in the
    loss); None while it is not known: nothing is held then. top (n,): each episode's largest norm, before
    the hold, in the same units."""

    def __init__(self, n, cap, device, usual=None, weight=1.):
        self.limit, self.weight, self.usual = None if usual is None else cap*usual*weight, weight, usual
        self.top = torch.zeros(n, device=device)
        self.capped = torch.zeros(n, dtype=torch.bool, device=device)
        self.broken = torch.zeros(n, dtype=torch.bool, device=device)

    def __call__(self, g):
        g, norm = self.finite(g)
        self.top = torch.maximum(self.top, norm/self.weight)
        return self.to_limit(g, norm)

    def finite(self, g):
        ok = torch.isfinite(g).all(-1)
        self.broken |= ~ok
        g = torch.where(ok[:, None], g, torch.zeros_like(g))
        return g, g.norm(dim=-1)

    def to_limit(self, g, norm):
        if self.limit is None:
            return g
        over = norm > self.limit
        self.capped |= over
        return torch.where(over[:, None], g*(self.limit/norm.clamp_min(1e-30))[:, None], g)

    def hold(self, g):
        """The same hold on the actor's output, where what came back through the plant (held above) meets what
        the later decisions send back through the last move they were given. A move that the next one answers
        (a valve turned back every decision) is a loop of its own: its gradient grows along the window without
        passing the plant (11-12, v2-stay-rl: steps of norm 1e3 to 1e11 with every move into the plant held).
        Not measured: the level stays that of the plant's gradient."""
        return self.to_limit(*self.finite(g))

    def held(self):
        return self.capped | self.broken

    def median(self):
        """The median over the episodes that had a gradient of their largest norm: this window's usual level."""
        some = self.top > 0
        return float(self.top[some].median()) if bool(some.any()) else None

    def peak(self):
        """The largest gradient of the window as a multiple of the usual level it was measured against."""
        return float(self.top.max()/self.usual) if self.usual else None


def running(old, new, keep=.9):
    return old if new is None else new if old is None else keep*old+(1-keep)*new


def td_targets(cost, v, gamma, lam):
    """TD(lambda) targets (n, H) of Q at the H decisions of a window: cost (n, H) of each, v (n, H+1) the
    values at them and at the decision after the last."""
    G, out = v[:, -1], []
    for j in range(cost.shape[1]-1, -1, -1):
        G = cost[:, j]+gamma*((1-lam)*v[:, j+1]+lam*G)
        out.append(G)
    return torch.stack(out[::-1], 1)


def reach(a, m=None):
    """Sum of the discounts over the decisions a value is of: all that are to come, or with heads of fixed horizons
    (--fixed S) the S*(m+1) of head m (None: the longest, the one the actor's objective has)."""
    S = getattr(a, 'fixed', 0)
    return (1.-a.gamma**(S*(a.heads if m is None else m+1)))/(1.-a.gamma) if S else 1./(1.-a.gamma)


def value_of(z, a):
    """The cost to come as the critic's heads (.., heads; symlog) give it to the actor's objective: their mean, or
    with heads of fixed horizons the longest."""
    return symexp(z[..., -1] if getattr(a, 'fixed', 0) else z.mean(-1))


def fixed_targets(cost, z, S, gamma):
    """Symlog targets (n, H, M) of heads of fixed horizons at the first H decisions of a series: head m the
    discounted cost of the S*(m+1) decisions from each on. cost (n, H+S) of the series' decisions, z (n, H+S, M)
    the heads' symlog values at them, as they were when they were run. Head 0 has costs alone in it; head m the
    costs of the first S decisions and head m-1's value S decisions on. No head is fitted towards itself, so none
    can raise itself along an episode whose end no cost ever answers (fixed-horizon TD: De Asis et al. 2019;
    11-12: a critic fitted towards its own value after the window, with a memory that knows how long a unit has
    been run, said 4 times the cost then paid at the end of one episode and 20 to 40 times it in the fourth)."""
    H = cost.shape[1]-S
    first = cost.unfold(1, S, 1)[:, :H]@(gamma**torch.arange(S, device=cost.device, dtype=cost.dtype))
    return symlog(torch.cat((first[..., None], first[..., None]+gamma**S*symexp(z[:, S:S+H, :-1]).clamp_min(0.)), -1))


class Due:
    """The windows of a lane the critic is yet to be fitted on (--fixed S: a window is fitted once the S decisions
    after its last have been run), and of every decision since the first of them what it cost and what the heads
    said there. ends: where the run of a unit ends (an episode that failed and starts again at rest; all of them
    when the lane is released) its cost is taken to go on as it was over its last S decisions, and the heads to
    say what that comes to: for the decisions before such an end the targets have the plant's answer alone in
    them, not the next run's and not the critic's."""

    def __init__(self, a):
        self.S, self.gamma, self.reach = a.fixed, a.gamma, [reach(a, m) for m in range(a.heads)]
        self.recs, self.cost, self.z, self.ends = [], None, None, []

    def add(self, rec, part=None):
        """A window just run. part: the episodes the critic is fitted on (None: all)."""
        H = rec['cost'].shape[1]
        cost, z = rec['cost'], rec['z'][:, :H]
        self.cost, self.z = (cost, z) if self.cost is None else (torch.cat((self.cost, cost), 1), torch.cat((self.z, z), 1))
        keys = ('priv',) if 'priv' in rec else ('obs', 'prev', 'act', 'lab', 'hc', 'feats')
        keep = {q: rec[q] for q in keys}
        if part is not None:
            keep = {q: (tuple(t[part] for t in x) if q == 'priv' else x if x is None else x[:, part] if q == 'hc' else x[part]) for q, x in keep.items()}
        self.recs.append((keep, part, H))

    def gone(self, who):
        """The run of these units (n,) ends with the last decision added."""
        if self.cost is not None and bool(who.any()):
            rate = self.cost[:, -self.S:].mean(1)
            self.ends.append((self.cost.shape[1], who, rate, symlog(rate[:, None]*rate.new_tensor(self.reach))))

    def last(self):
        """The lane is released: every unit's run ends, and the S decisions its last windows wait for are these."""
        if self.cost is not None:
            self.gone(torch.ones_like(self.cost[:, 0], dtype=torch.bool))
            _, _, rate, says = self.ends[-1]
            self.cost, self.z = torch.cat((self.cost, rate[:, None].expand(-1, self.S)), 1), torch.cat((self.z, says[:, None].expand(-1, self.S, -1)), 1)

    def ready(self):
        return bool(self.recs) and self.cost.shape[1] >= self.recs[0][2]+self.S

    def pop(self):
        """The first window, the episodes it is fitted on, its targets (of those episodes: n, H, M) and what the
        longest head said over it."""
        rec, part, H = self.recs.pop(0)
        cost, z = self.cost[:, :H+self.S].clone(), self.z[:, :H+self.S].clone()
        for at, who, rate, says in self.ends[::-1]:              # (the earliest end last: it is the one this window's units met)
            if at < H+self.S:
                cost[:, at:], z[:, at:] = torch.where(who[:, None], rate[:, None], cost[:, at:]), torch.where(who[:, None, None], says[:, None], z[:, at:])
        tgt, said = fixed_targets(cost, z, self.S, self.gamma), float(symexp(self.z[:, :H, -1]).mean())
        self.cost, self.z = self.cost[:, H:], self.z[:, H:]
        self.ends = [(at-H, *more) for at, *more in self.ends if at > H]    # an end no window left is before is done with
        return rec, part, (tgt if part is None else tgt[part]), said


def set_teacher(prog, meter=False):
    """The teacher of a batch that trains: the one that is imitated, whose E, K and V are the labels and which is told
    in the told episodes. Its model follows a process event 10-40 decisions late. --meter: its prediction is
    corrected from what the instruments report alone and does not see the unmeasured loads move
    (aio.sim.core.Program.set_teacher: what a controller in the field can know; 11-12, the same batch: 94 points
    against 99.9, the plants with a level 91). The costs, the scores and the true responses stay the plant's own."""
    return prog.set_teacher('late', False, True) if meter else prog.set_teacher('late')


def with_faults(prog, share):
    """--faults: one fault on a share of the episodes of a batch (aio.sim.faults.train_draw), by the batch's seed."""
    if share > 0:
        from aio.sim import faults as FT
        prog.fault = FT.train_draw(prog, share, prog.seed)


_LANE_IDS = itertools.count(1)


class Lane:
    """A batch of episodes on its way, with what is carried from one window to the next: the plant's state,
    the recurrent states of actor and critic, the last moves, and per episode how short its stretches are
    (level: the window halved that many times).
    teach: the share of the episodes that are for imitation (`label`): the teacher's move is the target there
    and nothing else is learned from them; the others are for the simulator's gradient and carry no imitation
    term. An episode is one or the other: an imitation term on an episode whose moves keep their graph sends
    its gradient through the plant too, towards states that are easy to imitate (11-11: at equal weights ten
    times the objective's own gradient at the moves, and against it). drive: the share of the imitation
    episodes the teacher drives."""

    def __init__(self, prog, a, gen, teach=0., drive=0.):
        self.tag = next(_LANE_IDS)
        n, dev = prog.n, prog.device
        self.meter = bool(getattr(a, 'meter', False))                                        # --meter: the teacher from the instruments alone (`set_teacher`)
        self.need_teacher = bool(teach > 0 or getattr(a, 'bc_iters', 1) > 0 or getattr(a, 'bc_floor', 0.) > 0 or getattr(a, 'bench', False)
                                 or getattr(a, 'after', 0) > 0 or getattr(a, 'lead', 0.) > 0 or getattr(a, 'law_keep', '') or getattr(a, 'law', False))
        if self.need_teacher:
            set_teacher(prog, self.meter)
        else:
            prog.Fr = None                                                                   # skip teacher prediction & 4000-iter meter Riccati when pure real-cost training
        self.faults = getattr(a, 'faults', 0.)                                               # --faults: a share of the episodes with a valve stuck or an instrument gone (aio.sim.faults)
        with_faults(prog, self.faults)
        self.prog, self.k, self.st = prog, 0, C.reset(prog)
        _, _, _, SSLOW, SROW, SARX, SDV, _ = N.sizes(prog.ncv, prog.nmv)                     # (of the batch's size: one size a batch)
        sets = 2 if getattr(a, 'slow', 0.) > 0 else 1
        more = N.FM.SMEM if any(getattr(a, q, False) for q in ('formula', 'solve', 'given')) else 0   # the actor's state holds the moves made too (aio.formula)
        more += SSLOW if sets > 1 else 0                                                     # and the slow set of the running sums (--slow)
        more += SROW+sets*SARX if getattr(a, 'arx', False) else 0                           # and the sums with the rows' own past (--arx)
        more += N.SDVF+sets*SDV if getattr(a, 'dv', False) else 0                           # and with the measured disturbances (--dv)
        if getattr(a, 'reader', False):                                                      # --reader: its memory where the sums' would be (aio.reader), not their size
            more += N.RD.size_of(prog.ncv, prog.nmv, **N.RD.said(a.reader))-N.sizes(prog.ncv, prog.nmv)[2]
        if getattr(a, 'seq', None):                                                          # --seq: the decisions it sees, where the sums would be (aio.seqnet)
            more += N.SQ_ctx(prog.ncv, prog.nmv, a.seq_window, a.seq)-N.sizes(prog.ncv, prog.nmv)[2]
        state = N.state_size(prog.ncv, prog.nmv)
        self.ha, self.hc = torch.zeros(1, n, state+more, device=dev), torch.zeros(1, n, state, device=dev)
        self.size = size_name(prog)                                                          # (--sizes: its buffer, guard and problems of design are of this size)
        self.nk = getattr(a, 'knots', N.NLAB)                                                # lags the responses are read out at (--knots)
        self.hs = None                                                                       # the running sums (aio.net.Stats)
        self.prev = torch.zeros(n, prog.nmv, device=dev)
        self.pm = torch.zeros(n, prog.nmv, device=dev)                                            # the last move before its noise (--smooth)
        self.level = torch.zeros(n, dtype=torch.long, device=dev)
        self.quiet = torch.zeros(n, dtype=torch.long, device=dev)                            # windows since its gradient was last held
        self.known = (torch.rand(n, generator=gen, device=dev) < a.known).float()           # told the true responses: actor
        self.cknown = (torch.rand(n, generator=gen, device=dev) < a.critic_known).float()   # and critic
        self.score = torch.zeros(n, device=dev)
        self.rows, self.link = prog.c['prc'].float(), prog.c['link'][:, :, :prog.nmv].float()
        # --unpaired: the episodes that are not told the pairing nor which way an MV acts (mixed.canonical); drawn only when asked for
        self.unpaired = (torch.rand(n, generator=gen, device=dev) < a.unpaired) if getattr(a, 'unpaired', 0.) > 0 else None
        self.label = torch.rand(n, generator=gen, device=dev) < teach if 0 < teach < 1 else torch.full((n,), teach >= 1, dtype=torch.bool, device=dev)
        if getattr(a, 'bc_told', False) and self.unpaired is not None:   # --bc-told: no blind episode imitates the teacher, which knows the plant (its moves never look for anything)
            self.label = self.label & ~self.unpaired
        # the episodes the teacher drives (--drive): the student is shown the states a good controller keeps a unit in
        self.drive = self.label & (torch.rand(n, generator=gen, device=dev) < drive) if drive > 0 and teach > 0 else None
        # --lead: the decision up to which the teacher has the plant in an episode of the simulator gradient (0: the network from the start)
        share = getattr(a, 'lead', 0.)
        self.lead = (torch.where(~self.label & (torch.rand(n, generator=gen, device=dev) < share), prog.k_first.to(dev).long()+torch.randint(1, a.lead_to+1, (n,), generator=gen, device=dev), 0)
                     if share > 0 else None)
        self.age, self.left, self.next, self.kw, self.mix = 0, 0, None, {}, None                            # episodes these units have been run before this one (--stay), those to come, the next one in the making, its menu (`menu_kw`) and which groups of --menu-mix it took
        self.group = None                                                                    # the group of --mech-groups its mechanistic episodes are of
        self.due = Due(a) if getattr(a, 'fixed', 0) else None                                # the windows the critic is yet to be fitted on (--fixed)

    def onto(self, prog):
        """The next episode of the same units (prog: built `after` this lane's program): the plant's state goes on
        (aio.sim.core.resume), and so do the memories, the running sums and the last moves; an episode that failed
        starts again at rest, its memories cleared. Which episodes are told, taught or driven stays as it is."""
        if self.need_teacher:
            set_teacher(prog, self.meter)
        else:
            prog.Fr = None
        with_faults(prog, self.faults)
        keep = (~self.st['fail']).float()
        if self.due is not None:                                  # an episode that failed does not go on: its next one is another run
            self.due.gone(self.st['fail'])
        st = C.resume(prog, self.st)
        self.prog, self.k, self.st = prog, 0, st
        self.ha, self.hc = self.ha*keep[None, :, None], self.hc*keep[None, :, None]
        self.hs = None if self.hs is None else self.hs*keep[:, None].to(self.hs.dtype)
        self.prev, self.pm = self.prev*keep[:, None], self.pm*keep[:, None]
        self.score = torch.zeros_like(self.score)
        self.age, self.left, self.next = self.age+1, self.left-1, None
        if bool((~keep.bool()).any()):
            self.tag = next(_LANE_IDS)

    def labels(self, k, s):
        """(n, 1, rows, MVs, knots) the true responses at decision k, direct-acting frame."""
        return pair_labels(self.prog.labels(k, self.nk).float()*s[:, :, None])[:, None]

    @property
    def done(self):
        return self.k >= HORIZON


def run(fn, lean, *args):
    """fn(*args). lean: its graph is not kept but rebuilt at the backward pass: the same gradient, and the
    memory of a window no longer grows with the networks' graphs of every decision (98% of it)."""
    return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False) if lean and torch.is_grad_enabled() else fn(*args)


def sums(net, obs, hs):
    """The running sums (aio.net.Stats) moved on by a decision's packets -> their features, for actor and
    critic alike, and their state; None and hs for a network without them."""
    return (None, hs) if net.pairs.stats is None else net.pairs.stats(obs[:, None], hs)


def steps_law(net, *args):
    """net.steps, and behind what it returns the law's readouts of E, K and V as they are read (aio.net.PairNet,
    `read_law`; a network that solves the design reads E alone): returned, so that through `run` a loss on them
    has their graph."""
    out = net.steps(*args)
    eh, kh, vh, _ = net.pairs.read_law
    return out+((eh,) if kh is None else (eh, kh, vh))+(() if net.pairs.unseen is None else (net.pairs.read_unseen,))


def decide(net, lane, obs, prev, ha, s, k, gen, noise, lean=False, feats=None, law=None, readl=None, unseen=None, need_read=True, lane_tag=None):
    """The actor at one decision -> the move in the direct-acting frame (with noise of that spread before the
    tanh), the mean before the tanh, the MV mask, the responses read (n, NCV, NMV, NLAB) or None when skipped, the state, the true
    ones. feats: of `sums`. law: the teacher's E, K and V there (aio.net.law_code), which the told episodes get too.
    readl: a list the law's readouts of this decision are put in (`steps_law`: E (n,NCV,NLAB), then K, V; last, of a
    network with --unseen, what it reads of what the instruments do not show). unseen: the true one there
    (aio.net.unseen_code), told with the law."""
    true = lane.labels(k, s)
    code = (true, lane.known) if law is None else (true, lane.known)+tuple(x[:, None] for x in law[:3])+(() if unseen is None else (unseen[:, None],))
    if readl is None:
        has_nr, has_lt = _steps_caps(net)
        skw = {}
        if has_nr and not need_read:
            skw['need_read'] = False
        if has_lt and lane_tag is not None:
            skw['lane_tag'] = lane_tag
        if skw:
            if lean and torch.is_grad_enabled():
                mean, _, mask, lab, ha = run(functools.partial(net.steps, **skw), lean, obs[:, None], prev[:, None], ha, code, feats)
            else:
                mean, _, mask, lab, ha = net.steps(obs[:, None], prev[:, None], ha, code, feats, **skw)
        else:
            mean, _, mask, lab, ha = run(net.steps, lean, obs[:, None], prev[:, None], ha, code, feats)
    else:
        mean, _, mask, lab, ha, *more = run(functools.partial(steps_law, net), lean, obs[:, None], prev[:, None], ha, code, feats)
        readl[:] = [x[:, 0] for x in more]
    mean, mask = mean[:, 0], mask[:, 0]
    scale = getattr(net.actor, 'act_scale_last', None)
    if scale is not None and (torch.is_tensor(noise) or noise):
        sc = scale[:, 0].detach().clamp(.02, 1.)
        if lane.unpaired is not None:
            sc = torch.where(lane.unpaired[:, None], sc.clamp_min(.15), sc)
        noise = noise*sc
    z = mean+noise*torch.randn(mean.shape, generator=gen, device=mean.device) if torch.is_tensor(noise) or noise else mean
    return torch.tanh(z)*mask, mean, mask, (None if lab is None else lab[:, 0]), ha, true


RESPONDS = .2                             # --id-sign: a pair answers where its response is this large at some lag (asinh units; a fifth of the pairs in service do not)


def read_error(net, read, true, a):
    """What the read-out is taught, per pair (..., NCV, NMV). The responses at the knots (squared, asinh units); or, --id-sign,
    two answers only: whether the row answers the MV at all (the read-out's second value, a logit) and, where it does, which
    way (its first). Which valve moves which row which way is all the packet leaves out when nothing is published about the
    pairing; a network is taught it in minutes where the history shows it (diagnostics/read-1117), and neither the teacher's
    moves nor the plant's gradient taught it in 200 rows."""
    if not getattr(a, 'id_sign', False):
        return (N.taught(net, read, 'S')-N.taught(net, true, 'S')).square().mean(-1)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    live = (true.abs().amax(-1) > RESPONDS).to(read.dtype)
    return bce(read[..., 1], live, reduction='none')+live*bce(read[..., 0], (true[..., -1] > 0).to(read.dtype), reduction='none')


def driven(lane, st, k, s, mask, u, spread, gen, law=None, own=False):
    """The teacher's move in the agent's frame, and the move that goes to the plant: in the episodes the teacher
    drives its own with noise of that spread before the tanh (the states around its path are seen too), the
    student's `u` in the others."""
    # --law: the law's move is what is imitated; --bc-own: the teacher's own, its plan within the limits (the law's move
    # held to the limits is another controller: on the instruments it lives at 87.8 against the teacher's 94.2, 78.0
    # against 92.4 where a level is a row, and with a valve at its limit it pushed a level the wrong way half the time)
    tch = (C.teacher(lane.prog, st, k).float()*s if law is None or own else law[4])*mask
    if lane.drive is None:
        return tch, u
    z = torch.atanh(tch.clamp(-.999, .999))+spread*torch.randn(tch.shape, generator=gen, device=tch.device)
    return tch, torch.where(lane.drive[:, None], torch.tanh(z)*mask, u)


def window(net, critic, lane, a, w, gen, usual=None, later=0, rel_probe=None):
    """One window of a lane with the graph kept. w: the weights of the terms (`weights`); usual: the guard's
    reference. Returns the loss, what the critic is then fitted on, the guard and the log. The lane is moved
    on to the window's end. later: the critic goes through the window after its last decision, that many episodes
    at a time over all its decisions at once, instead of a decision at a time (the same values and state; only
    where nothing is differentiated: `imitate`. All of 1,024 episodes at once were most of what a card held:
    11-11, 13.7 GB against 6.5)."""
    prog = lane.prog
    n, dev, c, nmv = prog.n, prog.device, prog.c, prog.nmv
    k0 = lane.k
    H = min(a.window, HORIZON-k0)
    st, ha, hc, hs, prev, pm = C.detach(lane.st), lane.ha.detach(), lane.hc.detach(), lane.hs, lane.prev.detach(), lane.pm.detach()
    hor = a.window >> lane.level if a.guard == 'cut' and not later else None    # (n,) decisions between two cuts of an episode's graph
    hor_set = set(hor.tolist()) if hor is not None else None                    # hoisted once per window: zero per-step GPU->CPU syncs
    lead_max = int(lane.lead.max()) if lane.lead is not None else -1
    opened = lane.level > 0 if a.guard == 'open' and not later and bool((lane.level > 0).any()) else None
    lab = lane.label if lane.drive is None else lane.label | lane.drive   # the imitation episodes
    taught, nsim = bool(lab.any()), float((~lab).sum())
    simw = w['sim'] if nsim else 0.
    # the guard's unit (GUARD_UNITS=3): batch-size-invariant per-episode objective gradient (multiplied by nsim / simw)
    guard, tail = Guard(n, a.cap, dev, usual, max(simw, 1e-9)/max(nsim, 1.)), reach(a)
    zero = torch.zeros(n, device=dev)
    tot, acc, wsum, disc, worst = zero, zero, zero, torch.ones(n, device=dev), zero
    l_bc = l_id = own = cross = 0.
    l_rel = torch.zeros((), device=dev)
    rel_n = torch.zeros((), device=dev)
    rel_stats = torch.zeros(4, dtype=torch.float64, device=dev)   # (live_ok, live_tot, sgn_ok, sgn_tot) accumulated on GPU
    split = dict(own_told=0., own_blind=0.)
    kept = getattr(a, 'law_keep', '') or ''                      # --law-keep: which of the law's readouts stay taught beside the simulator gradient
    keep = bool(kept) and net.pairs.law and simw > 0 and not later
    l_law, readl, trk = dict(head=0., gain=0., valve=0.), [] if keep else None, c['trk'][:, :nmv].float()
    priv = bool(getattr(critic, 'privileged', False))            # --priv-critic: training-only differentiable Markov V(st)
    rec, fs, pfs = dict(obs=[], prev=[], act=[], cost=[], z=[], lab=[], hc=hc), [], []
    probe = getattr(a, 'probe', 0.)                               # (a caller with the settings of an earlier checkpoint: aio.probe)
    can_probe = bool(probe) and (taught or (simw > 0 and (not hasattr(net.actor, 'when') or bool(net.actor.when.weight.detach().any() or net.actor.when.bias.detach().any()))))
    pre_cost = getattr(a, 'pre_cost', 1.)                         # --pre-cost: what the blind episodes pay before their first event, where they may try their valves
    k_first = prog.k_first.to(dev).long()
    free = (lane.unpaired & (k_first > k0)) if pre_cost != 1. and lane.unpaired is not None else None
    moved = torch.zeros(4, dtype=torch.float64, device=dev)
    blind = bool(getattr(a, 'ppo', False))                        # --ppo: the critic is not given the move (the value of the state, the baseline of `ppo`)
    need_read = bool(a.id_coef > 0)
    win_tag = (lane.tag, lane.age)
    rnd_buf = _prefetch_noise(prog, k0, H, dev)
    use_amp = bool(getattr(net, 'amp', False)) and dev.type == 'cuda'
    amp_ctx = torch.autocast('cuda', dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
    use_sim_c = bool(getattr(C, 'is_sim_compiled', lambda: False)()) and dev.type == 'cuda'
    with amp_ctx:
        for j in range(H):
            k = k0+j
            P, t = (None, None) if use_sim_c else C.step_ctx(prog, k)
            obs, s = canonical(C.observe(prog, st, k, t=t).float(), lane.unpaired)
            F, hs = sums(net, obs, hs)
            cut = (j % hor == 0) if (hor_set is not None and j and any(j % hh == 0 for hh in hor_set)) else None
            stop = cut if opened is None else opened                  # episodes whose actor does not pass gradient back from here
            if stop is not None and hasattr(net, 'cut'):              # --seq: the graph also runs through what its time attention keeps of the decisions before (11-23: the cut stopped nothing there)
                net.cut(stop)
            o, p, h = (obs, prev, ha) if stop is None else (cut_rows(obs, stop), cut_rows(prev, stop), cut_rows(ha[0], stop)[None])
            if taught:                                                # imitation: the last move is data, as the plant's answer is
                p = cut_rows(p, lab)
            if can_probe:                                             # (n,) before its first event: probed (only blind episodes when --unpaired > 0; never when `when` is untrained under --bc-iters 0)
                pre = ((k < k_first) & lane.unpaired).float() if lane.unpaired is not None else (k < k_first).float()
            else:
                pre = None
            noise = a.noise if pre is None else (a.noise+(probe-a.noise)*pre)[:, None]
            law = N.law_code(prog, st, k, s) if net.pairs.law and (taught or keep) else None    # the teacher's E, K, V of this decision, and the law's move
            hid = N.unseen_code(prog, st, k) if law is not None and net.pairs.unseen is not None else None    # --unseen: what the instruments do not show
            # (--meter: the told episodes are not told it. The teacher's E is then of its own estimate of these, not of the true ones: the E read-out
            #  takes the network's own reading in every episode, as it will when it runs; the reading's labels stay the true ones)
            u, mean, mask, read, ha, true = decide(net, lane, o, p, h, s, k, gen, noise, a.lean, F, law, readl, None if lane.meter else hid,
                                                   need_read=need_read, lane_tag=win_tag)
            if simw > 0 and u.requires_grad:
                u.register_hook(guard.hold)
            tch, took = driven(lane, st, k, s, mask, u, a.drive_noise, gen, law, getattr(a, 'bc_own', False)) if taught else (None, u)
            led = lane.lead is not None and k < lead_max and (k < lane.lead)[:, None]
            if led is not False:                                      # --lead: the teacher's move goes to the plant, and is the last move the network is given
                took = torch.where(led, C.teacher(prog, st, k).float().detach()*s*mask, took)
            if priv:
                if cut is not None and not later and w['value'] > 0:
                    z, pf = critic.eval_state(prog, st, k, P=P, t=t)
                else:
                    with torch.no_grad():
                        pf = tuple(x.detach() for x in N.priv_features(prog, st, k, P=P, t=t))
                    z = None
                pfs.append(pf)
            elif not later:
                z, hc = run(critic.steps, a.lean, obs[:, None], prev.detach()[:, None], (torch.zeros_like(took) if blind else took.detach())[:, None], hc, (true, lane.cknown), F)
                z = z[:, 0]
            if cut is not None:                                       # a stretch ends: what it cost and what the critic says follows
                if z is not None:
                    tot = tot+torch.where(cut, acc+w['value']*disc*value_of(z, a), zero)
                    wsum = wsum+torch.where(cut, w['value']*disc*tail, zero)
                else:
                    tot = tot+torch.where(cut, acc, zero)
                acc, disc = torch.where(cut, zero, acc), torch.where(cut, torch.ones_like(disc), disc)
                hc = cut_rows(hc[0], cut)[None]
                st, pm = {q: cut_rows(x, cut) for q, x in st.items()}, cut_rows(pm, cut)
            m = mask*(~st['fail']).float()[:, None]
            if taught:
                l_bc = l_bc+masked_mean((torch.tanh(mean)*mask-tch).square(), m*lab[:, None])
                rec.setdefault('tch', []).append(tch); rec.setdefault('m', []).append(m)
                if pre is not None:
                    rec.setdefault('pre', []).append(pre)
                if law is not None:                                   # of the tracked rows (the first nmv row slots)
                    for q, x in zip(('E', 'K', 'V', 'lm'), (law[0][:, :nmv], law[1][:, :nmv], law[2], law[3])):
                        rec.setdefault(q, []).append(x)
                    if hid is not None:
                        rec.setdefault('U', []).append(hid[:, :nmv]); rec.setdefault('since', []).append(prog.since(k).clamp(0, 9999).float())
            if keep:                                                  # E, K, V as read towards the teacher's, on every episode (as `learn` does where the teacher is all)
                if 'E' in kept:
                    eh = (N.taught(net, readl[0][:, :nmv].float(), 'E')-N.taught(net, law[0][:, :nmv], 'E')).square().mean(-1)
                    l_law['head'] = l_law['head']+masked_mean(eh, trk*m.amax(-1)[:, None])
                    for q, v in e_by_kind(eh.detach(), trk*m.amax(-1)[:, None], prog.ar < prog.nl).items():
                        l_law[q] = l_law.get(q, 0.)+v
                if 'E' in kept and hid is not None:                   # and what the instruments do not show, which the E read-out is given
                    l_law['unseen'] = l_law.get('unseen', 0.)+masked_mean(huber1(readl[-1][:, :nmv].float(), hid[:, :nmv]).mean(-1), trk*m.amax(-1)[:, None])
                if 'K' in kept and len(readl)-(hid is not None) > 1:
                    ek = (readl[1][:, :nmv].float()-law[1][:, :nmv]).square().mean(-1)
                    l_law['gain'] = l_law['gain']+masked_mean(ek, trk[:, :, None]*m[:, None, :])
                    for q, v in k_untold(ek.detach(), trk[:, :, None]*m[:, None, :], lane.known, lane.link[:, :nmv]).items():
                        l_law[q] = l_law.get(q, 0.)+v
                if 'V' in kept and len(readl)-(hid is not None) > 1:
                    l_law['valve'] = l_law['valve']+masked_mean((readl[2].float()-law[2]).square(), m)
            want_rel = rel_probe is not None and getattr(a, 'rel_id_coef', 0.) > 0 and getattr(net.actor, 'rel_feat_last', None) is not None and k >= 16
            if need_read or want_rel:
                pairs = lane.rows[:, :, None]*m.detach()[:, None, :]
            if need_read and read is not None:
                err = read_error(net, read, true[:, 0], a)                  # (--diff S: both as changes from knot to knot)
                l_id = l_id+masked_mean(err, pairs)
                ed = err.detach()
                own, cross = own+masked_mean(ed, pairs*lane.link), cross+masked_mean(ed, pairs*(1-lane.link))
                if lane.unpaired is not None:                             # told the pairing (the read-out has it given) and not told, apart
                    for q, e in (('own_told', ~lane.unpaired), ('own_blind', lane.unpaired)):
                        split[q] = split[q]+masked_mean(ed, pairs*lane.link*e.float()[:, None, None])
            if want_rel:
                # Training-only auxiliary supervision strictly on UNPUBLISHED pairs (branchless GPU math: zero per-step CUDA->CPU syncs!)
                unpub = pairs*(1.-lane.link) if lane.unpaired is None else pairs*torch.where(lane.unpaired[:, None, None], torch.ones_like(lane.link), 1.-lane.link)
                rf = net.actor.rel_feat_last
                lg_r = rel_probe(rf)
                lv_t = (true[:, 0].abs().amax(-1) > RESPONDS).to(lg_r.dtype)
                sg_t = (true[:, 0, ..., -1] > 0).to(lg_r.dtype)
                bce_l = torch.nn.functional.binary_cross_entropy_with_logits(lg_r[..., 0], lv_t, reduction='none')
                bce_s = torch.nn.functional.binary_cross_entropy_with_logits(lg_r[..., 1], sg_t, reduction='none')
                unpub_s = unpub.sum()
                m_pos, m_neg = unpub*lv_t, unpub*(1.-lv_t)
                sp, sn = m_pos.sum(), m_neg.sum()
                hp, hn = (sp > 0).to(lg_r.dtype), (sn > 0).to(lg_r.dtype)
                l_live = (hp*(bce_l*m_pos).sum()/sp.clamp_min(1.) + hn*(bce_l*m_neg).sum()/sn.clamp_min(1.))/(hp+hn).clamp_min(1.)
                l_sgn = (bce_s*m_pos).sum()/sp.clamp_min(1.)
                step_rel = l_live+l_sgn
                if lg_r.shape[-1] >= 3:
                    step_rel = step_rel+.5*(huber1(lg_r[..., 2], true[:, 0].abs().amax(-1).to(lg_r.dtype))*m_pos).sum()/sp.clamp_min(1.)
                has_u = (unpub_s > 0).to(lg_r.dtype)
                l_rel = l_rel+step_rel*has_u
                rel_n = rel_n+has_u
                with torch.no_grad():
                    pl, ps = (lg_r[..., 0] > 0).to(lv_t.dtype), (lg_r[..., 1] > 0).to(sg_t.dtype)
                    rel_stats = rel_stats+torch.stack((((pl == lv_t)*unpub).sum(), unpub_s, ((ps == sg_t)*m_pos).sum(), sp)).double()
            act = (u if simw > 0 else u.detach())*s                   # teacher alone: the plant is run without the graph
            if act.requires_grad:
                act.register_hook(guard)
            if taught:                                                # after the hook: an imitation episode's move reaches the plant as data
                act = torch.where(lab[:, None], took.detach()*s, act)
            if led is not False:
                act = torch.where(led, took.detach()*s, act)
            step_kw = dict(through=not getattr(a, 'hard', False), stops=getattr(a, 'stops', False), P=P, t=t)
            if rnd_buf is not None:
                step_kw['rnd'] = rnd_buf[j]
            st, out = C.step(prog, st, act.to(prog.dtype), k, **step_kw)
            cost, worst = fail_above(out['fail'], OR.objective(out, st, c, a.tau, a.cost_w, a.log_w).float(), worst, a.fail_cost)
            dm = torch.tanh(mean)*mask
            if led is not False:                                      # --lead: in the teacher's hands the network's move goes nowhere; the teacher's is the last one
                dm = torch.where(led, took.detach(), dm)
            if a.smooth:                                              # a move that turns back on the last one costs; a ramp does not (11-23: not on a move the teacher made, which
                cost = cost+a.smooth*(dm-pm).square().sum(-1)/mask.sum(-1).clamp_min(1.)    # pushed the network's unused move to 0, ten times over through the episode's own small level)
            if getattr(a, 'rev_smooth', 0.):                          # reversal-only anti-chatter penalty: zero drag on monotonic moves/steps, penalizes sign flips only
                cost = cost+a.rev_smooth*torch.relu(-dm*pm).sum(-1)/mask.sum(-1).clamp_min(1.)
            if free is not None:
                free = free & (k < k_first)
                cost = torch.where(free, pre_cost*cost, cost)
            with torch.no_grad():                                     # gauges: the size of the moves, how many stand at a limit, how far one is from the last
                moved = moved+torch.stack(((dm.square()*m).sum(), ((dm.abs() > .95).float()*m).sum(), ((dm-pm).square()*m).sum(), m.sum())).double()
            pm = dm
            acc, wsum, disc = acc+disc*cost, wsum+disc, disc*a.gamma
            lane.score += out['outside'].sum(-1)*(k >= prog.k_first)
            for q, x in (('obs', obs), ('prev', prev), ('act', torch.zeros_like(took) if blind else took), ('cost', cost), ('lab', true[:, 0]))+(() if (later or priv) else (('z', z),)):
                rec[q].append(x.detach())
            if blind:                                                 # the move as made, the mean it was drawn around, the MVs it is of
                for q, x in (('u', took), ('mean', mean), ('mk', m)):
                    rec.setdefault(q, []).append(x.detach())
            if F is not None:
                fs.append(F[:, 0])
            prev = took
    if later and not priv:                                        # the critic over the window's decisions in one call, from the state it had
        seq = [torch.stack(rec[q], 1) for q in ('obs', 'prev', 'act', 'lab')]+[torch.stack(fs, 1) if fs else None]
        parts = [slice(i, i+later) for i in range(0, n, later)]
        z, hc = zip(*(critic.steps(*(x[p] for x in seq[:3]), hc[:, p], (seq[3][p], lane.cknown[p]), None if seq[4] is None else seq[4][p]) for p in parts))
        hc = torch.cat(hc, 1)
        rec['z'] = list(torch.cat(z).detach().unbind(1))
    # after the window's last decision: the critic on the state reached and the move the actor would make there
    k1 = k0+H
    obs, s = canonical(C.observe(prog, st, k1).float(), lane.unpaired)
    F, _ = sums(net, obs, hs)                                     # not kept: the next window starts at this decision
    if not priv:
        with torch.no_grad(), (net.peek() if hasattr(net, 'peek') else contextlib.nullcontext()):    # (--seq: made again as the next window's first, not kept)
            u1, _, mask, _, _, true = decide(net, lane, obs, prev, ha, s, k1, gen, a.noise, feats=F, need_read=False)
            if lane.drive is not None and k1 < HORIZON:
                u1 = driven(lane, st, k1, s, mask, u1, a.drive_noise, gen)[1]
        if blind:
            u1 = torch.zeros_like(u1)
    else:
        u1, true = torch.zeros_like(prev), lane.labels(k1, s)
    more = min(getattr(a, 'after', 0), HORIZON-k1) if simw > 0 and not later else 0
    if priv:
        with torch.no_grad() if more > 0 or w['value'] == 0 else contextlib.nullcontext():
            z, pf1 = critic.eval_state(prog, st, k1)
        pfs.append(pf1)
        priv_seq = tuple(torch.stack([pf[i] for pf in pfs], 1) for i in range(len(pf1)))
        with torch.no_grad():
            z_all = critic(*priv_seq)                             # (n, H+1, heads) in one parallel batched call
        rec['priv'] = priv_seq
        rec['z'] = z_all.detach()
    else:
        with torch.no_grad() if more > 0 or w['value'] == 0 else contextlib.nullcontext():
            z, _ = run(critic.steps, a.lean, obs[:, None], prev.detach()[:, None], u1[:, None], hc, (true, lane.cknown), F)
        z = z[:, 0]
    if more > 0:                                                  # the teacher/hold takes the plant over from k1 (replaces z at k1: no double-counting; rec['z'] keeps k1's z for TD)
        acc, wsum, disc = handed(prog, st, k1, more, a, acc, wsum, disc)
        tot = tot+acc
    else:
        tot, wsum = tot+acc+w['value']*disc*value_of(z, a), wsum+w['value']*disc*tail
    for q, x in (('obs', obs), ('prev', prev), ('act', u1), ('lab', true[:, 0]))+(() if priv else (('z', z),)):
        rec[q].append(x.detach())
    rec = {q: torch.stack(x, 1) if isinstance(x, list) else x for q, x in rec.items()}
    rec['feats'] = torch.stack(fs, 1) if fs else None
    sim_m = (~lab).float()
    ep_loss = tot/(tot.detach()+a.floor*wsum)
    focal_alpha = float(getattr(a, 'focal_alpha', 0.))
    if focal_alpha > 0:
        fw = (tot.detach()/(a.floor*wsum).clamp_min(1e-6)).clamp(.5, 3.).pow(focal_alpha)
        fw = fw/masked_mean(fw, sim_m).clamp_min(1e-6)
        l_sim = masked_mean(ep_loss*fw, sim_m)
    else:
        l_sim = masked_mean(ep_loss, sim_m)                       # every episode relative to its own level
    loss = simw*l_sim+w['bc']*a.bc_coef*l_bc/H+(a.id_coef*l_id/H if need_read else 0.)
    rel_n_val = float(rel_n)
    if rel_n_val > 0:
        loss = loss+a.rel_id_coef*(l_rel/rel_n_val)
    log = dict(sim=_scalar(l_sim) if nsim else None, cost=float(rec['cost'].mean()), bc=_scalar(l_bc)/H if taught else None, id=_scalar(l_id)/H,
               id_own=_scalar(own)/H, id_cross=_scalar(cross)/H, steps=n*H)
    if rel_n_val > 0:
        rs = rel_stats.tolist()
        log.update(rel_id=_scalar(l_rel)/rel_n_val, rel_live=rs[0]/max(rs[1], 1.), rel_sign=rs[2]/max(rs[3], 1.))
    if lane.unpaired is not None:
        log.update({'id_'+q: _scalar(x)/H for q, x in split.items()})
    moved = moved.tolist()
    log.update(act_rms=(moved[0]/max(moved[3], 1.))**.5, act_sat=moved[1]/max(moved[3], 1.), act_step=(moved[2]/max(moved[3], 1.))**.5)
    if keep:
        loss = loss+a.law_coef*sum(x for x in l_law.values() if torch.is_tensor(x))/H
        log.update({q: _scalar(x)/H for q, x in l_law.items() if torch.is_tensor(x) or q in ('k_own', 'k_cross', 'head_lin', 'head_mech')})
    lane.st, lane.ha, lane.hc, lane.hs, lane.prev, lane.pm, lane.k = C.detach(st), ha.detach(), hc.detach(), hs, prev.detach(), pm.detach(), k1
    return loss, rec, guard, log


def fail_above(fail, cost, worst, fail_cost):
    """The objective of a decision and the largest paid so far while the unit ran -> (n,), (n,). A unit that failed
    pays --fail-cost on top of the most it ever paid running: a failure is never the cheaper way out of a row far
    outside (one zone row 4 bands out on a unit with one row is 16 a decision; 11-14)."""
    worst = torch.maximum(worst, torch.where(fail, torch.zeros_like(cost), cost.detach()))
    return torch.where(fail, fail_cost+worst, cost), worst


def handed(prog, st, k, steps, a, acc, wsum, disc):
    """The plant in the teacher's hands for `steps` decisions from the state a window ended in (--after): what
    that costs, discounted on from the window's own -> acc, wsum, disc. Nothing is learned for it and no network
    is run: it has the plant's own answer in it and cannot raise itself, as a value fitted to itself at the end
    of an episode does (11-12: along a stay the critic's value was 20 to 40 times the cost then paid, and 98%
    of the objective). The teacher's moves are worked out without a graph (core.teacher), so what comes back to
    the state is the cost of a deviation carried through the plant under those moves, not under the teacher's
    answer to it. --after-by hold: in nobody's hands instead, the MVs left where the window's last move put
    them (what a DMC's horizon takes for granted after its last move): a level still running at the window's
    end runs out of its band there and is paid for, and the gradient of that is exact (11-12: the distance to
    the DMC with the true model is in the plants with a level, 40 points against 87, more than half of it later
    than 64 decisions after an event). The plant's random draws are put back: the episode goes on from the same ones."""
    c, g, hold = prog.c, prog.g.get_state(), getattr(a, 'after_by', 'teacher') == 'hold'
    worst = torch.zeros_like(acc)
    Fr = prog.Fr
    if hold:                                                      # nobody predicts anything here: the teacher's model is not kept up along the way (its state is dropped with `st`)
        prog.Fr = None
    for j in range(k, k+steps):
        st, out = C.step(prog, st, torch.zeros(prog.n, prog.nmv, dtype=prog.dtype, device=prog.device) if hold else C.teacher(prog, st, j).to(prog.dtype), j,
                         stops=getattr(a, 'stops', False))
        cost, worst = fail_above(out['fail'], OR.objective(out, st, c, a.tau, a.cost_w, a.log_w).float(), worst, a.fail_cost)
        acc, wsum, disc = acc+disc*cost, wsum+disc, disc*a.gamma
    prog.Fr = Fr
    prog.g.set_state(g)
    return acc, wsum, disc


def top_level(a):
    return {'cut': max(int(math.log2(max(a.window//max(a.window_min, 1), 1))), 0), 'open': 1, 'cap': 0}[a.guard]


def adapt(lane, guard, a):
    """After a window: an episode whose gradient was held gets shorter stretches (one level up); after --relax
    windows without, it gets one level back."""
    held = guard.held()
    quiet = torch.where(held, torch.zeros_like(lane.quiet), lane.quiet+1)
    back = quiet >= a.relax
    lane.quiet = torch.where(back, torch.zeros_like(quiet), quiet)
    lane.level = (lane.level+held.long()-back.long()).clamp(0, top_level(a))


def _critic_items(critic, lane, rec, a, part=None):
    """Prepare one or more (slim_rec, known, tgt, said, chunk) tuples for `fit_to` / `fit_to_multi`."""
    if lane.due is not None:
        lane.due.add(rec, part)
        items = []
        while lane.due.ready():
            drec, dpart, tgt, said = lane.due.pop()
            items.append((drec, lane.cknown if dpart is None else lane.cknown[dpart], tgt, said, chunk_of(a, lane)))
        return items
    priv = bool(getattr(critic, 'privileged', False))
    keys = ('priv',) if priv else ('obs', 'prev', 'act', 'lab', 'hc', 'feats')
    if part is not None:
        slim = {q: (tuple(t[part] for t in rec[q]) if q == 'priv' else (rec[q][:, part] if q == 'hc' else rec[q][part]) if torch.is_tensor(rec[q]) else rec[q])
                for q in keys if q in rec}
        cost, zrec = rec['cost'][part], rec['z'][part]
    else:
        slim = {q: rec[q] for q in keys if q in rec}
        cost, zrec = rec['cost'], rec['z']
    with torch.no_grad():
        z_boot = zrec.mean(-1) if priv else zrec.amax(-1)
        v = symexp(z_boot).clamp_min(0.)
        tgt = symlog(td_targets(cost, v, a.gamma, a.lam))[..., None]
        said = float(v.mean())
    return [(slim, lane.cknown[:tgt.shape[0]], tgt, said, chunk_of(a, lane))]


def fit_critic(critic, copt, lane, rec, a, part=None):
    """The critic on the window just run: both heads towards the TD(lambda) targets built from the costs and
    the more pessimistic head's values (as they were before this fit), in symlog. The episodes go through
    a.fit_chunk at a time (their gradients added up: the same step, the memory of one part; `chunk_of`: as many
    pairs at a time whatever the lane's size). part: the episodes it is fitted on (None: all).
    --fixed: heads of fixed horizons (`fixed_targets`). The window waits for the decisions after it (`Due`), and
    what is fitted here is every window of the lane that has them by now (none: nothing is returned)."""
    if lane.due is not None:
        lane.due.add(rec, part)
        return fit_due(critic, copt, lane, a)
    if part is not None:
        rec = {q: (tuple(t[part] for t in x) if q == 'priv' else (x[:, part] if q == 'hc' else x[part]) if torch.is_tensor(x) else x) for q, x in rec.items()}
    with torch.no_grad():
        z_boot = rec['z'].mean(-1) if getattr(critic, 'privileged', False) else rec['z'].amax(-1)
        v = symexp(z_boot).clamp_min(0.)
        tgt = symlog(td_targets(rec['cost'], v, a.gamma, a.lam))
    return dict(value=float(v.mean()), **fit_to(critic, copt, lane.cknown[:tgt.shape[0]], rec, tgt[..., None], a, chunk_of(a, lane)))


def fit_due(critic, copt, lane, a, last=False):
    """--fixed: the critic on the windows of a lane that have the decisions after them. last: the lane is released,
    and all that are left are fitted (`Due.last`)."""
    out = {}
    if last:
        lane.due.last()
    while lane.due.ready():
        rec, part, tgt, said = lane.due.pop()
        out = dict(value=said, **fit_to(critic, copt, lane.cknown if part is None else lane.cknown[part], rec, tgt, a, chunk_of(a, lane)))
    return out


def _critic_feats(critic, rec, H):
    priv = bool(getattr(critic, 'privileged', False))
    feats = rec.get('feats')
    if not priv and feats is None and getattr(critic, 'pairs', None) is not None and critic.pairs.stats is not None:
        with torch.no_grad():
            ncv, nmv = N.shape_of(rec['obs'].shape[-1])
            base, spair = N.base_size(ncv, nmv), N.sizes(ncv, nmv)[7]
            ns = critic.pairs.stats.size_of(ncv, nmv)
            hs0 = rec['hc'][0, :, base+spair:base+spair+ns]
            feats, _ = critic.pairs.stats(rec['obs'][:, :H], hs0)
    return feats


def fit_to(critic, copt, known, rec, tgt, a, chunk=None):
    """--critic-iters steps of the critic on a window's record towards symlog targets (n, H, heads or 1). chunk: the
    episodes that go through at a time (None: --fit-chunk; `chunk_of`)."""
    n, H = tgt.shape[:2]
    chunk = a.fit_chunk if chunk is None else chunk
    parts = [slice(i, i+chunk) for i in range(0, n, chunk)] if chunk else [slice(None)]
    priv = bool(getattr(critic, 'privileged', False))
    feats = _critic_feats(critic, rec, H)
    out = {}
    for i in range(a.critic_iters):
        copt.zero_grad(set_to_none=True)
        loss, zs = 0., []
        for p in parts:
            if priv:
                z = critic(*(x[p, :H] for x in rec['priv']))
            else:
                z, _ = critic.steps(rec['obs'][p, :H], rec['prev'][p, :H], rec['act'][p, :H], rec['hc'][:, p], (rec['lab'][p, :H], known[p]),
                                    None if feats is None else feats[p])
            part = (z-tgt[p]).square().sum()/(n*H*z.shape[-1])
            if torch.isfinite(part):
                part.backward()
            loss, zs = loss+float(part.detach()), zs+[z.detach() if tgt.shape[-1] > 1 else z.detach().mean(-1, keepdim=True)]
        if i == 0:
            out.update(critic=loss, ev=float(1-(torch.cat(zs)-tgt).var()/tgt.var().clamp_min(1e-8)))
        if not math.isfinite(loss):
            copt.zero_grad(set_to_none=True)
            break
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(critic.parameters(), a.clip)):
            copt.step()
    return out


def fit_to_multi(critic, copt, items, a):
    """Deferred critic fit across multiple accumulated windows (--accum > 1): averages critic gradients over all
    buffered windows and steps `copt` once per critic iteration (exact 1:1 critic-to-actor parity with accum=1)."""
    if not items:
        return {}
    if len(items) == 1:
        rec, known, tgt, said, chunk = items[0]
        return dict(value=said, **fit_to(critic, copt, known, rec, tgt, a, chunk))
    priv = bool(getattr(critic, 'privileged', False))
    prep = []
    for rec, known, tgt, said, chunk in items:
        n, H = tgt.shape[:2]
        ck = a.fit_chunk if chunk is None else chunk
        parts = [slice(i, i+ck) for i in range(0, n, ck)] if ck else [slice(None)]
        prep.append((rec, known, tgt, said, n, H, parts, _critic_feats(critic, rec, H)))
    m = len(prep)
    out = {}
    for i in range(a.critic_iters):
        copt.zero_grad(set_to_none=True)
        loss, zs, tgts = 0., [], []
        for rec, known, tgt, _, n, H, parts, feats in prep:
            for p in parts:
                if priv:
                    z = critic(*(x[p, :H] for x in rec['priv']))
                else:
                    z, _ = critic.steps(rec['obs'][p, :H], rec['prev'][p, :H], rec['act'][p, :H], rec['hc'][:, p], (rec['lab'][p, :H], known[p]),
                                        None if feats is None else feats[p])
                part = (z-tgt[p]).square().sum()/(n*H*z.shape[-1]*m)
                if torch.isfinite(part):
                    part.backward()
                loss += float(part.detach())
                if i == 0:
                    zs.append((z.detach() if tgt.shape[-1] > 1 else z.detach().mean(-1, keepdim=True)).reshape(-1))
                    tgts.append(tgt[p].reshape(-1))
        if i == 0:
            zc, tc = torch.cat(zs), torch.cat(tgts)
            out.update(value=sum(x[3] for x in prep)/m, critic=loss, ev=float(1-(zc-tc).var()/tc.var().clamp_min(1e-8)))
        if not math.isfinite(loss):
            copt.zero_grad(set_to_none=True)
            break
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(critic.parameters(), a.clip)):
            copt.step()
    return out


GAUGES = ('act_rms', 'act_sat', 'act_step', 'gn_in', 'gn_lo', 'gn_mid', 'gn_hi', 'gn_out', 'w_out', 'x_norm')


def parts(net):
    """Gauges of a step of the --seq network (before the clip): the gradient's norm on what embeds the packet (gn_in), on
    the lower, middle and upper third of its layers (gn_lo, gn_mid, gn_hi) and on the head the moves are read from
    (gn_out), the norm of that head's last layer (w_out: 0 at the start, the network makes no move), and the residual
    stream norm before the final LayerNorm (x_norm). {} of another."""
    layers = getattr(net.actor, 'layers', None)
    if layers is None or not hasattr(net.actor, 'act'):
        return {}
    L = len(layers)
    dev = next(net.parameters()).device
    z0 = torch.zeros((), device=dev, dtype=torch.float32)

    def of_t(mods):
        gs = [p.grad.detach().float().square().sum() for m in mods for p in m.parameters() if p.grad is not None]
        return torch.stack(gs).sum().sqrt() if gs else z0

    cut = (0, max(L//3, 1), max(2*L//3, 1), L)
    xn = getattr(net.actor, 'x_norm_last', None)
    vals = [of_t((net.actor.row, net.actor.mv)), of_t(layers[cut[0]:cut[1]]), of_t(layers[cut[1]:cut[2]]),
            of_t(layers[cut[2]:cut[3]]), of_t((net.actor.norm, net.actor.act)), net.actor.act[-1].weight.detach().float().norm()]
    keys = ('gn_in', 'gn_lo', 'gn_mid', 'gn_hi', 'gn_out', 'w_out')
    if xn is not None:
        vals.append(xn.detach().float() if torch.is_tensor(xn) else torch.tensor(float(xn), device=dev))
        keys = keys+('x_norm',)
    return dict(zip(keys, torch.stack(vals).tolist()))


def _actor_grad_norm(net):
    """Total unclipped L2 gradient norm of actor parameters (without modifying `.grad`)."""
    gs = [p.grad.detach() for p in net.parameters() if p.grad is not None]
    if not gs:
        return torch.zeros((), device=next(net.parameters()).device)
    return torch.stack([n.float() for n in torch._foreach_norm(gs)]).norm()


def _clip_actor_grads(net, opt, clip):
    """Clip actor gradients in-place (separate Muon/AdamW norms for MuonAdamW) -> pre-clip combined norm."""
    if isinstance(opt, MuonAdamW):
        mu_ps = [p for g in opt.param_groups if g.get('kind') == 'muon' for p in g['params'] if p.grad is not None]
        ad_ps = [p for g in opt.param_groups if g.get('kind') != 'muon' for p in g['params'] if p.grad is not None]
        dev = next(net.parameters()).device
        gn_m = torch.nn.utils.clip_grad_norm_(mu_ps, clip) if mu_ps else torch.zeros((), device=dev)
        gn_a = torch.nn.utils.clip_grad_norm_(ad_ps, clip) if ad_ps else torch.zeros((), device=dev)
        return (gn_m.float().square()+gn_a.float().square()).sqrt()
    return torch.nn.utils.clip_grad_norm_(net.parameters(), clip)


def gathered(net, opt, temp, of, ok, rel_probe=None, rel_popt=None, clip=1.):
    """--accum: accumulate UNCLIPPED finite/spike-checked gradients across `of` windows, average them, and run
    `clip_grad_norm_` ONCE right before `opt.step()` (and `rel_popt.step()`). On CUDA, keeps the accumulation
    buffer on CPU (saving 524 MB VRAM during backward). Always resets `temp['acc']` to `dict(g=None, n=0, k=0)`
    at cycle boundaries. Returns whether `opt.step()` was called."""
    acc = temp.setdefault('acc', dict(g=None, n=0, k=0))
    ps = [q for q in net.parameters() if q.requires_grad]
    rps = [q for q in rel_probe.parameters() if q.requires_grad] if (rel_probe is not None and rel_popt is not None) else []
    offload = bool(ps and ps[0].is_cuda)
    if ok:
        if acc['g'] is None:
            acc['g'] = [None if q.grad is None else (q.grad.detach().cpu() if offload else q.grad.detach().clone()) for q in ps]
        else:
            for i, q in enumerate(ps):
                if q.grad is not None:
                    g = q.grad.detach().cpu() if offload else q.grad.detach()
                    if acc['g'][i] is None:
                        acc['g'][i] = g if offload else g.clone()
                    else:
                        acc['g'][i].add_(g)
        if rps:
            rg = acc.get('rg')
            if rg is None:
                acc['rg'] = [None if q.grad is None else q.grad.detach().clone() for q in rps]
            else:
                for i, q in enumerate(rps):
                    if q.grad is not None:
                        if rg[i] is None:
                            rg[i] = q.grad.detach().clone()
                        else:
                            rg[i].add_(q.grad.detach())
        acc['n'] += 1
    acc['k'] += 1
    stepped = False
    if acc['k'] >= of:
        if acc['n']:
            for q, g in zip(ps, acc['g']):
                q.grad = None if g is None else (g.to(q.device, non_blocking=True) if offload else g).div_(acc['n'])
            gn_acc = _clip_actor_grads(net, opt, clip)
            if bool(torch.isfinite(gn_acc)):
                opt.step()
                stepped = True
            rg = acc.get('rg')
            if stepped and rps and rg is not None:
                for q, g in zip(rps, rg):
                    q.grad = None if g is None else g.div_(acc['n'])
                if bool(torch.isfinite(torch.nn.utils.clip_grad_norm_(rps, clip))):
                    rel_popt.step()
        acc.pop('rg', None)
        acc.update(g=None, n=0, k=0)
    elif offload:
        opt.zero_grad(set_to_none=True)
    return stepped


def update(net, critic, opt, copt, lane, a, w, gen, temp):
    """One window of a lane: the actor's step, the horizons' adaptation, the critic's fit. temp: dict with usual
    (the guard's reference), moved on here; before_fit, where it has one: called between the actor's step and the fit."""
    for q in critic.parameters():                                 # the actor's objective passes through the critic, not into it
        q.requires_grad_(False)                                   # (until after the backward pass: --lean runs the critic again there)
    rel_probe, rel_popt = temp.get('rel_probe'), temp.get('rel_popt')
    accum = int(getattr(a, 'accum', 1) or 1)
    stepped = False
    try:
        loss, rec, guard, log = window(net, critic, lane, a, w, gen, temp['usual'], rel_probe=rel_probe)
        opt.zero_grad(set_to_none=True)
        if rel_popt is not None:
            rel_popt.zero_grad(set_to_none=True)
        ok = bool(torch.isfinite(loss))
        if ok:
            loss.backward()
            if hasattr(net, 'settle_rooms'):
                net.settle_rooms()
            log.update(parts(net))
            gn = _clip_actor_grads(net, opt, a.clip) if accum <= 1 else _actor_grad_norm(net)
            ok = bool(torch.isfinite(gn))                         # a gradient that is not finite is dropped, never stepped on
            if ok:
                if w['sim'] > 0 and guard.median() is not None:
                    # the simulator gradient of a few episodes can be many orders above the rest. The first window with
                    # it only measures the guard's level (nothing is held there yet) and is not stepped on; later, a
                    # step whose norm stands far above the recent ones (--spike times the median of the last 32) is
                    # one episode's, not the batch's, and is not taken either
                    past, spike = temp.setdefault('gns', []), getattr(a, 'spike', 0.)
                    mid = sorted(past)[len(past)//2] if past else None
                    far = spike > 0 and len(past) >= 8 and float(gn) > spike*mid
                    ok = temp['usual'] is not None and not far
                    # a step not taken counts as the most a step may be: the median moves up by at most --spike a step (11-23: the
                    # spikes themselves went into it, and in sq-big-s1 the guard's level ratcheted 10,616-fold over rows 122-139)
                    past.append(min(float(gn), spike*mid) if far else float(gn))
                    del past[:-32]
                log['gn' if ok else 'far'] = float(gn)            # gn: of the steps taken; far: of one that was not
            if ok and accum <= 1:
                opt.step()
                if rel_popt is not None:
                    torch.nn.utils.clip_grad_norm_(rel_probe.parameters(), a.clip)
                    rel_popt.step()
                stepped = True
        if accum > 1:
            stepped = gathered(net, opt, temp, accum, ok, rel_probe=rel_probe, rel_popt=rel_popt, clip=a.clip)
    finally:
        if hasattr(net, 'settle_rooms'):
            net.settle_rooms()
        for q in critic.parameters():
            q.requires_grad_(True)
    log['updates'] = 1 if stepped else 0
    log['skip'] = 0. if ok else 1.
    log['bad'] = 0. if bool(torch.isfinite(loss)) and ('gn' in log or 'far' in log) else 1.   # a loss or a gradient that was not finite
    log.update(capped=float(guard.held().float().mean()), peak=guard.peak(), short=float((lane.level > 0).float().mean()),
               hor=float((a.window >> lane.level).float().mean()) if a.guard == 'cut' else float(a.window))
    if w['sim'] > 0:                                              # (a window of imitation episodes alone has no median: nothing moves)
        m = guard.median()                                        # the level the gradient is held to: a window whose step was not taken, or one many times the level,
        if m is not None and temp['usual'] is not None and (not ok or m > 2*temp['usual']):   # moves it by at most a doubling (11-23: the ratchet above)
            m = min(m, 2*temp['usual'])
        temp['usual'] = running(temp['usual'], m)
    adapt(lane, guard, a)
    if 'before_fit' in temp:                                      # (--bench: the window's peak apart from the fit's)
        temp['before_fit']()
    if accum <= 1:
        log.update(fit_critic(critic, copt, lane, rec, a))
    else:
        acc = temp['acc']
        items = _critic_items(critic, lane, rec, a)
        if acc['k'] == 0:                                         # end of accum cycle: pop buffered critic batches and step ONLY if actor stepped
            buf = acc.pop('_crit', [])
            buf.extend(items)
            if stepped and buf:
                log.update(fit_to_multi(critic, copt, buf, a))
        elif items:
            acc.setdefault('_crit', []).extend(items)
    return log


def ppo(net, critic, opt, copt, lane, a, w, gen):
    """--ppo: one window of a lane without a teacher and without the plant's gradient (Schulman et al. 2017). The
    window is run with moves drawn around the actor's means (--noise, before the tanh) and no graph; the critic, not
    given the move, says the cost to come from each state; the advantage of a move is what it cost more than that
    (GAE, --gamma and --lam), every episode relative to its own level as the simulator's objective is; then
    --ppo-epochs passes over the window's episodes, --fit-chunk at a time, each a step on the clipped surrogate
    (--ppo-clip), stopped where the policy has moved by more than --ppo-kl; last the critic's fit."""
    h0 = lane.ha.detach()
    with torch.no_grad():
        _, rec, guard, log = window(net, critic, lane, a, dict(w, sim=0., bc=0.), gen, None, later=chunk_of(a, lane) or lane.prog.n)
        assert 'tch' not in rec, '--ppo: no episode is the teacher\'s (--bc-iters 0)'
        n, H = rec['cost'].shape
        v = symexp(rec['z'].mean(-1)).clamp_min(0.)                                         # (n,H+1) the cost to come
        delta, adv, last = rec['cost']+a.gamma*v[:, 1:]-v[:, :-1], torch.zeros_like(rec['cost']), 0.
        for j in reversed(range(H)):
            last = delta[:, j]+a.gamma*a.lam*last
            adv[:, j] = last
        adv = adv/((adv+v[:, :-1]).mean(1, keepdim=True).clamp_min(0.)+a.floor*reach(a))    # relative to the episode's own level
        alive = (rec['mk'].amax(-1) > 0).float()                                            # (n,H) decisions of an episode that has not failed
        cnt = alive.sum().clamp_min(1.)
        mu = (adv*alive).sum()/cnt
        adv = (adv-mu)/(((adv-mu).square()*alive).sum()/cnt).sqrt().clamp_min(1e-8)
        zs = torch.atanh(rec['u'].clamp(-1+1e-6, 1-1e-6))                                    # the moves before the tanh
        old = ((zs-rec['mean']).square()*rec['mk']).sum(-1)
    chunk, sd2 = chunk_of(a, lane) or n, 2.*a.noise**2
    steps, kls, clipped, gns, l_id, stop = 0, [], [], [], [], False
    for _ in range(a.ppo_epochs):
        order = torch.randperm(n, generator=gen, device=gen.device).to(h0.device)
        for i in range(0, n, chunk):
            p = order[i:i+chunk]
            mean, _, _, read, _ = net.steps(rec['obs'][p, :H], rec['prev'][p, :H], h0[:, p], (rec['lab'][p, :H], lane.known[p]), None if rec['feats'] is None else rec['feats'][p])
            logr = (old[p]-((zs[p]-mean).square()*rec['mk'][p]).sum(-1))/sd2
            ratio = logr.exp()
            kl = float(((ratio-1-logr)*alive[p]).sum()/alive[p].sum().clamp_min(1.))
            if not math.isfinite(kl) or kl > 1.5*a.ppo_kl:        # the policy has moved as far as the window's moves say anything about
                stop = True
                break
            surr = torch.maximum(ratio*adv[p], ratio.clamp(1-a.ppo_clip, 1+a.ppo_clip)*adv[p])       # a cost: the worse of the two
            pairs = lane.rows[p][:, None, :, None]*rec['mk'][p][:, :, None, :]
            err = masked_mean((N.taught(net, read, 'S')-N.taught(net, rec['lab'][p, :H], 'S')).square().mean(-1), pairs)
            loss = (surr*alive[p]).sum()/alive[p].sum().clamp_min(1.)+a.id_coef*err
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip)
            if bool(torch.isfinite(gn)):
                opt.step()
                steps += 1
                gns.append(float(gn))
            kls.append(kl); l_id.append(float(err))
            clipped.append(float((((ratio-1).abs() > a.ppo_clip).float()*alive[p]).sum()/alive[p].sum().clamp_min(1.)))
        if stop:
            break
    mid = lambda x: sum(x)/len(x) if x else None                  # noqa: E731
    log.update(sim=None, gn=mid(gns), skip=0. if steps else 1., updates=steps, kl=mid(kls), clipped=mid(clipped), capped=0., peak=None, short=0., hor=float(a.window))
    log.update(fit_critic(critic, copt, lane, rec, a))
    return log


def moved(prog, dev):
    """A batch of plants drawn for the CPU (by a builder process), on `dev` as if drawn for it."""
    dev = torch.device(dev)

    def to(o):
        if torch.is_tensor(o):
            return o.to(dev)
        if type(o) is dict:
            return {q: to(v) for q, v in o.items()}
        return type(o)(to(v) for v in o) if type(o) in (list, tuple) else o

    for q, v in list(vars(prog).items()):
        if q not in ('mg', 'tail'):                               # (tail: for the builder of these units' next episode, left where it was drawn)
            setattr(prog, q, to(v))
    prog.device = dev
    if prog.mech is not None:                                     # the engines of the mechanistic episodes, as made for this device
        prog.mg = prog.mech.runtime(prog.nl, dev, prog.dtype)
    return prog


def kind(dev):
    """'cpu' or 'cuda' of a device however it is written."""
    return torch.device(dev).type


@contextlib.contextmanager
def visit(dev, *progs):
    """Batches of plants on `dev` for a use, and back where they were after it (`moved`). What a run reads now and then
    stays off the card between its readings: the validation batches (--val-n, --val-sizes, every window of --val-stay
    and of the exams) are 2 MB a plant of the default size, several GB together, for a reading once a row, and a batch
    that is always there leaves that much less to the windows (11-13, a 16 GB card: three sizes taking turns did not
    fit it with anything kept on it)."""
    was = [kind(p.device) for p in progs]
    for p, d in zip(progs, was):
        if d != kind(dev):
            moved(p, dev)
    try:
        yield
    finally:
        for p, d in zip(progs, was):
            if d != kind(p.device):
                moved(p, d)


def parked(*progs):
    """Batches built on a card (their references run there), kept on the CPU from here on: `visit` brings them back for a use."""
    for p in progs:
        if kind(p.device) != 'cpu':
            moved(p, 'cpu')


def _builder(threads, qos):
    """At the start of a builder process: its CPU threads and where they run (--builder-qos, as --cpu-qos)."""
    torch.set_num_threads(threads)
    if qos != 'off':
        from monitoring import cpu_qos
        cpu_qos.apply(*qos.split(','))                           # never throttled: a builder that falls behind makes the GPU wait


SHAPE = ('nm', 'nd', 'nq', 'nz', 'nr')                           # counts a batch's plants can be kept to (aio.sim.prior.draw_roots): MVs, measured disturbances, unmeasured loads, zone rows, spare MVs
GO_ON = ('menu', 'nseg', 'blind')                               # of what a batch is built with, what the next windows of its plants are built with too


def menu_kw(items, nseg=0, always=False):
    """--menu rate=2,2,1,1,1 pace=2 turn=.5 drop=dead nd=1,2 -> what aio.sim.core.build is given for it: {} (the
    generator as it is, bit for bit), or menu=aio.sim.prior.Menu (with nseg where set), shape={root: (lo, hi)},
    blind=True. The fields are Menu's: rate (five shares: setpoint moves, measured disturbances, unmeasured
    loads, process changes, the rest), drop (names, comma separated), the others one number each; then the counts
    of SHAPE as lo,hi (nd=1,2: every plant has one or two measured disturbances) and blind=1 (the packet without
    the measured disturbances). always: nseg is given where there is no menu too (the windows of the lanes that
    train: prior.NSEG_STAY leaves a window that carries on room for its events to the end)."""
    menu, shape, kw = {}, {}, {}
    for it in items or ():
        q, v = it.split('=', 1)
        if q in SHAPE:
            shape[q] = tuple(int(x) for x in v.split(','))
            assert len(shape[q]) == 2, f'--menu: {q} takes lo,hi'
        elif q == 'stock':                                        # share of the plants with a conserving train of vessels (prior.STOCK)
            shape[q] = float(v)
        elif q == 'cls_p':                                        # 3-tuple probabilities for dead-time class (0, 1, 2)
            ps = tuple(float(x) for x in v.split(','))
            assert len(ps) == 3 and all(x >= 0 for x in ps) and sum(ps) > 0, '--menu: cls_p takes 3 non-negative probabilities'
            shape[q] = ps
        elif q in ('sep_p', 'integ_p'):                           # smooth prior overrides for separation arch & integrating levels
            pv = float(v)
            assert 0. <= pv <= 1., f'--menu: {q} must be in [0, 1]'
            shape[q] = pv
        elif q == 'blind':
            kw['blind'] = bool(int(v))
        else:
            assert q in PR.Menu._fields, f'--menu: {q} is not one of {PR.Menu._fields+SHAPE+("stock", "cls_p", "sep_p", "integ_p", "blind")}'
            menu[q] = tuple(x for x in v.split(',') if x) if q == 'drop' else tuple(float(x) for x in v.split(',')) if q == 'rate' else float(v)
    assert 'rate' not in menu or len(menu['rate']) == 5, '--menu: rate takes five shares'
    if menu:
        kw['menu'] = PR.Menu(**menu)
    if nseg and (menu or always):
        kw['nseg'] = nseg
    if shape:
        kw['shape'] = shape
    return kw


def mix_of(items):
    """--menu-mix 2/3:nd=1,2+rate=1,3,1,1,1 1/2:ride=.8+exc=.3 1/3:shift=.2 -> ((chance, (FIELD=VALUE, ..)), ..):
    groups of --menu's fields; every new batch of plants takes each group with that group's chance, one
    independently of the other."""
    out = []
    for it in items or ():
        p, fields = it.split(':', 1)
        p, fields = float(Fraction(p)), tuple(f for f in fields.split('+') if f)
        assert 0. <= p <= 1. and fields, f'--menu-mix: {it} is not CHANCE:FIELD=VALUE+FIELD=VALUE'
        menu_kw(fields)                                           # (the fields are --menu's: said now, not rows later)
        out.append((p, fields))
    return tuple(out)


def mixed(base, mix, seed):
    """The menu of the batch of plants with that seed -> (--menu's items, which groups came up as 0/1): `base`
    (--menu or --menu-stay) with the fields of the groups of `mix` that come up for it laid over (a later group
    over an earlier one). The draw is the seed's own: a batch has the same menu whatever was drawn before it,
    however far ahead the builders are, and in a run that resumes."""
    r = random.Random(f'menu-mix {seed}')
    on = tuple(int(r.random() < p) for p, _ in mix)
    return list(base or ())+[f for (_, fields), o in zip(mix, on) if o for f in fields], on


def groups_of(items):
    """--mech-groups 10:qtank,ph,vdv 1:column 1:rsr -> ((weight, (names, ..)), ..): the kinds of mechanistic
    plants a batch that has any takes together (the plants of aio.sim.plants, by name; no weight: 1)."""
    from aio.sim import plants as PL
    out = []
    for it in items or ():
        w, names = it.split(':', 1) if ':' in it else ('1', it)
        w, names = float(Fraction(w)), tuple(x for x in names.split(',') if x)
        assert w > 0 and names and all(q in PL.PLANTS or q in PL.LARGE for q in names), f'--mech-groups: {it} is not WEIGHT:NAME,NAME of {tuple(PL.PLANTS)+tuple(PL.LARGE)}'
        out.append((w, names))
    return tuple(out)


PLASTIC = 1.324717957244746                                       # the plastic number: x**3 = x+1


def evenly(k, d, salt):
    """The k-th point of an even sequence in [0, 1), coordinate d (0 or 1): the R2 sequence (Roberts 2018, steps of
    the powers of 1/PLASTIC), from a start of its own per salt. Any run of n of its points has close to n*w of them in
    any part of [0, 1) of width w, and the two coordinates are as even together: the sizes and the mechanistic plants
    of the new batches come in their shares all along a run. Drawn independently a share of 3 in 100 batches leaves a
    stretch of 50 without one about a fifth of the time; here never."""
    return (random.Random(f'even {salt} {d}').random()+(k+1)*PLASTIC**-(d+1)) % 1.


def pick(weights, u):
    """The index of the part of [0, 1) u falls in, the parts in proportion to the weights."""
    t, x = sum(weights), 0.
    for i, w in enumerate(weights):
        x += w/t
        if u < x:
            return i
    return len(weights)-1


def mech_of(share, chance, groups, seed, v=None):
    """The mechanistic plants of the batch of plants with that seed -> (the share of its episodes that are on
    them, their kinds or None for the pool's, the group's index or None). Without --mech-groups every batch has
    --mech of them, of all kinds. With it a batch has them with the chance --mech-batches, then --mech of its
    episodes, of the kinds of one group taken by the groups' weights: what the engines cost a decision goes
    with the kinds a batch has, not with how many plants are of them (11-12, the training machine: 502 ms a
    decision against 45 with all fifteen, at 64 plants as at 85), so the kinds come a few at a time. The draw
    is the seed's own, as --menu-mix's; v: a point of `evenly` it is taken by instead (below `chance` a group,
    the groups' parts in proportion to their weights)."""
    if not groups:
        return share, None, None
    if v is None:
        r = random.Random(f'mech {seed}')
        if r.random() >= chance:
            return 0., None, None
        g = r.choices(range(len(groups)), [w for w, _ in groups])[0]
    elif v >= chance:
        return 0., None, None
    else:
        g = pick([w for w, _ in groups], v/chance)
    return share, groups[g][1], g


def drawn(n, seed, dev='cpu', gen='cpu', names=None, envs=None, plant_seed=None, event_seed=None, **kw):
    """aio.sim.core.build; names: the kinds its mechanistic plants are of (`mech_of`; they are built in the process
    that draws the batch: a builder is given the names); envs: its plants, in place of n (--sizes: a batch of
    another size has as many of them as gives it about as many pairs). plant_seed/event_seed: mined hard plant
    dynamics paired with fresh event/noise schedules when Track A's `event_seed` is available."""
    if names:
        from aio.sim import mech as ME
        kw['plants'] = ME.pool(list(names))
    if plant_seed is not None:
        if event_seed is not None and _supports_event_seed():
            kw['event_seed'] = event_seed
        seed = plant_seed
    return C.build(envs or n, seed, dev, gen, **kw)


def expand_lag_b(v, old_lags, new_lags, shift=0.0):
    """Expand trained g1 lag_b (heads, len(old_lags)) to g2 (heads, len(new_lags)) by log-time linear
    interpolation between each head's own adjacent trained lags (plus `shift`). Preserves per-head lag
    specialization while avoiding the ~55x softmax gradient starvation caused by cold -4.5 init."""
    exp = v.new_empty(v.shape[0], len(new_lags))
    for j, tau in enumerate(new_lags):
        if tau in old_lags:
            exp[:, j] = v[:, old_lags.index(tau)]
        else:
            lo = max(i for i, t in enumerate(old_lags) if t < tau)
            hi = min(i for i, t in enumerate(old_lags) if t > tau)
            w = (math.log(tau) - math.log(old_lags[lo])) / (math.log(old_lags[hi]) - math.log(old_lags[lo]))
            exp[:, j] = (1. - w) * v[:, lo] + w * v[:, hi] + shift
    return exp


def expand_rel_in(v, new_f):
    """Expand trained r1 pairwise projection weights (..., REL_F) to r2 (..., REL_F2) by zero-padding the new
    physical log-gain columns so warm-started r1 -> r2 produces bit-identical actions at step 0."""
    out = v.new_zeros(*v.shape[:-1], new_f)
    out[..., :v.shape[-1]] = v
    return out


SQ_GROWN = {'n4', 'q_l', 'out_l', 'lag_b', 'lag_pair', 'lag_own', 'mem_n', 'mem_in', 'mem_out', 'mem_a',
            'mem_fb', 'edge_bias', 'edge_rn', 'edge_mn', 'edge_r', 'edge_m', 'edge_a',
            'rel_p', 'rel_a', 'rel_eta', 'rel_c', 'rel_rn', 'rel_mn',
            'rel_e_mv', 'rel_c_in', 'rel_mv_out', 'rel_e_cv', 'rel_m_in', 'rel_cv_out',
            'rel_bias', 'rel_lag_bias', 'rel_pipe_in', 'act_gate', 'rel_f2'}   # seqnet parts --init may add (zero start: the same moves)

DEFAULT = f'{IF.NCV}x{IF.NMV}'                                     # the size of a packet as the generator makes it unless asked (aio.sim.prior.SIZES)
BY_SIZE = ('bc', 'sim', 'cost', 'gain', 'k_own', 'k_cross', 'id_own', 'id_cross', 'id_own_told', 'id_own_blind')      # of a row's columns, those given per size too (--sizes: by_size)


def size_name(prog):
    return f'{prog.ncv}x{prog.nmv}'


def per_pairs(x, name):
    """x of the default size for a batch of that size: as many pairs (at least 1)."""
    ncv, nmv = PR.SIZES[name][:2]
    return max(1, round(x*IF.NCV*IF.NMV/(ncv*nmv)))


def chunk_of(a, lane):
    """--fit-chunk for a lane: the episodes of its size that are as many pairs (0: all at once). To the critic an
    episode of 48x20 is sixteen of the default size: a chunk counted in episodes was all of such a batch in one graph,
    and what `imitate` fits the critic on was all of it instead of a sixth. With --priv-critic (small non-recurrent
    Markov critic), default to full batch unless explicitly overridden to a smaller value."""
    if not a.fit_chunk:
        return 0
    if getattr(a, 'priv_critic', False):
        n = getattr(getattr(lane, 'prog', None), 'n', 0)
        if n and a.fit_chunk >= n:
            return 0
        return a.fit_chunk if getattr(a, '_fit_chunk_explicit', False) or (n and a.fit_chunk < n) else 0
    return per_pairs(a.fit_chunk, lane.size)


def sizes_of(items, envs):
    """--sizes 4:12x5 1:24x10 1:48x20@48 -> ((weight, size, plants), ..): every new batch of plants is of one size,
    taken by the weights (`size_of`); plants of a batch of that size: as given after @, else --envs times the pairs
    of the default size over its own (768, 192, 48: about as many pairs a batch). None: every batch is of the
    default size and of --envs plants, as before."""
    if items is None:
        return None
    out = []
    for it in items:
        w, rest = it.split(':', 1) if ':' in it else ('1', it)
        name, n = rest.split('@', 1) if '@' in rest else (rest, '')
        assert name in PR.SIZES and float(Fraction(w)) > 0, f'--sizes: {it} is not WEIGHT:SIZE[@PLANTS] of {tuple(PR.SIZES)}'
        out.append((float(Fraction(w)), name, int(n) if n else per_pairs(envs, name)))
    assert out and len({x[1] for x in out}) == len(out), '--sizes: each size once'
    return tuple(out)


def size_of(sizes, seed, u=None):
    """The size of the batch of plants with that seed and its plants, (name, n): the seed's own draw, as --menu-mix's;
    u: a point of `evenly` it is taken by instead."""
    if len(sizes) == 1:
        return sizes[0][1:]
    if u is not None:
        return sizes[pick([x[0] for x in sizes], u)][1:]
    r = random.Random(f'size {seed}')
    return r.choices([x[1:] for x in sizes], [x[0] for x in sizes])[0]


class Ahead:
    """--builders K: the batches of plants a run needs next, drawn ahead on the CPU by K processes. Drawing a
    batch of 1,024 is 35-50 s of Python and CPU tensor work on the training machine, once a row and in the
    run's own process: a third of every row with the GPU waiting (11-11), more with --life. Here the run
    only moves a drawn batch to its device. Batches are handed out in the order they were asked for, each
    with the next seed: the same plants however fast the builders are. seed(): the next seed."""

    def __init__(self, k, envs, seed, threads=8, qos='off'):
        import concurrent.futures as cf, multiprocessing as mp
        self.pool = cf.ProcessPoolExecutor(k, mp_context=mp.get_context('spawn'), initializer=_builder, initargs=(threads, qos))
        self.envs, self.seed, self.q = envs, seed, {}

    def __call__(self, mech=0., depth=1, **kw):
        """The next batch with that share of mechanistic episodes; `depth` more of the kind are then in the making.
        kw: what else aio.sim.core.build is given (`menu_kw`)."""
        q = self.q.setdefault((mech, repr(sorted(kw.items()))), collections.deque())
        while len(q) < depth+1:
            q.append(self.pool.submit(C.build, self.envs, self.seed(), 'cpu', 'cpu', mech=mech, **kw))
        return waited(q.popleft())

    def own(self, kind, kw_of, depth=1):
        """As a call, of batches each built its own way (--menu-mix, --mech-groups): kw_of(seed) -> (kw, tag) is
        asked when a batch is put in the making, kw is what `drawn` is given for it; kind names the queue (a
        lane's kind: its batches come in the order they were asked for). -> the batch, its kw and its tag."""
        q = self.q.setdefault(kind, collections.deque())
        while len(q) < depth+1:
            seed = self.seed()
            kw, tag = kw_of(seed)
            q.append((self.pool.submit(drawn, self.envs, seed, **kw), kw, tag))
        f, kw, tag = q.popleft()
        return waited(f), kw, tag

    def pids(self):
        return sorted(self.pool._processes or ())

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


class Average:
    """The actor's weights averaged over its steps (--ema D: exponentially, the last 1/(1-D) steps; fewer at the
    start, so the average is never behind by more than a tenth of the steps made). What a policy fitted to a
    teacher does in closed loop swings from one step of the optimizer to the next far more than its loss does
    (small changes of the weights add up along an episode; Block et al. 2024, "Butterfly effects of SGD noise");
    the average of the iterates does not. It is what is validated and kept as `model`; the steps go on from
    the weights as stepped (`raw` in a checkpoint). Called by the optimizer after every step."""

    def __init__(self, net, d, n=0, device=None):
        self.net, self.d, self.n = net, d, n
        self.avg = copy.deepcopy(net).requires_grad_(False)
        self.device = torch.device(device) if device is not None else next(net.parameters()).device
        if next(self.avg.parameters()).device != self.device:
            self.avg.to(self.device)

    @torch.no_grad()
    def update(self, *_):
        d = min(self.d, (1+self.n)/(10+self.n))
        self.n += 1
        w = 1.-d
        avg_ps = list(self.avg.parameters())
        net_ps = list(self.net.parameters())
        if avg_ps and avg_ps[0].device == net_ps[0].device:
            torch._foreach_lerp_(avg_ps, net_ps, w)
        else:
            for p, q in zip(avg_ps, net_ps):
                p.lerp_(q.detach().to(p.device), w)

    __call__ = update


class HardSeeds:
    """Mined hard plant seeds (--hard-seeds PATH, --hard-share P) with hot-reload on file mtime.
    Mixes mined `plant_seed`s with fresh `event_seed`s into non-plain 12x5 batches while preserving exact
    uniform anchor ratios alongside `--menu-mix`."""

    def __init__(self, path=None, share=0.):
        self.path = Path(path) if path else None
        self.share = float(share or 0.)
        self._mtime = -1.
        self.seeds = []
        self._idx = 0
        self.reload()

    @property
    def active(self):
        return self.path is not None and self.share > 0.

    def reload(self):
        if self.path is None or not self.path.exists():
            return self.seeds
        try:
            mt = self.path.stat().st_mtime
            if mt != self._mtime:
                data = json.loads(self.path.read_text(encoding='utf-8'))
                raw = data.get('seeds', []) if isinstance(data, dict) else data
                parsed = [int(x) for x in raw]
                if parsed:
                    self.seeds = parsed
                    self._mtime = mt
                    self._idx = self._idx % len(self.seeds)
        except (OSError, ValueError, TypeError):
            pass
        return self.seeds

    def pick(self, batch_seed, plain=False, is_default_size=True, menu_active=False, p_menu=0.):
        """Return a mined `plant_seed` or None. Never touches `--plain` anchor lanes, non-12x5 sizes, or batches
        already claimed by `--menu-mix`; scales conditional probability by `1 / (1 - p_menu)` so the marginal
        fractions (uniform anchor, parametric menu-mix, mined hard-seeds) match the configured shares exactly."""
        if plain or not is_default_size or self.share <= 0. or menu_active:
            return None
        self.reload()
        if not self.seeds:
            return None
        cond_p = min(1., self.share/max(1.-p_menu, 1e-6))
        if random.Random(f'hard-seeds {batch_seed}').random() >= cond_p:
            return None
        ps = self.seeds[self._idx % len(self.seeds)]
        self._idx += 1
        return ps


def pack_delta_int8(t, g=128):
    """Blockwise symmetric INT8 quantization (group size `g=128`, FP16 per-block scale) for MuLoCo pseudo-gradients."""
    x = t.detach().float().cpu().reshape(-1)
    numel = x.numel()
    pad = (-numel) % g
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    x = x.view(-1, g)
    s = x.abs().amax(-1).clamp_min(1e-8).to(torch.float16)
    q = (x/s.float()[:, None]*127.0).round().clamp(-127, 127).to(torch.int8)
    return dict(q=q, s=s, shape=tuple(t.shape), numel=numel, g=g)


def unpack_delta_int8(item):
    """Inverse of `pack_delta_int8` -> FP32 tensor of original shape."""
    q, s = item['q'], item['s']
    return (q.float()*(s.float()[:, None]/127.0)).reshape(-1)[:item['numel']].reshape(item['shape'])


def pack_actor_delta(anchor, current, compress='int8', g=128):
    """Compute pseudo-gradient `delta = anchor - current`, its L2 norm, finiteness flag, and packed transport dict."""
    packed, sq = {}, 0.
    finite = True
    for k, a_t in anchor.items():
        if k not in current or not a_t.is_floating_point():
            continue
        d = a_t.detach().float().cpu()-current[k].detach().float().cpu()
        if not bool(torch.isfinite(d).all()):
            finite = False
        sq += float(d.square().sum())
        if compress == 'int8':
            packed[k] = pack_delta_int8(d, g=g)
        else:
            packed[k] = d.to(torch.bfloat16) if d.ndim >= 2 else d.to(torch.float32)
    norm = math.sqrt(sq) if math.isfinite(sq) else float('inf')
    return packed, norm, finite and math.isfinite(norm)


def unpack_actor_delta(packed):
    """Decode a packed actor delta dict (`int8` dicts or `bf16`/`fp32` tensors) into FP32 tensors."""
    return {k: (unpack_delta_int8(v) if isinstance(v, dict) else v.detach().float().cpu()) for k, v in packed.items()}


class DiLoCoClient:
    """Elastic Equal-H MuLoCo worker client (--diloco-hub URL, --diloco-h H).
    Syncs pseudo-gradients `anchor - current` at exact row boundaries once `updates_since_sync >= H`, keeps
    local `MuonAdamW` and `PrivCritic` optimizer states intact, applies Hub-broadcast `inner_lr` and merged
    weights, and degrades gracefully to local training on any Hub network/timeout failure."""

    def __init__(self, hub_url, H, worker='', compress='int8', timeout=180., round_idx=0, hub_lr=None):
        self.hub_url = hub_url.rstrip('/')
        self.H = max(1, int(H))
        self.worker = worker or socket.gethostname()
        self.compress = compress
        self.timeout = float(timeout)
        self.round = int(round_idx)
        self.hub_lr = None if hub_lr is None else float(hub_lr)
        self.updates_since_sync = 0
        self.env_steps_since_sync = 0
        self.anchor = None

    def snapshot_anchor(self, net):
        self.anchor = {k: v.detach().float().cpu().clone() for k, v in net.actor.state_dict().items() if v.is_floating_point()}

    def record(self, updates, env_steps):
        self.updates_since_sync += int(updates)
        self.env_steps_since_sync += int(env_steps)

    def state_meta(self):
        return dict(round=self.round, hub_lr=self.hub_lr, worker=self.worker, H=self.H)

    def maybe_sync(self, net, critic, rel_probe=None, force=False):
        """Sync with Hub if `updates_since_sync >= H` (or `force=True`). Returns summary dict or None."""
        if self.anchor is None:
            self.snapshot_anchor(net)
        if not force and self.updates_since_sync < self.H:
            return None
        min_steps = int(.7*self.H)
        if self.updates_since_sync < min_steps:
            return dict(skipped='min_steps', steps=self.updates_since_sync, min_steps=min_steps)
        packed, norm, ok = pack_actor_delta(self.anchor, net.actor.state_dict(), compress=self.compress)
        if not ok:
            msg = dict(diloco_warn='non_finite_delta', worker=self.worker, round=self.round, steps=self.updates_since_sync)
            print(dumps(msg), file=sys.stderr, flush=True)
            if all(bool(torch.isfinite(v).all()) for v in net.actor.state_dict().values()):
                self.snapshot_anchor(net)
                self.updates_since_sync = 0
                self.env_steps_since_sync = 0
            return dict(skipped='non_finite', norm=None)
        req_dict = dict(
            worker=self.worker,
            round=self.round,
            steps=self.updates_since_sync,
            env_steps=self.env_steps_since_sync,
            compress=self.compress,
            actor_delta=packed,
            critic={k: v.detach().float().cpu() for k, v in critic.state_dict().items()} if critic is not None else None,
            rel_probe={k: v.detach().float().cpu() for k, v in rel_probe.state_dict().items()} if rel_probe is not None else None,
            norm=norm,
        )
        buf = io.BytesIO()
        torch.save(req_dict, buf)
        t0 = time.time()
        try:
            req = urllib.request.Request(f'{self.hub_url}/sync', data=buf.getvalue(),
                                         headers={'Content-Type': 'application/octet-stream'}, method='POST')
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw_resp = resp.read()
            reply = safe_torch_load(io.BytesIO(raw_resp), map_location='cpu')
        except Exception as exc:
            msg = dict(diloco_warn='hub_unreachable', worker=self.worker, round=self.round, error=repr(exc))
            print(dumps(msg), file=sys.stderr, flush=True)
            return dict(error=repr(exc), accepted=False, round=self.round)
        sync_s = round(time.time()-t0, 2)
        if reply.get('actor') is not None:
            net.actor.load_state_dict(reply['actor'])
            for m in net.modules():
                getattr(m, 'forget', lambda: None)()
        if reply.get('critic') is not None and critic is not None:
            critic.load_state_dict(reply['critic'])
        if reply.get('rel_probe') is not None and rel_probe is not None:
            rel_probe.load_state_dict(reply['rel_probe'])
        if reply.get('inner_lr') is not None:
            self.hub_lr = float(reply['inner_lr'])
        self.round = int(reply.get('round', self.round+1))
        steps_sent, env_sent = self.updates_since_sync, self.env_steps_since_sync
        self.updates_since_sync = 0
        self.env_steps_since_sync = 0
        self.snapshot_anchor(net)
        return dict(round=self.round, global_steps=reply.get('global_steps'), inner_lr=self.hub_lr,
                    accepted=bool(reply.get('accepted', True)), norm=round(norm, 4), steps=steps_sent,
                    env_steps=env_sent, sync_s=sync_s)


_RESUME_RESTORE_KEYS = (
    'ema', 'wd', 'muon_mom', 'beta2', 'rel_id_coef', 'unpaired',
    'meter', 'stops', 'fixed', 'heads', 'priv_critic', 'opt',
    'bias_lr_mult', 'knots', 'cosine_by', 'total_updates', 'total_samples',
    'focal_alpha', 'bc_iters', 'bc_floor', 'bc_hold', 'id_coef',
    'pre_cost', 'rev_smooth', 'smooth', 'floor', 'window_min', 'value', 'value_warm',
    'lanes', 'window', 'sim_window', 'replay', 'batch', 'batch_min', 'keep',
    'stay', 'after', 'after_by', 'cost_w', 'log_w', 'noise', 'faults', 'probe',
    'known', 'critic_known', 'gamma', 'lam', 'tau', 'fail_cost',
    'critic_lr', 'critic_iters', 'warm_steps', 'warmup', 'clip',
    'mech', 'mech_groups', 'mech_batches', 'menu', 'menu_mix', 'nseg',
    'builders', 'builder_threads', 'val_raw',
)


def restore_resume_config(a, ck, explicit=()):
    """Auto-restore structural & optimizer config from `ck['config']` / `pairs['seq']` on `--resume` unless explicitly overridden."""
    explicit = set(explicit or ())
    cfg = ck.get('config') or {}
    pairs = (ck.get('model') or {}).get('pairs') or {}
    if 'seq' in pairs or cfg.get('seq'):
        if 'seq' in pairs:
            from aio import seqnet as SQ
            asked, w = SQ.of_buffer(pairs['seq'])
            if 'seq' not in explicit:
                a.seq = asked
            if 'seq_window' not in explicit:
                a.seq_window = int(w)
        else:
            if 'seq' not in explicit:
                a.seq = cfg['seq']
            if 'seq_window' not in explicit and 'seq_window' in cfg:
                a.seq_window = int(cfg['seq_window'])
        if 'replay' not in explicit and cfg.get('replay'):
            a.replay = cfg['replay']
        if 'seq_ckpt' not in explicit and 'seq_ckpt' in cfg:
            a.seq_ckpt = bool(cfg['seq_ckpt'])
    for k in _RESUME_RESTORE_KEYS:
        if k in explicit:
            continue
        if k in ('cosine_by', 'total_updates', 'total_samples') and k in ck and ck[k] is not None:
            setattr(a, k, ck[k])
        elif k in cfg and cfg[k] is not None:
            setattr(a, k, cfg[k])
    return a


LAW = ('head', 'gain', 'valve', 'law', 'hbias', 'headf', 'k_own', 'k_cross', 'head_lin', 'head_mech')   # --law, in the rows: the errors of the readouts of E, K and V (as they are read) and of their product (asinh of the
                                                  # move in slew limits); hbias: the part of `head` that is a row's offset over the window.
                                                  # k_own, k_cross: `gain` on the episodes not told the responses, an MV's paired row and the others (`k_untold`).
                                                  # head_lin, head_mech: `head` on the plants of the prior and of the engines apart (`e_by_kind`).
                                                  # --given: head, gain, valve of the learned readouts on every episode; headf and hbias of the
                                                  # formula's E in the episodes told the responses; law of the path each episode takes


UNSEEN = ('unseen', 'gap_un', 'rate_un', 'rate_rk')+tuple(f'rate_{q}{lo}' for q in ('un', 'rk') for lo in (0, 16, 64, 128))
# --unseen, in the rows: the read-out's loss (Huber, in its units); gap_un: the unexplained share of the true value less the
# reading over the tracked rows; of the levels' rate now: rate_un the unexplained share, rate_rk the rank correlation, and
# both by the decisions since the last event that changed something (0-16, 16-64, 64-128, later). 11-12, the trunk before
# it was taught this: rank .77, unexplained 1.65; a slow filter of the readings with the true model: .87-.92 and .17-.22.


def huber1(x, y):
    """Huber's loss with its turn at 1, element by element: the read-out's units are those of its labels (aio.net.GAP_UNIT,
    RATE_UNIT: weighted bands), so what is common is squared and a spike, or a reading that is not there, counts by its size."""
    return torch.nn.functional.huber_loss(x, y, delta=1., reduction='none')


def ranks(x):
    return x.argsort().argsort().double()


@torch.no_grad()
def unseen_log(uh, U, since, lev, wt):
    """How well what the instruments do not show is read (the rows' UNSEEN but its first): uh, U (b, T, rows, 2) read and
    true, since (b, T) decisions since the last event, lev (b, rows) the tracked rows that integrate, wt (b, T, rows) the
    tracked rows at the decisions that count."""
    out = dict(gap_un=float(((uh[..., 0]-U[..., 0]).square()*wt).sum()/(U[..., 0].square()*wt).sum().clamp_min(1e-9)))
    r, t, on = uh[..., 1].double(), U[..., 1].double(), (wt*lev[:, None, :]) > 0
    for name, lo, hi in (('', 0, 1e9), ('0', 0, 16), ('16', 16, 64), ('64', 64, 128), ('128', 128, 1e9)):
        m = on & ((since >= lo) & (since < hi))[:, :, None]
        a, b = r[m], t[m]
        if a.numel() < 16 or float(b.square().sum()) <= 0:
            continue
        ra, rb = ranks(a), ranks(b)
        out['rate_un'+name] = float((a-b).square().sum()/b.square().sum())
        out['rate_rk'+name] = float(torch.corrcoef(torch.stack((ra, rb)))[0, 1].nan_to_num(0.))
    return out


class Buffer:
    """Windows gathered, as data: per episode and window the packets, the last moves, the teacher's moves, which
    of them count (`m`: MVs in service of episodes that have not failed, 0 past an episode's end), the true
    responses, the running sums' features (`feats`), and the network's state at the window's first decision
    (`h0`: with the running sums in their place, so the window can be run through `steps` at once). The oldest
    rows are written over. batch: the windows a step of `learn` draws from it (0: --batch; --sizes: a buffer of
    windows of one size, as many pairs a step whatever the size)."""

    def __init__(self, cap, batch=0):
        self.cap, self.x, self.n, self.at, self.batch = cap, None, 0, 0, batch

    def add(self, rows):
        first = next(iter(rows.values()))
        n = len(first)
        if self.x is None:
            self.x = {q: v.new_zeros((self.cap,)+v.shape[1:]) for q, v in rows.items()}
        i = (self.at+torch.arange(n, device=first.device)) % self.cap
        for q, v in rows.items():
            self.x[q][i] = v
        self.at, self.n = (self.at+n) % self.cap, min(self.n+n, self.cap)

    def draw(self, b, gen):
        i = torch.randint(self.n, (b,), generator=gen, device=next(iter(self.x.values())).device)
        return {q: v[i] for q, v in self.x.items()}

    def to(self, dev):
        """The rows on that device (`settle`: those of a size no lane is of wait on the CPU)."""
        if self.x is not None and kind(next(iter(self.x.values())).device) != kind(dev):
            self.x = {q: v.to(dev) for q, v in self.x.items()}
        return self


def design_set(n, seed, nk, kw=None, unpaired=0., meter=False):
    """n plants drawn and not run, each a problem of design (--design): the packet of its first decision after the
    reset (in the frame the network sees), the true responses the network is told with it, and the teacher's K of
    the plant's first model (aio.net.law_code); `pairs`: the tracked rows by the MVs in service, where K counts.
    kw: what else aio.sim.core.build is given. unpaired: the share of them with nothing published about the
    pairing, in the plant's own frame (--unpaired: as the episodes; which ones is the seed's own draw). On the
    CPU (a builder process), in half precision."""
    prog = C.build(n, seed, 'cpu', 'cpu', **(kw or {}))
    set_teacher(prog, meter)                                      # (K is the design of the model: the same either way)
    st = C.reset(prog)
    un = torch.rand(n, generator=torch.Generator().manual_seed(seed % (1 << 31))) < unpaired if unpaired > 0 else None
    obs, s = canonical(C.observe(prog, st, 0).float(), un)
    c, nmv = prog.c, prog.nmv
    return dict(obs=obs.half(), lab=N.pair_labels(prog.labels(0, nk).float()*s[:, :, None]).half(), K=N.law_code(prog, st, 0, s)[1][:, :nmv].half(),
                pairs=((c['trk'][:, :nmv, None] > 0) & (c['mvm'][:, None, :nmv] > 0)).half())


def design_error(net, x):
    """The K read-out's error on problems of design x (`design_set`): the network is told the responses and nothing
    of the law, has no memory and has made no move; what it reads as K against the teacher's, as `gain` is. The
    running sums' features of the packet, where x has them (`Designs`: worked out once as a set comes in, as a
    gathered window's are), are not worked out again at every step (their solves are most of its cost)."""
    b, nmv = len(x['obs']), N.shape_of(x['obs'].shape[-1])[1]
    feats = x['feats'][:, None].float() if 'feats' in x else None
    net.steps(x['obs'][:, None].float(), x['obs'].new_zeros(b, 1, nmv, dtype=torch.float32), None, (x['lab'][:, None].float(), torch.ones(b, device=x['obs'].device)), feats)
    return masked_mean((net.pairs.read_law[1][:, 0, :nmv].float()-x['K'].float()).square().mean(-1), x['pairs'].float())


class Designs:
    """--design B: problems of design, of plants that are not run. The teacher's K is the design of a plant's model
    (aio.law.tables): where the plant is has no part in it. An episode gives one such problem per model for its
    720 decisions of plant and network (about one new plant per step of learning); a plant drawn and not run
    gives one for .03 CPU seconds. The builders draw sets of --design-set of them in the time the batches leave
    (`ahead` sets in the making); the sets go into a pool of --design-keep plants, the oldest written over, and
    every step of `learn` takes B of them (`design` in the rows: the K read-out, told the responses, against the
    teacher's K). A set that arrives is read once before anything is learned from it: `design_new`, the error on
    plants never seen. Without builders a set is drawn in the run's own process, once a row (tests, --bench).
    seed(): of the next set; kw: what the plants are built with (a size has a pool of its own: --sizes, with b, n and
    keep scaled to its pairs, and ahead sets in the making by its share of the builders).
    --design-every K: one step of K takes them (`due`). A step's time on a card goes with how many operations it is
    more than with how much they hold, and the problems run the network a second time: B of them at every step were a
    third of a step for a sixteenth of its pairs (11-13, 768 plants: .087 s a step against .064 s without them)."""

    def __init__(self, a, pool, seed, dev, nk, kw=None, every=1, b=0, n=0, keep=0, ahead=0):
        self.b, self.n, self.pool, self.seed, self.dev = b or a.design, n or a.design_set, pool, seed, torch.device(dev)
        self.stride, self.steps = max(1, getattr(a, 'design_every', 1)), 0                    # --design-every; steps of learning so far
        self.args = (nk, kw, getattr(a, 'unpaired', 0.), bool(getattr(a, 'meter', False)))      # of `design_set`, behind the plants and the seed
        self.buf, self.q, self.new, self.every, self.calls = Buffer(keep or a.design_keep), collections.deque(), [], every, 0
        self.ahead, self.plants = ahead or max(1, a.builders-1), 0

    def take(self, net, x):
        x = {q: v.to(self.dev) for q, v in x.items()}
        with torch.no_grad():
            if net.pairs.stats is not None:                       # the running sums of the first packet, once (`design_error`)
                x['feats'] = net.pairs.stats(x['obs'][:, None].float())[0][:, 0].half()
            self.new.append(float(design_error(net, x)))
        self.buf.add(x)
        self.plants += len(x['obs'])

    def more(self, net):
        """Before a window's steps of learning: the sets that are ready go into the pool (the first is waited for)."""
        if self.pool is None:
            if self.buf.n < self.b or self.calls % self.every == 0:
                self.take(net, design_set(self.n, self.seed(), *self.args))
            self.calls += 1
            return
        while len(self.q) < self.ahead:
            self.q.append(self.pool.submit(design_set, self.n, self.seed(), *self.args))
        while self.q and (self.q[0].done() or self.buf.n < self.b):
            self.take(net, waited(self.q.popleft()))
            self.q.append(self.pool.submit(design_set, self.n, self.seed(), *self.args))

    def due(self):
        """Whether this step of learning takes problems: one step of --design-every, once the pool has a step's."""
        self.steps += 1
        return self.buf.n >= self.b and self.steps % self.stride == 0

    def loss(self, net, gen):
        return design_error(net, self.buf.draw(self.b, gen))

    def log(self):
        """Of the sets that arrived since the last call: their error before they were learned from, and how many plants the pool has had."""
        new, self.new = self.new, []
        return dict(design_new=sum(new)/len(new) if new else None)


def learn(net, opt, buf, a, gen, designs=None, log=True):
    """One step on --batch windows drawn from the buffer, of as many different episodes: each goes through the
    network at once from the state it was gathered in (no plant, no decision-by-decision loop), towards the
    teacher's move at every decision and, for the readout, the true responses. designs: with as many problems
    of design of its pool (--design; at one step of --design-every), the K read-out towards the teacher's K. log: with
    --unseen's rank correlations of the step (`unseen_log`: sorts, .005 s of a step of .064 s on a card; `imitate`
    takes them of a window's last step)."""
    x = buf.draw(buf.batch or a.batch, gen)
    feats = x['feats'].float() if 'feats' in x else None            # as gathered: not worked out again at every step
    code, law = (x['lab'], x['known']), None
    if 'E' in x:                                                  # --law: the teacher's E, K and V; E and K kept of the tracked rows
        E, K, V = x['E'].float(), x['K'].float(), x['V'].float()
        grow = N.shape_of(x['obs'].shape[-1])[0]-E.shape[2]      # the other row slots: 0
        code = code+(torch.nn.functional.pad(E, (0, 0, 0, grow)), torch.nn.functional.pad(K, (0, 0, 0, 0, 0, grow)), V)
        if 'U' in x and not getattr(a, 'meter', False):           # --unseen: what the instruments do not show, told with them (--meter: not told, `window`)
            code = code+(torch.nn.functional.pad(x['U'].float(), (0, 0, 0, grow)),)
    mean, second, mask, read, _ = net.steps(x['obs'], x['prev'], x['h0'][None], code, feats)
    l_bc = masked_mean((torch.tanh(mean)*mask-x['tch']).square(), x['m'])
    l_id = masked_mean(read_error(net, read, x['lab'], a), x['rows'][:, None, :, None]*x['m'][:, :, None, :])
    loss = a.bc_coef*l_bc+a.id_coef*l_id
    if 'E' in x:                                                  # the readouts towards them on every episode, and their product towards the law's move
        eh, kh, vh, mv = net.pairs.read_law
        on = x['m'].amax(-1)                                      # (b, T): decisions that count
        d, wt = N.taught(net, eh[:, :, :E.shape[2]].float(), 'E')-N.taught(net, E, 'E'), x['trk'][:, None, :]*on[:, :, None]      # (--diff E: both as changes from lag to lag)
        law = dict(head=masked_mean(d.square().mean(-1), wt),
                   law=masked_mean((torch.asinh(mv)-torch.asinh(x['lm'].float())).square(), x['m']), **e_by_kind(d.detach().square().mean(-1), wt, x['nl']))
        if 'U' in x:                                              # its read-out towards the true one, on every episode
            uh = net.pairs.read_unseen[:, :, :E.shape[2]].float()
            law['unseen'] = masked_mean(huber1(uh, x['U'].float()).mean(-1), wt)
            if log:
                law.update(unseen_log(uh.detach(), x['U'].float(), x['since'], x['lev'], wt))
        if kh is not None:                                        # (--solve: no K and V are read, the design is solved)
            ek, wk = (kh[:, :, :K.shape[2]].float()-K).square().mean(-1), x['trk'][:, None, :, None]*x['m'][:, :, None, :]
            law.update(gain=masked_mean(ek, wk), valve=masked_mean((vh.float()-V).square(), x['m']), **k_untold(ek.detach(), wk, x['known'], x['link']))
        if net.pairs.given:                                       # --given: the formula's E (after its factor and correction), in the episodes told the responses
            ef = net.pairs.read_given
            d, wt = torch.zeros_like(d) if ef is None else ef[:, :, :E.shape[2]].float()-E, wt*x['known'][:, None, None]
            law['headf'] = masked_mean(d.square().mean(-1), wt)
        loss = loss+a.law_coef*sum(v for q, v in law.items() if torch.is_tensor(v))
        with torch.no_grad():                                     # the part of E's error that stays the same over a row's window: an offset costs points where
            off = (d*wt[..., None]).sum(1)/wt.sum(1).clamp_min(1.)[..., None]      # noise of the same size does not (11-11: 40-60 against 97); not in the loss
            law['hbias'] = masked_mean(off.square().mean(-1), (wt.sum(1) > 0).to(d.dtype))
    l_pre = None
    if 'pre' in x:                                                # --probe: whether the unit has had its first event yet, said by the head's second output
        l_pre = masked_mean(torch.nn.functional.binary_cross_entropy_with_logits(N.probing(second), x['pre'][..., None].expand_as(second), reduction='none'), x['m'])
        loss = loss+l_pre
    l_des = None
    if designs is not None and designs.due():                     # --design (last: it runs the network again, and the readouts above are of the windows)
        l_des = designs.loss(net, gen)
        loss = loss+a.law_coef*a.design_coef*l_des
    opt.zero_grad(set_to_none=True)
    ok = bool(torch.isfinite(loss))
    if ok:
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip)
        ok = bool(torch.isfinite(gn))
        if ok:
            opt.step()
    return dict(fit=float(l_bc), gn=float(gn) if ok else None, skip=0. if ok else 1., pre=None if l_pre is None else float(l_pre),
                design=None if l_des is None else float(l_des),
                **{q: None if law is None or q not in law else float(law[q]) for q in LAW},
                **({q: None if q not in law else float(law[q]) for q in UNSEEN} if law is not None and 'unseen' in law else {}))


def imitate(net, critic, opt, copt, lane, buf, a, gen, designs=None):
    """One window of a lane while the teacher is all of the objective, gathering apart from learning (--replay).
    The window is run without a graph (the student's moves to the plant, the teacher's as labels: what `window`
    does, not differentiated) and put in the buffer; then --replay steps of `learn`. A window that is
    differentiated as it is run gives one step on the episodes of one lane, at the pace of the loop that runs
    plant and networks a decision at a time (one host thread: 11-11, a third of the GPU in use, 1,300 steps
    and 14,000 plants an hour); here the episodes gathered can be many and the steps are the GPU's.
    The critic is fitted on --fit-chunk of the window's episodes (`chunk_of`: as many pairs at any size), so it is there
    when the simulator gradient comes in."""
    n, ha, hs, ncv, nmv = lane.prog.n, lane.ha, lane.hs, lane.prog.ncv, lane.prog.nmv
    with torch.no_grad():
        _, rec, _, log = window(net, critic, lane, a, dict(bc=1., sim=0., value=0.), gen, later=chunk_of(a, lane) or n)
    H = rec['tch'].shape[1]
    pad = lambda x: x if H == a.window else torch.cat((x, x.new_zeros((n, a.window-H)+x.shape[2:])), 1)    # noqa: E731
    s = N.sizes(ncv, nmv)
    ns = net.pairs.size_of(ncv, nmv)-s[7]-(N.FM.SMEM if net.pairs.fm is not None else 0)   # the running sums, with their slow set (--slow), or the learned reader's memory (--reader)
    at = N.base_size(ncv, nmv)+s[7]                                                       # where they start: behind the actor's and the pairs' memories
    if net.pairs.reader is not None:                              # the learned reader's memory is in the actor's state already (nothing keeps it apart, as the sums are)
        sums0 = ha[0, :, at:at+ns]
    else:
        sums0 = torch.zeros(n, ns, device=ha.device) if hs is None else hs.to(ha.dtype)
    rows = dict(obs=pad(rec['obs'][:, :H]), prev=pad(rec['prev'][:, :H]), tch=pad(rec['tch']), m=pad(rec['m']), lab=pad(rec['lab'][:, :H]),
                h0=torch.cat((ha[0, :, :at], sums0, ha[0, :, at+ns:]), -1), known=lane.known, rows=lane.rows)    # (behind the sums: what the formula keeps)
    assert rows['h0'].shape[-1] == ha.shape[-1], 'the state a window starts at is the lane\'s, piece for piece'
    if 'pre' in rec:
        rows['pre'] = pad(rec['pre'])
    if lane.unpaired is not None:
        rows['unp'] = lane.unpaired
    if 'E' in rec:
        rows.update(E=pad(rec['E']).half(), K=pad(rec['K']).half(), V=pad(rec['V']), lm=pad(rec['lm']), trk=lane.prog.c['trk'][:, :nmv].float(), link=lane.link[:, :nmv],
                    nl=(lane.prog.ar < lane.prog.nl).float())                 # (nl: the plants of the prior, `e_by_kind`)
    if 'U' in rec:
        c = lane.prog.c
        rows.update(U=pad(rec['U']), since=pad(rec['since']), lev=((c['integ'] > 0) & (c['trk'] > 0))[:, :nmv].float())
    if rec['feats'] is not None:                                  # the running sums' features of every decision, in half precision (they are
        rows['feats'] = pad(rec['feats']).half()                  # of order 1): a loop of small solves over the window, once instead of at every step
    buf.add(rows)
    if designs is not None:
        designs.more(net)
    steps = [learn(net, opt, buf, a, gen, designs, log=i == a.replay-1) for i in range(a.replay)]
    mean = lambda q: (lambda v: sum(v)/len(v) if v else None)([x[q] for x in steps if x[q] is not None])   # noqa: E731
    log.update(fit=mean('fit'), gn=mean('gn'), pre=mean('pre'), design=mean('design'), skip=sum(x['skip'] for x in steps), updates=len(steps), capped=0., peak=None, short=0., hor=float(a.window),
               **{q: mean(q) for q in LAW+(UNSEEN if steps and 'unseen' in steps[0] else ())})
    log.update(fit_critic(critic, copt, lane, rec, a, slice(0, chunk_of(a, lane) or n)))
    return log


@torch.no_grad()
def advance(net, critic, lane, steps, gen, noise, drive_noise=0.):
    """Run a lane on without learning (the lanes start at different points of their episodes)."""
    prog = lane.prog
    adv_tag = (lane.tag, lane.age, 'adv', lane.k)
    for k in range(lane.k, min(lane.k+steps, HORIZON)):
        obs, s = canonical(C.observe(prog, lane.st, k).float(), lane.unpaired)
        F, lane.hs = sums(net, obs, lane.hs)
        u, mean, mask, _, lane.ha, true = decide(net, lane, obs, lane.prev, lane.ha, s, k, gen, noise, feats=F, need_read=False, lane_tag=adv_tag)
        if lane.drive is not None:
            u = driven(lane, lane.st, k, s, mask, u, drive_noise, gen)[1]
        if lane.lead is not None:
            u = torch.where((k < lane.lead)[:, None], C.teacher(prog, lane.st, k).float()*s*mask, u)
        if not getattr(critic, 'privileged', False):
            _, lane.hc = critic.steps(obs[:, None], lane.prev[:, None], u[:, None], lane.hc, (true, lane.cknown), F)
        lane.st, out = C.step(prog, lane.st, (u*s).to(prog.dtype), k)
        lane.score += out['outside'].sum(-1)*(k >= prog.k_first)
        lane.prev, lane.pm, lane.k = u, torch.tanh(mean)*mask, k+1


def weights(row, a):
    """The weights of the terms at a row: the teacher is all of the objective for --bc-hold rows, then fades out
    over --bc-iters rows as the simulator gradient fades in (no teacher: --bc-iters 0) down to --bc-floor, the
    critic's value enters the actor's objective over --value-warm, up to --value."""
    bc = max(a.bc_floor, min(1., 1.-(row-a.bc_hold)/a.bc_iters)) if a.bc_iters > 0 else 0.
    return dict(bc=bc, sim=1.-bc, value=getattr(a, 'value', 1.)*(min(1., row/a.value_warm) if a.value_warm > 0 else 1.))


def constrained_share(prog, r):
    """Share of the settled row-decisions at constrained operating points that a rollout r spent outside
    (the last 40 % of every such segment): whether the controller finds the reachable optimum. The points are the
    batch's own (aio.sim.core.Program.con): where an event took the unit and, with Menu.ride, where ordinary events
    found or left it (the generator's own batches: the same segments as before)."""
    exc = prog.con
    return float(r['seg_settled'][exc].sum()/r['seg_rows'][exc].sum().clamp_min(1))


METERED = ('dmc_meter', 'teacher_meter')    # references whose predictions are corrected from the instruments alone (aio.sim.stay.REFS): reported, not part of the scale


@torch.no_grad()
def references(prog, meter=False):
    """The references on a batch. meter: also the step-test DMC and the teacher from the instruments alone (--meter:
    the second is what is imitated); they are reported beside the others and are no part of the scale, which
    stays the one earlier runs were read on."""
    ctl = dict(hold=CT.Hold(), pi=CT.PI(), dmc=CT.Teacher('fixed'), teacher=CT.Teacher('late'), now=CT.Teacher('now'))
    if meter:
        ctl.update({q: ST.REFS[q]() for q in METERED})
    out = {}
    for q, c in ctl.items():
        out[q] = CT.rollout(prog, c); alive(references=prog.n, of=q)
    return out


def scale_of(ref, path=None):
    """The scale on a batch: the envelope of the references run on it, and of the file of aio.sim.oracle if it
    is of the same batch (None, False if it is not: it then says nothing here)."""
    per = {q: r['score'].tolist() for q, r in ref.items() if q not in METERED}
    if path:
        sc = SC.Scale.load(path, per['pi'])
        if sc is not None:
            return sc.join(per), True
    return SC.Scale.of(per), False if path else None


PARTS = ('before', 'after')


def before_after(r, ref):
    """`pts_before`, `pts_after` of a rollout r: its points on the part of the scored window before an episode's
    first process change (aio.sim.control.changed_k) and on the part from it on, over the batch. 0 is the hold on
    the same part, 100 the best of the references run here, per episode and part. (The Oracle's file has no
    parts, so this 100 is easier than the scale's: the two are read against the references' own, in `val_refs`.)"""
    best = {q: torch.stack([x[q] for name, x in ref.items() if name != 'hold' and name not in METERED]).amin(0) for q in PARTS}
    return {'pts_'+q: SC.points(r[q].tolist(), ref['hold'][q].tolist(), best[q].tolist()) for q in PARTS}


def levels(prog):
    """(n,) bool: the plants with a level among their scored rows, an integrating row that is tracked or held in a
    zone (the linear plants' constants; an episode on a plant of the engine: False)."""
    c, has = prog.c, torch.zeros(prog.n, dtype=torch.bool, device=prog.device)
    if all(q in c for q in ('integ', 'trk', 'zon')):
        lin = ((c['integ'] > 0) & ((c['trk'] > 0) | (c['zon'] > 0))).any(-1)
        has[:lin.shape[0]] = lin
        has[prog.nl:] = False
    return has


def kinds(prog):
    """The kind of plant of every episode: 'lin' for the generator's own, the plant's name for one of the engine
    (aio.sim.mech.Set.names; --per: an exam of the engine's plants read kind by kind)."""
    return ['lin']*prog.nl+(prog.mech.names() if prog.mech is not None else [])


SHADOW = 240    # decisions of memory the control of `Stay` keeps: one response's length (aio.sim.model.LAGS), more than the running sums' 200


class Stay:
    """--val-stay L: what the network gains by staying on a unit. A fixed batch of n plants is run for L episodes
    in a row (L x 720 decisions: each episode the next window on the same plants, aio.sim.core.Program `after`;
    the plant's state and the network's memory go on, every decision scored from the second episode on). Per
    episode:

      pts       the points with the memory carried from the unit's first decision
      fresh     the same episode from the same state of the plant by the network as it is when it has forgotten
                everything older than SHADOW decisions (aio.net.Policy, `forgetful`): pts - fresh is what the
                longer memory is worth (None in the first episode: there is nothing to forget)
      id_own, id_cross   the error of the responses read with the memory carried (published pairs, the others):
                whether the memory comes to know the unit, read off the network itself; `_near` and `_quiet`:
                the same within aio.net.NEAR decisions of the last event and in the quiet after it
      pts_level, pts_other   the points over the plants that have a level among their scored rows (an integrating
                row, tracked or held in a zone: `levels`) and over the others, each on its own 0 and 100 (None where
                there is no such plant); n_level: how many have one. Nearly all of the distance to a controller
                that knows the plant is on the first kind (11-12)
      move, flip   the moves of the episode (`Watch`)

    The scale of every episode is its own: 0 the MVs left where they are, 100 the best per unit of formula PI,
    step-test DMC (both with the plant as commissioned, all the stay) and teacher, all three taking the plant
    over in the state the teacher left it in at the end of the episode before (a hold from the very start would
    only drift further off: the scale would stretch with the stay and the points rise by themselves)."""

    def __init__(self, n, seed, links, dev, progs=None, fresh=True, first=0, refs=None, nseg=0):
        """progs: the windows themselves, where they are not the generator's own (an exam of aio.sim.stay: other
        changes on the plants); fresh: whether the control with the short memory is run (half the cost without);
        first: the window the summary `mean` starts at (an exam whose first windows only lead up to it); refs: the
        references of these windows where they are kept (aio.sim.stay.kept: run by the same rule as here); nseg
        (--val-nseg): segments the generator's own windows have room for (0: as they are; another batch otherwise).
        The windows are kept on the CPU and are on `dev` one at a time while they are run (`visit`)."""
        self.dev = dev
        if progs is None:
            kw = dict(nseg=nseg) if nseg else {}
            progs = [C.build(n, seed, dev, 'cpu', **kw)]
            for j in range(1, links):
                progs.append(C.build(n, seed+j, dev, 'cpu', after=progs[-1].tail, **kw))
        self.progs, self.fresh, self.first = list(progs), fresh, first
        self.level = levels(self.progs[0])
        st, self.hold, self.best, self.ref, self.per = None, [], [], [], []   # (per: every reference's scores, window by window and plant by plant)
        with torch.no_grad():
            for j, prog in enumerate(self.progs):
                if refs is not None:
                    per = {q: refs['score'][q][j] for q in ('hold', 'pi', 'dmc', 'teacher')}
                else:
                    r = {}
                    for q, c in dict(hold=CT.Hold(), pi=CT.PI(), dmc=CT.Teacher('fixed'), teacher=CT.Teacher('late')).items():
                        c.reset(prog)                             # each takes the plant over anew (the PI's loops are closed here)
                        r[q] = CT.rollout(prog, c, st=st)
                    st = r['teacher']['st']
                    per = {q: x['score'].tolist() for q, x in r.items()}
                self.hold.append(per['hold']); self.best.append(SC.envelope(per)); self.per.append(per)
                self.ref.append({q: round(SC.points(v, per['hold'], self.best[-1]), 2) for q, v in per.items() if q != 'hold'})
                parked(prog)
                alive(references=prog.n, link=j)

    def meta(self):
        return dict(n=self.progs[0].n, links=len(self.progs), shadow=SHADOW if self.fresh else 0, first=self.first, hold=[round(sum(h)/len(h), 1) for h in self.hold],
                    best=[round(sum(b)/len(b), 1) for b in self.best], **{q: [r[q] for r in self.ref] for q in self.ref[0]})

    @torch.no_grad()
    def __call__(self, net, per=None):
        """per: a list that takes the network's scores, window by window and plant by plant (--per)."""
        net.eval()
        own, st = N.Policy(net, stay=True, read=True, shadow=SHADOW if self.fresh else 0), None
        watch, lev = Watch(own), self.level.tolist()
        out = dict(pts=[], fresh=[], id_own=[], id_cross=[], failed=[], **{f'{q}_{w}': [] for q in ('id_own', 'id_cross') for w in ('near', 'quiet')},
                   pts_level=[], pts_other=[], move=[], flip=[], n_level=sum(lev))
        part = lambda v, yes: [x for x, k in zip(v, lev) if k == yes]     # noqa: E731
        for j, prog in enumerate(self.progs):
            fresh = None
            if j:
                own.restart(st['fail'])                           # (a unit that failed starts again at rest: aio.sim.core.resume)
            with visit(self.dev, prog):
                if j and self.fresh:
                    fresh = SC.points(CT.rollout(prog, own.forgetful(), st=st)['score'].tolist(), self.hold[j], self.best[j])
                r = CT.rollout(prog, watch, st=st)
            alive(validation=prog.n, link=j)
            st, score = r['st'], r['score'].tolist()
            if per is not None:
                per.append(score)
            kinds = {f'pts_{q}': SC.points(part(score, yes), part(self.hold[j], yes), part(self.best[j], yes)) if any(k == yes for k in lev) else None
                     for q, yes in (('level', True), ('other', False))}
            for q, x in dict(pts=SC.points(score, self.hold[j], self.best[j]), fresh=fresh, failed=int(r['failed'].sum()), **own.readout(), **kinds).items():
                out[q].append(None if x is None else round(x, 4 if q.startswith('id') else 2))
            for q, x in watch.stats().items():
                out[q].append(x)
        net.train()
        mean = lambda v: sum(v)/len(v) if v else None             # noqa: E731
        gain, half = [a-b for a, b in zip(out['pts'][1:], out['fresh'][1:])] if self.fresh else [], len(self.progs)//2
        # gain: what the memory older than SHADOW decisions is worth, over the episodes after the first; gain_late: over the later half of the stay;
        # rise: the later half's points over those of the episodes before it, the first left out (it starts at rest and is scored from its first event);
        # mean: the points over the windows from `first` on
        out.update(gain=rnd(mean(gain), 2), gain_late=rnd(mean(gain[half-1:]), 2) if half and gain else None,
                   rise=rnd(mean(out['pts'][half:])-mean(out['pts'][1:half]), 2) if half > 1 else None, mean=rnd(mean(out['pts'][self.first:]), 2))
        return out


def exams_of(names, n, links, dev, keep=None, nseg=0):
    """--val-exams: the fixed exams of a long stay that are named (aio.sim.stay.EXAMS), each a `Stay` on its own
    windows: the reading of --val-stay on kinds of change and plants the training ground does not have (pulse,
    dead, quiet, mech, mech_new), beside the same plants as the generator has them (usual). The control with the
    short memory is not run (half the cost); `quiet` is read from its third window on (the first two lead up to
    it). keep: a directory the references are kept in, per exam, size and device (aio.sim.stay.kept: read when
    they are those of this very batch, else run and written; a window of `mech` is minutes on a CPU). nseg
    (--val-nseg): segments their windows have room for where the exam does not ask for more (0: as they are)."""
    out = {}
    for q in names:
        assert q in ST.EXAMS, f'--val-exams: {q} is not one of {tuple(ST.EXAMS)}'
        progs = list(ST.exam(q, n, ST.EXAM_SEED, links, nseg=nseg or None, device=dev, gen_device='cpu'))
        refs = ST.kept(progs, Path(keep)/f'{q}-{n}x{links}{f"-nseg{nseg}" if nseg else ""}-{torch.device(dev).type}.json',
                       say=lambda w, *_, q=q: alive(references=n, exam=q, link=w)) if keep else None
        out[q] = Stay(n, ST.EXAM_SEED, links, dev, progs=progs, fresh=False, first=min(ST.EXAMS[q].get('read', 0), links-1), refs=refs)
    return out


def read_exams(exams, net, per=None):
    """Every exam's reading (`Stay`), without the entries only the short-memory control fills. per: a dict that takes, per
    exam, the network's scores by window and plant (--per)."""
    return {q: {k: v for k, v in e(net, None if per is None else per.setdefault(q, [])).items() if k not in ('fresh', 'gain', 'gain_late')} for q, e in exams.items()}


class Watch:
    """A controller with its moves watched: `move`, their mean size (share of the slew limit a decision), and
    `flip`, the share of an MV's decisions on which its move turns back on the last one (chatter; a teacher
    that knows the plant 0.05, the formula PI 0.24). Accumulated on GPU/device with a single sync in `stats()`."""

    def __init__(self, ctl):
        self.ctl = ctl

    def reset(self, prog):
        self.ctl.reset(prog)
        self.n = self.size = self.turn = None
        self.last = None

    def resume(self, prog):
        """The next window on the same plants (aio.sim.control.rollout keeps the memory of a controller that has this):
        the controller goes on as it does, the moves are counted anew."""
        (self.ctl.resume if hasattr(self.ctl, 'resume') else self.ctl.reset)(prog)
        self.n = self.size = self.turn = None
        self.last = None

    def restart(self, again):
        if hasattr(self.ctl, 'restart'):
            self.ctl.restart(again)

    def __call__(self, prog, st, k, obs):
        a = self.ctl(prog, st, k, obs)
        m = (prog.c['mvm'] > 0) & (k >= prog.k_first)[:, None]
        n = m.sum().double()
        sz = (a.abs()*m).sum().double()
        if self.n is None:
            self.n, self.size, self.turn = n, sz, torch.zeros((), dtype=torch.float64, device=a.device)
        else:
            self.n = self.n+n
            self.size = self.size+sz
        if self.last is not None:
            self.turn = self.turn+((a*self.last < 0) & (a.abs() > 1e-3) & (self.last.abs() > 1e-3) & m).sum().double()
        self.last = a
        return a

    def stats(self):
        if self.n is None:
            return dict(move=0.0, flip=0.0)
        n, sz, tr = torch.stack((self.n, self.size, self.turn)).tolist()
        return dict(move=round(sz/max(n, 1.), 4), flip=round(tr/max(n, 1.), 4))


def forget(*nets):
    """Drop the sequence networks' kept memories of their batches (seqnet Room: up to AIO_SEQ_ROOMS of them, 20 GB each at
    256 plants of d768 l12); a batch's next decision builds its own again."""
    for n in nets:
        for m in n.modules():
            getattr(m, 'forget', lambda: None)()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def validate(net, prog, ref, scale, known=True, paired=True, return_rollout=False):
    """The network on a batch beside its references `ref` (run where the network is). The batch may be kept on the CPU:
    it is on the network's device for the reading (`visit`)."""
    net.eval()
    with visit(ref['pi']['score'].device, prog):
        set_teacher(prog, getattr(net, 'meter', False))           # (the teacher whose E, K and V the rollout that is told gets: the one the run imitates)
        watch = Watch(N.Policy(net))
        forget(net)
        r = CT.rollout(prog, watch)
        watch.ctl.h = watch.ctl.hb = None                         # its memory of the rollout before the next one
        forget(net)
        rk = CT.rollout(prog, N.Policy(net, known=True)) if known else None
        forget(net)
        rp = CT.rollout(prog, N.Policy(net, paired=N.blind_main(net))) if paired and getattr(net, 'unpaired', 0.) > 0 else None      # --unpaired: the other way too (told, or blind)
        forget(net)
        net.train()
        pi = ref['pi']
        out = dict(pts=scale(r['score'].tolist()), score=float(r['score'].mean()), cost=float(r['cost'].mean()), failed=int(r['failed'].sum()),
                   x_pi=float(r['score'].mean()/pi['score'].mean()), cost_x_pi=float(r['cost'].mean()/pi['cost'].mean()),
                   win=float((r['score'] < pi['score']).float().mean()), loss=float((r['score'] > pi['score']).float().mean()),
                   con=round(constrained_share(prog, r), 4), con_pi=round(constrained_share(prog, pi), 4), **before_after(r, ref), **watch.stats())
        lev = levels(prog).tolist()
    if rk is not None:
        out.update(pts_known=scale(rk['score'].tolist()), score_known=float(rk['score'].mean()))
    if rp is not None:
        q = 'paired' if N.blind_main(net) else 'blind'           # pts_paired: told, beside a blind validation; pts_blind: blind, beside a told one
        out.update({'pts_'+q: scale(rp['score'].tolist()), 'score_'+q: float(rp['score'].mean())})
    sc = r['score'].tolist()                                      # the plants with a level among their scored rows, and the others: each on its own 0 and 100
    for q, yes in (('level', True), ('other', False)):
        if any(k == yes for k in lev):
            out['pts_'+q] = SC.points(*([x for x, k in zip(v, lev) if k == yes] for v in (sc, scale.hold, scale.best)))
    out['n_level'] = sum(lev)
    alive(validation=prog.n)
    return (out, r) if return_rollout else out


def ref_table(prog, ref, scale):
    """The references on a batch as run.json keeps them (`val_refs`): per reference its score, cost, points, points on
    the plants with a level and on the others, constrained share and points before and after the first change."""
    with visit(ref['pi']['score'].device, prog):                  # (a batch kept on the CPU: where its references are, for this)
        pi, lev = ref['pi']['score'].mean(), levels(prog).tolist()
        part = lambda r, yes: round(SC.points(*([x for x, k in zip(v, lev) if k == yes] for v in (r['score'].tolist(), scale.hold, scale.best))), 2) if yes in lev else None    # noqa: E731
        return {q: dict(score=float(r['score'].mean()), cost=float(r['cost'].mean()), x_pi=float(r['score'].mean()/pi), pts=round(scale(r['score'].tolist()), 2),
                        pts_level=part(r, True), pts_other=part(r, False),
                        con=round(constrained_share(prog, r), 4), **{k: round(x, 2) for k, x in before_after(r, ref).items()}) for q, r in ref.items()}


_REF_KEEP = ('score', 'cost', 'before', 'after', 'seg_settled', 'seg_rows')


def cached_refs(cache, key, prog, meter=False):
    """Run `references(prog, meter)` or load cached results from `cache/key.pt` (saves ~9 min on every launch/rollover)."""
    dev = prog.device
    path = (Path(cache)/f'{key}.pt') if cache else None
    if path is not None and path.exists():
        raw = safe_torch_load(path, map_location='cpu')
        return {q: {k: v.to(dev) for k, v in r.items()} for q, r in raw.items()}
    ref = references(prog, meter)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_save({q: {k: r[k].detach().cpu() for k in _REF_KEEP if k in r} for q, r in ref.items()}, path)
    return ref


def size_vals(items, seed, dev, meter=False, cache=None):
    """--val-sizes 24x10@64 48x20@32 -> {size: (batch, references, scale)}: a fixed batch of each size (the generator's
    own, drawn on the CPU), the references run on it and the scale they make (no Oracle file)."""
    out = {}
    for it in items or ():
        name, n = it.split('@', 1) if '@' in it else (it, '64')
        assert name in PR.SIZES, f'--val-sizes: {it} is not SIZE@PLANTS of {tuple(PR.SIZES)}'
        prog = C.build(int(n), seed, dev, 'cpu', dims=name)
        ref = cached_refs(cache, f'sz-{name}-n{n}-s{seed}-mt{int(bool(meter))}', prog, meter)
        out[name] = (prog, ref, scale_of(ref)[0])
        parked(prog)                                              # (`validate` has it on the card for its reading)
    return out


def read_sizes(vsizes, net):
    """The network on every batch of --val-sizes (`validate`, without the rollouts told the responses or the pairing)."""
    return {q: {k: round(x, 4) for k, x in validate(net, prog, ref, scale, False, False).items()} for q, (prog, ref, scale) in vsizes.items()}


def score(a, dev):
    """--score CKPT: no training. Evaluates the checkpoint's `model` (and optional `raw` with --val-raw) using
    `cached_refs`, `validate`, and `read_sizes` so the output matches in-loop `metrics.jsonl` schema identically
    while preserving legacy `all`/`lin`/`mech`/`score`/`before`/`after` fields and `--per`."""
    net, ck = N.load(a.score, dev)
    cfg = ck.get('config') or {}
    meter = bool(getattr(a, 'meter', False) or cfg.get('meter', False))
    net.meter = meter
    is_seq = 'seq' in ((ck.get('model') or {}).get('pairs') or {})
    known = (not getattr(a, 'no_known_val', False)) and not is_seq
    prog = C.build(a.val_n, a.val_seed, dev, 'cpu', mech=a.val_mech)
    ref = cached_refs(getattr(a, 'val_cache', None), f'val-n{a.val_n}-s{a.val_seed}-m{a.val_mech}-mt{int(bool(meter))}', prog, meter)
    scale, used = scale_of(ref, a.refs)
    v, model = validate(net, prog, ref, scale, known, return_rollout=True)
    run = {q: r['score'].tolist() for q, r in ref.items()}
    run['model'] = model['score'].tolist()
    split = {q: {name: round(x['pts_'+q], 2) for name, x in ((name, before_after(r, ref)) for name, r in dict(ref, model=model).items())} for q in PARTS}
    part = lambda q, i: round(SC.points(run[q][i], scale.hold[i], scale.best[i]), 2)     # noqa: E731
    parts = dict(all=slice(None), lin=slice(0, prog.nl), mech=slice(prog.nl, None))
    out = dict(ckpt=str(a.score), iter=ck.get('iter'), hours=ck.get('hours'), updates=ck.get('updates'), samples=ck.get('samples'),
               val_n=a.val_n, val_seed=a.val_seed, val_mech=a.val_mech, mech_n=prog.n-prog.nl,
               refs_used=used, scale=scale.meta(), val={q: round(x, 4) for q, x in v.items()}, val_refs=ref_table(prog, ref, scale),
               score={q: round(sum(v_s)/len(v_s), 2) for q, v_s in run.items()}, **split,
               **{k: {q: part(q, i) for q in run} for k, i in parts.items() if len(run['model'][i])})
    if getattr(a, 'val_raw', False) and 'raw' in ck:
        raw_net = N.of_state(ck['raw']['actor'], ck['raw'].get('pairs'), cfg, dev)
        raw_net.meter = meter
        vr = validate(raw_net, prog, ref, scale, False)
        out['raw'] = {q: round(x, 4) for q, x in vr.items()}
    if getattr(a, 'val_sizes', None):
        vsizes = size_vals(a.val_sizes, a.val_seed, dev, meter, getattr(a, 'val_cache', None))
        out['val_sizes'] = read_sizes(vsizes, net)
        out['val_size_refs'] = {q: ref_table(*vs) for q, vs in vsizes.items()}
    per, stays = {} if a.per else None, {}
    if a.val_stay > 1:                                            # what staying on a unit is worth to these weights (`Stay`)
        stays['stay'] = stay = Stay(a.val_stay_n, a.val_seed+1, a.val_stay, dev, nseg=a.val_nseg)
        out.update(stay_refs=stay.meta(), stay=stay(net, None if per is None else per.setdefault('stay', [])))
    if a.val_exams:                                               # and on the fixed exams of a long stay
        exams = exams_of(a.val_exams, a.val_stay_n, a.val_stay if a.val_stay > 1 else ST.EXAM_WINDOWS, dev, a.val_exams_refs, a.val_nseg)
        out.update(exam_refs={q: e.meta() for q, e in exams.items()}, exams=read_exams(exams, net, per))
        stays.update(exams)
    if per is not None:                                           # --per: what the points above are taken of, for a reading by another rule
        a.per.write_text(dumps(dict(ckpt=str(a.score), iter=ck.get('iter'), val=dict(level=levels(prog).tolist(), score=run),
                                    **{q: dict(level=e.level.tolist(), kind=kinds(e.progs[0]), refs=e.per, model=per[q]) for q, e in stays.items()})))
    print(dumps(out), flush=True)
    return out


def nets(a, dev):
    """The actor and the critic the flags ask for (a run and --bench: the same ones)."""
    assert not a.law or (a.replay and (not a.no_stats or getattr(a, 'reader', False))), '--law is for --replay, with the running sums (with --reader: its read-outs as labels alone)'
    assert a.law or not (a.formula or a.solve or a.given), '--formula, --solve and --given are steps of --law'
    assert not (a.slow or a.arx) or not a.no_stats, '--slow is a second set of the running sums, --arx a second fit of them'
    assert a.arx or not a.dv, '--dv widens the fits of --arx'
    assert not getattr(a, 'design', 0) or (a.law and a.replay and not a.solve), '--design teaches the K read-out of --law, in the steps of --replay'
    unseen = getattr(a, 'unseen', False)                          # (a caller with the settings of an earlier run has none)
    assert not unseen or (a.law and not (a.formula or a.given)), '--unseen is given to the E read-out that learns: --law without --formula or --given'
    if getattr(a, 'seq', None):                                   # --seq: the plain attention network (aio.seqnet), nothing of the pair pathway; the critic, which never runs, as it was
        from aio import seqnet as SQ
        assert a.replay and a.no_stats and not (a.law or a.reader or a.slow or a.arx or a.dv or a.tied), '--seq goes with --replay and --no-stats, and with none of the pair pathway\'s switches'
        net = SQ.SeqAIO(a.seq, a.seq_window, a.knots).to(dev)
        net.actor.grad_ckpt = bool(getattr(a, 'seq_ckpt', False))
        a.lean = False                                            # (a decision's graph goes through what the network keeps of the ones before: per-layer checkpointing via --seq-ckpt)
        net.probe, net.unpaired, net.meter = a.probe, getattr(a, 'unpaired', 0.), bool(getattr(a, 'meter', False))
        crit = N.PrivCritic(a.heads).to(dev) if getattr(a, 'priv_critic', False) else N.Critic(a.heads, True, 0, a.knots).to(dev)
        return net, crit
    reader = getattr(a, 'reader', False)                          # --reader: the learned reader in place of the running sums; the critic, which never runs, keeps them
    assert not reader or (a.no_stats and not (a.formula or a.solve or a.given)), '--reader goes with --no-stats and without --formula, --solve, --given; --law beside it: its read-outs as labels alone'
    net = N.AIO(not a.no_stats, a.tied, a.law, a.knots, a.formula, a.solve, a.given, a.slow, a.arx, a.dv, a.diff, unseen, getattr(a, 'told', False), reader,
                {'read': False, 'labels': True}.get(getattr(a, 'law_path', None)), getattr(a, 'hybrid', False)).to(dev)
    net.probe = a.probe if a.replay else 0.                       # how it is validated and runs (aio.net.Policy)
    net.unpaired = getattr(a, 'unpaired', 0.)                     # (a run with --unpaired is validated and runs with nothing published about the pairing)
    net.meter = bool(getattr(a, 'meter', False))                  # (--meter: the validation that is told gets the teacher of the instruments alone)
    crit = N.PrivCritic(a.heads).to(dev) if getattr(a, 'priv_critic', False) else N.Critic(a.heads, not a.no_stats or bool(reader), a.tied, a.knots).to(dev)
    return net, crit


def zeropower_newtonschulz5(G, steps=5):
    """5-step quintic Newton-Schulz matrix orthogonalization (Jordan 2024; Kimi Moonlight 2025).
    Projects 2D momentum update G to the nearest semi-orthogonal matrix (equalized singular values)."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() if G.is_cuda else G.float()
    trans = X.size(0) > X.size(1)
    if trans:
        X = X.T
    X = X/(X.norm()+1e-7)
    for _ in range(steps):
        A = X@X.T
        B = b*A+c*(A@A)
        X = a*X+B@X
    if trans:
        X = X.T
    return X.to(G.dtype)


class MuonAdamW(torch.optim.Optimizer):
    """Hybrid Muon (hidden 2D Transformer matrices) + decoupled AdamW (embeddings, readout, norms, biases).
    Uses Moonlight RMS-matched scaling `0.2 * sqrt(max(m, n))` so the same base `--lr` transfers cleanly
    from AdamW, while equalizing singular directions and balancing layer update rates across depth."""

    def __init__(self, param_groups, lr=1e-4):
        defaults = dict(lr=lr, lr_mult=1., weight_decay=0., kind='adamw', momentum=.95, betas=(.9, .95), eps=1e-8)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for g in self.param_groups:
            lr = g['lr']
            wd = g['weight_decay']
            if g.get('kind') == 'muon':
                mom = g['momentum']
                for p in g['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    st = self.state[p]
                    if 'buf' not in st:
                        st['buf'] = torch.zeros_like(grad)
                    buf = st['buf']
                    buf.lerp_(grad, 1.-mom)
                    u = zeropower_newtonschulz5(grad.lerp(buf, mom))
                    if wd:
                        p.mul_(1.-lr*wd)
                    p.add_(u, alpha=-lr*(.2*math.sqrt(max(p.shape[0], p.shape[1]))))
            else:
                b1, b2 = g['betas']
                eps = g['eps']
                for p in g['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    st = self.state[p]
                    if 'step' not in st:
                        st['step'] = 0
                        st['exp_avg'] = torch.zeros_like(grad)
                        st['exp_avg_sq'] = torch.zeros_like(grad)
                    st['step'] += 1
                    m, v = st['exp_avg'], st['exp_avg_sq']
                    m.lerp_(grad, 1.-b1)
                    v.mul_(b2).addcmul_(grad, grad, value=1.-b2)
                    bc1, bc2 = 1.-b1**st['step'], 1.-b2**st['step']
                    if wd:
                        p.mul_(1.-lr*wd)
                    p.addcdiv_(m, (v.sqrt()/math.sqrt(bc2)).add_(eps), value=-lr/bc1)
        return loss


_ROUTING_BIAS = ('.pair', '.lag_pair', '.lag_own', '.lag_b', '.edge_bias.weight',
                 '.rel_bias.weight', '.rel_lag_bias.weight', '.rel_p', '.rel_a', '.rel_eta', '.rel_c',
                 '.act_gate.weight', '.act_gate.bias')


def optimizer(net, a):
    """The actor's optimizer: Adam, AdamW (--wd), or hybrid Muon+AdamW (--opt muon).
    `--bias-lr-mult` scales LR on attention routing biases (pair, lag_pair, lag_own, lag_b).
    `--warm-steps`: each step's rate rises from 0 over the first so many steps of the run."""
    opt_kind = getattr(a, 'opt', 'adamw')
    bmult = float(getattr(a, 'bias_lr_mult', 1.))
    mom = float(getattr(a, 'muon_mom', .95))
    if opt_kind == 'muon':
        mu, dec, rest, rbias = [], [], [], []
        for q, p in net.named_parameters():
            if not p.requires_grad:
                continue
            if q.endswith(_ROUTING_BIAS):
                rbias.append(p)
            elif p.dim() == 2 and ('.layers.' in q or q.endswith('.act.0.weight')):
                mu.append(p)
            elif p.dim() >= 2:
                dec.append(p)
            else:
                rest.append(p)
        groups = []
        if mu:
            groups.append(dict(params=mu, kind='muon', weight_decay=a.wd, momentum=mom, lr_mult=1.))
        if dec:
            groups.append(dict(params=dec, kind='adamw', weight_decay=a.wd, betas=(.9, a.beta2), lr_mult=1.))
        if rest:
            groups.append(dict(params=rest, kind='adamw', weight_decay=0., betas=(.9, a.beta2), lr_mult=1.))
        if rbias:
            groups.append(dict(params=rbias, kind='adamw', weight_decay=0., betas=(.9, a.beta2), lr=a.lr*bmult, lr_mult=bmult))
        opt = MuonAdamW(groups, lr=a.lr)
    elif getattr(a, 'wd', 0.) > 0 or bmult != 1.:
        dec, rest, rbias = [], [], []
        for q, p in net.named_parameters():
            if q.endswith(_ROUTING_BIAS) and bmult != 1.:
                rbias.append(p)
            elif p.dim() >= 2 and not q.endswith('.lag_b'):
                dec.append(p)
            else:
                rest.append(p)
        groups = [dict(params=dec, weight_decay=a.wd, lr_mult=1.), dict(params=rest, weight_decay=0., lr_mult=1.)]
        if rbias:
            groups.append(dict(params=rbias, weight_decay=0., lr=a.lr*bmult, lr_mult=bmult))
        opt = torch.optim.AdamW(groups, lr=a.lr, betas=(.9, a.beta2), **_fused_kw(net))
    else:
        opt = torch.optim.Adam(net.parameters(), lr=a.lr, **_fused_kw(net))
    if getattr(a, 'warm_steps', 0) > 0:
        opt.warm_n = 0

        def warm(o, *_):
            f = min(1., (o.warm_n+1)/a.warm_steps)
            for g in o.param_groups:
                g['lr'] = g.get('set_lr', g['lr'])*f
            o.warm_n += 1
        opt.register_step_pre_hook(warm)
    return opt


def lr_factor(hours, of):
    """The share of --lr at this progress along a schedule of `of` units: down to a tenth along a cosine."""
    if of <= 0:
        return 1.
    return .1+.45*(1+math.cos(math.pi*min(1., hours/of)))


def prune_saves(out, keep):
    """--keep-saves N: keep only the N newest `row<int>.pt` snapshots in `out` (0: keep all). Never touches `last.pt` or `best.pt`."""
    if not keep or keep < 1:
        return []
    snaps = []
    for p in Path(out).glob('row*.pt'):
        stem = p.stem[3:]
        if stem.isdigit():
            snaps.append((int(stem), p))
    snaps.sort()
    dropped = []
    for _, p in snaps[:-keep]:
        p.unlink(missing_ok=True)
        dropped.append(p.name)
    return dropped


def bench_kinds(a):
    """--bench with --mech-groups: the kinds its mechanistic plants are of, those of the first group given (a batch of a
    run has one group's: `mech_of`); None: the pool's, as without --mech-groups."""
    groups = groups_of(a.mech_groups)
    return groups[0][1] if groups and a.mech else None


def bench(a, dev):
    """Speed and memory of every part at the given sizes, with the network and the plants the flags ask for (--law, --arx,
    --dv, --slow, --diff; --mech, --menu, --nseg; --mech-groups: --mech of the plants on the kinds of the first group, as a
    batch of a run that has that group's, what a decision costs going with the kinds; --bench-size: --envs plants of that
    size, the generator's own changes there as in a run, --batch, --fit-chunk and --design scaled to its pairs) and the
    window each phase has (--window with the teacher, --sim-window with the simulator gradient: --sim-window 8 for the
    phase with the teacher alone at more plants than the windows with their graphs have room for; then row_s is not
    a row's); trains nothing that is kept. buffer_gb: what --keep windows of
    a lane hold. The peaks by part: peak_gb_window the windows with their graphs up to the critic's fit, peak_gb_fit the
    fit's own, peak_gb_learn the steps of learning's. A step of learning by part: learn_step_s as a run takes it (the
    problems of design at one step of --design-every), learn_step_windows_s without them, unseen_log_s what --unseen
    logs of a window's last step."""
    cuda, gb = dev.type == 'cuda', lambda x: round(x/2**30, 2)
    sync, peak, reset = ((lambda: torch.cuda.synchronize()), torch.cuda.max_memory_allocated, torch.cuda.reset_peak_memory_stats) if cuda else ((lambda: None), (lambda: 0), (lambda: None))
    held = torch.cuda.max_memory_reserved if cuda else (lambda: 0)   # what the allocator took of the card (reserved_gb*: above the peaks by what it keeps between windows)
    wins = (a.window, a.sim_window or a.window)                   # with the teacher, with the simulator gradient
    out = dict(device=str(dev), card_gb=a.card_gb, card_alloc=a.card_alloc, envs=a.envs, window=wins[0], sim_window=wins[1], guard=a.guard, lean=a.lean, gen_device=a.gen_device,
               cpu_placement=a.cpu_placement, tied=a.tied, mech=a.mech, mech_kinds=bench_kinds(a), menu=a.menu, nseg=a.nseg, size=a.bench_size,
               net={q: getattr(a, q) for q in ('law', 'slow', 'arx', 'dv', 'diff', 'unseen', 'told', 'knots') if getattr(a, q)})
    gen = torch.Generator(device=dev).manual_seed(1)
    usual, size = a.bench_size == DEFAULT, a.bench_size
    kw = menu_kw(a.menu, a.nseg, True) if usual else dict(menu_kw((), a.nseg, True), dims=size)
    t = time.time(); prog = drawn(a.envs, 1, dev, a.gen_device, bench_kinds(a), mech=a.mech, **kw); sync(); out['build_s'] = round(time.time()-t, 2)
    net, critic = nets(a, dev)
    opt, copt = optimizer(net, a), torch.optim.Adam(critic.parameters(), lr=a.critic_lr)
    lane = Lane(prog, a, gen)
    out['fit_chunk'] = chunk_of(a, lane)
    alloc = retries(dev)
    t = time.time(); advance(net, critic, lane, 64, gen, a.noise); sync()
    out['plant_and_nets_ms_per_decision'] = round((time.time()-t)/64*1e3, 2)
    parts = dict(window=0, fit=0)
    reset()

    def before_fit():                                             # the window's peak so far is the actor's; what follows is the fit's
        parts['window'], parts['held'] = max(parts['window'], peak()), max(parts.get('held', 0), held()); reset()
    temp, ws = dict(usual=None, before_fit=before_fit), []
    for i, w in enumerate((dict(bc=0., sim=1., value=a.value), dict(bc=0., sim=1., value=a.value), dict(bc=1., sim=0., value=0.))):
        lane.label = torch.full_like(lane.label, w['bc'] > 0)
        a.window = wins[w['sim'] > 0]
        t = time.time(); update(net, critic, opt, copt, lane, a, w, gen, temp); sync(); ws.append(time.time()-t)
        parts['fit'], parts['held'] = max(parts['fit'], peak()), max(parts.get('held', 0), held()); reset()
    out.update(window_s=round(ws[1], 2), window_with_teacher_s=round(ws[2], 2), samples_per_s=round(a.envs*wins[1]/ws[1]),
               row_s=round(ws[1]*math.ceil(HORIZON/wins[1]), 1))
    if cuda:                                                      # the windows kept their graphs: their peak, and its two parts
        out.update(peak_gb_windows=gb(max(parts['window'], parts['fit'])), peak_gb_window=gb(parts['window']), peak_gb_fit=gb(parts['fit']), reserved_gb_windows=gb(parts['held']))
    if a.replay:                                                  # gathering apart from learning: a window gathered, a step of learning
        buf = Buffer(a.keep*a.envs, per_pairs(a.batch, size))
        if cuda:                                                  # this path's own peak
            opt.zero_grad(set_to_none=True); copt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); reset()
        designs = None
        if a.design:                                              # --design: one set, drawn here; its term is in every step
            designs = Designs(a, None, lambda: 1 << 31, dev, net.pairs.nk, None if usual else dict(dims=size), every=1 << 30,
                              b=per_pairs(a.design, size), n=per_pairs(a.design_set, size), keep=per_pairs(a.design_keep, size))
            t = time.time(); designs.more(net); sync(); out.update(design=designs.b, design_every=designs.stride, design_set_s=round(time.time()-t, 2))
            while designs.buf.n < designs.b:                      # (--design above a set: the sets a step takes)
                designs.more(net)
        t = time.time(); imitate(net, critic, opt, copt, lane, buf, a, gen, designs); sync(); whole = time.time()-t
        first, kept = peak(), held(); reset()
        t = time.time(); [learn(net, opt, buf, a, gen, designs, log=False) for _ in range(8)]; sync(); step = (time.time()-t)/8     # (8: a multiple of --design-every)
        rows = math.ceil(HORIZON/wins[0])
        out.update(replay=a.replay, batch=buf.batch, keep=a.keep, learn_step_s=round(step, 3), gather_window_s=round(whole-a.replay*step, 2),
                   buffer_gb=round(sum(v.numel()*v.element_size() for v in buf.x.values())/2**30, 2),
                   row_s_replay=round(whole*rows, 1), updates_per_hour=round(3600/whole*a.replay), episodes_per_hour=round(3600/(whole*rows)*a.envs))
        if designs is not None:                                   # what the problems of design cost a step
            t = time.time(); [learn(net, opt, buf, a, gen, log=False) for _ in range(8)]; sync(); out['learn_step_windows_s'] = round((time.time()-t)/8, 3)
        x = buf.draw(buf.batch, gen)
        if 'U' in x:                                              # and --unseen's log of a step (rank correlations by the time since an event)
            wt = x['trk'][:, None, :]*x['m'].amax(-1)[:, :, None]
            t = time.time(); [unseen_log(x['U'].float(), x['U'].float(), x['since'], x['lev'], wt) for _ in range(8)]; sync()
            out['unseen_log_s'] = round((time.time()-t)/8, 4)
        if cuda:
            last = peak()
            out.update(peak_gb=gb(max(first, last)), peak_gb_learn=gb(last), reserved_gb=gb(max(kept, held())))
    out['alloc'] = retries(dev, alloc)                            # (the allocator's counts over the windows and steps above: `retries`)
    t = time.time(); CT.rollout(prog, N.Policy(net), 60); sync(); out['validation_s_per_episode_batch'] = round((time.time()-t)*HORIZON/60, 1)
    print(dumps(out), flush=True)
    return out


def bench_mix(a, dev):
    """--bench with --sizes: lanes of different sizes taking turns in one process, as in a run. A run has --lanes batches
    of plants on the card at a time: every choice of that many of the sizes is run in turn (all of them at once where
    there are no more sizes than lanes), the other sizes' plants, windows kept and problems of design waiting on the
    CPU as a run's do (`settle`), and what the allocator kept given back before a window of another size than the last
    (a size alone on a 16 GB card, 11-13: 10 GB allocated at its peak and 13 GB held; three sizes in turn with
    nothing given back: 13 to 17 GB allocated, 18 to 21 held). Two rounds a choice of the windows with the simulator
    gradient (that phase is a process of its own in a run: no buffers); then, their memory given back, two rounds a
    choice of windows gathered with their steps of learning. Per size the seconds a row of it takes in each phase (its
    last window); per phase and choice the peak allocated and the most the allocator held, and the largest of them."""
    cuda, gb = dev.type == 'cuda', lambda x: round(x/2**30, 2)
    sync, peak, held, reset, free = ((lambda: torch.cuda.synchronize()), torch.cuda.max_memory_allocated, torch.cuda.max_memory_reserved,
                                     torch.cuda.reset_peak_memory_stats, torch.cuda.empty_cache) if cuda else ((lambda: None), (lambda: 0), (lambda: 0), (lambda: None), (lambda: None))
    wins, rounds = (a.window, a.sim_window or a.window), 2
    gen = torch.Generator(device=dev).manual_seed(1)
    net, critic = nets(a, dev)
    opt, copt = optimizer(net, a), torch.optim.Adam(critic.parameters(), lr=a.critic_lr)
    lanes = {}
    for _, q, n in sizes_of(a.sizes, a.envs):
        kw = menu_kw(a.menu, a.nseg, True) if q == DEFAULT else dict(menu_kw((), a.nseg, True), dims=q)
        lanes[q] = Lane(C.build(n, 1, dev, a.gen_device, mech=0., **kw), a, gen)
        advance(net, critic, lanes[q], 64, gen, a.noise)
        parked(lanes[q].prog)
    picks = list(itertools.combinations(lanes, min(max(a.lanes, 1), len(lanes))))
    out = dict(device=str(dev), card_gb=a.card_gb, card_alloc=a.card_alloc, plants={q: x.prog.n for q, x in lanes.items()}, fit_chunk={q: chunk_of(a, x) for q, x in lanes.items()}, window=wins[0], sim_window=wins[1],
               lanes=len(picks[0]), rounds=rounds, net={q: getattr(a, q) for q in ('law', 'slow', 'arx', 'dv', 'diff', 'unseen', 'told', 'knots') if getattr(a, q)})
    mark, counts = {}, {}

    def before_fit():                                             # (a window with the simulator gradient: its own part apart from the critic's fit)
        sync(); mark.update(t=time.time(), c=retries(dev))
    temps, took = {q: dict(usual=None, before_fit=before_fit) for q in lanes}, {q: 0. for q in lanes}
    bufs, designs = {}, {}

    def turns(pick, one):
        """The lanes of a choice taking turns, the rest off the card -> (peak allocated, most held) of it."""
        for q, lane in lanes.items():                             # the plants of the choice on the card, as a run's lanes are
            if q not in pick:
                parked(lane.prog)
            elif kind(lane.prog.device) != kind(dev):
                moved(lane.prog, dev)
        for q, b in list(bufs.items())+[(q, d.buf) for q, d in designs.items()]:
            b.to(dev if q in pick else 'cpu')
        free(); reset()
        for r in range(rounds):
            for q in pick:
                if len(pick) > 1:
                    free()
                mark.clear(); c0 = retries(dev)
                t = time.time(); one(q, lanes[q]); sync(); took[q] = time.time()-t
                end = retries(dev); c1 = {q2: end[q2]-c0[q2] for q2 in RETRIES}
                line = dict(phase=one.__name__, pick='+'.join(pick), size=q, round=r, s=round(took[q], 1), **{q2[4:]: v for q2, v in c1.items()})
                if 'c' in mark:                                   # the critic's fit apart
                    line.update(fit_s=round(t+took[q]-mark['t'], 1), fit_alloc_retries=end['num_alloc_retries']-mark['c']['num_alloc_retries'])
                if cuda:
                    line.update(gb=[gb(torch.cuda.max_memory_allocated()), gb(torch.cuda.max_memory_reserved())])
                k = f'{one.__name__} {line["pick"]}'
                counts[k] = {q2: counts.get(k, {}).get(q2, 0)+c1[q2] for q2 in RETRIES}
                print(dumps(line), file=sys.stderr, flush=True)   # (a bench stopped on the way still says what each window took)
                alive()                                           # (--stall: of a window)
        return gb(peak()), gb(held())

    def graph(q, lane):
        lane.label = torch.full_like(lane.label, False)
        update(net, critic, opt, copt, lane, a, dict(bc=0., sim=1., value=a.value), gen, temps[q])

    def gathered(q, lane):
        lane.label = torch.full_like(lane.label, True)
        if q not in bufs:
            bufs[q] = Buffer(a.keep*lane.prog.n, per_pairs(a.batch, q))
            if a.design:
                designs[q] = Designs(a, None, lambda: 1 << 31, dev, net.pairs.nk, None if q == DEFAULT else dict(dims=q), every=1 << 30,
                                     b=per_pairs(a.design, q), n=per_pairs(a.design_set, q), keep=per_pairs(a.design_keep, q))
                while designs[q].buf.n < designs[q].b:
                    designs[q].more(net)
        imitate(net, critic, opt, copt, lane, bufs[q], a, gen, designs.get(q))

    a.window = wins[1]
    by = {'+'.join(pick): turns(pick, graph) for pick in picks}
    out['row_s'] = {q: round(x*math.ceil(HORIZON/wins[1]), 1) for q, x in took.items()}
    if cuda:
        out.update(peak_gb_windows=max(x[0] for x in by.values()), reserved_gb_windows=max(x[1] for x in by.values()), gb_windows_by=by)
    if a.replay:
        opt.zero_grad(set_to_none=True); copt.zero_grad(set_to_none=True)
        a.window = wins[0]
        by = {'+'.join(pick): turns(pick, gathered) for pick in picks}
        out.update(row_s_replay={q: round(x*math.ceil(HORIZON/wins[0]), 1) for q, x in took.items()}, replay=a.replay, keep=a.keep,
                   batch={q: x.batch for q, x in bufs.items()}, design={q: x.b for q, x in designs.items()}, design_every=max(1, getattr(a, 'design_every', 1)),
                   buffer_gb={q: round(sum(v.numel()*v.element_size() for v in x.x.values())/2**30, 2) for q, x in bufs.items()})
        if cuda:
            out.update(peak_gb=max(x[0] for x in by.values()), reserved_gb=max(x[1] for x in by.values()), gb_by=by)
    out['alloc_by'] = counts
    print(dumps(out), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path)
    ap.add_argument('--envs', type=int, default=256, help='episodes of a lane')
    ap.add_argument('--lanes', type=int, default=3, help='batches of episodes taking turns, each at another point of its episodes')
    ap.add_argument('--mech', type=float, default=0., help='share of the training episodes on plants of the engine (as core.build); with --life, '
                    'of the lane that runs its episodes whole (they take seconds each to draw)')
    ap.add_argument('--mech-groups', nargs='*', default=None, metavar='WEIGHT:NAME,NAME', help='the mechanistic plants come a few kinds at a time: a batch that has '
                    'any (--mech-batches) has --mech of its episodes on the kinds of one of these groups, taken by the weights (10:qtank,ph,vdv,excstr,bio,evap,'
                    'furnace,gasnet,hex,comp 1:column 1:rsr). What the engines cost a decision goes with the kinds in a batch, not with the plants. Rows: mech, '
                    'per lane the group its plants took (None: none)')
    ap.add_argument('--mech-batches', type=float, default=1., help='with --mech-groups: the chance a new batch of plants has mechanistic plants at all '
                    '(--mech-batches .25 --mech .33: a twelfth of the episodes, in a quarter of the batches)')
    ap.add_argument('--mech-groups-stay', nargs='*', default=None, metavar='WEIGHT:NAME,NAME', help='the groups of the lanes that keep their plants (--stay), where '
                    'they are others: a batch there runs --stay episodes, so a slow kind that fits a share of the time on the lanes that take new plants '
                    'overruns it there in one batch. Rows: mech counts them on from the groups of --mech-groups (None: as --mech-groups)')
    ap.add_argument('--mech-batches-stay', type=float, default=None, help='--mech-batches on the lanes that keep their plants (None: --mech-batches)')
    ap.add_argument('--sizes', nargs='*', default=None, metavar='WEIGHT:SIZE[@PLANTS]', help='plants of these sizes (aio.sim.prior.SIZES; 4:12x5 1:24x10 1:48x20), '
                    'one size a batch, taken by the weights by the batch\'s own seed, in both phases alike. A batch of a size other than the default has as many '
                    'pairs as one of --envs plants (768, 192, 48) unless @PLANTS says, and the generator\'s own changes with room for them (--nseg; not --menu '
                    'or --menu-mix: they are not checked on those sizes). Each size has its own buffer of windows (--batch scaled to its pairs), reference of '
                    'the guard, and pool of problems of design (--design, --design-set, --design-keep scaled to its pairs). Rows: sizes (per lane), t_sizes '
                    '(seconds of the row per size), plants_by. None: every batch 12x5 of --envs plants')
    ap.add_argument('--val-sizes', nargs='*', default=None, metavar='SIZE@PLANTS', help='at every validation also a fixed batch of each of these sizes (24x10@64 48x20@32), '
                    'on its own references\' scale (rows: val_sizes; run.json: val_size_refs)')
    ap.add_argument('--size-stop', type=float, default=0., help='with --val-sizes: once the simulator gradient is in the objective, the run stops (and keeps its weights) '
                    'when the points on any of those sizes fall more than this below the last reading with the teacher alone, or the run\'s start where it had '
                    'none (a run from weights that imitated). 0: never')
    ap.add_argument('--bench-size', default=DEFAULT, help='--bench: the plants of this size (--envs of them)')
    ap.add_argument('--window', type=int, default=32, help='decisions differentiated through at a time')
    ap.add_argument('--window-min', type=int, default=4, help='shortest stretch of an episode whose gradient grows (--guard cut)')
    ap.add_argument('--no-lean', dest='lean', action='store_false', help='keep the networks\' graphs of the whole window (more memory, a little faster)')
    ap.add_argument('--fit-chunk', type=int, default=256, help='episodes of the critic\'s fit that go through at a time (0: all)')
    ap.add_argument('--guard', choices=('cut', 'open', 'cap'), default='cut', help='what is done about an episode whose gradient was held')
    ap.add_argument('--spike', type=float, default=10., help='a step of the simulator gradient whose norm is above this multiple of the median of the last 32 is not taken (0: all are)')
    ap.add_argument('--cap', type=float, default=10., help='an episode\'s gradient at a move is held to this multiple of what is usual (the running median of the episodes\' largest)')
    ap.add_argument('--relax', type=int, default=4, help='windows without being held after which an episode gets one halving back')
    ap.add_argument('--gamma', type=float, default=.99)
    ap.add_argument('--lam', type=float, default=.95)
    ap.add_argument('--tau', type=float, default=.25, help='bands over which the count of rows outside is softened (as the Oracle)')
    ap.add_argument('--cost-w', type=float, default=.1, help='share of the Huber cost in the objective (as the Oracle)')
    ap.add_argument('--log-w', type=float, default=0., help='weight of log(1 + bands outside) in the objective: a far term that does not grow with the deviation as the Huber cost does')
    ap.add_argument('--smooth', type=float, default=0., help='weight of the squared change of the move (before its noise) from one decision to the next, per MV: chatter costs, a ramp does not')
    ap.add_argument('--rev-smooth', type=float, default=0., help='reversal-only anti-chatter weight on relu(-dm*pm) per MV: zero drag on monotonic moves/ramps, penalizes sign flips only')
    ap.add_argument('--floor', type=float, default=.1, help='objective per decision below which an episode counts as settled')
    ap.add_argument('--fail-cost', type=float, default=10., help='objective per decision of an episode that failed, on top of the most it paid while it ran (every row outside is 1)')
    ap.add_argument('--stops', action='store_true', help='the simulator gradient passes straight through a level\'s physical stops as it does through the MV limits: a level held at its stop '
                    'answers no valve, and a quarter of the levels that drift off end there (11-17, aio.sim.core.step, `stops`). The plant is the same')
    ap.add_argument('--hard', action='store_true', help='the simulator gradient of what the plant really does: none through an MV limit or inside a valve\'s friction band (aio.sim.core.step, `through`); for comparison')
    ap.add_argument('--noise', type=float, default=.05, help='spread of the noise on every move in training, before the tanh (0: none)')
    ap.add_argument('--known', type=float, default=.25, help='share of the episodes whose actor is told the true responses')
    ap.add_argument('--critic-known', type=float, default=.75, help='the same for the critic')
    ap.add_argument('--id-coef', type=float, default=.1, help='weight of the identification readout')
    ap.add_argument('--id-sign', action='store_true', help='the identification readout is taught two answers per pair in place of the responses at the knots: whether '
                    'the row answers the MV, and which way (`read_error`). In training alone: no move is made from it, the network runs as before')
    ap.add_argument('--bc-own', action='store_true', help='with --law: the teacher\'s own move is what is imitated, not the law\'s held to the limits (the law\'s readouts stay labels)')
    ap.add_argument('--law-path', choices=('read', 'labels'), help='with --law: whether the MVs take in the product of the law\'s read-outs (read: as with the running sums) or the read-outs are taught and nothing running reads them (labels: as with --reader). Unsaid: as each has been. The two by two of sums or reader, read or labels, tells which of the two a difference comes from')
    ap.add_argument('--seq', nargs='?', const='d192,l4,h6', default=None, help='the plain attention network (aio.seqnet) in place of actor and pair pathway: every row and every MV a token at every '
                    'decision, attention over time and over the channels, the move read from the MV\'s token; no running sums, no reader, no law. Its spec: d width, l layers, h heads. With --replay --no-stats')
    ap.add_argument('--seq-ckpt', action='store_true', help='with --seq: checkpoint each Transformer layer body inside step_graph (4x-8x lower VRAM, enables large batch sizes on GPU)')
    ap.add_argument('--compile', action='store_true', help='compile fixed-arity per-layer step cores, Erlang/DeltaNet update, and critic via torch.compile (CUDA/Triton; auto-falls back if unavailable)')
    ap.add_argument('--priv-critic', action='store_true', help='training-only differentiable Markov state-value critic on true simulator state (exact gradients through yt, v, fl, modes, and delay line buf)')
    ap.add_argument('--opt', choices=('adamw', 'muon'), default='adamw', help='actor optimizer: adamw (default) or hybrid muon (Newton-Schulz orthogonalized updates on 2D hidden matrices + AdamW on rest)')
    ap.add_argument('--bias-lr-mult', type=float, default=1., help='LR multiplier on attention routing biases (pair, lag_pair, lag_own, lag_b)')
    ap.add_argument('--muon-mom', type=float, default=.95, help='with --opt muon: Nesterov momentum for Muon parameter group')
    ap.add_argument('--val-cache', type=Path, default=None, help='directory to cache validation reference rollouts across runs/rollovers')
    ap.add_argument('--ppo', action='store_true', help='no teacher and no gradient through the plant: proximal policy optimisation on windows run with moves drawn around the means '
                    '(--noise their spread before the tanh; with --bc-iters 0; `ppo`)')
    ap.add_argument('--ppo-epochs', type=int, default=4, help='--ppo: passes over a window\'s episodes')
    ap.add_argument('--ppo-clip', type=float, default=.2, help='--ppo: how far the ratio of the new policy to the old counts')
    ap.add_argument('--ppo-kl', type=float, default=.02, help='--ppo: the passes over a window stop where the policy is 1.5 times this far (KL) from the one that ran it')
    ap.add_argument('--faults', type=float, default=0., help='share of the training episodes with one fault after their first event: a valve stuck, an instrument gone bad, '
                    'frozen or drifting (aio.sim.faults.train_draw); the validation batches have none')
    ap.add_argument('--wd', type=float, default=0., help='AdamW with this weight decay on the weight matrices (0: Adam, as before)')
    ap.add_argument('--beta2', type=float, default=.95, help='with --wd: Adam\'s second beta')
    ap.add_argument('--warm-steps', type=int, default=0, help='steps over which each step\'s rate rises from 0 at the start of a run')
    ap.add_argument('--batch-min', type=int, default=1, help='--replay: windows a step takes at the least, whatever the size of the plants')
    ap.add_argument('--seq-window', type=int, default=128, help='with --seq: decisions of history a decision sees (kept in the state)')
    ap.add_argument('--reader', nargs='?', const=True, default=False, help='the learned reader (aio.reader) in place of the running sums: with --no-stats; --law beside it teaches E, K and V as labels that nothing running reads. With a value, comma-separated: const (a key that is always 1: the rows\' own drift), roundsN (N rounds of one shared step in place of 4 of their own)')
    ap.add_argument('--no-stats', action='store_true', help='the network without the running sums (aio.net.Stats); a resumed run keeps what it has')
    ap.add_argument('--slow', type=float, default=0., help='a second set of the running sums that forgets over this many decisions (the first: 200), read through a layer '
                    'that starts at zero (aio.net, `slow`); 0: none. For a unit the network has run for a while with its state carried on. A resumed run keeps what it has')
    ap.add_argument('--arx', action='store_true', help='every set of the running sums is also fitted with the rows\' own past beside the MVs\' moves, read through a layer that '
                    'starts at zero (aio.net, `arx`): in closed loop the plain sums put what the loop feeds back of a row\'s noise down to the MVs. Weights that have it keep it')
    ap.add_argument('--dv', action='store_true', help='with --arx: those fits take the rows that read a measured disturbance beside the MVs (aio.net, `dv`); the same features, '
                    'no new weights. Weights that have it keep it')
    ap.add_argument('--diff', default='', help='read-outs that give, and are taught, the change from one lag to the next instead of the value (aio.net, `diff`): S the responses '
                    '(knot to knot), E the E read-out (label lag to label lag), SE both; K and V as they are. The values are the changes added up; nothing else changes. '
                    'A run from scratch, or from weights that have it (they keep theirs); weights without it first get their read-outs fitted again: python -m aio.relabel')
    ap.add_argument('--unseen', action='store_true', help='with --law: a read-out of what the instruments do not show of every row (aio.net, `unseen`: the true value less the '
                    'reading; the rate the row goes on at, a level\'s net inflow), taught the true ones by Huber\'s loss in weighted bands, and given to the E read-out (the told '
                    'episodes: the true ones). Weights without it take it at its zero start. In the rows: unseen, gap_un, rate_un, rate_rk and the last two by decisions since an event')
    ap.add_argument('--told', action='store_true', help='the pairs are given the share of the responses they have that was told (aio.net, `told`: 1 in a told episode '
                    'and in a problem of design, 0 where the network reads them itself, as it does when it runs; through a layer without a bias that starts at '
                    'zero), so that the K read-out tells the two apart: taught both, problems of design and its own reading pull against each other without it '
                    '(11-12, offline: 60.7 / 86.1 against 69.4 / 87.8 fed the true responses, 55.4 / 82.5 against 62.8 / 81.4 its own reading; plants with a level '
                    '/ the others). Weights without it take it at its zero start. The critic has none')
    ap.add_argument('--tied', type=int, default=0, help='rounds of the control half with one set of weights and products in the messages (0: two rounds of their own weights); a run from weights keeps what they have')
    ap.add_argument('--drive', type=float, default=0., help='share of the imitation episodes the teacher drives; the student drives the others '
                    'and is labelled by the teacher on the states it gets to')
    ap.add_argument('--drive-noise', type=float, default=.2, help='spread of the noise on the teacher\'s moves in those episodes, before the tanh')
    ap.add_argument('--bc-floor', type=float, default=0., help='the teacher\'s weight does not fall below this: that share of the episodes stays '
                    'for imitation beside the simulator gradient')
    ap.add_argument('--bc-hold', type=int, default=0, help='rows the teacher is all of the objective before it fades: the simulator gradient '
                    'starts from a student that already holds a unit')
    ap.add_argument('--replay', type=int, default=0, help='while the teacher is all of the objective (--bc-hold): windows are gathered without '
                    'a graph and kept, and after each this many steps are made on windows drawn from those kept (0: a window is '
                    'differentiated as it is run, one step each)')
    ap.add_argument('--batch', type=int, default=256, help='windows of a step of --replay, each of another episode')
    ap.add_argument('--keep', type=int, default=12, help='windows of a lane kept for --replay (the oldest are written over)')
    ap.add_argument('--law', action='store_true', help='with --replay: the network reads out the things the teacher\'s move is the product of (aio.law: '
                    'E, how far every tracked row will be from where it can settle if the MVs stop now; K, every MV\'s first change of flow per unit '
                    'of it; V, the valve curve), is taught them on every episode (rows: head, gain, valve, law), multiplies them and feeds the product '
                    'to its head; the told episodes get the teacher\'s own; and the move imitated is the law\'s, not the teacher\'s (1.4 points '
                    'less in closed loop, and a function a network with E, K and V can make exactly). A run from scratch (a resumed run keeps what it has)')
    ap.add_argument('--law-coef', type=float, default=1., help='weight of the four terms of --law')
    ap.add_argument('--meter', action='store_true', help='the teacher that is imitated, whose E, K and V are the labels of --law, corrects its prediction from '
                    'what the instruments report alone and does not see the unmeasured loads move (aio.sim.core.Program.set_teacher: late, loads off, meter '
                    'on). With --unseen the told episodes are then not told what the instruments do not show: the E read-out takes the network\'s own '
                    'reading in every episode. The validation batch gets two more references, the step-test DMC and this teacher from the instruments '
                    'alone (dmc_meter, teacher_meter), beside the scale, which stays as it was')
    ap.add_argument('--design', type=int, default=0, help='--law, while the teacher is all of the objective (--replay): every step of learning also takes this many '
                    'problems of design, of plants that are drawn and not run: the packet of the first decision and the true responses in, the '
                    "teacher's K out (`Designs`; rows: `design`, and `design_new` on plants never seen). 0: none")
    ap.add_argument('--design-set', type=int, default=256, help='plants of a set of them, as the builders draw them')
    ap.add_argument('--design-keep', type=int, default=32768, help='plants of the pool they are taken from (the oldest are written over)')
    ap.add_argument('--design-coef', type=float, default=1., help='weight of their term, times --law-coef (1: as `gain`, which is the same error on the episodes)')
    ap.add_argument('--design-every', type=int, default=1, help='one step of learning of this many takes the problems of design (--design is what such a step takes: '
                    'with 4 here and four times as many there, as many problems for a quarter of what they cost; their pull on average is then --design-coef over this)')
    ap.add_argument('--law-keep', nargs='?', const='EKV', default='', metavar='EKV', help='with --law: once the simulator gradient is in the objective, these readouts '
                    '(letters of E, K, V; alone: all three) stay taught towards the teacher\'s on every episode (their product and the move are the cost\'s). '
                    'Without it nothing holds them there: v2-rl (11-12) moved E off the teacher\'s (slope at 16 decisions .89 -> .55) and its law did 1 %% of '
                    'the teacher\'s in steady state, from 7 %%. E alone is what to keep: E read right is worth 11.6 points and v2-rl\'s move of it under a '
                    'point, while its smaller K and V are worth 2 (readouts of one network told to the other, 256 episodes)')
    ap.add_argument('--knots', type=int, default=N.NLAB, choices=sorted(N.KNOTS), help='lags the responses are read out at and told at (aio.sim.model.KNOTS: '
                    'nine are too few for --formula). A run from scratch')
    ap.add_argument('--formula', action='store_true', help='with --law: E is worked out from the responses read and the moves made (aio.formula; no weights), '
                    'and the readout of E learns a factor and a correction on it, in place of the learned kernel on the filtered moves. A run from scratch')
    ap.add_argument('--hybrid', action='store_true', help='with --formula: the heading read-out gets the learned kernel\'s pending effects beside the formula\'s E (aio.net, `hybrid`)')
    ap.add_argument('--solve', action='store_true', help='with --law: the move is the teacher\'s design solved on the responses read for that E (aio.formula), '
                    'in place of the readouts of K and V (rows gain and valve stay empty). A run from scratch')
    ap.add_argument('--given', action='store_true', help='with --law: E by the formula and the move by the design solved in the told episodes, which are told '
                    'the responses alone (what a plant whose step tests are handed over gives); the learned readouts of E, K and V in the others, as '
                    'with --law alone. In place of --formula and --solve (on the responses a network reads itself the formula is far worse than the '
                    'learned readouts). A run from scratch')
    ap.add_argument('--unpaired', type=float, default=0., help='this share of the training episodes get the packet with nothing published about the pairing: '
                    'which row an MV is published with (the block behind the MVs) all zero, the sign slot +1, nothing mirrored, so that the '
                    'network works in the plant\'s own frame and has to find out which way a valve moves a row (mixed.canonical); the labels and '
                    'the teacher\'s law come in that frame. Both are the plant\'s true gains handed over, as the kind of equipment or a step test '
                    'would be. The network is validated and runs without them (aio.net.Policy); `pts_paired` is the same validation with '
                    'both published: what they are worth to it. 0: every episode is told (as before). A run from scratch')
    ap.add_argument('--lead', type=float, default=0., help='this share of the episodes of the simulator gradient start in the teacher\'s hands: its moves go to the plant '
                    '(and are the last moves the network is given) up to a decision drawn for each, 1 to --lead-to decisions after its first event; from there '
                    'the network has the plant, with a history in which the valves were moved the right way and the rows answered. What it does with '
                    'such a history is what the gradient is then of; nothing is imitated. The network is validated and runs on its own from the start')
    ap.add_argument('--pre-cost', type=float, default=1., help='with --unpaired: the share of the objective the blind episodes pay before their first event '
                    '(nothing is scored there): room to try the valves and read the answers, the network\'s own plant test, learned from the plant\'s gradient')
    ap.add_argument('--rel-id-coef', type=float, default=0., help='with r1 (--seq ...,r1): training-only auxiliary pairwise (live, sign) BCE loss on unpublished (link=0) and blind pairs; zero runtime parameters')
    ap.add_argument('--bc-told', action='store_true', help='with --unpaired: the blind episodes are never imitation episodes (the teacher knows the plant and never '
                    'moves a valve to learn something; imitated there, the best answer is not to move)')
    ap.add_argument('--lead-to', type=int, default=360, help='--lead: the latest hand-over, in decisions after the first event')
    ap.add_argument('--probe', type=float, default=0., help='with --replay: before an episode\'s first event (nothing is scored there) the moves made '
                    'carry noise of this spread before the tanh in place of --noise, the teacher\'s move on the state reached staying the target: '
                    'the unit is seen answering its valves before it has to be controlled. The network learns to say when (the second output '
                    'of its head) and runs so (aio.net.Policy): its own plant test. Imitation alone never moves a valve to learn something: '
                    'the teacher knows the plant')
    ap.add_argument('--ahead', type=int, default=1, help='with --builders: batches of new plants of a lane in the making beyond the one it takes '
                    '(a batch of the slowest mechanistic kinds takes 500 s or more to draw, longer than the two rows it lasts: drawn further ahead, the run '
                    'does not wait for it; rows: t_build, the seconds it waited)')
    ap.add_argument('--builders', type=int, default=0, help='processes that draw the next batches of plants ahead, on the CPU (0: drawn in the '
                    'run\'s own process when a lane needs them, the GPU waiting)')
    ap.add_argument('--builder-threads', type=int, default=8, help='CPU threads of each of them')
    ap.add_argument('--builder-qos', default='below,e', help='priority,cores of the builder processes (as --cpu-qos; "off": the run\'s own): on the '
                    'E-cores, out of the way of the thread that feeds the GPU')
    ap.add_argument('--life', type=int, default=0, help='with --replay: decisions after which the plants of a lane are replaced by new ones, on every lane '
                    'but the first, which runs its episodes whole (0: all whole). What is read on plants not seen rises with the number of '
                    'different plants, not with the decisions on each (11-11); an episode has 2 of its 8 events in its first 192 decisions')
    ap.add_argument('--menu', nargs='*', default=None, metavar='FIELD=VALUE', help='what changes on the training plants, as fields of aio.sim.prior.Menu: '
                    'rate=2,2,1,1,1 (setpoint moves, measured disturbances, unmeasured loads, process changes, the rest: each against its usual share) '
                    'pace=2 move= step= proc= wander= turn= joint= pulse= exc= drop=dead,... (kinds left out: they are then met in the exams alone); '
                    'and what plants they are: nd=1,2 (lo,hi of a count: nm, nd, nq, nz, nr), blind=1. None: the generator as it is. '
                    'The validation batches always keep the generator\'s own')
    ap.add_argument('--menu-stay', nargs='*', default=None, metavar='FIELD=VALUE', help='with --stay: the menu of the lanes that keep their plants (default: --menu)')
    ap.add_argument('--menu-plain', nargs='*', default=None, metavar='FIELD=VALUE', help='the menu of every --plain-th lane (default: the generator as it is)')
    ap.add_argument('--plain', type=int, default=0, help='every this-many-th lane (the last of each such group) takes --menu-plain instead: with --lanes 3 --plain 3, '
                    'a third of the plants (0: no lane)')
    ap.add_argument('--menu-mix', nargs='*', default=None, metavar='CHANCE:FIELD=VALUE+FIELD=VALUE', help='groups of --menu\'s fields every new batch of plants takes by chance, '
                    'each group independently, laid over --menu (--menu-stay on the lanes that keep their plants; never on a --plain lane): '
                    '2/3:nd=1,2+rate=1,3,1,1,1 1/2:ride=.8+exc=.3 1/3:shift=.2. The draw is the batch\'s seed\'s own (rows: mix, per lane which groups its plants took)')
    ap.add_argument('--nseg', type=int, default=0, help='operating points an episode of the training lanes has room for (0: the generator\'s 12; more events need more; '
                    'prior.NSEG_STAY = 16: a window that carries on has room for its events to the end). Not on a --plain lane without a menu')
    ap.add_argument('--val-nseg', type=int, default=0, help='the same for the windows of --val-stay and of --val-exams (0: the batches as they are; another number: other batches, '
                    'their references run anew and kept under their own names)')
    ap.add_argument('--stay', type=int, default=1, help='episodes in a row the plants of every other lane (lanes 0, 2, ..) are run for: each the next episode '
                    'of the same units, the plant\'s state and all memories carried on (1: every lane takes new plants every episode). Not with --life')
    ap.add_argument('--val-stay', type=int, default=0, help='episodes in a row a fixed batch is run for at a validation, to read what staying on a unit is worth '
                    '(`Stay`; in a row: stay; 0: off). With --score: of that checkpoint')
    ap.add_argument('--val-stay-n', type=int, default=64, help='plants of that batch')
    ap.add_argument('--val-stay-every', type=int, default=1, help='validations between two of them (it is --val-stay episodes of two rollouts each)')
    ap.add_argument('--val-exams', nargs='*', default=None, metavar='NAME', help='the fixed exams of a long stay, read at the same validations (aio.sim.stay.EXAMS: '
                    'pulse dead quiet mech mech_new, kinds of change and plants the training ground does not have; usual, the same plants as the generator '
                    'has them; blind sparse dense): --val-stay-n plants for --val-stay windows each (64 x 8 where --val-stay is not given), the reading of '
                    '--val-stay without its short-memory control (rows: exams; run.json: exam_refs). With --score: the checkpoint on them')
    ap.add_argument('--val-exams-refs', type=Path, default=None, help='a directory the references of the exams are kept in (run once per exam, size and device)')
    ap.add_argument('--ema', type=float, default=0., help='the actor\'s weights averaged over its steps with this decay (.999: the last 1,000 steps) '
                    'are what is validated and kept (0: the weights as stepped)')
    ap.add_argument('--ema-device', choices=('cuda', 'cpu'), default='cuda', help='device for EMA weights (cpu saves 524 MB VRAM on local GPUs)')
    ap.add_argument('--val-raw', action='store_true', help='with --ema: the weights as stepped are validated too (`raw` in a row)')
    ap.add_argument('--sim-window', type=int, default=0, help='the window once the simulator gradient is in the objective (0: --window). '
                    'Imitation needs no long one; the simulator gradient of a short one misses what the moves cost later')
    ap.add_argument('--bc-iters', type=int, default=30, help='rows over which the teacher fades out (0, and with --init: no teacher; a resumed run keeps its schedule). '
                    'Its weight is also the share of the episodes that are for imitation: an episode is for imitation or for the simulator gradient')
    ap.add_argument('--bc-coef', type=float, default=10., help='imitation weight while the teacher is in (the cost gradient is far larger)')
    ap.add_argument('--value-warm', type=int, default=3, help='rows over which the critic\'s value enters the actor\'s objective')
    ap.add_argument('--value', type=float, default=1., help='its weight there. 0: the cost paid in the window alone (and with --after the teacher\'s after it); the critic is '
                    'fitted all the same. With units that stay (--stay) its value rises along the stay to many times the cost then paid: 0 there until it is anchored')
    ap.add_argument('--after', type=int, default=0, help='decisions the teacher has the plant for after a window with the simulator gradient: what they cost is '
                    'added to the window\'s own (`handed`). 0: the window ends at its last decision')
    ap.add_argument('--after-by', choices=('teacher', 'hold'), default='teacher', help='who has the plant for --after: the teacher, or nobody (the MVs stay where the window left them)')
    ap.add_argument('--heads', type=int, default=2, help='heads of the critic')
    ap.add_argument('--fixed', type=int, default=0, help='the critic\'s heads are of fixed horizons: head m says the cost of the next this many times m+1 decisions and is fitted '
                    'on what they cost and on the head before it, never on itself (--heads of them: the actor\'s objective has the longest). 0: both heads the cost of all that '
                    'is to come, fitted on their own value after the window')
    ap.add_argument('--critic-iters', type=int, default=4, help='fits of the critic on each window')
    ap.add_argument('--critic-lr', type=float, default=3e-4)
    ap.add_argument('--lr', type=float, default=3e-4, help='after --warmup rows; falls to a tenth along a cosine over --hours')
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--lr-jump', action='store_true', help='with --resume: the next row may step at more than twice the rate the run ended on (refused without it; '
                    'run.json: lr_from)')
    ap.add_argument('--accum', type=int, default=1, help='windows whose gradients (each clipped) are averaged into one step of the actor: with --envs an Nth, '
                    'the batch of a step is as it was on a card N times the size (16 GB: docs/sim.md). The critic is fitted on every window as before')
    ap.add_argument('--clip', type=float, default=1., help='norm clip of the gradient of a step (actor and critic each)')
    ap.add_argument('--row', type=int, default=0, help='windows per row of metrics.jsonl (0: one episode\'s worth)')
    ap.add_argument('--rows', type=int, default=10**9)
    ap.add_argument('--hours', type=float, default=2.)
    ap.add_argument('--cosine-by', choices=('hours', 'updates', 'samples'), default='hours', help='progress unit for cosine LR decay (default: hours; updates/samples for cross-GPU parity)')
    ap.add_argument('--total-updates', type=int, default=0, help='target total optimizer updates when --cosine-by updates')
    ap.add_argument('--total-samples', type=int, default=0, help='target total env samples when --cosine-by samples')
    ap.add_argument('--eval-every', type=int, default=5, help='rows between validations (0: disable all in-loop/start/end validation for zero-CU async eval)')
    ap.add_argument('--eval-snap-every', type=int, default=0, help='rows between lightweight atomic BF16+FP32-1D eval snapshots in OUT/eval/rowN.pt (0: off)')
    ap.add_argument('--keep-eval-snaps', type=int, default=6, help='with --eval-snap-every: keep at most this many newest OUT/eval/rowN.pt snapshots (0: keep all)')
    ap.add_argument('--eval-snap-raw', action='store_true', help='with --eval-snap-every and --ema: also pack raw actor weights in eval snapshots')
    ap.add_argument('--focal-alpha', type=float, default=0., help='bounded online focal weighting exponent on episode sim loss (0: uniform; e.g. 0.5)')
    ap.add_argument('--hard-seeds', type=Path, default=None, help='JSON file of mined hard plant batch seeds ({"seeds": [...]}) for --hard-share mixing into 12x5 batches')
    ap.add_argument('--hard-share', type=float, default=0., help='share of non-plain 12x5 batches drawn from --hard-seeds with fresh event_seeds (0: off)')
    ap.add_argument('--diloco-hub', default='', help='Elastic MuLoCo Hub base URL (e.g. http://100.79.108.0:8765); empty disables distributed sync')
    ap.add_argument('--diloco-h', type=int, default=32, help='inner optimizer updates between MuLoCo Hub syncs (at exact row boundaries)')
    ap.add_argument('--diloco-worker', default='', help='worker identifier sent to --diloco-hub (default: OUT dir name or hostname)')
    ap.add_argument('--diloco-compress', choices=('int8', 'bf16'), default='int8', help='pseudo-gradient compression for --diloco-hub')
    ap.add_argument('--diloco-timeout', type=float, default=180., help='HTTP timeout in seconds for --diloco-hub /sync requests')
    ap.add_argument('--quiet-hours', default='', help='local quiet window (HH:MM-HH:MM, e.g. 00:00-10:30 for guomao) during which training pauses at 0%% GPU and resumes automatically')
    ap.add_argument('--quiet-tz', default='Asia/Shanghai', help='IANA timezone for --quiet-hours (default: Asia/Shanghai)')
    ap.add_argument('--save-every', type=int, default=0, help='every row whose number is a multiple of this keeps its checkpoint as rowN.pt beside last.pt, '
                    'and so does the row the run ends on (0: last.pt and best.pt alone; decoupled from --eval-every)')
    ap.add_argument('--keep-saves', type=int, default=0, help='with --save-every: keep at most this many newest rowN.pt snapshots (0: keep all; never deletes last.pt or best.pt)')
    ap.add_argument('--val-n', type=int, default=256)
    ap.add_argument('--val-seed', type=int, default=10**6)
    ap.add_argument('--val-mech', type=float, default=0., help='share of the validation episodes on plants of the engine')
    ap.add_argument('--refs', type=Path, help='file of aio.sim.oracle for the validation batch: the Oracle joins the scale\'s 100')
    ap.add_argument('--no-known-val', action='store_true', help='skip the validation rollout that is told the true responses')
    ap.add_argument('--lit', type=int, default=0, help='ignored (earlier command lines still run): the literature columns were evaluated on the previous simulator, which is gone')
    ap.add_argument('--init', type=Path, help='start from the weights of this checkpoint (any earlier AIO run)')
    ap.add_argument('--init-gauge', type=float, default=1., help='with --init --seq: scale residual-writing weights/biases (row, mv, out_t/c/l, ff.2) by this factor (0 < c <= 1) '
                    'to deflate Pre-LN residual stream norm while preserving the LayerNormed forward function (1: off)')
    ap.add_argument('--init-pairs', type=Path, help='start the pair pathway from the weights of this file where names and shapes are the same (the learned reader taught '
                    'alone first: experiments.diagnostics.reader_probe --save); the rest starts as it would, the teacher\'s schedule stays. Not with --resume\'s weights: a resumed run keeps its own')
    ap.add_argument('--resume', action='store_true', help='continue from OUT/last.pt (written at every row): weights, optimizers, counters')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--gen-device', default='cpu', help='where training episodes are drawn and their models designed (cuda: faster builds)')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--cpu-qos', default='normal,p',
                    help='priority,cores of this process (monitoring.cpu_qos, as rsac): the loop is one launch-bound host '
                         'thread, and a below-normal/EcoQoS thread lands on the E-cores of baolong\'s hybrid CPU (11-09: 2x '
                         'slower). "off": keep the launcher\'s placement. No effect off Windows; results are the same')
    ap.add_argument('--bench', action='store_true')
    ap.add_argument('--stall', type=float, default=900., help='seconds without a sign of life (a window, or a batch of a validation or of the references, done; a --bench: its whole '
                    'time) after which where every thread stands is written to OUT/stalls.txt (a --bench: stderr), and again after as long (0: never): a run that stops moving says where')
    ap.add_argument('--restarts', type=int, default=0, help='N: the run is watched by a process of its own (`watched`): after twice --stall without a sign of life it is ended and started '
                    'again with --resume from the last row, up to N times (OUT/restarts.jsonl). 0: not watched')
    ap.add_argument('--card-gb', type=float, default=0., help='the most of the card this process\'s allocator may hold, GB (`cap_card`: past it the blocks it keeps are '
                    'given back before an allocation fails; on Windows nothing else stops it running over into shared memory and slowing every row). '
                    '0: what is free when the process starts, less .5 GB; below 0: no cap')
    ap.add_argument('--card-alloc', default='garbage_collection_threshold:0.8', help='the allocator\'s settings, as PYTORCH_CUDA_ALLOC_CONF, set before anything is '
                    'on the card (\'\': its own). The default gives back the oldest blocks it keeps once it holds .8 of the cap (11-13, 24x10 and 48x20 on a '
                    'card: 12.4 GB held and no allocation tried twice, against 14.2 held at the cap and freed and tried again; as fast). '
                    'expandable_segments is not on Windows. Rows: alloc, its counts (`retries`)')
    ap.add_argument('--score', type=Path, help='no training: this checkpoint on the batch of --val-n, --val-seed, --val-mech beside the references (one JSON line)')
    ap.add_argument('--per', type=Path, help='with --score: a file for the scores themselves, plant by plant (and window by window on --val-stay and --val-exams), of the '
                    'checkpoint and of every reference, with which plants have a level: for a reading by another rule than the rows\' (the reference '
                    'tables add a plant\'s windows up before the points are taken; the rows take each window\'s points)')
    a = ap.parse_args()
    explicit = explicit_args(ap, sys.argv[1:])
    a._fit_chunk_explicit = 'fit_chunk' in explicit
    assert a.stall > 0 or not a.restarts, '--restarts: with --stall'
    assert a.mech_groups_stay is None or (a.mech_groups and a.mech_groups_stay), '--mech-groups-stay: groups, beside --mech-groups (none there: --mech-batches-stay 0)'
    if a.restarts and not (a.bench or a.score or os.environ.get(KEPT)):       # the run is a child of this process, which watches it
        a.out.mkdir(parents=True, exist_ok=True)
        sys.exit(watched([sys.executable, *sys.orig_argv[1:]], a.out, a.stall, a.restarts))
    a.cpu_placement, a.lr_from = None, None
    if a.cpu_qos != 'off':
        from monitoring import cpu_qos
        pri, cores = a.cpu_qos.split(',')
        a.cpu_placement = cpu_qos.apply(pri, cores, high_qos=True)
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    if dev.type == 'cuda' and os.environ.get('AIO_TF32', '1') != '0':
        torch.set_float32_matmul_precision('high')
    expandable = _has_expandable_segments(a.card_alloc)
    a.card_gb = cap_card(dev, a.card_gb, a.card_alloc)                         # (run.json, the bench's line: the cap it ran under)
    assert a.seed >= 1 and a.val_seed < 2**32 and 1 <= a.window <= HORIZON and 0 <= a.sim_window <= HORIZON
    assert a.stay >= 1 and not (a.stay > 1 and a.life), '--stay: whole episodes (no --life)'
    windows = (a.window, a.sim_window or a.window)               # while the teacher is alone, and with the simulator gradient
    if a.bench:
        if a.priv_critic and not a._fit_chunk_explicit:
            a.fit_chunk = 0
        SIGN.update(stall=a.stall); alive()
        return bench_mix(a, dev) if a.sizes else bench(a, dev)
    if a.score:
        return score(a, dev)
    a.out.mkdir(parents=True, exist_ok=True)
    SIGN.update(stall=a.stall, file=open(a.out/'stalls.txt', 'a') if a.stall else None, path=a.out/'alive')
    if os.environ.get(KEPT):
        ends_with(sys.stdin.fileno())
    alive(start=True)
    torch.manual_seed(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    last = a.out/'last.pt'
    resumed = a.resume and last.exists()
    assert not (a.probe > 0 and a.bc_iters == 0 and not (resumed or a.init)), (
        '--probe: needs imitation (--bc-iters > 0) or weights that trained `when`; for pure real-cost self-probing use --probe 0 --pre-cost .25')
    ck = safe_torch_load(last if resumed else a.init, map_location='cpu') if resumed or a.init else None
    grown = False
    if resumed:                                                   # a run goes on with the network it has
        restore_resume_config(a, ck, explicit)
        windows = (a.window, a.sim_window or a.window)
        a.no_stats = not N.has_stats(ck['model']['pairs'])
        if not a.seq:
            a.reader = N.reader_of(ck['model']['pairs'])
            a.law_path = ('labels' if N.apart_of(ck['model']['pairs']) else 'read') if N.law_of(ck['model']['pairs']) else None
            had = (N.slow_of(ck['model']['pairs']), N.arx_of(ck['model']['pairs']), N.dv_of(ck['model']['pairs']))
            seen = N.unseen_of(ck['model']['pairs'])                  # (and --unseen: its read-out, and the layer that gives it to E's at its zero start)
            told = N.told_of(ck['model']['pairs'])                    # (and --told: the layer the share told comes in by, at its zero start)
            grown = (bool(a.slow) and not had[0]) or (a.arx and not had[1]) or (a.dv and not had[2]) or (a.unseen and not seen) or (a.told and not told)   # but for the slow sums and the rows' own past: --slow, --arx add them (their layers start at zero: the same function), --dv widens the fits
            a.slow, a.arx, a.dv = had[0] or a.slow, had[1] or a.arx, had[2] or a.dv
            a.diff = N.diff_of(ck['model']['pairs'])
            a.law, a.knots = N.law_of(ck['model']['pairs']), N.knots_of(ck['model']['pairs'])
            a.formula, a.solve = N.worked_of(ck['model']['pairs'])
            a.given = N.given_of(ck['model']['pairs'])
            a.hybrid = N.hybrid_of(ck['model']['pairs'])
    elif ck is not None and 'pairs' in ck['model']:
        assert not set(a.diff)-set(N.diff_of(ck['model']['pairs'])), '--diff on weights whose read-outs give values: fit them again first (python -m aio.relabel)'
        a.diff = N.diff_of(ck['model']['pairs'])                  # the read-outs mean what they do in the weights
        a.slow = N.slow_of(ck['model']['pairs']) or a.slow        # weights that have the slow sums keep theirs; weights without take --slow at its zero start
        a.arx = N.arx_of(ck['model']['pairs']) or a.arx           # the same of --arx
        a.dv = N.dv_of(ck['model']['pairs']) or a.dv              # and of --dv
    if ck is not None and 'pairs' in ck['model'] and not a.seq:
        a.tied = N.tied_of(ck['model']['pairs'])
        a.unseen = N.unseen_of(ck['model']['pairs']) or a.unseen  # weights that have the read-out keep it; weights without take --unseen at its zero start
        a.told = N.told_of(ck['model']['pairs']) or a.told        # the same of --told
    if a.priv_critic and not a._fit_chunk_explicit:
        a.fit_chunk = 0
    assert not (a.cosine_by == 'updates' and a.total_updates <= 0 and not a.diloco_hub), '--cosine-by updates requires --total-updates > 0'
    assert not (a.cosine_by == 'samples' and a.total_samples <= 0 and not a.diloco_hub), '--cosine-by samples requires --total-samples > 0'
    shares = sizes_of(a.sizes, a.envs)                            # --sizes: a size a batch
    assert not (a.formula or a.solve or a.given) or all(x[1] == DEFAULT for x in shares or ()), '--formula, --solve and --given are of the default size alone (aio.formula)'
    assert not a.size_stop or a.val_sizes, '--size-stop reads --val-sizes'
    net, critic = nets(a, dev)
    opt, copt = optimizer(net, a), torch.optim.Adam(critic.parameters(), lr=a.critic_lr, **_fused_kw(critic))
    rel_probe = rel_popt = None
    if getattr(a, 'rel_id_coef', 0.) > 0 and getattr(net.actor, 'rel', 0) > 0:
        from aio import seqnet as SQ
        rel_probe = torch.nn.Linear(SQ.rel_feat_dim(net.actor.rel), 3 if net.actor.rel >= 2 else 2).to(dev)
        rel_popt = torch.optim.AdamW(rel_probe.parameters(), lr=3e-3, weight_decay=1e-2, **_fused_kw(rel_probe))
    c = dict(row=0, windows=0, samples=0, updates=0, drawn=0, best=-float('inf'), hours=0., stayed=0, designs=0,   # stayed: of the batches drawn, those that are a next episode of units already counted
             sized=0, plants_by={}, design_plants=0, turns={})    # turns: new batches drawn per kind of lane (`evenly`)              # sized: of the new batches, those drawn by --sizes, whose plants are counted by size
    acc = dict(g=None, n=0, k=0)                                  # shared across sizes (--accum): every accumulated window is at the same weights, no stale gradients
    temp = dict(usual=None, acc=acc, rel_probe=rel_probe, rel_popt=rel_popt)
    temps = {DEFAULT: temp}                                       # the guard's reference per size (--sizes: a batch's gradients are of its size)
    size_ref = None
    diloco_ck = {}
    if resumed:
        src = ck.get('raw') or ck['model']                        # (a run with --ema: the steps go on from the weights as stepped)
        net.actor.load_state_dict(src['actor']); net.pairs.load_state_dict(src['pairs'])
        heads = set(ck['model']['critic']) == set(critic.state_dict()) and a.fixed == ck['config'].get('fixed', 0)
        # a run that goes on with heads of another kind or number (--fixed, --heads): the critic's trunk as it was, its heads and its optimizer from their start
        critic.load_state_dict(ck['model']['critic'] if heads else {q: x for q, x in ck['model']['critic'].items() if not q.startswith('heads.')}, strict=heads)
        new = {'actor.'+q for q in net.actor.state_dict() if q not in src['actor']} if getattr(a, 'seq', None) else set()   # (--seq from before `when`)
        if grown or new:                                          # the optimizer's state of the weights the run had, by name; the new layer's starts with it
            was = [q for q, _ in net.named_parameters() if q not in new] if new else [
                q for q, _ in N.AIO(not a.no_stats, a.tied, a.law, a.knots, a.formula, a.solve, a.given, *had, a.diff, seen, told).named_parameters()]
            old, sd = ck['opt'], opt.state_dict()
            by = {q: old['state'][i] for q, i in zip(was, old['param_groups'][0]['params']) if i in old['state']}
            sd['state'] = {i: by[q] for (q, _), i in zip(net.named_parameters(), sd['param_groups'][0]['params']) if q in by}
            sd['param_groups'][0].update({q: v for q, v in old['param_groups'][0].items() if q != 'params'})
            opt.load_state_dict(sd)
        else:
            opt.load_state_dict(ck['opt'])
        if heads:
            copt.load_state_dict(ck['copt'])
        if rel_probe is not None and 'rel_probe' in ck:
            rel_probe.load_state_dict(ck['rel_probe'])
            if 'rel_popt' in ck:
                rel_popt.load_state_dict(ck['rel_popt'])
        ck_cfg = ck.get('config') or {}
        old_envs = ck_cfg.get('envs', a.envs)
        old_map = {x[1]: x[2] for x in (sizes_of(ck_cfg.get('sizes'), old_envs) or [(1., DEFAULT, old_envs)])}
        new_map = {x[1]: x[2] for x in (shares or [(1., DEFAULT, a.envs)])}
        same_units = ck.get('guard_units') == GUARD_UNITS
        temp.update(usual=ck['usual'] if same_units else None)   # an older one: measured again at the first window
        temp.update(gns=list(ck.get('gns') or []) if (same_units and old_map.get(DEFAULT) == new_map.get(DEFAULT)) else [])
        for q, x in (ck.get('usuals') or {}).items():             # (the other sizes')
            temps[q] = dict(usual=x['usual'] if same_units else None,
                            gns=list(x.get('gns') or []) if (same_units and old_map.get(q) == new_map.get(q)) else [],
                            acc=acc, rel_probe=rel_probe, rel_popt=rel_popt)
        c.update(row=ck['iter'], windows=ck['windows'], samples=ck['samples'], updates=ck.get('updates', ck['windows']), drawn=ck['drawn'], best=ck['best'], hours=ck['hours'],
                 stayed=ck.get('stayed', 0), designs=ck.get('designs', 0), sized=ck.get('sized', 0), plants_by=dict(ck.get('plants_by') or {}),
                 design_plants=ck.get('design_plants', ck.get('designs', 0)*a.design_set), val=ck.get('val'), turns=dict(ck.get('turns') or {}))
        temp.update(zero=ck.get('zero'))
        size_ref = ck.get('size_ref')
        diloco_ck = dict(ck.get('diloco') or {})
        pg = ck['opt']['param_groups'][0]
        was = pg.get('set_lr', pg['lr'])
        first = scheduled_lr(a, c, c['hours'], hub_lr=diloco_ck.get('hub_lr') if a.diloco_hub else None)
        a.lr_from = dict(was=was, first=first)                    # the rate the run ended on, and the one its next row would step at
        # the cosine counts over the new target: a run that had come down its own goes back up it (v2-stay-rl2, 11-12: six times, and the weights were gone in 36 steps)
        assert a.lr_jump or c['row'] < a.warmup or bool(a.diloco_hub) or first <= 2*was, (
            f'--resume: the next row would step at {first:.2e}, {first/was:.1f} times the {was:.2e} the run ended on: '
            f'--lr {a.lr*was/first:.2e} keeps it there (--lr-jump: as given)')
    elif a.init and a.seq:                                        # (aio.seqnet: the parts the spec adds and the weights lack, m g p f e, start where they change nothing)
        from aio import seqnet as SQ
        src_act = dict(ck['model']['actor'])
        own_act = net.actor.state_dict()
        # Seamless g1 -> g2 lag expansion: copy trained g1 lags (1,3,8,24,64) and log-linearly interpolate new lags (16,36,48)
        lag_grown = False
        rel_grown = False
        dropped_edge = False
        for k, v in list(src_act.items()):
            if k.endswith('.lag_b') and k in own_act and v.shape != own_act[k].shape and v.shape[0] == own_act[k].shape[0]:
                if v.shape[1] == len(SQ.LAGS_BY[1]) and own_act[k].shape[1] == len(SQ.LAGS_BY[2]):
                    src_act[k] = expand_lag_b(v, SQ.LAGS_BY[1], SQ.LAGS_BY[2])
                    lag_grown = True
            elif net.actor.rel >= 2 and k in own_act and v.dim() == 2 and v.shape[0] == own_act[k].shape[0] and v.shape[1] == SQ.REL_F and own_act[k].shape[1] == SQ.REL_F2:
                src_act[k] = expand_rel_in(v, SQ.REL_F2)
                rel_grown = True
            elif net.actor.rel > 0 and net.actor.edge == 0 and (k.startswith('edge_') or '.edge_bias.' in k):
                src_act.pop(k)
                dropped_edge = True
        got = net.actor.load_state_dict(src_act, strict=False)
        grew = {k.split('.')[2] if k.startswith('layers.') else k.split('.')[0] for k in got.missing_keys}
        if lag_grown:
            grew.add('lag_b')
        if rel_grown:
            grew.add('rel_f2')
        assert not got.unexpected_keys and grew <= SQ_GROWN, f'--init on another network: {sorted(grew-SQ_GROWN)} {got.unexpected_keys[:4]}'
        mup_scaled = False
        src_mup = bool(SQ.spec(SQ.of_buffer(ck['model']['pairs']['seq'])[0])['mup']) if 'pairs' in ck['model'] and 'seq' in ck['model']['pairs'] else False
        if net.actor.mup and not src_mup:                         # u0 -> u1: exact forward identity with O(1) readout weights and balanced gradients
            with torch.no_grad():
                s_mup = math.sqrt(net.actor.d)
                net.actor.act[-1].weight.mul_(s_mup)
                net.actor.act[-1].bias.mul_(s_mup)
            mup_scaled = True
        gauge = float(getattr(a, 'init_gauge', 1.))
        assert 0. < gauge <= 1., f'--init-gauge must be in (0, 1], got {gauge}'
        if gauge < 1.:                                            # gauge-invariant residual stream deflation (Pre-LN + final LayerNorm)
            with torch.no_grad():
                for m in (net.actor.row, net.actor.mv):
                    m.weight.mul_(gauge); m.bias.mul_(gauge)
                for lay in net.actor.layers:
                    for m in (lay.out_t, lay.out_c, lay.ff[-1]) + ((lay.out_l,) if lay.lags else ()):
                        m.weight.mul_(gauge); m.bias.mul_(gauge)
        if set(critic.state_dict()) == set(ck['model'].get('critic', ())):
            critic.load_state_dict(ck['model']['critic'])
        print(json.dumps(dict(init=str(a.init), grown=sorted(grew), mup_scaled=mup_scaled, gauge=gauge)), flush=True)
    elif a.init:
        net.actor.load_state_dict(ck['model']['actor'])
        if 'pairs' in ck['model']:                                # a checkpoint of this trainer: the pair pathway and the critic too
            net.pairs.load_state_dict(ck['model']['pairs'])       # (one from before the running sums: their layer stays at its zero start)
            if {k for k in critic.state_dict() if not N.has_stats([k])} <= set(ck['model'].get('critic', ())) <= set(critic.state_dict()):
                critic.load_state_dict(ck['model']['critic'])
    if getattr(a, 'init_pairs', None) and not resumed:
        got, own = safe_torch_load(a.init_pairs, map_location='cpu')['pairs'], net.pairs.state_dict()
        same = {k: v for k, v in got.items() if k in own and own[k].shape == v.shape}
        assert any(k.startswith('reader.') for k in same), '--init-pairs: none of the learned reader\'s weights fit (another size asked for with --reader?)'
        net.pairs.load_state_dict(same, strict=False)
        print(json.dumps(dict(init_pairs=str(a.init_pairs), taken=len(same), of_file=len(got), of_net=len(own))), flush=True)
    if a.init and not resumed and not a.bc_floor:                 # a resumed run goes on along its own schedule: its rows are counted on
        a.bc_iters = 0                                            # (--bc-floor: the teacher stays at it beside the simulator gradient)
    ema = None
    if a.ema:                                                     # (resumed from a run without one: the average starts at its weights)
        kept = resumed and 'raw' in ck
        ema_dev = torch.device('cpu') if a.ema_device == 'cpu' or dev.type == 'cpu' else dev
        ema = Average(net, a.ema, ck.get('ema_n', 0) if kept else 0, device=ema_dev)
        if kept:
            ema.avg.actor.load_state_dict(ck['model']['actor']); ema.avg.pairs.load_state_dict(ck['model']['pairs'])
        opt.register_step_post_hook(ema)
    del ck                                                        # release CPU checkpoint memory before building batches
    shown = ema.avg if ema else net                               # what is validated and kept as `model`
    if (getattr(a, 'compile', False) or os.environ.get('AIO_COMPILE') == '1') and dev.type == 'cuda':
        try:
            import triton  # noqa: F401
            if hasattr(net.actor, 'enable_compile'):
                net.actor.enable_compile(dynamic=True)
                if ema is not None and ema.device.type == 'cuda':
                    ema.avg.actor.enable_compile(dynamic=True)
            if getattr(critic, 'privileged', False):
                torch._dynamo.config.cache_size_limit = max(getattr(torch._dynamo.config, 'cache_size_limit', 8), 64)
                critic.forward = torch.compile(critic.forward, dynamic=False)
            if getattr(a, 'mech', 0.) == 0. and hasattr(C, 'enable_sim_compile'):
                C.enable_sim_compile(True, dynamic=False)
                if hasattr(N, 'enable_priv_compile'):
                    N.enable_priv_compile(True, dynamic=False)
            print(json.dumps(dict(compiled=True, triton=getattr(triton, '__version__', 'ok'), sim_compiled=bool(getattr(C, 'is_sim_compiled', lambda: False)()))), flush=True)
        except Exception as exc:
            print(json.dumps(dict(compiled=False, reason=repr(exc))), flush=True)
    quiet_paused_s = [0.0]

    def maybe_quiet_pause(forget_models=True, count_pause=True):
        if not getattr(a, 'quiet_hours', ''):
            return
        wait_s = quiet_sleep_seconds(a.quiet_hours, getattr(a, 'quiet_tz', 'Asia/Shanghai'))
        if wait_s <= 0:
            return
        if forget_models:
            forget(net, shown)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print(dumps(dict(quiet_pause=a.quiet_hours, tz=a.quiet_tz, resume_in_s=round(wait_s, 1))), flush=True)
        q_start = time.time()
        while True:
            rem = quiet_sleep_seconds(a.quiet_hours, getattr(a, 'quiet_tz', 'Asia/Shanghai'))
            if rem <= 0:
                break
            alive(row=c['row'], windows=c['windows'], quiet=True)
            time.sleep(min(60.0, rem))
        slept = time.time()-q_start
        if count_pause:
            quiet_paused_s[0] += slept
        print(dumps(dict(quiet_resume=a.quiet_hours, slept_s=round(slept, 1))), flush=True)

    maybe_quiet_pause(forget_models=False, count_pause=False)
    if a.eval_every > 0:
        vprog = C.build(a.val_n, a.val_seed, dev, 'cpu', mech=a.val_mech)     # always drawn on the CPU: the same episodes on every machine
        ref = cached_refs(a.val_cache, f'val-n{a.val_n}-s{a.val_seed}-m{a.val_mech}-mt{int(bool(a.meter))}', vprog, a.meter)
        scale, refs_used = scale_of(ref, a.refs)
        if resumed and temp.pop('zero') != SC.ZERO:                   # a checkpoint of the scale whose 0 was the formula PI: its best, read on this one
            c['best'] = scale.from_recovered(c['best'])
        refs = ref_table(vprog, ref, scale)                           # (the references on the plants with a level and on the others, as `validate` reads the network)
        parked(vprog)                                                 # the validation batches wait on the CPU (`visit`: on the card for a reading)
        vsizes = size_vals(a.val_sizes, a.val_seed, dev, a.meter, a.val_cache)     # --val-sizes: a batch of each of the larger sizes
        stay = Stay(a.val_stay_n, a.val_seed+1, a.val_stay, dev, nseg=a.val_nseg) if a.val_stay > 1 else None
        exams = exams_of(a.val_exams, a.val_stay_n, a.val_stay if a.val_stay > 1 else ST.EXAM_WINDOWS, dev, a.val_exams_refs, a.val_nseg) if a.val_exams else {}
        known = (not a.no_known_val) and not a.seq                    # SeqAIO ignores `code`: skip duplicate identical 256-plant rollout
        if c['row'] == 0:                                             # where the run starts from, on the same batch
            start = dict(val={q: round(x, 4) for q, x in validate(net, vprog, ref, scale, known).items()})
            if stay:
                start['stay'] = stay(net)
            if exams:
                start['exams'] = read_exams(exams, net)
            if vsizes:
                start['val_sizes'] = read_sizes(vsizes, net)
        else:                                                         # a resumed run keeps the record of its start
            old = a.out/'run.json'
            start = json.loads(old.read_text()).get('start') if old.exists() else None
            with on_device(shown, dev):
                if stay and not (start or {}).get('stay'):               # (a run that had no --val-stay: where it stands when it gets one)
                    start = dict(start or {}, stay=dict(stay(shown), iter=c['row']))
                if exams and not (start or {}).get('exams'):
                    start = dict(start or {}, exams=read_exams(exams, shown), exams_iter=c['row'])
                if vsizes and not (start or {}).get('val_sizes'):
                    start = dict(start or {}, val_sizes=read_sizes(vsizes, shown), val_sizes_iter=c['row'])
    else:                                                             # --eval-every 0: skip all validation setup & rollouts (0 s overhead)
        temp.pop('zero', None)
        vprog = ref = scale = refs_used = refs = stay = start = None
        vsizes, exams, known = {}, {}, False
    # --size-stop: the points on each larger size the simulator gradient is held to, the last reading with the teacher alone
    if size_ref is None and (start or {}).get('val_sizes'):
        size_ref = {q: v['pts'] for q, v in start['val_sizes'].items()}
    meta = sanitize_paths(dict({k: v for k, v in vars(a).items() if not k.startswith('_')},
                               out=str(a.out), init=str(a.init) if a.init else None, refs=str(a.refs) if a.refs else None, refs_used=refs_used,
                               trainer='aio.train', version=2, canonical=True, horizon=HORIZON, val_refs=refs, scale=scale.meta() if scale else None, start=start,
                               stay_refs=stay.meta() if stay else None, exam_refs={q: e.meta() for q, e in exams.items()} or None,
                               val_size_refs={q: ref_table(*v) for q, v in vsizes.items()} or None))
    (a.out/'run.json').write_text(dumps(meta, indent=1, default=str))
    print(dumps(meta, default=str), flush=True)
    t0 = time.time()
    hours = lambda: c['hours']+(time.time()-t0-quiet_paused_s[0])/3600              # noqa: E731
    diloco = None
    if a.diloco_hub:
        diloco = DiLoCoClient(a.diloco_hub, a.diloco_h, worker=a.diloco_worker or a.out.name,
                              compress=a.diloco_compress, timeout=a.diloco_timeout,
                              round_idx=diloco_ck.get('round', 0), hub_lr=diloco_ck.get('hub_lr'))
        diloco.snapshot_anchor(net)

    def save(name, val=None):
        more = dict(raw=dict(actor=net.actor.state_dict(), pairs=net.pairs.state_dict()), ema_n=ema.n) if ema else {}
        if rel_probe is not None:
            more.update(rel_probe=rel_probe.state_dict(), rel_popt=rel_popt.state_dict())
        if diloco is not None:
            more['diloco'] = diloco.state_meta()
        atomic_save(dict(model=dict(actor=shown.actor.state_dict(), pairs=shown.pairs.state_dict(), critic=critic.state_dict()), **more, opt=opt.state_dict(),
                         copt=copt.state_dict(), usual=temp['usual'], gns=temp.get('gns'), guard_units=GUARD_UNITS, zero=SC.ZERO,
                         config=sanitize_paths(dict(meta, canonical=True)), iter=c['row'], windows=c['windows'],
                         cosine_by=a.cosine_by, total_updates=a.total_updates, total_samples=a.total_samples,
                         usuals={q: dict(usual=x['usual'], gns=x.get('gns')) for q, x in temps.items() if q != DEFAULT}, size_ref=size_ref,
                         samples=c['samples'], updates=c['updates'], drawn=c['drawn'], stayed=c['stayed'], designs=c['designs'], sized=c['sized'], plants_by=c['plants_by'],
                         design_plants=design_plants(), best=c['best'], hours=hours(), val=val, turns=c['turns']), a.out/f'{name}.pt')

    def save_eval_snap(row_idx):
        eval_dir = a.out/'eval'
        snap = dict(
            model=dict(actor=pack_eval_state(shown.actor.state_dict()),
                       pairs={k: v.detach().cpu() for k, v in shown.pairs.state_dict().items()}),
            config=sanitize_paths(dict(meta, canonical=True)),
            iter=row_idx, hours=hours(), updates=c['updates'], samples=c['samples'], eval_snap=True,
        )
        if a.eval_snap_raw and ema is not None:
            snap['raw'] = dict(actor=pack_eval_state(net.actor.state_dict()),
                               pairs={k: v.detach().cpu() for k, v in net.pairs.state_dict().items()})
        atomic_save(snap, eval_dir/f'row{row_idx}.pt')
        prune_saves(eval_dir, a.keep_eval_snaps)

    def seed():                                                   # of the next batch of plants; fixed sets (validation) use seeds below 2^32
        c['drawn'] += 1
        return (a.seed << 32)+c['drawn']-1

    def design_seed():                                            # of the next set of problems of design: apart from the batches', which are the same plants with and without them
        c['designs'] += 1
        return (a.seed << 32)+(1 << 31)+c['designs']-1

    short = lambda i: bool(a.life and a.replay and i)             # noqa: E731  a lane whose plants are replaced after --life decisions
    ahead = Ahead(a.builders, a.envs, seed, a.builder_threads, a.builder_qos) if a.builders else None
    designs, bufs, total = {}, {}, sum(x[0] for x in shares or ())

    def designs_of(lane):                                         # --design: the pool of problems of the lane's size, made when it is first needed
        if not a.design:
            return None
        if lane.size not in designs:
            q, share = lane.size, next((x[0] for x in shares or () if x[1] == lane.size), total)/max(total, 1e-9)
            designs[q] = Designs(a, ahead.pool if ahead else None, design_seed, dev, net.pairs.nk, None if q == DEFAULT else dict(dims=q), every=math.ceil(HORIZON/windows[0]),
                                 b=per_pairs(a.design, q), n=per_pairs(a.design_set, q), keep=per_pairs(a.design_keep, q), ahead=max(1, round((a.builders-1)*share)))
        return designs[lane.size]

    def design_plants():
        return c['design_plants']+sum(d.plants for d in designs.values())

    def buffer_of(lane):                                          # --replay: the windows of the lane's size (one size a buffer: the windows of a step are stacked)
        if lane.size not in bufs:
            bufs[lane.size] = Buffer(a.keep*lane.prog.n, max(getattr(a, 'batch_min', 1), per_pairs(a.batch, lane.size)))   # (--batch-min: 48x20 had 4 windows a step)
        return bufs[lane.size]

    def temp_of(lane):
        return temps.setdefault(lane.size, dict(usual=None, acc=acc, rel_probe=rel_probe, rel_popt=rel_popt))

    def settle():                                                 # --sizes: the windows kept and the problems of design of a size no lane is of now wait on the CPU
        on = {x.size for x in lanes if x is not None}             # (2 GB and .4 GB a size; a size comes and goes with a lane's batch, an episode or --stay of them at a time)
        for q, b in list(bufs.items())+[(q, d.buf) for q, d in designs.items()]:
            b.to(dev if q in on else 'cpu')

    menu = menu_kw(a.menu, a.nseg, True)                          # what changes on the training plants (the validation batches keep the generator's own)
    menu_stay = menu_kw(a.menu_stay, a.nseg, True) if a.menu_stay is not None else menu      # on the lanes that keep their plants
    menu_plain = menu_kw(a.menu_plain, a.nseg)                    # on every --plain-th lane
    mix = mix_of(a.menu_mix)                                      # groups of fields a new batch takes by chance, over --menu (--menu-stay on the lanes that keep their plants)
    p_mix = 1.-math.prod(1.-p for p, _ in mix) if mix else 0.
    hard_seeds = HardSeeds(a.hard_seeds, a.hard_share)            # --hard-seeds / --hard-share: mined hard batch seeds paired with fresh event_seeds
    groups = groups_of(a.mech_groups)                             # kinds of mechanistic plants a batch that has any takes together
    stays = groups_of(a.mech_groups_stay) if a.mech_groups_stay is not None else None   # on the lanes that keep their plants, where they are others
    go_on = lambda kw: {q: v for q, v in kw.items() if q in GO_ON}    # noqa: E731  (the plants are there: only what changes on them is asked again)

    def then(lane):                                               # the next episode of a lane's units: a builder starts on it now, or it is drawn when it is needed
        if lane.left > 0 and ahead:
            c['stayed'] += 1
            lane.next = ahead.pool.submit(C.build, lane.prog.n, seed(), 'cpu', 'cpu', after=lane.prog.tail, **go_on(lane.kw))
        return lane

    def fresh(i=0):
        t, mech = time.time(), 0. if short(i) else a.mech
        plain, keeps = bool(a.plain and i % a.plain == a.plain-1), a.stay > 1 and i % 2 == 0      # --stay: the even lanes keep their plants
        kw, on, depth = menu_plain if plain else menu_stay if keeps else menu, None, a.builders if short(i) else a.ahead
        if (mix and not plain) or (groups and mech) or shares or (hard_seeds.active and not plain):    # --menu-mix, --mech-groups, --sizes, --hard-seeds
            base, same = a.menu_stay if keeps and a.menu_stay is not None else a.menu, kw
            kind = 'plain' if plain else 'stay' if keeps else 'new'
            gs, chance, first = (stays, a.mech_batches if a.mech_batches_stay is None else a.mech_batches_stay, len(groups)) if keeps and stays is not None \
                else (groups, a.mech_batches, 0)                  # (the stay lanes' groups are counted on from the others' in the rows' `mech`)

            def kw_of(sd):
                k = c['turns'].get(kind, 0)                       # the size and the mechanistic plants by an even sequence per kind of lane (`evenly`)
                c['turns'][kind] = k+1
                u, v = (evenly(k, d, f'{a.seed} {kind}') for d in (0, 1))
                size, n = size_of(shares, sd, u) if shares else (DEFAULT, a.envs)
                usual = size == DEFAULT                           # a larger size: the generator's own changes, with room for them (--nseg)
                items, on = mixed(base, mix, sd) if mix and not plain and usual else (None, None)
                m, names, g = mech_of(mech, chance, gs, sd, v)
                g = None if g is None else g+first
                kw = (same if items is None else menu_kw(items, a.nseg, True)) if usual else menu_kw((), a.nseg, True)
                if shares:
                    c['sized'] += 1
                    c['plants_by'][size] = c['plants_by'].get(size, 0)+n
                    kw = dict(kw, envs=n, **({} if usual else dict(dims=size)))
                ps = hard_seeds.pick(sd, plain=plain, is_default_size=usual, menu_active=bool(on and any(on)), p_menu=p_mix)
                extra = dict(plant_seed=ps, event_seed=sd) if ps is not None else {}
                return dict(kw, mech=m, names=names, **extra), (on, g)
            if ahead:
                prog, kw, on = ahead.own(('own', keeps, plain, mech), kw_of, depth)
                prog = moved(prog, dev)
            else:
                sd = seed()
                kw, on = kw_of(sd)
                prog = drawn(a.envs, sd, dev, a.gen_device, **kw)
        else:
            prog, on = moved(ahead(mech, depth, **kw), dev) if ahead else C.build(a.envs, seed(), dev, a.gen_device, mech=mech, **kw), (None, None)
        c['t_build'] = c.get('t_build', 0.)+time.time()-t        # the run's own time on new plants: drawing them, or waiting for and moving them
        lane = Lane(prog, a, gen, weights(c['row'], a)['bc'], a.drive)
        lane.left, lane.kw, (lane.mix, lane.group) = (a.stay-1 if i % 2 == 0 else 0), kw, on
        return then(lane)

    def follow(lane):                                             # --stay: the lane goes on to the next episode of its units
        t = time.time()
        if lane.next is None:
            c['stayed'] += 1
        prog = moved(waited(lane.next), dev) if lane.next is not None else C.build(lane.prog.n, seed(), dev, a.gen_device, after=lane.prog.tail, **go_on(lane.kw))
        c['t_build'] = c.get('t_build', 0.)+time.time()-t
        lane.onto(prog)
        return then(lane)

    lanes = []
    for i in range(a.lanes):                                      # lane i starts i/lanes of the way into its episodes (a short one: of its --life)
        lanes.append(fresh(i))
        skip = (i-1)*a.life//max(a.lanes-1, 1) if short(i) else i*HORIZON//a.lanes
        advance(net, critic, lanes[i], skip//a.window*a.window, gen, a.noise, a.drive_noise)
    if ahead:
        print(dumps(dict(builders=ahead.pids(), first_batches_s=round(c['t_build'], 1))), flush=True)
    log = open(a.out/'metrics.jsonl', 'a')
    done = dict(score=None, failed=None)
    stopped = None                                                # --size-stop: why the run stopped early
    alloc = retries(dev)                                          # (rows: alloc, the allocator's counts of the row)
    last_eval_snap_row = -1

    def keep_going():
        if c['row'] >= a.rows or stopped is not None:
            return False
        if a.cosine_by == 'hours':
            return hours() < a.hours
        if (time.time()-t0-quiet_paused_s[0])/3600 >= a.hours:    # wall-clock active session cap decoupled from LR schedule & quiet sleep
            return False
        if a.cosine_by == 'updates' and a.total_updates > 0 and c['updates'] >= a.total_updates:
            return False
        if a.cosine_by == 'samples' and a.total_samples > 0 and c['samples'] >= a.total_samples:
            return False
        return True

    while keep_going():
        maybe_quiet_pause(forget_models=True, count_pause=True)
        lr = scheduled_lr(a, c, hours(), hub_lr=None if diloco is None else diloco.hub_lr)
        for g in opt.param_groups:
            g['lr'] = g['set_lr'] = lr*g.get('lr_mult', 1.)
        if hasattr(opt, 'warm_n'):                                # --warm-steps: counted on from the run's steps
            opt.warm_n = max(opt.warm_n, c['updates'])
        w = weights(c['row'], a)
        a.window = windows[w['sim'] > 0]
        t, logs, c['t_build'], t_sizes = time.time(), [], 0., {}
        row_updates = row_samples = 0
        for _ in range(a.row or math.ceil(HORIZON/a.window)):
            i = c['windows'] % a.lanes
            alive(row=c['row'], windows=c['windows'])
            if lanes[i].done or short(i) and w['sim'] == 0 and lanes[i].k >= a.life:
                own = torch.ones_like(lanes[i].score, dtype=torch.bool) if lanes[i].drive is None else ~lanes[i].drive
                if lanes[i].lead is not None:                     # (--lead: not those the teacher had first)
                    own = own & (lanes[i].lead == 0)
                if lanes[i].done and bool(own.any()):             # the student's own episodes, run whole
                    done = dict(score=float(lanes[i].score[own].mean()), failed=float(lanes[i].st['fail'][own].float().mean()))
                if lanes[i].done and lanes[i].left > 0:
                    follow(lanes[i])
                else:
                    if lanes[i].due is not None:                  # --fixed: the critic on its last windows
                        fit_due(critic, copt, lanes[i], a, last=True)
                    lanes[i] = None                               # the old batch is released before the new one is built
                    lanes[i] = fresh(i)
            if shares:
                if dev.type == 'cuda' and lanes[i].size != c.get('size'):
                    maybe_empty_cache(dev, .85, expandable)
                c['size'] = lanes[i].size
                settle()
            t1 = time.time()
            if a.replay and w['sim'] == 0:                        # the teacher alone: gathered apart from learning
                logs.append(imitate(net, critic, opt, copt, lanes[i], buffer_of(lanes[i]), a, gen, designs_of(lanes[i])))
            elif a.ppo:
                logs.append(ppo(net, critic, opt, copt, lanes[i], a, w, gen))
            else:
                logs.append(update(net, critic, opt, copt, lanes[i], a, w, gen, temp_of(lanes[i])))
            t_sizes[lanes[i].size] = t_sizes.get(lanes[i].size, 0.)+time.time()-t1
            logs[-1].update(age=float(lanes[i].age), size=lanes[i].size)
            if lanes[i].age:                                      # the readout on units run for an episode or more (--stay), beside the row's over all
                logs[-1].update(id_own_stay=logs[-1]['id_own'], id_cross_stay=logs[-1]['id_cross'])
            u_inc = logs[-1].get('updates', 1)
            s_inc = logs[-1]['steps']
            c['windows'] += 1; c['samples'] += s_inc; c['updates'] += u_inc
            row_updates += u_inc; row_samples += s_inc
        c['row'] += 1
        avg = lambda q: (lambda v: sum(v)/len(v) if v else None)([x[q] for x in logs if x.get(q) is not None])   # noqa: E731
        eff_lr = lr*min(1., max(c['updates'], 1)/a.warm_steps) if getattr(a, 'warm_steps', 0) > 0 else lr
        row = dict(iter=c['row'], lr=round(eff_lr, 6), w_sim=round(w['sim'], 2), w_bc=round(w['bc'], 2), sim=rnd(avg('sim'), 4), cost=rnd(avg('cost'), 4),
                   window=a.window, bc=rnd(avg('bc'), 5), fit=rnd(avg('fit'), 5), pre=rnd(avg('pre'), 4), **{q: rnd(avg(q), 4) for q in LAW}, updates=c['updates'], id=rnd(avg('id'), 4), id_own=rnd(avg('id_own'), 4), id_cross=rnd(avg('id_cross'), 4), **{q: rnd(avg(q), 4) for q in ('id_own_told', 'id_own_blind')}, std=a.noise,
                   gn=rnd(avg('gn'), 3), skip=sum(x['skip'] for x in logs), far=rnd(max([x['far'] for x in logs if x.get('far') is not None], default=None), 1), capped=rnd(avg('capped'), 4),
                   short=rnd(avg('short'), 4), peak=rnd(max([x['peak'] for x in logs if x['peak'] is not None], default=None), 1), usual=rnd(temp['usual'], 5), hor=rnd(avg('hor'), 1), critic=rnd(avg('critic'), 4), ev=rnd(avg('ev'), 3),
                   value=rnd(avg('value'), 3), student_score=rnd(done['score'], 1), failed=rnd(done['failed'], 4), t_row=round(time.time()-t, 1), t_build=round(c['t_build'], 1),
                   samples=c['samples'], plants=(c['drawn']-c['stayed']-c['sized'])*a.envs+sum(c['plants_by'].values()), hours=round(hours(), 3))
        row.update({q: rnd(avg(q), 4) for q in GAUGES}, bad=sum(x.get('bad', 0.) for x in logs))   # gauges of the row (`parts`, `window`)
        if rel_probe is not None:
            row.update(rel_id=rnd(avg('rel_id'), 4), rel_live=rnd(avg('rel_live'), 4), rel_sign=rnd(avg('rel_sign'), 4))
        if a.ppo:                                                 # --ppo: how far a window's passes moved the policy, the share of decisions past the clip
            row.update(kl=rnd(avg('kl'), 5), clipped=rnd(avg('clipped'), 4))
        if net.pairs.unseen is not None:                          # --unseen: its read-out's loss and how much of the levels' rate it reads
            row.update({q: rnd(avg(q), 3) for q in UNSEEN})
        if a.design:                                              # --design: the K read-out on the problems of the pool, on the sets that came in this row, and the plants of the sets taken so far
            new = {q: d.log()['design_new'] for q, d in designs.items()}
            row.update(design=rnd(avg('design'), 4), design_new=rnd(new.get(DEFAULT), 4), designs=design_plants())
            if shares:
                row['design_by'] = {q: rnd(x, 4) for q, x in new.items()}
        if shares:                                                # --sizes: per lane the size of its plants, the row's time per size, the plants drawn per size
            row.update(sizes=[x.size for x in lanes], t_sizes={q: round(x, 1) for q, x in sorted(t_sizes.items())}, plants_by=dict(sorted(c['plants_by'].items())))
            of = lambda q, z: (lambda v: rnd(sum(v)/len(v), 4) if v else None)([x[q] for x in logs if x['size'] == z and x.get(q) is not None])   # noqa: E731
            row['by_size'] = {z: {q: of(q, z) for q in BY_SIZE if of(q, z) is not None} for z in sorted(t_sizes)}      # the row's windows on each size apart: a larger size has
                                                                  # tracked rows without a valve of their own, all of them in the 'cross' columns, so these are not read across sizes
            if len(temps) > 1:
                row['usual_by'] = {q: rnd(x['usual'], 5) for q, x in sorted(temps.items())}
        if a.stay > 1:
            row.update(age=rnd(avg('age'), 2), id_own_stay=rnd(avg('id_own_stay'), 4), id_cross_stay=rnd(avg('id_cross_stay'), 4))
        if mix:                                                   # --menu-mix: per lane, which groups the plants it is on took (None: a --plain lane)
            row['mix'] = [None if x.mix is None else list(x.mix) for x in lanes]
        if groups:                                                # --mech-groups: per lane, the group its plants' mechanistic episodes are of (None: it has none)
            row['mech'] = [x.group for x in lanes]
        if diloco is not None:                                    # Elastic MuLoCo sync at exact row boundary
            diloco.record(row_updates, row_samples)
            d_res = diloco.maybe_sync(net, critic, rel_probe)
            if d_res is not None:
                row['diloco'] = d_res
        alive(row=c['row'], windows=c['windows'])
        if a.eval_every > 0 and c['row'] % a.eval_every == 0:
            tv = row['t_val'] = {}                                # the seconds of each reading of this validation
            forget(net, shown)                                    # the training batches' kept memories are not held through the validation
            with on_device(shown, dev):
                v = timed(tv, 'val', validate, shown, vprog, ref, scale, known)
                row['val'] = {q: round(x, 4) for q, x in v.items()}
                if stay and (c['row']//a.eval_every) % a.val_stay_every == 0:
                    row['stay'] = timed(tv, 'stay', stay, shown)
                if exams and (c['row']//a.eval_every) % a.val_stay_every == 0:
                    row['exams'] = timed(tv, 'exams', read_exams, exams, shown)
                if ema and a.val_raw:
                    row['raw'] = {q: round(x, 4) for q, x in timed(tv, 'raw', validate, net, vprog, ref, scale, False).items()}
                if vsizes:                                            # the larger sizes, and the line the simulator gradient is held to on them
                    row['val_sizes'] = timed(tv, 'sizes', read_sizes, vsizes, shown)
                    now = {q: x['pts'] for q, x in row['val_sizes'].items()}
                    if w['sim'] == 0 or size_ref is None:
                        size_ref = now
                    elif a.size_stop:
                        drop = {q: round(size_ref[q]-x, 2) for q, x in now.items() if q in size_ref}
                        if any(x > a.size_stop for x in drop.values()):
                            stopped = row['stopped'] = dict(iter=c['row'], drop=drop, ref=size_ref, now=now)
            if v['pts'] > c['best']:
                c['best'] = v['pts']; save('best', v)
            c['val'] = v
        if a.save_every and c['row'] % a.save_every == 0:
            save(f'row{c["row"]}', c.get('val'))
            prune_saves(a.out, a.keep_saves)
        if a.eval_snap_every and c['row'] % a.eval_snap_every == 0:
            save_eval_snap(c['row'])
            last_eval_snap_row = c['row']
        save('last', c.get('val'))                                # every row: where a run that is stopped goes on from (`val`: its last validation)
        if dev.type == 'cuda':                                    # GB: the most in use at once in this row (its validation included), and what
            row['gb'] = [round(x/2**30, 2) for x in (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())]   # the allocator held (what the card shows);
            row['alloc'] = [x for x in retries(dev, alloc).values()]; alloc = retries(dev)         # retries, blocks asked for, given back (`retries`)
            maybe_empty_cache(dev, .85, expandable); torch.cuda.reset_peak_memory_stats()
        line = dumps(row); print(line, flush=True); log.write(line+'\n'); log.flush()
    log.close()
    if a.eval_every > 0:
        with on_device(shown, dev):
            v = validate(shown, vprog, ref, scale, known)
            if v['pts'] > c['best']:
                c['best'] = v['pts']; save('best', v)
            save('last', v)
            if a.save_every:                                              # the row a run ends on stays when the run is carried on in its directory
                save(f'row{c["row"]}', v)
                prune_saves(a.out, a.keep_saves)
            if a.eval_snap_every and c['row'] != last_eval_snap_row:
                save_eval_snap(c['row'])
            final = dict(iter=c['row'], updates=c['updates'], samples=c['samples'], hours=round(hours(), 3), val=v, val_refs=refs, scale=scale.meta(), best=c['best'], stopped=stopped)
            if vsizes:
                final.update(val_sizes=read_sizes(vsizes, shown), val_size_refs=meta['val_size_refs'])
            if stay:
                final.update(stay=stay(shown), stay_refs=stay.meta())
            if exams:
                final.update(exams=read_exams(exams, shown), exam_refs={q: e.meta() for q, e in exams.items()})
    else:
        save('last', c.get('val'))
        if a.save_every:
            save(f'row{c["row"]}', c.get('val'))
            prune_saves(a.out, a.keep_saves)
        if a.eval_snap_every and c['row'] != last_eval_snap_row:
            save_eval_snap(c['row'])
        final = dict(iter=c['row'], updates=c['updates'], samples=c['samples'], hours=round(hours(), 3), val=c.get('val'), val_refs=None, scale=None, best=c['best'], stopped=stopped)
    (a.out/'final.json').write_text(dumps(final, indent=1))
    print(dumps(dict(final=final)), flush=True)
    faulthandler.cancel_dump_traceback_later()
    alive(done=True)
    if ahead:
        ahead.close()


if __name__ == '__main__':
    main()

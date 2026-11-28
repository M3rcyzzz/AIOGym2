"""Where a trained network's score comes from, and whether following the trainer's objective lowers it.

A diagnosis (11-11): the first from-scratch run of aio.train stopped at 1.15 of the formula PI's score with
0.57 of its quadratic cost: the deviations were made smaller, the time outside the bands was not.

  where   on the validation batch, every reference and the checkpoint against the formula PI: the score; the
          parts of the objective (softened count, Huber, log) and how much of it the count is; the
          row-decisions outside within a band of the edge and beyond; the share of the rows outside before
          and after the unit can have lined out, and by decisions since the event.
  fit     a fixed batch of fresh episodes and the trainer's own update (aio.train.update) from the
          checkpoint: --critic-passes over the episodes fitting the critic alone (to the objective given),
          then --passes with the actor. After every pass the deterministic score on those episodes, at the
          start and the end on the validation batch too. Run once with the objective the checkpoint was
          trained on and once with another (--cost-w, --log-w): does the score on the very episodes that are
          trained on fall with one and not with the other.

  quiet   the stretches a unit has had time to line out in (--late decisions after an event): the rows outside
          and how far, whether an error keeps its sign (an offset) or turns (an oscillation), how much the MVs
          move, how often a move turns back on the last one (chatter), the MVs at a limit; of a tracked row's
          error over these stretches the mean (`offset`) and the spread (`swing`: a slow oscillation keeps its
          sign from one decision to the next too), medians over the rows, bands; for the MVs whose
          own row is outside `push`, their mean move counted the way the formula PI would move them, the
          share at a limit, and `closing`, the bands a decision an outside row comes nearer (is an offset
          worked off). With the network once more with its moves low-pass filtered (does the chatter cost
          score), and the score against the formula PI by kind of plant.

  horizon is what the trainer tells a move what the cost further out tells it? On a fixed batch at several
          points of the episodes, per episode the gradient at the moves of the trainer's own window (with
          the critic's value at its end, and without) against that of the cost actually paid over --long
          decisions: the ones after the window by a copy of the network whose weights are held (so the
          plant's and the policy's answer to the window's moves is in, as a value would have it). The
          cosines, how much the paid gradient grows with the stretch (a closed loop that is not stable
          grows without bound), and from which stretch on its direction is the long one's.

  python -m aio.probe where RUN/last.pt
  python -m aio.probe quiet RUN/last.pt
  python -m aio.probe horizon RUN/last.pt
  python -m aio.probe fit RUN/last.pt --cost-w .1 --log-w 0
  python -m aio.probe fit RUN/last.pt --cost-w 0 --log-w .2

Reads the checkpoint; trains a copy in memory; writes only --out (a json file).
"""
import argparse, copy, json, os, pathlib, time
from types import SimpleNamespace
import torch

from aio import net as N, train as T
from aio.sim import core as C, control as CT, oracle as OR
from aio.sim.prior import ARCH, HORIZON

SINCE = 240                                                       # decisions after an event that the profile follows
AT = (0, 5, 10, 20, 40, 80, 120, 160, 200, 239)


def load(path, dev):
    """(actor network, critic, checkpoint dict) of a checkpoint of aio.train."""
    if os.name != 'nt':                                           # written on the training machine: its paths are Windows paths
        pathlib.WindowsPath = pathlib.PureWindowsPath
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = ck['model']
    stats = N.has_stats(m['pairs'])
    tied = N.tied_of(m['pairs'])
    net, critic = N.AIO(stats, tied).to(dev), N.Critic(ck['config'].get('heads', 2), stats, tied).to(dev)
    net.actor.load_state_dict(m['actor']); net.pairs.load_state_dict(m['pairs']); critic.load_state_dict(m['critic'])
    return net, critic, ck


@torch.no_grad()
def measure(prog, ctl, tau=.25):
    """A controller on every episode -> per-episode sums from the first event on (dict of (n,)) and the share
    of the rows outside by decisions since the event that opened the segment (SINCE,)."""
    st = C.reset(prog)
    ctl.reset(prog)
    n, c, dev = prog.n, prog.c, prog.device
    rows = (c['trk']+c['zon']).sum(-1).clamp_min(1.)
    q = {k: torch.zeros(n, dtype=torch.float64, device=dev) for k in
         ('score', 'count', 'huber', 'log', 'quad', 'near', 'far', 'early', 'late', 'early_rows', 'late_rows')}
    prof, profn = torch.zeros(SINCE, dtype=torch.float64, device=dev), torch.zeros(SINCE, dtype=torch.float64, device=dev)
    for k in range(HORIZON):
        a = ctl(prog, st, k, C.observe(prog, st, k))
        st, out = C.step(prog, st, a, k)
        on = (k >= prog.k_first).double()
        ex = torch.maximum(out['alo']-1-out['err'], out['err']-out['ahi']-1)
        zx = torch.maximum(out['zlo']-st['yt'], st['yt']-out['zhi'])/c['bandf']-out['ahi']
        o = out['outside']
        beyond = torch.where(o & (c['trk'] > 0), ex, torch.where(o & (c['zon'] > 0), zx, torch.zeros_like(ex)))
        count = OR.objective(out, st, c, tau, 0.)
        seg = prog.seg_at(k)
        late = (k >= prog.settled_k[prog.ar, seg]).double()
        osum = o.sum(-1).double()
        for name, x in (('score', osum), ('count', count), ('huber', out['soft']/rows), ('log', OR.objective(out, st, c, tau, 0., 1.)-count),
                        ('quad', out['cost']), ('near', (o & (beyond < 1)).sum(-1)), ('far', (o & (beyond >= 1)).sum(-1)),
                        ('early', osum*(1-late)), ('late', osum*late), ('early_rows', rows*(1-late)), ('late_rows', rows*late)):
            q[name] += x.double()*on
        since, w = (k-prog.seg_k[prog.ar, seg]).clamp(0, SINCE-1), on > 0
        prof.index_add_(0, since[w], (osum/rows.double())[w]); profn.index_add_(0, since[w], torch.ones_like(osum)[w])
    q['failed'] = st['fail'].double()
    prog.set_teacher()
    return q, prof/profn.clamp_min(1)


def line(name, q, pi, cost_w, log_w):
    """One controller against the formula PI on the same episodes."""
    m = {k: float(v.mean()) for k, v in q.items()}
    obj = lambda r: r['count']+cost_w*r['huber']+log_w*r['log']   # noqa: E731
    o, op = obj(q), obj(pi)
    x = lambda k: round(m[k]/max(float(pi[k].mean()), 1e-9), 3)   # noqa: E731
    return dict(ctl=name, score=round(m['score'], 1), x_score=x('score'), x_objective=round(float(o.mean()/op.mean()), 3), x_count=x('count'),
                x_huber=x('huber'), x_log=x('log'), x_quad=x('quad'), count_share=round(m['count']/max(float(o.mean()), 1e-9), 3),
                near=round(m['near'], 1), far=round(m['far'], 1),
                early_rate=round(m['early']/max(m['early_rows'], 1e-9), 3), late_rate=round(m['late']/max(m['late_rows'], 1e-9), 3),
                wins_score=round(float((q['score'] < pi['score']).double().mean()), 3), wins_objective=round(float((o < op).double().mean()), 3),
                failed=int(q['failed'].sum()))


def where(a, dev):
    net, _, ck = load(a.ckpt, dev)
    cfg = ck['config']
    cost_w, log_w = cfg.get('cost_w', .1), cfg.get('log_w', 0.)
    prog = C.build(a.val_n, a.val_seed, dev, 'cpu')
    ctls = [('pi', CT.PI()), ('hold', CT.Hold()), ('dmc', CT.Teacher('fixed')), ('now', CT.Teacher('now')),
            ('net', N.Policy(net)), ('net_known', N.Policy(net, known=True))]
    res, prof = {}, {}
    for name, ctl in ctls:
        res[name], prof[name] = measure(prog, ctl)
        print(json.dumps(line(name, res[name], res['pi'], cost_w, log_w)), flush=True)
    print(json.dumps(dict(outside_share_by_decisions_since_event=AT, **{k: [round(float(p[i]), 3) for i in AT] for k, p in prof.items()})), flush=True)
    return dict(objective=dict(cost_w=cost_w, log_w=log_w), lines=[line(k, r, res['pi'], cost_w, log_w) for k, r in res.items()],
                profile={k: p.tolist() for k, p in prof.items()}, per_episode={k: {q: v.tolist() for q, v in r.items()} for k, r in res.items()})


class Slow(N.Policy):
    """The network with its moves low-pass filtered: what is applied (and what it is told it did) is `beta`
    of the last applied move and the rest of the new one. The sum of the moves is kept, a turn is not."""

    def __init__(self, net, beta=.5):
        super().__init__(net)
        self.beta = beta

    def __call__(self, prog, st, k, obs):
        from aio.frame import canonical
        last = self.prev
        super().__call__(prog, st, k, obs)
        self.prev = self.beta*last+(1-self.beta)*self.prev
        return (self.prev*canonical(obs.float())[1]).to(prog.dtype)


EDGES = (.5, 1., 2., 5., 10.)                                     # |error| of a tracked row, bands


@torch.no_grad()
def lined_out(prog, ctl, late):
    """A controller on every episode -> what it does `late` and more decisions after an event (dict)."""
    st = C.reset(prog)
    ctl.reset(prog)
    c, dev = prog.c, prog.device
    trk, mvm, rowm = c['trk'] > 0, c['mvm'] > 0, (c['trk']+c['zon']) > 0
    q = {k: torch.zeros((), dtype=torch.float64, device=dev) for k in
         ('rows', 'out', 'trk', 'same', 'pairs', 'mv', 'move', 'still', 'flip', 'limit', 'aa', 'ab', 'bb', 'push', 'own', 'own_limit', 'close')}
    hist, edges = torch.zeros(len(EDGES)+1, dtype=torch.float64, device=dev), torch.tensor(EDGES, device=dev)
    esum, esq, en = (torch.zeros_like(c['trk'], dtype=torch.float64) for _ in range(3))
    le = la = None
    row, way = prog.pi['row'], torch.sign(prog.pi['kc'])          # each MV's published row and the way its formula PI moves it
    for k in range(HORIZON):
        a = ctl(prog, st, k, C.observe(prog, st, k))
        e = torch.gather(st['target']-st['y'], 1, row)            # before the step, as the controller saw it
        st, out = C.step(prog, st, a, k)
        seg = prog.seg_at(k)
        lt = ((k-prog.seg_k[prog.ar, seg]) >= late) & (k >= prog.k_first)
        err, task = out['err'], prog.task(seg)
        if bool(lt.any()) and le is not None:
            r, t, m = lt[:, None] & rowm, lt[:, None] & trk, lt[:, None] & mvm
            o = out['outside'] & t
            q['rows'] += r.sum(); q['out'] += (out['outside'] & r).sum(); q['trk'] += t.sum()
            q['pairs'] += o.sum(); q['same'] += (o & (err*le > 0)).sum(); q['close'] += ((le.abs()-err.abs())*o).sum()
            hist.index_add_(0, torch.bucketize(err.abs()[t], edges), torch.ones(int(t.sum()), dtype=torch.float64, device=dev))
            ed = torch.where(t, err.double(), torch.zeros_like(esum))
            esum += ed; esq += ed*ed; en += t.double()
            q['mv'] += m.sum(); q['move'] += (a.abs()*m).sum(); q['still'] += ((a.abs() < 1e-3) & m).sum()
            q['flip'] += ((a*la < 0) & (a.abs() > 1e-3) & (la.abs() > 1e-3) & m).sum()
            span = task['uhi']-task['ulo']
            lim = (st['u']-task['ulo'] < .02*span) | (task['uhi']-st['u'] < .02*span)
            own = m & torch.gather(o, 1, row)                     # MVs whose own row is outside
            q['limit'] += (lim & m).sum(); q['own'] += own.sum(); q['own_limit'] += (lim & own).sum()
            q['push'] += (a*way*torch.sign(e)*own).sum()
            q['aa'] += (a*a*m).sum(); q['ab'] += (a*la*m).sum(); q['bb'] += (la*la*m).sum()
        le, la = err, a
    prog.set_teacher()
    f = lambda x, y: round(float(q[x]/q[y].clamp_min(1.)), 4)     # noqa: E731
    mean = esum/en.clamp_min(1.)
    off, swing = mean.abs()[en > 20], (esq/en.clamp_min(1.)-mean*mean).clamp_min(0.).sqrt()[en > 20]
    return dict(outside=f('out', 'rows'), keeps_sign=f('same', 'pairs'), rows_off_a_band=round(float((off > 1).double().mean()), 4),
                offset=round(float(off.median()), 3), swing=round(float(swing.median()), 3), rows_swinging_a_band=round(float((swing > 1).double().mean()), 4),
                move=f('move', 'mv'), still=f('still', 'mv'), flip=f('flip', 'mv'), at_limit=f('limit', 'mv'), push=f('push', 'own'),
                at_limit_own=f('own_limit', 'own'), closing=f('close', 'pairs'),
                move_follows_last=round(float(q['ab']/(q['aa']*q['bb']).sqrt().clamp_min(1e-12)), 3),
                error_bands=[round(float(x), 3) for x in hist/hist.sum().clamp_min(1.)])


def quiet(a, dev):
    net, _, ck = load(a.ckpt, dev)
    prog = C.build(a.val_n, a.val_seed, dev, 'cpu')
    ctls = [('pi', CT.PI()), ('dmc', CT.Teacher('fixed')), ('now', CT.Teacher('now')), ('net', N.Policy(net)),
            ('net_known', N.Policy(net, known=True))]+[(f'net_slow{b}', Slow(net, b)) for b in a.slow]
    print(json.dumps(dict(late=a.late, error_bands_edges=EDGES)), flush=True)
    res, score = {}, {}
    for name, ctl in ctls:
        res[name], score[name] = lined_out(prog, ctl, a.late), measure(prog, ctl)[0]['score']
        res[name]['x_score'] = round(float(score[name].mean()/score['pi'].mean()), 3)
        print(json.dumps(dict(ctl=name, **res[name])), flush=True)
    # the score against the formula PI by kind of plant: where the network loses
    c = prog.c
    groups = [(f'arch {ARCH[i]}', c['arch'] == i) for i in range(len(ARCH))]
    groups += [(f'{i} MVs', (c['mvm'] > 0).sum(-1) == i) for i in range(1, c['mvm'].shape[1]+1)]
    groups += [('integrating row', (c['integ']*c['trk']).sum(-1) > 0), ('no integrating row', (c['integ']*c['trk']).sum(-1) == 0),
               ('zone rows', c['zon'].sum(-1) > 0), ('constrained point', prog.con.any(-1)),
               ('no constrained point', ~prog.con.any(-1))]
    kind = []
    for name, g in groups:
        if int(g.sum()) >= 4:
            x = lambda k: round(float(score[k][g].sum()/score['pi'][g].sum().clamp_min(1e-9)), 3)      # noqa: E731
            kind.append(dict(kind=name, episodes=int(g.sum()), share_of_pi_score=round(float(score['pi'][g].sum()/score['pi'].sum()), 3),
                             **{k: x(k) for k in ('dmc', 'now', 'net', 'net_known')}))
            print(json.dumps(kind[-1]), flush=True)
    return dict(late=a.late, edges=EDGES, lines=res, by_kind=kind, per_episode={k: v.tolist() for k, v in score.items()})

def horizon(a, dev):
    net, critic, ck = load(a.ckpt, dev)
    cfg = dict(ck['config'], drive=0., drive_noise=.2, bc_floor=0.)
    cfg.update(bc_iters=0, id_coef=0., lean=False, noise=0., log_w=cfg.get('log_w', 0.), smooth=cfg.get('smooth', 0.),
               window=a.window or cfg.get('sim_window') or cfg['window'])     # the window its simulator gradient was taken over
    t = SimpleNamespace(**cfg)                                    # no noise: the gradients at the mean move, the same episode every time
    W, spans = t.window, sorted({x for x in a.spans if x > t.window} | {a.long})
    net.train()
    critic.requires_grad_(False)
    frozen = copy.deepcopy(net).requires_grad_(False)
    gen = torch.Generator(device=dev).manual_seed(1)
    lane = T.Lane(C.build(a.n or 64, a.seed, dev, 'cpu'), t, gen)
    prog, c, n = lane.prog, lane.prog.c, lane.prog.n
    keep = ('st', 'ha', 'hc', 'hs', 'prev', 'pm', 'k')

    def back():
        for q, x in snap.items():
            setattr(lane, q, x)
        lane.level, lane.quiet, lane.score = more[0].clone(), more[1].clone(), more[2].clone()
        prog.g.set_state(noise)

    def trainer(value):
        """The gradient the trainer's window sends into each of its moves, per episode (n, W, NMV); nothing held."""
        back()
        got, call = [], T.Guard.__call__
        T.Guard.__call__ = lambda self, g: (got.append(g.detach().clone()), g)[1]
        try:
            T.window(net, critic, lane, t, dict(bc=0., sim=1., value=value), gen)[0].backward()
        finally:
            T.Guard.__call__ = call
        net.zero_grad(set_to_none=True)
        return torch.stack(got[::-1], 1).double()*n

    def paid():
        """The cost actually paid over W and the stretches `spans` from the lane's point, each as the trainer
        weighs an episode -> its gradient at the window's moves: through the plant and through what the
        policy (a copy with its weights held, after the window) does about them later."""
        back()
        st, ha, hs, prev, k0 = C.detach(lane.st), lane.ha.detach(), lane.hs, lane.prev.detach(), lane.k
        tot = wsum = torch.zeros(n, device=dev)
        disc, acts, got = torch.ones(n, device=dev), [], {}
        for j in range(spans[-1]):
            k, m = k0+j, net if j < W else frozen
            obs, s = T.canonical(C.observe(prog, st, k).float())
            F, hs = T.sums(m, obs, hs)
            u, _, _, _, ha, _ = T.decide(m, lane, obs, prev, ha, s, k, gen, 0., feats=F)
            act = u*s
            if j < W:
                acts.append(act)
            st, out = C.step(prog, st, act.to(prog.dtype), k)
            cost = torch.where(out['fail'], torch.full_like(disc, t.fail_cost), OR.objective(out, st, c, t.tau, t.cost_w, t.log_w).float())
            tot, wsum, disc, prev = tot+disc*cost, wsum+disc, disc*t.gamma, u
            if j+1 in (W, *spans):
                g = torch.autograd.grad((tot/(tot.detach()+t.floor*wsum)).sum(), acts, retain_graph=j+1 < spans[-1])
                got[j+1] = torch.stack(g, 1).double()
        return got

    norm = lambda g: g.flatten(1).norm(dim=1)                     # noqa: E731
    cos = lambda x, y: (x*y).flatten(1).sum(1)/(norm(x)*norm(y)).clamp_min(1e-300)      # noqa: E731
    qt = lambda x, w=None: [round(float(v), 3) for v in torch.quantile(x if w is None else x[w], torch.tensor([.1, .5, .9], dtype=x.dtype, device=x.device))] \
        if (w is None or bool(w.any())) else None                 # noqa: E731
    rows = []
    print(json.dumps(dict(window=W, long=a.long, gamma=t.gamma, episodes=n, at=a.at, read='three numbers: over the episodes the 10 %, the median, the 90 %')), flush=True)
    for k0 in a.at:
        if k0+spans[-1] > HORIZON:
            continue
        with torch.no_grad():
            T.advance(net, critic, lane, k0-lane.k, gen, cfg_noise(ck))
        snap, more, noise = {q: getattr(lane, q) for q in keep}, (lane.level.clone(), lane.quiet.clone(), lane.score.clone()), prog.g.get_state()
        t0 = time.time()
        g1, g0, true = trainer(1.), trainer(0.), paid()
        far, grow = true[spans[-1]], norm(true[spans[-1]])/norm(true[W]).clamp_min(1e-300)
        ok = (norm(far) > 0) & (norm(g1) > 0) & torch.isfinite(norm(far)) & torch.isfinite(norm(g1))
        calm, wild = ok & (grow < 10), ok & (grow >= 10)
        r = dict(at=k0, episodes=int(ok.sum()), trainer_vs_paid=qt(cos(g1, far), ok), against_it=round(float((cos(g1, far)[ok] < 0).double().mean()), 3),
                 window_alone_vs_paid=qt(cos(g0, far), ok), check_window=qt(cos(g0, true[W]), ok),
                 paid_grows=qt(grow, ok), grows_tenfold=round(float(wild.sum()/ok.sum().clamp_min(1)), 3),
                 trainer_vs_paid_calm=qt(cos(g1, far), calm), trainer_vs_paid_grown=qt(cos(g1, far), wild),
                 size_trainer_to_paid=qt(norm(g1)/norm(far).clamp_min(1e-300), ok),
                 paid_over={str(x): qt(cos(true[x], far), ok) for x in (W, *spans[:-1])}, seconds=round(time.time()-t0, 1))
        rows.append(r)
        print(json.dumps(r), flush=True)
        back()
    return dict(window=W, long=a.long, rows=rows)


def cfg_noise(ck):
    return ck['config'].get('noise', .05)


def fit(a, dev):
    net, critic, ck = load(a.ckpt, dev)
    cfg = dict(ck['config'], drive=0., drive_noise=.2, bc_floor=0.)
    was = dict(cost_w=cfg.get('cost_w', .1), log_w=cfg.get('log_w', 0.))
    cfg.update(cost_w=was['cost_w'] if a.cost_w is None else a.cost_w, log_w=was['log_w'] if a.log_w is None else a.log_w,
               noise=cfg['noise'] if a.noise is None else a.noise, smooth=cfg.get('smooth', 0.) if a.smooth is None else a.smooth,
               window=a.window or cfg.get('sim_window') or cfg['window'], lean=bool(a.window) or cfg.get('lean', True), cap=a.cap or cfg['cap'], guard=a.guard or cfg['guard'], bc_iters=0)   # a longer window: its graphs rebuilt, not kept
    t = SimpleNamespace(**cfg)
    gen = torch.Generator(device=dev).manual_seed(1)
    prog, held = C.build(a.n or 256, a.seed, dev, 'cpu'), C.build(a.val_n, a.val_seed, dev, 'cpu')
    pi, pih = measure(prog, CT.PI())[0], measure(held, CT.PI())[0]
    opt, copt = torch.optim.Adam(net.parameters(), lr=a.lr), torch.optim.Adam(critic.parameters(), lr=cfg.get('critic_lr', 3e-4))
    still = torch.optim.SGD(net.parameters(), lr=0.)
    temp, rows = dict(usual=ck.get('usual') if ck.get('guard_units') == T.GUARD_UNITS else None), []   # an older one: measured again
    head = dict(trained_on=was, objective=dict(cost_w=t.cost_w, log_w=t.log_w, smooth=t.smooth), window=t.window, cap=t.cap, guard=t.guard, value=a.value, noise=t.noise, lr=a.lr,
                episodes=prog.n)

    def look(tag, both):
        net.eval()
        r = dict(head, at=tag, fixed=line('net', measure(prog, N.Policy(net))[0], pi, was['cost_w'], was['log_w']))
        if both:
            r['validation'] = line('net', measure(held, N.Policy(net))[0], pih, was['cost_w'], was['log_w'])
        net.train()
        rows.append(r)
        short = lambda d: {k: d[k] for k in ('x_score', 'x_count', 'x_huber', 'x_log', 'near', 'far', 'late_rate', 'failed')}      # noqa: E731
        print(json.dumps(dict(at=tag, fixed=short(r['fixed']), **({'validation': short(r['validation'])} if both else {}))), flush=True)

    print(json.dumps(head), flush=True)
    look('start', True)
    total = a.critic_passes+a.passes
    for p in range(total):
        only = p < a.critic_passes
        lane, t0, logs = T.Lane(prog, t, gen), time.time(), []
        while not lane.done:
            logs.append(T.update(net, critic, still if only else opt, copt, lane, t, dict(bc=0., sim=1., value=0. if only else a.value), gen, temp))
        avg = lambda k: round(sum(x[k] for x in logs if x.get(k) is not None)/max(len(logs), 1), 4)      # noqa: E731
        print(json.dumps(dict(pass_=p, critic_only=only, windows=len(logs), seconds=round(time.time()-t0, 1), cost=avg('cost'), gn=avg('gn'),
                              ev=avg('ev'), capped=avg('capped'), short=avg('short'), skip=avg('skip'))), flush=True)
        if not only:
            look(f'pass {p-a.critic_passes+1}', p == total-1)
    return dict(head, rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=('where', 'quiet', 'horizon', 'fit'))
    ap.add_argument('ckpt', type=pathlib.Path)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--val-n', type=int, default=256)
    ap.add_argument('--val-seed', type=int, default=10**6)
    ap.add_argument('--n', type=int, help='the fixed episodes (fit: 256, horizon: 64)')
    ap.add_argument('--long', type=int, default=256, help='horizon: decisions the cost actually paid is counted over')
    ap.add_argument('--spans', type=int, nargs='*', default=[64, 128], help='horizon: shorter stretches reported too')
    ap.add_argument('--at', type=int, nargs='*', default=[64, 200, 336, 464], help='horizon: the points of the episodes')
    ap.add_argument('--window', type=int, help='fit, horizon: decisions differentiated through at a time (default: the checkpoint\'s)')
    ap.add_argument('--cap', type=float, help='fit: the guard\'s cap, multiples of the usual gradient (default: the checkpoint\'s)')
    ap.add_argument('--guard', choices=('cut', 'open', 'cap'), help='fit: what is done about an episode whose gradient was held (default: the checkpoint\'s)')
    ap.add_argument('--seed', type=int, default=(7 << 32)+1, help='fit: their seed (training seeds are above 2^32)')
    ap.add_argument('--late', type=int, default=150, help='quiet: decisions after an event from which a stretch counts')
    ap.add_argument('--slow', type=float, nargs='*', default=[.5, .8], help='quiet: the low-pass shares the network is run with once more')
    ap.add_argument('--passes', type=int, default=3)
    ap.add_argument('--critic-passes', type=int, default=1)
    ap.add_argument('--cost-w', type=float, help='fit: the objective\'s Huber share (default: the checkpoint\'s)')
    ap.add_argument('--log-w', type=float, help='fit: its log share (default: the checkpoint\'s)')
    ap.add_argument('--value', type=float, default=1., help='fit: weight of the critic\'s value in the actor\'s objective')
    ap.add_argument('--noise', type=float)
    ap.add_argument('--smooth', type=float, help='fit: the weight on a move that turns back (default: the checkpoint\'s)')
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--out', type=pathlib.Path)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    res = dict(where=where, quiet=quiet, horizon=horizon, fit=fit)[a.what](a, torch.device(a.device))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res))


if __name__ == '__main__':
    main()

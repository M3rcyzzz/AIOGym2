"""sim-replay.json for the research site: one episode of the unified simulation core (aio/sim, docs/sim.md) under
every reference controller and the network, in the page's suite replay format (version 1, process 'sim').

aio.sim.replay runs the episode (torch, on the training machine); this file only reshapes its JSON (plain Python):

  python -m aio.sim.replay --out raw.json --ckpt D:/aiogym2-runs/sim-r1/seed1/best.pt          # baolong
  python3 -m experiments.shared_rl_view.research_site_sim_replay raw.json --out sim-replay.json  # anywhere
  python3 -m experiments.shared_rl_view.research_site_sim_replay raw.json --start-of sim-r1/seed1 --out sim-replay.json
      # the checkpoint replayed is the round's starting weights (aio.train --init), not one it trained

Every controller runs the same plant, events and noise (site rule 2). Each plant of the core is new, so the file
carries the plant's structure for the screen (the 'sim' screen in zooScreens.ts is drawn from it). Values are in
% of each channel's range (the core's units are 0-1 of range). Zone limits and priorities move with events: the
events carry them in `state`, the zone rows the limit steps in `lims`. The outside count of every row and step
is the core's own (aio.sim.replay writes it); the scores in the file must add up to the controllers' scores. A run
whose reachable reference differs from the first one's (dead times that follow a throughput MV) carries its own `ref`.
On a fat plant a spare MV's resting value is a tracked channel of its own (`rest`, tag R<MV>: the row reads that MV's
flow); the process CVs keep the numbering CV1.., and `start.nr` counts the resting rows. An event kind this file does
not know is refused, not skipped.
"""
import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

CST = timezone(timedelta(hours=8))
ARCH = dict(decoupled=('decoupled', '解耦'), series=('in series', '串级'), interacting=('interacting', '交互'),
            separation=('separation', '分离'), blending=('blending', '调和'), sparse=('sparse', '稀疏'), mechanism=('from the model', '按机理模型'))
GRADUAL = ('process', 'both', 'load', 'dv', 'over_load', 'return')   # aio.sim.replay gives a ramp only where it applies
NONE = 1e9   # a limit that is not there (state keys are numbers)
# id in the raw file -> page controller id, kind, label, label_zh, short, short_zh, note, note_zh
# mechanistic plants of the core (aio.sim.plants): page name, en and zh
PLANTS = dict(tep=('Tennessee Eastman (TEP)', 'TEP 田纳西–伊斯曼'), tep9=('Tennessee Eastman (TEP), 9 MVs', 'TEP 田纳西–伊斯曼（9 个 MV）'),
              bsm1=('Activated sludge (BSM1)', 'BSM1 活性污泥污水处理'), column=('Distillation column A (Skogestad)', '精馏塔 A（Skogestad）'),
              btcol=('Benzene–toluene column', '苯–甲苯精馏塔'), meoh=('Methanol–water column', '甲醇–水精馏塔'), debut=('Debutanizer', '脱丁烷塔'),
              excstr=('Exothermic CSTR', '放热 CSTR'), vdv=('Van de Vusse reactor', 'Van de Vusse 反应器'), qtank=('Quadruple tank', '四水箱'),
              ph=('pH neutralisation', 'pH 中和'), bio=('Bioreactor', '生物反应器'), evap=('Evaporator', '蒸发器'), furnace=('Two-pass fired heater', '双程加热炉'),
              gasnet=('Gas pressure-reducing station', '燃气调压站'), hex=('Heat-exchanger network', '换热网络'), bldg=('Building heating', '建筑供暖'),
              comp=('Centrifugal compressor', '离心压缩机'), rsr=('Reactor, column and recycle', '反应–精馏–循环'),
              amm=('Ammonia converter (quench-cooled fixed bed)', '氨合成塔（冷激式固定床）'), heater=('Crude heater, natural draft', '原油加热炉（自然通风）'))

CONTROLLERS = {
    'aio': ('rl', 'rl', 'AIO model', 'AIO 模型', 'AIO', 'AIO', '', ''),
    'pi': ('fpi', 'ref', 'Self-tuning PID (baseline)', '自整定PID（基线）', 'Self-tuning PID', '自整定PID',
           'One PI loop per published pairing, tuned once by the lambda rule on the starting plant\'s exact step response: what a self-tuner gets from a commissioning step test.',
           '按公布的配对每对一个 PI 回路，用起始装置的精确阶跃响应按 λ 规则整定一次：相当于投运阶跃试验后的自整定。'),
    'dmc': ('dmc', 'dmc', 'Static-model MPC', '静态模型MPC', 'Static MPC', '静态模型MPC',
            'Constrained MPC given the plant\'s exact model once at the start for free, corrected from the instruments alone; blind to unmeasured loads, its model never updated after events. The upper reference of the exam tables.',
            '开头白拿一次装置的准确模型、只用仪表读数校正的带约束 MPC；看不到不可测负荷，事件后模型不更新。考试表里的上参照。'),
    'teacher': ('teacher', 'oracle', 'Oracle MPC (training teacher)', 'oracle MPC（训练用的老师）', 'Oracle MPC', 'oracle MPC',
                'Knows the exact model and sees the unmeasured loads; its model catches up 10–40 decisions after a process change; does not know later events. A strong reference, not a bound, and absent at deployment.',
                '知道精确模型、看得见不可测负荷，过程变化后 10–40 个决策才跟上新模型，不知道后面的事件。强参照，不是上限；部署时不存在。'),
    'hold': ('hold', 'baseline', 'No action (MVs held)', '不动（MV 保持）', 'No action', '不动', '', ''),
}
ORDER = ('aio', 'pi', 'dmc', 'teacher', 'hold')


def decimals(res):
    return max(0, min(6, math.ceil(-math.log10(res))))


def pct(x, d):
    return None if x is None else round(100*x, d)


def fmt(x, d):
    return f'{x:.{d}f}%'


def recount(c, rows, seg_at):
    """The outside flags from the stored values (rounded to 5 decimals, bands to 4), as core.step counts them: a tracked
    row outside ok_lo..ok_hi (its band around the reachable reference, wider at a constrained operating point; files
    without it: the band), a zone row against the limits of the segment in force, widened by the segment's allowance
    (`allow`, bands: where the best steady point has the CV at its limit)."""
    out = []
    for k, (y, ref) in enumerate(zip(c['y'], c['ref'])):
        s = seg_at(k)
        ok = (c['ok_lo'][k], c['ok_hi'][k]) if c.get('ok_lo') else None

        def off(r, row):
            if ok and r < len(ok[0]):
                return int(y[r] < ok[0][r] or y[r] > ok[1][r])
            return int(abs(y[r]-ref[r]) > row['band'])
        give = lambda r, row: (s.get('allow') or [0]*len(rows))[r]*row['band']   # noqa: E731
        out.append([off(r, row) if row['role'] == 'tracked'
                    else int((s['lo'][r] is not None and y[r] < s['lo'][r]-give(r, row)) or (s['hi'][r] is not None and y[r] > s['hi'][r]+give(r, row)))
                    if row['role'] == 'zone' else 0 for r, row in enumerate(rows)])
    return out


def convert(raw, start_of=None, now=None):
    st, k0, dt = raw['struct'], raw['k_first'], round(raw['dt_s'], 2) if raw['dt_s'] < 10 else round(raw['dt_s'])
    rows, nm, H = st['rows'], st['nm'], raw['horizon']
    segs = sorted(raw['segments'], key=lambda s: s['k'])
    seg_at = lambda k: [s for s in segs if s['k'] <= k][-1]   # noqa: E731
    # ---- channels: tracked rows CV1.., resting values of spare MVs R<MV>, zone rows Z1.., measured disturbances D1..
    cvs, n = [], dict(tracked=0, zone=0, dv=0, rest=0)
    for r, row in enumerate(rows):
        role = row['role']; n[role] += 1
        band = 100*row['band']
        rest = role == 'tracked' and bool(row.get('rest'))
        if rest:               # a spare MV's resting value: the row reads that MV's flow (a fat plant, docs/sim.md)
            j = row['mv']; n['rest'] += 1
            label, label_zh, tag = f"Resting value of MV{j+1} (its flow)", f"MV{j+1} 的静置值（它的流量）", f'R{j+1}'
        elif role == 'tracked':
            j = row['mv']; k_ = n[role]-n['rest']
            label, label_zh = f"CV{k_} (paired with MV{j+1})", f"CV{k_}（与 MV{j+1} 配对）"
            if row['integrating']:
                label, label_zh = label + ', integrating', label_zh + '，积分型'
            tag = f'CV{k_}'
        elif role == 'zone':
            label, label_zh, tag = f"Zone CV Z{n[role]}", f"区间量 Z{n[role]}", f'Z{n[role]}'
        else:
            label, label_zh, tag = f"Measured disturbance D{n[role]}", f"实测扰动 D{n[role]}", f'D{n[role]}'
        if row.get('name'):    # a mechanistic plant: its own name of the output
            label, label_zh = f"{row['name']} · {label}", f"{row['name']} · {label_zh}"
            if role == 'dv':
                tag = row['name']
        dec = min(3, decimals(band/50)) if role == 'tracked' else 2
        c = dict(key=f'y{r}', row=r, name=f'cv{r}', tag=tag, label=label, label_zh=label_zh, unit='%', tracked=role == 'tracked',
                 zone=role == 'zone', band=round(band, 4) if role == 'tracked' else None, show=True, dec=dec, integrating=row['integrating'],
                 lower=None, upper=None)
        if rest:
            c['rest'] = True
        if role == 'zone':
            steps = []
            for s in segs:
                lo, hi = pct(s['lo'][r], 3), pct(s['hi'][r], 3)
                if not steps or (steps[-1][1], steps[-1][2]) != (lo, hi):
                    steps.append([s['k']*dt, lo, hi])
            c['lower'], c['upper'] = steps[0][1], steps[0][2]
            if len(steps) > 1:
                c['lims'] = steps
            # how far beyond its limit the CV may be before it counts (% of range), where the best steady point has it there: steps [t, give]
            give = [[s['k']*dt, round(band*(s.get('allow') or [0]*len(rows))[r], 3)] for s in segs]
            if any(g for _, g in give):
                c['allow'] = [g for j, g in enumerate(give) if j == 0 or g[1] != give[j-1][1]]
        cvs.append(c)
    nmz = lambda m: f" {m['name']}" if m.get('name') else ''        # noqa: E731
    mvs = [dict(key=f'u{j}', row=j, label=f"MV{j+1}{nmz(m)}" + (', reverse acting' if m['sign'] < 0 else ''), label_zh=f"MV{j+1}{nmz(m)}" + ('，反作用' if m['sign'] < 0 else ''),
                unit='%', lo=0, hi=100, slew=round(100*m['slew'], 3), sign=m['sign'], paired=f"cv{m['row']}", dec=1) for j, m in enumerate(st['mvs'])]
    # the operator's MV limits (5-95 % of range at commissioning, narrowed by an mv_limit event): steps [t, lower, upper]
    if all('u_lo' in s for s in segs):
        for m in mvs:
            steps = []
            for s in segs:
                lo, hi = pct(s['u_lo'][m['row']], 2), pct(s['u_hi'][m['row']], 2)
                if not steps or (steps[-1][1], steps[-1][2]) != (lo, hi):
                    steps.append([s['k']*dt, lo, hi])
            m['lower'], m['upper'] = steps[0][1], steps[0][2]
            if len(steps) > 1:
                m['lims'] = steps
    tag = {c['row']: c['tag'] for c in cvs}

    def state(s):
        """Zone limits and tracked priorities of a segment, for the screen (NONE: no limit on that side)."""
        out = {}
        for c in cvs:
            r = c['row']
            if c['zone']:
                lo, hi = pct(s['lo'][r], 3), pct(s['hi'][r], 3)
                out[f"lo_{c['name']}"], out[f"hi_{c['name']}"] = (NONE if lo is None else lo), (NONE if hi is None else hi)
            elif c['tracked']:
                out[f"w_{c['name']}"] = s['weight'][r]
        if 'u_lo' in s:
            for m in mvs:
                out[f"ulo_{m['key']}"], out[f"uhi_{m['key']}"] = pct(s['u_lo'][m['row']], 2), pct(s['u_hi'][m['row']], 2)
        return out
    # ---- events, with the targets, limits and priorities they set
    ctl0 = raw['controllers'][next(iter(raw['controllers']))]
    tgt = lambda k: ctl0['target'][max(0, k-1)]   # the target published for decision k (trace entry k-1 holds decision k's)  # noqa: E731
    events, nchg = [], 0
    for e in raw['events']:
        k, kind = e['k'], e['kind']
        t, s = k*dt, seg_at(k)
        ramp = e.get('ramp') or 0
        slow_en = f", gradual over {ramp} decisions ({ramp*dt/60:.0f} min)" if ramp and kind in GRADUAL else ''
        slow_zh = f"，{ramp} 个决策（{ramp*dt/60:.0f} 分钟）内渐变" if ramp and kind in GRADUAL else ''
        if kind in ('process', 'both'):
            nchg += 1
            events.append(dict(t=t, kind='process', params=dict(nchg=nchg), state=state(s), label='Plant change' + slow_en, label_zh='装置变化' + slow_zh,
                               short='Plant change', short_zh='装置变化'))
        if kind in ('setpoint', 'both'):
            a, b = tgt(k-1), tgt(k)
            moved = [c for c in cvs if c['tracked'] and abs(b[c['row']]-a[c['row']]) > 1e-9] or [c for c in cvs if c['tracked']]
            parts = [f"{c['tag']} {fmt(100*b[c['row']], min(c['dec'], 2))}" for c in moved]
            events.append(dict(t=t, kind='setpoint', state=state(s), target=[round(100*b[c['row']], 4) for c in cvs if c['tracked']],
                               label='Targets ' + ', '.join(parts), label_zh='目标 ' + '，'.join(parts),
                               short=parts[0] if len(parts) == 1 else f'{len(parts)} targets', short_zh=parts[0] if len(parts) == 1 else f'{len(parts)} 个目标'))
        elif kind in ('load', 'dv'):
            en, zh = ('Unmeasured load', '不可测负荷') if kind == 'load' else ('Measured disturbance', '可测扰动')
            step_en, step_zh = ('ramp' if slow_en else 'step'), ('渐变' if slow_zh else '阶跃')
            events.append(dict(t=t, kind='disturbance', state=state(s), label=f'{en} change' + slow_en if slow_en else f'{en} step',
                               label_zh=f'{zh}变化' + slow_zh if slow_zh else f'{zh}阶跃',
                               short=f'{en} {step_en}', short_zh=f'{zh}{step_zh}'))
        elif kind == 'limit':
            prev = seg_at(k-1)
            moved = [c for c in cvs if c['zone'] and (s['lo'][c['row']], s['hi'][c['row']]) != (prev['lo'][c['row']], prev['hi'][c['row']])] or [c for c in cvs if c['zone']]
            def lim(c, zh=False):
                lo, hi = pct(s['lo'][c['row']], 2), pct(s['hi'][c['row']], 2)
                if lo is not None and hi is not None:
                    return f"{c['tag']} {lo}–{hi}%"
                if hi is not None:
                    return f"{c['tag']} {'不高于' if zh else 'at most'} {hi}%"
                return f"{c['tag']} {'不低于' if zh else 'at least'} {lo}%" if lo is not None else c['tag']
            events.append(dict(t=t, kind='other', state=state(s), label='Limits ' + ', '.join(lim(c) for c in moved), label_zh='限值 ' + '，'.join(lim(c, True) for c in moved),
                               short='Limits move', short_zh='限值移动'))
        elif kind == 'priority':
            parts = [f"{c['tag']} {s['weight'][c['row']]:.2g}" for c in cvs if c['tracked']]
            events.append(dict(t=t, kind='other', state=state(s), label='Priorities ' + ', '.join(parts), label_zh='优先级 ' + '，'.join(parts),
                               short='Priorities change', short_zh='优先级变化'))
        # ---- constrained operating points (docs/sim.md): the targets cannot all be held until the unit comes back
        elif kind == 'over_setpoint':
            a, b = tgt(k-1), tgt(k)
            moved = [c for c in cvs if c['tracked'] and abs(b[c['row']]-a[c['row']]) > 1e-9] or [c for c in cvs if c['tracked']]
            parts = [f"{c['tag']} {fmt(100*b[c['row']], min(c['dec'], 2))}" for c in moved]
            events.append(dict(t=t, kind='setpoint', state=state(s), target=[round(100*b[c['row']], 4) for c in cvs if c['tracked']],
                               label='Targets beyond reach: ' + ', '.join(parts), label_zh='目标超出可达范围：' + '，'.join(parts),
                               short='Targets beyond reach', short_zh='目标超出可达范围'))
        elif kind == 'over_load':
            events.append(dict(t=t, kind='disturbance', state=state(s), label='Load beyond what the valves can follow' + slow_en,
                               label_zh='负荷超出阀门能力' + slow_zh, short='Load beyond the valves', short_zh='负荷超出阀门能力'))
        elif kind == 'mv_limit':
            prev = seg_at(k-1)
            parts = [f"MV{j+1} {pct(s['u_lo'][j], 1)}–{pct(s['u_hi'][j], 1)}%" for j in range(len(mvs))
                     if (s['u_lo'][j], s['u_hi'][j]) != (prev['u_lo'][j], prev['u_hi'][j])] if 'u_lo' in s else []
            events.append(dict(t=t, kind='other', state=state(s), label='Operator narrows ' + (', '.join(parts) or 'an MV limit'),
                               label_zh='操作员收窄 ' + ('，'.join(parts) or 'MV 限值'), short='MV limit narrowed', short_zh='MV 限值收窄'))
        elif kind == 'zone_limit':
            prev = seg_at(k-1)
            moved = [c for c in cvs if c['zone'] and (s['lo'][c['row']], s['hi'][c['row']]) != (prev['lo'][c['row']], prev['hi'][c['row']])] or [c for c in cvs if c['zone']]

            def past(c, zh=False):
                lo, hi = pct(s['lo'][c['row']], 2), pct(s['hi'][c['row']], 2)
                if hi is not None and hi != pct(prev['hi'][c['row']], 2):
                    return f"{c['tag']} {'不高于' if zh else 'at most'} {hi}%"
                return f"{c['tag']} {'不低于' if zh else 'at least'} {lo}%" if lo is not None else c['tag']
            events.append(dict(t=t, kind='other', state=state(s), label='Limit moved past where the unit runs: ' + ', '.join(past(c) for c in moved),
                               label_zh='区间限值收紧到现值之外：' + '，'.join(past(c, True) for c in moved), short='Limit past the unit', short_zh='区间限值收紧'))
        elif kind == 'mv_out':
            prev = seg_at(k-1)
            gone = [j for j in range(len(mvs)) if (s['u_lo'][j], s['u_hi'][j]) != (prev['u_lo'][j], prev['u_hi'][j])] if 'u_lo' in s else []
            en = ', '.join(f"MV{j+1} at {pct((s['u_lo'][j]+s['u_hi'][j])/2, 1)}%" for j in gone) or 'an MV'
            zh = '，'.join(f"MV{j+1} 停在 {pct((s['u_lo'][j]+s['u_hi'][j])/2, 1)}%" for j in gone) or '一个 MV'
            a, b = tgt(k-1), tgt(k)
            moved = [c for c in cvs if c['tracked'] and abs(b[c['row']]-a[c['row']]) > 1e-9]
            parts = [f"{c['tag']} {fmt(100*b[c['row']], min(c['dec'], 2))}" for c in moved]
            ev = dict(t=t, kind='setpoint' if moved else 'other', state=state(s), label=f'Out of service: {en}' + ('; targets ' + ', '.join(parts) if parts else ''),
                      label_zh=f'退出控制：{zh}' + ('；目标 ' + '，'.join(parts) if parts else ''), short='MV out of service', short_zh='MV 退出控制')
            if moved:
                ev['target'] = [round(100*b[c['row']], 4) for c in cvs if c['tracked']]
            events.append(ev)
        elif kind == 'push':   # the operator moves a spare MV's resting value to its valve's limit: the unit rides whatever stops it
            a, b = tgt(k-1), tgt(k)
            moved = [c for c in cvs if c.get('rest') and abs(b[c['row']]-a[c['row']]) > 1e-9] or [c for c in cvs if c.get('rest')]
            parts = [f"{c['tag']} {fmt(100*b[c['row']], min(c['dec'], 2))}" for c in moved]
            events.append(dict(t=t, kind='setpoint', state=state(s), target=[round(100*b[c['row']], 4) for c in cvs if c['tracked']],
                               label='Resting value moved to the valve limit: ' + ', '.join(parts), label_zh='静置值推到阀门限值：' + '，'.join(parts),
                               short='Resting value to the limit', short_zh='静置值推到限值'))
        elif kind == 'return':
            a, b = tgt(k-1), tgt(k)
            back = [c for c in cvs if c['tracked'] and abs(b[c['row']]-a[c['row']]) > 1e-9]
            ev = dict(t=t, kind='setpoint' if back else 'other', state=state(s), label='Back to the earlier operating point' + slow_en,
                      label_zh='回到原工况' + slow_zh, short='Back', short_zh='回到原工况')
            if back:
                ev['target'] = [round(100*b[c['row']], 4) for c in cvs if c['tracked']]
            events.append(ev)
        elif kind != 'process':
            raise ValueError(f'event kind {kind!r} is not known to the page format')
    # ---- runs: the true values, the reachable reference, the core's outside flags
    ref, runs = None, []
    for rid in [q for q in ORDER if q in raw['controllers']]:
        c = raw['controllers'][rid]
        cid, kind, label, label_zh, short, short_zh, note, note_zh = CONTROLLERS[rid]
        y, u = c['y'], c['u']
        out = c.get('outside') or recount(c, rows, seg_at)   # a file from before aio.sim.replay wrote the core's own flags
        assert len(y) == H and len(out) == H, f'{rid}: {len(y)} steps, want {H}'
        cols = {ch['key']: [round(100*y[0][ch['row']], ch['dec'])] + [round(100*v[ch['row']], ch['dec']) for v in y] for ch in cvs}
        cols.update({m['key']: [round(100*u[0][m['row']], m['dec'])] + [round(100*v[m['row']], m['dec']) for v in u] for m in mvs})
        on = [k >= k0 for k in range(H)]
        cols['out'] = [0] + [sum(o) if on[k] else 0 for k, o in enumerate(out)]
        r_ = {ch['key']: [round(100*c['ref'][0][ch['row']], ch['dec'])] + [round(100*v[ch['row']], ch['dec']) for v in c['ref']] for ch in cvs if ch['tracked']}
        # the reachable reference is the target one dead time back; on a load-dependent plant whose throughput is an MV the
        # dead time follows that MV, so a controller can have its own reference (then carried by the run)
        if ref is None:
            ref = r_
        per = {ch['name']: sum(o[ch['row']] for k, o in enumerate(out) if on[k]) for ch in cvs}
        counted = sum(cols['out'])
        if not c['failed'] and counted != c['score']:
            if c.get('outside'):
                sys.exit(f"refusing to write: {rid} counts {counted} outside decisions here and scores {c['score']}")
            print(f"warning: {rid}: recounted {counted} outside decisions from rounded values, the core scores {c['score']}", file=sys.stderr)
        run = dict(controller=cid, kind=kind, label=label, label_zh=label_zh, short=short, short_zh=short_zh, note=note, note_zh=note_zh,
                   n=H+1, cols=cols, total_s=c['score']*dt, failed=bool(c['failed']), score=c['score'], cost=c['cost'],
                   track_s=sum(per[ch['name']] for ch in cvs if ch['tracked'])*dt, viol_s=sum(per[ch['name']] for ch in cvs if ch['zone'])*dt,
                   **{f"{ch['name']}_s": per[ch['name']]*dt for ch in cvs if ch['tracked']}, **{f"{ch['name']}_viol_s": per[ch['name']]*dt for ch in cvs if ch['zone']})
        if r_ != ref:
            run['ref'] = r_
        # the stretch a tracked CV may take without counting as outside, where it is wider than the band around the
        # reference: at a constrained operating point, from the target to the reachable optimum and a band beyond
        if c.get('ok_lo'):
            ok = {}
            for ch in cvs:
                if not ch['tracked']:
                    continue
                r = ch['row']
                assert r < len(c['ok_lo'][0]), f'{rid}: no ok band for tracked row {r}'
                lo = [round(100*v[r], ch['dec']) for v in c['ok_lo']]
                hi = [round(100*v[r], ch['dec']) for v in c['ok_hi']]
                rr, b, tol = r_[ch['key']][1:], ch['band'], 1.5*10**-ch['dec']
                if any(abs(l_-(x-b)) > tol or abs(h_-(x+b)) > tol for l_, h_, x in zip(lo, hi, rr)):
                    ok[ch['key']] = [[lo[0]] + lo, [hi[0]] + hi]
            if ok:
                run['ok'] = ok
        if rid == 'aio':
            src = raw.get('source') or {}
            parts = (src.get('file') or '').split('/')
            if start_of:   # the round's starting weights (aio.train --init)
                run_id, seed = start_of.split('/')
                name = parts[-1] if parts else 'init'
                run['label'], run['label_zh'] = f'AIO model · starting weights ({name}), not trained on the core yet', f'AIO 模型 · 起点权重（{name}），还没在新核心上训练'
                run['note'], run['note_zh'] = 'The round starts from these weights; the trained checkpoint replaces them once it has one.', '本轮从这份权重出发，有了训练后的 checkpoint 就换掉。'
                meta = dict(run=run_id, seed=seed, which='init', file=src.get('file'), x_pi=None)
            else:
                assert len(parts) >= 3 and src.get('trainer') == 'aio.train', f"not an aio.train checkpoint ({src.get('file')}); pass --start-of for starting weights"
                run_id, seed, which = parts[-3], parts[-2], parts[-1].removesuffix('.pt')
                xs = src.get('x_pi')
                # plain words on the page (site rule: no run codes, no multiples of self-tuning PID); the run stays in `source`
                run['label'], run['label_zh'] = 'AIO model (current weights)', 'AIO 模型（现行权重）'
                sm = src.get('samples') or 0
                run['note'] = f"The weights in service, trained on {sm/1e6:,.0f} M plant steps; it reads the plant from its own moves, with no test."
                run['note_zh'] = f"现行的这份权重，训练见过 {sm/1e8:,.1f} 亿个装置步；不做测试，靠自己的动作读装置。"
                meta = dict(run=run_id, seed=seed, which=which, file=src.get('file'), iter=src.get('iter'), samples=sm, hours=src.get('hours'), x_pi=xs)
            run['source'] = dict(meta, generated=(now or datetime.now(CST)).isoformat(timespec='minutes'))
        runs.append(run)
    assert runs, 'no controllers in the file'
    # ---- the plant and the footnote
    arch_en, arch_zh = ARCH.get(st['arch'], (st['arch'], st['arch']))
    nt, nz, nd, nq, nr = n['tracked'], n['zone'], n['dv'], st['nq'], n['rest']
    ncv = nt-nr                 # the process CVs; the other tracked rows are resting values of spare MVs
    shape = f'{nm} MV{"s" if nm > 1 else ""}' + (f' ({nr} spare, with a resting value)' if nr else '') + f', {ncv} tracked CV{"s" if ncv > 1 else ""}' \
        + (f', {nz} zone CV{"s" if nz > 1 else ""}' if nz else '') \
        + (f', {nd} measured disturbance{"s" if nd > 1 else ""}' if nd else '') + (f' and {nq} unmeasured load{"s" if nq > 1 else ""}' if nq else '')
    shape_zh = f'{nm} 个 MV' + (f'（{nr} 个富余，有静置值）' if nr else '') + f'、{ncv} 个跟踪 CV' + (f'、{nz} 个区间量' if nz else '') \
        + (f'、{nd} 个实测扰动' if nd else '') + (f'、{nq} 个不可测负荷' if nq else '')
    rest_en = (" A spare MV has a resting value (R: the row reads its flow) that ranks below every CV: where the targets cannot all be held it gives first, "
               "and it counts as outside only beyond its own wide band.") if nr else ''
    rest_zh = "富余 MV 有静置值（R 行读它的流量），排在所有 CV 之后：目标不能全保住时它先让，超出自己的宽带才算带外。" if nr else ''
    nl, nl_zh = ('linear', '线性') if st['linear'] else ('nonlinear', '非线性')
    if st['load_dependent']:
        nl, nl_zh = nl + ', its dynamics scaled by the throughput', nl_zh + '、动态随处理量缩放'
    hours = H*dt/3600
    dur_en, dur_zh = (f'{hours:.0f} h', f'{hours:.0f} 小时') if hours >= 1 else (f'{hours*60:.0f} min', f'{hours*60:.0f} 分钟')
    ev_n = len(raw['events'])
    pk = raw.get('pick') or ''
    pick_en, pick_zh = (('of the episodes that reach a constrained operating point, the one with the median static-model MPC score', '在到过约束工况的回合里取静态模型MPC 成绩居中的那个')
                        if 'constrained' in pk else ('of the episodes with a level (an integrating CV), the one with the median static-model MPC score', '在带液位（积分量）的回合里取静态模型MPC 成绩居中的那个')
                        if 'level' in pk else ('the one with the median static-model MPC score', '取静态模型MPC 成绩居中的那个') if 'median' in pk
                        else ('chosen by index', '按序号指定'))
    con = [s_ for s_ in segs if s_.get('constrained')]
    con_en = (f" {len(con)} segment{'s' if len(con) > 1 else ''} at a constrained operating point, where the targets cannot all be held: there the reference is the "
              "reachable optimum, and a tracked CV anywhere from its target to the best reachable value, plus a band, is not outside (the wider shading).") if con else ''
    con_zh = (f"其中 {len(con)} 段是约束工况，目标不能全部保住：这时参考是可达最优点，跟踪 CV 落在目标到可达最优值之间、再外扩一个带宽以内都不算带外（较宽的阴影）。") if con else ''
    footnote = (f"Episode {raw['index']} of a fixed replay batch ({raw['n']} episodes, seed {raw['seed']:,}, apart from training and validation), {pick_en}: "
                f"one new plant with {shape}; coupling {arch_en}, time scale {st['time_scale']:g} decisions, dead time {st['dead_ratio']:g} of the time constant, {nl}; "
                f"one decision every {dt:g} s. Over {dur_en}, {ev_n} events.{con_en}{rest_en} Values are in % of each channel's range. References on the same episode: "
                "the self-tuning PID, the static-model MPC, the oracle MPC (the training teacher) and no action. "
                "Outside time counts what the core's score counts, from the first event: tracked CVs outside their band around the reachable reference "
                "(the target one dead time back) and zone CVs outside their limits, per CV; the seconds equal each controller's score times the step.")
    footnote_zh = (f"固定回放批（{raw['n']} 个回合，种子 {raw['seed']:,}，与训练、验证都不相交）的第 {raw['index']} 个回合，{pick_zh}："
                   f"一个新装置，{shape_zh}；耦合结构{arch_zh}，时间尺度 {st['time_scale']:g} 个决策，纯滞后为时间常数的 {st['dead_ratio']:g} 倍，{nl_zh}；"
                   f"每 {dt:g} 秒决策一次。{dur_zh}内 {ev_n} 个事件。{con_zh}{rest_zh}数值是各通道量程的 %。同一回合上的参照：自整定PID、静态模型MPC、"
                   "oracle MPC（训练用的老师）和不动。带外时间与新核心的评分同一口径，从第一个事件起算：跟踪 CV 偏离可达参考（一个纯滞后之前的目标）超出带宽、"
                   "区间 CV 越限，按 CV 逐个累计；秒数等于各控制器的评分乘以步长。")
    start = dict(dt=dt, nm=nm, nt=nt, nr=nr, nz=nz, nd=nd, nq=nq, nchg=0, T=st['time_scale'], dead=st['dead_ratio'], linear=int(st['linear']),
                 load=int(st['load_dependent']), **state(segs[0]))
    q = raw.get('plant')
    if q:                       # a mechanistic plant: named, not "a new plant"
        en, zh = PLANTS.get(q, (q, q))
        footnote = f"Plant: {en}, a mechanistic model of the core (aio.sim.plants), its MVs and outputs named as in the equations. " + footnote.replace('one new plant with', f'{en} with')
        footnote_zh = f"装置：{zh}，新核心里的机理模型（aio.sim.plants），MV 和输出用机理方程里的名字。" + footnote_zh.replace('一个新装置，', f'{zh}，')
    return dict(version=1, process='sim', set='replay', plant=q, plant_label=PLANTS.get(q, (q, q))[0] if q else None, plant_label_zh=PLANTS.get(q, (q, q))[1] if q else None, scenario=f"{raw['seed']}#{raw['index']}", pick=raw.get('pick'), pick_zh=raw.get('pick_zh'),
                dt=dt, horizon_s=H*dt, score_from=k0*dt, cvs=cvs, mvs=mvs, ref=ref, start=start, events=events, runs=runs,
                sim=dict(arch=st['arch'], arch_label=arch_en, arch_label_zh=arch_zh, nq=nq),
                label=f"Replay episode {raw['index']} · {len(events)} events", label_zh=f"回放回合 {raw['index']} · {len(events)} 个事件",
                footnote=footnote, footnote_zh=footnote_zh, generated=(now or datetime.now(CST)).isoformat(timespec='minutes'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw', type=Path, help='the JSON aio.sim.replay wrote')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--start-of', help='RUN/SEED: the replayed checkpoint is that round\'s starting weights (aio.train --init)')
    a = ap.parse_args()
    out = convert(json.loads(a.raw.read_text(encoding='utf-8')), a.start_of)
    a.out.write_text(json.dumps(out, ensure_ascii=True, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({r['controller']: dict(score=r['score'], total_s=r['total_s'], failed=r['failed']) for r in out['runs']}))


if __name__ == '__main__':
    main()

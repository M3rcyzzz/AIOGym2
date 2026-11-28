"""Run the validation and write its results.

  python -m aio.sim.validate.run static              # equations, published numbers, conservation (a minute)
  python -m aio.sim.validate.run numerics [plants]   # the integrator on training episodes (minutes per plant; --jobs)
  python -m aio.sim.validate.run stress [plants]     # the guarded step with any valve position at states the episodes reached
  python -m aio.sim.validate.run colamod DIR         # column A against the original MATLAB code in DIR (Octave)
  python -m aio.sim.validate.run realdata DIR        # records of real plants in DIR (the folders fetch.py writes)
  python -m aio.sim.validate.run boptest URLS SRC [--ctl MODULE:FACTORY --arg WEIGHTS]   # BOPTEST servers (comma separated), its source tree, optional packet controller
  python -m aio.sim.validate.run prior               # the stored real loops (BOPTEST step tests, DaISy records) against the prior as it is now
  python -m aio.sim.validate.run table               # docs/validation.md from reports/validation/*.json

Results are JSON files in reports/validation/, one per part; `table` only reads them.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'reports'/'validation'
ZH = dict(qtank='四水箱', ph='pH 中和', vdv='Van de Vusse 反应器', excstr='放热 CSTR', bio='生物反应器', evap='蒸发器', furnace='加热炉', gasnet='气网',
          hex='换热网络', column='精馏塔 A')
COLUMNS = dict(btcol='苯–甲苯塔', meoh='甲醇–水塔', debut='脱丁烷塔')       # plants.LARGE: the designed columns (section 6 of the document)
SRC = dict(qtank='Johansson 2000', ph='Henson–Seborg 1994', vdv='Klatt–Engell 1998', excstr='Seborg 等教材', bio='Bequette 1998', evap='Newell–Lee 1989',
           column='Skogestad colamod.m', furnace='自建', gasnet='自建', hex='自建')


def static():
    import torch
    from aio.sim import plants as PL
    from aio.sim.validate import checks as CK
    torch.set_num_threads(1)
    res = {}
    for name in PL.FIRST:
        res[name] = dict(equations=CK.equations(name), published=CK.published(name), conservation=CK.conservation(name))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'static.json').write_text(json.dumps(res, indent=1))
    return res


OPEN_LOOP_UNSTABLE = {'excstr'}       # LSODA cannot follow the open-loop replay of a diverging plant: no LSODA cross-check there


def numerics(plants, n, jobs):
    """`jobs` plants at a time, each in its own process; a slow plant does not hold the others up."""
    import tempfile
    import time
    from aio.sim import plants as PL
    OUT.mkdir(parents=True, exist_ok=True)
    todo, live = list(plants or PL.FIRST), {}
    while todo or live:
        while todo and len(live) < jobs:
            p, out = todo.pop(0), tempfile.TemporaryFile('w+')
            exact = '0' if p in OPEN_LOOP_UNSTABLE else '2'
            live[p] = (subprocess.Popen([sys.executable, '-m', 'aio.sim.validate.numerics', p, str(n), exact], stdout=out, text=True, cwd=ROOT), out)
        for p, (proc, out) in list(live.items()):
            if proc.poll() is not None:
                out.seek(0)
                (OUT/f'numerics-{p}.json').write_text(json.dumps(json.loads(out.read()), indent=1))
                print(p, 'done', flush=True)
                del live[p]
        time.sleep(1)


def stress(plants):
    """The guarded step off the training distribution (numerics.stress), one plant after another."""
    import torch
    from aio.sim import plants as PL
    from aio.sim.validate import numerics as NU
    torch.set_num_threads(1)
    f = OUT/'stress.json'
    res = json.loads(f.read_text()) if f.is_file() else {}
    for p in plants or PL.FIRST:
        res[p] = {str(j): NU.stress(p, jump=j) for j in ('slew', .2, 1.)}      # every valve by its own slew limit; a fifth of the span; any position
        f.write_text(json.dumps(res, indent=1))
        print(p, 'done', flush=True)


def colamod(cola_dir):
    import torch
    from aio.sim.validate import colamod as CM
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'colamod.json').write_text(json.dumps(CM.compare(cola_dir), indent=1))


def realdata(root):
    import torch
    from aio.sim.validate import realdata as RD
    torch.set_num_threads(1)
    root, res = Path(root), {}
    find = lambda pat: next(iter(sorted(root.rglob(pat))), None)      # noqa: E731
    for key, pat, fn in (('csth', 'CSTHMaster.mdl', RD.csth), ('tanks', 'dataBenchmark.mat', RD.tanks), ('tanks', 'dataBenchmark.csv', RD.tanks),
                         ('steps', 'T1Heater.xlsx', lambda d: RD.place_column(RD.steps(d))), ('records', 'dryer2.dat.gz', lambda d: RD.records(d.parent))):
        f = find(pat)
        if f is not None and key not in res:
            res[key] = fn(f.parent)
            print(key, 'done', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'realdata.json').write_text(json.dumps(res, indent=1))


def prior():
    """The stored real loops (BOPTEST's step tests, the DaISy records) against the prior as it is now: needs neither
    the servers nor the data. Run it when the generator changes."""
    import torch
    from aio.sim.validate import realdata as RD
    torch.set_num_threads(4)
    f = OUT/'boptest.json'
    if f.is_file():
        b = json.loads(f.read_text())
        b['steps'] = RD.replace(b['steps'])
        f.write_text(dump(b))
    f = OUT/'realdata.json'
    if f.is_file():
        r = json.loads(f.read_text())
        if 'records' in r:
            r['records'] = RD.replace(r['records'])
        if 'steps' in r:
            r['steps'] = RD.place_column(r['steps'])
        f.write_text(json.dumps(r, indent=1))


def dump(o, ind=0):
    """JSON at indent 1 with arrays of numbers on one line (the trajectories of the BOPTEST report)."""
    pad = ' '*(ind+1)
    if isinstance(o, dict) and o:
        return '{\n'+',\n'.join(f'{pad}{json.dumps(k)}: {dump(v, ind+1)}' for k, v in o.items())+'\n'+' '*ind+'}'
    if isinstance(o, list) and any(isinstance(v, (dict, str)) for v in o):
        return '[\n'+',\n'.join(pad+dump(v, ind+1) for v in o)+'\n'+' '*ind+']'
    return json.dumps(o, allow_nan=False)


def boptest(urls, src, ctl=None, arg=None, big=False):
    """Against running BOPTEST servers: the project's reference KPIs, every test case under its own
    controller, step tests of the loops in boptest.LOOPS against the prior, and (with ctl) the exam.
    big: the 15-zone office too (its five periods take 43 minutes on three servers)."""
    import torch
    from aio.sim.validate import boptest as B
    torch.set_num_threads(1)
    urls = urls.split(',')
    cases = [c for c in B.Client(urls[0]).cases() if big or c != 'multizone_office_complex_air']
    res = dict(version=B.Client(urls[0]).call('GET', 'version')['payload']['version'].strip())
    save = lambda: (OUT.mkdir(parents=True, exist_ok=True), (OUT/'boptest.json').write_text(dump(res)))      # noqa: E731
    for key, fn in (('reference', lambda: B.reference(urls, src)), ('steps', lambda: B.steps(urls, src)),
                    ('exam', lambda: B.exams(urls, ctl, res['steps'], arg=arg) if ctl else None), ('baseline', lambda: B.baseline(urls, src, cases))):
        res[key] = fn()
        save()
        print(key, 'done', flush=True)


def _f(v, d=2):
    return f'{v:.{d}f}' if v >= 10**-d else (f'{v:.0e}'.replace('e-0', 'e-') if v > 0 else '0')


def table():
    st = json.loads((OUT/'static.json').read_text())
    cola = json.loads((OUT/'colamod.json').read_text()) if (OUT/'colamod.json').exists() else None
    rows, notes = [], []
    for name in st:
        s = st[name]
        eq = s['equations']
        eqs = '无论文' if eq is None else ('一致（%s）' % _f(max(eq.values()), 0) if max(eq.values()) < 1e-9 else '**不一致**')
        pub = s['published']
        if pub is None:
            pubs = '无'
        else:
            bad = [r for r in pub if not r['ok']]
            pubs = f'{len(pub)-len(bad)}/{len(pub)}' + ('' if not bad else '，**%d 项不符**' % len(bad))
            notes += [f'{ZH[name]}：{r["what"]}，公布 {r["published"]}，引擎 {r["engine"]}。' for r in bad]
        con = s['conservation']
        cons = '不适用' if 'none' in con else '闭合（%s）' % _f(max(con.values()), 0)
        f = OUT/f'numerics-{name}.json'
        if f.exists():
            nm = json.loads(f.read_text())
            loc, share, glob = max(r['local_max'] for r in nm), max(r['local_share'] for r in nm), max(r['global_max'] for r in nm)
            nums = f'{_f(loc)} / {100*share:.1f}% / {_f(glob)}'
            if any(r['failed'] or r['at_state_bound'] for r in nm):
                nums += '（%d 回合失效，%d 次到状态界）' % (sum(r['failed'] for r in nm), sum(r['at_state_bound'] for r in nm))
        else:
            nums = '未跑'
        ext = '—'
        if name == 'column' and cola:
            r = cola['runs']
            ext = '原版 colamod.m：方程 %s，按训练步长 %s' % (_f(max(x['equations_lsoda'] for x in r), 0), _f(max(x['engine_float32'] for x in r)))
        rows.append(f'| {ZH[name]} | {SRC[name]} | {eqs} | {pubs} | {cons} | {nums} | {ext} |')
    head = ['| 装置 | 来源 | 方程对论文 | 公布数字 | 守恒 | 积分误差：单决策最大 / 占位移 / 整回合最大（带宽） | 外部原版代码轨迹（带宽） |', '|---|---|---|---|---|---|---|']
    return '\n'.join(head+rows), notes


def stress_table():
    """The table of the stress test of docs/validation.md from reports/validation/stress.json."""
    st = json.loads((OUT/'stress.json').read_text())
    cell = lambda r, k='guarded': '%d / %s' % (r[k+'_over_one_band'], _f(r[k+'_max']) if r[k+'_max'] < 100 else '%d' % round(r[k+'_max']))       # noqa: E731
    rows = ['| 装置 | 每个阀走满自己的速率限值：超过 1 带宽的决策 / 最大误差 | 一步动 20 % 量程 | 一步动到任意位置 | 任意位置、不带守卫 |', '|---|---|---|---|---|']
    for name, r in st.items():
        if name in ZH:
            rows.append('| %s | %s | %s | %s | %s |' % (ZH[name], cell(r['slew']), cell(r['0.2']), cell(r['1.0']), cell(r['1.0'], 'plain')))
    return '\n'.join(rows)


def columns_table():
    """The designed columns at the training step: integration error over training episodes, and one decision
    with every valve moved by its slew limit; the rows of the task, and every output shown."""
    st = json.loads((OUT/'stress.json').read_text())
    rows = ['| 塔 | 训练回合，任务行：单决策最大 / 整回合最大（带宽） | 训练回合，全部输出：单决策最大（出在） | 阀走满速率限值，任务行：超过 1 带宽 / 最大 | 阀走满速率限值，全部输出 |',
            '|---|---|---|---|---|']
    for name, zh in COLUMNS.items():
        f = OUT/f'numerics-{name}.json'
        if not f.exists() or name not in st:
            continue
        nm, a = json.loads(f.read_text()), st[name]['slew']
        w = max(nm, key=lambda r: r['local_max'])
        rows.append('| %s | %s / %s | %s（%s） | %d / %s | %d / %s |' % (
            zh, _f(max(r['local_max_task'] for r in nm)), _f(max(r['global_max_task'] for r in nm)), _f(w['local_max']), w['worst_output'],
            a['guarded_over_one_band_task'], _f(a['guarded_max_task']), a['guarded_over_one_band'], _f(a['guarded_max'])))
    return '\n'.join(rows)


def _where(v):
    return '低于 1 %' if v < 1 else '高于 99 %' if v > 99 else '%d %%' % round(v)


def _prior_line(pr):
    q, b = pr['pairs'], pr['batch']
    return '先验的逐对响应（%d 回合、种子 %d，%d 对；1 %% / 50 %% / 99 %%）：滞后/时间常数比 %s / %s / %s，时间比 %s / %s / %s。像它的配对：两项都在该回路 2 倍以内的占比。' % (
        b['episodes'], b['seed'], b['pairs'], _f(q['ratio'][0], 3), _f(q['ratio'][2]), _f(q['ratio'][4]), _f(q['spread'][0], 1), _f(q['spread'][2], 1), _f(q['spread'][4], 1))


RZH = dict(dryer='工业干燥器', evaporator='四效蒸发器', glassfurnace='玻璃窑', exchanger='蒸汽换热器', heating='灯加热钢板')


def records_table():
    """The DaISy records of reports/validation/realdata.json against the prior: every paired response, and the
    features of the plant that the prior does not generate."""
    rec = json.loads((OUT/'realdata.json').read_text())['records']
    rows = ['| 记录（配对数） | 留出段 R² | 各配对的滞后/时间常数比（先验分位） | 到 63 % 与到 20 % 的时间比（先验分位） | 先验里像它的配对 | 装置特征在 1–99 % 之外的 |', '|---|---|---|---|---|---|']
    for name, zh in RZH.items():
        r = rec[name]
        if 'pairs' not in r:
            continue
        out = ['%s %s（%s）' % (q, _f(v['value']), _where(v['percentile'])) for q, v in r['features'].items() if not 1 <= v['percentile'] <= 99] if len(r['pairs']) > 1 else []
        fit = [r['fit'][o] for _, o in r['pairing']]              # the outputs paired
        rows.append('| %s（%d） | %s | %s | %s | %s | %s |' % (
            zh, len(fit), '–'.join('%.2f' % v for v in (min(fit), max(fit))) if len(fit) > 1 else '%.2f' % fit[0],
            '、'.join('%s（%s）' % (_f(q['ratio']['value']), _where(q['ratio']['percentile'])) for q in r['pairs']),
            '、'.join('%s（%s）' % (_f(q['spread']['value'], 1), _where(q['spread']['percentile'])) for q in r['pairs']),
            '、'.join('%.1f%%' % (100*q['alike']) for q in r['pairs']), '；'.join(out) or '无'))
    col = json.loads((OUT/'realdata.json').read_text())['steps']
    q = rec['prior']['plants']
    return '\n'.join(rows)+'\n\n'+_prior_line(rec['prior'])+'\n\nManipal 中试塔：相对增益 %s（先验装置的 %s 分位，中位 %s），两回路快慢比 %s（%s 分位，中位 %s）。' % (
        _f(col['relative_gain']), _where(col['percentile']['relative_gain']), _f(q['rga_max'][2]), _f(col['speed'], 1), _where(col['percentile']['speed']), _f(q['speed'][2], 1))


BZH = dict(bestest_hydronic='散热器（锅炉供水温度 → 室温）', bestest_hydronic_heat_pump='热泵地暖（压缩机转速 → 室温）',
           singlezone_commercial_hydronic='商用单区（散热器阀 → 室温）', twozone_apartment_hydronic='两区公寓（两个地暖阀 → 两个室温）')
PZH = dict(peak_heat_day='最冷', typical_heat_day='典型', peak_cool_day='最热', typical_cool_day='典型制冷', mix_day='过渡', test_day='测试')


def boptest_tables():
    """The three tables of the BOPTEST section of docs/validation.md from reports/validation/boptest.json."""
    b = json.loads((OUT/'boptest.json').read_text())
    out = []
    ref = b['reference']
    with_ref = {c: r for c, r in ref.items() if r['values']}
    out.append('对上游参考值：%d 个考题 %d 个时段 %d 个 KPI，最大相对差 %s；没有这类参考文件的考题 %d 个。' % (
        len(with_ref), sum(r['periods'] for r in with_ref.values()), sum(r['values'] for r in with_ref.values()),
        _f(max(r['worst'] for r in with_ref.values()), 0), len(ref)-len(with_ref)))
    st = b['steps']
    rows = ['| 回路 | 决策周期 | 滞后 / 时间常数（决策） | 滞后/时间常数比（先验分位） | 到 63 % 与到 20 % 的时间比（先验分位） | 先验里像它的配对 | 第 240 个决策处是 10 天值的 | 配对相对增益 |',
            '|---|---|---|---|---|---|---|---|']
    for c, zh in BZH.items():
        x = st[c]
        f, pr = x['features'], x['pairs']
        rows.append('| %s | %d min | %s / %s | %s（%s） | %s（%s） | %s | %s | %s |' % (
            zh, x['dt']/60, _f(pr[0]['theta']['value'], 0), _f(pr[0]['tau']['value'], 0), _f(pr[0]['ratio']['value'], 3), '、'.join(_where(q['ratio']['percentile']) for q in pr),
            '、'.join(_f(q['spread']['value'], 1) for q in pr), '、'.join(_where(q['spread']['percentile']) for q in pr), '、'.join('%.2f%%' % (100*q['alike']) for q in pr),
            '、'.join('%d%%' % round(100*v) for v in x['reached']), _f(f['rga_max']['value']) if 'rga_max' in f else '—'))
    out.append('\n'.join(rows)+'\n\n'+_prior_line(st['prior']))
    ex = {(r['case'], r['period'], r['controller']): r for r in b['exam'] if 'kpi' in r}
    rows = ['| 回路 | 时段 | 不舒适度（K·h）：自带 / 公式 PI / 网络 / 网络半带宽 | 能耗（kWh/m²）：自带 / 公式 PI / 网络 | 室温平均高出设定值（K）：自带 / 网络 | 执行器停在下限的时间：自带 / 网络 |',
            '|---|---|---|---|---|---|']
    kinds = ('own', 'formula_pi', 'network', 'network_half_band')
    num = lambda v, d=2: '—' if v is None else _f(v, d)      # noqa: E731
    for c, zh in BZH.items():
        for p in ('peak_heat_day', 'typical_heat_day'):
            g = lambda k, q: ex[(c, p, k)]['kpi']['constant'][q] if (c, p, k) in ex else None      # noqa: E731
            per = lambda k, q, f: '、'.join(f(v) for v in ex[(c, p, k)][q]) if (c, p, k) in ex else '—'      # noqa: E731
            rows.append('| %s | %s | %s | %s | %s | %s |' % (
                zh, PZH[p], ' / '.join(num(g(k, 'tdis_tot'), 1) for k in kinds), ' / '.join(num(g(k, 'ener_tot')) for k in kinds[:3]),
                ' / '.join(per(k, 'mean_above_setpoint', lambda v: '%.1f' % v) for k in ('own', 'network')),
                ' / '.join(per(k, 'at_low_limit', lambda v: '%d%%' % round(100*v)) for k in ('own', 'network'))))
    out.append('\n'.join(rows))
    rows = ['| 考题 | 时段：不舒适度（K·h/区） / 能耗（kWh/m²） / 费用（€/m²，恒价） |', '|---|---|']
    base = {}
    for r in b['baseline']:
        if 'kpi' in r:
            k = r['kpi']['constant']
            base.setdefault(r['case'], []).append('%s %s / %s / %s' % (PZH[r['period']], _f(k['tdis_tot'], 1), _f(k['ener_tot']), _f(k['cost_tot'])))
    for c, v in base.items():
        rows.append('| `%s` | %s |' % (c, '；'.join(v)))
    out.append('\n'.join(rows))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('static', 'numerics', 'stress', 'colamod', 'realdata', 'boptest', 'prior', 'table'))
    ap.add_argument('args', nargs='*')
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--big', action='store_true')
    ap.add_argument('--ctl', default='', help='MODULE:FACTORY of a packet controller (aio.sim.plugin)')
    ap.add_argument('--arg', default=None, help='argument passed to FACTORY (e.g. weights path)')
    a = ap.parse_args()
    if a.cmd == 'static':
        static()
    elif a.cmd == 'numerics':
        numerics(a.args, a.n, a.jobs)
    elif a.cmd == 'stress':
        stress(a.args)
    elif a.cmd == 'colamod':
        colamod(a.args[0])
    elif a.cmd == 'realdata':
        realdata(a.args[0])
    elif a.cmd == 'boptest':
        if len(a.args) > 2 and not a.ctl:
            ap.error('boptest weights must now be passed via --ctl MODULE:FACTORY --arg PATH')
        boptest(*a.args[:2], ctl=a.ctl or None, arg=a.arg, big=a.big)
    elif a.cmd == 'prior':
        prior()
    else:
        t, notes = table()
        print(t)
        print('\n'.join(notes))
        if (OUT/'stress.json').is_file():
            print('\n'+stress_table()+'\n')
        if (OUT/'realdata.json').is_file() and 'prior' in json.loads((OUT/'realdata.json').read_text()).get('records', {}):
            print(records_table())
        if (OUT/'boptest.json').is_file():
            print('\n\n'.join(boptest_tables()))
        if (OUT/'refprops.json').is_file():                        # written by `python -m aio.sim.validate.refprops`, `... columns` (they need thermo and CoolProp)
            from aio.sim.validate import refprops
            print('\n'+refprops.table())
        if (OUT/'columns.json').is_file():
            from aio.sim.validate import columns
            print('\n'+columns.table())
        if (OUT/'numerics-btcol.json').is_file():
            print('\n'+columns_table())

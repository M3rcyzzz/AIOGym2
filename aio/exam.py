"""A model on one of the fixed exams of a long stay (aio.sim.stay.EXAMS), beside the references, with its tail.

    python -m aio.exam --exam stock --ckpt RUN/SEED/last.pt [--n 64 --windows 8 --out exam.json]

The teacher lives on the plants, the formula PI and the commissioning-model DMC take every window over from it; the
model and the two references that read the instruments alone each live their own stay. Points as the reference tables
have them: per plant summed over the windows, 0 the MVs left where they are, 100 the best of hold, PI, DMC and teacher
on that plant. Beside the points of the batch, each controller's tail (aio.scale.tail): the share of the plants it
leaves worse than untouched and its points on the tenth of the plants it does worst on. `levels`: the plants with a
level among their rows, and the rest.
"""
import argparse
import json

import torch

from aio import net as N, scale as SC
from aio.sim import stay as ST

BASE = ('hold', 'pi', 'dmc', 'teacher')


def read(name, ckpt=None, n=ST.EXAM_N, windows=ST.EXAM_WINDOWS, device='cpu', say=None):
    progs = list(ST.exam(name, n, ST.EXAM_SEED, windows, device=device, gen_device='cpu'))
    extra = dict(dmc_meter=ST.REFS['dmc_meter'](), teacher_meter=ST.REFS['teacher_meter']())
    if ckpt:
        extra['aio'] = N.Policy(N.load(ckpt, torch.device(device))[0], stay=True)
    out = ST.references(n, 0, windows, BASE, progs=progs, extra=extra, say=say)
    r0 = ST.EXAMS[name].get('read', 0)
    per = {q: v[r0:].sum(0).tolist() for q, v in out['score'].items()}
    best = SC.envelope({q: per[q] for q in BASE})
    lev = ((progs[0].c['integ'] > 0).sum(1) > 0).tolist()
    res = dict(exam=name, key=ST.key(progs), n=n, windows=windows, read_from=r0, per=per, levels=lev, failed={q: int(v.sum()) for q, v in out['failed'].items()}, table={})
    for part, m in (('all', [True]*n), ('levels', lev), ('rest', [not x for x in lev])):
        pick = lambda v: [x for x, k in zip(v, m) if k]           # noqa: E731
        if any(m):
            res['table'][part] = {q: dict(points=SC.points(pick(v), pick(per['hold']), pick(best)), **SC.tail(pick(v), pick(per['hold']), pick(best))) for q, v in per.items() if q != 'hold'}
            res['table'][part]['plants'] = sum(m)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exam', required=True, help=', '.join(ST.EXAMS))
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--n', type=int, default=ST.EXAM_N)
    ap.add_argument('--windows', type=int, default=ST.EXAM_WINDOWS)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    res = read(a.exam, a.ckpt, a.n, a.windows, a.device, say=lambda w, s, t: print(f'window {w}: '+'  '.join(f'{q} {v:.0f}' for q, v in s.items())+f'  ({t:.0f} s)', flush=True))
    print(f"{a.exam}  key {res['key']}  {a.n} plants x {a.windows} windows   points / worse than untouched / points on the worst tenth")
    for part, t in res['table'].items():
        print(f"{part:7s}{t['plants']:4d}  "+'   '.join(f"{q} {v['points']:.1f} / {v['worse']:.0%} / {v['low']:.0f}" for q, v in t.items() if q != 'plants'))
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(res, f)


if __name__ == '__main__':
    main()

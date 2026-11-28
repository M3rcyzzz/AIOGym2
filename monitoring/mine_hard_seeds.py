"""Zero-CU causal-regret hard-seed miner for `--hard-seeds` curriculum training.

Loads a checkpoint via `aio.net.load`, rolls out `N` fresh `12x5` plants starting at `--start-seed`
(default `10000000`) alongside `CT.Teacher('late')` and `CT.Hold()`, ranks candidate seeds by causal
regret:

    regret = (model - teacher) / max(hold - teacher, 100.0)

and atomically writes the top `--top-frac` (default 0.25) seeds to JSON (`{"seeds": [...], "meta": {...}}`).

By default (`--mode batch`), evaluates candidate batch seeds (`start_seed + k`, each rolled out with
`C.build(batch, seed)`) matching `aio.train --hard-seeds`'s `C.build(envs, plant_seed, event_seed=...)`
contract (`PR.Rng(seed)` hashes `(seed, i)` independently of `n`). Also supports `--mode plant` for
individual single-plant seeds via `VectorRng`.

  python -m monitoring.mine_hard_seeds --ckpt ~/colab-runs/sq-pure-10/run/eval/best_eval.pt \\
      --out ~/colab-runs/sq-pure-10/hard_seeds.json --n 512 --batch 32 --start-seed 10000000 --top-frac 0.25
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from aio import net as N
from aio.sim import control as CT, core as C, prior as PR


class VectorRng(PR.Rng):
    """Vectorized per-plant Rng whose row `i` is bit-identical to `PR.Rng(seeds[i]).bind(1)`."""

    def __init__(self, seeds, device='cpu'):
        dev = torch.device(device)
        if isinstance(seeds, torch.Tensor):
            vec = seeds.to(device=dev, dtype=torch.int64).reshape(-1)
        else:
            vec = torch.tensor([PR._s64(int(s)) for s in seeds], dtype=torch.int64, device=dev)
        super().__init__(int(vec[0].item()) if vec.numel() else 0, device=dev)
        self.seed_vec = vec

    def sub(self, idx, repeat=1):
        v = VectorRng.__new__(VectorRng)
        v.dev, v.seed, v.seed_vec, v.ctr, v.forks, v.live = (
            self.dev, self.seed, self.seed_vec, self.ctr, 0, None,
        )
        v.rows = self.rows[idx].repeat_interleave(repeat)
        lane = self.lane[idx].repeat_interleave(repeat)
        v.lane = lane * repeat + torch.arange(repeat, device=self.dev).repeat(idx.numel()) if repeat > 1 else lane
        return v

    def fork(self, device=None):
        self.forks += 1
        d = torch.device(device) if device is not None else self.dev
        delta = PR._mix(torch.tensor(-self.forks, dtype=torch.int64, device=self.dev))
        forked_vec = PR._mix(self.seed_vec + delta).to(d)
        return VectorRng(forked_vec, device=d)

    def _next(self, shape):
        m = shape[0]
        self.bind(m)
        k = math.prod(shape[1:])
        key = PR._mix(PR._mix(PR._mix(self.seed_vec[self.rows])) + self.lane)
        j = self.ctr[self.rows][:, None] + torch.arange(k, device=self.dev)[None]
        z = PR._mix(key[:, None] + PR._mix(j))
        step = torch.full_like(self.rows, k) if self.live is None else k * self.live.long()
        self.ctr[self.rows] = self.ctr[self.rows] + step
        return (PR._shr(z, 11).to(PR.F) * (1. / (1 << 53))).view(*shape)


def build_from_seeds(seeds, device='cpu', gen_device='cpu', dims='12x5', teacher=True):
    """Build a batched Program whose episode `i` has the exact plant dynamics of `C.build(1, seeds[i])`."""
    seeds = [int(s) for s in seeds]
    if not seeds:
        raise ValueError('seeds must be non-empty')
    rng = VectorRng(seeds, device=gen_device)
    return C.Program(rng, len(seeds), device=device, seed=seeds[0], teacher=teacher, dims=dims)


def rank_regret(seeds, model_scores, teacher_scores, hold_scores, *, top_frac=0.25, min_denom=100.0):
    """Rank plant seeds descending by `(model - teacher) / max(hold - teacher, min_denom)`."""
    if not (len(seeds) == len(model_scores) == len(teacher_scores) == len(hold_scores)):
        raise ValueError('seeds and score lists must have identical length')
    if not seeds:
        return [], [], {}
    rows = []
    for s, m, t, h in zip(seeds, model_scores, teacher_scores, hold_scores):
        mf, tf, hf = float(m), float(t), float(h)
        denom = max(hf - tf, float(min_denom))
        regret = (mf - tf) / denom
        if not math.isfinite(regret):
            regret = float('-inf')
        rows.append(dict(seed=int(s), regret=regret, model=mf, teacher=tf, hold=hf))
    rows.sort(key=lambda r: (-r['regret'], r['seed']))
    k = max(1, min(len(rows), int(round(len(rows) * float(top_frac)))))
    top = rows[:k]
    finite_all = [r['regret'] for r in rows if math.isfinite(r['regret'])]
    finite_top = [r['regret'] for r in top if math.isfinite(r['regret'])]
    stats = dict(
        mean_regret_all=round(sum(finite_all) / len(finite_all), 4) if finite_all else None,
        mean_regret_top=round(sum(finite_top) / len(finite_top), 4) if finite_top else None,
        min_regret_top=round(min(finite_top), 4) if finite_top else None,
        max_regret_top=round(max(finite_top), 4) if finite_top else None,
    )
    details = [
        dict(seed=r['seed'], regret=round(r['regret'], 4),
             model=round(r['model'], 2), teacher=round(r['teacher'], 2), hold=round(r['hold'], 2))
        for r in top if math.isfinite(r['regret'])
    ]
    return [r['seed'] for r in top], details, stats


@torch.no_grad()
def mine_hard_seeds(ckpt_path, out_path=None, *, n=512, batch=32, start_seed=10_000_000,
                    top_frac=0.25, dims='12x5', device='cpu', steps=C.HORIZON,
                    min_denom=100.0, mode='batch', ctl=None):
    """Roll out `n` fresh plants and select top `--top-frac` hardest seeds by causal regret.

    mode='batch' (default): mines candidate batch seeds (`C.build(batch, start_seed + k)`) directly
    compatible with `aio.train --hard-seeds` (`C.build(envs, plant_seed, event_seed=...)`).
    mode='plant': mines individual plant seeds (`VectorRng` bit-identical to `C.build(1, seed)`).
    """
    if mode not in ('batch', 'plant'):
        raise ValueError(f'mode must be "batch" or "plant", got {mode!r}')
    t0 = time.time()
    dev = torch.device(device)
    if ctl is None:
        net, _ = N.load(ckpt_path, dev)
        policy = N.Policy(net)
    else:
        policy = ctl
    teacher = CT.Teacher('late')
    hold = CT.Hold()

    eff_batch = max(1, min(int(batch), int(n)))
    if mode == 'batch':
        n_batches = max(1, int(n) // eff_batch)
        cand_seeds = [int(start_seed) + b for b in range(n_batches)]
        m_scores, t_scores, h_scores = [], [], []
        batch_regrets = []
        for s in cand_seeds:
            prog = C.build(eff_batch, s, device=dev, gen_device='cpu', dims=dims, teacher=True)
            rm = CT.rollout(prog, policy, steps=steps)['score'].detach().cpu().tolist()
            rt = CT.rollout(prog, teacher, steps=steps)['score'].detach().cpu().tolist()
            rh = CT.rollout(prog, hold, steps=steps)['score'].detach().cpu().tolist()
            per_r = [(float(m) - float(t)) / max(float(h) - float(t), float(min_denom)) for m, t, h in zip(rm, rt, rh)]
            mean_r = sum(per_r) / len(per_r)
            batch_regrets.append((s, mean_r, sum(rm) / len(rm), sum(rt) / len(rt), sum(rh) / len(rh)))
            m_scores.append(sum(rm) / len(rm))
            t_scores.append(sum(rt) / len(rt))
            h_scores.append(sum(rh) / len(rh))
        # Rank candidate batch seeds by mean per-plant causal regret
        batch_regrets.sort(key=lambda r: (-r[1], r[0]))
        k = max(1, min(len(batch_regrets), int(round(len(batch_regrets) * float(top_frac)))))
        top = batch_regrets[:k]
        top_seeds = [int(r[0]) for r in top]
        all_r = [r[1] for r in batch_regrets if math.isfinite(r[1])]
        top_r = [r[1] for r in top if math.isfinite(r[1])]
        stats = dict(
            mean_regret_all=round(sum(all_r) / len(all_r), 4) if all_r else None,
            mean_regret_top=round(sum(top_r) / len(top_r), 4) if top_r else None,
            min_regret_top=round(min(top_r), 4) if top_r else None,
            max_regret_top=round(max(top_r), 4) if top_r else None,
        )
        details = [
            dict(seed=int(r[0]), regret=round(r[1], 4),
                 model=round(r[2], 2), teacher=round(r[3], 2), hold=round(r[4], 2))
            for r in top if math.isfinite(r[1])
        ]
        total_plants = n_batches * eff_batch
    else:
        all_seeds = [int(start_seed) + i for i in range(int(n))]
        m_scores, t_scores, h_scores = [], [], []
        for i in range(0, len(all_seeds), eff_batch):
            chunk = all_seeds[i:i + eff_batch]
            prog = build_from_seeds(chunk, device=dev, gen_device='cpu', dims=dims, teacher=True)
            rm = CT.rollout(prog, policy, steps=steps)
            rt = CT.rollout(prog, teacher, steps=steps)
            rh = CT.rollout(prog, hold, steps=steps)
            m_scores.extend(rm['score'].detach().cpu().tolist())
            t_scores.extend(rt['score'].detach().cpu().tolist())
            h_scores.extend(rh['score'].detach().cpu().tolist())
        top_seeds, details, stats = rank_regret(
            all_seeds, m_scores, t_scores, h_scores, top_frac=top_frac, min_denom=min_denom,
        )
        n_batches = None
        total_plants = len(all_seeds)

    payload = {
        'seeds': top_seeds,
        'meta': {
            'ckpt': str(ckpt_path) if ckpt_path is not None else None,
            'mode': mode,
            'n': int(total_plants),
            'batch_size': int(eff_batch),
            **({'n_batches': int(n_batches)} if n_batches is not None else {}),
            'selected': len(top_seeds),
            'start_seed': int(start_seed),
            'top_frac': float(top_frac),
            'dims': str(dims),
            'steps': int(steps),
            'min_denom': float(min_denom),
            **stats,
            'seconds': round(time.time() - t0, 2),
            'created_at': round(time.time(), 3),
            'top_details': details,
        },
    }
    if out_path is not None:
        p = Path(out_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.parent / f'.{p.name}.tmp.{os.getpid()}'
        tmp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=1) + '\n', encoding='utf-8')
        os.replace(tmp, p)
    return payload


def build_parser():
    ap = argparse.ArgumentParser(description='Zero-CU regret-based hard-seed miner for --hard-seeds.')
    ap.add_argument('--ckpt', type=Path, required=True, help='Model checkpoint (.pt) loaded via aio.net.load')
    ap.add_argument('--out', type=Path, required=True, help='Output JSON file ({"seeds": [...], "meta": {...}})')
    ap.add_argument('--n', type=int, default=512, help='Total fresh plants to evaluate')
    ap.add_argument('--batch', type=int, default=32, help='Plants per candidate batch (default 32)')
    ap.add_argument('--mode', choices=('batch', 'plant'), default='batch',
                    help='batch: emit batch seeds for C.build(envs, seed) (default); plant: emit per-plant seeds')
    ap.add_argument('--start-seed', type=int, default=10_000_000, help='Starting seed (disjoint from validation)')
    ap.add_argument('--top-frac', type=float, default=0.25, help='Fraction of highest-regret seeds to emit')
    ap.add_argument('--dims', default='12x5', help='Plant size preset (default 12x5)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--steps', type=int, default=C.HORIZON, help='Rollout horizon in decisions (default 720)')
    ap.add_argument('--min-denom', type=float, default=100.0, help='Minimum (hold - teacher) denominator floor')
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    res = mine_hard_seeds(
        a.ckpt, a.out, n=a.n, batch=a.batch, start_seed=a.start_seed,
        top_frac=a.top_frac, dims=a.dims, device=a.device, steps=a.steps,
        min_denom=a.min_denom, mode=a.mode,
    )
    meta = res['meta']
    print(json.dumps(dict(
        out=str(a.out), n=meta['n'], selected=meta['selected'],
        mean_regret_all=meta['mean_regret_all'], mean_regret_top=meta['mean_regret_top'],
        seconds=meta['seconds'],
    )), flush=True)


if __name__ == '__main__':
    main()

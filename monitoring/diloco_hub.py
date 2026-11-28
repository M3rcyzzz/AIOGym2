"""Elastic Equal-H MuLoCo coordinator hub (Mac CPU FP32 master, <2.2 GB RAM for 131M SeqAIO).

Coordinates 1 or 2 local GPU workers (`guomao` + `baolong`) over Tailscale HTTP without stalling
when a worker detaches (`--min-quorum 1 --target-quorum 2 --grace-s 120`):
  - Wire protocol:
      GET  /status -> JSON telemetry
      POST /sync   -> binary `torch.save` BytesIO request & response
  - Compression: blockwise symmetric INT8 (group g=128) or BF16 pseudo-gradients
  - Outer optimizer: FP32 Nesterov SGD (--outer-lr 0.7 --outer-mom 0.6, wd=0) on MVUE
    sample-weighted actor deltas, anomaly norm gate (norm <= 3 * ema_norm after warmup),
    plain sample-weighted FedAvg on PrivCritic and rel_probe (no Nesterov), CPU FP32 Outer EMA
    (d_out = min(0.55, (1+t)/(5+t))), and Hub-managed WSD inner LR schedule.
  - Persistence: atomic `hub_last.pt` + 268 MB Outer-EMA eval snapshot `eval/outer_r{round}.pt`
    every outer step so `monitoring/eval_worker.py` evaluates every merged checkpoint automatically.

  python -m monitoring.diloco_hub --init ~/colab-runs/sq-pure-10/run/last.pt --out ~/diloco-hub \\
      --host 0.0.0.0 --port 8770 --outer-lr 0.7 --outer-mom 0.6 --min-quorum 1 --target-quorum 2 --grace-s 120
"""
import argparse
import io
import json
import math
import os
import pathlib
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch


def safe_torch_load(path_or_buf, map_location='cpu'):
    """Load PyTorch checkpoint/payload safely across Python 3.13->3.12 (`pathlib._local`) and Windows/POSIX."""
    sys.modules.setdefault('pathlib._local', pathlib)
    saved = pathlib.WindowsPath
    if os.name != 'nt':
        pathlib.WindowsPath = pathlib.PureWindowsPath
    try:
        return torch.load(path_or_buf, map_location=map_location, weights_only=False)
    finally:
        pathlib.WindowsPath = saved


def sanitize_config(cfg):
    """Convert any Path objects inside checkpoint config to plain strings."""
    if isinstance(cfg, Path):
        return str(cfg)
    if isinstance(cfg, dict):
        return {k: sanitize_config(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [sanitize_config(v) for v in cfg]
    if isinstance(cfg, tuple):
        return tuple(sanitize_config(v) for v in cfg)
    return cfg


def pack_int8(tensor, g=128):
    """Blockwise symmetric INT8 quantization (group size g=128) matching trainer-builder's wire contract."""
    flat = tensor.detach().float().cpu().reshape(-1)
    numel = flat.numel()
    pad = (-numel) % g
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blk = flat.view(-1, g)
    s = blk.abs().amax(-1).clamp_min(1e-8).to(torch.float16)
    q = (blk / s.float()[:, None] * 127.0).round().clamp(-127, 127).to(torch.int8)
    return {'q': q, 's': s, 'shape': tuple(tensor.shape), 'numel': int(numel), 'g': int(g)}


def unpack_int8(entry):
    """Exact inverse of `pack_int8`: returns FP32 tensor of original shape."""
    q = entry['q']
    s = entry['s']
    numel = int(entry['numel'])
    shape = tuple(entry['shape'])
    return (q.float() * (s.float()[:, None] / 127.0)).reshape(-1)[:numel].reshape(shape)


def unpack_delta_dict(packed, compress='int8'):
    """Unpack `actor_delta` dict (INT8 packed entries or raw BF16/FP32 tensors) -> FP32 CPU dict."""
    out = {}
    for k, v in packed.items():
        if isinstance(v, dict) and 'q' in v and 's' in v:
            out[k] = unpack_int8(v).cpu()
        elif isinstance(v, torch.Tensor):
            out[k] = v.detach().float().cpu() if v.is_floating_point() else v.detach().cpu()
        else:
            raise ValueError(f'unsupported delta entry for {k!r}: {type(v).__name__}')
    return out


def pack_actor_broadcast(actor_fp32):
    """Pack FP32 actor state dict: BF16 for >=2D float tensors, FP32 for <=1D float tensors."""
    out = {}
    for k, v in actor_fp32.items():
        t = v.detach().cpu()
        if t.is_floating_point():
            out[k] = t.to(torch.bfloat16) if t.ndim >= 2 else t.to(torch.float32)
        else:
            out[k] = t
    return out


def wsd_lr(global_steps, base_lr, warmup_steps=0, stable_steps=100_000, decay_steps=0, min_lr_frac=0.1):
    """Hub-managed Warmup-Stable-Decay inner learning rate schedule."""
    s = max(0, int(global_steps))
    if warmup_steps > 0 and s < warmup_steps:
        return float(base_lr) * min(1.0, float(s + 1) / float(warmup_steps))
    if decay_steps <= 0 or s <= warmup_steps + stable_steps:
        return float(base_lr)
    p = min(1.0, float(s - warmup_steps - stable_steps) / float(decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * p))
    return float(base_lr) * (float(min_lr_frac) + (1.0 - float(min_lr_frac)) * cosine)


def _tensor_dict_to_fp32(sd):
    if sd is None:
        return None
    out = {}
    for k, v in sd.items():
        t = v.detach().cpu()
        out[k] = t.float().clone() if t.is_floating_point() else t.clone()
    return out


def _all_finite_and_norm(delta_sd):
    sq = 0.0
    for v in delta_sd.values():
        if not v.is_floating_point():
            continue
        if not bool(torch.isfinite(v).all()):
            return False, float('inf')
        sq += float(v.double().pow(2).sum().item())
    return True, math.sqrt(sq)


def _atomic_torch_save(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.parent / f'.{p.name}.tmp.{os.getpid()}.{threading.get_ident()}'
    torch.save(obj, tmp)
    os.replace(tmp, p)


def _serialize_torch(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


class DiLoCoHub:
    """Thread-safe Elastic Equal-H MuLoCo coordinator holding FP32 master weights on CPU."""

    def __init__(self, actor_sd, *, critic_sd=None, rel_probe_sd=None, pairs=None, config=None,
                 out_dir=None, outer_lr=0.7, outer_mom=0.6, outer_ema_max=0.55,
                 base_lr=6e-5, warmup_steps=0, stable_steps=100_000, decay_steps=0, min_lr_frac=0.1,
                 min_quorum=1, target_quorum=2, grace_s=120.0,
                 norm_mult=3.0, norm_warmup=2, keep_eval_snaps=12):
        self.actor = _tensor_dict_to_fp32(actor_sd)
        self.outer_buf = {k: torch.zeros_like(v) for k, v in self.actor.items() if v.is_floating_point()}
        self.outer_ema = {k: v.clone() for k, v in self.actor.items()}
        self.critic = _tensor_dict_to_fp32(critic_sd)
        self.rel_probe = _tensor_dict_to_fp32(rel_probe_sd)
        self.pairs = {k: v.detach().cpu().clone() for k, v in pairs.items()} if pairs is not None else None
        self.config = sanitize_config(config or {})

        self.out_dir = Path(out_dir).expanduser() if out_dir is not None else None
        self.outer_lr = float(outer_lr)
        self.outer_mom = float(outer_mom)
        self.outer_ema_max = float(outer_ema_max)
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.stable_steps = int(stable_steps)
        self.decay_steps = int(decay_steps)
        self.min_lr_frac = float(min_lr_frac)
        self.min_quorum = max(1, int(min_quorum))
        self.target_quorum = max(self.min_quorum, int(target_quorum))
        self.grace_s = float(grace_s)
        self.norm_mult = float(norm_mult)
        self.norm_warmup = int(norm_warmup)
        self.keep_eval_snaps = int(keep_eval_snaps)

        self.round = 0
        self.global_steps = 0
        self.global_env_steps = 0
        self.ema_norm = None
        self.norm_updates = 0
        self.rejected_count = 0
        self.workers = {}
        self.started_at = time.time()
        self.last_outer_step_at = None

        self.cond = threading.Condition()
        self._pending = {}
        self._first_arrival = None
        self._round_bytes = {}
        self._closed = False

    @classmethod
    def from_checkpoint(cls, ckpt_path, *, init_from_raw=False, **kwargs):
        ck = safe_torch_load(ckpt_path, map_location='cpu')
        if 'model' in ck and isinstance(ck['model'], dict):
            raw = ck.get('raw') if isinstance(ck.get('raw'), dict) else {}
            actor = raw['actor'] if (init_from_raw and 'actor' in raw) else ck['model']['actor']
            critic = ck['model'].get('critic')
            pairs = ck['model'].get('pairs')
        elif 'actor' in ck:
            actor = ck['actor']
            critic = ck.get('critic')
            pairs = ck.get('pairs')
        else:
            raise ValueError(f'unrecognized checkpoint structure in {ckpt_path}')
        return cls(
            actor,
            critic_sd=critic,
            rel_probe_sd=ck.get('rel_probe'),
            pairs=pairs,
            config=ck.get('config'),
            **kwargs,
        )

    @classmethod
    def from_hub_state(cls, hub_pt, **overrides):
        st = safe_torch_load(hub_pt, map_location='cpu')
        hp = dict(st.get('hparams') or {})
        hp.update({k: v for k, v in overrides.items() if v is not None})
        hub = cls(
            st['actor'],
            critic_sd=st.get('critic'),
            rel_probe_sd=st.get('rel_probe'),
            pairs=st.get('pairs'),
            config=st.get('config'),
            **hp,
        )
        hub.outer_buf = _tensor_dict_to_fp32(st['outer_buf'])
        hub.outer_ema = _tensor_dict_to_fp32(st['outer_ema'])
        hub.round = int(st.get('round', 0))
        hub.global_steps = int(st.get('global_steps', 0))
        hub.global_env_steps = int(st.get('global_env_steps', 0))
        hub.ema_norm = st.get('ema_norm')
        hub.norm_updates = int(st.get('norm_updates', 0))
        hub.rejected_count = int(st.get('rejected_count', 0))
        hub.workers = dict(st.get('workers') or {})
        hub.started_at = float(st.get('started_at', time.time()))
        hub.last_outer_step_at = st.get('last_outer_step_at')
        return hub

    def current_inner_lr(self):
        return wsd_lr(
            self.global_steps, self.base_lr,
            warmup_steps=self.warmup_steps, stable_steps=self.stable_steps,
            decay_steps=self.decay_steps, min_lr_frac=self.min_lr_frac,
        )

    def current_outer_ema_decay(self, t=None):
        step = self.round if t is None else int(t)
        return min(self.outer_ema_max, float(1 + step) / float(5 + step))

    def _response_dict_locked(self, accepted):
        return {
            'round': int(self.round),
            'global_steps': int(self.global_steps),
            'inner_lr': float(self.current_inner_lr()),
            'accepted': bool(accepted),
            'actor': pack_actor_broadcast(self.actor),
            'critic': ({k: v.detach().cpu().clone() for k, v in self.critic.items()} if self.critic is not None else None),
            'rel_probe': ({k: v.detach().cpu().clone() for k, v in self.rel_probe.items()} if self.rel_probe is not None else None),
        }

    def status_dict(self):
        with self.cond:
            return {
                'round': int(self.round),
                'global_steps': int(self.global_steps),
                'global_env_steps': int(self.global_env_steps),
                'inner_lr': float(self.current_inner_lr()),
                'outer_lr': self.outer_lr,
                'outer_mom': self.outer_mom,
                'ema_norm': round(self.ema_norm, 6) if self.ema_norm is not None else None,
                'norm_updates': int(self.norm_updates),
                'outer_ema_decay': round(self.current_outer_ema_decay(), 4),
                'min_quorum': self.min_quorum,
                'target_quorum': self.target_quorum,
                'grace_s': self.grace_s,
                'pending_workers': sorted(self._pending.keys()),
                'workers': {k: dict(v) for k, v in self.workers.items()},
                'rejected_count': int(self.rejected_count),
                'last_outer_step_at': self.last_outer_step_at,
            }

    def _check_norm_gate_locked(self, norm):
        if not math.isfinite(norm) or norm < 0.0:
            return False, 'non-finite norm'
        if self.norm_updates >= self.norm_warmup and self.ema_norm is not None and self.ema_norm > 0.0:
            if norm > self.norm_mult * self.ema_norm:
                return False, f'norm {norm:.4g} > {self.norm_mult:.1f}x ema_norm {self.ema_norm:.4g}'
        return True, None

    def _apply_outer_step_locked(self):
        subs = list(self._pending.values())
        completed_round = self.round

        # 1. MVUE sample-weighted weights
        total_env = sum(max(0, s['env_steps']) for s in subs)
        total_steps = sum(max(0, s['steps']) for s in subs)
        if total_env > 0:
            weights = [max(0, s['env_steps']) / float(total_env) for s in subs]
        elif total_steps > 0:
            weights = [max(0, s['steps']) / float(total_steps) for s in subs]
        else:
            weights = [1.0 / len(subs)] * len(subs)

        # 2. Sample-weighted pseudo-gradient average + FP32 Outer Nesterov SGD (wd=0)
        for k, param in self.actor.items():
            if not param.is_floating_point():
                continue
            avg_d = torch.zeros_like(param, dtype=torch.float32)
            for sub, w in zip(subs, weights):
                d_k = sub['delta'].get(k)
                if d_k is not None:
                    avg_d.add_(d_k, alpha=w)
            buf = self.outer_buf[k]
            buf.mul_(self.outer_mom).add_(avg_d)
            step = avg_d.add(buf, alpha=self.outer_mom)
            param.sub_(step, alpha=self.outer_lr)

        # 3. Plain sample-weighted FedAvg on PrivCritic (no Nesterov)
        crit_subs = [(s, max(0, s['env_steps']), max(0, s['steps'])) for s in subs if s['critic'] is not None]
        if crit_subs:
            c_env = sum(e for _, e, _ in crit_subs)
            c_st = sum(st for _, _, st in crit_subs)
            if c_env > 0:
                cw = [e / float(c_env) for _, e, _ in crit_subs]
            elif c_st > 0:
                cw = [st / float(c_st) for _, _, st in crit_subs]
            else:
                cw = [1.0 / len(crit_subs)] * len(crit_subs)
            keys = list(crit_subs[0][0]['critic'].keys())
            new_critic = {}
            for k in keys:
                first = crit_subs[0][0]['critic'][k]
                if first.is_floating_point():
                    acc = torch.zeros_like(first, dtype=torch.float32)
                    for (sub, _, _), w in zip(crit_subs, cw):
                        if k in sub['critic']:
                            acc.add_(sub['critic'][k].float(), alpha=w)
                    new_critic[k] = acc
                else:
                    # Non-float buffers taken from highest-weight worker
                    best_i = max(range(len(crit_subs)), key=lambda i: cw[i])
                    new_critic[k] = crit_subs[best_i][0]['critic'][k].detach().cpu().clone()
            self.critic = new_critic

        # 4. Plain sample-weighted FedAvg on rel_probe (no Nesterov)
        probe_subs = [(s, max(0, s['env_steps']), max(0, s['steps'])) for s in subs if s['rel_probe'] is not None]
        if probe_subs:
            p_env = sum(e for _, e, _ in probe_subs)
            p_st = sum(st for _, _, st in probe_subs)
            if p_env > 0:
                pw = [e / float(p_env) for _, e, _ in probe_subs]
            elif p_st > 0:
                pw = [st / float(p_st) for _, _, st in probe_subs]
            else:
                pw = [1.0 / len(probe_subs)] * len(probe_subs)
            keys = list(probe_subs[0][0]['rel_probe'].keys())
            new_probe = {}
            for k in keys:
                first = probe_subs[0][0]['rel_probe'][k]
                if first.is_floating_point():
                    acc = torch.zeros_like(first, dtype=torch.float32)
                    for (sub, _, _), w in zip(probe_subs, pw):
                        if k in sub['rel_probe']:
                            acc.add_(sub['rel_probe'][k].float(), alpha=w)
                    new_probe[k] = acc
                else:
                    best_i = max(range(len(probe_subs)), key=lambda i: pw[i])
                    new_probe[k] = probe_subs[best_i][0]['rel_probe'][k].detach().cpu().clone()
            self.rel_probe = new_probe

        # 5. Advance counters, Outer EMA, and anomaly-gate norm EMA
        self.round += 1
        self.global_steps += sum(max(0, s['steps']) for s in subs)
        self.global_env_steps += sum(max(0, s['env_steps']) for s in subs)
        d_out = self.current_outer_ema_decay(self.round)
        for k, param in self.actor.items():
            if param.is_floating_point():
                self.outer_ema[k].mul_(d_out).add_(param, alpha=1.0 - d_out)
            else:
                self.outer_ema[k] = param.clone()

        avg_norm = sum(w * s['norm'] for s, w in zip(subs, weights))
        if self.ema_norm is None:
            self.ema_norm = avg_norm
        else:
            self.ema_norm = 0.8 * self.ema_norm + 0.2 * avg_norm
        self.norm_updates += 1
        self.last_outer_step_at = round(time.time(), 3)

        worker_weights = {}
        for sub, w in zip(subs, weights):
            wn = sub['worker']
            worker_weights[wn] = round(w, 4)
            rec = self.workers.setdefault(wn, dict(rounds=0, total_steps=0, total_env_steps=0, rejected=0))
            rec['rounds'] += 1
            rec['total_steps'] += sub['steps']
            rec['total_env_steps'] += sub['env_steps']
            rec['last_norm'] = round(sub['norm'], 6)
            rec['last_seen'] = self.last_outer_step_at
            rec['accepted'] = True

        # 6. Cache serialized response bytes for all threads waiting on `completed_round`
        resp_bytes = _serialize_torch(self._response_dict_locked(accepted=True))
        self._round_bytes[completed_round] = resp_bytes
        for old_r in list(self._round_bytes.keys()):
            if old_r < completed_round - 4:
                self._round_bytes.pop(old_r, None)

        self._pending.clear()
        self._first_arrival = None

        # 7. Persist hub state + Outer-EMA eval snapshot
        if self.out_dir is not None:
            self._save_state_and_snapshot_locked(worker_weights, avg_norm, d_out)

    def _save_state_and_snapshot_locked(self, worker_weights, avg_norm, d_out):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        hub_state = {
            'round': self.round,
            'global_steps': self.global_steps,
            'global_env_steps': self.global_env_steps,
            'actor': self.actor,
            'outer_buf': self.outer_buf,
            'outer_ema': self.outer_ema,
            'critic': self.critic,
            'rel_probe': self.rel_probe,
            'pairs': self.pairs,
            'config': self.config,
            'ema_norm': self.ema_norm,
            'norm_updates': self.norm_updates,
            'rejected_count': self.rejected_count,
            'workers': self.workers,
            'started_at': self.started_at,
            'last_outer_step_at': self.last_outer_step_at,
            'hparams': dict(
                outer_lr=self.outer_lr, outer_mom=self.outer_mom, outer_ema_max=self.outer_ema_max,
                base_lr=self.base_lr, warmup_steps=self.warmup_steps, stable_steps=self.stable_steps,
                decay_steps=self.decay_steps, min_lr_frac=self.min_lr_frac,
                min_quorum=self.min_quorum, target_quorum=self.target_quorum, grace_s=self.grace_s,
                norm_mult=self.norm_mult, norm_warmup=self.norm_warmup, keep_eval_snaps=self.keep_eval_snaps,
            ),
        }
        _atomic_torch_save(hub_state, self.out_dir / 'hub_last.pt')

        model_block = {'actor': pack_actor_broadcast(self.outer_ema)}
        if self.pairs is not None:
            model_block['pairs'] = self.pairs
        snap = {
            'model': model_block,
            'config': sanitize_config(self.config),
            'iter': int(self.round),
            'round': int(self.round),
            'updates': int(self.global_steps),
            'samples': int(self.global_env_steps),
            'hours': round((time.time() - self.started_at) / 3600.0, 4),
        }
        eval_dir = self.out_dir / 'eval'
        eval_dir.mkdir(parents=True, exist_ok=True)
        _atomic_torch_save(snap, eval_dir / f'outer_r{self.round}.pt')

        if self.keep_eval_snaps > 0:
            snaps = sorted(
                (p for p in eval_dir.glob('outer_r*.pt') if p.is_file()),
                key=lambda p: int(p.stem[len('outer_r'):]) if p.stem[len('outer_r'):].isdigit() else 0,
            )
            for old in snaps[:-self.keep_eval_snaps]:
                old.unlink(missing_ok=True)

        row = {
            'round': self.round,
            'global_steps': self.global_steps,
            'global_env_steps': self.global_env_steps,
            'inner_lr': round(self.current_inner_lr(), 10),
            'workers': worker_weights,
            'avg_norm': round(avg_norm, 6),
            'ema_norm': round(self.ema_norm, 6) if self.ema_norm is not None else None,
            'd_out': round(d_out, 4),
            'time': self.last_outer_step_at,
        }
        with open(self.out_dir / 'hub_metrics.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
            f.flush()

    def handle_sync_bytes(self, raw_bytes):
        """Process one POST /sync request payload and return serialized torch response bytes."""
        req = safe_torch_load(io.BytesIO(raw_bytes), map_location='cpu')
        if not isinstance(req, dict):
            raise ValueError('sync payload must be a dict')

        worker = str(req.get('worker') or 'unknown')
        req_round = int(req.get('round', -1))
        steps = int(req.get('steps', 0))
        env_steps = int(req.get('env_steps', 0))
        compress = str(req.get('compress', 'int8'))
        actor_delta_packed = req.get('actor_delta') or {}
        critic_sd = _tensor_dict_to_fp32(req.get('critic'))
        rel_probe_sd = _tensor_dict_to_fp32(req.get('rel_probe'))
        req_norm = req.get('norm')

        delta_sd = unpack_delta_dict(actor_delta_packed, compress=compress)
        finite_ok, computed_norm = _all_finite_and_norm(delta_sd)
        norm = float(req_norm) if (isinstance(req_norm, (int, float)) and math.isfinite(req_norm) and req_norm > 0) else computed_norm

        with self.cond:
            now = time.time()
            rec = self.workers.setdefault(worker, dict(rounds=0, total_steps=0, total_env_steps=0, rejected=0))
            rec['last_seen'] = round(now, 3)

            # Stale round or explicit pull (steps <= 0): reject cleanly and return current global weights
            if req_round != self.round or steps <= 0 or not actor_delta_packed:
                rec['accepted'] = False
                return _serialize_torch(self._response_dict_locked(accepted=False))

            # Anomaly norm gate + finiteness check
            gate_ok, reason = (self._check_norm_gate_locked(norm) if finite_ok else (False, 'non-finite delta tensor'))
            if not gate_ok:
                self.rejected_count += 1
                rec['rejected'] = rec.get('rejected', 0) + 1
                rec['last_norm'] = round(norm, 6) if math.isfinite(norm) else None
                rec['last_reject_reason'] = reason
                rec['accepted'] = False
                return _serialize_torch(self._response_dict_locked(accepted=False))

            my_round = self.round
            self._pending[worker] = dict(
                worker=worker, steps=steps, env_steps=env_steps,
                delta=delta_sd, critic=critic_sd, rel_probe=rel_probe_sd, norm=norm,
            )
            if self._first_arrival is None:
                self._first_arrival = time.monotonic()

            while self.round == my_round and not self._closed:
                n = len(self._pending)
                if n >= self.target_quorum:
                    self._apply_outer_step_locked()
                    self.cond.notify_all()
                    break
                if n >= self.min_quorum:
                    rem = self.grace_s - (time.monotonic() - self._first_arrival)
                    if rem <= 0.0:
                        self._apply_outer_step_locked()
                        self.cond.notify_all()
                        break
                    self.cond.wait(timeout=rem)
                else:
                    self.cond.wait(timeout=1.0)

            if my_round in self._round_bytes:
                return self._round_bytes[my_round]
            return _serialize_torch(self._response_dict_locked(accepted=False))

    def close(self):
        with self.cond:
            self._closed = True
            self.cond.notify_all()


class _HubHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def make_handler(hub):
    class HubHandler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, fmt, *args):
            if os.environ.get('AIO_HUB_QUIET') != '1':
                sys.stderr.write(f'[diloco_hub] {self.address_string()} - {fmt % args}\n')

        def _send(self, code, content_type, body):
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.split('?', 1)[0] == '/status':
                body = json.dumps(hub.status_dict(), allow_nan=False, indent=1).encode('utf-8')
                self._send(200, 'application/json', body)
            else:
                self._send(404, 'application/json', b'{"error":"not found"}')

        def do_POST(self):
            if self.path.split('?', 1)[0] != '/sync':
                self._send(404, 'application/json', b'{"error":"not found"}')
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length <= 0:
                    self._send(400, 'application/json', b'{"error":"empty body"}')
                    return
                raw = self.rfile.read(length)
                resp = hub.handle_sync_bytes(raw)
                self._send(200, 'application/octet-stream', resp)
            except Exception as exc:
                msg = json.dumps({'error': f'{type(exc).__name__}: {exc}'}).encode('utf-8')
                self._send(500, 'application/json', msg)

    return HubHandler


class DiLoCoServer:
    """Managed ThreadingHTTPServer wrapper around a DiLoCoHub instance."""

    def __init__(self, hub, host='0.0.0.0', port=8770):
        self.hub = hub
        self.httpd = _HubHTTPServer((host, int(port)), make_handler(hub))
        self.host, self.port = self.httpd.server_address[:2]
        self._thread = None

    @property
    def url(self):
        h = '127.0.0.1' if self.host in ('0.0.0.0', '') else self.host
        return f'http://{h}:{self.port}'

    def start(self):
        self._thread = threading.Thread(target=self.httpd.serve_forever, kwargs={'poll_interval': 0.1}, daemon=True)
        self._thread.start()
        return self

    def serve_forever(self):
        try:
            self.httpd.serve_forever()
        finally:
            self.hub.close()
            self.httpd.server_close()

    def shutdown(self):
        self.hub.close()
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def build_parser():
    ap = argparse.ArgumentParser(description='Elastic Equal-H MuLoCo Coordinator Hub.')
    ap.add_argument('--init', type=Path, default=None, help='Initial checkpoint (.pt) to initialize master weights')
    ap.add_argument('--init-from-raw', action='store_true', help='Initialize from raw.actor instead of model.actor')
    ap.add_argument('--out', type=Path, default=Path.home() / 'diloco-hub', help='Hub state & eval snapshot output dir')
    ap.add_argument('--resume', action='store_true', help='Resume from <out>/hub_last.pt if present')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8770)
    ap.add_argument('--outer-lr', type=float, default=0.7)
    ap.add_argument('--outer-mom', type=float, default=0.6)
    ap.add_argument('--outer-ema-max', type=float, default=0.55)
    ap.add_argument('--base-lr', type=float, default=6e-5)
    ap.add_argument('--warmup-steps', type=int, default=0)
    ap.add_argument('--stable-steps', type=int, default=100_000)
    ap.add_argument('--decay-steps', type=int, default=0)
    ap.add_argument('--min-lr-frac', type=float, default=0.1)
    ap.add_argument('--min-quorum', type=int, default=1)
    ap.add_argument('--target-quorum', type=int, default=2)
    ap.add_argument('--grace-s', type=float, default=120.0)
    ap.add_argument('--norm-mult', type=float, default=3.0)
    ap.add_argument('--norm-warmup', type=int, default=2)
    ap.add_argument('--keep-eval-snaps', type=int, default=12)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    hp = dict(
        out_dir=a.out, outer_lr=a.outer_lr, outer_mom=a.outer_mom, outer_ema_max=a.outer_ema_max,
        base_lr=a.base_lr, warmup_steps=a.warmup_steps, stable_steps=a.stable_steps,
        decay_steps=a.decay_steps, min_lr_frac=a.min_lr_frac,
        min_quorum=a.min_quorum, target_quorum=a.target_quorum, grace_s=a.grace_s,
        norm_mult=a.norm_mult, norm_warmup=a.norm_warmup, keep_eval_snaps=a.keep_eval_snaps,
    )
    hub_pt = a.out / 'hub_last.pt'
    if a.resume and hub_pt.exists():
        hub = DiLoCoHub.from_hub_state(hub_pt, **hp)
    elif a.init is not None:
        hub = DiLoCoHub.from_checkpoint(a.init, init_from_raw=a.init_from_raw, **hp)
    else:
        raise SystemExit('--init CKPT (or --resume with existing <out>/hub_last.pt) is required')
    srv = DiLoCoServer(hub, host=a.host, port=a.port)
    print(json.dumps(dict(listening=srv.url, out=str(a.out), round=hub.round, global_steps=hub.global_steps)), flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()

---
license: mit
library_name: pytorch
tags:
  - process-control
  - advanced-process-control
  - reinforcement-learning
  - safetensors
---

# AIO1: Pretrained End-to-End Neural Network for General Advanced Process Control

English | [中文](README_zh.md)

**AIO1** is a single pretrained end-to-end neural network (~128 million active runtime parameters) designed as a general multivariable Advanced Process Control (APC) controller. It controls arbitrary industrial plants through one dimensionless, plant-agnostic interface:

- **No commissioning step test**: runs closed-loop immediately on unseen plants.
- **No formula, identifier, or MPC solver at runtime**: no underlying PID loop, no recursive least squares, and no online optimization solver.
- **Plant-agnostic interface**: receives only normalized public instrument readings (controlled variables, setpoints, bands, limits, priorities, measured disturbances, valve openings, rate limits, optional loop pairing and action directions, and decision interval). It infers plant dynamics, dead time, and operating drift directly from its own closed-loop history.

- **Code repository**: [M3rcyzzz/AIOGym2](https://github.com/M3rcyzzz/AIOGym2)
- **Live research progress & replays**: https://m3rcyzzz.club/aio/
- **ModelScope mirror**: [M3rcyzzz/AIO1](https://www.modelscope.cn/models/M3rcyzzz/AIO1)

## Status (2026-11-28)

Research checkpoint (simulation only, single seed, **not** validated on real industrial plants). This repository always stores the latest active checkpoint; each update replaces the previous release.

## Architecture & Training Summary

- **Runtime network (~128.6M active parameters, ~131.0M total in file)**:
  - Shared dimensionless encoders map every controlled variable, measured disturbance, and valve into 768-dim tokens so the exact same weights adapt natively to plants of any size (trained jointly across 12×5, 24×10, and 48×20).
  - **12 spatio-temporal attention layers** (width 768, 12 heads, pre-LayerNorm residuals), each combining: (A) causal temporal attention over the past 128 steps with rotary positional embeddings (RoPE), QK normalization, and KV caching; (B) multi-lag cross-channel attention directly looking back 1, 3, 8, 16, 24, 36, 48, and 64 steps to handle long pure dead times; (C) simultaneous cross-channel interaction attention for multivariable decoupling and constraint trade-offs; and (D) a 3,072-wide SiLU feed-forward network.
  - **Two persistent feedback memories + in-flight pipe predictor + 1-hop decoupling messages** extending context far past 128 steps: a 64-dim multi-timescale per-channel exponential moving average (4–2,048 steps) fed back into Layer 0, and a gain-aware 8-stage causal state-space + symmetric Gated DeltaNet pairwise associative memory updated online in $O(1)$ per step, injecting 8-scale in-flight pipe displacements (a multivariable neural Smith predictor) into Layer 0, biasing all 12 layers' channel & delay attention, and driving direct 1-hop CV/MV decoupling messages.
  - **Precision-gated direct action head** with width-aware scaling ($1/\sqrt{768}$), a 3-decade multiplicative precision gate adapting step size to online physical plant gain, and `tanh` rate-limit bounding.
- **Training method (pure real control cost, zero imitation)**:
  - Trained from scratch by backpropagating real closed-loop control cost (out-of-band tracking error, zone constraint violations, trips, and reversal-only anti-chatter smoothing) over 64-step windows directly through a differentiable process simulator with sensitivity-scaled exploration noise.
  - Long-horizon guidance past 64 steps comes from a training-only 4-head dead-time privileged critic (64–256 future steps), and unpublished cross-coupling & physical gain calibration uses a training-only 75-parameter linear probe; both are completely removed at deployment.

## Verified Results (2026-11-28, Simulation)

Unified 0–100 score scale: **0 = all valves frozen**, **100 = oracle MPC** (which always knows the exact true plant equations and internal state). All controllers run the exact same unseen plants, events, and noise.

### Unseen Random Multivariable Plants (Stage 10 Round 80 checkpoint)

| Plant size (CVs × MVs) | Episodes | AIO1 (this checkpoint) | Self-tuning PID | Static-model MPC (upper ref) |
|---|---:|---:|---:|---:|
| **12×5 standard validation** | 256 | **47.0** (levels **40.0** / non-level **50.8**; win rate **46.5%**) | 48.6 | 91.2 |
| **24×10 chained units** | 64 | **57.7** (stage peak **58.8**; levels up to **60.7**, **50.0%** win rate) | 53.0 | 97.9 |
| **48×20 large chained units** | 32 | **63.1** (stage peak **64.6**, non-level **81.3**, violations **10.6%**) | 56.2 | 98.3 |

- Within **1.6 points** of self-tuning PID (48.6) on 12×5 with integrating levels breaking 40 for the first time (**40.0**), non-level hitting **50.8**, post-disturbance recovery reaching **49.9**, pre-event breaking 40 at **40.2**, and hidden-pairing (blind) control reaching **17.7** (stage peak **18.2**, >4.7× Stage 8).
- Ahead of self-tuning PID on both larger multivariable benchmarks (**+4.7 pts** on 24×10 and **+6.9 pts** on 48×20 at round 80; up to **+5.8** and **+8.4** across Stage 10).

### Zero-Shot Mechanistic Industrial Plants (never seen during training)

Evaluated across 8 published mechanistic process families using this exact Stage 10 round 80 checkpoint (8 episodes each = 64 episodes, **0 trips**):

- **Eight-family average**: AIO1 **70.5** (**first time past 70; +8.3 points ahead of Self-tuning PID 62.2**; Static-model MPC **90.9**).
- Per-family scores: Distillation column A **94.7** (**+13.5 ahead of static-model MPC 81.2** and +56.4 ahead of PID 38.3), BSM1 wastewater **93.0** (PID 86.2), Building heating **74.5** (PID 66.5), Quadruple tank **73.5** (+7.0 ahead of PID 66.5), Tennessee Eastman Process **70.7** (first time past 70; PID 72.4), pH neutralization **62.5** (**+22.2 ahead of PID 40.3**), Exothermic CSTR **50.7** (PID 60.3), Debutanizer **44.5** (+7.3 vs. Stage 9; PID 66.9).

## Files

| File | Description |
|---|---|
| `model.safetensors` | Standalone inference weights (~128.6M active / 131.0M total parameters, `safetensors` format) |
| `config.json` | Minimal loader configuration |
| `modeling_aio.py`, `code/` | Self-contained inference loader (`load()`, `Controller`, `encode()`) and slim runtime modules |
| `README.md`, `README_zh.md`, `LICENSE` | Bilingual model cards and MIT license |

## Usage

```python
# pip install torch numpy safetensors huggingface_hub
from huggingface_hub import snapshot_download
import sys
import numpy as np

path = snapshot_download("M3rcyzzz/AIO1")
sys.path.insert(0, path)
import modeling_aio as M

net = M.load(device="cpu")           # or device="cuda" / "mps"
ctl = M.Controller(net)              # keeps per-plant history & memory; call ctl.reset() for a new plant

# At each control step, pass normalized public signals (numpy arrays):
packet = M.encode(
    values=y, previous=y_prev, target=sp, last_target=sp_prev,
    scale=cv_scale, lower=cv_low, upper=cv_high, tracking=is_tracked,
    command=u, last_command=u_prev, u_low=u_min, u_high=u_max,
    slew=du_max, pairing=pairing, dt=dt,
    sign=paired_gain_sign,           # +1 / -1 per MV, 0 if unknown; optional dv=, weight=
)
move = ctl.step(packet)              # per-MV increment in [-1, 1] of slew limit (0 on unused slots)
u = np.clip(u + move[:len(u)] * du_max, u_min, u_max)
```

- **Reverse-acting loops**: any valve with `sign = -1` is mirrored automatically on input and output by `Controller`.
- **Variable plant sizes**: `encode()` packs up to 12 CVs × 5 MVs by default (unused slots masked); the network weights themselves are dimension-agnostic and accept larger canonical packets (`24×10`, `48×20`) natively.

## Limitations

- Simulation-only research checkpoint; **never validated on a real industrial plant** and **not safety-certified**.
- Trained on a single random seed and still actively improving on long-dead-time plants, integrating levels, and hidden-pairing control.

## License

MIT (Copyright (c) 2026 Mercy Zhang), see `LICENSE`.

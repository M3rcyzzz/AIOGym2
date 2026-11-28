# 云端与本地混合训练操作指南

本指南说明如何在 Colab G4、本地两台 GPU 机器（`guomao` RTX 5090 24 GB 与 `baolong` RTX 4090 16 GB，均为 WSL2 Linux 环境）以及 Mac 调度机之间分工协同，覆盖两套运行档位和跨机无缝续训。

## 总体分工：两套档位，一套工具

- **档位一（Colab 额度充足期）**：Colab G4 专注纯训练，关闭训练循环内的验证与结束后的轨迹检查，每 10 轮导出约 268 MB 的轻量验证快照；本地 `guomao`（白天）/`baolong`（全天候）与 Mac 负责零额度异步验证、机理装置考试、困难种子挖掘和上线前的本地对照实验。
- **档位二（Colab 额度空窗期）**：将主线权重无缝交接到本地，由 `guomao` 与 `baolong` 组成弹性双机等步长 MuLoCo 协同训练，Mac 运行协调中心 `monitoring/diloco_hub.py`。任意一台本地机器离线、进入夜间静音时段或抽调去做对照实验时，另一台自动按单机模式继续推进，不中断主线。

> **夜间静音硬约束（`guomao`）**：`guomao` 在每天 **00:00–10:30（北京时间 `Asia/Shanghai`）** 禁止运行任何训练、GPU 基准测试或 GPU 验证，避免风扇噪音打扰休息。该规则已直接内置到代码默认行为中：
> - `local_job.py` 在 `guomao` 启动训练时自动注入 `--quiet-hours 00:00-10:30`，到达 `00:00` 自动清空显存、保持 0% GPU 休眠，次日 `10:30` 自动唤醒续训（休眠时间不计入有效训练时长）；
> - `eval_worker.py` 在 `00:00–10:30` 期间自动将指向 `guomao` 的验证任务改派到 `baolong`（或 `--fallback-eval-host local` 本机 Mac）；
> - 在双机弹性 MuLoCo 模式下，白天 `10:30–00:00` 由 `guomao` 与 `baolong` 协同推进，夜间 `00:00–10:30` 自动由办公室的 `baolong` 单机接力，次日 `10:30` `guomao` 醒来后自动对齐到最新全局权重归队。

---

## 1. 本地 WSL2 环境一次准备

为避免 Windows `sshd.exe` 并发断开与路径转义问题，同时保证不改动 `baolong` 的 Windows `.venv`、`D:\aiogym2-runs` 和 `guomao` 的 8765 渲染端口，所有本地训练与缓存都放在 WSL2 Linux 原生 ext4 文件系统下（`~/.cache/aio/{inductor,triton,val-cache}` 与 `/root/aio-runs`）。

先查看准备脚本，再按需在 WSL2 内应用：

```bash
# 仅打印脚本，不执行改动
python3 -m monitoring.local_job setup-wsl guomao
python3 -m monitoring.local_job setup-wsl baolong

# 应用 Linux 侧目录、缓存迁移与 /etc/wsl.conf 的 systemd=true 配置
python3 -m monitoring.local_job setup-wsl guomao --apply
python3 -m monitoring.local_job setup-wsl baolong --apply
```

按输出提示在两台机器的 WSL2 内执行一次 `sudo tailscale up --ssh --hostname=guomao-wsl`（与 `baolong-wsl`）后，后续命令可直接把主机名写成 `guomao-wsl` / `baolong-wsl`，通过原生 Linux SSH 与 `rsync` 通信；未配置前仍自动兼容 `guomao` / `baolong` 经 Windows 代理进入 WSL2。

---

## 2. 档位一：Colab 零验证纯训练 + 本地异步验证

### 2.1 启动 Colab 纯训练任务

`monitoring/colab_job.py` 已升级为 4 线程并行拉取分片、换机续跑时自动跳过冗余 `in/*.pt` 上传、每轮同步轻量 `eval/row*.pt` 快照，并默认每 30 分钟（`--last-every 1800`）及换机/结束时同步 1.6 GB 的 `last.pt`：

```bash
python3 -m monitoring.colab_job start sq-pure-11 --ref main --gpu G4 \
  --every 600 --last-every 1800 --put ~/colab-runs/sq-pure-10/run/last.pt -- \
  python -m aio.train --out {run} --init {in}/last.pt --resume \
  --seq d768,l12,h12,r2 --seq-ckpt --compile --priv-critic --opt muon \
  --eval-every 0 --eval-snap-every 10 --keep-eval-snaps 6 \
  --cosine-by updates --total-updates 6000 --lr 6e-5 \
  --sizes 6:12x5@80 1:24x10@28 1:48x20@18
```

### 2.2 启动零额度异步验证与机理考试守护

`monitoring/eval_worker.py` 监听本地同步回来的 `eval/row*.pt` 快照，在本地或远端 WSL2 GPU 上运行 `aio.train --score`，结果只追加写入旁路文件 `run/eval.jsonl`（绝不修改 `metrics.jsonl`，避免与 `colab_job.py` 同步冲突），同步维护 `run/best_eval.json` 与 `run/eval/best_eval.pt`，并按里程碑自动运行机理装置考试 `aio.sim.famexam`：

```bash
python3 -m monitoring.eval_worker \
  --run-dir ~/colab-runs/sq-pure-11 \
  --eval-host guomao-wsl \
  --val-cache /root/.cache/aio/val-cache \
  --val-n 256 --val-sizes 24x10@64 48x20@32 \
  --famexam-every 50
```

如需在 Mac 本机一次性评测当前全部待评快照，可加 `--eval-host local --once`。

### 2.3 零额度挖掘困难装置种子

基于最新较优快照，在独立种子段（默认 `10000000` 起）评估 `12x5` 装置，按 `(模型得分 - 老师得分) / max(不动得分 - 老师得分, 100)` 选出前 25% 困难装置写入 JSON，供训练侧 `--hard-seeds` 使用：

```bash
python3 -m monitoring.mine_hard_seeds \
  --ckpt ~/colab-runs/sq-pure-11/run/eval/best_eval.pt \
  --out ~/colab-runs/sq-pure-11/hard_seeds.json \
  --n 512 --start-seed 10000000 --top-frac 0.25
```

### 2.4 统一进度与验证汇总推送

`monitoring/train_status.py` 自动合并每个任务的旁路 `eval.jsonl`，并可通过 `--workers` 并入本地 WSL2 任务（每台机器 5 秒超时，断连时平滑回退到缓存状态）：

```bash
python3 -m monitoring.train_status \
  --workers guomao-wsl,baolong-wsl \
  --push 52 --every 600
```

---

## 3. 跨机无缝交接（Colab <-> `guomao` <-> `baolong`）

交接命令会先校验源 `last.pt` 完整性与训练计数器，再传输 `last.pt`、`run.json`、`metrics.jsonl`（及已有 `eval.jsonl`），在目标 `metrics.jsonl` 末尾追加一条 `{"handoff": {...}}` 审计记录，并打印可直接复制的续训参数：

```bash
# Colab -> guomao 本地续训
python3 -m monitoring.local_job handoff \
  --from colab:sq-pure-11 \
  --to local:guomao-wsl:sq-pure-11

# 本地训练一段时间后再交回 Colab
python3 -m monitoring.local_job handoff \
  --from local:guomao-wsl:sq-pure-11 \
  --to colab:sq-pure-12
```

在本地 WSL2 启动 24×7 后台训练、查询温度心跳（达到 84 °C 自动在 `heartbeat.json` 与日志中告警）、拉取或停止：

```bash
# 启动后台训练（自动注入 ext4 缓存、expandable_segments 与 GPU 温度心跳）
python3 -m monitoring.local_job start guomao-wsl sq-pure-11 --systemd -- \
  /root/AIOGym2/.venv/bin/python -m aio.train --out {run} --resume \
  --cosine-by updates --total-updates 12000 --lr 6e-5 \
  --val-cache {val_cache} --eval-every 0 --eval-snap-every 10 \
  --card-alloc expandable_segments:True,garbage_collection_threshold:0.8

# 查看运行状态、最新轮次与 GPU 温度/功耗/显存
python3 -m monitoring.local_job status guomao-wsl sq-pure-11

# 同步日志、eval 旁路和快照回 Mac（需要 last.pt 时加 --with-last）
python3 -m monitoring.local_job sync guomao-wsl sq-pure-11

# 平滑停止远端任务
python3 -m monitoring.local_job stop guomao-wsl sq-pure-11
```

---

## 4. 档位二：Colab 空窗期的双机弹性 MuLoCo 协同

### 4.1 在 Mac 上启动协调中心

`monitoring/diloco_hub.py` 在 Mac 内存中以 FP32 维护全局主权重、外层 Nesterov 动量与外层 EMA（1.3 亿参数模型总内存小于 2.2 GB），监听 HTTP `/status` 与 `/sync`。每一轮外层更新完成后，自动原子保存 `hub_last.pt` 并导出约 268 MB 的外层 EMA 验证快照 `eval/outer_r{round}.pt`：

```bash
python3 -m monitoring.diloco_hub \
  --init ~/colab-runs/sq-pure-11/run/last.pt \
  --out ~/diloco-hub/sq-gap-1 \
  --host 0.0.0.0 --port 8770 \
  --outer-lr 0.7 --outer-mom 0.6 \
  --base-lr 6e-5 --warmup-steps 0 --stable-steps 100000 --decay-steps 0 \
  --min-quorum 1 --target-quorum 2 --grace-s 120
```

如协调中心重启，加 `--resume` 即可从 `~/diloco-hub/sq-gap-1/hub_last.pt` 无损恢复。

### 4.2 启动 `guomao` 与 `baolong` 工作节点

两台机器使用不同 `--seed`（保证抽样装置完全独立），按相同内层步数 `--diloco-h 32` 与 Mac 协调中心同步（传输按 128 分组对称 INT8 压缩的增量，仅约 133 MB，并同步平均 `PrivCritic` 与 `rel_probe`）：

```bash
# guomao 节点（24 GB RTX 5090，自动附加 --quiet-hours 00:00-10:30；将 MAC_TAILSCALE_IP 替换为 Mac 的 Tailscale 地址）
python3 -m monitoring.local_job start guomao-wsl muloco-g --systemd -- \
  /root/AIOGym2/.venv/bin/python -m aio.train --out {run} --resume \
  --seed 101 --ema 0 --eval-every 0 --compile \
  --sizes 6:12x5@72 1:24x10@28 1:48x20@16 --accum 1 \
  --diloco-hub http://MAC_TAILSCALE_IP:8770 --diloco-h 32 \
  --diloco-worker guomao --diloco-compress int8

# baolong 节点（16 GB RTX 4090，全天候运行，夜间 00:00-10:30 自动单机接力主线）
python3 -m monitoring.local_job start baolong-wsl muloco-b --systemd -- \
  /root/AIOGym2/.venv/bin/python -m aio.train --out {run} --resume \
  --seed 202 --ema 0 --eval-every 0 --compile \
  --sizes 6:12x5@44 1:24x10@20 1:48x20@8 --accum 2 \
  --diloco-hub http://MAC_TAILSCALE_IP:8770 --diloco-h 32 \
  --diloco-worker baolong --diloco-compress int8
```

### 4.3 弹性摘除、重新接入与自动验证

- **抽调 `baolong` 做对照实验**：直接执行 `local_job stop baolong-wsl muloco-b`。协调中心在首个节点到达后等待 `--grace-s 120` 秒，未等到第二个节点即自动以单机权重完成本轮更新，`guomao` 不会停机或报错。
- **`baolong` 重新回归主线**：直接重启 `muloco-b`。若其本地轮次落后，协调中心会返回 `accepted=False` 与最新全局权重；节点自动同步到最新轮次后，从下一轮起零陈旧度参与双机合并。
- **自动评估外层合并快照**：对协调中心目录运行 `eval_worker.py` 即可自动评估每一个 `eval/outer_r*.pt`：

```bash
python3 -m monitoring.eval_worker \
  --run-dir ~/diloco-hub/sq-gap-1 \
  --eval-host local --val-cache ~/.cache/aio/val-cache \
  --val-n 256 --val-sizes 24x10@64 48x20@32 --famexam-every 10
```

---

## 5. 验证与日常检查

```bash
# 运行完整运维与协调中心单元测试（零 GPU 依赖）
PYTHONPATH=. pytest tests/test_fleet_ops.py tests/test_colab_job.py tests/test_site_progress.py -q

# 查询协调中心实时状态
curl -s http://127.0.0.1:8770/status

# 检查本地与云端统一状态汇总（不推送）
python3 -m monitoring.train_status --workers guomao-wsl,baolong-wsl
```

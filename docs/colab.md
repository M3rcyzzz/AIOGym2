# Colab 远程训练调度指南

仓库内置 `monitoring/colab_job.py`，用于从本地 Mac 通过 `colab` CLI 自动启动、监控、同步与无损续跑长时间 GPU 训练任务。

## 常用命令（仓库根目录）

```bash
# 启动远程训练任务（自动打包指定 git 提交、上传输入权重并起后台看守）
python3 -m monitoring.colab_job start NAME --ref main --gpu G4 --put CKPT.pt -- \
  python -m aio.train --out {run} --init {in}/CKPT.pt --resume ...

# 纯 CPU 评测 / 参照计算任务
python3 -m monitoring.colab_job start NAME --ref main --gpu CPU --high-mem -- \
  python -m aio.sim.famexam ...

# 查看任务状态与实时显存/日志
python3 -m monitoring.colab_job status NAME

# 停止任务并回收远端机器
python3 -m monitoring.colab_job stop NAME
```

## 工作机制

- **占位符**：`{run}` 自动映射到远端运行目录（看守定期同步回本地 `~/colab-runs/NAME/run`）；`{in}` 映射 `--put` 上传的初始文件目录。
- **自动同步与断线续跑**：本地看守进程默认每 10 分钟拉回 `last.pt`、`best.pt`、`metrics.jsonl` 与 `job.log`（历史 `row*.pt` 快照默认跳过，需要时加 `--sync-rows`）。若远端实例中断或临近 `--max-hours`（默认 23 小时），看守会提前 3 小时预申新机，抢到后同步最新 `last.pt` 再无缝切换续跑。
- **训练命令建议**：配合 `--resume`（自动衔接最新 `last.pt`）与 `--restarts 20`（内部卡死超时自恢复），确保长跑过程无人值守稳定推进。
- **网站进度条推送**：本地另起 `python3 -m monitoring.train_status --push <HOST> --every 600` 可每 10 分钟将活跃任务进度汇总写入线上 `train-status.json`。

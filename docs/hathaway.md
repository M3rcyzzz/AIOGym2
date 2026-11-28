# Hathaway 仿真环境边界

`aio/sim` 提供的 GPU 加速、端到端可微过程控制仿真环境遵循严格的单向依赖边界，便于独立封装为标准基准包（Hathaway）：

## 依赖边界

- **环境层**（`aio/sim/`、`aio/interface.py`、`aio/scale.py`）：不认识任何控制网络或训练算法，只负责装置动力学、过程事件、故障注入、评分口径、工程参照控制器与仅训练期可见的特权状态信号。
- **训练与策略层**（`aio/seqnet.py`、`aio/net.py`、`aio/train.py`、`aio/actor.py`、`aio/frame.py` 等）：单向依赖环境层。
- `tests/test_env_boundary.py` 自动守护该单向边界：若 `aio/sim` 新增任何对训练侧模块的反向导入，测试会立即报错。

## 基准标准

环境标准版本为 `H4`（详见 `docs/standard.md`），提供固定考试批、参照成绩表、统一 0–100 评分与 Gymnasium 向量环境接口（`aio/sim/gym.py`、`aio/sim/bench.py`）。

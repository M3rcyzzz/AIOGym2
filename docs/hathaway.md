# Hflash（`apcgym`）仿真环境边界

`aio/sim` 提供的 GPU 加速、端到端可微过程控制仿真环境遵循严格的单向依赖边界，将拆分为独立开源仓库 **`M3rcyzzz/Hflash`**（PyPI 包名 **`apcgym`**；文件路径保留 `docs/hathaway.md` 以兼容已有引用）。

## 依赖边界（Phase 0 边界清零）

- **环境层**（`aio/sim/`、`aio/interface.py`、`aio/scale.py`）：不认识任何控制网络或训练算法，只负责装置动力学、随机先验与机理引擎、过程事件、故障注入、评分口径、工程参照控制器、训练期特权真值信号与通用控制器插件入口（`aio/sim/plugin.py`）。
- **训练与策略层**（`aio/seqnet.py`、`aio/net.py`、`aio/train.py`、`aio/actor.py`、`aio/frame.py`、`aio/plugin.py`、`aio/tiers.py` 等）：严格单向依赖环境层；评估与回放统一通过 `--ctl MODULE:FACTORY` 插件协议注入策略。
- **单向守卫**：`tests/test_env_boundary.py` 强制 `aio/sim/` 对训练侧零反向导入（`LEFT = set()`，静态与动态导入均受检查），任何新增反向导入都会令测试失败。

## 基准标准与合规分层

- **基准版本**：Phase 0 保持现行 `H4` 冻结考题与参照表不变（详见 `docs/standard.md`）；独立开源后基准标准将演进至 **`H5 / APC-Bench v1`**。
- **合规与打包**：第三方许可署名与全部 30 类机理装置文献清单见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) 与 [CITATIONS.bib](../CITATIONS.bib)（24 族 MIT 洁净机理实现含 `plants_bsm.py`，`gaslib` 遵循 CC BY 3.0 / BSD-2-Clause / Apache-2.0 署名，`tep`/`vam`/`badgwell`/`refrig` 将在 `Hflash`/`apcgym` 中物理隔离至 `contrib_suite`）；`pyproject.toml` 核心依赖仅保留 `numpy>=1.24`（`scipy` 迁入 `[test]`/`[validate]`），并显式排除 `aio/sim/validate/data/*` 与 `data/bsm1/*` 不进 wheel。

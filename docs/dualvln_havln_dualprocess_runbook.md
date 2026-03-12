# DualVLN x HA-VLN Dual-Process Runbook

## Goal

本文记录当前已经跑通的 `DualVLN -> HA-VLN` 双进程闭环，以及复现实验所需的最小命令、约束和已知限制。

截至 `2026-03-12`，该链路已经验证：

- `InternNav` 在 `habitat` 环境中加载 `DualVLN`
- `HA-VLN` 在 `havlnce` 环境中运行 `HAVLNCEDaggerEnv`
- 两边通过 HTTP 完成 `reset -> step -> metrics`
- 至少 `1` 个 `val_unseen` episode 已完整跑通

## Why Dual Process

单进程方案最终不可取，原因不是业务逻辑，而是运行时框架冲突：

- `HA-VLN` 依赖旧版 `habitat` / `habitat_baselines`
- `InternNav` 依赖新版 Habitat 栈
- 两边共享相同模块名与全局 registry
- 同一 Python 进程内混用时会出现导入污染和 config/task 冲突

因此当前采用的方案是：

- `havlnce` 负责仿真
- `habitat` 负责推理
- 中间只传稳定的数据接口

## Architecture

组件如下：

- Server:
  - 文件: `/root/backup/HA-VLN/scripts/havln_http_env_server.py`
  - 职责: 持有 `HAVLNCEDaggerEnv`，提供 `/metadata`、`/reset`、`/step`、`/close`
- Client env:
  - 文件: `/root/backup/InternNav/internnav/env/havln_http_env.py`
  - 职责: 把 HTTP 响应包装成 `InternNav` evaluator 可用的 env
- Eval config:
  - 文件: `/root/backup/InternNav/scripts/eval/configs/havln_http_dual_system_cfg.py`
  - 职责: 指定 `havln_http` env 和 DualVLN 评测参数
- HA-VLN task config:
  - 文件: `/root/backup/InternNav/scripts/eval/configs/vln_ha_r2r.yaml`
  - 职责: 定义 4-action 的 HA-VLN 评测侧配置假设

传输的数据目前包括：

- `rgb`
- `depth`
- `instruction_text`
- `scene_id`
- `episode_id`
- 基础 metrics

## Prerequisites

需要两个独立环境：

- `havlnce`
  - 用于运行 `HA-VLN`
- `habitat`
  - 用于运行 `InternNav`

同时需要保证：

- `HA-VLN` 数据已就位
- `HUMAN_GLB_PATH` 已指向 `../Data/HAPS2_0/human_motion_glbs_v3`
- `HUMAN_COUNTING=False`
- server 启动时带上你本机已经验证可用的 OpenGL 预加载环境

## Launch Commands

### 1. Start HA-VLN server

在第一个终端里执行：

```bash
conda activate havlnce
cd /root/backup/HA-VLN
export PYTHONPATH=/root/backup/HA-VLN/agent/VLN-CE:/root/backup/HA-VLN:$PYTHONPATH
export LD_PRELOAD=/lib/x86_64-linux-gnu/libGLX_nvidia.so.0:/lib/x86_64-linux-gnu/libGLdispatch.so.0
python scripts/havln_http_env_server.py --port 8899 --split val_unseen --max-episodes 1
```

说明：

- server 必须单线程运行，避免 `GL::Context::current(): no current context`
- 当前默认只跑 `1` 个 episode，便于 smoke test

### 2. Optional health check

在第二个终端先确认 server 存活：

```bash
conda activate habitat
python - <<'PY'
import requests
print(requests.get('http://127.0.0.1:8899/health', timeout=10).json())
print(requests.get('http://127.0.0.1:8899/metadata', timeout=10).json())
PY
```

### 3. Start DualVLN client

在第二个终端继续执行：

```bash
conda activate habitat
cd /root/backup/InternNav
export PYTHONPATH=/root/backup/InternNav:$PYTHONPATH
python scripts/eval/eval.py --config scripts/eval/configs/havln_http_dual_system_cfg.py
```

## Current Behavior

当前这条链路已经确认能产出 episode 级结果，例如：

- `success`
- `spl`
- `oracle_success`
- `distance_to_goal`

当前结果偏低是预期内现象，不应被解读为 “DualVLN 无法用于 HA-VLN”。更准确的解释是：

- 系统集成已经成功
- 策略层还处于最小兼容模式

## Current Compatibility Policy

当前桥接是“尽量不影响原有 InternNav 路径”的做法：

- 只有 `env_type='havln_http'` 时才会走双进程 env
- 原有标准 `habitat` env 不受影响
- evaluator 只有在缺失 pose sensor 或 pitch action 时才进入降级逻辑

## Known Limitations

截至当前，限制主要有这些：

- `HA-VLN` 只有 4-action，缺少 `LOOKUP/LOOKDOWN`
- 当前没有真正把 look-down 视角回传给 DualVLN 的局部轨迹分支
- `gps/compass` 缺失时只能走降级模式
- `↓` 和 pixel-goal 目前都是启发式映射，不是原始设计语义
- 当前更适合做“小样本可运行性验证”，不适合直接做论文级结论

## Recommended Next Steps

建议按这个顺序继续：

1. 先把 `max_eval_episodes` 提到 `8-32`，拿一个粗略平均指标。
2. 继续打磨 4-action fallback，把 `pixel goal -> 离散动作序列` 做得更稳。
3. 如果需要更接近原始 DualVLN，再考虑给双进程接口补 look-down 观测。
4. 最后才考虑完整评测和更大规模实验。

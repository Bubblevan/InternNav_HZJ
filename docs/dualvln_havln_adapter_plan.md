# DualVLN To HA-VLN Minimal Adapter Plan

## Goal

先不要追求“完整复现 HA-VLN 论文设定”，第一阶段只做一件事：

- 让 `/root/InternNav` 里的 DualVLN 能在 `HA-VLN` 的 `HAVLNCE` 环境上跑完少量 `val_unseen` episode
- 保留闭环评测
- 暂时关闭 human counting
- 暂时不要求作者原始 checkpoint 和全部论文指标

这个阶段的成功标准是：

- 能读取 `HA-R2R` 指令与 episode
- 能拿到 HA-VLN 的 RGBD 观测
- 能输出 4-action 控制
- 能得到 `success/spl/path_length` 等基础指标

## What Is Already Verified

HA-VLN 侧目前已经验证到：

- `HAVLNCEDaggerEnv.reset()` 可成功运行
- `HAVLNCEDaggerEnv.step()` 可成功运行
- 初始观测包含：
  - `rgb`
  - `depth`
  - `instruction`
  - `progress`
  - `shortest_path_sensor`
- task 配置是旧版 Habitat 0.1.7 风格
- 动作空间只有：
  - `STOP`
  - `MOVE_FORWARD`
  - `TURN_LEFT`
  - `TURN_RIGHT`

同时，HA-VLN 作者公开仓库里看起来没有直接提供核心导航 checkpoint，只给了训练命令和 `ddppo` 编码器权重。

## Current DualVLN Assumptions

`/root/InternNav` 当前 Habitat evaluator 默认假设的是新一点的 VLN-CE 环境，并且依赖以下条件：

- 观测里必须有 `gps`
- 观测里必须有 `compass`
- 动作空间里必须有 `look_up`
- 动作空间里必须有 `look_down`
- Habitat config 采用 0.2.x 的 structured/Hydra 体系

这些假设在 HA-VLN 当前环境里都不成立。

## Compatibility Gaps

### 1. Habitat version gap

HA-VLN 基于 Habitat 0.1.7。

DualVLN 当前 Habitat evaluator 基于 Habitat 0.2.x 配置方式开发。即使底层都是 Habitat-sim，task/config/registry 的接法也不同，不能直接复用。

### 2. Observation gap

DualVLN evaluator 在运行时会直接读取：

- `observations["gps"]`
- `observations["compass"]`

HA-VLN 当前观测里没有这两个键。

### 3. Action gap

DualVLN evaluator 会额外执行：

- `LOOKDOWN`
- `LOOKUP`

HA-VLN 任务当前只有 4 个动作，没有 pitch 动作。

### 4. System-1 input gap

当前 DualVLN 的局部轨迹分支依赖 look-down RGB/depth 视角。HA-VLN 当前默认只有前视 RGB/depth。

## Recommended Strategy

不要先尝试把 HA-VLN 补成与 DualVLN 现有 evaluator 完全一致。

第一阶段更稳的路线是：

1. 保留 HA-VLN 作为环境与数据源
2. 新增一个 `ha_vln` evaluator/config 分支
3. 在这个分支里先去掉 DualVLN 对 `gps/compass/look_up/look_down` 的硬依赖
4. 先把 DualVLN 当作“前视 RGBD + instruction -> 4-action”策略接起来
5. 跑通小样本 episode 后，再考虑是否补 pitch 动作或 pose 观测

## Phase 1 Scope

第一阶段只做最小闭环：

- 输入：
  - front `rgb`
  - front `depth`
  - `instruction`
- 输出：
  - `STOP`
  - `MOVE_FORWARD`
  - `TURN_LEFT`
  - `TURN_RIGHT`

在这个阶段里：

- `gps/compass` 全部降级为可选
- `look_down/look_up` 全部禁用
- 如果 System 2 输出 pixel goal，则先不要进入当前的 look-down System 1 路径
- 可以先退化成 “System 2 only” 或 “pixel goal -> nearest discrete action heuristic”

## Concrete Implementation Plan

### A. Add HA-VLN-specific eval config

新增一套 HA-VLN 专用配置，例如：

- `scripts/eval/configs/vln_ha_r2r.yaml`
- `scripts/eval/configs/habitat_dual_system_ha_cfg.py`

目标：

- dataset path 指向 `HA-R2R`
- scene path 指向 `HA-VLN/Data/scene_datasets`
- action space 定义为 4-action
- 不再声明 `gps_sensor`/`compass_sensor`
- 不再声明 `look_up`/`look_down`

### B. Split evaluator assumptions

在 `internnav/habitat_extensions/vln/habitat_vln_evaluator.py` 里把下面几类逻辑拆开：

- 与 pose 记录相关的逻辑
- 与 look-down 视角采集相关的逻辑
- 与 4-action 离散导航直接相关的逻辑

建议增加 capability flags，例如：

- `has_pose_sensors`
- `has_pitch_actions`
- `use_system1_local_policy`

这样 evaluator 可以在 HA-VLN 分支里自动降级，而不影响原有 R2R 路径。

### C. Add a 4-action fallback path

如果环境没有 pitch 动作：

- 不执行 `LOOKDOWN/LOOKUP`
- 不调用依赖 look-down 视角的局部轨迹推理
- 优先使用 System 2 的离散动作输出
- 若 System 2 输出 pixel goal，则临时做一个简单 fallback：
  - 解析像素目标
  - 映射到左转/右转/前进中的一个短动作
  - 或直接在 Phase 1 禁用 pixel-goal mode

### D. Delay paper-specific metrics

先不要把适配目标扩到：

- `CR`
- `TCR`
- 人体计数相关输出
- 作者训练管线复现

第一阶段只关心：

- env 能跑
- DualVLN 能闭环
- 基础指标能产出

## Suggested Order

1. 先把 HA-VLN 多 episode smoke test 跑稳定
2. 在 InternNav 增加 HA-VLN 专用 config
3. 修改 evaluator，让 `gps/compass/look_up/look_down` 变成可选能力
4. 先跑 1-5 个 `val_unseen` episode
5. 再决定是否扩回 DualVLN 的 System 1 分支

## Non-Goals For Phase 1

以下内容先不做：

- 训练 HA-VLN 原始 baseline
- 复现作者未公开 checkpoint 的结果
- 补齐 GroundingDINO
- 追求论文级最终指标

## Immediate Next Command Targets

HA-VLN 侧优先验证：

```bash
python scripts/verify_havlnce_env.py --episodes 10 --steps-per-episode 5
```

InternNav 侧第一批代码改动优先落在：

- `internnav/habitat_extensions/vln/habitat_vln_evaluator.py`
- `scripts/eval/configs/`


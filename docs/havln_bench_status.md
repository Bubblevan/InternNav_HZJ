# HAVLN Bench Status

## Scope

本文记录当前本地 `HA-VLN` bench 的实际运行状态、必要数据组成、已经验证通过的部分，以及把 `DualVLN` 接入该 bench 时真正需要解决的问题。

结论先行：

- `HA-VLN` bench 本地已经基本跑通
- 当前主要风险已从“环境起不来”转为“DualVLN 接口不兼容”
- 下一步工作的重点应放在 `InternNav` evaluator 适配，而不是继续折腾 `HA-VLN` 安装

## Bench Composition

当前 bench 由以下部分组成：

- `HA-R2R`
  - 指令与 episode 标注
- `scene_datasets`
  - Matterport3D 场景资源
- `HAPS2_0`
  - 人体 motion GLB 资源
- `Multi-Human-Annotations`
  - human motion 与场景的绑定信息
- `ddppo-models`
  - 深度编码器初始化权重

补充说明：

- `GroundingDINO` 只在 `HUMAN_COUNTING=True` 时才是硬依赖
- 对于当前“先跑环境、先做 DualVLN 接入”的目标，`GroundingDINO` 不是必要条件

## Local Runtime Status

截至 `2026-03-12`，本地已经确认：

- `import habitat, habitat_sim, habitat_baselines, habitat_extensions, vlnce_baselines` 正常
- `HAVLNCEDaggerEnv.reset()` 正常
- `HAVLNCEDaggerEnv.step()` 正常
- 无头 GPU OpenGL 上下文正常
- 多 episode 动态人环境 smoke test 正常

本地实际验证命令：

```bash
cd /root/backup/HA-VLN
export PYTHONPATH=/root/backup/HA-VLN/agent/VLN-CE:/root/backup/HA-VLN:$PYTHONPATH
python scripts/verify_havlnce_env.py --episodes 10 --steps-per-episode 5
```

验证结果摘要：

- 连续 `10` 个 episode reset/step 成功
- 每个 episode 执行 `5` 步离散动作
- `summary: requested_episodes=10 failures=0`

这说明：

- 动态人注入路径可工作
- 多轮 reset 没有立刻崩
- `HAVLNCEDaggerEnv` 这条封装链路不是一次性假成功

## Observations And Actions

当前 `HA-VLN` 运行时已观测到的初始 observation keys：

- `rgb`
- `depth`
- `instruction`
- `progress`
- `shortest_path_sensor`

当前 task 动作空间：

- `STOP`
- `MOVE_FORWARD`
- `TURN_LEFT`
- `TURN_RIGHT`

这和 `InternNav` 当前 DualVLN Habitat evaluator 的默认假设不同。后者当前默认还依赖：

- `gps`
- `compass`
- `LOOKUP`
- `LOOKDOWN`

## Metrics Currently Available

在当前本地 smoke test 和 nonlearning eval 中，已经见到以下基础指标可正常返回：

- `distance_to_goal`
- `success`
- `spl`
- `path_length`
- `oracle_success`
- `steps_taken`

这意味着第一阶段的接入目标可以先定义为：

- 只追求基础闭环评测
- 先产出 `success/spl/path_length`
- 不先追求论文里的全部扩展指标

## Dataset Bridge Feasibility

本地已检查 `HA-R2R/val_unseen.json.gz`，其结构包含：

- `episodes`
- `instruction_vocab`

单条 episode 中包含：

- `episode_id`
- `scene_id`
- `instruction`
- `reference_path`
- `trajectory_id`
- `goals`
- `start_position`
- `start_rotation`

这与 Habitat VLN 的常见 `R2R` 数据字段高度接近。

额外观察：

- 当前 `HA-R2R` 的 `scene_id` 形如 `mp3d/<scan>/<scan>.glb`
- 因此在 `InternNav` 一侧，数据层很可能可以先尝试沿用 `R2RVLN-v1` loader，只要 `scenes_dir` 指向 `/root/backup/HA-VLN/Data/scene_datasets`

也就是说：

- 数据 schema 不是当前最大障碍
- 观测与动作接口才是当前最大障碍

## Local Config Adjustments Already Applied

为了让 bench 先跑起来，当前本地已经做了两项必要调整：

1. `HUMAN_GLB_PATH` 改为：

```yaml
../Data/HAPS2_0/human_motion_glbs_v3
```

原因：

- 真实的人体 GLB 目录不在 `HAPS2_0` 顶层
- 代码会直接枚举该目录下的 motion 目录

2. `HUMAN_COUNTING` 改为：

```yaml
False
```

原因：

- 先绕开 `GroundingDINO`
- 不影响动态人环境本身运行
- 适合第一阶段 DualVLN 接入

## What This Means For DualVLN

当前状态下，`DualVLN -> HA-VLN` 的问题已经比较明确：

- 不是数据缺失
- 不是仿真器完全不通
- 不是动态人环境完全不稳定

真正需要处理的是：

1. Habitat 版本差异
2. 观测缺口
3. 动作缺口
4. evaluator 里对 `look-down` 局部轨迹分支的硬依赖

## Recommended Next Step

下一阶段应聚焦于 `/root/InternNav`：

- 新增 `HA-VLN` 专用 eval config
- 让 evaluator 支持“无 `gps/compass`、无 `LOOKUP/LOOKDOWN`”的降级模式
- 先跑通少量 `val_unseen` episode

第一阶段目标不应该是：

- 复现 HA-VLN 作者未公开 checkpoint
- 补齐 GroundingDINO
- 追求论文级最终指标

第一阶段目标应该是：

- `DualVLN` 在 `HA-VLN` bench 上完成最小闭环


# HA-VLN x InternNav HTTP Bridge Summary

## Scope

本文总结截至 `2026-03-13` 为止，为了让 `InternNav / DualVLN` 在 `HA-VLN` bench 上运行而做出的 HTTP 双进程桥接改动、当前已验证能力、遗留问题，以及建议的下一步方向。

## Why HTTP Bridge

最初尝试的是单进程直接在 `InternNav` 中加载 `HA-VLN` 旧版 Habitat 栈，但很快暴露出两个无法优雅解决的问题：

- `habitat` / `habitat_baselines` 在新旧版本中模块名相同
- registry、config、task import side effects 严重，单进程混用会污染全局状态

因此当前采用双进程方案：

- `havlnce` 环境运行 `HA-VLN` simulator server
- `habitat` 环境运行 `InternNav` / `DualVLN`
- 两边通过 HTTP 交换 observation、action、metrics

## Main Files Added Or Modified

### HA-VLN side

- `/root/backup/HA-VLN/scripts/verify_havlnce_env.py`
  - 扩展为多 episode smoke test
- `/root/backup/HA-VLN/scripts/havln_http_env_server.py`
  - 新增 HTTP server
  - 暴露 `/health`、`/metadata`、`/reset`、`/step`、`/close`
  - 透传 `rgb`、`depth`
  - 新增固定俯视 `lookdown_rgb`、`lookdown_depth`
  - 透传 HA-VLN metrics

### InternNav side

- `/root/backup/InternNav/internnav/env/havln_http_env.py`
  - 新增 HTTP client env
  - 负责 decode `rgb/depth/lookdown_rgb/lookdown_depth`
- `/root/backup/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py`
  - 增加 HA-VLN 降级能力探测
  - 支持 HTTP env
  - 支持外部 look-down 观测
  - 恢复 `↓ -> System 1` 的内部控制语义
  - 保留并记录 HA-VLN 原始 metrics
  - 补算 `SR / CR / TCR`
- `/root/backup/InternNav/scripts/eval/configs/havln_http_dual_system_cfg.py`
  - 双进程正式配置
- `/root/backup/InternNav/scripts/eval/configs/havln_http_dual_system_debug_cfg.py`
  - 带第一人称 debug mp4 的配置

## Existing Docs

当前与本项工作直接相关的文档包括：

- [havln_bench_status.md](/root/backup/InternNav/docs/havln_bench_status.md)
  - 记录 HA-VLN bench 本地可运行状态
- [dualvln_havln_adapter_plan.md](/root/backup/InternNav/docs/dualvln_havln_adapter_plan.md)
  - 记录最初的适配方向和阶段划分
- [dualvln_havln_dualprocess_runbook.md](/root/backup/InternNav/docs/dualvln_havln_dualprocess_runbook.md)
  - 记录双进程启动方式和运行命令

本文档主要补的是：

- HTTP 桥接已经改了什么
- 哪些能力已经从“理论可行”变成“实测可用”

## Verified Milestones

截至目前已确认：

- `HA-VLN` 环境本地可稳定 `reset/step`
- HTTP bridge 已经打通
- `DualVLN` 已经能在 `HA-VLN` 上跑完整 episode
- 外部 `lookdown_rgb/lookdown_depth` 已接入
- `↓` 已重新恢复为 evaluator 内部控制信号，而不是伪装成离散动作
- 至少小样本 `2` 个 episode 中已有 `1` 个成功，`SPL=0.68`

这说明当前问题已经从“系统兼容性”转移到“动态人环境中的策略行为”。

## Metrics Status

当前 HA-VLN task 中实际启用的关键 measure 包括：

- `distance_to_goal`
- `success`
- `spl`
- `path_length`
- `oracle_success`
- `steps_taken`
- `collisions`
- `collisions_detail`
- `distance_to_human`

其中：

- `collisions_detail` 和 `distance_to_human` 是原始明细
- 论文里的 `SR / CR / TCR` 不是 env 原生 measure，而是后处理得到

当前桥接已经支持：

- 保留原始 raw metrics
- 在 `InternNav` 侧补算：
  - `TCR`
  - `CR`
  - `SR`

## Current Known Issues

### 1. Some failure cases look like local deadlock

已经观察到典型失败模式：

- agent 出生在狭窄房间内部
- 门口附近有动态人
- DualVLN 高层意图看起来正确，但低层局部执行会在人群附近反复重试
- 最终耗尽 step budget

这更像策略泛化问题，而不是接口错误。

### 2. Look-down orientation likely needed sign correction

最初固定俯视 sensor 的 pitch 符号很可能取反，导致视频里像“抬头 30 度”。当前已改成负角度方向，但仍需继续用视频做确认。

### 3. Human motion may be subtle

HA-VLN 的人不是通过物理驱动“走动”，而是通过按 frame 替换 GLB 模型实现动画。很多 activity 本来就是站立、坐姿、举杯、按摩、冥想等小动作，因此第一人称视频里不一定表现为明显平移。

如果仍怀疑“完全没动”，需要进一步落更明确的 per-step human state debug。

## Recommended Next Steps

建议优先做：

1. 跑 `16-32` 个 episode 小样本统计。
2. 加 stuck detector，把“局部死锁”从“全局迷路”中区分出来。
3. 如果需要更强解释性，透传更多 crowd/collision 明细并补可视化。
4. 如果旧版 HA-VLN 在动态人或传感器层面继续卡住，再考虑用新版 Habitat 重新实现 HA-VLN task。

# Changelog 2026-03-10 Technical Trace

本文记录 2026-03-10 围绕 Habitat 仿真版 DualVLN 评估链路的代码修改、验证结果与当前结论，重点服务于后续 traceback、复现实验与继续迭代。

## 一、任务目标

本轮工作的目标不是训练，而是回答三个更具体的问题：

1. 在 `main` 分支上，是否可以在不下载 `vln_ce/traj_data/r2r` 的前提下，直接跑通 Habitat 的 VLN-CE R2R 闭环评估。
2. 能否在 `habitat_dual_system_cfg.py` 这条评估链中补充大小脑耗时、调用次数与行为统计，用于解释 realworld 部署时的高延迟问题。
3. 能否在闭环评估过程中导出稳定的 replay 子集，以支持后续在 L20 上做 DualVLN 纯推理 benchmark 和单变量优化实验。

## 二、关于数据依赖的结论

结论已经明确：当前 Habitat 的 VLN-CE R2R 评估链不依赖 `vln_ce/traj_data/r2r`，而是依赖以下两类数据：

- `data/vln_ce/raw_data/r2r/{split}/{split}.json.gz`
- `data/scene_data/...` 下的 Matterport3D 场景资源（`.glb` / `.navmesh` / `.house` 等）

证据如下：

- `scripts/eval/configs/vln_r2r.yaml` 只向 Habitat dataset 提供 `data_path` 和 `scenes_dir`
- Habitat 官方 `R2RVLN-v1` loader `habitat/datasets/vln/r2r_vln_dataset.py` 只读取 `json.gz` 和 `scenes_dir`
- `traj_data` 在本项目中主要进入 LeRobot / 训练数据管线，例如 `internnav/dataset/internvla_n1_lerobot_dataset.py`

因此，若当前目标仅是 Habitat 闭环评估和推理剖析，可以不下载 343GB 的 `traj_data/r2r`。

## 三、代码修改总览

本日共涉及以下文件：

- `internnav/env/habitat_env.py`
- `internnav/habitat_extensions/vln/habitat_vln_evaluator.py`
- `scripts/eval/configs/vln_r2r.yaml`
- `scripts/eval/configs/habitat_dual_system_cfg.py`
- `scripts/eval/configs/habitat_dual_system_mini_cfg.py`
- `scripts/eval/configs/habitat_dual_system_smoke_cfg.py`
- `scripts/eval/tools/build_r2r_subset.py`
- `scripts/eval/tools/benchmark_dualvln_replay.py`

下面分文件说明。

## 四、逐文件修改说明

### 1. `internnav/env/habitat_env.py`

修改目的：支持在不改 Habitat 内部 loader 的情况下，直接从评估配置里控制“只跑某些 scene / 某些 episode / 前 N 个 episode”。

具体修改：

- 在 `generate_episodes()` 中新增 `allowed_scene_ids`
- 新增 `allowed_episode_ids`
- 新增 `max_eval_episodes`
- 过滤逻辑放在 rank 分片之后，兼容现有 distributed evaluator

效果：

- 可以直接对完整 `val_unseen` 做 smoke test，而不必必须生成 mini json
- 也可以结合 mini json 做更小的闭环调试

对应位置：

- `internnav/env/habitat_env.py:41`

### 2. `scripts/eval/configs/vln_r2r.yaml`

修改目的：修正本地实际路径和配置不一致的问题。

具体修改：

- 将 `scenes_dir` 从 `data/scene_data/mp3d_ce` 改为 `data/scene_data`

原因：

- 本地磁盘上存在的是 `data/scene_data/mp3d/...`
- Habitat 的 R2R loader 会将 `scene_id` 中的相对路径拼到 `scenes_dir` 后面，因此此处应指向 `scene_data` 根目录，而不是额外拼一层错误目录

对应位置：

- `scripts/eval/configs/vln_r2r.yaml:73`

### 3. `scripts/eval/configs/habitat_dual_system_cfg.py`

修改目的：给 DualVLN Habitat 评估增加更灵活的运行控制和 profiling 开关。

具体新增字段：

- `dataset_path_override`
- `scenes_dir_override`
- `dataset_split_override`
- `allowed_scene_ids`
- `allowed_episode_ids`
- `max_eval_episodes`
- `profile_runtime`
- `profile_modules`
- `export_replay_subset`
- `replay_num_episodes`
- `replay_seed`

效果：

- 评估时可以临时切换 mini/full dataset
- 可以直接限制 episode 数量
- 可以打开/关闭 replay 导出与运行时统计

对应位置：

- `scripts/eval/configs/habitat_dual_system_cfg.py:25`

### 4. `scripts/eval/configs/habitat_dual_system_mini_cfg.py`

修改目的：提供一个可直接运行的小规模 mini 配置。

具体内容：

- 输出目录指向 `logs/habitat/test_dual_system_mini`
- `dataset_path_override` 指向 `val_unseen_mini.json.gz`
- `max_eval_episodes = 8`
- `replay_num_episodes = 8`

用途：

- 用于快速验证 DualVLN 闭环、profiling 和 replay 导出

### 5. `scripts/eval/configs/habitat_dual_system_smoke_cfg.py`

修改目的：提供一个更符合主线需求的 smoke 配置。

具体内容：

- 不切 mini json，直接使用完整 `val_unseen`
- 仅通过 `max_eval_episodes = 8` 截断 episode 数量

用途：

- 当确认完整主分支数据可用后，用它替代 mini 配置更合适
- 避免维护额外子集文件

### 6. `internnav/habitat_extensions/vln/habitat_vln_evaluator.py`

这是本轮最核心的修改文件。

#### 6.1 配置覆盖能力

在 evaluator 初始化阶段新增对以下配置项的支持：

- dataset path override
- scenes dir override
- split override
- mini subset 过滤参数

这样可以不动 Habitat 内部代码，直接从 eval config 控制闭环规模。

#### 6.2 运行时 profiling

新增 episode 级统计字段，记录：

- `success`
- `spl`
- `oracle_success`
- `navigation_error`
- 每个 episode 的总 wall-clock 时间
- 平均每步 wall-clock 时间
- `s2_call_count`
- `s2_generate_seconds`
- `s2_latent_seconds`
- `s1_call_count`
- `s1_generate_seconds`
- `pixel_goal_ratio`
- `discrete_ratio`
- `stop_ratio`
- `pixel_goal_cycles`
- `pixel_goal_active_steps`
- `avg_s1_steps_per_cycle`

其中：

- `s2_generate_seconds` 对应 System 2 文本生成阶段
- `s2_latent_seconds` 对应从文本输出进一步生成 latent 的阶段
- `s1_generate_seconds` 对应 `generate_traj(...)` 生成局部离散动作的阶段

这些统计写入：

- `runtime_rank0.jsonl`
- `runtime_summary_rank0.json`
- 同时将核心字段写回 `progress.json`

#### 6.3 功能指标写回增强

原先只写：

- `success`
- `spl`
- `os`
- `ne`

本轮新增同时写入：

- `oracle_success`
- `navigation_error`
- `episode_wall_clock_seconds`
- `avg_step_wall_clock_seconds`
- `s2_call_count`
- `s2_avg_seconds`
- `s1_avg_seconds`
- `pixel_goal_ratio`
- `discrete_ratio`
- `stop_ratio`
- `avg_s1_steps_per_cycle`

这样后续分析不必再读多个文件拼接。

#### 6.4 Navigation Error 口径修正

优先使用 Habitat task measurement 中的 `oracle_navigation_error`，若不存在则回退到 `distance_to_goal`。

原因：

- 用户要求额外记录的是 `Navigation Error`
- 当前评估配置已注册 `oracle_navigation_error`

#### 6.5 Replay 子集导出

新增 replay subset 导出逻辑。

当前每个被选中的 episode、每个 step 会保存：

- `episode_id`
- `instruction`
- `step_id`
- 当前 RGB 原图
- 当前 depth
- look-down RGB
- look-down depth
- pose（`gps` + `compass`）
- 历史帧索引
- baseline 输出：
  - `action`
  - `pixel_goal`
  - `output_kind`
  - `llm_output`
  - `local_actions_remaining`

输出位置：

- `logs/.../replay_subset/manifest_rank0.jsonl`
- `logs/.../replay_subset/<scene>_<episode>/...`

增加 look-down 观测的原因：

- 当前 DualVLN evaluator 中 S1 依赖 look-down 视角输入
- 若只存正前方 RGB/depth，离线重放无法严格复现闭环时的 S1 输入

#### 6.6 关键结论

本次 profiling 所依赖的真实闭环路径并不是 `internnav/agent/internvla_n1_agent.py`，而是 `habitat_vln_evaluator.py` 直接调用 DualVLN 模型。

这点非常重要，因为最初关于 realworld `step()` 中插计时点的思路是对的，但不适用于当前 Habitat VLN-CE 评估实现。当前仿真评估的大小脑拆解应以 evaluator 为准。

### 7. `scripts/eval/tools/build_r2r_subset.py`

修改目的：从完整 R2R `json.gz` 生成一个可用于 Habitat 的子集文件。

初版问题：

- 初版只保留了 `episodes`
- 丢失了顶层 `instruction_vocab`
- 导致 Habitat 的 `R2RVLN-v1` loader 在读取时触发 `KeyError: 'instruction_vocab'`

修复后逻辑：

- 读取完整 payload
- 保留所有顶层字段
- 仅替换 `payload["episodes"]`

当前已验证：

- `/root/backup/InternNav/data/vln_ce/raw_data/r2r/val_unseen_mini/val_unseen_mini.json.gz`
- 顶层同时包含 `episodes` 和 `instruction_vocab`

### 8. `scripts/eval/tools/benchmark_dualvln_replay.py`

修改目的：提供离线 replay benchmark 原型，用于在不依赖 Habitat 环境的情况下，只测 DualVLN 推理延迟和输出一致性。

当前脚本支持：

- 加载 replay manifest
- 重建多历史帧输入
- 统计 cold start 时间
- 统计 warm latency
- 统计 `tokens per second`
- 统计生成长度
- 统计 GPU 峰值显存
- 对 baseline action 做一致性比较
- 对 pixel goal 做误差比较

说明：

- 这仍是第一版离线 benchmark
- 后续可继续拆为 `S2-only` 与 `S2+S1` 两段 latency 分布

## 五、评估结果分析（基于 `logs_example/habitat/test_dual_system_mini`）

### 5.1 功能指标

8 个 mini episode 的闭环结果如下：

- Success: `0.875`
- SPL: `0.7751`
- Oracle Success: `0.875`
- Navigation Error: `1.0230`

说明：

- 在非常小的子集上，DualVLN 主链路已经跑通
- 功能指标没有出现系统级崩溃，说明补充的 profiling / replay 导出逻辑没有破坏主流程

### 5.2 延迟与行为统计

汇总均值如下：

- episode wall-clock: `44.21s`
- avg step wall-clock: `0.5086s`
- S2 avg: `0.5202s / call`
- S1 avg: `0.1530s / call`
- 平均每个 episode 的 S2 调用数：`22.25`
- 平均每个 episode 的 S1 调用数：`23.13`
- pixel goal ratio: `0.3751`
- discrete ratio: `0.6249`
- avg S1 steps per cycle: `8.596`

进一步汇总：

- 总 S2 调用数：`178`
- 总 S1 调用数：`185`
- 加权 pixel-goal ratio：`0.3708`
- 加权 discrete ratio：`0.6292`

### 5.3 对 realworld 假设的解释

原始怀疑是：

- realworld 的高延迟可能来自 S2 经常输出离散动作，而没有足够输出 pixel goal 进入 S1 循环

当前 Habitat 结果支持“这个方向基本成立，但需要更精确表述”：

1. **不是完全没有进入 S1。**
   当前 8 个 episode 中，S1 调用次数与 S2 调用次数处于同一量级，且平均每个 pixel-goal cycle 会连续执行约 `8.6` 步，说明双系统交接在仿真中是实际发生的。

2. **但 S2 的离散动作输出比例确实更高。**
   加权后离散输出约 `62.9%`，pixel-goal 输出约 `37.1%`。也就是说，System 2 仍更常直接控制行为，而不是把控制权交给 System 1 持续执行。

3. **S2 是更明显的模型级瓶颈。**
   单次平均耗时：
   - S2: `0.52s`
   - S1: `0.153s`

   仅从模型模块时间占比看：
   - `(S2 generate + S2 latent)` 约占模块时间的 `79.0%`
   - `S1 generate` 约占模块时间的 `21.0%`

4. **但闭环总延迟不只来自模型推理。**
   所有模型时间之和只占总 wall-clock 的约 `38.1%`，说明另外约 `61.9%` 的时间在以下环节中：
   - Habitat 环境步进
   - 取 look-down 观测的额外交互
   - 图像 / depth 预处理
   - Python 端流程组织与同步等待

因此，对于 realworld 的 `400ms` 问题，更准确的表述应是：

> 在仿真中，延迟主导项确实更接近 S2，而非 S1；同时 S2 直接输出离散动作的比例高于 pixel-goal，说明大小脑切换频率还不够高。但闭环端到端延迟并不只是模型推理本身，还包含大量交互和预处理开销。

## 六、当前未完成项

以下事项当前仍未完全闭环：

1. `benchmark_dualvln_replay.py` 还没有在真实 replay 数据上完成一轮 L20 baseline 跑测
2. 尚未将离线 benchmark 细分为：
   - `S2-only latency`
   - `S2+S1 latency`
   - cold / warm 更严格分桶
3. 尚未针对 realworld 版 `internvla_n1_agent.py` 做同口径 profiling

## 七、建议的下一步实验顺序

当前不建议直接先做量化，建议顺序如下。

### 第一步：先完成 replay 离线基线

原因：

- Habitat 闭环里只有约 `38%` 时间是模型本体
- 如果不先把模型和环境开销分离，直接做量化很难判断收益究竟落在模型还是系统交互

这一步应当先使用 `benchmark_dualvln_replay.py` 跑出：

- cold start
- warm p50 / p95 / max
- TPS
- 显存峰值
- baseline 一致性

### 第二步：优先测试 KV cache / 推理后端优化

优先级高于量化。

原因：

- S2 的主耗时来自 autoregressive 文本生成
- 这类瓶颈通常先受益于 cache / kernel / backend 优化
- 当前结果表明 S2 是主要模型级延迟来源，因此应首先优化 S2 路径

建议先做的单变量：

- KV cache 策略确认与强制开启
- `flash_attention_2` / vLLM / TensorRT-LLM / lmdeploy 等更快后端的可行性评估
- 减少 generation 长度上限与输出模板冗余

### 第三步：再做量化

量化仍值得做，但应放在第二优先级。

原因：

- 量化对显存与吞吐可能有效
- 但它往往带来输出一致性风险
- 当前更需要先确认 S2 纯推理链路的 latency ceiling

适合的量化方向：

- W8A8 / INT8 first
- 保持 visual encoder 和关键 cross-modal 模块尽量稳定
- 使用 replay baseline 检查 action / pixel-goal 一致性

### 第四步：将最优单变量放回 Habitat 闭环

指标仍然用：

- Success
- SPL
- Oracle Success
- Navigation Error
- episode wall-clock
- avg step wall-clock
- S2 / S1 模块耗时
- behavior ratio

只有这样，才能判断“推理变快了”是否伴随“闭环功能退化”。

## 八、今日产出清单

代码产出：

- mini/full smoke 两套 DualVLN Habitat 评估配置
- Habitat evaluator 级别的大小脑 profiling
- replay subset 导出
- replay 离线 benchmark 原型
- replay 离线 benchmark 强化版，支持分开统计 `S2 generate / S2 latent / S1 generate / total`，并输出更适合 L20 对比实验的结构化结果
- R2R subset 构造工具及 bug 修复
- 架构 walkthrough 文档
- 推理优化构思文档

结论产出：

- 明确确认 VLN-CE Habitat 评估不需要 `traj_data/r2r`
- 明确确认当前仿真中 S2 是主要模型级瓶颈
- 明确确认 S1 实际在发生，但 S2 离散输出比例仍偏高
- 明确确认闭环延迟中还有大量非模型开销

## 九、指标口径解释

为了避免后续复盘时混淆，下面对本次报告里出现的几个核心指标做统一解释。

### 9.1 `Success`

表示最终停下来的位置是否落在目标成功半径内。当前任务中它是最终完成率。

### 9.2 `SPL`

`Success weighted by Path Length`，表示既到达目标、又走得尽量高效的程度。它同时考虑：

- 成功与否
- 实际路径是否绕远

因此 `SPL` 通常比单纯的 `Success` 更严格。

### 9.3 `Oracle Success`

表示在整条轨迹中，是否曾经到达过目标区域。即使最终停点不好，只要中途到过，也可能是 `Oracle Success = 1`。

### 9.4 `Navigation Error`

表示 episode 结束时，当前位置与目标之间的距离。数值越低越好。

### 9.5 `wall-clock`

`wall-clock` 不是模型内部算了多久，而是现实世界时钟流逝的总时间。

例如：

- `episode_wall_clock_seconds`
  表示从 episode 开始到 episode 结束，整条闭环真实经过了多少秒

- `avg_step_wall_clock_seconds`
  表示平均每一步决策和执行花了多少秒

它包含的不只是模型推理，还包括：

- Habitat 环境步进
- look-down / look-up 取观测
- 图像和深度预处理
- Python 控制逻辑和同步等待

### 9.6 `s2_avg_seconds`

表示一次 System 2 文本决策平均花费的时间。它主要对应：

- prompt + 图像输入后
- 调用 `model.generate(...)`
- 得到离散动作或 pixel goal 文本输出

### 9.7 `s1_avg_seconds`

表示一次 System 1 局部轨迹生成平均花费的时间。它主要对应：

- 已经拿到 S2 latent
- 使用 look-down 视角
- 调用 `model.generate_traj(...)`

### 9.8 `pixel_goal_ratio` 和 `discrete_ratio`

它们反映 System 2 的输出行为分布：

- `pixel_goal_ratio`
  表示 S2 有多少次把控制权交给了 S1

- `discrete_ratio`
  表示 S2 有多少次直接给出离散动作而没有进入 S1

### 9.9 `avg_s1_steps_per_cycle`

表示每次进入 System 1 后，平均能连续执行多少步局部动作。这个指标越高，说明大小脑交接后 System 1 持续工作的时间越长。

## 十、2026-03-10 mini run 具体数据

基于 `logs_example/habitat/test_dual_system_mini` 的 8 个 episode，关键数据如下：

- Success: `0.875`
- SPL: `0.7751`
- Oracle Success: `0.875`
- Navigation Error: `1.0230`
- Episode wall-clock: `44.21s`
- Avg step wall-clock: `0.5086s`
- 平均每个 episode 的 S2 调用数：`22.25`
- 平均每个 episode 的 S1 调用数：`23.13`
- S2 平均单次耗时：`0.5202s`
- S1 平均单次耗时：`0.1530s`
- Pixel-goal ratio: `0.3751`
- Discrete ratio: `0.6249`
- Avg S1 steps per cycle: `8.596`

进一步汇总得到：

- 总 S2 调用数：`178`
- 总 S1 调用数：`185`
- 加权 pixel-goal ratio：`0.3708`
- 加权 discrete ratio：`0.6292`
- 模型模块时间中，S2 相关部分占约 `79.0%`
- 模型模块时间中，S1 相关部分占约 `21.0%`
- 模型模块总时间约占整条闭环 wall-clock 的 `38.1%`

## 十一、2026-03-10 新增文档

为了方便继续理解代码与设计后续实验，额外新增了两份 docs：

- `docs/dualvln_habitat_architecture.md`
  解释 Habitat 版 DualVLN 的大小脑数据流、关键代码入口和闭环结构

- `docs/dualvln_inference_optimization_plan.md`
  解释 KV cache / 推理后端 / 量化应如何分阶段推进

## 十二、离线 replay benchmark 补充结果

在补强后的 `benchmark_dualvln_replay.py` 上，对

- `logs_example/habitat/test_dual_system_mini/replay_subset/manifest_rank0.jsonl`

执行了一轮完整 replay benchmark。

### 12.1 这次 benchmark 实际覆盖范围

这次不是只跑 1 个 episode，而是跑了 replay 子集中的全部 8 个 episode，共 `697` 个 step。

分布如下：

- episode 1: `59` steps
- episode 2: `145` steps
- episode 3: `57` steps
- episode 13: `70` steps
- episode 14: `93` steps
- episode 15: `79` steps
- episode 25: `105` steps
- episode 26: `89` steps

### 12.2 为什么日志里几乎全是 `s1=0`

表面现象是：

- 进度日志里几乎所有 step 都显示
  - `kind=discrete`
  - `latent=0`
  - `s1=0`

但这并不等于 replay 数据里本来没有 pixel-goal。

进一步核对 manifest 可知：

- baseline `pixel_goal` steps: `496`
- baseline `discrete` steps: `201`

而 benchmark 输出中：

- `pixel_goal -> discrete`: `496`
- `discrete -> discrete`: `192`
- `discrete -> pixel_goal`: `9`

也就是说，这一版离线 replay benchmark 当前还没有严格复现闭环中的大小脑切换状态。

### 12.3 原因分析

根本原因主要有两点。

第一，manifest 保存的是“每个环境 step”，而不是“每次新的 S2 决策点”。

在闭环 evaluator 里，很多 step 实际处于同一个 ongoing pixel-goal / S1 cycle 中，此时 System 2 并不会重新做一次新的高层决策。但当前离线 benchmark 是把每个 step 都当成一个新的 S2 输入来独立重跑，因此天然更偏向离散动作输出。

第二，当前离线脚本还没有完整复现 evaluator 中的上下文状态，尤其包括：

- conversation continuation
- look-down continuation 分支
- 同一个 pixel-goal cycle 内的上下文演进

因此当前 replay benchmark 更适合解释为：

- 它可以给出稳定的“纯推理 baseline”
- 它可以用于比较 `num_history`、prompt 长度、`max_new_tokens`、KV cache、backend、量化等单变量对 latency 的影响
- 但它暂时不能直接等价于“完整复现闭环中的 S2 -> S1 切换”

### 12.4 当前这轮 replay benchmark 的使用方式

因此本轮 replay benchmark 的结论应该这样使用：

1. 把它当成 `S2-centric` 的纯推理测量工具
2. 用它做单变量延迟对比
3. 不把 `pixel_goal -> discrete` 的大量失配直接解释成模型已经退化
4. 真正闭环效果仍以 Habitat evaluator 的闭环指标为准

## 十三、`num_history / prompt / max_new_tokens` 缩减实验设计

本轮已形成一套建议实验矩阵，后续建议按单变量顺序执行。

### 13.1 `num_history` 实验

建议对比：

- `8`
- `4`
- `2`
- `0`

记录指标：

- `S2 generate p50 / p95 / max`
- `total_step p50 / p95 / max`
- `tokens_per_second`
- `output_kind_match_rate`
- `action_match_rate_all`
- `action_match_rate_pixel_goal`

### 13.2 prompt 长度实验

建议三档：

- `full`
- `short`
- `minimal`

记录指标：

- `S2 generate p50 / p95 / max`
- `generated_tokens_mean`
- `text_exact_match_rate`
- `output_kind_match_rate`

### 13.3 `max_new_tokens` 实验

建议对比：

- `128`
- `64`
- `32`
- `16`

记录指标：

- `generated_tokens_mean`
- `S2 generate p50 / p95 / max`
- `action_match_rate_all`
- `output_kind_match_rate`

### 13.4 推荐执行顺序

建议不要做全排列，而是：

1. 先扫 `num_history`
2. 在最佳 `num_history` 上扫 prompt 长度
3. 在前两轮最佳配置上扫 `max_new_tokens`

这样结果更容易解释。

## 十四、2026-03-10 新增 runner 与严格重放方案

### 14.1 新增统一 sweep runner

新增文件：

- `scripts/eval/tools/run_replay_benchmark_sweeps.py`

作用：

- 自动依次运行三组离线 replay benchmark：
  - `num_history`
  - `prompt_variant`
  - `max_new_tokens`
- 每组自动生成独立 summary
- 自动汇总成对比表：
  - `comparison_num_history.csv`
  - `comparison_prompt.csv`
  - `comparison_max_new_tokens.csv`
  - `comparison_all.json`

当前 runner 采用“单变量分组对比”而非全排列，目的是保证结果可解释。

### 14.2 benchmark 脚本新增可 sweep 参数

`benchmark_dualvln_replay.py` 已增加：

- `--prompt-variant`
- `--max-new-tokens`
- 读取 manifest 中已有的 `history_frame_indices`

同时输出 metadata 中也会记录：

- 当前 `num_history`
- 当前 `prompt_variant`
- 当前 `max_new_tokens`

### 14.3 新增严格重放设计文档

新增文件：

- `docs/offline_replay_strict_reproduction_plan.md`

该文档解释了：

1. 为什么当前第一版 replay benchmark 还不能严格复现闭环上下文
2. 如果要严格复现，为什么应该重新跑一遍 mini 闭环，导出 `replay_v2`
3. `replay_v2` 应额外保存哪些决策级状态
4. 为什么建议先做 `decision-only replay`，再做 `full-cycle replay`

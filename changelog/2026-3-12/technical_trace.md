# Changelog 2026-03-12 Technical Trace

本文记录 2026-03-12 基于 Habitat 闭环 `num_history` sweep 的结果分析，重点回答两个问题：

1. 离线 replay 中看起来很有效的 `num_history` 缩减，放回 Habitat 闭环后是否仍成立。
2. 如果闭环结果与离线结果不一致，下一步应该优先进入哪类优化。

## 一、实验背景

本次实验使用前一日新增的一键闭环 runner：

- `scripts/eval/tools/run_habitat_closed_loop_num_history_sweeps.py`

实际执行命令为：

```bash
python scripts/eval/tools/run_habitat_closed_loop_num_history_sweeps.py \
  --dataset mini \
  --num-history-values 4 2 1 0 \
  --output-root ./logs/habitat/closed_loop_num_history_sweeps \
  --max-eval-episodes 8 \
  --replay-num-episodes 8 \
  > ./logs/habitat/closed_loop_num_history_sweeps/run_mini_num_history.log 2>&1
```

输入数据：

- `mini` 模式
- `8` 个 episode

输出位置：

- `logs/habitat/closed_loop_num_history_sweeps/mini/comparison_num_history_closed_loop.csv`
- `logs/habitat/closed_loop_num_history_sweeps/mini/num_history_{4,2,1,0}/...`

对照 baseline：

- `logs/habitat/test_dual_system_mini/result.json`
- `logs/habitat/test_dual_system_mini/runtime_summary_rank0.json`

baseline 对应 `num_history=8`。

## 二、baseline 与闭环 sweep 对比

baseline（`num_history=8`）闭环结果：

- `Success = 1.0`
- `SPL = 0.911`
- `Oracle Success = 1.0`
- `Navigation Error = 0.916`
- `avg_step_wall_clock_seconds = 0.508`
- `s2_avg_seconds = 0.513`
- `pixel_goal_ratio = 0.370`
- `avg_s1_steps_per_cycle = 8.438`

本次 mini 闭环 sweep 汇总如下。

### 1. `num_history=4`

- `Success = 0.75`
- `SPL = 0.671`
- `Oracle Success = 0.75`
- `Navigation Error = 2.161`
- `avg_step_wall_clock_seconds = 0.534`
- `s2_avg_seconds = 0.359`
- `pixel_goal_ratio = 0.363`
- `avg_s1_steps_per_cycle = 8.604`

相对 baseline 的变化：

- `Success`: `-0.25`
- `SPL`: `-0.240`
- `Navigation Error`: `+1.245`
- `avg_step_wall_clock_seconds`: `+0.026`
- `s2_avg_seconds`: `-0.153`
- `pixel_goal_ratio`: `-0.0066`

### 2. `num_history=2`

- `Success = 0.0`
- `SPL = 0.0`
- `Oracle Success = 0.625`
- `Navigation Error = 4.111`
- `avg_step_wall_clock_seconds = 0.464`
- `s2_avg_seconds = 0.266`
- `pixel_goal_ratio = 0.215`
- `avg_s1_steps_per_cycle = 7.527`

相对 baseline 的变化：

- `Success`: `-1.0`
- `SPL`: `-0.911`
- `Navigation Error`: `+3.194`
- `avg_step_wall_clock_seconds`: `-0.044`
- `s2_avg_seconds`: `-0.247`
- `pixel_goal_ratio`: `-0.155`

### 3. `num_history=1`

- `Success = 0.0`
- `SPL = 0.0`
- `Oracle Success = 0.375`
- `Navigation Error = 5.436`
- `avg_step_wall_clock_seconds = 0.409`
- `s2_avg_seconds = 0.219`
- `pixel_goal_ratio = 0.043`
- `avg_s1_steps_per_cycle = 9.0`

### 4. `num_history=0`

- `Success = 0.0`
- `SPL = 0.0`
- `Oracle Success = 0.25`
- `Navigation Error = 6.721`
- `avg_step_wall_clock_seconds = 0.398`
- `s2_avg_seconds = 0.178`
- `pixel_goal_ratio = 0.039`
- `avg_s1_steps_per_cycle = 7.856`

## 三、关键结论

### 1. 离线 replay 的速度收益没有直接转化成闭环收益

前一日的 Level 1 replay baseline 中，`num_history` 缩减带来了很明显的 `S2 generate` 加速，看起来 `num_history=4` 是一个很有希望的低风险优化点。

但今天的闭环结果说明：

- `num_history=4` 虽然把 `s2_avg_seconds` 从 `0.513s` 降到了 `0.359s`
- 但整体 `avg_step_wall_clock_seconds` 不降反升，从 `0.508s` 上升到 `0.534s`
- 同时功能指标显著退化，`Success` 从 `1.0` 降到 `0.75`

这说明：

> 离线纯推理速度收益，并不能直接等价于闭环收益。

### 2. `num_history=4` 也不能直接采用

原本最有希望的 `num_history=4`，在闭环里虽然仍保留了接近 baseline 的 `pixel_goal_ratio` 和 `avg_s1_steps_per_cycle`，但功能指标已经明显下降：

- `Success` 下降 25 个点
- `SPL` 下降约 0.24
- `Navigation Error` 增加约 1.25 米

这个退化幅度已经不能视为“轻微波动”，因此当前不能把 `num_history=4` 直接当作部署候选。

### 3. `num_history<=2` 已经明显破坏大小脑切换

`num_history=2/1/0` 的共同特征是：

- `Success` 全部掉到 `0`
- `Navigation Error` 持续恶化
- `pixel_goal_ratio` 快速下降
- `discrete_ratio` 持续上升

尤其：

- `num_history=1/0` 时，`pixel_goal_ratio` 已经跌到 `4%` 左右

这说明当历史帧压得过低时，模型不仅变慢/变快的问题不再是重点，而是：

> 高层决策行为本身已经改变，系统越来越不愿意进入 `pixel_goal -> S1` 路径。

## 四、对系统瓶颈的再判断

这组闭环结果给出了一个更成熟的判断：

1. 历史帧长度确实会显著影响 `S2` 纯推理成本
2. 但历史帧也是模型维持导航能力的重要语义上下文
3. 因此“直接删上下文换速度”不是当前最优路线

换句话说，我们现在更适合追求的是：

- 尽量保持 `num_history=8` 的语义输入
- 同时通过更底层的推理优化去降低 `S2` 成本

## 五、下一步建议

基于今天的闭环 sweep，下一步建议顺序如下。

### 1. 不再继续压 `num_history`

当前没有必要继续围绕 `num_history=4/2/1/0` 做更多闭环实验，因为方向性已经很清楚：

- 速度提升存在
- 但功能退化过大

### 2. 正式进入“不删上下文”的优化阶段

现在更合理的下一步是：

1. 保持 `num_history=8`
2. 开始试验 `KV cache / backend` 一类优化

原因：

- 这类优化的目标是降低 `S2` 计算成本
- 但不直接删除视觉历史信息
- 更有希望拿到“时延下降而功能不掉”的结果

### 3. KV cache 方向的优先级已经被抬高

截至 2026-03-12，推荐优先级应调整为：

1. `KV cache / prefix reuse / backend`
2. 视情况再考虑更细粒度的 history 策略
3. 量化仍然放在后面

也就是说：

> 今天的闭环结果实际上为 KV cache 路线提供了更强的合理性，因为它说明我们需要的是“保留上下文、降低计算”，而不是“删掉上下文换速度”。

## 六、关于 `logs` 目录能否删除

今天额外对日志体积做了拆分。

整体体积分布：

- `/root/backup/InternNav/logs` 约 `70GB`
- 其中 `/root/backup/InternNav/logs/habitat/closed_loop_num_history_sweeps` 约 `68GB`
- `logs/habitat/test_dual_system_mini` 约 `1.8GB`

进一步看闭环 sweep：

- `num_history_0` 约 `22GB`
- `num_history_1` 约 `22GB`
- `num_history_2` 约 `22GB`
- `num_history_4` 约 `3.8GB`

真正的大头不是 summary json，而是每组导出的：

- `replay_subset`
- `replay_subset_v2`

对 `closed_loop_num_history_sweeps` 做 dry-run 清理后，若删除以下目录：

- `replay_subset`
- `replay_subset_v2`
- `check_sim_0`

预计可回收：

- `67.26GB`

### 1. 哪些可以删

如果你当前已经完成了闭环 `num_history` 分析，并且不准备再用这些 sweep 目录里的 replay 图像做离线重放，那么以下内容都可以删：

- `logs/habitat/closed_loop_num_history_sweeps/mini/num_history_*/replay_subset`
- `logs/habitat/closed_loop_num_history_sweeps/mini/num_history_*/replay_subset_v2`
- `logs/habitat/closed_loop_num_history_sweeps/mini/num_history_*/check_sim_0`

删掉后，仍会保留最关键的结论文件：

- `result.json`
- `runtime_summary_rank0.json`
- `progress.json`
- `comparison_num_history_closed_loop.csv`
- `run_mini_num_history.log`

也就是说：

- 分析结论会保留
- 巨大的逐帧回放资产会删除

### 2. 不建议立刻删什么

当前不建议立刻删除：

- `logs/habitat/test_dual_system_mini`

因为它仍然是：

- baseline 闭环结果
- baseline replay benchmark

的核心对照目录，体积也只有约 `1.8GB`，相对于 68GB 的 sweep 目录并不大。

### 3. 已新增的清理脚本

今天新增：

- `scripts/eval/tools/prune_habitat_logs.py`

dry-run 示例：

```bash
python scripts/eval/tools/prune_habitat_logs.py \
  --root /root/backup/InternNav/logs/habitat/closed_loop_num_history_sweeps
```

真正执行删除：

```bash
python scripts/eval/tools/prune_habitat_logs.py \
  --root /root/backup/InternNav/logs/habitat/closed_loop_num_history_sweeps \
  --apply
```

### 4. 后续如何避免再炸出几十 GB

今天还顺手修正了一点：

- `run_habitat_closed_loop_num_history_sweeps.py` 现在在 `--replay-num-episodes 0` 时，会自动关闭 replay 导出

所以以后若只是想做闭环功能对比，不需要 replay 子集，建议直接运行：

```bash
python scripts/eval/tools/run_habitat_closed_loop_num_history_sweeps.py \
  --dataset mini \
  --num-history-values 4 2 1 0 \
  --output-root ./logs/habitat/closed_loop_num_history_sweeps \
  --max-eval-episodes 8 \
  --replay-num-episodes 0
```

这样可以大幅减少磁盘占用。

## 七、KV cache 方向的实验设计与代码落点

今天已新增独立文档：

- `docs/dualvln_kv_cache_experiment_plan.md`

核心结论：

1. 当前 `use_cache=True` 只在单次 `generate()` 内生效
2. evaluator / replay benchmark / policy / realworld agent 里都显式传了 `past_key_values=None`
3. 因此当前实际上没有跨步 KV cache 复用

当前最重要的代码落点为：

- `internnav/habitat_extensions/vln/habitat_vln_evaluator.py`
- `scripts/eval/tools/benchmark_dualvln_replay.py`
- `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py`
- `internnav/agent/internvla_n1_agent_realworld.py`

推荐实验顺序为：

1. Phase A：先补输入统计，明确 prefill 压力
2. Phase B：只在 `look_down=True` 的 follow-up 路径尝试最小 cache 实验
3. Phase C：再考虑 instruction prefix reuse / partial prefill
4. Phase D：若上面风险高，直接转向 backend 路线

换句话说，今天之后的主要方向已经明确是：

- 保持 `num_history=8`
- 不再继续靠删上下文提速
- 转向“保留上下文、减少重复计算”的 KV cache / backend 优化

## 八、KV cache Phase A / B 已进入可执行状态

今天继续把 KV cache 路线从“设计方案”推进到“能直接跑实验”的状态。

### 1. Phase A：replay benchmark 新增输入压力统计

修改文件：

- `scripts/eval/tools/benchmark_dualvln_replay.py`
- `scripts/eval/tools/run_replay_benchmark_sweeps.py`

新增 summary 字段：

- `prefill.input_token_count`
- `prefill.input_image_count`
- `prefill.history_image_count`
- `prefill.prompt_char_count`
- `prefill.image_token_count`

新增 per-step details 字段：

- `input_token_count`
- `input_image_count`
- `history_image_count`
- `prompt_char_count`
- `image_token_count`

新增 sweep 表字段：

- `input_token_mean`
- `input_token_p95`
- `input_image_mean`
- `history_image_mean`
- `prompt_char_mean`
- `image_token_mean`

这一步的目标是明确：

> 当前 `S2 generate` 的成本更多来自文本 token，还是来自多图视觉 prefill。

### 2. Phase B：look-down follow-up 的实验型 KV cache

修改文件：

- `internnav/agent/internvla_n1_agent_realworld.py`
- `scripts/eval/tools/benchmark_dualvln_http_loopback.py`
- `scripts/realworld/http_internvla_server.py`

新增参数：

- `--kv-cache-mode disabled`
- `--kv-cache-mode lookdown_experimental`
- `--kv-cache-debug`

实现方式：

1. 默认仍是 `disabled`
2. 只有 `look_down=True` 的 follow-up 才尝试复用 cache
3. 先构造 full prompt
4. 再检查 full prompt 的 token 前缀是否与上一轮 `output_ids` 完全一致
5. 若完全一致，则只截取 delta token 和最后一张 look-down 图像进入 `generate()`
6. 若不一致、图像切片失败或运行时报错，则自动 fallback 到原始全量 prefill 路径

这个实现是刻意保守的。它的目标不是直接上线，而是先测两个问题：

- 当前 look-down continuation 到底有多少次具备可复用条件
- 即使可复用，收益是否足够大

### 3. 新增的 cache 统计项

`benchmark_dualvln_http_loopback.py` 的 summary 里，现在会在：

- `breakdown.kv_cache_stats`

输出这些统计：

- `lookdown_attempts`
- `lookdown_cache_hits`
- `lookdown_cache_fallbacks`
- `lookdown_prefix_mismatch`
- `lookdown_image_slice_error`
- `lookdown_cache_exceptions`
- `lookdown_delta_tokens_total`
- `lookdown_prefix_len_total`

含义分别是：

- 共有多少次 look-down continuation 尝试进入实验路径
- 其中多少次真的成功复用了 cache
- 多少次因为 token 前缀不一致或其他异常退回原始逻辑

## 九、现在怎么评估

### 1. Phase A：跑 replay baseline

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate habitat
cd /root/backup/InternNav
export TOKENIZERS_PARALLELISM=false
python -u scripts/eval/tools/benchmark_dualvln_replay.py \
  --manifest /root/backup/InternNav/logs/habitat/test_dual_system_mini/replay_subset/manifest_rank0.jsonl \
  --model-path /root/backup/InternNav/checkpoints/InternVLA-N1-DualVLN \
  --output /root/backup/InternNav/logs/habitat/test_dual_system_mini/kv_phase_a_replay_summary.json \
  --details-output /root/backup/InternNav/logs/habitat/test_dual_system_mini/kv_phase_a_replay_details.jsonl \
  --base-path /root/backup/InternNav/logs \
  --num-history 8 \
  --prompt-variant full \
  --max-new-tokens 128 \
  --verbose-every 20
```

优先看：

- `latency.s2_generate.p50 / p95`
- `prefill.input_token_count.mean / p95`
- `prefill.image_token_count.mean / p95`
- `prefill.history_image_count.mean`

### 2. Phase B：跑 HTTP loopback 的 AB 对比

baseline：

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate habitat
cd /root/backup/InternNav
export TOKENIZERS_PARALLELISM=false
python -u scripts/eval/tools/benchmark_dualvln_http_loopback.py \
  --manifest /root/backup/InternNav/logs/habitat/test_dual_system_mini/replay_subset_v2/manifest_rank0.jsonl \
  --model-path /root/backup/InternNav/checkpoints/InternVLA-N1-DualVLN \
  --output /root/backup/InternNav/logs/habitat/test_dual_system_mini/http_loopback_kv_disabled.json \
  --details-output /root/backup/InternNav/logs/habitat/test_dual_system_mini/http_loopback_kv_disabled.jsonl \
  --base-path /root/backup/InternNav/logs \
  --num-history 8 \
  --plan-step-gap 4 \
  --use-recorded-lookdown \
  --kv-cache-mode disabled \
  --verbose-every 20
```

实验模式：

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate habitat
cd /root/backup/InternNav
export TOKENIZERS_PARALLELISM=false
python -u scripts/eval/tools/benchmark_dualvln_http_loopback.py \
  --manifest /root/backup/InternNav/logs/habitat/test_dual_system_mini/replay_subset_v2/manifest_rank0.jsonl \
  --model-path /root/backup/InternNav/checkpoints/InternVLA-N1-DualVLN \
  --output /root/backup/InternNav/logs/habitat/test_dual_system_mini/http_loopback_kv_lookdown.json \
  --details-output /root/backup/InternNav/logs/habitat/test_dual_system_mini/http_loopback_kv_lookdown.jsonl \
  --base-path /root/backup/InternNav/logs \
  --num-history 8 \
  --plan-step-gap 4 \
  --use-recorded-lookdown \
  --kv-cache-mode lookdown_experimental \
  --verbose-every 20
```

评估时优先比较：

- `latency.server_model.p50 / p95`
- `latency.total_step.p50 / p95`
- `consistency.output_kind_match_rate`
- `breakdown.kv_cache_stats.lookdown_attempts`
- `breakdown.kv_cache_stats.lookdown_cache_hits`
- `breakdown.kv_cache_stats.lookdown_cache_fallbacks`
- `breakdown.kv_cache_stats.lookdown_prefix_mismatch`

### 3. 当前判断标准

1. 如果 `generated_tokens_mean` 很小，但 `input_token_count` 和 `image_token_count` 很高，则瓶颈更像 prefill，不像 decode。
2. 如果 `lookdown_attempts > 0`，但 `lookdown_cache_hits` 很低，而且主要失败原因是 `prefix_mismatch`，说明现有 chat template 还不适合直接续 cache。
3. 如果 `lookdown_cache_hits` 有一定比例，且 `server_model.p50/p95` 明显下降，同时输出一致性没有明显变差，这条路线才值得继续往主链推进。

## 十、关于 `replay_subset_v2` 没生成的原因

今天又遇到一个容易混淆的点：

- `logs/habitat/test_dual_system_mini/replay_subset_v2/manifest_rank0.jsonl` 不存在
- 重新执行 `python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_mini_cfg.py`
- 日志显示 `Eval Epoch 0 Rank 0: 0it [00:00, ?it/s]`

原因已经确认，不是 replay_v2 代码没生效，而是：

1. `test_dual_system_mini` 目录里已经有旧的 `progress.json`
2. `HabitatEnv` 在加载 episode 时会读取 `progress.json`
3. 对已经完成的 `(scene_id, episode_id)` 会直接跳过
4. 因此这次 rerun 根本没有真正进入 episode loop
5. 自然也就不会重新导出 `replay_subset_v2`

也就是说：

> 这次 `0it` 不是运行异常，而是 resume 机制把现有 8 个 episode 全部视为已完成。

为避免覆盖旧结果，今天补了一个新 config：

- `scripts/eval/configs/habitat_dual_system_mini_replay_v2_cfg.py`

它会把输出写到：

- `./logs/habitat/test_dual_system_mini_replay_v2`

因此现在要重新生成 replay_v2，正确命令应为：

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate habitat
cd /root/backup/InternNav
python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_mini_replay_v2_cfg.py
```

生成完成后，Phase B 应使用新的 manifest：

```bash
python -u scripts/eval/tools/benchmark_dualvln_http_loopback.py \
  --manifest /root/backup/InternNav/logs/habitat/test_dual_system_mini_replay_v2/replay_subset_v2/manifest_rank0.jsonl \
  --model-path /root/backup/InternNav/checkpoints/InternVLA-N1-DualVLN \
  --output /root/backup/InternNav/logs/habitat/test_dual_system_mini_replay_v2/http_loopback_kv_disabled.json \
  --details-output /root/backup/InternNav/logs/habitat/test_dual_system_mini_replay_v2/http_loopback_kv_disabled.jsonl \
  --base-path /root/backup/InternNav/logs \
  --num-history 8 \
  --plan-step-gap 4 \
  --use-recorded-lookdown \
  --kv-cache-mode disabled \
  --verbose-every 20
```

## 十一、HTTP loopback baseline 结果分析

今天已经拿到一份 Phase B 的 baseline：

- `logs/habitat/test_dual_system_mini_replay_v2/http_loopback_kv_disabled.json`

注意，这一轮只有：

- `kv_cache_mode = disabled`

还没有拿到：

- `kv_cache_mode = lookdown_experimental`

因此今天能下的结论，是“目标路径是否合理”，还不是“cache 是否有效”。

### 1. 整体结果

- `num_steps = 645`
- `num_episodes = 8`
- `cold_start_load_seconds = 4.41`
- `gpu_peak_memory_mb = 26428.87`
- `total_step mean = 0.544s`
- `total_step p50 = 0.263s`
- `total_step p95 = 1.999s`
- `server_model mean = 0.487s`
- `server_model p50 = 0.201s`
- `server_model p95 = 1.943s`
- `output_kind_match_rate = 0.826`
- `discrete_action_match_rate = 0.916`
- `pixel_goal_l2_mean = 73.97`

### 2. 长尾几乎全部来自 look-down continuation

summary 中：

- `lookdown_used_steps = 93`

进一步按 details jsonl 拆开：

`lookdown_used = true` 的 93 步：

- `total p50 = 1.993s`
- `model p50 = 1.932s`
- `model p95 = 1.950s`

`lookdown_used = false` 的 552 步：

- `total p50 = 0.261s`
- `model p50 = 0.201s`
- `model p95 = 0.579s`

这说明：

> HTTP loopback 的主要长尾几乎全部由 look-down continuation 贡献。

换句话说，我们把 KV cache Phase B 的第一刀切在：

- `look_down=True` follow-up

这个选择是对的，因为这正是当前整条链路里最明显的慢路径。

### 3. baseline 还证明了 plan-step-gap 确实在起作用

`kv_cache_stats` 中：

- `s2_calls_total = 282`
- `lookdown_attempts = 93`
- `lookdown_cache_disabled = 93`

而总 step 数是：

- `645`

这说明并不是每个 step 都重新进 S2，`plan_step_gap` 仍在生效。当前慢的不是“所有 step 都很慢”，而是：

- 少数 look-down continuation 非常慢

### 4. 当前阶段的结论

截至今天，Phase B baseline 已经给出两个明确信号：

1. 选 look-down continuation 作为第一条 cache 路径是合理的
2. 这条路径如果能成功命中 cache，理论上最有希望吃掉当前 loopback 的长尾

但今天还不能回答：

- cache hit rate 有多高
- prefix mismatch 是否会很多
- latency 是否真的能降

这些还需要下一轮：

- `kv_cache_mode = lookdown_experimental`

的 AB 对比结果才能确认。

## 十二、`lookdown_experimental` AB 对比结果

今天已经拿到了：

- `http_loopback_kv_disabled.json`
- `http_loopback_kv_lookdown.json`

两份完整结果。

### 1. 表面结果：latency 和一致性几乎不变

`disabled`：

- `total_step p50 = 0.2629s`
- `total_step p95 = 1.9987s`
- `server_model p50 = 0.2011s`
- `server_model p95 = 1.9435s`
- `output_kind_match_rate = 0.8264`

`lookdown_experimental`：

- `total_step p50 = 0.2606s`
- `total_step p95 = 2.0015s`
- `server_model p50 = 0.1987s`
- `server_model p95 = 1.9469s`
- `output_kind_match_rate = 0.8264`

从整体指标看，两者几乎没有差异。

### 2. 真正原因：所有 cache 尝试都异常并 fallback

`lookdown_experimental` 的 `kv_cache_stats` 为：

- `lookdown_attempts = 93`
- `lookdown_cache_ready = 93`
- `lookdown_cache_hits = 93`
- `lookdown_cache_exceptions = 93`
- `lookdown_used_cache_total = 0`

这说明：

1. 93 次 look-down continuation 全部进入了实验路径
2. 93 次 token 前缀都对齐成功
3. 93 次 delta token / 最后一张图像切分都成功
4. 但 93 次真实 cache generate 全部抛异常
5. 所以最终 93 次都 fallback 回了原始全量 prefill 路径

因此这次实验的真正结论不是：

- cache 没有收益

而是：

> 当前这条多模态 `generate(..., past_key_values=...)` 接法并不能直接工作。

### 3. 这轮实验排除了什么

这次已经基本排除掉以下怀疑：

- replay_v2 字段不够，导致无法构造 look-down continuation
- token 前缀对不齐，导致无法识别增量输入
- 只取最后一张 look-down 图像时，图像 token 切片逻辑完全错误

因为如果这些地方有问题，就不会出现：

- `lookdown_cache_ready = 93`
- `lookdown_cache_hits = 93`

而当前真正失败的是更底层的一层：

- 带 `past_key_values` 的多模态 `generate()` 调用本身

### 4. 对下一步路线的影响

这轮 AB 对比使得当前策略需要再收紧：

1. 当前不宜继续在“同一 Hugging Face 多模态 generate 接口上直接续 cache”这条路线上投入过多
2. 应优先补充异常日志，确认到底是 `cache_position`、vision token、还是 `generate()` 接口本身不兼容
3. 如果异常栈证明是 API 级限制，则下一步优先级应转向：
   - prefix reuse / partial prefill
   - 更快的 S2 backend

换句话说，Phase B 这轮实验虽然没有拿到 latency 收益，但它不是失败，而是一次高价值排除：

> 问题不在高层调度逻辑，而在底层多模态 cache 接口本身。

## 十三、异常日志落盘与首错调试脚本

为了把上面的判断真正坐实，今天继续补了两类调试能力。

### 1. 异常日志落盘

修改文件：

- `internnav/agent/internvla_n1_agent_realworld.py`
- `scripts/eval/tools/benchmark_dualvln_http_loopback.py`

新增内容：

- `kv_cache_stats.exception_type_counts`
- `kv_cache_stats.exception_samples`
- per-step details 中的 `kv_cache_event`

其中 `exception_samples` 会记录：

- `exception_type`
- `exception_message`
- `traceback`
- `full_input_tokens`
- `delta_input_tokens`
- `full_image_tokens`
- `delta_image_tokens`
- `episode_idx`

这意味着下一轮 loopback 不再只是告诉我们：

- 有多少次异常

而是能直接告诉我们：

- 具体是什么异常
- 发生时的输入规模
- 对应 traceback 是什么

### 2. stop-on-first-exception debug runner

新增脚本：

- `scripts/eval/tools/debug_dualvln_http_kv_cache.py`

作用：

1. 复用现有 HTTP loopback 路径
2. 一旦遇到首个 cache exception 立刻停止
3. 把对应的 scene/episode/record/step 和完整异常信息写到一个独立 json
4. 避免每次为了拿异常栈都重跑完整个 645 step benchmark

推荐命令：

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate habitat
cd /root/backup/InternNav
export TOKENIZERS_PARALLELISM=false
python -u scripts/eval/tools/debug_dualvln_http_kv_cache.py \
  --manifest /root/backup/InternNav/logs/habitat/test_dual_system_mini_replay_v2/replay_subset_v2/manifest_rank0.jsonl \
  --model-path /root/backup/InternNav/checkpoints/InternVLA-N1-DualVLN \
  --output /root/backup/InternNav/logs/habitat/test_dual_system_mini_replay_v2/http_loopback_kv_debug_first_exception.json \
  --details-output /root/backup/InternNav/logs/habitat/test_dual_system_mini_replay_v2/http_loopback_kv_debug_first_exception.jsonl \
  --base-path /root/backup/InternNav/logs \
  --num-history 8 \
  --plan-step-gap 4 \
  --use-recorded-lookdown \
  --kv-cache-mode lookdown_experimental \
  --max-steps 50
```

### 3. 这一步的意义

到这一步，下一轮实验的目标已经很明确：

- 不是继续重复证明“有异常”
- 而是抓到第一条完整异常栈，确认问题是：
  - vision token 接法
  - cache position / rope
  - 还是 HF `generate()` 本身的接口限制

### 4. 为什么终端上只看到 4 个 output 就结束

这次 debug runner 的表现是：

- 终端只打印了少量几次 `output ...`
- 没有直接把 traceback 打到终端
- 程序很快结束

这是符合脚本设计的，因为：

1. `debug_dualvln_http_kv_cache.py` 会在捕获到首个 cache exception 后
2. 把异常写进 json
3. 然后正常返回，不会继续跑剩余 step

当前这份 debug json 里：

- `steps_processed = 2`
- `first_exception.record_id = 1`
- `first_exception.step_id = 1`

同时：

- `s2_calls_total = 4`

这里的 `4` 并不是“处理了 4 个 replay step”，而是：

1. 初始化 warmup 的 1 次 S2
2. 第 0 条 replay step 的 1 次 S2
3. 第 1 条 replay step 的 1 次 S2
4. 第 1 条 replay step 中 look-down follow-up 的 1 次 S2

所以“终端上看到 4 个 output”与 “json 里只处理了 2 个 replay step” 是一致的。

## 十四、首错 traceback 的直接结论

今天已经拿到第一条首错 traceback，核心错误是：

```text
IndexError: index -1 is out of bounds for dimension 0 with size 0
```

位置在：

- `transformers/generation/utils.py`
- `_cache_dependant_input_preparation`

直接原因是：

- 传入了 `past_key_values`
- 但没有显式传 `cache_position`
- HF 默认按 `arange(past_length, input_len)` 推导
- 当 `past_length > delta_input_len` 时，得到空的 `cache_position`
- 随后访问 `cache_position[-1]` 直接报错

这一步已经据此补了代码：

- 在实验型 `generation_inputs` 中显式加入 `cache_position`

同时，这次 traceback 还暴露出一个更重要的结构性信号。Qwen2.5-VL 的：

- `prepare_inputs_for_generation()`

里有这样一段逻辑：

```python
if cache_position[0] != 0:
    model_inputs["pixel_values"] = None
    model_inputs["pixel_values_videos"] = None
```

这意味着只要走的是 continuation cache 路径：

- 新的图像输入会被直接丢弃

换句话说，当前 HF/Qwen2.5-VL 的标准 `generate + past_key_values` 接口，本身就不适合：

- look-down 后追加一张新图像再继续生成

### 当前结论进一步收敛为

1. 高层前缀对齐逻辑没有问题
2. delta token / delta image 切分逻辑也已基本走通
3. 当前真正的阻塞点已经非常靠近底层接口：
   - 一是 `cache_position`
   - 二是 continuation 时 vision input 被主动置空

因此下一步即使修完 `cache_position`，也很可能仍然会被：

- “continuation 不接受新图像”

这个接口限制卡住。

## 十五、prefix reuse 方案落点与最小实验版

基于前面的接口分析，现在更合理的下一条路线是：

- 不再尝试“带新图像的 continuation cache”
- 转而先分析“固定文本前缀能不能复用”

### 1. 为什么 prefix reuse 比继续试 `past_key_values` 更合理

当前 `past_key_values + multimodal generate` 的问题已经不是简单 bug，而是接口假设层面的冲突：

1. HF/Qwen2.5-VL continuation 默认要求 continuation 阶段主要是文本延续
2. 当前 look-down follow-up 需要 continuation 阶段接收一张新图像
3. Qwen2.5-VL 在 `cache_position[0] != 0` 时会主动把 `pixel_values` 置空

所以如果继续硬试这条路，下一步基本就不是“继续实验”，而是在重写：

- `prepare_inputs_for_generation`
- vision token 注入逻辑
- cache position / rope 行为

从工程投入比看，这条路线已经明显变重。

而 `prefix reuse` 的思路是：

- 不碰 continuation 语义
- 不要求 cache 阶段引入新图像
- 先看看 instruction / 模板前缀本身是否值得复用

### 2. prefix reuse 的实际代码落点

当前最关键的三个落点是：

- `internnav/habitat_extensions/vln/habitat_vln_evaluator.py`
- `internnav/agent/internvla_n1_agent_realworld.py`
- `internnav/model/basemodel/internvla_n1/internvla_n1.py`

分工理解如下：

1. evaluator / realworld agent 负责把当前 prompt 拆成：
   - `static_prefix_text`
   - `dynamic_suffix_text`
   - `current_images`
2. model 层如果未来做 runtime 版，需要提供：
   - 不完全依赖黑盒 `generate()`
   - 能显式控制“前缀先编码、动态部分后拼接”的入口

今天还没有做到 runtime 版，只先落了最小实验版。

### 3. 最小实验版脚本

新增脚本：

- `scripts/eval/tools/analyze_dualvln_prefix_reuse.py`

它会基于 `replay_subset_v2` 的真实 S2 决策上下文，统计：

- `full_input_tokens`
- `chat_text_tokens`
- `image_token_count`
- `static_prefix_tokens`
- `dynamic_suffix_tokens`
- `static_prefix_ratio_full`
- `static_prefix_ratio_text`
- `common_prefix_vs_prev_episode`
- `common_prefix_vs_prev_decision_group`

推荐命令：

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate habitat
cd /root/backup/InternNav
python -u scripts/eval/tools/analyze_dualvln_prefix_reuse.py \
  --manifest /root/backup/InternNav/logs/habitat/test_dual_system_mini_replay_v2/replay_subset_v2/manifest_rank0.jsonl \
  --model-path /root/backup/InternNav/checkpoints/InternVLA-N1-DualVLN \
  --output /root/backup/InternNav/logs/habitat/test_dual_system_mini_replay_v2/prefix_reuse_analysis.json \
  --details-output /root/backup/InternNav/logs/habitat/test_dual_system_mini_replay_v2/prefix_reuse_analysis.jsonl \
  --base-path /root/backup/InternNav/logs
```

### 4. 这个最小实验版到底回答什么

它不直接让模型更快，它回答的是：

1. 静态文本前缀在“真实多模态输入”里到底占多大比例
2. 如果只复用文本前缀，不复用图像 token，理论上最多能省多少 prefill
3. 这条路线有没有足够大的理论收益，值得继续推进 runtime 版

### 5. 分析脚本的容错修正

首次运行 `analyze_dualvln_prefix_reuse.py` 时，又暴露出一个 replay_v2 数据质量问题：

- 某些 `decision_input_image_paths` 会引用到未实际落盘的历史帧

例如：

- `record_0012_step_0011_rgb.png`

在 manifest 里被引用，但磁盘上并不存在。

为避免整个分析被这种个别缺口中断，今天把脚本改成：

1. 先用 manifest 中实际存在的 `rgb_path` 建索引
2. 尝试按原路径读取图像
3. 如果某条记录缺任意历史图，则整条记录跳过，不再把“缺图后的残缺输入”喂给 processor
4. 同时把缺失数量写进：
   - `data_quality.records_with_missing_images`
   - `data_quality.total_missing_images`

并且在 details jsonl 中对被跳过的记录写入：

- `skipped_reason = missing_decision_images`

因此当前这版 prefix reuse 分析脚本已经具备：

- 对 replay_v2 个别历史图缺失的容错能力
- 对“缺图导致图像占位符数量不一致”的记录进行整条跳过
- 在 summary 中显式记录数据质量：
  - `data_quality.records_with_missing_images`
  - `data_quality.total_missing_images`
- 在 details jsonl 中给跳过记录写入：
  - `skipped_reason = missing_decision_images`

## 十六、prefix reuse 分析结果

在修复缺图容错后，已完成一轮 prefix reuse 理论空间分析：

- summary: [prefix_reuse_analysis.json](/root/backup/InternNav/logs/habitat/test_dual_system_mini_replay_v2/prefix_reuse_analysis.json)
- details: [prefix_reuse_analysis.jsonl](/root/backup/InternNav/logs/habitat/test_dual_system_mini_replay_v2/prefix_reuse_analysis.jsonl)

关键数据如下：

- `metadata.num_records = 167`
- 实际完成 processor 对齐并纳入统计的记录数：
  - `prefix_reuse.full_input_tokens.count = 145`
- 因决策图缺失而被跳过：
  - `data_quality.records_with_missing_images = 22`
  - `data_quality.total_missing_images = 32`

整体 token 结构：

- `full_input_tokens mean = 3494.10`
- `chat_text_tokens mean = 145.48`
- `image_token_count mean = 13428.83`
- `static_prefix_tokens mean = 103.50`
- `dynamic_suffix_tokens mean = 3390.61`

prefix reuse 最关键的两个比例：

- `static_prefix_ratio_full mean = 0.0386`
- `static_prefix_ratio_text mean = 0.7153`

这两个数放在一起解释非常关键：

1. 从“纯文本 prompt”视角看，静态 instruction/template 前缀高度重复，平均占文本 token 的约 `71.5%`
2. 但从“真实多模态输入”视角看，这段静态文本前缀只占总输入 token 的约 `3.86%`
3. 说明当前 System 2 的 prefill 主成本不是文本前缀，而是图像 token

按决策类型拆开看：

- `groups.normal_decision.static_prefix_ratio_full mean = 0.0456`
- `groups.lookdown_followup.static_prefix_ratio_full mean = 0.0267`

这意味着：

1. 普通决策步里，纯文本 prefix reuse 仍然只有中低个位数收益空间
2. 真正的慢路径 `lookdown_followup` 里，这个理论占比反而更低
3. 也就是说，越是当前最慢、最值得优化的路径，单纯文本前缀复用越难带来决定性收益

截至目前的技术判断因此收敛为：

1. 纯文本 prefix reuse 不是没有价值，而是收益上限偏低
2. 它更适合作为低风险辅助手段，而不是当前主攻方向
3. 如果后续继续做 prefill 优化，应该优先考虑：
   - 更快的 System 2 backend
   - 更广义的 partial prefill / multimodal prefill 优化
   - 其次才是纯文本 prefix reuse

## 十七、vLLM 路线启动

基于以上结论，今天正式开始把下一条 backend 路线具体化，也就是：

- vLLM

这里的判断不是“整套 DualVLN 直接迁到 vLLM”，而是：

- 先做 S2-only 的 vLLM 尝试

### 1. 为什么这条路线值得试

当前最不建议继续主攻的路线是：

- `past_key_values + multimodal generate`

因为它已经暴露出较强的底层接口冲突，尤其是：

- continuation 阶段需要“复用旧 cache + 再喂一张新图”

而当前 HF/Qwen2.5-VL 这一层对这种输入模式并不友好。

相反，vLLM 更适合承接的，是：

- System 2 的文本生成提速
- KV/prefix 管理
- 更成熟的 serving/backend 优化

### 2. 为什么不能直接整套 DualVLN 切过去

检查 [config.json](/root/backup/InternNav/checkpoints/InternVLA-N1-DualVLN/config.json) 后确认：

- `model_type = internvla_n1`
- `architectures = ["InternVLAN1ForCausalLM"]`

同时 `model.safetensors.index.json` 里还存在大量额外模块：

- `model.latent_queries`
- `model.rgb_model.*`
- `model.memory_encoder.*`
- `model.rgb_resampler.*`
- `model.cond_projector.*`

这说明：

1. 当前 checkpoint 不是标准 `Qwen2.5-VL`
2. 但其 S2 主干仍然高度依赖 Qwen2.5-VL 的视觉语言主干
3. 因此最合理的切法是：
   - 只替换 S2
   - 保留 S1 的当前 PyTorch 路径

### 3. 本次新增脚本

新增：

- [check_dualvln_vllm_feasibility.py](/root/backup/InternNav/scripts/eval/tools/check_dualvln_vllm_feasibility.py)
- [benchmark_dualvln_s2_backends.py](/root/backup/InternNav/scripts/eval/tools/benchmark_dualvln_s2_backends.py)

其中：

`check_dualvln_vllm_feasibility.py` 负责：

- 读取 checkpoint config
- 统计是否存在大量 DualVLN/System1 额外模块
- 生成一个 S2-only 的 patched config view

patched view 的策略是：

1. 通过 symlink 复用原始权重文件
2. 只重写 `config.json`
3. 将：
   - `model_type -> qwen2_5_vl`
   - `architectures -> ["Qwen2_5_VLForConditionalGeneration"]`

用于 S2-only vLLM 尝试

`benchmark_dualvln_s2_backends.py` 负责：

- 用同一份 replay manifest 跑：
  - `--backend hf`
  - `--backend vllm`
- 只测 S2 文本生成
- 输出：
  - `cold_start_load_seconds`
  - `s2_generate p50/p95`
  - `tokens_per_second`
  - `output_kind_match_rate`
  - `text_exact_match_rate`
  - `discrete_action_match_rate`

### 4. 当前状态

截至本记录：

1. 两个脚本均已完成并通过 `py_compile`
2. 本地环境当前尚未安装 `vllm`
3. 因此还没有跑出真实的 HF vs vLLM 数值对比

所以当前阶段的完成度应理解为：

- 已完成 vLLM 实验入口与 checkpoint 兼容性准备
- 下一步等 `vllm` 安装后，直接跑 S2-only 的 replay A/B

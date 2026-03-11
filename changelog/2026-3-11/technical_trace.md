# Changelog 2026-03-11 Technical Trace

本文记录 2026-03-11 围绕 DualVLN 离线 replay、strict reproduction 方案以及实验执行命令的整理结果，目标是给后续 traceback、复现实验和 replay_v2 开发提供一份可以直接照着执行的 runbook。

## 一、今天的核心结论

今天需要先把一个边界讲清楚：

1. 当前已经能稳定跑通的是两条链路：
   - Habitat 闭环评估 + profiling + replay 导出
   - 基于导出 replay 的 Level 1 纯推理 benchmark 与单变量 sweep
2. 当前还不能直接跑的是 strict replay_v2：
   - `decision-only replay`
   - `full-cycle replay`
   - `HTTP loopback replay`
3. 原因不是文档缺失，而是 `replay_v2` 仍停留在设计层，当前 manifest 还没有保存足够的决策级上下文。

因此，今天的推荐执行顺序不是直接追 strict replay，而是：

1. 重新跑一遍 mini 闭环，生成一份新的 replay 子集
2. 在这份 replay 上先跑当前可用的 Level 1 纯推理 baseline
3. 再跑 `num_history / prompt / max_new_tokens` 三组单变量 sweep
4. 根据 sweep 结果决定先做 `replay_v2`，还是先做 `KV cache / backend` 优化

## 二、今天可以直接执行的命令

### 1. 重新跑 mini 闭环

这条命令会重新跑 8 个 mini episode，并在输出目录中生成：

- `result.json`
- `progress.json`
- `runtime_rank0.jsonl`
- `runtime_summary_rank0.json`
- `replay_subset/manifest_rank0.jsonl`

命令如下：

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate habitat
cd /root/backup/InternNav
python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_mini_cfg.py
```

对应输出目录：

```text
/root/backup/InternNav/logs/habitat/test_dual_system_mini
```

### 2. 如果不想依赖 mini json，而是直接用完整 `val_unseen` 做 8-episode smoke

命令如下：

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate habitat
cd /root/backup/InternNav
python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_smoke_cfg.py
```

对应输出目录：

```text
/root/backup/InternNav/logs/habitat/test_dual_system_smoke
```

### 3. 闭环完成后，运行当前可用的离线 replay baseline

这一步是 Level 1 纯推理 benchmark。它不会严格复现闭环状态，但会给出：

- cold start
- warm latency
- generated tokens
- tokens per second
- GPU peak memory
- 与 baseline 的输出一致性

推荐基线命令如下：

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate habitat
cd /root/backup/InternNav
export TOKENIZERS_PARALLELISM=false
python -u scripts/eval/tools/benchmark_dualvln_replay.py \
  --manifest /root/backup/InternNav/logs/habitat/test_dual_system_mini/replay_subset/manifest_rank0.jsonl \
  --model-path /root/backup/InternNav/checkpoints/InternVLA-N1-DualVLN \
  --output /root/backup/InternNav/logs/habitat/test_dual_system_mini/replay_summary.json \
  --base-path /root/backup/InternNav/logs \
  --num-history 8 \
  --prompt-variant full \
  --max-new-tokens 128 \
  --verbose-every 20
```

如果是用 `logs_example` 下已有的示例 replay，则只把 `--manifest`、`--output` 和 `--base-path` 三个路径改成 `logs_example` 即可。

### 4. 如果想先做一个 10-step 的快速 sanity check

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate habitat
cd /root/backup/InternNav
export TOKENIZERS_PARALLELISM=false
python -u scripts/eval/tools/benchmark_dualvln_replay.py \
  --manifest /root/backup/InternNav/logs/habitat/test_dual_system_mini/replay_subset/manifest_rank0.jsonl \
  --model-path /root/backup/InternNav/checkpoints/InternVLA-N1-DualVLN \
  --output /root/backup/InternNav/logs/habitat/test_dual_system_mini/replay_summary_debug.json \
  --base-path /root/backup/InternNav/logs \
  --num-history 8 \
  --prompt-variant full \
  --max-new-tokens 128 \
  --max-steps 10 \
  --verbose-every 1
```

### 5. 跑三组单变量 sweep

统一 runner 的命令如下：

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate habitat
cd /root/backup/InternNav
export TOKENIZERS_PARALLELISM=false
python scripts/eval/tools/run_replay_benchmark_sweeps.py \
  --manifest /root/backup/InternNav/logs/habitat/test_dual_system_mini/replay_subset/manifest_rank0.jsonl \
  --model-path /root/backup/InternNav/checkpoints/InternVLA-N1-DualVLN \
  --output-dir /root/backup/InternNav/logs/habitat/test_dual_system_mini/sweeps \
  --base-path /root/backup/InternNav/logs \
  --verbose-every 20
```

输出结果包括：

- `comparison_num_history.csv`
- `comparison_prompt.csv`
- `comparison_max_new_tokens.csv`
- `comparison_all.json`

## 三、今天建议的实际执行顺序

最推荐的 3-11 执行顺序如下：

### 路线 A：继续用 mini 闭环

1. 重新跑 mini 闭环
2. 检查 `result.json`、`progress.json`、`runtime_summary_rank0.json`
3. 确认新的 `replay_subset/manifest_rank0.jsonl` 已生成
4. 跑一轮 `benchmark_dualvln_replay.py` 基线
5. 跑 `run_replay_benchmark_sweeps.py`
6. 汇总哪一组变量最敏感，再决定是否先做 `KV cache / backend`

### 路线 B：切到 smoke 闭环

如果希望尽量贴近主分支完整数据，而不是继续依赖 `val_unseen_mini`，则：

1. 用 `habitat_dual_system_smoke_cfg.py` 重跑 8 个 episode
2. 后续 benchmark 和 sweep 的命令完全相同，只需把 manifest 和 output 路径替换为 `test_dual_system_smoke`

## 四、关于 strict replay 的整合说明

今天把 `docs/offline_replay_strict_reproduction_plan.md` 中和执行链最相关的部分整合如下。

### 1. 当前第一版 replay benchmark 为什么不能严格复现闭环

原因有四个：

1. 当前 manifest 保存的是每个环境 step，而不是每个“新 S2 决策点”
2. manifest 没有完整保存 conversation state
3. manifest 没有显式标记哪些 step 是新的 S2 决策，哪些只是 S1 连续执行
4. 离线脚本是重新构造 prompt，而不是复用闭环当时真正喂给模型的精确 prompt 和图像列表

因此，当前 replay benchmark 的定位应是：

- Level 1 纯推理 baseline
- 用于做 latency 和一致性横向对比

而不是：

- 严格等价于闭环的大小脑状态重放

### 2. strict replay_v2 需要什么

如果要做 strict replay_v2，下一步应该在 `habitat_vln_evaluator.py` 的 replay 导出中补充这些字段：

- `is_new_s2_decision`
- `decision_id`
- `pixel_goal_cycle_id`
- `cycle_step_offset`
- `is_lookdown_followup`
- `prompt_text`
- `input_image_paths`
- `conversation_messages`
- `s2_output_text`
- `s2_output_kind`
- `s2_output_pixel`
- `s2_action_seq`
- `s2_generated_token_count`
- `s1_action_seq`
- `s1_action_index_in_cycle`

### 3. strict replay 的推荐开发顺序

推荐顺序不是直接做最复杂的 full-cycle，而是：

1. 继续保留当前 Level 1 benchmark
2. 修改 evaluator，导出 `replay_v2`
3. 先做 `decision-only replay`
4. 稳定后再做 `full-cycle replay`
5. 如果要贴近 realworld 部署，再做 `HTTP loopback replay`

### 4. HTTP loopback replay 的定位

`http_internvla_client.py` 和 `http_internvla_server.py` 的模拟是有价值的，但它不是 Level 1 的替代，而应作为 Level 3 fidelity：

- Level 1：纯模型 replay
- Level 2：严格上下文 replay
- Level 3：HTTP loopback replay

这样三层可以同时保留：

- Level 1 负责测最纯粹的推理 latency
- Level 2 负责测大小脑切换是否被正确复现
- Level 3 负责估算真实部署里的通信与序列化开销

## 五、今天的结论与下一步

今天最重要的落点不是新增一个可直接运行的 strict replay 命令，而是把“现在能跑什么”和“下一步该补什么”彻底拆开了。

截至 2026-03-11：

1. 你今天就可以直接跑 mini 或 smoke 闭环
2. 闭环后你今天就可以直接跑 replay baseline 和三组 sweep
3. 你今天还不能直接跑 strict replay_v2，因为 replay_v2 导出和 decision-only runner 尚未实现

因此最合理的下一步是：

1. 重新生成一份新的 mini 或 smoke replay
2. 跑完整的 baseline + sweep
3. 根据结果判断：
   - 如果 `num_history / prompt / max_new_tokens` 已经拿到明显收益，则先推进这些轻量优化
   - 如果收益有限，再继续做 `KV cache / backend`
4. 并行规划 `replay_v2` 字段补充，准备进入 strict reproduction 阶段

## 六、3-11 晚更新：replay_v2 与 HTTP loopback 已落地

本日后续代码已经继续推进，前文中“`replay_v2` 和 `HTTP loopback replay` 尚未实现”的状态已不再成立，现补充实际落地情况。

### 1. 新增代码落地

本次新增和更新的核心文件如下：

- `internnav/habitat_extensions/vln/habitat_vln_evaluator.py`
- `scripts/eval/tools/benchmark_dualvln_http_loopback.py`

其中：

1. 原版 `replay_subset` 保持不变，保证旧版 `benchmark_dualvln_replay.py` 仍可直接使用
2. evaluator 现在会并行导出 `replay_subset_v2`
3. `replay_subset_v2` 在旧版字段之外，新增了决策级上下文字段：
   - `record_id`
   - `is_new_s2_decision`
   - `is_lookdown_followup`
   - `decision_id`
   - `decision_step_offset`
   - `pixel_goal_cycle_id`
   - `cycle_step_offset`
   - `action_source`
   - `s1_regenerated_this_step`
   - `decision_prompt_text`
   - `decision_chat_text`
   - `decision_input_image_paths`
   - `conversation_messages`
   - `s2_output_text`
   - `s2_generated_token_count`
   - `s2_output_kind`
   - `s2_output_pixel`
   - `s2_action_seq`
   - `s1_initial_action_seq`
   - `http_loopback` 请求元信息

### 2. 新增可直接运行的 HTTP loopback benchmark

新脚本：

- `scripts/eval/tools/benchmark_dualvln_http_loopback.py`

它的定位是：

- 不额外启动 Flask server
- 在本地进程内模拟 client 的 JPEG/PNG 编码
- 模拟 server 的解码与 `agent.step(...)`
- 如果模型返回 `LOOKDOWN`，可复用 `replay_v2` 中保存的 lookdown RGB/depth 再做一次 `look_down=True` 调用

因此它本质上是一个 in-process loopback benchmark，用来估算“编码/解码 + realworld agent 封装”这层额外开销。

### 3. 更新后的可执行命令

#### 3.1 重新导出 `replay_subset_v2`

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate habitat
cd /root/backup/InternNav
python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_mini_cfg.py
```

输出目录现在会同时包含：

- `replay_subset/manifest_rank0.jsonl`
- `replay_subset_v2/manifest_rank0.jsonl`

#### 3.2 原版纯推理 replay benchmark

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate habitat
cd /root/backup/InternNav
export TOKENIZERS_PARALLELISM=false
python -u scripts/eval/tools/benchmark_dualvln_replay.py \
  --manifest /root/backup/InternNav/logs/habitat/test_dual_system_mini/replay_subset/manifest_rank0.jsonl \
  --model-path /root/backup/InternNav/checkpoints/InternVLA-N1-DualVLN \
  --output /root/backup/InternNav/logs/habitat/test_dual_system_mini/replay_summary.json \
  --base-path /root/backup/InternNav/logs \
  --num-history 8 \
  --prompt-variant full \
  --max-new-tokens 128 \
  --verbose-every 20
```

#### 3.3 HTTP loopback replay benchmark

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate habitat
cd /root/backup/InternNav
export TOKENIZERS_PARALLELISM=false
python -u scripts/eval/tools/benchmark_dualvln_http_loopback.py \
  --manifest /root/backup/InternNav/logs/habitat/test_dual_system_mini/replay_subset_v2/manifest_rank0.jsonl \
  --model-path /root/backup/InternNav/checkpoints/InternVLA-N1-DualVLN \
  --output /root/backup/InternNav/logs/habitat/test_dual_system_mini/http_loopback_summary.json \
  --base-path /root/backup/InternNav/logs \
  --num-history 8 \
  --plan-step-gap 4 \
  --use-recorded-lookdown \
  --verbose-every 20
```

### 4. 本次 replay benchmark 结果分析

本次实际分析对象为：

- `logs/habitat/test_dual_system_mini/result.json`
- `logs/habitat/test_dual_system_mini/runtime_summary_rank0.json`
- `logs/habitat/test_dual_system_mini/replay_summary.json`

#### 4.1 闭环结果

当前 8 个 episode 的闭环指标为：

- `Success = 1.0`
- `SPL = 0.911`
- `Oracle Success = 1.0`
- `Navigation Error = 0.916`
- 平均每个 episode 的总 wall-clock 约 `44.26s`
- 平均每步 wall-clock 约 `0.508s`
- `S2` 单次平均约 `0.513s`
- `S1` 单次平均约 `0.153s`
- `pixel_goal_ratio ≈ 0.370`
- `discrete_ratio ≈ 0.630`
- `avg_s1_steps_per_cycle ≈ 8.44`

结论不变：Habitat 闭环里大小脑切换是真实发生的，且 `S2` 仍是主要模型级瓶颈。

#### 4.2 Level 1 replay baseline 结果

离线纯推理 replay 本次跑的是：

- `8` 个 episode
- `697` 个 replay step

关键结果如下：

- cold start 约 `3.51s`
- GPU 峰值显存约 `17.5GB`
- `total_step p50 ≈ 0.860s`
- `total_step p95 ≈ 0.933s`
- `total_step max ≈ 1.964s`
- `s2_generate p50 ≈ 0.860s`
- `s2_generate p95 ≈ 0.932s`
- `s2_generate max ≈ 1.006s`
- `s2_latent count = 24`
- `s1_generate count = 24`
- 平均生成长度约 `2.77 tokens`
- 平均 tokens/s 约 `3.74`

#### 4.3 一致性结果

- `output_kind_match_rate ≈ 0.298`
- `action_match_rate_all ≈ 0.280`
- `action_match_rate_discrete ≈ 0.803`
- `action_match_rate_pixel_goal ≈ 0.057`
- `text_exact_match_rate ≈ 0.221`
- `pixel_goal L2 mean ≈ 22.77`

进一步看 confusion：

- `pixel_goal -> discrete = 477`
- `pixel_goal -> pixel_goal = 12`
- `discrete -> discrete = 196`
- `discrete -> pixel_goal = 12`

#### 4.4 对结果的解释

这个结果非常清楚地说明了两件事：

1. 当前 Level 1 replay benchmark 仍然主要在测 `S2` 纯推理成本
2. 它依然不能直接替代 strict replay 或闭环重放

证据是：

- `total_step p50` 和 `s2_generate p50` 几乎相等，说明总时延几乎完全由 `S2` 主导
- 虽然闭环中 `pixel_goal_ratio` 约为 `37%`，但离线 replay 里 `s2_latent/s1` 只触发了 `24` 次
- 大量 baseline 为 `pixel_goal` 的 step 在离线 replay 中被重跑成了 `discrete`

因此当前 replay baseline 最适合拿来做：

- `S2` 纯推理延时基线
- `num_history / prompt / max_new_tokens / KV cache / backend` 的单变量筛选

但不适合直接拿来回答：

- “为什么 realworld 里 `S1` 很少发生”
- “当前离线 replay 是否已经严格复现了闭环里的大小脑交接”

### 5. 当前阶段的最合理下一步

基于这次结果，下一步优先级建议如下：

1. 保留当前 Level 1 replay baseline，继续跑三组 sweep
2. 单独补一个 `decision-only replay benchmark`，只消费 `replay_v2` 中 `is_new_s2_decision=true` 的节点
3. 再基于 `replay_v2` 增量实现 `full-cycle replay`
4. 最后再跑 `HTTP loopback replay`，估算序列化和 realworld agent 封装带来的额外开销

### 6. 关于当前 sweep 结果的解释

`logs/habitat/test_dual_system_mini/sweeps` 目录已经完整存在，因此同一套命令不需要为了“生成 sweep”再重跑一遍。

但这三组 sweep 的解释要分开看：

#### 6.1 `max_new_tokens` 结果已经足够明确

`128 / 64 / 32 / 16` 几乎没有差别，原因很直接：

- 当前平均生成长度只有 `2.77 tokens`
- 因此 `16` 已经远高于实际输出长度
- 这说明当前 latency 不是被 `max_new_tokens` 截断上限卡住的

结论：

- 这一组已经可以收敛，暂时没必要继续围绕 `max_new_tokens` 花时间

#### 6.2 prompt 长度结果也基本明确

`full / short / minimal` 之间只有极小差别，`minimal` 略快，但量级只有毫秒级，同时一致性没有出现结构性改善。

结论：

- 当前 Level 1 replay 下，prompt 冗余不是主要瓶颈

#### 6.3 `num_history` 这一组当前并不能被直接解读

这里有一个实现层面的关键细节：

- `benchmark_dualvln_replay.py` 在构造输入时，会优先使用 manifest 中已经保存的 `history_frame_indices`
- 只有在 manifest 没有该字段时，才会按 `--num-history` 重新计算

因此，当前这份 replay manifest 上做出来的 `num_history = 8 / 4 / 2 / 0` 几乎不变，并不表示“历史帧长度对时延没有影响”，而更可能表示：

- 当前这组 sweep 实际上没有真正改到输入历史帧

结论：

- 当前 `num_history` sweep 不能作为最终结论
- 如果要真正评估 `num_history`，需要新增一个 override 开关，让 benchmark 忽略 manifest 中记录的 history，按传入参数强制重建输入

### 7. 3-11 晚更新二：`num_history` override 已补上

后续代码已继续更新，当前 `benchmark_dualvln_replay.py` 新增：

- `--ignore-manifest-history`

作用是：

- 不使用 manifest 中保存的 `history_frame_indices`
- 而是严格按 `--num-history` 重新构造历史帧输入

同时，`run_replay_benchmark_sweeps.py` 现在会在 `num_history` 组自动启用这个参数，因此后续只需要重跑 `num_history` 这一组，而不需要把 prompt 和 `max_new_tokens` 两组也一起重跑。

对应的重跑命令如下：

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate habitat
cd /root/backup/InternNav
export TOKENIZERS_PARALLELISM=false
python scripts/eval/tools/run_replay_benchmark_sweeps.py \
  --manifest /root/backup/InternNav/logs/habitat/test_dual_system_mini/replay_subset/manifest_rank0.jsonl \
  --model-path /root/backup/InternNav/checkpoints/InternVLA-N1-DualVLN \
  --output-dir /root/backup/InternNav/logs/habitat/test_dual_system_mini/sweeps \
  --base-path /root/backup/InternNav/logs \
  --groups num_history \
  --verbose-every 20
```

### 8. 3-11 晚更新三：重跑后的 `num_history` 结果

修正后的 `comparison_num_history.csv` 已重新生成，且 `ignore_manifest_history=True`，说明这次 sweep 确实改到了历史帧输入。

关键结果如下：

#### 8.1 时延与显存

- `num_history=8`
  - `total_step p50 ≈ 0.860s`
  - `s2_generate p50 ≈ 0.859s`
  - `GPU peak ≈ 17.5GB`
- `num_history=4`
  - `total_step p50 ≈ 0.484s`
  - `s2_generate p50 ≈ 0.484s`
  - `GPU peak ≈ 16.9GB`
- `num_history=2`
  - `total_step p50 ≈ 0.319s`
  - `s2_generate p50 ≈ 0.319s`
  - `GPU peak ≈ 16.5GB`
- `num_history=0`
  - `total_step p50 ≈ 0.206s`
  - `s2_generate p50 ≈ 0.205s`
  - `GPU peak ≈ 16.2GB`

结论非常明确：

- 历史帧数量确实是当前 Level 1 replay 的主瓶颈之一
- `S2 generate` 与 `total_step` 仍几乎重合，说明收益主要来自 System 2 prefill 减少
- 从 `8 -> 4 -> 2 -> 0` 呈现近似单调下降趋势

#### 8.2 一致性与行为变化

- `num_history=8`
  - `action_match_rate_all ≈ 0.258`
  - `action_match_rate_discrete ≈ 0.774`
  - `output_kind_match_rate ≈ 0.281`
  - `s2_latent count = 12`
  - `s1_generate count = 12`
- `num_history=4`
  - `action_match_rate_all ≈ 0.263`
  - `action_match_rate_discrete ≈ 0.764`
  - `output_kind_match_rate ≈ 0.278`
  - `s2_latent count = 14`
  - `s1_generate count = 14`
- `num_history=2`
  - `action_match_rate_all ≈ 0.258`
  - `action_match_rate_discrete ≈ 0.702`
  - `output_kind_match_rate ≈ 0.297`
  - `s2_latent count = 1`
  - `s1_generate count = 1`
- `num_history=0`
  - `action_match_rate_all ≈ 0.212`
  - `action_match_rate_discrete ≈ 0.519`
  - `output_kind_match_rate ≈ 0.298`
  - `s2_latent count = 0`
  - `s1_generate count = 0`

这里最关键的现象不是单纯看一个 match rate，而是看：

1. `num_history=4` 时，时延几乎减半，但一致性没有出现结构性恶化
2. `num_history=2` 时，时延继续明显下降，但 pixel-goal / S1 几乎消失
3. `num_history=0` 时，时延最低，但实际上已经退化成“几乎纯 discrete、几乎不进 S1”的模式

#### 8.3 对结果的解释

这组结果说明：

1. 视觉历史输入对 `S2` 的计算成本影响非常大
2. 但把历史帧压得过低，会显著改变模型的输出行为
3. 真正有希望作为“低风险优化”的点是 `num_history=4`

因此当前最合理的 tradeoff 是：

- `num_history=4` 作为下一步优先验证配置

原因：

- 相比 `8`，`p50` 时延从约 `0.86s` 降到约 `0.48s`
- 峰值显存也下降约 `0.65GB`
- 整体一致性和输出类型没有出现灾难性崩塌
- 同时仍保留了 `pixel_goal -> S1` 路径

而：

- `num_history=2` 和 `0` 更像是“激进压缩”
- 它们虽然更快，但已经明显改变了大小脑切换行为，不适合直接作为部署候选

### 9. 当前推荐的下一步实验

基于目前全部结果，下一步建议顺序调整为：

1. 先把 `num_history=4` 放回 Habitat 闭环重跑一次
2. 对比：
   - `Success / SPL / Oracle Success / Navigation Error`
   - `pixel_goal_ratio / avg_s1_steps_per_cycle`
   - `episode wall-clock / avg_step wall-clock`
3. 如果闭环功能指标和大小脑切换没有明显退化，再进入 `KV cache / backend` 优化

原因是：

- `num_history=4` 已经是一个证据充分、改动极小、收益明显的优化点
- 它比直接做 KV cache 更便宜、更容易验证
- 只有当这个低成本收益吃完以后，再继续做更重的系统级优化才更划算

### 10. 3-11 晚更新四：新增一键闭环 runner

当前已新增：

- `scripts/eval/tools/run_habitat_closed_loop_num_history_sweeps.py`

作用：

- 直接在 Habitat 闭环里 sweep `num_history`
- 支持 `mini / full` 两种数据模式
- 默认跑 `4 / 2 / 1 / 0`
- 自动为每一组输出独立日志目录
- 跑完后自动汇总出 `comparison_num_history_closed_loop.csv`

#### 10.1 mini 一键闭环命令

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate habitat
cd /root/backup/InternNav
python scripts/eval/tools/run_habitat_closed_loop_num_history_sweeps.py \
  --dataset mini \
  --num-history-values 4 2 1 0 \
  --output-root ./logs/habitat/closed_loop_num_history_sweeps \
  --max-eval-episodes 8 \
  --replay-num-episodes 8
```

#### 10.2 full 一键闭环命令

如果想用完整 `val_unseen`，但仍限制 episode 数量，例如先跑 8 个：

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate habitat
cd /root/backup/InternNav
python scripts/eval/tools/run_habitat_closed_loop_num_history_sweeps.py \
  --dataset full \
  --num-history-values 4 2 1 0 \
  --output-root ./logs/habitat/closed_loop_num_history_sweeps \
  --max-eval-episodes 8 \
  --replay-num-episodes 8
```

如果要跑 full 且不限制 episode 数量，则把：

- `--max-eval-episodes 8`

改成：

- `--max-eval-episodes 0`

即可。

### 11. 3-11 晚更新五：闭环 runner 首次报错与修复

在首次运行 `run_habitat_closed_loop_num_history_sweeps.py` 时，出现了：

- `IndexError: list index out of range`

报错位置在：

- `internnav/habitat_extensions/vln/habitat_vln_evaluator.py`
- `replay_v2_state["rgb_frame_paths"][i]`

原因是：

- `rgb_list` 在 step 开头就会 append 当前图像
- 但 `replay_v2` 的 `rgb_frame_paths` 原先是在更晚的位置才 append
- 因此当同一步立即构造 `history_id -> input_image_paths` 时，路径列表长度会比图像列表短一拍

修复方式：

- 将 `rgb_frame_paths.append(current_rgb_input_path)` 前移到图像 append 之后、构造 `input_image_paths` 之前

这样 `rgb_list` 和 `rgb_frame_paths` 就能保持同步，不再在闭环 runner 中触发越界。

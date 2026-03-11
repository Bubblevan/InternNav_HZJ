# DualVLN Inference Optimization Plan

本文给出针对 DualVLN 的 KV cache / 推理后端优化思路，目标是在不明显破坏输出一致性的前提下，优先降低 System 2 的生成时延。

## 1. 为什么优先优化 System 2

从当前 Habitat mini replay 的 profiling 结果看：

- `S2 generate` 单次平均时延显著高于 `S1 generate`
- `S2` 输出离散动作的比例仍高于 `pixel_goal`
- 因此很多闭环步的关键路径仍由 `S2` 主导

所以优化优先级应当是：

1. 先减少 `S2` 的单次生成开销
2. 再考虑是否提升 `S2 -> S1` 的交接频率
3. 最后才是做更激进的量化压缩

## 2. KV cache 可以怎么优化

### 2.1 先确认当前是否真正吃到了 cache

虽然代码里 `model.generate(..., use_cache=True)` 已开启，但仍需确认以下几点：

1. 当前聊天模板和多图输入是否导致每步都完整 prefill
2. 每一步是否都显式传了 `past_key_values=None`，从而阻断跨步 cache 复用
3. 当前 evaluator / policy 的调用方式是否让 cache 只在单次 generate 内生效，而不能跨步复用

当前代码在每一步 S2 都重新构造完整 prompt 和 image tokens，因此很可能只有“单次生成内部”在用 cache，而没有“跨环境步复用”。

### 2.2 可以做的第一类优化：缩短每次 prefill

重点不是先改模型，而是减少每次重新编码的上下文长度。

建议：

1. 减少 instruction 模板冗余，压缩固定 prompt
2. 限制历史图数量，测试 `num_history=8 -> 4 -> 2`
3. 对 look-down 对话分支减少无必要的 conversation 累积

如果这些改动能显著降低 `S2 generate` latency，同时输出一致性保持稳定，就比直接上量化更稳。

### 2.3 可以做的第二类优化：跨步 cache 复用

理论上更激进的做法是：

1. 把 instruction 的静态文本部分单独 prefill 一次
2. 将历史视觉输入拆成“增量 append”形式
3. 在相邻环境步之间复用 `past_key_values`

但这条路改动会比较大，因为 Qwen-VL 多模态输入不是简单纯文本，图像 token 的位置编码和模板拼接都需要重新验证。因此建议先把它作为第二阶段实验，而不是第一阶段。

## 3. 推理后端应该怎么选

### 3.1 先做“最小改动”路线

在当前 Hugging Face + Flash Attention 2 路径上，优先确认：

1. `flash_attention_2` 是否真正启用
2. `torch_dtype=bfloat16` 是否是当前卡上最合适的配置
3. 是否存在多余的 CPU-GPU 同步或张量搬运

这一步的好处是：

- 改动最小
- 输出风险最低
- 可以先得到一个稳态 baseline

### 3.2 第二条路线：更高性能的 serving/backend

如果 `S2 generate` 仍明显偏慢，可以探索：

1. vLLM
2. TensorRT-LLM
3. lmdeploy

但要注意：

- DualVLN 不是纯文本模型，而是带视觉输入和自定义 `generate_latents / generate_traj` 的多模态模型
- 标准后端往往先优化 `generate`
- 对 `generate_latents` 和 `generate_traj` 的兼容性要单独验证

因此更现实的落地方式是：

> 先只把 System 2 文本生成部分接到更快后端，System 1 相关部分先保持现有 PyTorch 实现。

## 4. 推荐实验顺序

### 实验 A：离线 replay 基线

目标：

- 记录当前 `S2-only`、`S2+latent`、`S2+S1` 的 p50 / p95 / max
- 记录 cold start
- 记录显存峰值
- 记录输出一致性

输出：

- 作为后续所有优化实验的对照基线

### 实验 B：Prompt / history 缩减

变量：

- `num_history`
- prompt 模板长度
- 最大生成长度

目标：

- 判断 latency 是否主要受 prefill 长度影响

建议把这一轮做成严格的单变量实验，而不是一次同时改多个参数。推荐顺序如下。

#### B1. `num_history` 缩减实验

固定其余参数不变，仅改：

- `num_history = 8` 作为 baseline
- `num_history = 4`
- `num_history = 2`
- `num_history = 0`

每组记录：

- `S2 generate p50 / p95 / max`
- `total_step p50 / p95 / max`
- `tokens_per_second`
- `output_kind_match_rate`
- `action_match_rate_all`
- `action_match_rate_pixel_goal`

判断标准：

- 如果 `num_history` 减少后 `S2 generate` 明显下降，而输出一致性变化很小，则说明当前瓶颈有较大一部分来自视觉历史输入的 prefill 开销

#### B2. prompt 长度缩减实验

固定 `num_history = 8`，仅修改 prompt 模板，建议至少比较三档：

- `prompt_v0_full`
  即当前完整版
- `prompt_v1_short`
  保留任务描述和输出要求，但删除修饰性句子
- `prompt_v2_min`
  只保留最小必要格式，例如“instruction + 输出 pixel goal 或 STOP”

每组记录：

- `S2 generate p50 / p95 / max`
- `generated_tokens_mean`
- `text_exact_match_rate`
- `output_kind_match_rate`

判断标准：

- 如果 prompt 缩短显著降低 `S2 generate`，但 `output_kind_match_rate` 明显变差，说明 prompt 中仍有稳定输出所需的约束信息，不能简单删除

#### B3. `max_new_tokens` 缩减实验

固定 `num_history` 和 prompt，建议比较：

- `max_new_tokens = 128` 作为 baseline
- `max_new_tokens = 64`
- `max_new_tokens = 32`
- `max_new_tokens = 16`

每组记录：

- `generated_tokens_mean`
- `S2 generate p50 / p95 / max`
- `action_match_rate_all`
- `output_kind_match_rate`

判断标准：

- 如果降低 `max_new_tokens` 后 latency 下降，但一致性不变，说明当前生成长度存在冗余
- 如果一致性明显下降，则说明模型实际需要较长输出空间，不能简单截断

#### B4. 推荐实验执行矩阵

最稳妥的执行方式不是全排列，而是分三轮：

第一轮：

- 只扫 `num_history`

第二轮：

- 在第一轮最佳 `num_history` 上扫 prompt 长度

第三轮：

- 在前两轮最佳配置上扫 `max_new_tokens`

这样可以避免一次性引入太多变量，导致结果不可解释

### 实验 C：KV cache 强化

变量：

- 更严格的 cache 复用策略
- 尝试跨步复用静态前缀

目标：

- 观察 `S2 generate` 是否有明显下降

### 实验 D：更换推理后端

变量：

- vLLM 或 TensorRT-LLM 仅接管 S2 generate

目标：

- 判断在不改 S1 的前提下，System 2 是否还能继续明显加速

### 实验 E：量化

变量：

- W8A8 / INT8

目标：

- 同时看延迟、显存和输出一致性

## 5. 当前最值得优先做什么

基于现有结果，优先级建议是：

1. 先跑离线 replay benchmark 基线
2. 然后优先试 `num_history` / prompt 压缩
3. 再试 KV cache / S2 serving backend
4. 最后再做量化

原因很简单：

- 现在最重的是 S2
- S2 的主要瓶颈是生成
- 生成瓶颈通常先受益于 cache 和 backend
- 量化更适合放在已经明确主瓶颈之后做收益补充

## 5.1 当前离线 replay benchmark 的限制

需要特别说明：当前第一版离线 replay benchmark 更适合测“纯推理 baseline”，还不等于“严格复现闭环大小脑切换”。

原因是：

1. replay manifest 中保存的是每个环境 step，而不是只保存真正触发新 S2 决策的 step
2. 很多 `pixel_goal` 标记对应的是一个仍在持续执行的 S1 cycle，而不是新的 S2 文本生成
3. 当前脚本是按“每个 step 都重新构造一次输入并跑一次 S2”的方式做 benchmark，因此会天然偏向离散输出
4. 当前脚本还没有完整复现 evaluator 中的 conversation continuation 和 look-down 分支上下文

因此，当前 replay benchmark 的结果应理解为：

- 可以测 `S2 generate` 的纯推理速度上限
- 可以做 prompt / history / token 长度等单变量对比
- 但不能直接把其中的 `pixel_goal -> discrete` 失配，解释成模型真正失去了 System 1 能力

## 6. 成功标准

如果一个优化方案要被认为“值得继续推进”，建议同时满足：

1. `S2 generate p50/p95` 明显下降
2. `S2+S1 total latency` 也同步下降
3. `action_match_rate` 和 `output_kind_match_rate` 不显著恶化
4. 放回 Habitat 闭环后，`Success / SPL / Navigation Error` 不显著退化

否则，即使模型单测更快，也不一定真的适合实际部署。

## 7. 3-11 当前基线结果

### 7.1 Habitat 闭环基线

当前 `logs/habitat/test_dual_system_mini` 的闭环 summary 为：

- `Success = 1.0`
- `SPL = 0.911`
- `Oracle Success = 1.0`
- `Navigation Error = 0.916`
- 平均每步 wall-clock 约 `0.508s`
- `S2` 单次平均约 `0.513s`
- `S1` 单次平均约 `0.153s`
- `pixel_goal_ratio ≈ 0.370`
- `discrete_ratio ≈ 0.630`
- `avg_s1_steps_per_cycle ≈ 8.44`

这说明闭环里的大小脑切换是实际发生的，且 System 2 仍是模型级主瓶颈。

### 7.2 Level 1 replay baseline

当前 `replay_summary.json` 的关键结果为：

- `8` 个 episode，`697` 个 step
- cold start 约 `3.51s`
- GPU peak memory 约 `17.5GB`
- `total_step p50 ≈ 0.860s`
- `total_step p95 ≈ 0.933s`
- `s2_generate p50 ≈ 0.860s`
- `s2_generate p95 ≈ 0.932s`
- `s2_latent count = 24`
- `s1_generate count = 24`
- `tokens_per_second_mean ≈ 3.74`

一致性上：

- `output_kind_match_rate ≈ 0.298`
- `action_match_rate_all ≈ 0.280`
- `action_match_rate_discrete ≈ 0.803`
- `action_match_rate_pixel_goal ≈ 0.057`
- `text_exact_match_rate ≈ 0.221`

这组数字表明：

1. 当前 Level 1 replay 主要在测 `S2 generate`
2. 它已经足够适合作为 System 2 纯推理优化基线
3. 但它还不足以作为“严格大小脑切换重放”的依据

### 7.3 当前优化策略的直接含义

由于 `total_step p50` 基本与 `s2_generate p50` 重合，当前最值得优先优化的仍然是：

1. `num_history`
2. prompt 长度
3. `max_new_tokens`
4. `KV cache / backend`

而不是先把主要时间投到量化或 S1 上。

### 7.4 新增的 Level 3：HTTP loopback

当前代码里还新增了：

- `scripts/eval/tools/benchmark_dualvln_http_loopback.py`

它的作用不是替代 Level 1，而是补一层更接近 realworld agent 封装的成本测量。后续如果要判断通信/序列化开销是否显著，应先用这条链路打出一份独立 baseline，再与 Level 1 纯推理结果做对比。

## 8. 当前 sweep 应如何解读

### 8.1 `max_new_tokens` 已基本排除

当前 `128 / 64 / 32 / 16` 几乎没有差别，而平均生成长度只有 `2.77 tokens`。这说明：

- 当前 Level 1 replay 的输出非常短
- `max_new_tokens` 不是主要约束
- 继续围绕这一维做实验的优先级可以降低

### 8.2 prompt 长度不是主要瓶颈

`full / short / minimal` 的 latency 差异只有很小的毫秒级变化，说明在这条链路上：

- prompt 模板冗余不是主要时延来源

### 8.3 当前 `num_history` sweep 结果不能直接下结论

需要特别注意一个实现细节：

- 当前 `benchmark_dualvln_replay.py` 会优先使用 replay manifest 里已经保存的 `history_frame_indices`
- 只有 manifest 没有该字段时，才会按 `--num-history` 重建历史图输入

因此在当前这份 manifest 上，`num_history = 8 / 4 / 2 / 0` 几乎没有变化，更可能表示：

- 这组 sweep 并没有真正修改输入历史帧

而不是：

- 历史帧长度对时延完全没有影响

所以当前三组 sweep 的结论应当是：

1. `max_new_tokens` 基本可以先排除
2. prompt 长度不是主要瓶颈
3. `num_history` 需要修 benchmark 的 override 逻辑后再重新测

### 8.4 当前已补上的修正

当前代码已经新增：

- `benchmark_dualvln_replay.py --ignore-manifest-history`

该参数会忽略 manifest 中已经保存的 `history_frame_indices`，改为严格按 `--num-history` 重建输入历史帧。

同时：

- `run_replay_benchmark_sweeps.py` 会在 `num_history` 组自动加上这个参数

因此现在的正确做法是：

- 只重跑 `num_history` 这一组
- 不必重跑 prompt 和 `max_new_tokens` 两组

## 9. 修正后的 `num_history` sweep 结果

当前新的 `comparison_num_history.csv` 已在 `ignore_manifest_history=True` 条件下重新生成，因此这次结果可以作为有效结论使用。

### 9.1 速度收益

从 `num_history=8 -> 4 -> 2 -> 0`：

- `total_step p50`: `0.860s -> 0.484s -> 0.319s -> 0.206s`
- `s2_generate p50`: `0.859s -> 0.484s -> 0.319s -> 0.205s`
- `GPU peak memory`: `17.5GB -> 16.9GB -> 16.5GB -> 16.2GB`

这说明：

- 历史帧长度确实是当前 Level 1 replay 的主要性能控制杆
- 下降趋势几乎单调，且收益幅度非常明显

### 9.2 一致性与行为代价

同时也要看输出行为：

- `num_history=4`
  - 时延接近减半
  - 一致性没有结构性恶化
  - 仍保留 `pixel_goal -> S1` 路径
- `num_history=2`
  - 虽然更快
  - 但 `s2_latent/s1_generate` 只剩 `1` 次
  - 表明大小脑切换几乎被压没
- `num_history=0`
  - 最快
  - 但 `s2_latent/s1_generate = 0`
  - 实际上退化成“几乎纯 discrete”策略

### 9.3 当前最优候选

基于速度和行为保持的 tradeoff，当前最值得优先验证的是：

- `num_history=4`

原因：

1. 相比 baseline 速度收益已经非常大
2. 不需要对模型实现做侵入式改动
3. 仍保留 System 1 参与闭环的可能性

### 9.4 因此下一步怎么排

当前建议顺序是：

1. 先把 `num_history=4` 放回 Habitat 闭环验证
2. 若功能指标和大小脑切换都稳定，再做 `KV cache / backend`
3. 不建议优先把时间花在 `num_history=2` 或 `0` 上

也就是说，KV cache 仍然是下一阶段重点，但它现在应该排在：

- “先验证 `num_history=4` 这个已被数据证明有效的低成本优化”

之后，而不是之前。

## 10. 闭环验证的一键 runner

当前已新增：

- `scripts/eval/tools/run_habitat_closed_loop_num_history_sweeps.py`

这个脚本用于把离线筛出来的 `num_history` 候选，直接放回 Habitat 闭环验证。

### 10.1 mini 模式

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

### 10.2 full 模式

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

说明：

- `--dataset mini` 会使用 `val_unseen_mini`
- `--dataset full` 会使用完整 `val_unseen`
- `--max-eval-episodes 8` 适合先做 smoke
- `--max-eval-episodes 0` 表示不限制 episode 数量

跑完后会自动产出：

- 每个 `num_history` 的独立输出目录
- `comparison_num_history_closed_loop.csv`

# DualVLN KV Cache Experiment Plan

本文聚焦一个更具体的问题：

> 在当前 DualVLN 代码里，KV cache 到底有没有真正跨步复用？如果没有，下一步应该怎样分阶段实验，而不是一次性做高风险重构？

在 2026-03-12 的实验推进后，本文也额外覆盖一个相邻问题：

> 如果 `past_key_values + multimodal generate` 这条 continuation 路线被底层接口卡住，那么更稳的下一个方向应该是什么？

## 1. 当前状态判断

当前代码已经明确显示，System 2 生成阶段虽然在单次 `generate()` 内打开了：

- `use_cache=True`

但在所有关键入口里仍然显式传入：

- `past_key_values=None`

这意味着：

1. 单次生成内部的 decoder cache 是开的
2. 但跨环境步、跨决策步的 cache 复用实际上并没有发生

当前可见代码落点：

- `internnav/habitat_extensions/vln/habitat_vln_evaluator.py`
- `scripts/eval/tools/benchmark_dualvln_replay.py`
- `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py`
- `internnav/agent/internvla_n1_agent_realworld.py`

其中：

- `internvla_n1_agent_realworld.py` 里甚至已经有注释掉的 `past_key_values` 代码，说明作者此前考虑过这个方向，但并未真正启用

## 2. 为什么不能直接暴力打开跨步 cache

因为当前 S2 输入并不是纯文本，而是：

- instruction 模板
- 历史图像 token
- 当前图像 token
- 某些情况下还有 look-down continuation

而代码在每一步都重新构造完整的：

- `messages`
- `chat template`
- `processor(...)`

这意味着：

1. 每一步 prefill 都在重做
2. 即使 `past_key_values` 对象存在，也不能简单把“完整旧 prompt + 新 prompt”一起再喂一遍
3. 如果 cache 复用方式不对，很容易出现 token 位置、RoPE index、图像 token 对齐错误

所以当前不适合直接上“全局跨步 cache 复用”，而应该分阶段做。

## 3. 推荐实验分层

### Phase A: 零风险定位

目标：

- 确认 System 2 的时延里，究竟有多少来自 prefill，多少来自 decode

建议记录：

- `input_ids` 长度
- 历史图像数量
- 当前图像数量
- 每步 `generated_tokens`
- `s2_generate_seconds`

代码落点：

- `scripts/eval/tools/benchmark_dualvln_replay.py`

这是最小改动，因为 benchmark 里已经能拿到：

- `inputs.input_ids.shape[1]`
- `len(input_images)`
- `generated_token_count`

### Phase B: look-down follow-up cache

目标：

- 只在最局部、最安全的路径上尝试 cache

最合适的第一块不是跨任意环境步，而是：

- `LOOKDOWN -> look_down=True` 的 follow-up

原因：

1. 这条路径本来就在同一个决策上下文内
2. 当前 realworld agent 已有 `conversation_history` 和 `self.past_key_values` 变量
3. 代码里已有注释掉的 cache 相关位置

代码落点：

- `internnav/agent/internvla_n1_agent_realworld.py`

建议做法：

1. 新增实验 flag，例如 `kv_cache_mode`
2. 只在 `look_down=True` 时尝试复用上一轮的 cache
3. 先做离线/loopback 验证，不直接上真实部署链

注意：

- 这一步是否能真正启用，还取决于 Hugging Face `generate()` 与当前多模态输入拼接是否兼容
- 如果不兼容，不要强行硬上，而应停在 Phase A / C

### Phase C: prefix reuse / partial prefill

目标：

- 保留 `num_history=8`
- 但尽量避免每一步重做静态 instruction 前缀

核心思路：

1. 区分静态文本前缀与动态视觉输入
2. 评估 instruction 固定前缀是否能先 prefill 一次
3. 后续只重新编码变化部分

代码落点：

- `internnav/habitat_extensions/vln/habitat_vln_evaluator.py`
- `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py`

这一步会比 Phase B 难很多，因为它已经涉及：

- prompt 结构拆分
- cache_position / rope_deltas 处理
- 图像 token 位置的一致性验证

因此不建议作为第一步。

### Phase D: backend 替换

目标：

- 在保持上下文不变的情况下，直接降低 System 2 生成时延

候选：

- vLLM
- TensorRT-LLM
- lmdeploy

但前提是：

- 只先接管 `S2 generate`
- `generate_latents / generate_traj` 保持现有 PyTorch 路线

## 4. 当前最值得先动的代码

基于 2026-03-12 的结果，最合理的顺序是：

1. 先补 Phase A 的输入统计
2. 再评估 Phase B 的 look-down follow-up cache 是否能做最小实验
3. 若 Phase B 风险高，则直接转向 Phase D 的 backend 路线

也就是说，当前最优先的“代码落点”是：

- `benchmark_dualvln_replay.py`
- `internvla_n1_agent_realworld.py`

而不是一上来就大改 evaluator 主循环。

## 5. 与当前结论的关系

2026-03-12 的 Habitat 闭环结果已经说明：

- 不能简单通过删历史上下文来换速度

因此 KV cache 方向的价值进一步上升，因为它追求的是：

> 保持 `num_history=8` 的语义输入不变，只减少重复计算。

这也是为什么今天以后，KV cache / prefix reuse / backend 会成为比 `num_history` 缩减更优先的实验方向。

## 6. 2026-03-12 已落地的 Phase A / B

今天已经把两个最小可执行阶段真正落到代码里。

### 6.1 Phase A: replay benchmark 输入压力统计

代码文件：

- `scripts/eval/tools/benchmark_dualvln_replay.py`
- `scripts/eval/tools/run_replay_benchmark_sweeps.py`

新增统计：

- `prefill.input_token_count`
- `prefill.input_image_count`
- `prefill.history_image_count`
- `prefill.prompt_char_count`
- `prefill.image_token_count`

同时每个 step 的 details jsonl 也会额外记录：

- `input_token_count`
- `input_image_count`
- `history_image_count`
- `prompt_char_count`
- `image_token_count`

这一步解决的问题是：

> 当 `S2 generate` 变慢时，究竟是文本 token、多图历史、还是视觉 token prefill 在拉高成本。

### 6.2 Phase B: look-down continuation 的实验型 KV cache

代码文件：

- `internnav/agent/internvla_n1_agent_realworld.py`
- `scripts/eval/tools/benchmark_dualvln_http_loopback.py`
- `scripts/realworld/http_internvla_server.py`

新增开关：

- `--kv-cache-mode disabled`
- `--kv-cache-mode lookdown_experimental`
- `--kv-cache-debug`

当前 Phase B 的边界非常明确：

1. 只尝试在 `look_down=True` 的 follow-up 上复用 cache
2. 主链路默认仍是 `disabled`
3. 只有当“新 prompt 的 token 前缀完全等于上一轮输出序列”时，才会尝试走 cache
4. 一旦出现 prefix mismatch、图像切片失败或运行异常，就自动 fallback 到原始全量 prefill 路径

因此这一步的定位是：

> 一个安全的可观测实验分支，而不是默认启用的生产优化。

## 7. 现在怎么评估

### 7.1 Phase A: 先测 replay baseline 的 prefill 压力

命令：

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

重点查看：

- `latency.s2_generate.p50 / p95`
- `prefill.input_token_count.mean / p95`
- `prefill.image_token_count.mean / p95`
- `prefill.history_image_count.mean`

判断方式：

1. 如果 `generated_tokens_mean` 很小，但 `input_token_count` 和 `image_token_count` 很高，那么瓶颈主要在 prefill，不在 decode。
2. 如果 `history_image_count` 变化会带动 `s2_generate` 明显变化，说明视觉历史是主要成本来源。
3. 如果 `prompt_char_count` 变化很小、而 `s2_generate` 也几乎不变，则 prompt 文本长度不是主要瓶颈。

### 7.2 Phase B: 对比 HTTP loopback 的 disabled vs experimental

如果你之前已经在：

- `logs/habitat/test_dual_system_mini`

跑过闭环，并且目录里已有 `progress.json`，那么再次执行同一个 mini config 时，evaluator 会直接把这 8 个 episode 视为“已完成”并跳过，不会重新写 `replay_subset_v2`。这种情况下要先用一个新的 output 目录重跑闭环。

推荐直接使用：

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate habitat
cd /root/backup/InternNav
python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_mini_replay_v2_cfg.py
```

成功后，新的 manifest 会在：

- `/root/backup/InternNav/logs/habitat/test_dual_system_mini_replay_v2/replay_subset_v2/manifest_rank0.jsonl`

第一轮跑原始基线：

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

第二轮跑实验模式：

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

重点查看：

- `latency.server_model.p50 / p95`
- `latency.total_step.p50 / p95`
- `consistency.output_kind_match_rate`
- `breakdown.kv_cache_stats.lookdown_attempts`
- `breakdown.kv_cache_stats.lookdown_cache_hits`
- `breakdown.kv_cache_stats.lookdown_cache_fallbacks`
- `breakdown.kv_cache_stats.lookdown_prefix_mismatch`
- `breakdown.kv_cache_stats.lookdown_cache_exceptions`

判断方式：

1. 先看 `lookdown_attempts` 是否大于 0；否则这轮数据里根本没有触发可复用路径。
2. 再看 `lookdown_cache_hits / lookdown_attempts`；这是实验命中率。
3. 如果命中率高，但 `server_model.p50/p95` 几乎不降，说明当前瓶颈不在这条 follow-up。
4. 如果命中率低，而且主要是 `prefix_mismatch`，说明当前 conversation template 还不适合直接做 cache 续接。
5. 如果 latency 降了，但 `output_kind_match_rate` 或 `pixel_goal` 一致性明显变差，则这条 cache 方案暂时不能进主线。

## 8. 2026-03-12 当前已拿到的 Phase B baseline

截至 2026-03-12，目前已经拿到一份：

- `kv_cache_mode = disabled`

的 HTTP loopback baseline：

- `logs/habitat/test_dual_system_mini_replay_v2/http_loopback_kv_disabled.json`

核心结果：

- `num_steps = 645`
- `num_episodes = 8`
- `cold_start_load_seconds = 4.41s`
- `gpu_peak_memory_mb = 26428.87`
- `total_step p50 = 0.263s`
- `total_step p95 = 1.999s`
- `server_model p50 = 0.201s`
- `server_model p95 = 1.943s`
- `output_kind_match_rate = 0.826`
- `discrete_action_match_rate = 0.916`

更关键的是长尾拆解结果：

- `lookdown_used_steps = 93`
- 这 93 步的 `server_model p50 = 1.932s`
- 非 look-down 步的 `server_model p50 = 0.201s`
- 非 look-down 步的 `server_model p95 = 0.579s`

这意味着：

1. HTTP loopback 的主要长尾几乎全部来自 look-down continuation
2. 当前 Phase B 把 cache 实验聚焦在 `look_down=True` 路径上，是有数据支持的
3. 仅从 baseline 看，哪怕还没开 cache，这条路径本身就比普通 step 慢接近一个数量级

注意：

- 当前还只有 `disabled` 基线
- 还没有对应的 `lookdown_experimental` 对比结果

所以截至现在的正确结论是：

> Phase B 的目标路径选对了，但是否能真正靠 cache 降下来，还需要下一轮 AB 对比去验证。

## 9. 2026-03-12 `lookdown_experimental` 结果

今天已经补齐了 `lookdown_experimental` 的 AB 对比：

- `logs/habitat/test_dual_system_mini_replay_v2/http_loopback_kv_disabled.json`
- `logs/habitat/test_dual_system_mini_replay_v2/http_loopback_kv_lookdown.json`

### 9.1 结论先说

这轮结果说明：

1. Phase B 的“前缀对齐 + delta image 切分”逻辑是能走通的
2. 但真正进入 `generate(..., past_key_values=...)` 后，每次都会异常
3. 所以 93 次 look-down continuation 全部 fallback 回了原始全量 prefill
4. 因此时延和一致性几乎与 `disabled` 完全相同

也就是说，这轮实验的结论不是“KV cache 无效”，而是：

> 当前这条 Hugging Face 多模态 `generate + past_key_values` 接法，在 look-down continuation 上还不能直接工作。

### 9.2 关键数据

#### disabled baseline

- `total_step p50 = 0.2629s`
- `total_step p95 = 1.9987s`
- `server_model p50 = 0.2011s`
- `server_model p95 = 1.9435s`
- `output_kind_match_rate = 0.8264`
- `discrete_action_match_rate = 0.9157`

#### lookdown_experimental

- `total_step p50 = 0.2606s`
- `total_step p95 = 2.0015s`
- `server_model p50 = 0.1987s`
- `server_model p95 = 1.9469s`
- `output_kind_match_rate = 0.8264`
- `discrete_action_match_rate = 0.9157`

这些数字几乎完全一致，本身就说明：

- 当前实验分支没有真正改变最终执行路径

### 9.3 cache 统计最关键的解读

`lookdown_experimental` 的 `kv_cache_stats` 为：

- `lookdown_attempts = 93`
- `lookdown_cache_ready = 93`
- `lookdown_cache_hits = 93`
- `lookdown_cache_exceptions = 93`
- `lookdown_used_cache_total = 0`

这几个数字合起来的含义是：

1. 共有 93 次 look-down continuation
2. 这 93 次里，token 前缀检查都通过了
3. delta token 和最后一张图像的切片也都成功了
4. 但在真正执行 cache generate 时，93 次全部抛异常
5. 所以最终 93 次全部回退到原始全量 prefill

这说明当前问题不在：

- prefix 对不齐
- replay_v2 缺字段
- look-down 图像切分错误

而更可能在：

- Hugging Face `generate()` 对当前多模态 `past_key_values` 路径的兼容性
- vision token 与 cache position / rope 的拼接方式
- 当前 `generate()` API 不能简单接受“带 cache 的增量图像输入”

### 9.4 这轮实验的价值

虽然 latency 没降，但这轮并不是无效实验。它已经把问题缩小到了更具体的一层：

> 当前 Phase B 的阻塞点不是“找不到可复用路径”，而是“底层 generate 接口不接受这种多模态 cache 续接方式”。

后续首错 traceback 进一步说明了两个更具体的问题：

1. 默认不传 `cache_position` 时，HF 会按 `arange(past_length, input_len)` 推导，导致空张量并触发：
   - `IndexError: index -1 is out of bounds for dimension 0 with size 0`
2. 即使手动补 `cache_position`，Qwen2.5-VL 的 `prepare_inputs_for_generation()` 在：
   - `cache_position[0] != 0`

   时也会直接执行：

   - `model_inputs["pixel_values"] = None`

这意味着：

> 当前 HF/Qwen2.5-VL 这条标准 `generate + past_key_values` 路径，本身就会屏蔽 continuation 中的新图像输入。

因此，对于“look-down 后追加一张新图像再继续生成”这个需求，当前接口很可能是结构性不支持的。

这也是为什么现在不建议继续在这条 `past_key_values + multimodal generate` 接法上反复试。原因不是“它永远不可能成功”，而是：

1. 当前标准 HF/Qwen2.5-VL 接口已经明确假设 continuation 阶段不再接收新的图像输入
2. 而我们的 look-down continuation 恰好就是“文本续接 + 新图像”
3. 如果要强行打通，就需要绕过或重写：
   - `prepare_inputs_for_generation`
   - `cache_position`
   - vision token 注入
   - rope / position handling
4. 到那一步，其实已经不是“继续试现有接法”，而是在重写底层多模态 generate 流程

所以从工程投入比看，继续硬试这条路的性价比已经明显下降。

因此，下一步不应该继续调 `lookdown_attempts` 这类高层逻辑，而应该转向：

1. 先把异常栈完整打印出来
2. 确认是 `generate()` 本身、`pixel_values`、还是 `cache_position` / `rope` 出错
3. 如果确认是 Hugging Face 当前接口限制，就暂停这条直接 cache 续接路线
4. 优先转向 prefix reuse 或 backend 路线

### 9.5 当前阶段建议

截至这一步，推荐顺序调整为：

1. 先补 `lookdown_cache_exceptions` 的具体异常日志
2. 做一次最小 debug run，只跑少量 step，拿到完整报错
3. 如果异常证明这条 API 路线不可行，则不要继续在同一接口上硬攻
4. 直接把下一阶段重心转到：
   - prefix reuse / partial prefill
   - 更快的 S2 backend

## 11. 为什么接下来优先做 prefix reuse

`prefix reuse` 和 `past_key_values continuation` 最大的不同是：

- continuation cache 试图“接着上一次完整生成往后走”
- prefix reuse 试图“不要每次都重新编码那段完全没变的前缀”

它的优点在于：

1. 不依赖 HF/Qwen2.5-VL 的 continuation 语义
2. 不要求 continuation 阶段携带新图像
3. 可以把问题先收缩到“哪些 token 是静态的，值不值得复用”

也就是说，prefix reuse 并不需要先解决：

- 多模态 `past_key_values`
- continuation 图像输入

它更像是：

> 把每一步都重复编码的固定 instruction / 模板前缀先拆出来，再评估是否值得缓存或预编码。

## 12. prefix reuse 的代码落点

如果后续要做真正的 runtime 版本，优先应落在这几个位置：

### 12.1 输入构造层

- `internnav/habitat_extensions/vln/habitat_vln_evaluator.py`
- `internnav/agent/internvla_n1_agent_realworld.py`

目标：

1. 把当前 S2 prompt 拆成：
   - `static_prefix_text`
   - `dynamic_suffix_text`
   - `current_images`
2. 明确哪些部分在同一 episode 内不会变化
3. 为后面缓存“前缀 tokenization / 前缀 embedding / 前缀 prefill”做准备

### 12.2 离线评估层

- `scripts/eval/tools/benchmark_dualvln_replay.py`
- `scripts/eval/tools/benchmark_dualvln_http_loopback.py`

目标：

1. 在 replay 和 loopback 里复用相同的 prefix 拆分定义
2. 保证后续对比实验能在相同统计口径下进行

### 12.3 模型接口层

- `internnav/model/basemodel/internvla_n1/internvla_n1.py`

目标：

1. 如果 prefix reuse 继续向 runtime 版本推进，最终需要绕开纯 `generate()` 黑盒
2. 暴露更底层的“前缀先编码、后续再拼动态部分”的能力

当前还没有做到这一步，今天只做到分析版，不做高风险 runtime 改造。

## 13. prefix reuse 最小实验版

今天新增脚本：

- `scripts/eval/tools/analyze_dualvln_prefix_reuse.py`

这个脚本的定位不是直接提速，而是回答三个问题：

1. 真实 S2 决策里，静态文本前缀一共有多少 token
2. 这些静态前缀在总多模态输入里占比多少
3. 如果只复用文本前缀，不复用图像 token，理论上最多能省多少 prefill

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

它会输出：

- `full_input_tokens`
- `chat_text_tokens`
- `image_token_count`
- `static_prefix_tokens`
- `dynamic_suffix_tokens`
- `static_prefix_ratio_full`
- `static_prefix_ratio_text`
- `common_prefix_vs_prev_episode`
- `common_prefix_vs_prev_decision_group`
- `data_quality.records_with_missing_images`
- `data_quality.total_missing_images`

其中最重要的是：

- `static_prefix_ratio_full`

它表示：

> 在真实多模态输入里，第一张图像之前的静态文本前缀占总 token 的比例。

如果这个比例本来就很低，那么即使 prefix reuse 做成 runtime 优化，收益也不会太大；反之，这条路才值得继续做。

补充说明：

- 某些 `replay_v2` 里的 `decision_input_image_paths` 可能引用到未真正落盘的历史帧
- 当前分析脚本已经改成“只要某条记录缺任意历史图，就整条跳过并继续分析”
- 缺失数量会记录在：
  - `data_quality.records_with_missing_images`
  - `data_quality.total_missing_images`
- 被跳过的记录会在 details jsonl 中带：
  - `skipped_reason = missing_decision_images`

这意味着脚本不会再因为个别缺失历史帧直接中断，但在解读结果时，需要同时关注这两个数据质量指标。

## 14. 2026-03-12 prefix reuse 分析结果

今天已经跑完：

- `logs/habitat/test_dual_system_mini_replay_v2/prefix_reuse_analysis.json`

### 14.1 核心结论

这轮结果说明：

1. 静态文本前缀本身确实高度重复
2. 但在“总多模态输入”里，它只占很小一部分
3. 真正的大头仍然是图像 token，而不是文本前缀 token

因此当前最准确的判断是：

> prefix reuse 值得作为一个低风险辅助手段，但不太可能单独带来决定性的提速。

### 14.2 关键数据

有效分析记录：

- `145` 条

被跳过记录：

- `22` 条

缺失历史图总数：

- `32`

平均输入规模：

- `full_input_tokens mean = 3494.1`
- `chat_text_tokens mean = 145.5`
- `image_token_count mean = 13428.8`

静态前缀规模：

- `static_prefix_tokens mean = 103.5`
- `static_prefix_ratio_full mean = 0.0386`
- `static_prefix_ratio_text mean = 0.7153`

这几组数要放在一起理解：

1. 从“文本本身”看，静态前缀占到了约 `71.5%`
2. 但从“真实多模态输入”看，静态前缀只占约 `3.9%`
3. 这说明重复最多的是 instruction / 模板文字
4. 但这些文字在总成本里远没有图像 token 重要

### 14.3 normal decision vs look-down follow-up

普通决策步：

- `static_prefix_ratio_full mean = 0.0456`

look-down follow-up：

- `static_prefix_ratio_full mean = 0.0267`

这说明：

- look-down 路径里，静态文本前缀占比更低
- 也就是说，越是当前真正的慢路径，纯文本 prefix reuse 的理论收益越小

这和前面 HTTP loopback 的结论是吻合的：

- 当前长尾集中在 look-down continuation
- 但 look-down continuation 的主要负担更像是图像 token，而不是文本前缀

### 14.4 对路线选择的影响

截至这一步，当前几条路线的优先级应调整为：

1. 更快的 S2 backend
2. 更深一层的 prefix / prefill 优化，但不能只盯文本前缀
3. 纯文本 prefix reuse 可做，但收益预期要保守

换句话说：

- 如果只是做“缓存 instruction 文本前缀”
- 理论上最多只能动到总多模态输入的几个百分点

所以它可以做，但不应该被当成当前主力优化方向。

## 14.5 下一步转向 backend sweep

基于以上结果，接下来最合适的动作不是继续在纯文本 prefix reuse 上深挖，而是先做一轮低风险 backend sweep。

当前已新增统一 runner：

- `scripts/eval/tools/run_dualvln_backend_sweeps.py`

它统一暴露两类变量：

1. `attn_backend`
   - `flash_attention_2`
   - `sdpa`
   - `eager`
2. `processor_use_fast`
   - `auto`
   - `true`
   - `false`

之所以优先做这组实验，是因为：

1. 当前系统里已经观察到 `AutoProcessor` 的 slow processor 警告
2. 这组变量不会改动 DualVLN 的高层决策逻辑
3. 一旦其中某个组合在 replay 和 loopback 上都更快，就可以优先放回闭环验证

### 14.5.1 replay backend sweep

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate habitat
cd /root/backup/InternNav
export TOKENIZERS_PARALLELISM=false
python scripts/eval/tools/run_dualvln_backend_sweeps.py \
  --mode replay \
  --manifest /root/backup/InternNav/logs/habitat/test_dual_system_mini/replay_subset/manifest_rank0.jsonl \
  --model-path /root/backup/InternNav/checkpoints/InternVLA-N1-DualVLN \
  --output-dir /root/backup/InternNav/logs/habitat/test_dual_system_mini/backend_sweeps_replay \
  --base-path /root/backup/InternNav/logs \
  --attn-backends flash_attention_2 sdpa \
  --processor-fast-options auto true \
  --num-history 8 \
  --prompt-variant full \
  --max-new-tokens 128 \
  --verbose-every 20
```

### 14.5.2 loopback backend sweep

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate habitat
cd /root/backup/InternNav
export TOKENIZERS_PARALLELISM=false
python scripts/eval/tools/run_dualvln_backend_sweeps.py \
  --mode loopback \
  --manifest /root/backup/InternNav/logs/habitat/test_dual_system_mini_replay_v2/replay_subset_v2/manifest_rank0.jsonl \
  --model-path /root/backup/InternNav/checkpoints/InternVLA-N1-DualVLN \
  --output-dir /root/backup/InternNav/logs/habitat/test_dual_system_mini_replay_v2/backend_sweeps_loopback \
  --base-path /root/backup/InternNav/logs \
  --attn-backends flash_attention_2 sdpa \
  --processor-fast-options auto true \
  --num-history 8 \
  --plan-step-gap 4 \
  --use-recorded-lookdown \
  --kv-cache-mode disabled \
  --verbose-every 20
```

runner 会输出：

- `comparison_backend_replay.csv`
- `comparison_backend_loopback.csv`
- `comparison_backend_all.json`

建议优先关注：

- replay:
  - `s2_generate_p50 / p95`
  - `total_step_p50 / p95`
  - `gpu_peak_memory_mb`
  - `output_kind_match_rate`
- loopback:
  - `server_model_p50 / p95`
  - `total_step_p50 / p95`
  - `output_kind_match_rate`
  - `discrete_action_match_rate`

## 14.6 vLLM 作为下一条 backend 路线

在当前实验结论下，vLLM 是一条值得尽快验证的路线，而且不一定像“整套模型迁移”那样重，但切法必须收敛成：

- S2-only

而不是：

- 整套 DualVLN 直接替换为 vLLM serving

原因是当前 checkpoint 不是标准 `Qwen2.5-VL`：

- `model_type = internvla_n1`
- `architectures = ["InternVLAN1ForCausalLM"]`

同时权重里还包含大量 System1 相关模块：

- `latent_queries`
- `rgb_model`
- `memory_encoder`
- `rgb_resampler`
- `cond_projector`

因此这条路线的正确打开方式是：

1. 保留现有 S1 PyTorch 路径
2. 只把 S2 文本生成部分拿去做 vLLM 实验
3. 先在 replay 上验证 HF vs vLLM 的纯推理差异

为支持这一步，当前已新增：

- `scripts/eval/tools/check_dualvln_vllm_feasibility.py`
- `scripts/eval/tools/benchmark_dualvln_s2_backends.py`

### 14.6.1 feasibility + patched config view

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate habitat
cd /root/backup/InternNav
python scripts/eval/tools/check_dualvln_vllm_feasibility.py \
  --model-path /root/backup/InternNav/checkpoints/InternVLA-N1-DualVLN \
  --output /root/backup/InternNav/logs/habitat/test_dual_system_mini_replay_v2/vllm_feasibility.json \
  --patched-model-path /root/backup/InternNav/checkpoints/InternVLA-N1-DualVLN-qwen25vl-s2-view
```

这一步会生成一个：

- `InternVLA-N1-DualVLN-qwen25vl-s2-view`

它通过 symlink 复用原权重，只重写 `config.json`，把：

- `model_type -> qwen2_5_vl`
- `architectures -> ["Qwen2_5_VLForConditionalGeneration"]`

用于 S2-only 的 vLLM 尝试。

### 14.6.2 HF baseline

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate habitat
cd /root/backup/InternNav
export TOKENIZERS_PARALLELISM=false
python scripts/eval/tools/benchmark_dualvln_s2_backends.py \
  --backend hf \
  --manifest /root/backup/InternNav/logs/habitat/test_dual_system_mini/replay_subset/manifest_rank0.jsonl \
  --model-path /root/backup/InternNav/checkpoints/InternVLA-N1-DualVLN \
  --output /root/backup/InternNav/logs/habitat/test_dual_system_mini/s2_backend_hf.json \
  --base-path /root/backup/InternNav/logs \
  --num-history 8 \
  --prompt-variant full \
  --max-new-tokens 128 \
  --verbose-every 20
```

### 14.6.3 vLLM S2-only benchmark

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate habitat
cd /root/backup/InternNav
export TOKENIZERS_PARALLELISM=false
python scripts/eval/tools/benchmark_dualvln_s2_backends.py \
  --backend vllm \
  --manifest /root/backup/InternNav/logs/habitat/test_dual_system_mini/replay_subset/manifest_rank0.jsonl \
  --model-path /root/backup/InternNav/checkpoints/InternVLA-N1-DualVLN-qwen25vl-s2-view \
  --output /root/backup/InternNav/logs/habitat/test_dual_system_mini/s2_backend_vllm.json \
  --base-path /root/backup/InternNav/logs \
  --num-history 8 \
  --prompt-variant full \
  --max-new-tokens 128 \
  --processor-use-fast true \
  --trust-remote-code \
  --verbose-every 20
```

当前边界：

1. 本地环境里 `vllm` 尚未安装
2. 因此今天还没有跑出真实的 vLLM 数值
3. 但实验入口、可行性检查和 patched view 已经准备好

## 10. 2026-03-12 新增的异常调试能力

今天已经把 Phase B 的异常从“只记计数”补成了“可直接定位”的资产。

### 10.1 agent 现在会记录什么

`internnav/agent/internvla_n1_agent_realworld.py` 现在会在 cache exception 时记录：

- `exception_type`
- `exception_message`
- `traceback`
- `full_input_tokens`
- `delta_input_tokens`
- `full_image_tokens`
- `delta_image_tokens`
- `episode_idx`

这些信息会进入：

- `breakdown.kv_cache_stats.exception_type_counts`
- `breakdown.kv_cache_stats.exception_samples`

同时每个 loopback details 记录还会额外带：

- `kv_cache_event`

便于直接定位是哪一个 step 触发了哪种异常。

### 10.2 新增 stop-on-first-exception debug runner

新增脚本：

- `scripts/eval/tools/debug_dualvln_http_kv_cache.py`

用途：

- 复用现有 HTTP loopback 路径
- 一旦遇到首个 `cache_exception` 就立刻停止
- 输出该 step 的 scene/episode/record/step 元信息
- 同时保存完整 traceback 和当时的 token / image 规模

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

这个脚本的目标很单纯：

> 不再跑完整轮 benchmark，而是先抓到第一条真实异常栈，确认当前阻塞点到底是哪一层接口。

### 10.3 当前这份 debug json 怎么解读

如果你看到终端里只打印了少量几次 `output ...` 然后脚本就结束，而没有直接在终端抛 traceback，这通常是正常的。

原因是这个脚本的设计就是：

1. 一旦捕获到首个 cache exception
2. 就把异常信息写进 `http_loopback_kv_debug_first_exception.json`
3. 然后正常 `return`

所以它会表现为：

- 终端看上去“平静结束”
- 但 json 里已经记录了首错

当前这份 json 的关键信息是：

- `steps_processed = 2`
- `first_exception.record_id = 1`
- `first_exception.step_id = 1`
- `lookdown_used = true`
- `exception_type = IndexError`
- `exception_message = index -1 is out of bounds for dimension 0 with size 0`

此外：

- `s2_calls_total = 4`

并不表示处理了 4 个 replay step，而是表示 agent 一共触发了 4 次 S2 调用。这里包含：

1. loopback 初始化时的 warmup 调用
2. 第 1 个 replay step 的正常 S2
3. 第 2 个 replay step 的正常 S2
4. 第 2 个 replay step 里的 look-down follow-up S2

因此“只看到 4 个 output”与 `steps_processed = 2` 并不矛盾。

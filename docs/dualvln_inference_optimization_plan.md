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

## 5.2 2026-03-12 后的策略更新

截至 2026-03-12，`num_history` 闭环 sweep 已经把“删上下文换速度”这条路线基本排除了，因此当前优化顺序应更新为：

1. 先跑 KV cache Phase A，量化 prefill 压力
2. 再跑 KV cache Phase B，验证 `look_down=True` follow-up 是否存在可复用空间
3. 如果 Phase B 的命中率或收益太差，再转向 prefix reuse / 更快 backend
4. 量化继续放在后面

当前已经有可直接执行的两条实验链：

- `benchmark_dualvln_replay.py`
  负责 Phase A 的输入压力统计
- `benchmark_dualvln_http_loopback.py`
  负责 Phase B 的 `disabled` vs `lookdown_experimental` 对比

截至 2026-03-12，目前已经拿到一份 Phase B 的 `disabled` baseline。它说明：

- 普通非 look-down step 的 `server_model p50` 约为 `0.20s`
- look-down continuation 的 `server_model p50` 约为 `1.93s`

因此当前 loopback 的长尾几乎全部集中在 look-down follow-up 上。这进一步支持了当前策略：

> 优先把第一条 cache 实验放在 `look_down=True` 路径，而不是平均优化所有 step。

但同一天的 `lookdown_experimental` AB 对比也表明：

- `lookdown_attempts = 93`
- `lookdown_cache_hits = 93`
- `lookdown_cache_exceptions = 93`
- `lookdown_used_cache_total = 0`

这意味着当前实验实现虽然能够识别并构造出可复用的 delta 输入，但在真正调用带 `past_key_values` 的多模态 `generate()` 时全部异常，最终全部 fallback 到原始路径。

因此当前策略应再收紧一步：

1. 先补异常日志，确认 API 阻塞点
2. 如果确认是 Hugging Face 多模态 cache 路径不兼容，则不要继续在同一接法上投入太多
3. 更优先考虑：
   - prefix reuse / partial prefill
   - 更快的 S2 backend

目前首错 traceback 已经显示出一个很强的结构性信号：

- 默认 cache continuation 会因为空 `cache_position` 直接报错
- 即使手动补 `cache_position`，Qwen2.5-VL 在 `cache_position[0] != 0` 时会主动把 `pixel_values` 置空

这意味着当前标准 HF 接口并不自然支持：

> 带新图像输入的多模态 cache continuation

因此这条路线的工程风险已经进一步升高。

这也是为什么当前不建议继续在 `past_key_values + multimodal generate` 这条接法上反复试：

1. 这条路线当前不是“小修小补就能通”的状态
2. 它已经进入“需要改底层多模态 generate 假设”的状态
3. 而 prefix reuse 不依赖 continuation 图像输入语义，更适合作为下一条低风险路线

截至 2026-03-12，prefix reuse 已经有一个最小实验版本：

- `scripts/eval/tools/analyze_dualvln_prefix_reuse.py`

它不直接改模型推理，而是先用 `replay_v2` 精确统计：

- 静态文本前缀 token 数
- 图像 token 数
- 静态前缀在总多模态输入中的占比
- 与前一决策相比可复用的文本前缀长度

这一步的意义是先回答：

> prefix reuse 在 DualVLN 里有没有足够大的理论收益，值得继续做成 runtime 优化。

为支持这一步排查，当前代码已经新增：

- loopback summary 内的 `exception_type_counts` 和 `exception_samples`
- per-step details 内的 `kv_cache_event`
- `debug_dualvln_http_kv_cache.py` 的 stop-on-first-exception 调试脚本

因此当前最推荐的动作不是继续跑更多完整 benchmark，而是：

1. 先抓到第一条完整异常栈
2. 确认是否是 HF 多模态 cache API 的结构性限制
3. 再决定是否继续推进 cache，还是切 backend / prefix reuse

随后已完成一轮 prefix reuse 理论空间分析，结果来自：

- [prefix_reuse_analysis.json](/root/backup/InternNav/logs/habitat/test_dual_system_mini_replay_v2/prefix_reuse_analysis.json)

关键结果：

- `metadata.num_records = 167`
- 实际完成统计的记录数：`145`
- 因图像缺失被跳过：`22`
- 缺失图像总数：`32`

整体输入结构：

- `full_input_tokens mean = 3494.10`
- `chat_text_tokens mean = 145.48`
- `image_token_count mean = 13428.83`
- `static_prefix_tokens mean = 103.50`

关键比例：

- `static_prefix_ratio_full mean = 0.0386`
- `static_prefix_ratio_text mean = 0.7153`

按路径拆分：

- `normal_decision.static_prefix_ratio_full mean = 0.0456`
- `lookdown_followup.static_prefix_ratio_full mean = 0.0267`

这组结果说明：

1. 静态 instruction/template 文本在“文本空间”里确实高度重复
2. 但在“真实多模态输入”里，其占比只有约 `3.9%`
3. 当前 prefill 主成本主要来自图像 token，而不是静态文本前缀
4. 真正的慢路径 `lookdown_followup` 上，纯文本 prefix reuse 的理论收益更低

因此当前优先级应调整为：

1. 优先探索更快的 System 2 backend
2. 其次考虑更广义的 partial prefill / multimodal prefill 优化
3. 纯文本 prefix reuse 作为低风险辅助手段保留，但不应被视为主攻方向

## 5.2 下一步：S2 backend 低风险 sweep

基于当前结果，最先值得做的 backend 实验，不是直接接入全新的 serving 框架，而是先扫两类低风险变量：

1. `attn_backend`
   - `flash_attention_2`
   - `sdpa`
   - `eager`
2. `processor_use_fast`
   - `auto`
   - `true`
   - `false`

这样做的原因是：

1. 当前日志里已经出现了 `AutoProcessor` 的 slow processor 警告
2. 这类变量不会改变高层 DualVLN 逻辑，也不依赖新的部署基础设施
3. 可以先判断“仅靠 HF loader / processor 层面的 backend 变量，能不能拿到可观收益”

当前代码已经支持这些参数，并新增了统一 runner：

- `scripts/eval/tools/run_dualvln_backend_sweeps.py`

### 5.2.1 replay sweep 命令

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

### 5.2.2 loopback sweep 命令

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

### 5.2.3 结果文件

runner 会自动生成：

- `comparison_backend_replay.csv`
- `comparison_backend_loopback.csv`
- `comparison_backend_all.json`

建议优先比较：

1. replay:
   - `s2_generate_p50 / p95`
   - `total_step_p50 / p95`
   - `gpu_peak_memory_mb`
   - `output_kind_match_rate`
2. loopback:
   - `server_model_p50 / p95`
   - `total_step_p50 / p95`
   - `output_kind_match_rate`
   - `discrete_action_match_rate`

## 5.3 vLLM 路线

在 2026-03-12 的当前判断下，vLLM 是一条值得尽快验证的路线，而且工程量并不一定很重，但前提是切法要对。

### 5.3.1 为什么值得试

当前不建议继续主攻：

- `past_key_values + multimodal generate`

因为这条路已经暴露出较强的底层接口冲突，尤其是在：

- look-down continuation 里要“复用旧 cache + 再喂一张新图”

而 vLLM 更适合处理：

- 更快的 S2 文本生成
- 更成熟的 KV / prefix 管理
- 多请求推理 backend 优化

### 5.3.2 为什么不能一上来整套 DualVLN 迁过去

本地 checkpoint 的 [config.json](/root/backup/InternNav/checkpoints/InternVLA-N1-DualVLN/config.json) 显示：

- `model_type = internvla_n1`
- `architectures = ["InternVLAN1ForCausalLM"]`

这不是标准的 `Qwen2.5-VL` checkpoint。与此同时，权重里还带有：

- `model.latent_queries`
- `model.rgb_model.*`
- `model.memory_encoder.*`
- `model.rgb_resampler.*`
- `model.cond_projector.*`

也就是说：

1. 整个 DualVLN 不是一个“直接拿标准 Qwen2.5-VL serving 就能替换”的模型
2. 但其中的 S2 文本生成部分，仍然高度建立在 Qwen2.5-VL 视觉-语言主干上
3. 因此更合理的切法是：
   - 只把 S2 拿去试 vLLM
   - S1 继续保留现有 PyTorch 路径

### 5.3.3 当前新增脚本

已新增：

- `scripts/eval/tools/check_dualvln_vllm_feasibility.py`
- `scripts/eval/tools/benchmark_dualvln_s2_backends.py`

其中：

`check_dualvln_vllm_feasibility.py` 用于：

- 检查当前 checkpoint 是否是标准 Qwen2.5-VL
- 统计是否包含大量 DualVLN/System1 额外模块
- 可选生成一个“patched standard-Qwen2.5-VL config view”

`benchmark_dualvln_s2_backends.py` 用于：

- 在同一份 replay manifest 上对比 `hf` 和 `vllm`
- 只测 S2 文本生成，不牵涉 S1
- 输出：
  - `cold_start_load_seconds`
  - `s2_generate p50/p95`
  - `tokens_per_second`
  - `output_kind_match_rate`
  - `text_exact_match_rate`
  - `discrete_action_match_rate`

### 5.3.4 建议执行顺序

先做 feasibility + patched view：

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate habitat
cd /root/backup/InternNav
python scripts/eval/tools/check_dualvln_vllm_feasibility.py \
  --model-path /root/backup/InternNav/checkpoints/InternVLA-N1-DualVLN \
  --output /root/backup/InternNav/logs/habitat/test_dual_system_mini_replay_v2/vllm_feasibility.json \
  --patched-model-path /root/backup/InternNav/checkpoints/InternVLA-N1-DualVLN-qwen25vl-s2-view
```

再跑 HF baseline：

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

如果已安装 vLLM，再跑 S2-only vLLM：

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

### 5.3.5 当前边界

截至目前：

1. 这些脚本已经完成并通过静态检查
2. 本地环境尚未安装 `vllm`
3. 因此还没有完成真实的 HF vs vLLM 数值对比

所以现在的定位仍然是：

- 已完成实验入口与兼容性准备
- 下一步等安装 `vllm` 后做真实 A/B

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

## 11. 2026-03-12 闭环验证后的策略更新

今天的 Habitat 闭环 sweep 已经证明：

- `num_history` 压缩虽然能显著降低离线 `S2` 推理时延
- 但回到闭环后会带来明显功能退化

因此当前策略应更新为：

1. 保持 `num_history=8`
2. 不再把 history 压缩当作主线优化方向
3. 正式把重点转到 `KV cache / prefix reuse / backend`

## 12. 日志体积控制

当前大日志的主要来源不是 summary，而是：

- `replay_subset`
- `replay_subset_v2`

尤其在闭环 sweep 中，会快速膨胀到几十 GB。

因此：

1. 如果当前目标只是做闭环功能对比，不需要导出 replay
2. 应在闭环 runner 中使用：

```bash
--replay-num-episodes 0
```

当前代码已支持：

- 当 `replay_num_episodes=0` 时自动关闭 replay 导出

同时新增了清理脚本：

- `scripts/eval/tools/prune_habitat_logs.py`

可用于删除 bulky replay 资产，但保留：

- `result.json`
- `runtime_summary_rank0.json`
- `comparison_num_history_closed_loop.csv`

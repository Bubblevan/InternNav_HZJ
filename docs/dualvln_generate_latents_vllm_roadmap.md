# DualVLN `generate_latents()` -> vLLM Roadmap

本文档专门讨论下一阶段的核心问题：

> 如何把 `generate_latents()` 从“本地完整 DualVLN”迁移为“依赖 vLLM hidden states”，最终实现只开一个 vLLM server。

目标不是在一个大改动里一次完成，而是分阶段、逐步做精度对齐。

---

## 1. 最终目标

当前架构的问题是：

- vLLM server 中有一份 Qwen2.5-VL backbone
- Habitat 本地评估进程里又有一份完整 DualVLN

这意味着：

- 显存浪费
- 推理链路重复
- `generate_latents()` 的迁移价值非常高

最终目标架构应当是：

1. **vLLM server**
   - 负责 Qwen2.5-VL 多模态 backbone
   - 负责 S2 文本生成
   - 最好还能提供 `generate_latents()` 所需的 hidden states

2. **本地轻量模块**
   - 只保留 `latent_queries`
   - 只保留 S1 / DiT 相关模块
   - 不再加载完整 7B Qwen2.5-VL 主干

简化成一句话：

> 最终希望 Habitat 侧不再起第二份完整 DualVLN，只保留一个 vLLM server + 一本地 S1。

---

## 2. 为什么优先盯住 `generate_latents()`

因为当前 `generate_traj()` 虽然也慢，但它本质上是 DiT / denoising loop，不属于 vLLM 的核心能力范围。

而 `generate_latents()` 不同：

- 它不是自回归文本生成
- 但它仍然是 **Qwen2.5-VL backbone forward**
- 本质上是“跑完整 backbone，然后取最后一层 hidden states 的最后几个位置”

所以从结构上说，`generate_latents()` 是最可能被进一步迁移到 vLLM 一侧的那个环节。

---

## 3. `generate_latents()` 的本质再确认一次

当前实现核心流程是：

1. 对 `output_ids` 做 token embedding
2. 对图片做 visual embedding
3. 替换 image placeholder
4. 在序列末尾追加 4 个 `TRAJ_TOKEN`
5. 用 `latent_queries` 替换这 4 个 token 的 embedding
6. 跑 Qwen2.5-VL backbone
7. 取最后一层 hidden states 的最后 4 个位置

换句话说：

> `generate_latents()` 本质不是“生成”，而是“构造特殊输入后做一次 forward，并读 hidden states”

这里有一个后续实现里非常关键、也很容易被忽略的点：

- 序列末尾虽然会追加 4 个 `TRAJ_TOKEN (151667)`
- 但真正送进 backbone 的并不是这 4 个 token 的普通 embedding lookup
- 而是 `latent_queries` 这个可学习参数，直接替换掉这 4 个位置的输入 embedding

也就是说，`generate_latents()` 真正依赖的是：

1. 固定的完整多模态上下文 token 序列
2. 正确的视觉 embedding 替换
3. 末尾 4 个位置的 **custom input embeddings (`latent_queries`)**
4. 然后才是 last-layer hidden states 的最后 4 个位置

这意味着后续即便 vLLM 能做到：

- 多模态输入跑通
- token-level outputs 跑通
- `baseline_output_ids` 对齐跑通

也 still 不自动等于已经能复现 `generate_latents()`，因为还差一个核心能力：

- **能否对末尾 4 个位置注入自定义 `inputs_embeds`，而不是只靠 token id**

这也是原作者那句判断的来源：

> generate latent 其实就是取的大模型最后的 hidden state。

这个判断很有启发性，但当前还只是“工程假设”，不是已验证结论。

---

## 4. 当前最大的技术障碍

障碍不是数学定义不清楚，而是 **vLLM 标准接口不直接暴露所需能力**。

当前我们已经能用：

- `/v1/chat/completions`

拿到文本输出。

但要迁 `generate_latents()`，我们还需要这些能力中的至少一部分：

1. 能得到 per-token hidden states
2. 最好能控制或复现 `inputs_embeds`
3. 最好能兼容多模态输入与 position / rope 逻辑

而这些都不是 OpenAI-compatible chat API 默认提供的。

---

## 5. 分阶段推进原则

这条线不要一上来就改 Habitat 主流程，而应严格分成三层：

1. **定义层**
   - 先把当前 HF `generate_latents()` 的输入输出完全钉死
2. **离线验证层**
   - 在 Habitat 外部做小脚本对比 hidden states
3. **闭环替换层**
   - 只有离线数值对齐足够好，才进入 Habitat 闭环替换

原则是：

> 先在小范围里把“等价性”说清楚，再去动闭环。

---

## 6. 建议的分阶段路线

### Phase A：把 HF `generate_latents()` 做成明确基准

这一阶段只做一件事：

> 让当前本地 HF 路径成为稳定、可复用、可比对的黄金参考。

建议输出一个离线测试样本包，至少包含：

- 原始 `conversation_history` / prompt
- `output_ids`
- `pixel_values`
- `image_grid_thw`
- HF `generate_latents()` 输出的 tensor

当前第一步可以先基于 `replay_subset/manifest_rank0.jsonl` 完成，配套脚本建议放在：

- `scripts/eval/tools/export_hf_generate_latents_baseline.py`

一个最小运行示例可以是：

```bash
cd /root/backup/InternNav
conda activate habitat
python scripts/eval/tools/export_hf_generate_latents_baseline.py \
  --manifest logs/habitat/test_dual_system_mini/replay_subset/manifest_rank0.jsonl \
  --model-path checkpoints/InternVLA-N1-DualVLN \
  --output-dir logs/habitat/hf_generate_latents_baseline_replay1 \
  --device cuda:0 \
  --max-samples 8
```

导出后的每个 `.pt` 样本现在建议至少包含两套信息：

- `baseline_llm_output` 重建出的：
  - `baseline_output_ids`
  - `baseline_latent`
- 当前 HF 重新 `generate()` 得到的：
  - `hf_generate_output_ids`
  - `hf_generate_latent`

这样可以把两个问题拆开看：

1. **原始闭环记录的 latent 基准是什么**
2. **当前 HF 复现实验与原始闭环记录偏差有多大**

这一层完成后，后面每个候选方案都可以直接对这组样本做比较。

导出后建议立刻再跑一次质量分析：

```bash
cd /root/backup/InternNav
conda activate habitat
python scripts/eval/tools/analyze_generate_latents_baseline.py \
  --input-dir logs/habitat/hf_generate_latents_baseline_replay1 \
  --output logs/habitat/hf_generate_latents_baseline_replay1_analysis.json
```

这个分析脚本当前会帮助回答三个问题：

1. `baseline_llm_output` 与当前 HF `generate()` 文本是否一致
2. pixel goal 是否一致
3. `baseline_latent` 与 `hf_generate_latent` 是否 already allclose

如果这一步都不稳定，就说明 replay1 导出的这批样本还不能直接拿来做后续 vLLM hidden-state 严格对齐，必须先继续筛选样本或收紧导出逻辑。

特别注意 replay1 的一个坑：

- 很多 `pixel_goal` 记录本质上是 **look-down follow-up 第二轮**
- 如果导出脚本没有恢复：
  - 上一轮 assistant 文本
  - 当前 lookdown 图像

那么就会错误地把第二轮 `pixel_goal` 当成第一轮单轮输入去重放，常见现象就是：

- `text_exact_match_rate = 0`
- `pixel_goal_match_rate = 0`
- `latent_allclose_rate = 0`

这时优先说明的是**样本构造错位**，而不一定是模型本身已经不可复现。

在当前脚本已经补上 look-down follow-up 恢复之后，如果仍然出现下面这种现象：

- `text_exact_match_rate = 0`
- `pixel_goal_match_rate = 0`
- 但 HF 重放输出仍然大多是**相近的 pixel 坐标**

那更常见的解释是：

1. replay1 仍然没有保存足够多的“精确决策时上下文”
2. 当前重放更像是**语义接近复现**，不是**逐 token 严格复现**
3. 由于 `generate_latents()` 对 `output_ids` 非常敏感，只要坐标 token 略有偏移，latent 就很可能完全不 allclose

因此需要明确区分 replay1 的两个用途：

### replay1 适合做的

- 导出固定的 `baseline_output_ids`
- 导出固定的 `baseline_latent`
- 作为后续 vLLM hidden-state 路线的**静态黄金参考样本包**

### replay1 不适合做的

- 作为“当前 HF `generate()` 必须逐 token exact match 原闭环输出”的严格评测基准
- 作为“只要 HF regenerate 不 exact match，就说明导出失败”的判据

换句话说：

> replay1 更适合当“固定样本导出器”，不适合当“严格闭环复现实验器”。

成功标准：

- 能稳定复现同一个 HF latent 输出
- 输出样本可重复加载
- 基准样本可以被后续比较器脚本直接消费

基准样本导出后，建议统一通过比较器来做候选路线对齐，而不是每条路线都各写一套 diff 逻辑。配套脚本建议放在：

- `scripts/eval/tools/compare_generate_latents_candidates.py`

一个最小运行示例可以是：

```bash
cd /root/backup/InternNav
conda activate habitat
python scripts/eval/tools/compare_generate_latents_candidates.py \
  --input-dir logs/habitat/hf_generate_latents_baseline_replay1 \
  --candidate-key hf_generate_latent \
  --output logs/habitat/hf_generate_latents_baseline_replay1_compare_hf_generate.json
```

这样至少能先把当前样本包内部的：

- `baseline_latent`
- `hf_generate_latent`

做统一汇总。后面如果接入 vLLM 候选路线，只需要把候选 latent 保存成同名 `.pt` 文件，再通过 `--external-dir` 接到这套比较器里即可。

### Phase B：确认 vLLM 侧到底能不能拿到需要的 hidden states

这一阶段先不追求替换，只回答能力问题。

需要验证的最小问题是：

1. `vllm.LLM` Python API 能否拿到 per-token hidden states？
2. vLLM embedding/encode 路径是否可用于多模态 Qwen2.5-VL？
3. 是否能在不经过 chat API 的情况下，复现接近 HF 的 backbone 输入？

这一阶段可能的结论有三种：

1. **可行**
   - 能拿到需要的 hidden states
2. **部分可行**
   - 只能拿 pooled embeddings 或能力不完整
3. **暂不可行**
   - 现成接口无法满足要求

这一阶段的产出应该是“能力清单”和“接口边界”，而不是急着改主流程。

当前基于本地 `/root/backup/vllm` 源码的初步检查，可以先记下两个重要事实：

1. `vllm.LLM` 确实存在 `encode()` / `embed()` / pooling 相关接口，不是只有 chat/completions。
2. 但 `LLM.encode()` 明确要求 `runner_type == "pooling"`，返回的是 `PoolingRequestOutput`，语义上更接近 pooling / embedding / token-level pooling，而不是“标准生成模型任意位置最后层 hidden states”。

这意味着现在还不能直接下结论说：

- “vLLM 已经原生支持 `generate_latents()` 需要的 hidden states”

更准确的表述应该是：

- “vLLM 源码里存在一条值得继续探的 pooling / token-embed 路线，但它是否适用于多模态 Qwen2.5-VL、以及是否能严格等价到 `outputs.hidden_states[-1][:, -4:, :]`，目前都还没有验证。”

另外，针对当前实际 checkpoint：

- `checkpoints/InternVLA-N1-DualVLN-qwen25vl-s2-view`

已经补了一个静态探测脚本：

- `scripts/eval/tools/probe_vllm_pooling_capability.py`

当前静态探测结果是：

- architecture = `Qwen2_5_VLForConditionalGeneration`
- 结论 = `likely_requires_convert_embed`

也就是说，在本地 vLLM 这版源码里，它目前更像是：

- **需要尝试 `--runner pooling --convert embed` 的候选**

而不是：

- **直接已有原生 pooling registry 的模型**

因此，Phase B 的首要问题已经可以进一步收紧为：

> `Qwen2_5_VLForConditionalGeneration` 在 `--runner pooling --convert embed` 下，是否还能正确处理多模态输入，并返回可用于离线 latent 对齐的 token-level outputs？

在这个问题之后，马上接着的第二个关键问题应该是：

> 即使 token-level outputs 可得，vLLM 是否允许我们把末尾 4 个 `TRAJ_TOKEN` 的输入 embedding 替换成 `latent_queries`？

如果答案是否定的，那么当前这条路线最多只能做到：

- 接近 `baseline_output_ids` 的 token-level 表示提取

而不能做到：

- 与 HF `generate_latents()` 严格等价

### Phase C：做候选 latent 路径的离线数值对齐

如果 Phase B 表明 vLLM 有可用路径，那么下一步就是离线 allclose 对齐。

最小对齐目标：

- 输入相同样本
- 输出 shape 一致：`[1, 4, 3584]`
- 与 HF 基准做：
  - `max_abs_diff`
  - `mean_abs_diff`
  - `torch.allclose(atol=1e-4)` 或更合理阈值

这一阶段最重要的是：

- 不要只看任务指标
- 先把 latent tensor 本身对齐清楚

### Phase D：只替换 `generate_latents()`，闭环 A/B 再跑一次

只有在离线 latent 对齐足够好后，才进入这个阶段。

此时闭环链路变成：

- vLLM：`generate()`
- vLLM 或其扩展路径：latent hidden states
- 本地：S1 / DiT

这一步要重新做：

- 文本级对齐
- latent 数值级对齐
- 闭环任务级对齐

如果这里通过，才算真正消掉了“本地完整 DualVLN backbone”。

---

## 7. 可能的技术路径

### 路径 1：vLLM Python API / encode 路线

这是目前侵入性相对较低的第一候选。

想验证的问题：

- `vllm.LLM` 是否能不走 chat API，而是走更底层的 encode / embedding 路径
- 是否能返回足够细粒度的 per-token 输出

优点：

- 不一定需要改 server

缺点：

- 很可能接口能力不够

### 路径 2：给 vLLM server 增加 hidden-state 返回接口

这是更工程化但侵入性更高的路线。

思路是：

- 保留现有 server 架构
- 在服务端增加一个自定义 endpoint
- 返回 `generate_latents()` 所需 hidden states

优点：

- 一旦打通，更符合“最终只保留一个 server”的方向

缺点：

- 需要深入改 vLLM server 或自定义 wrapper

### 路径 3：过渡态瘦身版本地 backbone

如果 vLLM 短期拿不到 hidden states，仍然可以考虑中间态：

- vLLM 继续负责文本 `generate()`
- 本地只保留一个更瘦的 Qwen backbone forward 路径
- 不再保留完整 DualVLN 全模型加载方式

这不是最终目标，但可能是显存优化的中间站。

---

## 8. 精度对齐要怎么做

这条线的精度对齐建议分三层。

### 8.1 文本级

先确认：

- HF `generate()` 与 vLLM `generate()` 输出文本是否一致
- token ids 是否一致

这一步已经有现成脚本基础。

### 8.2 latent 数值级

这是核心。

需要固定同一组：

- prompt
- images
- `output_ids`
- `pixel_values`
- `image_grid_thw`

然后比较：

- HF `generate_latents()`
- 候选 vLLM latent 路径

只要 latent 数值级还没站稳，就不要进入大规模闭环结论。

### 8.3 闭环任务级

在文本级和 latent 级都基本稳定后，再看：

- SR
- SPL
- NE
- 平均步数

这样闭环结果才更有解释力。

---

## 9. 当前最推荐的下一步

如果只做一个最小可推进动作，我建议优先做：

> **先写一个独立小脚本，把 HF `generate_latents()` 的输入样本和输出 latent 保存下来，作为后续所有 vLLM hidden-state 方案的黄金基准。**

理由很简单：

- 这一步不需要先决定 vLLM 路线
- 但能立刻把后续所有实验都变得可比较
- 也最符合“先做精度对齐，再做大改”的原则

---

## 10. 一句话路线图

这条线不应该理解成“直接把 `generate_latents()` 塞给 vLLM”，而应该理解成：

1. 先把 HF latent 基准钉死
2. 再确认 vLLM 是否拿得到等价 hidden states
3. 再做离线 latent allclose
4. 最后才进 Habitat 闭环

最终目标仍然明确：

> 让 Habitat 侧不再加载第二份完整 DualVLN，只保留一个 vLLM server 和一本地 S1。

# DualVLN vLLM 深度分析与下一步规划

本文档基于与 InternNav 作者的技术交流，以及对 `generate()`、`generate_latents()`、`generate_traj()` 三个函数的逐行源码分析，总结当前认知、厘清技术边界、并给出下一步的精确执行计划。

---

## 1. 作者交流核心结论

### 1.1 KV cache / prefix caching 的收益预期

> kv cache 优化的是多轮对话形式下的速度，dualvln 当前只有产生 pixel 的时候才是多轮对话，不然正常就是一轮对话

**含义**：DualVLN 的大部分推理步都是"给 instruction + 历史图 + 当前图 → 一次性输出"的单轮形态，不是标准聊天机器人那种用户和 assistant 交替多轮、前缀高度重复的场景。只有 `look_down=True` 时才出现真正的多轮对话（在已有 context 上追加 assistant 回复 + 新一轮用户图+文本）。

**结论**：prefix caching 的理论收益集中在 look-down continuation 路径，对普通决策步收益有限。这与此前 `analyze_dualvln_prefix_reuse.py` 的数据一致（静态文本前缀占总多模态输入仅约 3.9%）。

### 1.2 `generate_latents()` 的本质

> generate latent 其实就是取的大模型最后的 hidden state，你把整个 dualvln 的标准 qwen25vl 分开，然后直接调用 vllm 来跑，并取 hidden state，应该就是等价于 generate latent

**含义**：`generate_latents()` 不是标准的自回归文本生成。它的核心操作是：跑一遍 Qwen2.5-VL backbone forward，取最后一层 hidden states 的特定位置（latent query tokens）。作者认为如果能从 vLLM 拿到等价的 hidden states，就不需要在本地再加载一遍完整模型单独跑这个函数。

**关键词："应该"**——这是思路级判断，不是已验证结论。需要自己确认数值等价性。

### 1.3 `generate_traj()` 不是 vLLM 的甜蜜点

> generate traj 这个函数不太好跑 kv cache 优化，可以试一下 dit 优化的方式

**含义**：`generate_traj()` 是 Flow Matching / DiT 的 iterative denoising 循环（10 步 Euler step），不是标准 transformer decoder 的 token-by-token 生成。它依赖 `action_encoder`、`traj_dit`、`action_decoder`、`noise_scheduler` 等 DualVLN 自定义模块。这类结构不在 vLLM 的优化范围内，应当用 DiT 专用的加速手段（如 torch.compile、CUDA graph、distillation 减步数等）。

### 1.4 精度对齐

> 最好写好测试用例，每一步修改后都得对齐精度

**含义**：当前阶段的改动已经进入"重构推理路径"，不是简单调参。每一步拆分都可能引入数值偏移，必须有可运行的等价性测试。

---

## 2. 三个函数的精确解剖

以下基于 `internvla_n1.py` 和 `internvla_n1_policy.py` 源码的逐行分析。

### 2.1 `model.generate()`

**调用位置**：`s2_step()` 第 237–245 行

**输入**：

| 张量 | 来源 | 说明 |
|------|------|------|
| `inputs.input_ids` | `processor(text, images)` | `[1, seq_len]`，含文本 token 和 image placeholder token |
| `inputs.pixel_values` | 同上 | 各图 patch 展平后的像素值 |
| `inputs.image_grid_thw` | 同上 | 每张图的 (t, h, w) grid 信息 |
| `inputs.attention_mask` | 同上 | `[1, seq_len]` |

**输出**：

| 张量 | shape | 说明 |
|------|-------|------|
| `output_ids` (sequences) | `[1, seq_len + gen_len]` | prompt token ids + 生成的 token ids |

**解码后**：`self.llm_output` — 纯文本字符串，如 `"(182, 213)"` 或 `"STOP"` 或 `"↑"` 等

**性质**：

- 标准 Qwen2.5-VL autoregressive generation
- 逐 token 解码，`do_sample=False`（greedy）
- `max_new_tokens=128`，实际平均生成约 2–5 tokens
- **适合 vLLM 加速**：这正是 vLLM 的核心优化点

**当前 vLLM 替换状态**：已完成（通过 `/v1/chat/completions` HTTP 调用）

---

### 2.2 `model.generate_latents()`

**调用位置**：`s2_step()` 第 260 行，仅在 S2 输出 pixel goal 时触发

**输入**：

| 张量 | shape (典型值) | 说明 |
|------|----------------|------|
| `output_ids` | `[1, seq_len + gen_len]` | 来自 `generate()` 的完整序列（prompt + 生成） |
| `pixel_values` | `[num_patches, 1176]` | 所有图像的 patch 特征 |
| `image_grid_thw` | `[num_images, 3]` | 每张图的 grid 维度 |

**内部执行流程**（逐步）：

```
1. text_embeds = embed_tokens(output_ids)       → [1, total_len, 3584]
2. latent_queries = self.latent_queries.repeat(bs, 1, 1)  → [1, 4, 3584]（n_query=4, hidden_size=3584）
3. image_embeds = self.visual(pixel_values, grid_thw)  → [num_visual_tokens, 3584]
4. text_embeds[image_positions] = image_embeds   → 替换 image placeholder
5. input_ids 末尾追加 4 个 TRAJ_TOKEN (151667)
6. text_embeds = cat([text_embeds, latent_queries], dim=1)  → [1, total_len + 4, 3584]
7. position_ids = get_rope_index(input_ids, image_grid_thw)
8. outputs = self.model(inputs_embeds=..., position_ids=..., output_hidden_states=True)
9. hidden_states = outputs.hidden_states[-1][:, -4:, :]  → [1, 4, 3584]
```

**输出**：`[1, n_query, hidden_size]` = `[1, 4, 3584]` — 最后一层 hidden states 的最后 4 个位置

**性质**：

- **不是自回归生成**，而是**一次 forward pass**
- 核心依赖的模块：
  - `embed_tokens()`：标准 Qwen2.5-VL 词嵌入 ✓ vLLM 有
  - `self.visual()`：标准 Qwen2.5-VL ViT ✓ vLLM 有
  - `self.latent_queries`：DualVLN 自定义参数 ✗ vLLM 没有
  - `self.model()`：标准 Qwen2.5-VL backbone ✓ vLLM 有（但需要 `output_hidden_states=True`）
  - `get_rope_index()`：标准 Qwen2.5-VL RoPE ✓ vLLM 有
- 关键技巧：末尾追加 4 个 `TRAJ_TOKEN`，用 `latent_queries` embedding 代替，然后取最后 4 个 hidden states
- **不直接适合 vLLM**：vLLM 的 `/v1/chat/completions` 不暴露 hidden states；需要用更底层的 API 或做模型拆分

---

### 2.3 `model.generate_traj()`

**调用位置**：`s1_step_latent()` 第 271 行

**输入**（nextdit_async 模式）：

| 张量 | shape (典型值) | 说明 |
|------|----------------|------|
| `traj_latents` | `[1, 4, 3584]` | 来自 `generate_latents()` 的 hidden states |
| `images_dp` | `[1, 2, 224, 224, 3]` | pixel goal 帧 + 当前帧 |
| `depths_dp` | `[1, 2, 224, 224, 1]` | 对应 depth |

**内部执行流程**（nextdit_async 模式，10 步 Flow Matching）：

```
1. traj_latents = cond_projector(traj_latents)           → [1, 4, 384]
2. images_dp → rgb_model(DINOv2) → images_dp_feat       → [1, 2, 256, 384]
3. memory_feat = memory_encoder(images_dp_feat)           → [1, 512, 384]
4. memory_feat = cat([images_dp_feat, memory_feat])       → [1, 512, 768]
5. memory_tokens = rgb_resampler/QFormer(memory_feat)     → [1, 32, 768]
6. hidden_states = cat([memory_tokens, traj_latents])     → [1, 36, 768?→384]
7. latents = randn([32, 32, 3])                           → 32 条候选轨迹，每条 32 步 × (x,y,θ)
8. for t in 10 Euler steps:
       noise_pred = traj_dit(action_features, t, hidden_states)
       latents = scheduler.step(noise_pred, t, latents)
9. return latents                                          → [32, 32, 3]
```

**输出**：`[num_sample_trajs, predict_step_nums, 3]` = `[32, 32, 3]` — 32 条候选轨迹

**性质**：

- **Flow Matching iterative denoising**，不是 transformer autoregressive
- 依赖全部 S1 自定义模块：`cond_projector`、`rgb_model (DINOv2)`、`memory_encoder`、`rgb_resampler (QFormer)`、`traj_dit (NextDiT)`、`action_encoder/decoder`、`noise_scheduler`
- 10 次迭代循环、32 条并行采样
- **完全不适合 vLLM**：不是 token generation 问题
- 优化方向：`torch.compile`、CUDA graph、减少 denoising 步数 (distillation)、DiT 专用加速

---

## 3. 当前架构与改造路线

### 3.1 当前已完成：S2 generate → vLLM（文本生成层）

```
已替换路径：
  s2_step() → _vllm_generate() → vLLM /v1/chat/completions → 文本输出

仍在本地：
  s2_step() → processor() → inputs  (本地构造 input_ids / pixel_values)
  s2_step() → generate_latents(output_ids, pixel_values, image_grid_thw)  (本地 forward)
  s1_step_latent() → generate_traj(latents, images, depths)  (本地 DiT)
```

**问题**：本地仍然需要加载**完整 DualVLN 模型**来跑 `generate_latents()`，导致 GPU 上同时存在 vLLM 副本 + 本地副本，显存浪费严重。

### 3.2 作者建议的进阶方向：把 `generate_latents()` 也迁到 vLLM

核心思路：`generate_latents()` 的 backbone forward 与 `generate()` 的 prefill 阶段在做**几乎相同的事情**——都是跑一遍 Qwen2.5-VL 编码器拿 hidden states。区别只是 `generate_latents()` 在末尾额外追加了 4 个 latent query token。

**如果能从 vLLM 拿到 hidden states**，本地只需保留：

- `latent_queries` 参数（4 × 3584，约 56KB）
- S1 全部自定义模块（约 1.5GB）

**而不再需要**在本地加载 7B 级 Qwen2.5-VL 主干。

### 3.3 拆分后的目标架构

```
┌──────────────────────────────────────────────────────┐
│  eval.py（Habitat 评估主循环）                         │
│  ┌─────────────────────────────────────────────────┐ │
│  │  s2_step()                                       │ │
│  │   ├─ 构造 messages                               │ │
│  │   ├─ vLLM: generate text  ──────── HTTP ──┐      │ │
│  │   ├─ vLLM: get hidden states ─── HTTP ──┐ │      │ │
│  │   └─ 本地: latent_queries 拼接           │ │      │ │
│  │                                          │ │      │ │
│  │  s1_step_latent()                        │ │      │ │
│  │   └─ 本地: generate_traj (DiT)           │ │      │ │
│  └──────────────────────────────────────────┘ │      │
│  GPU 占用: ~2GB (S1 模块 + latent_queries)    │      │
└───────────────────────────────────────────────┼──┼───┘
                                                │  │
                    ┌───────────────────────────┐│  │
                    │  vLLM Server :8001        ││  │
                    │  Qwen2.5-VL 完整 backbone ◄┘  │
                    │  + text generate          ◄───┘
                    │  + hidden states 提取
                    │  GPU: ~20GB
                    └───────────────────────────┘
```

---

## 4. 技术可行性分析：从 vLLM 获取 hidden states

### 4.1 vLLM 标准 API 的限制

vLLM 的 `/v1/chat/completions` 和 `/v1/completions` 只返回生成的 token/text，**不暴露 hidden states**。

### 4.2 可能的路径

| 路径 | 侵入性 | 说明 |
|------|--------|------|
| **A. vLLM `/v1/completions` + `prompt_logprobs` + 自定义钩子** | 高 | 需要 fork 或 plugin vLLM，在 forward 中截取 hidden states 并返回。vLLM 的 model runner 内部可以拿到 hidden states，但没有标准接口暴露出来。 |
| **B. 用 vLLM 的 `LLM` 类做离线推理（非 serve 模式）** | 中 | 直接用 `vllm.LLM` 做 Python 调用，绕过 HTTP。但标准 `LLM.generate()` 也不返回 hidden states。可能需要用 `LLM.encode()` 或自定义 `SamplingParams`。 |
| **C. 两阶段：先 vLLM generate 拿文本，再本地用轻量 forward 拿 hidden states** | 低 | 就是当前的做法。代价是本地仍需加载完整模型。 |
| **D. 本地只保留 backbone 权重的"瘦身版"** | 中 | 不走 vLLM 取 hidden states，而是在本地加载一个去掉 lm_head 的 backbone（不做 generate，只做 forward），与 vLLM 的 generate 共存。显存略省，但改动不小。 |
| **E. vLLM embedding API** | 低 | vLLM 已支持 embedding 模型的 `/v1/embeddings`，某些配置下可以拿到 pooled hidden states。但能否拿到 per-token hidden states、且兼容 Qwen2.5-VL 多模态，需要验证。 |

### 4.3 当前判断

- **短期（立即可做）**：维持当前方案 C，即 vLLM 只管 generate text，`generate_latents()` 继续走本地。这已经能加速 S2 的主瓶颈（autoregressive decode）。
- **中期**：探索路径 B 或 E，尝试从 vLLM 的 Python API 层面获取 hidden states，避免 HTTP 开销和二次加载。
- **长期**：如果 vLLM 社区或 Qwen 团队补上 hidden states 输出接口，直接迁移。

---

## 5. 下一步执行计划

### Phase 0: 精度对齐基础设施（最高优先级）

**目标**：建立三层测试框架，确保后续每一步改动都有数值基准可比。

**具体测试用例**：

| 测试层 | 测试内容 | 判断标准 |
|--------|----------|----------|
| **接口级** | 给定相同输入（rgb, instruction, history），本地 HF `generate()` vs vLLM `generate()` 输出的文本是否完全一致 | exact match |
| **数值级** | 给定相同 `output_ids` + `pixel_values` + `image_grid_thw`，本地 `generate_latents()` 输出的 `[1, 4, 3584]` 张量 | `torch.allclose(atol=1e-4)` |
| **任务级** | 跑相同 8 episode 闭环评估，对比 SR / SPL / NE | 差值 < 统计噪声 |

**实现**：

1. 编写 `scripts/eval/tools/test_vllm_s2_equivalence.py`
   - 用 replay manifest 中的几条数据
   - 分别用 HF backend 和 vLLM backend 跑 `s2_step()`
   - 输出：文本匹配率、action 匹配率、pixel goal 匹配率
2. 跑一轮 8 episode 闭环，用 `habitat_dual_system_vllm_cfg.py`，与 `habitat_dual_system_cfg.py` 的 baseline 比 SR/SPL/NE

### Phase 1: vLLM S2 generate 验证（当前阶段）

**目标**：确认当前 `s2_vllm_url` 分支在 Habitat 闭环中功能正确。

**步骤**：

1. 停掉当前 GPU 上的 eval 进程，释放显存
2. vLLM serve patched 模型（port 8001）
3. 跑 `habitat_dual_system_vllm_cfg.py`（8 episode mini）
4. 对比 baseline 的 SR / SPL / NE

**成功标准**：SR/SPL/NE 与不用 vLLM 时无显著差异。

### Phase 2: `generate_latents()` 迁移可行性验证

**目标**：确认"从 vLLM backbone 取 hidden states"是否真的等价于本地 `generate_latents()`。

**步骤**：

1. 在本地脚本中，用 `vllm.LLM` 的 Python API（非 HTTP）加载 patched 模型
2. 构造与 `generate_latents()` 相同的输入：在 prompt 末尾追加 4 个 TRAJ_TOKEN，用 latent_queries embedding 替换
3. 跑 forward，取 `outputs.hidden_states[-1][:, -4:, :]`
4. 与本地 HF 的 `generate_latents()` 输出做 `torch.allclose` 对比

**需要回答的问题**：

- vLLM 的 `LLM` 类是否支持自定义 `inputs_embeds`？（可能不支持，需要看代码）
- 是否有接口能拿到 per-layer hidden states？
- 如果都不支持，是否可以通过 vLLM 的 `encode()` / embedding 接口间接获取？

**可能的结论**：

- 如果等价且可获取 → 可以去掉本地 7B backbone，显存显著下降
- 如果不等价或接口不支持 → 维持当前方案，寻找其他优化手段

### Phase 3: `generate_traj()` 独立优化

**目标**：在不改推理逻辑的前提下加速 DiT 轨迹生成。

**候选方向**（与 vLLM 无关）：

| 方向 | 预期收益 | 工程量 |
|------|----------|--------|
| `torch.compile(traj_dit)` | 中等 | 低 |
| CUDA graph 包装 10 步 Euler loop | 中等 | 中 |
| 减少 `num_inference_steps` (10→5→3) + 验证精度 | 高（线性加速） | 低 |
| 减少 `num_sample_trajs` (32→16→8) | 高（线性加速） | 低 |
| DiT distillation（一步生成） | 非常高 | 非常高（需训练） |

**优先级**：先做低成本的 `num_inference_steps` / `num_sample_trajs` 减少实验，再考虑 compile/graph。

---

## 6. 函数解剖汇总表

| 维度 | `generate()` | `generate_latents()` | `generate_traj()` |
|------|-------------|---------------------|-------------------|
| **触发时机** | 每次 S2 决策 | 仅 pixel goal 输出时 | 仅 pixel goal 输出时 |
| **本质** | 标准 autoregressive 文本生成 | 单次 backbone forward + 取 hidden states | Flow Matching iterative denoising (DiT) |
| **输入** | input_ids, pixel_values, image_grid_thw | output_ids, pixel_values, image_grid_thw | traj_latents `[1,4,3584]`, images `[1,2,224,224,3]`, depths |
| **输出** | output_ids `[1, seq+gen]` → 文本 | hidden_states `[1, 4, 3584]` | trajectories `[32, 32, 3]` |
| **依赖 Qwen2.5-VL** | embed_tokens, visual, LM backbone, lm_head | embed_tokens, visual, LM backbone (无 lm_head) | 无 |
| **依赖 DualVLN 自定义** | 无 | latent_queries | cond_projector, rgb_model, memory_encoder, rgb_resampler, traj_dit, action_encoder/decoder, noise_scheduler |
| **计算特征** | prefill + 逐 token decode | 一次完整 prefill（无 decode） | 10 × DiT forward + 32 × 并行 |
| **主要时延来源** | 多图 prefill + decode | 一次多图 prefill | 10 × DiT forward |
| **适合 vLLM** | **是**，核心甜蜜点 | **部分**，backbone 等价但需 hidden states 接口 | **否**，完全不是 token generation |
| **优化方向** | vLLM serve | 模型拆分：vLLM backbone + 本地 latent_queries | DiT 加速：compile / graph / 减步数 / distill |

---

## 7. 风险与注意事项

### 7.1 vLLM 与 HF 的 generate 输出差异

- vLLM 使用 V1 engine，CUDA graph + torch.compile，内部数值路径可能与 HF 的 eager forward 有微小差异
- `temperature=0.0` 在 vLLM 中等价于 greedy，与 HF 的 `do_sample=False` 语义相同
- 但 BF16 运算顺序、FlashAttention 版本差异等可能导致非 bit-exact 输出
- **因此必须做 Phase 0 的等价性测试**

### 7.2 `generate_latents()` 中 output_ids 的拼接

当前 vLLM 分支中，`output_ids` 是通过 `tokenizer.encode(llm_output)` 重建的。如果 vLLM 和 HF 的 tokenizer 行为不完全一致（如 whitespace 处理、special token 插入等），可能导致 `output_ids` 微妙不同，进而影响 `generate_latents()` 的 embedding lookup 和 position_ids 计算。

**缓解措施**：在 Phase 0 中明确验证 token ids 是否一致。

### 7.3 显存预算

| 场景 | vLLM | 本地模型 | 合计 | L20 余量 |
|------|------|----------|------|----------|
| 当前（vLLM + 完整本地 DualVLN） | ~21GB | ~17GB | ~38GB | ~11GB |
| 目标（vLLM + 仅 S1 模块） | ~21GB | ~2GB | ~23GB | ~26GB |

如果 Phase 2 成功，显存下降约 15GB，可以更激进地调高 vLLM 的 `gpu-memory-utilization` 以获得更大 KV cache 容量。

---

## 8. 时间线建议

| 阶段 | 内容 | 预估时间 | 前置依赖 |
|------|------|----------|----------|
| Phase 0 | 精度对齐测试脚本 + 基线数据 | 0.5–1 天 | 无 |
| Phase 1 | vLLM S2 闭环验证 | 0.5 天 | Phase 0 |
| Phase 2 | `generate_latents()` vLLM 可行性 | 1–2 天 | Phase 1 |
| Phase 3 | `generate_traj()` DiT 加速 | 1–2 天 | 可与 Phase 2 并行 |

**核心原则**：每个 Phase 完成后都跑一次 Phase 0 的等价性测试，确认没引入回归。

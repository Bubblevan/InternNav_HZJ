# DualVLN `generate_latents()` 的 vLLM Hidden-State 能力探测

这份文档记录当前对本地 `/root/backup/vllm` 源码的第一轮结论。目标不是直接宣布“能做”或“不能做”，而是把现阶段已经确认的能力边界写清楚，避免后续重复判断。

---

## 1. 这一步想回答什么

问题不是“vLLM 能不能聊天”，而是：

> vLLM 能不能以某种方式给出足够接近 `generate_latents()` 所需的 token 级 hidden states，从而替掉 Habitat 侧本地完整 Qwen2.5-VL backbone？

我们关心的不是 pooled embedding，而是接近下面这件事：

- 对固定多模态输入做 backbone forward
- 取最后一层 hidden states
- 只拿最后追加的 4 个 `TRAJ_TOKEN` 位置

---

## 2. 当前已经确认的事实

基于本地 vLLM 源码，可以先确认这几件事。

### 2.1 `LLM.encode()` 不是假的，确实存在

本地源码里的 [llm.py](/root/backup/vllm/vllm/entrypoints/llm.py) 明确有：

- `LLM.encode()`
- `LLM.embed()`
- `pooling_task="token_embed"`
- `pooling_task="token_classify"`

这说明 vLLM 并不只有 `/v1/chat/completions` 这一条路。

### 2.2 `token_embed` 返回的是 token 级二维张量

源码里的 tokenwise pooler 说明得很清楚：

- `token_embed` 最终输出 shape 是 `[n_tokens, embedding_dimension]`
- 不是单个 pooled 向量

这对 `generate_latents()` 很重要，因为我们本来就只想拿最后几个特殊 token 的表示。

### 2.3 `STEP` token pooling 支持按 token 位置做筛选

本地源码里的 `StepPool` 还支持：

- `step_tag_id`
- `returned_token_ids`

也就是说，从机制上看，vLLM 的 tokenwise pooling 不是只能“把整段序列全吐出来”，它本身就支持做 token 级过滤。

### 2.4 多模态路径在源码里不是完全空白

`LLM.score()` 的 late-interaction 路线里有一句很关键的注释：

- `handles both text and multimodal`

而且本地 registry 里也有多模态 embedding / late-interaction 模型项。这至少说明：

- “pooling/token_embed 完全不能碰多模态” 这个结论现在下不了

---

## 3. 当前还不能确认的事

虽然上面这些信号都偏正面，但离“已经能等价 `generate_latents()`”还差一大截。

### 3.1 `LLM.encode()` 只支持 pooling runner

源码明确限制：

- `LLM.encode()` only supported for pooling models

这意味着如果当前服务起来的是标准 generate runner，那么：

- 不能直接指望现成 `LLM.encode()` 在同一套普通生成服务上顺手返回 hidden states

### 3.2 `token_embed` 不等于“最后层原始 hidden states”

虽然 `token_embed` 返回 token 级矩阵，但它还会经过 pooling head，可能包含：

- dtype 变换
- projector
- normalize
- 以及其他 pooling 配置影响

所以它更准确地说是：

- token-level pooled embeddings

而不是已经证明等价于：

- `outputs.hidden_states[-1][:, -4:, :]`

### 3.3 Qwen2.5-VL 是否能无损走 pooling runner 还没实测

本地 vLLM registry 里能看到：

- `Qwen2VLForConditionalGeneration` 在 embedding models 里
- 但我们当前实际 checkpoint 是 Qwen2.5-VL 体系

源码层面还能看到：

- 生成模型如果不是原生 pooling model，理论上可以走 `--runner pooling --convert embed`

但这只是“架构上可能可转”，不等于已经验证：

- 多模态图像输入还能正常工作
- M-RoPE / position 逻辑还能和 HF `generate_latents()` 对齐
- 返回的 token 表示正好就是我们想要的那四个位置

---

## 4. 目前最稳妥的判断

现在最准确的说法应该是：

> 本地 vLLM 源码里存在一条值得继续探的 pooling / token_embed 路线，它有可能提供 token 级表示，也不排除支持多模态；但它是否能严格等价 `generate_latents()` 需要的最后层 hidden states，目前还没有被证明。

所以这一步的结论不是：

- “能做”

也不是：

- “不能做”

而是：

- “值得继续做，而且已经有比 chat API 更接近目标的入口了”

---

## 5. 为什么还不能直接宣布成功

`generate_latents()` 要求其实非常具体：

1. 输入不是普通文本，而是多模态序列
2. 序列末尾还要人工追加 4 个 `TRAJ_TOKEN`
3. 这 4 个位置的 embedding 还要被 `latent_queries` 替换
4. 最终要读的是 last-layer hidden states 的最后 4 个位置

而当前 vLLM pooling 路线，即便能吐 token 级矩阵，也还没有证明这几件事全部成立：

1. 能否精确控制追加 token 的位置
2. 能否精确控制这几个 token 的输入 embedding
3. 输出是不是 HF backbone 的“原始最后层 hidden states”，而不是被 pooler 处理过的版本

这三条里，任何一条不成立，都会让它和 `generate_latents()` 失去严格等价性。

---

## 6. 下一步应该怎么做

下一步不应该直接改 Habitat 主链，而应该继续做能力探测。

推荐顺序是：

1. 先用静态探测脚本确认当前 checkpoint 在本地 vLLM 架构分类里是“直通 pooling 候选”还是“只能 convert-to-embed”
2. 再做一个最小 runtime probe，验证 pooling runner 下的多模态输入能不能真正跑起来
3. 如果能跑，再看 token 级输出的 shape 和 token 过滤能力
4. 最后才把候选输出接到 [compare_generate_latents_candidates.py](/root/backup/InternNav/scripts/eval/tools/compare_generate_latents_candidates.py) 做 baseline_latent 对齐

---

## 7. 配套脚本

为了把这一步固定下来，我加了一个静态探测脚本：

- [probe_vllm_pooling_capability.py](/root/backup/InternNav/scripts/eval/tools/probe_vllm_pooling_capability.py)

最小用法：

```bash
cd /root/backup/InternNav
python scripts/eval/tools/probe_vllm_pooling_capability.py \
  --model-path checkpoints/InternVLA-N1-DualVLN-qwen25vl-s2-view \
  --output logs/habitat/vllm_pooling_probe_qwen25vl.json
```

这个脚本不会真正跑 vLLM，只做静态判断，回答：

- 当前 checkpoint 的 architecture 看起来是直通 pooling、可 convert、还是未知
- 这条路线下一步最该验证什么

在静态探测之后，又补了一个真正的 runtime probe：

- [probe_vllm_pooling_runtime.py](/root/backup/InternNav/scripts/eval/tools/probe_vllm_pooling_runtime.py)

它会做这些事：

1. 从 replay1 manifest 里重建一条 `pixel_goal` 样本
2. 自动处理 look-down follow-up 第二轮
3. 用 `AutoProcessor.apply_chat_template(...)` 生成和当前 HF 路径同风格的 prompt text
4. 用本地 `vllm.LLM(..., runner="pooling", convert="embed")` 尝试跑 `pooling_task="token_embed"`
5. 输出：
   - 是否成功
   - prompt token 数
   - token-level 输出 shape
   - 或者明确的异常类型和报错

推荐运行方式：

```bash
cd /root/backup/InternNav
source /root/.venv/bin/activate
python scripts/eval/tools/probe_vllm_pooling_runtime.py \
  --manifest logs/habitat/test_dual_system_mini/replay_subset/manifest_rank0.jsonl \
  --model-path checkpoints/InternVLA-N1-DualVLN-qwen25vl-s2-view \
  --sample-index 0 \
  --gpu-memory-utilization 0.45 \
  --max-model-len 4096 \
  --output logs/habitat/vllm_pooling_runtime_probe_qwen25vl.json
```

当前已经跑出的第一条结果可以直接记下来：

- checkpoint: `checkpoints/InternVLA-N1-DualVLN-qwen25vl-s2-view`
- sample: `zsNo4HB9uLZ / episode 1 / step 3`
- 输入图片数: `5`
- 结果: **runtime probe succeeded**
- `prompt_token_count = 2078`
- `token_embed_shape = [2078, 3584]`
- `token_embed_dtype = torch.float32`

这说明至少有一件事已经不再是假设：

> 本地 vLLM 的 `runner="pooling" + convert="embed" + pooling_task="token_embed"`，对当前这版多模态 Qwen2.5-VL checkpoint 是真的可以跑起来的，而且会返回逐 token 的二维表示。

这是一个明显的正向信号，因为它把问题从“这条路是不是完全走不通”推进成了“这条路返回的 token-level outputs 是否足够等价 `generate_latents()` 需要的 hidden states”。

如果这一步成功，最先看的不是任务指标，而是这三个字段：

- `runtime_probe.success`
- `runtime_probe.prompt_token_count`
- `runtime_probe.token_embed_shape`

只要这里都拿不到，就说明还没资格谈“如何等价 `generate_latents()` 的最后 4 个位置”。

如果这里成功，再继续问：

1. 输出是不是稳定的 token-level matrix
2. 能不能筛到我们关心的特殊 token 位置
3. 能不能进一步逼近 `baseline_latent`

同时，这次 probe 也暴露出了一个很重要的风险点：

- baseline 样本里的 `input_ids` 长度是 `2073`
- 但 runtime probe 的 `prompt_token_count` 是 `2078`

这个偏差虽然不大，但已经足够说明：

- 现在的 HF baseline 路径和 vLLM pooling 路径，在 **multimodal processor / placeholder expansion** 这一层并没有做到完全同构

尤其是日志里还明确提示：

- vLLM 默认把 `Qwen2VLImageProcessor` 当成 fast processor 加载
- 这可能带来轻微输出差异

这意味着当前最合理的结论是：

1. **运行能力已验证**
2. **严格 token 对齐尚未验证**
3. **processor 差异已经是下一阶段必须显式处理的风险源**

因此，现阶段还不能直接把这条路线视为“已经可以替换 `generate_latents()`”，但它已经足够值得继续往下做第二层 probe：

- 尝试把固定 `baseline_output_ids` 送进 vLLM pooling 路线
- 观察输出长度和 token 对应关系是否稳定
- 再决定是否继续研究如何逼近最后 4 个 `TRAJ_TOKEN` 位置

这里还要额外强调一层原理风险：

- HF `generate_latents()` 并不是简单在 `baseline_output_ids` 后面再拼 4 个 `TRAJ_TOKEN`
- 它会把这 4 个位置的输入 embedding 直接替换成 `latent_queries`

所以第二层 probe 的意义不是“直接复现 latent”，而是更谨慎地回答：

1. 固定 `baseline_output_ids` 时，vLLM 能不能返回稳定的 token-level outputs？
2. 如果再额外拼 4 个原始 `TRAJ_TOKEN`，vLLM 能不能稳定返回最后 4 个位置？
3. 这些输出距离真正的 HF `generate_latents()` 还差的，是不是正好就是 `latent_queries` 注入这一步？

为此已经补了下一个脚本：

- [probe_vllm_token_prompt_runtime.py](/root/backup/InternNav/scripts/eval/tools/probe_vllm_token_prompt_runtime.py)

它会直接读取导出的 baseline `.pt` 样本里的：

- `baseline_output_ids`

并尝试把它们作为 `TokensPrompt(prompt_token_ids=...)` 喂给 vLLM 的 pooling/token_embed 路线。可选地，也可以再追加 4 个原始 `TRAJ_TOKEN`，用于观察长度和位置是否稳定。

---

## 8. 当前一句话结论

现阶段最靠谱的方向不是继续纠结 replay1 regenerate exact match，而是：

> 把 replay1 导出的 `baseline_latent` 当作黄金参考，然后逐步验证 vLLM 的 pooling / token_embed 路线能否给出接近 `generate_latents()` 的 token 级表示。  

---

## 9. `prompt_embeds` 路线的最新阻塞

为了进一步逼近 HF `generate_latents()` 的输入形式，又补了一个 probe：

- [probe_vllm_prompt_embeds_runtime.py](/root/backup/InternNav/scripts/eval/tools/probe_vllm_prompt_embeds_runtime.py)

这条 probe 做的事情是：

1. 从 vLLM 内部模型拿到 `baseline_output_ids (+ 4 个 TRAJ_TOKEN)` 的 token embedding
2. 重新以 `EmbedsPrompt(prompt_embeds=...)` 的方式送回 vLLM
3. 观察：
   - 请求长度能否严格保持
   - 自定义最后 4 个 embedding 是否能影响最后 4 个输出

这轮实验得到的不是数值偏差，而是一个非常明确的结构性 blocker：

- worker 侧在初始化 Qwen2.5-VL 的 M-RoPE 位置时，会报  
  `AssertionError: M-RoPE requires prompt_token_ids to be available.`

这个错误和源码能完全对上：

- [input_processor.py](/root/backup/vllm/vllm/v1/engine/input_processor.py#L246)
  对 `type == "embeds"` 的请求，vLLM 会显式设定：
  - `prompt_token_ids = None`
  - `prompt_embeds = decoder_inputs["prompt_embeds"]`
- [gpu_model_runner.py](/root/backup/vllm/vllm/v1/worker/gpu_model_runner.py#L1417)
  Qwen2.5-VL 初始化 M-RoPE 时又硬要求：
  - `req_state.prompt_token_ids is not None`

因此当前公开 `EmbedsPrompt` 路线在 Qwen2.5-VL 上的结论可以明确写成：

1. **模型 forward 本身并不排斥 `inputs_embeds`**
2. **但公开 `EmbedsPrompt` 请求在进入 worker 前就丢掉了 `prompt_token_ids`**
3. **M-RoPE 位置构造依赖 token ids**
4. **所以这条路当前会在 worker 侧提前失败**

这条结果很重要，因为它说明：

> 当前 `prompt_embeds` 路线不是“完全没希望”，而是“已经精确收敛到一个可 patch 的源码问题”。

最小源码 patch 方向现在已经比较清楚：

1. 允许 embeds 请求可选地同时携带 `prompt_token_ids`
2. 不要在 `input_processor` 中对 embeds 请求强制把 `prompt_token_ids` 设为 `None`
3. worker 继续使用 `prompt_token_ids` 构造 M-RoPE 位置
4. backbone 实际输入仍然走 `prompt_embeds`

如果这一步打通，`prompt_embeds` 路线才有资格继续回答下一个核心问题：

- 末尾 4 个位置的 custom embedding 注入，是否能够逼近 HF `latent_queries`

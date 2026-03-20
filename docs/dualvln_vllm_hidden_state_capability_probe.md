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

## 9. `prompt_embeds` 路线的两阶段阻塞

为了进一步逼近 HF `generate_latents()` 的输入形式，又补了一个 probe：

- [probe_vllm_prompt_embeds_runtime.py](/root/backup/InternNav/scripts/eval/tools/probe_vllm_prompt_embeds_runtime.py)

这条 probe 做的事情是：

1. 从 vLLM 内部模型拿到 `baseline_output_ids (+ 4 个 TRAJ_TOKEN)` 的 token embedding
2. 重新以 `EmbedsPrompt(prompt_embeds=...)` 的方式送回 vLLM
3. 观察：
   - 请求长度能否严格保持
   - 自定义最后 4 个 embedding 是否能影响最后 4 个输出

第一轮实验得到的不是数值偏差，而是一个非常明确的结构性 blocker：

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

在这个 patch 之后，又继续做了一轮 probe。第二轮结果是：

- `success = true`
- `requested_prompt_length = 2084`
- `base_output.token_embed_shape = [2084, 3584]`
- `custom_output.token_embed_shape = [2084, 3584]`
- `shape_matches_requested_length = true`

这说明：

1. `prompt_embeds + prompt_token_ids` 这条路已经能穿过 Qwen2.5-VL 的 M-RoPE 初始化
2. 长度也终于可以被严格保住，不再像 `TokensPrompt + multi_modal_data` 那样被重新展开

但第二轮 probe 同时暴露了一个更核心的新 blocker：

- `output_diff.max_abs_diff = 0.0`
- `last_n_output_diff.max_abs_diff = 0.0`

也就是说：

- 即便我们显式修改了最后 4 个 prompt embedding
- vLLM 返回的 token-level 输出仍然完全不变

这通常不意味着“模型对输入完全不敏感”，而更可能意味着：

- 当前 worker 在真正执行模型前，又把这些位置重新按 token id 做了一次 embedding lookup
- 从而把外部提供的 `prompt_embeds` 覆盖回去了

源码上也能对上这个判断：

- [gpu_input_batch.py](/root/backup/vllm/vllm/v1/worker/gpu_input_batch.py)
  当前只要 `prompt_token_ids` 存在，就会把 prompt 段标成 `is_token_ids=True`
- [gpu_model_runner.py](/root/backup/vllm/vllm/v1/worker/gpu_model_runner.py)
  在 `enable_prompt_embeds` 路径下，又会对这些 `is_token_ids=True` 的位置重新执行 `embed_input_ids(...)`

因此目前 `prompt_embeds` 路线的真实状态是：

1. **第一个 blocker 已解决**
   - `prompt_token_ids` 能保留，M-RoPE 可以构造
2. **第二个 blocker 已暴露**
   - 自定义 prompt embeddings 还没有真正生效，仍会被 token-id embedding 覆盖

这意味着下一步最小 patch 方向进一步收敛成：

1. 保留 `prompt_token_ids` 给 M-RoPE / metadata 使用
2. 但对已经提供了 `prompt_embeds` 的 prompt 位置，不再把它们标记为 `is_token_ids=True`
3. 这样 worker 才不会在执行前把这些位置重新 embedding 一次

如果这一步再打通，`prompt_embeds` 路线才有资格继续回答下一个核心问题：

- 末尾 4 个位置的 custom embedding 注入，是否能够逼近 HF `latent_queries`

在继续补上这层 patch 后，我们又加了一轮文件级 dump，把链路细化到：

- `gpu_input_batch_add_request`
- `gpu_model_runner_prepare_inputs`
- `gpu_model_runner_pre_forward`
- `gpu_model_runner_post_forward`
- `gpu_model_runner_pooler_output`

第三轮 probe 的结论比前两轮更明确：

1. 在 `gpu_input_batch_add_request`，两次请求的 `prompt_embeds` 差异仍然存在
2. 但一到 `gpu_model_runner_prepare_inputs`，`inputs_embeds_gpu` 就已经完全一致
3. 后面的 `pooler_output` 因此继续完全一致

这说明问题已经不在：

- pooler 是否抹平了差异
- 或 transformer forward 是否不敏感

而在更前面的输入准备阶段。

进一步对源码分支的定位表明，真正把差异覆盖掉的是 Qwen2.5-VL 的 multimodal 分支：

1. 该模型 `supports_mm_inputs=True`
2. `_preprocess()` 会优先走 multimodal 路线
3. 该路线会调用 `self.model.embed_input_ids(...)`
4. 然后把得到的 `inputs_embeds_scheduled` 整段 `copy_` 回 `self.inputs_embeds.gpu[:num_scheduled_tokens]`

因此当前第三个 blocker 可以写得很具体：

1. 第一个 blocker：embeds 请求需要同时保留 `prompt_token_ids`，否则 M-RoPE 过不去
2. 第二个 blocker：worker 不能把这些位置重新当成 token ids 再 embedding 一次
3. 第三个 blocker：**在 `supports_mm_inputs` 分支里，外部提供的 prompt embeddings 仍会被整段 multimodal embedding 覆盖**

所以第三个 patch 的目标也就非常明确：

- 不是继续改 pooler
- 而是要在 `_preprocess()` 的 multimodal 分支里，保住外部提供的 prompt embeddings
- 同时继续保留 `prompt_token_ids` 给 M-RoPE / position 使用

在第三个 patch 打通之后，下一步 probe 就不应该继续停留在“测试向量”层面，而应该尽量逼近 HF `generate_latents()` 的真实输入形式。

为此又补了一个更接近目标的脚本：

- [probe_vllm_generate_latents_hidden_states.py](/root/backup/InternNav/scripts/eval/tools/probe_vllm_generate_latents_hidden_states.py)

这条脚本会同时做两件事：

1. 用更接近 HF `generate_latents()` 的方式构造 `prompt_embeds`
   - 从 baseline sample 读取 `pixel_values` 和 `image_grid_thw`
   - 用 vLLM 内部 Qwen2.5-VL model 计算视觉 embedding
   - 替换 prompt 中的 image token 位置
   - 再把末尾 4 个位置替换成真实 checkpoint 里的 `latent_queries`
2. 自动开启 full-tensor dump，并比较
   - `prepare_inputs.inputs_embeds_gpu`
   - `post_forward.hidden_states`
   - `pooler_output`

推荐运行方式：

```bash
cd /root/backup/InternNav
source /root/.venv/bin/activate
python scripts/eval/tools/probe_vllm_generate_latents_hidden_states.py \
  --model-path checkpoints/InternVLA-N1-DualVLN-qwen25vl-s2-view \
  --hf-model-path checkpoints/InternVLA-N1-DualVLN \
  --sample-pt logs/habitat/hf_generate_latents_baseline_replay1/samples/sample_0000_zsNo4HB9uLZ_0001_step_0003.pt \
  --append-traj-tokens \
  --output logs/habitat/vllm_generate_latents_hidden_states_probe_qwen25vl.json
```

这一步真正想回答的问题已经升级为：

1. 真实 `latent_queries` 注入后，raw hidden states 的最后 4 个位置会不会变化
2. 这些 vLLM hidden states，和 HF baseline latent 的差距大概有多大

当前已经基于同一条 replay1 样本做了一轮实测：

- sample: `sample_0000_zsNo4HB9uLZ_0001_step_0003.pt`
- prompt length: `2084`
- image token count: `1955`
- visual embeddings count: `1955`

在普通模式下，已经确认：

1. 真实 `latent_queries` 注入后，`prepare_inputs.inputs_embeds_gpu` 的最后 4 个位置会明显变化
2. `pooler_output` 的最后 4 个位置也会随之变化

进一步在 `--enforce-eager` 模式下，为正式 pooling 请求路径补上 raw hidden-state dump 后，又拿到了更关键的一层：

- `Input-embed last-4 max abs diff = 3.984375`
- `Token-embed last-4 max abs diff = 0.215068`
- `Hidden-state last-4 max abs diff = 24.437500`
- `vLLM custom hidden last-4 vs HF baseline latent max abs diff = 19.500000`

这组数字说明两件事：

1. **第三个 patch 已经真正打通了“真实 latent_queries 注入 -> raw hidden states 改变”这条链路**
2. **但当前 vLLM hidden states 还没有和 HF baseline latent 对齐，数值差距仍然明显**

所以这一步的阶段性结论可以写得很清楚：

> 现在已经不再是“vLLM 能不能承接 custom latent query inputs”的问题，而是“在这条已经打通的输入路径上，如何继续做 hidden-state 数值对齐”的问题。

---

## 10. 当前 gap 更像来自哪里

结合现有源码比对和 probe 结果，当前最值得优先怀疑的不是同一个量纲里的小数值误差，而是输入语义层面的不一致。按优先级大致可以分成下面三类。

### 10.1 最可疑：`EmbedsPrompt` 路线当前没有把 multimodal metadata 传进 M-RoPE

这是当前最像主 gap 来源的一点。

虽然 `EmbedsPrompt` 在 schema 上继承了 `_PromptOptions`，因此**可以声明**：

- `multi_modal_data`
- `mm_processor_kwargs`

但当前实现里：

- [base.py](/root/backup/vllm/vllm/renderers/base.py)
  的 `_process_embeds()` 只返回
  - `prompt_embeds`
  - `prompt_token_ids`
  - `cache_salt`
- 它不会像 `_process_tokens()` 那样进入 `_process_multimodal(...)`
- [input_processor.py](/root/backup/vllm/vllm/v1/engine/input_processor.py)
  也只会在 `decoder_inputs["type"] == "multimodal"` 时构造 `mm_features`

这意味着当前 prompt-embeds probe 的真实状态很可能是：

1. 视觉 embedding 已经被我们手工正确替换到 prompt 中
2. 但 worker 侧构造 M-RoPE 时，`req_state.mm_features` 仍然是空的
3. 因此 `get_mrope_input_positions(...)` 很可能退化成“纯文本 1D 位置”
4. 也就是：
   - **embedding 是多模态的**
   - **position 却还是文本的**

如果这一点成立，它足以单独解释一大块 hidden-state gap。

更重要的是，这里的问题更像是：

- **M-RoPE 输入数据管线不一致**

而不是：

- **M-RoPE 数学公式本身有明显错误**

### 10.2 次可疑：visual replacement 本身看起来更接近 HF，同构程度比之前高

当前 probe 已经有几个正向信号：

1. baseline sample 里的 image token 数是 `1955`
2. vLLM 内部 `embed_multimodal(...)` 产出的 visual embedding 数也是 `1955`
3. 当前脚本是直接拿 baseline sample 里的：
   - `pixel_values`
   - `image_grid_thw`
   去构造 visual replacement
4. 没有再经过 prompt text / fast processor 重新展开一遍

因此和早期 `TokensPrompt + multi_modal_data` 路线不同，当前这条 probe 里：

- `Qwen2VLImageProcessor fast/slow` 的 warning 已经不是最核心问题
- visual replacement 这一步的结构同构性，实际上比之前高很多

这并不等于我们已经证明：

- HF visual tower 输出 tensor 和 vLLM visual tower 输出 tensor 数值 fully allclose

但至少从当前证据看，**visual replacement 本身不像是最大 gap 来源**。

### 10.3 较不可疑：我们抓到的 raw hidden states 定义与 HF `outputs.hidden_states[-1]` 更像是同义的

这一点通过源码比对已经比较清楚：

- HF 侧 [modeling_qwen2_5_vl.py](/root/miniforge3/envs/habitat/lib/python3.9/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)
  在 `Qwen2_5_VLModel.forward()` 里：
  - 先跑所有 decoder layers
  - 然后执行 `hidden_states = self.norm(hidden_states)`
  - `output_hidden_states=True` 时，`all_hidden_states += (hidden_states,)`
- 所以 HF 的 `outputs.hidden_states[-1]` 就是**最终 norm 之后的 last hidden states**

而 vLLM 侧：

- [qwen2.py](/root/backup/vllm/vllm/model_executor/models/qwen2.py)
  的 `Qwen2Model.forward()` 最后也执行：
  - `hidden_states, _ = self.norm(hidden_states, residual)`
  - 然后直接返回 `hidden_states`
- [qwen2_5_vl.py](/root/backup/vllm/vllm/model_executor/models/qwen2_5_vl.py)
  的 `forward()` 也只是把这个 language model 的输出继续向上返回

所以从张量定义上看，我们现在在正式 pooling 请求路径里 dump 到的 raw hidden states，更像是与 HF 的：

- `outputs.last_hidden_state`
- `outputs.hidden_states[-1]`

处在同一个语义层级。

因此这里更合理的判断是：

- **raw hidden states 的“定义”本身大概率不是主 gap**
- 真正的主 gap 更可能发生在：
  - `position_ids / M-RoPE`
  - 或更前面的 multimodal metadata 传递

### 10.4 当前最值得优先做的下一步

如果按“最可能快速缩小 gap”的优先级排，当前最值得先做的是：

1. 让 `EmbedsPrompt` 路线也能保留并传递 `multi_modal_data -> mm_features`
2. 让 worker 侧的 M-RoPE 构造真正看到这些图像项的 offset / grid 信息
3. 再复跑同一条 hidden-state probe

如果这一步之后 `vLLM custom hidden last-4 vs HF baseline latent` 明显收敛，
那就基本可以确认：

> 当前最大的 gap 主要不在 hidden-state 定义，也不在 visual replacement 本身，而在 M-RoPE 所需的 multimodal metadata 没有跟着 `EmbedsPrompt` 一起进入 engine。

---

## 11. 第四个 patch：让 `EmbedsPrompt` 也能携带 `mm_features`

基于上面的判断，我们继续补了第四个 patch，目标不是再改 hidden-state 本身，而是把：

- `EmbedsPrompt`
- `prompt_token_ids`
- multimodal placeholder / grid / hash 信息

一起送进 engine。

这次改动的核心点是：

1. `EmbedsInputs` 允许携带：
   - `mm_kwargs`
   - `mm_hashes`
   - `mm_placeholders`
2. renderer 在处理 `EmbedsPrompt(multi_modal_data=...)` 时，会先走 `_process_multimodal(...)`
3. `InputProcessor` 不再只在 `type == "multimodal"` 时构造 `mm_features`
4. 对 `type == "embeds"` 且带有 `mm_*` 字段的请求，也会生成 `req_state.mm_features`

这意味着现在 worker 侧构造 Qwen2.5-VL 的 M-RoPE 时，终于不再只是“看见 token ids”，而是也能看见：

- 图像 placeholder offset
- 每张图的 `image_grid_thw`
- 每个 multimodal item 的位置元信息

---

## 12. 新实验：补上 mm metadata 后，HF gap 明显收缩

对应实验结果文件：

- [vllm_generate_latents_hidden_states_probe_qwen25vl_eager.json](/root/backup/InternNav/logs/habitat/vllm_generate_latents_hidden_states_probe_qwen25vl_eager.json)
- [vllm_generate_latents_hidden_states_probe_qwen25vl_with_mm_metadata_eager.json](/root/backup/InternNav/logs/habitat/vllm_generate_latents_hidden_states_probe_qwen25vl_with_mm_metadata_eager.json)

两版最关键的对比如下：

| 指标 | 无 mm metadata | 有 mm metadata |
|---|---:|---:|
| token-embed last-4 max abs diff | `0.2151` | `0.3125` |
| raw hidden-state last-4 max abs diff | `24.4375` | `44.4375` |
| vLLM custom hidden last-4 vs HF baseline latent max abs diff | `19.5000` | `7.3438` |
| vLLM custom hidden last-4 vs HF baseline latent mean abs diff | `0.8178` | `0.3997` |

这组数里最重要的不是中间层彼此差多少，而是最后一列：

- **和 HF baseline latent 的 `max_abs_diff` 从 `19.5` 降到了 `7.34`**
- **和 HF baseline latent 的 `mean_abs_diff` 从 `0.818` 降到了 `0.400`**

这说明：

1. `EmbedsPrompt` 缺少 multimodal metadata 确实是一个实质性误差源
2. 补上 `mm_features` 后，vLLM hidden-state 路线和 HF `generate_latents()` 的数值距离明显缩小
3. 之前对 “主 gap 很可能在 M-RoPE / position metadata” 的判断，是被实验支持的

同时也要保持准确：

- 现在还没有 fully align
- `max_abs_diff ≈ 7.34` 仍然不够拿去直接替换本地第二个 DualVLN
- 但这一步已经把问题从“路径不通 / 定义不一致”推进到了“剩余数值偏差如何继续压缩”

---

## 13. 当前最可信的误差来源排序

在补上 `mm metadata` 之后，三个候选误差源的优先级可以重新排序：

1. **M-RoPE / multimodal position metadata**
   - 仍然重要，但已经不是“完全缺失”，而是可能还有细节没完全同构
2. **visual replacement 的细节差异**
   - 现在比之前更值得查，因为主 metadata 缺口已经补上了
3. **raw hidden-state 定义差异**
   - 目前仍然不像主因，因为 HF 和 vLLM 两边都指向 post-final-norm 的最后层状态

所以接下来的工作重点应该是：

1. 继续比对 HF 与 vLLM 的：
   - `position_ids / M-RoPE`
   - `image_grid_thw`
   - visual embedding 替换前后的关键张量
2. 不再把主要精力放在 pooler / hidden-state 定义上

一句话更新当前判断：

> 第四个 patch 证明了 `EmbedsPrompt + prompt_token_ids + mm_features` 这条路是有效的，而且它显著缩小了和 HF `generate_latents()` 的差距；剩下的 gap 更像是 position / visual 细节级对齐问题，而不是“vLLM 根本不适合做这件事”。
## 2026-03-19: Visual Tower Alignment Status

在前面把 `prompt_embeds + prompt_token_ids + mm metadata` 打通之后，剩余 gap 继续被拆到了 visual tower。

本轮新增两层验证：

1. 直接比较 visual tower 最终输出
2. 比较 visual tower 的关键中间量

结论如下：

- `patch_embed` 与 HF 完全一致
- `window_index` 与 `cu_window_seqlens` 与 HF 完全一致
- 但 final visual output 仍有稳定差异：
  - `max_abs_diff = 0.84375`
  - `mean_abs_diff ~= 0.0087`

这说明：

- 视觉 token 的顺序、数量、切片和 replacement 已经不是主要问题
- 剩余差异更像是在 visual transformer block / attention 内部逐层积累出来的

同时，HF `flash_attention_2 / sdpa / eager` 三种 backend 与 vLLM 的差异量级没有明显收敛，所以这不太像“只换一个 HF backend 就能消掉”的问题。

因此当前最合理的工程判断是：

> `generate_latents()` 的 backbone 路线在技术上已经打通；剩余精度 gap 主要卡在 Qwen2.5-VL visual tower 的实现细节，而不是语言侧位置、token replacement 或 latent query 注入。

## 2026-03-20: Visual Blocks Divergence Localization

为了避免继续无限深挖 visual tower 的单个 kernel，本轮只做了一次 block 级 hidden-state 定位，并把它和最终 `generate_latents()` latent gap 放在同一个结论里看。

使用脚本：

- `scripts/eval/tools/export_vllm_visual_block_states.py`
- `scripts/eval/tools/compare_hf_vllm_visual_block_states.py`

与当前端到端 probe 的绑定指标：

- `vLLM custom hidden last-4 vs HF baseline latent max_abs_diff = 0.75`

block 级结果显示：

- `patch_embed` 完全一致
- `pre_blocks` 完全一致
- **block 0 就开始有非零差异**
  - block 0: `max_abs_diff = 0.125`
  - block 5: `max_abs_diff = 0.5`
  - block 10: `max_abs_diff = 0.625`
  - block 15 (full): `max_abs_diff = 11.0`
  - block 31 (full): `max_abs_diff = 1588.0`
- final visual output 仍然是：
  - `max_abs_diff = 0.84375`

这说明：

1. visual tower 的分叉不是在 merger 才突然出现
2. 偏差是从 **第一个 visual block** 就开始，但前几层幅度小
3. 后续更像是在 visual blocks 内逐层放大，尤其是更深层 / full-attention block 处更明显

因此当前最合理的边界判断是：

> 继续往下追，大概率只会把问题进一步定位到 `MMEncoderAttention` / backend 级实现；在没有显著缩小最终 `last-4 latent diff` 之前，这已经足够支持“先停在这里，回到是否能去掉第二份 DualVLN 的主目标”的决策。

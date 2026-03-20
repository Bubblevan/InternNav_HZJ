# DualVLN `generate_latents()` 迁移到 vLLM 的阶段性阻塞分析

这份文档专门总结当前这几轮实验里最关键的结论：

1. `generate_latents()` 在 DualVLN 里到底扮演什么角色
2. 我们为什么想把它迁到 vLLM
3. 当前公开 vLLM 接口为什么还做不到严格等价
4. 后续是否值得改 vLLM 源码并从源码安装

---

## 1. `generate_latents()` 是做什么的

是的，`generate_latents()` 生成出来的东西，本质上是给 **System 1 / DiT** 用的条件 latent。

在当前 DualVLN 里，链路大致是：

1. **System 2**
   - 根据多模态上下文输出文本动作
   - 文本动作可能是：
     - 离散动作
     - 或 pixel goal，比如 `"301 225"`

2. **`generate_latents()`**
   - 当 S2 输出 pixel goal 时，把“完整多模态上下文 + 输出 token 序列”送进 Qwen2.5-VL backbone
   - 取出特定位置的最后层 hidden states
   - 作为后续轨迹生成条件

3. **`generate_traj()` / DiT**
   - 把上一步的 latent 当作条件
   - 做 denoising / flow matching
   - 生成轨迹

所以更准确地说：

> `generate_latents()` 不是最终动作头，而是 **S2 到 S1/DiT 之间的桥接表示提取步骤**。

---

## 2. 当前为什么要盯着 `generate_latents()`

因为当前系统里最浪费显存的地方，不只是 `generate()`，而是：

- vLLM server 里已经有一份 Qwen2.5-VL backbone
- Habitat 本地完整 DualVLN 里又有一份 Qwen2.5-VL backbone

而 `generate_latents()` 恰好还在本地完整 DualVLN 上运行。

所以当前“两个大模型副本”的根源是：

1. vLLM 负责 S2 文本生成
2. 本地完整 DualVLN 负责：
   - `generate_latents()`
   - `generate_traj()`

从优化目标上讲，我们真正想做的是：

> 让 vLLM 不只负责 `generate()`，还负责 `generate_latents()` 里依赖 backbone forward 的那一部分；本地只保留 S1/DiT。

也就是最终收敛到：

- **一个 vLLM server**
- **一个本地轻量 S1/DiT**

---

## 3. `generate_latents()` 的真实原理

这一步非常关键，因为它决定了“迁移难点”到底在哪。

当前 HF 实现并不是：

- 直接拿 `baseline_output_ids` 跑一遍模型
- 然后取最后 4 个 token hidden states

而是更复杂一些：

1. 对 `input_ids / output_ids` 做 text embedding
2. 对图片做 visual embedding
3. 用视觉 embedding 替换 image placeholder 对应位置
4. 在序列末尾追加 4 个 `TRAJ_TOKEN`
5. **但这 4 个位置真正送入 backbone 的并不是普通 token embedding**
6. **而是 `latent_queries` 这个可学习参数，直接替换掉这 4 个位置的输入 embedding**
7. 然后才跑 backbone
8. 取最后层 hidden states 的最后 4 个位置

所以真正的输入不是：

- token ids 就完了

而是：

- 固定的多模态序列
- 固定的视觉 embedding 替换
- 固定的末尾 4 个 custom `inputs_embeds`（`latent_queries`）

一句话总结：

> `generate_latents()` 的核心能力不是“输出 hidden states”，而是“允许我们精确控制 backbone 的输入 embedding，再读取指定位置的 hidden states”。

---

## 4. 为什么 vLLM 的公开接口还做不到严格等价

### 4.1 `token_embed` 路线虽然能跑，但只解决了“输出端”

我们已经验证：

- `runner="pooling" + convert="embed" + pooling_task="token_embed"`
- 对当前多模态 Qwen2.5-VL checkpoint 是 **真的能跑通**
- 它也确实返回逐 token 的二维矩阵

这说明：

- vLLM 不是完全没有 hidden-state 相关路径

但这还只解决了“输出端”，也就是：

- 能不能拿到 token-level outputs

### 4.2 `TokensPrompt + multi_modal_data` 并不会保住你给的 token 序列

更关键的问题出现在“输入端”。

我们做过一轮 probe：

- 直接把 baseline sample 里的 `baseline_output_ids` 作为 `TokensPrompt(prompt_token_ids=...)`
- 同时提供 `multi_modal_data={"image": ...}`
- 希望 vLLM 返回和这些 token 一一对应的 token-level outputs

结果是：

- 请求长度：`2084`
- 返回长度：`4034`

这不是偶然噪声，而是源码逻辑决定的。

本地 vLLM 源码里，[base.py](/root/backup/vllm/vllm/renderers/base.py) 的 `TokensPrompt` 只要带了 `multi_modal_data`，仍然会走 `_process_multimodal(...)`。也就是说：

- vLLM 不会“原封不动使用你传入的 token ids”
- 它会再经过一遍 multimodal processor
- 然后重新展开 placeholder / 图像 patch 对应 token

因此这条路线天然做不到：

- “固定输入 token 序列”

而 `generate_latents()` 恰恰非常依赖“固定输入序列”。

### 4.3 即使 token 序列能固定，`latent_queries` 仍然是第二道门槛

就算假设我们 somehow 解决了上面的问题，后面还有一层更本质的门槛：

- HF `generate_latents()` 最后的 4 个位置，用的是自定义 `latent_queries` embedding
- 不是普通 `TRAJ_TOKEN` 的 embedding lookup

所以就算 vLLM 允许：

- 固定 `baseline_output_ids`
- 追加 4 个 `TRAJ_TOKEN`
- 返回最后 4 个 token 的 outputs

它也不等价于 HF `generate_latents()`，因为还差：

- 对末尾 4 个位置注入 custom `inputs_embeds`

---

## 5. 当前最准确的阶段性结论

现在最准确的说法不是：

- “vLLM 做不到 `generate_latents()`”

而是：

- **vLLM 的公开高层接口还做不到严格等价的 `generate_latents()`**

更具体地说，当前阻塞点有两个：

1. **输入序列不可严格冻结**
   - `TokensPrompt + multi_modal_data` 会重新走 multimodal processor
2. **无法直接注入 `latent_queries` 这种 custom `inputs_embeds`**
   - 而这正是 HF `generate_latents()` 的关键

最近补的 `prompt_embeds` probe 又把这个结论进一步收紧了：

- 即便先不要求“严格冻结多模态 token 序列”
- 单纯想让 vLLM 接受外部给定的 `prompt_embeds`
- 在 Qwen2.5-VL 上也会先撞到 M-RoPE 初始化阶段的硬约束

原因不是模型 `forward(inputs_embeds=...)` 不存在，而是当前公开路径的中间层逻辑：

1. `EmbedsPrompt` 请求进入 engine 后，只保留 `prompt_embeds`
2. `prompt_token_ids` 会被置为 `None`
3. 但 worker 初始化 M-RoPE 时又硬要求 `prompt_token_ids` 必须存在

所以目前公开 `prompt_embeds` 路线的更精确结论是：

- **方向上是对的**
- **当前实现不通**
- **blocker 已经具体到可以通过源码 patch 解决**

在补上“embeds 请求同时保留 `prompt_token_ids`”这个最小 patch 后，新的实验结果又进一步收敛了问题：

- 请求可以成功执行
- 输出长度也可以严格保持为 `2084`
- 但即使显式修改最后 4 个 prompt embeddings，返回的 token-level 输出仍然完全不变

这说明当前问题已经不再是：

- “M-RoPE 过不去”

而变成了：

- “自定义 `prompt_embeds` 在 worker 执行前又被 token-id embedding 覆盖掉了”

因此，当前公开 `prompt_embeds` 路线的完整阶段性判断应该是：

1. **第一个 patch 已经打通 M-RoPE 所需的 token ids 保留**
2. **第二个 patch 还需要阻止这些位置被重新当作 token ids 处理**
3. **只有这样，外部给定的 custom embeddings 才可能真正影响输出**

在继续补 patch 之后，我们又加了一轮文件级 debug dump，并把链路细化到了：

- `gpu_input_batch_add_request`
- `gpu_model_runner_prepare_inputs`
- `gpu_model_runner_pre_forward`
- `gpu_model_runner_post_forward`
- `gpu_model_runner_pooler_output`

这一轮的关键观察是：

1. 在 `add_request` 阶段，两次请求的 `prompt_embeds` 最后 4 行差异是存在的
2. 但到了 `prepare_inputs`，`inputs_embeds_gpu` 已经完全一致
3. 后续 `pooler_output` 也因此继续保持完全一致

这说明当前问题已经进一步收敛为：

- 不是 pooler 抹平了差异
- 也不是 model forward 抹平了差异
- 而是 **在 `_preprocess()` 阶段，自定义 `prompt_embeds` 就已经被覆盖掉了**

进一步对源码分支的定位表明，真正覆盖它们的是 Qwen2.5-VL 的多模态分支：

- 只要 `supports_mm_inputs=True`
- `_preprocess()` 就会优先走 multimodal path
- 然后调用 `self.model.embed_input_ids(...)`
- 并把结果整段 `copy_` 回 `self.inputs_embeds.gpu[:num_scheduled_tokens]`

这会导致一个更具体的“第三个 blocker”：

1. 第一个 patch：让 `EmbedsPrompt` 同时保留 `prompt_token_ids`，供 M-RoPE 使用
2. 第二个 patch：阻止这些位置在通用 prompt-embed 路线上被重新当成 token ids
3. **第三个 patch：在 `supports_mm_inputs` 分支里，继续保住外部给定的 prompt embeddings，不要被整段 multimodal embedding 覆盖**

在第三个 patch 之后，我们又进一步把 probe 升级成了：

- 用真实 replay1 sample 的 `pixel_values / image_grid_thw`
- 用真实 checkpoint 里的 `latent_queries`
- 在正式 pooling 请求路径里直接抓 raw hidden states

这一步带来的阶段性变化非常关键：

1. **“custom latent_queries 是否真的进入 backbone”这个问题已经可以回答为：是**
2. **新的主问题已经从“路径能否打通”转成了“hidden-state 如何继续数值对齐”**

基于当前单样本实测（`sample_0000_zsNo4HB9uLZ_0001_step_0003.pt`）：

- `prepare_inputs` 最后 4 个位置差异：明显非零
- `token_embed` 最后 4 个位置差异：明显非零
- `raw hidden_states` 最后 4 个位置差异：明显非零
- 但 `vLLM custom hidden last-4` 与 HF baseline latent 的 `max_abs_diff` 仍约为 `19.5`

所以最新结论应该更新为：

> 目前已经证明 vLLM 源码改造后可以承接 DualVLN 风格的“真实 latent_queries 注入”，并且这会真实影响 raw hidden states；但它与 HF `generate_latents()` 的数值等价性还没有建立，接下来需要继续做 hidden-state 级别的精度对齐。

在这之后，我们又补了一个更关键的“第四个 patch”：

1. `EmbedsInputs` / `EmbedsPrompt` 允许一路携带：
   - `mm_kwargs`
   - `mm_hashes`
   - `mm_placeholders`
2. `InputProcessor` 对 `type == "embeds"` 且带有这些字段的请求，也会构造 `mm_features`
3. probe 侧用 replay1 manifest 重建真实输入图像，把 multimodal metadata 真正带回 engine

这一步的核心目的，是让 worker 在构造 Qwen2.5-VL 的 M-RoPE / position 时，不再处于“只有 token ids、没有 multimodal item metadata”的状态。

补完以后，同一条 eager hidden-state probe 的结果更新为：

- `vLLM custom hidden last-4 vs HF baseline latent max_abs_diff`
  - 从 `19.5`
  - 降到 `7.34375`
- `mean_abs_diff`
  - 从 `0.8178`
  - 降到 `0.3997`

这说明：

1. `EmbedsPrompt` 缺失 multimodal metadata，确实是之前的一个主要误差源
2. “主 gap 很大一部分来自 M-RoPE / multimodal metadata 不同构” 这个判断，被实验明显支持了
3. 当前问题已经进一步收敛成：
   - 剩余的 M-RoPE / position 细节
   - visual replacement 是否完全同构
   - 而不再是“这条路根本不成立”

所以目前这条结论可以写得很明确：

> 公开 vLLM API 可以提供“接近 hidden-state 提取”的能力，但还不足以原样承接 DualVLN 的 `generate_latents()`。

---

## 6. 那我们的目标到底是不是“让 vLLM 做 `generate_latents()`”？

是，但更精确一点：

我们的目标不是“把 `generate_latents()` 这个 Python 函数名字搬到 vLLM 上”。

真正目标是：

> 让 vLLM 负责 `generate_latents()` 中依赖 Qwen2.5-VL backbone 的那部分 forward / hidden-state 提取能力。

换句话说，我们要迁移的是它的 **backbone computation**，不是执着于某个函数名。

只要最后能做到：

- Habitat 本地不再加载第二份完整 Qwen2.5-VL
- vLLM 能给出与 HF `generate_latents()` 足够接近的 latent

那就算目标达成。

---

## 7. 改 vLLM 源码、从源码安装

### 7.1 为什么可以

我们现在用的本来就是本地 vLLM 源码树：

- `/root/backup/vllm`

而且已经验证过：

- 本地源码里有 pooling / token_embed / multimodal 这些路径

所以工程上完全可以：

1. 改本地 vLLM 源码
2. 在你的 vLLM 环境里从源码安装
3. 跑自定义 probe / 自定义 endpoint

这条路在工程上是可行的。

### 7.2 为什么值得改

因为当前真正缺的是“中间层能力”，不是最终模型权重本身。

如果我们只靠公开 API，就会被卡在：

- token 序列重处理
- 无法精确控制 `inputs_embeds`

而源码改造至少给了我们两个可能方向：

1. **自定义 offline API**
   - 新增一个更底层的接口
   - 允许传入已经构造好的多模态 `inputs_embeds`
   - 直接返回最后层 hidden states

2. **自定义 server endpoint**
   - 在 vLLM server 上加一个内部接口
   - 输入固定样本
   - 返回指定 token 位置 hidden states

### 7.3 但要注意，改源码不是小修小补

从当前原理看，真正要补的能力可能包括：

1. 支持“完全信任外部给定的 prompt_embeds / inputs_embeds”
2. 支持 embeds 请求同时保留 `prompt_token_ids`，供 M-RoPE / position 构造使用
3. 支持对“已由 prompt_embeds 提供的 prompt 位置”跳过 token-id re-embedding
4. 支持 embeds 请求携带完整的 multimodal metadata，构造 `mm_features`
5. 支持跳过或部分绕过默认 multimodal processor
6. 支持返回指定层、指定位置的 hidden states
7. 最好还能保住现有 Qwen2.5-VL 的 position / rope / multimodal 对齐逻辑

所以这不是改一个 flag 就完事，更像是：

- 给 vLLM 增加一条面向 DualVLN 的专用隐藏态提取路径

---

## 8. 当前最推荐的下一步

如果按“最有价值、也最符合现状”的顺序，我建议：

1. **先承认公开 vLLM API 不足以严格复现 `generate_latents()`**
2. **下一阶段转向 vLLM 源码改造 / from-source 安装路线**
3. **优先验证 `prompt_embeds` / `inputs_embeds` 能否打通**
4. **如果能打通，再返回 baseline latent 比较器做离线对齐**

从当前证据看，后续的核心问题已经很收敛了：

> 不是“能不能拿 hidden states”，而是“能不能在 vLLM 中以正确方式注入 DualVLN 所需的 custom input embeddings，并返回指定位置的 hidden states”。

并且现在还能再加一句更具体的：

> 当前 gap 已经被压缩了一大截，接下来优先该查的不是 hidden-state 定义，而是 M-RoPE / position 与 visual replacement 的剩余细节对齐。

---

## 9. 一句话总结

`generate_latents()` 确实是给 System 1 / DiT 提供条件 latent 的；我们也确实希望最终由 vLLM 来承担它背后的 backbone forward。

但当前公开 vLLM 接口还做不到严格等价，不是因为“vLLM 完全没能力”，而是因为：

- 它会重处理多模态 token 序列
- 它还不支持我们真正需要的那种 custom `inputs_embeds` 注入

所以当前最现实的路线已经逐渐收敛为：

> **改 vLLM 源码，并从源码安装，做一条面向 DualVLN `generate_latents()` 的专用 hidden-state / input-embed 路线。**
## 2026-03-19: Visual Gap Narrowing Update

本轮把剩余 gap 继续拆到了 visual tower 内部，新增了几组专门的对齐脚本：

- `scripts/eval/tools/export_vllm_visual_embeddings.py`
- `scripts/eval/tools/compare_hf_vllm_visual_embeddings.py`
- `scripts/eval/tools/export_vllm_visual_stages.py`
- `scripts/eval/tools/compare_hf_vllm_visual_stages.py`

当前最关键的新结论：

1. `generate_latents()` 剩余 gap 的主来源，已经进一步收敛到 visual tower 本身。
2. `patch_embed` 已经和 HF **完全一致**。
3. visual token 的 `window_index` 与 `cu_window_seqlens` 也已经和 HF **完全一致**。
4. 但 final visual output 仍存在稳定差异：
   - `HF vs vLLM visual embeddings max_abs_diff = 0.84375`
   - 平均绝对误差约 `0.0087`
5. 这个差异不是单纯 HF backend 选择造成的：
   - HF `flash_attention_2` / `sdpa` / `eager` 对 vLLM 的差异量级都接近
   - 因此它更像是 vLLM visual blocks / `MMEncoderAttention` 这条实现路径的数值差异，而不是简单的 HF kernel 选择问题

### 已验证不再是主嫌疑的点

- `M-RoPE / language-model positions`
  - 之前已经做到与 HF 完全一致
- `text token embeddings`
  - 已经做到与 HF 完全一致
- `latent_queries` 注入
  - 已经做到与 HF 完全一致
- visual replacement 的 token 对齐顺序
  - image token 数、replacement 顺序、window 索引都已经对上

### 当前更像主嫌疑的点

- `Qwen2.5-VL visual blocks`
- `MMEncoderAttention` / visual attention backend
- visual rotary 在 attention kernel 内部的实际应用方式

### 研究判断

这意味着当前问题已经从“序列/位置/替换错了”进一步收缩成：

> visual tower 前处理已经高度对齐，但 vLLM 与 HF 在视觉 transformer 主体里的数值轨迹仍未完全同构。

换句话说，接下来最值得继续做的，不是再反复调 prompt 或 image placeholder，而是：

1. 对比 visual blocks 的中间 hidden states
2. 重点检查 `MMEncoderAttention` 与 HF visual attention 的差异
3. 判断剩余 `0.84375` 的 visual output gap 是否会继续主导最终 `generate_latents()` latent gap

## 2026-03-20: Visual Blocks First-Divergence Update

本轮按照“不要无限深挖 visual kernel 细节”的原则，只额外做了一轮 **visual blocks 中间 hidden states** 定位，并且把结果显式绑定回：

- `logs/habitat/vllm_generate_latents_hidden_states_probe_qwen25vl_with_mm_metadata_eager.json`
  - `vLLM custom hidden last-4 vs HF baseline latent max_abs_diff = 0.75`

新增脚本：

- `scripts/eval/tools/export_vllm_visual_block_states.py`
- `scripts/eval/tools/compare_hf_vllm_visual_block_states.py`

核心结果：

1. `patch_embed` 仍然 **完全一致**
2. `pre_blocks`（进入 visual blocks 前的重排后 hidden states）也 **完全一致**
3. 但 **第 0 个 visual block 就已经开始出现差异**
   - block 0: `max_abs_diff = 0.125`
   - block 1: `max_abs_diff = 0.125`
   - block 2: `max_abs_diff = 0.25`
4. 差异在前几层是小幅累积，不是“前几层完全对、最后突然炸”
5. 到中后层开始明显放大：
   - block 10: `max_abs_diff = 0.625`
   - block 14: `max_abs_diff = 1.0`
   - block 15（full attention）: `max_abs_diff = 11.0`
   - block 17 之后出现更大数值分叉
6. 最终 visual output 仍是：
   - `max_abs_diff = 0.84375`
7. 对应的最终端到端 latent 指标仍是：
   - `vLLM custom hidden last-4 vs HF baseline latent max_abs_diff = 0.75`

### 这轮能回答什么

这轮已经足够回答用户最关心的那个结构性问题：

> 不是“前几层都对，后面在 merger 才突然放大”，而是 **从第一个 visual block 就开始偏，只是前几层偏得很小，随后在 visual blocks 内逐层积累并放大。**

### 这轮不再继续往下做什么

按当前证据，如果继续往下查，最自然会落到：

- `MMEncoderAttention`
- visual attention backend
- 更细粒度的视觉 block 内部 kernel 数值轨迹

但如果只是把问题进一步定位到这些 backend 级实现，而**没有继续明显缩小最终 `last-4 latent diff`**，这就已经超出了当前轮次“为主目标服务”的范围。

因此本轮建议在这里先停：

> 已经可以合理判断，剩余 gap 不是来自 prompt/token/position/replacement 这些上游拼接问题，而是来自 visual blocks 本体的数值轨迹差异；下一步应回到“在当前 gap 水平下，是否足以推进只保留一个 vLLM server 的主目标”这个更高层决策。

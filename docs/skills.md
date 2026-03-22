# DualVLN Single-vLLM 背景说明（当前阶段）

## 一、项目动机

本项目当前的核心目标非常明确：**让 Habitat 侧不再加载第二份完整 DualVLN，只保留一个 vLLM server 和一本地 S1**。  
也就是说，我们希望把原先 Habitat 进程内承担的完整 S2 / latent 生成职责迁移到服务端，从而减少本地重复加载整套 DualVLN 带来的显存、进程复杂度和工程维护成本。

这里需要特别强调最终目标的边界：  
**最终目标不是“找到一个大致可用的替代表示”，也不是“找到一个相关但不同义的近似 latent”，而是严格等价复现原版 HF `generate_latents()` 语义。**  
任何后续方案都必须以这一原则为准绳：服务端产出的 latent，最终应当能够在语义上替代原版 DualVLN 的 latent，而不是只在统计意义上相近。

## 二、当前系统现状

原版 InternNav / DualVLN 的主链路是纯 HF 语义：

1. `apply_chat_template`
2. `processor(..., images=...)`
3. `model.generate(...)`
4. `generate_latents(...)`
5. `generate_traj(...)`

在当前 fork 中，新增了3条与 vLLM 相关的路径：

- `s2_vllm_url`：文本生成走 vLLM，但仍由本地 HF 路径重建 `output_ids` 并继续执行原版 `generate_latents / generate_traj`
> 其中有一条是文本生成全走本地但是latents由vllm Server输入输出生成
- `dualvln_single_vllm_url`：S2 文本与 latent 都尝试由单个 vLLM server 产生，Habitat 本地只保留 S1

从工程目标看，真正需要打通的是第二条，也就是 **single-vLLM 路径**。

## 三、已经做过的实验

### 1. 闭环原型实验

我们已经跑通过 single-vLLM prototype，并与原版 DualVLN 进行了闭环对比。结果显示当前 prototype 相比原版显著退化，表现为：

- Success、SPL、OS 明显下降
- NE 变大
- 平均步数增加
- 个别 episode 出现明显 drift / loop

这说明当前 single-vLLM 路径虽然“能跑通”，但产出的 latent 还不能等价替代原版 latent。

### 2. token / prompt reconstruction 排查

随后对 single-vLLM 路径做了 1-sample parity 检查，确认过以下内容：

- `full_output_ids`
- `image token` 数量与位置
- `image_grid_thw`
- `position_ids`

其中最早发现过一个 token reconstruction 问题：single-vLLM 使用的 token 序列尾部比 HF reconstruction 多一个 token。该问题修复后，结构侧已经对齐，但 latent 结果几乎没有改善。  
这说明 **token reconstruction 不是主要矛盾**。

### 3. custom forward 路径排查

之后进一步定位 single-vLLM 的 custom latent 提取逻辑，发现原先路径本质上是在 worker 内手工调用：

- `set_forward_context(None, ...)`
- `model.forward(...)`

这条路径没有经过 vLLM 标准 multimodal prefill，也没有正常的 attention metadata / runner context。  
实验表明，这种方式下即便输入 embedding、position、latent query 拼接都看起来合理，forward 后的第一层输出就已经和 HF 路径明显偏离，因此这条路被判断为**不可靠，不应继续作为主线**。

### 4. `extract_hidden_states` 路线

后续改为尝试 vLLM 官方/半官方的 hidden-state 导出路径。  
这条线相较于裸 `apply_model` 明显更接近原版，但最佳层的 cosine 仍只有约 `0.33`，远不足以支持“严格等价复现”这一目标。  
因此可以认定：**`extract_hidden_states` 不是当前最优方向**。

### 5. pooling / `token_embed` 路线

再往后，我们验证了 pooling runner，并确认：

- 本地 vLLM 版本支持 `runner="pooling"`
- 支持 `prompt_embeds + LLM.encode(...)`
- `pooling_task="token_embed"` 能返回 token-wise 张量
- `token_embed` 对应的最后 4 个 token 与 HF `traj_latents` 的相似度显著高于 `extract_hidden_states`

单样本上，`token_embed` 的 tail cosine 可达约 `0.80`；  
在目前有效的 8 个样本上，raw cosine 平均约 `0.74`。  
这说明：**`token_embed` 是当前最接近 HF latent 的官方输出接口**。

## 四、当前发现与判断

到目前为止，已经可以形成以下阶段性结论：

1. **single-vLLM 路径当前尚未实现对原版 latent 的严格等价复现**
2. token、rope、image placeholder、latent query 拼接不是主要问题
3. 裸 `apply_model + model.forward(...)` 路线应停止继续投入
4. `extract_hidden_states` 虽比 custom forward 更合理，但仍与 HF latent 相距较远
5. `token_embed` 是目前最接近 HF latent 的 vLLM 官方可得表示，但它**不是原版 latent 本体**

也就是说，现阶段我们并没有完成“严格等价复现原版 latent”，只是找到了一个**最接近目标但仍不等价**的中间表示。

## 五、当前工作原则

为了避免后续方向跑偏，当前阶段需要坚持以下原则：

- 最终目标始终是：**严格等价复现原版 HF latent**
- 当前所有实验都只是围绕“为什么还没等价”以及“哪条路径最接近等价”展开
- 任何“近似表示”“相关表示”“可校准表示”都只能作为辅助分析结论，**不能被误认为最终方案**
- 在没有证明严格等价前，不应把当前 `token_embed` 路线直接接回 Habitat evaluator 作为正式替代

## 六、当前阶段总结

简而言之，项目目前已经完成了从“能否跑通”到“误差来源定位”的第一轮收敛：

- 我们已经知道哪些路径明显不对
- 也已经知道当前最接近目标的是哪条官方接口
- 但距离“严格等价复现原版 latent”仍有关键差距

因此，当前阶段的意义不是宣布问题已经解决，而是：  
**已经把搜索空间显著缩小，并把后续重点锁定在 vLLM 标准 runner 可得表征与原版 HF latent 之间的语义差距上。**
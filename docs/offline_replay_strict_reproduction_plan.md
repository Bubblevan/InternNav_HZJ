# Offline Replay Strict Reproduction Plan

本文回答一个具体问题：

> 当前离线 replay benchmark 为什么不能严格复现闭环里的上下文状态？如果要更严格地复现，需要怎么做？

## 1. 当前第一版 replay benchmark 的问题

当前 `scripts/eval/tools/benchmark_dualvln_replay.py` 的能力更接近：

- 固定输入上的纯推理 baseline
- 用于比较不同 `num_history / prompt / max_new_tokens` 配置下的 latency 和一致性

它还不等于严格闭环重放，原因主要有四个。

### 1.1 manifest 保存的是每个环境 step，不是每个“新决策点”

闭环里并不是每一步都会重新触发一次 System 2 文本决策。

很多 step 处于同一个 ongoing `pixel_goal -> S1` cycle 中，此时：

- System 2 不会重新生成
- System 1 只是在继续执行之前生成好的局部动作

但当前 replay benchmark 是按“每个 step 都重新喂一次模型”的方式运行，所以天然偏向把很多本来属于 S1 延续的步骤，错误地当成新的 S2 输入。

### 1.2 conversation state 没有被完整保存

在 evaluator 里，尤其是 look-down 分支，存在一个简化的多轮对话状态：

- 上一轮 assistant 的文本输出
- 当前 user 的图像和提示

当前 manifest 里只保存了最终 `llm_output`，没有保存完整的对话历史结构，因此离线脚本无法严格还原当时的上下文。

### 1.3 没有显式区分“新 S2 决策”和“S1 连续执行”

当前 manifest 里虽然保存了：

- `pixel_goal`
- `local_actions_remaining`
- `history_frame_indices`

但还没有显式字段告诉离线脚本：

- 这个 step 是否真的触发了新 S2
- 这个 step 是否只是沿用同一个 pixel-goal cycle
- 当前 step 属于哪个 cycle

### 1.4 离线脚本没有复用 evaluator 当时的精确 prompt

当前离线脚本是按规则重新构造 prompt 和历史图像，而不是直接复用闭环当时真正输入给模型的精确文本和图片列表。

只要 prompt、历史帧或对话状态和闭环时略有不同，输出类型就可能偏到 `discrete`。

## 2. 如果要“严格复现”，最好的方式是什么

最稳妥的做法不是在当前 manifest 上继续猜状态，而是：

> 重新跑一遍 mini 闭环，生成 replay_v2，把当时真正送进模型的决策级上下文一并存下来。

这是推荐路线。

## 3. replay_v2 里应该额外保存什么

建议在 `internnav/habitat_extensions/vln/habitat_vln_evaluator.py` 的 replay 导出逻辑里，新增以下字段。

### 3.1 step 层字段

- `is_new_s2_decision`
  当前 step 是否真正触发了新的一次 System 2

- `decision_id`
  当前 step 对应哪一次高层决策

- `pixel_goal_cycle_id`
  当前 step 是否属于某个持续中的 pixel-goal / S1 cycle

- `cycle_step_offset`
  当前 step 是这个 cycle 的第几步

- `is_lookdown_followup`
  当前 step 是否处于 look-down continuation 分支

### 3.2 S2 输入层字段

- `prompt_text`
  真正送给 processor 的原始文本

- `input_image_paths`
  真正参与该次 S2 的图像列表路径，按顺序保存

- `history_frame_indices`
  参与该次 S2 的历史帧索引

- `conversation_messages`
  若不嫌文件大，建议直接把当时的多轮 messages 结构序列化保存

### 3.3 S2 输出层字段

- `s2_output_text`
- `s2_output_kind`
- `s2_output_pixel`
- `s2_action_seq`
- `s2_generated_token_count`

### 3.4 S1 输入层字段

- `lookdown_rgb_path`
- `lookdown_depth_path`
- `latent_path` 或 `latent_sha`

如果未来只想测 S1，而不想重复生成 latent，那么把 latent 也存盘会很有用。

### 3.5 S1 输出层字段

- `s1_action_seq`
- `s1_action_index_in_cycle`
- `local_actions_remaining`

## 4. 推荐的严格重放流程

### 第一步：重新跑一遍 mini 闭环，生成 replay_v2

建议继续用当前 8-episode mini 或 smoke 配置，只是把 replay 导出升级成决策级版本。

推荐命名：

- `replay_subset_v2`

目标是得到一个足够小、但包含完整上下文状态的固定数据集。

### 第二步：先做 decision-only replay

这里只离线重放真正的 `is_new_s2_decision = true` 的节点。

这样可以先回答：

- 在完全固定的决策点上，S2 文本生成的 latency 和一致性是什么

这一步是最重要的，因为它直接对应 S2 纯推理优化。

### 第三步：再做 full-cycle replay

在 decision-only 通过之后，再做 full-cycle replay：

1. 对于每个新的 S2 决策，先重放 S2
2. 如果 S2 输出 pixel-goal，则继续重放 `generate_latents`
3. 再使用保存好的 look-down 输入重放 S1
4. 按记录下来的 cycle 步序持续消费 S1 动作

这样才能更严格地近似闭环里的大小脑交接过程。

## 5. 为什么建议先做 decision-only 再做 full-cycle

因为 full-cycle 更复杂，涉及：

- S2 决策节点识别
- pixel-goal cycle 边界识别
- 多轮上下文恢复
- look-down continuation 还原

如果一开始就做 full-cycle，调试成本很高。

而 decision-only 的价值已经很大：

- 可以稳定测 `S2 generate`
- 可以做 prompt / history / max_new_tokens / KV cache / backend 的横向对比
- 可以快速发现哪类优化最值得继续推进

## 5.1 是否需要模拟 `http_internvla_client.py / http_internvla_server.py`

可以，而且这是一个很好的“更贴近 realworld”的中间层，但建议把它当作第三层 fidelity，而不是替代原始纯推理 baseline。

推荐把离线 replay 分成三层：

### Level 1: 纯模型 replay

直接在 Python 进程内调用模型。

适合：

- 测最纯粹的 `S2 / S1` 推理 latency
- 做 prompt / history / token 长度 / KV cache / backend 单变量对比

### Level 2: 严格上下文 replay

仍在本地进程内运行，但严格复现：

- decision-only 节点
- full-cycle 状态
- conversation state
- look-down continuation

适合：

- 验证大小脑切换是否被正确复现

### Level 3: HTTP loopback replay

在同机上模拟：

- `http_internvla_client.py`
- `http_internvla_server.py`

即把离线保存的 RGB/depth 按照 realworld client 的方式：

- JPEG / PNG 编码
- `requests.post`
- Flask server 解析
- server 内部 agent.step(...)

全部再走一遍。

适合：

- 估计真实部署中的序列化 / 反序列化 / HTTP 往返成本
- 测 client/server 结构性额外开销

因此最合理的策略不是只选一种，而是三层并存：

- Level 1 保留，作为纯推理 baseline
- Level 2 用于严格复现大小脑上下文
- Level 3 用于估算通信层额外成本

## 6. 当前建议的执行顺序

基于现状，推荐顺序是：

1. 继续使用当前 replay benchmark 做第一轮纯推理横向对比
2. 同时改 evaluator，导出 `replay_v2`
3. 在 `replay_v2` 上实现 decision-only replay benchmark
4. 如果 decision-only 结果稳定，再实现 full-cycle replay benchmark

## 7. 一句话总结

如果你想让离线 replay 真正严格复现闭环上下文，核心不是继续在现有 manifest 上猜，而是：

> 重新跑一遍 mini 闭环，生成带有“决策级状态”的 replay_v2，然后先做 decision-only 重放，再做 full-cycle 重放。

## 8. 3-11 晚更新：已落地的代码状态

前文的设计方案现在已经有第一步代码落地。

### 8.1 已实现部分

当前 evaluator 已经能够在原版 `replay_subset` 之外，同时导出：

- `replay_subset_v2/manifest_rank0.jsonl`

`replay_v2` 当前已实际保存的关键信息包括：

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
- `http_loopback`

这意味着：

- “manifest 没有显式区分新 S2 决策和 S1 连续执行”这个问题，已经在 `replay_v2` 里有了第一版修复
- “conversation state 没有保存”这个问题，也已经通过 `conversation_messages` 和 prompt/chat 文本得到缓解

### 8.2 已实现的 Level 3：HTTP loopback

当前还新增了一条可执行的 in-process loopback benchmark：

- `scripts/eval/tools/benchmark_dualvln_http_loopback.py`

它会在本地进程内模拟：

- client 端 JPEG / PNG 编码
- server 端解码
- `InternVLAN1AsyncAgent.step(...)`
- `LOOKDOWN -> look_down=True` 的二次调用

如果 `replay_v2` 中存在保存下来的 lookdown 图像和深度，它可以直接复用这些观测。

### 8.3 仍未实现部分

虽然 `replay_v2` 已经落地，但以下两部分仍未真正完成：

1. `decision-only replay benchmark`
2. `full-cycle replay benchmark`

当前的 `benchmark_dualvln_replay.py` 仍然是 Level 1 纯推理 baseline，不会自动根据 `is_new_s2_decision` 过滤到只重放真实决策节点。

### 8.4 当前推荐执行顺序

截至 2026-03-11，最合理的使用顺序变为：

1. 重新跑 mini 或 smoke 闭环，生成新的 `replay_subset_v2`
2. 跑原版 `benchmark_dualvln_replay.py`，得到纯推理 baseline
3. 跑 `benchmark_dualvln_http_loopback.py`，得到编码/解码/agent 封装开销
4. 在 `replay_v2` 基础上继续实现 `decision-only replay`

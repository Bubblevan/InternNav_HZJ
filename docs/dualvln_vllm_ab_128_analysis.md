# DualVLN vLLM A/B 闭环分析（128 Episodes）

本文记录当前 DualVLN 在 Habitat 闭环中的 vLLM 接入方式、128-episode A/B 结果、结果解读、以及下一步优化计划。

---

## 1. 当前 vLLM 是怎么接入的

当前方案是 **S2-only vLLM**，不是把整套 DualVLN 都迁到 vLLM。

实际执行链路如下：

1. `s2_step()` 先构造多模态 prompt（instruction + history images + current image）
2. 如果配置了 `s2_vllm_url`，则 **S2 文本生成** 通过 vLLM 的 `/v1/chat/completions` 执行
3. vLLM 返回文本后，在本地重新 `tokenizer.encode()`，拼回 `output_ids`
4. 如果输出是离散动作，则直接走后续 Habitat action
5. 如果输出是 pixel goal，则仍在本地执行：
   - `generate_latents(output_ids, pixel_values, image_grid_thw)`
   - `generate_traj(traj_latents, images_dp, depths_dp)`

因此当前并不是“一个纯 vLLM 版 DualVLN”，而是：

- **vLLM 负责**：S2 的 autoregressive 文本生成
- **本地完整 DualVLN 仍负责**：
  - `generate_latents()`
  - `generate_traj()`
  - System 1 / DiT 轨迹生成

也就是说，当前闭环依然存在“两份大模型链路并存”的问题：

- vLLM server 中有一份 Qwen2.5-VL backbone
- Habitat 本地评估进程里还有一份完整 DualVLN

这也是后续要继续消除的主要显存和工程负担来源。

相关代码位置：

- [internvla_n1_policy.py](/root/backup/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1_policy.py)
- [habitat_vln_evaluator.py](/root/backup/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py)
- [internvla_n1.py](/root/backup/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1.py)

---

## 2. 本次 A/B 实验设置

本次对比采用两套配置：

- HF baseline: `scripts/eval/configs/habitat_ab_hf_cfg.py`
- vLLM S2: `scripts/eval/configs/habitat_ab_vllm_cfg.py`

vLLM 服务启动方式：

```bash
cd /root/backup/InternNav
source /root/.venv/bin/activate
vllm serve checkpoints/InternVLA-N1-DualVLN-qwen25vl-s2-view \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.45 \
  --limit-mm-per-prompt '{"image":16,"video":0}' \
  --port 8001
```

Habitat 评估方式：

```bash
cd /root/backup/InternNav
conda activate habitat
TOKENIZERS_PARALLELISM=false python scripts/eval/eval.py \
  --config scripts/eval/configs/habitat_ab_hf_cfg.py
```

```bash
cd /root/backup/InternNav
conda activate habitat
TOKENIZERS_PARALLELISM=false python scripts/eval/eval.py \
  --config scripts/eval/configs/habitat_ab_vllm_cfg.py
```

输出目录：

- HF: `logs/habitat/ab_test_hf`
- vLLM: `logs/habitat/ab_test_vllm`

---

## 3. 数据整理

### 3.1 HF 目录存在旧结果残留

- `ab_test_hf/result.json` 中 `length = 136`
- `ab_test_hf/progress.json` 实际也是 136 行
- `ab_test_vllm/result.json` 中 `length = 128`
- `ab_test_vllm/progress.json` 实际是 128 行

HF 目录比 vLLM 多出的 8 条 episode 是：

- `804, 805, 806, 807, 808, 809, 810, 832`

因此，**HF 原始总表不能直接与 vLLM 原始总表比较**。原因不是模型行为差异，而是 HF 输出目录中混入了上一次 run 的残留结果。

### 3.2 两边仍然有共同的 128 条 episode

虽然 HF 目录被污染，但两边的交集仍然是 **完全相同的 128 个 episode**。

因此本次文档中的主结果统一采用：

> **按 HF 与 vLLM 共同的 128 个 episode 重算后再比较**

这个口径是本次分析中最可靠的一组闭环结果。

---

## 4. 128 个共同 Episodes 的 A/B 结果

按共同 128 个 episode 重算后：

| 指标 | HF | vLLM | 差异 |
|------|----:|-----:|-----:|
| **SR** | 0.6641 | 0.6719 | +0.0078 |
| **SPL** | 0.5891 | 0.6139 | +0.0248 |
| **NE** | 3.5399 | 3.8268 | +0.2869 |
| **平均步数** | 84.62 | 85.56 | +0.95 |

补充统计：

- success 翻转：vLLM 相比 HF
  - 变好：10 个 episode
  - 变差：9 个 episode
  - 不变：109 个 episode
- steps 差异均值：`+0.95`
- `NE` 差异均值：`+0.2869`

从结果本身看：

1. `SR` 没有下降，反而略高
2. `SPL` 略高
3. `NE` 略差
4. `平均步数` 基本接近

因此，这组结果**不支持“vLLM 接入后精度明显崩坏”**这一判断。

更准确地说：

- 目前没有看到灾难性退化
- 结果呈现的是“整体接近、局部有波动”的状态
- 更像是一次粗粒度 sanity check 通过，而不是严格精度对齐已经完成

---

## 5. 结果怎么理解

如果 vLLM S2-only 接入真的让精度明显失真，典型现象通常会是：

- `SR` 明显下降
- `SPL` 明显下降
- `NE` 明显恶化
- 成功/失败翻转集中偏向单边

但本次并没有出现这种模式。

相反，当前模式更像：

- 部分 episode 更好
- 部分 episode 更差
- 大多数 episode 不变

这说明至少在当前闭环中，vLLM 替代 HF `generate()` 后，系统功能仍基本成立。

## 6. 当前实验中不严格的地方

### 6.1 输出目录未完全清空

HF 目录残留了旧结果，导致直接查看 `result.json` 会得到错误结论。

这里的教训很直接：

- 以后所有 A/B run 前都必须清空输出目录
- 或者改 evaluator 为覆盖写而不是追加写

### 6.2 配置文件里写了 `allowed_episode_ids`，但这次并未真正约束到 Habitat 数据集

当前配置中：

- [habitat_ab_hf_cfg.py](/root/backup/InternNav/scripts/eval/configs/habitat_ab_hf_cfg.py)
- [habitat_ab_vllm_cfg.py](/root/backup/InternNav/scripts/eval/configs/habitat_ab_vllm_cfg.py)

都在 `eval_settings` 里写了：

```python
"allowed_episode_ids": [10, 11, 12, 16, 17, 18, 43, 44]
```

但 `HabitatEnv.generate_episodes()` 实际读取的是：

- `self.config.habitat.dataset.allowed_episode_ids`

而不是 `eval_settings.allowed_episode_ids`。

因此这次实验**实际跑的是共同的 128 个 episode**，不是“8 个固定 episode”。

这并不影响“共同 128 episode”这组结果的有效性，但会影响对实验范围的描述。后续文档和脚本都应按真实口径修正。

### 6.3 闭环中仍有 System 1 / DiT 随机性

这次 A/B 不只是比 S2 文本输出，还经过了完整闭环。

而 pixel-goal 路径下会继续经过：

- 本地 `generate_latents()`
- 本地 `generate_traj()`

其中 `generate_traj()` 内部存在随机噪声采样。因此即使 S2 文本完全一致，单次闭环轨迹也未必严格 deterministic。

这意味着：

- 闭环 A/B 很适合做“功能可用性”和“粗精度稳定性”验证
- 但不够支撑“严格数值等价”结论

### 6.4 还没有把 vLLM 的性能收益拆成细粒度指标

目前这次分析主要关注任务指标：

- SR
- SPL
- NE
- steps

但 vLLM 的优势本来主要是推理性能，所以后续还必须补齐更细粒度的性能统计，例如：

- prefill latency
- decode latency
- end-to-end latency
- TTFT
- output tokens/s
- requests/s

否则只能说明“没明显掉点”，还不能完整说明“值不值得接入”。

---

## 7. 当前阶段结论

当前最稳妥的结论是：

1. **vLLM S2-only 集成已经能在闭环中跑通**
2. **在共同 128 个 episode 上，没有看到“精度明显 G 了”的证据**
3. **但这次实验还不属于严格精度对齐**
4. **当前最大的结构性问题仍是：vLLM 与本地 DualVLN 同时存在，`generate_latents()` 与 `generate_traj()` 还没有被进一步拆分优化**

---

## 8. 下一步计划

接下来建议把工作分成三条并行主线。

### 8.1 主线 A：消灭“双份模型”问题

这是当前最关键的结构优化方向。

原作者给出的思路是：

> `generate_latent` 其实就是取的大模型最后的 hidden state。把 DualVLN 里的标准 Qwen2.5-VL 主干拆出来，直接调用 vLLM 来跑，并取 hidden state，应该就等价于 `generate_latent`。

这个思路不一定最终完全成立，但非常值得作为第一优先级验证。

建议的执行顺序：

1. 明确 `generate_latents()` 的数学定义：
   - prompt + generated output
   - 末尾追加 4 个 `TRAJ_TOKEN`
   - 用 `latent_queries` 替换 embedding
   - 取最后一层 hidden states 的最后 4 个位置
2. 探索 vLLM 侧是否能获得等价 hidden states
3. 先做离线脚本验证，不直接改闭环
4. 只有在 latent tensor 足够接近时，再考虑在 Habitat 中替换

目标是把当前：

- vLLM 一份 Qwen2.5-VL
- 本地再一份完整 DualVLN

变成：

- vLLM 负责 Qwen2.5-VL backbone
- 本地只保留 `latent_queries + S1 / DiT`

如果这条路打通，就能真正解决当前“双份模型”的显存浪费。

### 8.2 主线 B：`generate_traj()` / S1 DiT 独立优化

`generate_traj()` 不属于 vLLM 的甜蜜点，它本质是 DiT / flow matching denoising loop。

这条线建议单独做，不要硬塞到 vLLM 路线里。

可优先尝试的方向：

1. `torch.compile`
2. CUDA graph
3. 减少 `num_inference_steps`
4. 减少 `num_sample_trajs`
5. 探索 S1 / DiT caching 或其他跨步复用策略

其中最先值得做的是低成本实验：

- `num_inference_steps: 10 -> 5 -> 3`
- `num_sample_trajs: 32 -> 16 -> 8`

如果任务指标波动小，就能直接获得线性加速。

### 8.3 主线 C：把 A/B 脚本和评测协议做严格

后续所有 HF vs vLLM 对齐实验，都应升级为“严格可复现”的协议。

至少要做到：

1. episode 集合固定
2. 输出目录强制清空
3. torch / numpy / random seed 固定
4. 显式区分：
   - 接口级一致性
   - latent 数值级一致性
   - 闭环任务级一致性
5. 统一收集性能指标：
   - prefill latency
   - decode latency
   - end-to-end latency
   - TTFT
   - output tokens/s
   - requests/s

理想的实验矩阵应至少包含三层：

1. **S2 文本级**
   - HF vs vLLM 文本 exact match
   - token ids match
2. **latent 数值级**
   - `generate_latents()` 输出 tensor allclose
3. **闭环任务级**
   - SR / SPL / NE / steps
   - latency / throughput

---

## 9. 建议的近期里程碑

### Milestone 1

修正 A/B 配置与脚本，使实验真正满足：

- 固定 episode
- 固定 seed
- 固定输出目录
- 自动汇总任务指标与性能指标

### Milestone 2

补齐 `generate_latents()` 的离线等价性验证脚本：

- 本地 HF `generate_latents()`
- 候选 vLLM hidden states 路径
- 做 tensor 级对比

### Milestone 3

对 `generate_traj()` 做独立 profiling 与减步数实验，判断 S1 优化空间。

---

## 10. 最终一句话总结

当前 vLLM 方案已经证明：**只替换 S2 文本生成并不会在共同 128 个闭环 episode 上表现出明显精度崩坏**。但这仍只是阶段性验证，真正的下一步重点不是“继续堆更多粗对比”，而是：

- 想办法把 `generate_latents()` 从本地完整 DualVLN 中拆出来
- 避免 vLLM 与本地 DualVLN 双份并存
- 同时把 S1 / DiT 和 A/B 对齐协议一起做严谨


# DualVLN HF vs vLLM A/B 使用说明

本文档整理当前仓库中 HF baseline 与 vLLM S2-only 的 A/B 实验使用方式，包括：

- 如何启动 vLLM server
- 如何运行 HF / vLLM 闭环评估
- 如何避免旧日志污染
- 如何汇总共同 episode 的结果
- 当前这套 A/B 实验能说明什么、不能说明什么

配套分析文档：

- [dualvln_vllm_ab_128_analysis.md](/root/backup/InternNav/docs/dualvln_vllm_ab_128_analysis.md)

---

## 1. 当前 A/B 实验实际在比较什么

当前 vLLM 接入方式是 **S2-only**：

- `generate()` 的文本生成走 vLLM
- `generate_latents()` 仍走本地完整 DualVLN
- `generate_traj()` 仍走本地完整 DualVLN

因此这不是“纯 vLLM 版 DualVLN”，而是：

- **HF baseline**
  - 本地完整 DualVLN 执行 `generate() + generate_latents() + generate_traj()`
- **vLLM A/B 版**
  - vLLM 执行 `generate()`
  - 本地完整 DualVLN 执行 `generate_latents() + generate_traj()`

所以这套 A/B 的主要目的，是回答：

> 在不改 `generate_latents()` 与 `generate_traj()` 的前提下，仅把 S2 文本生成替换为 vLLM，闭环行为会不会明显劣化？

---

## 2. 相关文件

配置文件：

- HF: [habitat_ab_hf_cfg.py](/root/backup/InternNav/scripts/eval/configs/habitat_ab_hf_cfg.py)
- vLLM: [habitat_ab_vllm_cfg.py](/root/backup/InternNav/scripts/eval/configs/habitat_ab_vllm_cfg.py)

汇总脚本：

- [analyze_dualvln_ab_results.py](/root/backup/InternNav/scripts/eval/tools/analyze_dualvln_ab_results.py)

核心实现：

- [habitat_vln_evaluator.py](/root/backup/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py)

---

## 3. 运行前的准备

### 3.1 确认 patched vLLM checkpoint 已存在

默认使用：

- `checkpoints/InternVLA-N1-DualVLN-qwen25vl-s2-view`

如果还没有这个目录，需要先按已有文档生成 patched S2-only 视图。

### 3.2 强烈建议先清空输出目录

当前 evaluator 的 `progress.json` 和 `result.json` 是**追加写**，不是覆盖写。

因此每次正式 A/B 前，建议都先清空：

```bash
cd /root/backup/InternNav
rm -rf logs/habitat/ab_test_hf
rm -rf logs/habitat/ab_test_vllm
```

如果不清空，后续分析时可能混入旧 run 的残留结果。

---

## 4. 启动 vLLM Server

在 vLLM 环境中执行：

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

建议检查服务是否起来：

```bash
curl -s http://127.0.0.1:8001/v1/models
```

---

## 5. 运行 HF baseline

在 Habitat 环境中执行：

```bash
cd /root/backup/InternNav
conda activate habitat
TOKENIZERS_PARALLELISM=false python scripts/eval/eval.py \
  --config scripts/eval/configs/habitat_ab_hf_cfg.py
```

输出目录：

- `logs/habitat/ab_test_hf`

---

## 6. 运行 vLLM S2-only A/B

确保上一节的 vLLM server 仍在运行，然后在 Habitat 环境中执行：

```bash
cd /root/backup/InternNav
conda activate habitat
TOKENIZERS_PARALLELISM=false python scripts/eval/eval.py \
  --config scripts/eval/configs/habitat_ab_vllm_cfg.py
```

输出目录：

- `logs/habitat/ab_test_vllm`

---

## 7. 当前配置里 fixed episode 是怎么生效的

当前配置中写的：

```python
"allowed_episode_ids": [10, 11, 12, 16, 17, 18, 43, 44]
```

现在已经通过 evaluator 显式写回 Habitat dataset config，因此后续重跑时这类约束会真正作用到 episode 选择。

也就是说，后续：

- `allowed_scene_ids`
- `allowed_episode_ids`
- `dataset_path_override`
- `dataset_split_override`
- `scenes_dir_override`
- `max_eval_episodes`

这些字段都不再只是“写在 config 里”，而是会真的影响 Habitat 取样。

---

## 8. 如何汇总 A/B 结果

运行汇总脚本：

```bash
cd /root/backup/InternNav
python scripts/eval/tools/analyze_dualvln_ab_results.py \
  --hf-dir logs/habitat/ab_test_hf \
  --vllm-dir logs/habitat/ab_test_vllm \
  --output logs/habitat/ab_test_compare.json
```

它会自动：

1. 读取两边 `progress.json`
2. 检查 episode 覆盖是否一致
3. 找出共同 episode 集合
4. 在共同 episode 上重算：
   - `SR`
   - `SPL`
   - `NE`
   - `平均步数`
5. 统计 success flip
6. 如果目录里存在 `runtime_summary_rank0.json` 或 `runtime_rank0.jsonl`，也会一起汇总 runtime 信息

典型输出示例：

```text
Coverage: hf_rows=136 vllm_rows=128 common=128
Coverage warning: episode sets differ between HF and vLLM outputs.
Common episodes: SR 0.6641 -> 0.6719 SPL 0.5891 -> 0.6139 NE 3.5399 -> 3.8268 Steps 84.62 -> 85.56
Pairwise delta: dSR=0.0078 dSPL=0.0248 dNE=0.2869 dSteps=0.95
Success flips: vllm_better=10 hf_better=9 unchanged=109
```

---

## 9. 如何解读汇总结果

### 9.1 如果 `Coverage warning` 出现

说明两边输出目录中的 episode 集合不一致。

最常见原因：

- 某一侧输出目录没有清空
- 某一侧配置改过
- 中途断跑后继续追加

这时不要直接比较原始 `result.json`，应以：

> **共同 episode 重算结果**

为准。

### 9.2 如果共同 episode 指标接近

说明当前 `vLLM -> S2 generate` 替换至少没有造成明显闭环崩坏。

但这还**不等于严格精度对齐**，因为：

- 闭环里仍包含本地 `generate_latents()`
- 闭环里仍包含本地 `generate_traj()`
- pixel-goal 路径存在 S1 / DiT 采样随机性

所以当前 A/B 的定位应当是：

- **系统级 sanity check**
- **粗粒度任务指标验证**

而不是数值级严格认证。

---

## 10. 当前这套 A/B 实验的边界

这套 A/B 实验适合回答：

- vLLM 替换 S2 文本生成后，闭环是否还能正常工作
- 是否出现明显任务指标崩坏

这套 A/B 实验暂时不能回答：

- `generate_latents()` 是否与未来的 vLLM hidden-state 路径数值等价
- `generate_traj()` 是否是主要误差来源
- vLLM 的 prefill / decode / TTFT / throughput 具体收益有多大

---

## 11. 推荐使用顺序

每次做新的 HF/vLLM A/B，建议按这个顺序：

1. 清空输出目录
2. 启动 vLLM server
3. 跑 HF baseline
4. 跑 vLLM S2-only
5. 用 `analyze_dualvln_ab_results.py` 做共同 episode 汇总
6. 再写实验结论，不要直接抄原始 `result.json`

---

## 12. 当前之后应该往哪里推进

当前 A/B 已经足够支撑下一步进入更关键的问题：

> 能不能把 `generate_latents()` 也从本地完整 DualVLN 中拆出来，最终做到只保留一个 vLLM server？

对应的下一份路线文档见：

- [dualvln_generate_latents_vllm_roadmap.md](/root/backup/InternNav/docs/dualvln_generate_latents_vllm_roadmap.md)


# DualVLN vLLM Hidden-Latents A/B Usage

本文档记录当前阶段的决策：

> **先冻结现有 vLLM hidden-state / visual 对齐 patch，不再继续深挖 visual kernel；转而优先做 end-to-end A/B。**

当前目标是回答：

1. `last-4 latent` 偏差在实际替换路径里是否还能接受
2. `generate_traj()` 输出是否发生不可接受退化
3. mini split 任务指标是否出现明显退化

只有当这些 end-to-end 指标仍明显不可接受时，才回到 visual backend 继续深查。

---

## 1. 当前可切换实现

已经接入一个可切换的 `generate_latents()` 替代后端：

- 默认：`hf`
- 备选：`vllm_hidden`

对应代码：

- `internnav/model/utils/vllm_hidden_latents.py`
- `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py`

在 `InternVLAN1Net` 中，当前逻辑变为：

- `generate_latents_backend == "hf"`  
  继续调用本地 `self.model.generate_latents(...)`

- `generate_latents_backend == "vllm_hidden"`  
  当前支持两种运行方式：
  - **同进程 local runner**
  - **跨环境 HTTP sidecar**

  推荐当前优先使用 **HTTP sidecar**：
  - Habitat / Python 3.9 只做客户端
  - uv vLLM / Python 3.12 起本地 hidden-latents 服务

  这样可以避免：
  - habitat 环境不能直接 `import vllm`
  - vLLM 只支持 Python >= 3.10

---

## 2. 新增配置项

以下 `model_settings` 现在可用：

| key | 含义 |
|---|---|
| `generate_latents_backend` | `"hf"` 或 `"vllm_hidden"` |
| `generate_latents_vllm_model_path` | patched Qwen2.5-VL 视图目录，例如 `checkpoints/InternVLA-N1-DualVLN-qwen25vl-s2-view` |
| `generate_latents_vllm_url` | 可选。本地 HTTP hidden-latents 服务地址，例如 `http://127.0.0.1:8011` |
| `generate_latents_vllm_dump_dir` | vLLM hidden-state dump 输出目录 |
| `generate_latents_vllm_max_model_len` | vLLM local runner `max_model_len` |
| `generate_latents_vllm_gpu_memory_utilization` | vLLM local runner 显存占用比例 |
| `generate_latents_vllm_limit_mm_per_prompt_image` | `limit_mm_per_prompt["image"]` |
| `generate_latents_vllm_dtype` | 传给 local vLLM runner 的 dtype |
| `generate_latents_vllm_enforce_eager` | 是否强制 eager，当前建议 `True` |

---

## 3. Mini Split 任务指标入口

新增 mini split config：

- `scripts/eval/configs/habitat_dual_system_mini_vllm_hidden_latents_cfg.py`
- `scripts/eval/configs/habitat_dual_system_mini_vllm_hidden_latents_http_cfg.py`

它基于：

- `scripts/eval/configs/habitat_dual_system_mini_cfg.py`

其中更推荐当前使用的是：

- `habitat_dual_system_mini_vllm_hidden_latents_http_cfg.py`

因为它不要求 `habitat` 环境直接 import `vllm`。

### HF baseline

```bash
cd /root/backup/InternNav
conda activate habitat
python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_mini_cfg.py
```

### vLLM hidden-latents（推荐：跨环境 HTTP）

先在 **uv/.venv 的 Python 3.12 环境** 起本地服务：

```bash
cd /root/backup/InternNav
source /root/.venv/bin/activate
python scripts/eval/tools/serve_vllm_hidden_latents_http.py \
  --model-path checkpoints/InternVLA-N1-DualVLN-qwen25vl-s2-view \
  --port 8011 \
  --enforce-eager
```

然后在 **habitat / Python 3.9 环境** 跑 mini split：

```bash
cd /root/backup/InternNav
conda activate habitat
python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_mini_vllm_hidden_latents_http_cfg.py
```

### vLLM hidden-latents（同进程 local runner）

如果你未来真的有一个能直接 import patched `vllm` 的环境，也仍然可以使用：

- `scripts/eval/configs/habitat_dual_system_mini_vllm_hidden_latents_cfg.py`

---

## 4. 小规模 Replay A/B

当前最推荐先跑的是 replay benchmark，因为它可以更快回答：

1. `last-4 latent` 与 HF 差多少
2. 同 seed 下 `generate_traj()` 输出差多少
3. 对 replay 基线动作预测是否明显退化

对应脚本：

- `scripts/eval/tools/benchmark_dualvln_replay.py`

新增关键参数：

| 参数 | 含义 |
|---|---|
| `--latent-backend {hf,vllm_hidden}` | 切换 `generate_latents()` 后端 |
| `--latent-vllm-model-path` | patched vLLM view 路径 |
| `--latent-vllm-url` | HTTP hidden-latents 服务地址；提供后优先走跨环境 sidecar |
| `--latent-vllm-dump-dir` | dump 目录 |
| `--compare-hf-reference` | 在 `vllm_hidden` 模式下，同时保留 HF latent / traj 参考 |
| `--traj-seed` | 对比 `generate_traj()` 时复用的随机种子 |

### HF reference

```bash
cd /root/backup/InternNav
conda activate habitat
python scripts/eval/tools/benchmark_dualvln_replay.py \
  --manifest logs/habitat/test_dual_system_mini/replay_subset/manifest_rank0.jsonl \
  --model-path checkpoints/InternVLA-N1-DualVLN \
  --device cuda:0 \
  --max-steps 32 \
  --latent-backend hf \
  --output logs/habitat/replay_ab_hf_latents.json
```

### vLLM hidden-latents A/B（推荐：跨环境 HTTP）

先在 uv/.venv 环境起服务：

```bash
cd /root/backup/InternNav
source /root/.venv/bin/activate
python scripts/eval/tools/serve_vllm_hidden_latents_http.py \
  --model-path checkpoints/InternVLA-N1-DualVLN-qwen25vl-s2-view \
  --port 8011 \
  --enforce-eager
```

然后在 habitat 环境跑 replay A/B：

```bash
cd /root/backup/InternNav
conda activate habitat
python scripts/eval/tools/benchmark_dualvln_replay.py \
  --manifest logs/habitat/test_dual_system_mini/replay_subset/manifest_rank0.jsonl \
  --model-path checkpoints/InternVLA-N1-DualVLN \
  --device cuda:0 \
  --max-steps 32 \
  --latent-backend vllm_hidden \
  --latent-vllm-url http://127.0.0.1:8011 \
  --compare-hf-reference \
  --output logs/habitat/replay_ab_vllm_hidden_latents.json
```

---

## 5. 当前最该看的指标

### 5.1 Latent

看 `summary["latent_compare"]`：

- `last4_max_abs_diff_mean`
- `last4_max_abs_diff_max`
- `last4_mean_abs_diff_mean`

### 5.2 Traj

看 `summary["traj_compare"]`：

- `traj_max_abs_diff_mean`
- `traj_mean_abs_diff_mean`
- `traj_first_action_match_rate`

### 5.3 Replay 一致性

看 `summary["consistency"]`：

- `action_match_rate_all`
- `action_match_rate_pixel_goal`
- `output_kind_match_rate`
- `text_exact_match_rate`

### 5.4 Mini split 任务指标

仍按闭环常规指标看：

- `SR`
- `SPL`
- `NE`
- `平均步数`

---

## 6. 决策边界

当前阶段的原则是：

1. 如果 replay A/B 和 mini split 指标都没有显示明显不可接受退化  
   就优先继续推进“去掉第二份 DualVLN”的集成路线

2. 只有当：
   - `last-4 latent` 差异过大
   - `generate_traj()` 输出明显漂移
   - mini split 任务指标明显恶化

   才回到 visual backend / `MMEncoderAttention` 继续细查

换句话说：

> 现在的优先级已经不再是“继续把 visual block 的局部 allclose 压到更漂亮”，而是“先看当前 patch 水平是否已经足以支撑 end-to-end 替换决策”。

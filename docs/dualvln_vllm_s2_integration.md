# DualVLN vLLM S2-Only 集成说明

本文档总结将 vLLM 作为 System 2（Qwen2.5-VL）推理后端的完整流程，包括 patched checkpoint 创建、vLLM 服务启动、以及 Habitat 评估中 S2 路径的替换方式。

**核心思路**：只把 S2 的文本生成交给 vLLM 加速，S1（DiT/NavDP）和 `generate_latents` 仍保留在本地 PyTorch 路径。

---

## 1. 背景与动机

- **DualVLN** 由 System 1（S1）和 System 2（S2）组成：S2 为 Qwen2.5-VL 视觉语言模型，负责输出 waypoint / STOP / 离散动作；S1 为 DiT 等，负责轨迹执行。
- 闭环评估中 **S2 单次生成时延** 明显高于 S1，是主要瓶颈。
- **vLLM** 提供 PagedAttention、CUDA graph、连续 batching 等优化，适合作为 S2 的推理后端。
- **不能** 把整套 DualVLN 直接丢给 vLLM：checkpoint 的 `model_type=internvla_n1`、且包含 S1 专用权重（latent_queries、memory_encoder、cond_projector 等），vLLM 只认标准 Qwen2.5-VL。因此采用 **S2-only** 方案：用「patched 标准 Qwen2.5-VL 视图」给 vLLM 加载，S1 逻辑留在原代码中。

---

## 2. Patched S2-Only Checkpoint 视图

### 2.1 作用

原始 checkpoint 路径：`checkpoints/InternVLA-N1-DualVLN`  
- `config.json` 中 `model_type=internvla_n1`，`architectures=["InternVLAN1ForCausalLM"]`
- 权重中包含 S2（Qwen2.5-VL 主干）与 S1 相关张量

vLLM 需要以 **标准 Qwen2.5-VL** 形式加载才能正确识别，因此使用工具生成一个「仅改 config、权重用 symlink」的目录，供 vLLM 使用。

### 2.2 生成 Patched 视图

使用项目内脚本检查可行性并生成 patched 目录：

```bash
cd /root/backup/InternNav
python scripts/eval/tools/check_dualvln_vllm_feasibility.py \
  --model-path checkpoints/InternVLA-N1-DualVLN \
  --output logs/vllm_feasibility.json \
  --patched-model-path checkpoints/InternVLA-N1-DualVLN-qwen25vl-s2-view
```

- **`--model-path`**：原始 DualVLN checkpoint 路径  
- **`--patched-model-path`**：输出的 patched 目录（若已存在会报错，需换名或删掉再跑）  
- **`--output`**：可选，将检查结果写入 JSON  

生成后的 **patched 目录**（例如 `checkpoints/InternVLA-N1-DualVLN-qwen25vl-s2-view`）结构：

- 除 `config.json` 外，其余文件（safetensors、tokenizer、preprocessor 等）均为指向原 checkpoint 的 **符号链接**
- `config.json` 被重写为：
  - `model_type`: `"qwen2_5_vl"`
  - `architectures`: `["Qwen2_5_VLForConditionalGeneration"]`
  - 并带有 `_comment` 说明此为 S2-only 实验用视图

vLLM 加载该目录时会按标准 Qwen2.5-VL 解析，自动忽略权重中与 S2 无关的 S1 张量（如 latent_queries、rgb_model、memory_encoder 等）。

### 2.3 权重分布说明（供参考）

- 总张量数约 1338；其中 **S2（Qwen2.5-VL 主干）** 约 1063，**S1 相关** 约 275  
- S1 权重分布在部分 shard 中，与 S2 权重共存；vLLM 只映射需要的参数，多余张量被忽略  

详见 `scripts/eval/tools/check_dualvln_vllm_feasibility.py` 及输出的 `vllm_feasibility.json`。

---

## 3. 启动 vLLM 服务

### 3.1 基本命令

使用 **patched 目录** 作为 `vllm serve` 的模型路径，例如：

```bash
vllm serve /root/backup/InternNav/checkpoints/InternVLA-N1-DualVLN-qwen25vl-s2-view \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --limit-mm-per-prompt '{"image":16,"video":0}' \
  --port 8001
```

参数说明：

| 参数 | 说明 |
|------|------|
| 模型路径 | 必须为 **patched 目录**（含 `model_type=qwen2_5_vl` 的 config），不能直接用原始 `InternVLA-N1-DualVLN` |
| `--dtype auto` | 通常为 bfloat16，与训练/原推理一致 |
| `--max-model-len 4096` | 序列长度上限，可按需调整 |
| `--gpu-memory-utilization` | 显存占用比例；若与本地 eval 同机需预留空间给本地模型，可适当减小（如 0.45） |
| `--limit-mm-per-prompt` | 多模态每 prompt 图像数上限；DualVLN 评估 `num_history=8` + 当前帧等，建议 ≥16 |
| `--port 8001` | 服务端口，与评估配置中的 `s2_vllm_url` 一致即可 |

### 3.2 与本地 eval 同机时的显存

- vLLM 进程（7B 级）约占 20–25GB（视 `gpu-memory-utilization` 而定）
- 本地 DualVLN 模型（用于 `generate_latents` + S1）约 16–17GB  
- 同机运行需保证总显存 ≥ 两者之和（例如 L20 49GB 可把 vLLM 设为 0.45，eval 约 23GB，总约 38GB）

### 3.3 验证服务

```bash
curl -s http://127.0.0.1:8001/v1/models
```

返回的 `data[].id` 即为当前加载的模型名（一般为 patched 目录的绝对路径），评估代码会通过 `/v1/models` 自动检测模型名，无需手填。

---

## 4. 评估代码中的 S2 路径替换（vLLM 分支）

### 4.1 修改概要

在 **不改变** 原有「无 vLLM」行为的前提下，增加「当配置了 vLLM URL 时，S2 文本生成走 HTTP 请求」的分支。

涉及文件与改动点：

1. **`internnav/model/basemodel/internvla_n1/internvla_n1_policy.py`**
   - 新增依赖：`base64`、`io`、`requests`（用于调用 vLLM HTTP API）
   - 在 `InternVLAN1Net.__init__` 中读取 `model_config.s2_vllm_url`、`s2_vllm_model`；若存在 URL，则可选调用 `_detect_vllm_model_name()` 从 `/v1/models` 拉取模型名
   - 新增方法：
     - `_detect_vllm_model_name()`：请求 `{s2_vllm_url}/v1/models` 取第一个模型 id
     - `_pil_to_data_url(image)`：PIL 转 base64 data URL（JPEG）
     - `_conversation_to_openai(conversation_history)`：将内部多模态对话格式转为 OpenAI 风格 messages（文本 + `image_url` 的 base64）
     - `_vllm_generate(conversation_history, max_new_tokens)`：对 vLLM 的 `/v1/chat/completions` 发 POST，返回生成文本
   - 在 **`s2_step()`** 中：
     - 仍按原逻辑构造 `conversation_history`、`inputs`（processor、images、input_ids 等）
     - **若配置了 `s2_vllm_url`**：  
       - 调用 `_vllm_generate(self.conversation_history, max_new_tokens=128)` 得到 `self.llm_output`  
       - 用本地 `tokenizer.encode(llm_output, add_special_tokens=False)` 得到生成 token id 列表，再与 `inputs.input_ids` 拼成完整 `output_ids`（与原先 `model.generate()` 返回的 `sequences` 形状一致）
     - **否则**：  
       - 保持原 `model.generate(...)` 调用，并照旧从 `output_ids` 解码得到 `self.llm_output`
     - 后续逻辑**完全共用**：  
       - 若 `llm_output` 含数字则解析为 pixel goal，并调用 `self.model.generate_latents(output_ids, inputs.pixel_values, image_grid_thw)`（仍在本地）  
       - 否则解析为离散动作  
       - S1 的 `s1_step_latent` / `generate_traj` 未做任何改动  

2. **配置与入口**
   - 新增评估配置（见下节），在 `agent.model_settings` 中增加 `s2_vllm_url`（及可选的 `s2_vllm_model`），即可在对应 eval 中启用 vLLM S2 分支。

这样，**仅 S2 的 autoregressive 文本生成** 被替换为 vLLM HTTP 调用；**latent 提取与 S1 轨迹生成** 仍使用本地加载的 DualVLN 模型。

### 4.2 配置项说明

在 `AgentCfg` 的 `model_settings` 中可增加：

| 键 | 类型 | 说明 |
|----|------|------|
| `s2_vllm_url` | str 或 None | vLLM 服务 base URL，例如 `"http://127.0.0.1:8001"`。为 `None` 或不设置时使用本地 HF `model.generate()` |
| `s2_vllm_model` | str 或 None | 可选。vLLM 侧模型名（与 `/v1/models` 返回的 id 一致）。为 `None` 时在首次使用前通过 `_detect_vllm_model_name()` 自动获取 |

`ModelCfg` 使用 `extra='allow'`，因此无需改 schema 即可传入上述字段。

---

## 5. 评估配置与运行方式

### 5.1 新增配置文件

- **`scripts/eval/configs/habitat_dual_system_vllm_cfg.py`**  
  - Habitat 环境 + DualVLN，且启用 vLLM S2：  
  - `model_settings` 中设 `s2_vllm_url": "http://127.0.0.1:8001"`、`s2_vllm_model": None`  
  - 输出目录示例：`./logs/habitat/test_dual_system_vllm`  
  - 默认 `max_eval_episodes=8`、`save_video=False`、`export_replay_subset=False`，便于快速验证  

- **`scripts/eval/configs/havln_http_dual_system_vllm_cfg.py`**  
  - HA-VLN HTTP 环境 + DualVLN + vLLM S2：  
  - 同样 `s2_vllm_url": "http://127.0.0.1:8001"`  
  - `env_type='havln_http'`，需配合 HA-VLN 的 HTTP env server（如 8899 端口）  
  - 输出目录示例：`./logs/habitat/test_dual_system_ha_http_vllm`  

### 5.2 运行 Habitat 评估（vLLM S2）

1. 先启动 vLLM（见第 3 节），确认 `http://127.0.0.1:8001/v1/models` 可访问。  
2. 若使用 **纯 Habitat**（非 HA-VLN HTTP）：

   ```bash
   cd /root/backup/InternNav
   python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_vllm_cfg.py
   ```

3. 若使用 **HA-VLN HTTP 环境**，需先启动 HA-VLN 的 env server，再跑：

   ```bash
   python scripts/eval/eval.py --config scripts/eval/configs/havln_http_dual_system_vllm_cfg.py
   ```

4. 若希望**不使用 vLLM**，仍用本地 S2：  
   - 使用原有 config（如 `habitat_dual_system_cfg.py`），或  
   - 在 vLLM 版 config 中把 `s2_vllm_url` 设为 `None` 并重启 eval。

### 5.3 数据与场景

- Habitat 相关路径、scene、dataset 等仍由各 config 的 `eval_settings` 和 env 的 `config_path` 决定（如 `vln_r2r.yaml`），与是否启用 vLLM 无关。  
- 使用 mini 子集或指定 episode 数时，在对应 config 中设置 `dataset_path_override`、`max_eval_episodes` 等即可。

---

## 6. 流程小结

| 步骤 | 内容 |
|------|------|
| 1 | 用 `check_dualvln_vllm_feasibility.py` 生成 **patched S2 视图** 目录（如 `InternVLA-N1-DualVLN-qwen25vl-s2-view`） |
| 2 | 使用该 **patched 目录** 启动 **vLLM serve**（端口如 8001），并设置 `limit_mm_per_prompt`、`gpu-memory-utilization` 等 |
| 3 | 在评估 config 中设置 **`s2_vllm_url`**（及可选 `s2_vllm_model`） |
| 4 | 运行 **eval.py** 时，S2 文本生成走 vLLM HTTP，`generate_latents` 与 S1 仍走本地模型 |

未设置 `s2_vllm_url` 时行为与改动前完全一致，便于对比与回退。

---

## 7. 相关文档与脚本

- **推理优化与实验顺序**：`docs/dualvln_inference_optimization_plan.md`（含 vLLM 路线与 5.3 节建议）  
- **KV cache 与 backend 实验**：`docs/dualvln_kv_cache_experiment_plan.md`  
- **S2 后端对比（HF vs vLLM）**：`scripts/eval/tools/benchmark_dualvln_s2_backends.py`（离线 replay 对比，不接 Habitat）  
- **Patched 视图生成与可行性检查**：`scripts/eval/tools/check_dualvln_vllm_feasibility.py`  

以上文档与脚本与本文档描述的「Habitat 评估中 S2 路径替换为 vLLM」互为补充：本文档侧重**在线评估集成**；benchmark 与优化计划侧重离线分析与实验设计。

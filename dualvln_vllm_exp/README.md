# DualVLN / vLLM 最小实验包

这套脚本按两个问题来拆：

1. 标准 vLLM generate runner 能不能直接吃 `prompt_embeds`
2. 如果能，能不能在标准 runner 上拿到 prompt hidden states

## 文件

- `01_export_hf_prompt_embeds_bundle.py`
  - 从 HF full DualVLN + 现有 sample `.pt` 导出：
    - `prompt_embeds`
    - `full_prompt_ids`
    - `position_ids`
    - `ref_traj_latents`
- `02_vllm_prompt_embeds_smoketest.py`
  - 只测标准 `LLM.generate()` 是否能吃 `prompt_embeds`
- `03_vllm_extract_hidden_states_from_prompt_embeds.py`
  - 只测标准 `generate()` + `extract_hidden_states` 能否导出 prompt hidden states

## 推荐跑法

### 1) 先导 bundle

```bash
python ./dualvln_vllm_exp/01_export_hf_prompt_embeds_bundle.py \
  --model-path checkpoints/InternVLA-N1-DualVLN \
  --sample-pt logs/habitat/hf_generate_latents_baseline_replay1/samples/sample_0000_zsNo4HB9uLZ_0001_step_0003.pt \
  --out logs/habitat/prompt_embeds_bundle_sample_0000.pt \
  --device cuda:0 \
  --dtype bfloat16
```

### 2) 再测 prompt_embeds smoke test

```bash
python ./dualvln_vllm_exp/02_vllm_prompt_embeds_smoketest.py \
  --model-path checkpoints/InternVLA-N1-DualVLN \
  --bundle logs/habitat/prompt_embeds_bundle_sample_0000.pt \
  --out-json logs/habitat/prompt_embeds_smoketest_sample_0000.json \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --model-impl transformers \
  --trust-remote-code \
  --enforce-eager \
  --try-enable-prompt-embeds
```

### 3) 最后测 hidden states extraction

```bash
python ./dualvln_vllm_exp/03_vllm_extract_hidden_states_from_prompt_embeds.py \
  --model-path checkpoints/InternVLA-N1-DualVLN \
  --bundle logs/habitat/prompt_embeds_bundle_sample_0000.pt \
  --out-json logs/habitat/prompt_embeds_hidden_states_sample_0000.json \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --model-impl transformers \
  --trust-remote-code \
  --enforce-eager \
  --layer-id 27 \
  --try-enable-prompt-embeds
```

## 结果解读

- 第 2 步成功：说明标准 runner 至少能接住 `prompt_embeds`
- 第 3 步成功且 tail 接近 `ref_traj_latents`：说明 `prompt_embeds + extract_hidden_states` 这条线可行
- 第 2 步成功但第 3 步失败：说明应该 patch runner 输出接口，而不是输入接口
- 第 2 步都失败：说明应该继续看 `prompt_embeds` 输入链路或当前模型/多模态 backend 支持情况

## 注意

这些脚本尽量做成通用版，但 sample `.pt` 的 key 名可能和你本地不同。如果 `01` 报找不到 key，直接看它打印的 `available_sample_keys`，然后用 `--ids-key / --pixel-key / --grid-key` 显式指定。

# DualVLN vLLM Experiment Runbook

This document records the code changes and the current execution status for:

- `01_export_hf_prompt_embeds_bundle.py`
- `02_vllm_prompt_embeds_smoketest.py`
- `03_vllm_extract_hidden_states_from_prompt_embeds.py`
- `05_compare_pooling_tasks.py`

## Goal

Run the three scripts in `dualvln_vllm_exp` against:

- model: `checkpoints/InternVLA-N1-DualVLN`
- sample: `logs/habitat/hf_generate_latents_baseline_replay1/samples/sample_0000_zsNo4HB9uLZ_0001_step_0003.pt`

## Files Changed

### InternNav

- `internnav/model/basemodel/internvla_n1/internvla_n1.py`
- `dualvln_vllm_exp/01_export_hf_prompt_embeds_bundle.py`
- `dualvln_vllm_exp/02_vllm_prompt_embeds_smoketest.py`

### vLLM

- `vllm/model_executor/models/qwen2_5_vl.py`
- `vllm/model_executor/models/registry.py`
- `vllm/transformers_utils/configs/extract_hidden_states.py`
- `vllm/v1/spec_decode/extract_hidden_states.py`
- `vllm/v1/utils.py`

## 01 Export HF Prompt Embeds Bundle

### Problems fixed

- `InternVLAN1ModelConfig` was missing top-level LM fields such as
  `hidden_size` and `num_attention_heads`.
- The loaded model did not expose `get_rope_index()`.
- HF checkpoint loading printed a very large `UNEXPECTED` list for
  task-specific weights that are intentionally unused in this experiment.
- The script used deprecated `torch_dtype`.
- The script needed to work with the newer Qwen2.5-VL embedding and visual
  interfaces.

### Code changes

In `internnav/model/basemodel/internvla_n1/internvla_n1.py`:

- Synced LM fields from `text_config` to top-level config aliases:
  `hidden_size`, `intermediate_size`, `num_attention_heads`,
  `num_hidden_layers`, `num_key_value_heads`, `vocab_size`, `hidden_act`,
  `max_position_embeddings`, `rms_norm_eps`, `rope_theta`,
  `sliding_window`, `max_window_layers`, `use_sliding_window`.
- Added `InternVLAN1ForCausalLM.get_rope_index(...)`.
- Added helpers for embedding lookup and visual feature extraction that match
  newer Qwen2.5-VL interfaces.
- Ignored task-specific checkpoint branches during HF loading:
  `action_*`, `cond_projector`, `memory_encoder`, `rgb_*`, `traj_dit`.
- Adjusted checkpoint conversion mapping so unrelated branches are not remapped
  into the language model namespace.

In `dualvln_vllm_exp/01_export_hf_prompt_embeds_bundle.py`:

- Added project root to `sys.path`.
- Switched `from_pretrained(..., dtype=...)`.
- Added compatibility helpers for token embeddings and visual outputs.

### Verified result

Command:

```bash
python ./dualvln_vllm_exp/01_export_hf_prompt_embeds_bundle.py \
  --model-path checkpoints/InternVLA-N1-DualVLN \
  --sample-pt logs/habitat/hf_generate_latents_baseline_replay1/samples/sample_0000_zsNo4HB9uLZ_0001_step_0003.pt \
  --out logs/habitat/prompt_embeds_bundle_sample_0000.pt \
  --device cuda:0 \
  --dtype bfloat16
```

Outputs:

- `logs/habitat/prompt_embeds_bundle_sample_0000.pt`
- `logs/habitat/prompt_embeds_bundle_sample_0000.pt.json`

Observed values:

- `prompt_embeds_shape = [2077, 3584]`
- `n_query = 4`
- `image_token_count = 1955`
- `hf_ref_vs_manual.mean_abs = 0.0`
- `hf_ref_vs_manual.max_abs = 0.0`

Interpretation:

- The manually rebuilt `prompt_embeds + rope` path matches the HF reference
  path exactly for this sample.

## 02 vLLM Prompt Embeds Smoketest

### Problems fixed

- vLLM could not load `InternVLAN1ForCausalLM` natively.
- The generic `transformers` backend was not the stable path for this model.
- The old request path that sent multimodal request metadata alongside
  `prompt_embeds` could hang or fail.
- The stable path required V2 model runner behavior.

### Code changes

In `vllm/model_executor/models/qwen2_5_vl.py`:

- Added `InternVLAN1ForCausalLM` as a vLLM adapter subclass of the native
  Qwen2.5-VL model implementation.
- Ignored these extra checkpoint branches while loading weights:
  `model.action_decoder.`
  `model.action_encoder.`
  `model.cond_projector.`
  `model.latent_queries`
  `model.memory_encoder.`
  `model.rgb_model.`
  `model.rgb_resampler.`
  `model.traj_dit.`

In `vllm/model_executor/models/registry.py`:

- Registered:
  `InternVLAN1ForCausalLM -> qwen2_5_vl.InternVLAN1ForCausalLM`

In `dualvln_vllm_exp/02_vllm_prompt_embeds_smoketest.py`:

- Enabled `VLLM_USE_V2_MODEL_RUNNER=1`.
- Forced `async_scheduling=False`.
- Enabled `prompt_embeds`.
- If the architecture is `InternVLAN1ForCausalLM` and the requested
  `model_impl` is `transformers`, the script overrides to `auto`.
- Stopped sending request-side multimodal metadata.

### Important behavioral note

The exported bundle from script `01` already contains prompt embeddings with
image embeddings fused into the sequence. For this smoketest, the stable vLLM
request is:

- `prompt_token_ids`
- `prompt_embeds`
- `prompt`

and not:

- `multi_modal_data`
- `mm_kwargs`
- `mm_placeholders`
- `mm_hashes`

### Verified result

Command:

```bash
python ./dualvln_vllm_exp/02_vllm_prompt_embeds_smoketest.py \
  --model-path checkpoints/InternVLA-N1-DualVLN \
  --bundle logs/habitat/prompt_embeds_bundle_sample_0000.pt \
  --out-json logs/habitat/prompt_embeds_smoketest_sample_0000.v15.json \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --model-impl transformers \
  --trust-remote-code \
  --enforce-eager \
  --try-enable-prompt-embeds
```

Output:

- `logs/habitat/prompt_embeds_smoketest_sample_0000.v15.json`

Observed values:

- `success = true`
- `generated_token_ids = [279]`
- `generated_text = " the"`
- `finish_reason = "length"`

## 03 Extract Hidden States From Prompt Embeds

### Current root cause identified

The current traceback:

- `SpeculativeConfig`
- `extract_hidden_states`
- `get_hf_text_config(...)`

comes from a config-compatibility bug in the `ExtractHiddenStatesConfig`
wrapper, not from the prompt embeddings themselves.

What happens:

1. The original InternVLA config has a valid `text_config`.
2. `ExtractHiddenStatesConfig` rebuilds the config through `to_dict()`.
3. The nested `text_config` becomes a plain Python `dict`.
4. `get_hf_text_config()` calls `config.get_text_config()`.
5. It receives that plain dict instead of a config object and fails because
   the dict has no `num_attention_heads` attribute.

### Additional fixes applied

In `vllm/transformers_utils/configs/extract_hidden_states.py`:

- Added a `get_text_config(...)` fallback for multimodal remote-code configs.
- If `text_config` was degraded into a plain dict by `to_dict()`, the wrapper
  now falls back to the top-level config object which still carries the
  language-model shape attributes required by vLLM.

In `vllm/v1/spec_decode/extract_hidden_states.py`:

- Fixed draft token shape normalization.
- `sampled_token_ids` may already arrive as `[batch, 1]`; the proposer now
  returns `[batch, 1]` consistently instead of accidentally producing
  `[batch, 1, 1]`.

In `dualvln_vllm_exp/03_vllm_extract_hidden_states_from_prompt_embeds.py`:

- Defaulted `VLLM_USE_V2_MODEL_RUNNER=0`.
- Reused the stable request shape from script `02`:
  `prompt_token_ids + prompt_embeds + prompt`.
- Removed the old request-side multimodal metadata path.
- Added automatic `model_impl=auto` override for `InternVLAN1ForCausalLM`.
- Added robust parsing and summary for the exported hidden states.
- Fixed tail-vs-reference comparison to use the last `n_query` token rows.

### Important behavioral note

For script `03`, the stable path is the legacy `GPUModelRunner`:

- `VLLM_USE_V2_MODEL_RUNNER=0`

Reason:

- `extract_hidden_states` is implemented in the legacy speculative path.
- The V2 model runner currently errors with:
  `NotImplementedError: extract_hidden_states is not supported yet.`

### Verified result

Command:

```bash
python ./dualvln_vllm_exp/03_vllm_extract_hidden_states_from_prompt_embeds.py \
  --model-path checkpoints/InternVLA-N1-DualVLN \
  --bundle logs/habitat/prompt_embeds_bundle_sample_0000.pt \
  --out-json logs/habitat/prompt_embeds_extract_hidden_states_sample_0000.json \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --model-impl transformers \
  --trust-remote-code \
  --enforce-eager \
  --try-enable-prompt-embeds
```

Output:

- `logs/habitat/prompt_embeds_extract_hidden_states_sample_0000.json`

Observed values:

- `success = true`
- `generated_token_ids = [220]`
- `generated_text = " "`
- `finish_reason = "length"`
- `hidden_states_shape = [2077, 1, 3584]`
- `token_ids_shape = [2077]`
- `tail_vs_ref_traj_latents.cosine = 0.31312644481658936`

Interpretation:

- vLLM successfully exported one requested hidden-state layer
  (`layer_id = 27`) for every prompt token.
- The output format is `[num_tokens, num_selected_layers, hidden_size]`.

## 05 Compare Pooling Tasks

### Goal

Use the local vLLM API exactly as exposed by this checkout and answer three
questions with the smallest possible experiment surface:

1. Does this version support `runner="pooling"`?
2. Does this version support `prompt_embeds + LLM.encode(...)`?
3. Which route actually returns token-wise tensors:
   `pooling_task="token_embed"`, `pooling_task="embed"`, or `LLM.reward(...)`?

### Local API probe

From `dualvln_vllm_exp/00_probe_local_vllm_api.py`:

- `LLM(...)` supports `runner="pooling"`.
- `LLM.encode(...)` exists and accepts `pooling_task=...`.
- Valid `pooling_task` values include `token_embed` and `embed`.
- `LLM.reward(...)` exists as a method.
- Constructor-side `convert=` only supports:
  `auto`, `none`, `embed`, `classify`.

Interpretation:

- `reward` must be tested through `LLM.reward(...)`, not via
  `convert="reward"`.
- The cleanest pooling probe is:
  `runner="pooling", convert="embed", enable_prompt_embeds=True`
  plus prompt payload:
  `prompt + prompt_token_ids + prompt_embeds`.

### Code changes

In `dualvln_vllm_exp/05_compare_pooling_tasks.py`:

- Resolved the reference latent tensor from the bundle using:
  `traj_latents`, `ref_traj_latents`, or `manual_traj_latents`.
- Switched the probe matrix to three routes only:
  `token_embed_encode`, `embed_encode`, `reward_call`.
- Aligned the constructor with the probed signature:
  `runner="pooling", convert="embed", enable_prompt_embeds=True`.
- Tested reward through `LLM.reward(...)` instead of unsupported
  constructor-side `convert="reward"`.
- Added summary fields that directly answer whether pooling runner works,
  whether `prompt_embeds + encode` works, and which routes are token-wise
  versus vector-only.

### Verified result

Command:

```bash
python /root/backup/InternNav/dualvln_vllm_exp/05_compare_pooling_tasks.py \
  --model-path /root/backup/InternNav/checkpoints/InternVLA-N1-DualVLN \
  --bundle /root/backup/InternNav/logs/habitat/prompt_embeds_bundle_sample_0000.pt \
  --out-json /root/backup/InternNav/logs/habitat/pooling_compare_sample_0000.json \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --trust-remote-code \
  --enforce-eager
```

Output:

- `logs/habitat/pooling_compare_sample_0000.json`

Observed values:

- `pooling_runner_supported = true`
- `prompt_embeds_encode_supported = true`
- `token_embed_returns_tokenwise = true`
- `embed_returns_tokenwise = false`
- `reward_supported = false`
- `token_wise_routes = ["token_embed_encode"]`
- `vector_routes = ["embed_encode"]`
- `unsupported_routes = ["reward_call"]`

Route details:

- `token_embed_encode`
  - `tensor_shape = [2077, 3584]`
  - `tail_vs_ref.cosine = 0.7981945276260376`
  - `best_window_vs_ref.start = 2073`
  - `tail_is_best_window = true`
- `embed_encode`
  - `tensor_shape = [3584]`
  - sequence-level vector, not token-wise
- `reward_call`
  - failed with:
    `Unsupported task: 'token_classify' Supported tasks: ['token_embed', 'embed']`

Interpretation:

- Pooling runner is supported in this checkout.
- `prompt_embeds + encode(...)` is supported.
- The practical token-wise pooling route is `pooling_task="token_embed"`.
- This route already outperforms the hidden-state layer sweep baseline
  (`0.7982` vs `0.3297` cosine on the trajectory-token tail), so further
  minimal validation should continue from pooling rather than blind layer
  scanning.

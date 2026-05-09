# vLLM Patchset Status

Base upstream commit used for the current fork snapshot:

- `57431d823`

Generated artifacts:

- `full-fork-vs-upstream.patch`: full patch against the upstream base commit
- `current-file-list.txt`: all currently changed vLLM files
- `current-stat.txt`: patch statistics

## Current Classification

### Patch-critical for current single-vLLM DualVLN path

These are the files that currently carry the unavoidable runtime ingress /
worker-side changes for prompt-embed prefill, hidden-state collection, or
same-request suffix continuation:

- `vllm/entrypoints/llm.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/engine/core_client.py`
- `vllm/v1/engine/llm_engine.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/request.py`
- `vllm/v1/worker/gpu_input_batch.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/gpu_worker.py`
- `vllm/inputs/data.py`
- `vllm/renderers/base.py`
- `vllm/v1/engine/input_processor.py`
- `vllm/outputs.py`
- `vllm/pooling_params.py`

### Thin model-adaptation patch

These let the InternVLA-N1 checkpoint load through a native Qwen2.5-VL-backed
vLLM path without carrying full business logic inside the fork:

- `vllm/model_executor/models/qwen2_5_vl.py`
- `vllm/model_executor/models/registry.py`
- `vllm/model_executor/models/transformers/base.py`

### Operational / debug / legacy-experiment patch debt

These do not define the primary adapter/runtime contract and should be treated
as cleanup candidates during later patch minimization:

- `vllm/config/speculative.py`
- `vllm/debug_dump.py`
- `vllm/envs.py`
- `vllm/forward_context.py`
- `vllm/model_executor/model_loader/base_loader.py`
- `vllm/transformers_utils/configs/extract_hidden_states.py`
- `vllm/v1/core/sched/output.py`
- `vllm/v1/engine/__init__.py`
- `vllm/v1/engine/output_processor.py`
- `vllm/v1/outputs.py`
- `vllm/v1/spec_decode/extract_hidden_states.py`
- `vllm/v1/utils.py`

## Patch-Minimization Direction

The current refactor in `InternNav` moves business adaptation and runtime
transport out of the fork first. The next vLLM-focused cleanup step should
compress the remaining patch set around three contract surfaces only:

1. embed-prompt ingress with explicit multimodal metadata
2. worker-side hidden-state collection for latent suffix extraction
3. optional same-request continuation scheduling

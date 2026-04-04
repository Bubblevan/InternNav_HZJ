# vLLM Native Dataflow

This document summarizes the native `vLLM` request path relevant to multimodal chat, `EmbedsPrompt`, prompt-embedding inputs, M-RoPE construction, and raw hidden-state return. The codebase referenced here is `/root/backup/vllm`.

## 1. File Map

The most relevant files are:

1. Prompt schemas:
   - [`/root/backup/vllm/vllm/inputs/data.py`](/root/backup/vllm/vllm/inputs/data.py)
2. User-facing `LLM` entrypoint:
   - [`/root/backup/vllm/vllm/entrypoints/llm.py`](/root/backup/vllm/vllm/entrypoints/llm.py)
3. Chat/text renderer and multimodal prompt processing:
   - [`/root/backup/vllm/vllm/renderers/base.py`](/root/backup/vllm/vllm/renderers/base.py)
4. Frontend input normalization into `EngineCoreRequest`:
   - [`/root/backup/vllm/vllm/v1/engine/input_processor.py`](/root/backup/vllm/vllm/v1/engine/input_processor.py)
5. Engine request structure:
   - [`/root/backup/vllm/vllm/v1/engine/__init__.py`](/root/backup/vllm/vllm/v1/engine/__init__.py)
6. Qwen2.5-VL model implementation:
   - [`/root/backup/vllm/vllm/model_executor/models/qwen2_5_vl.py`](/root/backup/vllm/vllm/model_executor/models/qwen2_5_vl.py)
7. GPU worker pooling and raw hidden-state return:
   - [`/root/backup/vllm/vllm/v1/worker/gpu_model_runner.py`](/root/backup/vllm/vllm/v1/worker/gpu_model_runner.py)
8. Pooling parameter and output objects:
   - [`/root/backup/vllm/vllm/pooling_params.py`](/root/backup/vllm/vllm/pooling_params.py)
   - [`/root/backup/vllm/vllm/outputs.py`](/root/backup/vllm/vllm/outputs.py)
   - [`/root/backup/vllm/vllm/v1/engine/output_processor.py`](/root/backup/vllm/vllm/v1/engine/output_processor.py)

## 2. High-Level Native Graph

For a multimodal chat generation request, the native path is:

1. `LLM.chat(...)`
2. `_preprocess_chat_one(...)` or `_preprocess_chat(...)`
3. `renderer.render_chat(...)`
4. renderer returns `ProcessorInputs`
5. `InputProcessor.process_inputs(...)`
6. `EngineCoreRequest(prompt_token_ids, prompt_embeds, mm_features, ...)`
7. engine/scheduler/worker batch the request
8. model executes forward
9. generation path returns `RequestOutput`

For pooling or prompt-embedding requests, the path is similar, but the output object is `PoolingRequestOutput`, and `PoolingOutput.hidden_states` can be populated when requested.

## 3. Input Schema Layer

The key user-facing prompt types are declared in [`inputs/data.py:65-99`](/root/backup/vllm/vllm/inputs/data.py#L65).

Three prompt forms matter here:

1. `TokensPrompt`
   - explicit `prompt_token_ids`
2. `EmbedsPrompt`
   - explicit `prompt_embeds`
   - optional `prompt_token_ids`
   - optional `mm_kwargs`
   - optional `mm_hashes`
   - optional `mm_placeholders`
3. prompt with `multi_modal_data`
   - raw multimodal payload to be processed by the renderer/processor

The existence of `EmbedsPrompt.prompt_token_ids` is important because some models, such as Qwen2.5-VL, still need token-level information to construct positions even when the actual backbone input is `prompt_embeds`.

## 4. `LLM.chat()` to `ProcessorInputs`

The native chat preprocessing entrypoint is [`llm.py:870-943`](/root/backup/vllm/vllm/entrypoints/llm.py#L870).

The flow is:

1. `_preprocess_chat(...)` builds `ChatParams`
2. it calls `renderer.render_chat(...)`
3. renderer returns `ProcessorInputs`
4. `_preprocess_chat_one(...)` is just the single-conversation wrapper

The main chat execution path then uses [`llm.py:1848-1904`](/root/backup/vllm/vllm/entrypoints/llm.py#L1848):

1. preprocess each conversation
2. pass prompts into `_render_and_run_requests(...)`
3. `_render_and_add_requests(...)`
4. `_add_request(...)`
5. run the engine

This is why reusing a previously built `processed_prompt` is possible and also why double preprocessing can create multimodal cache mismatches if handled incorrectly by downstream custom code.

## 5. Renderer Behavior for `EmbedsPrompt`

The most important renderer logic is `_process_embeds(...)` in [`renderers/base.py:688-840`](/root/backup/vllm/vllm/renderers/base.py#L688).

What it does:

1. validates `prompt_embeds` shape
2. validates that `prompt_token_ids` length matches `prompt_embeds.shape[0]`
3. supports two mutually exclusive multimodal modes:
   - explicit passthrough:
     - `mm_kwargs`
     - `mm_hashes`
     - `mm_placeholders`
   - raw `multi_modal_data`
4. if explicit metadata is provided:
   - require all three fields together
   - require `prompt_token_ids`
   - normalize `mm_kwargs`
5. if raw `multi_modal_data` is provided:
   - process it through the multimodal pipeline
   - recover or rebuild placeholder ranges if needed

This function is the key native support for “prompt embeddings plus exact multimodal metadata”.

## 6. InputProcessor: `ProcessorInputs` -> `EngineCoreRequest`

The frontend normalization step is [`input_processor.py:187-340`](/root/backup/vllm/vllm/v1/engine/input_processor.py#L187).

Important behaviors:

1. validate whether the request is generation or pooling
2. normalize `prompt_token_ids` and `prompt_embeds`
3. gather multimodal metadata from either:
   - `decoder_inputs["mm_kwargs/mm_placeholders/mm_hashes"]`
   - or embeds-based prompt equivalents
4. sort multimodal items by placeholder offset
5. build a flat `list[MultiModalFeatureSpec]`
6. return `EngineCoreRequest(...)`

The `MultiModalFeatureSpec` objects are built in [`input_processor.py:315-331`](/root/backup/vllm/vllm/v1/engine/input_processor.py#L315).

This is the point where native `vLLM` collapses renderer outputs into one canonical multimodal contract:

- `prompt_token_ids`
- `prompt_embeds`
- `mm_features`

## 7. `EngineCoreRequest` as the Canonical Engine Contract

The actual engine request object is declared in [`engine/__init__.py:66-110`](/root/backup/vllm/vllm/v1/engine/__init__.py#L66).

Its relevant fields are:

1. `prompt_token_ids`
2. `prompt_embeds`
3. `mm_features`
4. `sampling_params`
5. `pooling_params`

This is the native contract to keep in mind. Everything earlier in the stack is preprocessing; everything later assumes this structure already exists.

## 8. How Qwen2.5-VL Uses `mm_features`

The Qwen2.5-VL implementation exposes the relevant multimodal position helpers in [`qwen2_5_vl.py:1027-1095`](/root/backup/vllm/vllm/model_executor/models/qwen2_5_vl.py#L1027).

Two methods are central:

1. `iter_mm_grid_thw(mm_features)`
2. `get_mrope_input_positions(input_tokens, mm_features)`

The logic is:

1. sort multimodal features by placeholder offset
2. read each item’s `image_grid_thw` or `video_grid_thw`
3. convert those grids into merged LLM token-space dimensions
4. interleave:
   - plain text position ids
   - multimodal grid position ids
5. return final M-RoPE positions

This means that for Qwen2.5-VL, M-RoPE is fundamentally a function of:

1. token sequence
2. multimodal offsets
3. multimodal grid shapes

In native `vLLM`, that information is carried by `mm_features`.

## 9. Generation Path Output

For generation requests, the engine returns `RequestOutput` objects whose prompt and completion parts remain separated:

1. `request_output.prompt_token_ids`
2. `request_output.outputs[0].token_ids`
3. `request_output.outputs[0].text`

This is why, in a strict-alignment migration, `request_output.prompt_token_ids + completion.token_ids` is the canonical output-id sequence, not a tokenizer re-encode of `completion.text`.

## 10. Pooling Path and Raw Hidden States

Pooling requests are configured with [`pooling_params.py:65-72`](/root/backup/vllm/vllm/pooling_params.py#L65).

The crucial flag is:

- `return_raw_hidden_states: bool = False`

When enabled, the worker-side pooling path in [`gpu_model_runner.py:2940-3054`](/root/backup/vllm/vllm/v1/worker/gpu_model_runner.py#L2940) does:

1. run the model and get `hidden_states`
2. compute pooler output
3. determine which requests are finished
4. for requests that requested raw hidden states:
   - slice the per-request hidden-state span out of the packed `hidden_states`
5. copy both:
   - pooler output
   - raw hidden states
   to CPU

Then the output processor wraps the result as [`output_processor.py:410-418`](/root/backup/vllm/vllm/v1/engine/output_processor.py#L410):

`PoolingOutput(data=pooling_output, hidden_states=pooling_hidden_states)`

And the data container itself is declared in [`outputs.py:66-88`](/root/backup/vllm/vllm/outputs.py#L66).

So the native formal path for raw hidden states is:

1. set `PoolingParams(return_raw_hidden_states=True)`
2. worker slices raw per-request hidden states
3. output processor attaches them to `PoolingOutput.hidden_states`

## 11. Why `EmbedsPrompt` + `prompt_token_ids` + `mm_features` Is the Right Native Latent Interface

From the native `vLLM` point of view, this combination is the most semantically complete latent-prefill interface because:

1. `prompt_embeds` controls the exact backbone input vectors
2. `prompt_token_ids` still preserves token-level metadata for:
   - position construction
   - bookkeeping
3. `mm_kwargs/mm_hashes/mm_placeholders` or derived `mm_features` preserve:
   - multimodal offsets
   - image/video grid shapes
   - cache identity
4. the Qwen2.5-VL model already knows how to compute M-RoPE from `mm_features`

That is why native `vLLM` already contains most of the infrastructure needed for strict DualVLN latent migration.

## 12. Native vLLM Invariants Relevant to Human Debugging

If a human is debugging multimodal/latent mismatches, these are the first native invariants to check:

1. `EmbedsPrompt.prompt_token_ids` length must match `prompt_embeds`
2. explicit `mm_kwargs/mm_hashes/mm_placeholders` must either all exist or all be absent
3. `InputProcessor` sorts multimodal items by position before building `mm_features`
4. Qwen2.5-VL M-RoPE uses `mm_features`, not arbitrary external tensors
5. generation ids are best read from:
   - `request_output.prompt_token_ids`
   - `completion.token_ids`
6. raw pooling hidden states only exist if:
   - pooling task is used
   - `return_raw_hidden_states=True`

## 13. Minimal Native Inspection Order

The fastest inspection order in `vLLM` for multimodal latent semantics is:

1. prompt schema:
   - [`inputs/data.py:78-99`](/root/backup/vllm/vllm/inputs/data.py#L78)
2. renderer embeds path:
   - [`renderers/base.py:688-840`](/root/backup/vllm/vllm/renderers/base.py#L688)
3. input processor:
   - [`input_processor.py:187-340`](/root/backup/vllm/vllm/v1/engine/input_processor.py#L187)
4. engine request struct:
   - [`engine/__init__.py:66-110`](/root/backup/vllm/vllm/v1/engine/__init__.py#L66)
5. model M-RoPE:
   - [`qwen2_5_vl.py:1027-1095`](/root/backup/vllm/vllm/model_executor/models/qwen2_5_vl.py#L1027)
6. raw hidden-state return:
   - [`gpu_model_runner.py:2998-3050`](/root/backup/vllm/vllm/v1/worker/gpu_model_runner.py#L2998)


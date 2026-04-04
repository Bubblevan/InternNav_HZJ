# DualVLN Latents -> vLLM Migration Dataflow

This document describes the migration-specific dataflow in `/root/backup/InternNav`: how the original DualVLN S2/S1 chain is being moved onto a `vLLM`-backed System-2 path, how latents are currently extracted, and where the remaining semantic fault lines are.

This is not the original `/root/InternNav` path. It is the patched migration workspace that introduced:

1. single-vLLM HTTP S2 serving
2. canonical `request_output.prompt_token_ids + completion.token_ids`
3. explicit multimodal metadata handoff
4. shared-engine latent extraction
5. shadow diff against HF

## 1. File Map

The migration-specific files are:

1. Single-vLLM runtime and HTTP client:
   - [`/root/backup/InternNav/internnav/model/utils/dualvln_single_vllm.py`](/root/backup/InternNav/internnav/model/utils/dualvln_single_vllm.py)
2. Latent request bundling:
   - [`/root/backup/InternNav/internnav/model/utils/latents_request.py`](/root/backup/InternNav/internnav/model/utils/latents_request.py)
3. Multimodal alignment helpers:
   - [`/root/backup/InternNav/internnav/model/utils/vllm_latents_alignment.py`](/root/backup/InternNav/internnav/model/utils/vllm_latents_alignment.py)
4. Legacy separate-LLM hidden-state path:
   - [`/root/backup/InternNav/internnav/model/utils/vllm_hidden_latents.py`](/root/backup/InternNav/internnav/model/utils/vllm_hidden_latents.py)
5. Flask server:
   - [`/root/backup/InternNav/scripts/eval/tools/serve_dualvln_single_vllm.py`](/root/backup/InternNav/scripts/eval/tools/serve_dualvln_single_vllm.py)
6. Habitat evaluator integration and shadow diff:
   - [`/root/backup/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py`](/root/backup/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py)
7. Offline deterministic diff harness:
   - [`/root/backup/InternNav/scripts/eval/tools/diff_dualvln_single_vllm_deterministic.py`](/root/backup/InternNav/scripts/eval/tools/diff_dualvln_single_vllm_deterministic.py)

## 2. Online Runtime Topology

When `dualvln_single_vllm_url` is enabled, the evaluator no longer loads a local HF System-2 model. Instead, it does:

1. load local `InternVLAN1System1Runner`
2. construct `DualVLNSingleVLLMHTTPClient`
3. keep Habitat execution and System-1 local rollout in-process
4. outsource only the S2 text generation plus latent extraction to the single-vLLM HTTP server

This handoff is configured in [`habitat_vln_evaluator.py:150-216`](/root/backup/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py#L150).

The server itself exposes one primary endpoint:

- `/dualvln/step_s2`

See [`serve_dualvln_single_vllm.py:32-55`](/root/backup/InternNav/scripts/eval/tools/serve_dualvln_single_vllm.py#L32).

## 3. Dataflow of `/dualvln/step_s2`

### 3.1 HTTP payload

The client serializes messages into JSON and encodes images as base64 PNG, not JPEG. See [`dualvln_single_vllm.py:328-366`](/root/backup/InternNav/internnav/model/utils/dualvln_single_vllm.py#L328).

That means the online transport contract is now:

1. message list
2. text items unchanged
3. image items serialized losslessly as PNG

This was a deliberate migration fix to remove JPEG compression drift from S2.

### 3.2 Server-side generation path

The server delegates to `DualVLNSingleVLLMRunner.step_s2(...)`, implemented in [`dualvln_single_vllm.py:471-535`](/root/backup/InternNav/internnav/model/utils/dualvln_single_vllm.py#L471).

The current generation path is:

1. convert InternNav-style messages to vLLM chat items with `image_pil`
2. call `self.llm._preprocess_chat_one(vllm_messages)`
3. call `self.llm._render_and_run_requests(...)` directly on that already-built `processed_prompt`
4. read:
   - `request_output.prompt_token_ids`
   - `completion.token_ids`
   - `completion.text`
5. if output is non-numeric:
   - return only S2 text
6. if output is numeric:
   - build a latent bundle
   - attach multimodal metadata from the exact same `processed_prompt`
   - run one of the latent backends

This path deliberately avoids `llm.chat()` after a manual preprocess step, to avoid duplicate multimodal preprocessing and cache mismatches.

## 4. Canonical Token Contract in the Migration

The migration now treats these as canonical:

1. prompt ids:
   - `request_output.prompt_token_ids`
2. generated ids:
   - `completion.token_ids`
3. full ids:
   - `prompt_token_ids + generated_token_ids`

That is built in [`dualvln_single_vllm.py:484-488`](/root/backup/InternNav/internnav/model/utils/dualvln_single_vllm.py#L484) and then bundled in [`latents_request.py:85-123`](/root/backup/InternNav/internnav/model/utils/latents_request.py#L85).

This is a major migration change relative to the earlier prototype designs that reconstructed output ids by re-tokenizing `llm_output`.

## 5. `LatentsRequestBundle` as the Migration Boundary

The central handoff object is `LatentsRequestBundle`, declared in [`latents_request.py:8-28`](/root/backup/InternNav/internnav/model/utils/latents_request.py#L8).

It holds:

1. `prompt_token_ids`
2. `generated_token_ids`
3. `full_output_token_ids`
4. `full_output_ids`
5. `pixel_values`
6. `image_grid_thw`
7. `input_images`
8. `latent_queries`
9. `traj_token_index`
10. `n_query`
11. optional:
    - `prompt_embeds`
    - `mm_kwargs`
    - `mm_hashes`
    - `mm_placeholders`
    - `mm_features`

This object is intended to collect both:

1. HF-side tensor view of the request
2. vLLM-side multimodal metadata view of the same request

That mixed role is important when diagnosing current drift.

## 6. How Explicit Multimodal Metadata Is Attached

The main migration helper is [`attach_explicit_mm_metadata_from_processed_inputs(...)` in `latents_request.py:31-82`](/root/backup/InternNav/internnav/model/utils/latents_request.py#L31).

It does:

1. verify `processed_inputs["type"] == "multimodal"`
2. verify `processed_prompt_token_ids == bundle.prompt_token_ids`
3. copy:
   - `mm_kwargs`
   - `mm_hashes`
   - `mm_placeholders`
4. flatten them into `MultiModalFeatureSpec` objects sorted by placeholder offsets
5. store them as `bundle.mm_features`

This validation is one of the strictest migration checks currently in the code: it refuses to proceed if vLLM preprocessing rewrites the canonical prompt-token prefix.

## 7. Shared-Engine Latent Path

The current primary latent path is the shared-engine path in [`dualvln_single_vllm.py:450-462`](/root/backup/InternNav/internnav/model/utils/dualvln_single_vllm.py#L450), which uses:

`self.llm.apply_model(partial(_generate_latents_from_vllm_model, ...))`

The core function is [`_generate_latents_from_vllm_model(...)` in `dualvln_single_vllm.py:158-226`](/root/backup/InternNav/internnav/model/utils/dualvln_single_vllm.py#L158).

Its dataflow is:

1. take canonical `prompt_token_ids`
2. append `TRAJ_TOKEN_INDEX * n_query`
3. build `input_ids`
4. enter vLLM forward context with:
   - `set_current_vllm_config(...)`
   - `set_forward_context(...)`
5. build backbone input embeddings
6. build position ids
7. run `model.forward(...)`
8. slice the last `n_query` hidden states

### 7.1 Embedding reconstruction branch

There are two branches:

1. preferred path:
   - if `mm_features` exists, call `build_prompt_embeds_with_mm_features(...)`
2. fallback path:
   - embed plain token ids
   - run `model.embed_multimodal(pixel_values, image_grid_thw)`
   - scatter visual embeddings into image-token positions
   - write latent queries into the suffix

This branch point is visible at [`dualvln_single_vllm.py:181-205`](/root/backup/InternNav/internnav/model/utils/dualvln_single_vllm.py#L181).

### 7.2 Position construction branch

Position construction also has two branches:

1. preferred path:
   - `compute_mrope_positions_from_mm_features(...)`
2. fallback path:
   - `_compute_qwen2_5_vl_rope_index(...)`

See [`dualvln_single_vllm.py:207-220`](/root/backup/InternNav/internnav/model/utils/dualvln_single_vllm.py#L207).

This is the migration’s attempt to make the shared-engine latent path use the same multimodal position semantics as native vLLM Qwen2.5-VL.

## 8. The `vllm_latents_alignment.py` Helpers

The alignment helpers live in [`vllm_latents_alignment.py:31-147`](/root/backup/InternNav/internnav/model/utils/vllm_latents_alignment.py#L31).

They do three things:

1. derive an `is_multimodal` mask from placeholder ranges
2. reconstruct multimodal embeddings from `mm_features`
3. ask the model for M-RoPE positions via `get_mrope_input_positions(...)`

More specifically:

1. `build_is_multimodal_mask(...)`:
   - reads each placeholder’s offset and length
   - respects `mm_position.is_embed` if present
2. `build_multimodal_embeddings_from_mm_features(...)`:
   - unwraps each `mm_feature.data`
   - moves the payload onto the target device
   - runs `model.embed_multimodal(**mm_kwargs)`
3. `build_prompt_embeds_with_mm_features(...)`:
   - calls `model.embed_input_ids(...)`
   - injects multimodal embeddings
   - overwrites the suffix with `latent_queries`
4. `compute_mrope_positions_from_mm_features(...)`:
   - calls `model.get_mrope_input_positions(prompt_token_ids, mm_features)`

This file is the most explicit statement of the migration thesis:

“Do not approximate HF semantics through raw tensors if vLLM can already express the same multimodal layout through `mm_features`.”

## 9. Separate-LLM Hidden-State Path

The older path still exists in [`vllm_hidden_latents.py:116-253`](/root/backup/InternNav/internnav/model/utils/vllm_hidden_latents.py#L116).

It does:

1. create a second `LLM(...)` with:
   - `runner="pooling"`
   - `convert="embed"`
   - `enable_prompt_embeds=True`
2. attach explicit MM metadata, sometimes by re-running an input-processor path
3. build HF-like prompt embeddings
4. build an `EmbedsPrompt(...)`
5. call `llm.encode(..., PoolingParams(task="token_embed", return_raw_hidden_states=True))`
6. read `PoolingOutput.hidden_states`
7. fallback to debug dump if the official field is absent
8. slice `hidden_states[-n_query:, :]`

This path is valuable for debugging because it is closer to native pooling semantics, but it has structural drawbacks:

1. it creates a second engine
2. it doubles memory pressure
3. it has its own `max_model_len`
4. it introduces another preprocessing and cache boundary

That is why the migration currently treats it as a non-primary debug backend.

## 10. Habitat Integration and Shadow Diff

The patched evaluator adds two migration-specific features:

1. deterministic seeding
2. per-step shadow diff against an HF reference model

Initialization and guard rails are in [`habitat_vln_evaluator.py:131-260`](/root/backup/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py#L131).

Relevant behaviors:

1. fix:
   - `random`
   - `numpy`
   - `torch`
   - `torch.cuda`
2. optionally fix conjunction selection
3. enforce that `dualvln_single_vllm_url` disables local HF S2
4. load a separate HF shadow model only when `shadow_diff_enabled=True`

The shadow HF execution itself is in [`habitat_vln_evaluator.py:516-557`](/root/backup/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py#L516).

It records:

1. `prompt_token_ids`
2. `generated_token_ids`
3. `output_ids`
4. `llm_output`
5. `pixel_goal`
6. `traj_latents`

The divergence classifier is in [`habitat_vln_evaluator.py:559-576`](/root/backup/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py#L559), with stages:

1. `prompt_token_ids`
2. `generated_token_ids`
3. `s2_text`
4. `pixel_goal`
5. `latent`
6. `system1_rollout`
7. `match`

This is the exact logic that produced your `shadow_diff_summary_rank0.json`.

## 11. Deterministic Diff Harness

The offline harness in [`diff_dualvln_single_vllm_deterministic.py`](/root/backup/InternNav/scripts/eval/tools/diff_dualvln_single_vllm_deterministic.py) mirrors the same idea outside Habitat.

It exists to answer one question:

“At the same episode, same step, same prompt, same images, where does divergence first appear?”

It fixes:

1. `random`
2. `numpy`
3. `torch`

and records prompt ids, generated ids, text, pixel goals, latent diffs, and action prefixes.

## 12. Migration Intent vs Current Hybrid Reality

The migration is trying to move the original HF latent contract:

`full_output_ids + visual replacement + latent_queries + HF M-RoPE -> last_hidden_states[:, -N_QUERY:, :]`

into a vLLM-native contract:

`canonical prompt/generated ids + mm_features + prompt_embeds/shared-engine forward -> last hidden states`

However, the current shared-engine implementation is still hybrid in an important way:

1. token ids come from vLLM generation output
2. multimodal metadata comes from vLLM `processed_prompt`
3. but `pixel_values` and `image_grid_thw` inside the bundle are still rebuilt through the HF processor in [`latents_request.py:101-118`](/root/backup/InternNav/internnav/model/utils/latents_request.py#L101)

So the migration currently mixes:

1. vLLM token-space contract
2. vLLM multimodal placeholder contract
3. HF tensor reconstruction of visual inputs

This is a useful intermediate state, but it is not a pure one-engine native path yet.

## 13. Where the Current Code Matches the User’s Findings

Your shadow diff results map cleanly onto the migration architecture:

1. episodes 10/16/17 diverge first at `generated_token_ids`
   - this points upstream of latent extraction
   - the issue is in S2 generation token sequence
2. episodes 43/44 diverge first at `latent`
   - prompt ids match
   - generated ids match
   - text matches
   - pixel goal matches
   - therefore the remaining suspect area is the latent path itself

Within the current migration code, the most likely fault lines for a “latent-only divergence” are:

1. shared-engine hidden-state extraction semantics in `_generate_latents_from_vllm_model(...)`
2. the exact sequence of:
   - prompt embedding rebuild
   - latent-query suffix insertion
   - M-RoPE computation
3. the fact that bundle tensors are partly HF-derived while placeholder metadata is vLLM-derived
4. whether `apply_model(...)` plus manual forward context fully matches native prefill semantics

## 14. The Two Latent Paths, Side by Side

### 14.1 Shared-engine path

Files:

- [`dualvln_single_vllm.py:450-527`](/root/backup/InternNav/internnav/model/utils/dualvln_single_vllm.py#L450)
- [`vllm_latents_alignment.py:31-147`](/root/backup/InternNav/internnav/model/utils/vllm_latents_alignment.py#L31)

Pros:

1. no second engine
2. reuses the same vLLM instance as S2 generation
3. uses canonical generation output ids
4. can reuse `processed_prompt` metadata directly

Risks:

1. custom `apply_model(...)` path may not be fully equivalent to native prefill execution
2. still partly reconstructs the latent bundle through HF processor outputs

### 14.2 Separate-LLM pooling path

Files:

- [`vllm_hidden_latents.py:145-253`](/root/backup/InternNav/internnav/model/utils/vllm_hidden_latents.py#L145)

Pros:

1. aligns more closely with native `EmbedsPrompt + PoolingParams(return_raw_hidden_states=True)` semantics
2. easier to inspect hidden states directly

Risks:

1. second engine
2. OOM and max-model-len pressure
3. extra preprocessing and cache boundary
4. historically depended on debug dumps

## 15. Suggested Human Inspection Order

If a human takes over from here, the fastest order is:

1. canonical S2 token contract:
   - [`dualvln_single_vllm.py:475-516`](/root/backup/InternNav/internnav/model/utils/dualvln_single_vllm.py#L475)
2. explicit MM metadata capture:
   - [`latents_request.py:31-82`](/root/backup/InternNav/internnav/model/utils/latents_request.py#L31)
3. shared-engine latent extraction:
   - [`dualvln_single_vllm.py:158-226`](/root/backup/InternNav/internnav/model/utils/dualvln_single_vllm.py#L158)
4. native vLLM M-RoPE source of truth:
   - [`/root/backup/vllm/vllm/model_executor/models/qwen2_5_vl.py:1027-1095`](/root/backup/vllm/vllm/model_executor/models/qwen2_5_vl.py#L1027)
5. native input canonicalization:
   - [`/root/backup/vllm/vllm/v1/engine/input_processor.py:187-340`](/root/backup/vllm/vllm/v1/engine/input_processor.py#L187)
6. if needed, compare against the debug pooling path:
   - [`vllm_hidden_latents.py:195-253`](/root/backup/InternNav/internnav/model/utils/vllm_hidden_latents.py#L195)
7. then interpret the result through:
   - [`habitat_vln_evaluator.py:559-576`](/root/backup/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py#L559)

## 16. Bottom Line

The migration currently has a clear structure:

1. S2 text generation is already routed through native vLLM generation outputs.
2. The online service uses canonical prompt/generated token ids and lossless PNG transport.
3. The evaluator can diff primary single-vLLM against HF at per-step granularity.
4. The unresolved semantic gap now lives mainly in the latent extraction path, especially for cases where text and pixel-goal already match but latents do not.

That is exactly why the most important inspection target is no longer “prompt/token contract” in the abstract, but the narrower question:

“Given identical prompt ids, identical generated ids, identical text, and identical pixel-goal, why does the shared-engine latent extraction path still produce a different last-hidden-state suffix than original HF `generate_latents()`?”


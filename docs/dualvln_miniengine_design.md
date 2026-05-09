# DualVLN Mini-Engine Design

## Background

The current refactor already moved a meaningful amount of DualVLN-specific code
out of the private vLLM fork:

- `dualvln_vllm_adapter/` now owns S2 request canonicalization, latent bundle
  construction, multimodal attach, and placeholder alignment.
- `dualvln_runtime/` now owns HTTP transport, message serialization, and
  shared-memory image transport.
- `internnav/model/utils/*.py` now act mainly as compatibility shims.

That direction is correct, but it is not the whole story.

Two real constraints still remain in the current codebase:

1. Some DualVLN-specific execution semantics still live below the public vLLM
   API surface, especially around request state, scheduler state, and worker-side
   hidden-state extraction.
2. `dualvln_vllm_adapter/single_vllm.py` still contains a second copy of runtime
   transport/client logic, which shows that "pluginizing" alone does not fully
   solve the boundary problem.

This motivates a second track: define a minimal, task-specific, stateful
inference core that serves only the DualVLN / InternVLA-N1 S2 path.

## Why Not Keep Expanding The vLLM Fork

The current fork still touches a broad internal surface in vLLM, including:

- `vllm/renderers/base.py`
- `vllm/v1/request.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/worker/gpu_model_runner.py`

Those files are exactly where DualVLN-specific latent continuation and native
prefill semantics interact with the scheduler and worker runtime. Keeping every
task-specific behavior inside that fork makes upgrades harder and keeps the
story centered on a large private serving fork.

The goal is therefore not "zero patch", but "small, explainable, isolated
patches", with everything else moved either upward into the adapter or sideways
into a task-specific inference core.

## Why Not Stop At Pluginization

Moving code into `dualvln_vllm_adapter/` and `dualvln_runtime/` is necessary but
not sufficient.

The current adapter still owns orchestration that is more than simple business
format conversion:

- it decides whether to use same-request continuation
- it manages latent prefill fallback
- it coordinates prompt-embed prefill inputs and multimodal metadata
- it still contains a duplicate HTTP/runtime client implementation

That means the current adapter layer mixes two concerns:

1. "business adaptation"
2. "task-specific stateful execution"

`dualvln_miniengine/` is meant to absorb the second category.

## Mini-Engine Scope

### It should do

The mini-engine is a task-specific, stateful inference core for the DualVLN S2
path only. Its minimal responsibilities are:

- accept canonical S2 requests from the adapter
- organize `prompt_embeds + prompt_token_ids + explicit mm metadata`
- manage the minimal request lifecycle for:
  - text generation
  - optional same-request suffix continuation
  - latent prefill fallback
- return raw hidden states / latent tensors for the final query suffix
- expose stable state and metrics back to the adapter

### It should not do

The mini-engine explicitly does not aim to be:

- a general OpenAI-compatible API server
- a general continuous batching scheduler
- a general model registry
- a multi-tenant serving framework
- a generic implementation for every vLLM feature
- a home for LoRA, beam search, speculative decoding variants, or unrelated
  sampling features

## Proposed Layer Boundary

### 1. InternNav integration

Owns task control flow, evaluator logic, environment interaction, and System-1
trajectory generation.

It should know:

- how to build VLN messages
- when to call S2
- how to consume returned pixel goals and latents

It should not know:

- vLLM request internals
- worker hidden-state plumbing
- transport details

### 2. `dualvln_vllm_adapter`

Owns business-level adaptation and canonical request construction.

It should know:

- how DualVLN turns messages into canonical S2 requests
- how to build latent bundles
- how to align placeholders and `mm_features`
- how to translate between InternNav concepts and engine inputs

It should not own:

- HTTP server/client transport
- request-state machines
- scheduler semantics

### 3. `dualvln_runtime`

Owns sidecar transport only.

It should know:

- HTTP ingress/egress
- message serialization
- tensor serialization
- shared-memory image transport

It should not know:

- DualVLN prompt construction
- request lifecycle semantics
- scheduler or worker internals

### 4. `dualvln_miniengine`

Owns task-specific stateful S2 execution semantics.

It should know:

- the minimal S2 request lifecycle
- how to track external and internal request ids
- when continuation was armed or used
- when to fall back from same-request continuation to latent prefill
- how to return text result plus latents result as one coherent unit

It should not know:

- Habitat episode logic
- HTTP or shared-memory transport
- broad model registry concerns

### 5. `patches/vllm-main`

Owns the irreducible remaining kernel patch.

This layer should shrink toward three surfaces only:

1. embeds ingress with explicit multimodal metadata
2. worker-side hidden-state return for suffix latents
3. request/scheduler coordination for same-request continuation

## Minimal Input / Output Interface

The first skeleton uses the following concepts:

- `DualVLNMiniEngineRequest`
- `DualVLNMiniEngineGenerateResult`
- `DualVLNMiniEngineLatentsResult`
- `DualVLNMiniEngineStepResult`
- `DualVLNMiniEngineRequestState`

These are intentionally task-specific. They are not a public serving schema.

The important contract is:

1. the adapter builds one canonical request
2. the mini-engine executes the S2 lifecycle
3. the adapter receives:
   - generated text
   - token ids
   - optional latents
   - minimal debug/runtime metadata

## Request Lifecycle

The proposed minimal lifecycle is:

1. `received`
2. `preprocessed`
3. `generated_text`
4. `latent_prefill_ready`
5. `latent_continuation_armed` if same-request suffix is available
6. `latents_ready`
7. `finished`

Important notes:

- `latent_continuation_armed` and `same_request_continuation_used` are not the
  same thing.
- the mini-engine should record fallback reason explicitly when continuation
  cannot be used.
- request state should stay small and task-oriented, not become a full scheduler
  clone.

## Relationship To The Current vLLM Fork

In the near term, the mini-engine can still be backed by the patched vLLM path.
That is acceptable and expected for MVP.

The point is to invert the dependency:

- today: DualVLN semantics are largely encoded as special behavior inside the
  vLLM fork and a large runner file
- target: DualVLN semantics are expressed in the mini-engine contract, while the
  patched vLLM pieces become one backend implementation detail

This creates a cleaner migration path for later work:

- keep using thin patch where unavoidable
- move orchestration and task-specific state out of `single_vllm.py`
- eventually decide whether some patched worker/scheduler logic can move from
  vLLM into a dedicated mini-engine backend

## MVP Scope

The first implementation step should stay deliberately small:

1. add the `dualvln_miniengine/` package
2. define request/result/state contracts
3. define a base stateful engine interface
4. do not change the current production path yet

After that, the next practical migration can be:

1. split task-specific request-state orchestration out of
   `dualvln_vllm_adapter/single_vllm.py`
2. make the current patched-vLLM runner implement the mini-engine interface
3. only then evaluate whether parts of same-request continuation and native
   latent prefill should move further out of vLLM

## Current MVP Decision

The current implementation now goes one step further than the initial skeleton:

- the mini-engine request/result/state contracts exist
- `backends/vllm_patched_engine.py` implements the mini-engine interface on top
  of the existing patched-vLLM path
- the HTTP runtime entry now targets the mini-engine backend instead of directly
  targeting the old adapter runner
- `dualvln_vllm_adapter/single_vllm.py` has been reduced to a compatibility
  bridge instead of remaining the primary execution owner
- backend/model helper code that still belongs to adapter-side model adaptation
  has been moved into `dualvln_vllm_adapter/model_exec.py`, so the mini-engine
  backend no longer needs to import the compatibility runner module

Important limits still apply:

- the current backend still depends on the patched vLLM fork
- no new vLLM patch surface was added in this step
- GPU runtime validation has not been claimed here

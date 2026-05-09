# DualVLN Mini-Engine

This package now holds the task-specific execution abstraction for
DualVLN / InternVLA-N1 System-2 inference.

Current status:

- `contracts.py` defines the stable request/result objects.
- `request_state.py` defines the minimal lifecycle state tracked for one S2 request.
- `engine.py` defines the mini-engine interface.
- `backends/vllm_patched_engine.py` implements that interface on top of the
  current patched vLLM backend.

What this layer owns:

- text-generate plus latent-extract request lifecycle
- same-request continuation attempt / fallback bookkeeping
- minimal execution metadata such as external/internal request ids, fallback
  reason, and whether prefill reuse was used

What this layer does not own:

- HTTP or shared-memory transport
- Habitat / evaluator logic
- multimodal business adaptation helpers
- generic serving features outside the DualVLN S2 path

Current dependency direction:

- `internnav/` -> `dualvln_runtime/` -> `dualvln_miniengine/`
- `dualvln_miniengine/` reuses adapter helpers and the existing patched-vLLM
  backend

This is still not a standalone replacement for vLLM, and it has not been GPU
runtime-validated in this refactor step.

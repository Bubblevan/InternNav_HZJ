# DualVLN / InternVLA-N1 vLLM Refactor Overview

This refactor changes the project boundary from “a broad private vLLM fork”
into three clearer layers:

## 1. Adapter layer

Located in:

- `/root/backup/InternNav/dualvln_vllm_adapter`

Responsibilities:

- DualVLN S2 request canonicalization
- latent bundle construction
- multimodal attach / placeholder / `mm_features` alignment
- single-vLLM runner orchestration

## 2. Runtime / sidecar layer

Located in:

- `/root/backup/InternNav/dualvln_runtime`

Responsibilities:

- HTTP server/client
- shared-memory image transport
- tensor/message serialization

## 3. vLLM patch layer

Located in:

- `/root/backup/InternNav/patches/vllm-main`
- `/root/backup/vllm`

Responsibilities:

- only the runtime surfaces that cannot currently be expressed through public
  vLLM APIs

## 4. Mini-engine layer

Located in:

- `/root/backup/InternNav/dualvln_miniengine`

Responsibilities:

- minimal stateful S2 execution semantics
- text generation plus latent extraction lifecycle
- patched-vLLM backend integration without pushing that state machine back into
  the adapter or runtime layers

## Compatibility policy

Existing imports under:

- `internnav/model/utils/dualvln_single_vllm.py`
- `internnav/model/utils/latents_request.py`
- `internnav/model/utils/vllm_latents_alignment.py`
- `internnav/model/utils/vllm_hidden_latents.py`

are preserved as thin compatibility shims that re-export the new adapter /
runtime modules.

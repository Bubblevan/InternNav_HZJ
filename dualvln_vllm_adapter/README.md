# DualVLN vLLM Adapter

This package holds DualVLN / InternVLA-N1 specific adaptation logic that sits
above the mini-engine backend:

- canonical S2 request construction for single-vLLM serving
- latent bundle construction and multimodal metadata attachment
- multimodal placeholder / `mm_features` alignment helpers
- hidden-latents business wrappers and compatibility surfaces

The intent is to keep this layer business-aware, while avoiding direct
dependence on HTTP transport or stateful execution orchestration.

Current modules:

- `latents_request.py`: canonical latent request bundle + MM attach helpers
- `mm_alignment.py`: prompt-embed / M-RoPE reconstruction from `mm_features`
- `model_exec.py`: model-facing latent/prompt helper functions shared by adapter
  compatibility code and the mini-engine backend
- `hidden_latents.py`: optional separate pooling-engine latent path
- `single_vllm.py`: thin compatibility bridge that now delegates execution to
  `dualvln_miniengine.backends.vllm_patched_engine`

# DualVLN Single-vLLM Experiment Note

## Goal

The target of this experiment is **not** the previous 3-copy deployment:

- local full DualVLN in Habitat
- one vLLM S2 server
- one hidden-latents sidecar

Instead, the goal is:

- one custom vLLM server that handles both `S2 generate` and `generate_latents`
- one local S1-only runner inside Habitat
- original / default evaluator and server behavior must remain unchanged unless the new path is explicitly enabled

## What Was Implemented

### 1. S1-only local loader

Added a local runner that only loads the System-1 weights from the original DualVLN checkpoint:

- `latent_queries`
- `traj_dit`
- `action_encoder`
- `action_decoder`
- `cond_projector`
- `rgb_model`
- `memory_encoder`
- `rgb_resampler`

File:

- `internnav/model/basemodel/internvla_n1/system1_runner.py`

This avoids loading the entire 7B DualVLN backbone inside Habitat when the single-vLLM backend is enabled.

### 2. Single-engine vLLM server path

Added a custom runner and HTTP server:

- `internnav/model/utils/dualvln_single_vllm.py`
- `scripts/eval/tools/serve_dualvln_single_vllm.py`

This path:

1. uses vLLM `LLM.chat(...)` for S2 text generation
2. if the output is a pixel-goal, runs a second forward on the **same** vLLM-loaded model via `apply_model(...)`
3. returns:
   - `llm_output`
   - `pixel_goal`
   - `latents`

Important detail:

- the server may use a patched Qwen2.5-VL view for vLLM loading
- `latent_queries` are loaded from the original DualVLN checkpoint via `--hf-model-path`

### 3. Habitat evaluator opt-in branch

Updated:

- `internnav/habitat_extensions/vln/habitat_vln_evaluator.py`

New opt-in config keys:

- `dualvln_single_vllm_url`
- `dualvln_single_vllm_timeout`

When `dualvln_single_vllm_url` is set:

- Habitat no longer loads full `InternVLAN1ForCausalLM`
- Habitat loads S1-only local runner
- S2 + latents are requested over HTTP from the single-vLLM server

When the key is not set:

- original evaluator behavior remains unchanged

### 4. New experiment config

Added:

- `scripts/eval/configs/habitat_dual_system_32ep_single_vllm_cfg.py`

This is a dedicated config for the single-vLLM experiment and does not modify the previous configs.

## Runtime Issues Encountered And Fixed

During bring-up, several issues were found and fixed:

1. Python 3.9 incompatibility in Habitat env
   - new files initially used `A | B` annotations
   - Habitat env is Python 3.9
   - fixed by switching to Python 3.9-compatible typing

2. Habitat import failure when `vllm` is absent locally
   - the evaluator only needs the HTTP client, not local vLLM
   - fixed by moving server-only `vllm` imports to lazy imports

3. vLLM chat multimodal format mismatch
   - internal messages used `{"type":"image"}`
   - vLLM expects `image_pil` / `image_url`
   - fixed by converting internal messages before `LLM.chat(...)`

4. Missing vLLM forward context in custom `apply_model(...)` forward
   - fixed by wrapping custom forward with:
     - `set_current_vllm_config(...)`
     - `set_forward_context(...)`

5. S1-only runner returned wrong trajectory tensor shape
   - incorrectly returned `[B, N, T, 3]`
   - original path returns `[B * N, T, 3]`
   - fixed to match original `generate_traj()`

## Launch Commands

### Single-vLLM server

```bash
cd /root/backup/InternNav
source /root/.venv/bin/activate

python scripts/eval/tools/serve_dualvln_single_vllm.py \
  --model-path checkpoints/InternVLA-N1-DualVLN-qwen25vl-s2-view \
  --hf-model-path checkpoints/InternVLA-N1-DualVLN \
  --served-model-name dualvln-single-vllm \
  --port 8000 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --limit-mm-per-prompt-image 16 \
  --gpu-memory-utilization 0.35 \
  --seed 0 \
  --enforce-eager
```

### Habitat eval

```bash
cd /root/backup/InternNav
source /root/miniforge3/etc/profile.d/conda.sh
conda activate habitat

python scripts/eval/eval.py \
  --config scripts/eval/configs/habitat_dual_system_32ep_single_vllm_cfg.py
```

## 32-Episode Comparison

Comparison baseline:

- `/root/InternNav/logs/habitat/test_dual_system_32ep_base/progress.json`

Comparison experiment:

- `/root/backup/InternNav/logs/habitat/test_dual_system_32ep_single_vllm/progress.json`

### Aggregate metrics

| Setting | Success | SPL | Oracle Success | NE | Avg Steps |
|---|---:|---:|---:|---:|---:|
| Original DualVLN (no vLLM) | 0.65625 | 0.58234 | 0.78125 | 3.99963 | 81.0 |
| Single-vLLM prototype | 0.46875 | 0.37063 | 0.53125 | 5.97884 | 105.0 |

Absolute delta of single-vLLM prototype vs original:

- Success: `-0.1875`
- SPL: `-0.21171`
- Oracle success: `-0.25`
- NE: `+1.97921`
- Avg steps: `+24.0`

### Success flips

There are **12** episodes where success changed.

Regression (`success: 1 -> 0`):

- `16`
- `43`
- `63`
- `71`
- `76`
- `78`
- `115`
- `122`
- `140`

Improvement (`success: 0 -> 1`):

- `61`
- `62`
- `116`

### Notable failure patterns

Observed patterns from the current single-vLLM results:

- several previously successful episodes become large-NE failures
- some episodes terminate much later than baseline
- at least one episode (`76`) hits `501` steps, indicating loop / drift behavior
- some successes remain successes but NE becomes noticeably worse

## Current Interpretation

The current single-vLLM prototype is **functionally running**, but it is **not yet behaviorally faithful** to the original DualVLN pipeline.

The most likely reason is not the evaluator wiring anymore, but **numerical mismatch in the custom single-engine `generate_latents` path**.

Why this is the leading hypothesis:

- the default/original full-model baseline is known to work
- the S1-only runner is now structurally aligned with original `generate_traj()`
- the major novel component in this prototype is the custom hidden-state extraction on top of vLLM `apply_model(...)`
- the quality degradation is large enough to suggest latent / hidden-state mismatch, not only harmless decoding noise

This interpretation is an inference from the current results, not yet a strict proof.

## Recommended Next Step

Before trusting any more closed-loop numbers, do **offline sample-level equivalence checks**:

1. capture one or more Habitat replay samples
2. run original HF `generate_latents()`
3. run single-vLLM `generate_latents()` on the same sample
4. compare:
   - prompt token ids
   - image token alignment
   - RoPE / position ids
   - final last-4 hidden states / latents
5. only after this is numerically close should the 32-episode closed-loop be treated as meaningful

In short:

- the single-vLLM control path is now wired end-to-end
- but the current closed-loop quality shows that this prototype is still a research/debug build, not a faithful replacement for the original DualVLN inference path

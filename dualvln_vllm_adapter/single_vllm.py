import os
from typing import Optional
import time

from dualvln_miniengine.contracts import DualVLNMiniEngineRequest
from dualvln_vllm_adapter.model_exec import (
    TRAJ_TOKEN_INDEX,
    build_native_latent_prefill_prompt_embeds as _build_native_latent_prefill_prompt_embeds,
    build_native_latent_prefill_suffix_prompt_embeds as _build_native_latent_prefill_suffix_prompt_embeds,
    compute_qwen2_5_vl_rope_index as _compute_qwen2_5_vl_rope_index,
    generate_latents_from_vllm_model as _generate_latents_from_vllm_model,
    generate_latents_via_transformers_backend_apply_model as _generate_latents_via_transformers_backend_apply_model,
    inspect_transformers_backend_model_tree as _inspect_transformers_backend_model_tree,
    load_latent_queries_tensor as _load_latent_queries_tensor,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

class DualVLNSingleVLLMRunner:
    def __init__(
        self,
        model_path: str,
        *,
        hf_model_path: Optional[str] = None,
        dtype: str = "auto",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.45,
        limit_mm_per_prompt_image: int = 16,
        tensor_parallel_size: int = 1,
        model_impl: str = "auto",
        latent_backend: Optional[str] = None,
        trust_remote_code: bool = False,
        enforce_eager: bool = False,
        seed: int = 0,
        compilation_config: Optional[dict] = None,
        quantization: Optional[str] = None,
    ):
        from dualvln_miniengine.backends import PatchedVLLMDualVLNMiniEngine

        self.engine = PatchedVLLMDualVLNMiniEngine.from_pretrained(
            model_path=model_path,
            hf_model_path=hf_model_path,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt_image=limit_mm_per_prompt_image,
            tensor_parallel_size=tensor_parallel_size,
            model_impl=model_impl,
            latent_backend=latent_backend,
            trust_remote_code=trust_remote_code,
            enforce_eager=enforce_eager,
            seed=seed,
            compilation_config=compilation_config,
            quantization=quantization,
        )
        self.llm = self.engine.llm
        self.processor = self.engine.processor
        self.latent_queries = self.engine.latent_queries
        self.n_query = self.engine.n_query
        self.traj_token_index = self.engine.traj_token_index
        self.latent_backend = self.engine.latent_backend

    def step_s2(self, messages, *, max_new_tokens: int = 128, return_latents: bool = True):
        step_result = self.engine.step_s2(
            DualVLNMiniEngineRequest(
                external_request_id=f"dualvln-adapter-{time.time_ns()}",
                messages=messages,
                max_new_tokens=max_new_tokens,
                return_latents=return_latents,
                latent_query_count=self.n_query,
            )
        )
        generate = step_result.generate
        latents_result = step_result.latents
        return {
            "llm_output": generate.llm_output,
            "prompt_token_ids": generate.prompt_token_ids,
            "generated_token_ids": generate.generated_token_ids,
            "pixel_goal": generate.pixel_goal,
            "latents": None if latents_result is None else latents_result.latents,
            "runtime_metrics": (
                generate.runtime_metrics
                if latents_result is None
                else latents_result.runtime_metrics
            ),
            "vllm_kv_cache": step_result.vllm_kv_cache,
            "debug_mm": step_result.debug_mm,
        }

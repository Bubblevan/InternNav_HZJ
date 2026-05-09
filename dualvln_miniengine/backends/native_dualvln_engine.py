from __future__ import annotations

import os
import time
from typing import Optional

import torch
from transformers import AutoProcessor
import transformers

from dualvln_model_adapters import create_model_adapter
from dualvln_model_adapters.base import VLNModelAdapter
from dualvln_miniengine.backends.base import DualVLNMiniEngineBackend
from dualvln_miniengine.contracts import (
    DualVLNMiniEngineGenerateResult,
    DualVLNMiniEngineLatentsResult,
    DualVLNMiniEngineRequest,
)
from dualvln_miniengine.engine import StatefulDualVLNMiniEngine
from dualvln_vllm_adapter.model_exec import normalize_image_grid_thw
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM


def _resolve_torch_dtype(dtype: str) -> torch.dtype:
    if dtype in {"auto", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"float16", "half"}:
        return torch.float16
    if dtype in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype for native backend: {dtype}")


def _runtime_versions() -> dict[str, str]:
    return {
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
    }


def _is_known_good_native_runtime() -> bool:
    versions = _runtime_versions()
    torch_version = versions["torch"]
    transformers_version = versions["transformers"]
    return torch_version.startswith("2.6.") and transformers_version.startswith("4.51.")


def _validate_native_runtime() -> None:
    if _is_known_good_native_runtime():
        return
    if os.getenv("DUALVLN_NATIVE_ALLOW_UNVERIFIED_RUNTIME", "0") == "1":
        return
    versions = _runtime_versions()
    raise RuntimeError(
        "Native DualVLN backend semantic drift was reproduced under the current "
        f"runtime stack (torch={versions['torch']}, transformers={versions['transformers']}). "
        "Use the validated habitat runtime (torch 2.6.x + transformers 4.51.x), "
        "or set DUALVLN_NATIVE_ALLOW_UNVERIFIED_RUNTIME=1 to bypass this guard."
    )


class NativeDualVLNBackend(DualVLNMiniEngineBackend):
    backend_name = "native_dualvln"

    def __init__(
        self,
        *,
        model_path: str,
        processor,
        model,
        device: torch.device,
        dtype: torch.dtype,
        attn_implementation: Optional[str],
        generation_impl: str,
        processor_use_fast: bool,
        enable_same_request_continuation: bool,
        model_adapter: VLNModelAdapter,
    ) -> None:
        super().__init__(model_adapter=model_adapter)
        self.model_path = model_path
        self.processor = processor
        self.model = model
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.generation_impl = generation_impl
        self.processor_use_fast = bool(processor_use_fast)
        self.enable_same_request_continuation = bool(enable_same_request_continuation)
        self.traj_token_index = int(self.model_adapter.traj_token_index)
        self._request_counter = 0

    @property
    def n_query(self) -> int:
        return int(self.model_adapter.get_latent_query_count(self.model))

    @property
    def capabilities(self) -> dict[str, bool]:
        return {
            "text_generate": True,
            "hidden_states": True,
            "latent_path": True,
            "same_request_continuation": bool(
                self.enable_same_request_continuation
            ),
            "latent_prefill_reuse": bool(self.enable_same_request_continuation),
        }

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        attn_implementation: Optional[str] = None,
        generation_impl: str = "hf_generate",
        processor_use_fast: bool = True,
        enable_same_request_continuation: bool = False,
        device: Optional[str] = None,
        model_adapter: str | VLNModelAdapter = "dualvln",
    ) -> "NativeDualVLNBackend":
        _validate_native_runtime()
        resolved_adapter = (
            create_model_adapter(model_adapter)
            if isinstance(model_adapter, str)
            else model_adapter
        )
        torch_dtype = _resolve_torch_dtype(dtype)
        resolved_device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            use_fast=processor_use_fast,
        )
        processor.tokenizer.padding_side = "left"
        attn_impl = attn_implementation
        if attn_impl is None and resolved_device.type == "cuda":
            attn_impl = "sdpa"
        model = InternVLAN1ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        )
        model.to(device=resolved_device, dtype=torch_dtype)
        model.eval()
        return cls(
            model_path=model_path,
            processor=processor,
            model=model,
            device=resolved_device,
            dtype=torch_dtype,
            attn_implementation=attn_impl,
            generation_impl=generation_impl,
            processor_use_fast=processor_use_fast,
            enable_same_request_continuation=enable_same_request_continuation,
            model_adapter=resolved_adapter,
        )

    def _next_request_id(self) -> str:
        request_id = f"native-{self._request_counter}"
        self._request_counter += 1
        return request_id

    @staticmethod
    def _sanitize_model_inputs(model_inputs):
        if hasattr(model_inputs, "items"):
            return {
                key: value
                for key, value in model_inputs.items()
                if key not in {"mm_token_type_ids"}
            }
        return model_inputs

    def get_runtime_stats(self) -> dict[str, object] | None:
        memory_allocated = None
        memory_reserved = None
        if self.device.type == "cuda":
            memory_allocated = int(torch.cuda.memory_allocated(self.device))
            memory_reserved = int(torch.cuda.memory_reserved(self.device))
        return {
            "backend_name": self.backend_name,
            "model_adapter": self.model_adapter.describe(),
            "device": str(self.device),
            "dtype": str(self.dtype),
            "attn_implementation": self.attn_implementation,
            "generation_impl": self.generation_impl,
            "processor_use_fast": self.processor_use_fast,
            "enable_same_request_continuation": self.enable_same_request_continuation,
            "model_path": self.model_path,
            "n_query": int(self.n_query),
            "runtime_versions": _runtime_versions(),
            "runtime_stack_validated": _is_known_good_native_runtime(),
            "cuda_memory_allocated_bytes": memory_allocated,
            "cuda_memory_reserved_bytes": memory_reserved,
        }

    def _build_full_forward_context(
        self,
        *,
        output_ids: torch.Tensor,
        model_inputs,
        capture_hidden_states: bool,
    ) -> tuple[object, Optional[torch.Tensor], float]:
        image_grid_thw = normalize_image_grid_thw(
            getattr(model_inputs, "image_grid_thw", None)
        )
        context_start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(
                input_ids=output_ids,
                pixel_values=getattr(model_inputs, "pixel_values", None),
                image_grid_thw=image_grid_thw,
                use_cache=True,
                output_hidden_states=bool(capture_hidden_states),
                return_dict=True,
            )
        context_ms = (time.perf_counter() - context_start) * 1000.0
        hidden_states = None
        if capture_hidden_states and getattr(outputs, "hidden_states", None) is not None:
            hidden_states = outputs.hidden_states[-1].detach().cpu()
        return outputs, hidden_states, context_ms

    def _extract_latents_via_same_request_continuation(
        self,
        *,
        past_key_values,
        output_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        continuation_start = time.perf_counter()
        seq_len = int(output_ids.shape[1])
        suffix_ids = torch.full(
            (1, self.n_query),
            self.traj_token_index,
            dtype=torch.long,
            device=output_ids.device,
        )
        cache_position = torch.arange(
            seq_len,
            seq_len + self.n_query,
            dtype=torch.long,
            device=output_ids.device,
        )
        with torch.inference_mode():
            outputs = self.model(
                input_ids=suffix_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                cache_position=cache_position,
            )
        continuation_ms = (time.perf_counter() - continuation_start) * 1000.0
        latents = outputs.hidden_states[-1][:, -self.n_query :, :].detach().cpu()
        return latents, continuation_ms

    def _is_eos_token_id(self, token_id: int) -> bool:
        eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        if eos_token_id is None:
            return False
        if isinstance(eos_token_id, (list, tuple, set)):
            return int(token_id) in {int(x) for x in eos_token_id}
        return int(token_id) == int(eos_token_id)

    def _custom_greedy_generate(
        self,
        *,
        model_inputs,
        max_new_tokens: int,
    ) -> dict[str, object]:
        input_ids = model_inputs.input_ids
        attention_mask = getattr(model_inputs, "attention_mask", None)
        pixel_values = getattr(model_inputs, "pixel_values", None)
        image_grid_thw = normalize_image_grid_thw(
            getattr(model_inputs, "image_grid_thw", None)
        )

        sequences = input_ids.clone()
        generated_tokens: list[torch.Tensor] = []
        prefill_start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
        prefill_ms = (time.perf_counter() - prefill_start) * 1000.0

        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        decode_ms = 0.0
        decode_steps = 0
        cur_len = int(sequences.shape[1])

        while len(generated_tokens) < int(max_new_tokens):
            generated_tokens.append(next_token[:, None])
            sequences = torch.cat([sequences, next_token[:, None]], dim=1)
            token_id = int(next_token[0].item())
            if self._is_eos_token_id(token_id):
                break
            if len(generated_tokens) >= int(max_new_tokens):
                break

            decode_start = time.perf_counter()
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=next_token[:, None],
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=False,
                    return_dict=True,
                    cache_position=torch.tensor(
                        [cur_len],
                        dtype=torch.long,
                        device=next_token.device,
                    ),
                )
            decode_ms += (time.perf_counter() - decode_start) * 1000.0
            decode_steps += 1
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            cur_len += 1

        return {
            "sequences": sequences,
            "past_key_values": past_key_values,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "decode_steps": decode_steps,
            "generation_impl": "native_custom_greedy",
        }

    def generate_text(
        self,
        request: DualVLNMiniEngineRequest,
    ) -> DualVLNMiniEngineGenerateResult:
        preprocess_start = time.perf_counter()
        hf_inputs = self.model_adapter.build_generation_inputs(
            self.processor,
            request.messages,
            device=self.device,
        )
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
        model_inputs = hf_inputs["model_inputs"]
        sanitized_model_inputs = self._sanitize_model_inputs(model_inputs)
        preprocess_timings = dict(hf_inputs.get("timings") or {})
        prompt_build_ms = float(preprocess_timings.get("prompt_build_ms", 0.0) or 0.0)
        processor_ms = float(preprocess_timings.get("processor_ms", 0.0) or 0.0)
        to_device_ms = float(preprocess_timings.get("to_device_ms", 0.0) or 0.0)
        image_collect_ms = float(preprocess_timings.get("image_collect_ms", 0.0) or 0.0)

        generate_prefill_ms = None
        generate_decode_ms = None
        generate_decode_steps = None
        generation_impl = self.generation_impl
        if self.generation_impl == "hf_generate" or request.return_hidden_states:
            generate_start = time.perf_counter()
            with torch.inference_mode():
                generate_output = self.model.generate(
                    **sanitized_model_inputs,
                    max_new_tokens=request.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    past_key_values=None,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                )
            generate_ms = (time.perf_counter() - generate_start) * 1000.0
            output_ids = generate_output.sequences
            continuation_past_key_values = getattr(generate_output, "past_key_values", None)
        else:
            try:
                generate_start = time.perf_counter()
                custom_generate = self._custom_greedy_generate(
                    model_inputs=model_inputs,
                    max_new_tokens=request.max_new_tokens,
                )
                generate_ms = (time.perf_counter() - generate_start) * 1000.0
                output_ids = custom_generate["sequences"]
                continuation_past_key_values = custom_generate["past_key_values"]
                generate_prefill_ms = float(custom_generate["prefill_ms"])
                generate_decode_ms = float(custom_generate["decode_ms"])
                generate_decode_steps = int(custom_generate["decode_steps"])
                generation_impl = str(custom_generate["generation_impl"])
            except Exception as exc:
                generate_start = time.perf_counter()
                with torch.inference_mode():
                    generate_output = self.model.generate(
                        **sanitized_model_inputs,
                        max_new_tokens=request.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        past_key_values=None,
                        return_dict_in_generate=True,
                        output_hidden_states=False,
                    )
                generate_ms = (time.perf_counter() - generate_start) * 1000.0
                output_ids = generate_output.sequences
                continuation_past_key_values = getattr(generate_output, "past_key_values", None)
                generation_impl = f"hf_generate_fallback:{type(exc).__name__}"

        prompt_len = int(model_inputs.input_ids.shape[1])
        prompt_token_ids = model_inputs.input_ids[0].detach().cpu().tolist()
        generated_token_ids = output_ids[0][prompt_len:].detach().cpu().tolist()
        output_parse_start = time.perf_counter()
        llm_output = self.processor.tokenizer.decode(
            output_ids[0][prompt_len:],
            skip_special_tokens=True,
        )
        planner_output = self.model_adapter.parse_planner_output(llm_output)
        output_parse_ms = (time.perf_counter() - output_parse_start) * 1000.0
        pixel_goal = planner_output.pixel_goal

        hidden_states = None
        hidden_states_shape = None
        hidden_states_ms = 0.0
        continuation_prepare_ms = 0.0
        continuation_cache_source = (
            "generate_past_key_values"
            if continuation_past_key_values is not None
            else "missing_generate_past_key_values"
        )
        if request.return_hidden_states:
            _, hidden_states, continuation_prepare_ms = (
                self._build_full_forward_context(
                    output_ids=output_ids,
                    model_inputs=model_inputs,
                    capture_hidden_states=request.return_hidden_states,
                )
            )
            continuation_cache_source = (
                "generate_past_key_values+full_forward_hidden_states"
                if continuation_past_key_values is not None
                else "full_forward_hidden_states_only"
            )
        if request.return_hidden_states and hidden_states is not None:
            hidden_states_shape = list(hidden_states.shape)
            hidden_states_ms = continuation_prepare_ms

        internal_request_id = self._next_request_id()
        runtime_metrics = {
            "preprocess_ms": preprocess_ms,
            "image_collect_ms": image_collect_ms,
            "prompt_build_ms": prompt_build_ms,
            "processor_ms": processor_ms,
            "to_device_ms": to_device_ms,
            "generate_ms": generate_ms,
            "generate_prefill_ms": generate_prefill_ms,
            "generate_decode_ms": generate_decode_ms,
            "generate_driver_overhead_ms": (
                float(max(generate_ms - (generate_prefill_ms or 0.0) - (generate_decode_ms or 0.0), 0.0))
                if (generate_prefill_ms is not None or generate_decode_ms is not None)
                else None
            ),
            "generate_decode_steps": generate_decode_steps,
            "generation_impl": generation_impl,
            "bundle_build_ms": 0.0,
            "mm_attach_ms": 0.0,
            "latent_prefill_ms": 0.0,
            "output_parse_ms": output_parse_ms,
            "prompt_token_count": int(len(prompt_token_ids)),
            "generated_token_count": int(len(generated_token_ids)),
            "prefill_token_count": None,
            "n_query": int(self.n_query),
            "num_images": int(len(hf_inputs["input_images"])),
            "mm_feature_count": int(len(hf_inputs["input_images"])),
            "same_request_continuation_enabled": bool(
                self.enable_same_request_continuation
            ),
            "same_request_continuation_attempted": False,
            "same_request_continuation_used": False,
            "same_request_suffix_len": int(self.n_query),
            "same_request_external_request_id": request.external_request_id,
            "same_request_internal_request_id": internal_request_id,
            "same_request_request_ids_match": True,
            "same_request_fallback_reason": None,
            "same_request_result_missing": False,
            "reused_prefill": False,
            "latent_path": "native_continuation_candidate" if pixel_goal is not None else "hf_generate_latents",
            "continuation_prepare_ms": continuation_prepare_ms if request.return_hidden_states else None,
            "hidden_states_ms": hidden_states_ms if request.return_hidden_states else None,
            "hidden_states_shape": hidden_states_shape,
            "continuation_cache_source": continuation_cache_source,
            "full_forward_hidden_states_required": bool(request.return_hidden_states),
        }
        return DualVLNMiniEngineGenerateResult(
            external_request_id=request.external_request_id,
            internal_request_id=internal_request_id,
            llm_output=llm_output,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
            pixel_goal=pixel_goal,
            mm_feature_count=int(len(hf_inputs["input_images"])),
            runtime_metrics=runtime_metrics,
            metadata={
                "hf_inputs": hf_inputs,
                "output_ids": output_ids,
                "hidden_states": hidden_states,
                "continuation_past_key_values": continuation_past_key_values,
                "planner_output": planner_output,
            },
        )

    def extract_latents(
        self,
        request: DualVLNMiniEngineRequest,
        generate_result: DualVLNMiniEngineGenerateResult,
    ) -> DualVLNMiniEngineLatentsResult | None:
        hf_inputs = generate_result.metadata["hf_inputs"]
        output_ids = generate_result.metadata["output_ids"]
        runtime_metrics = dict(generate_result.runtime_metrics)
        continuation_past_key_values = generate_result.metadata.get("continuation_past_key_values")
        fallback_reason = None
        mode = "native_same_request_continuation"
        reused_prefill = False
        same_request_continuation_used = False
        runtime_metrics["same_request_continuation_attempted"] = bool(
            self.enable_same_request_continuation
        )
        try:
            if not self.enable_same_request_continuation:
                raise RuntimeError("same_request_continuation_disabled")
            if continuation_past_key_values is None:
                raise RuntimeError("missing_continuation_cache")
            latents, latent_prefill_ms = self._extract_latents_via_same_request_continuation(
                past_key_values=continuation_past_key_values,
                output_ids=output_ids,
            )
            runtime_metrics["same_request_continuation_used"] = True
            runtime_metrics["same_request_fallback_reason"] = None
            runtime_metrics["latent_path"] = "native_same_request_continuation"
            reused_prefill = True
            same_request_continuation_used = True
        except Exception as exc:
            fallback_reason = f"native_same_request_continuation_failed:{type(exc).__name__}:{exc}"
            runtime_metrics["same_request_fallback_reason"] = fallback_reason
            runtime_metrics["same_request_result_missing"] = True
            latent_start = time.perf_counter()
            with torch.inference_mode():
                latents = self.model.generate_latents(
                    output_ids,
                    getattr(hf_inputs["model_inputs"], "pixel_values"),
                    hf_inputs["image_grid_thw"].to(output_ids.device),
                )
            latent_prefill_ms = (time.perf_counter() - latent_start) * 1000.0
            runtime_metrics["latent_path"] = "hf_generate_latents_fallback"
            mode = "hf_generate_latents_fallback"
        runtime_metrics["latent_prefill_ms"] = latent_prefill_ms
        runtime_metrics["prefill_token_count"] = int(output_ids.shape[1] + self.n_query)
        runtime_metrics["reused_prefill"] = bool(reused_prefill)
        return DualVLNMiniEngineLatentsResult(
            external_request_id=request.external_request_id,
            internal_request_id=generate_result.internal_request_id,
            latents=latents.detach().cpu(),
            prefill_token_ids=output_ids[0].detach().cpu().tolist()
            + [self.traj_token_index] * self.n_query,
            latent_query_count=int(self.n_query),
            mode=mode,
            reused_prefill=reused_prefill,
            same_request_continuation_used=same_request_continuation_used,
            fallback_reason=fallback_reason,
            runtime_metrics=runtime_metrics,
            metadata={
                "hidden_states_shape": generate_result.runtime_metrics.get("hidden_states_shape"),
            },
        )


class NativeDualVLNMiniEngine(StatefulDualVLNMiniEngine):
    def __init__(self, backend: NativeDualVLNBackend) -> None:
        super().__init__(backend)

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "NativeDualVLNMiniEngine":
        return cls(NativeDualVLNBackend.from_pretrained(*args, **kwargs))

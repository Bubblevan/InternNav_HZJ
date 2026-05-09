from __future__ import annotations

import functools
import json
import logging
import os
import time
from typing import Optional

from transformers import AutoProcessor

from dualvln_model_adapters import create_model_adapter
from dualvln_model_adapters.base import VLNModelAdapter
from dualvln_runtime.protocol import extract_images_from_messages, to_vllm_chat_messages
from dualvln_vllm_adapter.hidden_latents import VLLMHiddenLatentsRunner
from dualvln_vllm_adapter.latents_request import (
    attach_explicit_mm_metadata,
    attach_explicit_mm_metadata_from_engine_core_request,
    attach_explicit_mm_metadata_from_processed_inputs,
    build_latents_request_bundle,
)
from dualvln_vllm_adapter.model_exec import (
    build_native_latent_prefill_prompt_embeds,
    build_native_latent_prefill_suffix_prompt_embeds,
    generate_latents_via_transformers_backend_apply_model,
)
from dualvln_miniengine.contracts import (
    DualVLNMiniEngineGenerateResult,
    DualVLNMiniEngineLatentsResult,
    DualVLNMiniEngineRequest,
)
from dualvln_miniengine.engine import StatefulDualVLNMiniEngine
from dualvln_miniengine.backends.base import DualVLNMiniEngineBackend

logger = logging.getLogger(__name__)


def _collect_step_s2_mm_debug(processed_prompt, bundle) -> dict:
    mm_placeholders = processed_prompt.get("mm_placeholders") or {}
    mm_kwargs = processed_prompt.get("mm_kwargs") or {}
    image_placeholders = mm_placeholders.get("image") or []
    image_kwargs = mm_kwargs.get("image") or []
    none_mm_kwargs = sum(item is None for item in image_kwargs)

    mm_features = bundle.mm_features
    mm_features_len = len(mm_features) if mm_features is not None else None
    mm_features_type = type(mm_features).__name__
    mm_features_num_data_none = (
        sum(getattr(feature, "data", None) is None for feature in mm_features)
        if mm_features is not None
        else None
    )

    feature0_offset = None
    feature0_length = None
    if mm_features:
        feature0_offset = int(mm_features[0].mm_position.offset)
        feature0_length = int(mm_features[0].mm_position.length)

    return {
        "processed_prompt_type": processed_prompt.get("type"),
        "processed_prompt_num_image_placeholders": len(image_placeholders),
        "processed_prompt_num_image_kwargs": len(image_kwargs),
        "processed_prompt_num_image_kwargs_none": none_mm_kwargs,
        "bundle_mm_features_is_none": mm_features is None,
        "bundle_mm_features_len": mm_features_len,
        "bundle_mm_features_type": mm_features_type,
        "bundle_mm_features_num_data_none": mm_features_num_data_none,
        "bundle_mm_feature0_offset": feature0_offset,
        "bundle_mm_feature0_length": feature0_length,
    }


def _bundle_mm_features_have_missing_payload(bundle) -> bool:
    mm_features = getattr(bundle, "mm_features", None)
    if not mm_features:
        return False
    return any(getattr(feature, "data", None) is None for feature in mm_features)


def _aggregate_vllm_worker_runtime_stats(worker_reports) -> dict:
    reports = [report for report in (worker_reports or []) if isinstance(report, dict)]
    if not reports:
        return {
            "available_kv_cache_memory_bytes": None,
            "requested_memory_bytes": None,
            "peak_activation_memory_bytes": None,
            "cudagraph_memory_estimate_bytes": None,
            "effective_kv_budget_bytes": None,
            "gpu_memory_utilization": None,
            "max_model_len": None,
            "num_gpu_blocks": None,
            "worker_count": 0,
        }

    def _sum_optional_int(key: str):
        values = [report.get(key) for report in reports if report.get(key) is not None]
        return int(sum(values)) if values else None

    head = reports[0]
    return {
        "available_kv_cache_memory_bytes": _sum_optional_int("available_kv_cache_memory_bytes"),
        "requested_memory_bytes": _sum_optional_int("requested_memory_bytes"),
        "peak_activation_memory_bytes": _sum_optional_int("peak_activation_memory_bytes"),
        "cudagraph_memory_estimate_bytes": _sum_optional_int("cudagraph_memory_estimate_bytes"),
        "effective_kv_budget_bytes": _sum_optional_int("effective_kv_budget_bytes"),
        "gpu_memory_utilization": head.get("gpu_memory_utilization"),
        "max_model_len": head.get("max_model_len"),
        "num_gpu_blocks": _sum_optional_int("num_gpu_blocks"),
        "worker_count": len(reports),
    }


def _make_runtime_stats_error_payload(exc: Exception) -> dict:
    return {
        "available_kv_cache_memory_bytes": None,
        "requested_memory_bytes": None,
        "peak_activation_memory_bytes": None,
        "cudagraph_memory_estimate_bytes": None,
        "effective_kv_budget_bytes": None,
        "gpu_memory_utilization": None,
        "max_model_len": None,
        "num_gpu_blocks": None,
        "worker_count": 0,
        "collect_error": repr(exc),
    }


def _extract_request_encoder_timing_ms(
    worker_reports,
    request_id: str | None,
) -> tuple[float | None, int | None]:
    if not request_id:
        return None, None

    total_secs = 0.0
    total_calls = 0
    found = False
    for report in worker_reports or []:
        if not isinstance(report, dict):
            continue
        req_stats = report.get(request_id)
        if not isinstance(req_stats, dict):
            continue
        secs = req_stats.get("encoder_forward_secs")
        calls = req_stats.get("num_encoder_calls")
        if secs is not None:
            total_secs += float(secs)
            found = True
        if calls is not None:
            total_calls += int(calls)
            found = True

    if not found:
        return None, None
    return total_secs * 1000.0, total_calls


def _extract_request_stage_timing_ms(
    worker_reports,
    request_id: str | None,
) -> dict[str, float | int | None]:
    fields = (
        "prefill_forward_secs",
        "prefill_forward_calls",
        "prefill_sample_secs",
        "prefill_sample_calls",
        "decode_forward_secs",
        "decode_forward_calls",
        "decode_sample_secs",
        "decode_sample_calls",
        "extend_forward_secs",
        "extend_forward_calls",
        "extend_sample_secs",
        "extend_sample_calls",
    )
    result: dict[str, float | int | None] = {field: None for field in fields}
    if not request_id:
        return result

    found = False
    aggregates: dict[str, float | int] = {field: 0 for field in fields}
    for report in worker_reports or []:
        if not isinstance(report, dict):
            continue
        req_stats = report.get(request_id)
        if not isinstance(req_stats, dict):
            continue
        for field in fields:
            value = req_stats.get(field)
            if value is None:
                continue
            found = True
            aggregates[field] += value

    if not found:
        return result

    for field in fields:
        value = aggregates[field]
        if field.endswith("_secs"):
            result[field] = float(value) * 1000.0
        else:
            result[field] = int(value)
    return result


class PatchedVLLMDualVLNBackend(DualVLNMiniEngineBackend):
    backend_name = "patched_vllm"

    def __init__(
        self,
        *,
        llm,
        processor,
        latent_queries,
        n_query: int,
        traj_token_index: int,
        latent_backend: str,
        hidden_latents_runner_kwargs: dict,
        model_adapter: VLNModelAdapter,
    ) -> None:
        super().__init__(model_adapter=model_adapter)
        self.llm = llm
        self.processor = processor
        self.latent_queries = latent_queries
        self._n_query = int(n_query)
        self.traj_token_index = int(traj_token_index)
        self.latent_backend = latent_backend
        self._hidden_latents_runner = None
        self._hidden_latents_runner_kwargs = hidden_latents_runner_kwargs
        self._last_step_s2_engine_request = None
        self._runtime_stats_refresh_mode = os.environ.get(
            "INTERNNAV_VLLM_RUNTIME_STATS_REFRESH_MODE",
            "init_once",
        )
        self._cached_vllm_runtime_stats = self._fetch_vllm_runtime_stats_once()

    @property
    def n_query(self) -> int:
        return int(self._n_query)

    @property
    def capabilities(self) -> dict[str, bool]:
        return {
            "text_generate": True,
            "hidden_states": True,
            "latent_path": True,
            "same_request_continuation": True,
            "latent_prefill_reuse": True,
        }

    @classmethod
    def from_pretrained(
        cls,
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
        model_adapter: str | VLNModelAdapter = "dualvln",
    ) -> "PatchedVLLMDualVLNBackend":
        from vllm import LLM

        resolved_adapter = (
            create_model_adapter(model_adapter)
            if isinstance(model_adapter, str)
            else model_adapter
        )
        resolved_hf_model_path = hf_model_path or model_path
        processor = AutoProcessor.from_pretrained(
            resolved_hf_model_path,
            trust_remote_code=trust_remote_code,
        )
        processor.tokenizer.padding_side = "left"
        latent_queries = resolved_adapter.load_latent_queries(resolved_hf_model_path)
        n_query = int(latent_queries.shape[0])
        requested_latent_backend = latent_backend or "shared_engine_forward"
        if requested_latent_backend == "vllm_hidden":
            requested_latent_backend = "shared_engine_forward"

        llm_kwargs = dict(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_model_len=max_model_len,
            compilation_config=compilation_config,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": limit_mm_per_prompt_image},
            model_impl=model_impl,
            trust_remote_code=trust_remote_code,
            enforce_eager=enforce_eager,
            seed=seed,
            disable_log_stats=True,
            async_scheduling=False,
        )
        if quantization is not None:
            llm_kwargs["quantization"] = quantization
        llm = LLM(**llm_kwargs)
        hidden_latents_runner_kwargs = {
            "model_path": model_path,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "limit_mm_per_prompt_image": limit_mm_per_prompt_image,
            "dtype": dtype,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": trust_remote_code,
            "enforce_eager": enforce_eager,
        }
        return cls(
            llm=llm,
            processor=processor,
            latent_queries=latent_queries,
            n_query=n_query,
            traj_token_index=resolved_adapter.traj_token_index,
            latent_backend=requested_latent_backend,
            hidden_latents_runner_kwargs=hidden_latents_runner_kwargs,
            model_adapter=resolved_adapter,
        )

    def _fetch_vllm_runtime_stats_once(self) -> dict:
        try:
            worker_reports = self.llm.collective_rpc("get_internnav_runtime_stats")
        except Exception as exc:
            return _make_runtime_stats_error_payload(exc)
        return _aggregate_vllm_worker_runtime_stats(worker_reports)

    def _get_cached_vllm_runtime_stats(self) -> dict:
        if self._runtime_stats_refresh_mode in {"manual", "init_once"}:
            return dict(self._cached_vllm_runtime_stats)
        if self._runtime_stats_refresh_mode == "lazy_once":
            if self._cached_vllm_runtime_stats is None:
                self._cached_vllm_runtime_stats = self._fetch_vllm_runtime_stats_once()
            return dict(self._cached_vllm_runtime_stats)
        raise ValueError(
            "Unsupported INTERNNAV_VLLM_RUNTIME_STATS_REFRESH_MODE: "
            f"{self._runtime_stats_refresh_mode}"
        )

    def _pop_request_encoder_timing_stats(
        self,
        request_id: str | None,
    ) -> tuple[float | None, int | None]:
        try:
            worker_reports = self.llm.collective_rpc("get_encoder_timing_stats")
        except Exception as exc:
            logger.warning(
                "Failed to collect encoder timing stats for request %s: %s",
                request_id,
                exc,
            )
            return None, None
        return _extract_request_encoder_timing_ms(worker_reports, request_id)

    def _pop_request_stage_timing_stats(
        self,
        request_id: str | None,
    ) -> dict[str, float | int | None]:
        try:
            worker_reports = self.llm.collective_rpc("get_request_timing_stats")
        except Exception as exc:
            logger.warning(
                "Failed to collect request timing stats for request %s: %s",
                request_id,
                exc,
            )
            return _extract_request_stage_timing_ms([], request_id)
        return _extract_request_stage_timing_ms(worker_reports, request_id)

    def get_runtime_stats(self) -> dict[str, object] | None:
        return self._get_cached_vllm_runtime_stats()

    def _generate_latents_via_shared_engine(self, bundle):
        prompt_embeds_mode = os.environ.get(
            "INTERNNAV_NATIVE_PREFILL_PROMPT_EMBEDS_MODE",
            "full_prompt",
        )
        prompt_embeds_soft_suffix_len = None

        if prompt_embeds_mode == "full_prompt":
            if (
                bundle.prompt_embeds is None
                or int(bundle.prompt_embeds.shape[0]) != len(bundle.prefill_token_ids)
            ):
                bundle.prompt_embeds = self.llm.apply_model(
                    functools.partial(
                        build_native_latent_prefill_prompt_embeds,
                        prompt_token_ids=bundle.prefill_token_ids,
                        latent_queries_cpu=bundle.latent_queries,
                        mm_features=bundle.mm_features,
                    )
                )[0]
        elif prompt_embeds_mode == "soft_suffix_only":
            if (
                bundle.prompt_embeds is None
                or int(bundle.prompt_embeds.shape[0]) != self.n_query
            ):
                bundle.prompt_embeds = self.llm.apply_model(
                    functools.partial(
                        build_native_latent_prefill_suffix_prompt_embeds,
                        latent_queries_cpu=bundle.latent_queries,
                    )
                )[0]
            prompt_embeds_soft_suffix_len = self.n_query
        else:
            raise ValueError(
                "Unsupported INTERNNAV_NATIVE_PREFILL_PROMPT_EMBEDS_MODE: "
                f"{prompt_embeds_mode}"
            )

        return self.llm.generate_latents_native_prefill(
            prompt_token_ids=bundle.prefill_token_ids,
            prompt_embeds=bundle.prompt_embeds,
            mm_features=bundle.mm_features,
            n_query=self.n_query,
            prompt_embeds_soft_suffix_len=prompt_embeds_soft_suffix_len,
        )[0]

    def _ensure_hidden_latents_runner(self):
        if self._hidden_latents_runner is None:
            self._hidden_latents_runner = VLLMHiddenLatentsRunner(
                **self._hidden_latents_runner_kwargs,
            )
        return self._hidden_latents_runner

    def generate_text(
        self,
        request: DualVLNMiniEngineRequest,
    ) -> DualVLNMiniEngineGenerateResult:
        from vllm import SamplingParams
        from vllm.outputs import RequestOutput

        preprocess_start = time.perf_counter()
        vllm_messages = to_vllm_chat_messages(request.messages)
        processed_prompt = self.llm._preprocess_chat_one(vllm_messages)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        generate_start = time.perf_counter()
        sampling_params = SamplingParams(max_tokens=request.max_new_tokens, temperature=0.0)
        continuation_enabled = (
            os.environ.get("INTERNNAV_DUALVLN_LATENT_CONTINUATION", "1") != "0"
        )
        continuation_backend_eligible = self.latent_backend in (
            "legacy_custom_forward",
            "shared_engine_forward",
        )
        continuation_attempted = continuation_enabled and continuation_backend_eligible
        continuation_result = None
        continuation_setup_error = None
        if continuation_attempted:
            try:
                outputs, continuation_result = (
                    self.llm._run_request_with_dualvln_latent_continuation(
                        prompt=processed_prompt,
                        params=sampling_params,
                        suffix_token_ids=[self.traj_token_index] * self.n_query,
                        use_tqdm=False,
                    )
                )
            except Exception as exc:
                continuation_setup_error = (
                    f"continuation_setup_error:{type(exc).__name__}:{exc}"
                )
                logger.warning(
                    "DualVLN step_s2 could not arm same-request latent continuation; "
                    "falling back to normal text generate path: %s",
                    continuation_setup_error,
                )
                outputs = self.llm._render_and_run_requests(
                    prompts=(processed_prompt,),
                    params=[sampling_params],
                    output_type=RequestOutput,
                    use_tqdm=False,
                )
        else:
            outputs = self.llm._render_and_run_requests(
                prompts=(processed_prompt,),
                params=[sampling_params],
                output_type=RequestOutput,
                use_tqdm=False,
            )
        generate_ms = (time.perf_counter() - generate_start) * 1000.0

        request_output = outputs[0]
        self._last_step_s2_engine_request = self.llm.llm_engine.pop_debug_engine_core_request(
            request_output.request_id
        )
        timing_request_id = (
            self._last_step_s2_engine_request.request_id
            if self._last_step_s2_engine_request is not None
            and getattr(self._last_step_s2_engine_request, "request_id", None)
            else request_output.request_id
        )
        completion = request_output.outputs[0]
        llm_output = completion.text
        prompt_token_ids = list(request_output.prompt_token_ids or [])
        generated_token_ids = list(completion.token_ids)
        input_images = extract_images_from_messages(request.messages)
        image_placeholders = (processed_prompt.get("mm_placeholders") or {}).get("image") or []
        vision_encode_ms, vision_encoder_calls = self._pop_request_encoder_timing_stats(
            timing_request_id
        )
        request_stage_timing = self._pop_request_stage_timing_stats(
            timing_request_id
        )
        llm_prefill_ms = None
        if (
            request_stage_timing.get("prefill_forward_secs") is not None
            or request_stage_timing.get("prefill_sample_secs") is not None
        ):
            llm_prefill_ms = float(
                (request_stage_timing.get("prefill_forward_secs") or 0.0)
                + (request_stage_timing.get("prefill_sample_secs") or 0.0)
            )
        llm_decode_ms = None
        if (
            request_stage_timing.get("decode_forward_secs") is not None
            or request_stage_timing.get("decode_sample_secs") is not None
        ):
            llm_decode_ms = float(
                (request_stage_timing.get("decode_forward_secs") or 0.0)
                + (request_stage_timing.get("decode_sample_secs") or 0.0)
            )
        llm_extend_ms = None
        if (
            request_stage_timing.get("extend_forward_secs") is not None
            or request_stage_timing.get("extend_sample_secs") is not None
        ):
            llm_extend_ms = float(
                (request_stage_timing.get("extend_forward_secs") or 0.0)
                + (request_stage_timing.get("extend_sample_secs") or 0.0)
            )
        generate_residual_ms = None
        if vision_encode_ms is not None:
            generate_residual_ms = max(generate_ms - vision_encode_ms, 0.0)

        runtime_metrics = {
            "preprocess_ms": preprocess_ms,
            "mm_processor_ms": preprocess_ms,
            "generate_ms": generate_ms,
            "vision_encode_ms": vision_encode_ms,
            "vision_encoder_calls": vision_encoder_calls,
            "llm_prefill_ms": llm_prefill_ms,
            "llm_prefill_forward_ms": request_stage_timing.get("prefill_forward_secs"),
            "llm_prefill_sample_ms": request_stage_timing.get("prefill_sample_secs"),
            "llm_prefill_forward_calls": request_stage_timing.get("prefill_forward_calls"),
            "llm_prefill_sample_calls": request_stage_timing.get("prefill_sample_calls"),
            "llm_decode_ms": llm_decode_ms,
            "llm_decode_forward_ms": request_stage_timing.get("decode_forward_secs"),
            "llm_decode_sample_ms": request_stage_timing.get("decode_sample_secs"),
            "llm_decode_forward_calls": request_stage_timing.get("decode_forward_calls"),
            "llm_decode_sample_calls": request_stage_timing.get("decode_sample_calls"),
            "llm_extend_ms": llm_extend_ms,
            "llm_extend_forward_ms": request_stage_timing.get("extend_forward_secs"),
            "llm_extend_sample_ms": request_stage_timing.get("extend_sample_secs"),
            "llm_extend_forward_calls": request_stage_timing.get("extend_forward_calls"),
            "llm_extend_sample_calls": request_stage_timing.get("extend_sample_calls"),
            "generate_residual_ms": generate_residual_ms,
            "bundle_build_ms": 0.0,
            "mm_attach_ms": 0.0,
            "latent_prefill_ms": 0.0,
            "total_ms": 0.0,
            "prompt_token_count": int(len(prompt_token_ids)),
            "generated_token_count": int(len(generated_token_ids)),
            "prefill_token_count": None,
            "n_query": int(self.n_query),
            "num_images": int(len(input_images)),
            "mm_feature_count": int(len(image_placeholders)),
            "prefill_share_of_total": None,
            "latent_prefill_share_of_total": None,
            "decode_share_of_total": None,
            "same_request_continuation_enabled": bool(continuation_enabled),
            "same_request_continuation_attempted": bool(continuation_attempted),
            "same_request_continuation_used": False,
            "same_request_suffix_len": int(self.n_query),
            "same_request_external_request_id": request.external_request_id,
            "same_request_internal_request_id": (
                self._last_step_s2_engine_request.request_id
                if self._last_step_s2_engine_request is not None
                else None
            ),
            "timing_lookup_request_id": timing_request_id,
            "same_request_request_ids_match": None,
            "same_request_fallback_reason": None,
            "same_request_result_missing": False,
            "reused_prefill": False,
            "latent_path": (
                "same_request_continuation_attempt"
                if continuation_attempted
                else "legacy_latent_path"
            ),
        }
        if continuation_result is not None:
            runtime_metrics["same_request_internal_request_id"] = (
                continuation_result.internal_request_id
            )
            runtime_metrics["same_request_request_ids_match"] = bool(
                continuation_result.external_request_id == request_output.request_id
            )
            runtime_metrics["same_request_fallback_reason"] = (
                continuation_result.fallback_reason
            )
        elif continuation_attempted:
            runtime_metrics["same_request_result_missing"] = True
            runtime_metrics["same_request_fallback_reason"] = "continuation_result_missing"
        if continuation_setup_error is not None:
            runtime_metrics["same_request_fallback_reason"] = continuation_setup_error

        planner_output = self.model_adapter.parse_planner_output(llm_output)
        pixel_goal = planner_output.pixel_goal

        return DualVLNMiniEngineGenerateResult(
            external_request_id=request.external_request_id,
            internal_request_id=request_output.request_id,
            llm_output=llm_output,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
            pixel_goal=pixel_goal,
            mm_feature_count=int(len(image_placeholders)),
            runtime_metrics=runtime_metrics,
            metadata={
                "processed_prompt": processed_prompt,
                "continuation_result": continuation_result,
                "request_output": request_output,
                "input_images": input_images,
                "planner_output": planner_output,
            },
        )

    def extract_latents(
        self,
        request: DualVLNMiniEngineRequest,
        generate_result: DualVLNMiniEngineGenerateResult,
    ) -> DualVLNMiniEngineLatentsResult | None:
        prompt_token_ids = generate_result.prompt_token_ids
        generated_token_ids = generate_result.generated_token_ids
        input_images = list(generate_result.metadata.get("input_images") or [])
        processed_prompt = generate_result.metadata.get("processed_prompt") or {}
        continuation_result = generate_result.metadata.get("continuation_result")
        runtime_metrics = generate_result.runtime_metrics

        bundle_start = time.perf_counter()
        bundle = build_latents_request_bundle(
            processor=self.processor,
            messages=request.messages,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
            input_images=input_images,
            latent_queries=self.latent_queries,
            traj_token_index=self.traj_token_index,
            n_query=self.n_query,
        )
        runtime_metrics["bundle_build_ms"] = (time.perf_counter() - bundle_start) * 1000.0

        mm_attach_source = os.environ.get("INTERNNAV_STEP_S2_MM_SOURCE", "engine_core_request")
        mm_attach_start = time.perf_counter()
        if mm_attach_source == "engine_core_request":
            attach_explicit_mm_metadata_from_engine_core_request(
                bundle,
                self._last_step_s2_engine_request,
            )
        elif mm_attach_source == "processed_prompt":
            attach_explicit_mm_metadata_from_processed_inputs(bundle, processed_prompt)
        elif mm_attach_source == "llm_input_processor":
            attach_explicit_mm_metadata(bundle, self.llm)
        else:
            raise ValueError(f"Unsupported INTERNNAV_STEP_S2_MM_SOURCE: {mm_attach_source}")

        mm_attach_backfill = None
        if mm_attach_source == "engine_core_request" and _bundle_mm_features_have_missing_payload(bundle):
            attach_explicit_mm_metadata_from_processed_inputs(bundle, processed_prompt)
            mm_attach_backfill = "processed_prompt_missing_data"
        if _bundle_mm_features_have_missing_payload(bundle):
            attach_explicit_mm_metadata(bundle, self.llm)
            if mm_attach_backfill is None:
                mm_attach_backfill = "llm_input_processor_missing_data"
            else:
                mm_attach_backfill = f"{mm_attach_backfill}+llm_input_processor_missing_data"
        runtime_metrics["mm_attach_ms"] = (time.perf_counter() - mm_attach_start) * 1000.0

        mm_debug = _collect_step_s2_mm_debug(processed_prompt, bundle)
        mm_debug["mm_attach_source"] = mm_attach_source
        mm_debug["mm_attach_backfill"] = mm_attach_backfill
        runtime_metrics["prefill_token_count"] = int(len(bundle.prefill_token_ids))
        runtime_metrics["mm_feature_count"] = int(len(bundle.mm_features or []))
        if os.environ.get("INTERNNAV_DEBUG_STEP_S2_MM"):
            print(
                "[DualVLN step_s2 mm_debug] "
                + json.dumps(mm_debug, ensure_ascii=False, sort_keys=True),
                flush=True,
            )

        latent_prefill_start = time.perf_counter()
        continuation_latents = (
            continuation_result.latents
            if continuation_result is not None
            and bool(continuation_result.continuation_used)
            and continuation_result.latents is not None
            else None
        )
        fallback_reason = runtime_metrics.get("same_request_fallback_reason")
        reused_prefill = False
        same_request_continuation_used = False
        mode = runtime_metrics["latent_path"]

        if continuation_latents is not None:
            runtime_metrics["same_request_continuation_used"] = True
            runtime_metrics["latent_path"] = "same_request_continuation"
            latents = continuation_latents
            runtime_metrics["prefill_token_count"] = int(
                len(prompt_token_ids) + len(generated_token_ids) + self.n_query
            )
            same_request_continuation_used = True
            mode = "same_request_continuation"
            logger.info(
                "DualVLN step_s2 used same-request latent continuation for request %s "
                "(internal=%s, suffix_len=%d).",
                generate_result.internal_request_id,
                continuation_result.internal_request_id,
                self.n_query,
            )
        elif self.latent_backend == "transformers_backend_apply_model":
            runtime_metrics["latent_path"] = "transformers_backend_apply_model"
            mode = "transformers_backend_apply_model"
            latents = self.llm.apply_model(
                functools.partial(
                    generate_latents_via_transformers_backend_apply_model,
                    full_output_ids_cpu=bundle.full_output_ids,
                    pixel_values_cpu=bundle.pixel_values,
                    image_grid_thw_cpu=bundle.image_grid_thw,
                )
            )[0]
        elif self.latent_backend in ("legacy_custom_forward", "shared_engine_forward"):
            runtime_metrics["latent_path"] = "native_latent_prefill_fallback"
            mode = "native_latent_prefill_fallback"
            if runtime_metrics["same_request_continuation_attempted"]:
                logger.info(
                    "DualVLN step_s2 falling back to native latent prefill for request %s: %s",
                    generate_result.internal_request_id,
                    fallback_reason,
                )
            latents = self._generate_latents_via_shared_engine(bundle)
            reused_prefill = True
        elif self.latent_backend == "vllm_hidden_separate_llm":
            runtime_metrics["latent_path"] = "vllm_hidden_separate_llm"
            mode = "vllm_hidden_separate_llm"
            latents = self._ensure_hidden_latents_runner().generate_latents_from_bundle(bundle)
        else:
            raise ValueError(f"Unsupported latent_backend: {self.latent_backend}")

        runtime_metrics["latent_prefill_ms"] = (time.perf_counter() - latent_prefill_start) * 1000.0
        runtime_metrics["reused_prefill"] = bool(reused_prefill)
        return DualVLNMiniEngineLatentsResult(
            external_request_id=request.external_request_id,
            internal_request_id=generate_result.internal_request_id,
            latents=latents,
            prefill_token_ids=bundle.prefill_token_ids,
            latent_query_count=self.n_query,
            mode=mode,
            reused_prefill=reused_prefill,
            same_request_continuation_used=same_request_continuation_used,
            fallback_reason=fallback_reason,
            runtime_metrics=runtime_metrics,
            metadata={"debug_mm": mm_debug},
        )

class PatchedVLLMDualVLNMiniEngine(StatefulDualVLNMiniEngine):
    def __init__(self, backend: PatchedVLLMDualVLNBackend) -> None:
        super().__init__(backend)

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "PatchedVLLMDualVLNMiniEngine":
        return cls(PatchedVLLMDualVLNBackend.from_pretrained(*args, **kwargs))

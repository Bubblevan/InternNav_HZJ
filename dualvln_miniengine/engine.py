from __future__ import annotations

from typing import Protocol
import time

from .contracts import (
    DualVLNMiniEngineGenerateResult,
    DualVLNMiniEngineLatentsResult,
    DualVLNMiniEngineRequest,
    DualVLNMiniEngineStepResult,
)
from .request_state import DualVLNMiniEnginePhase, DualVLNMiniEngineRequestState
from .backends.base import DualVLNMiniEngineBackend


class DualVLNMiniEngine(Protocol):
    def step_s2(self, request: DualVLNMiniEngineRequest) -> DualVLNMiniEngineStepResult:
        ...

    def generate_text(
        self,
        request: DualVLNMiniEngineRequest,
    ) -> DualVLNMiniEngineGenerateResult:
        ...

    def extract_latents(
        self,
        request: DualVLNMiniEngineRequest,
        generate_result: DualVLNMiniEngineGenerateResult,
    ) -> DualVLNMiniEngineLatentsResult | None:
        ...


class StatefulDualVLNMiniEngine:
    def __init__(self, backend: DualVLNMiniEngineBackend) -> None:
        self.backend = backend
        self._request_states: dict[str, DualVLNMiniEngineRequestState] = {}

    def __getattr__(self, name: str):
        return getattr(self.backend, name)

    @property
    def n_query(self) -> int:
        return int(self.backend.n_query)

    def get_request_state(self, external_request_id: str) -> DualVLNMiniEngineRequestState | None:
        return self._request_states.get(external_request_id)

    def _ensure_request_state(
        self,
        request: DualVLNMiniEngineRequest,
    ) -> DualVLNMiniEngineRequestState:
        state = self._request_states.get(request.external_request_id)
        if state is None:
            state = DualVLNMiniEngineRequestState(
                external_request_id=request.external_request_id,
                latent_query_count=int(request.latent_query_count or 0),
            )
            self._request_states[request.external_request_id] = state
        return state

    def _advance_phase(
        self,
        state: DualVLNMiniEngineRequestState,
        phase: DualVLNMiniEnginePhase,
    ) -> DualVLNMiniEngineRequestState:
        state.phase = phase
        return state

    def _finalize_state(
        self,
        state: DualVLNMiniEngineRequestState,
        *,
        fallback_reason: str | None = None,
    ) -> None:
        state.phase = DualVLNMiniEnginePhase.FINISHED
        state.fallback_reason = fallback_reason

    def step_s2(self, request: DualVLNMiniEngineRequest) -> DualVLNMiniEngineStepResult:
        state = self._ensure_request_state(request)
        state.latent_query_count = int(request.latent_query_count or self.n_query)
        self._advance_phase(state, DualVLNMiniEnginePhase.PREPROCESSED)
        total_start = time.perf_counter()
        generate_result = self.backend.generate_text(request)

        state.internal_request_id = generate_result.internal_request_id
        state.prompt_token_count = len(generate_result.prompt_token_ids)
        state.generated_token_count = len(generate_result.generated_token_ids)
        state.mm_feature_count = int(generate_result.mm_feature_count)
        self._advance_phase(state, DualVLNMiniEnginePhase.GENERATED_TEXT)

        latents_result = None
        if request.return_latents:
            state.prefill_token_count = (
                len(generate_result.prompt_token_ids)
                + len(generate_result.generated_token_ids)
                + int(request.latent_query_count or self.n_query)
            )
            self._advance_phase(state, DualVLNMiniEnginePhase.LATENT_PREFILL_READY)
            if generate_result.pixel_goal is not None:
                latents_result = self.backend.extract_latents(request, generate_result)
            if latents_result is not None:
                state.same_request_continuation_used = bool(
                    latents_result.same_request_continuation_used
                )
                state.fallback_reason = latents_result.fallback_reason
                self._advance_phase(state, DualVLNMiniEnginePhase.LATENTS_READY)

        total_ms = (time.perf_counter() - total_start) * 1000.0
        generate_result.runtime_metrics["total_ms"] = total_ms
        if (
            latents_result is not None
            and total_ms > 0.0
            and latents_result.runtime_metrics.get("latent_prefill_ms") is not None
        ):
            latents_result.runtime_metrics["total_ms"] = total_ms
            latents_result.runtime_metrics["latent_prefill_share_of_total"] = (
                latents_result.runtime_metrics["latent_prefill_ms"] / total_ms
            )

        self._finalize_state(
            state,
            fallback_reason=None if latents_result is None else latents_result.fallback_reason,
        )
        engine_metadata = {
            "engine_class": type(self).__name__,
            "backend_name": self.backend.backend_name,
            "external_request_id": request.external_request_id,
            "internal_request_id": generate_result.internal_request_id,
            "phase": state.phase.value,
            "same_request_continuation_used": bool(
                False if latents_result is None else latents_result.same_request_continuation_used
            ),
            "fallback_reason": None if latents_result is None else latents_result.fallback_reason,
        }
        return DualVLNMiniEngineStepResult(
            generate=generate_result,
            latents=latents_result,
            engine_metadata=engine_metadata,
            backend_metadata=self.backend.describe_backend(),
            backend_runtime=self.backend.get_runtime_stats(),
            vllm_kv_cache=(
                self.backend.get_runtime_stats()
                if self.backend.backend_name == "patched_vllm"
                else None
            ),
        )

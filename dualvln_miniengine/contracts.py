from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch


@dataclass
class DualVLNMiniEngineRequest:
    external_request_id: str
    messages: list[dict[str, Any]]
    max_new_tokens: int
    return_latents: bool = True
    return_hidden_states: bool = False
    prompt_token_ids: Optional[list[int]] = None
    generated_token_ids: Optional[list[int]] = None
    prompt_embeds: Optional[torch.Tensor] = None
    prompt_embeds_soft_suffix_len: Optional[int] = None
    mm_features: Optional[Sequence[Any]] = None
    latent_query_count: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DualVLNMiniEngineGenerateResult:
    external_request_id: str
    internal_request_id: Optional[str]
    llm_output: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    pixel_goal: Optional[list[int]] = None
    mm_feature_count: int = 0
    runtime_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DualVLNMiniEngineLatentsResult:
    external_request_id: str
    internal_request_id: Optional[str]
    latents: Optional[torch.Tensor]
    prefill_token_ids: list[int]
    latent_query_count: int
    mode: str
    reused_prefill: bool = False
    same_request_continuation_used: bool = False
    fallback_reason: Optional[str] = None
    runtime_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DualVLNMiniEngineStepResult:
    generate: DualVLNMiniEngineGenerateResult
    latents: Optional[DualVLNMiniEngineLatentsResult]
    debug_mm: Optional[dict[str, Any]] = None
    engine_metadata: dict[str, Any] = field(default_factory=dict)
    backend_metadata: dict[str, Any] = field(default_factory=dict)
    backend_runtime: Optional[dict[str, Any]] = None
    vllm_kv_cache: Optional[dict[str, Any]] = None

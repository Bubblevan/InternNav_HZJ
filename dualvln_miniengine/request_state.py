from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class DualVLNMiniEnginePhase(str, Enum):
    RECEIVED = "received"
    PREPROCESSED = "preprocessed"
    GENERATED_TEXT = "generated_text"
    LATENT_PREFILL_READY = "latent_prefill_ready"
    LATENT_CONTINUATION_ARMED = "latent_continuation_armed"
    LATENTS_READY = "latents_ready"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass
class DualVLNMiniEngineRequestState:
    external_request_id: str
    internal_request_id: Optional[str] = None
    phase: DualVLNMiniEnginePhase = DualVLNMiniEnginePhase.RECEIVED
    prompt_token_count: int = 0
    generated_token_count: int = 0
    prefill_token_count: int = 0
    mm_feature_count: int = 0
    latent_query_count: int = 0
    same_request_continuation_armed: bool = False
    same_request_continuation_used: bool = False
    fallback_reason: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

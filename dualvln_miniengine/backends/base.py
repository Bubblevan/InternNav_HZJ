from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from dualvln_miniengine.contracts import (
    DualVLNMiniEngineGenerateResult,
    DualVLNMiniEngineLatentsResult,
    DualVLNMiniEngineRequest,
)
from dualvln_model_adapters.base import VLNModelAdapter


class DualVLNMiniEngineBackend(ABC):
    backend_name = "unknown"

    def __init__(self, model_adapter: VLNModelAdapter | None = None) -> None:
        self.model_adapter = model_adapter

    @property
    @abstractmethod
    def n_query(self) -> int:
        raise NotImplementedError

    @property
    def capabilities(self) -> dict[str, bool]:
        return {
            "text_generate": True,
            "hidden_states": False,
            "latent_path": False,
            "same_request_continuation": False,
            "latent_prefill_reuse": False,
        }

    def describe_backend(self) -> dict[str, Any]:
        payload = {
            "backend_name": self.backend_name,
            "capabilities": dict(self.capabilities),
            "latent_query_count": int(self.n_query),
        }
        if self.model_adapter is not None:
            payload["model_adapter"] = self.model_adapter.describe()
        return payload

    def get_runtime_stats(self) -> dict[str, Any] | None:
        return None

    def close(self) -> None:
        return None

    @abstractmethod
    def generate_text(
        self,
        request: DualVLNMiniEngineRequest,
    ) -> DualVLNMiniEngineGenerateResult:
        raise NotImplementedError

    @abstractmethod
    def extract_latents(
        self,
        request: DualVLNMiniEngineRequest,
        generate_result: DualVLNMiniEngineGenerateResult,
    ) -> DualVLNMiniEngineLatentsResult | None:
        raise NotImplementedError

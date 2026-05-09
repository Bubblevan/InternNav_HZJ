from __future__ import annotations

from dualvln_model_adapters.base import VLNModelAdapter
from dualvln_miniengine.backends.base import DualVLNMiniEngineBackend
from dualvln_miniengine.contracts import (
    DualVLNMiniEngineGenerateResult,
    DualVLNMiniEngineLatentsResult,
    DualVLNMiniEngineRequest,
)


class TensorRTDualVLNBackendStub(DualVLNMiniEngineBackend):
    backend_name = "tensorrt_stub"

    def __init__(self, *, model_adapter: VLNModelAdapter) -> None:
        super().__init__(model_adapter=model_adapter)

    @property
    def n_query(self) -> int:
        raise NotImplementedError("TensorRT backend stub does not have a compiled engine yet.")

    @property
    def capabilities(self) -> dict[str, bool]:
        return {
            "text_generate": False,
            "hidden_states": False,
            "latent_path": False,
            "same_request_continuation": False,
            "latent_prefill_reuse": False,
        }

    def generate_text(
        self,
        request: DualVLNMiniEngineRequest,
    ) -> DualVLNMiniEngineGenerateResult:
        raise NotImplementedError("TensorRT backend stub is a platform placeholder only.")

    def extract_latents(
        self,
        request: DualVLNMiniEngineRequest,
        generate_result: DualVLNMiniEngineGenerateResult,
    ) -> DualVLNMiniEngineLatentsResult | None:
        raise NotImplementedError("TensorRT backend stub is a platform placeholder only.")

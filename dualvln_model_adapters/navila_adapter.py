from __future__ import annotations

import torch

from .base import PlannerOutput, VLNModelAdapter


class NaVILAAdapterStub(VLNModelAdapter):
    adapter_name = "navila_stub"
    model_family = "navila"

    @property
    def capabilities(self) -> dict[str, bool]:
        return {
            "pixel_goal": False,
            "discrete_actions": False,
            "hidden_states": False,
            "latents": False,
        }

    @property
    def traj_token_index(self) -> int:
        raise NotImplementedError("NaVILA adapter stub does not define traj token semantics yet.")

    def build_generation_inputs(self, processor, messages, *, device):
        raise NotImplementedError("NaVILA adapter stub is a platform placeholder only.")

    def parse_planner_output(self, output_text: str) -> PlannerOutput:
        raise NotImplementedError("NaVILA adapter stub is a platform placeholder only.")

    def load_latent_queries(self, model_path: str) -> torch.Tensor:
        raise NotImplementedError("NaVILA adapter stub is a platform placeholder only.")

    def get_latent_query_count(self, model) -> int:
        raise NotImplementedError("NaVILA adapter stub is a platform placeholder only.")

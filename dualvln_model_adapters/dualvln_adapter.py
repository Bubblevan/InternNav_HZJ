from __future__ import annotations

import re
from typing import Any

import torch

from dualvln_vllm_adapter.model_exec import (
    TRAJ_TOKEN_INDEX,
    build_hf_generation_inputs,
    load_latent_queries_tensor,
    parse_pixel_goal_from_text,
)

from .base import PlannerOutput, VLNModelAdapter


class DualVLNModelAdapter(VLNModelAdapter):
    adapter_name = "dualvln"
    model_family = "internvla_n1"

    @property
    def capabilities(self) -> dict[str, bool]:
        return {
            "pixel_goal": True,
            "discrete_actions": True,
            "hidden_states": True,
            "latents": True,
        }

    @property
    def traj_token_index(self) -> int:
        return int(TRAJ_TOKEN_INDEX)

    def build_generation_inputs(
        self,
        processor,
        messages,
        *,
        device,
    ) -> dict[str, Any]:
        return build_hf_generation_inputs(
            processor,
            messages,
            device=device,
        )

    def parse_planner_output(self, output_text: str) -> PlannerOutput:
        pixel_goal = parse_pixel_goal_from_text(output_text)
        discrete_actions: list[int] = []
        if pixel_goal is None:
            token_map = {
                "STOP": 0,
                "↑": 1,
                "←": 2,
                "→": 3,
                "↓": 5,
            }
            text = (output_text or "").strip()
            if text == "STOP":
                discrete_actions = [0]
            elif text:
                discrete_actions = [token_map[c] for c in text if c in token_map]
        output_type = "pixel_goal" if pixel_goal is not None else "discrete_action"
        return PlannerOutput(
            raw_text=output_text,
            output_type=output_type,
            pixel_goal=pixel_goal,
            discrete_actions=discrete_actions,
            metadata={
                "contains_digit": bool(re.search(r"\d", output_text or "")),
            },
        )

    def load_latent_queries(self, model_path: str) -> torch.Tensor:
        return load_latent_queries_tensor(model_path)

    def get_latent_query_count(self, model) -> int:
        if hasattr(model, "get_n_query"):
            return int(model.get_n_query())
        raise RuntimeError(
            "DualVLNModelAdapter could not infer latent query count from the model instance."
        )

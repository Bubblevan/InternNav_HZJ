from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class PlannerOutput:
    raw_text: str
    output_type: str
    pixel_goal: Optional[list[int]] = None
    discrete_actions: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class VLNModelAdapter(ABC):
    adapter_name = "unknown"
    model_family = "unknown"

    @property
    def capabilities(self) -> dict[str, bool]:
        return {
            "pixel_goal": True,
            "discrete_actions": False,
            "hidden_states": False,
            "latents": False,
        }

    def describe(self) -> dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "model_family": self.model_family,
            "capabilities": dict(self.capabilities),
        }

    @property
    @abstractmethod
    def traj_token_index(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def build_generation_inputs(
        self,
        processor,
        messages,
        *,
        device,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def parse_planner_output(self, output_text: str) -> PlannerOutput:
        raise NotImplementedError

    @abstractmethod
    def load_latent_queries(self, model_path: str) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_latent_query_count(self, model) -> int:
        raise NotImplementedError

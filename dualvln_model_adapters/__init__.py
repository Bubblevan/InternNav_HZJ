from .base import PlannerOutput, VLNModelAdapter
from .dualvln_adapter import DualVLNModelAdapter
from .navila_adapter import NaVILAAdapterStub


def create_model_adapter(name: str = "dualvln") -> VLNModelAdapter:
    normalized = str(name).strip().lower()
    if normalized in {"dualvln", "internvla_n1"}:
        return DualVLNModelAdapter()
    if normalized in {"navila", "navila_stub"}:
        return NaVILAAdapterStub()
    raise ValueError(f"Unsupported model adapter: {name}")


__all__ = [
    "PlannerOutput",
    "VLNModelAdapter",
    "DualVLNModelAdapter",
    "NaVILAAdapterStub",
    "create_model_adapter",
]

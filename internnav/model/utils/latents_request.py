from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class LatentsRequestBundle:
    prompt_token_ids: List[int]
    generated_token_ids: List[int]
    full_output_token_ids: List[int]
    full_output_ids: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    input_images: List[Any]
    latent_queries: torch.Tensor
    traj_token_index: int
    n_query: int
    prompt_embeds: Optional[torch.Tensor] = None
    mm_kwargs: Optional[Any] = None
    mm_hashes: Optional[Any] = None
    mm_placeholders: Optional[Any] = None
    mm_features: Optional[Any] = None

    @property
    def prefill_token_ids(self) -> List[int]:
        return self.full_output_token_ids + [self.traj_token_index] * self.n_query


def attach_explicit_mm_metadata_from_processed_inputs(
    bundle: LatentsRequestBundle,
    processed_inputs,
) -> LatentsRequestBundle:
    if not bundle.input_images:
        bundle.mm_kwargs = None
        bundle.mm_hashes = None
        bundle.mm_placeholders = None
        bundle.mm_features = None
        return bundle

    if processed_inputs.get("type") != "multimodal":
        raise RuntimeError(
            "Expected vLLM chat preprocessing to return multimodal inputs, "
            f"got type={processed_inputs.get('type')!r}."
        )

    processed_prompt_token_ids = list(processed_inputs.get("prompt_token_ids") or [])
    if processed_prompt_token_ids != bundle.prompt_token_ids:
        raise RuntimeError(
            "vLLM chat preprocessing prompt_token_ids do not match the canonical "
            "request_output.prompt_token_ids used by single-vLLM."
        )

    bundle.mm_kwargs = processed_inputs.get("mm_kwargs")
    bundle.mm_hashes = processed_inputs.get("mm_hashes")
    bundle.mm_placeholders = processed_inputs.get("mm_placeholders")

    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.multimodal.utils import argsort_mm_positions

    mm_features = []
    mm_kwargs = bundle.mm_kwargs or {}
    mm_hashes = bundle.mm_hashes or {}
    mm_placeholders = bundle.mm_placeholders or {}
    for modality, idx in argsort_mm_positions(mm_placeholders):
        item_hash = mm_hashes[modality][idx]
        item_data = mm_kwargs[modality][idx]
        mm_features.append(
            MultiModalFeatureSpec(
                data=item_data,
                modality=modality,
                identifier=item_hash,
                mm_position=mm_placeholders[modality][idx],
                mm_hash=item_hash,
            )
        )
    bundle.mm_features = mm_features or None
    return bundle


def _attach_mm_metadata_from_features(
    bundle: LatentsRequestBundle,
    mm_features,
) -> LatentsRequestBundle:
    mm_kwargs: Dict[str, List[Any]] = defaultdict(list)
    mm_hashes: Dict[str, List[str | None]] = defaultdict(list)
    mm_placeholders: Dict[str, List[Any]] = defaultdict(list)
    copied_mm_features = list(mm_features or [])

    for mm_feature in copied_mm_features:
        mm_kwargs[mm_feature.modality].append(mm_feature.data)
        mm_hashes[mm_feature.modality].append(mm_feature.mm_hash)
        mm_placeholders[mm_feature.modality].append(mm_feature.mm_position)

    bundle.mm_kwargs = dict(mm_kwargs) if mm_kwargs else None
    bundle.mm_hashes = dict(mm_hashes) if mm_hashes else None
    bundle.mm_placeholders = dict(mm_placeholders) if mm_placeholders else None
    bundle.mm_features = copied_mm_features or None
    return bundle


def attach_explicit_mm_metadata_from_engine_core_request(
    bundle: LatentsRequestBundle,
    engine_request,
) -> LatentsRequestBundle:
    if (
        bundle.mm_kwargs is not None
        and bundle.mm_hashes is not None
        and bundle.mm_placeholders is not None
        and bundle.mm_features is not None
    ):
        return bundle

    if engine_request is None:
        raise RuntimeError("Missing EngineCoreRequest for step_s2 multimodal metadata attach.")

    processed_prompt_token_ids = list(engine_request.prompt_token_ids or [])
    if processed_prompt_token_ids != bundle.prompt_token_ids:
        raise RuntimeError(
            "EngineCoreRequest prompt_token_ids do not match the canonical "
            "request_output.prompt_token_ids used by single-vLLM."
        )

    return _attach_mm_metadata_from_features(bundle, engine_request.mm_features)


def build_latents_request_bundle(
    *,
    processor,
    messages,
    prompt_token_ids: List[int],
    generated_token_ids: List[int],
    input_images,
    latent_queries: torch.Tensor,
    traj_token_index: int,
    n_query: int,
) -> LatentsRequestBundle:
    prompt_token_ids = list(prompt_token_ids)
    generated_token_ids = list(generated_token_ids)
    full_output_token_ids = prompt_token_ids + generated_token_ids
    full_output_ids = torch.tensor([full_output_token_ids], dtype=torch.long)

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    hf_inputs = processor(
        text=[text],
        images=input_images if input_images else None,
        return_tensors="pt",
    )

    return LatentsRequestBundle(
        prompt_token_ids=prompt_token_ids,
        generated_token_ids=generated_token_ids,
        full_output_token_ids=full_output_token_ids,
        full_output_ids=full_output_ids,
        pixel_values=hf_inputs.pixel_values.detach().cpu(),
        image_grid_thw=hf_inputs.image_grid_thw.detach().cpu(),
        input_images=list(input_images or []),
        latent_queries=latent_queries.detach().cpu(),
        traj_token_index=int(traj_token_index),
        n_query=int(n_query),
    )


def build_latents_request_bundle_from_tensors(
    *,
    output_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    input_images,
    latent_queries: torch.Tensor,
    traj_token_index: int,
    n_query: int,
) -> LatentsRequestBundle:
    if output_ids.ndim != 2 or output_ids.shape[0] != 1:
        raise ValueError(f"Only batch size 1 is supported, got shape={tuple(output_ids.shape)}")

    full_output_token_ids = output_ids[0].detach().cpu().tolist()
    return LatentsRequestBundle(
        prompt_token_ids=list(full_output_token_ids),
        generated_token_ids=[],
        full_output_token_ids=full_output_token_ids,
        full_output_ids=output_ids.detach().cpu(),
        pixel_values=pixel_values.detach().cpu(),
        image_grid_thw=image_grid_thw.detach().cpu(),
        input_images=list(input_images or []),
        latent_queries=latent_queries.detach().cpu(),
        traj_token_index=int(traj_token_index),
        n_query=int(n_query),
    )


def attach_explicit_mm_metadata(bundle: LatentsRequestBundle, llm) -> LatentsRequestBundle:
    if (
        bundle.mm_kwargs is not None
        and bundle.mm_hashes is not None
        and bundle.mm_placeholders is not None
        and bundle.mm_features is not None
    ):
        return bundle

    if not bundle.input_images:
        bundle.mm_kwargs = None
        bundle.mm_hashes = None
        bundle.mm_placeholders = None
        bundle.mm_features = None
        return bundle

    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.tasks import POOLING_TASKS

    prompt = {
        "prompt_token_ids": bundle.prefill_token_ids,
        "multi_modal_data": {"image": bundle.input_images},
    }
    supported_tasks = tuple(llm.supported_tasks)
    if any(task in POOLING_TASKS for task in supported_tasks):
        params = PoolingParams(task="token_embed")
    else:
        # We only need InputProcessor's multimodal normalization here. For a
        # generation-only engine, a trivial SamplingParams is enough to build
        # EngineCoreRequest.mm_features without ever submitting a decode.
        params = SamplingParams(max_tokens=1, temperature=0.0)
    engine_request = llm.input_processor.process_inputs(
        request_id="internnav-generate-latents-prefill",
        prompt=prompt,
        params=params,
        supported_tasks=supported_tasks,
    )

    processed_prompt_token_ids = list(engine_request.prompt_token_ids or [])
    if processed_prompt_token_ids != bundle.prefill_token_ids:
        raise RuntimeError(
            "vLLM preprocessor changed the canonical prefill token ids for generate_latents. "
            "This breaks the strict HF-aligned latent contract."
        )
    return _attach_mm_metadata_from_features(bundle, engine_request.mm_features)

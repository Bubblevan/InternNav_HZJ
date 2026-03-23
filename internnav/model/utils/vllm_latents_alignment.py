from typing import Any, Optional, Sequence

import torch


def _move_nested_to_device(value: Any, *, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, list):
        return [_move_nested_to_device(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_nested_to_device(item, device=device) for item in value)
    if isinstance(value, dict):
        return {
            key: _move_nested_to_device(item, device=device) for key, item in value.items()
        }
    return value


def _get_mm_item_data(mm_item: Any) -> dict:
    getter = getattr(mm_item, "get_data", None)
    if callable(getter):
        return getter()

    data = {}
    for key, value in mm_item.items():
        data[key] = getattr(value, "data", value)
    return data


def build_is_multimodal_mask(
    seq_len: int,
    mm_features: Optional[Sequence[Any]],
    *,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not mm_features:
        return None

    is_multimodal = torch.zeros(seq_len, dtype=torch.bool, device=device)
    for mm_feature in sorted(mm_features, key=lambda feature: feature.mm_position.offset):
        mm_position = mm_feature.mm_position
        offset = int(mm_position.offset)
        length = int(mm_position.length)
        is_embed = getattr(mm_position, "is_embed", None)
        if is_embed is None:
            is_multimodal[offset : offset + length] = True
            continue

        embed_mask = torch.as_tensor(is_embed, dtype=torch.bool, device=device).view(-1)
        if int(embed_mask.numel()) != length:
            raise RuntimeError(
                "mm_position.is_embed length does not match placeholder length: "
                f"{int(embed_mask.numel())} != {length}"
            )
        is_multimodal[offset : offset + length] |= embed_mask

    return is_multimodal


def build_multimodal_embeddings_from_mm_features(
    model,
    mm_features: Optional[Sequence[Any]],
    *,
    device: torch.device,
) -> tuple:
    if not mm_features:
        return ()

    multimodal_embeddings = []
    for mm_feature in sorted(mm_features, key=lambda feature: feature.mm_position.offset):
        mm_item = mm_feature.data
        if mm_item is None:
            raise RuntimeError(
                "Missing mm_feature.data while rebuilding generate_latents prompt embeds. "
                "This indicates vLLM cached away the multimodal payload."
            )

        mm_kwargs = {
            key: _move_nested_to_device(value, device=device)
            for key, value in _get_mm_item_data(mm_item).items()
        }
        item_embeddings = model.embed_multimodal(**mm_kwargs)
        if isinstance(item_embeddings, torch.Tensor):
            multimodal_embeddings.append(item_embeddings)
            continue
        if len(item_embeddings) != 1:
            raise RuntimeError(
                "Expected exactly one multimodal embedding tensor per feature, got "
                f"{len(item_embeddings)} for modality={mm_feature.modality!r}."
            )
        multimodal_embeddings.append(item_embeddings[0])

    return tuple(multimodal_embeddings)


def build_prompt_embeds_with_mm_features(
    *,
    model,
    input_ids: torch.Tensor,
    latent_queries: torch.Tensor,
    mm_features: Optional[Sequence[Any]],
) -> torch.Tensor:
    if input_ids.ndim != 1:
        raise ValueError(f"Expected rank-1 input_ids, got shape={tuple(input_ids.shape)}")

    device = input_ids.device
    multimodal_embeddings = build_multimodal_embeddings_from_mm_features(
        model,
        mm_features,
        device=device,
    )
    is_multimodal = build_is_multimodal_mask(
        int(input_ids.shape[0]),
        mm_features,
        device=device,
    )

    if multimodal_embeddings:
        inputs_embeds = model.embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        ).clone()
    else:
        inputs_embeds = model.embed_input_ids(input_ids).clone()

    latent_queries = latent_queries.to(device=device, dtype=inputs_embeds.dtype)
    inputs_embeds[-latent_queries.shape[0] :] = latent_queries
    return inputs_embeds


def compute_mrope_positions_from_mm_features(
    *,
    model,
    prompt_token_ids: Sequence[int],
    mm_features: Optional[Sequence[Any]],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not mm_features or not hasattr(model, "get_mrope_input_positions"):
        return None

    positions, _ = model.get_mrope_input_positions(
        list(prompt_token_ids),
        list(mm_features),
    )
    return positions.to(device=device)

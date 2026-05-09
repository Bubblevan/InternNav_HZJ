from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Optional

import torch
from safetensors.torch import load_file

from .mm_alignment import build_prompt_embeds_with_mm_features, compute_mrope_positions_from_mm_features

TRAJ_TOKEN_INDEX = 151667


def parse_pixel_goal_from_text(output_text: str) -> list[int] | None:
    if not re.search(r"\d", output_text or ""):
        return None
    coord = [int(c) for c in re.findall(r"\d+", output_text)]
    if len(coord) < 2:
        return None
    return [int(coord[1]), int(coord[0])]


def normalize_image_grid_thw(image_grid_thw) -> Optional[torch.LongTensor]:
    if image_grid_thw is None:
        return None
    if isinstance(image_grid_thw, torch.Tensor):
        if image_grid_thw.ndim == 2:
            return image_grid_thw
        if image_grid_thw.ndim == 3:
            return image_grid_thw.flatten(0, 1)
        raise RuntimeError(f"Unsupported image_grid_thw tensor shape: {tuple(image_grid_thw.shape)}")
    if isinstance(image_grid_thw, (list, tuple)):
        tensors = []
        for item in image_grid_thw:
            if isinstance(item, torch.Tensor):
                tensors.append(item.unsqueeze(0) if item.ndim == 1 else item)
        if not tensors:
            return None
        return torch.cat(tensors, dim=0)
    raise RuntimeError(f"Unsupported image_grid_thw type: {type(image_grid_thw)!r}")


def build_hf_generation_inputs(
    processor,
    messages,
    *,
    device,
):
    timings = {}
    image_collect_start = time.perf_counter()
    input_images = []
    for message in messages:
        for part in message.get("content", []):
            if part.get("type") == "image":
                input_images.append(part["image"])
    timings["image_collect_ms"] = (time.perf_counter() - image_collect_start) * 1000.0
    prompt_build_start = time.perf_counter()
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    timings["prompt_build_ms"] = (time.perf_counter() - prompt_build_start) * 1000.0
    processor_start = time.perf_counter()
    model_inputs = processor(
        text=[prompt_text],
        images=input_images,
        return_tensors="pt",
    )
    timings["processor_ms"] = (time.perf_counter() - processor_start) * 1000.0
    to_device_start = time.perf_counter()
    model_inputs = model_inputs.to(device)
    timings["to_device_ms"] = (time.perf_counter() - to_device_start) * 1000.0
    return {
        "prompt_text": prompt_text,
        "input_images": input_images,
        "model_inputs": model_inputs,
        "image_grid_thw": normalize_image_grid_thw(getattr(model_inputs, "image_grid_thw", None)),
        "timings": timings,
    }


def load_latent_queries_tensor(model_path: str) -> torch.Tensor:
    index_path = Path(model_path) / "model.safetensors.index.json"
    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index_data["weight_map"]
    tensor_name = "model.latent_queries"
    shard_name = weight_map[tensor_name]
    shard_path = Path(model_path) / shard_name
    state = load_file(str(shard_path), device="cpu")
    latent_queries = state[tensor_name]
    if latent_queries.ndim != 3 or latent_queries.shape[0] != 1:
        raise RuntimeError(f"Unexpected latent_queries shape: {tuple(latent_queries.shape)}")
    return latent_queries[0].contiguous()


def compute_qwen2_5_vl_rope_index(
    input_ids: torch.LongTensor,
    config,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    spatial_merge_size = config.vision_config.spatial_merge_size
    image_token_id = config.image_token_id
    video_token_id = config.video_token_id
    vision_start_token_id = config.vision_start_token_id
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is not None:
            attention_mask = attention_mask == 1
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        for i, sample_input_ids in enumerate(total_input_ids):
            sample_attention_mask = attention_mask[i] if attention_mask is not None else None
            if sample_attention_mask is not None:
                sample_input_ids = sample_input_ids[sample_attention_mask]
            vision_start_indices = torch.argwhere(sample_input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = sample_input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = sample_input_ids.tolist()
            llm_pos_ids_list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = image_grid_thw[image_index]
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = video_grid_thw[video_index]
                    second_per_grid_t = second_per_grid_ts[video_index] if second_per_grid_ts is not None else 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t = t.item()
                llm_grid_h = h.item() // spatial_merge_size
                llm_grid_w = w.item() // spatial_merge_size
                text_len = ed - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx)

                range_tensor = torch.arange(llm_grid_t, device=input_ids.device).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                second_per_grid_t = torch.as_tensor(
                    second_per_grid_t,
                    dtype=range_tensor.dtype,
                    device=range_tensor.device,
                )
                time_tensor = expanded_range * second_per_grid_t * config.vision_config.tokens_per_second
                t_index = time_tensor.long().flatten()
                h_index = torch.arange(llm_grid_h, device=input_ids.device).view(1, -1, 1).expand(
                    llm_grid_t, -1, llm_grid_w
                ).flatten()
                w_index = torch.arange(llm_grid_w, device=input_ids.device).view(1, 1, -1).expand(
                    llm_grid_t, llm_grid_h, -1
                ).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if sample_attention_mask is not None:
                position_ids[..., i, sample_attention_mask] = llm_positions.to(position_ids.device)
            else:
                position_ids[..., i, :] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
    else:
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(
            3, input_ids.shape[0], -1
        )
        mrope_position_deltas = torch.zeros([input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype)

    return position_ids, mrope_position_deltas


def generate_latents_from_vllm_model(
    model,
    prompt_token_ids,
    pixel_values_cpu,
    image_grid_thw_cpu,
    latent_queries_cpu,
    traj_token_index,
    n_query,
    mm_features=None,
):
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context

    device = next(model.parameters()).device
    full_prompt_token_ids = prompt_token_ids + [traj_token_index] * n_query
    input_ids = torch.tensor(full_prompt_token_ids, device=device, dtype=torch.long).unsqueeze(0)
    vllm_config = model.vllm_config

    with set_current_vllm_config(vllm_config), set_forward_context(
        None,
        vllm_config=vllm_config,
        num_tokens=input_ids.shape[1],
    ), torch.inference_mode():
        if mm_features:
            embeds = build_prompt_embeds_with_mm_features(
                model=model,
                input_ids=input_ids[0],
                latent_queries=latent_queries_cpu,
                mm_features=mm_features,
            )
        else:
            embeds = model.embed_input_ids(input_ids[0]).clone()

            if pixel_values_cpu is not None and image_grid_thw_cpu is not None:
                pixel_values = pixel_values_cpu.to(device=device, dtype=model.visual.dtype)
                image_grid_thw = image_grid_thw_cpu.to(device=device)
                multimodal_embeddings = model.embed_multimodal(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )
                flat_mm_embeddings = torch.cat(list(multimodal_embeddings), dim=0) if multimodal_embeddings else None
                if flat_mm_embeddings is not None:
                    image_idx = input_ids[0] == model.config.image_token_id
                    image_token_count = int(image_idx.sum().item())
                    embeds[image_idx] = flat_mm_embeddings[:image_token_count].to(embeds.dtype)

            latent_queries = latent_queries_cpu.to(device=device, dtype=embeds.dtype)
            embeds[-latent_queries.shape[0] :] = latent_queries

        position_ids = compute_mrope_positions_from_mm_features(
            model=model,
            prompt_token_ids=full_prompt_token_ids,
            mm_features=mm_features,
            device=device,
        )
        if position_ids is None:
            image_grid_thw = image_grid_thw_cpu.to(device=device) if image_grid_thw_cpu is not None else None
            position_ids, _ = compute_qwen2_5_vl_rope_index(
                input_ids,
                config=model.config,
                image_grid_thw=image_grid_thw,
            )
            position_ids = position_ids[:, 0, :]
        hidden_states = model.forward(
            input_ids=None,
            positions=position_ids,
            inputs_embeds=embeds,
        )
        return hidden_states[-n_query:, :].unsqueeze(0).cpu()


def build_native_latent_prefill_prompt_embeds(
    model,
    prompt_token_ids,
    latent_queries_cpu,
    mm_features,
):
    if mm_features is None:
        raise RuntimeError(
            "Native shared-engine latent prefill requires canonical mm_features."
        )

    device = next(model.parameters()).device
    input_ids = torch.tensor(
        prompt_token_ids,
        device=device,
        dtype=torch.long,
    )
    embeds = build_prompt_embeds_with_mm_features(
        model=model,
        input_ids=input_ids,
        latent_queries=latent_queries_cpu,
        mm_features=mm_features,
    )
    return embeds.detach().cpu()


def build_native_latent_prefill_suffix_prompt_embeds(
    model,
    latent_queries_cpu,
):
    embed_dtype = next(model.parameters()).dtype
    return latent_queries_cpu.to(dtype=embed_dtype).detach().cpu().contiguous()


def inspect_transformers_backend_model_tree(model) -> dict:
    wrapper_type = type(model).__name__
    wrapped = getattr(model, "model", None)
    wrapped_type = type(wrapped).__name__ if wrapped is not None else None
    inner = getattr(wrapped, "model", None)
    inner_type = type(inner).__name__ if inner is not None else None

    return {
        "wrapper_type": wrapper_type,
        "wrapped_type": wrapped_type,
        "inner_type": inner_type,
        "wrapper_has_generate_latents": hasattr(model, "generate_latents"),
        "wrapped_has_generate_latents": hasattr(wrapped, "generate_latents") if wrapped is not None else False,
        "wrapped_has_get_rope_index": hasattr(wrapped, "get_rope_index") if wrapped is not None else False,
        "wrapped_has_visual": hasattr(wrapped, "visual") if wrapped is not None else False,
        "wrapped_has_latent_queries": hasattr(wrapped, "latent_queries") if wrapped is not None else False,
        "inner_has_latent_queries": hasattr(inner, "latent_queries") if inner is not None else False,
    }


def generate_latents_via_transformers_backend_apply_model(
    model,
    full_output_ids_cpu,
    pixel_values_cpu,
    image_grid_thw_cpu,
):
    from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM

    wrapped = getattr(model, "model", None)
    if wrapped is None:
        raise RuntimeError(f"Transformers backend wrapper has no .model attribute: {type(model)}")

    if hasattr(wrapped, "generate_latents"):
        target = wrapped
        return target.generate_latents(
            full_output_ids_cpu.to(next(target.parameters()).device),
            pixel_values_cpu.to(next(target.parameters()).device),
            image_grid_thw_cpu.to(next(target.parameters()).device),
        ).detach().cpu()

    class _GenerateLatentsAdapter:
        class _ModelProxy:
            def __init__(self, wrapped_model, wrapper_model):
                self._wrapped_model = wrapped_model
                self._wrapper_model = wrapper_model
                self.embed_tokens = wrapped_model.language_model.get_input_embeddings()
                self.latent_queries = wrapped_model.latent_queries
                self.config = wrapped_model.config
                self.device = next(wrapped_model.parameters()).device

            def __call__(self, *args, **kwargs):
                kwargs.setdefault("attention_instances", self._wrapper_model.attention_instances)
                return self._wrapped_model(*args, **kwargs)

            def get_rope_index(self, *args, **kwargs):
                return self._wrapped_model.get_rope_index(*args, **kwargs)

        def __init__(self, wrapped_model):
            self.model = self._ModelProxy(wrapped_model, model)
            self.visual = wrapped_model.visual

        def get_model(self):
            return self.model

        def get_n_query(self):
            return int(self.model.config.n_query)

        def get_rope_index(self, *args, **kwargs):
            return self.model.get_rope_index(*args, **kwargs)

    adapter = _GenerateLatentsAdapter(wrapped)
    device = next(wrapped.parameters()).device
    try:
        return InternVLAN1ForCausalLM.generate_latents(
            adapter,
            full_output_ids_cpu.to(device),
            pixel_values_cpu.to(device),
            image_grid_thw_cpu.to(device),
        ).detach().cpu()
    except Exception as exc:
        raise RuntimeError(
            "transformers_backend_apply_model reached HF generate_latents logic, "
            "but the vLLM-backed forward still failed inside attention execution. "
            "This indicates apply_model is not yet supplying the full forward "
            "context / attention metadata expected by the transformers backend."
        ) from exc

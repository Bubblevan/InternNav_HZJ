import base64
import functools
import io
import json
import logging
import os
import re
import uuid
from multiprocessing import shared_memory
from pathlib import Path
from typing import Optional
import time

import requests as http_requests
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoProcessor

from internnav.model.utils.latents_request import (
    attach_explicit_mm_metadata,
    attach_explicit_mm_metadata_from_engine_core_request,
    attach_explicit_mm_metadata_from_processed_inputs,
    build_latents_request_bundle,
)
from internnav.model.utils.vllm_latents_alignment import (
    build_prompt_embeds_with_mm_features,
    compute_mrope_positions_from_mm_features,
)
from internnav.model.utils.vllm_hidden_latents import VLLMHiddenLatentsRunner

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

logger = logging.getLogger(__name__)

TRAJ_TOKEN_INDEX = 151667
DEFAULT_IMAGE_TRANSPORT_MODE = "base64"
IMAGE_TRANSPORT_ENV = "INTERNNAV_DUALVLN_IMAGE_TRANSPORT"


def _load_latent_queries_tensor(model_path: str) -> torch.Tensor:
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


def _compute_qwen2_5_vl_rope_index(
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


def _generate_latents_from_vllm_model(
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
            position_ids, _ = _compute_qwen2_5_vl_rope_index(
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


def _build_native_latent_prefill_prompt_embeds(
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


def _build_native_latent_prefill_suffix_prompt_embeds(
    model,
    latent_queries_cpu,
):
    embed_dtype = next(model.parameters()).dtype
    return latent_queries_cpu.to(dtype=embed_dtype).detach().cpu().contiguous()


def _inspect_transformers_backend_model_tree(model) -> dict:
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


def _generate_latents_via_transformers_backend_apply_model(
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


def encode_tensor_to_b64(tensor: torch.Tensor) -> str:
    buf = io.BytesIO()
    torch.save(tensor.detach().cpu(), buf)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_tensor_from_b64(payload: str) -> torch.Tensor:
    raw = base64.b64decode(payload.encode("utf-8"))
    return torch.load(io.BytesIO(raw), map_location="cpu")


def encode_pil_image_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_pil_image_from_b64(payload: str) -> Image.Image:
    raw = base64.b64decode(payload.encode("utf-8"))
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _get_image_transport_mode() -> str:
    mode = os.environ.get(IMAGE_TRANSPORT_ENV, DEFAULT_IMAGE_TRANSPORT_MODE).strip().lower()
    if mode not in {"base64", "image_shm"}:
        raise ValueError(
            f"Unsupported {IMAGE_TRANSPORT_ENV}={mode!r}; expected 'base64' or 'image_shm'."
        )
    return mode


def _build_image_shm_name() -> str:
    return f"internnav_img_{os.getpid()}_{time.time_ns()}_{uuid.uuid4().hex}"


def _best_effort_unregister_shared_memory(shm_handle: shared_memory.SharedMemory) -> None:
    try:
        from multiprocessing import resource_tracker

        resource_name = getattr(shm_handle, "_name", None) or shm_handle.name
        resource_tracker.unregister(resource_name, "shared_memory")
    except Exception:
        pass


def _encode_pil_image_to_shm(image: Image.Image, *, message_index: int, content_index: int):
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    raw = rgb_image.tobytes()
    shm_handle = shared_memory.SharedMemory(
        name=_build_image_shm_name(),
        create=True,
        size=len(raw),
    )
    shm_handle.buf[: len(raw)] = raw
    payload = {
        "type": "image",
        "image_transport": "image_shm",
        "shm_name": shm_handle.name,
        "shape": [height, width, 3],
        "dtype": "uint8",
        "mode": "RGB",
        "nbytes": len(raw),
        "message_index": int(message_index),
        "content_index": int(content_index),
    }
    return payload, shm_handle, len(raw)


def _decode_pil_image_from_shm(payload: dict) -> Image.Image:
    shm_name = payload["shm_name"]
    shape = payload["shape"]
    dtype = payload.get("dtype", "uint8")
    mode = payload.get("mode", "RGB")
    nbytes = int(payload["nbytes"])
    if dtype != "uint8":
        raise ValueError(f"Unsupported shared-memory image dtype: {dtype}")
    if mode != "RGB":
        raise ValueError(f"Unsupported shared-memory image mode: {mode}")
    if len(shape) != 3 or int(shape[2]) != 3:
        raise ValueError(f"Unsupported shared-memory image shape: {shape}")

    height = int(shape[0])
    width = int(shape[1])
    expected_nbytes = height * width * 3
    if nbytes != expected_nbytes:
        raise ValueError(
            f"Shared-memory image payload size mismatch: nbytes={nbytes}, expected={expected_nbytes}"
        )

    shm_handle = shared_memory.SharedMemory(name=shm_name, create=False)
    try:
        raw = bytes(shm_handle.buf[:nbytes])
        return Image.frombytes("RGB", (width, height), raw)
    finally:
        shm_handle.close()  # 这样 server 只负责读完关闭句柄，不负责删除共享内存


def _cleanup_client_shared_memory_handles(shared_memory_handles) -> None:
    for shm_handle in shared_memory_handles:
        try:
            shm_handle.close()
        finally:
            try:
                shm_handle.unlink()
            except FileNotFoundError:
                pass

def encode_messages(messages, *, image_transport_mode: Optional[str] = None):
    transport_mode = image_transport_mode or _get_image_transport_mode()
    encoded = []
    shared_memory_handles = []
    image_payload_bytes = 0
    image_count = 0
    for message in messages:
        content = []
        for content_index, item in enumerate(message["content"]):
            if item["type"] == "text":
                content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image":
                image_count += 1
                if transport_mode == "base64":
                    content.append(
                        {
                            "type": "image",
                            "image_transport": "base64",
                            "image": encode_pil_image_to_b64(item["image"]),
                        }
                    )
                elif transport_mode == "image_shm":
                    shm_payload, shm_handle, payload_bytes = _encode_pil_image_to_shm(
                        item["image"],
                        message_index=len(encoded),
                        content_index=content_index,
                    )
                    content.append(shm_payload)
                    shared_memory_handles.append(shm_handle)
                    image_payload_bytes += payload_bytes
                else:
                    raise ValueError(f"Unsupported image transport mode: {transport_mode}")
            else:
                raise ValueError(f"Unsupported message content type: {item['type']}")
        encoded.append({"role": message["role"], "content": content})
    return encoded, {
        "image_transport_mode": transport_mode,
        "shared_memory_handles": shared_memory_handles,
        "image_count": image_count,
        "image_payload_bytes": image_payload_bytes,
    }


def decode_messages(messages):
    decoded = []
    for message in messages:
        content = []
        for item in message["content"]:
            if item["type"] == "text":
                content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image":
                image_transport = item.get("image_transport", "base64")
                if image_transport == "base64":
                    content.append({"type": "image", "image": decode_pil_image_from_b64(item["image"])})
                elif image_transport == "image_shm":
                    content.append({"type": "image", "image": _decode_pil_image_from_shm(item)})
                else:
                    raise ValueError(f"Unsupported image transport type: {image_transport}")
            else:
                raise ValueError(f"Unsupported message content type: {item['type']}")
        decoded.append({"role": message["role"], "content": content})
    return decoded


def extract_images_from_messages(messages):
    images = []
    for message in messages:
        for item in message["content"]:
            if item["type"] == "image":
                images.append(item["image"])
    return images


def to_vllm_chat_messages(messages):
    converted = []
    for message in messages:
        content = []
        for item in message["content"]:
            if item["type"] == "text":
                content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image":
                content.append({"type": "image_pil", "image_pil": item["image"]})
            else:
                raise ValueError(f"Unsupported message content type for vLLM chat: {item['type']}")
        converted.append({"role": message["role"], "content": content})
    return converted


def _collect_step_s2_mm_debug(processed_prompt, bundle) -> dict:
    mm_placeholders = processed_prompt.get("mm_placeholders") or {}
    mm_kwargs = processed_prompt.get("mm_kwargs") or {}
    image_placeholders = mm_placeholders.get("image") or []
    image_kwargs = mm_kwargs.get("image") or []
    none_mm_kwargs = sum(item is None for item in image_kwargs)

    mm_features = bundle.mm_features
    mm_features_len = len(mm_features) if mm_features is not None else None
    mm_features_type = type(mm_features).__name__
    mm_features_num_data_none = (
        sum(getattr(feature, "data", None) is None for feature in mm_features)
        if mm_features is not None
        else None
    )

    feature0_offset = None
    feature0_length = None
    if mm_features:
        feature0_offset = int(mm_features[0].mm_position.offset)
        feature0_length = int(mm_features[0].mm_position.length)

    return {
        "processed_prompt_type": processed_prompt.get("type"),
        "processed_prompt_num_image_placeholders": len(image_placeholders),
        "processed_prompt_num_image_kwargs": len(image_kwargs),
        "processed_prompt_num_image_kwargs_none": none_mm_kwargs,
        "bundle_mm_features_is_none": mm_features is None,
        "bundle_mm_features_len": mm_features_len,
        "bundle_mm_features_type": mm_features_type,
        "bundle_mm_features_num_data_none": mm_features_num_data_none,
        "bundle_mm_feature0_offset": feature0_offset,
        "bundle_mm_feature0_length": feature0_length,
    }


def _bundle_mm_features_have_missing_payload(bundle) -> bool:
    mm_features = getattr(bundle, "mm_features", None)
    if not mm_features:
        return False
    return any(getattr(feature, "data", None) is None for feature in mm_features)


def _aggregate_vllm_worker_runtime_stats(worker_reports) -> dict:
    reports = [report for report in (worker_reports or []) if isinstance(report, dict)]
    if not reports:
        return {
            "available_kv_cache_memory_bytes": None,
            "requested_memory_bytes": None,
            "peak_activation_memory_bytes": None,
            "cudagraph_memory_estimate_bytes": None,
            "effective_kv_budget_bytes": None,
            "gpu_memory_utilization": None,
            "max_model_len": None,
            "num_gpu_blocks": None,
            "worker_count": 0,
        }

    def _sum_optional_int(key: str):
        values = [report.get(key) for report in reports if report.get(key) is not None]
        return int(sum(values)) if values else None

    head = reports[0]
    return {
        "available_kv_cache_memory_bytes": _sum_optional_int("available_kv_cache_memory_bytes"),
        "requested_memory_bytes": _sum_optional_int("requested_memory_bytes"),
        "peak_activation_memory_bytes": _sum_optional_int("peak_activation_memory_bytes"),
        "cudagraph_memory_estimate_bytes": _sum_optional_int("cudagraph_memory_estimate_bytes"),
        "effective_kv_budget_bytes": _sum_optional_int("effective_kv_budget_bytes"),
        "gpu_memory_utilization": head.get("gpu_memory_utilization"),
        "max_model_len": head.get("max_model_len"),
        "num_gpu_blocks": _sum_optional_int("num_gpu_blocks"),
        "worker_count": len(reports),
    }


def _make_runtime_stats_error_payload(exc: Exception) -> dict:
    return {
        "available_kv_cache_memory_bytes": None,
        "requested_memory_bytes": None,
        "peak_activation_memory_bytes": None,
        "cudagraph_memory_estimate_bytes": None,
        "effective_kv_budget_bytes": None,
        "gpu_memory_utilization": None,
        "max_model_len": None,
        "num_gpu_blocks": None,
        "worker_count": 0,
        "collect_error": repr(exc),
    }


class DualVLNSingleVLLMRunner:
    def __init__(
        self,
        model_path: str,
        *,
        hf_model_path: Optional[str] = None,
        dtype: str = "auto",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.45,
        limit_mm_per_prompt_image: int = 16,
        tensor_parallel_size: int = 1,
        model_impl: str = "auto",
        latent_backend: Optional[str] = None,
        trust_remote_code: bool = False,
        enforce_eager: bool = False,
        seed: int = 0,
        compilation_config: Optional[dict] = None,
    ):
        from vllm import LLM, SamplingParams

        self.model_path = model_path
        self.hf_model_path = hf_model_path or model_path
        self.processor = AutoProcessor.from_pretrained(
            self.hf_model_path,
            trust_remote_code=trust_remote_code,
        )
        self.processor.tokenizer.padding_side = "left"
        self.latent_queries = _load_latent_queries_tensor(self.hf_model_path)
        self.n_query = int(self.latent_queries.shape[0])
        self.traj_token_index = TRAJ_TOKEN_INDEX
        self.model_impl = model_impl
        requested_latent_backend = latent_backend or "shared_engine_forward"
        if requested_latent_backend == "vllm_hidden":
            requested_latent_backend = "shared_engine_forward"
        self.latent_backend = requested_latent_backend
        self._hidden_latents_runner = None
        self._hidden_latents_runner_kwargs = {
            "model_path": model_path,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "limit_mm_per_prompt_image": limit_mm_per_prompt_image,
            "dtype": dtype,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": trust_remote_code,
            "enforce_eager": enforce_eager,
        }
        self.sampling_params = SamplingParams(max_tokens=128, temperature=0.0)
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_model_len=max_model_len,
            compilation_config=compilation_config,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": limit_mm_per_prompt_image},
            model_impl=model_impl,
            trust_remote_code=trust_remote_code,
            enforce_eager=enforce_eager,
            seed=seed,
            disable_log_stats=True,
            async_scheduling=False,
        )
        self._last_step_s2_engine_request = None
        self._runtime_stats_refresh_mode = os.environ.get(
            "INTERNNAV_VLLM_RUNTIME_STATS_REFRESH_MODE",
            "init_once",
        )
        self._cached_vllm_runtime_stats = self._fetch_vllm_runtime_stats_once()

    def _fetch_vllm_runtime_stats_once(self) -> dict:
        try:
            worker_reports = self.llm.collective_rpc("get_internnav_runtime_stats")
        except Exception as exc:
            return _make_runtime_stats_error_payload(exc)
        return _aggregate_vllm_worker_runtime_stats(worker_reports)

    def _get_cached_vllm_runtime_stats(self) -> dict:
        if self._runtime_stats_refresh_mode == "manual":
            return dict(self._cached_vllm_runtime_stats)
        if self._runtime_stats_refresh_mode == "init_once":
            return dict(self._cached_vllm_runtime_stats)
        if self._runtime_stats_refresh_mode == "lazy_once":
            if self._cached_vllm_runtime_stats is None:
                self._cached_vllm_runtime_stats = self._fetch_vllm_runtime_stats_once()
            return dict(self._cached_vllm_runtime_stats)
        raise ValueError(
            "Unsupported INTERNNAV_VLLM_RUNTIME_STATS_REFRESH_MODE: "
            f"{self._runtime_stats_refresh_mode}"
        )

    def _generate_latents_via_shared_engine(self, bundle):
        prompt_embeds_mode = os.environ.get(
            "INTERNNAV_NATIVE_PREFILL_PROMPT_EMBEDS_MODE",
            "full_prompt",
        )
        prompt_embeds_soft_suffix_len = None

        if prompt_embeds_mode == "full_prompt":
            if (
                bundle.prompt_embeds is None
                or int(bundle.prompt_embeds.shape[0]) != len(bundle.prefill_token_ids)
            ):
                bundle.prompt_embeds = self.llm.apply_model(
                    functools.partial(
                        _build_native_latent_prefill_prompt_embeds,
                        prompt_token_ids=bundle.prefill_token_ids,
                        latent_queries_cpu=bundle.latent_queries,
                        mm_features=bundle.mm_features,
                    )
                )[0]
        elif prompt_embeds_mode == "soft_suffix_only":
            if (
                bundle.prompt_embeds is None
                or int(bundle.prompt_embeds.shape[0]) != self.n_query
            ):
                bundle.prompt_embeds = self.llm.apply_model(
                    functools.partial(
                        _build_native_latent_prefill_suffix_prompt_embeds,
                        latent_queries_cpu=bundle.latent_queries,
                    )
                )[0]
            prompt_embeds_soft_suffix_len = self.n_query
        else:
            raise ValueError(
                "Unsupported INTERNNAV_NATIVE_PREFILL_PROMPT_EMBEDS_MODE: "
                f"{prompt_embeds_mode}"
            )

        return self.llm.generate_latents_native_prefill(
            prompt_token_ids=bundle.prefill_token_ids,
            prompt_embeds=bundle.prompt_embeds,
            mm_features=bundle.mm_features,
            n_query=self.n_query,
            prompt_embeds_soft_suffix_len=prompt_embeds_soft_suffix_len,
        )[0]

    def _ensure_hidden_latents_runner(self):
        if self._hidden_latents_runner is None:
            self._hidden_latents_runner = VLLMHiddenLatentsRunner(
                **self._hidden_latents_runner_kwargs,
            )
        return self._hidden_latents_runner

    def step_s2(self, messages, *, max_new_tokens: int = 128):
        from vllm import SamplingParams
        from vllm.outputs import RequestOutput

        total_start = time.perf_counter()
        preprocess_start = total_start
        vllm_messages = to_vllm_chat_messages(messages)
        processed_prompt = self.llm._preprocess_chat_one(vllm_messages)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        generate_start = time.perf_counter()
        sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
        continuation_enabled = (
            os.environ.get("INTERNNAV_DUALVLN_LATENT_CONTINUATION", "1") != "0"
        )
        continuation_backend_eligible = self.latent_backend in (
            "legacy_custom_forward",
            "shared_engine_forward",
        )
        continuation_attempted = continuation_enabled and continuation_backend_eligible
        continuation_result = None
        continuation_setup_error = None
        if continuation_attempted:
            try:
                outputs, continuation_result = (
                    self.llm._run_request_with_dualvln_latent_continuation(
                        prompt=processed_prompt,
                        params=sampling_params,
                        suffix_token_ids=[self.traj_token_index] * self.n_query,
                        use_tqdm=False,
                    )
                )
            except Exception as exc:
                continuation_setup_error = (
                    f"continuation_setup_error:{type(exc).__name__}:{exc}"
                )
                logger.warning(
                    "DualVLN step_s2 could not arm same-request latent continuation; "
                    "falling back to normal text generate path: %s",
                    continuation_setup_error,
                )
                outputs = self.llm._render_and_run_requests(
                    prompts=(processed_prompt,),
                    params=[sampling_params],
                    output_type=RequestOutput,
                    use_tqdm=False,
                )
        else:
            outputs = self.llm._render_and_run_requests(
                prompts=(processed_prompt,),
                params=[sampling_params],
                output_type=RequestOutput,
                use_tqdm=False,
            )
        generate_ms = (time.perf_counter() - generate_start) * 1000.0
        request_output = outputs[0]
        self._last_step_s2_engine_request = self.llm.llm_engine.pop_debug_engine_core_request(
            request_output.request_id
        )
        completion = request_output.outputs[0]
        llm_output = completion.text
        prompt_token_ids = list(request_output.prompt_token_ids or [])
        generated_token_ids = list(completion.token_ids)
        input_images = extract_images_from_messages(messages)
        image_placeholders = (processed_prompt.get("mm_placeholders") or {}).get("image") or []

        runtime_metrics = {
            "preprocess_ms": preprocess_ms,
            "generate_ms": generate_ms,
            "bundle_build_ms": 0.0,
            "mm_attach_ms": 0.0,
            "latent_prefill_ms": 0.0,
            "total_ms": 0.0,
            "prompt_token_count": int(len(prompt_token_ids)),
            "generated_token_count": int(len(generated_token_ids)),
            "prefill_token_count": None,
            "n_query": int(self.n_query),
            "num_images": int(len(input_images)),
            "mm_feature_count": int(len(image_placeholders)),
            "prefill_share_of_total": None,
            "latent_prefill_share_of_total": None,
            "decode_share_of_total": None,
            "same_request_continuation_enabled": bool(continuation_enabled),
            "same_request_continuation_attempted": bool(continuation_attempted),
            "same_request_continuation_used": False,
            "same_request_suffix_len": int(self.n_query),
            "same_request_external_request_id": request_output.request_id,
            "same_request_internal_request_id": (
                self._last_step_s2_engine_request.request_id
                if self._last_step_s2_engine_request is not None
                else None
            ),
            "same_request_request_ids_match": None,
            "same_request_fallback_reason": None,
            "same_request_result_missing": False,
            "latent_path": (
                "same_request_continuation_attempt"
                if continuation_attempted
                else "legacy_latent_path"
            ),
        }
        if continuation_result is not None:
            runtime_metrics["same_request_internal_request_id"] = (
                continuation_result.internal_request_id
            )
            runtime_metrics["same_request_request_ids_match"] = bool(
                continuation_result.external_request_id == request_output.request_id
            )
            runtime_metrics["same_request_fallback_reason"] = (
                continuation_result.fallback_reason
            )
        elif continuation_attempted:
            runtime_metrics["same_request_result_missing"] = True
            runtime_metrics["same_request_fallback_reason"] = "continuation_result_missing"
        if continuation_setup_error is not None:
            runtime_metrics["same_request_fallback_reason"] = continuation_setup_error

        result = {
            "llm_output": llm_output,
            "prompt_token_ids": prompt_token_ids,
            "generated_token_ids": generated_token_ids,
            "pixel_goal": None,
            "latents": None,
            "runtime_metrics": runtime_metrics,
            "vllm_kv_cache": self._get_cached_vllm_runtime_stats(),
        }

        if not re.search(r"\d", llm_output):
            runtime_metrics["total_ms"] = (time.perf_counter() - total_start) * 1000.0
            return result

        coord = [int(c) for c in re.findall(r"\d+", llm_output)]
        if len(coord) < 2:
            runtime_metrics["total_ms"] = (time.perf_counter() - total_start) * 1000.0
            return result

        bundle_start = time.perf_counter()
        bundle = build_latents_request_bundle(
            processor=self.processor,
            messages=messages,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
            input_images=input_images,
            latent_queries=self.latent_queries,
            traj_token_index=self.traj_token_index,
            n_query=self.n_query,
        )
        runtime_metrics["bundle_build_ms"] = (time.perf_counter() - bundle_start) * 1000.0
        mm_attach_source = os.environ.get("INTERNNAV_STEP_S2_MM_SOURCE", "engine_core_request")
        mm_attach_start = time.perf_counter()
        if mm_attach_source == "engine_core_request":
            attach_explicit_mm_metadata_from_engine_core_request(
                bundle,
                self._last_step_s2_engine_request,
            )
        elif mm_attach_source == "processed_prompt":
            attach_explicit_mm_metadata_from_processed_inputs(bundle, processed_prompt)
        elif mm_attach_source == "llm_input_processor":
            attach_explicit_mm_metadata(bundle, self.llm)
        else:
            raise ValueError(f"Unsupported INTERNNAV_STEP_S2_MM_SOURCE: {mm_attach_source}")
        mm_attach_backfill = None
        if mm_attach_source == "engine_core_request" and _bundle_mm_features_have_missing_payload(bundle):
            attach_explicit_mm_metadata_from_processed_inputs(bundle, processed_prompt)
            mm_attach_backfill = "processed_prompt_missing_data"
        if _bundle_mm_features_have_missing_payload(bundle):
            attach_explicit_mm_metadata(bundle, self.llm)
            if mm_attach_backfill is None:
                mm_attach_backfill = "llm_input_processor_missing_data"
            else:
                mm_attach_backfill = f"{mm_attach_backfill}+llm_input_processor_missing_data"
        runtime_metrics["mm_attach_ms"] = (time.perf_counter() - mm_attach_start) * 1000.0
        mm_debug = _collect_step_s2_mm_debug(processed_prompt, bundle)
        mm_debug["mm_attach_source"] = mm_attach_source
        mm_debug["mm_attach_backfill"] = mm_attach_backfill
        runtime_metrics["prefill_token_count"] = int(len(bundle.prefill_token_ids))
        runtime_metrics["mm_feature_count"] = int(len(bundle.mm_features or []))
        if os.environ.get("INTERNNAV_DEBUG_STEP_S2_MM"):
            print(
                "[DualVLN step_s2 mm_debug] "
                + json.dumps(mm_debug, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
        latent_prefill_start = time.perf_counter()
        continuation_latents = (
            continuation_result.latents
            if continuation_result is not None
            and bool(continuation_result.continuation_used)
            and continuation_result.latents is not None
            else None
        )
        if continuation_latents is not None:
            runtime_metrics["same_request_continuation_used"] = True
            runtime_metrics["latent_path"] = "same_request_continuation"
            latents = continuation_latents
            runtime_metrics["prefill_token_count"] = int(
                len(prompt_token_ids) + len(generated_token_ids) + self.n_query
            )
            logger.info(
                "DualVLN step_s2 used same-request latent continuation for request %s "
                "(internal=%s, suffix_len=%d).",
                request_output.request_id,
                continuation_result.internal_request_id,
                self.n_query,
            )
        elif self.latent_backend == "transformers_backend_apply_model":
            runtime_metrics["latent_path"] = "transformers_backend_apply_model"
            latents = self.llm.apply_model(
                functools.partial(
                    _generate_latents_via_transformers_backend_apply_model,
                    full_output_ids_cpu=bundle.full_output_ids,
                    pixel_values_cpu=bundle.pixel_values,
                    image_grid_thw_cpu=bundle.image_grid_thw,
                )
            )[0]
        elif self.latent_backend in ("legacy_custom_forward", "shared_engine_forward"):
            runtime_metrics["latent_path"] = "native_latent_prefill_fallback"
            if continuation_attempted:
                logger.info(
                    "DualVLN step_s2 falling back to native latent prefill for request %s: %s",
                    request_output.request_id,
                    runtime_metrics["same_request_fallback_reason"],
                )
            latents = self._generate_latents_via_shared_engine(bundle)
        elif self.latent_backend == "vllm_hidden_separate_llm":
            runtime_metrics["latent_path"] = "vllm_hidden_separate_llm"
            latents = self._ensure_hidden_latents_runner().generate_latents_from_bundle(bundle)
        else:
            raise ValueError(f"Unsupported latent_backend: {self.latent_backend}")
        runtime_metrics["latent_prefill_ms"] = (time.perf_counter() - latent_prefill_start) * 1000.0
        runtime_metrics["total_ms"] = (time.perf_counter() - total_start) * 1000.0
        if runtime_metrics["total_ms"] > 0:
            runtime_metrics["latent_prefill_share_of_total"] = (
                runtime_metrics["latent_prefill_ms"] / runtime_metrics["total_ms"]
            )

        result["pixel_goal"] = [int(coord[1]), int(coord[0])]
        result["latents"] = latents
        result["debug_mm"] = mm_debug
        return result


class DualVLNSingleVLLMHTTPClient:
    def __init__(self, base_url: str, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.image_transport_mode = _get_image_transport_mode()

    def step_s2(self, messages, *, max_new_tokens: int = 128, target_device=None, target_dtype=None):
        client_total_start = time.perf_counter()
        encode_start = client_total_start
        encoded_messages, encode_state = encode_messages(
            messages,
            image_transport_mode=self.image_transport_mode,
        )
        client_encode_messages_ms = (time.perf_counter() - encode_start) * 1000.0
        payload = {
            "messages": encoded_messages,
            "max_new_tokens": int(max_new_tokens),
            "image_transport_mode": encode_state["image_transport_mode"],
        }
        data = None
        try:
            http_start = time.perf_counter()
            resp = http_requests.post(
                f"{self.base_url}/dualvln/step_s2",
                json=payload,
                timeout=self.timeout,
            )
            client_http_post_ms = (time.perf_counter() - http_start) * 1000.0
            resp.raise_for_status()
            response_json_start = time.perf_counter()
            data = resp.json()
            client_response_json_ms = (time.perf_counter() - response_json_start) * 1000.0
            latents = None
            decode_latents_start = time.perf_counter()
            if data.get("latents") is not None:
                latents = decode_tensor_from_b64(data["latents"])
                if target_dtype is not None:
                    latents = latents.to(dtype=target_dtype)
                if target_device is not None:
                    latents = latents.to(device=target_device)
            client_decode_latents_ms = (time.perf_counter() - decode_latents_start) * 1000.0
            client_total_ms = (time.perf_counter() - client_total_start) * 1000.0
            transport_metrics = dict(data.get("transport_metrics") or {})
            transport_metrics.update(
                {
                    "client_encode_messages_ms": client_encode_messages_ms,
                    "client_http_post_ms": client_http_post_ms,
                    "client_response_json_ms": client_response_json_ms,
                    "client_decode_latents_ms": client_decode_latents_ms,
                    "client_total_ms": client_total_ms,
                    "image_transport_mode": encode_state["image_transport_mode"],
                    "image_transport_count": int(encode_state["image_count"]),
                    "image_transport_payload_bytes": int(encode_state["image_payload_bytes"]),
                }
            )
            runtime_total_ms = ((data.get("runtime_metrics") or {}).get("total_ms"))
            server_total_ms = transport_metrics.get("server_total_ms")
            transport_metrics["client_side_overhead_ms"] = (
                float(max(client_total_ms - server_total_ms, 0.0))
                if server_total_ms is not None
                else None
            )
            transport_metrics["end_to_end_transport_overhead_ms"] = (
                float(max(client_total_ms - runtime_total_ms, 0.0))
                if runtime_total_ms is not None
                else None
            )
            data["latents"] = latents
            data["transport_metrics"] = transport_metrics
            return data
        finally:
            _cleanup_client_shared_memory_handles(encode_state["shared_memory_handles"])

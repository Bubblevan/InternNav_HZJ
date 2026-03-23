import base64
import functools
import io
import json
import os
import re
from pathlib import Path
from typing import Optional

import requests as http_requests
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoProcessor

from internnav.model.utils.latents_request import build_latents_request_bundle
from internnav.model.utils.vllm_hidden_latents import VLLMHiddenLatentsRunner

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

TRAJ_TOKEN_INDEX = 151667


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

        position_ids, _ = _compute_qwen2_5_vl_rope_index(
            input_ids,
            config=model.config,
            image_grid_thw=image_grid_thw_cpu.to(device=device) if image_grid_thw_cpu is not None else None,
        )
        hidden_states = model.forward(
            input_ids=None,
            positions=position_ids[:, 0, :],
            inputs_embeds=embeds,
        )
        return hidden_states[-n_query:, :].unsqueeze(0).cpu()


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
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_pil_image_from_b64(payload: str) -> Image.Image:
    raw = base64.b64decode(payload.encode("utf-8"))
    return Image.open(io.BytesIO(raw)).convert("RGB")


def encode_messages(messages):
    encoded = []
    for message in messages:
        content = []
        for item in message["content"]:
            if item["type"] == "text":
                content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image":
                content.append({"type": "image", "image": encode_pil_image_to_b64(item["image"])})
            else:
                raise ValueError(f"Unsupported message content type: {item['type']}")
        encoded.append({"role": message["role"], "content": content})
    return encoded


def decode_messages(messages):
    decoded = []
    for message in messages:
        content = []
        for item in message["content"]:
            if item["type"] == "text":
                content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image":
                content.append({"type": "image", "image": decode_pil_image_from_b64(item["image"])})
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
        enforce_eager: bool = True,
        seed: int = 0,
    ):
        from vllm import LLM, SamplingParams

        self.model_path = model_path
        self.hf_model_path = hf_model_path or model_path
        self.processor = AutoProcessor.from_pretrained(self.hf_model_path)
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
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": limit_mm_per_prompt_image},
            model_impl=model_impl,
            trust_remote_code=trust_remote_code,
            enforce_eager=enforce_eager,
            seed=seed,
            disable_log_stats=True,
        )

    def _generate_latents_via_shared_engine(self, bundle):
        return self.llm.apply_model(
            functools.partial(
                _generate_latents_from_vllm_model,
                prompt_token_ids=bundle.full_output_token_ids,
                pixel_values_cpu=bundle.pixel_values,
                image_grid_thw_cpu=bundle.image_grid_thw,
                latent_queries_cpu=bundle.latent_queries,
                traj_token_index=self.traj_token_index,
                n_query=self.n_query,
            )
        )[0]

    def _ensure_hidden_latents_runner(self):
        if self._hidden_latents_runner is None:
            self._hidden_latents_runner = VLLMHiddenLatentsRunner(
                **self._hidden_latents_runner_kwargs,
            )
        return self._hidden_latents_runner

    def step_s2(self, messages, *, max_new_tokens: int = 128):
        from vllm import SamplingParams

        vllm_messages = to_vllm_chat_messages(messages)
        outputs = self.llm.chat(
            vllm_messages,
            SamplingParams(max_tokens=max_new_tokens, temperature=0.0),
            use_tqdm=False,
        )
        request_output = outputs[0]
        completion = request_output.outputs[0]
        llm_output = completion.text
        prompt_token_ids = list(request_output.prompt_token_ids or [])
        generated_token_ids = list(completion.token_ids)

        result = {
            "llm_output": llm_output,
            "prompt_token_ids": prompt_token_ids,
            "generated_token_ids": generated_token_ids,
            "pixel_goal": None,
            "latents": None,
        }

        if not re.search(r"\d", llm_output):
            return result

        coord = [int(c) for c in re.findall(r"\d+", llm_output)]
        if len(coord) < 2:
            return result

        input_images = extract_images_from_messages(messages)
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
        if self.latent_backend == "transformers_backend_apply_model":
            latents = self.llm.apply_model(
                functools.partial(
                    _generate_latents_via_transformers_backend_apply_model,
                    full_output_ids_cpu=bundle.full_output_ids,
                    pixel_values_cpu=bundle.pixel_values,
                    image_grid_thw_cpu=bundle.image_grid_thw,
                )
            )[0]
        elif self.latent_backend in ("legacy_custom_forward", "shared_engine_forward"):
            latents = self._generate_latents_via_shared_engine(bundle)
        elif self.latent_backend == "vllm_hidden_separate_llm":
            latents = self._ensure_hidden_latents_runner().generate_latents_from_bundle(bundle)
        else:
            raise ValueError(f"Unsupported latent_backend: {self.latent_backend}")

        result["pixel_goal"] = [int(coord[1]), int(coord[0])]
        result["latents"] = latents
        return result


class DualVLNSingleVLLMHTTPClient:
    def __init__(self, base_url: str, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def step_s2(self, messages, *, max_new_tokens: int = 128, target_device=None, target_dtype=None):
        payload = {
            "messages": encode_messages(messages),
            "max_new_tokens": int(max_new_tokens),
        }
        resp = http_requests.post(
            f"{self.base_url}/dualvln/step_s2",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        latents = None
        if data.get("latents") is not None:
            latents = decode_tensor_from_b64(data["latents"])
            if target_dtype is not None:
                latents = latents.to(dtype=target_dtype)
            if target_device is not None:
                latents = latents.to(device=target_device)
        data["latents"] = latents
        return data

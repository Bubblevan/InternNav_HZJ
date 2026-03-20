import functools
import io
import os
import time
import base64
from pathlib import Path

import torch
import requests as http_requests
from PIL import Image

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def _build_hf_like_prompt_embeds(
    model,
    prompt_token_ids,
    pixel_values_cpu,
    image_grid_thw_cpu,
    latent_queries_cpu,
):
    device = next(model.parameters()).device
    input_ids = torch.tensor(prompt_token_ids, device=device, dtype=torch.long)

    with torch.inference_mode():
        embeds = model.embed_input_ids(input_ids).clone()

        if pixel_values_cpu is not None and image_grid_thw_cpu is not None:
            pixel_values = pixel_values_cpu.to(device=device, dtype=model.visual.dtype)
            image_grid_thw = image_grid_thw_cpu.to(device=device)
            multimodal_embeddings = model.embed_multimodal(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
            flat_mm_embeddings = (
                torch.cat(list(multimodal_embeddings), dim=0)
                if multimodal_embeddings
                else None
            )
            if flat_mm_embeddings is not None:
                image_idx = input_ids == model.config.image_token_id
                image_token_count = int(image_idx.sum().item())
                embeds[image_idx] = flat_mm_embeddings[:image_token_count].to(embeds.dtype)

        latent_queries = latent_queries_cpu.to(device=device, dtype=embeds.dtype)
        embeds[-latent_queries.shape[0] :] = latent_queries

    return embeds.cpu()


def _set_dump_env(dump_dir: Path, dump_prefix: str):
    os.environ["VLLM_DEBUG_DUMP_DIR"] = str(dump_dir)
    os.environ["VLLM_DEBUG_DUMP_PREFIX"] = dump_prefix
    os.environ["VLLM_DEBUG_DUMP_FULL_TENSORS"] = "1"
    os.environ["VLLM_DEBUG_DUMP_SLICE_ROWS"] = "8"


def _load_dump_records(dump_dir: Path, dump_prefix: str, tag: str):
    records = []
    for path in sorted(dump_dir.glob(f"{dump_prefix}_*_{tag}.pt")):
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception:
            continue
        payload["_path"] = str(path)
        records.append(payload)
    return records


def _window_records(records, start_ts, end_ts, tensor_key):
    candidates = []
    for record in records:
        ts = record.get("ts_ns")
        if ts is None or not (start_ts <= ts <= end_ts):
            continue
        tensor = record.get("tensors", {}).get(tensor_key)
        if not torch.is_tensor(tensor):
            continue
        candidates.append(record)
    candidates.sort(key=lambda r: r.get("ts_ns", 0))
    return candidates


def _aggregate_records(records, tensor_key):
    if not records:
        return None
    tensors = [record["tensors"][tensor_key] for record in records]
    if not all(torch.is_tensor(t) and t.ndim == 2 for t in tensors):
        return None
    hidden_size = tensors[0].shape[1]
    if not all(t.shape[1] == hidden_size for t in tensors):
        return None
    return torch.cat(tensors, dim=0)


class VLLMHiddenLatentsRunner:
    def __init__(
        self,
        model_path: str,
        dump_dir: str,
        *,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.45,
        limit_mm_per_prompt_image: int = 16,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = False,
        enforce_eager: bool = True,
    ):
        self.model_path = model_path
        self.dump_dir = Path(dump_dir)
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.limit_mm_per_prompt_image = limit_mm_per_prompt_image
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.trust_remote_code = trust_remote_code
        self.enforce_eager = enforce_eager
        self._llm = None

    def _ensure_llm(self):
        if self._llm is not None:
            return self._llm
        from vllm import LLM

        self._llm = LLM(
            model=self.model_path,
            runner="pooling",
            convert="embed",
            enable_prompt_embeds=True,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            limit_mm_per_prompt={"image": self.limit_mm_per_prompt_image},
            trust_remote_code=self.trust_remote_code,
            enforce_eager=self.enforce_eager,
            disable_log_stats=True,
        )
        return self._llm

    def generate_latents(
        self,
        *,
        output_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_images,
        latent_queries: torch.Tensor,
        traj_token_index: int,
        n_query: int,
        target_device=None,
        target_dtype=None,
    ):
        if output_ids.ndim != 2 or output_ids.shape[0] != 1:
            raise ValueError(f"Only batch size 1 is supported, got shape={tuple(output_ids.shape)}")

        prompt_token_ids = output_ids[0].detach().cpu().tolist() + [traj_token_index] * n_query
        dump_prefix = f"generate_latents_backend_{int(time.time() * 1000)}_{time.time_ns()}"
        _set_dump_env(self.dump_dir, dump_prefix)

        llm = self._ensure_llm()
        prompt_embeds = llm.apply_model(
            functools.partial(
                _build_hf_like_prompt_embeds,
                prompt_token_ids=prompt_token_ids,
                pixel_values_cpu=pixel_values.detach().cpu(),
                image_grid_thw_cpu=image_grid_thw.detach().cpu(),
                latent_queries_cpu=latent_queries.detach().cpu(),
            )
        )[0]

        from vllm.inputs.data import EmbedsPrompt

        start_ts = time.time_ns()
        llm.encode(
            [
                EmbedsPrompt(
                    prompt_embeds=prompt_embeds,
                    prompt_token_ids=prompt_token_ids,
                    multi_modal_data={"image": input_images} if input_images is not None else None,
                )
            ],
            pooling_task="token_embed",
            use_tqdm=False,
        )
        end_ts = time.time_ns()

        records_post = _load_dump_records(self.dump_dir, dump_prefix, "gpu_model_runner_actual_post_forward")
        post_records = _window_records(records_post, start_ts, end_ts, "hidden_states")
        post_agg = _aggregate_records(post_records, "hidden_states")
        if post_agg is None:
            raise RuntimeError(
                "Failed to collect vLLM hidden states from debug dump. "
                "Check patched vLLM dump hooks and the dump directory."
            )

        latents = post_agg[-n_query:, :]
        if target_dtype is not None:
            latents = latents.to(dtype=target_dtype)
        if target_device is not None:
            latents = latents.to(device=target_device)
        return latents.unsqueeze(0)


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


class VLLMHiddenLatentsHTTPClient:
    def __init__(self, base_url: str, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate_latents(
        self,
        *,
        output_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_images,
        latent_queries: torch.Tensor,
        traj_token_index: int,
        n_query: int,
        target_device=None,
        target_dtype=None,
    ):
        payload = {
            "output_ids": encode_tensor_to_b64(output_ids),
            "pixel_values": encode_tensor_to_b64(pixel_values),
            "image_grid_thw": encode_tensor_to_b64(image_grid_thw),
            "latent_queries": encode_tensor_to_b64(latent_queries),
            "traj_token_index": int(traj_token_index),
            "n_query": int(n_query),
            "input_images": [encode_pil_image_to_b64(image) for image in (input_images or [])],
        }
        resp = http_requests.post(
            f"{self.base_url}/generate_latents",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        latents = decode_tensor_from_b64(data["latents"])
        if target_dtype is not None:
            latents = latents.to(dtype=target_dtype)
        if target_device is not None:
            latents = latents.to(device=target_device)
        return latents

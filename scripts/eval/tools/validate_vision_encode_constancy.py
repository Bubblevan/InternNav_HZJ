import argparse
import gc
import json
import os
from pathlib import Path
import re
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

CONJUNCTIONS = [
    "you can see ",
    "in front of you is ",
    "there is ",
    "you can spot ",
    "you are toward the ",
    "ahead of you is ",
    "in your sight is ",
]
PROMPT_VARIANTS = {
    "full": (
        "You are an autonomous navigation assistant. Your task is to <instruction>. "
        "Where should you go next to stay on track? Please output the next waypoint's "
        "coordinates in the image. Please output STOP when you have successfully completed the task."
    ),
    "short": (
        "Your task is to <instruction>. "
        "Output the next waypoint coordinates in the image, or output STOP if the task is complete."
    ),
    "minimal": "<instruction> Output next waypoint coordinates or STOP.",
}
DEFAULT_IMAGE_TOKEN = "<image>"


def normalize_image_grid_thw(image_grid_thw):
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate whether vision encode latency stays approximately constant "
            "for fixed-history S2 prompts under native HF or patched vLLM."
        )
    )
    parser.add_argument("--backend", choices=["hf", "patched_vllm"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--details-output", default=None)
    parser.add_argument("--base-path", default=None)
    parser.add_argument("--hf-model-path", required=True)
    parser.add_argument("--vllm-model-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--num-history", type=int, default=8)
    parser.add_argument("--max-episodes", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--prompt-variant", choices=sorted(PROMPT_VARIANTS.keys()), default="full")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--ignore-manifest-history", action="store_true")
    parser.add_argument(
        "--allow-short-history",
        action="store_true",
        help="Include steps with fewer than num_history historical images.",
    )
    parser.add_argument(
        "--attn-backend",
        choices=["flash_attention_2", "sdpa", "eager"],
        default="flash_attention_2",
        help="HF-only attention backend",
    )
    parser.add_argument(
        "--processor-use-fast",
        choices=["auto", "true", "false"],
        default="auto",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--limit-mm-per-prompt-image", type=int, default=16)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quantization",
        default=None,
        choices=["awq", "fp8", "gptq", "squeezellm", "marlin", "gguf", "smoothquant"],
        help="Patched vLLM only.",
    )
    parser.add_argument(
        "--enable-mm-encoder-compile",
        action="store_true",
        help="Patched vLLM only.",
    )
    return parser.parse_args()


def default_details_output(output_path):
    stem, _ = os.path.splitext(output_path)
    return f"{stem}.jsonl"


def percentile(values, q):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(values):
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "stdev": 0.0,
            "cv": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    mean = float(np.mean(values))
    stdev = float(np.std(np.asarray(values, dtype=np.float64), ddof=0))
    return {
        "count": len(values),
        "mean": mean,
        "stdev": stdev,
        "cv": float(stdev / mean) if abs(mean) > 1e-12 else 0.0,
        "min": float(np.min(values)),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "max": float(np.max(values)),
    }


def _close_images(images):
    for image in images:
        try:
            image.close()
        except Exception:
            pass


def _cuda_sync_if_needed(device):
    if isinstance(device, str):
        use_cuda = device.startswith("cuda")
    else:
        use_cuda = isinstance(device, torch.device) and device.type == "cuda"
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_manifest(path, base_path=None):
    grouped = defaultdict(list)
    old_prefix = "./logs/"
    if base_path is not None:
        base_path = base_path.rstrip("/") + "/"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            path_fields = ["rgb_path", "depth_path", "lookdown_rgb_path", "lookdown_depth_path"]
            for field in path_fields:
                if field in item and base_path is not None and item[field].startswith(old_prefix):
                    item[field] = item[field].replace(old_prefix, base_path)
                if field in item:
                    item[field] = os.path.abspath(item[field])
            grouped[(item["scene_id"], int(item["episode_id"]))].append(item)
    for items in grouped.values():
        items.sort(key=lambda x: x["step_id"])
    ordered = dict(sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])))
    return ordered


def _load_rgb_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def build_messages(instruction, episode_steps, current_index, num_history, prompt_variant, ignore_manifest_history=False):
    sources = [{"from": "human", "value": PROMPT_VARIANTS[prompt_variant]}, {"from": "gpt", "value": ""}]
    sources[0]["value"] = sources[0]["value"].replace("<instruction>.", instruction[:-1])
    sources[0]["value"] = sources[0]["value"].replace("<instruction>", instruction.strip())

    manifest_history = None if ignore_manifest_history else episode_steps[current_index].get("history_frame_indices", None)
    if manifest_history:
        history_indices = sorted(manifest_history)
    elif current_index == 0 or num_history == 0:
        history_indices = []
    else:
        history_indices = np.unique(np.linspace(0, current_index - 1, num_history, dtype=np.int32)).tolist()

    if history_indices:
        placeholder = (DEFAULT_IMAGE_TOKEN + "\n") * len(history_indices)
        sources[0]["value"] += f" These are your historical observations: {placeholder}."

    history_indices = sorted(history_indices)
    input_images = [_load_rgb_image(episode_steps[i]["rgb_path"]) for i in history_indices]
    input_images.append(_load_rgb_image(episode_steps[current_index]["rgb_path"]))
    input_img_id = 0

    prompt = CONJUNCTIONS[0] + DEFAULT_IMAGE_TOKEN
    sources[0]["value"] += f" {prompt}."
    parts = re.split(r"(<image>)", sources[0]["value"])
    content = []
    for part in parts:
        if not part:
            continue
        if part == DEFAULT_IMAGE_TOKEN:
            content.append({"type": "image", "image": input_images[input_img_id]})
            input_img_id += 1
        else:
            content.append({"type": "text", "text": part})
    messages = [{"role": "user", "content": content}]
    return messages, input_images, history_indices


def make_processor(model_path, processor_use_fast, trust_remote_code):
    kwargs = {"trust_remote_code": bool(trust_remote_code)}
    if processor_use_fast != "auto":
        kwargs["use_fast"] = processor_use_fast == "true"
    processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    processor.tokenizer.padding_side = "left"
    return processor


class HFVisionEncodeBackend:
    def __init__(self, args):
        dtype_map = {
            "auto": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "half": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if args.dtype not in dtype_map:
            raise ValueError(f"Unsupported HF dtype: {args.dtype}")
        self.device = torch.device(args.device)
        self.processor = make_processor(args.hf_model_path, args.processor_use_fast, args.trust_remote_code)
        self.model = InternVLAN1ForCausalLM.from_pretrained(
            args.hf_model_path,
            torch_dtype=dtype_map[args.dtype],
            attn_implementation=args.attn_backend,
        )
        self.model.to(device=self.device, dtype=dtype_map[args.dtype])
        self.model.eval()
        self.name = "hf"

    def measure(self, messages, input_images, max_new_tokens):
        del max_new_tokens
        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        _cuda_sync_if_needed(self.device)
        preprocess_start = time.perf_counter()
        model_inputs = self.processor(text=[prompt_text], images=input_images, return_tensors="pt")
        model_inputs = model_inputs.to(self.device)
        _cuda_sync_if_needed(self.device)
        mm_processor_ms = (time.perf_counter() - preprocess_start) * 1000.0

        pixel_values = getattr(model_inputs, "pixel_values", None)
        image_grid_thw = normalize_image_grid_thw(getattr(model_inputs, "image_grid_thw", None))
        if pixel_values is None or image_grid_thw is None:
            raise RuntimeError("HF processor did not produce pixel_values/image_grid_thw.")

        visual = self.model._get_visual()
        pixel_values = pixel_values.type(visual.dtype)

        _cuda_sync_if_needed(self.device)
        vision_start = time.perf_counter()
        with torch.no_grad():
            image_embeds = self.model._extract_visual_features(visual(pixel_values, grid_thw=image_grid_thw))
        _cuda_sync_if_needed(self.device)
        vision_encode_ms = (time.perf_counter() - vision_start) * 1000.0

        image_token_count = int((model_inputs.input_ids == self.model.config.image_token_id).sum().item())
        image_feature_count = int(image_embeds.shape[0])
        return {
            "backend": self.name,
            "prompt_token_count": int(model_inputs.input_ids.shape[1]),
            "generated_token_count": None,
            "mm_processor_ms": float(mm_processor_ms),
            "vision_encode_ms": float(vision_encode_ms),
            "vision_encoder_calls": 1,
            "vision_encode_ms_per_image": float(vision_encode_ms / max(len(input_images), 1)),
            "image_token_count": image_token_count,
            "image_feature_count": image_feature_count,
            "image_grid_thw": (
                image_grid_thw.detach().cpu().tolist() if torch.is_tensor(image_grid_thw) else None
            ),
        }


class PatchedVLLMVisionEncodeBackend:
    def __init__(self, args):
        from internnav.model.utils.dualvln_single_vllm import DualVLNSingleVLLMRunner

        if not args.vllm_model_path:
            raise ValueError("--vllm-model-path is required for backend=patched_vllm")

        compilation_config = None
        if args.enable_mm_encoder_compile:
            compilation_config = {"compile_mm_encoder": True}

        self.runner = DualVLNSingleVLLMRunner(
            model_path=args.vllm_model_path,
            hf_model_path=args.hf_model_path,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            limit_mm_per_prompt_image=args.limit_mm_per_prompt_image,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=args.trust_remote_code,
            enforce_eager=args.enforce_eager,
            seed=args.seed,
            compilation_config=compilation_config,
            quantization=args.quantization,
        )
        self.name = "patched_vllm"

    def measure(self, messages, input_images, max_new_tokens):
        result = self.runner.step_s2(messages, max_new_tokens=max_new_tokens, return_latents=False)
        runtime = result.get("runtime_metrics") or {}
        vision_encode_ms = runtime.get("vision_encode_ms")
        mm_processor_ms = runtime.get("mm_processor_ms", runtime.get("preprocess_ms"))
        return {
            "backend": self.name,
            "prompt_token_count": runtime.get("prompt_token_count"),
            "generated_token_count": runtime.get("generated_token_count"),
            "mm_processor_ms": None if mm_processor_ms is None else float(mm_processor_ms),
            "vision_encode_ms": None if vision_encode_ms is None else float(vision_encode_ms),
            "vision_encoder_calls": runtime.get("vision_encoder_calls"),
            "vision_encode_ms_per_image": (
                None
                if vision_encode_ms is None
                else float(vision_encode_ms / max(len(input_images), 1))
            ),
            "image_token_count": None,
            "image_feature_count": None,
            "image_grid_thw": None,
            "runtime_metrics": runtime,
        }


def _compact_runtime_metrics(runtime_metrics):
    if runtime_metrics is None:
        return None
    keep_keys = [
        "preprocess_ms",
        "mm_processor_ms",
        "generate_ms",
        "vision_encode_ms",
        "vision_encoder_calls",
        "llm_prefill_ms",
        "llm_decode_ms",
        "llm_extend_ms",
        "generate_residual_ms",
        "prompt_token_count",
        "generated_token_count",
        "num_images",
        "mm_feature_count",
    ]
    return {key: runtime_metrics.get(key) for key in keep_keys if key in runtime_metrics}


def _iter_selected_steps(replay_steps, max_episodes):
    count = 0
    for key, episode in replay_steps.items():
        if count >= max_episodes:
            break
        yield key, episode
        count += 1


def _build_backend(args):
    if args.backend == "hf":
        return HFVisionEncodeBackend(args)
    if args.backend == "patched_vllm":
        return PatchedVLLMVisionEncodeBackend(args)
    raise ValueError(f"Unsupported backend: {args.backend}")


def _cleanup_backend(backend):
    if backend is None:
        return
    try:
        del backend
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    args = parse_args()
    replay_steps = load_manifest(args.manifest, args.base_path)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    details_output = args.details_output or default_details_output(args.output)
    expected_num_images = int(args.num_history + 1)

    backend = _build_backend(args)
    measured_records = []
    skipped_short_history = 0
    skipped_missing_vision_encode = 0
    warmup_seen = 0
    total_seen = 0

    try:
        with open(details_output, "w", encoding="utf-8") as detail_f:
            for (scene_id, episode_id), episode in _iter_selected_steps(replay_steps, args.max_episodes):
                for step_index, item in enumerate(episode):
                    if args.max_steps is not None and len(measured_records) >= args.max_steps:
                        break

                    messages, input_images, history_indices = build_messages(
                        item["instruction"],
                        episode,
                        step_index,
                        args.num_history,
                        args.prompt_variant,
                        args.ignore_manifest_history,
                    )
                    total_seen += 1
                    num_images = len(input_images)
                    has_full_history = len(history_indices) == args.num_history
                    if not args.allow_short_history and not has_full_history:
                        skipped_short_history += 1
                        _close_images(input_images)
                        continue

                    try:
                        metrics = backend.measure(messages, input_images, args.max_new_tokens)
                    finally:
                        _close_images(input_images)

                    if warmup_seen < args.warmup_steps:
                        warmup_seen += 1
                        continue

                    if metrics.get("vision_encode_ms") is None:
                        skipped_missing_vision_encode += 1
                        continue

                    record = {
                        "backend": args.backend,
                        "scene_id": scene_id,
                        "episode_id": int(episode_id),
                        "step_id": int(item["step_id"]),
                        "history_frame_indices": history_indices,
                        "num_history_images": int(len(history_indices)),
                        "num_images": int(num_images),
                        "expected_num_images": expected_num_images,
                        "has_full_history": bool(has_full_history),
                        "mm_processor_ms": metrics.get("mm_processor_ms"),
                        "vision_encode_ms": metrics.get("vision_encode_ms"),
                        "vision_encode_ms_per_image": metrics.get("vision_encode_ms_per_image"),
                        "vision_encoder_calls": metrics.get("vision_encoder_calls"),
                        "prompt_token_count": metrics.get("prompt_token_count"),
                        "generated_token_count": metrics.get("generated_token_count"),
                        "image_token_count": metrics.get("image_token_count"),
                        "image_feature_count": metrics.get("image_feature_count"),
                        "image_grid_thw": metrics.get("image_grid_thw"),
                        "runtime_metrics": _compact_runtime_metrics(metrics.get("runtime_metrics")),
                    }
                    measured_records.append(record)
                    detail_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if args.max_steps is not None and len(measured_records) >= args.max_steps:
                    break
    finally:
        _cleanup_backend(backend)

    vision_values = [record["vision_encode_ms"] for record in measured_records if record["vision_encode_ms"] is not None]
    per_image_values = [
        record["vision_encode_ms_per_image"]
        for record in measured_records
        if record["vision_encode_ms_per_image"] is not None
    ]
    processor_values = [record["mm_processor_ms"] for record in measured_records if record["mm_processor_ms"] is not None]

    grouped_by_num_images = defaultdict(list)
    for record in measured_records:
        grouped_by_num_images[int(record["num_images"])].append(record)

    grouped_summary = {}
    for num_images, rows in sorted(grouped_by_num_images.items()):
        grouped_summary[str(num_images)] = {
            "count": len(rows),
            "vision_encode_ms": summarize([row["vision_encode_ms"] for row in rows if row["vision_encode_ms"] is not None]),
            "vision_encode_ms_per_image": summarize(
                [row["vision_encode_ms_per_image"] for row in rows if row["vision_encode_ms_per_image"] is not None]
            ),
        }

    exact_rows = [record for record in measured_records if int(record["num_images"]) == expected_num_images]
    summary = {
        "metadata": {
            "backend": args.backend,
            "manifest": args.manifest,
            "base_path": args.base_path,
            "hf_model_path": args.hf_model_path,
            "vllm_model_path": args.vllm_model_path,
            "device": args.device,
            "dtype": args.dtype,
            "quantization": args.quantization,
            "num_history": int(args.num_history),
            "expected_num_images": expected_num_images,
            "prompt_variant": args.prompt_variant,
            "max_episodes": int(args.max_episodes),
            "max_steps": args.max_steps,
            "warmup_steps": int(args.warmup_steps),
            "ignore_manifest_history": bool(args.ignore_manifest_history),
            "allow_short_history": bool(args.allow_short_history),
            "enable_mm_encoder_compile": bool(args.enable_mm_encoder_compile),
            "details_output": details_output,
        },
        "counts": {
            "episodes_loaded": len(replay_steps),
            "episodes_selected": min(len(replay_steps), int(args.max_episodes)),
            "steps_seen": int(total_seen),
            "steps_measured": len(measured_records),
            "steps_skipped_short_history": int(skipped_short_history),
            "steps_skipped_missing_vision_encode": int(skipped_missing_vision_encode),
            "warmup_steps_skipped_after_measurement": int(min(args.warmup_steps, max(total_seen - skipped_short_history, 0))),
        },
        "overall": {
            "mm_processor_ms": summarize(processor_values),
            "vision_encode_ms": summarize(vision_values),
            "vision_encode_ms_per_image": summarize(per_image_values),
        },
        "exact_expected_image_count": {
            "num_images": expected_num_images,
            "count": len(exact_rows),
            "vision_encode_ms": summarize([row["vision_encode_ms"] for row in exact_rows]),
            "vision_encode_ms_per_image": summarize([row["vision_encode_ms_per_image"] for row in exact_rows]),
        },
        "grouped_by_num_images": grouped_summary,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

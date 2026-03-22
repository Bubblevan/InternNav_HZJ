#!/usr/bin/env python3
import argparse
import json
import os
import traceback
from pathlib import Path

import torch
from transformers import AutoConfig

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")

from vllm import LLM, SamplingParams


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Smoke-test vLLM generate() with prompt_embeds + token ids + multimodal metadata."
    )
    p.add_argument("--model-path", required=True)
    p.add_argument("--bundle", required=True)
    p.add_argument("--out-json", required=True)

    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-impl", default="transformers")
    p.add_argument("--max-tokens", type=int, default=1)

    p.add_argument("--max-num-seqs", type=int, default=1)
    p.add_argument("--max-num-batched-tokens", type=int, default=2304)

    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--try-enable-prompt-embeds", action="store_true")
    return p


def to_1d_long_list(x, name: str) -> list[int]:
    if isinstance(x, torch.Tensor):
        if x.ndim != 1:
            raise ValueError(f"Expected rank-1 {name}, got {tuple(x.shape)}")
        return x.long().tolist()
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    raise TypeError(f"Unsupported type for {name}: {type(x)}")


def to_grid_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.long()
    elif isinstance(x, (list, tuple)):
        t = torch.tensor(x, dtype=torch.long)
    else:
        raise TypeError(f"Unsupported type for image_grid_thw: {type(x)}")

    if t.ndim == 1:
        if t.numel() != 3:
            raise ValueError(f"Expected image_grid_thw with 3 elems, got {tuple(t.shape)}")
        t = t.unsqueeze(0)

    if t.ndim != 2 or t.shape[1] != 3:
        raise ValueError(f"Expected image_grid_thw shape [N,3], got {tuple(t.shape)}")

    return t


def infer_image_mm_data(
    *,
    prompt_embeds: torch.Tensor,
    full_prompt_ids: list[int],
    image_grid_thw: torch.Tensor,
    image_token_id: int,
) -> tuple[dict, dict, dict]:
    image_positions = [i for i, tok in enumerate(full_prompt_ids) if tok == image_token_id]
    if not image_positions:
        raise ValueError("No image_token_id found in full_prompt_ids")

    # Qwen2.x-VL family usually has placeholder token count = sum(t*h*w) / merge_factor.
    raw_sizes = [int(row[0].item() * row[1].item() * row[2].item()) for row in image_grid_thw]
    total_raw = sum(raw_sizes)
    total_img_tokens = len(image_positions)

    if total_raw % total_img_tokens != 0:
        raise ValueError(
            f"Cannot infer merge factor: total_raw={total_raw}, "
            f"total_img_tokens={total_img_tokens}"
        )

    merge_factor = total_raw // total_img_tokens
    if merge_factor <= 0:
        raise ValueError(f"Invalid inferred merge_factor={merge_factor}")

    item_lengths = []
    for raw in raw_sizes:
        if raw % merge_factor != 0:
            raise ValueError(
                f"Grid volume {raw} is not divisible by merge_factor={merge_factor}"
            )
        item_lengths.append(raw // merge_factor)

    if sum(item_lengths) != total_img_tokens:
        raise ValueError(
            f"Sum(item_lengths)={sum(item_lengths)} != total_img_tokens={total_img_tokens}"
        )

    image_embed_segments = []
    placeholder_offsets = []

    cursor = 0
    for i, length in enumerate(item_lengths):
        seg = image_positions[cursor: cursor + length]
        if len(seg) != length:
            raise ValueError(f"Image segment {i} length mismatch")

        # 要求每个图像对应的 image tokens 在 prompt 中是连续的一段
        expected = list(range(seg[0], seg[0] + length))
        if seg != expected:
            raise ValueError(
                f"Image token segment {i} is not contiguous. "
                f"start={seg[0]}, len={length}, actual_head={seg[:8]}"
            )

        image_embed_segments.append(prompt_embeds[seg[0]: seg[0] + length].clone())
        placeholder_offsets.append(seg[0])

        cursor += length

    image_embeds = torch.cat(image_embed_segments, dim=0).contiguous()
    multi_modal_data = {
        "image": {
            "image_embeds": image_embeds,
            "image_grid_thw": image_grid_thw.clone().contiguous(),
        }
    }
    multi_modal_uuids = {
        "image": [f"dualvln-img-{i}" for i in range(int(image_grid_thw.shape[0]))]
    }

    debug = {
        "image_token_id": image_token_id,
        "num_images": int(image_grid_thw.shape[0]),
        "image_token_count": total_img_tokens,
        "raw_sizes": raw_sizes,
        "merge_factor": int(merge_factor),
        "item_lengths": item_lengths,
        "placeholder_offsets": placeholder_offsets,
        "image_embeds_shape": list(image_embeds.shape),
    }
    return multi_modal_data, multi_modal_uuids, debug


def main() -> None:
    args = build_parser().parse_args()

    bundle = torch.load(args.bundle, map_location="cpu")

    prompt_embeds = bundle["prompt_embeds"]
    if not isinstance(prompt_embeds, torch.Tensor) or prompt_embeds.ndim != 2:
        raise ValueError(f"Expected rank-2 prompt_embeds, got {type(prompt_embeds)} / {getattr(prompt_embeds, 'shape', None)}")

    full_prompt_ids = to_1d_long_list(bundle["full_prompt_ids"], "full_prompt_ids")
    image_grid_thw = to_grid_tensor(bundle["image_grid_thw"])

    if len(full_prompt_ids) != prompt_embeds.shape[0]:
        raise ValueError(
            f"Length mismatch: len(full_prompt_ids)={len(full_prompt_ids)} "
            f"vs prompt_embeds.shape[0]={prompt_embeds.shape[0]}"
        )

    cfg = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if not hasattr(cfg, "image_token_id"):
        raise AttributeError("Model config has no image_token_id")
    image_token_id = int(cfg.image_token_id)
    architectures = list(getattr(cfg, "architectures", []) or [])

    multi_modal_data, multi_modal_uuids, mm_debug = infer_image_mm_data(
        prompt_embeds=prompt_embeds,
        full_prompt_ids=full_prompt_ids,
        image_grid_thw=image_grid_thw,
        image_token_id=image_token_id,
    )

    effective_model_impl = args.model_impl
    model_impl_note = None
    if "InternVLAN1ForCausalLM" in architectures and args.model_impl == "transformers":
        effective_model_impl = "auto"
        model_impl_note = (
            "Switched InternVLAN1ForCausalLM from generic transformers backend "
            "to native vLLM adapter for this prompt-embeds smoketest."
        )

    llm_kwargs = dict(
        model=args.model_path,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        model_impl=effective_model_impl,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        disable_log_stats=True,
        async_scheduling=False,
    )
    llm_kwargs["enable_prompt_embeds"] = True

    out = {
        "bundle": args.bundle,
        "model_path": args.model_path,
        "architectures": architectures,
        "prompt_embeds_shape": list(prompt_embeds.shape),
        "full_prompt_ids_len": len(full_prompt_ids),
        "llm_kwargs": llm_kwargs,
        "mm_debug": mm_debug,
    }
    if model_impl_note is not None:
        out["model_impl_override_note"] = model_impl_note
    out["v2_model_runner"] = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    out["request_mm_mode"] = "omitted"

    try:
        try:
            llm = LLM(**llm_kwargs)
        except TypeError as e:
            if "enable_prompt_embeds" in llm_kwargs:
                llm_kwargs.pop("enable_prompt_embeds", None)
                out["llm_init_retry_without_enable_prompt_embeds"] = str(e)
                llm = LLM(**llm_kwargs)
            else:
                raise

        prompts = [{
            "prompt_token_ids": full_prompt_ids,
            "prompt_embeds": prompt_embeds.to(dtype=getattr(torch, args.dtype)).contiguous(),
            "prompt": "<dualvln_prompt_embeds_smoketest>",
        }]

        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0.0,
        )

        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        output = outputs[0]
        completion = output.outputs[0]

        out.update({
            "success": True,
            "prompt_token_ids_len": None if output.prompt_token_ids is None else len(output.prompt_token_ids),
            "generated_token_ids": list(completion.token_ids),
            "generated_text": completion.text,
            "finish_reason": getattr(completion, "finish_reason", None),
        })

    except Exception as exc:
        out.update({
            "success": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

import argparse
import functools
import json
import os
from pathlib import Path

import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export vLLM Qwen2.5-VL visual intermediate tensors for one sample."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sample-pt", required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output-pt", required=True)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def _qualname(obj):
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}.{cls.__name__}"


def _extract_vllm_visual_stages(model, pixel_values_cpu, image_grid_thw_cpu):
    device = next(model.parameters()).device
    visual = model.visual
    pixel_values = pixel_values_cpu.to(device=device, dtype=visual.dtype)
    image_grid_thw = image_grid_thw_cpu.to(device=device)
    grid_thw_list = image_grid_thw.tolist()

    with torch.inference_mode():
        patch_embed = visual.patch_embed(pixel_values)

        rotary_cos = []
        rotary_sin = []
        window_index = []
        cu_window_seqlens = []
        window_index_id = 0
        cu_window_last = 0
        for t, h, w in grid_thw_list:
            cos_thw, sin_thw, window_index_thw, cu_window_thw, _ = visual.get_rope_by_thw(
                int(t),
                int(h),
                int(w),
            )
            rotary_cos.append(cos_thw)
            rotary_sin.append(sin_thw)
            window_index.append(window_index_thw + window_index_id)
            llm_h = int(h) // visual.spatial_merge_size
            llm_w = int(w) // visual.spatial_merge_size
            window_index_id += int(t) * llm_h * llm_w
            if not cu_window_seqlens:
                cu_window_seqlens.append(
                    torch.tensor([0], dtype=torch.int32, device=cu_window_thw.device)
                )
            cu_window_thw = cu_window_thw + cu_window_last
            cu_window_last = cu_window_thw[-1]
            cu_window_seqlens.append(cu_window_thw)

        rotary_cos = torch.cat(rotary_cos, dim=0)
        rotary_sin = torch.cat(rotary_sin, dim=0)
        window_index = torch.cat(window_index, dim=0)
        cu_window_seqlens = torch.unique_consecutive(torch.cat(cu_window_seqlens, dim=0))
        final_output = visual(pixel_values, grid_thw=grid_thw_list)

    return {
        "model_type": _qualname(model),
        "visual_type": _qualname(visual),
        "visual_dtype": str(visual.dtype),
        "patch_embed": patch_embed.detach().cpu(),
        "rotary_cos": rotary_cos.detach().cpu(),
        "rotary_sin": rotary_sin.detach().cpu(),
        "window_index": window_index.detach().cpu(),
        "cu_window_seqlens": cu_window_seqlens.detach().cpu(),
        "final_output": final_output.detach().cpu(),
    }


def main():
    args = parse_args()

    sample = torch.load(args.sample_pt, map_location="cpu")
    pixel_values = sample["pixel_values"]
    image_grid_thw = sample["image_grid_thw"]

    from vllm import LLM

    llm = LLM(
        model=args.model_path,
        runner="pooling",
        convert="embed",
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        disable_log_stats=True,
    )

    result = llm.apply_model(
        functools.partial(
            _extract_vllm_visual_stages,
            pixel_values_cpu=pixel_values,
            image_grid_thw_cpu=image_grid_thw,
        )
    )[0]

    output_pt = Path(args.output_pt)
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_path": args.model_path,
            "sample_pt": args.sample_pt,
            "image_grid_thw": image_grid_thw,
            **result,
        },
        output_pt,
    )

    report = {
        "model_path": args.model_path,
        "sample_pt": args.sample_pt,
        "output_pt": str(output_pt),
        "model_type": result["model_type"],
        "visual_type": result["visual_type"],
        "visual_dtype": result["visual_dtype"],
        "patch_embed_shape": list(result["patch_embed"].shape),
        "rotary_cos_shape": list(result["rotary_cos"].shape),
        "rotary_sin_shape": list(result["rotary_sin"].shape),
        "window_index_shape": list(result["window_index"].shape),
        "cu_window_seqlens_shape": list(result["cu_window_seqlens"].shape),
        "final_output_shape": list(result["final_output"].shape),
    }

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print("=" * 72)
    print("Export vLLM visual stages")
    print("=" * 72)
    print(f"Patch embed shape: {list(result['patch_embed'].shape)}")
    print(f"Rotary cos shape: {list(result['rotary_cos'].shape)}")
    print(f"Window index shape: {list(result['window_index'].shape)}")
    print(f"Final output shape: {list(result['final_output'].shape)}")
    print(f"Saved PT to {output_pt}")
    if args.output_json:
        print(f"Saved JSON to {args.output_json}")
    print("=" * 72)


if __name__ == "__main__":
    main()

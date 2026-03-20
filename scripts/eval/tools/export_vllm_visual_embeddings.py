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
        description="Export raw vLLM Qwen2.5-VL visual embeddings for a baseline sample."
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


def _extract_vllm_visual_embeddings(model, pixel_values_cpu, image_grid_thw_cpu):
    device = next(model.parameters()).device
    pixel_values = pixel_values_cpu.to(device=device, dtype=model.visual.dtype)
    image_grid_thw = image_grid_thw_cpu.to(device=device)

    with torch.inference_mode():
        multimodal_embeddings = model.embed_multimodal(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    split_embeddings = tuple(emb.detach().cpu() for emb in multimodal_embeddings)
    flat_embeddings = (
        torch.cat(list(split_embeddings), dim=0) if split_embeddings else torch.empty(0)
    )
    return {
        "model_type": _qualname(model),
        "visual_type": _qualname(model.visual),
        "visual_dtype": str(model.visual.dtype),
        "num_items": len(split_embeddings),
        "split_shapes": [list(emb.shape) for emb in split_embeddings],
        "flat_embeddings": flat_embeddings,
        "split_embeddings": split_embeddings,
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
            _extract_vllm_visual_embeddings,
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
            "model_type": result["model_type"],
            "visual_type": result["visual_type"],
            "visual_dtype": result["visual_dtype"],
            "num_items": result["num_items"],
            "split_shapes": result["split_shapes"],
            "flat_embeddings": result["flat_embeddings"],
            "split_embeddings": result["split_embeddings"],
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
        "num_items": result["num_items"],
        "split_shapes": result["split_shapes"],
        "flat_shape": list(result["flat_embeddings"].shape),
    }

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print("=" * 72)
    print("Export vLLM visual embeddings")
    print("=" * 72)
    print(f"Model path: {args.model_path}")
    print(f"Num items: {result['num_items']}")
    print(f"Flat shape: {list(result['flat_embeddings'].shape)}")
    print(f"Saved PT to {output_pt}")
    if args.output_json:
        print(f"Saved JSON to {args.output_json}")
    print("=" * 72)


if __name__ == "__main__":
    main()

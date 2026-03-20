import argparse
import json
from pathlib import Path

import torch

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare HF visual embeddings against exported vLLM visual embeddings."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sample-pt", required=True)
    parser.add_argument("--vllm-pt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-backend", choices=["flash_attention_2", "sdpa", "eager"], default="flash_attention_2")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _torch_dtype(name: str):
    if name == "auto":
        return None
    return getattr(torch, name)


def _tensor_diff(a: torch.Tensor, b: torch.Tensor):
    diff = (a.float() - b.float()).abs()
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def main():
    args = parse_args()

    sample = torch.load(args.sample_pt, map_location="cpu")
    vllm_payload = torch.load(args.vllm_pt, map_location="cpu")

    pixel_values = sample["pixel_values"]
    image_grid_thw = sample["image_grid_thw"]

    model = InternVLAN1ForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=_torch_dtype(args.dtype),
        attn_implementation=args.attn_backend,
    ).eval().to(args.device)

    with torch.no_grad():
        hf_flat = model.visual(
            pixel_values.to(args.device, dtype=model.visual.dtype),
            grid_thw=image_grid_thw.to(args.device),
        ).detach().cpu()

    vllm_flat = vllm_payload["flat_embeddings"]
    if list(hf_flat.shape) != list(vllm_flat.shape):
        raise RuntimeError(
            f"Shape mismatch: HF {list(hf_flat.shape)} vs vLLM {list(vllm_flat.shape)}"
        )

    sizes = [shape[0] for shape in vllm_payload["split_shapes"]]
    hf_splits = list(hf_flat.split(sizes, dim=0))
    vllm_splits = list(vllm_flat.split(sizes, dim=0))
    per_image = []
    for idx, (hf_emb, vllm_emb) in enumerate(zip(hf_splits, vllm_splits)):
        per_image.append(
            {
                "index": idx,
                "shape": list(hf_emb.shape),
                "diff": _tensor_diff(hf_emb, vllm_emb),
            }
        )

    report = {
        "model_path": args.model_path,
        "sample_pt": args.sample_pt,
        "vllm_pt": args.vllm_pt,
        "hf_flat_shape": list(hf_flat.shape),
        "vllm_flat_shape": list(vllm_flat.shape),
        "flat_diff": _tensor_diff(hf_flat, vllm_flat),
        "per_image": per_image,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 72)
    print("Compare HF vs vLLM visual embeddings")
    print("=" * 72)
    print(f"HF flat shape: {list(hf_flat.shape)}")
    print(f"vLLM flat shape: {list(vllm_flat.shape)}")
    print(f"Flat max abs diff: {report['flat_diff']['max_abs_diff']:.6f}")
    print(f"Flat mean abs diff: {report['flat_diff']['mean_abs_diff']:.6f}")
    if per_image:
        worst = max(per_image, key=lambda item: item["diff"]["max_abs_diff"])
        print(
            f"Worst image diff: index={worst['index']} max_abs_diff={worst['diff']['max_abs_diff']:.6f}"
        )
    if args.output:
        print(f"Saved JSON summary to {args.output}")
    print("=" * 72)


if __name__ == "__main__":
    main()

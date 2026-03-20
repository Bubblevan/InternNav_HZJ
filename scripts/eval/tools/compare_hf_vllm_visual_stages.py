import argparse
import json
from pathlib import Path

import torch

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare HF vs vLLM Qwen2.5-VL visual intermediate tensors."
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
    visual = model.visual

    with torch.no_grad():
        pixel_values_gpu = pixel_values.to(args.device, dtype=visual.dtype)
        image_grid_thw_gpu = image_grid_thw.to(args.device)
        patch_embed = visual.patch_embed(pixel_values_gpu).detach().cpu()
        rotary = visual.rot_pos_emb(image_grid_thw_gpu)
        rotary_cos = rotary.cos().detach().cpu()
        rotary_sin = rotary.sin().detach().cpu()
        window_index, cu_window_seqlens = visual.get_window_index(image_grid_thw_gpu)
        window_index = window_index.detach().cpu()
        cu_window_seqlens = torch.tensor(cu_window_seqlens, dtype=torch.int32)
        final_output = visual(pixel_values_gpu, grid_thw=image_grid_thw_gpu).detach().cpu()

    report = {
        "model_path": args.model_path,
        "sample_pt": args.sample_pt,
        "vllm_pt": args.vllm_pt,
        "patch_embed_diff": _tensor_diff(patch_embed, vllm_payload["patch_embed"]),
        "rotary_cos_diff": _tensor_diff(rotary_cos, vllm_payload["rotary_cos"]),
        "rotary_sin_diff": _tensor_diff(rotary_sin, vllm_payload["rotary_sin"]),
        "window_index_equal": bool(torch.equal(window_index, vllm_payload["window_index"])),
        "cu_window_seqlens_equal": bool(torch.equal(cu_window_seqlens, vllm_payload["cu_window_seqlens"])),
        "final_output_diff": _tensor_diff(final_output, vllm_payload["final_output"]),
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 72)
    print("Compare HF vs vLLM visual stages")
    print("=" * 72)
    print(f"Patch embed max abs diff: {report['patch_embed_diff']['max_abs_diff']:.6f}")
    print(f"Rotary cos max abs diff: {report['rotary_cos_diff']['max_abs_diff']:.6f}")
    print(f"Rotary sin max abs diff: {report['rotary_sin_diff']['max_abs_diff']:.6f}")
    print(f"Window index equal: {report['window_index_equal']}")
    print(f"CU window seqlens equal: {report['cu_window_seqlens_equal']}")
    print(f"Final output max abs diff: {report['final_output_diff']['max_abs_diff']:.6f}")
    if args.output:
        print(f"Saved JSON summary to {args.output}")
    print("=" * 72)


if __name__ == "__main__":
    main()

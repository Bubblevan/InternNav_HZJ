import argparse
import json
import os
from pathlib import Path

import torch

from internnav.model.basemodel.internvla_n1.internvla_n1 import (
    IMAGE_TOKEN_INDEX,
    InternVLAN1ForCausalLM,
    TRAJ_TOKEN_INDEX,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare HF generate_latents input embeddings against vLLM prepare_inputs dump."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sample-pt", required=True)
    parser.add_argument("--probe-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-backend", choices=["flash_attention_2", "sdpa", "eager"], default="flash_attention_2")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--append-traj-tokens", action="store_true")
    parser.add_argument("--traj-token-count", type=int, default=4)
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


def _load_vllm_prepare_inputs(probe_json_path: str):
    project_root = Path.cwd().resolve()
    probe = json.loads(Path(probe_json_path).read_text(encoding="utf-8"))
    prepare = probe["dump_diffs"]["prepare_inputs"]
    if prepare is None:
        raise RuntimeError("probe_json has no prepare_inputs dump diff")
    custom_path = Path(prepare["custom_path"])
    if not custom_path.is_absolute():
        custom_path = project_root / custom_path
    payload = torch.load(custom_path, map_location="cpu")
    return {
        "path": str(custom_path),
        "inputs_embeds": payload["tensors"]["inputs_embeds_gpu"],
        "positions": payload["tensors"].get("positions_gpu"),
    }


def main():
    args = parse_args()

    sample = torch.load(args.sample_pt, map_location="cpu")
    input_ids = sample["baseline_output_ids"][0].clone()
    if args.append_traj_tokens:
        input_ids = torch.cat(
            [input_ids, torch.full((args.traj_token_count,), TRAJ_TOKEN_INDEX, dtype=torch.long)],
            dim=0,
        )
    input_ids = input_ids.unsqueeze(0)

    pixel_values = sample["pixel_values"]
    image_grid_thw = sample["image_grid_thw"]
    vllm_dump = _load_vllm_prepare_inputs(args.probe_json)

    model = InternVLAN1ForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=_torch_dtype(args.dtype),
        attn_implementation=args.attn_backend,
    ).eval().to(args.device)

    with torch.no_grad():
        text_embeds = model.get_model().embed_tokens(input_ids.to(args.device)).clone()
        image_embeds = model.visual(
            pixel_values.to(args.device, dtype=model.visual.dtype),
            grid_thw=image_grid_thw.to(args.device),
        ).unsqueeze(0)
        image_idx = input_ids.to(args.device) == IMAGE_TOKEN_INDEX
        text_embeds[image_idx] = image_embeds.to(text_embeds.dtype)[: image_idx.sum(), :]
        latent_queries = model.get_model().latent_queries.repeat(text_embeds.shape[0], 1, 1)
        text_embeds[:, -latent_queries.shape[1] :, :] = latent_queries.to(text_embeds.dtype)
        hf_inputs_embeds = text_embeds[0].detach().cpu()

    report = {
        "model_path": args.model_path,
        "sample_pt": args.sample_pt,
        "probe_json": args.probe_json,
        "vllm_prepare_inputs_path": vllm_dump["path"],
        "hf_inputs_embeds_shape": list(hf_inputs_embeds.shape),
        "vllm_inputs_embeds_shape": list(vllm_dump["inputs_embeds"].shape),
        "inputs_embeds_diff": _tensor_diff(hf_inputs_embeds, vllm_dump["inputs_embeds"]),
        "inputs_embeds_last4_diff": _tensor_diff(
            hf_inputs_embeds[-args.traj_token_count :],
            vllm_dump["inputs_embeds"][-args.traj_token_count :],
        ),
        "inputs_embeds_image_diff": None,
        "inputs_embeds_text_diff": None,
        "hf_baseline_latent_shape": list(sample["baseline_latent"][0].shape),
    }

    image_mask = input_ids[0] == IMAGE_TOKEN_INDEX
    text_mask = ~image_mask
    if image_mask.any():
        report["inputs_embeds_image_diff"] = _tensor_diff(
            hf_inputs_embeds[image_mask.cpu()],
            vllm_dump["inputs_embeds"][image_mask.cpu()],
        )
    if text_mask.any():
        report["inputs_embeds_text_diff"] = _tensor_diff(
            hf_inputs_embeds[text_mask.cpu()],
            vllm_dump["inputs_embeds"][text_mask.cpu()],
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 72)
    print("Compare HF vs vLLM generate_latents inputs")
    print("=" * 72)
    print(f"HF inputs_embeds shape: {list(hf_inputs_embeds.shape)}")
    print(f"vLLM inputs_embeds shape: {list(vllm_dump['inputs_embeds'].shape)}")
    print(f"Inputs max abs diff: {report['inputs_embeds_diff']['max_abs_diff']:.6f}")
    print(f"Inputs mean abs diff: {report['inputs_embeds_diff']['mean_abs_diff']:.6f}")
    print(f"Last-{args.traj_token_count} max abs diff: {report['inputs_embeds_last4_diff']['max_abs_diff']:.6f}")
    if report["inputs_embeds_image_diff"] is not None:
        print(f"Image-token max abs diff: {report['inputs_embeds_image_diff']['max_abs_diff']:.6f}")
    if report["inputs_embeds_text_diff"] is not None:
        print(f"Text-token max abs diff: {report['inputs_embeds_text_diff']['max_abs_diff']:.6f}")
    if args.output:
        print(f"Saved JSON summary to {args.output}")
    print("=" * 72)


if __name__ == "__main__":
    main()

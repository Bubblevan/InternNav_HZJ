import argparse
import json
from pathlib import Path

import torch

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare HF vs vLLM Qwen2.5-VL visual block hidden states."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sample-pt", required=True)
    parser.add_argument("--vllm-pt", required=True)
    parser.add_argument("--probe-json", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attn-backend",
        choices=["flash_attention_2", "sdpa", "eager"],
        default="flash_attention_2",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument("--small-threshold", type=float, default=1e-3)
    parser.add_argument("--obvious-threshold", type=float, default=0.1)
    parser.add_argument("--strong-threshold", type=float, default=0.5)
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


def _first_over(block_diffs, threshold: float):
    for block in block_diffs:
        if block["max_abs_diff"] > threshold:
            return int(block["layer_idx"])
    return None


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

        hidden_states = visual.patch_embed(pixel_values_gpu)
        patch_embed = hidden_states.detach().cpu()

        rotary_pos_emb = visual.rot_pos_emb(image_grid_thw_gpu)
        window_index, cu_window_seqlens = visual.get_window_index(image_grid_thw_gpu)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=image_grid_thw_gpu.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        pre_blocks = hidden_states.detach().cpu()

        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            image_grid_thw_gpu[:, 1] * image_grid_thw_gpu[:, 2],
            image_grid_thw_gpu[:, 0],
        ).cumsum(
            dim=0,
            dtype=image_grid_thw_gpu.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        hf_block_outputs = []
        block_attention_modes = []
        for layer_num, blk in enumerate(visual.blocks):
            if layer_num in visual.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
                block_attention_modes.append("full")
            else:
                cu_seqlens_now = cu_window_seqlens
                block_attention_modes.append("window")

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
            )
            hf_block_outputs.append(hidden_states.detach().cpu())

        pre_merger = hidden_states.detach().cpu()
        final_output = visual.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        final_output = final_output[reverse_indices, :].detach().cpu()

    vllm_block_outputs = vllm_payload["block_outputs"]
    block_diffs = []
    for layer_idx, (hf_tensor, vllm_tensor) in enumerate(zip(hf_block_outputs, vllm_block_outputs)):
        diff = _tensor_diff(hf_tensor, vllm_tensor)
        block_diffs.append(
            {
                "layer_idx": int(layer_idx),
                "attention_mode": block_attention_modes[layer_idx],
                **diff,
            }
        )

    report = {
        "model_path": args.model_path,
        "sample_pt": args.sample_pt,
        "vllm_pt": args.vllm_pt,
        "probe_json": args.probe_json,
        "patch_embed_diff": _tensor_diff(patch_embed, vllm_payload["patch_embed"]),
        "pre_blocks_diff": _tensor_diff(pre_blocks, vllm_payload["pre_blocks"]),
        "num_blocks": len(block_diffs),
        "block_diffs": block_diffs,
        "pre_merger_diff": _tensor_diff(pre_merger, vllm_payload["pre_merger"]),
        "final_output_diff": _tensor_diff(final_output, vllm_payload["final_output"]),
        "first_block_over_small_threshold": _first_over(block_diffs, args.small_threshold),
        "first_block_over_obvious_threshold": _first_over(block_diffs, args.obvious_threshold),
        "first_block_over_strong_threshold": _first_over(block_diffs, args.strong_threshold),
    }

    if args.probe_json:
        probe = json.loads(Path(args.probe_json).read_text(encoding="utf-8"))
        report["latent_probe_summary"] = {
            "vllm_custom_hidden_last4_vs_hf_baseline_latent": probe["hf_baseline_compare"][
                "vllm_custom_hidden_last4_vs_hf_baseline_latent"
            ],
            "post_forward_hidden_states_last4": probe["dump_diffs"]["post_forward_hidden_states"]["last4"],
        }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 72)
    print("Compare HF vs vLLM visual block states")
    print("=" * 72)
    print(f"Patch embed max abs diff: {report['patch_embed_diff']['max_abs_diff']:.6f}")
    print(f"Pre-blocks max abs diff: {report['pre_blocks_diff']['max_abs_diff']:.6f}")
    print(f"First block over {args.small_threshold}: {report['first_block_over_small_threshold']}")
    print(f"First block over {args.obvious_threshold}: {report['first_block_over_obvious_threshold']}")
    print(f"First block over {args.strong_threshold}: {report['first_block_over_strong_threshold']}")
    print(f"Pre-merger max abs diff: {report['pre_merger_diff']['max_abs_diff']:.6f}")
    print(f"Final output max abs diff: {report['final_output_diff']['max_abs_diff']:.6f}")
    if args.probe_json:
        latent = report["latent_probe_summary"]["vllm_custom_hidden_last4_vs_hf_baseline_latent"]
        print(f"Final latent last-4 max abs diff: {latent['max_abs_diff']:.6f}")
    if args.output:
        print(f"Saved JSON summary to {args.output}")
    print("=" * 72)


if __name__ == "__main__":
    main()

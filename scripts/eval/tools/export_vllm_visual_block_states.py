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
        description="Export vLLM Qwen2.5-VL visual block hidden states for one sample."
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


def _cpu_bf16(tensor):
    return tensor.detach().to(dtype=torch.bfloat16).cpu()


def _invert_permutation_device(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
    return inv


def _extract_vllm_visual_block_states(model, pixel_values_cpu, image_grid_thw_cpu):
    device = next(model.parameters()).device
    visual = model.visual
    pixel_values = pixel_values_cpu.to(device=device, dtype=visual.dtype)
    image_grid_thw = image_grid_thw_cpu.to(device=device)
    grid_thw_list = image_grid_thw.tolist()

    with torch.inference_mode():
        hidden_states = visual.patch_embed(pixel_values)
        patch_embed = hidden_states.clone()

        rotary_pos_emb_cos = []
        rotary_pos_emb_sin = []
        window_index = []
        cu_window_seqlens = [torch.tensor([0], dtype=torch.int32, device=device)]
        cu_seqlens = []

        window_index_id = 0
        cu_window_seqlens_last = 0
        for t, h, w in grid_thw_list:
            t, h, w = int(t), int(h), int(w)
            llm_h = h // visual.spatial_merge_size
            llm_w = w // visual.spatial_merge_size

            (
                cos_thw,
                sin_thw,
                window_index_thw,
                cu_seqlens_window_thw,
                cu_seqlens_thw,
            ) = visual.get_rope_by_thw(t, h, w)
            window_index_thw = window_index_thw.to(device=device)
            cu_seqlens_window_thw = cu_seqlens_window_thw.to(device=device)
            cu_seqlens_thw = cu_seqlens_thw.to(device=device)

            window_index.append(window_index_thw + window_index_id)
            window_index_id += t * llm_h * llm_w

            cu_seqlens_window_thw = cu_seqlens_window_thw + cu_window_seqlens_last
            cu_window_seqlens_last = cu_seqlens_window_thw[-1]
            cu_window_seqlens.append(cu_seqlens_window_thw)

            rotary_pos_emb_cos.append(cos_thw)
            rotary_pos_emb_sin.append(sin_thw)
            cu_seqlens.append(cu_seqlens_thw)

        rotary_pos_emb_cos = torch.cat(rotary_pos_emb_cos).to(device=device, non_blocking=True)
        rotary_pos_emb_sin = torch.cat(rotary_pos_emb_sin).to(device=device, non_blocking=True)
        window_index = torch.cat(window_index).to(device=device, non_blocking=True)
        reverse_indices = _invert_permutation_device(window_index)
        cu_window_seqlens = torch.unique_consecutive(torch.cat(cu_window_seqlens)).to(device=device, non_blocking=True)
        cu_seqlens = torch.cat(cu_seqlens)
        cu_seqlens = torch.cumsum(cu_seqlens, dim=0, dtype=torch.int32)
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), "constant", 0).to(
            device=device, non_blocking=True
        )

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(
            seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1
        )
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        pre_blocks = hidden_states.clone()
        hidden_states = hidden_states.unsqueeze(1)

        max_seqlen_full = visual.compute_attn_mask_seqlen(cu_seqlens)
        max_seqlen_window = visual.compute_attn_mask_seqlen(cu_window_seqlens)

        block_outputs = []
        block_attention_modes = []
        for layer_num, blk in enumerate(visual.blocks):
            if layer_num in visual.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
                max_seqlen_now = max_seqlen_full
                block_attention_modes.append("full")
            else:
                cu_seqlens_now = cu_window_seqlens
                max_seqlen_now = max_seqlen_window
                block_attention_modes.append("window")

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                max_seqlen=max_seqlen_now,
            )
            block_outputs.append(_cpu_bf16(hidden_states.squeeze(1)))

        pre_merger = hidden_states.squeeze(1).clone()
        merged = visual.merger(hidden_states)
        final_output = merged[reverse_indices, :]

    return {
        "model_type": _qualname(model),
        "visual_type": _qualname(visual),
        "visual_dtype": str(visual.dtype),
        "num_blocks": len(block_outputs),
        "fullatt_block_indexes": list(map(int, visual.fullatt_block_indexes)),
        "block_attention_modes": block_attention_modes,
        "patch_embed": _cpu_bf16(patch_embed),
        "pre_blocks": _cpu_bf16(pre_blocks),
        "block_outputs": block_outputs,
        "pre_merger": _cpu_bf16(pre_merger),
        "final_output": _cpu_bf16(final_output),
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
            _extract_vllm_visual_block_states,
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
        "num_blocks": result["num_blocks"],
        "fullatt_block_indexes": result["fullatt_block_indexes"],
        "pre_blocks_shape": list(result["pre_blocks"].shape),
        "block_output_shape": list(result["block_outputs"][0].shape) if result["block_outputs"] else None,
        "pre_merger_shape": list(result["pre_merger"].shape),
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
    print("Export vLLM visual block states")
    print("=" * 72)
    print(f"Num blocks: {result['num_blocks']}")
    print(f"Pre-blocks shape: {list(result['pre_blocks'].shape)}")
    print(f"Block output shape: {list(result['block_outputs'][0].shape) if result['block_outputs'] else None}")
    print(f"Final output shape: {list(result['final_output'].shape)}")
    print(f"Saved PT to {output_pt}")
    if args.output_json:
        print(f"Saved JSON to {args.output_json}")
    print("=" * 72)


if __name__ == "__main__":
    main()

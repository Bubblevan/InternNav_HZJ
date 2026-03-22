import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from internnav.model.utils.dualvln_single_vllm import (  # noqa: E402
    TRAJ_TOKEN_INDEX,
    _compute_qwen2_5_vl_rope_index,
    _load_latent_queries_tensor,
)
from scripts.eval.tools.test_vllm_s2_equivalence import (  # noqa: E402
    build_messages,
    load_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="1-sample deep parity for HF generate_latents vs single-vLLM custom forward."
    )
    parser.add_argument(
        "--sample-pt",
        default="logs/habitat/hf_generate_latents_baseline_replay1/samples/sample_0000_zsNo4HB9uLZ_0001_step_0003.pt",
    )
    parser.add_argument(
        "--manifest",
        default="logs/habitat/test_dual_system_mini/replay_subset/manifest_rank0.jsonl",
    )
    parser.add_argument(
        "--hf-model-path",
        default="checkpoints/InternVLA-N1-DualVLN",
    )
    parser.add_argument(
        "--vllm-model-path",
        default="checkpoints/InternVLA-N1-DualVLN-qwen25vl-s2-view",
    )
    parser.add_argument(
        "--llm-output",
        default="311 222",
        help="Text used to reconstruct full_output_ids for all paths.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--limit-mm-per-prompt-image", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--output-json",
        default="logs/habitat/single_vllm_deep_parity_sample_0000.json",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
    )
    return parser.parse_args()


def _tensor_stats(t: torch.Tensor) -> dict:
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
    }


def _compare_tensors(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> dict:
    a_cpu = a.detach().float().cpu()
    b_cpu = b.detach().float().cpu()
    diff = (a_cpu - b_cpu).abs()
    cosine = None
    if a_cpu.numel() > 0 and b_cpu.numel() > 0:
        cosine = float(
            F.cosine_similarity(a_cpu.reshape(1, -1), b_cpu.reshape(1, -1)).item()
        )
    return {
        "a": _tensor_stats(a_cpu),
        "b": _tensor_stats(b_cpu),
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "cosine_similarity": cosine,
        "allclose": bool(torch.allclose(a_cpu, b_cpu, atol=atol, rtol=rtol)),
    }


def _build_unified_inputs(sample, manifest_path: Path, processor, llm_output: str):
    replay = load_manifest(str(manifest_path), base_path="logs")
    steps = replay[(sample["scene_id"], int(sample["episode_id"]))]
    target_item = None
    prev_llm = None
    for step in steps:
        if (
            int(step["step_id"]) == int(sample["step_id"])
            and step["baseline_output"]["output_kind"] == sample["baseline_output"]["output_kind"]
        ):
            target_item = step
            break
        prev_llm = step["baseline_output"]["llm_output"]
    if target_item is None:
        raise RuntimeError("Could not locate matching replay item for sample.")

    messages, input_images = build_messages(
        sample["instruction"],
        target_item,
        steps,
        num_history=8,
        is_lookdown=bool(sample["is_inferred_lookdown_followup"]),
        prev_llm_output=prev_llm,
    )
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    hf_inputs = processor(text=[text], images=input_images, return_tensors="pt")
    generated_ids = processor.tokenizer.encode(llm_output, add_special_tokens=False)
    full_output_ids = torch.cat(
        [hf_inputs.input_ids, torch.tensor([generated_ids], dtype=torch.long)],
        dim=1,
    )
    return {
        "messages": messages,
        "input_images": input_images,
        "hf_inputs": hf_inputs,
        "full_output_ids": full_output_ids,
        "generated_ids": generated_ids,
    }


def _trace_hf_manual(model, latent_queries_cpu, full_output_ids, pixel_values, image_grid_thw):
    device = next(model.parameters()).device
    full_output_ids = full_output_ids.to(device)
    pixel_values = pixel_values.to(device)
    image_grid_thw = image_grid_thw.to(device)

    with torch.no_grad():
        text_embeds_before_mm = model.get_input_embeddings()(full_output_ids)
        image_embeds = model.visual(pixel_values.type(model.visual.dtype), grid_thw=image_grid_thw).unsqueeze(0)
        image_idx = full_output_ids == model.config.image_token_id
        text_embeds_after_mm = text_embeds_before_mm.clone()
        text_embeds_after_mm[image_idx] = image_embeds.to(text_embeds_after_mm.dtype)[: image_idx.sum(), :]

        latent_queries = latent_queries_cpu.to(device=device, dtype=text_embeds_after_mm.dtype).unsqueeze(0)
        latent_queries = latent_queries.repeat(text_embeds_after_mm.shape[0], 1, 1)
        input_ids_with_traj = torch.cat(
            [
                full_output_ids,
                torch.full(
                    (full_output_ids.shape[0], latent_queries.shape[1]),
                    TRAJ_TOKEN_INDEX,
                    device=device,
                    dtype=full_output_ids.dtype,
                ),
            ],
            dim=1,
        )
        inputs_embeds_after_latent_queries = torch.cat([text_embeds_after_mm, latent_queries], dim=1)
        position_ids, _ = model.model.get_rope_index(input_ids_with_traj, image_grid_thw)
        outputs = model.model(
            inputs_embeds=inputs_embeds_after_latent_queries,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        final_hidden_states_last_layer = outputs.hidden_states[-1]
        first_layer_output = outputs.hidden_states[1]
        traj_latents = final_hidden_states_last_layer[:, -latent_queries.shape[1] :, :]

    return {
        "input_ids_with_traj": input_ids_with_traj.detach().cpu(),
        "text_embeds_before_mm": text_embeds_before_mm.detach().cpu(),
        "image_embeds": image_embeds.detach().cpu(),
        "text_embeds_after_mm": text_embeds_after_mm.detach().cpu(),
        "inputs_embeds_after_latent_queries": inputs_embeds_after_latent_queries.detach().cpu(),
        "position_ids": position_ids.detach().cpu(),
        "first_layer_output": first_layer_output.detach().cpu(),
        "final_hidden_states_last_layer": final_hidden_states_last_layer.detach().cpu(),
        "traj_latents": traj_latents.detach().cpu(),
    }


def _trace_hf_reference(model, latent_queries_cpu, full_output_ids, pixel_values, image_grid_thw):
    device = next(model.parameters()).device
    with torch.no_grad():
        full_output_ids = full_output_ids.to(device)
        pixel_values = pixel_values.to(device)
        image_grid_thw = image_grid_thw.to(device)
        text_embeds = model.get_input_embeddings()(full_output_ids)
        image_idx = full_output_ids == model.config.image_token_id
        n_query = int(latent_queries_cpu.shape[0])
        input_ids_with_traj = torch.cat(
            [
                full_output_ids,
                torch.full(
                    (full_output_ids.shape[0], n_query),
                    TRAJ_TOKEN_INDEX,
                    device=device,
                    dtype=full_output_ids.dtype,
                ),
            ],
            dim=1,
        )
        image_embeds = model.visual(pixel_values.type(model.visual.dtype), grid_thw=image_grid_thw).unsqueeze(0)
        text_embeds[image_idx] = image_embeds.to(text_embeds.dtype)[: image_idx.sum(), :]
        latent_queries = latent_queries_cpu.to(device=device, dtype=text_embeds.dtype).unsqueeze(0)
        inputs_embeds = torch.cat([text_embeds, latent_queries], dim=1)
        position_ids, _ = model.model.get_rope_index(input_ids_with_traj, image_grid_thw)
        outputs = model.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        traj_latents = outputs.hidden_states[-1][:, -n_query:, :]
    return traj_latents.detach().cpu()


def _trace_vllm_custom(
    model,
    full_output_ids_cpu,
    pixel_values_cpu,
    image_grid_thw_cpu,
    latent_queries_cpu,
):
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import get_forward_context, set_forward_context

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    n_query = int(latent_queries_cpu.shape[0])
    full_output_ids = full_output_ids_cpu.to(device)
    latent_queries = latent_queries_cpu.to(device=device, dtype=model_dtype)
    input_ids_with_traj = torch.cat(
        [
            full_output_ids,
            torch.full(
                (full_output_ids.shape[0], n_query),
                TRAJ_TOKEN_INDEX,
                dtype=full_output_ids.dtype,
                device=device,
            ),
        ],
        dim=1,
    )

    vllm_config = model.vllm_config
    with set_current_vllm_config(vllm_config), set_forward_context(
        None,
        vllm_config=vllm_config,
        num_tokens=input_ids_with_traj.shape[1],
    ), torch.inference_mode():
        forward_context = get_forward_context()
        text_embeds_before_mm = model.embed_input_ids(full_output_ids[0]).clone()
        multimodal_embeddings = model.embed_multimodal(
            pixel_values=pixel_values_cpu.to(device=device, dtype=model.visual.dtype),
            image_grid_thw=image_grid_thw_cpu.to(device=device),
        )
        flat_image_embeds = torch.cat(list(multimodal_embeddings), dim=0)

        text_embeds_after_mm = text_embeds_before_mm.clone()
        image_idx = full_output_ids[0] == model.config.image_token_id
        text_embeds_after_mm[image_idx] = flat_image_embeds[: int(image_idx.sum().item())].to(text_embeds_after_mm.dtype)

        inputs_embeds_after_latent_queries = model.embed_input_ids(input_ids_with_traj[0]).clone()
        inputs_embeds_after_latent_queries[: text_embeds_after_mm.shape[0]] = text_embeds_after_mm
        inputs_embeds_after_latent_queries[-n_query:] = latent_queries
        inputs_embeds_before_forward = inputs_embeds_after_latent_queries.clone()

        position_ids, _ = _compute_qwen2_5_vl_rope_index(
            input_ids_with_traj,
            config=model.config,
            image_grid_thw=image_grid_thw_cpu.to(device=device),
        )

        qwen_model = model.language_model.model
        positions_1d = position_ids[:, 0, :]
        hidden_states = inputs_embeds_after_latent_queries
        residual = None
        first_layer_output = None
        for idx, layer in enumerate(qwen_model.layers):
            hidden_states, residual = layer(positions_1d, hidden_states, residual)
            layer_output = hidden_states if residual is None else hidden_states + residual
            if idx == 0:
                first_layer_output = layer_output.clone()

        final_hidden_states_last_layer, _ = qwen_model.norm(hidden_states, residual)
        traj_latents = final_hidden_states_last_layer[-n_query:, :].unsqueeze(0)

    return {
        "input_ids_with_traj": input_ids_with_traj.detach().cpu(),
        "text_embeds_before_mm": text_embeds_before_mm.unsqueeze(0).detach().cpu(),
        "image_embeds": flat_image_embeds.unsqueeze(0).detach().cpu(),
        "text_embeds_after_mm": text_embeds_after_mm.unsqueeze(0).detach().cpu(),
        "inputs_embeds_after_latent_queries": inputs_embeds_before_forward.unsqueeze(0).detach().cpu(),
        "position_ids": position_ids.detach().cpu(),
        "first_layer_output": first_layer_output.unsqueeze(0).detach().cpu(),
        "final_hidden_states_last_layer": final_hidden_states_last_layer.unsqueeze(0).detach().cpu(),
        "traj_latents": traj_latents.detach().cpu(),
        "forward_context": {
            "attn_metadata_is_none": forward_context.attn_metadata is None,
            "slot_mapping_type": type(forward_context.slot_mapping).__name__,
            "slot_mapping_len": len(forward_context.slot_mapping),
        },
    }


def _load_hf_model(model_path: str):
    device = torch.device("cuda:0")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = model.to(device)
    model.eval()
    return model


def _load_vllm_model(args):
    from vllm import LLM

    llm = LLM(
        model=args.vllm_model_path,
        tensor_parallel_size=1,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": args.limit_mm_per_prompt_image},
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        seed=0,
        disable_log_stats=True,
    )
    return llm


def _free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    sample = torch.load(args.sample_pt, map_location="cpu")

    processor = AutoProcessor.from_pretrained(args.hf_model_path, use_fast=False)
    processor.tokenizer.padding_side = "left"
    unified = _build_unified_inputs(sample, Path(args.manifest), processor, args.llm_output)
    latent_queries = _load_latent_queries_tensor(args.hf_model_path)

    pixel_values = unified["hf_inputs"].pixel_values
    image_grid_thw = unified["hf_inputs"].image_grid_thw
    full_output_ids = unified["full_output_ids"]

    hf_model = _load_hf_model(args.vllm_model_path)
    hf_reference_latents = _trace_hf_reference(
        hf_model,
        latent_queries,
        full_output_ids,
        pixel_values,
        image_grid_thw,
    )
    hf_manual = _trace_hf_manual(
        hf_model,
        latent_queries,
        full_output_ids,
        pixel_values,
        image_grid_thw,
    )
    del hf_model
    _free_cuda()

    vllm_llm = _load_vllm_model(args)
    vllm_custom = vllm_llm.apply_model(
        lambda model: _trace_vllm_custom(
            model,
            full_output_ids,
            pixel_values,
            image_grid_thw,
            latent_queries,
        )
    )[0]
    del vllm_llm
    _free_cuda()

    comparisons = {
        "hf_reference_vs_hf_manual.traj_latents": _compare_tensors(
            hf_reference_latents,
            hf_manual["traj_latents"],
            args.atol,
            args.rtol,
        ),
        "hf_manual_vs_vllm_custom.input_ids_with_traj": _compare_tensors(
            hf_manual["input_ids_with_traj"],
            vllm_custom["input_ids_with_traj"],
            args.atol,
            args.rtol,
        ),
        "hf_manual_vs_vllm_custom.text_embeds_before_mm": _compare_tensors(
            hf_manual["text_embeds_before_mm"],
            vllm_custom["text_embeds_before_mm"],
            args.atol,
            args.rtol,
        ),
        "hf_manual_vs_vllm_custom.image_embeds": _compare_tensors(
            hf_manual["image_embeds"],
            vllm_custom["image_embeds"],
            args.atol,
            args.rtol,
        ),
        "hf_manual_vs_vllm_custom.text_embeds_after_mm": _compare_tensors(
            hf_manual["text_embeds_after_mm"],
            vllm_custom["text_embeds_after_mm"],
            args.atol,
            args.rtol,
        ),
        "hf_manual_vs_vllm_custom.inputs_embeds_after_latent_queries": _compare_tensors(
            hf_manual["inputs_embeds_after_latent_queries"],
            vllm_custom["inputs_embeds_after_latent_queries"],
            args.atol,
            args.rtol,
        ),
        "hf_manual_vs_vllm_custom.inputs_embeds_prefix_without_latent_queries": _compare_tensors(
            hf_manual["inputs_embeds_after_latent_queries"][:, :- latent_queries.shape[0], :],
            vllm_custom["inputs_embeds_after_latent_queries"][:, :- latent_queries.shape[0], :],
            args.atol,
            args.rtol,
        ),
        "hf_manual_vs_vllm_custom.inputs_embeds_latent_query_tail": _compare_tensors(
            hf_manual["inputs_embeds_after_latent_queries"][:, -latent_queries.shape[0] :, :],
            vllm_custom["inputs_embeds_after_latent_queries"][:, -latent_queries.shape[0] :, :],
            args.atol,
            args.rtol,
        ),
        "hf_manual_vs_vllm_custom.position_ids": _compare_tensors(
            hf_manual["position_ids"],
            vllm_custom["position_ids"],
            args.atol,
            args.rtol,
        ),
        "hf_manual_vs_vllm_custom.first_layer_output": _compare_tensors(
            hf_manual["first_layer_output"],
            vllm_custom["first_layer_output"],
            args.atol,
            args.rtol,
        ),
        "hf_manual_vs_vllm_custom.final_hidden_states_last_layer": _compare_tensors(
            hf_manual["final_hidden_states_last_layer"],
            vllm_custom["final_hidden_states_last_layer"],
            args.atol,
            args.rtol,
        ),
        "hf_manual_vs_vllm_custom.traj_latents": _compare_tensors(
            hf_manual["traj_latents"],
            vllm_custom["traj_latents"],
            args.atol,
            args.rtol,
        ),
    }

    first_mismatch = None
    ordered_keys = [
        "hf_manual_vs_vllm_custom.text_embeds_before_mm",
        "hf_manual_vs_vllm_custom.image_embeds",
        "hf_manual_vs_vllm_custom.text_embeds_after_mm",
        "hf_manual_vs_vllm_custom.inputs_embeds_after_latent_queries",
        "hf_manual_vs_vllm_custom.position_ids",
        "hf_manual_vs_vllm_custom.first_layer_output",
        "hf_manual_vs_vllm_custom.final_hidden_states_last_layer",
        "hf_manual_vs_vllm_custom.traj_latents",
    ]
    for key in ordered_keys:
        if not comparisons[key]["allclose"]:
            first_mismatch = {
                "name": key,
                "stats": comparisons[key],
            }
            break

    report = {
        "sample_pt": args.sample_pt,
        "manifest": args.manifest,
        "hf_model_path": args.hf_model_path,
        "vllm_model_path": args.vllm_model_path,
        "llm_output": args.llm_output,
        "generated_ids_len": len(unified["generated_ids"]),
        "full_output_ids_len": int(full_output_ids.shape[1]),
        "image_grid_thw": image_grid_thw.tolist(),
        "latent_queries_shape": list(latent_queries.shape),
        "vllm_forward_context": vllm_custom["forward_context"],
        "first_mismatch": first_mismatch,
        "comparisons": comparisons,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 72)
    print("Deep single-vLLM latent parity")
    print("=" * 72)
    print(f"llm_output: {args.llm_output}")
    print(f"full_output_ids_len: {int(full_output_ids.shape[1])}")
    print(f"first mismatch: {first_mismatch['name'] if first_mismatch else 'none'}")
    for key in ordered_keys:
        stats = comparisons[key]
        print(
            f"{key}: cos={stats['cosine_similarity']:.6f} "
            f"mean_abs={stats['mean_abs_diff']:.6f} "
            f"max_abs={stats['max_abs_diff']:.6f} "
            f"allclose={stats['allclose']}"
        )
    print(
        "HF reference vs HF manual traj_latents: "
        f"cos={comparisons['hf_reference_vs_hf_manual.traj_latents']['cosine_similarity']:.6f} "
        f"mean_abs={comparisons['hf_reference_vs_hf_manual.traj_latents']['mean_abs_diff']:.6f} "
        f"max_abs={comparisons['hf_reference_vs_hf_manual.traj_latents']['max_abs_diff']:.6f}"
    )
    print(f"attn_metadata_is_none: {vllm_custom['forward_context']['attn_metadata_is_none']}")
    print(f"Saved JSON to {output_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

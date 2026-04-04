import argparse
import gc
import json
import multiprocessing as mp
import os
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from internnav.model.utils.dualvln_single_vllm import (  # noqa: E402
    _load_latent_queries_tensor,
    to_vllm_chat_messages,
)
from internnav.model.utils.latents_request import (  # noqa: E402
    attach_explicit_mm_metadata_from_engine_core_request,
    build_latents_request_bundle,
)
from internnav.model.utils.vllm_latents_alignment import (  # noqa: E402
    _get_mm_feature_cache_key,
    _get_mm_item_data,
    _move_nested_to_device,
    _normalize_mm_kwargs_for_embed_multimodal,
    build_is_multimodal_mask,
    compute_mrope_positions_from_mm_features,
    materialize_mm_features_with_cached_data,
)
from scripts.eval.tools.test_vllm_s2_equivalence import (  # noqa: E402
    build_messages,
    load_manifest,
)


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Bisect where generate-model and pooling-model first diverge on the "
            "same token sequence and same resolved generation mm_features."
        )
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
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--limit-mm-per-prompt-image", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        default="logs/habitat/generate_vs_pooling_mm_bisect_sample_0000",
    )
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser.parse_args()


def _free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _json_ready(value):
    if torch.is_tensor(value):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _tensor_stats(tensor: torch.Tensor) -> dict:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def _compare_tensors(a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float) -> dict:
    a_cpu = a.detach().float().cpu()
    b_cpu = b.detach().float().cpu()
    diff = (a_cpu - b_cpu).abs()
    cosine = None
    if a_cpu.numel() and b_cpu.numel():
        cosine = float(F.cosine_similarity(a_cpu.reshape(1, -1), b_cpu.reshape(1, -1)).item())
    return {
        "a": _tensor_stats(a_cpu),
        "b": _tensor_stats(b_cpu),
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "cosine_similarity": cosine,
        "allclose": bool(torch.allclose(a_cpu, b_cpu, atol=atol, rtol=rtol)),
    }


def _compare_positions(a: torch.Tensor, b: torch.Tensor) -> dict:
    a_cpu = a.detach().cpu()
    b_cpu = b.detach().cpu()
    per_token_mismatch = (a_cpu != b_cpu).any(dim=0)
    mismatch_count = int(per_token_mismatch.sum().item())
    first_mismatch_index = None
    if mismatch_count:
        first_mismatch_index = int(torch.nonzero(per_token_mismatch, as_tuple=False)[0].item())
    return {
        "shape_a": list(a_cpu.shape),
        "shape_b": list(b_cpu.shape),
        "mismatch_count": mismatch_count,
        "first_mismatch_index": first_mismatch_index,
    }


def _build_replay_context(sample: dict, manifest_path: Path):
    replay = load_manifest(str(manifest_path), base_path="logs")
    steps = replay[(sample["scene_id"], int(sample["episode_id"]))]

    target_item = None
    prev_discrete_llm = None
    for item in steps:
        if (
            int(item["step_id"]) == int(sample["step_id"])
            and item["baseline_output"]["output_kind"] == sample["baseline_output"]["output_kind"]
        ):
            target_item = item
            break
        if item["baseline_output"]["output_kind"] != "pixel_goal":
            prev_discrete_llm = item["baseline_output"]["llm_output"]

    if target_item is None:
        raise RuntimeError("Could not locate matching replay item for sample.")

    lookdown = bool(sample.get("is_inferred_lookdown_followup", False))
    prev_llm_output = prev_discrete_llm if lookdown else None
    messages, input_images = build_messages(
        sample["instruction"],
        target_item,
        steps,
        num_history=8,
        is_lookdown=lookdown,
        prev_llm_output=prev_llm_output,
    )
    return {
        "messages": messages,
        "input_images": input_images,
    }


def _summarize_mm_features(mm_features) -> list[dict]:
    rows = []
    if not mm_features:
        return rows
    for idx, feature in enumerate(mm_features):
        rows.append(
            {
                "index": idx,
                "modality": feature.modality,
                "identifier": getattr(feature, "identifier", None),
                "mm_hash": getattr(feature, "mm_hash", None),
                "data_is_none": getattr(feature, "data", None) is None,
                "data_type": type(getattr(feature, "data", None)).__name__,
                "offset": int(feature.mm_position.offset),
                "length": int(feature.mm_position.length),
            }
        )
    return rows


def _summarize_normalized_payload(mm_kwargs: dict) -> dict:
    rows = {}
    for key, value in mm_kwargs.items():
        if torch.is_tensor(value):
            rows[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        else:
            rows[key] = {
                "type": type(value).__name__,
            }
    return rows


def _capture_generation_bundle_and_request(args, replay_context):
    from vllm import LLM, SamplingParams
    from vllm.outputs import RequestOutput

    processor = AutoProcessor.from_pretrained(
        args.hf_model_path,
        trust_remote_code=args.trust_remote_code,
    )
    latent_queries = _load_latent_queries_tensor(args.hf_model_path)

    llm = LLM(
        model=args.vllm_model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": args.limit_mm_per_prompt_image},
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        seed=0,
        disable_log_stats=True,
    )

    processed_prompt = llm._preprocess_chat_one(to_vllm_chat_messages(replay_context["messages"]))
    outputs = llm._render_and_run_requests(
        prompts=(processed_prompt,),
        params=[SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)],
        output_type=RequestOutput,
        use_tqdm=False,
    )
    request_output = outputs[0]
    completion = request_output.outputs[0]
    engine_core_request = llm.llm_engine.pop_debug_engine_core_request(request_output.request_id)
    if engine_core_request is None:
        raise RuntimeError("Failed to capture generation EngineCoreRequest for bisect.")

    bundle = build_latents_request_bundle(
        processor=processor,
        messages=replay_context["messages"],
        prompt_token_ids=list(request_output.prompt_token_ids or []),
        generated_token_ids=list(completion.token_ids),
        input_images=replay_context["input_images"],
        latent_queries=latent_queries,
        traj_token_index=151667,
        n_query=int(latent_queries.shape[0]),
    )
    attach_explicit_mm_metadata_from_engine_core_request(bundle, engine_core_request)

    return {
        "llm": llm,
        "processor": processor,
        "latent_queries": latent_queries,
        "bundle": bundle,
        "processed_prompt": processed_prompt,
        "request_output": request_output,
        "completion": completion,
        "engine_core_request": engine_core_request,
    }


def _trace_model_micro_bisect(
    model,
    *,
    prefill_token_ids,
    latent_queries_cpu,
    mm_features,
):
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context

    device = next(model.parameters()).device
    input_ids = torch.tensor(prefill_token_ids, device=device, dtype=torch.long)
    latent_queries = latent_queries_cpu.to(device=device)
    resolved_mm_features = materialize_mm_features_with_cached_data(mm_features)
    vllm_config = model.vllm_config

    mm_item_embeddings = []
    mm_item_payloads = []
    with set_current_vllm_config(vllm_config), set_forward_context(
        None,
        vllm_config=vllm_config,
        num_tokens=input_ids.shape[0],
    ), torch.inference_mode():
        text_only_embeds = model.embed_input_ids(input_ids).clone()

        for idx, mm_feature in enumerate(resolved_mm_features):
            mm_kwargs = {
                key: _move_nested_to_device(value, device=device)
                for key, value in _get_mm_item_data(mm_feature.data).items()
            }
            mm_kwargs = _normalize_mm_kwargs_for_embed_multimodal(mm_kwargs)
            item_embeddings = model.embed_multimodal(**mm_kwargs)
            if isinstance(item_embeddings, torch.Tensor):
                item_embeddings = item_embeddings
            else:
                if len(item_embeddings) != 1:
                    raise RuntimeError(
                        "Expected exactly one multimodal embedding tensor per feature, got "
                        f"{len(item_embeddings)} for idx={idx}."
                    )
                item_embeddings = item_embeddings[0]

            mm_item_embeddings.append(item_embeddings.detach().cpu())
            mm_item_payloads.append(
                {
                    "index": idx,
                    "cache_key": _get_mm_feature_cache_key(mm_feature),
                    "offset": int(mm_feature.mm_position.offset),
                    "length": int(mm_feature.mm_position.length),
                    "normalized_payload": _summarize_normalized_payload(mm_kwargs),
                }
            )

        is_multimodal = build_is_multimodal_mask(
            int(input_ids.shape[0]),
            resolved_mm_features,
            device=device,
        )
        prompt_embeds_before_latent = model.embed_input_ids(
            input_ids,
            multimodal_embeddings=tuple(item.to(device=device) for item in mm_item_embeddings),
            is_multimodal=is_multimodal,
        ).clone()
        final_prompt_embeds = prompt_embeds_before_latent.clone()
        final_prompt_embeds[-latent_queries.shape[0] :] = latent_queries.to(
            dtype=final_prompt_embeds.dtype
        )

        positions = compute_mrope_positions_from_mm_features(
            model=model,
            prompt_token_ids=prefill_token_ids,
            mm_features=resolved_mm_features,
            device=device,
        )
        if positions is None:
            raise RuntimeError("Expected M-RoPE positions for multimodal Qwen2.5-VL bisect.")

        hidden_states = model.forward(
            input_ids=None,
            positions=positions,
            inputs_embeds=final_prompt_embeds,
        )

    return {
        "model_type": type(model).__name__,
        "is_pooling_model": bool(getattr(model, "is_pooling_model", False)),
        "text_only_embeds": text_only_embeds.detach().cpu(),
        "mm_item_embeddings": mm_item_embeddings,
        "mm_item_payloads": mm_item_payloads,
        "prompt_embeds_before_latent": prompt_embeds_before_latent.detach().cpu(),
        "final_prompt_embeds": final_prompt_embeds.detach().cpu(),
        "positions": positions.detach().cpu(),
        "hidden_last4": hidden_states[-4:, :].detach().cpu(),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_path = PROJECT_ROOT / args.sample_pt
    manifest_path = PROJECT_ROOT / args.manifest
    sample = torch.load(sample_path, map_location="cpu")
    replay_context = _build_replay_context(sample, manifest_path)

    generation = _capture_generation_bundle_and_request(args, replay_context)
    bundle = generation["bundle"]
    prefill_token_ids = bundle.prefill_token_ids
    mm_features = bundle.mm_features
    latent_queries = bundle.latent_queries

    generate_trace = generation["llm"].apply_model(
        partial(
            _trace_model_micro_bisect,
            prefill_token_ids=prefill_token_ids,
            latent_queries_cpu=latent_queries,
            mm_features=mm_features,
        )
    )[0]

    del generation["llm"]
    _free_cuda()

    from vllm import LLM

    pooling_llm = LLM(
        model=args.vllm_model_path,
        runner="pooling",
        convert="embed",
        enable_prompt_embeds=True,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": args.limit_mm_per_prompt_image},
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        seed=0,
        disable_log_stats=True,
    )
    pooling_trace = pooling_llm.apply_model(
        partial(
            _trace_model_micro_bisect,
            prefill_token_ids=prefill_token_ids,
            latent_queries_cpu=latent_queries,
            mm_features=mm_features,
        )
    )[0]

    del pooling_llm
    _free_cuda()

    mm_item_diffs = []
    for idx, (generate_item, pooling_item) in enumerate(
        zip(generate_trace["mm_item_embeddings"], pooling_trace["mm_item_embeddings"], strict=True)
    ):
        mm_item_diffs.append(
            {
                "index": idx,
                "payload": generate_trace["mm_item_payloads"][idx],
                "compare": _compare_tensors(
                    generate_item,
                    pooling_item,
                    atol=args.atol,
                    rtol=args.rtol,
                ),
                "generate_first8_row0": generate_item[0, :8].float().tolist(),
                "pooling_first8_row0": pooling_item[0, :8].float().tolist(),
            }
        )

    report = {
        "metadata": {
            "sample_pt": str(sample_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "hf_model_path": str((PROJECT_ROOT / args.hf_model_path).resolve()),
            "vllm_model_path": str((PROJECT_ROOT / args.vllm_model_path).resolve()),
            "output_dir": str(output_dir.resolve()),
            "llm_output": generation["completion"].text,
            "prompt_token_count": len(bundle.prompt_token_ids),
            "generated_token_count": len(bundle.generated_token_ids),
            "prefill_token_count": len(prefill_token_ids),
        },
        "generation_request": {
            "request_output_id": generation["request_output"].request_id,
            "engine_core_request_id": generation["engine_core_request"].request_id,
            "engine_core_external_req_id": generation["engine_core_request"].external_req_id,
            "mm_features": _summarize_mm_features(mm_features),
            "resolved_mm_features": _summarize_mm_features(
                materialize_mm_features_with_cached_data(mm_features)
            ),
        },
        "model_identities": {
            "generate": {
                "model_type": generate_trace["model_type"],
                "is_pooling_model": generate_trace["is_pooling_model"],
            },
            "pooling": {
                "model_type": pooling_trace["model_type"],
                "is_pooling_model": pooling_trace["is_pooling_model"],
            },
        },
        "diffs": {
            "generate_text_only_embeds_vs_pooling_text_only_embeds": _compare_tensors(
                generate_trace["text_only_embeds"],
                pooling_trace["text_only_embeds"],
                atol=args.atol,
                rtol=args.rtol,
            ),
            "generate_mm_items_vs_pooling_mm_items": mm_item_diffs,
            "generate_prompt_embeds_before_latent_vs_pooling_prompt_embeds_before_latent": _compare_tensors(
                generate_trace["prompt_embeds_before_latent"],
                pooling_trace["prompt_embeds_before_latent"],
                atol=args.atol,
                rtol=args.rtol,
            ),
            "generate_final_prompt_embeds_vs_pooling_final_prompt_embeds": _compare_tensors(
                generate_trace["final_prompt_embeds"],
                pooling_trace["final_prompt_embeds"],
                atol=args.atol,
                rtol=args.rtol,
            ),
            "generate_positions_vs_pooling_positions": _compare_positions(
                generate_trace["positions"],
                pooling_trace["positions"],
            ),
            "generate_hidden_last4_vs_pooling_hidden_last4": _compare_tensors(
                generate_trace["hidden_last4"],
                pooling_trace["hidden_last4"],
                atol=args.atol,
                rtol=args.rtol,
            ),
        },
        "artifacts": {
            "generate_snapshot": str((output_dir / "generate_trace.snapshot.pt").resolve()),
            "pooling_snapshot": str((output_dir / "pooling_trace.snapshot.pt").resolve()),
        },
    }

    torch.save(generate_trace, output_dir / "generate_trace.snapshot.pt")
    torch.save(pooling_trace, output_dir / "pooling_trace.snapshot.pt")
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(_json_ready(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_ready(report), indent=2, ensure_ascii=False))
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()

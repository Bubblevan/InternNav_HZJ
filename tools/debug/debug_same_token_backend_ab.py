import argparse
import functools
import gc
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from internnav.model.basemodel.internvla_n1.internvla_n1 import (  # noqa: E402
    IMAGE_TOKEN_INDEX,
    InternVLAN1ForCausalLM,
)
from internnav.model.utils.dualvln_single_vllm import (  # noqa: E402
    DualVLNSingleVLLMRunner,
    _collect_step_s2_mm_debug,
    _compute_qwen2_5_vl_rope_index,
    to_vllm_chat_messages,
)
from internnav.model.utils.latents_request import (  # noqa: E402
    attach_explicit_mm_metadata,
    attach_explicit_mm_metadata_from_engine_core_request,
    build_latents_request_bundle,
)
from internnav.model.utils.vllm_hidden_latents import (  # noqa: E402
    _aggregate_records,
    _build_hf_like_prompt_embeds,
    _load_dump_records,
    _set_dump_env,
    _window_records,
)
from internnav.model.utils.vllm_latents_alignment import (  # noqa: E402
    build_prompt_embeds_with_mm_features,
    compute_mrope_positions_from_mm_features,
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
            "Run same-token latent backend A/B on the current step_s2 output and "
            "dump mm metadata, prompt embeds, full positions, and hidden-state diffs."
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
    parser.add_argument(
        "--output-dir",
        default="logs/habitat/same_token_backend_ab_sample_0000_real_engine_mm",
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


def _pick_record(records, start_ts: int, end_ts: int, tensor_key: str, first_dim: int):
    candidates = []
    for record in records:
        ts = record.get("ts_ns")
        if ts is None or not (start_ts <= ts <= end_ts):
            continue
        tensor = record.get("tensors", {}).get(tensor_key)
        if not torch.is_tensor(tensor):
            continue
        if tensor.ndim >= 1 and int(tensor.shape[0]) == int(first_dim):
            candidates.append(record)
    if not candidates:
        return None
    candidates.sort(
        key=lambda record: (
            record["tensors"][tensor_key].numel(),
            record.get("ts_ns", 0),
        )
    )
    return candidates[-1]


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


def _summarize_mm_placeholders(mm_placeholders) -> list[dict]:
    rows = []
    if not mm_placeholders:
        return rows
    for modality, placeholders in mm_placeholders.items():
        for idx, placeholder in enumerate(placeholders):
            rows.append(
                {
                    "modality": modality,
                    "index": idx,
                    "offset": int(placeholder.offset),
                    "length": int(placeholder.length),
                }
            )
    return rows


def _summarize_mm_features(mm_features) -> list[dict]:
    rows = []
    if not mm_features:
        return rows
    for idx, feature in enumerate(mm_features):
        position = feature.mm_position
        rows.append(
            {
                "index": idx,
                "modality": feature.modality,
                "identifier": getattr(feature, "identifier", None),
                "mm_hash": getattr(feature, "mm_hash", None),
                "data_is_none": getattr(feature, "data", None) is None,
                "data_type": type(getattr(feature, "data", None)).__name__,
                "offset": int(position.offset),
                "length": int(position.length),
            }
        )
    return rows


def _summarize_engine_core_request(engine_request) -> dict:
    if engine_request is None:
        return {
            "present": False,
            "prompt_token_ids_len": None,
            "mm_feature_count": None,
            "mm_features_data_none_count": None,
            "mm_features": [],
        }

    mm_features = list(engine_request.mm_features or [])
    return {
        "present": True,
        "request_id": engine_request.request_id,
        "external_req_id": engine_request.external_req_id,
        "prompt_token_ids_len": len(engine_request.prompt_token_ids or []),
        "mm_feature_count": len(mm_features),
        "mm_features_data_none_count": sum(
            getattr(feature, "data", None) is None for feature in mm_features
        ),
        "mm_features": _summarize_mm_features(mm_features),
    }


def _clone_bundle(bundle, *, clear_mm: bool):
    return type(bundle)(
        prompt_token_ids=list(bundle.prompt_token_ids),
        generated_token_ids=list(bundle.generated_token_ids),
        full_output_token_ids=list(bundle.full_output_token_ids),
        full_output_ids=bundle.full_output_ids.clone(),
        pixel_values=bundle.pixel_values.clone(),
        image_grid_thw=bundle.image_grid_thw.clone(),
        input_images=list(bundle.input_images),
        latent_queries=bundle.latent_queries.clone(),
        traj_token_index=int(bundle.traj_token_index),
        n_query=int(bundle.n_query),
        prompt_embeds=None if clear_mm else bundle.prompt_embeds,
        mm_kwargs=None if clear_mm else bundle.mm_kwargs,
        mm_hashes=None if clear_mm else bundle.mm_hashes,
        mm_placeholders=None if clear_mm else bundle.mm_placeholders,
        mm_features=None if clear_mm else bundle.mm_features,
    )


def _collect_processed_prompt_debug(processed_prompt, bundle) -> dict:
    mm_debug = _collect_step_s2_mm_debug(processed_prompt, bundle)
    mm_kwargs = processed_prompt.get("mm_kwargs") or {}
    image_kwargs = mm_kwargs.get("image") or []
    mm_debug["processed_prompt_image_kwargs_types"] = [
        type(item).__name__ if item is not None else "NoneType" for item in image_kwargs
    ]
    mm_debug["bundle_mm_placeholders"] = _summarize_mm_placeholders(bundle.mm_placeholders)
    return mm_debug


def _compare_token_id_lists(expected_ids: list[int], actual_ids: list[int], *, window: int = 16) -> dict:
    max_len = max(len(expected_ids), len(actual_ids))
    mismatch_indices = []
    for idx in range(max_len):
        expected = expected_ids[idx] if idx < len(expected_ids) else None
        actual = actual_ids[idx] if idx < len(actual_ids) else None
        if expected != actual:
            mismatch_indices.append(idx)

    first_mismatch_index = mismatch_indices[0] if mismatch_indices else None
    expected_window = None
    actual_window = None
    if first_mismatch_index is not None:
        start = max(0, first_mismatch_index - window)
        end = first_mismatch_index + window + 1
        expected_window = expected_ids[start:end]
        actual_window = actual_ids[start:end]

    return {
        "expected_len": len(expected_ids),
        "actual_len": len(actual_ids),
        "mismatch_count": len(mismatch_indices),
        "first_mismatch_index": first_mismatch_index,
        "expected_window_pm16": expected_window,
        "actual_window_pm16": actual_window,
        "expected_tail16": expected_ids[-16:],
        "actual_tail16": actual_ids[-16:],
    }


def _probe_generate_engine_explicit_metadata(bundle, llm) -> dict:
    from vllm.sampling_params import SamplingParams

    prompt = {
        "prompt_token_ids": bundle.prefill_token_ids,
        "multi_modal_data": {"image": bundle.input_images},
    }
    engine_request = llm.input_processor.process_inputs(
        request_id="internnav-generate-latents-prefill-generate-probe",
        prompt=prompt,
        params=SamplingParams(max_tokens=1, temperature=0.0),
        supported_tasks=tuple(llm.supported_tasks),
    )
    actual_prompt_token_ids = list(engine_request.prompt_token_ids or [])
    mm_features = list(engine_request.mm_features or [])
    return {
        "status": "ok",
        "supported_tasks": list(llm.supported_tasks),
        "prompt_token_ids_compare": _compare_token_id_lists(
            bundle.prefill_token_ids,
            actual_prompt_token_ids,
        ),
        "mm_feature_count": len(mm_features),
        "mm_features": _summarize_mm_features(mm_features),
    }


def _probe_generate_engine_direct_attach(bundle, llm) -> dict:
    probe_bundle = _clone_bundle(bundle, clear_mm=True)
    try:
        attach_explicit_mm_metadata(probe_bundle, llm)
    except Exception as exc:
        return {
            "status": "error",
            "type": type(exc).__name__,
            "message": str(exc),
        }
    return {
        "status": "ok",
        "mm_debug": _collect_processed_prompt_debug({"type": None}, probe_bundle),
    }


def _trace_hf_manual(
    model,
    *,
    full_output_ids_cpu: torch.Tensor,
    pixel_values_cpu: torch.Tensor,
    image_grid_thw_cpu: torch.Tensor,
    latent_queries_cpu: torch.Tensor,
):
    device = next(model.parameters()).device
    full_output_ids = full_output_ids_cpu.to(device)
    pixel_values = pixel_values_cpu.to(device)
    image_grid_thw = image_grid_thw_cpu.to(device)

    with torch.no_grad():
        text_embeds = model._embed_tokens(full_output_ids)
        image_idx = full_output_ids == IMAGE_TOKEN_INDEX
        image_embeds = model._extract_visual_features(
            model._get_visual()(
                pixel_values.type(model._get_visual().dtype),
                grid_thw=image_grid_thw,
            )
        ).unsqueeze(0)
        text_embeds[image_idx] = image_embeds.to(text_embeds.dtype)[: image_idx.sum(), :]

        latent_queries = latent_queries_cpu.to(
            device=device,
            dtype=text_embeds.dtype,
        ).unsqueeze(0)
        full_ids_with_traj = torch.cat(
            [
                full_output_ids,
                torch.full(
                    (full_output_ids.shape[0], latent_queries.shape[1]),
                    151667,
                    device=device,
                    dtype=full_output_ids.dtype,
                ),
            ],
            dim=1,
        )
        inputs_embeds = torch.cat([text_embeds, latent_queries], dim=1)
        position_ids, _ = model.get_rope_index(full_ids_with_traj, image_grid_thw)
        outputs = model.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        final_hidden_states = outputs.hidden_states[-1]
        latents = final_hidden_states[:, -latent_queries.shape[1] :, :]

    return {
        "prefill_token_ids": full_ids_with_traj[0].detach().cpu(),
        "inputs_embeds": inputs_embeds[0].detach().cpu(),
        "position_ids": position_ids[:, 0, :].detach().cpu(),
        "final_hidden_states_last4": final_hidden_states[0, -4:, :].detach().cpu(),
        "latents": latents.detach().cpu(),
    }


def _trace_shared_engine(model, bundle):
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context

    device = next(model.parameters()).device
    full_prompt_token_ids = bundle.full_output_token_ids + [bundle.traj_token_index] * bundle.n_query
    input_ids = torch.tensor(full_prompt_token_ids, device=device, dtype=torch.long).unsqueeze(0)
    vllm_config = model.vllm_config

    with set_current_vllm_config(vllm_config), set_forward_context(
        None,
        vllm_config=vllm_config,
        num_tokens=input_ids.shape[1],
    ), torch.inference_mode():
        if bundle.mm_features:
            embeds = build_prompt_embeds_with_mm_features(
                model=model,
                input_ids=input_ids[0],
                latent_queries=bundle.latent_queries,
                mm_features=bundle.mm_features,
            )
        else:
            embeds = model.embed_input_ids(input_ids[0]).clone()

            if bundle.pixel_values is not None and bundle.image_grid_thw is not None:
                pixel_values = bundle.pixel_values.to(device=device, dtype=model.visual.dtype)
                image_grid_thw = bundle.image_grid_thw.to(device=device)
                multimodal_embeddings = model.embed_multimodal(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )
                flat_mm_embeddings = (
                    torch.cat(list(multimodal_embeddings), dim=0)
                    if multimodal_embeddings
                    else None
                )
                if flat_mm_embeddings is not None:
                    image_idx = input_ids[0] == model.config.image_token_id
                    image_token_count = int(image_idx.sum().item())
                    embeds[image_idx] = flat_mm_embeddings[:image_token_count].to(embeds.dtype)

            latent_queries = bundle.latent_queries.to(device=device, dtype=embeds.dtype)
            embeds[-latent_queries.shape[0] :] = latent_queries

        position_ids = compute_mrope_positions_from_mm_features(
            model=model,
            prompt_token_ids=full_prompt_token_ids,
            mm_features=bundle.mm_features,
            device=device,
        )
        if position_ids is None:
            position_ids, _ = _compute_qwen2_5_vl_rope_index(
                input_ids,
                config=model.config,
                image_grid_thw=bundle.image_grid_thw.to(device=device),
            )
            position_ids = position_ids[:, 0, :]

        hidden_states = model.forward(
            input_ids=None,
            positions=position_ids,
            inputs_embeds=embeds,
        )

    return {
        "prefill_token_ids": input_ids[0].detach().cpu(),
        "inputs_embeds": embeds.detach().cpu(),
        "position_ids": position_ids.detach().cpu(),
        "final_hidden_states_last4": hidden_states[-4:, :].detach().cpu(),
        "latents": hidden_states[-bundle.n_query :, :].unsqueeze(0).detach().cpu(),
    }


def _run_native_like_separate_probe(
    bundle,
    *,
    input_images,
    model_path: str,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    limit_mm_per_prompt_image: int,
    tensor_parallel_size: int,
    trust_remote_code: bool,
    enforce_eager: bool,
    dump_dir: Path,
):
    dump_prefix = f"same_token_probe_{int(time.time() * 1000)}_{time.time_ns()}"
    dump_dir.mkdir(parents=True, exist_ok=True)
    _set_dump_env(dump_dir, dump_prefix)

    from vllm import LLM
    from vllm.inputs.data import EmbedsPrompt
    from vllm.pooling_params import PoolingParams

    llm = LLM(
        model=model_path,
        runner="pooling",
        convert="embed",
        enable_prompt_embeds=True,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": limit_mm_per_prompt_image},
        trust_remote_code=trust_remote_code,
        enforce_eager=enforce_eager,
        disable_log_stats=True,
    )

    prefill_token_ids = bundle.full_output_token_ids + [bundle.traj_token_index] * bundle.n_query
    prompt_embeds = llm.apply_model(
        functools.partial(
            _build_hf_like_prompt_embeds,
            prompt_token_ids=prefill_token_ids,
            pixel_values_cpu=bundle.pixel_values,
            image_grid_thw_cpu=bundle.image_grid_thw,
            latent_queries_cpu=bundle.latent_queries,
            mm_features=bundle.mm_features,
        )
    )[0]

    prompt = EmbedsPrompt(
        prompt_embeds=prompt_embeds,
        prompt_token_ids=prefill_token_ids,
        multi_modal_data={"image": input_images} if input_images else None,
    )

    start_ts = time.time_ns()
    outputs = llm.encode(
        [prompt],
        pooling_params=PoolingParams(
            task="token_embed",
            return_raw_hidden_states=True,
        ),
        pooling_task="token_embed",
        use_tqdm=False,
    )
    end_ts = time.time_ns()

    hidden_states = getattr(outputs[0].outputs, "hidden_states", None)

    records_prepare = _load_dump_records(dump_dir, dump_prefix, "gpu_model_runner_prepare_inputs")
    records_post = _load_dump_records(dump_dir, dump_prefix, "gpu_model_runner_actual_post_forward")
    records_request = _load_dump_records(dump_dir, dump_prefix, "gpu_input_batch_add_request")

    positions_record = _pick_record(
        records_prepare,
        start_ts,
        end_ts,
        "positions_gpu",
        3,
    )
    post_records = _window_records(records_post, start_ts, end_ts, "hidden_states")
    post_hidden = _aggregate_records(post_records, "hidden_states")
    request_record = _pick_record(
        records_request,
        start_ts,
        end_ts,
        "prompt_token_ids",
        len(prefill_token_ids),
    )

    if hidden_states is None:
        hidden_states = post_hidden
    else:
        hidden_states = hidden_states.detach().cpu()

    if hidden_states is None:
        raise RuntimeError("Failed to collect native-like separate hidden_states.")
    if positions_record is None:
        raise RuntimeError("Failed to collect native-like separate positions_gpu dump.")

    result = {
        "prefill_token_ids": (
            request_record["tensors"]["prompt_token_ids"].detach().cpu()
            if request_record is not None
            else torch.tensor(prefill_token_ids, dtype=torch.long)
        ),
        "prompt_embeds": prompt_embeds.detach().cpu(),
        "position_ids": positions_record["tensors"]["positions_gpu"].detach().cpu(),
        "final_hidden_states_last4": hidden_states[-4:, :].detach().cpu(),
        "latents": hidden_states[-bundle.n_query :, :].unsqueeze(0).detach().cpu(),
        "dump_paths": {
            "prepare_inputs": positions_record.get("_path"),
            "actual_post_forward": post_records[-1]["_path"] if post_records else None,
            "input_batch_add_request": request_record.get("_path") if request_record else None,
        },
        "dump_prefix": dump_prefix,
    }

    del llm
    _free_cuda()
    return result


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


def _window_tensor_rows(tensor: torch.Tensor, start: int, end: int) -> list[list[int]]:
    return tensor[:, start:end].detach().cpu().tolist()


def _position_windows_around_placeholders(
    *,
    shared_positions: torch.Tensor,
    separate_positions: torch.Tensor,
    hf_positions: torch.Tensor,
    mm_placeholders: list[dict],
    pad: int = 32,
) -> list[dict]:
    rows = []
    seq_len = int(shared_positions.shape[1])
    for item in mm_placeholders:
        start = max(0, int(item["offset"]) - pad)
        end = min(seq_len, int(item["offset"]) + int(item["length"]) + pad)
        rows.append(
            {
                "index": int(item["index"]),
                "offset": int(item["offset"]),
                "length": int(item["length"]),
                "window_start": start,
                "window_end": end,
                "shared": _window_tensor_rows(shared_positions, start, end),
                "separate": _window_tensor_rows(separate_positions, start, end),
                "hf": _window_tensor_rows(hf_positions, start, end),
            }
        )
    return rows


def _embed_placeholder_segments(
    *,
    shared_embeds: torch.Tensor,
    separate_embeds: torch.Tensor,
    mm_placeholders: list[dict],
) -> list[dict]:
    rows = []
    for item in mm_placeholders:
        offset = int(item["offset"])
        length = int(item["length"])
        start_idx = offset
        mid_idx = offset + length // 2
        end_idx = offset + length - 1
        shared_span = shared_embeds[offset : offset + length].float()
        separate_span = separate_embeds[offset : offset + length].float()
        diff = (shared_span - separate_span).abs()
        rows.append(
            {
                "index": int(item["index"]),
                "offset": offset,
                "length": length,
                "span_mean_abs_diff": float(diff.mean().item()),
                "span_max_abs_diff": float(diff.max().item()),
                "shared_start_first8": shared_embeds[start_idx, :8].float().tolist(),
                "separate_start_first8": separate_embeds[start_idx, :8].float().tolist(),
                "shared_mid_first8": shared_embeds[mid_idx, :8].float().tolist(),
                "separate_mid_first8": separate_embeds[mid_idx, :8].float().tolist(),
                "shared_end_first8": shared_embeds[end_idx, :8].float().tolist(),
                "separate_end_first8": separate_embeds[end_idx, :8].float().tolist(),
            }
        )
    return rows


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_path = PROJECT_ROOT / args.sample_pt
    manifest_path = PROJECT_ROOT / args.manifest
    sample = torch.load(sample_path, map_location="cpu")
    replay_context = _build_replay_context(sample, manifest_path)

    os.environ["INTERNNAV_DEBUG_STEP_S2_MM"] = "1"
    runner = DualVLNSingleVLLMRunner(
        model_path=args.vllm_model_path,
        hf_model_path=args.hf_model_path,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt_image=args.limit_mm_per_prompt_image,
        tensor_parallel_size=args.tensor_parallel_size,
        latent_backend="shared_engine_forward",
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        seed=0,
    )

    step_s2_result = runner.step_s2(replay_context["messages"], max_new_tokens=128)
    prompt_token_ids = list(step_s2_result["prompt_token_ids"])
    generated_token_ids = list(step_s2_result["generated_token_ids"])
    engine_core_request = runner._last_step_s2_engine_request

    processed_prompt = runner.llm._preprocess_chat_one(
        to_vllm_chat_messages(replay_context["messages"])
    )
    base_bundle = build_latents_request_bundle(
        processor=runner.processor,
        messages=replay_context["messages"],
        prompt_token_ids=prompt_token_ids,
        generated_token_ids=generated_token_ids,
        input_images=replay_context["input_images"],
        latent_queries=runner.latent_queries,
        traj_token_index=runner.traj_token_index,
        n_query=runner.n_query,
    )
    processed_prompt_mm_debug = dict(step_s2_result.get("debug_mm") or {})
    shared_bundle = _clone_bundle(base_bundle, clear_mm=True)
    attach_explicit_mm_metadata_from_engine_core_request(shared_bundle, engine_core_request)

    if list(processed_prompt.get("prompt_token_ids") or []) != prompt_token_ids:
        raise RuntimeError("processed_prompt prompt_token_ids do not match step_s2 prompt_token_ids.")
    if engine_core_request is None:
        raise RuntimeError("step_s2 did not capture a real generation EngineCoreRequest.")

    current_llm_output = step_s2_result["llm_output"]
    latent_queries = runner.latent_queries.detach().cpu()

    separate_trace = _run_native_like_separate_probe(
        shared_bundle,
        input_images=replay_context["input_images"],
        model_path=args.vllm_model_path,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt_image=args.limit_mm_per_prompt_image,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        dump_dir=output_dir / "separate_probe_dump",
    )
    engine_core_attach_debug = _collect_processed_prompt_debug(processed_prompt, shared_bundle)
    engine_core_attach_debug["bundle_mm_features"] = _summarize_mm_features(shared_bundle.mm_features)
    generation_engine_core_request = _summarize_engine_core_request(engine_core_request)
    generation_engine_core_request["prompt_token_ids_compare_vs_step_s2"] = _compare_token_id_lists(
        prompt_token_ids,
        list(engine_core_request.prompt_token_ids or []),
    )

    shared_trace = runner.llm.apply_model(
        functools.partial(_trace_shared_engine, bundle=shared_bundle)
    )[0]

    del runner
    _free_cuda()

    hf_model = InternVLAN1ForCausalLM.from_pretrained(
        args.hf_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=args.trust_remote_code,
    ).to(torch.device("cuda:0"))
    hf_model.eval()
    hf_trace = _trace_hf_manual(
        hf_model,
        full_output_ids_cpu=base_bundle.full_output_ids,
        pixel_values_cpu=base_bundle.pixel_values,
        image_grid_thw_cpu=base_bundle.image_grid_thw,
        latent_queries_cpu=latent_queries,
    )
    del hf_model
    _free_cuda()

    shared_vs_hf = _compare_tensors(
        shared_trace["latents"],
        hf_trace["latents"],
        atol=args.atol,
        rtol=args.rtol,
    )
    separate_vs_hf = _compare_tensors(
        separate_trace["latents"],
        hf_trace["latents"],
        atol=args.atol,
        rtol=args.rtol,
    )
    shared_vs_separate_embeds = _compare_tensors(
        shared_trace["inputs_embeds"],
        separate_trace["prompt_embeds"],
        atol=args.atol,
        rtol=args.rtol,
    )
    shared_vs_hf_positions = _compare_positions(shared_trace["position_ids"], hf_trace["position_ids"])
    separate_vs_hf_positions = _compare_positions(separate_trace["position_ids"], hf_trace["position_ids"])
    shared_vs_separate_positions = _compare_positions(
        shared_trace["position_ids"],
        separate_trace["position_ids"],
    )

    mm_placeholders = _summarize_mm_placeholders(shared_bundle.mm_placeholders)
    position_windows = _position_windows_around_placeholders(
        shared_positions=shared_trace["position_ids"],
        separate_positions=separate_trace["position_ids"],
        hf_positions=hf_trace["position_ids"],
        mm_placeholders=mm_placeholders,
    )
    embed_segments = _embed_placeholder_segments(
        shared_embeds=shared_trace["inputs_embeds"],
        separate_embeds=separate_trace["prompt_embeds"],
        mm_placeholders=mm_placeholders,
    )

    shared_snapshot_path = output_dir / "shared_current.snapshot.pt"
    separate_snapshot_path = output_dir / "separate_current.snapshot.pt"
    hf_snapshot_path = output_dir / "hf_current.snapshot.pt"

    torch.save(
        {
            "llm_output": current_llm_output,
            "full_output_token_ids": torch.tensor(shared_bundle.full_output_token_ids, dtype=torch.long),
            "prefill_token_ids": shared_trace["prefill_token_ids"],
            "mm_placeholders": mm_placeholders,
            "mm_features": engine_core_attach_debug["bundle_mm_features"],
            "inputs_embeds": shared_trace["inputs_embeds"],
            "position_ids": shared_trace["position_ids"],
            "final_hidden_states_last4": shared_trace["final_hidden_states_last4"],
            "latents": shared_trace["latents"],
        },
        shared_snapshot_path,
    )
    torch.save(
        {
            "llm_output": current_llm_output,
            "full_output_token_ids": torch.tensor(shared_bundle.full_output_token_ids, dtype=torch.long),
            "prefill_token_ids": separate_trace["prefill_token_ids"],
            "mm_placeholders": mm_placeholders,
            "prompt_embeds": separate_trace["prompt_embeds"],
            "position_ids": separate_trace["position_ids"],
            "final_hidden_states_last4": separate_trace["final_hidden_states_last4"],
            "latents": separate_trace["latents"],
            "dump_paths": separate_trace["dump_paths"],
        },
        separate_snapshot_path,
    )
    torch.save(
        {
            "llm_output": current_llm_output,
            "full_output_token_ids": torch.tensor(base_bundle.full_output_token_ids, dtype=torch.long),
            "prefill_token_ids": hf_trace["prefill_token_ids"],
            "inputs_embeds": hf_trace["inputs_embeds"],
            "position_ids": hf_trace["position_ids"],
            "final_hidden_states_last4": hf_trace["final_hidden_states_last4"],
            "latents": hf_trace["latents"],
        },
        hf_snapshot_path,
    )

    report = {
        "metadata": {
            "sample_pt": str(sample_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "hf_model_path": str((PROJECT_ROOT / args.hf_model_path).resolve()),
            "vllm_model_path": str((PROJECT_ROOT / args.vllm_model_path).resolve()),
            "output_dir": str(output_dir.resolve()),
            "llm_output": current_llm_output,
            "prompt_token_count": len(prompt_token_ids),
            "generated_token_count": len(generated_token_ids),
            "full_output_token_count": len(base_bundle.full_output_token_ids),
            "prefill_token_count": len(base_bundle.full_output_token_ids) + base_bundle.n_query,
            "shared_forward_mm_source": "generation_engine_core_request",
        },
        "group1_mm_features_after_attach": {
            "processed_prompt_path": processed_prompt_mm_debug,
            "generation_engine_core_request": generation_engine_core_request,
            "engine_core_request_attach_result": engine_core_attach_debug,
        },
        "group2_same_token_backend_ab": {
            "shared_engine_forward_with_engine_mm_vs_hf": shared_vs_hf,
            "native_like_separate_vs_hf": separate_vs_hf,
        },
        "group3_inputs_embeds_compare": {
            "shared_vs_native_like_separate": shared_vs_separate_embeds,
            "last4_shared_first8": shared_trace["inputs_embeds"][-4:, :8].float().tolist(),
            "last4_separate_first8": separate_trace["prompt_embeds"][-4:, :8].float().tolist(),
            "placeholder_embed_segments": embed_segments,
        },
        "group4_full_position_ids_compare": {
            "shared_vs_hf": shared_vs_hf_positions,
            "native_like_separate_vs_hf": separate_vs_hf_positions,
            "shared_vs_native_like_separate": shared_vs_separate_positions,
            "placeholder_windows_pm32": position_windows,
        },
        "artifacts": {
            "shared_snapshot": str(shared_snapshot_path.resolve()),
            "separate_snapshot": str(separate_snapshot_path.resolve()),
            "hf_snapshot": str(hf_snapshot_path.resolve()),
            "separate_dump_paths": separate_trace["dump_paths"],
        },
    }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(_json_ready(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(_json_ready(report), indent=2, ensure_ascii=False))
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()

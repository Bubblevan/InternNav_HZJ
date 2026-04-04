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
from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from internnav.model.basemodel.internvla_n1.internvla_n1 import (  # noqa: E402
    IMAGE_TOKEN_INDEX,
    InternVLAN1ForCausalLM,
)
from internnav.model.utils.dualvln_single_vllm import (  # noqa: E402
    TRAJ_TOKEN_INDEX,
    _compute_qwen2_5_vl_rope_index,
    _inspect_transformers_backend_model_tree,
    _load_latent_queries_tensor,
    to_vllm_chat_messages,
)
from internnav.model.utils.latents_request import (  # noqa: E402
    LatentsRequestBundle,
    attach_explicit_mm_metadata,
    attach_explicit_mm_metadata_from_processed_inputs,
    build_latents_request_bundle,
)
from internnav.model.utils.vllm_hidden_latents import (  # noqa: E402
    _build_hf_like_prompt_embeds,
    _load_dump_records,
    _set_dump_env,
    _window_records,
    _aggregate_records,
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
            "Run 1-sample latent backend A/B on shared_engine_forward vs "
            "vllm_hidden_separate_llm and dump comparable snapshots."
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
    parser.add_argument(
        "--llm-output",
        default=None,
        help="Overrides sample baseline llm_output when provided.",
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--limit-mm-per-prompt-image", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--skip-separate", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="logs/habitat/latent_backend_ab_sample_0000",
    )
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser.parse_args()


def _free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def _json_ready(value):
    if torch.is_tensor(value):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _build_replay_context(sample: dict, manifest_path: Path):
    replay = load_manifest(str(manifest_path), base_path="logs")
    steps = replay[(sample["scene_id"], int(sample["episode_id"]))]

    target_item = None
    prev_llm = None
    prev_discrete_llm = None
    prev_step_id = None
    for item in steps:
        if (
            int(item["step_id"]) == int(sample["step_id"])
            and item["baseline_output"]["output_kind"] == sample["baseline_output"]["output_kind"]
        ):
            target_item = item
            break
        if item["baseline_output"]["output_kind"] != "pixel_goal":
            prev_discrete_llm = item["baseline_output"]["llm_output"]
        prev_llm = item["baseline_output"]["llm_output"]
        prev_step_id = int(item["step_id"])

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
        "target_item": target_item,
        "steps": steps,
        "prev_llm": prev_llm,
        "prev_step_id": prev_step_id,
    }


def _clone_bundle(bundle: LatentsRequestBundle, *, clear_mm: bool) -> LatentsRequestBundle:
    return LatentsRequestBundle(
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


def _summarize_mm_placeholders(mm_placeholders) -> list[dict]:
    rows = []
    if not mm_placeholders:
        return rows
    for modality, placeholders in mm_placeholders.items():
        for idx, placeholder in enumerate(placeholders):
            is_embed = getattr(placeholder, "is_embed", None)
            is_embed_true = None
            is_embed_numel = None
            if is_embed is not None:
                is_embed_tensor = torch.as_tensor(is_embed, dtype=torch.bool).view(-1)
                is_embed_true = int(is_embed_tensor.sum().item())
                is_embed_numel = int(is_embed_tensor.numel())
            rows.append(
                {
                    "modality": modality,
                    "index": idx,
                    "offset": int(placeholder.offset),
                    "length": int(placeholder.length),
                    "is_embed_true": is_embed_true,
                    "is_embed_numel": is_embed_numel,
                }
            )
    return rows


def _summarize_mm_features(mm_features) -> list[dict]:
    rows = []
    if not mm_features:
        return rows
    for idx, feature in enumerate(mm_features):
        position = feature.mm_position
        is_embed = getattr(position, "is_embed", None)
        is_embed_true = None
        is_embed_numel = None
        if is_embed is not None:
            is_embed_tensor = torch.as_tensor(is_embed, dtype=torch.bool).view(-1)
            is_embed_true = int(is_embed_tensor.sum().item())
            is_embed_numel = int(is_embed_tensor.numel())
        rows.append(
            {
                "index": idx,
                "modality": feature.modality,
                "identifier": getattr(feature, "identifier", None),
                "mm_hash": getattr(feature, "mm_hash", None),
                "offset": int(position.offset),
                "length": int(position.length),
                "is_embed_true": is_embed_true,
                "is_embed_numel": is_embed_numel,
            }
        )
    return rows


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
                    TRAJ_TOKEN_INDEX,
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
        "position_ids_last16": position_ids[:, 0, -16:].detach().cpu(),
        "final_hidden_states_last4": final_hidden_states[0, -4:, :].detach().cpu(),
        "latents": latents.detach().cpu(),
    }


def _trace_shared_engine(model, bundle: LatentsRequestBundle):
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context

    device = next(model.parameters()).device
    full_prompt_token_ids = bundle.prefill_token_ids
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
        "position_ids_last16": position_ids[:, -16:].detach().cpu(),
        "final_hidden_states_last4": hidden_states[-4:, :].detach().cpu(),
        "latents": hidden_states[-bundle.n_query :, :].unsqueeze(0).detach().cpu(),
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


def _run_separate_backend(
    bundle: LatentsRequestBundle,
    *,
    model_path: str,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    limit_mm_per_prompt_image: int,
    tensor_parallel_size: int,
    trust_remote_code: bool,
    enforce_eager: bool,
    dump_dir: Path,
    strict_probe_bundle: LatentsRequestBundle | None = None,
):
    dump_prefix = f"latent_backend_ab_{int(time.time() * 1000)}_{time.time_ns()}"
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

    strict_probe = None
    if strict_probe_bundle is not None:
        try:
            attach_explicit_mm_metadata(strict_probe_bundle, llm)
            strict_probe = {
                "status": "ok",
                "mm_placeholders": _summarize_mm_placeholders(strict_probe_bundle.mm_placeholders),
                "mm_features": _summarize_mm_features(strict_probe_bundle.mm_features),
            }
        except Exception as exc:
            strict_probe = {
                "status": "error",
                "type": type(exc).__name__,
                "message": str(exc),
            }

    attach_explicit_mm_metadata(bundle, llm)
    prompt_embeds = llm.apply_model(
        functools.partial(
            _build_hf_like_prompt_embeds,
            prompt_token_ids=bundle.prefill_token_ids,
            pixel_values_cpu=bundle.pixel_values,
            image_grid_thw_cpu=bundle.image_grid_thw,
            latent_queries_cpu=bundle.latent_queries,
            mm_features=bundle.mm_features,
        )
    )[0]
    bundle.prompt_embeds = prompt_embeds

    prompt = EmbedsPrompt(
        prompt_embeds=prompt_embeds,
        prompt_token_ids=bundle.prefill_token_ids,
        mm_kwargs=bundle.mm_kwargs,
        mm_hashes=bundle.mm_hashes,
        mm_placeholders=bundle.mm_placeholders,
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
    positions_record = _pick_record(
        records_prepare,
        start_ts,
        end_ts,
        "positions_gpu",
        3,
    )
    post_records = _window_records(records_post, start_ts, end_ts, "hidden_states")
    post_hidden = _aggregate_records(post_records, "hidden_states")
    if hidden_states is None:
        hidden_states = post_hidden
    else:
        hidden_states = hidden_states.detach().cpu()

    if hidden_states is None:
        raise RuntimeError("Failed to collect hidden_states for vllm_hidden_separate_llm.")
    if positions_record is None:
        raise RuntimeError("Failed to collect positions_gpu dump for vllm_hidden_separate_llm.")

    llm_tree = llm.apply_model(_inspect_transformers_backend_model_tree)[0]
    del llm
    _free_cuda()

    return {
        "latents": hidden_states[-bundle.n_query :, :].unsqueeze(0),
        "position_ids_last16": positions_record["tensors"]["positions_gpu"][:, -16:].detach().cpu(),
        "final_hidden_states_last4": hidden_states[-4:, :].detach().cpu(),
        "model_tree": llm_tree,
        "dump_dir": str(dump_dir),
        "dump_prefix": dump_prefix,
        "strict_prefill_probe": strict_probe,
    }


def main():
    args = parse_args()

    sample_path = Path(args.sample_pt)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = torch.load(sample_path, map_location="cpu")
    replay_context = _build_replay_context(sample, manifest_path)
    latent_queries = _load_latent_queries_tensor(args.hf_model_path)
    processor = AutoProcessor.from_pretrained(
        args.hf_model_path,
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
    )
    processor.tokenizer.padding_side = "left"

    from vllm import LLM
    from vllm import SamplingParams
    from vllm.outputs import RequestOutput

    shared_llm = LLM(
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
    processed_prompt = shared_llm._preprocess_chat_one(
        to_vllm_chat_messages(replay_context["messages"])
    )
    request_outputs = shared_llm._render_and_run_requests(
        prompts=(processed_prompt,),
        params=[SamplingParams(max_tokens=128, temperature=0.0)],
        output_type=RequestOutput,
        use_tqdm=False,
    )
    request_output = request_outputs[0]
    completion = request_output.outputs[0]
    llm_output = completion.text
    prompt_token_ids = list(request_output.prompt_token_ids or [])
    generated_token_ids = list(completion.token_ids)
    if not generated_token_ids:
        raise RuntimeError("Current vLLM generation returned no output tokens for the target sample.")

    if args.llm_output is not None and args.llm_output != llm_output:
        raise RuntimeError(
            f"--llm-output={args.llm_output!r} does not match current vLLM output {llm_output!r}."
        )

    bundle = build_latents_request_bundle(
        processor=processor,
        messages=replay_context["messages"],
        prompt_token_ids=prompt_token_ids,
        generated_token_ids=generated_token_ids,
        input_images=replay_context["input_images"],
        latent_queries=latent_queries,
        traj_token_index=TRAJ_TOKEN_INDEX,
        n_query=int(latent_queries.shape[0]),
    )

    shared_bundle = _clone_bundle(bundle, clear_mm=True)
    attach_explicit_mm_metadata_from_processed_inputs(shared_bundle, processed_prompt)
    shared_tree = shared_llm.apply_model(_inspect_transformers_backend_model_tree)[0]
    shared_trace = shared_llm.apply_model(
        functools.partial(_trace_shared_engine, bundle=shared_bundle)
    )[0]
    del shared_llm
    _free_cuda()

    hf_model = InternVLAN1ForCausalLM.from_pretrained(
        args.hf_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(torch.device("cuda:0"))
    hf_model.eval()
    hf_trace = _trace_hf_manual(
        hf_model,
        full_output_ids_cpu=bundle.full_output_ids,
        pixel_values_cpu=bundle.pixel_values,
        image_grid_thw_cpu=bundle.image_grid_thw,
        latent_queries_cpu=bundle.latent_queries,
    )
    hf_vs_sample = None
    if "baseline_latent" in sample:
        hf_vs_sample = _compare_tensors(
            hf_trace["latents"],
            sample["baseline_latent"],
            atol=args.atol,
            rtol=args.rtol,
        )
    del hf_model
    _free_cuda()

    separate_trace = None
    separate_snapshot = None
    separate_snapshot_path = output_dir / "vllm_hidden_separate_llm.snapshot.pt"
    if not args.skip_separate:
        separate_bundle = _clone_bundle(shared_bundle, clear_mm=False)
        separate_dump_dir = output_dir / "vllm_hidden_separate_llm_dump"
        separate_trace = _run_separate_backend(
            separate_bundle,
            model_path=args.vllm_model_path,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            limit_mm_per_prompt_image=args.limit_mm_per_prompt_image,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=args.trust_remote_code,
            enforce_eager=args.enforce_eager,
            dump_dir=separate_dump_dir,
            strict_probe_bundle=_clone_bundle(bundle, clear_mm=True),
        )
        separate_snapshot = {
            "full_output_token_ids": torch.tensor(shared_bundle.full_output_token_ids, dtype=torch.long),
            "mm_placeholders": _summarize_mm_placeholders(separate_bundle.mm_placeholders),
            "mm_features": _summarize_mm_features(separate_bundle.mm_features),
            "position_ids_last16": separate_trace["position_ids_last16"],
            "final_hidden_states_last4": separate_trace["final_hidden_states_last4"],
        }

    shared_snapshot = {
        "full_output_token_ids": torch.tensor(shared_bundle.full_output_token_ids, dtype=torch.long),
        "mm_placeholders": _summarize_mm_placeholders(shared_bundle.mm_placeholders),
        "mm_features": _summarize_mm_features(shared_bundle.mm_features),
        "position_ids_last16": shared_trace["position_ids_last16"],
        "final_hidden_states_last4": shared_trace["final_hidden_states_last4"],
    }
    hf_snapshot = {
        "full_output_token_ids": torch.tensor(bundle.full_output_token_ids, dtype=torch.long),
        "position_ids_last16": hf_trace["position_ids_last16"],
        "final_hidden_states_last4": hf_trace["final_hidden_states_last4"],
        "latents": hf_trace["latents"],
    }

    shared_snapshot_path = output_dir / "shared_engine_forward.snapshot.pt"
    hf_snapshot_path = output_dir / "hf_reference.snapshot.pt"
    torch.save(shared_snapshot, shared_snapshot_path)
    if separate_snapshot is not None:
        torch.save(separate_snapshot, separate_snapshot_path)
    torch.save(hf_snapshot, hf_snapshot_path)

    shared_vs_hf = _compare_tensors(
        shared_trace["latents"],
        hf_trace["latents"],
        atol=args.atol,
        rtol=args.rtol,
    )
    separate_vs_hf = (
        _compare_tensors(
            separate_trace["latents"],
            hf_trace["latents"],
            atol=args.atol,
            rtol=args.rtol,
        )
        if separate_trace is not None
        else None
    )

    report = {
        "metadata": {
            "sample_pt": str(sample_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "hf_model_path": str((PROJECT_ROOT / args.hf_model_path).resolve())
            if not os.path.isabs(args.hf_model_path)
            else str(Path(args.hf_model_path).resolve()),
            "vllm_model_path": str((PROJECT_ROOT / args.vllm_model_path).resolve())
            if not os.path.isabs(args.vllm_model_path)
            else str(Path(args.vllm_model_path).resolve()),
            "output_dir": str(output_dir.resolve()),
            "llm_output": llm_output,
            "sample_id": {
                "scene_id": sample["scene_id"],
                "episode_id": int(sample["episode_id"]),
                "step_id": int(sample["step_id"]),
            },
            "current_vllm_output": llm_output,
            "sample_baseline_output": sample["baseline_output"]["llm_output"],
            "sample_hf_output": sample.get("hf_output_text"),
            "prompt_token_count": len(prompt_token_ids),
            "generated_token_count": len(generated_token_ids),
            "dtype": args.dtype,
            "gpu_memory_utilization": float(args.gpu_memory_utilization),
            "max_model_len": int(args.max_model_len),
            "limit_mm_per_prompt_image": int(args.limit_mm_per_prompt_image),
            "enforce_eager": bool(args.enforce_eager),
            "vllm_import_path": __import__("vllm").__file__,
        },
        "group1_backend_ab": {
            "hf_manual_vs_sample_baseline_latent": hf_vs_sample,
            "shared_engine_forward_vs_hf": shared_vs_hf,
            "vllm_hidden_separate_llm_vs_hf": separate_vs_hf,
            "closer_backend_by_mean_abs_diff": (
                "vllm_hidden_separate_llm"
                if separate_vs_hf is not None and separate_vs_hf["mean_abs_diff"] < shared_vs_hf["mean_abs_diff"]
                else "shared_engine_forward"
            ),
            "closer_backend_by_cosine_similarity": (
                "vllm_hidden_separate_llm"
                if separate_vs_hf is not None
                and (separate_vs_hf["cosine_similarity"] or float("-inf"))
                > (shared_vs_hf["cosine_similarity"] or float("-inf"))
                else "shared_engine_forward"
            ),
        },
        "group2_model_tree": {
            "shared_engine_apply_model_tree": shared_tree,
            "separate_pooling_apply_model_tree": (
                separate_trace["model_tree"] if separate_trace is not None else None
            ),
        },
        "group3_snapshots": {
            "shared_engine_forward": {
                "snapshot_path": str(shared_snapshot_path.resolve()),
                "mm_placeholders": shared_snapshot["mm_placeholders"],
                "mm_features": shared_snapshot["mm_features"],
                "position_ids_last16": _json_ready(shared_snapshot["position_ids_last16"]),
            },
            "vllm_hidden_separate_llm": {
                "snapshot_path": str(separate_snapshot_path.resolve()) if separate_snapshot is not None else None,
                "mm_placeholders": separate_snapshot["mm_placeholders"] if separate_snapshot is not None else None,
                "mm_features": separate_snapshot["mm_features"] if separate_snapshot is not None else None,
                "position_ids_last16": (
                    _json_ready(separate_snapshot["position_ids_last16"])
                    if separate_snapshot is not None
                    else None
                ),
                "strict_prefill_probe": (
                    separate_trace["strict_prefill_probe"] if separate_trace is not None else None
                ),
            },
            "hf_reference": {
                "snapshot_path": str(hf_snapshot_path.resolve()),
                "position_ids_last16": _json_ready(hf_snapshot["position_ids_last16"]),
            },
        },
    }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_path = output_dir / "report.md"
    md_lines = [
        "# Latent Backend A/B Result",
        "",
        f"- Sample: `{sample['scene_id']}` / ep `{int(sample['episode_id'])}` / step `{int(sample['step_id'])}`",
        f"- current vLLM llm_output: `{llm_output}`",
        f"- sample baseline llm_output: `{sample['baseline_output']['llm_output']}`",
        "",
        "## Group 1: Backend A/B vs HF",
        "",
        f"- `shared_engine_forward`: cosine `{shared_vs_hf['cosine_similarity']:.6f}`, mean_abs `{shared_vs_hf['mean_abs_diff']:.6f}`, max_abs `{shared_vs_hf['max_abs_diff']:.6f}`",
        (
            f"- `vllm_hidden_separate_llm`: cosine `{separate_vs_hf['cosine_similarity']:.6f}`, "
            f"mean_abs `{separate_vs_hf['mean_abs_diff']:.6f}`, max_abs `{separate_vs_hf['max_abs_diff']:.6f}`"
            if separate_vs_hf is not None
            else "- `vllm_hidden_separate_llm`: skipped in this run"
        ),
        "",
        "## Group 2: Model Tree",
        "",
        "```json",
        json.dumps(report["group2_model_tree"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Group 3: Snapshot Files",
        "",
        f"- shared_engine_forward: `{shared_snapshot_path.resolve()}`",
        (
            f"- vllm_hidden_separate_llm: `{separate_snapshot_path.resolve()}`"
            if separate_snapshot is not None
            else "- vllm_hidden_separate_llm: skipped in this run"
        ),
        f"- hf_reference: `{hf_snapshot_path.resolve()}`",
        "",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps(report["group1_backend_ab"], indent=2, ensure_ascii=False))
    print(f"Saved report to {report_path}")
    print(f"Saved markdown to {md_path}")


if __name__ == "__main__":
    main()

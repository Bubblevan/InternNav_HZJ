import argparse
import gc
import json
import multiprocessing as mp
import os
import sys
import time
from enum import Enum
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from internnav.model.utils.vllm_hidden_latents import (  # noqa: E402
    _build_hf_like_prompt_embeds,
    _aggregate_records,
    _load_dump_records,
    _set_dump_env,
    _window_records,
)
from internnav.model.utils.vllm_latents_alignment import (  # noqa: E402
    compute_mrope_positions_from_mm_features,
)
from tools.debug.debug_generate_vs_pooling_mm_bisect import (  # noqa: E402
    _capture_generation_bundle_and_request,
    _compare_positions,
    _compare_tensors,
    _free_cuda,
    _json_ready,
    _tensor_stats,
    _build_replay_context,
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
            "Runtime bisect for native pooling encode vs manual forward on the "
            "same pooling LLM and same request."
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
        default="logs/habitat/pooling_runtime_bisect_sample_0000",
    )
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser.parse_args()


def _pick_prepare_record(records, start_ts: int, end_ts: int):
    candidates = []
    for record in records:
        ts = record.get("ts_ns")
        if ts is None or not (start_ts <= ts <= end_ts):
            continue
        tensors = record.get("tensors", {})
        if not torch.is_tensor(tensors.get("inputs_embeds_gpu")):
            continue
        if not torch.is_tensor(tensors.get("positions_gpu")):
            continue
        candidates.append(record)

    if not candidates:
        return None
    candidates.sort(key=lambda record: record.get("ts_ns", 0))
    return candidates[-1]


def _summarize_bool_tensor(value: torch.Tensor | None) -> dict | None:
    if value is None:
        return None
    value = value.detach().cpu().bool().view(-1)
    return {
        "shape": list(value.shape),
        "num_true": int(value.sum().item()),
        "num_false": int((~value).sum().item()),
        "head16": value[:16].tolist(),
        "tail16": value[-16:].tolist(),
    }


def _manual_prompt_inputs_from_pooling_model(
    model,
    *,
    prefill_token_ids,
    pixel_values_cpu,
    image_grid_thw_cpu,
    latent_queries_cpu,
    mm_features,
):
    device = next(model.parameters()).device
    prompt_embeds = _build_hf_like_prompt_embeds(
        model,
        prompt_token_ids=prefill_token_ids,
        pixel_values_cpu=pixel_values_cpu,
        image_grid_thw_cpu=image_grid_thw_cpu,
        latent_queries_cpu=latent_queries_cpu,
        mm_features=mm_features,
    ).to(device=device)
    positions = compute_mrope_positions_from_mm_features(
        model=model,
        prompt_token_ids=prefill_token_ids,
        mm_features=mm_features,
        device=device,
    )
    if positions is None:
        raise RuntimeError("Expected mrope positions for pooling runtime bisect.")
    return {
        "prompt_embeds": prompt_embeds.detach().cpu(),
        "positions": positions.detach().cpu(),
    }


def _manual_forward_from_runtime_inputs(
    model,
    *,
    inputs_embeds_cpu,
    positions_cpu,
):
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context

    device = next(model.parameters()).device
    vllm_config = model.vllm_config
    inputs_embeds = inputs_embeds_cpu.to(device=device)
    positions = positions_cpu.to(device=device)

    with set_current_vllm_config(vllm_config), set_forward_context(
        None,
        vllm_config=vllm_config,
        num_tokens=int(inputs_embeds.shape[0]),
    ), torch.inference_mode():
        hidden_states = model.forward(
            input_ids=None,
            positions=positions,
            inputs_embeds=inputs_embeds,
        )
    return hidden_states.detach().cpu()


_CONTEXT_KEY_SUBSTRINGS = (
    "query_start_loc",
    "seq_lens",
    "slot_mapping",
    "block_table",
    "cu_seq",
    "cum_seq",
    "qo_indptr",
    "paged_kv",
)


def _should_capture_context_tensor(path: str) -> bool:
    if path.startswith("slot_mapping"):
        return True
    return any(token in path for token in _CONTEXT_KEY_SUBSTRINGS)


def _walk_context_value(
    value,
    path: str,
    *,
    tensors: dict,
    scalars: dict,
    visited: set[int],
    depth: int = 0,
    max_depth: int = 6,
):
    if value is None:
        return

    if torch.is_tensor(value):
        if _should_capture_context_tensor(path):
            tensors[path] = value.detach().cpu()
        return

    if isinstance(value, Enum):
        scalars[path] = str(value)
        return

    if isinstance(value, (str, int, float, bool)):
        scalars[path] = value
        return

    if depth >= max_depth:
        scalars[f"{path}.__type__"] = f"{type(value).__module__}.{type(value).__name__}"
        return

    object_id = id(value)
    if object_id in visited:
        return
    visited.add(object_id)

    if isinstance(value, dict):
        scalars[f"{path}.__type__"] = type(value).__name__
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            _walk_context_value(
                item,
                child,
                tensors=tensors,
                scalars=scalars,
                visited=visited,
                depth=depth + 1,
                max_depth=max_depth,
            )
        return

    if isinstance(value, (list, tuple)):
        scalars[f"{path}.__type__"] = type(value).__name__
        for idx, item in enumerate(value):
            child = f"{path}[{idx}]"
            _walk_context_value(
                item,
                child,
                tensors=tensors,
                scalars=scalars,
                visited=visited,
                depth=depth + 1,
                max_depth=max_depth,
            )
        return

    if hasattr(value, "__dict__"):
        scalars[f"{path}.__type__"] = f"{type(value).__module__}.{type(value).__name__}"
        for name, item in vars(value).items():
            if name.startswith("_"):
                continue
            child = f"{path}.{name}" if path else name
            _walk_context_value(
                item,
                child,
                tensors=tensors,
                scalars=scalars,
                visited=visited,
                depth=depth + 1,
                max_depth=max_depth,
            )


def _worker_capture_last_forward_context(worker):
    from vllm.forward_context import get_last_debug_forward_context

    forward_context = get_last_debug_forward_context()
    if forward_context is None:
        raise RuntimeError("No captured forward context is available on the worker.")

    tensors: dict[str, torch.Tensor] = {}
    scalars: dict[str, object] = {}
    visited: set[int] = set()

    _walk_context_value(
        forward_context.attn_metadata,
        "attn_metadata",
        tensors=tensors,
        scalars=scalars,
        visited=visited,
    )
    _walk_context_value(
        forward_context.slot_mapping,
        "slot_mapping",
        tensors=tensors,
        scalars=scalars,
        visited=visited,
    )
    _walk_context_value(
        forward_context.batch_descriptor,
        "batch_descriptor",
        tensors=tensors,
        scalars=scalars,
        visited=visited,
    )
    _walk_context_value(
        forward_context.additional_kwargs,
        "additional_kwargs",
        tensors=tensors,
        scalars=scalars,
        visited=visited,
    )

    return {
        "meta": {
            "attn_metadata_type": f"{type(forward_context.attn_metadata).__module__}.{type(forward_context.attn_metadata).__name__}",
            "slot_mapping_type": f"{type(forward_context.slot_mapping).__module__}.{type(forward_context.slot_mapping).__name__}",
            "batch_descriptor_type": (
                None
                if forward_context.batch_descriptor is None
                else f"{type(forward_context.batch_descriptor).__module__}.{type(forward_context.batch_descriptor).__name__}"
            ),
            "cudagraph_runtime_mode": str(forward_context.cudagraph_runtime_mode),
            "skip_compiled": bool(forward_context.skip_compiled),
            "virtual_engine": int(forward_context.virtual_engine),
            "additional_kwargs_keys": sorted(forward_context.additional_kwargs.keys()),
            "scalar_fields": scalars,
        },
        "tensors": tensors,
    }


def _worker_manual_forward_with_last_context(
    worker,
    prepare_snapshot_path,
):
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import (
        get_last_debug_forward_context,
        override_forward_context,
    )

    forward_context = get_last_debug_forward_context()
    if forward_context is None:
        raise RuntimeError("No captured forward context is available on the worker.")

    model = worker.get_model()
    model_runner = worker.model_runner
    device = next(model.parameters()).device
    prepare_snapshot = torch.load(prepare_snapshot_path, map_location="cpu")
    inputs_embeds_cpu = prepare_snapshot["inputs_embeds_gpu"]
    positions_cpu = prepare_snapshot["positions_gpu"]
    inputs_embeds = inputs_embeds_cpu.to(device=device)
    positions = positions_cpu.to(device=device)
    model_kwargs = model_runner._init_model_kwargs()

    with (
        set_current_vllm_config(model_runner.vllm_config),
        override_forward_context(forward_context),
        torch.inference_mode(),
    ):
        hidden_states = model(
            input_ids=None,
            positions=positions,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

    model_kwargs_cpu = {}
    for key, value in model_kwargs.items():
        model_kwargs_cpu[key] = value.detach().cpu() if torch.is_tensor(value) else value

    return {
        "hidden_states": hidden_states.detach().cpu(),
        "model_kwargs": model_kwargs_cpu,
        "model_kwargs_keys": sorted(model_kwargs_cpu.keys()),
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

    del generation["llm"]
    _free_cuda()

    dump_prefix = f"pooling_runtime_bisect_{int(time.time() * 1000)}_{time.time_ns()}"
    dump_dir = output_dir / "pooling_dump"
    dump_dir.mkdir(parents=True, exist_ok=True)
    _set_dump_env(dump_dir, dump_prefix)

    from vllm import LLM
    from vllm.inputs.data import EmbedsPrompt
    from vllm.pooling_params import PoolingParams

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

    manual_prompt = pooling_llm.apply_model(
        partial(
            _manual_prompt_inputs_from_pooling_model,
            prefill_token_ids=bundle.prefill_token_ids,
            pixel_values_cpu=bundle.pixel_values,
            image_grid_thw_cpu=bundle.image_grid_thw,
            latent_queries_cpu=bundle.latent_queries,
            mm_features=bundle.mm_features,
        )
    )[0]

    prompt = EmbedsPrompt(
        prompt_embeds=manual_prompt["prompt_embeds"],
        prompt_token_ids=bundle.prefill_token_ids,
        multi_modal_data={"image": bundle.input_images} if bundle.input_images else None,
    )

    start_ts = time.time_ns()
    outputs = pooling_llm.encode(
        [prompt],
        pooling_params=PoolingParams(task="token_embed", return_raw_hidden_states=True),
        pooling_task="token_embed",
        use_tqdm=False,
    )
    end_ts = time.time_ns()

    native_return_hidden_states = getattr(outputs[0].outputs, "hidden_states", None)
    if native_return_hidden_states is not None:
        native_return_hidden_states = native_return_hidden_states.detach().cpu()

    forward_context_snapshot = pooling_llm.collective_rpc(
        _worker_capture_last_forward_context
    )[0]

    records_prepare = _load_dump_records(dump_dir, dump_prefix, "gpu_model_runner_prepare_inputs")
    prepare_record = _pick_prepare_record(records_prepare, start_ts, end_ts)
    if prepare_record is None:
        raise RuntimeError("Failed to capture native pooling prepare_inputs dump.")

    native_inputs_embeds = prepare_record["tensors"]["inputs_embeds_gpu"].detach().cpu()
    native_positions = prepare_record["tensors"]["positions_gpu"].detach().cpu()
    native_is_token_ids = prepare_record["tensors"].get("is_token_ids_gpu")
    if torch.is_tensor(native_is_token_ids):
        native_is_token_ids = native_is_token_ids.detach().cpu()
    else:
        native_is_token_ids = None

    native_prepare_snapshot_path = output_dir / "native_prepare.snapshot.pt"
    torch.save(
        {
            "inputs_embeds_gpu": native_inputs_embeds,
            "positions_gpu": native_positions,
            "is_token_ids_gpu": native_is_token_ids,
        },
        native_prepare_snapshot_path,
    )

    records_post = _load_dump_records(dump_dir, dump_prefix, "gpu_model_runner_actual_post_forward")
    post_records = _window_records(records_post, start_ts, end_ts, "hidden_states")
    native_post_hidden_states = _aggregate_records(post_records, "hidden_states")
    if native_post_hidden_states is None:
        raise RuntimeError("Failed to capture native pooling post-forward hidden_states.")

    manual_forward_reused_context = pooling_llm.collective_rpc(
        _worker_manual_forward_with_last_context,
        args=(str(native_prepare_snapshot_path),),
    )[0]
    manual_forward_reused_context_hidden_states = manual_forward_reused_context[
        "hidden_states"
    ]

    manual_forward_hidden_states = pooling_llm.apply_model(
        partial(
            _manual_forward_from_runtime_inputs,
            inputs_embeds_cpu=native_inputs_embeds,
            positions_cpu=native_positions,
        )
    )[0]

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
            "prefill_token_count": len(bundle.prefill_token_ids),
            "dump_prefix": dump_prefix,
            "native_prompt_mode": "multi_modal_data",
        },
        "step1_native_prepare_inputs_vs_manual": {
            "manual_prompt_embeds": _tensor_stats(manual_prompt["prompt_embeds"]),
            "native_inputs_embeds": _tensor_stats(native_inputs_embeds),
            "prompt_embeds_compare": _compare_tensors(
                manual_prompt["prompt_embeds"],
                native_inputs_embeds,
                atol=args.atol,
                rtol=args.rtol,
            ),
            "manual_positions": _tensor_stats(manual_prompt["positions"]),
            "native_positions": _tensor_stats(native_positions),
            "positions_compare": _compare_positions(
                manual_prompt["positions"],
                native_positions,
            ),
            "native_is_token_ids_gpu": _summarize_bool_tensor(native_is_token_ids),
            "prepare_record_path": prepare_record.get("_path"),
        },
        "step2_forward_context_dump": {
            "context_meta": forward_context_snapshot["meta"],
            "captured_tensor_keys": sorted(forward_context_snapshot["tensors"].keys()),
            "captured_tensor_stats": {
                key: _tensor_stats(value)
                for key, value in sorted(forward_context_snapshot["tensors"].items())
            },
        },
        "step3_native_forward_vs_manual_forward_reused_context": {
            "manual_forward_reused_context_hidden_states": _tensor_stats(
                manual_forward_reused_context_hidden_states
            ),
            "model_kwargs_keys": manual_forward_reused_context["model_kwargs_keys"],
            "native_vs_reused_context_full": _compare_tensors(
                native_post_hidden_states,
                manual_forward_reused_context_hidden_states,
                atol=args.atol,
                rtol=args.rtol,
            ),
            "native_vs_reused_context_last4": _compare_tensors(
                native_post_hidden_states[-4:, :],
                manual_forward_reused_context_hidden_states[-4:, :],
                atol=args.atol,
                rtol=args.rtol,
            ),
        },
        "step4_native_forward_vs_manual_forward_fresh_context": {
            "native_post_hidden_states": _tensor_stats(native_post_hidden_states),
            "manual_forward_hidden_states": _tensor_stats(manual_forward_hidden_states),
            "full_hidden_states_compare": _compare_tensors(
                native_post_hidden_states,
                manual_forward_hidden_states,
                atol=args.atol,
                rtol=args.rtol,
            ),
            "last4_compare": _compare_tensors(
                native_post_hidden_states[-4:, :],
                manual_forward_hidden_states[-4:, :],
                atol=args.atol,
                rtol=args.rtol,
            ),
            "native_post_forward_path": post_records[-1]["_path"] if post_records else None,
            "fresh_vs_reused_context_full": _compare_tensors(
                manual_forward_hidden_states,
                manual_forward_reused_context_hidden_states,
                atol=args.atol,
                rtol=args.rtol,
            ),
        },
        "step5_native_output_extraction": {
            "native_return_hidden_states_present": native_return_hidden_states is not None,
            "returned_vs_post_forward_full": (
                _compare_tensors(
                    native_return_hidden_states,
                    native_post_hidden_states,
                    atol=args.atol,
                    rtol=args.rtol,
                )
                if native_return_hidden_states is not None
                else None
            ),
            "returned_vs_manual_forward_full": (
                _compare_tensors(
                    native_return_hidden_states,
                    manual_forward_hidden_states,
                    atol=args.atol,
                    rtol=args.rtol,
                )
                if native_return_hidden_states is not None
                else None
            ),
            "returned_vs_reused_context_full": (
                _compare_tensors(
                    native_return_hidden_states,
                    manual_forward_reused_context_hidden_states,
                    atol=args.atol,
                    rtol=args.rtol,
                )
                if native_return_hidden_states is not None
                else None
            ),
            "returned_vs_post_forward_last4": (
                _compare_tensors(
                    native_return_hidden_states[-4:, :],
                    native_post_hidden_states[-4:, :],
                    atol=args.atol,
                    rtol=args.rtol,
                )
                if native_return_hidden_states is not None
                else None
            ),
        },
        "artifacts": {
            "manual_prompt_snapshot": str((output_dir / "manual_prompt.snapshot.pt").resolve()),
            "native_prepare_snapshot": str((output_dir / "native_prepare.snapshot.pt").resolve()),
            "forward_context_snapshot": str((output_dir / "forward_context.snapshot.pt").resolve()),
            "native_post_forward_snapshot": str((output_dir / "native_post_forward.snapshot.pt").resolve()),
            "manual_forward_reused_context_snapshot": str((output_dir / "manual_forward_reused_context.snapshot.pt").resolve()),
            "manual_forward_snapshot": str((output_dir / "manual_forward.snapshot.pt").resolve()),
        },
    }

    torch.save(manual_prompt, output_dir / "manual_prompt.snapshot.pt")
    torch.save(
        {
            "inputs_embeds_gpu": native_inputs_embeds,
            "positions_gpu": native_positions,
            "is_token_ids_gpu": native_is_token_ids,
        },
        output_dir / "native_prepare.snapshot.pt",
    )
    torch.save(forward_context_snapshot, output_dir / "forward_context.snapshot.pt")
    torch.save(native_post_hidden_states, output_dir / "native_post_forward.snapshot.pt")
    torch.save(
        manual_forward_reused_context,
        output_dir / "manual_forward_reused_context.snapshot.pt",
    )
    torch.save(manual_forward_hidden_states, output_dir / "manual_forward.snapshot.pt")

    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(_json_ready(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_ready(report), indent=2, ensure_ascii=False))
    print(f"Saved report to {report_path}")

    del pooling_llm
    _free_cuda()


if __name__ == "__main__":
    main()

import argparse
import gc
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from internnav.model.utils.dualvln_single_vllm import to_vllm_chat_messages  # noqa: E402
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
            "Capture the real generation EngineCoreRequest prompt_token_ids and "
            "mm_features for one DualVLN step_s2 sample."
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
        default="logs/habitat/generation_request_mm_capture_sample_0000",
    )
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


def _summarize_placeholders(mm_placeholders) -> list[dict]:
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


def _summarize_processed_prompt(processed_prompt) -> dict:
    mm_placeholders = processed_prompt.get("mm_placeholders") or {}
    mm_kwargs = processed_prompt.get("mm_kwargs") or {}
    image_placeholders = mm_placeholders.get("image") or []
    image_kwargs = mm_kwargs.get("image") or []
    return {
        "type": processed_prompt.get("type"),
        "prompt_token_ids_len": len(processed_prompt.get("prompt_token_ids") or []),
        "image_placeholder_count": len(image_placeholders),
        "image_kwargs_count": len(image_kwargs),
        "image_kwargs_none_count": sum(item is None for item in image_kwargs),
        "image_placeholder_spans": _summarize_placeholders({"image": image_placeholders}),
        "image_kwargs_types": [
            type(item).__name__ if item is not None else "NoneType"
            for item in image_kwargs
        ],
    }


def _load_log_records(dump_dir: Path, prefix: str, tag: str) -> list[dict]:
    records = []
    for path in sorted(dump_dir.glob(f"{prefix}_*_{tag}.log")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload["_path"] = str(path)
        records.append(payload)
    return records


def _load_tensor_records(dump_dir: Path, prefix: str, tag: str) -> list[dict]:
    records = []
    for path in sorted(dump_dir.glob(f"{prefix}_*_{tag}.pt")):
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception:
            continue
        payload["_path"] = str(path)
        records.append(payload)
    return records


def _pick_generation_record(records: list[dict]) -> dict | None:
    candidates = []
    for record in records:
        supported_tasks = record.get("supported_tasks") or record.get("meta", {}).get("supported_tasks") or []
        params_type = record.get("params_type") or record.get("meta", {}).get("params_type")
        if "generate" in supported_tasks and params_type == "SamplingParams":
            candidates.append(record)
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.get("ts_ns", 0))
    return candidates[-1]


def _compare_token_lists(expected_ids: list[int], actual_ids: list[int], *, window: int = 16) -> dict:
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


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dump_prefix = f"generation_request_capture_{int(time.time() * 1000)}"
    os.environ["VLLM_DEBUG_DUMP_DIR"] = str(output_dir)
    os.environ["VLLM_DEBUG_DUMP_PREFIX"] = dump_prefix
    os.environ["VLLM_DEBUG_DUMP_FULL_TENSORS"] = "1"
    os.environ["VLLM_DEBUG_DUMP_SLICE_ROWS"] = "8"

    sample_path = PROJECT_ROOT / args.sample_pt
    manifest_path = PROJECT_ROOT / args.manifest
    sample = torch.load(sample_path, map_location="cpu")
    replay_context = _build_replay_context(sample, manifest_path)

    from vllm import LLM, SamplingParams
    from vllm.outputs import RequestOutput

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

    processed_prompt = llm._preprocess_chat_one(
        to_vllm_chat_messages(replay_context["messages"])
    )
    processed_prompt_summary = _summarize_processed_prompt(processed_prompt)

    outputs = llm._render_and_run_requests(
        prompts=(processed_prompt,),
        params=[SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)],
        output_type=RequestOutput,
        use_tqdm=False,
    )
    request_output = outputs[0]
    completion = request_output.outputs[0]
    request_output_prompt_token_ids = list(request_output.prompt_token_ids or [])
    request_output_generated_token_ids = list(completion.token_ids)

    log_records = _load_log_records(output_dir, dump_prefix, "input_processor_engine_core_request")
    tensor_records = _load_tensor_records(output_dir, dump_prefix, "input_processor_engine_core_request")
    generation_log = _pick_generation_record(log_records)
    generation_tensor = _pick_generation_record(tensor_records)
    if generation_log is None or generation_tensor is None:
        raise RuntimeError("Failed to capture generation EngineCoreRequest debug dump.")

    captured_prompt_token_ids_tensor = generation_tensor.get("tensors", {}).get("prompt_token_ids")
    if not torch.is_tensor(captured_prompt_token_ids_tensor):
        raise RuntimeError("Captured EngineCoreRequest prompt_token_ids tensor is missing.")
    captured_prompt_token_ids = captured_prompt_token_ids_tensor.detach().cpu().tolist()

    report = {
        "metadata": {
            "sample_pt": str(sample_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "vllm_model_path": str((PROJECT_ROOT / args.vllm_model_path).resolve()),
            "output_dir": str(output_dir.resolve()),
            "dump_prefix": dump_prefix,
            "llm_output": completion.text,
            "request_output_prompt_token_count": len(request_output_prompt_token_ids),
            "request_output_generated_token_count": len(request_output_generated_token_ids),
        },
        "processed_prompt": processed_prompt_summary,
        "generation_engine_core_request": {
            "log_path": generation_log["_path"],
            "tensor_path": generation_tensor["_path"],
            "request_id": generation_log.get("request_id"),
            "supported_tasks": generation_log.get("supported_tasks"),
            "params_type": generation_log.get("params_type"),
            "processed_inputs_type": generation_log.get("processed_inputs_type"),
            "decoder_inputs_type": generation_log.get("decoder_inputs_type"),
            "prompt_token_ids_len": generation_log.get("prompt_token_ids_len"),
            "mm_feature_count": generation_log.get("mm_feature_count"),
            "mm_features": generation_log.get("mm_features"),
            "prompt_token_ids_compare_to_request_output": _compare_token_lists(
                request_output_prompt_token_ids,
                captured_prompt_token_ids,
            ),
            "prompt_token_ids_compare_to_processed_prompt": _compare_token_lists(
                list(processed_prompt.get("prompt_token_ids") or []),
                captured_prompt_token_ids,
            ),
            "captured_prompt_token_ids_head16": captured_prompt_token_ids[:16],
            "captured_prompt_token_ids_tail16": captured_prompt_token_ids[-16:],
        },
    }

    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(_json_ready(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_ready(report), indent=2, ensure_ascii=False))
    print(f"Saved report to {report_path}")

    del llm
    _free_cuda()


if __name__ == "__main__":
    main()

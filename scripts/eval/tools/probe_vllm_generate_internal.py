import argparse
import inspect
import json
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe vLLM generate runner internals: model forward exposure, worker-visible KV caches, and prefix cache reuse."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", default="Please answer with a single word: hello.")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _qualname(obj):
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}.{cls.__name__}"


def _summarize_cache_container(cache_obj):
    summary = {
        "container_type": _qualname(cache_obj),
        "length": None,
        "non_none_entries": None,
        "sample_entries": [],
    }

    if isinstance(cache_obj, dict):
        items = list(cache_obj.items())
        summary["length"] = len(items)
        summary["non_none_entries"] = sum(1 for _, v in items if v is not None)
        sample = items[:3]
    elif isinstance(cache_obj, (list, tuple)):
        summary["length"] = len(cache_obj)
        summary["non_none_entries"] = sum(1 for v in cache_obj if v is not None)
        sample = list(enumerate(cache_obj[:3]))
    else:
        sample = []

    for key, value in sample:
        entry = {
            "key": str(key),
            "type": _qualname(value) if value is not None else "NoneType",
        }
        if value is not None and hasattr(value, "shape"):
            try:
                entry["shape"] = list(value.shape)
                entry["dtype"] = str(value.dtype)
                entry["device"] = str(value.device)
            except Exception:
                pass
        summary["sample_entries"].append(entry)
    return summary


def inspect_worker(worker):
    report = {
        "worker_type": _qualname(worker),
        "worker_attrs_subset": sorted(
            [name for name in dir(worker) if any(k in name for k in ("runner", "cache", "model", "executor"))]
        )[:64],
        "paths": {},
    }

    path_specs = {
        "worker": worker,
        "worker.model_runner": getattr(worker, "model_runner", None),
        "worker.model_executor": getattr(worker, "model_executor", None),
        "worker.gpu_worker": getattr(worker, "gpu_worker", None),
    }

    nested_worker = getattr(worker, "worker", None)
    if nested_worker is not None:
        path_specs["worker.worker"] = nested_worker
        path_specs["worker.worker.model_runner"] = getattr(nested_worker, "model_runner", None)

    for path, obj in path_specs.items():
        if obj is None:
            continue
        info = {
            "type": _qualname(obj),
            "has_kv_caches": hasattr(obj, "kv_caches"),
            "has_model": hasattr(obj, "model"),
            "has_input_batch": hasattr(obj, "input_batch"),
        }
        if hasattr(obj, "kv_caches"):
            try:
                info["kv_caches"] = _summarize_cache_container(obj.kv_caches)
            except Exception as exc:
                info["kv_caches_error"] = f"{type(exc).__name__}: {exc}"
        if hasattr(obj, "model") and obj.model is not None:
            try:
                info["model_type"] = _qualname(obj.model)
            except Exception:
                pass
        report["paths"][path] = info
    return report


def inspect_model(model):
    report = {
        "model_type": _qualname(model),
        "has_forward": hasattr(model, "forward"),
        "forward_signature": None,
        "language_model_model_type": None,
        "language_model_forward_signature": None,
        "direct_forward_trials": [],
    }

    if hasattr(model, "forward"):
        try:
            report["forward_signature"] = str(inspect.signature(model.forward))
        except Exception as exc:
            report["forward_signature"] = f"<error: {type(exc).__name__}: {exc}>"

    lm_model = None
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        lm_model = model.language_model.model
        report["language_model_model_type"] = _qualname(lm_model)
        try:
            report["language_model_forward_signature"] = str(inspect.signature(lm_model.forward))
        except Exception as exc:
            report["language_model_forward_signature"] = f"<error: {type(exc).__name__}: {exc}>"

    try:
        import torch

        device = next(model.parameters()).device
        input_ids = torch.tensor([1, 2, 3, 4], device=device, dtype=torch.long)
        positions_1d = torch.arange(4, device=device, dtype=torch.long)
        positions_3d = torch.stack([positions_1d, positions_1d, positions_1d], dim=0)

        def _record_trial(name, fn):
            trial = {"name": name}
            try:
                with torch.inference_mode():
                    output = fn()
                trial["success"] = True
                trial["output_type"] = type(output).__name__
                if hasattr(output, "shape"):
                    trial["output_shape"] = list(output.shape)
                elif isinstance(output, tuple):
                    trial["tuple_len"] = len(output)
                    if output and hasattr(output[0], "shape"):
                        trial["first_output_shape"] = list(output[0].shape)
            except Exception as exc:
                trial["success"] = False
                trial["error_type"] = type(exc).__name__
                trial["error_message"] = str(exc)
            report["direct_forward_trials"].append(trial)

        if hasattr(model, "forward"):
            _record_trial(
                "model.forward_input_ids_positions_1d",
                lambda: model(input_ids=input_ids, positions=positions_1d),
            )
            _record_trial(
                "model.forward_input_ids_positions_3d",
                lambda: model(input_ids=input_ids, positions=positions_3d),
            )

        if lm_model is not None and hasattr(lm_model, "embed_input_ids"):
            inputs_embeds = lm_model.embed_input_ids(input_ids)
            _record_trial(
                "model.forward_inputs_embeds_positions_1d",
                lambda: model(input_ids=None, positions=positions_1d, inputs_embeds=inputs_embeds),
            )
            _record_trial(
                "model.forward_inputs_embeds_positions_3d",
                lambda: model(input_ids=None, positions=positions_3d, inputs_embeds=inputs_embeds),
            )
    except Exception as exc:
        report["inspection_error"] = f"{type(exc).__name__}: {exc}"

    return report


def main():
    args = parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        disable_log_stats=True,
    )

    before_worker = llm.collective_rpc(inspect_worker)
    model_info = llm.apply_model(inspect_model)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
    )
    out1 = llm.generate([args.prompt], sampling_params=sampling_params, use_tqdm=False)[0]
    mid_worker = llm.collective_rpc(inspect_worker)
    out2 = llm.generate([args.prompt], sampling_params=sampling_params, use_tqdm=False)[0]
    after_worker = llm.collective_rpc(inspect_worker)

    report = {
        "model_path": args.model_path,
        "prompt": args.prompt,
        "model_info": model_info,
        "worker_before": before_worker,
        "first_generate": {
            "prompt_token_count": len(out1.prompt_token_ids),
            "num_cached_tokens": out1.num_cached_tokens,
            "output_text": out1.outputs[0].text,
        },
        "worker_after_first_generate": mid_worker,
        "second_generate": {
            "prompt_token_count": len(out2.prompt_token_ids),
            "num_cached_tokens": out2.num_cached_tokens,
            "output_text": out2.outputs[0].text,
        },
        "worker_after_second_generate": after_worker,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 72)
    print("Probe vLLM generate internals")
    print("=" * 72)
    print(f"Model path: {args.model_path}")
    print(f"Prompt length: {len(out1.prompt_token_ids)}")
    print(f"Model has forward: {model_info[0].get('has_forward') if model_info else 'n/a'}")
    if model_info:
        print(f"Model forward signature: {model_info[0].get('forward_signature')}")
    print(f"First generate num_cached_tokens: {out1.num_cached_tokens}")
    print(f"Second generate num_cached_tokens: {out2.num_cached_tokens}")
    if args.output:
        print(f"Saved JSON summary to {args.output}")
    print("=" * 72)


if __name__ == "__main__":
    main()

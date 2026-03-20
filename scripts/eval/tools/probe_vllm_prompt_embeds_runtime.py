import argparse
import functools
import json
import os
from pathlib import Path

import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

TRAJ_TOKEN_INDEX = 151667


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe vLLM EmbedsPrompt path for exact-length preservation and custom last-4 embeddings."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sample-pt", required=True)
    parser.add_argument("--append-traj-tokens", action="store_true")
    parser.add_argument("--traj-token-count", type=int, default=4)
    parser.add_argument("--custom-last-n", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _qualname(obj):
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}.{cls.__name__}"


def _extract_token_embeddings(model, prompt_token_ids):
    device = next(model.parameters()).device
    input_ids = torch.tensor(prompt_token_ids, device=device, dtype=torch.long)

    if not hasattr(model, "language_model") or not hasattr(model.language_model, "model"):
        raise RuntimeError(f"Unexpected model structure: {type(model).__name__}")

    lm_model = model.language_model.model
    if not hasattr(lm_model, "embed_input_ids"):
        raise RuntimeError(f"language_model.model has no embed_input_ids: {type(lm_model).__name__}")

    with torch.inference_mode():
        embeds = lm_model.embed_input_ids(input_ids)

    return {
        "model_type": _qualname(model),
        "language_model_model_type": _qualname(lm_model),
        "hidden_size": int(embeds.shape[-1]),
        "embed_dtype": str(embeds.dtype),
        "embeds": embeds.cpu(),
    }


def _make_custom_last_n(embeds, n):
    modified = embeds.clone()
    if n <= 0:
        return modified
    hidden_size = modified.shape[-1]
    base = torch.linspace(-1.0, 1.0, steps=hidden_size, dtype=modified.dtype)
    custom = torch.stack([base + (i * 0.01) for i in range(n)], dim=0)
    modified[-n:] = custom
    return modified


def _tensor_diff(a, b):
    diff = (a.float() - b.float()).abs()
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def main():
    args = parse_args()

    from vllm import LLM
    from vllm.inputs.data import EmbedsPrompt

    sample = torch.load(args.sample_pt, map_location="cpu")
    prompt_token_ids = sample["baseline_output_ids"][0].tolist()
    if args.append_traj_tokens:
        prompt_token_ids = prompt_token_ids + [TRAJ_TOKEN_INDEX] * args.traj_token_count

    llm = LLM(
        model=args.model_path,
        runner="pooling",
        convert="embed",
        enable_prompt_embeds=True,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        disable_log_stats=True,
    )

    report = {
        "model_path": args.model_path,
        "sample_pt": args.sample_pt,
        "requested_prompt_length": len(prompt_token_ids),
        "append_traj_tokens": bool(args.append_traj_tokens),
        "traj_token_count": int(args.traj_token_count if args.append_traj_tokens else 0),
        "custom_last_n": int(args.custom_last_n),
        "success": False,
        "embedding_source": None,
        "base_prompt_embeds_shape": None,
        "custom_prompt_embeds_shape": None,
        "base_output": None,
        "custom_output": None,
        "output_diff": None,
        "last_n_output_diff": None,
        "error_type": None,
        "error_message": None,
        "blocker_analysis": None,
    }

    try:
        extracted = llm.apply_model(
            functools.partial(_extract_token_embeddings, prompt_token_ids=prompt_token_ids)
        )
        base_prompt_embeds = extracted[0]["embeds"]
        custom_prompt_embeds = _make_custom_last_n(base_prompt_embeds, args.custom_last_n)

        base_output = llm.encode(
            [EmbedsPrompt(prompt_embeds=base_prompt_embeds, prompt_token_ids=prompt_token_ids)],
            pooling_task="token_embed",
            use_tqdm=False,
        )[0]
        custom_output = llm.encode(
            [EmbedsPrompt(prompt_embeds=custom_prompt_embeds, prompt_token_ids=prompt_token_ids)],
            pooling_task="token_embed",
            use_tqdm=False,
        )[0]

        base_data = base_output.outputs.data.cpu()
        custom_data = custom_output.outputs.data.cpu()

        report["success"] = True
        report["embedding_source"] = {
            "model_type": extracted[0]["model_type"],
            "language_model_model_type": extracted[0]["language_model_model_type"],
            "hidden_size": extracted[0]["hidden_size"],
            "embed_dtype": extracted[0]["embed_dtype"],
        }
        report["base_prompt_embeds_shape"] = list(base_prompt_embeds.shape)
        report["custom_prompt_embeds_shape"] = list(custom_prompt_embeds.shape)
        report["base_output"] = {
            "num_cached_tokens": int(base_output.num_cached_tokens),
            "token_embed_shape": list(base_data.shape),
            "shape_matches_requested_length": bool(base_data.shape[0] == len(prompt_token_ids)),
        }
        report["custom_output"] = {
            "num_cached_tokens": int(custom_output.num_cached_tokens),
            "token_embed_shape": list(custom_data.shape),
            "shape_matches_requested_length": bool(custom_data.shape[0] == len(prompt_token_ids)),
        }
        report["output_diff"] = _tensor_diff(base_data, custom_data)
        report["last_n_output_diff"] = _tensor_diff(
            base_data[-args.custom_last_n :], custom_data[-args.custom_last_n :]
        )
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error_message"] = str(exc)
        msg = str(exc)
        if "M-RoPE requires prompt_token_ids to be available" in msg:
            report["blocker_analysis"] = {
                "blocker": "mrope_requires_prompt_token_ids",
                "summary": "Public EmbedsPrompt path drops prompt_token_ids before worker-side M-RoPE initialization.",
                "implication": "Qwen2.5-VL cannot currently use prompt_embeds-only requests for strict generate_latents equivalence.",
                "minimum_patch_direction": [
                    "Allow embeds requests to optionally carry prompt_token_ids.",
                    "Preserve prompt_token_ids through renderer/input_processor instead of forcing them to None.",
                    "Use prompt_token_ids for M-RoPE position construction while still using prompt_embeds as actual backbone inputs.",
                ],
            }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 72)
    print("Probe vLLM prompt_embeds runtime")
    print("=" * 72)
    print(f"Model path: {args.model_path}")
    print(f"Requested prompt length: {len(prompt_token_ids)}")
    if report["success"]:
        print(f"Base output shape: {report['base_output']['token_embed_shape']}")
        print(f"Custom output shape: {report['custom_output']['token_embed_shape']}")
        print(f"Base shape matches requested: {report['base_output']['shape_matches_requested_length']}")
        print(f"Custom shape matches requested: {report['custom_output']['shape_matches_requested_length']}")
        print(f"Last-{args.custom_last_n} max abs diff: {report['last_n_output_diff']['max_abs_diff']:.6f}")
        print(f"Last-{args.custom_last_n} mean abs diff: {report['last_n_output_diff']['mean_abs_diff']:.6f}")
    else:
        print("Runtime probe failed")
        print(f"{report['error_type']}: {report['error_message']}")
        if report["blocker_analysis"] is not None:
            print(f"Blocker: {report['blocker_analysis']['blocker']}")
    if args.output:
        print(f"Saved JSON summary to {args.output}")
    print("=" * 72)


if __name__ == "__main__":
    main()

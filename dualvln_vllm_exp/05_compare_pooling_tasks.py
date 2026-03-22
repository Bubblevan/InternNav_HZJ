import argparse
import gc
import inspect
import json
import os
from pathlib import Path

import torch

TOKEN_DUMP_PATH = Path("/root/backup/InternNav/logs/habitat/token_embed_last_tensor.pt")

def cosine(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def find_best_window(t: torch.Tensor, ref: torch.Tensor, n_query: int):
    if t.ndim != 2 or ref.ndim != 3 or t.shape[0] < n_query or t.shape[1] != ref.shape[-1]:
        return None
    best = None
    for start in range(t.shape[0] - n_query + 1):
        cand = t[start:start + n_query].unsqueeze(0)
        score = cosine(cand, ref)
        if best is None or score > best["cosine"]:
            best = {
                "start": int(start),
                "end": int(start + n_query),
                "cosine": float(score),
            }
    return best

def load_bundle(path):
    bundle = torch.load(path, map_location="cpu")
    out = {}
    for k, v in bundle.items():
        out[k] = v
    return out

def make_prompt(bundle):
    prompt = bundle.get("prompt", "")
    prompt_token_ids = bundle.get("prompt_token_ids")
    if prompt_token_ids is None:
        prompt_token_ids = bundle.get("full_prompt_ids")
    if prompt_token_ids is None:
        prompt_token_ids = bundle.get("full_output_ids")
    if isinstance(prompt_token_ids, torch.Tensor):
        prompt_token_ids = prompt_token_ids.tolist()

    prompt_embeds = bundle.get("prompt_embeds")
    if prompt_embeds is None:
        raise RuntimeError("bundle missing prompt_embeds")

    if isinstance(prompt_embeds, torch.Tensor):
        prompt_embeds = prompt_embeds.contiguous()

    return {
        "prompt": prompt,
        "prompt_token_ids": prompt_token_ids,
        "prompt_embeds": prompt_embeds,
    }


def resolve_reference_latents(bundle):
    ref = bundle.get("traj_latents")
    if ref is None:
        ref = bundle.get("ref_traj_latents")
    if ref is None:
        ref = bundle.get("manual_traj_latents")
    if ref is None:
        raise KeyError(
            "Bundle missing trajectory latents. Expected one of: "
            "traj_latents, ref_traj_latents, manual_traj_latents"
        )
    if not isinstance(ref, torch.Tensor):
        ref = torch.tensor(ref)
    return ref.cpu()

def extract_tensor_from_output(obj):
    candidates = []

    for attr in ["outputs", "data", "embedding", "embeddings", "hidden_states"]:
        if hasattr(obj, attr):
            candidates.append(getattr(obj, attr))

    if hasattr(obj, "__dict__"):
        for _, v in obj.__dict__.items():
            candidates.append(v)

    queue = list(candidates)
    seen = set()
    while queue:
        x = queue.pop(0)
        if id(x) in seen:
            continue
        seen.add(id(x))

        if isinstance(x, torch.Tensor):
            return x

        if isinstance(x, (list, tuple)):
            for y in x:
                queue.append(y)
            continue

        if hasattr(x, "__dict__"):
            for _, y in x.__dict__.items():
                queue.append(y)

    return None

def call_encode(llm, prompt, pooling_task=None):
    sig = inspect.signature(llm.encode)
    kwargs = {}
    if "use_tqdm" in sig.parameters:
        kwargs["use_tqdm"] = False
    if pooling_task is not None and "pooling_task" in sig.parameters:
        kwargs["pooling_task"] = pooling_task
    return llm.encode([prompt], **kwargs)


def call_reward(llm, prompt):
    sig = inspect.signature(llm.reward)
    kwargs = {}
    if "use_tqdm" in sig.parameters:
        kwargs["use_tqdm"] = False
    return llm.reward([prompt], **kwargs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--enforce-eager", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

    from vllm import LLM
    bundle = load_bundle(args.bundle)
    prompt = make_prompt(bundle)
    ref = resolve_reference_latents(bundle)
    n_query = int(bundle.get("meta", {}).get("n_query", bundle.get("n_query", ref.shape[1])))

    configs = [
        {"name": "token_embed_encode", "api": "encode", "pooling_task": "token_embed"},
        {"name": "embed_encode", "api": "encode", "pooling_task": "embed"},
        {"name": "reward_call", "api": "reward", "pooling_task": None},
    ]

    results = []
    for cfg in configs:
        row = {
            "name": cfg["name"],
            "api": cfg["api"],
            "pooling_task": cfg["pooling_task"],
        }
        try:
            llm = LLM(
                model=args.model_path,
                runner="pooling",
                convert="embed",
                trust_remote_code=args.trust_remote_code,
                dtype=args.dtype,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                enforce_eager=args.enforce_eager,
                enable_prompt_embeds=True,
            )
            row["supported_tasks"] = list(getattr(llm, "supported_tasks", []))
            if cfg["api"] == "encode":
                outs = call_encode(llm, prompt, cfg["pooling_task"])
            elif cfg["api"] == "reward":
                outs = call_reward(llm, prompt)
            else:
                raise ValueError(f"Unsupported api: {cfg['api']}")
            first = outs[0]
            tensor = extract_tensor_from_output(first)
            row["output_type"] = str(type(first))
            row["tensor_found"] = tensor is not None

            if tensor is not None:
                t = tensor.detach().cpu()
                row["tensor_shape"] = list(t.shape)
                if cfg["name"] == "token_embed_encode":
                    TOKEN_DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"tensor": t.contiguous()}, TOKEN_DUMP_PATH)
                    row["tensor_dump_path"] = str(TOKEN_DUMP_PATH)

                tail = None
                if t.ndim == 2 and t.shape[0] >= n_query and t.shape[-1] == ref.shape[-1]:
                    tail = t[-n_query:]
                    if ref.ndim == 3:
                        tail = tail.unsqueeze(0)
                elif t.ndim == 3 and t.shape[-2] >= n_query and t.shape[-1] == ref.shape[-1]:
                    tail = t[0, -n_query:, :]
                    if ref.ndim == 3:
                        tail = tail.unsqueeze(0)
                elif t.ndim == 1 and t.shape[0] == ref.numel():
                    tail = t.reshape_as(ref)

                if tail is not None and tuple(tail.shape) == tuple(ref.shape):
                    row["tail_vs_ref"] = {
                        "cosine": cosine(tail, ref),
                        "mean_abs": (tail.float() - ref.float()).abs().mean().item(),
                        "max_abs": (tail.float() - ref.float()).abs().max().item(),
                    }
                    best_window = find_best_window(t, ref, n_query)
                    if best_window is not None:
                        row["best_window_vs_ref"] = best_window
                        row["tail_is_best_window"] = (
                            best_window["start"] == t.shape[0] - n_query
                        )
                else:
                    row["tail_vs_ref"] = None

            del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            row["error"] = repr(e)

        results.append(row)

    summary = {
        "pooling_runner_supported": any("supported_tasks" in row for row in results),
        "prompt_embeds_encode_supported": any(
            row.get("api") == "encode" and "error" not in row for row in results
        ),
        "token_embed_returns_tokenwise": any(
            row.get("name") == "token_embed_encode"
            and row.get("tensor_shape")
            and len(row["tensor_shape"]) >= 2
            for row in results
        ),
        "embed_returns_tokenwise": any(
            row.get("name") == "embed_encode"
            and row.get("tensor_shape")
            and len(row["tensor_shape"]) >= 2
            for row in results
        ),
        "reward_supported": any(
            row.get("name") == "reward_call" and "error" not in row for row in results
        ),
        "token_wise_routes": [
            row["name"]
            for row in results
            if row.get("tensor_shape") and len(row["tensor_shape"]) >= 2
        ],
        "vector_routes": [
            row["name"]
            for row in results
            if row.get("tensor_shape") and len(row["tensor_shape"]) == 1
        ],
        "unsupported_routes": [
            row["name"]
            for row in results
            if "error" in row
        ],
    }
    out = {"summary": summary, "results": results}
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

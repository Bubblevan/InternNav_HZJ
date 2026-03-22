import argparse
import json
from pathlib import Path

import torch

def cosine(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--compare-json", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    bundle = torch.load(args.bundle, map_location="cpu")
    compare = json.loads(Path(args.compare_json).read_text())

    ref = bundle.get("traj_latents", bundle.get("ref_traj_latents"))
    if not isinstance(ref, torch.Tensor):
        ref = torch.tensor(ref)
    if ref.ndim == 3 and ref.shape[0] == 1:
        ref = ref[0]
    ref = ref.cpu().float()

    token_row = None
    for r in compare.get("results", []):
        name = r.get("name") or r.get("pooling_task")
        if name in ("token_embed_encode", "token_embed"):
            token_row = r
            break

    out = {
        "have_token_embed_route": token_row is not None,
        "token_embed_summary": token_row,
    }

    # Optional tensor dump path support if Codex later extends 05 to save token tensor.
    token_tensor_path = token_row.get("tensor_dump_path") if token_row else None
    if token_tensor_path and Path(token_tensor_path).exists():
        token = torch.load(token_tensor_path, map_location="cpu")
        if isinstance(token, dict) and "tensor" in token:
            token = token["tensor"]
        if token.ndim == 3 and token.shape[0] == 1:
            token = token[0]
        token = token.float().cpu()

        tail = token[-ref.shape[0]:]
        out["tail_shape"] = list(tail.shape)
        out["ref_shape"] = list(ref.shape)

        # Raw compare
        out["raw"] = {
            "cosine": cosine(tail, ref),
            "mean_abs": (tail - ref).abs().mean().item(),
            "max_abs": (tail - ref).abs().max().item(),
        }

        # L2 normalize per token
        tail_n = torch.nn.functional.normalize(tail, dim=-1)
        ref_n = torch.nn.functional.normalize(ref, dim=-1)
        out["l2_normalized"] = {
            "cosine": cosine(tail_n, ref_n),
            "mean_abs": (tail_n - ref_n).abs().mean().item(),
            "max_abs": (tail_n - ref_n).abs().max().item(),
        }

        # Center per token
        tail_c = tail - tail.mean(dim=-1, keepdim=True)
        ref_c = ref - ref.mean(dim=-1, keepdim=True)
        out["mean_centered"] = {
            "cosine": cosine(tail_c, ref_c),
            "mean_abs": (tail_c - ref_c).abs().mean().item(),
            "max_abs": (tail_c - ref_c).abs().max().item(),
        }

    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

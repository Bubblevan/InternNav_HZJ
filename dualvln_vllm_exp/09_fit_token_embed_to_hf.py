import argparse
import json
from pathlib import Path

import torch

def cosine(a, b):
    return torch.nn.functional.cosine_similarity(
        a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1), dim=1
    )

def solve_linear_regression(X, Y, ridge=1e-4):
    # X: [M, D], Y: [M, D]
    D = X.shape[1]
    I = torch.eye(D, device=X.device, dtype=X.dtype)
    W = torch.linalg.solve(X.T @ X + ridge * I, X.T @ Y)
    return W

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-pt", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = torch.load(args.dataset_pt, map_location="cpu")
    X = data["X_token_embed"].float()   # [N, Q, H]
    Y = data["Y_hf_latents"].float()    # [N, Q, H]

    N, Q, H = X.shape
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(N, generator=g)
    train_n = max(1, int(N * args.train_ratio))
    tr, te = perm[:train_n], perm[train_n:]

    Xtr, Ytr = X[tr], Y[tr]
    Xte, Yte = X[te], Y[te]

    results = {}

    # 1) raw
    raw_cos = cosine(Xte, Yte)
    results["raw"] = {
        "mean_cosine": raw_cos.mean().item(),
        "min_cosine": raw_cos.min().item(),
        "max_cosine": raw_cos.max().item(),
    }

    # 2) global scalar alpha
    alpha = (Xtr * Ytr).sum() / (Xtr * Xtr).sum().clamp_min(1e-8)
    Xte_alpha = Xte * alpha
    cos_alpha = cosine(Xte_alpha, Yte)
    results["scalar_alpha"] = {
        "alpha": alpha.item(),
        "mean_cosine": cos_alpha.mean().item(),
        "min_cosine": cos_alpha.min().item(),
        "max_cosine": cos_alpha.max().item(),
    }

    # 3) per-dim diagonal scale
    num = (Xtr * Ytr).sum(dim=(0,1))
    den = (Xtr * Xtr).sum(dim=(0,1)).clamp_min(1e-8)
    d = num / den
    Xte_diag = Xte * d
    cos_diag = cosine(Xte_diag, Yte)
    results["diag_scale"] = {
        "mean_cosine": cos_diag.mean().item(),
        "min_cosine": cos_diag.min().item(),
        "max_cosine": cos_diag.max().item(),
    }

    # 4) full linear map shared across all query positions
    Xtr2 = Xtr.reshape(-1, H)
    Ytr2 = Ytr.reshape(-1, H)
    Xte2 = Xte.reshape(-1, H)

    W = solve_linear_regression(Xtr2, Ytr2)
    Yhat = (Xte2 @ W).reshape_as(Yte)
    cos_lin = cosine(Yhat, Yte)
    results["full_linear"] = {
        "mean_cosine": cos_lin.mean().item(),
        "min_cosine": cos_lin.min().item(),
        "max_cosine": cos_lin.max().item(),
    }

    # 5) query-specific linear maps
    q_stats = []
    Yhat_q = torch.empty_like(Yte)
    for q in range(Q):
        Wq = solve_linear_regression(Xtr[:, q, :], Ytr[:, q, :])
        Yhat_q[:, q, :] = Xte[:, q, :] @ Wq
        q_cos = cosine(Yhat_q[:, q:q+1, :], Yte[:, q:q+1, :])
        q_stats.append({
            "query_idx": q,
            "mean_cosine": q_cos.mean().item(),
            "min_cosine": q_cos.min().item(),
            "max_cosine": q_cos.max().item(),
        })
    cos_q = cosine(Yhat_q, Yte)
    results["query_specific_linear"] = {
        "mean_cosine": cos_q.mean().item(),
        "min_cosine": cos_q.min().item(),
        "max_cosine": cos_q.max().item(),
        "per_query": q_stats,
    }

    Path(args.out_json).write_text(json.dumps({
        "N": N, "Q": Q, "H": H,
        "train_n": train_n,
        "test_n": len(te),
        "results": results,
    }, ensure_ascii=False, indent=2))
    print(json.dumps({
        "train_n": train_n,
        "test_n": len(te),
        "results": results,
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

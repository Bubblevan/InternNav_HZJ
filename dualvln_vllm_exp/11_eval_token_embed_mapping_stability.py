import argparse
import json
from pathlib import Path

import torch

def cosine_batch(a, b):
    return torch.nn.functional.cosine_similarity(
        a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1), dim=1
    )

def solve_linear_regression(X, Y, ridge=1e-4):
    D = X.shape[1]
    I = torch.eye(D, device=X.device, dtype=X.dtype)
    return torch.linalg.solve(X.T @ X + ridge * I, X.T @ Y)

def eval_mapping(Xtr, Ytr, Xte, Yte):
    Xtr2 = Xtr.reshape(-1, Xtr.shape[-1])
    Ytr2 = Ytr.reshape(-1, Ytr.shape[-1])
    Xte2 = Xte.reshape(-1, Xte.shape[-1])

    raw = cosine_batch(Xte, Yte)

    alpha = (Xtr * Ytr).sum() / (Xtr * Xtr).sum().clamp_min(1e-8)
    raw_alpha = cosine_batch(Xte * alpha, Yte)

    num = (Xtr * Ytr).sum(dim=(0, 1))
    den = (Xtr * Xtr).sum(dim=(0, 1)).clamp_min(1e-8)
    d = num / den
    raw_diag = cosine_batch(Xte * d, Yte)

    W = solve_linear_regression(Xtr2, Ytr2)
    yhat_lin = (Xte2 @ W).reshape_as(Yte)
    raw_lin = cosine_batch(yhat_lin, Yte)

    return {
        "raw": raw.tolist(),
        "scalar_alpha": raw_alpha.tolist(),
        "diag_scale": raw_diag.tolist(),
        "full_linear": raw_lin.tolist(),
    }

def stats(vals):
    t = torch.tensor(vals, dtype=torch.float32)
    return {
        "mean": t.mean().item(),
        "min": t.min().item(),
        "max": t.max().item(),
        "std": t.std(unbiased=False).item() if t.numel() > 1 else 0.0,
        "count": t.numel(),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-pt", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--num-seeds", type=int, default=20)
    args = ap.parse_args()

    data = torch.load(args.dataset_pt, map_location="cpu")
    X = data["X_token_embed"].float()
    Y = data["Y_hf_latents"].float()
    N = X.shape[0]

    split_runs = []
    for seed in range(args.num_seeds):
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(N, generator=g)
        train_n = max(1, int(N * args.train_ratio))
        if train_n >= N:
            train_n = N - 1
        tr, te = perm[:train_n], perm[train_n:]
        metrics = eval_mapping(X[tr], Y[tr], X[te], Y[te])
        split_runs.append({
            "seed": seed,
            "train_idx": tr.tolist(),
            "test_idx": te.tolist(),
            "metrics": {k: stats(v) for k, v in metrics.items()},
        })

    loocv_runs = []
    if N >= 2:
        for i in range(N):
            tr = [j for j in range(N) if j != i]
            te = [i]
            metrics = eval_mapping(X[tr], Y[tr], X[te], Y[te])
            loocv_runs.append({
                "heldout_idx": i,
                "metrics": {k: stats(v) for k, v in metrics.items()},
            })

    def aggregate(runs):
        out = {}
        for key in ["raw", "scalar_alpha", "diag_scale", "full_linear"]:
            means = [r["metrics"][key]["mean"] for r in runs]
            mins = [r["metrics"][key]["min"] for r in runs]
            out[key] = {
                "mean_of_mean": stats(means)["mean"] if means else None,
                "min_of_mean": stats(means)["min"] if means else None,
                "max_of_mean": stats(means)["max"] if means else None,
                "std_of_mean": stats(means)["std"] if means else None,
                "worst_case_min": min(mins) if mins else None,
            }
        return out

    out = {
        "N": N,
        "split_runs": split_runs,
        "split_summary": aggregate(split_runs),
        "loocv_runs": loocv_runs,
        "loocv_summary": aggregate(loocv_runs),
    }
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps({
        "N": N,
        "split_summary": out["split_summary"],
        "loocv_summary": out["loocv_summary"],
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

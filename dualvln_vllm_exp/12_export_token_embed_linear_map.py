import argparse
import json
from pathlib import Path

import torch

def solve_linear_regression(X, Y, ridge=1e-4):
    D = X.shape[1]
    I = torch.eye(D, device=X.device, dtype=X.dtype)
    return torch.linalg.solve(X.T @ X + ridge * I, X.T @ Y)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-pt", required=True)
    ap.add_argument("--out-pt", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    data = torch.load(args.dataset_pt, map_location="cpu")
    X = data["X_token_embed"].float()   # [N,Q,H]
    Y = data["Y_hf_latents"].float()    # [N,Q,H]

    N, Q, H = X.shape
    X2 = X.reshape(-1, H)
    Y2 = Y.reshape(-1, H)
    W = solve_linear_regression(X2, Y2)

    torch.save({
        "W": W.cpu(),
        "hidden_size": H,
        "n_query": Q,
        "num_samples": N,
    }, args.out_pt)

    out = {
        "weight_pt": args.out_pt,
        "hidden_size": H,
        "n_query": Q,
        "num_samples": N,
        "weight_shape": list(W.shape),
    }
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

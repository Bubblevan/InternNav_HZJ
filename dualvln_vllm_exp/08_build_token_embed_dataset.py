import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--samples-dir", required=True)
    ap.add_argument("--out-pt", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    repo = Path("/root/backup/InternNav")
    export_script = repo / "dualvln_vllm_exp/01_export_hf_prompt_embeds_bundle.py"
    compare_script = repo / "dualvln_vllm_exp/05_compare_pooling_tasks.py"

    tmp_dir = repo / "logs/habitat/token_embed_dataset/tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    sample_files = sorted(Path(args.samples_dir).glob("sample_*.pt"))[:args.limit]

    rows = []
    X = []
    Y = []

    for sample_pt in sample_files:
        stem = sample_pt.stem
        bundle_pt = tmp_dir / f"{stem}.bundle.pt"
        compare_json = tmp_dir / f"{stem}.compare.json"

        cmd1 = [
            sys.executable, str(export_script),
            "--model-path", args.model_path,
            "--sample-pt", str(sample_pt),
            "--out", str(bundle_pt),
            "--device", "cuda:0",
            "--dtype", args.dtype,
        ]
        p1 = subprocess.run(cmd1, cwd=str(repo), capture_output=True, text=True)
        row = {
            "sample": stem,
            "export_returncode": p1.returncode,
            "export_stderr_tail": p1.stderr[-2000:],
        }
        if p1.returncode != 0 or not bundle_pt.exists():
            rows.append(row)
            continue

        cmd2 = [
            sys.executable, str(compare_script),
            "--model-path", args.model_path,
            "--bundle", str(bundle_pt),
            "--out-json", str(compare_json),
            "--dtype", args.dtype,
            "--max-model-len", "4096",
            "--trust-remote-code",
            "--enforce-eager",
        ]
        p2 = subprocess.run(cmd2, cwd=str(repo), capture_output=True, text=True)
        row["compare_returncode"] = p2.returncode
        row["compare_stderr_tail"] = p2.stderr[-2000:]

        if p2.returncode != 0 or not compare_json.exists():
            rows.append(row)
            continue

        compare = json.loads(compare_json.read_text())
        token_row = None
        for r in compare.get("results", []):
            name = r.get("name") or r.get("pooling_task")
            if name in ("token_embed_encode", "token_embed"):
                token_row = r
                break

        if not token_row or "tensor_dump_path" not in token_row:
            row["missing_token_embed_tensor"] = True
            rows.append(row)
            continue

        token_tensor = torch.load(token_row["tensor_dump_path"], map_location="cpu")
        if isinstance(token_tensor, dict) and "tensor" in token_tensor:
            token_tensor = token_tensor["tensor"]
        if token_tensor.ndim == 3 and token_tensor.shape[0] == 1:
            token_tensor = token_tensor[0]

        bundle = torch.load(bundle_pt, map_location="cpu")
        ref = bundle.get("traj_latents", bundle.get("ref_traj_latents"))
        if not isinstance(ref, torch.Tensor):
            ref = torch.tensor(ref)
        if ref.ndim == 3 and ref.shape[0] == 1:
            ref = ref[0]

        n_query = ref.shape[0]
        tail = token_tensor[-n_query:].float().cpu()
        ref = ref.float().cpu()

        X.append(tail)   # token_embed tail [n_query, hidden]
        Y.append(ref)    # HF traj_latents [n_query, hidden]

        row["tail_shape"] = list(tail.shape)
        row["ref_shape"] = list(ref.shape)
        row["cosine_raw"] = torch.nn.functional.cosine_similarity(
            tail.reshape(-1), ref.reshape(-1), dim=0
        ).item()
        rows.append(row)

    out = {
        "X_token_embed": torch.stack(X) if X else None,   # [N, n_query, hidden]
        "Y_hf_latents": torch.stack(Y) if Y else None,    # [N, n_query, hidden]
        "rows": rows,
    }
    torch.save(out, args.out_pt)

    summary = {
        "dataset_pt": args.out_pt,
        "num_requested": len(sample_files),
        "num_valid": len(X),
        "rows": rows,
    }
    Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({
        "dataset_pt": args.out_pt,
        "num_requested": len(sample_files),
        "num_valid": len(X),
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

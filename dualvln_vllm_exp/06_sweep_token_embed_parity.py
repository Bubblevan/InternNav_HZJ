import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--samples-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    repo = Path("/root/backup/InternNav")
    export_script = repo / "dualvln_vllm_exp/01_export_hf_prompt_embeds_bundle.py"
    compare_script = repo / "dualvln_vllm_exp/05_compare_pooling_tasks.py"

    out_dir = Path(args.out_dir)
    bundle_dir = out_dir / "bundles"
    json_dir = out_dir / "json"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    sample_files = sorted(Path(args.samples_dir).glob("sample_*.pt"))[:args.limit]
    rows = []

    for sample_pt in sample_files:
        stem = sample_pt.stem
        bundle_pt = bundle_dir / f"{stem}.bundle.pt"
        compare_json = json_dir / f"{stem}.json"

        export_cmd = [
            sys.executable, str(export_script),
            "--model-path", args.model_path,
            "--sample-pt", str(sample_pt),
            "--out", str(bundle_pt),
            "--device", "cuda:0",
            "--dtype", args.dtype,
        ]
        print("EXPORT", " ".join(export_cmd), flush=True)
        p1 = subprocess.run(export_cmd, cwd=str(repo), text=True, capture_output=True)

        row = {
            "sample": stem,
            "sample_pt": str(sample_pt),
            "export_returncode": p1.returncode,
            "export_stdout_tail": p1.stdout[-1000:],
            "export_stderr_tail": p1.stderr[-2000:],
            "bundle_exists": bundle_pt.exists(),
        }

        if p1.returncode != 0 or not bundle_pt.exists():
            rows.append(row)
            continue

        compare_cmd = [
            sys.executable, str(compare_script),
            "--model-path", args.model_path,
            "--bundle", str(bundle_pt),
            "--out-json", str(compare_json),
            "--dtype", args.dtype,
            "--max-model-len", str(args.max_model_len),
        ]
        if args.trust_remote_code:
            compare_cmd.append("--trust-remote-code")
        if args.enforce_eager:
            compare_cmd.append("--enforce-eager")

        print("COMPARE", " ".join(compare_cmd), flush=True)
        p2 = subprocess.run(compare_cmd, cwd=str(repo), text=True, capture_output=True)

        row.update({
            "compare_returncode": p2.returncode,
            "compare_stdout_tail": p2.stdout[-1000:],
            "compare_stderr_tail": p2.stderr[-2000:],
            "compare_json_exists": compare_json.exists(),
        })

        if compare_json.exists():
            try:
                data = json.loads(compare_json.read_text())
                by_name = {}
                for r in data.get("results", []):
                    by_name[r.get("name") or r.get("pooling_task")] = r

                token_row = by_name.get("token_embed_encode", by_name.get("token_embed"))
                embed_row = by_name.get("embed_encode", by_name.get("embed"))

                row["token_embed_tensor_shape"] = token_row.get("tensor_shape") if token_row else None
                row["token_embed_tail_vs_ref"] = token_row.get("tail_vs_ref") if token_row else None
                row["embed_tensor_shape"] = embed_row.get("tensor_shape") if embed_row else None
                row["embed_tail_vs_ref"] = embed_row.get("tail_vs_ref") if embed_row else None
            except Exception as e:
                row["compare_json_parse_error"] = repr(e)

        rows.append(row)

    valid = []
    for r in rows:
        tvr = r.get("token_embed_tail_vs_ref")
        if isinstance(tvr, dict) and isinstance(tvr.get("cosine"), (int, float)):
            valid.append((r["sample"], tvr["cosine"], tvr.get("mean_abs"), tvr.get("max_abs")))

    cosines = [x[1] for x in valid]
    summary = {
        "num_samples": len(sample_files),
        "num_valid_token_embed": len(valid),
        "mean_cosine": sum(cosines) / len(cosines) if cosines else None,
        "min_cosine": min(cosines) if cosines else None,
        "max_cosine": max(cosines) if cosines else None,
        "valid_samples": [
            {"sample": s, "cosine": c, "mean_abs": ma, "max_abs": xa}
            for s, c, ma, xa in valid
        ],
        "rows": rows,
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({
        "summary_json": str(summary_path),
        "num_valid_token_embed": summary["num_valid_token_embed"],
        "mean_cosine": summary["mean_cosine"],
        "min_cosine": summary["min_cosine"],
        "max_cosine": summary["max_cosine"],
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

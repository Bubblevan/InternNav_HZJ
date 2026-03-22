import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--model-impl", default="transformers")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--try-enable-prompt-embeds", action="store_true")
    ap.add_argument("--start-layer", type=int, default=0)
    ap.add_argument("--end-layer", type=int, default=27)
    args = ap.parse_args()

    repo = Path("/root/backup/InternNav")
    script03 = repo / "dualvln_vllm_exp/03_vllm_extract_hidden_states_from_prompt_embeds.py"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for layer_id in range(args.start_layer, args.end_layer + 1):
        out_json = out_dir / f"layer_{layer_id:02d}.json"
        cmd = [
            sys.executable, str(script03),
            "--model-path", args.model_path,
            "--bundle", args.bundle,
            "--out-json", str(out_json),
            "--dtype", args.dtype,
            "--max-model-len", str(args.max_model_len),
            "--model-impl", args.model_impl,
            "--layer-id", str(layer_id),
        ]
        if args.trust_remote_code:
            cmd.append("--trust-remote-code")
        if args.enforce_eager:
            cmd.append("--enforce-eager")
        if args.try_enable_prompt_embeds:
            cmd.append("--try-enable-prompt-embeds")

        print("RUN", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=str(repo), text=True, capture_output=True)

        row = {
            "layer_id": layer_id,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-4000:],
            "json_exists": out_json.exists(),
        }

        if out_json.exists():
            try:
                data = json.loads(out_json.read_text())
                tail = data.get("tail_vs_ref_traj_latents", {})
                row.update({
                    "success": data.get("success"),
                    "hidden_states_shape": data.get("hidden_states_shape"),
                    "token_ids_shape": data.get("token_ids_shape"),
                    "generated_text": data.get("generated_text"),
                    "finish_reason": data.get("finish_reason"),
                    "cosine": tail.get("cosine"),
                    "mean_abs": tail.get("mean_abs"),
                    "max_abs": tail.get("max_abs"),
                })
            except Exception as e:
                row["json_parse_error"] = repr(e)

        rows.append(row)

    valid = [r for r in rows if isinstance(r.get("cosine"), (int, float))]
    valid_sorted = sorted(valid, key=lambda r: r["cosine"], reverse=True)

    summary = {
        "rows": rows,
        "best_by_cosine": valid_sorted[:10],
        "best_layer": valid_sorted[0]["layer_id"] if valid_sorted else None,
        "best_cosine": valid_sorted[0]["cosine"] if valid_sorted else None,
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({
        "summary_json": str(summary_path),
        "best_layer": summary["best_layer"],
        "best_cosine": summary["best_cosine"],
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

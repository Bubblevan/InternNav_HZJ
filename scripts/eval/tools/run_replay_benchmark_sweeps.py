import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_NUM_HISTORY = [8, 4, 2, 0]
DEFAULT_PROMPT_VARIANTS = ["full", "short", "minimal"]
DEFAULT_MAX_NEW_TOKENS = [128, 64, 32, 16]


def parse_args():
    parser = argparse.ArgumentParser(description="Run replay benchmark ablation sweeps and build comparison tables.")
    parser.add_argument("--manifest", required=True, help="Replay manifest jsonl path")
    parser.add_argument("--model-path", required=True, help="DualVLN checkpoint path")
    parser.add_argument("--output-dir", required=True, help="Directory for sweep outputs")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--base-path", default=None, help="Optional base path for manifest path rewrite")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on replay steps")
    parser.add_argument(
        "--groups",
        nargs="*",
        choices=["num_history", "prompt", "max_new_tokens"],
        default=["num_history", "prompt", "max_new_tokens"],
        help="Which sweep groups to execute",
    )
    parser.add_argument("--verbose-every", type=int, default=20, help="Forwarded to replay benchmark")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def flatten_summary(summary, extra):
    return {
        **extra,
        "ignore_manifest_history": summary["metadata"].get("ignore_manifest_history", False),
        "cold_start_load_seconds": summary["startup"]["cold_start_load_seconds"],
        "gpu_peak_memory_mb": summary["startup"]["gpu_peak_memory_mb"],
        "total_step_p50": summary["latency"]["total_step"]["p50"],
        "total_step_p95": summary["latency"]["total_step"]["p95"],
        "total_step_max": summary["latency"]["total_step"]["max"],
        "s2_generate_p50": summary["latency"]["s2_generate"]["p50"],
        "s2_generate_p95": summary["latency"]["s2_generate"]["p95"],
        "s2_generate_max": summary["latency"]["s2_generate"]["max"],
        "s2_latent_p50": summary["latency"]["s2_latent"]["p50"],
        "s2_latent_p95": summary["latency"]["s2_latent"]["p95"],
        "s1_generate_p50": summary["latency"]["s1_generate"]["p50"],
        "s1_generate_p95": summary["latency"]["s1_generate"]["p95"],
        "tokens_per_second_mean": summary["generation"]["tokens_per_second_mean"],
        "generated_tokens_mean": summary["generation"]["generated_tokens_mean"],
        "action_match_rate_all": summary["consistency"]["action_match_rate_all"],
        "action_match_rate_discrete": summary["consistency"]["action_match_rate_discrete"],
        "action_match_rate_pixel_goal": summary["consistency"]["action_match_rate_pixel_goal"],
        "output_kind_match_rate": summary["consistency"]["output_kind_match_rate"],
        "text_exact_match_rate": summary["consistency"]["text_exact_match_rate"],
    }


def run_single_benchmark(
    args,
    output_dir,
    run_name,
    num_history,
    prompt_variant,
    max_new_tokens,
    ignore_manifest_history=False,
):
    ensure_dir(output_dir)
    summary_path = os.path.join(output_dir, f"{run_name}.json")
    details_path = os.path.join(output_dir, f"{run_name}.jsonl")

    cmd = [
        sys.executable,
        "scripts/eval/tools/benchmark_dualvln_replay.py",
        "--manifest",
        args.manifest,
        "--model-path",
        args.model_path,
        "--output",
        summary_path,
        "--details-output",
        details_path,
        "--device",
        args.device,
        "--num-history",
        str(num_history),
        "--prompt-variant",
        prompt_variant,
        "--max-new-tokens",
        str(max_new_tokens),
        "--verbose-every",
        str(args.verbose_every),
    ]
    if ignore_manifest_history:
        cmd.append("--ignore-manifest-history")
    if args.base_path:
        cmd.extend(["--base-path", args.base_path])
    if args.max_steps is not None:
        cmd.extend(["--max-steps", str(args.max_steps)])

    print("Running:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    subprocess.run(cmd, check=True, env=env)

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    return flatten_summary(
        summary,
        {
            "run_name": run_name,
            "num_history": num_history,
            "prompt_variant": prompt_variant,
            "max_new_tokens": max_new_tokens,
            "summary_path": summary_path,
            "details_path": details_path,
        },
    )


def write_table(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    out_root = Path(args.output_dir)
    ensure_dir(out_root)

    results = {}

    if "num_history" in args.groups:
        rows = []
        group_dir = out_root / "num_history_runs"
        for value in DEFAULT_NUM_HISTORY:
            rows.append(
                run_single_benchmark(
                    args,
                    str(group_dir),
                    run_name=f"num_history_{value}",
                    num_history=value,
                    prompt_variant="full",
                    max_new_tokens=128,
                    ignore_manifest_history=True,
                )
            )
        write_table(str(out_root / "comparison_num_history.csv"), rows)
        results["num_history"] = rows

    if "prompt" in args.groups:
        rows = []
        group_dir = out_root / "prompt_runs"
        for value in DEFAULT_PROMPT_VARIANTS:
            rows.append(
                run_single_benchmark(
                    args,
                    str(group_dir),
                    run_name=f"prompt_{value}",
                    num_history=8,
                    prompt_variant=value,
                    max_new_tokens=128,
                )
            )
        write_table(str(out_root / "comparison_prompt.csv"), rows)
        results["prompt"] = rows

    if "max_new_tokens" in args.groups:
        rows = []
        group_dir = out_root / "max_new_tokens_runs"
        for value in DEFAULT_MAX_NEW_TOKENS:
            rows.append(
                run_single_benchmark(
                    args,
                    str(group_dir),
                    run_name=f"max_new_tokens_{value}",
                    num_history=8,
                    prompt_variant="full",
                    max_new_tokens=value,
                )
            )
        write_table(str(out_root / "comparison_max_new_tokens.csv"), rows)
        results["max_new_tokens"] = rows

    with open(out_root / "comparison_all.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

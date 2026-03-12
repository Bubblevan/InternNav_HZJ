import argparse
import csv
import json
import os
import subprocess
import sys


REPLAY_SCRIPT = "scripts/eval/tools/benchmark_dualvln_replay.py"
LOOPBACK_SCRIPT = "scripts/eval/tools/benchmark_dualvln_http_loopback.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Run DualVLN backend sweeps for replay and/or HTTP loopback.")
    parser.add_argument("--manifest", required=True, help="Replay manifest path")
    parser.add_argument("--model-path", required=True, help="DualVLN checkpoint path")
    parser.add_argument("--output-dir", required=True, help="Output directory for summaries and comparisons")
    parser.add_argument("--base-path", default=None, help="Base path used to rewrite ./logs/ manifest entries")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument(
        "--mode",
        choices=["replay", "loopback", "both"],
        default="both",
        help="Which benchmark family to run",
    )
    parser.add_argument(
        "--attn-backends",
        nargs="+",
        default=["flash_attention_2", "sdpa"],
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention backend variants",
    )
    parser.add_argument(
        "--processor-fast-options",
        nargs="+",
        default=["auto", "true"],
        choices=["auto", "true", "false"],
        help="AutoProcessor use_fast variants",
    )
    parser.add_argument("--num-history", type=int, default=8, help="Shared num_history")
    parser.add_argument("--prompt-variant", default="full", choices=["full", "short", "minimal"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--plan-step-gap", type=int, default=4)
    parser.add_argument("--use-recorded-lookdown", action="store_true")
    parser.add_argument("--kv-cache-mode", default="disabled", choices=["disabled", "lookdown_experimental"])
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--verbose-every", type=int, default=20)
    return parser.parse_args()


def run_command(cmd):
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_variants(attn_backends, processor_fast_options):
    variants = []
    for attn_backend in attn_backends:
        for processor_use_fast in processor_fast_options:
            variants.append(
                {
                    "name": f"attn_{attn_backend}__proc_{processor_use_fast}",
                    "attn_backend": attn_backend,
                    "processor_use_fast": processor_use_fast,
                }
            )
    return variants


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def replay_row(variant, summary_path):
    summary = load_json(summary_path)
    return {
        "variant": variant["name"],
        "attn_backend": variant["attn_backend"],
        "processor_use_fast": variant["processor_use_fast"],
        "cold_start_load_seconds": summary["startup"]["cold_start_load_seconds"],
        "gpu_peak_memory_mb": summary["startup"]["gpu_peak_memory_mb"],
        "total_step_p50": summary["latency"]["total_step"]["p50"],
        "total_step_p95": summary["latency"]["total_step"]["p95"],
        "s2_generate_p50": summary["latency"]["s2_generate"]["p50"],
        "s2_generate_p95": summary["latency"]["s2_generate"]["p95"],
        "tokens_per_second_mean": summary["generation"]["tokens_per_second_mean"],
        "output_kind_match_rate": summary["consistency"]["output_kind_match_rate"],
        "action_match_rate_all": summary["consistency"]["action_match_rate_all"],
        "text_exact_match_rate": summary["consistency"]["text_exact_match_rate"],
    }


def loopback_row(variant, summary_path):
    summary = load_json(summary_path)
    return {
        "variant": variant["name"],
        "attn_backend": variant["attn_backend"],
        "processor_use_fast": variant["processor_use_fast"],
        "cold_start_load_seconds": summary["startup"]["cold_start_load_seconds"],
        "gpu_peak_memory_mb": summary["startup"]["gpu_peak_memory_mb"],
        "total_step_p50": summary["latency"]["total_step"]["p50"],
        "total_step_p95": summary["latency"]["total_step"]["p95"],
        "server_model_p50": summary["latency"]["server_model"]["p50"],
        "server_model_p95": summary["latency"]["server_model"]["p95"],
        "output_kind_match_rate": summary["consistency"]["output_kind_match_rate"],
        "discrete_action_match_rate": summary["consistency"]["discrete_action_match_rate"],
        "lookdown_used_steps": summary["breakdown"]["lookdown_used_steps"],
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    variants = build_variants(args.attn_backends, args.processor_fast_options)

    replay_rows = []
    loopback_rows = []
    comparison_all = {"replay": [], "loopback": []}

    for variant in variants:
        if args.mode in ("replay", "both"):
            summary_path = os.path.join(args.output_dir, f"{variant['name']}__replay.json")
            cmd = [
                sys.executable,
                REPLAY_SCRIPT,
                "--manifest",
                args.manifest,
                "--model-path",
                args.model_path,
                "--output",
                summary_path,
                "--device",
                args.device,
                "--num-history",
                str(args.num_history),
                "--prompt-variant",
                args.prompt_variant,
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--attn-backend",
                variant["attn_backend"],
                "--processor-use-fast",
                variant["processor_use_fast"],
                "--verbose-every",
                str(args.verbose_every),
            ]
            if args.base_path:
                cmd.extend(["--base-path", args.base_path])
            if args.max_steps is not None:
                cmd.extend(["--max-steps", str(args.max_steps)])
            run_command(cmd)
            row = replay_row(variant, summary_path)
            replay_rows.append(row)
            comparison_all["replay"].append({"summary_path": summary_path, **row})

        if args.mode in ("loopback", "both"):
            summary_path = os.path.join(args.output_dir, f"{variant['name']}__loopback.json")
            cmd = [
                sys.executable,
                LOOPBACK_SCRIPT,
                "--manifest",
                args.manifest,
                "--model-path",
                args.model_path,
                "--output",
                summary_path,
                "--device",
                args.device,
                "--num-history",
                str(args.num_history),
                "--plan-step-gap",
                str(args.plan_step_gap),
                "--kv-cache-mode",
                args.kv_cache_mode,
                "--attn-backend",
                variant["attn_backend"],
                "--processor-use-fast",
                variant["processor_use_fast"],
                "--verbose-every",
                str(args.verbose_every),
            ]
            if args.base_path:
                cmd.extend(["--base-path", args.base_path])
            if args.max_steps is not None:
                cmd.extend(["--max-steps", str(args.max_steps)])
            if args.use_recorded_lookdown:
                cmd.append("--use-recorded-lookdown")
            run_command(cmd)
            row = loopback_row(variant, summary_path)
            loopback_rows.append(row)
            comparison_all["loopback"].append({"summary_path": summary_path, **row})

    if replay_rows:
        write_csv(os.path.join(args.output_dir, "comparison_backend_replay.csv"), replay_rows)
    if loopback_rows:
        write_csv(os.path.join(args.output_dir, "comparison_backend_loopback.csv"), loopback_rows)
    with open(os.path.join(args.output_dir, "comparison_backend_all.json"), "w", encoding="utf-8") as f:
        json.dump(comparison_all, f, indent=2)


if __name__ == "__main__":
    main()

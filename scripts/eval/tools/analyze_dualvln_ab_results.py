import argparse
import json
from pathlib import Path
from statistics import mean


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize DualVLN HF vs vLLM A/B results.")
    parser.add_argument("--hf-dir", required=True, help="Path to HF output dir containing progress.json")
    parser.add_argument("--vllm-dir", required=True, help="Path to vLLM output dir containing progress.json")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_json(path):
    if not path.exists():
        return None
    text = path.read_text().strip()
    if not text:
        return None
    if "\n" in text:
        text = text.splitlines()[-1]
    return json.loads(text)


def episode_key(row):
    return (row["scene_id"], int(row["episode_id"]))


def summarize_progress(rows):
    n = len(rows)
    if n == 0:
        return {
            "episodes": 0,
            "sr": None,
            "spl": None,
            "ne": None,
            "avg_steps": None,
        }
    return {
        "episodes": n,
        "sr": float(sum(r["success"] for r in rows) / n),
        "spl": float(sum(r["spl"] for r in rows) / n),
        "ne": float(sum(r["ne"] for r in rows) / n),
        "avg_steps": float(sum(r["steps"] for r in rows) / n),
    }


def summarize_runtime(runtime_summary, runtime_rows):
    if runtime_summary is None and not runtime_rows:
        return {
            "available": False,
            "message": "No runtime_summary_rank0.json or runtime_rank0.jsonl found.",
        }

    out = {"available": True}
    if runtime_summary is not None:
        out["runtime_summary"] = runtime_summary
    if runtime_rows:
        out["runtime_jsonl_summary"] = {
            "episodes": len(runtime_rows),
            "episode_wall_clock_seconds_mean": float(mean([r["episode_wall_clock_seconds"] for r in runtime_rows])),
            "avg_step_wall_clock_seconds_mean": float(mean([r["avg_step_wall_clock_seconds"] for r in runtime_rows])),
            "s2_avg_seconds_mean": float(mean([r["s2_avg_seconds"] for r in runtime_rows])),
            "s1_avg_seconds_mean": float(mean([r["s1_avg_seconds"] for r in runtime_rows])),
            "s2_latent_seconds_mean": float(mean([r.get("s2_latent_seconds", 0.0) for r in runtime_rows])),
            "s2_call_count_mean": float(mean([r["s2_call_count"] for r in runtime_rows])),
            "s1_call_count_mean": float(mean([r["s1_call_count"] for r in runtime_rows])),
        }
    return out


def summarize_pairwise(common_keys, hf_map, vllm_map):
    if not common_keys:
        return {
            "episodes": 0,
            "delta_sr": None,
            "delta_spl": None,
            "delta_ne": None,
            "delta_avg_steps": None,
            "success_flip_counts": {"vllm_better": 0, "hf_better": 0, "unchanged": 0},
        }

    success_diffs = [vllm_map[k]["success"] - hf_map[k]["success"] for k in common_keys]
    spl_diffs = [vllm_map[k]["spl"] - hf_map[k]["spl"] for k in common_keys]
    ne_diffs = [vllm_map[k]["ne"] - hf_map[k]["ne"] for k in common_keys]
    step_diffs = [vllm_map[k]["steps"] - hf_map[k]["steps"] for k in common_keys]
    return {
        "episodes": len(common_keys),
        "delta_sr": float(mean(success_diffs)),
        "delta_spl": float(mean(spl_diffs)),
        "delta_ne": float(mean(ne_diffs)),
        "delta_avg_steps": float(mean(step_diffs)),
        "success_flip_counts": {
            "vllm_better": int(sum(d > 0 for d in success_diffs)),
            "hf_better": int(sum(d < 0 for d in success_diffs)),
            "unchanged": int(sum(d == 0 for d in success_diffs)),
        },
    }


def main():
    args = parse_args()
    hf_dir = Path(args.hf_dir)
    vllm_dir = Path(args.vllm_dir)

    hf_progress = load_jsonl(hf_dir / "progress.json")
    vllm_progress = load_jsonl(vllm_dir / "progress.json")
    hf_result = load_json(hf_dir / "result.json")
    vllm_result = load_json(vllm_dir / "result.json")

    hf_map = {episode_key(row): row for row in hf_progress}
    vllm_map = {episode_key(row): row for row in vllm_progress}
    common_keys = sorted(hf_map.keys() & vllm_map.keys())
    hf_only = sorted(hf_map.keys() - vllm_map.keys())
    vllm_only = sorted(vllm_map.keys() - hf_map.keys())

    hf_common_rows = [hf_map[k] for k in common_keys]
    vllm_common_rows = [vllm_map[k] for k in common_keys]

    report = {
        "inputs": {
            "hf_dir": str(hf_dir),
            "vllm_dir": str(vllm_dir),
        },
        "raw_results": {
            "hf_result_json": hf_result,
            "vllm_result_json": vllm_result,
        },
        "progress_coverage": {
            "hf_rows": len(hf_progress),
            "vllm_rows": len(vllm_progress),
            "hf_unique_episodes": len(hf_map),
            "vllm_unique_episodes": len(vllm_map),
            "common_episodes": len(common_keys),
            "hf_only_episodes": [{"scene_id": s, "episode_id": e} for s, e in hf_only],
            "vllm_only_episodes": [{"scene_id": s, "episode_id": e} for s, e in vllm_only],
        },
        "common_episode_metrics": {
            "hf": summarize_progress(hf_common_rows),
            "vllm": summarize_progress(vllm_common_rows),
            "pairwise": summarize_pairwise(common_keys, hf_map, vllm_map),
        },
        "runtime": {
            "hf": summarize_runtime(
                load_json(hf_dir / "runtime_summary_rank0.json"),
                load_jsonl(hf_dir / "runtime_rank0.jsonl"),
            ),
            "vllm": summarize_runtime(
                load_json(vllm_dir / "runtime_summary_rank0.json"),
                load_jsonl(vllm_dir / "runtime_rank0.jsonl"),
            ),
        },
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print("=" * 72)
    print("DualVLN HF vs vLLM A/B Summary")
    print("=" * 72)
    print(
        "Coverage:",
        f"hf_rows={report['progress_coverage']['hf_rows']}",
        f"vllm_rows={report['progress_coverage']['vllm_rows']}",
        f"common={report['progress_coverage']['common_episodes']}",
    )
    if hf_only or vllm_only:
        print("Coverage warning: episode sets differ between HF and vLLM outputs.")
    hf_common = report["common_episode_metrics"]["hf"]
    vllm_common = report["common_episode_metrics"]["vllm"]
    pair = report["common_episode_metrics"]["pairwise"]
    print(
        "Common episodes:",
        f"SR {hf_common['sr']:.4f} -> {vllm_common['sr']:.4f}",
        f"SPL {hf_common['spl']:.4f} -> {vllm_common['spl']:.4f}",
        f"NE {hf_common['ne']:.4f} -> {vllm_common['ne']:.4f}",
        f"Steps {hf_common['avg_steps']:.2f} -> {vllm_common['avg_steps']:.2f}",
    )
    print(
        "Pairwise delta:",
        f"dSR={pair['delta_sr']:.4f}",
        f"dSPL={pair['delta_spl']:.4f}",
        f"dNE={pair['delta_ne']:.4f}",
        f"dSteps={pair['delta_avg_steps']:.2f}",
    )
    flips = pair["success_flip_counts"]
    print(
        "Success flips:",
        f"vllm_better={flips['vllm_better']}",
        f"hf_better={flips['hf_better']}",
        f"unchanged={flips['unchanged']}",
    )
    for label in ("hf", "vllm"):
        runtime = report["runtime"][label]
        if runtime["available"]:
            src = "runtime_summary_rank0.json" if "runtime_summary" in runtime else "runtime_rank0.jsonl"
            print(f"{label} runtime: available via {src}")
        else:
            print(f"{label} runtime: unavailable")
    if args.output:
        print(f"Saved JSON summary to {args.output}")
    print("=" * 72)


if __name__ == "__main__":
    main()

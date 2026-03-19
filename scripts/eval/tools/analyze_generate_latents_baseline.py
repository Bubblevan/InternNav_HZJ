import argparse
import json
from pathlib import Path
from statistics import mean

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze exported generate_latents baseline samples.")
    parser.add_argument("--input-dir", required=True, help="Directory created by export_hf_generate_latents_baseline.py")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def load_metadata(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    metadata_path = input_dir / "metadata.jsonl"
    samples_dir = input_dir / "samples"
    metadata = load_metadata(metadata_path)

    if not metadata:
        raise SystemExit(f"No metadata found at {metadata_path}")

    max_abs_diffs = []
    mean_abs_diffs = []
    latent_allclose_count = 0
    checked_samples = 0

    for item in metadata:
        sample_path = Path(item["sample_path"])
        if not sample_path.is_absolute():
            sample_path = Path.cwd() / sample_path
        if not sample_path.exists():
            sample_path = samples_dir / sample_path.name
        sample = torch.load(sample_path, map_location="cpu")

        baseline_latent = sample["baseline_latent"].float()
        hf_generate_latent = sample["hf_generate_latent"].float()
        diff = (baseline_latent - hf_generate_latent).abs()

        max_abs_diffs.append(float(diff.max().item()))
        mean_abs_diffs.append(float(diff.mean().item()))
        latent_allclose_count += int(torch.allclose(baseline_latent, hf_generate_latent, atol=1e-4, rtol=1e-4))
        checked_samples += 1

    report = {
        "input_dir": str(input_dir),
        "counts": {
            "metadata_records": len(metadata),
            "checked_samples": checked_samples,
            "text_exact_matches": int(sum(item["text_exact_match"] for item in metadata)),
            "pixel_goal_matches": int(sum(item["pixel_goal_match"] for item in metadata)),
            "latent_allclose_matches": latent_allclose_count,
        },
        "rates": {
            "text_exact_match_rate": float(sum(item["text_exact_match"] for item in metadata) / len(metadata)),
            "pixel_goal_match_rate": float(sum(item["pixel_goal_match"] for item in metadata) / len(metadata)),
            "latent_allclose_rate": float(latent_allclose_count / checked_samples) if checked_samples else None,
        },
        "latent_diff": {
            "max_abs_diff_mean": float(mean(max_abs_diffs)) if max_abs_diffs else None,
            "max_abs_diff_max": max(max_abs_diffs) if max_abs_diffs else None,
            "mean_abs_diff_mean": float(mean(mean_abs_diffs)) if mean_abs_diffs else None,
            "mean_abs_diff_max": max(mean_abs_diffs) if mean_abs_diffs else None,
        },
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 72)
    print("Analyze generate_latents baseline")
    print("=" * 72)
    print(f"Samples checked: {checked_samples}")
    print(f"Text exact match rate: {report['rates']['text_exact_match_rate']:.4f}")
    print(f"Pixel-goal match rate: {report['rates']['pixel_goal_match_rate']:.4f}")
    print(f"Latent allclose rate: {report['rates']['latent_allclose_rate']:.4f}")
    print(f"Latent max abs diff (mean/max): {report['latent_diff']['max_abs_diff_mean']:.6f} / {report['latent_diff']['max_abs_diff_max']:.6f}")
    print(f"Latent mean abs diff (mean/max): {report['latent_diff']['mean_abs_diff_mean']:.6f} / {report['latent_diff']['mean_abs_diff_max']:.6f}")
    if args.output:
        print(f"Saved JSON summary to {args.output}")
    print("=" * 72)


if __name__ == "__main__":
    main()

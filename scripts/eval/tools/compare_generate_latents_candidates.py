import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Optional


import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare candidate latent tensors against exported generate_latents baseline samples."
    )
    parser.add_argument("--input-dir", required=True, help="Directory created by export_hf_generate_latents_baseline.py")
    parser.add_argument("--reference-key", default="baseline_latent", help="Tensor key used as the reference latent")
    parser.add_argument(
        "--candidate-key",
        default="hf_generate_latent",
        help="Tensor key used as the candidate latent when reading directly from each sample",
    )
    parser.add_argument(
        "--external-dir",
        default=None,
        help="Optional directory of external candidate .pt files named the same as sample files",
    )
    parser.add_argument(
        "--external-key",
        default="latent",
        help="Tensor key to read from external candidate .pt files when --external-dir is provided",
    )
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for torch.allclose")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for torch.allclose")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def load_metadata(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_sample_paths(input_dir: Path, metadata):
    samples_dir = input_dir / "samples"
    if metadata:
        paths = []
        for item in metadata:
            sample_path = Path(item["sample_path"])
            if not sample_path.is_absolute():
                sample_path = samples_dir / sample_path.name
            paths.append((item.get("sample_name", sample_path.stem), sample_path))
        return paths
    return [(path.stem, path) for path in sorted(samples_dir.glob("*.pt"))]


def load_candidate_tensor(
    sample_name: str,
    sample_path: Path,
    sample: dict,
    candidate_key: str,
    external_dir: Optional[Path],
    external_key: str,
):
    if external_dir is None:
        if candidate_key not in sample:
            raise KeyError(f"{sample_path}: missing candidate key {candidate_key!r}")
        return sample[candidate_key]

    external_path = external_dir / sample_path.name
    if not external_path.exists():
        raise FileNotFoundError(f"Missing external candidate file for {sample_name}: {external_path}")

    external_sample = torch.load(external_path, map_location="cpu")
    if external_key not in external_sample:
        raise KeyError(f"{external_path}: missing external key {external_key!r}")
    return external_sample[external_key]


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    metadata = load_metadata(input_dir / "metadata.jsonl")
    sample_items = resolve_sample_paths(input_dir, metadata)
    if not sample_items:
        raise SystemExit(f"No .pt samples found under {input_dir / 'samples'}")

    external_dir = Path(args.external_dir) if args.external_dir else None

    per_sample = []
    shape_match_count = 0
    allclose_count = 0
    max_abs_diffs = []
    mean_abs_diffs = []

    for sample_name, sample_path in sample_items:
        sample = torch.load(sample_path, map_location="cpu")
        if args.reference_key not in sample:
            raise KeyError(f"{sample_path}: missing reference key {args.reference_key!r}")

        reference = sample[args.reference_key].float()
        candidate = load_candidate_tensor(
            sample_name=sample_name,
            sample_path=sample_path,
            sample=sample,
            candidate_key=args.candidate_key,
            external_dir=external_dir,
            external_key=args.external_key,
        ).float()

        shape_match = list(reference.shape) == list(candidate.shape)
        record = {
            "sample_name": sample_name,
            "sample_path": str(sample_path),
            "reference_key": args.reference_key,
            "candidate_key": args.external_key if external_dir else args.candidate_key,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "shape_match": bool(shape_match),
        }

        if shape_match:
            diff = (reference - candidate).abs()
            max_abs_diff = float(diff.max().item())
            mean_abs_diff = float(diff.mean().item())
            is_allclose = bool(torch.allclose(reference, candidate, atol=args.atol, rtol=args.rtol))

            shape_match_count += 1
            allclose_count += int(is_allclose)
            max_abs_diffs.append(max_abs_diff)
            mean_abs_diffs.append(mean_abs_diff)

            record.update(
                {
                    "allclose": is_allclose,
                    "max_abs_diff": max_abs_diff,
                    "mean_abs_diff": mean_abs_diff,
                }
            )
        else:
            record.update(
                {
                    "allclose": False,
                    "max_abs_diff": None,
                    "mean_abs_diff": None,
                }
            )

        per_sample.append(record)

    report = {
        "input_dir": str(input_dir),
        "external_dir": str(external_dir) if external_dir else None,
        "reference_key": args.reference_key,
        "candidate_key": args.external_key if external_dir else args.candidate_key,
        "tolerance": {"atol": args.atol, "rtol": args.rtol},
        "counts": {
            "samples": len(per_sample),
            "shape_matches": shape_match_count,
            "allclose_matches": allclose_count,
        },
        "rates": {
            "shape_match_rate": float(shape_match_count / len(per_sample)),
            "allclose_rate": float(allclose_count / shape_match_count) if shape_match_count else None,
        },
        "diff": {
            "max_abs_diff_mean": float(mean(max_abs_diffs)) if max_abs_diffs else None,
            "max_abs_diff_max": max(max_abs_diffs) if max_abs_diffs else None,
            "mean_abs_diff_mean": float(mean(mean_abs_diffs)) if mean_abs_diffs else None,
            "mean_abs_diff_max": max(mean_abs_diffs) if mean_abs_diffs else None,
        },
        "per_sample": per_sample,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 72)
    print("Compare generate_latents candidates")
    print("=" * 72)
    print(f"Samples: {report['counts']['samples']}")
    print(f"Shape match rate: {report['rates']['shape_match_rate']:.4f}")
    allclose_rate = report["rates"]["allclose_rate"]
    print(f"Allclose rate: {allclose_rate:.4f}" if allclose_rate is not None else "Allclose rate: n/a")
    print(
        f"Max abs diff (mean/max): {report['diff']['max_abs_diff_mean']:.6f} / {report['diff']['max_abs_diff_max']:.6f}"
        if report["diff"]["max_abs_diff_mean"] is not None
        else "Max abs diff (mean/max): n/a"
    )
    print(
        f"Mean abs diff (mean/max): {report['diff']['mean_abs_diff_mean']:.6f} / {report['diff']['mean_abs_diff_max']:.6f}"
        if report["diff"]["mean_abs_diff_mean"] is not None
        else "Mean abs diff (mean/max): n/a"
    )
    if args.output:
        print(f"Saved JSON summary to {args.output}")
    print("=" * 72)


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path


DIRECT_POOLING_ARCHES = {
    "Qwen2VLForConditionalGeneration": {
        "support": "direct_embedding_registry",
        "notes": "In local vLLM registry, this multimodal architecture appears in embedding models.",
    },
}

CONVERTIBLE_GENERATIVE_ARCH_SUFFIXES = (
    "ForCausalLM",
    "ForConditionalGeneration",
    "ChatModel",
    "LMHeadModel",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Static probe for whether a model is likely usable with local vLLM pooling/token_embed paths."
    )
    parser.add_argument("--model-path", required=True, help="HF model directory containing config.json")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def load_architectures(model_path: Path):
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise SystemExit(f"Missing config.json under {model_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    if not architectures:
        raise SystemExit(f"No architectures found in {config_path}")
    return config, architectures


def classify_architecture(arch: str):
    if arch in DIRECT_POOLING_ARCHES:
        return {
            "architecture": arch,
            "status": "direct_pooling_candidate",
            "reason": DIRECT_POOLING_ARCHES[arch]["notes"],
        }

    if arch.endswith(CONVERTIBLE_GENERATIVE_ARCH_SUFFIXES):
        return {
            "architecture": arch,
            "status": "convertible_pooling_candidate",
            "reason": (
                "This looks like a generative architecture. "
                "Local vLLM source allows `--runner pooling --convert embed|classify|reward` "
                "for models that are not native pooling models."
            ),
        }

    return {
        "architecture": arch,
        "status": "unknown",
        "reason": "This architecture does not match the current static heuristics.",
    }


def summarize(records):
    statuses = [item["status"] for item in records]
    if any(status == "direct_pooling_candidate" for status in statuses):
        return {
            "overall_status": "has_direct_pooling_path",
            "summary": (
                "At least one architecture is directly registered as a pooling/embedding candidate in local vLLM source."
            ),
        }
    if any(status == "convertible_pooling_candidate" for status in statuses):
        return {
            "overall_status": "likely_requires_convert_embed",
            "summary": (
                "No direct pooling registration was found by this script, but the architecture looks generative and "
                "may be adaptable via `--runner pooling --convert embed`."
            ),
        }
    return {
        "overall_status": "unknown",
        "summary": "Static probe could not establish a likely pooling path.",
    }


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    config, architectures = load_architectures(model_path)

    per_arch = [classify_architecture(arch) for arch in architectures]
    overall = summarize(per_arch)

    report = {
        "model_path": str(model_path),
        "architectures": architectures,
        "model_type": config.get("model_type"),
        "is_multimodal_hint": any("vl" in arch.lower() or "vision" in arch.lower() for arch in architectures),
        "source_based_findings": {
            "llm_encode_requires_pooling_runner": True,
            "token_embed_returns_token_matrix": True,
            "token_embed_step_pooling_can_filter_tokens": True,
            "multimodal_score_path_exists_in_local_source": True,
        },
        "per_architecture": per_arch,
        "overall": overall,
        "next_checks": [
            "Confirm whether the exact served checkpoint architecture is direct-pooling or convert-to-embed only.",
            "If using convert-to-embed, verify that multimodal Qwen2.5-VL still accepts image inputs in pooling runner.",
            "If pooling runner works, test whether token_embed can isolate the appended TRAJ token positions.",
            "Even if token_embed runs, it still must be compared against baseline_latent with the offline comparator.",
        ],
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 72)
    print("Probe vLLM pooling capability")
    print("=" * 72)
    print(f"Model path: {model_path}")
    print(f"Architectures: {architectures}")
    print(f"Overall: {overall['overall_status']}")
    print(overall["summary"])
    if args.output:
        print(f"Saved JSON summary to {args.output}")
    print("=" * 72)


if __name__ == "__main__":
    main()

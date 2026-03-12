import argparse
import json
import os
import shutil
from collections import Counter


STANDARD_QWEN25VL_MODEL_TYPE = "qwen2_5_vl"
STANDARD_QWEN25VL_ARCH = "Qwen2_5_VLForConditionalGeneration"


def parse_args():
    parser = argparse.ArgumentParser(description="Check whether a DualVLN checkpoint is a good vLLM S2-only candidate.")
    parser.add_argument("--model-path", required=True, help="Checkpoint directory")
    parser.add_argument("--output", default=None, help="Optional JSON summary output")
    parser.add_argument(
        "--patched-model-path",
        default=None,
        help="Optional output directory for a standard-Qwen2.5-VL config view used by S2-only vLLM experiments",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_weight_map(index_json):
    keys = list(index_json["weight_map"].keys())
    counts = Counter()
    for key in keys:
        root = key.split(".", 1)[0]
        counts[root] += 1

    custom_patterns = {
        "latent_queries": sum("latent_queries" in k for k in keys),
        "rgb_model": sum("rgb_model" in k for k in keys),
        "memory_encoder": sum("memory_encoder" in k for k in keys),
        "rgb_resampler": sum("rgb_resampler" in k for k in keys),
        "cond_projector": sum("cond_projector" in k for k in keys),
    }
    return {
        "num_tensors": len(keys),
        "top_level_prefix_counts": dict(counts),
        "custom_module_tensor_counts": custom_patterns,
    }


def build_assessment(config, weights_summary):
    model_type = config.get("model_type")
    archs = config.get("architectures") or []
    has_custom_modules = any(weights_summary["custom_module_tensor_counts"].values())

    blockers = []
    cautions = []
    positives = []

    if model_type != STANDARD_QWEN25VL_MODEL_TYPE:
        blockers.append(
            f"config.model_type={model_type} is custom; vLLM will not see this as a standard Qwen2.5-VL checkpoint."
        )
    else:
        positives.append("config.model_type already matches standard qwen2_5_vl.")

    if STANDARD_QWEN25VL_ARCH not in archs:
        blockers.append(f"architectures={archs} does not include {STANDARD_QWEN25VL_ARCH}.")
    else:
        positives.append("architectures already include standard Qwen2.5-VL causal LM.")

    if has_custom_modules:
        cautions.append(
            "checkpoint contains extra DualVLN/System1 tensors; vLLM S2-only experiments must treat them as non-serving extras."
        )

    positives.append("vision_config.model_type is qwen2_5_vl, suggesting the visual tower is still Qwen2.5-VL-based.")

    recommendation = (
        "Good S2-only candidate with a patched standard-Qwen2.5-VL config view."
        if blockers
        else "Can try direct vLLM loading without config patching."
    )

    return {
        "blockers": blockers,
        "cautions": cautions,
        "positives": positives,
        "recommendation": recommendation,
    }


def create_patched_view(model_path, output_path):
    if os.path.exists(output_path):
        raise FileExistsError(f"patched model path already exists: {output_path}")
    os.makedirs(output_path, exist_ok=False)

    for name in os.listdir(model_path):
        src = os.path.join(model_path, name)
        dst = os.path.join(output_path, name)
        if name == "config.json":
            continue
        os.symlink(src, dst)

    config = load_json(os.path.join(model_path, "config.json"))
    patched = dict(config)
    patched["model_type"] = STANDARD_QWEN25VL_MODEL_TYPE
    patched["architectures"] = [STANDARD_QWEN25VL_ARCH]
    patched["_comment"] = (
        "Patched view for S2-only vLLM experiments. "
        "Original checkpoint is InternVLAN1/DualVLN; this view only targets Qwen2.5-VL text+vision generation."
    )
    with open(os.path.join(output_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(patched, f, indent=2)

    return {
        "patched_model_path": output_path,
        "patched_model_type": patched["model_type"],
        "patched_architectures": patched["architectures"],
    }


def main():
    args = parse_args()
    config_path = os.path.join(args.model_path, "config.json")
    index_path = os.path.join(args.model_path, "model.safetensors.index.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(config_path)
    if not os.path.exists(index_path):
        raise FileNotFoundError(index_path)

    config = load_json(config_path)
    index_json = load_json(index_path)
    weights_summary = summarize_weight_map(index_json)
    assessment = build_assessment(config, weights_summary)

    result = {
        "model_path": args.model_path,
        "config": {
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "vision_config_model_type": (config.get("vision_config") or {}).get("model_type"),
        },
        "weights_summary": weights_summary,
        "assessment": assessment,
    }

    if args.patched_model_path:
        result["patched_view"] = create_patched_view(args.model_path, args.patched_model_path)

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

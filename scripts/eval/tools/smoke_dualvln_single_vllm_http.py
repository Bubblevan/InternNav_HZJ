import argparse
import json
from pathlib import Path

from PIL import Image

from dualvln_runtime.http import DualVLNSingleVLLMHTTPClient


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal smoke test for DualVLN single-vLLM HTTP server.")
    parser.add_argument("--base-url", required=True, help="Server base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--image", required=True, help="Path to an RGB image used in the request")
    parser.add_argument(
        "--instruction",
        default=(
            "Where should you go next to stay on track? "
            "Please output the next waypoint coordinates in the image as two integers."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--return-hidden-states", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    image = Image.open(Path(args.image)).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.instruction},
            ],
        }
    ]
    client = DualVLNSingleVLLMHTTPClient(args.base_url, timeout=args.timeout)
    result = client.step_s2(
        messages,
        max_new_tokens=args.max_new_tokens,
        return_latents=True,
        return_hidden_states=args.return_hidden_states,
    )
    latents = result.get("latents")
    runtime_metrics = result.get("runtime_metrics") or {}
    summary = {
        "backend_name": ((result.get("backend_metadata") or {}).get("backend_name")),
        "model_adapter": (((result.get("backend_metadata") or {}).get("model_adapter")) or {}).get("adapter_name"),
        "backend_capabilities": ((result.get("backend_metadata") or {}).get("capabilities")),
        "llm_output": result.get("llm_output"),
        "pixel_goal": result.get("pixel_goal"),
        "prompt_token_count": len(result.get("prompt_token_ids") or []),
        "generated_token_count": len(result.get("generated_token_ids") or []),
        "latents_shape": list(latents.shape) if latents is not None else None,
        "hidden_states_shape": runtime_metrics.get("hidden_states_shape"),
        "runtime_metrics": runtime_metrics,
        "transport_metrics": result.get("transport_metrics"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

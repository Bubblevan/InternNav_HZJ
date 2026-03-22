import argparse
import time

from flask import Flask, jsonify, request

from internnav.model.utils.dualvln_single_vllm import (
    DualVLNSingleVLLMRunner,
    decode_messages,
    encode_tensor_to_b64,
)

app = Flask(__name__)
runner = None
served_model_name = None


@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify(
        {
            "object": "list",
            "data": [
                {
                    "id": served_model_name,
                    "object": "model",
                }
            ],
        }
    )


@app.route("/dualvln/step_s2", methods=["POST"])
def dualvln_step_s2():
    payload = request.get_json(force=True)
    messages = decode_messages(payload["messages"])
    max_new_tokens = int(payload.get("max_new_tokens", 128))
    start_time = time.time()
    result = runner.step_s2(messages, max_new_tokens=max_new_tokens)
    latency = time.time() - start_time
    print(
        "[DualVLN Single vLLM] /dualvln/step_s2 "
        f"latency={latency:.3f}s "
        f"pixel_goal={result['pixel_goal']} "
        f"gen_tokens={len(result['generated_token_ids'])}",
        flush=True,
    )
    return jsonify(
        {
            "llm_output": result["llm_output"],
            "prompt_token_ids": result["prompt_token_ids"],
            "generated_token_ids": result["generated_token_ids"],
            "pixel_goal": result["pixel_goal"],
            "latents": encode_tensor_to_b64(result["latents"]) if result["latents"] is not None else None,
        }
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Single-engine vLLM server for DualVLN S2 generate + latents.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--hf-model-path",
        default=None,
        help="Optional original DualVLN checkpoint path used to load latent_queries; defaults to --model-path.",
    )
    parser.add_argument("--served-model-name", default="dualvln-single-vllm")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--limit-mm-per-prompt-image", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--model-impl", default="auto")
    parser.add_argument(
        "--latent-backend",
        default=None,
        help="Latent extraction backend. Defaults to transformers_backend_apply_model when --model-impl=transformers, otherwise legacy_custom_forward.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    served_model_name = args.served_model_name
    runner = DualVLNSingleVLLMRunner(
        model_path=args.model_path,
        hf_model_path=args.hf_model_path,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt_image=args.limit_mm_per_prompt_image,
        tensor_parallel_size=args.tensor_parallel_size,
        model_impl=args.model_impl,
        latent_backend=args.latent_backend,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    print(
        "[DualVLN Single vLLM] ready "
        f"model={args.model_path} "
        f"hf_model_path={args.hf_model_path or args.model_path} "
        f"model_impl={args.model_impl} "
        f"latent_backend={args.latent_backend or ('transformers_backend_apply_model' if args.model_impl == 'transformers' else 'legacy_custom_forward')} "
        f"served_model_name={served_model_name} "
        f"port={args.port}",
        flush=True,
    )
    app.run(host=args.host, port=args.port)

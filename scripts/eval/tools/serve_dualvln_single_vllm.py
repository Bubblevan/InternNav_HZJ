import argparse
import time
import os

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
    server_total_start = time.perf_counter()
    request_parse_start = server_total_start
    payload = request.get_json(force=True, cache=False)
    server_request_parse_ms = (time.perf_counter() - request_parse_start) * 1000.0
    image_transport_mode = str(payload.get("image_transport_mode", "base64"))

    decode_start = time.perf_counter()
    messages = decode_messages(payload["messages"])
    server_decode_messages_ms = (time.perf_counter() - decode_start) * 1000.0
    max_new_tokens = int(payload.get("max_new_tokens", 128))

    runner_start = time.perf_counter()
    result = runner.step_s2(messages, max_new_tokens=max_new_tokens)
    server_runner_step_s2_ms = (time.perf_counter() - runner_start) * 1000.0

    encode_start = time.perf_counter()
    latents_payload = encode_tensor_to_b64(result["latents"]) if result["latents"] is not None else None
    runtime_total_ms = ((result.get("runtime_metrics") or {}).get("total_ms"))
    transport_metrics = {
        "server_request_parse_ms": server_request_parse_ms,
        "server_decode_messages_ms": server_decode_messages_ms,
        "server_runner_step_s2_ms": server_runner_step_s2_ms,
        "server_encode_response_ms": None,
        "server_total_ms": None,
        "server_outer_overhead_ms": None,
        "image_transport_mode": image_transport_mode,
    }
    response_payload = {
        "llm_output": result["llm_output"],
        "prompt_token_ids": result["prompt_token_ids"],
        "generated_token_ids": result["generated_token_ids"],
        "pixel_goal": result["pixel_goal"],
        "latents": latents_payload,
        "runtime_metrics": result.get("runtime_metrics"),
        "vllm_kv_cache": result.get("vllm_kv_cache"),
        "transport_metrics": transport_metrics,
        "debug_mm": result.get("debug_mm"),
    }
    transport_metrics["server_encode_response_ms"] = (time.perf_counter() - encode_start) * 1000.0
    transport_metrics["server_total_ms"] = (time.perf_counter() - server_total_start) * 1000.0
    transport_metrics["server_outer_overhead_ms"] = (
        float(max(transport_metrics["server_total_ms"] - runtime_total_ms, 0.0))
        if runtime_total_ms is not None
        else None
    )
    print(
        "[DualVLN Single vLLM] /dualvln/step_s2 "
        f"latency={transport_metrics['server_total_ms'] / 1000.0:.3f}s "
        f"pixel_goal={result['pixel_goal']} "
        f"gen_tokens={len(result['generated_token_ids'])}",
        flush=True,
    )
    return jsonify(response_payload)


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
        help=(
            "Latent extraction backend. Defaults to shared_engine_forward; "
            "vllm_hidden is kept as an alias to that shared-engine path, and "
            "vllm_hidden_separate_llm keeps the old second-LLM behavior for debugging."
        ),
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--enable-mm-encoder-compile",
        action="store_true",
        help="Enable vLLM torch.compile for multimodal encoder.",
    )
    parser.add_argument(
        "--force-v1",
        action="store_true",
        help="Set VLLM_USE_V1=1 before engine init.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.force_v1:
        os.environ["VLLM_USE_V1"] = "1"
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
        compilation_config=(
        {"compile_mm_encoder": True}
        if args.enable_mm_encoder_compile
        else None
    ),
    )
    print(
        "[DualVLN Single vLLM] ready "
        f"model={args.model_path} "
        f"hf_model_path={args.hf_model_path or args.model_path} "
        f"model_impl={args.model_impl} "
        f"latent_backend={args.latent_backend or 'shared_engine_forward'} "
        f"served_model_name={served_model_name} "
        f"port={args.port}",
        flush=True,
    )
    app.run(host=args.host, port=args.port)

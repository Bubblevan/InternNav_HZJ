import argparse
import os

from dualvln_runtime.http import create_dualvln_app


def parse_args():
    parser = argparse.ArgumentParser(description="Single-engine vLLM server for DualVLN S2 generate + latents.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--backend",
        choices=("patched_vllm", "native"),
        default="patched_vllm",
        help="Execution backend used by the miniengine.",
    )
    parser.add_argument(
        "--model-adapter",
        choices=("dualvln", "navila_stub"),
        default="dualvln",
        help="Model adapter used by the platform engine.",
    )
    parser.add_argument(
        "--hf-model-path",
        default=None,
        help="Optional original DualVLN checkpoint path used to load latent_queries; defaults to --model-path.",
    )
    parser.add_argument("--served-model-name", default="dualvln-single-vllm")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--native-attn-implementation",
        default=None,
        help="Optional HF attention implementation for the native backend, e.g. flash_attention_2 or sdpa.",
    )
    parser.add_argument(
        "--native-generation-impl",
        choices=("hf_generate", "custom_greedy"),
        default="hf_generate",
        help="Native backend generation path. Keep hf_generate as the semantic-reference default.",
    )
    parser.add_argument(
        "--native-processor-use-fast",
        choices=("true", "false"),
        default="true",
        help="Whether the native backend should use the fast Qwen processor. Default true to match patched-vLLM preprocessing.",
    )
    parser.add_argument(
        "--native-enable-same-request-continuation",
        choices=("true", "false"),
        default="false",
        help=(
            "Whether the native backend should reuse generate() cache for the latent "
            "suffix continuation path. Set false to force exact full-prefill latent "
            "extraction. Default false because continuation currently causes "
            "semantic drift in navigation."
        ),
    )
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
    parser.add_argument(
        "--quantization",
        default=None,
        choices=("awq", "fp8", "gptq", "squeezellm", "marlin", "gguf", "smoothquant"),
        help="vLLM quantization method for the model weights.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.force_v1:
        os.environ["VLLM_USE_V1"] = "1"
    if args.backend == "patched_vllm":
        from dualvln_miniengine.backends import PatchedVLLMDualVLNMiniEngine

        engine = PatchedVLLMDualVLNMiniEngine.from_pretrained(
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
            model_adapter=args.model_adapter,
            compilation_config=(
                {"compile_mm_encoder": True}
                if args.enable_mm_encoder_compile
                else None
            ),
            quantization=args.quantization,
        )
    else:
        from dualvln_miniengine.backends import NativeDualVLNMiniEngine

        engine = NativeDualVLNMiniEngine.from_pretrained(
            model_path=args.model_path,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.native_attn_implementation,
            generation_impl=args.native_generation_impl,
            processor_use_fast=args.native_processor_use_fast == "true",
            enable_same_request_continuation=(
                args.native_enable_same_request_continuation == "true"
            ),
            model_adapter=args.model_adapter,
        )
    app = create_dualvln_app(
        engine=engine,
        served_model_name=args.served_model_name,
    )
    print(
        "[DualVLN MiniEngine] ready "
        f"backend={args.backend} "
        f"adapter={args.model_adapter} "
        f"model={args.model_path} "
        f"hf_model_path={args.hf_model_path or args.model_path} "
        f"model_impl={args.model_impl} "
        f"latent_backend={args.latent_backend or 'shared_engine_forward'} "
        f"native_processor_use_fast={getattr(args, 'native_processor_use_fast', 'n/a')} "
        f"native_enable_same_request_continuation={getattr(args, 'native_enable_same_request_continuation', 'n/a')} "
        f"served_model_name={args.served_model_name} "
        f"port={args.port}",
        flush=True,
    )
    app.run(host=args.host, port=args.port)

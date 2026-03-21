import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import time
import traceback
import sys

import torch

from internnav.model.utils.vllm_hidden_latents import (
    VLLMHiddenLatentsRunner,
    decode_pil_image_from_b64,
    decode_tensor_from_b64,
    encode_tensor_to_b64,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Serve patched vLLM hidden-state generate_latents over local HTTP."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--dump-dir",
        default="./logs/habitat/vllm_generate_latents_http_dump",
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--limit-mm-per-prompt-image", type=int, default=16)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


class HiddenLatentsHandler(BaseHTTPRequestHandler):
    runner = None

    def _send_json(self, status_code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"ok": True})
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self):
        if self.path != "/generate_latents":
            self._send_json(404, {"error": "not_found"})
            return

        try:
            start_time = time.perf_counter()
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))

            output_ids = decode_tensor_from_b64(payload["output_ids"])
            pixel_values = decode_tensor_from_b64(payload["pixel_values"])
            image_grid_thw = decode_tensor_from_b64(payload["image_grid_thw"])
            latent_queries = decode_tensor_from_b64(payload["latent_queries"])
            input_images = [decode_pil_image_from_b64(x) for x in payload.get("input_images", [])]

            print(
                "[HiddenLatents] POST /generate_latents "
                f"output_ids={tuple(output_ids.shape)} "
                f"pixel_values={tuple(pixel_values.shape)} "
                f"image_grid_thw={tuple(image_grid_thw.shape)} "
                f"latent_queries={tuple(latent_queries.shape)} "
                f"input_images={len(input_images)}",
                flush=True,
            )

            latents = self.runner.generate_latents(
                output_ids=output_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                input_images=input_images,
                latent_queries=latent_queries,
                traj_token_index=int(payload["traj_token_index"]),
                n_query=int(payload["n_query"]),
            )

            self._send_json(
                200,
                {
                    "latents": encode_tensor_to_b64(latents),
                    "shape": list(latents.shape),
                },
            )
            print(
                "[HiddenLatents] completed "
                f"latents={tuple(latents.shape)} "
                f"elapsed_s={time.perf_counter() - start_time:.3f}",
                flush=True,
            )
        except Exception as exc:
            tb = traceback.format_exc()
            print(tb, file=sys.stderr, flush=True)
            self._send_json(
                500,
                {
                    "error": type(exc).__name__,
                    "message": str(exc),
                    "traceback": tb,
                },
            )


def main():
    args = parse_args()
    HiddenLatentsHandler.runner = VLLMHiddenLatentsRunner(
        model_path=args.model_path,
        dump_dir=args.dump_dir,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt_image=args.limit_mm_per_prompt_image,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
    )

    server = ThreadingHTTPServer((args.host, args.port), HiddenLatentsHandler)
    print("=" * 72)
    print("Serve vLLM hidden latents HTTP")
    print("=" * 72)
    print(f"Listening on http://{args.host}:{args.port}")
    print(f"Model path: {args.model_path}")
    print("=" * 72)
    server.serve_forever()


if __name__ == "__main__":
    main()

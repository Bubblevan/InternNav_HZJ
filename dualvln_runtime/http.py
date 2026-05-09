import ipaddress
import logging
import time
from urllib.parse import urlparse

import requests as http_requests
import torch
from flask import Flask, jsonify, request

from dualvln_miniengine.contracts import DualVLNMiniEngineRequest

from .protocol import (
    cleanup_client_shared_memory_handles,
    decode_messages,
    decode_tensor_from_b64,
    encode_messages,
    encode_tensor_to_b64,
    get_image_transport_mode,
)

logger = logging.getLogger(__name__)


class DualVLNSingleVLLMHTTPClient:
    def __init__(self, base_url: str, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.image_transport_mode = get_image_transport_mode()
        parsed = urlparse(self.base_url)
        hostname = parsed.hostname or ""
        disable_env_proxy = hostname in {"localhost", "::1"}
        if not disable_env_proxy:
            try:
                disable_env_proxy = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                disable_env_proxy = False
        self._disable_env_proxy = bool(disable_env_proxy)
        self._session = http_requests.Session()
        if self._disable_env_proxy:
            self._session.trust_env = False
            logger.info(
                "DualVLNSingleVLLMHTTPClient disabling env proxy for local base_url=%s",
                self.base_url,
            )

    def step_s2(
        self,
        messages,
        *,
        max_new_tokens: int = 128,
        target_device=None,
        target_dtype=None,
        return_latents: bool = True,
        return_hidden_states: bool = False,
    ):
        client_total_start = time.perf_counter()
        encode_start = client_total_start
        encode_state = {"shared_memory_handles": []}
        encoded_messages, encode_state = encode_messages(
            messages,
            image_transport_mode=self.image_transport_mode,
        )
        client_encode_messages_ms = (time.perf_counter() - encode_start) * 1000.0
        payload = {
            "messages": encoded_messages,
            "max_new_tokens": int(max_new_tokens),
            "image_transport_mode": encode_state["image_transport_mode"],
            "return_latents": bool(return_latents),
            "return_hidden_states": bool(return_hidden_states),
        }
        data = None
        try:
            http_start = time.perf_counter()
            resp = self._session.post(
                f"{self.base_url}/dualvln/step_s2",
                json=payload,
                timeout=self.timeout,
            )
            client_http_post_ms = (time.perf_counter() - http_start) * 1000.0
            if not resp.ok:
                logger.error(
                    "DualVLN step_s2 HTTP %s from %s/dualvln/step_s2; response body prefix=%r",
                    resp.status_code,
                    self.base_url,
                    resp.text[:500],
                )
            resp.raise_for_status()
            response_json_start = time.perf_counter()
            data = resp.json()
            client_response_json_ms = (time.perf_counter() - response_json_start) * 1000.0
            latents = None
            decode_latents_start = time.perf_counter()
            if data.get("latents") is not None:
                latents = decode_tensor_from_b64(data["latents"])
                if target_dtype is not None:
                    latents = latents.to(dtype=target_dtype)
                if target_device is not None:
                    latents = latents.to(device=target_device)
            client_decode_latents_ms = (time.perf_counter() - decode_latents_start) * 1000.0
            client_total_ms = (time.perf_counter() - client_total_start) * 1000.0
            transport_metrics = dict(data.get("transport_metrics") or {})
            transport_metrics.update(
                {
                    "client_encode_messages_ms": client_encode_messages_ms,
                    "client_http_post_ms": client_http_post_ms,
                    "client_response_json_ms": client_response_json_ms,
                    "client_decode_latents_ms": client_decode_latents_ms,
                    "client_total_ms": client_total_ms,
                    "image_transport_mode": encode_state["image_transport_mode"],
                    "image_transport_count": int(encode_state["image_count"]),
                    "image_transport_payload_bytes": int(encode_state["image_payload_bytes"]),
                }
            )
            runtime_total_ms = ((data.get("runtime_metrics") or {}).get("total_ms"))
            server_total_ms = transport_metrics.get("server_total_ms")
            transport_metrics["client_side_overhead_ms"] = (
                float(max(client_total_ms - server_total_ms, 0.0))
                if server_total_ms is not None
                else None
            )
            transport_metrics["end_to_end_transport_overhead_ms"] = (
                float(max(client_total_ms - runtime_total_ms, 0.0))
                if runtime_total_ms is not None
                else None
            )
            data["latents"] = latents
            data["transport_metrics"] = transport_metrics
            return data
        finally:
            cleanup_client_shared_memory_handles(encode_state["shared_memory_handles"])


def _step_result_to_response_payload(step_result):
    generate = step_result.generate
    latents_result = step_result.latents
    latents_tensor = None if latents_result is None else latents_result.latents
    latents_payload = encode_tensor_to_b64(latents_tensor) if latents_tensor is not None else None
    runtime_metrics = dict(generate.runtime_metrics or {})
    if latents_result is not None and latents_result.runtime_metrics:
        runtime_metrics = dict(latents_result.runtime_metrics)

    return {
        "llm_output": generate.llm_output,
        "prompt_token_ids": generate.prompt_token_ids,
        "generated_token_ids": generate.generated_token_ids,
        "pixel_goal": generate.pixel_goal,
        "latents": latents_payload,
        "runtime_metrics": runtime_metrics,
        "engine_metadata": step_result.engine_metadata,
        "backend_metadata": step_result.backend_metadata,
        "backend_runtime": step_result.backend_runtime,
        "vllm_kv_cache": step_result.vllm_kv_cache,
        "debug_mm": step_result.debug_mm,
    }


def create_dualvln_app(*, engine, served_model_name: str) -> Flask:
    app = Flask(__name__)

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
        return_latents = bool(payload.get("return_latents", True))
        return_hidden_states = bool(payload.get("return_hidden_states", False))

        runner_start = time.perf_counter()
        step_result = engine.step_s2(
            DualVLNMiniEngineRequest(
                external_request_id=f"dualvln-http-{time.time_ns()}",
                messages=messages,
                max_new_tokens=max_new_tokens,
                return_latents=return_latents,
                return_hidden_states=return_hidden_states,
                latent_query_count=int(getattr(engine, "n_query", 0) or 0),
            )
        )
        server_runner_step_s2_ms = (time.perf_counter() - runner_start) * 1000.0

        encode_start = time.perf_counter()
        response_payload = _step_result_to_response_payload(step_result)
        runtime_total_ms = ((response_payload.get("runtime_metrics") or {}).get("total_ms"))
        transport_metrics = {
            "server_request_parse_ms": server_request_parse_ms,
            "server_decode_messages_ms": server_decode_messages_ms,
            "server_runner_step_s2_ms": server_runner_step_s2_ms,
            "server_encode_response_ms": None,
            "server_total_ms": None,
            "server_outer_overhead_ms": None,
            "image_transport_mode": image_transport_mode,
        }
        response_payload["transport_metrics"] = transport_metrics
        transport_metrics["server_encode_response_ms"] = (time.perf_counter() - encode_start) * 1000.0
        transport_metrics["server_total_ms"] = (time.perf_counter() - server_total_start) * 1000.0
        transport_metrics["server_outer_overhead_ms"] = (
            float(max(transport_metrics["server_total_ms"] - runtime_total_ms, 0.0))
            if runtime_total_ms is not None
            else None
        )
        print(
            "[DualVLN MiniEngine] /dualvln/step_s2 "
            f"latency={transport_metrics['server_total_ms'] / 1000.0:.3f}s "
            f"pixel_goal={response_payload['pixel_goal']} "
            f"gen_tokens={len(response_payload['generated_token_ids'])}",
            flush=True,
        )
        return jsonify(response_payload)

    return app

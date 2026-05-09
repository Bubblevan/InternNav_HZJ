import base64
import io
import os
import time
import uuid
from multiprocessing import shared_memory
from typing import Optional
from unittest.mock import patch

import torch
from PIL import Image

DEFAULT_IMAGE_TRANSPORT_MODE = "image_shm"
IMAGE_TRANSPORT_ENV = "INTERNNAV_DUALVLN_IMAGE_TRANSPORT"


def encode_tensor_to_b64(tensor: torch.Tensor) -> str:
    buf = io.BytesIO()
    torch.save(tensor.detach().cpu(), buf)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_tensor_from_b64(payload: str) -> torch.Tensor:
    raw = base64.b64decode(payload.encode("utf-8"))
    return torch.load(io.BytesIO(raw), map_location="cpu")


def encode_pil_image_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_pil_image_from_b64(payload: str) -> Image.Image:
    raw = base64.b64decode(payload.encode("utf-8"))
    return Image.open(io.BytesIO(raw)).convert("RGB")


def get_image_transport_mode() -> str:
    mode = os.environ.get(IMAGE_TRANSPORT_ENV, DEFAULT_IMAGE_TRANSPORT_MODE).strip().lower()
    if mode not in {"base64", "image_shm"}:
        raise ValueError(
            f"Unsupported {IMAGE_TRANSPORT_ENV}={mode!r}; expected 'base64' or 'image_shm'."
        )
    return mode


def _build_image_shm_name() -> str:
    return f"internnav_img_{os.getpid()}_{time.time_ns()}_{uuid.uuid4().hex}"


def _attach_shared_memory_without_tracking(name: str) -> shared_memory.SharedMemory:
    # Python's resource_tracker incorrectly registers attached shared-memory
    # blocks as if the current process owned their lifecycle.
    with patch("multiprocessing.resource_tracker.register", lambda *args, **kwargs: None):
        return shared_memory.SharedMemory(name=name, create=False)


def encode_pil_image_to_shm(image: Image.Image, *, message_index: int, content_index: int):
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    raw = rgb_image.tobytes()
    shm_handle = shared_memory.SharedMemory(
        name=_build_image_shm_name(),
        create=True,
        size=len(raw),
    )
    shm_handle.buf[: len(raw)] = raw
    payload = {
        "type": "image",
        "image_transport": "image_shm",
        "shm_name": shm_handle.name,
        "shape": [height, width, 3],
        "dtype": "uint8",
        "mode": "RGB",
        "nbytes": len(raw),
        "message_index": int(message_index),
        "content_index": int(content_index),
    }
    return payload, shm_handle, len(raw)


def decode_pil_image_from_shm(payload: dict) -> Image.Image:
    shm_name = payload["shm_name"]
    shape = payload["shape"]
    dtype = payload.get("dtype", "uint8")
    mode = payload.get("mode", "RGB")
    nbytes = int(payload["nbytes"])
    if dtype != "uint8":
        raise ValueError(f"Unsupported shared-memory image dtype: {dtype}")
    if mode != "RGB":
        raise ValueError(f"Unsupported shared-memory image mode: {mode}")
    if len(shape) != 3 or int(shape[2]) != 3:
        raise ValueError(f"Unsupported shared-memory image shape: {shape}")

    height = int(shape[0])
    width = int(shape[1])
    expected_nbytes = height * width * 3
    if nbytes != expected_nbytes:
        raise ValueError(
            f"Shared-memory image payload size mismatch: nbytes={nbytes}, expected={expected_nbytes}"
        )

    shm_handle = _attach_shared_memory_without_tracking(shm_name)
    try:
        raw = bytes(shm_handle.buf[:nbytes])
        return Image.frombytes("RGB", (width, height), raw)
    finally:
        shm_handle.close()


def cleanup_client_shared_memory_handles(shared_memory_handles) -> None:
    for shm_handle in shared_memory_handles:
        try:
            shm_handle.close()
        finally:
            try:
                shm_handle.unlink()
            except FileNotFoundError:
                pass


def encode_messages(messages, *, image_transport_mode: Optional[str] = None):
    transport_mode = image_transport_mode or get_image_transport_mode()
    encoded = []
    shared_memory_handles = []
    image_payload_bytes = 0
    image_count = 0
    for message in messages:
        content = []
        for content_index, item in enumerate(message["content"]):
            if item["type"] == "text":
                content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image":
                image_count += 1
                if transport_mode == "base64":
                    content.append(
                        {
                            "type": "image",
                            "image_transport": "base64",
                            "image": encode_pil_image_to_b64(item["image"]),
                        }
                    )
                elif transport_mode == "image_shm":
                    shm_payload, shm_handle, payload_bytes = encode_pil_image_to_shm(
                        item["image"],
                        message_index=len(encoded),
                        content_index=content_index,
                    )
                    content.append(shm_payload)
                    shared_memory_handles.append(shm_handle)
                    image_payload_bytes += payload_bytes
                else:
                    raise ValueError(f"Unsupported image transport mode: {transport_mode}")
            else:
                raise ValueError(f"Unsupported message content type: {item['type']}")
        encoded.append({"role": message["role"], "content": content})
    return encoded, {
        "image_transport_mode": transport_mode,
        "shared_memory_handles": shared_memory_handles,
        "image_count": image_count,
        "image_payload_bytes": image_payload_bytes,
    }


def decode_messages(messages):
    decoded = []
    for message in messages:
        content = []
        for item in message["content"]:
            if item["type"] == "text":
                content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image":
                image_transport = item.get("image_transport", "base64")
                if image_transport == "base64":
                    content.append({"type": "image", "image": decode_pil_image_from_b64(item["image"])})
                elif image_transport == "image_shm":
                    content.append({"type": "image", "image": decode_pil_image_from_shm(item)})
                else:
                    raise ValueError(f"Unsupported image transport type: {image_transport}")
            else:
                raise ValueError(f"Unsupported message content type: {item['type']}")
        decoded.append({"role": message["role"], "content": content})
    return decoded


def extract_images_from_messages(messages):
    images = []
    for message in messages:
        for item in message["content"]:
            if item["type"] == "image":
                images.append(item["image"])
    return images


def to_vllm_chat_messages(messages):
    converted = []
    for message in messages:
        content = []
        for item in message["content"]:
            if item["type"] == "text":
                content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image":
                content.append({"type": "image_pil", "image_pil": item["image"]})
            else:
                raise ValueError(f"Unsupported message content type for vLLM chat: {item['type']}")
        converted.append({"role": message["role"], "content": content})
    return converted

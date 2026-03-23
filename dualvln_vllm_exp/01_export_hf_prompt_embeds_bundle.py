#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from transformers import AutoModelForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_IDS_KEYS = [
    "full_output_ids",
    "output_ids",
    "input_ids",
    "prompt_token_ids",
]
DEFAULT_PIXEL_KEYS = [
    "pixel_values",
    "pixel_values_cpu",
]
DEFAULT_GRID_KEYS = [
    "image_grid_thw",
    "image_grid_thw_cpu",
]


def first_present(d: dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def as_tensor(x: Any, name: str) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        return torch.tensor(x)
    raise TypeError(f"Unsupported type for {name}: {type(x)}")


def squeeze_batch(t: torch.Tensor, expected_rank: Optional[int] = None) -> torch.Tensor:
    if t.ndim >= 1 and t.shape[0] == 1:
        t = t.squeeze(0)
    if expected_rank is not None and t.ndim != expected_rank:
        raise ValueError(f"Expected rank {expected_rank}, got shape {tuple(t.shape)}")
    return t


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    denom = a.norm() * b.norm()
    if denom.item() == 0:
        return float("nan")
    return float(torch.dot(a, b) / denom)


def resolve_embed_tokens_module(model: torch.nn.Module, inner: torch.nn.Module) -> torch.nn.Module:
    if hasattr(inner, "embed_tokens"):
        return inner.embed_tokens
    if hasattr(model, "get_input_embeddings"):
        embed_tokens = model.get_input_embeddings()
        if embed_tokens is not None:
            return embed_tokens
    raise AttributeError("Could not resolve token embedding module from loaded model.")


def resolve_visual_module(model: torch.nn.Module, inner: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "visual"):
        return model.visual
    if hasattr(inner, "visual"):
        return inner.visual
    raise AttributeError("Could not resolve visual module from loaded model.")


def extract_visual_features(visual_outputs: Any) -> torch.Tensor:
    if hasattr(visual_outputs, "pooler_output"):
        visual_outputs = visual_outputs.pooler_output
    if isinstance(visual_outputs, (list, tuple)):
        visual_outputs = torch.cat(list(visual_outputs), dim=0)
    if not isinstance(visual_outputs, torch.Tensor):
        raise TypeError(f"Unsupported visual output type: {type(visual_outputs)}")
    return visual_outputs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export HF-built prompt_embeds bundle for DualVLN/vLLM experiments.")
    p.add_argument("--model-path", required=True)
    p.add_argument("--sample-pt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--traj-token-index", type=int, default=151667)
    p.add_argument("--ids-key", default=None, help="Explicit key inside sample .pt for full_output_ids.")
    p.add_argument("--pixel-key", default=None)
    p.add_argument("--grid-key", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sample = torch.load(args.sample_pt, map_location="cpu")
    if not isinstance(sample, dict):
        raise TypeError(f"Expected dict sample, got {type(sample)}")

    ids_val = sample.get(args.ids_key) if args.ids_key else first_present(sample, DEFAULT_IDS_KEYS)
    pixel_val = sample.get(args.pixel_key) if args.pixel_key else first_present(sample, DEFAULT_PIXEL_KEYS)
    grid_val = sample.get(args.grid_key) if args.grid_key else first_present(sample, DEFAULT_GRID_KEYS)

    if ids_val is None:
        raise KeyError(f"Could not find ids in sample. Available keys: {sorted(sample.keys())}")
    if pixel_val is None:
        raise KeyError(f"Could not find pixel_values in sample. Available keys: {sorted(sample.keys())}")
    if grid_val is None:
        raise KeyError(f"Could not find image_grid_thw in sample. Available keys: {sorted(sample.keys())}")

    full_output_ids = as_tensor(ids_val, "full_output_ids").long()
    full_output_ids = squeeze_batch(full_output_ids)
    if full_output_ids.ndim != 1:
        raise ValueError(f"Expected 1D full_output_ids, got {tuple(full_output_ids.shape)}")

    pixel_values = as_tensor(pixel_val, "pixel_values")
    image_grid_thw = as_tensor(grid_val, "image_grid_thw").long()
    if pixel_values.ndim >= 1 and pixel_values.shape[0] == 1:
        pixel_values = pixel_values.squeeze(0)
    if image_grid_thw.ndim >= 1 and image_grid_thw.shape[0] == 1 and image_grid_thw.ndim > 2:
        image_grid_thw = image_grid_thw.squeeze(0)

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    if not hasattr(model, "generate_latents"):
        raise AttributeError("Loaded model does not expose generate_latents().")
    if not hasattr(model, "get_rope_index"):
        raise AttributeError("Loaded model does not expose get_rope_index().")

    inner = model.get_model() if hasattr(model, "get_model") else getattr(model, "model", None)
    if inner is None:
        raise AttributeError("Could not locate inner model via get_model() or .model")
    if not hasattr(inner, "latent_queries"):
        raise AttributeError("Inner model has no latent_queries")
    embed_tokens = resolve_embed_tokens_module(model, inner)
    visual = resolve_visual_module(model, inner)

    latent_queries = inner.latent_queries
    if latent_queries.ndim == 2:
        latent_queries = latent_queries.unsqueeze(0)
    elif latent_queries.ndim != 3:
        raise ValueError(f"Unexpected latent_queries shape: {tuple(latent_queries.shape)}")
    n_query = latent_queries.shape[1]

    full_output_ids_b = full_output_ids.unsqueeze(0).to(device)
    pixel_values_b = pixel_values.to(device=device, dtype=getattr(visual, "dtype", dtype))
    image_grid_thw_b = image_grid_thw.to(device=device)

    with torch.inference_mode():
        ref_traj_latents = model.generate_latents(
            full_output_ids_b,
            pixel_values_b,
            image_grid_thw_b,
        ).detach().cpu()

        traj_placeholders = torch.full(
            (1, n_query),
            args.traj_token_index,
            dtype=torch.long,
            device=device,
        )
        full_prompt_ids = torch.cat([full_output_ids_b, traj_placeholders], dim=1)

        inputs_embeds = embed_tokens(full_prompt_ids)

        image_embeds = extract_visual_features(visual(pixel_values_b, grid_thw=image_grid_thw_b))
        image_embeds = image_embeds.to(device=device, dtype=inputs_embeds.dtype)

        image_mask = (full_prompt_ids[0] == model.config.image_token_id)
        image_token_count = int(image_mask.sum().item())
        if image_token_count != int(image_embeds.shape[0]):
            raise ValueError(
                f"Image token count mismatch: mask={image_token_count}, image_embeds={tuple(image_embeds.shape)}"
            )
        inputs_embeds[0, image_mask, :] = image_embeds
        inputs_embeds[:, -n_query:, :] = latent_queries.to(device=device, dtype=inputs_embeds.dtype)

        attention_mask = torch.ones_like(full_prompt_ids)
        position_ids, _ = model.get_rope_index(
            full_prompt_ids,
            image_grid_thw=image_grid_thw_b,
            attention_mask=attention_mask,
        )

        manual_outputs = inner(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict=True,
        )
        manual_traj_latents = manual_outputs.hidden_states[-1][:, -n_query:, :].detach().cpu()

    bundle = {
        "prompt_embeds": inputs_embeds[0].detach().cpu().contiguous(),
        "full_prompt_ids": full_prompt_ids[0].detach().cpu().contiguous(),
        "full_output_ids": full_output_ids.detach().cpu().contiguous(),
        "position_ids": position_ids.detach().cpu().contiguous(),
        "attention_mask": attention_mask.detach().cpu().contiguous(),
        "pixel_values": pixel_values.detach().cpu().contiguous(),
        "image_grid_thw": image_grid_thw.detach().cpu().contiguous(),
        "ref_traj_latents": ref_traj_latents.contiguous(),
        "manual_traj_latents": manual_traj_latents.contiguous(),
        "meta": {
            "model_path": args.model_path,
            "sample_pt": args.sample_pt,
            "dtype": args.dtype,
            "device": str(device),
            "traj_token_index": args.traj_token_index,
            "n_query": int(n_query),
            "image_token_count": image_token_count,
            "prompt_embeds_shape": list(inputs_embeds[0].shape),
            "full_prompt_len": int(full_prompt_ids.shape[1]),
            "full_output_len": int(full_output_ids.shape[0]),
        },
    }
    torch.save(bundle, out_path)

    summary = {
        "saved_bundle": str(out_path),
        "prompt_embeds_shape": list(bundle["prompt_embeds"].shape),
        "full_prompt_len": bundle["meta"]["full_prompt_len"],
        "n_query": bundle["meta"]["n_query"],
        "image_token_count": image_token_count,
        "hf_ref_vs_manual": {
            "cosine": cosine_similarity(ref_traj_latents, manual_traj_latents),
            "mean_abs": float((ref_traj_latents - manual_traj_latents).abs().mean()),
            "max_abs": float((ref_traj_latents - manual_traj_latents).abs().max()),
        },
        "available_sample_keys": sorted(sample.keys()),
    }
    summary_path = out_path.with_suffix(out_path.suffix + ".json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

import argparse
import functools
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from safetensors.torch import load_file

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

TRAJ_TOKEN_INDEX = 151667


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Probe vLLM with HF-like generate_latents inputs: visual replacement + "
            "real latent_queries injection + raw hidden-state dump comparison."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sample-pt", required=True)
    parser.add_argument("--hf-model-path", default="checkpoints/InternVLA-N1-DualVLN")
    parser.add_argument("--manifest", default=None, help="Optional replay1 manifest for rebuilding input image metadata")
    parser.add_argument(
        "--base-path",
        default=None,
        help="Optional base path to replace ./logs/ in manifest paths",
    )
    parser.add_argument("--append-traj-tokens", action="store_true")
    parser.add_argument("--traj-token-count", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--limit-mm-per-prompt-image", type=int, default=16)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dump-dir", default=None)
    parser.add_argument("--dump-prefix", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _qualname(obj):
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}.{cls.__name__}"


def _tensor_diff(a, b):
    diff = (a.float() - b.float()).abs()
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def _load_model_config_dict(model_path: str) -> dict:
    config_path = Path(model_path) / "config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def _compute_hf_like_position_ids(
    prompt_token_ids: list[int],
    image_grid_thw: Optional[torch.Tensor],
    model_config: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor(prompt_token_ids, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    vision_config = model_config["vision_config"]
    spatial_merge_size = int(vision_config["spatial_merge_size"])
    tokens_per_second = float(vision_config.get("tokens_per_second", 1.0))
    image_token_id = int(model_config["image_token_id"])
    video_token_id = int(model_config["video_token_id"])
    vision_start_token_id = int(model_config["vision_start_token_id"])

    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(torch.long)

    total_input_ids = input_ids
    position_ids = torch.ones(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
    )
    image_index = 0
    mrope_position_deltas = []

    for i, sample_input_ids in enumerate(total_input_ids):
        sample_input_ids = sample_input_ids[attention_mask[i] == 1]
        vision_start_indices = torch.argwhere(sample_input_ids == vision_start_token_id).squeeze(1)
        vision_tokens = sample_input_ids[vision_start_indices + 1]
        image_nums = int((vision_tokens == image_token_id).sum().item())
        video_nums = int((vision_tokens == video_token_id).sum().item())
        input_tokens = sample_input_ids.tolist()
        llm_pos_ids_list = []
        st = 0
        remain_images = image_nums
        remain_videos = video_nums

        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1

            if ed_image < ed_video:
                if image_grid_thw is None:
                    raise RuntimeError("image_grid_thw is required for multimodal position comparison")
                t, h, w = (
                    int(image_grid_thw[image_index][0].item()),
                    int(image_grid_thw[image_index][1].item()),
                    int(image_grid_thw[image_index][2].item()),
                )
                second_per_grid_t = 0.0
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                raise RuntimeError("Video inputs are not expected in this probe")

            llm_grid_t = t
            llm_grid_h = h // spatial_merge_size
            llm_grid_w = w // spatial_merge_size
            text_len = ed - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            llm_pos_ids_list.append(
                torch.arange(text_len, dtype=torch.long).view(1, -1).expand(3, -1) + st_idx
            )

            range_tensor = torch.arange(llm_grid_t, dtype=torch.float32).view(-1, 1)
            expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
            time_tensor = expanded_range * second_per_grid_t * tokens_per_second
            t_index = time_tensor.long().flatten()
            h_index = (
                torch.arange(llm_grid_h, dtype=torch.long)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w, dtype=torch.long)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len, dtype=torch.long).view(1, -1).expand(3, -1) + st_idx
            )

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., i, attention_mask[i] == 1] = llm_positions
        mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

    delta = torch.tensor(mrope_position_deltas, dtype=torch.long).unsqueeze(1)
    return position_ids, delta


def _load_manifest(path, base_path=None):
    grouped = defaultdict(list)
    old_prefix = "./logs/"
    if base_path is not None:
        base_path = base_path.rstrip("/") + "/"

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            path_fields = [
                "rgb_path",
                "depth_path",
                "lookdown_rgb_path",
                "lookdown_depth_path",
            ]
            for field in path_fields:
                if field in item and base_path is not None and item[field].startswith(old_prefix):
                    item[field] = item[field].replace(old_prefix, base_path)
                if field in item:
                    item[field] = os.path.abspath(item[field])
            grouped[(item["scene_id"], int(item["episode_id"]))].append(item)
    for items in grouped.values():
        items.sort(key=lambda x: (int(x["step_id"]), x["baseline_output"]["output_kind"]))
    return grouped


def _build_input_images_from_sample(sample, manifest_path, base_path=None):
    replay_steps = _load_manifest(manifest_path, base_path)
    key = (sample["scene_id"], int(sample["episode_id"]))
    episode = replay_steps[key]
    target_item = None
    for item in episode:
        if (
            int(item["step_id"]) == int(sample["step_id"])
            and item["baseline_output"]["output_kind"] == sample["baseline_output"]["output_kind"]
        ):
            target_item = item
            break
    if target_item is None:
        raise RuntimeError(
            f"Could not locate manifest item for {sample['scene_id']} / ep {sample['episode_id']} / step {sample['step_id']}"
        )

    input_images = []
    history_indices = sample.get("history_frame_indices") or []
    for history_index in history_indices:
        input_images.append(Image.open(episode[int(history_index)]["rgb_path"]).convert("RGB"))
    input_images.append(Image.open(target_item["rgb_path"]).convert("RGB"))

    if sample.get("is_inferred_lookdown_followup", False):
        input_images.append(Image.open(target_item["lookdown_rgb_path"]).convert("RGB"))

    return input_images


def _load_latent_queries_tensor(hf_model_path: str) -> torch.Tensor:
    index_path = Path(hf_model_path) / "model.safetensors.index.json"
    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index_data["weight_map"]
    tensor_name = "model.latent_queries"
    shard_name = weight_map[tensor_name]
    shard_path = Path(hf_model_path) / shard_name
    state = load_file(str(shard_path), device="cpu")
    latent_queries = state[tensor_name]
    if latent_queries.ndim != 3 or latent_queries.shape[0] != 1:
        raise RuntimeError(
            f"Unexpected latent_queries shape: {tuple(latent_queries.shape)}"
        )
    return latent_queries[0].contiguous()


def _build_hf_like_prompt_embeds(
    model,
    prompt_token_ids,
    pixel_values_cpu,
    image_grid_thw_cpu,
    latent_queries_cpu=None,
):
    device = next(model.parameters()).device
    input_ids = torch.tensor(prompt_token_ids, device=device, dtype=torch.long)

    with torch.inference_mode():
        embeds = model.embed_input_ids(input_ids).clone()

        mm_info = {
            "image_token_id": int(getattr(model.config, "image_token_id")),
            "num_image_tokens_in_prompt": int((input_ids == model.config.image_token_id).sum().item()),
            "num_mm_embeddings": 0,
        }

        if pixel_values_cpu is not None and image_grid_thw_cpu is not None:
            pixel_values = pixel_values_cpu.to(device=device, dtype=model.visual.dtype)
            image_grid_thw = image_grid_thw_cpu.to(device=device)
            multimodal_embeddings = model.embed_multimodal(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
            flat_mm_embeddings = (
                torch.cat(list(multimodal_embeddings), dim=0)
                if multimodal_embeddings
                else None
            )
            if flat_mm_embeddings is not None:
                image_idx = input_ids == model.config.image_token_id
                image_token_count = int(image_idx.sum().item())
                embeds[image_idx] = flat_mm_embeddings[:image_token_count].to(embeds.dtype)
                mm_info["num_mm_embeddings"] = int(flat_mm_embeddings.shape[0])

        if latent_queries_cpu is not None:
            latent_queries = latent_queries_cpu.to(device=device, dtype=embeds.dtype)
            embeds[-latent_queries.shape[0] :] = latent_queries

    return {
        "embeds": embeds.cpu(),
        "info": mm_info,
        "model_type": _qualname(model),
    }


def _set_dump_env(dump_dir: Path, dump_prefix: str):
    os.environ["VLLM_DEBUG_DUMP_DIR"] = str(dump_dir)
    os.environ["VLLM_DEBUG_DUMP_PREFIX"] = dump_prefix
    os.environ["VLLM_DEBUG_DUMP_FULL_TENSORS"] = "1"
    os.environ["VLLM_DEBUG_DUMP_SLICE_ROWS"] = "8"


def _load_dump_records(dump_dir: Path, dump_prefix: str, tag: str):
    records = []
    for path in sorted(dump_dir.glob(f"{dump_prefix}_*_{tag}.pt")):
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception:
            continue
        payload["_path"] = str(path)
        records.append(payload)
    return records


def _pick_record(records, start_ts, end_ts, tensor_key, first_dim):
    candidates = []
    for record in records:
        ts = record.get("ts_ns")
        if ts is None or not (start_ts <= ts <= end_ts):
            continue
        tensor = record.get("tensors", {}).get(tensor_key)
        if not torch.is_tensor(tensor):
            continue
        if tensor.ndim >= 1 and tensor.shape[0] == first_dim:
            candidates.append(record)
    if not candidates:
        return None
    candidates.sort(
        key=lambda r: (
            r["tensors"][tensor_key].numel(),
            r.get("ts_ns", 0),
        )
    )
    return candidates[-1]


def _window_records(records, start_ts, end_ts, tensor_key):
    candidates = []
    for record in records:
        ts = record.get("ts_ns")
        if ts is None or not (start_ts <= ts <= end_ts):
            continue
        tensor = record.get("tensors", {}).get(tensor_key)
        if not torch.is_tensor(tensor):
            continue
        candidates.append(record)
    candidates.sort(key=lambda r: r.get("ts_ns", 0))
    return candidates


def _aggregate_records(records, tensor_key):
    if not records:
        return None
    tensors = [record["tensors"][tensor_key] for record in records]
    if not all(torch.is_tensor(t) and t.ndim == 2 for t in tensors):
        return None
    hidden_size = tensors[0].shape[1]
    if not all(t.shape[1] == hidden_size for t in tensors):
        return None
    return torch.cat(tensors, dim=0)


def _record_diff(a_record, b_record, tensor_key):
    if a_record is None or b_record is None:
        return None
    a = a_record["tensors"][tensor_key]
    b = b_record["tensors"][tensor_key]
    report = _tensor_diff(a, b)
    if a.ndim >= 2:
        report["last4"] = _tensor_diff(a[-4:], b[-4:])
    report["base_path"] = a_record["_path"]
    report["custom_path"] = b_record["_path"]
    return report


def main():
    args = parse_args()

    sample = torch.load(args.sample_pt, map_location="cpu")
    prompt_token_ids = sample["baseline_output_ids"][0].tolist()
    if args.append_traj_tokens:
        prompt_token_ids = prompt_token_ids + [TRAJ_TOKEN_INDEX] * args.traj_token_count

    pixel_values = sample.get("pixel_values")
    image_grid_thw = sample.get("image_grid_thw")
    baseline_latent = sample.get("baseline_latent")
    latent_queries = _load_latent_queries_tensor(args.hf_model_path)
    input_images = None
    if args.manifest:
        input_images = _build_input_images_from_sample(sample, args.manifest, args.base_path)

    output_path = Path(args.output) if args.output else None
    dump_dir = (
        Path(args.dump_dir)
        if args.dump_dir
        else (output_path.parent / f"{output_path.stem}_dump" if output_path else Path("logs/habitat/vllm_generate_latents_probe_dump"))
    )
    dump_prefix = args.dump_prefix or f"generate_latents_probe_{int(time.time())}"
    dump_dir.mkdir(parents=True, exist_ok=True)
    _set_dump_env(dump_dir, dump_prefix)

    from vllm import LLM
    from vllm.inputs.data import EmbedsPrompt

    llm = LLM(
        model=args.model_path,
        runner="pooling",
        convert="embed",
        enable_prompt_embeds=True,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": args.limit_mm_per_prompt_image},
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        disable_log_stats=True,
    )

    base_prompt = llm.apply_model(
        functools.partial(
            _build_hf_like_prompt_embeds,
            prompt_token_ids=prompt_token_ids,
            pixel_values_cpu=pixel_values,
            image_grid_thw_cpu=image_grid_thw,
            latent_queries_cpu=None,
        )
    )[0]
    custom_prompt = llm.apply_model(
        functools.partial(
            _build_hf_like_prompt_embeds,
            prompt_token_ids=prompt_token_ids,
            pixel_values_cpu=pixel_values,
            image_grid_thw_cpu=image_grid_thw,
            latent_queries_cpu=latent_queries,
        )
    )[0]

    base_embeds = base_prompt["embeds"]
    custom_embeds = custom_prompt["embeds"]

    base_start_ts = time.time_ns()
    base_output = llm.encode(
        [
            EmbedsPrompt(
                prompt_embeds=base_embeds,
                prompt_token_ids=prompt_token_ids,
                multi_modal_data={"image": input_images} if input_images is not None else None,
            )
        ],
        pooling_task="token_embed",
        use_tqdm=False,
    )[0]
    base_end_ts = time.time_ns()

    custom_start_ts = time.time_ns()
    custom_output = llm.encode(
        [
            EmbedsPrompt(
                prompt_embeds=custom_embeds,
                prompt_token_ids=prompt_token_ids,
                multi_modal_data={"image": input_images} if input_images is not None else None,
            )
        ],
        pooling_task="token_embed",
        use_tqdm=False,
    )[0]
    custom_end_ts = time.time_ns()

    base_data = base_output.outputs.data.cpu()
    custom_data = custom_output.outputs.data.cpu()

    records_prepare = _load_dump_records(dump_dir, dump_prefix, "gpu_model_runner_prepare_inputs")
    records_post = _load_dump_records(
        dump_dir, dump_prefix, "gpu_model_runner_actual_post_forward"
    )
    records_pool = _load_dump_records(dump_dir, dump_prefix, "gpu_model_runner_pooler_output")

    base_prepare = _pick_record(
        records_prepare,
        base_start_ts,
        base_end_ts,
        "inputs_embeds_gpu",
        len(prompt_token_ids),
    )
    custom_prepare = _pick_record(
        records_prepare,
        custom_start_ts,
        custom_end_ts,
        "inputs_embeds_gpu",
        len(prompt_token_ids),
    )
    base_post = _pick_record(
        records_post,
        base_start_ts,
        base_end_ts,
        "hidden_states",
        len(prompt_token_ids),
    )
    custom_post = _pick_record(
        records_post,
        custom_start_ts,
        custom_end_ts,
        "hidden_states",
        len(prompt_token_ids),
    )
    base_pool = _pick_record(
        records_pool,
        base_start_ts,
        base_end_ts,
        "pooler_output_0",
        len(prompt_token_ids),
    )
    custom_pool = _pick_record(
        records_pool,
        custom_start_ts,
        custom_end_ts,
        "pooler_output_0",
        len(prompt_token_ids),
    )
    base_post_records = _window_records(
        records_post,
        base_start_ts,
        base_end_ts,
        "hidden_states",
    )
    custom_post_records = _window_records(
        records_post,
        custom_start_ts,
        custom_end_ts,
        "hidden_states",
    )
    base_post_agg = _aggregate_records(base_post_records, "hidden_states")
    custom_post_agg = _aggregate_records(custom_post_records, "hidden_states")
    qwen_config = _load_model_config_dict(args.model_path)
    hf_position_ids, hf_mrope_delta = _compute_hf_like_position_ids(
        prompt_token_ids,
        image_grid_thw,
        qwen_config,
    )
    base_positions_record = _pick_record(
        records_prepare,
        base_start_ts,
        base_end_ts,
        "positions_gpu",
        3,
    )
    custom_positions_record = _pick_record(
        records_prepare,
        custom_start_ts,
        custom_end_ts,
        "positions_gpu",
        3,
    )

    report = {
        "model_path": args.model_path,
        "hf_model_path": args.hf_model_path,
        "sample_pt": args.sample_pt,
        "requested_prompt_length": len(prompt_token_ids),
        "append_traj_tokens": bool(args.append_traj_tokens),
        "traj_token_count": int(args.traj_token_count if args.append_traj_tokens else 0),
        "dump_dir": str(dump_dir),
        "dump_prefix": dump_prefix,
        "timing_windows": {
            "base_start_ts": int(base_start_ts),
            "base_end_ts": int(base_end_ts),
            "custom_start_ts": int(custom_start_ts),
            "custom_end_ts": int(custom_end_ts),
        },
        "embedding_source": {
            "base_model_type": base_prompt["model_type"],
            "custom_model_type": custom_prompt["model_type"],
            "hidden_size": int(base_embeds.shape[-1]),
            "dtype": str(base_embeds.dtype),
        },
        "multimodal_info": {
            "base": base_prompt["info"],
            "custom": custom_prompt["info"],
        },
        "input_image_metadata": {
            "used_manifest": bool(args.manifest),
            "manifest": args.manifest,
            "num_input_images": len(input_images) if input_images is not None else None,
        },
        "input_embed_diff": {
            **_tensor_diff(base_embeds, custom_embeds),
            "last4": _tensor_diff(base_embeds[-latent_queries.shape[0] :], custom_embeds[-latent_queries.shape[0] :]),
        },
        "token_embed_diff": {
            **_tensor_diff(base_data, custom_data),
            "last4": _tensor_diff(base_data[-latent_queries.shape[0] :], custom_data[-latent_queries.shape[0] :]),
        },
        "dump_diffs": {
            "prepare_inputs": _record_diff(base_prepare, custom_prepare, "inputs_embeds_gpu"),
            "positions": _record_diff(base_positions_record, custom_positions_record, "positions_gpu"),
            "post_forward_hidden_states": (
                {
                    **_tensor_diff(base_post_agg, custom_post_agg),
                    "last4": _tensor_diff(
                        base_post_agg[-latent_queries.shape[0] :],
                        custom_post_agg[-latent_queries.shape[0] :],
                    ),
                    "base_num_chunks": len(base_post_records),
                    "custom_num_chunks": len(custom_post_records),
                    "base_total_tokens": int(base_post_agg.shape[0]),
                    "custom_total_tokens": int(custom_post_agg.shape[0]),
                    "base_last_path": base_post_records[-1]["_path"] if base_post_records else None,
                    "custom_last_path": custom_post_records[-1]["_path"] if custom_post_records else None,
                }
                if base_post_agg is not None and custom_post_agg is not None
                else None
            ),
            "pooler_output": _record_diff(base_pool, custom_pool, "pooler_output_0"),
        },
        "hf_position_compare": (
            {
                **_tensor_diff(
                    custom_positions_record["tensors"]["positions_gpu"],
                    hf_position_ids.squeeze(1),
                ),
                "last4": _tensor_diff(
                    custom_positions_record["tensors"]["positions_gpu"][:, -latent_queries.shape[0] :],
                    hf_position_ids[:, :, -latent_queries.shape[0] :].squeeze(1),
                ),
                "vllm_positions_shape": list(custom_positions_record["tensors"]["positions_gpu"].shape),
                "hf_position_ids_shape": list(hf_position_ids.shape),
                "hf_mrope_delta": hf_mrope_delta.flatten().tolist(),
                "custom_positions_path": custom_positions_record["_path"],
            }
            if custom_positions_record is not None
            else None
        ),
        "hf_baseline_compare": None,
    }

    if custom_post_agg is not None and baseline_latent is not None:
        custom_hidden_last4 = custom_post_agg[-latent_queries.shape[0] :]
        hf_baseline_last4 = baseline_latent[0].cpu()
        report["hf_baseline_compare"] = {
            "vllm_custom_hidden_last4_vs_hf_baseline_latent": _tensor_diff(
                custom_hidden_last4, hf_baseline_last4
            ),
            "vllm_custom_hidden_last4_shape": list(custom_hidden_last4.shape),
            "hf_baseline_latent_shape": list(hf_baseline_last4.shape),
        }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 72)
    print("Probe vLLM generate_latents hidden states")
    print("=" * 72)
    print(f"Model path: {args.model_path}")
    print(f"HF model path: {args.hf_model_path}")
    print(f"Requested prompt length: {len(prompt_token_ids)}")
    print(f"Input-embed last-4 max abs diff: {report['input_embed_diff']['last4']['max_abs_diff']:.6f}")
    print(f"Token-embed last-4 max abs diff: {report['token_embed_diff']['last4']['max_abs_diff']:.6f}")
    if report["dump_diffs"]["post_forward_hidden_states"] is not None:
        print(
            "Hidden-state last-4 max abs diff: "
            f"{report['dump_diffs']['post_forward_hidden_states']['last4']['max_abs_diff']:.6f}"
        )
    else:
        print("Hidden-state dump comparison unavailable")
    if report["hf_position_compare"] is not None:
        print(
            "vLLM positions vs HF position_ids max abs diff: "
            f"{report['hf_position_compare']['max_abs_diff']:.6f}"
        )
    if report["hf_baseline_compare"] is not None:
        print(
            "vLLM custom hidden last-4 vs HF baseline latent max abs diff: "
            f"{report['hf_baseline_compare']['vllm_custom_hidden_last4_vs_hf_baseline_latent']['max_abs_diff']:.6f}"
        )
    if output_path:
        print(f"Saved JSON summary to {output_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

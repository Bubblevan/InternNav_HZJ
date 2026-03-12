import argparse
import json
import os
import statistics
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze prefix reuse opportunities from replay_v2 manifests.")
    parser.add_argument("--manifest", required=True, help="Replay_v2 manifest jsonl path")
    parser.add_argument("--model-path", required=True, help="DualVLN checkpoint path for AutoProcessor")
    parser.add_argument("--output", required=True, help="Summary json output path")
    parser.add_argument("--details-output", default=None, help="Optional per-record jsonl output path")
    parser.add_argument("--base-path", default=None, help="Optional base path to replace ./logs/ in manifest paths")
    parser.add_argument("--max-records", type=int, default=None, help="Optional cap on analyzed decision records")
    parser.add_argument(
        "--include-continuations",
        action="store_true",
        help="Analyze all records with decision context, not only is_new_s2_decision=true records",
    )
    return parser.parse_args()


def default_details_output(output_path):
    stem, _ = os.path.splitext(output_path)
    return f"{stem}.jsonl"


def percentile(values, q):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(values):
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def rewrite_path(path, base_path):
    if base_path is None:
        return os.path.abspath(path)
    base_path = base_path.rstrip("/") + "/"
    if path.startswith("./logs/"):
        path = path.replace("./logs/", base_path, 1)
    return os.path.abspath(path)


def load_records(path, base_path=None):
    records = []
    existing_rgb_paths = set()
    rgb_basename_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            item["rgb_path"] = rewrite_path(item["rgb_path"], base_path)
            if "decision_input_image_paths" in item:
                item["decision_input_image_paths"] = [rewrite_path(p, base_path) for p in item["decision_input_image_paths"]]
            existing_rgb_paths.add(item["rgb_path"])
            rgb_basename_map[os.path.basename(item["rgb_path"]).lower()] = item["rgb_path"]
            records.append(item)
    records.sort(key=lambda x: (x["scene_id"], int(x["episode_id"]), int(x.get("record_id", 0))))
    return records, existing_rgb_paths, rgb_basename_map


def load_images(paths, existing_rgb_paths, rgb_basename_map):
    images = []
    missing_paths = []
    for path in paths:
        resolved = path if os.path.exists(path) else None
        if resolved is None and path in existing_rgb_paths:
            resolved = path
        if resolved is None:
            resolved = rgb_basename_map.get(os.path.basename(path).lower())
        if resolved is None:
            missing_paths.append(path)
            continue
        images.append(Image.open(resolved).convert("RGB"))
    return images, missing_paths


def count_image_tokens(image_grid_thw):
    if image_grid_thw is None:
        return 0
    if isinstance(image_grid_thw, torch.Tensor):
        if image_grid_thw.ndim == 1:
            image_grid_thw = image_grid_thw.unsqueeze(0)
        return int(torch.prod(image_grid_thw.to(torch.int64), dim=-1).sum().item())
    total = 0
    for thw in image_grid_thw:
        tensor = thw if isinstance(thw, torch.Tensor) else torch.as_tensor(thw)
        total += int(torch.prod(tensor.to(torch.int64)).item())
    return total


def find_first_marker_index(input_ids, marker_ids):
    for idx, token_id in enumerate(input_ids.tolist()):
        if token_id in marker_ids:
            return idx
    return len(input_ids)


def common_prefix_len(lhs, rhs):
    prefix = 0
    limit = min(len(lhs), len(rhs))
    while prefix < limit and lhs[prefix] == rhs[prefix]:
        prefix += 1
    return prefix


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    details_output = args.details_output or default_details_output(args.output)

    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    marker_ids = {vision_start_id, image_pad_id}

    records, existing_rgb_paths, rgb_basename_map = load_records(args.manifest, args.base_path)
    if not args.include_continuations:
        records = [r for r in records if r.get("is_new_s2_decision")]
    if args.max_records is not None:
        records = records[: args.max_records]

    full_input_tokens = []
    chat_text_tokens = []
    image_token_counts = []
    static_prefix_tokens = []
    dynamic_suffix_tokens = []
    static_prefix_ratio_full = []
    static_prefix_ratio_text = []
    prefix_vs_prev_episode = []
    prefix_vs_prev_decision = []
    group_metrics = defaultdict(lambda: defaultdict(list))
    prev_text_ids_by_episode = {}
    prev_text_ids_by_decision_group = {}
    records_with_missing_images = 0
    total_missing_images = 0

    with open(details_output, "w", encoding="utf-8") as detail_f:
        for item in tqdm(records, desc="Prefix Reuse Analysis", dynamic_ncols=True):
            input_image_paths = item.get("decision_input_image_paths", [])
            if not input_image_paths or "decision_chat_text" not in item:
                continue

            images, missing_paths = load_images(input_image_paths, existing_rgb_paths, rgb_basename_map)
            if missing_paths:
                records_with_missing_images += 1
                total_missing_images += len(missing_paths)
                detail_f.write(
                    json.dumps(
                        {
                            "scene_id": item["scene_id"],
                            "episode_id": item["episode_id"],
                            "record_id": item.get("record_id"),
                            "step_id": item["step_id"],
                            "decision_id": item.get("decision_id"),
                            "is_lookdown_followup": bool(item.get("is_lookdown_followup", False)),
                            "action_source": item.get("action_source"),
                            "decision_input_image_count": len(input_image_paths),
                            "loaded_decision_image_count": len(images),
                            "missing_decision_image_count": len(missing_paths),
                            "missing_decision_image_paths": missing_paths,
                            "skipped_reason": "missing_decision_images",
                        }
                    )
                    + "\n"
                )
                for image in images:
                    image.close()
                continue
            if not images:
                continue

            inputs = processor(
                text=[item["decision_chat_text"]],
                images=images,
                return_tensors="pt",
            )
            for image in images:
                image.close()

            full_ids = inputs.input_ids[0]
            full_token_count = int(full_ids.shape[0])
            image_token_count = count_image_tokens(inputs.image_grid_thw)
            first_marker_idx = find_first_marker_index(full_ids, marker_ids)

            text_ids = tokenizer(item["decision_chat_text"], add_special_tokens=False)["input_ids"]
            text_token_count = int(len(text_ids))
            static_prefix_token_count = int(first_marker_idx)
            dynamic_suffix_token_count = int(full_token_count - static_prefix_token_count)

            episode_key = (item["scene_id"], int(item["episode_id"]))
            decision_group_key = (
                item["scene_id"],
                int(item["episode_id"]),
                int(item.get("decision_id", -1)),
                bool(item.get("is_lookdown_followup", False)),
            )

            prefix_episode = 0
            if episode_key in prev_text_ids_by_episode:
                prefix_episode = common_prefix_len(prev_text_ids_by_episode[episode_key], text_ids)
                prefix_vs_prev_episode.append(prefix_episode)
            prev_text_ids_by_episode[episode_key] = text_ids

            prefix_decision = 0
            if decision_group_key in prev_text_ids_by_decision_group:
                prefix_decision = common_prefix_len(prev_text_ids_by_decision_group[decision_group_key], text_ids)
                prefix_vs_prev_decision.append(prefix_decision)
            prev_text_ids_by_decision_group[decision_group_key] = text_ids

            full_input_tokens.append(full_token_count)
            chat_text_tokens.append(text_token_count)
            image_token_counts.append(image_token_count)
            static_prefix_tokens.append(static_prefix_token_count)
            dynamic_suffix_tokens.append(dynamic_suffix_token_count)
            static_prefix_ratio_full.append(static_prefix_token_count / max(full_token_count, 1))
            static_prefix_ratio_text.append(static_prefix_token_count / max(text_token_count, 1))

            group_name = "lookdown_followup" if item.get("is_lookdown_followup") else "normal_decision"
            group_metrics[group_name]["full_input_tokens"].append(full_token_count)
            group_metrics[group_name]["image_token_count"].append(image_token_count)
            group_metrics[group_name]["static_prefix_tokens"].append(static_prefix_token_count)
            group_metrics[group_name]["static_prefix_ratio_full"].append(static_prefix_token_count / max(full_token_count, 1))

            detail_f.write(
                json.dumps(
                    {
                        "scene_id": item["scene_id"],
                        "episode_id": item["episode_id"],
                        "record_id": item.get("record_id"),
                        "step_id": item["step_id"],
                        "decision_id": item.get("decision_id"),
                        "is_lookdown_followup": bool(item.get("is_lookdown_followup", False)),
                        "action_source": item.get("action_source"),
                        "decision_input_image_count": len(input_image_paths),
                        "loaded_decision_image_count": len(images),
                        "missing_decision_image_count": len(missing_paths),
                        "missing_decision_image_paths": missing_paths,
                        "full_input_tokens": full_token_count,
                        "chat_text_tokens": text_token_count,
                        "image_token_count": image_token_count,
                        "static_prefix_tokens": static_prefix_token_count,
                        "dynamic_suffix_tokens": dynamic_suffix_token_count,
                        "static_prefix_ratio_full": static_prefix_token_count / max(full_token_count, 1),
                        "static_prefix_ratio_text": static_prefix_token_count / max(text_token_count, 1),
                        "common_prefix_vs_prev_episode": prefix_episode,
                        "common_prefix_vs_prev_decision_group": prefix_decision,
                    }
                )
                + "\n"
            )

    group_summary = {}
    for group_name, metrics in group_metrics.items():
        group_summary[group_name] = {
            key: summarize(values) for key, values in metrics.items()
        }

    summary = {
        "metadata": {
            "manifest": args.manifest,
            "model_path": args.model_path,
            "include_continuations": bool(args.include_continuations),
            "num_records": len(records),
            "details_output": details_output,
        },
        "prefix_reuse": {
            "full_input_tokens": summarize(full_input_tokens),
            "chat_text_tokens": summarize(chat_text_tokens),
            "image_token_count": summarize(image_token_counts),
            "static_prefix_tokens": summarize(static_prefix_tokens),
            "dynamic_suffix_tokens": summarize(dynamic_suffix_tokens),
            "static_prefix_ratio_full": summarize(static_prefix_ratio_full),
            "static_prefix_ratio_text": summarize(static_prefix_ratio_text),
            "common_prefix_vs_prev_episode": summarize(prefix_vs_prev_episode),
            "common_prefix_vs_prev_decision_group": summarize(prefix_vs_prev_decision),
        },
        "data_quality": {
            "records_with_missing_images": int(records_with_missing_images),
            "total_missing_images": int(total_missing_images),
        },
        "groups": group_summary,
        "interpretation": {
            "meaning_of_static_prefix_ratio_full": (
                "Share of actual multimodal input tokens that sit before the first vision marker. "
                "This is the upper bound for text-only prefix reuse without reusing image tokens."
            ),
            "meaning_of_static_prefix_ratio_text": (
                "Share of chat-text tokens that sit before the first vision marker. "
                "This estimates how much of the textual prompt is static before vision input."
            ),
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

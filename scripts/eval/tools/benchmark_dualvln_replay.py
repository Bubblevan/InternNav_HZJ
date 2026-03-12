import argparse
import json
import os
import re
import statistics
import time
from collections import OrderedDict, defaultdict
from enum import IntEnum

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from internnav.habitat_extensions.vln.utils import preprocess_depth_image_v2
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.utils.vln_utils import split_and_clean, traj_to_actions


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


CONJUNCTIONS = [
    "you can see ",
    "in front of you is ",
    "there is ",
    "you can spot ",
    "you are toward the ",
    "ahead of you is ",
    "in your sight is ",
]
PROMPT_VARIANTS = {
    "full": (
        "You are an autonomous navigation assistant. Your task is to <instruction>. "
        "Where should you go next to stay on track? Please output the next waypoint's "
        "coordinates in the image. Please output STOP when you have successfully completed the task."
    ),
    "short": (
        "Your task is to <instruction>. "
        "Output the next waypoint coordinates in the image, or output STOP if the task is complete."
    ),
    "minimal": "<instruction> Output next waypoint coordinates or STOP.",
}
DEFAULT_IMAGE_TOKEN = "<image>"
ACTIONS2IDX = OrderedDict({"STOP": [0], "↑": [1], "←": [2], "→": [3], "↓": [5]})


class action_code(IntEnum):
    STOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    LOOKUP = 4
    LOOKDOWN = 5


def parse_args():
    parser = argparse.ArgumentParser(description="Offline DualVLN replay benchmark.")
    parser.add_argument("--manifest", required=True, help="Replay manifest jsonl path")
    parser.add_argument("--model-path", required=True, help="DualVLN checkpoint path")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument(
        "--attn-backend",
        choices=["flash_attention_2", "sdpa", "eager"],
        default="flash_attention_2",
        help="Attention backend passed to from_pretrained",
    )
    parser.add_argument(
        "--processor-use-fast",
        choices=["auto", "true", "false"],
        default="auto",
        help="Forward use_fast to AutoProcessor when not set to auto",
    )
    parser.add_argument("--num-history", type=int, default=8, help="History length")
    parser.add_argument(
        "--prompt-variant",
        choices=sorted(PROMPT_VARIANTS.keys()),
        default="full",
        help="Prompt template variant for ablations",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Generation length cap for S2")
    parser.add_argument(
        "--ignore-manifest-history",
        action="store_true",
        help="Ignore saved history_frame_indices in the manifest and rebuild history from --num-history",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on replay steps")
    parser.add_argument("--output", required=True, help="Summary json output path")
    parser.add_argument(
        "--details-output",
        default=None,
        help="Optional per-step jsonl output path. Defaults to <output stem>.jsonl",
    )
    parser.add_argument(
        "--base-path",
        default=None,
        help="Optional base path to replace ./logs/ in manifest paths (e.g., /root/backup/InternNav/logs_example)",
    )
    parser.add_argument("--verbose-every", type=int, default=20, help="Print one timing line every N replay steps")
    return parser.parse_args()


def parse_actions(output):
    action_patterns = "|".join(re.escape(action) for action in ACTIONS2IDX)
    regex = re.compile(action_patterns)
    matches = regex.findall(output)
    actions = [ACTIONS2IDX[match] for match in matches]
    return [item for action in actions for item in action]


def percentile(values, q):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_latencies(values):
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def summarize_numeric(values):
    return summarize_latencies(values)


def count_image_tokens(image_grid_thw):
    if image_grid_thw is None:
        return 0
    if isinstance(image_grid_thw, torch.Tensor):
        if image_grid_thw.ndim == 1:
            image_grid_thw = image_grid_thw.unsqueeze(0)
        counts = torch.prod(image_grid_thw.to(torch.int64), dim=-1)
        return int(counts.sum().item())
    total = 0
    for thw in image_grid_thw:
        tensor = thw if isinstance(thw, torch.Tensor) else torch.as_tensor(thw)
        total += int(torch.prod(tensor.to(torch.int64)).item())
    return total


def load_manifest(path, base_path=None):
    """Load manifest and optionally replace ./logs/ with a user-provided base path."""
    grouped = defaultdict(list)
    old_prefix = "./logs/"
    if base_path is not None:
        base_path = base_path.rstrip("/") + "/"
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            path_fields = ["rgb_path", "depth_path", "lookdown_rgb_path", "lookdown_depth_path"]
            for field in path_fields:
                if field in item and base_path is not None and item[field].startswith(old_prefix):
                    item[field] = item[field].replace(old_prefix, base_path)
                if field in item:
                    item[field] = os.path.abspath(item[field])
            grouped[(item["scene_id"], int(item["episode_id"]))].append(item)
    for items in grouped.values():
        items.sort(key=lambda x: x["step_id"])
    return grouped


def count_total_steps(replay_steps):
    return sum(len(v) for v in replay_steps.values())


def build_messages(instruction, episode_steps, current_index, num_history, prompt_variant, ignore_manifest_history=False):
    sources = [{"from": "human", "value": PROMPT_VARIANTS[prompt_variant]}, {"from": "gpt", "value": ""}]
    sources[0]["value"] = sources[0]["value"].replace("<instruction>.", instruction[:-1])
    sources[0]["value"] = sources[0]["value"].replace("<instruction>", instruction.strip())

    manifest_history = None if ignore_manifest_history else episode_steps[current_index].get("history_frame_indices", None)
    if manifest_history:
        history_indices = sorted(manifest_history)
    elif current_index == 0 or num_history == 0:
        history_indices = []
    else:
        history_indices = np.unique(np.linspace(0, current_index - 1, num_history, dtype=np.int32)).tolist()

    if history_indices:
        placeholder = (DEFAULT_IMAGE_TOKEN + "\n") * len(history_indices)
        sources[0]["value"] += f" These are your historical observations: {placeholder}."

    history_indices = sorted(history_indices)
    input_images = [Image.open(episode_steps[i]["rgb_path"]).convert("RGB") for i in history_indices]
    input_images.append(Image.open(episode_steps[current_index]["rgb_path"]).convert("RGB"))
    input_img_id = 0

    prompt = CONJUNCTIONS[0] + DEFAULT_IMAGE_TOKEN
    sources[0]["value"] += f" {prompt}."
    parts = split_and_clean(sources[0]["value"])
    content = []
    for part in parts:
        if part == DEFAULT_IMAGE_TOKEN:
            content.append({"type": "image", "image": input_images[input_img_id]})
            input_img_id += 1
        else:
            content.append({"type": "text", "text": part})
    messages = [{"role": "user", "content": content}]
    return messages, input_images, history_indices


def load_lookdown_depth(path):
    lookdown_depth = np.load(path)
    lookdown_depth, _ = preprocess_depth_image_v2(
        Image.fromarray((lookdown_depth[:, :, 0] * 1000).astype(np.uint16), mode="I;16"),
        do_depth_scale=True,
        depth_scale=1000,
        target_height=224,
        target_width=224,
    )
    lookdown_depth = torch.as_tensor(np.ascontiguousarray(lookdown_depth)).float()
    lookdown_depth[lookdown_depth > 5.0] = 5.0
    return lookdown_depth


def default_details_output(output_path):
    stem, _ = os.path.splitext(output_path)
    return f"{stem}.jsonl"


def main():
    args = parse_args()
    replay_steps = load_manifest(args.manifest, args.base_path)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    details_output = args.details_output or default_details_output(args.output)
    total_planned_steps = count_total_steps(replay_steps)

    load_start = time.perf_counter()
    processor_kwargs = {}
    if args.processor_use_fast != "auto":
        processor_kwargs["use_fast"] = args.processor_use_fast == "true"
    processor = AutoProcessor.from_pretrained(args.model_path, **processor_kwargs)
    processor.tokenizer.padding_side = "left"
    model = InternVLAN1ForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_backend,
        device_map={"": torch.device(args.device)},
    )
    model.eval()
    cold_start_seconds = time.perf_counter() - load_start

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    total_latencies = []
    s2_generate_latencies = []
    s2_latent_latencies = []
    s1_generate_latencies = []
    discrete_total_latencies = []
    pixel_total_latencies = []
    input_token_counts = []
    input_image_counts = []
    history_image_counts = []
    prompt_char_counts = []
    image_token_counts = []
    tokens_per_second = []
    gen_lengths = []
    action_matches = 0
    action_total = 0
    discrete_action_matches = 0
    discrete_action_total = 0
    pixel_action_matches = 0
    pixel_action_total = 0
    output_kind_matches = 0
    output_kind_total = 0
    text_exact_matches = 0
    text_exact_total = 0
    pixel_goal_l1_errors = []
    pixel_goal_l2_errors = []
    records = []

    with open(details_output, "w", encoding="utf-8") as detail_f:
        progress = tqdm(total=total_planned_steps, desc="Replay Benchmark", dynamic_ncols=True)
        for _, episode in replay_steps.items():
            for step_index, item in enumerate(episode):
                if args.max_steps is not None and len(records) >= args.max_steps:
                    break

                baseline = item["baseline_output"]
                baseline_kind = baseline["output_kind"]

                messages, input_images, history_indices = build_messages(
                    item["instruction"],
                    episode,
                    step_index,
                    args.num_history,
                    args.prompt_variant,
                    args.ignore_manifest_history,
                )
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=input_images, return_tensors="pt").to(model.device)
                input_token_count = int(inputs.input_ids.shape[1])
                input_image_count = int(len(input_images))
                history_image_count = int(len(history_indices))
                prompt_char_count = int(len(text))
                image_token_count = count_image_tokens(getattr(inputs, "image_grid_thw", None))

                input_token_counts.append(input_token_count)
                input_image_counts.append(input_image_count)
                history_image_counts.append(history_image_count)
                prompt_char_counts.append(prompt_char_count)
                image_token_counts.append(image_token_count)

                step_start = time.perf_counter()
                s2_start = time.perf_counter()
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        past_key_values=None,
                        return_dict_in_generate=True,
                    ).sequences
                s2_generate_seconds = time.perf_counter() - s2_start
                s2_generate_latencies.append(s2_generate_seconds)

                output_text = processor.tokenizer.decode(
                    output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                )
                gen_token_count = int(output_ids.shape[1] - inputs.input_ids.shape[1])
                gen_lengths.append(gen_token_count)
                tokens_per_second.append(gen_token_count / max(s2_generate_seconds, 1e-6))

                predicted_action = None
                predicted_pixel_goal = None
                predicted_kind = "discrete"
                s2_latent_seconds = 0.0
                s1_generate_seconds = 0.0

                if bool(re.search(r"\d", output_text)):
                    predicted_kind = "pixel_goal"
                    coord = [int(c) for c in re.findall(r"\d+", output_text)]
                    if len(coord) >= 2:
                        predicted_pixel_goal = [int(coord[1]), int(coord[0])]

                    lookdown_rgb = Image.open(item["lookdown_rgb_path"]).convert("RGB")
                    lookdown_depth = load_lookdown_depth(item["lookdown_depth_path"])

                    pixel_values = inputs.pixel_values
                    image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)

                    latent_start = time.perf_counter()
                    with torch.no_grad():
                        traj_latents = model.generate_latents(output_ids, pixel_values, image_grid_thw)
                    s2_latent_seconds = time.perf_counter() - latent_start
                    s2_latent_latencies.append(s2_latent_seconds)

                    image_dp = torch.tensor(np.array(lookdown_rgb.resize((224, 224)))).to(torch.bfloat16) / 255
                    images_dp = torch.stack([image_dp, image_dp]).unsqueeze(0).to(model.device)
                    depth_dp = lookdown_depth.unsqueeze(-1).to(torch.bfloat16)
                    depths_dp = torch.stack([depth_dp, depth_dp]).unsqueeze(0).to(model.device)

                    s1_start = time.perf_counter()
                    with torch.no_grad():
                        dp_actions = model.generate_traj(traj_latents, images_dp, depths_dp)
                    s1_generate_seconds = time.perf_counter() - s1_start
                    s1_generate_latencies.append(s1_generate_seconds)

                    action_list = traj_to_actions(dp_actions)
                    predicted_action = int(action_list[0]) if action_list else int(action_code.STOP)
                else:
                    actions = parse_actions(output_text)
                    predicted_action = int(actions[0]) if actions else int(action_code.STOP)

                total_step_seconds = time.perf_counter() - step_start
                total_latencies.append(total_step_seconds)
                if predicted_kind == "discrete":
                    discrete_total_latencies.append(total_step_seconds)
                else:
                    pixel_total_latencies.append(total_step_seconds)

                baseline_action = int(baseline["action"])
                action_total += 1
                if predicted_action == baseline_action:
                    action_matches += 1

                output_kind_total += 1
                if predicted_kind == baseline_kind:
                    output_kind_matches += 1

                text_exact_total += 1
                if output_text.strip() == baseline["llm_output"].strip():
                    text_exact_matches += 1

                if baseline_kind == "discrete":
                    discrete_action_total += 1
                    if predicted_action == baseline_action:
                        discrete_action_matches += 1
                else:
                    pixel_action_total += 1
                    if predicted_action == baseline_action:
                        pixel_action_matches += 1

                if predicted_pixel_goal is not None and baseline["pixel_goal"] is not None:
                    base_goal = baseline["pixel_goal"]
                    dx = float(predicted_pixel_goal[0] - base_goal[0])
                    dy = float(predicted_pixel_goal[1] - base_goal[1])
                    pixel_goal_l1_errors.append(abs(dx) + abs(dy))
                    pixel_goal_l2_errors.append(float(np.sqrt(dx * dx + dy * dy)))

                record = {
                    "scene_id": item["scene_id"],
                    "episode_id": item["episode_id"],
                    "step_id": item["step_id"],
                    "history_frame_indices": history_indices,
                    "baseline_output_kind": baseline_kind,
                    "predicted_output_kind": predicted_kind,
                    "total_latency_seconds": total_step_seconds,
                    "s2_generate_seconds": s2_generate_seconds,
                    "s2_latent_seconds": s2_latent_seconds,
                    "s1_generate_seconds": s1_generate_seconds,
                    "generated_tokens": gen_token_count,
                    "tokens_per_second": gen_token_count / max(s2_generate_seconds, 1e-6),
                    "input_token_count": input_token_count,
                    "input_image_count": input_image_count,
                    "history_image_count": history_image_count,
                    "prompt_char_count": prompt_char_count,
                    "image_token_count": image_token_count,
                    "output_text": output_text,
                    "baseline_text": baseline["llm_output"],
                    "predicted_action": predicted_action,
                    "baseline_action": baseline_action,
                    "predicted_pixel_goal": predicted_pixel_goal,
                    "baseline_pixel_goal": baseline["pixel_goal"],
                }
                records.append(record)
                detail_f.write(json.dumps(record) + "\n")
                progress.update(1)
                if args.verbose_every > 0 and len(records) % args.verbose_every == 0:
                    print(
                        f"[step {len(records)}] kind={predicted_kind} total={total_step_seconds:.3f}s "
                        f"s2={s2_generate_seconds:.3f}s latent={s2_latent_seconds:.3f}s s1={s1_generate_seconds:.3f}s",
                        flush=True,
                    )

            if args.max_steps is not None and len(records) >= args.max_steps:
                break
        progress.close()

    summary = {
        "metadata": {
            "manifest": args.manifest,
            "model_path": args.model_path,
            "device": args.device,
            "attn_backend": args.attn_backend,
            "processor_use_fast": args.processor_use_fast,
            "num_history": args.num_history,
            "prompt_variant": args.prompt_variant,
            "max_new_tokens": args.max_new_tokens,
            "ignore_manifest_history": bool(args.ignore_manifest_history),
            "num_steps": len(records),
            "num_episodes": len(replay_steps),
            "details_output": details_output,
        },
        "startup": {
            "cold_start_load_seconds": cold_start_seconds,
            "gpu_peak_memory_mb": (
                float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
            ),
        },
        "latency": {
            "total_step": summarize_latencies(total_latencies),
            "s2_generate": summarize_latencies(s2_generate_latencies),
            "s2_latent": summarize_latencies(s2_latent_latencies),
            "s1_generate": summarize_latencies(s1_generate_latencies),
            "discrete_total_step": summarize_latencies(discrete_total_latencies),
            "pixel_goal_total_step": summarize_latencies(pixel_total_latencies),
        },
        "generation": {
            "generated_tokens_mean": statistics.mean(gen_lengths) if gen_lengths else 0.0,
            "generated_tokens_p50": percentile(gen_lengths, 50),
            "generated_tokens_p95": percentile(gen_lengths, 95),
            "tokens_per_second_mean": statistics.mean(tokens_per_second) if tokens_per_second else 0.0,
            "tokens_per_second_p50": percentile(tokens_per_second, 50),
            "tokens_per_second_p95": percentile(tokens_per_second, 95),
        },
        "prefill": {
            "input_token_count": summarize_numeric(input_token_counts),
            "input_image_count": summarize_numeric(input_image_counts),
            "history_image_count": summarize_numeric(history_image_counts),
            "prompt_char_count": summarize_numeric(prompt_char_counts),
            "image_token_count": summarize_numeric(image_token_counts),
        },
        "consistency": {
            "action_match_rate_all": (action_matches / action_total) if action_total else 0.0,
            "action_match_rate_discrete": (
                discrete_action_matches / discrete_action_total if discrete_action_total else 0.0
            ),
            "action_match_rate_pixel_goal": (pixel_action_matches / pixel_action_total if pixel_action_total else 0.0),
            "output_kind_match_rate": (output_kind_matches / output_kind_total) if output_kind_total else 0.0,
            "text_exact_match_rate": (text_exact_matches / text_exact_total) if text_exact_total else 0.0,
            "pixel_goal_l1_mean": statistics.mean(pixel_goal_l1_errors) if pixel_goal_l1_errors else 0.0,
            "pixel_goal_l1_p50": percentile(pixel_goal_l1_errors, 50),
            "pixel_goal_l1_p95": percentile(pixel_goal_l1_errors, 95),
            "pixel_goal_l2_mean": statistics.mean(pixel_goal_l2_errors) if pixel_goal_l2_errors else 0.0,
            "pixel_goal_l2_p50": percentile(pixel_goal_l2_errors, 50),
            "pixel_goal_l2_p95": percentile(pixel_goal_l2_errors, 95),
        },
        "breakdown": {
            "steps_discrete": discrete_action_total,
            "steps_pixel_goal": pixel_action_total,
            "steps_output_kind_match": output_kind_matches,
        },
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

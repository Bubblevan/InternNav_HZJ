import argparse
import json
import os
import re
import statistics
import time
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from internnav.habitat_extensions.vln.habitat_vln_evaluator import DEFAULT_IMAGE_TOKEN
from internnav.habitat_extensions.vln.habitat_vln_evaluator import action_code
from internnav.habitat_extensions.vln.habitat_vln_evaluator import split_and_clean
from internnav.habitat_extensions.vln.habitat_vln_evaluator import traj_to_actions
from internnav.habitat_extensions.vln.utils import preprocess_depth_image_v2
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM


CONJUNCTIONS = [
    "you can see ",
    "in front of you is ",
    "there is ",
    "you can spot ",
    "you are toward the ",
    "ahead of you is ",
    "in your sight is ",
]
PROMPT = (
    "You are an autonomous navigation assistant. Your task is to <instruction>. "
    "Where should you go next to stay on track? Please output the next waypoint's "
    "coordinates in the image. Please output STOP when you have successfully completed the task."
)
ACTIONS2IDX = OrderedDict({"STOP": [0], "↑": [1], "←": [2], "→": [3], "↓": [5]})


def parse_args():
    parser = argparse.ArgumentParser(description="Offline DualVLN replay benchmark.")
    parser.add_argument("--manifest", required=True, help="Replay manifest jsonl path")
    parser.add_argument("--model-path", required=True, help="DualVLN checkpoint path")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--num-history", type=int, default=8, help="History length")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on replay steps")
    parser.add_argument("--output", required=True, help="Summary json output path")
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


def load_manifest(path):
    grouped = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            grouped[(item["scene_id"], int(item["episode_id"]))].append(item)
    for items in grouped.values():
        items.sort(key=lambda x: x["step_id"])
    return grouped


def build_messages(instruction, episode_steps, current_index, num_history):
    sources = [{"from": "human", "value": PROMPT}, {"from": "gpt", "value": ""}]
    sources[0]["value"] = sources[0]["value"].replace("<instruction>.", instruction[:-1])

    if current_index == 0:
        history_indices = []
    else:
        history_indices = np.unique(np.linspace(0, current_index - 1, num_history, dtype=np.int32)).tolist()
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


def main():
    args = parse_args()
    replay_steps = load_manifest(args.manifest)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"
    model = InternVLAN1ForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": torch.device(args.device)},
    )
    model.eval()
    cold_start_seconds = time.perf_counter() - load_start

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    warm_latencies = []
    s2_latencies = []
    tokens_per_second = []
    gen_lengths = []
    action_matches = 0
    action_total = 0
    pixel_goal_errors = []
    records = []

    for _, episode in replay_steps.items():
        for step_index, item in enumerate(episode):
            if args.max_steps is not None and len(records) >= args.max_steps:
                break

            messages, input_images, history_indices = build_messages(
                item["instruction"], episode, step_index, args.num_history
            )
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=input_images, return_tensors="pt").to(model.device)

            step_start = time.perf_counter()
            s2_start = time.perf_counter()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    use_cache=True,
                    past_key_values=None,
                    return_dict_in_generate=True,
                ).sequences
            s2_seconds = time.perf_counter() - s2_start
            s2_latencies.append(s2_seconds)

            output_text = processor.tokenizer.decode(
                output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
            gen_token_count = int(output_ids.shape[1] - inputs.input_ids.shape[1])
            gen_lengths.append(gen_token_count)
            tokens_per_second.append(gen_token_count / max(s2_seconds, 1e-6))

            predicted_action = None
            predicted_pixel_goal = None
            if bool(re.search(r"\d", output_text)):
                coord = [int(c) for c in re.findall(r"\d+", output_text)]
                if len(coord) >= 2:
                    predicted_pixel_goal = [int(coord[1]), int(coord[0])]

                lookdown_rgb = Image.open(item["lookdown_rgb_path"]).convert("RGB")
                lookdown_depth = np.load(item["lookdown_depth_path"])
                lookdown_depth, _ = preprocess_depth_image_v2(
                    Image.fromarray((lookdown_depth[:, :, 0] * 1000).astype(np.uint16), mode="I;16"),
                    do_depth_scale=True,
                    depth_scale=1000,
                    target_height=224,
                    target_width=224,
                )
                lookdown_depth = torch.as_tensor(np.ascontiguousarray(lookdown_depth)).float()
                lookdown_depth[lookdown_depth > 5.0] = 5.0

                pixel_values = inputs.pixel_values
                image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)
                with torch.no_grad():
                    traj_latents = model.generate_latents(output_ids, pixel_values, image_grid_thw)

                image_dp = torch.tensor(np.array(lookdown_rgb.resize((224, 224)))).to(torch.bfloat16) / 255
                images_dp = torch.stack([image_dp, image_dp]).unsqueeze(0).to(model.device)
                depth_dp = lookdown_depth.unsqueeze(-1).to(torch.bfloat16)
                depths_dp = torch.stack([depth_dp, depth_dp]).unsqueeze(0).to(model.device)
                with torch.no_grad():
                    dp_actions = model.generate_traj(traj_latents, images_dp, depths_dp)
                action_list = traj_to_actions(dp_actions)
                predicted_action = int(action_list[0]) if action_list else int(action_code.STOP)
            else:
                actions = parse_actions(output_text)
                predicted_action = int(actions[0]) if actions else int(action_code.STOP)

            total_step_seconds = time.perf_counter() - step_start
            warm_latencies.append(total_step_seconds)

            baseline = item["baseline_output"]
            baseline_action = int(baseline["action"])
            action_total += 1
            if predicted_action == baseline_action:
                action_matches += 1

            if predicted_pixel_goal is not None and baseline["pixel_goal"] is not None:
                base_goal = baseline["pixel_goal"]
                pixel_goal_errors.append(
                    float(abs(predicted_pixel_goal[0] - base_goal[0]) + abs(predicted_pixel_goal[1] - base_goal[1]))
                )

            records.append(
                {
                    "scene_id": item["scene_id"],
                    "episode_id": item["episode_id"],
                    "step_id": item["step_id"],
                    "history_frame_indices": history_indices,
                    "latency_seconds": total_step_seconds,
                    "s2_seconds": s2_seconds,
                    "generated_tokens": gen_token_count,
                    "output_text": output_text,
                    "predicted_action": predicted_action,
                    "baseline_action": baseline_action,
                    "predicted_pixel_goal": predicted_pixel_goal,
                    "baseline_pixel_goal": baseline["pixel_goal"],
                }
            )
        if args.max_steps is not None and len(records) >= args.max_steps:
            break

    summary = {
        "cold_start_load_seconds": cold_start_seconds,
        "warm_latency_seconds_p50": percentile(warm_latencies, 50),
        "warm_latency_seconds_p95": percentile(warm_latencies, 95),
        "warm_latency_seconds_max": max(warm_latencies) if warm_latencies else 0.0,
        "s2_latency_seconds_p50": percentile(s2_latencies, 50),
        "s2_latency_seconds_p95": percentile(s2_latencies, 95),
        "tokens_per_second_mean": statistics.mean(tokens_per_second) if tokens_per_second else 0.0,
        "generated_tokens_mean": statistics.mean(gen_lengths) if gen_lengths else 0.0,
        "gpu_peak_memory_mb": (
            float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
        ),
        "action_match_rate": (action_matches / action_total) if action_total else 0.0,
        "pixel_goal_l1_mean": statistics.mean(pixel_goal_errors) if pixel_goal_errors else 0.0,
        "num_steps": len(records),
        "records": records,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

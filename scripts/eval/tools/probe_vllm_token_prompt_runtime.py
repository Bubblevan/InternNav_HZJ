import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

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
}
TRAJ_TOKEN_INDEX = 151667


def parse_args():
    parser = argparse.ArgumentParser(
        description="Runtime probe for feeding baseline output token ids into vLLM pooling/token_embed."
    )
    parser.add_argument("--sample-pt", required=True, help="Exported baseline .pt sample path")
    parser.add_argument("--manifest", required=True, help="Replay1 manifest jsonl path")
    parser.add_argument("--model-path", required=True, help="Model path used by vLLM")
    parser.add_argument("--num-history", type=int, default=8, help="History length when rebuilding prompt")
    parser.add_argument("--prompt-variant", choices=sorted(PROMPT_VARIANTS.keys()), default="full")
    parser.add_argument("--ignore-manifest-history", action="store_true")
    parser.add_argument("--append-traj-tokens", action="store_true", help="Append 4 raw TRAJ_TOKEN ids after baseline_output_ids")
    parser.add_argument("--traj-token-count", type=int, default=4, help="Number of TRAJ_TOKEN ids to append")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit-mm-per-prompt-image", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--base-path", default=None, help="Optional base path to replace ./logs/ in manifest paths")
    return parser.parse_args()


def load_manifest(path, base_path=None):
    grouped = defaultdict(list)
    old_prefix = "./logs/"
    if base_path is not None:
        base_path = base_path.rstrip("/") + "/"

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            for field in ("rgb_path", "depth_path", "lookdown_rgb_path", "lookdown_depth_path"):
                if field in item and base_path is not None and item[field].startswith(old_prefix):
                    item[field] = item[field].replace(old_prefix, base_path)
                if field in item:
                    item[field] = os.path.abspath(item[field])
            grouped[(item["scene_id"], int(item["episode_id"]))].append(item)
    for items in grouped.values():
        items.sort(key=lambda x: x["step_id"])
    return grouped


def annotate_inferred_decision_points(replay_steps):
    for episode in replay_steps.values():
        previous_llm_output = None
        for idx, item in enumerate(episode):
            llm_output = item["baseline_output"]["llm_output"]
            previous_item = episode[idx - 1] if idx > 0 else None
            is_lookdown_followup = (
                previous_item is not None
                and item["baseline_output"]["output_kind"] == "pixel_goal"
                and previous_item["step_id"] == item["step_id"]
                and previous_item["baseline_output"]["output_kind"] == "discrete"
                and "↓" in previous_item["baseline_output"]["llm_output"]
            )
            item["_is_inferred_decision_point"] = (llm_output != previous_llm_output) or is_lookdown_followup
            item["_is_inferred_lookdown_followup"] = is_lookdown_followup
            previous_llm_output = llm_output


def build_messages(instruction, episode_steps, current_index, num_history, prompt_variant, ignore_manifest_history=False):
    prompt_template = PROMPT_VARIANTS[prompt_variant]
    prompt = prompt_template.replace("<instruction>.", instruction[:-1]).replace("<instruction>", instruction.strip())
    manifest_history = None if ignore_manifest_history else episode_steps[current_index].get("history_frame_indices", None)
    if manifest_history:
        history_indices = sorted(manifest_history)
    elif current_index == 0 or num_history == 0:
        history_indices = []
    else:
        history_indices = np.unique(np.linspace(0, current_index - 1, num_history, dtype=np.int32)).tolist()

    if history_indices:
        placeholder = ("<image>\n") * len(history_indices)
        prompt += f" These are your historical observations: {placeholder}."

    history_indices = sorted(history_indices)
    input_images = [Image.open(episode_steps[i]["rgb_path"]).convert("RGB") for i in history_indices]
    input_images.append(Image.open(episode_steps[current_index]["rgb_path"]).convert("RGB"))
    prompt += f" {CONJUNCTIONS[0]}<image>."

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return messages, input_images, history_indices


def build_lookdown_input_images(episode_steps, current_index, num_history, ignore_manifest_history=False):
    manifest_history = None if ignore_manifest_history else episode_steps[current_index].get("history_frame_indices", None)
    if manifest_history:
        history_indices = sorted(manifest_history)
    elif current_index == 0 or num_history == 0:
        history_indices = []
    else:
        history_indices = np.unique(np.linspace(0, current_index - 1, num_history, dtype=np.int32)).tolist()

    history_indices = sorted(history_indices)
    input_images = [Image.open(episode_steps[i]["rgb_path"]).convert("RGB") for i in history_indices]
    input_images.append(Image.open(episode_steps[current_index]["rgb_path"]).convert("RGB"))
    input_images.append(Image.open(episode_steps[current_index]["lookdown_rgb_path"]).convert("RGB"))
    return input_images, history_indices


def find_step_index(episode, step_id, baseline_llm_output):
    for idx, item in enumerate(episode):
        if int(item["step_id"]) == int(step_id) and item["baseline_output"]["llm_output"] == baseline_llm_output:
            return idx
    raise KeyError(f"Could not match step_id={step_id} baseline_llm_output={baseline_llm_output!r} in episode")


def main():
    args = parse_args()

    from vllm import LLM
    from vllm.inputs.data import TokensPrompt

    sample = torch.load(args.sample_pt, map_location="cpu")
    replay_steps = load_manifest(args.manifest, args.base_path)
    annotate_inferred_decision_points(replay_steps)

    key = (sample["scene_id"], int(sample["episode_id"]))
    if key not in replay_steps:
        raise SystemExit(f"Episode {key} not found in manifest")
    episode = replay_steps[key]
    step_index = find_step_index(episode, sample["step_id"], sample["baseline_output"]["llm_output"])
    item = episode[step_index]

    if item.get("_is_inferred_lookdown_followup", False):
        input_images, history_indices = build_lookdown_input_images(
            episode,
            step_index,
            args.num_history,
            args.ignore_manifest_history,
        )
    else:
        _, input_images, history_indices = build_messages(
            item["instruction"],
            episode,
            step_index,
            args.num_history,
            args.prompt_variant,
            args.ignore_manifest_history,
        )

    prompt_token_ids = sample["baseline_output_ids"][0].tolist()
    if args.append_traj_tokens:
        prompt_token_ids = prompt_token_ids + [TRAJ_TOKEN_INDEX] * args.traj_token_count

    prompt = TokensPrompt(
        prompt_token_ids=prompt_token_ids,
        prompt=sample.get("prompt_text"),
        multi_modal_data={"image": input_images},
    )

    report = {
        "model_path": args.model_path,
        "sample_pt": args.sample_pt,
        "manifest": args.manifest,
        "scene_id": sample["scene_id"],
        "episode_id": int(sample["episode_id"]),
        "step_id": int(sample["step_id"]),
        "baseline_output_kind": sample["baseline_output"]["output_kind"],
        "baseline_llm_output": sample["baseline_output"]["llm_output"],
        "history_frame_indices": history_indices,
        "num_input_images": len(input_images),
        "requested_prompt_token_count": len(prompt_token_ids),
        "baseline_output_ids_count": int(sample["baseline_output_ids"].shape[1]),
        "append_traj_tokens": bool(args.append_traj_tokens),
        "traj_token_count": int(args.traj_token_count if args.append_traj_tokens else 0),
        "runtime_probe": {},
    }

    try:
        llm = LLM(
            model=args.model_path,
            runner="pooling",
            convert="embed",
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            limit_mm_per_prompt={"image": args.limit_mm_per_prompt_image},
            trust_remote_code=args.trust_remote_code,
            enforce_eager=args.enforce_eager,
        )
        outputs = llm.encode([prompt], pooling_task="token_embed", use_tqdm=False)
        output = outputs[0]
        data = output.outputs.data
        report["runtime_probe"] = {
            "success": True,
            "prompt_token_count": len(output.prompt_token_ids),
            "num_cached_tokens": int(output.num_cached_tokens),
            "token_embed_shape": list(data.shape),
            "token_embed_dtype": str(data.dtype),
            "length_matches_requested_tokens": bool(len(output.prompt_token_ids) == len(prompt_token_ids)),
            "shape_matches_requested_tokens": bool(data.shape[0] == len(prompt_token_ids)),
            "first_prompt_token_ids": output.prompt_token_ids[:32],
            "last_prompt_token_ids": output.prompt_token_ids[-16:],
        }
    except Exception as exc:
        report["runtime_probe"] = {
            "success": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 72)
    print("Probe vLLM token-prompt runtime")
    print("=" * 72)
    print(f"Model path: {args.model_path}")
    print(f"Sample: {sample['scene_id']} / ep {int(sample['episode_id'])} / step {int(sample['step_id'])}")
    print(f"Requested prompt token count: {len(prompt_token_ids)}")
    if report["runtime_probe"]["success"]:
        print("Runtime probe succeeded")
        print(f"Returned prompt token count: {report['runtime_probe']['prompt_token_count']}")
        print(f"Token embed shape: {report['runtime_probe']['token_embed_shape']}")
        print(f"Length matches requested: {report['runtime_probe']['length_matches_requested_tokens']}")
        print(f"Shape matches requested: {report['runtime_probe']['shape_matches_requested_tokens']}")
    else:
        print("Runtime probe failed")
        print(f"{report['runtime_probe']['error_type']}: {report['runtime_probe']['error_message']}")
    if args.output:
        print(f"Saved JSON summary to {args.output}")
    print("=" * 72)


if __name__ == "__main__":
    main()

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoProcessor

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
DEFAULT_IMAGE_TOKEN = "<image>"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Runtime probe for vLLM pooling/token_embed on a replay1 DualVLN sample."
    )
    parser.add_argument("--manifest", required=True, help="Replay1 manifest jsonl path")
    parser.add_argument("--model-path", required=True, help="Model path used for both AutoProcessor and vLLM")
    parser.add_argument("--sample-index", type=int, default=0, help="Which inferred pixel-goal decision sample to use")
    parser.add_argument("--num-history", type=int, default=8, help="History length when rebuilding prompt")
    parser.add_argument("--prompt-variant", choices=sorted(PROMPT_VARIANTS.keys()), default="full")
    parser.add_argument("--ignore-manifest-history", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit-mm-per-prompt-image", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument(
        "--base-path",
        default=None,
        help="Optional base path to replace ./logs/ in manifest paths",
    )
    return parser.parse_args()


def load_manifest(path, base_path=None):
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


def split_and_clean(text):
    parts = text.split(DEFAULT_IMAGE_TOKEN)
    cleaned = []
    for i, part in enumerate(parts):
        if part:
            cleaned.append(part)
        if i != len(parts) - 1:
            cleaned.append(DEFAULT_IMAGE_TOKEN)
    return cleaned


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
        placeholder = (DEFAULT_IMAGE_TOKEN + "\n") * len(history_indices)
        prompt += f" These are your historical observations: {placeholder}."

    history_indices = sorted(history_indices)
    input_images = [Image.open(episode_steps[i]["rgb_path"]).convert("RGB") for i in history_indices]
    input_images.append(Image.open(episode_steps[current_index]["rgb_path"]).convert("RGB"))
    prompt += f" {CONJUNCTIONS[0]}{DEFAULT_IMAGE_TOKEN}."

    parts = split_and_clean(prompt)
    content = []
    image_idx = 0
    for part in parts:
        if part == DEFAULT_IMAGE_TOKEN:
            content.append({"type": "image", "image": input_images[image_idx]})
            image_idx += 1
        else:
            content.append({"type": "text", "text": part})
    return [{"role": "user", "content": content}], input_images, history_indices


def build_lookdown_messages(
    instruction,
    episode_steps,
    current_index,
    num_history,
    prompt_variant,
    previous_assistant_text,
    ignore_manifest_history=False,
):
    messages, input_images, history_indices = build_messages(
        instruction,
        episode_steps,
        current_index,
        num_history,
        prompt_variant,
        ignore_manifest_history,
    )
    lookdown_image = Image.open(episode_steps[current_index]["lookdown_rgb_path"]).convert("RGB")
    messages.append({"role": "assistant", "content": [{"type": "text", "text": previous_assistant_text}]})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f" {CONJUNCTIONS[0]}"},
                {"type": "image", "image": lookdown_image},
                {"type": "text", "text": "."},
            ],
        }
    )
    input_images = list(input_images) + [lookdown_image]
    return messages, input_images, history_indices


def select_target_sample(replay_steps, sample_index):
    candidates = []
    for _, episode in replay_steps.items():
        for step_index, item in enumerate(episode):
            if (
                item["baseline_output"]["output_kind"] == "pixel_goal"
                and item.get("_is_inferred_decision_point", False)
            ):
                candidates.append((episode, step_index, item))
    if sample_index >= len(candidates):
        raise SystemExit(f"sample_index={sample_index} out of range, only {len(candidates)} candidates found")
    return candidates[sample_index], len(candidates)


def main():
    args = parse_args()

    from vllm import LLM
    from vllm.inputs.data import TextPrompt

    replay_steps = load_manifest(args.manifest, args.base_path)
    annotate_inferred_decision_points(replay_steps)
    (episode, step_index, item), total_candidates = select_target_sample(replay_steps, args.sample_index)

    previous_item = episode[step_index - 1] if step_index > 0 else None
    if item.get("_is_inferred_lookdown_followup", False) and previous_item is not None:
        messages, input_images, history_indices = build_lookdown_messages(
            item["instruction"],
            episode,
            step_index,
            args.num_history,
            args.prompt_variant,
            previous_item["baseline_output"]["llm_output"],
            args.ignore_manifest_history,
        )
    else:
        messages, input_images, history_indices = build_messages(
            item["instruction"],
            episode,
            step_index,
            args.num_history,
            args.prompt_variant,
            args.ignore_manifest_history,
        )

    processor = AutoProcessor.from_pretrained(args.model_path)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text_prompt = TextPrompt(prompt=prompt_text, multi_modal_data={"image": input_images})

    report = {
        "model_path": args.model_path,
        "manifest": args.manifest,
        "sample_index": args.sample_index,
        "total_candidates": total_candidates,
        "scene_id": item["scene_id"],
        "episode_id": int(item["episode_id"]),
        "step_id": int(item["step_id"]),
        "baseline_output_kind": item["baseline_output"]["output_kind"],
        "baseline_llm_output": item["baseline_output"]["llm_output"],
        "is_inferred_lookdown_followup": bool(item.get("_is_inferred_lookdown_followup", False)),
        "history_frame_indices": history_indices,
        "num_input_images": len(input_images),
        "prompt_preview": prompt_text[:500],
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
        outputs = llm.encode([text_prompt], pooling_task="token_embed", use_tqdm=False)
        output = outputs[0]
        data = output.outputs.data
        report["runtime_probe"] = {
            "success": True,
            "prompt_token_count": len(output.prompt_token_ids),
            "num_cached_tokens": int(output.num_cached_tokens),
            "token_embed_shape": list(data.shape),
            "token_embed_dtype": str(data.dtype),
            "first_prompt_token_ids": output.prompt_token_ids[:32],
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
    print("Probe vLLM pooling runtime")
    print("=" * 72)
    print(f"Model path: {args.model_path}")
    print(f"Sample: {item['scene_id']} / ep {int(item['episode_id'])} / step {int(item['step_id'])}")
    print(f"Images: {len(input_images)}")
    if report["runtime_probe"]["success"]:
        print("Runtime probe succeeded")
        print(f"Prompt token count: {report['runtime_probe']['prompt_token_count']}")
        print(f"Token embed shape: {report['runtime_probe']['token_embed_shape']}")
        print(f"Token embed dtype: {report['runtime_probe']['token_embed_dtype']}")
    else:
        print("Runtime probe failed")
        print(f"{report['runtime_probe']['error_type']}: {report['runtime_probe']['error_message']}")
    if args.output:
        print(f"Saved JSON summary to {args.output}")
    print("=" * 72)


if __name__ == "__main__":
    main()

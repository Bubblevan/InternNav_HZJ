import argparse
import json
import os
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.utils.vln_utils import split_and_clean


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export HF generate_latents() baseline samples from replay_subset (replay1)."
    )
    parser.add_argument("--manifest", required=True, help="Replay1 manifest jsonl path")
    parser.add_argument("--model-path", required=True, help="DualVLN checkpoint path")
    parser.add_argument("--output-dir", required=True, help="Directory to save exported latent baseline samples")
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
        help="Prompt template variant",
    )
    parser.add_argument(
        "--ignore-manifest-history",
        action="store_true",
        help="Ignore saved history_frame_indices and rebuild history from --num-history",
    )
    parser.add_argument(
        "--decision-filter",
        choices=["pixel_goal_only", "all", "decision_points_only"],
        default="pixel_goal_only",
        help="Which replay1 steps to export. replay1 pixel_goal_only now means inferred new pixel-goal decisions only.",
    )
    parser.add_argument("--max-samples", type=int, default=8, help="Maximum number of exported samples")
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


def should_export(item, decision_filter):
    if decision_filter == "all":
        return True
    if decision_filter == "decision_points_only":
        return bool(item.get("_is_inferred_decision_point", False))
    return item["baseline_output"]["output_kind"] == "pixel_goal" and bool(
        item.get("_is_inferred_decision_point", False)
    )


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


def main():
    args = parse_args()
    replay_steps = load_manifest(args.manifest, args.base_path)
    annotate_inferred_decision_points(replay_steps)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    summary_path = output_dir / "summary.json"

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

    metadata_records = []
    exported = 0
    examined = 0

    total_candidates = sum(
        1
        for episode in replay_steps.values()
        for item in episode
        if should_export(item, args.decision_filter)
    )

    with open(metadata_path, "w", encoding="utf-8") as metadata_f:
        progress = tqdm(total=min(total_candidates, args.max_samples), desc="Export HF latents", dynamic_ncols=True)
        for _, episode in replay_steps.items():
            for step_index, item in enumerate(episode):
                if exported >= args.max_samples:
                    break
                if not should_export(item, args.decision_filter):
                    continue

                examined += 1
                baseline = item["baseline_output"]
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
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=input_images, return_tensors="pt").to(model.device)
                image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)

                baseline_generated_ids = processor.tokenizer.encode(baseline["llm_output"], add_special_tokens=False)
                baseline_output_ids = torch.cat(
                    [
                        inputs.input_ids,
                        torch.tensor([baseline_generated_ids], device=inputs.input_ids.device, dtype=inputs.input_ids.dtype),
                    ],
                    dim=1,
                )

                with torch.no_grad():
                    baseline_latent = model.generate_latents(baseline_output_ids, inputs.pixel_values, image_grid_thw)

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        use_cache=True,
                        past_key_values=None,
                        return_dict_in_generate=True,
                    ).sequences

                output_text = processor.tokenizer.decode(
                    output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                )

                with torch.no_grad():
                    hf_generate_latent = model.generate_latents(output_ids, inputs.pixel_values, image_grid_thw)

                generated_ids = output_ids[0][inputs.input_ids.shape[1] :].detach().cpu()
                pixel_goal_match = baseline["pixel_goal"] is None
                exported_pixel_goal = None
                if bool(re.search(r"\d", output_text)):
                    coord = [int(c) for c in re.findall(r"\d+", output_text)]
                    if len(coord) >= 2:
                        exported_pixel_goal = [int(coord[1]), int(coord[0])]
                if baseline["pixel_goal"] is not None:
                    pixel_goal_match = exported_pixel_goal == baseline["pixel_goal"]

                latent_allclose = torch.allclose(
                    baseline_latent.detach().cpu(),
                    hf_generate_latent.detach().cpu(),
                    atol=1e-4,
                    rtol=1e-4,
                )
                latent_max_abs_diff = float(
                    (baseline_latent.detach().cpu() - hf_generate_latent.detach().cpu()).abs().max().item()
                )

                sample_name = (
                    f"sample_{exported:04d}_{item['scene_id']}_{int(item['episode_id']):04d}_step_{int(item['step_id']):04d}"
                )
                sample_path = samples_dir / f"{sample_name}.pt"
                torch.save(
                    {
                        "scene_id": item["scene_id"],
                        "episode_id": int(item["episode_id"]),
                        "step_id": int(item["step_id"]),
                        "instruction": item["instruction"],
                        "prompt_variant": args.prompt_variant,
                        "history_frame_indices": history_indices,
                        "prompt_text": text,
                        "is_inferred_decision_point": bool(item["_is_inferred_decision_point"]),
                        "is_inferred_lookdown_followup": bool(item.get("_is_inferred_lookdown_followup", False)),
                        "baseline_output": baseline,
                        "hf_output_text": output_text,
                        "baseline_output_ids": baseline_output_ids.detach().cpu(),
                        "baseline_generated_ids": torch.tensor(baseline_generated_ids, dtype=torch.long),
                        "baseline_latent": baseline_latent.detach().cpu(),
                        "input_ids": inputs.input_ids.detach().cpu(),
                        "hf_generate_output_ids": output_ids.detach().cpu(),
                        "hf_generate_generated_ids": generated_ids,
                        "pixel_values": inputs.pixel_values.detach().cpu(),
                        "image_grid_thw": image_grid_thw.detach().cpu(),
                        "hf_generate_latent": hf_generate_latent.detach().cpu(),
                    },
                    sample_path,
                )

                record = {
                    "sample_name": sample_name,
                    "sample_path": str(sample_path),
                    "scene_id": item["scene_id"],
                    "episode_id": int(item["episode_id"]),
                    "step_id": int(item["step_id"]),
                    "instruction": item["instruction"],
                    "history_frame_indices": history_indices,
                    "input_image_paths": [img["image"].filename if hasattr(img["image"], "filename") else None for img in messages[0]["content"] if img["type"] == "image"],
                    "is_inferred_decision_point": bool(item["_is_inferred_decision_point"]),
                    "is_inferred_lookdown_followup": bool(item.get("_is_inferred_lookdown_followup", False)),
                    "input_token_count": int(inputs.input_ids.shape[1]),
                    "image_token_count": count_image_tokens(image_grid_thw),
                    "num_input_images": len(input_images),
                    "baseline_output_kind": baseline["output_kind"],
                    "baseline_llm_output": baseline["llm_output"],
                    "baseline_generated_token_count": len(baseline_generated_ids),
                    "hf_output_text": output_text,
                    "text_exact_match": output_text.strip() == baseline["llm_output"].strip(),
                    "baseline_pixel_goal": baseline["pixel_goal"],
                    "hf_pixel_goal": exported_pixel_goal,
                    "pixel_goal_match": pixel_goal_match,
                    "generated_token_count": int(output_ids.shape[1] - inputs.input_ids.shape[1]),
                    "latent_shape": list(baseline_latent.shape),
                    "latent_allclose_baseline_vs_hf_generate": bool(latent_allclose),
                    "latent_max_abs_diff_baseline_vs_hf_generate": latent_max_abs_diff,
                }
                metadata_records.append(record)
                metadata_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                exported += 1
                progress.update(1)

            if exported >= args.max_samples:
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
            "ignore_manifest_history": bool(args.ignore_manifest_history),
            "decision_filter": args.decision_filter,
            "max_samples": args.max_samples,
            "output_dir": str(output_dir),
        },
        "counts": {
            "examined_candidates": examined,
            "exported_samples": exported,
            "text_exact_matches": int(sum(r["text_exact_match"] for r in metadata_records)),
            "pixel_goal_matches": int(sum(r["pixel_goal_match"] for r in metadata_records)),
            "latent_allclose_matches": int(
                sum(r["latent_allclose_baseline_vs_hf_generate"] for r in metadata_records)
            ),
        },
        "match_rates": {
            "text_exact_match_rate": (
                float(sum(r["text_exact_match"] for r in metadata_records) / exported) if exported else None
            ),
            "pixel_goal_match_rate": (
                float(sum(r["pixel_goal_match"] for r in metadata_records) / exported) if exported else None
            ),
            "latent_allclose_rate_baseline_vs_hf_generate": (
                float(sum(r["latent_allclose_baseline_vs_hf_generate"] for r in metadata_records) / exported)
                if exported
                else None
            ),
        },
        "artifacts": {
            "metadata_jsonl": str(metadata_path),
            "samples_dir": str(samples_dir),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 72)
    print("Export HF generate_latents baseline")
    print("=" * 72)
    print(f"Exported samples: {exported}")
    print(f"Text exact match rate: {summary['match_rates']['text_exact_match_rate']}")
    print(f"Pixel-goal match rate: {summary['match_rates']['pixel_goal_match_rate']}")
    print(f"Summary:  {summary_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Samples:  {samples_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()

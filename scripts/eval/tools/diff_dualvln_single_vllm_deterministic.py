import argparse
import base64
import hashlib
import io
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from internnav.habitat_extensions.vln.utils import preprocess_depth_image_v2
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.utils.dualvln_single_vllm import DualVLNSingleVLLMHTTPClient
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
}
DEFAULT_IMAGE_TOKEN = "<image>"
MAX_STEPS = 8
MAX_LOCAL_STEPS = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deterministic per-step diff harness for original DualVLN HF vs single-vLLM HTTP."
    )
    parser.add_argument("--manifest", required=True, help="Replay manifest jsonl path")
    parser.add_argument("--hf-model-path", required=True, help="Original DualVLN checkpoint path")
    parser.add_argument("--single-vllm-url", required=True, help="single-vLLM HTTP service URL")
    parser.add_argument("--output", required=True, help="Summary json output path")
    parser.add_argument("--details-output", default=None, help="Optional per-step jsonl output path")
    parser.add_argument("--base-path", default=None, help="Optional base path to replace ./logs/")
    parser.add_argument("--scene-id", default=None, help="Optional scene id filter")
    parser.add_argument(
        "--episode-ids",
        default=None,
        help="Optional comma-separated episode id filter, e.g. 10,16,17,43,44",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument(
        "--attn-backend",
        choices=["flash_attention_2", "sdpa", "eager"],
        default="flash_attention_2",
        help="Attention backend for the HF reference model",
    )
    parser.add_argument(
        "--processor-use-fast",
        choices=["auto", "true", "false"],
        default="auto",
        help="Forward use_fast to AutoProcessor when not set to auto",
    )
    parser.add_argument("--num-history", type=int, default=8, help="History length")
    parser.add_argument("--prompt-variant", choices=sorted(PROMPT_VARIANTS.keys()), default="full")
    parser.add_argument("--ignore-manifest-history", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0, help="Global deterministic seed")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on processed records")
    parser.add_argument("--decision-points-only", action="store_true", help="Only diff inferred new S2 decisions")
    parser.add_argument("--latent-max-abs-tol", type=float, default=1e-4)
    parser.add_argument("--latent-min-cosine", type=float, default=0.9999)
    return parser.parse_args()


def default_details_output(output_path):
    stem, _ = os.path.splitext(output_path)
    return f"{stem}.jsonl"


def parse_episode_ids(raw_episode_ids):
    if raw_episode_ids is None or not str(raw_episode_ids).strip():
        return None
    return {int(item.strip()) for item in str(raw_episode_ids).split(",") if item.strip()}


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
                    item[field] = item[field].replace(old_prefix, base_path, 1)
                if field in item:
                    item[field] = os.path.abspath(item[field])
            grouped[(item["scene_id"], int(item["episode_id"]))].append(item)
    for items in grouped.values():
        items.sort(key=lambda x: (int(x["step_id"]), x["baseline_output"]["output_kind"]))
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
                {"type": "text", "text": " " + CONJUNCTIONS[0]},
                {"type": "image", "image": lookdown_image},
                {"type": "text", "text": "."},
            ],
        }
    )
    return messages, list(input_images) + [lookdown_image], history_indices


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


def parse_pixel_goal(output_text):
    if not re.search(r"\d", output_text):
        return None
    coord = [int(c) for c in re.findall(r"\d+", output_text)]
    if len(coord) < 2:
        return None
    return [int(coord[1]), int(coord[0])]


def make_step_seed(base_seed, scene_id, episode_id, step_id, salt):
    payload = f"{base_seed}:{scene_id}:{episode_id}:{step_id}:{salt}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def make_torch_generator(device, seed):
    if isinstance(device, torch.device):
        generator_device = device.type
    else:
        generator_device = torch.device(device).type
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    return generator


def tensor_norm(tensor):
    return float(tensor.detach().float().norm().item())


def tensor_diff_report(lhs, rhs):
    lhs_cpu = lhs.detach().float().cpu()
    rhs_cpu = rhs.detach().float().cpu()
    diff = (lhs_cpu - rhs_cpu).abs()
    cosine = None
    if lhs_cpu.numel() > 0 and rhs_cpu.numel() > 0:
        cosine = float(F.cosine_similarity(lhs_cpu.reshape(1, -1), rhs_cpu.reshape(1, -1)).item())
    return {
        "lhs_norm": tensor_norm(lhs_cpu),
        "rhs_norm": tensor_norm(rhs_cpu),
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "cosine_similarity": cosine,
    }


def serialize_messages(messages):
    serialized = []
    for message in messages:
        content = []
        for item in message["content"]:
            if item["type"] == "text":
                content.append({"type": "text", "text": item["text"]})
                continue
            image = item["image"]
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            raw = buf.getvalue()
            content.append(
                {
                    "type": "image",
                    "size": [int(image.width), int(image.height)],
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "base64_preview": base64.b64encode(raw[:48]).decode("utf-8"),
                }
            )
        serialized.append({"role": message["role"], "content": content})
    return serialized


def plan_local_actions(model, traj_latents, item, device, seed):
    lookdown_rgb = Image.open(item["lookdown_rgb_path"]).convert("RGB")
    lookdown_depth = load_lookdown_depth(item["lookdown_depth_path"])

    image_dp = torch.tensor(np.array(lookdown_rgb.resize((224, 224)))).to(torch.bfloat16) / 255
    images_dp = torch.stack([image_dp, image_dp]).unsqueeze(0).to(device)
    depth_dp = lookdown_depth.unsqueeze(-1).to(torch.bfloat16)
    depths_dp = torch.stack([depth_dp, depth_dp]).unsqueeze(0).to(device)
    generator = make_torch_generator(device, seed)

    with torch.no_grad():
        dp_actions = model.generate_traj(
            traj_latents,
            images_dp,
            depths_dp,
            generator=generator,
        )

    action_list = traj_to_actions(dp_actions)
    if len(action_list) < MAX_STEPS:
        action_list = list(action_list) + [0] * (MAX_STEPS - len(action_list))

    return {
        "seed": int(seed),
        "action_prefix": [int(action) for action in action_list[:MAX_LOCAL_STEPS]],
        "dp_actions_shape": list(dp_actions.shape),
    }


def classify_divergence(record, args):
    shadow = record.get("shadow")
    if not shadow or shadow.get("error"):
        return "shadow_request"
    if record["hf"]["prompt_token_ids"] != shadow["prompt_token_ids"]:
        return "prompt_token_ids"
    if record["hf"]["generated_token_ids"] != shadow["generated_token_ids"]:
        return "generated_token_ids"
    if record["hf"]["llm_output"] != shadow["llm_output"]:
        return "s2_text"
    if record["hf"]["pixel_goal"] != shadow["pixel_goal"]:
        return "pixel_goal"

    latent_diff = record.get("latent_diff")
    if latent_diff is not None:
        cosine = latent_diff.get("cosine_similarity")
        if (
            latent_diff["max_abs_diff"] > args.latent_max_abs_tol
            or (cosine is not None and cosine < args.latent_min_cosine)
        ):
            return "latent"

    hf_actions = record.get("hf_local_actions_prefix")
    shadow_actions = record.get("shadow_local_actions_prefix")
    if hf_actions is not None and shadow_actions is not None and hf_actions != shadow_actions:
        return "system1_rollout"
    return "match"


def main():
    args = parse_args()
    set_global_seed(args.seed)

    replay_steps = load_manifest(args.manifest, args.base_path)
    annotate_inferred_decision_points(replay_steps)

    processor_kwargs = {}
    if args.processor_use_fast != "auto":
        processor_kwargs["use_fast"] = args.processor_use_fast == "true"
    processor = AutoProcessor.from_pretrained(args.hf_model_path, **processor_kwargs)
    processor.tokenizer.padding_side = "left"
    model = InternVLAN1ForCausalLM.from_pretrained(
        args.hf_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_backend,
        device_map={"": torch.device(args.device)},
    )
    model.eval()

    client = DualVLNSingleVLLMHTTPClient(args.single_vllm_url, timeout=600.0)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    details_output = Path(args.details_output or default_details_output(args.output))
    details_output.parent.mkdir(parents=True, exist_ok=True)

    allowed_episode_ids = parse_episode_ids(args.episode_ids)
    selected_records = []
    for (scene_id, episode_id), episode in replay_steps.items():
        if args.scene_id and scene_id != args.scene_id:
            continue
        if allowed_episode_ids is not None and int(episode_id) not in allowed_episode_ids:
            continue
        for step_index, item in enumerate(episode):
            if args.decision_points_only and not item.get("_is_inferred_decision_point", False):
                continue
            selected_records.append((scene_id, int(episode_id), episode, step_index, item))

    summary = {
        "metadata": {
            "manifest": args.manifest,
            "hf_model_path": args.hf_model_path,
            "single_vllm_url": args.single_vllm_url,
            "scene_id": args.scene_id,
            "episode_ids": sorted(allowed_episode_ids) if allowed_episode_ids is not None else None,
            "seed": int(args.seed),
            "num_history": int(args.num_history),
            "prompt_variant": args.prompt_variant,
            "decision_points_only": bool(args.decision_points_only),
            "details_output": str(details_output),
            "num_records": 0,
        },
        "episodes": {},
        "stage_counts": defaultdict(int),
    }

    processed = 0
    with open(details_output, "w", encoding="utf-8") as detail_f:
        progress = tqdm(total=len(selected_records), desc="Deterministic Diff", dynamic_ncols=True)
        for scene_id, episode_id, episode, step_index, item in selected_records:
            if args.max_steps is not None and processed >= args.max_steps:
                break

            previous_item = episode[step_index - 1] if step_index > 0 else None
            if item.get("_is_inferred_lookdown_followup", False):
                previous_text = previous_item["baseline_output"]["llm_output"] if previous_item is not None else ""
                messages, input_images, history_indices = build_lookdown_messages(
                    item["instruction"],
                    episode,
                    step_index,
                    args.num_history,
                    args.prompt_variant,
                    previous_text,
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

            prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt_text], images=input_images, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    past_key_values=None,
                    return_dict_in_generate=True,
                ).sequences

            prompt_token_ids = inputs.input_ids[0].detach().cpu().tolist()
            generated_token_ids = output_ids[0][inputs.input_ids.shape[1] :].detach().cpu().tolist()
            hf_output_ids = output_ids[0].detach().cpu().tolist()
            hf_output_text = processor.tokenizer.decode(output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
            hf_pixel_goal = parse_pixel_goal(hf_output_text)

            hf_latents = None
            shadow_latents = None
            latent_diff = None
            hf_local_plan = None
            shadow_local_plan = None

            shadow_error = None
            shadow_result = None
            try:
                shadow_result = client.step_s2(
                    messages,
                    max_new_tokens=args.max_new_tokens,
                    target_device=model.device,
                    target_dtype=model.dtype if hasattr(model, "dtype") else torch.bfloat16,
                )
            except Exception as exc:
                shadow_error = {"type": type(exc).__name__, "message": str(exc)}

            if hf_pixel_goal is not None:
                image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)
                with torch.no_grad():
                    hf_latents = model.generate_latents(output_ids, inputs.pixel_values, image_grid_thw)

                local_seed = make_step_seed(args.seed, scene_id, episode_id, int(item["step_id"]), salt=17)
                hf_local_plan = plan_local_actions(model, hf_latents, item, model.device, local_seed)

            if shadow_result is not None and shadow_result.get("latents") is not None:
                shadow_latents = shadow_result["latents"]
                local_seed = make_step_seed(args.seed, scene_id, episode_id, int(item["step_id"]), salt=17)
                shadow_local_plan = plan_local_actions(model, shadow_latents, item, model.device, local_seed)

            if hf_latents is not None and shadow_latents is not None:
                latent_diff = tensor_diff_report(hf_latents[0], shadow_latents[0])

            record = {
                "scene_id": scene_id,
                "episode_id": int(episode_id),
                "step_id": int(item["step_id"]),
                "step_index": int(step_index),
                "history_frame_indices": history_indices,
                "is_inferred_decision_point": bool(item.get("_is_inferred_decision_point", False)),
                "is_inferred_lookdown_followup": bool(item.get("_is_inferred_lookdown_followup", False)),
                "messages": serialize_messages(messages),
                "prompt_text": prompt_text,
                "baseline_output": item["baseline_output"],
                "hf": {
                    "llm_output": hf_output_text,
                    "pixel_goal": hf_pixel_goal,
                    "prompt_token_ids": prompt_token_ids,
                    "generated_token_ids": generated_token_ids,
                    "output_ids": hf_output_ids,
                },
                "shadow": {
                    "error": shadow_error,
                    "llm_output": None if shadow_result is None else shadow_result.get("llm_output"),
                    "pixel_goal": None if shadow_result is None else shadow_result.get("pixel_goal"),
                    "prompt_token_ids": [] if shadow_result is None else list(shadow_result.get("prompt_token_ids") or []),
                    "generated_token_ids": [] if shadow_result is None else list(shadow_result.get("generated_token_ids") or []),
                },
                "latent_diff": latent_diff,
                "hf_latent_norm": None if hf_latents is None else tensor_norm(hf_latents),
                "shadow_latent_norm": None if shadow_latents is None else tensor_norm(shadow_latents),
                "hf_local_actions_prefix": None if hf_local_plan is None else hf_local_plan["action_prefix"],
                "shadow_local_actions_prefix": None if shadow_local_plan is None else shadow_local_plan["action_prefix"],
                "hf_local_seed": None if hf_local_plan is None else hf_local_plan["seed"],
                "shadow_local_seed": None if shadow_local_plan is None else shadow_local_plan["seed"],
            }
            record["earliest_divergence_stage"] = classify_divergence(record, args)

            episode_key = f"{scene_id}:{episode_id}"
            episode_summary = summary["episodes"].setdefault(
                episode_key,
                {
                    "scene_id": scene_id,
                    "episode_id": int(episode_id),
                    "records": 0,
                    "first_divergence": None,
                },
            )
            episode_summary["records"] += 1
            if episode_summary["first_divergence"] is None and record["earliest_divergence_stage"] != "match":
                episode_summary["first_divergence"] = {
                    "step_id": int(item["step_id"]),
                    "step_index": int(step_index),
                    "stage": record["earliest_divergence_stage"],
                    "hf_output": hf_output_text,
                    "shadow_output": None if shadow_result is None else shadow_result.get("llm_output"),
                }

            summary["stage_counts"][record["earliest_divergence_stage"]] += 1
            detail_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            processed += 1
            progress.update(1)

        progress.close()

    summary["metadata"]["num_records"] = processed
    summary["stage_counts"] = dict(summary["stage_counts"])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

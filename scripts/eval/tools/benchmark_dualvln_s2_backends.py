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
from tqdm import tqdm
from transformers import AutoProcessor

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM


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


def parse_args():
    parser = argparse.ArgumentParser(description="S2-only backend benchmark for HF vs vLLM.")
    parser.add_argument("--backend", choices=["hf", "vllm"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--details-output", default=None)
    parser.add_argument("--base-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-history", type=int, default=8)
    parser.add_argument("--prompt-variant", choices=sorted(PROMPT_VARIANTS.keys()), default="full")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--verbose-every", type=int, default=20)
    parser.add_argument("--ignore-manifest-history", action="store_true")
    parser.add_argument(
        "--attn-backend",
        choices=["flash_attention_2", "sdpa", "eager"],
        default="flash_attention_2",
        help="HF-only attention backend",
    )
    parser.add_argument(
        "--processor-use-fast",
        choices=["auto", "true", "false"],
        default="auto",
    )
    parser.add_argument(
        "--vllm-limit-mm-per-prompt",
        type=int,
        default=16,
        help="vLLM limit_mm_per_prompt.image",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code=True to vLLM LLM loader",
    )
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


def summarize(values):
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def default_details_output(output_path):
    stem, _ = os.path.splitext(output_path)
    return f"{stem}.jsonl"


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
    parts = re.split(r"(<image>)", sources[0]["value"])
    content = []
    for part in parts:
        if not part:
            continue
        if part == DEFAULT_IMAGE_TOKEN:
            content.append({"type": "image", "image": input_images[input_img_id]})
            input_img_id += 1
        else:
            content.append({"type": "text", "text": part})
    messages = [{"role": "user", "content": content}]
    return messages, input_images, history_indices


def make_processor(model_path, processor_use_fast):
    kwargs = {}
    if processor_use_fast != "auto":
        kwargs["use_fast"] = processor_use_fast == "true"
    processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    processor.tokenizer.padding_side = "left"
    return processor


class HFBackend:
    def __init__(self, args):
        self.processor = make_processor(args.model_path, args.processor_use_fast)
        self.model = InternVLAN1ForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_backend,
            device_map={"": torch.device(args.device)},
        )
        self.model.eval()
        self.device = self.model.device
        self.name = "hf"

    def run_s2(self, messages, input_images, max_new_tokens):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=input_images, return_tensors="pt").to(self.device)
        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                past_key_values=None,
                return_dict_in_generate=True,
            )
        latency = time.perf_counter() - start
        output_ids = outputs.sequences
        output_text = self.processor.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        gen_tokens = int(output_ids.shape[1] - inputs.input_ids.shape[1])
        return {
            "prompt_text": text,
            "output_text": output_text,
            "generated_tokens": gen_tokens,
            "latency_seconds": latency,
        }


class VLLMBackend:
    def __init__(self, args):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "vLLM is not installed in the current environment. Install vllm first, then rerun this benchmark."
            ) from exc

        self.processor = make_processor(args.model_path, args.processor_use_fast)
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
        llm_kwargs = {
            "model": args.model_path,
            "trust_remote_code": bool(args.trust_remote_code),
            "limit_mm_per_prompt": {"image": int(args.vllm_limit_mm_per_prompt)},
        }
        self.llm = LLM(**llm_kwargs)
        self.name = "vllm"

    def run_s2(self, messages, input_images, max_new_tokens):
        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        request = {
            "prompt": prompt_text,
            "multi_modal_data": {"image": input_images},
        }
        self.sampling_params.max_tokens = max_new_tokens
        start = time.perf_counter()
        outputs = self.llm.generate([request], self.sampling_params)
        latency = time.perf_counter() - start
        result = outputs[0].outputs[0]
        return {
            "prompt_text": prompt_text,
            "output_text": result.text,
            "generated_tokens": len(getattr(result, "token_ids", []) or []),
            "latency_seconds": latency,
        }


def main():
    args = parse_args()
    replay_steps = load_manifest(args.manifest, args.base_path)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    details_output = args.details_output or default_details_output(args.output)
    total_planned_steps = count_total_steps(replay_steps)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    load_start = time.perf_counter()
    backend = HFBackend(args) if args.backend == "hf" else VLLMBackend(args)
    cold_start_seconds = time.perf_counter() - load_start

    total_latencies = []
    gen_lengths = []
    tokens_per_second = []
    action_matches = 0
    action_total = 0
    output_kind_matches = 0
    output_kind_total = 0
    text_exact_matches = 0
    text_exact_total = 0
    records = []

    with open(details_output, "w", encoding="utf-8") as detail_f:
        progress = tqdm(total=total_planned_steps, desc=f"S2 Backend {args.backend}", dynamic_ncols=True)
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
                result = backend.run_s2(messages, input_images, args.max_new_tokens)
                output_text = result["output_text"]
                latency = result["latency_seconds"]
                gen_tokens = result["generated_tokens"]

                predicted_kind = "pixel_goal" if bool(re.search(r"\d", output_text)) else "discrete"
                if predicted_kind == "pixel_goal":
                    coord = [int(c) for c in re.findall(r"\d+", output_text)]
                    predicted_action = None
                    predicted_pixel_goal = [int(coord[1]), int(coord[0])] if len(coord) >= 2 else None
                else:
                    actions = parse_actions(output_text)
                    predicted_action = int(actions[0]) if actions else 0
                    predicted_pixel_goal = None

                total_latencies.append(latency)
                gen_lengths.append(gen_tokens)
                tokens_per_second.append(gen_tokens / max(latency, 1e-6))
                output_kind_total += 1
                if predicted_kind == baseline_kind:
                    output_kind_matches += 1
                text_exact_total += 1
                if output_text.strip() == baseline["llm_output"].strip():
                    text_exact_matches += 1
                if predicted_kind == "discrete":
                    action_total += 1
                    if predicted_action == int(baseline["action"]):
                        action_matches += 1

                record = {
                    "scene_id": item["scene_id"],
                    "episode_id": item["episode_id"],
                    "step_id": item["step_id"],
                    "history_frame_indices": history_indices,
                    "baseline_output_kind": baseline_kind,
                    "predicted_output_kind": predicted_kind,
                    "latency_seconds": latency,
                    "generated_tokens": gen_tokens,
                    "tokens_per_second": gen_tokens / max(latency, 1e-6),
                    "output_text": output_text,
                    "baseline_text": baseline["llm_output"],
                    "predicted_action": predicted_action,
                    "baseline_action": baseline["action"],
                    "predicted_pixel_goal": predicted_pixel_goal,
                    "baseline_pixel_goal": baseline["pixel_goal"],
                }
                records.append(record)
                detail_f.write(json.dumps(record) + "\n")
                progress.update(1)
                if args.verbose_every > 0 and len(records) % args.verbose_every == 0:
                    print(
                        f"[step {len(records)}] backend={args.backend} latency={latency:.3f}s "
                        f"kind={predicted_kind} gen_tokens={gen_tokens}",
                        flush=True,
                    )

            if args.max_steps is not None and len(records) >= args.max_steps:
                break
        progress.close()

    summary = {
        "metadata": {
            "backend": args.backend,
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
            "s2_generate": summarize(total_latencies),
        },
        "generation": {
            "generated_tokens_mean": statistics.mean(gen_lengths) if gen_lengths else 0.0,
            "generated_tokens_p50": percentile(gen_lengths, 50),
            "generated_tokens_p95": percentile(gen_lengths, 95),
            "tokens_per_second_mean": statistics.mean(tokens_per_second) if tokens_per_second else 0.0,
            "tokens_per_second_p50": percentile(tokens_per_second, 50),
            "tokens_per_second_p95": percentile(tokens_per_second, 95),
        },
        "consistency": {
            "discrete_action_match_rate": (action_matches / action_total) if action_total else 0.0,
            "output_kind_match_rate": (output_kind_matches / output_kind_total) if output_kind_total else 0.0,
            "text_exact_match_rate": (text_exact_matches / text_exact_total) if text_exact_total else 0.0,
        },
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

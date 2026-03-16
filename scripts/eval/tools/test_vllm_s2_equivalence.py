"""
Phase 0: S2 等价性测试 — HF baseline (from manifest) vs vLLM HTTP

利用 replay manifest 中已有的 HF baseline_output 作为 ground truth，
对相同输入调 vLLM HTTP API，对比文本、action、pixel goal 等输出。
不需要在本地额外加载 HF 模型。

用法:
  conda activate habitat
  cd /root/backup/InternNav
  python scripts/eval/tools/test_vllm_s2_equivalence.py \
    --manifest logs/habitat/test_dual_system_mini/replay_subset/manifest_rank0.jsonl \
    --model-path checkpoints/InternVLA-N1-DualVLN \
    --vllm-url http://127.0.0.1:8001 \
    --base-path logs \
    --max-steps 30 \
    --output logs/phase0_equivalence.json
"""

import argparse
import base64
import io
import json
import os
import re
import time
from collections import OrderedDict, defaultdict

import numpy as np
import requests as http_requests
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_IMAGE_TOKEN = "<image>"
CONJUNCTIONS = ["you can see "]
PROMPT_TEMPLATE = (
    "You are an autonomous navigation assistant. Your task is to <instruction>. "
    "Where should you go next to stay on track? Please output the next waypoint's "
    "coordinates in the image. Please output STOP when you have successfully completed the task."
)
ACTIONS2IDX = OrderedDict({"STOP": [0], "↑": [1], "←": [2], "→": [3], "↓": [5]})


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0: HF baseline vs vLLM S2 equivalence")
    p.add_argument("--manifest", required=True)
    p.add_argument("--model-path", required=True, help="For loading tokenizer/processor only")
    p.add_argument("--vllm-url", required=True, help="e.g. http://127.0.0.1:8001")
    p.add_argument("--vllm-model", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--base-path", default=None)
    p.add_argument("--num-history", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--max-new-tokens", type=int, default=128)
    return p.parse_args()


def load_manifest(path, base_path=None):
    grouped = defaultdict(list)
    old_prefix = "./logs/"
    if base_path is not None:
        base_path = base_path.rstrip("/") + "/"
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
            for field in ("rgb_path", "depth_path", "lookdown_rgb_path", "lookdown_depth_path"):
                if field in item and base_path and item[field].startswith(old_prefix):
                    item[field] = item[field].replace(old_prefix, base_path)
                if field in item:
                    item[field] = os.path.abspath(item[field])
            grouped[(item["scene_id"], int(item["episode_id"]))].append(item)
    for v in grouped.values():
        v.sort(key=lambda x: x["step_id"])
    return grouped


def parse_actions(text):
    pattern = "|".join(re.escape(a) for a in ACTIONS2IDX)
    return [item for m in re.findall(pattern, text) for item in ACTIONS2IDX[m]]


def classify_output(text):
    if bool(re.search(r"\d", text)):
        coords = [int(c) for c in re.findall(r"\d+", text)]
        pixel = [int(coords[1]), int(coords[0])] if len(coords) >= 2 else None
        return "pixel_goal", None, pixel
    actions = parse_actions(text)
    action = int(actions[0]) if actions else 0
    return "discrete", action, None


def build_messages(instruction, item, all_steps, num_history, is_lookdown=False, prev_llm_output=None):
    """Build conversation messages for S2 inference.
    
    For normal steps (is_lookdown=False): single-turn with instruction + history + current image.
    For look-down steps (is_lookdown=True): two-turn conversation:
      turn 1 = same as normal step (instruction + history + current rgb)
      turn 2 = assistant reply (prev_llm_output) + user sends look-down image
    """
    text = PROMPT_TEMPLATE
    if instruction.endswith("."):
        text = text.replace("<instruction>.", instruction)
    else:
        text = text.replace("<instruction>.", instruction + ".")
    text = text.replace("<instruction>", instruction.strip())

    history = item.get("history_frame_indices") or []
    step_id = item["step_id"]
    if not history:
        if step_id > 0 and num_history > 0:
            history = np.unique(np.linspace(0, step_id - 1, num_history, dtype=np.int32)).tolist()
    history = sorted(history)

    # Build step_id → first manifest entry map for loading rgb images by step_id
    step_map = {}
    for s in all_steps:
        sid = s["step_id"]
        if sid not in step_map:
            step_map[sid] = s

    valid_history = [h for h in history if h in step_map and os.path.exists(step_map[h]["rgb_path"])]
    if valid_history:
        placeholder = (DEFAULT_IMAGE_TOKEN + "\n") * len(valid_history)
        text += f" These are your historical observations: {placeholder}."

    images = [Image.open(step_map[h]["rgb_path"]).convert("RGB") for h in valid_history]
    images.append(Image.open(item["rgb_path"]).convert("RGB"))

    text += f" {CONJUNCTIONS[0]}{DEFAULT_IMAGE_TOKEN}."

    parts = re.split(r"(<image>)", text)
    content = []
    img_id = 0
    for part in parts:
        if not part:
            continue
        if part == DEFAULT_IMAGE_TOKEN:
            content.append({"type": "image", "image": images[img_id]})
            img_id += 1
        else:
            content.append({"type": "text", "text": part})
    messages = [{"role": "user", "content": content}]

    if is_lookdown and prev_llm_output is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": prev_llm_output}]})
        lookdown_img = Image.open(item["lookdown_rgb_path"]).convert("RGB")
        images.append(lookdown_img)
        turn2_text = f" {CONJUNCTIONS[0]}{DEFAULT_IMAGE_TOKEN}."
        turn2_parts = re.split(r"(<image>)", turn2_text)
        turn2_content = []
        for part in turn2_parts:
            if not part:
                continue
            if part == DEFAULT_IMAGE_TOKEN:
                turn2_content.append({"type": "image", "image": lookdown_img})
            else:
                turn2_content.append({"type": "text", "text": part})
        messages.append({"role": "user", "content": turn2_content})

    return messages, images


def pil_to_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def messages_to_openai(messages):
    out = []
    for msg in messages:
        items = []
        for p in msg["content"]:
            if p["type"] == "text":
                items.append({"type": "text", "text": p["text"]})
            elif p["type"] == "image":
                items.append({"type": "image_url", "image_url": {"url": pil_to_data_url(p["image"])}})
        out.append({"role": msg["role"], "content": items})
    return out


def run_vllm(vllm_url, vllm_model, messages, max_new_tokens):
    openai_msgs = messages_to_openai(messages)
    payload = {
        "model": vllm_model,
        "messages": openai_msgs,
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    resp = http_requests.post(f"{vllm_url}/v1/chat/completions", json=payload, timeout=120)
    latency = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    output_text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return output_text, latency, usage


def detect_vllm_model(vllm_url):
    resp = http_requests.get(f"{vllm_url}/v1/models", timeout=5)
    models = resp.json().get("data", [])
    return models[0]["id"] if models else "default"


def main():
    args = parse_args()
    replay = load_manifest(args.manifest, args.base_path)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    vllm_model = args.vllm_model or detect_vllm_model(args.vllm_url)
    print(f"vLLM model: {vllm_model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    print(f"Tokenizer loaded from {args.model_path}")

    # Only test actual S2 decision points — where llm_output changes within an episode
    # Also mark lookdown steps and pair them with their preceding discrete output
    test_items = []
    for ep_key, steps in replay.items():
        prev_llm = None
        prev_discrete_llm = None
        prev_step_id = None
        for s in steps:
            llm = s["baseline_output"]["llm_output"]
            if llm != prev_llm:
                kind = s["baseline_output"]["output_kind"]
                is_lookdown = (kind == "pixel_goal" and prev_step_id == s["step_id"])
                test_items.append({
                    "ep_key": ep_key,
                    "item": s,
                    "steps": steps,
                    "is_lookdown": is_lookdown,
                    "prev_llm": prev_discrete_llm if is_lookdown else None,
                })
                if not is_lookdown:
                    prev_discrete_llm = llm
                prev_step_id = s["step_id"]
                prev_llm = llm
            if len(test_items) >= args.max_steps:
                break
        if len(test_items) >= args.max_steps:
            break
    n_lookdown = sum(1 for t in test_items if t["is_lookdown"])
    print(f"Filtered to {len(test_items)} S2 decision points ({n_lookdown} lookdown) from {sum(len(v) for v in replay.values())} total steps")

    records = []
    counters = {
        "total": 0,
        "text_exact_match": 0,
        "kind_match": 0,
        "action_match": 0,
        "action_total": 0,
        "pixel_match": 0,
        "pixel_total": 0,
        "token_ids_match": 0,
    }
    vllm_latencies = []

    print(f"\nRunning {len(test_items)} steps through vLLM, comparing with HF baseline from manifest...\n")
    for i, test in enumerate(test_items):
        ep_key = test["ep_key"]
        item = test["item"]
        episode_steps = test["steps"]
        is_lookdown = test["is_lookdown"]
        prev_llm = test["prev_llm"]

        step_idx = item["step_id"]
        instruction = item["instruction"]
        baseline = item["baseline_output"]
        hf_text = baseline["llm_output"]
        hf_kind = baseline["output_kind"]
        hf_action = baseline["action"]
        hf_pixel = baseline["pixel_goal"]

        messages, images = build_messages(
            instruction, item, episode_steps, args.num_history,
            is_lookdown=is_lookdown, prev_llm_output=prev_llm,
        )

        vllm_text, vllm_lat, vllm_usage = run_vllm(args.vllm_url, vllm_model, messages, args.max_new_tokens)
        vllm_kind, vllm_action, vllm_pixel = classify_output(vllm_text)

        hf_gen_ids = tokenizer.encode(hf_text, add_special_tokens=False)
        vllm_gen_ids = tokenizer.encode(vllm_text, add_special_tokens=False)
        token_match = hf_gen_ids == vllm_gen_ids

        text_match = hf_text.strip() == vllm_text.strip()
        kind_match = hf_kind == vllm_kind

        counters["total"] += 1
        if text_match:
            counters["text_exact_match"] += 1
        if kind_match:
            counters["kind_match"] += 1
        if token_match:
            counters["token_ids_match"] += 1

        action_match = None
        if hf_kind == "discrete":
            counters["action_total"] += 1
            if vllm_kind == "discrete" and hf_action == vllm_action:
                counters["action_match"] += 1
                action_match = True
            else:
                action_match = False

        pixel_match = None
        if hf_kind == "pixel_goal" and hf_pixel is not None:
            counters["pixel_total"] += 1
            if vllm_kind == "pixel_goal" and vllm_pixel == hf_pixel:
                counters["pixel_match"] += 1
                pixel_match = True
            else:
                pixel_match = False

        vllm_latencies.append(vllm_lat)
        status = "✓" if text_match else ("≈" if kind_match else "✗")
        ld_tag = " [lookdown]" if is_lookdown else ""
        print(
            f"[{i+1:3d}/{len(test_items)}] {status} "
            f"ep={ep_key[1]} step={step_idx:3d}{ld_tag}  "
            f"vllm={vllm_lat:.3f}s  "
            f"kind={hf_kind}→{vllm_kind}  "
            f"hf=\"{hf_text[:35]}\"  vllm=\"{vllm_text[:35]}\""
        )

        records.append({
            "scene_id": item["scene_id"],
            "episode_id": item["episode_id"],
            "step_id": step_idx,
            "is_lookdown": is_lookdown,
            "hf_text": hf_text,
            "vllm_text": vllm_text,
            "text_exact_match": text_match,
            "hf_kind": hf_kind,
            "vllm_kind": vllm_kind,
            "kind_match": kind_match,
            "hf_action": hf_action,
            "vllm_action": vllm_action,
            "action_match": action_match,
            "hf_pixel": hf_pixel,
            "vllm_pixel": vllm_pixel,
            "pixel_match": pixel_match,
            "hf_gen_token_ids": hf_gen_ids,
            "vllm_gen_token_ids": vllm_gen_ids,
            "token_ids_match": token_match,
            "vllm_latency": vllm_lat,
        })

    n = counters["total"]
    summary = {
        "metadata": {
            "manifest": args.manifest,
            "model_path": args.model_path,
            "vllm_url": args.vllm_url,
            "vllm_model": vllm_model,
            "num_history": args.num_history,
            "max_new_tokens": args.max_new_tokens,
            "num_steps": n,
            "hf_source": "manifest baseline_output (previously recorded HF generate results)",
        },
        "equivalence": {
            "text_exact_match_rate": counters["text_exact_match"] / n if n else 0,
            "output_kind_match_rate": counters["kind_match"] / n if n else 0,
            "token_ids_match_rate": counters["token_ids_match"] / n if n else 0,
            "discrete_action_match_rate": counters["action_match"] / counters["action_total"] if counters["action_total"] else None,
            "pixel_goal_match_rate": counters["pixel_match"] / counters["pixel_total"] if counters["pixel_total"] else None,
            "counts": {
                "text_exact_match": f"{counters['text_exact_match']}/{n}",
                "kind_match": f"{counters['kind_match']}/{n}",
                "token_ids_match": f"{counters['token_ids_match']}/{n}",
                "action_match": f"{counters['action_match']}/{counters['action_total']}",
                "pixel_match": f"{counters['pixel_match']}/{counters['pixel_total']}",
            },
        },
        "latency": {
            "vllm_mean": float(np.mean(vllm_latencies)) if vllm_latencies else 0,
            "vllm_p50": float(np.median(vllm_latencies)) if vllm_latencies else 0,
            "vllm_p95": float(np.percentile(vllm_latencies, 95)) if vllm_latencies else 0,
        },
    }

    mismatches = [r for r in records if not r["text_exact_match"]]
    summary["mismatches_sample"] = mismatches[:10]

    print("\n" + "=" * 70)
    print("PHASE 0 — S2 EQUIVALENCE SUMMARY")
    print("=" * 70)
    print(f"  Steps tested:          {n}")
    print(f"  Text exact match:      {counters['text_exact_match']}/{n} ({summary['equivalence']['text_exact_match_rate']:.1%})")
    print(f"  Output kind match:     {counters['kind_match']}/{n} ({summary['equivalence']['output_kind_match_rate']:.1%})")
    print(f"  Token IDs match:       {counters['token_ids_match']}/{n} ({summary['equivalence']['token_ids_match_rate']:.1%})")
    if counters["action_total"]:
        print(f"  Discrete action match: {counters['action_match']}/{counters['action_total']} ({counters['action_match']/counters['action_total']:.1%})")
    if counters["pixel_total"]:
        print(f"  Pixel goal match:      {counters['pixel_match']}/{counters['pixel_total']} ({counters['pixel_match']/counters['pixel_total']:.1%})")
    print(f"\n  vLLM latency mean={summary['latency']['vllm_mean']:.3f}s  p50={summary['latency']['vllm_p50']:.3f}s  p95={summary['latency']['vllm_p95']:.3f}s")
    if mismatches:
        print(f"\n  First mismatch example:")
        m = mismatches[0]
        print(f"    ep={m['episode_id']} step={m['step_id']}")
        print(f"    HF:   \"{m['hf_text']}\"")
        print(f"    vLLM: \"{m['vllm_text']}\"")
    print("=" * 70)

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    details_path = args.output.replace(".json", "_details.json")
    with open(details_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\nSummary → {args.output}")
    print(f"Details → {details_path}")


if __name__ == "__main__":
    main()

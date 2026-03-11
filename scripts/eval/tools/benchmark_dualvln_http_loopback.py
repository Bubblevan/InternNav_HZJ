import argparse
import io
import json
import os
import statistics
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args():
    parser = argparse.ArgumentParser(description="In-process HTTP loopback benchmark for DualVLN replay_v2.")
    parser.add_argument("--manifest", required=True, help="Replay manifest jsonl path, preferably replay_subset_v2")
    parser.add_argument("--model-path", required=True, help="DualVLN checkpoint path")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--output", required=True, help="Summary json output path")
    parser.add_argument("--details-output", default=None, help="Optional per-step details jsonl path")
    parser.add_argument("--base-path", default=None, help="Optional base path to replace ./logs/ in manifest paths")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on replay steps")
    parser.add_argument("--resize-w", type=int, default=384, help="Agent RGB resize width")
    parser.add_argument("--resize-h", type=int, default=384, help="Agent RGB resize height")
    parser.add_argument("--num-history", type=int, default=8, help="Agent history length")
    parser.add_argument("--plan-step-gap", type=int, default=4, help="Realworld agent plan step gap")
    parser.add_argument("--camera-height", type=float, default=0.4, help="Camera height for agent intrinsic")
    parser.add_argument("--use-recorded-lookdown", action="store_true", help="Use recorded lookdown RGB/depth if present")
    parser.add_argument("--verbose-every", type=int, default=20, help="Print one timing line every N replay steps")
    return parser.parse_args()


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


def default_details_output(output_path):
    stem, _ = os.path.splitext(output_path)
    return f"{stem}.jsonl"


def _rewrite_path(path, base_path):
    if base_path is None:
        return os.path.abspath(path)
    base_path = base_path.rstrip("/") + "/"
    if path.startswith("./logs/"):
        path = path.replace("./logs/", base_path, 1)
    return os.path.abspath(path)


def load_manifest(path, base_path=None):
    grouped = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            for field in ("rgb_path", "depth_path", "lookdown_rgb_path", "lookdown_depth_path"):
                if field in item and item[field]:
                    item[field] = _rewrite_path(item[field], base_path)
            grouped[(item["scene_id"], int(item["episode_id"]))].append(item)
    for items in grouped.values():
        items.sort(key=lambda x: (int(x.get("record_id", -1)), int(x.get("step_id", 0))))
    return grouped


def count_total_steps(replay_steps):
    return sum(len(v) for v in replay_steps.values())


def encode_rgb_jpeg(path):
    image = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def encode_depth_png(path):
    depth = np.load(path)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    depth = np.clip(depth.astype(np.float32), 0.0, None)
    depth_mm = np.clip(depth * 10000.0, 0.0, 65535.0).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(depth_mm, mode="I;16").save(buf, format="PNG")
    return buf.getvalue()


class LocalHTTPDualLoopback:
    def __init__(self, args):
        agent_args = SimpleNamespace(
            device=args.device,
            model_path=args.model_path,
            resize_w=args.resize_w,
            resize_h=args.resize_h,
            num_history=args.num_history,
            plan_step_gap=args.plan_step_gap,
        )
        self.camera_intrinsic = np.array(
            [[386.5, 0.0, 328.9, 0.0], [0.0, 386.5, 244.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        load_start = time.perf_counter()
        self.agent = InternVLAN1AsyncAgent(agent_args)
        warm_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        warm_depth = np.zeros((480, 640), dtype=np.float32)
        self.agent.step(warm_rgb, warm_depth, np.eye(4, dtype=np.float32), "hello", intrinsic=self.camera_intrinsic)
        self.agent.reset()
        self.cold_start_seconds = time.perf_counter() - load_start
        self.use_recorded_lookdown = bool(args.use_recorded_lookdown)

    def reset(self):
        self.agent.reset()

    def _decode_request(self, rgb_bytes, depth_bytes):
        image = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
        rgb = np.asarray(image)

        depth_image = Image.open(io.BytesIO(depth_bytes)).convert("I")
        depth = np.asarray(depth_image).astype(np.float32) / 10000.0
        return rgb, depth

    def _build_response(self, dual_sys_output):
        if dual_sys_output.output_action is not None:
            return {
                "output_kind": "discrete",
                "discrete_action": [int(a) for a in dual_sys_output.output_action],
                "pixel_goal": None,
                "trajectory": None,
            }
        trajectory = dual_sys_output.output_trajectory.tolist() if dual_sys_output.output_trajectory is not None else []
        pixel_goal = dual_sys_output.output_pixel
        if pixel_goal is not None:
            pixel_goal = [int(pixel_goal[0]), int(pixel_goal[1])]
        return {
            "output_kind": "pixel_goal",
            "discrete_action": None,
            "pixel_goal": pixel_goal,
            "trajectory": trajectory,
        }

    def handle(self, rgb_bytes, depth_bytes, instruction, lookdown_rgb_bytes=None, lookdown_depth_bytes=None):
        t_decode_start = time.perf_counter()
        rgb, depth = self._decode_request(rgb_bytes, depth_bytes)
        decode_seconds = time.perf_counter() - t_decode_start

        t_model_start = time.perf_counter()
        dual_sys_output = self.agent.step(
            rgb,
            depth,
            np.eye(4, dtype=np.float32),
            instruction,
            intrinsic=self.camera_intrinsic,
            look_down=False,
        )

        lookdown_used = False
        if dual_sys_output.output_action is not None and dual_sys_output.output_action == [5]:
            lookdown_used = True
            if self.use_recorded_lookdown and lookdown_rgb_bytes is not None and lookdown_depth_bytes is not None:
                look_rgb, look_depth = self._decode_request(lookdown_rgb_bytes, lookdown_depth_bytes)
            else:
                look_rgb, look_depth = rgb, depth
            dual_sys_output = self.agent.step(
                look_rgb,
                look_depth,
                np.eye(4, dtype=np.float32),
                instruction,
                intrinsic=self.camera_intrinsic,
                look_down=True,
            )
        model_seconds = time.perf_counter() - t_model_start

        return self._build_response(dual_sys_output), {
            "decode_seconds": decode_seconds,
            "model_seconds": model_seconds,
            "lookdown_used": lookdown_used,
        }


def main():
    args = parse_args()
    replay_steps = load_manifest(args.manifest, args.base_path)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    details_output = args.details_output or default_details_output(args.output)
    total_planned_steps = count_total_steps(replay_steps)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    server = LocalHTTPDualLoopback(args)

    total_latencies = []
    encode_latencies = []
    decode_latencies = []
    model_latencies = []
    discrete_total_latencies = []
    pixel_total_latencies = []
    kind_matches = 0
    kind_total = 0
    discrete_action_matches = 0
    discrete_action_total = 0
    pixel_goal_l2_errors = []
    lookdown_count = 0
    records = []

    with open(details_output, "w", encoding="utf-8") as detail_f:
        progress = tqdm(total=total_planned_steps, desc="HTTP Loopback", dynamic_ncols=True)
        for _, episode in replay_steps.items():
            server.reset()
            for item in episode:
                if args.max_steps is not None and len(records) >= args.max_steps:
                    break

                step_start = time.perf_counter()
                encode_start = time.perf_counter()
                rgb_bytes = encode_rgb_jpeg(item["rgb_path"])
                depth_bytes = encode_depth_png(item["depth_path"])
                lookdown_rgb_bytes = None
                lookdown_depth_bytes = None
                if args.use_recorded_lookdown and os.path.exists(item["lookdown_rgb_path"]) and os.path.exists(
                    item["lookdown_depth_path"]
                ):
                    lookdown_rgb_bytes = encode_rgb_jpeg(item["lookdown_rgb_path"])
                    lookdown_depth_bytes = encode_depth_png(item["lookdown_depth_path"])
                encode_seconds = time.perf_counter() - encode_start

                response, timings = server.handle(
                    rgb_bytes=rgb_bytes,
                    depth_bytes=depth_bytes,
                    instruction=item["instruction"],
                    lookdown_rgb_bytes=lookdown_rgb_bytes,
                    lookdown_depth_bytes=lookdown_depth_bytes,
                )
                total_seconds = time.perf_counter() - step_start

                baseline = item["baseline_output"]
                baseline_kind = baseline["output_kind"]
                predicted_kind = response["output_kind"]

                kind_total += 1
                if predicted_kind == baseline_kind:
                    kind_matches += 1

                if predicted_kind == "discrete":
                    discrete_total_latencies.append(total_seconds)
                    predicted_action = int(response["discrete_action"][0]) if response["discrete_action"] else 0
                    if baseline_kind == "discrete":
                        discrete_action_total += 1
                        if predicted_action == int(baseline["action"]):
                            discrete_action_matches += 1
                else:
                    pixel_total_latencies.append(total_seconds)
                    if response["pixel_goal"] is not None and baseline.get("pixel_goal") is not None:
                        dx = float(response["pixel_goal"][0] - baseline["pixel_goal"][0])
                        dy = float(response["pixel_goal"][1] - baseline["pixel_goal"][1])
                        pixel_goal_l2_errors.append(float(np.sqrt(dx * dx + dy * dy)))

                if timings["lookdown_used"]:
                    lookdown_count += 1

                encode_latencies.append(encode_seconds)
                decode_latencies.append(timings["decode_seconds"])
                model_latencies.append(timings["model_seconds"])
                total_latencies.append(total_seconds)

                record = {
                    "scene_id": item["scene_id"],
                    "episode_id": item["episode_id"],
                    "record_id": item.get("record_id"),
                    "step_id": item["step_id"],
                    "is_new_s2_decision": item.get("is_new_s2_decision"),
                    "baseline_output_kind": baseline_kind,
                    "predicted_output_kind": predicted_kind,
                    "encode_seconds": encode_seconds,
                    "decode_seconds": timings["decode_seconds"],
                    "model_seconds": timings["model_seconds"],
                    "total_seconds": total_seconds,
                    "lookdown_used": timings["lookdown_used"],
                    "predicted_discrete_action": response["discrete_action"],
                    "predicted_pixel_goal": response["pixel_goal"],
                    "baseline_action": baseline["action"],
                    "baseline_pixel_goal": baseline["pixel_goal"],
                }
                records.append(record)
                detail_f.write(json.dumps(record) + "\n")
                progress.update(1)

                if args.verbose_every > 0 and len(records) % args.verbose_every == 0:
                    print(
                        f"[step {len(records)}] kind={predicted_kind} total={total_seconds:.3f}s "
                        f"encode={encode_seconds:.3f}s decode={timings['decode_seconds']:.3f}s "
                        f"model={timings['model_seconds']:.3f}s lookdown={timings['lookdown_used']}",
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
            "num_history": args.num_history,
            "plan_step_gap": args.plan_step_gap,
            "use_recorded_lookdown": bool(args.use_recorded_lookdown),
            "num_steps": len(records),
            "num_episodes": len(replay_steps),
            "details_output": details_output,
        },
        "startup": {
            "cold_start_load_seconds": server.cold_start_seconds,
            "gpu_peak_memory_mb": (
                float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
            ),
        },
        "latency": {
            "total_step": summarize_latencies(total_latencies),
            "client_encode": summarize_latencies(encode_latencies),
            "server_decode": summarize_latencies(decode_latencies),
            "server_model": summarize_latencies(model_latencies),
            "discrete_total_step": summarize_latencies(discrete_total_latencies),
            "pixel_goal_total_step": summarize_latencies(pixel_total_latencies),
        },
        "consistency": {
            "output_kind_match_rate": (kind_matches / kind_total) if kind_total else 0.0,
            "discrete_action_match_rate": (
                discrete_action_matches / discrete_action_total if discrete_action_total else 0.0
            ),
            "pixel_goal_l2_mean": statistics.mean(pixel_goal_l2_errors) if pixel_goal_l2_errors else 0.0,
            "pixel_goal_l2_p50": percentile(pixel_goal_l2_errors, 50),
            "pixel_goal_l2_p95": percentile(pixel_goal_l2_errors, 95),
        },
        "breakdown": {
            "lookdown_used_steps": lookdown_count,
        },
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

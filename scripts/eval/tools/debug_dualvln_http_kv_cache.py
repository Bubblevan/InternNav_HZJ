import argparse
import json
import os
import time

from benchmark_dualvln_http_loopback import (
    LocalHTTPDualLoopback,
    count_total_steps,
    default_details_output,
    encode_depth_png,
    encode_rgb_jpeg,
    load_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Stop-on-first-exception debug runner for DualVLN HTTP KV cache.")
    parser.add_argument("--manifest", required=True, help="Replay manifest jsonl path, preferably replay_subset_v2")
    parser.add_argument("--model-path", required=True, help="DualVLN checkpoint path")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--output", required=True, help="Summary json output path")
    parser.add_argument("--details-output", default=None, help="Optional per-step details jsonl output path")
    parser.add_argument("--base-path", default=None, help="Optional base path to replace ./logs/ in manifest paths")
    parser.add_argument("--max-steps", type=int, default=200, help="Safety cap on total replay steps")
    parser.add_argument("--resize-w", type=int, default=384, help="Agent RGB resize width")
    parser.add_argument("--resize-h", type=int, default=384, help="Agent RGB resize height")
    parser.add_argument("--num-history", type=int, default=8, help="Agent history length")
    parser.add_argument("--plan-step-gap", type=int, default=4, help="Realworld agent plan step gap")
    parser.add_argument(
        "--kv-cache-mode",
        choices=["disabled", "lookdown_experimental"],
        default="lookdown_experimental",
        help="KV cache experiment mode forwarded to the realworld agent",
    )
    parser.add_argument("--kv-cache-debug", action="store_true", help="Enable verbose KV cache debug logging")
    parser.add_argument("--use-recorded-lookdown", action="store_true", help="Use recorded lookdown RGB/depth if present")
    return parser.parse_args()


def main():
    args = parse_args()
    replay_steps = load_manifest(args.manifest, args.base_path)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    details_output = args.details_output or default_details_output(args.output)
    total_planned_steps = count_total_steps(replay_steps)

    server = LocalHTTPDualLoopback(args)

    debug_summary = {
        "metadata": {
            "manifest": args.manifest,
            "model_path": args.model_path,
            "device": args.device,
            "num_history": args.num_history,
            "plan_step_gap": args.plan_step_gap,
            "kv_cache_mode": args.kv_cache_mode,
            "use_recorded_lookdown": bool(args.use_recorded_lookdown),
            "max_steps": args.max_steps,
            "total_planned_steps": total_planned_steps,
        },
        "steps_processed": 0,
        "first_exception": None,
        "kv_cache_stats": {},
    }

    with open(details_output, "w", encoding="utf-8") as detail_f:
        for _, episode in replay_steps.items():
            server.reset()
            for item in episode:
                if debug_summary["steps_processed"] >= args.max_steps:
                    break

                rgb_bytes = encode_rgb_jpeg(item["rgb_path"])
                depth_bytes = encode_depth_png(item["depth_path"])
                lookdown_rgb_bytes = None
                lookdown_depth_bytes = None
                if args.use_recorded_lookdown and os.path.exists(item["lookdown_rgb_path"]) and os.path.exists(
                    item["lookdown_depth_path"]
                ):
                    lookdown_rgb_bytes = encode_rgb_jpeg(item["lookdown_rgb_path"])
                    lookdown_depth_bytes = encode_depth_png(item["lookdown_depth_path"])

                step_start = time.perf_counter()
                response, timings = server.handle(
                    rgb_bytes=rgb_bytes,
                    depth_bytes=depth_bytes,
                    instruction=item["instruction"],
                    lookdown_rgb_bytes=lookdown_rgb_bytes,
                    lookdown_depth_bytes=lookdown_depth_bytes,
                )
                kv_event = server.get_last_kv_cache_event()
                record = {
                    "scene_id": item["scene_id"],
                    "episode_id": item["episode_id"],
                    "record_id": item.get("record_id"),
                    "step_id": item["step_id"],
                    "is_new_s2_decision": item.get("is_new_s2_decision"),
                    "response_kind": response["output_kind"],
                    "lookdown_used": timings["lookdown_used"],
                    "model_seconds": timings["model_seconds"],
                    "total_seconds": time.perf_counter() - step_start,
                    "kv_cache_event": kv_event,
                }
                detail_f.write(json.dumps(record) + "\n")
                debug_summary["steps_processed"] += 1

                if kv_event.get("exception_type"):
                    debug_summary["first_exception"] = {
                        "scene_id": item["scene_id"],
                        "episode_id": item["episode_id"],
                        "record_id": item.get("record_id"),
                        "step_id": item["step_id"],
                        "is_new_s2_decision": item.get("is_new_s2_decision"),
                        "lookdown_used": timings["lookdown_used"],
                        "kv_cache_event": kv_event,
                    }
                    debug_summary["kv_cache_stats"] = server.get_kv_cache_stats()
                    with open(args.output, "w", encoding="utf-8") as f:
                        json.dump(debug_summary, f, indent=2)
                    return

            if debug_summary["steps_processed"] >= args.max_steps:
                break

    debug_summary["kv_cache_stats"] = server.get_kv_cache_stats()
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(debug_summary, f, indent=2)


if __name__ == "__main__":
    main()

import argparse
import copy
import json
import os
import sys
from pathlib import Path

from internnav.evaluator import Evaluator
from scripts.eval.eval import load_eval_cfg


DEFAULT_NUM_HISTORY = [4, 2, 1, 0]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Habitat closed-loop DualVLN sweeps over num_history.")
    parser.add_argument(
        "--dataset",
        choices=["mini", "full"],
        default="mini",
        help="mini uses val_unseen_mini; full uses the full val_unseen dataset.",
    )
    parser.add_argument(
        "--num-history-values",
        nargs="+",
        type=int,
        default=DEFAULT_NUM_HISTORY,
        help="List of num_history values to sweep.",
    )
    parser.add_argument(
        "--base-config",
        default="scripts/eval/configs/habitat_dual_system_cfg.py",
        help="Base eval config path.",
    )
    parser.add_argument(
        "--output-root",
        default="./logs/habitat/closed_loop_num_history_sweeps",
        help="Root directory for per-run outputs.",
    )
    parser.add_argument(
        "--max-eval-episodes",
        type=int,
        default=8,
        help="Episode cap for each run. Use 0 or a negative value to disable the cap.",
    )
    parser.add_argument(
        "--replay-num-episodes",
        type=int,
        default=8,
        help="How many episodes to export into replay_subset/replay_subset_v2 for each run.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Whether to save Habitat videos for each run.",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=None,
        help="Optional override for max steps per episode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved run plan without executing evaluation.",
    )
    return parser.parse_args()


def build_run_cfg(base_config_path, dataset, num_history, output_root, max_eval_episodes, replay_num_episodes, save_video,
                  max_steps_per_episode):
    cfg = load_eval_cfg(base_config_path, attr_name="eval_cfg")
    cfg = copy.deepcopy(cfg)

    cfg.agent.model_settings["num_history"] = int(num_history)
    cfg.eval_settings["save_video"] = bool(save_video)
    if max_steps_per_episode is not None:
        cfg.eval_settings["max_steps_per_episode"] = int(max_steps_per_episode)

    if dataset == "mini":
        cfg.eval_settings["dataset_path_override"] = "data/vln_ce/raw_data/r2r/val_unseen_mini/val_unseen_mini.json.gz"
        cfg.eval_settings["output_path"] = os.path.join(output_root, "mini", f"num_history_{num_history}")
    elif dataset == "full":
        cfg.eval_settings["dataset_path_override"] = None
        cfg.eval_settings["output_path"] = os.path.join(output_root, "full", f"num_history_{num_history}")
    else:
        raise ValueError(f"Unsupported dataset mode: {dataset}")

    cfg.eval_settings["allowed_scene_ids"] = []
    cfg.eval_settings["allowed_episode_ids"] = []
    cfg.eval_settings["max_eval_episodes"] = None if max_eval_episodes is None else int(max_eval_episodes)
    cfg.eval_settings["replay_num_episodes"] = int(replay_num_episodes)
    cfg.eval_settings["export_replay_subset"] = int(replay_num_episodes) > 0
    cfg.eval_settings["scenes_dir_override"] = "data/scene_data"
    return cfg


def run_eval(cfg):
    evaluator = Evaluator.init(cfg)
    evaluator.eval()


def summarize_outputs(output_root, dataset, num_history_values):
    rows = []
    for value in num_history_values:
        run_dir = Path(output_root) / dataset / f"num_history_{value}"
        result_path = run_dir / "result.json"
        runtime_path = run_dir / "runtime_summary_rank0.json"
        result = {}
        runtime = {}
        if result_path.exists():
            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)
        if runtime_path.exists():
            with open(runtime_path, "r", encoding="utf-8") as f:
                runtime = json.load(f)
        rows.append(
            {
                "num_history": value,
                "output_path": str(run_dir),
                "success": result.get("sucs_all"),
                "spl": result.get("spls_all"),
                "oracle_success": result.get("oss_all"),
                "navigation_error": result.get("nes_all"),
                "avg_step_wall_clock_seconds": runtime.get("avg_step_wall_clock_seconds"),
                "s2_avg_seconds": runtime.get("s2_avg_seconds"),
                "s1_avg_seconds": runtime.get("s1_avg_seconds"),
                "pixel_goal_ratio": runtime.get("pixel_goal_ratio"),
                "discrete_ratio": runtime.get("discrete_ratio"),
                "avg_s1_steps_per_cycle": runtime.get("avg_s1_steps_per_cycle"),
            }
        )
    return rows


def write_summary(path, rows):
    import csv

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.max_eval_episodes is not None and args.max_eval_episodes <= 0:
        max_eval_episodes = None
    else:
        max_eval_episodes = args.max_eval_episodes

    output_root = os.path.abspath(args.output_root)
    planned = []
    for value in args.num_history_values:
        cfg = build_run_cfg(
            base_config_path=args.base_config,
            dataset=args.dataset,
            num_history=value,
            output_root=output_root,
            max_eval_episodes=max_eval_episodes,
            replay_num_episodes=args.replay_num_episodes,
            save_video=args.save_video,
            max_steps_per_episode=args.max_steps_per_episode,
        )
        planned.append(
            {
                "num_history": value,
                "output_path": cfg.eval_settings["output_path"],
                "dataset_path_override": cfg.eval_settings["dataset_path_override"],
                "max_eval_episodes": cfg.eval_settings["max_eval_episodes"],
            }
        )

    print(json.dumps({"dataset": args.dataset, "runs": planned}, indent=2), flush=True)
    if args.dry_run:
        return

    for value in args.num_history_values:
        cfg = build_run_cfg(
            base_config_path=args.base_config,
            dataset=args.dataset,
            num_history=value,
            output_root=output_root,
            max_eval_episodes=max_eval_episodes,
            replay_num_episodes=args.replay_num_episodes,
            save_video=args.save_video,
            max_steps_per_episode=args.max_steps_per_episode,
        )
        print(f"[closed-loop] dataset={args.dataset} num_history={value} output={cfg.eval_settings['output_path']}",
              flush=True)
        run_eval(cfg)

    rows = summarize_outputs(output_root, args.dataset, args.num_history_values)
    summary_path = os.path.join(output_root, args.dataset, "comparison_num_history_closed_loop.csv")
    write_summary(summary_path, rows)
    print(f"Wrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()

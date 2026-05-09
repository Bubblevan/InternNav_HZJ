import argparse
import glob
import importlib.util
import json
import os
import sys
from pathlib import Path


def load_eval_cfg(config_path, attr_name="eval_cfg"):
    spec = importlib.util.spec_from_file_location("eval_config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["eval_config_module"] = config_module
    spec.loader.exec_module(config_module)
    return getattr(config_module, attr_name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a per-episode control timeline / gantt chart from timeline_trace jsonl."
    )
    parser.add_argument("--config", required=True, help="Eval config path.")
    parser.add_argument("--trace-jsonl", default=None, help="Optional explicit trace jsonl path.")
    parser.add_argument("--output", default=None, help="Optional explicit output png path.")
    parser.add_argument("--start-s", type=float, default=None, help="Window start time in seconds.")
    parser.add_argument("--window-s", type=float, default=None, help="Window width in seconds.")
    return parser.parse_args()


def resolve_trace_path(eval_cfg, explicit_trace):
    if explicit_trace:
        return Path(explicit_trace).resolve()
    eval_settings = eval_cfg.eval_settings
    output_path = Path(eval_settings["output_path"]).resolve()
    dump_path = eval_settings.get("timeline_trace_dump_path")
    if dump_path:
        dump_path = str(dump_path)
        if "{rank}" in dump_path:
            dump_path = dump_path.format(rank=0)
        elif dump_path.endswith(".jsonl"):
            dump_path = dump_path[:-6] + "_rank0.jsonl"
        else:
            dump_path = os.path.join(dump_path, "timeline_trace_rank0.jsonl")
        return Path(dump_path).resolve()
    matches = sorted(output_path.glob("timeline_trace_rank*.jsonl"))
    if not matches:
        raise FileNotFoundError(
            f"Could not locate timeline trace under {output_path}. "
            "Please rerun eval with timeline_trace_enabled=True."
        )
    return matches[0]


def load_rows(trace_path, scene_id, episode_id):
    rows = []
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("scene_id")) != str(scene_id):
                continue
            if int(row.get("episode_id")) != int(episode_id):
                continue
            rows.append(row)
    if not rows:
        raise ValueError(
            f"No timeline rows for scene_id={scene_id!r}, episode_id={episode_id} in {trace_path}."
        )
    rows.sort(key=lambda row: float(row["t_rel_ms"]))
    return rows


def pair_intervals(rows, start_type, end_type, key_field, lane_name):
    pending = {}
    out = []
    for row in rows:
        event_type = row.get("event_type")
        key = row.get(key_field)
        if key is None:
            continue
        if event_type == start_type:
            pending[key] = row
        elif event_type == end_type and key in pending:
            start_row = pending.pop(key)
            out.append(
                {
                    "lane": lane_name,
                    "start_ms": float(start_row["t_rel_ms"]),
                    "end_ms": float(row["t_rel_ms"]),
                    "duration_ms": float(row["t_rel_ms"]) - float(start_row["t_rel_ms"]),
                    "start_row": start_row,
                    "end_row": row,
                }
            )
    return out


def build_wait_intervals(rows):
    responses = [row for row in rows if row.get("event_type") == "s2_response_received"]
    dispatches = [row for row in rows if row.get("event_type") == "action_dispatch_start"]
    out = []
    dispatch_idx = 0
    for response in responses:
        response_t = float(response["t_rel_ms"])
        while dispatch_idx < len(dispatches) and float(dispatches[dispatch_idx]["t_rel_ms"]) < response_t:
            dispatch_idx += 1
        if dispatch_idx >= len(dispatches):
            break
        dispatch = dispatches[dispatch_idx]
        dispatch_t = float(dispatch["t_rel_ms"])
        if dispatch_t >= response_t:
            out.append(
                {
                    "lane": "Waiting Gap",
                    "start_ms": response_t,
                    "end_ms": dispatch_t,
                    "duration_ms": dispatch_t - response_t,
                    "start_row": response,
                    "end_row": dispatch,
                }
            )
    return out


def plot_timeline(rows, output_path, title, start_s, window_s):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s2_intervals = pair_intervals(rows, "s2_request_sent", "s2_response_received", "request_seq", "S2 Request")
    s1_intervals = pair_intervals(rows, "s1_rollout_start", "s1_rollout_end", "rollout_seq", "S1 Rollout")
    action_intervals = pair_intervals(
        rows, "action_dispatch_start", "action_dispatch_end", "action_seq_id", "Action Dispatch"
    )
    wait_intervals = build_wait_intervals(rows)

    window_start_ms = float(start_s) * 1000.0
    window_end_ms = window_start_ms + float(window_s) * 1000.0

    def in_window(interval):
        return interval["end_ms"] >= window_start_ms and interval["start_ms"] <= window_end_ms

    s2_intervals = [it for it in s2_intervals if in_window(it)]
    s1_intervals = [it for it in s1_intervals if in_window(it)]
    action_intervals = [it for it in action_intervals if in_window(it)]
    wait_intervals = [it for it in wait_intervals if in_window(it)]

    lane_order = ["S2 Request", "S1 Rollout", "Action Dispatch", "Waiting Gap"]
    lane_y = {name: idx for idx, name in enumerate(reversed(lane_order))}
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=180)

    def draw_intervals(intervals, facecolor, edgecolor, alpha=0.85):
        for item in intervals:
            start = max(item["start_ms"], window_start_ms) / 1000.0
            end = min(item["end_ms"], window_end_ms) / 1000.0
            width = max(end - start, 1e-6)
            y = lane_y[item["lane"]]
            ax.broken_barh([(start, width)], (y - 0.35, 0.7), facecolors=facecolor, edgecolors=edgecolor, alpha=alpha)

    draw_intervals(s2_intervals, facecolor="#4C78A8", edgecolor="#2C4D6D")
    draw_intervals(s1_intervals, facecolor="#F58518", edgecolor="#A85A10")
    draw_intervals(action_intervals, facecolor="#54A24B", edgecolor="#2D6B28")
    draw_intervals(wait_intervals, facecolor="#B279A2", edgecolor="#7C5071", alpha=0.55)

    for item in s2_intervals:
        mid = (max(item["start_ms"], window_start_ms) + min(item["end_ms"], window_end_ms)) / 2000.0
        row = item["end_row"]
        label = "follow-up" if row.get("is_lookdown_followup") else "primary"
        if row.get("same_request_continuation_used"):
            label += " cont-hit"
        else:
            label += " cont-miss"
        ax.text(mid, lane_y["S2 Request"], label, ha="center", va="center", fontsize=7, color="white")

    event_styles = {
        "s2_output_ready": ("#111111", "--", 1.0),
        "pixel_goal_ready": ("#E45756", "-.", 1.2),
        "predicted_goal_ready": ("#EECA3B", ":", 1.4),
        "episode_end": ("#000000", "-", 1.5),
    }
    for row in rows:
        t_s = float(row["t_rel_ms"]) / 1000.0
        if t_s < window_start_ms / 1000.0 or t_s > window_end_ms / 1000.0:
            continue
        event_type = row.get("event_type")
        if event_type not in event_styles:
            continue
        color, linestyle, linewidth = event_styles[event_type]
        ax.axvline(t_s, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.85)
        if event_type == "pixel_goal_ready":
            goal = row.get("pixel_goal")
            if goal is not None:
                ax.text(t_s, lane_y["S2 Request"] + 0.42, f"goal {goal}", rotation=90, va="bottom", ha="right", fontsize=6)
        elif event_type == "episode_end":
            ax.text(t_s, lane_y["Waiting Gap"] - 0.5, "STOP/END", rotation=90, va="bottom", ha="right", fontsize=7)

    ax.set_xlim(window_start_ms / 1000.0, window_end_ms / 1000.0)
    ax.set_ylim(-0.8, len(lane_order) - 0.2)
    ax.set_yticks([lane_y[name] for name in lane_order])
    ax.set_yticklabels(lane_order)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    args = parse_args()
    eval_cfg = load_eval_cfg(args.config, attr_name="eval_cfg")
    eval_settings = eval_cfg.eval_settings
    scene_id = eval_settings.get("timeline_trace_scene_id")
    episode_id = eval_settings.get("timeline_trace_episode_id")
    if scene_id is None or episode_id is None:
        raise ValueError(
            "Config must set eval_cfg.eval_settings['timeline_trace_scene_id'] and "
            "eval_cfg.eval_settings['timeline_trace_episode_id']."
        )
    trace_path = resolve_trace_path(eval_cfg, args.trace_jsonl)
    rows = load_rows(trace_path, scene_id, episode_id)
    start_s = (
        float(args.start_s)
        if args.start_s is not None
        else float(eval_settings.get("timeline_plot_start_s", 0.0))
    )
    window_s = (
        float(args.window_s)
        if args.window_s is not None
        else float(eval_settings.get("timeline_plot_window_s", 20.0))
    )
    default_output = Path(eval_settings["output_path"]).resolve() / (
        f"timeline_gantt_{scene_id}_{int(episode_id):04d}.png"
    )
    output_path = Path(args.output).resolve() if args.output else default_output
    title = eval_settings.get(
        "timeline_plot_title",
        f"Episode Timeline: {scene_id} / {int(episode_id):04d}",
    )
    plot_timeline(rows, output_path, title, start_s, window_s)
    print(f"Wrote timeline gantt to: {output_path}")


if __name__ == "__main__":
    main()

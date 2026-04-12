import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize history probe jsonl and inventory jsonl into json/csv/png/md artifacts."
    )
    parser.add_argument("--input-jsonl", required=True, help="Path to history_probe_rank*.jsonl")
    parser.add_argument(
        "--inventory-jsonl",
        default=None,
        help=(
            "Optional inventory jsonl path. Defaults to a sibling "
            "history_probe_inventory_rank*.jsonl when present."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for summary outputs. Defaults to <input stem>_summary beside the jsonl.",
    )
    parser.add_argument("--title", default="History Probe Summary")
    return parser.parse_args()


def load_rows(path: Optional[Path]):
    if path is None or not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def safe_mean(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def safe_rate(num, den):
    if den <= 0:
        return None
    return float(num) / float(den)


def summarize_probe_output(row, prefix="probe"):
    output_type = row.get(f"{prefix}_output_type")
    if output_type == "pixel_goal":
        pixel_goal = row.get(f"{prefix}_pixel_goal")
        if pixel_goal is not None:
            return f"pixel_goal:{pixel_goal[0]},{pixel_goal[1]}"
        return "pixel_goal"
    action_seq = row.get(f"{prefix}_action_seq")
    if action_seq:
        if list(action_seq) == [5]:
            return "lookdown"
        return f"discrete:{','.join(str(int(action)) for action in action_seq)}"
    llm_output = str(row.get(f"{prefix}_llm_output") or "").strip()
    if llm_output:
        return f"text:{llm_output}"
    return str(output_type or "unknown")


def group_probe_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        probe_mode = row.get("probe_mode", "unknown")
        intervention_variant = (
            row.get("intervention_variant")
            or row.get("intervention_type")
            or "strong_null_history_image_replacement"
        )
        history_index = row.get("history_index")
        if history_index is None or int(history_index) < 0:
            continue
        grouped[(probe_mode, intervention_variant, int(history_index))].append(row)
    return grouped


def build_group_rows(rows):
    grouped = group_probe_rows(rows)
    out_rows = []
    for (probe_mode, intervention_variant, history_index), group in sorted(grouped.items()):
        total = len(group)
        valid = [row for row in group if not row.get("probe_error")]
        valid_count = len(valid)
        followup_metrics_enabled = any(
            bool(row.get("history_probe_run_followup_replay"))
            or row.get("followup_replay_attempted") is not None
            or row.get("followup_replay_executed") is not None
            for row in group
        )
        changed_output_counts = Counter(
            summarize_probe_output(row, prefix="probe")
            for row in valid
            if row.get("gateway_action_changed")
        )
        followup_executed = [row for row in valid if row.get("followup_replay_executed") is True]
        followup_executed_count = len(followup_executed)
        changed_followup_output_counts = Counter(
            summarize_probe_output(row, prefix="probe_followup")
            for row in followup_executed
            if row.get("followup_pixel_goal_preserved") is False
            or row.get("followup_pixel_goal_to_discrete") is True
        )
        out_rows.append(
            {
                "probe_mode": probe_mode,
                "intervention_variant": intervention_variant,
                "history_index": history_index,
                "history_probe_target": (
                    group[0].get("history_probe_target") if group else None
                ),
                "intervention_is_strong": (
                    group[0].get("intervention_is_strong") if group else None
                ),
                "total_rows": total,
                "valid_rows": valid_count,
                "error_rows": total - valid_count,
                "error_rate": safe_rate(total - valid_count, total),
                "gateway_action_preserved_rate": safe_rate(
                    sum(1 for row in valid if row.get("gateway_action_preserved")),
                    valid_count,
                ),
                "gateway_action_changed_rate": safe_rate(
                    sum(1 for row in valid if row.get("gateway_action_changed")),
                    valid_count,
                ),
                "first_generated_token_change_rate": safe_rate(
                    sum(1 for row in valid if row.get("first_generated_token_changed")),
                    valid_count,
                ),
                "mean_generated_token_count_delta": safe_mean(
                    [row.get("generated_token_count_delta") for row in valid]
                ),
                "mean_abs_generated_token_count_delta": safe_mean(
                    [row.get("generated_token_count_abs_delta") for row in valid]
                ),
                "mean_probe_wall_time_ms": safe_mean(
                    [row.get("probe_wall_time_ms") for row in valid]
                ),
                "mean_probe_runtime_total_ms": safe_mean(
                    [row.get("probe_runtime_total_ms") for row in valid]
                ),
                "followup_replay_attempted_rows": sum(
                    1 for row in valid if row.get("followup_replay_attempted") is True
                ),
                "followup_replay_executed_rows": followup_executed_count,
                "followup_replay_executed_rate": (
                    safe_rate(followup_executed_count, total)
                    if followup_metrics_enabled
                    else None
                ),
                "conditional_followup_pixel_goal_preserved_rate": (
                    safe_rate(
                        sum(
                            1
                            for row in followup_executed
                            if row.get("followup_pixel_goal_preserved") is True
                        ),
                        followup_executed_count,
                    )
                    if followup_metrics_enabled
                    else None
                ),
                "unconditional_end_to_end_preserved_rate": (
                    safe_rate(
                        sum(
                            1
                            for row in valid
                            if row.get("end_to_end_gateway_and_followup_preserved") is True
                        ),
                        total,
                    )
                    if followup_metrics_enabled
                    else None
                ),
                "mean_followup_pixel_goal_l2_shift": (
                    safe_mean([row.get("followup_pixel_goal_l2_shift") for row in followup_executed])
                    if followup_metrics_enabled
                    else None
                ),
                "changed_output_counts": dict(changed_output_counts),
                "changed_followup_output_counts": dict(changed_followup_output_counts),
            }
        )
    return out_rows


def build_inventory_summary(inventory_rows):
    if not inventory_rows:
        return {
            "inventory_rows": 0,
            "unique_step_ids": 0,
            "primary_call_rows": 0,
            "followup_call_rows": 0,
            "history_primary_rows": 0,
            "gateway_candidate_rows": 0,
            "gateway_with_pixel_goal_followup_rows": 0,
            "selected_gateway_rows": 0,
            "gateway_selected_reason_counts": {},
            "gateway_skipped_reason_counts": {},
            "followup_output_type_counts": {},
        }

    unique_steps = {
        (row["scene_id"], int(row["episode_id"]), int(row["step_id"]))
        for row in inventory_rows
    }
    primary_rows = [
        row for row in inventory_rows if row.get("call_role") == "history_conditioned_primary"
    ]
    followup_rows = [
        row for row in inventory_rows if row.get("call_role") == "lookdown_followup"
    ]
    return {
        "inventory_rows": len(inventory_rows),
        "unique_step_ids": len(unique_steps),
        "primary_call_rows": len(primary_rows),
        "followup_call_rows": len(followup_rows),
        "history_primary_rows": sum(1 for row in primary_rows if row.get("has_history")),
        "gateway_candidate_rows": sum(
            1
            for row in primary_rows
            if row.get("has_history")
            and row.get("baseline_output_type") == "discrete_action"
            and row.get("baseline_is_single_lookdown_action")
        ),
        "gateway_with_pixel_goal_followup_rows": sum(
            1
            for row in primary_rows
            if row.get("has_history")
            and row.get("baseline_output_type") == "discrete_action"
            and row.get("baseline_is_single_lookdown_action")
            and row.get("has_followup_same_step")
            and row.get("followup_has_pixel_goal") is True
        ),
        "selected_gateway_rows": sum(
            1 for row in primary_rows if row.get("gateway_event_selected")
        ),
        "gateway_selected_reason_counts": dict(
            Counter(
                row.get("gateway_selected_reason")
                for row in primary_rows
                if row.get("gateway_selected_reason")
            )
        ),
        "gateway_skipped_reason_counts": dict(
            Counter(
                row.get("gateway_skipped_reason")
                for row in primary_rows
                if row.get("gateway_skipped_reason")
            )
        ),
        "followup_output_type_counts": dict(
            Counter(row.get("baseline_output_type") for row in followup_rows)
        ),
    }


def write_csv(path: Path, rows):
    fieldnames = [
        "probe_mode",
        "intervention_variant",
        "history_index",
        "history_probe_target",
        "intervention_is_strong",
        "total_rows",
        "valid_rows",
        "error_rows",
        "error_rate",
        "gateway_action_preserved_rate",
        "gateway_action_changed_rate",
        "first_generated_token_change_rate",
        "mean_generated_token_count_delta",
        "mean_abs_generated_token_count_delta",
        "mean_probe_wall_time_ms",
        "mean_probe_runtime_total_ms",
        "followup_replay_attempted_rows",
        "followup_replay_executed_rows",
        "followup_replay_executed_rate",
        "conditional_followup_pixel_goal_preserved_rate",
        "unconditional_end_to_end_preserved_rate",
        "mean_followup_pixel_goal_l2_shift",
        "changed_output_counts",
        "changed_followup_output_counts",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["changed_output_counts"] = json.dumps(
                row.get("changed_output_counts") or {},
                ensure_ascii=False,
                sort_keys=True,
            )
            serialized["changed_followup_output_counts"] = json.dumps(
                row.get("changed_followup_output_counts") or {},
                ensure_ascii=False,
                sort_keys=True,
            )
            writer.writerow({key: serialized.get(key) for key in fieldnames})


def maybe_plot(group_rows, output_dir: Path):
    plot_paths = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return plot_paths, "matplotlib unavailable"

    by_mode = defaultdict(list)
    for row in group_rows:
        by_mode[(row["probe_mode"], row["intervention_variant"])].append(row)
    if not by_mode:
        return plot_paths, None

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    for (probe_mode, intervention_variant), rows in sorted(by_mode.items()):
        rows = sorted(rows, key=lambda row: row["history_index"])
        ax.plot(
            [row["history_index"] for row in rows],
            [
                0.0 if row["gateway_action_preserved_rate"] is None else row["gateway_action_preserved_rate"]
                for row in rows
            ],
            marker="o",
            label=f"{probe_mode}:{intervention_variant}",
        )
    ax.set_xlabel("History Index")
    ax.set_ylabel("Gateway Action Preserved Rate")
    ax.set_title("Gateway Probe: Preserve Single LOOKDOWN")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    if len(by_mode) > 1:
        ax.legend()
    preserved_path = output_dir / "gateway_action_preserved_rate.png"
    fig.tight_layout()
    fig.savefig(preserved_path)
    plt.close(fig)
    plot_paths.append(str(preserved_path))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    for (probe_mode, intervention_variant), rows in sorted(by_mode.items()):
        rows = sorted(rows, key=lambda row: row["history_index"])
        ax.plot(
            [row["history_index"] for row in rows],
            [
                0.0 if row["first_generated_token_change_rate"] is None else row["first_generated_token_change_rate"]
                for row in rows
            ],
            marker="o",
            label=f"{probe_mode}:{intervention_variant}",
        )
    ax.set_xlabel("History Index")
    ax.set_ylabel("First Token Changed Rate")
    ax.set_title("Gateway Probe: First Token Change Rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    if len(by_mode) > 1:
        ax.legend()
    token_path = output_dir / "first_generated_token_change_rate.png"
    fig.tight_layout()
    fig.savefig(token_path)
    plt.close(fig)
    plot_paths.append(str(token_path))

    if any(
        row.get("conditional_followup_pixel_goal_preserved_rate") is not None
        for row in group_rows
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
        for (probe_mode, intervention_variant), rows in sorted(by_mode.items()):
            rows = sorted(rows, key=lambda row: row["history_index"])
            ax.plot(
                [row["history_index"] for row in rows],
                [
                    0.0
                    if row["conditional_followup_pixel_goal_preserved_rate"] is None
                    else row["conditional_followup_pixel_goal_preserved_rate"]
                    for row in rows
                ],
                marker="o",
                label=f"{probe_mode}:{intervention_variant}",
            )
        ax.set_xlabel("History Index")
        ax.set_ylabel("Conditional Follow-up Preserve Rate")
        ax.set_title("Follow-up Replay: Preserve Pixel Goal")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
        if len(by_mode) > 1:
            ax.legend()
        followup_path = output_dir / "followup_pixel_goal_preserved_rate.png"
        fig.tight_layout()
        fig.savefig(followup_path)
        plt.close(fig)
        plot_paths.append(str(followup_path))

    return plot_paths, None


def write_markdown(path: Path, title: str, summary: dict):
    lines = [f"# {title}", ""]
    lines.append(f"- Probe jsonl: `{summary['probe_jsonl']}`")
    lines.append(f"- Inventory jsonl: `{summary['inventory_jsonl']}`")
    lines.append(f"- Total probe rows: {summary['total_probe_rows']}")
    lines.append(f"- Valid probe rows: {summary['valid_probe_rows']}")
    lines.append(f"- Error rows: {summary['error_probe_rows']}")
    lines.append(f"- Intervention variants: {summary['intervention_variants']}")
    lines.append(
        f"- Strong intervention: {summary['intervention_is_strong']}"
    )
    lines.append(
        f"- Follow-up replay enabled in probe rows: {summary['history_probe_run_followup_replay']}"
    )
    lines.append(f"- Intervention note: {summary['intervention_note']}")
    inv = summary["inventory_summary"]
    lines.append(f"- Inventory rows: {inv['inventory_rows']}")
    lines.append(f"- Unique step_ids: {inv['unique_step_ids']}")
    lines.append(f"- Primary call rows: {inv['primary_call_rows']}")
    lines.append(f"- Follow-up call rows: {inv['followup_call_rows']}")
    lines.append(f"- Gateway candidate primary rows: {inv['gateway_candidate_rows']}")
    lines.append(
        f"- Gateway rows with pixel-goal follow-up: {inv['gateway_with_pixel_goal_followup_rows']}"
    )
    lines.append(f"- Selected gateway rows: {inv['selected_gateway_rows']}")
    lines.append(
        f"- Gateway skipped reason counts: {json.dumps(inv['gateway_skipped_reason_counts'], ensure_ascii=False, sort_keys=True)}"
    )
    if summary["per_history_index"]:
        lines.append("")
        lines.append("## Per History Index")
        lines.append("")
        for row in summary["per_history_index"]:
            changed_output_counts = json.dumps(
                row.get("changed_output_counts") or {},
                ensure_ascii=False,
                sort_keys=True,
            )
            changed_followup_output_counts = json.dumps(
                row.get("changed_followup_output_counts") or {},
                ensure_ascii=False,
                sort_keys=True,
            )
            lines.append(
                f"- {row['probe_mode']} idx={row['history_index']}: "
                f"intervention={row['intervention_variant']}, "
                f"preserve={row['gateway_action_preserved_rate']}, "
                f"changed={row['gateway_action_changed_rate']}, "
                f"followup_exec={row['followup_replay_executed_rate']}, "
                f"followup_preserve={row['conditional_followup_pixel_goal_preserved_rate']}, "
                f"end_to_end={row['unconditional_end_to_end_preserved_rate']}, "
                f"followup_l2={row['mean_followup_pixel_goal_l2_shift']}, "
                f"first_token_changed={row['first_generated_token_change_rate']}, "
                f"token_delta={row['mean_generated_token_count_delta']}, "
                f"changed_outputs={changed_output_counts}, "
                f"changed_followups={changed_followup_output_counts}"
            )

    if summary["plots"]:
        lines.append("")
        lines.append("## Plots")
        lines.append("")
        for plot_path in summary["plots"]:
            lines.append(f"- `{plot_path}`")

    if summary["plot_note"]:
        lines.append("")
        lines.append(f"- Plot note: {summary['plot_note']}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    input_jsonl = Path(args.input_jsonl).resolve()
    inventory_jsonl = (
        Path(args.inventory_jsonl).resolve()
        if args.inventory_jsonl
        else input_jsonl.with_name(
            input_jsonl.name.replace("history_probe_rank", "history_probe_inventory_rank")
        )
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else input_jsonl.with_name(f"{input_jsonl.stem}_summary")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_rows = load_rows(input_jsonl)
    inventory_rows = load_rows(inventory_jsonl)
    group_rows = build_group_rows(probe_rows)
    inventory_summary = build_inventory_summary(inventory_rows)

    summary = {
        "probe_jsonl": str(input_jsonl),
        "inventory_jsonl": str(inventory_jsonl),
        "total_probe_rows": len(probe_rows),
        "valid_probe_rows": sum(1 for row in probe_rows if not row.get("probe_error")),
        "error_probe_rows": sum(1 for row in probe_rows if row.get("probe_error")),
        "probe_modes": sorted({row.get("probe_mode") for row in probe_rows if row.get("probe_mode")}),
        "history_probe_targets": sorted(
            {row.get("history_probe_target") for row in probe_rows if row.get("history_probe_target")}
        ),
        "intervention_variants": sorted(
            {
                row.get("intervention_variant") or row.get("intervention_type")
                for row in probe_rows
                if row.get("intervention_variant") or row.get("intervention_type")
            }
        ),
        "history_probe_run_followup_replay": any(
            bool(row.get("history_probe_run_followup_replay")) for row in probe_rows
        ),
        "intervention_type": (
            probe_rows[0].get("intervention_type")
            if probe_rows
            else "strong_null_history_image_replacement"
        ),
        "intervention_is_strong": any(
            bool(row.get("intervention_is_strong")) for row in probe_rows
        ) if probe_rows else True,
        "intervention_note": (
            "Per-row intervention notes may differ; inspect intervention_variant and intervention_note "
            "fields in the jsonl for exact semantics."
            if len(
                {
                    row.get("intervention_variant") or row.get("intervention_type")
                    for row in probe_rows
                }
            ) > 1
            else (
                probe_rows[0].get("intervention_note")
                if probe_rows
                else "History frames are replaced with black null images; this is a strong OOD intervention."
            )
        ),
        "inventory_summary": inventory_summary,
        "per_history_index": group_rows,
    }

    plot_paths, plot_note = maybe_plot(group_rows, output_dir)
    summary["plots"] = plot_paths
    summary["plot_note"] = plot_note

    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "per_history_index.csv"
    md_path = output_dir / "summary.md"

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(csv_path, group_rows)
    write_markdown(md_path, args.title, summary)


if __name__ == "__main__":
    main()

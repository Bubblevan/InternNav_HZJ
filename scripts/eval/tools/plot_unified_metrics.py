import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _summary_metadata(plot_generated_files=None):
    return {
        "schema_version": 1,
        "backend_filter_default": "dual_system_*",
        "placeholder_fields": [
            "cache_metrics.dit_cache.*",
            "runtime_breakdown.prefill_share_of_total",
            "runtime_breakdown.decode_share_of_total",
        ],
        "metrics_collection_mode": "hot_path_local_timing + cached_worker_stats",
        "hot_path_rpc_free": True,
        "deprecated_fields": [
            "s2_metrics.s1_invocation_rate",
            "s2_metrics.s1_invocation_count",
        ],
        "plot_generated_files": list(plot_generated_files or []),
        "notes": [
            "mm_processor_ms aliases preprocess_ms for single-vLLM System-2 requests.",
            "vision_encode_ms comes from vLLM worker encoder-forward timing and only covers multimodal encoder forward, not processor work.",
            "llm_prefill_ms/llm_decode_ms/llm_extend_ms come from worker-side per-request forward+sample timing classified by scheduler stage.",
            "generate_residual_ms is generate_ms minus vision_encode_ms; it still mixes LLM prefill, decode, and engine scheduling.",
            "s1_trigger_rate is the recommended collaboration metric.",
            "avg_s1_rollout_calls_per_trigger explains repeated local S1 replanning under one trigger.",
            "legacy rollout-call-based S1 counters are deprecated if still present.",
            "transport_metrics quantify non-vLLM-core overhead outside runtime_breakdown.total_ms.",
        ],
    }


def _expand_inputs(inputs):
    paths = []
    for item in inputs:
        expanded = sorted(glob.glob(item))
        if expanded:
            paths.extend(expanded)
        else:
            paths.append(item)
    return [Path(path) for path in paths]


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _mean_or_none(values):
    vals = [float(v) for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def _percentile_or_none(values, pct):
    vals = [float(v) for v in values if v is not None]
    return float(np.percentile(np.asarray(vals, dtype=np.float64), pct)) if vals else None


def _aggregate_episode_jsonl(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"No records found in {path}")

    backend_values = sorted({record.get("backend") for record in records if record.get("backend")})
    backend = backend_values[0] if len(backend_values) == 1 else "mixed"

    nav_metrics = {
        "sr": _mean_or_none([(record.get("nav_metrics") or {}).get("sr") for record in records]),
        "spl": _mean_or_none([(record.get("nav_metrics") or {}).get("spl") for record in records]),
        "osr": _mean_or_none([(record.get("nav_metrics") or {}).get("osr") for record in records]),
        "ne": _mean_or_none([(record.get("nav_metrics") or {}).get("ne") for record in records]),
        "ndtw": _mean_or_none([(record.get("nav_metrics") or {}).get("ndtw") for record in records]),
        "collision_count": _mean_or_none(
            [(record.get("nav_metrics") or {}).get("collision_count") for record in records]
        ),
        "psi_rate": _mean_or_none([(record.get("nav_metrics") or {}).get("psi_rate") for record in records]),
    }

    s2_latency_trace = []
    control_gap_trace = []
    runtime_samples = []
    transport_samples = []
    s1_call_metrics = []
    for record in records:
        traces = record.get("traces") or {}
        s2_latency_trace.extend(traces.get("s2_step_latency_ms") or [])
        control_gap_trace.extend(traces.get("end_to_end_control_gap_ms") or [])
        runtime_samples.extend(traces.get("s2_runtime_samples") or [])
        transport_samples.extend(traces.get("transport_metric_samples") or [])
        s1_call_metrics.extend(traces.get("s1_call_metrics") or [])

    total_s2_requests = int(sum((record.get("s2_metrics") or {}).get("s2_requests", 0) or 0 for record in records))
    total_pixel_goal_count = int(
        sum((record.get("s2_metrics") or {}).get("pixel_goal_count", 0) or 0 for record in records)
    )
    total_latent_success_count = int(
        sum((record.get("s2_metrics") or {}).get("latent_success_count", 0) or 0 for record in records)
    )
    total_s1_trigger_count = int(
        sum((record.get("s2_metrics") or {}).get("s1_trigger_count", 0) or 0 for record in records)
    )
    total_s1_rollout_call_count = int(
        sum((record.get("s2_metrics") or {}).get("s1_rollout_call_count", 0) or 0 for record in records)
    )
    total_s1_actions = int(sum((record.get("s2_metrics") or {}).get("s1_actions_total", 0) or 0 for record in records))
    total_s2_discrete_action_count = int(
        sum((record.get("s2_metrics") or {}).get("s2_discrete_action_count", 0) or 0 for record in records)
    )

    s2_metrics = {
        "s2_requests": total_s2_requests,
        "pixel_goal_yield_rate": (
            float(total_pixel_goal_count / total_s2_requests) if total_s2_requests > 0 else None
        ),
        "latent_success_rate": (
            float(total_latent_success_count / total_pixel_goal_count) if total_pixel_goal_count > 0 else None
        ),
        "s1_trigger_rate": (
            float(total_s1_trigger_count / total_s2_requests) if total_s2_requests > 0 else None
        ),
        "avg_s1_actions_per_trigger": (
            float(total_s1_actions / total_s1_trigger_count) if total_s1_trigger_count > 0 else None
        ),
        "avg_s1_rollout_calls_per_trigger": (
            float(total_s1_rollout_call_count / total_s1_trigger_count) if total_s1_trigger_count > 0 else None
        ),
        "s2_discrete_action_rate": (
            float(total_s2_discrete_action_count / total_s2_requests) if total_s2_requests > 0 else None
        ),
        "s2_step_latency_ms_mean": _mean_or_none(s2_latency_trace),
        "s2_step_latency_ms_p50": _percentile_or_none(s2_latency_trace, 50),
        "s2_step_latency_ms_p90": _percentile_or_none(s2_latency_trace, 90),
        "s2_step_latency_ms_p99": _percentile_or_none(s2_latency_trace, 99),
        "s2_step_latency_ms_max": max(s2_latency_trace) if s2_latency_trace else None,
        "end_to_end_control_gap_ms_mean": _mean_or_none(control_gap_trace),
        "end_to_end_control_gap_ms_p90": _percentile_or_none(control_gap_trace, 90),
        "end_to_end_control_gap_ms_max": max(control_gap_trace) if control_gap_trace else None,
        "episode_wall_time_s": _mean_or_none(
            [(record.get("s2_metrics") or {}).get("episode_wall_time_s") for record in records]
        ),
        "effective_low_level_hz": _mean_or_none(
            [(record.get("s2_metrics") or {}).get("effective_low_level_hz") for record in records]
        ),
        "s1_trigger_count": total_s1_trigger_count,
        "s1_rollout_call_count": total_s1_rollout_call_count,
        "s1_actions_total": total_s1_actions,
        "s1_invocation_rate": (
            float(total_s1_rollout_call_count / total_s2_requests) if total_s2_requests > 0 else None
        ),
        "s1_invocation_count": total_s1_rollout_call_count,
    }

    runtime_breakdown = {}
    for field in (
        "preprocess_ms",
        "mm_processor_ms",
        "generate_ms",
        "vision_encode_ms",
        "vision_encoder_calls",
        "llm_prefill_ms",
        "llm_prefill_forward_ms",
        "llm_prefill_sample_ms",
        "llm_prefill_forward_calls",
        "llm_prefill_sample_calls",
        "llm_decode_ms",
        "llm_decode_forward_ms",
        "llm_decode_sample_ms",
        "llm_decode_forward_calls",
        "llm_decode_sample_calls",
        "llm_extend_ms",
        "llm_extend_forward_ms",
        "llm_extend_sample_ms",
        "llm_extend_forward_calls",
        "llm_extend_sample_calls",
        "generate_residual_ms",
        "bundle_build_ms",
        "mm_attach_ms",
        "latent_prefill_ms",
        "total_ms",
        "prefill_share_of_total",
        "latent_prefill_share_of_total",
        "decode_share_of_total",
        "prompt_token_count",
        "generated_token_count",
        "prefill_token_count",
        "n_query",
        "num_images",
        "mm_feature_count",
    ):
        runtime_breakdown[field] = _mean_or_none([sample.get(field) for sample in runtime_samples])

    transport_metrics = {
        "client_encode_messages_ms": _mean_or_none([sample.get("client_encode_messages_ms") for sample in transport_samples]),
        "client_http_post_ms": _mean_or_none([sample.get("client_http_post_ms") for sample in transport_samples]),
        "client_response_json_ms": _mean_or_none([sample.get("client_response_json_ms") for sample in transport_samples]),
        "client_decode_latents_ms": _mean_or_none([sample.get("client_decode_latents_ms") for sample in transport_samples]),
        "client_total_ms": _mean_or_none([sample.get("client_total_ms") for sample in transport_samples]),
        "server_request_parse_ms": _mean_or_none([sample.get("server_request_parse_ms") for sample in transport_samples]),
        "server_decode_messages_ms": _mean_or_none([sample.get("server_decode_messages_ms") for sample in transport_samples]),
        "server_runner_step_s2_ms": _mean_or_none([sample.get("server_runner_step_s2_ms") for sample in transport_samples]),
        "server_encode_response_ms": _mean_or_none([sample.get("server_encode_response_ms") for sample in transport_samples]),
        "server_total_ms": _mean_or_none([sample.get("server_total_ms") for sample in transport_samples]),
        "server_outer_overhead_ms": _mean_or_none([sample.get("server_outer_overhead_ms") for sample in transport_samples]),
        "client_side_overhead_ms": _mean_or_none([sample.get("client_side_overhead_ms") for sample in transport_samples]),
        "end_to_end_transport_overhead_ms": _mean_or_none(
            [sample.get("end_to_end_transport_overhead_ms") for sample in transport_samples]
        ),
    }

    s1_metrics = {
        "s1_generate_traj_ms_total": _mean_or_none([metric.get("s1_generate_traj_ms_total") for metric in s1_call_metrics]),
        "s1_memory_encode_ms": _mean_or_none([metric.get("s1_memory_encode_ms") for metric in s1_call_metrics]),
        "s1_rgb_encode_ms": _mean_or_none([metric.get("s1_rgb_encode_ms") for metric in s1_call_metrics]),
        "s1_cond_project_ms": _mean_or_none([metric.get("s1_cond_project_ms") for metric in s1_call_metrics]),
        "s1_dit_loop_ms": _mean_or_none([metric.get("s1_dit_loop_ms") for metric in s1_call_metrics]),
        "s1_action_decode_ms": _mean_or_none([metric.get("s1_action_decode_ms") for metric in s1_call_metrics]),
        "diffusion_steps_total": int(sum(metric.get("diffusion_steps_total", 0) or 0 for metric in s1_call_metrics))
        if s1_call_metrics
        else None,
        "diffusion_steps_reused": int(sum(metric.get("diffusion_steps_reused", 0) or 0 for metric in s1_call_metrics))
        if s1_call_metrics
        else None,
        "diffusion_steps_executed": int(
            sum(metric.get("diffusion_steps_executed", 0) or 0 for metric in s1_call_metrics)
        )
        if s1_call_metrics
        else None,
        "s1_generator_seed": None,
        "s1_deterministic_mode": any(bool(metric.get("s1_deterministic_mode")) for metric in s1_call_metrics)
        if s1_call_metrics
        else None,
    }

    vllm_kv_cache = {}
    for record in reversed(records):
        cache_block = ((record.get("cache_metrics") or {}).get("vllm_kv_cache") or {})
        if cache_block:
            vllm_kv_cache = cache_block
            break

    return {
        "backend": backend,
        "episode_count": len(records),
        "metadata": records[0].get("metadata") or _summary_metadata(),
        "nav_metrics": nav_metrics,
        "s2_metrics": s2_metrics,
        "s1_metrics": s1_metrics,
        "cache_metrics": {
            "dit_cache": ((records[0].get("cache_metrics") or {}).get("dit_cache") or {}),
            "vllm_kv_cache": vllm_kv_cache,
        },
        "runtime_breakdown": runtime_breakdown,
        "transport_metrics": transport_metrics,
        "source_path": str(path),
    }


def _load_summary(path: Path):
    if path.suffix == ".jsonl":
        return _aggregate_episode_jsonl(path)
    payload = _load_json(path)
    payload["source_path"] = str(path)
    return payload


def _normalize_backend_label(summary, seen):
    backend = summary.get("backend") or "unknown"
    parent = Path(summary.get("source_path", "")).resolve().parent.name
    label = backend
    if label in seen:
        label = f"{backend}@{parent}"
    seen.add(label)
    return label


def _filter_summaries(summaries, include_system2):
    if include_system2:
        return summaries
    filtered = [summary for summary in summaries if str(summary.get("backend", "")).startswith("dual_system_")]
    return filtered


def _plot_metric_panels(summaries, metric_specs, title, output_stem, output_dir):
    labels = [summary["_plot_label"] for summary in summaries]
    fig, axes = plt.subplots(2, int(np.ceil(len(metric_specs) / 2)), figsize=(4 * len(metric_specs), 8))
    axes = np.asarray(axes).reshape(-1)
    for ax, (section, key, title_text) in zip(axes, metric_specs):
        values = [(summary.get(section) or {}).get(key) for summary in summaries]
        xs = np.arange(len(labels))
        plot_values = [np.nan if value is None else float(value) for value in values]
        ax.bar(xs, plot_values, color="#4E79A7")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title_text)
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[len(metric_specs):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [str(png_path), str(pdf_path)]


def _plot_stacked_runtime(summaries, title, output_stem, output_dir, fields, *, section):
    labels = [summary["_plot_label"] for summary in summaries]
    fig, ax = plt.subplots(figsize=(10, 6))
    bottoms = np.zeros(len(labels), dtype=np.float64)
    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]
    for color, field in zip(colors, fields):
        values = []
        for summary in summaries:
            value = (summary.get(section) or {}).get(field)
            values.append(0.0 if value is None else float(value))
        ax.bar(labels, values, bottom=bottoms, label=field, color=color)
        bottoms += np.asarray(values, dtype=np.float64)
    ax.set_title(title)
    ax.set_ylabel("ms")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [str(png_path), str(pdf_path)]


def _plot_kv_table(summaries, output_dir):
    fields = [
        "available_kv_cache_memory_bytes",
        "requested_memory_bytes",
        "peak_activation_memory_bytes",
        "cudagraph_memory_estimate_bytes",
        "effective_kv_budget_bytes",
        "gpu_memory_utilization",
        "max_model_len",
        "num_gpu_blocks",
    ]
    labels = [summary["_plot_label"] for summary in summaries]
    rows = []
    for field in fields:
        row = [field]
        for summary in summaries:
            value = ((summary.get("cache_metrics") or {}).get("vllm_kv_cache") or {}).get(field)
            row.append("null" if value is None else str(value))
        rows.append(row)

    fig, ax = plt.subplots(figsize=(2.5 + 2 * len(labels), 0.5 * len(rows) + 1.5))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["field", *labels],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    ax.set_title("vLLM KV / Memory Stats")
    fig.tight_layout()
    png_path = output_dir / "vllm_kv_memory.png"
    pdf_path = output_dir / "vllm_kv_memory.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [str(png_path), str(pdf_path)]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot DualVLN unified metrics summaries.")
    parser.add_argument("inputs", nargs="+", help="unified_summary.json or episode_metrics_rank*.jsonl")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--include-system2", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    input_paths = _expand_inputs(args.inputs)
    summaries = [_load_summary(path) for path in input_paths]
    summaries = _filter_summaries(summaries, include_system2=args.include_system2)
    if not summaries:
        raise RuntimeError("No summaries left after backend filtering.")

    seen_labels = set()
    for summary in summaries:
        summary["_plot_label"] = _normalize_backend_label(summary, seen_labels)

    output_dir = Path(args.output_dir) if args.output_dir else input_paths[0].resolve().parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_files = []
    plot_files.extend(
        _plot_metric_panels(
            summaries,
            [
                ("nav_metrics", "sr", "SR"),
                ("nav_metrics", "spl", "SPL"),
                ("nav_metrics", "osr", "OSR"),
                ("nav_metrics", "ne", "NE"),
                ("nav_metrics", "ndtw", "NDTW"),
                ("nav_metrics", "collision_count", "Collision Count"),
                ("nav_metrics", "psi_rate", "PSI Rate"),
            ],
            title="Navigation Quality",
            output_stem="nav_quality",
            output_dir=output_dir,
        )
    )
    plot_files.extend(
        _plot_metric_panels(
            summaries,
            [
                ("s2_metrics", "s2_step_latency_ms_mean", "S2 Latency Mean"),
                ("s2_metrics", "s2_step_latency_ms_p90", "S2 Latency P90"),
                ("s2_metrics", "s2_step_latency_ms_p99", "S2 Latency P99"),
                ("s2_metrics", "end_to_end_control_gap_ms_mean", "Control Gap Mean"),
                ("s2_metrics", "end_to_end_control_gap_ms_p90", "Control Gap P90"),
                ("s2_metrics", "effective_low_level_hz", "Low-level Hz"),
            ],
            title="S2 Latency And Control",
            output_stem="s2_latency",
            output_dir=output_dir,
        )
    )
    plot_files.extend(
        _plot_metric_panels(
            summaries,
            [
                ("s2_metrics", "pixel_goal_yield_rate", "Pixel Goal Yield"),
                ("s2_metrics", "latent_success_rate", "Latent Success"),
                ("s2_metrics", "s1_trigger_rate", "S1 Trigger"),
                ("s2_metrics", "avg_s1_actions_per_trigger", "Avg S1 Actions"),
                ("s2_metrics", "avg_s1_rollout_calls_per_trigger", "Avg S1 Rollout Calls"),
                ("s2_metrics", "s2_discrete_action_rate", "S2 Discrete Action"),
            ],
            title="Dual-System Collaboration",
            output_stem="collaboration",
            output_dir=output_dir,
        )
    )
    plot_files.extend(
        _plot_stacked_runtime(
            summaries,
            title="S2 Runtime Breakdown (Core Service Time)",
            output_stem="runtime_breakdown",
            output_dir=output_dir,
            section="runtime_breakdown",
            fields=[
                "mm_processor_ms",
                "vision_encode_ms",
                "llm_prefill_ms",
                "llm_decode_ms",
                "llm_extend_ms",
                "generate_residual_ms",
                "bundle_build_ms",
                "mm_attach_ms",
                "latent_prefill_ms",
            ],
        )
    )
    plot_files.extend(
        _plot_metric_panels(
            summaries,
            [
                ("runtime_breakdown", "total_ms", "Core Runtime Total"),
                ("transport_metrics", "server_total_ms", "Server Total"),
                ("transport_metrics", "client_total_ms", "Client Total"),
                ("transport_metrics", "end_to_end_transport_overhead_ms", "End-to-End Overhead"),
            ],
            title="Transport And Wrapper Overhead",
            output_stem="transport_overhead",
            output_dir=output_dir,
        )
    )
    plot_files.extend(
        _plot_stacked_runtime(
            summaries,
            title="S1 Breakdown",
            output_stem="s1_breakdown",
            output_dir=output_dir,
            section="s1_metrics",
            fields=[
                "s1_memory_encode_ms",
                "s1_rgb_encode_ms",
                "s1_cond_project_ms",
                "s1_dit_loop_ms",
                "s1_action_decode_ms",
            ],
        )
    )
    plot_files.extend(_plot_kv_table(summaries, output_dir))

    meta_payload = _summary_metadata(plot_generated_files=plot_files)
    meta_payload["source_inputs"] = [str(path) for path in input_paths]
    meta_payload["include_system2"] = bool(args.include_system2)
    meta_payload["skipped_placeholder_plots"] = ["dit_cache_benefit"]
    meta_path = output_dir / "unified_summary_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2, ensure_ascii=False)

    print(json.dumps({"output_dir": str(output_dir), "plot_generated_files": plot_files}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

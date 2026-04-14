#!/usr/bin/env python3
"""Compare profiling results across multiple VLM generation experiments.

Usage:
    uv run python examples/train/visgym_relaxed/compare_experiments.py \
        inference-exps/baseline inference-exps/semaphore-16 \
        --output inference-exps/comparison/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_experiment(exp_dir: Path) -> dict[str, Any]:
    summary_path = exp_dir / "summary.json"
    raw_path = exp_dir / "raw_profiles.json"
    if not summary_path.exists():
        sys.exit(f"ERROR: {summary_path} not found")
    data: dict[str, Any] = {"name": exp_dir.name, "dir": exp_dir}
    data["summary"] = json.loads(summary_path.read_text())
    if raw_path.exists():
        data["raw"] = json.loads(raw_path.read_text())
    else:
        data["raw"] = None
    return data


def per_turn_means(data: dict[str, Any], field: str) -> tuple[list[int], list[float]]:
    """Extract per-turn mean values from summary or raw profiles."""
    per_turn = data["summary"].get("per_turn", {})
    if per_turn:
        turns = sorted(int(t) for t in per_turn)
        means = [per_turn[str(t)][field]["mean"] for t in turns if field in per_turn[str(t)]]
        return turns[: len(means)], means

    if data["raw"] is None:
        return [], []

    from collections import defaultdict
    buckets: dict[int, list[float]] = defaultdict(list)
    for traj in data["raw"]:
        for tp in traj:
            buckets[tp["turn"]].append(tp[field])
    turns = sorted(buckets)
    means = [float(np.mean(buckets[t])) for t in turns]
    return turns, means


def per_turn_derived_means(
    data: dict[str, Any], num_field: str, denom_field: str
) -> tuple[list[int], list[float]]:
    """Compute per-turn mean of (num_field / denom_field) from raw profiles."""
    per_turn = data["summary"].get("per_turn", {})
    derived_key = f"{num_field.replace('_s', '')}_per_{denom_field.replace('num_', '')}_s"
    if per_turn:
        turns = sorted(int(t) for t in per_turn)
        means = []
        for t in turns:
            entry = per_turn[str(t)]
            if derived_key in entry:
                means.append(entry[derived_key]["mean"])
            elif num_field in entry and denom_field in entry:
                n = entry[num_field]["mean"]
                d = entry[denom_field]["mean"]
                means.append(n / d if d > 0 else 0.0)
            else:
                break
        return turns[: len(means)], means

    if data["raw"] is None:
        return [], []

    from collections import defaultdict
    buckets: dict[int, list[float]] = defaultdict(list)
    for traj in data["raw"]:
        for tp in traj:
            d = tp.get(denom_field, 0)
            if d > 0:
                buckets[tp["turn"]].append(tp[num_field] / d)
    turns = sorted(buckets)
    means = [float(np.mean(buckets[t])) for t in turns]
    return turns, means


def write_comparison_summary(experiments: list[dict], output_dir: Path) -> None:
    rollout_metrics = [
        ("Wall-clock (s)", "wall_clock_s"),
        ("Output tok/s", "output_tokens_per_second"),
        ("Input tok/s", "input_tokens_per_second"),
        ("Trajectories/s", "trajectories_per_second"),
        ("s / output tok", "seconds_per_output_token"),
        ("Total output tokens", "total_output_tokens"),
        ("Total input tokens", "total_input_tokens"),
        ("Num trajectories", "num_trajectories"),
    ]

    names = [e["name"] for e in experiments]
    header = "| Metric | " + " | ".join(names) + " |"
    sep = "|" + "---|" * (len(names) + 1)

    lines = ["# Experiment Comparison", "", header, sep]
    for label, key in rollout_metrics:
        row = f"| {label} "
        for e in experiments:
            rollout = e["summary"].get("rollout", {})
            val = rollout.get(key, "N/A")
            if isinstance(val, float):
                row += f"| {val:.4f} "
            else:
                row += f"| {val} "
        row += "|"
        lines.append(row)

    totals_fields = [
        ("Render (CPU-s)", "render_time_s"),
        ("Generate (CPU-s)", "generate_time_s"),
        ("Env step (CPU-s)", "env_step_time_s"),
        ("Deserialize (CPU-s)", "deserialize_time_s"),
        ("Total (CPU-s)", "total_time_s"),
    ]
    lines.append("")
    lines.append("### Component CPU-seconds (summed across all trajectories)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for label, key in totals_fields:
        row = f"| {label} "
        for e in experiments:
            totals = e["summary"].get("totals", {})
            val = totals.get(key, "N/A")
            if isinstance(val, float):
                row += f"| {val:.2f} "
            else:
                row += f"| {val} "
        row += "|"
        lines.append(row)

    (output_dir / "comparison_summary.md").write_text("\n".join(lines) + "\n")


def plot_headline_bars(experiments: list[dict], output_dir: Path) -> None:
    metrics = [
        ("Wall-clock (s)", "wall_clock_s"),
        ("Output tok/s", "output_tokens_per_second"),
        ("Trajectories/s", "trajectories_per_second"),
    ]
    names = [e["name"] for e in experiments]
    x = np.arange(len(metrics))
    width = 0.8 / max(len(experiments), 1)

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (label, key) in zip(axes, metrics):
        vals = []
        for e in experiments:
            rollout = e["summary"].get("rollout", {})
            vals.append(rollout.get(key, 0))
        bars = ax.bar(names, vals, color=plt.cm.Set2(np.linspace(0, 1, len(names))))
        ax.set_title(label)
        ax.set_ylabel(label)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Headline Metrics", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "headline_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_turn_generate_time(experiments: list[dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for e in experiments:
        turns, means = per_turn_means(e, "generate_time_s")
        if turns:
            ax.plot(turns, means, marker="o", label=e["name"])
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean generate time (s)")
    ax.set_title("Per-turn Generate Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "per_turn_generate_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_turn_time_per_output_token(experiments: list[dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for e in experiments:
        turns, means = per_turn_derived_means(e, "generate_time_s", "num_output_tokens")
        if turns:
            ax.plot(turns, means, marker="o", label=e["name"])
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean generate time / output token (s)")
    ax.set_title("Per-turn Time per Output Token")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        output_dir / "per_turn_time_per_output_token.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def plot_component_breakdown(experiments: list[dict], output_dir: Path) -> None:
    components = [
        ("Render", "render_time_s"),
        ("Generate", "generate_time_s"),
        ("Env step", "env_step_time_s"),
        ("Deserialize", "deserialize_time_s"),
    ]
    names = [e["name"] for e in experiments]
    x = np.arange(len(names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(8, 2 * len(names)), 5))

    for i, (label, key) in enumerate(components):
        vals = [e["summary"].get("totals", {}).get(key, 0) for e in experiments]
        ax.bar(x + i * width, vals, width, label=label)

    ax.set_xlabel("Experiment")
    ax.set_ylabel("Total CPU-seconds")
    ax.set_title("Component Time Breakdown")
    ax.set_xticks(x + width * (len(components) - 1) / 2)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "component_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare profiling results across VLM generation experiments"
    )
    parser.add_argument(
        "experiments",
        nargs="+",
        type=Path,
        help="Paths to experiment directories (each must contain summary.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("inference-exps/comparison"),
        help="Output directory for comparison artifacts",
    )
    args = parser.parse_args()

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = [load_experiment(p) for p in args.experiments]
    print(f"Comparing {len(experiments)} experiments: {[e['name'] for e in experiments]}")

    write_comparison_summary(experiments, output_dir)
    print(f"  -> {output_dir / 'comparison_summary.md'}")

    plot_headline_bars(experiments, output_dir)
    print(f"  -> {output_dir / 'headline_bars.png'}")

    plot_per_turn_generate_time(experiments, output_dir)
    print(f"  -> {output_dir / 'per_turn_generate_time.png'}")

    plot_per_turn_time_per_output_token(experiments, output_dir)
    print(f"  -> {output_dir / 'per_turn_time_per_output_token.png'}")

    plot_component_breakdown(experiments, output_dir)
    print(f"  -> {output_dir / 'component_breakdown.png'}")

    print("Done.")


if __name__ == "__main__":
    main()

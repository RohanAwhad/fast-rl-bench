"""Generate the report's plots + tables from collected_metrics.json /
reward_curves.csv (see collect_metrics.py) plus standalone eval JSON results
(eval_reverse_text.sh / eval_sciknoweval.sh outputs).

Usage:
    python3 make_plots.py --results-dir results --out-dir ../report/figures
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONDITION_LABELS = {
    "baseline": "Baseline (GRPO)",
    "duet": "DUET",
    "greso": "GRESO",
    "difficulty_targeted": "Difficulty-Targeted +Replay",
    "experience_replay": "Experience Replay",
    "mu_grpo": "µ-GRPO",
}
CONDITION_ORDER = ["baseline", "duet", "greso", "difficulty_targeted", "experience_replay", "mu_grpo"]
COLORS = {
    "baseline": "#444444",
    "duet": "#1f77b4",
    "greso": "#ff7f0e",
    "difficulty_targeted": "#2ca02c",
    "experience_replay": "#d62728",
    "mu_grpo": "#9467bd",
}


def load_reward_curves(path: Path) -> dict[tuple[str, str], list[tuple[int, float]]]:
    curves: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["task"], row["condition"])
            curves[key].append((int(row["step"]), float(row["reward"])))
    for key in curves:
        curves[key].sort()
    return curves


def plot_reward_curves(curves: dict[tuple[str, str], list[tuple[int, float]]], task: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for condition in CONDITION_ORDER:
        key = (task, condition)
        if key not in curves or not curves[key]:
            continue
        steps, rewards = zip(*curves[key])
        ax.plot(steps, rewards, label=CONDITION_LABELS[condition], color=COLORS[condition], linewidth=1.8)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Train reward (mean, per-step)")
    ax.set_title(f"{task.replace('_', ' ')}: reward vs. step (5-min training budget)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reward_vs_time(
    curves: dict[tuple[str, str], list[tuple[int, float]]],
    summaries: dict,
    task: str,
    out_path: Path,
) -> None:
    """Reward vs. wall-clock (approximated as cumulative time/step), which is
    the metric all 5 papers actually optimize for."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for condition in CONDITION_ORDER:
        run_name = f"{task}-{condition}"
        s = summaries.get(run_name)
        key = (task, condition)
        if s is None or key not in curves or not curves[key]:
            continue
        step_to_reward = dict(curves[key])
        cum_times = s.get("cumulative_step_times", {})
        if not cum_times:
            continue
        xs, ys = [], []
        for step in sorted(step_to_reward):
            if str(step) in cum_times:
                xs.append(cum_times[str(step)])
                ys.append(step_to_reward[step])
        if xs:
            ax.plot(xs, ys, label=CONDITION_LABELS[condition], color=COLORS[condition], linewidth=1.8)
    ax.axvline(300, color="red", linestyle="--", alpha=0.6, label="5 min cutoff")
    ax.set_xlabel("Training wall-clock (s, cumulative time/step)")
    ax.set_ylabel("Train reward (mean, per-step)")
    ax.set_title(f"{task.replace('_', ' ')}: reward vs. wall-clock time")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_summary_table(summaries: dict, task: str) -> str:
    lines = [
        "| Condition | Max step | Final train reward | Total training time (s) | Eval accuracy/reward |",
        "|---|---|---|---|---|",
    ]
    for condition in CONDITION_ORDER:
        run_name = f"{task}-{condition}"
        s = summaries.get(run_name)
        if s is None:
            lines.append(f"| {CONDITION_LABELS[condition]} | — | — | — | — |")
            continue
        eval_metric = s.get("eval_metric", "—")
        lines.append(
            f"| {CONDITION_LABELS[condition]} | {s.get('max_step', '—')} | "
            f"{s.get('final_reward', 0.0):.4f} | {s.get('total_step_time_s', 0.0):.1f} | {eval_metric} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "collected_metrics.json") as f:
        summaries = json.load(f)
    curves = load_reward_curves(results_dir / "reward_curves.csv")

    for task in ("reverse_text", "sciknoweval"):
        plot_reward_curves(curves, task, out_dir / f"{task}_reward_vs_step.png")
        plot_reward_vs_time(curves, summaries, task, out_dir / f"{task}_reward_vs_time.png")
        table = make_summary_table(summaries, task)
        (out_dir / f"{task}_summary_table.md").write_text(table + "\n")
        print(f"\n=== {task} ===")
        print(table)

    print(f"\nFigures + tables written to {out_dir}")


if __name__ == "__main__":
    main()

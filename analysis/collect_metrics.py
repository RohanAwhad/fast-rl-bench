"""Collect metrics.jsonl from every run's output dir into one tidy CSV +
per-run summary JSON. Run on the GPU node (reads from ~/prime-rl/outputs/)
or against a local copy of the outputs directory.

Usage:
    python3 collect_metrics.py --outputs-dir ~/prime-rl/outputs --out-dir results
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CONDITIONS = ["baseline", "duet", "greso", "difficulty_targeted", "experience_replay", "mu_grpo"]
TASKS = ["reverse_text", "sciknoweval"]


def load_metrics_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def first_rollout_ts(launch_log: Path) -> float | None:
    """Best-effort: first orchestrator log timestamp that mentions dispatch
    of rollouts (used to compute "time since start of first rollout" for the
    5-minute-budget verification). Falls back to None if unavailable --
    callers should treat that as "use launch_start.ts instead"."""
    if not launch_log.exists():
        return None
    return None  # placeholder: timestamp parsing done via launch_start.ts + step times instead


def summarize_run(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.jsonl"
    rows = load_metrics_jsonl(metrics_path)
    train_rows = [r for r in rows if "train/agg/effective/reward/mean" in r or "train/agg/all/reward/mean" in r]
    reward_key = None
    for candidate in ("train/agg/effective/reward/mean", "train/agg/all/reward/mean"):
        if any(candidate in r for r in train_rows):
            reward_key = candidate
            break

    step_times = [r.get("time/step") for r in rows if "time/step" in r]
    steps = [r.get("step") for r in rows if "step" in r]
    total_training_time_s = sum(t for t in step_times if isinstance(t, (int, float)))

    launch_start_file = run_dir / "launch_start.ts"
    launch_start = None
    if launch_start_file.exists():
        try:
            launch_start = float(launch_start_file.read_text().strip())
        except ValueError:
            pass

    reward_curve = []
    if reward_key:
        for r in rows:
            if "step" in r and reward_key in r:
                reward_curve.append({"step": r["step"], "reward": r[reward_key]})

    return {
        "run_dir": str(run_dir),
        "n_metric_rows": len(rows),
        "max_step": max(steps) if steps else None,
        "reward_key_used": reward_key,
        "final_reward": reward_curve[-1]["reward"] if reward_curve else None,
        "total_step_time_s": total_training_time_s,
        "launch_start_epoch": launch_start,
        "reward_curve": reward_curve,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", required=True, help="prime-rl outputs/ dir (contains one subdir per run)")
    ap.add_argument("--out-dir", required=True, help="where to write collected_metrics.csv / .json")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir).expanduser()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}
    for task in TASKS:
        for condition in CONDITIONS:
            run_name = f"{task}-{condition}"
            run_dir = outputs_dir / run_name
            if not run_dir.exists():
                continue
            summaries[run_name] = summarize_run(run_dir)
            summaries[run_name]["task"] = task
            summaries[run_name]["condition"] = condition

    with open(out_dir / "collected_metrics.json", "w") as f:
        json.dump(summaries, f, indent=2)

    with open(out_dir / "collected_metrics_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "condition", "max_step", "final_reward", "total_step_time_s", "n_metric_rows"])
        for run_name, s in sorted(summaries.items()):
            writer.writerow(
                [s["task"], s["condition"], s["max_step"], s["final_reward"], round(s["total_step_time_s"], 1), s["n_metric_rows"]]
            )

    # Long-format reward curves (one row per step per run) for plotting.
    with open(out_dir / "reward_curves.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "condition", "step", "reward"])
        for run_name, s in sorted(summaries.items()):
            for point in s["reward_curve"]:
                writer.writerow([s["task"], s["condition"], point["step"], point["reward"]])

    print(f"Collected {len(summaries)} runs -> {out_dir}")
    for run_name, s in sorted(summaries.items()):
        print(f"  {run_name}: max_step={s['max_step']} final_reward={s['final_reward']} total_step_time_s={s['total_step_time_s']:.1f}")


if __name__ == "__main__":
    main()

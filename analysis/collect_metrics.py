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


def summarize_run(run_dir: Path) -> dict:
    """NOTE on metrics.jsonl shape (verified against a real run): the orchestrator
    AND the trainer both append rows to the same metrics.jsonl, and both use a
    "step" field with overlapping/restarting numbering, so naively summing every
    row's "time/step" double-counts (trainer emits several sub-rows per orchestrator
    step with a *different* meaning of time/step - its own microbatch/optim timing).
    The orchestrator's own per-step row is uniquely identifiable as the row that
    carries the aggregate reward key (train/agg/{effective,all}/agent/reward/mean) --
    only those rows' "time/step" match the "Step N | Xs |" line in orchestrator.log
    (this is what calibrate.sh's log-parsing measures, and it matches these values
    exactly), so we filter to those rows before summing/reading step numbers."""
    metrics_path = run_dir / "metrics.jsonl"
    rows = load_metrics_jsonl(metrics_path)

    reward_key = None
    for candidate in ("train/agg/effective/agent/reward/mean", "train/agg/all/agent/reward/mean"):
        if any(candidate in r for r in rows):
            reward_key = candidate
            break

    orch_rows = [r for r in rows if reward_key and reward_key in r]
    step_times = [r["time/step"] for r in orch_rows if isinstance(r.get("time/step"), (int, float))]
    steps = [r["step"] for r in orch_rows if "step" in r]
    total_training_time_s = sum(step_times)

    launch_start_file = run_dir / "launch_start.ts"
    launch_start = None
    if launch_start_file.exists():
        try:
            launch_start = float(launch_start_file.read_text().strip())
        except ValueError:
            pass

    # dedupe by step (keep the last row seen for a given step, in case of restarts/reruns)
    reward_by_step: dict[int, float] = {}
    if reward_key:
        for r in orch_rows:
            if "step" in r:
                reward_by_step[r["step"]] = r[reward_key]
    reward_curve = [{"step": s, "reward": reward_by_step[s]} for s in sorted(reward_by_step)]

    return {
        "run_dir": str(run_dir),
        "n_metric_rows": len(rows),
        "n_orchestrator_rows": len(orch_rows),
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

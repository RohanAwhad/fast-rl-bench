"""Summarize a v1 `eval` CLI run (`-o <dir>`) into the same JSON summary shape
eval_sciknoweval.py produces, so collect_metrics.py can treat both eval paths
uniformly.

The v1 `eval` CLI (NOT the legacy `vf-eval` -- see eval_reverse_text.sh for
why) writes into a subdirectory it names itself under -o/--output-dir
(`<env>--<model>--<harness>--<short-id>`) containing `traces.jsonl` (one row
per episode: `{"id", "env", "ok", "errors", "traces": [<per-agent trace>]}`,
each trace's `"rewards"` a dict of `{name: {"score": float, "weight":
float}}` -- see verifiers/v1/session.py) and `configs/eval.json` (resolved
run config, not a summary). Since we always point -o at a fresh directory
per eval invocation, there is exactly one subdirectory to find.

Usage:
    python3 summarize_vfeval_results.py <eval_output_dir> --model <name> --out <summary.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_traces_jsonl(output_dir: Path) -> Path:
    matches = sorted(output_dir.glob("**/traces.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No traces.jsonl found under {output_dir}")
    if len(matches) > 1:
        # Shouldn't happen if output_dir is used fresh per eval, but guard anyway.
        matches = sorted(matches, key=lambda p: p.stat().st_mtime)
    return matches[-1]


def episode_reward(episode: dict) -> float | None:
    """Weighted sum of an episode's reward components, summed across its
    traces (one trace per agent seat; single-agent envs have exactly one).
    None if the episode has no reward components at all (e.g. it errored
    before scoring)."""
    total = 0.0
    seen = False
    for trace in episode.get("traces", []):
        for component in trace.get("rewards", {}).values():
            total += component["score"] * component["weight"]
            seen = True
    return total if seen else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", help="the -o/--output-dir passed to `eval`")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    traces_path = find_traces_jsonl(Path(args.output_dir))
    rewards: list[float] = []
    n_not_ok = 0
    with open(traces_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episode = json.loads(line)
            if not episode.get("ok", False):
                n_not_ok += 1
            reward = episode_reward(episode)
            if reward is not None:
                rewards.append(reward)

    summary = {
        "model": args.model,
        "n_rows": len(rewards),
        "n_not_ok": n_not_ok,
        "overall_reward": sum(rewards) / len(rewards) if rewards else None,
    }
    print(json.dumps(summary, indent=2))
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "traces_jsonl": str(traces_path)}, f, indent=2)
    print(f"summarize_vfeval_results: wrote {args.out}")


if __name__ == "__main__":
    main()

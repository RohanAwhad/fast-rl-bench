"""Summarize a vf-eval `--save-results --output-dir <dir>` run into the same
JSON summary shape eval_sciknoweval.py produces, so collect_metrics.py can
treat both eval paths uniformly.

vf-eval writes into a subdirectory it names itself under --output-dir (env_id
+ model + config hash) containing `results.jsonl` (one row per rollout, each
row a `RolloutOutput` dict with a top-level "reward": float field -- see
verifiers/legacy/types.py) and `metadata.json` (run config, not a summary).
Since we always point --output-dir at a fresh directory per eval invocation,
there is exactly one subdirectory to find.

Usage:
    python3 summarize_vfeval_results.py <vfeval_output_dir> --model <name> --out <summary.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_results_jsonl(output_dir: Path) -> Path:
    matches = sorted(output_dir.glob("**/results.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No results.jsonl found under {output_dir}")
    if len(matches) > 1:
        # Shouldn't happen if output_dir is used fresh per eval, but guard anyway.
        matches = sorted(matches, key=lambda p: p.stat().st_mtime)
    return matches[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", help="the --output-dir passed to vf-eval")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results_path = find_results_jsonl(Path(args.output_dir))
    rewards: list[float] = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "reward" in row and isinstance(row["reward"], (int, float)):
                rewards.append(row["reward"])

    summary = {
        "model": args.model,
        "n_rows": len(rewards),
        "overall_reward": sum(rewards) / len(rewards) if rewards else None,
    }
    print(json.dumps(summary, indent=2))
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "results_jsonl": str(results_path)}, f, indent=2)
    print(f"summarize_vfeval_results: wrote {args.out}")


if __name__ == "__main__":
    main()

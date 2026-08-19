"""Verify the 5-minute hard training-time cutoff empirically for a run.

Uses the exact same methodology as scripts/calibrate.sh: sum the orchestrator's
own "Step N | Xs | Reward ..." per-step durations from orchestrator.log. This
is the true end-to-end (rollout dispatch -> trained + policy published)
sink-to-sink cycle time for each step, and it already EXCLUDES dispatcher
backpressure/pause time (verified against real logs: e.g. a ~14s wall-clock
gap between two consecutive "Step N" lines, most of it spent in "Pausing
dispatcher.../Holding batch..." pipeline-depth throttling, reported as only
~4.4s of actual step time) -- so it is not the same as (last log timestamp -
first log timestamp), which would inflate the number with pipeline-tuning
idle time that isn't "training time" in any meaningful sense.

Deliberately does NOT read trainer.log: the trainer emits its own "Step N"
lines with different numbering/meaning (GPU-compute-only, no Reward field) --
mixing the two logs was a real bug caught during calibration (see devlogs.md).

Usage:
    python3 verify_time_budget.py <run_dir>   # e.g. outputs/reverse_text-baseline
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"Step (\d+) \|\s*([\d.]+)s\s*\|")


def find_log(run_dir: Path, name: str) -> Path | None:
    matches = sorted(run_dir.glob(f"**/{name}"))
    return matches[-1] if matches else None


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    run_dir = Path(sys.argv[1])

    orch_log = find_log(run_dir, "orchestrator.log")
    if orch_log is None:
        print(f"Could not find orchestrator.log under {run_dir}")
        sys.exit(1)

    times: list[tuple[int, float]] = []
    with open(orch_log, errors="ignore") as f:
        for line in f:
            clean = ANSI_RE.sub("", line)
            m = STEP_RE.search(clean)
            if m:
                times.append((int(m.group(1)), float(m.group(2))))

    if not times:
        print(f"No 'Step N | Xs |' lines found in {orch_log} -- inspect manually.")
        sys.exit(1)

    times.sort()
    elapsed = sum(t for _, t in times)
    print(f"run_dir: {run_dir}")
    print(f"orchestrator.log: {orch_log}")
    print(f"steps observed: {len(times)} (last step: {times[-1][0]})")
    print(f"TRAINING_TIME_SECONDS={elapsed:.1f}")
    print(f"UNDER_5MIN={'yes' if elapsed <= 300 else 'NO -- exceeds budget'}")


if __name__ == "__main__":
    main()

"""Verify the 5-minute hard training-time cutoff empirically for a run:
"time since the start of first rollout to end of training."

Operationalized as: first orchestrator log timestamp (pipeline startup
complete, dispatcher begins scheduling) -> timestamp of the last "Step N"
trainer success line (end of the last optimizer step), EXCLUDING the final
checkpoint write (which the orchestrator logs separately, after training).

Usage:
    python3 verify_time_budget.py <run_dir>   # e.g. outputs/reverse_text-baseline
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

TIME_RE = re.compile(r"^(\d{2}:\d{2}:\d{2})")
STEP_RE = re.compile(r"Step (\d+) \|")


def parse_times(log_path: Path) -> list[tuple[str, int]]:
    """Return (HH:MM:SS, line_no) for every line, to find first/last events."""
    out = []
    if not log_path.exists():
        return out
    with open(log_path, errors="ignore") as f:
        for i, line in enumerate(f):
            m = TIME_RE.match(line.strip("\x1b[0-9;]*m") if False else line)
            # logs are ANSI-colored; strip escape codes first
            clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
            m = TIME_RE.match(clean)
            if m:
                out.append((m.group(1), i))
    return out


def hms_to_seconds(hms: str) -> int:
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s


def find_log(run_dir: Path, name: str) -> Path | None:
    matches = sorted(run_dir.glob(f"**/{name}"))
    return matches[-1] if matches else None


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    run_dir = Path(sys.argv[1])

    orch_log = find_log(run_dir, "orchestrator.log")
    trainer_log = find_log(run_dir, "trainer.log")
    if orch_log is None or trainer_log is None:
        print(f"Could not find orchestrator.log / trainer.log under {run_dir}")
        sys.exit(1)

    with open(orch_log, errors="ignore") as f:
        orch_lines = f.readlines()
    first_dispatch_ts = None
    for line in orch_lines:
        clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
        if "Starting orchestrator loop" in clean or "orchestrator loop" in clean:
            m = TIME_RE.match(clean)
            if m:
                first_dispatch_ts = m.group(1)
                break
    if first_dispatch_ts is None:
        # fall back to the very first timestamped orchestrator line
        for line in orch_lines:
            clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
            m = TIME_RE.match(clean)
            if m:
                first_dispatch_ts = m.group(1)
                break

    with open(trainer_log, errors="ignore") as f:
        trainer_lines = f.readlines()
    last_step_ts = None
    last_step_n = None
    for line in trainer_lines:
        clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = STEP_RE.search(clean)
        if m:
            tm = TIME_RE.match(clean)
            if tm:
                last_step_ts = tm.group(1)
                last_step_n = int(m.group(1))

    if first_dispatch_ts is None or last_step_ts is None:
        print("Could not extract timestamps -- inspect logs manually.")
        print(f"orchestrator.log: {orch_log}")
        print(f"trainer.log: {trainer_log}")
        sys.exit(1)

    start_s = hms_to_seconds(first_dispatch_ts)
    end_s = hms_to_seconds(last_step_ts)
    elapsed = end_s - start_s
    if elapsed < 0:
        elapsed += 24 * 3600  # midnight wraparound
    print(f"run_dir: {run_dir}")
    print(f"first orchestrator activity: {first_dispatch_ts}")
    print(f"last training step ({last_step_n}): {last_step_ts}")
    print(f"TRAINING_TIME_SECONDS={elapsed}")
    print(f"UNDER_5MIN={'yes' if elapsed <= 300 else 'NO -- exceeds budget'}")


if __name__ == "__main__":
    main()

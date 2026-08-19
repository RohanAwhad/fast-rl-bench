#!/usr/bin/env bash
# Calibrate max_steps for a task's 5-minute training-time budget on THIS
# node's hardware (2x L40S -- different throughput than the reference
# guides' H100s, so their own step counts don't transfer). Runs the
# baseline condition for a handful of steps, reads per-step wall-clock from
# trainer.log, and prints a recommended --max-steps for run_condition.sh.
#
# See run_condition.sh's header for why tracking files live outside
# prime-rl's own --output-dir/--run.name directory.
#
# Usage: calibrate.sh <task> [n_calib_steps]
set -euo pipefail

TASK="${1:?task: reverse_text|sciknoweval}"
N="${2:-6}"
PRIME_RL_DIR="${PRIME_RL_DIR:-$HOME/prime-rl}"
REPO_DIR="${REPO_DIR:-$HOME/fast-rl-bench}"
TOML="$REPO_DIR/configs/$TASK/base.toml"
RUN_NAME="${TASK}-calib"
RUN_REL_DIR="outputs/${RUN_NAME}"
TRACK_DIR="$PRIME_RL_DIR/outputs/_runlogs/${RUN_NAME}"

mkdir -p "$TRACK_DIR"
rm -rf "${PRIME_RL_DIR:?}/${RUN_REL_DIR}"
SESSION="calib-${TASK}"
tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "Running $N calibration steps for $TASK..."
tmux new-session -d -s "$SESSION" "cd $PRIME_RL_DIR && export CUDA_VISIBLE_DEVICES=0,1 && timeout --kill-after=30 900 uv run --no-sync rl @ $TOML --max-steps $N --output-dir outputs --run.name $RUN_NAME --ckpt.interval 999999 > $TRACK_DIR/launch.log 2>&1; echo EXIT_CODE_\$? >> $TRACK_DIR/launch.log"
# This node's tmux.conf sets remain-on-exit -- override for this session so
# `tmux has-session` below actually goes false once the command finishes.
tmux set-option -t "$SESSION" remain-on-exit off 2>/dev/null || true

echo "tmux session: $SESSION -- waiting for it to finish (up to 15 min)..."
for _ in $(seq 1 180); do
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    break
  fi
  sleep 5
done

LOG="$(find "$PRIME_RL_DIR/$RUN_REL_DIR/logs" -name orchestrator.log 2>/dev/null | sort | tail -1)"
if [ -z "$LOG" ] || [ ! -f "$LOG" ]; then
  echo "no orchestrator.log found under $PRIME_RL_DIR/$RUN_REL_DIR/logs -- check $TRACK_DIR/launch.log"
  exit 1
fi
echo "using orchestrator.log: $LOG"

echo "--- orchestrator.log tail ---"
tail -40 "$LOG"

# Orchestrator's "Step N | <time> | Reward ..." lines are the real end-to-end
# (rollout + train) sink-to-sink cycle time -- what "training time" means for
# the 5-minute budget, not the trainer's own GPU-compute-only step time.
python3 - "$LOG" "$N" <<'PYEOF'
import re, sys
log_path, n = sys.argv[1], int(sys.argv[2])
times = []
ansi = re.compile(r"\x1b\[[0-9;]*m")
pat = re.compile(r"Step (\d+) \|\s*([\d.]+)s\s*\|")
with open(log_path, errors="ignore") as f:
    for line in f:
        clean = ansi.sub("", line)
        m = pat.search(clean)
        if m:
            times.append((int(m.group(1)), float(m.group(2))))
if not times:
    print("no 'Step N | Xs |' lines found in orchestrator.log -- inspect manually")
    sys.exit(1)
times.sort()
total = sum(t for _, t in times)
# First 1-2 steps carry one-off synchronization/compile overhead; use the
# later steps for the steady-state per-step rate if we have enough samples.
warmup = min(2, max(len(times) - 3, 0))
steady = [t for _, t in times[warmup:]] or [t for _, t in times]
avg_steady = sum(steady) / len(steady)
warmup_total = sum(t for _, t in times[:warmup])
print(f"Calibration: {len(times)} steps observed, total={total:.1f}s, warmup(first {warmup})={warmup_total:.1f}s, steady-state avg={avg_steady:.2f}s/step")
budget_s = 280.0  # target under the 300s hard cap, leaving margin
remaining = max(budget_s - warmup_total, 0.0)
extra_steps = int(remaining // avg_steady)
recommended = warmup + extra_steps
print(f"RECOMMENDED_MAX_STEPS={recommended}")
print(f"(targets ~{warmup_total + extra_steps * avg_steady:.0f}s of training loop time)")
PYEOF

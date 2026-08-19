#!/usr/bin/env bash
# Calibrate max_steps for a task's 5-minute training-time budget on THIS
# node's hardware (2x L40S -- different throughput than the reference
# guides' H100s, so their own step counts don't transfer). Runs the
# baseline condition for a handful of steps, reads per-step wall-clock from
# trainer.log, and prints a recommended --max-steps for run_condition.sh.
#
# Usage: calibrate.sh <task> [n_calib_steps]
set -euo pipefail

TASK="${1:?task: reverse_text|sciknoweval}"
N="${2:-6}"
PRIME_RL_DIR="${PRIME_RL_DIR:-$HOME/prime-rl}"
REPO_DIR="${REPO_DIR:-$HOME/fast-rl-bench}"
TOML="$REPO_DIR/configs/$TASK/base.toml"
RUN_NAME="${TASK}-calib"
OUT_DIR="outputs/${RUN_NAME}"

mkdir -p "$PRIME_RL_DIR/$OUT_DIR"
SESSION="calib-${TASK}"
tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "Running $N calibration steps for $TASK..."
tmux new-session -d -s "$SESSION" "cd $PRIME_RL_DIR && export CUDA_VISIBLE_DEVICES=0,1 && timeout --kill-after=30 900 uv run --no-sync rl @ $TOML --max-steps $N --output-dir outputs --run.name $RUN_NAME --ckpt.interval 999999 > $OUT_DIR/launch.log 2>&1; echo EXIT_CODE_\$? >> $OUT_DIR/launch.log"

echo "tmux session: $SESSION -- waiting for it to finish (up to 15 min)..."
for _ in $(seq 1 180); do
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    break
  fi
  sleep 5
done

LOG="$(find "$PRIME_RL_DIR/$OUT_DIR/logs" -name trainer.log 2>/dev/null | sort | tail -1)"
if [ -z "$LOG" ] || [ ! -f "$LOG" ]; then
  echo "no trainer.log found under $PRIME_RL_DIR/$OUT_DIR/logs -- check $PRIME_RL_DIR/$OUT_DIR/launch.log"
  exit 1
fi
echo "using trainer.log: $LOG"

echo "--- trainer.log tail ---"
tail -40 "$LOG"

python3 - "$LOG" "$N" <<'PYEOF'
import re, sys
log_path, n = sys.argv[1], int(sys.argv[2])
times = []
pat = re.compile(r"Step (\d+) \| Time: ([\d.]+)s")
with open(log_path) as f:
    for line in f:
        m = pat.search(line)
        if m:
            times.append((int(m.group(1)), float(m.group(2))))
if not times:
    print("no 'Step N | Time: Xs' lines found in trainer.log -- inspect manually")
    sys.exit(1)
times.sort()
# step 0 is always slow (compile/warmup); use steady-state (steps >= 1) for the rate
steady = [t for s, t in times if s >= 1] or [t for _, t in times]
avg_steady = sum(steady) / len(steady)
step0 = times[0][1] if times[0][0] == 0 else avg_steady
print(f"Calibration: {len(times)} steps observed. step0={step0:.2f}s, steady-state avg={avg_steady:.2f}s/step")
budget_s = 280.0  # target under the 300s hard cap, leaving margin
remaining = max(budget_s - step0, 0.0)
extra_steps = int(remaining // avg_steady)
recommended = 1 + extra_steps
print(f"RECOMMENDED_MAX_STEPS={recommended}")
print(f"(targets ~{step0 + extra_steps * avg_steady:.0f}s of training loop time)")
PYEOF

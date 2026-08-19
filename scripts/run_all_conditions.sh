#!/usr/bin/env bash
# Sequentially run all 6 conditions for one task (only 2 GPUs on this node,
# so no parallelism across conditions) and wait for each to finish before
# starting the next. Prints a one-line status per condition at the end.
#
# Usage: run_all_conditions.sh <task> <max_steps> <outer_timeout_s>
set -euo pipefail

TASK="${1:?task: reverse_text|sciknoweval}"
MAX_STEPS="${2:?max_steps (int)}"
OUTER_TIMEOUT="${3:?outer timeout seconds (int)}"
REPO_DIR="${REPO_DIR:-$HOME/fast-rl-bench}"
PRIME_RL_DIR="${PRIME_RL_DIR:-$HOME/prime-rl}"

CONDITIONS=(baseline duet greso difficulty_targeted experience_replay mu_grpo)

for cond in "${CONDITIONS[@]}"; do
  echo "############################################"
  echo "### $TASK / $cond ($MAX_STEPS steps, timeout ${OUTER_TIMEOUT}s)"
  echo "############################################"
  bash "$REPO_DIR/scripts/run_condition.sh" "$TASK" "$cond" "$MAX_STEPS" "$OUTER_TIMEOUT"
  SESSION="run-${TASK}-${cond}"
  # Wait for the tmux session to end (remain-on-exit disabled by run_condition.sh)
  for _ in $(seq 1 $((OUTER_TIMEOUT / 5 + 30))); do
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
      break
    fi
    sleep 5
  done
  RUN_DIR="$PRIME_RL_DIR/outputs/${TASK}-${cond}"
  TRACK_DIR="$PRIME_RL_DIR/outputs/_runlogs/${TASK}-${cond}"
  EXIT_LINE=$(tail -5 "$TRACK_DIR/launch.log" 2>/dev/null | grep -o "EXIT_CODE_[0-9]*" || echo "EXIT_CODE_UNKNOWN")
  METRICS_EXISTS="no"
  [ -f "$RUN_DIR/metrics.jsonl" ] && METRICS_EXISTS="yes"
  echo ">>> $TASK/$cond done: $EXIT_LINE, metrics.jsonl=$METRICS_EXISTS"
done

echo "############################################"
echo "### ALL $TASK CONDITIONS DONE"
echo "############################################"

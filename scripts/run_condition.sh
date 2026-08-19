#!/usr/bin/env bash
# Launch one (task, condition) run. Runs ON the GPU node (inside a tmux
# session so it survives an ssh disconnect), inside ~/prime-rl.
#
# The 5-minute hard training-time cutoff (start of first rollout -> end of
# last training step) is enforced primarily via --max-steps, calibrated
# separately per task (see calibrate.sh) so the training loop itself targets
# ~270-285s: prime-rl has no wall-clock stop, only max_steps. The outer
# `timeout` here is a generous SAFETY NET (startup + training + checkpoint
# write), not the primary cutoff -- see devlogs.md.
#
# IMPORTANT: our own tracking files (env.sh, launch.log, launch_start.ts) live
# under outputs/_runlogs/<run_name>/, NOT inside outputs/<run_name>/ (the
# directory prime-rl's --output-dir/--run.name combo resolves to) -- writing
# anything into that dir before `rl` starts trips its own
# already-contains-artifacts guard on every single launch.
#
# Usage: run_condition.sh <task> <condition> <max_steps> <outer_timeout_s> [run_suffix]
#   task:      reverse_text | sciknoweval
#   condition: baseline | duet | greso | difficulty_targeted | experience_replay | mu_grpo
set -euo pipefail

TASK="${1:?task: reverse_text|sciknoweval}"
CONDITION="${2:?condition: baseline|duet|greso|difficulty_targeted|experience_replay|mu_grpo}"
MAX_STEPS="${3:?max_steps (int)}"
OUTER_TIMEOUT="${4:?outer timeout seconds (int)}"
SUFFIX="${5:-}"

PRIME_RL_DIR="${PRIME_RL_DIR:-$HOME/prime-rl}"
REPO_DIR="${REPO_DIR:-$HOME/fast-rl-bench}"

if [ "$CONDITION" = "difficulty_targeted" ]; then
  TOML="$REPO_DIR/configs/$TASK/difficulty_targeted.toml"
else
  TOML="$REPO_DIR/configs/$TASK/base.toml"
fi
[ -f "$TOML" ] || { echo "config not found: $TOML" >&2; exit 1; }

RUN_NAME="${TASK}-${CONDITION}${SUFFIX:+-$SUFFIX}"
RUN_REL_DIR="outputs/${RUN_NAME}"          # prime-rl's own space -- never touched before launch
TRACK_DIR="$PRIME_RL_DIR/outputs/_runlogs/${RUN_NAME}"  # our tracking space
mkdir -p "$TRACK_DIR"

# Remove a prior attempt's prime-rl run dir so re-launching the same
# condition doesn't trip validate_run_dir's already-exists guard.
rm -rf "${PRIME_RL_DIR:?}/${RUN_REL_DIR}"

ENV_FILE="$TRACK_DIR/env.sh"
{
  echo "export CUDA_VISIBLE_DEVICES=0,1"
  case "$CONDITION" in
    baseline) ;;
    duet)
      echo "export EFFRL_DUET=on"
      ;;
    greso)
      echo "export EFFRL_GRESO=on"
      ;;
    difficulty_targeted)
      echo "export REPLAY_MODE=on"
      echo "export REPLAY_PRIORITY=staleness"
      echo "export REPLAY_EPS=0"
      echo "export REPLAY_FRESH_TARGET=96"
      echo "export REPLAY_USES_CAP=3"
      echo "export REPLAY_BUFFER_SIZE=512"
      echo "export REPLAY_RUN_NAME=$RUN_NAME"
      echo "export REPLAY_METRICS=$TRACK_DIR/replay_metrics.jsonl"
      ;;
    experience_replay)
      echo "export REPLAY_MODE=on"
      echo "export REPLAY_PRIORITY=staleness"
      echo "export REPLAY_EPS=0"
      echo "export REPLAY_FRESH_TARGET=96"
      echo "export REPLAY_USES_CAP=3"
      echo "export REPLAY_BUFFER_SIZE=512"
      echo "export REPLAY_RUN_NAME=$RUN_NAME"
      echo "export REPLAY_METRICS=$TRACK_DIR/replay_metrics.jsonl"
      ;;
    mu_grpo)
      echo "export REPLAY_MODE=on"
      echo "export REPLAY_PRIORITY=staleness"
      echo "export REPLAY_EPS=0"
      echo "export REPLAY_FRESH_TARGET=128"
      echo "export REPLAY_USES_CAP=4"
      echo "export REPLAY_BUFFER_SIZE=256"
      echo "export EFFRL_MUGRPO_CYCLE_K=4"
      echo "export REPLAY_RUN_NAME=$RUN_NAME"
      echo "export REPLAY_METRICS=$TRACK_DIR/replay_metrics.jsonl"
      ;;
    *)
      echo "unknown condition: $CONDITION" >&2; exit 1;;
  esac
} > "$ENV_FILE"

echo "=== $RUN_NAME ==="
echo "toml=$TOML max_steps=$MAX_STEPS outer_timeout=${OUTER_TIMEOUT}s"
cat "$ENV_FILE"

SESSION="run-${RUN_NAME}"
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" "cd $PRIME_RL_DIR && source $ENV_FILE && (date +%s.%N > $TRACK_DIR/launch_start.ts) && timeout --kill-after=60 $OUTER_TIMEOUT uv run --no-sync rl @ $TOML --max-steps $MAX_STEPS --output-dir outputs --run.name $RUN_NAME > $TRACK_DIR/launch.log 2>&1; echo EXIT_CODE_\$? >> $TRACK_DIR/launch.log"
# This node's tmux.conf sets remain-on-exit -- override for this session so
# a `tmux has-session` wait-loop actually goes false once the command finishes.
tmux set-option -t "$SESSION" remain-on-exit off 2>/dev/null || true

echo "tmux session: $SESSION (attach with: tmux attach -t $SESSION)"
echo "log: $TRACK_DIR/launch.log"
echo "run dir (prime-rl): $PRIME_RL_DIR/$RUN_REL_DIR"

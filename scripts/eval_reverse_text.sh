#!/usr/bin/env bash
# Evaluate a trained reverse-text checkpoint (20 examples x 3 rollouts,
# matching the Prime-RL run guide's protocol) and save the JSON result.
# Runs ON the GPU node.
#
# NOTE: the run guide's `vf-eval reverse-text` (legacy CLI) does not work on
# this checkout -- `reverse_text` (0.1.0, bundled under
# prime-rl/deps/verifiers/environments/reverse_text/) is a v1-style
# Taskset/Task module (no legacy `load_environment`), so it needs the
# separate v1 `eval` console script instead. Two of that CLI's own defaults
# also need overriding for a fair, apples-to-apples eval:
#   --env.agent.harness.id null    -- the CLI's own default is a "bash"
#     coding-agent harness (tool calls, edit/search, 600s tool timeout) --
#     totally wrong for this plain single-turn text task, and would eval a
#     materially different (harness-wrapped) interaction than what the model
#     was actually trained under via prime-rl's own orchestrator. `null` is
#     the framework's plain "one chat completion, no tools" harness.
#   --env.agent.runtime.type subprocess -- the CLI's own default is a
#     `prime` cloud sandbox per rollout, which requires real Prime Intellect
#     platform credentials ($PRIME_API_KEY) we don't have/want here, and
#     which isn't needed at all for a harness-less rollout. `subprocess`
#     runs locally with no sandbox/auth; it also makes `_runs_local()` true,
#     which skips the tunnel/interception setup that otherwise crashes at
#     teardown with "not authenticated with prime" even when nothing was
#     actually pushed (--no-push). See devlogs.md for the full trace.
#
# Usage: eval_reverse_text.sh <run_name> <step> [gpu_id]
set -euo pipefail

RUN_NAME="${1:?run_name, e.g. reverse_text-baseline}"
STEP="${2:?checkpoint step, e.g. 40}"
GPU="${3:-0}"
PRIME_RL_DIR="${PRIME_RL_DIR:-$HOME/prime-rl}"
REPO_DIR="${REPO_DIR:-$HOME/fast-rl-bench}"
WEIGHTS="outputs/${RUN_NAME}/weights/step_${STEP}"
RESULTS_DIR="$REPO_DIR/analysis/results"
VFEVAL_RAW_DIR="$RESULTS_DIR/vfeval_raw/${RUN_NAME}_step${STEP}"
mkdir -p "$RESULTS_DIR"
rm -rf "$VFEVAL_RAW_DIR"

if [ ! -d "$PRIME_RL_DIR/$WEIGHTS" ]; then
  echo "weights not found: $PRIME_RL_DIR/$WEIGHTS" >&2
  exit 1
fi

SESSION="eval-${RUN_NAME}"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "cd $PRIME_RL_DIR && CUDA_VISIBLE_DEVICES=$GPU uv run --no-sync inference --vllm.model $WEIGHTS --server.port 18100 > /tmp/${RUN_NAME}_eval_server.log 2>&1"
tmux set-option -t "$SESSION" remain-on-exit off 2>/dev/null || true

echo "Waiting for eval inference server (port 18100)..."
for _ in $(seq 1 60); do
  if curl -s http://localhost:18100/health >/dev/null 2>&1; then
    echo "server ready"
    break
  fi
  sleep 5
done

cd "$PRIME_RL_DIR"
uv run --no-sync eval reverse-text \
  -m "$WEIGHTS" \
  --client.base-url http://localhost:18100/v1 \
  --client.api-key-var DUMMY_API_KEY \
  --env.agent.harness.id null \
  --env.agent.runtime.type subprocess \
  --env.agent.max-output-tokens 1024 \
  -n 20 -r 3 --no-push --rich false \
  -o "$VFEVAL_RAW_DIR" \
  2>&1 | tee "$RESULTS_DIR/${RUN_NAME}_step${STEP}_vfeval.log"

tmux kill-session -t "$SESSION" 2>/dev/null || true

uv run --no-sync python3 "$REPO_DIR/scripts/summarize_vfeval_results.py" \
  "$VFEVAL_RAW_DIR" --model "$WEIGHTS" \
  --out "$RESULTS_DIR/${RUN_NAME}_step${STEP}_vfeval.json"

echo "done: $RESULTS_DIR/${RUN_NAME}_step${STEP}_vfeval.json"

#!/usr/bin/env bash
# Evaluate a trained reverse-text checkpoint with vf-eval (20 examples x 3
# rollouts, matching the Prime-RL run guide's protocol) and save the JSON
# result. Runs ON the GPU node.
#
# Usage: eval_reverse_text.sh <run_name> <step> [gpu_id]
set -euo pipefail

RUN_NAME="${1:?run_name, e.g. reverse_text-baseline}"
STEP="${2:?checkpoint step, e.g. 40}"
GPU="${3:-0}"
PRIME_RL_DIR="${PRIME_RL_DIR:-$HOME/prime-rl}"
WEIGHTS="outputs/${RUN_NAME}/weights/step_${STEP}"
RESULTS_DIR="$HOME/fast-rl-bench/analysis/results"
mkdir -p "$RESULTS_DIR"

if [ ! -d "$PRIME_RL_DIR/$WEIGHTS" ]; then
  echo "weights not found: $PRIME_RL_DIR/$WEIGHTS" >&2
  exit 1
fi

SESSION="eval-${RUN_NAME}"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "cd $PRIME_RL_DIR && CUDA_VISIBLE_DEVICES=$GPU uv run --no-sync inference --model.name $WEIGHTS --server.port 18100 > /tmp/${RUN_NAME}_eval_server.log 2>&1"

echo "Waiting for eval inference server (port 18100)..."
for _ in $(seq 1 60); do
  if curl -s http://localhost:18100/health >/dev/null 2>&1; then
    echo "server ready"
    break
  fi
  sleep 5
done

cd "$PRIME_RL_DIR"
uv run --no-sync vf-eval reverse-text \
  -m "$WEIGHTS" \
  -b http://localhost:18100/v1 \
  -n 20 --max-tokens 1024 \
  2>&1 | tee "$RESULTS_DIR/${RUN_NAME}_step${STEP}_vfeval.log"

tmux kill-session -t "$SESSION" 2>/dev/null || true
echo "done: $RESULTS_DIR/${RUN_NAME}_step${STEP}_vfeval.log"

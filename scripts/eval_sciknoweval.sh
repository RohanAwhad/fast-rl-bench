#!/usr/bin/env bash
# Evaluate a trained sciknoweval checkpoint against the held-out eval split
# (800 questions) via a plain vLLM server + eval_sciknoweval.py. Runs ON the
# GPU node.
#
# Usage: eval_sciknoweval.sh <run_name> <step> [gpu_id] [port]
set -euo pipefail

RUN_NAME="${1:?run_name, e.g. sciknoweval-baseline}"
STEP="${2:?checkpoint step, e.g. 12}"
GPU="${3:-0}"
PORT="${4:-18200}"
PRIME_RL_DIR="${PRIME_RL_DIR:-$HOME/prime-rl}"
REPO_DIR="${REPO_DIR:-$HOME/fast-rl-bench}"
WEIGHTS="outputs/${RUN_NAME}/weights/step_${STEP}"
RESULTS_DIR="$REPO_DIR/analysis/results"
mkdir -p "$RESULTS_DIR"

if [ ! -d "$PRIME_RL_DIR/$WEIGHTS" ]; then
  echo "weights not found: $PRIME_RL_DIR/$WEIGHTS" >&2
  exit 1
fi

SERVED_NAME="${RUN_NAME}-step${STEP}"
SESSION="eval-${RUN_NAME}"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "cd $PRIME_RL_DIR && CUDA_VISIBLE_DEVICES=$GPU uv run --no-sync vllm serve $WEIGHTS --served-model-name $SERVED_NAME --port $PORT --gpu-memory-utilization 0.85 --max-model-len 4096 > /tmp/${RUN_NAME}_eval_server.log 2>&1"

echo "Waiting for vLLM eval server (port $PORT)..."
for _ in $(seq 1 60); do
  if curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
    echo "server ready"
    break
  fi
  sleep 5
done

cd "$PRIME_RL_DIR"
uv run --no-sync python3 "$REPO_DIR/scripts/eval_sciknoweval.py" \
  --base-url "http://localhost:$PORT/v1" --model "$SERVED_NAME" \
  --n-samples 1 --concurrency 32 \
  --out "$RESULTS_DIR/${RUN_NAME}_step${STEP}_eval.json"

tmux kill-session -t "$SESSION" 2>/dev/null || true
echo "done: $RESULTS_DIR/${RUN_NAME}_step${STEP}_eval.json"

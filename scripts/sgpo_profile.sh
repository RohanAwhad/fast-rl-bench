#!/usr/bin/env bash
# sGPO offline profiling pass (paper §4.1): serve the task's *initial*
# checkpoint with plain vLLM, generate N=8 samples per train query under it,
# compute the empirical success rate p̂(q), and write the JSONL profile that
# the training run consumes (EFFRL_SGPO=on reads SGPO_PROFILE_FILE). The
# profiling wall-clock is OUTSIDE the 5-minute training budget — it is the
# paper's cheap one-time inference cost, amortized over the run.
#
# Usage: sgpo_profile.sh <task> [gpu_id] [port]
#   task: reverse_text | sciknoweval
set -euo pipefail

TASK="${1:?task: reverse_text|sciknoweval}"
GPU="${2:-0}"
PORT="${3:-18100}"
PRIME_RL_DIR="${PRIME_RL_DIR:-$HOME/prime-rl}"
REPO_DIR="${REPO_DIR:-$HOME/fast-rl-bench}"
PROFILE_DIR="$REPO_DIR/sgpo_profiles"
mkdir -p "$PROFILE_DIR"

case "$TASK" in
  reverse_text)
    MODEL="PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
    THRESHOLD="${SGPO_SUCCESS_THRESHOLD:-0.5}"
    ;;
  sciknoweval)
    MODEL="PrimeIntellect/Qwen3-0.6B"
    THRESHOLD="0"  # exact-match; threshold unused
    ;;
  *)
    echo "unknown task: $TASK" >&2; exit 1;;
esac

OUT="$PROFILE_DIR/${TASK}.jsonl"
SUMMARY="$PROFILE_DIR/${TASK}_summary.json"
SERVED_NAME="${TASK}-sgpo-profile"
SESSION="sgpo-profile-${TASK}"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "cd $PRIME_RL_DIR && CUDA_VISIBLE_DEVICES=$GPU uv run --no-sync vllm serve $MODEL --served-model-name $SERVED_NAME --port $PORT --gpu-memory-utilization 0.85 --max-model-len 4096 > /tmp/${TASK}_sgpo_profile_server.log 2>&1"
tmux set-option -t "$SESSION" remain-on-exit off

echo "Waiting for vLLM profiling server (port $PORT)..."
for _ in $(seq 1 120); do
  if curl -s -X POST "http://localhost:$PORT/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\": \"$SERVED_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"max_tokens\": 1}" \
      2>/dev/null | grep -q '"choices"'; then
    echo "server ready"
    break
  fi
  sleep 5
done

cd "$PRIME_RL_DIR"
uv run --no-sync python3 "$REPO_DIR/scripts/sgpo_profile.py" \
  --task "$TASK" --base-url "http://localhost:$PORT/v1" --model "$SERVED_NAME" \
  --success-threshold "$THRESHOLD" --concurrency 32 \
  --out "$OUT" --summary-out "$SUMMARY" --resume

tmux kill-session -t "$SESSION" 2>/dev/null || true
echo "done: $OUT"
echo "summary: $SUMMARY"

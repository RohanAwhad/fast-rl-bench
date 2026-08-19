#!/usr/bin/env bash
# Install this patch into a prime-rl checkout on the given host.
#
# A clean `git clone` (not an rsync'd copy) only has one real location for
# `prime_rl.orchestrator`/`configs` (under `src/` and the separate
# `packages/prime-rl-configs/` workspace member respectively) -- verified on
# this project's node (no shadowing top-level `prime_rl/` dir, unlike a prior,
# unrelated rsync'd checkout). So unlike some prior patches for this
# codebase, we deploy to exactly two places, not three.
set -euo pipefail
HOST="${1:?usage: deploy.sh <ssh-host-alias-or-user@host> [remote-prime-rl-dir]}"
REMOTE_DIR="${2:-~/prime-rl}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$PATCH_DIR/src/prime_rl"

echo "Deploying $PATCH_DIR -> $HOST:$REMOTE_DIR"

# orchestrator/ + efficient_rl/ (new) -> src/prime_rl/
for sub in orchestrator efficient_rl; do
  if [ -d "$SRC/$sub" ]; then
    tar -C "$SRC" -cf - "$sub" | ssh "$HOST" "tar -C $REMOTE_DIR/src/prime_rl -xf -"
  fi
done

# configs/ -> the separate prime-rl-configs workspace package
if [ -d "$SRC/configs" ]; then
  tar -C "$SRC" -cf - configs | ssh "$HOST" "tar -C $REMOTE_DIR/packages/prime-rl-configs/src/prime_rl -xf -"
fi

# sciknoweval env -> auto-discovered workspace member location
# (deps/prime-envs/environments/*/* is a glob workspace member in prime-rl's
# root pyproject.toml; any new directory there needs zero edits to any
# tracked prime-rl file, just `uv sync --all-packages` to pick it up).
if [ -d "$PATCH_DIR/sciknoweval_env" ]; then
  ssh "$HOST" "mkdir -p $REMOTE_DIR/deps/prime-envs/environments/science/sciknoweval"
  tar -C "$PATCH_DIR/sciknoweval_env" -cf - . | ssh "$HOST" "tar -C $REMOTE_DIR/deps/prime-envs/environments/science/sciknoweval -xf -"
fi

echo "Deployed. On the node, re-sync to register the sciknoweval workspace member (always re-specify every extra):"
echo "  cd $REMOTE_DIR && uv sync --all-packages --extra flash-attn"

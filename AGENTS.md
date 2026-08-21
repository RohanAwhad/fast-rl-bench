# AGENTS.md

Benchmarks 5 "fast RL" paper mechanisms (DUET, GRESO, Difficulty-Targeted +Replay, Experience Replay, µ-GRPO) as additive patches on Prime-RL GRPO, under a hard 5-minute training budget, on two tasks (`reverse-text`, `sciknoweval`). **Project complete**: all 12 runs (2 tasks x 6 conditions) trained + evaluated; results in `report/report.md`.

**Read `devlogs.md` before changing anything** — it holds the full design rationale, every bug found (with root causes), and the traces behind the gotchas below.

## Architecture

- This repo is **not the runtime**. It holds the patch (`prime_rl_patch/`), TOML configs (`configs/`), and run/eval/analysis scripts (`scripts/`, `analysis/`).
- Execution happens on a remote GPU node (2x L40S, `ssh -J jump@52.117.124.209 ronny_romeo@10.0.140.14`) inside `~/prime-rl` — a clean clone of PrimeIntellect-ai/prime-rl **pinned at commit `d8f3d010`**. Patches target that commit's API; do not casually upgrade prime-rl.
- The node has no rsync — git is the only sync mechanism: node clones this repo to `~/fast-rl-bench` (`git pull` before running any script sweep).
- `prime_rl_patch/deploy.sh <host> [remote-dir]` installs the patch into the prime-rl checkout, to two different locations:
  - `src/prime_rl/{orchestrator,efficient_rl}` → `~/prime-rl/src/prime_rl/`
  - `src/prime_rl/configs` → `~/prime-rl/packages/prime-rl-configs/src/prime_rl/` (separate uv workspace package)
  - `sciknoweval_env/` → `deps/prime-envs/environments/science/sciknoweval` (glob workspace member; zero tracked prime-rl edits needed)
- After deploy (or adding any env dir): `cd ~/prime-rl && uv sync --all-packages --extra flash-attn --extra disagg` — `--all-packages` is required or workspace-member envs (e.g. `reverse_text`) aren't installed; `disagg` provides the `vllm-router` binary even single-node.
- Submodule URLs are SSH and the node has no GitHub key: `git config --global url."https://github.com/".insteadOf "git@github.com:"` first.

## Running (node, inside tmux)

- `scripts/calibrate.sh <task> [n_steps]` — derive `max_steps` for the 5-min budget on this hardware.
- `scripts/run_condition.sh <task> <condition> <max_steps> <outer_timeout_s> [suffix]` — one run; `scripts/run_all_conditions.sh <task> <max_steps> <outer_timeout_s>` — all 6 conditions, strictly sequential (only 2 GPUs).
- Calibrated step budgets: **reverse_text=30, sciknoweval=25** (target ~270-285s of a 300s cap).
- Conditions are gated **purely by env vars** set in `run_condition.sh` (`EFFRL_DUET`, `EFFRL_GRESO`, `EFFRL_MUGRPO_CYCLE_K`, `REPLAY_*`) — not TOML. Exception: `difficulty_targeted` also needs `configs/<task>/difficulty_targeted.toml` (adds the `difficulty_band` filter). All patch mechanisms are off by default; baseline = zero behavior change.
- Scripts take `PRIME_RL_DIR` / `REPO_DIR` env overrides (default `~/prime-rl` / `~/fast-rl-bench`).

## Eval (node)

- `scripts/eval_reverse_text.sh <run_name> <step> [gpu]` — uses the **v1 `eval` CLI, not `vf-eval`** (`reverse_text` is a v1 Taskset package; `vf-eval` crashes with "does not expose load_environment"). Keep `--env.agent.harness.id null` and `--env.agent.runtime.type subprocess`: the CLI's own defaults (bash coding-agent harness, Prime cloud sandbox) silently eval a different interaction and crash on auth.
- `scripts/eval_sciknoweval.sh <run_name> <step> [gpu] [port]` — plain vLLM server + `scripts/eval_sciknoweval.py` (800-question held-out set); its health check must poll an actual `POST /v1/chat/completions`, not `/v1/models`.
- Checkpoints land at `outputs/<run_name>/weights/step_<N>/`.

## Hard-won gotchas (full traces in devlogs.md)

- "Training time" for the 5-min budget = sum of orchestrator.log's `Step N | Xs |` lines. NOT trainer.log (its "Step N" lines are GPU-compute-only with different numbering), NOT wall-clock log-timestamp deltas (they include dispatcher pause time). `scripts/calibrate.sh` and `analysis/verify_time_budget.py <run_dir>` both implement this methodology — keep them consistent.
- `metrics.jsonl` contains rows from BOTH the orchestrator and the trainer, with overlapping/restarting step numbers. Filter to rows carrying `train/agg/effective/agent/reward/mean` before summing `time/step` or reading step numbers.
- Never write anything into `outputs/<run_name>/` before launch — prime-rl's `validate_run_dir` raises `FileExistsError` on a non-empty run dir. Tracking files live in `outputs/_runlogs/<run_name>/` (gitignored).
- `rl` CLI: pass `--output-dir outputs --run.name <run>` — passing `outputs/<name>` as the output dir double-nests it.
- The node's tmux.conf sets `remain-on-exit on` globally; every script runs `tmux set-option -t <session> remain-on-exit off` right after creating its session. Don't remove those lines or `tmux has-session` wait-loops hang forever.
- Don't run two sweeps sharing `~/fast-rl-bench` concurrently — one `git pull` race across parallel ssh sessions caused a 404 storm that first looked like a vLLM readiness bug. Pull once, sequentially, first.
- Any new dispatcher/train_sink bookkeeping change needs a 6-8 step smoke test on the node before a full budgeted run (the DUET group-finalization bug cost a full run to catch; fixed via `Rollout.group_target_size` in `orchestrator/types.py`).
- SciKnowEval `difficulty_targeted`'s cold-start bootstrapping trap (bursty 0/N trainable early, narrow band drops ~all cold-start groups, starving its own replay buffer) is a **documented mechanism finding, not a bug** — don't "fix" the band or seed the buffer to make results look nicer.

## Analysis (plain python3, no package install)

- `python3 analysis/collect_metrics.py --outputs-dir ~/prime-rl/outputs --out-dir analysis/results`
- `python3 analysis/make_plots.py --results-dir analysis/results --out-dir report/figures` (needs matplotlib)
- Raw per-run outputs (metrics.jsonl, eval traces) live on the node; committed `analysis/results/` is the snapshot that feeds `report/` — regenerating figures without the raw node outputs is not possible.

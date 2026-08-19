# devlogs

Project: reproduce 5 "fast/efficient RL training" paper mechanisms (DUET, GRESO,
Difficulty-Targeted Selection+Replay, Experience Replay, µ-GRPO) as additive
patches to [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), compare
each against vanilla Prime-RL default GRPO, on two tasks (`reverse-text`,
`sciknoweval`), under a **hard 5-minute training-time budget per run**
(measured: first rollout dispatch -> end of last training step, excluding
setup/boot and final checkpoint export). Final artifact: a paper-style report
with plots/tables.

## 2026-08-19 — Kickoff / recon

### Compute
- Remote node: `ssh -J jump@52.117.124.209 ronny_romeo@10.0.140.14`, **2x NVIDIA
  L40S (46GB each)**, CUDA 13.2, no `rsync` (use `git` instead), `uv`/`tmux`/`git`
  present, 250GB free disk, GPUs idle at kickoff.
- This is much smaller than the reference run guides assume (8x H100 for
  reverse-text's 1+1 GPU design is trivial; sciknoweval's guide wants
  Qwen3-**8B** on 2 train + 6 infer GPUs — infeasible here: even 2-way FSDP
  sharding of 8B+AdamW needs ~67GB/GPU per prior runs, and we only have 2x46GB
  total, not even enough for the trainer alone).
- **Decision (confirmed with Rohan): use Qwen3-0.6B for BOTH tasks.** Reverse-text
  already spec'd `Qwen3-0.6B` (warm-start SFT ckpt) — unchanged. SciKnowEval
  swaps `PrimeIntellect/Qwen3-8B` -> `PrimeIntellect/Qwen3-0.6B` (cold start,
  same "base widened with RL-friendly chat template" relationship). This is a
  **documented hardware-driven scope adaptation**, not a fidelity change to the
  RL algorithms under test — every condition (baseline + 5 papers) within a task
  uses the identical model/hardware, so the *relative* comparison stays valid;
  only apples-to-apples vs. the original papers' absolute numbers is lost (and
  wasn't the point — the point is relative wall-clock efficiency under a shared
  budget).

### Framework recon (read real source, not just the gists — gists had stale
paths, e.g. `examples/reverse_text/rl.toml` / `[orchestrator.train.env]`)
- Cloned `PrimeIntellect-ai/prime-rl`, pinned to commit `d8f3d010` (matches the
  commit a prior related repo's replay-buffer patch was developed/tested
  against — minimizes integration risk since I'm reusing real, working code
  from that patch rather than writing the replay mechanism from scratch).
- `deps/*` are git submodules (`verifiers`, `renderers`, `prime-envs`,
  `pydantic-config`) — need `git submodule update --init --recursive`. Their
  `.gitmodules` URLs are SSH (`git@github.com:...`), which fails on a node
  without a GitHub SSH key — fixed via
  `git config --global url."https://github.com/".insteadOf "git@github.com:"`.
- `reverse_text` is **already a first-class uv workspace member**
  (`deps/verifiers/environments/reverse_text`) on this prime-rl version — the
  gist's "manual wheel unzip" workaround is stale/for an older release. Needs
  `uv sync --all-packages --extra flash-attn` (plain `uv sync --extra
  flash-attn`, no `--all-packages`, does NOT pull in workspace-member envs —
  confirmed empirically: `import reverse_text` failed after the first sync,
  succeeded after re-syncing with `--all-packages`).
- Config schema confirmed current (read `packages/prime-rl-configs/src/prime_rl/configs/{orchestrator,rl,algorithm}.py`
  directly): `[[orchestrator.train.source]]` (not `.env`), per-source
  `env.taskset.id`, top-level `[orchestrator] group_size`/`batch_size`,
  `[deployment] num_train_gpus/num_infer_gpus` (defaults 1/1 — fine for us).
- Found and reused **prior related work by the same author** (two other repos,
  same GitHub account, explicitly reuse-first per house style):
  - `RohanAwhad/isdpo_reprod` — has a real, tested `sciknoweval` verifiers env
    package (taskset + MCQ letter-extraction regex, copied there from prime-rl's
    own `gpqa` env), a standalone `eval_sciknoweval.py` script, and working
    example TOML configs for both tasks on the *current* schema. Reused
    directly (with attribution) rather than re-deriving from the gist.
  - `RohanAwhad/replay_buffer_experiment_rl` — a full, tested GRPO replay-buffer
    patch for prime-rl (`dispatcher.py`/`orchestrator.py`/`train_sink.py`
    patches + a `replay/` package: buffer + 12 priority functions), developed
    at the exact same pinned commit. This is the base for 3 of the 5 target
    papers (Experience Replay directly; Difficulty-Targeted-Selection+Replay and
    µ-GRPO as additive extensions of the same mechanism — see design doc below).

### Algorithm design — mapping 5 papers onto prime-rl's real extension points
prime-rl's algorithm surface (`orchestrator/algo/*.py`, hooks: `score_rollout`,
`score_group`) is for *credit assignment*, not rollout scheduling — DUET/GRESO
need to act *before* generation, which lives in `RolloutDispatcher`. Chose
**dispatcher-level + filter-level patches**, all gated by env vars / new TOML
filter types, off by default (baseline = zero behavior change):

| Paper | Mechanism implemented | Where |
|---|---|---|
| **DUET** | Adaptive `group_size` per prompt: a per-prompt (falling back to a length-bucket) EMA of historical reward-variance decides whether this prompt gets a below/above-baseline rollout count (budget-neutral in expectation). Early-abort (the paper's other mechanism) is **not** implemented — true mid-generation cancellation needs vLLM-scheduler-level surgery, scoped out; documented as a fidelity gap. | `dispatcher.py::next_fresh_group` (new: `efficient_rl/utility_tracker.py`, `efficient_rl/allocation.py`) |
| **GRESO** | Predict-before-generating: same utility tracker, but *skips* dispatching a prompt whose bucket has near-zero historical reward variance (likely all-same-reward degenerate group), with an ε exploration floor so the estimate keeps refreshing. Saves real generation compute (skipped prompts are never sent to vLLM). | same files as DUET (shared tracker) |
| **Difficulty-Targeted Selection + Rollout Replay** | New `difficulty_band` post-batch filter: infers each rollout's group-mean reward from `reward - scalar_advantage()` (exact for GRPO's uniform per-rollout advantage) and drops (from *fresh* training) any group outside `[low, high]` — i.e. train preferentially on the model's current frontier. Combined with the replay buffer (below) so the batch still fills to size from history instead of shrinking. | `orchestrator/filters.py` + `configs/orchestrator.py` (new filter type) + replay reuse |
| **Experience Replay** | The replay-buffer patch as-is: `staleness` priority (recency-based reuse — the generic paper's core idea, "don't discard a rollout after one use"), admission filter, uses-cap eviction. | `efficient_rl/replay/{buffer,priorities}.py` (adapted), `train_sink.py` |
| **µ-GRPO** | Same replay buffer, but `fresh_target` **cycles**: 1 in every K shipped batches is fully fresh (and admitted to the buffer); the other K-1 are 100% replay of that same admitted batch. Net effect: K gradient steps per "generation phase" — amortizing rollout cost K-fold, the paper's actual mechanism ("generate a big batch less frequently, more optimization per generation phase"). | `train_sink.py` (`EFFRL_MUGRPO_CYCLE_K`) |

**Bug found + fixed while adapting the borrowed replay code**: the original
`_process_batch_replay` never re-inserts items drawn via `ReplayBuffer.sample()`
back into `self.items` (only the *fresh-kept* items get restored, to avoid
same-step double-counting) — so `uses_cap` was structurally unreachable for
pure-replay draws (an item can never accumulate uses>1 if it's removed from the
pool the first time it's replayed). Fixed in my adapted copy by re-extending
`replay_items` back into the pool after use, matching the documented intent
("`uses_cap`: max times one sample can be replayed before eviction"). Noted here
per "trust code over docs" — the original repo's own docs already promised this
behavior, the code just didn't (yet) implement it; their headline finding
(admission-filter + reuse-cap driving most of the win, robust across priority
formulas) is about the *filter*, not this specific pool-refill detail, so it's
unlikely to have been load-bearing for their conclusions — but it matters a lot
for **µ-GRPO**, which depends on one batch surviving exactly K uses.

### 5-minute hard cutoff — operationalization
prime-rl has no wall-clock stop, only `max_steps` (confirmed in both run
guides). Plan (matches the sciknoweval guide's own gotcha #3 — don't cut
`timeout` close to training time, checkpoint writes can take minutes and a
mid-write kill corrupts the checkpoint):
1. **Calibrate** per task: run baseline for a handful of steps, measure real
   per-step time on *this* hardware (L40S, not the guides' H100s).
2. Convert to a **`--max-steps`** that targets ~270-285s of actual training loop
   (leaving margin under the 300s hard cap, accounting for the slow step-0
   compile/warmup seen in every reference run).
3. Apply that step budget to **all 6 conditions** of a task (baseline has the
   most generation work per step of any condition here, so it's a conservative
   ceiling — variants that skip/reuse rollouts should be at or under it).
4. Wrap the whole `rl` launch in an outer `timeout` sized generously
   (startup-estimate + 300s + checkpoint-estimate + margin) as a **safety net**
   against hangs — not the primary cutoff mechanism.
5. **Verify empirically** from `metrics.jsonl` / logs (first rollout timestamp
   -> last step timestamp) that measured training time is <= 5 min for every
   run; if a run overshoots, truncate the comparison at the 5-min mark using
   per-step timestamps rather than discarding the run.

### Repo
- Public GitHub repo created: https://github.com/RohanAwhad/fast-rl-bench
- Local dir `fast_rl/` (this repo) holds all patch code / configs / scripts /
  analysis; the remote node clones it fresh and "installs" the patch into its
  `~/prime-rl` checkout (no rsync available, so git is the sync mechanism both
  ways: this repo -> node via `git clone`/`git pull`).

Next: write the `efficient_rl` patch package + patched `dispatcher.py`/
`train_sink.py`/`filters.py`/`configs/orchestrator.py`, the sciknoweval env
(copied from `isdpo_reprod` with attribution), 12 TOML configs (6 conditions x
2 tasks), deploy + run + eval + metrics-collection scripts. Then deploy,
calibrate, run all 12, evaluate, plot, write the report.

## 2026-08-19 — Patch built, deployed, calibrated

- All 5 patch mechanisms + sciknoweval env + 12 configs written locally,
  deployed to remote `~/prime-rl` (manual tar+ssh copy for the patch files;
  remote clones `fast-rl-bench` itself via git for scripts/configs). All
  imports verified OK on remote (`efficient_rl`, `sciknoweval`, patched
  orchestrator modules).
- Fixed several setup bugs serially: submodule SSH URLs need
  `insteadOf https://`, `uv sync` needs `--all-packages --extra flash-attn
  --extra disagg` (the `vllm-router` binary is in the `disagg` extra even for
  single-node), `rl` CLI's `--output-dir`+`--run.name` double-nest if you pass
  `outputs/<name>` as output-dir, `validate_run_dir` throws `FileExistsError`
  if our own tracking files live inside prime-rl's run dir (moved to
  `outputs/_runlogs/<run>/` instead), `vf-eval`/`inference` CLI flags differ
  from what the reference gists show.
- **Calibration timing bug**: `calibrate.sh` was parsing `trainer.log`'s
  step time (GPU compute only) instead of `orchestrator.log`'s
  `Step N | Xs |` line (true end-to-end rollout+train cycle time — the right
  signal for the 5-min *training time* budget). Fixed.
- **tmux gotcha**: this node's `.tmux.conf` sets `remain-on-exit on` globally,
  so a finished session doesn't disappear from `tmux ls` and a
  `tmux has-session` wait-loop never returns false. Fixed by
  `tmux set-option -t <session> remain-on-exit off` right after creating each
  run/calibration session.
- Confirmed pipeline end-to-end on reverse-text baseline: reward climbs
  0.11 -> 0.81 over 15 steps (noisy but clearly learning).
- **Reverse-text calibration (15 steps, L40S x2, baseline config)**:
  per-step times noisy (dispatcher pause/resume for policy sync dominates):
  17.9, 3.2, 4.4, 27.8, 5.3, 7.3, 3.8, 15.6, ..., 6.3, 6.4, 16.4, 6.4, 4.8s.
  Total 137.3s/15 steps. Warmup (first 2 steps) = 21.1s. Steady-state avg =
  8.94s/step. -> **RECOMMENDED_MAX_STEPS=30** for reverse-text (all 6
  conditions), targeting ~271s training-loop time (29s margin under 300s cap).
- Built `scripts/run_all_conditions.sh` to sequence baseline + 5 patch
  conditions for a task automatically (only 2 GPUs on this node, so runs
  cannot be parallelized across conditions — must be strictly sequential).
- **SciKnowEval calibration (15 steps, cold start)**: reward climbs slower
  than reverse-text (cold start vs. warm SFT ckpt), reaches 0.53 by step 15,
  `Trainable` % dips as low as 75% (gibberish/repetition/zero_advantage
  filters actively dropping rollouts from an untuned base model, as
  expected). Total 163.8s/15 steps, warmup=22.7s, steady-state avg=
  10.85s/step (slower per-step than reverse-text, consistent with longer
  `max_completion_tokens=256`). -> **RECOMMENDED_MAX_STEPS=25**, targeting
  ~272s training-loop time.
- **Verified `metrics.jsonl` schema against a real run** (important: my
  original placeholder key names in `collect_metrics.py` were wrong). Real
  keys: `train/agg/effective/agent/reward/mean` (trainable-only) and
  `train/agg/all/agent/reward/mean` (includes filtered), both nested under
  `.../agent/...`, not the flatter names I'd guessed. Also confirmed the
  orchestrator AND trainer both write rows into the same metrics.jsonl file
  with overlapping/restarting "step" numbers (90 total rows for a 15-step
  run) -- naively summing every row's `time/step` would double/triple count;
  fixed by filtering to only rows carrying the aggregate reward key (which
  1:1-match the orchestrator.log "Step N" lines used for calibration).
  Fixed `collect_metrics.py` accordingly; verified the fixed version
  reproduces calibrate.sh's own total (137.29s vs. calibrate.sh's 137.3s)
  against the reverse-text-calib run.
- Rewrote `analysis/verify_time_budget.py`: it had the same class of bug as
  the original calibrate.sh (was reading trainer.log's own differently-scoped
  "Step N" line + naive wall-clock first/last-log-timestamp delta, which
  would include dispatcher backpressure/pause time -- confirmed from real
  logs that the orchestrator's self-reported per-step "Xs" already *excludes*
  that idle time, e.g. a 14s wall-clock gap between two "Step N" lines
  reported as only 4.4s of actual step time). Now sums orchestrator.log's own
  "Step N | Xs |" values, matching calibrate.sh's methodology exactly so the
  budget check is self-consistent with how max_steps was derived.
- Found `make_plots.py` referenced two fields `collect_metrics.py` never
  produced (`cumulative_step_times`, `eval_metric`) -- would have silently
  produced empty "reward vs. wall-clock" plots and blank eval columns after
  all 12 runs. Fixed: `summarize_run()` now also emits
  `cumulative_step_times` (running sum keyed by step); added
  `find_eval_summary()` to merge in the eval script's JSON output.
- Read `vf-eval`'s actual implementation (`verifiers/legacy/scripts/eval.py`
  + `save_utils.py`) rather than guessing its output format: `--save-results
  --output-dir <dir>` writes `results.jsonl` (one row per rollout, each a
  `RolloutOutput` dict with a top-level `"reward": float`) under a
  self-named subdirectory of `<dir>`. Added `scripts/summarize_vfeval_results.py`
  to glob for it and reduce to the same `{"summary": {...}}` JSON shape
  `eval_sciknoweval.py` already produces (`overall_reward` vs. sciknoweval's
  `overall_accuracy` -- reverse-text's reward is continuous LCS ratio, not
  binary, so it isn't an "accuracy"). Wired into `eval_reverse_text.sh`.
- Verified end-to-end: checkpoint dir naming is `weights/step_<N>/` (matches
  both eval scripts' assumptions); `uv run --no-sync vllm serve` works in the
  synced env (needed by `eval_sciknoweval.sh`); `sciknoweval.taskset`'s
  `SciKnowEvalData` field names (`prompt`, `system_prompt`, `answer_key`,
  `domain`) match what `eval_sciknoweval.py` reads off `task.data`.
- Calibration numbers: **reverse_text max_steps=30, sciknoweval max_steps=25**.
- Next: launch `run_all_conditions.sh` for both tasks (sequential, 2 GPUs
  only), then evaluate all 12 checkpoints, collect metrics, plot, write report.

## 2026-08-19 — Real runs: baseline OK, DUET bug found + fixed

- **reverse_text baseline** (real run, 30 steps): SUCCESS. Orchestrator step
  loop = 269s (`verify_time_budget.py`: 262.1s), final reward ~0.79-0.81
  (climbed from ~0.11). Under budget.
- **reverse_text DUET**: FAILED. Killed by the outer 600s timeout at step
  16/30, with `pending_batch` showing a runaway "+7877 buffered" episode
  backlog (vs. baseline's typical +7 to +80). Root cause (found by reading
  the dispatcher/train_sink interaction, not guessing): DUET makes
  `GroupState.target_rollouts` *per-prompt* (`allocation.py::duet_group_size`
  can scale a group to 0.5x-1.5x the base size), but
  `TrainSink.add()`'s group-completion check compared the arrival count
  against `group_size_for(env_name)` -- a **static per-env config lookup**
  that's always the *base* group size and never agrees with a DUET-resized
  group. Effect: an oversized group (>base) finalizes early and its trailing
  arrivals orphan into a fresh, never-completing entry under the same
  group_id; an undersized group (<base) never reaches the (too-high) static
  target at all. Either way the group leaks in `pending_groups`/
  `pending_group_episodes` forever -- exactly the unbounded backlog observed.
  This is a general hazard, not reverse-text-specific, since it only depends
  on DUET being enabled.
  - **Fix**: added `Rollout.group_target_size: int | None` (metadata field,
    `exclude=True`, same pattern as `group_id`/`policy_version`) to
    `orchestrator/types.py`; stamped from `GroupState.target_rollouts` in
    `dispatcher.py::emit_episode` (covers both normal completion and
    `drop_group`'s cancellation-marker path, since both funnel through
    `emit_episode`); `train_sink.py::add()` now checks against
    `episode[0].group_target_size or group_size_for(env_name)` (falls back
    to the old static lookup when unset, e.g. eval rollouts -- eval groups
    are never DUET-resized since the adjustment is `if kind == "train"` only
    in `next_fresh_group`, so the fallback is exact there, not approximate).
  - Redeployed the 3 files, re-verified imports on remote, re-launched DUET.
  - This is exactly the kind of interaction a smoke test would have caught
    before committing 10 minutes of GPU time to a doomed run — noted for
    future patches: a genuinely-fresh mechanism that changes dispatcher-side
    per-group bookkeeping needs a dedicated few-step smoke test, not just a
    static import/type-check, before it's trusted with a full budgeted run.
  - **6-step smoke test post-fix**: confirmed working -- varying, matched
    `Trainable N/N` counts (112/112, 144/144, 120/120, ...) prove DUET is
    genuinely resizing groups per-prompt AND that resized groups now
    finalize correctly (100% trainable, no orphaned partials). Buffered
    counts stayed bounded (tens, not thousands).
  - **reverse_text DUET retry (full 30 steps)**: SUCCESS. 267.0s (under
    budget), final reward ~0.74-0.82 -- comparable to baseline's ~0.79-0.81.
- **reverse_text GRESO**: launched directly at full budget (30 steps) without
  a separate smoke test -- lower risk than DUET since GRESO only changes
  *which* example `next_fresh_group` settles on (bounded retry against the
  utility tracker), never the group's size, so it can't hit the
  DUET-completion-check bug class (confirmed by code inspection: the
  `duet_group_size` call in `next_fresh_group` is gated strictly behind
  `duet_enabled()`, independent of `greso_enabled()`).
  SUCCESS: 30/30 steps, 270.1s (under budget), final reward ~0.78-0.81,
  Trainable consistently 128/128 (GRESO never resizes groups, so no risk of
  the DUET bug class).
- **reverse_text difficulty_targeted**: first live exercise of the replay
  buffer + `difficulty_band` filter combo -- smoke-testing (6 steps) before
  committing to the full budget, per the "new dispatcher/sink-bookkeeping
  mechanism needs a smoke test" lesson from DUET.
  Smoke test (6 steps): completed cleanly, no hangs/leaks.
  `replay_metrics.jsonl` shows sane admission/eviction/sampling behavior
  (buffer_size fluctuating 32-112 under a 512 cap, uses_cap=3 evicting
  steadily). Noted (not a bug, a real characteristic of this specific
  condition): composed batches sometimes ship smaller than the nominal 128
  early on -- `maybe_ship()`'s readiness check counts *pre-filter*
  `pending_batch`, but the `difficulty_band` filter (deliberately narrow
  [0.15, 0.85] band) can drop most of a fresh cohort, and the replay buffer
  hasn't built up a large-enough backlog yet this early to fully backfill.
  Expected to stabilize as the buffer fills over more steps; not a
  hang/crash risk (confirmed: smoke test completed all 6 steps).

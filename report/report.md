# Fast RL: Reproducing Time-Efficient RL Training Methods Under a 5-Minute Budget

*A comparison of DUET, GRESO, Difficulty-Targeted Selection + Rollout Replay,
Experience Replay, and µ-GRPO against vanilla Prime-RL GRPO, on `reverse-text`
and `SciKnowEval`, each run capped at 5 minutes of training wall-clock.*

## Abstract

*(filled in after runs complete)*

## 1. Introduction

Prime-RL's own recommended defaults for two representative RL tasks —
`reverse-text` (warm-started Qwen3-0.6B) and `SciKnowEval` (cold-started
Qwen3, multi-domain scientific MCQ) — serve as the baseline against which we
reproduce the core mechanisms of five recent RL-training-efficiency papers as
additive patches to Prime-RL. Every condition trains for a **hard 5-minute
wall-clock budget**, measured from the start of the first rollout dispatch to
the end of the last training step (excluding process setup and final
checkpoint export). This isolates the question each paper actually asks:
*given a fixed time budget, which training-loop design gets furthest?*

## 2. Methods

### 2.1 Tasks

| | reverse-text | SciKnowEval |
|---|---|---|
| Model | `Qwen3-0.6B` (warm-started from `PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT`) | `PrimeIntellect/Qwen3-0.6B` (cold start) |
| Reward | continuous LCS ratio [0,1] | binary exact-match on MCQ letter |
| batch_size / group_size | 128 / 16 | 128 / 16 |
| max_completion_tokens | 128 | 256 |
| seq_len | 2048 | 4096 |
| lr | 3e-6 | 5e-6 |

**Hardware deviation (documented):** the SciKnowEval task's originally
recommended model is `PrimeIntellect/Qwen3-8B` on 2 train + 6 inference GPUs.
Our node has 2x NVIDIA L40S (46GB each) total — not enough even for the 8B
trainer alone (2-way FSDP-sharded 8B+AdamW needs ~67GB/GPU per prior
measurements on this exact codebase). We use Qwen3-0.6B for both tasks
instead. Every condition on a task uses the identical model/hardware, so the
*relative* comparison between conditions remains valid; only apples-to-apples
comparison against the original papers' absolute numbers is affected.

### 2.2 The 5-minute hard cutoff

Prime-RL has no native wall-clock stop (only `--max-steps`). We calibrate
`--max-steps` per task from a short baseline run's steady-state per-step
timing on this hardware (2x L40S — throughput differs from the reference
guides' H100 nodes), targeting ~270-285s of training-loop time, and verify
after the fact from `metrics.jsonl`'s `time/step` field that every run's
summed training time is <= 300s. Each launch is additionally wrapped in an
outer `timeout` sized generously above the expected total (startup +
training + checkpoint export) as a safety net against hangs, not as the
primary cutoff mechanism.

Reverse-text calibration (baseline config, 15 steps, 2x L40S): steady-state
avg 8.94s/step after a 21.1s warmup (first 2 steps) -> **`--max-steps 30`**
(~271s training-loop time).

SciKnowEval calibration (baseline config, 15 steps, cold start): steady-state
avg 10.85s/step after a 22.7s warmup -> **`--max-steps 25`** (~272s
training-loop time). Slower per-step than reverse-text, consistent with its
longer `max_completion_tokens` (256 vs. 128).

The same step budget is applied to all 6 conditions of a task: baseline does
the most generation work per step of any condition (nothing is skipped or
reused), so it is a conservative ceiling — variants that skip/reuse rollouts
should train in equal or less wall-clock time for the same step count.

### 2.3 Conditions

All five paper mechanisms are implemented as additive, env-var/TOML-gated
patches on top of vanilla Prime-RL (commit `d8f3d010`) — a baseline run
touches none of this code. See `devlogs.md` for the full design rationale and
documented fidelity gaps.

| Condition | Mechanism | Implementation |
|---|---|---|
| Baseline | Prime-RL default GRPO | vanilla, unmodified |
| DUET | Adaptive per-prompt rollout allocation from a historical reward-variance tracker | `dispatcher.py::next_fresh_group` |
| GRESO | Skip prompts predicted to yield zero-variance groups, before generation | same tracker, dispatch-time skip |
| Difficulty-Targeted Selection + Replay | Train preferentially on groups in `[0.15, 0.85]` mean-reward band; backfill batch from a replay buffer | new `difficulty_band` post-batch filter + replay buffer |
| Experience Replay | Reuse recently-generated rollouts (`staleness` priority, uses-cap 3) instead of discarding after one step | replay buffer |
| µ-GRPO | 1 fully-fresh batch every K=4 ships; the other 3 ship 100% replay of that batch (4 gradient steps per generation phase) | replay buffer + fresh/replay cycling |

### 2.4 Evaluation

- reverse-text: `vf-eval reverse-text`, 20 examples x 3 rollouts (matches
  Prime-RL's own run guide protocol), on the final checkpoint of each
  condition.
- SciKnowEval: standalone `eval_sciknoweval.py` against the held-out 800-row
  eval split (200/domain), mean@1, on the final checkpoint of each condition.

## 3. Results

### 3.1 reverse-text

*(reward-vs-step plot, reward-vs-wall-clock plot, summary table — filled in
after runs)*

### 3.2 SciKnowEval

*(same — filled in after runs)*

## 4. Discussion

*(filled in after runs — what worked, what didn't, honest fidelity-gap
retrospective)*

## 5. Fidelity gaps (documented up front, not discovered after the fact)

- **DUET**: mid-generation early-abort of unproductive trajectories is not
  implemented (would need vLLM-scheduler-level cancellation inside a still-
  dispatching group); only the rollout-*reallocation* mechanism is
  implemented.
- **GRESO / DUET utility signal**: approximated by an online per-prompt (with
  length-bucket fallback) EMA of historical within-group reward variance,
  not a learned predictor as in the original paper.
- **Replay-based conditions**: priority fixed to `staleness` (simple
  recency) across all three replay-based conditions to isolate each paper's
  own distinguishing mechanism (selection filter for Difficulty-Targeted,
  fresh/replay cycling for µ-GRPO) from priority-formula choice.
- **SciKnowEval model size**: Qwen3-0.6B instead of the original Qwen3-8B
  (hardware constraint, see 2.1).

## Appendix: reproduction

See `README.md` / `devlogs.md` for the full setup. All configs under
`configs/`, patch code under `prime_rl_patch/`, run/eval/analysis scripts
under `scripts/` and `analysis/`.

# fast-rl-bench

Reproducing six "fast/efficient RL training" paper mechanisms as additive
patches to [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), compared
against vanilla Prime-RL default GRPO, under a **hard 5-minute training-time
budget per run**, on two tasks: `reverse-text` (Qwen3-0.6B, warm-start) and
`sciknoweval` (Qwen3-0.6B, cold-start).

Papers compared:
- **DUET** — adaptive per-prompt rollout allocation
- **GRESO** — skip likely-degenerate (zero-variance) prompts before generation
- **Difficulty-Targeted Selection + Rollout Replay** — train on the model's
  current frontier, backfill batches from a replay buffer
- **Experience Replay** — reuse recently-generated rollouts instead of
  discarding after one gradient step
- **µ-GRPO** — generate a big batch infrequently, take several gradient steps
  per generation phase
- **sGPO** — offline-profiled data selection + adaptive group size (G ∈
  {2,4,8} by 1/p̂ bucket) + easy-to-hard curriculum phases

See `devlogs.md` for the full design rationale, hardware constraints, and
fidelity-gap notes. See `report/` for the final paper-style writeup with plots
and tables once runs complete.

## Layout

```
prime_rl_patch/        # additive patch on top of prime-rl @ d8f3d010 (env-var / TOML gated, off by default)
  deploy.sh             #   installs the patch into a prime-rl checkout
  src/prime_rl/         #   patched dispatcher.py / train_sink.py / filters.py / configs
  sciknoweval_env/       #  verifiers taskset package for SciKnowEval
configs/                # 7 conditions x 2 tasks, TOML
scripts/                # calibration / run / eval / metrics-collection
analysis/               # plots + collected results
report/                 # final report
```

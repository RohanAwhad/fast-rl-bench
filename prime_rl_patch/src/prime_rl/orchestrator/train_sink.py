"""TrainSink: three-level rollout sink for the training side.

1. ``process_rollout`` — eager per-rollout tokenization (overlaps with
   dispatcher producing more rollouts), then the env algorithm's
   ``finalize_rollout`` (rollout-local scoring + any reference I/O). Errored
   and untrainable rollouts skip this.
2. ``process_group`` — filters errored rollouts, hands the trainable
   survivors to the env algorithm's ``finalize_group`` (advantages +
   per-sample wire stamping), runs the pre-batch filter pass.
3. ``process_batch`` — applies post-batch filter annotations and assembles
   the trainer-bound ``TrainingSample`` list. Returns a ``TrainBatch``.

``add()`` takes one episode (``list[Rollout]``) and returns
``TrainBatch | None``; group accounting counts episodes, never loose traces.
I/O concerns (ship to trainer, monitors.log) live on the
orchestrator.

--- efficient_rl (Experience Replay / Difficulty-Targeted Replay / mu-GRPO) ---
Replay integration adapted from a prior, tested replay-buffer harness for
prime-rl by the same author (``RohanAwhad/replay_buffer_experiment_rl``,
pinned to the same prime-rl base). Two changes from that harness:

1. **Bug fix**: the source harness's ``_process_batch_replay`` never
   reinserted items drawn via ``ReplayBuffer.sample()`` back into the pool
   (only *fresh-kept* items were restored, to dodge same-step double
   counting) — so ``uses_cap`` was structurally unreachable for pure-replay
   draws. Fixed here via ``self.replay.readmit(replay_items)`` after use, so
   an item is only retired once it has genuinely been used
   ``uses_cap`` times (see ``efficient_rl/replay/buffer.py`` docstring).
2. **New**: mu-GRPO's fresh/replay cycling (``EFFRL_MUGRPO_CYCLE_K``) — every
   Kth ship is fully fresh (and admitted to the buffer); the other K-1 ship
   100% replay of that same admitted batch, i.e. K gradient steps per
   generation phase instead of prime-rl's default 1:1.

Both are no-ops (identical to vanilla) when ``REPLAY_MODE`` is unset.
"""

from __future__ import annotations

import asyncio
import os
import random
import uuid
from collections import defaultdict

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.efficient_rl.replay.buffer import ReplayBuffer, ReplayParams
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.filters import RolloutFilter, apply_filters
from prime_rl.orchestrator.metrics import TrainRollouts
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.types import Rollout, TrainBatch
from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger


def replay_params_from_env(run_name: str = "") -> ReplayParams:
    """Build ReplayParams from environment variables."""
    return ReplayParams(
        enabled=os.environ.get("REPLAY_MODE", "off") == "on",
        buffer_size=int(os.environ.get("REPLAY_BUFFER_SIZE", "2048")),
        fresh_target=int(os.environ.get("REPLAY_FRESH_TARGET", "32")),
        uses_cap=int(os.environ.get("REPLAY_USES_CAP", "3")),
        eps=float(os.environ.get("REPLAY_EPS", "0.05")),
        priority=os.environ.get("REPLAY_PRIORITY", "staleness"),
        priority_params=dict(
            (k, float(v))
            for kv in os.environ.get("REPLAY_PRIORITY_PARAMS", "").split(",")
            if kv
            for k, v in [kv.split("=", 1)]
        ),
        sampling=os.environ.get("REPLAY_SAMPLING", "proportional"),
        temp=float(os.environ.get("REPLAY_TEMP", "1.0")),
        group_cap=int(os.environ.get("REPLAY_GROUP_CAP", "16")),
        metrics_path=os.environ.get("REPLAY_METRICS", ""),
        run_name=run_name,
    )


def _mugrpo_cycle_k() -> int:
    return int(os.environ.get("EFFRL_MUGRPO_CYCLE_K", "0") or 0)


def payload_tokens(rollout: Rollout) -> int:
    """Token cost of the rollout's trainer-bound payload — the samples built by
    ``process_rollout``. This is what actually ships: forked traces can drop
    branches with no trainable tokens, so ``Trace.num_total_tokens`` (which sums
    over all branches) may overcount. For linear traces the two agree.

    Zero-payload rollouts (no trainable samples at all) fall back to the trace
    total so they still advance token batching — a degenerate all-zero-payload
    stream then ships empty batches and trips the orchestrator's
    consecutive-empty-batch abort instead of stalling the readiness check."""
    return sum(len(sample.token_ids) for sample in rollout.samples) or rollout.num_total_tokens


class TrainSink:
    """Three-level train sink. Constructed once, fed via ``add(rollout)``."""

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        tokenizer,
        train_envs: TrainEnvs,
        mm_token_type_ids_mapping: dict[int, int] | None,
        batch_size: int | None,
        token_batch_size: int | None,
        pre_filters: list[RolloutFilter],
        post_filters: list[RolloutFilter],
    ) -> None:
        assert (batch_size is None) != (token_batch_size is None), (
            "Exactly one of batch_size / token_batch_size must be set"
        )
        self.config = config
        self.tokenizer = tokenizer
        self.train_envs = train_envs
        self.mm_token_type_ids_mapping = mm_token_type_ids_mapping
        self.batch_size = batch_size
        self.token_batch_size = token_batch_size
        self.pre_filters = pre_filters
        self.post_filters = post_filters

        # Observation window for the next shipped batch: rollouts of groups
        # finalized since the last ship (errored + filtered + survivors).
        # In-progress groups stay out until they finalize.
        self.pending_rollouts: TrainRollouts = TrainRollouts()
        # Keyed by the dispatcher's group UUID. ``(env_name, task_idx)``
        # isn't unique — the same task can be re-sampled while an
        # earlier group is still in flight
        self.pending_groups: dict[uuid.UUID, list[Rollout]] = defaultdict(list)
        # Episodes arrived per group — the finalization count (an episode may
        # add several traces to ``pending_groups`` but counts once here).
        self.pending_group_episodes: dict[uuid.UUID, int] = defaultdict(int)
        self.pending_batch: list[Rollout] = []
        # Running payload-token total of ``pending_batch`` (token-batched
        # runs), kept in sync on append/pop so the readiness check never
        # re-sums per arrival.
        self.pending_tokens: int = 0

        # --- efficient_rl replay (off unless REPLAY_MODE=on) ---
        self.replay: ReplayBuffer | None = None
        self._group_stats: dict[uuid.UUID, tuple[float, float]] = {}
        self._shipped = 0
        self._mugrpo_cycle_k = _mugrpo_cycle_k()
        p = replay_params_from_env(run_name=os.environ.get("REPLAY_RUN_NAME", ""))
        if p.enabled:
            self.replay = ReplayBuffer(p, random.Random(int(os.environ.get("REPLAY_SEED", "0"))))
            get_logger().info(
                f"Replay buffer ENABLED: priority={p.priority} params={p.priority_params} "
                f"eps={p.eps} uses_cap={p.uses_cap} buffer={p.buffer_size} fresh_target={p.fresh_target}"
                + (f" mugrpo_cycle_k={self._mugrpo_cycle_k}" if self._mugrpo_cycle_k > 1 else "")
            )

        # Reset by the orchestrator after each ship via ``reset_pre_filter_stats``
        self.pre_filter_seen = 0
        self.pre_filter_dropped = 0
        self.pre_filter_dropped_by_name: dict[str, int] = {}

    def group_size_for(self, env_name: str) -> int:
        return self.train_envs.get(env_name).config.group_size

    def batch_progress(self) -> tuple[int, int, str]:
        """``(current, target, unit)`` for the train batch — counts only
        ``pending_batch`` (survivors of finalized groups, queued for the
        trainer), so it's an honest 0→target fill. Partial-group arrivals are
        reported separately by ``buffered_count()``."""
        if self.batch_size is not None:
            return len(self.pending_batch), self.batch_size, "rollouts"
        assert self.token_batch_size is not None
        return self.pending_tokens, self.token_batch_size, "tokens"

    def buffered_count(self) -> int:
        """Episodes that have arrived but sit in not-yet-complete groups —
        buffered in the sink ahead of the batch."""
        return sum(self.pending_group_episodes.values())

    def pending_batch_by_env(self) -> dict[str, int]:
        """Per-env breakdown of ``batch_progress()`` (``pending_batch`` only);
        values sum to the aggregate."""
        counts: dict[str, int] = defaultdict(int)
        for r in self.pending_batch:
            counts[r.env_name] += 1
        return dict(counts)

    async def add(self, episode: list[Rollout]) -> TrainBatch | None:
        """Process one episode arrival; finalize the group on the
        ``group_size``-th episode; return a ``TrainBatch`` if the finalization
        pushed (or left) the batch over its threshold. Arrivals into
        still-incomplete groups never ship a batch."""
        group_id = episode[0].group_id
        env_name = episode[0].env_name
        for rollout in episode:
            await self.process_rollout(rollout)
        self.pending_groups[group_id].extend(episode)
        self.pending_group_episodes[group_id] += 1
        if self.pending_group_episodes[group_id] < self.group_size_for(env_name):
            return None
        await self.process_group(group_id)
        # ``pending_batch`` only grows on group finalization, so readiness is
        # only re-checked here — the window of a shipped batch then always
        # contains at least the group that finalized it.
        if self.replay is not None:
            return self.maybe_ship()
        ready = (
            len(self.pending_batch) >= self.batch_size
            if self.batch_size is not None
            else self.pending_tokens >= (self.token_batch_size or 0)
        )
        if ready:
            return self.process_batch()
        return None

    def maybe_ship(self) -> TrainBatch | None:
        """Replay mode: ship a composed batch whenever fresh pending + buffer
        can fill one; used both after episode arrival and from the
        orchestrator's idle hook (pure starvation: replay-only steps -- this
        is what lets mu-GRPO's replay-only ships fire back-to-back without
        waiting on fresh generation)."""
        if self.replay is None:
            return None
        target = self.batch_size if self.batch_size is not None else self.token_batch_size or 0
        fresh = len(self.pending_batch) if self.batch_size is not None else self.pending_tokens
        if fresh + self.replay.available() < target:
            return None
        return self.process_batch()

    async def process_rollout(self, rollout: Rollout) -> None:
        """Build training samples from the rollout's Trace (one per branch), walking the
        message graph. Training is renderer-only across all modes (RL/OPD student, SFT teacher),
        so every node already carries its tokens. Errored rollouts are dropped at the group
        level, so skip them here; untrainable traces never become training data."""
        if rollout.has_error or not rollout.agent.trainable:
            return
        samples = await asyncio.to_thread(
            trace_to_samples,
            rollout,
            env_name=rollout.env_name,
            mm_token_type_ids_mapping=self.mm_token_type_ids_mapping,
        )
        rollout.samples = samples or []
        # Arrival phase: rollout-local scoring (raw reward, echo observation
        # weighting, opd/opsd reference logprobs) runs as soon as the rollout is
        # tokenized — before its group is complete.
        await self.train_envs.get(rollout.env_name).algorithm.finalize_rollout(rollout)

    async def process_group(self, group_id: uuid.UUID) -> None:
        """Finalize one GRPO group: drop errored rollouts, assign advantages,
        run pre-batch filters, append survivors to ``pending_batch``."""
        group = self.pending_groups.pop(group_id, [])
        self.pending_group_episodes.pop(group_id, None)
        if not group:
            return
        # Window membership follows group finalization, not arrival: a rollout
        # only becomes observable (metrics / persistence) once its whole group
        # is finalized, so a batch's window never claims rollouts of a group
        # that ships later. Dropped groups still land here — they were observed.
        for r in group:
            self.pending_rollouts.append(r)
        env_name = group[0].env_name
        task_idx = group[0].task.data.idx
        survivors = [r for r in group if not r.has_error]
        num_errored = len(group) - len(survivors)

        env = self.train_envs.get(env_name)
        # Untrainable traces carry no samples and must not skew the group baseline.
        survivors = [r for r in survivors if r.agent.trainable]
        if not survivors:
            get_logger().debug(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"rollouts={len(group)} (errored={num_errored}) | dropped: no trainable survivors"
            )
            return

        # Advantages + per-sample wire stamping (advantage stream, loss
        # routing) are the algorithm's job (finalize_group); the sink only
        # owns the grouping mechanics.
        await env.algorithm.finalize_group(survivors)

        if self.replay is not None:
            rewards = [r.reward for r in survivors]
            mean = sum(rewards) / max(len(rewards), 1)
            std = (sum((r - mean) ** 2 for r in rewards) / max(len(rewards), 1)) ** 0.5
            self._group_stats[group_id] = (mean, std)

        # The env has a single sampling temperature; fan it out per token
        # (context tokens are masked out, so their temperature is don't-care).
        temperature = env.sampling_args["temperature"]
        for r in survivors:
            for sample in r.samples:
                sample.temperatures = [temperature] * len(sample.token_ids)

        if self.pre_filters:
            apply_filters(self.pre_filters, survivors)
        filtered_by_name: dict[str, int] = {}
        num_filtered = 0
        for r in survivors:
            self.pre_filter_seen += 1
            if r.is_filtered:
                self.pre_filter_dropped += 1
                num_filtered += 1
                for name, hit in r.filter_results.items():
                    if hit:
                        self.pre_filter_dropped_by_name[name] = self.pre_filter_dropped_by_name.get(name, 0) + 1
                        filtered_by_name[name] = filtered_by_name.get(name, 0) + 1
                continue
            # Reset annotations so the post-batch filter pass starts clean
            r.filter_results = {}
            r.is_filtered = False
            self.pending_batch.append(r)
            if self.token_batch_size is not None:
                self.pending_tokens += payload_tokens(r)

        # Per-group summary. One line per finalized group; per-filter
        # detection breakdown lives at debug level in ``apply_filters``
        rewards = [r.reward for r in survivors]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        filter_str = ", ".join(f"{n}={c}" for n, c in filtered_by_name.items()) if filtered_by_name else "—"
        get_logger().debug(
            f"Finished group | env={env_name} task_idx={task_idx} | "
            f"rollouts={len(group)} (errored={num_errored}, filtered={num_filtered}) | "
            f"reward={avg_reward:.4f} | filters: {filter_str}"
        )

    def process_batch(self) -> TrainBatch:
        """Pop a cohort off ``pending_batch`` (by rollout count when
        ``batch_size`` is set, by token count when ``token_batch_size`` is
        set), apply post-batch filter annotations, and assemble the
        trainer-bound ``TrainingSample`` list. Overflow stays for the next
        batch."""
        if self.replay is not None:
            return self._process_batch_replay()
        if self.batch_size is not None:
            cohort = self.pending_batch[: self.batch_size]
            self.pending_batch = self.pending_batch[self.batch_size :]
        else:
            assert self.token_batch_size is not None
            cut = 0
            running = 0
            for i, r in enumerate(self.pending_batch):
                running += payload_tokens(r)
                cut = i + 1
                if running >= self.token_batch_size:
                    break
            cohort = self.pending_batch[:cut]
            self.pending_batch = self.pending_batch[cut:]
            self.pending_tokens -= running

        if self.post_filters:
            apply_filters(self.post_filters, cohort)

        # Samples are pre-built by ``process_rollout``; ``process_group`` already stamped the
        # advantage stream and loss routing on each sample. Filtered rollouts don't ship.
        samples: list[TrainingSample] = [sample for r in cohort if not r.is_filtered for sample in r.samples]

        # ``rollouts`` is the observation window — every rollout of every group finalized since the
        # last ship (errored + filtered + survivors) — while ``samples`` is the shipped cohort's
        # trainable payload. ``rollouts.effective`` / ``rollouts.metrics`` derive the clean subset +
        # metric views on demand. Reset the window only when the batch actually ships (non-empty
        # samples) — an empty batch is dropped unlogged by the orchestrator, so keep accumulating its
        # finalized groups (and any overflow) into the next shipped batch's window.
        rollouts = self.pending_rollouts
        if samples:
            self.pending_rollouts = TrainRollouts()
        return TrainBatch(rollouts=rollouts, samples=samples)

    def _process_batch_replay(self) -> TrainBatch:
        """Replay mode: pop pending fresh cohort, admit all to the buffer,
        ship ``fresh_target`` fresh + priority-sampled replay fill.

        efficient_rl additions vs. the source harness: (a) mu-GRPO's
        ``fresh_target`` cycling (see module docstring), (b) re-admitting
        replay-drawn items to the pool after use (bug fix, see module
        docstring) so ``uses_cap`` retires an item only after it's genuinely
        been used that many times."""
        assert self.replay is not None and self.batch_size is not None
        cohort = self.pending_batch[: self.batch_size]
        self.pending_batch = self.pending_batch[self.batch_size :]

        if self.post_filters:
            apply_filters(self.post_filters, cohort)

        fresh_samples: list[TrainingSample] = []
        meta: list[dict] = []
        for r in cohort:
            if r.is_filtered:
                continue
            gmean, gstd = self._group_stats.get(r.group_id, (0.0, 1.0))
            for sample in r.samples:
                if not sample.mask:
                    continue
                # Advantage is a scalar broadcast over trainable (mask=True)
                # tokens and 0.0 elsewhere (see stamp_advantages) — read it
                # from a trainable position, not index 0 (usually prompt/
                # context, always 0.0, which would zero every sample here).
                adv_stream = sample.advantages
                advantage = 0.0
                if adv_stream:
                    for a, m in zip(adv_stream, sample.mask):
                        if m:
                            advantage = float(a)
                            break
                fresh_samples.append(sample)
                meta.append(
                    {
                        "reward": float(r.reward),
                        "advantage": advantage,
                        "group_mean": gmean,
                        "group_std": gstd,
                        "group_id": str(r.group_id),
                        "completion_len": sum(1 for m in sample.mask if m),
                    }
                )

        self.replay.update(self._shipped)
        admitted_items, admitted = self.replay.admit(fresh_samples, meta)
        target = self.batch_size

        # mu-GRPO: override the configured fresh_target with a periodic
        # cycle -- 1 fully-fresh ship every K, the other K-1 pure replay of
        # that same admitted batch. cycle_k<=1 (default) reduces exactly to
        # the source harness's static fresh_target (Experience Replay /
        # Difficulty-Targeted behavior).
        if self._mugrpo_cycle_k > 1:
            effective_fresh_target = target if (self._shipped % self._mugrpo_cycle_k == 0) else 0
        else:
            effective_fresh_target = self.replay.p.fresh_target

        n_fresh_kept = min(effective_fresh_target, len(admitted_items))
        fresh_kept_items = admitted_items[:n_fresh_kept]
        # shipped-fresh items also count as a use (they are backproped now)
        for it in fresh_kept_items:
            it.uses += 1
            it.sampled_count += 1
        self.replay.total_uses_accumulated += n_fresh_kept
        self.replay.total_fresh_shipped += n_fresh_kept
        # exclude items shipped-as-fresh this step from the replay draw pool
        # (else the same physical sample could ship twice in one batch: once
        # via fresh_kept, once via sample()) — they stay in the buffer for
        # later steps, just not double-counted in this composition.
        fresh_kept_ids = {id(it) for it in fresh_kept_items}
        held_out = [it for it in self.replay.items if id(it) in fresh_kept_ids]
        self.replay.items = [it for it in self.replay.items if id(it) not in fresh_kept_ids]
        replay_items = self.replay.sample(max(target - n_fresh_kept, 0))
        self.replay.items.extend(held_out)  # restore fresh-kept items for future steps
        # efficient_rl fix: return replay-drawn items to the pool too (post
        # uses-cap check inside readmit) so they remain eligible for future
        # draws instead of vanishing after their first use.
        self.replay.readmit(replay_items)
        composed = [it.sample for it in fresh_kept_items] + [it.sample for it in replay_items]

        rollouts = self.pending_rollouts
        if composed:
            self.pending_rollouts = TrainRollouts()
        self._shipped += 1
        stats = self.replay.step_stats(fresh_count=len(fresh_samples), n_fresh_kept=n_fresh_kept)
        fresh_rewards = [it.reward for it in fresh_kept_items]
        replay_rewards = [it.reward for it in replay_items]
        stats["fresh_mean_reward"] = sum(fresh_rewards) / max(len(fresh_rewards), 1)
        stats["replay_mean_reward"] = sum(replay_rewards) / max(len(replay_rewards), 1)
        stats["effective_fresh_target"] = effective_fresh_target
        self.replay.log_step(stats, {"shipped": len(composed), "replayed": len(replay_items)})
        return TrainBatch(rollouts=rollouts, samples=composed)

    def reset_pre_filter_stats(self) -> None:
        self.pre_filter_seen = 0
        self.pre_filter_dropped = 0
        self.pre_filter_dropped_by_name.clear()

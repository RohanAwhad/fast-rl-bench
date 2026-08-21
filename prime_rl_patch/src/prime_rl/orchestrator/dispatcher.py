"""RolloutDispatcher: schedules rollouts under a shared permit counter.

- Capacity (``max_inflight_episodes``) is shared across train + eval. One permit is
  one episode: one ``run`` request against an env server.
- Optional rate limiting via ``AsyncLimiter(tasks_per_minute, 60)``.
- Emit-everything invariant: every dispatched episode eventually reaches
  ``out_q`` exactly once, as a ``list[Rollout]``. Failures
  (env error, empty trajectory, task exception, off-policy cancel) carry
  ``trace.last_error`` set; sinks decide drop / partial-train policy.
- ``DispatcherMode.PREFER_TRAIN`` / ``PREFER_EVAL`` controls which kind to
  schedule next. Transitions are level-triggered (driven by the eval
  source's emptiness), so in-flight rollouts of the opposite kind drain
  naturally on either side of an eval boundary.
- ``on_version_pending`` (called by the watcher before the engines pause for
  the weight update) bumps ``off_policy_steps`` on in-flight train rollouts and
  drops groups past ``max_off_policy_steps``.
  Eval rollouts are measurements for the policy version they started with,
  so they are allowed to finish even if training advances. Train rollouts
  sampled from a frozen model never age — their sampler doesn't change
  with policy updates.
  Cancellations surface as synthetic ``Cancelled`` markers so the sink's
  count-to-``group_size`` finalization still fires.

--- efficient_rl (DUET / GRESO) ---
Two additive, env-var-gated hooks live in ``next_fresh_group`` (train only):
  - GRESO (``EFFRL_GRESO=on``): before opening a fresh group, consult the
    shared utility tracker; if this prompt's bucket looks likely to produce a
    zero-variance group, skip it (pull the next example instead) rather than
    paying for its rollouts. Bounded retries so a small/exhausted dataset
    can't spin forever.
  - DUET (``EFFRL_DUET=on``): resolve this prompt's group_size adaptively
    from the same tracker instead of the env's fixed config value.
  - sGPO (``EFFRL_SGPO=on``): same skip/group-size surface, driven by the
    offline profiling file instead of an online tracker — trivial queries
    (p̂ > 0.75) and out-of-phase-cluster queries are skipped before
    generation, unsolved queries (p̂ = 0) are accepted with probability
    ``SGPO_UNSOLVED_MIX``, and per-query group sizes come from the paper's
    1/p̂ buckets; the curriculum phase (easy→hard) advances at
    ``SGPO_PHASE_BOUNDS`` step boundaries (see efficient_rl/sgpo.py).
All are no-ops (identical to vanilla) when their env var is unset, and the
observation side (updating the tracker once a group finishes) only runs when
at least one of them is enabled, so a baseline run pays zero cost.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal

import verifiers.v1 as vf
from aiolimiter import AsyncLimiter

from prime_rl.efficient_rl.allocation import (
    duet_enabled,
    duet_group_size,
    greso_enabled,
    greso_should_skip,
)
from prime_rl.efficient_rl.sgpo import (
    sgpo_enabled,
    sgpo_entry,
    sgpo_group_size,
    sgpo_phase_bounds,
    sgpo_phase_bucket,
    sgpo_should_skip,
)
from prime_rl.efficient_rl.utility_tracker import tracker_singleton
from prime_rl.orchestrator.envs import EvalEnvs, TrainEnvs
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import (
    GroupState,
    InflightRollout,
    Policy,
    Rollout,
    RolloutKind,
)
from prime_rl.utils.async_utils import safe_cancel, safe_cancel_all
from prime_rl.utils.client import InferencePool
from prime_rl.utils.logger import get_logger

# Bound on GRESO's "pull a different example instead" retries per dispatch
# attempt, so a dataset where every bucket currently looks low-utility can't
# spin the event loop forever.
_GRESO_MAX_SKIP_RETRIES = 32

# Bound on sGPO's same pattern: phase-gated dispatch can skip a large share
# of pops (queries outside the current curriculum cluster + trivial ones),
# so the bound is looser than GRESO's. Tripping it logs a warning and
# accepts the popped example anyway (degraded curriculum, training keeps
# moving) instead of stalling dispatch.
_SGPO_MAX_SKIP_RETRIES = 256


class DispatcherMode(Enum):
    """Which kind of work the dispatcher schedules next."""

    PREFER_TRAIN = auto()
    PREFER_EVAL = auto()


@dataclass
class DispatcherMetrics:
    """Per-tick drain counters for the orchestrator's periodic log.
    ``drained()`` returns the current values and clears them; point-in-time
    gauges live on ``RolloutDispatcher.gauges`` instead."""

    cancelled_by_kind_env: dict[tuple[Literal["train", "eval"], str], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    errored_by_kind_env: dict[tuple[Literal["train", "eval"], str], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    sgpo_skipped: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    sgpo_accepted: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record_cancellation(self, *, kind: Literal["train", "eval"], env_name: str, n: int = 1) -> None:
        self.cancelled_by_kind_env[(kind, env_name)] += n

    def record_error(self, *, kind: Literal["train", "eval"], env_name: str) -> None:
        self.errored_by_kind_env[(kind, env_name)] += 1

    def drained(self, *, train_envs: set[str], eval_envs: set[str]) -> dict[str, float]:
        """Return per-tick counters and clear them. Emits the full pre-
        registered key set every tick (zero when no activity) so the wandb
        time axis stays dense and ``define_metric`` lines up."""
        out: dict[str, float] = {}
        for kind in ("train", "eval"):
            envs = train_envs if kind == "train" else eval_envs
            cancelled_total = sum(self.cancelled_by_kind_env.get((kind, e), 0) for e in envs)
            errored_total = sum(self.errored_by_kind_env.get((kind, e), 0) for e in envs)
            out[f"dispatcher/cancelled/{kind}"] = float(cancelled_total)
            out[f"dispatcher/errored/{kind}"] = float(errored_total)
        for env in train_envs | eval_envs:
            out[f"dispatcher/cancelled/{env}"] = float(
                self.cancelled_by_kind_env.get(("train", env), 0) + self.cancelled_by_kind_env.get(("eval", env), 0)
            )
            out[f"dispatcher/errored/{env}"] = float(
                self.errored_by_kind_env.get(("train", env), 0) + self.errored_by_kind_env.get(("eval", env), 0)
            )
        for reason in ("trivial", "phase", "unsolved"):
            out[f"sgpo/skipped/{reason}"] = float(self.sgpo_skipped.get(reason, 0))
        out["sgpo/skipped/total"] = float(sum(self.sgpo_skipped.values()))
        for bucket in ("2", "4", "8", "unsolved"):
            out[f"sgpo/accepted/{bucket}"] = float(self.sgpo_accepted.get(bucket, 0))
        out["sgpo/accepted/total"] = float(sum(self.sgpo_accepted.values()))
        self.cancelled_by_kind_env.clear()
        self.errored_by_kind_env.clear()
        self.sgpo_skipped.clear()
        self.sgpo_accepted.clear()
        return out

    @staticmethod
    def drain_keys(*, train_envs: set[str], eval_envs: set[str]) -> list[str]:
        """Full set of keys ``drained`` may emit; used by the periodic
        logger for ``wandb.define_metric``."""
        keys = [
            "dispatcher/cancelled/train",
            "dispatcher/cancelled/eval",
            "dispatcher/errored/train",
            "dispatcher/errored/eval",
        ]
        for env in train_envs | eval_envs:
            keys.append(f"dispatcher/cancelled/{env}")
            keys.append(f"dispatcher/errored/{env}")
        for reason in ("trivial", "phase", "unsolved"):
            keys.append(f"sgpo/skipped/{reason}")
        keys.append("sgpo/skipped/total")
        for bucket in ("2", "4", "8", "unsolved"):
            keys.append(f"sgpo/accepted/{bucket}")
        keys.append("sgpo/accepted/total")
        return keys


class RolloutDispatcher:
    """``await dispatcher.start()`` runs the dispatch loop until ``stop()``.
    Pulls examples from ``TrainSource`` / ``EvalSource``, schedules
    rollouts under shared capacity, and emits ``Rollout``\\ s to
    ``out_q``. The watcher drives ``on_version_pending`` for off-policy
    cancellation; the orchestrator triggers eval epochs."""

    def __init__(
        self,
        *,
        train_envs: TrainEnvs,
        eval_envs: EvalEnvs | None,
        train_source: TrainSource,
        eval_source: EvalSource | None,
        policy_pool: InferencePool,
        policy: Policy,
        max_inflight_episodes: int,
        tasks_per_minute: float | None,
        max_off_policy_steps: int,
    ) -> None:
        self.policy = policy
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        # Train rollouts go to the env sampler's pool; eval always
        # evaluates the policy.
        self.policy_pool = policy_pool
        self.train_source = train_source
        self.eval_source = eval_source
        self.max_off_policy_steps = max_off_policy_steps

        self.max_inflight = max_inflight_episodes
        self.inflight_permits = 0
        self.rate_limiter: AsyncLimiter | None = (
            AsyncLimiter(tasks_per_minute, time_period=60) if tasks_per_minute else None
        )

        self.inflight: dict[asyncio.Task, InflightRollout] = {}
        self.groups: dict[uuid.UUID, GroupState] = {}

        # Bounded so the dispatcher backpressures on a slow sink. One entry per
        # episode — the sinks count episodes, never loose traces.
        self.out_q: asyncio.Queue[list[Rollout]] = asyncio.Queue(maxsize=max(8, self.max_inflight))

        self.mode: DispatcherMode = DispatcherMode.PREFER_TRAIN
        # Set by the orchestrator after the final train step; pipeline then
        # winds down without scheduling new train rollouts
        self.train_scheduling_disabled: bool = False
        self.metrics = DispatcherMetrics()

        # Orchestrator-owned gate. When clear, ``fill_inflight`` returns
        # without scheduling new groups. The dispatcher itself doesn't know
        # *why* — the orchestrator toggles this based on step / policy lead.
        self.dispatch_allowed = asyncio.Event()
        self.dispatch_allowed.set()

        self.stopped = asyncio.Event()
        self.task: asyncio.Task | None = None

        # --- efficient_rl (DUET / GRESO) ---
        self._effrl_tracking = duet_enabled() or greso_enabled()

        # --- efficient_rl (sGPO) ---
        self._effrl_sgpo = sgpo_enabled()
        self._sgpo_phase = 0

    def _train_pool_for(self, env_name: str) -> tuple[InferencePool, str, bool]:
        """``(pool, model_name, is_live)`` for *train* rollouts of this env —
        the env sampler's pool. (Eval always uses the policy.)"""
        sampler = self.train_envs.get(env_name).sampler
        if sampler.samples_from_live_policy:
            return sampler.pool, self.policy.model_name, True
        return sampler.pool, sampler.pool.model_name, False

    @property
    def inflight_train_count(self) -> int:
        return sum(1 for m in self.inflight.values() if m.kind == "train")

    @property
    def inflight_eval_count(self) -> int:
        return sum(1 for m in self.inflight.values() if m.kind == "eval")

    @property
    def available_permits(self) -> int:
        return self.max_inflight - self.inflight_permits

    @property
    def inflight_by_env(self) -> dict[tuple[RolloutKind, str], int]:
        counts: dict[tuple[RolloutKind, str], int] = defaultdict(int)
        for meta in self.inflight.values():
            counts[(meta.kind, meta.env_name)] += 1
        return dict(counts)

    @property
    def queued_eval_examples(self) -> int:
        return len(self.eval_source) if self.eval_source is not None else 0

    @property
    def eval_has_work(self) -> bool:
        """Eval has work while its source queue is non-empty OR any opened eval group still has
        rollouts to schedule. An example leaves ``eval_source`` when its group opens
        (``next_fresh_group``), but its ``group_size`` rollouts dispatch one at a time across
        ``fill_inflight`` passes — so the queue can be empty while a group is still mid-schedule."""
        return bool(self.eval_source) or any(
            g.kind == "eval" and g.rollouts_to_schedule > 0 for g in self.groups.values()
        )

    @property
    def is_idle(self) -> bool:
        """True once nothing is in flight, no eval work remains (queued *or* a partly-scheduled eval
        group), and ``out_q`` is empty — the pipeline has fully drained."""
        return not self.inflight and not self.eval_has_work and self.out_q.empty()

    def disable_train_scheduling(self) -> None:
        """Stop scheduling new train rollouts; in-flight train + any
        triggered eval drain naturally."""
        self.train_scheduling_disabled = True

    @property
    def max_off_policy_level(self) -> int:
        steps = [m.off_policy_steps for m in self.inflight.values() if m.kind == "train"]
        return max(steps) if steps else 0

    @property
    def mean_off_policy_level(self) -> float:
        steps = [m.off_policy_steps for m in self.inflight.values() if m.kind == "train"]
        return sum(steps) / len(steps) if steps else 0.0

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Single dispatch loop: schedule, wait, collect, repeat."""
        self.task = asyncio.current_task()
        try:
            while not self.stopped.is_set():
                await self.fill_inflight()
                if not self.inflight:
                    # No work — sleep briefly. Eval triggers from the
                    # orchestrator wake the next iteration via a mode flip
                    try:
                        await asyncio.wait_for(self.stopped.wait(), timeout=0.1)
                    except asyncio.TimeoutError:
                        pass
                    continue

                done, _pending = await asyncio.wait(
                    list(self.inflight.keys()),
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=0.5,  # wake periodically to re-check fill (mode flips)
                )
                for task in done:
                    await self.handle_completed_rollout(task)
        except asyncio.CancelledError:
            return

    async def stop(self) -> None:
        self.stopped.set()
        await self.cancel_inflight_rollouts()
        if self.task is not None:
            await safe_cancel(self.task)
            self.task = None

    async def on_version_pending(self, step: int) -> None:
        """Bump off-policy counters and drop groups past
        ``max_off_policy_steps`` (drop_group emits ``Cancelled`` markers so
        the sink still finalizes the partial group). Eval rollouts are not
        aged because they are tied to their start-time policy version.

        Runs *before* the inference engines are paused for the weight update so
        the resulting aborts are processed while the engine is still stepping —
        otherwise the orphaned KV transfers crash the decode engine on resume
        (see ``WeightWatcher.apply_policy_update``)."""
        self._advance_sgpo_phase(step)
        stale_groups: set[uuid.UUID] = set()
        cancelled = 0
        for meta in self.inflight.values():
            if meta.kind != "train":
                continue
            # Frozen-sourced rollouts never go stale — their sampler doesn't
            # change with policy updates.
            if not self.train_envs.get(meta.env_name).sampler.samples_from_live_policy:
                continue
            meta.off_policy_steps += 1
            if meta.off_policy_steps > self.max_off_policy_steps:
                stale_groups.add(meta.group_id)

        for gid in stale_groups:
            removed = await self.drop_group(gid)
            cancelled += removed

        if cancelled:
            get_logger().warning(
                f"Cancelled {cancelled} train rollouts past max_off_policy_steps={self.max_off_policy_steps}. "
                "Consider increasing it to avoid this."
            )

    async def on_new_version(self, step: int) -> None:
        """No-op: the dispatcher drains in ``on_version_pending`` (pre-pause)."""

    def _advance_sgpo_phase(self, step: int) -> None:
        """Step-scheduled easy-to-hard curriculum (sGPO §4.3.3): phases
        advance when the training step crosses ``SGPO_PHASE_BOUNDS``. The
        dispatcher gates *dispatch* on the phase; in-flight groups finish at
        their start-time group size regardless."""
        if not self._effrl_sgpo:
            return
        phase = sum(1 for b in sgpo_phase_bounds() if step > b)
        if phase != self._sgpo_phase:
            get_logger().info(
                f"sGPO curriculum: phase {self._sgpo_phase} -> {phase} at step {step} "
                f"(bucket G={sgpo_phase_bucket(phase)})"
            )
            self._sgpo_phase = phase

    async def fill_inflight(self) -> None:
        """Schedule new rollouts up to ``max_inflight``, honoring
        ``self.mode``. Eval scheduling ignores the orchestrator's dispatch
        gate (evals are version-pinned measurements); only train scheduling
        respects it. When ``PREFER_EVAL``'s source exhausts we flip back to
        ``PREFER_TRAIN`` so the eval tail drains alongside fresh train."""
        while True:
            if self.available_permits <= 0:
                return

            if self.mode == DispatcherMode.PREFER_EVAL:
                # PREFER_EVAL is only entered when the orchestrator triggers
                # eval, which requires ``eval_source`` to be configured
                assert self.eval_source is not None
                if not self.eval_has_work:
                    # Eval source + all eval groups fully dispatched. Flip
                    # to PREFER_TRAIN so any remaining permits go to train
                    # while the in-flight eval tail completes naturally
                    self.switch_mode(DispatcherMode.PREFER_TRAIN, reason="the eval queue drained")
                    continue
                scheduled = await self.try_schedule("eval")
                if not scheduled:
                    return
            else:  # PREFER_TRAIN — respects the orchestrator's dispatch gate
                if not self.dispatch_allowed.is_set():
                    return
                scheduled = await self.try_schedule("train")
                if not scheduled:
                    return

    def switch_mode(self, new_mode: DispatcherMode, *, reason: str) -> None:
        if new_mode == self.mode:
            return
        prefer = "eval" if new_mode == DispatcherMode.PREFER_EVAL else "train"
        get_logger().info(f"Switching dispatcher mode to prefer {prefer} rollouts because {reason}")
        self.mode = new_mode

    async def try_schedule(self, kind: RolloutKind) -> bool:
        """Schedule one rollout of ``kind``: prefer continuing an existing
        group (keeps prefix-cache hits); otherwise open a fresh group from
        the corresponding source. Returns False if nothing could be
        scheduled."""
        if kind == "train" and self.train_scheduling_disabled:
            return False
        envs = self.train_envs if kind == "train" else self.eval_envs
        if envs is None:
            return False

        for gid, group in list(self.groups.items()):
            if group.kind != kind or group.rollouts_to_schedule <= 0:
                continue
            return await self.schedule_group_rollout(gid, group)

        fresh = self.next_fresh_group(kind, envs)
        if fresh is None:
            return False
        gid = uuid.uuid4()
        self.groups[gid] = fresh
        return await self.schedule_group_rollout(gid, fresh)

    def next_fresh_group(self, kind: RolloutKind, envs) -> GroupState | None:
        """Pop the next example from the corresponding source and wrap it in
        a ``GroupState``. Returns ``None`` if the source is empty.

        efficient_rl: for train-kind groups only, GRESO may retry with a
        different example (bounded) instead of the one just popped, and DUET
        may resolve a non-default ``group_size`` for the example it settles
        on. Both are no-ops unless their env var is enabled."""
        if kind == "train":
            source = self.train_source
        else:
            assert self.eval_source is not None
            source = self.eval_source

        example = source.next_example()
        if kind == "train" and example is not None and greso_enabled():
            retries = 0
            while retries < _GRESO_MAX_SKIP_RETRIES:
                prompt = _example_prompt(example)
                if not greso_should_skip(prompt):
                    break
                example = source.next_example()
                retries += 1
                if example is None:
                    break
        if kind == "train" and example is not None and sgpo_enabled():
            retries = 0
            while retries < _SGPO_MAX_SKIP_RETRIES:
                idx = _example_idx(example)
                skip, reason = sgpo_should_skip(idx, self._sgpo_phase)
                if not skip:
                    break
                self.metrics.sgpo_skipped[reason] += 1
                example = source.next_example()
                retries += 1
                if example is None:
                    break
            if retries >= _SGPO_MAX_SKIP_RETRIES and example is not None:
                get_logger().warning(
                    "sGPO: skip retry bound reached; accepting an example outside the current phase "
                    f"cluster (phase={self._sgpo_phase}, G={sgpo_phase_bucket(self._sgpo_phase)})"
                )
        if example is None:
            return None

        env_name = example["env_name"]
        base_group_size = envs.get(env_name).config.group_size
        if kind == "train" and duet_enabled():
            group_size = duet_group_size(_example_prompt(example), base_group_size)
        elif kind == "train" and sgpo_enabled():
            group_size = sgpo_group_size(_example_idx(example), self._sgpo_phase, base_group_size)
        else:
            group_size = base_group_size
        if kind == "train" and self._effrl_sgpo:
            entry = sgpo_entry(_example_idx(example))
            if entry is not None:
                key = "unsolved" if entry["decision"] == "unsolved" else f"G{entry['bucket']}"
                self.metrics.sgpo_accepted[key] += 1
        eval_step: int | None = example.get("eval_step") if kind == "eval" else None

        return GroupState(
            kind=kind,
            env_name=env_name,
            task=example["task"],
            rollouts_to_schedule=group_size,
            target_rollouts=group_size,
            eval_step=eval_step,
            policy_version_at_start=self.policy.version,
        )

    async def schedule_group_rollout(self, group_id: uuid.UUID, group: GroupState) -> bool:
        """Dispatch one ``run`` task for this group.

        Returns False only if we couldn't even schedule one rollout (no clients
        ready, no permits). Returns True after issuing one task — the caller
        loops to keep scheduling.
        """
        # Train rollouts use the env sampler's pool via the
        # renderer/token train client. Eval always evaluates the policy and
        # goes through the eval client (chat-completions) so eval scores stay
        # comparable.
        if group.kind == "eval":
            pool, model_name = self.policy_pool, self.policy.model_name
            live_sourced = True
        else:
            pool, model_name, live_sourced = self._train_pool_for(group.env_name)

        if group.pinned_client is None:
            group.pinned_client = pool.eval_client if group.kind == "eval" else pool.train_client
        client = group.pinned_client

        env_collection = self.train_envs if group.kind == "train" else self.eval_envs
        if env_collection is None:
            return False
        env = env_collection.get(group.env_name)
        # Frozen-sourced train rollouts hit a frozen pool; salting per policy
        # version would invalidate its prefix cache every weight update for
        # no reason.
        if live_sourced:
            cache_salt = str(group.policy_version_at_start)
        else:
            cache_salt = None

        group.rollouts_to_schedule -= 1
        await self.acquire()
        task = asyncio.create_task(
            env.run(
                client=client,
                model_name=model_name,
                cache_salt=cache_salt,
                task_data=group.task.data.model_dump(mode="json"),
            )
        )

        self.inflight[task] = InflightRollout(
            kind=group.kind,
            env_name=group.env_name,
            group_id=group_id,
            policy_version=group.policy_version_at_start,
            client_config=client,
            eval_step=group.eval_step,
        )
        return True

    async def acquire(self) -> None:
        """Reserve one permit + rate-limit it. Caller must precheck
        ``available_permits >= 1``; this is not a blocking acquire."""
        if self.rate_limiter is not None:
            await self.rate_limiter.acquire()
        self.inflight_permits += 1

    def release(self) -> None:
        self.inflight_permits -= 1

    async def handle_completed_rollout(self, task: asyncio.Task) -> None:
        """Emit every dispatched episode exactly once to ``out_q``. Task
        exceptions synthesize an error-marker episode so the sink's
        count-to-``group_size`` finalization still triggers. Cancelled tasks
        (popped by ``drop_group``) raise ``CancelledError`` and are discarded —
        ``drop_group`` already emitted their markers.
        """
        meta = self.inflight.pop(task, None)
        if meta is None:
            return  # already handled by drop_group / cancel_inflight_rollouts
        self.release()
        group = self.groups.get(meta.group_id)

        is_synth_exception = False
        try:
            result = task.result()
            rollouts: list[Rollout] = result if isinstance(result, list) else [result]
            if not rollouts:
                raise RuntimeError("env run returned an empty episode (no traces)")
        except asyncio.CancelledError:
            return
        except Exception as exc:
            get_logger().warning(f"Rollout task failed in group {meta.group_id} ({meta.env_name}): {exc!r}")
            task_idx = group.task.data.idx if group is not None else -1
            rollouts = [
                Rollout(
                    task=vf.TraceTask(type="Task", data=vf.TaskData(idx=task_idx, prompt=None)),
                    agent=vf.AgentInfo(config=vf.AgentConfig()),
                )
            ]
            for r in rollouts:
                r.record_error(exc)
            is_synth_exception = True

        for r in rollouts:
            if not r.has_error and r.num_turns == 0:
                # Empty trajectory: promote to an explicit error so the sink
                # treats it like any other failure (``has_error`` reads ``ok``)
                r.errors.append(vf.Error(type="EmptyTrajectory", message="Rollout returned with no trajectory steps"))
                r.ok = False
                get_logger().warning(f"Empty trajectory in group {meta.group_id} ({meta.env_name})")
            if r.has_error:
                self.metrics.record_error(kind=meta.kind, env_name=meta.env_name)
                if not is_synth_exception and r.last_error is not None:
                    get_logger().warning(
                        f"Rollout failed in group {meta.group_id} ({meta.env_name}) — {r.last_error.type}: {r.last_error.message}"
                    )
        await self.emit_episode(meta, group, rollouts)

    async def emit_episode(self, meta: InflightRollout, group: GroupState | None, rollouts: list[Rollout]) -> None:
        """Stamp prime-rl metadata onto one completed episode and put it on
        ``out_q``. Pops the group from ``self.groups`` once every owed episode
        has been emitted."""
        eval_step = meta.eval_step
        policy_version = meta.policy_version
        if group is not None:
            eval_step = group.eval_step
            policy_version = group.policy_version_at_start
            group.emitted += 1
            # efficient_rl (DUET / GRESO): accumulate this group's rewards as
            # they arrive, and update the shared utility tracker with the
            # full group's reward spread once every owed rollout is in.
            if meta.kind == "train" and self._effrl_tracking:
                group.rewards_seen.extend(r.reward for r in rollouts if not r.has_error)
                if group.emitted >= group.target_rollouts and group.rewards_seen:
                    tracker_singleton().observe(group.task.data.prompt or "", group.rewards_seen)
            if group.emitted >= group.target_rollouts:
                self.groups.pop(meta.group_id, None)

        for rollout in rollouts:
            rollout.kind = meta.kind
            rollout.env_name = meta.env_name
            rollout.group_id = meta.group_id
            rollout.policy_version = policy_version
            rollout.off_policy_steps = meta.off_policy_steps
            # efficient_rl (DUET): see types.py::Rollout.group_target_size docstring.
            rollout.group_target_size = group.target_rollouts if group is not None else None
            if meta.kind == "eval":
                assert eval_step is not None, "eval rollout missing eval_step"
                rollout.eval_step = eval_step
        await self.out_q.put(rollouts)

    async def drop_group(self, group_id: uuid.UUID) -> int:
        """Cancel remaining in-flight tasks for this group and emit a
        ``Cancelled`` marker for every rollout it still owes the sink
        (both in-flight and not-yet-scheduled). Returns the count for
        off-policy metrics."""
        group = self.groups.pop(group_id, None)
        task_idx = group.task.data.idx if group is not None else -1

        # Sync claim phase: pop matching tasks from ``self.inflight`` and
        # release their permits in one non-yielding sweep. After this loop
        # the dropped tasks are no longer reachable from ``self.inflight``,
        # so ``handle_completed_rollout``'s existing None-guard makes the
        # subsequent async emit phase race-free.
        claimed: list[tuple[asyncio.Task, InflightRollout]] = []
        for task, meta in list(self.inflight.items()):
            if meta.group_id != group_id:
                continue
            del self.inflight[task]
            self.release()
            claimed.append((task, meta))

        tasks_to_cancel = [task for task, _ in claimed]
        inflight_cancelled = len(claimed)
        last_meta: InflightRollout | None = claimed[-1][1] if claimed else None
        for _, meta in claimed:
            trace = Rollout(
                task=vf.TraceTask(type="Task", data=vf.TaskData(idx=task_idx, prompt=None)),
                agent=vf.AgentInfo(config=vf.AgentConfig()),
                ok=False,
                errors=[vf.Error(type="Cancelled", message="Off-policy cancel")],
                stop_condition="error",
            )
            await self.emit_episode(meta, group, [trace])

        # The group may have rollouts that were never dispatched
        # (``rollouts_to_schedule > 0``). Emit markers for those too so the
        # sink hits ``target_rollouts``
        #
        # ``last_meta`` can be ``None`` if the only inflight task for this
        # group completed naturally between ``on_version_pending``'s snapshot
        # and us reaching it — synthesize a stand-in from the group state
        unscheduled_cancelled = 0
        if group is not None and group.rollouts_to_schedule > 0:
            fallback_meta = last_meta or InflightRollout(
                kind=group.kind,
                env_name=group.env_name,
                group_id=group_id,
                policy_version=group.policy_version_at_start,
                eval_step=group.eval_step,
            )
            unscheduled_cancelled = group.rollouts_to_schedule
            for _ in range(unscheduled_cancelled):
                trace = Rollout(
                    task=vf.TraceTask(type="Task", data=vf.TaskData(idx=task_idx, prompt=None)),
                    agent=vf.AgentInfo(config=vf.AgentConfig()),
                    ok=False,
                    errors=[vf.Error(type="Cancelled", message="Off-policy cancel")],
                    stop_condition="error",
                )
                await self.emit_episode(fallback_meta, group, [trace])

        cancelled = inflight_cancelled + unscheduled_cancelled
        if cancelled > 0:
            meta_for_log = last_meta or (
                InflightRollout(
                    kind=group.kind,
                    env_name=group.env_name,
                    group_id=group_id,
                    policy_version=group.policy_version_at_start if group else 0,
                    eval_step=group.eval_step,
                )
                if group is not None
                else None
            )
            if meta_for_log is not None:
                self.metrics.record_cancellation(kind=meta_for_log.kind, env_name=meta_for_log.env_name, n=cancelled)
                get_logger().debug(
                    f"drain {meta_for_log.kind} | group={str(group_id)[:8]} env={meta_for_log.env_name} | "
                    f"cancelled={cancelled} (inflight={inflight_cancelled} unscheduled={unscheduled_cancelled})"
                )

        if tasks_to_cancel:
            await safe_cancel_all(tasks_to_cancel)
        return cancelled

    async def cancel_inflight_rollouts(self) -> None:
        """Cancel all in-flight rollouts. Used on shutdown — doesn't emit
        markers since the sinks are being torn down anyway."""
        for meta in self.inflight.values():
            self.metrics.record_cancellation(kind=meta.kind, env_name=meta.env_name)
            self.release()
        tasks = list(self.inflight.keys())
        self.inflight.clear()
        self.groups.clear()
        if tasks:
            await safe_cancel_all(tasks)

    async def cancel_inflight_train_rollouts(self) -> int:
        """Cancel in-flight train rollouts, leaving eval alone. Used by the
        orchestrator at ``max_steps`` so triggered eval can still complete
        through the pipeline while wasted train inference is short-circuited."""
        train_tasks: list[asyncio.Task] = []
        train_group_ids: set[uuid.UUID] = set()
        cancelled = 0
        for task, meta in list(self.inflight.items()):
            if meta.kind != "train":
                continue
            self.inflight.pop(task, None)
            self.release()
            self.metrics.record_cancellation(kind="train", env_name=meta.env_name)
            cancelled += 1
            train_tasks.append(task)
            train_group_ids.add(meta.group_id)
        for gid in train_group_ids:
            self.groups.pop(gid, None)
        if train_tasks:
            await safe_cancel_all(train_tasks)
        return cancelled

    # ── metrics ────────────────────────────────────────────────────────────

    def gauges(self) -> dict[str, float]:
        """Instantaneous, read-only gauges sampled by the periodic logger."""
        return {
            "dispatcher/inflight_train": float(self.inflight_train_count),
            "dispatcher/inflight_eval": float(self.inflight_eval_count),
            "dispatcher/queued/eval": float(self.queued_eval_examples),
            "dispatcher/mode": float(self.mode == DispatcherMode.PREFER_EVAL),
            "dispatcher/groups_in_flight": float(len(self.groups)),
            "dispatcher/off_policy_level_max": float(self.max_off_policy_level),
            "dispatcher/off_policy_level_mean": self.mean_off_policy_level,
        }


def _example_prompt(example: dict) -> str:
    """Best-effort prompt text for an example dict (``{"task": vf.Task,
    "env_name": str}``), used only by the efficient_rl bucket key — never
    raises (falls back to an empty string, which just means "unknown
    bucket", not a crash) so a task-data shape we didn't anticipate degrades
    to baseline-equivalent (no skip, no resize) instead of breaking dispatch.
    """
    try:
        return example["task"].data.prompt or ""
    except Exception:
        return ""


def _example_idx(example: dict) -> int:
    """Best-effort dataset index for an example dict, used only by sGPO's
    per-query profile lookup — never raises (falls back to -1, which just
    means "no profile entry", not a crash) so a task-data shape we didn't
    anticipate degrades to baseline-equivalent instead of breaking dispatch.
    """
    try:
        return int(example["task"].data.idx)
    except Exception:
        return -1

"""Shared per-prompt/bucket utility (reward-variance) tracker for DUET and GRESO.

Both DUET (adaptive rollout allocation) and GRESO (skip low-value prompts
before generation) need the same primitive: an online estimate of "how
informative would generating more rollouts for this prompt be" -- approximated
as the historical variance of rewards observed in past groups for that prompt
(near-zero variance ~= every rollout in the group tied, i.e. GRPO's advantage
would collapse to ~0 -- the degenerate case both papers target).

Two-level key: an exact prompt hash (most faithful -- matches both papers'
per-prompt framing) falling back to a coarser bucket (a cheap, env-agnostic
length-decile of the prompt text) once the exact prompt hasn't been seen often
enough yet. This matters because our two tasks have very different dataset
sizes relative to a ~minutes-long, few-hundred-rollout run: reverse-text's pool
is small enough to cycle within a run (exact-prompt stats accumulate real
signal), while SciKnowEval's ~18k-row pool rarely repeats an exact prompt in
that time (the coarse bucket is what actually carries signal there). Falling
back keeps the mechanism meaningfully active on both rather than silently
degenerating to a no-op on the larger dataset.
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
from dataclasses import dataclass


def _hash_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def coarse_bucket(prompt: str, n_buckets: int = 12) -> str:
    """Log-scale length-decile bucket: a cheap, env-agnostic difficulty proxy
    computable from just the prompt string, with zero extra model calls. Works
    identically for reverse-text's raw paragraph and SciKnowEval's rendered
    question+choices block."""
    length = max(len(prompt), 1)
    bucket = min(n_buckets - 1, int(math.log2(length)))
    return f"len_{bucket}"


@dataclass
class _Stat:
    n: int = 0
    mean_reward: float = 0.0
    ema_var: float = 0.0  # EMA of within-group reward variance


class PromptUtilityTracker:
    """Process-local, in-memory, one instance per orchestrator process
    (dispatcher owns it). asyncio is single-threaded so the lock is cheap
    insurance, not a real contention concern."""

    def __init__(self, ema_decay: float = 0.3, min_exact_count: int = 2) -> None:
        self.ema_decay = ema_decay
        self.min_exact_count = min_exact_count
        self._exact: dict[str, _Stat] = {}
        self._coarse: dict[str, _Stat] = {}
        self._lock = threading.Lock()

    def _update_one(self, table: dict[str, _Stat], key: str, mean: float, var: float) -> None:
        s = table.setdefault(key, _Stat())
        s.n += 1
        d = self.ema_decay
        if s.n == 1:
            s.mean_reward, s.ema_var = mean, var
        else:
            s.mean_reward = (1 - d) * s.mean_reward + d * mean
            s.ema_var = (1 - d) * s.ema_var + d * var

    def observe(self, prompt: str, rewards: list[float]) -> None:
        """Call once a group finishes, with the rewards of that group."""
        if not rewards:
            return
        mean = sum(rewards) / len(rewards)
        var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        key = _hash_key(prompt)
        bucket = coarse_bucket(prompt)
        with self._lock:
            self._update_one(self._exact, key, mean, var)
            self._update_one(self._coarse, bucket, mean, var)

    def utility(self, prompt: str) -> tuple[float, int]:
        """Returns (estimated within-group reward variance, observation count)
        preferring exact-prompt history once it has >= min_exact_count
        samples, else falling back to the coarse length bucket.
        observation_count == 0 means "no history anywhere" (always the case
        the very first time any prompt in that bucket is seen)."""
        key = _hash_key(prompt)
        bucket = coarse_bucket(prompt)
        with self._lock:
            exact = self._exact.get(key)
            if exact is not None and exact.n >= self.min_exact_count:
                return exact.ema_var, exact.n
            coarse = self._coarse.get(bucket)
            if coarse is not None:
                return coarse.ema_var, coarse.n
        return 0.0, 0


_SINGLETON: PromptUtilityTracker | None = None


def tracker_singleton() -> PromptUtilityTracker:
    """One tracker per orchestrator process, lazily constructed from env vars
    so DUET and GRESO share the exact same accumulated state (relevant if a
    run ever enabled both; our conditions only enable one at a time, but the
    primitive is genuinely shared, not accidentally duplicated)."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = PromptUtilityTracker(
            ema_decay=float(os.environ.get("EFFRL_UTILITY_EMA_DECAY", "0.3")),
            min_exact_count=int(os.environ.get("EFFRL_UTILITY_MIN_EXACT_COUNT", "2")),
        )
    return _SINGLETON

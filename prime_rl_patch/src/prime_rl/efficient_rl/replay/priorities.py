"""Priority function registry for the replay buffer.

Adapted (trimmed to the formulas this project actually uses) from a prior,
tested replay-buffer harness for prime-rl by the same author
(``RohanAwhad/replay_buffer_experiment_rl``, commit-pinned to the same
prime-rl base this project targets). That harness screened 12 priority
formulas and found the *admission filter + reuse cap* (not the specific
priority formula) drives most of the effect -- so we default to the simplest
one (``staleness``: newest-admitted-first) for every replay-based condition
here, isolating each paper's own distinguishing mechanism (selection filter,
fresh/replay cycling) instead of conflating it with priority-formula choice.

Each function: (item, ctx) -> float >= 0. Higher = more likely to be sampled.
All inputs are forward-pass-free metadata already available at admission time
(reward, advantage, age, uses) -- no extra model calls to score a replay
candidate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

registry: dict[str, object] = {}


def register(name: str):
    def deco(fn):
        registry[name] = fn
        return fn

    return deco


@dataclass
class ReplayItem:
    sample: object  # TrainingSample
    reward: float
    advantage: float  # as computed by the algo (here: r - group_mean)
    group_mean: float
    group_std: float
    group_id: str
    completion_len: int
    admitted_step: int
    uses: int = 0
    sampled_count: int = 0
    priority: float = 0.0
    age: int = 0  # step - admitted_step, refreshed each step


@dataclass
class PriorityCtx:
    step: int
    params: dict[str, float] = field(default_factory=dict)


def _tau(ctx: PriorityCtx) -> float:
    return max(float(ctx.params.get("tau", 3.0)), 1e-6)


@register("staleness")
def p_staleness(item: ReplayItem, ctx: PriorityCtx) -> float:
    """Newest admitted data first -- the generic "Experience Replay" idea:
    don't discard a rollout after one gradient step, keep recent ones around."""
    return 1.0 / (1.0 + item.age)


@register("signal")
def p_signal(item: ReplayItem, ctx: PriorityCtx) -> float:
    """Learning-signal only: |A| = |r - group_mean|."""
    return abs(item.advantage)


@register("combined")
def p_combined(item: ReplayItem, ctx: PriorityCtx) -> float:
    """|A| * exp(-age/tau) / sqrt(1+uses): signal + staleness + reuse penalty
    (the reference harness's best-performing formula, offered here as an
    alternative to ``staleness`` for follow-up experiments; not the default)."""
    return abs(item.advantage) * math.exp(-item.age / _tau(ctx)) / math.sqrt(1.0 + item.uses)


def make_priority(name: str, params: dict[str, float] | None = None) -> object:
    if name not in registry:
        raise ValueError(f"Unknown priority '{name}'; available: {sorted(registry)}")
    fn = registry[name]
    params = dict(params or {})

    def wrapped(item: ReplayItem, ctx: PriorityCtx) -> float:
        ctx.params = params
        return max(fn(item, ctx), 0.0)

    return wrapped

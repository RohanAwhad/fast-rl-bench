"""ReplayBuffer: state container + behavior, adapted from
``RohanAwhad/replay_buffer_experiment_rl`` (same author, same pinned prime-rl
base). Admission filter (|A| >= eps), priority computation per step
(forward-pass-free metadata only), probabilistic sampling, uses-cap + capacity
eviction, metrics.

Fix vs. the source harness (documented per "trust code over docs" -- the
harness's own INTEGRATION.md already promised this behavior, the code didn't
implement it): ``sample()`` pops drawn items out of ``self.items`` and the
caller (``train_sink.py``) never reinserted them, so ``uses_cap`` was
structurally unreachable for pure-replay draws -- an item removed from the
pool the first time it's replayed can never accumulate a 2nd/3rd use. Fixed by
re-extending sampled-and-shipped items back into the pool in the caller (see
``train_sink.py``); ``uses_cap`` eviction (below, in ``update()``) then does
the real work of retiring an item once it's actually been used the configured
number of times. This matters most for mu-GRPO here, which depends on one
admitted batch surviving exactly K uses.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from prime_rl.efficient_rl.replay.priorities import PriorityCtx, ReplayItem, make_priority


@dataclass
class ReplayParams:
    enabled: bool = False
    buffer_size: int = 2048
    fresh_target: int = 32  # per-step fresh portion of the composed batch
    uses_cap: int = 3
    eps: float = 0.05  # |A| admission threshold (0 disables the filter)
    priority: str = "staleness"
    priority_params: dict[str, float] = field(default_factory=dict)
    sampling: str = "proportional"  # proportional | topk
    temp: float = 1.0
    group_cap: int = 16  # max samples per group per draw
    metrics_path: str = ""
    run_name: str = ""


class ReplayBuffer:
    def __init__(self, params: ReplayParams, rng: random.Random):
        self.p = params
        self.rng = rng
        self.priority_fn = make_priority(params.priority, params.priority_params)
        self.items: list[ReplayItem] = []
        self.step = 0
        self.total_admitted = 0
        self.total_rejected_eps = 0
        self.total_evicted_uses = 0
        self.total_evicted_capacity = 0
        self.total_sampled = 0
        self.total_uses_accumulated = 0
        self.total_fresh_shipped = 0

    def available(self) -> int:
        return len(self.items)

    def admit(self, samples: list, meta: list[dict]) -> tuple[list[ReplayItem], dict]:
        """Admit samples (with parallel meta); returns (admitted_items, counts)."""
        admitted_items: list[ReplayItem] = []
        rejected = 0
        for sample, m in zip(samples, meta):
            if self.p.eps > 0 and abs(m["advantage"]) < self.p.eps:
                rejected += 1
                continue
            item = ReplayItem(
                sample=sample,
                reward=m["reward"],
                advantage=m["advantage"],
                group_mean=m["group_mean"],
                group_std=m["group_std"],
                group_id=m["group_id"],
                completion_len=m["completion_len"],
                admitted_step=self.step,
            )
            self.items.append(item)
            admitted_items.append(item)
        self.total_admitted += len(admitted_items)
        self.total_rejected_eps += rejected
        if len(self.items) > self.p.buffer_size:
            self._evict_capacity()
        return admitted_items, {"admitted": len(admitted_items), "rejected_eps": rejected}

    def _evict_capacity(self) -> None:
        over = len(self.items) - self.p.buffer_size
        if over <= 0:
            return
        self._update_priorities()
        self.items.sort(key=lambda it: it.priority)
        del self.items[:over]
        self.total_evicted_capacity += over

    def update(self, step: int) -> None:
        self.step = step
        for it in self.items:
            it.age = step - it.admitted_step
        self._update_priorities()
        before = len(self.items)
        self.items = [it for it in self.items if it.uses < self.p.uses_cap]
        self.total_evicted_uses += before - len(self.items)

    def _update_priorities(self) -> None:
        ctx = PriorityCtx(step=self.step, params=dict(self.p.priority_params))
        for it in self.items:
            it.priority = self.priority_fn(it, ctx)

    def sample(self, n: int) -> list[ReplayItem]:
        """Draw n items without replacement from the CURRENT pool. Caller is
        responsible for re-inserting them (via ``readmit``) if they should
        remain eligible for future draws (see module docstring)."""
        if n <= 0 or not self.items:
            return []
        n = min(n, len(self.items))
        if self.p.sampling == "topk":
            self.items.sort(key=lambda it: it.priority, reverse=True)
            picked = self.items[:n]
            rest = self.items[n:]
        else:
            picked, rest = self._proportional_draw(n)
        self.items = rest
        for it in picked:
            it.uses += 1
            it.sampled_count += 1
        self.total_sampled += len(picked)
        self.total_uses_accumulated += len(picked)
        return picked

    def readmit(self, items: list[ReplayItem]) -> None:
        """Return previously-sampled items to the pool (post uses-cap check --
        an item that just hit the cap is dropped here, not re-added)."""
        for it in items:
            if it.uses < self.p.uses_cap:
                self.items.append(it)
            else:
                self.total_evicted_uses += 1

    def _proportional_draw(self, n: int) -> tuple[list[ReplayItem], list[ReplayItem]]:
        """Sample without replacement with weights = priority^(1/temp), capped
        per group to keep batches from fragmenting one group."""
        if self.p.temp <= 0:
            self.items.sort(key=lambda it: it.priority, reverse=True)
            return self.items[:n], self.items[n:]
        items = list(self.items)
        exp = 1.0 / self.p.temp
        picked: list[ReplayItem] = []
        group_taken: dict[str, int] = {}
        for _ in range(n):
            if not items:
                break
            weights = [max(it.priority, 1e-9) ** exp for it in items]
            total = sum(weights)
            r = self.rng.random() * total
            acc = 0.0
            idx = len(items) - 1
            for i, w in enumerate(weights):
                acc += w
                if r <= acc:
                    idx = i
                    break
            it = items.pop(idx)
            gc = group_taken.get(it.group_id, 0)
            if gc >= self.p.group_cap:
                continue
            group_taken[it.group_id] = gc + 1
            picked.append(it)
        return picked, items

    def step_stats(self, fresh_count: int, n_fresh_kept: int) -> dict:
        ages = [it.age for it in self.items]
        abs_as = [abs(it.advantage) for it in self.items]
        return {
            "step": self.step,
            "buffer_size": len(self.items),
            "buffer_cap": self.p.buffer_size,
            "fresh_in": fresh_count,
            "fresh_shipped": n_fresh_kept,
            "buffer_mean_age": sum(ages) / max(len(ages), 1),
            "buffer_max_age": max(ages) if ages else 0,
            "buffer_mean_abs_a": sum(abs_as) / max(len(abs_as), 1),
            "total_admitted": self.total_admitted,
            "total_rejected_eps": self.total_rejected_eps,
            "total_evicted_uses": self.total_evicted_uses,
            "total_evicted_capacity": self.total_evicted_capacity,
            "total_sampled": self.total_sampled,
            "total_uses_accumulated": self.total_uses_accumulated,
            "total_fresh_shipped": self.total_fresh_shipped,
            "group_count": len(set(it.group_id for it in self.items)),
        }

    def log_step(self, stats: dict, replay_stats: dict) -> None:
        if not self.p.metrics_path:
            return
        line = {
            "run": self.p.run_name,
            "priority": self.p.priority,
            "priority_params": self.p.priority_params,
            "eps": self.p.eps,
            "uses_cap": self.p.uses_cap,
            "fresh_target": self.p.fresh_target,
            "buffer_cap": self.p.buffer_size,
            **stats,
            "replay": replay_stats,
        }
        with open(self.p.metrics_path, "a") as f:
            f.write(json.dumps(line) + "\n")

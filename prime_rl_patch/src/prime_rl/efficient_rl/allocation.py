"""DUET (adaptive rollout allocation) and GRESO (pre-generation skip) dispatch
policies, both built on the shared ``PromptUtilityTracker``. Off by default
(``EFFRL_DUET`` / ``EFFRL_GRESO`` env vars); a baseline run touches none of
this code path.

Fidelity note (documented, not silently dropped): DUET's paper also includes
mid-generation early-abort of unproductive trajectories. That needs
cancelling an in-flight vLLM completion inside a group, which prime-rl's
dispatcher does support for *off-policy* drops (``drop_group``) but not for
"this group turned out low-value" drops -- doing it faithfully means
surgery on ``RolloutDispatcher.schedule_group_rollout``'s per-rollout
scheduling loop and was scoped out given the project's time budget. What is
implemented -- reallocating the rollout budget across prompts by predicted
utility -- is DUET's primary, more impactful mechanism per the paper summary
this project started from.
"""

from __future__ import annotations

import os
import random

from prime_rl.efficient_rl.utility_tracker import tracker_singleton


def duet_enabled() -> bool:
    return os.environ.get("EFFRL_DUET", "off") == "on"


def greso_enabled() -> bool:
    return os.environ.get("EFFRL_GRESO", "off") == "on"


def duet_group_size(prompt: str, base_group_size: int) -> int:
    """Reallocate the rollout budget across prompts based on observed utility
    (within-group reward variance) instead of a fixed group size for every
    prompt. No history yet -> base group size (unbiased default, matches
    baseline). Low variance (near-degenerate historically) -> shrink; high
    variance (informative) -> grow. Bounded to [min_frac, max_frac] x base so
    a single step's total rollout budget doesn't swing wildly -- the
    allocation is meant to be roughly budget-neutral on average across a
    step's batch, reallocating *where* budget goes rather than inflating it."""
    tracker = tracker_singleton()
    var, n = tracker.utility(prompt)
    if n == 0:
        return base_group_size
    low = float(os.environ.get("EFFRL_DUET_LOW_VAR", "0.01"))
    high = float(os.environ.get("EFFRL_DUET_HIGH_VAR", "0.05"))
    min_frac = float(os.environ.get("EFFRL_DUET_MIN_FRAC", "0.5"))
    max_frac = float(os.environ.get("EFFRL_DUET_MAX_FRAC", "1.5"))
    if var <= low:
        frac = min_frac
    elif var >= high:
        frac = max_frac
    else:
        t = (var - low) / (high - low)
        frac = min_frac + t * (max_frac - min_frac)
    return max(2, round(base_group_size * frac))


def greso_should_skip(prompt: str) -> bool:
    """Predict (from history) whether this prompt is likely to yield a
    zero/near-zero-variance group (every rollout ties -> no GRPO gradient) and
    skip generating it before paying the rollout cost. A small exploration
    floor keeps the estimate refreshing instead of permanently blacklisting a
    prompt on early noisy evidence."""
    tracker = tracker_singleton()
    var, n = tracker.utility(prompt)
    if n == 0:
        return False  # no history at all -> always try it once
    threshold = float(os.environ.get("EFFRL_GRESO_VAR_THRESHOLD", "0.01"))
    explore_eps = float(os.environ.get("EFFRL_GRESO_EXPLORE_EPS", "0.1"))
    if var > threshold:
        return False
    return random.random() > explore_eps

"""sGPO (sorted Group Policy Optimization) — offline-profiled data selection,
adaptive per-query rollout allocation, and an easy-to-hard curriculum, driven
by a single offline profiling signal p̂(q). arXiv:2606.08854 (Sudalairaj et
al., Red Hat AI Innovation / IBM).

One offline profiling pass (scripts/sgpo_profile.py) generates N=8 samples
per train query under the *initial* policy and computes an empirical success
rate p̂(q) = n_success/N. Training-time decisions are then all read from that
one file (``SGPO_PROFILE_FILE``), off by default (``EFFRL_SGPO=on``):

- Data selection (paper §4.3.1, threshold t = 0.75): trivial queries
  (p̂ > 0.75) are never dispatched — near-zero advantage, wasted training
  FLOPs. Unsolved queries (p̂ = 0) are dispatched with probability
  ``SGPO_UNSOLVED_MIX`` (paper: α = 10%), keeping long-horizon exploration
  alive without drowning the batch in zero-signal queries.
- Adaptive group size (paper §4.3.2, Eq. 7/10): learnable queries
  (0 < p̂ ≤ 0.75) get G ∈ {2, 4, 8} from the paper's power-of-two buckets of
  1/p̂ — the smallest group that still surfaces a success, i.e. max advantage
  per generated rollout.
- Curriculum (paper §4.3.3): phases advance at step boundaries
  (``SGPO_PHASE_BOUNDS``, comma-separated steps), easy (G=2 cluster) →
  hard (G=8 cluster), with unsolved queries mixed into every phase at the
  phase's group size (paper Eq. 9/13: G_j = g for all q_j ∈ C̄_g).

Fidelity notes vs. the paper (documented, not silently dropped): (i) the
paper trains each cluster to convergence (``SeqTrain``); here phases are
fixed step intervals within the run's ``max_steps`` budget, so the
curriculum is a step-scheduled approximation; (ii) the paper's unsolved
subsample is a fixed random subset per phase — here it is an online
Bernoulli(α) accept, the same set in expectation and self-replenishing
across dataset epochs; (iii) on reverse-text the paper's binary verifiable
reward is approximated by binarizing the continuous LCS reward at
``SGPO_SUCCESS_THRESHOLD`` during profiling (exact-match is used verbatim
for sciknoweval).
"""

from __future__ import annotations

import json
import os
import random
from functools import lru_cache

SGPO_BUCKETS = (2, 4, 8)
_TRIVIAL_THRESHOLD = 0.75


def sgpo_enabled() -> bool:
    return os.environ.get("EFFRL_SGPO", "off") == "on"


@lru_cache(maxsize=1)
def sgpo_profile() -> dict[int, dict]:
    """``{task_idx: entry}`` from ``SGPO_PROFILE_FILE`` (JSONL, one row per
    profiled query). Entry keys: ``idx``, ``prompt``, ``n_success``,
    ``p_hat``, ``bucket`` (2/4/8 for learnable, else null), ``decision``
    (``trivial`` | ``unsolved`` | ``learnable``). Cached for the process
    lifetime — profiling is offline and immutable within a run."""
    path = os.environ.get("SGPO_PROFILE_FILE", "")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"EFFRL_SGPO=on but SGPO_PROFILE_FILE not found: {path!r}")
    entries: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            entries[int(row["idx"])] = row
    return entries


def sgpo_entry(idx: int) -> dict | None:
    try:
        return sgpo_profile().get(idx)
    except Exception:
        return None


def sgpo_unsolved_mix() -> float:
    return float(os.environ.get("SGPO_UNSOLVED_MIX", "0.1"))


def sgpo_phase_bounds() -> list[int]:
    """Step numbers at which the curriculum advances to the next cluster.
    ``SGPO_PHASE_BOUNDS`` is a comma-separated list, e.g. ``10,20`` for three
    phases of 10 steps each within a 30-step run. Empty list = no
    curriculum (single phase over all learnable queries)."""
    raw = os.environ.get("SGPO_PHASE_BOUNDS", "")
    if not raw:
        return []
    return [int(x) for x in raw.split(",") if x.strip()]


def sgpo_phase_bucket(phase: int) -> int:
    """Group size of the cluster the given phase trains (paper Eq. 10:
    easy → hard). Phases past the last bucket stay in the hardest cluster."""
    return SGPO_BUCKETS[min(phase, len(SGPO_BUCKETS) - 1)]


def sgpo_should_skip(idx: int, phase: int) -> tuple[bool, str]:
    """Decide whether the popped example should be skipped at dispatch time
    (before any generation cost is paid). Returns ``(skip, reason)`` with
    reason in ``{"trivial", "phase", "unsolved", ""}``. A missing profile
    entry degrades to no-skip (baseline-equivalent) rather than stalling."""
    entry = sgpo_entry(idx)
    if entry is None:
        return False, ""
    if entry["decision"] == "trivial":
        return True, "trivial"
    if entry["decision"] == "unsolved":
        # Online Bernoulli(α) stand-in for the paper's fixed α-subsample of
        # D_unsolved mixed into every phase.
        skip = random.random() >= sgpo_unsolved_mix()
        return skip, "unsolved" if skip else ""
    if sgpo_phase_bounds() and entry["bucket"] != sgpo_phase_bucket(phase):
        return True, "phase"
    return False, ""


def sgpo_group_size(idx: int, phase: int, base_group_size: int) -> int:
    """Per-query group size: the query's own bucket for learnable queries
    (equal to the phase bucket by construction — ``sgpo_should_skip`` gates
    on that), the phase bucket for unsolved queries mixed in (paper Eq. 13),
    the base size for unprofiled queries (baseline-equivalent fallback)."""
    entry = sgpo_entry(idx)
    if entry is None:
        return base_group_size
    if entry["decision"] == "learnable":
        return int(entry["bucket"])
    return sgpo_phase_bucket(phase)

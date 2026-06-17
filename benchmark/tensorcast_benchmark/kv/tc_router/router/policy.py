"""Routing policy protocol + Phase 7 stub.

The four hooks declared in arch § 6.4:

  - `should_rebalance(loads, sessions, now_ts) -> list[MigrationDecision]`
  - `pick_target_instance(session, candidates, loads) -> InstanceId`
  - `pick_session_for_initial_home(session_id, candidates, loads) -> InstanceId`
  - `should_consider_session_for_migration(session, now_ts) -> bool`

Phase 7 ships `_NeverRebalance`: a stub that never proposes migrations and
picks the initial home via the same power-of-two rule the gateway baseline
uses, so `tc_router` is observably equivalent to `gw_load_aware`. Real
policies replace this file later without touching the rest of the router.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Sequence, runtime_checkable

from .state import (
    InstanceId,
    LoadSample,
    MigrationDecision,
    SessionId,
    SessionState,
)


@runtime_checkable
class Policy(Protocol):
    """Pluggable routing policy. All hooks are pure (no I/O)."""

    name: str

    def should_rebalance(
        self,
        loads: Mapping[InstanceId, LoadSample],
        sessions: Mapping[SessionId, SessionState],
        now_ts: float,
    ) -> list[MigrationDecision]: ...

    def pick_target_instance(
        self,
        session: SessionState,
        candidates: Sequence[InstanceId],
        loads: Mapping[InstanceId, LoadSample],
    ) -> InstanceId: ...

    def pick_session_for_initial_home(
        self,
        session_id: SessionId,
        candidates: Sequence[InstanceId],
        loads: Mapping[InstanceId, LoadSample],
    ) -> InstanceId: ...

    def should_consider_session_for_migration(
        self,
        session: SessionState,
        now_ts: float,
    ) -> bool: ...


# --- helpers --------------------------------------------------------------


def power_of_two_pick(
    candidates: Sequence[InstanceId],
    loads: Mapping[InstanceId, LoadSample],
    rng: random.Random,
) -> InstanceId:
    """Power-of-two-choices: sample 2 candidates uniformly, pick the less-loaded one.

    Mirrors `sgl-model-gateway` `--policy power_of_two`. If load samples are
    missing for any candidate, treat that candidate as `queue_depth = 0`
    (best load) so a fresh fleet doesn't get stuck routing to the same host.
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")
    if len(candidates) == 1:
        return candidates[0]
    idx_a = rng.randrange(len(candidates))
    # Pick a different index for idx_b.
    idx_b = (idx_a + 1 + rng.randrange(len(candidates) - 1)) % len(candidates)
    a, b = candidates[idx_a], candidates[idx_b]
    qa = loads[a].queue_depth if a in loads else 0
    qb = loads[b].queue_depth if b in loads else 0
    return a if qa <= qb else b


# --- _NeverRebalance --------------------------------------------------------


@dataclass
class _NeverRebalance:
    """Phase 7 stub policy.

    - Never proposes any migration (`should_rebalance` returns []).
    - Initial home picked via power-of-two over the live load map.
    - `should_consider_session_for_migration` is permanently False so the
      rebalancer wouldn't act even if `should_rebalance` were stubbed.

    Acceptance criteria (plan § 7 validation gate): wired into `tc_router`,
    must behave identically to `gw_load_aware` — same prompt stream, same
    TTFT shape, zero migrations recorded.
    """

    name: str = "NeverRebalance"
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def should_rebalance(
        self,
        loads: Mapping[InstanceId, LoadSample],
        sessions: Mapping[SessionId, SessionState],
        now_ts: float,
    ) -> list[MigrationDecision]:
        return []

    def pick_target_instance(
        self,
        session: SessionState,
        candidates: Sequence[InstanceId],
        loads: Mapping[InstanceId, LoadSample],
    ) -> InstanceId:
        # Never called by the rebalancer in Phase 7 (we never migrate), but
        # implement it sensibly anyway in case future code paths invoke it.
        return power_of_two_pick(candidates, loads, self._rng)

    def pick_session_for_initial_home(
        self,
        session_id: SessionId,
        candidates: Sequence[InstanceId],
        loads: Mapping[InstanceId, LoadSample],
    ) -> InstanceId:
        return power_of_two_pick(candidates, loads, self._rng)

    def should_consider_session_for_migration(
        self,
        session: SessionState,
        now_ts: float,
    ) -> bool:
        return False


def make_policy(spec: Optional[dict]) -> Policy:
    """Build a Policy from a `benchmark.yaml`-style `policy:` dict.

    Phase 7 supports only `kind: never_rebalance` (or absent → default to
    NeverRebalance). Later phases register threshold / eager / etc. here.
    """
    if spec is None:
        return _NeverRebalance()
    kind = (spec.get("kind") or "never_rebalance").strip()
    if kind in {"never_rebalance", "never", "stub"}:
        seed = int(spec.get("seed", 0))
        return _NeverRebalance(seed=seed)
    raise ValueError(
        f"unknown policy kind {kind!r}; Phase 7 supports only 'never_rebalance'"
    )

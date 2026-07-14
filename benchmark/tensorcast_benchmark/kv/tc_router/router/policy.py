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
    pending_migration_wait_timeout_s: float
    plan_deadline_ms: int
    publish_ttl_ms: int

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
    pending_migration_wait_timeout_s: float = 0.0
    plan_deadline_ms: int = 30_000
    publish_ttl_ms: int = 600_000

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


@dataclass
class MigrateOnceAfterTurnPolicy:
    """Correctness policy: migrate each eligible session at most once."""

    name: str = "MigrateOnceAfterTurnPolicy"
    seed: int = 0
    after_turn_count: int = 1
    target_strategy: str = "next_instance"
    max_migrations_per_session: int = 1
    pending_migration_wait_timeout_s: float = 30.0
    plan_deadline_ms: int = 30_000
    publish_ttl_ms: int = 600_000

    def __post_init__(self) -> None:
        if self.after_turn_count < 1:
            raise ValueError("after_turn_count must be >= 1")
        if self.max_migrations_per_session < 1:
            raise ValueError("max_migrations_per_session must be >= 1")
        if self.target_strategy not in {"next_instance", "least_loaded"}:
            raise ValueError(
                "target_strategy must be one of: next_instance, least_loaded"
            )
        if self.pending_migration_wait_timeout_s < 0:
            raise ValueError("pending_migration_wait_timeout_s must be >= 0")
        if self.plan_deadline_ms <= 0:
            raise ValueError("plan_deadline_ms must be > 0")
        if self.publish_ttl_ms <= 0:
            raise ValueError("publish_ttl_ms must be > 0")
        self._rng = random.Random(self.seed)

    def should_rebalance(
        self,
        loads: Mapping[InstanceId, LoadSample],
        sessions: Mapping[SessionId, SessionState],
        now_ts: float,
    ) -> list[MigrationDecision]:
        decisions: list[MigrationDecision] = []
        candidates = sorted(
            set(loads) | {session.home_instance for session in sessions.values()}
        )
        for session in sessions.values():
            if not self.should_consider_session_for_migration(session, now_ts):
                continue
            target = self.pick_target_instance(session, candidates, loads)
            if target == session.home_instance:
                continue
            decisions.append(
                MigrationDecision(
                    session_id=session.session_id,
                    source_instance=session.home_instance,
                    target_instance=target,
                    decided_by=self.name,
                )
            )
        return decisions

    def pick_target_instance(
        self,
        session: SessionState,
        candidates: Sequence[InstanceId],
        loads: Mapping[InstanceId, LoadSample],
    ) -> InstanceId:
        non_source = [
            candidate for candidate in candidates if candidate != session.home_instance
        ]
        if not non_source:
            return session.home_instance
        if self.target_strategy == "least_loaded":
            return min(
                non_source,
                key=lambda instance_id: (
                    loads[instance_id].queue_depth if instance_id in loads else 0
                ),
            )
        ordered = sorted(candidates)
        try:
            idx = ordered.index(session.home_instance)
        except ValueError:
            return non_source[0]
        for offset in range(1, len(ordered) + 1):
            candidate = ordered[(idx + offset) % len(ordered)]
            if candidate != session.home_instance:
                return candidate
        return session.home_instance

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
        return (
            bool(session.home_instance)
            and bool(session.last_engine_request_id)
            and session.turn_count >= self.after_turn_count
            and session.pending_migration is None
            and session.migration_attempt_count < self.max_migrations_per_session
        )


def make_policy(spec: Optional[dict]) -> Policy:
    """Build a Policy from a `benchmark.yaml`-style `policy:` dict.

    `never_rebalance` preserves the Phase 7 sticky stub. Phase 8 adds
    `migrate_once_after_turn` to validate the migration primitive before
    tuning a real load-aware policy.
    """
    if spec is None:
        return _NeverRebalance()
    kind = (spec.get("kind") or "never_rebalance").strip()
    if kind in {"never_rebalance", "never", "stub"}:
        seed = int(spec.get("seed", 0))
        return _NeverRebalance(seed=seed)
    if kind in {"migrate_once_after_turn", "migrate_once", "migration_smoke"}:
        return MigrateOnceAfterTurnPolicy(
            seed=int(spec.get("seed", 0)),
            after_turn_count=int(spec.get("after_turn_count", 1)),
            target_strategy=str(spec.get("target_strategy", "next_instance")),
            max_migrations_per_session=int(spec.get("max_migrations_per_session", 1)),
            pending_migration_wait_timeout_s=float(
                spec.get("pending_migration_wait_timeout_s", 30.0)
            ),
            plan_deadline_ms=int(spec.get("plan_deadline_ms", 30_000)),
            publish_ttl_ms=int(spec.get("publish_ttl_ms", 600_000)),
        )
    raise ValueError(
        "unknown policy kind "
        f"{kind!r}; supported: 'never_rebalance', 'migrate_once_after_turn'"
    )

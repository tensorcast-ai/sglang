"""Tests for router/policy.py (NeverRebalance stub + power-of-two helper)."""

from __future__ import annotations

import time

import pytest

from tensorcast_benchmark.kv.tc_router.router.policy import (
    MigrateOnceAfterTurnPolicy,
    Policy,
    _NeverRebalance,
    make_policy,
    power_of_two_pick,
)
from tensorcast_benchmark.kv.tc_router.router.state import (
    LoadSample,
    SessionState,
)


def _ls(instance_id: str, q: int) -> LoadSample:
    return LoadSample(
        instance_id=instance_id,
        num_waiting_reqs=q // 2,
        num_running_reqs=q - q // 2,
        token_usage=0.0,
        utilization=0.0,
        gen_throughput=0.0,
        timestamp_monotonic=0.0,
    )


# --- power_of_two_pick ------------------------------------------------------


def test_pot_single_candidate_returns_it() -> None:
    import random as r

    assert power_of_two_pick(["only"], {}, r.Random(0)) == "only"


def test_pot_picks_less_loaded() -> None:
    import random as r

    loads = {"a": _ls("a", 8), "b": _ls("b", 2)}
    # With only 2 candidates, sampling always picks both → least-loaded wins.
    rng = r.Random(0)
    counts = {"a": 0, "b": 0}
    for _ in range(50):
        counts[power_of_two_pick(["a", "b"], loads, rng)] += 1
    assert counts["b"] == 50  # always picks b


def test_pot_missing_load_treated_as_zero() -> None:
    import random as r

    loads = {"a": _ls("a", 10)}  # b missing
    # b's missing load is treated as queue_depth=0 → b wins against a's 10.
    rng = r.Random(0)
    counts = {"a": 0, "b": 0}
    for _ in range(50):
        counts[power_of_two_pick(["a", "b"], loads, rng)] += 1
    assert counts["b"] == 50


def test_pot_empty_candidates_raises() -> None:
    import random as r

    with pytest.raises(ValueError, match="non-empty"):
        power_of_two_pick([], {}, r.Random(0))


# --- _NeverRebalance --------------------------------------------------------


def test_never_rebalance_implements_protocol() -> None:
    pol = _NeverRebalance()
    assert isinstance(pol, Policy)
    assert pol.name == "NeverRebalance"


def test_never_rebalance_returns_no_migrations() -> None:
    pol = _NeverRebalance()
    decisions = pol.should_rebalance(loads={}, sessions={}, now_ts=time.monotonic())
    assert decisions == []


def test_never_rebalance_session_not_migratable() -> None:
    pol = _NeverRebalance()
    s = SessionState(session_id="x", home_instance="a", turn_count=10)
    assert (
        pol.should_consider_session_for_migration(s, now_ts=time.monotonic()) is False
    )


def test_never_rebalance_initial_home_uses_power_of_two() -> None:
    pol = _NeverRebalance(seed=0)
    loads = {"a": _ls("a", 9), "b": _ls("b", 1)}
    home = pol.pick_session_for_initial_home(
        session_id="s",
        candidates=["a", "b"],
        loads=loads,
    )
    # 2 candidates + always-picks-less-loaded → b
    assert home == "b"


def test_never_rebalance_seed_is_deterministic() -> None:
    loads = {}  # no loads → uniform random pick
    a = _NeverRebalance(seed=42)
    b = _NeverRebalance(seed=42)
    seq_a = [
        a.pick_session_for_initial_home("s", ["x", "y", "z"], loads) for _ in range(20)
    ]
    seq_b = [
        b.pick_session_for_initial_home("s", ["x", "y", "z"], loads) for _ in range(20)
    ]
    assert seq_a == seq_b


# --- make_policy factory ----------------------------------------------------


def test_make_policy_default_is_never_rebalance() -> None:
    pol = make_policy(None)
    assert isinstance(pol, _NeverRebalance)


def test_make_policy_explicit_never_rebalance() -> None:
    pol = make_policy({"kind": "never_rebalance", "seed": 7})
    assert isinstance(pol, _NeverRebalance)
    assert pol.seed == 7


def test_make_policy_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown policy"):
        make_policy({"kind": "magical_rebalance"})


# --- MigrateOnceAfterTurnPolicy ---------------------------------------------


def _eligible_session(
    *,
    home_instance: str = "a",
    turn_count: int = 2,
    last_engine_request_id: str = "rid-1",
    migration_attempt_count: int = 0,
) -> SessionState:
    state = SessionState(
        session_id="sess",
        home_instance=home_instance,
        turn_count=turn_count,
        last_engine_request_id=last_engine_request_id,
    )
    state.migration_attempt_count = migration_attempt_count
    return state


def test_make_policy_builds_migrate_once_after_turn() -> None:
    pol = make_policy(
        {
            "kind": "migrate_once_after_turn",
            "seed": 7,
            "after_turn_count": 2,
            "target_strategy": "least_loaded",
            "max_migrations_per_session": 2,
            "pending_migration_wait_timeout_s": 5,
            "plan_deadline_ms": 123,
            "publish_ttl_ms": 456,
        }
    )

    assert isinstance(pol, MigrateOnceAfterTurnPolicy)
    assert pol.seed == 7
    assert pol.after_turn_count == 2
    assert pol.target_strategy == "least_loaded"
    assert pol.max_migrations_per_session == 2
    assert pol.pending_migration_wait_timeout_s == 5
    assert pol.plan_deadline_ms == 123
    assert pol.publish_ttl_ms == 456


def test_migrate_once_waits_until_turn_threshold() -> None:
    pol = MigrateOnceAfterTurnPolicy(after_turn_count=2)
    session = _eligible_session(turn_count=1)

    decisions = pol.should_rebalance(
        loads={"a": _ls("a", 0), "b": _ls("b", 0)},
        sessions={"sess": session},
        now_ts=time.monotonic(),
    )

    assert decisions == []


def test_migrate_once_proposes_next_instance_after_eligibility() -> None:
    pol = MigrateOnceAfterTurnPolicy(
        after_turn_count=2, target_strategy="next_instance"
    )
    session = _eligible_session(turn_count=2)

    decisions = pol.should_rebalance(
        loads={"a": _ls("a", 0), "b": _ls("b", 0), "c": _ls("c", 0)},
        sessions={"sess": session},
        now_ts=time.monotonic(),
    )

    assert len(decisions) == 1
    assert decisions[0].source_instance == "a"
    assert decisions[0].target_instance == "b"


def test_migrate_once_least_loaded_excludes_current_home() -> None:
    pol = MigrateOnceAfterTurnPolicy(target_strategy="least_loaded")
    session = _eligible_session(home_instance="a")

    decisions = pol.should_rebalance(
        loads={"a": _ls("a", 0), "b": _ls("b", 5), "c": _ls("c", 1)},
        sessions={"sess": session},
        now_ts=time.monotonic(),
    )

    assert decisions[0].target_instance == "c"


def test_migrate_once_respects_pending_migration() -> None:
    pol = MigrateOnceAfterTurnPolicy()
    session = _eligible_session()
    session.pending_migration = object()  # type: ignore[assignment]

    decisions = pol.should_rebalance(
        loads={"a": _ls("a", 0), "b": _ls("b", 0)},
        sessions={"sess": session},
        now_ts=time.monotonic(),
    )

    assert decisions == []


def test_migrate_once_respects_max_migrations() -> None:
    pol = MigrateOnceAfterTurnPolicy(max_migrations_per_session=1)
    session = _eligible_session(migration_attempt_count=1)

    decisions = pol.should_rebalance(
        loads={"a": _ls("a", 0), "b": _ls("b", 0)},
        sessions={"sess": session},
        now_ts=time.monotonic(),
    )

    assert decisions == []

"""Tests for router/policy.py (NeverRebalance stub + power-of-two helper)."""

from __future__ import annotations

import time

import pytest

from tensorcast_benchmark.kv.tc_router.router.policy import (
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
    assert pol.should_consider_session_for_migration(s, now_ts=time.monotonic()) is False


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
    seq_a = [a.pick_session_for_initial_home("s", ["x", "y", "z"], loads) for _ in range(20)]
    seq_b = [b.pick_session_for_initial_home("s", ["x", "y", "z"], loads) for _ in range(20)]
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

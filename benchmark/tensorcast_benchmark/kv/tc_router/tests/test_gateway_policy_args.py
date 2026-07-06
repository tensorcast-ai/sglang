"""Tests for gateway baseline policy knob translation."""

from __future__ import annotations

import pytest

from tensorcast_benchmark.kv.tc_router.driver.benchmark_loop import (
    _gateway_extra_args,
)
from tensorcast_benchmark.kv.tc_router.driver.config import ConfigSpec


def test_gateway_extra_args_empty_when_policy_absent() -> None:
    spec = ConfigSpec(kind="gw_cache_aware")

    assert _gateway_extra_args(spec) == ()


def test_gateway_extra_args_translates_cache_aware_knobs() -> None:
    spec = ConfigSpec(
        kind="gw_cache_aware",
        policy={
            "cache_threshold": 0.0,
            "balance_abs_threshold": 1_000_000,
            "balance_rel_threshold": 1.5,
        },
    )

    assert _gateway_extra_args(spec) == (
        "--cache-threshold",
        "0.0",
        "--balance-abs-threshold",
        "1000000",
        "--balance-rel-threshold",
        "1.5",
    )


def test_gateway_extra_args_rejects_unknown_cache_aware_knob() -> None:
    spec = ConfigSpec(kind="gw_cache_aware", policy={"unknown": 1})

    with pytest.raises(ValueError, match="unknown gw_cache_aware"):
        _gateway_extra_args(spec)


def test_gateway_extra_args_rejects_load_aware_policy_knobs() -> None:
    spec = ConfigSpec(kind="gw_load_aware", policy={"cache_threshold": 0.0})

    with pytest.raises(ValueError, match="does not accept"):
        _gateway_extra_args(spec)

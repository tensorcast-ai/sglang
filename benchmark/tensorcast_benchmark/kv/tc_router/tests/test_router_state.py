"""Tests for router/state.py dataclasses."""

from __future__ import annotations

import asyncio

import pytest

from tensorcast_benchmark.kv.tc_router.router.state import (
    LoadSample,
    MigrationDecision,
    MigrationFuture,
    SessionState,
)


def test_session_state_defaults() -> None:
    s = SessionState(session_id="x")
    assert s.home_instance == ""
    assert s.last_active_ts == 0.0
    assert s.turn_count == 0
    assert s.last_engine_request_id == ""
    assert s.last_published_manifest is None
    assert s.pending_migration is False


def test_session_state_is_mutable() -> None:
    s = SessionState(session_id="x")
    s.home_instance = "inst-0"
    s.turn_count += 1
    s.pending_migration = True
    assert s.home_instance == "inst-0"
    assert s.turn_count == 1
    assert s.pending_migration is True


def test_load_sample_queue_depth() -> None:
    ls = LoadSample(
        instance_id="inst-0",
        num_waiting_reqs=3,
        num_running_reqs=5,
        token_usage=0.1,
        utilization=0.2,
        gen_throughput=0.3,
        timestamp_monotonic=0.0,
    )
    assert ls.queue_depth == 8


def test_migration_decision_is_frozen() -> None:
    d = MigrationDecision(
        session_id="s",
        source_instance="A",
        target_instance="B",
        decided_by="ThresholdPolicy",
    )
    with pytest.raises(Exception):
        d.session_id = "other"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_migration_future_signals_completion() -> None:
    f = MigrationFuture(
        session_id="s",
        source_instance="A",
        target_instance="B",
        started_monotonic=0.0,
    )
    assert not f.completion_event.is_set()
    assert f.success is None

    async def waiter() -> bool:
        await f.completion_event.wait()
        return f.success is True

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0)  # let waiter start
    f.mark_success()
    assert await task is True
    assert f.completion_event.is_set()


@pytest.mark.asyncio
async def test_migration_future_signals_failure() -> None:
    f = MigrationFuture(
        session_id="s",
        source_instance="A",
        target_instance="B",
        started_monotonic=0.0,
    )
    f.mark_failure("publish timed out")
    assert f.success is False
    assert f.error == "publish timed out"
    assert f.completion_event.is_set()

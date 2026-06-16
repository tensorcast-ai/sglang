"""Tests for metrics/summary.py — quantile correctness + integration with workload driver."""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from typing import Any

import pytest

from tensorcast_benchmark.kv.tc_router.metrics.per_turn import TurnRecordWriter
from tensorcast_benchmark.kv.tc_router.metrics.summary import (
    RunSummary,
    _quantile,
    aggregate_cell,
    write_summary_csv,
)
from tensorcast_benchmark.kv.tc_router.router.interface import GenerateResult
from tensorcast_benchmark.kv.tc_router.workload.generator import WorkloadDriver
from tensorcast_benchmark.kv.tc_router.workload.trajectory_pool import Trajectory


# --- pure helper -------------------------------------------------------------


def test_quantile_matches_known_values() -> None:
    xs = list(range(1, 11))  # 1..10
    # Linear interpolation between order statistics (numpy 'linear').
    assert _quantile(xs, 0.50) == pytest.approx(5.5)
    assert _quantile(xs, 0.0) == 1
    assert _quantile(xs, 1.0) == 10
    assert _quantile(xs, 0.95) == pytest.approx(9.55)


def test_quantile_empty_returns_none() -> None:
    assert _quantile([], 0.5) is None


# --- aggregate_cell ---------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _turn(success: bool = True, ttft_ms: float | None = 50.0, prompt_tokens: int = 1000, cached_tokens: int = 800, **extra: Any) -> dict:
    base = {
        "ts": 0.0,
        "session_id": "s",
        "instance_id": "task",
        "turn_index": 0,
        "prompt_messages_count": 2,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": 32,
        "served_instance": "inst-0",
        "ttft_ms": ttft_ms,
        "latency_ms": 100.0,
        "cached_tokens": cached_tokens,
        "used_hydrated_bundle": False,
        "was_just_migrated": False,
        "rid": "r",
        "success": success,
        "error_message": "",
    }
    base.update(extra)
    return base


def test_aggregate_basic_quantiles(tmp_path: Path) -> None:
    turns = [_turn(ttft_ms=t, prompt_tokens=1000, cached_tokens=800) for t in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
    turns_path = tmp_path / "turns.jsonl"
    _write_jsonl(turns_path, turns)

    summary = aggregate_cell(
        turns_path=turns_path,
        migrations_path=None,
        config="gw_load_aware",
        c_target=4,
        trial=0,
        inter_turn_delay_preset="agent_medium",
        transport_mode="rdma",
    )

    assert summary.total_turns_completed == 10
    assert summary.total_requests_failed == 0
    assert summary.ttft_p50_ms == pytest.approx(55.0)
    assert summary.ttft_p95_ms == pytest.approx(95.5)
    assert summary.ttft_p99_ms == pytest.approx(99.1)
    assert summary.ttft_mean_ms == pytest.approx(55.0)
    assert summary.cached_token_ratio_mean == pytest.approx(0.8)
    assert summary.migration_count == 0
    assert summary.migration_utilization is None
    assert summary.mean_publish_latency_ms is None
    assert summary.mean_hydrate_latency_ms is None


def test_aggregate_excludes_failed_turns_from_ttft(tmp_path: Path) -> None:
    turns = [
        _turn(success=True, ttft_ms=10.0),
        _turn(success=False, ttft_ms=None, error_message="boom"),
        _turn(success=True, ttft_ms=20.0),
    ]
    turns_path = tmp_path / "turns.jsonl"
    _write_jsonl(turns_path, turns)

    summary = aggregate_cell(
        turns_path=turns_path,
        migrations_path=None,
        config="gw_load_aware",
        c_target=2,
        trial=0,
        inter_turn_delay_preset="agent_fast",
        transport_mode="rdma",
    )
    assert summary.total_turns_completed == 3
    assert summary.total_requests_failed == 1
    # Mean computed only over the two successful turns.
    assert summary.ttft_mean_ms == pytest.approx(15.0)


def test_aggregate_handles_empty_turns_jsonl(tmp_path: Path) -> None:
    turns_path = tmp_path / "turns.jsonl"
    turns_path.write_text("")  # empty
    summary = aggregate_cell(
        turns_path=turns_path,
        migrations_path=None,
        config="x",
        c_target=1,
        trial=0,
        inter_turn_delay_preset="agent_medium",
        transport_mode="tcp",
    )
    assert summary.total_turns_completed == 0
    assert summary.total_requests_failed == 0
    assert summary.ttft_p50_ms is None
    assert summary.ttft_mean_ms is None
    assert summary.cached_token_ratio_mean is None


def test_aggregate_handles_missing_migrations_file(tmp_path: Path) -> None:
    turns_path = tmp_path / "turns.jsonl"
    _write_jsonl(turns_path, [_turn()])
    # `migrations_path` exists in plan but file does not → treated as empty
    summary = aggregate_cell(
        turns_path=turns_path,
        migrations_path=tmp_path / "missing.jsonl",
        config="tc_router",
        c_target=4,
        trial=0,
        inter_turn_delay_preset="agent_medium",
        transport_mode="rdma",
    )
    assert summary.migration_count == 0
    assert summary.migration_utilization is None


def test_aggregate_with_migrations(tmp_path: Path) -> None:
    turns_path = tmp_path / "turns.jsonl"
    _write_jsonl(turns_path, [_turn() for _ in range(5)])

    migs = [
        # consumed
        {"ts": 0, "session_id": "s1", "source_instance": "A", "target_instance": "B",
         "publish_latency_ms": 30.0, "hydrate_latency_ms": 80.0,
         "transferred_bytes_estimated": 1024, "decided_by": "T",
         "consumed_by_turn_rid": "r1", "consumed_within_s": 5.0, "wasted": False},
        # consumed
        {"ts": 0, "session_id": "s2", "source_instance": "A", "target_instance": "C",
         "publish_latency_ms": 50.0, "hydrate_latency_ms": 100.0,
         "transferred_bytes_estimated": 2048, "decided_by": "T",
         "consumed_by_turn_rid": "r2", "consumed_within_s": 6.0, "wasted": False},
        # wasted (rejected from utilization numerator)
        {"ts": 0, "session_id": "s3", "source_instance": "A", "target_instance": "B",
         "publish_latency_ms": 40.0, "hydrate_latency_ms": 90.0,
         "transferred_bytes_estimated": 1024, "decided_by": "T",
         "consumed_by_turn_rid": None, "consumed_within_s": None, "wasted": True},
    ]
    migrations_path = tmp_path / "migrations.jsonl"
    _write_jsonl(migrations_path, migs)

    summary = aggregate_cell(
        turns_path=turns_path,
        migrations_path=migrations_path,
        config="tc_router",
        c_target=8,
        trial=1,
        inter_turn_delay_preset="agent_medium",
        transport_mode="rdma",
    )
    assert summary.migration_count == 3
    assert summary.migration_utilization == pytest.approx(2 / 3)
    assert summary.mean_publish_latency_ms == pytest.approx(40.0)
    assert summary.mean_hydrate_latency_ms == pytest.approx(90.0)


def test_write_summary_csv_round_trip(tmp_path: Path) -> None:
    rows = [
        RunSummary(
            config="gw_load_aware",
            c_target=3,
            trial=0,
            inter_turn_delay_preset="agent_medium",
            transport_mode="rdma",
            ttft_p50_ms=10.0,
            ttft_p95_ms=20.0,
            ttft_p99_ms=22.0,
            ttft_mean_ms=12.0,
            cached_token_ratio_mean=0.7,
            total_turns_completed=15,
            total_requests_failed=0,
        )
    ]
    out = tmp_path / "summary.csv"
    write_summary_csv(rows, out)

    with out.open() as fh:
        records = list(csv.DictReader(fh))
    assert len(records) == 1
    assert records[0]["config"] == "gw_load_aware"
    assert float(records[0]["ttft_p50_ms"]) == pytest.approx(10.0)
    assert int(records[0]["c_target"]) == 3
    assert records[0]["mean_publish_latency_ms"] == ""  # None serializes as empty


# --- integration: WorkloadDriver → turns.jsonl → aggregator ----------------


class _IntegrationMockRouter:
    """MockRouter producing varied prompt/cached tokens to exercise the aggregator."""

    def __init__(self) -> None:
        self.n = 0

    async def generate(self, *, session_id, messages, tools, sampling_params) -> GenerateResult:
        self.n += 1
        await asyncio.sleep(0)
        # Vary TTFT so quantiles differ.
        return GenerateResult(
            text="ok",
            ttft_ms=10.0 * self.n,
            latency_ms=20.0 * self.n,
            served_instance="mock:0",
            prompt_tokens=1000 * self.n,
            cached_tokens=800 * self.n,
            success=True,
        )

    async def close(self) -> None: pass


def _msg(role: str, content: str = "", tool_calls=None, tool_call_id=None, name=None) -> dict:
    return {"role": role, "content": content, "tool_calls": tool_calls,
            "tool_call_id": tool_call_id, "name": name, "function_call": None}


def _build_pool(n: int, num_assistants: int = 3, content_len: int = 200) -> list[Trajectory]:
    pool: list[Trajectory] = []
    for i in range(n):
        messages = [_msg("system", "sys"), _msg("user", "u" * content_len)]
        for k in range(num_assistants):
            messages.append(_msg("assistant", "", tool_calls=[
                {"id": f"c{k}", "index": 0, "type": "function",
                 "function": {"name": "f", "arguments": "{}"}}]))
            messages.append(_msg("tool", "t" * content_len, tool_call_id=f"c{k}", name="f"))
        assistant_idx = tuple(j for j, m in enumerate(messages) if m["role"] == "assistant")
        pool.append(Trajectory(
            session_id=f"sess{i}", instance_id=f"task{i}",
            messages=tuple(messages), tools=(),
            assistant_indices=assistant_idx,
            total_chars=sum(len(m.get("content") or "") for m in messages),
            estimated_tokens=1000, resolved=False,
        ))
    return pool


@pytest.mark.asyncio
async def test_aggregator_reads_workload_driver_output(tmp_path: Path) -> None:
    """Plan validation gate: WorkloadDriver writes turns.jsonl, summary.py
    aggregates a non-trivial RunSummary row.
    """
    pool = _build_pool(3, num_assistants=3)
    turns_path = tmp_path / "turns.jsonl"

    with TurnRecordWriter(turns_path) as writer:
        driver = WorkloadDriver(
            _IntegrationMockRouter(),
            pool,
            lambda: 0.0,
            c_target=2,
            wall_seconds=0.5,
            warmup_seconds=0.0,
            start_jitter_s=0.0,
            max_new_tokens_clip=64,
            record_sink=writer.write,
            rng_seed=0,
            supervisor_tick_s=0.01,
        )
        outcome = await driver.run()

    assert outcome.total_turns >= 3

    summary = aggregate_cell(
        turns_path=turns_path,
        migrations_path=None,
        config="gw_load_aware",
        c_target=2,
        trial=0,
        inter_turn_delay_preset="agent_medium",
        transport_mode="rdma",
    )
    assert summary.total_turns_completed == outcome.total_turns
    assert summary.total_requests_failed == 0
    assert summary.ttft_p50_ms is not None and summary.ttft_p50_ms > 0
    assert summary.ttft_mean_ms is not None and summary.ttft_mean_ms > 0
    assert summary.cached_token_ratio_mean == pytest.approx(0.8)

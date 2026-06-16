"""WorkloadDriver dry-run tests.

Use a `MockRouter` that returns instantly so tests don't need a live SGLang.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import pytest

from tensorcast_benchmark.kv.tc_router.metrics.per_turn import (
    TurnRecord,
    TurnRecordWriter,
)
from tensorcast_benchmark.kv.tc_router.router.interface import GenerateResult
from tensorcast_benchmark.kv.tc_router.workload.generator import WorkloadDriver
from tensorcast_benchmark.kv.tc_router.workload.trajectory_pool import Trajectory


class MockRouter:
    """Router stub that returns an immediate, deterministic GenerateResult."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate(
        self,
        *,
        session_id: str,
        messages,
        tools,
        sampling_params,
    ) -> GenerateResult:
        # Tiny await so we don't block the supervisor refill loop tightly.
        await asyncio.sleep(0)
        self.calls.append(
            {
                "session_id": session_id,
                "messages_count": len(messages),
                "tools_count": len(tools) if tools is not None else 0,
                "sampling_params": dict(sampling_params),
            }
        )
        return GenerateResult(
            text="ok",
            ttft_ms=10.0,
            latency_ms=20.0,
            served_instance="mock:0",
            prompt_tokens=len(messages) * 100,
            cached_tokens=max(0, (len(messages) - 1) * 100),
            success=True,
        )

    async def close(self) -> None:
        pass


def _msg(role: str, content: str = "", tool_calls=None, tool_call_id=None, name=None) -> dict:
    return {
        "role": role,
        "content": content,
        "tool_calls": tool_calls,
        "tool_call_id": tool_call_id,
        "name": name,
        "function_call": None,
    }


def _tool_call(call_id: str = "c1") -> dict:
    return {
        "id": call_id,
        "index": 0,
        "type": "function",
        "function": {"name": "f", "arguments": "{}"},
    }


def _build_pool(n: int, *, num_assistants: int = 4, content_len: int = 200) -> list[Trajectory]:
    pool: list[Trajectory] = []
    for i in range(n):
        messages: list[dict] = [
            _msg("system", "sys"),
            _msg("user", "u" * content_len),
        ]
        for k in range(num_assistants):
            messages.append(_msg("assistant", "", tool_calls=[_tool_call(f"c{k}")]))
            messages.append(_msg("tool", "t" * content_len, tool_call_id=f"c{k}", name="f"))
        assistant_idx = tuple(j for j, m in enumerate(messages) if m["role"] == "assistant")
        pool.append(
            Trajectory(
                session_id=f"sess{i}",
                instance_id=f"task{i}",
                messages=tuple(messages),
                tools=(),
                assistant_indices=assistant_idx,
                total_chars=sum(len(m.get("content") or "") for m in messages),
                estimated_tokens=1000,
                resolved=False,
            )
        )
    return pool


# --- core integration tests --------------------------------------------------


@pytest.mark.asyncio
async def test_dry_run_records_turns_per_session() -> None:
    """Plan validation gate: 5 sessions, ~1s wall-time, MockRouter — assert
    multiple turns per session and TurnRecord fields populated."""
    pool = _build_pool(20, num_assistants=4)

    # Tiny inter-turn delay so we get several turns per session in 1s.
    delay = lambda: 0.01

    router = MockRouter()
    sink_records: list[TurnRecord] = []
    driver = WorkloadDriver(
        router,
        pool,
        delay,
        c_target=5,
        wall_seconds=1.0,
        warmup_seconds=0.0,
        start_jitter_s=0.0,
        max_new_tokens_clip=64,
        record_sink=sink_records.append,
        rng_seed=0,
        supervisor_tick_s=0.01,
    )
    outcome = await driver.run()

    assert outcome.total_turns >= 5  # at least one turn per active session
    assert outcome.successful_turns == outcome.total_turns
    assert outcome.failed_turns == 0
    assert outcome.distinct_sessions_started >= 5
    # Driver-internal records and sink records agree.
    assert len(sink_records) == outcome.total_turns
    assert len(driver.records) == outcome.total_turns

    # At least one session must have run more than one turn (multi-turn proof).
    per_session = {}
    for r in sink_records:
        per_session.setdefault(r.session_id, 0)
        per_session[r.session_id] += 1
    assert max(per_session.values()) > 1, f"expected multi-turn; got {per_session}"


@pytest.mark.asyncio
async def test_turn_record_fields_populated_correctly() -> None:
    pool = _build_pool(3, num_assistants=2)
    router = MockRouter()
    driver = WorkloadDriver(
        router,
        pool,
        lambda: 0.0,
        c_target=1,
        wall_seconds=0.5,
        warmup_seconds=0.0,
        start_jitter_s=0.0,
        max_new_tokens_clip=64,
        rng_seed=0,
        supervisor_tick_s=0.01,
    )
    await driver.run()

    rec = driver.records[0]
    assert rec.session_id.startswith("sess")
    assert rec.instance_id.startswith("task")
    assert rec.turn_index == 0
    assert rec.prompt_messages_count == 2  # [system, user] before first assistant
    assert rec.served_instance == "mock:0"
    assert rec.ttft_ms == 10.0
    assert rec.latency_ms is not None and rec.latency_ms > 0
    assert rec.rid.startswith("tcrouter:sess")
    assert rec.success is True
    assert rec.used_hydrated_bundle is False
    assert rec.was_just_migrated is False


@pytest.mark.asyncio
async def test_router_failure_recorded_as_failed_turn() -> None:
    class FlakyRouter:
        def __init__(self) -> None:
            self.n = 0

        async def generate(self, *, session_id, messages, tools, sampling_params):
            self.n += 1
            if self.n == 2:
                raise RuntimeError("boom")
            return GenerateResult(text="ok", ttft_ms=1.0, served_instance="x:0")

        async def close(self) -> None:
            pass

    pool = _build_pool(1, num_assistants=4)
    driver = WorkloadDriver(
        FlakyRouter(),
        pool,
        lambda: 0.0,
        c_target=1,
        wall_seconds=0.3,
        start_jitter_s=0.0,
        max_new_tokens_clip=64,
        rng_seed=0,
        supervisor_tick_s=0.01,
    )
    outcome = await driver.run()
    assert outcome.failed_turns == 1
    assert any(not r.success and "boom" in r.error_message for r in driver.records)


@pytest.mark.asyncio
async def test_jsonl_sink_writes_records(tmp_path: Path) -> None:
    pool = _build_pool(3, num_assistants=3)
    out_path = tmp_path / "turns.jsonl"
    with TurnRecordWriter(out_path) as writer:
        driver = WorkloadDriver(
            MockRouter(),
            pool,
            lambda: 0.0,
            c_target=2,
            wall_seconds=0.3,
            start_jitter_s=0.0,
            max_new_tokens_clip=64,
            record_sink=writer.write,
            rng_seed=0,
            supervisor_tick_s=0.01,
        )
        outcome = await driver.run()

    assert out_path.exists()
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == outcome.total_turns
    parsed = [json.loads(line) for line in lines]
    assert all("session_id" in p for p in parsed)
    assert all("ttft_ms" in p for p in parsed)


@pytest.mark.asyncio
async def test_pool_exhaustion_does_not_crash() -> None:
    """If pool is smaller than c_target we should still finish cleanly."""
    pool = _build_pool(2, num_assistants=2)
    driver = WorkloadDriver(
        MockRouter(),
        pool,
        lambda: 0.0,
        c_target=10,
        wall_seconds=0.3,
        start_jitter_s=0.0,
        max_new_tokens_clip=64,
        rng_seed=0,
        supervisor_tick_s=0.01,
    )
    outcome = await driver.run()
    assert outcome.distinct_sessions_started == 2

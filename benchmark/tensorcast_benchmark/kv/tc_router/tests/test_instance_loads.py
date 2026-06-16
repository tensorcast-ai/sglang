"""Tests for router/instance_loads.py.

We use `aiohttp.test_utils.TestServer` to host a fake `/v1/loads` endpoint
on `localhost`, so the poller exercises the real HTTP path without needing
SGLang.
"""

from __future__ import annotations

import asyncio

import pytest
from aiohttp import web

from tensorcast_benchmark.kv.tc_router.router.instance_loads import (
    InstanceLoadPoller,
    _parse_loads_response,
)


# --- pure-helper test --------------------------------------------------------


def test_parse_loads_response_aggregates_per_dp_rank() -> None:
    payload = {
        "timestamp": "2026-01-01T00:00:00Z",
        "loads": [
            {"num_waiting_reqs": 1, "num_running_reqs": 3, "token_usage": 0.5, "utilization": 0.4, "gen_throughput": 100.0},
            {"num_waiting_reqs": 2, "num_running_reqs": 4, "token_usage": 0.7, "utilization": 0.6, "gen_throughput": 120.0},
        ],
    }
    s = _parse_loads_response("inst-0", payload)
    assert s.instance_id == "inst-0"
    assert s.num_waiting_reqs == 3
    assert s.num_running_reqs == 7
    assert s.queue_depth == 10
    assert s.token_usage == pytest.approx(0.6)
    assert s.utilization == pytest.approx(0.5)
    assert s.gen_throughput == pytest.approx(110.0)


def test_parse_loads_response_handles_empty_loads() -> None:
    s = _parse_loads_response("inst-0", {"loads": []})
    assert s.queue_depth == 0
    assert s.token_usage == 0.0


# --- poller integration via fake aiohttp server -----------------------------


class _FakeServer:
    """Hosts `/v1/loads` returning a dynamically updatable payload."""

    def __init__(self) -> None:
        self.payload = {"loads": []}
        self.calls = 0

    async def _handler(self, request: web.Request) -> web.Response:
        self.calls += 1
        return web.json_response(self.payload)

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/v1/loads", self._handler)
        return app


async def _start_fake(server: _FakeServer) -> tuple[web.AppRunner, str]:
    runner = web.AppRunner(server.app())
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    # Resolve the bound port.
    sockname = site._server.sockets[0].getsockname()  # type: ignore[attr-defined]
    base = f"http://127.0.0.1:{sockname[1]}"
    return runner, base


@pytest.mark.asyncio
async def test_poller_captures_initial_sample() -> None:
    fake = _FakeServer()
    fake.payload = {"loads": [{"num_waiting_reqs": 5, "num_running_reqs": 1}]}
    runner, base = await _start_fake(fake)

    poller = InstanceLoadPoller({"inst-0": base}, period_ms=50)
    try:
        await poller.start()
        # Wait for the first poll.
        for _ in range(50):  # up to ~1s
            if poller.get("inst-0") is not None:
                break
            await asyncio.sleep(0.02)
        sample = poller.get("inst-0")
        assert sample is not None
        assert sample.num_waiting_reqs == 5
        assert sample.num_running_reqs == 1
        assert sample.queue_depth == 6
    finally:
        await poller.stop()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_poller_observes_payload_change() -> None:
    fake = _FakeServer()
    fake.payload = {"loads": [{"num_waiting_reqs": 0, "num_running_reqs": 0}]}
    runner, base = await _start_fake(fake)

    poller = InstanceLoadPoller({"inst-0": base}, period_ms=50)
    try:
        await poller.start()
        # Wait for first sample
        for _ in range(50):
            if poller.get("inst-0") is not None:
                break
            await asyncio.sleep(0.02)
        first = poller.get("inst-0")
        assert first is not None
        assert first.queue_depth == 0

        # Update payload, wait for next poll cycle.
        fake.payload = {"loads": [{"num_waiting_reqs": 7, "num_running_reqs": 2}]}
        await asyncio.sleep(0.2)  # >= 4 poll periods
        second = poller.get("inst-0")
        assert second is not None
        assert second.queue_depth == 9
    finally:
        await poller.stop()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_poller_handles_failing_endpoint_silently() -> None:
    """Polling a closed port should not raise; the sample stays absent."""
    poller = InstanceLoadPoller(
        {"inst-down": "http://127.0.0.1:1"},  # port 1: nothing listening
        period_ms=50,
        request_timeout_s=0.2,
    )
    try:
        await poller.start()
        await asyncio.sleep(0.2)
        assert poller.get("inst-down") is None
    finally:
        await poller.stop()


def test_poller_rejects_empty_endpoint_map() -> None:
    with pytest.raises(ValueError, match="at least one"):
        InstanceLoadPoller({})


def test_poller_rejects_too_low_period() -> None:
    with pytest.raises(ValueError, match="period_ms"):
        InstanceLoadPoller({"x": "http://127.0.0.1"}, period_ms=1)


@pytest.mark.asyncio
async def test_poller_snapshot_is_a_copy() -> None:
    fake = _FakeServer()
    fake.payload = {"loads": [{"num_waiting_reqs": 1, "num_running_reqs": 1}]}
    runner, base = await _start_fake(fake)
    poller = InstanceLoadPoller({"inst-0": base}, period_ms=50)
    try:
        await poller.start()
        for _ in range(50):
            if poller.get("inst-0") is not None:
                break
            await asyncio.sleep(0.02)
        snap = poller.snapshot()
        snap.clear()  # mutating snapshot must not affect internal state
        assert poller.get("inst-0") is not None
    finally:
        await poller.stop()
        await runner.cleanup()

"""Tests for router/tc_router.py routing logic.

Use a `MockChatServer` (fake aiohttp app) so we don't need a real SGLang.
Tensorcast Runtime is mocked via monkeypatching `tc.connect`.
"""

from __future__ import annotations

import json
import sys
import types

import pytest
from aiohttp import web

from tensorcast_benchmark.kv.tc_router.router import tc_router as tc_router_module
from tensorcast_benchmark.kv.tc_router.router.migration import MigrationResult
from tensorcast_benchmark.kv.tc_router.router.policy import (
    MigrateOnceAfterTurnPolicy,
    _NeverRebalance,
)
from tensorcast_benchmark.kv.tc_router.router.tc_router import (
    TcRouter,
    TcRouterConfig,
)


class _FakeRuntime:
    closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_tensorcast(monkeypatch):
    """Stub `tensorcast.connect` to avoid needing a real daemon in tests."""
    fake_tc = types.ModuleType("tensorcast")
    instances: list[_FakeRuntime] = []

    def connect(*, daemon_address: str):
        r = _FakeRuntime()
        instances.append(r)
        return r

    fake_tc.connect = connect  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tensorcast", fake_tc)
    return instances


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


def test_format_migration_exception_expands_tensorcast_plan_failure() -> None:
    from tensorcast.api.operation import OperationError, OperationStatus
    from tensorcast.api.plan import PlanFailedError, PlanResult, PlanStepResult

    error = PlanFailedError(
        PlanResult(
            ok=False,
            request_id="req-1",
            steps={
                "step-1": PlanStepResult(
                    step_id="step-1",
                    target_id="10.0.10.58:62104",
                    action="publish",
                    status=OperationStatus(
                        state="failed",
                        message="publish failed",
                        error=OperationError(
                            status_code="NOT_FOUND",
                            message="request bundle not found",
                            retryable=False,
                        ),
                    ),
                )
            },
        )
    )

    message = tc_router_module._format_migration_exception(error)

    assert "PlanFailedError: Plan execution failed" in message
    assert "plan_request_id=req-1" in message
    assert "action=publish" in message
    assert "target=10.0.10.58:62104" in message
    assert "error_code=NOT_FOUND" in message
    assert "error_message=request bundle not found" in message


class _FakeChatServer:
    """Returns a small valid streaming chat completion + load endpoint."""

    def __init__(self, instance_label: str, *, cached_tokens: int = 0) -> None:
        self.instance_label = instance_label
        self.cached_tokens = cached_tokens
        self.requests: list[dict] = []
        self.routing_keys: list[str | None] = []
        self.loads_payload = {"loads": [{"num_waiting_reqs": 0, "num_running_reqs": 0}]}

    async def _chat_handler(self, request: web.Request) -> web.StreamResponse:
        self.requests.append(await request.json())
        self.routing_keys.append(request.headers.get("X-SMG-Routing-Key"))
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await resp.prepare(request)
        await resp.write(
            _sse(
                {
                    "choices": [{"delta": {"content": "hi"}, "index": 0}],
                    "usage": None,
                }
            )
        )
        await resp.write(
            _sse(
                {
                    "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
                    "usage": None,
                }
            )
        )
        await resp.write(
            _sse(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 1,
                        "total_tokens": 101,
                        "prompt_tokens_details": {"cached_tokens": self.cached_tokens},
                    },
                }
            )
        )
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    async def _loads_handler(self, request: web.Request) -> web.Response:
        return web.json_response(self.loads_payload)

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._chat_handler)
        app.router.add_get("/v1/loads", self._loads_handler)
        return app


async def _start_fake(server: _FakeChatServer) -> tuple[web.AppRunner, str]:
    runner = web.AppRunner(server.app())
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[attr-defined]
    return runner, f"http://127.0.0.1:{port}"


# --- core tests --------------------------------------------------------------


@pytest.mark.asyncio
async def test_tc_router_routes_same_session_to_same_home(fake_tensorcast) -> None:
    """First call assigns a home; every later call goes to the same instance."""
    s_a = _FakeChatServer("a")
    s_b = _FakeChatServer("b")
    runners = []
    try:
        ra, url_a = await _start_fake(s_a)
        rb, url_b = await _start_fake(s_b)
        runners += [ra, rb]

        endpoints = {"a:0": url_a, "b:0": url_b}
        router = TcRouter(
            TcRouterConfig(
                instance_endpoints=endpoints,
                default_model="model-x",
                daemon_address="127.0.0.1:1",
                load_polling_period_ms=50,
            ),
            policy=_NeverRebalance(seed=0),
        )
        await router.start()
        try:
            results = []
            for i in range(5):
                r = await router.generate(
                    rid=f"tcrouter:sess1:turn{i:03d}",
                    session_id="sess1",
                    messages=[{"role": "user", "content": "hi"}],
                    tools=None,
                    sampling_params={"max_tokens": 1, "temperature": 0.0},
                )
                results.append(r)
            # All 5 should go to the same served_instance.
            assert all(r.success for r in results)
            home = results[0].served_instance
            assert home in endpoints
            assert all(r.served_instance == home for r in results)
            # Exactly one server should have received all 5 requests.
            req_counts = {url_a: len(s_a.requests), url_b: len(s_b.requests)}
            assert sorted(req_counts.values()) == [0, 5]
        finally:
            await router.close()
    finally:
        for r in runners:
            await r.cleanup()


@pytest.mark.asyncio
async def test_tc_router_distinct_sessions_can_land_on_different_instances(
    fake_tensorcast,
) -> None:
    s_a = _FakeChatServer("a")
    s_b = _FakeChatServer("b")
    runners = []
    try:
        ra, url_a = await _start_fake(s_a)
        rb, url_b = await _start_fake(s_b)
        runners += [ra, rb]

        endpoints = {"a:0": url_a, "b:0": url_b}
        router = TcRouter(
            TcRouterConfig(
                instance_endpoints=endpoints,
                default_model="model-x",
                daemon_address="127.0.0.1:1",
                load_polling_period_ms=50,
            ),
            policy=_NeverRebalance(seed=42),
        )
        await router.start()
        try:
            homes = []
            for i in range(20):
                r = await router.generate(
                    rid=f"tcrouter:sess{i}:turn000",
                    session_id=f"sess{i}",
                    messages=[{"role": "user", "content": "hi"}],
                    tools=None,
                    sampling_params={"max_tokens": 1, "temperature": 0.0},
                )
                homes.append(r.served_instance)
            # With no observed load, power-of-two falls back to roughly uniform
            # — both endpoints should see at least one session.
            assert len(set(homes)) == 2
        finally:
            await router.close()
    finally:
        for r in runners:
            await r.cleanup()


@pytest.mark.asyncio
async def test_tc_router_close_releases_runtime(fake_tensorcast) -> None:
    s_a = _FakeChatServer("a")
    runner, url = await _start_fake(s_a)
    try:
        router = TcRouter(
            TcRouterConfig(
                instance_endpoints={"a:0": url},
                default_model="m",
                daemon_address="127.0.0.1:1",
                load_polling_period_ms=50,
            ),
            policy=_NeverRebalance(),
        )
        await router.start()
        assert len(fake_tensorcast) == 1
        assert fake_tensorcast[0].closed is False
        await router.close()
        assert fake_tensorcast[0].closed is True
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_tc_router_records_turn_count(fake_tensorcast) -> None:
    s_a = _FakeChatServer("a")
    runner, url = await _start_fake(s_a)
    try:
        router = TcRouter(
            TcRouterConfig(
                instance_endpoints={"a:0": url},
                default_model="m",
                daemon_address="127.0.0.1:1",
                load_polling_period_ms=50,
            ),
            policy=_NeverRebalance(),
        )
        await router.start()
        try:
            for i in range(3):
                r = await router.generate(
                    rid=f"tcrouter:sx:turn{i:03d}",
                    session_id="sx",
                    messages=[{"role": "user", "content": "hi"}],
                    tools=None,
                    sampling_params={"max_tokens": 1, "temperature": 0.0},
                )
                assert r.success
            snap = router.session_state_snapshot()
            assert snap["sx"].turn_count == 3
            assert snap["sx"].home_instance == "a:0"
            assert snap["sx"].last_engine_request_id == "tcrouter:sx:turn002"
            assert [request["rid"] for request in s_a.requests] == [
                "tcrouter:sx:turn000",
                "tcrouter:sx:turn001",
                "tcrouter:sx:turn002",
            ]
            assert s_a.routing_keys == ["sx", "sx", "sx"]
        finally:
            await router.close()
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_tc_router_migrates_session_and_marks_next_turn(
    fake_tensorcast,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMigrationClient:
        instances: list["FakeMigrationClient"] = []

        def __init__(self, runtime) -> None:
            del runtime
            self.calls: list[dict] = []
            self.instances.append(self)

        def migrate(self, **kwargs) -> MigrationResult:
            self.calls.append(dict(kwargs))
            return MigrationResult(
                publish_latency_ms=1.0,
                hydrate_latency_ms=2.0,
                publish_manifest_digest="publish-digest",
                artifact_manifest_digest="artifact-digest",
                published_cutoff_token_count=100,
                tail_valid_tokens=0,
            )

    monkeypatch.setattr(
        tc_router_module,
        "TensorcastMigrationClient",
        FakeMigrationClient,
    )

    s_a = _FakeChatServer("a", cached_tokens=10)
    s_b = _FakeChatServer("b", cached_tokens=10)
    runners = []
    try:
        ra, url_a = await _start_fake(s_a)
        rb, url_b = await _start_fake(s_b)
        runners += [ra, rb]

        endpoints = {"a:0": url_a, "b:0": url_b}
        router = TcRouter(
            TcRouterConfig(
                instance_endpoints=endpoints,
                default_model="model-x",
                daemon_address="127.0.0.1:1",
                load_polling_period_ms=50,
            ),
            policy=MigrateOnceAfterTurnPolicy(
                seed=0,
                after_turn_count=1,
                target_strategy="next_instance",
                pending_migration_wait_timeout_s=1.0,
            ),
        )
        await router.start()
        try:
            first = await router.generate(
                rid="tcrouter:sess1:turn000",
                session_id="sess1",
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
                sampling_params={"max_tokens": 1, "temperature": 0.0},
            )
            second = await router.generate(
                rid="tcrouter:sess1:turn001",
                session_id="sess1",
                messages=[{"role": "user", "content": "again"}],
                tools=None,
                sampling_params={"max_tokens": 1, "temperature": 0.0},
            )

            assert first.success
            assert second.success
            assert first.served_instance in endpoints
            assert second.served_instance in endpoints
            assert second.served_instance != first.served_instance
            assert second.was_just_migrated is True
            assert second.used_hydrated_bundle is True

            client = FakeMigrationClient.instances[0]
            assert len(client.calls) == 1
            assert client.calls[0]["source_instance_id"] == first.served_instance
            assert client.calls[0]["target_instance_id"] == second.served_instance
            assert client.calls[0]["engine_request_id"] == "tcrouter:sess1:turn000"

            snap = router.session_state_snapshot()
            assert snap["sess1"].home_instance == second.served_instance
            assert snap["sess1"].migration_attempt_count == 1
        finally:
            await router.close()
    finally:
        for r in runners:
            await r.cleanup()


def test_tc_router_rejects_empty_endpoints() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        TcRouter(
            TcRouterConfig(
                instance_endpoints={},
                default_model="m",
                daemon_address="x:1",
            ),
            policy=_NeverRebalance(),
        )

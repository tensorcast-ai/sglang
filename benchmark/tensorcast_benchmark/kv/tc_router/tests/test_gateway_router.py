"""Tests for router/gateway_router.py.

Use a fake aiohttp server emitting an OpenAI-compatible streaming
chat-completions response so we exercise the real HTTP + SSE parsing
path without needing a live SGLang.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp import web

from tensorcast_benchmark.kv.tc_router.router.gateway_router import (
    GatewayRouter,
    _extract_cached_tokens,
)


# --- _extract_cached_tokens -------------------------------------------------


def test_extract_cached_tokens_openai_canonical() -> None:
    assert _extract_cached_tokens(
        {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 7}}
    ) == 7


def test_extract_cached_tokens_sglang_flat() -> None:
    assert _extract_cached_tokens({"cached_tokens": 5}) == 5


def test_extract_cached_tokens_meta_info_nested() -> None:
    assert _extract_cached_tokens({"meta_info": {"cached_tokens": 9}}) == 9


def test_extract_cached_tokens_absent() -> None:
    assert _extract_cached_tokens({"prompt_tokens": 100}) == 0


def test_extract_cached_tokens_handles_non_dict() -> None:
    assert _extract_cached_tokens("oops") == 0  # type: ignore[arg-type]


# --- streaming response integration ----------------------------------------


def _sse_chunk(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


class _FakeChatServer:
    """Streams an OpenAI-format response with two content chunks + final usage."""

    def __init__(self, prompt_tokens: int, cached_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.cached_tokens = cached_tokens
        self.last_request: dict | None = None

    async def _handler(self, request: web.Request) -> web.StreamResponse:
        self.last_request = await request.json()
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "x-served-by": "10.0.0.1:30001",
            },
        )
        await resp.prepare(request)

        # First content delta
        await resp.write(_sse_chunk({
            "choices": [{"delta": {"content": "Hello"}, "index": 0, "finish_reason": None}],
            "usage": None,
        }))
        await asyncio.sleep(0.01)
        # Second content delta
        await resp.write(_sse_chunk({
            "choices": [{"delta": {"content": " world"}, "index": 0, "finish_reason": None}],
            "usage": None,
        }))
        # Final stop chunk
        await resp.write(_sse_chunk({
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
            "usage": None,
        }))
        # Final usage chunk
        await resp.write(_sse_chunk({
            "choices": [],
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": 2,
                "total_tokens": self.prompt_tokens + 2,
                "prompt_tokens_details": {"cached_tokens": self.cached_tokens},
            },
        }))
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._handler)
        return app


async def _start_fake(server: _FakeChatServer) -> tuple[web.AppRunner, str]:
    runner = web.AppRunner(server.app())
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockname = site._server.sockets[0].getsockname()  # type: ignore[attr-defined]
    base = f"http://127.0.0.1:{sockname[1]}"
    return runner, base


@pytest.mark.asyncio
async def test_gateway_router_parses_stream_and_usage() -> None:
    fake = _FakeChatServer(prompt_tokens=1234, cached_tokens=900)
    runner, base = await _start_fake(fake)
    router = GatewayRouter(base, default_model="Qwen3-32B")
    try:
        result = await router.generate(
            session_id="s1",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            sampling_params={"max_tokens": 32, "temperature": 0.0},
        )
        assert result.success is True
        assert result.text == "Hello world"
        assert result.ttft_ms is not None and result.ttft_ms > 0
        assert result.latency_ms is not None and result.latency_ms >= result.ttft_ms
        assert result.prompt_tokens == 1234
        assert result.cached_tokens == 900
        assert result.served_instance == "10.0.0.1:30001"
    finally:
        await router.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_gateway_router_includes_tools_in_request() -> None:
    fake = _FakeChatServer(prompt_tokens=10, cached_tokens=0)
    runner, base = await _start_fake(fake)
    router = GatewayRouter(base, default_model="m")
    try:
        await router.generate(
            session_id="s",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
            sampling_params={"max_tokens": 8, "temperature": 0.0},
        )
        assert fake.last_request is not None
        assert fake.last_request["stream"] is True
        assert fake.last_request["stream_options"] == {"include_usage": True}
        assert fake.last_request["tools"][0]["function"]["name"] == "f"
    finally:
        await router.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_gateway_router_returns_failure_on_http_error() -> None:
    async def err_handler(request: web.Request) -> web.Response:
        return web.Response(status=503, text="overloaded")

    app = web.Application()
    app.router.add_post("/v1/chat/completions", err_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    base = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"  # type: ignore[attr-defined]

    router = GatewayRouter(base, default_model="m")
    try:
        result = await router.generate(
            session_id="s",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            sampling_params={"max_tokens": 1, "temperature": 0.0},
        )
        assert result.success is False
        assert "503" in result.error_message
    finally:
        await router.close()
        await runner.cleanup()

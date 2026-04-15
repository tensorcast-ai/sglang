from __future__ import annotations

import time

import pytest

from tensorcast_benchmark.kv.sgl_client import SGLangClient


class _FakeSession:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


@pytest.mark.asyncio
async def test_keepalive_refresh_skips_active_streams() -> None:
    client = SGLangClient("http://127.0.0.1:1")
    session = _FakeSession()
    client._session = session  # type: ignore[assignment]
    client._owns_session = True
    client._last_request_completed_monotonic = time.monotonic() - 10.0
    client._active_request_count = 1

    await client._ensure_session(refresh_if_idle=True)

    assert client._session is session
    assert session.close_calls == 0

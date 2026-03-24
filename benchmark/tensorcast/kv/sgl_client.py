from __future__ import annotations

import asyncio
import json
import time
from contextlib import suppress
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import aiohttp
from pydantic import BaseModel, ConfigDict, Field

from tensorcast.kv.models import GenerateMetrics


class SGLangHTTPError(RuntimeError):
    """HTTP error from an SGLang serving instance."""


class _GenerateChunk(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = ""
    meta_info: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class StreamGenerateResult:
    """Measured streaming generate result with TTFT and full latency."""

    text: str
    meta_info: dict[str, Any]
    ttft_ms: float | None
    latency_ms: float

    def _abort_message(self) -> str:
        finish_reason = self.meta_info.get("finish_reason")
        if isinstance(finish_reason, dict) and finish_reason.get("type") == "abort":
            message = finish_reason.get("message")
            return str(message) if message else "request aborted"
        return ""

    def to_metrics(self) -> GenerateMetrics:
        abort_message = self._abort_message()
        success = bool(self.text) and not abort_message
        error_message = ""
        if abort_message:
            error_message = abort_message
        elif not self.text:
            error_message = "empty generation output"
        return GenerateMetrics(
            success=success,
            text=self.text,
            ttft_ms=self.ttft_ms,
            latency_ms=self.latency_ms,
            meta_info=self.meta_info,
            error_message=error_message,
        )


class SGLangClient:
    """Thin async client for SGLang's native `/generate` endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        request_timeout_seconds: float = 120.0,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=request_timeout_seconds)
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self) -> "SGLangClient":
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
            self._owns_session = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        if self._session is not None and self._owns_session:
            await self._session.close()
        self._session = None

    async def health(self) -> bool:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
            self._owns_session = True
        url = f"{self._base_url}/health"
        async with self._session.get(url) as response:
            return response.status == 200

    async def wait_ready(
        self,
        *,
        timeout_seconds: float = 1000.0,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            with suppress(Exception):
                if await self.health():
                    return
            await asyncio.sleep(poll_interval_seconds)
        raise TimeoutError(f"Timed out waiting for {self._base_url}/health")

    async def generate_stream(
        self,
        text: str,
        *,
        sampling_params: dict[str, Any],
        extra_body: dict[str, Any] | None = None,
    ) -> StreamGenerateResult:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
            self._owns_session = True

        payload: dict[str, Any] = {
            "text": text,
            "sampling_params": sampling_params,
            "stream": True,
        }
        if extra_body:
            payload.update(extra_body)

        start = time.perf_counter()
        first_token_ms: float | None = None
        last_text = ""
        final_chunk: _GenerateChunk | None = None
        url = f"{self._base_url}/generate"
        async with self._session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise SGLangHTTPError(
                    f"POST {url} failed: {response.status} {error_text}"
                )
            async for chunk_bytes in response.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue
                chunk = chunk_bytes.decode("utf-8")
                if not chunk.startswith("data:"):
                    continue
                body = chunk[5:].strip()
                if body == "[DONE]":
                    break
                parsed = _GenerateChunk.model_validate(json.loads(body))
                if parsed.text and first_token_ms is None and parsed.text != last_text:
                    first_token_ms = (time.perf_counter() - start) * 1000.0
                last_text = parsed.text
                final_chunk = parsed

        latency_ms = (time.perf_counter() - start) * 1000.0
        if final_chunk is None:
            return StreamGenerateResult(
                text="",
                meta_info={},
                ttft_ms=None,
                latency_ms=latency_ms,
            )
        return StreamGenerateResult(
            text=final_chunk.text,
            meta_info=final_chunk.meta_info,
            ttft_ms=first_token_ms,
            latency_ms=latency_ms,
        )

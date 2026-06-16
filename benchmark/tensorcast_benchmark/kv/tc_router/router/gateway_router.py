"""Gateway-baseline `Router` implementation.

Forwards `Router.generate(...)` to the SGL model gateway via streaming
`/v1/chat/completions`. Measures TTFT at the first non-empty content
delta, latency at the [DONE] line, and reads `prompt_tokens` /
`cached_tokens` from the `usage` field of the final pre-DONE chunk.

Per arch § 5.2.3 we deliberately do NOT enable any tool-call parsing on
the SGLang side. The gateway forwards `tools` verbatim, so the prefix
shape is realistic; we discard whatever the model emits.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

import aiohttp

from .interface import GenerateResult


logger = logging.getLogger(__name__)


class GatewayRouter:
    """Talks to a sgl-model-gateway over OpenAI-compatible HTTP."""

    def __init__(
        self,
        gateway_url: str,
        *,
        default_model: str,
        request_timeout_s: float = 600.0,
    ) -> None:
        self._url = gateway_url.rstrip("/")
        self._model = default_model
        self._timeout = aiohttp.ClientTimeout(total=request_timeout_s)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self._timeout, trust_env=False
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def generate(
        self,
        *,
        session_id: str,
        messages: list[dict],
        tools: Optional[list[dict]],
        sampling_params: dict,
    ) -> GenerateResult:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            # Ask SGLang to include final usage chunk in the SSE stream.
            "stream_options": {"include_usage": True},
            "max_tokens": int(sampling_params.get("max_tokens", 64)),
            "temperature": float(sampling_params.get("temperature", 0.0)),
        }
        if tools:
            body["tools"] = tools

        url = f"{self._url}/v1/chat/completions"
        session = await self._ensure_session()

        t0 = time.monotonic()
        first_token_ms: Optional[float] = None
        text_buf: list[str] = []
        usage: dict = {}
        served_instance = ""

        try:
            async with session.post(url, json=body, proxy=None) as resp:
                if resp.status != 200:
                    err_text = await resp.text()
                    return GenerateResult(
                        success=False,
                        error_message=f"HTTP {resp.status}: {err_text[:300]}",
                        latency_ms=(time.monotonic() - t0) * 1000.0,
                    )
                # SGLang gateway propagates the upstream worker's identifier
                # via headers; field name varies by version.
                served_instance = (
                    resp.headers.get("x-served-by")
                    or resp.headers.get("x-routed-by")
                    or resp.headers.get("x-router-target")
                    or ""
                )
                async for raw_line in resp.content:
                    line = raw_line.strip()
                    if not line:
                        continue
                    if not line.startswith(b"data:"):
                        continue
                    body_b = line[5:].strip()
                    if body_b == b"[DONE]":
                        break
                    try:
                        chunk = json.loads(body_b)
                    except json.JSONDecodeError:
                        continue

                    # Capture usage when present (typically in the final
                    # pre-DONE chunk if `stream_options.include_usage=True`).
                    chunk_usage = chunk.get("usage")
                    if chunk_usage:
                        usage = chunk_usage

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        if first_token_ms is None:
                            first_token_ms = (time.monotonic() - t0) * 1000.0
                        text_buf.append(content)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            return GenerateResult(
                success=False,
                error_message=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.monotonic() - t0) * 1000.0,
            )

        latency_ms = (time.monotonic() - t0) * 1000.0
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        cached_tokens = _extract_cached_tokens(usage)

        return GenerateResult(
            text="".join(text_buf),
            ttft_ms=first_token_ms,
            latency_ms=latency_ms,
            served_instance=served_instance,
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            raw_meta_info=usage,
            success=True,
        )


def _extract_cached_tokens(usage: dict) -> int:
    """Pull `cached_tokens` from a usage dict.

    Tries several field shapes, since SGLang/OpenAI compat layers vary:
      - OpenAI canonical: `usage.prompt_tokens_details.cached_tokens`
      - SGLang older: `usage.cached_tokens`
      - SGLang `meta_info` bag: `usage.meta_info.cached_tokens` / nested
    """
    if not isinstance(usage, dict):
        return 0
    for getter in (
        lambda u: (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
        lambda u: u.get("cached_tokens"),
        lambda u: (u.get("meta_info") or {}).get("cached_tokens"),
    ):
        try:
            v = getter(usage)
        except (AttributeError, TypeError):
            v = None
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
    return 0

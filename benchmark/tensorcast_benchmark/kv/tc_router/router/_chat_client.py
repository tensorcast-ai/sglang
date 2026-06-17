"""Shared streaming `/v1/chat/completions` client.

Both `GatewayRouter` (Phase 5) and `TcRouter` (Phase 7) post requests to a
SGLang-backed OpenAI-compat HTTP endpoint and need identical
SSE-streaming + usage-extraction logic. The difference is only the URL
(gateway proxy vs. direct instance).
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

import aiohttp

from .interface import GenerateResult


def build_chat_completions_body(
    *,
    model: str,
    messages: list[dict],
    tools: Optional[list[dict]],
    sampling_params: dict,
) -> dict:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        # Required so SGLang puts the final `usage` chunk in the SSE stream
        # AND so we can read prompt_tokens / cached_tokens. Combined with
        # the `--enable-cache-report` server flag (per services/sglang.py)
        # this puts `usage.prompt_tokens_details.cached_tokens` on the wire.
        "stream_options": {"include_usage": True},
        "max_tokens": int(sampling_params.get("max_tokens", 64)),
        "temperature": float(sampling_params.get("temperature", 0.0)),
    }
    if tools:
        body["tools"] = tools
    return body


def extract_cached_tokens(usage: dict) -> int:
    """Pull `cached_tokens` from a usage dict, tolerating field-name variants."""
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


async def chat_completion_stream(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    body: dict,
) -> GenerateResult:
    """POST `body` to `<base_url>/v1/chat/completions` (streaming) and return GenerateResult.

    Discards content-by-policy except for the concatenated `text` field —
    callers (faithful replay) use only TTFT / latency / token counts.
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
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
            served_instance = (
                resp.headers.get("x-served-by")
                or resp.headers.get("x-routed-by")
                or resp.headers.get("x-router-target")
                or ""
            )
            async for raw_line in resp.content:
                line = raw_line.strip()
                if not line or not line.startswith(b"data:"):
                    continue
                body_b = line[5:].strip()
                if body_b == b"[DONE]":
                    break
                try:
                    chunk = json.loads(body_b)
                except json.JSONDecodeError:
                    continue
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

    return GenerateResult(
        text="".join(text_buf),
        ttft_ms=first_token_ms,
        latency_ms=(time.monotonic() - t0) * 1000.0,
        served_instance=served_instance,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        cached_tokens=extract_cached_tokens(usage),
        raw_meta_info=usage,
        success=True,
    )

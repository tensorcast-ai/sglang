"""Gateway-baseline `Router` implementation.

Forwards `Router.generate(...)` to a sgl-model-gateway via streaming
`/v1/chat/completions`. Per arch § 5.2.3 we do NOT enable any tool-call
parsing on the SGLang side; tools are forwarded verbatim and the model's
output is discarded by the workload driver.
"""

from __future__ import annotations

from typing import Optional

import aiohttp

from ._chat_client import build_chat_completions_body, chat_completion_stream
# Re-exported for backward-compat with tests written against gateway_router.
from ._chat_client import extract_cached_tokens as _extract_cached_tokens  # noqa: F401
from .interface import GenerateResult


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
        body = build_chat_completions_body(
            model=self._model,
            messages=messages,
            tools=tools,
            sampling_params=sampling_params,
        )
        session = await self._ensure_session()
        return await chat_completion_stream(session, base_url=self._url, body=body)

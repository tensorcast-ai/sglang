"""TcRouter — Phase 7 stub.

Maintains:
  - a `session_id -> SessionState` map
  - an `InstanceLoadPoller` for the per-instance `/v1/loads` stream
  - a Tensorcast `Runtime` connected to the daemon (proves wiring; with
    `_NeverRebalance` no plans are issued, so the runtime stays idle)

Routing rule:
  - First request for a session: ask `policy.pick_session_for_initial_home`
    using the current load snapshot, then stick the session to that
    `home_instance` for its lifetime.
  - Subsequent requests: route to `home_instance`. With `_NeverRebalance`
    the home is never changed.

Per the plan, Phase 7's success criterion is that `tc_router` with a
NeverRebalance stub behaves identically to `gw_load_aware` end-to-end.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Mapping, Optional

import aiohttp

from ._chat_client import build_chat_completions_body, chat_completion_stream
from .instance_loads import InstanceLoadPoller
from .interface import GenerateResult
from .policy import Policy
from .state import InstanceId, SessionId, SessionState


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TcRouterConfig:
    """Inputs to TcRouter that aren't already on the Policy."""

    instance_endpoints: dict[InstanceId, str]  # {instance_id: serving_http_url}
    default_model: str
    daemon_address: str  # "<host>:<port>" for `tc.connect`
    request_timeout_s: float = 600.0
    load_polling_period_ms: int = 250


class TcRouter:
    """Tensorcast-backed `Router` implementation.

    Phase 7 stub: holds a Tensorcast Runtime but never issues plans.
    """

    def __init__(self, config: TcRouterConfig, *, policy: Policy) -> None:
        if not config.instance_endpoints:
            raise ValueError("TcRouterConfig.instance_endpoints must be non-empty")
        self._config = config
        self._policy = policy

        self._instance_ids: list[InstanceId] = list(config.instance_endpoints.keys())
        self._endpoints: dict[InstanceId, str] = dict(config.instance_endpoints)

        # State protected by `_state_lock` because the workload driver runs
        # many concurrent sessions; supervisor + per-session coroutines all
        # touch `_session_state`.
        self._session_state: dict[SessionId, SessionState] = {}
        self._state_lock = asyncio.Lock()

        # Background load poller.
        self._load_poller = InstanceLoadPoller(
            self._endpoints, period_ms=config.load_polling_period_ms
        )

        # HTTP session reused across requests.
        self._timeout = aiohttp.ClientTimeout(total=config.request_timeout_s)
        self._http_session: Optional[aiohttp.ClientSession] = None

        # Tensorcast Runtime — created in `start()` to avoid event-loop
        # binding at __init__ time. Phase 7 holds it idle.
        self._runtime: object | None = None

    # --- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        await self._load_poller.start()
        # Connect to the Tensorcast daemon. This is sync in the SDK, run
        # in a thread executor so we don't block the asyncio loop.
        loop = asyncio.get_event_loop()
        self._runtime = await loop.run_in_executor(None, self._connect_runtime)
        logger.info(
            "tc_router connected to tensorcast daemon at %s",
            self._config.daemon_address,
        )

    def _connect_runtime(self) -> object:
        # Imported lazily so unit tests don't require the SDK.
        import tensorcast as tc

        return tc.connect(daemon_address=self._config.daemon_address)

    async def close(self) -> None:
        # Stop background poller first.
        try:
            await self._load_poller.stop()
        except Exception:  # noqa: BLE001
            logger.exception("instance load poller stop failed")
        # Close HTTP session.
        if self._http_session is not None and not self._http_session.closed:
            try:
                await self._http_session.close()
            except Exception:  # noqa: BLE001
                logger.exception("http session close failed")
            self._http_session = None
        # Close Tensorcast runtime.
        if self._runtime is not None:
            try:
                loop = asyncio.get_event_loop()
                runtime = self._runtime
                self._runtime = None
                await loop.run_in_executor(None, runtime.close)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                logger.exception("tensorcast runtime close failed")

    # --- routing -----------------------------------------------------------

    async def _ensure_http(self) -> aiohttp.ClientSession:
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession(
                timeout=self._timeout, trust_env=False
            )
        return self._http_session

    async def _resolve_home(self, session_id: SessionId) -> InstanceId:
        async with self._state_lock:
            existing = self._session_state.get(session_id)
            if existing is not None:
                return existing.home_instance

            chosen = self._policy.pick_session_for_initial_home(
                session_id=session_id,
                candidates=self._instance_ids,
                loads=self._load_poller.snapshot(),
            )
            self._session_state[session_id] = SessionState(
                session_id=session_id,
                home_instance=chosen,
                last_active_ts=time.monotonic(),
                turn_count=0,
            )
            return chosen

    async def generate(
        self,
        *,
        session_id: str,
        messages: list[dict],
        tools: Optional[list[dict]],
        sampling_params: dict,
    ) -> GenerateResult:
        home_instance = await self._resolve_home(session_id)
        base_url = self._endpoints[home_instance]

        body = build_chat_completions_body(
            model=self._config.default_model,
            messages=messages,
            tools=tools,
            sampling_params=sampling_params,
        )
        session = await self._ensure_http()
        result = await chat_completion_stream(session, base_url=base_url, body=body)

        # Always tag served_instance with our home_instance — gateway
        # baselines may omit the upstream header, but tc_router knows
        # exactly where it sent the request.
        result.served_instance = home_instance

        # Update session state (only on success — failed turns shouldn't
        # bump the activity clock as if a real call landed).
        if result.success:
            async with self._state_lock:
                state = self._session_state.get(session_id)
                if state is not None:
                    state.last_active_ts = time.monotonic()
                    state.turn_count += 1
                    state.last_prompt_tokens = result.prompt_tokens
                    state.last_engine_request_id = (
                        f"tcrouter:{session_id}:turn{state.turn_count - 1:03d}"
                    )
        return result

    # --- introspection (handy for tests / driver logging) ------------------

    def session_state_snapshot(self) -> Mapping[SessionId, SessionState]:
        return dict(self._session_state)

    @property
    def policy(self) -> Policy:
        return self._policy

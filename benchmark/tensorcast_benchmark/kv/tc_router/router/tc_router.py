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
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import aiohttp

from ..metrics.migrations import MigrationRecordSink
from ._chat_client import build_chat_completions_body, chat_completion_stream
from .instance_loads import InstanceLoadPoller
from .interface import GenerateResult
from .migration import TensorcastMigrationClient
from .policy import Policy
from .state import (
    InstanceId,
    LoadSample,
    MigrationDecision,
    MigrationFuture,
    SessionId,
    SessionState,
)


logger = logging.getLogger(__name__)


def _format_migration_exception(exc: Exception) -> str:
    base_message = f"{type(exc).__name__}: {exc}"
    try:
        from tensorcast.api.plan import PlanFailedError
    except Exception:  # noqa: BLE001
        return base_message

    if not isinstance(exc, PlanFailedError):
        return base_message

    step_messages: list[str] = []
    for step in exc.result.steps.values():
        status = step.status
        pieces = [
            f"step_id={step.step_id}",
            f"action={step.action}",
            f"target={step.target_id}",
            f"state={status.state}",
        ]
        if status.message:
            pieces.append(f"message={status.message}")
        if status.error is not None:
            pieces.extend(
                [
                    f"error_code={status.error.status_code}",
                    f"error_message={status.error.message}",
                ]
            )
        step_messages.append("{" + ", ".join(pieces) + "}")

    plan_request_id = exc.result.request_id
    if not step_messages:
        return f"{base_message}; plan_request_id={plan_request_id}; plan_steps=[]"
    return (
        f"{base_message}; plan_request_id={plan_request_id}; "
        f"plan_steps=[{'; '.join(step_messages)}]"
    )


@dataclass(frozen=True)
class TcRouterConfig:
    """Inputs to TcRouter that aren't already on the Policy."""

    instance_endpoints: dict[InstanceId, str]  # {instance_id: serving_http_url}
    default_model: str
    daemon_address: str  # "<host>:<port>" for `tc.connect`
    request_timeout_s: float = 600.0
    load_polling_period_ms: int = 250
    migrations_path: str | None = None


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
        # binding at __init__ time.
        self._runtime: object | None = None
        self._migration_client: TensorcastMigrationClient | None = None
        self._migration_tasks: set[asyncio.Task[None]] = set()
        self._migration_sink = (
            MigrationRecordSink(Path(config.migrations_path))
            if config.migrations_path
            else None
        )

    # --- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        await self._load_poller.start()
        # Connect to the Tensorcast daemon. This is sync in the SDK, run
        # in a thread executor so we don't block the asyncio loop.
        loop = asyncio.get_event_loop()
        self._runtime = await loop.run_in_executor(None, self._connect_runtime)
        self._migration_client = TensorcastMigrationClient(self._runtime)
        logger.info(
            "tc_router connected to tensorcast daemon at %s",
            self._config.daemon_address,
        )

    def _connect_runtime(self) -> object:
        # Imported lazily so unit tests don't require the SDK.
        import tensorcast as tc

        return tc.connect(daemon_address=self._config.daemon_address)

    async def close(self) -> None:
        try:
            await self.finalize_migrations()
        except Exception:  # noqa: BLE001
            logger.exception("migration finalization failed")
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

    async def _wait_for_pending_migration(self, session_id: SessionId) -> None:
        async with self._state_lock:
            state = self._session_state.get(session_id)
            future = state.pending_migration if state is not None else None
        if future is None:
            return
        timeout_s = self._policy.pending_migration_wait_timeout_s
        if timeout_s <= 0:
            return
        try:
            await asyncio.wait_for(future.completion_event.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            logger.warning(
                "pending migration wait timed out session=%s source=%s target=%s",
                session_id,
                future.source_instance,
                future.target_instance,
            )

    async def _maybe_schedule_migrations(self) -> None:
        now_ts = time.monotonic()
        loads = dict(self._load_poller.snapshot())
        for instance_id in self._instance_ids:
            if instance_id not in loads:
                loads[instance_id] = LoadSample(
                    instance_id=instance_id,
                    num_waiting_reqs=0,
                    num_running_reqs=0,
                    token_usage=0.0,
                    utilization=0.0,
                    gen_throughput=0.0,
                    timestamp_monotonic=now_ts,
                )
        async with self._state_lock:
            sessions = dict(self._session_state)
        decisions = self._policy.should_rebalance(
            loads=loads,
            sessions=sessions,
            now_ts=now_ts,
        )
        for decision in decisions:
            await self._schedule_migration(decision)

    async def _schedule_migration(self, decision: MigrationDecision) -> None:
        async with self._state_lock:
            state = self._session_state.get(decision.session_id)
            if state is None:
                return
            if state.pending_migration is not None:
                return
            if state.home_instance != decision.source_instance:
                return
            if not state.last_engine_request_id:
                return
            migration_number = state.migration_attempt_count + 1
            migration_id = f"migrate:{decision.session_id}:{migration_number:06d}"
            source_engine_request_id = state.last_engine_request_id
            future = MigrationFuture(
                session_id=decision.session_id,
                source_instance=decision.source_instance,
                target_instance=decision.target_instance,
                started_monotonic=time.monotonic(),
            )
            state.pending_migration = future
            state.migration_attempt_count = migration_number
            state.last_migration_id = migration_id
            state.last_migration_source_instance = decision.source_instance
            state.last_migration_target_instance = decision.target_instance

        if self._migration_sink is not None:
            self._migration_sink.start(
                migration_id=migration_id,
                session_id=decision.session_id,
                source_instance=decision.source_instance,
                target_instance=decision.target_instance,
                source_engine_request_id=source_engine_request_id,
                decided_by=decision.decided_by,
            )

        task = asyncio.create_task(
            self._run_migration(
                decision=decision,
                migration_id=migration_id,
                source_engine_request_id=source_engine_request_id,
                future=future,
            )
        )
        self._migration_tasks.add(task)
        task.add_done_callback(self._migration_tasks.discard)

    async def _run_migration(
        self,
        *,
        decision: MigrationDecision,
        migration_id: str,
        source_engine_request_id: str,
        future: MigrationFuture,
    ) -> None:
        client = self._migration_client
        if client is None:
            await self._mark_migration_failed(
                decision=decision,
                migration_id=migration_id,
                future=future,
                error_message="Tensorcast migration client is not initialized",
                status="publish_failed",
            )
            return
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: client.migrate(
                    migration_id=migration_id,
                    source_instance_id=decision.source_instance,
                    target_instance_id=decision.target_instance,
                    engine_request_id=source_engine_request_id,
                    deadline_ms=self._policy.plan_deadline_ms,
                    publish_ttl_ms=self._policy.publish_ttl_ms,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            error_message = _format_migration_exception(exc)
            status = (
                "hydrate_failed" if "hydrate" in str(exc).lower() else "publish_failed"
            )
            await self._mark_migration_failed(
                decision=decision,
                migration_id=migration_id,
                future=future,
                error_message=error_message,
                status=status,
            )
            return

        async with self._state_lock:
            state = self._session_state.get(decision.session_id)
            if state is not None and state.pending_migration is future:
                state.home_instance = decision.target_instance
                state.last_published_manifest = result.publish_manifest_digest
                state.pending_consumed_migration_id = migration_id
                state.pending_migration = None
                state.last_migration_completed_monotonic = time.monotonic()
        if self._migration_sink is not None:
            self._migration_sink.mark_success(
                migration_id=migration_id,
                publish_latency_ms=result.publish_latency_ms,
                hydrate_latency_ms=result.hydrate_latency_ms,
                publish_manifest_digest=result.publish_manifest_digest,
                artifact_manifest_digest=result.artifact_manifest_digest,
                published_cutoff_token_count=result.published_cutoff_token_count,
                tail_valid_tokens=result.tail_valid_tokens,
            )
        future.mark_success()

    async def _mark_migration_failed(
        self,
        *,
        decision: MigrationDecision,
        migration_id: str,
        future: MigrationFuture,
        error_message: str,
        status: str,
    ) -> None:
        async with self._state_lock:
            state = self._session_state.get(decision.session_id)
            if state is not None and state.pending_migration is future:
                state.pending_migration = None
        if self._migration_sink is not None:
            self._migration_sink.mark_failure(
                migration_id=migration_id,
                status=status,
                error_message=error_message,
            )
        future.mark_failure(error_message)

    async def generate(
        self,
        *,
        rid: str,
        session_id: str,
        messages: list[dict],
        tools: Optional[list[dict]],
        sampling_params: dict,
    ) -> GenerateResult:
        await self._wait_for_pending_migration(session_id)
        home_instance = await self._resolve_home(session_id)
        base_url = self._endpoints[home_instance]

        body = build_chat_completions_body(
            model=self._config.default_model,
            rid=rid,
            messages=messages,
            tools=tools,
            sampling_params=sampling_params,
        )
        session = await self._ensure_http()
        result = await chat_completion_stream(
            session,
            base_url=base_url,
            body=body,
            headers={"X-SMG-Routing-Key": session_id},
        )

        # Always tag served_instance with our home_instance — gateway
        # baselines may omit the upstream header, but tc_router knows
        # exactly where it sent the request.
        result.served_instance = home_instance

        # Update session state (only on success — failed turns shouldn't
        # bump the activity clock as if a real call landed).
        if result.success:
            consumed_migration_id = ""
            consumed_within_s = 0.0
            async with self._state_lock:
                state = self._session_state.get(session_id)
                if state is not None:
                    state.last_active_ts = time.monotonic()
                    state.turn_count += 1
                    state.last_prompt_tokens = result.prompt_tokens
                    state.last_engine_request_id = rid
                    if (
                        state.pending_consumed_migration_id
                        and home_instance == state.home_instance
                    ):
                        consumed_migration_id = state.pending_consumed_migration_id
                        consumed_within_s = (
                            state.last_active_ts
                            - state.last_migration_completed_monotonic
                        )
                        state.pending_consumed_migration_id = ""
            if consumed_migration_id:
                result.was_just_migrated = True
                result.used_hydrated_bundle = result.cached_tokens > 0
                if self._migration_sink is not None:
                    self._migration_sink.mark_consumed(
                        migration_id=consumed_migration_id,
                        rid=rid,
                        cached_tokens=result.cached_tokens,
                        consumed_within_s=consumed_within_s,
                        used_hydrated_bundle=result.used_hydrated_bundle,
                    )
            await self._maybe_schedule_migrations()
        return result

    async def finalize_migrations(self) -> None:
        if self._migration_tasks:
            done, pending = await asyncio.wait(
                self._migration_tasks,
                timeout=max(1.0, self._policy.pending_migration_wait_timeout_s),
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                with suppress(Exception):
                    task.result()
        if self._migration_sink is not None:
            self._migration_sink.finalize()

    # --- introspection (handy for tests / driver logging) ------------------

    def session_state_snapshot(self) -> Mapping[SessionId, SessionState]:
        return dict(self._session_state)

    @property
    def policy(self) -> Policy:
        return self._policy

"""Router protocol consumed by the workload generator.

Phase 3 only needs the call-shape contract; concrete routers are built in
Phase 5 (gateway baselines) and Phase 7 (tc_router). Phase 4 will add the
metrics scaffolding around this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass
class GenerateResult:
    """Result of one `Router.generate` call.

    `text` is the model's streamed output, kept for completeness but
    discarded by the workload driver under faithful-replay (arch § 5.2.1).
    Tokens / cached_tokens come from SGLang's response `meta_info`.
    `served_instance` is the `instance_id` (`<host>:<port>`) of the
    serving SGLang worker that handled the request.
    `used_hydrated_bundle` and `was_just_migrated` are populated by
    `tc_router` only; gateway baselines leave them False.
    """

    text: str = ""
    ttft_ms: Optional[float] = None
    latency_ms: Optional[float] = None
    served_instance: str = ""
    prompt_tokens: int = 0
    cached_tokens: int = 0
    used_hydrated_bundle: bool = False
    was_just_migrated: bool = False
    raw_meta_info: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: str = ""


class Router(Protocol):
    """Minimal Router protocol per arch § 6.2."""

    async def generate(
        self,
        *,
        session_id: str,
        messages: list[dict],
        tools: Optional[list[dict]],
        sampling_params: dict,
    ) -> GenerateResult: ...

    async def close(self) -> None: ...

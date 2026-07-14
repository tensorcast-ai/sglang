# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

LogicalSessionIdSource = Literal[
    "disabled",
    "routing_key",
    "routing_key_then_rid_regex",
]
LogicalSessionResolutionReason = Literal[
    "disabled",
    "routing_key",
    "regex",
    "missing_metadata",
    "regex_no_match",
    "invalid_generation",
    "invalid_source",
]


class LogicalSessionResolution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    logical_session_id: str | None = None
    session_generation: int | None = None
    reason: LogicalSessionResolutionReason


def resolve_logical_session(
    *,
    source: LogicalSessionIdSource | str,
    routing_key: str | None,
    rid: str,
    rid_regex: str | None = None,
) -> LogicalSessionResolution:
    normalized_routing_key = str(routing_key).strip() if routing_key else ""
    if source == "disabled":
        return LogicalSessionResolution(reason="disabled")
    if source == "routing_key":
        if normalized_routing_key:
            return LogicalSessionResolution(
                logical_session_id=normalized_routing_key,
                reason="routing_key",
            )
        return LogicalSessionResolution(reason="missing_metadata")
    if source != "routing_key_then_rid_regex":
        return LogicalSessionResolution(reason="invalid_source")

    if normalized_routing_key:
        return LogicalSessionResolution(
            logical_session_id=normalized_routing_key,
            reason="routing_key",
        )
    if not rid_regex:
        return LogicalSessionResolution(reason="missing_metadata")
    match = re.match(rid_regex, rid)
    if match is None:
        return LogicalSessionResolution(reason="regex_no_match")
    groups = match.groupdict()
    logical_session_id = str(groups.get("session_id") or "").strip()
    if not logical_session_id:
        return LogicalSessionResolution(reason="missing_metadata")
    generation_text = groups.get("generation")
    if generation_text is None or str(generation_text).strip() == "":
        return LogicalSessionResolution(
            logical_session_id=logical_session_id,
            reason="regex",
        )
    try:
        session_generation = int(str(generation_text).strip())
    except ValueError:
        return LogicalSessionResolution(reason="invalid_generation")
    return LogicalSessionResolution(
        logical_session_id=logical_session_id,
        session_generation=session_generation,
        reason="regex",
    )

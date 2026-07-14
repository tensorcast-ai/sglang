from __future__ import annotations

from sglang.srt.tensorcast.request_bundle.logical_session import (
    resolve_logical_session,
)


def test_resolver_uses_routing_key() -> None:
    result = resolve_logical_session(
        source="routing_key",
        routing_key="session-a",
        rid="rid-1",
    )

    assert result.logical_session_id == "session-a"
    assert result.session_generation is None
    assert result.reason == "routing_key"


def test_resolver_disabled() -> None:
    result = resolve_logical_session(
        source="disabled",
        routing_key="session-a",
        rid="rid-1",
    )

    assert result.logical_session_id is None
    assert result.reason == "disabled"


def test_resolver_reports_missing_metadata() -> None:
    result = resolve_logical_session(
        source="routing_key",
        routing_key=None,
        rid="rid-1",
    )

    assert result.logical_session_id is None
    assert result.reason == "missing_metadata"


def test_regex_fallback_extracts_session_and_generation() -> None:
    result = resolve_logical_session(
        source="routing_key_then_rid_regex",
        routing_key=None,
        rid="sess-42-turn-7",
        rid_regex=r"^sess-(?P<session_id>\d+)-turn-(?P<generation>\d+)$",
    )

    assert result.logical_session_id == "42"
    assert result.session_generation == 7
    assert result.reason == "regex"


def test_routing_key_wins_before_regex_fallback() -> None:
    result = resolve_logical_session(
        source="routing_key_then_rid_regex",
        routing_key="header-session",
        rid="sess-42-turn-7",
        rid_regex=r"^sess-(?P<session_id>\d+)-turn-(?P<generation>\d+)$",
    )

    assert result.logical_session_id == "header-session"
    assert result.session_generation is None
    assert result.reason == "routing_key"


def test_regex_malformed_generation_rejects_session() -> None:
    result = resolve_logical_session(
        source="routing_key_then_rid_regex",
        routing_key=None,
        rid="sess-42-turn-bad",
        rid_regex=r"^sess-(?P<session_id>\d+)-turn-(?P<generation>[^-]+)$",
    )

    assert result.logical_session_id is None
    assert result.reason == "invalid_generation"


def test_regex_no_match_does_not_hide_router_specific_parser() -> None:
    result = resolve_logical_session(
        source="routing_key_then_rid_regex",
        routing_key=None,
        rid="router-specific-rid-without-configured-shape",
        rid_regex=r"^sess-(?P<session_id>\d+)-turn-(?P<generation>\d+)$",
    )

    assert result.logical_session_id is None
    assert result.reason == "regex_no_match"

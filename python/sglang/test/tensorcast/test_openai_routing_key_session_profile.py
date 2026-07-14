from __future__ import annotations

from types import SimpleNamespace

from sglang.srt.entrypoints.openai.routing_key import extract_smg_routing_key


def test_openai_serving_extracts_smg_routing_key_header() -> None:
    raw_request = SimpleNamespace(headers={"x-smg-routing-key": "session-openai"})

    assert extract_smg_routing_key(raw_request) == "session-openai"


def test_openai_serving_missing_routing_key_header_returns_none() -> None:
    raw_request = SimpleNamespace(headers={})

    assert extract_smg_routing_key(raw_request) is None

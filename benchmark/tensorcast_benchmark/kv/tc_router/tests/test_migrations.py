"""Tests for per-migration JSONL records."""

from __future__ import annotations

import json
from pathlib import Path

from tensorcast_benchmark.kv.tc_router.metrics.migrations import MigrationRecordSink


def test_migration_record_sink_writes_consumed_and_wasted_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "migrations.jsonl"
    sink = MigrationRecordSink(path)

    sink.start(
        migration_id="m1",
        session_id="s1",
        source_instance="a",
        target_instance="b",
        source_engine_request_id="r0",
        decided_by="unit",
    )
    sink.mark_success(
        migration_id="m1",
        publish_latency_ms=1.0,
        hydrate_latency_ms=2.0,
        publish_manifest_digest="publish",
        artifact_manifest_digest="artifact",
        published_cutoff_token_count=100,
        tail_valid_tokens=0,
    )
    sink.mark_consumed(
        migration_id="m1",
        rid="r1",
        cached_tokens=80,
        consumed_within_s=0.5,
        used_hydrated_bundle=True,
    )
    sink.start(
        migration_id="m2",
        session_id="s2",
        source_instance="a",
        target_instance="c",
        source_engine_request_id="r2",
        decided_by="unit",
    )

    sink.finalize()
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert rows[0]["migration_id"] == "m1"
    assert rows[0]["status"] == "consumed"
    assert rows[0]["consumed_by_turn_rid"] == "r1"
    assert rows[0]["prepared_bundle_attached"] is True
    assert rows[0]["wasted"] is False
    assert rows[1]["migration_id"] == "m2"
    assert rows[1]["status"] == "unconsumed"
    assert rows[1]["wasted"] is True

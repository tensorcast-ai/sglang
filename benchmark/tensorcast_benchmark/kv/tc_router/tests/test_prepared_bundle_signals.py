"""Tests for prepared-bundle log verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorcast_benchmark.kv.tc_router.metrics.prepared_bundle_signals import (
    extract_prepared_bundle_signals,
    verify_prepared_bundle_signals_for_migrations,
)


def test_extract_prepared_bundle_signals_matches_manifest_and_request() -> None:
    text = "\n".join(
        [
            "Tensorcast prepared-bundle attached logical_request_id=other manifest=d1 prepared_bundle_key=x",
            "Tensorcast prepared-bundle attached logical_request_id=rid-source manifest=d2 prepared_bundle_key=x",
            "Tensorcast prepared-bundle consume failed rid=rid-target manifest=d2",
        ]
    )

    signals = extract_prepared_bundle_signals(
        log_text=text,
        publish_manifest_digest="d2",
        logical_request_ids=("rid-source", "rid-target"),
    )

    assert signals.attached is True
    assert signals.consume_failed is True
    assert signals.fallback is False
    assert signals.fail_closed is False


def test_extract_prepared_bundle_signals_detects_matching_fallback() -> None:
    text = (
        "Tensorcast prepared-bundle falling back to normal generate path: stale "
        "logical_request_id=rid-source manifest=d1 stale=True tainted=False"
    )

    signals = extract_prepared_bundle_signals(
        log_text=text,
        publish_manifest_digest="d1",
        logical_request_ids=("rid-source",),
    )

    assert signals.fallback is True
    assert signals.attached is False


@pytest.mark.asyncio
async def test_verify_prepared_bundle_signals_updates_migrations_jsonl(
    tmp_path: Path,
) -> None:
    migrations_path = tmp_path / "migrations.jsonl"
    log_path = tmp_path / "target.log"
    migration = {
        "migration_id": "m1",
        "status": "consumed",
        "session_id": "s1",
        "source_instance": "a",
        "target_instance": "b",
        "source_engine_request_id": "rid-source",
        "publish_manifest_digest": "digest",
        "consumed_by_turn_rid": "rid-target",
        "prepared_bundle_attached": False,
        "prepared_bundle_fallback": False,
        "prepared_bundle_fail_closed": False,
        "prepared_bundle_consume_failed": False,
        "wasted": True,
    }
    migrations_path.write_text(json.dumps(migration) + "\n", encoding="utf-8")
    log_path.write_text(
        "Tensorcast prepared-bundle attached logical_request_id=rid-source "
        "manifest=digest prepared_bundle_key=x\n",
        encoding="utf-8",
    )

    await verify_prepared_bundle_signals_for_migrations(
        migrations_path=migrations_path,
        instance_log_paths={"b": log_path},
        timeout_s=0.01,
        poll_interval_s=0.01,
    )

    row = json.loads(migrations_path.read_text(encoding="utf-8").strip())
    assert row["prepared_bundle_attached"] is True
    assert row["prepared_bundle_fallback"] is False
    assert row["prepared_bundle_fail_closed"] is False
    assert row["prepared_bundle_consume_failed"] is False
    assert row["wasted"] is False

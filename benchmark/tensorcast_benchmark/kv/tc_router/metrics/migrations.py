"""Per-cell migration records for Tensorcast-backed tc_router runs."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MigrationRecord:
    ts: float
    migration_id: str
    is_warmup: bool
    session_id: str
    source_instance: str
    target_instance: str
    source_engine_request_id: str
    status: str
    publish_latency_ms: Optional[float] = None
    hydrate_latency_ms: Optional[float] = None
    publish_manifest_digest: str = ""
    artifact_manifest_digest: str = ""
    published_cutoff_token_count: Optional[int] = None
    tail_valid_tokens: Optional[int] = None
    transferred_bytes_estimated: Optional[int] = None
    decided_by: str = ""
    consumed_by_turn_rid: str = ""
    consumed_within_s: Optional[float] = None
    target_turn_cached_tokens: Optional[int] = None
    prepared_bundle_attached: bool = False
    prepared_bundle_fallback: bool = False
    prepared_bundle_fail_closed: bool = False
    prepared_bundle_consume_failed: bool = False
    wasted: bool = False
    error_message: str = ""


class MigrationRecordSink:
    """Collect migration records in memory and write finalized JSONL rows."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._records: dict[str, MigrationRecord] = {}
        self._finalized = False

    @property
    def path(self) -> Path:
        return self._path

    def start(
        self,
        *,
        migration_id: str,
        session_id: str,
        source_instance: str,
        target_instance: str,
        source_engine_request_id: str,
        decided_by: str,
        is_warmup: bool = False,
    ) -> None:
        self._records[migration_id] = MigrationRecord(
            ts=time.time(),
            migration_id=migration_id,
            is_warmup=is_warmup,
            session_id=session_id,
            source_instance=source_instance,
            target_instance=target_instance,
            source_engine_request_id=source_engine_request_id,
            status="running",
            decided_by=decided_by,
        )

    def mark_success(
        self,
        *,
        migration_id: str,
        publish_latency_ms: float,
        hydrate_latency_ms: float,
        publish_manifest_digest: str,
        artifact_manifest_digest: str,
        published_cutoff_token_count: int | None,
        tail_valid_tokens: int | None,
    ) -> None:
        record = self._records[migration_id]
        record.status = "unconsumed"
        record.publish_latency_ms = publish_latency_ms
        record.hydrate_latency_ms = hydrate_latency_ms
        record.publish_manifest_digest = publish_manifest_digest
        record.artifact_manifest_digest = artifact_manifest_digest
        record.published_cutoff_token_count = published_cutoff_token_count
        record.tail_valid_tokens = tail_valid_tokens

    def mark_failure(
        self,
        *,
        migration_id: str,
        status: str,
        error_message: str,
        publish_latency_ms: float | None = None,
        hydrate_latency_ms: float | None = None,
    ) -> None:
        record = self._records[migration_id]
        record.status = status
        record.error_message = error_message
        record.publish_latency_ms = publish_latency_ms
        record.hydrate_latency_ms = hydrate_latency_ms
        record.wasted = True

    def mark_consumed(
        self,
        *,
        migration_id: str,
        rid: str,
        cached_tokens: int,
        consumed_within_s: float,
        used_hydrated_bundle: bool,
    ) -> None:
        record = self._records[migration_id]
        record.status = "consumed"
        record.consumed_by_turn_rid = rid
        record.consumed_within_s = consumed_within_s
        record.target_turn_cached_tokens = cached_tokens
        record.prepared_bundle_attached = used_hydrated_bundle
        record.wasted = False

    def finalize(self) -> None:
        if self._finalized:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        for record in self._records.values():
            if record.status in {"running", "unconsumed"}:
                record.status = "unconsumed"
                record.wasted = True
        with self._path.open("w", encoding="utf-8") as file:
            for record in self._records.values():
                file.write(json.dumps(asdict(record)) + "\n")
        self._finalized = True

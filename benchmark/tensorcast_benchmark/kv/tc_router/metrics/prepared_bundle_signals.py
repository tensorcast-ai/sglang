"""Prepared-bundle log verification for tc_router migrations."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class PreparedBundleSignals:
    attached: bool = False
    fallback: bool = False
    fail_closed: bool = False
    consume_failed: bool = False

    @property
    def observed(self) -> bool:
        return self.attached or self.fallback or self.fail_closed or self.consume_failed

    def merged(self, other: "PreparedBundleSignals") -> "PreparedBundleSignals":
        return PreparedBundleSignals(
            attached=self.attached or other.attached,
            fallback=self.fallback or other.fallback,
            fail_closed=self.fail_closed or other.fail_closed,
            consume_failed=self.consume_failed or other.consume_failed,
        )


def extract_prepared_bundle_signals(
    *,
    log_text: str,
    publish_manifest_digest: str,
    logical_request_ids: tuple[str, ...],
) -> PreparedBundleSignals:
    """Extract SGLang prepared-bundle signals for one migration manifest.

    In the session-scoped profile, SGLang may log the source request id for
    attached/fallback records and the target request id for consume failures.
    Accept either id, but always require the publish manifest digest for
    manifest-specific signals.
    """
    attached = False
    fallback = False
    fail_closed = False
    consume_failed = False
    request_ids = tuple(rid for rid in logical_request_ids if rid)
    for line in log_text.splitlines():
        has_manifest = bool(publish_manifest_digest) and publish_manifest_digest in line
        has_request_id = any(rid in line for rid in request_ids)
        if (
            "Tensorcast prepared-bundle attached" in line
            and has_manifest
            and has_request_id
        ):
            attached = True
        if (
            "Tensorcast prepared-bundle falling back to normal generate path:" in line
            and has_manifest
            and has_request_id
        ):
            fallback = True
        if (
            "Tensorcast prepared-bundle fail-closed during ordinary generate admission:"
            in line
            and (has_manifest or has_request_id)
        ):
            fail_closed = True
        if (
            "Tensorcast prepared-bundle consume failed rid=" in line
            and has_manifest
            and has_request_id
        ):
            consume_failed = True
    return PreparedBundleSignals(
        attached=attached,
        fallback=fallback,
        fail_closed=fail_closed,
        consume_failed=consume_failed,
    )


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


async def _wait_for_signals(
    *,
    log_path: Path,
    publish_manifest_digest: str,
    logical_request_ids: tuple[str, ...],
    timeout_s: float,
    poll_interval_s: float,
) -> PreparedBundleSignals:
    deadline = asyncio.get_running_loop().time() + timeout_s
    aggregate = PreparedBundleSignals()
    while True:
        observed = extract_prepared_bundle_signals(
            log_text=_read_text_if_exists(log_path),
            publish_manifest_digest=publish_manifest_digest,
            logical_request_ids=logical_request_ids,
        )
        aggregate = aggregate.merged(observed)
        if aggregate.observed or asyncio.get_running_loop().time() >= deadline:
            return aggregate
        await asyncio.sleep(poll_interval_s)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


async def verify_prepared_bundle_signals_for_migrations(
    *,
    migrations_path: Path,
    instance_log_paths: Mapping[str, Path],
    timeout_s: float = 5.0,
    poll_interval_s: float = 0.25,
) -> None:
    """Update migration rows with target-side prepared-bundle log signals."""
    rows = _load_jsonl(migrations_path)
    if not rows:
        return

    changed = False
    for row in rows:
        if row.get("status") != "consumed":
            continue
        publish_manifest_digest = str(row.get("publish_manifest_digest") or "")
        if not publish_manifest_digest:
            continue
        target_instance = str(row.get("target_instance") or "")
        log_path = instance_log_paths.get(target_instance)
        if log_path is None:
            continue
        signals = await _wait_for_signals(
            log_path=Path(log_path),
            publish_manifest_digest=publish_manifest_digest,
            logical_request_ids=(
                str(row.get("source_engine_request_id") or ""),
                str(row.get("consumed_by_turn_rid") or ""),
            ),
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
        )
        row["prepared_bundle_attached"] = signals.attached
        row["prepared_bundle_fallback"] = signals.fallback
        row["prepared_bundle_fail_closed"] = signals.fail_closed
        row["prepared_bundle_consume_failed"] = signals.consume_failed
        if signals.attached:
            row["wasted"] = False
        if signals.fallback or signals.fail_closed or signals.consume_failed:
            row["wasted"] = True
        changed = True

    if changed:
        _write_jsonl(migrations_path, rows)

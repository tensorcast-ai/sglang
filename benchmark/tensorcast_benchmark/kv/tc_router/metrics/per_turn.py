"""Per-turn record schema and JSONL sink.

`TurnRecord` matches arch § 10.1. The writer flushes on every record so
that a crashed run still has data on disk.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Optional


@dataclass
class TurnRecord:
    """One LLM call inside a session.

    Mirrors arch § 10.1 verbatim. Field order kept stable so that JSON
    diffs across runs stay sane.
    """

    ts: float
    session_id: str
    instance_id: str
    turn_index: int
    prompt_messages_count: int
    prompt_tokens: int
    max_new_tokens: int
    served_instance: str
    ttft_ms: Optional[float]
    latency_ms: Optional[float]
    cached_tokens: int
    used_hydrated_bundle: bool
    was_just_migrated: bool
    rid: str
    success: bool
    error_message: str = ""


class TurnRecordWriter:
    """Append-mode JSONL writer that flushes after each record.

    Use as a context manager:

        with TurnRecordWriter("turns.jsonl") as w:
            w.write(record)
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: Optional[IO[str]] = self._path.open("a", encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    def write(self, record: TurnRecord) -> None:
        if self._fh is None:
            raise RuntimeError("TurnRecordWriter is closed")
        self._fh.write(json.dumps(asdict(record)) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "TurnRecordWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

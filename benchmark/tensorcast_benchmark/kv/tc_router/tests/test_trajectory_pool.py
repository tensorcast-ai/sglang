"""Trajectory pool loader tests against a synthetic in-memory parquet."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tensorcast_benchmark.kv.tc_router.workload.trajectory_pool import (
    Trajectory,
    _assistant_indices,
    _strip_nulls_deep,
    _trajectory_total_chars,
    _turn_chars,
    load_pool,
)


def _msg(role: str, content: str = "", tool_calls=None, tool_call_id=None, name=None) -> dict:
    return {
        "role": role,
        "content": content,
        "tool_calls": tool_calls,
        "tool_call_id": tool_call_id,
        "name": name,
        "function_call": None,
    }


def _tool_call(name: str, arguments: str, call_id: str) -> dict:
    return {
        "id": call_id,
        "index": 0,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _make_long_trajectory(session_id: str, instance_id: str, *, num_assistants: int = 5, content_len: int = 5000, resolved: bool = False) -> dict:
    """Build a synthetic trajectory with `num_assistants` LLM-call boundaries."""
    messages = [
        _msg("system", "you are an SWE agent"),
        _msg("user", "X" * content_len),  # initial issue
    ]
    for i in range(num_assistants):
        messages.append(
            _msg(
                "assistant",
                content="",
                tool_calls=[_tool_call("str_replace_editor", '{"command":"view"}', f"call_{i}")],
            )
        )
        messages.append(_msg("tool", "Y" * content_len, tool_call_id=f"call_{i}", name="str_replace_editor"))
    return {
        "instance_id": instance_id,
        "run_id": session_id,
        "resolved": resolved,
        "messages": messages,
        "tools": [],
    }


def _write_parquet(tmp_path: Path, rows: list[dict]) -> Path:
    """Write rows as parquet at `<tmp_path>/data/test.parquet`."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    path = data_dir / "test.parquet"
    pq.write_table(table, path)
    return path


# --- pure helpers ------------------------------------------------------------


def test_turn_chars_includes_content_and_tool_call_args() -> None:
    m = _msg("assistant", "hi", tool_calls=[_tool_call("foo", "abcd", "x1")])
    # 2 (content "hi") + 3 ("foo") + 4 ("abcd") = 9
    assert _turn_chars(m) == 9


def test_assistant_indices_returns_only_assistant_positions() -> None:
    msgs = [
        _msg("system"),
        _msg("user", "u"),
        _msg("assistant", "a1"),
        _msg("tool", "t"),
        _msg("assistant", "a2"),
    ]
    assert _assistant_indices(msgs) == (2, 4)


def test_trajectory_total_chars_sums_all_messages() -> None:
    msgs = [_msg("user", "abc"), _msg("assistant", "de", tool_calls=[_tool_call("n", "ar", "i")])]
    # 3 + 2 + 1 + 2 = 8
    assert _trajectory_total_chars(msgs) == 8


# --- load_pool ---------------------------------------------------------------


def test_filter_keeps_long_trajectory_rejects_short_one(tmp_path: Path) -> None:
    long_traj = _make_long_trajectory("sess_long", "task_long", num_assistants=5, content_len=5000)
    short_traj = _make_long_trajectory("sess_short", "task_short", num_assistants=2, content_len=100)
    _write_parquet(tmp_path, [long_traj, short_traj])

    pool = load_pool(
        tmp_path,
        min_turns=8,
        min_total_tokens=8000,
    )
    assert len(pool) == 1
    assert pool[0].session_id == "sess_long"


def test_assistant_indices_in_returned_trajectory(tmp_path: Path) -> None:
    traj = _make_long_trajectory("s", "t", num_assistants=3, content_len=5000)
    _write_parquet(tmp_path, [traj])
    pool = load_pool(tmp_path, min_turns=4, min_total_tokens=0)
    assert len(pool) == 1
    # Layout: [system, user, asst0, tool0, asst1, tool1, asst2, tool2]
    # assistants at indices 2, 4, 6.
    assert pool[0].assistant_indices == (2, 4, 6)


def test_messages_preserved_verbatim(tmp_path: Path) -> None:
    traj = _make_long_trajectory("s", "t", num_assistants=2, content_len=5000)
    _write_parquet(tmp_path, [traj])
    pool = load_pool(tmp_path, min_turns=4, min_total_tokens=0)
    assert pool[0].messages[0]["role"] == "system"
    assert pool[0].messages[1]["role"] == "user"
    assert pool[0].messages[2]["role"] == "assistant"
    # Tool call structure preserved.
    assert pool[0].messages[2]["tool_calls"][0]["function"]["name"] == "str_replace_editor"


def test_shuffle_is_deterministic(tmp_path: Path) -> None:
    rows = [
        _make_long_trajectory(f"s{i}", f"t{i}", num_assistants=3, content_len=5000)
        for i in range(10)
    ]
    _write_parquet(tmp_path, rows)
    pool_a = load_pool(tmp_path, min_turns=4, min_total_tokens=0, seed=42)
    pool_b = load_pool(tmp_path, min_turns=4, min_total_tokens=0, seed=42)
    pool_c = load_pool(tmp_path, min_turns=4, min_total_tokens=0, seed=99)
    ids_a = [t.session_id for t in pool_a]
    ids_b = [t.session_id for t in pool_b]
    ids_c = [t.session_id for t in pool_c]
    assert ids_a == ids_b
    assert ids_a != ids_c


def test_skip_trajectories_with_no_assistant_when_required(tmp_path: Path) -> None:
    bad = {
        "instance_id": "x",
        "run_id": "y",
        "resolved": False,
        "messages": [_msg("system"), _msg("user", "u" * 100000)],
        "tools": [],
    }
    _write_parquet(tmp_path, [bad])
    pool = load_pool(tmp_path, min_turns=2, min_total_tokens=0)
    assert pool == []


def test_missing_dataset_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_pool(tmp_path / "nope")


# --- _strip_nulls_deep ------------------------------------------------------


def test_strip_nulls_deep_removes_nested_nones() -> None:
    obj = {
        "function": {
            "name": "execute_bash",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "enum": None},
                    "file_text": None,
                    "view_range": None,
                },
                "required": ["command"],
            },
        },
        "type": "function",
    }
    cleaned = _strip_nulls_deep(obj)
    # Top-level shape preserved.
    assert cleaned["function"]["name"] == "execute_bash"
    assert cleaned["function"]["parameters"]["type"] == "object"
    # Per-property None entries gone.
    assert "file_text" not in cleaned["function"]["parameters"]["properties"]
    assert "view_range" not in cleaned["function"]["parameters"]["properties"]
    # Inner enum=None gone but `command` still present with its other fields.
    assert cleaned["function"]["parameters"]["properties"]["command"] == {
        "type": "string"
    }
    assert cleaned["function"]["parameters"]["required"] == ["command"]


def test_strip_nulls_deep_preserves_falsy_non_none_values() -> None:
    obj = {"a": 0, "b": False, "c": "", "d": None, "e": [0, "", False, None]}
    assert _strip_nulls_deep(obj) == {"a": 0, "b": False, "c": "", "e": [0, "", False]}


def test_strip_nulls_deep_handles_lists() -> None:
    obj = [{"x": 1, "y": None}, None, {"z": [None, 2, None]}]
    assert _strip_nulls_deep(obj) == [{"x": 1}, {"z": [2]}]


def test_strip_nulls_deep_returns_none_for_none() -> None:
    assert _strip_nulls_deep(None) is None

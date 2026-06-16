"""SGLang launch command construction tests.

These do NOT invoke `sglang.launch_server` or any subprocess; they exercise
the pure command builder. Live tests against a real SGLang instance live
in `tools/live_check_sglang.py`.
"""

from __future__ import annotations

import json

import pytest

from tensorcast_benchmark.kv.tc_router.services.sglang import (
    FORBIDDEN_ARGS,
    SGLangLaunchSpec,
    build_launch_command,
)


def make_spec(**overrides) -> SGLangLaunchSpec:
    defaults = dict(
        model_path="/path/to/model",
        host="10.0.0.1",
        port=30000,
    )
    defaults.update(overrides)
    return SGLangLaunchSpec(**defaults)


# --- Required-arg presence ---------------------------------------------------


def test_basic_command_has_required_args() -> None:
    cmd = build_launch_command(make_spec())
    assert "sglang.launch_server" in cmd
    assert "--host 10.0.0.1" in cmd
    assert "--port 30000" in cmd
    assert "--model-path /path/to/model" in cmd
    assert "--tp 1" in cmd
    assert "--page-size 32" in cmd
    assert "--mem-fraction-static 0.85" in cmd


def test_command_always_enables_cache_report() -> None:
    """Required so /v1/chat/completions populates `usage.prompt_tokens_details.cached_tokens`.

    Without this flag the `cached_token_ratio_mean` summary metric is
    permanently 0 (arch § 10.3 + plan §13 risk register).
    """
    cmd = build_launch_command(make_spec())
    assert "--enable-cache-report" in cmd


def test_uses_uv_with_offline_no_project() -> None:
    cmd = build_launch_command(make_spec())
    # Must invoke through uv with the share_remote-conventional flags.
    assert "uv " in cmd  # default uv_bin ends with `uv`
    assert "run" in cmd
    assert "--active" in cmd
    assert "--no-project" in cmd
    assert "--offline" in cmd


def test_command_activates_workspace_venv() -> None:
    """build_launch_command must prepend `cd <sglang>; source .venv/bin/activate; export PYTHONPATH=...; export PATH=...`
    so that `uv run --active` picks up the workspace venv (and sglang.launch_server
    is importable).
    """
    cmd = build_launch_command(make_spec(workspace_root="/home/u/tot"))
    assert "cd /home/u/tot/thirdparty/sglang;" in cmd
    assert "source /home/u/tot/.venv/bin/activate" in cmd
    assert "export PYTHONPATH=/home/u/tot/thirdparty/sglang/python" in cmd
    assert "export PATH=" in cmd


def test_uv_bin_override() -> None:
    cmd = build_launch_command(make_spec(uv_bin="/opt/custom/uv"))
    assert "/opt/custom/uv" in cmd


# --- Tool-call-parser guarantee (arch § 5.2.3 / plan validation gate) -------


def test_default_command_does_not_pass_tool_call_parser() -> None:
    cmd = build_launch_command(make_spec())
    for forbidden in FORBIDDEN_ARGS:
        assert forbidden not in cmd, (
            f"build_launch_command leaked forbidden flag {forbidden!r}: {cmd}"
        )


def test_extra_args_with_tool_call_parser_is_rejected() -> None:
    with pytest.raises(ValueError, match="forbidden flag"):
        build_launch_command(
            make_spec(extra_args=("--tool-call-parser=qwen",))
        )


def test_extra_args_with_tool_call_parser_two_token_form_rejected() -> None:
    with pytest.raises(ValueError, match="forbidden flag"):
        build_launch_command(
            make_spec(extra_args=("--tool-call-parser", "qwen"))
        )


# --- HiCache flag gating -----------------------------------------------------


def test_hicache_off_by_default() -> None:
    cmd = build_launch_command(make_spec())
    assert "--enable-hierarchical-cache" not in cmd
    assert "--hicache-mem-layout" not in cmd
    assert "--hicache-storage-backend" not in cmd


def test_hicache_on_emits_all_hicache_flags() -> None:
    cmd = build_launch_command(make_spec(enable_hierarchical_cache=True))
    assert "--enable-hierarchical-cache" in cmd
    assert "--hicache-mem-layout page_first_direct" in cmd
    assert "--hicache-io-backend direct" in cmd
    assert "--hicache-ratio 2.0" in cmd
    assert "--hicache-size 0" in cmd
    assert "--hicache-storage-prefetch-policy wait_complete" in cmd


# --- Storage backend gating --------------------------------------------------


def test_storage_backend_none_emits_no_backend_flag() -> None:
    cmd = build_launch_command(make_spec())
    assert "--hicache-storage-backend" not in cmd
    assert "--hicache-storage-backend-extra-config" not in cmd


def test_storage_backend_mooncake_with_extra_config() -> None:
    cmd = build_launch_command(
        make_spec(
            hicache_storage_backend="mooncake",
            hicache_storage_backend_extra_config={"foo": 1, "bar": "baz"},
        )
    )
    assert "--hicache-storage-backend mooncake" in cmd
    # JSON is sorted + compact; assert presence of canonical form.
    expected_json = json.dumps({"bar": "baz", "foo": 1}, separators=(",", ":"), sort_keys=True)
    assert expected_json in cmd


def test_storage_backend_tensorcast_emits_correct_flag() -> None:
    cmd = build_launch_command(make_spec(hicache_storage_backend="tensorcast"))
    assert "--hicache-storage-backend tensorcast" in cmd


# --- Misc --------------------------------------------------------------------


def test_trust_remote_code_off_by_default() -> None:
    cmd = build_launch_command(make_spec())
    assert "--trust-remote-code" not in cmd


def test_trust_remote_code_when_enabled() -> None:
    cmd = build_launch_command(make_spec(trust_remote_code=True))
    assert "--trust-remote-code" in cmd


def test_extra_args_appended() -> None:
    cmd = build_launch_command(
        make_spec(extra_args=("--log-level", "debug"))
    )
    assert "--log-level debug" in cmd


def test_tp_size_serialized() -> None:
    cmd = build_launch_command(make_spec(tp_size=4))
    assert "--tp 4" in cmd

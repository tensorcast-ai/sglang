"""LocalProvider behavior tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

from tensorcast_benchmark.kv.tc_router.resource.local import (
    LocalProcessError,
    LocalProvider,
)


def _cluster_yaml(
    tmp_path: Path,
    *,
    base_env: dict[str, str] | None = None,
    env_path_prepend: dict[str, list[str]] | None = None,
    env_unset: list[str] | None = None,
) -> dict:
    return {
        "provider": {"kind": "local"},
        "driver_host": {"scratch_dir": str(tmp_path / "driver")},
        "mount": {"path": str(tmp_path / "mount"), "spec": "local"},
        "workers": [
            {
                "id": "local_test",
                "address": "127.0.0.1",
                "node": "local_test_node",
                "process_handle": "local",
                "gpu_indices": [0, 1, 2, 3],
                "scratch_dir": str(tmp_path / "worker"),
                "base_env": base_env or {},
                "env_path_prepend": env_path_prepend or {},
                "env_unset": env_unset or [],
            }
        ],
        "service_placement": {
            "global_store_worker_id": "local_test",
            "mooncake_master_worker_id": "local_test",
        },
    }


def _write_cluster(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "cluster.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_local_provider_health_check_creates_local_dirs(tmp_path: Path) -> None:
    provider = LocalProvider.from_cluster_config(
        _write_cluster(tmp_path, _cluster_yaml(tmp_path))
    )

    await provider.health_check()

    worker = provider.workers()[0]
    assert Path(worker.scratch_dir).is_dir()
    assert (tmp_path / "driver").is_dir()
    assert (tmp_path / "mount").is_dir()


@pytest.mark.asyncio
async def test_local_worker_run_merges_base_and_call_env(tmp_path: Path) -> None:
    provider = LocalProvider.from_cluster_config(
        _write_cluster(
            tmp_path,
            _cluster_yaml(tmp_path, base_env={"BASE_VALUE": "base"}),
        )
    )
    worker = provider.workers()[0]

    script = (
        "import os; "
        "print(os.environ['BASE_VALUE'] + ':' + os.environ['CALL_VALUE'], end='')"
    )
    proc = await worker.run(
        [
            sys.executable,
            "-c",
            script,
        ],
        env={"CALL_VALUE": "call"},
    )

    assert proc.returncode == 0
    assert proc.stdout == "base:call"
    assert proc.stderr == ""


@pytest.mark.asyncio
async def test_local_worker_run_prepends_path_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/lib")
    provider = LocalProvider.from_cluster_config(
        _write_cluster(
            tmp_path,
            _cluster_yaml(
                tmp_path,
                env_path_prepend={"LD_LIBRARY_PATH": ["/compat/lib"]},
            ),
        )
    )
    worker = provider.workers()[0]

    proc = await worker.run(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ['LD_LIBRARY_PATH'], end='')",
        ]
    )

    assert proc.returncode == 0
    assert proc.stdout == os.pathsep.join(["/compat/lib", "/existing/lib"])


@pytest.mark.asyncio
async def test_local_worker_run_unsets_configured_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy_keys = [
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "https_proxy",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
    ]
    for key in proxy_keys:
        monkeypatch.setenv(key, "http://proxy.invalid:8080")
    provider = LocalProvider.from_cluster_config(
        _write_cluster(
            tmp_path,
            _cluster_yaml(tmp_path, env_unset=proxy_keys),
        )
    )
    worker = provider.workers()[0]

    proc = await worker.run(
        [
            sys.executable,
            "-c",
            "import os; print(any(key in os.environ for key in "
            f"{proxy_keys!r}), end='')",
        ],
        env={"HTTP_PROXY": "http://call.invalid:8080"},
    )

    assert proc.returncode == 0
    assert proc.stdout == "False"


@pytest.mark.asyncio
async def test_local_worker_run_accepts_shell_string(tmp_path: Path) -> None:
    provider = LocalProvider.from_cluster_config(
        _write_cluster(tmp_path, _cluster_yaml(tmp_path))
    )
    worker = provider.workers()[0]

    proc = await worker.run("printf '%s' shell-ok")

    assert proc.returncode == 0
    assert proc.stdout.endswith("shell-ok")


@pytest.mark.asyncio
async def test_local_worker_run_check_false_returns_nonzero(
    tmp_path: Path,
) -> None:
    provider = LocalProvider.from_cluster_config(
        _write_cluster(tmp_path, _cluster_yaml(tmp_path))
    )
    worker = provider.workers()[0]

    proc = await worker.run(
        ["bash", "-lc", "echo bad >&2; exit 7"],
        check=False,
    )

    assert proc.returncode == 7
    assert "bad" in proc.stderr


@pytest.mark.asyncio
async def test_local_worker_run_check_true_raises(tmp_path: Path) -> None:
    provider = LocalProvider.from_cluster_config(
        _write_cluster(tmp_path, _cluster_yaml(tmp_path))
    )
    worker = provider.workers()[0]

    with pytest.raises(LocalProcessError, match="rc=7"):
        await worker.run(["bash", "-lc", "exit 7"])


@pytest.mark.asyncio
async def test_local_worker_background_lifecycle(tmp_path: Path) -> None:
    provider = LocalProvider.from_cluster_config(
        _write_cluster(tmp_path, _cluster_yaml(tmp_path))
    )
    worker = provider.workers()[0]
    log_path = tmp_path / "logs" / "bg.log"
    pid_path = tmp_path / "logs" / "bg.pid"

    pid = await worker.start_background(
        "while true; do sleep 1; done",
        name="background_test",
        log_path=str(log_path),
        pid_path=str(pid_path),
    )

    assert pid_path.read_text(encoding="utf-8").strip() == str(pid)
    os.kill(pid, 0)

    await worker.stop_background(pid_path=str(pid_path))

    assert not pid_path.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.asyncio
async def test_local_worker_file_roundtrip(tmp_path: Path) -> None:
    provider = LocalProvider.from_cluster_config(
        _write_cluster(tmp_path, _cluster_yaml(tmp_path))
    )
    worker = provider.workers()[0]
    src = tmp_path / "src.txt"
    remote = tmp_path / "remote" / "payload.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("abcdef", encoding="utf-8")

    await worker.put_file(src, str(remote))
    assert await worker.read_file(str(remote), max_bytes=3) == b"abc"
    await worker.get_file(str(remote), dst)

    assert dst.read_text(encoding="utf-8") == "abcdef"

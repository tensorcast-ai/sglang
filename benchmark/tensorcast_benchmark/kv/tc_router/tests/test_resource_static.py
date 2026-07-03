"""StaticProvider behavior tests."""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

import pytest
import yaml

from tensorcast_benchmark.kv.tc_router.resource import factory
from tensorcast_benchmark.kv.tc_router.resource.base import (
    ClusterConfig,
    WorkerConfig,
    load_cluster_config,
)
from tensorcast_benchmark.kv.tc_router.resource.static import (
    LocalStaticWorker,
    SshStaticWorker,
    StaticProvider,
)


def _static_cluster() -> dict:
    return {
        "provider": {"kind": "static"},
        "driver_host": {"scratch_dir": "/mnt/data/tc_router_unit/driver"},
        "mount": {"path": "/mnt/data", "spec": "shared"},
        "workers": [
            {
                "id": "local_h800",
                "address": "127.0.0.1",
                "node": "local_h800",
                "process_handle": "local",
                "execution": "local",
                "gpu_indices": [0, 1],
                "scratch_dir": "/mnt/data/tc_router_unit/local",
                "base_env": {},
                "env_unset": ["HTTP_PROXY", "HTTPS_PROXY"],
                "env_path_prepend": {"LD_LIBRARY_PATH": ["/compat"]},
            },
            {
                "id": "remote_10_0_10_58",
                "address": "10.0.10.58",
                "node": "remote_10_0_10_58",
                "process_handle": "ssh-yuhan-10.0.10.58",
                "execution": "ssh",
                "ssh_user": "yuhan",
                "ssh_host": "10.0.10.58",
                "ssh_port": 22,
                "connect_timeout_s": 10,
                "gpu_indices": [0, 1],
                "scratch_dir": "/mnt/data/tc_router_unit/remote",
                "base_env": {"BASE_VALUE": "base"},
                "env_unset": ["HTTP_PROXY", "HTTPS_PROXY"],
                "env_path_prepend": {"LD_LIBRARY_PATH": ["/compat"]},
            },
        ],
        "service_placement": {
            "global_store_worker_id": "local_h800",
            "mooncake_master_worker_id": "local_h800",
        },
    }


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "cluster.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _ssh_worker_config() -> WorkerConfig:
    return WorkerConfig.model_validate(_static_cluster()["workers"][1])


def test_static_cluster_yaml_parses_and_factory_dispatches(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, _static_cluster())

    cfg = load_cluster_config(path)
    provider = factory.from_cluster_config(path)

    assert isinstance(cfg, ClusterConfig)
    assert cfg.provider.kind == "static"
    assert isinstance(provider, StaticProvider)
    workers = provider.workers()
    assert [worker.id for worker in workers] == [
        "local_h800",
        "remote_10_0_10_58",
    ]
    assert isinstance(workers[0], LocalStaticWorker)
    assert isinstance(workers[1], SshStaticWorker)


def test_shipped_static_cluster_yaml_loads() -> None:
    example = (
        Path(__file__).parent.parent
        / "configs"
        / "cluster_static_local_h800_plus_10_0_10_58.yaml"
    )

    cfg = load_cluster_config(example)

    assert cfg.provider.kind == "static"
    assert len(cfg.workers) == 2
    assert cfg.workers[0].execution == "local"
    assert cfg.workers[1].execution == "ssh"
    assert cfg.workers[1].ssh_user == "yuhan"
    assert cfg.workers[1].ssh_host == "10.0.10.58"
    for worker in cfg.workers:
        path_prepend = worker.env_path_prepend["PATH"]
        assert "/home/yuhan/.local/bin" in path_prepend
        assert "/usr/local/cuda-13/bin" in path_prepend
        assert "/usr/local/cuda-12.8/bin" not in path_prepend
        assert worker.env_path_prepend["LD_LIBRARY_PATH"] == (
            "/usr/local/cuda-13/compat",
        )
        assert worker.env_unset == (
            "HTTPS_PROXY",
            "HTTP_PROXY",
            "https_proxy",
            "http_proxy",
            "ALL_PROXY",
            "all_proxy",
        )


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("ssh_user", "ssh_user is required"),
        ("ssh_host", "ssh_host is required"),
    ],
)
def test_static_rejects_malformed_ssh_worker(
    tmp_path: Path,
    field: str,
    expected: str,
) -> None:
    data = _static_cluster()
    data["workers"][1][field] = ""

    with pytest.raises(Exception, match=expected):
        load_cluster_config(_write_yaml(tmp_path, data))


def test_static_rejects_non_shared_scratch(tmp_path: Path) -> None:
    data = _static_cluster()
    data["workers"][0]["scratch_dir"] = "/tmp/not_shared"

    with pytest.raises(Exception, match="scratch_dir must be under /mnt/data"):
        load_cluster_config(_write_yaml(tmp_path, data))


@pytest.mark.asyncio
async def test_local_static_worker_run_merges_base_path_and_call_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing")
    cfg = WorkerConfig.model_validate(
        {
            "id": "local",
            "address": "127.0.0.1",
            "node": "local",
            "process_handle": "local",
            "execution": "local",
            "gpu_indices": [0],
            "scratch_dir": "/mnt/data/tc_router_unit/local",
            "base_env": {"BASE_VALUE": "base"},
            "env_unset": ["HTTP_PROXY"],
            "env_path_prepend": {"LD_LIBRARY_PATH": ["/compat"]},
        }
    )
    worker = LocalStaticWorker(cfg)

    script = (
        "import os; "
        "print(os.environ['BASE_VALUE'] + ':' + os.environ['CALL_VALUE'] + ':' + "
        "os.environ['LD_LIBRARY_PATH'], end='')"
    )
    proc = await worker.run(
        [sys.executable, "-c", script],
        env={"CALL_VALUE": "call", "HTTP_PROXY": "http://call.invalid:8080"},
        cwd=str(tmp_path),
    )

    assert proc.returncode == 0
    assert proc.stdout == f"base:call:{os.pathsep.join(['/compat', '/existing'])}"


@pytest.mark.asyncio
async def test_local_static_worker_run_unsets_configured_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy_keys = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]
    for key in proxy_keys:
        monkeypatch.setenv(key, "http://proxy.invalid:8080")
    cfg = WorkerConfig.model_validate(
        {
            "id": "local",
            "address": "127.0.0.1",
            "node": "local",
            "process_handle": "local",
            "execution": "local",
            "gpu_indices": [0],
            "scratch_dir": "/mnt/data/tc_router_unit/local",
            "base_env": {},
            "env_unset": proxy_keys,
        }
    )
    worker = LocalStaticWorker(cfg)

    proc = await worker.run(
        [
            sys.executable,
            "-c",
            "import os; print(any(key in os.environ for key in "
            f"{proxy_keys!r}), end='')",
        ],
        env={"HTTP_PROXY": "http://call.invalid:8080"},
        cwd=str(tmp_path),
    )

    assert proc.returncode == 0
    assert proc.stdout == "False"


def test_ssh_static_worker_builds_deterministic_ssh_argv() -> None:
    worker = SshStaticWorker(_ssh_worker_config())
    script = worker._command_script(
        ["printf", "%s", "hello world"],
        env={"CALL_VALUE": "call value"},
        cwd="/mnt/data/work dir",
    )
    argv = worker._ssh_argv(script)

    assert argv[:7] == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
    ]
    assert "-p" in argv
    assert "22" in argv
    assert "yuhan@10.0.10.58" in argv
    assert argv[-5:-1] == ["bash", "--noprofile", "--norc", "-lc"]
    assert shlex.split(argv[-1]) == [script]
    assert "export BASE_VALUE=base" in script
    assert "export LD_LIBRARY_PATH=/compat:${LD_LIBRARY_PATH:-}" in script
    assert "export CALL_VALUE='call value'" in script
    assert "unset HTTP_PROXY HTTPS_PROXY" in script
    assert "cd '/mnt/data/work dir' && printf %s 'hello world'" in script


def test_static_background_scripts_use_pid_files_and_escalation() -> None:
    worker = SshStaticWorker(_ssh_worker_config())

    start_script = worker._background_start_script(
        "sleep 60",
        "/mnt/data/logs/service.log",
        "/mnt/data/logs/service.pid",
    )
    stop_script = worker._background_stop_script("/mnt/data/logs/service.pid")

    assert "nohup setsid bash --noprofile --norc -lc" in start_script
    assert "&;" not in start_script
    assert "> /mnt/data/logs/service.log 2>&1" in start_script
    assert "printf '%s\\n' \"$pid\" > /mnt/data/logs/service.pid" in start_script
    assert 'kill -TERM -- "-$pid"' in stop_script
    assert "kill -KILL" in stop_script
    assert "rm -f /mnt/data/logs/service.pid" in stop_script


@pytest.mark.asyncio
async def test_static_file_operations_use_shared_filesystem(tmp_path: Path) -> None:
    worker = SshStaticWorker(_ssh_worker_config())
    src = tmp_path / "src.txt"
    remote = tmp_path / "shared" / "payload.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("abcdef", encoding="utf-8")

    await worker.put_file(src, str(remote))
    assert await worker.read_file(str(remote), max_bytes=3) == b"abc"
    await worker.get_file(str(remote), dst)

    assert dst.read_text(encoding="utf-8") == "abcdef"

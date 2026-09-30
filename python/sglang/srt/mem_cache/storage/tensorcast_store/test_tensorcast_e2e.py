# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest


_REAL_DAEMON_CHILD = textwrap.dedent(
    """
    import os

    import torch

    from sglang.srt.mem_cache.pool_host.common import HostTensorAllocator
    from sglang.srt.mem_cache.storage.tensorcast_store.host_allocator import (
        TensorcastHostTensorAllocator,
        build_tensorcast_session_options,
        claim_tensorcast_store_session,
        create_tensorcast_host_allocator,
        normalize_tensorcast_config,
        terminate_tensorcast_store_session,
    )

    transfer_mode = os.environ["SGLANG_TENSORCAST_CHILD_TRANSFER_MODE"]
    source = {
        "tensorcast": {
            "daemon_address": os.environ["SGLANG_TENSORCAST_DAEMON_ADDRESS"],
            "transfer_mode": transfer_mode,
            "scratch": {"capacity_bytes": 4096},
            "session_name_prefix": "sglang-phase2-e2e",
            "region_name_prefix": "sglang_phase2_e2e",
        }
    }
    allocator = create_tensorcast_host_allocator(
        source,
        host_memory_mode="cache",
        host_layout="page_first",
        io_backend="kernel",
        platform_name="linux",
        is_cuda_backend=True,
        world_rank=0,
        world_size=1,
    )
    repeated_allocator = create_tensorcast_host_allocator(
        source,
        host_memory_mode="cache",
        host_layout="page_first",
        io_backend="kernel",
        platform_name="linux",
        is_cuda_backend=True,
        world_rank=0,
        world_size=1,
    )

    owner = object()
    config = normalize_tensorcast_config(source)
    options = build_tensorcast_session_options(
        config,
        world_rank=0,
        world_size=1,
    )
    session = claim_tensorcast_store_session(options, owner=owner)

    if transfer_mode == "scratch":
        assert type(allocator) is HostTensorAllocator
        assert type(repeated_allocator) is HostTensorAllocator
    else:
        assert isinstance(allocator, TensorcastHostTensorAllocator)
        assert isinstance(repeated_allocator, TensorcastHostTensorAllocator)
        assert allocator.session is session
        assert repeated_allocator.session is session
        first = allocator.allocate((257,), torch.uint8, "cpu")
        second = allocator.allocate((3, 5), torch.float32, "cpu")
        first.fill_(19)
        second.fill_(2.5)
        assert first.device.type == "cpu"
        assert second.device.type == "cpu"
        assert first.shape == (257,)
        assert second.shape == (3, 5)
        assert first.tolist() == [19] * 257
        assert torch.equal(second, torch.full((3, 5), 2.5))

    assert terminate_tensorcast_store_session(owner=owner) is True
    assert terminate_tensorcast_store_session(owner=owner) is False
    print(f"phase2-real-daemon-{transfer_mode}=ok", flush=True)
    """
)


_REAL_DAEMON_REGISTER_AND_EXISTS_CHILD = textwrap.dedent(
    """
    import os
    from types import SimpleNamespace

    import torch

    import sglang.srt.runtime_context as runtime_context
    from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
    from sglang.srt.mem_cache.storage.tensorcast_store.host_allocator import (
        create_tensorcast_host_allocator,
        terminate_tensorcast_store_session,
    )
    from sglang.srt.mem_cache.storage.tensorcast_store.tensorcast_store import (
        TensorcastStore,
    )

    runtime_context.get_parallel = lambda: SimpleNamespace(
        world_rank=0,
        world_size=1,
    )
    source = {
        "tensorcast": {
            "daemon_address": os.environ["SGLANG_TENSORCAST_DAEMON_ADDRESS"],
            "transfer_mode": "scratch",
            "scratch": {"capacity_bytes": 4096},
            "model_id": "sglang-phase3-e2e-model",
            "session_name_prefix": "sglang-phase3-e2e",
            "region_name_prefix": "sglang_phase3_e2e",
        }
    }
    allocator = create_tensorcast_host_allocator(
        source,
        host_memory_mode="cache",
        host_layout="page_first",
        io_backend="kernel",
        platform_name="linux",
        is_cuda_backend=True,
    )

    pool = object.__new__(MHATokenToKVPoolHost)
    pool.layout = "page_first"
    pool.page_num = 3
    pool.page_size = 2
    pool.size = 6
    pool.layer_num = 1
    pool.head_num = 1
    pool.head_dim = 2
    pool.dtype = torch.float32
    pool.mtp_draft_device_pools = ()
    pool.dcp_size = 1
    pool.dcp_rank = 0
    pool.allocator = allocator
    pool.kv_buffer = torch.empty((2, 6, 1, 1, 2), dtype=torch.float32)

    storage_config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="unused-fallback",
        extra_config=source,
    )
    store = TensorcastStore(storage_config)
    store.register_mem_pool_host(pool)
    assert store.batch_exists(["phase3-guaranteed-missing-page"]) == 0
    assert terminate_tensorcast_store_session(owner=store) is True
    assert terminate_tensorcast_store_session(owner=store) is False
    print("phase3-real-daemon-register-and-exists-miss=ok", flush=True)
    """
)


_REAL_DAEMON_SCRATCH_ROUND_TRIP_CHILD = textwrap.dedent(
    """
    import os
    from types import SimpleNamespace

    import torch

    import sglang.srt.runtime_context as runtime_context
    from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
    from sglang.srt.mem_cache.storage.tensorcast_store.host_allocator import (
        create_tensorcast_host_allocator,
        terminate_tensorcast_store_session,
    )
    from sglang.srt.mem_cache.storage.tensorcast_store.tensorcast_store import (
        TensorcastStore,
    )

    runtime_context.get_parallel = lambda: SimpleNamespace(
        world_rank=0,
        world_size=1,
    )
    source = {
        "tensorcast": {
            "daemon_address": os.environ["SGLANG_TENSORCAST_DAEMON_ADDRESS"],
            "transfer_mode": "scratch",
            "scratch": {"capacity_bytes": 4096},
            "model_id": f"sglang-phase4-e2e-{os.getpid()}",
            "session_name_prefix": "sglang-phase4-e2e",
            "region_name_prefix": "sglang_phase4_e2e",
        }
    }
    allocator = create_tensorcast_host_allocator(
        source,
        host_memory_mode="cache",
        host_layout="page_first",
        io_backend="kernel",
        platform_name="linux",
        is_cuda_backend=True,
    )

    pool = object.__new__(MHATokenToKVPoolHost)
    pool.layout = "page_first"
    pool.page_num = 3
    pool.page_size = 2
    pool.size = 6
    pool.layer_num = 1
    pool.head_num = 1
    pool.head_dim = 2
    pool.dtype = torch.float32
    pool.mtp_draft_device_pools = ()
    pool.dcp_size = 1
    pool.dcp_rank = 0
    pool.allocator = allocator
    pool.kv_buffer = torch.arange(
        2 * 6 * 1 * 1 * 2,
        dtype=torch.float32,
    ).reshape(2, 6, 1, 1, 2)

    storage_config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="unused-fallback",
        extra_config=source,
    )
    store = TensorcastStore(storage_config)
    store.register_mem_pool_host(pool)
    keys = ["phase4-page-0", "phase4-page-1"]
    host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    expected = pool.kv_buffer[:, :4].clone()

    assert store.batch_exists(keys) == 0
    assert store.batch_set_v1(keys, host_indices) == [True, True]
    assert store.batch_exists(keys) == 2
    pool.kv_buffer[:, :4].zero_()
    assert store.batch_get_v1(keys, host_indices) == [True, True]
    assert torch.equal(pool.kv_buffer[:, :4], expected)
    assert terminate_tensorcast_store_session(owner=store) is True
    assert terminate_tensorcast_store_session(owner=store) is False
    print("phase4-real-daemon-scratch-full-round-trip=ok", flush=True)
    """
)


_REAL_DAEMON_ALLOCATOR_ROUND_TRIP_CHILD = textwrap.dedent(
    """
    import os
    from types import SimpleNamespace

    import torch

    import sglang.srt.runtime_context as runtime_context
    from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
    from sglang.srt.mem_cache.pool_host.mha import (
        AsymmetricMHATokenToKVPoolHost,
        MHATokenToKVPoolHost,
    )
    from sglang.srt.mem_cache.storage.tensorcast_store.host_allocator import (
        TensorcastHostTensorAllocator,
        create_tensorcast_host_allocator,
        terminate_tensorcast_store_session,
    )
    from sglang.srt.mem_cache.storage.tensorcast_store.tensorcast_store import (
        TensorcastStore,
    )

    runtime_context.get_parallel = lambda: SimpleNamespace(
        world_rank=0,
        world_size=1,
    )
    family = os.environ["SGLANG_TENSORCAST_CHILD_POOL_FAMILY"]
    source = {
        "tensorcast": {
            "daemon_address": os.environ["SGLANG_TENSORCAST_DAEMON_ADDRESS"],
            "model_id": f"sglang-phase5-e2e-{family}-{os.getpid()}",
            "session_name_prefix": "sglang-phase5-e2e",
            "region_name_prefix": "sglang_phase5_e2e",
        }
    }
    allocator = create_tensorcast_host_allocator(
        source,
        host_memory_mode="cache",
        host_layout="page_first_direct",
        io_backend="direct",
        platform_name="linux",
        is_cuda_backend=True,
    )
    assert isinstance(allocator, TensorcastHostTensorAllocator)

    page_num = 3
    page_size = 2
    size = page_num * page_size
    layer_num = 1
    head_num = 1
    head_dim = 2
    if family == "mha":
        pool = object.__new__(MHATokenToKVPoolHost)
        root = allocator.allocate(
            (2, page_num, layer_num, page_size, head_num, head_dim),
            torch.float32,
            "cpu",
        )
        root.copy_(torch.arange(root.numel(), dtype=root.dtype).reshape(root.shape))
        pool.kv_buffer = root
        roots = (root,)
    elif family == "asymmetric_mha":
        pool = object.__new__(AsymmetricMHATokenToKVPoolHost)
        v_head_dim = 3
        k_root = allocator.allocate(
            (page_num, layer_num, page_size, head_num, head_dim),
            torch.float32,
            "cpu",
        )
        v_root = allocator.allocate(
            (page_num, layer_num, page_size, head_num, v_head_dim),
            torch.float32,
            "cpu",
        )
        k_root.copy_(
            torch.arange(k_root.numel(), dtype=k_root.dtype).reshape(k_root.shape)
        )
        v_root.copy_(
            torch.arange(v_root.numel(), dtype=v_root.dtype).reshape(v_root.shape)
            + 1000
        )
        pool.kv_buffer = (k_root, v_root)
        pool.v_head_dim = v_head_dim
        roots = (k_root, v_root)
    else:
        raise AssertionError(f"unexpected pool family {family!r}")

    pool.layout = "page_first_direct"
    pool.page_num = page_num
    pool.page_size = page_size
    pool.size = size
    pool.layer_num = layer_num
    pool.head_num = head_num
    pool.head_dim = head_dim
    pool.dtype = torch.float32
    pool.mtp_draft_device_pools = ()
    pool.dcp_size = 1
    pool.dcp_rank = 0
    pool.allocator = allocator

    storage_config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="unused-fallback",
        extra_config=source,
    )
    store = TensorcastStore(storage_config)
    store.register_mem_pool_host(pool)
    registered = store._require_registered()
    if family == "mha":
        assert registered.roots[0] is roots[0]
        assert registered.roots[1] is roots[0]
    else:
        assert registered.roots[0] is roots[0]
        assert registered.roots[1] is roots[1]

    keys = [f"phase5-{family}-page-0", f"phase5-{family}-page-1"]
    host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    transferred_views = (
        (roots[0][:, :2],)
        if family == "mha"
        else (roots[0][:2], roots[1][:2])
    )
    expected = tuple(view.clone() for view in transferred_views)

    assert store.batch_exists(keys) == 0
    assert store.batch_set_v1(keys, host_indices) == [True, True]
    assert store.batch_exists(keys) == 2
    for view in transferred_views:
        view.zero_()
    assert store.batch_get_v1(keys, host_indices) == [True, True]
    assert all(
        torch.equal(view, saved)
        for view, saved in zip(transferred_views, expected)
    )
    assert terminate_tensorcast_store_session(owner=store) is True
    assert terminate_tensorcast_store_session(owner=store) is False
    print(f"phase5-real-daemon-{family}=ok", flush=True)
    """
)


_REAL_DAEMON_PHASE6_CHILD = textwrap.dedent(
    """
    import logging
    import os
    from types import SimpleNamespace

    import torch
    from tensorcast.api.store import (
        ByteArtifactSpec,
        HostMemorySpan,
        RegionArtifactTransfer,
        RegionBackedArtifactSession,
        RegionSessionFailedError,
        RegionSessionHealth,
        RegionSessionLifecycleState,
        RegionSessionTerminatedError,
    )

    import sglang.srt.runtime_context as runtime_context
    from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
    from sglang.srt.mem_cache.storage.tensorcast_store.host_allocator import (
        TensorcastHostTensorAllocator,
        create_tensorcast_host_allocator,
    )
    from sglang.srt.mem_cache.storage.tensorcast_store.tensorcast_store import (
        TensorcastStore,
    )

    logging.basicConfig(level=logging.ERROR)
    runtime_context.get_parallel = lambda: SimpleNamespace(
        world_rank=0,
        world_size=1,
    )
    scenario = os.environ["SGLANG_TENSORCAST_PHASE6_SCENARIO"]
    source = {
        "tensorcast": {
            "daemon_address": os.environ["SGLANG_TENSORCAST_DAEMON_ADDRESS"],
            "model_id": f"sglang-phase6-e2e-{scenario}-{os.getpid()}",
            "session_name_prefix": "sglang-phase6-e2e",
            "region_name_prefix": "sglang_phase6_e2e",
        }
    }
    allocator = create_tensorcast_host_allocator(
        source,
        host_memory_mode="cache",
        host_layout="page_first_direct",
        io_backend="direct",
        platform_name="linux",
        is_cuda_backend=True,
    )
    assert isinstance(allocator, TensorcastHostTensorAllocator)

    page_num = 2
    page_size = 2
    root = allocator.allocate(
        (2, page_num, 1, page_size, 1, 2),
        torch.float32,
        "cpu",
    )
    root.copy_(torch.arange(root.numel(), dtype=root.dtype).reshape(root.shape))
    expected_root = root.clone()
    pool = object.__new__(MHATokenToKVPoolHost)
    pool.layout = "page_first_direct"
    pool.page_num = page_num
    pool.page_size = page_size
    pool.size = page_num * page_size
    pool.layer_num = 1
    pool.head_num = 1
    pool.head_dim = 2
    pool.dtype = torch.float32
    pool.mtp_draft_device_pools = ()
    pool.dcp_size = 1
    pool.dcp_rank = 0
    pool.allocator = allocator
    pool.kv_buffer = root

    storage_config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="unused-fallback",
        extra_config=source,
    )
    store = TensorcastStore(storage_config)
    store.register_mem_pool_host(pool)
    session = store.session
    keys = [f"phase6-{scenario}-page"]
    host_indices = torch.tensor([0, 1], dtype=torch.int64)

    if scenario == "terminal":
        store.close()
        store.close()
        assert session.lifecycle_state is RegionSessionLifecycleState.TERMINATED
        assert store.batch_exists(keys) == 0
        assert store.batch_get_v1(keys, host_indices) == [False]
        assert store.batch_set_v1(keys, host_indices) == [False]
        assert torch.equal(root, expected_root)
        root.add_(1)
        assert torch.equal(root, expected_root + 1)
        try:
            RegionBackedArtifactSession.attach(store._session_options)
        except RegionSessionTerminatedError:
            pass
        else:
            raise AssertionError("terminated Session was reattached")
    elif scenario == "failure":
        assert store.batch_set_v1(keys, host_indices) == [True]
        original = store._build_artifact_specs(keys)[0]
        conflicting = ByteArtifactSpec(
            keyspace=original.keyspace,
            engine_key=original.engine_key,
            byte_length=4,
        )
        conflict_root = allocator.allocate((1,), torch.float32, "cpu")
        conflict_transfer = RegionArtifactTransfer(
            artifact=conflicting,
            span=HostMemorySpan.from_tensor(
                conflict_root,
                offset_bytes=0,
                byte_length=4,
            ),
        )
        try:
            session.batch_put_from((conflict_transfer,))
        except RegionSessionFailedError as error:
            first_failure = error.failure
        else:
            raise AssertionError("daemon identity conflict did not fail Session")
        assert session.health is RegionSessionHealth.FAILED
        assert store.batch_exists(keys) == 0
        assert store.batch_get_v1(keys, host_indices) == [False]
        assert store.batch_set_v1(keys, host_indices) == [False]
        assert session.failure == first_failure
        store.close()
        assert session.lifecycle_state is RegionSessionLifecycleState.TERMINATED
        assert torch.equal(root, expected_root)
    else:
        raise AssertionError(f"unknown phase6 scenario {scenario!r}")

    print(f"phase6-real-daemon-{scenario}=ok", flush=True)
    """
)


def _run_real_daemon_child(transfer_mode: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["SGLANG_TENSORCAST_CHILD_TRANSFER_MODE"] = transfer_mode
    return subprocess.run(
        [sys.executable, "-c", _REAL_DAEMON_CHILD],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )


def _run_register_and_exists_child() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _REAL_DAEMON_REGISTER_AND_EXISTS_CHILD],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=os.environ.copy(),
    )


def _run_scratch_round_trip_child() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _REAL_DAEMON_SCRATCH_ROUND_TRIP_CHILD],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=os.environ.copy(),
    )


def _run_allocator_round_trip_child(
    family: str,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["SGLANG_TENSORCAST_CHILD_POOL_FAMILY"] = family
    return subprocess.run(
        [sys.executable, "-c", _REAL_DAEMON_ALLOCATOR_ROUND_TRIP_CHILD],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )


def _run_phase6_child(scenario: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["SGLANG_TENSORCAST_PHASE6_SCENARIO"] = scenario
    return subprocess.run(
        [sys.executable, "-c", _REAL_DAEMON_PHASE6_CHILD],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )


@pytest.mark.skipif(
    os.getenv("SGLANG_TENSORCAST_E2E") != "1",
    reason="set SGLANG_TENSORCAST_E2E=1 with a ready StoreDaemon",
)
def test_real_daemon_early_attach_and_allocator() -> None:
    daemon_address = os.getenv("SGLANG_TENSORCAST_DAEMON_ADDRESS")
    if not daemon_address:
        pytest.fail("SGLANG_TENSORCAST_DAEMON_ADDRESS is required")

    for transfer_mode in ("scratch", "allocator"):
        result = _run_real_daemon_child(transfer_mode)
        assert result.returncode == 0, (
            f"{transfer_mode} child failed\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        assert f"phase2-real-daemon-{transfer_mode}=ok" in result.stdout


@pytest.mark.skipif(
    os.getenv("SGLANG_TENSORCAST_E2E") != "1",
    reason="set SGLANG_TENSORCAST_E2E=1 with a ready StoreDaemon",
)
def test_real_daemon_register_and_exists_miss() -> None:
    if not os.getenv("SGLANG_TENSORCAST_DAEMON_ADDRESS"):
        pytest.fail("SGLANG_TENSORCAST_DAEMON_ADDRESS is required")

    result = _run_register_and_exists_child()
    assert result.returncode == 0, (
        f"phase3 child failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "phase3-real-daemon-register-and-exists-miss=ok" in result.stdout


@pytest.mark.skipif(
    os.getenv("SGLANG_TENSORCAST_E2E") != "1",
    reason="set SGLANG_TENSORCAST_E2E=1 with a ready StoreDaemon",
)
def test_real_daemon_scratch_full_round_trip() -> None:
    if not os.getenv("SGLANG_TENSORCAST_DAEMON_ADDRESS"):
        pytest.fail("SGLANG_TENSORCAST_DAEMON_ADDRESS is required")

    result = _run_scratch_round_trip_child()
    assert result.returncode == 0, (
        f"phase4 child failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "phase4-real-daemon-scratch-full-round-trip=ok" in result.stdout


@pytest.mark.skipif(
    os.getenv("SGLANG_TENSORCAST_E2E") != "1",
    reason="set SGLANG_TENSORCAST_E2E=1 with a ready StoreDaemon",
)
def test_real_daemon_allocator_full_round_trip() -> None:
    if not os.getenv("SGLANG_TENSORCAST_DAEMON_ADDRESS"):
        pytest.fail("SGLANG_TENSORCAST_DAEMON_ADDRESS is required")

    result = _run_allocator_round_trip_child("mha")
    assert result.returncode == 0, (
        f"phase5 MHA child failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "phase5-real-daemon-mha=ok" in result.stdout


@pytest.mark.skipif(
    os.getenv("SGLANG_TENSORCAST_E2E") != "1",
    reason="set SGLANG_TENSORCAST_E2E=1 with a ready StoreDaemon",
)
def test_real_daemon_asymmetric_multi_region() -> None:
    if not os.getenv("SGLANG_TENSORCAST_DAEMON_ADDRESS"):
        pytest.fail("SGLANG_TENSORCAST_DAEMON_ADDRESS is required")

    result = _run_allocator_round_trip_child("asymmetric_mha")
    assert result.returncode == 0, (
        "phase5 asymmetric MHA child failed\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "phase5-real-daemon-asymmetric_mha=ok" in result.stdout


@pytest.mark.skipif(
    os.getenv("SGLANG_TENSORCAST_E2E") != "1",
    reason="set SGLANG_TENSORCAST_E2E=1 with a ready StoreDaemon",
)
def test_real_daemon_terminal_lifecycle() -> None:
    if not os.getenv("SGLANG_TENSORCAST_DAEMON_ADDRESS"):
        pytest.fail("SGLANG_TENSORCAST_DAEMON_ADDRESS is required")

    result = _run_phase6_child("terminal")
    assert result.returncode == 0, (
        "phase6 terminal child failed\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "phase6-real-daemon-terminal=ok" in result.stdout
    assert "TensorCast L3 unavailable:" not in result.stderr


@pytest.mark.skipif(
    os.getenv("SGLANG_TENSORCAST_E2E") != "1",
    reason="set SGLANG_TENSORCAST_E2E=1 with a ready StoreDaemon",
)
def test_real_daemon_failure_latches_adapter() -> None:
    if not os.getenv("SGLANG_TENSORCAST_DAEMON_ADDRESS"):
        pytest.fail("SGLANG_TENSORCAST_DAEMON_ADDRESS is required")

    result = _run_phase6_child("failure")
    assert result.returncode == 0, (
        "phase6 failure child failed\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "phase6-real-daemon-failure=ok" in result.stdout
    output = result.stdout + result.stderr
    assert output.count("TensorCast L3 unavailable:") == 1
    assert "category=session_failed" in output

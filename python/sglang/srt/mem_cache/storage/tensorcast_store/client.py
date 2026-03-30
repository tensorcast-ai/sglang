# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

import atexit
import collections
import hashlib
import logging
import os
import re
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import Callable, Protocol

import torch

from sglang.srt.mem_cache.storage.tensorcast_store.config import (
    TensorcastHiCacheConfig,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TensorcastBatchExistsResult:
    existence_mask: tuple[bool, ...]
    rpc_elapsed_s: float


@dataclass(frozen=True)
class TensorcastBatchTransferResult:
    success_mask: tuple[bool, ...]
    adopted_duplicate_count: int
    pack_elapsed_s: float
    stage_copy_elapsed_s: float
    rpc_elapsed_s: float
    host_fill_elapsed_s: float
    operation_id: str = ""


class TensorcastPageClient(Protocol):
    def artifact_id_for(self, logical_key: str) -> str: ...

    def batch_exists(self, logical_keys: list[str]) -> TensorcastBatchExistsResult: ...

    def batch_put(
        self,
        logical_keys: list[str],
        pages: list[torch.Tensor],
    ) -> TensorcastBatchTransferResult: ...

    def batch_get_into(
        self,
        logical_keys: list[str],
        targets: list[torch.Tensor],
    ) -> TensorcastBatchTransferResult: ...


@dataclass
class _StagingRegionState:
    tensor: torch.Tensor
    region_id: str
    capacity_bytes: int
    mapping_base_offset: int


_CGID_SAFE_PATTERN = re.compile(r"[^-._~A-Za-z0-9]+")


def _cpu_tensor_as_uint8_view(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.reshape(-1)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    return flat.view(torch.uint8)


def _sanitize_cgid_segment(value: str) -> str:
    sanitized = _CGID_SAFE_PATTERN.sub("_", str(value).strip())
    sanitized = sanitized.strip("_")
    return sanitized or "default"


def _compact_cgid_segment(value: str, *, prefix: str, max_len: int = 24) -> str:
    sanitized = _sanitize_cgid_segment(value)
    if len(sanitized) <= max_len:
        return sanitized
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _engine_key_payload(engine_key_prefix: str, logical_key: str) -> bytes:
    prefix = str(engine_key_prefix).encode("utf-8")
    logical_key_text = str(logical_key).strip()
    if logical_key_text and len(logical_key_text) % 2 == 0:
        try:
            key_bytes = bytes.fromhex(logical_key_text)
        except ValueError:
            key_bytes = logical_key_text.encode("utf-8")
    else:
        key_bytes = logical_key_text.encode("utf-8")
    return prefix + b":" + key_bytes


def _resolve_daemon_device_id(local_device_id: int) -> int:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible_devices:
        return int(local_device_id)
    device_tokens = [
        token.strip() for token in visible_devices.split(",") if token.strip()
    ]
    if local_device_id < 0 or local_device_id >= len(device_tokens):
        return int(local_device_id)
    try:
        return int(device_tokens[local_device_id])
    except ValueError:
        return int(local_device_id)


class _StagingRegionManager:
    def __init__(
        self,
        *,
        client,
        device: torch.device,
        export_device_id: int,
        daemon_device_id: int,
        ttl_ms: int,
        name: str,
        get_cuda_memory_handle: Callable[[int, int], bytes],
        get_cuda_memory_handle_with_offset: Callable[[int, int], tuple[bytes, int]],
    ) -> None:
        self._client = client
        self._device = device
        self._export_device_id = int(export_device_id)
        self._daemon_device_id = int(daemon_device_id)
        self._ttl_ms = int(ttl_ms)
        self._name = name
        self._get_cuda_memory_handle = get_cuda_memory_handle
        self._get_cuda_memory_handle_with_offset = get_cuda_memory_handle_with_offset
        self._state: _StagingRegionState | None = None
        self._lock = threading.Lock()

    def ensure_capacity(self, required_bytes: int) -> _StagingRegionState:
        if required_bytes <= 0:
            raise ValueError("required_bytes must be positive")
        with self._lock:
            state = self._state
            if state is not None and state.capacity_bytes >= required_bytes:
                return state
            self._release_locked()
            tensor = torch.empty(
                required_bytes,
                dtype=torch.uint8,
                device=self._device,
            )
            base_ptr_value = int(tensor.data_ptr())
            size_value = int(required_bytes)
            mapping_base_offset = 0
            try:
                handle_bytes, mapping_base_offset = (
                    self._get_cuda_memory_handle_with_offset(
                        self._export_device_id,
                        base_ptr_value,
                    )
                )
            except Exception:  # noqa: BLE001
                handle_bytes = self._get_cuda_memory_handle(
                    self._export_device_id,
                    base_ptr_value,
                )
                mapping_base_offset = 0
            if mapping_base_offset:
                size_value += int(mapping_base_offset)
            handle = self._client.register_vram_region(
                device_id=self._daemon_device_id,
                size_bytes=size_value,
                ttl_ms=self._ttl_ms,
                cuda_ipc_handle=handle_bytes,
                region_name=self._name,
            )
            self._state = _StagingRegionState(
                tensor=tensor,
                region_id=str(handle.region_id),
                capacity_bytes=int(required_bytes),
                mapping_base_offset=int(mapping_base_offset),
            )
            return self._state

    def close(self) -> None:
        with self._lock:
            self._release_locked()

    def _release_locked(self) -> None:
        state = self._state
        self._state = None
        if state is None:
            return
        with suppress(Exception):
            self._client.unregister_vram_region(state.region_id, force=True)


class DefaultTensorcastPageClient:
    def __init__(
        self,
        config: TensorcastHiCacheConfig,
        *,
        layout_id: str,
        engine_key_prefix: str,
    ) -> None:
        import tensorcast as tc
        from tensorcast._c_ext import (
            get_cuda_memory_handle,
            get_cuda_memory_handle_with_offset,
        )
        from tensorcast.api._device import device_uuid_for
        from tensorcast.common.identity import build_byte_artifact_cgid
        from tensorcast.common.selection_contract import build_artifact_selection
        from tensorcast.proto.common.v1 import common_pb2
        from tensorcast.proto.daemon.v2 import store_daemon_pb2

        self._tc = tc
        self._common_pb2 = common_pb2
        self._store_daemon_pb2 = store_daemon_pb2
        self._build_artifact_selection = build_artifact_selection
        self._build_byte_artifact_cgid = build_byte_artifact_cgid
        self._device_uuid_for = device_uuid_for
        self._config = config
        self._layout_id = _compact_cgid_segment(layout_id, prefix="ly")
        self._namespace = _compact_cgid_segment(config.namespace, prefix="ns")
        self._engine = _compact_cgid_segment(config.engine, prefix="en")
        self._engine_key_prefix = str(engine_key_prefix)
        self._artifact_id_cache: dict[str, str] = {}
        self._selection_cache: dict[str, object] = {}
        self._store = tc.Store(config.daemon_address)
        self._client = self._store._runtime.ensure_client()
        current_device = torch.cuda.current_device()
        self._device = torch.device(f"cuda:{current_device}")
        self._export_device_id = int(current_device)
        self._daemon_device_id = _resolve_daemon_device_id(current_device)
        self._device_uuid = str(self._device_uuid_for(current_device))
        region_ttl_ms = int(config.staging_region_ttl_ms)
        self._put_staging = _StagingRegionManager(
            client=self._client,
            device=self._device,
            export_device_id=self._export_device_id,
            daemon_device_id=self._daemon_device_id,
            ttl_ms=region_ttl_ms,
            name="sglang_tensorcast_put_staging",
            get_cuda_memory_handle=get_cuda_memory_handle,
            get_cuda_memory_handle_with_offset=get_cuda_memory_handle_with_offset,
        )
        self._get_staging = _StagingRegionManager(
            client=self._client,
            device=self._device,
            export_device_id=self._export_device_id,
            daemon_device_id=self._daemon_device_id,
            ttl_ms=region_ttl_ms,
            name="sglang_tensorcast_get_staging",
            get_cuda_memory_handle=get_cuda_memory_handle,
            get_cuda_memory_handle_with_offset=get_cuda_memory_handle_with_offset,
        )
        atexit.register(self.close)

    def close(self) -> None:
        self._put_staging.close()
        self._get_staging.close()

    def artifact_id_for(self, logical_key: str) -> str:
        artifact_id = self._artifact_id_cache.get(logical_key)
        if artifact_id is not None:
            return artifact_id
        artifact_id = self._build_byte_artifact_cgid(
            namespace=self._namespace,
            engine=self._engine,
            model_id=self._config.model_id,
            layout_id=self._layout_id,
            engine_key=_engine_key_payload(self._engine_key_prefix, logical_key),
        )
        self._artifact_id_cache[logical_key] = artifact_id
        return artifact_id

    def batch_exists(self, logical_keys: list[str]) -> TensorcastBatchExistsResult:
        if not logical_keys:
            return TensorcastBatchExistsResult(existence_mask=(), rpc_elapsed_s=0.0)
        selections = [
            self._selection_for(self.artifact_id_for(key)) for key in logical_keys
        ]
        started_at = time.perf_counter()
        response = self._client.batch_exists(
            selections=selections,
            timeout_s=float(self._config.batch_exists_timeout_s),
        )
        rpc_elapsed_s = time.perf_counter() - started_at
        outcome_by_artifact = {
            str(outcome.artifact_id): int(outcome.status)
            for outcome in response.outcomes
        }
        ok_status = int(self._store_daemon_pb2.BATCH_ITEM_STATUS_OK)
        miss_status = int(self._store_daemon_pb2.BATCH_ITEM_STATUS_MISS)
        status_counts = collections.Counter(outcome_by_artifact.values())
        existence_mask: list[bool] = []
        for logical_key in logical_keys:
            artifact_id = self.artifact_id_for(logical_key)
            status = outcome_by_artifact.get(artifact_id, miss_status)
            existence_mask.append(status == ok_status)
        unexpected_statuses = {
            status: count
            for status, count in status_counts.items()
            if status not in {ok_status, miss_status}
        }
        if unexpected_statuses:
            status_summary = {
                self._store_daemon_pb2.BatchItemStatus.Name(status): count
                for status, count in unexpected_statuses.items()
            }
            logger.debug(
                "Tensorcast batch_exists unexpected statuses=%s first_key=%s first_artifact_id=%s",
                status_summary,
                logical_keys[0],
                self.artifact_id_for(logical_keys[0]),
            )
        return TensorcastBatchExistsResult(
            existence_mask=tuple(existence_mask),
            rpc_elapsed_s=rpc_elapsed_s,
        )

    def batch_put(
        self,
        logical_keys: list[str],
        pages: list[torch.Tensor],
    ) -> TensorcastBatchTransferResult:
        if len(logical_keys) != len(pages):
            raise ValueError("logical_keys and pages must have the same length")
        if not logical_keys:
            return TensorcastBatchTransferResult(
                success_mask=(),
                adopted_duplicate_count=0,
                pack_elapsed_s=0.0,
                stage_copy_elapsed_s=0.0,
                rpc_elapsed_s=0.0,
                host_fill_elapsed_s=0.0,
            )
        pack_started_at = time.perf_counter()
        packed_pages: list[tuple[str, str, torch.Tensor, int, str]] = []
        total_bytes = 0
        for logical_key, page in zip(logical_keys, pages, strict=True):
            artifact_id = self.artifact_id_for(logical_key)
            page_bytes = _cpu_tensor_as_uint8_view(page)
            byte_length = int(page_bytes.numel())
            digest_hex = hashlib.sha256(memoryview(page_bytes.numpy())).hexdigest()
            packed_pages.append(
                (
                    logical_key,
                    artifact_id,
                    page_bytes,
                    byte_length,
                    digest_hex,
                )
            )
            total_bytes += byte_length
        pack_elapsed_s = time.perf_counter() - pack_started_at
        staging = self._put_staging.ensure_capacity(total_bytes)
        source_layout = self._store_daemon_pb2.TargetLayout(
            layout_kind=self._store_daemon_pb2.TargetLayout.LAYOUT_KIND_COALESCED_UNSPECIFIED,
            index_kind=self._store_daemon_pb2.TargetLayout.INDEX_KIND_CANONICAL_UNSPECIFIED,
            tensor_spec_kind=self._store_daemon_pb2.TargetLayout.TENSOR_SPEC_KIND_OFFSETS,
        )
        storage = source_layout.storages.add()
        storage.storage_id = "storage-0"
        storage.device_id = int(self._daemon_device_id)
        storage.storage_length = int(total_bytes)
        storage.vram_region_id = staging.region_id
        storage.mapping_base_offset = int(staging.mapping_base_offset)
        items = []
        stage_started_at = time.perf_counter()
        cursor = 0
        for _, artifact_id, page_bytes, byte_length, digest_hex in packed_pages:
            staging.tensor[cursor : cursor + byte_length].copy_(
                page_bytes,
                non_blocking=True,
            )
            offset = source_layout.offsets.add()
            offset.name = artifact_id
            offset.storage_id = "storage-0"
            offset.storage_offset = int(cursor)
            offset.logical_length = int(byte_length)
            item = self._store_daemon_pb2.BatchPutIfAbsentFromRegionItem(
                selection=self._selection_for(artifact_id),
                invariant=self._store_daemon_pb2.PutIfAbsentInvariant(
                    layout_id=self._layout_id,
                    byte_length=int(byte_length),
                    payload_digest_alg="sha256",
                    payload_digest_hex=digest_hex,
                ),
            )
            items.append(item)
            cursor += byte_length
        torch.cuda.synchronize(self._device)
        stage_copy_elapsed_s = time.perf_counter() - stage_started_at
        rpc_started_at = time.perf_counter()
        response = self._client.batch_put_if_absent_from_region(
            items=items,
            source_layout=source_layout,
            pid=os.getpid(),
            device_uuid=self._device_uuid,
            operation_id=uuid.uuid4().hex,
            timeout_s=float(self._config.batch_transfer_timeout_s),
        )
        rpc_elapsed_s = time.perf_counter() - rpc_started_at
        ok_status = int(self._store_daemon_pb2.BATCH_ITEM_STATUS_OK)
        outcome_by_artifact = {
            str(outcome.artifact_id): outcome for outcome in response.outcomes
        }
        status_counts = collections.Counter(
            int(outcome.status) for outcome in response.outcomes
        )
        success_mask: list[bool] = []
        adopted_duplicate_count = 0
        for _, artifact_id, _, _, _ in packed_pages:
            outcome = outcome_by_artifact.get(artifact_id)
            status = int(outcome.status) if outcome is not None else 0
            if status == ok_status:
                if (
                    outcome is not None
                    and str(outcome.message).strip().lower() == "joined"
                ):
                    adopted_duplicate_count += 1
                success_mask.append(True)
                continue
            success_mask.append(False)
        failed_statuses = {
            self._store_daemon_pb2.BatchItemStatus.Name(status): count
            for status, count in status_counts.items()
            if status != ok_status
        }
        if failed_statuses:
            failure_messages = []
            for artifact_id, outcome in outcome_by_artifact.items():
                status = int(outcome.status)
                if status == ok_status:
                    continue
                failure_messages.append(
                    f"{artifact_id}:{self._store_daemon_pb2.BatchItemStatus.Name(status)}:{outcome.message}"
                )
                if len(failure_messages) >= 3:
                    break
            logger.debug(
                "Tensorcast batch_put failures statuses=%s first_key=%s first_artifact_id=%s samples=%s",
                failed_statuses,
                logical_keys[0],
                packed_pages[0][1],
                failure_messages,
            )
        return TensorcastBatchTransferResult(
            success_mask=tuple(success_mask),
            adopted_duplicate_count=adopted_duplicate_count,
            pack_elapsed_s=pack_elapsed_s,
            stage_copy_elapsed_s=stage_copy_elapsed_s,
            rpc_elapsed_s=rpc_elapsed_s,
            host_fill_elapsed_s=0.0,
        )

    def batch_get_into(
        self,
        logical_keys: list[str],
        targets: list[torch.Tensor],
    ) -> TensorcastBatchTransferResult:
        if len(logical_keys) != len(targets):
            raise ValueError("logical_keys and targets must have the same length")
        if not logical_keys:
            return TensorcastBatchTransferResult(
                success_mask=(),
                adopted_duplicate_count=0,
                pack_elapsed_s=0.0,
                stage_copy_elapsed_s=0.0,
                rpc_elapsed_s=0.0,
                host_fill_elapsed_s=0.0,
            )
        pack_started_at = time.perf_counter()
        packed_targets: list[tuple[str, str, torch.Tensor, int]] = []
        total_bytes = 0
        for logical_key, target in zip(logical_keys, targets, strict=True):
            artifact_id = self.artifact_id_for(logical_key)
            target_bytes = _cpu_tensor_as_uint8_view(target)
            byte_length = int(target_bytes.numel())
            packed_targets.append((logical_key, artifact_id, target_bytes, byte_length))
            total_bytes += byte_length
        pack_elapsed_s = time.perf_counter() - pack_started_at
        staging = self._get_staging.ensure_capacity(total_bytes)
        target_layout = self._store_daemon_pb2.TargetLayout(
            layout_kind=self._store_daemon_pb2.TargetLayout.LAYOUT_KIND_COALESCED_UNSPECIFIED,
            index_kind=self._store_daemon_pb2.TargetLayout.INDEX_KIND_CANONICAL_UNSPECIFIED,
            tensor_spec_kind=self._store_daemon_pb2.TargetLayout.TENSOR_SPEC_KIND_OFFSETS,
        )
        storage = target_layout.storages.add()
        storage.storage_id = "storage-0"
        storage.device_id = int(self._daemon_device_id)
        storage.storage_length = int(total_bytes)
        storage.vram_region_id = staging.region_id
        storage.mapping_base_offset = int(staging.mapping_base_offset)
        selections = []
        cursor = 0
        for _, artifact_id, _, byte_length in packed_targets:
            selections.append(self._selection_for(artifact_id))
            offset = target_layout.offsets.add()
            offset.name = artifact_id
            offset.storage_id = "storage-0"
            offset.storage_offset = int(cursor)
            offset.logical_length = int(byte_length)
            cursor += byte_length
        operation_id = uuid.uuid4().hex
        rpc_started_at = time.perf_counter()
        response = self._client.batch_get_into_region(
            selections=selections,
            target_layout=target_layout,
            pid=os.getpid(),
            device_uuid=self._device_uuid,
            operation_id=operation_id,
            timeout_s=float(self._config.batch_transfer_timeout_s),
        )
        torch.cuda.synchronize(self._device)
        rpc_elapsed_s = time.perf_counter() - rpc_started_at
        ok_status = int(self._store_daemon_pb2.BATCH_ITEM_STATUS_OK)
        outcome_by_artifact = {
            str(outcome.artifact_id): outcome for outcome in response.outcomes
        }
        status_counts = collections.Counter(
            int(outcome.status) for outcome in response.outcomes
        )
        host_fill_started_at = time.perf_counter()
        cursor = 0
        success_mask: list[bool] = []
        stop_copying = False
        for _, artifact_id, target_bytes, byte_length in packed_targets:
            outcome = outcome_by_artifact.get(artifact_id)
            status = (
                int(outcome.status)
                if outcome is not None
                else int(self._store_daemon_pb2.BATCH_ITEM_STATUS_MISS)
            )
            if stop_copying or status != ok_status:
                stop_copying = True
                success_mask.append(False)
                cursor += byte_length
                continue
            target_bytes.copy_(
                staging.tensor[cursor : cursor + byte_length],
                non_blocking=True,
            )
            success_mask.append(True)
            cursor += byte_length
        torch.cuda.synchronize(self._device)
        host_fill_elapsed_s = time.perf_counter() - host_fill_started_at
        failed_statuses = {
            self._store_daemon_pb2.BatchItemStatus.Name(status): count
            for status, count in status_counts.items()
            if status != ok_status
        }
        if failed_statuses:
            failure_messages = []
            for artifact_id, outcome in outcome_by_artifact.items():
                status = int(outcome.status)
                if status == ok_status:
                    continue
                failure_messages.append(
                    f"{artifact_id}:{self._store_daemon_pb2.BatchItemStatus.Name(status)}:{outcome.message}"
                )
                if len(failure_messages) >= 3:
                    break
            logger.debug(
                "Tensorcast batch_get failures statuses=%s first_key=%s first_artifact_id=%s samples=%s",
                failed_statuses,
                logical_keys[0],
                packed_targets[0][1],
                failure_messages,
            )
        return TensorcastBatchTransferResult(
            success_mask=tuple(success_mask),
            adopted_duplicate_count=0,
            pack_elapsed_s=pack_elapsed_s,
            stage_copy_elapsed_s=0.0,
            rpc_elapsed_s=rpc_elapsed_s,
            host_fill_elapsed_s=host_fill_elapsed_s,
            operation_id=operation_id,
        )

    def _selection_for(self, artifact_id: str):
        selection = self._selection_cache.get(artifact_id)
        if selection is not None:
            cloned = self._common_pb2.ArtifactSelection()
            cloned.CopyFrom(selection)
            return cloned
        selection = self._build_artifact_selection(
            artifact_id=artifact_id,
            canonical_index_bytes=b"",
            layout_index_bytes=None,
            view_spec=None,
            tensor_names=None,
        )
        self._selection_cache[artifact_id] = selection
        cloned = self._common_pb2.ArtifactSelection()
        cloned.CopyFrom(selection)
        return cloned

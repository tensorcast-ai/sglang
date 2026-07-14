from __future__ import annotations

# ruff: noqa: E402

import pytest
from pydantic import ValidationError

from sglang.test.tensorcast.test_support import install_memory_pool_host_stub

install_memory_pool_host_stub()

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.tensorcast_store.config import (
    TensorcastHiCacheConfig,
)


def _storage_config(extra_config: dict[str, object]) -> HiCacheStorageConfig:
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        is_mla_model=False,
        is_page_first_layout=True,
        model_name="Qwen3-32B",
        extra_config=extra_config,
    )


def test_tensorcast_config_defaults_preserve_passive_prefix_share() -> None:
    config = TensorcastHiCacheConfig.from_storage_config(
        _storage_config({"daemon_address": "127.0.0.1:50052"})
    )

    assert config.tensorcast_kv_mode == "passive_prefix_share"
    assert config.background_page_publish_enabled
    assert config.ordinary_storage_prefetch_enabled
    assert config.record_host_residency_for_publish
    assert config.logical_session_id_source == "routing_key"
    assert config.model_id == "Qwen3-32B"
    assert config.model_version == "default"


def test_explicit_mode_applies_safe_defaults() -> None:
    config = TensorcastHiCacheConfig.from_storage_config(
        _storage_config(
            {
                "daemon_address": "127.0.0.1:50052",
                "tensorcast_kv_mode": "explicit_request_transfer",
            }
        )
    )

    assert config.explicit_request_transfer_enabled
    assert not config.background_page_publish_enabled
    assert not config.ordinary_storage_prefetch_enabled
    assert config.record_host_residency_for_publish


@pytest.mark.parametrize(
    "payload, message",
    [
        (
            {
                "daemon_address": "127.0.0.1:50052",
                "tensorcast_kv_mode": "invalid",
            },
            "tensorcast_kv_mode",
        ),
        (
            {
                "daemon_address": "127.0.0.1:50052",
                "logical_session_id_source": "routing_key_then_rid_regex",
            },
            "logical_session_id_rid_regex",
        ),
        (
            {
                "daemon_address": "127.0.0.1:50052",
                "logical_session_id_source": "routing_key_then_rid_regex",
                "logical_session_id_rid_regex": r"^rid-(?P<turn>\d+)$",
            },
            "session_id",
        ),
        (
            {
                "daemon_address": "127.0.0.1:50052",
                "logical_session_id_source": "routing_key_then_rid_regex",
                "logical_session_id_rid_regex": r"(?P<session_id>",
            },
            "invalid logical_session_id_rid_regex",
        ),
        (
            {
                "daemon_address": "127.0.0.1:50052",
                "tensorcast_kv_mode": "explicit_request_transfer",
                "record_host_residency_for_publish": False,
            },
            "record_host_residency_for_publish",
        ),
        (
            {
                "daemon_address": "127.0.0.1:50052",
                "tensorcast_kv_mode": "explicit_request_transfer",
                "background_page_publish": True,
            },
            "background_page_publish",
        ),
    ],
)
def test_tensorcast_config_rejects_invalid_combinations(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        TensorcastHiCacheConfig.from_storage_config(_storage_config(payload))

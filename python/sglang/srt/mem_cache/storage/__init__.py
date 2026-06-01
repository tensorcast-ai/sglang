# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Storage backend module for SGLang HiCache."""

from sglang.srt.mem_cache.storage.tensorcast_store import TensorcastStore

from .backend_factory import StorageBackendFactory

__all__ = [
    "StorageBackendFactory",
    "TensorcastStore",
]

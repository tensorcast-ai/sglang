# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Storage backend module for SGLang HiCache."""

from .backend_factory import StorageBackendFactory
from sglang.srt.mem_cache.storage.tensorcast_store import TensorcastStore

__all__ = [
    "StorageBackendFactory",
    "TensorcastStore",
]

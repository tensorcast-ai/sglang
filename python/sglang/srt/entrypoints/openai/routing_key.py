# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

from typing import Any


def extract_smg_routing_key(raw_request: Any | None) -> str | None:
    if raw_request is None:
        return None
    return raw_request.headers.get("x-smg-routing-key")

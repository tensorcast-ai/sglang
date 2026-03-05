from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_loader.tensorcast_runtime import parse_tensorcast_extra_config
from sglang.srt.model_loader.tensorcast_trace import (
    TracePlan,
    trace_plan_from_json,
    trace_plan_to_json,
)

logger = logging.getLogger(__name__)

_TRACE_PLAN_CACHE_SCHEMA_VERSION = 1


def get_trace_plan_cache_dir(extra_config: dict[str, Any]) -> Path | None:
    cfg = parse_tensorcast_extra_config(extra_config)
    raw = cfg.tensorcast_tp_slice_plan_cache_dir
    if raw is None:
        return None
    raw = str(raw).strip()
    if not raw:
        return None
    return Path(raw)


def build_trace_plan_cache_key(
    *,
    model_config: ModelConfig,
    load_config: LoadConfig,
    extra_config: dict[str, Any],
) -> str:
    try:
        arch = model_config.hf_config.architectures[0]
    except Exception:  # noqa: BLE001
        arch = "unknown_arch"

    tp_rank, tp_world_size, pp_rank, pp_world_size = _try_get_tp_pp()
    if load_config.tp_rank is not None:
        tp_rank = int(load_config.tp_rank)

    payload = {
        "cache_schema_version": _TRACE_PLAN_CACHE_SCHEMA_VERSION,
        "model_path": str(model_config.model_path),
        "revision": None if model_config.revision is None else str(model_config.revision),
        "arch": str(arch),
        "dtype": str(model_config.dtype),
        "quantization": None
        if model_config.quantization is None
        else str(model_config.quantization),
        "model_override_args": getattr(model_config, "model_override_args", {}),
        "tensorcast_model_name": parse_tensorcast_extra_config(extra_config).tensorcast_model_name,
        "tp_world_size": int(tp_world_size),
        "tp_rank": int(tp_rank),
        "pp_world_size": int(pp_world_size),
        "pp_rank": int(pp_rank),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:32]
    safe_arch = str(arch).replace("/", "_").replace(" ", "_")
    return f"{safe_arch}-{digest}"


def load_trace_plan_from_cache(cache_dir: Path, cache_key: str) -> TracePlan | None:
    path = cache_dir / f"{cache_key}.json"
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read tensorcast trace plan cache: %s: %s", path, exc)
        return None

    version = data.get("cache_schema_version")
    if int(version or 0) != _TRACE_PLAN_CACHE_SCHEMA_VERSION:
        logger.info(
            "Ignoring tensorcast trace plan cache with schema version mismatch: %s version=%s expected=%s",
            path,
            version,
            _TRACE_PLAN_CACHE_SCHEMA_VERSION,
        )
        return None

    plan_obj = data.get("trace_plan")
    if not isinstance(plan_obj, dict):
        logger.warning("Invalid tensorcast trace plan cache format: %s", path)
        return None

    try:
        return trace_plan_from_json(plan_obj)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse tensorcast trace plan cache: %s: %s", path, exc)
        return None


def save_trace_plan_to_cache(cache_dir: Path, cache_key: str, trace_plan: TracePlan) -> None:
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to create tensorcast trace plan cache dir: %s: %s", cache_dir, exc)
        return

    path = cache_dir / f"{cache_key}.json"
    tmp = cache_dir / f"{cache_key}.json.tmp"
    data = {
        "cache_schema_version": _TRACE_PLAN_CACHE_SCHEMA_VERSION,
        "trace_plan": trace_plan_to_json(trace_plan),
    }
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, sort_keys=True)
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write tensorcast trace plan cache: %s: %s", path, exc)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass


def _try_get_tp_pp() -> tuple[int, int, int, int]:
    try:
        from sglang.srt.distributed.parallel_state import (
            get_pipeline_model_parallel_rank,
            get_pipeline_model_parallel_world_size,
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        tp_rank = int(get_tensor_model_parallel_rank())
        tp_world_size = int(get_tensor_model_parallel_world_size())
        pp_rank = int(get_pipeline_model_parallel_rank())
        pp_world_size = int(get_pipeline_model_parallel_world_size())
        return tp_rank, tp_world_size, pp_rank, pp_world_size
    except Exception:  # noqa: BLE001
        return 0, 1, 0, 1


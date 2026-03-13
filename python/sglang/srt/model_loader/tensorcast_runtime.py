from __future__ import annotations

import logging
import os
import time
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

logger = logging.getLogger(__name__)

_INIT_LOCK = Lock()
_INIT_KWARGS: dict[str, Any] | None = None


if TYPE_CHECKING:
    from tensorcast.api.store.artifact import Artifact
    from tensorcast.api.store.types import FallbackOptions


class TensorcastExtraConfig(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    # Runtime init
    tensorcast_init_mode: Literal["connect", "create", "auto"] = "auto"
    tensorcast_daemon_address: str | None = None
    tensorcast_daemon_config_path: str | None = None
    tensorcast_show_daemon_logs: bool = False
    tensorcast_global_store_address: str | None = None

    tensorcast_init_max_attempts: int = 1
    tensorcast_init_retry_backoff_s: float = 0.5

    # Artifact keying
    tensorcast_model_name: str | None = None
    tensorcast_key_template: str | None = None
    tensorcast_artifact_key: str | None = None

    # Materialization preference
    tensorcast_allow_disk_fallback: bool = True
    tensorcast_allow_p2p_fallback: bool = True
    tensorcast_verify_checksums: bool = True
    tensorcast_fallback_prefer: Literal["auto", "local", "p2p", "disk"] = "auto"

    tensorcast_get_prefer: Literal["auto", "local", "p2p", "disk"] = "auto"
    tensorcast_export_policy: Literal["never", "auto", "force"] = "auto"
    tensorcast_need_view_data_hash: bool = False

    # Disk fallback publisher behavior (best-effort)
    tensorcast_disk_fallback_auto_put: bool = True
    tensorcast_put_policy: str | None = "durable"
    tensorcast_put_stage_on_gpu: bool = False

    # Trace plan cache (optional)
    tensorcast_tp_slice_plan_cache_dir: str | None = None

    # Online-update rollback preflight policy.
    # False (default): best-effort rollback preflight; update may proceed if
    # current-version key is not readable.
    # True: reject update when current-version rollback key preflight fails.
    tensorcast_update_require_rollback_preflight: bool = False

    # Online-update materialization fallback policy.
    # True (default): on GPU materialization OOM, retry materialization on CPU.
    # This improves reliability at the cost of slower updates.
    tensorcast_update_allow_cpu_fallback: bool = True

    # Bootstrap load materialization fallback policy.
    # True (default): on GPU materialization OOM during initial model load,
    # retry materialization on CPU and then copy into the already-initialized
    # model storages.
    tensorcast_load_allow_cpu_fallback: bool = True


def parse_tensorcast_extra_config(extra_config: dict[str, Any]) -> TensorcastExtraConfig:
    try:
        return TensorcastExtraConfig.model_validate(extra_config)
    except ValidationError as exc:
        raise ValueError(
            "Invalid Tensorcast extra config (model_loader_extra_config)."
        ) from exc


def build_tensorcast_init_kwargs(extra_config: dict[str, Any]) -> dict[str, Any]:
    cfg = parse_tensorcast_extra_config(extra_config)

    kwargs: dict[str, Any] = {
        "mode": cfg.tensorcast_init_mode,
        "show_daemon_logs": cfg.tensorcast_show_daemon_logs,
    }

    if cfg.tensorcast_daemon_address:
        kwargs["address"] = cfg.tensorcast_daemon_address

    if cfg.tensorcast_daemon_config_path and cfg.tensorcast_init_mode in {"create", "auto"}:
        kwargs["daemon_config_path"] = cfg.tensorcast_daemon_config_path

    if cfg.tensorcast_global_store_address:
        kwargs["global_store_mode"] = "connect"
        kwargs["global_store_address"] = cfg.tensorcast_global_store_address
    else:
        kwargs["global_store_mode"] = "none"

    return kwargs


def ensure_tensorcast_runtime_initialized(extra_config: dict[str, Any]) -> None:
    try:
        import tensorcast as tc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Tensorcast runtime requested but `tensorcast` is not importable in this environment. "
            'Install with: `uv pip install "sglang[tensorcast]"` '
            '(or `pip install "sglang[tensorcast]"`).'
        ) from exc

    cfg = parse_tensorcast_extra_config(extra_config)
    init_kwargs = build_tensorcast_init_kwargs(extra_config)

    with _INIT_LOCK:
        global _INIT_KWARGS
        if tc.is_initialized():
            if _INIT_KWARGS is None:
                _INIT_KWARGS = dict(init_kwargs)
            elif init_kwargs != _INIT_KWARGS:
                raise RuntimeError(
                    "Tensorcast runtime already initialized with different settings. "
                    f"existing={_INIT_KWARGS}, requested={init_kwargs}"
                )
            return

        last_exc: Exception | None = None
        for attempt in range(max(1, int(cfg.tensorcast_init_max_attempts))):
            try:
                tc.init(**init_kwargs)
                _INIT_KWARGS = dict(init_kwargs)
                logger.info(
                    "Initialized tensorcast runtime: mode=%s address=%s",
                    init_kwargs.get("mode"),
                    init_kwargs.get("address"),
                )
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt + 1 >= cfg.tensorcast_init_max_attempts:
                    break
                backoff_s = max(0.0, float(cfg.tensorcast_init_retry_backoff_s)) * (
                    2**attempt
                )
                logger.warning(
                    "Tensorcast init failed (attempt %d/%d): %s; retrying in %.2fs",
                    attempt + 1,
                    cfg.tensorcast_init_max_attempts,
                    exc,
                    backoff_s,
                )
                time.sleep(backoff_s)

        raise RuntimeError(
            f"Failed to initialize tensorcast runtime with settings: {init_kwargs}"
        ) from last_exc


def build_tensorcast_fallback_options(extra_config: dict[str, Any]) -> "FallbackOptions":
    import tensorcast as tc

    cfg = parse_tensorcast_extra_config(extra_config)
    return tc.FallbackOptions(
        prefer=cfg.tensorcast_fallback_prefer,
        allow_p2p=cfg.tensorcast_allow_p2p_fallback,
        allow_disk=cfg.tensorcast_allow_disk_fallback,
        verify_checksums=cfg.tensorcast_verify_checksums,
    )


def resolve_tensorcast_artifact_key(
    extra_config: dict[str, Any],
    *,
    weight_version: int | None,
) -> str:
    cfg = parse_tensorcast_extra_config(extra_config)
    if cfg.tensorcast_artifact_key:
        return str(cfg.tensorcast_artifact_key)

    if weight_version is None:
        raise ValueError(
            "Tensorcast artifact key resolution requires weight_version when tensorcast_artifact_key is not set."
        )
    if not cfg.tensorcast_key_template:
        raise ValueError(
            "Tensorcast artifact key resolution requires tensorcast_key_template when tensorcast_artifact_key is not set."
        )
    if not cfg.tensorcast_model_name:
        raise ValueError(
            "Tensorcast artifact key resolution requires tensorcast_model_name when tensorcast_artifact_key is not set."
        )

    try:
        return str(
            cfg.tensorcast_key_template.format(
                model_name=cfg.tensorcast_model_name,
                weight_version=int(weight_version),
            )
        )
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "Failed to format tensorcast_key_template with {model_name} and {weight_version}."
        ) from exc


def resolve_tensorcast_versioned_artifact_key(
    extra_config: dict[str, Any],
    *,
    weight_version: int,
) -> str:
    """Resolve versioned artifact key from template, ignoring tensorcast_artifact_key override."""
    cfg = parse_tensorcast_extra_config(extra_config)
    if not cfg.tensorcast_key_template:
        raise ValueError(
            "Tensorcast versioned artifact key resolution requires tensorcast_key_template."
        )
    if not cfg.tensorcast_model_name:
        raise ValueError(
            "Tensorcast versioned artifact key resolution requires tensorcast_model_name."
        )

    try:
        return str(
            cfg.tensorcast_key_template.format(
                model_name=cfg.tensorcast_model_name,
                weight_version=int(weight_version),
            )
        )
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "Failed to format tensorcast_key_template with {model_name} and {weight_version}."
        ) from exc


def open_tensorcast_artifact_by_key(
    extra_config: dict[str, Any],
    *,
    artifact_key: str,
) -> "Artifact":
    import tensorcast as tc

    ensure_tensorcast_runtime_initialized(extra_config)
    fallback = build_tensorcast_fallback_options(extra_config)

    artifact = tc.artifact(key=str(artifact_key), fallback=fallback)
    artifact.describe()
    return artifact


def iter_exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None:
        cur_id = id(cur)
        if cur_id in seen:
            break
        seen.add(cur_id)
        chain.append(cur)
        cur = cur.__cause__ or cur.__context__
    return chain


def is_materialize_oom_error(exc: BaseException) -> bool:
    patterns = (
        "out of memory",
        "cudaerrormemoryallocation",
        "failed uma gpu allocation",
        "resource_exhausted",
    )
    for err in iter_exception_chain(exc):
        msg = str(err).lower()
        if any(token in msg for token in patterns):
            return True
    return False


def _is_tensorcast_key_not_found(exc: BaseException) -> bool:
    try:
        import grpc  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        grpc = None  # type: ignore[assignment]

    for err in iter_exception_chain(exc):
        msg = str(err).lower()
        if "key not found" in msg and "statuscode.not_found" in msg:
            return True

        if grpc is None:
            continue

        if isinstance(err, grpc.RpcError):
            try:
                code = err.code()
            except Exception:  # noqa: BLE001
                continue
            if code != grpc.StatusCode.NOT_FOUND:
                continue
            try:
                details = str(err.details() or "").lower()
            except Exception:  # noqa: BLE001
                details = ""
            if "key not found" in details:
                return True

    return False


def is_tensorcast_key_not_found(exc: BaseException) -> bool:
    """Public helper for callers that need NOT_FOUND classification."""
    return _is_tensorcast_key_not_found(exc)


def open_tensorcast_artifact_for_bootstrap(
    extra_config: dict[str, Any],
    *,
    artifact_key: str,
    model_path: str | None,
) -> tuple["Artifact", bool]:
    """Open a Tensorcast artifact by key, importing from disk on key-miss.

    Bootstrap convenience only: on ResolveKeyMapping NOT_FOUND and when
    `tensorcast_allow_disk_fallback` is enabled, import the model from local
    `model_path` via `tc.from_disk(path)`.

    NOTE: We do NOT re-open by key after import. Tensorcast's `from_disk` does
    not guarantee key mapping publication (and may be rejected by Global Store
    until after at least one successful materialization/replica registration).
    Callers should continue bootstrap with the returned artifact_id and publish
    key mapping later as a best-effort step.

    Online updates MUST remain key-only (no disk reads).
    """
    try:
        return open_tensorcast_artifact_by_key(extra_config, artifact_key=artifact_key), False
    except Exception as exc:  # noqa: BLE001
        cfg = parse_tensorcast_extra_config(extra_config)
        if not _is_tensorcast_key_not_found(exc):
            raise
        if not cfg.tensorcast_allow_disk_fallback:
            raise
        if model_path is None or not str(model_path).strip():
            raise

        resolved_path = os.fspath(model_path)
        if not os.path.isdir(resolved_path):
            raise ValueError(
                "Tensorcast bootstrap disk fallback requires --model-path to be a local directory. "
                f"got model_path={model_path!r}"
            ) from exc

        import tensorcast as tc

        logger.warning(
            "Tensorcast artifact key not found; importing from disk for bootstrap. key=%s path=%s",
            artifact_key,
            resolved_path,
        )
        try:
            artifact = tc.from_disk(
                resolved_path,
                verify_checksums=cfg.tensorcast_verify_checksums,
                show_progress=True,
            )
        except Exception as import_exc:  # noqa: BLE001
            raise RuntimeError(
                "Tensorcast bootstrap failed to import artifact from disk. "
                f"key={artifact_key!r} path={resolved_path!r}"
            ) from import_exc

        fallback = build_tensorcast_fallback_options(extra_config)
        return artifact.with_fallback(fallback), True

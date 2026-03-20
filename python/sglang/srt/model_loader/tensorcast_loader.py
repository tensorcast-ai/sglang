from __future__ import annotations

import gc
import logging
from typing import Any

import torch
from torch import nn

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_loader.loader import (
    BaseModelLoader,
    _initialize_model,
    device_loading_context,
)
from sglang.srt.model_loader.tensorcast_runtime import (
    build_tensorcast_call_context,
    ensure_tensorcast_runtime_initialized,
    is_materialize_oom_error,
    open_tensorcast_artifact_by_key,
    open_tensorcast_artifact_for_bootstrap,
    parse_tensorcast_extra_config,
    resolve_tensorcast_artifact_key,
)
from sglang.srt.model_loader.tensorcast_trace import (
    CopyPlanEntry,
    Range,
    RangeSpec,
    TracePlan,
    ViewSpec,
    trace_model_load,
)
from sglang.srt.model_loader.tensorcast_trace_cache import (
    build_trace_plan_cache_key,
    get_trace_plan_cache_dir,
    load_trace_plan_from_cache,
    save_trace_plan_to_cache,
)
from sglang.srt.model_loader.utils import post_load_weights, set_default_torch_dtype
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)


class TensorcastLoaderError(RuntimeError):
    """Base error for Tensorcast loader/bootstrap/update paths."""


class TensorcastArtifactError(TensorcastLoaderError):
    """Artifact open/describe/materialize failures."""


class TensorcastTraceError(TensorcastLoaderError):
    """TraceMode/plan generation failures."""


class TensorcastPlanMismatchError(TensorcastLoaderError):
    """Plan <-> model or plan <-> artifact mismatch failures."""


class TensorcastApplyError(TensorcastLoaderError):
    """Copy-plan apply failures after materialization."""


class TensorcastModelLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra = load_config.model_loader_extra_config or {}
        if not isinstance(extra, dict):
            raise ValueError(
                "tensorcast model_loader_extra_config must be a dict, "
                f"got {type(extra)}: {extra}"
            )
        self.extra_config = extra

    def download_model(self, model_config: ModelConfig) -> None:
        # Tensorcast loader does not download HF weights.
        return

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        cfg = parse_tensorcast_extra_config(self.extra_config)
        ensure_tensorcast_runtime_initialized(self.extra_config)
        bootstrap_weight_version = _get_numeric_weight_version_for_bootstrap(cfg)

        artifact_key = resolve_tensorcast_artifact_key(
            self.extra_config,
            weight_version=bootstrap_weight_version,
        )
        allow_materialize_cpu_fallback = bool(cfg.tensorcast_load_allow_cpu_fallback)
        logger.info("Tensorcast bootstrap: opening artifact key=%s", artifact_key)
        try:
            artifact, used_disk_fallback = open_tensorcast_artifact_for_bootstrap(
                self.extra_config,
                artifact_key=artifact_key,
                model_path=model_config.model_path,
            )
            descriptor = artifact.describe()
        except Exception as exc:  # noqa: BLE001
            raise TensorcastArtifactError(
                f"Failed to open/describe tensorcast artifact for bootstrap: key={artifact_key!r}"
            ) from exc

        target_device = torch.device(device_config.device)
        if (
            target_device.type == "cuda"
            and device_config.gpu_id is not None
            and device_config.gpu_id >= 0
        ):
            target_device = torch.device("cuda", device_config.gpu_id)

        trace_plan: TracePlan | None = None
        cache_dir = get_trace_plan_cache_dir(self.extra_config)
        cache_key = build_trace_plan_cache_key(
            model_config=model_config,
            load_config=self.load_config,
            extra_config=self.extra_config,
        )
        if cache_dir is not None:
            trace_plan = load_trace_plan_from_cache(cache_dir, cache_key)
            if trace_plan is not None:
                missing = trace_plan.expected_src_names - set(descriptor.tensor_names)
                if missing:
                    logger.warning(
                        "Ignoring cached tensorcast trace plan due to missing tensors in artifact. "
                        "cache_key=%s missing_tensors=%s",
                        cache_key,
                        sorted(missing),
                    )
                    trace_plan = None

        if trace_plan is None:
            with set_default_torch_dtype(model_config.dtype):
                with torch.device("meta"):
                    meta_model = _initialize_model(model_config, self.load_config)
            try:
                trace_plan = trace_model_load(
                    meta_model,
                    ordered_names=list(descriptor.tensor_names),
                    meta_by_name=descriptor.tensor_metas,
                )
            except Exception as exc:  # noqa: BLE001
                # Best-effort cleanup: avoid leaving meta RoPE modules in the
                # global cache if trace fails.
                _evict_meta_rotary_cache()
                raise TensorcastTraceError(
                    f"Failed to trace model.load_weights for tensorcast slice plan: key={artifact_key!r}"
                ) from exc
            if cache_dir is not None:
                save_trace_plan_to_cache(cache_dir, cache_key, trace_plan)

        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                evicted = _evict_meta_rotary_cache()
                if evicted and _is_tp_pp_rank0_best_effort():
                    logger.info(
                        "Evicted %d meta RoPE cache entries before real model init.",
                        evicted,
                    )
                model = _initialize_model(model_config, self.load_config)
                _assert_no_meta_rotary_cache(model)

            tensor_dict: dict[str, torch.Tensor] | None = None
            try:
                tensor_dict = _materialize_tensor_dict(
                    artifact=artifact,
                    trace_plan=trace_plan,
                    target_device=target_device,
                    extra_config=self.extra_config,
                )
            except Exception as exc:  # noqa: BLE001
                if (
                    allow_materialize_cpu_fallback
                    and target_device.type == "cuda"
                    and is_materialize_oom_error(exc)
                ):
                    logger.warning(
                        "Tensorcast GPU materialization OOM during bootstrap; retrying on CPU. key=%s",
                        artifact_key,
                    )
                    try:
                        tensor_dict = _materialize_tensor_dict(
                            artifact=artifact,
                            trace_plan=trace_plan,
                            target_device=torch.device("cpu"),
                            extra_config=self.extra_config,
                        )
                    except Exception as fallback_exc:  # noqa: BLE001
                        raise TensorcastArtifactError(
                            "Failed to materialize tensor dict from tensorcast artifact "
                            f"after GPU OOM and CPU fallback retry: key={artifact_key!r}"
                        ) from fallback_exc
                    logger.warning(
                        "Tensorcast CPU fallback materialization succeeded during bootstrap. key=%s",
                        artifact_key,
                    )
                else:
                    raise TensorcastArtifactError(
                        f"Failed to materialize tensor dict from tensorcast artifact: key={artifact_key!r}"
                    ) from exc

            try:
                assert tensor_dict is not None
                _apply_copy_plan(model, trace_plan, tensor_dict)
            except TensorcastLoaderError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise TensorcastApplyError(
                    f"Failed to apply tensorcast copy plan: key={artifact_key!r}"
                ) from exc
            finally:
                if tensor_dict is not None:
                    del tensor_dict
                gc.collect()
                if target_device.type == "cuda":
                    torch.cuda.empty_cache()

            _best_effort_publish_after_disk_fallback(
                used_disk_fallback=used_disk_fallback,
                artifact_key=artifact_key,
                model_path=model_config.model_path,
                weight_version=bootstrap_weight_version,
                extra_config=self.extra_config,
            )

            _postprocess_after_loading(model, model_config, target_device)

        return model.eval()


def _evict_meta_rotary_cache() -> int:
    """Evict meta rotary embedding modules from the global RoPE cache.

    `sglang.srt.layers.rotary_embedding.get_rope(...)` caches rotary modules
    globally in `_ROPE_DICT`. Tensorcast builds a meta model first for tracing;
    without eviction, the real model init can reuse meta rotary modules, leaving
    `cos_sin_cache` on meta and breaking runtime CUDA kernels.
    """

    import sys

    rope_mod = sys.modules.get("sglang.srt.layers.rotary_embedding")
    if rope_mod is None:
        return 0

    rope_dict = getattr(rope_mod, "_ROPE_DICT", None)
    if not isinstance(rope_dict, dict):
        return 0

    removed = 0
    for key, rope in list(rope_dict.items()):
        cache = getattr(rope, "cos_sin_cache", None)
        if isinstance(cache, torch.Tensor) and cache.is_meta:
            rope_dict.pop(key, None)
            removed += 1
    return removed


def _is_tp_pp_rank0_best_effort() -> bool:
    try:
        from sglang.srt.distributed import (
            get_pipeline_model_parallel_rank,
            get_tensor_model_parallel_rank,
        )

        return (
            int(get_tensor_model_parallel_rank()) == 0
            and int(get_pipeline_model_parallel_rank()) == 0
        )
    except Exception:  # noqa: BLE001
        return True


def _assert_no_meta_rotary_cache(model: nn.Module) -> None:
    for module in model.modules():
        cache = getattr(module, "cos_sin_cache", None)
        if isinstance(cache, torch.Tensor) and cache.is_meta:
            raise RuntimeError(
                "Detected a RotaryEmbedding with cos_sin_cache on meta after real model init. "
                "This is likely caused by global RoPE cache pollution from the meta trace model. "
                "Ensure meta RoPE cache eviction runs before real init."
            )


def _get_numeric_weight_version_for_bootstrap(cfg: Any) -> int | None:
    if getattr(cfg, "tensorcast_artifact_key", None):
        return None
    raw = get_global_server_args().weight_version
    try:
        return int(str(raw))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "Tensorcast bootstrap requires numeric --weight-version when tensorcast_artifact_key is not set."
        ) from exc




def _materialize_tensor_dict(
    *,
    artifact: Any,
    trace_plan: TracePlan,
    target_device: torch.device,
    extra_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    import tensorcast as tc

    materialize_names = sorted(trace_plan.expected_src_names)
    tensorcast_slices = {
        name: [(rng.dim, slice(rng.start, rng.end))]
        for name, rng in trace_plan.tensorcast_slices.items()
    }
    artifact_tp = artifact.subset(materialize_names).view(slices=tensorcast_slices)
    cfg = parse_tensorcast_extra_config(extra_config)
    options = tc.GetArtifactOptions(
        export_policy=cfg.tensorcast_export_policy,
        need_view_data_hash=cfg.tensorcast_need_view_data_hash,
    )
    ctx = build_tensorcast_call_context(extra_config)
    return artifact_tp.tensor_dict(device=str(target_device), options=options, ctx=ctx)


def _postprocess_after_loading(
    model: nn.Module,
    model_config: ModelConfig,
    target_device: torch.device,
) -> None:
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            with device_loading_context(module, target_device):
                quant_method.process_weights_after_loading(module)
    post_load_weights(model, model_config)


def _best_effort_publish_after_disk_fallback(
    *,
    used_disk_fallback: bool,
    artifact_key: str,
    model_path: str | None,
    weight_version: int | None,
    extra_config: dict[str, Any],
) -> None:
    """Best-effort: publish canonical checkpoint key mapping after disk fallback.

    Motivation:
    - `tc.from_disk(...)` during loader bootstrap may not establish key mapping
      immediately.
    - `WeightPublisher.publish_from_disk(...)` provides a public, canonical-name
      path to publish key mapping for a versioned model key.

    Safety:
    - Best-effort (never raises).
    - Runs on TP/PP rank0 only to avoid duplicate heavy import work.
    """

    if not used_disk_fallback:
        return

    cfg = parse_tensorcast_extra_config(extra_config)
    if not cfg.tensorcast_disk_fallback_auto_put:
        logger.info(
            "Tensorcast disk fallback publish disabled by config. key=%s",
            artifact_key,
        )
        return

    if not _is_tp_pp_rank0_best_effort():
        return

    if not model_path or not str(model_path).strip():
        logger.warning(
            "Tensorcast disk fallback publish skipped: missing local model_path. key=%s",
            artifact_key,
        )
        return
    if weight_version is None:
        logger.warning(
            "Tensorcast disk fallback publish skipped: numeric weight_version is required. key=%s",
            artifact_key,
        )
        return
    if not cfg.tensorcast_model_name or not cfg.tensorcast_key_template:
        logger.warning(
            "Tensorcast disk fallback publish skipped: model_name/key_template is required for WeightPublisher. key=%s",
            artifact_key,
        )
        return

    resolved_path = str(model_path)

    try:
        # If another process already published this key, skip disk import publish.
        open_tensorcast_artifact_by_key(extra_config, artifact_key=artifact_key)
        logger.info(
            "Tensorcast disk fallback publish skipped: artifact key already exists. key=%s",
            artifact_key,
        )
        return
    except Exception:
        # Key not found or daemon unavailable; continue to best-effort publish.
        pass

    try:
        from tensorcast.tools.weight_publisher import WeightPublisher, WeightPublisherConfig

        publish_cfg = WeightPublisherConfig(
            model_name=str(cfg.tensorcast_model_name),
            key_template=str(cfg.tensorcast_key_template),
            policy=str(cfg.tensorcast_put_policy or "durable"),
            from_disk_verify_checksums=bool(cfg.tensorcast_verify_checksums),
            trigger_reload=False,
            verify_key_mapping=True,
            stage_on_gpu=bool(cfg.tensorcast_put_stage_on_gpu),
        )
        logger.warning(
            "Tensorcast disk fallback publish: publishing canonical key mapping via WeightPublisher.publish_from_disk (best-effort). key=%s version=%s path=%s",
            artifact_key,
            weight_version,
            resolved_path,
        )
        publisher = WeightPublisher(publish_cfg)
        artifact_id = publisher.publish_from_disk(
            resolved_path, version=int(weight_version)
        )
        logger.info(
            "Tensorcast disk fallback publish completed (best-effort). key=%s artifact_id=%s",
            artifact_key,
            artifact_id,
        )
    except Exception:
        logger.exception(
            "Tensorcast disk fallback publish failed (best-effort). key=%s",
            artifact_key,
        )


def _apply_copy_plan(
    model: nn.Module,
    trace_plan: TracePlan,
    tensor_dict: dict[str, torch.Tensor],
) -> None:
    dst_by_name: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        dst_by_name[name] = param.data
    for name, buf in model.named_buffers(remove_duplicate=False):
        dst_by_name[name] = buf

    for entry in trace_plan.copy_plan:
        if entry.dst_name not in dst_by_name:
            raise TensorcastPlanMismatchError(
                f"Missing destination tensor in model: {entry.dst_name}"
            )
        dst_base = dst_by_name[entry.dst_name]
        dst_view = dst_base if entry.dst_range is None else _narrow_by_range_spec(
            dst_base, entry.dst_range
        )

        if entry.op == "copy":
            if entry.ckpt_name is None:
                raise TensorcastPlanMismatchError(
                    f"Missing source name for copy into {entry.dst_name}"
                )
            if entry.ckpt_name not in tensor_dict:
                raise TensorcastPlanMismatchError(
                    f"Missing source tensor in materialized tensor_dict: {entry.ckpt_name}"
                )
            src_base = tensor_dict[entry.ckpt_name]
            if entry.ckpt_view is not None:
                src_view = _as_strided_src_view(
                    src_base,
                    entry.ckpt_name,
                    entry.ckpt_view,
                    entry.ckpt_range,
                    trace_plan.src_hull,
                )
            else:
                src_view = _narrow_src_view(
                    src_base,
                    entry.ckpt_name,
                    entry.ckpt_range,
                    trace_plan.src_hull,
                )
            if src_view.ndim == 0 and dst_view.numel() == 1:
                src_view = src_view.reshape(dst_view.shape)
            if tuple(src_view.shape) != tuple(dst_view.shape):
                raise TensorcastApplyError(
                    f"Copy shape mismatch for {entry.ckpt_name} -> {entry.dst_name}: "
                    f"{tuple(src_view.shape)} vs {tuple(dst_view.shape)}"
                )
            dst_view.copy_(src_view)
            continue

        if entry.op == "fill":
            if entry.fill_value is not None:
                fill_value = entry.fill_value
            else:
                if entry.ckpt_name is None:
                    raise TensorcastPlanMismatchError(
                        f"Missing fill source for {entry.dst_name}"
                    )
                if entry.ckpt_name not in tensor_dict:
                    raise TensorcastPlanMismatchError(
                        f"Missing source tensor in materialized tensor_dict: {entry.ckpt_name}"
                    )
                src_base = tensor_dict[entry.ckpt_name]
                if entry.ckpt_view is not None:
                    src_view = _as_strided_src_view(
                        src_base,
                        entry.ckpt_name,
                        entry.ckpt_view,
                        entry.ckpt_range,
                        trace_plan.src_hull,
                    )
                else:
                    src_view = _narrow_src_view(
                        src_base,
                        entry.ckpt_name,
                        entry.ckpt_range,
                        trace_plan.src_hull,
                    )
                if src_view.numel() != 1:
                    raise TensorcastApplyError(
                        f"Fill source {entry.ckpt_name} is not scalar: numel={src_view.numel()}"
                    )
                fill_value = src_view.reshape(()).item()
            dst_view.fill_(fill_value)
            continue

        raise TensorcastApplyError(
            f"Unknown tensorcast plan op {entry.op} for {entry.dst_name}"
        )


def _iter_ranges(range_spec: RangeSpec) -> tuple[Range, ...]:
    if isinstance(range_spec, Range):
        return (range_spec,)
    return range_spec.ranges


def _narrow_by_range_spec(tensor: torch.Tensor, range_spec: RangeSpec) -> torch.Tensor:
    out = tensor
    for rng in _iter_ranges(range_spec):
        out = out.narrow(rng.dim, rng.start, rng.end - rng.start)
    return out


def _narrow_src_view(
    src_base: torch.Tensor,
    ckpt_name: str,
    ckpt_range: RangeSpec | None,
    src_hull: dict[str, Range],
) -> torch.Tensor:
    if ckpt_range is None:
        return src_base
    if isinstance(ckpt_range, Range):
        hull = src_hull.get(ckpt_name)
        if hull is not None:
            if hull.dim != ckpt_range.dim:
                raise RuntimeError(f"Slice dim mismatch for {ckpt_name}")
            rel_start = ckpt_range.start - hull.start
            length = ckpt_range.end - ckpt_range.start
            return src_base.narrow(hull.dim, rel_start, length)
    return _narrow_by_range_spec(src_base, ckpt_range)


def _as_strided_src_view(
    src_base: torch.Tensor,
    ckpt_name: str,
    ckpt_view: ViewSpec,
    ckpt_range: RangeSpec | None,
    src_hull: dict[str, Range],
) -> torch.Tensor:
    def _required_storage_numel(
        size: tuple[int, ...], stride: tuple[int, ...], storage_offset: int
    ) -> int:
        if len(size) != len(stride):
            raise TensorcastApplyError(
                f"Invalid ViewSpec for {ckpt_name}: size/stride rank mismatch "
                f"{len(size)} vs {len(stride)}"
            )
        if not size:
            return storage_offset + 1
        max_index = storage_offset
        for dim_size, dim_stride in zip(size, stride):
            if dim_size <= 0:
                return storage_offset + 1
            max_index += (int(dim_size) - 1) * int(dim_stride)
        return max_index + 1

    def _numel(size: tuple[int, ...]) -> int:
        total = 1
        for dim_size in size:
            total *= int(dim_size)
        return total

    offset = int(ckpt_view.storage_offset)
    hull = src_hull.get(ckpt_name)
    if hull is not None:
        stride_dim = src_base.stride()[hull.dim]
        offset -= int(hull.start) * int(stride_dim)

    available_numel = int(src_base.untyped_storage().nbytes() // src_base.element_size())
    required_numel = _required_storage_numel(
        ckpt_view.size, ckpt_view.stride, offset
    )

    # Tensorcast may have already materialized a sliced hull as a compact tensor.
    # In that case, ViewSpec strides recorded from the original full checkpoint
    # can exceed compact storage bounds. Prefer a hull-relative narrow fallback
    # before failing hard.
    if offset < 0 or required_numel > available_numel:
        if ckpt_range is not None:
            try:
                narrowed = _narrow_src_view(src_base, ckpt_name, ckpt_range, src_hull)
                if tuple(narrowed.shape) == tuple(ckpt_view.size):
                    logger.debug(
                        "Tensorcast as_strided fallback to narrow for %s: "
                        "required_numel=%d available_numel=%d offset=%d",
                        ckpt_name,
                        required_numel,
                        available_numel,
                        offset,
                    )
                    return narrowed
            except Exception:  # noqa: BLE001
                pass

        if src_base.numel() == _numel(ckpt_view.size):
            logger.debug(
                "Tensorcast as_strided fallback to reshape for %s: "
                "required_numel=%d available_numel=%d offset=%d",
                ckpt_name,
                required_numel,
                available_numel,
                offset,
            )
            return src_base.reshape(ckpt_view.size)

        raise TensorcastApplyError(
            "Tensorcast source view is out of bounds after slice materialization: "
            f"name={ckpt_name!r} size={ckpt_view.size} stride={ckpt_view.stride} "
            f"storage_offset={offset} required_numel={required_numel} "
            f"available_numel={available_numel}"
        )

    return src_base.as_strided(
        size=ckpt_view.size,
        stride=ckpt_view.stride,
        storage_offset=offset,
    )

from __future__ import annotations

import dataclasses
import logging
import weakref
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
from torch.utils._python_dispatch import TorchDispatchMode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Range:
    dim: int
    start: int
    end: int


@dataclass(frozen=True)
class MultiRange:
    ranges: tuple[Range, ...]


RangeSpec = Range | MultiRange


@dataclass(frozen=True)
class ViewSpec:
    size: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int


def _view_spec_from_tensor(tensor: torch.Tensor) -> ViewSpec:
    return ViewSpec(
        size=tuple(int(v) for v in tensor.size()),
        stride=tuple(int(v) for v in tensor.stride()),
        storage_offset=int(tensor.storage_offset()),
    )


@dataclass(frozen=True)
class CopyPlanEntry:
    op: str
    ckpt_name: str | None
    ckpt_range: RangeSpec | None
    ckpt_view: ViewSpec | None
    dst_name: str
    dst_range: RangeSpec | None
    fill_value: int | float | None = None


@dataclass
class TracePlan:
    copy_plan: list[CopyPlanEntry]
    expected_src_names: set[str]
    expected_dst_names: set[str]
    unsliceable_src_names: set[str]
    tensorcast_slices: dict[str, Range]
    src_hull: dict[str, Range]


class ScalarProxy(float):
    __slots__ = ("ckpt_name", "ckpt_range", "ckpt_view")
    ckpt_name: str
    ckpt_range: RangeSpec | None
    ckpt_view: ViewSpec | None

    def __new__(
        cls,
        value: float,
        ckpt_name: str,
        ckpt_range: RangeSpec | None,
        ckpt_view: ViewSpec | None,
    ):
        obj = float.__new__(cls, value)
        obj.ckpt_name = ckpt_name
        obj.ckpt_range = ckpt_range
        obj.ckpt_view = ckpt_view
        return obj


@dataclass(frozen=True)
class _CatPart:
    kind: str  # "ckpt" or "const"
    length: int
    base_dim: int | None = None
    ckpt_name: str | None = None
    ckpt_range: RangeSpec | None = None
    fill_value: int | float | None = None


@dataclass(frozen=True)
class _CatMeta:
    dim: int
    parts: tuple[_CatPart, ...]


@dataclass(frozen=True)
class _StackPart:
    kind: str  # "ckpt" or "const"
    ckpt_name: str | None = None
    ckpt_range: RangeSpec | None = None
    fill_value: int | float | None = None


@dataclass(frozen=True)
class _StackMeta:
    dim: int
    parts: tuple[_StackPart, ...]


class TraceMode(TorchDispatchMode):
    _PASSTHROUGH = {
        "aten::view",
        "aten::reshape",
        "aten::_reshape_alias",
        "aten::view_as",
        "aten::reshape_as",
        "aten::alias",
        "aten::detach",
        "aten::contiguous",
        "aten::clone",
        "aten::flatten",
        "aten::to",
        "aten::_to_copy",
        "aten::type_as",
    }
    _COPYING_PASSTHROUGH = {
        "aten::contiguous",
        "aten::clone",
        "aten::to",
        "aten::_to_copy",
        "aten::type_as",
    }
    _SLICE_OPS = {
        "aten::slice",
        "aten::select",
        "aten::narrow",
        "aten::split",
        "aten::split_with_sizes",
        "aten::chunk",
        "aten::unbind",
    }

    def __init__(
        self,
        *,
        src_registry: dict[int, str],
        dst_registry: dict[int, str],
    ) -> None:
        super().__init__()
        self._src_registry = src_registry
        self._dst_registry = dst_registry
        # id(tensor) -> (weakref(tensor), base_id, range, dim_map)
        # dim_map maps current tensor dims to base tensor dims.
        self._view_meta: dict[
            int,
            tuple[
                weakref.ReferenceType[torch.Tensor],
                int,
                RangeSpec | None,
                tuple[int, ...] | None,
            ],
        ] = {}
        self._cat_meta: dict[int, _CatMeta] = {}
        self._stack_meta: dict[int, _StackMeta] = {}
        self._const_meta: dict[int, int | float] = {}
        self._passthrough_origin: dict[int, weakref.ReferenceType[torch.Tensor]] = {}

        self.copy_plan: list[CopyPlanEntry] = []
        self.expected_src_names: set[str] = set()
        self.expected_dst_names: set[str] = set()
        self.unsliceable_src_names: set[str] = set()
        self._warned_unsupported_ops: set[str] = set()

        self._prev_tensor_item: Any | None = None
        self._pending_scalar_proxy: ScalarProxy | None = None

    def __enter__(self):
        super().__enter__()

        self._prev_tensor_item = torch.Tensor.item

        def _item_with_trace(tensor: torch.Tensor, *args, **kwargs):
            prev_tensor_item = self._prev_tensor_item
            if prev_tensor_item is None:
                raise RuntimeError("tensorcast trace mode is not initialized")
            value = prev_tensor_item(tensor, *args, **kwargs)
            if not isinstance(value, (int, float)):
                return value
            base_id, base_range = self._resolve_base_and_range(tensor)
            ckpt_name = self._src_registry.get(base_id)
            if ckpt_name is None:
                return value
            proxy = ScalarProxy(
                float(value),
                ckpt_name,
                base_range,
                _view_spec_from_tensor(self._resolve_passthrough_origin(tensor)),
            )
            self._pending_scalar_proxy = proxy
            return proxy

        torch.Tensor.item = _item_with_trace
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            if self._prev_tensor_item is not None:
                torch.Tensor.item = self._prev_tensor_item
                self._prev_tensor_item = None
            self._pending_scalar_proxy = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func._schema.name

        if name == "aten::copy_":
            dst = args[0]
            src = args[1]
            self._record_copy(dst, src)
            return dst

        if name == "aten::_local_scalar_dense":
            tensor = args[0]
            base_id, base_range = self._resolve_base_and_range(tensor)
            if base_id not in self._src_registry:
                raise RuntimeError("Trace encountered scalar from unknown source tensor")
            ckpt_name = self._src_registry[base_id]
            proxy = ScalarProxy(
                0.0,
                ckpt_name,
                base_range,
                _view_spec_from_tensor(self._resolve_passthrough_origin(tensor)),
            )
            self._pending_scalar_proxy = proxy
            return proxy

        if name == "aten::fill_":
            dst = args[0]
            value = args[1]
            if isinstance(value, ScalarProxy):
                self._record_fill_from_ckpt(dst, value)
                self._pending_scalar_proxy = None
            elif isinstance(value, (int, float)):
                pending = self._pending_scalar_proxy
                if pending is not None and float(value) == float(pending):
                    self._record_fill_from_ckpt(dst, pending)
                else:
                    self._record_fill_const(dst, value)
                self._pending_scalar_proxy = None
            else:
                raise RuntimeError("Trace encountered fill_ with unsupported value type")
            return dst

        if name in {"aten::zeros", "aten::zeros_like"}:
            result = func(*args, **kwargs)
            if isinstance(result, torch.Tensor):
                self._const_meta[id(result)] = 0
            return result

        if name in {"aten::transpose", "aten::t"}:
            return self._handle_transpose(name, func, args, kwargs)

        if name == "aten::permute":
            return self._handle_permute(func, args, kwargs)

        if name == "aten::cat":
            return self._handle_cat(func, args, kwargs)

        if name == "aten::stack":
            return self._handle_stack(func, args, kwargs)

        if name in self._PASSTHROUGH:
            result = func(*args, **kwargs)
            if name in self._COPYING_PASSTHROUGH and isinstance(args[0], torch.Tensor):
                self._record_passthrough_origin(result, args[0])
            return self._propagate_view(result, args[0])

        if name in self._SLICE_OPS:
            return self._handle_slice_op(name, func, args, kwargs)

        if self._has_tracked_tensor(args, kwargs):
            return self._handle_unsupported_op(name, func, args, kwargs)
        return func(*args, **kwargs)

    @staticmethod
    def _iter_tensors(obj: object) -> Iterable[torch.Tensor]:
        if isinstance(obj, torch.Tensor):
            yield obj
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                if isinstance(item, torch.Tensor):
                    yield item

    @staticmethod
    def _ultimate_base(tensor: torch.Tensor) -> torch.Tensor:
        cur = tensor
        for _ in range(32):
            base = getattr(cur, "_base", None)
            if base is None:
                return cur
            cur = base
        return cur

    def _handle_unsupported_op(self, op_name: str, func, args, kwargs):
        tracked_inputs: list[torch.Tensor] = []
        for obj in list(args) + list(kwargs.values()):
            tracked_inputs.extend(list(self._iter_tensors(obj)))

        ckpt_names: set[str] = set()
        tracked_base_ids: set[int] = set()
        for t in tracked_inputs:
            base_id, _, _ = self._resolve_base_and_meta(t)
            tracked_base_ids.add(base_id)
            ckpt_name = self._src_registry.get(base_id)
            if ckpt_name is not None:
                ckpt_names.add(ckpt_name)

        if ckpt_names:
            self.unsliceable_src_names.update(ckpt_names)
            if op_name not in self._warned_unsupported_ops:
                logger.warning(
                    "Unsupported op in tensorcast trace: %s. "
                    "Falling back to full materialization (no artifact.view slices) for sources=%s",
                    op_name,
                    sorted(ckpt_names),
                )
                self._warned_unsupported_ops.add(op_name)

        result = func(*args, **kwargs)

        if not tracked_inputs:
            return result

        if len(tracked_base_ids) != 1:
            raise RuntimeError(
                f"Unsupported op in tensorcast trace: {op_name}. "
                "The op consumed multiple tracked tensors; this cannot be safely "
                "treated as a view-only fallback."
            )

        source = tracked_inputs[0]
        base_source = self._ultimate_base(source)
        base_id, _, _ = self._resolve_base_and_meta(source)

        result_tensors = list(self._iter_tensors(result))
        if not result_tensors:
            return result

        for out in result_tensors:
            if self._ultimate_base(out) is not base_source:
                raise RuntimeError(
                    f"Unsupported op in tensorcast trace: {op_name}. "
                    "Fallback requires outputs to share storage with the checkpoint tensor, "
                    "but this op produced a new tensor."
                )
            # Mark as unsliceable: we may still copy correct views using ViewSpec,
            # but we must not attempt to derive 1-D slice plans for Tensorcast.
            self._assign_view_meta(out, base_id, None, None)

        return result

    def _lookup_view_meta(
        self, tensor: torch.Tensor
    ) -> tuple[int, RangeSpec | None, tuple[int, ...] | None] | None:
        tid = id(tensor)
        item = self._view_meta.get(tid)
        if item is None:
            return None
        wr, base_id, base_range, dim_map = item
        if wr() is not tensor:
            self._view_meta.pop(tid, None)
            return None
        return base_id, base_range, dim_map

    def _has_tracked_tensor(self, args, kwargs) -> bool:
        for obj in list(args) + list(kwargs.values()):
            if isinstance(obj, torch.Tensor):
                if self._lookup_view_meta(obj) is not None:
                    return True
                if id(obj) in self._src_registry or id(obj) in self._dst_registry:
                    return True
        return False

    def _resolve_base_and_meta(
        self, tensor: torch.Tensor
    ) -> tuple[int, RangeSpec | None, tuple[int, ...] | None]:
        meta = self._lookup_view_meta(tensor)
        if meta is not None:
            return meta
        base = getattr(tensor, "_base", None)
        if base is not None:
            base_id, base_range, dim_map = self._resolve_base_and_meta(base)
            if tensor.numel() != base.numel():
                raise RuntimeError("Trace encountered untracked view that changes numel")
            if dim_map is not None and tensor.dim() != len(dim_map):
                dim_map = None
            return base_id, base_range, dim_map
        return id(tensor), None, tuple(range(tensor.dim()))

    def _resolve_base_and_range(self, tensor: torch.Tensor) -> tuple[int, RangeSpec | None]:
        base_id, base_range, _ = self._resolve_base_and_meta(tensor)
        return base_id, base_range

    def _propagate_view(self, result: Any, source: torch.Tensor):
        base_id, base_range, dim_map = self._resolve_base_and_meta(source)

        def _map_for_result(t: torch.Tensor) -> tuple[int, ...] | None:
            if dim_map is None:
                return None
            if t.dim() != len(dim_map):
                return None
            return dim_map

        if isinstance(result, torch.Tensor):
            self._view_meta[id(result)] = (
                weakref.ref(result),
                base_id,
                base_range,
                _map_for_result(result),
            )
        elif isinstance(result, (list, tuple)):
            for item in result:
                if isinstance(item, torch.Tensor):
                    self._view_meta[id(item)] = (
                        weakref.ref(item),
                        base_id,
                        base_range,
                        _map_for_result(item),
                    )
        return result

    def _record_passthrough_origin(self, result: Any, source: torch.Tensor) -> None:
        if isinstance(result, torch.Tensor):
            self._passthrough_origin[id(result)] = weakref.ref(source)
            return
        if isinstance(result, (list, tuple)):
            for item in result:
                if isinstance(item, torch.Tensor):
                    self._passthrough_origin[id(item)] = weakref.ref(source)

    def _resolve_passthrough_origin(self, tensor: torch.Tensor) -> torch.Tensor:
        cur = tensor
        for _ in range(8):
            ref = self._passthrough_origin.get(id(cur))
            if ref is None:
                return cur
            origin = ref()
            if origin is None:
                return cur
            cur = origin
        return cur

    def _assign_view_meta(
        self,
        tensor: torch.Tensor,
        base_id: int,
        base_range: RangeSpec | None,
        dim_map: tuple[int, ...] | None,
    ) -> None:
        if dim_map is not None and tensor.dim() != len(dim_map):
            dim_map = None
        self._view_meta[id(tensor)] = (weakref.ref(tensor), base_id, base_range, dim_map)

    def _handle_transpose(self, name: str, func, args, kwargs):
        tensor = args[0]
        if name == "aten::t":
            dim0, dim1 = 0, 1
        else:
            dim0 = args[1] if len(args) > 1 else kwargs.get("dim0")
            dim1 = args[2] if len(args) > 2 else kwargs.get("dim1")
        dim0 = self._normalize_dim(int(dim0), tensor.dim())
        dim1 = self._normalize_dim(int(dim1), tensor.dim())

        base_id, base_range, dim_map = self._resolve_base_and_meta(tensor)
        result = func(*args, **kwargs)
        if not isinstance(result, torch.Tensor):
            return result
        if dim_map is None:
            new_dim_map = None
        else:
            dims = list(dim_map)
            dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
            new_dim_map = tuple(dims)
        self._assign_view_meta(result, base_id, base_range, new_dim_map)
        return result

    def _handle_permute(self, func, args, kwargs):
        tensor = args[0]
        dims = args[1] if len(args) > 1 else kwargs.get("dims")
        base_id, base_range, dim_map = self._resolve_base_and_meta(tensor)
        result = func(*args, **kwargs)
        if not isinstance(result, torch.Tensor):
            return result
        if dim_map is None:
            new_dim_map = None
        else:
            order = [self._normalize_dim(int(d), tensor.dim()) for d in list(dims)]
            new_dim_map = tuple(dim_map[d] for d in order)
        self._assign_view_meta(result, base_id, base_range, new_dim_map)
        return result

    def _handle_cat(self, func, args, kwargs):
        tensors = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        if not isinstance(tensors, (list, tuple)):
            return func(*args, **kwargs)
        first_tensor = next((t for t in tensors if isinstance(t, torch.Tensor)), None)
        if first_tensor is None:
            return func(*args, **kwargs)
        dim = self._normalize_dim(int(dim), first_tensor.dim())

        any_tracked = False
        parts: list[_CatPart] = []
        for t in tensors:
            if not isinstance(t, torch.Tensor):
                raise RuntimeError("Trace encountered cat with non-tensor input")
            length = int(t.size(dim))
            base_id, base_range, dim_map = self._resolve_base_and_meta(t)
            if base_id in self._src_registry:
                any_tracked = True
                parts.append(
                    _CatPart(
                        kind="ckpt",
                        length=length,
                        base_dim=self._map_dim_to_base(dim_map, dim),
                        ckpt_name=self._src_registry[base_id],
                        ckpt_range=base_range,
                    )
                )
            else:
                const = self._const_meta.get(base_id)
                if const is None:
                    const = self._const_meta.get(id(t))
                if const is None:
                    raise RuntimeError(
                        "Trace encountered cat with an untracked input tensor. "
                        "Only checkpoint tensors and traced constants (e.g. zeros) are supported."
                    )
                any_tracked = True
                parts.append(_CatPart(kind="const", length=length, fill_value=const))

        result = func(*args, **kwargs)
        if any_tracked and isinstance(result, torch.Tensor):
            self._cat_meta[id(result)] = _CatMeta(dim=dim, parts=tuple(parts))
        return result

    def _handle_stack(self, func, args, kwargs):
        tensors = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        if not isinstance(tensors, (list, tuple)):
            return func(*args, **kwargs)
        first_tensor = next((t for t in tensors if isinstance(t, torch.Tensor)), None)
        if first_tensor is None:
            return func(*args, **kwargs)
        dim = self._normalize_dim(int(dim), first_tensor.dim() + 1)

        any_tracked = False
        parts: list[_StackPart] = []
        for t in tensors:
            if not isinstance(t, torch.Tensor):
                raise RuntimeError("Trace encountered stack with non-tensor input")
            base_id, base_range, _ = self._resolve_base_and_meta(t)
            if base_id in self._src_registry:
                any_tracked = True
                parts.append(
                    _StackPart(
                        kind="ckpt",
                        ckpt_name=self._src_registry[base_id],
                        ckpt_range=base_range,
                    )
                )
                continue

            const = self._const_meta.get(base_id)
            if const is None:
                const = self._const_meta.get(id(t))
            if const is None:
                raise RuntimeError(
                    "Trace encountered stack with an untracked input tensor. "
                    "Only checkpoint tensors and traced constants (e.g. zeros) are supported."
                )
            any_tracked = True
            parts.append(_StackPart(kind="const", fill_value=const))

        result = func(*args, **kwargs)
        if any_tracked and isinstance(result, torch.Tensor):
            self._stack_meta[id(result)] = _StackMeta(dim=dim, parts=tuple(parts))
        return result

    def _record_copy(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        if dst.numel() == 0 or src.numel() == 0:
            return
        src_base, src_range = self._resolve_base_and_range(src)
        if src_base in self._cat_meta:
            self._record_copy_from_cat(dst, src, src_base, src_range)
            return
        if src_base in self._stack_meta:
            self._record_copy_from_stack(dst, src, src_base, src_range)
            return

        dst_base, dst_range = self._resolve_base_and_range(dst)
        if src_base not in self._src_registry:
            raise RuntimeError("Trace encountered copy_ from unknown source")
        if dst_base not in self._dst_registry:
            raise RuntimeError("Trace encountered copy_ to unknown dst")
        ckpt_name = self._src_registry[src_base]
        dst_name = self._dst_registry[dst_base]

        if not (src.ndim == 0 and dst.numel() == 1) and tuple(src.shape) != tuple(dst.shape):
            raise RuntimeError(
                f"Shape mismatch in trace copy_ for {ckpt_name} -> {dst_name}: "
                f"{tuple(src.shape)} vs {tuple(dst.shape)}"
            )

        self.copy_plan.append(
            CopyPlanEntry(
                op="copy",
                ckpt_name=ckpt_name,
                ckpt_range=src_range,
                ckpt_view=_view_spec_from_tensor(self._resolve_passthrough_origin(src)),
                dst_name=dst_name,
                dst_range=dst_range,
            )
        )
        self.expected_src_names.add(ckpt_name)
        self.expected_dst_names.add(dst_name)

    def _record_copy_from_cat(
        self,
        dst: torch.Tensor,
        src: torch.Tensor,
        src_base: int,
        src_range: RangeSpec | None,
    ) -> None:
        if not (src.ndim == 0 and dst.numel() == 1) and tuple(src.shape) != tuple(dst.shape):
            dst_base, _ = self._resolve_base_and_range(dst)
            dst_name = self._dst_registry.get(dst_base, "<unknown>")
            raise RuntimeError(
                f"Shape mismatch in trace copy_ for cat -> {dst_name}: "
                f"{tuple(src.shape)} vs {tuple(dst.shape)}"
            )

        dst_base, dst_range, dst_dim_map = self._resolve_base_and_meta(dst)
        if dst_base not in self._dst_registry:
            raise RuntimeError("Trace encountered copy_ to unknown dst")
        dst_name = self._dst_registry[dst_base]

        cat = self._cat_meta[src_base]
        cat_dim = cat.dim
        dst_base_dim = self._map_dim_to_base(dst_dim_map, cat_dim)

        total_src_len = sum(int(p.length) for p in cat.parts)
        src_cat = self._range_for_dim(src_range, cat_dim)
        src_start = 0 if src_cat is None else src_cat.start
        src_end = total_src_len if src_cat is None else src_cat.end

        dst_cat = self._range_for_dim(dst_range, dst_base_dim)
        dst_base_start = 0 if dst_cat is None else dst_cat.start

        offset = 0
        for part in cat.parts:
            part_start = offset
            part_end = offset + int(part.length)
            offset = part_end
            overlap_start = max(part_start, src_start)
            overlap_end = min(part_end, src_end)
            if overlap_start >= overlap_end:
                continue

            overlap_len = overlap_end - overlap_start
            dst_seg_start = dst_base_start + (overlap_start - src_start)
            dst_seg_end = dst_seg_start + overlap_len
            dst_seg_range = self._set_absolute_range(
                dst_range, dst_base_dim, dst_seg_start, dst_seg_end
            )

            if part.kind == "ckpt":
                if part.ckpt_name is None:
                    raise RuntimeError("Internal error: missing ckpt_name for cat part")
                if part.base_dim is None:
                    raise RuntimeError("Internal error: missing base_dim for cat part")
                ckpt_base_range = part.ckpt_range
                part_rel_start = overlap_start - part_start
                part_rel_end = overlap_end - part_start
                ckpt_existing = self._range_for_dim(ckpt_base_range, part.base_dim)
                ckpt_base_start = 0 if ckpt_existing is None else ckpt_existing.start
                ckpt_seg_range = self._set_absolute_range(
                    ckpt_base_range,
                    part.base_dim,
                    ckpt_base_start + part_rel_start,
                    ckpt_base_start + part_rel_end,
                )
                self.copy_plan.append(
                    CopyPlanEntry(
                        op="copy",
                        ckpt_name=part.ckpt_name,
                        ckpt_range=ckpt_seg_range,
                        ckpt_view=None,
                        dst_name=dst_name,
                        dst_range=dst_seg_range,
                    )
                )
                self.expected_src_names.add(part.ckpt_name)
                self.expected_dst_names.add(dst_name)
            elif part.kind == "const":
                if part.fill_value is None:
                    raise RuntimeError("Internal error: missing fill_value for cat const")
                self.copy_plan.append(
                    CopyPlanEntry(
                        op="fill",
                        ckpt_name=None,
                        ckpt_range=None,
                        ckpt_view=None,
                        dst_name=dst_name,
                        dst_range=dst_seg_range,
                        fill_value=part.fill_value,
                    )
                )
                self.expected_dst_names.add(dst_name)
            else:
                raise RuntimeError(f"Internal error: unknown cat part kind {part.kind!r}")

    def _record_copy_from_stack(
        self,
        dst: torch.Tensor,
        src: torch.Tensor,
        src_base: int,
        src_range: RangeSpec | None,
    ) -> None:
        if not (src.ndim == 0 and dst.numel() == 1) and tuple(src.shape) != tuple(dst.shape):
            dst_base, _ = self._resolve_base_and_range(dst)
            dst_name = self._dst_registry.get(dst_base, "<unknown>")
            raise RuntimeError(
                f"Shape mismatch in trace copy_ for stack -> {dst_name}: "
                f"{tuple(src.shape)} vs {tuple(dst.shape)}"
            )

        dst_base, dst_range, dst_dim_map = self._resolve_base_and_meta(dst)
        if dst_base not in self._dst_registry:
            raise RuntimeError("Trace encountered copy_ to unknown dst")
        dst_name = self._dst_registry[dst_base]

        stack = self._stack_meta[src_base]
        stack_dim = stack.dim
        dst_base_dim = self._map_dim_to_base(dst_dim_map, stack_dim)

        if isinstance(src_range, MultiRange):
            raise RuntimeError("Trace encountered stack copy with multi-dim slicing")

        num_parts = len(stack.parts)
        src_stack = self._range_for_dim(src_range, stack_dim)
        src_start = 0 if src_stack is None else src_stack.start
        src_end = num_parts if src_stack is None else src_stack.end
        if src_start < 0 or src_end > num_parts:
            raise RuntimeError("Trace encountered stack range out of bounds")

        dst_stack = self._range_for_dim(dst_range, dst_base_dim)
        dst_base_start = 0 if dst_stack is None else dst_stack.start

        for idx, part in enumerate(stack.parts):
            if idx < src_start or idx >= src_end:
                continue
            dst_seg_start = dst_base_start + (idx - src_start)
            dst_seg_end = dst_seg_start + 1
            dst_seg_range = self._set_absolute_range(
                dst_range, dst_base_dim, dst_seg_start, dst_seg_end
            )

            if part.kind == "ckpt":
                if part.ckpt_name is None:
                    raise RuntimeError("Internal error: missing ckpt_name for stack part")
                self.copy_plan.append(
                    CopyPlanEntry(
                        op="copy",
                        ckpt_name=part.ckpt_name,
                        ckpt_range=part.ckpt_range,
                        ckpt_view=None,
                        dst_name=dst_name,
                        dst_range=dst_seg_range,
                    )
                )
                self.expected_src_names.add(part.ckpt_name)
                self.expected_dst_names.add(dst_name)
            elif part.kind == "const":
                if part.fill_value is None:
                    raise RuntimeError("Internal error: missing fill_value for stack const")
                self.copy_plan.append(
                    CopyPlanEntry(
                        op="fill",
                        ckpt_name=None,
                        ckpt_range=None,
                        ckpt_view=None,
                        dst_name=dst_name,
                        dst_range=dst_seg_range,
                        fill_value=part.fill_value,
                    )
                )
                self.expected_dst_names.add(dst_name)
            else:
                raise RuntimeError(
                    f"Internal error: unknown stack part kind {part.kind!r}"
                )

    def _record_fill_from_ckpt(self, dst: torch.Tensor, value: ScalarProxy) -> None:
        if dst.numel() == 0:
            return
        dst_base, dst_range = self._resolve_base_and_range(dst)
        if dst_base not in self._dst_registry:
            raise RuntimeError("Trace encountered fill_ to unknown dst")
        dst_name = self._dst_registry[dst_base]
        ckpt_name = value.ckpt_name
        ckpt_range = value.ckpt_range
        self.copy_plan.append(
            CopyPlanEntry(
                op="fill",
                ckpt_name=ckpt_name,
                ckpt_range=ckpt_range,
                ckpt_view=value.ckpt_view,
                dst_name=dst_name,
                dst_range=dst_range,
            )
        )
        self.expected_src_names.add(ckpt_name)
        self.expected_dst_names.add(dst_name)

    def _record_fill_const(self, dst: torch.Tensor, value: int | float) -> None:
        if dst.numel() == 0:
            return
        dst_base, dst_range = self._resolve_base_and_range(dst)
        if dst_base not in self._dst_registry:
            raise RuntimeError("Trace encountered fill_ to unknown dst")
        dst_name = self._dst_registry[dst_base]
        self.copy_plan.append(
            CopyPlanEntry(
                op="fill",
                ckpt_name=None,
                ckpt_range=None,
                ckpt_view=None,
                dst_name=dst_name,
                dst_range=dst_range,
                fill_value=value,
            )
        )
        self.expected_dst_names.add(dst_name)

    def _normalize_dim(self, dim: int, ndim: int) -> int:
        dim = int(dim)
        if dim < 0:
            dim += ndim
        return dim

    def _normalize_index(self, index: int, size: int) -> int:
        index = int(index)
        if index < 0:
            index += size
        return index

    def _normalize_slice(
        self,
        start: int | None,
        end: int | None,
        step: int | None,
        size: int,
    ) -> tuple[int, int, int]:
        if step not in (None, 1):
            raise RuntimeError("Trace does not support slice step != 1")
        if start is None:
            start = 0
        if end is None:
            end = size
        start = int(start)
        end = int(end)
        if start < 0:
            start += size
        if end < 0:
            end += size
        start = max(start, 0)
        end = min(end, size)
        if start > end:
            raise RuntimeError("Trace encountered invalid slice")
        return start, end, 1

    def _merge_range(self, prev: RangeSpec | None, dim: int, start: int, end: int) -> RangeSpec:
        if dim < 0:
            raise RuntimeError(f"Trace encountered slice on invalid dim {dim}")
        if prev is None:
            if start > end:
                raise RuntimeError("Trace encountered invalid slice")
            return Range(dim=dim, start=start, end=end)
        ranges = [prev] if isinstance(prev, Range) else list(prev.ranges)
        for i, old in enumerate(ranges):
            if old.dim != dim:
                continue
            start = old.start + start
            end = old.start + end
            if start > end:
                raise RuntimeError("Trace encountered invalid slice")
            ranges[i] = Range(dim=dim, start=start, end=end)
            break
        else:
            if start > end:
                raise RuntimeError("Trace encountered invalid slice")
            ranges.append(Range(dim=dim, start=start, end=end))
        ranges.sort(key=lambda r: r.dim)
        if len(ranges) == 1:
            return ranges[0]
        return MultiRange(ranges=tuple(ranges))

    def _range_for_dim(self, prev: RangeSpec | None, dim: int) -> Range | None:
        if prev is None:
            return None
        if isinstance(prev, Range):
            return prev if prev.dim == dim else None
        for r in prev.ranges:
            if r.dim == dim:
                return r
        return None

    def _set_absolute_range(
        self,
        prev: RangeSpec | None,
        dim: int,
        start: int,
        end: int,
    ) -> RangeSpec:
        if dim < 0:
            raise RuntimeError(f"Trace encountered slice on invalid dim {dim}")
        if start > end:
            raise RuntimeError("Trace encountered invalid slice")
        if prev is None:
            return Range(dim=dim, start=start, end=end)
        ranges = [prev] if isinstance(prev, Range) else list(prev.ranges)
        for i, old in enumerate(ranges):
            if old.dim != dim:
                continue
            ranges[i] = Range(dim=dim, start=start, end=end)
            break
        else:
            ranges.append(Range(dim=dim, start=start, end=end))
        ranges.sort(key=lambda r: r.dim)
        if len(ranges) == 1:
            return ranges[0]
        return MultiRange(ranges=tuple(ranges))

    @staticmethod
    def _remove_dim(dim_map: tuple[int, ...] | None, dim: int) -> tuple[int, ...] | None:
        if dim_map is None:
            return None
        if dim < 0 or dim >= len(dim_map):
            return None
        return dim_map[:dim] + dim_map[dim + 1 :]

    @staticmethod
    def _map_dim_to_base(dim_map: tuple[int, ...] | None, dim: int) -> int:
        if dim_map is None:
            return dim
        if dim < 0 or dim >= len(dim_map):
            raise RuntimeError("Trace encountered invalid dim mapping")
        return dim_map[dim]

    def _set_view_range(
        self, result: Any, tensor: torch.Tensor, dim: int, start: int, end: int, step: int
    ) -> None:
        base_id, base_range, dim_map = self._resolve_base_and_meta(tensor)
        if start == 0 and end == tensor.size(dim):
            if isinstance(result, torch.Tensor):
                self._assign_view_meta(result, base_id, base_range, dim_map)
            elif isinstance(result, (list, tuple)):
                for item in result:
                    if isinstance(item, torch.Tensor):
                        self._assign_view_meta(item, base_id, base_range, dim_map)
            return
        base_dim = self._map_dim_to_base(dim_map, dim)
        if isinstance(result, torch.Tensor):
            new_range = self._merge_range(base_range, base_dim, start, end)
            self._assign_view_meta(result, base_id, new_range, dim_map)
        elif isinstance(result, (list, tuple)):
            for item in result:
                if isinstance(item, torch.Tensor):
                    new_range = self._merge_range(base_range, base_dim, start, end)
                    self._assign_view_meta(item, base_id, new_range, dim_map)

    def _set_select_range(self, result: Any, tensor: torch.Tensor, dim: int, index: int) -> None:
        size = tensor.size(dim)
        index = self._normalize_index(index, size)
        base_id, base_range, dim_map = self._resolve_base_and_meta(tensor)
        base_dim = self._map_dim_to_base(dim_map, dim)
        new_range = self._merge_range(base_range, base_dim, index, index + 1)
        new_dim_map = self._remove_dim(dim_map, dim)
        if isinstance(result, torch.Tensor):
            self._assign_view_meta(result, base_id, new_range, new_dim_map)
        elif isinstance(result, (list, tuple)):
            for item in result:
                if isinstance(item, torch.Tensor):
                    self._assign_view_meta(item, base_id, new_range, new_dim_map)

    def _set_split_ranges(self, result: Any, tensor: torch.Tensor, dim: int, split_size: int) -> None:
        if not isinstance(result, (list, tuple)):
            return
        size = tensor.size(dim)
        start = 0
        for out in result:
            end = min(start + split_size, size)
            self._set_view_range(out, tensor, dim, start, end, 1)
            start = end

    def _set_split_with_sizes_ranges(
        self,
        result: Any,
        tensor: torch.Tensor,
        dim: int,
        split_sizes: Iterable[int],
    ) -> None:
        if not isinstance(result, (list, tuple)):
            return
        start = 0
        for out, size in zip(result, split_sizes):
            end = start + int(size)
            self._set_view_range(out, tensor, dim, start, end, 1)
            start = end

    def _set_chunk_ranges(self, result: Any, tensor: torch.Tensor, dim: int, chunks: int) -> None:
        if not isinstance(result, (list, tuple)):
            return
        size = tensor.size(dim)
        if chunks <= 0:
            raise RuntimeError("Trace encountered invalid chunk count")
        chunk_size = (size + chunks - 1) // chunks
        start = 0
        for out in result:
            end = min(start + chunk_size, size)
            self._set_view_range(out, tensor, dim, start, end, 1)
            start = end

    def _set_unbind_ranges(self, result: Any, tensor: torch.Tensor, dim: int) -> None:
        if not isinstance(result, (list, tuple)):
            return
        size = tensor.size(dim)
        base_id, base_range, dim_map = self._resolve_base_and_meta(tensor)
        base_dim = self._map_dim_to_base(dim_map, dim)
        new_dim_map = self._remove_dim(dim_map, dim)
        for i, out in enumerate(result):
            if i >= size:
                break
            if not isinstance(out, torch.Tensor):
                continue
            new_range = self._merge_range(base_range, base_dim, i, i + 1)
            self._assign_view_meta(out, base_id, new_range, new_dim_map)

    def _parse_slice_args(self, args, kwargs):
        tensor = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        start = args[2] if len(args) > 2 else kwargs.get("start")
        end = args[3] if len(args) > 3 else kwargs.get("end")
        step = args[4] if len(args) > 4 else kwargs.get("step")
        dim = self._normalize_dim(int(dim), tensor.dim())
        start, end, step = self._normalize_slice(start, end, step, tensor.size(dim))
        return tensor, dim, start, end, step

    def _parse_select_args(self, args, kwargs):
        tensor = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        index = args[2] if len(args) > 2 else kwargs.get("index")
        dim = self._normalize_dim(int(dim), tensor.dim())
        return tensor, dim, index

    def _parse_narrow_args(self, args, kwargs):
        tensor = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        start = args[2] if len(args) > 2 else kwargs.get("start")
        length = args[3] if len(args) > 3 else kwargs.get("length")
        dim = self._normalize_dim(int(dim), tensor.dim())
        return tensor, dim, int(start), int(length)

    def _parse_split_args(self, args, kwargs):
        tensor = args[0]
        split_size = args[1] if len(args) > 1 else kwargs.get("split_size")
        dim = args[2] if len(args) > 2 else kwargs.get("dim", 0)
        dim = self._normalize_dim(int(dim), tensor.dim())
        return tensor, int(split_size), dim

    def _parse_split_with_sizes_args(self, args, kwargs):
        tensor = args[0]
        split_sizes = args[1] if len(args) > 1 else kwargs.get("split_sizes")
        dim = args[2] if len(args) > 2 else kwargs.get("dim", 0)
        dim = self._normalize_dim(int(dim), tensor.dim())
        return tensor, split_sizes, dim

    def _parse_chunk_args(self, args, kwargs):
        tensor = args[0]
        chunks = args[1] if len(args) > 1 else kwargs.get("chunks")
        dim = args[2] if len(args) > 2 else kwargs.get("dim", 0)
        dim = self._normalize_dim(int(dim), tensor.dim())
        return tensor, int(chunks), dim

    def _parse_unbind_args(self, args, kwargs):
        tensor = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        dim = self._normalize_dim(int(dim), tensor.dim())
        return tensor, dim

    def _handle_slice_op(self, name, func, args, kwargs):
        if name == "aten::slice":
            tensor, dim, start, end, step = self._parse_slice_args(args, kwargs)
            result = func(*args, **kwargs)
            self._set_view_range(result, tensor, dim, start, end, step)
            return result

        if name == "aten::select":
            tensor, dim, index = self._parse_select_args(args, kwargs)
            result = func(*args, **kwargs)
            self._set_select_range(result, tensor, dim, index)
            return result

        if name == "aten::narrow":
            tensor, dim, start, length = self._parse_narrow_args(args, kwargs)
            result = func(*args, **kwargs)
            end = start + length
            self._set_view_range(result, tensor, dim, start, end, 1)
            return result

        if name == "aten::split":
            tensor, split_size, dim = self._parse_split_args(args, kwargs)
            result = func(*args, **kwargs)
            self._set_split_ranges(result, tensor, dim, split_size)
            return result

        if name == "aten::split_with_sizes":
            tensor, split_sizes, dim = self._parse_split_with_sizes_args(args, kwargs)
            result = func(*args, **kwargs)
            self._set_split_with_sizes_ranges(result, tensor, dim, split_sizes)
            return result

        if name == "aten::chunk":
            tensor, chunks, dim = self._parse_chunk_args(args, kwargs)
            result = func(*args, **kwargs)
            self._set_chunk_ranges(result, tensor, dim, chunks)
            return result

        if name == "aten::unbind":
            tensor, dim = self._parse_unbind_args(args, kwargs)
            result = func(*args, **kwargs)
            self._set_unbind_ranges(result, tensor, dim)
            return result

        raise RuntimeError(f"Unhandled slice op: {name}")


def meta_weights_iterator(
    ordered_names: list[str],
    meta_by_name: Mapping[str, Any],
    *,
    src_registry: dict[int, str],
    device: str = "meta",
    keepalive: list[torch.Tensor] | None = None,
):
    for name in ordered_names:
        meta = meta_by_name[name]
        shape = tuple(int(d) for d in meta.shape)
        tensor = torch.empty(shape, dtype=meta.dtype, device=device)
        src_registry[id(tensor)] = name
        if keepalive is not None:
            keepalive.append(tensor)
        yield name, tensor


def trace_model_load(
    model: torch.nn.Module,
    *,
    ordered_names: list[str],
    meta_by_name: Mapping[str, Any],
) -> TracePlan:
    src_registry: dict[int, str] = {}
    dst_registry: dict[int, str] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        dst_registry.setdefault(id(param), name)
    for name, buf in model.named_buffers(remove_duplicate=False):
        dst_registry.setdefault(id(buf), name)

    # NOTE: Some model load_weight implementations iterate `weights` more than
    # once (e.g. via `filter(..., weights)` for tying), so use a re-iterable
    # container rather than a single-pass generator.
    #
    # Also sort source names to avoid relying on artifact-provided ordering.
    src_keepalive: list[torch.Tensor] = []
    weights = list(
        meta_weights_iterator(
            sorted(ordered_names),
            meta_by_name,
            src_registry=src_registry,
            keepalive=src_keepalive,
        )
    )
    trace_mode = TraceMode(src_registry=src_registry, dst_registry=dst_registry)
    with trace_mode:
        model.load_weights(weights)

    if not trace_mode.copy_plan:
        raise RuntimeError("tensorcast trace produced empty copy plan")
    if not trace_mode.expected_src_names:
        raise RuntimeError("tensorcast trace produced empty source set")
    if not trace_mode.expected_dst_names:
        raise RuntimeError("tensorcast trace produced empty dst set")

    tensorcast_slices, src_hull = build_tensorcast_slices(
        trace_mode.copy_plan,
        meta_by_name,
        unsliceable_src_names=trace_mode.unsliceable_src_names,
    )
    return TracePlan(
        copy_plan=trace_mode.copy_plan,
        expected_src_names=trace_mode.expected_src_names,
        expected_dst_names=trace_mode.expected_dst_names,
        unsliceable_src_names=trace_mode.unsliceable_src_names,
        tensorcast_slices=tensorcast_slices,
        src_hull=src_hull,
    )


def build_tensorcast_slices(
    copy_plan: list[CopyPlanEntry],
    meta_by_name: Mapping[str, Any],
    *,
    unsliceable_src_names: set[str] | None = None,
) -> tuple[dict[str, Range], dict[str, Range]]:
    full_reads: set[str] = set(unsliceable_src_names or ())
    ranges_by_name: dict[str, list[Range]] = {}
    for entry in copy_plan:
        if entry.ckpt_name is None:
            continue
        if entry.ckpt_name in full_reads:
            continue
        if entry.ckpt_range is None:
            full_reads.add(entry.ckpt_name)
            continue
        if isinstance(entry.ckpt_range, MultiRange):
            full_reads.add(entry.ckpt_name)
            continue
        ranges_by_name.setdefault(entry.ckpt_name, []).append(entry.ckpt_range)

    src_hull: dict[str, Range] = {}
    for name, ranges in ranges_by_name.items():
        dim = ranges[0].dim
        for r in ranges[1:]:
            if r.dim != dim:
                raise RuntimeError(f"tensorcast trace saw multi-dim slices for {name}")
        start = min(r.start for r in ranges)
        end = max(r.end for r in ranges)
        if start >= end:
            raise RuntimeError(f"tensorcast trace saw empty slice for {name}")
        src_hull[name] = Range(dim=dim, start=start, end=end)

    tensorcast_slices: dict[str, Range] = {}
    for name, hull in src_hull.items():
        meta = meta_by_name.get(name)
        if meta is None:
            raise RuntimeError(f"Missing metadata for {name}")
        if name in full_reads:
            continue
        if len(meta.shape) == 0:
            continue
        if hull.dim >= len(meta.shape):
            raise RuntimeError(f"Slice dim {hull.dim} out of range for {name}")
        extent = int(meta.shape[hull.dim])
        if hull.start < 0 or hull.end > extent:
            raise RuntimeError(f"Slice out of range for {name}: {hull.start}:{hull.end}")
        if hull.start == 0 and hull.end == extent:
            continue
        tensorcast_slices[name] = hull

    return tensorcast_slices, src_hull


def trace_plan_to_json(trace_plan: TracePlan) -> dict[str, Any]:
    return {
        "copy_plan": [dataclasses.asdict(entry) for entry in trace_plan.copy_plan],
        "expected_src_names": sorted(trace_plan.expected_src_names),
        "expected_dst_names": sorted(trace_plan.expected_dst_names),
        "unsliceable_src_names": sorted(trace_plan.unsliceable_src_names),
        "tensorcast_slices": {name: dataclasses.asdict(rng) for name, rng in trace_plan.tensorcast_slices.items()},
        "src_hull": {name: dataclasses.asdict(rng) for name, rng in trace_plan.src_hull.items()},
    }


def trace_plan_from_json(data: Mapping[str, Any]) -> TracePlan:
    def _range_from_dict(obj: Mapping[str, Any]) -> Range:
        return Range(dim=int(obj["dim"]), start=int(obj["start"]), end=int(obj["end"]))

    def _range_spec_from_obj(obj: object) -> RangeSpec:
        if isinstance(obj, dict) and "ranges" in obj:
            ranges = tuple(_range_from_dict(r) for r in obj["ranges"])
            return MultiRange(ranges=ranges)
        if isinstance(obj, dict):
            return _range_from_dict(obj)
        raise TypeError(f"Invalid range spec: {obj!r}")

    def _view_spec_from_obj(obj: object) -> ViewSpec:
        if not isinstance(obj, dict):
            raise TypeError(f"Invalid view spec: {obj!r}")
        return ViewSpec(
            size=tuple(int(v) for v in obj["size"]),
            stride=tuple(int(v) for v in obj["stride"]),
            storage_offset=int(obj["storage_offset"]),
        )

    copy_plan: list[CopyPlanEntry] = []
    for entry in data.get("copy_plan", []):
        ckpt_range_obj = entry.get("ckpt_range")
        ckpt_view_obj = entry.get("ckpt_view")
        dst_range_obj = entry.get("dst_range")
        copy_plan.append(
            CopyPlanEntry(
                op=str(entry["op"]),
                ckpt_name=entry.get("ckpt_name"),
                ckpt_range=None if ckpt_range_obj is None else _range_spec_from_obj(ckpt_range_obj),
                ckpt_view=None if ckpt_view_obj is None else _view_spec_from_obj(ckpt_view_obj),
                dst_name=str(entry["dst_name"]),
                dst_range=None if dst_range_obj is None else _range_spec_from_obj(dst_range_obj),
                fill_value=entry.get("fill_value"),
            )
        )

    return TracePlan(
        copy_plan=copy_plan,
        expected_src_names=set(data.get("expected_src_names", [])),
        expected_dst_names=set(data.get("expected_dst_names", [])),
        unsliceable_src_names=set(data.get("unsliceable_src_names", [])),
        tensorcast_slices={name: _range_from_dict(rng) for name, rng in data.get("tensorcast_slices", {}).items()},
        src_hull={name: _range_from_dict(rng) for name, rng in data.get("src_hull", {}).items()},
    )

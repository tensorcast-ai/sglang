"""Memory pool configurators for KV cache sizing.

Each configurator computes a bytes_per_full_token cost — the amortized memory
cost of one context token across all pools. This enables lossless conversion
between available_bytes and max_total_num_tokens, so external constraints
(user cap, page align, PP sync, draft override) can be applied and then
all pool sizes recomputed via the same code path.

Passing model_runner directly is a temporary approach — future work should
introduce a lightweight dataclass (e.g. ModelMemorySpec) to decouple from
the god class.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.configs.model_config import get_nsa_index_head_dim, is_deepseek_nsa
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
from sglang.srt.utils.common import is_float4_e2m1fn_x2

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class MemoryPoolConfigurator:
    """Base class. Subclasses compute bytes_per_full_token in __init__,
    then implement calculate_pool_sizes to convert available_bytes into
    pool size fields.

    calculate_pool_sizes_from_max_tokens converts a constrained
    max_total_num_tokens back to bytes and calls calculate_pool_sizes,
    so both profiling and constraint paths use the same computation.
    """

    max_total_num_tokens: int = 0

    def calculate_pool_sizes(self, available_bytes: int, page_size: int) -> None:
        raise NotImplementedError

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Default: MHA / MLA / NSA / FP4
# ---------------------------------------------------------------------------


class DefaultPoolConfigurator(MemoryPoolConfigurator):
    """Handles MHA, MLA, NSA, and FP4 models.

    bytes_per_full_token = cell_size (bytes per token across all KV layers).
    """

    def __init__(self, model_runner: ModelRunner):
        mc = model_runner.model_config
        kv_cache_dtype = model_runner.kv_cache_dtype
        kv_elem_size = torch._utils._element_size(kv_cache_dtype)
        is_fp4 = is_float4_e2m1fn_x2(kv_cache_dtype)
        use_mla = model_runner.use_mla_backend

        num_kv_heads = mc.get_num_kv_heads(get_attention_tp_size())
        head_dim = mc.head_dim
        v_head_dim = mc.v_head_dim
        kv_lora_rank = mc.kv_lora_rank
        qk_rope_head_dim = mc.qk_rope_head_dim

        is_nsa = is_deepseek_nsa(mc.hf_config)
        index_head_dim = get_nsa_index_head_dim(mc.hf_config) if is_nsa else 0

        # num_layers — absorb draft/mamba/PP logic
        if model_runner.is_draft_worker:
            num_layers = getattr(
                mc.hf_config,
                "num_nextn_predict_layers",
                model_runner.num_effective_layers,
            )
        elif mambaish := model_runner.mambaish_config:
            effective_layer_ids = [
                i
                for i in mambaish.full_attention_layer_ids
                if model_runner.start_layer <= i < model_runner.end_layer
            ]
            num_layers = len(effective_layer_ids)
        else:
            num_layers = model_runner.num_effective_layers

        # Compute cell_size (= bytes_per_full_token)
        if use_mla:
            cell_size = (kv_lora_rank + qk_rope_head_dim) * num_layers * kv_elem_size
            if is_fp4:
                scale_block_size = 16
                cell_size = (cell_size // 2) + (
                    (kv_lora_rank + qk_rope_head_dim)
                    // scale_block_size
                    * num_layers
                    * kv_elem_size
                )
            if is_nsa:
                indexer_size_per_token = (
                    index_head_dim
                    + index_head_dim // NSATokenToKVPool.quant_block_size * 4
                )
                element_size = torch._utils._element_size(
                    NSATokenToKVPool.index_k_with_scale_buffer_dtype
                )
                cell_size += indexer_size_per_token * num_layers * element_size
        else:
            cell_size = (
                num_kv_heads * (head_dim + v_head_dim) * num_layers * kv_elem_size
            )
            if is_fp4:
                scale_block_size = 16
                cell_size = (cell_size // 2) + (
                    (num_kv_heads * head_dim * num_layers * 2 * kv_elem_size)
                    // scale_block_size
                )

        self._cell_size = cell_size

    def calculate_pool_sizes(self, available_bytes: int, page_size: int) -> None:
        self.max_total_num_tokens = available_bytes // self._cell_size

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> None:
        available_bytes = max_total_num_tokens * self._cell_size
        self.calculate_pool_sizes(available_bytes, page_size)


# ---------------------------------------------------------------------------
# Hybrid SWA (Gemma2, Command-R, MiMo)
# ---------------------------------------------------------------------------


class HybridSWAPoolConfigurator(MemoryPoolConfigurator):
    """Handles hybrid sliding-window attention models.

    bytes_per_full_token = F * n_full + r * S * n_swa, where:
      F = full layer per-token memory
      S = SWA layer per-token memory
      r = swa_full_tokens_ratio
      n_full, n_swa = number of full / SWA layers

    Both profiling and constraint paths use calculate_pool_sizes.
    """

    full_max_total_num_tokens: int = 0
    swa_max_total_num_tokens: int = 0

    def __init__(self, model_runner: ModelRunner):
        mc = model_runner.model_config
        kv_size = torch._utils._element_size(model_runner.kv_cache_dtype)

        self._full_layers_num = len(mc.full_attention_layer_ids)
        self._swa_layers_num = len(mc.swa_attention_layer_ids)
        assert (
            self._swa_layers_num > 0
        ), "Hybrid SWA model must have at least one SWA layer"

        self._swa_full_tokens_ratio = model_runner.server_args.swa_full_tokens_ratio

        self._full_per_token = (
            mc.get_num_kv_heads(get_attention_tp_size())
            * (mc.head_dim + mc.v_head_dim)
            * kv_size
        )
        self._swa_per_token = (
            mc.get_swa_num_kv_heads(get_attention_tp_size())
            * (mc.swa_head_dim + mc.swa_v_head_dim)
            * kv_size
        )

        # bytes_per_full_token: the denominator from the memory equation solve
        if self._full_layers_num == 0:
            self._bytes_per_full_token = self._swa_per_token * self._swa_layers_num
        else:
            self._bytes_per_full_token = (
                self._full_per_token * self._full_layers_num
                + self._swa_full_tokens_ratio
                * self._swa_per_token
                * self._swa_layers_num
            )

    def calculate_pool_sizes(self, available_bytes: int, page_size: int) -> None:
        def align(x: int) -> int:
            return (x // page_size) * page_size

        if self._full_layers_num == 0:
            raw_tokens = (
                available_bytes // self._bytes_per_full_token
                if self._bytes_per_full_token > 0
                else 0
            )
            self.swa_max_total_num_tokens = align(raw_tokens)
            self.full_max_total_num_tokens = 0
            self.max_total_num_tokens = self.swa_max_total_num_tokens
            logger.info(
                f"Use sliding window memory pool (all SWA). "
                f"swa_layer_tokens={self.swa_max_total_num_tokens}"
            )
            return

        self.full_max_total_num_tokens = align(
            int(available_bytes / self._bytes_per_full_token)
        )
        self.swa_max_total_num_tokens = align(
            int(self.full_max_total_num_tokens * self._swa_full_tokens_ratio)
        )
        self.max_total_num_tokens = self.full_max_total_num_tokens

        logger.info(
            f"Use sliding window memory pool. "
            f"full_layer_tokens={self.full_max_total_num_tokens}, "
            f"swa_layer_tokens={self.swa_max_total_num_tokens}"
        )

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> None:
        available_bytes = max_total_num_tokens * self._bytes_per_full_token
        self.calculate_pool_sizes(available_bytes, page_size)


def create_memory_pool_configurator(
    model_runner: ModelRunner,
) -> MemoryPoolConfigurator:
    """Create the appropriate configurator for this model_runner."""
    if model_runner.model_config.is_hybrid_swa:
        return HybridSWAPoolConfigurator(model_runner)

    return DefaultPoolConfigurator(model_runner)


def profile_available_bytes(
    device: str,
    gpu_id: int,
    total_gpu_memory: float,
    mem_fraction_static: float,
    distributed: bool = False,
    cpu_group=None,
) -> int:
    """Profile available memory bytes for KV cache after model loading.

    available = (current_free - total_gpu_memory * (1 - mem_fraction_static))
    converted to bytes.
    """
    from sglang.srt.utils.common import get_available_gpu_memory

    available_gpu_memory = get_available_gpu_memory(
        device, gpu_id, distributed=distributed, cpu_group=cpu_group
    )
    rest_memory = available_gpu_memory - total_gpu_memory * (1 - mem_fraction_static)

    available_bytes = int(rest_memory * (1 << 30))

    logger.info(
        f"Memory profiling: available_gpu_memory={available_gpu_memory:.2f} GB, "
        f"total_gpu_memory={total_gpu_memory:.2f} GB, "
        f"mem_fraction_static={mem_fraction_static:.2f}, "
        f"rest_memory={rest_memory:.2f} GB"
    )

    return available_bytes

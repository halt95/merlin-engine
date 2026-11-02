# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""NVIDIA QSA owner with Triton kernels."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections.abc import Callable, Collection, Iterable, Mapping
from pathlib import Path
from typing import ClassVar, cast

import torch
from torch import nn
from transformers import Qwen4ExpTextConfig

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.config.cache import CacheConfig, CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding, get_rope
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.model_executor.models.qwen3_vl import mrope_positions_factor
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

from ..common.qsa_cache import QSAForwardMetadata
from .indexer_qsa import QSAIndexer
# Separate calibration namespaces.
from .mtp_kv_scales import qsa_kv_scale_entries
from .ops.qsa import qsa_e4m3_uses_native_cast

logger = init_logger(__name__)

_QSA_CLIP_ENV = "VLLM_QSA_KV_CLIP_COUNT"
_qsa_clip_raw = os.environ.get(_QSA_CLIP_ENV, "").strip()
try:
    _qsa_clip_interval = float(_qsa_clip_raw) if _qsa_clip_raw else 0.0
except ValueError as exc:  # fail closed: a set-but-broken gate must not be silent
    raise ValueError(
        f"{_QSA_CLIP_ENV} must be a read interval in seconds, got {_qsa_clip_raw!r}"
    ) from exc
if _qsa_clip_raw and not (math.isfinite(_qsa_clip_interval) and _qsa_clip_interval > 0.0):
    raise ValueError(
        f"{_QSA_CLIP_ENV} must be a finite positive number of seconds, got {_qsa_clip_raw!r}"
    )
_qsa_clip_counters: dict[str, torch.Tensor] = {}
# The reader must never block on the compute stream; see
# guard_telemetry.NonBlockingReader.
from vllm.v1.worker.guard_telemetry import (  # noqa: E402
    NonBlockingReader as _NonBlockingReader,
)

_qsa_clip_nb = _NonBlockingReader()
_qsa_clip_reader_started = False
_qsa_clip_lock = threading.Lock()


def _qsa_clip_reader() -> None:
    last: dict[str, tuple[int, int]] = {}
    passes = 0
    while True:
        time.sleep(_qsa_clip_interval)
        passes += 1
        try:
            totals_k = totals_v = 0
            for name, ctr in list(_qsa_clip_counters.items()):
                if not _qsa_clip_nb.registered(name):
                    _qsa_clip_nb.register(name, ctr)  # reader thread: never capturing
            ready = _qsa_clip_nb.poll()
            for name, ctr in list(_qsa_clip_counters.items()):
                row = ready.get(name)
                if row is None:
                    # copy still in flight: report on the next pass
                    prev_k, prev_v = last.get(name, (0, 0))
                    totals_k += prev_k
                    totals_v += prev_v
                    continue
                k_total = int(row[0])
                v_total = int(row[1])
                totals_k += k_total
                totals_v += v_total
                prev_k, prev_v = last.get(name, (0, 0))
                if k_total > prev_k or v_total > prev_v:
                    logger.warning(
                        "QSA KV CLIPPING on %s: k +%d (total %d), v +%d (total %d)"
                        " -- inputs exceeded the calibration absmax; the scales"
                        " sidecar may need re-calibration",
                        name,
                        k_total - prev_k,
                        k_total,
                        v_total - prev_v,
                        v_total,
                    )
                last[name] = (k_total, v_total)
            if passes % 10 == 1:
                logger.info(
                    "QSA clip counters alive: %d layer(s), totals k=%d v=%d",
                    len(_qsa_clip_counters),
                    totals_k,
                    totals_v,
                )
        except Exception:
            logger.exception("QSA clip reader pass failed; thread continues")


def _qsa_clip_start_reader() -> None:
    """Start the reader thread. Runs after warmup + capture (post-capture hook); idempotent."""
    global _qsa_clip_reader_started
    with _qsa_clip_lock:
        if _qsa_clip_reader_started:
            return
        _qsa_clip_reader_started = True
        threading.Thread(
            target=_qsa_clip_reader,
            name="qsa-clip-reader",
            daemon=True,
        ).start()
    # Log readers match the leading "QSA clip counter armed (read interval Ns)";
    # keep it verbatim when editing this line.
    logger.info(
        "QSA clip counter armed (read interval %.0fs); reader started after capture "
        "(%d layer(s) registered)",
        _qsa_clip_interval,
        len(_qsa_clip_counters),
    )


if _qsa_clip_interval > 0.0:
    from vllm.v1.worker.guard_telemetry import register_post_capture_hook  # noqa: E402

    register_post_capture_hook(_qsa_clip_start_reader)


def _qsa_clip_count(
    layer: Qwen4ExpQSAAttention,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor | None = None,
) -> None:
    ctr = _qsa_clip_counters.get(layer.layer_name)
    if ctr is None:
        ctr = torch.zeros(2, dtype=torch.int64, device=key.device)
        _qsa_clip_counters[layer.layer_name] = ctr  # registered with the reader by the reader thread (never during capture)
    k_ceiling = (448.0 * layer._k_scale).to(torch.float32).view(1)
    v_ceiling = (448.0 * layer._v_scale).to(torch.float32).view(1)
    # Non-finite K/V count too: abs(NaN) > ceiling is False, and a NaN code in
    # the cache is a fault that must stay visible.
    k_over = (key.detach().abs() > k_ceiling) | ~torch.isfinite(key.detach())
    v_over = (value.detach().abs() > v_ceiling) | ~torch.isfinite(value.detach())
    if slot_mapping is not None:
        n = min(slot_mapping.shape[0], key.shape[0])
        written = (slot_mapping[:n] >= 0).view(-1, 1, 1)
        k_over = k_over[:n] & written
        v_over = v_over[:n] & written
    ctr[0] += k_over.sum()
    ctr[1] += v_over.sum()
# ------------------------------------------------------------------------------

_QSA_FP8_CACHE_DTYPES = ("fp8", "fp8_e4m3")
_QSA_SUPPORTED_CACHE_DTYPES = ("auto", "bfloat16", *_QSA_FP8_CACHE_DTYPES)


def _is_qsa_fp8_cache_dtype(cache_dtype: str) -> bool:
    return cache_dtype in _QSA_FP8_CACHE_DTYPES


def _validated_qsa_fp8_dtype(cache_dtype: str) -> torch.dtype | None:
    """Return the platform E4M3 dtype after enforcing the supported HW gate.

    SM80/SM86 use the byte decoder; SM89+ uses FP8
    (vllm#55557). On SM80 the FP8 cache is used only when an FP8 KV cache
    dtype is requested; it has not been tested on SM80 hardware.
    SM87, ROCm and other unsupported capabilities remain refused.
    """
    if not _is_qsa_fp8_cache_dtype(cache_dtype):
        return None
    if not current_platform.is_cuda():
        raise ValueError(
            "Qwen4Exp QSA E4M3 KV cache is supported only on CUDA; "
            "ROCm and other platforms have not been validated"
        )

    capability = current_platform.get_device_capability()
    if capability is None or (
        capability.to_int() not in (80, 86) and not qsa_e4m3_uses_native_cast()
    ):
        cap_str = (
            capability.as_version_str() if capability is not None else "unknown"
        )
        raise ValueError(
            "Qwen4Exp QSA E4M3 KV cache needs SM80/SM86 (byte decoder) or SM89+ "
            f"(native FP8), but {current_platform.get_device_name()} has "
            f"compute capability {cap_str}."
        )
    return current_platform.fp8_dtype()


class Qwen4ExpQSAQKVIndexerLinear(MergedColumnParallelLinear):
    """Pack sharded Q/gate, K, V and replicated indexer Q/K in one GEMM."""

    def __init__(
        self,
        qkv_proj: QKVParallelLinear,
        index_size: int,
        quant_config: QuantizationConfig | None,
    ) -> None:
        self.num_kv_head_replicas = qkv_proj.num_kv_head_replicas
        super().__init__(
            input_size=qkv_proj.input_size,
            output_sizes=[*qkv_proj.output_sizes, index_size * qkv_proj.tp_size],
            bias=False,
            params_dtype=qkv_proj.params_dtype,
            quant_config=quant_config,
            prefix=qkv_proj.prefix,
        )

    def _load_shard(
        self,
        loader: Callable[..., None],
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | tuple[int, ...] | None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id == 3:
            shard_rank = 0
        elif loaded_shard_id in (1, 2):
            shard_rank = tp_rank // self.num_kv_head_replicas
        else:
            shard_rank = tp_rank
        self.tp_rank = shard_rank
        if param_tp_rank is not None:
            param.tp_rank = shard_rank
        try:
            loader(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank

    def weight_loader(self, param, loaded_weight, loaded_shard_id=None) -> None:
        self._load_shard(super().weight_loader, param, loaded_weight, loaded_shard_id)

    def weight_loader_v2(self, param, loaded_weight, loaded_shard_id=None) -> None:
        self._load_shard(
            super().weight_loader_v2, param, loaded_weight, loaded_shard_id
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        def remap_shards():
            for name, weight in weights:
                shard_id = getattr(weight, "shard_id", None)
                if isinstance(shard_id, str):
                    weight = weight.detach()
                    weight.shard_id = {"q": 0, "k": 1, "v": 2}[shard_id]
                yield name, weight

        return super().load_weights(remap_shards())


def qsa_kv_cache_dtype(cache_config: CacheConfig, prefix: str) -> CacheDType:
    """The layer's KV cache dtype, honoring ``--kv-cache-dtype-skip-layers``."""
    if str(extract_layer_index(prefix)) in cache_config.kv_cache_dtype_skip_layers:
        return "auto"
    return cache_config.cache_dtype


class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Flash metadata supporting uniform decode and target-verify graphs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class Qwen4ExpQSAFlashAttentionBackend(FlashAttentionBackend):
    """FullAttentionSpec backend used by the merged QSA owner."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    # fp8/fp8_e4m3: e4m3 bytes in a uint8 cache, written by reshape_and_cache
    # with the layer's per-tensor scales and dequantized on load inside the QSA
    # Triton kernel. flash-attn never runs over this cache, so its fp8 probe
    # does not apply (see supports_kv_cache_dtype and the impl constructor).
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype is None or kv_cache_dtype in cls.supported_kv_cache_dtypes

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        probe_kv_cache_dtype = kv_cache_dtype
        if probe_kv_cache_dtype is not None and _is_qsa_fp8_cache_dtype(
            probe_kv_cache_dtype
        ):
            probe_kv_cache_dtype = None
        return super().supports_combination(
            head_size,
            dtype,
            probe_kv_cache_dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            use_mm_prefix,
            device_capability,
        )

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA_TRITON"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        # QSA consumes manager pages directly and does not use FA4 paged attention.
        return [MultipleOf(16)]

    @staticmethod
    def get_impl_cls() -> type[Qwen4ExpQSAFlashAttentionImpl]:
        return Qwen4ExpQSAFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Qwen4ExpQSAMetadataBuilder]:
        return Qwen4ExpQSAMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
    """Run paged sparse GQA with the QSA Triton kernel."""

    supports_dcp: bool = False
    supports_pcp: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        real_kv_cache_dtype = kv_cache_dtype
        if _is_qsa_fp8_cache_dtype(kv_cache_dtype):
            kv_cache_dtype = "auto"
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            sinks,
        )
        self.kv_cache_dtype = real_kv_cache_dtype
        if not is_flash_attn_varlen_func_available():
            raise NotImplementedError("Qwen4Exp QSA requires FlashAttention")
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                "Qwen4Exp QSA does not support decode context parallelism"
            )
        if self.kv_cache_dtype not in _QSA_SUPPORTED_CACHE_DTYPES:
            raise NotImplementedError(
                "Qwen4Exp QSA supports BF16 or static per-tensor E4M3 main "
                f"KV caches, not {self.kv_cache_dtype!r}"
            )
        self.qsa_fp8_dtype = _validated_qsa_fp8_dtype(self.kv_cache_dtype)
        self.supports_quant_query_input = False

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        token_to_req: torch.Tensor,
        use_prefill_config: bool,
        output_gate: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA does not support fused output quantization")
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("QSA does not support ALiBi or attention sinks")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError("QSA does not support sliding-window attention")

        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output

        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        logical_indices = topk_buffer[:num_tokens]
        token_to_req = token_to_req[:num_tokens]
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        if query.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA requires a BF16 query")
        if key_cache.dtype != value_cache.dtype:
            raise TypeError("Qwen4Exp QSA K and V cache dtypes must match")
        if _is_qsa_fp8_cache_dtype(self.kv_cache_dtype):
            if self.qsa_fp8_dtype is None:
                raise RuntimeError("QSA FP8 dtype was not initialized")
            if key_cache.dtype not in (torch.uint8, self.qsa_fp8_dtype):
                raise NotImplementedError(
                    "Qwen4Exp QSA E4M3 cache storage must use raw uint8 or "
                    f"the platform FP8 dtype, not {key_cache.dtype}"
                )
        elif key_cache.dtype != torch.bfloat16:
            raise NotImplementedError(
                "Qwen4Exp QSA BF16 mode requires a BF16 main KV cache"
            )

        from .ops.qsa import qsa_sparse_paged_attention

        if _is_qsa_fp8_cache_dtype(self.kv_cache_dtype) and not getattr(
            self, "_qsa_scales_logged", False
        ):
            self._qsa_scales_logged = True
            logger.info(
                "Qwen4Exp QSA E4M3 KV active on %s: k_scale=%.6g v_scale=%.6g%s",
                getattr(layer, "layer_name", "<unnamed>"),
                float(layer._k_scale),
                float(layer._v_scale),
                "" if float(layer._k_scale) != 1.0 or float(layer._v_scale) != 1.0
                else "  (DEFAULT 1.0 -- no static calibration loaded)",
            )

        qsa_sparse_paged_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            use_prefill_config,
            out=output[:num_tokens],
            k_scale=layer._k_scale,
            v_scale=layer._v_scale,
            output_gate=output_gate[:num_tokens],
        )
        return output


class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch."""

    supports_dcp = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen4Exp QSA requires a paged KV cache")
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA currently requires BF16")
        if cache_config.cache_dtype not in _QSA_SUPPORTED_CACHE_DTYPES:
            raise NotImplementedError(
                "Qwen4Exp QSA supports BF16 or static per-tensor E4M3 main "
                f"KV caches, not {cache_config.cache_dtype!r}"
            )
        # The SM86 E4M3 gate applies to this layer's effective dtype: a layer
        # named in --kv-cache-dtype-skip-layers runs a BF16 cache (vllm#60023).
        qsa_fp8_dtype = _validated_qsa_fp8_dtype(
            qsa_kv_cache_dtype(cache_config, prefix)
        )
        kv_cache_scheme = getattr(quant_config, "kv_cache_scheme", None)
        if kv_cache_scheme is not None:
            raise NotImplementedError(
                "Qwen4Exp QSA does not implement kv_cache_scheme variants. "
                "Dynamic, per-head, per-token, and other non-scalar KV scales "
                "are unsupported; use static scalar per-layer K/V scales from "
                "load_qsa_static_kv_scales()."
            )
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA does not support context parallelism"
            )
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("Qwen4Exp QSA requires causal decoder attention")

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        # Decode/verify batches have at most 1 + num_spec query tokens per
        # request; use_prefill_config (max_query_len > this) steers the
        # config table. Shorter batches take the decode profile — harmless,
        # the difference is tile-shape tuning, not correctness.
        self._max_decode_query_len = 1 + vllm_config.num_speculative_tokens
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support dual-chunk RoPE")
        # Qwen4Exp full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True

        # deferred: model.py imports this module at top level.
        from . import model

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=model.without_modelopt_fp4(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            mrope_positions_factor=mrope_positions_factor(vllm_config),
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        mm_config = model_config.multimodal_config
        text_only = mm_config is None or mm_config.language_model_only
        mrope_section = getattr(self.rotary_emb, "mrope_section", None)
        supports_mrope = bool(
            type(self.rotary_emb) is MRotaryEmbedding
            and mrope_section
            and len(mrope_section) == 3
            and sum(mrope_section) == self.rotary_emb.rotary_dim // 2
            and getattr(self.rotary_emb, "mrope_interleaved", False)
        )
        supports_dtype = getattr(self.rotary_emb, "dtype", None) in (
            torch.float16,
            torch.bfloat16,
        )
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and supports_dtype
            and (text_only or supports_mrope)
        )

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = qsa_kv_cache_dtype(cache_config, prefix)
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        if _is_qsa_fp8_cache_dtype(self.kv_cache_dtype):
            if qsa_fp8_dtype is None:
                raise RuntimeError("QSA FP8 dtype was not initialized")
            if self.kv_cache_torch_dtype not in (torch.uint8, qsa_fp8_dtype):
                raise NotImplementedError(
                    "Qwen4Exp QSA E4M3 cache storage resolved to unsupported "
                    f"dtype {self.kv_cache_torch_dtype}"
                )
        elif self.kv_cache_torch_dtype != torch.bfloat16:
            raise NotImplementedError(
                "Qwen4Exp QSA BF16 mode requires BF16 cache storage"
            )
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)

        self.attn_backend = Qwen4ExpQSAFlashAttentionBackend
        self.impl = Qwen4ExpQSAFlashAttentionImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            None,
            None,
            self.kv_cache_dtype,
            None,
            AttentionType.DECODER,
            None,
        )
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        # One launch does the indexer prepare, the main QK-norm/RoPE/gate and
        fp8_layer = _is_qsa_fp8_cache_dtype(self.kv_cache_dtype)
        self.use_fused_qsa_prepare = (
            self.use_fused_qk_norm_rope_gate
            and self.indexer.use_fused_pre_indexer
            and (not fp8_layer or qsa_e4m3_uses_native_cast())
            and not (fp8_layer and _qsa_clip_interval > 0.0)
        )
        self.fuse_indexer_projection = vllm_config.lora_config is None
        # Port of vllm#62318: without a
        # quant config here (BF16, or ModelOpt FP4, which keeps attention in
        # BF16) the fused layer is unquantized. Otherwise the indexer shares
        # the q/k/v GEMM only if it uses the same quant method.
        fused_quant_config = model.without_modelopt_fp4(quant_config)
        projection_method = type(self.qkv_proj.quant_method)
        if fused_quant_config is not None and projection_method is not type(
            self.indexer.index_qk_proj.quant_method
        ):
            self.fuse_indexer_projection = False
        if self.fuse_indexer_projection:
            self.index_qk_size = self.indexer.index_qk_proj.output_size
            self.qkv_proj = Qwen4ExpQSAQKVIndexerLinear(
                self.qkv_proj,
                self.index_qk_size,
                fused_quant_config,
            )
            del self.indexer.index_qk_proj

        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        # PACKED selection buffer: the trailing column holds each row's
        # valid-entry count (written by the expand kernel) — never a token
        # index; the sparse attention kernel reads it as its loop bound.
        # MTP skip_topk steps reuse rows frozen from step 0; the count is
        # a row column, so compaction/reuse keep it paired with the content.
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(
                max_tokens,
                self.indexer.packed_output_width,
                dtype=torch.int32,
            ),
            persistent=False,
        )

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = None
        if self.fuse_indexer_projection:
            mapper = WeightsMapper(
                orig_to_new_stacked={
                    "indexer.index_qk_proj.": ("qkv_proj.", 3),
                }
            )
        return AutoWeightsLoader(self).load_weights(weights, mapper=mapper)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    @eager_break_during_capture
    def _run_qsa(
        self,
        projected_qk: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor | None,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        output: torch.Tensor,
        output_gate: torch.Tensor | None,
        qkv: torch.Tensor,
    ) -> None:
        # query/key/value/output_gate are None when the fused prepare runs
        # inside the indexer launch.
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            return
        main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(
            QSAForwardMetadata,
            metadata[self.indexer.raw_key_cache.prefix],
        )
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        selected, main_outputs = self.indexer(
            projected_qk,
            positions,
            self.topk_indices_buffer[:num_tokens],
            attn=self,
            qkv=qkv,
            slot_mapping=main_metadata.slot_mapping,
        )
        if selected.shape != (num_tokens, self.indexer.packed_output_width):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        if _qsa_clip_interval > 0.0 and _is_qsa_fp8_cache_dtype(
            self.kv_cache_dtype
        ):
            # The clip counter keeps E4M3 layers off the fused prepare.
            assert key is not None and value is not None
            _qsa_clip_count(self, key, value, main_metadata.slot_mapping)
        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        if main_outputs is None:
            assert key is not None and value is not None
            impl.do_kv_cache_update(
                self,
                key,
                value,
                self.kv_cache,
                main_metadata.slot_mapping,
            )
        else:
            query, output_gate = main_outputs
        assert query is not None and output_gate is not None
        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
            use_prefill_config=main_metadata.max_query_len > self._max_decode_query_len,
            output_gate=output_gate,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if self.fuse_indexer_projection:
            qkv, projected_qk = qkv.split(
                [2 * self.q_size + 2 * self.kv_size, self.index_qk_size], dim=-1
            )
        else:
            projected_qk, _ = self.indexer.index_qk_proj(hidden_states)
        num_tokens = hidden_states.shape[0]
        if not self.use_fused_qsa_prepare:
            q, k, v, gate = self._project_qkv_gate(qkv, positions)
            assert gate is not None
            query = q.view(num_tokens, self.num_heads, self.head_dim)
            key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
            value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        else:
            # Norm/RoPE/gate and the K/V cache write happen inside _run_qsa.
            query = key = value = gate = None
        attn_output = qkv.new_empty(num_tokens, self.num_heads, self.head_dim)
        encoded_layer_name = _encode_layer_name(self.layer_name)
        if current_platform.opaque_attention_op():
            torch.ops.vllm.qwen4_exp_qsa_with_output(
                projected_qk,
                positions,
                query,
                key,
                value,
                attn_output,
                gate,
                qkv,
                encoded_layer_name,
            )
        else:
            qwen4_exp_qsa_with_output(
                projected_qk,
                positions,
                query,
                key,
                value,
                attn_output,
                gate,
                qkv,
                encoded_layer_name,
            )
        flat_output = attn_output.view(num_tokens, -1)
        output, _ = self.o_proj(flat_output)
        return output


def load_qsa_static_kv_scales(
    layers: nn.Module | Mapping[str, nn.Module],
    sidecar_path: str | Path,
    *,
    strict: bool = True,
    expected_layer_names: Collection[str] | None = None,
    mtp: bool = False,
) -> list[str]:
    """Load static scalar QSA K/V scales into already-loaded model layers.

    Call this on every model worker after checkpoint weight loading finishes
    and before any profiling, warmup, or cudagraph capture. ``layers`` may be
    the loaded root module or ``compilation_config.static_forward_context``.

    The JSON schema is::

        {
          "model.layers.3.self_attn.attn": {
            "k_scale": 0.25,
            "v_scale": 2.0
          }
        }

    ``strict=False`` permits a deliberate subset; omitted layers keep the
    vLLM default scale of 1.0. Unknown names and non-scalar values are always
    errors.

    Pipeline parallelism: a rank holds only its stage's layers, so the sidecar
    legitimately names layers this rank does not own. Pass
    ``expected_layer_names`` = the QSA layer names of the WHOLE model; names are
    validated against that set, ``strict`` requires the sidecar to cover all
    of it (checkable on every rank), and the scales are applied to the local
    intersection only. Without it (TP-only callers) the local layers are the
    universe, as before.
    """

    path = Path(sidecar_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object keyed by layer name")
    # Targets ignore the optional mtp object.
    raw = qsa_kv_scale_entries(raw, path, mtp=mtp)

    if isinstance(layers, nn.Module):
        qsa_layers = {
            layer.layer_name: layer
            for layer in layers.modules()
            if isinstance(layer, Qwen4ExpQSAAttention)
        }
    else:
        qsa_layers = {
            name: layer
            for name, layer in layers.items()
            if isinstance(name, str)
            and isinstance(layer, Qwen4ExpQSAAttention)
        }

    if not qsa_layers and expected_layer_names is None:
        raise ValueError("No Qwen4Exp QSA attention layers were supplied")
    non_fp8_layers = sorted(
        name
        for name, layer in qsa_layers.items()
        if not _is_qsa_fp8_cache_dtype(layer.kv_cache_dtype)
    )
    if non_fp8_layers:
        raise ValueError(
            "Static QSA KV scales may only be injected into E4M3 layers; "
            f"non-FP8 layers: {non_fp8_layers}"
        )

    if expected_layer_names is None:
        universe = set(qsa_layers)
    else:
        universe = set(expected_layer_names)
        not_expected = sorted(name for name in qsa_layers if name not in universe)
        if not_expected:
            raise ValueError(
                "Local QSA layers outside expected_layer_names: "
                f"{not_expected}"
            )
    unknown = sorted(name for name in raw if name not in universe)
    if unknown:
        raise ValueError(f"Unknown QSA layer names in {path}: {unknown}")
    missing = sorted(name for name in universe if name not in raw)
    if strict and missing:
        raise ValueError(f"Missing QSA layer scales in {path}: {missing}")
    remote = sorted(name for name in raw if name not in qsa_layers)
    if remote:
        logger.info(
            "Qwen4Exp QSA: %d sidecar layer(s) belong to other pipeline stages "
            "and are skipped on this rank: %s%s",
            len(remote),
            ", ".join(remote[:4]),
            " ..." if len(remote) > 4 else "",
        )

    validated: dict[str, tuple[float, float]] = {}
    for layer_name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{layer_name} scales must be a JSON object")
        if set(entry) != {"k_scale", "v_scale"}:
            raise ValueError(
                f"{layer_name} must contain exactly k_scale and v_scale"
            )

        values: list[float] = []
        for field in ("k_scale", "v_scale"):
            value = entry[field]
            if type(value) not in (int, float):
                raise ValueError(
                    f"{layer_name}.{field} must be a scalar JSON number"
                )
            scale = float(value)
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    f"{layer_name}.{field} must be finite and positive"
                )
            values.append(scale)

        layer = qsa_layers.get(layer_name)
        if layer is None:
            continue  # another pipeline stage's layer: schema-checked only
        for field in ("_k_scale", "_v_scale"):
            tensor = getattr(layer, field, None)
            if not isinstance(tensor, torch.Tensor) or tensor.numel() != 1:
                raise ValueError(
                    f"{layer_name}.{field} must be an existing scalar tensor"
                )
        validated[layer_name] = (values[0], values[1])

    with torch.no_grad():
        for layer_name, (k_scale, v_scale) in validated.items():
            layer = qsa_layers[layer_name]
            layer._k_scale.fill_(k_scale)
            layer._v_scale.fill_(v_scale)
            # The SM89+ fused prepare (upstream's path) quantizes with the
            # host-float copies; keep them equal to the tensors.
            layer._k_scale_float = k_scale
            layer._v_scale_float = v_scale

    return sorted(validated)


def qwen4_exp_qsa_with_output(
    projected_qk: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor | None,
    key: torch.Tensor | None,
    value: torch.Tensor | None,
    output: torch.Tensor,
    output_gate: torch.Tensor | None,
    qkv: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Run the complete QSA state/update/attend transaction.

    query/key/value/output_gate are None when the layer uses the fused QSA
    prepare (main QK-norm/RoPE/gate and K/V cache write inside the indexer
    launch, which reads ``qkv``).
    """
    layer_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen4ExpQSAAttention):
        raise TypeError(f"{layer_name} is not a Qwen4Exp QSA owner")
    layer._run_qsa(
        projected_qk,
        positions,
        query,
        key,
        value,
        output,
        output_gate,
        qkv,
    )


def qwen4_exp_qsa_with_output_fake(
    projected_qk: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor | None,
    key: torch.Tensor | None,
    value: torch.Tensor | None,
    output: torch.Tensor,
    output_gate: torch.Tensor | None,
    qkv: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    del projected_qk, positions, query, key, value, output, output_gate, qkv
    del layer_name


direct_register_custom_op(
    op_name="qwen4_exp_qsa_with_output",
    op_func=qwen4_exp_qsa_with_output,
    mutates_args=["output"],
    fake_impl=qwen4_exp_qsa_with_output_fake,
)


__all__ = [
    "QSAIndexer",
    "Qwen4ExpQSAAttention",
    "Qwen4ExpQSAFlashAttentionBackend",
    "Qwen4ExpQSAFlashAttentionImpl",
    "load_qsa_static_kv_scales",
    "qwen4_exp_qsa_with_output",
]

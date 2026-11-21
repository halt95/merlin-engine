# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Model-private Hy4 sparse MLA backend.

Pass the backend class explicitly to MLAAttention. Distributed ownership is
provided by the DCP policy in ``vllm.merlin.hy4_kvshard``; PCP remains unsupported.
"""

import torch

from vllm.config import get_current_vllm_config
from vllm.merlin.hy4_kernels import (
    DECODE_MAX_ROWS,
    reserve_sparse_workspace,
    sparse_mla,
    sparse_mla_decode,
)
from vllm.merlin.hy4_kvshard import Hy4ReplicatedKVImplMixin
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
    SparseMLACommonMetadata,
    SparseMLACommonMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    align_blocks_to_rows,
    flat_kv_row_view,
)
from vllm.v1.attention.ops.metadata import compute_token_to_req_indices


def _require_sparse_mqa(config):
    if not config.attention_config.sparse_mla_force_mqa:
        raise ValueError(
            "Hy4 sinks require sparse_mla_force_mqa (dense prefill drops the sink)"
        )


class Hy4TritonMetadataBuilder(SparseMLACommonMetadataBuilder):
    metadata_cls = SparseMLACommonMetadata
    # FULL graphs for uniform decode / spec-verify batches, also under DCP: the
    # query gather, DCP index filter, sparse_mla with the owner sink and the LSE
    # combine take shapes from the padded batch and read only device tensors.
    # Mixed batches run as breakable PIECEWISE graphs with this layer eager.
    # tests (development tree): merlin_cpu/test_hy4_cudagraph.py,
    # merlin/test_hy4_dcp_cudagraph_gpu.py
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    # This backend never gathers a dense prefill context: MLAAttention's
    # profile-run dummy reads the size below instead of the 64K-token default.
    never_gathers_prefill_context = True

    @staticmethod
    def determine_chunked_prefill_workspace_size(vllm_config):
        # This backend never gathers a dense prefill context.
        return vllm_config.cache_config.block_size or 64

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        _require_sparse_mqa(vllm_config)
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        layer = vllm_config.compilation_config.static_forward_context[layer_names[0]]
        if layer.topk_buffer_width is None:
            raise ValueError("Hy4 requires an explicit logical top-k buffer")
        self.topk_tokens = layer.topk_buffer_width
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        self.supports_draft_decode_metadata_update = self.dcp_world_size == 1

    def update_draft_decode_metadata(self, metadata):
        count = metadata.num_decode_tokens
        if count:
            compute_token_to_req_indices(
                metadata.query_start_loc, metadata.req_id_per_token, count, count
            )


class Hy4TritonSparseBackend(AttentionBackend):
    supported_dtypes = [torch.bfloat16, torch.float16]
    supported_kv_cache_dtypes = ["auto", "bfloat16", "float16", "fp8_ds_mla"]

    @staticmethod
    def get_name():
        return "HY4_TRITON_SPARSE"

    @staticmethod
    def get_impl_cls():
        return Hy4TritonSparseImpl

    @staticmethod
    def get_builder_cls():
        return Hy4TritonMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        return [MultipleOf(16)]

    @classmethod
    def customize_spec(cls, spec):
        return align_blocks_to_rows(spec)

    @classmethod
    def get_supported_head_sizes(cls):
        return [576]

    @classmethod
    def is_mla(cls):
        return True

    @classmethod
    def is_sparse(cls):
        return True

    @classmethod
    def supports_sink(cls):
        return True

    @classmethod
    def supports_compute_capability(cls, capability):
        return capability >= DeviceCapability(8, 0)

    @classmethod
    def supports_pcp(cls):
        return False

    @classmethod
    def supports_dcp(cls):
        return True


class Hy4TritonSWABackend(Hy4TritonSparseBackend):
    @staticmethod
    def get_impl_cls():
        return Hy4TritonSWAImpl

    @classmethod
    def supports_sliding_window(cls):
        return True


def sparse_decode_settings(environ=None) -> tuple[bool, int | None]:
    """``MERLIN_HY4_SPARSE_DECODE`` (1 = decode path, unset/blank/0 = stock) and
    ``MERLIN_HY4_SPARSE_DECODE_HEAD_BLOCK`` (16 or 32; unset = 32 when the layer
    has at least 32 heads)."""
    import os

    env = os.environ if environ is None else environ
    raw = (env.get("MERLIN_HY4_SPARSE_DECODE", "") or "").strip()
    if raw not in ("", "0", "1"):
        raise ValueError(f"MERLIN_HY4_SPARSE_DECODE must be 0 or 1 (got {raw!r})")
    block = (env.get("MERLIN_HY4_SPARSE_DECODE_HEAD_BLOCK", "") or "").strip()
    if block not in ("", "16", "32"):
        raise ValueError(
            f"MERLIN_HY4_SPARSE_DECODE_HEAD_BLOCK must be 16 or 32 (got {block!r})"
        )
    return raw == "1", int(block) if block else None


_SPARSE_DECODE, _SPARSE_DECODE_HEAD_BLOCK = sparse_decode_settings()


class Hy4TritonSparseImpl(SparseMLACommonImpl):
    supports_dcp = True
    group_conversion_decode_only = True
    supports_dense_mha_prefill = False
    supports_quant_query_input = False
    can_return_lse_for_decode = True
    lse_base_on_e = True
    # Spec-verify (DFly2 K+1 query tokens per
    # request) under DCP at any cp_kv_cache_interleave_size: causality comes
    # only from each token's own top-k positions, never a per-request seq len.
    # Under DCP SparseMLACommonImpl._forward_mqa (sparse_mla_attention.py)
    # maps every index of every token to its owner rank and local row with
    # triton_filter_and_convert_dcp_index(..., cp_kv_cache_interleave_size)
    # (v1/attention/backends/mla/sparse_utils.py), the same path
    # FlashMLASparseImpl declares this for; the LSE combine is per token. The
    # SWA impl is replicated (dcp_world_size 1) and needs no interleave at all.
    supports_mtp_with_cp_non_trivial_interleave_size = True

    def __init__(self, *, sinks=None, **kwargs):
        _require_sparse_mqa(get_current_vllm_config())
        if self.pcp_world_size != 1:
            raise ValueError("Hy4 does not support PCP")
        if kwargs["attn_type"] != AttentionType.DECODER:
            raise ValueError("Hy4 supports decoder self-attention")
        if kwargs["alibi_slopes"] is not None or kwargs["logits_soft_cap"] is not None:
            raise ValueError("Hy4 does not support ALiBi or logits soft caps")
        supported = Hy4TritonSparseBackend.supported_kv_cache_dtypes
        if kwargs["kv_cache_dtype"] not in supported:
            raise ValueError("Hy4 requires auto/BF16/FP16 or explicit fp8_ds_mla")
        if (
            kwargs["kv_cache_dtype"] == "fp8_ds_mla"
            and torch.get_default_dtype() != torch.bfloat16
        ):
            raise ValueError("Hy4 fp8_ds_mla requires BF16 model inputs")
        if kwargs["kv_lora_rank"] != 512 or kwargs["qk_rope_head_dim"] != 64:
            raise ValueError("Hy4 requires latent512 + rotary64")
        heads = kwargs["num_heads"]
        if sinks is not None and (
            sinks.shape != (heads * self.dcp_world_size,)
            or sinks.dtype != torch.float32
        ):
            raise ValueError("Hy4 sinks must be FP32 [local_heads * dcp_world_size]")
        super().__init__(**kwargs)
        if self.topk_indices_buffer is None:
            raise ValueError("Hy4 requires a logical top-k buffer")
        self.sinks = sinks
        self.masked_mha_available = False
        # The split partials of the largest call
        # this layer serves (a full chunk; fewer than 32 rows split 8 ways), reserved
        # here so they are resident before the KV pin is allocated
        if torch.cuda.is_available():
            chunk = get_current_vllm_config().scheduler_config.max_num_batched_tokens
            reserve_sparse_workspace(
                torch.device("cuda", torch.cuda.current_device()),
                max(int(chunk), 31 * 8), heads * self.dcp_world_size,
            )

    def _forward_mqa_kernel(
        self,
        q,
        kv_cache,
        topk_indices,
        valid_counts,
        *,
        layer,
        block_size,
        is_decode,
    ):
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        rows, _ = flat_kv_row_view(kv_cache, block_size)
        # SWA can have leading -1 holes. valid_counts is not a prefix length.
        # Opt-in decode path (same contract)
        if is_decode and _SPARSE_DECODE and q.shape[0] <= DECODE_MAX_ROWS:
            output, lse = sparse_mla_decode(
                q, rows, topk_indices, self.scale, sinks=self.sinks,
                head_block=_SPARSE_DECODE_HEAD_BLOCK,
            )
            return output, lse if self.need_to_return_lse_for_decode else None
        output, lse = sparse_mla(q, rows, topk_indices, self.scale, sinks=self.sinks)
        return output, lse if self.need_to_return_lse_for_decode else None


class Hy4TritonSWAImpl(Hy4ReplicatedKVImplMixin, Hy4TritonSparseImpl):
    """Replicated SWA KV uses local TP heads and a local sink."""

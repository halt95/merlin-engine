# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""GDN attention metadata for RecoverSSM speculative decode (Qwen-style Gated DeltaNet).

Wraps the generic GDN builder: every active decode row runs through the spec path (a
draft-less step is a T=1 window), the spec state indices keep only the checkpoint
column, num_accepted_tokens is always 1 (the checkpoint and the compacted conv window
already hold the accepted state), and the metadata carries what the post-sampling commit
needs. Mirrors the Kimi-K3 KDA builder's RecoverSSM handling."""

from dataclasses import dataclass, field, replace
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.recoverssm_utils import (
    recoverssm_binding_signature as _binding_signature,
)
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
    clone_commit_tensors,
    recoverssm_request_indices,
    recoverssm_spec_rows,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    mamba_align_gather_index,
    mamba_get_block_table_tensor,
)

logger = init_logger(__name__)


@dataclass
class GDNRecoverSSMCommitMetadata:
    state_indices: torch.Tensor  # [num_spec_decodes, 1]
    query_start_loc: torch.Tensor  # [num_spec_decodes + 1]
    request_indices: (
        torch.Tensor | None
    )  # batch rows of the spec requests (None: rows 0..n-1)
    block_table: torch.Tensor | None  # align mode
    num_computed_tokens: torch.Tensor | None
    block_size: int | None


@dataclass
class GDNRecoverSSMMetadata(GDNAttentionMetadata, RecoverSSMMetadata):
    recoverssm_commit: GDNRecoverSSMCommitMetadata | None = None
    recoverssm_context: Any = field(default=None, repr=False, compare=False)
    # The batch's seq_lens, so `update_block_table` can derive another group's
    # align-mode state indices exactly as `build` does.
    recoverssm_seq_lens: torch.Tensor | None = field(
        default=None, repr=False, compare=False
    )
    recoverssm_align_gather_index: torch.Tensor | None = field(
        default=None, repr=False, compare=False
    )

    def commit_recoverssm_state(
        self,
        num_accepted_tokens: torch.Tensor,
        *,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> RecoverSSMPostprocessMetadata | None:
        """Run this group's commit. ``num_computed_tokens`` replaces the align-mode
        count tensor with a staged copy of the same values (the commit-graph path,
        whose replays read fixed addresses); None keeps the metadata's own tensor."""
        c = self.recoverssm_commit
        if c is None:
            return None
        n = self.num_spec_decodes
        nc = c.num_computed_tokens
        if num_computed_tokens is not None:
            if nc is None:
                raise ValueError(
                    "GDN RecoverSSM: num_computed_tokens override outside align mode"
                )
            nc = num_computed_tokens
        self.recoverssm_context.commit(
            num_accepted_tokens,
            c.state_indices[:n, 0],
            c.query_start_loc[: n + 1],
            request_indices=c.request_indices,
            block_table=c.block_table,
            num_computed_tokens=nc,
            mamba_block_size=c.block_size,
        )
        if c.block_table is None:
            return None
        assert c.block_size is not None and nc is not None
        return RecoverSSMPostprocessMetadata(
            num_spec_decodes=n,
            request_indices=c.request_indices,
            block_table=c.block_table,
            num_computed_tokens=nc,
            block_size=c.block_size,
        )

    def snapshot_for_deferred_commit(
        self, copy: Any = None
    ) -> "GDNRecoverSSMMetadata":
        c = self.recoverssm_commit
        if c is None:
            return self
        return replace(
            self,
            recoverssm_commit=clone_commit_tensors(c, self.num_spec_decodes, copy),
        )


def _check_context_binding(
    context: Any, checkpoints: list[torch.Tensor], bound: tuple | None = None
) -> None:
    """A cached commit context must point at the layers' live KV cache.

    The context stores raw base addresses of every layer's conv, state and
    replay tensors. One that outlived its cache (the profiling KV cache is
    torn down and re-allocated before serving) would keep that cache alive
    and, worse, commit into freed memory. Builders are re-created with the
    real cache, so a mismatch is a lifecycle bug: fail loudly.

    ``bound`` is the checkpoints' signature recorded when the context was
    built. Without it the context's own tensors are signed now, which cannot
    see a storage change made to those same tensor objects."""
    if bound is None:
        bound = _binding_signature(context.checkpoints)
    live = _binding_signature(checkpoints)
    if bound != live:
        raise RuntimeError(
            "GDN RecoverSSM commit context is bound to a different KV cache than "
            "its layers (stale context after a KV cache re-allocation)"
        )


def recoverssm_reuse_gathers(
    block_table_tensor: torch.Tensor,
    masks: torch.Tensor | None,
    *,
    num_prefills: int,
    num_decodes: int,
    num_spec_decodes: int,
    num_spec: int,
    fastpath_prefix_len: int | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """(spec, non-spec, prefill) state-index gathers for a reused group.

    A pure spec-decode batch (no prefill and no ordinary-decode rows) whose
    spec rows are exactly the leading ``num_spec_decodes`` rows takes the
    prefix slice, as upstream's ``GDNAttentionMetadataBuilder.update_block_table``
    and ``build()`` do (no host-mask H2D copy and gather). The leading-rows
    check makes the slice the same rows as the mask gather by construction
    (``num_spec_decodes`` is the mask's popcount); otherwise the mask gather
    is used as before. ``fastpath_prefix_len`` is the optional decode fast
    path's own slice (VLLM_GDN_DECODE_FASTPATH), unchanged.
    """
    spec_indices = non_spec_indices = prefill_indices = None
    if masks is None:
        non_spec_indices = block_table_tensor[:, 0]
        if num_prefills > 0:
            prefill_indices = non_spec_indices[num_decodes:]
    elif fastpath_prefix_len is not None:
        # Decode fast path batch: the spec rows are rows 0..n-1 and there
        # are no prefill rows, so the mask gather is this slice.
        spec_indices = block_table_tensor[:fastpath_prefix_len, : num_spec + 1]
    elif (
        num_prefills == 0 and num_decodes == 0 and bool(masks[:num_spec_decodes].all())
    ):
        # Padded rows trail the spec decodes: the mask gather is this slice.
        spec_indices = block_table_tensor[:num_spec_decodes, : num_spec + 1]
    else:
        spec_indices = block_table_tensor[masks, : num_spec + 1]
        if num_prefills > 0:
            non_spec_indices = prefill_indices = block_table_tensor[~masks, 0]
    return spec_indices, non_spec_indices, prefill_indices


def _promote(meta: GDNAttentionMetadata, **extra: Any) -> GDNRecoverSSMMetadata:
    """The base builder's metadata as GDNRecoverSSMMetadata: the same field values
    plus ``extra``. Equivalent to ``GDNRecoverSSMMetadata(**{f.name: getattr(meta,
    f.name) for f in fields(meta)}, **extra)`` (both are plain dataclasses without
    ``__post_init__`` or slots) without the per-step ``fields()`` walk and keyword
    re-binding of ~30 fields."""
    out = GDNRecoverSSMMetadata.__new__(GDNRecoverSSMMetadata)
    d = out.__dict__
    d.update(meta.__dict__)
    d["recoverssm_commit"] = None
    d["recoverssm_context"] = None
    d["recoverssm_seq_lens"] = None
    d["recoverssm_align_gather_index"] = None
    d.update(extra)
    return out


class GDNRecoverSSMMetadataBuilder(GDNAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    _overrides_num_accepted_tokens = True

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        max_reqs = vllm_config.scheduler_config.max_num_seqs
        self._ones = torch.ones(max_reqs, dtype=torch.int32, device=device)
        self.shared_align_index: bool = (
            bool(envs.VLLM_GDN_SHARED_ALIGN_INDEX)
            and vllm_config.cache_config.mamba_cache_mode == "align"
        )
        self._align_offsets: torch.Tensor | None = None
        if self.shared_align_index:
            self._align_offsets = torch.arange(
                1 + kv_cache_spec.num_speculative_blocks,
                device=device,
                dtype=torch.int32,
            )
        # Handed from `build` to the base build's `_mamba_state_block_table`.
        self._pending_state_block_table: torch.Tensor | None = None
        self._context: Any = None
        self._context_binding: tuple | None = None
        self._logged = False
        self._capturing = False
        self.supports_update_block_table = (
            vllm_config.use_v2_model_runner and device.type == "cuda"
        )

    def _get_context(self):
        if self._context is not None:
            fc = self.vllm_config.compilation_config.static_forward_context
            live = [fc[name].kv_cache[1] for name in self.layer_names]
            _check_context_binding(self._context, live, self._context_binding)
        if self._context is None:
            from vllm.model_executor.layers.mamba.gdn.recoverssm_gdn import (
                GDNRecoverSSMCommitContext,
            )

            fc = self.vllm_config.compilation_config.static_forward_context
            layers = [fc[name] for name in self.layer_names]
            self._context = GDNRecoverSSMCommitContext.create(
                layers,
                spec_query_len=1 + self.num_spec,
                max_num_reqs=self.vllm_config.scheduler_config.max_num_seqs,
            )
            self._context_binding = _binding_signature(
                [layer.kv_cache[1] for layer in layers]
            )
            logger.info(
                "GDN RecoverSSM commit context: %d layers, spec_query_len %d",
                len(layers),
                1 + self.num_spec,
            )
        return self._context

    def _mamba_state_block_table(self, m: CommonAttentionMetadata) -> torch.Tensor:
        table = self._pending_state_block_table
        if table is not None:
            self._pending_state_block_table = None
            return table
        return super()._mamba_state_block_table(m)

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> "GDNRecoverSSMMetadata":
        m = common_attn_metadata
        rows = None
        if self.use_spec_decode and num_decode_draft_tokens_cpu is not None:
            if self._capturing and m.is_prefilling is None:
                # CUDA graph capture: dummy decode batches, every row a spec window.
                m = m.replace(is_prefilling=torch.zeros(m.num_reqs, dtype=torch.bool))
            if m.is_prefilling is None or m.is_prefilling.device.type != "cpu":
                raise ValueError("GDN RecoverSSM needs the CPU is_prefilling mask")
            num_decode_draft_tokens_cpu, rows = recoverssm_spec_rows(
                m.is_prefilling,
                m.query_start_loc_cpu,
                num_decode_draft_tokens_cpu,
                m.num_reqs,
            )
        align_index = None
        if (
            self.shared_align_index
            and getattr(self, "mamba_aligned_state_indices", None) is None
        ):
            assert self._align_offsets is not None
            align_index = mamba_align_gather_index(
                m.seq_lens, self.kv_cache_spec, self._align_offsets
            )
            # == mamba_get_block_table_tensor(m.block_table_tensor, m.seq_lens,
            # spec, "align"), the table the base build derives otherwise.
            self._pending_state_block_table = torch.gather(
                m.block_table_tensor, 1, align_index
            )
        try:
            meta = super().build(
                common_prefix_len,
                m,
                num_accepted_tokens=num_accepted_tokens,
                num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
                fast_build=fast_build,
            )
        finally:
            self._pending_state_block_table = None
        extra: dict[str, Any] = {"recoverssm_seq_lens": m.seq_lens}
        if align_index is not None:
            extra["recoverssm_align_gather_index"] = align_index
        if meta.num_spec_decodes > 0:
            n = meta.num_spec_decodes
            assert meta.spec_state_indices_tensor is not None
            state_indices = meta.spec_state_indices_tensor[:, :1]
            extra["spec_state_indices_tensor"] = state_indices
            # One accepted token per row: the checkpoint already holds the accepted
            # state. Sized like the (possibly graph-padded) state indices.
            extra["num_accepted_tokens"] = self._ones[: state_indices.shape[0]]
            if rows is None:
                raise ValueError("GDN RecoverSSM: spec rows without draft counts")
            request_indices = recoverssm_request_indices(
                rows, n, m.query_start_loc.device
            )
            align = self.vllm_config.cache_config.mamba_cache_mode == "align"
            extra["recoverssm_commit"] = GDNRecoverSSMCommitMetadata(
                state_indices=state_indices,
                query_start_loc=meta.spec_query_start_loc,
                request_indices=request_indices,
                block_table=m.block_table_tensor if align else None,
                num_computed_tokens=m.compute_num_computed_tokens() if align else None,
                block_size=self.kv_cache_spec.block_size if align else None,
            )
            extra["recoverssm_context"] = self._get_context()
            if not self._logged:
                self._logged = True
                logger.info(
                    "GDN RecoverSSM path taken: %d spec rows, align=%s", n, align
                )
        return _promote(meta, **extra)

    def update_block_table(  # type: ignore[override]
        self,
        metadata: GDNRecoverSSMMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> GDNRecoverSSMMetadata:
        """Metadata for another GDN group of the batch `metadata` was built for.

        Every field `build` derives from this group's block table is re-derived
        the same way: `mamba_get_block_table_tensor` on this group's table and
        the batch seq_lens, the same spec/non-spec/prefill gathers, staging into
        this builder's own state-index buffers, the checkpoint column, and a
        commit with this group's state indices, block table, block size and
        context. The rest is batch-level (the same query offsets, seq_lens, CPU
        classification and accepted counts feed every group), so it is shared
        with `metadata` instead of being rebuilt: the spec-row classification,
        query offsets, request rows and computed-token counts of the commit."""
        m = metadata
        if not isinstance(m, GDNRecoverSSMMetadata):
            raise TypeError("GDN RecoverSSM can only reuse RecoverSSM metadata")
        align_index = m.recoverssm_align_gather_index
        if align_index is not None and self.shared_align_index:
            block_table_tensor = torch.gather(blk_table, 1, align_index)
        else:
            block_table_tensor = mamba_get_block_table_tensor(
                blk_table,
                m.recoverssm_seq_lens,
                self.kv_cache_spec,
                self.vllm_config.cache_config.mamba_cache_mode,
            )
        # Same gathers as build() / GDNAttentionMetadataBuilder.update_block_table.
        spec_indices, non_spec_indices, prefill_indices = recoverssm_reuse_gathers(
            block_table_tensor,
            m.spec_sequence_masks_cpu,
            num_prefills=m.num_prefills,
            num_decodes=m.num_decodes,
            num_spec_decodes=m.num_spec_decodes,
            num_spec=self.num_spec,
            fastpath_prefix_len=(
                m.spec_mask_prefix_len if self.decode_fastpath else None
            ),
        )

        if self._stage_spec_decode(
            m.num_prefills, m.num_decodes, m.num_spec_decodes, m.num_spec_decode_tokens
        ):
            assert m.spec_state_indices_tensor is not None
            batch_size = m.spec_state_indices_tensor.shape[0]
            self.spec_state_indices_tensor[: m.num_spec_decodes].copy_(
                spec_indices, non_blocking=True
            )
            spec_indices = self.spec_state_indices_tensor[:batch_size]
            spec_indices[m.num_spec_decodes :].fill_(NULL_BLOCK_ID)

        if self._stage_decode(m.num_prefills, m.num_decodes, m.num_spec_decodes):
            assert m.non_spec_state_indices_tensor is not None
            batch_size = m.non_spec_state_indices_tensor.shape[0]
            self.non_spec_state_indices_tensor[: m.num_decodes].copy_(
                non_spec_indices, non_blocking=True
            )
            non_spec_indices = self.non_spec_state_indices_tensor[:batch_size]
            non_spec_indices[m.num_decodes :].fill_(NULL_BLOCK_ID)

        changes: dict[str, Any] = {
            "spec_state_indices_tensor": spec_indices,
            "non_spec_state_indices_tensor": non_spec_indices,
            "prefill_state_indices": prefill_indices,
            "checkpoint": (
                m.checkpoint.regather_state_indices(blk_table)
                if m.checkpoint is not None
                else None
            ),
        }
        first = m.recoverssm_commit
        if m.num_spec_decodes > 0:
            assert spec_indices is not None and first is not None
            state_indices = spec_indices[:, :1]
            changes["spec_state_indices_tensor"] = state_indices
            changes["num_accepted_tokens"] = self._ones[: state_indices.shape[0]]
            align = first.block_table is not None
            changes["recoverssm_commit"] = GDNRecoverSSMCommitMetadata(
                state_indices=state_indices,
                query_start_loc=first.query_start_loc,
                request_indices=first.request_indices,
                block_table=blk_table if align else None,
                num_computed_tokens=first.num_computed_tokens,
                block_size=self.kv_cache_spec.block_size if align else None,
            )
            changes["recoverssm_context"] = self._get_context()
        return replace(m, **changes)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        self._capturing = True
        try:
            return super().build_for_cudagraph_capture(common_attn_metadata)
        finally:
            self._capturing = False


class GDNRecoverSSMAttentionBackend(GDNAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_RECOVERSSM"

    @staticmethod
    def get_builder_cls() -> type[GDNRecoverSSMMetadataBuilder]:
        return GDNRecoverSSMMetadataBuilder

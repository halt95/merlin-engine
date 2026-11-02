# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Mamba-2 attention metadata for RecoverSSM speculative decode (NemotronH, Nemotron
3.5 Super VL).

Wraps the Mamba-2 builder the way ``gdn_recoverssm.py`` wraps the GDN one: every active
decode row is a verify window (a draft-less step is a T=1 window) read from the
checkpoint column of the state indices, ``num_accepted_tokens`` is always 1 (the
checkpoint and the compacted conv window already hold the accepted state), and the
metadata carries what the post-sampling commit needs. The Mamba-2 prefill path (chunk
metadata, internal prefill checkpoints) is the parent builder's, unchanged.

Row classification. Mamba-2 metadata has one decode class (``state_indices_tensor_d``,
``query_start_loc_d``), and every decode row runs the verify kernel and later commits
``num_sampled`` tokens. So a decode row must be a row that samples: exactly the
``recoverssm_spec_rows`` rows (post-prefill rows and the final prompt tails the runner
already marked as decodes). The parent's own reclassification of one-token stateful
prefill rows is turned off (``_reclassify_one_token_prefills``), since a non-final
one-token chunk samples nothing and its token would never reach the state."""

from dataclasses import dataclass, field, replace
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.recoverssm_utils import (
    recoverssm_binding_signature as _binding_signature,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mamba2_attn import (
    Mamba2AttentionBackend,
    Mamba2AttentionMetadata,
    Mamba2AttentionMetadataBuilder,
)
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
    clone_commit_tensors,
    recoverssm_request_indices,
    recoverssm_spec_rows,
)

logger = init_logger(__name__)


@dataclass
class Mamba2RecoverSSMCommitMetadata:
    state_indices: torch.Tensor  # [num_spec_decodes (+graph padding), 1]
    query_start_loc: torch.Tensor  # [num_spec_decodes (+graph padding) + 1]
    # Batch rows of the spec requests; always None here: Mamba-2 decode rows are the
    # leading batch rows (the field keeps the generic commit/snapshot contract).
    request_indices: torch.Tensor | None
    block_table: torch.Tensor | None  # align mode
    num_computed_tokens: torch.Tensor | None  # align mode, pre-verify counts
    block_size: int | None
    # Activation tokens the verify ran with (x.shape[0] of the decode rows); the
    # commit caps every row to what the verify recorded.
    num_tokens: int | None = None


@dataclass
class Mamba2RecoverSSMMetadata(Mamba2AttentionMetadata, RecoverSSMMetadata):
    # Real verify rows: the leading decode rows. `num_decodes` also counts the
    # zero-length padding rows of a FULL decode graph, which commit nothing.
    num_spec_decodes: int = 0
    recoverssm_commit: Mamba2RecoverSSMCommitMetadata | None = None
    recoverssm_context: Any = field(default=None, repr=False, compare=False)

    def commit_recoverssm_state(
        self,
        num_accepted_tokens: torch.Tensor,
        *,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> RecoverSSMPostprocessMetadata | None:
        """Run this group's commit (the GDN contract, ``gdn_recoverssm.py``).
        ``num_computed_tokens`` replaces the align-mode count tensor with a staged
        copy of the same values; None keeps the metadata's own tensor."""
        c = self.recoverssm_commit
        if c is None:
            return None
        n = self.num_spec_decodes
        nc = c.num_computed_tokens
        if num_computed_tokens is not None:
            if nc is None:
                raise ValueError(
                    "Mamba-2 RecoverSSM: num_computed_tokens override outside "
                    "align mode"
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
            num_tokens=c.num_tokens,
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

    def snapshot_for_deferred_commit(self) -> "Mamba2RecoverSSMMetadata":
        """The commit inputs cloned (trimmed to the real decode rows), so the
        builds between this step and its PP-deferred commit cannot overwrite
        them: the state indices may be the builder's persistent graph buffer and
        the block table / query offsets are the runner's per-step buffers. The
        context (KV-cache address tables) is shared, as for GDN."""
        c = self.recoverssm_commit
        if c is None:
            return self
        return replace(
            self, recoverssm_commit=clone_commit_tensors(c, self.num_spec_decodes)
        )


def _check_context_binding(
    context: Any, checkpoints: list[torch.Tensor], bound: tuple | None = None
) -> None:
    """A cached commit context must point at the layers' live KV cache (the
    profiling cache is torn down and re-allocated before serving; a stale
    context would commit into freed memory). Same rule as GDN."""
    if bound is None:
        bound = _binding_signature(context.checkpoints)
    if bound != _binding_signature(checkpoints):
        raise RuntimeError(
            "Mamba-2 RecoverSSM commit context is bound to a different KV cache "
            "than its layers (stale context after a KV cache re-allocation)"
        )


def _promote(meta: Mamba2AttentionMetadata, **extra: Any) -> Mamba2RecoverSSMMetadata:
    """The parent's metadata as Mamba2RecoverSSMMetadata (same field values plus
    ``extra``). The parent builds ``metadata_cls`` instances already; this keeps
    the type right for a parent that returns its own class."""
    if isinstance(meta, Mamba2RecoverSSMMetadata):
        return replace(meta, **extra) if extra else meta
    out = Mamba2RecoverSSMMetadata.__new__(Mamba2RecoverSSMMetadata)
    d = out.__dict__
    d.update(meta.__dict__)
    d["num_spec_decodes"] = 0
    d["recoverssm_commit"] = None
    d["recoverssm_context"] = None
    d.update(extra)
    return out


class Mamba2RecoverSSMMetadataBuilder(Mamba2AttentionMetadataBuilder):
    metadata_cls = Mamba2RecoverSSMMetadata
    # Every metadata with decode rows carries self._ones as num_accepted_tokens.
    _overrides_num_accepted_tokens = True
    # Rows are classified by `recoverssm_spec_rows` (module docstring).
    _reclassify_one_token_prefills = False

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        max_rows = max(
            vllm_config.scheduler_config.max_num_seqs, self.decode_cudagraph_max_bs
        )
        self._ones = torch.ones(max_rows, dtype=torch.int32, device=device)
        if (
            vllm_config.cache_config.mamba_cache_mode == "align"
            and kv_cache_spec.block_size < 1 + self.num_spec_tokens
        ):
            # The commit plan needs a window to cross at most one boundary.
            raise ValueError(
                "Mamba-2 RecoverSSM: the Mamba block size must cover one verify "
                f"window ({kv_cache_spec.block_size} < {1 + self.num_spec_tokens})"
            )
        self._context: Any = None
        self._context_binding: tuple | None = None
        self._capturing = False
        self._logged = False

    def _get_context(self):
        fc = self.vllm_config.compilation_config.static_forward_context
        if self._context is not None:
            live = [fc[name].kv_cache[1] for name in self.layer_names]
            _check_context_binding(self._context, live, self._context_binding)
            return self._context
        from vllm.model_executor.layers.mamba.recoverssm_mamba2 import (
            Mamba2RecoverSSMCommitContext,
        )

        layers = [fc[name] for name in self.layer_names]
        self._context = Mamba2RecoverSSMCommitContext.create(
            layers,
            spec_query_len=1 + self.num_spec_tokens,
            max_num_reqs=self.vllm_config.scheduler_config.max_num_seqs,
        )
        self._context_binding = _binding_signature(
            [layer.kv_cache[1] for layer in layers]
        )
        logger.info(
            "Mamba-2 RecoverSSM commit context: %d layers, spec_query_len %d",
            len(layers),
            1 + self.num_spec_tokens,
        )
        return self._context

    def _classify_rows(
        self,
        m: CommonAttentionMetadata,
        num_decode_draft_tokens_cpu: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
        """(metadata whose is_prefilling marks exactly the non-spec real rows,
        draft counts for the parent, spec rows in batch order). Real rows have
        tokens; zero-length graph-padding rows stay non-prefilling, so the
        parent's split keeps them in a padded decode-only batch."""
        if self._capturing and m.is_prefilling is None:
            # CUDA graph capture: dummy decode batches, every row a spec window.
            m = m.replace(is_prefilling=torch.zeros(m.num_reqs, dtype=torch.bool))
        if m.is_prefilling is None or m.is_prefilling.device.type != "cpu":
            raise ValueError("Mamba-2 RecoverSSM needs the CPU is_prefilling mask")
        drafts, rows = recoverssm_spec_rows(
            m.is_prefilling, m.query_start_loc_cpu, num_decode_draft_tokens_cpu, m.num_reqs
        )
        query_lens = torch.diff(m.query_start_loc_cpu[: m.num_reqs + 1])
        # A row wider than one verify window cannot be a spec row. Real decode
        # rows never are; compile-warmup dummies are non-prefilling rows of any
        # width (input_batch.make_dummy), so they run as prefills.
        wide = query_lens[rows] > 1 + self.num_spec_tokens
        if bool(wide.any()):
            drafts = drafts.clone()
            drafts[rows[wide]] = -1
            rows = rows[~wide]
        is_prefilling = m.is_prefilling.clone()
        is_prefilling[: m.num_reqs] = query_lens > 0
        is_prefilling[rows] = False
        return m.replace(is_prefilling=is_prefilling), drafts, rows

    @staticmethod
    def _check_decode_rows(
        m: CommonAttentionMetadata, rows: torch.Tensor, num_decodes: int
    ) -> int:
        """The parent's decode rows must be the spec rows (leading, in batch
        order) plus only zero-length graph padding. Returns the spec row count."""
        n = rows.numel()
        # The parent's torch.split takes the leading rows as decodes.
        if recoverssm_request_indices(rows, n, rows.device) is not None:
            raise ValueError(
                "Mamba-2 RecoverSSM: the spec rows must be the leading batch rows"
            )
        qsl = m.query_start_loc_cpu
        if num_decodes < n or int(qsl[num_decodes]) != int(qsl[n]):
            raise ValueError(
                f"Mamba-2 RecoverSSM: {n} spec rows, but the builder produced "
                f"{num_decodes} decode rows with tokens beyond them"
            )
        return n

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs: Any,
    ) -> Mamba2RecoverSSMMetadata:
        m = common_attn_metadata
        rows = None
        drafts_cpu = kwargs.get("num_decode_draft_tokens_cpu")
        if self.use_spec_decode and drafts_cpu is not None:
            m, kwargs["num_decode_draft_tokens_cpu"], rows = self._classify_rows(
                m, drafts_cpu
            )
        if m.num_reqs > self._ones.numel():
            raise ValueError("Mamba-2 RecoverSSM: batch exceeds the builder capacity")
        meta = super().build(common_prefix_len, m, fast_build, **kwargs)
        n = 0
        if rows is not None:
            n = self._check_decode_rows(m, rows, meta.num_decodes)
        if not self.use_spec_decode or meta.num_decodes == 0:
            return _promote(meta)
        if rows is None:
            if not self._capturing:
                raise ValueError("Mamba-2 RecoverSSM: decode rows without draft counts")
            n = meta.num_decodes  # capture dummies: every row is a real window
        assert meta.state_indices_tensor_d is not None
        assert meta.query_start_loc_d is not None
        # The checkpoint column only (the verify's S0 and the commit's source);
        # a view of the persistent graph buffer under FULL decode graphs.
        state_indices = meta.state_indices_tensor_d[:, :1]
        extra: dict[str, Any] = {
            "num_spec_decodes": n,
            "state_indices_tensor_d": state_indices,
            # One accepted token per row: the checkpoint holds the accepted
            # state. Sized like the (possibly graph-padded) state indices.
            "num_accepted_tokens": self._ones[: state_indices.shape[0]],
        }
        if n > 0:
            align = self.vllm_config.cache_config.mamba_cache_mode == "align"
            extra["recoverssm_commit"] = Mamba2RecoverSSMCommitMetadata(
                state_indices=state_indices,
                query_start_loc=meta.query_start_loc_d,
                request_indices=None,
                block_table=m.block_table_tensor if align else None,
                num_computed_tokens=(
                    m.compute_num_computed_tokens() if align else None
                ),
                block_size=self.kv_cache_spec.block_size if align else None,
                num_tokens=meta.num_decode_tokens,
            )
            extra["recoverssm_context"] = self._get_context()
            if not self._logged and not self._capturing:
                self._logged = True
                logger.info(
                    "Mamba-2 RecoverSSM path taken: %d spec rows, align=%s", n, align
                )
        return _promote(meta, **extra)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        self._capturing = True
        try:
            return super().build_for_cudagraph_capture(common_attn_metadata)
        finally:
            self._capturing = False


class Mamba2RecoverSSMAttentionBackend(Mamba2AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "MAMBA2_RECOVERSSM"

    @staticmethod
    def get_builder_cls() -> type[Mamba2RecoverSSMMetadataBuilder]:
        return Mamba2RecoverSSMMetadataBuilder


__all__ = [
    "Mamba2RecoverSSMAttentionBackend",
    "Mamba2RecoverSSMCommitMetadata",
    "Mamba2RecoverSSMMetadata",
    "Mamba2RecoverSSMMetadataBuilder",
]

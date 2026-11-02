# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""RecoverSSM commit context for Mamba-2 layers (NemotronH / Nemotron Super VL).

Holds, per attention group, the raw address tables of every layer's conv state,
fp32 SSM checkpoint and fp32 replay record (``layer.kv_cache[0..2]``) plus the
per-step plan scratch, and runs the post-sampling commit chain: the generic
``prepare_commit_plan_kernel`` (commit length, final and align-boundary blocks), the
generic ``compact_conv_state_kernel`` (accepted conv suffix to the front) and the
Mamba-2 ``_commit_mamba2_state_kernel`` (one traversal: the align-boundary state is
stored when reached, then the final state; either may alias the source). It follows
the GDN commit path; the metadata/builder (``Mamba2RecoverSSMMetadata``)
calls ``commit`` exactly as the GDN metadata does."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm.merlin.envs import read_registered
from vllm.model_executor.layers.mamba.ops.recoverssm import (
    compact_conv_state_kernel,
    prepare_commit_plan_kernel,
)
from vllm.model_executor.layers.mamba.ops.recoverssm_mamba2 import (
    MAMBA2_RECOVERSSM_RECORD_DTYPE,
    _commit_mamba2_state_kernel,
    mamba2_recoverssm_record_shape,
    mamba2_recoverssm_verify,
)
from vllm.model_executor.layers.mamba.recoverssm_utils import recoverssm_require
from vllm.triton_utils import triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

# Debug only: assert the source-block ownership invariant on every commit (host sync).
_CHECK_OWNERSHIP = bool(read_registered("VLLM_RECOVERSSM_CHECK_OWNERSHIP"))


def _require(cond: bool, msg: str) -> None:
    recoverssm_require(cond, msg, "Mamba2 RecoverSSM")


@dataclass(frozen=True)
class Mamba2RecoverSSMCommitContext:
    conv_states: tuple[torch.Tensor, ...]
    conv_state_base_addrs: torch.Tensor
    conv_state_block_strides: torch.Tensor
    conv_state_dim_strides: torch.Tensor
    conv_state_token_strides: torch.Tensor
    conv_history_len: int
    checkpoints: tuple[torch.Tensor, ...]
    state_base_addrs: torch.Tensor
    state_block_strides: torch.Tensor
    replays: tuple[torch.Tensor, ...]
    replay_base_addrs: torch.Tensor
    replay_block_strides: torch.Tensor
    commit_lens: torch.Tensor
    final_state_indices: torch.Tensor
    boundary_state_indices: torch.Tensor
    boundary_recovery_lens: torch.Tensor
    spec_query_len: int

    @classmethod
    def from_tensors(
        cls,
        conv_states: Sequence[torch.Tensor],
        checkpoints: Sequence[torch.Tensor],
        replays: Sequence[torch.Tensor],
        *,
        spec_query_len: int,
        max_num_reqs: int,
        conv_dim_first: bool = True,
    ) -> "Mamba2RecoverSSMCommitContext":
        _require(len(checkpoints) > 0, "commit requires at least one layer")
        _require(
            len(conv_states) == len(checkpoints) == len(replays),
            "conv, state and replay lists differ",
        )
        _require(spec_query_len >= 1, "spec_query_len must be at least 1")
        if not conv_dim_first:
            # SD layout [blocks, window, dim] -> a [blocks, dim, window] view, so the
            # stride tables below are (dim, token) for either layout.
            conv_states = [s.transpose(-1, -2) for s in conv_states]
        ref = checkpoints[0]
        _require(ref.ndim == 4, "checkpoint must be [blocks, heads, head_dim, dstate]")
        _require(ref.dtype == torch.float32, "checkpoint must use float32")
        nb, HL, P, N = ref.shape
        dev = ref.device
        for s in checkpoints:
            _require(
                s.shape == ref.shape
                and s.dtype == ref.dtype
                and s.stride()[1:] == ref.stride()[1:]
                and s.device == dev,
                "layers need matching checkpoints",
            )
        rr = replays[0]
        _require(rr.ndim == 4 and rr.shape[0] == nb, "replay must be [blocks, G, T, R]")
        G = rr.shape[1]
        _require(G > 0 and HL % G == 0, "heads must be a multiple of the record groups")
        rec = (nb, *mamba2_recoverssm_record_shape(HL, G, P, N, spec_query_len - 1))
        for r in replays:
            _require(
                tuple(r.shape) == rec
                and r.dtype == MAMBA2_RECOVERSSM_RECORD_DTYPE
                and r.stride()[1:] == rr.stride()[1:]
                and r.device == dev,
                f"layers need matching float32 replay buffers of shape {rec}",
            )
        conv_ref = conv_states[0]
        _require(
            conv_ref.ndim == 3 and conv_ref.shape[0] == nb,
            "conv state must be [blocks, dim, window] with the checkpoint's block count",
        )
        for c in conv_states:
            _require(
                c.shape == conv_ref.shape
                and c.dtype == conv_ref.dtype
                and c.stride()[1:] == conv_ref.stride()[1:]
                and c.device == dev,
                "layers need matching conv states",
            )
        conv_len = conv_ref.shape[2]
        hist = conv_len - spec_query_len + 1
        _require(hist > 0, "conv state is shorter than its window")
        addr = lambda ts: torch.tensor(
            [t.data_ptr() for t in ts], dtype=torch.int64, device=dev
        )
        bstr = lambda ts: torch.tensor(
            [t.stride(0) for t in ts], dtype=torch.int64, device=dev
        )
        z = lambda: torch.empty(max_num_reqs, dtype=torch.int32, device=dev)
        return cls(
            tuple(conv_states),
            addr(conv_states),
            bstr(conv_states),
            torch.tensor([s.stride(1) for s in conv_states], dtype=torch.int64, device=dev),
            torch.tensor([s.stride(2) for s in conv_states], dtype=torch.int64, device=dev),
            hist,
            tuple(checkpoints),
            addr(checkpoints),
            bstr(checkpoints),
            tuple(replays),
            addr(replays),
            bstr(replays),
            z(),
            z(),
            z(),
            z(),
            spec_query_len,
        )

    @classmethod
    def create(
        cls, layers: Sequence[Any], *, spec_query_len: int, max_num_reqs: int
    ) -> "Mamba2RecoverSSMCommitContext":
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

        if any(len(layer.kv_cache) != 3 for layer in layers):
            raise ValueError(
                "Mamba2 RecoverSSM pages must contain conv, state and replay"
            )
        return cls.from_tensors(
            [layer.kv_cache[0] for layer in layers],
            [layer.kv_cache[1] for layer in layers],
            [layer.kv_cache[2] for layer in layers],
            spec_query_len=spec_query_len,
            max_num_reqs=max_num_reqs,
            conv_dim_first=is_conv_state_dim_first(),
        )

    def commit(
        self,
        num_accepted_tokens: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        request_indices: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        num_computed_tokens: torch.Tensor | None = None,
        mamba_block_size: int | None = None,
        commit_conv: bool = True,
        num_tokens: int | None = None,
    ) -> None:
        """Replay the accepted prefix into every layer's checkpoint (and the align
        boundary block) and compact the conv windows. Same contract as
        ``GDNRecoverSSMCommitContext.commit``.

        ``num_tokens`` is the activation token count the verify ran with
        (``x.shape[0]``); pass it so the commit never replays record slots the
        verify did not write. Rows are capped exactly as the verify capped them
        (negative or non-monotonic offsets commit nothing; endpoints clamp into
        ``[0, num_tokens]``), on device, with no host sync. Ownership invariant:
        a positive source block belongs to one request from verify to
        commit; ``VLLM_RECOVERSSM_CHECK_OWNERSHIP=1`` asserts it (debug, syncs)."""
        batch = state_indices.shape[0]
        if batch == 0:
            return
        _require(
            state_indices.ndim == 1
            and state_indices.dtype in (torch.int32, torch.int64),
            "state indices must be a one-dimensional integer tensor",
        )
        _require(
            batch <= self.commit_lens.shape[0], "commit batch exceeds its plan capacity"
        )
        _require(
            all(
                t_ is None or t_.dtype in (torch.int32, torch.int64)
                for t_ in (num_accepted_tokens, query_start_loc, request_indices,
                           block_table, num_computed_tokens)
            ),
            "commit metadata must be integer tensors",
        )
        _require(
            query_start_loc.ndim == 1 and query_start_loc.shape[0] == batch + 1,
            "commit metadata is incompatible",
        )
        _require(
            request_indices is None or request_indices.shape[0] >= batch,
            "request mapping is too short",
        )
        _require(
            num_accepted_tokens.ndim == 1
            and (request_indices is not None or num_accepted_tokens.shape[0] >= batch),
            "accepted-token counts are too short",
        )
        _align = (block_table, num_computed_tokens, mamba_block_size)
        _require(
            all(a is None for a in _align) or all(a is not None for a in _align),
            "align metadata is incomplete",
        )
        _require(
            mamba_block_size is None or mamba_block_size >= self.spec_query_len,
            "align block size must cover one speculative window",
        )
        _require(
            block_table is None or block_table.ndim == 2,
            "block table must be two-dimensional",
        )
        dev = self.checkpoints[0].device
        _require(
            all(
                t_ is None or t_.device == dev
                for t_ in (
                    num_accepted_tokens,
                    state_indices,
                    query_start_loc,
                    request_indices,
                    block_table,
                    num_computed_tokens,
                )
            ),
            "commit inputs must be on the same device",
        )
        if _CHECK_OWNERSHIP:
            live = state_indices[state_indices > NULL_BLOCK_ID]
            _require(
                live.unique().numel() == live.numel(),
                "a source block is shared by two rows in one commit",
            )
        # The shared plan derives each row's commit length from its offsets. Hand it
        # offsets rebuilt from the lengths the verify actually recorded, so a
        # malformed row can never replay stale record slots.
        qsl = query_start_loc[: batch + 1].to(torch.int64)
        bos, eos = qsl[:-1], qsl[1:]
        if num_tokens is not None:
            bos_c, eos_c = bos.clamp(0, num_tokens), eos.clamp(0, num_tokens)
        else:
            bos_c, eos_c = bos, eos
        rec_lens = torch.where(
            (bos < 0) | (eos < bos), 0, (eos_c - bos_c).clamp_min(0)
        )
        plan_qsl = torch.zeros(batch + 1, dtype=torch.int64, device=dev)
        torch.cumsum(rec_lens, 0, out=plan_qsl[1:])
        bt_stride = (0, 0) if block_table is None else block_table.stride()
        prepare_commit_plan_kernel[(batch,)](
            num_accepted_tokens,
            request_indices,
            state_indices,
            plan_qsl,
            block_table,
            num_computed_tokens,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            mamba_block_size or 1,
            block_table.shape[1] if block_table is not None else 1,
            num_accepted_tokens.stride(0),
            request_indices.stride(0) if request_indices is not None else 0,
            state_indices.stride(0),
            plan_qsl.stride(0),
            bt_stride[0],
            bt_stride[1],
            0 if num_computed_tokens is None else num_computed_tokens.stride(0),
            SPEC_QUERY_LEN=self.spec_query_len,
            num_warps=1,
        )
        num_layers = len(self.checkpoints)
        align = block_table is not None
        if commit_conv:
            conv_ref = self.conv_states[0]
            conv_dim = conv_ref.shape[1]
            compact_conv_state_kernel[(triton.cdiv(conv_dim, 256), batch, num_layers)](
                conv_ref,
                self.conv_state_base_addrs,
                self.conv_state_block_strides,
                self.conv_state_dim_strides,
                self.conv_state_token_strides,
                state_indices,
                self.commit_lens,
                self.final_state_indices,
                self.boundary_state_indices,
                self.boundary_recovery_lens,
                NULL_BLOCK_ID,
                conv_dim,
                self.conv_history_len,
                state_indices.stride(0),
                BLOCK_D=256,
                BLOCK_HISTORY=triton.next_power_of_2(self.conv_history_len),
                ALIGN_MODE=align,
                num_warps=4,
            )
        ref = self.checkpoints[0]
        _, HL, P, N = ref.shape
        rr = self.replays[0]
        G = rr.shape[1]
        HPG = HL // G
        # One 16-row tile per program (GDN's commit choice): the commit streams each
        # state once and smaller tiles keep more of them in flight.
        BLOCK_SIZE_M = min(triton.next_power_of_2(P), 16)
        args = (
            ref,
            self.state_base_addrs,
            self.state_block_strides,
            rr,
            self.replay_base_addrs,
            self.replay_block_strides,
            state_indices,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            ref.stride(1),
            ref.stride(2),
            ref.stride(3),
            rr.stride(1),
            rr.stride(2),
            rr.stride(3),
            state_indices.stride(0),
        )
        grid = (triton.cdiv(P, BLOCK_SIZE_M), batch, num_layers * HL)
        # One launch replays each tile once and stores the boundary state (align) and
        # the final state from registers, so either may alias the source block.
        _commit_mamba2_state_kernel[grid](
            *args,
            P=P,
            N=N,
            HPG=HPG,
            NUM_HEADS=HL,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_DSTATE=triton.next_power_of_2(N),
            ALIGN_MODE=align,
            num_warps=4,
            num_stages=2,
        )


__all__ = [
    "MAMBA2_RECOVERSSM_RECORD_DTYPE",
    "Mamba2RecoverSSMCommitContext",
    "mamba2_recoverssm_record_shape",
    "mamba2_recoverssm_verify",
]

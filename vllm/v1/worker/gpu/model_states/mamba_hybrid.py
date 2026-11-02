# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm import envs
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.merlin.flashinfer_smallq import (
    scheduled_prefill_flags,
)
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFuncsByType
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionMetadataBuilder,
    ShortConvAttentionMetadataBuilder,
)
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState
from vllm.v1.worker.gpu.pp_utils import compute_need_sampled_mask
from vllm.v1.worker.mamba_utils import (
    MambaSpecDecodeGPUContext,
    get_mamba_group_ids,
    get_mamba_groups,
    preprocess_mamba_align_fused_kernel,
    validate_mamba_state_copy_funcs,
)
from vllm.v1.worker.utils import AttentionGroup


@dataclass
class MambaHybridAttnMetadata(ModelSpecificAttnMetadata):
    is_prefilling: torch.Tensor
    num_accepted_tokens: torch.Tensor | None = None
    num_decode_draft_tokens_cpu: torch.Tensor | None = None

    workload_prefill_flags: list[bool] | None = None

    def get_extra_common_attn_kwargs(
        self, kv_cache_group_id: int, num_reqs: int
    ) -> dict[str, Any]:
        return {"is_prefilling": self.is_prefilling[:num_reqs]}

    def get_extra_attn_kwargs(
        self, attn_metadata_builder: Any, num_reqs: int
    ) -> dict[str, Any]:
        if getattr(attn_metadata_builder, "_workload_tile", False):
            return {"workload_prefill_flags": self.workload_prefill_flags}
        if not isinstance(
            attn_metadata_builder,
            (
                Mamba2AttentionMetadataBuilder,
                GDNAttentionMetadataBuilder,
                ShortConvAttentionMetadataBuilder,
                PleShortConvAttentionMetadataBuilder,
            ),
        ):
            return {}
        return {
            "num_accepted_tokens": None
            if self.num_accepted_tokens is None
            else self.num_accepted_tokens[:num_reqs],
            "num_decode_draft_tokens_cpu": None
            if self.num_decode_draft_tokens_cpu is None
            else self.num_decode_draft_tokens_cpu[:num_reqs],
        }


class MambaHybridModelState(DefaultModelState):
    """Model state for hybrid attention + Mamba / linear-attention models."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self.cache_config = vllm_config.cache_config
        self.num_accepted_tokens_gpu = torch.ones(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        # Handed to the Mamba builders instead of the per-step gather when every
        # one of them discards it (_mamba_builders_drop_num_accepted).
        self._unused_num_accepted: torch.Tensor | None = None
        # Pre-copy "align" prefix-cache state (V2). The migration of each
        # request's mamba state across block boundaries runs as a fused GPU
        # kernel reusing the postprocess copy machinery, so the per-step src
        # columns and the running state_idx are kept GPU-resident.
        self._align_mode = self.cache_config.mamba_cache_mode == "align"
        # Set by the runner via `set_block_tables`; the align post-process needs
        # the persistent per-req-slot tables, not the per-step gather.
        self._runner_block_tables: Any = None
        self.recoverssm = (
            RecoverSSMState() if self.cache_config.use_recoverssm else None
        )
        # (idx_mapping, num_sampled, num_computed_tokens, recorded step) of a
        # postprocess tail held until flush_deferred_postprocess().
        self._deferred_tail: tuple[Any, ...] | None = None
        # Set when a deferred tail could not be enqueued: the GDN/conv state is
        # not the committed state, so the runner must not run another step.
        self._deferred_tail_error: str | None = None
        self._step_has_sampled_outputs = False
        if self._align_mode:
            self._mamba_state_idx_gpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )
            self._mamba_src_col_gpu = torch.full(
                (self.max_num_reqs,), -1, dtype=torch.int32, device=self.device
            )
            self._mamba_src_off_gpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )
            self._mamba_ctx: MambaSpecDecodeGPUContext | None = None
            self._mamba_group_ids: list[int] = []
            self._mamba_spec: MambaSpec | None = None
            self._mamba_state_copy_funcs: MambaStateCopyFuncsByType | None = None

    def set_block_tables(self, block_tables: Any) -> None:
        self._runner_block_tables = block_tables

    @staticmethod
    def _mamba_builders_drop_num_accepted(
        attn_groups: list[list[AttentionGroup]],
    ) -> bool:
        """True when every Mamba builder runs the decode fast path
        (VLLM_GDN_DECODE_FASTPATH) and replaces num_accepted_tokens in all the
        metadata it returns (RecoverSSM GDN: ones; see
        GDNAttentionMetadataBuilder._overrides_num_accepted_tokens). The
        gathered counts are then never read, so the per-step gather is skipped."""
        found = False
        for groups in attn_groups:
            for group in groups:
                if not isinstance(group.kv_cache_spec, MambaSpec):
                    continue
                builder = group.get_metadata_builder(0)
                if not (
                    getattr(builder, "decode_fastpath", False)
                    and getattr(builder, "_overrides_num_accepted_tokens", False)
                ):
                    return False
                found = True
        return found

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        # Must reset the speculative acceptance count in this idx which could be stale.
        self.num_accepted_tokens_gpu[req_index].fill_(1)
        if self._align_mode:
            # Seed the running state block from the resumed/prefilled position.
            # The column is in mamba blocks. cache_config.block_size is the
            # min over prefix-cacheable groups, so a drafter group with its own
            # smaller block (e.g. DFlash) pulls it below mamba_block_size.
            mamba_block_size = self.cache_config.mamba_block_size
            assert mamba_block_size is not None
            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // mamba_block_size
            )

    def _get_mamba_group_info(
        self, kv_cache_config: KVCacheConfig
    ) -> tuple[list[int], MambaSpec]:
        if self._mamba_spec is None:
            mamba_groups = get_mamba_groups(kv_cache_config)
            mamba_spec = next(iter(mamba_groups))
            assert all(
                spec.block_size == mamba_spec.block_size
                and spec.num_speculative_blocks == mamba_spec.num_speculative_blocks
                and spec.mamba_cache_mode == mamba_spec.mamba_cache_mode
                for spec in mamba_groups
            ), "all mamba groups must share cache scheduling parameters"
            self._mamba_group_ids = get_mamba_group_ids(mamba_groups)
            self._mamba_spec = mamba_spec
        return self._mamba_group_ids, self._mamba_spec

    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig, block_tables: BlockTables
    ) -> None:
        if not self._align_mode:
            return
        mamba_group_ids, _ = self._get_mamba_group_info(kv_cache_config)
        if self._mamba_state_copy_funcs is None:
            mamba_groups = get_mamba_groups(kv_cache_config)
            mamba_types = {spec.mamba_type for spec in mamba_groups}
            copy_funcs = self.model.get_mamba_state_copy_funcs(mamba_types)
            validate_mamba_state_copy_funcs(mamba_groups, copy_funcs)
            self._mamba_state_copy_funcs = copy_funcs
        # Both SD and DS conv layouts support a >0 spec-decode shift: the
        # fused pre-copy kernel (``_copy_mamba_state_block``) applies the
        # ``token_bias = num_accepted - 1`` window shift per conv layout
        # (SD: contiguous slice; DS: per-dim-row strided slice), matching
        # the V1 ``get_conv_copy_spec`` semantics.
        self._mamba_ctx = MambaSpecDecodeGPUContext.create(
            max_num_reqs=self.max_num_reqs,
            kv_cache_config=kv_cache_config,
            copy_funcs=self._mamba_state_copy_funcs,
            device=self.device,
            make_buffer=lambda n, dtype: CpuGpuBuffer(
                n, dtype=dtype, device=self.device
            ),
        )
        self._mamba_ctx.initialize_from_forward_context(
            kv_cache_config,
            self.vllm_config.compilation_config.static_forward_context,
            self._mamba_state_copy_funcs,
            [block_tables.input_block_tables[gid] for gid in mamba_group_ids],
            persistent_block_tables=[
                block_tables.block_tables[gid].gpu for gid in mamba_group_ids
            ],
        )

    def preprocess_state(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        """Migrate each request's mamba state across block boundaries before the
        forward (V1 align semantics, done on GPU). Runs on real batches only
        (dummy DP/profiling runs skip preprocess_state), and before
        ``prepare_attn`` gathers ``num_accepted_tokens``, so the boundary reset
        is visible to the forward kernels.
        """
        if not self._align_mode:
            return
        num_reqs = input_batch.num_reqs
        if num_reqs == 0:
            return
        _, mamba_spec = self._get_mamba_group_info(kv_cache_config)
        ctx = self._mamba_ctx
        assert ctx is not None

        # The state-advance + pre-copy kernels run every step; they fast-exit per
        # request when src_col < 0 or src_col == dst_col, so no copy happens on
        # steps that don't cross a block boundary. (Skipping the launch entirely
        # would need a V1-style async-D2H of the actual num_computed, since
        # num_computed_tokens_np is an optimistic mirror under async scheduling;
        # the launch cost is ~0.3% of TPOT, so the GPU fast-exit suffices.)
        block = 256
        grid = (triton.cdiv(num_reqs, block),)
        preprocess_mamba_align_fused_kernel[grid](
            input_batch.idx_mapping,
            self._mamba_state_idx_gpu,
            num_computed_tokens,
            input_batch.query_start_loc,
            self.num_accepted_tokens_gpu,
            self._mamba_src_col_gpu,
            self._mamba_src_off_gpu,
            num_reqs,
            BLOCK_SIZE=block,
            MAMBA_BLOCK_SIZE=mamba_spec.block_size,
        )
        ctx.run_fused_precopy(
            num_reqs,
            self._mamba_state_idx_gpu,
            self._mamba_src_col_gpu,
            self._mamba_src_off_gpu,
            input_batch.idx_mapping,
        )

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        assert ubatch_idx == 0, "DBO is not supported"
        assert model_specific_attn_metadata is None
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        # Prefer the promised bound: a capture dummy's measured max is its even
        # split, not the length the graph must replay.
        max_query_len = input_batch.max_query_len
        if max_query_len is None:
            max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        if for_capture:
            # Capture with worst-case max_seq_len so the graph is valid at any replay.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()

        is_prefilling_np = input_batch.is_prefilling_np
        if input_batch.prefill_runs_as_decode_np is not None:
            # A prompt tail the scheduler padded with placeholder drafts must run
            # as a spec-decode row: the prefill kernels can't roll them back.
            is_prefilling_np = is_prefilling_np & ~input_batch.prefill_runs_as_decode_np
        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool, device="cpu")
        is_prefilling[: input_batch.num_reqs] = torch.from_numpy(is_prefilling_np)
        # During CUDAGraph capture, num_decode_draft_tokens_cpu and num_accepted_tokens
        # are created by attn_metadata_builder.build_for_cudagraph_capture, so we only
        # compute them during actual (non-capture) forward execution.
        num_accepted_tokens = None
        num_decode_draft_tokens_cpu = None
        if not for_capture and self.vllm_config.num_speculative_tokens > 0:
            if self._mamba_builders_drop_num_accepted(attn_groups):
                # Never read (see the helper); a persistent ones view keeps the
                # builders' contract (an int32 [num_reqs] tensor) with no ops.
                if (
                    self._unused_num_accepted is None
                    or self._unused_num_accepted.numel() < num_reqs
                ):
                    self._unused_num_accepted = torch.ones(
                        max(num_reqs, self.max_num_reqs),
                        dtype=torch.int32,
                        device=self.device,
                    )
                num_accepted_tokens = self._unused_num_accepted[:num_reqs]
            else:
                num_accepted_tokens = self.num_accepted_tokens_gpu.new_ones(num_reqs)
                num_accepted_tokens[: input_batch.num_reqs] = (
                    self.num_accepted_tokens_gpu[input_batch.idx_mapping]
                )

            # GDN uses >= 0 to select spec-decode rows, so non-decode rows
            # need the -1 sentinel. Decode rows without drafts stay spec rows:
            # only that path applies the previous step's accepted offset.
            num_decode_draft_tokens_np = np.full(num_reqs, -1, dtype=np.int32)
            num_draft_tokens_per_req = input_batch.num_draft_tokens_per_req
            if num_draft_tokens_per_req is None:
                num_draft_tokens_per_req = np.zeros(input_batch.num_reqs, np.int32)
            # Test request state, not num_scheduled_tokens == draft_count+1:
            # adaptive rewrites num_scheduled_tokens to an even split, so that
            # equality rarely holds and would demote every verify row to decode.
            is_decode = (
                ~is_prefilling_np
                & (input_batch.num_scheduled_tokens > 0)
                & (
                    input_batch.num_scheduled_tokens
                    <= self.vllm_config.num_speculative_tokens + 1
                )
            )
            num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(
                is_decode, num_draft_tokens_per_req, -1
            )
            num_decode_draft_tokens_cpu = torch.from_numpy(num_decode_draft_tokens_np)

        if self._align_mode:
            mamba_group_ids, _ = self._get_mamba_group_info(kv_cache_config)
            aligned_index_builders = []
            for group_idx, group_id in enumerate(mamba_group_ids):
                for group in attn_groups[group_id]:
                    builder = group.get_metadata_builder(0)
                    if hasattr(builder, "mamba_aligned_state_indices"):
                        aligned_index_builders.append((group_idx, builder))
            if aligned_index_builders:
                assert self._mamba_ctx is not None
                all_group_indices = self._mamba_ctx.compute_aligned_state_indices(
                    input_batch.seq_lens, num_reqs
                )
                for group_idx, builder in aligned_index_builders:
                    builder.mamba_aligned_state_indices = all_group_indices[group_idx]

        mamba_attn_metadata = MambaHybridAttnMetadata(
            is_prefilling=is_prefilling,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        )
        if envs.MERLIN_FLASHINFER_WORKLOAD_TILE:
            mamba_attn_metadata.workload_prefill_flags = scheduled_prefill_flags(
                input_batch, num_reqs
            )
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            positions=input_batch.positions,
            model_specific_attn_metadata=mamba_attn_metadata,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
            seq_lens_cpu_exact=None if for_capture else input_batch.seq_lens_cpu_exact,
        )
        if self.recoverssm is not None:
            self._step_has_sampled_outputs = (
                not for_capture and compute_need_sampled_mask(input_batch) is not None
            )
            self.recoverssm.record_step(
                attn_metadata,
                attn_groups,
                for_capture=for_capture,
                state_indices=(self._mamba_state_idx_gpu if self._align_mode else None),
                num_accepted_tokens=self.num_accepted_tokens_gpu,
            )
        return attn_metadata

    def detach_deferred_state(self) -> Any:
        # Non-last PP rank: the RecoverSSM commit of this step has to wait for
        # the accepted counts of the PP broadcast; snapshot its inputs now.
        if self.recoverssm is None:
            return None
        return self.recoverssm.detach_step(
            has_sampled_outputs=self._step_has_sampled_outputs
        )

    def restore_deferred_state(self, state: Any) -> None:
        if state is None:
            return
        if self.recoverssm is None:
            raise RuntimeError("deferred RecoverSSM state without RecoverSSM")
        self.recoverssm.restore_step(state)

    supports_deferred_postprocess = True

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
        *,
        defer_tail: bool = False,
    ) -> None:
        self.check_deferred_postprocess()
        # Chunked prefill does not sample a token, so num_sampled can be 0.
        # Mamba treats num_accepted_tokens=1 as the neutral non-spec value.
        num_reqs = idx_mapping.shape[0]
        if num_reqs:
            if not isinstance(num_sampled, int):
                # idx_mapping may contain -1 sentinels (filtered rows) under PP; the
                # kernel skips them rather than scattering with a host-side gather.
                _scatter_num_accepted_kernel[(num_reqs,)](
                    idx_mapping,
                    num_sampled,
                    self.num_accepted_tokens_gpu,
                )
            else:
                # Fill with single value.
                _fill_num_accepted_kernel[(num_reqs,)](
                    idx_mapping,
                    self.num_accepted_tokens_gpu,
                    max(num_sampled, 1),
                )

        if defer_tail:
            self._deferred_tail = (
                idx_mapping,
                num_sampled,
                num_computed_tokens,
                None if self.recoverssm is None else self.recoverssm._step,
            )
            return
        self._postprocess_tail(idx_mapping, num_sampled, num_computed_tokens)

    def check_deferred_postprocess(self) -> None:
        """Raise unless the previous step's deferred tail was enqueued. The
        runner calls this before it mutates request state or builds metadata
        for the next step: a pending tail (its step raised between postprocess
        and flush) or a failed one means the next forward would read state the
        commit has not written. Not recoverable: the runner stays poisoned."""
        if self._deferred_tail_error is not None:
            raise RuntimeError(
                "RecoverSSM: a deferred postprocess tail failed "
                f"({self._deferred_tail_error}); the GDN state is not committed"
            )
        if self._deferred_tail is not None:
            raise RuntimeError(
                "RecoverSSM: a deferred postprocess tail was never flushed"
            )

    def flush_deferred_postprocess(self) -> None:
        if self._deferred_tail_error is not None:
            self.check_deferred_postprocess()  # raises: already failed once
        tail = self._deferred_tail
        if tail is None:
            return
        idx_mapping, num_sampled, num_computed_tokens, step = tail
        try:
            if self.recoverssm is not None and self.recoverssm._step is not step:
                raise RuntimeError(
                    "RecoverSSM: the recorded step changed between postprocess "
                    "and the deferred commit"
                )
            self._postprocess_tail(idx_mapping, num_sampled, num_computed_tokens)
        except BaseException as e:
            # Keep the marker: the state is mid-transition.
            self._deferred_tail_error = f"{type(e).__name__}: {e}"
            raise
        self._deferred_tail = None

    def _postprocess_tail(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None,
    ) -> None:
        num_reqs = idx_mapping.shape[0]
        if self.recoverssm is not None:
            self.recoverssm.commit_step(
                num_sampled,
                idx_mapping,
                state_indices=(self._mamba_state_idx_gpu if self._align_mode else None),
                num_accepted_tokens=self.num_accepted_tokens_gpu,
            )

        if not num_reqs:
            return

        # Align: save the running state to the block-aligned position when
        # spec-decode acceptance leaves the sequence non-block-aligned (mirrors
        # the V1 align postprocess). num_computed_tokens already holds the
        # post-step advanced count.
        if (
            self._align_mode
            and num_computed_tokens is not None
            and self._mamba_ctx is not None
        ):
            self._mamba_ctx.run_fused_postprocess_align(
                num_reqs,
                self.num_accepted_tokens_gpu,
                self._mamba_state_idx_gpu,
                num_computed_tokens,
                idx_mapping,
            )


@triton.jit
def _scatter_num_accepted_kernel(
    idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx (-1 to skip)
    num_sampled_ptr,  # [num_reqs]
    num_accepted_ptr,  # [max_num_reqs]
):
    row = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    if req_state_idx < 0:
        return
    num_sampled = tl.load(num_sampled_ptr + row)
    tl.store(num_accepted_ptr + req_state_idx, tl.maximum(num_sampled, 1))


@triton.jit
def _fill_num_accepted_kernel(
    idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx (-1 to skip)
    num_accepted_ptr,  # [max_num_reqs]
    num_sampled,
):
    row = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    if req_state_idx < 0:
        return
    tl.store(num_accepted_ptr + req_state_idx, num_sampled)

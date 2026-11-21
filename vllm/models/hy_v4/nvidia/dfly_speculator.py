# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""The DFly2 proposer for Hy4 Flash on the V2 GPU model runner.

Adapted from #68296 ``vllm/models/hy_v4/nvidia/dfly_speculator.py:1-587``.

One query block per request per step: the bonus token fused with the
target's final hidden state at slot 0, then ``num_speculative_tokens`` mask
tokens. The draft layers attend to the target's sparse-layer KV (aliased
storage) through the indices built from the target's top-k snapshots, and
each draft token comes from the hidden state of its query slot corrected
with the previous draft token's embedding.

Differences from upstream, for this tree's runner (Merlin 0.33):
- ``DraftModelSpeculator`` here has no ``max_model_len_gpu``, no draft-logits
  row mapping and a 5-argument ``set_attn``; sampling goes through the base
  ``sample_draft`` (greedy argmax or Gumbel with the draft logits cache).
- Query-block slots come from ``BlockTables.compute_slot_mappings``, which is
  DCP-aware (see ``dfly_inputs``); upstream's own slot kernel assumes DCP 1.
- Eager by default. ``MERLIN_HY4_DFLY_GRAPHS=1``
  captures the whole query block (embed, fuse, the draft layers, every
  correction + sample step) as FULL graphs per uniform decode size, through
  the DFlash graph manager; the target already runs the Hy4 sparse backend in
  FULL graphs (UNIFORM_BATCH). Breakable-PIECEWISE draft graphs are not ported.
- The draft is loaded with prefix "" so its module names equal the
  checkpoint's (``model.dfly_layers.N``).
- Under pipeline parallelism (draft and target layers 25-41 on the last
  stage), the input embedding is loaded for the last stage from the target
  checkpoint (``spec_decode/dflash/utils._maybe_load_target_embed_on_pp``).
"""

from copy import copy
from typing import Any, NamedTuple

import torch
from torch import nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_pp_group
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.utils import get_draft_load_config
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import DPSyncState, dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    get_target_lm_head,
    maybe_share_target_embed,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.utils import get_pp_safe_draft_load_config
from vllm.v1.worker.utils import AttentionGroup

from .dfly import HYV4DFlyForCausalLM, run_dfly_query
from .dfly_inputs import prepare_dfly_inputs
from .dfly_target_reuse import HYV4DFlyTargetReuse

logger = init_logger(__name__)


def dfly_graphs_requested(environ=None) -> bool:
    """``MERLIN_HY4_DFLY_GRAPHS``: 1 = FULL draft graphs, unset/blank/0 = eager."""
    import os

    raw = ((os.environ if environ is None else environ).get("MERLIN_HY4_DFLY_GRAPHS", "") or "").strip()
    if raw in ("", "0"):
        return False
    if raw == "1":
        return True
    raise ValueError(f"MERLIN_HY4_DFLY_GRAPHS must be 0 or 1 (got {raw!r})")


def dfly_graph_mode(requested: bool, decode_mode: CUDAGraphMode, attn_support) -> CUDAGraphMode:
    """FULL_DECODE_ONLY when asked for, the target decodes in FULL graphs and
    every draft attention backend supports uniform-batch graphs; else NONE."""
    if not requested or decode_mode != CUDAGraphMode.FULL:
        return CUDAGraphMode.NONE
    if attn_support.value < AttentionCGSupport.UNIFORM_BATCH.value:
        return CUDAGraphMode.NONE
    return CUDAGraphMode.FULL_DECODE_ONLY


class _DraftSlotTables:
    """the target's BlockTables with one change for
    draft-graph capture: the dummy slot mappings are the draft's own persistent
    buffer (``_slot_mappings``, which ``_prepare_query`` fills before every
    replay: PAD for inactive and padded rows, DCP-owner slots otherwise), not
    the target's ``slot_mappings``. Captured against the target's buffer, every
    replay wrote the draft's query-block KV through the target's last-step slots
    into the aliased target cache (otherwise T=0 outputs differ and draft acceptance collapses)."""

    def __init__(self, block_tables: BlockTables, slots: torch.Tensor) -> None:
        self._block_tables, self._slots = block_tables, slots

    def get_dummy_slot_mappings(self, num_tokens: int) -> torch.Tensor:
        self._slots.fill_(PAD_SLOT_ID)
        return self._slots[:, :num_tokens]

    def __getattr__(self, name: str):
        return getattr(self._block_tables, name)


class DFlyProposeInputs(NamedTuple):
    """The target step's results a DFly query block is built from."""

    query_start_loc: torch.Tensor
    positions: torch.Tensor
    idx_mapping: torch.Tensor
    num_sampled: torch.Tensor
    num_rejected: torch.Tensor
    last_sampled: torch.Tensor
    last_hidden_states: torch.Tensor
    temperature: torch.Tensor
    seeds: torch.Tensor


class HYV4DFlySpeculator(DraftModelSpeculator):
    supports_step_keyed_rng = False

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        cache_dtype = vllm_config.cache_config.cache_dtype
        if cache_dtype not in ("auto", "bfloat16", "fp8_ds_mla"):
            raise ValueError("DFly requires BF16 or fp8_ds_mla MLA KV cache")
        spec = vllm_config.speculative_config
        assert spec is not None
        draft_cache_dtype = spec.kv_cache_dtype or cache_dtype
        normalize = lambda d: "bfloat16" if d == "auto" else d  # noqa: E731
        if normalize(draft_cache_dtype) != normalize(cache_dtype):
            raise ValueError(
                "DFly draft and target must use the same MLA KV cache dtype; "
                "omit speculative_config.kv_cache_dtype to inherit the target"
            )
        if vllm_config.kv_transfer_config is not None:
            raise ValueError("DFly does not support PD disaggregation")
        if (
            vllm_config.lora_config is not None
            or vllm_config.parallel_config.use_ubatching
        ):
            raise ValueError("DFly does not support LoRA or microbatching")
        super().__init__(vllm_config, device)
        # The draft runs at the target's DCP size: its layers read the target's
        # DCP-sharded cache.
        assert self.dcp_size == vllm_config.parallel_config.decode_context_parallel_size

        config = copy(vllm_config)
        config.model_config = copy(self.draft_model_config)
        config.parallel_config = copy(vllm_config.parallel_config)
        config.parallel_config.enable_expert_parallel = False
        config.attention_config = copy(vllm_config.attention_config)
        # Every draft row is a sparse MQA query over the target's latent cache.
        config.attention_config.sparse_mla_force_mqa = True
        config.attention_config.use_non_causal = False
        config.cache_config = vllm_config.cache_config
        config.quant_config = None
        self._draft_vllm_config = config

        self.num_query_per_req = self.num_speculative_steps + 1
        self.target_hidden_states = torch.zeros(
            self.max_num_reqs, self.hidden_size, dtype=self.dtype, device=device
        )
        self.active_requests = torch.zeros(
            self.max_num_reqs, dtype=torch.bool, device=device
        )
        # The target batch row of each request's last accepted token, and the
        # position of its bonus (anchor) query.
        self.source_rows = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self.anchors = torch.zeros(self.max_num_reqs, dtype=torch.int64, device=device)
        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )
        self.reuse: HYV4DFlyTargetReuse | None = None
        self._storage_validated = False
        self._slot_mappings: torch.Tensor | None = None

    @property
    def attn_vllm_config(self) -> VllmConfig:
        return self._draft_vllm_config

    # ------------------------------------------------------------------ load
    def load_draft_model(
        self, target_model: nn.Module, target_attn_layer_names: set[str]
    ) -> nn.Module:
        from vllm.v1.worker.gpu.spec_decode.dflash.utils import (
            _maybe_load_target_embed_on_pp,
        )

        config = self._draft_vllm_config
        load_config = get_pp_safe_draft_load_config(
            get_draft_load_config(self.vllm_config)
        )
        config.quant_config = VllmConfig.get_quantization_config(
            self.draft_model_config, load_config
        )
        draft = get_model(
            vllm_config=config,
            model_config=self.draft_model_config,
            load_config=load_config,
            prefix="",
        )
        if not isinstance(draft, HYV4DFlyForCausalLM):
            raise TypeError("DFly speculator requires HYV4DFlyForCausalLM")
        language_model = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        target_inner = getattr(language_model, "model", language_model)
        if draft.model.embed_tokens is None and get_pp_group().world_size > 1:
            # The last stage holds no target table: give the loader one to fill.
            with set_current_vllm_config(config), torch.device(self.device):
                draft.model.embed_tokens = VocabParallelEmbedding(
                    draft.config.vocab_size,
                    draft.config.hidden_size,
                    params_dtype=self.dtype,
                )
        if not _maybe_load_target_embed_on_pp(
            draft.model, self.vllm_config, target_inner
        ):
            maybe_share_target_embed(draft, draft.model, target_inner)
        head = get_target_lm_head(target_model, language_model)
        if head is None or draft.model.embed_tokens is None:
            raise ValueError("DFly requires the target embedding and LM head")
        if (
            head.weight.dtype != self.dtype
            or draft.model.embed_tokens.weight.dtype != self.dtype
        ):
            raise ValueError("DFly requires BF16 shared embedding and LM head weights")
        draft.lm_head = head
        if hasattr(draft.logits_processor, "head_dtype"):
            draft.logits_processor.head_dtype = self.vllm_config.model_config.head_dtype
        self.reuse = HYV4DFlyTargetReuse(target_model, draft, self.vllm_config, config)
        logger.info(
            "HYV4 DFly: %d layers, N=%d, TP=%d, DCP=%d, target index/KV=%s/%s, "
            "KV cache %s (aliased, not replicated), draft quantization=%s %s, "
            "draft sampling=%s, %s",
            len(draft.model.dfly_layers),
            self.num_speculative_steps,
            config.parallel_config.tensor_parallel_size,
            self.dcp_size,
            draft.config.target_index_layer,
            draft.config.target_kv_layers,
            config.cache_config.cache_dtype,
            self.draft_model_config.quantization or "none",
            sorted({b for b in draft.quant_bits.values() if b is not None}) or "",
            self.speculative_config.draft_sample_method,
            # The requested mode (the capture logs when it runs)
            "draft graphs requested" if dfly_graphs_requested() else "eager",
        )
        return draft

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        assert self.reuse is not None
        self.reuse.validate_groups(kv_cache_config)
        self._storage_validated = False
        self._slot_mappings = torch.full(
            (len(kv_cache_config.kv_cache_groups), self.max_num_tokens),
            PAD_SLOT_ID,
            dtype=torch.int64,
            device=self.device,
        )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:

        self.cudagraph_manager = None
        mode = dfly_graph_mode(
            dfly_graphs_requested(),
            cudagraph_mode.decode_mode(),
            self.attn_cg_support.min_cg_support,
        )
        if mode == CUDAGraphMode.NONE:
            if cudagraph_mode != CUDAGraphMode.NONE:
                logger.info_once(
                    "DFly draft runs eagerly (MERLIN_HY4_DFLY_GRAPHS=1 for FULL draft graphs)"
                )
            return
        from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import (
            DFlashCudaGraphManager,
        )

        self.cudagraph_manager = DFlashCudaGraphManager(
            self.vllm_config,
            self.device,
            mode,
            decode_query_len=self.num_query_per_req,
        )
        # Plain info: the draft lives on the last PP stage, which never holds the
        # (local) first rank that info_once and the capture progress bar need.
        logger.info("DFly draft: FULL CUDA graphs per uniform decode size (rank-local)")

    def capture(self) -> None:

        if getattr(self, "cudagraph_manager", None) is None:
            return
        # Every row inactive: the draft indices built from the dummy anchors stay
        # in range (-1 padded) for every captured size, and no row is a request.
        self._prepare_query(
            self._dummy_propose_inputs(), 0, self.max_num_reqs, with_slots=False
        )
        self.draft_tokens.fill_(-1)
        assert self._slot_mappings is not None
        self.cudagraph_manager.capture(
            self._graph_forward,
            self.input_buffers,
            _DraftSlotTables(self.block_tables, self._slot_mappings),
            self.attn_groups,
            self.kv_cache_config,
            self.max_model_len,
            causal=True,
            # Captured ahead of the forward: nothing, the eager propose stages
            # every input before a replay.
            prepare_forward=lambda num_reqs, attn_metadata: None,
            progress_bar_desc="Capturing DFly draft CUDA graphs",
        )
        logger.info("DFly draft: captured FULL graphs")

    def _graph_forward(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        """The captured query block: inputs and outputs are the persistent
        buffers ``_prepare_query`` fills (input ids, positions, target hidden,
        active rows, draft indices) and ``draft_tokens``."""
        with set_forward_context(
            attn_metadata,
            self._draft_vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
            slot_mapping=slot_mappings,
            is_padding=self.input_buffers.is_padding[:num_tokens],
        ):
            run_dfly_query(
                self.model,
                self.input_buffers.input_ids[:num_tokens],
                self.input_buffers.positions[:num_tokens],
                self.target_hidden_states[:num_reqs],
                self.active_requests[:num_reqs],
                self.draft_tokens,
                self._sample_step,
            )

    # --------------------------------------------------------------- propose
    def _dummy_propose_inputs(self) -> DFlyProposeInputs:
        """Placeholders for dummy runs; never read with ``num_reqs=0``."""
        return DFlyProposeInputs(
            query_start_loc=self.input_buffers.query_start_loc,
            positions=self.input_buffers.positions,
            idx_mapping=self.idx_mapping,
            num_sampled=self.source_rows,
            num_rejected=self.source_rows,
            last_sampled=self.draft_tokens[:, 0],
            last_hidden_states=self.target_hidden_states,
            temperature=self.temperature,
            seeds=self.seeds,
        )

    def _prepare_query(
        self, inputs: DFlyProposeInputs, num_reqs: int, batch: int, with_slots: bool
    ) -> None:
        assert self.reuse is not None
        buffers = self.input_buffers
        prepare_dfly_inputs(
            input_ids=buffers.input_ids,
            positions=buffers.positions,
            is_padding=buffers.is_padding,
            query_start_loc=buffers.query_start_loc,
            seq_lens=buffers.seq_lens,
            active=self.active_requests,
            source_rows=self.source_rows,
            anchors=self.anchors,
            target_hidden_out=self.target_hidden_states,
            draft_idx_mapping=self.idx_mapping,
            out_temperature=self.temperature,
            out_seeds=self.seeds,
            target_query_start_loc=inputs.query_start_loc,
            target_positions=inputs.positions,
            idx_mapping=inputs.idx_mapping,
            num_sampled=inputs.num_sampled,
            num_rejected=inputs.num_rejected,
            last_sampled=inputs.last_sampled,
            target_hidden=inputs.last_hidden_states,
            temperature=inputs.temperature,
            seeds=inputs.seeds,
            max_model_len=self.max_model_len,
            mask_token_id=self.model.config.mask_token_id,
            query_width=self.num_query_per_req,
            num_reqs=num_reqs,
            padded_batch=batch,
        )
        self.idx_mapping[batch:].fill_(-1)
        count = batch * self.num_query_per_req
        if with_slots:
            assert self._slot_mappings is not None
            # The target's slot kernel: an inactive row (idx -1) gets PAD, and
            # under DCP only the owner rank of a position gets a local slot.
            self.block_tables.compute_slot_mappings(
                self.idx_mapping[:batch],
                buffers.query_start_loc[: batch + 1],
                buffers.positions[:count],
                count,
                out=self._slot_mappings,
            )
        self.reuse.update_indices(
            self.active_requests,
            self.source_rows,
            self.anchors,
            batch,
            inputs.query_start_loc,
        )

    def _sample_step(self, step: int, hidden: torch.Tensor) -> torch.Tensor:
        """Draft tokens at query offset ``step + 1``, keyed by the prior position."""
        batch = hidden.shape[0]
        return self.sample_draft(
            hidden,
            self.anchors[:batch] + step,
            self.idx_mapping[:batch],
            self.temperature,
            self.seeds,
            self._step_cols[step],
            self.draft_logits,
            spec_step_idx=step,
            num_steps=self.num_speculative_steps,
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        batch = input_batch.num_reqs
        width = self.num_query_per_req
        count = batch * width
        result = self.draft_tokens[:batch]
        if batch == 0 or num_speculative_tokens == 0:
            return result.fill_(-1)
        if count > self.max_num_tokens:
            raise ValueError("DFly query block exceeds max_num_batched_tokens")
        assert self.reuse is not None
        with_attn = not (dummy_run and skip_attn_for_dummy_run)
        if with_attn and not dummy_run and not self._storage_validated:
            self.reuse.validate_storage()
            self._storage_validated = True
        if dummy_run:
            inputs, num_reqs = self._dummy_propose_inputs(), 0
        else:
            inputs = DFlyProposeInputs(
                query_start_loc=input_batch.query_start_loc,
                positions=input_batch.positions,
                idx_mapping=input_batch.idx_mapping,
                num_sampled=num_sampled,
                num_rejected=num_rejected,
                last_sampled=last_sampled,
                last_hidden_states=last_hidden_states,
                temperature=temperature,
                seeds=seeds,
            )
            num_reqs = batch
        # A FULL replay when a draft graph covers the
        # batch: stage the inputs at the graph's padded size, rebuild the
        # metadata into the builders' persistent buffers, replay.
        manager = getattr(self, "cudagraph_manager", None)
        if manager is not None and with_attn and not dummy_run and not is_profile:
            desc, _ = dispatch_cg_and_sync_dp(
                manager,
                batch,
                count,
                uniform_token_count=width,
                dp_size=self.dp_size,
                dp_rank=self.dp_rank,
            )
            if desc.cg_mode == CUDAGraphMode.FULL:
                return self._propose_full(input_batch, inputs, batch, desc)
        self._prepare_query(inputs, num_reqs, batch, with_slots=with_attn)

        metadata = None
        slots_by_layer = None
        descriptor = BatchExecutionDescriptor(CUDAGraphMode.NONE, count, batch)
        if with_attn:
            upper = torch.zeros(batch, dtype=torch.int32, device="cpu")
            upper.copy_(input_batch.seq_lens_cpu_upper_bound[:batch])
            self.draft_max_seq_len = min(int(upper.max()) + width, self.max_model_len)
            assert self._slot_mappings is not None
            metadata = self._build_attn_metadata(
                num_reqs=batch,
                batch_desc=descriptor,
                query_start_loc_np=self.arange_np[: batch + 1] * width,
                seq_lens_cpu_upper_bound=upper,
                step=width,
                causal=True,
                slot_mappings=self._slot_mappings[:, :count],
            )
            slots_by_layer = build_slot_mappings_by_layer(
                self._slot_mappings[:, :count], self.kv_cache_config
            )
        self._prepare_eplb_forward(count)
        with set_forward_context(
            metadata,
            self._draft_vllm_config,
            num_tokens=count,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=BatchDescriptor(num_tokens=count),
            slot_mapping=slots_by_layer,
            is_padding=self.input_buffers.is_padding[:count],
        ):
            run_dfly_query(
                self.model,
                self.input_buffers.input_ids[:count],
                self.input_buffers.positions[:count],
                self.target_hidden_states[:batch],
                self.active_requests[:batch],
                self.draft_tokens,
                self._sample_step,
            )
        return result


    def _propose_full(
        self,
        input_batch: InputBatch,
        inputs: DFlyProposeInputs,
        batch: int,
        desc: BatchExecutionDescriptor,
    ) -> torch.Tensor:
        width = self.num_query_per_req
        graph_batch = desc.num_reqs or batch
        assert graph_batch >= batch and desc.num_tokens == graph_batch * width
        # Rows batch..graph_batch are inactive padding (-1 drafts, -1 indices).
        self._prepare_query(inputs, batch, graph_batch, with_slots=True)
        upper = torch.zeros(batch, dtype=torch.int32, device="cpu")
        upper.copy_(input_batch.seq_lens_cpu_upper_bound[:batch])
        self.draft_max_seq_len = min(int(upper.max()) + width, self.max_model_len)
        assert self._slot_mappings is not None
        # The DCP-local lengths of the padded rows too: the graph reads all
        # graph_batch rows, and the base builder fills only the live ones.
        dcp_local = None
        if self.dcp_size > 1:
            from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens

            dcp_local = prepare_dcp_local_seq_lens(
                self.input_buffers.dcp_local_seq_lens,
                self.input_buffers.seq_lens,
                batch,
                self.dcp_size,
                self.block_tables.cp_rank,
                self.block_tables.cp_interleave,
                num_reqs_padded=graph_batch,
            )
        self._build_attn_metadata(
            num_reqs=batch,
            batch_desc=desc,
            query_start_loc_np=self.arange_np[: batch + 1] * width,
            seq_lens_cpu_upper_bound=upper,
            step=width,
            causal=True,
            dcp_local_seq_lens=dcp_local,
            # the buffer the graphs were captured against (_DraftSlotTables)
            slot_mappings=self._slot_mappings[:, : desc.num_tokens],
        )
        self._prepare_eplb_forward(desc.num_tokens)
        self.cudagraph_manager.run_fullgraph(desc)
        return self.draft_tokens[:batch]

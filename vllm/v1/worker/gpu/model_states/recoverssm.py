# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.recoverssm_metadata import RecoverSSMMetadata
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


@dataclass(frozen=True)
class DeferredRecoverSSMStep:
    """One verify step's RecoverSSM commits, detached on a non-last PP rank.

    The rank verified the window at step T but receives the accepted counts
    only when the PP broadcast of step T is consumed (pp_size steps later).
    The metadata is a snapshot (`snapshot_for_deferred_commit`), so the steps
    in between cannot overwrite its commit inputs. It rides on the PP
    handler's pending entry for step T and is restored right before that
    entry's postprocess."""

    metadata: tuple[RecoverSSMMetadata, ...]


class RecoverSSMState:
    """Coordinates RecoverSSM metadata between attention and postprocessing."""

    def __init__(self) -> None:
        from vllm.v1.worker.gpu.model_states.recoverssm_commit_graph import (
            RecoverSSMCommitGraphs,
        )

        self._step: tuple[RecoverSSMMetadata, ...] | None = None
        self._step_is_deferred = False
        # Commit execution mode (eager | staged | graph) and the captured
        # commit graphs. Default eager = the unchanged launch sequence below.
        self._commit_graphs = RecoverSSMCommitGraphs(self._commit_groups)

    def reset(self) -> None:
        """Drop the recorded step. Its metadata holds the commit contexts, which
        reference the KV cache tensors they were built for (so do the captured
        commit graphs: drop them too)."""
        self._step = None
        self._step_is_deferred = False
        self._commit_graphs.reset()

    def detach_step(
        self, *, has_sampled_outputs: bool = True
    ) -> DeferredRecoverSSMStep | None:
        """Take this step's commits for a deferred (non-last PP rank) commit.

        Returns None when the step has nothing to commit (no spec rows).
        An unsampled step must have no commits; reject it before cloning."""
        step = self._step
        live = (
            () if step is None else tuple(m for m in step if m.has_recoverssm_commit())
        )
        if live and not has_sampled_outputs:
            raise RuntimeError(
                "RecoverSSM: cannot defer a commit without sampled-output rows"
            )
        self._step = None
        self._step_is_deferred = False
        if not live:
            return None
        return DeferredRecoverSSMStep(
            tuple(m.snapshot_for_deferred_commit() for m in live)
        )

    def restore_step(self, deferred: DeferredRecoverSSMStep | None) -> None:
        """Make a detached step the one the next `commit_step` commits."""
        if self._step is not None:
            logger.warning_once(
                "RecoverSSM: dropping a recorded step that was never committed "
                "(expected only after a PIECEWISE capture)"
            )
        self._step = None if deferred is None else deferred.metadata
        self._step_is_deferred = deferred is not None

    def record_step(
        self,
        attn_metadata: dict[str, Any],
        attn_groups: list[list[AttentionGroup]],
        *,
        for_capture: bool,
        state_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> None:
        step: list[RecoverSSMMetadata] = []
        for group_list in attn_groups:
            for group in group_list:
                metadata = attn_metadata[group.layer_names[0]]
                if isinstance(metadata, RecoverSSMMetadata):
                    step.append(metadata)
        if for_capture:
            self._step = None
            if num_accepted_tokens is not None:
                self._commit_graphs.capture_warmup(
                    tuple(step),
                    state_indices=state_indices,
                    num_accepted_tokens=num_accepted_tokens,
                )
            return
        self._step = tuple(step)

    def commit_step(
        self,
        num_sampled: torch.Tensor | int,
        idx_mapping: torch.Tensor,
        *,
        state_indices: torch.Tensor | None,
        num_accepted_tokens: torch.Tensor,
    ) -> None:
        step = self._step
        deferred = self._step_is_deferred
        self._step = None
        self._step_is_deferred = False
        if isinstance(num_sampled, int) or step is None:
            if deferred and step is not None:
                raise RuntimeError(
                    "RecoverSSM: a deferred PP commit was restored without "
                    "sampled counts"
                )
            return

        if deferred:
            num_sampled = torch.where(
                idx_mapping >= 0, num_sampled, torch.zeros_like(num_sampled)
            )

        if not deferred and self._commit_graphs.run(
            step,
            num_sampled,
            idx_mapping,
            state_indices=state_indices,
            num_accepted_tokens=num_accepted_tokens,
        ):
            return
        self._commit_groups(
            step,
            num_sampled,
            idx_mapping,
            state_indices=state_indices,
            num_accepted_tokens=num_accepted_tokens,
        )

    @staticmethod
    def _commit_groups(
        step: tuple[RecoverSSMMetadata, ...],
        num_sampled: torch.Tensor,
        idx_mapping: torch.Tensor,
        *,
        state_indices: torch.Tensor | None,
        num_accepted_tokens: torch.Tensor,
        num_computed_tokens: Sequence[torch.Tensor | None] | None = None,
    ) -> None:
        """Run each backend's commit, then align postprocess.
        GDN fuses the final and boundary state folds in its commit kernel.
        The eager path calls this with the step's own tensors; the
        staged and graph paths with staged copies (``num_computed_tokens``: one
        per group, replacing the metadata's align count tensor)."""
        for i, metadata in enumerate(step):
            if num_computed_tokens is None or num_computed_tokens[i] is None:
                postprocess_meta = metadata.commit_recoverssm_state(num_sampled)
            else:
                postprocess_meta = metadata.commit_recoverssm_state(  # type: ignore[call-arg]
                    num_sampled, num_computed_tokens=num_computed_tokens[i]
                )
            if postprocess_meta is None:
                continue
            assert state_indices is not None
            # RecoverSSM already restored the accepted state. Update its running
            # column and reset the next-step copy bias to the neutral value.
            _postprocess_recoverssm_align_kernel[(postprocess_meta.num_spec_decodes,)](
                idx_mapping,
                num_sampled,
                postprocess_meta.request_indices,
                postprocess_meta.num_computed_tokens,
                state_indices,
                num_accepted_tokens,
                MAMBA_BLOCK_SIZE=postprocess_meta.block_size,
                BLOCK_TABLE_WIDTH=postprocess_meta.block_table.shape[1],
            )


@triton.heuristics(
    {"HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None}
)
@triton.jit
def _postprocess_recoverssm_align_kernel(
    idx_mapping_ptr,
    num_sampled_ptr,
    request_indices_ptr,
    num_computed_ptr,
    state_idx_ptr,
    num_accepted_ptr,
    HAS_REQUEST_INDICES: tl.constexpr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    BLOCK_TABLE_WIDTH: tl.constexpr,
):
    spec_idx = tl.program_id(0)
    batch_idx = spec_idx
    if HAS_REQUEST_INDICES:
        batch_idx = tl.load(request_indices_ptr + spec_idx)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    if req_state_idx < 0:
        return
    num_sampled = tl.load(num_sampled_ptr + batch_idx)
    num_computed = tl.load(num_computed_ptr + batch_idx)
    # Match the commit plan: the running state lives in the block holding the
    # last computed token, so the next step migrates it across a boundary.
    tl.store(
        state_idx_ptr + req_state_idx,
        tl.minimum(
            tl.maximum(num_computed + num_sampled - 1, 0) // MAMBA_BLOCK_SIZE,
            BLOCK_TABLE_WIDTH - 1,
        ),
    )
    tl.store(num_accepted_ptr + req_state_idx, 1)

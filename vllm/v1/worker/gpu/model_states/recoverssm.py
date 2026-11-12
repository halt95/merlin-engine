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


    # ``snapshot_ring_depth`` sizes the PP-deferred commit snapshot ring (None: no
    # ring).
    def __init__(self, snapshot_ring_depth: int | None = None) -> None:
        from vllm.v1.worker.gpu.model_states.recoverssm_commit_graph import (
            RecoverSSMCommitGraphs,
        )
        from vllm.v1.worker.gpu.model_states.recoverssm_snapshot_ring import (
            RecoverSSMSnapshotRing,
        )

        self._step: tuple[RecoverSSMMetadata, ...] | None = None
        # Snapshot ring, deferred-step ownership and
        # commit execution mode state
        self._step_is_deferred = False
        # Persistent snapshot storage for the
        # PP deferral (None: clone() per snapshot), and the restored
        # deferred step, held until its commit is enqueued so its ring slot is
        # not reused before then.
        self._snapshot_ring = (
            None
            if snapshot_ring_depth is None
            else RecoverSSMSnapshotRing(snapshot_ring_depth)
        )
        self._step_owner: DeferredRecoverSSMStep | None = None
        # Commit execution mode (eager | staged | graph) and the captured
        # commit graphs. Default eager = the unchanged launch sequence below.
        self._commit_graphs = RecoverSSMCommitGraphs(self._commit_groups)
        # The eager warm-up keys already launched.
        self._warmed: set[tuple[Any, ...]] = set()

    def _warm_eager_commit(
        self,
        step: tuple[RecoverSSMMetadata, ...],
        *,
        state_indices: torch.Tensor | None,
        num_accepted_tokens: torch.Tensor,
    ) -> None:
        """#
        JIT/module-load the eager commit chain (plan, conv compact, fold, align
        postprocess) on this rank at boot, once per capture geometry, when the
        commit-graph capture did not (eager mode, or a step the planner rejects,
        e.g. PLE). Every PP rank captures its decode graphs, so this also warms
        a non-last rank, whose first real commit is a deferred one (those
        kernels loaded mid-serving, which can hang a PP step).

        A no-op on persistent state, with the commit-graph warm-up's stand-ins:
        0 sampled tokens makes every commit length 0 (no fold, no conv write)
        and idx -1 makes the align postprocess skip its row. It runs during the
        capture phase, before any request exists, on the capture dummy metadata.
        The dtypes are the serving ones (int32 counts, int64 idx_mapping), so the
        warmed kernels are the ones a serving commit launches."""
        if not step:
            return
        ctx0 = getattr(step[0], "recoverssm_context", None)
        n = getattr(step[0], "num_spec_decodes", 0)
        if ctx0 is None or n <= 0:
            return
        key = (
            n,
            tuple(type(m).__name__ for m in step),
            state_indices is None,
        )
        warmed = self.__dict__.setdefault("_warmed", set())
        if key in warmed:
            return
        from vllm.v1.worker.gpu.model_states.recoverssm_commit_graph import (
            IDX_MAPPING_DTYPE,
            NUM_SAMPLED_DTYPE,
        )

        dev = num_accepted_tokens.device
        num_sampled = torch.zeros(n, dtype=NUM_SAMPLED_DTYPE, device=dev)
        idx_mapping = torch.full((n,), -1, dtype=IDX_MAPPING_DTYPE, device=dev)
        self._commit_groups(
            step,
            num_sampled,
            idx_mapping,
            state_indices=state_indices,
            num_accepted_tokens=num_accepted_tokens,
        )
        warmed.add(key)

    def reset(self) -> None:
        """Drop the recorded step. Its metadata holds the commit contexts, which
        reference the KV cache tensors they were built for (so do the captured
        commit graphs: drop them too)."""
        self._step = None
        self._step_is_deferred = False
        self._commit_graphs.reset()

        self._step_owner = None
        if self._snapshot_ring is not None:
            self._snapshot_ring.reset()

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
        # Copy into a free ring slot (same
        # values and layouts as clone; see recoverssm_snapshot_ring), else clone.
        ring = getattr(self, "_snapshot_ring", None)
        slot, copier = (None, None) if ring is None else ring.acquire()
        if copier is None:
            return DeferredRecoverSSMStep(
                tuple(m.snapshot_for_deferred_commit() for m in live)
            )
        deferred = DeferredRecoverSSMStep(
            tuple(m.snapshot_for_deferred_commit(copier) for m in live)
        )
        ring.bind(slot, deferred)
        return deferred

    def restore_step(self, deferred: DeferredRecoverSSMStep | None) -> None:
        """Make a detached step the one the next `commit_step` commits."""
        if self._step is not None:
            logger.warning_once(
                "RecoverSSM: dropping a recorded step that was never committed "
                "(expected only after a PIECEWISE capture)"
            )
        self._step = None if deferred is None else deferred.metadata
        self._step_is_deferred = deferred is not None
        self._step_owner = deferred

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
        # Commit kernels warmed at capture (graph or
        # eager) on every PP rank
        if for_capture:
            self._step = None
            if num_accepted_tokens is not None:
                captured = self._commit_graphs.capture_warmup(
                    tuple(step),
                    state_indices=state_indices,
                    num_accepted_tokens=num_accepted_tokens,
                )
                if not captured:

                    self._warm_eager_commit(
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
        # Deferred step released only once its commit
        # is enqueued
        self._step_is_deferred = False
        # Keep the deferred step (its ring
        # slot) alive until this commit has been enqueued.
        owner = getattr(self, "_step_owner", None)
        self._step_owner = None
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
        del owner  # enqueued: slot reusable

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
        # Align postprocess launches already
        # issued this step: (request_indices, num_computed_tokens, geometry).
        # The tensors themselves are held (compared with `is`), so an id can
        # never be recycled by a freed tensor within the loop.
        aligned: list[tuple[Any, ...]] = []
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

            # The launch reads idx_mapping, num_sampled, request_indices and
            # num_computed_tokens and writes state_indices[req] and
            # num_accepted_tokens[req]; no commit in this chain writes any of
            # its inputs. A launch whose per-group inputs are the SAME tensors
            # (identity, not equal values) with the same geometry as an earlier
            # one rewrites the same values to the same addresses: skipping it is
            # bit-identical. Groups sharing the batch's align count tensor
            # (gdn_recoverssm metadata reuse, the snapshot ring) dedupe;
            # staged copies (commit-graph path) are distinct tensors and keep
            # every launch.
            align_key = (
                postprocess_meta.request_indices,
                postprocess_meta.num_computed_tokens,
                postprocess_meta.num_spec_decodes,
                postprocess_meta.block_size,
                postprocess_meta.block_table.shape[1],
            )
            if any(
                k[0] is align_key[0] and k[1] is align_key[1] and k[2:] == align_key[2:]
                for k in aligned
            ):
                continue
            aligned.append(align_key)
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

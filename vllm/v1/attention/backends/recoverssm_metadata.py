# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.

import abc
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import torch


@dataclass(frozen=True)
class RecoverSSMPostprocessMetadata:
    """Metadata used during postprocessing for align-mode prefix caching."""

    num_spec_decodes: int
    request_indices: torch.Tensor | None
    block_table: torch.Tensor
    num_computed_tokens: torch.Tensor
    block_size: int


class RecoverSSMMetadata(abc.ABC):
    @abc.abstractmethod
    def commit_recoverssm_state(
        self, num_accepted_tokens: torch.Tensor
    ) -> RecoverSSMPostprocessMetadata | None:
        raise NotImplementedError

    def has_recoverssm_commit(self) -> bool:
        """Whether this step's metadata carries a commit (any spec row)."""
        return getattr(self, "recoverssm_commit", None) is not None

    def snapshot_for_deferred_commit(
        self, copy: "SnapshotCopy | None" = None
    ) -> "RecoverSSMMetadata":
        """A copy whose commit inputs no longer alias per-step buffers.

        Under pipeline parallelism a non-last rank learns the accepted counts
        pp_size steps after its verify (the PP sampled-token broadcast), and
        the steps in between rebuild the builders' persistent, graph-staged
        buffers (spec state indices, query offsets, the batch-order block
        table). The deferred commit must read this step's values, so they are
        cloned here, on the current stream, before the next build overwrites
        them. Implementations that do not support the PP deferral raise.

        ``copy`` replaces ``Tensor.clone`` for
        every cloned tensor: the snapshot ring copies into persistent buffers
        with the same values and layout (see clone_commit_tensors)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support the pipeline-parallel "
            "RecoverSSM commit deferral"
        )


def recoverssm_spec_rows(
    is_prefilling_cpu: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    num_decode_draft_tokens_cpu: torch.Tensor,
    num_reqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Which batch rows take the RecoverSSM spec path, shared by the RecoverSSM metadata
    builders (GDN, PLE).

    Every active decode row does (a draft-less step is a window of one token), plus
    every row that has drafts. Only the first ``num_reqs`` rows are real; padding rows
    never take the spec path. Returns the draft-count vector to hand to the base builder
    (1 for spec rows, -1 otherwise, padding included) and the spec rows in batch order,
    which is also the order of the base builder's spec state indices.

    Computed with NumPy views of the CPU tensors: the same values and dtypes as
    the torch formulation (drafts keep the input dtype, rows are int64), at a
    fraction of the per-op dispatch cost of six small CPU torch ops."""
    query_lens = np.diff(query_start_loc_cpu.numpy()[: num_reqs + 1])
    active_decode = ~is_prefilling_cpu.numpy()[:num_reqs] & (query_lens > 0)
    draft_counts = num_decode_draft_tokens_cpu.numpy()
    spec_mask = (draft_counts[:num_reqs] >= 0) | active_decode
    drafts = np.full_like(draft_counts, -1)
    drafts[:num_reqs][spec_mask] = 1
    rows = np.flatnonzero(spec_mask).astype(np.int64, copy=False)
    return torch.from_numpy(drafts), torch.from_numpy(rows)


def recoverssm_request_indices(
    rows: torch.Tensor, num_spec_decodes: int, device: torch.device
) -> torch.Tensor | None:
    """None when the spec rows are exactly 0..n-1, else their batch rows (the commit
    plan's request mapping). Raises if the base builder counted a different number of
    spec rows than the classification above."""
    if rows.numel() != num_spec_decodes:
        raise ValueError(
            f"RecoverSSM: {rows.numel()} spec rows classified, but the builder "
            f"produced {num_spec_decodes}"
        )
    if num_spec_decodes == 0 or int(rows[-1]) == num_spec_decodes - 1:
        return None
    return rows.to(torch.int32).to(device, non_blocking=True)


SnapshotCopy = Callable[[torch.Tensor], torch.Tensor]


def clone_commit_tensors(
    commit, num_spec_decodes: int, copy: SnapshotCopy | None = None
):
    """Clone the commit inputs a deferred (PP) commit reads, trimmed to the real
    spec rows. Works for every RecoverSSM commit dataclass with the fields
    state_indices, query_start_loc, request_indices, block_table and
    num_computed_tokens. ``copy`` replaces
    ``Tensor.clone``; None keeps clone."""
    n = num_spec_decodes
    cp: SnapshotCopy = torch.Tensor.clone if copy is None else copy

    def c(t: torch.Tensor | None) -> torch.Tensor | None:
        return None if t is None else cp(t)

    return replace(
        commit,
        state_indices=cp(commit.state_indices[:n]),
        query_start_loc=cp(commit.query_start_loc[: n + 1]),
        request_indices=c(commit.request_indices),
        block_table=c(commit.block_table),
        num_computed_tokens=c(commit.num_computed_tokens),
    )

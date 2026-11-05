# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""IPC message definitions for PLE CPU offload (host-mapped pull design).

The offload process owns no CUDA context. Every GPU worker owns, per PLE
layer, a shared host output buffer and a shared host int32 flag, both
page-locked and device-mapped in the worker's context. The offload process
waits for ``flag == 0``, writes the layer result into the host buffer and
stores ``flag = 1`` with plain CPU stores; the GPU's captured forward waits on
the host flag, DMAs host -> GPU on the model stream, and ``release_outputs``
resets the flag after the forward.
"""

from dataclasses import dataclass

import msgspec
import torch


@dataclass
class PleOffloadRegistration:
    """Sent once from each GPU worker during offload setup."""

    worker_id: int
    tp_rank: int
    dp_rank: int
    # Shared CPU tensors per PLE layer, filled by the offload process.
    host_output_buffers: dict[str, torch.Tensor]
    # Shared CPU int32 tensors per PLE layer; element 0 is the ready flag.
    host_flag_tensors: dict[str, torch.Tensor]
    # CPU tensors are allocated in shared memory and registered once.
    input_ids_buf: torch.Tensor
    query_start_loc_buf: torch.Tensor
    ngram_context_buf: torch.Tensor | None


@dataclass
class PleOffloadRequest:
    """Sent by each DP rank's TP rank zero at every inference step."""

    dp_rank: int
    num_tokens: int
    num_reqs: int


_PLE_OFFLOAD_REQUEST_DECODER = msgspec.msgpack.Decoder(PleOffloadRequest)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""One QSA top-k index buffer per pipeline stage instead of one per layer.

Each ``Qwen4ExpQSAAttention`` registered its own persistent
``topk_indices_buffer`` of ``[max_num_batched_tokens, packed_output_width]``
int32 (Flash-Next at a 1024-token chunk: 1024 x 2052 x 4 B = 8.02 MiB), about
six per stage. Inside one target forward a layer's selection is written by its
indexer (``expand_qsa_block_indices``) and read by its own sparse attention
kernel within the same ``_run_qsa`` call, before that layer's output exists;
the next QSA layer's indexer input is computed from that output, so its write
is ordered after the read on any stream. No target-layer selection is read
after its layer returns, so the target layers of one stage can share one
buffer: same values, same addresses within a step, same kernels.

The MTP drafter is NOT shared: with ``index_share_for_mtp_iteration`` its
step-0 rows are compacted and reused by draft steps 1+ (``skip_topk``), i.e.
they live across forward calls. It is built outside the target model's scope
(``Qwen4ExpMultiTokenPredictor``), so it keeps a private buffer. The target
model must not expose a ``topk_indices_buffer`` attribute either: the
spec-decode loaders hand such an attribute to every draft module.

Usage: ``Qwen4ExpModel.__init__`` builds its layers inside
``shared_qsa_index_buffers()``; ``Qwen4ExpQSAAttention`` allocates through
``qsa_topk_index_buffer()``, which returns the scope's buffer for that exact
shape/dtype/device, or a private one outside any scope. Only identical shapes
share, so per-layer widths never alias a strided view.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator
from dataclasses import dataclass, field

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_MIB = 1 << 20

_BufferKey = tuple[int, int, torch.dtype, str]


@dataclass
class QSAIndexBufferScope:
    """The buffers handed out inside one ``shared_qsa_index_buffers`` scope."""

    label: str = ""
    buffers: dict[_BufferKey, torch.Tensor] = field(default_factory=dict)
    requests: int = 0
    requested_bytes: int = 0

    @property
    def distinct(self) -> int:
        return len(self.buffers)

    @property
    def allocated_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.buffers.values())

    @property
    def saved_bytes(self) -> int:
        return self.requested_bytes - self.allocated_bytes


_ACTIVE: contextvars.ContextVar[QSAIndexBufferScope | None] = (
    contextvars.ContextVar("merlin_qsa_index_buffer_scope", default=None)
)


def _default_device() -> torch.device:
    get = getattr(torch, "get_default_device", None)
    return torch.device(get()) if get is not None else torch.empty(0).device


def qsa_topk_index_buffer(
    rows: int, cols: int, dtype: torch.dtype = torch.int32
) -> torch.Tensor:
    """The ``topk_indices_buffer`` for one QSA layer.

    Allocated exactly as before (``torch.empty`` on the default device).
    Inside a scope, layers asking for the same shape, dtype and device get
    the same tensor.
    """
    scope = _ACTIVE.get()
    if scope is None:
        return torch.empty(rows, cols, dtype=dtype)
    key: _BufferKey = (int(rows), int(cols), dtype, str(_default_device()))
    buffer = scope.buffers.get(key)
    if buffer is None:
        buffer = torch.empty(rows, cols, dtype=dtype)
        scope.buffers[key] = buffer
    scope.requests += 1
    scope.requested_bytes += buffer.numel() * buffer.element_size()
    return buffer


@contextlib.contextmanager
def shared_qsa_index_buffers(label: str = "") -> Iterator[QSAIndexBufferScope]:
    """Share QSA top-k index buffers among the layers built in this scope.

    Logs, once at exit, the number of QSA layers, the distinct buffers and
    the MiB saved on this rank.
    """
    scope = QSAIndexBufferScope(label=label)
    token = _ACTIVE.set(scope)
    try:
        yield scope
    finally:
        _ACTIVE.reset(token)
    if scope.requests:
        logger.info(
            "Qwen4Exp QSA shared top-k index buffers%s: %d QSA layer(s) "
            "-> %d distinct buffer(s), %.2f MiB allocated, %.2f MiB saved "
            "per rank",
            f" [{label}]" if label else "",
            scope.requests,
            scope.distinct,
            scope.allocated_bytes / _MIB,
            scope.saved_bytes / _MIB,
        )


__all__ = [
    "QSAIndexBufferScope",
    "qsa_topk_index_buffer",
    "shared_qsa_index_buffers",
]

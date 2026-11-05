# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.

from functools import lru_cache

import torch

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import direct_register_custom_op


def _custom_collective(name: str, x: torch.Tensor) -> torch.Tensor | None:
    device_communicator = get_tp_group().device_communicator
    if device_communicator is None:
        return None
    collective = getattr(device_communicator, name, None)
    return None if collective is None else collective(x)


@lru_cache
def is_sequence_parallel_active(is_enabled: bool, num_tokens: int) -> bool:
    """Whether a batch of ``num_tokens`` shards the stream along the sequence.

    Below one token per rank the shards are mostly padding, so the batch
    stays replicated.
    """
    return is_enabled and num_tokens >= get_tensor_model_parallel_world_size()


def sp_all_gather(x: torch.Tensor) -> torch.Tensor:
    output = _custom_collective("custom_all_gather", x)
    if output is not None:
        return output
    return tensor_model_parallel_all_gather(x, 0)


def sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    if torch.compiler.is_compiling():
        # See _sp_pad_rows_op.
        x = torch.ops.vllm.sp_pad_rows(x)
    else:
        tp_size = get_tensor_model_parallel_world_size()
        sp_pad = (-x.shape[0]) % tp_size
        if sp_pad > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, sp_pad))
    output = _custom_collective("custom_reduce_scatter", x)
    if output is not None:
        return output
    return tensor_model_parallel_reduce_scatter(x, 0)


def sp_shard(x: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        # See _sp_shard_rows_op.
        return torch.ops.vllm.sp_shard_rows(x, False)
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    sp_pad = (-x.shape[0]) % tp_size
    if sp_pad > 0:
        pad = (0, 0) * (x.ndim - 1) + (0, sp_pad)
        x = torch.nn.functional.pad(x, pad)
    chunk = x.shape[0] // tp_size
    return x[tp_rank * chunk : (tp_rank + 1) * chunk]


def sp_padding_mask(
    is_padding: torch.Tensor | None,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    if is_padding is None:
        is_padding = hidden_states.new_zeros(num_tokens, dtype=torch.bool)
    assert is_padding.shape[0] == num_tokens

    if torch.compiler.is_compiling():
        # See _sp_shard_rows_op; SP padding rows count as padding.
        return torch.ops.vllm.sp_shard_rows(is_padding, True)
    tp_size = get_tensor_model_parallel_world_size()
    sp_pad = (-num_tokens) % tp_size
    if sp_pad > 0:
        is_padding = torch.nn.functional.pad(is_padding, (0, sp_pad), value=True)
    chunk = is_padding.shape[0] // tp_size
    tp_rank = get_tensor_model_parallel_rank()
    return is_padding[tp_rank * chunk : (tp_rank + 1) * chunk]


# Opaque SP padding for compiled models.
#
# The helpers above branch in Python on ``(-N) % tp``. Dynamo turns that into
# a guard on the parity of N; vLLM's compile wrapper drops guards, so the
# branch taken at the compile-time dummy run would be frozen for every later
# N (an odd N then loses its last token, N=1 at TP2 shards to zero rows).
# Inside a Dynamo trace the pad/shard therefore goes through these opaque ops
# -- the pattern upstream uses for ``sequence_parallel_chunk`` -- whose fake
# impls give the output size as a symbolic ``ceil(N/tp)`` without branching.
# Eager callers (including every model that is not torch.compiled) keep the
# exact code above. Pure device work on the current stream, no host sync and
# no request metadata, so the ops are safe inside piecewise CUDA graphs and
# need no splitting-op entry. CompositeExplicitAutograd: one impl for every
# backend, not decomposed by the compiler.


def _pad_rows(x: torch.Tensor, value: bool) -> torch.Tensor:
    """Zero- (or ``value``-) pad dim 0 to a multiple of TP; always a copy."""
    tp_size = get_tensor_model_parallel_world_size()
    sp_pad = (-x.shape[0]) % tp_size
    pad = (0, 0) * (x.ndim - 1) + (0, sp_pad)
    return torch.nn.functional.pad(x, pad, value=float(value))


def _sp_pad_rows_op(x: torch.Tensor) -> torch.Tensor:
    return _pad_rows(x, False)


def _sp_pad_rows_fake(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    rows = cdiv(x.shape[0], tp_size) * tp_size
    return x.new_empty((rows, *x.shape[1:]))


def _sp_shard_rows_op(x: torch.Tensor, pad_value: bool) -> torch.Tensor:
    """``sp_shard`` (``pad_value=False``) / ``sp_padding_mask`` (``True``)."""
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    padded = _pad_rows(x, pad_value)
    chunk = padded.shape[0] // tp_size
    # Contiguous rows of a fresh tensor: never an alias of the input.
    return padded[tp_rank * chunk : (tp_rank + 1) * chunk]


def _sp_shard_rows_fake(x: torch.Tensor, pad_value: bool) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    return x.new_empty((cdiv(x.shape[0], tp_size), *x.shape[1:]))


direct_register_custom_op(
    op_name="sp_pad_rows",
    op_func=_sp_pad_rows_op,
    fake_impl=_sp_pad_rows_fake,
    dispatch_key="CompositeExplicitAutograd",
    tags=(torch.Tag.needs_fixed_stride_order,),
)

direct_register_custom_op(
    op_name="sp_shard_rows",
    op_func=_sp_shard_rows_op,
    fake_impl=_sp_shard_rows_fake,
    dispatch_key="CompositeExplicitAutograd",
    tags=(torch.Tag.needs_fixed_stride_order,),
)

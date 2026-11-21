# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""Adapted from #68296 nvidia/engram.py:209-451.

GPU-only imports are lazy so storage and projection loading can be tested on CPU.

Table formats (#68296 engram.py:259-348; sglang #51159 uses the same): BF16
rows, or FP8 E4M3 rows ``weight`` [rows, head_dim] with one BF16 scale per row
``weight_scale_inv`` [rows]. A looked-up row dequantizes as
``weight[row].float() * weight_scale_inv[row].float()``. The table keeps the
checkpoint's dtype in host RAM; only the looked-up rows are dequantized.
"""

import math
import mmap
import sys
import weakref

import torch
from torch import nn

from vllm.models.hy_v4_flash.hashing import head_range

# Exact-size pinned host tables. torch's pinned
# allocator rounds every block up to a power of two (CachingHostAllocator.h,
# PowerOf2Ceil), so a 7.4 GiB FP8 half-table would pin 8 GiB and a 14.8 GiB
# BF16 one 16 GiB. An anonymous mapping registered with cudaHostRegister pins
# exactly the table's bytes (the pattern of deepseek_v41/common/engram.py).


def _unregister(mapping, pointer):
    result = torch.cuda.cudart().cudaHostUnregister(pointer)
    if result.value != 0:
        from vllm.logger import init_logger

        init_logger(__name__).warning(
            "Flash Engram cudaHostUnregister failed: %s", result
        )
    del mapping


def host_table_empty(shape, dtype):
    """Uninitialised host storage of exactly ``prod(shape) * itemsize`` bytes.

    Pinned (CUDA-registered) when CUDA is available, else plain CPU memory (the
    CPU platform has no pinning; its UVA view is the CPU tensor itself).
    """
    shape = tuple(shape)
    nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if not torch.cuda.is_available():
        return torch.empty(shape, dtype=dtype, device="cpu")
    if nbytes == 0 or sys.platform != "linux":
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
    import numpy as np

    mapping = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    owner = np.frombuffer(mapping, dtype=np.uint8)
    pointer = owner.ctypes.data
    result = torch.cuda.cudart().cudaHostRegister(pointer, nbytes, 0)
    if result.value != 0:
        from vllm.logger import init_logger

        init_logger(__name__).warning(
            "Flash Engram cudaHostRegister failed (%s); using the pinned "
            "allocator (rounds up to a power of two)",
            result,
        )
        del owner
        mapping.close()
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
    # The storage keeps the numpy owner alive (through views and UVA views); the
    # mapping stays alive until CUDA has released the registration.
    finalizer = weakref.finalize(owner, _unregister, mapping, pointer)
    finalizer.atexit = False
    tensor = torch.from_numpy(owner)
    if not tensor.is_pinned():
        raise RuntimeError("cudaHostRegister did not pin the Flash Engram table")
    return tensor.view(dtype).view(shape)


def dummy_table_dtype(config):
    """Table dtype of a dummy-weight load (#68296 engram.py:367-371)."""
    return (
        torch.float8_e4m3fn
        if getattr(config, "engram_fp8_embedding", False)
        else torch.bfloat16
    )


def _copy(parameter, value):
    if parameter.shape != value.shape:
        raise ValueError(
            f"Engram shape mismatch: {tuple(value.shape)} != {tuple(parameter.shape)}"
        )
    with torch.no_grad():
        parameter.copy_(value)


class FlashEmbedding(nn.Module):
    def __init__(self, sizes, dim, *, tp_size=1, rank=0, cpu_offload=True, dtype=None):
        super().__init__()
        self.tp_size = tp_size
        self.head_start, self.head_end = head_range(sizes, tp_size, rank)
        self.vocab_start = sum(sizes[: self.head_start])
        self.vocab_end = sum(sizes[: self.head_end])
        self.num_embeddings = sum(sizes)
        self.n_hash_cols = len(sizes)
        self.dim = dim
        self.cpu_offload = cpu_offload
        if cpu_offload:
            from vllm.utils.platform_utils import is_uva_available

            if not is_uva_available():
                raise RuntimeError("Flash Engram host offload requires UVA")
        self.embedding = nn.Module()
        rows = self.vocab_end - self.vocab_start
        self.embedding.weight = nn.Parameter(
            self._empty(
                (0,) if dtype is None else (rows, dim),
                dtype or torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.embedding.register_parameter("weight_scale_inv", None)
        self.embedding.weight.weight_loader = self._load_shard
        self.fp8 = dtype == torch.float8_e4m3fn
        self._loaded = set()
        if self.fp8:
            self._make_scale(rows)
        self._uva_weight = None
        self._uva_scale = None

    def _empty(self, shape, dtype, device=None):
        if self.cpu_offload:
            return host_table_empty(shape, dtype)
        return torch.empty(shape, dtype=dtype, device=device)

    def host_bytes(self):
        """Bytes this rank's table part holds (rows plus FP8 row scales)."""
        return sum(
            parameter.untyped_storage().nbytes()
            for parameter in self.embedding.parameters()
        )

    def _make_scale(self, rows=0):
        self.embedding.weight_scale_inv = nn.Parameter(
            self._empty((rows,), torch.bfloat16, self.embedding.weight.device),
            requires_grad=False,
        )
        self.embedding.weight_scale_inv.weight_loader = self._load_shard

    def load_weight(self, leaf, value):
        if leaf not in ("weight", "weight_scale_inv"):
            raise ValueError(f"Unknown Engram table leaf: {leaf}")
        if leaf == "weight_scale_inv" and self.embedding.weight_scale_inv is None:
            self._make_scale()
        self._load_shard(getattr(self.embedding, leaf), value)

    def _load_shard(self, parameter, value):
        is_weight = parameter is self.embedding.weight
        leaf = "weight" if is_weight else "weight_scale_inv"
        shape = (self.num_embeddings, self.dim) if is_weight else (self.num_embeddings,)
        dtypes = (
            (torch.bfloat16, torch.float8_e4m3fn) if is_weight else (torch.bfloat16,)
        )
        if tuple(value.shape) != shape or value.dtype not in dtypes:
            raise ValueError(
                f"Invalid Engram {leaf}: {tuple(value.shape)}/{value.dtype}"
            )
        if leaf in self._loaded and parameter.dtype != value.dtype:
            raise ValueError("Engram table dtype changed during reload")
        if is_weight:
            fp8 = value.dtype == torch.float8_e4m3fn
            if not fp8 and self.embedding.weight_scale_inv is not None:
                raise ValueError("BF16 Engram tables must not have FP8 row scales")
            self.fp8 = fp8
        elif "weight" in self._loaded and not self.fp8:
            raise ValueError("BF16 Engram tables must not have FP8 row scales")
        if parameter.numel() == 0:
            # The checkpoint's dtype (FP8 stays FP8): no BF16 copy is made.
            parameter.data = self._empty(
                (self.vocab_end - self.vocab_start, *shape[1:]),
                value.dtype,
                parameter.device,
            )
        _copy(parameter, value[self.vocab_start : self.vocab_end])
        self._loaded.add(leaf)
        if is_weight and self.fp8 and self.embedding.weight_scale_inv is None:
            self._make_scale()
        self._uva_weight = None
        self._uva_scale = None

    def _lookup(self, parameter, indices, out, *, scale=False):
        row_bytes = parameter.element_size() * (1 if scale else self.dim)
        if indices.device.type == "cpu":
            local = (indices - self.vocab_start).clamp(
                0, self.vocab_end - self.vocab_start - 1
            )
            raw = parameter.view(torch.uint8).reshape(parameter.shape[0], row_bytes)
            selected = raw.index_select(0, local.flatten())
            out.copy_(selected.view(parameter.dtype).reshape_as(out))
            return
        from vllm.models.qwen4_exp.common.ngram_embedding import (
            _lookup_ple_embedding_from_pinned_kernel,
        )
        from vllm.triton_utils import triton
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        name = "_uva_scale" if scale else "_uva_weight"
        view = getattr(self, name)
        if view is None:
            view = (
                get_accelerator_view_from_cpu_tensor(parameter)
                if self.cpu_offload
                else parameter
            )
            setattr(self, name, view)
        _lookup_ple_embedding_from_pinned_kernel[(indices.numel(),)](
            view.view(torch.uint8),
            indices,
            out.view(torch.uint8),
            row_bytes,
            self.vocab_start,
            self.vocab_end,
            BLOCK_D=triton.next_power_of_2(row_bytes),
        )

    def forward(self, hashes):
        if "weight" not in self._loaded and self.embedding.weight.numel() == 0:
            raise RuntimeError("Engram table has not been loaded")
        if self.fp8 and "weight_scale_inv" not in self._loaded:
            raise RuntimeError("Missing Engram FP8 row scales")
        indices = hashes[:, self.head_start : self.head_end].long().contiguous()
        rows = torch.empty(
            (*indices.shape, self.dim),
            dtype=self.embedding.weight.dtype,
            device=hashes.device,
        )
        if indices.numel() == 0:
            return rows.float()
        self._lookup(self.embedding.weight, indices, rows)
        result = rows.float()
        if self.fp8:
            scales = torch.empty(
                indices.shape, dtype=torch.bfloat16, device=hashes.device
            )
            self._lookup(self.embedding.weight_scale_inv, indices, scales, scale=True)
            result = result * scales.float().unsqueeze(-1)
        owned = (indices >= self.vocab_start) & (indices < self.vocab_end)
        return result.masked_fill(~owned.unsqueeze(-1), 0)


class FlashEngram(nn.Module):
    def __init__(
        self,
        config,
        layout,
        layer_id,
        *,
        tp_size=1,
        rank=0,
        cpu_offload=True,
        dummy_dtype=None,
    ):
        super().__init__()
        index = layout.layer_ids.index(layer_id)
        sizes = [size for order in layout.primes[index] for size in order]
        self.multi_head_embedding = FlashEmbedding(
            sizes,
            layout.head_dim,
            tp_size=tp_size,
            rank=rank,
            cpu_offload=cpu_offload,
            dtype=dummy_dtype,
        )
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.rank = rank
        local = (
            self.multi_head_embedding.head_end - self.multi_head_embedding.head_start
        )
        self.lookup_per_head_norms = nn.ModuleList(
            [nn.RMSNorm(layout.head_dim) for _ in range(local)]
        )
        self.value_proj = nn.Linear(local * layout.head_dim, self.hidden_size)
        self.key_projs = nn.ModuleList(
            [
                nn.Linear(local * layout.head_dim, self.hidden_size)
                for _ in range(self.hc_mult)
            ]
        )
        self.norm_key = nn.ModuleList(
            [nn.RMSNorm(self.hidden_size) for _ in range(self.hc_mult)]
        )
        self.norm_query = nn.ModuleList(
            [nn.RMSNorm(self.hidden_size) for _ in range(self.hc_mult)]
        )
        self.unified_pre_gamma = nn.Parameter(torch.empty(self.hidden_size))
        self.unified_per_hc_scales = nn.ParameterList(
            [nn.Parameter(torch.empty(self.hidden_size)) for _ in range(self.hc_mult)]
        )
        self.total_width = len(sizes) * layout.head_dim
        self.start = self.multi_head_embedding.head_start * layout.head_dim
        self.stop = self.multi_head_embedding.head_end * layout.head_dim
        for projection in [self.value_proj, *self.key_projs]:
            projection.weight.weight_loader = self._load_projection
            projection.bias.weight_loader = self._load_bias

    def _load_projection(self, parameter, value):
        if value.shape != (self.hidden_size, self.total_width):
            raise ValueError("Flash Engram projection weight shape mismatch")
        _copy(parameter, value[:, self.start : self.stop])

    def _load_bias(self, parameter, value):
        _copy(parameter, value if self.rank == 0 else torch.zeros_like(value))

    def _reduce(self, value):
        if self.multi_head_embedding.tp_size == 1:
            return value
        from vllm.distributed import tensor_model_parallel_all_reduce

        return tensor_model_parallel_all_reduce(value)

    def forward(self, hidden, hashes):
        rows = self.multi_head_embedding(hashes)
        eps = torch.finfo(torch.float32).eps
        gamma = torch.stack(
            [norm.weight for norm in self.lookup_per_head_norms]
        ).float()
        x = rows * (rows.square().mean(-1, keepdim=True) + eps).rsqrt() * gamma
        x = x.to(hidden.dtype).flatten(1)
        gates = []
        for channel in range(self.hc_mult):
            key = self._reduce(self.key_projs[channel](x))
            gate = (
                self.norm_key[channel](key)
                * self.norm_query[channel](hidden[:, channel])
            ).sum(-1) * self.hidden_size**-0.5
            gates.append((gate.abs().clamp_min(1e-6).sqrt() * gate.sign()).sigmoid())
        gate = torch.stack(gates, dim=1)
        value = self._reduce(self.value_proj(x)).float()
        signal = (
            self.unified_pre_gamma.float()
            * value
            * (value.square().mean(-1, keepdim=True) + eps).rsqrt()
        )
        scales = torch.stack(list(self.unified_per_hc_scales)).float()
        result = ((gate.float().unsqueeze(-1) + scales) * signal.unsqueeze(1)).to(
            hidden.dtype
        )
        return hidden + result.masked_fill((hashes < 0).all(-1)[:, None, None], 0)

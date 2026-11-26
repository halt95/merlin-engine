# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 lebeche hill-climb (fork of Merlin 3.0.2.2)
"""Lossless TP=2 PCIe P2P CUDA-IPC AllGather for sm_8x (CuTe-DSL).

Replaces NCCL Ring LL (`ncclDevKernel_AllGather_RING_LL`) for TP=2 all-gathers
up to 8 MiB/rank on single-node P2P-capable sm_8x GPUs (e.g. 4x RTX 3090).

Two 100% bit-exact, CUDA-graph capture & replay safe kernels share one
symmetric per-rank CUDA IPC slab (~16.3 MiB, allocated once at TP group init):

1. Small payloads (<= 64 KiB/rank, 4-byte divisible):
   ``CuteTp2Atomic64AllGather`` packs each 32-bit payload word with an odd
   per-call tag into ONE single-copy-atomic 64-bit PTX store
   (``st.volatile.global.u64``) and polls ``ld.volatile.global.u64`` with zero
   system fences and a single PCIe round-trip (~2.4-3.8 us).

2. Medium/large payloads (64 KiB < bytes <= 8 MiB/rank, 16-byte divisible):
   ``CuteTp2Flag128AllGather`` pushes raw 128-bit ``uint4`` packs with zero
   bandwidth overhead, synchronizes per-CTA with system-scope release/acquire
   flags (``fence.acq_rel.sys`` + ``st.release.sys`` / ``ld.acquire.sys``), and
   copies the peer shard to output (~18.9 us at 242 KB, ~50.8 us at 970 KB).

Both kernels use device-side call-level double-buffered slot selection (safe
under arbitrary CUDA graph replay and dynamic grid sizes across calls) and
fuse ``dim=-1`` scatter directly on write when aligned, eliminating the
``movedim`` + ``reshape`` copy kernel.

Collective contract: ``should_use(input_, dim)`` depends strictly on
init-agreed state and ``(input_.shape, input_.element_size(), dim)`` — never on
rank-local pointers or contiguity — so both ranks always take the identical
branch (P2P fast path or stock PyNccl fallback).
"""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# ---- Workspace geometry ----
_MAX_BLOCKS = 64
_SMALL_MAX_BYTES = 64 * 1024          # 64 KiB raw payload -> Atomic64 path
_LARGE_MAX_BYTES = 8 * 1024 * 1024    # 8 MiB raw payload -> Flag128 path

# Signal region layout (bytes, 4096-aligned):
#   [0]:          flag_call_epoch (u32)
#   [4]:          flag_blocks_arrived (u32)
#   [8]:          tag_call_epoch (u32)
#   [12]:         tag_blocks_arrived (u32)
#   [128..383]:   flag_self_counter[0..63] (u32 each)
#   [512..16895]: flag_slots[slot 0..1][block 0..63] (128 B cacheline each)
_TAG_EPOCH_OFF = 8
_TAG_ARRIVED_OFF = 12
_SELF_COUNTER_OFF = 128
_FLAG_BASE_OFF = 512
_FLAG_STRIDE_BYTES = 128
_SIGNAL_BYTES = 20480  # 20 KiB, 4096-aligned

_TAG_SLOT_BYTES = _SMALL_MAX_BYTES * 2  # 128 KiB per slot (8 B per 4 B word)
_FLAG_SLOT_BYTES = _LARGE_MAX_BYTES     # 8 MiB per slot

_TAG_STG0_OFF = _SIGNAL_BYTES
_TAG_STG1_OFF = _TAG_STG0_OFF + _TAG_SLOT_BYTES
_FLAG_STG0_OFF = _TAG_STG1_OFF + _TAG_SLOT_BYTES
_FLAG_STG1_OFF = _FLAG_STG0_OFF + _FLAG_SLOT_BYTES
_SLAB_TOTAL_BYTES = _FLAG_STG1_OFF + _FLAG_SLOT_BYTES  # ~16.27 MiB

# ---- CuTe-DSL kernels ----
try:
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from cutlass import Int32, Int64, Uint32, Uint64, const_expr

    _HAS_CUTE = True
except Exception:  # noqa: BLE001
    _HAS_CUTE = False


if _HAS_CUTE:
    def _ld_global_nc_u32(addr_u64):
        return cute.arch.inline_ptx(
            "ld.global.nc.u32 {$w0}, [{$r0}];",
            write_only_types=[Uint32],
            read_only_args=[addr_u64],
        )

    def _st_global_u32(addr_u64, val_u32):
        cute.arch.inline_ptx(
            "st.global.u32 [{$r0}], {$r1};",
            read_only_args=[addr_u64, val_u32],
        )

    def _ld_global_u32(addr_u64):
        return cute.arch.inline_ptx(
            "ld.global.u32 {$w0}, [{$r0}];",
            write_only_types=[Uint32],
            read_only_args=[addr_u64],
        )

    def _ld_relaxed_gpu_u32(addr_u64):
        return cute.arch.inline_ptx(
            "ld.relaxed.gpu.global.u32 {$w0}, [{$r0}];",
            write_only_types=[Uint32],
            read_only_args=[addr_u64],
        )

    def _atom_inc_u32(addr_u64, limit_u32):
        return cute.arch.inline_ptx(
            "atom.global.inc.u32 {$w0}, [{$r0}], {$r1};",
            write_only_types=[Uint32],
            read_only_args=[addr_u64, limit_u32],
        )

    def _st_vol_u64_pair(addr_u64, lo_u32, hi_u32):
        cute.arch.inline_ptx(
            "{\n"
            ".reg .b64 packed;\n"
            "mov.b64 packed, {{$r1}, {$r2}};\n"
            "st.volatile.global.u64 [{$r0}], packed;\n"
            "}",
            read_only_args=[addr_u64, lo_u32, hi_u32],
        )

    def _st_vol_u64_zero(addr_u64):
        cute.arch.inline_ptx(
            "st.volatile.global.u64 [{$r0}], 0;",
            read_only_args=[addr_u64],
        )

    def _ld_vol_u64_pair(addr_u64):
        return cute.arch.inline_ptx(
            "{\n"
            ".reg .b64 packed;\n"
            "ld.volatile.global.u64 packed, [{$r0}];\n"
            "mov.b64 {{$w0}, {$w1}}, packed;\n"
            "}",
            write_only_types=[Uint32, Uint32],
            read_only_args=[addr_u64],
        )

    def _ld_global_nc_u4(addr_u64):
        return cute.arch.inline_ptx(
            "ld.global.nc.v4.b32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
            write_only_types=[Uint32, Uint32, Uint32, Uint32],
            read_only_args=[addr_u64],
        )

    def _ld_global_cv_u4(addr_u64):
        return cute.arch.inline_ptx(
            "ld.global.cv.v4.b32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
            write_only_types=[Uint32, Uint32, Uint32, Uint32],
            read_only_args=[addr_u64],
        )

    def _st_global_u4(addr_u64, v0, v1, v2, v3):
        cute.arch.inline_ptx(
            "st.global.v4.b32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
            read_only_args=[addr_u64, v0, v1, v2, v3],
        )

    def _st_global_wt_u4(addr_u64, v0, v1, v2, v3):
        cute.arch.inline_ptx(
            "st.global.wt.v4.b32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
            read_only_args=[addr_u64, v0, v1, v2, v3],
        )

    def _st_release_sys_u32(addr_u64, val_u32):
        cute.arch.inline_ptx(
            "st.release.sys.global.u32 [{$r0}], {$r1};",
            read_only_args=[addr_u64, val_u32],
        )

    def _ld_acquire_sys_u32(addr_u64):
        return cute.arch.inline_ptx(
            "ld.acquire.sys.global.u32 {$w0}, [{$r0}];",
            write_only_types=[Uint32],
            read_only_args=[addr_u64],
        )

    class CuteTp2Atomic64AllGather:
        """Zero-fence single-RTT bit-exact TP=2 AllGather (<= 64 KiB/rank)."""

        def __init__(self, *, threads: int = 64, fuse_dim_last: bool = False):
            self.threads = threads
            self.fuse_dim_last = fuse_dim_last

        @cute.jit
        def __call__(
            self,
            inp_ptr: Int64,
            out_ptr: Int64,
            self_sig_ptr: Int64,
            self_stg0_ptr: Int64,
            self_stg1_ptr: Int64,
            peer_stg0_ptr: Int64,
            peer_stg1_ptr: Int64,
            total_words: Int32,
            row_words: Int32,
            rank: Int32,
            grid_x: Int32,
            stream: CUstream,
        ):
            self.kernel(
                inp_ptr,
                out_ptr,
                self_sig_ptr,
                self_stg0_ptr,
                self_stg1_ptr,
                peer_stg0_ptr,
                peer_stg1_ptr,
                total_words,
                row_words,
                rank,
            ).launch(grid=(grid_x, 1, 1), block=(self.threads, 1, 1), stream=stream)

        @cute.kernel
        def kernel(
            self,
            inp_ptr: Int64,
            out_ptr: Int64,
            self_sig_ptr: Int64,
            self_stg0_ptr: Int64,
            self_stg1_ptr: Int64,
            peer_stg0_ptr: Int64,
            peer_stg1_ptr: Int64,
            total_words: Int32,
            row_words: Int32,
            rank: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            gdim, _, _ = cute.arch.grid_dim()
            bdim = const_expr(self.threads)

            epoch_addr = Uint64(self_sig_ptr) + Uint64(_TAG_EPOCH_OFF)
            arrived_addr = Uint64(self_sig_ptr) + Uint64(_TAG_ARRIVED_OFF)
            epoch_val = _ld_relaxed_gpu_u32(epoch_addr)
            slot = epoch_val & Uint32(1)
            # Odd per-call tag (never 0); local slot is reset to 0 after read.
            tag = (epoch_val << Uint32(1)) | Uint32(1)
            cute.arch.sync_threads()
            if tidx == Int32(0):
                if gdim == Int32(1):
                    _st_global_u32(epoch_addr, epoch_val + Uint32(1))
                else:
                    last = Uint32(gdim - Int32(1))
                    prior = _atom_inc_u32(arrived_addr, last)
                    if prior == last:
                        _st_global_u32(epoch_addr, epoch_val + Uint32(1))

            peer_stg = Uint64(peer_stg0_ptr)
            local_stg = Uint64(self_stg0_ptr)
            if slot != Uint32(0):
                peer_stg = Uint64(peer_stg1_ptr)
                local_stg = Uint64(self_stg1_ptr)

            inp_u64 = Uint64(inp_ptr)
            out_u64 = Uint64(out_ptr)
            peer = rank ^ Int32(1)

            w = bidx * Int32(bdim) + tidx
            stride = gdim * Int32(bdim)
            while w < total_words:
                val = _ld_global_nc_u32(inp_u64 + Uint64(Int64(w) * Int64(4)))
                stg_off = Uint64(Int64(w) * Int64(8))
                _st_vol_u64_pair(peer_stg + stg_off, val, tag)

                if const_expr(self.fuse_dim_last):
                    row = w // row_words
                    col = w - row * row_words
                    out_self_w = (row * Int32(2) + rank) * row_words + col
                    out_peer_w = (row * Int32(2) + peer) * row_words + col
                else:
                    out_self_w = rank * total_words + w
                    out_peer_w = peer * total_words + w

                _st_global_u32(out_u64 + Uint64(Int64(out_self_w) * Int64(4)), val)

                p_val, p_tag = _ld_vol_u64_pair(local_stg + stg_off)
                while p_tag != tag:
                    p_val, p_tag = _ld_vol_u64_pair(local_stg + stg_off)
                _st_global_u32(out_u64 + Uint64(Int64(out_peer_w) * Int64(4)), p_val)
                _st_vol_u64_zero(local_stg + stg_off)
                w = w + stride

    class CuteTp2Flag128AllGather:
        """Raw 128-bit PCIe P2P TP=2 AllGather (64 KiB < bytes <= 8 MiB/rank)."""

        def __init__(self, *, threads: int = 256, fuse_dim_last: bool = False):
            self.threads = threads
            self.fuse_dim_last = fuse_dim_last

        @cute.jit
        def __call__(
            self,
            inp_ptr: Int64,
            out_ptr: Int64,
            self_sig_ptr: Int64,
            peer_sig_ptr: Int64,
            self_stg0_ptr: Int64,
            self_stg1_ptr: Int64,
            peer_stg0_ptr: Int64,
            peer_stg1_ptr: Int64,
            num_rows: Int32,
            row_packs: Int32,
            rank: Int32,
            grid_x: Int32,
            stream: CUstream,
        ):
            self.kernel(
                inp_ptr,
                out_ptr,
                self_sig_ptr,
                peer_sig_ptr,
                self_stg0_ptr,
                self_stg1_ptr,
                peer_stg0_ptr,
                peer_stg1_ptr,
                num_rows,
                row_packs,
                rank,
            ).launch(grid=(grid_x, 1, 1), block=(self.threads, 1, 1), stream=stream)

        @cute.kernel
        def kernel(
            self,
            inp_ptr: Int64,
            out_ptr: Int64,
            self_sig_ptr: Int64,
            peer_sig_ptr: Int64,
            self_stg0_ptr: Int64,
            self_stg1_ptr: Int64,
            peer_stg0_ptr: Int64,
            peer_stg1_ptr: Int64,
            num_rows: Int32,
            row_packs: Int32,
            rank: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            gdim, _, _ = cute.arch.grid_dim()
            bdim = const_expr(self.threads)

            self_sig = Uint64(self_sig_ptr)
            peer_sig = Uint64(peer_sig_ptr)

            # 1. Select call-level double-buffer slot (0 or 1)
            epoch_val = _ld_relaxed_gpu_u32(self_sig)
            slot = epoch_val & Uint32(1)
            cute.arch.sync_threads()
            if tidx == Int32(0):
                if gdim == Int32(1):
                    _st_global_u32(self_sig, epoch_val + Uint32(1))
                else:
                    last = Uint32(gdim - Int32(1))
                    prior = _atom_inc_u32(self_sig + Uint64(4), last)
                    if prior == last:
                        _st_global_u32(self_sig, epoch_val + Uint32(1))

            peer_stg = Uint64(peer_stg0_ptr)
            local_stg = Uint64(self_stg0_ptr)
            if slot != Uint32(0):
                peer_stg = Uint64(peer_stg1_ptr)
                local_stg = Uint64(self_stg1_ptr)

            inp_u64 = Uint64(inp_ptr)
            out_u64 = Uint64(out_ptr)
            total_packs = Int64(num_rows) * Int64(row_packs)
            rank64 = Int64(rank)
            peer64 = Int64(rank ^ Int32(1))
            rp64 = Int64(row_packs)
            c_stride = gdim * Int32(bdim)

            # 2. Push input -> peer_staging[slot] and write local shard -> output
            if const_expr(self.fuse_dim_last):
                r = Int32(0)
                while r < num_rows:
                    r64 = Int64(r)
                    inp_b = inp_u64 + Uint64(r64 * rp64 * Int64(16))
                    stg_b = peer_stg + Uint64(r64 * rp64 * Int64(16))
                    out_b = out_u64 + Uint64((r64 * Int64(2) + rank64) * rp64 * Int64(16))
                    c = bidx * Int32(bdim) + tidx
                    while c < row_packs:
                        off = Uint64(Int64(c) * Int64(16))
                        v0, v1, v2, v3 = _ld_global_nc_u4(inp_b + off)
                        _st_global_wt_u4(stg_b + off, v0, v1, v2, v3)
                        _st_global_u4(out_b + off, v0, v1, v2, v3)
                        c = c + c_stride
                    r = r + Int32(1)
            else:
                out_self_b = out_u64 + Uint64(rank64 * total_packs * Int64(16))
                p = bidx * Int32(bdim) + tidx
                total_i32 = Int32(total_packs)
                while p < total_i32:
                    off = Uint64(Int64(p) * Int64(16))
                    v0, v1, v2, v3 = _ld_global_nc_u4(inp_u64 + off)
                    _st_global_wt_u4(peer_stg + off, v0, v1, v2, v3)
                    _st_global_u4(out_self_b + off, v0, v1, v2, v3)
                    p = p + c_stride

            # 3. Per-CTA peer barrier (system-scope release/acquire)
            cute.arch.sync_threads()
            if tidx == Int32(0):
                cute.arch.fence_acq_rel_sys()
                ctr_addr = self_sig + Uint64(_SELF_COUNTER_OFF + bidx * Int32(4))
                gen = _ld_global_u32(ctr_addr) + Uint32(1)
                _st_global_u32(ctr_addr, gen)
                fslot = Int32(gen & Uint32(1))
                flag_off = Uint64(
                    _FLAG_BASE_OFF
                    + (fslot * Int32(_MAX_BLOCKS) + bidx) * Int32(_FLAG_STRIDE_BYTES)
                )
                _st_release_sys_u32(peer_sig + flag_off, gen)
                local_flag_addr = self_sig + flag_off
                obs = _ld_acquire_sys_u32(local_flag_addr)
                while Int32(obs - gen) < Int32(0):
                    obs = _ld_acquire_sys_u32(local_flag_addr)
            cute.arch.sync_threads()

            # 4. Copy peer shard local_staging[slot] -> output
            if const_expr(self.fuse_dim_last):
                r = Int32(0)
                while r < num_rows:
                    r64 = Int64(r)
                    stg_b = local_stg + Uint64(r64 * rp64 * Int64(16))
                    out_b = out_u64 + Uint64((r64 * Int64(2) + peer64) * rp64 * Int64(16))
                    c = bidx * Int32(bdim) + tidx
                    while c < row_packs:
                        off = Uint64(Int64(c) * Int64(16))
                        v0, v1, v2, v3 = _ld_global_cv_u4(stg_b + off)
                        _st_global_u4(out_b + off, v0, v1, v2, v3)
                        c = c + c_stride
                    r = r + Int32(1)
            else:
                out_peer_b = out_u64 + Uint64(peer64 * total_packs * Int64(16))
                p = bidx * Int32(bdim) + tidx
                total_i32 = Int32(total_packs)
                while p < total_i32:
                    off = Uint64(Int64(p) * Int64(16))
                    v0, v1, v2, v3 = _ld_global_cv_u4(local_stg + off)
                    _st_global_u4(out_peer_b + off, v0, v1, v2, v3)
                    p = p + c_stride


_ALL_KERNEL_KEYS: tuple[tuple[str, int, bool], ...] = (
    ("atomic64", 64, False),
    ("atomic64", 64, True),
    ("atomic64", 128, False),
    ("atomic64", 128, True),
    ("flag128", 256, False),
    ("flag128", 256, True),
)


class _P2pAllGatherKernelCache:
    """Process-wide compiled CuTe-DSL kernel cache + cutedsl_warmup provider."""

    def __init__(self) -> None:
        self._compiled: dict[tuple[str, int, bool], Any] = {}
        self._warmup_registered = False

    def _compile(self, key: tuple[str, int, bool]) -> Any:
        fn = self._compiled.get(key)
        if fn is not None:
            return fn
        kind, threads, fuse = key
        s_ptr = torch.cuda.current_stream().cuda_stream
        if kind == "atomic64":
            obj = CuteTp2Atomic64AllGather(threads=threads, fuse_dim_last=fuse)
            fn = cute.compile(
                obj,
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int32(1),
                Int32(1),
                Int32(0),
                Int32(1),
                CUstream(s_ptr),
                options="--enable-tvm-ffi",
            )
        else:
            obj = CuteTp2Flag128AllGather(threads=threads, fuse_dim_last=fuse)
            fn = cute.compile(
                obj,
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int32(1),
                Int32(1),
                Int32(0),
                Int32(1),
                CUstream(s_ptr),
                options="--enable-tvm-ffi",
            )
        self._compiled[key] = fn
        return fn

    def get(self, key: tuple[str, int, bool]) -> Any:
        fn = self._compiled.get(key)
        if fn is None:
            fn = self._compile(key)
        return fn

    def register_warmup(self) -> None:
        if self._warmup_registered or not _HAS_CUTE:
            return
        from vllm.model_executor.warmup.cutedsl_warmup import (
            register_cutedsl_warmup_provider,
        )

        register_cutedsl_warmup_provider(self)
        self._warmup_registered = True

    def get_cutedsl_warmup_compile_units(self):
        from vllm.model_executor.warmup.cutedsl_warmup import CuTeDSLCompileUnit

        return tuple(
            CuTeDSLCompileUnit(
                name=f"Lebeche TP2 P2P AllGather {k[0]} thr={k[1]} fuse={k[2]}",
                key=("lebeche-tp2-p2p-allgather", *k),
                compile=partial(self._compile, k),
            )
            for k in _ALL_KERNEL_KEYS
        )


_KERNEL_CACHE = _P2pAllGatherKernelCache()


def _get_cudart():
    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    cudart = CudaRTLibrary()
    if not hasattr(cudart, "_lebeche_ipc_close"):
        fn = cudart.lib.cudaIpcCloseMemHandle
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p]
        cudart._lebeche_ipc_close = fn
    return cudart


def _reaches_peer(group: ProcessGroup, device: torch.device) -> bool:
    mine = device.index if device.index is not None else torch.cuda.current_device()
    devices: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(devices, int(mine), group=group)
    try:
        return all(
            torch.cuda.can_device_access_peer(mine, peer)
            for peer in devices
            if peer != mine
        )
    except Exception as err:  # noqa: BLE001
        logger.warning("Lebeche P2P AllGather peer-access probe failed: %s", err)
        return False


def _is_sm8x(device: torch.device) -> bool:
    try:
        major, _ = torch.cuda.get_device_capability(device)
        return major == 8
    except Exception:  # noqa: BLE001
        return False


class LebecheP2pAllGather:
    """TP=2 single-node PCIe P2P CUDA-IPC AllGather communicator."""

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
    ) -> None:
        self.disabled = True
        self.group = group
        self.device = torch.device(device)
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.peer = self.rank ^ 1
        self._ipc_ptrs: list[int] | None = None
        self._cudart: Any = None
        self._self_sig_ptr = 0
        self._peer_sig_ptr = 0
        self._self_tag0_ptr = 0
        self._self_tag1_ptr = 0
        self._peer_tag0_ptr = 0
        self._peer_tag1_ptr = 0
        self._self_flag0_ptr = 0
        self._self_flag1_ptr = 0
        self._peer_flag0_ptr = 0
        self._peer_flag1_ptr = 0

        if not _HAS_CUTE or not current_platform.is_cuda() or self.world_size != 2:
            return

        from vllm.distributed.device_communicators.custom_all_reduce import (
            _all_ranks_true,
        )
        from vllm.distributed.parallel_state import in_the_same_node_as

        if not all(in_the_same_node_as(group, source_rank=0)):
            return
        if not _all_ranks_true(group, _is_sm8x(self.device)):
            return
        if not _all_ranks_true(group, _reaches_peer(group, self.device)):
            return

        # Allocate and exchange one symmetric CUDA IPC slab per rank.
        alloc_ok = True
        local_ptr: ctypes.c_void_p | None = None
        ptrs: list[int] | None = None
        try:
            cudart = _get_cudart()
            with torch.cuda.device(self.device):
                local_ptr = cudart.cudaMalloc(_SLAB_TOTAL_BYTES)
                cudart.cudaMemset(local_ptr, 0, _SLAB_TOTAL_BYTES)
                torch.cuda.synchronize(self.device)
                handle = cudart.cudaIpcGetMemHandle(local_ptr)
            handles: list[Any] = [None, None]
            dist.all_gather_object(handles, bytes(handle.internal), group=group)
            ptrs = [0, 0]
            with torch.cuda.device(self.device):
                for i, h_bytes in enumerate(handles):
                    if i == self.rank:
                        ptrs[i] = int(local_ptr.value)
                    else:
                        peer_h = type(handle)()
                        ctypes.memmove(peer_h.internal, h_bytes, ctypes.sizeof(peer_h.internal))
                        ptrs[i] = int(cudart.cudaIpcOpenMemHandle(peer_h).value)
            self._cudart = cudart
        except Exception as err:  # noqa: BLE001
            logger.warning(
                "Lebeche P2P AllGather IPC init failed (%s); falling back to NCCL.",
                err,
            )
            alloc_ok = False
            if ptrs is not None and self._cudart is not None:
                for i, p in enumerate(ptrs):
                    if i != self.rank and p:
                        try:
                            self._cudart._lebeche_ipc_close(ctypes.c_void_p(p))
                        except Exception:  # noqa: BLE001
                            pass
            if local_ptr is not None and self._cudart is not None:
                try:
                    self._cudart.cudaFree(local_ptr)
                except Exception:  # noqa: BLE001
                    pass
            ptrs = None

        if not _all_ranks_true(group, alloc_ok):
            if ptrs is not None and self._cudart is not None:
                for i, p in enumerate(ptrs):
                    if i != self.rank and p:
                        try:
                            self._cudart._lebeche_ipc_close(ctypes.c_void_p(p))
                        except Exception:  # noqa: BLE001
                            pass
                dist.barrier(group=group)
                if ptrs[self.rank]:
                    try:
                        self._cudart.cudaFree(ctypes.c_void_p(ptrs[self.rank]))
                    except Exception:  # noqa: BLE001
                        pass
            return

        dist.barrier(group=group)
        assert ptrs is not None
        self._ipc_ptrs = ptrs
        self._self_sig_ptr = ptrs[self.rank]
        self._peer_sig_ptr = ptrs[self.peer]
        self._self_tag0_ptr = self._self_sig_ptr + _TAG_STG0_OFF
        self._self_tag1_ptr = self._self_sig_ptr + _TAG_STG1_OFF
        self._peer_tag0_ptr = self._peer_sig_ptr + _TAG_STG0_OFF
        self._peer_tag1_ptr = self._peer_sig_ptr + _TAG_STG1_OFF
        self._self_flag0_ptr = self._self_sig_ptr + _FLAG_STG0_OFF
        self._self_flag1_ptr = self._self_sig_ptr + _FLAG_STG1_OFF
        self._peer_flag0_ptr = self._peer_sig_ptr + _FLAG_STG0_OFF
        self._peer_flag1_ptr = self._peer_sig_ptr + _FLAG_STG1_OFF
        self.disabled = False

        _KERNEL_CACHE.register_warmup()
        logger.info_once(
            "Enabled Lebeche TP=2 PCIe P2P IPC AllGather (slab=%.2f MiB/rank, "
            "atomic64<=%d KiB, flag128<=%d MiB).",
            _SLAB_TOTAL_BYTES / (1024 * 1024),
            _SMALL_MAX_BYTES // 1024,
            _LARGE_MAX_BYTES // (1024 * 1024),
        )

    def should_use(self, input_: torch.Tensor, dim: int) -> bool:
        """Pure collective-safe predicate (depends ONLY on shape/dtype/dim)."""
        if self.disabled or not input_.is_cuda:
            return False
        numel = input_.numel()
        if numel <= 0:
            return False
        nbytes = numel * input_.element_size()
        if nbytes <= _SMALL_MAX_BYTES:
            return (nbytes & 3) == 0
        if nbytes <= _LARGE_MAX_BYTES:
            return (nbytes & 15) == 0
        return False

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            dim += input_.dim()
        inp_shape = input_.size()
        ndim = input_.dim()
        elem_size = input_.element_size()
        nbytes = input_.numel() * elem_size
        out_shape = (
            inp_shape[:dim] + (2 * inp_shape[dim],) + inp_shape[dim + 1 :]
        )

        outer = 1
        for i in range(ndim - 1):
            outer *= inp_shape[i]
        row_bytes = inp_shape[-1] * elem_size
        stream_val = torch.cuda.current_stream(self.device).cuda_stream

        if nbytes <= _SMALL_MAX_BYTES:
            # Ensure contiguous & 4-byte aligned (purely rank-local fixup;
            # never diverges collective control flow).
            inp = input_.contiguous()
            if (inp.data_ptr() & 3) != 0:
                inp = inp.clone(memory_format=torch.contiguous_format)
            total_words = nbytes >> 2
            threads = 64 if nbytes <= 12288 else 128
            grid_x = min(_MAX_BLOCKS, (total_words + threads - 1) // threads)
            if grid_x < 1:
                grid_x = 1

            direct_flat = (dim == 0) or (outer == 1 and dim == ndim - 1)
            fuse_last = (
                (not direct_flat)
                and (dim == ndim - 1)
                and ((row_bytes & 3) == 0)
            )
            if direct_flat or fuse_last:
                out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
                fn = _KERNEL_CACHE.get(("atomic64", threads, fuse_last))
                row_words = (row_bytes >> 2) if fuse_last else total_words
                fn(
                    inp.data_ptr(),
                    out.data_ptr(),
                    self._self_sig_ptr,
                    self._self_tag0_ptr,
                    self._self_tag1_ptr,
                    self._peer_tag0_ptr,
                    self._peer_tag1_ptr,
                    total_words,
                    row_words,
                    self.rank,
                    grid_x,
                    CUstream(stream_val),
                )
                return out

            tmp = torch.empty((2,) + inp_shape, dtype=inp.dtype, device=inp.device)
            fn = _KERNEL_CACHE.get(("atomic64", threads, False))
            fn(
                inp.data_ptr(),
                tmp.data_ptr(),
                self._self_sig_ptr,
                self._self_tag0_ptr,
                self._self_tag1_ptr,
                self._peer_tag0_ptr,
                self._peer_tag1_ptr,
                total_words,
                total_words,
                self.rank,
                grid_x,
                CUstream(stream_val),
            )
            return tmp.movedim(0, dim).reshape(out_shape).contiguous()

        # Medium / large path: Flag128 (64 KiB < nbytes <= 8 MiB)
        inp = input_.contiguous()
        if (inp.data_ptr() & 15) != 0:
            inp = inp.clone(memory_format=torch.contiguous_format)
        total_packs = nbytes >> 4
        threads = 256
        row_packs = row_bytes >> 4

        direct_flat = (dim == 0) or (outer == 1 and dim == ndim - 1)
        # Use zero-division fused row-loop whenever each row is wide enough to
        # saturate >= 16 CTAs (e.g. vocab logits row_packs=15520), avoiding the
        # post-gather movedim+reshape copy on c1/c4/c8 batches.
        fuse_last = (
            (not direct_flat)
            and (dim == ndim - 1)
            and ((row_bytes & 15) == 0)
            and (row_packs >= 16 * threads)
        )
        if nbytes <= (1536 * 1024):
            grid_x = 16
        elif fuse_last:
            grid_x = min(_MAX_BLOCKS, max(16, (row_packs + threads - 1) // threads))
        else:
            grid_x = 64
        if direct_flat or fuse_last:
            out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
            fn = _KERNEL_CACHE.get(("flag128", threads, fuse_last))
            n_rows_arg = outer if fuse_last else 1
            r_packs_arg = row_packs if fuse_last else total_packs
            fn(
                inp.data_ptr(),
                out.data_ptr(),
                self._self_sig_ptr,
                self._peer_sig_ptr,
                self._self_flag0_ptr,
                self._self_flag1_ptr,
                self._peer_flag0_ptr,
                self._peer_flag1_ptr,
                n_rows_arg,
                r_packs_arg,
                self.rank,
                grid_x,
                CUstream(stream_val),
            )
            return out

        tmp = torch.empty((2,) + inp_shape, dtype=inp.dtype, device=inp.device)
        fn = _KERNEL_CACHE.get(("flag128", threads, False))
        fn(
            inp.data_ptr(),
            tmp.data_ptr(),
            self._self_sig_ptr,
            self._peer_sig_ptr,
            self._self_flag0_ptr,
            self._self_flag1_ptr,
            self._peer_flag0_ptr,
            self._peer_flag1_ptr,
            1,
            total_packs,
            self.rank,
            grid_x,
            CUstream(stream_val),
        )
        return tmp.movedim(0, dim).reshape(out_shape).contiguous()

    @contextmanager
    def capture(self):
        """Synchronize around CUDA graph capture stream transitions."""
        if self.disabled:
            yield
            return
        torch.accelerator.synchronize(self.device)
        try:
            yield
        finally:
            torch.accelerator.synchronize(self.device)

    def destroy(self) -> None:
        ptrs = self._ipc_ptrs
        self._ipc_ptrs = None
        self.disabled = True
        if ptrs is None or self._cudart is None:
            return
        try:
            torch.cuda.synchronize(self.device)
        except Exception:  # noqa: BLE001
            pass
        for i, p in enumerate(ptrs):
            if i != self.rank and p:
                try:
                    self._cudart._lebeche_ipc_close(ctypes.c_void_p(p))
                except Exception:  # noqa: BLE001
                    pass
        if dist.is_initialized():
            try:
                dist.barrier(group=self.group)
            except Exception:  # noqa: BLE001
                pass
        if ptrs[self.rank]:
            try:
                self._cudart.cudaFree(ctypes.c_void_p(ptrs[self.rank]))
            except Exception:  # noqa: BLE001
                pass
        if dist.is_initialized():
            try:
                dist.barrier(group=self.group)
            except Exception:  # noqa: BLE001
                pass

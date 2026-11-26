# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 lebeche hill-climb (fork of Merlin 3.0.2.2)
"""Small-M (M <= 8) W8 channelwise dense Marlin replacement on sm_8x (lebeche fork).

Dispatches inside the opaque ``vllm::marlin_gemm`` custom op at run time, so
dynamic-shape ``torch.compile`` traces with dropped guards pick the fast path
at M <= 8 and fall back to stock Marlin at M > 8 (and whenever any gate fails).

``MERLIN_LEBECHE_W8_SMALL_M`` (default "1"; "0" = stock Marlin) is read once at
import and is a hashed compile factor. Inert on non-sm_8x devices.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import torch

from vllm.merlin.envs import read_registered
from vllm.scalar_type import scalar_types

SM8X_W8_CAPABILITIES: frozenset[tuple[int, int]] = frozenset({(8, 0), (8, 6), (8, 9)})
W8_SMALL_M_SWITCH: bool = bool(read_registered("MERLIN_LEBECHE_W8_SMALL_M"))
MAX_W8_SMALL_M: int = 8
_UINT8B128_ID: int = scalar_types.uint8b128.id

_active: bool | None = None


def w8_small_m_active() -> bool:
    """Switch on and current CUDA device is sm_80/86/89."""
    global _active
    active = _active
    if active is None:
        active = False
        if W8_SMALL_M_SWITCH:
            from vllm.platforms import current_platform

            if current_platform.is_cuda():
                cap = current_platform.get_device_capability()
                if cap is None:
                    return False
                active = (cap.major, cap.minor) in SM8X_W8_CAPABILITIES
        if not torch.compiler.is_compiling():
            _active = active
    return active


@dataclass(frozen=True)
class W8SmallMGemmConfig:
    max_m: int
    cols_per_block: int
    block_size: int
    k_unroll: int = 2
    static_k: int | None = None
    pingpong: bool = True
    contig_k: bool = False
    maxrregcount: int = 64


# Measured on RTX 3090 (sm_86, 82 SMs, 1500 MHz cap, 24 distinct cold-L2 weights).
# Each (N, K) maps to a tuple of configs ordered by ascending max_m.
_SM86_PLANS: dict[tuple[int, int], tuple[W8SmallMGemmConfig, ...]] = {
    # Flash-Next GDN in_proj_qkvz (TP=2): 25.93-25.97 us at M=1..4 (~808 GB/s), 26.84 us at M=8
    (8192, 2560): (
        W8SmallMGemmConfig(
            max_m=4,
            cols_per_block=32,
            block_size=160,
            k_unroll=1,
            static_k=2560,
            pingpong=True,
            contig_k=False,
            maxrregcount=64,
        ),
        W8SmallMGemmConfig(
            max_m=8,
            cols_per_block=32,
            block_size=128,
            k_unroll=2,
            static_k=None,
            pingpong=False,
            contig_k=False,
            maxrregcount=64,
        ),
    ),
    # Flash-Next GDN out_proj (TP=2): 11.09-11.31 us at M=1..4 (~695-709 GB/s), 11.72 us at M=8
    (2560, 3072): (
        W8SmallMGemmConfig(
            max_m=4,
            cols_per_block=64,
            block_size=768,
            k_unroll=2,
            static_k=3072,
            pingpong=True,
            contig_k=False,
            maxrregcount=64,
        ),
        W8SmallMGemmConfig(
            max_m=8,
            cols_per_block=64,
            block_size=384,
            k_unroll=1,
            static_k=3072,
            pingpong=True,
            contig_k=False,
            maxrregcount=64,
        ),
    ),
}

_GENERIC_CONFIG = W8SmallMGemmConfig(
    max_m=MAX_W8_SMALL_M,
    cols_per_block=64,
    block_size=256,
    k_unroll=2,
    static_k=None,
    pingpong=False,
    contig_k=False,
    maxrregcount=64,
)


def configs_for_shape(size_n: int, size_k: int) -> tuple[W8SmallMGemmConfig, ...]:
    plan = _SM86_PLANS.get((size_n, size_k))
    if plan is not None:
        return plan
    return (_GENERIC_CONFIG,)


def select_config(size_m: int, size_n: int, size_k: int) -> W8SmallMGemmConfig | None:
    for cfg in configs_for_shape(size_n, size_k):
        if size_m <= cfg.max_m:
            return cfg
    return None


class W8SmallMGemmRunner:
    """Compiled kernel cache and CuTeDSL warmup provider."""

    def __init__(self) -> None:
        self._compiled: dict[W8SmallMGemmConfig, Any] = {}
        self._warmup_configs: set[W8SmallMGemmConfig] = set()
        self._warmup_registered = False

    def _compile(self, cfg: W8SmallMGemmConfig) -> Any:
        fn = self._compiled.get(cfg)
        if fn is not None:
            return fn

        import cutlass
        import cutlass.cute as cute
        from cuda.bindings.driver import CUstream
        from quack.compile_utils import make_fake_tensor

        from vllm.model_executor.kernels.linear.cute_dsl._w8_skinny_gemm import (
            CuteW8MarlinGemm,
        )

        m_sym = cute.sym_int(divisibility=1)
        n_sym = cute.sym_int(divisibility=64)
        k_tiles_sym = (
            cfg.static_k // 16 if cfg.static_k is not None else cute.sym_int(divisibility=1)
        )
        k_half_sym = (
            cfg.static_k // 2 if cfg.static_k is not None else cute.sym_int(divisibility=8)
        )
        n4_sym = cute.sym_int(divisibility=256)

        a_fake = make_fake_tensor(cutlass.Int32, (m_sym, k_half_sym), divisibility=4)
        b_fake = make_fake_tensor(cutlass.Int32, (k_tiles_sym, n4_sym), divisibility=4)
        s_fake = make_fake_tensor(cutlass.BFloat16, (n_sym,), divisibility=1)
        c_fake = make_fake_tensor(cutlass.BFloat16, (m_sym, n_sym), divisibility=1)

        kernel = CuteW8MarlinGemm(
            max_m=cfg.max_m,
            cols_per_block=cfg.cols_per_block,
            block_size=cfg.block_size,
            k_unroll=cfg.k_unroll,
            static_k=cfg.static_k,
            pingpong=cfg.pingpong,
            contig_k=cfg.contig_k,
        )
        stream = CUstream(torch.cuda.current_stream().cuda_stream)
        opts = (
            f"--enable-tvm-ffi --ptxas-options -maxrregcount={cfg.maxrregcount}"
            if cfg.maxrregcount
            else "--enable-tvm-ffi"
        )
        fn = cute.compile(kernel, a_fake, b_fake, s_fake, c_fake, stream, options=opts)
        self._compiled[cfg] = fn
        return fn

    def request_warmup(self, size_n: int, size_k: int) -> None:
        if not w8_small_m_active():
            return
        if size_n <= 0 or size_k <= 0 or size_n % 64 != 0 or size_k % 128 != 0:
            return
        self._warmup_configs.update(configs_for_shape(size_n, size_k))
        if self._warmup_registered:
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
                name="lebeche sm_8x W8 channelwise small-M GEMM",
                key=("lebeche-w8-small-m", cfg),
                compile=partial(self._compile, cfg),
            )
            for cfg in sorted(
                self._warmup_configs,
                key=lambda c: (
                    c.static_k or 0,
                    c.max_m,
                    c.cols_per_block,
                    c.block_size,
                    c.k_unroll,
                ),
            )
        )

    def run(
        self,
        a: torch.Tensor,
        b_q_weight: torch.Tensor,
        b_scales: torch.Tensor,
        size_m: int,
        size_n: int,
        size_k: int,
    ) -> torch.Tensor | None:
        cfg = select_config(size_m, size_n, size_k)
        if cfg is None:
            return None
        fn = self._compiled.get(cfg)
        if fn is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            fn = self._compile(cfg)
        from cuda.bindings.driver import CUstream

        out = torch.empty((size_m, size_n), device=a.device, dtype=torch.bfloat16)
        stream = CUstream(torch.cuda.current_stream(a.device).cuda_stream)
        fn(a.view(torch.int32), b_q_weight, b_scales.view(-1), out, stream)
        return out


w8_small_m_gemm = W8SmallMGemmRunner()


def request_w8_small_m_warmup(size_n: int, size_k: int) -> None:
    w8_small_m_gemm.request_warmup(size_n, size_k)


def maybe_w8_small_m_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_bias: torch.Tensor | None,
    b_scales: torch.Tensor,
    a_scales: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    b_zeros: torch.Tensor | None,
    b_q_type_id: int,
    size_m: int,
    size_n: int,
    size_k: int,
    is_zp_float: bool,
) -> torch.Tensor | None:
    """Fast path for sm_8x small-M W8 channelwise dense Marlin GEMMs.

    Returns a fresh contiguous ``[size_m, size_n]`` BF16 tensor on hit, or
    ``None`` to fall back to stock ``torch.ops._C.marlin_gemm``.
    """
    if (
        size_m < 1
        or size_m > MAX_W8_SMALL_M
        or b_q_type_id != _UINT8B128_ID
        or not w8_small_m_active()
        or b_bias is not None
        or a_scales is not None
        or global_scale is not None
        or is_zp_float
        or (b_zeros is not None and b_zeros.numel() != 0)
        or a.device.type != "cuda"
        or a.dtype != torch.bfloat16
        or b_q_weight.dtype != torch.int32
        or b_scales.dtype != torch.bfloat16
        or b_scales.ndim != 2
        or b_scales.shape[0] != 1
        or b_scales.shape[1] != size_n
        or size_n <= 0
        or size_k <= 0
        or (size_n & 63) != 0
        or (size_k & 127) != 0
        or a.ndim != 2
        or a.shape[0] != size_m
        or a.shape[1] != size_k
        or b_q_weight.ndim != 2
        or b_q_weight.shape[0] != (size_k >> 4)
        or b_q_weight.shape[1] != (size_n << 2)
        or not a.is_contiguous()
        or not b_q_weight.is_contiguous()
        or not b_scales.is_contiguous()
        or (a.data_ptr() & 15) != 0
        or (b_q_weight.data_ptr() & 15) != 0
    ):
        return None
    return w8_small_m_gemm.run(a, b_q_weight, b_scales, size_m, size_n, size_k)


__all__ = [
    "MAX_W8_SMALL_M",
    "W8_SMALL_M_SWITCH",
    "W8SmallMGemmConfig",
    "maybe_w8_small_m_gemm",
    "request_w8_small_m_warmup",
    "w8_small_m_active",
    "w8_small_m_gemm",
]

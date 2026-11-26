# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 lebeche hill-climb (fork of Merlin 3.0.2.2)
"""Small-M fused HyperConnection GEMMs for sm_80/86/89 (lebeche fork).

At decode token counts a ``GatedResidual`` runs five kernels: combine+norm,
skinny down GEMM, ``hc_silu``, skinny up GEMM, ``hc_gate_mix``. This module
folds the two glue kernels into the GEMMs as epilogues
(ops/cute_dsl/_hc_small_fused.py):

* ``qwen4_exp_hc_down_silu``    down(+inject) GEMM -> SiLU, logits passed through
* ``qwen4_exp_hc_up_gate_mix``  up GEMM -> sigmoid gate -> mean over HC streams

Both are opaque custom ops, so the token-count dispatch happens at run time
(the model forward is traced once with a dynamic batch dim and no guards; a
Python ``if M <= 4`` in ``GatedResidual`` would be frozen at trace time). For
token counts without a plan, or inputs the kernels cannot take, the ops run the
unfused sequence of commit b5519de (``qwen4_exp_low_latency_gemm`` followed by
the Triton glue kernel), bit-identically.

The plans reuse ``(block_size, vector_width)`` of the skinny plans of the same
weight shapes, which fixes the FP32 accumulation order, and the epilogues keep
the BF16 rounding points and the operation order of the Triton kernels: the
fused results are bit-identical to the unfused skinny sequence.

``MERLIN_LEBECHE_HC_FUSE`` (default "1"; "0" = behaviour of commit b5519de) is
read once at import and is a hashed compile factor. It needs
``MERLIN_LEBECHE_SM86_LL`` on and an sm_80/86/89 device; inert otherwise.
"""

from collections.abc import Iterable
from functools import partial
from typing import Any

import torch
from torch import nn

import vllm.envs as envs
from vllm.merlin.envs import read_registered
from vllm.merlin.lebeche_sm86 import sm86_ll_active
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    row_stride_ok,
    shape_dynamic_skinny_gemm,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .low_latency_gemm import (
    Qwen4ExpLowLatencyLinearMethod,
    _is_packed_row_major,
    _qwen4_exp_low_latency_gemm,
)
from .ops.hc import hc_combine_norm, hc_gate_mix, hc_silu

HC_FUSE_SWITCH: bool = bool(read_registered("MERLIN_LEBECHE_HC_FUSE"))
HC_SM86_TUNE_SWITCH: bool = bool(read_registered("MERLIN_LEBECHE_HC_SM86_TUNE"))

_DOWN = "down_silu"
_UP = "up_gate_mix"
_WARP = 32

# RTX 3090 plans, CUDA graph replay (24 distinct weights, best of 40). Keyed by
# the weight shape (N, K) and the token count. block_size / vector_width / K
# mode equal the skinny plan of the same shape (QWEN4_EXP_SM86_GEMM_PLANS), so
# the GEMM values are unchanged; only outputs_per_block is free (swept: the
# skinny plan's value stays best for the down shapes). us per call, unfused
# pair -> fused: down+SiLU 10.9/11.0/11.8/13.0 -> 9.5/9.6/10.3/11.5 and
# up+gate-mix 10.9/11.0/11.2/11.8 -> 9.7/9.9/10.2/10.5 at M = 1/2/3/4; a whole
# combine_and_mix 24.3/24.6/25.5/27.3 -> 21.7/22.1/23.0/24.2.
HC_DOWN_SILU_SM86_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
    # Merged HC down+inject projection (320 + 4 logits, padded to 336 rows).
    (336, 10240): {
        1: SkinnyGemmConfig(1, 64, 1, static_k=10240),
        2: SkinnyGemmConfig(2, 64, 1, static_k=10240),
        3: SkinnyGemmConfig(3, 160, 2, static_k=10240),
        4: SkinnyGemmConfig(
            4, 128, 1 if HC_SM86_TUNE_SWITCH else 2, static_k=10240
        ),
    },
    # Final HC mixer down projection.
    (320, 10240): {
        1: SkinnyGemmConfig(1, 64, 2, vector_width=2, static_k=10240),
        2: SkinnyGemmConfig(2, 160, 1, static_k=10240),
        3: SkinnyGemmConfig(3, 256, 2, k_unroll=2),
        4: SkinnyGemmConfig(
            4, 128, 1 if HC_SM86_TUNE_SWITCH else 2, static_k=10240
        ),
    },
}
# outputs_per_block counts weight rows: hc streams x inner indices per block.
HC_UP_GATE_MIX_SM86_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
    (10240, 320): {
        1: SkinnyGemmConfig(1, 32, 4, vector_width=2, static_k=320),
        2: SkinnyGemmConfig(2, 32, 4, vector_width=2, static_k=320),
        3: SkinnyGemmConfig(3, 32, 4, vector_width=2, static_k=320),
        4: SkinnyGemmConfig(4, 32, 4, vector_width=2, static_k=320),
    },
}

# (epilogue, config, rank, hc)
_CompileKey = tuple[str, SkinnyGemmConfig, int, int]


def _config_fits(config: SkinnyGemmConfig, n: int, k: int) -> bool:
    """Static (shape-only) validity of a plan entry for N computed rows."""
    return (
        n % config.outputs_per_block == 0
        and config.num_rows * config.outputs_per_block <= _WARP
        and k % (config.block_size * config.vector_width) == 0
        and (config.static_k is None or config.static_k == k)
    )


class HcSmallFused:
    """Compile cache and CuTe-DSL warmup provider of the fused HC kernels."""

    def __init__(self) -> None:
        self._compiled: dict[_CompileKey, Any] = {}
        self._warmup: set[_CompileKey] = set()
        self._warmup_registered = False

    def _compile(self, key: _CompileKey) -> None:
        if key in self._compiled:
            return
        import cutlass.cute as cute
        from cutlass import BFloat16
        from quack.compile_utils import make_fake_tensor

        from .ops.cute_dsl._hc_small_fused import CuteHcSmallFused

        epilogue, config, rank, hc = key
        m = config.num_rows
        n = cute.sym_int(divisibility=config.outputs_per_block)
        k = (
            config.static_k
            if config.static_k is not None
            else cute.sym_int(divisibility=config.block_size * config.vector_width)
        )
        a = make_fake_tensor(BFloat16, (m, k), divisibility=config.vector_width)
        b = make_fake_tensor(BFloat16, (n, k), divisibility=config.vector_width)
        x = make_fake_tensor(BFloat16, (m, cute.sym_int()), divisibility=1)
        c = make_fake_tensor(BFloat16, (m, cute.sym_int()), divisibility=1)
        logits = make_fake_tensor(BFloat16, (m, cute.sym_int()), divisibility=1)
        kernel = CuteHcSmallFused(
            element_type=BFloat16,
            epilogue=epilogue,
            num_rows=m,
            block_size=config.block_size,
            outputs_per_block=config.outputs_per_block,
            hc=hc,
            rank=rank,
            vector_width=config.vector_width,
            k_unroll=config.k_unroll,
            use_pdl=shape_dynamic_skinny_gemm._use_pdl(),
            static_k=config.static_k,
        )
        self._compiled[key] = cute.compile(
            kernel,
            a,
            b,
            x,
            c,
            logits,
            shape_dynamic_skinny_gemm._stream(),
            options="--enable-tvm-ffi --ptxas-options -maxrregcount=64",
        )

    def request_warmup(self, keys: Iterable[_CompileKey]) -> None:
        """Request compilation before CUDA graph capture."""
        self._warmup.update(keys)
        if self._warmup_registered or not self._warmup:
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
                name=f"Qwen4Exp HC small-M fused {key[0]} GEMM",
                key=("qwen4-exp-hc-small-fused", *key),
                compile=partial(self._compile, key),
            )
            for key in sorted(
                self._warmup,
                key=lambda item: (
                    item[0],
                    item[1].num_rows,
                    item[1].block_size,
                    item[1].outputs_per_block,
                    item[1].k_unroll,
                    item[1].vector_width,
                    item[1].static_k or 0,
                    item[2],
                    item[3],
                ),
            )
        )

    def down_silu(
        self,
        x: torch.Tensor,  # [M, K]
        weight: torch.Tensor,  # [N, K], N >= rank + n_logits
        rank: int,
        n_logits: int,
        hc: int,
        config: SkinnyGemmConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (_DOWN, config, rank, hc)
        if key not in self._compiled:
            self._compile(key)
        m = x.shape[0]
        lowrank = torch.empty((m, rank), dtype=x.dtype, device=x.device)
        # Same layout as the unfused GEMM output; only the logit columns are
        # written (the SiLU columns live in lowrank, the pad rows of the merged
        # weight are never computed).
        down = torch.empty((m, weight.shape[0]), dtype=x.dtype, device=x.device)
        self._compiled[key](
            x,
            weight[: rank + n_logits],
            lowrank,
            lowrank,
            down[:, rank : rank + n_logits] if n_logits else lowrank,
            shape_dynamic_skinny_gemm._stream(),
        )
        return lowrank, down

    def up_gate_mix(
        self,
        lowrank: torch.Tensor,  # [M, K]
        weight: torch.Tensor,  # [hc * hc_dim, K], stream-major rows
        xn: torch.Tensor,  # [M, hc * hc_dim]
        hc: int,
        config: SkinnyGemmConfig,
    ) -> torch.Tensor:
        key = (_UP, config, 0, hc)
        if key not in self._compiled:
            self._compile(key)
        out = torch.empty(
            (lowrank.shape[0], weight.shape[0] // hc),
            dtype=lowrank.dtype,
            device=lowrank.device,
        )
        self._compiled[key](
            lowrank, weight, xn, out, out, shape_dynamic_skinny_gemm._stream()
        )
        return out


hc_small_fused = HcSmallFused()


def _plans(
    table: dict[tuple[int, int], dict[int, SkinnyGemmConfig]],
) -> dict[tuple[int, int], dict[int, SkinnyGemmConfig]]:
    return table if HC_FUSE_SWITCH and sm86_ll_active() else {}


def _gemm_ok(x: torch.Tensor, weight: torch.Tensor, config: SkinnyGemmConfig) -> bool:
    return (
        not envs.VLLM_BATCH_INVARIANT
        and x.dim() == 2
        and row_stride_ok(x, config)
        and _is_packed_row_major(weight)
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_cuda
        and x.device == weight.device
        and x.shape[1] == weight.shape[1]
    )


def _qwen4_exp_hc_down_silu(
    x: torch.Tensor, weight: torch.Tensor, rank: int, n_logits: int, hc: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``silu((x @ weight[:rank].T) / hc)`` and a GEMM-shaped ``[M, N]`` buffer.

    Only columns ``[rank, rank + n_logits)`` of the second output are defined
    on every path (the caller slices them in the traced graph, like the
    ``split`` of the unfused module). Both outputs are freshly allocated and
    contiguous on every path: compiled backends assert the fake impl's strides.
    """
    plan = _plans(HC_DOWN_SILU_SM86_PLANS).get((weight.shape[0], weight.shape[1]))
    config = None if plan is None else plan.get(x.shape[0])
    if (
        config is not None
        and _gemm_ok(x, weight, config)
        and _config_fits(config, rank + n_logits, weight.shape[1])
    ):
        return hc_small_fused.down_silu(x, weight, rank, n_logits, hc, config)
    # Unfused sequence of b5519de: Linear (low-latency op) -> split -> hc_silu.
    out = _qwen4_exp_low_latency_gemm(x, weight)
    return hc_silu(out[..., :rank], hc), out


def _qwen4_exp_hc_down_silu_fake(
    x: torch.Tensor, weight: torch.Tensor, rank: int, n_logits: int, hc: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        x.new_empty((*x.shape[:-1], rank)),
        x.new_empty((*x.shape[:-1], weight.shape[0])),
    )


def _qwen4_exp_hc_up_gate_mix(
    lowrank: torch.Tensor, weight: torch.Tensor, xn: torch.Tensor, hc: int
) -> torch.Tensor:
    """``mean_s(sigmoid(lowrank @ weight.T)_s * xn_s)`` over the hc streams."""
    plan = _plans(HC_UP_GATE_MIX_SM86_PLANS).get((weight.shape[0], weight.shape[1]))
    config = None if plan is None else plan.get(lowrank.shape[0])
    if (
        config is not None
        and _gemm_ok(lowrank, weight, config)
        and _config_fits(config, weight.shape[0], weight.shape[1])
        and config.outputs_per_block % hc == 0
        and xn.shape == (lowrank.shape[0], weight.shape[0])
        and xn.stride(1) == 1
        and xn.dtype == torch.bfloat16
        and xn.device == weight.device
    ):
        return hc_small_fused.up_gate_mix(lowrank, weight, xn, hc, config)
    # Unfused sequence of b5519de: Linear (low-latency op) -> hc_gate_mix.
    return hc_gate_mix(xn, _qwen4_exp_low_latency_gemm(lowrank, weight), hc)


def _qwen4_exp_hc_up_gate_mix_fake(
    lowrank: torch.Tensor, weight: torch.Tensor, xn: torch.Tensor, hc: int
) -> torch.Tensor:
    return xn.new_empty((*xn.shape[:-1], weight.shape[0] // hc))


direct_register_custom_op(
    op_name="qwen4_exp_hc_down_silu",
    op_func=_qwen4_exp_hc_down_silu,
    fake_impl=_qwen4_exp_hc_down_silu_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_hc_up_gate_mix",
    op_func=_qwen4_exp_hc_up_gate_mix,
    fake_impl=_qwen4_exp_hc_up_gate_mix_fake,
)


_CombineNormKey = tuple[bool, int, int, float]


class HcCombineNormCute:
    """Compile cache and CuTe-DSL warmup provider for dynamic-M combine+norm."""

    def __init__(self) -> None:
        self._compiled: dict[_CombineNormKey, Any] = {}
        self._warmup: set[_CombineNormKey] = set()
        self._warmup_registered = False

    def _compile(self, key: _CombineNormKey) -> None:
        if key in self._compiled:
            return
        import cutlass.cute as cute
        from cutlass import BFloat16
        from quack.compile_utils import make_fake_tensor

        from .ops.cute_dsl._hc_combine_norm import CuteHcCombineNorm

        has_inj, hc, hc_dim, eps = key
        k = hc * hc_dim
        m_sym = cute.sym_int()
        kernel = CuteHcCombineNorm(
            element_type=BFloat16,
            has_inj=has_inj,
            warps_per_stream=10,
            hc=hc,
            hc_dim=hc_dim,
            eps=eps,
        )
        self._compiled[key] = cute.compile(
            kernel,
            make_fake_tensor(BFloat16, (m_sym, k), divisibility=8),
            make_fake_tensor(BFloat16, (m_sym, hc_dim), divisibility=8),
            make_fake_tensor(BFloat16, (m_sym, hc), divisibility=1),
            make_fake_tensor(BFloat16, (1, k), divisibility=8),
            make_fake_tensor(BFloat16, (m_sym, k), divisibility=8),
            make_fake_tensor(BFloat16, (m_sym, k), divisibility=8),
            shape_dynamic_skinny_gemm._stream(),
            options="--enable-tvm-ffi",
        )

    def request_warmup(self, keys: Iterable[_CombineNormKey]) -> None:
        self._warmup.update(keys)
        if self._warmup_registered or not self._warmup:
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
                name="Qwen4Exp HC dynamic-M combine+norm",
                key=("qwen4-exp-hc-combine-norm", *key),
                compile=partial(self._compile, key),
            )
            for key in sorted(self._warmup)
        )

    def __call__(
        self,
        residual: torch.Tensor,
        block_output: torch.Tensor,
        injection_logits: torch.Tensor | None,
        norm_weight: torch.Tensor,
        eps: float,
        hc_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hc_dim = residual.shape[1] // hc_count
        has_inj = injection_logits is not None
        key: _CombineNormKey = (has_inj, hc_count, hc_dim, float(eps))
        if key not in self._compiled:
            self._compile(key)
        out = residual.new_empty(residual.shape)
        xn = residual.new_empty(residual.shape)
        inj_arg = (
            injection_logits
            if injection_logits is not None
            else residual[:, :hc_count]
        )
        self._compiled[key](
            residual,
            block_output,
            inj_arg,
            norm_weight.view(1, -1),
            out,
            xn,
            shape_dynamic_skinny_gemm._stream(),
        )
        return out, xn


hc_combine_norm_cute = HcCombineNormCute()


def _combine_norm_cute_ok(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    norm_weight: torch.Tensor,
    hc_count: int,
) -> bool:
    if (
        not HC_SM86_TUNE_SWITCH
        or not sm86_ll_active()
        or envs.VLLM_BATCH_INVARIANT
        or not shape_dynamic_skinny_gemm.is_available()
    ):
        return False
    if (
        residual.dim() != 2
        or block_output.dim() != 2
        or residual.shape[0] < 1
        or hc_count != 4
        or residual.shape[1] != 10240
        or block_output.shape != (residual.shape[0], 2560)
        or norm_weight.numel() != 10240
        or residual.dtype != torch.bfloat16
        or block_output.dtype != torch.bfloat16
        or norm_weight.dtype != torch.bfloat16
        or not residual.is_cuda
        or block_output.device != residual.device
        or norm_weight.device != residual.device
        or not residual.is_contiguous()
        or not block_output.is_contiguous()
        or not norm_weight.is_contiguous()
        or (residual.data_ptr() % 16) != 0
        or (block_output.data_ptr() % 16) != 0
        or (norm_weight.data_ptr() % 16) != 0
    ):
        return False
    if injection_logits is not None and (
        injection_logits.dim() != 2
        or injection_logits.shape != (residual.shape[0], hc_count)
        or injection_logits.dtype != torch.bfloat16
        or injection_logits.device != residual.device
        or injection_logits.stride(1) != 1
    ):
        return False
    return True


def _qwen4_exp_hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _combine_norm_cute_ok(
        residual, block_output, injection_logits, norm_weight, hc_count
    ):
        return hc_combine_norm_cute(
            residual, block_output, injection_logits, norm_weight, eps, hc_count
        )
    return hc_combine_norm(
        residual, block_output, injection_logits, norm_weight, eps, hc_count
    )


def _qwen4_exp_hc_combine_norm_fake(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        residual.new_empty(residual.shape),
        residual.new_empty(residual.shape),
    )


direct_register_custom_op(
    op_name="qwen4_exp_hc_combine_norm",
    op_func=_qwen4_exp_hc_combine_norm,
    fake_impl=_qwen4_exp_hc_combine_norm_fake,
)


def _patched(linear: nn.Module | None) -> bool:
    """The Linear already dispatches through ``qwen4_exp_low_latency_gemm``."""
    return (
        linear is not None
        and type(getattr(linear, "quant_method", None))
        is Qwen4ExpLowLatencyLinearMethod
        and getattr(linear, "bias", None) is None
        and getattr(linear, "weight", None) is not None
        and linear.weight.dim() == 2
    )


def enable_qwen4_exp_hc_small_fused(module: nn.Module, dtype: torch.dtype) -> None:
    """Route planned ``GatedResidual`` projections through the fused ops.

    Called by ``enable_qwen4_exp_low_latency_gemm`` after it patched the Linear
    modules: a projection is fused only if it already uses the low-latency op,
    so the ops' run-time fallback equals what the module did before.
    """
    if dtype != torch.bfloat16 or (not HC_FUSE_SWITCH and not HC_SM86_TUNE_SWITCH) or not sm86_ll_active():
        return
    if not shape_dynamic_skinny_gemm.is_available():
        return
    from .hyperconnection import GatedResidual

    warmup: set[_CompileKey] = set()
    cn_warmup: set[_CombineNormKey] = set()
    for child in module.modules():
        if not isinstance(child, GatedResidual):
            continue
        if (
            HC_SM86_TUNE_SWITCH
            and child.hc_count == 4
            and child.hidden_size == 2560
            and child.hc_norm.weight.numel() == 10240
        ):
            child._hc_fast_combine_norm = True
            eps = float(child.config.rms_norm_eps)
            cn_warmup.add((True, 4, 2560, eps))
            cn_warmup.add((False, 4, 2560, eps))
        if not HC_FUSE_SWITCH:
            continue
        rank, hc = child.hc_lowrank, child.hc_count
        if child.use_combine:
            down = child.input_mix_weight_down_block_inject
            n_logits = hc
        else:
            down = child.input_mix_weight_down
            n_logits = 0
        if _patched(down):
            n, k = down.weight.shape
            plan = HC_DOWN_SILU_SM86_PLANS.get((n, k), {})
            configs = [c for c in plan.values() if _config_fits(c, rank + n_logits, k)]
            if configs and rank + n_logits <= n:
                child._hc_fuse_down = True
                warmup.update((_DOWN, c, rank, hc) for c in configs)
        up = child.input_mix_weight_up
        if _patched(up):
            n, k = up.weight.shape
            plan = HC_UP_GATE_MIX_SM86_PLANS.get((n, k), {})
            configs = [
                c
                for c in plan.values()
                if _config_fits(c, n, k) and c.outputs_per_block % hc == 0
            ]
            if configs and n == child.hyper_hidden_size:
                child._hc_fuse_up = True
                warmup.update((_UP, c, 0, hc) for c in configs)
    hc_small_fused.request_warmup(warmup)
    if cn_warmup:
        hc_combine_norm_cute.request_warmup(cn_warmup)


__all__ = [
    "HC_FUSE_SWITCH",
    "HC_SM86_TUNE_SWITCH",
    "enable_qwen4_exp_hc_small_fused",
    "hc_combine_norm_cute",
    "hc_small_fused",
]

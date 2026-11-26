# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 lebeche hill-climb (fork of Merlin 3.0.2.2)
"""Fused small-M shared-expert gate_up_proj + silu_and_mul for sm_80/86/89.

In Qwen3.8-Flash-Next (TP=2), each MoE layer runs a BF16 shared expert
(``Qwen2MoeMLP``) whose ``gate_up_proj`` has local shape ``(640, 2560)``
followed by ``SiluAndMul`` (``silu(gate) * up`` -> ``[M, 320]``). Under
``VLLM_COMPILE`` the activation executes as Inductor's standalone Triton kernel
``triton_poi_fused_mul_silu_slice_0`` (52.6 calls/step, ~0.175 ms/step).

This module fuses the skinny BF16 GEMM and the SwiGLU epilogue into one
CuTe-DSL kernel (``CuteSharedGateUpSilu``) behind the opaque custom op
``vllm.lebeche_shared_gate_up_silu``:

* Dispatch on the token count happens at RUN time inside the opaque op, so
  ``torch.compile`` dynamic-shape traces with dropped guards still engage the
  fused kernel at small M and fall back cleanly at large M.
* Each CTA computes ``outputs_per_block // 2`` pairs of ``(gate_j, up_j)``
  rows (weight rows ``j`` and ``j + N//2``). The epilogue rounds each GEMM
  output to BF16 (the unfused GEMM rounding point), then evaluates
  ``(gate / (exp(-gate) + 1.0)) * up`` using the exact PTX sequence of
  ``triton_poi_fused_mul_silu_slice_0`` (libdevice ``__nv_expf`` +
  ``div.full.f32`` + ``mul.f32``).
* At M = 1..4 the plans match the ``(block_size, vector_width)`` of
  ``QWEN4_EXP_SM86_GEMM_PLANS[(640, 2560)]``, so the fused output is
  bit-identical to commit ``fbc1a32``. At M = 6, 8 (e.g. c2 decode) the fused
  kernel replaces cuBLAS + Triton with ~2x lower mean FP64 error and saves
  ~2.4-3.2 us/layer. Above the plan table the op falls back to the unfused
  ``fbc1a32`` sequence.

``MERLIN_LEBECHE_SHARED_ACT_FUSE`` (default "1"; "0" = behaviour of commit
``fbc1a32``) is read once at import, is a hashed compile factor, and requires
``MERLIN_LEBECHE_SM86_LL`` on an sm_80/86/89 device.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

import torch
from torch import nn

import vllm.envs as envs
from vllm.merlin.envs import read_registered
from vllm.merlin.lebeche_sm86 import SM86_LL_SWITCH, sm86_ll_active
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    row_stride_ok,
    shape_dynamic_skinny_gemm,
)
from vllm.utils.torch_utils import direct_register_custom_op

SHARED_ACT_SWITCH: bool = bool(read_registered("MERLIN_LEBECHE_SHARED_ACT_FUSE"))
SHARED_GATE_SWITCH: bool = bool(read_registered("MERLIN_LEBECHE_SHARED_GATE_FUSE"))
_WARP = 32
_PREFETCH_TILES = 2

# Test/diagnostic counters; never read on the hot path by the engine.
CALLS = {"fused": 0, "fallback": 0, "fused_gate": 0, "fallback_gate": 0}

# RTX 3090 (sm_86) plans for TP=2 shared-expert gate_up_proj (640, 2560),
# selected by CUDA-graph replay sweeps (24 distinct weights, best of 60).
# M = 1..4 reuse (block_size, vector_width) of QWEN4_EXP_SM86_GEMM_PLANS so
# GEMM accumulation is bit-identical to commit fbc1a32.
SHARED_GATE_UP_SILU_SM86_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
    (640, 2560): {
        1: SkinnyGemmConfig(1, 320, 2, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 2, vector_width=4, static_k=2560),
        3: SkinnyGemmConfig(3, 32, 2, vector_width=8, static_k=2560),
        4: SkinnyGemmConfig(4, 64, 2, vector_width=4, static_k=2560),
        6: SkinnyGemmConfig(6, 64, 2, vector_width=8, static_k=2560),
        8: SkinnyGemmConfig(8, 64, 2, vector_width=8, static_k=2560),
    },
}


# Virtual-thread (gate_block_size, gate_vector_width) for shared_expert_gate (1, 2560),
# matching QWEN4_EXP_SM86_GEMM_PLANS[(1, 2560)] bit-for-bit at M = 1..4.
SHARED_GATE_MATCH_PLANS: dict[int, tuple[int, int]] = {
    1: (256, 2),
    2: (320, 4),
    3: (128, 4),
    4: (128, 4),
    6: (64, 8),
    8: (64, 8),
}


def shared_act_active() -> bool:
    """Both switches on and the current CUDA device is sm_80/86/89."""
    return SHARED_ACT_SWITCH and SM86_LL_SWITCH and sm86_ll_active()


def shared_gate_active() -> bool:
    """Shared gate+act fusion switches on and the current CUDA device is sm_80/86/89."""
    return SHARED_GATE_SWITCH and shared_act_active()


def _is_packed_row_major(tensor: torch.Tensor) -> bool:
    return tensor.dim() == 2 and tensor.stride() == (tensor.shape[1], 1)


def _config_fits(config: SkinnyGemmConfig, n: int, k: int) -> bool:
    return (
        config.outputs_per_block % 2 == 0
        and n % config.outputs_per_block == 0
        and config.num_rows * config.outputs_per_block <= _WARP
        and k % (config.block_size * config.vector_width) == 0
        and (config.static_k is None or config.static_k == k)
    )


def _runtime_ok(x: torch.Tensor, weight: torch.Tensor, config: SkinnyGemmConfig) -> bool:
    return (
        not envs.VLLM_BATCH_INVARIANT
        and x.dim() == 2
        and row_stride_ok(x, config)
        and _is_packed_row_major(weight)
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.shape[1] == weight.shape[1]
    )


def _build_cute_kernel(config: SkinnyGemmConfig):
    import cutlass.cute as cute
    from cutlass import BFloat16
    from quack.compile_utils import make_fake_tensor

    from vllm.merlin._lebeche_shared_act_cute import CuteSharedGateUpSilu

    m = config.num_rows
    n = cute.sym_int(divisibility=config.outputs_per_block)
    k = (
        config.static_k
        if config.static_k is not None
        else cute.sym_int(divisibility=config.block_size * config.vector_width)
    )
    a = make_fake_tensor(BFloat16, (m, k), divisibility=config.vector_width)
    b = make_fake_tensor(BFloat16, (n, k), divisibility=config.vector_width)
    c = make_fake_tensor(BFloat16, (m, cute.sym_int()), divisibility=1)
    kern = CuteSharedGateUpSilu(
        element_type=BFloat16,
        num_rows=m,
        block_size=config.block_size,
        outputs_per_block=config.outputs_per_block,
        vector_width=config.vector_width,
        k_unroll=config.k_unroll,
        use_pdl=shape_dynamic_skinny_gemm._use_pdl(),
        static_k=config.static_k,
    )
    return cute.compile(
        kern,
        a,
        b,
        c,
        shape_dynamic_skinny_gemm._stream(),
        options="--enable-tvm-ffi --ptxas-options -maxrregcount=64",
    )


def _build_cute_gate_kernel(
    config: SkinnyGemmConfig, gate_block_size: int, gate_vector_width: int
):
    import cutlass.cute as cute
    from cutlass import BFloat16
    from quack.compile_utils import make_fake_tensor

    from vllm.merlin._lebeche_shared_act_cute import CuteSharedGateUpSiluGate

    assert config.static_k is not None
    m = config.num_rows
    n = cute.sym_int(divisibility=config.outputs_per_block)
    k = config.static_k
    a_div = max(config.vector_width, gate_vector_width)
    a = make_fake_tensor(BFloat16, (m, k), divisibility=a_div)
    b = make_fake_tensor(BFloat16, (n, k), divisibility=config.vector_width)
    gw = make_fake_tensor(BFloat16, (1, k), divisibility=gate_vector_width)
    c = make_fake_tensor(BFloat16, (m, cute.sym_int()), divisibility=1)
    go = make_fake_tensor(BFloat16, (m, 1), divisibility=1)
    kern = CuteSharedGateUpSiluGate(
        element_type=BFloat16,
        num_rows=m,
        block_size=config.block_size,
        outputs_per_block=config.outputs_per_block,
        vector_width=config.vector_width,
        k_unroll=config.k_unroll,
        use_pdl=shape_dynamic_skinny_gemm._use_pdl(),
        static_k=k,
        gate_block_size=gate_block_size,
        gate_vector_width=gate_vector_width,
    )
    return cute.compile(
        kern,
        a,
        b,
        gw,
        c,
        go,
        shape_dynamic_skinny_gemm._stream(),
        options="--enable-tvm-ffi --ptxas-options -maxrregcount=64",
    )


class SharedActFused:
    """Compile cache and CuTe-DSL warmup provider for fused shared-expert gate_up+SiLU."""

    def __init__(self) -> None:
        self._compiled: dict[SkinnyGemmConfig, Any] = {}
        self._warmup: set[SkinnyGemmConfig] = set()
        self._compiled_gate: dict[tuple[SkinnyGemmConfig, int, int], Any] = {}
        self._warmup_gate: set[tuple[SkinnyGemmConfig, int, int]] = set()
        self._warmup_registered = False

    def _compile(self, config: SkinnyGemmConfig) -> None:
        if config in self._compiled:
            return
        self._compiled[config] = _build_cute_kernel(config)

    def _compile_gate(
        self, key: tuple[SkinnyGemmConfig, int, int]
    ) -> None:
        if key in self._compiled_gate:
            return
        cfg, gbs, gvw = key
        self._compiled_gate[key] = _build_cute_gate_kernel(cfg, gbs, gvw)

    def _ensure_warmup_registered(self) -> None:
        if self._warmup_registered or (not self._warmup and not self._warmup_gate):
            return
        from vllm.model_executor.warmup.cutedsl_warmup import (
            register_cutedsl_warmup_provider,
        )

        register_cutedsl_warmup_provider(self)
        self._warmup_registered = True

    def request_warmup(self, configs: Iterable[SkinnyGemmConfig]) -> None:
        self._warmup.update(configs)
        self._ensure_warmup_registered()

    def request_gate_warmup(
        self, items: Iterable[tuple[SkinnyGemmConfig, int, int]]
    ) -> None:
        self._warmup_gate.update(items)
        self._ensure_warmup_registered()

    def get_cutedsl_warmup_compile_units(self):
        from vllm.model_executor.warmup.cutedsl_warmup import CuTeDSLCompileUnit

        def _cfg_key(c: SkinnyGemmConfig):
            return (
                c.num_rows,
                c.block_size,
                c.outputs_per_block,
                c.vector_width,
                c.k_unroll,
                c.static_k or 0,
            )

        units = [
            CuTeDSLCompileUnit(
                name=f"Lebeche shared-expert fused gate_up+SiLU M={cfg.num_rows}",
                key=("lebeche-shared-gate-up-silu", cfg),
                compile=partial(self._compile, cfg),
            )
            for cfg in sorted(self._warmup, key=_cfg_key)
        ]
        units.extend(
            CuTeDSLCompileUnit(
                name=f"Lebeche shared-expert fused gate_up+SiLU+gate M={item[0].num_rows}",
                key=("lebeche-shared-gate-up-silu-gate", *item),
                compile=partial(self._compile_gate, item),
            )
            for item in sorted(
                self._warmup_gate, key=lambda it: (_cfg_key(it[0]), it[1], it[2])
            )
        )
        return tuple(units)

    def gate_up_silu(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        config: SkinnyGemmConfig,
    ) -> torch.Tensor:
        if config not in self._compiled:
            self._compile(config)
        out = torch.empty(
            (x.shape[0], weight.shape[0] // 2), dtype=x.dtype, device=x.device
        )
        self._compiled[config](x, weight, out, shape_dynamic_skinny_gemm._stream())
        return out

    def gate_up_silu_gate(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        gate_weight: torch.Tensor,
        config: SkinnyGemmConfig,
        gate_block_size: int,
        gate_vector_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (config, gate_block_size, gate_vector_width)
        if key not in self._compiled_gate:
            self._compile_gate(key)
        out = torch.empty(
            (x.shape[0], weight.shape[0] // 2), dtype=x.dtype, device=x.device
        )
        gate_out = torch.empty((x.shape[0], 1), dtype=x.dtype, device=x.device)
        self._compiled_gate[key](
            x, weight, gate_weight, out, gate_out, shape_dynamic_skinny_gemm._stream()
        )
        return out, gate_out


shared_act_fused = SharedActFused()
_FALLBACK_ACT: nn.Module | None = None


def _fallback_silu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
    if _FALLBACK_ACT is not None:
        return _FALLBACK_ACT(gate_up)
    d = gate_up.shape[-1] // 2
    g = gate_up[..., :d].float()
    u = gate_up[..., d:].float()
    return (torch.nn.functional.silu(g) * u).to(gate_up.dtype)


def _lebeche_shared_gate_up_silu(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """``silu((x @ weight[:D].T)) * (x @ weight[D:].T)`` with runtime M dispatch."""
    if shared_act_active() and shape_dynamic_skinny_gemm.is_available():
        plan = SHARED_GATE_UP_SILU_SM86_PLANS.get((weight.shape[0], weight.shape[1]))
        config = None if plan is None else plan.get(x.shape[0])
        if (
            config is not None
            and _runtime_ok(x, weight, config)
            and _config_fits(config, weight.shape[0], weight.shape[1])
        ):
            CALLS["fused"] += 1
            return shared_act_fused.gate_up_silu(x, weight, config)
    CALLS["fallback"] += 1
    from vllm.models.qwen4_exp.nvidia.low_latency_gemm import (
        _qwen4_exp_low_latency_gemm,
    )

    gate_up = _qwen4_exp_low_latency_gemm(x, weight)
    return _fallback_silu_and_mul(gate_up)


def _lebeche_shared_gate_up_silu_fake(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0] // 2))


direct_register_custom_op(
    op_name="lebeche_shared_gate_up_silu",
    op_func=_lebeche_shared_gate_up_silu,
    fake_impl=_lebeche_shared_gate_up_silu_fake,
)


def _gate_runtime_ok(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    config: SkinnyGemmConfig,
    gate_block_size: int,
    gate_vector_width: int,
) -> bool:
    a_div = max(config.vector_width, gate_vector_width)
    return (
        config.static_k is not None
        and config.static_k == x.shape[1]
        and config.static_k % (gate_block_size * gate_vector_width) == 0
        and x.stride(1) == 1
        and (x.shape[0] == 1 or x.stride(0) % a_div == 0)
        and _is_packed_row_major(gate_weight)
        and gate_weight.shape == (1, x.shape[1])
        and gate_weight.dtype == torch.bfloat16
        and gate_weight.is_cuda
        and gate_weight.device == x.device
    )


def _lebeche_shared_gate_up_silu_gate(
    x: torch.Tensor, weight: torch.Tensor, gate_weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused shared-expert ``gate_up+SiLU`` and ``shared_expert_gate`` with runtime M dispatch."""
    if shared_gate_active() and shape_dynamic_skinny_gemm.is_available():
        plan = SHARED_GATE_UP_SILU_SM86_PLANS.get((weight.shape[0], weight.shape[1]))
        config = None if plan is None else plan.get(x.shape[0])
        gmatch = SHARED_GATE_MATCH_PLANS.get(x.shape[0])
        if (
            config is not None
            and gmatch is not None
            and _runtime_ok(x, weight, config)
            and _config_fits(config, weight.shape[0], weight.shape[1])
            and _gate_runtime_ok(x, gate_weight, config, gmatch[0], gmatch[1])
        ):
            CALLS["fused_gate"] += 1
            return shared_act_fused.gate_up_silu_gate(
                x, weight, gate_weight, config, gmatch[0], gmatch[1]
            )
    CALLS["fallback_gate"] += 1
    from vllm.models.qwen4_exp.nvidia.low_latency_gemm import (
        _qwen4_exp_low_latency_gemm,
    )

    out = _lebeche_shared_gate_up_silu(x, weight)
    gate = _qwen4_exp_low_latency_gemm(x, gate_weight)
    return out, gate


def _lebeche_shared_gate_up_silu_gate_fake(
    x: torch.Tensor, weight: torch.Tensor, gate_weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        x.new_empty((*x.shape[:-1], weight.shape[0] // 2)),
        x.new_empty((*x.shape[:-1], gate_weight.shape[0])),
    )


direct_register_custom_op(
    op_name="lebeche_shared_gate_up_silu_gate",
    op_func=_lebeche_shared_gate_up_silu_gate,
    fake_impl=_lebeche_shared_gate_up_silu_gate_fake,
)


def maybe_register_mlp(mlp: nn.Module) -> bool:
    """Check static eligibility of a ``Qwen2MoeMLP`` and register warmup configs."""
    if not shared_act_active():
        return False
    if not shape_dynamic_skinny_gemm.is_available():
        return False
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        UnquantizedLinearMethod,
    )

    gate_up = getattr(mlp, "gate_up_proj", None)
    act_fn = getattr(mlp, "act_fn", None)
    if not isinstance(gate_up, MergedColumnParallelLinear):
        return False
    if not isinstance(act_fn, SiluAndMul):
        return False
    qm = getattr(gate_up, "quant_method", None)
    if not isinstance(qm, UnquantizedLinearMethod):
        return False
    if getattr(gate_up, "bias", None) is not None:
        return False
    if getattr(gate_up, "gather_output", False):
        return False
    output_sizes = getattr(gate_up, "output_sizes", None)
    if not output_sizes or len(output_sizes) != 2 or output_sizes[0] != output_sizes[1]:
        return False
    weight = getattr(gate_up, "weight", None)
    if weight is None or weight.dim() != 2 or weight.dtype != torch.bfloat16:
        return False
    n, k = int(weight.shape[0]), int(weight.shape[1])
    plan = SHARED_GATE_UP_SILU_SM86_PLANS.get((n, k))
    if not plan:
        return False
    configs = [c for c in plan.values() if _config_fits(c, n, k)]
    if not configs:
        return False
    global _FALLBACK_ACT
    if _FALLBACK_ACT is None:
        _FALLBACK_ACT = act_fn
    shared_act_fused.request_warmup(configs)
    return True


def maybe_register_mlp_with_gate(mlp: nn.Module) -> bool:
    """Check static eligibility of a ``Qwen2MoeMLP`` with ``expert_gate`` and register warmup."""
    if not shared_gate_active():
        return False
    if not shape_dynamic_skinny_gemm.is_available():
        return False
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        ReplicatedLinear,
        UnquantizedLinearMethod,
    )

    gate_up = getattr(mlp, "gate_up_proj", None)
    act_fn = getattr(mlp, "act_fn", None)
    expert_gate = getattr(mlp, "expert_gate", None)
    if not isinstance(gate_up, MergedColumnParallelLinear):
        return False
    if not isinstance(act_fn, SiluAndMul):
        return False
    if not isinstance(expert_gate, ReplicatedLinear):
        return False
    if not isinstance(getattr(gate_up, "quant_method", None), UnquantizedLinearMethod):
        return False
    if not isinstance(
        getattr(expert_gate, "quant_method", None), UnquantizedLinearMethod
    ):
        return False
    if getattr(gate_up, "bias", None) is not None:
        return False
    if getattr(expert_gate, "bias", None) is not None:
        return False
    if getattr(gate_up, "gather_output", False):
        return False
    output_sizes = getattr(gate_up, "output_sizes", None)
    if not output_sizes or len(output_sizes) != 2 or output_sizes[0] != output_sizes[1]:
        return False
    weight = getattr(gate_up, "weight", None)
    gweight = getattr(expert_gate, "weight", None)
    if weight is None or weight.dim() != 2 or weight.dtype != torch.bfloat16:
        return False
    if gweight is None or gweight.dim() != 2 or gweight.dtype != torch.bfloat16:
        return False
    n, k = int(weight.shape[0]), int(weight.shape[1])
    if int(gweight.shape[0]) != 1 or int(gweight.shape[1]) != k:
        return False
    plan = SHARED_GATE_UP_SILU_SM86_PLANS.get((n, k))
    if not plan:
        return False
    items: list[tuple[SkinnyGemmConfig, int, int]] = []
    for m_val, c in plan.items():
        gmatch = SHARED_GATE_MATCH_PLANS.get(m_val)
        if (
            gmatch is not None
            and c.static_k == k
            and _config_fits(c, n, k)
            and k % (gmatch[0] * gmatch[1]) == 0
        ):
            items.append((c, gmatch[0], gmatch[1]))
    if not items:
        return False
    global _FALLBACK_ACT
    if _FALLBACK_ACT is None:
        _FALLBACK_ACT = act_fn
    shared_act_fused.request_gate_warmup(items)
    return True


__all__ = [
    "CALLS",
    "SHARED_ACT_SWITCH",
    "SHARED_GATE_MATCH_PLANS",
    "SHARED_GATE_SWITCH",
    "SHARED_GATE_UP_SILU_SM86_PLANS",
    "maybe_register_mlp",
    "maybe_register_mlp_with_gate",
    "shared_act_active",
    "shared_gate_active",
    "shared_act_fused",
]

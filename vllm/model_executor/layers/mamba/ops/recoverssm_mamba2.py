# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""RecoverSSM for Mamba-2 (NemotronH / Nemotron Super VL): speculative verify from one
read-only fp32 checkpoint and accepted-state recovery after sampling.

Mamba-2 port of the GDN RecoverSSM kernels (gdn/recoverssm_gdn.py). The verify kernel
runs the stock selective-state-update recurrence (ops/mamba_ssm.py:402-454, tied head
dim) over a speculative window of ``T = 1 + K`` tokens with the state carried in
registers, emits every token's output and, instead of storing per-token states, writes
one fp32 "replay" record per token: per head ``x_t`` (P floats), ``dt_t`` (after bias
and softplus) and ``dA_t = exp(A dt_t)``; per group ``B_t`` (N floats). ``C`` is
output-only and not recorded; the checkpoint is never written.

The commit kernel reloads the checkpoint and replays ``S = S * dA_t + (B_t dt_t) x_t`` for
the accepted prefix with the very floats the verify multiplied, so the committed state is
the verify's per-token state bit for bit (and equal to serial stock execution of the
accepted tokens when the compiler contracts both kernels alike). In align mode the same
single traversal also stores the block-boundary state from registers when it reaches it,
then the final state, so either destination may alias the source block (the V2 pre-copy
makes ``source == final != boundary`` normal).

Record row layout (``replay[blk, g, t, :]``, ``HPG = HL // G`` heads per group):
``[0:N] = B_t``; for head ``j`` of the group at ``base = N + j * (P + 2)``:
``[base:base+P] = x_t``, ``[base+P] = dt_t``, ``[base+P+1] = dA_t``."""

import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    softplus,
    try_get_optimal_ssm_config,
)
from vllm.model_executor.layers.mamba.ops.triton_helpers import fast_exp
from vllm.model_executor.layers.mamba.recoverssm_utils import recoverssm_require
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

MAMBA2_RECOVERSSM_RECORD_DTYPE = torch.float32


def _require(cond: bool, msg: str) -> None:
    recoverssm_require(cond, msg, "Mamba2 RecoverSSM")


def mamba2_recoverssm_record_shape(
    local_nheads: int,
    local_ngroups: int,
    head_dim: int,
    state_size: int,
    num_spec: int,
) -> tuple[int, int, int]:
    """Per-block shape of the fp32 replay record: ``(G, 1 + num_spec, N + HPG * (P + 2))``."""
    _require(
        local_ngroups > 0 and local_nheads % local_ngroups == 0,
        "local heads must be a positive multiple of the local groups",
    )
    hpg = local_nheads // local_ngroups
    return (local_ngroups, 1 + num_spec, state_size + hpg * (head_dim + 2))


@triton.jit
def _recoverssm_mamba2_verify_kernel(
    state_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    out_ptr,
    replay_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    null_block_id,
    num_tokens,
    stride_state_block,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    stride_x_token,
    stride_x_head,
    stride_x_dim,
    stride_dt_token,
    stride_dt_head,
    stride_dt_bias_head,
    stride_A_head,
    stride_B_token,
    stride_B_group,
    stride_B_dstate,
    stride_C_token,
    stride_C_group,
    stride_C_dstate,
    stride_D_head,
    stride_out_token,
    stride_out_head,
    stride_out_dim,
    stride_replay_block,
    stride_replay_group,
    stride_replay_pos,
    stride_replay_elem,
    stride_qsl,
    stride_si,
    P: tl.constexpr,
    N: tl.constexpr,
    HPG: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_g = pid_h // HPG
    head_in_group = pid_h % HPG

    bos = tl.load(query_start_loc_ptr + pid_b * stride_qsl).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (pid_b + 1) * stride_qsl).to(tl.int64)
    # Device-side guard (no host sync): a row never reads or writes activations
    # outside [0, num_tokens) even with malformed offsets. Both endpoints are
    # clamped into [0, num_tokens] before the subtraction, so extreme int64
    # offsets cannot wrap; a negative or non-monotonic row gets an empty window.
    # The commit applies the same rule (Mamba2RecoverSSMCommitContext.commit).
    bos_c = tl.minimum(tl.maximum(bos, 0), num_tokens)
    eos_c = tl.minimum(tl.maximum(eos, 0), num_tokens)
    query_len = tl.where((bos < 0) | (eos < bos), 0, tl.maximum(eos_c - bos_c, 0))
    state_idx = tl.load(state_indices_ptr + pid_b * stride_si).to(tl.int64)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    mask_m = offs_m < P
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    out_ptr += pid_h * stride_out_head
    if state_idx <= null_block_id:
        # Graph padding / freed row: deterministic zeros, no record.
        for t in tl.static_range(SPEC_QUERY_LEN):
            tl.store(
                out_ptr + (bos + t) * stride_out_token + offs_m * stride_out_dim,
                tl.zeros([BLOCK_SIZE_M], dtype=tl.float32),
                mask=(t < query_len) & mask_m,
            )
        return

    state_ptrs = (
        state_ptr
        + state_idx * stride_state_block
        + pid_h * stride_state_head
        + offs_m[:, None] * stride_state_dim
        + offs_n[None, :] * stride_state_dstate
    )
    state = tl.load(state_ptrs, mask=mask, other=0.0).to(tl.float32)

    x_ptr += pid_h * stride_x_head
    dt_ptr += pid_h * stride_dt_head
    dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    B_ptr += pid_g * stride_B_group
    C_ptr += pid_g * stride_C_group
    D_ptr += pid_h * stride_D_head
    rec_base = (
        replay_ptr + state_idx * stride_replay_block + pid_g * stride_replay_group
    )
    rec_head = N + head_in_group * (P + 2)

    for t in tl.static_range(SPEC_QUERY_LEN):
        valid = t < query_len
        tok = bos + t
        # Stock operation order (ops/mamba_ssm.py:402-454, TIE_HDIM branch).
        x = tl.load(
            x_ptr + tok * stride_x_token + offs_m * stride_x_dim,
            mask=valid & mask_m,
            other=0.0,
        ).to(tl.float32)
        dt = tl.load(dt_ptr + tok * stride_dt_token, mask=valid, other=0.0).to(
            tl.float32
        )
        dt += tl.load(dt_bias_ptr).to(tl.float32)
        if DT_SOFTPLUS:
            dt = softplus(dt)
        A = tl.load(A_ptr).to(tl.float32)
        dA = fast_exp(A * dt)
        B = tl.load(
            B_ptr + tok * stride_B_token + offs_n * stride_B_dstate,
            mask=valid & mask_n,
            other=0.0,
        ).to(tl.float32)
        C = tl.load(
            C_ptr + tok * stride_C_token + offs_n * stride_C_dstate,
            mask=valid & mask_n,
            other=0.0,
        ).to(tl.float32)
        D = tl.load(D_ptr).to(tl.float32)

        dB = B * dt
        new_state = state * dA + dB * x[:, None]
        state = tl.where(valid, new_state, state)

        out = tl.sum(state * C[None, :], axis=1)
        out += x * D
        tl.store(
            out_ptr + tok * stride_out_token + offs_m * stride_out_dim,
            out,
            mask=valid & mask_m,
        )

        # Replay record: x per head tile, dt/dA once per head, B once per group.
        rec = rec_base + t * stride_replay_pos
        tl.store(
            rec + (rec_head + offs_m) * stride_replay_elem, x, mask=valid & mask_m
        )
        if pid_m == 0:
            tl.store(rec + (rec_head + P) * stride_replay_elem, dt, mask=valid)
            tl.store(rec + (rec_head + P + 1) * stride_replay_elem, dA, mask=valid)
            if head_in_group == 0:
                tl.store(rec + offs_n * stride_replay_elem, B, mask=valid & mask_n)


def _tied_per_head(t: torch.Tensor, name: str, lead: int) -> torch.Tensor:
    """The per-head vector behind the mixer's expanded Mamba-2 parameter views.

    ``dt`` arrives as ``[tokens, HL, P]`` (stride(-1) == 0), ``A`` as ``[HL, P, N]``
    (two zero strides), ``dt_bias``/``D`` as ``[HL, P]``; ``lead`` is the number of
    leading real dimensions (2 for dt, 1 otherwise)."""
    if t.ndim == lead:
        return t
    _require(
        t.ndim > lead and all(t.stride(d) == 0 for d in range(lead, t.ndim)),
        f"{name} must be tied across the head dim (Mamba-2 TIE_HDIM form)",
    )
    idx = (slice(None),) * lead + (0,) * (t.ndim - lead)
    return t[idx]


def mamba2_recoverssm_verify(
    ssm_state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    replay_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    out: torch.Tensor,
    dt_softplus: bool = True,
    is_blackwell: bool = False,
) -> torch.Tensor:
    """Verify Mamba-2 speculative windows against their checkpoints without modifying
    them. Same tensors as the stock ``selective_state_update`` call in the mixer
    (``mamba_mixer2.py:1080-1096``): x/out ``[tokens, HL, P]``, dt ``[tokens, HL(, P)]``,
    A ``[HL(, P, N)]``, B/C ``[tokens, G, N]``, D/dt_bias ``[HL(, P)]``; plus the record
    ``replay_cache`` ``[blocks, G, T, N + HPG*(P+2)]`` fp32, the spec rows'
    ``query_start_loc`` ``[n+1]`` and their checkpoint blocks ``state_indices`` ``[n]``.
    Writes ``out`` in place and returns it."""
    _require(ssm_state.ndim == 4, "checkpoint must be [blocks, heads, head_dim, dstate]")
    nb, HL, P, N = ssm_state.shape
    _require(ssm_state.dtype == torch.float32, "checkpoint must use float32")
    _require(
        x.ndim == 3 and x.shape[1:] == (HL, P),
        "x must have shape [tokens, heads, head_dim]",
    )
    tokens = x.shape[0]
    _require(
        out.shape == x.shape and out.stride(-1) == 1,
        "out must match x with a contiguous head dim",
    )
    _require(x.stride(-1) == 1, "x head dim must be contiguous")
    dt2 = _tied_per_head(dt, "dt", 2)
    _require(tuple(dt2.shape) == (tokens, HL), "dt must have shape [tokens, heads]")
    A1 = _tied_per_head(A, "A", 1)
    D1 = _tied_per_head(D, "D", 1)
    bias1 = _tied_per_head(dt_bias, "dt_bias", 1)
    _require(
        tuple(A1.shape) == (HL,) and tuple(D1.shape) == (HL,) and tuple(bias1.shape) == (HL,),
        "A, D and dt_bias must be one value per head",
    )
    _require(
        B.ndim == 3 and B.shape[0] == tokens and B.shape[2] == N and C.shape == B.shape,
        "B and C must have shape [tokens, groups, dstate]",
    )
    G = B.shape[1]
    _require(G > 0 and HL % G == 0, "heads must be a multiple of the groups")
    _require(
        B.stride(-1) == 1 and C.stride(-1) == 1, "B and C dstate must be contiguous"
    )
    HPG = HL // G
    rec = (nb, G, spec_query_len, N + HPG * (P + 2))
    _require(
        tuple(replay_cache.shape) == rec, f"replay buffer needs shape {rec}"
    )
    _require(
        replay_cache.dtype == MAMBA2_RECOVERSSM_RECORD_DTYPE,
        "replay buffer must use float32",
    )
    _require(
        state_indices.ndim == 1 and state_indices.dtype in (torch.int32, torch.int64),
        "state indices must be a one-dimensional integer tensor",
    )
    _require(
        query_start_loc.dtype in (torch.int32, torch.int64),
        "query_start_loc must be an integer tensor",
    )
    rows = state_indices.shape[0]
    _require(
        query_start_loc.ndim == 1 and query_start_loc.shape[0] == rows + 1,
        "query metadata is incompatible",
    )
    _require(
        tokens <= rows * spec_query_len or rows == 0,
        "speculative decode input exceeds its activation capacity",
    )
    dev = ssm_state.device
    _require(
        all(
            t_.device == dev
            for t_ in (x, dt2, A1, B, C, D1, bias1, replay_cache, query_start_loc, state_indices, out)
        ),
        "inputs must be on the same device",
    )
    if rows == 0 or tokens == 0:
        return out

    BLOCK_SIZE_M, num_warps = try_get_optimal_ssm_config(
        P, N, rows, HL, "float32", is_blackwell
    )
    grid = (triton.cdiv(P, BLOCK_SIZE_M), rows, HL)
    with torch.accelerator.device_index(x.device.index):
        _recoverssm_mamba2_verify_kernel[grid](
            ssm_state,
            x,
            dt2,
            bias1,
            A1,
            B,
            C,
            D1,
            out,
            replay_cache,
            query_start_loc,
            state_indices,
            NULL_BLOCK_ID,
            tokens,
            ssm_state.stride(0),
            ssm_state.stride(1),
            ssm_state.stride(2),
            ssm_state.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            dt2.stride(0),
            dt2.stride(1),
            bias1.stride(0),
            A1.stride(0),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            D1.stride(0),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            replay_cache.stride(0),
            replay_cache.stride(1),
            replay_cache.stride(2),
            replay_cache.stride(3),
            query_start_loc.stride(0),
            state_indices.stride(0),
            P=P,
            N=N,
            HPG=HPG,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_DSTATE=triton.next_power_of_2(N),
            SPEC_QUERY_LEN=spec_query_len,
            DT_SOFTPLUS=dt_softplus,
            num_warps=num_warps,
        )
    return out


@triton.jit
def _commit_mamba2_state_kernel(
    state_ref_ptr,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    replay_ref_ptr,
    replay_base_addrs_ptr,
    replay_block_strides_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    stride_replay_group,
    stride_replay_pos,
    stride_replay_elem,
    stride_si,
    P: tl.constexpr,
    N: tl.constexpr,
    HPG: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
):
    # One program replays one head tile of one row for one layer, in a single
    # traversal: S0 is loaded once into registers, the block-boundary state (align
    # mode, after boundary_recovery_len tokens) is stored from the registers when the
    # traversal reaches it, and the final state after commit_len tokens. Either
    # destination may alias the source block (the V2 pre-copy puts the checkpoint in
    # the block of the scheduled window's end, so final == src with a distinct
    # boundary block is the common crossing case; boundary == src is the other
    # direction): no reload of the source happens after a store, so both are safe.
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_lh = tl.program_id(2)
    pid_l = pid_lh // NUM_HEADS
    pid_h = pid_lh % NUM_HEADS
    pid_g = pid_h // HPG
    head_in_group = pid_h % HPG

    src = tl.load(state_indices_ptr + pid_b * stride_si).to(tl.int64)
    if src <= null_block_id:
        return
    n = tl.load(commit_lens_ptr + pid_b)
    if n == 0:
        return
    final_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    if final_idx <= null_block_id:
        return
    boundary_len = 0
    boundary_idx = null_block_id
    if ALIGN_MODE:
        boundary_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
        boundary_len = tl.load(boundary_recovery_lens_ptr + pid_b)
        # A window ending exactly on the boundary keeps its final state in that
        # block (boundary == final): the final store covers it.
        if boundary_idx <= null_block_id or boundary_idx == final_idx:
            boundary_len = 0

    state_ptr = tl.load(state_base_addrs_ptr + pid_l).to(
        tl.pointer_type(state_ref_ptr.dtype.element_ty)
    )
    sbs = tl.load(state_block_strides_ptr + pid_l)
    replay_ptr = tl.load(replay_base_addrs_ptr + pid_l).to(
        tl.pointer_type(replay_ref_ptr.dtype.element_ty)
    )
    rbs = tl.load(replay_block_strides_ptr + pid_l)
    rec_base = replay_ptr + src * rbs + pid_g * stride_replay_group
    rec_head = N + head_in_group * (P + 2)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    mask_m = offs_m < P
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    sp = (
        pid_h * stride_state_head
        + offs_m[:, None] * stride_state_dim
        + offs_n[None, :] * stride_state_dstate
    )
    state = tl.load(state_ptr + src * sbs + sp, mask=mask, other=0.0).to(tl.float32)
    # Replay the verify's own update with the floats it multiplied, in the same
    # expression form, so the committed state is the per-token state bit for bit.
    for t in range(n):
        rec = rec_base + t * stride_replay_pos
        x = tl.load(rec + (rec_head + offs_m) * stride_replay_elem, mask=mask_m, other=0.0)
        dt = tl.load(rec + (rec_head + P) * stride_replay_elem)
        dA = tl.load(rec + (rec_head + P + 1) * stride_replay_elem)
        B = tl.load(rec + offs_n * stride_replay_elem, mask=mask_n, other=0.0)
        dB = B * dt
        state = state * dA + dB * x[:, None]
        if ALIGN_MODE:
            # boundary_len <= n whenever the plan set a boundary (the window crossed).
            tl.store(
                state_ptr + boundary_idx * sbs + sp,
                state.to(state_ref_ptr.dtype.element_ty),
                mask=mask & (t + 1 == boundary_len),
            )
    tl.store(
        state_ptr + final_idx * sbs + sp,
        state.to(state_ref_ptr.dtype.element_ty),
        mask=mask,
    )


__all__ = [
    "MAMBA2_RECOVERSSM_RECORD_DTYPE",
    "_commit_mamba2_state_kernel",
    "_recoverssm_mamba2_verify_kernel",
    "mamba2_recoverssm_record_shape",
    "mamba2_recoverssm_verify",
]

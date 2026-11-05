# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Decode-first step budget. ``VLLM_MERLIN_DECODE_FIRST``, strict 0/1, default off.

The scheduler's RUNNING loop walks ``self.running`` in admission order and
gives each request ``min(wanted, token_budget)``. A decoder admitted after a
long prefill that is still chunking therefore gets whatever the prefill chunk
left, which is usually nothing: it stalls until the prefill ends.

With the knob on, every running decoder that is eligible this step (its PP
decode cadence, the row cap, not already done) has its tokens reserved
before any running prefill chunk is sized: a prefill chunk takes at most
``budget - tokens reserved for eligible decoders behind it``. Decoders ahead of
the chunk are served before it anyway, so the knob changes nothing unless a
decoder sits behind a prefill chunk. Spec-decode verify tokens (MTP, DFlash2)
are part of a decoder's want and are reserved with it.

The iteration order, preemption order and batch order stay as they are; only
the size of a prefill chunk that has decoders behind it changes. The chunk is
never cut to nothing: if the reserved size would not advance the prefill (no
budget left, or a Mamba block-aligned split that rounds it to 0) the chunk keeps
its unreserved size for that step, as with the knob off, so the prefill always
advances whenever it would have without the knob.

Not batch-invariant: a smaller chunk moves the prefill's
chunk boundaries, and decoders join different steps, so T=0 outputs are not
proven identical across the knob. Hence default off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request

logger = init_logger(__name__)

KNOB = "VLLM_MERLIN_DECODE_FIRST"


@dataclass
class DecodeFirstStats:
    """Cumulative counters (reported in the one log line per 1000 steps)."""

    # Prefill tokens withheld from chunks so decoders behind them could run.
    reserved_tokens: int = 0
    # Steps on which a prefill chunk with decoders behind it was shrunk.
    shrunk_chunks: int = 0
    # Times the reservation was dropped so the prefill still advanced.
    kept_prefill: int = 0
    _logged: tuple[int, int, int] = (0, 0, 0)
    _next_log_step: int = 0


def state() -> DecodeFirstStats | None:
    """The knob's per-scheduler state, or None when it is off (the default)."""
    import vllm.envs as envs

    if not envs.VLLM_MERLIN_DECODE_FIRST:
        return None
    logger.info(
        "Decode-first step budget: knob on; running decoders' tokens "
        "are reserved before prefill chunks are sized"
    )
    return DecodeFirstStats()


def reservations(
    sched: Scheduler, decode_cap: int | None, long_prefill_token_threshold: int
) -> dict[str, int]:
    """Tokens each running decoder will ask for this step, keyed by request
    id, for the decoders the RUNNING loop would schedule (same eligibility
    tests and the same ``num_new_tokens`` expression, before the budget)."""
    out: dict[str, int] = {}
    rows = 0
    for request in sched.running:
        if request.is_prefill_chunk:
            continue
        if (
            request.num_output_placeholders > 0
            and request.num_computed_tokens + 2 - request.num_output_placeholders
            >= request.num_prompt_tokens + request.max_tokens
        ):
            continue
        if sched.current_step < request.next_decode_eligible_step:
            continue
        if decode_cap is not None and rows >= decode_cap:
            break
        want = (
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens
        )
        if 0 < long_prefill_token_threshold < want:
            want = long_prefill_token_threshold
        want = min(
            want,
            sched.max_model_len
            - request.num_computed_tokens
            - sched.num_sampled_tokens_per_step,
        )
        if want <= 0:
            continue
        out[request.request_id] = want
        rows += 1
    return out


def cap_prefill(
    sched: Scheduler,
    reserve: dict[str, int],
    req_index: int,
    request: Request,
    num_new_tokens: int,
    token_budget: int,
    input_avail: int,
    draft_slots: int,
) -> int:
    """The prefill chunk of ``request`` (at ``req_index`` in ``running``),
    leaving the reserved tokens of every eligible decoder behind it."""
    stats = sched.decode_first
    later = [
        reserve[r.request_id]
        for r in sched.running[req_index + 1 :]
        if r.request_id in reserve
    ]
    if not later:
        return num_new_tokens
    held = sum(later)
    capped = max(
        0,
        min(
            num_new_tokens,
            token_budget - held,
            input_avail - held - draft_slots * len(later),
        ),
    )
    if capped >= num_new_tokens:
        return num_new_tokens
    # Compare what each size really schedules (a Mamba block-aligned split
    # may clip both to the same boundary: then nothing was withheld).
    if sched.need_mamba_block_aligned_split:
        full = sched._mamba_block_aligned_split(request, num_new_tokens)
        kept = sched._mamba_block_aligned_split(request, capped) if capped else 0
    else:
        full, kept = num_new_tokens, capped
    if kept >= full:
        return capped
    if kept == 0:
        # The prefill must still advance whenever it would have without the
        # knob: keep its unreserved size this step.
        stats.kept_prefill += 1
        return num_new_tokens
    stats.reserved_tokens += full - kept
    stats.shrunk_chunks += 1
    _maybe_log(sched, stats)
    return capped


def _maybe_log(sched: Scheduler, stats: DecodeFirstStats) -> None:
    now = (stats.reserved_tokens, stats.shrunk_chunks, stats.kept_prefill)
    if now != stats._logged and sched.current_step >= stats._next_log_step:
        logger.info(
            "Decode-first: %d prefill tokens reserved for decoders over "
            "%d chunks; reservation dropped %d times to keep a prefill moving",
            *now,
        )
        stats._logged = now
        stats._next_log_step = sched.current_step + 1000

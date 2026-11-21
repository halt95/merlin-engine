# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Rejection-sampler residual diagnostic (``MERLIN_SPEC_RESIDUAL_DIAG``).

Observe-only. The counters read the rejection kernels' outputs after they ran
(the sampled ids, the per-block maxima of the residual draw, the processed
target logits); they never launch a kernel of their own, never draw a random
number and never write a tensor the sampler reads, so the sampled tokens are
the same with the switch on or off, with the stable residual on or off.

Init line (every rank that builds a rejection sampler, switch on or off)::

    merlin-spec: rejection sampler stable_residual=<True|False>
    effective=<True|False> residual_diag=<0|1> method=<method>
    draft=<draft_sample_method> rank=<PP<p>_TP<t>>

on one line. ``stable_residual`` is the value the sampler read at init;
``effective`` is what the kernels receive after the guard that turns the
repair off for synthetic and FLY acceptance.

Counter line (switch on only; every ``LOG_EVERY_STEPS`` verify steps of this
rank, and once more at shutdown with ``final=1``)::

    merlin-spec-residual-diag: rank=<tag> steps=<n> final=<0|1>
    rejection_events=<n> residual_mass_zero=<n> residual_outside_support=<n>
    token0_at_rejection=<n> ring_events=<n> ring_dropped=<n>

on one line. Counts are cumulative per rank: take the latest line per
``rank=``. A counter that could not be read is written as -1, never 0.

Counters, per verify row of a request (one per request per verify step):

* ``rejection_events``: a draft was rejected and the replacement was drawn
  from the residual: the row is not the bonus row, the rejected draft is a
  real token (not a -1 placeholder or a grammar-invalid draft), and the
  request samples (temperature != 0; an all-greedy batch has none).
* ``residual_mass_zero``: of those, the residual carried no mass: every
  vocabulary block of the residual draw has a -inf (or NaN) maximum, so the
  draw fell back to an arbitrary index. A -inf normaliser lands here too
  (it makes every residual logit NaN).
* ``residual_outside_support``: of those, the drawn token has a -inf (or NaN)
  processed target logit (grammar mask, top-k/top-p, allowed ids included) or
  lies outside the sampled vocabulary.
* ``token0_at_rejection``: of those, the drawn token is id 0.

Event ring: the first ``RING_CAPACITY`` ``residual_mass_zero`` events, as
(verify step, request slot, token position, output index, draft position);
the host resolves the slot to the request id from the per-step batch, so a
slot reused by a later request is never mislabelled. ``output_index`` is the
token position minus the prompt length (-1 when the prompt length is not
available); ``draft_position`` is the 0-based index of the rejected draft.

Dump: with ``MERLIN_SPEC_RESIDUAL_DIAG_OUT`` set, every counter line also
rewrites ``<dir>/<rank tag>.json`` (atomically) with the counters and the
resolved events; the shutdown one has ``"final": true``.

Cost, switch on: about 50 small elementwise/gather launches over num_reqs
rows (one 2-D gather of num_reqs logits, one reduction over the residual
draw's per-block maxima) per verify chunk on the sampler's stream, no host
sync; host work per step is one pass over the batch's request ids. Each
counter line costs one device sync (a 4-element read, plus the new ring
rows). Switch off: the sampler holds ``None`` and the verify path does two
``is not None`` checks per step.
"""

from __future__ import annotations

import bisect
import json
import logging
import os
import weakref
from collections.abc import Sequence
from typing import Any

import torch

logger = logging.getLogger(__name__)

LOG_EVERY_STEPS = 512
RING_CAPACITY = 4096
COUNTER_NAMES = (
    "rejection_events",
    "residual_mass_zero",
    "residual_outside_support",
    "token0_at_rejection",
)
INIT_PREFIX = "merlin-spec: rejection sampler"
LINE_PREFIX = "merlin-spec-residual-diag:"
_RING_COLUMNS = ("step", "slot", "token_position", "output_index", "draft_position")

_LIVE: weakref.WeakSet[ResidualDiag] = weakref.WeakSet()


def effective_stable_residual(
    stable_residual: bool, synthetic: bool, fly_window_size: int
) -> bool:
    """The value rejection_sample passes to the kernels (its first guard)."""
    return bool(stable_residual and not synthetic and fly_window_size == 0)


def init_line(
    stable_residual: bool,
    effective: bool,
    diag_on: bool,
    method: str,
    draft: str,
    rank: str,
) -> str:
    return (
        f"{INIT_PREFIX} stable_residual={bool(stable_residual)} "
        f"effective={bool(effective)} residual_diag={int(bool(diag_on))} "
        f"method={method} draft={draft} rank={rank}"
    )


def rank_tag() -> str:
    try:
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group

        return f"PP{get_pp_group().rank_in_group}_TP{get_tp_group().rank_in_group}"
    except Exception:
        # Before distributed init or in a single process: say so.
        return "PP?_TP?"


class ResidualDiag:
    """Per-rank device counters; created only with the switch on."""

    def __init__(
        self,
        device: torch.device | str,
        *,
        rank: str,
        out_dir: str = "",
        prompt_len: Any = None,
        header: dict[str, Any] | None = None,
        log_every: int = LOG_EVERY_STEPS,
        ring_capacity: int = RING_CAPACITY,
    ):
        self.rank = rank
        self.out_dir = out_dir
        # The request-state holder (its .gpu view is re-read on every step,
        # because the holder replaces it on each host-to-device copy).
        self.prompt_len = prompt_len
        self.header = dict(header or {})
        self.log_every = max(1, int(log_every))
        self.ring_capacity = int(ring_capacity)
        self.counts = torch.zeros(len(COUNTER_NAMES), dtype=torch.int64, device=device)
        # One scratch row past the end takes every write that is not an event.
        self.ring = torch.zeros(
            (self.ring_capacity + 1, len(_RING_COLUMNS)),
            dtype=torch.int64,
            device=device,
        )
        self.steps = 0
        self._last_flush_step = 0
        self._ring_read = 0
        self._owners: dict[int, list[tuple[int, str]]] = {}
        self.events: list[dict[str, Any]] = []
        self._warned = False
        _LIVE.add(self)

    # Host side, once per verify step, before the chunks are verified.
    def begin_step(self, req_ids: Sequence[str], idx_mapping_np: Any) -> None:
        if self.steps - self._last_flush_step >= self.log_every:
            self.flush(final=False)
        self.steps += 1
        slots = idx_mapping_np[: len(req_ids)].tolist()
        for slot, req_id in zip(slots, req_ids):
            history = self._owners.setdefault(int(slot), [])
            if not history or history[-1][1] != req_id:
                history.append((self.steps, req_id))

    # Device side, once per verify chunk, after the rejection kernels ran.
    @torch.no_grad()
    def observe(
        self,
        target_logits: torch.Tensor,
        draft_sampled: torch.Tensor,
        cu_num_logits: torch.Tensor,
        pos: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        resampled_local_max: torch.Tensor,
        vocab_size: int,
        all_greedy: bool,
    ) -> None:
        num_reqs = cu_num_logits.shape[0] - 1
        num_logits = target_logits.shape[0]
        if all_greedy or num_reqs <= 0 or num_logits == 0:
            return
        cu = cu_num_logits.to(torch.int64)
        n = num_sampled.to(torch.int64)
        row = cu[:-1] + n - 1
        is_bonus = row == cu[1:] - 1
        row_c = row.clamp(0, num_logits - 1)
        next_c = (row + 1).clamp(0, num_logits - 1)
        draft_ok = draft_sampled[next_c] >= 0
        slot = expanded_idx_mapping[row_c].to(torch.int64)
        sampling = temperature[slot] != 0
        rejected = ~is_bonus & draft_ok & sampling

        draft_pos = (n - 1).clamp(min=0)
        token = sampled.gather(1, draft_pos.unsqueeze(1)).squeeze(1).to(torch.int64)
        no_mass = ~(resampled_local_max > float("-inf")).any(dim=1)
        in_vocab = (token >= 0) & (token < vocab_size)
        token_c = token.clamp(0, target_logits.shape[1] - 1)
        supported = target_logits[row_c, token_c] > float("-inf")
        zero = rejected & no_mass
        outside = rejected & (~in_vocab | ~supported)
        token0 = rejected & (token == 0)

        # Ring slots for this chunk's zero-mass events, after the earlier ones.
        position = pos[row_c].to(torch.int64) + 1
        output_index = torch.full_like(position, -1)
        holder = self.prompt_len
        prompt = getattr(holder, "gpu", holder) if holder is not None else None
        if isinstance(prompt, torch.Tensor) and prompt.device == position.device:
            output_index = position - prompt[slot].to(torch.int64)
        zero_i = zero.to(torch.int64)
        index = self.counts[1] + zero_i.cumsum(0) - 1
        write = zero & (index < self.ring_capacity)
        dest = torch.where(write, index, torch.full_like(index, self.ring_capacity))
        values = torch.stack(
            (
                torch.full_like(position, self.steps),
                slot,
                position,
                output_index,
                draft_pos,
            ),
            dim=1,
        )
        self.ring[dest] = values
        self.counts += torch.stack(
            (rejected.sum(), zero_i.sum(), outside.sum(), token0.sum())
        ).to(torch.int64)

    def _resolve(self, row: Sequence[int]) -> dict[str, Any]:
        step, slot, position, output_index, draft_position = (int(x) for x in row)
        req_id = None
        history = self._owners.get(slot, [])
        i = bisect.bisect_right([s for s, _ in history], step) - 1
        if i >= 0:
            req_id = history[i][1]
        return {
            "req_id": req_id,
            "output_index": output_index,
            "token_position": position,
            "draft_position": draft_position,
            "step": step,
            "slot": slot,
        }

    def snapshot(self) -> list[int]:
        """Read the counters and resolve the new ring rows (one device sync)."""
        try:
            counts = [int(x) for x in self.counts.tolist()]
            hi = min(counts[1], self.ring_capacity)
            if hi > self._ring_read:
                for row in self.ring[self._ring_read : hi].tolist():
                    self.events.append(self._resolve(row))
                self._ring_read = hi
        except Exception as exc:  # an instrument never breaks a step
            if not self._warned:
                self._warned = True
                logger.warning("spec residual diag read failed: %s", exc)
            return [-1] * len(COUNTER_NAMES)
        # Every recorded event is resolved: one owner per slot is enough now.
        for slot, history in self._owners.items():
            if len(history) > 1:
                self._owners[slot] = history[-1:]
        return counts

    def line(self, counts: Sequence[int], final: bool) -> str:
        dropped = max(0, counts[1] - self.ring_capacity) if counts[1] >= 0 else -1
        fields = " ".join(f"{k}={v}" for k, v in zip(COUNTER_NAMES, counts))
        return (
            f"{LINE_PREFIX} rank={self.rank} steps={self.steps} final={int(final)} "
            f"{fields} ring_events={len(self.events)} ring_dropped={dropped}"
        )

    def flush(self, final: bool) -> str:
        counts = self.snapshot()
        self._last_flush_step = self.steps
        text = self.line(counts, final)
        logger.info("%s", text)
        if self.out_dir:
            self._dump(counts, final)
        return text

    def _dump(self, counts: Sequence[int], final: bool) -> None:
        body = {
            **self.header,
            "rank": self.rank,
            "pid": os.getpid(),
            "final": bool(final),
            "verify_steps": self.steps,
            "counters": dict(zip(COUNTER_NAMES, counts)),
            "ring_capacity": self.ring_capacity,
            "ring_dropped": (
                max(0, counts[1] - self.ring_capacity) if counts[1] >= 0 else -1
            ),
            "events": self.events,
        }
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            path = os.path.join(self.out_dir, f"{self.rank}.json")
            tmp = f"{path}.tmp{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(body, f, sort_keys=True)
            os.replace(tmp, path)
        except OSError as exc:
            if not self._warned:
                self._warned = True
                logger.warning("spec residual diag dump failed: %s", exc)


def build(
    *,
    stable_residual: bool,
    effective: bool,
    diag_on: bool,
    method: str,
    draft: str,
    rank: str,
    device: torch.device | str,
    out_dir: str = "",
    prompt_len: Any = None,
) -> ResidualDiag | None:
    """Log the init line; return the counters only with the switch on."""
    logger.info(
        "%s", init_line(stable_residual, effective, diag_on, method, draft, rank)
    )
    if not diag_on:
        return None
    header = {
        "stable_residual": bool(stable_residual),
        "effective": bool(effective),
        "residual_diag": 1,
        "method": method,
        "draft": draft,
    }
    return ResidualDiag(
        device, rank=rank, out_dir=out_dir, prompt_len=prompt_len, header=header
    )


def at_init(
    sampler: Any, spec_config: Any, device: torch.device
) -> ResidualDiag | None:
    """RejectionSampler.__init__ hook: reads the switches in this process."""
    from vllm import envs

    return build(
        stable_residual=sampler.stable_residual,
        effective=effective_stable_residual(
            sampler.stable_residual,
            sampler.synthetic_conditional_rates is not None,
            sampler.fly_window_size,
        ),
        diag_on=bool(envs.MERLIN_SPEC_RESIDUAL_DIAG),
        method=spec_config.rejection_sample_method,
        draft=spec_config.draft_sample_method,
        rank=rank_tag(),
        device=device,
        out_dir=envs.MERLIN_SPEC_RESIDUAL_DIAG_OUT,
        prompt_len=getattr(
            getattr(getattr(sampler, "sampler", None), "req_states", None),
            "prompt_len",
            None,
        ),
    )


def shutdown_all() -> None:
    """Model-runner shutdown: the last counter line (and dump) of each rank."""
    for diag in list(_LIVE):
        diag.flush(final=True)

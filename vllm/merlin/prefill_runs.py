# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Prefill runs. ``VLLM_MERLIN_PREFILL_RUNS``, strict 0/1, default off.

With pipeline parallelism a decoding session advances once every ``pp_size``
steps, and a step that carries prefill tokens runs eagerly and costs several
times a graphed decode-only step. While a long cold prompt prefills, every
step carries a chunk, so the sessions that ride along keep only a small share
of their decode speed whatever the chunk size. Taking the chunk out of single
steps does not help either: an isolated chunk exposes the whole pipeline
latency, so the prompt's first token takes about twice as long.

With the switch on, the scheduler time-slices instead:

* A step's prefill tokens (running chunks and admissions together) never
  exceed ``CHUNK_TOKENS`` (Q). Run with ``--max-num-batched-tokens`` at least
  Q + max_num_seqs x (1 + speculative tokens): the headroom above Q is for
  decode rows only, so riding decoders never shrink a chunk and a chunk
  stays on the Mamba block grid (no short "runt" step before each block
  boundary). Without decoders a step therefore schedules what an engine
  with a budget of Q schedules (the adaptive threshold floor included),
  except that an admission with one token left (a full prefix-cache hit or
  a landed restore, padded to its verify rows) is a decode row and takes the
  headroom instead of the cap.
* A long prefill (more than ``CHUNKS`` x Q tokens left when first seen) is
  time-sliced while at least one session decodes: it advances in a run of
  ``CHUNKS`` (P) consecutive full chunks, with the decoders riding on the
  steps their pipeline cadence allows, then waits for a decode-only window.
* The window lasts ``rounds`` decoder rounds (one round = every decoder
  decodes once), counted in scheduled decode rows, never in time:

      f_n    = DECODE_SHARE / (number of pending long prefills)
      rounds = round(f_n / (1 - f_n) x (P + fill) x prefill_tick / decode_round)

  where ``TICK_COSTS`` = "prefill_tick ms,decode_round ms,fill ticks" are
  per-profile constants measured offline. The runs read no timer: their
  decisions follow from the scheduler-visible arrivals, completions and
  cache state (other admission rules keep their own timeouts).
* One long prefill holds the runs at a time (the first in ``running`` order);
  other long prefills, running or waiting, wait for it.
* A long prefill's step is all or nothing: it takes min(Q, left) tokens or it
  waits. A short prefill (a new turn, a late session, a landed restore)
  is scheduled ahead of it and delays it by whole steps; it is never resized.
* While the prefill runs, the long-prefill token threshold does not apply
  (the runs replace it); it applies as before whenever runs are inactive.

``DECODE_SHARE=0`` keeps only the first point (the per-step prefill cap with
decode headroom): no runs, no windows.

Start-up refuses combinations that would fight over the same steps: the
decode-first budget (``VLLM_MERLIN_DECODE_FIRST``), a prefill schedule
interval above 1 and data parallelism; it also refuses a scheduled- or
batched-token budget without the decode headroom above Q, a chunk or a
nonzero long-prefill threshold below ``MIN_PREFILL_TOKENS``, a chunk that does
not tile the Mamba block, a chunk below the drafter's prefill lookahead (a
tail of Q+1..lookahead tokens could never be scheduled), chunked prefill off
with prompts longer than Q, whole multimodal items
(``--disable-chunked-mm-input``), bounded prefix replay, and tick costs or a
share that give no decoder round per window or one too large to count.

One long prefill holds the runs: the first in ``running`` order (after a
preemption another running long prefill may take over). A steady stream of
short prefills keeps it waiting, by the short-first rule. Run and window
counters belong to the scheduler, not to a request: a new holder continues
the current run. Long membership covers one prefill: it ends when that
prefill's tokens are computed, whatever the request's status.

With the switch off nothing here runs (``state`` returns None and every
scheduler hook is guarded): the scheduling decisions are those of the tree
without this module, which is still imported.

A lone request computes the same chunks it computes with the switch off on
an engine whose budget is Q; with decoders the steps it shares with them
change, so outputs at concurrency above 1 are not proven identical. Hence
default off in the engine; the Flash-Next launcher rows turn it on
(tools/merlin/launch.py, PREFILL_RUNS=0 turns it off there).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request

logger = init_logger(__name__)

KNOB = "VLLM_MERLIN_PREFILL_RUNS"
# Smallest per-request prefill cap the runs accept: the chunk Q, and a nonzero
# --long-prefill-token-threshold (which caps a request while the runs are
# inactive; its adaptive floor min(input budget, Q) // requests only raises
# it). A request stalls under a cap c exactly when it has between c + 1 and
# the drafter's prefill lookahead (K <= 3 on the served drafters) tokens left:
# _reserve_prefill_lookahead takes back every token the cap allows. Every cap
# >= this floor is far above any served lookahead (the lookahead check below
# stays for larger ones). Mamba block alignment never cuts a full-cap step to
# zero by itself; its stop at a block boundary within the lookahead of the
# prompt end does not depend on the cap and is the same with the runs off.
# The shipped chunks are 1024 and 2048 and the shipped threshold 512.
MIN_PREFILL_TOKENS = 256
# Waiting requests inspected per step for a short prefill to put first.
WAITING_SCAN = 8


@dataclass(frozen=True)
class RunsConfig:
    chunk: int  # Q: prefill tokens per step
    run_chunks: int  # P: full chunks per run
    decode_share: float  # f
    prefill_tick_ms: float
    decode_round_ms: float
    fill_ticks: float

    @property
    def long_tokens(self) -> int:
        """A prefill with more tokens left than one run is long."""
        return self.run_chunks * self.chunk


def window_rounds(cfg: RunsConfig, num_long: int) -> int:
    """Decoder rounds in one decode-only window with ``num_long`` long
    prefills pending (f is divided among them)."""
    share = cfg.decode_share / max(1, num_long)
    if share <= 0.0:
        return 0
    run_ms = (cfg.run_chunks + cfg.fill_ticks) * cfg.prefill_tick_ms
    window_ms = share / (1.0 - share) * run_ms
    rounds = window_ms / cfg.decode_round_ms + 0.5
    if not math.isfinite(rounds):
        # Each tunable is finite on its own, but their product overflowed.
        raise ValueError(
            f"VLLM_MERLIN_PREFILL_RUNS_TICK_COSTS {cfg.prefill_tick_ms:g},"
            f"{cfg.decode_round_ms:g},{cfg.fill_ticks:g} with "
            f"VLLM_MERLIN_PREFILL_RUNS_CHUNKS {cfg.run_chunks} and "
            f"VLLM_MERLIN_PREFILL_RUNS_DECODE_SHARE {cfg.decode_share:g} give "
            "a decode window too large to count: set the tick costs in "
            "milliseconds (the default is 139,25,1)"
        )
    return int(rounds)


def config_from_env() -> RunsConfig:
    import vllm.envs as envs

    prefill_tick, decode_round, fill = envs.VLLM_MERLIN_PREFILL_RUNS_TICK_COSTS
    return RunsConfig(
        chunk=envs.VLLM_MERLIN_PREFILL_RUNS_CHUNK_TOKENS,
        run_chunks=envs.VLLM_MERLIN_PREFILL_RUNS_CHUNKS,
        decode_share=envs.VLLM_MERLIN_PREFILL_RUNS_DECODE_SHARE,
        prefill_tick_ms=prefill_tick,
        decode_round_ms=decode_round,
        fill_ticks=fill,
    )


def refusals(sched: Scheduler, cfg: RunsConfig) -> list[str]:
    """Why this engine cannot run prefill runs as configured (empty = it can)."""
    out = []
    if sched.decode_first is not None:
        out.append(
            "VLLM_MERLIN_DECODE_FIRST=1: decode-first sizes prefill chunks "
            "around decoders, prefill runs keep chunks whole and give "
            "decoders the budget headroom; set one of them"
        )
    interval = sched.scheduler_config.prefill_schedule_interval
    if interval > 1:
        out.append(
            f"--prefill-schedule-interval {interval}: the interval holds every "
            "prefill, short ones included; prefill runs replace it, drop the flag"
        )
    if sched.parallel_config.data_parallel_size > 1:
        out.append("data parallelism: the DP engine core drives its own prefill cadence")
    spec = sched.vllm_config.speculative_config
    draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
    rows = sched.max_num_running_reqs
    per_row = 1 + sched.num_spec_tokens
    need = cfg.chunk + rows * per_row
    if sched.max_num_scheduled_tokens < need:
        set_by = (
            "--max-num-scheduled-tokens"
            if sched.scheduler_config.max_num_scheduled_tokens is not None
            else "--max-num-batched-tokens (--max-num-scheduled-tokens unset)"
        )
        out.append(
            f"token budget {sched.max_num_scheduled_tokens} < {need} "
            f"(chunk {cfg.chunk} + {rows} sequences x {per_row} tokens), set "
            f"by {set_by}: raise --max-num-scheduled-tokens and "
            f"--max-num-batched-tokens to {need} or more, lower --max-num-seqs, "
            "or lower VLLM_MERLIN_PREFILL_RUNS_CHUNK_TOKENS"
        )
    need_input = need + (rows + 1) * draft_slots
    if sched.scheduler_config.max_num_batched_tokens < need_input:
        out.append(
            f"input budget {sched.scheduler_config.max_num_batched_tokens} < "
            f"{need_input} (as above plus {draft_slots} drafting slots per row)"
        )
    if sched.need_mamba_block_aligned_split:
        block = sched.mamba_block_size
        if block % cfg.chunk and cfg.chunk % block:
            out.append(
                f"chunk {cfg.chunk} does not tile the Mamba block {block}: "
                "chunk ends would fall off the block grid"
            )
    if (
        not sched.scheduler_config.enable_chunked_prefill
        and sched.max_model_len > cfg.chunk
    ):
        # Without chunking a prompt is admitted whole or not at all, and no
        # step's prefill may exceed the chunk: a longer prompt would wait
        # forever.
        out.append(
            f"chunked prefill is off and --max-model-len {sched.max_model_len} "
            f"exceeds the chunk {cfg.chunk}: enable chunked prefill, or keep "
            "every prompt within the chunk"
        )
    floor = MIN_PREFILL_TOKENS
    if cfg.chunk < floor:
        out.append(
            f"chunk {cfg.chunk} is below the {floor}-token floor for the "
            "per-step prefill cap: raise VLLM_MERLIN_PREFILL_RUNS_CHUNK_TOKENS "
            f"to {floor} or more"
        )
    lpt = getattr(sched.scheduler_config, "long_prefill_token_threshold", 0)
    if 0 < lpt < floor:
        # While the runs are inactive the threshold caps each request's
        # step; the adaptive floor only ever raises it (scheduler.py).
        out.append(
            f"--long-prefill-token-threshold {lpt} is below the {floor}-token "
            f"floor for a per-request prefill cap: set it to {floor} or more, "
            "or 0"
        )
    lookahead = getattr(sched, "num_prefill_lookahead", 0)
    if cfg.chunk < lookahead:
        # _reserve_prefill_lookahead never ends a chunk with fewer than
        # `lookahead` prefill tokens left; with c tokens left, Q < c <= lookahead,
        # the cap leaves Q and the reservation takes them all back: the
        # prefill could never advance. Q >= lookahead always leaves a step.
        out.append(
            f"chunk {cfg.chunk} is below the drafter's prefill lookahead "
            f"{lookahead} (a chunk never ends within {lookahead} tokens of the "
            "prompt end): raise VLLM_MERLIN_PREFILL_RUNS_CHUNK_TOKENS to "
            f"{lookahead} or more"
        )
    if getattr(sched.scheduler_config, "disable_chunked_mm_input", False):
        # A multimodal item is then scheduled whole or not at all, and an
        # item longer than Q would wait forever under the cap.
        out.append(
            "--disable-chunked-mm-input: an image longer than the chunk "
            f"{cfg.chunk} could never be scheduled; drop the flag"
        )
    if getattr(sched, "prefix_replay_tokens", 0):
        # A replayed hit grows an admission after its lookup, past the
        # one-token test that keeps decode rows off the prefill cap.
        out.append("bounded prefix replay (sliding-window prefix hits) is not supported")
    try:
        # With n long prefills the share is f / n: n = 1 gives the longest window.
        rounds = window_rounds(cfg, 1)
    except ValueError as e:
        out.append(str(e))
        rounds = 1
    if cfg.decode_share > 0 and rounds <= 0:
        out.append(
            f"decode share {cfg.decode_share} with tick costs "
            f"{cfg.prefill_tick_ms}/{cfg.decode_round_ms} ms gives no decoder "
            "round per window: raise the share or set it to 0"
        )
    return out


def state(sched: Scheduler) -> PrefillRuns | None:
    """The switch's per-scheduler state, or None when it is off (the default).
    Raises ValueError for a combination it refuses."""
    import vllm.envs as envs

    if not envs.VLLM_MERLIN_PREFILL_RUNS:
        return None
    cfg = config_from_env()
    bad = refusals(sched, cfg)
    if bad:
        raise ValueError(
            f"{KNOB}=1 refused: "
            + "; ".join(bad)
            + f". {KNOB}=0 turns the runs off (on the Flash-Next launcher rows, "
            "where they are the default: PREFILL_RUNS=0)"
        )
    logger.info(
        "Prefill runs: switch on; chunk %d, %d chunks per run, decode share "
        "%.2f (%d decoder rounds per window with one long prefill), tick "
        "costs %.1f / %.1f ms, fill %.1f ticks",
        cfg.chunk,
        cfg.run_chunks,
        cfg.decode_share,
        window_rounds(cfg, 1),
        cfg.prefill_tick_ms,
        cfg.decode_round_ms,
        cfg.fill_ticks,
    )
    return PrefillRuns(cfg)


@dataclass
class RunsStats:
    """Cumulative counters (one log line per 1000 steps when they move)."""

    runs: int = 0  # windows opened (a run reached P chunks)
    windows_cut: int = 0  # windows ended early: no decoder could decode
    long_chunks: int = 0  # full chunks given to the long prefill in runs
    long_deferred: int = 0  # steps the holding long prefill waited
    short_first: int = 0  # of those, steps a short prefill went first
    _logged: tuple[int, ...] = ()
    _next_log_step: int = 0


@dataclass
class RunsStep:
    """One schedule() call's view: the prefill budget left and the decisions
    taken at the start of the step."""

    prefill_budget: int
    active: bool = False
    window: bool = False
    holder: str | None = None
    short_running: bool = False
    short_waiting: frozenset[str] = frozenset()
    long_scheduled: bool = False
    short_admitted: bool = False
    decode_rows: int = 0

    @property
    def short_first(self) -> bool:
        """A short prefill (running, or waiting at the front of the queue)
        takes this step ahead of the long one."""
        return self.short_running or bool(self.short_waiting)

    def note(self, request: Request, num_tokens: int, prefill: bool) -> None:
        """A RUNNING row was scheduled (``num_tokens`` > 0) or withdrawn by a
        preemption later in the step (``num_tokens`` < 0)."""
        sign = 1 if num_tokens > 0 else -1
        if prefill:
            self.prefill_budget -= num_tokens
            if request.request_id == self.holder:
                self.long_scheduled = sign > 0
        else:
            self.decode_rows += sign

    def note_admitted(
        self, request: Request, num_tokens: int, num_computed_tokens: int
    ) -> None:
        """An admission was scheduled. One with a single token left (a full
        prefix-cache hit, a landed restore) is a decode row: it takes the
        decode headroom, not the step's prefill cap."""
        if num_computed_tokens >= request.num_tokens - 1:
            self.decode_rows += 1
            return
        self.prefill_budget -= num_tokens
        if request.request_id == self.holder:
            self.long_scheduled = True
        else:
            self.short_admitted = True


@dataclass
class PrefillRuns:
    cfg: RunsConfig
    # Long prefills not done yet: request id -> the token count its prefill
    # ends at (insertion-ordered). Membership covers that prefill only: a
    # later turn of the same session is classified afresh.
    long_ids: dict[str, int] = field(default_factory=dict)
    in_window: bool = False
    run_done: int = 0  # full chunks in the current run
    window_rows: int = 0  # decode rows scheduled in the current window
    window_target: int = 0  # rounds of the current window
    # Waiting short candidates that went first and were not admitted, each
    # keyed by its status and readiness (see _key): they do not hold the long
    # prefill again until the candidate set or any member's state changes.
    short_miss: frozenset[tuple] | None = None
    # Free KV blocks when the miss was recorded: more free blocks later (a
    # request finished, was aborted or preempted) may let it fit, so it ends.
    miss_free_blocks: int | None = None
    stats: RunsStats = field(default_factory=RunsStats)

    @staticmethod
    def _free_blocks(sched: Scheduler) -> int | None:
        manager = getattr(sched, "kv_cache_manager", None)
        pool = getattr(manager, "block_pool", None)
        return pool.get_num_free_blocks() if pool is not None else None

    def _left(self, request: Request) -> int:
        return request.num_tokens - request.num_computed_tokens

    @staticmethod
    def _key(request: Request, landed: set[str]) -> tuple:
        """A waiting candidate's identity and state: any change of status,
        computed tokens, length or load readiness makes it a new candidate."""
        return (
            request.request_id,
            request.status,
            request.num_computed_tokens,
            request.num_tokens,
            request.request_id in landed,
        )

    def _check_miss(self, sched: Scheduler) -> None:
        """Drop the miss once any member left or changed state. Called at the
        start and the end of every step, so a member that starts a load (and
        later fails back to WAITING with its old state) is never stale."""
        if self.short_miss is None:
            return
        free = self._free_blocks(sched)
        if free is not None and self.miss_free_blocks is not None and (
            free > self.miss_free_blocks
        ):
            self.short_miss = None
            return
        landed = sched.finished_recving_kv_req_ids
        for key in self.short_miss:
            req = sched.requests.get(key[0])
            if req is None or req.is_finished() or self._key(req, landed) != key:
                self.short_miss = None
                return

    @staticmethod
    def _decodes(request: Request) -> bool:
        """A running request past its prefill that still needs a decode step
        (the scheduler's own test for a request already at max_tokens)."""
        return not request.is_prefill_chunk and not (
            request.num_output_placeholders > 0
            and request.num_computed_tokens + 2 - request.num_output_placeholders
            >= request.num_prompt_tokens + request.max_tokens
        )

    def begin_step(self, sched: Scheduler) -> RunsStep:
        from vllm.v1.request import RequestStatus

        cfg = self.cfg
        step = RunsStep(prefill_budget=cfg.chunk)
        self._check_miss(sched)
        for rid, end in list(self.long_ids.items()):
            req = sched.requests.get(rid)
            if (
                req is None
                or req.is_finished()
                # Its prefill is done, whatever the status now (running,
                # or a streaming session parked or continued with a new turn).
                or req.num_computed_tokens >= end
                or (req.status == RequestStatus.RUNNING and not req.is_prefill_chunk)
            ):
                del self.long_ids[rid]
        for req in sched.running:
            if req.is_prefill_chunk and self._left(req) > cfg.long_tokens:
                self.long_ids.setdefault(req.request_id, req.num_tokens)
        num_decoders = sum(1 for r in sched.running if self._decodes(r))
        # A long prompt not yet admitted (its prefix-cache hit is not known
        # yet, so this is an upper bound): its first chunk must already be a
        # full one, not one sized by the long-prefill threshold.
        long_arriving = not self.long_ids and any(
            self._left(r) > cfg.long_tokens
            for queue in (sched.kv_holding_waiting, sched.waiting)
            for r in itertools.islice(queue, WAITING_SCAN)
        )
        if cfg.decode_share <= 0.0 or not num_decoders or not (
            self.long_ids or long_arriving
        ):
            self.in_window = False
            self.run_done = 0
            return step

        if self.in_window and self.window_rows >= self.window_target * num_decoders:
            self.in_window = False
            self.run_done = 0
        if not self.in_window and self.run_done >= cfg.run_chunks:
            self.run_done = 0
            target = window_rounds(cfg, max(1, len(self.long_ids)))
            if target > 0:
                self.in_window = True
                self.window_rows = 0
                self.window_target = target
                self.stats.runs += 1

        step.active = True
        step.window = self.in_window
        step.holder = next(
            (
                r.request_id
                for r in sched.running
                if r.is_prefill_chunk and r.request_id in self.long_ids
            ),
            None,
        )
        step.short_running = any(
            r.is_prefill_chunk and r.request_id not in self.long_ids
            for r in sched.running
        )
        if (
            len(sched.running) + sched.num_waiting_for_streaming_input
            < sched.max_num_active_reqs
        ):
            landed = sched.finished_recving_kv_req_ids
            waiting = frozenset(
                self._key(r, landed)
                for queue in (sched.kv_holding_waiting, sched.waiting)
                for r in itertools.islice(queue, WAITING_SCAN)
                if (
                    r.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED)
                    # A load that has landed is admitted this step.
                    or (
                        r.status == RequestStatus.WAITING_FOR_REMOTE_KVS
                        and r.request_id in landed
                    )
                )
                and r.request_id not in self.long_ids
                # One token left is a decode row: it needs no prefill step.
                and not (0 < r.num_computed_tokens >= r.num_tokens - 1)
            )
            if waiting and waiting != self.short_miss:
                step.short_waiting = frozenset(key[0] for key in waiting)
        return step

    def defers(self, step: RunsStep, request: Request, token_avail: int) -> bool:
        """Whether the RUNNING prefill chunk of ``request`` waits this step."""
        if not step.active or request.request_id not in self.long_ids:
            return False
        holder = request.request_id == step.holder
        if (
            step.window
            or not holder
            or step.short_first
            or min(step.prefill_budget, token_avail)
            < min(self.cfg.chunk, self._left(request))
        ):
            if holder:
                self.stats.long_deferred += 1
                self.stats.short_first += step.short_first and not step.window
            return True
        return False

    def defers_admission(
        self, step: RunsStep, request: Request, left: int, token_avail: int
    ) -> bool:
        """Whether a waiting request with ``left`` prefill tokens after its
        prefix-cache lookup stays waiting this step (a long prefill while
        another holds the runs, in a window, or behind a short prefill; any
        prefill once the step's prefill budget is spent). Never called for an
        async KV load."""
        if not step.active or left <= 1:
            # A single token left is a decode row, admitted on the decode
            # headroom (see RunsStep.note_admitted).
            return False
        if step.prefill_budget <= 0:
            # The step's prefill is spent; the scan goes on so async KV
            # loads (no compute this step) still start.
            return True
        rid = request.request_id
        if left <= self.cfg.long_tokens and rid not in self.long_ids:
            return False
        self.long_ids.setdefault(rid, request.num_tokens)
        if (
            step.window
            or step.holder is not None
            # Not behind itself: a long prompt seen for the first time was
            # a waiting candidate until its lookup.
            or step.short_running
            or bool(step.short_waiting - {rid})
            or min(step.prefill_budget, token_avail) < min(self.cfg.chunk, left)
        ):
            return True
        step.holder = rid
        return False

    def end_step(self, sched: Scheduler, step: RunsStep) -> None:
        # Whether or not a candidate went first this step, a member of an
        # older miss that changed state here (started a load, was admitted,
        # aborted) ends that miss.
        self._check_miss(sched)
        if step.short_waiting:
            from vllm.v1.request import RequestStatus

            # A miss is a candidate still waiting to compute after the step. One
            # that started an async KV load is not: when it lands it is a new
            # candidate and goes ahead of the long prefill again.
            landed = sched.finished_recving_kv_req_ids
            still = frozenset(
                self._key(r, landed)
                for rid in step.short_waiting
                if (r := sched.requests.get(rid)) is not None
                and r.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED)
            )
            self.short_miss = None if step.short_admitted or not still else still
            self.miss_free_blocks = self._free_blocks(sched)
        if not step.active:
            return
        if step.long_scheduled and not step.window:
            self.run_done += 1
            self.stats.long_chunks += 1
        if step.window:
            self.window_rows += step.decode_rows
            if step.decode_rows == 0 and not any(
                self._decodes(r) and sched.current_step < r.next_decode_eligible_step
                for r in sched.running
            ):
                # No decoder decoded and none is waiting for its pipeline
                # cadence: nothing can use the window, end it.
                self.in_window = False
                self.run_done = 0
                self.stats.windows_cut += 1
        self._maybe_log(sched)

    def _maybe_log(self, sched: Scheduler) -> None:
        s = self.stats
        now = (s.runs, s.windows_cut, s.long_chunks, s.long_deferred, s.short_first)
        if now != s._logged and sched.current_step >= s._next_log_step:
            logger.info(
                "Prefill runs: %d windows (%d cut short), %d long chunks in "
                "runs, long prefill held %d steps (%d behind a short prefill)",
                *now,
            )
            s._logged = now
            s._next_log_step = sched.current_step + 1000

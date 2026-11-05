# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""RecoverSSM post-sampling commit as a standalone CUDA graph.

The commit chain of one decode step (per GDN group: plan, conv compaction, fused
final/boundary state fold, align postprocess) runs after sampling, outside the
forward graph. The eager path launches these kernels with Python checks. This
module replays the SAME chain from a CUDA graph: same kernels, same grids, same
arguments, same order, on the same stream, so the committed state is bit-identical
to the eager chain. The chain is rank-local (no collectives), so a rank replaying
while another runs eagerly changes timing only.

Modes (``VLLM_RECOVERSSM_COMMIT_GRAPH``, flippable at runtime through the worker
method ``set_recoverssm_commit_mode``):

- ``eager``  (default): the unchanged launch sequence, nothing staged, and no
  commit graph is captured at boot.
- ``staged``: copy the pointer-unstable inputs into persistent staging buffers,
  then run the chain eagerly on them. The control arm: it does everything the graph
  arm does except the replay.
- ``graph``: stage, then replay the graph captured for this step's key.

Graphs are captured ONLY in the engine's CUDA-graph capture phase (quiescent: no
request is in flight, so no output thread is issuing CUDA work), from the dummy
uniform-decode metadata of each FULL forward capture, in ``global`` capture mode.
That happens when the boot-time mode is ``staged`` or ``graph`` (a process booted
in ``eager`` has no graphs; flipping it to ``graph`` runs staged-eager). Serving
never captures: a step whose key was not captured runs staged-eager. Capture
errors propagate (the boot fails loudly); only a step that is not eligible is
skipped, before any capture API is called.

A step is graph-eligible only when it is a uniform all-decode RecoverSSM step:
every recorded metadata is GDN RecoverSSM with a commit, all groups have the same
``n`` spec rows, ``n`` equals the batch's request count, and the spec rows are
0..n-1 (no request mapping). Everything else (prefill/mixed batches, PP-deferred
commits, other RecoverSSM backends, an uncaptured key) runs an eager chain. Fail
closed: never replay a graph whose recorded addresses are not exactly the current
step's tensors.

Staged inputs (fresh tensors every step): the sampler's ``num_sampled``, the
batch's ``idx_mapping`` and each group's align ``num_computed_tokens``. Every
other tensor the chain reads is passed through and pinned by the key (builder-
staged spec state indices and query offsets, the persistent block-table slice,
the mamba running-state column and accepted counts: data pointer, dtype, strides,
the shape entries the launches use) or, for the commit context, by its identity:
the context is a frozen dataclass (no field can be rebound), and the addresses of
every context tensor a launch takes as an argument are re-checked before each
replay."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.worker import guard_telemetry

logger = init_logger(__name__)

ENV_NAME = "VLLM_RECOVERSSM_COMMIT_GRAPH"
COMMIT_MODES = ("eager", "staged", "graph")
_MODE_ALIASES = {
    "": "eager",
    "0": "eager",
    "eager": "eager",
    "staged": "staged",
    "1": "graph",
    "graph": "graph",
}
# Dtypes of the serving-time staged sources, assumed by the warmup capture: the
# rejection sampler's num_sampled (rejection_sampler_utils, new_empty int32) and
# idx_mapping (async_tensor_h2d of an np.intp array). A serving step with other
# dtypes has a different key and runs staged-eager.
NUM_SAMPLED_DTYPE = torch.int32
IDX_MAPPING_DTYPE = torch.int64


def parse_commit_mode(value: str | None) -> str:
    """Map an env/RPC value to a mode. Unknown values raise: a typo must not
    silently run a different arm."""
    key = "" if value is None else str(value).strip().lower()
    if key not in _MODE_ALIASES:
        raise ValueError(
            f"{ENV_NAME}: unknown RecoverSSM commit mode {value!r} "
            f"(expected one of {COMMIT_MODES} or 0/1)"
        )
    return _MODE_ALIASES[key]


_mode: str | None = None


def get_commit_mode() -> str:
    global _mode
    if _mode is None:
        import vllm.envs as envs

        _mode = parse_commit_mode(envs.VLLM_RECOVERSSM_COMMIT_GRAPH)
    return _mode


def set_commit_mode(mode: str) -> str:
    """Set the process-wide commit mode; takes effect at the next commit."""
    global _mode
    new = parse_commit_mode(mode)
    old = get_commit_mode()
    _mode = new
    if new != old:
        logger.info("[recoverssm-commit] mode=%s (was %s)", new, old)
    return new


@dataclass
class CommitGraphStats:
    eager_steps: int = 0  # mode eager, or ineligible in staged/graph mode
    staged_steps: int = 0  # staged-eager runs (staged mode, or uncaptured key)
    replays: int = 0
    captures: int = 0
    fallbacks: dict[str, int] = field(default_factory=dict)

    def fallback(self, reason: str) -> None:
        self.fallbacks[reason] = self.fallbacks.get(reason, 0) + 1


_stats = CommitGraphStats()


def commit_stats() -> dict[str, Any]:
    return {
        "mode": get_commit_mode(),
        "pid": os.getpid(),
        "device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "eager_steps": _stats.eager_steps,
        "staged_steps": _stats.staged_steps,
        "replays": _stats.replays,
        "captures": _stats.captures,
        "fallbacks": dict(_stats.fallbacks),
    }


def _tsig(t: torch.Tensor | None, *, shape_dims: tuple[int, ...] = ()) -> Any:
    """What a captured launch bakes in from a pass-through tensor: its address,
    dtype, strides and the listed shape entries (the ones passed as launch args)."""
    if t is None:
        return None
    return (
        t.data_ptr(),
        t.dtype,
        t.device,
        tuple(t.stride()),
        tuple(t.shape[d] for d in shape_dims),
    )


def context_bound_tensors(ctx: Any) -> tuple[torch.Tensor, ...]:
    """The context tensors the commit launches take as arguments (the per-layer
    tensors are reached only through the address tables, on device, by both the
    eager and the replayed chain)."""
    return (
        ctx.conv_states[0],
        ctx.conv_state_base_addrs,
        ctx.conv_state_block_strides,
        ctx.conv_state_dim_strides,
        ctx.conv_state_token_strides,
        ctx.checkpoints[0],
        ctx.state_base_addrs,
        ctx.state_block_strides,
        ctx.replays[0],
        ctx.replay_base_addrs,
        ctx.replay_block_strides,
        ctx.commit_lens,
        ctx.final_state_indices,
        ctx.boundary_state_indices,
        ctx.boundary_recovery_lens,
    )


def _context_ptrs(ctx: Any) -> tuple[int, ...]:
    return tuple(t.data_ptr() for t in context_bound_tensors(ctx))


@dataclass
class _Plan:
    """One eligible step, resolved: the key, and what to stage."""

    n: int
    key: tuple
    # Per group: staging slot of its num_computed_tokens (None: not align).
    nc_slots: tuple[int | None, ...]
    nc_sources: list[torch.Tensor]  # one per slot
    contexts: tuple[Any, ...]


@dataclass
class _Entry:
    graph: Any  # torch.cuda.CUDAGraph
    # Everything whose address the graph recorded, kept alive with it: the
    # contexts, the staging buffers and the pass-through tensors of the step it
    # was captured from.
    contexts: tuple[Any, ...]
    staging: tuple[torch.Tensor, ...]
    bindings: tuple[Any, ...]
    # Per context, the addresses of every tensor a launch takes as an argument,
    # as the context's fields held them at capture; the fields are re-read and
    # compared before each replay.
    context_ptrs: tuple[tuple[int, ...], ...]


class RecoverSSMCommitGraphs:
    """Stages, captures (at warmup) and replays the RecoverSSM commit chain."""

    def __init__(self, launch: Callable[..., None]) -> None:
        # launch(step, num_sampled, idx_mapping, *, state_indices,
        #        num_accepted_tokens, num_computed_tokens) -> None
        self._launch = launch
        self._graphs: dict[tuple, _Entry] = {}
        self._staging: dict[tuple, torch.Tensor] = {}
        self._capture_stream: torch.cuda.Stream | None = None
        self._logged_replay = False
        self._logged_uncaptured = False
        self._contexts: tuple[Any, ...] | None = None  # of the last eligible step
        self._require_cuda = True  # CPU unit tests of the eligibility rules clear it
        # Read the mode now: a bad env value fails the boot, not the first step.
        # The boot-time mode decides whether warmup captures graphs.
        self.capture_at_warmup = get_commit_mode() != "eager"

    # -- lifecycle -------------------------------------------------------------

    def reset(self) -> None:
        """Drop every graph and staging buffer (KV cache teardown/re-allocation:
        the graphs hold addresses inside the old cache)."""
        self._graphs.clear()
        self._staging.clear()
        self._contexts = None

    @property
    def num_graphs(self) -> int:
        return len(self._graphs)

    # -- eligibility -----------------------------------------------------------

    def _plan(
        self,
        step: Sequence[Any],
        num_sampled: torch.Tensor,
        idx_mapping: torch.Tensor,
        state_indices: torch.Tensor | None,
        num_accepted_tokens: torch.Tensor,
    ) -> _Plan | str:
        """The step's key and staging plan, or the reason it is not eligible."""
        from vllm.v1.attention.backends.gdn_recoverssm import GDNRecoverSSMMetadata

        if not step:
            return "empty_step"
        if not isinstance(num_sampled, torch.Tensor) or num_sampled.ndim != 1:
            return "num_sampled"
        if idx_mapping.ndim != 1:
            return "idx_mapping"
        num_reqs = idx_mapping.shape[0]
        if num_reqs == 0 or num_sampled.shape[0] < num_reqs:
            return "num_sampled"
        dev = num_sampled.device
        if idx_mapping.device != dev:
            return "device"
        group_sigs = []
        nc_slots: list[int | None] = []
        nc_sources: list[torch.Tensor] = []
        nc_ptrs: dict[tuple, int] = {}
        contexts = []
        for m in step:
            if type(m) is not GDNRecoverSSMMetadata:
                return "metadata_type"
            c = m.recoverssm_commit
            ctx = m.recoverssm_context
            if c is None or ctx is None:
                return "no_commit"
            n = m.num_spec_decodes
            if n != num_reqs:
                return "not_uniform_decode"
            if c.request_indices is not None:
                return "request_indices"
            if (
                c.state_indices.ndim != 2
                or c.state_indices.shape[0] < n
                or c.query_start_loc.ndim != 1
                or c.query_start_loc.shape[0] < n + 1
            ):
                return "commit_shape"
            if n > ctx.commit_lens.shape[0]:
                return "capacity"
            nc = c.num_computed_tokens
            slot = None
            if nc is not None:
                if nc.ndim != 1 or nc.shape[0] < n or nc.device != dev:
                    return "num_computed_tokens"
                k = (nc.data_ptr(), nc.dtype, tuple(nc.stride()))
                if k not in nc_ptrs:
                    nc_ptrs[k] = len(nc_sources)
                    nc_sources.append(nc)
                slot = nc_ptrs[k]
            nc_slots.append(slot)
            contexts.append(ctx)
            group_sigs.append(
                (
                    id(ctx),
                    n,
                    _tsig(c.state_indices),
                    _tsig(c.query_start_loc),
                    _tsig(c.block_table, shape_dims=(1,)),
                    c.block_size,
                    None if nc is None else (slot, nc.dtype),
                )
            )
        if any(s is not None for s in nc_slots) and state_indices is None:
            return "state_indices"
        if dev.type != "cuda" and self._require_cuda:
            return "device"
        key = (
            num_reqs,
            num_sampled.dtype,
            idx_mapping.dtype,
            dev,
            _tsig(state_indices),
            _tsig(num_accepted_tokens),
            tuple(group_sigs),
        )
        return _Plan(num_reqs, key, tuple(nc_slots), nc_sources, tuple(contexts))

    # -- staging ---------------------------------------------------------------

    def _buffer(self, name: str, dtype: torch.dtype, dev: torch.device, cap: int):
        k = (name, dtype, dev)
        buf = self._staging.get(k)
        if buf is None or buf.shape[0] < cap:
            # A captured graph reads the old buffer's address: drop them all.
            self._graphs.clear()
            buf = torch.zeros(cap, dtype=dtype, device=dev)
            self._staging[k] = buf
        return buf

    def _stage(
        self,
        plan: _Plan,
        num_sampled: torch.Tensor,
        idx_mapping: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Copy the step's pointer-unstable inputs into the staging buffers (same
        stream as the commit, so the next step's copies are ordered after it)."""
        n = plan.n
        dev = num_sampled.device
        cap = max(int(ctx.commit_lens.shape[0]) for ctx in plan.contexts)
        ns = self._buffer("num_sampled", num_sampled.dtype, dev, cap)[:n]
        im = self._buffer("idx_mapping", idx_mapping.dtype, dev, cap)[:n]
        ncs = [
            self._buffer(f"num_computed_{i}", src.dtype, dev, cap)[:n]
            for i, src in enumerate(plan.nc_sources)
        ]
        dst = [ns, im, *ncs]
        src = [num_sampled[:n], idx_mapping[:n], *(s[:n] for s in plan.nc_sources)]
        # One fused launch per dtype group where the foreach kernel applies.
        by_dtype: dict[torch.dtype, tuple[list, list]] = {}
        for d, s in zip(dst, src):
            if d.dtype != s.dtype:
                raise AssertionError("staging dtype mismatch")
            by_dtype.setdefault(d.dtype, ([], []))
            by_dtype[d.dtype][0].append(d)
            by_dtype[d.dtype][1].append(s)
        for ds, ss in by_dtype.values():
            if len(ds) == 1:
                ds[0].copy_(ss[0], non_blocking=True)
            else:
                torch._foreach_copy_(ds, ss, non_blocking=True)
        return ns, im, ncs

    def _staged_args(self, plan: _Plan, ncs: list[torch.Tensor]):
        return [None if s is None else ncs[s] for s in plan.nc_slots]

    def _check_contexts(self, plan: _Plan) -> None:
        last = self._contexts
        if (
            last is None
            or len(last) != len(plan.contexts)
            or any(a is not b for a, b in zip(last, plan.contexts))
        ):
            # New commit contexts (re-created builders / KV cache): graphs of the
            # old ones are dead and would keep the old cache alive.
            self._graphs.clear()
            self._contexts = plan.contexts

    # -- warmup capture --------------------------------------------------------

    def capture_warmup(
        self,
        step: Sequence[Any],
        *,
        state_indices: torch.Tensor | None,
        num_accepted_tokens: torch.Tensor,
    ) -> bool:
        """Capture the commit chain for one FULL forward capture's dummy decode
        metadata. Called from the engine's CUDA-graph capture phase, outside any
        forward capture. Returns False when the step is not eligible (checked
        before any capture API is touched); every capture error propagates."""
        if not self.capture_at_warmup or not step:
            return False
        ctx0 = getattr(step[0], "recoverssm_context", None)
        if ctx0 is None or not hasattr(ctx0, "commit_lens"):
            return False
        dev = ctx0.commit_lens.device
        n = getattr(step[0], "num_spec_decodes", 0)
        if n <= 0:
            return False
        # Stand-ins for the serving-time sources, valued so the warm-up run is a
        # no-op on persistent state: 0 accepted tokens makes every commit length
        # 0 (no fold, no conv write), idx -1 makes the align postprocess skip.
        ns_src = torch.zeros(n, dtype=NUM_SAMPLED_DTYPE, device=dev)
        im_src = torch.full((n,), -1, dtype=IDX_MAPPING_DTYPE, device=dev)
        plan = self._plan(step, ns_src, im_src, state_indices, num_accepted_tokens)
        if isinstance(plan, str):
            _stats.fallback("warmup_" + plan)
            return False
        self._check_contexts(plan)
        if plan.key in self._graphs:
            return False
        ns, im, ncs = self._stage(plan, ns_src, im_src)
        args = dict(
            state_indices=state_indices,
            num_accepted_tokens=num_accepted_tokens,
            num_computed_tokens=self._staged_args(plan, ncs),
        )
        # JIT warm-up with the exact arguments the capture records.
        self._launch(step, ns, im, **args)
        graph = self._capture_graph(dev, step, ns, im, args)
        bindings = tuple(
            t
            for m in step
            for t in (
                m.recoverssm_commit.state_indices,
                m.recoverssm_commit.query_start_loc,
                m.recoverssm_commit.block_table,
            )
            if t is not None
        ) + tuple(t for t in (state_indices, num_accepted_tokens) if t is not None)
        bindings += tuple(
            t for ctx in plan.contexts for t in context_bound_tensors(ctx)
        )
        context_ptrs = tuple(_context_ptrs(ctx) for ctx in plan.contexts)
        self._graphs[plan.key] = _Entry(
            graph, plan.contexts, (ns, im, *ncs), bindings, context_ptrs
        )
        _stats.captures += 1
        logger.info(
            "[recoverssm-commit] captured commit graph n=%d at warmup (%d total)",
            plan.n,
            len(self._graphs),
        )
        return True

    def _capture_graph(self, dev, step, ns, im, args):
        """Record the chain (no execution). Errors propagate: torch's
        capture_end ends the allocator's pool routing before it raises
        (CUDAGraph.cpp, torch 2.13), and the boot should fail loudly anyway."""
        if self._capture_stream is None:
            self._capture_stream = torch.cuda.Stream(device=dev)
        side = self._capture_stream
        torch.cuda.synchronize(dev)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(side):
            # Private pool (nothing in the chain allocates); global mode: the
            # capture phase is quiescent, so any other CUDA work is an error.
            graph.capture_begin(capture_error_mode="global")
            try:
                self._launch(step, ns, im, **args)
            finally:
                graph.capture_end()
        torch.cuda.synchronize(dev)
        return graph

    # -- serving ---------------------------------------------------------------

    def run(
        self,
        step: Sequence[Any],
        num_sampled: torch.Tensor,
        idx_mapping: torch.Tensor,
        *,
        state_indices: torch.Tensor | None,
        num_accepted_tokens: torch.Tensor,
    ) -> bool:
        """Commit the step in staged or graph mode. False: the caller runs the
        eager chain (eager mode, or the step is not eligible). Never captures."""
        mode = get_commit_mode()
        if mode == "eager":
            _stats.eager_steps += 1
            return False
        plan = self._plan(
            step, num_sampled, idx_mapping, state_indices, num_accepted_tokens
        )
        if isinstance(plan, str):
            _stats.eager_steps += 1
            _stats.fallback(plan)
            return False
        self._check_contexts(plan)
        ns, im, ncs = self._stage(plan, num_sampled, idx_mapping)
        if mode == "graph":
            entry = self._graphs.get(plan.key)
            if entry is not None and (
                any(
                    a.data_ptr() != b.data_ptr()
                    for a, b in zip(entry.staging, (ns, im, *ncs), strict=True)
                )
                or any(
                    _context_ptrs(ctx) != ptrs
                    for ctx, ptrs in zip(entry.contexts, entry.context_ptrs)
                )
            ):
                _stats.fallback("binding_moved")
                del self._graphs[plan.key]
                entry = None
            if entry is not None:
                # The graph holds the align postprocess (guarded mamba_utils
                # kernels); a replay inherits the caller's stream, so the
                # producer-coverage check runs here like at every other replay
                # site.
                guard_telemetry.check_replay_stream()
                entry.graph.replay()
                _stats.replays += 1
                if not self._logged_replay:
                    self._logged_replay = True
                    logger.info(
                        "[recoverssm-commit] mode=graph captured=%d (device %s, "
                        "first replay n=%d)",
                        len(self._graphs),
                        ns.device,
                        plan.n,
                    )
                return True
            _stats.fallback("not_captured")
            if not self._logged_uncaptured:
                self._logged_uncaptured = True
                logger.info(
                    "[recoverssm-commit] mode=graph: no graph for this step "
                    "(n=%d, %d captured); running staged-eager",
                    plan.n,
                    len(self._graphs),
                )
        # Staged-eager: the real commit of this step on the staged buffers.
        self._launch(
            step,
            ns,
            im,
            state_indices=state_indices,
            num_accepted_tokens=num_accepted_tokens,
            num_computed_tokens=self._staged_args(plan, ncs),
        )
        _stats.staged_steps += 1
        return True


__all__ = [
    "COMMIT_MODES",
    "ENV_NAME",
    "RecoverSSMCommitGraphs",
    "commit_stats",
    "get_commit_mode",
    "parse_commit_mode",
    "set_commit_mode",
]

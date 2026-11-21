# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Static cold-expert offload: hot experts on the GPU, cold ones on UVA.

Default OFF (``MERLIN_COLD_EXPERTS`` unset, blank or ``0``): :func:`attach` registers nothing and
:func:`dispatch` returns its arguments after one empty-dict test.

Placement (once, after the post-load repack; ``attach`` from
``model_loader/utils.py``). For every MoE layer named by the profile
(``vllm/merlin/cold_experts_plan.py``), on each rank that builds it:

* the packed weights ``w13_weight_packed`` / ``w2_weight_packed`` move whole to
  pinned host RAM and the parameters are rebound to their UVA views (the same
  view helper the UVA offloader uses). The parameter shapes do not change, so
  ``E = w1.size(0)`` and everything keyed on it (block size, the
  MERLIN_MARLIN_MOE_CFG table, workspaces) is the resident value;
* the hot rows (the plan's ``hot_local``) are copied to a GPU tensor; the scales
  stay on the GPU (full) and get a hot-row copy too;
* two int32 maps: global -> hot row, and global -> master row for the local cold
  experts (-1 elsewhere).

Dispatch (``fused_marlin_moe``). After ``E`` and the
block size are chosen from the master tensor, a registered layer runs:

1. ``moe_align_block_size`` with the hot map and with the cold map (same block
   size, same global expert count);
2. ``_fused_marlin_moe`` on the hot GPU tensors, then on the master (UVA) tensors,
   both writing their gemm2 rows into ONE ``[M * topk, K]`` buffer (each call
   writes only the rows of its own experts: padded and unmapped entries are
   skipped, ``use_atomic_add=False``);
3. the caller's own ``moe_sum`` over that buffer with the ORIGINAL expert map.

So every (token, k) row comes from the same GEMM with the same ``M``, ``N``,
``K``, top-k, block size and thread config as the resident call, and the top-k
reduction runs once, in k order, over the same rows: the output is the resident
output, bit for bit, provided the Marlin MoE row result does not depend on which
other experts' blocks share the launch (``use_full_k``, no atomics). That
proviso is checked on a GPU by a bitwise test against the resident output. Summing two separately reduced
calls would NOT be exact (two bf16 partial sums plus an add).

CUDA graphs: the hot tensors, UVA views, maps and the row buffer are allocated
before the memory profile and capture and never move; the cold call always launches,
and the hot call too whenever the layer has hot experts (a call with no routed rows
launches empty blocks); no host sync and no Python branch on routing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch

from vllm.merlin import cold_experts_plan as P

logger = logging.getLogger("vllm.merlin.cold_experts")

WEIGHT_NAMES = ("w13_weight_packed", "w2_weight_packed")
SCALE_NAMES = ("w13_weight_scale", "w2_weight_scale")
# Per-expert tensors the split does not carry: a layer that has any is refused.
EXTRA_MARKERS = ("zero_point", "bias", "global_scale")
# Router-side parameters RoutedExperts registers as its
# own (the aux-free routing bias the gate passes in, e.g. Hy4's expert_bias): read before
# top-k, never by the expert GEMMs, so the split leaves them where they are. Mirrors the
# non-expert names RoutedExperts keeps out of EPLB.
ROUTER_SIDE = ("e_score_correction_bias",)

_REG: dict[int, "ColdSplit"] = {}
_ROWS: dict[tuple, torch.Tensor] = {}
_STATE = {"hint_tokens": 0, "captured": False, "ignored_warned": False}


class _Split:
    """moe_align_block_size results of the hot and cold halves, carried through
    the three alignment variables of fused_marlin_moe."""

    __slots__ = ("hot", "cold")

    def __init__(self, hot, cold):
        self.hot, self.cold = hot, cold


@dataclass
class ColdSplit:
    name: str
    layer: int
    plan: P.LayerPlan
    master_w1: torch.Tensor  # UVA view, all local experts (the layer parameter)
    master_w2: torch.Tensor
    scale1: torch.Tensor  # GPU, all local experts (the layer parameter)
    scale2: torch.Tensor
    hot_w1: torch.Tensor | None  # GPU, plan.hot_local rows
    hot_w2: torch.Tensor | None
    hot_s1: torch.Tensor | None
    hot_s2: torch.Tensor | None
    hot_map: torch.Tensor  # int32 [global]
    cold_map: torch.Tensor  # int32 [global]
    expert_map_ptr: int | None
    # Set when MERLIN_COLD_EXPERTS_ADAPTIVE=1:
    # decode-window distinct-step counts per global expert, and each hot row's
    # global id and master row (the maps' inverse), all fixed device tensors.
    counts: torch.Tensor | None = None  # fp32 [global]
    row_global: torch.Tensor | None = None  # int64 [hot rows]
    row_local: torch.Tensor | None = None  # int64 [hot rows]
    # Swaps applied so far (device int64 [1])
    moved: torch.Tensor | None = None

    def _check(self, kw) -> None:
        if kw["w2"].data_ptr() != self.master_w2.data_ptr():
            raise RuntimeError(f"cold experts {self.name}: w2 is not the registered master")
        for k, t in (("w1_scale", self.scale1), ("w2_scale", self.scale2)):
            if kw[k].data_ptr() != t.data_ptr():
                raise RuntimeError(f"cold experts {self.name}: {k} is not the registered scale")
        for k in ("bias1", "bias2", "w1_zeros", "w2_zeros", "global_scale1", "global_scale2",
                  "input_global_scale1", "input_global_scale2"):
            if kw.get(k) is not None:
                raise RuntimeError(f"cold experts {self.name}: {k} is not split; unset {P.ENV_PROFILE}")

    def align_fn(self, align):
        def run(topk_ids, block_size, num_experts, expert_map=None, *args, **kwargs):
            got = None if expert_map is None else expert_map.data_ptr()
            if got != self.expert_map_ptr:
                raise RuntimeError(f"cold experts {self.name}: the expert map is not the layer's")
            if self.counts is not None:
                count_decode(self, topk_ids)
            hot = (align(topk_ids, block_size, num_experts, self.hot_map, *args, **kwargs)
                   if self.hot_w1 is not None else None)
            cold = align(topk_ids, block_size, num_experts, self.cold_map, *args, **kwargs)
            return _Split(hot, cold), None, None

        return run

    def gemms_fn(self, gemms):
        def run(**kw):
            self._check(kw)
            split = kw["sorted_token_ids"]
            if not isinstance(split, _Split):
                raise RuntimeError(f"cold experts {self.name}: alignment did not go through the split")
            hs = kw["hidden_states"]
            n_rows, K = hs.size(0) * kw["num_topk"], hs.size(1)
            rows = _rows(n_rows, K, hs.dtype, hs.device, kw["num_topk"])
            common = dict(kw)
            common["output"] = rows
            if split.hot is not None:
                gemms(**{**common, "w1": self.hot_w1, "w2": self.hot_w2, "w1_scale": self.hot_s1,
                         "w2_scale": self.hot_s2, "expert_map": self.hot_map,
                         "sorted_token_ids": split.hot[0], "expert_ids": split.hot[1],
                         "num_tokens_post_padded": split.hot[2]})
            gemms(**{**common, "expert_map": self.cold_map,
                     "sorted_token_ids": split.cold[0], "expert_ids": split.cold[1],
                     "num_tokens_post_padded": split.cold[2]})
            return rows

        return run


def _capturing() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def _rows(n_rows: int, K: int, dtype, device, topk: int) -> torch.Tensor:
    """The shared [n_rows, K] gemm2 output. One buffer per (device, dtype, K), sized
    on first use to max(this call, max_num_batched_tokens x topk) -- the memory
    profile's dummy run comes first, so the buffer exists before capture and is
    counted. It never moves once a graph was captured."""
    if _capturing():
        _STATE["captured"] = True
    key = (str(device), dtype, K)
    buf = _ROWS.get(key)
    need = n_rows * K
    if buf is None or buf.numel() < need:
        if _STATE["captured"]:
            raise RuntimeError(
                f"cold experts: the row buffer would grow to {n_rows} rows after CUDA graph capture"
            )
        size = max(need, _STATE["hint_tokens"] * topk * K)
        buf = torch.empty(size, dtype=dtype, device=device)
        _ROWS[key] = buf
    return buf[:need].view(n_rows, K)


def dispatch(w1, align, gemms):
    """The fused_marlin_moe hook: (align, gemms) for an unregistered ``w1``;
    the split pair for a registered master."""
    if not _REG:
        return align, gemms
    s = _REG.get(w1.data_ptr())
    if s is None:
        return align, gemms
    return s.align_fn(align), s.gemms_fn(gemms)


# ---------------------------------------------------------------- adaptive hot set
# MERLIN_COLD_EXPERTS_ADAPTIVE=1 (with the static
# offload on): per registered layer, decode calls (rows <= MMAX, the graph's size
# at capture) add their distinct routed experts to a window counter, inside the
# captured graphs. Every PERIOD steps, between steps on the main stream, outside
# every graph (adaptive_step, called from the runner's execute_model), each layer
# swaps up to SWAPS of its hottest cold experts with its coldest hot rows when the
# candidate's count beats the victim's by more than MARGIN: the candidate's
# master bytes and scales are copied into the victim's hot row and the four map
# entries are rewritten, all from device indices (no host sync). Counts halve
# every WINDOW steps. Captured graphs read the maps and hot rows at fixed
# addresses, and a hot row is a byte copy of the master, so every (token, k) row
# is computed exactly as before: placement only, output unchanged.
ADAPTIVE_ENV = "MERLIN_COLD_EXPERTS_ADAPTIVE"
ADAPTIVE_DEFAULTS = {"PERIOD": 8, "SWAPS": 4, "WINDOW": 64, "MARGIN": 2, "MMAX": 32}
_ADAPTIVE = {"steps": 0, "settings": "unset", "atexit": False}
# The swap totals reach the host without a sync:
# after each swap pass the per-layer totals are copied to pinned memory with an
# event; the next pass logs them (INFO, only when a total changed) once the
# event has completed. The shutdown report reads them after the runner's sync.
_MOVED = {"host": None, "event": None, "logged": None, "step": 0, "reported": False}


@dataclass(frozen=True)
class AdaptiveSettings:
    period: int
    swaps: int
    window: int
    margin: float
    mmax: int


def adaptive_settings(environ=None) -> AdaptiveSettings | None:
    """None when off. The ``MERLIN_COLD_EXPERTS_ADAPTIVE_PERIOD``, ``_SWAPS``,
    ``_WINDOW``, ``_MARGIN`` and ``_MMAX`` knobs tune it (WINDOW 0 = never decay)."""
    import os

    cache = environ is None
    if cache and _ADAPTIVE["settings"] != "unset":
        return _ADAPTIVE["settings"]
    env = os.environ if environ is None else environ
    raw = (env.get(ADAPTIVE_ENV, "") or "").strip()
    if raw not in ("", "0", "1"):
        raise ValueError(f"{ADAPTIVE_ENV} must be 0 or 1 (got {raw!r})")
    out = None
    if raw == "1":
        vals = {}
        for key, default in ADAPTIVE_DEFAULTS.items():
            text = (env.get(f"{ADAPTIVE_ENV}_{key}", "") or "").strip()
            try:
                vals[key] = float(text) if key == "MARGIN" and text else int(text) if text else default
            except ValueError:
                raise ValueError(f"{ADAPTIVE_ENV}_{key}={text!r} is not a number") from None
            low = 0 if key in ("WINDOW", "MARGIN") else 1
            if not math.isfinite(vals[key]):  # inf / nan refused
                raise ValueError(f"{ADAPTIVE_ENV}_{key}={text!r} is not a finite number")
            if vals[key] < low:
                raise ValueError(f"{ADAPTIVE_ENV}_{key}={vals[key]} is below {low}")
        out = AdaptiveSettings(vals["PERIOD"], vals["SWAPS"], vals["WINDOW"], float(vals["MARGIN"]),
                               vals["MMAX"])
    if cache:
        _ADAPTIVE["settings"] = out
    return out


def count_decode(s: ColdSplit, topk_ids: torch.Tensor) -> None:
    """Add a decode call's distinct routed experts to ``s.counts`` (device ops only;
    the row-count branch is taken on the Python shape, i.e. at capture)."""
    st = adaptive_settings()
    if st is None or topk_ids.shape[0] > st.mmax:
        return
    ids = topk_ids.reshape(-1).long()
    live = (ids >= 0) & (ids < s.counts.numel())
    mark = torch.zeros_like(s.counts)
    mark.index_put_((torch.where(live, ids, 0),), live.float(), accumulate=True)
    s.counts.add_(mark.clamp_(max=1.0))


def swap_round(s: ColdSplit, swaps: int, margin: float) -> torch.Tensor | None:
    """One swap pass over one layer, from device values only. Returns the number
    of swaps applied (a device scalar), or None when the layer cannot swap."""
    n_hot = s.row_global.numel()
    n_cold = s.plan.n_local - n_hot
    k = min(swaps, n_hot, n_cold)
    if k <= 0:
        return None
    cold_score = torch.where(s.cold_map >= 0, s.counts, torch.full_like(s.counts, -1.0))
    cand_val, cand_g = cold_score.topk(k)
    vic_val, vic_r = s.counts[s.row_global].topk(k, largest=False)
    swap = cand_val > vic_val + margin
    cand_l = s.cold_map[cand_g].long().clamp_min(0)
    vic_g, vic_l = s.row_global[vic_r], s.row_local[vic_r]
    # bytes first (the candidate's master row into the victim's hot row) ...
    for hot, master in ((s.hot_w1, s.master_w1), (s.hot_w2, s.master_w2),
                        (s.hot_s1, s.scale1), (s.hot_s2, s.scale2)):
        shape = (k,) + (1,) * (hot.dim() - 1)
        rows = torch.where(swap.view(shape), master.index_select(0, cand_l),
                           hot.index_select(0, vic_r))
        hot.index_copy_(0, vic_r, rows)
    # ... then the maps (same stream: no forward sees one without the other)
    hot_r = vic_r.to(s.hot_map.dtype)
    s.hot_map[cand_g] = torch.where(swap, hot_r, s.hot_map[cand_g])
    s.cold_map[cand_g] = torch.where(swap, torch.full_like(hot_r, -1), s.cold_map[cand_g])
    s.hot_map[vic_g] = torch.where(swap, torch.full_like(hot_r, -1), s.hot_map[vic_g])
    s.cold_map[vic_g] = torch.where(swap, vic_l.to(s.cold_map.dtype), s.cold_map[vic_g])
    s.row_global[vic_r] = torch.where(swap, cand_g, vic_g)
    s.row_local[vic_r] = torch.where(swap, cand_l, vic_l)
    return swap.sum()


def adaptive_step() -> None:
    """Once per engine step on every rank (the runner's execute_model, outside
    graphs): every PERIOD steps a swap pass, every WINDOW steps a decay."""
    st = adaptive_settings()
    if st is None or not _REG:
        return
    _ADAPTIVE["steps"] += 1
    step = _ADAPTIVE["steps"]
    if step % st.period == 0:
        _log_moved(final=False)  # the last pass's totals
        for s in _REG.values():
            if s.counts is not None:
                n = swap_round(s, st.swaps, st.margin)
                if n is not None:
                    if s.moved is None:
                        s.moved = torch.zeros(1, dtype=torch.long, device=s.counts.device)
                    s.moved.add_(n)
        _post_moved(step)
    if st.window and step % st.window == 0:
        for s in _REG.values():
            if s.counts is not None:
                s.counts.mul_(0.5)


def _adaptive_layers() -> list[ColdSplit]:
    return [s for s in _REG.values() if s.counts is not None and s.moved is not None]


def _post_moved(step: int) -> None:
    """Copy the per-layer swap totals to pinned host memory, non-blocking."""
    layers = _adaptive_layers()
    if not layers:
        return
    dev = torch.stack([s.moved[0] for s in layers])
    if dev.device.type != "cuda":  # CPU tensors (tests): nothing to wait for
        _MOVED["host"], _MOVED["event"], _MOVED["step"] = dev.clone(), "ready", step
        return
    host = _MOVED["host"]
    if host is None or host.numel() != dev.numel():
        host = torch.zeros(dev.numel(), dtype=torch.long, pin_memory=True)
        _MOVED["host"] = host
    host.copy_(dev, non_blocking=True)
    event = torch.cuda.Event()
    event.record()
    _MOVED["event"], _MOVED["step"] = event, step


def _log_moved(final: bool) -> None:
    """INFO line when a layer's swap total changed since the last line; never
    waits (a pending copy is read at the next pass) unless ``final``."""
    event = _MOVED["event"]
    if event is None:
        return
    if event != "ready":
        if final:
            event.synchronize()
        elif not event.query():
            return
    totals = _MOVED["host"].tolist()
    if totals == _MOVED["logged"] and not final:
        return
    names = [s.name for s in _adaptive_layers()][: len(totals)]
    previous = _MOVED["logged"] or [0] * len(totals)
    _MOVED["logged"] = totals
    logger.info("cold experts adaptive: step %d: swaps %d total (%s)", _MOVED["step"],
                sum(totals), ", ".join(f"{n} {t} (+{t - p})"
                                       for n, t, p in zip(names, totals, previous)))


def adaptive_report(once: bool = False) -> list[tuple[str, int, int]]:
    """(layer name, hot rows now holding an expert the static plan did not put
    there, hot rows) per adaptive layer, plus the swap totals; logged once at
    shutdown (the model runner's shutdown, after its device sync; atexit is the
    fallback) as the GPU check's proof that swaps ran. Host reads: never call it
    between steps. ``once``: the shutdown / atexit call, which logs at most once."""
    if once and _MOVED["reported"]:
        return []
    out = []
    for s in _REG.values():
        if s.counts is None:
            continue
        now = s.row_local.to("cpu", copy=True).tolist()
        out.append((s.name, sum(a != b for a, b in zip(now, s.plan.hot_local)), len(now)))
    if out:
        _MOVED["reported"] = _MOVED["reported"] or once
        _log_moved(final=True)
        swaps = sum(int(s.moved.item()) for s in _adaptive_layers())
        logger.info("cold experts adaptive: %d steps, %d swaps; rows moved off the static "
                       "plan: %s", _ADAPTIVE["steps"], swaps,
                       ", ".join(f"{n} {m}/{h}" for n, m, h in out))
    return out


def registered() -> list[ColdSplit]:
    return list(_REG.values())


def clear() -> None:
    _REG.clear()
    _ROWS.clear()
    _STATE.update(hint_tokens=0, captured=False, ignored_warned=False)
    _ADAPTIVE.update(steps=0, settings="unset")

    _MOVED.update(host=None, event=None, logged=None, step=0, reported=False)


# ---------------------------------------------------------------------------- attach
def _pin_and_view(t: torch.Tensor) -> torch.Tensor:
    """A pinned host copy of ``t`` and its UVA view (the UVA offloader's path)."""
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    host = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
    host.copy_(t)
    return get_accelerator_view_from_cpu_tensor(host)


def _layer_index(name: str) -> int | None:
    parts = name.split(".")
    for i in range(len(parts) - 1):
        if parts[i] == "layers" and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def _marlin_kernel(mod) -> bool:
    qm = getattr(mod, "quant_method", None)
    kernel = getattr(qm, "moe_kernel", None)
    fe = getattr(kernel, "fused_experts", None)
    return fe is not None and type(fe).__name__ == "MarlinExperts"


def _first_rank() -> bool:
    dist = torch.distributed
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def _hint_tokens() -> int:
    try:
        from vllm.config import get_current_vllm_config

        return int(get_current_vllm_config().scheduler_config.max_num_batched_tokens or 0)
    except Exception:  # no config in scope (tests): size on first use
        return 0


def split_layer(name: str, layer: int, params: dict, expert_map, global_experts: int,
                ranking: list[int], hot_frac: float, pin_and_view=_pin_and_view) -> ColdSplit:
    """Split one layer's parameters in place (``params``: name -> nn.Parameter) and
    register it. ``pin_and_view`` is injectable for the CPU tests."""
    w13, w2 = params[WEIGHT_NAMES[0]], params[WEIGHT_NAMES[1]]
    s13, s2 = params[SCALE_NAMES[0]], params[SCALE_NAMES[1]]
    if expert_map is None:
        local_of = list(range(global_experts))
    else:
        local_of = [int(x) for x in expert_map.detach().to("cpu").tolist()]
    if len(local_of) != global_experts:
        raise RuntimeError(f"cold experts {name}: expert map has {len(local_of)} entries, not {global_experts}")
    plan = P.plan_layer(layer, ranking, local_of, hot_frac)
    if plan.n_local != w13.shape[0]:
        raise RuntimeError(f"cold experts {name}: {plan.n_local} local experts, master has {w13.shape[0]} rows")
    dev = w13.device
    idx = torch.tensor(plan.hot_local, dtype=torch.long, device=dev)
    if plan.n_hot:
        hot = [t.data.index_select(0, idx).contiguous() for t in (w13, w2, s13, s2)]
    else:
        hot = [None] * 4
    for p in (w13, w2):
        old = p.data
        p.data = pin_and_view(old)
        p._merlin_cold_split = True
        del old
    hot_map = torch.tensor(plan.hot_map, dtype=torch.int32, device=dev)
    cold_map = torch.tensor(plan.cold_map, dtype=torch.int32, device=dev)
    s = ColdSplit(name, layer, plan, w13.data, w2.data, s13.data, s2.data, *hot, hot_map, cold_map,
                  None if expert_map is None else expert_map.data_ptr())
    # The window counters and the hot rows' inverse maps
    if adaptive_settings() is not None and plan.n_hot:
        global_of = {loc: g for g, loc in enumerate(local_of) if loc >= 0}
        s.counts = torch.zeros(global_experts, dtype=torch.float32, device=dev)
        s.row_local = torch.tensor(plan.hot_local, dtype=torch.long, device=dev)
        s.row_global = torch.tensor([global_of[r] for r in plan.hot_local], dtype=torch.long,
                                    device=dev)
        if not _ADAPTIVE["atexit"]:
            import atexit

            atexit.register(adaptive_report, True)
            _ADAPTIVE["atexit"] = True
    _REG[s.master_w1.data_ptr()] = s
    return s


def attach(model: torch.nn.Module, checkpoint_dir: str | None = None, environ=None) -> int:
    """After weight load and repack: split the profile's layers on this rank.
    Returns the number of layers split (0 when off). Raises on a bad profile, a
    refused layer, or a checkpoint mismatch."""
    st = P.settings_from_env(environ)
    if st is None:
        ignored = P.ignored_overrides(environ)
        # Once per engine: the first model of the process (target, not drafter), on the first rank only.
        if ignored and not _STATE["ignored_warned"] and _first_rank():
            _STATE["ignored_warned"] = True
            logger.warning("cold experts: %s set but %s is off (unset, blank or 0): ignored",
                           " and ".join(ignored), P.ENV_PROFILE)
        return 0
    if "MTP" in type(model).__name__:  # the drafter's experts stay resident
        return 0
    status = P.check_checkpoint(st.profile, checkpoint_dir)
    if status != "ok":
        logger.warning("cold experts: profile %s is %s for this checkpoint (%s)", st.path,
                       "a placeholder, not tied to a checkpoint" if status == "placeholder"
                       else "unverified: no local config.json", checkpoint_dir)
    E = int(st.profile["num_experts"])
    wanted = set(st.layers)
    done: list[ColdSplit] = []
    for name, mod in model.named_modules():
        own = dict(mod.named_parameters(recurse=False))
        if not all(k in own for k in WEIGHT_NAMES + SCALE_NAMES):
            continue
        layer = _layer_index(name)
        if layer is None or layer not in wanted:
            continue
        if getattr(own[WEIGHT_NAMES[0]], "_merlin_cold_split", False):
            continue  # a second post-load pass over the same weights
        # Router-side names are not per-expert tensors
        extra = [k for k in own if k not in ROUTER_SIDE and any(m in k for m in EXTRA_MARKERS)]
        if extra:
            raise RuntimeError(f"cold experts: {name} has {extra}, which the split does not carry")
        if not _marlin_kernel(mod):
            raise RuntimeError(f"cold experts: {name} does not run the Marlin MoE kernel")
        par = getattr(getattr(mod, "moe_config", None), "moe_parallel_config", None)
        if getattr(par, "enable_eplb", False):
            raise RuntimeError(f"cold experts: {name} has EPLB on, which moves experts at runtime")
        if any(getattr(own[k], "_vllm_is_uva_offloaded", False) for k in WEIGHT_NAMES):
            raise RuntimeError(f"cold experts: {name} is already UVA-offloaded (--cpu-offload-params)")
        if own[WEIGHT_NAMES[0]].device.type != "cuda":
            raise RuntimeError(f"cold experts: {name} is not on a CUDA device")
        G = int(getattr(mod, "global_num_experts", E))
        if G != E:
            raise RuntimeError(f"cold experts: {name} has {G} experts, the profile {E}")
        s = split_layer(name, layer, own, getattr(mod, "expert_map", None), G,
                        st.profile["ranking"][str(layer)], st.hot_frac)
        done.append(s)
    if done:
        _STATE["hint_tokens"] = max(_STATE["hint_tokens"], _hint_tokens())
        torch.cuda.empty_cache()
        per = sum(s.master_w1[0].numel() * s.master_w1.element_size()
                  + s.master_w2[0].numel() * s.master_w2.element_size() for s in done[:1])
        freed = sum(s.plan.n_cold for s in done) * per
        logger.info(
            "cold experts: %s; layers %s; hot_frac %g; %s; GPU bytes freed on this rank %d",
            st.path, ",".join(str(s.layer) for s in done), st.hot_frac,
            " ".join(f"L{s.layer}:{s.plan.n_hot}hot/{s.plan.n_cold}cold" for s in done), freed,
        )
    else:
        logger.info("cold experts: none of layers %s is built on this rank", list(st.layers))
    return len(done)

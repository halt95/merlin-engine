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
from dataclasses import dataclass

import torch

from vllm.merlin import cold_experts_plan as P

logger = logging.getLogger("vllm.merlin.cold_experts")

WEIGHT_NAMES = ("w13_weight_packed", "w2_weight_packed")
SCALE_NAMES = ("w13_weight_scale", "w2_weight_scale")
# Per-expert tensors the split does not carry: a layer that has any is refused.
EXTRA_MARKERS = ("zero_point", "bias", "global_scale")

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


def registered() -> list[ColdSplit]:
    return list(_REG.values())


def clear() -> None:
    _REG.clear()
    _ROWS.clear()
    _STATE.update(hint_tokens=0, captured=False, ignored_warned=False)


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
        extra = [k for k in own if any(m in k for m in EXTRA_MARKERS)]
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
